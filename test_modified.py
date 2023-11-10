"""
Computation of performance metrics (nDSC, lesion F1 score, nDSC R-AUC) 
for an ensemble of models.
Metrics are displayed in console.
"""

import argparse
import re
import os
import torch
import torch.nn as nn
from joblib import Parallel
from monai.data import write_nifti
from monai.networks.blocks.convolutions import Convolution
from monai.inferers import sliding_window_inference
# from monai.networks.nets import UNet
import numpy as np
from data_load import remove_connected_components, get_val_dataloader, ForeverDataIterator
from metrics import dice_norm_metric, lesion_f1_score, ndsc_aac_metric
from uncertainty import ensemble_uncertainties_classification
from torchvision import models, transforms, utils
from unet_model_for_loop import UNet, iVAE, MLP
from torch.utils.data import DataLoader
from collections import OrderedDict

parser = argparse.ArgumentParser(description='Get all command line arguments.')
# save options

parser.add_argument('--path_pred', type=str, default="/l/users/umaima.rahman/shifts/data/dev_out/predictions_test",
                    help='Specify the path to the directory to store predictions')
parser.add_argument('--num_models', type=int, default=1,
                    help='Number of models in ensemble')
parser.add_argument('--path_model', type=str, default="/l/users/umaima.rahman/shifts/checkpoints/debugging", 
                    help='Specify the dir to all the trained models')
# data
parser.add_argument('--path_data', type=str, default="/l/users/umaima.rahman/shifts/data/",
                    help='Specify the path to the directory with FLAIR images')
parser.add_argument('--path_gt', type=str, default="/l/users/umaima.rahman/shifts/data/dev_out/gt",
                    help='Specify the path to the directory with FLAIR images')
parser.add_argument('--path_bm', type=str, default="/l/users/umaima.rahman/shifts/data/dev_out/fg_mask/",
                    help='Specify the path to the directory with brain masks')
# parallel computation
parser.add_argument('--num_workers', type=int, default=2,
                    help='Number of workers to preprocess images')
parser.add_argument('--n_jobs', type=int, default=1,
                    help='Number of parallel workers for F1 score computation')
# hyperparameters
parser.add_argument('--threshold', type=float, default=0.35,
                    help='Probability threshold')

parser.add_argument('--source', type=str, default="train,dev_in,eval_in",
                    help='Source domains')
parser.add_argument('--target', type=str, default="dev_out", 
                    help='Target domain')

parser.add_argument('--z_dim', type=int, default=1024, metavar='N')
parser.add_argument('--s_dim', type=int, default=24, metavar='N')
parser.add_argument('--hidden_dim', type=int, default=1024, metavar='N')
parser.add_argument('--beta', type=float, default=1., metavar='N')
parser.add_argument('--name', type=str, default='', metavar='N')
parser.add_argument('--flow', type=str, default='ddsf', metavar='N')
parser.add_argument('--flow_dim', type=int, default=24, metavar='N')
parser.add_argument('--flow_nlayer', type=int, default=2, metavar='N')
parser.add_argument('--init_value', type=float, default=0.0, metavar='N')
parser.add_argument('--flow_bound', type=int, default=5, metavar='N')
parser.add_argument('--flow_bins', type=int, default=8, metavar='N')
parser.add_argument('--flow_order', type=str, default='linear', metavar='N')
parser.add_argument('--net', type=str, default='dirt', metavar='N')
parser.add_argument('--n_flow', type=int, default=2, metavar='N')
parser.add_argument('--lambda_vae', type=float, default=1e-3, metavar='N')
parser.add_argument('--lambda_cls', type=float, default=1., metavar='N')
parser.add_argument('--lambda_ent', type=float, default=0.1, metavar='N')
parser.add_argument('--entropy_thr', type=float, default=0.5, metavar='N')
parser.add_argument('--C_max', type=float, default=20., metavar='N')
parser.add_argument('--C_stop_iter', type=int, default=10000, metavar='N')   


def get_default_device():
    """ Set device """
    if torch.cuda.is_available():
        print("Got CUDA!")
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def main(args):
    os.makedirs(args.path_pred, exist_ok=True)
    device = get_default_device()
    torch.multiprocessing.set_sharing_strategy('file_system')

    total_files = 25
    ind_vec = np.arange(1, 26)
    np.random.shuffle(ind_vec)
    train_ind_vec = ind_vec[0:15]
    val_ind_vec = ind_vec[15:]

    '''' Initialise dataloaders '''
    
    val_dataset = get_val_dataloader(root_path=args.path_data,
                                        num_workers=args.num_workers,
                                        source = args.source,
                                        target = args.target,
                                        tasks = args.target,
                                        indices = val_ind_vec,
                                        bm_path= args.path_bm)
    
    val_loader = DataLoader(val_dataset, batch_size=1,
                            shuffle=True, num_workers=args.num_workers, drop_last=True,
                            )  
    
    val_iter = ForeverDataIterator(val_loader) 

    ''' Load trained models  '''
    K = args.num_models
    models = []
    for i in range(K):
        models.append(UNet(args).to(device))
                      

    for i, model in enumerate(models):
        state_dict = torch.load(os.path.join(args.path_model,"best.pth"))
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove module.
            new_state_dict[name] = v
        msg = model.load_state_dict(new_state_dict)
        print(msg)
        model.eval()

    act = torch.nn.Softmax(dim=1)
    th = args.threshold
    roi_size = (96, 96, 96)
    sw_batch_size = 4

    ndsc, f1, ndsc_aac = [], [], []
    list_val_acc = []
    count = 0
    ''' Evaluatioin loop '''
    with Parallel(n_jobs=args.n_jobs) as parallel_backend:
        with torch.no_grad():
            
            for i in range(len(val_loader)):
                count+=1
                inputs, gt, brain_mask, d_v, image_meta_dicts = next(val_iter)
                # inputs, gt, brain_mask = (
                #     batch_data["image"].to(device),
                #     batch_data["label"].cpu().numpy(),
                #     batch_data["brain_mask"].cpu().numpy()
                # )
                inputs = inputs.to(device)
                gt = gt.to(device)
                brain_mask = brain_mask.to(device)
                d_v = d_v.to(device)

                input = (inputs, d_v, True, True)
                # get ensemble predictions
                all_outputs = []
                for model in models:
                    # f_maps = get_conv_layers(model)
                    # print(f_maps.shape)
                    # outputs = sliding_window_inference(inputs, roi_size,
                    #                                    sw_batch_size, model,
                    #                                    mode='gaussian')


                    _, val_outputs = model(input)

                    # outputs = act(val_outputs).cpu().numpy()
                    # outputs = np.squeeze(outputs[0])
                    # all_outputs.append(outputs)
                    model_seg = val_outputs[0,0].cpu().numpy()
                    model_seg[model_seg >= 0] = 1
                    model_seg[model_seg < 0] = 0
                    all_outputs.append(model_seg)
                all_outputs = np.asarray(all_outputs)

                # get image metadata
                original_affine = image_meta_dicts['original_affine'][0]
                affine = image_meta_dicts['affine'][0]
                spatial_shape = image_meta_dicts['spatial_shape'][0]
                filename_or_obj = image_meta_dicts['filename_or_obj'][0]
                filename_or_obj = os.path.basename(filename_or_obj)

                # obtain and save probability maps averaged across models in an ensemble
                outputs_mean = np.mean(all_outputs, axis=0) #outputs_mean.shape = (96,96,96)
                print(outputs_mean)

                filename = re.sub("FLAIR_isovox.nii.gz", 'pred_prob.nii.gz',
                                    filename_or_obj)
                filepath = os.path.join(args.path_pred, filename)
                write_nifti(outputs_mean, filepath,
                            affine=original_affine,
                            target_affine=affine,
                            output_spatial_shape=spatial_shape)

                # obtain binary segmentation mask
               
                seg = outputs_mean.copy()
                seg[seg >= th] = 1
                seg[seg < th] = 0
                seg = np.squeeze(seg)
                seg = remove_connected_components(seg)

                gt = np.squeeze(gt[0])
                brain_mask = np.squeeze(brain_mask[0])

                filename = re.sub("FLAIR_isovox.nii.gz", 'pred_seg.nii.gz',
                              filename_or_obj)
                filepath = os.path.join(args.path_pred, filename)
                write_nifti(seg, filepath,
                        affine=original_affine,
                        target_affine=affine,
                        mode='nearest',
                        output_spatial_shape=spatial_shape)

                # compute reverse mutual information uncertainty map
                uncs_map = ensemble_uncertainties_classification(np.concatenate(
                    (np.expand_dims(all_outputs, axis=-1),
                     np.expand_dims(1. - all_outputs, axis=-1)),
                    axis=-1))['reverse_mutual_information']

                filename = re.sub("FLAIR_isovox.nii.gz", 'uncs_rmi.nii.gz',
                              filename_or_obj)
                filepath = os.path.join(args.path_pred, filename)
                write_nifti(uncs_map * gt, filepath,
                            affine=original_affine,
                            target_affine=affine,
                            output_spatial_shape=spatial_shape)

                # compute metrics
                
                ndsc += [dice_norm_metric(ground_truth=gt, predictions=seg)]
                f1 += [lesion_f1_score(ground_truth=gt,
                                       predictions=seg,
                                       IoU_threshold=0.5,
                                       parallel_backend=parallel_backend)]
                ndsc_aac += [ndsc_aac_metric(ground_truth=gt[brain_mask == 1].flatten(),
                                             predictions=seg[brain_mask == 1].flatten(),
                                             uncertainties=uncs_map[brain_mask == 1].flatten(),
                                             parallel_backend=parallel_backend)]

                # for nervous people
                if count % 10 == 0:
                    print(f"Processed {count}/{len(val_loader)}")

    ndsc = np.asarray(ndsc) * 100.
    f1 = np.asarray(f1) * 100.
    ndsc_aac = np.asarray(ndsc_aac) * 100.

    print(f"nDSC:\t{np.mean(ndsc):.4f} +- {np.std(ndsc):.4f}")
    print(f"Lesion F1 score:\t{np.mean(f1):.4f} +- {np.std(f1):.4f}")
    print(f"nDSC R-AUC:\t{np.mean(ndsc_aac):.4f} +- {np.std(ndsc_aac):.4f}")


# %%
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
