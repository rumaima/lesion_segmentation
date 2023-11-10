"""
Perform inference for an ensemble of baseline models and save 3D Nifti images of
predicted probability maps averaged across ensemble models (saved to "*pred_prob.nii.gz" files),
binary segmentation maps predicted obtained by thresholding of average predictions and 
removing all connected components smaller than 9 voxels (saved to "pred_seg.nii.gz"),
uncertainty maps for reversed mutual information measure (saved to "uncs_rmi.nii.gz").
"""

import argparse
import os
import re
import torch
from monai.inferers import sliding_window_inference
# from monai.networks.nets import UNet
from monai.data import write_nifti
import numpy as np
from data_load import remove_connected_components, get_flair_dataloader, ForeverDataIterator
from uncertainty import ensemble_uncertainties_classification
from unet_model import UNet, iVAE, MLP
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='Get all command line arguments.')
# save options
parser.add_argument('--path_pred', type=str, required=True,
                    help='Specify the path to the directory to store predictions')
# model
parser.add_argument('--num_models', type=int, default=3,
                    help='Number of models in ensemble')
parser.add_argument('--path_model', type=str, default='',
                    help='Specify the dir to al the trained models')
# data
parser.add_argument('--path_data', type=str, required=True,
                    help='Specify the path to the directory with FLAIR images')
parser.add_argument('--path_bm', type=str, required=True,
                    help='Specify the path to the directory with brain masks')
# parallel computation
parser.add_argument('--num_workers', type=int, default=10,
                    help='Number of workers to preprocess images')
# hyperparameters
parser.add_argument('--threshold', type=float, default=0.35,
                    help='Probability threshold')

parser.add_argument('--source', type=str, default="train,dev_in,eval_in",
                    help='Source domains')
parser.add_argument('--target', type=str, default="dev_out", 
                    help='Target domain')

parser.add_argument('--z_dim', type=int, default=256, metavar='N')
parser.add_argument('--s_dim', type=int, default=32, metavar='N')
parser.add_argument('--hidden_dim', type=int, default=1024, metavar='N')
parser.add_argument('--beta', type=float, default=1., metavar='N')
parser.add_argument('--name', type=str, default='', metavar='N')
parser.add_argument('--flow', type=str, default='ddsf', metavar='N')
parser.add_argument('--flow_dim', type=int, default=32, metavar='N')
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

    '''' Initialise dataloaders '''
    
    val_dataset = get_flair_dataloader(flair_path=args.path_data,
                                        num_workers=args.num_workers,
                                        bm_path= args.path_bm)
    
    val_loader = DataLoader(val_dataset, batch_size=4,
                            shuffle=True, num_workers=args.num_workers, drop_last=True,
                            )  
    
    val_iter = ForeverDataIterator(val_loader)    

    ''' Load trained models  '''
    K = args.num_models
    models = []
    for i in range(K):
        models.append(UNet(args).to(device)
                      )

    for i, model in enumerate(models):
        model.load_state_dict(torch.load(os.path.join(args.path_model,
                                                      f"seed{i + 1}",
                                                      "Best_model_finetuning.pth")))
        model.eval()

    act = torch.nn.Softmax(dim=1)
    th = args.threshold
    roi_size = (96, 96, 96)
    sw_batch_size = 4

    ''' Predictions loop '''
    with torch.no_grad():
        for i in range(len(val_loader)):
            count+=1
            inputs, gt, foreground_mask, d_pred = next(val_iter)
    
            inputs = inputs.to(device)
            foreground_mask = foreground_mask.numpy()[0, 0]

            input = (inputs, d_pred, True, True)
            # get ensemble predictions
            all_outputs = []
            for model in models:
                _, val_outputs, _, _, _, _, _, _ = model(input)
                # outputs = sliding_window_inference(inputs, roi_size, sw_batch_size, model, mode='gaussian')
                outputs = act(val_outputs).cpu().numpy()
                outputs = np.squeeze(outputs[0])
                all_outputs.append(outputs)
            all_outputs = np.asarray(all_outputs)

            # get image metadata
            original_affine = inputs['image_meta_dict']['original_affine'][0]
            affine = inputs['image_meta_dict']['affine'][0]
            spatial_shape = inputs['image_meta_dict']['spatial_shape'][0]
            filename_or_obj = inputs['image_meta_dict']['filename_or_obj'][0]
            filename_or_obj = os.path.basename(filename_or_obj)

            # obtain and save probability maps averaged across models in an ensemble
            outputs_mean = np.mean(all_outputs, axis=0)

            filename = re.sub("FLAIR_isovox.nii.gz", 'pred_prob.nii.gz',
                              filename_or_obj)
            filepath = os.path.join(args.path_pred, filename)
            write_nifti(outputs_mean, filepath,
                        affine=original_affine,
                        target_affine=affine,
                        output_spatial_shape=spatial_shape)

            # obtain and save binary segmentation masks
            seg = outputs_mean.copy()
            seg[seg >= th] = 1
            seg[seg < th] = 0
            seg = np.squeeze(seg)
            seg = remove_connected_components(seg)

            filename = re.sub("FLAIR_isovox.nii.gz", 'pred_seg.nii.gz',
                              filename_or_obj)
            filepath = os.path.join(args.path_pred, filename)
            write_nifti(seg, filepath,
                        affine=original_affine,
                        target_affine=affine,
                        mode='nearest',
                        output_spatial_shape=spatial_shape)

            # obtain and save uncertainty map (voxel-wise reverse mutual information)
            uncs_map = ensemble_uncertainties_classification(np.concatenate(
                (np.expand_dims(all_outputs, axis=-1),
                 np.expand_dims(1. - all_outputs, axis=-1)),
                axis=-1))['reverse_mutual_information']

            filename = re.sub("FLAIR_isovox.nii.gz", 'uncs_rmi.nii.gz',
                              filename_or_obj)
            filepath = os.path.join(args.path_pred, filename)
            write_nifti(uncs_map * foreground_mask, filepath,
                        affine=original_affine,
                        target_affine=affine,
                        output_spatial_shape=spatial_shape)


# %%
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
