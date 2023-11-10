"""
@author: Originally by Francesco La Rosa
         Adapted by Vatsal Raina, Nataliia Molchanova
"""

import argparse
import os
import torch
import shutil
import time
from torch import nn
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
# from monai.networks.nets import UNet
from torch.optim.lr_scheduler import LambdaLR
# from torch.utils.data import DataLoader
# from monai.data import CacheDataset, DataLoader
import torch.nn.functional as F
import numpy as np
import random
from metrics import dice_metric
from data_load import get_train_dataloader, get_val_dataloader, ForeverDataIterator, remove_connected_components
from torchsummary import summary
from unet_model_for_loop import UNet, iVAE, MLP
from torch.utils.data import DataLoader
from joblib import Parallel
from metrics import dice_norm_metric, lesion_f1_score, ndsc_aac_metric
from uncertainty import ensemble_uncertainties_classification
from matplotlib import pyplot as plt
from importlib import reload
import wandb
import logging
    


def get_default_device():
    """ Set device """
    if torch.cuda.is_available():
        print("Got CUDA!")
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    


    

parser = argparse.ArgumentParser(description='Get all command line arguments.')
# trainining

parser.add_argument('--n_epochs', type=int, default=10, 
                    help='Specify the number of epochs to train for')
# initialisation                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
parser.add_argument('--seed', type=int, default=1, help='Specify the global random seed')

# data
parser.add_argument('--path_train_data', type=str, default="/l/users/umaima.rahman/shifts/data/",
                    help='Specify the path to the training data files directory')
parser.add_argument('--path_train_gts', type=str, 
                    help='Specify the path to the training gts files directory')
parser.add_argument('--path_val_data', type=str, required=False, 
                    help='Specify the path to the validation data files directory')
parser.add_argument('--path_val_gts', type=str, required=False, 
                    help='Specify the path to the validation gts files directory')
parser.add_argument('--num_workers', type=int, default=10, 
                    help='Number of workers')
# logging
parser.add_argument('--path_save', type=str, default="/l/users/umaima.rahman/shifts/checkpoints/debugging/", 
                    help='Specify the path to the trained model will be saved')
parser.add_argument('--plot_path_save', type=str, default="/l/users/umaima.rahman/shifts/output_partial/", 
                    help='Specify the path to the trained model will be saved')

parser.add_argument('--output_logs', default="/l/users/umaima.rahman/shifts/logs/", 
                    help='Specify the path to the output logs will be saved')
parser.add_argument('--resume', action="store_true", help='')
parser.add_argument('--resume_checkpoint_path', type=str, default="/l/users/umaima.rahman/shifts/checkpoints/debugging/model_finetuning_0.pth")
parser.add_argument('--val_interval', type=int, default=10, 
                    help='Validation every n-th epochs')
parser.add_argument('--threshold', type=float, default=0.4, 
                    help='Probability threshold')
parser.add_argument('--source', type=str, default="train,dev_in,eval_in",
                    help='Source domains')
parser.add_argument('--target', type=str, default="dev_out", 
                    help='Target domain')
parser.add_argument('--lr', type=float, default=1e-3, 
                    help='Specify the initial learning rate')
parser.add_argument('--lr-gamma', default=0.0003, type=float, help='parameter for lr scheduler')
parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-3)',
                    dest='weight_decay')
parser.add_argument('--n_jobs', type=int, default=1,
                    help='Number of parallel workers for F1 score computation')
parser.add_argument('--z_dim', type=int, default=1024, metavar='N')
parser.add_argument('--train_batch_size', default=1, type=int)
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
parser.add_argument('-i', '--iters_per_epoch', default=1, type=int,
                        help='Number of iterations per epoch')
parser.add_argument('-d', '--divide_step', default=100, type=int,
                        help='Number of iterations per epoch')                       
# parser.add_argument('--root', type=str, default='../da_datasets/domainnet',
#                         help='root path of dataset')
# parser.add_argument('-d', '--data', metavar='DATA', default='DomainNet', choices=utils.get_dataset_names(),
#                     help='dataset: ' + ' | '.join(utils.get_dataset_names()) +
#                          ' (default: Office31)')
# parser.add_argument('-s', '--source', help='source domain(s)', default='i,p,q,r,s')
# parser.add_argument('-t', '--target', help='target domain(s)', default='c')


def main(args):
    
    seed_val = args.seed
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    
    device = get_default_device()
    torch.multiprocessing.set_sharing_strategy('file_system')

    path_save = args.path_save

    #now we will Create and configure logger 
    logging.basicConfig(filename="outputs_for_loop.log", 
					format='%(asctime)s %(message)s', 
					filemode='w') 

    #Let us Create an object 
    logger=logging.getLogger() 

    #Now we are going to Set the threshold of logger to DEBUG 
    logger.setLevel(logging.DEBUG) 
    
    '''' Initialise dataloaders '''
    # train_loader = get_train_dataloader(flair_path=args.path_train_data, 
    #                                     gts_path=args.path_train_gts, 
    #                                     num_workers=args.num_workers)
    # val_loader = get_val_dataloader(flair_path=args.path_val_data, 
    #                                 gts_path=args.path_val_gts, 
    #                                 num_workers=args.num_workers)

    'shuffling'

    total_files = 25
    ind_vec = np.arange(1, 26)
    np.random.shuffle(ind_vec)
    train_ind_vec = ind_vec[0:15]
    val_ind_vec = ind_vec[15:]


    '''Initialise dataloaders for different domains'''

    train_source_dataset = get_train_dataloader(root_path=args.path_train_data,
                                        num_workers=args.num_workers,
                                        source = args.source,
                                        target = args.target,
                                        tasks = args.source)
    # before is right
    train_target_dataset = get_train_dataloader(root_path=args.path_train_data,
                                        num_workers=args.num_workers,
                                        source = args.source,
                                        target = args.target,
                                        tasks = args.target)
    
    train_source_loader = DataLoader(train_source_dataset, batch_size=args.train_batch_size,
                                     num_workers=args.num_workers, drop_last=True,
                                     #sampler=_make_balanced_sampler(train_source_dataset.domain_ids)
                                     shuffle=True,
                                     )
    
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.train_batch_size,
                                     shuffle=True, num_workers=args.num_workers, drop_last=True,
                                     )

    val_dataset = get_val_dataloader(root_path=args.path_train_data,
                                        num_workers=args.num_workers,
                                        source = args.source,
                                        target = args.target,
                                        tasks = args.target,
                                        bm_path= args.path_train_data)
    
    val_loader = DataLoader(val_dataset, batch_size=args.train_batch_size,
                            shuffle=True, num_workers=args.num_workers, drop_last=True,
                            )  
    
    val_iter = ForeverDataIterator(val_loader)                                

    train_source_iter = ForeverDataIterator(train_source_loader)
    
    train_target_iter = ForeverDataIterator(train_target_loader)
    
  
    ''' Initialise the model '''

    model = UNet(args).to(device)

    # classifier = iVAE(args, backbone_net=None).to(device)
    
    #multiple gpus
    # parallel_net = nn.DataParallel(model)
    # parallel_net = parallel_net.to(device)
    # model = parallel_net
    
    loss_function = DiceLoss(to_onehot_y=True, 
                             softmax=True, sigmoid=False,
                             include_background=False)
    
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    #define the scheduler modification UR
    # print(optimizer.param_groups[0]['lr'], ' *** lr')
    lr_scheduler = LambdaLR(optimizer, lambda x:  args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    # print(optimizer.param_groups[0]['lr'], ' *** lr')
    # test_logger = '%s/test.txt' % (args.log)

    act = nn.Softmax(dim=1)
    
    epoch_num = args.n_epochs
    val_interval = args.val_interval
    thresh = args.threshold
    gamma_focal = 2.0
    dice_weight = 0.5
    focal_weight = 1
    kl_weight = 1e-3
    roi_size = (96, 96, 96)
    divide_step = args.divide_step
    
    best_loss = 1
    best_metric, best_metric_epoch = -1, -1
    epoch_loss_values, metric_values = [], []
    n_domains = 4

    # start training
    best_acc1 = 0.
    total_iter = 0
    trainingEpoch_loss = []
    validationEpoch_loss = []
    
    start_epoch = 0
 
    if args.resume:
        print("=> loading checkpoint ")
        checkpoint = torch.load(args.resume_checkpoint_path)
        print(checkpoint.keys())
        start_epoch = checkpoint['last_epoch']
        lr_scheduler.load_state_dict(checkpoint['scheduler'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    
    ''' Training loop '''
    for epoch in range(start_epoch, epoch_num):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epoch_num}")

        # train for every epoch
        model.train()

        epoch_loss = 0
        step = 0
        end = time.time()
        total_iter = 0


        for step in range(args.iters_per_epoch):
            model.train()
            
            images_s, labels_s, d_s = next(train_source_iter)
            images_t, labels_t, d_t = next(train_target_iter)


            n_samples = images_s.size(0)
            
            img_all = torch.cat([images_s, images_t], 0).to(device)
            label_all = torch.cat([labels_s, labels_t], 0).to(device)
            d_all = torch.cat([d_s, d_t], 0).to(device)

            dice_loss = []
            focal_loss = []
            kl_loss = []
            loss_kl  =0
            is_val = False
            
            
            for id in range(n_domains):
                domain_id = id
                is_target = domain_id == n_domains-1
                """
                if id == 0:
                    index = (d_all != target_domain_id)
                else:
                    index = (d_all == target_domain_id)
                """
                index = d_all == id
                label_dom = label_all[index] if not is_target else None
                img_dom = img_all[index]
                d_dom = d_all[index]
                
                #img_dom.shape = torch.Size([2, 1, 96, 96, 96]), d_dom.shape = torch.Size([2])
                if len(img_dom) != 0:    
                    data_input = (img_dom, d_dom, is_target, is_val)
                    outputs_sigmoid, outputs, kl = model(data_input)


                    if not is_target:
                        # Dice loss
                        dice_loss.append(loss_function(outputs, label_dom))
                        
                        # Focal loss
                        ce_loss = nn.BCEWithLogitsLoss(reduction='none')
                        ce = ce_loss(outputs,label_dom)
                        pt = torch.exp(-ce)
                        loss2 = (1 - pt)**gamma_focal * ce 
                        focal_loss.append(torch.mean(loss2))

                        # KL Loss
                        C = torch.clamp(torch.tensor(args.C_max) / args.C_stop_iter * total_iter, 0, args.C_max)
                        loss_kl = args.beta * (kl - C).abs()
                        kl_loss.append(loss_kl)

        #mean kl loss
        mean_loss_kl = torch.stack(kl_loss, dim=0).mean()

        #mean dice loss
        mean_loss_dice = torch.stack(dice_loss, dim=0).mean()

        #mean focal loss
        mean_loss_focal = torch.stack(focal_loss, dim=0).mean()

        #total loss
        # loss = dice_weight * mean_loss_dice + focal_weight * mean_loss_focal + kl_weight * mean_loss_kl
        loss =  focal_weight * mean_loss_focal + kl_weight * mean_loss_kl
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step() 

        # epoch_loss += loss.item()
        # if step % divide_step == 0:
        #     # step_print = int(step/2)
        #     print(f"{step}/{iters_per_epoch}, train_loss: {loss.item():.4f}, dice_loss: {mean_loss_dice:.4f}, focal_loss: {mean_loss_focal:.4f}")

        epoch_loss = loss.mean()
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        logger.info(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")



        # ''' Validation on the target data'''
        if (epoch + 1) % val_interval == 0:
            model.eval()
            img_t = img_all[d_all==n_domains-1]
            labels_t = label_all[d_all==n_domains-1]
            d_t = d_all[d_all==n_domains-1]

            sw_batch_size = 4

            target_input = (img_t, d_t, True, True)
            train_target_acc = []  
            with torch.no_grad():
                metric_sum = 0.0
                metric_count = 0

                _, train_tar_out, kl_tar = model(target_input)
 
                seg = train_tar_out[0,0].cpu().numpy()
                seg[seg >= 0] = 1
                seg[seg < 0] = 0

                gt = np.squeeze(labels_t.cpu().numpy())
            
                train_tar_acc = dice_metric(ground_truth=gt.flatten(), predictions=seg.flatten())
                train_target_acc.append(train_tar_acc)
            
            model.train()
        
        total_iter += args.iters_per_epoch

        count = 0
        ''' Evaluatioin loop '''
        with Parallel(n_jobs=args.n_jobs) as parallel_backend:
            with torch.no_grad():
                metric_sum = 0.0
                metric_count = 0

                dice_loss = []
                focal_loss = []
                validation_loss = []
                kl_loss = []
                loss_kl = 0
                
                for i in range(len(val_loader)):
                    
                    count+=1
                    inputs_v, labels_v, brain_mask_v, d_v , _= next(val_iter)
                   
                    inputs_v = inputs_v.to(device)
                    labels_v = labels_v.to(device)
                    brain_mask_v = brain_mask_v.to(device)
                    d_v = d_v.to(device)

                    val_input = (inputs_v, d_v, True, True)
                    # get ensemble predictions
                    all_outputs = []

                    _, val_outputs, kl_val = model(val_input)

                    gt = np.squeeze(labels_v.cpu().numpy())
        
                    outputs_seg = val_outputs[0,0].cpu().numpy()
                    outputs_seg[outputs_seg >= 0] = 1
                    outputs_seg[outputs_seg < 0] = 0

                
                    val_acc = dice_metric(ground_truth=gt.flatten(), predictions=outputs_seg.flatten())
                    print("dice score: ", val_acc)
                    metric_count += 1
                    metric_sum += val_acc.sum().item()

                    #calculating the validation loss
                    dice_loss.append(loss_function(val_outputs, labels_v))
                    ce_loss = nn.BCEWithLogitsLoss(reduction='none')
                    
                    ce = ce_loss(outputs,labels_v)
                    pt = torch.exp(-ce)
                    loss2 = (1 - pt)**gamma_focal * ce 
                    focal_loss.append(torch.mean(loss2))

                    # KL Loss
                    C = torch.clamp(torch.tensor(args.C_max) / args.C_stop_iter * total_iter, 0, args.C_max)
                    loss_kl = args.beta * (kl_val - C).abs()
                    kl_loss.append(loss_kl)

                #mean kl loss
                mean_loss_kl = torch.stack(kl_loss, dim=0).mean()

                #mean dice loss
                mean_loss_dice = torch.stack(dice_loss, dim=0).mean()

                #mean focal loss
                mean_loss_focal = torch.stack(focal_loss, dim=0).mean()

                #total loss
                loss = focal_weight * mean_loss_focal + kl_weight * mean_loss_kl

                checkpoint = {}
                metric = metric_sum / metric_count
                metric_values.append(metric)
                
                # if metric > best_metric:
                #     best_metric = metric
                if loss < best_loss:
                    best_loss = loss
                    best_metric_epoch = epoch + 1
                    checkpoint["model_state_dict"]= model.state_dict()
                    checkpoint["optimizer_state_dict"] = optimizer.state_dict()
                    checkpoint["scheduler"] = lr_scheduler.state_dict()
                    torch.save(checkpoint, os.path.join(path_save, f"Best_model_finetuning_{epoch}.pth"))
                    print("saved new best metric model")
                print(f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                                    f"\nbest mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}"
                                    )
                                
                if (epoch+1) % divide_step == 0:
                    checkpoint["model_state_dict"]= model.state_dict()
                    checkpoint["optimizer_state_dict"] = optimizer.state_dict()
                    checkpoint["scheduler"] = lr_scheduler.state_dict()
                    checkpoint["last_epoch"] = epoch
                    torch.save(checkpoint, os.path.join(path_save, f"model_finetuning_{epoch}.pth"))
                    print("saved new model")


                
    plt.plot(trainingEpoch_loss, label='train_loss')    
    plt.plot(validationEpoch_loss,label='val_loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(range(1, epoch_num))
    plt.title('Train and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(args.plot_path_save,f'loss_plot_seed{args.seed}.png'))
    plt.show   


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

