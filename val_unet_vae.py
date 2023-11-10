
import time
import torch
import numpy as np
from common.utils.meter import AverageMeter, ProgressMeter
from joblib import Parallel
from metrics import dice_metric
from torch import nn

def validate(val_loader, val_iter, model, args, device, act, loss_function) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Validation: ')
    is_val = True
    # switch to evaluate mode
    model.eval()
    count = 0
    ''' Evaluatioin loop '''
    with Parallel(n_jobs=args.n_jobs) as parallel_backend:
        with torch.no_grad():
            end = time.time()
            metric_sum = 0.0
            metric_count = 0
            thresh = args.threshold
            dice_loss = []
            focal_loss = []
            validationEpoch_loss = []

            for i in range(len(val_loader)):
                validation_loss = []
                count+=1
                inputs_v, labels_v, brain_mask_v, d_v = next(val_iter)
                # inputs, gt, brain_mask = (
                #     batch_data["image"].to(device),
                #     batch_data["label"].cpu().numpy(),
                #     batch_data["brain_mask"].cpu().numpy()
                # )
                inputs_v = inputs_v.to(device)
                labels_v = labels_v.to(device)
                brain_mask_v = brain_mask_v.to(device)
                d_v = d_v.to(device)

                data_input = (inputs_v, d_v, True, True)
                # data_input = inputs_v
                # get ensemble predictions
                all_outputs = []
                print(inputs_v.shape)
                val_out_sigmoid, val_outputs, _, _, _, _, _, _ = model(data_input)
                
                
                gt = np.squeeze(labels_v.cpu().numpy())
                # gt= np.squeeze(gt)

                outputs_seg = act(val_outputs).cpu().numpy()
                outputs_seg = np.squeeze(outputs_seg[0])
                outputs_seg[outputs_seg >= thresh] = 1
                outputs_seg[outputs_seg < thresh] = 0

                print(outputs_seg.shape)
                print(labels_v.shape)

                val_acc = dice_metric(ground_truth=gt.flatten(), predictions=outputs_seg.flatten())

                # print('val acc: ', val_acc)
                

                metric_count += 1
                metric_sum += val_acc.sum().item()

                

                #calculating the validation loss
                dice_loss.append(loss_function(val_out_sigmoid, labels_v))
                ce_loss = nn.BCEWithLogitsLoss(reduction='none')
                
                ce = ce_loss(val_outputs,labels_v)
                pt = torch.exp(-ce)
                loss2 = (1 - pt)**args.gamma_focal * ce 
                focal_loss.append(torch.mean(loss2))

                #mean dice loss
                mean_loss_dice = torch.stack(dice_loss, dim=0).mean()

                #mean focal loss
                mean_loss_focal = torch.stack(focal_loss, dim=0).mean()

                #total loss
                loss = args.dice_weight * mean_loss_dice + args.focal_weight * mean_loss_focal

                validation_loss.append(loss.item())

                # if i % args.print_freq == 0:
                    # progress.display(i)

            validationEpoch_loss.append(np.array(validation_loss).mean())

            # print('metric_sum: ', metric_sum)
            # print('metric_count: ', metric_count)
            
            metric = metric_sum / metric_count

    validationEpoch_loss = "{:.4f}".format(validationEpoch_loss[-1])
    print('Validation Loss:', validationEpoch_loss)
            
    return metric, validationEpoch_loss

      