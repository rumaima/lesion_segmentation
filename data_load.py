"""
Contains implementations of transforms and dataloaders needed for training, validation and inference.
"""
import numpy as np
import os
from glob import glob
import re
from monai.data import CacheDataset, DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AddChanneld, Compose, LoadImaged, RandCropByPosNegLabeld,
    Spacingd, ToTensord, NormalizeIntensityd, RandFlipd,
    RandRotate90d, RandShiftIntensityd, RandAffined, RandSpatialCropd,
    RandScaleIntensityd)
from scipy import ndimage
from customdataset import CustomDataset
from torch.utils.data import DataLoader
import sys
import torch
import torch.nn as nn
from metrics import dice_metric


def send_to_device(tensor, device):
    """
    Recursively sends the elements in a nested list/tuple/dictionary of tensors to a given device.

    Args:
        tensor (nested list/tuple/dictionary of :obj:`torch.Tensor`):
            The data to send to a given device.
        device (:obj:`torch.device`):
            The device to send the data to

    Returns:
        The same data structure as :obj:`tensor` with all tensors sent to the proper device.
    """
    if isinstance(tensor, (list, tuple)):
        return type(tensor)(send_to_device(t, device) for t in tensor)
    elif isinstance(tensor, dict):
        return type(tensor)({k: send_to_device(v, device) for k, v in tensor.items()})
    elif not hasattr(tensor, "to"):
        return tensor
    return tensor.to(device)

class ForeverDataIterator:
    r"""A data iterator that will never stop producing data"""

    def __init__(self, data_loader: DataLoader, device=None):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)
        self.device = device

    def __next__(self):
        try:
            data = next(self.iter)
            
            if self.device is not None:
                data = send_to_device(data, self.device)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
            if self.device is not None:
                data = send_to_device(data, self.device)
        return data

    def __len__(self):
        return len(self.data_loader)

class UniformDataset(torch.utils.data.Dataset):
    def __init__(self, datasets=[]):
        super(UniformDataset, self).__init__()
        self.datasets = datasets
        total = 0
        for ds in datasets:
            total += len(ds)
        self.total = total
        self.domain_id = 0
        self.domain_ids = []

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        #domain_id = np.random.choice(len(self.datasets))
        domain_id = self.domain_id % len(self.datasets)
        self.domain_id += 1
        self.domain_id = self.domain_id % len(self.datasets)
        idx = idx % len(self.datasets[domain_id])
        return self.datasets[domain_id][idx]
    
def get_train_transforms():
    """ Get transforms for training on FLAIR images and ground truth:
    - Loads 3D images from Nifti file
    - Adds channel dimention
    - Normalises intensity
    - Applies augmentations
    - Crops out 32 patches of shape [96, 96, 96] that contain lesions
    - Converts to torch.Tensor()
    """
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            NormalizeIntensityd(keys=["image"], nonzero=True),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandCropByPosNegLabeld(keys=["image", "label"],
                                   label_key="label", image_key="image",
                                   spatial_size=(128,128,128), num_samples=1,
                                   pos=4, neg=1),
            RandSpatialCropd(keys=["image", "label"],
                                roi_size=(96,96,96),
                             random_center=True, random_size=False),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=(0, 1, 2)),
            RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 1)),
            RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(1, 2)),
            RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 2)),
            RandAffined(keys=['image', 'label'], mode=('bilinear', 'nearest'),
                        prob=1.0, spatial_size=(96,96,96),
                        rotate_range=(np.pi / 12, np.pi / 12, np.pi / 12),
                        scale_range=(0.1, 0.1, 0.1), padding_mode='border'),
            ToTensord(keys=["image", "label"]),
        ]
    )


def get_val_transforms(keys=["image", "label"], image_keys=["image"]):
    """ Get transforms for testing on FLAIR images and ground truth:
    - Loads 3D images and masks from Nifti file
    - Adds channel dimention
    - Applies intensity normalisation to scans
    - Converts to torch.Tensor()
    """
    return Compose(
        [
            LoadImaged(keys=keys),
            AddChanneld(keys=keys),
            NormalizeIntensityd(keys=image_keys, nonzero=True),
            RandSpatialCropd(keys=keys,
                                roi_size=(96, 96, 96),
                             random_center=True, random_size=False),
            # RandAffined(keys=keys, mode=('bilinear', 'nearest'),
            #             prob=1.0, spatial_size=(96, 96, 96),
            #             rotate_range=(np.pi / 12, np.pi / 12, np.pi / 12),
            #             scale_range=(0.1, 0.1, 0.1), padding_mode='border'),                
            ToTensord(keys=keys),
        ]
    )


def get_train_dataloader(root_path, num_workers, source, target, tasks, indices=None, cache_rate=0.1):
    """
    Get dataloader for training 
    Args:
      flair_path: `str`, path to directory with FLAIR images from Train set.
      gts_path:  `str`, path to directory with ground truth lesion segmentation 
                    binary masks images from Train set.
      num_workers:  `int`,  number of worker threads to use for parallel processing
                    of images
      cache_rate:  `float` in (0.0, 1.0], percentage of cached data in total.
    Returns:
      monai.data.DataLoader() class object.
    """
  
    domain_list = tasks.split(",")
    domain_ids = []
    dataset_list = []
    all_domains = source+','+target
    all_domains = all_domains.split(",")

    

    for task in domain_list:
        flair_path = os.path.join(root_path,task,'flair')
        gts_path = os.path.join(root_path,task,'gt')

        # if indices is None:
        flair = sorted(glob(os.path.join(flair_path, "*FLAIR_isovox.nii.gz")),
                    key=lambda i: int(re.sub('\D', '', i)))  # Collect all flair images sorted
        segs = sorted(glob(os.path.join(gts_path, "*gt_isovox.nii.gz")),
                    key=lambda i: int(re.sub('\D', '', i)))  # Collect all corresponding ground truths
        # else:
        #     flair = sorted([os.path.join(flair_path, str(i) + '_FLAIR_isovox.nii.gz') for i in indices])
        #     segs = sorted([os.path.join(gts_path, str(i) + '_gt_isovox.nii.gz') for i in indices])

        files = [{"image": fl, "label": seg} for fl, seg in zip(flair, segs)]

        print("Number of training files:", len(files))

        print('In train loader',files[0].keys())
        
        ds = Dataset(data=files, transform=get_train_transforms(),
                    domain_index=all_domains.index(task), val=False, pred=False)
        domain_ids += [all_domains.index(task)] * len(ds)
        dataset_list.append(ds)
        
    x = UniformDataset(dataset_list)
    x.domain_ids = domain_ids
                      
    # return DataLoader(x, batch_size=1, shuffle=True,num_workers=num_workers)
    return x

def get_val_dataloader(root_path, num_workers, source, target, tasks, indices=None,cache_rate=0.1,bm_path=None):
    """
    Get dataloader for training 
    Args:
      flair_path: `str`, path to directory with FLAIR images from Train set.
      gts_path:  `str`, path to directory with ground truth lesion segmentation 
                    binary masks images from Train set.
      num_workers:  `int`,  number of worker threads to use for parallel processing
                    of images
      cache_rate:  `float` in (0.0, 1.0], percentage of cached data in total.
    Returns:
      monai.data.DataLoader() class object.
    """
  
    domain_list = tasks.split(",")
    domain_ids = []
    dataset_list = []
    all_domains = source+','+target
    all_domains = all_domains.split(",")

    for task in domain_list:
        flair_path = os.path.join(root_path,task,'flair')
        gts_path = os.path.join(root_path,task,'gt')
        bms_path = os.path.join(root_path,task,'fg_mask')
        
        # if indices is None:
        flair = sorted(glob(os.path.join(flair_path, "*FLAIR_isovox.nii.gz")),
                    key=lambda i: int(re.sub('\D', '', i)))  # Collect all flair images sorted
        segs = sorted(glob(os.path.join(gts_path, "*gt_isovox.nii.gz")),
                    key=lambda i: int(re.sub('\D', '', i)))  # Collect all corresponding ground truths
        
        if bm_path is not None:
            bms = sorted(glob(os.path.join(bms_path, "*isovox_fg_mask.nii.gz")),
                    key=lambda i: int(re.sub('\D', '', i)))  # Collect all foreground masks sorted
            
            assert len(flair) == len(segs) == len(bms), f"Some files must be missing: {[len(flair), len(segs), len(bms)]}"

            files = [{"image": fl, "label": seg, "brain_mask": bm} for fl, seg, bm in zip(flair, segs, bms)]

            val_transforms = get_val_transforms(keys=["image", "label", "brain_mask"])

        else:
            assert len(flair) == len(segs), f"Some files must be missing: {[len(flair), len(segs)]}"
            
            files = [{"image": fl, "label": seg} for fl, seg in zip(flair, segs)]

            val_transforms = get_val_transforms()
        # else:
        #     flair = sorted([os.path.join(flair_path, str(i) + '_FLAIR_isovox.nii.gz') for i in indices])
        #     segs = sorted([os.path.join(gts_path, str(i) + '_gt_isovox.nii.gz') for i in indices])

        #     if bm_path is not None:
        #         bms = sorted([os.path.join(bms_path, str(i) + '_isovox_fg_mask.nii.gz') for i in indices])

        #         assert len(flair) == len(segs) == len(bms), f"Some files must be missing: {[len(flair), len(segs), len(bms)]}"

        #         files = [{"image": fl, "label": seg, "brain_mask": bm} for fl, seg, bm in zip(flair, segs, bms)]

        #         val_transforms = get_val_transforms(keys=["image", "label", "brain_mask"])

        #     else:
        #         assert len(flair) == len(segs), f"Some files must be missing: {[len(flair), len(segs)]}"
                
        #         files = [{"image": fl, "label": seg} for fl, seg in zip(flair, segs)]

        #         val_transforms = get_val_transforms()

        print("Number of validation files:", len(files))

        print('In val loader',files[0].keys())
        ds = Dataset(data=files, transform=val_transforms,
                    domain_index=all_domains.index(task),val=True, pred=False)
        domain_ids += [all_domains.index(task)] * len(ds)
        dataset_list.append(ds)

    x = UniformDataset(dataset_list)
    x.domain_ids = domain_ids

    # return DataLoader(x, batch_size=1, shuffle=True,num_workers=num_workers)
    return x

def get_org_val_dataloader(flair_path, gts_path, num_workers, cache_rate=0.1, bm_path=None):
    """
    Get dataloader for validation and testing. Either with or without brain masks.

    Args:
      flair_path: `str`, path to directory with FLAIR images.
      gts_path:  `str`, path to directory with ground truth lesion segmentation 
                    binary masks images.
      num_workers:  `int`,  number of worker threads to use for parallel processing
                    of images
      cache_rate:  `float` in (0.0, 1.0], percentage of cached data in total.
      bm_path:   `None|str`. If `str`, then defines path to directory with
                 brain masks. If `None`, dataloader does not return brain masks. 
    Returns:
      monai.data.DataLoader() class object.
    """
    flair = sorted(glob(os.path.join(flair_path, "*FLAIR_isovox.nii.gz")),
                   key=lambda i: int(re.sub('\D', '', i)))  # Collect all flair images sorted
    segs = sorted(glob(os.path.join(gts_path, "*_isovox.nii.gz")),
                  key=lambda i: int(re.sub('\D', '', i)))  # Collect all corresponding ground truths

    if bm_path is not None:
        bms = sorted(glob(os.path.join(bm_path, "*isovox_fg_mask.nii.gz")),
                     key=lambda i: int(re.sub('\D', '', i)))  # Collect all corresponding brain masks

        assert len(flair) == len(segs) == len(bms), f"Some files must be missing: {[len(flair), len(segs), len(bms)]}"

        files = [
            {"image": fl, "label": seg, "brain_mask": bm} for fl, seg, bm
            in zip(flair, segs, bms)
        ]

        val_transforms = get_val_transforms(keys=["image", "label", "brain_mask"])
    else:
        assert len(flair) == len(segs), f"Some files must be missing: {[len(flair), len(segs)]}"

        files = [{"image": fl, "label": seg} for fl, seg in zip(flair, segs)]

        val_transforms = get_val_transforms()

    print("Number of validation files:", len(files))

    ds = CacheDataset(data=files, transform=val_transforms,
                      cache_rate=cache_rate, num_workers=num_workers)
    return DataLoader(ds, batch_size=1, shuffle=False,
                      num_workers=num_workers)


def get_flair_dataloader(flair_path, num_workers, cache_rate=0.1, bm_path=None):
    """
    Get dataloader with FLAIR images only for inference
    
    Args:
      flair_path: `str`, path to directory with FLAIR images from Train set.
      num_workers:  `int`,  number of worker threads to use for parallel processing
                    of images
      cache_rate:  `float` in (0.0, 1.0], percentage of cached data in total.
      bm_path:   `None|str`. If `str`, then defines path to directory with
                 brain masks. If `None`, dataloader does not return brain masks.
    Returns:
      monai.data.DataLoader() class object.
    """
    dataset_list = []
    domain_ids = []
    
    flair = sorted(glob(os.path.join(flair_path, "*FLAIR_isovox.nii.gz")),
                   key=lambda i: int(re.sub('\D', '', i)))  # Collect all flair images sorted

    if bm_path is not None:
        bms = sorted(glob(os.path.join(bm_path, "*isovox_fg_mask.nii.gz")),
                     key=lambda i: int(re.sub('\D', '', i)))  # Collect all corresponding brain masks

        assert len(flair) == len(bms), f"Some files must be missing: {[len(flair), len(bms)]}"

        files = [{"image": fl, "brain_mask": bm} for fl, bm in zip(flair, bms)]

        val_transforms = get_val_transforms(keys=["image", "brain_mask"])
    else:
        files = [{"image": fl} for fl in flair]

        val_transforms = get_val_transforms(keys=["image"])

    print("Number of FLAIR files:", len(files))

    ds = Dataset(data=files, transform=val_transforms,
                 domain_index = 3, val=False, pred=True)
    domain_ids += [3] * len(ds)
    dataset_list.append(ds)

    x = UniformDataset(dataset_list)
    x.domain_ids = domain_ids

    return x
    # return DataLoader(ds, batch_size=1, shuffle=False,
    #                   num_workers=num_workers)


def remove_connected_components(segmentation, l_min=9):
    """
    Remove all lesions with less or equal amount of voxels than `l_min` from a 
    binary segmentation mask `segmentation`.
    Args:
      segmentation: `numpy.ndarray` of shape [H, W, D], with a binary lesions segmentation mask.
      l_min:  `int`, minimal amount of voxels in a lesion.
    Returns:
      Binary lesion segmentation mask (`numpy.ndarray` of shape [H, W, D])
      only with connected components that have more than `l_min` voxels.
    """
    labeled_seg, num_labels = ndimage.label(segmentation)
    label_list = np.unique(labeled_seg)
    num_elements_by_lesion = ndimage.labeled_comprehension(segmentation, labeled_seg, label_list, np.sum, float, 0)

    seg2 = np.zeros_like(segmentation)
    for i_el, n_el in enumerate(num_elements_by_lesion):
        if n_el > l_min:
            current_voxels = np.stack(np.where(labeled_seg == i_el), axis=1)
            seg2[current_voxels[:, 0],
                 current_voxels[:, 1],
                 current_voxels[:, 2]] = 1
    return seg2
