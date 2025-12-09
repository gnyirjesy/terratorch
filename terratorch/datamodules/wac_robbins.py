from torchgeo.datasets.utils import (
    Path,
    check_integrity,
    download_and_extract_archive,
    download_url,
    lazy_import,
    percentile_normalization,
)

from collections.abc import Callable
from typing import Any, ClassVar

from torch import Tensor
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib import patches
from functools import partial

from terratorch.datasets.wac_robbins import WACVisRobbins

from torchgeo.datamodules import NonGeoDataModule

from collections.abc import Sequence
from torchvision.transforms import v2
import yaml

import albumentations as A
from albumentations.pytorch import transforms as T
import torchvision.transforms as orig_transforms

from torch.utils.data import DataLoader

import torch
from torch import nn
import numpy as np

import pdb

def collate_fn_detection(batch, boxes_tag='boxes', labels_tag='labels', masks_tag='masks'):
    images = [item["image"] for item in batch]
    new_batch = {
        "image": images,
        boxes_tag: [item[boxes_tag] for item in batch],
        labels_tag: [item[labels_tag] for item in batch],
        masks_tag: [item[masks_tag] for item in batch],
    }
    return new_batch


def get_transform(train, image_size=300, pad=True, labels_tag='labels'):

    transforms = []
    if pad:
        transforms.append(A.PadIfNeeded(min_height=image_size, min_width=image_size, value=0, border_mode=0))
    else:
        transforms.append(A.Resize(height=image_size, width=image_size))
    if train:
        transforms.append(A.CenterCrop(width=image_size, height=image_size))
        transforms.append(A.HorizontalFlip(p=0.5))
    else:
        transforms.append(A.CenterCrop(width=image_size, height=image_size, p=1.0))
    transforms.append(T.ToTensorV2())
    # print(labels_tag)
    return A.Compose(transforms, bbox_params=A.BboxParams(format="pascal_voc", label_fields=[labels_tag]), is_check_shapes=False)

def apply_transforms(sample, transforms, boxes_tag='boxes', labels_tag='labels', masks_tag='masks'):
    # pdb.set_trace()
    # Change shape for albumentations
    image = sample["image"].permute(1,2,0).cpu().numpy() # Change from (C,H,W) to (H,W,C)
    masks = [m.cpu().numpy().astype(np.uint8) for m in sample[masks_tag]]
    boxes = sample[boxes_tag].cpu().numpy().astype(np.float32)
    labels = sample[labels_tag].cpu().numpy().astype(np.int64)
    # Final transformed['image'] shape should be C, H, W
    transformed = transforms(
        image=image,
        masks=masks,
        bboxes=boxes,
        labels=labels
    )

    # If crop causes bbox/mask misalignment then skip the sample and return and empty target
    if len(transformed["masks"]) != len(transformed["bboxes"]):
        return {
            "image": torch.from_numpy(transformed["image"]).float().permute(2,0,1),
            boxes_tag: torch.zeros((0,4), dtype=torch.float32),
            labels_tag: torch.zeros((0,), dtype=torch.int64),
            masks_tag: [],
        }
    # ----- HANDLE EMPTY CASE (Albumentations dropped everything) -----
    if len(transformed["masks"]) == 0:
        img = transformed["image"]
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).float().permute(2,0,1)

        return {
            "image": img,
            boxes_tag: torch.zeros((0,4), dtype=torch.float32),
            labels_tag: torch.zeros((0,), dtype=torch.int64),
            masks_tag: [],
        }  
    
    # Normal case 
    img = transformed["image"]
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
    # Ensure CHW
    if img.ndim == 3 and img.shape[0] != sample["image"].shape[0]:
        img = img.permute(2,0,1)
    image_t = img.float()

    boxes = np.array(transformed["bboxes"], dtype=np.float32).reshape(-1,4)
    labels = np.array(transformed["labels"], dtype=np.int64)
    masks = transformed["masks"]

    # Filter invalid boxes/masks together
    keep = []
    for m, b in zip(masks, boxes):
        if m.sum() == 0:
            keep.append(False)
        elif b[2] <= b[0] or b[3] <= b[1]:
            keep.append(False)
        else:
            keep.append(True)

    boxes = boxes[keep]
    labels = labels[keep]
    masks = [m for m,k in zip(masks, keep) if k]
    masks_t = []
    for m in masks:
        if isinstance(m, np.ndarray):
            masks_t.append(torch.from_numpy(m).to(torch.uint8))
        elif isinstance(m, torch.Tensor):
            masks_t.append(m.to(torch.uint8))
        else:
            raise TypeError(f"Unexpected mask type: {type(m)}")

    return {
        "image": image_t,
        boxes_tag: torch.from_numpy(boxes),
        labels_tag: torch.from_numpy(labels),
        masks_tag: masks_t,
    }


class Normalize(Callable):
    def __init__(self, means, stds, max_pixel_value=None):
        super().__init__()
        self.means = means
        self.stds = stds
        self.max_pixel_value = max_pixel_value

    def __call__(self, batch):
        # pdb.set_trace() # Confirmed do use this
        batch['image']=torch.stack(tuple(batch["image"]))
        image = batch["image"]/self.max_pixel_value if self.max_pixel_value is not None else batch["image"]
        if len(image.shape) == 5:
            means = torch.tensor(self.means, device=image.device).view(1, -1, 1, 1, 1)
            stds = torch.tensor(self.stds, device=image.device).view(1, -1, 1, 1, 1)
        elif len(image.shape) == 4:
            means = torch.tensor(self.means, device=image.device).view(1, -1, 1, 1)
            stds = torch.tensor(self.stds, device=image.device).view(1, -1, 1, 1)
        else:
            msg = f"Expected batch to have 5 or 4 dimensions, but got {len(image.shape)}"
            raise Exception(msg)
        # pdb.set_trace() 
        batch["image"] = (image - means) / stds
        # for m in batch['masks'][0][:3]:
        #     print("*****batch check", m.dtype, m.min().item(), m.max().item(), m.shape)
        # pdb.set_trace()
        return batch

class IdentityTransform(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class WACVisRobbinsDataModule(NonGeoDataModule):

    all_band_names = ("415","566", "604", "643", "689")
    rgb_bands = ("415", "415", "415")

    CHANNELS = {"vis": ("415","566", "604", "643", "689"),
                "aspect": ("ASPECT_SIN","ASPECT_COS"),
                "slope": ("SLOPE",),
                 "dtm": ("DTM",),
                 }

    num_classes = 2 # crater class and non-crater class
    splits = {"train": "training", "val": "validation"}

    categories = ("background", 
                  "crater",)

    def __init__(
        self,
        data_roots: dict,
        stats_path: str,
        splits_path: str,
        annotations_path: str,
        split: str = 'train',
        channels= CHANNELS,
        # transforms: v2.Compose | None = None,
        no_data_replace: float | None = 0,
        # use_metadata: bool = False,
        batch_size: int = 4,
        num_workers: int = 0,
        # pad = False, # Don't apply padding because all images are 300 x 300 or 301 x 301
        image_size=300,
        collate_fn = None,
        boxes_output_tag='boxes',
        labels_output_tag='labels',
        masks_output_tag='masks',
        scores_output_tag='scores',
        apply_norm_in_datamodule=True, # objectdetection tasks assume normalization in datamodule
        percentile_normalize = False, # Normalize in dataset percentile based
        *args,
        **kwargs):

        super().__init__(WACVisRobbins,
                         batch_size=batch_size,
                         num_workers=num_workers,
                        #  coco_data_root=coco_data_root,
                        splits_path= splits_path,
                        annotations_path= annotations_path, 
                        data_roots=data_roots,
                         split=split,
                         channels=channels,
                         percentile_normalize = percentile_normalize,
                        #  transforms=transforms,
                         no_data_replace=no_data_replace,
                         boxes_output_tag='boxes',
                         labels_output_tag='labels',
                         masks_output_tag='masks',
                         scores_output_tag='scores',
                        #  use_metadata=use_metadata
                         **kwargs
        )
        # pdb.set_trace()
        self.train_transform = partial(apply_transforms,transforms=get_transform(True, image_size, labels_tag=labels_output_tag), boxes_tag=boxes_output_tag, labels_tag=labels_output_tag, masks_tag=masks_output_tag)
        self.val_transform = partial(apply_transforms,transforms=get_transform(False, image_size, labels_tag=labels_output_tag), boxes_tag=boxes_output_tag, labels_tag=labels_output_tag, masks_tag=masks_output_tag)
        self.test_transform = partial(apply_transforms,transforms=get_transform(False, image_size, labels_tag= labels_output_tag), boxes_tag=boxes_output_tag, labels_tag=labels_output_tag, masks_tag=masks_output_tag)

        with open(stats_path, 'r') as f:
            self.stats = yaml.safe_load(f)
        
        self.channels = channels
        self.percentile_normalize = percentile_normalize


        if apply_norm_in_datamodule:
            means = []
            stds = []
            for channel in self.channels:
                means += self.stats[channel]['mean']
                stds += self.stats[channel]['std']
            # Apply with the imagenet values - the data before this point should be scaled 0-1
            # self.aug = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), max_pixel_value=1) # ImageNet defaults
            self.aug = Normalize(means=means, stds=stds, max_pixel_value=1)
        else:
            self.aug = Normalize(
                tuple(torch.zeros(sum(len(value) for value in self.channels.values()))),
                tuple(torch.ones(sum(len(value) for value in self.channels.values()))),
                max_pixel_value=1)

        # self.coco_data_root = coco_data_root
        self.splits_path = splits_path
        self.annotations_path = annotations_path
        self.data_roots = data_roots
        self.stats_path = stats_path
        self.split = split
        self.batch_size = batch_size
        self.num_workers = num_workers
    
        if collate_fn is None:
            self.collate_fn = partial(collate_fn_detection, boxes_tag=boxes_output_tag, labels_tag=labels_output_tag, masks_tag=masks_output_tag) #lambda b: collate_fn_detection(b, boxes_tag=boxes_output_tag, labels_tag=labels_output_tag, masks_tag=masks_output_tag)
        else:
            self.collate_fn = collate_fn

        self.no_data_replace = no_data_replace
        # self.use_metadata = use_metadata

    def setup(self, stage: str) -> None:

        if stage in ["fit"]:
            self.train_dataset = WACVisRobbins(
                # coco_data_root=self.coco_data_root, 
                splits_path = self.splits_path,
                annotations_path = self.annotations_path,
                data_roots=self.data_roots, 
                stats_path=self.stats_path,
                split= "train", 
                channels= self.channels,
                percentile_normalize = self.percentile_normalize,
                transforms=self.train_transform,
                no_data_replace=self.no_data_replace,
                # use_metadata=self.use_metadata,
            )            
        if stage in ["fit", "validate"]:
            self.val_dataset = WACVisRobbins(
                # coco_data_root=self.coco_data_root, 
                splits_path = self.splits_path,
                annotations_path = self.annotations_path,
                data_roots=self.data_roots, 
                stats_path=self.stats_path,
                split="val", 
                channels= self.channels,
                percentile_normalize = self.percentile_normalize,
                transforms=self.val_transform,
                no_data_replace=self.no_data_replace,
                # use_metadata=self.use_metadata,
            )    
        if stage in ["test"]:
            self.test_dataset = WACVisRobbins(
                # coco_data_root=self.coco_data_root, 
                splits_path = self.splits_path,
                annotations_path = self.annotations_path,
                data_roots=self.data_roots,
                stats_path=self.stats_path,
                split="test", 
                channels= self.channels,
                percentile_normalize = self.percentile_normalize,
                transforms=self.test_transform,
                no_data_replace=self.no_data_replace,
                # use_metadata=self.use_metadata,
            )  

    
    def _dataloader_factory(self, split: str) -> DataLoader[dict[str, Tensor]]:
        """Implement one or more PyTorch DataLoaders.

        Args:
            split: Either 'train', 'val', 'test', or 'predict'.

        Returns:
            A collection of data loaders specifying samples.

        Raises:
            MisconfigurationException: If :meth:`setup` does not define a
                dataset or sampler, or if the dataset or sampler has length 0.
        """
        dataset = self._valid_attribute(f"{split}_dataset", "dataset")
        batch_size = self.batch_size

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )