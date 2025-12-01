from collections.abc import Callable
from typing import Any, ClassVar
from torch import Tensor
from functools import partial
from terratorch.datasets.wac_geomaps import WACVisGeomaps
from torchgeo.datamodules import NonGeoDataModule
from collections.abc import Sequence
import yaml
import albumentations as A
from albumentations.pytorch import transforms as T
from torch.utils.data import DataLoader
import torch
import numpy as np
from terratorch.datamodules.utils import wrap_in_compose_is_list
import pdb

# def apply_transforms(sample, transforms):
#     # Change shape for albumentations
#     image = sample["image"].permute(1,2,0).cpu().numpy() # Change from (C,H,W) to (H,W,C)
#     label = sample["mask"].squeeze(0).cpu().numpy()
#     # Final transformed['image'] shape should be C, H, W
#     transformed = transforms(
#         image=image,
#         mask=label
#     )

#     out_img = transformed["image"]
#     out_mask = transformed["mask"]

#     # Albumentations returns CHW if ToTensorV2 is in pipeline.
#     # Otherwise, it returns HWC.
#     if isinstance(out_img, np.ndarray):
#         if out_img.ndim == 3 and out_img.shape[-1] == sample["image"].shape[0]:
#             # HWC -> CHW
#             out_img = out_img.transpose(2,0,1)
#         out_img = torch.from_numpy(out_img).float()

#     out_mask = torch.from_numpy(out_mask).long().unsqueeze(0)

#     return {
#         "image": out_img,
#         "mask": out_mask
#     }

class Normalize(Callable):
    def __init__(self, means, stds, max_pixel_value=None):
        super().__init__()
        self.means = means
        self.stds = stds
        self.max_pixel_value = max_pixel_value

    def __call__(self, batch):
        # batch['image']=torch.stack(tuple(batch["image"]))
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
        batch["image"] = (image - means) / stds
        return batch

class WACVisGeomapsDataModule(NonGeoDataModule):

    num_classes = 15

    # categories = ("background", 
    #               "crater",)

    def __init__(
        self,
        wac_data_root: str,
        labels_root: str,
        stats_path: str,
        splits_path: str,
        num_classes: int,
        bands: Sequence[str] = ("415","415","415"),
        train_transform: A.Compose | None | list[A.BasicTransform] = None,
        val_transform: A.Compose | None | list[A.BasicTransform] = None,
        test_transform: A.Compose | None | list[A.BasicTransform] = None,
        no_data_replace: float | None = 0,
        no_label_replace: float | None = -1,
        batch_size: int = 4,
        num_workers: int = 0,
        # image_size=300,
        apply_norm_in_datamodule=True, # objectdetection tasks assume normalization in datamodule
        percentile_normalize = False, # Normalize in dataset percentile based
        **kwargs):

        super().__init__(WACVisGeomaps,
                        batch_size=batch_size,
                        num_workers= num_workers,
                        **kwargs
        )
        self.wac_data_root = wac_data_root
        self.labels_root = labels_root
        self.stats_path = stats_path
        with open(stats_path, 'r') as f:
            self.stats = yaml.safe_load(f)

        self.splits_path = splits_path
        self.num_classes = num_classes
        self.bands = bands

        self.train_transform = wrap_in_compose_is_list(train_transform) if train_transform else None
        self.val_transform   = wrap_in_compose_is_list(val_transform) if val_transform else None
        self.test_transform  = wrap_in_compose_is_list(test_transform) if test_transform else None

        self.no_data_replace = no_data_replace
        self.no_label_replace = no_label_replace
        self.batch_size = batch_size
        self.num_workers = num_workers

        if apply_norm_in_datamodule:
            means = tuple(self.stats[b]['mean'] for b in self.bands)
            stds  = tuple(self.stats[b]['std']  for b in self.bands)
            # Apply with the imagenet values - the data before this point should be scaled 0-1
            # self.aug = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), max_pixel_value=1) # ImageNet defaults
            self.aug = Normalize(means=means, stds=stds, max_pixel_value=1)
        else:
            self.aug = Normalize((0, 0, 0), (1, 1, 1), max_pixel_value=1)

        self.percentile_normalize = percentile_normalize

    def setup(self, stage: str) -> None:

        if stage in ["fit"]:
            self.train_dataset = WACVisGeomaps(
                split= "train",
                wac_data_root=self.wac_data_root, 
                labels_root=self.labels_root,
                splits_path = self.splits_path,
                num_classes = self.num_classes,
                bands= self.bands,
                percentile_normalize = self.percentile_normalize,
                transforms=self.train_transform,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace
            )            
        if stage in ["fit", "validate"]:
            self.val_dataset = WACVisGeomaps(
                split="val", 
                wac_data_root=self.wac_data_root, 
                labels_root=self.labels_root,
                splits_path = self.splits_path,
                num_classes = self.num_classes,
                bands= self.bands,
                percentile_normalize = self.percentile_normalize,
                transforms=self.val_transform,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace
            )    
        if stage in ["test"]:
            self.test_dataset = WACVisGeomaps(
                split="test", 
                wac_data_root=self.wac_data_root, 
                labels_root=self.labels_root,
                splits_path = self.splits_path,
                num_classes = self.num_classes,
                bands= self.bands,
                percentile_normalize = self.percentile_normalize,
                transforms=self.test_transform,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace
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