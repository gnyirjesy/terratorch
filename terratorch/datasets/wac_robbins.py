from torchvision.transforms import v2 # apply the v2.normalize code
from terratorch.datasets.utils import validate_bands
from pathlib import Path
import numpy as np
from torchgeo.datasets import NonGeoDataset
from typing import Any
import h5netcdf
import xarray as xr
import numpy as np
import hdf5plugin
from collections.abc import Sequence
from torchgeo.datasets.utils import lazy_import, percentile_normalization
from torchgeo.datasets.vhr10 import ConvertCocoAnnotations
from torchgeo.datasets.vhr10 import convert_coco_poly_to_mask
import torch
import yaml
import json
import pdb
from torch import Tensor
from matplotlib.figure import Figure
from matplotlib import patches
import matplotlib.pyplot as plt
from functools import lru_cache
import pdb

@lru_cache(maxsize=None)
def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)

def normalize_percentile_tensor(img):
    # Output is 0-1 range tensor
    x = img.permute(1,2,0).cpu().numpy()
    x = percentile_normalization(x)
    return torch.from_numpy(x).permute(2,0,1).float()

def normalize_image(img):
    x = img.cpu().numpy()
    x = (x-x.mean())/x.std()
    return torch.from_numpy(x).float()

class WACVisRobbins(NonGeoDataset):
    """NonGeo dataset implementation for WAC visible data and Robbins crater detection"""
    all_band_names = ("415","566", "604", "643", "689")
    rgb_bands = ("415", "415", "415")

    BAND_SETS = {"all": all_band_names, "rgb": rgb_bands}

    num_classes = 2 # crater class and non-crater class
    splits = {"train": "training", "val": "validation", "test": "testing"}

    categories = ("background", 
                  "crater",)

    def __init__(
        self,
        wac_data_root: str, # /rtmp/hpatil/data/WAC_ELV_SLP_SIN_COS/vis
        stats_path: str,
        splits_path: str,
        annotations_path: str,
        split: str = "train", 
        bands: Sequence[str] = BAND_SETS["rgb"],
        percentile_normalize: bool = False,
        transforms: v2.Compose | None = None,
        no_data_replace: float | None = 0,
        boxes_output_tag='boxes',
        labels_output_tag='labels',
        masks_output_tag='masks',
        scores_output_tag='scores',
    ) -> None:
        """Constructor

        Args:
            data_root (str): Path to the data root directory.
            bands (list[str]): Bands that should be output by the dataset. Defaults to all bands.
            transform (A.Compose | None): Albumentations transform to be applied.
                Should end with ToTensorV2(). If used through the corresponding data module,
                should not include normalization. Defaults to None, which applies ToTensorV2().
            no_data_replace (float | None): Replace nan values in input images with this value.
                If None, does no replacement. Defaults to 0.
            use_metadata (bool): whether to return metadata info (time and location).
        """
        super().__init__()
        if split not in self.splits:
            msg = f"Incorrect split '{split}', please choose one of {self.splits}."
            raise ValueError(msg)
        split_name = self.splits[split]
        self.split = split

        validate_bands(bands, self.all_band_names)
        self.bands = bands
        self.band_indices = np.asarray([self.all_band_names.index(b) for b in bands])

        self.wac_data_root = wac_data_root
        self.no_data_replace = no_data_replace
        self.transforms = transforms

        with open(stats_path, 'r') as f:
            self.stats = yaml.safe_load(f)

        pc = lazy_import('pycocotools.coco')
        self.coco = pc.COCO()                 # init empty
        self.coco.dataset = load_json(annotations_path) 
        self.coco.createIndex()
        self.coco_convert = ConvertCocoAnnotations()

        splits = load_json(splits_path) 
        
        split_imgs = splits[self.split]
        self.ids = [int(k) for k in sorted(split_imgs.keys(), key=lambda x: int(x))]

        self.percentile_normalize = percentile_normalize
        self.boxes_output_tag = boxes_output_tag
        self.labels_output_tag = labels_output_tag
        self.masks_output_tag = masks_output_tag
        self.scores_output_tag = scores_output_tag

    def __len__(self) -> int:
        return(len(self.ids))

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        img_id = int(self.ids[index])
        # HAD TO MAKE THIS CHANGE FOR CCC
        filename = self.coco.loadImgs([img_id])[0]['file_name'].replace("vis/","")
        image = self._load_file(filename, nan_replace=self.no_data_replace)
        image = torch.from_numpy(image.to_numpy()).float()

        sample: dict[str, Any] = {
            # Image is (C, H, W)
            "image": image,
            "label": self._load_target(img_id),
        }

        if sample['label']['annotations']:
            # pdb.set_trace()
            sample = self.coco_convert(sample)
            # Tensor that is [n_boxes, 4] -> set to zero if no boxes
            sample[self.boxes_output_tag] = sample['label']['boxes']
            # Tensor that is [n_craters, H, W] -> set to zero if no craters
            sample[self.masks_output_tag] = sample['label']['masks']
            # Tensor that is n_boxes long -> set to zero if no craters
            sample[self.labels_output_tag] = sample['label']['labels']
            if self.labels_output_tag != 'label':
                del sample['label']
        
        # # Percentile normalize to deal with data issues:
        if self.percentile_normalize:
            img = normalize_percentile_tensor(sample['image']) # version 25
            sample['image'] = img

        if self.transforms is not None:
            sample = self.transforms(sample)

        sample['filename'] = filename

        return sample


    def _load_file(self, filename: str, nan_replace: int | float | None = None) -> xr.DataArray:
        """Load a single image.

        Modified from torchgeo.datasets.vhr10.py

        Args:
            id_: unique ID of the image

        Returns:
            the image
        
        """
        path = Path(self.wac_data_root)/filename

        with h5netcdf.File(path) as ds:
            bands = [ds[b][()] for b in self.bands]
            arr = np.stack(bands)
            xp = ds['x'][()]
            yp = ds['y'][()]
            if nan_replace is not None:
                arr = np.nan_to_num(arr, nan=nan_replace)
        ds_base = xr.DataArray(
            data = arr,
            dims=["band","y","x"],
            coords={"y": yp, "x": xp}
                    )
        ds_base=ds_base.transpose('band', 'y', 'x')

        return ds_base
    
    def _load_target(self, img_id: int) -> dict[str, Any]:
        """Load the annotations for a single image.
        
        Modified from torchgeo.datasets.vhr10.py

        Args:
            id_: unique ID of the image

        Returns:
            the annotations
        """
        # pdb.set_trace()
        annot = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[int(img_id)]))
        target = dict(image_id=int(img_id), annotations=annot)

        return target

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: str | None = None,
        show_feats: str | None = 'both',
        box_alpha: float = 0.7,
        mask_alpha: float = 0.7,
    ) -> Figure:
        """Plot a sample from the dataset.

        Code taken from torchgeo.datasets.vhr10.py

        Args:
            sample: a sample returned by :meth:`__getitem__`
            suptitle: optional string to use as a suptitle
            show_titles: flag indicating whether to show titles above each panel
            show_feats: optional string to pick features to be shown: boxes, masks, both
            box_alpha: alpha value of box
            mask_alpha: alpha value of mask

        Returns:
            a matplotlib Figure with the rendered sample

        Raises:
            AssertionError: if ``show_feats`` argument is invalid
            DependencyNotFoundError: If plotting masks and scikit-image is not installed.

        .. versionadded:: 0.4
        """
        assert show_feats in {'boxes', 'masks', 'both'}
        image = percentile_normalization(sample['image'].permute(1, 2, 0).numpy())

        if show_feats != 'boxes':
            skimage = lazy_import('skimage')

        boxes = sample[self.boxes_output_tag].cpu().numpy()
        labels = sample[self.labels_output_tag].cpu().numpy()
        masks = []
        if self.masks_output_tag in sample:
            M = sample[self.masks_output_tag]
            if isinstance(M, torch.Tensor):
                if M.ndim == 2:  # single mask, no batch dim
                    masks = [M.cpu().numpy()]
                elif M.ndim == 3:  # multiple masks: (N, H, W)
                    masks = [m.cpu().numpy() for m in M]
            elif isinstance(M, (list, tuple)):
                masks = [
                    (m.cpu().numpy() if isinstance(m, torch.Tensor) else np.asarray(m))
                    for m in M
                ]

        n_gt = len(boxes)

        ncols = 1
        show_predictions = 'prediction_' + self.labels_output_tag in sample

        if show_predictions:
            show_pred_boxes = False
            show_pred_masks = False
            prediction_labels = sample['prediction_' + self.labels_output_tag].numpy()
            prediction_scores = sample['prediction_' + self.scores_output_tag].numpy()
            if 'prediction_' + self.boxes_output_tag in sample:
                prediction_boxes = sample['prediction_' + self.boxes_output_tag].numpy()
                show_pred_boxes = True
            if 'prediction_' + self.masks_output_tag in sample:
                prediction_masks = sample['prediction_' + self.masks_output_tag].numpy()
                show_pred_masks = True

            n_pred = len(prediction_labels)
            ncols += 1

        # Display image
        fig, axs = plt.subplots(ncols=ncols, squeeze=False, figsize=(ncols * 10, 13))
        axs[0, 0].imshow(image)
        axs[0, 0].axis('off')

        cm = plt.get_cmap('gist_rainbow')
        for i in range(n_gt):
            class_num = labels[i]
            color = cm(class_num / len(self.categories))

            # Add bounding boxes
            x1, y1, x2, y2 = boxes[i]
            if show_feats in {'boxes', 'both'}:
                r = patches.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=2,
                    alpha=box_alpha,
                    linestyle='dashed',
                    edgecolor=color,
                    facecolor='none',
                )
                axs[0, 0].add_patch(r)

            # Add labels
            label = self.categories[class_num]
            caption = label
            axs[0, 0].text(
                x1, y1 - 8, caption, color='white', size=11, backgroundcolor='none'
            )

            # Add masks
            if show_feats in {'masks', 'both'} and self.masks_output_tag in sample:
                mask = masks[i]
                axs[0,0].imshow(np.ma.masked_where(mask == 0, mask),
                cmap="Reds", alpha=mask_alpha, zorder=3)

            if show_titles:
                axs[0, 0].set_title('Ground Truth')

        if show_predictions:
            axs[0, 1].imshow(image)
            axs[0, 1].axis('off')
            for i in range(n_pred):
                score = prediction_scores[i]
                if score < 0.1:
                    continue

                class_num = prediction_labels[i]
                color = cm(class_num / len(self.categories))

                if show_pred_boxes:
                    # Add bounding boxes
                    x1, y1, x2, y2 = prediction_boxes[i]
                    r = patches.Rectangle(
                        (x1, y1),
                        x2 - x1,
                        y2 - y1,
                        linewidth=2,
                        alpha=box_alpha,
                        linestyle='dashed',
                        edgecolor=color,
                        facecolor='none',
                    )
                    axs[0, 1].add_patch(r)

                    # Add labels
                    label = self.categories[class_num]
                    caption = f'{label} {score:.3f}'
                    axs[0, 1].text(
                        x1,
                        y1 - 8,
                        caption,
                        color='white',
                        size=11,
                        backgroundcolor='none',
                    )

                # Add masks
                if show_pred_masks:
                    mask = prediction_masks[i][0]
                    contours = skimage.measure.find_contours(mask, 0.5)
                    for verts in contours:
                        verts = np.fliplr(verts)
                        p = patches.Polygon(
                            verts, facecolor=color, alpha=mask_alpha, edgecolor='white'
                        )
                        axs[0, 1].add_patch(p)

            if show_titles:
                axs[0, 1].set_title('Prediction')

        if suptitle is not None:
            plt.suptitle(suptitle)

        plt.tight_layout()

        return fig
