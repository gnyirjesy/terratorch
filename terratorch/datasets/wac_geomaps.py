from terratorch.datasets.utils import validate_bands
from torchgeo.datasets import NonGeoDataset

import numpy as np
import pandas as pd
import h5netcdf
import torch

from pathlib import Path
from typing import Any, Sequence

from torchgeo.datasets.utils import percentile_normalization
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle

from typing import Dict, Optional

# def normalize_percentile_tensor(img):
#     # Output is 0-1 range tensor
#     x = img.permute(1,2,0).cpu().numpy()
#     x = percentile_normalization(x)
#     return torch.from_numpy(x).permute(2,0,1).float()
GEOLABEL_UNIT_MAPPING = {
        1: 10,
        2: 10,
        3: 10,
        4: 14,
        5: 0,
        6: 0,
        7: 13,
        8: 0,
        9: 11,
        10: 11,
        11: 2,
        12: 2,
        13: 1,
        14: 1,
        15: 1,
        16: 1,
        17: 1,
        18: 6,
        19: 6,
        20: 2,
        21: 2,
        22: 1,
        23: 2,
        24: 3,
        25: 3,
        26: 3,
        27: 2,
        28: 2,
        29: 2,
        30: 2,
        31: 2,
        32: 2,
        33: 6,
        34: 1,
        35: 6,
        36: 6,
        37: 4,
        38: 4,
        39: 4,
        40: 4,
        41: 5,
        42: 4,
        43: 7,
        44: 7,
        45: 7,
        46: 12,
        47: 12,
        48: 8,
        49: 9,
    }

# Just terrain
# GEOLABEL_UNIT_MAPPING = {1: 0, 2: 0, 3: 0, 4: 3, 5: 0, 6: 0, 7: 2, 8: 0, 9: 3, 10: 3, 11: 1, 12: 1, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 3, 19: 3, 20: 1, 21: 1, 22: 0, 23: 1, 24: 2, 25: 2, 26: 2, 27: 1, 28: 1, 29: 1, 30: 1, 31: 1, 32: 1, 33: 3, 34: 0, 35: 3, 36: 3, 37: 1, 38: 1, 39: 1, 40: 1, 41: 0, 42: 1, 43: 3, 44: 3, 45: 3, 46: 1, 47: 1, 48: 0, 49: 3}

# Building mapping for the geolabels
def build_geolabel_lut(mapping_dict, ignore_index, extra_ignore_values=(255,)):
    # force to int
    ignore_index = int(ignore_index)
    extra_ignore_values = tuple(int(v) for v in extra_ignore_values)
    mapping_dict = {int(k): int(v) for k, v in mapping_dict.items()}

    # Fast Path
    if ignore_index < 0 and ignore_index > -10:
        offset = -ignore_index  # offset becomes int

        valid_keys = list(mapping_dict.keys())
        max_key = max(max(valid_keys), max(extra_ignore_values))

        lut_size = int(max_key + offset + 1)   # <--- cast here

        lut = np.full(lut_size, fill_value=ignore_index, dtype=np.int32)

        for k, v in mapping_dict.items():
            lut[k + offset] = v

        for iv in extra_ignore_values:
            lut[iv + offset] = ignore_index

        return {
            "mode": "offset",
            "lut": lut,
            "offset": offset,
            "ignore_index": ignore_index,
        }

    # Slow Path
    mapping_arr = np.full(max(mapping_dict.keys()) + 1, ignore_index, dtype=np.int32)
    for k, v in mapping_dict.items():
        mapping_arr[k] = v

    return {
        "mode": "mask",
        "mapping_arr": mapping_arr,
        "ignore_index": ignore_index,
        "extra_ignore_values": extra_ignore_values,
    }

def apply_geolabel_lut(label_arr, lut_cfg):
    """
    Apply LUT to label_arr using the chosen mode.
    label_arr must be int32.
    """
    label_arr = label_arr.astype(np.int32)
    if lut_cfg["mode"] == "offset":
        lut = lut_cfg["lut"]
        offset = lut_cfg["offset"]
        return lut[label_arr + offset]

    elif lut_cfg["mode"] == "mask":
        mapping_arr = lut_cfg["mapping_arr"]
        ignore_index = lut_cfg["ignore_index"]
        extra_ignore_values = lut_cfg["extra_ignore_values"]

        label_out = np.full_like(label_arr, ignore_index)

        # valid mask: labels within mapping_arr
        valid = (label_arr >= 0) & (label_arr < len(mapping_arr))
        label_out[valid] = mapping_arr[label_arr[valid]]

        # extra ignore values
        for iv in extra_ignore_values:
            label_out[label_arr == iv] = ignore_index

        return label_out

    else:
        raise ValueError("Invalid lut_cfg mode.")


def percentile_norm(img: torch.Tensor) -> torch.Tensor:
    """Apply percentile normalization on CHW tensor."""
    # Output is 0-1 range tensor
    img_np = img.cpu().numpy().transpose(1,2,0)
    img_np = percentile_normalization(img_np)
    return torch.from_numpy(img_np.transpose(2,0,1)).float()

class WACVisGeomaps(NonGeoDataset):
    """NonGeo dataset implementation for WAC visible data and Robbins crater detection"""
    all_band_names = ("415","566", "604", "643", "689")

    # num_classes = 2 # crater class and non-crater class
    splits = {"train": "training", "val": "validation", "test": "testing"}


    def __init__(
        self,
        wac_data_root: str,
        labels_root: str,
        splits_path: str,
        num_classes: int,
        split: str = "train", 
        bands: Sequence[str] = ("415", "415", "415"),
        percentile_normalize: bool = False,
        transforms= None,
        no_data_replace: float | None = 0,
        no_label_replace: float | None = -1,
        geolabel_unit_mapping: Optional[Dict[int, int]] = None,
        **_,
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
        """
        super().__init__()
        # pdb.set_trace()
        if split not in self.splits:
            msg = f"Incorrect split '{split}', please choose one of {self.splits}."
            raise ValueError(msg)
        
        self.split = split
        self.num_classes = num_classes

        validate_bands(bands, self.all_band_names)
        self.bands = bands
        self.band_indices = np.asarray([self.all_band_names.index(b) for b in bands])

        self.wac_data_root = Path(wac_data_root)
        self.labels_root = Path(labels_root)
        self.no_data_replace = no_data_replace
        self.no_label_replace = no_label_replace
        self.percentile_normalize = percentile_normalize
        self.transforms = transforms

        if geolabel_unit_mapping:
            self.geolabel_unit_mapping = geolabel_unit_mapping
        else:
            self.geolabel_unit_mapping =GEOLABEL_UNIT_MAPPING

        # Use the parquet that contains the WAC_VIS_TILE with a path to the filename
        df = pd.read_parquet(splits_path)
        self.imgs = [
            p.replace("vis/", "") for p in df[df["dataset"] == split]["WAC_VIS_TILE"]
        ]

        # print(">>> sanity: mapping_dict[4] =", geolabel_unit_mapping.get(4))
        # print(">>> sanity: mapping_dict[23] =", geolabel_unit_mapping.get(23))
        # print(">>> sanity: mapping_dict[25] =", geolabel_unit_mapping.get(25))

        self.lut_cfg = build_geolabel_lut(
            self.geolabel_unit_mapping,
            ignore_index=self.no_label_replace,   # -1
            extra_ignore_values=(255,)
        )
        # print(">>> lut_cfg mode:", self.lut_cfg["mode"])

        # if self.lut_cfg["mode"] == "offset":
        #     lut = self.lut_cfg["lut"]; off = self.lut_cfg["offset"]
        #     print(">>> LUT[4+off] =", lut[4 + off])     # should be 14
        #     print(">>> LUT[23+off] =", lut[23 + off])   # should be 2
        #     print(">>> LUT[25+off] =", lut[25 + off])   # should be 3
        # elif self.lut_cfg["mode"] == "mask":
        #     arr = self.lut_cfg["mapping_arr"]
        #     print(">>> ARR[4] =", arr[4])               # should be 14
        #     print(">>> ARR[23] =", arr[23])             # should be 2
        #     print(">>> ARR[25] =", arr[25])             # should be 3

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        # pdb.set_trace()
        fname = self.imgs[index]
        image = self._load_h5(fname, self.bands, self.wac_data_root, nan_replace=self.no_data_replace)
        image = torch.from_numpy(image).float()

        label = self._load_h5(fname, ['GEOMAP'], self.labels_root, nan_replace=self.no_label_replace)
        # print(label)
        # Remap to the smaller number of label categories
        label = label.astype(np.int32)
        # print(f"{fname} labels before lut: {np.unique(label)}\n")
        label = apply_geolabel_lut(label, self.lut_cfg)
        label = torch.from_numpy(label).long()
        # print(f"{fname} lalels after lut: {np.unique(label)}\n")

        # print(f"pre-%norm image min {fname}: {image.min()} | max: {image.max()} | mean: {image.mean()} | std: {image.std()}\n")
        if self.percentile_normalize:
            image = percentile_norm(image)
        # print(f"post-transform image min {fname}: {image.min()} | max: {image.max()} | mean: {image.mean()} | std: {image.std()}\n")

        sample = {"image": image, "mask": label}

        if self.transforms is not None:
            # Albumentations expects HWC numpy
            img = sample["image"].cpu().numpy().transpose(1,2,0)
            mask = sample["mask"].squeeze(0).cpu().numpy()

            transformed = self.transforms(image=img, mask=mask)

            # image
            out_img = transformed["image"]
            if isinstance(out_img, np.ndarray):
                out_img = torch.from_numpy(out_img).float().permute(2,0,1)
            else:  # already a torch tensor (ToTensorV2 case)
                if out_img.ndim == 3 and out_img.shape[-1] == 3:   # HWC → CHW
                    out_img = out_img.permute(2,0,1)
            image_t = out_img.float()

            # mask
            out_mask = transformed["mask"]
            if isinstance(out_mask, np.ndarray):
                out_mask = torch.from_numpy(out_mask).long()
            mask_t = out_mask.unsqueeze(0)

            sample["image"] = image_t
            sample["mask"]  = mask_t

        sample['filename'] = fname

        return sample


    def _load_h5(self, filename: str, bands: Sequence[str], root: Path, nan_replace):
        """Load a stack of bands from HDF5."""
        with h5netcdf.File(root / filename) as ds:
            arr = np.stack([ds[b][()] for b in bands])
            if nan_replace is not None:
                arr = np.nan_to_num(arr, nan=nan_replace)
        return arr
    
    def _remap_vals(self, image: np.ndarray):
        pass

    def plot(
        self,
        sample: dict[str, torch.Tensor],
        suptitle: str | None = None,
    ):
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        ignore_index = self.no_label_replace
        num_classes = self.num_classes
        num_images = 4 # legend, wac vis, labels, labels on wac_vis
        image = percentile_normalization(sample['image'].permute(1, 2, 0).numpy())
        label = sample['mask'].permute(1,2,0).numpy()

        prediction = sample.get("prediction", None)
        if prediction is not None:
            num_images += 1
            pred_arr = prediction
            if hasattr(pred_arr, "cpu"):  # torch tensor
                pred_arr = pred_arr.cpu().numpy()
        else:
            pred_arr = None

        # Map the ignore_index to an extra label at the end of the labels
        vis_label = label.copy()
        vis_label[vis_label == ignore_index] = num_classes

        if pred_arr is not None:
            vis_pred = pred_arr.copy()
            vis_pred[vis_pred == ignore_index] = num_classes

        # Create colormap with one extra for the ignore index
        base_cmap = mpl.colormaps.get_cmap("jet").resampled(num_classes)
        colors = list(base_cmap(range(num_classes)))  # range -> indices into LUT
        colors.append((0, 0, 0, 1.0))
        extended_cmap = mpl.colors.ListedColormap(colors)
        norm = mpl.colors.Normalize(vmin=0, vmax=num_classes)

        fig, ax = plt.subplots(1, num_images, figsize=(12, 5), layout="compressed")

        ax[0].axis("off")

        ax[1].axis("off")
        ax[1].title.set_text("Image")
        ax[1].imshow(image)

        ax[2].axis("off")
        ax[2].title.set_text("Ground Truth Labels")
        ax[2].imshow(vis_label, cmap=extended_cmap, norm=norm)

        ax[3].axis("off")
        ax[3].title.set_text("GT Labels on Image")
        ax[3].imshow(image)
        ax[3].imshow(vis_label, cmap=extended_cmap, alpha=0.3, norm=norm)

        if pred_arr is not None:
            ax[4].axis("off")
            ax[4].title.set_text("Predicted Labels")
            ax[4].imshow(vis_pred, cmap=extended_cmap, norm=norm)

        legend_data = [(i, colors[i], str(i)) for i in range(num_classes)]
        legend_data.append((num_classes, colors[num_classes], "nan"))

        handles = [Rectangle((0, 0), 1, 1, color=c) for _, c, _ in legend_data]
        labels  = [n for _, _, n in legend_data]
        ax[0].legend(handles, labels, loc="center")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig