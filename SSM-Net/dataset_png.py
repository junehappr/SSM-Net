import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from utils.utils import cvtColor, preprocess_input


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
MASK_EXTENSIONS = (".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg")


def _find_by_stem(folder, stem, extensions):
    folder = Path(folder)
    for ext in extensions:
        path = folder / f"{stem}{ext}"
        if path.exists():
            return str(path)
    return None


def list_png_pairs(image_dir, mask_dir, missing_mask_as_negative=False):
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image folder not found: {image_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask folder not found: {mask_dir}")

    pairs = []
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        mask_path = _find_by_stem(mask_dir, image_path.stem, MASK_EXTENSIONS)
        if mask_path is None:
            if not missing_mask_as_negative:
                raise FileNotFoundError(f"Mask not found for image: {image_path.name}")
            pairs.append((str(image_path), None))
            continue
        pairs.append((str(image_path), mask_path))

    if not pairs:
        raise ValueError(f"No image/mask pairs found in {image_dir} and {mask_dir}")
    return pairs


class PngSegmentationDataset(Dataset):
    def __init__(
        self,
        image_dir,
        mask_dir,
        input_shape,
        num_classes,
        train=True,
        binary_mask=True,
        missing_mask_as_negative=False,
    ):
        super().__init__()
        self.pairs = list_png_pairs(image_dir, mask_dir, missing_mask_as_negative=missing_mask_as_negative)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train = train
        self.binary_mask = binary_mask

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        image = Image.open(image_path)
        if mask_path is None:
            mask = Image.new("L", image.size, 0)
        else:
            mask = Image.open(mask_path)

        image, mask = self.get_random_data(image, mask, self.input_shape, random=self.train)
        image = np.transpose(preprocess_input(np.array(image, np.float32)), [2, 0, 1])
        mask = self._normalize_mask(np.array(mask))

        labels = np.eye(self.num_classes + 1, dtype=np.float32)[mask.reshape([-1])]
        labels = labels.reshape((self.input_shape[0], self.input_shape[1], self.num_classes + 1))
        return image, mask, labels

    def _normalize_mask(self, mask):
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        if self.binary_mask and self.num_classes == 2:
            mask = (mask > 0).astype(np.int64)
        else:
            mask = mask.astype(np.int64)
        mask[mask >= self.num_classes] = self.num_classes
        return mask

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, mask, input_shape, jitter=0.3, hue=0.1, sat=0.7, val=0.3, random=True):
        image = cvtColor(image)
        mask = Image.fromarray(np.array(mask)).convert("L")
        iw, ih = image.size
        h, w = input_shape

        if not random:
            scale = min(w / iw, h / ih)
            nw = max(int(iw * scale), 1)
            nh = max(int(ih * scale), 1)

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new("RGB", (w, h), (128, 128, 128))
            new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

            mask = mask.resize((nw, nh), Image.NEAREST)
            new_mask = Image.new("L", (w, h), 0)
            new_mask.paste(mask, ((w - nw) // 2, (h - nh) // 2))
            return new_image, new_mask

        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(0.25, 2)
        if new_ar < 1:
            nh = max(int(scale * h), 1)
            nw = max(int(nh * new_ar), 1)
        else:
            nw = max(int(scale * w), 1)
            nh = max(int(nw / new_ar), 1)

        image = image.resize((nw, nh), Image.BICUBIC)
        mask = mask.resize((nw, nh), Image.NEAREST)

        if self.rand() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new("RGB", (w, h), (128, 128, 128))
        new_mask = Image.new("L", (w, h), 0)
        new_image.paste(image, (dx, dy))
        new_mask.paste(mask, (dx, dy))

        image_data = np.array(new_image, np.uint8)
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        hue_ch, sat_ch, val_ch = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        image_data = cv2.merge((cv2.LUT(hue_ch, lut_hue), cv2.LUT(sat_ch, lut_sat), cv2.LUT(val_ch, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        return image_data, new_mask


def png_segmentation_collate(batch):
    images, masks, labels = [], [], []
    for image, mask, label in batch:
        images.append(image)
        masks.append(mask)
        labels.append(label)
    images = torch.from_numpy(np.array(images)).float()
    masks = torch.from_numpy(np.array(masks)).long()
    labels = torch.from_numpy(np.array(labels)).float()
    return images, masks, labels
