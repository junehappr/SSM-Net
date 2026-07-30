import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from dataset_png import list_png_pairs
from nets.model_factory import get_model
from utils.utils import cvtColor, preprocess_input


class Config:
    model_name = "segman_swin_u2net"
    model_path = r"C:\Users\kd\Desktop\WJ_model\unet-pytorch-main_0605-version5\unet-pytorch-main_0426\unet-pytorch-main\train_result_png_segman_swin_u2net\best_epoch_weights.pth"

    dataset_root = r"C:\Users\kd\Desktop\WJ_model\traindatasat\png_datasat"
    split = "val"
    base_save_dir = r"results\png_eval"

    num_classes = 2
    input_channels = 3
    input_shape = [256, 256]
    cuda = True
    binary_mask = True
    missing_mask_as_negative = True
    save_visuals = True
    vis_limit = None


def set_rgb_edge_order(model):
    for module in model.modules():
        if hasattr(module, "edge_channel_order"):
            module.edge_channel_order = "rgb"


def normalize_mask(mask, num_classes, binary_mask=True):
    mask = np.array(mask)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if binary_mask and num_classes == 2:
        mask = (mask > 0).astype(np.int64)
    else:
        mask = mask.astype(np.int64)
    mask[mask >= num_classes] = num_classes
    return mask


def letterbox_image(image, input_shape):
    image = cvtColor(image)
    iw, ih = image.size
    h, w = input_shape
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    dx = (w - nw) // 2
    dy = (h - nh) // 2

    resized = image.resize((nw, nh), Image.BICUBIC)
    boxed = Image.new("RGB", (w, h), (128, 128, 128))
    boxed.paste(resized, (dx, dy))
    return boxed, (iw, ih), (nw, nh), (dx, dy)


def unletterbox_mask(mask, original_size, resized_size, offset):
    iw, ih = original_size
    nw, nh = resized_size
    dx, dy = offset
    crop = mask[dy:dy + nh, dx:dx + nw]
    crop_img = Image.fromarray(crop.astype(np.uint8))
    return np.array(crop_img.resize((iw, ih), Image.NEAREST), dtype=np.int64)


def predict_one(model, image, cfg, device):
    boxed, original_size, resized_size, offset = letterbox_image(image, cfg.input_shape)
    image_data = np.transpose(preprocess_input(np.array(boxed, np.float32)), [2, 0, 1])
    image_tensor = torch.from_numpy(image_data).float().unsqueeze(0).to(device)

    output = model(image_tensor)
    if output.shape[1] == 1:
        pred = (torch.sigmoid(output) > 0.5).long().squeeze(0).squeeze(0)
    else:
        pred = torch.argmax(F.softmax(output, dim=1), dim=1).squeeze(0)
    pred = pred.cpu().numpy().astype(np.uint8)
    return unletterbox_mask(pred, original_size, resized_size, offset)


def fast_hist(gt, pred, num_classes):
    valid = (gt >= 0) & (gt < num_classes)
    return np.bincount(
        num_classes * gt[valid].astype(int) + pred[valid].astype(int),
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)


def colorize_mask(mask):
    colors = np.array(
        [
            [0, 0, 0],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [255, 0, 255],
            [0, 255, 255],
        ],
        dtype=np.uint8,
    )
    return colors[mask % len(colors)]


def save_visual(image, gt, pred, save_dir, stem):
    image = cvtColor(image)
    gt_img = Image.fromarray(colorize_mask(gt))
    pred_img = Image.fromarray(colorize_mask(pred))
    overlay = Image.blend(image.convert("RGB"), pred_img.convert("RGB"), 0.45)

    image.save(os.path.join(save_dir, "original", f"{stem}.png"))
    gt_img.save(os.path.join(save_dir, "ground_truth", f"{stem}.png"))
    pred_img.save(os.path.join(save_dir, "prediction", f"{stem}.png"))
    overlay.save(os.path.join(save_dir, "overlay", f"{stem}.png"))


def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.cuda else "cpu")

    image_dir = os.path.join(cfg.dataset_root, "images", cfg.split)
    mask_dir = os.path.join(cfg.dataset_root, "masks", cfg.split)
    pairs = list_png_pairs(image_dir, mask_dir, missing_mask_as_negative=cfg.missing_mask_as_negative)

    save_dir = os.path.join(cfg.base_save_dir, cfg.model_name, cfg.split)
    for sub in ["original", "ground_truth", "prediction", "overlay"]:
        os.makedirs(os.path.join(save_dir, sub), exist_ok=True)

    model = get_model(
        cfg.model_name,
        cfg.num_classes,
        cfg.input_channels,
        pretrained=False,
        img_size=cfg.input_shape[0],
    )
    set_rgb_edge_order(model)
    model.load_state_dict(torch.load(cfg.model_path, map_location=device))
    model.to(device)
    model.eval()

    hist = np.zeros((cfg.num_classes, cfg.num_classes), dtype=np.float64)

    with torch.no_grad():
        for index, (image_path, mask_path) in enumerate(tqdm(pairs, desc="Predict")):
            image = Image.open(image_path)
            if mask_path is None:
                gt = np.zeros((image.height, image.width), dtype=np.int64)
            else:
                gt = normalize_mask(Image.open(mask_path), cfg.num_classes, cfg.binary_mask)
            pred = predict_one(model, image, cfg, device)
            hist += fast_hist(gt, pred, cfg.num_classes)

            if cfg.save_visuals and (cfg.vis_limit is None or index < cfg.vis_limit):
                stem = Path(image_path).stem
                save_visual(image, gt, pred, save_dir, stem)

    iou = np.diag(hist) / np.maximum(hist.sum(1) + hist.sum(0) - np.diag(hist), 1)
    precision = np.diag(hist) / np.maximum(hist.sum(0), 1)
    recall = np.diag(hist) / np.maximum(hist.sum(1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    miou = np.nanmean(iou)

    print("\n" + "=" * 50)
    print(f"PNG Eval Report: {cfg.model_name}")
    print("=" * 50)
    print(f"mIoU: {miou:.4f}")
    for cls_id in range(cfg.num_classes):
        print(
            f"class {cls_id}: IoU={iou[cls_id]:.4f}, "
            f"Precision={precision[cls_id]:.4f}, Recall={recall[cls_id]:.4f}, F1={f1[cls_id]:.4f}"
        )
    print("=" * 50)

    with open(os.path.join(save_dir, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"Model: {cfg.model_name}\n")
        f.write(f"Weights: {cfg.model_path}\n")
        f.write(f"Split: {cfg.split}\n")
        f.write(f"mIoU: {miou:.6f}\n")
        for cls_id in range(cfg.num_classes):
            f.write(
                f"class {cls_id}: IoU={iou[cls_id]:.6f}, "
                f"Precision={precision[cls_id]:.6f}, Recall={recall[cls_id]:.6f}, F1={f1[cls_id]:.6f}\n"
            )


if __name__ == "__main__":
    main()
