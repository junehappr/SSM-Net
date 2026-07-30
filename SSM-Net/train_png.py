import datetime
import os
from functools import partial

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset_png import PngSegmentationDataset, png_segmentation_collate
from nets.model_factory import get_model
from nets.unet_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import LossHistory
from utils.utils import seed_everything, worker_init_fn
from utils.utils_fit import fit_one_epoch


def set_rgb_edge_order(model):
    for module in model.modules():
        if hasattr(module, "edge_channel_order"):
            module.edge_channel_order = "rgb"


if __name__ == "__main__":
    Cuda = True
    seed = 42
    fp16 = False

    model_name = "segman_swin_u2net"  # only supported model in this project
    num_classes = 2
    input_channels = 3
    input_shape = [256, 256]
    pretrained = False
    model_path = ""

    Init_Epoch = 0
    UnFreeze_Epoch = 200
    batch_size = 8

    Init_lr = 1e-4
    Min_lr = Init_lr * 0.01
    optimizer_type = "adam"
    momentum = 0.9
    weight_decay = 1e-4
    lr_decay_type = "cos"

    dice_loss = True
    tversky_loss    = False
    tversky_alpha   = 0.3
    tversky_beta    = 0.7
    focal_loss = False
    cls_weights = np.ones([num_classes], np.float32)

    save_period = 2
    save_dir = r"train_result_png_segman_swin_u2net"
    early_stopping_patience = 20
    num_workers = 4

    dataset_root = r"C:\Users\kd\Desktop\WJ_model\traindatasat\png_datasat"
    missing_mask_as_negative = True
    train_image_dir = os.path.join(dataset_root, "images", "train")
    train_mask_dir = os.path.join(dataset_root, "masks", "train")
    val_image_dir = os.path.join(dataset_root, "images", "val")
    val_mask_dir = os.path.join(dataset_root, "masks", "val")

    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and Cuda else "cpu")
    local_rank =  0
    rank = 0
    fp16 = fp16 and device.type == "cuda"
    os.makedirs(save_dir, exist_ok=True)

    model = get_model(model_name, num_classes, input_channels, pretrained, img_size=input_shape[0])
    set_rgb_edge_order(model)

    if not pretrained:
        weights_init(model)

    if model_path:
        print(f"Load weights {model_path}.")
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for key, value in pretrained_dict.items():
            if key in model_dict and np.shape(model_dict[key]) == np.shape(value):
                temp_dict[key] = value
                load_key.append(key)
            else:
                no_load_key.append(key)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        print(f"Successful Load Key Num: {len(load_key)}")
        print(f"Fail To Load Key Num: {len(no_load_key)}")

    time_str = datetime.datetime.strftime(datetime.datetime.now(), "%Y_%m_%d_%H_%M_%S")
    log_dir = os.path.join(save_dir, "loss_" + str(time_str))
    loss_history = LossHistory(log_dir, model, input_shape=input_shape)

    if fp16:
        try:
            scaler = torch.amp.GradScaler("cuda")
        except (AttributeError, TypeError):
            from torch.cuda.amp import GradScaler
            scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()
    if device.type == "cuda":
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    train_dataset = PngSegmentationDataset(
        train_image_dir,
        train_mask_dir,
        input_shape,
        num_classes,
        train=True,
        binary_mask=(num_classes == 2),
        missing_mask_as_negative=missing_mask_as_negative,
    )
    val_dataset = PngSegmentationDataset(
        val_image_dir,
        val_mask_dir,
        input_shape,
        num_classes,
        train=False,
        binary_mask=(num_classes == 2),
        missing_mask_as_negative=missing_mask_as_negative,
    )

    num_train = len(train_dataset)
    num_val = len(val_dataset)
    print(f"Num Train: {num_train}, Num Val: {num_val}")
    print(f"Current Model: {model_name}")
    print(f"Input: RGB {input_shape[0]}x{input_shape[1]}")

    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size
    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("Dataset is too small for the selected batch_size.")

    nbs = 16
    lr_limit_max = 1e-4 if optimizer_type in ["adam", "adamw"] else 1e-1
    lr_limit_min = 1e-6 if optimizer_type in ["adam", "adamw"] else 5e-4
    Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    optimizer = {
        "adam": optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
        "adamw": optim.AdamW(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
        "sgd": optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True, weight_decay=weight_decay),
    }[optimizer_type]
    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

    gen = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=png_segmentation_collate,
        worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed),
    )
    gen_val = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=png_segmentation_collate,
        worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed),
    )

    best_val_loss = float("inf")
    early_stopping_counter = 0

    for epoch in range(Init_Epoch, UnFreeze_Epoch):
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
        fit_one_epoch(
            model_train,
            model,
            loss_history,
            None,
            optimizer,
            epoch,
            epoch_step,
            epoch_step_val,
            gen,
            gen_val,
            UnFreeze_Epoch,
            device.type == "cuda",
            dice_loss,
            focal_loss,
            cls_weights,
            num_classes,
            fp16,
            scaler,
            save_period,
            save_dir,
            local_rank,
            tversky_loss,
            tversky_alpha,
            tversky_beta,
        )

        current_val_loss = loss_history.val_loss[-1]
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            print(f"Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")
            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break

    loss_history.writer.close()
