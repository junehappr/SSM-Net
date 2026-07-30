import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial  # [补齐] 之前缺失的引用


def CE_Loss(inputs, target, cls_weights, num_classes=21):
    """
    通用交叉熵损失
    inputs: [N, C, H, W]
    target: [N, H, W] (索引)
    """
    n, c, h, w = inputs.size()

    # 兼容性处理：如果 target 是 one-hot [N, H, W, C]，转回索引 [N, H, W]
    if target.ndim == 4:
        target = torch.argmax(target, dim=-1)

    nt, ht, wt = target.size()

    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    # 转换形状用于 Loss 计算
    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    loss = nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes)(temp_inputs, temp_target)
    return loss


def Focal_Loss(inputs, target, cls_weights, num_classes=21, alpha=0.5, gamma=2):
    n, c, h, w = inputs.size()

    # -----------------------------------------------------------
    # [核心修复] 兼容 [N, H, W] 的 3维 Label
    # -----------------------------------------------------------
    if target.ndim == 3:
        # 转为 One-hot [N, H, W, C]
        target = F.one_hot(target, num_classes=c).float()
    # -----------------------------------------------------------

    nt, ht, wt, ct = target.size()

    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1, ct)

    # 计算 Softmax
    logpt = -nn.CrossEntropyLoss(weight=cls_weights, reduction='none')(temp_inputs, torch.argmax(temp_target, -1))
    pt = torch.exp(logpt)

    if alpha is not None:
        logpt *= alpha

    loss = -((1 - pt) ** gamma) * logpt
    loss = loss.mean()
    return loss


def Dice_loss(inputs, target, beta=1, smooth=1e-5):
    n, c, h, w = inputs.size()

    # -----------------------------------------------------------
    # [核心修复] 兼容 [N, H, W] 的 3维 Label
    # -----------------------------------------------------------
    if target.ndim == 3:
        # target: [N, H, W] -> One-hot [N, H, W, C]
        target = F.one_hot(target, num_classes=c).float()
    # -----------------------------------------------------------

    # 现在可以安全解包了，target 是 [N, H, W, C]
    nt, ht, wt, ct = target.size()

    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    # inputs: [N, C, H, W] -> [N, H, W, C] -> [N*H*W, C]
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    # target: [N, H, W, C] -> [N, H, W, C] -> [N*H*W, C]
    temp_target = target.view(n, -1, ct)

    # --- 针对二分类滑坡检测的修正 ---
    if c == 2:
        # 只计算 class 1 (滑坡) 的 Dice Loss
        tp = torch.sum(temp_target[..., 1:] * temp_inputs[..., 1:], axis=[0, 1])
        fp = torch.sum(temp_inputs[..., 1:], axis=[0, 1]) - tp
        fn = torch.sum(temp_target[..., 1:], axis=[0, 1]) - tp
    else:
        # 多分类，计算所有类别
        tp = torch.sum(temp_target * temp_inputs, axis=[0, 1])
        fp = torch.sum(temp_inputs, axis=[0, 1]) - tp
        fn = torch.sum(temp_target, axis=[0, 1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = torch.mean(score)
    return 1 - score


def Tversky_Loss(inputs, target, alpha=0.3, beta=0.7, smooth=1e-5):
    n, c, h, w = inputs.size()

    if target.ndim == 3:
        target = F.one_hot(target, num_classes=c).float()

    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    temp_target = target.view(n, -1, ct)

    if c == 2:
        temp_inputs = temp_inputs[..., 1:]
        temp_target = temp_target[..., 1:]

    tp = torch.sum(temp_target * temp_inputs, axis=[0, 1])
    fp = torch.sum((1 - temp_target) * temp_inputs, axis=[0, 1])
    fn = torch.sum(temp_target * (1 - temp_inputs), axis=[0, 1])

    score = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1 - torch.mean(score)


# ---------------------------------------------------
# 辅助函数：学习率与权重初始化
# ---------------------------------------------------
def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s type' % init_type)
    net.apply(init_func)


def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio=0.05, warmup_lr_ratio=0.1,
                     no_aug_iter_ratio=0.05, step_num=10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                    1.0 + np.cos(
                np.pi * (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n = iters // step_size
        out_lr = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr, lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate = (min_lr / lr) ** (1 / (step_num - 1))
        step_size = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func


def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
