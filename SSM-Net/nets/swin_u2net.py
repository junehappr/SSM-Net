import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.u2net import REBNConv, U2NET, _upsample_like


def _choose_heads(channels):
    for heads in (8, 4, 2, 1):
        if channels % heads == 0 and channels // heads >= 8:
            return heads
    return 1


def _window_partition(x, window_size):
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows.view(-1, window_size, window_size, c)


def _window_reverse(windows, window_size, h, w, b):
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(b, h, w, -1)


class Mlp(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.0):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class LayerNorm2d(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size=7, num_heads=4, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        b_windows, n, c = x.shape
        qkv = self.qkv(x).reshape(b_windows, n, 3, self.num_heads, c // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            num_windows = mask.shape[0]
            attn = attn.view(b_windows // num_windows, num_windows, self.num_heads, n, n)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_windows, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock2D(nn.Module):
    def __init__(
        self,
        dim,
        window_size=7,
        shift_size=0,
        num_heads=None,
        mlp_ratio=4.0,
        dropout=0.0,
        attention_cls=WindowAttention,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = attention_cls(
            dim,
            window_size=window_size,
            num_heads=num_heads or _choose_heads(dim),
            attn_drop=dropout,
            proj_drop=dropout,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, hidden_dim=int(dim * mlp_ratio), dropout=dropout)

    def _attention_mask(self, hp, wp, device):
        if self.shift_size <= 0:
            return None

        img_mask = torch.zeros((1, hp, wp, 1), device=device)
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = _window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x):
        b, c, h, w = x.shape
        shortcut = x
        x = x.permute(0, 2, 3, 1).contiguous()

        pad_b = (self.window_size - h % self.window_size) % self.window_size
        pad_r = (self.window_size - w % self.window_size) % self.window_size
        if pad_b > 0 or pad_r > 0:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        hp, wp = x.shape[1], x.shape[2]

        shift_size = self.shift_size if min(hp, wp) > self.window_size else 0
        shifted_x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2)) if shift_size > 0 else x
        x_windows = _window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)

        attn_mask = self._attention_mask(hp, wp, x.device) if shift_size > 0 else None
        attn_windows = self.attn(self.norm1(x_windows), mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = _window_reverse(attn_windows, self.window_size, hp, wp, b)

        if shift_size > 0:
            x = torch.roll(shifted_x, shifts=(shift_size, shift_size), dims=(1, 2))
        else:
            x = shifted_x
        if pad_b > 0 or pad_r > 0:
            x = x[:, :h, :w, :].contiguous()

        x = x.permute(0, 3, 1, 2).contiguous()
        x = shortcut + x

        y = x.permute(0, 2, 3, 1).contiguous()
        y = y + self.mlp(self.norm2(y))
        return y.permute(0, 3, 1, 2).contiguous()


class SwinBottleneck(nn.Module):
    def __init__(self, channels, window_size=7, dropout=0.0, attention_cls=WindowAttention):
        super().__init__()
        shift_size = window_size // 2
        self.blocks = nn.Sequential(
            SwinTransformerBlock2D(
                channels,
                window_size=window_size,
                shift_size=0,
                dropout=dropout,
                attention_cls=attention_cls,
            ),
            SwinTransformerBlock2D(
                channels,
                window_size=window_size,
                shift_size=shift_size,
                dropout=dropout,
                attention_cls=attention_cls,
            ),
        )

    def forward(self, x):
        return self.blocks(x)


class LocalNeighborhoodAttention2D(nn.Module):
    def __init__(self, dim, kernel_size=7, num_heads=None, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.num_heads = num_heads or _choose_heads(dim)
        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.lepe = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        lepe = self.lepe(v)

        q = q.view(b, self.num_heads, self.head_dim, h * w).transpose(-1, -2)
        k = F.unfold(k, kernel_size=self.kernel_size, padding=self.kernel_size // 2)
        v = F.unfold(v, kernel_size=self.kernel_size, padding=self.kernel_size // 2)
        k = k.view(b, self.num_heads, self.head_dim, self.kernel_size * self.kernel_size, h * w)
        v = v.view(b, self.num_heads, self.head_dim, self.kernel_size * self.kernel_size, h * w)
        k = k.permute(0, 1, 4, 3, 2).contiguous()
        v = v.permute(0, 1, 4, 3, 2).contiguous()

        attn = (q.unsqueeze(-2) * k).sum(dim=-1) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = (attn.unsqueeze(-1) * v).sum(dim=-2)
        out = out.transpose(-1, -2).contiguous().view(b, c, h, w)
        out = out + lepe
        out = self.proj(out)
        return self.proj_drop(out)


class DirectionalScanContext(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(dim * 4, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
        )
        self.out = nn.Conv2d(dim, dim, kernel_size=1)

    @staticmethod
    def _scan_mean(x, dim):
        length = x.size(dim)
        denom = torch.arange(1, length + 1, device=x.device, dtype=x.dtype)
        shape = [1] * x.ndim
        shape[dim] = length
        denom = denom.view(*shape)
        return torch.cumsum(x, dim=dim) / denom

    def forward(self, x):
        left_right = self._scan_mean(x, dim=3)
        right_left = torch.flip(self._scan_mean(torch.flip(x, dims=[3]), dim=3), dims=[3])
        top_bottom = self._scan_mean(x, dim=2)
        bottom_top = torch.flip(self._scan_mean(torch.flip(x, dims=[2]), dim=2), dims=[2])
        context = torch.cat((left_right, right_left, top_bottom, bottom_top), dim=1)
        return self.out(self.fuse(context))


class SegMANLASSBlock(nn.Module):
    def __init__(self, dim, kernel_size=7, num_heads=None, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.cpe1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm1 = LayerNorm2d(dim)
        self.local_attn = LocalNeighborhoodAttention2D(
            dim,
            kernel_size=kernel_size,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.norm2 = LayerNorm2d(dim)
        self.scan_context = DirectionalScanContext(dim)
        self.local_global_fuse = nn.Conv2d(dim * 2, dim, kernel_size=1)
        self.cpe2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm3 = LayerNorm2d(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, dim, kernel_size=1),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.cpe1(x)
        local = self.local_attn(self.norm1(x))
        global_context = self.scan_context(self.norm2(local))
        x = x + self.local_global_fuse(torch.cat((local, local + global_context), dim=1))
        x = x + self.cpe2(x)
        x = x + self.ffn(self.norm3(x))
        return x


class SegMANSwinBottleneck(SwinBottleneck):
    def __init__(self, channels, window_size=7, dropout=0.0):
        super().__init__(channels, window_size=window_size, dropout=dropout)
        self.lass = SegMANLASSBlock(channels, kernel_size=window_size, dropout=dropout)

    def forward(self, x):
        x = super().forward(x)
        return self.lass(x)


class SwinRSU7(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, window_size=7):
        super().__init__()
        self.rebnconvin = REBNConv(in_channels, out_channels, dilation=1)
        self.rebnconv1 = REBNConv(out_channels, mid_channels, dilation=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNConv(mid_channels, mid_channels, dilation=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNConv(mid_channels, mid_channels, dilation=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNConv(mid_channels, mid_channels, dilation=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv5 = REBNConv(mid_channels, mid_channels, dilation=1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv6 = REBNConv(mid_channels, mid_channels, dilation=1)
        self.rebnconv7 = REBNConv(mid_channels, mid_channels, dilation=2)
        self.swin = SwinBottleneck(mid_channels, window_size=window_size)
        self.rebnconv6d = REBNConv(mid_channels * 2, mid_channels, dilation=1)
        self.rebnconv5d = REBNConv(mid_channels * 2, mid_channels, dilation=1)
        self.rebnconv4d = REBNConv(mid_channels * 2, mid_channels, dilation=1)
        self.rebnconv3d = REBNConv(mid_channels * 2, mid_channels, dilation=1)
        self.rebnconv2d = REBNConv(mid_channels * 2, mid_channels, dilation=1)
        self.rebnconv1d = REBNConv(mid_channels * 2, out_channels, dilation=1)

    def forward(self, x):
        hxin = self.rebnconvin(x)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)
        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)
        hx6 = self.rebnconv6(hx)
        hx7 = self.swin(self.rebnconv7(hx6))
        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), dim=1))
        hx6dup = _upsample_like(hx6d, hx5)
        hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), dim=1))
        hx5dup = _upsample_like(hx5d, hx4)
        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), dim=1))
        hx4dup = _upsample_like(hx4d, hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), dim=1))
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), dim=1))
        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), dim=1))
        return hx1d + hxin


class SwinRSU6(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, window_size=7):
        super().__init__()
        self.rebnconvin = REBNConv(in_channels, out_channels, dilation=1)
        self.rebnconv1 = REBNConv(out_channels, mid_channels, dilation=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNConv(mid_channels, mid_channels, dilation=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNConv(mid_channels, mid_channels, dilation=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNConv(mid_channels, mid_channels, dilation=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv5 = REBNConv(mid_channels, mid_channels, dilation=1)
        self.rebnconv6 = REBNConv(mid_channels, mid_channels, dilation=2)
        self.swin = SwinBottleneck(mid_channels, window_size=window_size)
        self.rebnconv5d = REBNConv(mid_channels * 2, mid_channels, dilation=1)
        self.rebnconv4d = REBNConv(mid_channels * 2, mid_channels, dilation=1)
        self.rebnconv3d = REBNConv(mid_channels * 2, mid_channels, dilation=1)
        self.rebnconv2d = REBNConv(mid_channels * 2, mid_channels, dilation=1)
        self.rebnconv1d = REBNConv(mid_channels * 2, out_channels, dilation=1)

    def forward(self, x):
        hxin = self.rebnconvin(x)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)
        hx5 = self.rebnconv5(hx)
        hx6 = self.swin(self.rebnconv6(hx5))
        hx5d = self.rebnconv5d(torch.cat((hx6, hx5), dim=1))
        hx5dup = _upsample_like(hx5d, hx4)
        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), dim=1))
        hx4dup = _upsample_like(hx4d, hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), dim=1))
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), dim=1))
        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), dim=1))
        return hx1d + hxin


class SwinRSU5(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, window_size=7):
        super().__init__()
        self.rebnconvin = REBNConv(in_channels, out_channels, dilation=1)
        self.rebnconv1 = REBNConv(out_channels, mid_channels, dilation=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNConv(mid_channels, mid_channels, dilation=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNConv(mid_channels, mid_channels, dilation=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNConv(mid_channels, mid_channels, dilation=1)
        self.rebnconv5 = REBNConv(mid_channels, mid_channels, dilation=2)
        self.swin = SwinBottleneck(mid_channels, window_size=window_size)
        self.rebnconv4d = REBNConv(mid_channels * 2, mid_channels, dilation=1)
        self.rebnconv3d = REBNConv(mid_channels * 2, mid_channels, dilation=1)
        self.rebnconv2d = REBNConv(mid_channels * 2, mid_channels, dilation=1)
        self.rebnconv1d = REBNConv(mid_channels * 2, out_channels, dilation=1)

    def forward(self, x):
        hxin = self.rebnconvin(x)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)
        hx5 = self.swin(self.rebnconv5(hx4))
        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), dim=1))
        hx4dup = _upsample_like(hx4d, hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), dim=1))
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), dim=1))
        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), dim=1))
        return hx1d + hxin


class SwinRSU4(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, window_size=7):
        super().__init__()
        self.rebnconvin = REBNConv(in_channels, out_channels, dilation=1)
        self.rebnconv1 = REBNConv(out_channels, mid_channels, dilation=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNConv(mid_channels, mid_channels, dilation=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNConv(mid_channels, mid_channels, dilation=1)
        self.rebnconv4 = REBNConv(mid_channels, mid_channels, dilation=2)
        self.swin = SwinBottleneck(mid_channels, window_size=window_size)
        self.rebnconv3d = REBNConv(mid_channels * 2, mid_channels, dilation=1)
        self.rebnconv2d = REBNConv(mid_channels * 2, mid_channels, dilation=1)
        self.rebnconv1d = REBNConv(mid_channels * 2, out_channels, dilation=1)

    def forward(self, x):
        hxin = self.rebnconvin(x)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx4 = self.swin(self.rebnconv4(hx3))
        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), dim=1))
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), dim=1))
        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), dim=1))
        return hx1d + hxin


class SwinRSU4F(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, window_size=7):
        super().__init__()
        self.rebnconvin = REBNConv(in_channels, out_channels, dilation=1)
        self.rebnconv1 = REBNConv(out_channels, mid_channels, dilation=1)
        self.rebnconv2 = REBNConv(mid_channels, mid_channels, dilation=2)
        self.rebnconv3 = REBNConv(mid_channels, mid_channels, dilation=4)
        self.rebnconv4 = REBNConv(mid_channels, mid_channels, dilation=8)
        self.swin = SwinBottleneck(mid_channels, window_size=window_size)
        self.rebnconv3d = REBNConv(mid_channels * 2, mid_channels, dilation=4)
        self.rebnconv2d = REBNConv(mid_channels * 2, mid_channels, dilation=2)
        self.rebnconv1d = REBNConv(mid_channels * 2, out_channels, dilation=1)

    def forward(self, x):
        hxin = self.rebnconvin(x)
        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)
        hx4 = self.swin(self.rebnconv4(hx3))
        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), dim=1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), dim=1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), dim=1))
        return hx1d + hxin


def _replace_with_swin_rsu(model, window_size=7):
    model.stage1 = SwinRSU7(model.stage1.rebnconvin.block[0].in_channels, 32, 64, window_size)
    model.stage2 = SwinRSU6(64, 32, 128, window_size)
    model.stage3 = SwinRSU5(128, 64, 256, window_size)
    model.stage4 = SwinRSU4(256, 128, 512, window_size)
    model.stage5 = SwinRSU4F(512, 256, 512, window_size)
    model.stage6 = SwinRSU4F(512, 256, 512, window_size)
    model.stage5d = SwinRSU4F(1024, 256, 512, window_size)
    model.stage4d = SwinRSU4(1024, 128, 256, window_size)
    model.stage3d = SwinRSU5(512, 64, 128, window_size)
    model.stage2d = SwinRSU6(256, 32, 64, window_size)
    model.stage1d = SwinRSU7(128, 16, 64, window_size)



def _replace_with_segman_swin_bottleneck(model, window_size=7):
    model.stage1.swin = SegMANSwinBottleneck(32, window_size=window_size)
    model.stage2.swin = SegMANSwinBottleneck(32, window_size=window_size)
    model.stage3.swin = SegMANSwinBottleneck(64, window_size=window_size)
    model.stage4.swin = SegMANSwinBottleneck(128, window_size=window_size)
    model.stage5.swin = SegMANSwinBottleneck(256, window_size=window_size)
    model.stage6.swin = SegMANSwinBottleneck(256, window_size=window_size)
    model.stage5d.swin = SegMANSwinBottleneck(256, window_size=window_size)
    model.stage4d.swin = SegMANSwinBottleneck(128, window_size=window_size)
    model.stage3d.swin = SegMANSwinBottleneck(64, window_size=window_size)
    model.stage2d.swin = SegMANSwinBottleneck(32, window_size=window_size)
    model.stage1d.swin = SegMANSwinBottleneck(16, window_size=window_size)


class SwinU2NET(U2NET):
    def __init__(self, in_channels=3, num_classes=1, window_size=7):
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        _replace_with_swin_rsu(self, window_size=window_size)


# Only SegMANSwinU2NET models exposed for standalone project.

class SegMANSwinU2NET(SwinU2NET):
    def __init__(self, in_channels=3, num_classes=1, window_size=7):
        super().__init__(in_channels=in_channels, num_classes=num_classes, window_size=window_size)
        _replace_with_segman_swin_bottleneck(self, window_size=window_size)


class SegMANSwinU2NET_Single(SegMANSwinU2NET):
    def forward(self, x):
        outputs = super().forward(x)
        return outputs[0]
