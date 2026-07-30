# SSM-Net: SegMAN-Swin-U²Net

A nested U-shaped network combining **Swin Transformer** window self-attention with **SegMAN** local-global hybrid attention for semantic segmentation.

## Model Architecture

```
Input Image
    │
    ├── Stage1: SwinRSU7(3→64)  ─── hx1 ────────┐
    │       └── MaxPool(2)                        │
    ├── Stage2: SwinRSU6(64→128) ─── hx2 ────────┤
    │       └── MaxPool(2)                        │ Skip
    ├── Stage3: SwinRSU5(128→256) ─── hx3 ───────┤ Connections
    │       └── MaxPool(2)                        │
    ├── Stage4: SwinRSU4(256→512) ─── hx4 ───────┤
    │       └── MaxPool(2)                        │
    ├── Stage5: SwinRSU4F(512→512) ── hx5 ───────┤
    │       └── MaxPool(2)                        │
    ├── Stage6: SwinRSU4F(512→512) ── hx6 ───────┤
    │                                            │
    │    (Decoder: Upsample + Skip Concat)        │
    │                                            │
    ├── Stage5d: SwinRSU4F(1024→512) ← cat(hx6↑,hx5)
    ├── Stage4d: SwinRSU4(1024→256)  ← cat(hx5d↑,hx4)
    ├── Stage3d: SwinRSU5(512→128)   ← cat(hx4d↑,hx3)
    ├── Stage2d: SwinRSU6(256→64)    ← cat(hx3d↑,hx2)
    ├── Stage1d: SwinRSU7(128→64)    ← cat(hx2d↑,hx1)
    │
    └── Multi-scale Fusion → Output
```

**Key components:**
- **SwinRSU**: Residual U-Block with Swin Transformer bottleneck
- **SwinBottleneck**: W-MSA + SW-MSA window self-attention
- **SegMANLASSBlock**: Local neighborhood attention + directional scan context

Each SwinRSU bottleneck contains:
```
SegMANSwinBottleneck = SwinBottleneck (W-MSA → SW-MSA) + SegMANLASSBlock (CPE → LNA → Scan → FFN)
```

## Directory Structure

```
SSM-Net/
├── train_png.py              # Training script for PNG datasets
├── predict_and_eval_png.py   # Prediction and evaluation script
├── dataset_png.py            # Dataset loader for PNG format
├── requirements.txt
├── nets/
│   ├── __init__.py
│   ├── u2net.py              # Base U²-Net with RSU blocks
│   ├── swin_u2net.py         # SegMANSwinU2NET model definition
│   ├── model_factory.py      # Model registry
│   └── unet_training.py      # Loss functions and LR scheduling
└── utils/
    ├── utils.py
    ├── utils_metrics.py
    ├── callbacks.py
    └── utils_fit.py
```

## Usage

### Training

```bash
python train_png.py
```

Configure `train_png.py`:
- `dataset_path`: Path to dataset root
- `num_classes`: Number of segmentation classes
- `input_shape`: Input image size [height, width]
- `batch_size`, `Init_lr`, `UnFreeze_Epoch`, etc.

### Prediction

```bash
python predict_and_eval_png.py
```

### Dataset Format

PNG dataset structure:
```
dataset_root/
├── images/
│   ├── train/
│   │   ├── img1.png
│   │   └── ...
│   └── val/
│       └── ...
└── masks/
    ├── train/
    │   ├── img1.png
    │   └── ...
    └── val/
        └── ...
```

## Model Parameters

| Variant | Parameters |
|---------|-----------|
| SegMANSwinU2NET | ~54.6M |

## Citation

If you use this model, please cite:

```
@article{ssmnet2026,
  title={SSM-Net: SegMAN-Swin-U²Net for Landslide Detection},
  author={},
  journal={},
  year={2026}
}
```

## License

See LICENSE file.
