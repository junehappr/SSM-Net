from nets.swin_u2net import SegMANSwinU2NET_Single


def get_model(model_name, num_classes, in_channels, pretrained=False, img_size=128):
    """Get a segmentation model by name.

    Currently supports only SegMANSwinU2NET.
    """
    if model_name == "segman_swin_u2net":
        return SegMANSwinU2NET_Single(
            in_channels=in_channels,
            num_classes=num_classes,
        )

    raise ValueError(f"Unknown model name: {model_name}")
