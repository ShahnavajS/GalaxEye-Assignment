from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

MODEL_REGISTRY = {
    "unet": smp.Unet,
    "unetplusplus": smp.UnetPlusPlus,
    "deeplabv3plus": smp.DeepLabV3Plus,
}


class ChangeSegmentationModel(nn.Module):
    def __init__(
        self,
        architecture: str = "unet",
        encoder_name: str = "resnet18",
        encoder_weights: str | None = None,
        in_channels: int = 4,
        classes: int = 1,
    ) -> None:
        super().__init__()
        architecture = architecture.lower()
        if architecture not in MODEL_REGISTRY:
            supported = ", ".join(sorted(MODEL_REGISTRY))
            raise ValueError(f"Unsupported architecture '{architecture}'. Supported: {supported}")

        model_class = MODEL_REGISTRY[architecture]
        self.architecture = architecture
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.in_channels = in_channels
        self.classes = classes
        self.model = model_class(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image)


def build_segmentation_model(
    architecture: str = "unet",
    encoder_name: str = "resnet18",
    encoder_weights: str | None = None,
    in_channels: int = 4,
    classes: int = 1,
) -> ChangeSegmentationModel:
    return ChangeSegmentationModel(
        architecture=architecture,
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
    )


def build_unet_model(
    encoder_name: str = "resnet18",
    encoder_weights: str | None = None,
    in_channels: int = 4,
    classes: int = 1,
) -> ChangeSegmentationModel:
    return build_segmentation_model(
        architecture="unet",
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
    )
