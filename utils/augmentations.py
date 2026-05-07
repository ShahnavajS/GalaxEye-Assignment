from __future__ import annotations

import albumentations as A


def build_train_transforms(
    crop_size: int | None = None,
    use_transpose: bool = True,
    affine_probability: float = 0.25,
    affine_scale: float = 0.05,
    affine_translate: float = 0.05,
    horizontal_flip_probability: float = 0.5,
    vertical_flip_probability: float = 0.5,
    rotate90_probability: float = 0.5,
) -> A.Compose:
    transforms: list[A.BasicTransform] = []

    if crop_size is not None:
        transforms.append(A.RandomCrop(height=crop_size, width=crop_size))

    if affine_probability > 0.0:
        transforms.append(
            A.Affine(
                scale=(1.0 - affine_scale, 1.0 + affine_scale),
                translate_percent=(-affine_translate, affine_translate),
                rotate=0.0,
                shear=0.0,
                interpolation=1,
                mask_interpolation=0,
                p=affine_probability,
            )
        )

    transforms.extend(
        [
            A.HorizontalFlip(p=horizontal_flip_probability),
            A.VerticalFlip(p=vertical_flip_probability),
            A.RandomRotate90(p=rotate90_probability),
        ]
    )

    if use_transpose:
        transforms.append(A.Transpose(p=0.25))

    return A.Compose(transforms)


def build_val_transforms(crop_size: int | None = None) -> A.Compose:
    transforms: list[A.BasicTransform] = []

    if crop_size is not None:
        transforms.append(A.CenterCrop(height=crop_size, width=crop_size))

    return A.Compose(transforms)
