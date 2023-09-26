from typing import Tuple

import albumentations as albu
import numpy as np
from albumentations.pytorch import ToTensorV2
from torch import Tensor


def cv_image_to_tensor(img: np.ndarray, normalize: bool = True) -> Tensor:
    ops = [
        ToTensorV2(),
    ]
    if normalize:
        ops.insert(0, albu.Normalize())
    to_tensor = albu.Compose(ops)
    return to_tensor(image=img)['image']


def denormalize(
    img: np.ndarray,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    max_value: int = 255,
):
    denorm = albu.Normalize(
        mean=[-me / st for me, st in zip(mean, std)],
        std=[1.0 / st for st in std],
        always_apply=True,
        max_pixel_value=1.0,
    )
    return (denorm(image=img)['image'] * max_value).astype(np.uint8)


def tensor_to_cv_image(tensor: Tensor) -> np.ndarray:
    return tensor.permute(1, 2, 0).cpu().numpy()
