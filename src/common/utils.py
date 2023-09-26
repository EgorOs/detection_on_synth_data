import contextlib
import random
import ssl
from copy import deepcopy
from itertools import product
from typing import List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, random_split


@contextlib.contextmanager
def set_unverified_ssl_context():
    orig_context = deepcopy(ssl._create_default_https_context)
    ssl._create_default_https_context = ssl._create_unverified_context
    yield
    ssl._create_default_https_context = orig_context


def grid_from_images(  # noqa: WPS210
    images: List[np.ndarray],
    cols: int = 3,
) -> np.ndarray:
    if len({img.shape for img in images}) != 1:
        raise ValueError('Images should have identical shapes.')

    height, width, channels = images[0].shape
    rows = len(images) // cols + 1
    grid = np.zeros((rows * height, cols * width, channels))
    for img, pos in zip(images, list(product(range(rows), range(cols)))):  # noqa: WPS221
        row, col = pos
        grid[row * height : (row + 1) * height, col * width : (col + 1) * width] = img  # noqa: WPS221
    return grid


def get_torch_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_data_split_indexes(dataset: Dataset, proportion: Sequence[float]) -> List[List[int]]:
    splits = random_split(dataset, proportion)
    return [list(sp.indices) for sp in splits]
