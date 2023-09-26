from copy import deepcopy
from dataclasses import dataclass
from typing import Callable
import numpy as np
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from flash import InputTransform, RunningStage, DataKeys
from torch import Tensor

from src.config import DataConfig
from src.annotations import Bbox


@dataclass
class SynthDataTransform(InputTransform):
    def __init__(self, data_config: DataConfig = DataConfig()):
        super().__init__()
        self.data_config = data_config
        self._train_transforms = get_det_train_transforms(*self.data_config.img_size)
        self._valid_transforms = get_det_valid_transforms(*self.data_config.img_size)

    def _per_sample_transform(self, sample: dict, stage: RunningStage) -> dict:
        sample = deepcopy(sample)

        transform_fn = self._train_transforms if stage is RunningStage.TRAINING else self._valid_transforms
        img = np.array(sample['input'])
        target = sample['target']
        bboxes = target['bboxes']

        try:
            # In most cases this transformation is required, otherwise error will be raised.
            bboxes = [Bbox.from_icevision_bbox(bbox) for bbox in bboxes]
        except TypeError:
            pass

        transformed = transform_fn(image=img, labels=target['labels'], bboxes=bboxes)

        sample[DataKeys.TARGET]['bboxes'] = transformed['bboxes']
        sample[DataKeys.TARGET]['labels'] = transformed['labels']
        sample[DataKeys.METADATA]['preproc_image'] = transformed['image']

        sample[DataKeys.INPUT] = cv_image_to_tensor(transformed['image'])
        return sample

    def collate(self) -> Callable:
        return self._identity


def get_det_train_transforms(img_width: int, img_height: int) -> albu.Compose:
    return albu.Compose(
        [
            albu.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            albu.Rotate(limit=30),
            albu.GaussianBlur(),
            albu.RandomResizedCrop(height=img_height, width=img_width, always_apply=True),
        ],
        bbox_params=albu.BboxParams(
            format='pascal_voc',
            min_visibility=0,
            min_area=0,
            check_each_transform=True,
            label_fields=['labels'],
        ),
    )


def get_det_valid_transforms(img_width: int, img_height: int) -> albu.Compose:
    return albu.Compose(
        [albu.Resize(height=img_height, width=img_width)],
        bbox_params=albu.BboxParams(
            format='pascal_voc',
            min_visibility=0,
            min_area=0,
            label_fields=['labels'],
        ),
    )


def cv_image_to_tensor(img: np.ndarray, normalize: bool = True) -> Tensor:
    ops = [
        ToTensorV2(),
    ]
    if normalize:
        ops.insert(0, albu.Normalize())
    to_tensor = albu.Compose(ops)
    return to_tensor(image=img)['image']
