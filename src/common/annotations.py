import json
import os
from enum import Enum
from pathlib import Path
from typing import List, NamedTuple, Tuple, Union

import cv2
import numpy as np
from pydantic import BaseModel

STOP_SIGN_ID = 0
STOP_SIGN_CLS_NAME = 'stop_sign'


class Color(Tuple, Enum):
    black = (0, 0, 0)
    white = (255, 255, 255)
    green = (16, 172, 132)
    blue = (10, 189, 227)
    red = (238, 82, 83)
    violet = (95, 39, 205)


class AbsBboxXYXY(NamedTuple):
    x0: int
    y0: int
    x1: int
    y1: int


class AbsBboxXYWH(NamedTuple):
    x0: int
    y0: int
    w: int  # noqa: WPS111
    h: int  # noqa: WPS111


class SignAnnotation(BaseModel):
    code: int
    bbox: AbsBboxXYXY


class COCOImage(BaseModel):
    id: int
    width: int
    height: int
    file_name: str


class COCOAnnotation(BaseModel):
    id: int
    image_id: int
    category_id: int
    area: float
    bbox: AbsBboxXYWH
    iscrowd: int


class COCOCategory(BaseModel):
    id: int
    name: str


class COCODatasetMetadata(BaseModel):
    images: List[COCOImage]
    annotations: List[COCOAnnotation]
    categories: List[COCOCategory]

    @classmethod
    def from_samples(cls, samples: List[Tuple[Path, dict]]) -> 'COCODatasetMetadata':  # noqa: WPS210
        category = COCOCategory(id=STOP_SIGN_ID, name=STOP_SIGN_CLS_NAME)
        fields = {'images': [], 'annotations': [], 'categories': [category]}
        for fpath, ann in samples:
            idx = fpath.stem

            img_arr = cv2.imread(str(fpath))
            img_arr: np.ndarray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
            img_h, img_w = img_arr.shape[:2]

            fields['images'].append(
                COCOImage(id=idx, width=img_w, height=img_h, file_name=os.path.join(*fpath.parts[-3:])),  # noqa: WPS221
            )

            x0 = ann['bbox']['x0']
            y0 = ann['bbox']['y0']
            x1 = ann['bbox']['x1']
            y1 = ann['bbox']['y1']
            fields['annotations'].append(
                COCOAnnotation(
                    id=idx,
                    image_id=idx,
                    category_id=category.id,
                    area=img_h * img_w,
                    bbox=AbsBboxXYWH(x0, y0, x1 - x0, y1 - y0),
                    iscrowd=0,
                ),  # noqa: WPS221
            )
        return cls(**fields)

    def to_json(self, path: Union[str, Path], indent: int = 0):
        with open(path, 'w') as out_file:
            out_file.write(json.dumps(self.dict(), indent=indent))
