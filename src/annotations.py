from dataclasses import dataclass
from typing import NamedTuple, Dict, Any


class Bbox(NamedTuple):
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def icevision_bbox(self) -> 'IcevisionBbox':
        x0, y0, x1, y1 = self
        return {'xmin': x0, 'ymin': y0, 'width': (x1 - x0), 'height': (y1 - y0)}

    @classmethod
    def from_icevision_bbox(cls, bbox) -> 'Bbox':
        return cls(bbox['xmin'], bbox['ymin'], bbox['xmin'] + bbox['width'], bbox['ymin'] + bbox['height'])


def is_icevision_bbox(bbox: Any) -> bool:
    is_icevision = False
    if isinstance(bbox, dict):
        if bbox.get('xmin') is not None:
            # Roughly matched schema
            is_icevision = True
    return is_icevision



# @dataclass
# class IcevisionBbox:
#     xmin: float
#     ymin: float
#     width: float
#     height: float
#
#     @property
#     def xmax(self) -> float:
#         return self.xmin + self.width
#
#     @property
#     def ymax(self) -> float:
#         return self.ymin + self.height
#
#     @property
#     def as_bbox(self) -> 'Bbox':
#         return Bbox(x0=self.xmin, y0=self.ymin, x1=self.xmax, y1=self.ymax)
