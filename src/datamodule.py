import json
from pathlib import Path
from typing import List, Optional, Any, Dict, Type, NamedTuple

from flash import DataKeys, Input, RunningStage
from flash.core.data.io.classification_input import ClassificationInputMixin
from flash.core.data.utilities.classification import TargetFormatter
from flash.core.data.utilities.loading import IMG_EXTENSIONS
from flash.core.data.utilities.paths import PATH_TYPE, filter_valid_files
from flash.core.finetuning import LightningEnum
from flash.core.integrations.icevision.transforms import IceVisionInputTransform
from flash.core.utilities.types import INPUT_TRANSFORM_TYPE
from flash.image import ObjectDetectionData
from flash.image.data import ImageFilesInput


class DataSplit(LightningEnum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'


class Bbox(NamedTuple):
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def icevision_bbox(self) -> Dict[str, float]:
        x0, y0, x1, y1 = self
        return {'xmin': x0, 'ymin': y0, 'width': (x1 - x0), 'height': (y1 - y0)}


class SynthFilesInput(ClassificationInputMixin, ImageFilesInput):
    def load_data(
        self,
        files: List[PATH_TYPE],
        targets: Optional[List[List[Any]]] = None,
        bboxes: Optional[List[List[Dict[str, int]]]] = None,
        target_formatter: Optional[TargetFormatter] = None,
    ) -> List[Dict[str, Any]]:
        if targets is None:
            return super().load_data(files)
        files, targets, bboxes = filter_valid_files(
            files, targets, bboxes, valid_extensions=IMG_EXTENSIONS
        )
        self.load_target_metadata(
            [t for target in targets for t in target], add_background=True, target_formatter=target_formatter
        )

        return [
            {DataKeys.INPUT: file, DataKeys.TARGET: {"bboxes": bbox, "labels": label}}
            for file, label, bbox in zip(files, targets, bboxes)
        ]

    def load_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample = super().load_sample(sample)
        if DataKeys.TARGET in sample:
            sample[DataKeys.TARGET]["labels"] = [
                self.format_target(label) for label in sample[DataKeys.TARGET]["labels"]
            ]
        return sample


class SignsDatamodule(ObjectDetectionData):
    @classmethod
    def from_synth_data(
        cls,
        images_path: Path,
        target_formatter: Optional[TargetFormatter] = None,
        input_cls: Type[Input] = SynthFilesInput,
        transform: INPUT_TRANSFORM_TYPE = IceVisionInputTransform,  # FIXME check these transforms
        transform_kwargs: Optional[Dict] = None,
        annotations_file='annotations.json',
        **data_module_kwargs: Any,
    ) -> 'SignsDatamodule':

        ds_kw = {
            "target_formatter": target_formatter,
        }

        with open(images_path.parent / annotations_file, 'rb') as ann_file:
            annotations = json.load(ann_file)

        train_files, train_targets, train_bboxes = _load_split(images_path, DataSplit.TRAIN, annotations)
        val_files, val_targets, val_bboxes = _load_split(images_path, DataSplit.VAL, annotations)
        test_files, test_targets, test_bboxes = _load_split(images_path, DataSplit.TEST, annotations)

        train_input = input_cls(
            RunningStage.TRAINING,
            train_files,
            train_targets,
            train_bboxes,
            **ds_kw,
        )
        ds_kw["target_formatter"] = getattr(train_input, "target_formatter", None)

        return cls(
            train_input,
            input_cls(
                RunningStage.VALIDATING,
                val_files,
                val_targets,
                val_bboxes,
                **ds_kw,
            ),
            input_cls(
                RunningStage.TESTING,
                test_files,
                test_targets,
                test_bboxes,
                **ds_kw,
            ),
            transform=transform,
            transform_kwargs=transform_kwargs,
            **data_module_kwargs,
        )


def _load_split(images_path: Path, split: DataSplit, annotations: Dict[str, Any]):
    split = split.value
    files = list((images_path / split).glob('*.png'))

    bboxes = []
    cls_indexes = []
    for ann in annotations[split]:
        bboxes.append(
            [Bbox(**ann['bbox']).icevision_bbox]
        )

        cls_indexes.append(
            [ann['cls_idx']]
        )

    return files, cls_indexes, bboxes
