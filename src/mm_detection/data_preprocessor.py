import json
from pathlib import Path
from typing import Any, Dict, List

from src.common.annotations import COCODatasetMetadata


def get_samples(images_path: Path, img_pattern: str, annotations: List[Dict[str, Any]]):
    return list(
        zip(
            sorted(images_path.rglob(img_pattern), key=lambda pth: int(pth.stem)),
            annotations,
        ),
    )


def prepare_coco_dataset(  # noqa: WPS432, WPS210
    data_path: Path,
    annotations_path: str,
):
    img_folder = data_path / 'images'
    with open(data_path / annotations_path, 'rb') as ann_file:
        annotations = json.load(ann_file)

    for split in ('train', 'val', 'test'):
        samples = get_samples(img_folder / split, img_pattern='*.png', annotations=annotations[split])
        coco_annotations = COCODatasetMetadata.from_samples(samples)
        coco_annotations.to_json(data_path / f'{split}.json')
