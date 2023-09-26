from typing import Any, Dict

from src.mm_detection.constants import DETECTION_CLASSES

BATCH_SIZE = 12
NUM_WORKERS = 2
MIN_ANNOTATION_COUNT = 1


def build_dataloader_cfg(
    pipeline: Dict[str, Any],
    data_root: str,
    ann_file: str,
    data_prefix: str = '',
    pin_memory: bool = False,
) -> Dict[str, Any]:
    return dict(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        sampler=dict(type='DefaultSampler', shuffle=True),
        batch_sampler=None,
        dataset=dict(
            type='CocoDataset',
            data_root=data_root,
            ann_file=ann_file,
            data_prefix=dict(img=data_prefix),
            filter_cfg=dict(filter_empty_gt=True, min_size=MIN_ANNOTATION_COUNT),
            pipeline=pipeline,
            backend_args=None,
            metainfo=dict(classes=DETECTION_CLASSES),
        ),
        pin_memory=pin_memory,
    )
