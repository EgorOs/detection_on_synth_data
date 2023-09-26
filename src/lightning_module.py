from typing import Any

import torch.optim
from flash import DataKeys
from flash.core.integrations.icevision.transforms import to_icevision_record
from flash.image import ObjectDetector

from src.annotations import Bbox, is_icevision_bbox
from src.schedulers import get_cosine_schedule_with_warmup


def _patched_collate(collate_fn, samples):
    metadata = [sample.get(DataKeys.METADATA, None) for sample in samples]

    for sample in samples:
        bboxes = sample[DataKeys.TARGET]['bboxes']

        if bboxes:
            if not is_icevision_bbox(bboxes[0]):
                bboxes = [Bbox(*bbox).icevision_bbox for bbox in sample[DataKeys.TARGET]['bboxes']]
                sample[DataKeys.TARGET]['bboxes'] = bboxes

    new_samples = {
        DataKeys.INPUT: collate_fn([to_icevision_record(sample) for sample in samples]),
        DataKeys.METADATA: metadata,
    }
    return new_samples


class SignDetectorModule(ObjectDetector):
    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.adapter._wrap_collate_fn = _patched_collate

    def configure_optimizers(self) -> dict:
        # TODO: parametrize optimizer and lr scheduler.
        optimizer = torch.optim.SGD(self.parameters(), lr=2e-4)  # noqa: WPS432 will be parametrized
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=50,  # noqa: WPS432 will be parametrized
            num_training_steps=self.trainer.estimated_stepping_batches,
            num_cycles=0.4,  # noqa: WPS432 will be parametrized
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            },
        }
