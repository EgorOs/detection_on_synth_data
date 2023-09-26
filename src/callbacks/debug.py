import json
import random
from pathlib import Path
from typing import Tuple, List

import albumentations as albu
import cv2
import imageio
import torch
from albumentations.pytorch import ToTensorV2
from flash import DataKeys
from pytorch_lightning import Callback, LightningModule, Trainer
from torch import Tensor
from torchvision.utils import draw_bounding_boxes, make_grid

from src.datamodule import DataSplit, get_samples
from src.annotations import Bbox


class PreviewRawImages(Callback):
    def __init__(
        self,
        dataset_path: Path,
        img_pattern: str,
        annotations_file: str = 'annotations.json',
        image_folder: str = 'images',
        split: DataSplit = DataSplit.TRAIN,
        n_images: int = 5,
        preview_size: Tuple[int, int] = (512, 512),
    ):
        super().__init__()
        self.img_pattern = img_pattern
        self.split = split.value
        self.images_path = dataset_path / image_folder / self.split
        self.n_images = n_images

        with open(dataset_path / annotations_file, 'rb') as ann_file:
            self.annotations = json.load(ann_file)[self.split]

        self.transform = albu.Compose([albu.Resize(*preview_size), ToTensorV2()])

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule):  # noqa: WPS210
        visualizations = []

        samples = get_samples(self.images_path, self.img_pattern, self.annotations)
        random.shuffle(samples)

        for img_path, ann in samples[: self.n_images]:
            img = imageio.imread(img_path)
            x0, y0, x1, y1 = Bbox(**ann['bbox'])

            img = cv2.rectangle(img, (x0, y0), (x1, y1), (255, 0, 0), 3)
            img_tensor = self.transform(image=img)['image']
            visualizations.append(img_tensor)

        grid = make_grid(visualizations, normalize=False)
        trainer.logger.experiment.add_image(
            f'Raw images / {self.split}',
            img_tensor=grid,
            global_step=trainer.global_step,
        )


class VisualizeBatch(Callback):
    def __init__(self, every_n_epochs: int):
        super().__init__()
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule):  # noqa: WPS210
        if trainer.current_epoch % self.every_n_epochs == 0:
            # batch = next(iter(trainer.train_dataloader))['input'][0]
            # images, targets = batch

            batch = next(iter(trainer.train_dataloader))
            visualizations = prepare_batch_visualizations(batch)

            grid = make_grid(visualizations, normalize=False)
            trainer.logger.experiment.add_image(
                'Batch preview / Train',
                img_tensor=grid,
                global_step=trainer.global_step,
            )

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule):  # noqa: WPS210
        if trainer.current_epoch % self.every_n_epochs == 0:
            batch = next(iter(trainer.val_dataloaders[0]))
            visualizations = prepare_batch_visualizations(batch)

            grid = make_grid(visualizations, normalize=False)
            trainer.logger.experiment.add_image(
                'Batch preview / Val',
                img_tensor=grid,
                global_step=trainer.global_step,
            )


# class VisualizeDetectionsCallback(pl.Callback):
#     def __init__(self, every_n_epochs=1, top_k=3):
#         super().__init__()
#         self.every_n_epochs = every_n_epochs
#         self.top_k = top_k
#
#     def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: DetectionLightningModule):
#         if trainer.current_epoch % self.every_n_epochs == 0:
#             batch: tuple = next(iter(trainer.val_dataloaders))
#             with torch.no_grad():
#                 with override_training_status(pl_module.model.model, training=False):
#                     preds = pl_module(batch)
#
#             visualizations: List[np.ndarray] = []
#             for image, pred in zip(batch[0], preds):
#                 image = denormalize(tensor_to_cv_image(image))
#                 visualizations.append(draw_top_k_predictions(image, pred, self.top_k))
#
#             trainer.logger.experiment.add_image(
#                 'Val set predictions',
#                 cv_image_to_tensor(grid_from_images(visualizations), normalize=False).to(torch.uint8),
#                 global_step=trainer.global_step,
#             )


def prepare_batch_visualizations(batch) -> List[Tensor]:
    metadata = batch[DataKeys.METADATA]
    images, targets = batch[DataKeys.INPUT][0]
    preproc_images = [rec['preproc_image'] for rec in metadata]
    visualizations = []
    for img, tgt in zip(preproc_images, targets):
        processed_img = Tensor(img).permute(2, 0, 1).to(torch.uint8)
        visualizations.append(draw_bounding_boxes(processed_img, boxes=tgt['boxes']))
    return visualizations
