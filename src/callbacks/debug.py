import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from torchvision.utils import draw_bounding_boxes, make_grid


class VisualizeBatch(Callback):
    def __init__(self, every_n_epochs: int):
        super().__init__()
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule):  # noqa: WPS210
        if trainer.current_epoch % self.every_n_epochs == 0:
            batch = next(iter(trainer.train_dataloader))['input'][0]
            images, targets = batch

            visualizations = []
            for img, tgt in zip(images, targets):
                processed_img = img.mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8)

                visualizations.append(draw_bounding_boxes(processed_img, boxes=tgt['boxes']))
            grid = make_grid(visualizations, normalize=False)
            trainer.logger.experiment.add_image(
                'Batch preview',
                img_tensor=grid,
                global_step=trainer.global_step,
            )
