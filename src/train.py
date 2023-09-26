import os
from pathlib import Path

import flash
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor

from src.callbacks.debug import PreviewRawImages, VisualizeBatch
from src.callbacks.experiment_tracking import ClearMLTracking
from src.config import ExperimentConfig
from src.constants import PROJECT_ROOT
from src.datamodule import DataSplit, SignsDatamodule
from src.lightning_module import SignDetectorModule


DATA_PATH = Path('/home/egor/Projects/detection_on_synth_data/dataset/35')


def train(cfg: ExperimentConfig):
    pl.seed_everything(0)

    datamodule = SignsDatamodule.from_synth_data(
        images_path=DATA_PATH / 'images',
        transform_kwargs={'data_config': cfg.data_config},
        batch_size=cfg.data_config.batch_size,
    )

    # model = SignDetectorModule(
    #     head="efficientdet", backbone="d0", pretrained=True, num_classes=datamodule.num_classes, image_size=cfg.data_config.img_size
    # )
    model = SignDetectorModule(
        head='retinanet',
        backbone='resnet18_fpn',
        pretrained=True,
        num_classes=datamodule.num_classes,
        image_size={'image_size': cfg.data_config.img_size},
    )

    callbacks = [
        VisualizeBatch(every_n_epochs=1),
        LearningRateMonitor(logging_interval='step'),
    ]
    callbacks += [
        PreviewRawImages(
            DATA_PATH, img_pattern='*.png', split=split,
        )
        for split in (DataSplit.TRAIN, DataSplit.VAL, DataSplit.TEST)
    ]

    if cfg.track_in_clearml:
        callbacks += [ClearMLTracking(cfg=cfg)]

    trainer = flash.Trainer(**dict(cfg.trainer_config), accelerator='auto', callbacks=callbacks)
    # trainer.finetune(model, datamodule=datamodule, strategy="freeze")
    trainer.fit(model, datamodule=datamodule)

    trainer.save_checkpoint('object_detection_model.pt')


if __name__ == '__main__':
    cfg_path = os.getenv('TRAIN_CFG_PATH', PROJECT_ROOT / 'configs' / 'train.yaml')
    train(cfg=ExperimentConfig.from_yaml(cfg_path))
