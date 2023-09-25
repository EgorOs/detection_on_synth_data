from pathlib import Path

import flash
import pytorch_lightning as pl
from flash.image import ObjectDetectionData, ObjectDetector
from pytorch_lightning.callbacks import LearningRateMonitor

from src.callbacks.debug import VisualizeBatch
from src.datamodule import SignsDatamodule


def train():
    pl.seed_everything(0)

    datamodule = ObjectDetectionData.from_coco(
        train_folder='data/coco128/images/train2017/',
        train_ann_file='data/coco128/annotations/instances_train2017.json',
        val_split=0.1,
        transform_kwargs={'image_size': 512},
        batch_size=16,
    )

    datamodule = SignsDatamodule.from_synth_data(
        images_path=Path('/home/egor/Projects/detection_on_synth_data/dataset/25/images'),
        transform_kwargs={'image_size': 512},
        batch_size=16,
    )

    # model = ObjectDetector(head="efficientdet", backbone="d0", num_classes=datamodule.num_classes, image_size=512)
    model = ObjectDetector(
        head='retinanet', backbone='resnet18_fpn', pretrained=True, num_classes=datamodule.num_classes, image_size=512,
    )

    # FIXME need sampler
    callbacks = [
        VisualizeBatch(every_n_epochs=2),
        LearningRateMonitor(logging_interval='step'),
    ]
    trainer = flash.Trainer(accelerator='auto', max_epochs=10, callbacks=callbacks, log_every_n_steps=5)
    # trainer.finetune(model, datamodule=datamodule, strategy="freeze")
    trainer.fit(model, datamodule=datamodule)

    # datamodule = ObjectDetectionData.from_files(
    #     predict_files=[
    #         "/home/egor/Projects/detection_on_synth_data/sample_dataset/images/test/6.png",
    #         "/home/egor/Projects/detection_on_synth_data/sample_dataset/images/test/7.png",
    #         "/home/egor/Projects/detection_on_synth_data/sample_dataset/images/test/8.png",
    #     ],
    #     transform_kwargs={"image_size": 512},
    #     batch_size=4,
    # )
    # predictions = trainer.predict(model, datamodule=datamodule)
    # print(predictions)

    trainer.save_checkpoint('object_detection_model.pt')


if __name__ == '__main__':
    train()
