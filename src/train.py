from pathlib import Path

import flash
from flash.image import ObjectDetector
from pytorch_lightning.callbacks import LearningRateMonitor

from src.datamodule import SignsDatamodule


def train():
    flash.seed_everything(0)

    datamodule = SignsDatamodule.from_synth_data(
        images_path=Path('/home/egor/Projects/detection_on_synth_data/dataset/25/images'),
        transform_kwargs={"image_size": 512},
        batch_size=4,
    )

    # model = ObjectDetector(head="efficientdet", backbone="d0", num_classes=datamodule.num_classes, image_size=512)
    model = ObjectDetector(head="retinanet", backbone="resnet18_fpn", pretrained=True, num_classes=datamodule.num_classes, image_size=512)

    callbacks = [
        LearningRateMonitor(logging_interval='step'),
    ]
    trainer = flash.Trainer(max_epochs=10, callbacks=callbacks)
    trainer.finetune(model, datamodule=datamodule, strategy="freeze")

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

    trainer.save_checkpoint("object_detection_model.pt")


if __name__ == '__main__':
    train()
