import flash
from flash.core.data.utils import download_data
from flash.image import ObjectDetectionData, ObjectDetector


def train():

    # 1. Create the DataModule
    # Dataset Credit: https://www.kaggle.com/ultralytics/coco128

    download_data("https://github.com/zhiqwang/yolov5-rt-stack/releases/download/v0.3.0/coco128.zip", "data/")

    datamodule = ObjectDetectionData.from_coco(
        train_folder="data/coco128/images/train2017/",
        train_ann_file="data/coco128/annotations/instances_train2017.json",
        val_split=0.1,
        transform_kwargs={"image_size": 512},
        batch_size=4,
    )

    # 2. Build the task
    model = ObjectDetector(head="efficientdet", backbone="d0", num_classes=datamodule.num_classes, image_size=512)
    # model = ObjectDetector(head="retinanet", backbone="resnet18_fpn", num_classes=datamodule.num_classes, image_size=512)

    # 3. Create the trainer and finetune the model
    trainer = flash.Trainer(max_epochs=1)
    trainer.finetune(model, datamodule=datamodule, strategy="freeze")

    # 4. Detect objects in a few images!
    datamodule = ObjectDetectionData.from_files(
        predict_files=[
            "data/coco128/images/train2017/000000000625.jpg",
            "data/coco128/images/train2017/000000000626.jpg",
            "data/coco128/images/train2017/000000000629.jpg",
        ],
        transform_kwargs={"image_size": 512},
        batch_size=4,
    )
    predictions = trainer.predict(model, datamodule=datamodule)
    print(predictions)

    # 5. Save the model!
    trainer.save_checkpoint("object_detection_model.pt")


if __name__ == '__main__':
    train()
