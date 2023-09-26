from pathlib import Path

import cv2
import imageio
import numpy as np
import pyrootutils
from mmdet.apis import inference_detector, init_detector

PROJECT_ROOT = pyrootutils.setup_root(__file__, pythonpath=True)
MODEL_PATH = PROJECT_ROOT / 'model'
DEMO_IMAGES_IN = PROJECT_ROOT / 'media' / 'demo_images' / 'input'
NOISE_THRESHOLD = 0.2


def run_inference(imgs_path: Path, pattern: str = '*.jpg', device: str = 'cpu'):  # noqa: WPS210: Fixme
    model = init_detector(MODEL_PATH / 'config.py', str(MODEL_PATH / 'weights.pth'), device)

    for impath in imgs_path.rglob(pattern):
        img = imageio.v3.imread(impath)
        pred_instances = inference_detector(model, img).pred_instances

        bboxes = pred_instances.bboxes.cpu().numpy().astype(np.float32).tolist()
        labels = pred_instances.labels.cpu().numpy().astype(np.int32).tolist()
        scores = pred_instances.scores.cpu().numpy().astype(np.float32).tolist()
        for cls_score, bbox, _cls_label in zip(scores, bboxes, labels):
            if cls_score >= NOISE_THRESHOLD:
                x0, y0, x1, y1 = [int(coord) for coord in bbox]
                img = cv2.rectangle(img, (x0, y0), (x1, y1), (255, 0, 0), 3)  # noqa: WPS221

        save_to = impath.parents[1] / 'output' / impath.name
        imageio.v3.imwrite(save_to, img)
        print(f'Image saved to {save_to}')  # noqa: WPS421


if __name__ == '__main__':
    run_inference(imgs_path=DEMO_IMAGES_IN)
