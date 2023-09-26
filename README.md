# Object detection on synthetic data

<img src=media/header.png>

Creating synthetic dataset in blender and training object detection model

<a href="https://github.com/open-mmlab/mmdetection"><img alt="PytorchLightning" src="https://img.shields.io/badge/MMDetection-dcdde1?logo=pytorch&style=flat"></a>
<a href="https://clear.ml/docs/latest/"><img alt="Config: Hydra" src="https://img.shields.io/badge/MLOps-Clear%7CML-%2309173c"></a>

# Getting started

1. Follow [instructions](https://github.com/python-poetry/install.python-poetry.org)
   to install Poetry:
   ```bash
   # Unix/MacOs installation
   curl -sSL https://install.python-poetry.org | python3 -
   ```
1. Check that poetry was installed successfully:
   ```bash
   poetry --version
   ```
1. Setup workspace:
   ```bash
   make setup_ws
   ```
1. Setup ClearML:
   ```bash
   clearml-init
   ```
1. Download dataset to your local workspace:
   ```bash
   make migrate_dataset
   ```

# Training

```bash
make run_training
```

# Inference

```bash
make get_weights
make run_inference
```

# Results

| Image 1                                            | Image 2                                            | Image 3                                            |
|----------------------------------------------------|----------------------------------------------------|----------------------------------------------------|
| <img src=media/demo_images/output/stop_sign_1.jpg> | <img src=media/demo_images/output/stop_sign_2.jpg> | <img src=media/demo_images/output/stop_sign_3.jpg> |
