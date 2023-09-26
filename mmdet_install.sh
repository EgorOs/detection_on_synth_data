#bin/bash

# OpenMMLab rely on their own package manager to solve dependencies.
# https://github.com/open-mmlab/mim/tree/main/mim

# TODO: check whether it's possible to handle all the dependencies with `poetry`.
mim install mmengine==0.7.4
mim install mmcv==2.0.0
mim install mmdet==3.0.0
