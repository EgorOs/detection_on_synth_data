from pathlib import Path

import pyrootutils
from clearml import Task
from mmdet.utils import setup_cache_size_limit_of_dynamo

# Adds project root to "PYTHONPATH". Required for remote execution by ClearML agent.
PROJECT_ROOT = pyrootutils.setup_root(__file__, pythonpath=True)
DET_PROJ_PATH = PROJECT_ROOT / 'src' / 'mm_detection'
DET_CFG_PATH = DET_PROJ_PATH / 'config'

from src.common.utils import (  # noqa: E402 source imports should come after `pyrootutils.setup_root`.
    set_random_seed,
    set_unverified_ssl_context,
)
from src.mm_detection.builder import RunnerBuilder  # noqa: E402
from src.mm_detection.data_preprocessor import (  # noqa: E402
    prepare_coco_dataset,
)


def run_training(data_path: Path):
    # Reduce the number of repeated compilations and improve training speed.
    setup_cache_size_limit_of_dynamo()

    Task.force_requirements_env_freeze()

    set_random_seed(0)

    prepare_coco_dataset(data_path, annotations_path='annotations.json')

    runner = RunnerBuilder(
        data_path=data_path,
        model_cfg=DET_CFG_PATH / 'model.py',
        trainer_cfg=DET_CFG_PATH / 'trainer.py',
        work_dir=str(DET_PROJ_PATH),
    ).build_runner()

    # TODO: Register custom ClearML hooks with `runner.register_hook()`.

    # start training
    with set_unverified_ssl_context():
        runner.train()


if __name__ == '__main__':
    run_training(data_path=PROJECT_ROOT / 'dataset' / '35')
