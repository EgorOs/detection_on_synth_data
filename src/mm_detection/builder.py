import os
import time
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, Union

from mmengine import Config, ConfigDict
from mmengine.runner import Runner

from src.mm_detection.constants import DET_RES_DIR, DETECTION_CLASSES
from src.mm_detection.dataloaders import build_dataloader_cfg
from src.mm_detection.hooks import CUSTOM_HOOKS
from src.mm_detection.transforms import test_pipeline, train_pipeline

ConfigSourceType = Union[str, Path, Config]


class RunnerBuilder:  # noqa: WPS306
    def __init__(
        self,
        data_path: Path,
        model_cfg: ConfigSourceType,
        trainer_cfg: ConfigSourceType,
        work_dir: Optional[str] = None,
    ):
        self._data_path = data_path
        self._model_cfg = _read_cfg(model_cfg)
        self._trainer_cfg = _read_cfg(trainer_cfg)

        base_work_dir = work_dir if work_dir else os.getcwd()
        timestamp = str(int(time.time()))
        self._work_dir = os.path.join(base_work_dir, DET_RES_DIR, timestamp)

    def get_dataloaders(self) -> Tuple[Dict[str, Any], ...]:
        data_path = str(self._data_path)

        train_dataloader = build_dataloader_cfg(
            train_pipeline,
            data_root=data_path,
            ann_file=os.path.join(data_path, 'train.json'),
            pin_memory=True,
        )
        val_dataloader = build_dataloader_cfg(
            test_pipeline,
            data_root=data_path,
            ann_file=os.path.join(data_path, 'val.json'),
            pin_memory=False,
        )
        test_dataloader = build_dataloader_cfg(
            test_pipeline,
            data_root=data_path,
            ann_file=os.path.join(data_path, 'test.json'),
            pin_memory=False,
        )
        return train_dataloader, val_dataloader, test_dataloader

    def get_evaluator(self, name: Literal['val', 'test']):
        return dict(
            type='CocoMetric',
            ann_file=str(self._data_path / f'{name}.json'),
            metric='bbox',
            format_only=False,
            backend_args=None,
            proposal_nums=(100, 1, 10),
        )

    def build_runner(self) -> Runner:
        self._override_model_config()

        train_dataloader, val_dataloader, test_dataloader = self.get_dataloaders()
        # TODO: turn it into a config (cfg from dict) and log to experiment tracking + pretty print
        cfg = Config(
            cfg_dict=dict(
                model=self._model_cfg['model'],
                work_dir=self._work_dir,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                test_dataloader=test_dataloader,
                train_cfg=self._trainer_cfg['train_cfg'],
                val_cfg=self._trainer_cfg['val_cfg'],
                test_cfg=self._trainer_cfg['test_cfg'],
                auto_scale_lr=self._trainer_cfg['auto_scale_lr'],
                optim_wrapper=self._trainer_cfg['optim_wrapper'],
                param_scheduler=self._trainer_cfg['param_scheduler'],
                val_evaluator=self.get_evaluator('val'),
                test_evaluator=self.get_evaluator('test'),
                default_hooks=self._trainer_cfg['default_hooks'],
                custom_hooks=CUSTOM_HOOKS,
                data_preprocessor=None,  # Defined in model config.
                load_from=self._trainer_cfg['load_from'],
                resume=self._trainer_cfg['resume'],
                launcher='none',
                env_cfg=self._trainer_cfg['env_cfg'],
                log_processor=None,
                log_level=self._trainer_cfg['log_level'],
                visualizer=self._trainer_cfg['visualizer'],
                default_scope=self._trainer_cfg['default_scope'],
                randomness={'seed': 0},
                experiment_name='experiment_1',
                cfg=None,
            ),
        )
        return Runner.from_cfg(cfg)

    def _override_model_config(self):
        self._model_cfg['model']['bbox_head']['num_classes'] = len(DETECTION_CLASSES)


def _read_cfg(cfg: ConfigSourceType) -> ConfigDict:
    if not isinstance(cfg, Config):
        cfg = Config.fromfile(cfg)
    return Config._dict_to_config_dict(dict(cfg))  # noqa: WPS437
