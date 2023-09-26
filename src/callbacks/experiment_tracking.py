from typing import Dict, Optional

import pytorch_lightning as pl
from clearml import OutputModel, Task

from src.config import ExperimentConfig


class ClearMLTracking(pl.Callback):
    def __init__(
        self,
        cfg: ExperimentConfig,
        label_enumeration: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.label_enumeration = label_enumeration
        self.task: Optional[Task] = None
        self.output_model: Optional[OutputModel] = None

    def on_fit_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        self._setup_task()

    def _setup_task(self):
        Task.force_requirements_env_freeze()  # or use task.set_packages() for more granular control.
        self.task = Task.init(
            project_name=self.cfg.project_name,
            task_name=self.cfg.experiment_name,
            # If `output_uri=True` uses default ClearML output URI,
            # can use string value to specify custom storage URI like S3.
            output_uri=True,
            auto_connect_frameworks={'tensorboard': True, 'detect_repository': True}
        )
        self.task.connect_configuration(configuration=self.cfg.model_dump())
        self.output_model = OutputModel(task=self.task, label_enumeration=self.label_enumeration)
