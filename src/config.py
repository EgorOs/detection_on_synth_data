from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import yaml
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field


class _BaseValidatedConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')  # Disallow unexpected arguments.


class DataConfig(_BaseValidatedConfig):
    img_size: Tuple[int, int] = (224, 224)
    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = True


class TrainerConfig(_BaseValidatedConfig):
    min_epochs: int = 7  # prevents early stopping
    max_epochs: int = 20

    # perform a validation loop every N training epochs
    check_val_every_n_epoch: int = 3

    log_every_n_steps: int = 50

    gradient_clip_val: Optional[float] = None
    gradient_clip_algorithm: Optional[Literal['norm', 'value']] = None

    # set True to ensure deterministic results
    # makes training slower but gives more reproducibility than just setting seeds
    deterministic: bool = False

    fast_dev_run: bool = False
    default_root_dir: Optional[Path] = None

    detect_anomaly: bool = False


class ExperimentConfig(_BaseValidatedConfig):
    project_name: str = 'synth_data_obj_det'
    experiment_name: str = 'obj_det'
    track_in_clearml: bool = True
    trainer_config: TrainerConfig = Field(default=TrainerConfig())
    data_config: DataConfig = Field(default=DataConfig())

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> 'ExperimentConfig':
        cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        return cls(**cfg)

    def to_yaml(self, path: Union[str, Path]):
        with open(path, 'w') as out_file:
            yaml.safe_dump(self.model_dump(), out_file, default_flow_style=False, sort_keys=False)