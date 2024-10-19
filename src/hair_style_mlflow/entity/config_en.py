from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class data_config:
    root_dir:Path
    file_url:str
    zipped_file:Path
    unzipped:Path


@dataclass(frozen=True)
class base_model_config:
    root_dir:Path
    base_path:Path
    updated_base_path:Path
    params_image_size:list
    params_include_top:bool
    params_learning_rate:float
    params_weights:str
    params_classes:int

@dataclass(frozen=True)
class trained_model_conf:
    root_dir:Path
    updated_base_model_path:Path
    trained_model_path:Path
    training_data:Path
    params_epochs:int
    params_image_size:list
    params_augmentation:bool


@dataclass(frozen=True)
class evaluation_conf:
    model_path: Path
    data_path: Path
    all_params: dict
    mlflow_uri: str
    params_image_size: list
    



