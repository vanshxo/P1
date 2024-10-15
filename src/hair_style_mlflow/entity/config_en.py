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