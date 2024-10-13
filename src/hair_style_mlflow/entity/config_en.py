from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class data_config:
    root_dir:Path
    file_url:str
    zipped_file:Path
    unzipped:Path
    
