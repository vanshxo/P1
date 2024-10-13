from pathlib import Path
from hair_style_mlflow.constants import *
from hair_style_mlflow.entity.config_en import data_config
from hair_style_mlflow.utils.common_utils import readYaml,create_dir

class config_manager:
    def __init__(self):
        self.config_file_path=CONFIG_FILE_PATH
        self.params_file_path=PARAMS_FILE_PATH
        self.config= readYaml(self.config_file_path)
        self.params=readYaml(self.params_file_path)
        create_dir([self.config.artifacts_rootdir])
        


    def data_conf(self)->data_config:
        data_conf=self.config.stage_01
        create_dir([data_conf.root_dir])
        return data_config(root_dir=data_conf.root_dir,
                    file_url=data_conf.file_url,
                    zipped_file=data_conf.zipped_dir,
                    unzipped=data_conf.unzipped
                    )
        


    
