from pathlib import Path
from hair_style_mlflow.constants import *
from hair_style_mlflow.entity.config_en import (data_config,base_model_config,trained_model_conf)
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
    
    def base_model_conf(self)->base_model_config:
        base_conf=self.config.stage_02
        create_dir([base_conf.root_dir])

        return base_model_config(root_dir=base_conf.root_dir,
                                 base_path=base_conf.base_model,
                                 updated_base_path=base_conf.updated_base_model,
                                 params_image_size=base_conf.params_image_size,
                                 params_include_top=base_conf.params_include_top,
                                 params_weights=base_conf.params_weights,
                                 params_learning_rate=base_conf.params_learning_rate,
                                 params_classes=base_conf.params_classes
        )
    
    def trained_model_conf(self)->trained_model_conf:
        trained_config=self.config.stage_03
        create_dir([trained_config.root_dir])

        return trained_model_conf(root_dir=trained_config.root_dir,
                                  updated_base_model_path=trained_config.updated_base_model_path,
                                  trained_model_path=trained_config.trained_model_path,
                                  training_data=trained_config.training_data,
                                  params_epochs=self.params.epochs,
                                  params_image_size=self.params.image_size,
                                  params_augmentation=self.params.augmentation
                                  )


        


    
