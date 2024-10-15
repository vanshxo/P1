from pathlib import Path
from hair_style_mlflow.config.configuration import config_manager
from hair_style_mlflow.entity.config_en import base_model_config
from hair_style_mlflow.components.stage_02 import stage_02
from hair_style_mlflow import logger
STAGE_NAME='Model pretraining stage'
class stage_02_p:
    def __init__(self):
        pass

    def main(self):
        logger.info(f">>Stage 02 {STAGE_NAME} is started<<")
        manager=config_manager()
        base_conf=manager.base_model_conf()
        st2=stage_02(base_conf)
        st2.get_base_model()
        st2.update_base_model()


if __name__=='__main__':
    try:
        obj=stage_02_p()
        obj.main()
        
    except Exception as e:
        raise e


