from hair_style_mlflow.config.configuration import config_manager
from hair_style_mlflow.components.stage_01 import stage_01
from hair_style_mlflow import logger
STAGE_NAME='data_download_stage'

class Stage_01_p:
    
    def __init__(self):
        pass

    def main(self):
        manager=config_manager()
        data_conf=manager.data_conf()
        st1=stage_01(data_conf)
        st1.download()
        st1.unzip()


if __name__=='__main__':
    try:
        logger.info("<<Starting stage 1 data download stage>>")
        obj=Stage_01_p()
        obj.main()
    except Exception as e:
        raise e

        