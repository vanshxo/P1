STAGE_03='Model training stage'
from hair_style_mlflow.components.stage_03 import stage_03
from hair_style_mlflow.config.configuration import config_manager
from hair_style_mlflow import logger
class stage_03_p:
    def __init__(self):
        pass
    def main(self):
        manager=config_manager()
        train_conf=manager.trained_model_conf()
        st3=stage_03(train_conf)
        st3.load_updated_base_model()
        st3.data_gen()
        
        st3.training()


if __name__ =='__main__':
    logger.info(">>Stage 03 has started!!!<<")
    obj=stage_03_p()
    obj.main()