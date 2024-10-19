from hair_style_mlflow.config.configuration import config_manager
from hair_style_mlflow.components.stage_04 import stage_04
from hair_style_mlflow import logger
STAGE_NAME='model evaluation stage'

class stage_04_p:
    def __init__(self):
        pass

    def main(self):
        conf=config_manager()
        eval_conf=conf.evaluation_config()
        st4=stage_04(eval_conf)
        st4.evaluation()
        st4.save_score()
        st4.login_mlflow()

if __name__=='__main__':
    logger.info('>>>Stage 04 model_evaluation has started<<<')
    obj=stage_04_p()
    obj.main()
    logger.info(">>Stage 04 is completed!!!<<")