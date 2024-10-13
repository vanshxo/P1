from hair_style_mlflow.pipeline.stage_01p import Stage_01_p
from hair_style_mlflow import logger
if __name__=='__main__':
    try:
        logger.info("<<Starting stage 1 data download stage>>")
        obj=Stage_01_p()
        obj.main()
    except Exception as e:
        raise e