from hair_style_mlflow.pipeline.stage_01p import (Stage_01_p)
from hair_style_mlflow.pipeline.stage_02p import stage_02_p
from hair_style_mlflow.pipeline.stage_03p import stage_03_p
from hair_style_mlflow.pipeline.stage_04p import stage_04_p
from hair_style_mlflow import logger

try:
    logger.info("<<Starting stage 1 data download stage>>")
    obj=Stage_01_p()
    obj.main()
except Exception as e:
    raise e


try:
    obj=stage_02_p()
    obj.main()
    logger.info(f">>Stage 02  is completed<<")
except Exception as e:
    raise e

try:
    obj=stage_03_p()
    obj.main()
    logger.info(f">>Stage 03  is completed<<")
except Exception as e:
    raise e

try:
    logger.info('>>>Stage 04 model_evaluation has started<<<')
    obj=stage_04_p()
    obj.main()
    logger.info(">>Stage 04 is completed!!!<<")
except Exception as e:
    raise e
