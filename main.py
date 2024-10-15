from hair_style_mlflow.pipeline.stage_01p import (Stage_01_p)
from hair_style_mlflow.pipeline.stage_02p import stage_02_p
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