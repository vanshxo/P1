from hair_style_mlflow.entity.config_en import data_config
import gdown
from hair_style_mlflow import logger
import os
import zipfile
class stage_01:
    def __init__(self,config:data_config):
        self.config=config

    def download(self):

        try:
            url=self.config.file_url
            file_id=url.split('/')[-2]
            download_url='https://drive.google.com/uc?export=download&id='
            download_url=download_url+file_id


            logger.info(f"downloading data to {self.config.root_dir} ")
            os.makedirs(self.config.root_dir,exist_ok=True)
            gdown.download(download_url,self.config.zipped_file)

        except Exception as e:
            raise e
        
    def unzip(self):
        unzip_path=self.config.unzipped
        with zipfile.ZipFile(self.config.zipped_file,'r') as file:
            file.extractall(unzip_path)
        logger.info("data in successfully unzipped!!>>")





        