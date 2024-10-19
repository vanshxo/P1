from urllib.parse import urlparse
import mlflow
import tensorflow
from pathlib import Path
from hair_style_mlflow.utils.common_utils import save_json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import mlflow.keras
from hair_style_mlflow.entity.config_en import evaluation_conf

class stage_04:
    def __init__(self,config:evaluation_conf):
        self.config=config
        self.valid_datagen=None
    def valid_gen(self):
        self.image_data=ImageDataGenerator(    
            rescale=1./255,
            validation_split=0.3
            )
        self.valid_datagen=self.image_data.flow_from_directory(self.config.data_path,target_size=(self.config.params_image_size[0],self.config.params_image_size[1]),subset='validation')

    @staticmethod
    def load_model(model_path:Path)-> tensorflow.keras.Model:
        return tensorflow.keras.models.load_model(model_path)
    
    
    def evaluation(self):
        self.model=self.load_model(self.config.model_path)
        self.valid_gen()
        self.score=self.model.evaluate(self.valid_datagen)
        self.save_score()
    
    def save_score(self):
        score={"loss":self.score[0],"accuracy":self.score[1]}
        save_json(Path('score.json'),score)

    def login_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )
            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.keras.log_model(self.model,artifact_path='artifacts/models/trained_model.h5',keras_model_kwargs={"save_format": "h5"}, registered_model_name="MobileNetV2")
            else:
               
               mlflow.keras.log_model(self.model, artifact_path='artifacts/models/trained_model.h5', keras_model_kwargs={"save_format": "h5"})



