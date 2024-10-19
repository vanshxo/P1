from hair_style_mlflow.entity.config_en import trained_model_conf
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from pathlib import Path

class stage_03:
    def __init__(self,config:trained_model_conf):
        self.config=config
        
        self.image_data=None
        self.train_datagen=None
        self.valid_datagen=None

    def load_updated_base_model(self):
        self.updated=load_model(self.config.updated_base_model_path)
    
    @staticmethod
    def save_model(save_path:Path,model:tensorflow.keras.Model):
        model.save(save_path)


    def data_gen(self):
        if self.config.params_augmentation:
            self.image_data=ImageDataGenerator(    
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=0.2
            )
        else:
            self.image_data=ImageDataGenerator(    
            rescale=1./255,
            validation_split=0.2
            )
        self.train_datagen=self.image_data.flow_from_directory(self.config.training_data,target_size=(self.config.params_image_size[0],self.config.params_image_size[1]),subset='training')
        self.valid_datagen=self.image_data.flow_from_directory(self.config.training_data,target_size=(self.config.params_image_size[0],self.config.params_image_size[1]),subset='validation')
    def training(self):
        opti=Adam(learning_rate=0.0001)
        self.updated.compile(optimizer=opti, loss='categorical_crossentropy', metrics=['accuracy'])
        self.updated.fit(self.train_datagen,validation_data=self.valid_datagen,epochs=self.config.params_epochs)
        self.save_model(self.config.trained_model_path,self.updated)




        