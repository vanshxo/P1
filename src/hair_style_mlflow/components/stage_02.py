from hair_style_mlflow import logger
from hair_style_mlflow.entity.config_en import base_model_config
from tensorflow.keras.applications import MobileNetV2
from pathlib import Path
from tensorflow.keras.layers import Dense,Flatten,GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow

class stage_02:
    def __init__(self,config:base_model_config):
        self.base_config=config
        self.base_model=None


    def get_base_model(self):
        self.base_model=MobileNetV2(
            input_shape=self.base_config.params_image_size,
            include_top=self.base_config.params_include_top,
            weights=self.base_config.params_weights
        )
        self.save_model(self.base_config.base_path,self.base_model)

    @staticmethod
    def save_model(model_path:Path,model:tensorflow.keras.Model):
        model.save(model_path)
    def update_base_model(self):
        self.base_model.trainable=False
        # Add custom layers on top of the base model
        x = self.base_model.output
        x = GlobalAveragePooling2D()(x)  # Add a global spatial average pooling layer
        x = Dense(1024, activation='relu')(x)  # Add a fully connected layer

        # Add the final classification layer (change 10 to the number of your classes)
        predictions = Dense(self.base_config.params_classes, activation='softmax')(x)

        # Combine the base model and the new top layers into a single model
        model = Model(inputs=self.base_model.input, outputs=predictions)
        opti=Adam(learning_rate=0.0001)
        model.compile(optimizer=opti, loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        self.save_model(self.base_config.updated_base_path,model)

            
