import os
import sys
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from ANNClassifier.entity.config_entity import PrepareBaseModelConfig
from pathlib import Path


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def build_model(self, input_shape: int) -> tf.keras.Model:
        """Builds a dense neural network for regression tasks"""
        model = Sequential()
        model.add(Dense(128, activation='relu', input_shape=(input_shape,)))
        model.add(Dropout(0.3))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1))  # Regression output
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss='mean_squared_error',
            metrics=['mean_absolute_error']
        )

        return model

    # def save_model(self, model: tf.keras.Model):
    #     """Saves the compiled model to the specified path"""
    #     # model_path = self.config.updated_base_model_path

    #     model_path = Path("artifacts/prepare_base_model/base_model.h5")

        
    #     os.makedirs(model_path.parent, exist_ok=True)
    #     model.save(model_path)
    #     print(f"✅ Base model saved at: {model_path}")


    def save_model(self, model: tf.keras.Model):
        """Saves the compiled model to the specified path"""
        model_path = Path("artifacts/prepare_base_model/base_model.h5")
        os.makedirs(model_path.parent, exist_ok=True)
        model.save(model_path)
        print(f"✅ Base model saved at: {model_path}")

    def prepare_model(self):
        """Main method to build, print, and save the base model"""
        model = self.build_model(input_shape=self.config.input_shape)

        print("\n🧠 Model Architecture:\n")
        model.summary()  # ✅ This prints the architecture
        sys.stdout.flush()

        self.save_model(model)