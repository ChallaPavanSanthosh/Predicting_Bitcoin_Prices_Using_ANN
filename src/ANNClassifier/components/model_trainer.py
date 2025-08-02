import os
import urllib.request as request
import tensorflow as tf
import time
import pickle

from pathlib import Path
from ANNClassifier.entity.config_entity import DataIngestionConfig, PrepareBaseModelConfig, ANNTrainingConfig
from tensorflow.keras.callbacks import ReduceLROnPlateau
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense, Dropout


class ANNModelTrainer:
    def __init__(self, config: ANNTrainingConfig):
        self.config = config
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
        self.config.updated_base_model_path
    )


    def load_data(self):
        """Load preprocessed and saved training data"""
        with open("artifacts/data_preparation/train_val_data.pkl", "rb") as f:
            data = pickle.load(f)

        self.X_train = data["X_train"]
        self.X_val = data["X_val"]
        self.y_train = data["y_train"]
        self.y_val = data["y_val"]

    def load_model(self):
        self.model = tf.keras.models.load_model(self.config.updated_base_model_path)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def train(self):
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-5,
            verbose=1
        )

        history = self.model.fit(
            self.X_train,
            self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=self.config.epochs_final,
            batch_size=self.config.batch_size,
            callbacks=[reduce_lr],
            verbose=1
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )

        print("✅ Model training complete. Model saved to:", self.config.trained_model_path)