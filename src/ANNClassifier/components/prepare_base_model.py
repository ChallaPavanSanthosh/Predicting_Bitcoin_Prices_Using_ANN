import os
import urllib.request as request
import tensorflow as tf

from pathlib import Path
from ANNClassifier.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig, input_dim: int):
        self.config = config
        self.input_dim = input_dim

        # Flatten and ensure dropout values are valid floats
        # self.config.dropout_range = self._flatten_and_cast(self.config.dropout_range)
        self.dropout_range = self._flatten_and_cast(self.config.dropout_range)

    @staticmethod
    def _flatten_and_cast(dropout_range):
        # Support both list and string formats
        if isinstance(dropout_range, list):
            flat = []
            for d in dropout_range:
                if isinstance(d, list):
                    flat.extend(d)
                else:
                    flat.append(d)
            return [float(val) for val in flat]
        elif isinstance(dropout_range, str):
            import ast
            return [float(x) for x in ast.literal_eval(dropout_range)]
        else:
            raise ValueError("Invalid format for dropout_range")

    def get_base_model(self):
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.Input(shape=(self.input_dim,)))

        num_layers = max(self.config.num_layers_range)
        units = max(self.config.units_range)
        # dropout = min(self.config.dropout_range)  # safest (most regularized) option
        dropout = min(self.dropout_range)

        # Add configurable dense layers
        for i in range(num_layers):
            self.model.add(tf.keras.layers.Dense(units, activation='relu'))
            self.model.add(tf.keras.layers.Dropout(dropout))

        # Final regression layer
        self.model.add(tf.keras.layers.Dense(1, activation='linear'))

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=float(self.config.learning_rate)),
            loss='mean_squared_error',
            metrics=['mean_absolute_error']
        )

        # ✅ Show model summary in terminal
        print("\n🧠 Model Architecture:\n")
        self.model.summary()  # <-- This line prints the architecture

        # Save base (untrained) model
        self.save_model(path=self.config.model_save_dir / "base_model.h5", model=self.model)

    def update_base_model(self):
        # Final trained model can be the same model in your setup
        self.full_model = self.model
        self.save_model(path=self.config.model_save_dir / "updated_model.h5", model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
