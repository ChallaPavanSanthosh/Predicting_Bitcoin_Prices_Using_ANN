from pathlib import Path
from urllib.parse import urlparse

import tensorflow as tf
import matplotlib.pyplot as plt
import pickle

from ANNClassifier.entity.config_entity import EvaluationConfig
from ANNClassifier.utils.common import save_json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def load_model(self):
        self.model = tf.keras.models.load_model(self.config.path_of_model)

    def load_data(self):
        with open(self.config.training_data, "rb") as f:
            data = pickle.load(f)
        self.X_val = data["X_val"]
        self.y_val = data["y_val"]

    def evaluate(self):
        self.y_pred = self.model.predict(self.X_val)
        self.mse = mean_squared_error(self.y_val, self.y_pred)
        self.mae = mean_absolute_error(self.y_val, self.y_pred)
        self.r2 = r2_score(self.y_val, self.y_pred)


        # Create directory if it doesn't exist
        output_dir = Path("artifacts/plots")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Plot actual vs predicted
        plt.figure(figsize=(12, 6))
        plt.plot(self.y_val, label="Actual")
        plt.plot(self.y_pred, label="Predicted")
        plt.title("Model Prediction vs Actual")
        plt.xlabel("Time Steps")
        plt.ylabel("Bitcoin Price")
        plt.legend()
        plt.grid(True)

        # Save to file
        plot_path = output_dir / "actual_vs_predicted.png"
        plt.savefig(plot_path)
        plt.close()

        print(f"📈 Plot saved at: {plot_path}")



    def save_score(self):
        scores = {
            "mean_squared_error": self.mse,
            "mean_absolute_error": self.mae,
            "r2_score": self.r2
        }
        save_json(path=Path("artifacts/scores.json"), data=scores)

    def run(self):
        self.load_model()
        self.load_data()
        self.evaluate()
        self.save_score()