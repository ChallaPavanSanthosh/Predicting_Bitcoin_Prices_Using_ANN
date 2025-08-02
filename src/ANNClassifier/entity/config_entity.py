from dataclasses import dataclass
from pathlib import Path
from typing import List

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: str
    base_model_path: str
    updated_base_model_path: str
    num_layers_range: List[int]
    units_range: List[int]
    dropout_range: List[float]
    learning_rate: float
    model_save_dir: Path
    input_shape: int

    model_save_dir: Path
    updated_base_model_path: Path

@dataclass(frozen=True)
class ANNTrainingConfig:
    root_dir: Path
    data_file: Path
    model_save_dir: Path
    updated_base_model_path: Path
    trained_model_path: Path

    # Data parameters
    sample_size: int
    target_column: str

    # Training parameters
    epochs_tuning: int
    epochs_final: int
    batch_size: int
    learning_rate: float

    # Hyperparameter tuning ranges
    max_trials: int
    num_layers_range: List[int]
    units_range: List[int]
    dropout_range: List[float]



@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model: Path                 # Path to the trained .h5 model
    training_data: Path                 # Path to train_val_data.pkl
    all_params: dict                    # All hyperparameters
    params_learning_rate: float
    # mlflow_uri: str                     # URI for MLflow tracking
    params_epochs: int                 # Number of epochs used
    params_batch_size: int 