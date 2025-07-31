from dataclasses import dataclass
from pathlib import Path
from typing import List

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path



# @dataclass(frozen=True)
# class PrepareBaseModelConfig:
#     root_dir: str
#     base_model_path: str
#     updated_base_model_path: str
#     params_image_size: list
#     params_learning_rate: float
#     params_include_top: bool
#     params_weights: str
#     params_classes: int

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




# @dataclass(frozen=True)
# class ANNTrainingConfig:
#     root_dir: Path
#     data_file: Path
#     model_save_dir: Path
#     project_name: str

#     # Data parameters
#     sample_size: int
#     target_column: str

#     # Training parameters
#     epochs_tuning: int
#     epochs_final: int
#     batch_size: int
#     learning_rate: float

#     # Hyperparameter tuning ranges
#     max_trials: int
#     num_layers_range: List[int]
#     units_range: List[int]
#     dropout_range: List[float]