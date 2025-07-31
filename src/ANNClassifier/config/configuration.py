from ANNClassifier.constants import *
from ANNClassifier.utils.common import read_yaml, create_directories
from ANNClassifier.entity.config_entity import DataIngestionConfig, PrepareBaseModelConfig


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)


        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
            config = self.config.data_ingestion

            create_directories([config.root_dir])

            data_ingestion_config = DataIngestionConfig(
                root_dir= config.root_dir,
                source_URL= config.source_URL,
                local_data_file= config.local_data_file,
            )
            return data_ingestion_config
    
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config['prepare_base_model']
        params = self.params

        create_directories([config['root_dir']])

        return PrepareBaseModelConfig(
            root_dir=config['root_dir'],
            base_model_path=config['base_model_path'],
            updated_base_model_path=config['updated_base_model_path'],
            num_layers_range=params['NUM_LAYERS_RANGE'],
            units_range=params['UNITS_RANGE'],
            dropout_range=params['DROPOUT_RANGE'],
            learning_rate=params['LEARNING_RATE'],
            model_save_dir=Path(config['root_dir'])
        )


    
    # def get_ann_training_config(self) -> ANNTrainingConfig:
    #     config = self.config.ann_training  # section in config.yaml
    #     params = self.params

    #     create_directories([config.model_save_dir])

    #     ann_training_config = ANNTrainingConfig(
    #         root_dir=Path(config.root_dir),
    #         data_file=Path(config.data_file),
    #         model_save_dir=Path(config.model_save_dir),
    #         project_name=config.project_name,

    #         sample_size=params.SAMPLE_SIZE,
    #         target_column=params.TARGET_COLUMN,

    #         epochs_tuning=params.EPOCHS_TUNING,
    #         epochs_final=params.EPOCHS_FINAL,
    #         batch_size=params.BATCH_SIZE,
    #         learning_rate=params.LEARNING_RATE,

    #         max_trials=params.MAX_TRIALS,
    #         num_layers_range=params.NUM_LAYERS_RANGE,
    #         units_range=params.UNITS_RANGE,
    #         dropout_range=params.DROPOUT_RANGE
    #     )

    #     return ann_training_config
