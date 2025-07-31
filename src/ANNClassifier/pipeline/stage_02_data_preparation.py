from ANNClassifier.components.data_preparation import DataPreparation
from ANNClassifier.config.configuration import ConfigurationManager
from ANNClassifier import logger
import json

STAGE_NAME = "Data Preparation Stage"

class DataPreparationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_path = config.get_data_ingestion_config().local_data_file
        sample_size = config.params['SAMPLE_SIZE']
        target_column = config.params['TARGET_COLUMN']

        # Prepare data
        prep = DataPreparation(
            file_path=data_path,
            sample_size=sample_size,
            target_column=target_column
        )
        result = prep.prepare()
        
        # Unpack and store globally or pass to next stages
        global X_train, X_val, X_test, y_train, y_val, y_test, scaler_y, input_dim
        X_train, X_val, X_test, y_train, y_val, y_test, scaler_y, input_dim = result


        # ✅ Save input_dim to a JSON file
        with open("artifacts/input_dim.json", "w") as f:
            json.dump({"input_dim": input_dim}, f)
        
        logger.info(f"✅ Data preparation completed. Input dim: {input_dim}")

if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataPreparationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
