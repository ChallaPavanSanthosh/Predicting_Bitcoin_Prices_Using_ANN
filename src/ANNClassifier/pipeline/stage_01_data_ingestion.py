from ANNClassifier.config.configuration import ConfigurationManager
from ANNClassifier.components.data_ingestion import DataIngestion
from ANNClassifier import logger


STAGE_NAME = "Data Ingestion stage"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass 

    def main(self):
        # Step 1: Load config
        config = ConfigurationManager()

        # Step 2: Get data ingestion configuration
        data_ingestion_config = config.get_data_ingestion_config()

        # Step 3: Initialize ingestion
        data_ingestion = DataIngestion(config=data_ingestion_config)

        # Step 4: Download the CSV file
        data_ingestion.download_file()

        # Step 5: Load the CSV into a DataFrame
        df = data_ingestion.load_csv()

        # Optional: Preview or log DataFrame shape
        print(df.head())  # or logger.info(df.head())
        print(f"✅ CSV loaded successfully. Shape: {df.shape}")


if __name__ == '__main__':
    try:
        logger.info(f">> stage {STAGE_NAME} started <<")
        obj =DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">> stage {STAGE_NAME} completed <<\n\nx====x")

    except Exception as e:
        logger.exception(e)
        raise e

