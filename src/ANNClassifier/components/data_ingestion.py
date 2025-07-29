import os
import pandas as pd
import zipfile
import gdown
from ANNClassifier import logger
from ANNClassifier.utils.common import get_size
from ANNClassifier.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config


    # def download_file(self) -> str:
    #     '''
    #     Fetch data from the url
    #     '''

    #     try:
    #         dataset_url = self.config.source_URL
    #         download_path = self.config.local_data_file
    #         os.makedirs("artifacts/data_ingestion", exist_ok=True)
    #         logger.info(f"Downloading data from {dataset_url} into {download_path}")

            
    #         gdown.download(dataset_url, str(download_path), quiet=False, fuzzy=True)

    #         file_id = dataset_url.split("/")[-2]
    #         prefix = 'https://drive.google.com/uc?export=download&id='
    #         gdown.download(prefix+file_id)

    #         logger.info(f"Downloaded data from {dataset_url} into file {download_path}")

    #     except Exception as e:
    #         raise e

    def download_file(self) -> str:
        """
        Fetch data from the URL and save to local_data_file path.
        """
        try:
            dataset_url = self.config.source_URL
            download_path = self.config.local_data_file
            os.makedirs(os.path.dirname(download_path), exist_ok=True)

            logger.info(f"Downloading data from {dataset_url} into {download_path}")
            gdown.download(dataset_url, str(download_path), quiet=False, fuzzy=True)

            logger.info(f"Downloaded data from {dataset_url} into file {download_path}")
        except Exception as e:
            raise e
    
    def load_csv(self) -> pd.DataFrame:
        try:
            logger.info(f"Loading CSV file from {self.config.local_data_file}")
            df = pd.read_csv(self.config.local_data_file)
            logger.info(f"CSV file loaded with shape: {df.shape}")
            return df
        
        except Exception as e:
            logger.error("Failed to load CSV")
            raise e