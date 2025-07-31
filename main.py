# from ANNClassifier import logger
# from ANNClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
# # from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
# # from cnnClassifier.pipeline.stage_03_model_trainer import ModelTrainingPipeline
# # from cnnClassifier.pipeline.stage_04_model_evaluation import EvaluationPipeline



# STAGE_NAME = "Data Ingestion stage"


# try:
#     logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#     obj = DataIngestionTrainingPipeline()
#     obj.main()
#     logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#     logger.exception(e)
#     raise e


from ANNClassifier import logger
from ANNClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from ANNClassifier.pipeline.stage_02_data_preparation import DataPreparationTrainingPipeline
from ANNClassifier.pipeline.stage_03_prepare_base_model import PrepareBaseModelTrainingPipeline
# from ANNClassifier.pipeline.stage_03_model_trainer import ModelTrainingPipeline
# from ANNClassifier.pipeline.stage_04_model_evaluation import EvaluationPipeline



STAGE_NAME = "Data Ingestion stage"


try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e



STAGE_NAME = "Data Preparation"

if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataPreparationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e



STAGE_NAME = "Prepare base model"
try: 
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   prepare_base_model = PrepareBaseModelTrainingPipeline()
   prepare_base_model.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)

