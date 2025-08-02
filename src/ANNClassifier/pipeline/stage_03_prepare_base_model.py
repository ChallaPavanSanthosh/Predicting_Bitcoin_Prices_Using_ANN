# from ANNClassifier.config.configuration import ConfigurationManager
# from ANNClassifier.components.prepare_base_model import PrepareBaseModel
# # from ANNClassifier.pipeline.stage_02_data_preparation import input_dim
# from ANNClassifier import logger
# import json

# STAGE_NAME = "Prepare base model"


# class PrepareBaseModelTrainingPipeline:
#     def __init__(self):
#         pass

#     def main(self):
#         config = ConfigurationManager()
#         prepare_base_model_config = config.get_prepare_base_model_config()
#         # ✅ Load input_dim from JSON file
#         with open("artifacts/input_dim.json", "r") as f:
#             input_dim = json.load(f)["input_dim"]

#         prepare_base_model = PrepareBaseModel(config=prepare_base_model_config, input_dim=input_dim)
#         # prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
#         prepare_base_model.get_base_model()
#         prepare_base_model.update_base_model()



# if __name__ == '__main__':
#     try:
#         logger.info(f"*******************")
#         logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#         obj = PrepareBaseModelTrainingPipeline()
#         obj.main()
#         logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
#     except Exception as e:
#         logger.exception(e)
#         raise e



from ANNClassifier.config.configuration import ConfigurationManager
from ANNClassifier.components.prepare_base_model import PrepareBaseModel
from ANNClassifier import logger
import json

STAGE_NAME = "Prepare base model"

class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()

        

        # ✅ Load input_dim if needed
        with open("artifacts/input_dim.json", "r") as f:
            input_dim = json.load(f)["input_dim"]

        # ✅ Initialize and call the correct method to print architecture
        # prepare_base_model = PrepareBaseModel(config=prepare_base_model_config, input_dim=input_dim)
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.prepare_model()  # ✅ This builds, prints, and saves the model

        

if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e