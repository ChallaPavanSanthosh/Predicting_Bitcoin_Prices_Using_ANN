from ANNClassifier.config.configuration import ConfigurationManager
# from ANNClassifier.components.model_evaluation_mlflow import Evaluation
from ANNClassifier.components.model_evaluation import Evaluation
from ANNClassifier import logger



STAGE_NAME = "Evaluation stage"


class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        evaluation = Evaluation(eval_config)
        # evaluation.evaluation()
        # evaluation.save_score()
        evaluation.run()
        print("✅ Evaluation complete. Scores saved to scores.json")




if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e