from src.config.configuration import ConfigurationManager
from src.components.model_evaluate import ModelEvaluate



class ModelEvaluatePipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_evaluate_config = config.get_model_evaluate_config()
        model_evaluate = ModelEvaluate(config=model_evaluate_config)
        model_evaluate.load_test_gen_models()
        model_evaluate.evaluate_models()
        model_evaluate.convert_results_csv()
        model_evaluate.visualize_the_scores()