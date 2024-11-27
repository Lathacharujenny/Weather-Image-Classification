from src.config.configuration import ConfigurationManager
from src.components.model_training import ModelTraining


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_training_config = config.get_model_training_config()
        model_training = ModelTraining(config=model_training_config)
        model_training.loading_data()
        model_training.efficientnet_model()
        model_training.resnet50_model()
        model_training.resnet152v2_model()
        model_training.mobilenet_model()
        model_training.vgg19_model()
        model_training.vgg16_model()
        model_training.inceptionresnetv2_model()
        model_training.xception_model()
        model_training.resnet101_model()
        model_training.densenet201_model()