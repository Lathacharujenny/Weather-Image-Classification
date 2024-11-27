from src.entity.config_entity import DataIngestionConfig, DataTransformationConfig, DataSplittingConfig, ModelTrainingConfig, ModelEvaluateConfig
import sys
import os
from src.logger import logging
from src.exception import CustomException
from src.utils.common import read_yaml, create_directories
from src.constants import *

class ConfigurationManager:
    def __init__(self, config=CONFIG_FILE_PATH):
        self.config = read_yaml(config)
        create_directories(self.config['artifacts_root'])

    def get_data_ingestion_config(self):
        config = self.config['data_ingestion']
        create_directories(config['root_dir'])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config['root_dir'],
            data_zip_path=config['data_zip_path']
        )

        return data_ingestion_config
    
    def get_data_transformation_config(self):
        config = self.config['data_transformation']
        create_directories(config['root_dir'])

        data_transformation_config = DataTransformationConfig(
            root_dir=config['root_dir'],
            data_path = config['data_path']
        )

        return data_transformation_config
    
    def get_data_splitting_config(self):
        config = self.config['data_splitting']
        create_directories(config['root_dir'])

        data_splitting_config = DataSplittingConfig(
            root_dir=config['root_dir'],
            data_path=config['data_path']
        )

        return data_splitting_config
    
    def get_model_training_config(self):
        config = self.config['model_training']
        create_directories(config['root_dir'])
        create_directories(config['labels_path'])
        create_directories(config['models'])
        create_directories(config['visualize_results'])
        create_directories(config['efficientnet_model'])
        create_directories(config['resnet50_model'])
        create_directories(config['resnet152v2_model'])
        create_directories(config['mobilenet_model'])
        create_directories(config['vgg19_model'])
        create_directories(config['xception_model'])
        create_directories(config['inceptionresnetv2_model'])
        create_directories(config['vgg16_model'])
        create_directories(config['resnet101_model'])
        create_directories(config['densenet201_model'])

        model_training_config = ModelTrainingConfig(
            root_dir=config['root_dir'],
            train_data_path=config['train_data_path'],
            test_data_path=config['test_data_path'],
            labels_path=config['labels_path'],
            models=config['models'],
            visualize_results=config['visualize_results'],
            efficientnet_model = config['efficientnet_model'],
            resnet50_model = config['resnet50_model'],
            resnet152v2_model = config['resnet152v2_model'],
            mobilenet_model = config['mobilenet_model'],
            vgg19_model = config['vgg19_model'],
            xception_model = config['xception_model'],
            inceptionresnetv2_model = config['inceptionresnetv2_model'],
            vgg16_model = config['vgg16_model'],
            resnet101_model = config['resnet101_model'],
            densenet201_model = config['densenet201_model']
        )

        return model_training_config
    
    def get_model_evaluate_config(self):
        config = self.config['model_evaluate']
        create_directories(config['root_dir'])
        create_directories(config['scores_json'])
        create_directories(config['scores_csv'])
        create_directories(config['visualize_scores'])

        model_evaluate_config = ModelEvaluateConfig(
            root_dir=config['root_dir'],
            scores_json=config['scores_json'],
            scores_csv=config['scores_csv'],
            visualize_scores=config(['visualize_scores'])
        )

        return model_evaluate_config