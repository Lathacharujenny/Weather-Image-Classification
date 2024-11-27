import os
import sys
import glob
import json
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.entity.config_entity import ModelEvaluateConfig
from src.config.configuration import ConfigurationManager
from src.components.model_training import ModelTraining
from src.functions.evaluate_test_data import evaluate_test_data
from src.utils.common import save_csv_data
import matplotlib.pyplot as plt



class ModelEvaluate:
    def __init__(self, config:ModelEvaluateConfig):
        self.config = config

    def load_test_gen_models(self):
        try:
            logging.info('Getting test_gen and models for evaluating')
            config = ConfigurationManager()
            model_training_config = config.get_model_training_config()
            model_training = ModelTraining(config=model_training_config)
            self.efficient_test_gen,self.efficient_model = model_training.efficientnet_model()
            self.resnet50_test_gen, self.resnet50_model = model_training.resnet50_model()
            self.resnet152v2_test_gen, self.resnet152v2_model=model_training.resnet152v2_model()
            self.mobilenet_test_gen, self.mobilenet_model = model_training.mobilenet_model()
            self.vgg19_test_gen, self.vgg19_model = model_training.vgg19_model()
            self.vgg16_test_gen, self.vgg16_model = model_training.vgg16_model()
            self.inception_test_gen, self.inception_model = model_training.inceptionresnetv2_model()
            self.xception_test_gen, self.xception_model= model_training.xception_model()
            self.resnet101_test_gen, self.resnet101_model = model_training.resnet101_model()
            self.densenet_test_gen, self.densenet_model= model_training.densenet201_model()
        except Exception as e:
            logging.error(f'Error Occurred: {e}')
            raise CustomException(e,sys)
        
    def evaluate_models(self):
        try:
            logging.info('Evaluating test data')
            evaluate_test_data(self.efficient_test_gen, self.efficient_model, self.config.scores_json)
            evaluate_test_data(self.resnet50_test_gen, self.resnet50_model,self.config.scores_json)
            evaluate_test_data(self.resnet152v2_test_gen, self.resnet152v2_model, self.config.scores_json)
            evaluate_test_data( self.mobilenet_test_gen, self.mobilenet_model, self.config.scores_json)
            evaluate_test_data(self.vgg19_test_gen, self.vgg19_model, self.config.scores_json)
            evaluate_test_data(self.vgg16_test_gen, self.vgg16_model, self.config.scores_json)
            evaluate_test_data(self.inception_test_gen, self.inception_model, self.config.scores_json)
            evaluate_test_data(self.xception_test_gen, self.xception_model, self.config.scores_json)
            evaluate_test_data(self.resnet101_test_gen, self.resnet101_model, self.config.scores_json)
            evaluate_test_data(self.densenet_test_gen, self.densenet_model, self.config.scores_json)
        except Exception as e:
            logging.error(f'Error Occurred: {e}')
            raise CustomException(e,sys)
        
    def convert_results_csv(self):
        try:
            path = self.config.scores_json
            json_path = glob.glob(path+'/*.json')
            scores=[]

            for path in json_path:
                #print(path)
                root_path,model_name = os.path.split(path)
                model_name = model_name.replace('.json', "")
                with open(path, 'r') as file:
                    json_file = json.load(file)
                
                results = {
                    'Model': model_name,
                    'Test_Loss': json_file['Test Loss'],
                    'Test_Accuracy': json_file['Test Accuracy']
                }

                scores.append(results)

            self.scores_df = pd.DataFrame(scores, columns=['Model', 'Test_Loss','Test_Accuracy'])
            csv_path = os.path.join(self.config.scores_csv, 'scores.csv')
            save_csv_data(self.scores_df, csv_path)
        except Exception as e:
            logging.error(f'Error Occurred: {e}')
            raise CustomException(e,sys)
        
    def visualize_the_scores(self):
        try:
            logging.info('Visualizing the test scores')
            plt.figure(figsize=(20,5))
            plt.bar([0,1,2,3,4,5,6,7,8,9], self.scores_df['Test_Accuracy'], width=0.4, label='Test_Accuracy')
            plt.bar([0.4,1.4,2.4,3.4,4.4,5.4,6.4,7.4,8.4,9.4], self.scores_df['Test_Loss'], width=0.4, label='Test_Loss')
            plt.xticks([0,1,2,3,4,5,6,7,8,9], self.scores_df['Model'])
            plt.legend()

            path = os.path.join(self.config.visualize_scores, 'test_scores.png')
            plt.savefig(path)
            plt.close()
        except Exception as e:
            logging.error(f'Error Occurred: {e}')
            raise CustomException(e,sys)
        

