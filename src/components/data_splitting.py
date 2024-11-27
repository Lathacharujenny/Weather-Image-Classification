import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.utils.common import load_csv_data, save_csv_data
from src.entity.config_entity import DataSplittingConfig
from sklearn.model_selection import train_test_split



class DataSplitting:
    def __init__(self, config:DataSplittingConfig):
        self.config = config

    def load_data(self):
        try:
            logging.info('Loading data for splitting into train and test')
            data = load_csv_data(self.config.data_path)
            return data
        except Exception as e:
            logging.error(f'Error Occurred: {e}')
            raise CustomException(e,sys)
        
    def data_splitting(self):
        try:
            data = self.load_data()
            train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

            logging.info(f'Saving the train and test data into {self.config.root_dir}')
            train_data_path = os.path.join(self.config.root_dir, 'train.csv')
            test_data_path = os.path.join(self.config.root_dir, 'test.csv')

            save_csv_data(train_data, train_data_path)
            save_csv_data(test_data, test_data_path)
        except Exception as e:
            logging.error(f'Error Occurred: {e}')
            raise CustomException(e,sys)
