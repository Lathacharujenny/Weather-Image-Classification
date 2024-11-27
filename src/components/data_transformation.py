import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.entity.config_entity import DataTransformationConfig
from src.utils.common import save_csv_data
import glob
import pandas as pd

class DataTransformation:
    def __init__(self, config:DataTransformationConfig):
        self.config = config

    def load_data(self):
        try:
            logging.info('Loading the data for transformation.................')
            data_path=self.config.data_path
            path_imgs = list(glob.glob(data_path+'/**/*.jpg'))
            labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], path_imgs))
            logging.info(f'Labels: {len(labels)}')

            images = pd.Series(path_imgs, name='Images').astype(str)
            labels = pd.Series(labels, name='Labels')
            data = pd.concat([images, labels], axis=1).reset_index(drop=True)
            logging.info(f'Converted data: \n {data.head()}')

            logging.info('Extracing only few data since the data is very large')
            data = data.groupby('Labels').apply(lambda x: x.sample(min(len(x), 400), random_state=42))
            data = data.sample(frac=1).reset_index(drop=True)
            logging.info(f'Taking data only 400 per each label: {data["Labels"].value_counts()}')

            logging.info(f'Saving the data into csv')
            data_csv_path = os.path.join(self.config.root_dir, 'img_data.csv')
            save_csv_data(data, data_csv_path)
        except Exception as e:
            logging.error(f'Error Occurred: {e}')
            raise CustomException(e,sys)
