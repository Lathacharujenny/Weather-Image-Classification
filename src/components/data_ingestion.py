import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.entity.config_entity import DataIngestionConfig
import zipfile


class DataIngestion:
    def __init__(self, config:DataIngestionConfig):
        self.config = config

    def data_loading(self):
        try:
            data_zip_path = self.config.data_zip_path
            logging.info(f'Getting zip data: {data_zip_path} for extracting')
            data_unzip_path = self.config.root_dir
            with zipfile.ZipFile(data_zip_path, 'r') as zip_file:
                zip_file.extractall(path=data_unzip_path)
            logging.info(f'Successfully extracted data from zip file')
        except Exception as e:
            logging.error(f'Error Occurred: {e}')
            raise CustomException(e,sys)

        
