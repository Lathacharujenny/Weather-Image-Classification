import os
from src.logger import logging
from src.exception import CustomException
import yaml
import sys
import pandas as pd
import pickle

def read_yaml(filepath):
    # Loading the yaml file
    try:
        with open(filepath) as file_obj:
            content = yaml.safe_load(file_obj)
            logging.info(f'Loaded yaml {file_obj} successfully')
            return content
    except Exception as e:
        logging.error(f'Error Occured: {e}')
        raise CustomException(e,sys)
    

def create_directories(filepath):
    # Creating the directories
    try:
        if not os.path.exists(filepath):
          os.makedirs(filepath, exist_ok=True)
        else:
            logging.info(f'{filepath} already exists')
    except Exception as e:
        logging.error(f'Error occured: {e}')
        raise CustomException(e,sys)
    
def load_csv_data(filepath):
    # To load the csv data
    try:
        if not os.path.exists(filepath):
            logging.info(f'Data filepath {filepath} does not exist')
        else:
             data = pd.read_csv(filepath)
             logging.info(f'Successfully loaded the data from {filepath}')
             return data
    except Exception as e:
        logging.error(f'Error Occurred: {e}')
        raise CustomException(e,sys)

def save_csv_data(data,filepath):
    # To save the csv data
    try:
        data.to_csv(filepath, index=False)
        logging.info(f'Successfully saved the data into {filepath}')
    except Exception as e:
        logging.error(f'Error Occurred: {e}')
        raise CustomException(e,sys)

def save_into_pickle_file(model, model_path):
    # Saving the pickle file
    try:
        logging.info(f'Saving the model into {model_path}')
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)
    except Exception as e:
        logging.error(f'Error Occurred: {e}')
        raise CustomException(e,sys)

def load_from_pickle_file(model_path):
    # Loading the pickle file
    try:
        logging.info(f'Loading the objest from {model_path}')
        with open(model_path, 'rb') as file:
            object = pickle.load(file)
        return(object)
    except Exception as e:
        logging.error(f'Error Occurred: {e}')
        raise CustomException(e,sys)