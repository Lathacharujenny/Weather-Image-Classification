import os
import sys
import json
from src.logger import logging
from src.exception import CustomException
from src.entity.config_entity import ModelTrainingConfig
from src.utils.common import load_csv_data, save_into_pickle_file
from src.functions.model_structure import model_structure
from src.functions.data_generator import data_generator
from src.functions.visualize_results import visualize_results
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess_input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input
from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.applications.resnet import preprocess_input as resnet152v2_preprocess_input
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_preprocess_input
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess_input
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess_input
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as inceptionresnetv2_preprocess_input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications.resnet import preprocess_input as resnet101_preprocess_input
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.applications.densenet import preprocess_input as densenet101_preprocess_input

class ModelTraining:
    def __init__(self, config:ModelTrainingConfig):
        self.config = config
    
    def loading_data(self):
        try:
            logging.info('Loading train and test data for model training..............')
            train_data = load_csv_data(self.config.train_data_path)
            test_data = load_csv_data(self.config.test_data_path)
            return train_data, test_data
        except Exception as e:
            logging.error(f'Error Occurred: {e}')
            raise CustomException(e,sys)
        
    def efficientnet_model(self):
        try:
            train_data, test_data = self.loading_data()

            logging.info('Transforming the train and test data into ImageDataGenerator for EfficientNetB7 model')
            train_gen, valid_gen, test_gen = data_generator(efficientnet_preprocess_input, train_data, test_data, 600)
      
            labels = train_gen.class_indices
            self.labels_dict = {v:k for k,v in labels.items()}
            labels_path = os.path.join(self.config.labels_path,'labels.json')
            logging.info(f'Saving the labels with their indices into {labels_path}')
            with open(labels_path, 'w') as file:
                json.dump(self.labels_dict, file)
            
            logging.info('Started training the EfficientNetB7 model......................')
            enet_model, callback = model_structure(EfficientNetB7, 600)
            history = enet_model.fit(train_gen, validation_data=valid_gen, epochs=100, callbacks=callback)
            
            model_path = os.path.join(self.config.models, 'Efficient_model.h5')
            logging.info(f'Dumping the trained model into {model_path}')
            save_into_pickle_file(enet_model, model_path)

            logging.info(f'Visualizing the results: {self.config.efficientnet_model}')
            visualize_results(history, self.labels_dict, test_gen, test_data, enet_model, self.config.efficientnet_model)

            logging.info('Ended training the EfficientNetB7 model......................')

            return test_gen, enet_model

        except Exception as e:
            logging.error(f'Error Occurred: {e}')
            raise CustomException(e,sys)
        
    def resnet50_model(self):
        try:
            train_data, test_data = self.loading_data()

            logging.info('Transforming the train and test data into ImageDataGenerator for ResNet50 model')
            train_gen, valid_gen, test_gen = data_generator(resnet50_preprocess_input, train_data, test_data, 224)

            logging.info('Started training the ResNet50 model......................')
            resnet50_model, callback = model_structure(ResNet50, 224)
            history = resnet50_model.fit(train_gen, validation_data=valid_gen, epochs=100, callbacks=callback)
            
            model_path = os.path.join(self.config.models, 'ResNet50.h5')
            logging.info(f'Dumping the trained model into {model_path}')
            save_into_pickle_file(resnet50_model, model_path)

            logging.info(f'Visualizing the results: {self.config.resnet50_model}')
            visualize_results(history, self.labels_dict, test_gen, test_data, resnet50_model, self.config.resnet50_model)

            logging.info('Ended training the ResNet50 model......................')

            return test_gen, resnet50_model

        except Exception as e:
            logging.error(f'Error Occurred: {e}')
            raise CustomException(e,sys)
        
    def resnet152v2_model(self):
        try:
            train_data, test_data = self.loading_data()

            logging.info('Transforming the train and test data into ImageDataGenerator for ResNet152V2 model')
            train_gen, valid_gen, test_gen = data_generator(resnet152v2_preprocess_input, train_data, test_data, 256)

            logging.info('Started training the ResNet152V2 model......................')
            resnet152v2_model, callback = model_structure(ResNet152V2, 256)
            history = resnet152v2_model.fit(train_gen, validation_data=valid_gen, epochs=100, callbacks=callback)
            
            model_path = os.path.join(self.config.models, 'ResNet152V2.h5')
            logging.info(f'Dumping the trained model into {model_path}')
            save_into_pickle_file(resnet152v2_model, model_path)

            logging.info(f'Visualizing the results: {self.config.resnet152v2_model}')
            visualize_results(history, self.labels_dict, test_gen, test_data, resnet152v2_model, self.config.resnet152v2_model)

            logging.info('Ended training the ResNet152V2 model......................')

            return test_gen, resnet152v2_model

        except Exception as e:
            logging.error(f'Error Occurred: {e}')
            raise CustomException(e,sys)
        
    def mobilenet_model(self):
        try:
            train_data, test_data = self.loading_data()

            logging.info('Transforming the train and test data into ImageDataGenerator for MobileNet model')
            train_gen, valid_gen, test_gen = data_generator(mobilenet_preprocess_input, train_data, test_data, 224)

            logging.info('Started training the MobileNet model......................')
            mobilenet_model, callback = model_structure (MobileNet, 224)
            history = mobilenet_model.fit(train_gen, validation_data=valid_gen, epochs=100, callbacks=callback)
            
            model_path = os.path.join(self.config.models,  'MobileNet.h5')
            logging.info(f'Dumping the trained model into {model_path}')
            save_into_pickle_file(mobilenet_model, model_path)

            logging.info(f'Visualizing the results: {self.config.mobilenet_model}')
            visualize_results(history, self.labels_dict, test_gen, test_data, mobilenet_model, self.config.mobilenet_model)

            logging.info('Ended training the MobileNet model......................')

            return test_gen, mobilenet_model

        except Exception as e:
            logging.error(f'Error Occurred: {e}')
            raise CustomException(e,sys)
        
    def vgg19_model(self):
        try:
            train_data, test_data = self.loading_data()

            logging.info('Transforming the train and test data into ImageDataGenerator for VGG19 model')
            train_gen, valid_gen, test_gen = data_generator(vgg19_preprocess_input, train_data, test_data, 224)

            logging.info('Started training the VGG19 model......................')
            vgg19_model, callback = model_structure (VGG19, 224)
            history = vgg19_model.fit(train_gen, validation_data=valid_gen, epochs=100, callbacks=callback)
            
            model_path = os.path.join(self.config.models,  'VGG19.h5')
            logging.info(f'Dumping the trained model into {model_path}')
            save_into_pickle_file(vgg19_model, model_path)

            logging.info(f'Visualizing the results: {self.config.vgg19_model}')
            visualize_results(history, self.labels_dict, test_gen, test_data, vgg19_model, self.config.vgg19_model)

            logging.info('Ended training the VGG19 model......................')

            return test_gen, vgg19_model

        except Exception as e:
            logging.error(f'Error Occurred: {e}')
            raise CustomException(e,sys)
        
    def xception_model(self):
        try:
            train_data, test_data = self.loading_data()

            logging.info('Transforming the train and test data into ImageDataGenerator for Xception model')
            train_gen, valid_gen, test_gen = data_generator(xception_preprocess_input, train_data, test_data, 224)

            logging.info('Started training the Xception model......................')
            xception_model, callback = model_structure (Xception, 224)
            history = xception_model.fit(train_gen, validation_data=valid_gen, epochs=100, callbacks=callback)
            
            model_path = os.path.join(self.config.models,  'Xception.h5')
            logging.info(f'Dumping the trained model into {model_path}')
            save_into_pickle_file(xception_model, model_path)

            logging.info(f'Visualizing the results: {self.config.xception_model}')
            visualize_results(history, self.labels_dict, test_gen, test_data, xception_model, self.config.xception_model)

            logging.info('Ended training the Xception model......................')

            return test_gen, xception_model

        except Exception as e:
            logging.error(f'Error Occurred: {e}')
            raise CustomException(e,sys)


    def inceptionresnetv2_model(self):
        try:
            train_data, test_data = self.loading_data()

            logging.info('Transforming the train and test data into ImageDataGenerator for InceptionResNetV2 model')
            train_gen, valid_gen, test_gen = data_generator(inceptionresnetv2_preprocess_input, train_data, test_data, 299)

            logging.info('Started training the InceptionResNetV2 model......................')
            inceptionresnetv2_model, callback = model_structure (InceptionResNetV2, 299)
            history = inceptionresnetv2_model.fit(train_gen, validation_data=valid_gen, epochs=100, callbacks=callback)
            
            model_path = os.path.join(self.config.models,  'InceptionResNetV2.h5')
            logging.info(f'Dumping the trained model into {model_path}')
            save_into_pickle_file(inceptionresnetv2_model, model_path)

            logging.info(f'Visualizing the results: {self.config.inceptionresnetv2_model}')
            visualize_results(history, self.labels_dict, test_gen, test_data, inceptionresnetv2_model, self.config.inceptionresnetv2_model)

            logging.info('Ended training the InceptionResNetV2 model......................')

            return test_gen, inceptionresnetv2_model

        except Exception as e:
            logging.error(f'Error Occurred: {e}')
            raise CustomException(e,sys)
        
    def vgg16_model(self):
        try:
            train_data, test_data = self.loading_data()

            logging.info('Transforming the train and test data into ImageDataGenerator for VGG16 model')
            train_gen, valid_gen, test_gen = data_generator(vgg16_preprocess_input, train_data, test_data, 224)

            logging.info('Started training the VGG16 model......................')
            vgg16_model, callback = model_structure (VGG16, 224)
            history = vgg16_model.fit(train_gen, validation_data=valid_gen, epochs=100, callbacks=callback)
            
            model_path = os.path.join(self.config.models,  'VGG16.h5')
            logging.info(f'Dumping the trained model into {model_path}')
            save_into_pickle_file(vgg16_model, model_path)

            logging.info(f'Visualizing the results: {self.config.vgg16_model}')
            visualize_results(history, self.labels_dict, test_gen, test_data, vgg16_model, self.config.vgg16_model)

            logging.info('Ended training the VGG16 model......................')

            return test_gen, vgg16_model

        except Exception as e:
            logging.error(f'Error Occurred: {e}')
            raise CustomException(e,sys)
    
    def resnet101_model(self):
        try:
            train_data, test_data = self.loading_data()

            logging.info('Transforming the train and test data into ImageDataGenerator for ResNet101 model')
            train_gen, valid_gen, test_gen = data_generator(resnet101_preprocess_input, train_data, test_data, 224)

            logging.info('Started training the ResNet101 model......................')
            resnet101_model, callback = model_structure (ResNet101, 224)
            history = resnet101_model.fit(train_gen, validation_data=valid_gen, epochs=100, callbacks=callback)
            
            model_path = os.path.join(self.config.models,  'ResNet101.h5')
            logging.info(f'Dumping the trained model into {model_path}')
            save_into_pickle_file(resnet101_model, model_path)

            logging.info(f'Visualizing the results: {self.config.resnet101_model}')
            visualize_results(history, self.labels_dict, test_gen, test_data, resnet101_model, self.config.resnet101_model)

            logging.info('Ended training the ResNet101 model......................')

            return test_gen, resnet101_model

        except Exception as e:
            logging.error(f'Error Occurred: {e}')
            raise CustomException(e,sys)

        
    def densenet201_model(self):
        try:
            train_data, test_data = self.loading_data()

            logging.info('Transforming the train and test data into ImageDataGenerator for DenseNet201 model')
            train_gen, valid_gen, test_gen = data_generator(resnet101_preprocess_input, train_data, test_data, 224)

            logging.info('Started training the DenseNet201 model......................')
            densenet201_model, callback = model_structure (DenseNet201, 224)
            history = densenet201_model.fit(train_gen, validation_data=valid_gen, epochs=100, callbacks=callback)
            
            model_path = os.path.join(self.config.models,  'DenseNet201.h5')
            logging.info(f'Dumping the trained model into {model_path}')
            save_into_pickle_file(densenet201_model, model_path)

            logging.info(f'Visualizing the results: {self.config.densenet201_model}')
            visualize_results(history, self.labels_dict, test_gen, test_data, densenet201_model, self.config.densenet201_model)

            logging.info('Ended training the DenseNet201 model......................')

            return test_gen, densenet201_model

        except Exception as e:
            logging.error(f'Error Occurred: {e}')
            raise CustomException(e,sys)
        
    


