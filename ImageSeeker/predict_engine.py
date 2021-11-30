'''
@Author: Naveen Vinayak S
Date: 30 Nov 2021
Email: naveenvinayak.2195@gmail.com
'''

from utils import data_manager as dm
from utils.config import configureData
from utils.config import configureModel
import tensorflow as tf
import os
import numpy as np

config_data = configureData()
config_model = configureModel()

#Manage Image
image_list = os.listdir(config_data['PREDICTION_DATA_DIR'])


def predict():



    # load model
    model_path = f"New_trained_model/{'new' + config_model['MODEL_NAME'] + '.h5'}"
    model = tf.keras.models.load_model(model_path)
    for image in image_list:
        predict = dm.manage_input_data(os.path.join(config_data['PREDICTION_DATA_DIR'],image))
        result = model.predict(predict)
        results = np.argmax(result, axis=-1)
        print(f"Original image : {image}. Predicted as {results}")



