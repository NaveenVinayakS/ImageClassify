'''
@Author: Naveen Vinayak S
Date: 30 Nov 2021
Email: naveenvinayak.2195@gmail.com
'''


from utils import model
from utils import data_manager as dm
from utils.config import configureModel
from utils import callbacks
import tensorflow as tf

config_model = configureModel()


def train():

    model_obj = model.load_pretrain_model()
    my_model = model_obj
    train_data, valid_data = dm.train_valid_generator()

    #callbacks
    log_dir = callbacks.get_log_path()
    tb_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    ckp = callbacks.checkpoint()

    call = [tb_cb, ckp]

    #Calculating steps_per_epoch & validation_steps
    steps_per_epoch = train_data.samples // train_data.batch_size
    validation_steps = valid_data.samples // valid_data.batch_size

    my_model.fit(
        train_data,
        validation_data=valid_data,
        epochs=config_model['EPOCHS'],
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=call
    )

    new_path = f"New_trained_model/{'new'+config_model['MODEL_NAME']+'.h5'}"
    my_model.save(new_path)
    print(f"Model saved at the following location : {new_path}")
