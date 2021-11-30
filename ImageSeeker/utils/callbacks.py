'''
@Author: Naveen Vinayak S
Date: 30 Nov 2021
Email: naveenvinayak.2195@gmail.com
'''
import os
from utils.config import configureModel
import time
import tensorflow as tf

config_model = configureModel()

def get_log_path(log_dir="Tensorboard/logs/fit"):
  fileName = time.strftime("log_%Y_%m_%d_%H_%M_%S")
  logs_path = os.path.join(log_dir, fileName)
  print(f"Saving logs at {logs_path}")
  return logs_path


def checkpoint():
    CKPT_path = "Checkpoint/Model_ckpt.h5"
    checkpointing_cb = tf.keras.callbacks.ModelCheckpoint(CKPT_path, save_best_only=True)
    return  checkpointing_cb

