#Evaluate the model on ETIS-LaribPolypDB test dataset

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.metrics import *
from glob import glob
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from model import build_model
from metrics import *
from prepare_dataset import prepare_dataset
from tensorflow.keras.losses import binary_crossentropy

if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    
    batch_size = 8#change it if required
    
    ##prepare test dataset
    test_path = "../Double_Unet/new_data/test"
    test_x = sorted(glob(os.path.join(test_path, "image", "*.jpg")))
    test_y = sorted(glob(os.path.join(test_path, "mask", "p*.jpg")))
    test_dataset = prepare_dataset(test_x, test_y, batch=batch_size)

    test_steps = (len(test_x)//batch_size)
    if len(test_x) % batch_size != 0:
        test_steps += 1

    model_path = "../Double_Unet/files_model/model.h5"

    model = build_model((288,384,3))
    metrics = [dice_coef, iou, Recall(), Precision()]
    
    model.compile(loss=binary_crossentropy, optimizer=Nadam(1e-4), metrics=metrics)
    model.load_weights(model_path)
    model.evaluate(test_dataset, steps=test_steps)