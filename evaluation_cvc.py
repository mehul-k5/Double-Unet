#Evaluate the model on CVC-ClinicDB test dataset
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

def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y

if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)

    batch_size = 8 #change it if required
    train_path = "../Double_Unet/new_data/train"

    ## Training set paths
    train_x = sorted(glob(os.path.join(train_path, "image", "*.jpg")))
    train_y = sorted(glob(os.path.join(train_path, "mask", "*.jpg")))

    ## Shuffling
    train_x, train_y = shuffling(train_x, train_y)

    ## Path Split
    ## Split CVC-ClinicDB - 80:10:10 (train:test:validation)
    train_x,valid_test_x = train_test_split(train_x, test_size=0.2, random_state=42)
    train_y,valid_test_y = train_test_split(train_y, test_size=0.2, random_state=42)
    valid_x,test_x = train_test_split(valid_test_x, test_size=0.5, random_state=42)
    valid_y,test_y = train_test_split(valid_test_y, test_size=0.5, random_state=42)
    
    ##Prepare test dataset
    test_dataset = prepare_dataset(test_x,test_y,batch_size)

    test_steps = (len(test_x)//batch_size)
    if len(test_x) % batch_size != 0:
        test_steps += 1

    model_path = "../Double_Unet/files_model/model.h5"

    model = build_model((288,384,3))
    metrics = [dice_coef, iou, Recall(), Precision()]

    model.compile(loss=binary_crossentropy, optimizer=Nadam(1e-4), metrics=metrics)
    model.load_weights(model_path)
    model.evaluate(test_dataset, steps=test_steps)