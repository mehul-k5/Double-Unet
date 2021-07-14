#Generate prediction images on CVC-ClinicDB test dataset
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
from tensorflow.keras.losses import binary_crossentropy
from generate_results import generate_results

def create_dir(path):
    """ Create a directory. """
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print(f"Error: creating directory with name {path}")
        
def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y
    
if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    create_dir("../Double_Unet/results_CVC/")
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
    
    model_path = "../Double_Unet/files_model/model.h5"

    model = build_model((288,384,3))
    metrics = [dice_coef, iou, Recall(), Precision()]

    model.compile(loss=binary_crossentropy, optimizer=Nadam(1e-4), metrics=metrics)
    model.load_weights(model_path)
    generate_results(model, test_x, test_y,name = 'CVC')
  