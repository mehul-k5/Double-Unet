#Generate prediction images on ETIS-LaribPolypDB test dataset
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
    create_dir("../Double_Unet/results_ETIS/")

    ##prepare test dataset
    test_path = "../Double_Unet/new_data/test"
    test_x = sorted(glob(os.path.join(test_path, "image", "*.jpg")))
    test_y = sorted(glob(os.path.join(test_path, "mask", "p*.jpg")))
    
    model_path = "../Double_Unet/files_model/model.h5"

    model = build_model((288,384,3))
    metrics = [dice_coef, iou, Recall(), Precision()]

    model.compile(loss=binary_crossentropy, optimizer=Nadam(1e-4), metrics=metrics)
    model.load_weights(model_path)
    generate_results(model, test_x, test_y,name = 'ETIS')
  