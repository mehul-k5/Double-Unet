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
    create_dir("../Double_Unet/files_model")

    train_path = "../Double_Unet/new_data/train"
    
    ## Training on CVC-ClinicDB
    
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
    
    ##Prepare Dataset
    train_dataset = prepare_dataset(train_x,train_y,batch_size)
    valid_dataset = prepare_dataset(valid_x,valid_y,batch_size)

    model_path = "../Double_Unet/files_model/model.h5"
    batch_size = 8
    epochs = 35
    lr = 1e-4
    shape = (288,384,3)


    model = build_model(shape)
    metrics = [dice_coef, iou, Recall(), Precision()]

    model.compile(loss=binary_crossentropy, optimizer=Nadam(lr), metrics=metrics)
    
    ##Load weights if continuing training
    #model.load_weights(model_path)
    
    callbacks = [
    ModelCheckpoint(model_path),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20, min_lr=1e-5),
    CSVLogger("../Double_Unet/files_model/data.csv",append=True),
    TensorBoard(),
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False)
    ]

    train_steps = (len(train_x)//batch_size)
    valid_steps = (len(valid_x)//batch_size)

    if len(train_x) % batch_size != 0:
    train_steps += 1

    if len(valid_x) % batch_size != 0:
    valid_steps += 1

    model.fit(train_dataset,
        epochs=epochs,
        validation_data=valid_dataset,
        steps_per_epoch=train_steps,
        validation_steps=valid_steps,
        callbacks=callbacks,
        shuffle=False)
   