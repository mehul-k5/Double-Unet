#data_augmentation.py
!pip install albumentations
!pip install tiffile
import os
import random
import numpy as np
import cv2
from tqdm import tqdm
from glob import glob
import tiffile as tif
from sklearn.model_selection import train_test_split

from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,
    CenterCrop,
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomBrightnessContrast,
    RandomGamma,
    HueSaturationValue,
    RGBShift,
    RandomBrightness,
    RandomContrast,
    MotionBlur,
    MedianBlur,
    GaussianBlur,
    GaussNoise,
    ChannelShuffle,
    CoarseDropout
)

def data_augmentation(images, masks, save_path):
    """ Performs data augmentation. Takes list of paths of images and masks and save_path as input and generates
        augmented images and saves them"""
    crop_size = (288-32, 384-32)
    size = (384, 288)

    for image, mask in tqdm(zip(images, masks), total=len(images)):
        image_name = image.split("/")[-1].split(".")[0]
        mask_name = mask.split("/")[-1].split(".")[0]

        x, y = read_data(image, mask)
        try:
            h, w, c = x.shape
        except Exception as e:
            image = image[:-1]
            x, y = read_data(image, mask)
            h, w, c = x.shape

        ## Crop
        x_min = 0
        y_min = 0
        x_max = x_min + size[0]
        y_max = y_min + size[1]

        augment = Crop(p=1, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
        augmented = augment(image=x, mask=y)
        x1 = augmented['image']
        y1 = augmented['mask']
        
        ## Center Crop
        augment = CenterCrop(p=1, height=crop_size[0], width=crop_size[1])
        augmented = augment(image=x, mask=y)
        x2 = augmented['image']
        y2 = augmented['mask']

        ## Transpose
        augment = Transpose(p=1)
        augmented = augment(image=x, mask=y)
        x3 = augmented['image']
        y3 = augmented['mask']
       
        ## Random Rotate 90 degree
        augment = RandomRotate90(p=1)
        augmented = augment(image=x, mask=y)
        x4 = augmented['image']
        y4 = augmented['mask']

        ## ElasticTransform
        augment = ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)
        augmented = augment(image=x, mask=y)
        x5 = augmented['image']
        y5 = augmented['mask']

        ## Optical Distortion
        augment = OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)
        augmented = augment(image=x, mask=y)
        x6 = augmented['image']
        y6 = augmented['mask']

        ## Grid Distortion
        augment = GridDistortion(p=1)
        augmented = augment(image=x, mask=y)
        x7 = augmented['image']
        y7 = augmented['mask']

        ## Grayscale
        x8 = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
        y8 = y

        ## Grayscale Vertical Flip
        augment = VerticalFlip(p=1)
        augmented = augment(image=x8, mask=y8)
        x9 = augmented['image']
        y9 = augmented['mask']

        ## Grayscale Horizontal Flip
        augment = HorizontalFlip(p=1)
        augmented = augment(image=x8, mask=y8)
        x10 = augmented['image']
        y10 = augmented['mask']

        ## Grayscale Center Crop
        augment = CenterCrop(p=1, height=crop_size[0], width=crop_size[1])
        augmented = augment(image=x8, mask=y8)
        x11 = augmented['image']
        y11 = augmented['mask']

        ## Vertical Flip
        augment = VerticalFlip(p=1)
        augmented = augment(image=x, mask=y)
        x12 = augmented['image']
        y12 = augmented['mask']

        ## Horizontal Flip
        augment = HorizontalFlip(p=1)
        augmented = augment(image=x, mask=y)
        x13 = augmented['image']
        y13 = augmented['mask']

        ##Randomly change gamma
        augment = RandomGamma(p=1)
        augmented = augment(image=x, mask=y)
        x14 = augmented['image']
        y14 = augmented['mask']

        ##Randomly change hue, saturation and value
        augment = HueSaturationValue(p=1)
        augmented = augment(image=x, mask=y)
        x15 = augmented['image']
        y15 = augmented['mask']

        ##Randomly change contrast
        augment = RandomContrast(p=1)
        augmented = augment(image=x, mask=y)
        x16 = augmented['image']
        y16 = augmented['mask']

        ##Randomly change brightness
        augment = RandomBrightness(p=1)
        augmented = augment(image=x, mask=y)
        x17 = augmented['image']
        y17 = augmented['mask']

        ##Randomly change brightness and contrast
        augment = RandomBrightnessContrast(p=1)
        augmented = augment(image=x, mask=y)
        x18 = augmented['image']
        y18 = augmented['mask']

        ##Randomly shift values for each channel of the input RGB image.
        augment = RGBShift(p=1)
        augmented = augment(image=x, mask=y)
        x19 = augmented['image']
        y19 = augmented['mask']

        ##CoarseDropout of the rectangular regions in the image.
        augment = CoarseDropout(p=1, max_holes=8, max_height=32, max_width=32)
        augmented = augment(image=x, mask=y)
        x20 = augmented['image']
        y20 = augmented['mask']

        ##Randomly rearrange channels of the input RGB image
        augment = ChannelShuffle(p=1)
        augmented = augment(image=x, mask=y)
        x21 = augmented['image']
        y21 = augmented['mask']

        ##Blur using a median filter with a random aperture linear size
        augment = MedianBlur(p=1, blur_limit=9)
        augmented = augment(image=x, mask=y)
        x22 = augmented['image']
        y22 = augmented['mask']

        ##Apply motion blur
        augment = MotionBlur(p=1, blur_limit=7)
        augmented = augment(image=x, mask=y)
        x23 = augmented['image']
        y23 = augmented['mask']

        ##Blur the input image using a Gaussian filter with a random kernel size
        augment = GaussianBlur(p=1, blur_limit=9)
        augmented = augment(image=x, mask=y)
        x24 = augmented['image']
        y24 = augmented['mask']

        ##Apply gaussian noise to the input image
        augment = GaussNoise(p=1)
        augmented = augment(image=x, mask=y)
        x25 = augmented['image']
        y25 = augmented['mask']

        images = [
            x, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10,
            x11, x12, x13, x14, x15, x16, x17, x18, x19, x20,
            x21, x22, x23, x24, x25
        ]
        masks  = [
            y, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10,
            y11, y12, y13, y14, y15, y16, y17, y18, y19, y20,
            y21, y22, y23, y24, y25
        ]

        idx = 0
        for i, m in zip(images, masks):
            i = cv2.resize(i, size)
            m = cv2.resize(m, size)

            temp_image_name = f"{image_name}_{idx}.jpg"
            temp_mask_name  = f"{mask_name}_{idx}.jpg"

            img_path = os.path.join(save_path, "image/", temp_image_name)
            mask_path  = os.path.join(save_path, "mask/", temp_mask_name)

            cv2.imwrite(img_path, i)
            cv2.imwrite(mask_path, m)

            idx += 1

def create_dir(path):
    """ Create a directory. """
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print(f"Error: creating directory with name {path}")
        
def read_data(x, y):
    """ Read the image and mask from the given path. """
    image = tif.imread(x)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    mask = tif.imread(y)
    return image, mask

def get_data(train_path,test_path):
    """ Get train and test data paths. """
    train_x = glob(os.path.join(train_path, "Original/*"))
    train_y = glob(os.path.join(train_path, "Ground Truth/*"))

    test_x = glob(os.path.join(test_path, "Original/*"))
    test_y = glob(os.path.join(test_path, "Ground Truth/p*"))

    return (train_x, train_y), (test_x, test_y)

def main():
    np.random.seed(42)

    #change the paths
    train_path = "../Double_Unet/CVC-ClinicDB"
    test_path = "../Double_Unet/ETIS-LaribPolypDB"
    (train_x, train_y), (test_x, test_y) = get_data(train_path,test_path)

    create_dir("../Double_Unet/new_data/train/image/")
    create_dir("../Double_Unet/new_data/train/mask/")

    create_dir("../Double_Unet/new_data/test/image/")
    create_dir("../Double_Unet/new_data/test/mask/")

    data_augmentation(train_x, train_y, "../Double_Unet/new_data/train/")
    
    data_augmentation(test_x, test_y, "../Double_Unet/new_data/test/")

if __name__ == "__main__":
    main()