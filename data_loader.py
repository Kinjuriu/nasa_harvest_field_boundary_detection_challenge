# Importing the needed libraries
import getpass
import os
import glob
import pickle
import random
from radiant_mlhub import Dataset

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio

import tensorflow as tf
import segmentation_models as sm

from pathlib import Path
from random import choice
from scipy.ndimage import gaussian_filter
from sklearn.model_selection import train_test_split

import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import *
from tensorflow.keras.preprocessing import image

from sklearn.model_selection import train_test_split
from typing import List, Any, Callable, Tuple

dataset_id = 'nasa_rwanda_field_boundary_competition'
assets = ['labels']

#Append your MLHUB_API_KEY after this cell is executed to download dataset
os.environ['MLHUB_API_KEY'] = getpass.getpass(prompt="MLHub API Key: ")
dataset = Dataset.fetch(dataset_id)
dataset.download(if_exists='overwrite')

input_dir = "/Users/stephanenjoki/Desktop/field_boundary_detection_challenge/field_boundary_detection/data/input"
# Use the input_dir variable in the functions responsible for downloading and saving the data.


#image snapshot dimensions
IMG_WIDTH = 256 
IMG_HEIGHT = 256 
IMG_CHANNELS = 4 #we have the rgba bands

class BoundaryDataset:
    """NASA Field Boundary Dataset. Read images."""
    
    def __init__(self, dataset_id: str, train_source_elements: str, train_label_elements: str):
        self.dataset_id = dataset_id
        self.train_source_elements = train_source_elements
        self.train_label_elements = train_label_elements
    
    # Add methods specific to the dataset, e.g., clean_string, normalize, loading data, etc.
    train_source_elements = f"{dataset_id}/{dataset_id}_source_train"
    train_label_elements = f"{dataset_id}/{dataset_id}_labels_train"

    def clean_string(s: str) -> str:
        """
        extract the tile id and timestamp from a source image folder
        e.g extract 'ID_YYYY_MM' from 'nasa_rwanda_field_boundary_competition_source_train_ID_YYYY_MM'
        """
        s = s.replace(f"{dataset_id}_source_", '').split('_')[1:]
        return '_'.join(s) 

    # Train data
    train_tiles = [clean_string(s) for s in next(os.walk(train_source_elements))[1]]

    # Normalize the data
    def normalize(array: np.ndarray) -> np.ndarray:
        """Normalize image to give a meaningful output."""
        array_min, array_max = array.min(), array.max()
        return (array - array_min) / (array_max - array_min) 

    # Loading the 4 bands of the image
    tile = random.choice(train_tiles)

    band1 = rio.open(f"{train_source_elements}/{dataset_id}_source_train_{tile}/B01.tif")
    band1_array = band1.read(1)
    band2 = rio.open(f"{train_source_elements}/{dataset_id}_source_train_{tile}/B02.tif")
    band2_array = band2.read(1)
    band3 = rio.open(f"{train_source_elements}/{dataset_id}_source_train_{tile}/B03.tif")
    band3_array = band3.read(1)
    band4 = rio.open(f"{train_source_elements}/{dataset_id}_source_train_{tile}/B04.tif")
    band4_array = band4.read(1)

    band1_norm = normalize(band1_array)
    band2_norm = normalize(band2_array)
    band3_norm = normalize(band3_array)
    band4_norm = normalize(band4_array)

    field = np.dstack((band4_norm, band3_norm, band2_norm, band1_norm))
    mask = rio.open(Path.cwd() / f"{train_label_elements}/{dataset_id}_labels_train_{tile.split('_')[0]}/raster_labels.tif").read(1)


class DataAugmentation:
    """Module to perform data augmentation."""

    @staticmethod
    def t_linear(field: np.ndarray, mask: np.ndarray, _: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Apply a linear (i.e., no) transformation and save."""
        return field, mask

    @staticmethod
    def t_rotation(field: np.ndarray, mask: np.ndarray, rot: int) -> Tuple[np.ndarray, np.ndarray]:
        """Rotate the data."""
        assert rot in range(0, 3 + 1)
        for _ in range(rot):
            field = np.rot90(field)
            mask = np.rot90(mask)
        return field, mask
    
    @staticmethod
    def t_flip(field: np.ndarray, mask: np.ndarray, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Flip the data."""
        assert idx in range(0, 2 + 1)
        if idx == 0:  # Diagonal
            field = np.rot90(np.fliplr(field))
            mask = np.rot90(np.fliplr(mask))
        if idx == 1:  # Horizontal
            field = np.flip(field, axis=0)
            mask = np.flip(mask, axis=0)
        if idx == 2:  # Vertical
            field = np.flip(field, axis=1)
            mask = np.flip(mask, axis=1)
        return field, mask
    
    @staticmethod
    def t_brightness(field: np.ndarray, mask: np.ndarray, value: float) -> Tuple[np.ndarray, np.ndarray]:
        """Adjust brightness of the data."""
        assert -0.5 <= value <= 0.5
        field_t = np.clip(field + value, 0, 1)
        return field_t, mask
    
    def generate(self, field: np.ndarray, mask: np.ndarray, write_folder: Path, prefix: str = "") -> None:
        """
        Generate data augmentations of the provided field and corresponding mask which includes:
         - Linear (no) transformation
         - Rotation
         - Horizontal or vertical flip
         - Gaussian filter (blur)
        :param field: Input array of the field to augment
        :param mask: Input array of the corresponding mask to augment
        :param write_folder: Folder (path) to write the results (augmentations) to
        :param prefix: Field-specific prefix used when writing the augmentation results
        """
        # Generate transformations
        f, m = [0,1,2,3], [0,1,2,3] #dummy data. will be replaced
        f[0],m[0] = t_linear(field, mask) #no augmentation 
        f[1],m[1] = t_rotation(field, mask, rot=1) #rotation
        f[2],m[2] = t_flip(field, mask, idx=0) #flipping
        f[3],m[3] = t_brightness(field, mask, brightness_value) #brightness
        for i in range(len(f)):        
            with open(write_folder +'/'+ f"fields/{str(prefix).zfill(2)}_{i}.pkl", 'wb') as handle:
                pickle.dump(f[i], handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(write_folder +'/'+ f"masks/{str(prefix).zfill(2)}_{i}.pkl", 'wb') as handle:
            pickle.dump(m[i], handle, protocol=pickle.HIGHEST_PROTOCOL)

    def apply_augmentations(self, train_tiles: List[str], train_source_elements: str, train_label_elements: str) -> None:
        """Apply the augmentations to the dataset."""
        #apply augmentation effects to training set
        for tile in train_tiles:
            bd1 = rio.open(f"{train_source_elements}/{dataset_id}_source_train_{tile}/B01.tif")
            bd1_array = bd1.read(1)
            bd2 = rio.open(f"{train_source_elements}/{dataset_id}_source_train_{tile}/B02.tif")
            bd2_array = bd2.read(1)
            bd3 = rio.open(f"{train_source_elements}/{dataset_id}_source_train_{tile}/B03.tif")
            bd3_array = bd3.read(1)
            bd4 = rio.open(f"{train_source_elements}/{dataset_id}_source_train_{tile}/B04.tif")
            bd4_array = bd4.read(1)
            b01_norm = normalize(bd1_array)
            b02_norm = normalize(bd2_array)
            b03_norm = normalize(bd3_array)
            b04_norm = normalize(bd4_array)

            ids_list  = tile.split('_') # XX_YYYY_MM where XX is the training file id and YYYY_MM is the timestamp
            tile_id   = ids_list[0]
            timestamp = f"{ids_list[1]}_{ids_list[2]}"

            field = np.dstack((b04_norm, b03_norm, b02_norm, b01_norm))
            mask  = rio.open(Path.cwd() / f"{train_label_elements}/{dataset_id}_labels_train_{tile_id}/raster_labels.tif").read(1) 

            #create a folder for the augmented images
            if not os.path.isdir(f"./augmented_data/{timestamp}"):
                os.makedirs(f"./augmented_data/{timestamp}")
            if not os.path.isdir(f"./augmented_data/{timestamp}/fields"):
                os.makedirs(f"./augmented_data/{timestamp}/fields")
            if not os.path.isdir(f"./augmented_data/{timestamp}/masks"):
                os.makedirs(f"./augmented_data/{timestamp}/masks")

            main( #applying augmentation effects
                field  = field,
                mask   = mask,
                prefix = tile_id,
                write_folder = f"./augmented_data/{timestamp}"
            ) #approximately 30 seconds

    timestamps = next(os.walk(f"./augmented_data"))[1] #Get all timestamps
    augmented_files = next(os.walk(f"./augmented_data/{timestamps[0]}/fields"))[2] #Get all augmented tile ids. can just use one timestamp
    X = np.empty((len(augmented_files), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS*len(timestamps)), dtype=np.float32) #time-series image
    y = np.empty((len(augmented_files), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8) #mask for each scene
    i = 0
    for file in augmented_files:
        idx = 0
        augmented_id = file.split('.pkl')[0] #id without .pkl extension
        temporal_fields = []
        for timestamp in timestamps:
            with open(f"./augmented_data/{timestamp}/fields/{augmented_id}.pkl", 'rb') as field:
                field = pickle.load(field) 
            X[i][:,:,idx:idx+IMG_CHANNELS] = field
            idx += IMG_CHANNELS
        with open(f"./augmented_data/{timestamp}/masks/{augmented_id}.pkl", 'rb') as mask:
            mask = pickle.load(mask)
        y[i] = mask.reshape(IMG_HEIGHT, IMG_WIDTH, 1)
        i+=1

class BoundaryDataset:
    # ... other methods and attributes ...
    test_source_elements = f"{dataset_id}/{dataset_id}_source_test"
    test_tiles = [clean_string(s) for s in next(os.walk(test_source_elements))[1]]

    @staticmethod
    def clean_string(s: str) -> str:
        return s.strip()

    @staticmethod
    def normalize(array: np.ndarray) -> np.ndarray:
        return (array - np.min(array)) / (np.max(array) - np.min(array))

    def load_test_chips(self, test_source_elements: str, timestamps: List[str], IMG_HEIGHT: int, IMG_WIDTH: int, IMG_CHANNELS: int) -> Tuple[np.ndarray, List[str]]:
        test_tiles = [self.clean_string(s) for s in next(os.walk(test_source_elements))[1]]

        test_tile_ids = set()
        for tile in test_tiles:
            test_tile_ids.add(tile.split('_')[0])

        X_test = np.empty((len(test_tile_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS * len(timestamps)), dtype=np.float32)
        i = 0
        loaded_tiles = []
        for tile_id in test_tile_ids:
            idx = 0
            for timestamp in timestamps:
                bd1 = rio.open(f"{test_source_elements}/{self.dataset_id}_source_test_{tile_id}_{timestamp}/B01.tif")
                bd1_array = bd1.read(1)
                bd2 = rio.open(f"{test_source_elements}/{self.dataset_id}_source_test_{tile_id}_{timestamp}/B02.tif")
                bd2_array = bd2.read(1)
                bd3 = rio.open(f"{test_source_elements}/{self.dataset_id}_source_test_{tile_id}_{timestamp}/B03.tif")
                bd3_array = bd3.read(1)
                bd4 = rio.open(f"{test_source_elements}/{self.dataset_id}_source_test_{tile_id}_{timestamp}/B04.tif")
                bd4_array = bd4.read(1)
                b01_norm = self.normalize(bd1_array)
                b02_norm = self.normalize(bd2_array)
                b03_norm = self.normalize(bd3_array)
                b04_norm = self.normalize(bd4_array)

                field = np.dstack((b04_norm, b03_norm, b02_norm, b01_norm))
                X_test[i][:, :, idx:idx + IMG_CHANNELS] = field
                idx += IMG_CHANNELS
            loaded_tiles.append(str(tile_id).zfill(2))  # track order test tiles are loaded into X to make sure tile id matches
            i += 1

        return X_test, loaded_tiles