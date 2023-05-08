import os
import numpy as np
import rasterio as rio

dataset_id = 'nasa_rwanda_field_boundary_competition'
assets = ['labels']

# Train and test dataset
train_source_elements = f"{dataset_id}/{dataset_id}_source_train"
train_label_elements = f"{dataset_id}/{dataset_id}_labels_train"

def clean_string(s: str) -> str:
    """
    extract the tile id and timestamp from a source image folder
    e.g extract 'ID_YYYY_MM' from 'nasa_rwanda_field_boundary_competition_source_train_ID_YYYY_MM'
    """
    s = s.replace(f"{dataset_id}_source_", '').split('_')[1:]
    return '_'.join(s)

def normalize(array: np.ndarray) -> np.ndarray:
    """Normalize image to give a meaningful output."""
    array_min, array_max = array.min(), array.max()
    return (array - array_min) / (array_max - array_min)
    pass

def load_test_chips(test_source_items, timestamps, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
    test_tiles = [clean_string(s) for s in next(os.walk(test_source_items))[1]]

    test_tile_ids = set()
    for tile in test_tiles:
        test_tile_ids.add(tile.split('_')[0])

    X_test = np.empty((len(test_tile_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS * len(timestamps)), dtype=np.float32)
    
    i = 0
    loaded_tiles = []
    for tile_id in test_tile_ids:
        idx = 0
    for timestamp in timestamps:
        bd1 = rio.open(f"{test_source_items}/{dataset_id}_source_test_{tile_id}_{timestamp}/B01.tif")
        bd1_array = bd1.read(1)
        bd2 = rio.open(f"{test_source_items}/{dataset_id}_source_test_{tile_id}_{timestamp}/B02.tif")
        bd2_array = bd2.read(1)
        bd3 = rio.open(f"{test_source_items}/{dataset_id}_source_test_{tile_id}_{timestamp}/B03.tif")
        bd3_array = bd3.read(1)
        bd4 = rio.open(f"{test_source_items}/{dataset_id}_source_test_{tile_id}_{timestamp}/B04.tif")
        bd4_array = bd4.read(1)
        b01_norm = normalize(bd1_array)
        b02_norm = normalize(bd2_array)
        b03_norm = normalize(bd3_array)
        b04_norm = normalize(bd4_array)
        
        field = np.dstack((b04_norm, b03_norm, b02_norm, b01_norm))
        X_test[i][:,:,idx:idx+IMG_CHANNELS] = field
        idx+=IMG_CHANNELS
    loaded_tiles.append(str(tile_id).zfill(2)) #track order test tiles are loaded into X to make sure tile id matches 
    i+=1
    
    return X_test, loaded_tiles

def pad_image(array, pad_height, pad_width):
    return np.pad(array, ((0, 0), (pad_height // 2, pad_height // 2), (pad_width // 2, pad_width // 2), (0, 0)), mode='constant')

def load_test_data_padded(test_source_items, timestamps, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, pad_height, pad_width):
    X_test, loaded_tiles = load_test_chips(test_source_items, timestamps, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    X_test_padded = pad_image(X_test, pad_height, pad_width)
    return X_test_padded, loaded_tiles
