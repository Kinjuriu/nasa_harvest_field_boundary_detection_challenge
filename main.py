import numpy as np
import os
from data_loader import BoundaryDataset
from models import PSPNet, UNET
from predict import predict_unet, predict_pspnet
from tensorflow.keras.models import load_model
# from metrics import recall, f1
from UNET import train_unet, recall, f1
from load_test_data import pad_image
from load_test_data import load_test_chips, load_test_data_padded
from load_test_data import test_tile_ids

# Constants
UNET_MODEL_PATH = "path/to/unet/model/file.h5"
PSPNET_MODEL_PATH = "path/to/pspnet/model/file.h5"

dataset_id = 'nasa_rwanda_field_boundary_competition'
assets = ['labels']

timestamps = next(os.walk(f"./augmented_data"))[1] #Get all timestamps

# Image dimensions
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 4

# Create a BoundaryDataset instance
boundary_dataset = BoundaryDataset()

# Load test data
test_source_items = f"{dataset_id}/{dataset_id}_source_test"
X_test, loaded_tiles = load_test_chips(test_source_items, timestamps, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

source_items = f"{dataset_id}/{dataset_id}_source"
X, y, _ = load_test_chips(source_items, timestamps, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# Load the U-Net and PSPNet models
unet_model = load_model(UNET_MODEL_PATH, custom_objects={"recall": recall, "f1": f1})
pspnet_model = load_model(PSPNET_MODEL_PATH)

# Pad height and pad width for PSPNet model
pad_height = 288 - X.shape[1]
pad_width = 288 - X.shape[2]
X_padded = pad_image(X, pad_height, pad_width)

# Run predictions for both models
unet_predictions = predict_unet(unet_model, X_test, test_tile_ids, loaded_tiles)
pspnet_predictions = predict_pspnet(pspnet_model, X_test, test_tile_ids, loaded_tiles, pad_height, pad_width)

# Save predictions to CSV files
unet_predictions.to_csv("./unet_harvest_sample_submission.csv", index=False)
pspnet_predictions.to_csv("./pspnet_harvest_sample_submission.csv", index=False)
