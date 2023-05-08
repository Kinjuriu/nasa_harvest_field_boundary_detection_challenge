import numpy as np
import pandas as pd
from skimage.transform import resize
from load_test_data import pad_image

# Image snapshot dimensions
IMG_WIDTH = 256 
IMG_HEIGHT = 256 
IMG_CHANNELS = 4 #we have the rgba bands

def predict_unet(model, X_test, test_tile_ids, loaded_tiles):
    predictions_dictionary = {}
    for i in range(len(test_tile_ids)):
        model_pred = model.predict(np.expand_dims(X_test[i], 0))
        model_pred = model_pred[0]
        model_pred = (model_pred >= 0.5).astype(np.uint8)
        model_pred = model_pred.reshape(IMG_HEIGHT, IMG_WIDTH)
        predictions_dictionary.update([(str(loaded_tiles[i]), pd.DataFrame(model_pred))])

    dfs = []
    for key, value in predictions_dictionary.items():
        ftd = value.unstack().reset_index().rename(columns={'level_0': 'row', 'level_1': 'column', 0: 'label'})
        ftd['tile_row_column'] = f'Tile{key}_' + ftd['row'].astype(str) + '_' + ftd['column'].astype(str)
        ftd = ftd[['tile_row_column', 'label']]
        dfs.append(ftd)

    sub = pd.concat(dfs)
    
    return sub

def predict_pspnet(model, X_test, test_tile_ids, loaded_tiles, pad_height, pad_width):
    # Preprocess the test dataset: pad the images
    X_test_padded = pad_image(X_test, pad_height, pad_width)
    predictions_dictionary = {}
    # Preprocess the test dataset: pad the images
    X_test_padded = pad_image(X_test, pad_height, pad_width)

    predictions_dictionary = {}
    for i in range(len(test_tile_ids)):
        model_pred = model.predict(np.expand_dims(X_test_padded[i], 0))
        model_pred = model_pred[0]

        # Select the field boundary class (channel 1) and resize the prediction back to 256x256
        model_pred = resize(model_pred[:, :, 1], (256, 256), order=0, preserve_range=True)

        model_pred = (model_pred >= 0.5).astype(np.uint8)
        model_pred = model_pred.reshape(256, 256)
        predictions_dictionary.update([(str(loaded_tiles[i]), pd.DataFrame(model_pred))])

    dfs = []
    for key, value in predictions_dictionary.items():
        ftd = value.unstack().reset_index().rename(columns={'level_0': 'row', 'level_1': 'column', 0: 'label'})
        ftd['tile_row_column'] = f'Tile{key}_' + ftd['row'].astype(str) + '_' + ftd['column'].astype(str)
        ftd = ftd[['tile_row_column', 'label']]
        dfs.append(ftd)

    sub = pd.concat(dfs)
    return sub
