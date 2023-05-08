import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from segmentation_models import PSPNet
import segmentation_models as sm
from load_test_data import pad_image

output_dir = "/Users/stephanenjoki/Desktop/field_boundary_detection_challenge/field_boundary_detection/data/output"

# you can change number of epochs
def train_pspnet(X, y, input_shape=(288, 288, 24), num_classes=2, backbone='efficientnetb3', batch_size=2, epochs=50):

    # Pad the data and masks
    pad_height = 288 - X.shape[1]
    pad_width = 288 - X.shape[2]
    X_padded = pad_image(X, pad_height, pad_width)

    # One-hot encode the masks
    y_padded = pad_image(y, pad_height, pad_width)
    y_padded = tf.keras.utils.to_categorical(y_padded, num_classes)

    # Split the data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_padded, y_padded, test_size=0.2, random_state=42)

    # Load the pre-trained model with your desired input shape, number of classes, and softmax activation
    model = sm.PSPNet(input_shape=input_shape, classes=num_classes, backbone_name=backbone, encoder_weights=None, activation='softmax')

    # Define optimizer and compile the model with categorical cross-entropy loss and other metrics
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss=sm.losses.CategoricalCELoss(), metrics=[sm.metrics.IOUScore(), sm.metrics.FScore()])

    # Train the model
    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[tf.keras.callbacks.ModelCheckpoint("best_model.h5", save_best_only=True, monitor="val_f1-score", mode="max")],
    )

    # # Load the best model
    # best_model = tf.keras.models.load_model("best_model.h5", custom_objects={"CategoricalCELoss": sm.losses.CategoricalCELoss(), "iou_score": sm.metrics.IOUScore(), "f1-score": sm.metrics.FScore()})

    # return best_model

    # Save the best model
    best_model_path = os.path.join(output_dir, "pspnet_best_model.h5")
    best_model.save(best_model_path)

    # Load the best model
    best_model = tf.keras.models.load_model(best_model_path, custom_objects={"CategoricalCELoss": sm.losses.CategoricalCELoss(), "iou_score": sm.metrics.IOUScore(), "f1-score": sm.metrics.FScore()})

