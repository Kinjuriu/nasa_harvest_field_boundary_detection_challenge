# NASA Harvest Field Boundary Detection Challenge

**Note: This repository is a work in progress. More models and improvements will be added in the future.**

This repository contains code to train and evaluate deep learning models for the Field Boundary Detection Challenge using U-Net and PSPNet architectures. The goal is to detect field boundaries in satellite imagery to assist in precision agriculture.

This repository contains the code for field boundary detection using deep learning and semantic segmentation models. The project uses U-Net and PSPNet for image segmentation to identify field boundaries in satellite imagery.

# Installation
The project uses Poetry for dependency management. To set up the project, follow these steps:

1. Install Poetry if you haven't already:
pip install poetry

2. Clone this repository:
git clone https://github.com/your-username/field_boundary_detection.git
cd field_boundary_detection

3. Install the required dependencies:
poetry install

# Radiant MLHub API Key
To download the data from Radiant MLHub, you need to have an API key. If you don't have one, you can register for a free API key on the Radiant MLHub website.

After obtaining the API key, you should add it to your environment variables. To do this, open a terminal and run the following command:
export MLHUB_API_KEY="your_api_key_here"

Make sure to replace your_api_key_here with your actual API key.

Note: You will need to set the environment variable each time you open a new terminal session. To avoid this, consider adding the export command to your shell's configuration file (e.g., .bashrc or .zshrc).

#Data
- Input Data
The input data for this project consists of source imagery and corresponding field boundary labels. The data is organized as follows:

Source Imagery
Train: nasa_rwanda_field_boundary_competition_source_train

{chip_id}_{timestamp}
B01.tif
B02.tif
B03.tif
B04.tif
stac.json
Test: nasa_rwanda_field_boundary_competition_source_test

{chip_id}_{timestamp}
B01.tif
B02.tif
B03.tif
B04.tif
stac.json

Train Labels
nasa_rwanda_field_boundary_competition_labels_train
{chip_id}
stac.json
raster_labels.tif

# Output Data
The output data for this project consists of the trained models and their corresponding predictions.

- Trained Models
U-Net: unet_best_model.h5
PSPNet: pspnet_best_model.h5
- Predictions
U-Net predictions: unet_predictions
{chip_id}_unet_prediction.tif
PSPNet predictions: pspnet_predictions
{chip_id}_pspnet_prediction.tif

# Submission File
The submission file should be in CSV format and include the following columns:

Tile_row_column: The identifier for each tile, in the format "Tile{number}{row}{column}".
Label: The predicted label for each tile (0 for no boundary, 1 for boundary).

# Usage

To run the project, execute the following command:

poetry run python main.py

This command will train and evaluate the models using the data in the specified data directory.

# Running on Google Cloud Platform
Instructions on setting up the environment and running the code on a Google Cloud virtual machine will be added later.

# Contributing
If you would like to contribute to this project, please follow these steps:

Fork the repository.
Create a new branch with a descriptive name.
Make changes or add new features to the code.
Commit your changes and create a pull request.

# License
This project is released under the MIT License. Please refer to the LICENSE file for more information.
