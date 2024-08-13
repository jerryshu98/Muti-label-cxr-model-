# README


## File Descriptions

### 1. `Model_training.py`

This script contains the core functionality for training a ResNet model on medical image data. It includes the following features:

- **GPU Setup**: Ensures that TensorFlow is configured to use the GPU, if available.
- **ResNet Model Definition**: Uses a pre-trained ResNet50 model from TensorFlow's Keras API as the base, with additional layers added for multi-label classification.
- **Data Loading**: Loads the training data from a pickle file and prepares it for input to the model.
- **Training Loop**: Trains the model in epochs, iterating through the training data in batches, and saving the model if the performance improves.

### 2. `TFrecord_data_process.py`

This script handles the processing of raw TFRecord data into a format suitable for training. Key functionalities include:

- **Data Labeling**: Converts raw data into a labeled format for multi-label classification.
- **Dataset Conversion**: Converts processed data into dictionary format, making it easier to manipulate and split.
- **Dataset Splitting**: Splits the data into training, validation, and test sets while ensuring that the split is subject-wise consistent.
- **Pickle Saving**: Saves the processed datasets into pickle files for easy loading during training.

### 3. `Testing_analysis.ipynb`

This Jupyter notebook contains code for analyzing the performance of the trained model. It includes:

- **Model Loading**: Loads the trained ResNet model from the saved Keras file.
- **Prediction and Evaluation**: Runs the model on the test dataset, computes accuracy, AUC, precision, and recall metrics.
- **Visualization**: Plots and visualizes the performance metrics and possibly the ROC and Precision-Recall curves.

## How to Run

1. **Prepare the Dataset**: Run `TFrecord_data_process.py` to process the raw TFRecord data and split it into training, validation, and test sets. The output will be stored in pickle files in the `./new_data/` directory.

   ```bash
   python data_processing.py
2. **Train the Model**: Run `Model_training.py'` to train the ResNet model using the processed dataset. The model will be saved in the ./models/resnetv0/ directory.
