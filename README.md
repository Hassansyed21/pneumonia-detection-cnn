# pneumonia-detection-cnn
Pneumonia detection from X-rays using a CNN built with Keras/TensorFlow and trained on the Chest X-Ray (Pneumonia) dataset. Includes data augmentation, model training, evaluation, and a saved model file.

Key Features
Data Preprocessing:Uses ImageDataGenerator for efficient loading and augmentation.
Data Augmentation:Applies rotation, horizontal flipping, and zooming to the training set to prevent overfitting.
Model Design:A feedforward CNN with three Conv2D/MaxPooling2D blocks followed by Dense layers.
Training & Evaluation:Includes training/validation accuracy and loss plots, a confusion matrix, and a detailed classification report.
Model Checkpoint:Saves the trained model as a keras file for inference.

Project Steps

1. Setup and Requirements
Installs tensorflow, keras, and matplotlib.
2. Data Processing
Mounts Google Drive to access the dataset.
Defines paths for train, val, and test directories.
Sets up ImageDataGenerator for training (with augmentation) and testing (with only rescaling).
Loads data using flow_from_directory().
3. Model Training
Defines a Sequential CNN model.
Compiles the model using adam optimizer and binary_crossentropy loss.
Trains the model for 10 epochs.
4. Evaluation & Visualization
 Evaluates the model on the test set.
 Plots "Model Accuracy" and "Model Loss" graphs.
Generates and plots a Confusion Matrix.
Prints a Classification Report (precision, recall, f1-score).
5. Model Saving & Loading
 Saves the final trained model as my_chest_xray_model.keras.

Setup Instructions

1. Clone the Repository
   git clone [https://github.com/Hassansyed21/pneumonia-detection-cnn.git](https://github.com/Hassansyed21/pneumonia-detection-cnn.git)
    cd pneumonia-detection-cnn
2. Install Dependencies
   pip install -r requirements.txt
3. Run the Notebook
   Open pneumonia_detection_model.ipynb in Jupyter or Google Colab.
   Important: This project requires the pre-trained model file my_chest_xray_model.keras. Download it from the link in the README.md and place it in your project folder.
   You must also upload the chest_xray dataset to your environment.

Follow the code cells for training and evaluation.

File Structure
pneumonia-detection-cnn

|-- README.md

|-- pneumonia_detection_model.ipynb <-- The main project notebook

|-- my_chest_xray_model.keras     <-- (Model file is too large for GitHub)
                                     [Download from Google Drive](https://drive.google.com/file/d/13Zu3TeIpwT0IqUQjaENAKOfkVsLx7PgA/view?usp=drive_link)
  
|-- requirements.txt <--Python libraries

|-- .gitignore <--Files to ignore

|-- LICENSE <-- MIT License

Results
Test Accuracy: ~88.49%
Confusion Matrix:Visualizes prediction results (Normal vs. Pneumonia).
Classification Report:Shows precision, recall, and F1-score for both classes.

Note on Dataset
This model was trained on the "Chest X-Ray Images (Pneumonia)" dataset. Due to its large size, the dataset is not included in this repository.

The notebook is configured to load the data from a Google Drive path. To run this project, you must download the dataset separately and update the path in the notebook.

A popular version of this dataset can be found on Kaggle:
[https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

License
This project is licensed under the MIT License- see the LICENSE file for details.
