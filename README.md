# Age And Gender Classification using ResNet-50

## Description

This project utilizes a pre-trained ResNet-50 model for age and gender prediction on the UTKFace dataset. The model is fine-tuned to predict the age and gender of individuals in facial images. The age is predicted as a regression task, while gender is treated as a binary classification problem.

## Requirements

Make sure you have the following dependencies installed:

- Python (>=3.6)
- TensorFlow (>=2.0)
- NumPy
- Pillow (PIL)
- scikit-learn

## Usage

1. **Dataset Preparation:**
   - Download the UTKFace dataset.
   - Organize the dataset into a directory named "UTKFace."

2. **Preprocessing:**
   - Adjust the `batch_size` variable to your liking.
   - Run the preprocessing script to load and process the images.

4. **Model Training:**
   - The pre-trained ResNet-50 model is loaded and fine-tuned on the UTKFace dataset.
   - Age is predicted using Mean Squared Error loss, and gender is predicted using Categorical Crossentropy loss.

5. **Save Model:**
   - The fine-tuned model is saved as 'model.h5.'
   
##Testing

     This Python script (`gui.py`) provides a graphical user interface (GUI) for testing the trained deep learning model used for gender and age prediction. The model has been fine-tuned on the UTKFace dataset and is loaded from the 'model.h5' file.

## Requirements

Make sure you have the following dependencies installed:

- Python (>=3.6)
- OpenCV
- NumPy
- Keras
- Pillow (PIL)
- Matplotlib
- Tkinter

Ensure that the 'model.h5' file containing the trained model is available.

## Usage

1. **Run the GUI(gui.py file):**
   - Execute the script to launch the Tkinter GUI.
   - Use the "Real-time" button to start capturing video from your webcam with real-time predictions.
   - Alternatively, use the "Upload an Image" button to select an image for prediction.

2. **Real-time Prediction:**
   - Press 'q' to exit the real-time prediction loop.

3. **Image Upload:**
   - Select an image using the file dialog.
   - The application will display the original image and predictions for gender and age.

## Notes

- The application utilizes a pre-trained ResNet-50 model fine-tuned for gender and age prediction.
- Real-time prediction uses OpenCV for video capturing.
- Tkinter is used for the graphical user interface.
- Ensure the necessary dependencies are installed before running the application.
