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
