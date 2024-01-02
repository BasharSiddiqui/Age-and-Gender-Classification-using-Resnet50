import tkinter as tk
from tkinter import filedialog
import numpy as np
import cv2
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt

# Load the trained model
model = load_model("model.h5")

gender_dict = {0: "Male", 1: "Female"}

def preprocess_image(image):
    img = cv2.resize(image, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

# Function to handle real-time video capturing
def start_realtime():
    # Initialize the camera
    camera = cv2.VideoCapture(0)

    while True:
        # Capture frame from the camera
        ret, frame = camera.read()

        # Preprocess the captured frame
        preprocessed_image = preprocess_image(frame)

        # Make predictions
        predictions = model.predict(preprocessed_image)
        gender_prediction = gender_dict[round(predictions[0][0][0])]
        age_prediction = round(predictions[1][0][0])
        age_lower = age_prediction - 3
        age_upper = age_prediction + 3

        # Display the original frame and predicted gender/age
        cv2.putText(frame, f"Predicted Gender: {gender_prediction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(frame, f"Predicted Age: {age_lower} to {age_upper}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.imshow("Gender and Age Prediction", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(700) & 0xFF == ord('q'):
            break

    camera.release()

    cv2.destroyAllWindows()

# Function to handle image uploading
def upload_image():
    file_path = filedialog.askopenfilename()

    if file_path:
        image = Image.open(file_path)
        image = np.array(image)
        preprocessed_image = preprocess_image(image)
        predictions = model.predict(preprocessed_image)
        gender_prediction = gender_dict[round(predictions[0][0][0])]
        age_prediction = round(predictions[1][0][0])
        age_lower = age_prediction - 3
        age_upper = age_prediction + 3

        # Display the original image and predicted gender/age
        plt.imshow(image)
        plt.title(f"Predicted Gender: {gender_prediction} Predicted Age: {age_lower} to {age_upper}")
        plt.axis("off")
        plt.show()

        print(f"Predicted Gender: {gender_prediction}")
        print(f"Predicted Age: {age_lower} to {age_upper}")

window = tk.Tk()
window.geometry("800x600")
menu_screen = tk.Frame(window)
menu_screen.pack()

realtime_button = tk.Button(menu_screen, text="Real-time", command=start_realtime)
realtime_button.pack(pady=10)

upload_button = tk.Button(menu_screen, text="Upload an Image", command=upload_image)
upload_button.pack(pady=10)

window.mainloop()