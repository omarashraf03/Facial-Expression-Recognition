from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np
import os

# Ensure the model path is correct
model_path = '/Users/mac/Desktop/AI/Facial Expression Recognition/model.keras'
face_cascade_path = '/Users/mac/Desktop/AI/Facial Expression Recognition/haarcascade_frontalface_default.xml'

# Check if the model file exists
if not os.path.exists(model_path):
    print(f"⚠️ Model file not found at {model_path}")
    exit()

# Load the saved Keras model
classifier = load_model(model_path)

# Check if face cascade file exists
if not os.path.exists(face_cascade_path):
    print(f"⚠️ Haar cascade file not found at {face_cascade_path}")
    exit()

# Load the Haar cascade for face detection
face_classifier = cv2.CascadeClassifier(face_cascade_path)

# Emotion labels
class_labels = ['happy', 'sad','surprise', 'neutral']

# Open the camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("⚠️ Failed to open camera")
    exit()

print("Camera is working...")

while True:
    # Grab a frame from the video capture
    ret, frame = cap.read()

    if not ret or frame is None:
        print("⚠️ Failed to grab frame")
        continue

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    # Loop through each detected face
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Extract the region of interest (ROI) for face
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            # Preprocess the face for prediction
            roi = roi_gray.astype('float32') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # Make a prediction on the ROI (face)
            preds = classifier.predict(roi)[0]
            label = class_labels[preds.argmax()]
            label_position = (x, y)

            # Display the predicted emotion on the frame
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, 'No Face Found', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    # Display the frame with emotion labels
    cv2.imshow('Emotion Detector', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
