import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('asl_model.h5')

# Define the label mapping (same as in training)
label_mapping = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
    8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P',
    16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W',
    23: 'X', 24: 'Y', 25: 'Z', 26: 'del', 27: 'nothing', 28: 'space'
}

# Preprocess the frame
def preprocess_frame(frame):
    img = cv2.resize(frame, (64, 64))  # Resize to match model input size
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Start capturing video from webcam
cap = cv2.VideoCapture(0)  # 0 for default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video frame")
        break

    # Flip the frame (optional, depending on your webcam)
    frame = cv2.flip(frame, 1)

    # Define a larger Region of Interest (ROI) for hand detection
    height, width, _ = frame.shape
    x, y, w, h = 50, 50, width - 100, height - 100  # Larger ROI
    roi = frame[y:y+h, x:x+w]
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Preprocess the ROI
    preprocessed_roi = preprocess_frame(roi)

    # Predict the ASL letter
    prediction = model.predict(preprocessed_roi)
    predicted_label = np.argmax(prediction)
    predicted_class = label_mapping[predicted_label]
    confidence = prediction[0][predicted_label]

    # Only display and print if confidence is above threshold (e.g., 70%)
    if confidence > 0.7:
        print(f"Prediction: {predicted_class} with confidence: {confidence:.2f}")

        # Display the prediction on the video feed
        cv2.putText(frame, f'Prediction: {predicted_class}', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the video feed
    cv2.imshow('ASL Recognition', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
