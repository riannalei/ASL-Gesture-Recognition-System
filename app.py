import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp

# Load the trained model
model = load_model("asl_model.h5")

# Define label mapping
label_mapping = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H",
    8: "I", 9: "J", 10: "K", 11: "L", 12: "M", 13: "N", 14: "O", 15: "P",
    16: "Q", 17: "R", 18: "S", 19: "T", 20: "U", 21: "V", 22: "W",
    23: "X", 24: "Y", 25: "Z", 26: "del", 27: "nothing", 28: "space"
}

# Preprocess the frame
def preprocess_frame(frame):
    img = cv2.resize(frame, (64, 64))  # Resize to model input size
    img = img / 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Start capturing video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror the frame
    height, width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands
    result = hands.process(rgb_frame)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get bounding box coordinates
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * width) - 20
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * height) - 20
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * width) + 20
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * height) + 20

            # Ensure ROI is within frame bounds
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(width, x_max), min(height, y_max)

            roi = frame[y_min:y_max, x_min:x_max]

            # Preprocess ROI and predict
            preprocessed_roi = preprocess_frame(roi)
            prediction = model.predict(preprocessed_roi)
            predicted_label = np.argmax(prediction)
            predicted_class = label_mapping[predicted_label]

            # Debugging prints
            print(f"Raw probabilities: {prediction[0]}")
            print(f"Predicted class: {predicted_class}, Confidence: {max(prediction[0]):.2f}")

            # Add a threshold for "nothing"
            if predicted_class == "nothing" and max(prediction[0]) < 0.90:  # Adjust threshold if needed
                predicted_class = "Uncertain"

            # Print the detected letter if it's valid
            if predicted_class not in ["nothing", "Uncertain"]:
                print(f"Detected letter: {predicted_class}")

            # Display prediction
            cv2.putText(frame, f"Prediction: {predicted_class} ({max(prediction[0]):.2f})",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Draw bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

    cv2.imshow("ASL Recognition", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
