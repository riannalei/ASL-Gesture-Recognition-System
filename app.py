import cv2
import numpy as np
from tensorflow.keras.models import load_model

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

# Start capturing video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror the frame
    height, width, _ = frame.shape
    x, y, w, h = 150, 150, 300, 300  # Adjust ROI
    roi = frame[y:y+h, x:x+w]

    # Preprocess ROI and predict
    preprocessed_roi = preprocess_frame(roi)
    prediction = model.predict(preprocessed_roi)
    predicted_label = np.argmax(prediction)
    predicted_class = label_mapping[predicted_label]

    # Add a threshold for "nothing"
    if predicted_class == "nothing" and max(prediction[0]) < 0.95:
        predicted_class = "Uncertain"

    # Display prediction
    cv2.putText(frame, f"Prediction: {predicted_class} ({max(prediction[0]):.2f})", 
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Draw ROI rectangle and show frames
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.imshow("ASL Recognition", frame)
    cv2.imshow("Region of Interest", cv2.resize(roi, (128, 128)))  # Upscale for visibility

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
