import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 1. Load model
model = load_model('banana_model.h5', compile=False)
print("Model loaded successfully!")

def preprocess_frame(frame):
    target_size = (224, 224)
    resized = cv2.resize(frame, target_size)
    img_array = resized.astype(np.float32) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch

def predict_days(frame):
    processed = preprocess_frame(frame)
    pred = model.predict(processed, verbose=0)[0][0]
    return round(float(pred), 1)

def draw_overlay(frame, days):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (w-10, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (10, 10), (w-10, 120), (255, 255, 255), 2)

    if days >= 6:
        status = "Fresh"
    elif 4 <= days < 6:
        status = "Ripe"
    elif 2 <= days < 4:
        status = "Overripe"
    else:
        status = "About to spoil"

    text = f"Days remaining: {days:.1f}  ({status})"
    cv2.putText(frame, text, (30, 55), cv2.FONT_HERSHEY_SIMPLEX,
                0.95, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(frame, "SPACE = Capture result | Q = Quit", (30, h - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)

    return frame

def show_result_screen(days, captured_frame):
    # create a blank image for the result screen
    result_img = np.zeros((720, 1000, 3), dtype=np.uint8)

    # Resize the captured frame to fit into the result screen
    img_display = cv2.resize(captured_frame, (420, 420))
    result_img[80:500, 50:470] = img_display

    # Draw a rectangle around the image display area
    cv2.rectangle(result_img, (50, 80), (470, 500), (255, 255, 255), 2)

    # Determine color and message based on days
    if days >= 6:
        color = (0, 255, 0)
        status = "Fresh"
        message = "The banana is still very fresh!"
    elif 4 <= days < 6:
        color = (0, 255, 255)
        status = "Ripe"
        message = "The banana is perfectly ripe!"
    elif 2 <= days < 4:
        color = (0, 165, 255)
        status = "Overripe"
        message = "The banana is overripe, eat soon"
    else:
        color = (0, 0, 255)
        status = "About to spoil"
        message = "The banana is about to spoil!"

    # Title 
    cv2.putText(result_img, "PREDICTION RESULT", (520, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)

    # Days remaining
    cv2.putText(result_img, f"{days:.1f} days", (540, 250),
                cv2.FONT_HERSHEY_SIMPLEX, 2.2, color, 5)

    # Status
    cv2.putText(result_img, status, (560, 330),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

    # Message
    cv2.putText(result_img, message, (520, 420),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (200, 200, 200), 2)

    # Exit instruction
    cv2.putText(result_img, "Press any key to exit", (580, 650),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (140, 140, 140), 2)

    cv2.imshow("Result - Banana Ripeness", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#MAIN
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

print("Camera is running...")
print("Press SPACE to capture | Press Q to quit")

final_days = None
captured_frame = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    days_left = predict_days(frame)
    display_frame = draw_overlay(frame.copy(), days_left)

    cv2.imshow('Banana Ripeness Predictor', display_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == 32:  # SPACE
        final_days = days_left
        captured_frame = frame.copy()   # Saving the screen shot
        break

cap.release()
cv2.destroyAllWindows()

#Print the result
if final_days is not None and captured_frame is not None:
    show_result_screen(final_days, captured_frame)