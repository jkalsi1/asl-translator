import cv2
from hands import process
import numpy as np
import random
from keras.api.models import load_model

model = load_model('asl-model.keras')

model.summary()
key = {i: str(i) for i in range(10)} 
key.update({i + 10: chr(97 + i) for i in range(26)})

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Failed to grab frame.")
                break

            frame_with_hands, landmarks, bounding_box = process(frame, True, 2)
            prediction = None
            # Make prediction based on hand landmarks
            if len(landmarks) == 63:
                landmarks = np.array(landmarks, dtype=np.float32)
                landmarks = landmarks.reshape(1, -1)
                res = model.predict(landmarks)
                prediction = key[np.argmax(res)]
                
            # Draw prediction on returned frame if it exists
            if bounding_box and prediction:
                x_min, y_min, _, _ = bounding_box
                cv2.putText(
                    frame_with_hands,
                    prediction,
                    (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )
            # Display image with frame, skeleton, and prediction
            cv2.imshow('Hand Detection', cv2.cvtColor(frame_with_hands, cv2.COLOR_RGB2BGR))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
