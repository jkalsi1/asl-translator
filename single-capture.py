import cv2
from hands import process
from keras.api.models import load_model
import numpy as np

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
        frame_with_hands = ""
        for i in range(1):
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            frame_with_hands, landmarks = process(frame, True, 2)

            # Predict here
            if landmarks is not None:
                landmarks = np.array(landmarks, dtype=np.float32)
                landmarks = landmarks.reshape(1, -1)
                print(landmarks.shape)
                res = model.predict(landmarks)
                
                print(key[np.argmax(res)])

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
