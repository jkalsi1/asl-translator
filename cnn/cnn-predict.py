import cv2
from hands_crop import process
import numpy as np
from keras.api.models import load_model

simpleModel = load_model('/Users/jagan-kalsi/Desktop/class/csc/csc487/asl/cnn/simple-model.keras')

# Dict for getting prediction from np.argmax
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

            frame_with_hands, landmarks, bounding_box, cropped_frame = process(frame, True, 2)
            prediction = None

            try:
                if cropped_frame is not None and cropped_frame.size > 0:
                    cropped_frame= cv2.resize(cropped_frame, (224,224))
                    cropped_frame = cropped_frame.astype(np.float32) / 255.0
                    cropped_frame = np.expand_dims(cropped_frame, axis=0)
                    res = simpleModel.predict(cropped_frame)
                    prediction = key[np.argmax(res)]
            except Exception as e:
                print(f'Invalid cropped frame: {e}')
            
                
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
            
            cv2.imshow('American Sign Language Translator', frame_with_hands)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except:
        print("couldn't find hand.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
