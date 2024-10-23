import cv2
from hands import process
import numpy as np

def main():
    # Initialize webcam capture
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    try:
        frame_with_hands = ""
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Failed to grab frame.")
                break

            # Process the frame for hand detection
            frame_with_hands, hands_arr = process(frame)
            # print(hands_arr)
            
            # Display the frame with hand landmarks
            cv2.imshow('Hand Detection', frame_with_hands)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.imshow('Hand Detection', frame_with_hands)
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
