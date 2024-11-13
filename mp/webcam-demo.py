import cv2
from hands import process

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
            frame_with_hands, hands_arr = process(frame, True, 2)
            
            # Display the frame with hand landmarks
            cv2.imshow('Hand Detection',  cv2.cvtColor(frame_with_hands, cv2.COLOR_RGB2BGR))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
