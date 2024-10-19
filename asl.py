import cv2
import mediapipe as mp

# Initialize MediaPipe Hands and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Start hand detection
with mp_hands.Hands(
    static_image_mode=False,  # Set to False for video feed
    max_num_hands=2,          # Detect up to 2 hands
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            # Flip the frame horizontally for a mirrored view
            frame = cv2.flip(frame, 1)

            # Convert the BGR image to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame to detect hands
            results = hands.process(frame_rgb)

            # Draw hand landmarks on the frame if hands are detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Display the frame with hand annotations
            cv2.imshow('Hand Detection', frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # When everything is done, release the capture
        cap.release()
        cv2.destroyAllWindows()
