import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Start hand detection
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    try:
        for i in range(1):
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                continue

            # Flip the frame horizontally for correct-handedness
            frame = cv2.flip(frame, 1)

            # Convert the BGR image to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame to detect hands
            results = hands.process(frame_rgb)

            # Draw hand landmarks on the frame if hands are detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            print(f"landmarks",results.multi_hand_landmarks)
            # Display the frame with hand annotations
            cv2.imshow('Hand Detection', frame)

    finally:
        cap.release()
        cv2.destroyAllWindows()
