import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def process(frame):
    """Process a frame to detect and annotate hand landmarks."""
    
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

        # Flip the frame horizontally for a correct handedness
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to detect hands
        results = hands.process(frame_rgb)
        flattened_landmarks = []
        # If hands are detected, draw landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    # Append x, y, z coordinates of each landmark to the flattened list
                    flattened_landmarks.extend([landmark.x, landmark.y, landmark.z])
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        return frame, flattened_landmarks
