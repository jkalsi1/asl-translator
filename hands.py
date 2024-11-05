import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def process(frame, is_webcam, num_hands):
    """Process a frame to detect and annotate hand landmarks and return bounding box coordinates."""
    
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=num_hands,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

        # Flip the frame horizontally if using webcam
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB (for webcam only)
        if is_webcam:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to detect hands
        results = hands.process(frame)
        flattened_landmarks = []
        bounding_box = None

        # If hands are detected, process landmarks and calculate bounding box
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_coords = [landmark.x for landmark in hand_landmarks.landmark]
                y_coords = [landmark.y for landmark in hand_landmarks.landmark]

                # Get bounding box coordinates
                x_min = int(min(x_coords) * frame.shape[1])
                x_max = int(max(x_coords) * frame.shape[1])
                y_min = int(min(y_coords) * frame.shape[0])
                y_max = int(max(y_coords) * frame.shape[0])

                # Draw the bounding box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

                # Store the bounding box coordinates for use in main
                bounding_box = (x_min, y_min, x_max, y_max)

                # Flatten the landmarks for further processing
                for landmark in hand_landmarks.landmark:
                    flattened_landmarks.extend([landmark.x, landmark.y, landmark.z])
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Return the processed frame, flattened landmarks, and bounding box coordinates
        return frame, flattened_landmarks, bounding_box
