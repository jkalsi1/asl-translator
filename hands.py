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

        # Webcam input, apply transformations to make input friendly to MP
        if is_webcam:
            frame = cv2.flip(frame, 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame)
        flattened_landmarks = []
        bounding_box = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw bounding box
                x_coords = [landmark.x for landmark in hand_landmarks.landmark]
                y_coords = [landmark.y for landmark in hand_landmarks.landmark]

                x_min = int(min(x_coords) * frame.shape[1])
                x_max = int(max(x_coords) * frame.shape[1])
                y_min = int(min(y_coords) * frame.shape[0])
                y_max = int(max(y_coords) * frame.shape[0])

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                
                bounding_box = (x_min, y_min, x_max, y_max)
                # End draw bounding box 

                # Draw hand landmarks skeleton
                for landmark in hand_landmarks.landmark:
                    # Add landmarks to flattened, regular array
                    flattened_landmarks.extend([landmark.x, landmark.y, landmark.z])
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        # frame with hand skeleton and bounding box, hand landmarks list, bounding box coordinates
        return frame, flattened_landmarks, bounding_box
