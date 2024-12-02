import cv2
import mediapipe as mp
import numpy as np
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation

segmentor = SelfiSegmentation()
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def process(frame, is_webcam, num_hands):
    """Process a frame to detect hand landmarks and return the original frame with bounding box and a cropped frame around the bounding box."""
    
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=num_hands,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

        # Apply transformations if frame is from webcam
        if is_webcam:
            frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        flattened_landmarks = []
        cropped_image = None
        original_frame_with_box = frame.copy()
        bounding_box = None
        edges = None
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Calculate bounding box coordinates
                x_coords = [landmark.x for landmark in hand_landmarks.landmark]
                y_coords = [landmark.y for landmark in hand_landmarks.landmark]

                x_min = int(min(x_coords) * frame.shape[1])
                x_max = int(max(x_coords) * frame.shape[1])
                y_min = int(min(y_coords) * frame.shape[0])
                y_max = int(max(y_coords) * frame.shape[0])

                # Increase bounding box by 10%
                width = x_max - x_min
                height = y_max - y_min
                x_min = max(x_min - int(0.2 * width), 0)
                x_max = min(x_max + int(0.2 * width), frame.shape[1])
                y_min = max(y_min - int(0.2 * height), 0)
                y_max = min(y_max + int(0.2 * height), frame.shape[0])

                # Draw bounding box on the original frame
                cv2.rectangle(original_frame_with_box, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

                # Crop the frame to the bounding box
                cropped_image = frame[y_min:y_max, x_min:x_max]
                green = (0,255,0)
                
                try:
                    if len(cropped_image.shape) == 3 and cropped_image.shape[2] == 3:
                        cropped_image = cv2.GaussianBlur(cropped_image, (9, 9), 3)
                        cropped_image = segmentor.removeBG(cropped_image, green, cutThreshold=0.2)
                        gray_img = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                        
                        
                        gray_img = gray_img.astype(np.uint8)
                        edges = cv2.Canny(gray_img, 100, 200)
                        
                        edges = edges.astype(np.float32)
                        edges = np.expand_dims(edges, axis=-1)
                except Exception as e:
                    print(f'error pre processing webcam input, {e}')
                    

                # Add landmarks to flattened list
                for landmark in hand_landmarks.landmark:
                    flattened_landmarks.extend([landmark.x, landmark.y, landmark.z])

                bounding_box = (x_min, y_min, x_max, y_max)
                break 

        return original_frame_with_box, flattened_landmarks, bounding_box, edges
