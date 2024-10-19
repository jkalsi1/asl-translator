import cv2
import mediapipe as mp

# Initialize MediaPipe Hands and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

imlist = []

with mp_hands.Hands(
    static_image_mode=False,  # Set to False for video feed
    max_num_hands=1,          # Detect up to 2 hands
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    try:
        for frame in imlist:
            pass
    finally:
    cv2.destroyAllWindows()