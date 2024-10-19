import cv2
import mediapipe as mp
import hands

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

imlist = []


cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    try:
        ret, frame = cap.read()
        if not ret:
                print("Failed to grab frame.")
                exit(1)
        frame = hands.process_frame
        cv2.imshow('Hand Detection', frame)
    finally:
        cv2.destroyAllWindows()