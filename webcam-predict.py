import cv2
import numpy as np
import mediapipe as mp
import pickle
import random
from keras.api.models import load_model

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)
labels_dict = {0: 'A', 1: 'B', 2: 'C'}

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
try:
    while True:
        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()

        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            try:
                right_hand_landmarks = None
                for hand_landmarks in results.multi_hand_landmarks:
                    handedness = hands.process(frame_rgb).multi_handedness
                    for hand_label in handedness:
                        if hand_label.classification[0].label == 'Right':
                            right_hand_landmarks = hand_landmarks
                            break
                    if right_hand_landmarks:
                        break

                if right_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        right_hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

                    for i in range(len(right_hand_landmarks.landmark)):
                        x = right_hand_landmarks.landmark[i].x
                        y = right_hand_landmarks.landmark[i].y
                        data_aux.append(x)
                        data_aux.append(y)
                        x_.append(x)
                        y_.append(y)

                    x1 = int(min(x_) * W) - 10
                    y1 = int(min(y_) * H) - 10

                    x2 = int(max(x_) * W) - 10
                    y2 = int(max(y_) * H) - 10

                    prediction = model.predict([np.asarray(data_aux)])

                    predicted_char = labels_dict[int(prediction[0])]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                    cv2.putText(frame, predicted_char, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                                cv2.LINE_AA)
            except ValueError as e:
                print(e)

        cv2.imshow('frame', frame)
        cv2.waitKey(25)
finally:
    cap.release()
    cv2.destroyAllWindows()