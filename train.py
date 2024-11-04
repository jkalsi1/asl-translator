# import tensorflow as tf
from hands import process
import os
import imageio.v3 as imageio
from matplotlib import pyplot as plt
import numpy as np
import string
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.src.models import Sequential
from keras.src.layers import Dense

asl_dict = {char: [] for char in string.digits + string.ascii_lowercase}
asl_dir = "asl_dataset"

sub_dirs = sorted([d for d in os.listdir(asl_dir) if os.path.isdir(os.path.join(asl_dir, d))])

for dir in sub_dirs:
    cur = os.path.join(asl_dir,dir)
    for img in os.listdir(cur):
        image_path = os.path.join(cur, img)
        im = imageio.imread(image_path)
        frame, landmarks = process(im, False, 1)
        try:
            asl_dict[dir].append(landmarks)
        except Exception as e:
            print(f'{e}: error adding to dict')


X = []
y = []

for label, landmark_arrays in asl_dict.items():
    for landmarks in landmark_arrays:
        X.append(landmarks)  
        y.append(label)    

X = np.array(X, dtype=np.float32) 
y = np.array(y)                   

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Dense(128, activation='relu', input_shape=(63,)), 
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

plt.figure(figsize=(10, 5))

plt.plot(history.history['accuracy'], label='Training Accuracy')

plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
