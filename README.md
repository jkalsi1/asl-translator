# How to run:
    - pip install -r requirements.txt
    - To predict with your own hand images dataset:
        - python3 collect-hand-imgs.py
        - python3 create-hands-data.py <-- Make sure DATA_DIR = './data'
        - python3 train-classifier.py 
        - python3 webcam-predict.py
    - To predict with asl_dataset: 
        - python3 create-hands-data.py <-- Make sure DATA_DIR = './asl_dataset'
        - python3 train-classifier.py 
        - python3 webcam-predict.py
    - mediapipe webcam demo:
        - $python3 webcam-demo.py 
    - single frame capture from webcam with prediction:
        - $python3 single-capture-py
    - CNN Model
        - cnn-model.ipynb
    - DNN Model x Mediapipe Preprocessing
        - mp-model.ipynb

# Data Source:
https://www.kaggle.com/datasets/ayuraj/asl-dataset
