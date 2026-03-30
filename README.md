# American Sign Language Translator

This project aims to translate ASL letters with help from neural networks. I contrast two inference pipelines comparing accuracy:

# Data Source:
https://www.kaggle.com/datasets/ayuraj/asl-dataset

### Convolutional Neural Network

This model is comprised of 3 convolutional layers, 3 pooling layers, and 2 dense layers. It achieved a final validation accuracy of ~83%, based on images from the training dataset.

### Deep Neural Network

This model is comprised of 6 dense layers, with as many dropout layers in an effort to prevent overfitting. It achieved a final validation accuracy of ~90%.

# How to run:
    - pip install -r requirements.txt
    - To predict with your own hand images dataset:
        - python3 collect-hand-imgs.py
        - python3 create-hands-data.py <-- Make sure DATA_DIR = './data'
        - python3 train-model.py 
        - python3 webcam-predict1.py
    - To predict with asl_dataset: 
        - python3 create-hands-data.py <-- Make sure DATA_DIR = './asl_dataset'
        - python3 train-model.py 
        - python3 webcam-predict.py
    - mediapipe webcam demo:
        - $python3 webcam-demo.py 
    - single frame capture from webcam with prediction:
        - $python3 single-capture-py
    - CNN Model
        - cnn-model.ipynb
    - DNN Model x Mediapipe Preprocessing
        - mp-model.ipynb

