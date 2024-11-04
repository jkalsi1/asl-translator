# import tensorflow as tf
from hands import process
import os
import imageio.v3 as imageio
from matplotlib import pyplot as plt
import cv2

asl_dir = "asl_dataset"
sub_dirs = sorted([d for d in os.listdir(asl_dir) if os.path.isdir(os.path.join(asl_dir, d))])

for dir in sub_dirs:
    cur = os.path.join(asl_dir,dir)
    for img in os.listdir(cur):
        print(cur, img)
        image_path = os.path.join(cur, img)
        im = imageio.imread(image_path)
        frame, landmarks = process(im)
        plt.imshow( frame)
        if landmarks:
            plt.show()