# import tensorflow as tf
from hands import process
import os
import imageio.v3 as imageio
from matplotlib import pyplot as plt
import numpy as np
import string

asl_dict = {char: [] for char in string.digits + string.ascii_lowercase}
asl_dir = "asl_dataset"

sub_dirs = sorted([d for d in os.listdir(asl_dir) if os.path.isdir(os.path.join(asl_dir, d))])

for dir in sub_dirs:
    cur = os.path.join(asl_dir,dir)
    for img in os.listdir(cur):
        # print(cur, img)
        image_path = os.path.join(cur, img)
        im = imageio.imread(image_path)
        frame, landmarks = process(im, False)
        np_landmarks = np.array(landmarks)
        try:
            asl_dict[dir].append(np_landmarks)
        except Exception as e:
            print(f'{e}: error adding to dict')
        # plt.imshow(frame)
        # if landmarks:
        #     plt.show()

print(asl_dict)