# import tensorflow as tf
from hands import process
import os
import imageio.v3 as imageio
from matplotlib import pyplot as plt
import cv2

asl_dir = "asl_dataset"
sub_dirs = sorted([d for d in os.listdir(asl_dir) if os.path.isdir(os.path.join(asl_dir, d))])

dir_0 = os.path.join(asl_dir,sub_dirs[0])
for img in os.listdir(dir_0):
    print(dir_0, img)
    image_path = os.path.join(dir_0, img)
    im = imageio.imread(image_path)
    frame, landmarks = process(im)
    plt.imshow( frame)
    plt.show()
    print(landmarks)

# # Iterate through each subdirectory in alphabetical order
# for sub_dir in sub_dirs:
#     sub_dir_path = os.path.join(asl_dir, sub_dir)

#     images = sorted([img for img in os.listdir(sub_dir_path) if img.endswith(('.jpg', '.jpeg', '.png'))])
    
#     # Call the function on each image
#     for img in images:
#         image_path = os.path.join(sub_dir_path, img)
#         im = imageio.imread(image_path)
#         frame, arr = process(im)
#         print(arr)