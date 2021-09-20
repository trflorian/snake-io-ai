import os
import random
import time
import cv2
import numpy as np
import keyboard

latest_dir = os.listdir("data")[-1]
data_path = "data/"+latest_dir+"/"

images = os.listdir(data_path+"img")
img_ind = random.randint(0, len(images))
img_path = images[img_ind]
inp = np.loadtxt(data_path+"inputs.txt")[img_ind]

img = cv2.imread(data_path+"img/"+img_path)
center = np.array(img.shape[:2])/2

angle = inp*np.pi
offset = np.array((np.cos(angle), np.sin(angle))) * 80

img = cv2.arrowedLine(img, center.astype(int), (center+offset).astype(int), color=(255,0,0))

cv2.imshow(img_path, img)
cv2.waitKey()
