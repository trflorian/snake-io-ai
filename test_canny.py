import os
import time
import cv2
import keyboard
import pyautogui
import numpy as np
from datetime import datetime
from mss import mss

# for whole browser window (200,0,1920,800)
monitor = {"top": 200, "left": 560, "width": 800, "height": 800}

time.sleep(4)

counter = 0
angle_inputs = []
with mss() as sct:
    screenshot = np.array(sct.grab(monitor))

    image = cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY)
    image = cv2.Canny(image, threshold1=120, threshold2=250)
    image[40:170, 200:600] = 0

    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

#resized = cv2.resize(image, (600, 600), interpolation=cv2.INTER_AREA)
cv2.imshow("Canny Test", image)
cv2.imwrite("test.png", image)
cv2.waitKey()
