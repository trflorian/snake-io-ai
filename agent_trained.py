import math
import random
import numpy as np
import pyautogui
import keyboard
import time
import cv2
from pathlib import Path
from mss import mss
from fastai.vision.all import *

# for whole browser window (200,0,1920,800)
monitor = {"top": 300, "left": 610, "width": 700, "height": 700}

# wait until user presses t to start the program
keyboard.wait('t')
print('Started agent....')

running = True
pause = False

def on_key_press(event):
    global running, pause
    if event.name == 'p':
        pause = not pause
    if event.name == 't':
        running = False

keyboard.on_press(on_key_press)

# load model
learn = load_learner(Path("brains/"))

with mss() as sct:
    while running:
        while not pause:
            screenshot = np.array(sct.grab(monitor))
            image = cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY)
            image = cv2.Canny(image, threshold1=119, threshold2=250)

            image = cv2.resize(image, (224, 224))

            prediction_angle = learn.predict(image)

            print(f"predicted angle: {np.rad2deg(prediction_angle):.0f}")

            angle = prediction_angle
            offset = np.array((np.cos(angle), np.sin(angle))) * 100
            center = np.array((1920, 1080))/2

            target = center + offset

            pyautogui.moveTo(target[0], target[1])
            time.sleep(0.2)
