import os
import time
import cv2
import keyboard
import pyautogui
import numpy as np
from datetime import datetime
from mss import mss
import winsound

# for whole browser window (200,0,1920,800)
monitor = {"top": 275, "left": 560, "width": 650, "height": 650}

# create data folder
folder_name = datetime.now().strftime('%Y%m%d_%H%M%S')
path = "data/"+folder_name+"/"
os.mkdir(path)
img_path = path+"img/"
os.mkdir(img_path)
inputs_path = path+"inputs.txt"

# wait until user presses t to start the program
keyboard.wait('t')
print('Started collecting data....')
winsound.Beep(1000, 200)

collecting = True

def on_key_press(event):
    global collecting
    print(event.name)
    if event.name == 't':
        collecting = False

keyboard.on_press(on_key_press)

counter = 0
angle_inputs = []
with mss() as sct:
    while collecting:
        screenshot = np.array(sct.grab(monitor))

        image = cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY)
        image = cv2.Canny(image, threshold1=120, threshold2=250)
        image[:130, 200:600] = 0

        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

        mouse_pos = pyautogui.position()
        delta = np.array((mouse_pos.x - 1920/2, mouse_pos.y - 1080/2))
        delta /= np.linalg.norm(delta)

        # [-pi, pi]
        angle = np.arctan2(delta[0], delta[1])

        # [-1, 1]
        my_input = angle / np.pi

        angle_inputs.append(my_input)

        cv2.imwrite(f'{img_path}{counter:05d}.png', image)
        counter += 1

to_remove = min(100, counter)

print(f'Stopped collecting data. Removing last {to_remove} data points')

files = os.listdir(img_path)

for i in range(-to_remove, 0):
    os.remove(img_path + files[i])

angle_inputs = angle_inputs[:-to_remove]

np.savetxt(inputs_path, np.array(angle_inputs))
winsound.Beep(1000, 200)
time.sleep(0.01)
winsound.Beep(1000, 200)
