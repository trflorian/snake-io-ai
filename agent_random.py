import math
import random
import numpy as np
import pyautogui
import keyboard
import time

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

while running:
    while not pause:
        angle = (random.random() * 2 - 1) * math.pi
        offset = np.array((np.cos(angle), np.sin(angle))) * 100
        center = np.array((1920, 1080))/2

        target = center + offset

        pyautogui.moveTo(target[0], target[1])
        time.sleep(0.2)
