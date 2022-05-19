import ctypes
import os
import time
import cv2
import mediapipe as mp
import numpy as np
from mss import mss
from PIL import Image
import pyautogui as pag
import win32api
import win32con
import ait

import keys

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

sct = mss()
cTime = 0
pTime = 0

# cap = cv2.VideoCapture(0)
## Setup mediapipe instance
with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:

        # ret, image = cap.read()

        ###########
        sct_img = sct.grab({
            "left": 704,
            "top": 284,
            "width": 512,
            "height": 512
        })

        # Create an Image
        img = Image.new("RGB", sct_img.size)

        # Best solution: create a list(tuple(R, G, B), ...) for putdata()
        pixels = zip(sct_img.raw[2::4], sct_img.raw[1::4], sct_img.raw[0::4])
        img.putdata(list(pixels))

        ### image => cv2.image => np.ndarray =>

        img = np.array(img)

        # Recolor image to RGB
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        imgHeight = img.shape[0]
        imgWidth = img.shape[1]
        ###########
        # #
        # # # Make detection
        results = pose.process(image)
        # imgHeight = image.shape[0]
        # imgWidth = image.shape[1]
        # #
        ###########
        # Recolor back to BGR
        image.flags.writeable = True
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        ###########

        if results.pose_landmarks:
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )
            for i, p in enumerate(results.pose_landmarks.landmark):
                xPos = int(p.x * imgWidth)
                yPos = int(p.y * imgHeight)

                cv2.putText(image, str(i), (xPos - 25, yPos + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)

                if i == 9:
                    print(xPos)
                    print(yPos)
                    keys.mouseMoveTo(256, 256, xPos, yPos)
                    keys.mouseClick('L')

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(image, f"FPS : {int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        cv2.imshow('Mediapipe Feed', image)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cv2.destroyAllWindows()
