import cv2 as cv
import mediapipe as mp
import HandTrackingModule as htm
import os
import time
import numpy as np

folderPath="Menu"
myList=os.listdir(folderPath)
# print(myList)
headImg=[]

for imPath in myList:
    image=cv.imread(f'{folderPath}/{imPath}')
    headImg.append(image)

header=headImg[0]
drawColor=(255,0,255)
brushThickness=15

cap=cv.VideoCapture(0)
cap.set(3,1278)
cap.set(4,720)

detector=htm.handDetector(detectionCon=0.85)

tipIds=[4,8,12,16,20] # the value of thqe tip of each of the finger is stored int his list except thumb which serves a special case

xp,yp=0,0
imgCanvas=np.zeros((720,1278,3),np.uint8)

while True:
    # 1. Import the image
    success,img=cap.read()
    img=cv.flip(img,1)

    # 2. Finding the hand landmarks
    img=detector.findHands(img)
    lmList=detector.findPosition(img,draw=False)

    if lmList:
        # print(lmList)
        x1,y1=lmList[8][1],lmList[8][2] # tip of index finger
        x2,y2=lmList[12][1],lmList[12][2] # tip of middle finger

        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)

        # 4. If selection mode - Two fingers are up
        if fingers[1] and fingers[2]:
            # cv.rectangle(img,(x1,y1-25),(x2,y2+25),drawColor,cv.FILLED)
            # print("Selection Mode")
            if y1<125: # when the index finger is on the header
                if 150<x1<315:
                    header=headImg[0]
                    drawColor=(255,0,255)
                elif 400<x1<635:
                    header=headImg[1]
                    drawColor=(255,0,0)
                elif 700<x1<955:
                    header=headImg[2]
                    drawColor=(0,255,0)
                elif 1050<x1<1279:
                    header=headImg[3]
                    drawColor=(0,0,0)
                    brushThickness=50

        # 5. If drawing mode - Index finger is up
        if fingers[1] and fingers[2]==False:
            cv.circle(img,(x1,y1),brushThickness,drawColor,cv.FILLED)
            # print("Drawing Mode")
            if xp==0 and yp==0:
                xp,yp=x1,y1

            cv.line(img,(xp,yp),(x1,y1),drawColor,brushThickness)
            cv.line(imgCanvas,(xp,yp),(x1,y1),drawColor,brushThickness)
            xp,yp=x1,y1

    # Setting the menu of colors
    img[0:126,0:1279]=header
    cv.imshow("Image",img)
    cv.imshow("Canvas",imgCanvas)
    if cv.waitKey(1) & 0xFF==ord('q'):
        break