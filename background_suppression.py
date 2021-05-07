import cv2 
import sys
import numpy as np
import pandas as pd
import argparse
import json
from matplotlib import pyplot as plt

nb_passage = 0

cap = cv2.VideoCapture('videos/zoom.mp4')
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

backSub = cv2.createBackgroundSubtractorMOG2(history =(nb_passage+1)*length,varThreshold = 80)
#backSub = cv2.createBackgroundSubtractorKNN(history=(nb_passage+1)*length,dist2Threshold = 2000)

cap.release()
cv2.destroyAllWindows()

for i in range(nb_passage) :

    cap = cv2.VideoCapture('videos/zoom.mp4')
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        num_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        print(str(num_frame)+' sur '+str(length))
        
        if ret :
    
            #update the background model
            fgMask = backSub.apply(frame)
            
        else :
            break
            
        
    
    cap.release()
    cv2.destroyAllWindows()


cap = cv2.VideoCapture('videos/zoom.mp4')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float `width`
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter('background_suppression.avi', fourcc, fps, (int(width),  int(height)),isColor=False)
out_background = cv2.VideoWriter('background_suppression_model.avi', fourcc, fps, (int(width),  int(height)),isColor=False)


length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    num_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    print(str(num_frame)+' sur '+str(length))
    
    if ret :

        #update the background model
        fgMask = backSub.apply(frame)
        #fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel)
        
        #write the output
        out.write(fgMask)
        #out_background.write(cv2.backSub.getBackgroundImage(frame))
        
    else :
        break
        
    

cap.release()
out.release()
out_background.release()
cv2.destroyAllWindows()
