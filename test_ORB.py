import cv2 
import sys
import numpy as np
import pandas as pd
import argparse
import json
from matplotlib import pyplot as plt


def corriger_reflet(img, threshold, inpaintradius) :
     
    # Thresholding
    mask = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY ), threshold, 255, cv2.THRESH_BINARY)[1]
    
    # Closing 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Inpainting
    result = cv2.inpaint(img, mask, inpaintradius, cv2.INPAINT_TELEA) 

    # return the corrected image
    return result
    



#%% Matching sur une video

cap = cv2.VideoCapture('zoom.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) 
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('outputORB.avi', fourcc, fps, (int(width),  int(height)))
length = cap.get(cv2.CAP_PROP_FRAME_COUNT)

img1 = cv2.imread('template1.png',0) #template
w, h = img1.shape[::-1]


orb = cv2.ORB_create(nfeatures=5,edgeThreshold=0,patchSize=100) # Initiate ORB detector for the template

# find the keypoints with ORB
kp1 = orb.detect(img1,None)

# compute the descriptors with ORB
kp1, des1 = orb.compute(img1, kp1)

orb = cv2.ORB_create(nfeatures=10,edgeThreshold=0,patchSize=100) # Initiate ORB detector for the frame

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    num_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    print(str(num_frame)+' sur '+str(length))
    
    if ret :
        img2= corriger_reflet(frame, 200,20) #correction des reflets
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) #box containing swimmer
       
        
        # find the keypoints and descriptors with ORB
        kp2, des2 = orb.detectAndCompute(img2,None)
            
        
        if not type(des2)==type(None) :
            # Match descriptors.
            matches = bf.match(des1,des2)
                
            # Sort them in the order of their distance.
            matches = sorted(matches, key = lambda x:x.distance)
            list_kp2 = [kp2[mat.trainIdx].pt for mat in matches] # coord in pixels of matching keypoints in img2 
            
               
            # Draw rectangle around the match
            x,y = list_kp2[0]
            top_left = (int(x - w//2), int(y - h//2))
            bottom_right = (int(x + w//2), int(y + h//2))
                
            #Write the output (bouding box)
            out.write(cv2.rectangle(frame, top_left,bottom_right , (0,255,0), 2))
        
        else :
            out.write(frame)
        

    else :
        break

cap.release()
out.release()
cv2.destroyAllWindows()


