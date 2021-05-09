import cv2 
from reflets import corriger_reflet


def background_suppression(videopath, dist) :
    cap = cv2.VideoCapture(videopath)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float `width`
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter('background_suppression.avi', fourcc, fps, (int(width),  int(height)),isColor=False)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    backSub = cv2.createBackgroundSubtractorMOG2(history =length,varThreshold = dist)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))    
    
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        num_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        print(str(num_frame)+' sur '+str(length)) 
        
        
        if ret :
            frame = corriger_reflet(frame, 180,20) #si on veut corriger les reflets
            #update the background model
            fgMask = backSub.apply(frame)
            fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel)
            
            #write the output
            out.write(fgMask)
            
        else :
            break
            
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    
videopath = 'zoom2.mp4'
dist = 100

background_suppression(videopath, dist)
