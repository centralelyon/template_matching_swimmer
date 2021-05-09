import cv2 

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
    

img = cv2.imread('reflets.png')
cv2.imshow('resultat', corriger_reflet(img,180,20))
cv2.imshow('reflet',img)
cv2.waitKey(0)
cv2.imwrite('reflet_corrige.png',corriger_reflet(img,180,20))