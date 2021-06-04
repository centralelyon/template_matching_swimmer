import cv2
import numpy as np
import pandas as pd
import argparse
import json
from matplotlib import pyplot as plt

#resolution video above
resx=1920
resy=1080


def get_index(list_dict, vid_name):
    """helper to read the json file."""
    index = -1
    for i in range(len(list_dict)):
        if list_dict[i]['name'] == vid_name:
            index = i
    return index


def zoom(video, start_time, swimmer_data, num_swimmer, save_path, size_box):
    origins = [] #coordonnees de l'origine dans l'image zoomee dans la grande imagine pour chaque frame
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    time_shift = round((start_time - 1) * fps)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    compt = 0
    # output video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(save_path, fourcc, fps, size_box)
    length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    # we read frames until synchronised
    for _ in range(abs(time_shift)):
        cap.read()
        

    while(True):
        ret, frame = cap.read()
        num_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        print(str(num_frame)+' sur '+str(length))

        if ret is not True:
            break
        
        else:
            # zoom
            if compt < len(swimmer_data) :
                x = swimmer_data[compt][1]
                to_save = np.zeros((size_box[1], size_box[0], 3)).astype(np.uint8)
                
                if x != -1:
                    x = int((50 - x) * frame.shape[1] / 50) # abscisse calculee avec la distance parcourue jusqua 50m
                    w = size_box[1]
                    y = int(frame.shape[0] * ((num_swimmer+1)-0.5) / 8) # on regarde le milieu de la voie associee au nageur
                    h = size_box[0]
                    # to_save = np.zeros(size_box)
                    if x - w//2 >= 0 :
                        to_save = frame[y - h//2:y + h//2, x - w//2:x + w//2]
                        origins.append([num_frame,y - h//2, x - w//2])
                    else :
                        to_save = frame[y - h//2:y + h//2, 0:size_box[1]]
                        origins.append([num_frame, y - h//2, 0])
                    
    
                # write the new image
                out.write(to_save)

        compt += 1
    
    cap.release()
    out.release()
    return origins

def zoom_two_videos(videog, videod, start_timeg, start_timel, swimmer_data, num_swimmer, hm_right, hm_left, save_path, size_box,
                    start_size_vid):
    capg = cv2.VideoCapture(videog)
    capd = cv2.VideoCapture(videod)
    fps = capg.get(cv2.CAP_PROP_FPS)
    time_shiftg = round((start_timeg - 1) * fps)
    time_shiftd = round((start_timel - 1) * fps)
    compt = 0
    # output video
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(save_path, fourcc, fps, size_box)
    new_hm_right = np.linalg.inv(hm_right)
    new_hm_left = np.linalg.inv(hm_left)
    width = int(capg.get(cv2.CAP_PROP_FRAME_WIDTH))  # float `width`
    height = int(capd.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # we read frames until synchronised
    for _ in range(abs(time_shiftg)):
        capg.read()
    for _ in range(abs(time_shiftd)):
        capd.read()
        
    resy=height
    resx=width

    while capd.isOpened() and capg.isOpened():
        retd, framed = capd.read()
        retg, frameg = capg.read()

        if retd is not True or compt >= len(swimmer_data):
            break
        else:
            # zoom
            x = swimmer_data[compt][1]
            to_save = np.zeros((size_box[1], size_box[0], 3)).astype(np.uint8)
            if x != -1 and x < 25:
                # convert x to a position that the homography maps
                # coor vue dessus
                if start_size_vid == 'right':
                    x = (50 - x) * resx / 50
                else:
                    x = x * resx / 50

                w = size_box[1]
                y = resy * (((num_swimmer+1)-0.5) / 8)
                h = size_box[0]
                to_transform = np.float32([[[x, y]]]) #np.array([x, y, 1])
                coord = cv2.perspectiveTransform(to_transform, new_hm_right)
                coor_maind = np.dot(new_hm_right, np.array([x, y, 1]))
                coor_maind = (coor_maind / coor_maind[-1]).astype(int)
                coord = np.squeeze(coord).astype(int)
                x_side, y_side = coor_maind[0], coor_maind[1]
                # to_save = np.zeros(size_box)
                to_save = framed[y_side - h//2:y_side + h//2, x_side - w//2:x_side + w//2]
            elif x != -1:
                # convert x to a position that the homography maps
                # coor vue dessus
                if start_size_vid == 'right':
                    x = (50 - x) * resx / 50
                else:
                    x = x * resx / 50

                w = size_box[1]
                y = resy * (((num_swimmer+1)-0.5) / 8)
                h = size_box[0]
                to_transform = np.float32([[[x, y]]])  # np.array([x, y, 1])
                coorg = cv2.perspectiveTransform(to_transform, new_hm_left)
                coor_maing = np.dot(new_hm_left, np.array([x, y, 1]))
                coor_maing = (coor_maing / coor_maing[-1])
                coorg = np.squeeze(coorg).astype(int)
                x_side, y_side = coorg[0], coorg[1]
                # to_save = np.zeros(size_box)
                to_save = frameg[y_side - h // 2:y_side + h // 2, x_side - w // 2:x_side + w // 2]
                
        # write the new image
        out.write(to_save)

        compt += 1

    capd.release()
    capg.release()
    out.release()
    
    
def zoom_two_videos_mask(videog, videod, start_timeg, start_timel, swimmer_data, num_swimmer, hm_right, hm_left, save_path, size_box,
                    start_size_vid):
    capg = cv2.VideoCapture(videog)
    capd = cv2.VideoCapture(videod)
    fps = capg.get(cv2.CAP_PROP_FPS)
    time_shiftg = round((start_timeg - 1) * fps)
    time_shiftd = round((start_timel - 1) * fps)
    width  =int( capd.get(cv2.CAP_PROP_FRAME_WIDTH) ) 
    height =int( capd.get(cv2.CAP_PROP_FRAME_HEIGHT) ) 
    compt = 0
    # output video
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(save_path, fourcc, fps, size_box)
    new_hm_right = np.linalg.inv(hm_right)
    new_hm_left = np.linalg.inv(hm_left)
    
    
    width = int(capg.get(cv2.CAP_PROP_FRAME_WIDTH))  # float `width`
    height = int(capd.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # we read frames until synchronised
    for _ in range(abs(time_shiftg)):
        capg.read()
    for _ in range(abs(time_shiftd)):
        capd.read()
        
    w = size_box[1]
    y = resy * (((num_swimmer+1)-0.5) / 8)
    h = size_box[0]
    maskd = np.zeros((height, width),dtype=np.uint8)
    maskg = np.zeros((height, width),dtype=np.uint8)

    kernel = np.ones((5,5),np.uint8)
    

    for i in range(int(max(0,y-h//2)),int(min(y+h//2, maskd.shape[0]))):
        for j in range(0,maskd.shape[1]):
            to_transform = np.float32([[[j, i]]])
            coord = cv2.perspectiveTransform(to_transform, new_hm_right)
            coor_maind = np.dot(new_hm_right, np.array([j, i, 1]))
            coor_maind = (coor_maind / coor_maind[-1]).astype(int)
            coord = np.squeeze(coord).astype(int)
            xx,yy = coor_maind[0], coor_maind[1]
            

            
            if  xx>=0 and yy >=0 and xx<maskd.shape[1] and yy<maskd.shape[0] :
                maskd[yy,xx] = 255
                
            coorg = cv2.perspectiveTransform(to_transform, new_hm_left)
            coor_maing = np.dot(new_hm_left, np.array([j, i, 1]))
            coor_maing = (coor_maing / coor_maing[-1]).astype(int)
            coorg = np.squeeze(coorg).astype(int)
            xx,yy = coor_maing[0], coor_maing[1]
            
            if  xx>=0 and yy >=0 and xx<maskg.shape[1] and yy<maskg.shape[0] :
                maskg[yy,xx] = 255
    
    maskd = cv2.dilate(maskd,kernel,iterations = 1)
    maskg = cv2.dilate(maskg,kernel,iterations = 1)

    k=1
    while capd.isOpened() and capg.isOpened():
        retd, framed = capd.read()
        retg, frameg = capg.read()

        print(k)
        if retd is not True or compt >= len(swimmer_data):
            break
        
        else:

            # zoom
            x = swimmer_data[compt][1] #distance parcourue par le nageur en m
            to_save = np.zeros((size_box[1], size_box[0], 3)).astype(np.uint8)
            
            if x != -1 and x < 25:
                # convert x to a position that the homography maps
                # coor vue dessus
                if start_size_vid == 'right':
                    x = (50 - x) * resx / 50
                else:
                    x = x * resx / 50

                w = size_box[1]
                y = resy * (((num_swimmer+1)-0.5) / 8) #milieu de la ligne sur laquelle zoomer, /8 car 8 lignes
                h = size_box[0]
                
            
                to_transform = np.float32([[[x, y]]]) #np.array([x, y, 1])
                coord = cv2.perspectiveTransform(to_transform, new_hm_right)
                coor_maind = np.dot(new_hm_right, np.array([x, y, 1]))
                coor_maind = (coor_maind / coor_maind[-1]).astype(int)
                coord = np.squeeze(coord).astype(int)
                x_side, y_side = coor_maind[0], coor_maind[1]
                
                to_save = cv2.bitwise_and(framed[y_side - h // 2:y_side + h // 2, x_side - w // 2:x_side + w // 2],
                                          framed[y_side - h // 2:y_side + h // 2, x_side - w // 2:x_side + w // 2],
                                          mask = maskd[y_side - h // 2:y_side + h // 2, x_side - w // 2:x_side + w // 2])
                
            elif x != -1:
                # convert x to a position that the homography maps
                # coor vue dessus
                if start_size_vid == 'right':
                    x = (50 - x) * resx / 50
                else:
                    x = x * resx / 50

                w = size_box[1]
                y = resy * (((num_swimmer+1)-0.5) / 8)
                h = size_box[0]
                
                to_transform = np.float32([[[x, y]]])  # np.array([x, y, 1])
                coorg = cv2.perspectiveTransform(to_transform, new_hm_left)
                coor_maing = np.dot(new_hm_left, np.array([x, y, 1]))
                coor_maing = (coor_maing / coor_maing[-1])
                coorg = np.squeeze(coorg).astype(int)
                x_side, y_side = coorg[0], coorg[1]

              
                to_save = cv2.bitwise_and(frameg[y_side - h // 2:y_side + h // 2, x_side - w // 2:x_side + w // 2],
                                          frameg[y_side - h // 2:y_side + h // 2, x_side - w // 2:x_side + w // 2],
                                          mask = maskg[y_side - h // 2:y_side + h // 2, x_side - w // 2:x_side + w // 2])
                
        # write the new image
        out.write(to_save)

        compt += 1
        k=k+1

    capd.release()
    capg.release()
    out.release()
    



if __name__ == '__main__':
    
    #above ='2021_Marseille_brasse_hommes_50_finaleA_from_above.mp4'
    #gauche ='2021_Marseille_brasse_hommes_50_finaleA_fixeGauche.mp4'
    #droite ='2021_Marseille_brasse_hommes_50_finaleA_fixeDroite.mp4'
    #csvv = '2021_Marseille_brasse_hommes_50_finaleA_automatique.csv'
    #jjson = '2021_Marseille_brasse_hommes_50_finaleA.json'
    
    #above ='2021_Marseille_brasse_dames_50_finaleA_from_above.mp4'
    #gauche ='2021_Marseille_brasse_dames_50_finaleA_fixeGauche.mp4'
    #droite ='2021_Marseille_brasse_dames_50_finaleA_fixeDroite.mp4'
    #csvv = '2021_Marseille_brasse_dames_50_finaleA_automatique.csv'
    #jjson = '2021_Marseille_brasse_dames_50_finaleA.json'

    #above ='2015_Kazan_brasse_dames_50_finale_from_above.mp4'
    #gauche ='2015_Kazan_brasse_dames_50_finale_fixeGauche.mp4'
    #droite ='2015_Kazan_brasse_dames_50_finale_fixeDroite.mp4'
    #csvv = '2015_Kazan_brasse_dames_50_finale_automatique.csv'
    #jjson = '2015_Kazan_brasse_dames_50_finale.json'
    
    above ='2021_Marseille_papillon_dames_50_finaleA_from_above.mp4'
    gauche ='2021_Marseille_papillon_dames_50_finaleA_fixeGauche.mp4'
    droite ='2021_Marseille_papillon_dames_50_finaleA_fixeDroite.mp4'
    csvv = '2021_Marseille_papillon_dames_50_finaleA_automatique.csv'
    jjson = '2021_Marseille_papillon_dames_50_finaleA.json'
    
    video = 'videos/'+above
    videog = 'videos/'+gauche
    videod = 'videos/'+droite
    csv = 'videos/'+csvv
    json_path = 'videos/'+jjson
    with open(json_path) as json_file:
        json_course = json.load(json_file)
    index_vid = get_index(json_course['videos'],above)
    start_time = json_course['videos'][index_vid]['start_moment']
    index_vidg = get_index(json_course['videos'], gauche)
    index_vidd = get_index(json_course['videos'], droite)
    src_ptsg = np.float32(json_course['videos'][index_vidg]["srcPts"])
    dest_ptsg = np.float32(json_course['videos'][index_vidg]["destPts"])
    src_ptsd = np.float32(json_course['videos'][index_vidd]["srcPts"])
    dest_ptsd = np.float32(json_course['videos'][index_vidd]["destPts"])

    size_box = (128, 128)
    data = pd.read_csv(csv)  # id, frame_number, swimmer, x1, x2, y1, y2, event, cycles
    data = data.to_numpy()
    all_swimmers = [[] for i in range(8)]
    for i in range(8):
        all_swimmers[i] = np.squeeze(data[np.argwhere(data[:, 2] == i)])[:, (1, 3)]
    all_swimmers = np.array(all_swimmers)
    

    num_swimmer = 5 #entre 0 et 7 
    swimmer = all_swimmers[num_swimmer] #Choose the swimmer
    #♠origines = zoom(video, start_time, swimmer, num_swimmer, 'videos/zoom_papillon.mp4', size_box)

    # we need to convert the points of the calibration to make them correspond the destination image size
    shape_output_img = (resx, resy)
    size_image_ref = (900, 360)
    dest_ptsg[:, 0] = dest_ptsg[:, 0] * shape_output_img[0] / size_image_ref[0]
    dest_ptsg[:, 1] = dest_ptsg[:, 1] * shape_output_img[1] / size_image_ref[1]
    dest_ptsd[:, 0] = dest_ptsd[:, 0] * shape_output_img[0] / size_image_ref[0]
    dest_ptsd[:, 1] = dest_ptsd[:, 1] * shape_output_img[1] / size_image_ref[1]

    # generating the homography matrices
    hm_left = cv2.getPerspectiveTransform(src_ptsg, dest_ptsg)
    hm_right = cv2.getPerspectiveTransform(src_ptsd, dest_ptsd)

    start_timeg = json_course['videos'][index_vidg]['start_moment']
    start_timel = json_course['videos'][index_vidd]['start_moment']
    # side where the swimmers start on the video
    start_side = json_course['videos'][index_vidg]['start_side']
    zoom_two_videos_mask(videog, videod,start_timeg, start_timel, swimmer, num_swimmer, hm_right, hm_left, 'videos/zoom_good.mp4',size_box, start_side)

    x = resx
    y = resy
    to_transform = np.array([x, y, 1])
    new_hm = np.linalg.inv(hm_right)
    coor = np.dot(new_hm, to_transform)
    coor = coor / coor[-1]
    coor = coor.astype(np.uint8)
    print(coor)

    points = np.float32([[[x, y]]])
    detransformed = cv2.perspectiveTransform(points, new_hm)
    print(np.squeeze(detransformed), np.squeeze(detransformed).astype(int))
    
#%%
origines_each_swimmer = []
for num in range(3,8) :
    origines_each_swimmer.append(zoom(video, start_time, all_swimmers[num], num, 'videos/zoom'+'swimmer'+str(num)+'papillon'+'.mp4', size_box))

for num in range(8):
    print(len(origines_each_swimmer[num]))
