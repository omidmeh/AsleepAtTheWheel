# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 18:01:46 2019

@author: OMIDMEH
"""

#%% Imports
from imutils.video import FPS
import numpy as np

from imutils.video import FileVideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2
import pandas as pd


#%% Variables

path_haar = r'build\\etc\\haarcascades\\'

path_pred_5  = r'facial-landmarks/shape_predictor_5_face_landmarks.dat'
path_pred_68 = r'facial-landmarks/shape_predictor_68_face_landmarks.dat'
path_pred = path_pred_68

#%% Setup
face_cascade = cv2.CascadeClassifier(path_haar + 'haarcascade_frontalface_default.xml') 
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(path_pred)


print("[INFO] starting video file thread...")
#fvs = FileVideoStream(args["video"]).start()
fvs = FileVideoStream(r'raw_data\\Fold3_part2\\33\\0.mp4').start()
time.sleep(1.0)
 
# start the FPS timer
final_list = []
header = []
header.append("participant")
header.append("mood")
header.append('fps')
header.append('frame_no')
header.append('time')

header.append('face_x')
header.append('face_y')
header.append('face_w')
header.append('face_h')

for i in range(68):
    header.append('px_'+str(i+1))
    header.append('py_'+str(i+1))

final_list.append(header)

#%% Processes

def process(frame):
    return_list = []
    # reads frames from a camera 
    img = frame  
  
    # convert to gray scale of each frames 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
  
    
    #CV2
#    faces = face_cascade.detectMultiScale(gray, 1.2, 5) 
#    for (x,y,w,h) in faces: 
    # DLIB2
    rects = detector(gray, 0)
    if len(rects) == 0:
        return_list = ['-1' for x in range(4+68)]
    if len(rects) > 1 : 
        rects = [rects[0]]
    
    for (x,y,w,h) in [face_utils.rect_to_bb(x) for x in rects]:
        return_list += [x, y, w, h]
#        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)  
        roi_gray = gray[y:y+h, x:x+w] 
        roi_color = img[y:y+h, x:x+w]
        
        shape_img = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))
        shape = predictor(img, shape_img)
        shape = face_utils.shape_to_np(shape)
#        
#        final_list.append(shape)
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        if len(shape) ==0:
            return_list += ['-1' for x in range(68)]
        else:    
            assert(len(shape) == 68)
            return_list += [item for sublist in shape.tolist() for item in sublist]
#            for (xx, yy) in shape:
#                cv2.circle(img, (xx, yy), 1, (0, 0, 255), -1)

        
    # Display an image in a window 
#    cv2.imshow('img',img) 
#    cv2.imshow('gray cut',roi_gray) 
    return return_list
    
#%% Merged

participent = 33
mood = 0

cap = cv2.VideoCapture(f'raw_data\\Fold3_part2\\{participent}\\{mood}.mp4')
video_fps = cap.get(cv2.CAP_PROP_FPS)
video_frame_length = 1/video_fps
frame_no = 0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
result_df = pd.DataFrame(columns=header, data=np.empty(shape=(total_frames-1, 68*2+4+5)))
idx = 0

fps = FPS().start()
for i in range(frame_no, total_frames - 1):
    frame_row = []
    assert(fvs.more())
    frame = fvs.read()
    assert(frame is not None)
    frame = imutils.resize(frame, width=350)
    frame = imutils.rotate_bound(frame, -90)
    
    frame_row.append(participent)
    frame_row.append(mood)
    frame_row.append(video_fps)
    frame_row.append(frame_no)
    frame_row.append(frame_no*video_frame_length)
    
    if (frame_no % 50) == 0: print(frame_no)
  
    frame_row += process(frame)
    result_df.iloc[idx] = frame_row
    
    
    # Press Q on keyboard to  exit 
#    if cv2.waitKey(25) & 0xFF == ord('q'): 
#        break

    idx += 1
    frame_no += 1
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
 
# do a bit of cleanup
cv2.destroyAllWindows()
fvs.stop()

