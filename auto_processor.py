# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 15:45:50 2019

@author: OMIDMEH
"""

#%% Imports
from imutils.video import FPS
import numpy as np

from imutils.video import FileVideoStream
from imutils import face_utils
import imutils
import time
import dlib
import cv2
import pandas as pd
import glob,os.path
from moviepy.editor import VideoFileClip
import traceback
import argparse

#%% Arguments
# initiate the parser
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--directory", help="path to start from")

# read arguments from the command line
args = parser.parse_args()

if args.directory:
    print(args.directory)

#%% Configuration
path_haar = os.path.join('build','etc','haarcascades')

path_pred_5  = os.path.join('facial-landmarks','shape_predictor_5_face_landmarks.dat')
path_pred_68 = os.path.join('facial-landmarks','shape_predictor_68_face_landmarks.dat')

path_pred = path_pred_68
landmark_pts_count = 68

header = ['participant', 'mood', 'fps', 
          'size_x', 'size_y', 
          'frame_no', 'time', 'face_x','face_y','face_w','face_h']

width = 350
progress_report = None

csv_output_path = os.path.join('.', 'output', 'csv')
report_output_path = os.path.join('.', 'output', 'report')

#%% Initialization
face_cascade = cv2.CascadeClassifier(os.path.join(path_haar, 'haarcascade_frontalface_default.xml'))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(path_pred)


## Table Header
for i in range(landmark_pts_count):
    header.append('px_'+str(i+1))
    header.append('py_'+str(i+1))

final_list = []
final_list.append(header)

for directory in [csv_output_path, report_output_path]:
    if not os.path.exists(directory):
        os.makedirs(directory)
#%% data_class

class participent:
    def __init__(self, participent, location):
        self.location = location
        self.participant_no = participent

        
    def get_video_metadata(self, video_path):
        c = VideoFileClip(video_path)
        rotation = c.rotation
        fps = c.fps
        if rotation in [0,180]:
            size_x = c.size[0]
            size_y = c.size[1]
        else:
            size_x = c.size[1]
            size_y = c.size[0]
        c.close()
        
        print(rotation, fps, size_x, size_y)
        return rotation, fps, size_x, size_y
       
        
    def process_all_moods(self, width, progress_report):
        #Find all that needs processesing
        videos = glob.glob(os.path.join(self.location,'*'))
        for video_path in videos:
            print(f"Path: {video_path}")
            try:
                self.process_one_mood(video_path, width, progress_report)
            except Exception as  e:
                print(f"failed processing {video_path}: {e}")
                traceback.print_tb(e.__traceback__)

        
    def process_one_mood(self, video_path, width, progress_report):
        # Reporting results
        dropped_frames = []
        
        # Metadata
        video_rotation, video_fps, size_x, size_y = self.get_video_metadata(video_path)
        print("rotation: ", video_rotation)
        print("fps: ", video_fps)
        print("size_x: ", size_x)
        print("size_y: ", size_y)
        
        assert(video_fps != 0)         

        mood = os.path.basename(video_path).split('.')[0]
        frame_len = 1/video_fps
        
        print("[INFO] starting video file thread...")
        fvs = FileVideoStream(video_path).start()
        cap = cv2.VideoCapture(video_path)
        time.sleep(1.0)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        

        # variables
#        result_df = pd.DataFrame(columns=header, 
#                                 data=np.empty(shape=(total_frames-1, len(header))))
        result_np = np.empty(shape=(total_frames-1, len(header)))
#        
        fps = FPS().start()
        frame_no = 0
        for i in range(frame_no, total_frames - 1):
            frame_row = []
            assert(fvs.more())
            frame = fvs.read()
            assert(frame is not None)
            
            frame = imutils.rotate_bound(frame, video_rotation)
            frame = imutils.resize(frame, width=width)
            
            frame_row.append(self.participant_no)
            frame_row.append(mood)
            frame_row.append(video_fps)
            frame_row.append(size_x)
            frame_row.append(size_y)
            
            frame_row.append(frame_no)
            frame_row.append(frame_no*frame_len)
            
            if progress_report is not None:
                if (frame_no % progress_report) == 0: 
                    print("processing frame: ", frame_no)
          
            frame_row += self.process_frame(frame)
#            result_df.iloc[frame_no] = frame_row
            result_np[frame_no] = frame_row
            if '-1' in frame_row: 
                dropped_frames.append(frame_no)
            
            
            # Press Q on keyboard to  exit 
        #    if cv2.waitKey(25) & 0xFF == ord('q'): 
        #        break
        
            frame_no += 1
            fps.update()
            
        fps.stop()
        print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
        fvs.stop()
        
        # Process one mood
        # display if needed
        result_df = pd.DataFrame(columns=header, data=result_np)
        result_df.to_csv(os.path.join(csv_output_path, f"{self.participant_no}_{mood}.csv"))
        
        with open(os.path.join(report_output_path, f"{self.participant_no}_{mood}.txt"),"w") as report:
            report.write(f"video: {video_path}\n")
            report.write(f"metadata: {video_fps} fps, {video_rotation} deg, {size_x}x{size_y}")
            report.write(f"dropped: {dropped_frames}\n")
            report.write("elasped time: {:.2f}\n".format(fps.elapsed()))
            report.write("approx. FPS: {:.2f}\n".format(fps.fps()))
            
            report.write("-"*20)
        # Save to df
        pass
    
    def process_frame(self, frame):
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
            return_list = ['-1' for x in range(4+(landmark_pts_count*2))]
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
                return_list += ['-1' for x in range(landmark_pts_count * 2)]
            else:    
                assert(len(shape) == landmark_pts_count)
                return_list += [item for sublist in shape.tolist() for item in sublist]
    #            for (xx, yy) in shape:
    #                cv2.circle(img, (xx, yy), 1, (0, 0, 255), -1)
    
            
        # Display an image in a window 
    #    cv2.imshow('img',img) 
    #    cv2.imshow('gray cut',roi_gray) 
        if not (len(return_list) == landmark_pts_count * 2 + 4):
            print(f"len{len(return_list)}")
            print(f'{return_list}')
        return return_list
    
    
#%% Main

## Find all folders
if args.directory:
    filesDepth3 = glob.glob(os.path.join(args.directory,'*'))
else:
    filesDepth3 = glob.glob(os.path.join('raw_data','*'))
dirsDepth3 = filter(lambda f: os.path.isdir(f), filesDepth3)

## Process files
#for path in dirsDepth3:
#    p_no = (os.path.basename(path))
#    p = participent(p_no, path)
#    p.process_all_moods(width, progress_report)
    
    
def run_participant(path):
    print("processing", path)
    p_no = (os.path.basename(path))
    p = participent(p_no, path)
    p.process_all_moods(width, progress_report)
    
from multiprocessing.dummy import Pool as ThreadPool 
print("start pooling")
pool = ThreadPool(4) 
print("Started")
#print([x for x in dirsDepth3])
results = pool.map(run_participant, dirsDepth3)
print(results)

# close the pool and wait for the work to finish 
pool.close() 
pool.join() 
