#!/usr/bin/env python
import PySimpleGUI as sg
from PIL import Image
import cv2 as cv
import io
import imutils
import pandas as pd
import random
import dlib
from imutils import face_utils
import numpy as np
import datetime


## VARIABLES
participant = 31
drowsiness = 'drowsy'
mood_code = 0

fe_pred_path = './prediction.csv'
cnn_pred_path = './prediction.csv'

filename = None
filename = r'C:/Users/OMIDMEH/Development/UT/drowsiness_detection/raw_data/Fold3_part2/31/10.mp4'
filename = r'C:/Users/OMIDMEH/Development/UT/drowsiness_detection/raw_data/Fold3_part2/00/32_10.mp4'
filename = r'C:/Users/OMIDMEH/Development/UT/drowsiness_detection/raw_data/Fold3_part2/00/__.mp4'

# L2

STEP_SIZE = 1
SAMPLES = 300
SAMPLE_MAX = 0.5
CANVAS_SIZE = (500, 100)
NUMBER_MARKER_FREQUENCY = 20

path_haar = r'build\\etc\\haarcascades\\'
path_pred_5  = r'facial-landmarks/shape_predictor_5_face_landmarks.dat'
path_pred_68 = r'facial-landmarks/shape_predictor_68_face_landmarks.dat'
path_pred = path_pred_68


def dist(mx, my ,nx, ny):
    return np.sqrt(np.square(mx-nx) + np.square(my-ny))

def mid(x1, x2):
    return (x1+x2)/2

def ratio_6(table, t1,t2,b1,b2,l,r):
    x1_m= mid(table[t1][0], table[t2][0])
    y1_m = mid(table[t1][1], table[t2][1])
    x2_m = mid(table[b1][0], table[b2][0])
    y2_m = mid(table[b1][1], table[b2][1])

    return dist(x1_m,y1_m,x2_m,y2_m) / dist(table[l][0], table[l][1], table[r][0], table[r][1])

def ratio_4(table,t,b,l,r):
    return(dist(table[t][0],table[t][1],table[b][0],table[b][1]) / dist(table[l][0], table[l][1], table[r][0], table[r][1]))



def main2():
    
    ##################
    ### Video Prep ###
    ##################
    # ---===--- Get the filename --- #
    global filename
    if filename is None:
        filename = sg.popup_get_file('Filename to play')
        if filename is None:
            return
    
    vidFile = cv.VideoCapture(filename)
    # ---===--- Get some Stats --- #
    num_frames = vidFile.get(cv.CAP_PROP_FRAME_COUNT)
    fps = vidFile.get(cv.CAP_PROP_FPS)
    p_fe = Predictions(participant, mood_code, fps, fe_pred_path)
    p_cnn = Predictions(participant, mood_code, fps, cnn_pred_path)

    
    ##################
    ### Detectors ####
    ##################
    face_cascade = cv.CascadeClassifier(path_haar + 'haarcascade_frontalface_default.xml') 
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(path_pred)


    ##################
    ### UI Setup #####
    ##################
    sg.change_look_and_feel('Black')
    sg.SetOptions(element_padding=(5, 5))


    info_table_header = ['frame', 'left_eye_ratio', 'right_eye_ratio', 'mouth_ratio']
    info_table_values = (150,0.5,0.5,0.2)

    info_table = sg.Table(values=[info_table_values],
                        headings=info_table_header,
                        max_col_width=15,
                        auto_size_columns=False,
                        justification='center',
                        # alternating_row_color='lightblue',
                        num_rows=1,
                        hide_vertical_scroll=True,
                        key='infobar')

    eye_graph = sg.Graph(CANVAS_SIZE, (0, 0), (SAMPLES, SAMPLE_MAX), background_color='black', key='graph')
    mouth_graph = sg.Graph(CANVAS_SIZE, (0, 0), (SAMPLES, SAMPLE_MAX*1.5), background_color='black', key='graph2')


    layout2 = [
        [sg.Text('Drowsy Driver Detector', size=(40, 1), font='Helvetica 20')],
        [sg.Text(f'    Participant: {participant}', size=(40, 1), font='Helvetica 12')],
        [sg.Text(f'    Nominal Drowsiness: {drowsiness}', size=(40, 1), font='Helvetica 12')],
        [
            sg.Column([
                [sg.Image(filename=r'C:\Users\OMIDMEH\Pictures\PC_Wallpaper_gray.png', key='-image-')],
                [sg.Slider(range=(0, num_frames), # Num Frames,
                        size=(40, 10), orientation='h', key='-slider-')],
                [sg.Button('Exit', size=(7, 1), pad=((120, 0), 3), font='Helvetica 14')]
            ]),
            sg.Column([
                [info_table],
                [sg.Text('Aspect Ratio: Eyes', size=(50, 1), font='Helvetica 8')],
                [eye_graph],
                [sg.Text('Aspect Ratio: Mouth', size=(50, 1), font='Helvetica 8')],
                [mouth_graph],
                [sg.Text('Predictions:', size=(50, 1), font='Helvetica 16')],
                [sg.Text('  Feature Engineering:', size=(50, 1), font='Helvetica 14', key='-pred_fe-')],
                [sg.Text('  Inception + RNN:    ', size=(50, 1), font='Helvetica 14', key='-pred_cnn-')],
            ])
        ]
    ]


    # create the window and show it without the plot
    window = sg.Window('Demo Application - OpenCV Integration',
                       layout2,
                       no_titlebar=False,
                       location=(0, 0))


    ##################
    ### UI Update ####
    ##################

    # locate the elements we'll be updating. Does the search only 1 time
    image_elem = window['-image-']
    slider_elem = window['-slider-']
    info_elem = window['infobar']
    
    ## RANDOM DATA FOR GRAPH
    
    graph = window['graph']
    graph2 = window['graph2']
    
    pred_fe = window['-pred_fe-']
    pred_cnn = window['-pred_cnn-']
    # graph.erase()
    # draw_axis(graph)




    # ---===--- LOOP through video file by frame --- #
    cur_frame = 0
    i = 0
    # prev_x, prev_y = 0, 0
    # graph_value = 250

    prev_x = 0
    prev_eye_r = 0
    prev_mouth = 0

    while vidFile.isOpened():
        event, values = window.read(timeout=0)
        if event in ('Exit', None):
            break
        ret, frame = vidFile.read()
        if not ret:  # if out of data stop looping
            break
        # if someone moved the slider manually, the jump to that frame
        if int(values['-slider-']) != cur_frame-1:
            cur_frame = int(values['-slider-'])
            vidFile.set(cv.CAP_PROP_POS_FRAMES, cur_frame)
        slider_elem.update(cur_frame)
        cur_frame += 1
        
        


        ## VIDEO PROCESSING
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) 
        rects = detector(gray, 0)
        
        for (x,y,w,h) in [face_utils.rect_to_bb(x) for x in rects]:
            # cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),5)  
            # roi_gray = gray[y:y+h, x:x+w] 
            # roi_color = img[y:y+h, x:x+w]
            
            shape_img = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))
            shape = predictor(frame, shape_img)
            shape = face_utils.shape_to_np(shape)

            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            for (xx, yy) in shape:
                cv.circle(frame, (xx, yy), 1, (0, 0, 255), 1)

            if len(shape) > 0:
                ratio_eye_l = ratio_6(shape, 37, 38, 40, 41, 36, 39)
                ratio_eye_r = ratio_6(shape, 43, 44, 46, 47, 42, 45)
                ratio_mouth = ratio_4(shape, 51, 57, 48, 54)
                info_elem.Update([(cur_frame,f'{ratio_eye_l:.2f}',f'{ratio_eye_r:.2f}',f'{ratio_mouth:.2f}')])
                
                
                if i >= SAMPLES:
                    graph.move(-STEP_SIZE, 0)
                    graph2.move(-STEP_SIZE, 0)
                    prev_x = prev_x - STEP_SIZE

                graph.draw_line((prev_x, prev_eye_r), (i, ratio_eye_l), color='white')
                graph2.draw_line((prev_x, prev_mouth), (i, ratio_mouth), color='white')

                prev_x = i
                prev_eye_r = ratio_eye_l
                prev_mouth = ratio_mouth
                i += STEP_SIZE if i < SAMPLES else 0

        prediction_drowsy_fe = p_fe.get_pred(cur_frame)['drowsy']
        prediction_drowsy_cnn = p_cnn.get_pred(cur_frame)['drowsy']

        
        # print(prediction_drowsy)
        pred_fe.Update(f'  Feature Engineering: {prediction_drowsy_fe:.2f} | {drowsy_level_to_string(prediction_drowsy_fe)}')
        pred_cnn.Update(f'  Inception Net + RNN: {prediction_drowsy_cnn:.2f} | {drowsy_level_to_string(prediction_drowsy_cnn)}')


        frame_resized = imutils.resize(frame, width=300)
        # img = cv.imencode('.png', frame)[1].tobytes()
        imgbytes = cv.imencode('.png', frame_resized)[1].tobytes()  # ditto
        image_elem.update(data=imgbytes)


  




def draw_axis(graph):
    graph.draw_line((-CANVAS_SIZE[0], 0), (CANVAS_SIZE[0], 0))                # axis lines
    graph.draw_line((0, -CANVAS_SIZE[1]), (0, CANVAS_SIZE[1]))

    for x in range(-CANVAS_SIZE[0], CANVAS_SIZE[0]+1, NUMBER_MARKER_FREQUENCY):
        graph.draw_line((x, -0.5), (x, 0.5))                       # tick marks
        if x != 0:
            # numeric labels
            graph.draw_text(str(x), (x, 0), color='green')

    for y in range(-CANVAS_SIZE[1], CANVAS_SIZE[1]+1, NUMBER_MARKER_FREQUENCY):
        graph.draw_line((-0.5, y), (0.5, y))
        if y != 0:
            graph.draw_text(str(y), (0, y), color='blue')


class Predictions():
    def __init__(self, participant, mood, fps, filepath):
        self.participant = participant
        self.fps = fps
        self.mood = mood

        self.df = pd.read_csv(filepath)
        self.df = self.df[self.df.participant == self.participant]
        self.df = self.df[self.df.mood == self.mood]
        self.df['date'] = pd.to_datetime(self.df.date)
        self.df.set_index('date', inplace = True)


    def get_pred(self, frame_no):
        sec = frame_no / self.fps
        t = datetime.datetime(1970,1,1)
        t += datetime.timedelta(seconds=sec)

        try:
            idx = self.df.index.get_loc(t,method='nearest')
            # print('t,', t)
            # print('index,', idx)
            return self.df.iloc[idx]
        except Exception as ex:
            print(ex)
            return {'drowsy' : -1}


def drowsy_level_to_string(level, threshold = 0.50):
    if level >= threshold:
        return "Drowsy"
    else:
        return "Awake"

main2()