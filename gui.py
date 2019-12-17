#!/usr/bin/env python
#%%
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
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


## VARIABLES


participant = 31
mood_code = 1

filename = None
filename = f'C:/Users/OMIDMEH/Development/UT/drowsiness_detection/demo/{participant}_{mood_code}_short.mp4'
# filename = r'C:/Users/OMIDMEH/Development/UT/drowsiness_detection/demo/31_0.mp4'
# filename = r'C:/Users/OMIDMEH/Development/UT/drowsiness_detection/demo/31_10.mp4'
# L2

STEP_SIZE = 1
SAMPLES = 300
SAMPLE_MAX = 0.7
CANVAS_SIZE = (500, 100)
NUMBER_MARKER_FREQUENCY = 20

icon_warning = r"./demo/alarm.png"
icon_ok = r"./demo/ok.png"


path_haar = r'build\\etc\\haarcascades\\'
path_pred_5  = r'facial-landmarks/shape_predictor_5_face_landmarks.dat'
path_pred_68 = r'facial-landmarks/shape_predictor_68_face_landmarks.dat'
path_pred = path_pred_68

# Backup Vars
fe_pred_path = './demo/pred_mel.csv'
cnn_pred_path = './demo/pred_jay.csv'


# filename = r'C:/Users/OMIDMEH/Development/UT/drowsiness_detection/demo/00_10.mp4'
# fe_pred_path = './demo/crash_prediction.csv'
# cnn_pred_path = './demo/crash_prediction_cnn.csv'
# participant = 0
# mood_code = 1


# filename = r'C:/Users/OMIDMEH/Development/UT/drowsiness_detection/raw_data/Fold3_part2/31/10.mp4'
# filename = r'C:/Users/OMIDMEH/Development/UT/drowsiness_detection/raw_data/Fold3_part2/00/32_10.mp4'
# filename = r'C:/Users/OMIDMEH/Development/UT/drowsiness_detection/raw_data/Fold3_part2/00/__.mp4'
# filename = r'C:/Users/OMIDMEH/Development/UT/drowsiness_detection/raw_data/Fold3_part2/31/10.mp4'

#%%


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


    info_table_header = ['frame', 'time', 'left_eye_ratio', 'right_eye_ratio', 'mouth_ratio']
    info_table_values = (150,0, 0.5,0.5,0.2)

    info_table = sg.Table(values=[info_table_values],
                        headings=info_table_header,
                        max_col_width=15,
                        auto_size_columns=False,
                        justification='center',
                        # alternating_row_color='lightblue',
                        num_rows=1,
                        hide_vertical_scroll=True,
                        key='infobar')
    pred_table = sg.Table(values=[(0,0)],
                        headings=['FFE+RNN', 'CNN+RNN'],
                        max_col_width=7,
                        auto_size_columns=False,
                        justification='center',
                        # alternating_row_color='lightblue',
                        num_rows=1,
                        hide_vertical_scroll=True,
                        key='predbar')

    eye_graph = sg.Graph(CANVAS_SIZE, (0, 0), (SAMPLES, SAMPLE_MAX), background_color='black', key='graph')
    mouth_graph = sg.Graph(CANVAS_SIZE, (0, 0), (SAMPLES, SAMPLE_MAX*1.5), background_color='black', key='graph2')


    layout2 = [
        [sg.Text('Drowsy Driver Detector', size=(40, 1), font='Helvetica 20')],
        [sg.Text(f'    Participant: {participant}', size=(40, 1), font='Helvetica 12')],
        [sg.Text(f'    Nominal Drowsiness: {drowsy_level_to_string(mood_code)}', size=(40, 1), font='Helvetica 12')],
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
                [sg.Text('Predictions:', size=(50, 1), font='Helvetica 14')],
                [
                    sg.Column([
                        [sg.Canvas(size=(100, 200), key='-bar_graph-')]
                        ]),
                    sg.Column([ 
                        # [sg.Text('  FFN Confidence:', size=(30, 1), font='Helvetica 12', key='-pred_fe-')],
                        [pred_table],
                        [sg.Image(filename=r'./demo/ok.png', key='-pred_icon-', pad=(70,25))],
                        [sg.Text('Result:', pad=(30,0), size=(20, 1), font='Helvetica 18', key='-pred_result-')],
                        ])
                ]
               
                
            ])
        ]
    ]


    # create the window and show it without the plot
    window = sg.Window('Demo Application - OpenCV Integration',
                       layout2,
                       no_titlebar=False,
                       location=(0, 0))
    window.Finalize()
    ##################
    ### Plot Setup ###
    ##################
    canvas_elem = window['-bar_graph-']
    canvas = canvas_elem.TKCanvas
    
    # draw the initial plot in the window
    plt.style.use('dark_background')
    fig = plt.Figure()
    ax = fig.add_subplot(111)
    ax.set_title('Drowsy Level Prediction')
    ax.figure.set_size_inches(3, 2)
    ax.set_ylabel('Confidence')
    # ax.set_xlabel("X axis")
    ax.set_xticks((0,1))
    ax.set_xticklabels(('FFE Confidence', 'CNN Confidence'))
    ax.set_yticks(np.arange(0, 1, 0.1))
    ax.set_ylim(0,1)
    # ax.grid()

    fig_agg = draw_figure(canvas, fig)

    
    
    # values_to_plot = (0.5, 0.5)
    width = 0.4
    # ind = np.arange(len(values_to_plot))

    

    # p1 = plt.bar(ind, values_to_plot, width)
    
    # plt.style.use('dark_background')
    # plt.ylabel('Confidence')
    # plt.title('Drowsy Level Prediction')
    # plt.xticks(ind, ('Drowsy - FFE', 'Drowsy - CNN'))
    # plt.yticks(np.arange(0, 1, 0.1))
    # plt.legend((p1[0],), ('Data Group 1',))

    # fig = plt.gcf()  # if using Pyplot then get the figure from the plot
    # figure_x, figure_y, figure_w, figure_h = fig.bbox.bounds

    ##################
    ### UI Update ####
    ##################

    # locate the elements we'll be updating. Does the search only 1 time
    image_elem = window['-image-']
    icon_elem = window['-pred_icon-']
    slider_elem = window['-slider-']
    info_elem = window['infobar']
    pred_elem = window['predbar']
    
    ## RANDOM DATA FOR GRAPH
    
    graph = window['graph']
    graph2 = window['graph2']
    
    pred_result = window['-pred_result-']
    # pred_fe = window['-pred_fe-']
    # pred_cnn = window['-pred_cnn-']
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

    prev_pred_ffe = 0
    prev_pred_cnn = 0

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
                info_elem.Update([(cur_frame,str(datetime.timedelta(seconds=round(cur_frame/fps))), f'{ratio_eye_l:.2f}',f'{ratio_eye_r:.2f}',f'{ratio_mouth:.2f}')])
                
                
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

        # Plot results
        if  prediction_drowsy_fe != prev_pred_ffe or prediction_drowsy_cnn != prev_pred_cnn:
            # plt.close()

            ind = np.arange(len((prediction_drowsy_fe, prediction_drowsy_cnn)))
            # p1 = plt.bar(ind, (prediction_drowsy_fe, prediction_drowsy_cnn), width)
            
            # plt.style.use('dark_background')
            # plt.ylabel('Confidence')
            # plt.title('Drowsy Level Prediction')
            # plt.xticks(ind, ('Drowsy - FFE', 'Drowsy - CNN'))
            # plt.yticks(np.arange(0, 1, 0.1))
            # plt.legend((p1[0],), ('Data Group 1',))

            # fig = plt.gcf()  # if using Pyplot then get the figure from the plot
            # fig.set_size_inches(3,2)

            # canvas = window['-bar_graph-'].TKCanvas
            # figure_canvas_agg = FigureCanvasTkAgg(fig, canvas)
            # figure_canvas_agg.draw()
            # figure_canvas_agg.get_tk_widget().
            # figure_canvas_agg.get_tk_widget().pack_forget()
            # # figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
            # figure_canvas_agg.get_tk_widget().pack()
            # return figure_canvas_agg

            ax.cla()                    # clear the subplot
            # ax.grid()                   # draw the grid
            ax.set_title('Drowsy Level Prediction')
            ax.set_ylabel('Confidence')
            # ax.set_xlabel("X axis")
            ax.set_ylim(0,1)
            ax.set_yticks(np.arange(0, 1, 0.1))
            ax.bar(ind, (prediction_drowsy_fe, prediction_drowsy_cnn), width)
            ax.set_xticks((0,1))
            ax.set_xticklabels(('FFE Confidence', 'CNN Confidence'))
            # ax.figure.set_size_inches(2, 3)
            fig_agg.draw()

            if prediction_drowsy_fe > 0.5 or prediction_drowsy_cnn > 0.5:
                icon_elem.Update(filename=icon_warning)
                pred_result.Update('Result: Drowsy')
            else:
                icon_elem.Update(filename=icon_ok)
                pred_result.Update('Result: Alert')


            
            # fig_photo = draw_figure(window['-bar_graph-'].TKCanvas, fig)

        prev_pred_ffe = prediction_drowsy_fe
        prev_pred_cnn = prediction_drowsy_cnn
        
        # print(prediction_drowsy)
        # pred_fe.Update(f'  FFN Confidence: {prediction_drowsy_fe:.2f} | {drowsy_level_to_string(prediction_drowsy_fe)}')
        # pred_cnn.Update(f'  CNN Confidence: {prediction_drowsy_cnn:.2f} | {drowsy_level_to_string(prediction_drowsy_cnn)}')
        pred_elem.Update([(f'{prev_pred_ffe:.3f}', f'{prev_pred_cnn:.3f}')])


        frame_resized = imutils.resize(frame, width=300)
        # img = cv.imencode('.png', frame)[1].tobytes()
        imgbytes = cv.imencode('.png', frame_resized)[1].tobytes()  # ditto
        image_elem.Update(data=imgbytes)


  




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

#%%
class Predictions():
    def __init__(self, participant, mood, fps, filepath, f=False):
        self.participant = participant
        self.fps = fps
        self.mood = mood
        self.f = f

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

def draw_figure(canvas, figure, loc=(0, 0)):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

#%%
main2()