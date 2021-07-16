import cv2
import numpy as np
import dlib
import pandas as pd
import time
from math import hypot


detector=dlib.get_frontal_face_detector()
cap=cv2.VideoCapture(0)
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def create_line_hor(landmarks,value0,value1):
    x = landmarks.part(value0).x
    y = landmarks.part(value0).y

    x1 = landmarks.part(value1).x
    y1 = landmarks.part(value1).y
    length = hypot((x1 - x), (y1 - y))
    cv2.line(frame, (x, y), (x1, y1), (0, 255, 0), 2)
    return length


def create_line_ver(landmarks, param, param1, param2, param3):
    x = landmarks.part(param).x
    y = landmarks.part(param).y

    x1 = landmarks.part(param1).x
    y1 = landmarks.part(param1).y

    x1mid=int((x+x1)/2)
    y1mid=int((y+y1)/2)

    x = landmarks.part(param2).x
    y = landmarks.part(param2).y

    x1 = landmarks.part(param3).x
    y1 = landmarks.part(param3).y

    x2mid = int((x + x1) / 2)
    y2mid = int((y + y1) / 2)
    length=hypot((x2mid-x1mid),(y2mid-y1mid))
    cv2.line(frame, (x1mid, y1mid), (x2mid, y2mid), (0, 255, 0), 2)
    return length
font=cv2.FONT_HERSHEY_COMPLEX


def detect_eye_gaze(param, param1, param2, param3, param4, param5, s):
    eye_region = np.array([(landmarks.part(param).x, landmarks.part(param).y),
                                 (landmarks.part(param1).x, landmarks.part(param1).y),
                                 (landmarks.part(param2).x, landmarks.part(param2).y),
                                 (landmarks.part(param3).x, landmarks.part(param3).y),
                                 (landmarks.part(param4).x, landmarks.part(param4).y),
                                 (landmarks.part(param5).x, landmarks.part(param5).y)])
    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [eye_region], True, 255, 2)
    cv2.fillPoly(mask, [eye_region], 255)
    left_eye = cv2.bitwise_and(gray, gray, mask=mask)

    xmin = np.min(eye_region[:, 0])
    xmax = np.max(eye_region[:, 0])
    ymax = np.max(eye_region[:, 1])
    ymin = np.min(eye_region[:, 1])

    gray_eye = left_eye[ymin: ymax, xmin: xmax]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)

    threshold_eye = cv2.resize(threshold_eye, None, fx=5, fy=5)
    cv2.imshow(s, threshold_eye)
    return threshold_eye


def get_ratio_threshold(threshold):
    height,width=threshold.shape
    left_threshold=threshold[0: height,0: int(width/2)]
    left_side_white=cv2.countNonZero(left_threshold)
    right_threshold=threshold[0:height,int(width/2):width]
    right_side_white=cv2.countNonZero(right_threshold)
    if right_side_white==0:
        ratio=0
    else:
        ratio=left_side_white/right_side_white
    return ratio


def checkerror(df,s):
    error=0
    if s in df[['Looking Direction']].tail(12).values:
        error = 0
    else:
        error=1
    return error



while True:
    blink=0
    data = pd.read_excel(r'Data4_Morning.xlsx')
    df = pd.DataFrame(data)
    eye_direc=""
    car_direc=""
    error=0
    key = cv2.waitKey(1)
    av=0
    _, frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces=detector(gray)
    for face in faces:
        x, y=face.left(), face.top()
        x1,y1=face.right(),face.bottom()
        cv2.rectangle(frame,(x,y),(x1,y1),(0,123,0),5)

        landmarks=predictor(gray,face)
        left_eye_hor_length=create_line_hor(landmarks,36,39)
        right_eye_hor_length=create_line_hor(landmarks, 42, 45)

        right_eye_vert_length=create_line_ver(landmarks,43,44,47,46)
        left_eye_vert_length=create_line_ver(landmarks, 37, 38, 40, 41)

        #Eye blink
        ratio_left=left_eye_hor_length/left_eye_vert_length
        ratio_right=right_eye_hor_length/right_eye_vert_length
        avg_ratio=(ratio_right+ratio_left)/2
        if(avg_ratio>4.5):
            cv2.putText(frame,"Blink",(50,150),font,3,(255,0,0))
            blink=1

        #Gaze Detection
        threshold_right_eye=detect_eye_gaze(36,37,38,39,40,41,"Right Eye")
        threshold_left_eye=detect_eye_gaze(42, 43, 44, 45, 46, 47, "Left Eye")

        ratio_left_eye=get_ratio_threshold(threshold_left_eye)
        ratio_right_eye = get_ratio_threshold(threshold_right_eye)
        avg_ratio_dir=(ratio_left_eye+ratio_right_eye)/2



        av=avg_ratio_dir
        if 0.1<avg_ratio_dir<0.65:
            eye_direc="right"
            cv2.putText(frame, "right", (50, 100), font, 2, (0, 0, 225, 3))
            #df.append({'Time' : 'asd' , 'Ratio' : avg_ratio_dir , 'Looking Direction' : 'right', 'Blinking' : blink , 'Driving Direction' : 'forward', 'Error' : 0} , ignore_index=True)

            #print(avg_ratio_dir,"right ",time.strftime("%H:%M:%S", t))
        elif 0.65<avg_ratio_dir<1.2:
            cv2.putText(frame, "center", (50, 100), font, 2, (0, 0, 225, 3))
            eye_direc ="center"

            #df.append({'Time': 'asd', 'Ratio': avg_ratio_dir, 'Looking Direction': 'center', 'Blinking': blink, 'Driving Direction': 'forward', 'Error': 0}, ignore_index=True)
            #print(avg_ratio_dir,"center ",time.strftime("%H:%M:%S", t))
        elif avg_ratio_dir>1.2:
            cv2.putText(frame, "left", (50, 100), font, 2, (0, 0, 225, 3))
            eye_direc ="left"

            #df.append({'Time': 'asd', 'Ratio': avg_ratio_dir, 'Looking Direction': 'left', 'Blinking': blink, 'Driving Direction': 'forward', 'Error': 0}, ignore_index=True)
            #print(avg_ratio_dir,"left ",time.strftime("%H:%M:%S", t))
        if key == 97:
            car_direc = "left"
            error=checkerror(df,"left")
        if key == 100:
            car_direc = "right"
            error = checkerror(df, "right")
        t = time.localtime()
        df = df.append(pd.Series([time.strftime("%H:%M:%S", t), av, eye_direc, blink, car_direc, error], index=df.columns),
                       ignore_index=True)
        df.to_excel(r'Data4_Morning.xlsx', index=False, header=True)




    cv2.imshow("Tracking",frame)



    if key==27:
        #print(df)
        break

cap.release()
cv2.destroyAllWindows()