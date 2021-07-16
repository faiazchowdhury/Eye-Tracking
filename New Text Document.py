import cv2
import numpy as np
import dlib
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


while True:
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

        #Gaze Detection
        threshold_right_eye=detect_eye_gaze(36,37,38,39,40,41,"Right Eye")
        threshold_left_eye=detect_eye_gaze(42, 43, 44, 45, 46, 47, "Left Eye")

        ratio_left_eye=get_ratio_threshold(threshold_left_eye)
        ratio_right_eye = get_ratio_threshold(threshold_right_eye)
        avg_ratio_dir=(ratio_left_eye+ratio_right_eye)/2
        if avg_ratio_dir<=0.6:
            cv2.putText(frame, "right", (50, 100), font, 2, (0, 0, 225, 3))
            print(avg_ratio_dir,"right")
        elif 0.6<avg_ratio_dir<=1.2:
            cv2.putText(frame, "center", (50, 100), font, 2, (0, 0, 225, 3))
            print(avg_ratio_dir,"center")
        elif avg_ratio_dir>1.2:
            cv2.putText(frame, "left", (50, 100), font, 2, (0, 0, 225, 3))
            print(avg_ratio_dir,"left")





    cv2.imshow("Tracking",frame)

    key = cv2.waitKey(1)
    if key==27:
        break

cap.release()
cv2.destroyAllWindows()