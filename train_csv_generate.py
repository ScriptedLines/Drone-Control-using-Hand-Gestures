import pandas as pd
import numpy as np
import cv2 as cv
import mediapipe as mp

mphand=mp.solutions.hands
hands=mphand.Hands(max_num_hands=1)
mpDraw=mp.solutions.drawing_utils

np_arr=np.zeros((225,43),dtype=object)


n=0
entry_number=0
gesture_name=""


vid=cv.VideoCapture(0)
vid.set(cv.CAP_PROP_FRAME_WIDTH,1920)
vid.set(cv.CAP_PROP_FRAME_HEIGHT,1080)


while True:
    isTrue,frame=vid.read()
    frame=cv.resize(frame,(1280,720))
    frame=cv.flip(frame,1)
    rgb_frame=cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    results=hands.process(rgb_frame)

    if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                 mpDraw.draw_landmarks(frame,handLms,mphand.HAND_CONNECTIONS)
                 

    if cv.waitKey(1) & 0xFF==ord("p"):
        if results.multi_hand_landmarks:
            l=[]
            if n==0:
                gesture_name=input("Enter gesture_name:")

            for handLms in results.multi_hand_landmarks:
                for id,lm in enumerate(handLms.landmark):
                    l.append(lm.x)
                    l.append(lm.y)
            l.append(gesture_name)
            np_arr[entry_number]=l
            entry_number+=1

            n+=1
            if n==25:
                n=0
            print(f"Success!{n}")
        
    if entry_number==225:
         break



    cv.imshow("webcam",frame)




vid.release()
cv.destroyAllWindows()

col_names=[]
for i in range(21):
     col_names.append(f"x{i}")
     col_names.append(f"y{i}")
col_names.append("Gesture")

my_data=pd.DataFrame(np_arr,columns=col_names)
my_data.to_csv("Training_Data_Left_Hand.csv")