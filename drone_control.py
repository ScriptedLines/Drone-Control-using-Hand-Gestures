import cv2 as cv
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pandas as pd
import speech_recognition as sr
import requests
from io import BytesIO
from PIL import Image
from transformers import AutoProcessor, ViltForQuestionAnswering
from transformers import BlipProcessor, BlipForQuestionAnswering
import torch

processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-capfilt-large")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-capfilt-large")


recog=sr.Recognizer()



def audio_to_text():


    with sr.Microphone() as source:
        print("listening...")
        audio=recog.listen(source)
    try:
        text=recog.recognize_whisper(audio)
        print(text)
        return text
    except:
        return None

# API_TOKEN = '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'
# API_URL = "https://api-inference.huggingface.co/models/uclanlp/visualbert-vqa-coco-pre"

def image_qa(img):

    ques=audio_to_text()
    pil_image = Image.fromarray(img)

    inputs = processor(images=pil_image, text=ques, return_tensors="pt")


    with torch.no_grad():
        outputs = model.generate(**inputs)
        answer = processor.decode(outputs[0], skip_special_tokens=True)


    print(f"Predicted Answer: {answer}")




right_hand_model=tf.keras.models.load_model("gesture_recog_model_right_hand.h5")
left_hand_model=tf.keras.models.load_model("gesture_recog_model_left_hand.h5")


mphands=mp.solutions.hands
hands_right=mphands.Hands(min_detection_confidence=0.4,max_num_hands=1)
hands_left=mphands.Hands(min_detection_confidence=0.4,max_num_hands=1)


drone1=np.zeros((720,1280),dtype="uint8")
cv.rectangle(drone1,(871,605),(1215,190),(255,255,255),-1)

drone2=np.zeros((720,1280),dtype="uint8")
cv.rectangle(drone2,(53,705),(447,190),(255,255,255),-1)

gestures_list=["Pitch_Forward","Pitch_Backward","Roll_Left","Roll_Right","Yaw_Left","Yaw_Right","Stop","Thrust_Up","Down"]



vid=cv.VideoCapture(0)
vid.set(cv.CAP_PROP_FRAME_WIDTH,1920)
vid.set(cv.CAP_PROP_FRAME_HEIGHT,1080)

username = "Aaryaman"
password = "Qwerty@12345"
ip_address = "192.168.1.38"
port = "8080"
url = f"http://{username}:{password}@{ip_address}:{port}/video"

drone_vid=cv.VideoCapture(url)
drone_vid.set(cv.CAP_PROP_FRAME_WIDTH,1920)
drone_vid.set(cv.CAP_PROP_FRAME_HEIGHT,1080)

def get_coordinates(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:  
        print(f"Coordinates: x={x}, y={y}")
cv.namedWindow("webcam")
cv.setMouseCallback('webcam', get_coordinates)

check_hand_right=0
check_hand_left=0

while True:
    isTrue,frame=vid.read()
    istrue,drone_frame=drone_vid.read()
    drone_frame=cv.cvtColor(drone_frame,cv.COLOR_BGR2RGB)

    frame=cv.flip(frame,1)
    frame=cv.resize(frame,(1280,720))
    rgb_frame=cv.cvtColor(frame,cv.COLOR_BGR2RGB)

    drone1_detect=cv.bitwise_and(rgb_frame,rgb_frame,mask=drone1)
    drone2_detect=cv.bitwise_and(rgb_frame,rgb_frame,mask=drone2)

    result_drone1=hands_right.process(drone1_detect)
    result_drone2=hands_left.process(drone2_detect)

    if result_drone1.multi_hand_landmarks:
        l=[]
        for handLms in result_drone1.multi_hand_landmarks:
            for lm in handLms.landmark:
                l.append(lm.x)
                l.append(lm.y)
        if l[0]>l[8]:
            check_hand_right=1
        elif l[0]<l[8]:
            check_hand_right=-1
        l=np.array(l)
        for i in range(2,41,2):
            l[i]=l[i]-l[0]
        for i in range(3,42,2):
            l[i]=l[i]-l[1]
        l[0]=0
        l[1]=0
        l = l.reshape(1, -1)

        if check_hand_right==1:
            pred_y=right_hand_model.predict(l)[0]
        elif check_hand_right==-1:
            pred_y=left_hand_model.predict(l)[0]
            
        if(max(pred_y)>=0.99):
            pred_y=np.array(pred_y)

            pred_ind=np.argmax(pred_y)
            cv.putText(frame,gestures_list[pred_ind],(740,650),fontFace=cv.FONT_HERSHEY_SIMPLEX,fontScale=2,color=(255,0,0),thickness=5)
        else:
            cv.putText(frame,"Unknown",(740,650),fontFace=cv.FONT_HERSHEY_SIMPLEX,fontScale=2,color=(255,0,0),thickness=5)


        cv.rectangle(frame,(871,605),(1215,190),(0,255,0),5)
    
    if result_drone1.multi_hand_landmarks is None:
        cv.rectangle(frame,(871,605),(1215,190),(0,0,255),5)
    
    if result_drone2.multi_hand_landmarks:
        l=[]
        for handLms in result_drone2.multi_hand_landmarks:
            for lm in handLms.landmark:
                l.append(lm.x)
                l.append(lm.y)
        if l[0]>l[8]:
            check_hand_left=1
        elif l[0]<l[8]:
            check_hand_left=-1
        l=np.array(l)
        for i in range(2,41,2):
            l[i]=l[i]-l[0]
        for i in range(3,42,2):
            l[i]=l[i]-l[1]
        l[0]=0
        l[1]=0
        l = l.reshape(1, -1)

        if check_hand_left==1:
            pred_y=right_hand_model.predict(l)[0]
        elif check_hand_left==-1:
            pred_y=left_hand_model.predict(l)[0]
        
        if(max(pred_y)>=0.99):

            pred_y=np.array(pred_y)

            pred_ind=np.argmax(pred_y)
            cv.putText(frame,gestures_list[pred_ind],(100,650),fontFace=cv.FONT_HERSHEY_SIMPLEX,fontScale=2,color=(255,0,0),thickness=5)
        else:
            cv.putText(frame,"Unkown",(100,650),fontFace=cv.FONT_HERSHEY_SIMPLEX,fontScale=2,color=(255,0,0),thickness=5)

        cv.rectangle(frame,(103,605),(447,190),(0,255,0),5)
    if result_drone2.multi_hand_landmarks is None:
        cv.rectangle(frame,(103,605),(447,190),(0,0,255),5)
    




    cv.imshow("webcam",frame)
    if cv.waitKey(1) & 0xFF==ord("q"):

        image_qa(drone_frame)

    elif cv.waitKey(1) & 0xFF==ord("d"):
        break
vid.release()
cv.destroyAllWindows()