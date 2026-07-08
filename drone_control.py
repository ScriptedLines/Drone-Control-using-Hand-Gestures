import os
import time
import cv2 as cv
import numpy as np
import tensorflow as tf
import pandas as pd
import speech_recognition as sr
from io import BytesIO
from PIL import Image
import torch

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image as ROSImage
try:
    from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand
except ImportError:
    print("px4_msgs not found. Make sure you have sourced your ROS 2 workspace.")
    OffboardControlMode = None
    TrajectorySetpoint = None
    VehicleCommand = None

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

username = "Aaryaman"
password = "Qwerty@12345"
ip_address = "192.168.1.38"
port = "8080"
url = f"http://{username}:{password}@{ip_address}:{port}/video"

processor = None
model = None
vqa_ready = False

recog=sr.Recognizer()


def load_vqa_model():
    global processor, model, vqa_ready
    if vqa_ready:
        return True
    try:
        from transformers import BlipProcessor, BlipForQuestionAnswering
        print("Loading VQA model...")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-capfilt-large")
        model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-capfilt-large")
        vqa_ready = True
        print("VQA model loaded.")
        return True
    except Exception as e:
        print(f"Unable to load VQA model: {e}")
        return False



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


class PX4Controller(Node):
    def __init__(self):
        super().__init__('drone_gesture_controller')
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        if OffboardControlMode is not None:
            self.offboard_control_mode_publisher = self.create_publisher(
                OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
            self.trajectory_setpoint_publisher = self.create_publisher(
                TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
            self.vehicle_command_publisher = self.create_publisher(
                VehicleCommand, '/fmu/in/vehicle_command', qos_profile)
            
            self.timer = self.create_timer(0.1, self.timer_callback)
            
        self.setpoint = [0.0, 0.0, 0.0] # vx, vy, vz
        self.yaw_speed = 0.0
        self.setup_counter = 0
        self.is_taking_off = False

    def publish_vehicle_command(self, command, **params):
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = float(params.get("param1", 0.0))
        msg.param2 = float(params.get("param2", 0.0))
        msg.param3 = float(params.get("param3", 0.0))
        msg.param4 = float(params.get("param4", 0.0))
        msg.param5 = float(params.get("param5", 0.0))
        msg.param6 = float(params.get("param6", 0.0))
        msg.param7 = float(params.get("param7", 0.0))
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_publisher.publish(msg)

    def arm_and_offboard(self):
        print("Automatically Arming and switching to Offboard Mode...")
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)

    def set_velocity(self, vx, vy, vz, yaw_speed):
        self.setpoint = [float(vx), float(vy), float(vz)]
        self.yaw_speed = float(yaw_speed)

    def timer_callback(self):
        # Publish offboard control mode
        offboard_msg = OffboardControlMode()
        offboard_msg.position = False
        offboard_msg.velocity = True
        offboard_msg.acceleration = False
        offboard_msg.attitude = False
        offboard_msg.body_rate = False
        offboard_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(offboard_msg)

        current_velocity = self.setpoint
        current_yaw = self.yaw_speed

        self.setup_counter += 1
        if self.setup_counter == 20: # After 2 seconds of publishing setpoints
            self.arm_and_offboard()
            self.is_taking_off = True
            print("Auto-takeoff initiated...")

        if self.is_taking_off:
            if self.setup_counter < 50: # Fly up for 3 seconds (2.0s to 5.0s)
                current_velocity = [0.0, 0.0, -1.5]
                current_yaw = 0.0
            else:
                self.is_taking_off = False
                print("Auto-takeoff complete. Hand gesture control is now active!")

        # Publish trajectory setpoint
        setpoint_msg = TrajectorySetpoint()
        setpoint_msg.position = [float('nan'), float('nan'), float('nan')]
        setpoint_msg.acceleration = [float('nan'), float('nan'), float('nan')]
        setpoint_msg.jerk = [float('nan'), float('nan'), float('nan')]
        setpoint_msg.yaw = float('nan')
        setpoint_msg.velocity = current_velocity
        setpoint_msg.yawspeed = current_yaw
        setpoint_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(setpoint_msg)

# API_TOKEN = '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'
# API_URL = "https://api-inference.huggingface.co/models/uclanlp/visualbert-vqa-coco-pre"

def image_qa(img):
    if not load_vqa_model():
        print("Skipping VQA because the model could not be loaded.")
        return

    ques=audio_to_text()
    if not ques:
        print("No question received.")
        return

    pil_image = Image.fromarray(img)
    inputs = processor(images=pil_image, text=ques, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(**inputs)
        answer = processor.decode(outputs[0], skip_special_tokens=True)

    print(f"Predicted Answer: {answer}")


try:
    right_hand_model=tf.keras.models.load_model("gesture_recog_model_right_hand.h5")
    left_hand_model=tf.keras.models.load_model("gesture_recog_model_left_hand.h5")
    print("Gesture models loaded successfully.")
except Exception as e:
    print(f"Unable to load gesture models: {e}")
    right_hand_model = None
    left_hand_model = None


model_path = os.path.join(os.path.dirname(__file__), 'hand_landmarker.task')
base_options = mp_python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.4,
    min_hand_presence_confidence=0.4,
    min_tracking_confidence=0.4)
hands_right = vision.HandLandmarker.create_from_options(options)
hands_left = vision.HandLandmarker.create_from_options(options)


drone1=np.zeros((720,1280),dtype="uint8")
cv.rectangle(drone1,(871,605),(1215,190),(255,255,255),-1)

drone2=np.zeros((720,1280),dtype="uint8")
cv.rectangle(drone2,(53,705),(447,190),(255,255,255),-1)

gestures_list=["Pitch_Forward","Pitch_Backward","Roll_Left","Roll_Right","Yaw_Left","Yaw_Right","Stop","Thrust_Up","Down"]



vid=cv.VideoCapture(0)
if not vid.isOpened():
    print("Webcam not available. Exiting.")
    raise SystemExit(0)
vid.set(cv.CAP_PROP_FRAME_WIDTH,1920)
vid.set(cv.CAP_PROP_FRAME_HEIGHT,1080)

username = "Aaryaman"
password = "Qwerty@12345"
ip_address = "192.168.1.38"
port = "8080"
url = f"http://{username}:{password}@{ip_address}:{port}/video"

drone_vid = cv.VideoCapture(url)
drone_available = drone_vid.isOpened()
if not drone_available:
    print("Drone video stream is not available. Continuing without it.")
drone_vid.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
drone_vid.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)

def get_coordinates(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:  
        print(f"Coordinates: x={x}, y={y}")
cv.namedWindow("webcam")
cv.setMouseCallback('webcam', get_coordinates)

check_hand_right=0
check_hand_left=0

# Initialize ROS 2 Node
try:
    rclpy.init()
    px4_node = PX4Controller()
except Exception as e:
    print(f"Failed to initialize ROS 2: {e}")
    px4_node = None


last_gesture_right = "Stop"
last_gesture_left = "Stop"

while True:
    isTrue,frame=vid.read()
    if not isTrue or frame is None:
        print("Unable to read webcam frame. Retrying...")
        continue

    if drone_available:
        istrue, drone_frame = drone_vid.read()
        if istrue and drone_frame is not None:
            drone_frame = cv.cvtColor(drone_frame, cv.COLOR_BGR2RGB)
        else:
            drone_frame = None
    else:
        drone_frame = None

    frame=cv.flip(frame,1)
    frame=cv.resize(frame,(1280,720))
    rgb_frame=cv.cvtColor(frame,cv.COLOR_BGR2RGB)

    drone1_detect=cv.bitwise_and(rgb_frame,rgb_frame,mask=drone1)
    drone2_detect=cv.bitwise_and(rgb_frame,rgb_frame,mask=drone2)

    mp_image1 = mp.Image(image_format=mp.ImageFormat.SRGB, data=drone1_detect)
    mp_image2 = mp.Image(image_format=mp.ImageFormat.SRGB, data=drone2_detect)
    timestamp_ms = int(time.time() * 1000)
    result_drone1 = hands_right.detect_for_video(mp_image1, timestamp_ms)
    result_drone2 = hands_left.detect_for_video(mp_image2, timestamp_ms)

    if right_hand_model is not None and left_hand_model is not None and result_drone1.hand_landmarks:
        l=[]
        for handLms in result_drone1.hand_landmarks:
            for lm in handLms:
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
            gesture = gestures_list[pred_ind]
            cv.putText(frame,gesture,(740,650),fontFace=cv.FONT_HERSHEY_SIMPLEX,fontScale=2,color=(255,0,0),thickness=5)
            
            vx, vy, vz, yaw = 0.0, 0.0, 0.0, 0.0
            if gesture == "Pitch_Forward": vx = 1.0
            elif gesture == "Pitch_Backward": vx = -1.0
            elif gesture == "Roll_Left": vy = -1.0
            elif gesture == "Roll_Right": vy = 1.0
            elif gesture == "Thrust_Up": vz = -1.0
            elif gesture == "Down": vz = 1.0
            elif gesture == "Yaw_Left": yaw = -1.0
            elif gesture == "Yaw_Right": yaw = 1.0
            if px4_node: px4_node.set_velocity(vx, vy, vz, yaw)
            last_gesture_right = gesture
        else:
            cv.putText(frame,last_gesture_right,(740,650),fontFace=cv.FONT_HERSHEY_SIMPLEX,fontScale=2,color=(255,0,0),thickness=5)


        cv.rectangle(frame,(871,605),(1215,190),(0,255,0),5)
    
    if not result_drone1.hand_landmarks:
        cv.rectangle(frame,(871,605),(1215,190),(0,0,255),5)
        cv.putText(frame,last_gesture_right,(740,650),fontFace=cv.FONT_HERSHEY_SIMPLEX,fontScale=2,color=(255,0,0),thickness=5)
    
    if right_hand_model is not None and left_hand_model is not None and result_drone2.hand_landmarks:
        l=[]
        for handLms in result_drone2.hand_landmarks:
            for lm in handLms:
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
            gesture = gestures_list[pred_ind]
            cv.putText(frame,gesture,(100,650),fontFace=cv.FONT_HERSHEY_SIMPLEX,fontScale=2,color=(255,0,0),thickness=5)
            
            vx, vy, vz, yaw = 0.0, 0.0, 0.0, 0.0
            if gesture == "Pitch_Forward": vx = 1.0
            elif gesture == "Pitch_Backward": vx = -1.0
            elif gesture == "Roll_Left": vy = -1.0
            elif gesture == "Roll_Right": vy = 1.0
            elif gesture == "Thrust_Up": vz = -1.0
            elif gesture == "Down": vz = 1.0
            elif gesture == "Yaw_Left": yaw = -1.0
            elif gesture == "Yaw_Right": yaw = 1.0
            if px4_node: px4_node.set_velocity(vx, vy, vz, yaw)
            last_gesture_left = gesture
        else:
            cv.putText(frame,last_gesture_left,(100,650),fontFace=cv.FONT_HERSHEY_SIMPLEX,fontScale=2,color=(255,0,0),thickness=5)

        cv.rectangle(frame,(103,605),(447,190),(0,255,0),5)
    if not result_drone2.hand_landmarks:
        cv.rectangle(frame,(103,605),(447,190),(0,0,255),5)
        cv.putText(frame,last_gesture_left,(100,650),fontFace=cv.FONT_HERSHEY_SIMPLEX,fontScale=2,color=(255,0,0),thickness=5)
    




    if px4_node:
        rclpy.spin_once(px4_node, timeout_sec=0.0)

    cv.imshow("webcam",frame)
    if drone_frame is not None:
        bgr_drone_frame = cv.cvtColor(drone_frame, cv.COLOR_RGB2BGR)
        cv.imshow("Drone Camera", bgr_drone_frame)
        
    key = cv.waitKey(1) & 0xFF
    if key == ord("q"):
        if drone_frame is not None:
            image_qa(drone_frame)
        else:
            print("Drone frame unavailable; skipping VQA.")
    elif key == ord("d"):
        break
vid.release()
if drone_available:
    drone_vid.release()
cv.destroyAllWindows()

if px4_node:
    px4_node.destroy_node()
    rclpy.shutdown()