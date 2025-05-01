import pandas as pd

data=pd.read_csv("Training_Data_Left_Hand.csv")

for i in range(1,21):
    data[f"x{i}"]=data[f"x{i}"]-data["x0"]
    data[f"y{i}"]=data[f"y{i}"]-data["y0"]

data["x0"]=data["x0"]-data["x0"]
data["y0"]=data["y0"]-data["y0"]
data.to_csv("normalized_training_set_left_hand.csv")