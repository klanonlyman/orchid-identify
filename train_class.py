import pandas as pd
import cv2
import os
import data_augment_tool
import csv
df=pd.read_csv("data_augmentation//label.csv")
if os.path.exists("./train") is False:
      os.makedirs("./train")
for i in range(0,len(df)):
    sub=df.loc[i]
    filename=sub["filename"]
    category=str(sub["category"])
    floder=os.path.join("train",category)
    if not(os.path.exists(floder)):
        os.mkdir(floder)
    img=cv2.imread(os.path.join("data_augmentation",filename))
    cv2.imwrite(os.path.join(floder,filename),img)
