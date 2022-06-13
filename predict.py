#!/usr/bin/env python
# coding: utf-8

# In[1]:



import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd
from model import swin_large_patch4_window12_384_in22k as create_model
import numpy as np
import csv
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x
def test():
    test_path = "./test"
    json_path = './class_indices.json'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_classes = 219
    img_size = 384
    data_transform = transforms.Compose([transforms.Resize([img_size,img_size]),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)
    class_list=list(class_indict.keys())
    class_list.append("img")
    model = create_model(num_classes=num_classes).to(device)
    list_= ["55","57","62"]
    for item in list_:
        print("model: %s\n"%item)
        model_name = 'model-%s'%item
        model_weight_path = "./weights/%s.pth"%model_name
        model.load_state_dict(torch.load(model_weight_path, map_location=device))
        model.eval()
        model.cuda()
        file = os.listdir(test_path)
        number=0
        with open("result//"+item+'.csv', 'a+', newline='') as student_file:
            writer = csv.writer(student_file)
            writer.writerow(class_list)
            for img_name in file:
                if number%1000==0:
                    print(number)
                img_path=os.path.join(test_path,img_name)
                img = Image.open(img_path)
                img = data_transform(img)
                img = torch.unsqueeze(img, dim=0)
                with torch.no_grad():
                    output = torch.squeeze(model(img.to(device))).cpu()
                ensemble_output = output.numpy()
                ensemble_output=list(ensemble_output)
                ensemble_output.append(img_name)
                writer.writerow(ensemble_output)
                number+=1
if __name__ == '__main__':
    if os.path.exists("./result") is False:
        os.makedirs("./result")
    test()
    
    json_path = './class_indices.json'
    json_file = open(json_path, "r")
    class_indict = json.load(json_file)
    class_list=list(class_indict.keys())
    df_55=pd.read_csv("result//55.csv")
    df_57=pd.read_csv("result//57.csv")
    df_62=pd.read_csv("result//62.csv")
    
    with open("result//55_57_62_Sub.csv", 'a+', newline='') as student_file:
        writer = csv.writer(student_file)
        writer.writerow(["filename","category"])
        for i in range(0,len(df_57)):
            sub_55=df_55.loc[i][class_list]
            sub_55=np.array(sub_55).astype("float32")
            sub_57=df_57.loc[i][class_list]
            sub_57=np.array(sub_57).astype("float32")
            sub_62=df_62.loc[i][class_list]
            sub_62=np.array(sub_62).astype("float32")

            
            sub=(sub_62+sub_57+sub_55)/3
            sub=torch.from_numpy(sub).cpu()
            predict = torch.softmax(sub, dim=0)
            predict_cla = torch.argmax(predict).numpy()
            predict=class_indict[str(predict_cla)]
            writer.writerow([df_57.loc[i]["img"],predict])


# In[5]:





# In[ ]:





# In[ ]:





# In[ ]:




