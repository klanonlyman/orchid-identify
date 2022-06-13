#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import pandas as pd
import cv2
import os
import data_augment_tool
import csv
df=pd.read_csv("data//label.csv")
filename=df["filename"]
category=df["category"]
if os.path.exists("./data_augmentation") is False:
      os.makedirs("./data_augmentation")
with open("data_augmentation//label.csv","a+",newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename","category"])
    for i in range(0,len(filename)):
        label=category[i]
        name=filename[i]
        img_path=os.path.join("data//",name)
        img=cv2.imread(img_path)
        
 
        cv2.imwrite(os.path.join("data_augmentation//",name),img)
        writer.writerow([name,label])
        
        after=data_augment_tool.colorjitter(img,"b")
        after_name="b_"+name
        cv2.imwrite(os.path.join("data_augmentation//",after_name),after)
        writer.writerow([after_name,label])

        after=data_augment_tool.colorjitter(img,"s")
        after_name="s_"+name
        cv2.imwrite(os.path.join("data_augmentation//",after_name),after)
        writer.writerow([after_name,label])

        after=data_augment_tool.colorjitter(img,"c")
        after_name="c_"+name
        cv2.imwrite(os.path.join("data_augmentation//",after_name),after)
        writer.writerow([after_name,label])

        after=data_augment_tool.noisy(img,"gauss")
        after_name="gauss_"+name
        cv2.imwrite(os.path.join("data_augmentation//",after_name),after)
        writer.writerow([after_name,label])

        after=data_augment_tool.noisy(img,"sp")
        after_name="sp_"+name
        cv2.imwrite(os.path.join("data_augmentation//",after_name),after)
        writer.writerow([after_name,label])

        after=data_augment_tool.filters(img,"blur")
        after_name="blur_"+name
        cv2.imwrite(os.path.join("data_augmentation//",after_name),after)
        writer.writerow([after_name,label])

        after=data_augment_tool.filters(img,"gaussian")
        after_name="gaussian_"+name
        cv2.imwrite(os.path.join("data_augmentation//",after_name),after)
        writer.writerow([after_name,label])

        after=data_augment_tool.filters(img,"median")
        after_name="median_"+name
        cv2.imwrite(os.path.join("data_augmentation//",after_name),after)
        writer.writerow([after_name,label])
        
        after=data_augment_tool.h_mirror(img)
        after_name="mirrorh_"+name
        cv2.imwrite(os.path.join("data_augmentation//",after_name),after)
        writer.writerow([after_name,label])
        
        
        after=data_augment_tool.v_mirror(img)
        after_name="mirrorv_"+name
        cv2.imwrite(os.path.join("data_augmentation//",after_name),after)
        writer.writerow([after_name,label])
        
        


# In[2]:





# In[17]:





# In[ ]:





# In[ ]:





# In[ ]:




