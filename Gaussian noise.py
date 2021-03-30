
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import train
import svm
import colorsys
from PIL import Image
import random




def HSV_noise(save_road,sta):
    save_road_1 = str(save_road)+str("/CATS")
    save_road_2 = str(save_road)+str("/DOGS")
    fs=os.listdir(save_road_1)
    
    for f in fs:
        if f.endswith('.png'):
            fullname = os.path.join(save_road_1, f)
            img = cv2.imread(fullname)
            img =cv2.imwrite(fullname,img)
            img = cv2.imread(fullname)
            img = np.array(img,dtype=float)
            
            noise = np.random.normal(1,sta,img.shape)
            out = img + noise
            out = cv2.normalize(out,0,1,norm_type=cv2.NORM_MINMAX)
            out = np.uint8(out*255)
            cv2.imwrite("data/gasnoise/cat/"+f,out)
    
    
    
    fs=os.listdir(save_road_2)
    for f in fs:
        if f.endswith('.png'):
            fullname = os.path.join(save_road_2, f)
            img = cv2.imread(fullname)
            img =cv2.imwrite(fullname,img)
            img = cv2.imread(fullname)
            img = np.array(img,dtype=float)
            
            noise = np.random.normal(1,sta,img.shape)
            out = img + noise
            out = cv2.normalize(out,0,1,norm_type=cv2.NORM_MINMAX)
            out = np.uint8(out*255)
            cv2.imwrite("data/gasnoise/dog/"+f,out)



var_noise =[0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
acc_1=[]
acc_2=[]
save_road = "data/catdog"
for i in range(len(var_noise)):    
    HSV_noise(save_road,var_noise[i])
    AC_1=train.main("data/gasnoise")
    acc_1.append(AC_1)
    AC_2=svm.SVMTEST("data/gasnoise")
    acc_2.append(AC_2)
   
y_1 = np.array(acc_1)
y_2 = np.array(acc_2)
x = np.array(var_noise)

plt.ylabel('ACC')
plt.xlim(var_noise[0],var_noise[-1])
plt.ylim(0,1)
plt.plot(x,y_1,'r')
plt.plot(x,y_2,'b')
plt.show

    
    