
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import train
from scipy import signal
import svm




def Gaussianblur(save_road,time):
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
            
            if time ==0:
                cv2.imwrite("data/gasnoise/cat/"+f,img)
            else:
                n=0
                while  n <time:                   
                    img = cv2.GaussianBlur(img,ksize=(3,3),
                                        sigmaX=0,sigmaY=0)
                    n=n+1
                cv2.imwrite("data/gasnoise/cat/"+f,img)
    
    
    fs=os.listdir(save_road_2)
    for f in fs:
        if f.endswith('.png'):
            fullname = os.path.join(save_road_2, f)

            img = cv2.imread(fullname)
            img =cv2.imwrite(fullname,img)
            img = cv2.imread(fullname)
            img = np.array(img,dtype=float) 
            
            if time ==0:
                cv2.imwrite("data/gasnoise/dog/"+f,img)
            else:
                n=0
                while  n <time:                   
                    img = cv2.GaussianBlur(img,ksize=(3,3),
                                        sigmaX=0,sigmaY=0)
                    n=n+1
                    
                cv2.imwrite("data/gasnoise/dog/"+f,img)




times = [0,1,2,3,4,5,6,7,8,9]
acc_1=[]
acc_2=[]
save_road = "data/catdog"
for i in range(len(times)):
    Gaussianblur(save_road,times[i])   
    AC_1=train.main("data/gasnoise")
    acc_1.append(AC_1)
    AC_2=svm.SVMTEST("data/gasnoise")
    acc_2.append(AC_2)
   
y_1 = np.array(acc_1)
y_2 = np.array(acc_2)
x = np.array(times)

plt.ylabel('ACC')
plt.xlim(times[0],times[-1])
plt.ylim(0,1)
plt.plot(x,y_1,'r')
plt.plot(x,y_2,'b')
plt.show

    
    