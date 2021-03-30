import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import train
import svm




def contrastincrease(save_road,time):
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
            img = np.uint8(np.clip((time * img), 0, 255))
            cv2.imwrite("data/gasnoise/cat/"+f,img)
    
    
    fs=os.listdir(save_road_2)
    for f in fs:
        if f.endswith('.png'):
            fullname = os.path.join(save_road_2, f)

            img = cv2.imread(fullname)
            img =cv2.imwrite(fullname,img)
            img = cv2.imread(fullname)
            img = np.array(img,dtype=float) 
            img = np.uint8(np.clip((time * img), 0, 255))        
            cv2.imwrite("data/gasnoise/dog/"+f,img)




times_1 = [1.0, 1.03, 1.06, 1.09, 1.12, 1.15, 1.18, 1.21,1.24, 1.27]
times_2 = [1.0, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30,0.20, 0.10]
acc_1=[]
acc_2=[]
save_road = "data/catdog"
a=1



if a==1:
    for i in range(len(times_2)):
        contrastincrease(save_road,times_2[i])   
        AC_1=train.main("data/gasnoise")
        acc_1.append(AC_1)
        AC_2=svm.SVMTEST("data/gasnoise")
        acc_2.append(AC_2)
   
    y_1 = np.array(acc_1)
    y_2 = np.array(acc_2)
    x = np.array(times_1)

    plt.ylabel('ACC')
    plt.xlim(times_1[0],times_1[-1])
    plt.ylim(0,1)
    plt.plot(x,y_1,'r')
    plt.plot(x,y_2,'b')
    plt.show
    
    