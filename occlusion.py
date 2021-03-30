import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import train
import svm
from PIL import Image
import random
import svm




def occlusion(save_road,sta):
    save_road_1 = str(save_road)+str("/CATS")
    save_road_2 = str(save_road)+str("/DOGS")
    fs=os.listdir(save_road_1)
    
    for f in fs:
        if f.endswith('.png'):
            fullname = os.path.join(save_road_1, f)
            image = Image.open(fullname).convert('RGB')
            if sta==0:
                image.save("data/gasnoise/cat/"+f)
                break
            img= np.zeros([sta, sta, 3], np.uint8)
            img = Image.fromarray(img)
            x=random.randint(0, 224-sta)
            y=random.randint(0,224-sta)
            image.paste(img,(x,y,x+sta,y+sta))
            image.save("data/gasnoise/cat/"+f)
    
    
    
    fs=os.listdir(save_road_2)
    for f in fs:
        if f.endswith('.png'):
            fullname = os.path.join(save_road_2, f)
            image = Image.open(fullname).convert('RGB')
            if sta==0:
                image.save("data/gasnoise/dog/"+f)
                break
            img= np.zeros([sta, sta, 3], np.uint8)
            img = Image.fromarray(img)
            x=random.randint(0, 224-sta)
            y=random.randint(0,224-sta)
            image.paste(img,(x,y,x+sta,y+sta))
            image.save("data/gasnoise/dog/"+f)



sta_noise = [0,5,10,15,20,25,30,35,40,45]
acc_1=[]
acc_2=[]
save_road = "data/catdog"
for i in range(len(sta_noise)):    
    occlusion(save_road,sta_noise[i])
    AC_1=train.main("data/gasnoise")
    acc_1.append(AC_1)
    AC_2=svm.SVMTEST("data/gasnoise")
    acc_2.append(AC_2)
   
y_1 = np.array(acc_1)
y_2 = np.array(acc_2)
x = np.array(sta_noise)

plt.ylabel('ACC')
plt.xlim(sta_noise[0],sta_noise[-1])
plt.ylim(0,1)
plt.plot(x,y_1,'r')
plt.plot(x,y_2,'b')
plt.show

    
    
