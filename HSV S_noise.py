import numpy as np
import matplotlib.pyplot as plt
import os
import train
import svm
import colorsys
from PIL import Image
import random




def S_noise(save_road,sta):
    save_road_1 = str(save_road)+str("/CATS")
    save_road_2 = str(save_road)+str("/DOGS")
    fs=os.listdir(save_road_1)
    
    for f in fs:
        if f.endswith('.png'):
            fullname = os.path.join(save_road_1, f)
            image = Image.open(fullname).convert('RGB')
            image.load()
            r, g, b = image.split()
            result_r, result_g, result_b = [], [], []
            for pixel_r, pixel_g, pixel_b in zip(r.getdata(), g.getdata(), b.getdata()):
                
                h, s, v = colorsys.rgb_to_hsv(pixel_r / 255., pixel_b / 255., pixel_g / 255.)
                noise = np.random.normal(0,sta)
                s=h+noise
                if s>1:
                    s=1
                if s<0:
                    s=0
                rgb = colorsys.hsv_to_rgb(h, s, v)
                pixel_r, pixel_g, pixel_b = [int(x * 255.) for x in rgb]
                result_r.append(pixel_r)
                result_g.append(pixel_g)
                result_b.append(pixel_b)

            r.putdata(result_r)
            g.putdata(result_g)
            b.putdata(result_b)
            image = Image.merge("RGB", (r, g, b))
            image.save("data/gasnoise/cat/"+f)
    
    
    
    fs=os.listdir(save_road_2)
    for f in fs:
        if f.endswith('.png'):
            fullname = os.path.join(save_road_2, f)
            image = Image.open(fullname).convert('RGB')
            image.load()
            r, g, b = image.split()
            result_r, result_g, result_b = [], [], []
            for pixel_r, pixel_g, pixel_b in zip(r.getdata(), g.getdata(), b.getdata()):
                
                h, s, v = colorsys.rgb_to_hsv(pixel_r / 255., pixel_b / 255., pixel_g / 255.)
                noise = np.random.normal(0,sta)
                s=h+noise
                if s>1:
                    s=1
                if s<0:
                    s=0
                rgb = colorsys.hsv_to_rgb(h, s, v)
                pixel_r, pixel_g, pixel_b = [int(x * 255.) for x in rgb]
                result_r.append(pixel_r)
                result_g.append(pixel_g)
                result_b.append(pixel_b)

            r.putdata(result_r)
            g.putdata(result_g)
            b.putdata(result_b)
            image = Image.merge("RGB", (r, g, b))
            image.save("data/gasnoise/dog/"+f)



sta_noise =[0,0.02,0.04,0.06,0.08,0.10,0.12,0.14,0.16,0.18]
acc_1=[]
acc_2=[]
save_road = "data/catdog"
for i in range(len(sta_noise)):    
    S_noise(save_road,sta_noise[i])
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

    
    
