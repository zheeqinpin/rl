#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 09:05:21 2018

@author: qingping
"""

import fnmatch
import os
import pandas as pd
import numpy as np  
import sys
import cv2

def ReadSaveAddr(Stra,Strb):
    #print(Stra)
    #print(Strb)
    print("Read :",Stra,Strb)
    a_list = fnmatch.filter(os.listdir(Stra),Strb)
    print("Find = ",len(a_list))
    df = pd.DataFrame(np.arange(len(a_list)).reshape((len(a_list),1)),columns=['Addr'])  
    df.Addr = a_list

    for i in range(len(a_list)):
        path = Stra+'/'+a_list[i]
        #print(path)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        t = a_list[i]
        t = t[:-4]
        t = '/home/qingping/rl/reinforcement_learning_basic_book-master/6-value_function_approximate/deep_learning_flappy_bird/assets/sprites/'+t+'.bmp'
        cv2.imwrite(t,img)

    df.to_csv('Get.lst',columns=['Addr'],index=False,header=False)
    print("Write To Get.lst !")

ReadSaveAddr("/home/qingping/rl/reinforcement_learning_basic_book-master/6-value_function_approximate/deep_learning_flappy_bird/assets/sprites/","*.png")
