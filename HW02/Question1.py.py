#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
pic = Image.open('Lenna.jpg')

# print image
#Question 1
pic_arr = np.asarray(pic)

Lennagray= np.zeros(shape=(256,256))

#NTSC method 0.21 R + 0.72 G + 0.07 B 
for i in pic_arr: 
    for j in pic_arr:
        Lennagray[i,j] = 0.21*pic_arr[i,j,0] + 0.72*pic_arr[i,j,1] + 0.07*pic_arr[i,j,2]

#show gray scale image        
plt.imshow(Lennagray,cmap='gray')

# Question 2

Lennagrayds = np.zeros(shape=(64,64))

for i in range(64):
    for j in range(64):
        Lennagrayds[i,j] = Lennagray[i*4,j*4]
        
plt.imshow(Lennagrayds,cmap ='gray')

Lennagrayds.shape

# Question 3

sobel_vertical =np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
sobel_horizontal = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])

Lenna_edge = np.zeros(shape=(256,256))
v_edge = np.zeros(shape=(256,256))
h_edge = np.zeros(shape=(256,256))

pic_array = Lennagray
kernal    = sobel_vertical
kernal2   = sobel_horizontal
for i in range(1,255):
    for j in range(1,255):
        v_edge[i,j]=(pic_array[i-1,j-1]*kernal[0,0]+pic_array[i-1,j]*kernal[0,1]+pic_array[i-1,j+1]*kernal[0,2]
                     + pic_array[i,j-1]*kernal[1,0]+pic_array[i,j]*kernal[1,1]+pic_array[i,j+1]*kernal[1,2]
                     +pic_array[i+1,j-1]*kernal[2,0]+pic_array[i+1,j]*kernal[2,1]+pic_array[i+1,j+1]*kernal[2,2]
                     )/4
        
        h_edge[i,j]=(pic_array[i-1,j-1]*kernal2[0,0]+pic_array[i-1,j]*kernal2[0,1]+pic_array[i-1,j+1]*kernal2[0,2]
                    +pic_array[i,j-1]*kernal2[1,0]+pic_array[i,j]*kernal2[1,1]+pic_array[i,j+1]*kernal2[1,2]
                    +pic_array[i+1,j-1]*kernal2[2,0]+pic_array[i+1,j]*kernal2[2,1]+pic_array[i+1,j+1]*kernal2[2,2]
                    )/4
        Lenna_edge[i,j] = math.sqrt(v_edge[i,j]**2 + h_edge[i,j]**2)

#Lenna_edge = Lenna_edge*256/(np.amax(Lenna_edge))
        
plt.imshow(Lenna_edge,cmap='gray')
#Lenna_edge.shape

