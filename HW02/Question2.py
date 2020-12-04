#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math

pic = Image.open('Lenna.jpg')
pic_arr = np.asarray(pic)
Lennagray= np.zeros(shape=(256,256))
#NTSC method 0.21 R + 0.72 G + 0.07 B 
for i in pic_arr: 
    for j in pic_arr:
        Lennagray[i,j] = 0.21*pic_arr[i,j,0] + 0.72*pic_arr[i,j,1] + 0.07*pic_arr[i,j,2]
#plt.imshow(Lennagray,cmap='gray')

# Question 1

bins = np.zeros(256)
for i in range(256):
    for j in range(256):
        n = int(Lennagray[i,j])
        bins[n] = bins[n]+1
        
langs = np.arange(0,256,1)

plt.plot(langs,bins)
plt.xlim([0,256])
plt.ylim([0,650])
plt.xlabel('Intensity')
plt.ylabel('Frequency of occurance')

# Question 2
bins_acc = np.zeros(256)

bins_acc[0]=bins[0]

for i in range (0,256):
    bins_acc[i] = bins[i]+bins_acc[i-1]
        
langs = np.arange(0,256,1)

plt.plot(langs,bins_acc)
plt.xlim([0,256])
plt.ylim([0,70000])
plt.xlabel('Intensity')
plt.ylabel('Cummulative frequency of occurance')


# Question 3

bins_normalized = bins / (Lennagray.shape[0]*Lennagray.shape[1])           #pdf

#plt.subplot(121)
plt.figure('normalized histograph')
plt.plot(langs,bins_normalized)
plt.xlim([0,256])
plt.ylim([0,0.01])
plt.xlabel('Intensity')
plt.ylabel('Frequency of occurance')

bins_acc_normalized = np.zeros(256)
bins_acc_normalized[0]=bins[0]
for i in range (0,256):
    bins_acc_normalized[i] = bins_normalized[i]+bins_acc_normalized[i-1]   #cdf
        
langs = np.arange(0,256,1)

plt.figure('normalized cummulative histograph')
#plt.subplot(122)
plt.plot(langs,bins_acc_normalized)
plt.xlim([0,256])
plt.ylim([0,1.2])
plt.xlabel('Intensity')
plt.ylabel('Cummulative frequency of occurance')

Lennagray_equalized = np.zeros(shape=(256,256))

for i in range(256):
    for j in range(256):
        x = np.int(Lennagray[i,j])
        Lennagray_equalized[i,j] = bins_acc_normalized[x]*(256-1)
        
plt.imshow(Lennagray_equalized,cmap='gray') 

bins_new = np.zeros(256)
for i in range(256):
    for j in range(256):
        n = int(Lennagray_equalized[i,j])
        bins_new[n] = bins_new[n]+1
        
langs = np.arange(0,256,1)
plt.figure()
plt.plot(langs,bins_new)
plt.xlim([0,256])
plt.ylim([0,750])
plt.xlabel('Intensity')
plt.ylabel('Frequency of occurance')

bins_new_acc = np.zeros(256)
bins_new_acc[0]=bins[0]
for i in range (0,256):
    bins_new_acc[i] = bins_new[i]+bins_new_acc[i-1]
        
langs = np.arange(0,256,1)
plt.figure()
plt.plot(langs,bins_new_acc)
plt.xlim([0,256])
plt.ylim([0,70000])
plt.xlabel('Intensity')
plt.ylabel('Cummulative frequency of occurance')



