#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
import cv2 as cv
from scipy import misc
import scipy.ndimage.filters as filters
import scipy.ndimage as ndimage
parkinglot = cv.imread('ParkingLot.jpg')
parkinglot.shape
plt.imshow(parkinglot)


#convert to gray scale image
parkinglotgray= np.zeros(shape=(parkinglot.shape[0],parkinglot.shape[1]))

for i in range (parkinglot.shape[0]): 
    for j in range (parkinglot.shape[1]):
        parkinglotgray[i,j] = 0.21*parkinglot[i,j,0] + 0.72*parkinglot[i,j,1] + 0.07*parkinglot[i,j,2]
plt.imshow(parkinglotgray,cmap='gray')

# Question 1 histogram
plt.hist(parkinglot.ravel(),256,[0,256])
plt.show()

# threshold
threshold = 215

parkinglotbinary = np.zeros(shape=(parkinglot.shape[0],parkinglot.shape[1]))

for i in range (parkinglot.shape[0]): 
    for j in range (parkinglot.shape[1]):
        if parkinglotgray[i,j] >threshold:
            parkinglotbinary[i,j]=1
            
plt.imshow(parkinglotbinary,cmap='gray')


# dimension of given image
picshape = parkinglotbinary.shape
picshape_xmax = picshape[0]
picshape_ymax = picshape[1]


# define max and min r and theta ranges
theta_max = math.pi
theta_min = 0

r_min = 0
r_max = math.hypot(picshape_xmax,picshape_ymax)

# hough space dimension
hough_x = 300   # theta dim
hough_y = 300   # range dim

houghspace = np.zeros((hough_y,hough_x))          

for x in range(picshape_xmax):
    for y in range(picshape_ymax):
        if parkinglotbinary[x,y] == 1:
            for t in range(hough_x):
                theta = t * theta_max / hough_x 
                r = x*math.cos(theta)+y*math.sin(theta)
                r_plot = int(r * (hough_y/r_max))
                houghspace[r_plot,t] = houghspace[r_plot,t] + 1     # vote
                
plt.imshow(houghspace,origin='lower')


neighborhood_size = 20
threshold = 115

data_max = filters.maximum_filter(houghspace, neighborhood_size)
maxima = (houghspace == data_max)

data_min = filters.minimum_filter(houghspace, neighborhood_size)
diff = ((data_max - data_min) > threshold)
maxima[diff == 0] = 0

labeled, num_objects = ndimage.label(maxima)
slices = ndimage.find_objects(labeled)

x, y = [], []
for dy,dx in slices:
    x_center = (dx.start + dx.stop - 1)/2
    x.append(x_center)
    y_center = (dy.start + dy.stop - 1)/2    
    y.append(y_center)

print x
print y

plt.imshow(houghspace, origin='lower')
plt.savefig('hough_space_i_j.png', bbox_inches = 'tight')

plt.autoscale(False)
plt.plot(x,y, 'ro')
plt.savefig('hough_space_maximas.png', bbox_inches = 'tight')

plt.close()


# manual method to extract lines and index
line_index = 1

for i,j in zip(y, x):

    r = round( (1.0 * i * r_max ) / hough_y,1)
    theta = round( (1.0 * j * theta_max) / hough_x,1)

    fig, ax = plt.subplots()

    ax.imshow(parkinglotgray,cmap='gray')

    ax.autoscale(False)

    px = []
    py = []
    for i in range(-picshape_ymax-40,picshape_ymax+40,1):
        px.append( math.cos(-theta) * i - math.sin(-theta) * r ) 
        py.append( math.sin(-theta) * i + math.cos(-theta) * r )

    ax.plot(px,py, linewidth=10)

    plt.savefig("image_line_"+ "%02d" % line_index +".png",bbox_inches='tight')

    #plt.show()

    plt.close()

    line_index = line_index + 1

