#!/usr/bin/env python
# coding: utf-8

# In[107]:


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
import cv2 as cv2
image = cv2.imread('ParkingLot.jpg')
image.shape


#grayscale
grayimage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

#blur
blurImg = cv2.GaussianBlur(grayimage, (3,3),0)

#threshold
for i in range (blurImg.shape[0]):
    for j in range (blurImg.shape[1]):
        if blurImg[i,j]<205:
            blurImg[i,j]=0
            
plt.imshow(blurImg,cmap='gray')

#Canny to find edges
edges = cv2.Canny(blurImg,75,150)
plt.imshow(edges,cmap='gray')

#Hough to find lines
lines = cv2.HoughLinesP(edges,1,np.pi/180,40,minLineLength=80, maxLineGap=20)
newImage = image.copy()

for line in lines:
    x1,y1,x2,y2 = line[0]
    cv2.line(newImage,(x1,y1),(x2,y2),(0,255,0),3)
    
plt.figure()
plt.imshow(newImage)

#find conituous line segment from multiple line obtained from hough transform

def line_segments(lines):
    
    m = np.zeros((lines.shape[0],2),dtype=float)
    lines_sort = []
    
    #return m (slope and intercept of line obtained from hough transform)
    for i in range (lines.shape[0]):

        m[i,0]= np.float32(lines[i,0,3]-lines[i,0,1] )/ ( lines[i,0,2]-lines[i,0,0] )
        m[i,1]= np.float32(lines[i,0,3] - m[i,0]*lines[i,0,2])
        
    #return find m and b common line segments
    check = np.array((0,3), int)
    i_list = np.empty((lines.shape[0]*2,0))
    
    for i in range (lines.shape[0]):
        i_list = np.append(i_list,1000)
        
        if (i not in i_list):
            i_list = np.append(i_list,i)
            
        for j in range(i+1,lines.shape[0]): 
            if (j not in i_list):
                if abs(m[i,0]-m[j,0])<0.2 and abs(m[i,1]-m[j,1])<15:   #input in def
                    i_list = np.append(i_list,j)
                    check  = np.append(check,j)
    
    final_list = np.empty((0,4),dtype=float)
   
    #find xmax, xmin , m, b of line segments
    for i in range (i_list.shape[0]):
        if (i_list[i] == 1000):
            for j in range(i+1,i_list.shape[0]):
                if (i_list[j] == 1000):
                    if i != j-1: 
                    
                        a=i_list[i+1:j]
                        a=a.astype(int)

                        x_max  = max( max(lines[a,0,0]),max(lines[a,0,2]) )
                        x_min  = min( min(lines[a,0,0]),min(lines[a,0,2]) )
                        m_line = m[(a[0]),0]
                        b_line = m[(a[0]),1]
                    
                        final_list = np.append(final_list,[x_max,x_min,m_line,b_line])
                        break
                    break     

    final_list2 = final_list.reshape((final_list.shape[0]/4),4)

    lines_xymb = np.empty((0,4), dtype=float)
    for i in range (final_list.shape[0]/4):
        x1 = final_list[i*4]
        y1 = final_list[i*4]*final_list[i*4+2]+final_list[i*4+3]
        
        x2 = final_list[i*4+1]
        y2 = final_list[i*4+1]*final_list[i*4+2]+final_list[i*4+3]
        
        m_slope = final_list[i*4+2]
        b_in    = final_list[i*4+3]
        
        lines_xymb = np.append(lines_xymb,[x1,y1,x2,y2,m_slope,b_in])
            
    lines_xymb = lines_xymb.reshape((lines_xymb.shape[0]/6),6)    
    return lines_xymb
            
a1= line_segments(lines) #unique_lines = [0,1 16,2 7,3 4,5 8 18,6 11 14,9,10 12 15,13,17,] 
 

index_arr = np.empty(())
new_a1 = np.empty((0,6))

reshape_a1 = a1.shape[0]

for i in range (a1.shape[0]):
    j = np.argmin(a1[:,0])
    new_a1 = np.append(new_a1,a1[j])
    a1 = np.delete(a1,j, 0)

a1      = new_a1.reshape(reshape_a1,6)
a1_temp = a1[5]
a1=np.delete(a1,5,0)
a1=(np.append(a1,a1_temp)).reshape(reshape_a1,6)

a1[:,4]

for i in range(a1.shape[0]):
    
    #lowest edge points
    center = (int(a1[i,2]),int(a1[i,3]))
    img_circle =  cv2.circle(image, center, 8, (255,0,0),2)
    
    #higher edge points
    center2 = (int(a1[i,0]),int(a1[i,1]))
    img_circle =  cv2.circle(image, center2, 8, (255,0,0),2)

plt.imshow(img_circle)


def line_intersection(line11, line22):
    
    line1 =[ [line11[0],line11[1]],[line11[2],line11[3]] ]
    line2 =[ [line22[0],line22[1]],[line22[2],line22[3]] ]
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    
    if div != 0:
        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return x, y

intersection_points = np.empty((0,3))

for i in range (a1.shape[0]-1):
    line22 = a1[6]
    line11 = a1[i]
    x,y = line_intersection(line11, line22)
    intersection_points = np.append(intersection_points,[x,y,i])
  
intersection_points=intersection_points.reshape((intersection_points.shape[0]/3),3)
intersection_points

for i in range(intersection_points.shape[0]):
    
    #lowest edge points
    center = (int(intersection_points[i,0]),int(intersection_points[i,1]))
    img_circle =  cv2.circle(image, center, 8, (0,0,235),2)

plt.imshow(img_circle)

# a1 ( xlow,ylow, x high, y high)
# intersection_points

#draw rectangle top left, bottom right
for i in range (intersection_points.shape[0]-1): 
    top_left     = [int(intersection_points[i,0]),int(intersection_points[i,1])]
    top_right    = [int(intersection_points[i+1,0]),int(intersection_points[i+1,1])]
    bottom_left  = [int(a1[i,0]), int(a1[i,1])]
    bottom_right = [int(a1[i+1,0]), int(a1[i+1,1])]
    bottom_left1  = [int(a1[i,2]), int(a1[i,3])]
    bottom_right1 = [int(a1[i+1,2]), int(a1[i+1,3])]
    
    
    pts = np.array([bottom_left,top_left ,top_right ,bottom_right],np.int32)
    pts2 = np.array([bottom_left1,top_left ,top_right ,bottom_right1],np.int32)
    
    pts = pts.reshape((-1,1,2))
    pts2 = pts2.reshape((-1,1,2))
    
    img_circle = cv2.fillPoly(img_circle,[pts],(255-33*i,10,255-40*i))
    img_circle = cv2.fillPoly(img_circle,[pts2],(2,23*i,10))
    
plt.imshow(img_circle )





