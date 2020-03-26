#!/usr/bin/env python
# coding: utf-8

# # HW03
# ### Question 1: Select a frame (or a few frames) of LiDAR data file, parse the file and visualize the 3D point cloud of this frame, colored by its reflectivity value.

# In[13]:


import numpy as np 
import argparse

filename   = str("bin_files/002_00000021.bin")
pointcloud = np.fromfile(filename, dtype=np.float32)
pointcloud = pointcloud.reshape([-1,4])

print('LiDAR data loaded as a variable pointcloud')

str1= str('\nLidar data file : ') + str(filename) + str('\nSize of pointcloud data = ') + str(pointcloud.shape)
print(str1)


# In[11]:


def visualize_3d(pointcloud,cloud_color,Point_size):
    import pptk
    import numpy as np 
    
    # Extract first three points as x y z inputs and reflectivity value
    P = pointcloud[:,0:3]
    
    a = pointcloud.shape[0]
    R = np.ones((a))*20
    
    if pointcloud.shape[1]==4:
        R   = pointcloud[:,3]               # take intensity values from pointcloud for plot
    # 
    rgb = np.ones((P.shape))*cloud_color    # for grayish effect based on reflectivity[200,200,200]
    
    rgb[:,0] = rgb[:,0]*(255-R)/255
    rgb[:,1] = rgb[:,1]*(255-R)/255
    rgb[:,2] = rgb[:,2]*(255-R)/255
    
    # Visualize point cloud
    v = pptk.viewer(P)
    v.attributes(rgb / 255, R)
    v.set(lookat = [0,0,0])
    v.set(point_size = Point_size)        #for better visualization point_size = 0.001
    v.color_map('jet', scale=[0, 5])


# In[14]:


visualize_3d(pointcloud,[200,200,200],0.001)

