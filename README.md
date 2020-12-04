# AuE8930_Perception_and_Intelligence

## Intereseting problems from Homework

### 1) Design and implement the approaches to find all park space frames with the four vertex points of each frame.( without deep learning algorithms) 

1.1 Canny transform on gray scale image to extract out edges.
<p align="center">
  <img width="400" height="250" src="https://github.com/vipulkumbhar/AuE893_Perception_and_Intelligence/blob/master/HW02/Result_images/Screen%20Shot%202020-12-04%20at%203.32.00%20PM.png">
</p>

1.2 Hough transform to find lines in image and superimpose it on original image
<p align="center">
  <img width="400" height="250" src="https://github.com/vipulkumbhar/AuE893_Perception_and_Intelligence/blob/master/HW02/Result_images/Screen%20Shot%202020-12-04%20at%203.32.12%20PM.png">
</p>

1.3 Finding out continuous lines from various line segments and finding out end points
<p align="center">
  <img width="400" height="250" src="https://github.com/vipulkumbhar/AuE893_Perception_and_Intelligence/blob/master/HW02/Result_images/Screen%20Shot%202020-12-04%20at%203.32.21%20PM.png">
</p>

1.4 Line intersections (blue circles)
<p align="center">
  <img width="400" height="250" src="https://github.com/vipulkumbhar/AuE893_Perception_and_Intelligence/blob/master/HW02/Result_images/Screen%20Shot%202020-12-04%20at%203.32.27%20PM.png">
</p>

1.5 Mapping rectangles in parking space
<p align="center">
  <img width="400" height="250" src="https://github.com/vipulkumbhar/AuE893_Perception_and_Intelligence/blob/master/HW02/Result_images/Screen%20Shot%202020-12-04%20at%203.32.37%20PM.png">
</p>

### 2) LiDAR data processing

2.1 Point cloud view
<p align="center">
  <img width="400" height="250" src="https://github.com/vipulkumbhar/AuE893_Perception_and_Intelligence/blob/master/HW03/result_images/Screen%20Shot%202020-12-04%20at%203.34.00%20PM.png">
</p>

2.2 Voxel filter (or box grid filter) to downsample all the 3D point cloud points to the 3D voxel space points
<p align="center">
  <img width="400" height="250" src="https://github.com/vipulkumbhar/AuE893_Perception_and_Intelligence/blob/master/HW03/result_images/Screen%20Shot%202020-12-04%20at%203.34.26%20PM.png">
</p>

2.3 RANSAC algorithm to the 3D voxel space points to find a ground plane model
<p align="center">
  <img width="400" height="250" src="https://github.com/vipulkumbhar/AuE893_Perception_and_Intelligence/blob/master/HW03/result_images/Screen%20Shot%202020-12-04%20at%203.34.37%20PM.png">
</p>

2.4 Remove all the ground planes points in the 3D voxel space points
<p align="center">
  <img width="400" height="250" src="https://github.com/vipulkumbhar/AuE893_Perception_and_Intelligence/blob/master/HW03/result_images/Screen%20Shot%202020-12-04%20at%203.34.50%20PM.png">
</p>

2.5 Off ground points with ground plane points
<p align="center">
  <img width="400" height="250" src="https://github.com/vipulkumbhar/AuE893_Perception_and_Intelligence/blob/master/HW03/result_images/Screen%20Shot%202020-12-04%20at%203.34.58%20PM.png">
</p>

2.6 Top view projection of point cloud data
<p align="center">
  <img width="400" height="250" src="https://github.com/vipulkumbhar/AuE893_Perception_and_Intelligence/blob/master/HW03/result_images/Screen%20Shot%202020-12-04%20at%203.35.18%20PM.png">
</p>

2.7 Front view image with color based on depth i.e. distance from ego vehicle
<p align="center">
  <img width="1000" height="175" src="https://github.com/vipulkumbhar/AuE893_Perception_and_Intelligence/blob/master/HW03/result_images/Screen%20Shot%202020-12-04%20at%203.35.38%20PM.png">
</p>




