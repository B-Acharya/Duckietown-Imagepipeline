# Duckietown-Imagepipeline
Contains the imaging pipeline of a Line detector node in Duckietown-Lanefollowing model  from www.duckietown.org
## Input Image
![Input Image](Images/image4.png)
## Color Segmentation
![](Images/colorSegment.png)
## Edge Detection
![](Images/edge.png)
## Image with normals point and detected lines
![](Images/image_with_lines.png)
The following gif should give a idea as to how the algorithm works.The heat map on the top is a representation of the  values corresponding to the estimates distance and angle with respect to the right lane center.Each line segment that is detected is used to get the estimate.The distance and angle that has the highest vote is considered for the estimation.
![](ezgif.com-gif-maker.gif )

