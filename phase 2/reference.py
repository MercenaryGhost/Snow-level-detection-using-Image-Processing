#!/usr/bin/env python
# coding: utf-8

# In[259]:


import numpy as np
import cv2
                # we will read the image directly as a grayscale image
#img = cv2.imread('reference 1.jpg', cv2.IMREAD_UNCHANGED) # gaussian filter is not used explicitly as it is already included in the canny function itself
img = cv2.imread('field 1-2.jpg', cv2.IMREAD_UNCHANGED)
roi = img[900:1650, 350:550]
roi = cv2.GaussianBlur(roi,(5,5),0)
roi2 = roi.copy()
roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

#b,g,r = cv2.split(roi)
#roi = cv2.Sobel(roi,cv2.CV_8U,1,1)
#roi = cv2.adaptiveThreshold(roi,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,15)
roi = cv2.Canny(roi,200,550,3)


# In[20]:


img.shape


# In[254]:


lines = cv2.HoughLinesP(roi,1,np.pi/180,200,maxLineGap = 80)  # HoughLines function gives an output of a 2d-array containing rho and theta values for the corresponding line equations which crossed the vote limit of 200.  
no_of_lines = len(lines)
start_point = (int((lines[0][0][0]+lines[1][0][0])/2), int((lines[0][0][1]+lines[1][0][1])/2))
end_point = (int((lines[0][0][2]+lines[1][0][2])/2), int((lines[0][0][3]+lines[1][0][3])/2))

cv2.line(roi2, start_point, end_point, (0,255,0), 1)
cv2.line(roi2, (81,680), (29,624), (0,255,0), 1)
cv2.line(roi2, (99,149),(35,160), (0,255,0), 1)
cv2.line(roi2, (29,624),(35,160), (0,255,0), 1)
pole_length = (6**2 + 464**2)**0.5

cv2.putText(roi2, str(pole_length), (43,442), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
cv2.putText(roi2, "angle=89.26", (43,470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
print(start_point,end_point,lines)


# In[243]:


cv2.line(roi2, (81,683), (89,415), (0,255,0), 1)
pixel_length = (8**2 + 268**2)**0.5
cv2.putText(roi2, str(pixel_length/2), (95,550), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
pixel_length


# In[6]:


line_lengths = np.zeros(no_of_lines, dtype = float)

for i in range(0,no_of_lines):
    y = lines[i][0][3] - lines[i][0][1]
    x = lines[i][0][2] - lines[i][0][0]
    line_lengths[i] = (y**2 + x**2)**0.5
    
line_lengths


# In[192]:


cv2.namedWindow('img',cv2.WINDOW_NORMAL)
cv2.imshow('img',img)

cv2.waitKey(0)


# In[260]:


cv2.namedWindow('roi',cv2.WINDOW_NORMAL)
cv2.imshow('roi',roi)

cv2.waitKey(0)


# In[256]:


#cv2.imwrite("field contour outline.jpg", roi2)
#cv2.imwrite("field hough line.jpg", roi)
#cv2.imwrite("Image pixel to metric ratio.jpg", roi2)
#cv2.imwrite("pole inital length.jpg", roi2)


# In[ ]:


snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

print("[ Top 5 ]")
for stat in top_stats[:10]:
    print(stat)

