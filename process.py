import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import os
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
import pickle
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
from Vehicle import Vehicle
from help_functions import *



#load the saved model
with open('svc_pickle.pickle','rb') as fr:
	dist_pickle = pickle.load(fr)

svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]



def process_image(img):
    # the box list for all the regions
    oneimg_box=[]
    boxes_small= find_cars_box_r(img, 380, 500, 1.5, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    oneimg_box.extend(boxes_small)
    boxes_middle= find_cars_box_r(img, 500, 656, 2.3, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    oneimg_box.extend(boxes_middle)
    #push back one frame detected result into the recent boxes list
    total_boxes, num = car_box.append_box(oneimg_box)
    #draw all the recent boxes in one frame and draw the heat map
    draw_img, heatmap = apply_box_r(img,total_boxes)
    #apply threshould to the heat map
    heatmap = apply_threshold(heatmap,2*num)
    #label the heat map for find the cars
    labels = label(heatmap)
    #draw the optimized boxes on one frame
    final_img = draw_labeled_bboxes(img,labels)

    return final_img


car_box = Vehicle()
test_output = 'output_video/output_project.mp4'
clip1 = VideoFileClip("project_video.mp4")
test_clip = clip1.fl_image(process_image)
test_clip.write_videofile(test_output, audio=False)