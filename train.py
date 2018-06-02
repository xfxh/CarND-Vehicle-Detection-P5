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
from help_functions import *
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
# from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split


# Read in cars and notcars
#images are divided up into vehicles and non-vehicles folders (each of whitch contains subfolders)
#first locate vehicle images
basedir = 'vehicles/'
#different folders represent different sources for images e.g. GTI, Kitti.
image_types = os.listdir(basedir)
cars = []
for imtype in image_types:
	cars.extend(glob.glob(basedir+imtype+'/*'))

print('Number of Vehicle Images found:',len(cars))
with open("cars.txt",'w') as f:
	for fn in cars:
		f.write(fn+'\n')

#do the same thing for non-vehicle images
basedir = 'non-vehicles/'
image_types = os.listdir(basedir)
notcars = []
for imtype in image_types:
	notcars.extend(glob.glob(basedir+imtype+'/*'))

print('Number of Non-Vehicle Images found:',len(notcars))
with open("notcars.txt",'w') as f:
	for fn in notcars:
		f.write(fn+'\n')

#Tweak these parameters and see how the results change.
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL" Need care!!!!!!
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [None, None] # Min and max in y to search in slide_window()

# #choose random car /not -car indices
# car_ind = np.random.randint(0, len(cars))
# notcar_ind = np.random.randint(0, len(notcars))

# #read in car / not-car images
# car_image = mpimg.imread(cars[car_ind])
# notcar_image = mpimg.imread(notcars[notcar_ind])

# car_features, car_hog_image = single_img_features(car_image, color_space=color_space, spatial_size=spatial_size,
#                                                   hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell,
#                                                    cell_per_block=cell_per_block, hog_channel=0, spatial_feat=spatial_feat,
#                                                    hist_feat=hist_feat, hog_feat=hog_feat, vis=True)
# notcar_features, notcar_hog_image = single_img_features(notcar_image, color_space=color_space, spatial_size=spatial_size,
#                                                   hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell,
#                                                    cell_per_block=cell_per_block, hog_channel=0, spatial_feat=spatial_feat,
#                                                    hist_feat=hist_feat, hog_feat=hog_feat, vis=True)
# images = [car_image,car_hog_image,notcar_image,notcar_hog_image]
# images1 = [car_image,notcar_image]
# titles = ['car image', 'car HOG image','notcar_image', 'notcar HOG image']
# titles1 = ['car image', 'notcar image']
# fig = plt.figure(figsize=(8,3))
# visualize(fig, 1, 2, images1, titles1)
# Check the extract features time
t=time.time()
car_features = extract_features(cars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)

notcar_features = extract_features(notcars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)

t2 = time.time()
print(round(t2-t, 2), 'Seconds to extract features...')

# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=rand_state)

# Fit a per-column scaler
X_scaler = StandardScaler().fit(X_train)
# Apply the scaler to X
X_train = X_scaler.transform(X_train)
X_test = X_scaler.transform(X_test)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))

# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
Save model and var
dist_pickle = {}
dist_pickle['svc'] = svc
dist_pickle['scaler'] = X_scaler
dist_pickle['orient'] = orient
dist_pickle['pix_per_cell'] = pix_per_cell
dist_pickle['cell_per_block'] = cell_per_block
dist_pickle['spatial_size'] = spatial_size
dist_pickle['hist_bins'] = hist_bins
with open('svc_pickle.pickle','wb') as fw:
	pickle.dump(dist_pickle, fw)

# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

# Check the prediction time for a single sample
t=time.time()

image = mpimg.imread('test_images/test1.jpg')     


ystart = 400
ystop = 656
scale = 1.5

out_img = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)


t2 = time.time()
print(round(t2-t, 2), 'Seconds to predict')

plt.imshow(out_img)
plt.show()
