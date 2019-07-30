import matplotlib.pyplot as plt
import sys
import numpy as np
import cv2
import glob
import os
import re
import fnmatch

def hazy_image(rgbImage):
	[rows, columns] = rgbImage.shape
	maxValue = float(np.amax(rgbImage))
	rgbImage.astype(float)
	rgbImage2 = np.add(rgbImage, 350)
	maxValue2 = float(np.amax(rgbImage2))
	mulval = maxValue / maxValue2
	rgbImage2 = np.multiply(rgbImage2, mulval)
	return rgbImage2
 
def compare_images(imageA, imageB):
    imageA1 = imageA - imageA.mean()
    imageB1 = imageB - imageB.mean()
    product = np.mean((imageA1) * (imageB1))
    stds = imageA.std() * imageB.std()
    if stds is not 0:
        return product / stds
    else:
    	return 0

################################################################################################

frame_set = []
FRAMES_DIR = sys.argv[2]
# load images, and start other processing
image_names = []
for root, dirnames, filenames in os.walk(FRAMES_DIR):
	for filename in fnmatch.filter(filenames, "*.*"):
		image_names.append((os.path.join(root, filename),filename))
for image_name,filename in image_names:
	img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE).astype(np.float64)
	frame_set.append((image_name,img,filename))

################################################################################################
SLIDES_DIR  = sys.argv[1]
slide_set = []
# load images, and start other processing
image_names = []
for root, dirnames, filenames in os.walk(SLIDES_DIR):
	for filename in fnmatch.filter(filenames, "*.*"):
		image_names.append((os.path.join(root, filename),filename))
for image_name,filename in image_names:
	img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE).astype(np.float64)
	img = hazy_image(img)
	slide_set.append((image_name,img,filename))



################################################################################################

fp= open("20171192_20171191_20171064.txt","w+")

try:
	for i in range(len(frame_set)):
		frame = cv2.imread(frame_set[i][0])
		height, width, channels = frame.shape
		dim = (width, height)
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		match_slide = ''
		max_match = -10
		for j in range(len(slide_set)):
			slide = cv2.imread(slide_set[j][0])
			slide = cv2.resize(slide, dim, interpolation = cv2.INTER_AREA) 
			slide = cv2.cvtColor(slide, cv2.COLOR_BGR2GRAY)
			if max_match < compare_images(frame, slide):
				max_match = compare_images(frame, slide)
				match_slide = slide_set[j][2]	
		fp.write(frame_set[i][2]+" "+match_slide+"\n")
	fp.close()
except:
	fp.close()

#################################################################################################