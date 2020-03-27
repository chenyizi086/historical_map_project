# import the necessary packages

import numpy as np
import os

import cv2.cv2 as cv2
from PIL import Image
import matplotlib.pyplot as plt
from keras.preprocessing import image


def crop_and_create_image_batches(path, superpixel_image):
	'''
	Description: A function to crop image into batches based on superpixel images.
	
	Parameters
	----------
	path: Absolute path of the selected image.
	superpixel_segments: image created by superpixel algorithm, each group of identity number represent to each
	superpixel element.

	Returns
	-------
	normalize_window: A window size which represent to the biggest size of image batches.
	
	'''
	# Set of classes in the image of superpixels
	set_of_classes = set(superpixel_image.flatten())
	# Split path and get the filename
	file_index = path.split('/')[-1].split('.')[0]
	# Create target directory & all intermediate directories if don't exists
	if not os.path.exists(file_index):
		os.makedirs(file_index)
		print("Directory ", file_index, " Created ")
	else:
		print("Directory ", file_index, " already exists")
	
	# Crop the images based on classes
	index = 1
	for s in set_of_classes:
		image_map = superpixel_image == s
		
		# Apply mask to the image
		image_copy = image.copy()
		image_copy[:, :, 0] *= image_map.astype(np.uint8)
		image_copy[:, :, 1] *= image_map.astype(np.uint8)
		image_copy[:, :, 2] *= image_map.astype(np.uint8)
		
		min_x, max_y, max_x, min_y = find_bounding_box(image_map)
		
		# Crop image into batches
		image_batch = image_copy[min_x:max_x, min_y:max_y]
		
		normalize_window = (0, 0)
		if index == 0:
			normalize_window = ((max_x-min_x), (max_y-min_y))
		else:
			if (max_x-min_x)*(max_y-min_y) > normalize_window[0]*normalize_window[1]:
				normalize_window = ((max_x - min_x), (max_y - min_y))
		
		file_name = str(index) + ".png"
		file_path = os.path.join(str(file_index), file_name)
		cv2.imwrite(file_path, cv2.cvtColor(image_batch, cv2.COLOR_RGB2BGR))
		index += 1
		return normalize_window


def normalized_image_size(path, normalize_window):
	'''
	Description: A function to normalize all the image into same size for better comparing the simialrity between images
	and featureextraction.
	
	Parameters
	----------
	path: Relative path of the selected image
	normalize_window: Size of normalized window

	Returns
	-------
	None
	
	'''
	# Split path and get the filename
	file_name_orignal = path.split('/')[-1].split('.')[0]
	file_index = file_name_orignal + '_normalize'
	# Create target directory & all intermediate directories if don't exists
	if not os.path.exists(file_index):
		os.makedirs(file_index)
		print("Directory ", file_index, " Created ")
	else:
		print("Directory ", file_index, " already exists")
	
	file_dir = os.path.join(os.getcwd(), file_name_orignal) + '/'
	file_names = os.listdir(file_dir)
	file_names.sort()
	print(file_names)
	print('The number of croped images: ', len(file_names))
	
	for i, file_name in enumerate(file_names):
		abs_file_path = os.path.join(os.getcwd(), file_name_orignal) + '/' + file_name
		img = Image.open(abs_file_path)
		img = img.resize(normalize_window)
		img = np.array(img)
		
		file_path = os.getcwd() + '/' + os.path.join(str(file_index), str(file_name))
		cv2.imwrite(file_path, img)
		

def find_bounding_box(image_batch):
	'''
	Description: Find rectangle bounding box of image batch.
	
	Parameters
	----------
	image_batch: image batch cropped by superpixels

	Returns
	-------
	(min_x, max_y): Point in left top corner
	(max_x, min_y): Point in down right corner
	
	'''
	cord_x = []
	cord_y = []
	x, y = image_batch.shape
	for i in range(x):
		for j in range(y):
			if image_batch[i][j]:
				cord_x.append(i)
				cord_y.append(j)

	min_x = min(cord_x)
	max_y = max(cord_y)
	max_x = max(cord_x)
	min_y = min(cord_y)
	return min_x, max_y, max_x, min_y


def plot_image_batch(path):
	'''
	Description: Plot image batches
	
	Parameters
	----------
	path: Relative path of the selected image

	Returns
	-------
	None
	
	'''
	file_names = os.listdir(path)
	file_names.sort()
	print(file_names)
	print('The number of fruit images: ', len(file_names))
	
	c = 8
	r = len(file_names) // c + 1
	plt.figure(figsize=(70, 70))
	for i, file_name in enumerate(file_names):
		abs_file_path = path + file_name
		img = image.load_img(abs_file_path, target_size=(224, 224))
		x = image.img_to_array(img)
		x = x.astype('float32') / 255
		plt.subplot(r, c, i + 1)
		plt.title(i + 1)
		plt.xticks([])
		plt.yticks([])
		plt.imshow(x)
	plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
	plt.show()
