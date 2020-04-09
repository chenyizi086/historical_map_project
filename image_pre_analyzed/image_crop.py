# import the necessary packages

import numpy as np
import os
import pickle
from pathlib import Path

import cv2.cv2 as cv2
from PIL import Image
import matplotlib.pyplot as plt
from keras.preprocessing import image
import image_slicer
from image_slicer import join


def create_save_image_batches(path, batch_size=(1000, 1000)):
	'''
	Description: Crop and save image batches into files
	Parameters
	----------
	path: image director
	batch_size: approximate size for each image

	Returns
	-------
	None
	'''
	# List file names
	file_names = os.listdir(path)
	
	for file in file_names:
		# Split path and get the filename
		file_index = file.split('.')[0]
		# Create target directory & all intermediate directories if don't exists
		if not os.path.exists(file_index):
			os.makedirs(file_index)
			print("Directory ", file_index, " Created ")
		else:
			print("Directory ", file_index, " already exists")
		
		# Read image check directory exist or not
		dir_save_image_batch = str(Path(os.getcwd()).parent) + '/image_generator/' + str(file.split('.')[0]) + '/image_batches/'
		if not os.path.exists(dir_save_image_batch):
			os.makedirs(dir_save_image_batch)
			print("Directory ", dir_save_image_batch, " Created ")
		else:
			print("Directory ", dir_save_image_batch, " already exists")
		
		file_orinal_image = path + '/' + file
		
		# Get size of image
		image_shape = np.array(cv2.imread(file_orinal_image)).shape
		batch_number = int(np.floor(image_shape[0]/batch_size[0])) * int(np.floor(image_shape[1]/batch_size[1]))
		
		# Slice and save tiles
		tiles = image_slicer.slice(file_orinal_image, batch_number, save=False)
		
		# Join the tiles
		image = join(tiles)
		
		plt.imshow(image)
		plt.show()
		
		# Save tile file in pickle
		tile_path = str(Path(os.getcwd()).parent) + '/image_generator/' + file_index + '/tile_info/'
		if not os.path.exists(tile_path):
			os.makedirs(tile_path)
			print("Directory ", tile_path, " Created ")
		else:
			print("Directory ", tile_path, " already exists")

		tile_file_path = tile_path + file_index + '_tile.p'
		# / home / yizi / Documents / phd / historical_map_project / image_generator / BHdV_PL_ATL20Ardt_1929_0003 / tile_info
		image_slicer.save_tiles(tiles, directory=dir_save_image_batch, format='JPEG')
		pickle.dump(tiles, open(tile_file_path, "wb"))


def join_image_batches(path):
	# Read the tile information
	tile_file_path = path + 'tile_info/' + os.listdir(path+'tile_info/')[0]
	
	# Load the pickles information
	df = open(tile_file_path, 'rb')
	tiles = pickle.load(df)
	image_batches_path = path + 'color_quantization_result_batches/'
	
	for index in range(1, len(os.listdir(image_batches_path))+1):
		join_image_bath_path = image_batches_path + os.listdir(image_batches_path)[index-1] + '/'
		for t in tiles:
			with open(join_image_bath_path+t.filename.split('/')[-1].split('.')[0]+'.p', 'rb') as handle:
				image_array = pickle.load(handle)
			t.image = Image.fromarray(image_array)
			t.filename = join_image_bath_path+t.filename.split('/')[-1]
		
		image_join_dir_path = path + 'color_quantization_result_join/' + os.listdir(image_batches_path)[index-1] + '/'
		
		if not os.path.exists(image_join_dir_path):
			os.makedirs(image_join_dir_path)
			print("Directory ", image_join_dir_path, " Created ")
		else:
			print("Directory ", image_join_dir_path, " already exists")
		
		# Uncompress data format 'tif'
		image_join_file_dir = image_join_dir_path + os.listdir(image_batches_path)[index-1] + '.tif'
		
		image_join = join(tiles)
		image_join.save(image_join_file_dir)
		
		plt.imshow(image_join)
		plt.show()


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
