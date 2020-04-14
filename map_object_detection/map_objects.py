import numpy as np
import time
import multiprocessing as mp
from functools import partial
import pickle
import os
from pathlib import Path

import cv2.cv2 as cv2
import matplotlib.pyplot as plt


def scale(x, x_min, x_max):
	nom = (x-x.min(axis=0))*(x_max-x_min)
	denom = X.max(axis=0) - X.min(axis=0)
	denom[denom == 0] = 1
	return x_min + nom/denom


def pixel_overlap_percentage(image, line_segment, overlap_threshold=0.8):
	# Detecting horizontal and vertical lines in the images.
	blank_image_gray = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
	
	rho, theta = line_segment[0]
	# Set threshold based on the angle theta
	# if np.abs((np.pi / 2) - theta) < 0.02 or np.abs(0 - theta) < 0.02:
	# if (np.pi / 2) - 0.02 < theta < (np.pi / 2) + 0.02 or - 0.02 < theta < 0.02:
	a = np.cos(theta)
	b = np.sin(theta)
	x0 = a * rho
	y0 = b * rho
	x1 = int(x0 + 1000 * (-b))
	y1 = int(y0 + 1000 * (a))
	x2 = int(x0 - 1000 * (-b))
	y2 = int(y0 - 1000 * (a))
	
	img_line = cv2.line(blank_image_gray, (x1, y1), (x2, y2), 1, 1)
	img_overlap = np.logical_and(img_line, image).astype(np.uint8)
	
	# Calculate overlapped pixel percentage
	number_horizontal_pixels = image.shape[0]
	
	# number_original_pixels = list(img_line.flatten()).count(1)
	number_overlapped_pixels = list(img_overlap.flatten()).count(1)
	percentage = number_overlapped_pixels / number_horizontal_pixels
	
	if overlap_threshold <= percentage <= 1:
		line = ((x1, y1), (x2, y2))
	else:
		line = None
	return line
	

# Detect longitude and latitude
def detect_longitude_latitude(image_path):
	print('Start detecting longitude and latitude')
	image = pickle.load(open(image_path, "rb"))
	
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	# Invert gray scale image
	gray_image = 255 - np.array(gray_image)
	gray_image = scale(gray_image, 0, 1).astype(np.uint8)
	
	gray_image_copy = gray_image.copy()
	
	# Detecting horizontal and vertical lines in the images.
	output_image_gray = np.zeros((gray_image.shape[0], gray_image.shape[1]), dtype=np.uint8)
	
	# Hough transform
	horizontal_lines = cv2.HoughLines(gray_image_copy, 1, np.pi / 180, 200, min_theta=(np.pi / 2)-0.02, max_theta=(np.pi / 2)+0.02)
	vertical_lines_1 = cv2.HoughLines(gray_image_copy, 1, np.pi / 180, 200, min_theta=-0.02, max_theta=0.02)
	vertical_lines_2 = cv2.HoughLines(gray_image_copy, 1, np.pi / 180, 200, min_theta=np.pi-0.02, max_theta=np.pi)
	
	vertical_lines = np.concatenate((vertical_lines_1, vertical_lines_2), axis=0)
	
	no_lines = False
	if horizontal_lines is None:
		if vertical_lines is None:
			selected_lines = None
			no_lines = True
			print('Not found any lines')
		else:
			selected_lines = vertical_lines
			print('Not found any horizontal lines')
	else:
		if vertical_lines is None:
			selected_lines = horizontal_lines
			print('Not found any vertical lines')
		else:
			# Cocatenated line change data shape
			selected_lines = np.concatenate((horizontal_lines, vertical_lines), axis=0)
	
	if not(no_lines):
		start_time = time.time()
		print("--- start ---")
		
		# Multi-processing
		pool = mp.Pool(14)
		func = partial(pixel_overlap_percentage, gray_image)
		lines = pool.map(func, selected_lines)
		
		# Clean memory and collect garbage
		pool.close()
		pool.join()
		
		print("--- %s seconds ---" % (time.time() - start_time))
		print("\n")
	
		# Remove none value in list
		lines = [l for l in lines if l]
		for l in lines:
			cv2.line(output_image_gray, l[0], l[1], 1, 2)
		print(lines)
		
	# Draw lines
	fig = plt.figure(figsize=(25, 25))
	
	ax = fig.add_subplot(1, 2, 1)
	ax.imshow(gray_image, cmap='gray')
	ax.axis('off')
	ax.set_title('Orignal gray-scale image')
	
	ax = fig.add_subplot(1, 2, 2)
	ax.imshow(output_image_gray, 'gray')
	ax.axis('off')
	ax.set_title('Extracted longitude and latitude')
	plt.show()
	
	current_dir = os.getcwd()
	image_name = image_path.split('/')[-4]
	file_name = image_path.split('/')[-1]
	image_quantization_result_dir = str(Path(current_dir).parent) + '/image_generator/' + image_name + \
									'/color_quantization_result_batches/' + 'logitude_and_latitude/'

	if not os.path.exists(image_quantization_result_dir):
		os.makedirs(image_quantization_result_dir)
		print("Directory ", image_quantization_result_dir, " Created ")
	else:
		print("Directory ", image_quantization_result_dir, " already exists")

	save_path = image_quantization_result_dir + file_name

	# Save image into pickle file for saving memeory
	output_image_gray = scale(output_image_gray, 0, 255).astype(np.uint8)
	output_image_gray = cv2.cvtColor(output_image_gray, cv2.COLOR_GRAY2RGB)
	with open(save_path, 'wb') as handle:
		pickle.dump(output_image_gray, handle, protocol=pickle.HIGHEST_PROTOCOL)
	
	return output_image_gray
