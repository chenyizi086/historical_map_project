import numpy as np
import time
import multiprocessing as mp
from functools import partial
import pickle
import os
from pathlib import Path

import cv2.cv2 as cv2
import matplotlib.pyplot as plt
from scipy import signal


def scale(x, x_min, x_max):
	nom = (x-x.min(axis=0))*(x_max-x_min)
	denom = x.max(axis=0) - x.min(axis=0)
	denom[denom == 0] = 1
	return x_min + nom/denom


def pixel_overlap_percentage(image, line_segment):
	# Detecting horizontal and vertical lines in the images.
	blank_image_gray = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
	
	rho, theta = line_segment[0]
	
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
	
	number_overlapped_pixels = list(img_overlap.flatten()).count(1)
	percentage = number_overlapped_pixels / number_horizontal_pixels
	line = ((x1, y1), (x2, y2))
	return line, percentage
	

# Detect longitude and latitude
def detect_longitude_latitude_hough_line_transform(image_path, threshold=0.6):
	print('Start detecting longitude and latitude')
	image = pickle.load(open(image_path, "rb"))
	
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	plt.imshow(gray_image, cmap='gray')
	plt.show()
	# Invert gray scale image
	# gray_image = 255 - np.array(gray_image)
	gray_image = scale(gray_image, 0, 1).astype(np.uint8)
	
	gray_image_copy = gray_image.copy()
	
	# Detecting horizontal and vertical lines in the images.
	output_image_gray = np.zeros((gray_image.shape[0], gray_image.shape[1]), dtype=np.uint8)
	
	# Hough transform
	horizontal_lines = cv2.HoughLines(gray_image_copy, 1, np.pi / 180, 200, min_theta=(np.pi / 2)-0.01, max_theta=(np.pi / 2)+0.01)
	vertical_lines = cv2.HoughLines(gray_image_copy, 1, np.pi / 180, 200, min_theta=-0.02, max_theta=0.02)
	
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
		lines = {l for l in lines if l}
		lines = {l: p for (l, p) in lines}
		max_percentage_line = max(list(lines.values()))
		for l, l_p in lines.items():
			if l_p >= max_percentage_line * threshold:
				(x1, y1), (x2, y2) = l
				k = (y2-y1)/(x2-x1)
				b = y1 - k * x1
				if 0 <= b <= image.shape[1]:
					# y = kx + b
					edge_point_1 = (0, int(y1 - k * x1))
					edge_point_2 = (image.shape[1], int(k * image.shape[1] + b))
				else:
					# y = kx + b
					edge_point_1 = (int((y1 - b)/k), 0)
					edge_point_2 = (int((image.shape[0]-b)/k), image.shape[0])
				
				cv2.line(output_image_gray, edge_point_1, edge_point_2, 1, 2)
			else:
				pass
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


def detect_longitude_latitude_convolution_based(image_path):
	print('Start detecting longitude and latitude')
	image = pickle.load(open(image_path, "rb"))

	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	# Invert gray scale image
	gray_image = 255 - np.array(gray_image)
	gray_image = scale(gray_image, 0, 1).astype(np.uint8)
	
	gray_image_copy = gray_image.copy()
	kernel = np.ones((5, 5), np.float32) / 25
	gray_image_copy = cv2.filter2D(gray_image_copy, -1, kernel)
	plt.imshow(gray_image_copy)
	plt.show()
	
	# Detecting horizontal and vertical lines in the images.
	output_image_gray = np.zeros((gray_image.shape[0], gray_image.shape[1]), dtype=np.uint8)
	
	# Define vertical and horizontal kernel
	vertical_kernel = np.array([-1, 2, -1, -1, 2, -1, -1, 2, -1]).reshape(3, 3).astype(np.uint8)
	horizontal_kernel = np.array([-1, -1, -1, 2, 2, 2, -1, -1, -1]).reshape(3, 3).astype(np.uint8)
	
	horizontal_conv = signal.convolve2d(gray_image_copy, horizontal_kernel, boundary='symm', mode='same')
	
	plt.imshow(horizontal_conv)
	plt.show()
	print()


if __name__ == '__main__':
	# Test: _01_02;
	image_path = '/home/yizi/Documents/phd/historical_map_project/image_generator/BHdV_PL_ATL20Ardt_1929_0003/color_quantization_result_batches/2_layer/_05_06.p'
	# image_path = '/home/yizi/Documents/phd/historical_map_project/image_generator/BHdV_PL_ATL20Ardt_1929_0003/color_quantization_result_batches/0_layer/_01_02.p'
	detect_longitude_latitude_hough_line_transform(image_path)