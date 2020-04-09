import numpy as np
import time
import multiprocessing as mp
from functools import partial

import cv2.cv2 as cv2
import matplotlib.pyplot as plt


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
	
	img_line = cv2.line(blank_image_gray, (x1, y1), (x2, y2), 1, 2)
	img_overlap = np.logical_and(img_line, image).astype(np.uint8)
	
	# Calculate overlapped pixel percentage
	number_original_pixels = list(img_line.flatten()).count(1)
	number_overlapped_pixels = list(img_overlap.flatten()).count(1)
	percentage = number_overlapped_pixels / number_original_pixels
	
	if percentage > overlap_threshold:
		line = ((x1, y1), (x2, y2))
	else:
		line = None
	return line


def pixel_overlap_percentage_probability(image, line_segment, overlap_threshold=0.8):
	# Detecting horizontal and vertical lines in the images.
	blank_image_gray = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
	x1, y1, x2, y2 = line_segment[0]
	img_line = cv2.line(blank_image_gray, (x1, y1), (x2, y2), 1, 2)
	img_overlap = np.logical_and(img_line, image).astype(np.uint8)
	
	# Calculate overlapped pixel percentage
	number_original_pixels = list(img_line.flatten()).count(1)
	number_overlapped_pixels = list(img_overlap.flatten()).count(1)
	percentage = number_overlapped_pixels / number_original_pixels
	
	if percentage > overlap_threshold:
		line = ((x1, y1), (x2, y2))
	else:
		line = None
	return line


def filter_lines(image, line_segment):
	fil_lines = []
	output_image_gray = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
	for l in line_segment:
		x1, y1, x2, y2 = l[0]
		theta = (y2 - y1) / (x2 - x1)
		if (np.pi / 2) - 0.02 < theta < (np.pi / 2) + 0.02 or - 0.02 < theta < +0.02:
			fil_lines.append(l[0])
			cv2.line(output_image_gray, (l[0][0], l[0][1]), (l[0][2], l[0][3]), 1, 2)
	plt.imshow(output_image_gray)
	plt.show()
	return fil_lines
	

# Detect longitude and latitude
def detect_longitude_latitude(gray_image):
	print('Start detecting longitude and latitude')
	# Detecting horizontal and vertical lines in the images.
	output_image_gray = np.zeros((gray_image.shape[0], gray_image.shape[1]), dtype=np.uint8)
	
	# Hough transform
	horizontal_lines = cv2.HoughLines(gray_image, 1, np.pi / 180, 200, min_theta=(np.pi / 2) - 0.02, max_theta=(np.pi / 2)+0.02)
	vertical_lines = cv2.HoughLines(gray_image, 1, np.pi / 180, 200, min_theta=-0.02, max_theta=0.02)
	
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
	ax.imshow(gray_image)
	ax.axis('off')
	ax.set_title('Orignal gray-scale image')
	
	ax = fig.add_subplot(1, 2, 2)
	ax.imshow(output_image_gray)
	ax.axis('off')
	ax.set_title('Extracted longitude and latitude')
	plt.show()
	return output_image_gray


if __name__ == '__main__':
	image = cv2.imread('/home/yizi/Documents/phd/historical_map_project/image_generator/BHdV_PL_ATL20Ardt_1929_0003/color_quantization_result_join/3_layer/3_layer.tif')

	gray_scale_image = cv2.bitwise_not(image)
	gray_scale_image = cv2.cvtColor(gray_scale_image, cv2.COLOR_BGR2GRAY)
	gray_scale_image = np.array(gray_scale_image)  # Change image objects into array
	
	gray_scale_image_1 = gray_scale_image[500:1000, 500:1000]
	plt.imshow(gray_scale_image_1, cmap='gray')
	plt.show()
	gray_image = detect_longitude_latitude(gray_scale_image_1)
