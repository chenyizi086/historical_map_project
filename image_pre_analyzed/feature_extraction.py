# import the necessary packages

import numpy as np

import skimage.feature as ft
import cv2.cv2 as cv2
import pylab as plt


def plot_key_point(img):
	'''
	Description: A function to plot find and plot the key points in image by using SIFT as a feature extractor.
	
	Parameters
	----------
	img : A three channel source image.

	Returns
	-------
	img_output: image is drawn with keypoints

	'''
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	sift = cv2.xfeatures2d.SIFT_create()
	kp = sift.detect(gray, None)
	
	img_output = cv2.drawKeypoints(gray, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	return img_output


def calculate_self_similarity_matrix(mask_dict, link_info):
	'''
	Description: A function to calculate the similarity matrix between image batches that are attached to each other
	or un-attached.
	
	Parameters
	----------
	mask_dict: Image segments created by superpixel method saved in dictionary.
	link_info: Linking information between image segments created by superpixel method.

	Returns
	-------
	self_feat_matrix: Self similarity matrix of image segments
	link_feat_matrix: Similarity matrix between linked images
	
	'''
	number_keys = len(list(mask_dict.keys()))
	matrix_value = np.array(list(mask_dict.values()))
	self_feat_matrix = np.zeros((len(matrix_value), len(matrix_value)))
	link_feat_matrix = np.zeros((number_keys, number_keys))
	for i in range(0, len(matrix_value)-1):
		for j in range(i+1, len(matrix_value)):
			a_image = matrix_value[i].flatten()
			b_image = matrix_value[j].flatten()
			self_feat_matrix[i][j] = np.linalg.norm(a_image-b_image)
	for li in link_info:
		c_image = mask_dict[li[0]].flatten()
		d_image = mask_dict[li[1]].flatten()
		if li[0] > li[1]:
			link_feat_matrix[li[0]][li[1]] = np.linalg.norm(c_image-d_image)
	plt.imshow(link_feat_matrix, interpolation='nearest')
	plt.show()
	return self_feat_matrix, link_feat_matrix


def line_segment_detector(image_gray):
	'''
	Description: A line segment detector which can detect lines in grayscale image.
	
	Parameters
	----------
	image_gray: gray-scale image

	Returns
	-------
	lsd_gray_output: gray scale image is drawn with line segments
	'''
	# Create line segments from the images
	blank_image_gray = np.ones((image_gray.shape[0], image_gray.shape[1]), dtype=np.uint8)
	lsd = cv2.createLineSegmentDetector(0)
	seg_gray = lsd.detect(image_gray)[0]  # Position 0 of the returned tuple are the detected lines
	lsd_gray_output = lsd.drawSegments(blank_image_gray, seg_gray).astype(np.uint8)
	return lsd_gray_output


def local_binary_pattern(image):
	'''
	Description: A function to extract local binary pattern (LBP) which can be used to compare texture information
	in images.
	
	Parameters
	----------
	image: A three channel RGB source image.

	Returns
	-------
	lbp: Local binary pattern for selected image.
	
	'''
	radius = 3
	n_points = 8 * radius
	
	def hist(ax, lbp):
		n_bins = int(lbp.max() + 1)
		return ax.hist(lbp.ravel(), density=True, bins=n_bins, range=(0, n_bins), facecolor='0.5')
	
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	lbp = ft.local_binary_pattern(image, radius, n_points)
	return lbp
