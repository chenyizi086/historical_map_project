import numpy as np
import os
from pathlib import Path
import operator
import pickle
import numba as nb

import matplotlib.pyplot as plt
from matplotlib import colors
import cv2.cv2 as cv2
import logging
from statistics import mean
from PIL import Image
from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans
from copy import deepcopy

from color_picker_tool.color_picker import color_picker


class ColorCube(object):
	def __init__(self, colors):
		self.colors = colors or []
		self.red = [r[0] for r in colors]
		self.green = [g[1] for g in colors]
		self.blue = [b[2] for b in colors]
		self.size = (max(self.red) - min(self.red),
					 max(self.green) - min(self.green),
					 max(self.blue) - min(self.blue))
		self.max_range = max(self.size)
		self.max_channel = self.size.index(self.max_range)
	
	def average(self):
		logging.info('Averaging cube with {} colors'.format(len(self.colors)))
		r = int(mean(self.red))
		g = int(mean(self.green))
		b = int(mean(self.blue))
		return r, g, b
	
	def split(self):
		middle = len(self.colors) // 2
		colors = sorted(self.colors, key=lambda c: c[self.max_channel])
		return ColorCube(colors[:middle]), ColorCube(colors[middle:])
	
	def __lt__(self, other):
		return self.max_range < other.max_range


def median_cut(img, num_colors, unique=False):
	# If unique is true then multiple instances of a single RGB value will only be counted once
	# and the rest discarded. This is MUCH faster and creates a more diverse pallete, but is not
	# a "true" median cut.
	# For example if an image had 99 blue pixels (0,0,255) and a single red pixel (255,0,0) the
	# respective median cuts would be
	# unique = False: [(0,0,255),(5,0,249)]
	# unique = True:  [(0,0,255),(255,0,0)]
	# False would produce 2 almost identical shades of blue, True would result in pure blue/red
	colors = []
	logging.info('Creating list of colors')
	for color_count, color in img.getcolors(img.width * img.height):
		if unique:
			colors += [color]
		else:
			colors += [color] * color_count
	logging.info('Created list of {} colors'.format(len(colors)))
	logging.info('Creating ColorCube')
	cubes = [ColorCube(colors)]
	logging.info('ColorCube created')
	
	while len(cubes) < num_colors:
		logging.info('Performing split {}/{}'.format(len(cubes), num_colors - 1))
		cubes.sort()
		cubes += cubes.pop().split()
	
	return [c.average() for c in cubes]


# Recreate image
def recreate_image(codebook, labels, w, h):
	"""Recreate the (compressed) image from the code book & labels"""
	d = codebook.shape[1]
	image = np.zeros((w, h, d))
	label_idx = 0
	for i in range(w):
		for j in range(h):
			image[i][j] = codebook[labels[label_idx]]
			label_idx += 1
	return image


def color_quantization(path, exe_median_cut=True, plot=True):
	# Read image by cv2
	image = cv2.imread(path)
	image_file_name = os.path.basename(path).split('.')[0]
	
	# Change color space from BGR to RGB to HLS
	image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
	image = np.array(image)  # Change image objects into array
	
	# The following bandwidth can be automatically detected using
	image_reshape = image.reshape((image.shape[0] * image.shape[1], image.shape[2]))
	bandwidth = estimate_bandwidth(image_reshape, quantile=0.2, n_samples=500)
	
	ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=10)
	ms.fit(image_reshape)
	labels = ms.predict(image_reshape)
	
	# Normalized color
	label_norm = deepcopy(labels)
	norm = colors.Normalize(vmin=-1., vmax=1.)
	norm.autoscale(label_norm)
	label_norm = norm(label_norm).tolist()

	cluster_centers = ms.cluster_centers_.astype(np.uint8)
	
	labels_unique = np.unique(labels)
	n_clusters_ = len(labels_unique)
	
	image_mean_shift = recreate_image(cluster_centers, labels, image.shape[0], image.shape[1]).astype(np.uint8)
	print("Reduce color through mean-shift: %d" % n_clusters_)
	
	h_o, s_o, v_o = cv2.split(image)
	
	# Normalize color space
	pixel_colors = image.reshape((np.shape(image)[0] * np.shape(image)[1], 3))
	norm = colors.Normalize(vmin=-1., vmax=1.)
	norm.autoscale(pixel_colors)
	pixel_colors = norm(pixel_colors).tolist()
	
	if plot:
		fig = plt.figure(figsize=(25, 25))
		
		ax = fig.add_subplot(2, 3, 1)
		ax.imshow(image)
		ax.axis('off')
		ax.set_title('Original image')
		
		ax = fig.add_subplot(2, 3, 2)
		ax.imshow(image_mean_shift)
		ax.axis('off')
		ax.set_title('Image after mean shift')
		
		ax = fig.add_subplot(2, 3, 4, projection="3d")
		ax.scatter(h_o.flatten(), s_o.flatten(), v_o.flatten(), facecolors=pixel_colors, marker=".")
		ax.set_xlabel("Hue")
		ax.set_ylabel("Saturation")
		ax.set_zlabel("Value")
		ax.set_title('Color space of original images')
		
		ax = fig.add_subplot(2, 3, 5, projection='3d')
		ax.scatter(h_o.flatten(), s_o.flatten(), v_o.flatten(), c=label_norm)
		ax.set_xlabel("Hue")
		ax.set_ylabel("Saturation")
		ax.set_zlabel("Value")
		ax.set_title('Color space after mean-shift')
	# Median cut
	# There is no need to do median cut if there are not too many colors
	if exe_median_cut:
		image_array = Image.fromarray(image_mean_shift.astype(np.uint8))
		image_median_cut_label = np.array(image_array.quantize(colors=10, method=0, kmeans=0, palette=None))
		
		# Calculate the clustered centered
		pixel_class = {index: [] for index in list(set(image_median_cut_label.flatten()))}
		image_median_cut = image_median_cut_label.flatten()
		image_mean_shift_shape = image_mean_shift.reshape(image_mean_shift.shape[0]*image_mean_shift.shape[1],
														  image_mean_shift.shape[2])
		
		for i, j in zip(image_median_cut, image_mean_shift_shape):
			pixel_class[i].append(j)
		
		clustered_center_median_cut = np.array(list({key: tuple(np.average(np.array(value), axis=0).astype(np.uint8))
													 for key, value in pixel_class.items()}.values()))
		
		image_median_cut = recreate_image(clustered_center_median_cut, image_median_cut_label.flatten(),
										  image_mean_shift.shape[0], image_mean_shift.shape[1]).astype(np.uint8)
		
		image_median_cut_h, image_median_cut_l, image_median_cut_s = cv2.split(image_median_cut)

		print("Reduce color through median cut: %d" % len(list(set(image_median_cut_label.flatten()))))
		image_mean_shift = image_median_cut
		
	# K-Means
	image_kmeans = image_mean_shift.reshape((image_mean_shift.shape[0] * image_mean_shift.shape[1], image_mean_shift.shape[2]))
	kmeans = KMeans(n_clusters=3, random_state=0, n_jobs=10).fit(image_kmeans)
	labels = kmeans.predict(image_kmeans)
	print("Reduce color through K-means: %d" % len(list(set(labels.flatten()))))
	
	cluster_centers = kmeans.cluster_centers_.astype(np.uint8)
	image_mean_shift = image_mean_shift.astype(np.uint8)
	k_means_image = recreate_image(cluster_centers, labels, image_mean_shift.shape[0], image_mean_shift.shape[1]).astype(np.uint8)
	h_km, s_km, v_km = cv2.split(k_means_image)

	if plot:
		ax = fig.add_subplot(2, 3, 3)
		ax.imshow(k_means_image)
		ax.axis('off')
		ax.set_title('Image after K-means')

		ax = fig.add_subplot(2, 3, 6, projection='3d')
		ax.scatter(h_km.flatten(), s_km.flatten(), v_km.flatten(), c=labels)
		ax.set_xlabel("Hue")
		ax.set_ylabel("Saturation")
		ax.set_zlabel("Value")

		fig.tight_layout()
		plt.show()
	
	segmentation_image = seperate_layers(k_means_image)
	
	for nl in range(1, len(segmentation_image)):
		current_dir = os.getcwd()
		file_name = path.split('/')[-3]
		image_quantization_result_dir = str(Path(current_dir).parent) + '/image_generator/' + file_name + \
										'/color_quantization_result_batches/' + str(nl) + '_layer/'
		
		if not os.path.exists(image_quantization_result_dir):
			os.makedirs(image_quantization_result_dir)
			print("Directory ", image_quantization_result_dir, " Created ")
		else:
			print("Directory ", image_quantization_result_dir, " already exists")
		
		save_path = image_quantization_result_dir + image_file_name + '.p'
		
		# Save image into pickle file for saving memeory
		with open(save_path, 'wb') as handle:
			pickle.dump(segmentation_image[list(segmentation_image.keys())[nl]], handle, protocol=pickle.HIGHEST_PROTOCOL)
	return segmentation_image


def color_quantization_click_and_select(path):
	# Recreate image
	def recreate_image(codebook, labels, w, h):
		"""Recreate the (compressed) image from the code book & labels"""
		d = codebook.shape[1]
		image = np.zeros((w, h, d))
		label_idx = 0
		for i in range(w):
			for j in range(h):
				image[i][j] = codebook[labels[label_idx]]
				label_idx += 1
		return image
		
	# Read image by cv2
	image = cv2.imread(path)
	image_file_name = os.path.basename(path).split('.')[0]
	
	# Change color space from BGR to RGB to HLS
	image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
	image = np.array(image)  # Change image objects into array
	
	# The following bandwidth can be automatically detected using
	image_reshape = image.reshape((image.shape[0] * image.shape[1], image.shape[2]))
	bandwidth = estimate_bandwidth(image_reshape, quantile=0.2, n_samples=500)
	
	ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=10)
	ms.fit(image_reshape)
	labels = ms.predict(image_reshape)
	cluster_centers = ms.cluster_centers_.astype(np.uint8)
	
	labels_unique = np.unique(labels)
	n_clusters_ = len(labels_unique)
	
	image = recreate_image(cluster_centers, labels, image.shape[0], image.shape[1]).astype(np.uint8)
	print("Reduce color through mean-shift: %d" % n_clusters_)
	image_reshape = image.reshape(image.shape[0]*image.shape[1], image.shape[2])
	# 1. Pick color
	select_color = color_picker(image)
	print(select_color)
	
	@nb.njit
	def euc(a, b):
		return ((b - a) ** 2).sum(axis=0) ** 0.5
	
	segmentation_image = {key: None for key in list(select_color.keys())}
	for key, value in select_color.items():
		image_copy = image.copy().astype(np.uint8)
		if list(value) == list(np.array([0, 0, 0])):
			segmentation_image[key] = np.zeros((image.shape[0], image.shape[1], image.shape[2]))
		
		# Define the color you're looking for
		pattern = np.array(value).astype(np.uint8)
		
		# Make a mask to use with where
		mask = (image_copy == pattern).all(axis=2)
		newshape = mask.shape + (1,)
		mask = mask.reshape(newshape)
		image_copy = np.where(mask, [255, 255, 255], [0, 0, 0])
		segmentation_image[key] = image_copy.astype(np.uint8)
		
	fig = plt.figure(figsize=(25, 25))
	for index, (layer, seg_image) in enumerate(segmentation_image.items()):
		ax = fig.add_subplot(2, 2, index + 1)
		ax.imshow(seg_image)
		ax.axis('off')
		ax.set_title(' %s _layer' % layer)
	plt.show()
	
	# label = []
	# ecu_dist = []
	# for i in image_reshape:
	# 	for s in selected_centroid:
	# 		ecu_dist.append(euc(i, s))
	# 	label.append(np.argmin(ecu_dist))
	# 	ecu_dist = []
	#
	# image = image.astype(np.uint8)
	# recreate_image = recreate_image(selected_centroid, label, image.shape[0], image.shape[1]).astype(np.uint8)
	#
	# plt.imshow(recreate_image)
	# plt.show()
	
	# segmentation_image = seperate_layers(recreate_image)
	
	for nl in range(len(segmentation_image)):
		current_dir = os.getcwd()
		file_name = path.split('/')[-3]
		image_quantization_result_dir = str(Path(current_dir).parent) + '/image_generator/' + file_name + \
										'/color_quantization_result_batches/' + str(nl) + '_layer/'

		if not os.path.exists(image_quantization_result_dir):
			os.makedirs(image_quantization_result_dir)
			print("Directory ", image_quantization_result_dir, " Created ")
		else:
			print("Directory ", image_quantization_result_dir, " already exists")

		save_path = image_quantization_result_dir + image_file_name + '.p'

		# Save image into pickle file for saving memeory
		with open(save_path, 'wb') as handle:
			pickle.dump(segmentation_image[list(segmentation_image.keys())[nl]],
						handle, protocol=pickle.HIGHEST_PROTOCOL)


def seperate_layers(image, plot=True):
	image = image.astype(np.uint8)
	image_set = list(set([tuple(j) for i in image for j in i]))  # find the group of color
	image_blank = np.zeros((len(image_set), image.shape[0], image.shape[1], image.shape[2])).astype(np.uint8)
	# Seperate different layers
	for z in range(len(image_set)):
		for i in range(image.shape[0]):
			for j in range(image.shape[1]):
				if list(image[i][j]) == list(image_set[z]):
					image_blank[z][i][j] = np.array([255, 255, 255])
	
	# Classification layers through
	image_type = {'Original_image': None, 'red_legend': None, 'black_text': None, 'background': None}
	number_zeros_pixels = []
	for index, img in enumerate(image_blank):
		number_zeros_pixels.append((index, list(img.flatten()).count(0)))
	
	number_zeros_pixels.sort(key=operator.itemgetter(1))
	
	# Image layer with maximum zeroes should be classified as red-legend or edge pixels
	# Image layer with second maximum zeroes should be classified as text&lines
	# Image layer with minimum zeroes should be classified as background
	image_type['Original_image'] = image
	image_type['background'] = image_blank[number_zeros_pixels[0][0]]
	image_type['black_text'] = image_blank[number_zeros_pixels[1][0]]
	# image_type['edge'] = image_blank[number_zeros_pixels[2][0]]
	image_type['red_legend'] = image_blank[number_zeros_pixels[2][0]]
	
	if plot:
		fig = plt.figure(figsize=(25, 25))
		
		for index, (type, image) in enumerate(image_type.items()):
			ax = fig.add_subplot(3, 2, index+1)
			ax.imshow(image)
			ax.axis('off')
			ax.set_title('%s_layers ' % type)
		plt.show()
	return image_type


# histogram analysis of color space
def histogram_color_space(path):
	image_BGR = np.array(cv2.imread(path))
	image_RGB = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)
	
	fig = plt.figure()
	
	ax = fig.add_subplot(2, 1, 1)
	ax.hist(image_RGB.ravel(), bins=256, color='orange', )
	ax.hist(image_RGB[:, :, 0].ravel(), bins=256, color='red', alpha=0.5)
	ax.hist(image_RGB[:, :, 1].ravel(), bins=256, color='Green', alpha=0.5)
	ax.hist(image_RGB[:, :, 2].ravel(), bins=256, color='Blue', alpha=0.5)
	ax.set_xlabel('Intensity Value')
	ax.set_ylabel('Count')
	ax.legend(['Total', 'Red_Channel', 'Green_Channel', 'Blue_Channel'])
	
	ax = fig.add_subplot(2, 1, 2)
	image_HSL = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2HSV)
	ax.hist(image_HSL.ravel(), bins=256, color='orange', )
	ax.hist(image_HSL[:, :, 0].ravel(), bins=256, color='red', alpha=0.5)
	ax.hist(image_HSL[:, :, 1].ravel(), bins=256, color='Green', alpha=0.5)
	ax.hist(image_HSL[:, :, 2].ravel(), bins=256, color='Blue', alpha=0.5)
	ax.set_xlabel('Intensity Value')
	ax.set_ylabel('Count')
	ax.legend(['Total', 'Hue_Channel', 'Lightning_Channel', 'Saturation_Channel'])
	
	plt.show()
