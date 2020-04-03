import numpy as np
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
import cv2.cv2 as cv2
import logging
from statistics import mean
from PIL import Image
from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans
from copy import deepcopy


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


def color_quantization(path, exe_median_cut=True, plot=False):
	# Read image by cv2
	image = cv2.imread(path)
	image_file_name = os.path.basename(path).split('.')[0]
	
	# Change color space from BGR to RGB to HLS
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
	
	image = np.array(image)  # Change image objects into array
	
	# Mean shift
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

	# The following bandwidth can be automatically detected using
	image_reshape = image.reshape((image.shape[0] * image.shape[1], image.shape[2]))
	bandwidth = estimate_bandwidth(image_reshape, quantile=0.2, n_samples=500)
	
	ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
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
	print("Reduce color through K-means: %d" % n_clusters_)
	
	h_o, s_o, v_o = cv2.split(image)
	h_s, s_s, v_s = cv2.split(image_mean_shift)
	
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
		
		fig = plt.figure()
		ax = fig.add_subplot(1, 1, 1, projection='3d')
		ax.scatter(image_median_cut_h.flatten(), image_median_cut_l.flatten(), image_median_cut_s.flatten())

		ax.set_xlabel("Hue")
		ax.set_ylabel("Saturation")
		ax.set_zlabel("Value")

		plt.show()
		print("Reduce color through median cut: %d" % len(list(set(image_median_cut_label.flatten()))))

	# K-Means
	image_kmeans = image_mean_shift.reshape((image_median_cut.shape[0] * image_median_cut.shape[1], image_median_cut.shape[2]))
	kmeans = KMeans(n_clusters=3, random_state=0).fit(image_kmeans)
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
	
	# fig.savefig(save_path, dpi=100)
	segmentation_image = seperate_layers(k_means_image)
	
	# num_layers = len(segmentation_image)
	num_layers = 0
	current_dir = os.getcwd()
	file_name = path.split('/')[-3]
	image_quantization_result_dir = str(Path(current_dir).parent) + '/image_generator/' + file_name + \
									'/color_quantization_result_batches/' + str(num_layers) + '_layer/'
	
	if not os.path.exists(image_quantization_result_dir):
		os.makedirs(image_quantization_result_dir)
		print("Directory ", image_quantization_result_dir, " Created ")
	else:
		print("Directory ", image_quantization_result_dir, " already exists")
	
	save_path = image_quantization_result_dir + image_file_name + '.jpg'
	
	# Save image into jpg file
	im = Image.fromarray(segmentation_image[num_layers])
	im.save(save_path)
	return segmentation_image


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
	
	if plot:
		image_type = ['Original_image', 'red_legend', 'black_text', 'background']
		fig = plt.figure(figsize=(25, 25))
		
		ax = fig.add_subplot(2, 2, 1)
		ax.imshow(image)
		ax.axis('off')
		ax.set_title('%s layers ' % image_type[0])
		
		for it, z in zip(range(1, len(image_type)), range(len(image_set))):
			ax = fig.add_subplot(2, 2, z+2)
			ax.imshow(image_blank[z])
			ax.axis('off')
			ax.set_title('%s layers ' % image_type[it])
		plt.show()
	return image_blank


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
	image_HSL = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2HLS)
	ax.hist(image_HSL.ravel(), bins=256, color='orange', )
	ax.hist(image_HSL[:, :, 0].ravel(), bins=256, color='red', alpha=0.5)
	ax.hist(image_HSL[:, :, 1].ravel(), bins=256, color='Green', alpha=0.5)
	ax.hist(image_HSL[:, :, 2].ravel(), bins=256, color='Blue', alpha=0.5)
	ax.set_xlabel('Intensity Value')
	ax.set_ylabel('Count')
	ax.legend(['Total', 'Hue_Channel', 'Lightning_Channel', 'Saturation_Channel'])
	
	plt.show()
	

if __name__ == "__main__":
	# path = '/home/yizi/Documents/phd/historical_map_project/image_generator/_01_01/image_batches/'
	# file_names = os.listdir(path)
	# for file in file_names:
	# 	file_path = path + file
	# 	quant_image = color_quantization(file_path)
	
	path = '/home/yizi/Documents/phd/historical_map_project/test/BHdV_PL_ATL20Ardt_1929_0003.jpg'
	histogram_color_space(path)
