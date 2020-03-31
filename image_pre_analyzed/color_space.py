import numpy as np

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


def color_quantization(image):
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
	
	image_mean_shift = recreate_image(cluster_centers, labels, image.shape[0], image.shape[1])
	print("number of estimated clusters : %d" % n_clusters_)
	
	h_o, s_o, v_o = cv2.split(image)
	h_s, s_s, v_s = cv2.split(image_mean_shift)
	
	fig = plt.figure(figsize=plt.figaspect(0.5))
	ax = fig.add_subplot(1, 2, 1, projection='3d')
	ax.scatter(h_o.flatten(), s_o.flatten(), v_o.flatten(), c=label_norm)
	ax.set_xlabel("Hue")
	ax.set_ylabel("Saturation")
	ax.set_zlabel("Value")
	
	ax = fig.add_subplot(1, 2, 2, projection='3d')
	ax.scatter(h_s.flatten(), s_s.flatten(), v_s.flatten(), c=label_norm)
	ax.set_xlabel("Hue")
	ax.set_ylabel("Saturation")
	ax.set_zlabel("Value")
	plt.show()
	print()
	
	# Median cut
	image_mean_shift_imojbect = Image.fromarray(image_mean_shift.astype(np.uint8))
	image_median_cut = median_cut(image_mean_shift_imojbect, 3)
	image_median_cut_h = np.array([imc[0] for imc in image_median_cut])
	image_median_cut_s = np.array([imc[1] for imc in image_median_cut])
	image_median_cut_v = np.array([imc[2] for imc in image_median_cut])
	
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1, projection='3d')
	ax.scatter(image_median_cut_h.flatten(), image_median_cut_s.flatten(), image_median_cut_v.flatten())
	
	ax.set_xlabel("Hue")
	ax.set_ylabel("Saturation")
	ax.set_zlabel("Value")
	
	plt.show()

	# K-Means
	image_kmeans = image_mean_shift.reshape((image_mean_shift.shape[0] * image_mean_shift.shape[1], image_mean_shift.shape[2]))
	kmeans = KMeans(n_clusters=3, random_state=0).fit(image_kmeans)
	labels = kmeans.predict(image_kmeans)
	
	cluster_centers = kmeans.cluster_centers_.astype(np.uint8)
	image_mean_shift = image_mean_shift.astype(np.uint8)
	k_means_image = recreate_image(cluster_centers, labels, image_mean_shift.shape[0], image_mean_shift.shape[1])
	h_km, s_km, v_km = cv2.split(k_means_image)
	
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1, projection='3d')
	
	ax.scatter(h_km.flatten(), s_km.flatten(), v_km.flatten(), c=labels)
	
	ax.set_xlabel("Hue")
	ax.set_ylabel("Saturation")
	ax.set_zlabel("Value")
	
	fig = plt.figure()
	plt.imshow((k_means_image*255).astype(np.uint8))
	plt.show()
	print()
	return k_means_image


def remove_element(image):
	image = image.astype(np.uint8)
	image_set = list(set([tuple(j) for i in image for j in i]))  # find the group of color
	image_blank = np.zeros((len(image_set), image.shape[0], image.shape[1], image.shape[2])).astype(np.uint8)
	# Seperate different layers
	for z in range(len(image_set)):
		for i in range(image.shape[0]):
			for j in range(image.shape[1]):
				if list(image[i][j]) == list(image_set[z]):
					image_blank[z][i][j] = np.array(image_set[0])
	
	fig = plt.figure()
	for z in range(len(image_set)):
		ax = fig.add_subplot(1, len(image_set), z+1)
		ax.imshow((image_blank[z]*255).astype(np.uint8))
	plt.show()
	return image_blank


if __name__ == "__main__":
	image = cv2.imread('/home/yizi/Documents/phd/map/service-du-plan-Paris-1929/Service-du-Plan-Paris-1929/BHdV_PL_ATL20Ardt_1929_0003.jpg')
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	fig = plt.figure()
	plt.imshow(np.array(image))
	plt.show()
	image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
	
	image = image[3000: 4000, 3000: 4000]  # Crop the image
	image = np.array(image)  # Change image objects into array
	
	quant_image = color_quantization(image)
	image_blank = remove_element(quant_image)
	
	fig = plt.figure()
	plt.imshow(np.array(image_blank[0]))
	plt.show()
