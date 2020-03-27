# import the necessary packages
import numpy as np

import matplotlib.pyplot as plt
from skimage.segmentation import slic, find_boundaries
from skimage.color import rgb2gray
from sklearn.cluster import KMeans, MeanShift
from PIL import Image, ImageCms
import cv2.cv2 as cv2


def image_to_superpixel(path, color_quant=False, color_format='RGB', method='SLIC'):
	'''
	Description: Change image to superpixel matrix
	
	Parameters
	----------
	path: The relative path of three channel source image.
	color_quant: True->color quantization; False->none color quantization
	color_format: 'RGB' ('default'); CIELAB
	method: superpixel method

	Returns
	-------
	image: A three channel image
	superpixel_image: Same value in the matrix correspond to a superpixel element.
	
	'''
	# load the image and convert it to a floating point data type
	if color_format == 'CIELAB':
		image = rgb_to_lab(path)
		image = np.array(image)
	elif color_format == 'RGB':
		image = Image.open(path)
		# Crop image to shrink the size
		image = np.array(image)
		plt.imshow(image)
		plt.show()
	else:
		raise NameError('Please enter appropriate color format.')
	
	if color_quant:
		fig = plt.figure(figsize=(24, 24))
		index = 2
		for num in range(9):
			image_quan = color_quantization(image, index)
			fig.add_subplot(3, 3, num+1)
			plt.title(index)
			plt.axis('off')
			plt.imshow(image_quan)
			index += 1
		image = color_quantization(image, 3)
		plt.show()
	else:
		pass
	
	# TODO: Add more superpixel methods
	if method == 'SLIC':
		superpixel_segments = slic(image, n_segments=200, sigma=5).astype(np.uint8)
		plt.imshow(superpixel_segments)
		plt.show()
	else:
		raise NameError('Please enter image segmentation method.')
	return image, superpixel_segments


def rgb_to_lab(path):
	'''
	Description: Change color space to 'RGB' to 'CIELAB'
	
	Parameters
	----------
	path: The relative path of three channel source image.

	Returns
	-------
	lab_image: image in 'CIELAB' color space

	'''
	# Create input and output colour profiles.
	rgb_profile = ImageCms.createProfile(colorSpace='sRGB')
	lab_profile = ImageCms.createProfile(colorSpace='LAB')
	
	# Create a transform object from the input and output profiles
	rgb_to_lab_transform = ImageCms.buildTransform(
		inputProfile=rgb_profile,
		outputProfile=lab_profile,
		inMode='RGB',
		outMode='LAB'
	)
	
	# Open the source image
	rgb_image = Image.open(path)
	
	# Create a new image by applying the transform object to the source image
	lab_image = ImageCms.applyTransform(
		im=rgb_image,
		transform=rgb_to_lab_transform
	)
	return lab_image

	
def color_quantization(image, n_c, mode='KMeans'):
	'''
	Description: Quantized image color by clustering similar pixels
	
	Parameters
	----------
	image: Three channel source image.
	n_c: Number of clusters used in clustering algorithm
	mode: Color quantization methods ('Kmeans' )

	Returns
	-------
	quantized_image: quantized image
	
	'''
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

	image_shape = image.shape
	image_array = np.reshape(image, (image_shape[0] * image_shape[1], image_shape[2]))
	if mode == 'MS':
		meanshift = MeanShift(bandwidth=2).fit(image_array)
		labels = meanshift.predict(image_array)
		quantized_image = recreate_image(meanshift.cluster_centers_, labels, image_shape[0], image_shape[1])
	elif mode == 'KMeans':
		kmeans = KMeans(n_clusters=n_c, random_state=0).fit(image_array)
		labels = kmeans.predict(image_array)
		quantized_image = recreate_image(kmeans.cluster_centers_, labels, image_shape[0], image_shape[1])
	else:
		pass
	return quantized_image


def boundary_of_superpixel(superpixel_segments):
	'''
	Description: Create boundary of superpixels
	
	Parameters
	----------
	superpixel_segments: Superpixels matrix generated by superpixel algorithm

	Returns
	-------
	segments_boundary_image_gray: Boundary image in gray-scale
	segments_boundary_image_rgb: Boundary image in 'RGB'
	
	'''
	# Find boundary of the superpixels segments
	segments_boundary = np.invert(find_boundaries(superpixel_segments))
	segments_boundary_image_gray = (segments_boundary * 255).astype(np.uint8)
	segments_boundary_image_rgb = cv2.cvtColor(segments_boundary_image_gray, cv2.COLOR_GRAY2RGB)
	plt.imshow(segments_boundary_image_gray)
	plt.show()
	return segments_boundary_image_gray, segments_boundary_image_rgb


def image_blender(img_1, img_2, weight_1, weight_2):
	'''
	
	Parameters
	----------
	img_1: first RGB three channel image
	img_2: Second RGB three channel image
	weight_1: Alpha value of first image
	weight_2: Alpha value of second image

	Returns
	-------
	blend_image: blend image result (three channel)

	'''
	# Blender two images together with weight number
	blend_image = cv2.addWeighted(img_1, weight_1, img_2, weight_2, 0)
	plt.imshow(blend_image, vmin=0, vmax=255)
	plt.show()
	return blend_image


def create_region_linking(superpixel_segments):
	'''
	Description: To know the neighbour between superpixel segments
	
	Parameters
	----------
	superpixel_segments: Superpixels matrix generated by superpixel algorithm

	Returns
	-------
	link_info: the neighbouring or linking information between superpixel segments
	
	'''
	# Creating the region linking information
	# Vertical + Horizontal
	link_info = []
	for l_i in superpixel_segments:
		tmp = l_i[0]
		for i in range(1, len(l_i)):
			if tmp != l_i[i]:
				link_info.append(tuple(set([tmp, l_i[i]])))
				tmp = l_i[i]
	segments_copy = np.swapaxes(superpixel_segments, 0, 1)
	for l_j in segments_copy:
		tmp = l_j[0]
		for i in range(1, len(l_j)):
			if tmp != l_j[i]:
				link_info.append(tuple(set([tmp, l_j[i]])))
				tmp = l_j[i]
	link_info = sorted(list(set(link_info)))
	return link_info


def seperate_superpixels_into_layers(superpixel_segments):
	'''
	Description: Seperate each superpixel element into layers of binary mask
	
	Parameters
	----------
	superpixel_segments: Superpixels matrix generated by superpixel algorithm

	Returns
	-------
	layers: Superpixel element to binary mask
	layers_dict: Superpixel element save in dictionary
	
	'''
	# Seperating the superpixel image into different binary layers
	set_of_classes = set(superpixel_segments.flatten())
	first_class = list(set_of_classes)[0]
	layers = np.array([])
	layers_dict = {i: None for i in set_of_classes}
	for s in set_of_classes:
		if s == first_class:
			layers = superpixel_segments == first_class
			layers = layers.astype(np.int)
			layers_dict[s] = layers
		else:
			temp = superpixel_segments == s
			temp = temp.astype(np.int)
			layers = np.vstack((layers, temp))
			layers_dict[s] = temp
	layers = layers.reshape(len(set_of_classes), superpixel_segments.shape[0], superpixel_segments.shape[1])
	return layers, layers_dict


def find_contours(image, segments_boundary):
	'''
	
	Parameters
	----------
	image: rgb image
	segments_boundary: boundaries of superpixel matrix

	Returns
	-------
	countour_image: image is drawn with contour
	
	'''
	# Find contours
	ret, thresh = cv2.threshold(segments_boundary, 10, 255, cv2.THRESH_BINARY)
	_, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	contour_image = np.zeros(image.shape)
	cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 3)
	contour_image = rgb2gray(contour_image).astype('uint8')
	return contour_image
