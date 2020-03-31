import matplotlib.pyplot as plt
import matplotlib
from matplotlib import colors

from feature_extraction import line_segment_detector
from PIL import Image
import numpy as np
import cv2.cv2 as cv2

from super_pixel import color_quantization
from color_space import visualize_color_space

if __name__ == "__main__":
	# image, superpixel_segments = image_to_superpixel('/home/yizi/Documents/phd/map/Jacoubet/ftp3.ign.fr/Data/BHVP/Jacoubet/Atlas_de_Jacoubet_-_03._Partie_de_la_commune_de_Monceau_et_de_ses_environs_-_BHVP.jpg')
	image = cv2.imread('/home/yizi/Documents/phd/map/service-du-plan-Paris-1929/Service-du-Plan-Paris-1929/BHdV_PL_ATL20Ardt_1929_0003.jpg')
	image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
	
	image = image[3000: 4000, 3000: 4000]  # Crop the image
	image = np.array(image)  # Change image objects into array
	
	plt.imshow(image[:, :, 0], cmap='gray')
	plt.show()
	
	visualize_color_space(image)
	
	# Remove third channel
	image[:, :, 1] = np.zeros((image.shape[0], image.shape[1]))
	image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
	# image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	
	visualize_color_space(image)
	
	image = color_quantization(image, 2, mode='KMeans')  # Quantized colors
	# image = image.astype(np.uint8)  # Change color into uint8 format
	#
	# image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	# plt.imshow(image, cmap='gray')
	# plt.show()
	# # visualize_color_space(image)
	#
	# image_set = set([tuple(j) for i in image for j in i])  # find the group of color
	#
	# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	#
	# image_set = set(image.flatten())
	# image = np.where(image == list(image_set)[0], image, list(image_set)[2])
	#
	# plt.imshow(image, cmap='gray')
	# plt.show()
	#
	# image_flatten = image.flatten()
	# gray_image_class = list(set(image_flatten))
	# image = np.where(image == gray_image_class[0], 0, image)
	# image = np.where(image == gray_image_class[1], 255, image)
	#
	# plt.imshow(image)
	# plt.show()
	# # plt.imsave('Jacoubet_quantize.png', image, cmap='gray')
	#
	# fig = plt.figure(figsize=(24, 24))
	# # s_gray, s_rgb = plot_boundary_with_superpixel(superpixel_segments)
	# drawn_img = line_segment_detector(image)
	# # merge_1 = image_blender(image, s_rgb, 0.7, 0.3)
