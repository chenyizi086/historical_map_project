import matplotlib.pyplot as plt
import matplotlib

from feature_extraction import line_segment_detector
from PIL import Image
import numpy as np
import cv2.cv2 as cv2

from super_pixel import color_quantization

if __name__ == "__main__":
	# image, superpixel_segments = image_to_superpixel('/home/yizi/Documents/phd/map/Jacoubet/ftp3.ign.fr/Data/BHVP/Jacoubet/Atlas_de_Jacoubet_-_03._Partie_de_la_commune_de_Monceau_et_de_ses_environs_-_BHVP.jpg')
	image = Image.open('/home/yizi/Documents/phd/map/Jacoubet/ftp3.ign.fr/Data/BHVP/Jacoubet/Atlas_de_Jacoubet_-_03._Partie_de_la_commune_de_Monceau_et_de_ses_environs_-_BHVP.jpg')
	# image = image.crop((0, 0, 1000, 1000))
	image = np.array(image)
	
	image = color_quantization(image, 2, mode='KMeans')
	image = image.astype(np.uint8)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	image_flatten = image.flatten()
	gray_image_class = list(set(image_flatten))
	image = np.where(image == gray_image_class[0], 0, image)
	image = np.where(image == gray_image_class[1], 255, image)
	
	plt.imshow(image)
	plt.show()
	# plt.imsave('Jacoubet_quantize.png', image, cmap='gray')

	fig = plt.figure(figsize=(24, 24))
	# s_gray, s_rgb = plot_boundary_with_superpixel(superpixel_segments)
	drawn_img = line_segment_detector(image)
	# merge_1 = image_blender(image, s_rgb, 0.7, 0.3)
