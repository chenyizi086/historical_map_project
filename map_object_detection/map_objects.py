import numpy as np

import cv2.cv2 as cv2
import matplotlib.pyplot as plt

from color_space import color_quantization


# Detect longitude and latitude
def detect_longitude_latitude(gray_image):
	print('Start detecting longitude and latitude')
	# Detecting horizontal and vertical lines in the images.
	blank_image_gray = np.zeros((gray_image.shape[0], gray_image.shape[1]), dtype=np.uint8)
	output_image_gray = np.zeros((gray_image.shape[0], gray_image.shape[1]), dtype=np.uint8)
	
	# Hough transform
	lines = cv2.HoughLines(gray_image, 1, np.pi / 180, 200)
	for l in lines:
		rho, theta = l[0]
		# Set threshold based on the angle theta
		if np.abs((np.pi / 2) - theta) < 0.02 or np.abs(0 - theta) < 0.02:
			a = np.cos(theta)
			b = np.sin(theta)
			x0 = a * rho
			y0 = b * rho
			x1 = int(x0 + 1000 * (-b))
			y1 = int(y0 + 1000 * (a))
			x2 = int(x0 - 1000 * (-b))
			y2 = int(y0 - 1000 * (a))
			
			img_line = cv2.line(blank_image_gray, (x1, y1), (x2, y2), 1, 2)
			img_overlap = np.logical_and(img_line, gray_image).astype(np.uint8)
			
			# Calculate overlapped pixel percentage
			number_original_pixels = list(img_line.flatten()).count(1)
			number_overlapped_pixels = list(img_overlap.flatten()).count(1)
			percentage = number_overlapped_pixels / number_original_pixels
		
			if percentage > 0.7:
				cv2.line(output_image_gray, (x1, y1), (x2, y2), 1, 2)
			else:
				# Reset image blank_image_gray and overlapped image
				blank_image_gray = np.zeros((gray_image.shape[0], gray_image.shape[1]), dtype=np.uint8)
				img_line = np.zeros((gray_image.shape[0], gray_image.shape[1]), dtype=np.uint8)
	
	fig = plt.figure(figsize=(25, 25))
	
	ax = fig.add_subplot(1, 2, 1)
	ax.imshow(gray_image)
	ax.axis('off')
	ax.set_title('Orignal gray-scale image')
	
	ax = fig.add_subplot(1, 2, 2)
	ax.imshow(output_image_gray)
	ax.axis('off')
	ax.set_title('Extracted longitude and latitude')

	plt.imshow(output_image_gray)
	plt.show()
	fig.savefig('longitude_latitude.png', dpi=100)
	print('Save figure')
	return output_image_gray


if __name__ == '__main__':
	image = cv2.imread('/home/yizi/Documents/phd/map/Service-du-Plan-Paris-1929/BHdV_PL_ATL20Ardt_1929_0003.jpg')
	
	# Change color space from BGR to RGB to HLS
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
	
	image = image[3500: 4000, 3500: 4000]  # Crop the image
	image = np.array(image)  # Change image objects into array
	quant_image = color_quantization(image)
	RGB_image = cv2.cvtColor(quant_image[1], cv2.COLOR_HLS2BGR)
	gray_scale_image = cv2.cvtColor(RGB_image, cv2.COLOR_BGR2GRAY)
	
	gray_image = detect_longitude_latitude(gray_scale_image)
