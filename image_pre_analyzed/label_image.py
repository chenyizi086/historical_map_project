import cv2.cv2 as cv2
import matplotlib.pyplot as plt

from super_pixel import rgb2gray


def fill_contours_callback(contours_image):
	'''
	Description: The user can click the contour in order to fill the contour in the image.
	
	Parameters
	----------
	contours_image: image with contour information

	Returns
	-------
	contours
	

	'''
	# drawing = False # true if mouse is pressed
	# mode = True # if True, draw rectangle. Press 'm' to toggle to curve
	# ix,iy = -1,-1
	fill_image = rgb2gray(contours_image).astype('uint8')
	
	def click_event(event, x, y, flags, param):
		if event == cv2.EVENT_LBUTTONDOWN:
			# Add text
			# font = cv2.FONT_HERSHEY_SIMPLEX
			# str = np.str(x)
			# cv2.putText(contours_image, str, (x, y),font ,1, (255, 255, 0), 2)
			seedpoint = x, y
			floodfill_color = 255, 255, 0
			# Get the value in the seedpoint
			# seed_point_value = contours_image[x][y]
			cv2.floodFill(contours_image, None, seedpoint, floodfill_color)
			cv2.imshow('original', contours_image)
	
	cv2.namedWindow('original')
	cv2.imshow('original', fill_image)
	cv2.setMouseCallback("original", click_event)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
	# Plot labeled result
	plt.figure()
	plt.imshow(fill_image)
	plt.show()
	return fill_image
