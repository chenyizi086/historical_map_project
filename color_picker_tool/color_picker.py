import cv2
import numpy as np

index = 0


def color_picker(img):
	# This will display all the available mouse click events
	events = [i for i in dir(cv2) if 'EVENT' in i]
	print(events)
	
	# Pick three different layers from maps
	# Layers include: Background, text, special_legend
	# Global variable -> select color
	select_color = {'background': np.array([0, 0, 0]),
					'red_legend': np.array([0, 0, 0]),
					'black_text': np.array([0, 0, 0])}
	global index
	
	# click event function
	def click_event(event, x, y, flags, param):
		global index
		if event == cv2.EVENT_LBUTTONDOWN and (flags & cv2.EVENT_FLAG_CTRLKEY):
			if index < len(select_color):
				print('There is no %s' % str(list(select_color.keys())[index]))
				cv2.imshow("image", img)
				index += 1
				if index == len(select_color):
					print('Job done, please press 0')
				else:
					print('Please select color of %s layer' % (str(list(select_color.keys())[index])))
			else:
				print('Job done, please press 0')
		if event == cv2.EVENT_LBUTTONDBLCLK:
			if index < len(select_color):
				red = img[y, x, 2]
				green = img[y, x, 1]
				blue = img[y, x, 0]
				font = cv2.FONT_HERSHEY_SIMPLEX
				strBGR = str(red) + ", " + str(green) + "," + str(blue)
				# cv2.putText(img, strBGR, (x, y), font, 0.5, (255, 255, 255), 2)
				cv2.imshow("image", img)
				select_color[list(select_color.keys())[index]] = (red, green, blue)
				print('Assign color %s to %s layer' % (str(select_color[list(select_color.keys())[index]]), str(list(select_color.keys())[index])))
				index += 1
				print(index)
				if index == len(select_color):
					print('Job done, please press 0')
				else:
					print('Please select color of %s layer' % (str(list(select_color.keys())[index])))
			else:
				print('Job done, please press 0')

	cv2.imshow("image", img)
	print('Please select color of %s layer' %(str(list(select_color.keys())[index])))
	# calling the mouse click event
	cv2.setMouseCallback("image", click_event)
	
	cv2.waitKey(0)
	
	cv2.destroyAllWindows()
	
	index = 0
	
	return select_color


if __name__ == '__main__':
	file_path = '/home/yizi/Documents/phd/historical_map_project/test/BHdV_PL_ATL20Ardt_1929_0003.jpg'
	image = cv2.imread(file_path)
	select_color = color_picker(image)
	print(select_color)
