import os

from color_space import color_quantization_click_and_select
from image_crop import create_save_image_batches, join_image_batches
from map_object_detection import map_objects as mp

from config import config

if __name__ == '__main__':
	# 0. Slice and save images
	# map_input_path = config.MAP_SOURCE
	# create_save_image_batches(map_input_path)
	
	# 1. quantization
	image_root_path = config.MAP_SAVE
	for image_index in os.listdir(image_root_path):
		batch_dir = image_root_path + image_index + '/image_batches/'
		for file in os.listdir(batch_dir):
			file_path = batch_dir + file
			# Seperate different layers of the map with human intervention
			color_quantization_click_and_select(file_path)
	
		# 2. Extract horizontal and vertical lines
		result_batch_path = image_root_path + image_index + '/color_quantization_result_batches/0_layer/'
		img_name = os.listdir(result_batch_path)
		for file in img_name:
			file_path = result_batch_path + file
			output_image_gray = mp.detect_longitude_latitude_hough_line_transform(file_path)
		
		# 3. Join and merge
		join_image_path = image_root_path + image_index + '/'
		join_image_batches(join_image_path)
