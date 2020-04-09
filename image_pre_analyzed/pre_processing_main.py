import os
from pathlib import Path

from color_space import color_quantization, color_quantization_initial_centroid
from image_crop import create_save_image_batches, join_image_batches


if __name__ == '__main__':
	# 0. Slice and save images
	path = '/home/yizi/Documents/phd/historical_map_project/test/'
	create_save_image_batches(path, batch_size=(1000, 1000))

	# 1. quantization
	path = '/home/yizi/Documents/phd/historical_map_project/image_generator/BHdV_PL_ATL20Ardt_1929_0003/image_batches/'
	file_names = os.listdir(path)
	for file in file_names:
		file_path = path + file
		# Seperate different layers of the map with human intervention
		color_quantization_initial_centroid(file_path)
		# Seperate different layers of the map automatically
		# quant_image = color_quantization(file_path, exe_median_cut=True)

	# 3. Join and merge
	current_dir = str(Path(os.getcwd()).parent) + '/image_generator/BHdV_PL_ATL20Ardt_1929_0003/'
	join_image_batches(current_dir)
