import numpy as np
import os
import pickle
from pathlib import Path

import cv2.cv2 as cv2
from PIL import Image
import matplotlib.pyplot as plt

from color_space import color_quantization
from image_crop import create_save_image_batches, join_image_batches


if __name__ == '__main__':
	# Create and save smaller image
	# path = '/home/yizi/Documents/phd/historical_map_project/test/BHdV_PL_ATL20Ardt_1929_0003.jpg'
	# img = cv2.imread(path)
	# img = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
	#
	# plt.imshow(img)
	# plt.show()
	#
	# img = img[3000:5000, 3000:5000]
	# img = Image.fromarray(img)
	# img.save('/home/yizi/Documents/phd/historical_map_project/test/BHdV_PL_ATL20Ardt_1929_0003.jpg')
	
	# # 1. crop
	# path = '/home/yizi/Documents/phd/historical_map_project/test/'
	# create_save_image_batches(path)
	
	# 2. quantization
	path = '/home/yizi/Documents/phd/historical_map_project/image_generator/BHdV_PL_ATL20Ardt_1929_0003/image_batches/'
	file_names = os.listdir(path)
	for file in file_names:
		file_path = path + file
		quant_image = color_quantization(file_path, exe_median_cut=True)

	# 3. Join and merge
	current_dir = str(Path(os.getcwd()).parent) + '/image_generator/BHdV_PL_ATL20Ardt_1929_0003/'
	join_image_batches(current_dir)
