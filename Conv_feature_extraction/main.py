import os
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras.utils import plot_model


# consine similarity
def cosine_similarity(ratings):
	sim = ratings.dot(ratings.T)
	norms = np.array([np.sqrt(np.diagonal(sim))])
	return (sim / norms / norms.T)


if __name__ == '__main__':
	dir = '../image_generator/AD075CA_STDF0685/'
	file_names = os.listdir(dir)
	file_names.sort()
	print(file_names)
	print('The number of croped images: ', len(file_names))
	
	y_test = []
	x_test = []
	for file_name in file_names:
		abs_file_path = dir + file_name
		img = image.load_img(abs_file_path, target_size=(224, 224))
		y_test.append(int(file_name.split('.')[0]))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		if len(x_test) > 0:
			x_test = np.concatenate((x_test, x))
		else:
			x_test = x
	
	# convert input to VGG format
	x_test = preprocess_input(x_test)
	
	# include_top=False: exclude top(last) 3 fully-connected layers. get features dim=(1,7,7,512)
	model = VGG16(weights='imagenet', include_top=False)
	plot_model(model, show_shapes=True, to_file='model.png')
	print(model.summary())
	
	# use VGG to extract features
	features = model.predict(x_test)

	# flatten as one dimension
	features_compress = features.reshape(len(y_test), 7 * 7 * 512)
	
	# compute consine similarity
	cos_sim = cosine_similarity(features_compress)
	
	# random choose 5 samples to test
	inputNos = np.random.choice(len(y_test), 5, replace=False)
	
	fig = plt.figure(figsize=(24, 24))
	index = 1
	for inputNo in inputNos:
		# select two best similar images
		top = np.argsort(-cos_sim[inputNo], axis=0)[0:2]
		recommend = [y_test[i] for i in top]
		output = 'input: \'{}\', recommend: {}'.format(inputNo, recommend)
		print(output)
		input_path = dir + str(inputNo) + '.png'
		fig.add_subplot(5, 3, index)
		
		image_input = np.array(Image.open(input_path))
		
		plt.title(str(inputNo+1))
		plt.axis('off')
		plt.imshow(image_input)
		index += 1
		
		for re in recommend:
			recommend_path = dir + str(re) + '.png'
			image_output = np.array(Image.open(recommend_path))
			
			fig.add_subplot(5, 3, index)
			plt.title(index)
			plt.axis('off')
			plt.imshow(image_output)
			index += 1
	plt.show()
