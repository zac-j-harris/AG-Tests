from tensorflow import keras
import numpy as np
import os, csv, pickle, lzma
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten

data_fname = '~/Datasets/mnist'
datasets = ['mnist_train', 'mnist_test']

def save_dat_compressed(data_out, fname):
	with lzma.open(fname, "wb") as file:
		pickle.dump(data_out, file)
def load_dat_compressed(fname):
	with lzma.open(fname, "rb") as file:
		data_out = pickle.load(file)
	return data_out

def save_dat(data_out, fname):
	with open(fname, "wb") as file:
		pickle.dump(data_out, file)
def load_dat(fname):
	with open(fname, "rb") as file:
		data_out = pickle.load(file)
	return data_out

def convert_csv(compressed=False):
	for cur_file in datasets:
		path = os.path.join(data_fname, cur_file)
		with open(path + '.csv', 'r') as csvfile:
			reader = csv.reader(csvfile, delimiter=',')
			images = [i for i in reader]
			labels = [i[0] for i in images]
			images = [i[1:] for i in images]
			if compressed:
				save_dat_compressed(labels, path + '_labels' + '.xz')
				save_dat_compressed(images, path + '_images' + '.xz')
			else:
				save_dat(labels, path + '_labels' + '.pkl')
				save_dat(images, path + '_images' + '.pkl')


def gather_data(scale_subset=1):
	x = [None, None]
	y = [None, None]
	for i in range(2):
		f = datasets[i]
		path = os.path.join(data_fname, f)
		l = int(len(load_dat_compressed(path + '_labels' + '.xz'))/scale_subset)

		y[i] = np.asarray(load_dat_compressed(path + '_labels' + '.xz'))
		x[i] = np.asarray(load_dat_compressed(path + '_images' + '.xz'))
		t = list(zip(x[i], y[i]))
		np.random.shuffle(t)
		x[i], y[i] = zip(*t)

		y[i] = y[i][:l]
		y[i] = np.asarray(keras.utils.to_categorical(y[i], 10))
		print(y[i].shape)

		x[i] = np.reshape(x[i][:l], (l, 28, 28, 1))
		x[i] = x[i].astype('float32')
		x[i] /= 255
	return x, y


def build_model():
	'''
	https://www.analyticsvidhya.com/blog/2020/02/learn-image-classification-cnn-convolutional-neural-networks-3-datasets/

	'''
	# building a linear stack of layers with the sequential model
	model = Sequential()
	# convolutional layer
	model.add(
		Conv2D(25, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu', input_shape=(28, 28, 1)))
	model.add(MaxPool2D(pool_size=(1, 1)))
	# flatten output of conv
	model.add(Flatten())
	# hidden layer
	model.add(Dense(100, activation='relu'))
	# output layer
	model.add(Dense(10, activation='softmax'))

	# compiling the sequential model
	model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
	return model

# new_base_image_shape = (128, 128)
# new_overall_img_shape = (1024, 1024)
# Temporal data (y) includes: actual digit, upper left x, upper left y, box width, box length
# After mapping all of these, we can then begin running a couple (< 4) perturbations per image and training
#   Perhaps also include images with multiple characters and only go for one in the upper left
# Accuracies above 90 percent are good enough

# For actual model, predict and then cut while there is at least one decent prediction


def add_temporal_data(x, y):
	for i in range(2):
		pass


def main():
	# x, y = gather_data(scale_subset=1000)
	# x_train, x_test = x[0], x[1]
	# y_train, y_test = y[0], y[1]

	(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
	assert x_train.shape == (60000, 28, 28)
	assert x_test.shape == (10000, 28, 28)
	assert y_train.shape == (60000,)
	assert y_test.shape == (10000,)

	model = build_model()

	# training the model for 10 epochs
	model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_test, y_test))
	# model.fit(x_train, y_train, batch_size=2, epochs=10, validation_data=(x_test, y_test))

	model.save('cnn_model')

	# model = keras.models.load_model('path/to/location')


if __name__=='__main__':
	main()