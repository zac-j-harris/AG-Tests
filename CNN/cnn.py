import numpy as np
import os, csv, pickle, lzma, math

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
	import tensorflow as tf
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
		y[i] = np.asarray(tf.keras.utils.to_categorical(y[i], 10))
		print(y[i].shape)

		x[i] = np.reshape(x[i][:l], (l, 28, 28, 1))
		x[i] = x[i].astype('float32')
		x[i] /= 255
	return x, y


def build_model():
	from tensorflow.keras import Sequential
	from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
	'''
	https://www.analyticsvidhya.com/blog/2020/02/learn-image-classification-cnn-convolutional-neural-networks-3-datasets/

	'''
	model = Sequential()
	model.add(Conv2D(25, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu', input_shape=(28, 28, 1)))
	model.add(MaxPool2D(pool_size=(1, 1)))
	model.add(Flatten())
	model.add(Dense(100, activation='relu'))
	model.add(Dense(10, activation='softmax'))
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
	import tensorflow as tf

	# x, y = gather_data(scale_subset=1000)
	# x_train, x_test = x[0], x[1]
	# y_train, y_test = y[0], y[1]

	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
	assert x_train.shape == (60000, 28, 28)
	assert x_test.shape == (10000, 28, 28)
	assert y_train.shape == (60000,)
	assert y_test.shape == (10000,)
	x_train = np.reshape(x_train, (60000, 28, 28, 1))
	x_test = np.reshape(x_test, (10000, 28, 28, 1))
	y_train = np.asarray(tf.keras.utils.to_categorical(y_train, 10))
	y_test = np.asarray(tf.keras.utils.to_categorical(y_test, 10))
	x_train = x_train / 255.0
	x_test = x_test / 255.0
	model = build_model()

	# training the model for 10 epochs
	model.fit(x_train, y_train, batch_size=128, epochs=12, validation_data=(x_test, y_test))
	# model.fit(x_train, y_train, batch_size=2, epochs=10, validation_data=(x_test, y_test))

	model.save('cnn_model')



def color_corrected():
	from PIL import Image
	for i in range(1, 281):
	# for i in range(1, 31):
		rgb = np.asarray(Image.open("CNN_Data/split_data/" + str(i) + ".png")).copy()
		alpha = np.asarray([[[0 if min(col) > 170 else 255 for _ in col] for col in row] for row in rgb[:,:,:3]], dtype=np.uint8)
		img = Image.fromarray(alpha, 'RGB')
		img.show()
		# quit()
		img.save("CNN_Data/color_corrected/" + str(i) + ".png")
		quit()


def cropped():
	from PIL import Image
	for i in range(1, 281):
	# for i in range(69, 90):
		rgb = np.asarray(Image.open("CNN_Data/color_corrected/" + str(i) + ".png")).copy()
		bounds = [-1, -1, -1, -1]# left, right, up, down
		rows = [sum(row) for row in rgb[:,:,0]]
		# print(rows)
		for j in range(len(rows)):
			if rows[j] >=3 and bounds[2] == -1:
				bounds[2] = j

			if rows[j] >= 3:
				bounds[3] = j
		bounds[2] = max(bounds[2]-1, 0)
		bounds[3] = min(bounds[3]+1, len(rows)-1)

		cols = [sum(rgb[:,j,0]) for j in range(len(rgb[0]))]
		for j in range(len(cols)):
			if cols[j] >=3 and bounds[0] == -1:
				bounds[0] = j

			if cols[j] >= 3:
				bounds[1] = j
		bounds[0] = max(bounds[0]-1, 0)
		bounds[1] = min(bounds[1]+1, len(cols)-1)
		old_shape = rgb.shape
		rgb = rgb[bounds[2]:bounds[3], bounds[0]:bounds[1], :]
		# print(rgb.shape, old_shape, bounds)
		# if min(rgb.shape) < 1:
		# 	print(rows)
		img = Image.fromarray(rgb, 'RGB')
		# img.show()

		img.save("CNN_Data/cropped/" + str(i) + ".png")



def scaled():
	from PIL import Image
	for f_num in range(1, 281):
	# for f_num in range(1, 31):
		rgb = np.asarray(Image.open("CNN_Data/cropped/" + str(f_num) + ".png")).copy()
		new_dim = max(rgb.shape)
		if new_dim == rgb.shape[1]:
			old_dim = rgb.shape[0]
			half = (new_dim - old_dim) // 2
			rgb = np.asarray([[ rgb[row - half, col, :] if row >= half and (row - half < old_dim) else [0, 0, 0]
			                         for col in range(new_dim)] for row in range(new_dim)], dtype=np.uint8)
		else:
			old_dim = rgb.shape[1]
			half = (new_dim - old_dim) // 2
			rgb = np.asarray([[rgb[row, col-half, :] if col >= half and (col - half < old_dim) else [0, 0, 0]
			                   for col in range(new_dim)] for row in range(new_dim)], dtype=np.uint8)
		img = Image.fromarray(rgb, 'RGB')
		img = img.resize((28, 28), Image.ANTIALIAS)
		# img.show()
		# quit()
		img.save("CNN_Data/final_scaled/" + str(f_num) + ".png")


def get_labels():
	import csv
	import tensorflow as tf
	with open('CNN_Data/Labels.csv', 'r') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		out = [i for i in reader]
	out = out[1:]
	out = [int(i[1]) for i in out]
	return np.asarray(tf.keras.utils.to_categorical(out, 10))


def test_model():
	import tensorflow as tf
	from PIL import Image
	model = tf.keras.models.load_model('cnn_model')
	x = np.asarray([np.asarray(Image.open("CNN_Data/final_scaled/" + str(f_num) + ".png")).copy()[:,:,0]/255 for f_num in range(1, 281)])
	x = np.reshape(x, (280, 28, 28, 1))
	pred_labels = model.predict(x)
	labels = get_labels()
	mse = lambda a, b: (np.square(np.subtract(a, b))).mean()
	print("MSE = ", mse(labels, pred_labels))
	def indexOfMax(l):
		m = max(l)
		for i in range(len(l)):
			if l[i] == m:
				return i
	predictions = [indexOfMax(i) for i in pred_labels]
	original = [indexOfMax(i) for i in labels]
	num_matching = 0
	for i in range(len(original)):
		if original[i] == predictions[i]:
			num_matching += 1
	print("Accuracy = ", num_matching / len(original) * 100.0)



if __name__=='__main__':
	main()
	# test_model()
	# scaled()
	# test_model()