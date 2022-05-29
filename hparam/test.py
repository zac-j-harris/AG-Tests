# from skopt import BayesSearchCV
import autokeras as ak
from tensorflow.keras import datasets
import tensorflow as tf
# from sklearn.model_selection import RandomizedSearchCV
# from skopt import gp_minimize
# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.backend import clear_session
import os, random, skopt, time
from threading import Thread
# from keras import metrics
from tensorflow.keras import callbacks
import numpy as np

'''
	Setup Parallel Processing
'''
GPUS = tf.config.list_logical_devices('GPU')


'''
	Setup Project Name
'''
MY_DIR = '/gscratch/zharris1/Workspace/prior_runs/'
project_name = 'auto_model'

def set_proj_name():
	'''
		Creates a custom project name after checking against currently used names.
		Sets global variable project_name for use as model save filename
	'''
	global project_name
	try:
		if (not (overwrite_num is None)):
			project_name = 'auto_model_' + str(overwrite_num)
		else:
			project_name = 'auto_model_'
			i = 0
			dir_files = os.listdir(MY_DIR)
			while project_name + str(i) in dir_files:
				i += 1
			project_name = project_name + str(i)
			os.system("mkdir " + MY_DIR + project_name)
	except:
		project_name = 'auto_model'

'''
	Setup project defaults
'''
EPOCHS = None
MAIN = True

overwrite_num = None		# Which prior AK model to overwrite, None if create new model save file

SEED = int(random.random() * 1000.0)
print('Seed:', SEED)

hparam_vals = {}
hparam_check_list = []



class ThreadWithReturnValue(Thread):
	'''
		Thread Class that returns value from internal function
	'''
	def __init__(self, group=None, target=None, name=None,
				 args=(), kwargs=None, *, daemon=None):
		Thread.__init__(self, group, target, name, args, kwargs, daemon=daemon)
		self._return = None
	def run(self):
		try:
			if self._target:
				self._return = self._target(*self._args, **self._kwargs)
		finally:
			del self._target, self._args, self._kwargs
	def join(self, timeout=None):
		Thread.join(self, timeout=timeout)
		return self._return


class HPO_Callback(callbacks.Callback):
	'''
		Callback H-Param modifier class
	'''
	def __init__(self, lr):
		super(HPO_Callback, self).__init__()
		self.lr = lr
		# tf.keras.backend.set_value(self.model.optimizer.lr, lr)

	def on_train_begin(self, logs=None):
		# Set the current learning rate
		tf.keras.backend.set_value(self.model.optimizer.lr, self.lr)
		# Get the current learning rate from model's optimizer.
		lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
		# Print the current learning rate.
		print("\nLearning rate is %6.4f." % (lr))

	# def on_epoch_begin(self, epoch, logs=None):
	# 	if not hasattr(self.model.optimizer, "lr"):
	# 		raise ValueError('Optimizer must have a "lr" attribute.')
	# 	# Get the current learning rate from model's optimizer.
	# 	lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
	# 	# Print the current learning rate.
	# 	print("Learning rate is %6.4f." % (lr))


def threaded_min_func(hparams):
	'''
		Function to internalize each AutoKeras run into a process.
	'''
	thread = ThreadWithReturnValue(target=minimizable_func, args=(hparams,))
	thread.daemon = True
	thread.start()
	while thread.is_alive():
		time.sleep(5)
	out = thread.join()
	return out


def log_output(model_eval):
	'''
		Function to log the output after each AK run.
	'''
	if type(model_eval) == list:
		if model_eval[1] is None:
			out = 10.0
		else:
			out = 1 - model_eval[1]
		print('Validation (Loss, Acc): ', model_eval)
		print('Output (1 - acc): ', out)
	else:
		if model_eval is None:
			out = 10.0
		else:
			out = model_eval
		print('Validation Loss: ', model_eval)
	return out


def minimizable_func(hparams):
	'''
		Function defining a single AK training and evaluation run.
	'''
	global project_name, hparam_vals, SEED, hparam_check_list

	set_proj_name()

	# Add hparam configuration to currently checked (log hparam search configuration in order w/ reused values)
	hparam_check_list.append(tuple(hparams))

	# Check for reuse of hparam configuration. If reused, return prior value.
	if not (hparam_vals == {}):
		if tuple(hparams) in hparam_vals.keys():
			return hparam_vals[tuple(hparams)]
	
	# Separate out the hyperparameters
	loss = hparams[0]
	tuner = hparams[1]
	learning_rate=hparams[2]

	# Code dealing with AutoModel reuse or new model creation
	if (not (overwrite_num is None)):
		overwrite = False
	else:
		overwrite = True
	
	# tf.debugging.set_log_device_placement(True)
	try:
		# Build a custom vanilla CNN model
		input_node, output_node = build_custom_search_space()
		
		# Server AutoModel creation
		clf = ak.AutoModel(inputs=input_node, outputs=output_node, objective='val_loss', loss=loss, tuner=tuner, seed=SEED, project_name=project_name, directory=MY_DIR, overwrite=overwrite, max_trials=1)
		
		# Local AutoModel creation
		# clf = ak.AutoModel(inputs=input_node, outputs=output_node, objective='val_accuracy', loss=loss, tuner=tuner, seed=SEED, overwrite=True, max_trials=1)
		
		# Fit the model w/ set learning rate
		clf.fit(train_data, epochs=None, callbacks=[HPO_Callback(lr=learning_rate)])

		# Evaluate the model
		model_eval = clf.evaluate(val_data)

		# Get and log the model evaluation loss
		out = log_output(model_eval)
	except Exception as e:
		print(e)
		out = 10.0

	# Clear TF session (just in case it helps to clear CUDA GPU memory)
	clear_session()
	
	# Store known hparam configuration worth
	hparam_vals[tuple(hparams)] = (out)

	return out


def build_custom_search_space():
	'''
		Function that creates and returns a custom Image Classifier model for AutoKeras
	'''
	input_node = ak.ImageInput()
	output_node = ak.ImageBlock(
		# Only search ResNet architectures.
		block_type="vanilla",
		# Normalize the dataset.
		normalize=False,
		# Do not do data augmentation.
		augment=False,
	)(input_node)
	# output_node = ak.ClassificationHead(metrics=['loss', 'accuracy', 'val_loss', 'val_accuracy'])(output_node)
	output_node = ak.ClassificationHead()(output_node)
	return input_node, output_node



def main():
	'''
		Objectives: val_accuracy, val_loss, https://faroit.com/keras-docs/1.2.2/objectives/#available-objectives
		Loss: keras loss function
		Tuners: greedy', 'bayesian', 'hyperband' or 'random'
		Learning Rate: [1e-4, 5.0]
	'''
	global hparam_check_list

	batchSize = [4, 8, 16, 32, 64]
	# objectives = ['val_accuracy']
	loss = ['categorical_crossentropy', 'binary_crossentropy']
	tuners = ['greedy', 'bayesian', 'hyperband', 'random']
	learning_rate = (1e-4, 5.0)


	dims = [loss, tuners, learning_rate]

	ret = skopt.gp_minimize(threaded_min_func, x0=[loss[0], tuners[0], 5e-3], dimensions=dims)
	print(ret.x)
	print(ret.fun)
	

	# for _ in range(30):
	# 	for loss_fun in loss:
	# 		for tuner in tuners:
	# 			hp = (loss_fun, tuner)
	# 			threaded_min_func(hp)
	# hp = (loss[1], tuners[0])
	# for i in range(10):
		# threaded_min_func(hp)
	print('hparam vals: ', hparam_check_list)



def run_base():
	'''
		Non-Optimized Hyperparameter AutoModel run over the same dataset
	'''
	global project_name

	set_proj_name()

	# Code dealing with AutoModel reuse or new model creation
	if (not (overwrite_num is None)):
		overwrite = False
	else:
		overwrite = True
	input_node, output_node = build_custom_search_space()

	# Server AutoModel creation
	model = ak.AutoModel(inputs=input_node, outputs=output_node, objective='val_loss', overwrite=overwrite, max_trials=1, seed=SEED, project_name=project_name, directory=MY_DIR)

	# Local AutoModel creation
	# model = ak.AutoModel(inputs=input_node, outputs=output_node, objective='val_loss', overwrite=overwrite, max_trials=1, seed=SEED)
	
	# 
	model.fit(train_data, epochs=EPOCHS)
	

	predicted_y = model.predict(x_test)
	# print(predicted_y)
	model_eval = model.evaluate(val_data)
	# print('Metrics: ', model.metrics_names)
	print('Eval output: ', model_eval)
	print('Val Accuracy: ', model_eval[1])


def get_norm_data():
	(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
	# (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()  # 'label_mode' param glitches it

	min_max_norm = lambda i, j: (i-np.min(j)) / (np.max(j)-np.min(j))
	x_train = min_max_norm(x_train, x_train)
	x_test = min_max_norm(x_test, x_train)
	y_train = min_max_norm(y_train, y_train)
	y_test = min_max_norm(y_test, y_train)
	return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
	# Gather data
	(x_train, y_train), (x_test, y_test) = get_norm_data()

	if MAIN:
		# Wrap data in Dataset objects.
		train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
		# y_train = tf.data.Dataset.from_tensor_slices(y_train)
		val_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))

		main()
	
	else:
		run_base()

