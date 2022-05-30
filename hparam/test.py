# from skopt import BayesSearchCV
import autokeras as ak
from tensorflow.keras import datasets
import tensorflow as tf
from autokeras import tuners
# from sklearn.model_selection import RandomizedSearchCV
# from skopt import gp_minimize
# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.backend import clear_session
import os, random, skopt, time, traceback
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

def reset_search_mem():
	global hparam_vals, hparam_check_list
	hparam_vals = {}
	hparam_check_list = []


'''
	Setup project defaults
'''
EPOCHS = None
MAIN = True

overwrite_num = None		# Which prior AK model to overwrite, None if create new model save file

SEED = int(random.random() * 1000.0)
print('Seed:', SEED)

reset_search_mem()

# TUNER_CLASSES = {
# 	"bayesian": tuners.BayesianOptimization,
# 	"random": tuners.RandomSearch,
# 	"hyperband": tuners.Hyperband,
# 	"greedy": tuners.Greedy,
# }



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


# class HPO_Callback(callbacks.Callback):
# 	'''
# 		Callback H-Param modifier class
# 	'''
# 	optimizers = {
# 	"SGD": tf.keras.optimizers.SGD(), 
# 	"RMSprop": tf.keras.optimizers.RMSprop(), 
# 	"Adam": tf.keras.optimizers.Adam(), 
# 	"Adadelta": tf.keras.optimizers.Adadelta(), 
# 	"Adagrad": tf.keras.optimizers.Adagrad(), 
# 	"Adamax": tf.keras.optimizers.Adamax(), 
# 	"Nadam": tf.keras.optimizers.Nadam(), 
# 	"Ftrl": tf.keras.optimizers.Ftrl()
# 	}

# 	def __init__(self, lr, optimizer):
# 		super(HPO_Callback, self).__init__()
# 		self.lr = lr
# 		self.optimizer = optimizer

# 	def on_train_begin(self, logs=None):
# 		# Set the current optimizer
# 		self.model.optimizer = self.optimizers[self.optimizer]

# 		# Set the current learning rate
# 		tf.keras.backend.set_value(self.model.optimizer.lr, self.lr)
# 		# Get the current learning rate from model's optimizer.
# 		lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
# 		# Print the current learning rate.
# 		print("\nOptimizer is " + self.optimizer + ". Learning rate is %6.4f." % (lr))

# 	# def on_epoch_begin(self, epoch, logs=None):
# 	# 	if not hasattr(self.model.optimizer, "lr"):
# 	# 		raise ValueError('Optimizer must have a "lr" attribute.')
# 	# 	# Get the current learning rate from model's optimizer.
# 	# 	lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
# 	# 	# Print the current learning rate.
# 	# 	print("Learning rate is %6.4f." % (lr))


class customTuner(ak.engine.tuner.AutoTuner):
	def __init__(self, oracle, hypermodel, **kwargs):
		super().__init__(oracle, hypermodel, **kwargs)



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
	
	# Separate out the hyperparameters from [loss, tuners, alpha, beta, factor, hyperband_iterations]
	loss = hparams[0]
	oracle = hparams[1]
	alpha = hparams[2]
	beta = hparams[3]
	factor = hparams[4]
	hyperband_iterations = hparams[5]


	# learning_rate=hparams[2]
	# optimizer=hparams[3]

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
		clf = ak.AutoModel(inputs=input_node, outputs=output_node, objective='val_loss', loss=loss, tuner=oracle, seed=SEED, project_name=project_name, directory=MY_DIR, overwrite=overwrite, max_trials=1)

		# Local AutoModel creation
		# clf = ak.AutoModel(inputs=input_node, outputs=output_node, objective='val_accuracy', loss=loss, tuner=oracle, seed=SEED, overwrite=True, max_trials=1)

		if oracle == 'bayesian':
			clf.tuner.oracle.alpha = float(alpha)
			clf.tuner.oracle.beta = float(beta)
		elif oracle == 'hyperband':
			clf.tuner.oracle.factor = int(factor)
			clf.tuner.oracle.hyperband_iterations = int(hyperband_iterations)
		
		# Fit the model w/ set learning rate
		clf.fit(train_data, epochs=None)
		# clf.fit(train_data, epochs=None, callbacks=[HPO_Callback(lr=learning_rate, optimizer=optimizer)])

		# Evaluate the model
		model_eval = clf.evaluate(val_data)

		# Get and log the model evaluation loss
		out = log_output(model_eval)
	except Exception as e:
		out = str(traceback.format_exc())
		print(out)
		print('\n', e)
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
		normalize=True,
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

	# batchSize = [4, 8, 16, 32, 64]
	# objectives = ['val_accuracy']
	loss = ['categorical_crossentropy', 'binary_crossentropy']
	tuners = ['greedy', 'bayesian', 'hyperband', 'random']
	alpha = (1e-6, 10.0)
	beta = (1e-4, 100)
	factor = (2, 10)
	hyperband_iterations = (1, 10)

	# Greedy Oracle Hparams:	Objective, Max Trials, seed
	# Bayesian OHP:				Objective, Max Trials, seed, alpha, beta
	# 		alpha: 		Float, the value added to the diagonal of the kernel matrix during fitting. 
	# 						It represents the expected amount of noise in the observed performances in Bayesian optimization. Defaults to 1e-4.
	# 		beta: 		Float, the balancing factor of exploration and exploitation. The larger it is, the more explorative it is. Defaults to 2.6.
	# Hyperband OHP: 			Objective, Max Trials, seed, factor, hyperband_iterations
	# 		hyperband_iterations:	Integer, at least 1, the number of times to iterate over the full Hyperband algorithm. 
	# 								One iteration will run approximately max_epochs * (math.log(max_epochs, factor) ** 2) cumulative epochs across all trials. 
	# 								It is recommended to set this to as high a value as is within your resource budget. Defaults to 1.
	# 		factor:	Integer, the reduction factor for the number of epochs and number of models for each bracket. Defaults to 3.
	# Random OHP: 				Objective, Max Trials, seed
	# 
	# total list: [alpha, beta, factor, hyperband_iterations]


	x0 = [loss[0], tuners[1], alpha[0], beta[0], factor[0], 2]
	dims = [loss, tuners, alpha, beta, factor, hyperband_iterations]


	print('*'*50, '\nBeginning Bayesian Hyperparameter Optimization\n', '*'*50)

	# Bayesian HPO
	ret = skopt.gp_minimize(threaded_min_func, x0=x0, dimensions=dims)
	print(ret.x)
	print(ret.fun)
	print('hparam vals: ', hparam_check_list)

	# Reset all known hparam configurations
	reset_search_mem()
	print('*'*50, '\nBeginning Random Search Hyperparameter Optimization\n', '*'*50)

	# Random Search HPO
	ret = skopt.dummy_minimize(threaded_min_func, x0=x0, dimensions=dims)
	print(ret.x)
	print(ret.fun)
	print('hparam vals: ', hparam_check_list)

	# Random search cv using sklearn
	# {'C': scipy.stats.expon(scale=100), 'gamma': scipy.stats.expon(scale=.1), 'kernel': ['rbf'], 'class_weight':['balanced', None]}
	# rnd_search_rf = RandomizedSearchCV(rf_clf, param_distributions=param_distribs, n_iter=10, cv=5, scoring='accuracy', random_state=42)
	# rnd_search_rf.fit(X_train,y_train)
	

	# for _ in range(30):
	# 	for loss_fun in loss:
	# 		for tuner in tuners:
	# 			hp = (loss_fun, tuner)
	# 			threaded_min_func(hp)
	# hp = (loss[1], tuners[0])
	# for i in range(10):
		# threaded_min_func(hp)
	# print('hparam vals: ', hparam_check_list)



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


def get_data():
	# (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
	(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()  # 'label_mode' param glitches it
	return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
	# Gather data
	(x_train, y_train), (x_test, y_test) = get_data()

	if MAIN:
		# Wrap data in Dataset objects.
		train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
		# y_train = tf.data.Dataset.from_tensor_slices(y_train)
		val_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))

		main()
	
	else:
		run_base()

