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
	global project_name
	# return
	try:
		if (not (overwrite_num is None)) and (not overwrite_check):
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
MAIN = False
# SEED = 67 # 17
overwrite_num = None
overwrite_check = False
SEED = int(random.random() * 1000.0)
print(SEED)
hparam_vals = []

# def get_fit_model(x_train, y_train, h_params=None):
# 	clf = ak.ImageClassifier(overwrite=True, max_trials=1)
# 	clf.fit(x_train, y_train, epochs=EPOCHS)
# 	return clf


# def test_model(clf):
# 	predicted_y = clf.predict(x_test)
# 	print(predicted_y)

# 	# Evaluate the best model with testing data.
# 	print(clf.evaluate(x_test, y_test))


'''
Thread Class
'''
class ThreadWithReturnValue(Thread):
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


def threaded_min_func(hparams):
	thread = ThreadWithReturnValue(target=minimizable_func, args=(hparams,))
	thread.daemon = True
	thread.start()
	while thread.is_alive():
		time.sleep(5)
	out = thread.join()
	return out


def minimizable_func(hparams):
	global project_name, overwrite_check, hparam_vals
	set_proj_name()
	# (x_train, y_train), (x_test, y_test) = data
	# objective = hparams[0]
	loss = hparams[0]
	tuner = hparams[1]
	# epochs = hparams[2]
	# objective, loss, tuner, epochs = hparams
	if (not (overwrite_num is None)) and (not overwrite_check):
		overwrite = False
		overwrite_check = True
	else:
		overwrite = True
	# tf.debugging.set_log_device_placement(True)
	# strategy = tf.distribute.MirroredStrategy(gpus)
	# with strategy.scope():
	try:
		input_node, output_node = build_custom_search_space()
		# clf = ak.AutoModel(inputs=input_node, outputs=output_node, metrics=['loss', 'accuracy', 'val_loss', 'val_accuracy'], objective='val_accuracy', loss=loss, tuner=tuner, seed=SEED, project_name=project_name, directory=MY_DIR, overwrite=True, max_trials=1, distribution_strategy=tf.distribute.MirroredStrategy(GPUS))
		clf = ak.AutoModel(inputs=input_node, outputs=output_node, objective='val_loss', loss=loss, tuner=tuner, seed=SEED, project_name=project_name, directory=MY_DIR, overwrite=overwrite, max_trials=1)
		# clf = ak.AutoModel(inputs=input_node, outputs=output_node, objective='val_accuracy', loss=loss, tuner=tuner, seed=SEED, overwrite=True, max_trials=1)
		clf.fit(train_data, epochs=None)
		# clf.export_model()
		# return 1-clf.evaluate(x_test, y_test)[1]
		model_eval = clf.evaluate(val_data)
		if type(model_eval) == list:
			out = 1 - model_eval[1]
			print('Validation (Loss, Acc): ', model_eval)
			print('Output (1 - acc): ', out)
		else:
			out = model_eval
			print('Validation Loss: ', model_eval)
	except Exception as e:
		print(e)
		out = 1.0
	clear_session()
	hparam_vals.append(out)
	return out


def build_custom_search_space():
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
	'''
	batchSize = [4, 8, 16, 32, 64]
	# objectives = ['val_accuracy']
	loss = ['categorical_crossentropy', 'binary_crossentropy']
	# max_trials = [2**i for i in range(6)]
	tuners = ['greedy', 'bayesian', 'hyperband', 'random']
	# learning_rate = (1e-4, 5.0)

	# epochs = [1, 200]

	dims = [loss, tuners]

	ret = skopt.gp_minimize(threaded_min_func, x0=[loss[0], tuners[0]], dimensions=dims)
	print(ret.x)
	print(ret.fun)
	print('hparam vals: ', hparam_vals)




	# h_params = {'objective': objectives, 'tuner': tuners, 'loss': loss, 'max_trials': max_trials}
	# create a dictionary from the hyperparameter grid
	# model = get_fit_model(x_train, y_train)
	# test_model(model)
	# quit()
	# model = KerasClassifier(build_fn=get_model, verbose=1)

	# grid = dict(
	# 	'objective'=objectives,
	# 	'tuner'=tuners,
	# 	'loss'=loss,
	# 	'max_trials'=max_trials
	# )

	# learnRate = [1e-2, 1e-3, 1e-4]
	# epochs = [10, 20, 30, 40]
	# grid = dict(
	# 	# learnRate=learnRate,
	# 	batch_size=batchSize,
	# 	loss=loss,
	# 	epochs=epochs
	# )

	# grid = GridSearchCV(
	# 	n_jobs=-1, 
	# 	verbose=1,
	# 	return_train_score=True,
	# 	cv=kfold_splits,  #StratifiedKFold(n_splits=kfold_splits, shuffle=True)
	# 	param_grid=param_grid,
	# )

	# searcher = RandomizedSearchCV(estimator=model, n_jobs=-1, cv=3,
	# param_distributions=grid, scoring="accuracy")
	# searchResults = searcher.fit(x_train, y_train)

	# # initialize a random search with a 3-fold cross-validation and then
	# # start the hyperparameter search process
	# print("[INFO] performing random search...")
	# searcher = RandomizedSearchCV(estimator=model, n_jobs=-1, cv=3,
	# 	param_distributions=grid, scoring="accuracy")
	# searchResults = searcher.fit(x_train, y_train)
	
	# # summarize grid search information
	# bestScore = searchResults.best_score_
	# bestParams = searchResults.best_params_
	# print("[INFO] best score is {:.2f} using {}".format(bestScore,
	# 	bestParams))


	# # extract the best model, make predictions on our data, and show a classification report
	# print("[INFO] evaluating the best model...")
	# bestModel = searchResults.best_estimator_
	# accuracy = bestModel.score(x_test, y_test)
	# print("accuracy: {:.2f}%".format(accuracy * 100))



def run_base():
	global project_name, overwrite_check
	set_proj_name()
	# model = ak.ImageClassifier(overwrite=True, max_trials=1, seed=SEED, project_name=project_name, directory=MY_DIR)
	if (not (overwrite_num is None)) and (not overwrite_check):
		overwrite = False
		overwrite_check = True
	else:
		overwrite = True
	input_node, output_node = build_custom_search_space()
	# model = ak.AutoModel(inputs=input_node, outputs=output_node, metrics=['loss', 'accuracy', 'val_loss', 'val_accuracy'], objective='val_accuracy', overwrite=overwrite, max_trials=1, seed=SEED, project_name=project_name, directory=MY_DIR)
	model = ak.AutoModel(inputs=input_node, outputs=output_node, objective='val_loss', overwrite=overwrite, max_trials=1, seed=SEED, project_name=project_name, directory=MY_DIR)
	# model = ak.AutoModel(inputs=input_node, outputs=output_node, objective='val_loss', overwrite=overwrite, max_trials=1, seed=SEED)
	model.fit(x_train, y_train, epochs=EPOCHS)
	predicted_y = model.predict(x_test)
	# print(predicted_y)
	model_eval = model.evaluate(x_test, y_test)
	# print('Metrics: ', model.metrics_names)
	print('Eval output: ', model_eval)
	print('Val Accuracy: ', model_eval[1])




if __name__ == "__main__":
	# Gather data
	(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

	# set_proj_name()
	if MAIN:
		# Wrap data in Dataset objects.
		train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
		# y_train = tf.data.Dataset.from_tensor_slices(y_train)
		val_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))

		# The batch size must now be set on the Dataset objects.
		# batch_size = 256
		# train_data = train_data.batch(batch_size)
		# # y_train = y_train.batch(batch_size)
		# val_data = val_data.batch(batch_size)

		# # Disable AutoShard.
		# options = tf.data.Options()
		# options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
		# train_data = train_data.with_options(options)
		# val_data = val_data.with_options(options)
		
		main()
	
	else:
		run_base()

