# from skopt import BayesSearchCV
import autokeras as ak
from tensorflow.keras import datasets
import tensorflow as tf
# from sklearn.model_selection import RandomizedSearchCV
# from skopt import gp_minimize
# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.backend import clear_session
import os, random, skopt


'''
	Setup Parallel Processing
'''
GPUS = tf.config.list_logical_devices('GPU')


'''
	Setup Project Name
'''
MY_DIR = '/home/zharris1/Documents/Jobs/Workspace/prior_runs/'
project_name = 'prior_runs/image_classifier_'
i = 0
dir_files = os.listdir(MY_DIR)
while project_name + str(i) in dir_files:
	i += 1
project_name = project_name + str(i)
os.system("mkdir " + project_name)

'''
	Setup project defaults
'''
EPOCHS = None
MAIN = False
SEED = 17
# SEED = int(random.random() * 100.0)
# print(SEED)

# def get_fit_model(x_train, y_train, h_params=None):
# 	clf = ak.ImageClassifier(overwrite=True, max_trials=1)
# 	clf.fit(x_train, y_train, epochs=EPOCHS)
# 	return clf


# def test_model(clf):
# 	predicted_y = clf.predict(x_test)
# 	print(predicted_y)

# 	# Evaluate the best model with testing data.
# 	print(clf.evaluate(x_test, y_test))


def minimizable_func(hparams):
	# (x_train, y_train), (x_test, y_test) = data
	# objective = hparams[0]
	loss = hparams[0]
	tuner = hparams[1]
	# epochs = hparams[2]
	# objective, loss, tuner, epochs = hparams
	# tf.debugging.set_log_device_placement(True)
	# strategy = tf.distribute.MirroredStrategy(gpus)
	# with strategy.scope():
	clf = ak.ImageClassifier(objective='val_accuracy', loss=loss, tuner=tuner, seed=SEED, project_name=project_name, overwrite=True, max_trials=1, distribution_strategy=tf.distribute.MirroredStrategy(GPUS))
	clf.fit(train_data, epochs=int(50))
	# clf.export_model()
	# return 1-clf.evaluate(x_test, y_test)[1]
	out = 1.0-clf.evaluate(val_data)[1]
	clear_session()
	return out



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

	# epochs = [1, 200]

	dims = [loss, tuners]

	ret = skopt.gp_minimize(minimizable_func, x0=[loss[0], tuners[0]], dimensions=dims)
	print(ret.x)
	print(ret.fun)




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
	model = ak.ImageClassifier(overwrite=True, max_trials=1, seed=SEED, project_name=project_name)
	model.fit(x_train, y_train, epochs=EPOCHS)
	predicted_y = model.predict(x_test)
	print(predicted_y)
	print(model.evaluate(x_test, y_test))




if __name__ == "__main__":
	# Gather data
	(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()


	if MAIN:
		# Wrap data in Dataset objects.
		train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
		# y_train = tf.data.Dataset.from_tensor_slices(y_train)
		val_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))

		# The batch size must now be set on the Dataset objects.
		batch_size = 32
		train_data = train_data.batch(batch_size)
		# y_train = y_train.batch(batch_size)
		val_data = val_data.batch(batch_size)

		# Disable AutoShard.
		options = tf.data.Options()
		options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
		train_data = train_data.with_options(options)
		val_data = val_data.with_options(options)
		
		main()
	
	else:
		run_base()

