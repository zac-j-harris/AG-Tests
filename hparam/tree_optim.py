# from skopt import BayesSearchCV
# import autokeras as ak
# from tensorflow.keras import datasets
# import tensorflow as tf
# from autokeras import tuners
# from sklearn.model_selection import RandomizedSearchCV
from skopt import gp_minimize
from sklearn.tree import DecisionTreeClassifier
# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
# from tensorflow.keras.backend import clear_session
import os, random, skopt, time, traceback
from threading import Thread
# from keras import metrics
# from tensorflow.keras import callbacks
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

'''
	Setup project defaults
'''

MAIN = True


SEED = int(random.random() * 1000.0)
# SEED = 814
print('Seed:', SEED)

N_CALLS = 11
N_CALLS0 = 50
rand_starts = 1


max_depth = (1, 3)
min_samples_split = (2, 20)
min_samples_leaf = (1, 10)
min_weight_fraction_leaf = (0, 0.5)
# max_leaf_nodes = (0, 100)
max_leaf_nodes = (2, 3)
dims = [max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_leaf_nodes]
# dims = [max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf]
x0 = [1, 2, 1, 0, max_leaf_nodes[0]]
# x0 = [1, 2, 1, 0]

tested = []
tested2 = []
all_outs = []


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


def tree_func(hparams):
	global tested2
	max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_leaf_nodes = hparams
	# max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf= hparams
	max_leaf_nodes = max_leaf_nodes if max_leaf_nodes > 1 else None

	outs = [i[0] for i in tested2]
	if hparams in outs:
		for i in range(len(tested2)):
			if tested2[i][0] == hparams:
				return tested2[i][1]

	clf_model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split,
		min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_leaf_nodes=max_leaf_nodes)
	clf_model.fit(x_train,y_train)
	predictions = clf_model.predict(x_test)
	out = 1.0-accuracy_score(y_test, predictions)
	tested2.append((hparams, out))
	return out



def minimizable_func(hparams, log=True):
	'''
		Function defining a single AK training and evaluation run.
	'''
	global tested2, tested
	kappa = hparams
	outs = [i[0] for i in tested]
	if hparams in outs:
		for i in range(len(tested)):
			if tested[i][0] == hparams:
				return tested[i][1]
	ret = skopt.gp_minimize(tree_func, x0=x0, dimensions=dims, n_random_starts=rand_starts, random_state=SEED, acq_func='LCB', kappa=kappa, n_calls=N_CALLS0)
	if log:
		print(ret.x)
		print(1.0 - ret.fun)
		# print('hparam vals: ', hparam_check_list)
		outs = [i[1] for i in tested2]
		all_outs.append(outs)
		# plt.plot(outs)
		# plt.show()
	tested2 = []
	tested.append((hparams, ret.fun))
	return ret.fun



def main():
	global SEED, N_CALLS, tested

	kappa = (0, 10)
	x02 = [1.96]
	dims2 = [kappa]


	print('*'*50, '\nBeginning Bayesian Hyperparameter Optimization\n', '*'*50)

	# Bayesian HPO
	ret = skopt.gp_minimize(threaded_min_func, x0=x02, dimensions=dims2, random_state=SEED, n_calls=N_CALLS)
	outs = [i[1] for i in tested]
	# plt.plot(outs)
	# plt.show()
	tested = []
	print(ret.x)
	print(1.0 - ret.fun)
	# print('hparam vals: ', hparam_check_list)

	# print('*'*50, '\nBeginning Random Search Hyperparameter Optimization\n', '*'*50)

	# Random Search HPO
	# ret = skopt.dummy_minimize(threaded_min_func, x0=x0, n_calls=N_CALLS, dimensions=dims, random_state=SEED)
	# print(ret.x)
	# print(ret.fun)
	# print('hparam vals: ', hparam_check_list)





def run_base():
	'''
		Non-Optimized Hyperparameter AutoModel run over the same dataset
	'''
	global tested, tested2
	print('*'*50, '\nBeginning Base Hyperparameter Optimization\n', '*'*50)
	minimizable_func(1.96, log=True)


def get_data(pima=True):
	
	if not pima:
		data = pd.read_csv(
			'https://github.com/datagy/data/raw/main/titanic.csv', 
			usecols=['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare'])
		data = data.dropna()
		X = data.copy()
		y = X.pop('Survived')
	else:
		data = pd.read_csv('~/Datasets/pima-indians-diabetes/pima-indians-diabetes.csv')
		data = data.dropna()
		X = data.copy()
		y = X.pop('HasDiabetes')

	x_train, x_test, y_train, y_test = train_test_split(X, y, random_state = 100)
	return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
	# Gather data
	(x_train, y_train), (x_test, y_test) = get_data(pima=True)

	if MAIN:
		run_base()
		tested = []
		main()
		for i in all_outs:
			plt.plot(i)
		plt.show()
	else:
		run_base()

