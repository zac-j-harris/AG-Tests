import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import HuberRegressor
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error #, accuracy_score, mean_absolute_error
import csv
import random
from time import perf_counter
import logging
import warnings
# from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from skopt import gp_minimize, dummy_minimize, gbrt_minimize


logger = logging.getLogger("MainLogger")
logging.basicConfig(level=logging.INFO)


_server_path = "/home/zharris1/Documents/Github/Arcc-git-tests/hw-clustering/HW/"
_pc_path = "../"
_current_path = _server_path

ORIG_DATAPATH = _current_path+"Data/wine_quality/winequality-whites.csv"
FIX_DATAPATH = _current_path+"Data/wine_quality/fixed-winequality-whites.csv"

warnings.filterwarnings("ignore", message="")

fig = plt.figure()
ax = plt.axes()

STEP_SIZE = 5
RUNS = 300
TEST_SIZE = 0.20

RF_hps = [(1, 100), ["squared_error", "absolute_error", "friedman_mse", "poisson"], (2, 100), (1, 100), (0.0, 0.5),
          ["sqrt", "log2", None], (2, int(1e8)), (0.0, 1e6), [False, True], (0.0, 1e6)]
SVM_hps = [(0.0, 1e4), (1e-7, 1e2), (1e-4, 1e5), ["epsilon_insensitive", "squared_epsilon_insensitive"],
           [False, True], (1e-4, 1e4), (1, int(1e6))]
MLR_hps = [[False, True], [False, True]]
HR_hps = [(1.0, 1e8), (1, int(1e6)), (0.0, 1e8), [False, True], [False, True], (1e-7, 1e2)]

Bayes_Opt = gp_minimize
Random_Opt = dummy_minimize
GBRT_Opt = gbrt_minimize

test_data, test_labels = [], []
test_arr = []

# 15 -> 39 -> 100 (exponential)

"""
Help received:
Generally, info on sklearn methods, xgboost, and numpy APIs was taken from the API-specific online documentation.
"""


class HPO_Class():
	# gp_minimize, dummy_minimize, forest_minimize, gbrt_minimize
	def __init__(self, hpo_fn, opt_fn, hps, data, labels, seed):
		self.hpo_fn = hpo_fn
		self.opt_fn = opt_fn
		self.data = data
		self.labels = labels
		self.seed = seed
		self.h_params = hps

	def optimize(self):
		global test_arr
		test_arr = []
		result = self.hpo_fn(func=self.__minimizable_func__, dimensions=self.h_params, n_calls=RUNS)
		print(self.__class__.__name__, result.x)
		print(result.fun)
		print(result.x_iters)
		print(result.func_vals)
		print(test_arr)
	def plot(self) -> None:
		# plot(means, stds, k, label=label, runs=runs)
		pass
	def __minimizable_func__(self, hparams) -> float:
		return train_and_get_test_MSE(func=self.opt_fn, data=self.data, labels=self.labels,
		                              seed=self.seed, hparams=hparams)

def timed(func):
	"""Wrapper to time function runtime"""
	
	# Info on decorators: https://realpython.com/primer-on-python-decorators/
	def wrapper(*args, **kwargs):
		runtime = perf_counter()
		out = func(*args, **kwargs)
		logger.info("{0} function took {1:.2f} secs".format(func.__name__, (perf_counter()-runtime)))
		return out
	return wrapper


def get_data(fname=""):
	"""Returns 2D numpy arrays of the training and test data"""
	out = None
	with open(fname, 'r') as file:
		csv_reader = csv.reader(file, delimiter=',')
		out = np.array(list(csv_reader))
	columns = out[0,1:]
	data = out[1:,1:-1].astype(float)
	labels = out[1:, -1:].astype(int).flatten()
	return data, labels, columns

# @timed
def SVM(X, y, seed, hparams) -> LinearSVR:
	"""Returns a fit sklearn Linear SVM Model"""
	# SVM = SVR()
	SVM = LinearSVR(random_state=seed, epsilon=hparams[0], tol=hparams[1], C=hparams[2], loss=hparams[3],
	                fit_intercept=hparams[4], intercept_scaling=hparams[5], max_iter=hparams[6])
	SVM.fit(X, y)
	return SVM

# @timed
def RF(X, y, seed, hparams) -> RandomForestRegressor:
	"""Returns a fit sklearn RF Model"""
	RF = RandomForestRegressor(random_state=seed, n_estimators=hparams[0], criterion=hparams[1],
	                           min_samples_split=hparams[2], min_samples_leaf=hparams[3],
	                           min_weight_fraction_leaf=hparams[4], max_features=hparams[5], max_leaf_nodes=hparams[6],
	                           min_impurity_decrease=hparams[7], warm_start=hparams[8], ccp_alpha=hparams[9],
	                           verbose=1, n_jobs=-1)
	RF.fit(X, y)
	return RF

# @timed
def MLR(X, y, seed, hparams) -> LinearRegression:
	"""Returns a fit sklearn Linear Model"""
	MLR = LinearRegression(fit_intercept=hparams[0], positive=hparams[1])
	MLR.fit(X, y)
	return MLR


# @timed
def HR(X, y, seed, hparams) -> HuberRegressor:
	"""Returns a fit sklearn Linear Model"""
	HR = HuberRegressor(epsilon=hparams[0], max_iter=hparams[1], alpha=hparams[2], warm_start=hparams[3],
	                    fit_intercept=hparams[4], tol=hparams[5])
	HR.fit(X, y)
	return HR


def test_model(model, val_data, val_labels):
	"""Simple method to test a model"""
	pred_labels = model.predict(val_data)
	MSE = mean_squared_error(val_labels, pred_labels)
	print(type(model), " gives MSE: %.3f" % MSE)


def train_and_get_test_MSE(func, data, labels, seed, subarray_size=None, **kwargs):
	global test_arr
	"""Trains a model with the given arguments, and returns the test MSE"""
	train_data, val_data = data
	train_labels, val_labels = labels
	# if subarray_size:
	# 	# generate k-sized subarrays
	# 	model = func(X=train_data[:subarray_size,:], y=train_labels[:subarray_size], seed=seed, **kwargs)
	# else:
	model = func(X=train_data, y=train_labels, seed=seed, **kwargs)
	test_mse = mean_squared_error(test_labels, model.predict(test_data))
	test_arr.append(test_mse)
	return mean_squared_error(val_labels, model.predict(val_data))  # unrounded MSE
	# return mean_squared_error(val_labels, np.round(model.predict(val_data)))  # rounded MSE
	# return accuracy_score(val_labels, np.round(model.predict(val_data)))  # accuracy
	# return mean_absolute_error(val_labels, model.predict(val_data))  # unrounded MAE
	# return mean_absolute_error(val_labels, np.round(model.predict(val_data)))  # rounded MAE


def get_subset_array(size, ind):
	"""Returns a random subset of indices of an array of some size"""
	arr = np.arange(size)
	np.random.shuffle(arr)
	return arr[:ind]


def save_plot(fname, title):
	"""Saves the plot, and labels it"""
	plt.ylabel('MSE')
	plt.xlabel('Epochs')
	# plt.ylim(0, 2.5)
	plt.title(title)
	plt.legend(loc='upper right')
	from io import BytesIO
	figfile = BytesIO()
	plt.savefig(figfile, format='png', dpi=200)
	try:
		f = open(fname + '.png', 'wb')
		f.write(figfile.getvalue())
	finally:
		f.close()
	plt.clf()
	fig = plt.figure()
	ax = plt.axes()


def plot(mse, label):
	"""Plot the given MSE values with a bounds for the known std"""
	x = [i for i in range(len(mse))]
	plt.plot(x, mse, lw=1.5, label=label)
	# stds = np.array(stds)
	# means = np.array(means)
	# print(stds)
	# if runs > 1:
	# 	ax.fill_between(k, means+stds, means-stds, alpha=0.3)
	# plt.errorbar(k, means, yerr=stds, color="black", capsize=3, lw=1.0)
	# save_plot(fname, fig)


def HPO(hpo_fn, data, labels, seed):
	# print(label, means, k)
	# plot(means, stds, k, label=label, runs=runs)
	optimizers = [(RF, RF_hps)]
	# optimizers = [(SVM, SVM_hps)]
	# optimizers = [(MLR, MLR_hps), (HR, HR_hps)]
	for optimizer in optimizers:
		hpo_inst = HPO_Class(hpo_fn=hpo_fn, opt_fn=optimizer[0], hps=optimizer[1], data=data, labels=labels, seed=seed)
		hpo_inst.optimize()
		# hpo_inst.plot()
		# save_plot(fname="../Plots/comb_plot")


def save_data(data, fname):
	"""Saves data using pickle"""
	with open(fname, 'wb') as file:
		pickle.dump(data, file)


def load_data(fname):
	"""Loads data using pickle"""
	try:
		with open(fname, 'wb') as file:
			data = pickle.load(file)
		return data
	except:
		return None


def main():
	global test_data, test_labels
	# Generate seed
	# seed = random.randint(0, 1e9)
	# print("Seed: ", seed)
	seed = 71115919

	random.seed(seed)
	np.random.seed(seed)
	
	# Grab Data
	data, labels, columns = get_data(fname=ORIG_DATAPATH)
	logger.info("Data collected.")

	train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state=seed,
	                                                                    test_size=TEST_SIZE)

	train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, random_state=seed, test_size=TEST_SIZE)
	data = (train_data, val_data)
	labels = (train_labels, val_labels)
	# logger.info("Beginning Bayesian HPO")
	# HPO(Bayes_Opt, data=data, labels=labels, seed=seed)
	# logger.info("Beginning Random HPO")
	# HPO(Random_Opt, data=data, labels=labels, seed=seed)
	logger.info("Beginning GB Regression Tree HPO")
	HPO(GBRT_Opt, data=data, labels=labels, seed=seed)


	# bo_svm = [35.10695742, 35.10695742, 35.10695742, 35.10695742, 35.10695742, 35.10695742,  35.10695742, 35.10695742, 35.10695742, 35.10695742, 35.10695742, 0.65844139, 35.10695742, 0.69199919, 35.10695742, 35.10695742, 0.76209636, 0.65844139, 0.63248508, 0.63248508, 0.63248508, 0.63248508, 0.63248508, 0.63248508, 0.63248508, 0.63248508, 0.63248508, 0.63248508, 0.63248508, 0.65852053]
	# bo_rf = [0.80166147, 0.80166147, 0.80166147, 0.78307336, 0.7831544, 0.78304847, 0.78317585, 0.78321504, 0.78303652, 0.78321504, 0.78317383, 0.78311032, 0.78309728, 0.78314256, 0.78399165, 0.78315229, 0.78314256, 0.78315495, 0.78316413, 0.78316594, 0.78314256, 0.80166147, 0.80166147, 0.7832252, 0.78371391, 0.80166147, 0.78317066, 0.78318656, 0.78318058, 0.7831743]
	# bo_mlr = [0.56171071, 0.58100835, 0.56171071, 0.56171071, 0.56171071, 0.56171071, 0.56171071, 0.58100835, 0.56605826, 0.58129822, 0.56171071, 0.58129822, 0.58100835, 0.58100835, 0.58100835, 0.56171071, 0.58100835, 0.56171071, 0.56171071, 0.56171071, 0.56171071, 0.56605826, 0.58129822, 0.56605826, 0.56171071, 0.58100835, 0.58129822, 0.56171071, 0.56171071, 0.58100835]
	# bo_hr = [0.74695986, 0.74622979, 0.74816118, 12.51552337, 24.56852896, 24.98710882, 21.05057012, 0.74802205, 0.74613838, 8.92246702, 4.0588664,  0.74650179, 7.80308585, 0.76427435, 0.57764902, 0.76428218, 0.74853181, 26.98601076, 0.7452242,  0.74888325, 0.76419622, 0.74888325, 0.74840837, 0.74888325, 0.76427435, 0.76427435, 0.74888325, 0.76427435, 0.74888325, 12.95021277]
	# dum_svm = [35.10695742, 35.10695742, 35.10695742, 35.10695742, 35.10695742, 35.10695742, 35.10695742, 35.10695742, 35.10695742, 35.10695742, 35.10695742, 35.10695742, 35.10695742, 35.10695742, 35.10695742, 35.10695742, 35.10695742, 35.10695742, 35.10695742, 35.10695742, 35.10695742, 35.10695742, 35.10695742, 35.10695742, 35.10695742, 35.10695742, 35.10695742, 35.10695742, 35.10695742, 35.10695742]
	# gbrt_svm = [35.10695742, 35.10695742, 35.10695742, 35.10695742, 35.10695742, 35.10695742, 35.10695742, 35.10695742, 35.10695742, 35.10695742, 35.10695742, 35.10695742, 35.10695742, 35.10695742, 35.10695742, 35.10695742, 35.10695742, 35.10695742, 35.10695742, 35.10695742, 35.10695742, 35.10695742, 35.10695742, 35.10695742, 35.10695742, 35.10695742, 35.10695742, 35.10695742, 35.10695742, 35.10695742]

	# bo_hr = [i for i in bo_hr if i < 2]
	# bo_svm = [i for i in bo_svm if i < 2]


	# plot(bo_svm, "RF")
	# plot(bo_rf, "SVM")
	# plot(bo_mlr, "MLR")
	# plot(bo_hr, "HR")
	# save_plot(fname="../Plots/comb_plot", title="MSE for Bayesian HPO")





if __name__ == "__main__":
	main()
	pass


# 21 - 100RF
# 20 - 50RF
# 19 - 25RF
# 22 - SVM
# 17 - MLR/HR

# GBRT only
# RF100 - 27
# SVM - 28


