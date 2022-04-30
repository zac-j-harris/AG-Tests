# from skopt import BayesSearchCV
import autokeras as ak
from tensorflow.keras import datasets
# from sklearn.model_selection import RandomizedSearchCV
# from skopt import gp_minimize
# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


EPOCHS = 100


def get_fit_model(x_train, y_train, h_params=None):
	clf = ak.ImageClassifier(overwrite=True, max_trials=1)
	clf.fit(x_train, y_train, epochs=EPOCHS)
	return clf


def test_model(clf):
	# Predict with the best model.
	predicted_y = clf.predict(x_test)
	print(predicted_y)


	# Evaluate the best model with testing data.
	print(clf.evaluate(x_test, y_test))


def get_model():
	model = ak.ImageClassifier(overwrite=True, max_trials=1)
	return model

def main(x_train, y_train):
	'''
	Objectives: val_accuracy, val_loss, https://faroit.com/keras-docs/1.2.2/objectives/#available-objectives
	Loss: keras loss function
	Tuners: greedy', 'bayesian', 'hyperband' or 'random'
	'''
	batchSize = [4, 8, 16, 32, 64]
	objectives = ['val_accuracy', 'val_loss']
	loss = ['categorical_crossentropy', 'binary_crossentropy']
	max_trials = [1, 2]
	tuners = ['greedy', 'bayesian', 'hyperband', 'random']

	# h_params = {'objective': objectives, 'tuner': tuners, 'loss': loss, 'max_trials': max_trials}
	# create a dictionary from the hyperparameter grid
	# model = get_fit_model(x_train, y_train)
	# test_model(model)
	# quit()
	# model = KerasClassifier(build_fn=get_model, verbose=1)

	grid = dict(
		'objective'=objectives,
		'tuner'=tuners,
		'loss'=loss,
		'max_trials'=max_trials
	)

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

	searcher = RandomizedSearchCV(estimator=model, n_jobs=-1, cv=3,
	param_distributions=grid, scoring="accuracy")
	searchResults = searcher.fit(x_train, y_train)

	# initialize a random search with a 3-fold cross-validation and then
	# start the hyperparameter search process
	print("[INFO] performing random search...")
	searcher = RandomizedSearchCV(estimator=model, n_jobs=-1, cv=3,
		param_distributions=grid, scoring="accuracy")
	searchResults = searcher.fit(x_train, y_train)
	
	# summarize grid search information
	bestScore = searchResults.best_score_
	bestParams = searchResults.best_params_
	print("[INFO] best score is {:.2f} using {}".format(bestScore,
		bestParams))


	# extract the best model, make predictions on our data, and show a classification report
	print("[INFO] evaluating the best model...")
	bestModel = searchResults.best_estimator_
	accuracy = bestModel.score(x_test, y_test)
	print("accuracy: {:.2f}%".format(accuracy * 100))



def run_base(x_train, y_train, x_test, y_test):
	model = ak.ImageClassifier(overwrite=True, max_trials=1)
	model.fit(x_train, y_train, epochs=EPOCHS)
	predicted_y = clf.predict(x_test)
	print(predicted_y)
	print(clf.evaluate(x_test, y_test))


if __name__ == "__main__":
	# Gather data
	(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

	# main(x_train, y_train)
	run_base(x_train, y_train, x_test, y_test)

