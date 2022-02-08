# from skopt import BayesSearchCV
import autokeras as ak
from tensorflow.keras import datasets
from sklearn.model_selection import RandomizedSearchCV


def get_fit_model(h_params=None):
	clf = ak.ImageClassifier(overwrite=True, max_trials=1)
	# Feed the image classifier with training data.
	clf.fit(x_train, y_train, epochs=1)
	return clf


def test_model(clf):
	# Predict with the best model.
	predicted_y = clf.predict(x_test)
	print(predicted_y)


	# Evaluate the best model with testing data.
	print(clf.evaluate(x_test, y_test))




def main():
	'''
	Objectives: val_accuracy, val_loss, https://faroit.com/keras-docs/1.2.2/objectives/#available-objectives
	Loss: keras loss function
	Tuners: greedy', 'bayesian', 'hyperband' or 'random'
	'''
	batchSize = [4, 8, 16, 32, 64]
	objectives = ['val_accuracy', 'val_loss']
	loss = ['categorical_crossentropy']
	max_trials = [1]
	tuners = ['greedy', 'bayesian', 'hyperband', 'random']

	# h_params = {'objective': objectives, 'tuner': tuners, 'loss': loss, 'max_trials': max_trials}
	# create a dictionary from the hyperparameter grid
	model = get_fit_model()
	# test_model(model)
	# quit()

	grid = dict(
		objective=objectives,
		tuner=tuners,
		loss=loss,
		max_trials=max_trials
	)

	searcher = RandomizedSearchCV(estimator=model, n_jobs=-1, cv=3,
	param_distributions=grid, scoring="accuracy")
	searchResults = searcher.fit(trainData, trainLabels)

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


if __name__ == "__main__":
	# Gather data
	(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

	main()
