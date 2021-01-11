from _csv import reader
from random import seed
from random import randrange
from math import sqrt
from math import exp
from math import pi
import numpy as np

##################################
# Step 1 - separate by class =>
# Split the dataset by class values, returns a dictionary
def separate_by_class(filename):
	separated = dict()
	dataset = open(filename, "r")
	line = dataset.readline()
	while line != '':  # The EOF char is an empty string
		line = line[:-1] # removing the new line char
		data = line.split(r",")
		vector = data[1:] # those are all the attributes
		class_value = "edible" if data[0] == "e" else "poisonous"
		if (class_value not in separated):
			separated[class_value] = list()
		separated[class_value].append(vector)
		line = dataset.readline()
	return separated

# Step 2 - summarize Dataset =>


# Calculate the standard deviation of a list of numbers


# Calculate the mean, stdev and count for each column in a dataset
def summarize_dataset(dataset):
	summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
	del(summaries[-1])
	return summaries

# Step 3 - Summarize Data By Class =>
# Split dataset by class then calculate statistics for each row
def summarize_by_class(dataset):
	separated = separate_by_class(dataset)
	summaries = dict()
	for class_value, rows in separated.items():
		summaries[class_value] = summarize_dataset(rows)
	return summaries

# Step 4 - Gaussian Probability Density Function =>
# calculate the Gaussian probability distribution function for x
def calculate_probability(x, mean, stdev):
	exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
	return (1 / (sqrt(2 * pi) * stdev)) * exponent

# Step 5 - Class Probabilities
def calculate_class_probabilities(summaries, row):
	total_rows = sum([summaries[label][0][2] for label in summaries])
	probabilities = dict()
	for class_value, class_summaries in summaries.items():
		probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
		for i in range(len(class_summaries)):
			mean, stdev, count = class_summaries[i]
			probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
	return probabilities

# Few more functions for implementing the algorithm
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for _ in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores

# Predict the class for a given row
def predict(summaries, row):
	probabilities = calculate_class_probabilities(summaries, row)
	best_label, best_prob = None, -1
	for class_value, probability in probabilities.items():
		if best_label is None or probability > best_prob:
			best_prob = probability
			best_label = class_value
	return best_label

# Naive Bayes Algorithm
def naive_bayes(train, test):
	summarize = summarize_by_class(train)
	predictions = list()
	for row in test:
		output = predict(summarize, row)
		predictions.append(output)
	return(predictions)

def load_dataset(filename):
	dataset = list()
	with open(filename, 'r') as file:
		readerDataset = reader(file)
		for row in readerDataset:
			if not row:
				continue
			dataset.append(row)
	return np.array(dataset)

# Test Naive Bayes on Mushroom dataset
seed(1)
filename = "Mushroom dataset/agaricus-lepiota.data"
dataset = load_dataset(filename)
print(dataset)
# for i in range(len(dataset[0])-1):
# 	str_column_to_float(dataset, i)
# # convert class column to integers
# str_column_to_int(dataset, len(dataset[0])-1)
# # evaluate algorithm
# n_folds = 5
# scores = evaluate_algorithm(dataset, naive_bayes, n_folds)
# print('Scores: %s' % scores)
# print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))


############
# Im writing here psudocode that we can use after
# first in order to predict a new mushroom we need to make a model
# i guess that would be just -
# for each attrubute take the frequency of Poisonous|that attribute and Edible|that attribute
# the if we have sth like : p,x,s,n,t,p,f,c,n,k,e,e,s,s,w,w,p,w,o,p,k,s,u
# and we need to clasify that we need to take :
# where P is poisonous=> P(P) * P(p|P) * P(x|P) * P(s|P)..... / P(p) * P(x) * P(s)... and do the same for Edible and check which has the higer probbaility?
