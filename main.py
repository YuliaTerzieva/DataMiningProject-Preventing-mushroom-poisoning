from _csv import reader
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D


# Load the dataset into a numpy array
def load_dataset(filename):
    dataset = list()
    with open(filename, 'r') as file:
        readerDataset = reader(file)
        for row in readerDataset:
            if not row:
                continue
            dataset.append(row)
    return np.array(dataset)


# LabelEncoder for the y labels
def encode_labels(y):
    le = preprocessing.LabelEncoder()
    le.fit(y)
    encoded = le.transform(y)
    return le, encoded


# One hot encoding for the attributes
def encode_attributes(X):
    enc = preprocessing.OneHotEncoder(handle_unknown='ignore')
    enc.fit(X)
    encoded = enc.transform(X)
    return enc, encoded.toarray()


# Encode the train and test data
def encode_train_test(X_train, X_test, y, y_train, y_test):
    onehot_encoder, X_train_encoded = encode_attributes(X_train)
    X_test_encoded = onehot_encoder.transform(X_test).toarray()
    le, y_encoded = encode_labels(y)
    y_train_encoded = le.transform(y_train)
    y_test_encoded = le.transform(y_test)
    return X_train_encoded, X_test_encoded, y_train_encoded, y_test_encoded


# Tunig the C parameter for Logistic regression
def tuningCLR(X_train_encoded, X_test_encoded, y_train_encoded, y_test_encoded):
    # We look at C values between 0.5 and 5
    C_param_range = np.linspace(0.5, 5.0, endpoint=True)
    train_results = []
    test_results = []

    # For all C values we look at the AUC score on the training and test data
    for i in C_param_range:
        model = LogisticRegression(C=i, max_iter=200)
        model.fit(X_train_encoded, y_train_encoded)

        train_pred = model.predict(X_train_encoded)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train_encoded, train_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        # Add auc score to previous train results
        train_results.append(roc_auc)
        y_pred = model.predict(X_test_encoded)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test_encoded, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        # Add auc score to previous test results
        test_results.append(roc_auc)

    # Plotting the results
    line1, = plt.plot(C_param_range, train_results, "b", label="TrainAUC")
    line2, = plt.plot(C_param_range, test_results, "r", label="TestAUC")
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel("AUC score")
    plt.xlabel("C")
    plt.show()


# Logistic regression using sklearn
def logistic_regression(X_train, X_test, y, y_train, y_test, name):
    # Encode the train and test data
    X_train_encoded, X_test_encoded, y_train_encoded, y_test_encoded = encode_train_test(X_train, X_test, y, y_train,
                                                                                         y_test)

    # Fit the model
    model = LogisticRegression(C=2.5, max_iter=200)
    model.fit(X_train_encoded, y_train_encoded)

    # Predict the test cases
    y_predicted = model.predict(X_test_encoded)

    # Evaluate predictions
    accuracy = accuracy_score(y_test_encoded, y_predicted)
    print('Accuracy of logistic regression on ', name, ' dataset: ', accuracy * 100, "%")


# Tuning the max depth for the decision tree classifier
def tuningMaxDepthDTC(X_train_encoded, y_train_encoded, X_test_encoded, y_test_encoded):
    # We look at values between 1 and n with n our number of attributes
    max_depths = np.linspace(1, 22, 22, endpoint=True)
    train_results = []
    test_results = []

    # For all values for max_depth we look at AUC score on training and test data
    for max_depth in max_depths:
        dt = DecisionTreeClassifier(max_depth=max_depth)
        dt.fit(X_train_encoded, y_train_encoded)
        train_pred = dt.predict(X_train_encoded)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train_encoded, train_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        # Add auc score to previous train results
        train_results.append(roc_auc)
        y_pred = dt.predict(X_test_encoded)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test_encoded, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        # Add auc score to previous test results
        test_results.append(roc_auc)

    # Plotting the results
    line1, = plt.plot(max_depths, train_results, "b", label="TrainAUC")
    line2, = plt.plot(max_depths, test_results, "r", label="TestAUC")
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel("AUC score")
    plt.xlabel("Tree depth")
    plt.show()


# Tuning the minimum samples split value
def tuningMinSampleSplit(X_train_encoded, y_train_encoded, X_test_encoded, y_test_encoded):
    # We look at values between 0.1 and 1
    min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)
    train_results = []
    test_results = []

    # For all values for min_samples_split we look at AUC score on training and test data
    for min_samples_split in min_samples_splits:
        dt = DecisionTreeClassifier(min_samples_split=min_samples_split)
        dt.fit(X_train_encoded, y_train_encoded)
        train_pred = dt.predict(X_train_encoded)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train_encoded, train_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        # Add auc score to previous train results
        train_results.append(roc_auc)
        y_pred = dt.predict(X_test_encoded)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test_encoded, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        # Add auc score to previous test results
        test_results.append(roc_auc)

    # Plotting the results
    line1, = plt.plot(min_samples_splits, train_results, "b", label="TrainAUC")
    line2, = plt.plot(min_samples_splits, test_results, "r", label="TestAUC")
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel("AUC score")
    plt.xlabel("Minimul Samples Split")
    plt.show()


# Decision tree classifier using sklearn
def decision_tree(X_train, X_test, y, y_train, y_test, name):
    # Encode the train and test data
    X_train_en, X_test_en, y_train_en, y_test_en = encode_train_test(X_train, X_test, y, y_train, y_test)

    # Fit the model
    model = DecisionTreeClassifier(max_depth=7, min_samples_split=11)
    model.fit(X_train_en, y_train_en)

    # Predict the test cases
    y_predicted = model.predict(X_test_en)

    # Evaluate predictions
    accuracy = accuracy_score(y_test_en, y_predicted)
    print('Accuracy of decision trees on ', name, ' dataset: ', accuracy * 100, "%")


# Clustering using KMeans from sklearn
def clustering(X, y, name):
    # Create clustering of 2 clusters and predict labels
    kmeans_labels = KMeans(n_clusters=2, random_state=1).fit_predict(encode_attributes(X)[1])

    # Calculate and show rand score of clustering
    rand_score = adjusted_rand_score(encode_labels(y)[1], kmeans_labels)
    print('Rand score of KMeans clustering of the data on ', name, ' dataset: ', rand_score)


# Tuning the hyperparameters
def tuning(X_train, X_test, y, y_train, y_test):
    X_train_encoded, X_test_encoded, y_train_encoded, y_test_encoded = encode_train_test(X_train, X_test, y, y_train,
                                                                                         y_test)

    tuningCLR(X_train_encoded, X_test_encoded, y_train_encoded, y_test_encoded)
    tuningMaxDepthDTC(X_train_encoded, y_train_encoded, X_test_encoded, y_test_encoded)
    tuningMinSampleSplit(X_train_encoded, y_train_encoded, X_test_encoded, y_test_encoded)


# Diagnostics
def diagnostics(X_train, X_test, y_train, y_test):
    # Showing the split between poisonous and edible
    plt.hist(y_train, color="purple", label="Train")
    plt.hist(y_test, color="yellow", label="Test")
    plt.xlabel("Label -> P=poisonous, E=edible")
    plt.ylabel("Number of instances")
    plt.title("Distribution of mushrooms in test and train sets")
    plt.legend()
    plt.show()


# Running the tests
def run_tests(name, title):
    dataset = load_dataset(name)
    X = dataset[:, 1:]
    y = dataset[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    logistic_regression(X_train, X_test, y, y_train, y_test, title)
    decision_tree(X_train, X_test, y, y_train, y_test, title)
    clustering(X, y, title)
    # diagnostics(X_train, X_test, y_train, y_test)
    # tuning(X_train, X_test, y, y_train, y_test)


# Test Algorithms on dataset
mushroom = "Mushroom dataset/agaricus-lepiota.data"
run_tests(mushroom, "mushroom")
