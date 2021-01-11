from _csv import reader
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import svm


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


# Logistic regression using sklearn
def logistic_regression(X_train, X_test, y, y_train, y_test, name):
    # Encode the train and test data
    X_train_encoded, X_test_encoded, y_train_encoded, y_test_encoded = encode_train_test(X_train, X_test, y, y_train,
                                                                                         y_test)

    # Fit the model
    model = LogisticRegression(max_iter=500)
    model.fit(X_train_encoded, y_train_encoded)

    # Predict the test cases
    y_predicted = model.predict(X_test_encoded)

    # Evaluate predictions
    accuracy = accuracy_score(y_test_encoded, y_predicted)
    print('Accuracy of logistic regression on ', name, ' dataset: ', accuracy * 100, "%")


def random_forest(X_train, X_test, y, y_train, y_test, name):
    # Encode the train and test data
    X_train_encoded, X_test_encoded, y_train_encoded, y_test_encoded = encode_train_test(X_train, X_test, y, y_train,
                                                                                         y_test)

    # Fit the model
    regressor = RandomForestRegressor(n_estimators=20, random_state=1)
    regressor.fit(X_train_encoded, y_train_encoded)

    # Predict the test cases
    y_predicted = regressor.predict(X_test_encoded).astype(int)

    # Evaluate predictions
    accuracy = accuracy_score(y_test_encoded, y_predicted)
    print('Accuracy of random forests on ', name, ' dataset: ', accuracy * 100, "%")


def support_vector_machines(X_train, X_test, y, y_train, y_test, name):
    # Encode the train and test data
    X_train_encoded, X_test_encoded, y_train_encoded, y_test_encoded = encode_train_test(X_train, X_test, y, y_train,
                                                                                         y_test)

    # Fit the model
    SVM = svm.LinearSVC()
    SVM.fit(X_train_encoded, y_train_encoded)

    # Predict the test cases
    y_predicted = SVM.predict(X_test_encoded)

    # Evaluate predictions
    accuracy = accuracy_score(y_test_encoded, y_predicted)
    print('Accuracy of Support Vector Machines on ', name, ' dataset: ', accuracy * 100, '%')


def run_test(name, title):
    dataset = load_dataset(name)
    X = dataset[:, 1:]
    y = dataset[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    logistic_regression(X_train, X_test, y, y_train, y_test, title)
    random_forest(X_train, X_test, y, y_train, y_test, title)
    support_vector_machines(X_train, X_test, y, y_train, y_test, title)


# Test Algorithms on datasets
mushroom = "Mushroom dataset/agaricus-lepiota.data"
tumor = "Mushroom dataset/primary-tumor.data"
run_test(mushroom, "mushroom")
print()
run_test(tumor, "tumor")
