import csv
import sys
import math
import pandas as pd 

pima_data = pd.read_csv("pima.csv", header = None)

#make sure that the dataset is randomly sampled
shuffled_pima = pima_data.sample(frac = 1)

#separate dataset into X and y, X is the values without the classifications
X = shuffled_pima.drop([8], axis = 1)
y = shuffled_pima[8]

#let's make a 70/30 split on our dataset
train_index = int(0.7 * len(X))

X_train, X_test = X[:train_index], X[train_index:]
y_train, y_test = y[:train_index], y[train_index:]

X_train = X_train.values.tolist()
X_test = X_test.values.tolist()

y_train = y_train.values.tolist()
y_test = y_test.values.tolist()

def euclidean_distance(n1, n2):
    distance = 0
    for i in range(len(n1)):
        distance += (n1[i] - n2[i])**2
    return np.sqrt(distance)

def get_distances(x, train_set):
    distances = []
    for x in train_set:
        distance = euclidean_distance(x, train_set)
        distances.append(distance)
    return distances

for row in X_train:
    distance = euclidean_distance(first_row, row)
    print(distance)

def classify_nn(training_filename, testing_filename, k):
    # Read the training dataset
    training_data = pd.read_csv(training_filename, header=None)
    X_train = training_data.drop([training_data.columns[-1]], axis=1).values.tolist()
    y_train = training_data[training_data.columns[-1]].values.tolist()
    
    # Read the testing dataset
    testing_data = pd.read_csv(testing_filename, header=None)
    X_test = testing_data.values.tolist()
    
    # Classify each testing instance
    predictions = []
    for test_instance in X_test:
        # Calculate distances between the test instance and all training instances
        distances = get_distances(test_instance, X_train)
        # Get indices of k nearest neighbors
        nearest_indices = np.argsort(distances)[:k]
        # Get the corresponding labels of nearest neighbors
        nearest_labels = [y_train[i] for i in nearest_indices]
        # Predict the class label based on the majority vote
        prediction = max(set(nearest_labels), key=nearest_labels.count)
        predictions.append(prediction)
    
    return predictions