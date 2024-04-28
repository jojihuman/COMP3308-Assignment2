import csv
import math

def read_data(filename):
    # reads content of the file and return as list
    lines = []
    with open(filename) as f:
        lines = f.readlines()
    return lines

def get_data1(filename):
  # for training data
    lines = read_data(filename)
    data = []
    for line in lines:
        line = line.strip()
        data.append(line.split(","))
    # Convert non-class columns to float
    for dt in data:
        for i in range(len(dt) - 1):  # Exclude the last column (class label)
            dt[i] = float(dt[i])
    return data

def get_data2(filename):
  # for testing data
    lines = read_data(filename)
    data = []
    for line in lines:
        line = line.strip()
        data.append(line.split(","))
    # Convert non-class columns to float
    for dt in data:
        for i in range(len(dt)):
            dt[i] = float(dt[i])
    return data

def euclidean_distance(training_instance, testing_instance):
  # Measure euclidean distance between two points
    distance = 0
    for i in range(0,len(testing_instance)):
      # Calculates squared difference for each dimension and sum up (refer to formula)
        distance = distance + math.pow(training_instance[i]-testing_instance[i], 2)
    return math.sqrt(distance)

def classify_nn(training_filename, testing_filename, k):
    training_data = get_data1(training_filename)
    testing_data = get_data2(testing_filename)

    # Classify each testing instance
    predictions = []
    for testing_instance in testing_data:
        distances = []
        for training_instance in training_data:
            # Calculate distances between the test instance and all training instances
            distance = euclidean_distance(training_instance, testing_instance)
            distances.append({"distance": distance, "class": training_instance[-1]})  # Store distance and class label
        # Sort distances and get the k-nearest neighbors
        distances.sort(key=lambda x: x["distance"])
        nearest_neighbors = distances[:k]
        # Count the number of 'yes' and 'no' classes among the nearest neighbors
        yes_count = sum(1 for neighbor in nearest_neighbors if neighbor["class"] == "yes")
        no_count = sum(1 for neighbor in nearest_neighbors if neighbor["class"] == "no")
        # Predict the class label based on the majority vote
        prediction = "yes" if yes_count >= no_count else "no"
        predictions.append(prediction)
    
    return predictions
