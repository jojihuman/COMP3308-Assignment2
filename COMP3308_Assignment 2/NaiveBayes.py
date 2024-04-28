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

def mean_standard_deviation(training_data):
    # For each attribute, calculate mean and standard deviation
    num_of_features = len(training_data[0]) - 1 #Number of features excluding class label
    means = [{"yes": 0, "no": 0} for _ in range(num_of_features)]
    std_devs = [{"yes": 0, "no": 0} for _ in range(num_of_features)]
    counts = {"yes": 0, "no": 0}

    # Calculate mean
    for data in training_data:
        counts[data[-1]] += 1
        for i in range(num_of_features):
            means[i][data[-1]] += data[i]

    num_examples = len(training_data)
    for i in range(num_of_features):
        means[i]["yes"] /= counts["yes"]
        means[i]["no"] /= counts["no"]

    # Calculate standard deviation
    for data in training_data:
        for i in range(num_of_features):
            std_devs[i][data[-1]] += (data[i] - means[i][data[-1]]) ** 2

    for i in range(num_of_features):
        std_devs[i]["yes"] = math.sqrt(std_devs[i]["yes"] / (counts["yes"] - 1))
        std_devs[i]["no"] = math.sqrt(std_devs[i]["no"] / (counts["no"] - 1))

    return means, std_devs, counts

def probability_density_function(mean, std_dev, x):
    # Returns the PDF for a given attribute value (formula)
    exponent = -(((x - mean)**2)/ (2 * std_dev ** 2))
    return (1 / (math.sqrt(2 * math.pi) * std_dev)) * math.exp(exponent)

def classify_nb(training_filename, testing_filename):
    # Naive Bayes classification
    training_data = get_data1(training_filename)
    testing_data = get_data2(testing_filename)
    
    # Calculate mean and standard deviation
    means, std_devs, counts = mean_standard_deviation(training_data)

    num_of_yes = counts["yes"]
    num_of_no = counts["no"]

    result = []
    # Classify each testing data point
    for data in testing_data:
      # Initialise probabilities
        prob_yes = num_of_yes / (num_of_yes + num_of_no)
        prob_no = num_of_no / (num_of_yes + num_of_no)
        for i in range(len(data)):  # Iterate over features
            prob_yes *= probability_density_function(means[i]["yes"], std_devs[i]["yes"], data[i])
            prob_no *= probability_density_function(means[i]["no"], std_devs[i]["no"], data[i])

        # Decision-making
        result.append("yes" if prob_yes >= prob_no else "no")

    return result
