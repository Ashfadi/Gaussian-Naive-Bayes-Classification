#!/usr/bin/env python
# coding: utf-8

# #                                                             Q2(d)

# ### Importing Libraries and Data

# In[1]:


import numpy as np
import math
from math import sqrt
from math import exp
from math import pi
def clean_data(line):
    return line.replace('(', '').replace(')', '').replace(' ', '').strip().split(',')
def fetch_data(filename):
    with open(filename, 'r') as f:
        input_data = f.readlines()
        clean_input = list(map(clean_data, input_data))
        f.close()
    return clean_input
def readFile(dataset_path):
    input_data = fetch_data(dataset_path)
    input_np = np.array(input_data)
    return input_np

training = r"C:\Users\alish\OneDrive\Documents\Alishbah\CSE6363_Machine Learning\Project-1\axf4185_project_1\dataset\Program Data.txt"
Training_Data = readFile(training)

print("Training Data:")
print(Training_Data)


# ### Replacing 'W' and 'M' to '1' and '0' respectively

# In[2]:


for i in Training_Data:
    if i[3]=='W':
        i[3]=i[3].replace('W','1')
        i[3]=int(i[3])
    else:
        i[3]=i[3].replace('M','0')
        i[3]=int(i[3])
Training_Data=Training_Data.astype(float)
Training_Data = np.delete(Training_Data, 2, 1)


# ### Split a dataset into k folds

# In[3]:


from random import randrange
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# ### Calculate accuracy percentage

# In[4]:


def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# ### Evaluate an algorithm using a cross validation split

# In[5]:


def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        def remove_values_from_list(train_set, fold):
            return [value for value in train_set if value != fold]
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


# ### Split Training data by class 

# In[6]:


def separate_by_class(Trainingdata):
    separated = dict()
    for i in range(len(Trainingdata)):
        vector = Trainingdata[i]
        class_value = vector[-1]
        if (class_value not in separated):
            separated[class_value] = list()
        separated[class_value].append(vector)
    return separated
splitted_data = separate_by_class(Training_Data)
for label in splitted_data:
    print(label)
    for row in splitted_data[label]:
        print(row)


# ### Split dataset by class then calculate statistics for each Feature

# In[7]:


# Calculating mean
def mean(numbers):
    return sum(numbers)/float(len(numbers))

# Calculating the standard deviation
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
    return sqrt(variance)

# Calculating mean, stdev and count for each column in a dataset
def summarize_dataset(dataset):
    summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
    del(summaries[-1])
    return summaries

# Split dataset by class then calculate statistics for each Feature
def summarize_by_class(dataset):
    separated = separate_by_class(dataset)
    summaries = dict()
    for class_value, rows in separated.items():
        summaries[class_value] = summarize_dataset(rows)
    return summaries
summary = summarize_by_class(Training_Data)
for label in summary:
    print(label)
    for row in summary[label]:
        print(row)


# ### Calculating probabilities of predicting each class for given Test Data

# In[8]:


# Calculating Gaussian probability distribution function
def calculate_probability(x, mean, stdev):
    exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
    return (1 / (sqrt(2 * pi) * stdev)) * exponent

# Calculating probabilities of predicting each class for given Test Data
def calculate_class_probabilities(summaries, row):
    total_rows = sum([summaries[label][0][2] for label in summaries])
    probabilities = dict()
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
        for i in range(len(class_summaries)):
            mean, stdev, _ = class_summaries[i]
            probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
    return probabilities
probabilities = calculate_class_probabilities(summary, Training_Data[0])
print(probabilities)


# ### Predict the class for given Test Data

# In[9]:


def predict(summaries, row):
    probabilities = calculate_class_probabilities(summary, Training_Data[0])
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label


# ### Naive Bayes Algorithm

# In[10]:


def naive_bayes(train, test):
    summarize = summarize_by_class(train)
    predictions = list()
    for row in test:
        output = predict(summarize, row)
        predictions.append(output)
    return(predictions)


# ### Result

# In[11]:


n_folds = 120
scores = evaluate_algorithm(Training_Data, naive_bayes, n_folds)
Accuracy = (sum(scores)/float(len(scores)))
print('Accuracy: %.3f%%' % Accuracy)


# ###### The removal of the 'Age' feature has no effect on the accuracy of the Naive Bayes Model.
