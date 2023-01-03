#!/usr/bin/env python
# coding: utf-8

# #                                                             Q2(a)

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

training = r"C:\Users\alish\OneDrive\Documents\Alishbah\CSE6363_Machine Learning\Project-1\axf4185_project_1\dataset\Training_Data.txt"
test = r"C:\Users\alish\OneDrive\Documents\Alishbah\CSE6363_Machine Learning\Project-1\axf4185_project_1\dataset\Test Data.txt"
Training_Data = readFile(training)
Test_Data = readFile(test)

print("Training Data:")
print(Training_Data)
print()
print("Test Data:")
print(Test_Data)


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


# ### Split Training data by class 

# In[3]:


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


# ### Calculating mean

# In[4]:


def mean(numbers):
    return sum(numbers)/float(len(numbers))


# ### Calculating the standard deviation

# In[5]:


def stdev(numbers):
    avg = mean(numbers)
    variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
    return sqrt(variance)


# ### Calculating mean, stdev and count for each column in a dataset

# In[6]:


def summarize_dataset(dataset):
    summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
    del(summaries[-1])
    return summaries


# ### Split dataset by class then calculate statistics for each Feature

# In[7]:


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


# ### Calculating Gaussian probability distribution function

# In[8]:


def calculate_probability(x, mean, stdev):
    exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
    return (1 / (sqrt(2 * pi) * stdev)) * exponent


# ### Calculate  probabilities of predicting each class for given Test Data

# In[9]:


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

# In[10]:


def predict(summaries, row):
    probabilities = calculate_class_probabilities(summary, Training_Data[0])
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label


# ### Naive Bayes Algorithm

# In[11]:


def naive_bayes(train, test):
    summarize = summarize_by_class(train)
    predictions = list()
    for row in test:
        output = predict(summarize, row)
        predictions.append(output)
    return(predictions)
 
print (naive_bayes(Training_Data,Test_Data), "--->" , "[' W' ' W' ' W' ' W']")

