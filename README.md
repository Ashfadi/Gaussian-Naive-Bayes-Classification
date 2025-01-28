# Gaussian Naive Bayes Classification Project

## Overview
This project applies the Gaussian Naive Bayes algorithm to classify individuals into gender categories based on their physical measurements. The project evaluates the classifier's accuracy using various feature sets and compares the effectiveness of different Gaussian Naive Bayes configurations.

## Objectives
1. Implement the Gaussian Naive Bayes classifier.
2. Analyze the impact of different features on classification accuracy.
3. Assess the classifier's performance with cross-validation.
4. Determine the effect of removing the 'Age' feature on accuracy.

## Tools and Technologies
- **Programming Language**: Python
- **Libraries**: 
  - `numpy` for numerical operations
  - `math` for mathematical functions

## Dataset Details
- **Training Data**: Includes features like height, weight, age, and gender.
- **Test Data**: Used to evaluate the classifier, consisting of similar features without labels.

## Key Components

### **1. Data Preparation**
- **Script**: `Project1_Q2(a)_Alishbah_Fahad.py`
- Processes raw data, converting gender labels from characters to binary classes (`1` for female, `0` for male).

### **2. Gaussian Naive Bayes Implementation**
- **Script**: `Project1_Q2(b)_Alishbah_Fahad.py`
- Implements the classifier, calculating probabilities using Gaussian distribution.

### **3. Cross-Validation**
- **Script**: `Project1_Q2(c)_Alishbah_Fahad.py`
- Applies k-fold cross-validation to assess model performance.

### **4. Feature Analysis**
- **Script**: `Project1_Q2(d)_Alishbah_Fahad.py`
- Analyzes the impact of removing the 'Age' feature on the classifier's accuracy.

### **5. Comparative Analysis**
- **Document**: `Project1_Q2(e)_Alishbah_Fahad.pdf`
- Discusses findings, noting that the KNN model generally outperforms Gaussian Naive Bayes unless the 'Age' feature is removed.
