#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm,datasets
from sklearn.metrics import accuracy_score


# In[2]:


data=pd.read_csv("twitter_training.csv")


# In[3]:


data.head()


# In[4]:


data.info()


# In[5]:


data.describe()


# In[6]:


data.isnull().sum()


# In[7]:


data.dropna()


# In[8]:


# Importing the necessary libraries 
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score 

# Importing the dataset from the sklearn library into a local variable called dataset
dataset = load_digits()

# Splitting the data test into train 70% and test 30%.
# x_train, y_train are training data and labels respectively 
# x_test, y_test are testing data and labels respectively 
x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.30, random_state=4)

# Making the SVM Classifer
Classifier = SVC(kernel="linear")

# Training the model on the training data and labels
Classifier.fit(x_train, y_train)

# Using the model to predict the labels of the test data
y_pred = Classifier.predict(x_test)

# Evaluating the accuracy of the model using the sklearn functions
accuracy = accuracy_score(y_test,y_pred)*100
confusion_mat = confusion_matrix(y_test,y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Printing the results
print(f"Accuracy for SVM is: {accuracy:.4f}")
print("Confusion Matrix")
print(confusion_mat)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


# In[9]:


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score 

# Load data
data = pd.read_csv("twitter_training.csv")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2)

# Create a Gaussian Naive Bayes classifier
clf = GaussianNB()

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test,y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print the accuracy
print(f"Accuracy of the Naive Bayes classifier: {accuracy:.4f}")
print("Confusion Matrix")
print(confusion_mat)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


# In[10]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer

# Load the dataset
data = pd.read_csv('twitter_training.csv')

# Split the data into features and target
X = data.iloc[:, 2]  # Assuming the text data is in the 4th column (index 3)
y = data.iloc[:, 1]  # Assuming the sentiment is in the 3rd column (index 2)

# Convert X to strings
X = X.astype(str)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the Random Forest Classifier
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train_vec, y_train)

# Predict the sentiment on the test set
y_pred = rf_classifier.predict(X_test_vec)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test,y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("Confusion Matrix")
print(confusion_mat)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


# In[11]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer

# Load the dataset
data = pd.read_csv('twitter_training.csv')

# Split the data into features and target
X = data.iloc[:, 2]  # Assuming the text data is in the 4th column (index 3)
y = data.iloc[:, 1]  # Assuming the sentiment is in the 3rd column (index 2)

# Convert X to strings
X = X.astype(str)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Vectorize the text data
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the Neural Network Classifier
nn_classifier = MLPClassifier()
nn_classifier.fit(X_train_vec, y_train)

# Predict the sentiment on the test set
y_pred = nn_classifier.predict(X_test_vec)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test,y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("Confusion Matrix")
print(confusion_mat)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


# In[ ]:




