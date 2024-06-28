#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess

features_train, features_test, labels_train, labels_test = preprocess()

def DTAccuracy(features_train, labels_train, features_test, labels_test):
    from sklearn import tree
    from sklearn.metrics import accuracy_score

    clf = tree.DecisionTreeClassifier(min_samples_split=40)
    t0 = time()
    clf.fit(features_train, labels_train)
    print("Training Time:", round(time()-t0, 3), "s")
    t0 = time()
    predictions = clf.predict(features_test)
    print("Predicting Time:", round(time()-t0, 3), "s")
    accuracy = accuracy_score(labels_test, predictions)
    return accuracy

accuracy = DTAccuracy(features_train, labels_train, features_test, labels_test)

print(f"Accuracy: {round(accuracy, 3)}")

print(len(features_train[0]))


