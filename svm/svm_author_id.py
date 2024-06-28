#!/usr/bin/python3

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

# features_train = features_train[:int(len(features_train)/100)]
# labels_train = labels_train[:int(len(labels_train)/100)]

#########################################################
### your code goes here ###
def CVMAccuracy(features_train, labels_train, features_test, labels_test):
    from sklearn import svm
    from sklearn.metrics import accuracy_score

    clf = svm.SVC(kernel='rbf', C=10000)
    t0 = time()
    clf.fit(features_train, labels_train)
    print("Training Time:", round(time()-t0, 3), "s")
    t0 = time()
    predictions = clf.predict(features_test)
    print("Predicting Time:", round(time()-t0, 3), "s")
    accuracy = accuracy_score(labels_test, predictions)
    print(sum(predictions))
    return accuracy

def submitAccuracy():
    accuracy = CVMAccuracy(features_train, labels_train, features_test, labels_test)
    return accuracy

print("Accuracy:")
print(submitAccuracy())



#########################################################

#########################################################
'''
You'll be Provided similar code in the Quiz
But the Code provided in Quiz has an Indexing issue
The Code Below solves that issue, So use this one
'''

# features_train = features_train[:int(len(features_train)/100)]
# labels_train = labels_train[:int(len(labels_train)/100)]

#########################################################
