#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""
import os
import joblib
import sys
sys.path.append(os.path.abspath("../tools/"))
from feature_format import featureFormat, targetFeatureSplit

data_dict = joblib.load(open("../final_project/final_project_dataset.pkl", "rb") )


### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list, sort_keys = '../tools/python2_lesson14_keys.pkl')
labels, features = targetFeatureSplit(data)



### your code goes here 


from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.3, random_state=42)

from sklearn.metrics import *
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(features_train, labels_train)
# accuracy = clf.score(features_test, labels_test)
# print(accuracy)
# predict = clf.predict(features_test)

# print(classification_report(labels_test, predict))
# print(confusion_matrix(labels_test, predict))

# print(labels_test)
# print(predict)

# print(precision_score(labels_test,predict))
# print(recall_score(labels_test,predict))


predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

print(classification_report(true_labels, predictions))
print(confusion_matrix(true_labels, predictions))

print(true_labels)
print(predictions)

print(precision_score(true_labels,predictions))
print(recall_score(true_labels,predictions))