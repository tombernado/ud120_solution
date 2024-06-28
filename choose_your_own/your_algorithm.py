#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from time import time

features_train, labels_train, features_test, labels_test = makeTerrainData()

# features_train = features_train[:int(len(features_train)/100)]
# labels_train = labels_train[:int(len(labels_train)/100)]


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################

def AlgorithmAccuracy(features_train, labels_train, features_test, labels_test):
    # from sklearn.ensemble import AdaBoostClassifier
    # from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    # from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score

    # clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=50, algorithm="SAMME.R", random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    # clf = KNeighborsClassifier(n_neighbors=5)
    t0 = time()
    clf.fit(features_train, labels_train)
    print("Training Time:", round(time()-t0, 3), "s")
    t0 = time()
    predictions = clf.predict(features_test)
    print("Predicting Time:", round(time()-t0, 3), "s")
    accuracy = accuracy_score(labels_test, predictions)
    return accuracy, clf

accuracy, clf = AlgorithmAccuracy(features_train, labels_train, features_test, labels_test)

print(f"Accuracy: {round(accuracy, 3)}")

try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
