#!/usr/bin/python3
import os
import joblib
import sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.append(os.path.abspath("../tools/"))
enron_data = joblib.load(open("../final_project/final_project_dataset.pkl", "rb"))
from feature_format import featureFormat, targetFeatureSplit

person = list(enron_data.keys())

selected_values =[]

for person, features in enron_data.items():
    selected_person_values = {key: features[key] for key in ['exercised_stock_options', 'bonus'] if key in features}
    selected_person_values['person'] = person
    selected_values.append(selected_person_values)

salaries = []
bonuses = []
persons = []

# Extract the values for 'salary', 'bonus', and 'person'
for person, features in enron_data.items():
    if 'exercised_stock_options' in features:
        salaries.append(features['exercised_stock_options'])
    if 'bonus' in features:
        bonuses.append(features['bonus'])
    persons.append(person)

# Convert lists to NumPy arrays
salaries_array = np.array(salaries).reshape(-1, 1)
bonuses_array = np.array(bonuses).reshape(-1, 1)
persons_array = np.array(persons).reshape(-1, 1)

# Concatenate the arrays
concatenated_array = np.hstack((salaries_array, bonuses_array, persons_array))

sorted_indices = np.argsort(concatenated_array[:, 0].astype(float))  # Ensure salary is numeric
sorted_array = concatenated_array[sorted_indices]

print("Sorted array by salary:")
print(sorted_array)


### read in data dictionary, convert to numpy array
data_dict = joblib.load( open("../final_project/final_project_dataset.pkl", "rb") )
features = ["salary", "bonus"]
data_dict.pop( 'TOTAL', 0 )
data = featureFormat(data_dict, features)


### your code below
plt.scatter(data[:, 0],data[:, 1])
plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()

