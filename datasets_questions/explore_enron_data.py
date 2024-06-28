#!/usr/bin/python3

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import joblib

enron_data = joblib.load(open("../final_project/final_project_dataset.pkl", "rb"))

# Get the first key and its corresponding value
first_key = list(enron_data.keys())[0]
first_item = enron_data[first_key]
first_value = list(enron_data.values())[0]

# Print the first key and its corresponding value
# print("First key:", first_key)
# print("First item:", first_item)
# print("Number of features:", len(first_item))
# print("First Value:", first_value)
# Initialize the counter
# poi_count = 0

# Iterate through the dictionary and count entries where "poi" is 1
# for key, value in enron_data.items():
#     if value.get("poi") == 1:
#         poi_count += 1

# Print the number of POIs
# print("Number of POIs:", poi_count)
# print(enron_data.items()["JAMES PRENTICE"]["total_stock_value"])

# total_stock_value_skilling = enron_data['SKILLING JEFFREY K']['total_payments']
# print(f"exercised_stock_option Skilling {total_stock_value_skilling}")
# total_stock_value_lay = enron_data['LAY KENNETH L']['total_payments']
# print(f"exercised_stock_option Lay {total_stock_value_lay}")
# total_stock_value_fastow = enron_data['FASTOW ANDREW S']['total_payments']
# print(f"exercised_stock_option Fastow {total_stock_value_fastow}")

# Specify the person name
# person_name = 'SKILLING JEFFREY K'

# Access the "total_stock_value" of the specified person
# if person_name in enron_data:
#     exercised_stock_options = enron_data[person_name].get('exercised_stock_options')
#     print(f"exercised_stock_options {person_name}: {exercised_stock_options}")
# else:
#     print(f"{person_name} not found in the dataset.")

import math

# Initialize the counter
not_nan_count = 0

# Loop through dictionary values and count those that are not NaN
# for key, value in enron_data.items():
#     if value.get("salary") != "NaN":
#         not_nan_count += 1

# print("Number of non-NaN salaries:", not_nan_count)

# Initialize the counter
not_nan_count1 = 0

# Loop through dictionary values and count those that are not NaN
for key, value in enron_data.items():
    if value.get("email_address") != "NaN":
        not_nan_count1 += 1

print("Number of non-NaN email:", not_nan_count1)

nan_count1 = 0

# Loop through dictionary values and count those that are not NaN
# for key, value in enron_data.items():
#     if value.get("total_payments") == "NaN" and value.get("poi") == 1:
#         nan_count1 += 1

# print("Number of NaN total payment:", nan_count1)

poi_count = 0

# Loop through dictionary values and count those that are not NaN
for key, value in enron_data.items():
    if value.get("poi") == 1:
        poi_count += 1

print("Number of NaN total payment:", poi_count)
