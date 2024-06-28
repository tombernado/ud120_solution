#!/usr/bin/python
import numpy as np
import random 
import math

### Tests
# predictions = [random.randint(100, 1000) for _ in range(90)]
# ages = [random.randint(30, 60) for _ in range(90)]
# net_worths = [random.randint(100, 1000) for _ in range(90)]

# print(predictions)
# print(ages)
# print(net_worths)

def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

   
    predictions_array = np.reshape(np.array(predictions), (len(predictions), 1))
    ages_array = np.reshape(np.array(ages), (len(ages), 1))
    net_worths_array = np.reshape(np.array(net_worths), (len(net_worths), 1))

    # print(predictions_array)
    # print(ages_array)
    # print(net_worths_array)

    cleaned_data = np.concatenate((ages_array, net_worths_array, (predictions_array-net_worths_array)**2), axis=1)
    # Sort by the first element of each sub-array
    cleaned_data = sorted(cleaned_data, key=lambda x: x[2])
    # Calculate the number of elements to keep (90% of the total elements)
    num_elements_to_keep = math.ceil(len(cleaned_data) * 0.9)
    # Prune the array by keeping only the first 90%
    cleaned_data = cleaned_data[:num_elements_to_keep]

    cleaned_data = [tuple(row) for row in cleaned_data]

    return cleaned_data
