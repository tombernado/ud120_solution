def featureScaling(arr):
    import numpy as np
    array = np.array(arr)
    print("Original array:", array)
    
    x_max = np.max(array)
    x_min = np.min(array)
    print("Max value:", x_max)
    print("Min value:", x_min)
    
    rescaled_array = (array - x_min) / (x_max - x_min)
    print("Rescaled array:", rescaled_array)
    
    return rescaled_array

# Tests of your feature scaler--line below is input data
data = [115, 140, 175]
print("Rescaled Data:", featureScaling(data))