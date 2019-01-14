

import numpy as np

# Introduction to numpy
"""

my_mat = [[1,2,3],[4,5,6],[7,8,9]]

print(np.array(my_mat))

# creating a custom array
print(np.arange(0,10,3))

# creating a custom array of zero values

print(np.zeros((5,5)))

# creating a cutom array of one values

print(np.ones((3,4)))

# creates custom array of evenly spaced values

print(np.linspace(0,5,10))

# creates an identity matrix (useful when dealing with linear
# algebra problems)

print(np.eye(4))

# array of random values between 0 and 1 (uniform distribution)

print(np.random.rand(5,5))

# array of random values with normal distribution centered around 0

print(np.random.randn(4,4))

# array of random integers between specified values

print(np.random.randint(1,100,10))

# reshaping arrays

arr = np.arange(25)

ranarr = np.random.randint(0,50,10)

print(arr.reshape(5,5))

# max value in array
print(ranarr.max())
# index of max value in array
print(ranarr.argmax())

"""

# Numpy Indexing and Selection

"""

arr = np.arange(0,11)

print(arr[:5])
print(arr[5:])

# replaces the first five values in the array with a specified value (called broadcasting)
arr[0:5] = 100

# Code below also affects original arr variable instead of only changing the slice_of_arr variable
slice_of_arr = arr[0:6]
slice_of_arr[:] = 99

print(arr)

# to leave original variable unchanged, use copy method

arr_copy = arr.copy()
arr_copy[:] = 100

print(arr_copy, arr)



arr_2d = np.array([[5,10,15], [20,25,30], [35,40,45]])

print(arr_2d)

print(arr_2d[0][0])
# alternative way to access value in array
print(arr_2d[0,0])

# slicing the array

# grab rows 0 and 1 (not including 2) and grab columns 1 and 2 (not 0)
print(arr_2d[:2,1:])

# grab row 0,1 and 2 and grab column 2
print(arr_2d[:,2:])


arr = np.arange(1,11)

print(arr)

# generates array in which every element is checked against condition within brackets
print(arr>5)

bool_arr = arr > 5

# prints out the elements of the array for which the condition was True
print(arr[bool_arr])

"""

# NumPy Operations


import numpy as np

arr = np.arange(0,11)

# will perform an adding / subtraction / multiplication operation on every element
print(arr+arr)
print(arr-arr)
print(arr*arr)

# scalar addition / subtraction / multiplication / division (performs operation on every single element in array)
print(arr+100)

# division by 0 with numpy will not prevent all code from running


# Universal Array Function

# square root
print(np.sqrt(arr))

# exp
print(np.exp(arr))

#

































