
"""
### Python Crash Course Exercises


#
power = 7**4

print(power)


#
s = "Hi there Sam!"

print(s.split())

#
planet = "Earth"
diameter = 12742

print('The diameter of {} is {} kilometers'.format(planet, diameter))

#
lst = [1,2,[3,4],[5,[100,200,['hello']],23,11],1,7]

print(lst[3][1][2][0])


#
d = {'k1':[1,2,3,{'tricky':['oh','man','inception',{'target':[1,2,3,'hello']}]}]}

print(d['k1'][3]['tricky'][3]['target'][3])

#
def domainGet(string):
    return string.split("@")[1]

#
def findDog(string):
    if "dog" in string.lower():
        return True
    else:
        return False



#
def countDog(string):
    count=0
    for n in string.split():
        if n.lower() == "dog":
            count+=1
    return count


#
seq = ['soup','dog','salad','cat','great']

print(list(filter(lambda x: x.startswith("s"), seq)))


#
def caught_speeding(speed, is_birthday):

    if is_birthday == True:

        if speed <= 65:
            print("No Ticket")

        elif speed > 65 and speed <= 85:

            print("Small Ticket")

        else:
            print("Big Ticket")

    else:

        if speed <= 60:
            print("No Ticket")

        elif speed > 60 and speed <= 80:

            print("Small Ticket")

        else:
            print("Big Ticket")

"""

# NumPy Exercises

import numpy as np

# Array of 10 zeros
print(np.zeros(10))

# Array of 10 ones
print(np.ones(10))

# Array of 10 fives
np.ones(10) * 5

# Array of even integers from 10 to 50
print(np.arange(10,51))

# 3x3 matric with values ranging from 0 to 8
print(np.arange(9).reshape(3,3))

# 3x3 identity matrix

print(np.identity(3))

# create random number between 0 and 1

print(np.random.rand(1))

# array of 25 random numbers sampled from a standard normal distribution

print(np.random.randn(5,5))

# create the following matrix

"""

array([[ 0.01,  0.02,  0.03,  0.04,  0.05,  0.06,  0.07,  0.08,  0.09,  0.1 ],
       [ 0.11,  0.12,  0.13,  0.14,  0.15,  0.16,  0.17,  0.18,  0.19,  0.2 ],
       [ 0.21,  0.22,  0.23,  0.24,  0.25,  0.26,  0.27,  0.28,  0.29,  0.3 ],
       [ 0.31,  0.32,  0.33,  0.34,  0.35,  0.36,  0.37,  0.38,  0.39,  0.4 ],
       [ 0.41,  0.42,  0.43,  0.44,  0.45,  0.46,  0.47,  0.48,  0.49,  0.5 ],
       [ 0.51,  0.52,  0.53,  0.54,  0.55,  0.56,  0.57,  0.58,  0.59,  0.6 ],
       [ 0.61,  0.62,  0.63,  0.64,  0.65,  0.66,  0.67,  0.68,  0.69,  0.7 ],
       [ 0.71,  0.72,  0.73,  0.74,  0.75,  0.76,  0.77,  0.78,  0.79,  0.8 ],
       [ 0.81,  0.82,  0.83,  0.84,  0.85,  0.86,  0.87,  0.88,  0.89,  0.9 ],
       [ 0.91,  0.92,  0.93,  0.94,  0.95,  0.96,  0.97,  0.98,  0.99,  1.  ]])

"""

print(np.arange(1,101).reshape(10,10) / 100)


# create array of 20 linearly spaced points between 0 and 1

print(np.linspace(0,1,20))

# Now you will be given a few matrices, and be asked to replicate the resulting matrix outputs:

mat = np.arange(1,26).reshape(5,5)

"""

array([[12, 13, 14, 15],
       [17, 18, 19, 20],
       [22, 23, 24, 25]])

"""

print(mat[2:,1:])

"""

20

"""

print(mat[-2,-1])

"""

array([[ 2],
       [ 7],
       [12]])

"""

print(mat[:3,1:2])

"""

array([21, 22, 23, 24, 25])

"""

print(mat[4:,:])

# sum of all values in mat

print(np.sum(mat))

# standard deviation of values in mat

print(np.std(mat))

# get sum of all columns in mat


