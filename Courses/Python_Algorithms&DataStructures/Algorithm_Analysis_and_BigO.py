def sum1(n):
    final_sum = 0

    for x in range(n+1):
        final_sum+=x

    return final_sum

#print(sum1(10))


def sum2(n):
    return (n*(n+1))/2

#print(sum2(10))

import timeit

# timeit module allows to measure time it takes to execute function
# maybe not the ideal solution !

#print(timeit.timeit("sum1(10)", setup="from __main__ import sum1"))
#print(timeit.timeit("sum2(10)", setup="from __main__ import sum2"))

# Big-O Notation


from math import log
import numpy as np
import matplotlib.pyplot as plt

"""
n = np.linspace(1, 10, 1000)

big_o = np.log(n)

plt.plot(n,big_o)
plt.show()
"""

# Examples of Big-O

# O(1) Constant

# this function is constant because regardless of the input, it will always only take
# one value

def func_constant(values):
    print(values[0])

#func_constant([1,2,3])

# O(n) Linear

def func_lin(lst):
    for val in lst:
        print(val)

lst = [1,2,3]

#func_lin(lst)

# O(n^2) Quadratic

def func_quad(lst):

    for item_1 in lst:
        for item_2 in lst:
            print(item_1, item_2)

#func_quad(lst)


def print_once(lst):

    for val in lst:
        print(val)


# is O(2n) but if n is infinite then 2 can be ignored so it is O(n)
def print_2(lst):

    for val in lst:
        print(val)

    for val in lst:
        print(val)

#print_2(lst)

def comp(lst):

    print(lst[0]) #O(1): Constant

    ### O(n/2)
    midpoint = int(len(lst)/2)

    for val in lst[:midpoint]:
        print(val)
    ###

    # O(10)
    for x in range(10):
        print("hello world")

lst = [1,2,3,4,5,6,7,8,9,10]
#comp(lst)

# How to estimate Big O of entire function
"""

O(1 + n/2 + 10) 

if n is infinite 1 and 10 can be ignored and the factor 1/2 also becomes 
irrelevant so simplified it can be called O(n) 

"""

def matcher (lst, match):

    for item in lst:
        if item == match:
            return True

    return False

# best case
#print(matcher(lst,1)) # O(1)

# worst case
#print(matcher(lst, 11)) # O(n)

# space complexity and time complexity

# Time complexity O(n)
# Space complexity O(1)
def create_list(n):

    new_list = []

    for num in range(n):
        new_list.append('new')

    return new_list

#print(create_list(5))

# Big O for Python Data Structures

# built-in functions often provide fastest solution to problem






