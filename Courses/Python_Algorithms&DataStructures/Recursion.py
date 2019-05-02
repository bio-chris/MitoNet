
# Overview

"""

two main instances of recursion

1) function makes one or more calls to itself

2) data structure uses smaller instances of exact same type of data structure


factorial function: n! (3! = 1*2*3)


"""

def factorial_func(n):

    if n == 0:
        return 1

    else:
        return n*factorial_func(n-1)


#print(factorial_func(1))


# homework section


"""

Problem 1

Write a recursive function which takes an integer and computes the cumulative sum of 0 to that integer

For example, if n=4 , return 4+3+2+1+0, which is 10.

This problem is very similar to the factorial problem presented during the introduction to recursion. Remember, 
always think of what the base case will look like. In this case, we have a base case of n =0 (Note, you could have also 
designed the cut off to be 1).

In this case, we have: n + (n-1) + (n-2) + .... + 0

Fill out a sample solution:

"""


def rec_sum(n):

    if n == 0:
        return 0

    else:
        return n+rec_sum(n-1)


#print(rec_sum(100))


"""

Problem 2

Given an integer, create a function which returns the sum of all the individual digits in that integer. For example:
if n = 4321, return 4+3+2+1

"""

import numpy as np

def sum_func(n):

    le = len(str(n))

    if n == 0:
        return 0

    else:

        return n%10 + sum_func(int(n/10))


n = 4321
#print(sum_func(n))


"""

Note, this is a more advanced problem than the previous two! It aso has a lot of variation possibilities and we're 
ignoring strict requirements here.

Create a function called word_split() which takes in a string phrase and a set list_of_words. The function will then 
determine if it is possible to split the string in a way in which words can be made from the list of words. 
You can assume the phrase will only contain words found in the dictionary if it is completely splittable.

For example:

word_split('themanran',['the','ran','man'])
["the", "man", "ran"]

"""

def word_split(phrase,list_of_words, output = None):

    if output is None:
        output = []

    for word in list_of_words:

        if phrase.startswith(word):

            output.append(word)

            return word_split (phrase[len(word):], list_of_words, output)

    return output



# Memoization


def memoize(f):
    memo = {}
    def helper(x):
        if x not in memo:
            memo[x] = f(x)
        return memo[x]
    return helper

#fib = memoize(fib)

# alternative

from functools import lru_cache

@lru_cache(maxsize=1000)
def fib(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fib(n-1) + fib(n-2)

"""
for i in range(1,50):
    print(fib(i))
"""

# reverse a string

"""

This interview question requires you to reverse a string using recursion. Make sure to think of the base case here.

Again, make sure you use recursion to accomplish this. Do not slice (e.g. string[::-1]) or use iteration, there must be
a recursive call for the function.

example

reverse('hello world')
'dlrow olleh'

"""


def reverse(s):

    if s == "":
        return ""

    else:
        return s[-1] + reverse(s[:-1])


# string permutation

"""

String Permutation
Problem Statement

Given a string, write a function that uses recursion to output a list of all the possible permutations of that string.

For example, given s='abc' the function should return ['abc', 'acb', 'bac', 'bca', 'cab', 'cba']

Note: If a character is repeated, treat each occurence as distinct, for example an input of 'xxx' would return a list 
with 6 "versions" of 'xxx'
Fill Out Your Solution Below

Let's think about what the steps we need to take here are:

example 

permute('abc')
['abc', 'acb', 'bac', 'bca', 'cab', 'cba']

"""

def permute(s):

    l = []

    if s == "":
        return []

    else:
        pass


s = "abc"

from itertools import permutations

for i in permutations(s):
    print(i)
