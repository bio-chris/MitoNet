

"""

All Code below is from:

- ProjectEuler.net
- CodingBat.com
- Python Bootcamp (Udemy course)

"""

'''

def pos_neg(a,b,negative):
    if negative is False:
        if (a<0 and b>0) or (a>0 and b<0):
            return True
        else:
            return False

    else:
        if a<0 and b<0:
            return True
        else:
            return False

print(pos_neg(-4, -5, True))

a = "not my cake"

print(a.startswith("not"))


def front_back(str):
  middle = str[1:(len(str)-1)]
  front = str[len(str)-1]
  back = str[0]
  return front + middle + back

print(front_back("Change"))

def front3(str):
  if len(str) >= 3:
    front = str[:3]
    return front*3
  else:
    front = str[:len(str)]
    return front*3


print(front3("Canada"))

def string_times(str,n):
    if n > 0:
        return n*str

print(string_times("Hi",10))

def front_times(str, n):
  if len(str) >= 3:
    front = str[:3]
    return n*front
  else:
    front = str[:len(str)]
    return n*front

print(front_times("Chocolate",3))

def string_bits(str):
    bits = str[::2]
    return bits

print(string_bits("Heeololeo"))


def string_splosion(str):
    j = 0
    splosion = ""
    for i in str:
        splosion+=str[:j]
        j+=1
    splosion+=str
    return splosion

print(string_splosion("abc"))

def array_count9(nums):
    count9 = 0
    for i in nums:
        if i == 9:
            count9+=1
    return count9

print(array_count9([9,9,9,9,9]))

def array_front9(nums):
    if len(nums) >= 4:
        for i in nums[:4]:
            if i == 9:
                return True
                break
            else:
                return False

    else:
        for i in nums:
            if i == 9:
                return True
                break
            else:
                return False

#print(array_front9([1, 2, 9, 3, 4]))


list = [1,3,3,4,5,6]

if len(list)>=4:
    for i in list[:4]:
        if i == 9:
            print(True)
            break

    else:
        print(False)



list = [2,2,3,1,2,1,2,3,5,6]

def array123(nums):
    j=0
    for i in nums:
        if len(nums[j:j+3]) == 3:
            if nums[j:j+3] == [1,2,3]:
                return True
            j+=1
    else:
        return False


print(array123([1, 1, 2, 1, 2, 3]))


j = 0
for i in list:
    if len(list[j:j+3]) == 3:
        print(list[j:j+3])
        j+=1


# Project Euler (Problem 1)
# Find the sum of all the multiples of 3 or 5 below 1000

i=0
sum=0
while i < 1000:
    if i%3 ==0 or i%5 ==0:
        sum+=i
    i += 1
print(sum)

# Project Euler (Problem 2)
# Fibonacci numbers (1,2,3,5,8,13 ...)
# By considering the terms in the Fibonacci
# sequence whose values do not exceed four
# million, find the sum of the even-valued terms.

i0=0
i1=1
sum=0

while i<4*10**6:

    i=i0+i1
    i0=i1
    i1=i

    if i <4*10**6:
        #print(i)
        if i%2 == 0:
            sum+=i

print(sum)

# Project Euler (Problem 3)
# What is the largest prime factor of the number
# 600851475143




def is_prime(number):
    new_number=1
    while i<=number:
        new_number=number%i
        if not (i == 1 or i == number):
            if new_number == 0:
                return False
            else:
                return True



i = 1
number = 9
while i<=number:
    print(i)
    new_number=number%i
    if not (i == 1 or i == number):
        if new_number == 0:
            print("Not Prime")
        else:
            print("Is Prime")
    i+=1



# Code below checks if number is prime


#number = 54499

def is_prime(number):
    i = 2
    while i < number:
        new_number = number % i
        if new_number == 0:
            return(False)
            break
        i+= 1
    else:
        return(True)




# List prime numbers until specified value

value = 100

i = 1
j=2
not_prime = None

while i < value:
    #print(i)
    #print(not_prime)
    while j < i:
        new_number = i % j
        if new_number == 0:
            not_prime = True
            #break
        j += 1
    if not_prime == None:
        if i != 1:
            print(i)

    i+=1
    j=2

    not_prime = None



# Largest prime factor

value = 600851475

i = 1
j=2
not_prime = None

while i < value:
    #print(i)
    #print(not_prime)
    while j < i:
        new_number = i % j
        if new_number == 0:
            not_prime = True
            #break
        j += 1
    if not_prime == None:
        if i != 1:
            print(i)
            print(value)
            value = value/i
            #print(new_number2)


            if is_prime(new_number2) == False:
                print(i)
                #break



    i+=1
    j=2

    not_prime = None




#

number = 600851475
i = 1

while i <= number:
    new_number = number % i
    if not (i == 1 or i == number):
        if new_number == 0:
            print(i)
        if new_number != 0:
            None
            #print(i)

    i += 1



i = 1
j = 1

while i <= number:
    new_number = number % i
    if not (i == 1 or i == number):
        if new_number == 0:
            #None
            print(i)



            number2 = number/i
            while j<=number2:
                new_number2 = number%j
                if not (i == 1 or i == number):
                    if (new_number2 == 0) and (new_number2*i == number):
                        print(i)
                        break
                    if new_number2 != 0:
                        number2 = number % i
                j+=1

        if new_number != 0:
            None

            #print(i)

    i += 1




# Looking for largest factor first

i = number
while i >= 0:
    new_number = number % i
    if not (i == 1 or i == number):
        if new_number == 0:
            print(i)

        if new_number != 0:
            None
            #print(i)

    i -= 1






n = 600851475
i = 2
while i * i < n:
    while n % i == 0:
        n = n / i
        print(n)
        print(i)
    i = i + 1
    print(i)

#print(n)


'''

def is_prime(number):
    i = 2
    while i < number:
        new_number = number % i
        if new_number == 0:
            return(False)
            break
        i+= 1

    else:
        return(True)

#print(is_prime(54499))

'''
value = 54499

i = 1
j=2
not_prime = None

while i < value:
    #print(i)
    #print(not_prime)
    while j < i:
        new_number = i % j
        if new_number == 0:
            not_prime = True
            #break
        j += 1
    if not_prime == None:
        if i != 1:
            print(i)


    i+=1
    j=2

    not_prime = None




def is_palindrome(n):
    if int(str(n)[::-1]) == n:
        return True
    else:
        return False

#print(is_palindrome(5005))

# Euler Project 4 (Finding the largest palindrom of
# a product of two three digit numbers)

n = 1000

i = n
j = n
list = []

while i > 0:
    while j > 0:
        if is_palindrome(i*j) == True:
            list.append(i*j)
        j-=1
    i-=1
    j=n

print(max(list))




square = lambda num: num**2

print(square(10))

x = 50

def func():
    global x

    x = 23

func()
print(x)

import math

def vol(rad):
    V = (4/3)*math.pi*rad**3
    return("%.3f" % V)


def ran_check(num,low,high):
    if num in range(low,high):
        return True
    else:
        return False


def up_low(s):
    count_lower = 0
    count_upper = 0
    for letters in s:
        if letters.isupper() == True:
            count_upper+=1
        else:
            count_lower+=1

    return("Upper case characters: ", count_upper)
    return("Lower case characters: ", count_lower)


def unique_list(l):
    return set(l)

def multiply(numbers):
    result = 1
    for i in numbers:
        result*= i
    return result

def palindrome(s):
    if s == s[::-1]:
        return True
    else:
        return False

import string

def ispangram(str1, alphabet=string.ascii_lowercase):

    new_str1 = []
    for i in sorted(str1.lower()):
        if i != ' ':
            new_str1.append(i)

    if (''.join(sorted(set(new_str1)))) == alphabet:
        return True
    else:
        return False



# Object oriented programming

# Class
class Test(object):
    pass

x = Test()
print (type(x))

# Attributes: self.attribute = something

class Dog(object):

    # Class Object Attribute
    species = "mammal"

    # special method __init__()
    def __init__(self, breed, name):
        self.breed = breed
        self.name = name

sam = Dog("Lab", "Sammie")

print(sam.name)
print(sam.species)

class Sphere(object):
    pi = 3.14

    # Methods
    def __init__(self, radius=10):
        self.radius = radius

    def volume(self):
        return self.radius**3*Sphere.pi*(4/3)

vol = Sphere()

print(vol.volume())

# Inheritance

class Animal(object):
    def __init__(self):
        print("Animal created")

    def whoAmI(self):
        print("Animal")

    def eat(self):
        print("Eating")

# instead of Dog(object) we use Dog(Animal) for inheritance
class Dog(Animal):
    def __init__(self):
        Animal.__init__(self)
        print("Dog created")


print(Dog())



# Objects homework Problem 1
import math

class Line(object):

    def __init__(self, coor1, coor2):

        self.coor1 = coor1
        self.coor2 = coor2


    def distance(self):

        return math.sqrt((self.coor2[0]-self.coor1[0])**2+(self.coor2[1]-self.coor1[1])**2)


    def slope(self):

        return (self.coor2[1]-self.coor1[1])/(self.coor2[0]-self.coor1[0])


coordinate1=(3,2)
coordinate2=(8,10)

li = Line(coordinate1, coordinate2)

#print(li.distance())

# Objects homework Problem 2
class Cylinder(object):

    pi = math.pi

    def __init__(self, height, radius):
        self.height = height
        self.radius = radius

    def volume(self):

        return Cylinder.pi*self.height*self.radius**2


    def surface_area(self):

        return 2*Cylinder.pi*self.radius*(self.height+self.radius)

c = Cylinder(2,3)

print(c.volume())
print(c.surface_area())



# Errors and Exceptions Homework

# Problem 1
for i in ['a', 'b', 'c']:
    try:
        print (i**2)
    except TypeError:
        print("Math operations can't be performed on strings!")
        break

# Problem 2
x = 5
y = 0

try:
    z = x/y

except ZeroDivisionError:

    print("Can't divide by 0!")

# Problem 3
def ask():

    while True:
        try:
            number = int(input("Enter an integer: "))
            print(number**2)
            break
        except:
            print("Entered value is not an integer!")
            continue

ask()





class Circle(object):
    pi = 3.14

    # Circle get instantiated with a radius (default is 1)
    def __init__(self, radius=1):
        self.radius = radius

    # Area method calculates the area. Note the use of self.
    def area(self):
        return self.radius * self.radius * Circle.pi

    # Method for resetting Radius
    def setRadius(self, radius):
        self.radius = radius

    # Method for getting radius (Same as just calling .radius)
    def getRadius(self):
        return self.radius


c = Circle()

c.setRadius(2)
print ('Radius is: ',c.getRadius())
print ('Area is: ',c.area())


class Lightspeed(object):

    c = 300000

    def __init__(self, distance):

        self.distance = distance

    def time(self):

        print((self.distance/Lightspeed.c)/60)

    def get_distance(self):
        return self.distance


lspeed = Lightspeed(150000000)

lspeed.time()


'''

"""
def fahrenheit(T):
    return( (9.0/5)*T+32 )


temp = [1,2,3]

print(list(map(fahrenheit, temp)))

print(map(lambda T: (9.0/5)*T+32, temp))



lst = [34, 23, 24, 24, 100, 2333, 2, 11]

max_find = lambda a,b: a if (a>b) else b

print(max_find(100,1000))

from functools import reduce

print(reduce(max_find, lst))



even_check = lambda num: True if num%2 == 0 else False


print(list(filter(even_check, [1,2,3,4,5])))



list1 = [1,2,3]
list2 = [4,5,6]

for (i, count) in enumerate(list1):
    print(i, count)



"""


# Advanced Functions Test - Write all functions in one line

# Problem 1:
# Use map to create a function which finds the length of each word in the phrase (broken by spaces)
# and return the values in a list.

"""
# Function without map or .split()

def word_lengths(phrase):
    list = []
    i = 0
    for count, letter in enumerate(phrase):

        if letter != " " and count + 1 != len(phrase):
            i += 1

        elif letter == " ":
            list.append(i)
            i = 0

        elif count + 1 == len(phrase):
            i += 1
            list.append(i)

    return list



# Function using map and .split()

def word_lengths(phrase):
    return list(map(len, phrase.split()))

# Lambda function using map and .split()

lenghts = lambda phrase: list(map(len, phrase.split()))



# Problem 2:
# Use reduce to take a list of digits and return the number that they correspond to.
# Do not convert the integers to strings!

from functools import reduce

def digits_to_num(digits):
    return reduce(lambda x,y: x*10 + y,digits)
    pass

print(digits_to_num([1,2,3]))

# My solution (but conversion to string not permitted)

digits_to_num = reduce((lambda x,y : str(x)+str(y)), [1,2,3,4])

print(digits_to_num)



# Problem 3
# Use filter to return the words from a list of words which start with a target letter.

def filter_words(word_list, letter):
    return list(filter(lambda word: word.startswith(letter), word_list))
    pass

l = ['hello','are','cat','dog','ham','hi','go','to','heart']

print(filter_words(l, 'g'))



# Problem 4
# Use zip and list comprehension to return a list of the same length where each value is the two strings from L1 and L2
# concatenated together with connector between them.


def concatenate(L1, L2, connector):
    return [x+connector+y for x, y in zip(L1, L2)]

print(concatenate(['A','B', 'C'],['a','b','c'],'-'))



# Problem 5
# Use enumerate and other skills to return a dictionary which has the values of the list as keys and the index
# as the value. You may assume that a value will only appear once in the given list.

from collections import OrderedDict

def d_list(L):
    return {key: count for count,key in enumerate(L)}

print(d_list(['a','b','c']))


# Problem 6
# Use enumerate and other skills from above to return the count of the number of items in the list whose value
# equals its index.

def count_match_index(L):
    return len([x for count,x in enumerate(L) if count == x])

print(count_match_index([0,2,2,1,5,5,6,10]))





s = "Test"

def func():
    print(locals())

func()

def hello(name="Chris"):
    print("Hello " +name)

hello()

greet = hello

greet()

def name(my_name="Chris"):
    print("Something")

    def new_greet():
        print("Something new")

    def welcome():
        print("This is welcome")

    print(greet())
    print(welcome())
    print("Inside name function")

name()

def hello():
    return("Hi Chris")

def other(func):
    print("Other code")
    print(func())


print(other(hello))


def new_decorator(func):
    def wrap_func():
        print("Some Code for func")

        func()

        print("Some Code after func")

    return wrap_func()

def func_needs_decorator():
    print("This needs a decorator")

func_needs_decorator = new_decorator(func_needs_decorator)

print(func_needs_decorator)


@new_decorator # same as func_needs_decorator = new_decorator(func_needs_decorator)
def func_needs_decorator():
    print("Something")



def gencubes(n):
    for num in range(n):
        yield num**3


for x in gencubes(10):
    print(x)


def simple_gen():
    for x in range(3):
        yield x



g = simple_gen()

print(next(g))
print(next(g))

s = "hello"

s_iter = iter(s)

print(next(s_iter))
print(next(s_iter))



# Iterators and Generators Homework

# Problem 1
# Create a generator that generates the squares of numbers up to some number N

def gensquares(N):
    for num in range(N):
        yield num**2


for x in gensquares(10):
    print(x)

# Problem 2
# Create a generator that yields "n" random numbers between a low and high number (that are inputs).


import random

def rand_num(low, high, n):
    for number in range(n):
        yield random.randint(low,high)

for num in rand_num(1,10,12):
    print(num)

# Problem 3
# Use the iter() function to convert the string below

s = 'hello'

s_iter = iter(s)

print(s_iter)



# Counter

from collections import Counter

l = [1,1,2,3,4]

print(Counter(l))

s = "abcdefga"

print(Counter(s))

s = "How many times does each word shop up in this show sentence word"

word = s.split()

print(Counter(word))

c = Counter(word)

print(c.most_common(3))



# defaultdict
# defaultdict can be used to avoid a key error when trying to open non-existing key in dictionary

from collections import defaultdict

d = {'k1':1}

d = defaultdict(object)



# OrderedDict

d = {}

d['a'] = 1
d['b'] = 2
d['c'] = 3
d['d'] = 4
d['e'] = 5

for k,v in d.items():
    print(k,v)


# namedtuple

t = (1,2,3)

print(t[0])

from  collections import namedtuple

#similar to creating a short class 
Dog = namedtuple('Dog', 'age breed name')

sam = Dog(age=2, breed="Lab", name="Sammy")

print(sam)



# Datetime

import datetime

t = datetime.time(5,25,1)

print(t)

print(t.hour)

today = datetime.date.today()

print(today)
print(today.timetuple())



# Python Debugger

import pdb

x = [1,3,4]
y = 2
z = 3

result = y+z
print(result)

pdb.set_trace()

result2 = y+x
print(result2)


# Timing your code

import timeit

print(timeit.timeit("-".join(str(n) for n in range(100)), number=10000))

print(timeit.timeit('"-".join(map(str, range(100)))',number=10000))

print('"-".join(map(str, range(100))')


# Regular Expressions

import re

patterns = ["term1", "term2"]

text = "This is a string with term1 and the other term"



import re

split_term = "@"

phrase = "c.fischer.1991@gmx.de"

# most efficient way to get a specific string after a certain element within that string
print(re.split(split_term, phrase))

test = "This is a test phrase"

# finds all instances that match to 'This' in string test
print(re.findall('This', test))


# code below finds specific patterns within a string

def multi_re_find(patterns,phrase):
    '''
    Takes in a list of regex patterns
    Prints a list of all matches
    '''
    for pattern in patterns:
        print ('Searching the phrase using the re check: %r' %pattern)
        print (re.findall(pattern,phrase))
        print ('\n')



test_phrase = 'sdsd..sssddd...sdddsddd...dsds...dsssss...sdddd'

test_patterns = [ 'sd*',     # s followed by zero or more d's
                'sd+',          # s followed by one or more d's
                'sd?',          # s followed by zero or one d's
                'sd{3}',        # s followed by three d's
                'sd{2,3}',      # s followed by two to three d's
                ]

multi_re_find(test_patterns,test_phrase)

# Exclusion

test_phrase = "This is a funny string."

# [^] notation removes specific elements within the string
print(re.findall('[^as]+', test_phrase))
print('\n')

# Character ranges

test_phrase = 'This is an example sentence. Lets see if we can find some letters.'

test_patterns = [ r'\d+',  # sequences of digits
                 '[A-Z]+',  # sequences of upper case letters
                 '[a-zA-Z]+',  # sequences of lower or upper case letters
                 '[A-Z][a-z]+']  # one upper case letter followed by lower case letters

multi_re_find(test_patterns, test_phrase)



# StringIO module

# Advanced Numbers

# Hexadecimal numbers
print(hex(100))

# Binary numbers
print(bin(20))

# to the power

print(2**4)
# different way of expressing to the power
print(pow(2,4))

# absolute value

print(abs(-100))

# rounding

print(round(3.4999,2))


# Advanced Strings

s = 'Hello World'

# printing first element as upper case
print(s.capitalize())
# printing all elements as upper case
print(s.upper())

# all elements as lower case
print(s.lower())

# count number of elements
print(s.count('o'))

# find index of elements
print(s.find('o'))

# center method
# placing string between specified number of elements (20 is the total length of the string)
print(s.center(20,'z'))

# isalnum method, check if all elements are alphanumeric

print(s.isalnum())

# check if all elements are lower or upper case

print(s.islower())
print(s.isupper())

# check if string is only space

print(s.isspace())

# title. returns true if s is a title case string (all first letters of all words have to be upper case to return true)

print(s.istitle())

# split method

print(s.split('e'))

# partition

print(s.partition('e'))


# Advanced Sets

s=set()

s.add(1)
s.add(2)

print(s)

s.clear()

print(s)

s = {1,2,3}

print(type(s))

sc = s.copy()

print(sc)

# checking difference between two sets

s.add(4)
print(s.difference(sc))



d = {'k1':1,'k2':2}


# dictionary comprehesion

print({x:x**2 for x in range(10)})

print({k:v**2 for k,v in zip(['a','b'],range(10))})

# iteritems

for k in d.items():
    print(k)



# Advanced Python Object Test

# Convert 1024 to binary and hexadecimal representation

print(bin(1024), hex(1024))

#  Round 5.23222 to two decimal places

print(round(5.23222,2))

# Check if every letter in the string s is lower case

s = 'hello how are you Mary, are you feeling okay?'

print(s.islower())

#Problem 4: How many times does the letter 'w' show up in the string below?

s = 'twywywtwywbwhsjhwuwshshwuwwwjdjdid'

print(s.count('w'))


#Problem 5: Find the elements in set1 that are not in set2:

set1 = {2,3,1,5,6,8}
set2 = {3,1,7,5,6,8}

print(set1.difference(set2))

# Problem 6: Find all elements that are in either set:

print(set1.union(set2))


#Problem 7: Create this dictionary: {0: 0, 1: 1, 2: 8, 3: 27, 4: 64} using dictionary comprehension.

print({x:x**3 for x in range(5)})

# Problem 8: Reverse the list below:

l = [1,2,3,4]
l.reverse()
print(l)

# Problem 9: Sort the list below

l = [3,4,2,5,1]
l.sort()
print(l)

"""











