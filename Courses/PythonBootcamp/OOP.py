

# Object Oriented Programming (OOP)


l = [1,2,3]

# calling methods on list l

l.count(1)


# function

def square(num):
    return num**2


# OBJECTS

"""

In Python EVERYTHING is an object (including a function) 

Many objects are already predefined, but you can also create your own objects

https://jeffknupp.com/blog/2014/06/18/improve-your-python-python-classes-and-object-oriented-programming/

"""

# Classes

# class names are by convention always capitalized // functions always lowercase
class Class(object):
    pass

x = Class()
#print(type(x))

"""

class attributes and methods

- an attribute, which is a characteristic of an object

self.attribute = something

- a method, which is an operation (or in other words a function within a class)

def __init__()
def something()

"""

class Calculator(object):

    # class object attribute
    isnumber = True

    # special method (this will initialize the attributes of the class)
    # all classes begin with the init (special) method
    def __init__(self, number):
        self.number = number

    def addition(self):
        return self.number*2

num = Calculator(number=2)

print(num.addition())



class Circle(object):

    # class object attributes (coa)
    pi = 3.14

    def __init__(self, radius=1):
        self.radius = radius

    def area(self):
        # radius**2 * pi
        return self.radius**2*Circle.pi

c = Circle(radius=50)

print(c.area())
















