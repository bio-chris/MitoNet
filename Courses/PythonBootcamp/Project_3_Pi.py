# Project 3: Find PI to the Nth Digit

# Enter a number and have the program generate PI up to that many decimal places. Keep a limit to how far the program
# will go.

"""

Using the Bailey-Borwein-Plouffe formula 

"""
from decimal import *

def pi(i):

    pi_value = 0
    getcontext().prec = i

    for n in range(i+1):

        pi_value += Decimal((1/16**n)*((4/(8*n+1))-(2/(8*n+4))-(1/(8*n+5))-(1/(8*n+6))))

    return pi_value


#print(pi(1000))


"""

Need to determine accuracy of above function !!!

"""

