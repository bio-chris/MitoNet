"""

Reverse digits of an integer.

Example1: x = 123, return 321
Example2: x = -123, return -321

click to show spoilers.

Note:
The input is assumed to be a 32-bit signed integer. Your function should return 0 when the reversed integer overflows.

Maximum representable 32-bit value is 2^32 (4,294,967,295)
"""

"""
x = 10012

x = list(str(x))

#print(x)
#print(x[::-1])

minus = False
if "-" in x:
    del x[0]
    minus = True

#print(x[::-1])

n = int(''.join(x[::-1]))

if minus == True:
    n = n*-1

print(n)
"""

# Answer Accepted 


def reverse(x):

    x = list(str(x))

    minus = False
    if "-" in x:
        del x[0]
        minus = True

    n = int(''.join(x[::-1]))

    if minus == True:
        n = n * -1

    if n > 2147483647 or n < -(2147483647):
        return 0
    else:
        return n

print(reverse(-1563847412))




