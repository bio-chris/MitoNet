"""

Determine whether an integer is a palindrome. Do this without extra space.

"""

x = -1


new_x = list(str(x))

if "-" in new_x:
    print(False)

else:

    n = int(''.join(new_x[::-1]))

    # Section for 32-bit integers to prevent overflowing when reversing the integer
    """
    if n > 2147483647 or n < -(2147483647):
        print(0)
    else:  
        print(n)
    """

    if x == n:
        print(True)

    else:
        print(False)


# Answer Accepted 

def isPalindrome(x):
    new_x = list(str(x))

    if "-" in new_x:
        return False

    else:
        n = int(''.join(new_x[::-1]))

        if x == n:
            return True

        else:
            return False

print(isPalindrome(-2147447412))


