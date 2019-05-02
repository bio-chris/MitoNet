# Fibonacci Sequence - Enter a number and have the program generate the Fibonacci
# sequence to that number or to the Nth number.


def fibonacci(n):
    n0 = 1
    n1 = 1

    for i in range(n):

        n1_2 = n0+n1

        n0 = n1
        n1 = n1_2

        print(n1_2)


fibonacci(100)



