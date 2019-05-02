"""

implementation of stack class (last in, first  out)


stack abstract data type

"""

class Stack:

    def __init__(self):

        # base of stack is empty list
        self.items = []

    def isEmpty(self):

        return self.items == []

    def push(self, item):

        self.items.append(item)

    def pop(self):

        return self.items.pop()

    def peek(self):

        return self.items[len(self.items)-1]

    def size(self):

        return len(self.items)


s = Stack()


""""

queues 

is an ordered collection of items (first in, first out)


implementation of a queue class 
"""

class Queue:

    def __init__(self):

        self.items = []

    def isEmpty(self):

        return self.items == []

    def enqueue(self, item):

        self.items.insert(0,item)

    def dequeue(self):

        return self.items.pop()

    def size(self):
        return len(self.items)


v = Queue()


"""

deque (linear data structure): double-ended queue 

unrestrictive nature of adding and removing items (items can be added and removed from both ends) 

"""

# see jupyter notebook for code


"""

interview questions

"""

# implement a stack

"""

Implement a Stack

A very common interview question is to begin by just implementing a Stack! Try your best to implement your own stack!

It should have the methods:

    Check if its empty
    Push a new item
    Pop an item
    Peek at the top item
    Return the size


"""

class Stack:

    def __init__(self):

        self.items = []

    def isempty(self):

        return self.items == []

    def push(self, item):

        self.items.append(item)

    def pop(self):

        self.items.pop()

    def peek(self):

        return self.items[len(self.items)-1]

    def size(self):

        return len(self.items)


# implement a queue class

class Queue:

    def __init__(self):

        self.items = []

    def isempty(self):

        return self.items == []

    def enqueue(self, item):

        self.items.insert(0,item)

    def dequeue(self):

        self.items.pop()

    def size(self):

        return len(self.items)




""""

Implement a Queue - Using Two Stacks

Given the Stack class below, implement a Queue class using two stacks! Note, this is a "classic" interview problem. 
Use a Python list data structure as your Stack.


"""

# Uses lists instead of your own Stack class.
stack1 = []
stack2 = []


class Queue2Stacks(object):

    def __init__(self):
        # Two Stacks
        self.stack1 = []
        self.stack2 = []

    def enqueue(self, element):
        # FILL OUT CODE HERE
        pass



    def dequeue(self):
        # FILL OUT CODE HERE
        pass




#RUN THIS CELL TO CHECK THAT YOUR SOLUTION OUTPUT MAKES SENSE AND BEHAVES AS A QUEUE
"""
q = Queue2Stacks()

for i in range(5):
    q.enqueue(i)

for i in range(5):
    print(q.dequeue())
    
"""

"""

Implement a Deque

Finally, implement a Deque class! It should be able to do the following:

    Check if its empty
    Add to both front and rear
    Remove from Front and Rear
    Check the Size

"""

class Deque:

    def __init__(self):

        self.items = []

    def isempty(self):

        return self.items == []

    def front(self, item):

        self.items.append(item)

    def rear(self, item):

        self.items.insert(0, item)

    def remfront(self):

        self.items.pop()

    def remrear(self):

        self.items.pop(0)

    def size(self):

        return len(self.items)


"""

Balanced Parentheses Check
Problem Statement

Given a string of opening and closing parentheses, check whether it’s balanced. 
We have 3 types of parentheses: round brackets: (), square brackets: [], and curly brackets: {}. 
Assume that the string doesn’t contain any other character than these, no spaces words or numbers. 
As a reminder, balanced parentheses require every opening parenthesis to be closed in the reverse order opened. 
For example ‘([])’ is balanced but ‘([)]’ is not.

You can assume the input string has no spaces.
Solution

Fill out your solution below:


"""

def balance_check(s):

    if len(s)%2 == 0:

        count=0
        for i in range(int(len(s) / 2)):

            if s[i] == "(":
                if s[-i - 1] == ")":
                    count += 1

            if s[i] == "[":
                if s[-i - 1] == "]":
                    count += 1

            if s[i] == "{":
                if s[-i - 1] == "}":
                    count += 1

        if count == int(len(s) / 2):
            return True

        else:
            return False

    else:

        return False

s = "([])([])"

#print(balance_check(s))

if len(s)%2 != 0:
    print(False)

count = 0
for i in range(int(len(s)/2)):


    if s[i] == "(":
        if s[-i - 1] == ")":
            count+=1

    if s[i] == "[":
        if s[-i - 1] == "]":
            count+=1

    if s[i] == "{":
        if s[-i - 1] == "}":
            count+=1


if count == int(len(s)/2):
    print(True)

s = "([])([])"

p_1 = []
p_2 = []

b_1 = []
b_2 = []

c_1 = []
c_2 = []

count=0
for i in range(int(len(s))):

    if s[i] == "(":
        p_1.append(1)

    elif s[i] == ")" and p_1 != []:
        count+=1
        p_1 = []

    elif s[i] == "(" and p_2 != []:
        count+=1
        p_2 = []

    elif s[i] == ")":
        p_2.append(1)



    if s[i] == "[":
        b_1.append(1)

    elif s[i] == "]" and b_1 != []:
        count += 1
        b_1 = []

    elif s[i] == "[" and b_2 != []:
        count += 1
        b_2 = []

    elif s[i] == "]":
        b_2.append(1)



    if s[i] == "{":
        c_1.append(1)

    elif s[i] == "}" and c_1 != []:
        count += 1
        c_1 = []

    elif s[i] == "{" and c_2 != []:
        count += 1
        c_2 = []

    elif s[i] == "}":
        c_2.append(1)









def balance_check(s):


    p_1 = []
    p_2 = []

    b_1 = []
    b_2 = []

    c_1 = []
    c_2 = []

    if len(s)%2 == 0:

        count = 0
        for i in range(int(len(s))):

            if s[i] == "(":
                p_1.append(1)

            elif s[i] == ")" and p_1 != []:
                count += 1
                #p_1 = []
                p_1.pop()

            elif s[i] == "(" and p_2 != []:
                count += 1
                #p_2 = []
                p_2.pop()

            elif s[i] == ")":
                p_2.append(1)



            if s[i] == "[":
                b_1.append(1)

            elif s[i] == "]" and b_1 != []:
                count += 1
                #b_1 = []
                b_1.pop()

            elif s[i] == "[" and b_2 != []:
                count += 1
                #b_2 = []
                b_2.pop()

            elif s[i] == "]":
                b_2.append(1)


            if s[i] == "{":
                c_1.append(1)

            elif s[i] == "}" and c_1 != []:
                count += 1
                #c_1 = []
                c_1.pop()

            elif s[i] == "{" and c_2 != []:
                count += 1
                #c_2 = []
                c_2.pop()

            elif s[i] == "}":
                c_2.append(1)


        if count == int(len(s)/2):
            return True

        else:
            return False

    else:
        return False


print(balance_check("[[[]])]"))





#RUN THIS CELL TO TEST YOUR SOLUTION

from nose.tools import assert_equal


class TestBalanceCheck(object):

    def test(self, sol):
        assert_equal(sol('[](){([[[]]])}('), False)
        assert_equal(sol('[{{{(())}}}]((()))'), True)
        assert_equal(sol('[[[]])]'), False)
        print('ALL TEST CASES PASSED')


# Run Tests

t = TestBalanceCheck()
t.test(balance_check)
