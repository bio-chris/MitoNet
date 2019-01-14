

# Singly Linked List

# Definition: collection of nodes that form a linear sequence

"""

the list instance maintains a member named head that identifies first node of list (tail: last node of list)

going through linked list is called traversing linked list (also known as link or pointer hopping)

each node is a unique object with that instance storing a reference to its element and a reference to the next
node (or None for last node)

in a singly linked list only the head node can be removed

for removal of tail node, a doubly linked list is needed

"""

# singly linked list implementation

class Node:

    def __init__(self, value):

        self.value = value
        self.nextnode = None


"""
# giving each node a value
a = Node(1)
b = Node(2)
c = Node(3)

# setting links between nodes
a.nextnode = b
b.nextnode = c
c.nextnode = a

#print(a.value)
"""


# Doubly linked lists

# definition: doubly linked lists have nodes that keep reference to the node before and after it

"""

allow greater variety O(1) time update operations (including deletions and insertions) 

prev: previous node 

"""

# doubly linked list implementation
"""
class DoublyLinkedListNode:

    def __init__(self, value):

        self.value = value
        self.nextnode = None
        self.prevnode = None


link = DoublyLinkedListNode

a = link(1)
b = link(2)
c = link(3)


a.nextnode = b
b.prevnode = a

b.nextnode = c
c.prevnode = b
"""

# Interview Problems Section

# singly linked list cycle check (a cycle is when a nodes next point points back to a previous node in the list)
# write function that return a boolean indicating if linked list contains a cycle


"""
l = []

# setting list to first node
list = a

# traversing a list
while list != None:

    n = list.nextnode

    print(n, list)

    if n in l:
        pass

    l.append(n.nextnode)

    list = list.nextnode


print(l)
"""



# solution works
def cycle_check_1(node):

    l = []
    list = node

    while list:
        if list in l:
            return True
            break

        l.append(list)
        list = list.nextnode

    else:
        return False


"""
def cycle_check(node):
    # Begin both markers at the first node
    marker1 = node
    marker2 = node

    # Go until end of list
    while marker2 != None and marker2.nextnode != None:

        # Note
        marker1 = marker1.nextnode
        marker2 = marker2.nextnode.nextnode

        # Check if the markers have matched
        if marker2 == marker1:
            return True

    # Case where marker ahead reaches the end of the list
    return False

"""

"""
a = Node(1)
b = Node(3)
c = Node(1)
d = Node(4)
e = Node(1)

# setting links between nodes
a.nextnode = b
b.nextnode = c
c.nextnode = d
d.nextnode = e
e.nextnode = None
"""


#print(cycle_check_1(a))

# linked list reversal

"""

write function to reverse a linked list in place. function will take in head of list and return new head of list

"""

# Create a list of 4 nodes
a = Node(1)
b = Node(2)
c = Node(3)
d = Node(4)

# Set up order a,b,c,d with values 1,2,3,4
a.nextnode = b
b.nextnode = c
c.nextnode = d

"""
l = []

while l_list:
    print(l_list.value)
    l.append(l_list.value)
    l_list = l_list.nextnode

l = (list(reversed(l)))

l_list = a
count = 0
while l_list:
    l_list.value = l[count]
    l_list = l_list.nextnode
    count+=1
"""

# solution works (but two while loops, so O(2*n))
def reverse(head):

    l_list = head
    l = []

    while l_list:
        l.append(l_list.value)
        l_list = l_list.nextnode

    l = (list(reversed(l)))

    l_list = head
    count = 0
    while l_list:
        l_list.value = l[count]
        l_list = l_list.nextnode
        count += 1

    return head

#reverse(a)

#nextnode = None
previous = None

#print(a.nextnode.value)

# solution with O(n) [do not quite understand this algorithm]

# if linked list is 1-2-3-4
"""

in first round a is 1
nextnode is set to 2
a.nextnode is set to None (meaning that 1 will know be linked to None, thereby becoming the tail)

previous is set to 1
and a is set to 2 (that is how it is traversing the list)

this continues until it reaches 4 



while a:


    nextnode = a.nextnode
    a.nextnode = previous

    previous = a
    a = nextnode

"""

# interview problem: linked list nth to last node

"""
Write a function that takes a head node and an integer value **n** and then returns the nth to last node in the linked 
list. For example, given:

in other words: give value (negative index) that will return the node value  

"""

class Node:

    def __init__(self, value):
        self.value = value
        self.nextnode  = None



def nth_to_last_node(n, head):

    l = []

    while head:

        l.append(head.value)
        head = head.nextnode

    if n <= len(l):
        return l[-n]

    else:
        raise LookupError("n larger than linked list")



"""
RUN THIS CELL TO TEST YOUR SOLUTION AGAINST A TEST CASE 

PLEASE NOTE THIS IS JUST ONE CASE
"""

from nose.tools import assert_equal

a = Node(1)
b = Node(2)
c = Node(3)
d = Node(4)
e = Node(5)

a.nextnode = b
b.nextnode = c
c.nextnode = d
d.nextnode = e


####

class TestNLast(object):

    def test(self, sol):
        assert_equal(sol(2, a), d)
        print('ALL TEST CASES PASSED')



# Run tests
#t = TestNLast()
#t.test(nth_to_last_node)

#print(nth_to_last_node(3, a))


# interview problem: implement a singly linked list

# singly linked list implementation

class Node:

    def __init__(self, value):

        self.value = value
        self.nextnode = None

# doubly linked list

class DoublyLinkedListNode:

    def __init__(self, value):

        self.value = value
        self.nextnode = None
        self.prevnode = None


