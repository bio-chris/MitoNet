"""

You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order
and each of their nodes contain a single digit. Add the two numbers and return it as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.

Example

Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
Output: 7 -> 0 -> 8
Explanation: 342 + 465 = 807.


NOT YET ACCEPTED

maybe try and implement for more than 3 values

"""

# code from internet
"""
class Node:
    def __init__(self,val):
        self.val = val
        self.next = None # the pointer initially points to nothing
"""

class Node:
    def __init__(self,val):
        self.val = val
        self.next = None

    def getData(self):
        return self.data

    def getNext(self):
        return self.next

    def setData(self,newdata):
        self.data = newdata

    def setNext(self,newnext):
        self.next = newnext




l_list1 = Node(2)
l_list1.next = Node(4)
l_list1.next.next = Node(3)


l_list2 = Node(5)
l_list2.next = Node(6)
l_list2.next.next = Node(4)






# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:

    def __init__(self):
        self.head = None


    """
    def add(self, item):
        temp = Node(item)
        temp.setNext(self.head)
        self.head = temp
    """


    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """

        l_1 = []
        l_2 = []

        list1 = l1
        while list1:
            i = list1.val
            list1 = list1.next
            l_1.append(i)

        list2 = l2
        while list2:
            j = list2.val
            list2 = list2.next
            l_2.append(j)

        l_1.reverse()
        l_2.reverse()

        def list_to_int(l):

            new_str = ""
            for i in l:
                new_str += str(i)

            return int(new_str)

        # add int values and convert to string
        new_l_list = str(list_to_int(l_1) + list_to_int(l_2))

        # convert string into single string elements in list
        new_l_list = list(new_l_list)

        # convert all string elements in list to int values
        new_l_list = [int(i) for i in new_l_list]

        # reverse list
        new_l_list.reverse()


        """
         val = carry
            if l1:
                val += l1.val
                l1 = l1.next
        """

        val = 0
        for index, i in enumerate(new_l_list):

            #val += Node(i)
            #new_l_list1 = new_l_list1.next

            if index == 0:
                new_l_list1 = Node(i)
                #print(i)

            if index > 0:
                new_l_list1.next = Node(i)


        """

        new_l_list1 = Node(new_l_list[0])
        new_l_list1.next = Node(new_l_list[1])
        new_l_list1.next.next = Node(new_l_list[2])

        """

        def add(item):
            temp = Node(item)
            temp.setNext(self.head)
            self.head = temp



        return new_l_list1



test = Solution()


result = test.addTwoNumbers(l_list1, l_list2)



final = result
while final:
    print(final.val)
    final = final.next