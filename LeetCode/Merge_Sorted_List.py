""""

Merge two sorted linked lists and return it as a new list. The new list should be made by splicing together the
nodes of the first two lists.

"""

"""
list1 = [1,3,5,7,9]

list2 = [2,4,6,8,10]

def mergesort(list1,list2):
    pass

new_list = []
index=0

for i in list1:
    #print(i)
    for i2 in list2[index:]:
        print(i, i2)
        if i < i2:
            new_list.append(i)
            new_list.append(i2)
            break

        else:
            new_list.append(i2)
            new_list.append(i)
            break
    index+=1

print(new_list)

x1 = False
x2 = False

for i in list1:
    for i2 in list2:
        if i2 > i:
            x1 = True
        else:
            pass
    if x1 == True:
        new_list.append(i)
"""


########################
# Sort Algorithm 1: Using the min Method

list = [6,1,3,8,4]
list_copy = list[:]

new_list = []

for i in list:

    x = min(list_copy)
    list_copy.remove(x)
    new_list.append(x)

#print(new_list)

def sort1(list):

    list_copy = list[:]
    new_list = []

    for i in list:
        x = min(list_copy)
        list_copy.remove(x)
        new_list.append(x)

    return new_list

#print(sort1([4,10,34,2,3,1]))

print(list)

index = 0
true_list = []
new_list = [0]*len(list)
checking = False
for i in list:
    for i2 in list[index:]:
        if i > i2:
            true_list.append(1)
        else:
            true_list.append(0)

        #print(i,i2)

    if true_list == []:
        break

    print(new_list, i)

    """
    if new_list[sum(true_list)] == 0:

        new_list[sum(true_list)] = i

    elif new_list[sum(true_list)+1] == 0:

        new_list[sum(true_list)+1] = i
    """

    n = 1
    while checking == False:

        #print(sum(true_list))

        if len(list) == sum(true_list)+1:
            new_list[-1] = i

        if new_list[sum(true_list)] == 0:
            new_list[sum(true_list)] = i
            break

        elif new_list[sum(true_list)+n] == 0:
            new_list[sum(true_list)+n] = i
            break

        n+=1

    #while new_list[sum(true_list)+n] == 0:




    true_list = []

    index+=1

print(new_list)