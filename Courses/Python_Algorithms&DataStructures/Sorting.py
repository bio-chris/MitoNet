from timeit import default_timer as timer


# bubble sort



l = [9, 3, 1, 14, 2, 5, 8, 7, 6]

#print(sorted(l))


#for index, n in enumerate(l):
#    print(l[:-index-1])


# my implementation
def bubble_sort(l):

    for index, n in enumerate(l):
        for index2, n in enumerate(l[:-index-1]):

            if l[index2+1] < l[index2]:

                var = l[index2]
                l[index2] = l[index2+1]
                l[index2+1] = var

    return l



# udemy implementation
def bubble_sort_u(arr):

    for n in range(len(arr)-1,0,-1):
        for k in range(n):

            if arr[k] > arr[k+1]:
                temp = arr[k]
                arr[k] = arr[k+1]
                arr[k+1] = temp


    return arr

#print(bubble_sort_u(l))



"""

iterate through list

take first element and set it to curr_min

compare each of the following elements with curr_min and check if it is smaller

if any element is smaller than curr_min, set curr_min to that element 

when iterated through list

move the curr_min to the left side of the list and the original min to the previous pos of curr_min

repeat iteration, but this time take only list[1:] (left part is considered sorted)
next iteration will be list [2:] and so on ...


"""


l = [9,6,7,99,3,15,32,100,2,1]

# my implementation of selection sort



def selection_sort(l):

    for index, n in enumerate(l):
        if index == len(l)-1:
            break

        curr_min = l[index]

        swap = False
        for index2, n2 in enumerate(l[index+1:]):

            if l[index+1:][index2] < curr_min:

                swap = True
                curr_min = l[index+1:][index2]
                pos = index2+index+1

        if swap == True:

            temp = l[index]
            l[index] = curr_min
            l[pos] = temp


    return l


# udemy implementation of selection sort


def sel_sort(l):

    for fillslot in list(range(len(l)-1, 0, -1)):

        pos_max = 0

        for location in list(range(1, fillslot+1)):

            if l[location] > l[pos_max]:
                pos_max = location

        temp = l[fillslot]
        l[fillslot] = l[pos_max]
        l[pos_max] = temp


# insertion sort

# my implementation of insertion sort

"""

generate sub list

iterate through list

append first element to sub list 

second iteration only after index > 0 

check if element is smaller than any element in sublist

"""


#l = [6,4,1,3,2]


"""
def insertion_sort(l):
    sub_l = []

    for index, n in enumerate(l):

        if index == 0:
            sub_l.append(n)

        else:
            print(sub_l, n)

            insert = False
            end_insert = False

            for index2, n2 in enumerate(sub_l):

                if n < n2:

                    new_index = index2
                    insert = True
                    break

            if insert == True:
                sub_l.insert(new_index, n)

            else:
                sub_l.append(n)

    return sub_l


#print(sub_l)
"""


# course implementation of insertion sort


def insertion_sort(arr):

    for i in range(1,len(arr)):

        currentvalue = arr[i]
        position = i

        while position > 0 and arr[position-1] > currentvalue:

            arr[position] = arr[position-1]
            position = position-1

        arr[position] = currentvalue

    return arr




"""

# shell sort

# my implementation of shell sort


l = [9,8,5,2,4,10,1,3]

#print(len(l))

print(list(range(0,int(len(l)/2))))
#"""

l = [4,5,1,3,9,2,6,11]

#print(insertion_sort(l))

n = int(len(l)/2)

while n > 1:

    for count, i in enumerate(range(0,n)):

        #print(l[count::n])

        new_l = l[count::n]

        for i in range(1, len(new_l)):

            currentvalue = new_l[i]
            position = i

            """
            while position > 0 and new_l[position - 1] > currentvalue:

                #print("pos:", position, "value at pos:", new_l[position], "pos*n:", position*n, "n:", n, count)


                new_l[position] = new_l[position - 1]


                # currently only works for [x1,x2] but fails for longer lists

                
                print(count, n, position, position*n)

                temp = l[count]
                l[count] = l[count+n]
                l[count + n] = temp
                


                position -= 1

            new_l[position] = currentvalue
            """


        #print(new_l)
        #print(l)


    n = int(n/2)

    """
    if n == 2:
        n-=1
    """


#print(l)


# new insertion sort

l = [6,4,3,1,5,7, 99, 33, 22, 21, 100]

# my own implementation of insertion sort
def insertion_sort(l):

    for count, i in enumerate(l):

        pos = count
        # compare i to each value in sublist (starting from right to left)
        for i2 in l[:count][::-1]:

            # if i smaller than any value i2 in sublist reduce pos by value of -1 and swap values
            if i < i2:
                pos-=1

                temp = l[pos]
                l[pos+1] = temp

        l[pos] = i

    return l

#print(insertion_sort(l))

l = [3,5,4,1,6,8]


def shell_sort(l):

    n = int(len(l)/2)

    while n >= 1:
        for count, i in enumerate(range(n)):
            for i2 in range(i + n, len(l), n):

                pos = i2
                curr_val = l[i2]

                while pos >= n and l[pos-n] > curr_val:

                    l[pos] = l[pos - n]
                    pos -= n

                l[pos] = curr_val

        n = int(n / 2)

    return l




# merge sort

# my implementation of merge sort


m = int(len(l)/2)

print(m, l)




















