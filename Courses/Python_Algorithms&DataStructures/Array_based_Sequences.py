# three main sequence classes in python

# list, tuple and strings

#list = [1,2,3]

#list = [0]*3

#list.extend([4,5])

#print(list)

tuple = (1,2,3)

string = "1,2,3"

# How computers store information

"""

memory of a computer stored in bits (one byte is 8 bits)

computers use a memory address 

Each byte associated with unique address (Byte #2144 vs Byte #2147)

RAM: random access memory (individual byte of memory can be retrieved in O(1) time) 

arrays group related variables in the computers memory 

each unicode character is represented (in python) by 16 bits (or 2 bytes)

each cell of an array uses same number of bytes 

appropriate memory address calculation: start + cellsize*index 

# Referential Arrays

create array of strings (but strings have different sizes), so instead use object references

each element (or index in an array) is referencing to the actual string object

when computing the slice of a list, the new list references the same elements that the original list is 

# Dynamic Arrays 

no need to specify how large an array is beforehand 

"""
# coding
"""
import sys

# set n
n = 100

data = []

for i in range(n):

    a = len(data)

    b = sys.getsizeof(data)

    print("Length: {0:3}; Size in bytes: {1:3}".format(a,b))

    data.append(n)

# python allocates more memory to the array (than necessary) in order it to not having to specify how large
# the array will be
"""

"""

# dynamic array implementation

1) allocate new array B with larger capacity (twice the capacity of the old array) 

2) set B[i] = A[i] for i = 0, ..., n-1 where n denotes current number of items

3) set A = B that is, we now use B as the array supporting the list

4) insert new element in new array


"""
# coding implementation of dynamic array




"""

Amortization 

algorithmic design pattern (amortization) (how efficient is the doubling of array size every time, the list is about
to overflow (all locations in list are filled))

"""


"""
Interview Problems: 
"""

# Anagram Check: given two strings, check if they are anagrams (contain the exact same letters)
# ignore lower and uppercase

s1 = "aabbcc"
s2 = "aabbc"

# using python methods
def anagram(s1, s2):

    sl_1 = [i.lower() for i in s1 if i != " "]
    s2_l = [i.lower() for i in s2 if i != " "]

    return sorted(sl_1) == sorted(s2_l)



# alternative anagram method (counting number of letters in string)

def alternative_anagram(s1, s2):

    sl_1 = [i.lower() for i in s1 if i != " "]
    s2_l = [i.lower() for i in s2 if i != " "]

    if len(sl_1) != len(s2_l):
        return False

    d1 = {}
    d1 = {n: 0 for n in sl_1 if n not in d1}
    d2 = {}
    d2 = {n: 0 for n in s2_l if n not in d2}


    for s1, s2 in zip(sl_1, s2_l):
        d1[s1] += 1
        d2[s2] += 1


    count = 0
    for i in d1:

        try:

            if d1[i] == d2[i]:
                count += 1

            else:
                break

        except KeyError:

            return False

    return len(d1) == count

#print(alternative_anagram(s1, s2))



# Array Pair Sum
# given an integer array, output all the unique pairs that sum up to a specific value k

"""
example 

pair_sum([1,3,2,2],4) 

would return 2 pairs

(1,3)
(2,2) 

should also return number of pairs to return 

"""

l = [1, 2, 3, 1]
k = 3

def pair_sum(l, k):

    count = 0
    index = 1

    pair_sum_list = []

    for i in l:
        for n in l[index:]:

            if i + n == k and (n,i) not in pair_sum_list:

                print((i, n))
                pair_sum_list.append((i,n))
                count += 1

        index += 1

    return count


#print(pair_sum(l,k))



# find the missing element

"""

Consider an array of non-negative integers. A second array is formed by shuffling the elements of the first array and 
deleting a random element. Given these two arrays, find which element is missing in the second array.

Here is an example input, the first array is shuffled and the number 5 is removed to construct the second array.

Input:

finder([1,2,3,4,5,6,7],[3,7,2,1,4,6])

Output:

5 is the missing number

"""

arr1 = [1,2,3,4,5,6,7]
arr2 = [3,7,2,1,4,6]

def finder(arr1, arr2):

    arr1.sort()
    arr2.sort()

    for n1, n2 in zip(arr1, arr2):
        if n1 != n2:
            return n1

    return arr1[-1]


# largest continuous sum

# given an array of integers (positive and negative) find the largest continuous sum

l = [1,2,-1,3,4,10,10,-10,-1]

def large_cont_sum(l):

    l_r = l[::-1]

    l_f = []
    l_b = []

    sums = []

    for index, i in enumerate(l):

        l_f.append(l[:index + 1])
        l_b.append(l_r[:index + 1])

        sums.append(sum(l_f[index]))
        sums.append(sum(l_b[index]))

    return max(sums)

#print(large_cont_sum(l))


# sentence reversal

# given a string of words, reverse all the words

sent = '  This is        the best'



def rev_word(s):

    l = s.split()
    return (' '.join(l[::-1]))


#print(rev_word(sent))

def rev_word_alt(s):

    word = ''
    list = []

    for i in s:

        if i != ' ':
            word+=i

        if i == ' ':
            if word != '':
                list.append(word)
            word = ''

    list.append(word)

    rev_sent = ''

    for i in list[::-1]:
        rev_sent+=i + ' '

    return rev_sent

#print(rev_word_alt(sent))


# string compression

"""
Given a string in the form 'AAAABBBBCCCCCDDEEEE' compress it to become 'A4B4C5D2E4'. 
For this problem, you can falsely "compress" strings of single or double letters. For instance, it is okay for 'AAB' 
to return 'A2B1' even though this technically takes more space.

The function should also be case sensitive, so that a string 'AAAaaa' returns 'A3a3'.
"""

def compress(s):

    # convert string to list
    l = [n for n in s]
    # remove multiple occurrences of each element in list
    x = sorted(list(set(l)))
    # create dictionary based on each letter in string
    d = {n: 0 for n in x}

    for i in s:
        if i in d:
            up = d[i]
            up += 1

            d.update({i: up})

    word = ''

    for i in d:
        word += i+str(d[i])

    return word

#print(compress('AAaa'))


"""
Unique Characters in String
Problem

Given a string,determine if it is comprised of all unique characters. For example, the string 'abcde' has all 
unique characters and should return True. The string 'aabcde' contains duplicate characters and should return false.

"""

def uni_char(s):

    count=0

    for i in s:

        if i in s[:count]:
            return False
            break

        count+=1

    else:
        return True



print(uni_char('abcde'))

