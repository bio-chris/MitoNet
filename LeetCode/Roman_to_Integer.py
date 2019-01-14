"""

Given a roman numeral, convert it to an integer.

Input is guaranteed to be within the range from 1 to 3999.

I: 1
II: 2
III: 3
IV: 4

V: 5
VI: 6
VII: 7
VIII: 8

IX: 9
X: 10

L: 50

C: 100

D: 500

M: 1000

LIV

50-1+5 54



"""

dic = {"I":1, "V":5, "X":10, "L":50, "C":100, "D":500, "M":1000}

numeral = "CXXXVIII"

old_i = None
number = 0

for i in numeral:
    if i in dic:

        #print(dic[i])

        if old_i != None and old_i >= dic[i]:
            number+=old_i

        if old_i != None and  old_i < dic[i]:
            number -= old_i

    old_i = dic[i]

number+=old_i

print(number)

# Accepted!

def romanToInt(s):

    dic = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}

    old_i = None
    number = 0

    for i in s:
        if i in dic:

            # print(dic[i])

            if old_i != None and old_i >= dic[i]:
                number += old_i

            if old_i != None and old_i < dic[i]:
                number -= old_i

        old_i = dic[i]

    number += old_i

    return number


print(romanToInt("CCIX"))
