"""

A self-dividing number is a number that is divisible by every digit it contains.

For example, 128 is a self-dividing number because 128 % 1 == 0, 128 % 2 == 0, and 128 % 8 == 0.

Also, a self-dividing number is not allowed to contain the digit zero.

Given a lower and upper number bound, output a list of every possible self dividing number, including the bounds if
possible.

Example 1:

Input:
left = 1, right = 22
Output: [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 15, 22]

Note:
The boundaries of each input argument are 1 <= left <= right <= 10000.

"""


class Solution:
    def selfDividingNumbers(self, left, right):
        """
        :type left: int
        :type right: int
        :rtype: List[int]
        """


left = 1
right = 22


output = []

while left <= 200:

    if str(0) not in str(left):

        count = 0
        for i in str(left):

            if left%int(i) == 0:
                count+=1

            if count == len(str(left)):
                output.append(left)



    left += 1

print(output)
print(135%5)



class Solution:

    def selfDividingNumber(self, left, right):

        if left >= 1 and right <= 10000:

            output = []

            while left <= right:

                if str(0) not in str(left):

                    count = 0
                    for i in str(left):

                        if left % int(i) == 0:
                            count += 1

                        if count == len(str(left)):
                            output.append(left)

                left += 1

            return output

        else:

            return("Values outside of boundaries")



answer = Solution()
print(answer.selfDividingNumber(2,22))



