"""

The Hamming distance between two integers is the number of positions at which the corresponding bits are different.

Given two integers x and y, calculate the Hamming distance.

Note:
0 ≤ x, y < 2^31.


Example:

    Input: x = 1, y = 4

Output: 2

Explanation:
1   (0 0 0 1)
4   (0 1 0 0)
       ↑   ↑

"""


class Solution:

    # loops through two lists and compares each object at the same index to each other. if different, count +1
    def loop(self, x_list, y_list):
        hamming_distance = 0

        for x, y in zip(x_list, y_list):
            if x != y:
                hamming_distance += 1

        return hamming_distance


    def hammingDistance(self, x,y):

        x = int(x)
        y = int(y)

        if x >= 0 and y < 2**31:

            # converts decimal into a binary list of 1s and 0s (removes the b annotation)
            x_bin_list = [x for x in bin(x) if x != "b"]
            y_bin_list = [y for y in bin(y) if y != "b"]

            # equalizes the length of the two lists

            if len(x_bin_list) > len(y_bin_list):

                len_difference = len(x_bin_list) - len(y_bin_list)
                new_y_bin_list = ['0'] * len_difference + y_bin_list

                hamming_distance = self.loop(x_bin_list, new_y_bin_list)

                #print(x_bin_list)
                #print(new_y_bin_list)

            else:

                len_difference = len(y_bin_list) - len(x_bin_list)
                new_x_bin_list = ['0'] * len_difference + x_bin_list

                hamming_distance = self.loop(new_x_bin_list, y_bin_list)

                #print(new_x_bin_list)
                #print(y_bin_list)

            return hamming_distance

        else:

            print("")

#Solution = (hamming_distance(1,4))

#print(Solution)

answer = Solution()

print(answer.hammingDistance(10,10))




