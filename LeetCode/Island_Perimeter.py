"""

You are given a map in form of a two-dimensional integer grid where 1 represents land and 0 represents water.
Grid cells are connected horizontally/vertically (not diagonally). The grid is completely surrounded by water,
and there is exactly one island (i.e., one or more connected land cells). The island doesn't have "lakes"
(water inside that isn't connected to the water around the island). One cell is a square with side length 1.
The grid is rectangular, width and height don't exceed 100. Determine the perimeter of the island.

Example:

[[0,1,0,0],
 [1,1,1,0],
 [0,1,0,0],
 [1,1,0,0]]

Answer: 16
Explanation: The perimeter is the 16 yellow stripes in the image below:


NOT COMPLETED (13/02/18) 

"""

island = [[0,1,0,0],
          [1,1,1,0],
          [0,1,0,0],
          [1,1,0,0]]

dic = {}

for index, i in enumerate(island):
    dic.update({index: i})

import numpy as np

island = np.array(island)



it = np.nditer(island, flags=['multi_index'])




old_origin = []
count = 0



"""
while not it.finished:

    # origin = (row, column)
    origin = it.multi_index

    x = island.item(origin[0]-1, origin[1])

    # do not check against an origin that was already checked


    # upper left corner
    if origin[0] - 1 < 0 and origin[1] - 1 < 0:

        if island.item(origin) == 1:
            count+=2

        #print(island.item(origin), origin)

        if island.item(origin) != island.item(origin[0],origin[1]+1):
            count+=1

        if island.item(origin) != island.item(origin[0]+1,origin[1]):
            count+=1



    # upper right corner
    elif origin[0] - 1 < 0 and origin[1] + 1 > len(island[0])-1:

        if island.item(origin) == 1:
            count+=2


        if island.item(origin) != island.item(origin[0]+1,origin[1]):
            count+=1



    # upper side (no corners)
    elif origin[0]-1 < 0:

        # adds +1 to count for every value in array that is 1 at the side
        if island.item(origin) == 1:
            count+=1

        if island.item(origin) != island.item(origin[0],origin[1]+1):

            if (origin[0],origin[1]+1) not in old_origin:

                count+=1

        if island.item(origin) != island.item(origin[0],origin[1]-1):

            if (origin[0],origin[1]-1) not in old_origin:

                count+=1

        if island.item(origin) != island.item(origin[0]+1,origin[1]):

            if (origin[0]+1,origin[1]) not in old_origin:

                count+=1



    # lower left corner
    elif origin[0] + 1 > len(island)-1 and origin[1] -1 < 0:

        if island.item(origin) == 1:
            count+=2

        if island.item(origin) != island.item(origin[0], origin[1] + 1):
            count+=1



    # left side (no corners)
    elif origin[1]-1 < 0:

        # adds +1 to count for every value in array that is 1 at the side
        if island.item(origin) == 1:
            count += 1


        if island.item(origin) != island.item(origin[0],origin[1]+1):

            if (origin[0],origin[1]+1) not in old_origin:

                count+=1

        if island.item(origin) != island.item(origin[0]-1,origin[1]):

            if (origin[0]-1,origin[1]) not in old_origin:

                count+=1

        if island.item(origin) != island.item(origin[0]+1,origin[1]):

            if (origin[0]+1,origin[1]) not in old_origin:

                count+=1



    # lower right corner
    elif origin[0] + 1 > len(island)-1 and origin[1] + 1 > len(island[0])-1:


        if island.item(origin) == 1:
            count+=2




    # lower side (no corners)
    elif origin[0]+1 > len(island)-1:

        # adds +1 to count for every value in array that is 1 at the side
        if island.item(origin) == 1:
            count += 1

        if island.item(origin) != island.item(origin[0], origin[1] + 1):
            if (origin[0], origin[1] + 1) not in old_origin:
                count+=1


        if island.item(origin) != island.item(origin[0],origin[1]-1):
            if (origin[0],origin[1]-1) not in old_origin:
                count+=1

        if island.item(origin) != island.item(origin[0]-1,origin[1]):
            if (origin[0]-1,origin[1]) not in old_origin:
                count+=1


    # right side (no corners)
    elif origin[1]+1 > len(island[0])-1:

        # adds +1 to count for every value in array that is 1 at the side
        if island.item(origin) == 1:
            count += 1


    # all values with no border contact
    else:

        if island.item(origin) != island.item(origin[0]+1,origin[1]):
            if (origin[0]+1,origin[1]) not in old_origin:
                count+=1

        if island.item(origin) != island.item(origin[0]-1,origin[1]):
            if (origin[0]-1,origin[1]) not in old_origin:
                count+=1

        if island.item(origin) != island.item(origin[0],origin[1]+1):
            if (origin[0],origin[1]+1) not in old_origin:
                count+=1

        if island.item(origin) != island.item(origin[0],origin[1]-1):
            if (origin[0],origin[1]-1) not in old_origin:
                count+=1



    old_origin.append(origin)


    it.iternext()


print(count)
"""




# exchange of 1 and 0 positions would currently lead to the same outcome (but it should not!)
#





class Solution(object):

    old_origin = []
    count = 0

    def corner_count(self, value):

        global count
        global old_origin

        if island.item(value) == 1:
            count += 2
        return count

    def side_count(self, value):

        global count
        global old_origin

        if island.item(value) == 1:
            count += 1
        return count


    def right_column(self, value):

        global count
        global old_origin

        if island.item(value) != island.item(value[0], value[1] + 1):
            if (value[0], value[1] + 1) not in old_origin:
                count += 1

        return count

    def left_column(self, value):

        global count
        global old_origin

        if island.item(value) != island.item(value[0], value[1] - 1):
            if (value[0], value[1] - 1) not in old_origin:
                count += 1

        return count

    def upper_row(self, value):

        global count
        global old_origin

        if island.item(value) != island.item(value[0]-1, value[1]):
            if (value[0]-1, value[1]) not in old_origin:
                count += 1

        return count

    def lower_row(self, value):

        global count
        global old_origin

        if island.item(value) != island.item(value[0]+1, value[1]):
            if (value[0]+1, value[1]) not in old_origin:
                count += 1

        return count

    def array_iteration(self, array):

        island = np.array(array)
        it = np.nditer(island, flags=['multi_index'])

        while not it.finished:

            origin = it.multi_index

            # upper left corner
            if origin[0] - 1 < 0 and origin[1] - 1 < 0:

                self.corner_count(origin)
                self.right_column(origin)
                self.lower_row(origin)


            # upper right corner
            elif origin[0] - 1 < 0 and origin[1] + 1 > len(island[0]) - 1:

                self.corner_count(origin)
                self.lower_row(origin)


            # upper side (no corners)
            elif origin[0] - 1 < 0:

                self.side_count(origin)
                self.right_column(origin)
                self.left_column(origin)
                self.lower_row(origin)


            # lower left corner
            elif origin[0] + 1 > len(island) - 1 and origin[1] - 1 < 0:

                self.corner_count(origin)
                self.right_column(origin)


            # left side (no corners)
            elif origin[1] - 1 < 0:

                # adds +1 to count for every value in array that is 1 at the side
                self.side_count(origin)
                self.right_column(origin)
                self.upper_row(origin)
                self.lower_row(origin)


            # lower right corner
            elif origin[0] + 1 > len(island) - 1 and origin[1] + 1 > len(island[0]) - 1:

                self.corner_count(origin)


            # lower side (no corners)
            elif origin[0] + 1 > len(island) - 1:

                # adds +1 to count for every value in array that is 1 at the side
                self.side_count(origin)
                self.right_column(origin)
                self.left_column(origin)
                self.upper_row(origin)


            # right side (no corners)
            elif origin[1] + 1 > len(island[0]) - 1:

                # adds +1 to count for every value in array that is 1 at the side
                self.side_count(origin)


            # all values with no border contact
            else:

                self.upper_row(origin)
                self.lower_row(origin)
                self.right_column(origin)
                self.left_column(origin)

            old_origin.append(origin)

            it.iternext()

        return count



x = Solution()

print(x.array_iteration(island))







"""
while not it.finished:

    # origin = (row, column)
    origin = it.multi_index

    # do not check against an origin that was already checked

    # upper left corner
    if origin[0] - 1 < 0 and origin[1] - 1 < 0:

        test.corner_count(origin)
        

        test.right_column(origin)

        test.lower_row(origin)


    # upper right corner
    elif origin[0] - 1 < 0 and origin[1] + 1 > len(island[0])-1:

        corner_count(origin)

        lower_row(origin)


    # upper side (no corners)
    elif origin[0]-1 < 0:

        side_count(origin)

        right_column(origin)

        left_column(origin)

        lower_row(origin)


    # lower left corner
    elif origin[0] + 1 > len(island)-1 and origin[1] -1 < 0:

        corner_count(origin)

        right_column(origin)


    # left side (no corners)
    elif origin[1]-1 < 0:

        # adds +1 to count for every value in array that is 1 at the side
        side_count(origin)

        right_column(origin)

        upper_row(origin)

        lower_row(origin)


    # lower right corner
    elif origin[0] + 1 > len(island)-1 and origin[1] + 1 > len(island[0])-1:

        corner_count(origin)


    # lower side (no corners)
    elif origin[0]+1 > len(island)-1:

        # adds +1 to count for every value in array that is 1 at the side
        side_count(origin)

        right_column(origin)

        left_column(origin)

        upper_row(origin)


    # right side (no corners)
    elif origin[1]+1 > len(island[0])-1:

        # adds +1 to count for every value in array that is 1 at the side
        side_count(origin)


    # all values with no border contact
    else:

        upper_row(origin)

        lower_row(origin)

        right_column(origin)

        left_column(origin)


    old_origin.append(origin)


    it.iternext()


print(count)
"""




















class Solution:

    def __init__(self):
        pass


    def islandPerimeter(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
