"""

 Initially, there is a Robot at position (0, 0).
 Given a sequence of its moves, judge if this robot makes a circle,
 which means it moves back to the original place.

The move sequence is represented by a string. And each move is represent by
a character. The valid robot moves are
R (Right), L (Left), U (Up) and D (down).
The output should be true or false representing whether the robot makes a
circle.

Example 1:

Input: "UD"
Output: true

Example 2:

Input: "LL"
Output: false


"""

import numpy as np
class Solution:
    def judgeCircle(self, moves):
        """
        :type moves: str
        :rtype: bool
        """

        x, y = len(moves), len(moves)

        org_x, org_y = len(moves), len(moves)

        for n in moves:

            if n == "L":
                x -= 1

            if n == "R":
                x += 1

            if n == "U":
                y += 1

            if n == "D":
                y -= 1

        if (x, y) == (org_x, org_y):
            return True

        else:
            return False




