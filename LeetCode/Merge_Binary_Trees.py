"""

NOT FINISHED YET (20/12/2017) 

Given two binary trees and imagine that when you put one of them to cover the other, some nodes of the two trees are
overlapped while the others are not.

You need to merge them into a new binary tree. The merge rule is that if two nodes overlap, then sum node values up as
the new value of the merged node. Otherwise, the NOT null node will be used as the node of new tree.

Example 1:

Input:
	Tree 1                     Tree 2
          1                         2
         / \                       / \
        3   2                     1   3
       /                           \   \
      5                             4   7
Output:
Merged tree:
	     3
	    / \
	   4   5
	  / \   \
	 5   4   7

Note: The merging process must start from the root nodes of both trees.


"""

#tree1 = [[1,3], [1,2], [3,5]]
#tree2 = [[2,1], [2,3], [1,4], [3,7]]

tree1 = {0: 1, 1: 3, 2: 2, 3: 5, 4: None, 5: None, 6: None}
tree2 = {0: 2, 1: 1, 2: 3, 3: None, 4: 4, 5: None, 6: 7}

new_tree = {}

if len(tree1) == len(tree2):

    for i in tree1:
        print(i)

        if tree1[i] != None and tree2[i] != None:

            new_tree.update({i: tree1[i] + tree2[i]})

        else:

            if tree1[i] != None:

                new_tree.update({i: tree1[i]})

            else:

                new_tree.update({i: tree2[i]})



print(new_tree)

"""

	     3
	    / \
	   4   5
	  / \   \
	 5   4   7

"""

#print(" "*len(new_tree)+"test")

for index, i in enumerate(new_tree):

    #print(2**i)

    if i == 0:
        print(" "*len(new_tree), new_tree[i])
        print((i+1)*(" "*len(new_tree)+ "/" + " \\"))

    else:

        count = 0

        while count <= 2**i:

            if count >= i:

                #print(count, i)

                print(" " * len(new_tree), new_tree[count])
                print((index + 1) * (" " * len(new_tree) + "/" + " \\"))

            count+=1




        #print(" " * len(new_tree), new_tree[i])
        #print((index + 1) * (" " * len(new_tree) + "/" + " \\"))




"""

values per row

0: 1
1: 2
2: 4
3: 8
4: 16

"""

def rows(dict):
    total = 0
    for i in dict:
        total+=2**i

        if total == len(new_tree):
            rows = i+1
            break

    return rows






