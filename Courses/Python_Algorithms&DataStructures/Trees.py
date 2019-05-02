

# representing a tree through lists

"""
def BinaryTree(r):
    return [r, [], []]


def insertLeft(root, newBranch):
    t = root.pop(1)

    if len(t) > 1:
        root.insert(1,[newBranch, t, []])
    else:
        root.insert(1,[newBranch, [], []])
    return root


def insertRight(root, newBranch):
    t = root.pop(2)

    if len(t) > 1:
        root.insert(2,[newBranch, [], t])
    else:
        root.insert(2, [newBranch, [], []])
    return root


def getRootVal(root):
    return root[0]


def setRootVal(root, newVal):
    root[0] = newVal

def getLeftChild(root):
    return root[1]

def getRightChild(root):
    return root[2]

"""

#r = BinaryTree(3)
#insertLeft(r,4)
#insertLeft(r,5)

#insertRight(r,6)
#insertRight(r,7)

#print(r)

# nodes and references implementation of trees (using OOP)

class BinaryTree(object):

    def __init__(self, rootObj):

        self.key = rootObj
        self.left = None
        self.right = None

    """
    def insertLeft(self, newNode):

        if self.leftChild == None:
            self.leftChild = BinaryTree(newNode)
        else:
            t = BinaryTree(newNode)
            t.leftChild = self.leftChild
            self.leftChild = t

    def insertRight(self, newNode):

        if self.rightChild == None:
            self.rightChild = BinaryTree(newNode)
        else:
            t = BinaryTree(newNode)
            t.rightChild = self.rightChild
            self.rightChild = t

    
    def getRightChild(self):
        return self.rightChild

    def getLeftChild(self):
        return self.leftChild

    def setRootVal(self, obj):
        self.key = obj

    """

    def getRootVal(self):
        return self.key



#print(r.getRootVal())
#r.insertLeft('b')
#print(r.getLeftChild().getRootVal( ))


# binary heap implementation


# interview problems
# binary search tree check: given a binary tree, check whether it is a binary search tree or not


# creating the binary tree
tree = [5,1,4,None,None,3,6]

bt = BinaryTree(25)
bt.left = BinaryTree(20)
bt.right = BinaryTree(36)
bt.left.left = BinaryTree(10)
bt.left.right = BinaryTree(22)
bt.right.left = BinaryTree(30)
bt.right.right = BinaryTree(36)

#print(bt.right.right.getRootVal())

"""

Criteria of BST (binary search trees)

• The left subtree of a node contains only nodes with keys less than the node’s key.
• The right subtree of a node contains only nodes with keys greater than the node’s key.
• Both the left and right subtrees must also be binary search trees.

"""


"""
# returns the left branch, then the right
class Solution(object):
    def inorderTraversal(self, root):
        res = []
        if root:
            res = self.inorderTraversal(root.left)
            res.append(root.key)
            res = res + self.inorderTraversal(root.right)
        return res

    def PreorderTraversal(self, root):
        res = []
        if root:
            res.append(root.key)
            res = res + self.PreorderTraversal(root.left)
            res = res + self.PreorderTraversal(root.right)


        return res


    def isBST(self, bt):

        nr_nodes = len(self.PreorderTraversal(bt))
        print(nr_nodes)

        for count, i in enumerate(self.PreorderTraversal(bt)):

            # left subtree
            if count <= (nr_nodes-1)/2:

                # set root value
                if count == 0:
                    root = i

                if count == 1:

                    if i < root:
                        continue
                    else:
                        break

                l = []
                if i == None:
                    l.append(i)

                if len(l) == 2:
                    pass

                old_i = i

            # right subtree
            else:
                pass


    def sort_check(self, bt):
        return self.inorderTraversal(bt) == sorted(self.inorderTraversal(bt))


so = Solution()

print(so.inorderTraversal(bt))
print(sorted(so.inorderTraversal(bt)))

print(so.sort_check(bt))

#print(so.isBST(bt))
"""

# tree level order print

"""
class Node:
    def __init__(self, val=None):
        self.left = None
        self.right = None
        self.val = val



tree = Node(1)
tree.left = Node(2)
tree.right = Node(3)
tree.left.left = Node(4)
tree.right.left = Node(5)
tree.right.right = Node(6)

import collections

def levelOrderPrint(tree):

    if not tree:
        return

    nodes = collections.deque([tree])

    currentCount = 1
    nextCount = 0


    while len(nodes) != 0:

        currentNode = nodes.popleft()

        print(len(nodes), currentNode.val)

        if currentNode.left:
            nodes.append(currentNode.left)

        if currentNode.right:
            nodes.append(currentNode.right)

levelOrderPrint(tree)
"""


# trim a binary search tree

class Node:
    def __init__(self, val=None):
        self.left = None
        self.right = None
        self.val = val


tree = Node(8)
tree.left = Node(3)
tree.right = Node(10)

tree.left.left = Node(1)
tree.left.right = Node(6)

tree.left.right.left = Node(4)
tree.left.right.right = Node(7)

tree.right.right = Node(14)
tree.right.right.left = Node(13)

def trimBST(tree, minVal, maxVal):

    if not tree:
        return

    tree.left = trimBST(tree.left, minVal, maxVal)
    tree.right = trimBST(tree.right, minVal, maxVal)

    if minVal <= tree.val <= maxVal:
        return tree

    if tree.val <= minVal:
        return tree.right

    if tree.val >= maxVal:
        return tree.left


new_tree = trimBST(tree, 5, 13)

