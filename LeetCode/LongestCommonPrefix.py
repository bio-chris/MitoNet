""""

Write a function to find the longest common prefix string amongst an array of strings.

"""

array = ["endanger", "irregulgar", "undo", "tricycle", "triangle"]



from difflib import SequenceMatcher
from difflib import get_close_matches

"""
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


print(similar("triangle","tricycle"))
"""

print(get_close_matches('enda', array))

