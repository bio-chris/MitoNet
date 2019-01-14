from timeit import default_timer as timer

# sequential search

def ordered_seq_search(arr, item):

    pos = 0
    found = False
    stopped = False

    while pos < len(arr) and not found and not stopped:

        if arr[pos] == item:
            found = True

        else:

            if arr[pos] > item:
                stopped = True

            pos+=1

    return found


arr = [1,2,3,4,5]

"""
start = timer()
print(ordered_seq_search(arr, 3))
end = timer()
print(end-start)
"""

# binary search

# only usable on sorted array
def binary_search(arr, item):

    first = 0
    last = len(arr)-1

    found = False

    while first <= last and not found:

        mid = int((first+last)/2)

        if arr[mid] == item:
            found = True

        else:
            if item < arr[mid]:
                last = mid-1
            else:
                last = mid+1


    return found

"""
start = timer()
print(binary_search(arr, 3))
end = timer()
print(end-start)
"""


def rec_bin_search(arr, item):

    if len(arr) == 0:
        return False

    else:

        mid = int(len(arr)/2)

        if arr[mid] == item:
            return True

        else:

            if item < arr[mid]:
                return rec_bin_search(arr[:mid], item)

            else:
                return rec_bin_search(arr[mid+1:], item)

"""
start = timer()
print(rec_bin_search(arr, 3))
end = timer()
print(end-start)
"""

# hashing


class HashTable():

    def __init__(self, size):
        self.size = size
        self.slots = [None]*self.size
        self.data = [None]*self.size


    def put(self, key, data):

        hashvalue = self.hashfunction(key, len(self.slots))

        if self.slots[hashvalue] == None:
            self.slots[hashvalue] = key
            self.data[hashvalue] = data

        else:

            if self.slots[hashvalue] == key:
                self.data[hashvalue] = data

            else:

                nextslot = self.rehash(hashvalue, len(self.slots))

                while self.slots[nextslot] != None and self.slots[nextslot] != key:
                    nextslot = self.rehash(nextslot, len(self.slots))

                if self.slots[nextslot] == None:
                    self.slots[nextslot] = key
                    self.data[nextslot] = data

                else:
                    self.data[nextslot] = data

    # remainder method
    def hashfunction(self, key, size):

        return key%size

    def rehash(self, oldhash, size):

        return (oldhash+1)%size


    def get(self, key):

        startslot = self.hashfunction(key, len(self.slots))
        data = None
        stop = False
        found = False
        position = startslot

        while self.slots[position] != None and not found and not stop:

            if self.slots[position] == key:
                found = True
                data = self.data[position]

            else:
                position = self.rehash(position, len(self.slots))

                if position == startslot:
                    stop = True


        return data


    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, data):
        self.put(key, data)


# binary search interview problem


def binary_search(arr, item):

    found = False

    if item == arr[int(len(arr)/2)]:
        found = True

    else:

        while not found and arr != []:

            mid = int(len(arr)/2)

            if item == arr[mid]:
                found = True
                break

            elif item > arr[mid]:
                arr = arr[mid+1:]

            else:
                arr = arr[:mid]

    return found


l = [1,2,3,4,5,6]
"""
start = timer()
print(binary_search(l, 5))
end = timer()
print(end-start)
"""










