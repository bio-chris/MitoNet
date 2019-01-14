

# scenario 1

cities = ['Marseille', 'Amsterdam', 'New York', 'London']

# the bad way
i = 0
for city in cities:
    print(i, city)
    i+=1

# the good way

for count, city in enumerate(cities):
    print(count, city)


# scenario 2

x = 10
y = -10
print('Before: x = %d, y=%d' % (x,y))

# the bad way

#tmp = y
#y = x
#x = tmp

# the good way

# swaps variable values
x,y = y,x

print('After: x = %d, y=%d' % (x,y))

# scenario 4

# find key in dictionary even if possibly not existing

ages = {'john': 38, 'laura': 28}

age = ages.get('Dick', 'Unknown')
print(age)

# scenario 5

# else after for loop

needle = 'd'
haystack = ['a', 'b', 'c', 'd']

for letter in haystack:
    if needle == letter:
        print("Found")
        break
else:
    print("Not found")



