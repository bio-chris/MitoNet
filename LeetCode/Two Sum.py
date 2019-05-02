""""

Given an array of integers, return indices of the two numbers such that they add up to a specific target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

Example:

Given nums = [2, 7, 11, 15], target = 9,

Because nums[0] + nums[1] = 2 + 7 = 9,
return [0, 1].

"""

# Solution below correct, but two slow because it goes through all values in every loop (should only look at values that
# have a different index (not yet accepted by LeetCode)

nums = [3,2,4]
target = 6

index=0
stop = False
for i in nums:

    index2=0

    for i2 in nums:

        if index != index2:

            if i + i2 == target:
                #print(index,index2)
                stop = True

        index2+=1

    if stop == True:
        break

    index+=1



def twoSum(nums,target):
    index = 0
    stop = False
    for i in nums:

        index2 = 0

        for i2 in nums[index+1:]:

            print(i, i2)
            #print(index, index2)

            if index != index2:

                if i + i2 == target:
                    return [index, index2]
                    stop = True
                    break

            index2 += 1

        if stop == True:
            break

        index += 1

print(twoSum([3,2,4],6))


x = [1,2,3]

index = 0
for i in x:

    for i2 in x[index+1:]:
        print(i, i2)

    index+=1

