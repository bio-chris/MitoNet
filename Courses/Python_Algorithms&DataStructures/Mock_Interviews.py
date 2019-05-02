

# e-commerce on-site question 1

prices = [12,11,15,3,10]

def max_profit(l):

    x_min = min(l)

    min_index = prices.index(x_min)

    x_max = max(prices[min_index:])

    max_profit = x_max-x_min

    return max_profit

#print(max_profit(prices))

# on-site question 2

def int_prod(l):

    new_l = []
    for count, n in enumerate(l):

        new_el = 1
        for count2, n2 in enumerate(l):

            if count != count2:
                new_el *= l[count2]

        new_l.append(new_el)

    return new_l

l = [1,2,3,4]

print(int_prod(l))




