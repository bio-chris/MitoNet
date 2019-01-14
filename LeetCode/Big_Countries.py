
"""

country is big if it has an area of bigger than 3 million square km or a population of more than 25 million.

Write a SQL solution to output big countries' name, population and area.

For example, according to the above table, we should output:

"""


# pandas solution


import pandas as pd

table = pd.DataFrame([['Afghanistan', 'Asia', 652230, 25500100, '20343000000'],
                      ['Albania', 'Europe', 28748, 2831741, '12960000000'],
                      ['Algeria', 'Africa', 2381741, 37100000, '188681000000'],
                      ['Andorra', 'Europe', 468, 78115, '3712000000'],
                      ['Angola', 'Africa', 1246700, 20609294, '100990000000']],
                     columns=["name", "continent", "area", "population", "gdp"])



print(table[(table["area"] > 3000000) | (table["population"] > 25000000)])

