"""

22/08/19

short script for simon

crawls through large dataset and collects data of interest

"""

import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)



table = pd.read_excel("20190820_MitoDynamicsLipidsPos_Results.xlsx", sheet_name="Results")

#print(table)


new_table = pd.DataFrame(columns=["Row Names", "drp-1 vs N2 fold change", "fzo-1 vs N2 fold change", "spg-7 vs N2 fold change",
                                  "Metabolite name", "#C", "#DB", "Welch N2 vs fzo-1 P-Value", "Welch N2 vs drp-1 P-Value",
                                  "Welch N2 vs spg-7 P-Value"])

temp_table = pd.DataFrame(columns=["drp1", "fzo-1", "N2", "spg-7"])


# rows / columns
#print(table.iloc[0:,1:7])

start = 1
end = 7


for phenotype in temp_table:

    val_table = table.iloc[2:, start:end]

    # 35 = metabolite name, 47 = alternative metabolite name, 48 = welch n2 vs fzo1, 51 = n2 vs drp1, 54 = n2 vs spg7

    #print(spg_p_table)

    add = []
    for index, row in val_table.iterrows():
        l = row.tolist()


        if l.count(np.NaN) <= 3:

            av = np.nanmean(l)

            add.append(av)

        else:
            add.append("-")

    temp_table[phenotype] = add

    start+=6
    end+=6

#print(temp_table)


mn_l = table.iloc[2:, 35].tolist()
amn_l = table.iloc[2:, 47].tolist()

#print(mn_l)
#print(amn_l)

fzo_p_l = table.iloc[2:, 48].tolist()
drp_p_l = table.iloc[2:, 51].tolist()
spg_p_l = table.iloc[2:, 54].tolist()




def fold_change(pval, val, cont):
    if pval < 0.05 and cont != "-":

        fc = val / cont

    else:
        fc = "-"

    return fc

def parse_string(val):

    tmp = val.split("(")[1].split(")")[0].split(":")

    l1 = []
    for i in tmp[0]:
        if i.isdigit():
            l1.append(i)

    l2 = []
    for i2 in tmp[1]:
        if i2.isdigit():
            l2.append(i2)

    sep = ""

    return sep.join(l1), sep.join(l2)


fc_f_l = []
fc_d_l = []
fc_s_l = []

new_mn_l = []

c_l = []
db_l = []

for (index, rows), mn_val, amn_val, fp_val, dp_val, sp_val in zip(temp_table.iterrows(), mn_l, amn_l, fzo_p_l, drp_p_l,
                                                                  spg_p_l):



    fc_f = fold_change(fp_val, rows["fzo-1"], rows["N2"])
    fc_d = fold_change(dp_val, rows["drp1"], rows["N2"])
    fc_s = fold_change(sp_val, rows["spg-7"], rows["N2"])

    fc_f_l.append(fc_f)
    fc_d_l.append(fc_d)
    fc_s_l.append(fc_s)


    if str(mn_val) == "nan":

        if str(amn_val) != "nan":

            new_mn_l.append(amn_val)

            c, db = parse_string(amn_val)

            c_l.append(c)
            db_l.append(db)

        else:
            new_mn_l.append("-")

            c_l.append("-")
            db_l.append("-")
    else:
        new_mn_l.append(mn_val)

        c, db = parse_string(mn_val)

        c_l.append(c)
        db_l.append(db)



new_table["Row Names"] = table.iloc[2:,0]


new_table["drp-1 vs N2 fold change"] = fc_d_l
new_table["fzo-1 vs N2 fold change"] = fc_f_l
new_table["spg-7 vs N2 fold change"] = fc_s_l

new_table["Metabolite name"] = new_mn_l

new_table["#C"] = c_l
new_table["#DB"] = db_l

new_table["Welch N2 vs fzo-1 P-Value"] = fzo_p_l
new_table["Welch N2 vs drp-1 P-Value"] = drp_p_l
new_table["Welch N2 vs spg-7 P-Value"] = spg_p_l


writer = pd.ExcelWriter("20190820_MitoDynamicsLipidsPos_SortedResults.xlsx", engine='xlsxwriter')
new_table.to_excel(writer, sheet_name="Sorted_Results")
writer.save()

