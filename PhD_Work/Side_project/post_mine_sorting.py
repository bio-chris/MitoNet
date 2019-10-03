"""

intended to be used after table_miner.py

1)hich metabolites of drp1 and fzo1 are either (statistically significant) up or downregulated
2)which metabolites of spg7 are (stat. sign) up or downregulated
3)which metabolites are upregulated in both drp1 and fzo1, and downregulated in spg7 (or vice versa)

"""


import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)

table = pd.read_excel("20190820_MitoDynamicsLipidsPos_SortedResults.xlsx", sheet_name="Sorted_Results")




new_table_1 = pd.DataFrame(columns=["Row Names", "drp-1 vs N2 fold change", "fzo-1 vs N2 fold change", "Metabolite name",
                                  "#C", "#DB", "Welch N2 vs fzo-1 P-Value", "Welch N2 vs drp-1 P-Value"])


new_table_2 = pd.DataFrame(columns=["Row Names", "spg-7 vs N2 fold change", "Metabolite name", "#C", "#DB",
                                    "Welch N2 vs spg-7 P-Value",])

new_table_3 = pd.DataFrame(columns=["Row Names", "drp-1 vs N2 fold change", "fzo-1 vs N2 fold change", "spg-7 vs N2 fold change",
                                  "Metabolite name", "#C", "#DB", "Welch N2 vs fzo-1 P-Value", "Welch N2 vs drp-1 P-Value",
                                  "Welch N2 vs spg-7 P-Value"])

row_l = []
fc_f_l = []
fc_d_l = []
mn_l = []
c_l = []
db_l = []
fzo1_p = []
drp1_p = []

row_2 = []
fc_s_2 = []
mn_2 = []
c_2 = []
db_2 = []
spg7_p = []

row_3 = []
fc_f_3 = []
fc_d_3 = []
fc_s_3 = []
mn_3 = []
c_3 = []
db_3 = []
fzo1_p_3 = []
drp1_p_3 = []
spg7_p_3 = []



for index, row in table.iterrows():

    #print(row)
    #l = row.tolist()

    # 1
    ##########################

    if row["drp-1 vs N2 fold change"] != "-" and row["fzo-1 vs N2 fold change"] != "-":

        row_l.append(row["Row Names"])

        fc_f_l.append(row["fzo-1 vs N2 fold change"])
        fc_d_l.append(row["drp-1 vs N2 fold change"])

        mn_l.append(row["Metabolite name"])
        c_l.append(row["#C"])
        db_l.append(row["#DB"])
        fzo1_p.append(row["Welch N2 vs fzo-1 P-Value"])
        drp1_p.append(row["Welch N2 vs drp-1 P-Value"])

    ##########################

    # 2
    ##########################
    if row["spg-7 vs N2 fold change"] != "-":

        row_2.append(row["Row Names"])

        fc_s_2.append(row["spg-7 vs N2 fold change"])

        mn_2.append(row["Metabolite name"])
        c_2.append(row["#C"])
        db_2.append(row["#DB"])
        spg7_p.append(row["Welch N2 vs spg-7 P-Value"])

    ##########################


    # 3
    ##########################

    if row["drp-1 vs N2 fold change"] != "-" and row["fzo-1 vs N2 fold change"] != "-" and row["spg-7 vs N2 fold change"] != "-":

        add = False

        # case 1
        if row["drp-1 vs N2 fold change"] > 1 and row["fzo-1 vs N2 fold change"] > 1 and row["spg-7 vs N2 fold change"] < 1:

            add = True

        elif row["drp-1 vs N2 fold change"] < 1 and row["fzo-1 vs N2 fold change"] < 1 and row["spg-7 vs N2 fold change"] > 1:

            add = True


        if add == True:

            row_3.append(row["Row Names"])

            fc_f_3.append(row["fzo-1 vs N2 fold change"])
            fc_d_3.append(row["drp-1 vs N2 fold change"])
            fc_s_3.append(row["spg-7 vs N2 fold change"])

            mn_3.append(row["Metabolite name"])
            c_3.append(row["#C"])
            db_3.append(row["#DB"])
            fzo1_p_3.append(row["Welch N2 vs fzo-1 P-Value"])
            drp1_p_3.append(row["Welch N2 vs drp-1 P-Value"])
            spg7_p_3.append(row["Welch N2 vs spg-7 P-Value"])

    ##########################

new_table_1["Row Names"] = row_l
new_table_1["drp-1 vs N2 fold change"] = fc_d_l
new_table_1["fzo-1 vs N2 fold change"] = fc_f_l
new_table_1["Metabolite name"] = mn_l
new_table_1["#C"] = c_l
new_table_1["#DB"] = db_l
new_table_1["Welch N2 vs fzo-1 P-Value"] = fzo1_p
new_table_1["Welch N2 vs drp-1 P-Value"] = drp1_p


new_table_2["Row Names"] = row_2
new_table_2["spg-7 vs N2 fold change"] = fc_s_2
new_table_2["Metabolite name"] = mn_2
new_table_2["#C"] = c_2
new_table_2["#DB"] = db_2
new_table_2["Welch N2 vs spg-7 P-Value"] = spg7_p

new_table_3["Row Names"] = row_3
new_table_3["drp-1 vs N2 fold change"] = fc_d_3
new_table_3["fzo-1 vs N2 fold change"] = fc_f_3
new_table_3["spg-7 vs N2 fold change"] = fc_s_3
new_table_3["Metabolite name"] = mn_3
new_table_3["#C"] = c_3
new_table_3["#DB"] = db_3
new_table_3["Welch N2 vs fzo-1 P-Value"] = fzo1_p_3
new_table_3["Welch N2 vs drp-1 P-Value"] = drp1_p_3
new_table_3["Welch N2 vs spg-7 P-Value"] = spg7_p_3


writer = pd.ExcelWriter("20190820_MitoDynamicsLipidsPos_SortedResults.xlsx", engine='xlsxwriter')

table.to_excel(writer, sheet_name="Sorted_Results")

new_table_1 = new_table_1.sort_values(by=["drp-1 vs N2 fold change", "fzo-1 vs N2 fold change"])
new_table_2 = new_table_2.sort_values(by=["spg-7 vs N2 fold change"])

new_table_1.to_excel(writer, sheet_name="drp-1_fzo-1")
new_table_2.to_excel(writer, sheet_name="spg-7")
new_table_3.to_excel(writer, sheet_name="drp-1_fzo-1_up-down_spg-7")


writer.save()