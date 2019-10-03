import os

path = "C:/Users/Christian (Home)/Pictures/iPhone_Pictures_25032019/Reorganised"
folder = "Weekend_Trento_2203-230319"

os.mkdir(path + os.sep + folder + os.sep + "aae")

for files in os.listdir(path + os.sep + folder):

    if ".AAE" in files:
        os.rename(path + os.sep + folder + os.sep + files, path + os.sep + folder + os.sep + "aae" + os.sep + files)


