"""

Class Control

Contains all functions that are necessary for the entire program

Class Advanced Mode

Contains all functions necessary for the advanced mode

Class Easy Mode

Contains all function necessary for the easy mode

"""


from tkinter import *
import tkinter.font
import tkinter.messagebox
import tkinter.filedialog
import os
from skimage.measure import regionprops, label
import webbrowser
import pandas as pd
import numpy as np
import cv2


# GUI
####################

class Control():

    """
    Adds functions that can be accessed from all windows in the program
    """

    def __init__(self):
        pass

    # close currently open window
    def close_window(self, window):

        window.destroy()

    # opens link to documentation of how to use the program
    def help(self, window):

        webbrowser.open_new("https://github.com/bio-chris/MitoSegNet")

    # go to main window
    def go_back(self, current_window, root):

        current_window.quit()
        root.quit()

        try:
            root = start_window()
            root.mainloop()

        except:
            pass

    # open new window with specified width and height
    def new_window(self, window, title, width, height):

        window.title(title)

        window.minsize(width=int(width/2), height=int(height/2))
        window.geometry(str(width)+"x"+str(height)+"+0+0")

    # adds menu to every window, which contains the above functions close_window, help and go_back
    def small_menu(self, window):

        menu = Menu(window)
        window.config(menu=menu)

        submenu = Menu(menu)
        menu.add_cascade(label="Menu", menu=submenu)

        submenu.add_command(label="Help", command=lambda: self.help(window))

        # creates line to separate group items
        submenu.add_separator()

        submenu.add_command(label="Go Back", command=lambda: self.go_back(window, root))
        submenu.add_command(label="Exit", command=lambda: self.close_window(window))
  
    def place_text(self, window, text, x, y, height, width):

        if height is None or width is None:
            Label(window, text=text, bd=1).place(bordermode=OUTSIDE, x=x, y=y)
        else:
            Label(window, text=text, bd=1).place(bordermode=OUTSIDE, x=x, y=y, height=height, width=width)

    def place_button(self, window, text, func, x, y, height, width):

        Button(window, text=text, command=func).place(bordermode=OUTSIDE, x=x, y=y, height=height, width=width)

    def place_entry(self, window, text, x, y, height, width):

        Entry(window, textvariable=text).place(bordermode=OUTSIDE, x=x, y=y, height=height, width=width)



if __name__ == '__main__':

    """
    Main (starting) window
    """

    control_class = Control()

    root = Tk()

    control_class.new_window(root, "MitoSegNet Segmentation Measurements", 500, 200)
    control_class.small_menu(root)

    imgpath = StringVar(root)
    labpath = StringVar(root)

    def askopenimgs():
        set_imgpath = tkinter.filedialog.askdirectory(parent=root, title='Choose a directory')
        imgpath.set(set_imgpath)

    def askopenlabels():
        set_labpath = tkinter.filedialog.askdirectory(parent=root, title='Choose a directory')
        labpath.set(set_labpath)

    #### browse for raw image data

    control_class.place_text(root, "Select directory in which 8-bit raw images are stored", 15, 20, None, None)
    control_class.place_button(root, "Browse", askopenimgs, 435, 50, 30, 50)
    control_class.place_entry(root, imgpath, 25, 50, 30, 400)

    #### browse for pretrained model

    control_class.place_text(root, "Select directory in which segmented images are stored", 15, 90, None, None)
    control_class.place_button(root, "Browse", askopenlabels, 435, 120, 30, 50)
    control_class.place_entry(root, labpath, 25, 120, 30, 400)

    def get_measurements():

        img_list = os.listdir(imgpath.get())

        dataframe = pd.DataFrame(columns=["Image", "Measurement", "Average", "Median", "Standard Deviation",
                                          "Standard Error", "Minimum", "Maximum", "N"])

        n = 0
        for i, img in enumerate(img_list):

            read_img = cv2.imread(imgpath.get() + os.sep + img, -1)
            read_lab = cv2.imread(labpath.get() + os.sep + img, cv2.IMREAD_GRAYSCALE)

            labelled_img = label(read_lab)

            labelled_img_props = regionprops(label_image=labelled_img, intensity_image=read_img, coordinates='xy')

            area = [obj.area for obj in labelled_img_props]
            minor_axis_length = [obj.minor_axis_length for obj in labelled_img_props]
            major_axis_length = [obj.major_axis_length for obj in labelled_img_props]
            eccentricity = [obj.eccentricity for obj in labelled_img_props]
            perimeter = [obj.perimeter for obj in labelled_img_props]
            solidity = [obj.solidity for obj in labelled_img_props]
            mean_int = [obj.mean_intensity for obj in labelled_img_props]
            max_int = [obj.max_intensity for obj in labelled_img_props]
            min_int = [obj.min_intensity for obj in labelled_img_props]



            def add_to_dataframe(measure_str, measure, n):

                dataframe.loc[n] = [img] + [measure_str, np.average(measure), np.median(measure), np.std(measure),
                                            np.std(measure) / np.sqrt(len(measure)), np.min(measure), np.max(measure),
                                            len(measure)]


            meas_str_l = ["Area", "Minor Axis Length", "Major Axis Length", "Eccentricity", "Perimeter", "Solidity",
                          "Mean Intensity", "Max Intensity", "Min Intensity"]
            meas_l = [area, minor_axis_length, major_axis_length, eccentricity, perimeter, solidity, mean_int, max_int,
                      min_int]

            for m_str, m in zip(meas_str_l, meas_l):

                add_to_dataframe(m_str, m, n)
                n+=1

        parentDirectory = os.path.abspath(os.path.join(imgpath.get(), os.pardir))

        folder_name = parentDirectory.split(os.sep)[-1]

        dataframe.to_csv(parentDirectory + os.sep + folder_name+ "_Analysis_Table.csv")
        dataframe.to_excel(parentDirectory + os.sep + folder_name + "_Analysis_Table.xlsx")

        tkinter.messagebox.showinfo("Done", "Table generated", parent=root)


    control_class.place_button(root, "Get Measurements", get_measurements, 200, 160, 30, 110)

    root.mainloop()



