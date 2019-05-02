from tkinter import *
import tkinter.font
import tkinter.messagebox
import tkinter.filedialog
import os
import cv2
from Create_Project import *
from Training_DataGenerator_GUI import *
from MitoSegNet_GUI import *


# GUI
####################

def close_window(window):
    window.destroy()


def help():
    tkinter.messagebox.showinfo("Help", "A detailed tutorial on how to use this software can be found under ...")


def go_back(current_window, root):
    current_window.quit()
    root.quit()

    root = start_window()
    root.mainloop()


def new_window(window, title, width, height):
    window.title(title)

    window.minsize(width=int(width / 2), height=int(height / 2))
    window.geometry(str(width) + "x" + str(height) + "+0+0")


### menu section

def small_menu(window):
    menu = Menu(window)
    window.config(menu=menu)

    submenu = Menu(menu)
    menu.add_cascade(label="Menu", menu=submenu)

    submenu.add_command(label="Help", command=help)

    # creates line to separate group items
    submenu.add_separator()

    submenu.add_command(label="Exit", command=lambda: close_window(window))
    submenu.add_command(label="Go Back", command=lambda: go_back(window, root))


def get_image_info(path):
    tiles_path = path + os.sep + "train" + os.sep + "image"
    tiles_list = os.listdir(tiles_path)

    images_path = path + os.sep + "train" + os.sep + "RawImgs" + os.sep + "image"
    images_list = os.listdir(images_path)

    for tiles in tiles_list:
        tile = cv2.imread(tiles_path + os.sep + tiles, cv2.IMREAD_GRAYSCALE)
        tile_size = tile.shape[0]
        break

    for images in images_list:
        image = cv2.imread(images_path + os.sep + images, cv2.IMREAD_GRAYSCALE)
        y = image.shape[0]
        x = image.shape[1]
        break

    return tile_size, y, x, tiles_list, images_list


# window 1: start new mitosegnet project
def start_new_project():
    root.quit()

    start_root = Tk()

    new_window(start_root, "MitoSegNet Navigator - Start new project", 600, 300)

    dirpath = StringVar()
    orgpath = StringVar()
    labpath = StringVar()

    def askopendir():
        set_dirpath = tkinter.filedialog.askdirectory(parent=start_root, title='Choose a directory')
        dirpath.set(set_dirpath)

    def askopenorg():
        set_orgpath = tkinter.filedialog.askdirectory(parent=start_root, title='Choose a directory')
        orgpath.set(set_orgpath)

    def askopenlab():
        set_labpath = tkinter.filedialog.askdirectory(parent=start_root, title='Choose a directory')
        labpath.set(set_labpath)

    ### menu section
    ######
    small_menu(start_root)
    ######

    ####
    browse_label = Label(start_root, text="Select directory in which MitoSegNet project files should be generated",
                         bd=1)
    browse_label.place(bordermode=OUTSIDE, x=30, y=20)

    browsedir_button = Button(start_root, text='Browse', command=askopendir)
    browsedir_button.place(bordermode=OUTSIDE, x=440, y=40, height=30, width=50)

    entry = Entry(start_root, textvariable=dirpath)
    entry.place(bordermode=OUTSIDE, x=30, y=40, height=30, width=400)
    ####

    ####
    browse_org_label = Label(start_root, text="Select directory in which 8-bit raw images are stored",
                             bd=1)
    browse_org_label.place(bordermode=OUTSIDE, x=30, y=80)

    browse_org_button = Button(start_root, text='Browse', command=askopenorg)
    browse_org_button.place(bordermode=OUTSIDE, x=440, y=100, height=30, width=50)

    entry_org = Entry(start_root, textvariable=orgpath)
    entry_org.place(bordermode=OUTSIDE, x=30, y=100, height=30, width=400)
    ####

    ####
    browse_lab_label = Label(start_root, text="Select directory in which ground truth (hand-labeled) images are stored",
                             bd=1)
    browse_lab_label.place(bordermode=OUTSIDE, x=30, y=140)

    browse_lab_button = Button(start_root, text='Browse', command=askopenlab)
    browse_lab_button.place(bordermode=OUTSIDE, x=440, y=160, height=30, width=50)

    entry_lab = Entry(start_root, textvariable=labpath)
    entry_lab.place(bordermode=OUTSIDE, x=30, y=160, height=30, width=400)

    ####

    def generate():

        str_dirpath = entry.get()
        str_orgpath = entry_org.get()
        str_labpath = entry_lab.get()

        create_project = CreateProject()

        copy = False
        if str_dirpath != "":
            cr_folders = create_project.create_folders(path=str_dirpath)

            if cr_folders == True:

                if str_orgpath != "" and str_labpath != "":
                    create_project.copy_data(path=str_dirpath, orgpath=str_orgpath, labpath=str_labpath)
                    copy = True

                else:
                    tkinter.messagebox.showinfo("Note", "You have not entered any paths")

            else:
                pass

        else:
            tkinter.messagebox.showinfo("Note", "You have not entered any path")

        if cr_folders == True and copy == True:

            tkinter.messagebox.showinfo("Done", "Generation of project folder and copying of files successful!")
            answer = tkinter.messagebox.askquestion("Done", "Do you want to continue with model training?")

            if answer == YES:

                cont_project()


            else:
                start_root.quit()

    generate_button = Button(start_root, text="Generate", command=generate)
    generate_button.place(bordermode=OUTSIDE, x=265, y=220, height=50, width=70)

    start_root.mainloop()


preprocess = Preprocess()


# global tile_size
# global tile_number


# window 3: creating augmented data
def cont_data(old_window):
    old_window.destroy()

    data_root = Tk()

    new_window(data_root, "MitoSegNet Data Augmentation", 450, 550)
    small_menu(data_root)

    dir_data_path = StringVar(data_root)

    # tile size and tile number list
    tkvar = StringVar(data_root)

    # tile size
    tile_size = IntVar(data_root)
    # tile number
    tile_number = IntVar(data_root)

    n_aug = IntVar(data_root)

    # augmentation operation parameters
    width_shift = DoubleVar(data_root)
    height_shift = DoubleVar(data_root)
    shear_range = DoubleVar(data_root)
    rotation_range = IntVar(data_root)
    zoom_range = DoubleVar(data_root)

    tkvar.set('')  # set the default option

    def askopendir():

        set_dir_data_path = tkinter.filedialog.askdirectory(parent=data_root, title='Choose a directory')
        dir_data_path.set(set_dir_data_path)

        print(dir_data_path.get())

        pr_list, val_List = preprocess.poss_tile_sizes(set_dir_data_path + os.sep + "train" + os.sep + "RawImgs")

        if set_dir_data_path != "":
            tkvar.set(list(pr_list)[0])  # set the default option
            choices = pr_list

            popupMenu = OptionMenu(data_root, tkvar, *choices)
            popupMenu.place(bordermode=OUTSIDE, x=30, y=90, height=30, width=300)

    # on change dropdown value
    def change_dropdown(*args):

        tile_inf = tkvar.get()

        l = (tile_inf.split(" "))

        tile_size.set(int(l[3]))
        tile_number.set(int(l[-1]))

        # print(tile_size.get(), tile_number.get())

    # link function to change dropdown (tile size and number)
    tkvar.trace('w', change_dropdown)

    ####
    browse_label = Label(data_root, text="Select MitoSegNet Project directory", bd=1)
    browse_label.place(bordermode=OUTSIDE, x=20, y=10)

    browsedir_button = Button(data_root, text='Browse', command=askopendir)
    browsedir_button.place(bordermode=OUTSIDE, x=390, y=30, height=30, width=50)

    entry = Entry(data_root, textvariable=dir_data_path)
    entry.place(bordermode=OUTSIDE, x=30, y=30, height=30, width=350)
    ####

    ####
    Label(data_root, text="Choose the tile size and corresponding tile number").place(bordermode=OUTSIDE, x=20, y=70)

    Label(data_root, text="Choose the number of augmentation operations").place(bordermode=OUTSIDE, x=20, y=130)

    entry_aug_n = Entry(data_root, textvariable=n_aug)
    entry_aug_n.place(bordermode=OUTSIDE, x=30, y=150, height=30, width=50)
    ####

    """

    shear_range=0.2,  # originally set to 0.1
            rotation_range=180,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            width_shift_range=0.2,
            height_shift_range=0.2,
            fill_mode='reflect')  # pixels outside boundary are set to 0

    """

    ####
    Label(data_root, text="Specify augmentation operations").place(bordermode=OUTSIDE, x=20, y=190)

    horizontal_flip = StringVar(data_root)
    hf_button = Checkbutton(data_root, text="Horizontal flip", variable=horizontal_flip, onvalue=True, offvalue=False)
    hf_button.place(bordermode=OUTSIDE, x=30, y=210, height=30, width=120)

    vertical_flip = StringVar(data_root)
    vf_button = Checkbutton(data_root, text="Vertical flip", variable=vertical_flip, onvalue=True, offvalue=False)
    vf_button.place(bordermode=OUTSIDE, x=150, y=210, height=30, width=120)

    ws_label1 = Label(data_root, text="Width shift range", bd=1).place(bordermode=OUTSIDE, x=30, y=240)
    ws_label2 = Label(data_root, text="(fraction of total width, if < 1, or pixels if >= 1)", bd=1).place(
        bordermode=OUTSIDE, x=30, y=260)

    ws_entry = Entry(data_root, textvariable=width_shift).place(bordermode=OUTSIDE, x=370, y=250, height=30, width=50)

    hs_label1 = Label(data_root, text="Height shift range", bd=1).place(bordermode=OUTSIDE, x=30, y=280)
    hs_label2 = Label(data_root, text="(fraction of total height, if < 1, or pixels if >= 1)", bd=1).place(
        bordermode=OUTSIDE, x=30, y=300)

    hs_entry = Entry(data_root, textvariable=height_shift).place(bordermode=OUTSIDE, x=370, y=290, height=30, width=50)

    shear_label = Label(data_root, text="Shear range (Shear intensity)", bd=1).place(bordermode=OUTSIDE, x=30, y=340)
    shear_entry = Entry(data_root, textvariable=shear_range).place(bordermode=OUTSIDE, x=370, y=330, height=30,
                                                                   width=50)

    rot_label = Label(data_root, text="Rotation range (Degree range for random rotations)", bd=1).place(
        bordermode=OUTSIDE, x=30, y=380)
    rot_entry = Entry(data_root, textvariable=rotation_range).place(bordermode=OUTSIDE, x=370, y=370, height=30,
                                                                    width=50)

    zoom_label = Label(data_root, text="Zoom range (Range for random zoom)", bd=1).place(
        bordermode=OUTSIDE, x=30, y=420)
    zoom_entry = Entry(data_root, textvariable=zoom_range).place(bordermode=OUTSIDE, x=370, y=410, height=30,
                                                                 width=50)

    ####

    check_weights = StringVar(data_root)
    weights_button = Checkbutton(data_root, text="Create weight map", variable=check_weights,  # command=cb,
                                 onvalue=True, offvalue=False)
    weights_button.place(bordermode=OUTSIDE, x=30, y=450, height=30, width=150)

    def generate_data():

        if dir_data_path.get() != "":

            if int(horizontal_flip.get()) == 1:
                hf = True
            else:
                hf = False

            if int(vertical_flip.get()) == 1:
                vf = True
            else:
                vf = False

            preprocess.splitImgs(dir_data_path.get(), tile_size.get(), tile_number.get())

            aug = Augment(dir_data_path.get(), shear_range.get(), rotation_range.get(), zoom_range.get(),
                          hf, vf, width_shift.get(), height_shift.get())

            if int(check_weights.get()) == 1:
                wmap = True
            else:
                wmap = False

            aug.start_augmentation(imgnum=n_aug.get(), wmap=wmap)
            aug.splitMerge(wmap=wmap)

            mydata = Create_npy_files(dir_data_path.get())

            mydata.create_train_data(wmap, tile_size.get(), tile_size.get())
            # mydata.check_class_balance()

            tkinter.messagebox.showinfo("Done", "Augmented data successfully generated")

        else:
            tkinter.messagebox.showinfo("Error", "Entries missing or not correct")

    gen_data_button = Button(data_root, text='Start data augmentation', command=generate_data)
    gen_data_button.place(bordermode=OUTSIDE, x=150, y=500, height=30, width=150)


def cont_training(old_window):
    old_window.destroy()

    cont_training = Tk()

    new_window(cont_training, "MitoSegNet Navigator - Training", 500, 400)
    small_menu(cont_training)

    dir_data_path_train = StringVar(cont_training)
    epochs = IntVar(cont_training)
    balancer = DoubleVar(cont_training)
    popup_var = StringVar(cont_training)
    popup_newex_var = StringVar(cont_training)
    model_name = StringVar(cont_training)
    use_weight_map = StringVar(cont_training)

    def askopendir_train():
        set_dir_data_path = tkinter.filedialog.askdirectory(parent=cont_training, title='Choose a directory')
        dir_data_path_train.set(set_dir_data_path)

        mydata = Create_npy_files(dir_data_path_train.get())

        try:

            zero_perc, fg_bg_ratio = mydata.check_class_balance()

            inf1_label = Label(cont_training, text="Average percentage of background pixels in augmented label data: " +
                                                   str(round(zero_perc * 100, 2)), bd=1).place(bordermode=OUTSIDE, x=30,
                                                                                               y=290)

            inf2_label = Label(cont_training,
                               text="Foreground to background pixel ratio: 1 to " + str(fg_bg_ratio) + " " * 30,
                               bd=1).place(bordermode=OUTSIDE, x=30, y=310)

            popup_newex_var.set("New")
            popupMenu_new_ex = OptionMenu(cont_training, popup_newex_var, *set(["New", "Existing"]))
            popupMenu_new_ex.place(bordermode=OUTSIDE, x=30, y=90, height=30, width=100)

            weight_images = os.listdir(dir_data_path_train.get() + os.sep + "aug_weights")

            if len(weight_images) == 0:

                weights_label = Label(cont_training,
                                      text="No weight map images detected.",
                                      bd=1).place(bordermode=OUTSIDE, x=30, y=170, height=30, width=180)

                use_weight_map.set(0)

            else:

                hf_button = Checkbutton(cont_training, text="Use weight map", variable=use_weight_map, onvalue=True,
                                        offvalue=False)
                hf_button.place(bordermode=OUTSIDE, x=30, y=170, height=30, width=120)

                bal_label = Label(cont_training, text="Class balance weight factor", bd=1).place(bordermode=OUTSIDE,
                                                                                                 x=30, y=200)
                bal_entry = Entry(cont_training, textvariable=balancer).place(bordermode=OUTSIDE, x=200, y=195,
                                                                              height=30, width=50)

        except:

            err_label = Label(cont_training, text="Error: Please choose the MitoSegNet Project directory", bd=1).place(
                bordermode=OUTSIDE, width=500, height=30, x=20, y=290)

        """
        inf3_label = Label(cont_training, text="To get foreground to background ratio of 1 to 1,\nset class balance "
                                               "weight factor to " + str(round(1/fg_bg_ratio, 3)), bd=1).place(
            bordermode=OUTSIDE, x=30, y=460)
        """

    ####
    browse_label = Label(cont_training, text="Select MitoSegNet Project directory", bd=1)
    browse_label.place(bordermode=OUTSIDE, x=20, y=10)

    browsedir_button = Button(cont_training, text='Browse', command=askopendir_train)
    browsedir_button.place(bordermode=OUTSIDE, x=440, y=30, height=30, width=50)

    entry = Entry(cont_training, textvariable=dir_data_path_train)
    entry.place(bordermode=OUTSIDE, x=30, y=30, height=30, width=400)
    ####

    new_ex_label = Label(cont_training, text="Train new or existing model", bd=1).place(bordermode=OUTSIDE, x=20, y=70)

    def change_dropdown_newex(*args):

        if dir_data_path_train.get() != '':

            if popup_newex_var.get() == "New":

                model_name.set("")
                model_entry = Entry(cont_training, textvariable=model_name)
                model_entry.place(bordermode=OUTSIDE, x=300, y=90, height=25, width=150)

                model_entry_label = Label(cont_training, text="Enter model name\n(without file extension)",
                                          bd=1).place(bordermode=OUTSIDE, x=150, y=90, height=25)

            else:

                file_list = os.listdir(dir_data_path_train.get())

                found = False
                for files in file_list:
                    if ".hdf5" in files:
                        model_label = Label(cont_training, text="Found " + files + " " * 30,
                                            bd=1).place(bordermode=OUTSIDE, x=150, y=85, height=35)
                        found = True

                        model_name.set(files)

                if found == False:
                    model_label = Label(cont_training, text="No existing model file found" + " " * 40,
                                        bd=1).place(bordermode=OUTSIDE, x=150, y=90, height=25)

        # print(model_name.get())

    popup_newex_var.trace('w', change_dropdown_newex)

    ep_label = Label(cont_training, text="Number of epochs", bd=1).place(bordermode=OUTSIDE, x=30, y=140)
    ep_entry = Entry(cont_training, textvariable=epochs).place(bordermode=OUTSIDE, x=140, y=135, height=30,
                                                               width=50)

    popup_var.set("GPU")
    bal_label = Label(cont_training, text="Train on", bd=1).place(bordermode=OUTSIDE, x=30, y=230)
    popupMenu_train = OptionMenu(cont_training, popup_var, *set(["GPU", "CPU"]))
    popupMenu_train.place(bordermode=OUTSIDE, x=30, y=250, height=30, width=100)

    def change_dropdown(*args):
        pass
        # print(popup_var.get())

    popup_var.trace('w', change_dropdown)

    def start_training():

        if dir_data_path_train.get() != "" and use_weight_map.get() != "":

            # print(use_weight_map.get())

            if int(use_weight_map.get()) == 1:
                weight_map = True
            else:
                weight_map = False

            tile_size, y, x, tiles_list, images_list = get_image_info(dir_data_path_train.get())

            train_mitosegnet = MitoSegNet(dir_data_path_train.get(), img_rows=tile_size, img_cols=tile_size,
                                          org_img_rows=y, org_img_cols=x)

            set_gpu_or_cpu = gpu_or_cpu(popup_var.get())
            set_gpu_or_cpu.ret_mode()

            # def train(self, epochs, wmap, vbal):
            train_mitosegnet.train(epochs.get(), weight_map, balancer.get(), model_name.get())

            tkinter.messagebox.showinfo("Done", "Training completed")

        else:

            tkinter.messagebox.showinfo("Error", "Entries missing or not correct")

    gen_data_button = Button(cont_training, text='Start training', command=start_training)
    gen_data_button.place(bordermode=OUTSIDE, x=200, y=340, height=30, width=100)


def cont_prediction(old_window):
    old_window.destroy()

    cont_prediction_window = Tk()

    new_window(cont_prediction_window, "MitoSegNet Navigator - Prediction", 500, 350)
    small_menu(cont_prediction_window)

    dir_data_path_prediction = StringVar(cont_prediction_window)
    popup_var = StringVar(cont_prediction_window)
    batch_var = StringVar(cont_prediction_window)
    model_name = StringVar(cont_prediction_window)
    dir_data_path_test_prediction = StringVar(cont_prediction_window)
    found = IntVar()
    found.set(0)

    def askopendir_pred():

        set_dir_data_path = tkinter.filedialog.askdirectory(parent=cont_prediction_window, title='Choose a directory')
        dir_data_path_prediction.set(set_dir_data_path)

        if dir_data_path_prediction.get() != "":

            file_list = os.listdir(dir_data_path_prediction.get())

            for files in file_list:
                if ".hdf5" in files:
                    model_label = Label(cont_prediction_window, text="Found " + files + " " * 30,
                                        bd=1).place(bordermode=OUTSIDE, x=40, y=60, height=35)
                    # found = True
                    found.set(1)

                    model_name.set(files)

            if found.get() == 0:
                found_label_pred = Label(cont_prediction_window, text="No model found", bd=1)
                found_label_pred.place(bordermode=OUTSIDE, x=40, y=70)

    ####
    browse_label_pred = Label(cont_prediction_window, text="Select MitoSegNet Project directory", bd=1)
    browse_label_pred.place(bordermode=OUTSIDE, x=20, y=10)

    browsedir_button_pred = Button(cont_prediction_window, text='Browse', command=askopendir_pred)
    browsedir_button_pred.place(bordermode=OUTSIDE, x=440, y=30, height=30, width=50)

    entry = Entry(cont_prediction_window, textvariable=dir_data_path_prediction)
    entry.place(bordermode=OUTSIDE, x=30, y=30, height=30, width=400)

    batch_label_pred = Label(cont_prediction_window, text="Apply model prediction on one folder or multiple folders?",
                             bd=1)
    batch_label_pred.place(bordermode=OUTSIDE, x=20, y=100)

    batch_var.set("One folder")
    popupMenu_batch_pred = OptionMenu(cont_prediction_window, batch_var, *set(["One folder", "Multiple folders"]))
    popupMenu_batch_pred.place(bordermode=OUTSIDE, x=30, y=120, height=30, width=130)

    def askopendir_test_pred():

        set_dir_data_path_test = tkinter.filedialog.askdirectory(parent=cont_prediction_window,
                                                                 title='Choose a directory')
        dir_data_path_test_prediction.set(set_dir_data_path_test)

    def change_dropdown_batch_test_pred(*args):

        if batch_var.get() == "One folder":

            browse_label_test_pred = Label(cont_prediction_window,
                                           text="Select folder containing 8-bit images to be segmented" + " " * 30,
                                           bd=1)
            browse_label_test_pred.place(bordermode=OUTSIDE, x=20, y=155)

        else:

            browse_label_test_pred = Label(cont_prediction_window,
                                           text="Select folder containing sub-folders with 8-bit images to be segmented",
                                           bd=1)
            browse_label_test_pred.place(bordermode=OUTSIDE, x=20, y=155)

    browsedir_button_test_pred = Button(cont_prediction_window, text='Browse', command=askopendir_test_pred)
    browsedir_button_test_pred.place(bordermode=OUTSIDE, x=440, y=175, height=30, width=50)

    browse_label_test_pred = Label(cont_prediction_window,
                                   text="Select folder containing 8-bit images to be segmented" + " " * 30, bd=1)
    browse_label_test_pred.place(bordermode=OUTSIDE, x=20, y=155)

    entry_test = Entry(cont_prediction_window, textvariable=dir_data_path_test_prediction)
    entry_test.place(bordermode=OUTSIDE, x=30, y=175, height=30, width=400)

    batch_var.trace('w', change_dropdown_batch_test_pred)

    def start_prediction():

        if dir_data_path_prediction.get() != "" and found.get() == 1 and dir_data_path_test_prediction.get() != "":

            tile_size, y, x, tiles_list, images_list = get_image_info(dir_data_path_prediction.get())

            pred_mitosegnet = MitoSegNet(dir_data_path_prediction.get(), img_rows=tile_size, img_cols=tile_size,
                                         org_img_rows=y, org_img_cols=x)

            set_gpu_or_cpu = gpu_or_cpu(popup_var.get())
            set_gpu_or_cpu.ret_mode()

            if batch_var.get() == "One folder":

                if not os.path.lexists(dir_data_path_test_prediction.get() + os.sep + "Prediction"):
                    os.mkdir(dir_data_path_test_prediction.get() + os.sep + "Prediction")

                pred_mitosegnet.predict(dir_data_path_test_prediction.get(), False, tile_size,
                                        len(tiles_list) / len(images_list), model_name.get())

            else:

                for subfolders in os.listdir(dir_data_path_test_prediction.get()):

                    if not os.path.lexists(
                            dir_data_path_test_prediction.get() + os.sep + subfolders + os.sep + "Prediction"):
                        os.mkdir(dir_data_path_test_prediction.get() + os.sep + subfolders + os.sep + "Prediction")

                    pred_mitosegnet.predict(dir_data_path_test_prediction.get() + os.sep + subfolders, False, tile_size,
                                            len(tiles_list) / len(images_list), model_name.get())

            tkinter.messagebox.showinfo("Done", "Prediction successful! Check " + dir_data_path_test_prediction.get() +
                                        os.sep + "Prediction" + " for segmentation results")

        else:

            tkinter.messagebox.showinfo("Error", "Entries not completed")

    popup_var.set("GPU")
    bal_label = Label(cont_prediction_window, text="Predict on", bd=1).place(bordermode=OUTSIDE, x=20, y=220)
    popupMenu_train = OptionMenu(cont_prediction_window, popup_var, *set(["GPU", "CPU"]))
    popupMenu_train.place(bordermode=OUTSIDE, x=30, y=240, height=30, width=100)

    def change_dropdown(*args):
        pass
        # print(popup_var.get())

    popup_var.trace('w', change_dropdown)

    # style = tkinter.font.Font(family='Helvetica', size=10, weight='bold')

    start_pred_button = Button(cont_prediction_window, text='Start prediction', command=start_prediction)
    # start_pred_button.config(font=("helvetica", 30, "bold italic"))
    start_pred_button.place(bordermode=OUTSIDE, x=200, y=290, height=30, width=110)

    # start_pred_button['font'] = style


# window 2: continue project: 3 option menu: create augmented data, train or predict
def cont_project():
    # print(root.state())
    # root.quit()

    cont_root = Tk()

    new_window(cont_root, "MitoSegNet Navigator - Continue", 300, 200)
    small_menu(cont_root)

    data_button = Button(cont_root, text="Create augmented data", command=lambda: cont_data(cont_root))
    train_button = Button(cont_root, text="Train model", command=lambda: cont_training(cont_root))
    pred_button = Button(cont_root, text="Model prediction", command=lambda: cont_prediction(cont_root))

    h = 50
    w = 150

    data_button.place(bordermode=OUTSIDE, x=87, y=10, height=h, width=w)
    train_button.place(bordermode=OUTSIDE, x=87, y=70, height=h, width=w)
    pred_button.place(bordermode=OUTSIDE, x=87, y=130, height=h, width=w)


# blank window
def start_window():
    root = Tk()

    new_window(root, "MitoSegNet Navigator - Start", 400, 400)

    small_menu(root)

    ### button section
    start_new_button = Button(root, text="Start new project", command=start_new_project)
    cont_button = Button(root, text="Continue working on\nexisting project", command=cont_project)

    start_new_button.place(bordermode=OUTSIDE, x=125, y=50, height=130, width=150)
    cont_button.place(bordermode=OUTSIDE, x=125, y=200, height=130, width=150)

    return root


if __name__ == '__main__':
    root = start_window()
    root.mainloop()



