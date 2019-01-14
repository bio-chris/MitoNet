import tkinter
from tkinter import messagebox
top = tkinter.Tk()
# Code to add widgets will go here...

def helloCallBack():
   messagebox.showinfo( "Hello Python", "Hello World")

B = tkinter.Button(top, text ="Hello", command = helloCallBack)

B.pack()

top.mainloop()

