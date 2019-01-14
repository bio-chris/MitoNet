"""

This code allows to transform python files into executable files 

To use this code, open command line in directory in which the python file that you want to convert is stored. 

Then type: python Setup_cx_Freeze.py build 

"""

from cx_Freeze import setup, Executable

setup(name='test',
      version='0.1',
      description='Stuff ...',
      # variable needs to be changed according to file one wishes to convert 
      executables = [Executable("test_exec.py")])




