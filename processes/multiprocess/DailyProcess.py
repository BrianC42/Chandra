'''
Created on Jul 27, 2023

@author: Brian
'''

import sys

import tensorflow as tf
import keras

import pandas as pd

from DailyProcessUI import DailyProcessUI

if __name__ == '__main__':
    print("The version of python is {}".format(sys.version))
    print("The version of tensorflow installed is {}".format(tf.__version__))
    print("\tThere are {} GPUs available to tensorflow: {}".format(len(tf.config.list_physical_devices('GPU')), tf.config.list_physical_devices('GPU')))
    print("The version of keras installed is {}".format(keras.__version__))
    print("\nDisplaying the process control panel\n")
    
    #Set print parameters for Pandas dataframes 
    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', 20)
    
    UI = DailyProcessUI()
                    
    print ("\nAll requested processes have completed")