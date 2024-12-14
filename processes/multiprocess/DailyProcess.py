'''
Created on Jul 27, 2023

@author: Brian
'''
'''
import os
import glob
import json
import time
import tkinter as tk
from configuration import read_config_json
from Workbooks import investments, optionTrades
from multiProcess import updateMarketData 
import tensorflow as tf
'''

import sys

import tensorflow as tf
import keras

import pandas as pd

from DailyProcessUI import DailyProcessUI
'''
import re
import datetime as dt
from configuration import get_ini_data
from GoogleSheets import googleSheet
from pretrainedModels import rnnCategorization, rnnPrediction
PROCESS_CONFIGS = "processes"
MODEL_CONFIGS = "models"

PROCESS_ID = "name"
PROCESS_DESCRIPTION = "Description"
RUN = "run"
RUN_COL = 2
MODEL_FILE = "file"
MODEL = "model"
OUTPUT_FIELDS = "Outputs"
CONFIG = "json"

MARKET_DATA_UPDATE = "Market data update"
MARK_TO_MARKET = "Mark to market"
DERIVED_MARKET_DATA = "Calculate derived market data"
SECURED_PUTS = "Secured puts"
COVERED_CALLS = "Covered calls"
OPTION_TRADES = "Option trade review"
BOLLINGER_BAND_PREDICTION = "Bollinger Band"
MACD_TREND_CROSS = "MACD Trend"
'''

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