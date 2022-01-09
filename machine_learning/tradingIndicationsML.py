'''
Created on Dec 27, 2021

@author: brian
'''
import sys
import os
import glob
import logging
import networkx as nx
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, StandardScaler

GM_HISTORY = "d:\\brian\\AI Projects\\tda\\market_analysis_data\\gm.csv"
GM_MODEL = "d:\\brian\\AI Projects\\models\\GM_mkt""d:\\brian\\AI Projects\\models\\GM_mkt"
MODEL_DATA = ["Close"]

def selectTrainingDataFields(df_data):
    
    np_data = np.array(df_data)
    return

def tradingIndications():
    print("\nUsing trained ML models to look for trading opportunities")
    
    ''' error handling '''
    try:
        err_txt = "*** An exception occurred collecting and selecting the data ***"
    
        if os.path.isfile(GM_HISTORY):
            df_data = pd.read_csv(GM_HISTORY)

            np_data = selectTrainingDataFields(df_data)
    
    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        if isinstance(exc_str, str):
            exc_txt = err_txt + "\n\t" + exc_str
        elif isinstance(exc_str, tuple):
            exc_txt = err_txt + "\n\t"
            for s in exc_str:
                exc_txt += " " + s
        logging.debug(exc_txt)
        sys.exit(exc_txt)

    return

if __name__ == '__main__':
    print ("Affirmative, Dave. I read you\n")
    tradingIndications()
    
