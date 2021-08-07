'''
Created on Jan 31, 2018

@author: Brian
'''
import pandas as pd
import numpy as np    

def simple_moving_average(df_data, value_label=None, \
        avg_interval=None, \
        SMA_data_label=None):
    
    df_data[SMA_data_label] = df_data[value_label].rolling(window=avg_interval, min_periods=1).mean()
    
    return (df_data)

def exponential_moving_average(df_data, value_label=None, \
        date_label=None, interval=None, \
        EMA_data_label=None):
    '''
    print ("\nEMA:...\ndf_data length:", len(df_data), \
        "\ndate label: ", date_label, \
        "\nvalue label: ", value_label, \
        "\nEMA data label: ", EMA_data_label, \
        "\ninterval:", interval, \
        df_data.head(2))
    '''
    
    if date_label in df_data.columns:
        df_data.index = pd.to_datetime(df_data.pop(date_label))

    # The following works but produces a setting on copy warning
    df_data[EMA_data_label] = df_data[value_label].ewm(span=interval, min_periods=1).mean()
    
    # print ("EMA calculated - label: ", EMA_data_label, "interval:", "interval:", interval, "\n", df_data.head(2))
    
    return (df_data)