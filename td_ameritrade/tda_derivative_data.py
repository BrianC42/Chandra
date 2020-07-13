'''
Created on Jul 13, 2020

@author: Brian
'''
from numpy import NaN
from moving_average import simple_moving_average
from moving_average import exponential_moving_average

def add_trending_data(df_data):
    df_data.insert(loc=0, column='EMA12', value=NaN)
    df_data = exponential_moving_average(df_data[:], value_label="Close", interval=12, EMA_data_label='EMA12')
    df_data.insert(loc=0, column='EMA26', value=NaN)
    df_data = exponential_moving_average(df_data[:], value_label="Close", interval=26, EMA_data_label='EMA26')
    df_data.insert(loc=0, column='SMA20', value=NaN)
    df_data = simple_moving_average(df_data[:], value_label="Close", avg_interval=20, SMA_data_label='SMA20')
    return df_data

def add_change_data(df_data):
    df_data.insert(loc=0, column='1 day change', value=NaN)
    df_data.insert(loc=0, column='5 day change', value=NaN)
    df_data.insert(loc=0, column='10 day change', value=NaN)
    df_data.insert(loc=0, column='20 day change', value=NaN)
    idx = 1
    while idx < len(df_data):
        if not df_data.loc[idx, "Close"] == 0:
            if idx + int(1) < len(df_data):
                df_data.loc[idx, '1 day change'] = (df_data.loc[idx + int(1), "Close"] - df_data.loc[idx, "Close"]) / df_data.loc[idx, "Close"]                    
            if idx + int(5) < len(df_data):
                df_data.loc[idx, '5 day change'] = (df_data.loc[idx + int(5), "Close"] - df_data.loc[idx, "Close"]) / df_data.loc[idx, "Close"]                    
            if idx + int(10) < len(df_data):
                df_data.loc[idx, '10 day change'] = (df_data.loc[idx + int(10), "Close"] - df_data.loc[idx, "Close"]) / df_data.loc[idx, "Close"]                    
            if idx + int(20) < len(df_data):
                df_data.loc[idx, '20 day change'] = (df_data.loc[idx + int(20), "Close"] - df_data.loc[idx, "Close"]) / df_data.loc[idx, "Close"]                    
        idx += 1

    return df_data
