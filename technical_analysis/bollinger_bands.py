'''
Created on Jan 31, 2018

@author: Brian

https://en.wikipedia.org/wiki/Bollinger_Bands
http://www.investopedia.com/terms/b/bollingerbands.asp

Bollinger Bands are a volatility indicator similar to the Keltner channel.

Bollinger Bands consist of:
an N-period moving average (MA)
an upper band at K times an N-period standard deviation above the moving average (MA + K*sd)
a lower band at K times an N-period standard deviation below the moving average (MA - K*sd)

Typical values for N and K are 20 and 2, respectively. The default choice for the average 
is a simple moving average, but other types of averages can be employed as needed. 
Exponential moving averages are a common second choice. Usually the same period is 
used for both the middle band and the calculation of standard deviation.

'''
import pandas as pd

from technical_analysis_utilities import add_results_index
from technical_analysis_utilities import find_sample_index

def eval_bollinger_bands(df_data, eval_results):
    r1_index = 'Bollinger Bands, Squeeze, 1 day'
    r5_index = 'Bollinger Bands, Squeeze, 5 day'
    r10_index = 'Bollinger Bands, Squeeze, 10 day'
    r20_index = 'Bollinger Bands, Squeeze, 20 day'
    trend_A_index = 'Bollinger Bands, Trend A, 20 day'
    trend_B_index = 'Bollinger Bands, Trend B, 20 day'
    if not r10_index in eval_results.index:
        eval_results = eval_results.append(add_results_index(eval_results, r1_index))
        eval_results = eval_results.append(add_results_index(eval_results, r5_index))
        eval_results = eval_results.append(add_results_index(eval_results, r10_index))
        eval_results = eval_results.append(add_results_index(eval_results, r20_index))
        eval_results = eval_results.append(add_results_index(eval_results, trend_A_index))
        eval_results = eval_results.append(add_results_index(eval_results, trend_B_index))

    '''
    TREND = 100
    NEAR = 0.2
    REVERSAL_INDICATOR = 0.25
    THRESHOLD = TREND * 0.75
    Bollinger Bands, Reversal down, 20 day            0.0     0.0      6.0      6.0      12.00      35.00      94.00      32.00      22.00      9.0      3.0     0.0      0.0
    Bollinger Bands, Reversal up, 20 day              0.0     0.0      3.0     11.0       2.00       1.00       1.00       1.00       0.00      0.0      0.0     0.0      0.0

    TREND = 50
    NEAR = 0.2
    REVERSAL_INDICATOR = 0.25
    THRESHOLD = TREND * 0.75
    Range Min                                     -1000.0    -1.0     -0.5     -0.2      -0.10      -0.05      -0.01       0.01       0.05      0.1      0.2     0.5      1.0
    Range Max                                        -1.0    -0.5     -0.2     -0.1      -0.05      -0.01       0.01       0.05       0.10      0.2      0.5     1.0   1000.0
    Bollinger Bands, Reversal down, 20 day            0.0     4.0     74.0    233.0     469.00     803.00     882.00    1312.00     652.00    337.0    147.0     4.0      0.0
    Bollinger Bands, Reversal up, 20 day              0.0     8.0     71.0    164.0     194.00     185.00     174.00     260.00     259.00    238.0    134.0    23.0      0.0
    '''
    TREND = 100
    NEAR = 0.2
    REVERSAL_INDICATOR = 0.25
    THRESHOLD = TREND * 0.75

    ds_volatility = df_data["BB_Upper"] - df_data["BB_Lower"]
    ds_lower = abs(df_data["Close"] - df_data["BB_Lower"])
    ds_upper = abs(df_data["BB_Upper"] - df_data["Close"])
    
    ds_near_upper = pd.Series(0 for i in range(0, df_data.shape[0]))
    ds_reversing = pd.Series(False for i in range(0, df_data.shape[0]))
    ds_near_lower = pd.Series(0 for i in range(0, df_data.shape[0]))
    
    idx = 1
    while idx < len(df_data):
        if df_data.at[idx, "Close"] > df_data.at[idx, "BB_Upper"] - (ds_volatility[idx] * NEAR):
            ds_near_upper[idx] = 1
        
        if df_data.at[idx, "Close"] < (df_data.at[idx, "SMA20"] + (ds_volatility[idx] * REVERSAL_INDICATOR)) and \
            df_data.at[idx, "Close"] > (df_data.at[idx, "SMA20"] - (ds_volatility[idx] * REVERSAL_INDICATOR)):
            ds_reversing.at[idx] = True
        
        if df_data.at[idx, "Close"] < df_data.at[idx, "BB_Lower"] + (ds_volatility[idx] * NEAR):
            ds_near_lower.at[idx] = 1
        
        if idx > TREND and ds_reversing.at[idx]:
            if ds_near_upper.loc[idx-TREND : idx].sum() > THRESHOLD:
                eval_results.at[trend_A_index, find_sample_index(eval_results, df_data.at[idx, '20 day change'])] += 1
            if ds_near_lower.loc[idx-TREND : idx].sum() > THRESHOLD:
                eval_results.at[trend_B_index, find_sample_index(eval_results, df_data.at[idx, '20 day change'])] += 1
        
        if not ds_volatility[idx] == 0:
            if idx >= TREND:
                if ds_volatility.iloc[idx] < ds_volatility.iloc[idx-TREND:idx].min():
                    eval_results.at[r1_index, find_sample_index(eval_results, df_data.at[idx, '1 day change'])] += 1
                    eval_results.at[r5_index, find_sample_index(eval_results, df_data.at[idx, '5 day change'])] += 1
                    eval_results.at[r10_index, find_sample_index(eval_results, df_data.at[idx, '10 day change'])] += 1
                    eval_results.at[r20_index, find_sample_index(eval_results, df_data.at[idx, '20 day change'])] += 1
        idx += 1
    return eval_results

def bollinger_bands(df_data=None, value_label=None, sma_interval=None):
    ds_std_dev = df_data[value_label]
    ds_value = df_data[value_label]
    ds_std_dev = ds_std_dev.rolling(window=sma_interval, min_periods=1).std()
    df_data.insert(0, 'BB_Upper', ds_value + (ds_std_dev * 2))
    df_data.insert(0, 'BB_Lower', ds_value - (ds_std_dev * 2))

    return df_data