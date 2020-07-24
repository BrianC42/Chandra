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

from tda_api_library import format_tda_datetime

from technical_analysis_utilities import add_results_index
from technical_analysis_utilities import find_sample_index

def bb_oversold(guidance, symbol, df_data):
    OVERSOLD = 0.1

    trigger_status = ""
    trade = False
    trigger_date = ""
    close = 0.0
        
    # Oversold indicator
    idx = len(df_data) - 1
    if idx >= 0:
        if df_data.loc[idx, "Close"] < df_data.loc[idx, "BB_Lower"] * (1 - OVERSOLD):
            trade = True
            trigger_status = "85% chance of 1 day price move >1%"
            trigger_date = format_tda_datetime(df_data.at[df_data.shape[0], 'DateTime'])

    if trade:
        guidance = guidance.append([[trade, symbol, 'BB Oversold', trigger_date, trigger_status, close]])
    return guidance

def bb_signal_b(guidance, symbol, df_data):
    TREND = 100
    NEAR = 0.2
    REVERSAL_INDICATOR = 0.25
    THRESHOLD = TREND * 0.75

    trigger_status = ""
    trade = False
    trigger_date = ""
    
    ds_price_range = df_data["BB_Upper"] - df_data["BB_Lower"]
    ds_reversing = pd.Series(False for i in range(0, df_data.shape[0]))
    ds_near_upper = pd.Series(0 for i in range(0, df_data.shape[0]))
    ds_reversing = pd.Series(False for i in range(0, df_data.shape[0]))
    ds_near_lower = pd.Series(0 for i in range(0, df_data.shape[0]))

    if len(df_data) < TREND:
        idx = 0
    else:
        idx = len(df_data) - TREND
        
    while idx < len(df_data):
        if df_data.at[idx, "Close"] > df_data.at[idx, "BB_Upper"] - (ds_price_range[idx] * NEAR):
            ds_near_upper[idx] = 1
        
        if df_data.at[idx, "Close"] < (df_data.at[idx, "EMA20"] + (ds_price_range[idx] * REVERSAL_INDICATOR)) and \
            df_data.at[idx, "Close"] > (df_data.at[idx, "EMA20"] - (ds_price_range[idx] * REVERSAL_INDICATOR)):
            ds_reversing.at[idx] = True
        
        if df_data.at[idx, "Close"] < df_data.at[idx, "BB_Lower"] + (ds_price_range[idx] * NEAR):
            ds_near_lower.at[idx] = 1        
        idx += 1        
    
    idx = df_data.shape[0] - 1
    if idx > TREND and ds_reversing.at[idx]:
        if ds_near_lower.loc[idx-TREND : idx].sum() > THRESHOLD:
            trade = True
            trigger_status = "90% probable decline in 20 days"
            trigger_date = format_tda_datetime(df_data.at[df_data.shape[0], 'DateTime'])

    if trade:
        guidance = guidance.append([[trade, symbol, 'BB condition 1', trigger_date, trigger_status, df_data.at[idx, "Close"]]])
    return guidance

def trade_on_bb(guidance, symbol, df_data):
    guidance = bb_signal_b(guidance, symbol, df_data)
    guidance = bb_oversold(guidance, symbol, df_data)
    return guidance

def eval_bollinger_bands(df_data, eval_results):
    r1_index = 'Bollinger Bands, Squeeze, 1 day'
    r5_index = 'Bollinger Bands, Squeeze, 5 day'
    r10_index = 'Bollinger Bands, Squeeze, 10 day'
    r20_index = 'Bollinger Bands, Squeeze, 20 day'
    oversold1_index = "Bollinger Bands, Oversold, 1 day"
    oversold5_index = "Bollinger Bands, Oversold, 5 day"
    oversold10_index = "Bollinger Bands, Oversold, 10 day"
    trend_A_index = 'Bollinger Bands, Trend A, 20 day'
    trend_B_index = 'Bollinger Bands, Trend B, 20 day'
    oversold_squeeze_index = "Bollinger Bands, Oversold Squeeze, 20 day"
    if not r10_index in eval_results.index:
        eval_results = eval_results.append(add_results_index(eval_results, r1_index))
        eval_results = eval_results.append(add_results_index(eval_results, r5_index))
        eval_results = eval_results.append(add_results_index(eval_results, r10_index))
        eval_results = eval_results.append(add_results_index(eval_results, r20_index))
        eval_results = eval_results.append(add_results_index(eval_results, oversold1_index))
        eval_results = eval_results.append(add_results_index(eval_results, oversold5_index))
        eval_results = eval_results.append(add_results_index(eval_results, oversold10_index))
        eval_results = eval_results.append(add_results_index(eval_results, trend_A_index))
        eval_results = eval_results.append(add_results_index(eval_results, trend_B_index))
        eval_results = eval_results.append(add_results_index(eval_results, oversold_squeeze_index))

    TREND = 100
    NEAR = 0.2
    REVERSAL_INDICATOR = 0.25
    THRESHOLD = TREND * 0.75
    OVERSOLD = 0.1
    
    ds_price_range = df_data["BB_Upper"] - df_data["BB_Lower"]    
    ds_near_upper = pd.Series(0 for i in range(0, df_data.shape[0]))
    ds_reversing = pd.Series(False for i in range(0, df_data.shape[0]))
    ds_near_lower = pd.Series(0 for i in range(0, df_data.shape[0]))
    
    idx = 1
    while idx < len(df_data):
        if df_data.at[idx, "Close"] > df_data.at[idx, "BB_Upper"] - (ds_price_range[idx] * NEAR):
            ds_near_upper[idx] = 1
        
        if df_data.at[idx, "Close"] < (df_data.at[idx, "EMA20"] + (ds_price_range[idx] * REVERSAL_INDICATOR)) and \
            df_data.at[idx, "Close"] > (df_data.at[idx, "EMA20"] - (ds_price_range[idx] * REVERSAL_INDICATOR)):
            ds_reversing.at[idx] = True
        
        if df_data.at[idx, "Close"] < df_data.at[idx, "BB_Lower"] + (ds_price_range[idx] * NEAR):
            ds_near_lower.at[idx] = 1
        
        if idx > TREND and ds_reversing.at[idx]:
            # trade signal A
            if ds_near_upper.loc[idx-TREND : idx].sum() > THRESHOLD:
                eval_results.at[trend_A_index, find_sample_index(eval_results, df_data.at[idx, '20 day change'])] += 1
            # trade signal B
            if ds_near_lower.loc[idx-TREND : idx].sum() > THRESHOLD:
                eval_results.at[trend_B_index, find_sample_index(eval_results, df_data.at[idx, '20 day change'])] += 1
        
        # Oversold indicator
        if df_data.at[idx, "Close"] < df_data.at[idx, "BB_Lower"] * (1 - OVERSOLD):
            eval_results.at[oversold1_index, find_sample_index(eval_results, df_data.at[idx, '1 day change'])] += 1
            eval_results.at[oversold5_index, find_sample_index(eval_results, df_data.at[idx, '5 day change'])] += 1
            eval_results.at[oversold10_index, find_sample_index(eval_results, df_data.at[idx, '10 day change'])] += 1
        
        if not ds_price_range[idx] == 0:
            if idx >= TREND:
                # Trade signal C - minimal volatility 'squeeze'
                if ds_price_range.iloc[idx] < ds_price_range.iloc[idx-TREND:idx].min():
                    eval_results.at[r1_index, find_sample_index(eval_results, df_data.at[idx, '1 day change'])] += 1
                    eval_results.at[r5_index, find_sample_index(eval_results, df_data.at[idx, '5 day change'])] += 1
                    eval_results.at[r10_index, find_sample_index(eval_results, df_data.at[idx, '10 day change'])] += 1
                    eval_results.at[r20_index, find_sample_index(eval_results, df_data.at[idx, '20 day change'])] += 1
                    # Oversold squeeze indicator
                    if df_data.at[idx, "Close"] < df_data.at[idx, "BB_Lower"] * (1 - OVERSOLD):
                        eval_results.at[oversold_squeeze_index, find_sample_index(eval_results, df_data.at[idx, '20 day change'])] += 1                    
                    
        idx += 1
    return eval_results

def bollinger_bands(df_data=None, value_label=None, ema_interval=None):
    ds_std_dev = df_data[value_label]
    ds_value = df_data[value_label]
    ds_std_dev = ds_std_dev.rolling(window=ema_interval, min_periods=1).std()
    df_data.insert(0, 'BB_Upper', ds_value + (ds_std_dev * 2))
    df_data.insert(0, 'BB_Lower', ds_value - (ds_std_dev * 2))

    return df_data