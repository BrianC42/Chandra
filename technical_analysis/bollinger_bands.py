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

from technical_analysis_utilities import increment_sample_counts

OVERSOLD = 0.25
TREND = 100
NEAR = 0.25
REVERSAL_INDICATOR = 0.25
THRESHOLD = TREND * 0.75

def bb_oversold(guidance, symbol, df_data):
    trigger_status = ""
    trade = False
    trigger_date = ""
    close = 0.0
        
    # Oversold indicator
    idx = len(df_data) - 1
    if idx > 0:
        if df_data.loc[idx, "Close"] < df_data.loc[idx, "BB_Lower"] * (1 - OVERSOLD):
            trade = True
            close = df_data.at[idx, 'Close']
            trigger_status = "Oversold: 85% chance of 1 day price move >1%"
            trigger_date = format_tda_datetime(df_data.at[idx, 'DateTime'])

    if trade:
        guidance = guidance.append([[trade, symbol, 'BB', trigger_date, trigger_status, df_data.at[idx, "Close"]]])
    return guidance

def bb_trend(guidance, symbol, df_data):
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
        trigger_date = format_tda_datetime(df_data.at[idx, 'DateTime'])
            
        if ds_near_lower.loc[idx-TREND : idx].sum() > THRESHOLD:
            trade = True
            trigger_status = "lower band trend: probable decline in 20 days"
            guidance = guidance.append([[trade, symbol, 'BB', trigger_date, trigger_status, df_data.at[idx, "Close"]]])
            
        if ds_near_upper.loc[idx-TREND : idx].sum() > THRESHOLD:
            trade = True
            trigger_status = "Upper band trend: elevated chance of stable price"
            guidance = guidance.append([[trade, symbol, 'BB', trigger_date, trigger_status, df_data.at[idx, "Close"]]])

        if df_data.loc[idx, "Close"] < df_data.loc[idx, "BB_Lower"] * (1 - OVERSOLD):
            trade = True
            close = df_data.at[idx, 'Close']
            trigger_status = "Oversold: 85% chance of 1 day price move >1%"
            guidance = guidance.append([[trade, symbol, 'BB', trigger_date, trigger_status, df_data.at[idx, "Close"]]])

    return guidance

def trade_on_bb(guidance, symbol, df_data):
    '''
    guidance = bb_trend(guidance, symbol, df_data)
    guidance = bb_oversold(guidance, symbol, df_data)
    '''
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

    '''        
    while idx < len(df_data):
        if df_data.at[idx, "Close"] > df_data.at[idx, "BB_Upper"] - (ds_price_range[idx] * NEAR):
            ds_near_upper[idx] = 1
        
        if df_data.at[idx, "Close"] < (df_data.at[idx, "EMA20"] + (ds_price_range[idx] * REVERSAL_INDICATOR)) and \
            df_data.at[idx, "Close"] > (df_data.at[idx, "EMA20"] - (ds_price_range[idx] * REVERSAL_INDICATOR)):
            ds_reversing.at[idx] = True
        
        if df_data.at[idx, "Close"] < df_data.at[idx, "BB_Lower"] + (ds_price_range[idx] * NEAR):
            ds_near_lower.at[idx] = 1        
        idx += 1        
    '''
        
    idx = df_data.shape[0] - 1
    if idx > TREND and ds_reversing.at[idx]:
        trigger_date = format_tda_datetime(df_data.at[idx, 'DateTime'])
        
        '''
        if ds_near_lower.loc[idx-TREND : idx].sum() > THRESHOLD:
            trade = True
            trigger_status = "lower trend"
            guidance = guidance.append([[trade, symbol, 'BB', trigger_date, trigger_status, df_data.at[idx, "Close"]]])
        '''
        
        if ds_near_upper.loc[idx-TREND : idx].sum() > THRESHOLD:
            trade = True
            trigger_status = "Upper trend"
            guidance = guidance.append([[trade, symbol, 'BB', trigger_date, trigger_status, df_data.at[idx, "Close"]]])

        if df_data.loc[idx, "Close"] < df_data.loc[idx, "BB_Lower"] * (ds_price_range[idx] * OVERSOLD):
            trade = True
            trigger_status = "Oversold"
            guidance = guidance.append([[trade, symbol, 'BB', trigger_date, trigger_status, df_data.at[idx, "Close"]]])
            
    return guidance

def eval_bollinger_bands(symbol, df_data, eval_results):
    rsqeeze_index = ['BB', 'Squeeze']
    oversold_index = ['BB', 'Oversold']
    trend_Upper_index = ['BB', 'Upper Trend']
    trend_Lower_index = ['BB', 'Lower Trend']
    oversold_squeeze_index = ['BB', 'Oversold Squeeze']

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
                eval_results = increment_sample_counts(symbol, eval_results, trend_Upper_index, df_data.iloc[idx, :]) 
            # trade signal B
            if ds_near_lower.loc[idx-TREND : idx].sum() > THRESHOLD:
                eval_results = increment_sample_counts(symbol, eval_results, trend_Lower_index, df_data.iloc[idx, :]) 
        
        # Oversold indicator
        if df_data.at[idx, "Close"] < df_data.at[idx, "BB_Lower"] - (ds_price_range[idx] * OVERSOLD):
            eval_results = increment_sample_counts(symbol, eval_results, oversold_index, df_data.iloc[idx, :]) 
        
        if not ds_price_range[idx] == 0:
            if idx >= TREND:
                # Trade signal C - minimal volatility 'squeeze'
                if ds_price_range.iloc[idx] < ds_price_range.iloc[idx-TREND:idx].min():
                    eval_results = increment_sample_counts(symbol, eval_results, rsqeeze_index, df_data.iloc[idx, :]) 
                    # Oversold squeeze indicator
                    if df_data.at[idx, "Close"] < df_data.at[idx, "BB_Lower"] - (ds_price_range[idx] * OVERSOLD):
                        eval_results = increment_sample_counts(symbol, eval_results, oversold_squeeze_index, df_data.iloc[idx, :]) 
                    
        idx += 1
    return eval_results

def bollinger_bands(df_data=None, value_label=None, ema_interval=None):
    ds_std_dev = df_data[value_label]
    ds_value = df_data[value_label]
    ds_std_dev = ds_std_dev.rolling(window=ema_interval, min_periods=1).std()
    df_data.insert(0, 'BB_Upper', ds_value + (ds_std_dev * 2))
    df_data.insert(0, 'BB_Lower', ds_value - (ds_std_dev * 2))

    return df_data