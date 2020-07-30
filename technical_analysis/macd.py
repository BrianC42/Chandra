'''
Created on Jan 31, 2018

@author: Brian

What is the 'Moving Average Convergence Divergence - MACD'
http://www.investopedia.com/terms/m/macd.asp?ad=dirN&qo=investopediaSiteSearch&qsrc=0&o=40186

Moving average convergence divergence (MACD) is a trend-following momentum indicator that shows 
the relationship between two moving averages of prices. The MACD is calculated by subtracting 
the 26-day exponential moving average (EMA) from the 12-day EMA. A nine-day EMA of the MACD, 
called the "signal line", is then plotted on top of the MACD, functioning as a trigger for 
buy and sell signals.

BREAKING DOWN 'Moving Average Convergence Divergence - MACD'
There are three common methods used to interpret the MACD:

1. Crossovers - As shown in the chart above, when the MACD falls below the signal line, 
it is a bearish signal, which indicates that it may be time to sell. Conversely, when the 
MACD rises above the signal line, the indicator gives a bullish signal, which suggests 
that the price of the asset is likely to experience upward momentum. Many traders wait for a 
confirmed cross above the signal line before entering into a position to avoid getting getting 
"faked out" or entering into a position too early, as shown by the first arrow.

2. Divergence - When the security price diverges from the MACD. It signals the end of the current trend.

3. Dramatic rise - When the MACD rises dramatically - that is, the shorter moving average pulls 
away from the longer-term moving average - it is a signal that the security is overbought and 
will soon return to normal levels.

Traders also watch for a move above or below the zero line because this signals the position of the 
short-term average relative to the long-term average. 
When the MACD is above zero, the short-term average is above the long-term average, which signals upward momentum. 
The opposite is true when the MACD is below zero. As you can see from the chart above, 
the zero line often acts as an area of support and resistance for the indicator.

'''
import pandas as pd
from numpy import NaN
import sys

from tda_api_library import format_tda_datetime

from moving_average import exponential_moving_average
from technical_analysis_utilities import add_results_index
from technical_analysis_utilities import find_sample_index

def return_macd_flags():
    MACD_flags = dict(buy='1', sell='-1', neutral='0')
    return (MACD_flags)

def return_macd_ind():
    MACD_ind = dict(buy=1, sell=-1, neutral=0)
    return (MACD_ind)

def trade_on_crossovers(guidance, symbol, df_data):
    return

def trade_on_macd(guidance, symbol, df_data):
    day1 = df_data.shape[0]-3
    day2 = df_data.shape[0]-2
    day3 = df_data.shape[0]-1
    trigger_status = ""
    trade = False
    trigger_date = ""
    close = 0.0
    if day1 >= 0:
        trigger_date = format_tda_datetime(df_data.at[day3, 'DateTime'])
        '''
        if df_data.at[day3, 'MACD_Buy'] == True:
            trade = True
            trigger_status = "Potential recommendation in 2 trading days"
        if df_data.at[day2, 'MACD_Buy'] == True:
            if df_data.at[day3, 'MACD'] > df_data.at[day3, 'MACD_Signal']:
                trade = True
                trigger_status = "Potential recommendation on next trading day"
        '''
        if df_data.at[day1, 'MACD_Buy']:
            if df_data.at[day2, 'MACD'] > df_data.at[day2, 'MACD_Signal']:
                if df_data.at[day3, 'MACD'] > df_data.at[day3, 'MACD_Signal']:
                    trade = True
                    close = df_data.at[day3, 'Close']
                    trigger_status = "confirmed positive Cross"
                    guidance = guidance.append([[trade, symbol, 'MACD', trigger_date, trigger_status, close]])
    
        if df_data.at[day1, 'MACD_Sell']:
            if df_data.at[day2, 'MACD'] < df_data.at[day2, 'MACD_Signal']:
                if df_data.at[day3, 'MACD'] < df_data.at[day3, 'MACD_Signal']:
                    trade = True
                    close = df_data.at[day3, 'Close']
                    trigger_status = "confirmed negative Cross"
                    guidance = guidance.append([[trade, symbol, 'MACD', trigger_date, trigger_status, close]])
    
    return guidance

def eval_macd_positive_cross(df_data, eval_results):
    DRAMATIC_RISE = 1.05
    DRAMATIC_DECLINE = 0.95
    UPWARD_DIVERGENCE = 1.1
    DOWNWARD_DIVERGENCE = 0.9

    index_1 = 'MACD confirmed positive Cross - 10 day max'
    index_2 = 'MACD confirmed negative Cross - 10 day max'
    index_3 = 'MACD dramatic rise - 10 day max'
    index_4 = 'MACD dramatic decline - 10 day max'
    index_5 = 'MACD positive divergence - 10 day max'
    index_6 = 'MACD negative divergence - 10 day max'
    if not index_1 in eval_results.index:
        eval_results = eval_results.append(add_results_index(eval_results, index_1))
        eval_results = eval_results.append(add_results_index(eval_results, index_2))
        eval_results = eval_results.append(add_results_index(eval_results, index_3))
        eval_results = eval_results.append(add_results_index(eval_results, index_4))
        eval_results = eval_results.append(add_results_index(eval_results, index_5))
        eval_results = eval_results.append(add_results_index(eval_results, index_6))

    ndx = 10
    while ndx < df_data.shape[0] - 1:
        # confirmed crossovers
        if  df_data.at[ndx-2, 'MACD_Buy'] and \
            df_data.at[ndx-1, 'MACD'] > df_data.at[ndx-1, 'MACD_Signal'] and \
            df_data.at[ndx, 'MACD'] > df_data.at[ndx, 'MACD_Signal']:
            eval_results.at[index_1, find_sample_index(eval_results, df_data.at[ndx, '10 day max'])] += 1
        if  df_data.at[ndx-2, 'MACD_Sell'] and \
            df_data.at[ndx-1, 'MACD'] < df_data.at[ndx-1, 'MACD_Signal'] and \
            df_data.at[ndx, 'MACD'] < df_data.at[ndx, 'MACD_Signal']:
            eval_results.at[index_2, find_sample_index(eval_results, df_data.at[ndx, '10 day max'])] += 1

        # dramatic rise and decline
        if df_data.at[ndx, "EMA12"] > (df_data.at[ndx, "EMA26"] * DRAMATIC_RISE):
            eval_results.at[index_3, find_sample_index(eval_results, df_data.at[ndx, '10 day max'])] += 1
        if df_data.at[ndx, "EMA12"] < (df_data.at[ndx, "EMA26"] * DRAMATIC_DECLINE):
            eval_results.at[index_4, find_sample_index(eval_results, df_data.at[ndx, '10 day max'])] += 1
        
        # divergence
        if df_data.at[ndx, "Close"] > (df_data.at[ndx, "EMA12"] * UPWARD_DIVERGENCE):
            eval_results.at[index_5, find_sample_index(eval_results, df_data.at[ndx, '10 day max'])] += 1
        if df_data.at[ndx, "Close"] < (df_data.at[ndx, "EMA12"] * DOWNWARD_DIVERGENCE):
            eval_results.at[index_6, find_sample_index(eval_results, df_data.at[ndx, '10 day max'])] += 1
        
        ndx += 1
    return eval_results

def macd(df_src=None, value_label=None):
        
    sys.path.append("../Utilities/")

    #str_prediction = prediction_interval + ' day change'
    df_src.insert(loc=0, column='MACD_Buy', value=False)
    df_src.insert(loc=0, column='MACD_Sell', value=False)
    df_src.insert(loc=0, column='MACD_flag', value=NaN)
    df_src.insert(loc=0, column='MACD', value=NaN)
    df_src.insert(loc=0, column='MACD_Signal', value=NaN)

    MACD_flags = return_macd_flags()
    idx = 1 ### EMA calculation forces 1st entries to be equal, will always show as crossover on 2nd row
    while idx < len(df_src):
        df_src.at[idx,'MACD'] = df_src.at[idx, "EMA12"] - df_src.at[idx, "EMA26"]
        idx += 1
    df_src = exponential_moving_average(df_src[:], value_label='MACD', interval=9, EMA_data_label='MACD_Signal')  
    idx = 1 ### EMA calculation forces 1st entries to be equal, will always show as crossover on 2nd row
    while idx < len(df_src):
        df_src.at[idx, 'MACD_flag'] = MACD_flags.get('neutral')
        if df_src.at[idx, 'MACD'] < df_src.at[idx, 'MACD_Signal']:
            if df_src.at[idx-1, 'MACD'] >= df_src.at[idx-1, 'MACD_Signal']:
                ### Downward crossover
                df_src.at[idx, 'MACD_Sell'] = True
                df_src.at[idx, 'MACD_flag'] = MACD_flags.get('sell')
        if df_src.at[idx, 'MACD'] > df_src.at[idx, 'MACD_Signal']:
            if df_src.at[idx-1, 'MACD'] <= df_src.at[idx-1, 'MACD_Signal']:
                ### Upward crossover
                df_src.at[idx, 'MACD_Buy'] = True
                df_src.at[idx, 'MACD_flag'] = MACD_flags.get('buy')   
        idx += 1
        
    return df_src