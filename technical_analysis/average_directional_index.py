'''
Created on Jan 31, 2018

@author: Brian

http://www.investopedia.com/terms/a/adx.asp?ad=dirN&qo=investopediaSiteSearch&qsrc=0&o=40186

What is the 'Average Directional Index - ADX'
The average directional index (ADX) is an indicator used in technical analysis as an objective 
value for the strength of a trend. ADX is non-directional, so it quantifies a trend's 
strength regardless of whether it is up or down. It is usually plotted in a chart window 
along with two lines known as the DMI (Directional Movement Indicators).

BREAKING DOWN 'Average Directional Index - ADX'
Analysis of ADX is a method of evaluating trends and can help traders choose the strongest trends.
There are two forms of stock analysis: fundamental and technical. Fundamental analysis 
selects stocks based on business performance, whereas technical analysis selects stocks 
based on price movement. When technicians look for patterns over time to confirm a continuation 
or trend reversal, they often use the average directional index as an indicator. Developed by 
Welles Wilder for use with commodities and daily prices, the indicator is also used for 
stock selection. Wilder used the plus directional movement (+DM) and the minus direction 
movement (-DM) to determine the ADX.

The ADX
Traders believe that the trend is your friend. As a result, there are numerous trading 
indicators that are meant to confirm a trend. Once the trend is identified, however, the 
challenge is determining the best time to enter and exit a trade. The ADX is used to measure 
the strength or weakness of a trend and is therefore used alongside the +DM and -DM to 
determine the best course of action. Put together, Wilder created a framework for trading 
out of the ADX, DI+ and DI- trio of lines.

ADX Strategies
Traders start by using the ADX to determine if there is a trend. A strong trend is occurring 
when the ADX is over 25; likewise, there is no trend when the ADX falls below 20. When 
the +DI line is greater than the -DI line, the bulls have the directional edge. However, 
when the -DI line is above the +DI line, the bears have the directional edge. As with all 
technical trends, traders use several indicators to confirm a movement. One option is to sell 
when -DI is up and the major trend is down. Another option is to buy when +DI is higher 
than -DI, but only when the larger trend is also moving up. In other words, it is possible 
to use the ADX as a way to time an entry on a market that is already confirmed to be trading 
in a particular direction.

'''
import pandas as pd
from technical_analysis_utilities import increment_sample_counts
from moving_average import exponential_moving_average

ADI_PERIODS = 14
ADI_TREND_MIN = 25
ADI_TREND_CONFIRMNATION = 5

def eval_average_directional_index(symbol, df_data, eval_results):
    ADI_UI = ['ADI', 'Up Trend']
    ADI_DI = ['ADI', 'Down Trend']
    ADI_TS = ['ADI', 'Trend Strengthening']
    ADI_TW = ['ADI', 'Trend Weakening']
    ADI_TR = ['ADI', 'Trend reversal']
    ADI_TRU = ['ADI', 'Trend reversal Up']
    ADI_TRD = ['ADI', 'Trend reversalDown']

    ndx = 1
    while ndx < df_data.shape[0]:
        if df_data.at[ndx, "ADX"] > ADI_TREND_MIN:
            if df_data.at[ndx-1, "ADX"] <= ADI_TREND_MIN:
                eval_results = increment_sample_counts(symbol, eval_results, ADI_TR, df_data.iloc[ndx, :])
                if df_data.at[ndx, 'ADI +DI'] > df_data.at[ndx, 'ADI -DI']:
                    eval_results = increment_sample_counts(symbol, eval_results, ADI_TRU, df_data.iloc[ndx, :]) 
                else:
                    eval_results = increment_sample_counts(symbol, eval_results, ADI_TRD, df_data.iloc[ndx, :])
            if df_data.at[ndx, 'ADI +DI'] > df_data.at[ndx, 'ADI -DI']:
                eval_results = increment_sample_counts(symbol, eval_results, ADI_UI, df_data.iloc[ndx, :]) 
            else:
                eval_results = increment_sample_counts(symbol, eval_results, ADI_DI, df_data.iloc[ndx, :])
            if ndx > ADI_TREND_CONFIRMNATION:
                tndx = 0
                ts = True
                tw = True
                while tndx < ADI_TREND_CONFIRMNATION:
                    if df_data.at[ndx-(tndx+1), "ADX"] > df_data.at[ndx-tndx, "ADX"]:
                        ts = False
                    if df_data.at[ndx-(tndx+1), "ADX"] < df_data.at[ndx-tndx, "ADX"]:
                        tw = False
                    tndx += 1
                if ts:
                    eval_results = increment_sample_counts(symbol, eval_results, ADI_TS, df_data.iloc[ndx, :])                
                if tw:
                    eval_results = increment_sample_counts(symbol, eval_results, ADI_TW, df_data.iloc[ndx, :])
        ndx += 1

    return eval_results

def average_directional_index(df_data=None):
    df_data.insert(loc=0, column='ADX', value=0.0)
    df_data.insert(loc=0, column='DX', value=0.0)
    df_data.insert(loc=0, column='ADI +DI', value=0.0)
    df_data.insert(loc=0, column='ADI -DI', value=0.0)
    df_data.insert(loc=0, column='ADI Smooth TR', value=0.0)
    df_data.insert(loc=0, column='ADI TR', value=0.0)
    df_data.insert(loc=0, column='ADI Smooth +DM', value=0.0)
    df_data.insert(loc=0, column='ADI +DM', value=0.0)
    df_data.insert(loc=0, column='ADI Smooth -DM', value=0.0)
    df_data.insert(loc=0, column='ADI -DM', value=0.0)
    
    #return df_data

    ndx = 1
    while ndx < df_data.shape[0]:
        if df_data.at[ndx, 'High'] > df_data.at[ndx-1, 'High']:
            df_data.at[ndx, 'ADI +DM'] = df_data.at[ndx, 'High'] - df_data.at[ndx-1, 'High']
            
        if df_data.at[ndx-1, 'Low'] > df_data.at[ndx, 'Low']:
            df_data.at[ndx, 'ADI -DM'] = df_data.at[ndx- 1, 'Low'] - df_data.at[ndx, 'Low']            
        if df_data.at[ndx, 'ADI +DM'] < df_data.at[ndx, 'ADI -DM']:
            df_data.at[ndx, 'ADI +DM'] = 0.0
        else:
            df_data.at[ndx, 'ADI -DM'] = 0.0
            
        df_data.at[ndx, 'ADI TR'] = max(df_data.at[ndx, 'High'] - df_data.at[ndx, 'Low'], \
                                        abs(df_data.at[ndx, 'High'] - df_data.at[ndx-1, 'Close']), \
                                        abs(df_data.at[ndx, 'Low'] - df_data.at[ndx-1, 'Close']))
        ndx += 1

    df_data = exponential_moving_average(df_data[:], value_label="ADI -DM", interval=ADI_PERIODS, EMA_data_label='ADI Smooth -DM')
    df_data = exponential_moving_average(df_data[:], value_label="ADI +DM", interval=ADI_PERIODS, EMA_data_label='ADI Smooth +DM')
    df_data = exponential_moving_average(df_data[:], value_label="ADI TR", interval=ADI_PERIODS, EMA_data_label='ADI Smooth TR')
    ndx = ADI_PERIODS
    while ndx < df_data.shape[0]:
        if df_data.at[ndx, 'ADI Smooth TR'] == 0.0:
            df_data.at[ndx, 'ADI +DI'] = 0.0
            df_data.at[ndx, 'ADI -DI'] = 0.0
            df_data.at[ndx, 'DX'] = 0.0
        else:
            df_data.at[ndx, 'ADI +DI'] = (df_data.at[ndx, 'ADI Smooth +DM'] * 100) / df_data.at[ndx, 'ADI Smooth TR']
            df_data.at[ndx, 'ADI -DI'] = (df_data.at[ndx, 'ADI Smooth -DM'] * 100) / df_data.at[ndx, 'ADI Smooth TR']
            df_data.at[ndx, 'DX'] = abs((df_data.at[ndx, 'ADI +DI'] - df_data.at[ndx, 'ADI -DI']) / \
                                        (df_data.at[ndx, 'ADI +DI'] + df_data.at[ndx, 'ADI -DI']))
        ndx += 1
    df_data = exponential_moving_average(df_data[:], value_label="DX", interval=ADI_PERIODS, EMA_data_label='ADX')
    
    ndx = 1
    while ndx < df_data.shape[0]:
        df_data.at[ndx, 'ADX'] = df_data.at[ndx, 'ADX'] * 100
        ndx += 1
        
    return df_data
