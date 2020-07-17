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
from technical_analysis_utilities import add_results_index
from technical_analysis_utilities import find_sample_index

def eval_bollinger_bands(df_data, eval_results):
    result_index = 'Bollinger Bands, Close<SMA20, 10 day'
    if not result_index in eval_results.index:
        BB_results = add_results_index(eval_results, result_index)
        eval_results = eval_results.append(BB_results)
        
    rows = df_data.iterrows()
    for nrow in rows:
        if nrow[1]['Close'] < nrow[1]['SMA20']:
            cat_str = find_sample_index(eval_results, nrow[1]['10 day change'])
            eval_results.at[result_index, cat_str] += 1
    return eval_results

def bollinger_bands(df_data=None, value_label=None, sma_interval=None):
    
    ds_std_dev = df_data[value_label]
    ds_value = df_data[value_label]
    ds_std_dev = ds_std_dev.rolling(window=sma_interval, min_periods=1).std()
    df_data.insert(0, 'BB_Upper', ds_value + (ds_std_dev * 2))
    df_data.insert(0, 'BB_Lower', ds_value - (ds_std_dev * 2))

    return df_data