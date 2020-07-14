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

def bollinger_bands(df_data=None, value_label=None, sma_interval=None):
    
    ds_std_dev = df_data[value_label]
    ds_value = df_data[value_label]
    ds_std_dev = ds_std_dev.rolling(window=sma_interval, min_periods=1).std()
    df_data.insert(0, 'BB_Upper', ds_value + (ds_std_dev * 2))
    df_data.insert(0, 'BB_Lower', ds_value - (ds_std_dev * 2))

    return df_data