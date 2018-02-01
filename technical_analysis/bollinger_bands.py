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

from moving_average import simple_moving_average
from numpy import NaN

def add_bb_fields(df_data, sma_label):
    '''
    Only one additional field is added to the digest
        Bollinger band fields: 
            Simple Moving Average
            BB_Upper
            BB_Lower
    '''
    df_data.insert(loc=0, column=sma_label, value=NaN)
    df_data.insert(loc=0, column='BB_Upper', value=NaN)
    df_data.insert(loc=0, column='BB_Lower', value=NaN)

    return (df_data)

def bollinger_bands(df_data=None, \
                    value_label=None, \
                    sma_interval=None, \
                    sma_label=None):
    
    #data_points = len(df_data)
    #print ("bollinger_bands:", data_points, "data points", value_label, sma_interval, sma_label)
    
    df_data = add_bb_fields(df_data, sma_label)
    df_data = simple_moving_average(df_data[:], value_label, sma_interval, sma_label)
    
    ds_std_dev = df_data[value_label]
    ds_value = df_data[value_label]
    #print (ds_std_dev.head(sma_interval), ds_std_dev.tail(sma_interval))
    ds_std_dev = ds_std_dev.rolling(window=sma_interval, min_periods=1).std()
    #print (ds_std_dev.head(sma_interval), ds_std_dev.tail(sma_interval))
    df_data['BB_Upper'] = ds_value + (ds_std_dev * 2) 
    df_data['BB_Lower'] = ds_value - (ds_std_dev * 2)

    '''
    print ("\nbollinger_bands df_data head\n",df_data[:].head(sma_interval), \
           "\n\nand tail ...\n", df_data[:].tail(sma_interval))
    '''
    
    return df_data