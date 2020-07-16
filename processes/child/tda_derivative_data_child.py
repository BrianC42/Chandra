'''
Created on Jul 15, 2020

@author: Brian
'''
import os
import pandas as pd

from tda_derivative_data import add_trending_data
from tda_derivative_data import add_change_data
from macd import macd
from on_balance_volume import on_balance_volume
from bollinger_bands import bollinger_bands
from accumulation_distribution import accumulation_distribution
from aroon_indicator import aroon_indicator
from average_directional_index import average_directional_index
from stochastic_oscillator import stochastic_oscillator
from relative_strength import relative_strength

def tda_derivative_data_child(mpipe_feed):
    #print ("Worker starting")
    try:
        while True:
            '''================ Work on data as long as the manager has more
            ==============================================================='''
            str_filenames = mpipe_feed.recv()
            
            '''================ Perform technical analysis ==================='''
            data_in = str_filenames[0]
            data_out = str_filenames[1]
            #print("Worker process reading: %s and creating: %s" % (data_in, data_out))
            if os.path.isfile(data_in):
                #print("File: %s" % data_in)
                df_data = pd.read_csv(data_in)
                #print("EOD data for %s\n%s" % (filename, df_data))
                df_data = add_trending_data(df_data)
                df_data = add_change_data(df_data)
                df_data = macd(df_data[:], value_label="Close")
                df_data = on_balance_volume(df_data[:], value_label='Close', volume_lable='Volume')
                df_data = bollinger_bands(df_data[:], value_label="Close", sma_interval=20)
                df_data = accumulation_distribution(df_data)
                df_data = aroon_indicator(df_data)
                df_data = average_directional_index(df_data)
                df_data = stochastic_oscillator(df_data)
                df_data = relative_strength(df_data, value_label="Close", relative_to='d:\\brian\AI Projects\\tda\market_data\\$spx.x.csv')
                df_data.to_csv(data_out, index=False)

            '''================ Return processed data ====================='''
            mpipe_feed.send(1)
            
    except EOFError:
        #print ("The Master has no more work for me ...")
        pass
        
    finally:
        #print ("Cleaning up after myself")
        pass

    #print ("Worker completing")

    return