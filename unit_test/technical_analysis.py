'''
Created on Jan 31, 2018

@author: Brian
'''

from macd import macd
from accumulation_distribution import accumulation_distribution
from aroon_indicator import aroon_indicator
from average_directional_index import average_directional_index
from on_balance_volume import on_balance_volume
from relative_strength import relative_strength
from stochastic_oscillator import stochastic_oscillator

def perform_technical_analysis (df_data=None, df_tickers=None):
    print ("\nBeginning technical analysis ...")
    idx = 0
    indices = df_tickers.index.get_values()
    #print ("indices:\n", indices)

    while idx < df_tickers.shape[0]:
        slice_start = indices[idx]
    
        if idx < df_tickers.shape[0] - 1:
            slice_end = indices[idx+1]
            #print ("ticker", tickers.ix[slice_start, 'ticker'], "Not last ticker. Slice:", slice_start, "to", slice_end)
        else:
            slice_end = len(df_data) - 1
            #print ("ticker", tickers.ix[slice_start, 'ticker'], "Last ticker. Slice", slice_start, "to", slice_end)
        
        macd(df_data[slice_start:slice_end], "adj_close", 12, 'EMA12', 26, 'EMA26', "date", 30)
        print ("MACD information calculated for", df_tickers.ix[slice_start, 'ticker'], idx+1, "of", df_tickers.shape[0])

        accumulation_distribution(df_data[slice_start:slice_end])
        
        aroon_indicator(df_data[slice_start:slice_end])
        
        average_directional_index(df_data[slice_start:slice_end])
        
        on_balance_volume(df_data[slice_start:slice_end])
        
        relative_strength(df_data[slice_start:slice_end])
        
        stochastic_oscillator(df_data[slice_start:slice_end])
        
        ''' temporary test '''
        #break
           
        idx += 1
        
    print ("Technical analysis complete")

    return
