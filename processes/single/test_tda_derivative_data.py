'''
Created on Aug 25, 2020

@author: Brian
'''
import os
import pandas as pd

#from tda_derivative_data import add_trending_data
#from tda_derivative_data import add_change_data
from tda_derivative_data import add_derived_data
from macd import macd
from on_balance_volume import on_balance_volume
from bollinger_bands import bollinger_bands
from accumulation_distribution import accumulation_distribution
from aroon_indicator import aroon_indicator
from average_directional_index import average_directional_index
from average_directional_index import eval_average_directional_index
from stochastic_oscillator import stochastic_oscillator
from relative_strength import relative_strength

from technical_analysis_utilities import initialize_eval_results

if __name__ == '__main__':
    df_data = pd.read_csv('d:\\brian\AI Projects\\tda\market_data\\c.csv')
    df_data = add_derived_data(df_data)
    '''
    df_data = macd(df_data[:], value_label="Close")
    df_data = on_balance_volume(df_data[:], value_label='Close', volume_lable='Volume')
    df_data = bollinger_bands(df_data[:], value_label="EMA20", ema_interval=20)
    df_data = accumulation_distribution(df_data)
    df_data = aroon_indicator(df_data)
    df_data = stochastic_oscillator(df_data)
    df_data = relative_strength(df_data, value_label="Close", relative_to='d:\\brian\AI Projects\\tda\market_data\\$spx.x.csv')
    '''
    df_data = average_directional_index(df_data)
    
    '''
    test evaluation functions
    '''
    eval_results = initialize_eval_results()
    eval_average_directional_index('c', df_data, eval_results)
    # output data for external viewing and validation
    df_data.to_csv('d:\\brian\Google Drive\\Documents\development\\tda_data.csv', index=False)
