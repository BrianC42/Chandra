'''
Created on May 4, 2019

@author: Brian

Code specific to building, training, evaluating and using a model capable of returning an action flag
    Buy (1): data is indicating an increase in price >2% in the coming 30 days
    Hold (0) data is indicating the price will remain within 2% of the current price for the coming 30 days
    Sell (-1): data is indicating an decrease in price >2% in the coming 30 days
'''
import numpy as np

def calculate_single_bsh_flag(current_price, future_price):
    
    bsh_change = future_price / current_price
    if bsh_change >= 1.2 :
        # 3% increase
        bsh_flag = 1
    elif bsh_change <= 0.8 :
        # 3% decline
        bsh_flag = -1
    else :
        # change between -3% and +3%
        bsh_flag = 0

    return bsh_flag

def calculate_sample_bsh_flag(sample_single_flags):
    
    bsh_flag = 0
    
    bsh_flag_max = np.amax(sample_single_flags)
    bsh_flag_min = np.amin(sample_single_flags)
    
    if (bsh_flag_max == 1) :
        bsh_flag = 1
    elif (bsh_flag_min == -1) :
        bsh_flag = -1
    else :
        bsh_flag = 0
    
    return bsh_flag