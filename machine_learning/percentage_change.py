'''
Created on May 16, 2019

@author: Brian

Functions required to create, train, test and use a regression model deigned to predict the percentage change
in the forecast feature
 
'''
import logging

def bsh_data_check () :
    '''
    '''
    return

def balance_pct_change (np_data, np_forecast) :
    '''
    balance the training y-axis values to improve training
    '''
    return np_data, np_forecast

def build_bsh_classification_model () :
    '''
    '''
    return

def build_regression_model () :
    '''
    '''
    return

def calculate_sample_pct_change() :
    '''
    '''
    return

def calculate_single_actual_pct_change (current_price, future_price) :
    '''
    Calculate forecast y-axis characteristic value
    '''
    pct_change = future_price / current_price

    return pct_change

def calculate_single_pct_change(current_price, future_price):
    '''
    Calculate the percentage change between the current_price and future_price
    '''
    bsh_change = future_price / current_price

    return bsh_change

def pct_change_multiple():
    return