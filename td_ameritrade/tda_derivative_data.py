'''
Created on Jul 13, 2020

@author: Brian
'''
from datetime import date
from numpy import NaN
import datetime
from moving_average import simple_moving_average
from moving_average import exponential_moving_average
from tda_api_library import format_tda_datetime

def add_derived_data(df_data):
    df_data.insert(loc=0, column='1 day change', value=NaN)
    df_data.insert(loc=0, column='5 day change', value=NaN)
    df_data.insert(loc=0, column='10 day change', value=NaN)
    df_data.insert(loc=0, column='10 day max', value=NaN)
    df_data.insert(loc=0, column='10 day min', value=NaN)
    df_data.insert(loc=0, column='20 day change', value=NaN)
    df_data.insert(loc=0, column='20 day max', value=NaN)
    df_data.insert(loc=0, column='20 day min', value=NaN)
    df_data.insert(loc=0, column='40 day change', value=NaN)
    df_data.insert(loc=0, column='40 day max', value=NaN)
    df_data.insert(loc=0, column='40 day min', value=NaN)
    df_data.insert(loc=0, column='date', value="")
    #df_data.insert(loc=0, column='month', value="")
    df_data.insert(loc=0, column='day', value="")
    #df_data.insert(loc=0, column='weekday', value="")
    df_data.insert(loc=0, column='10day5pct', value=False)
    df_data.insert(loc=0, column='10day10pct', value=False)
    df_data.insert(loc=0, column='10day25pct', value=False)
    df_data.insert(loc=0, column='10day50pct', value=False)
    df_data.insert(loc=0, column='10day100pct', value=False)
    
    df_data.insert(loc=0, column='EMA12', value=NaN)
    df_data.insert(loc=0, column='EMA20', value=NaN)
    df_data.insert(loc=0, column='EMA26', value=NaN)
    df_data.insert(loc=0, column='SMA20', value=NaN)

    idx = 0
    while idx < len(df_data):
        df_data.at[idx, 'date'] = format_tda_datetime( df_data.at[idx, 'DateTime'] )
        idx += 1
    #print(df_data)
    df_data = df_data.drop_duplicates(subset=['date'], keep='last', inplace=False)
    #print(df_data)
    #df_data = df_data.reset_index()
    df_data = df_data.set_index(i for i in range(0, df_data.shape[0]))
    #print(df_data)
    df_data = exponential_moving_average(df_data[:], value_label="Close", interval=12, EMA_data_label='EMA12')
    df_data = exponential_moving_average(df_data[:], value_label="Close", interval=20, EMA_data_label='EMA20')
    df_data = exponential_moving_average(df_data[:], value_label="Close", interval=26, EMA_data_label='EMA26')
    df_data = simple_moving_average(df_data[:], value_label="Close", avg_interval=20, SMA_data_label='SMA20')
    
    idx = 0
    while idx < len(df_data):
        '''
        df_data.at[idx, 'month'] = date.month(df_data.at[idx, 'date'])
        df_data.at[idx, 'weekday'] = date.weekday(df_data.at[idx, 'date'])
        '''
        df_data.at[idx, "day"] = date.fromtimestamp(df_data.at[idx, "DateTime"]/1000).timetuple().tm_yday
        closing_price = df_data.at[idx, "Close"]
        try:
            if not closing_price == 0:
                if idx < len(df_data) - 1:
                    df_data.loc[idx, '1 day change'] = (df_data.loc[idx + int(1), "Close"] - closing_price) / closing_price                    
                if idx < len(df_data) - 5:
                    df_data.loc[idx, '5 day change'] = (df_data.loc[idx + int(5), "Close"] - closing_price) / closing_price
                if idx < len(df_data) - 10:
                    df_data.loc[idx, '10 day change'] = (df_data.loc[idx + int(10), "Close"] - closing_price) / closing_price
                    df_data.loc[idx, '10 day max'] = df_data.iloc[idx:idx+10].get('High').max()
                    df_data.loc[idx, '10 day min'] = df_data.iloc[idx:idx+10].get('Low').min()                
                    if df_data.loc[idx, '10 day max'] > df_data.loc[idx, 'Close'] * 1.1:
                        df_data.loc[idx, '10day10pct'] = True
                        if df_data.loc[idx, '10 day max'] > df_data.loc[idx, 'Close'] * 1.25:
                            df_data.loc[idx, '10day25pct'] = True
                            if df_data.loc[idx, '10 day max'] > df_data.loc[idx, 'Close'] * 1.50:
                                df_data.loc[idx, '10day50pct'] = True
                                if df_data.loc[idx, '10 day max'] > df_data.loc[idx, 'Close'] * 2:
                                    df_data.loc[idx, '10day100pct'] = True
                if idx < len(df_data) - 14:
                    df_data.loc[idx, '14 day max'] = df_data.iloc[idx:idx+14].get('High').max()
                    df_data.loc[idx, '14 day min'] = df_data.iloc[idx:idx+14].get('Low').min()
                if idx < len(df_data) - 20:
                    df_data.loc[idx, '20 day change'] = (df_data.loc[idx + 20, "Close"] - closing_price) / closing_price
                    df_data.loc[idx, '20 day max'] = df_data.iloc[idx:idx + 20].get('High').max()
                    df_data.loc[idx, '20 day min'] = df_data.iloc[idx:idx + 20].get('Low').min()
                if idx < len(df_data) - 40:
                    df_data.loc[idx, '40 day change'] = (df_data.loc[idx + 40, "Close"] - closing_price) / closing_price
                    df_data.loc[idx, '40 day max'] = df_data.iloc[idx:idx + 40].get('High').max()
                    df_data.loc[idx, '40 day min'] = df_data.iloc[idx:idx + 40].get('Low').min()
            pass
        except:
            print("error")
        
        idx += 1

    return df_data
