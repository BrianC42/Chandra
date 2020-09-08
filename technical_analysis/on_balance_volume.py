'''
Created on Jan 31, 2018

@author: Brian

What is On-Balance Volume (OBV)
On-balance volume (OBV) is a momentum indicator that uses volume flow to predict 
changes in stock price. Joseph Granville developed the OBV metric in the 1960s. 
He believed that, when volume increases sharply without a significant change 
in the stock's price, the price will eventually jump upward, and vice versa.

BREAKING DOWN On-Balance Volume (OBV)
The theory behind OBV is based on the distinction between smart money - namely,
institutional investors - and less sophisticated retail investors. As mutual
funds and pension funds begin to buy into an issue that retail investors are selling,
volume may increase even as the price remains relatively level. Eventually, volume
drives the price upward. At that point, larger investors begin to sell, and smaller
investors begin buying.

The OBV is a running total of volume (positive and negative). 
There are three rules implemented when calculating the OBV. They are:

1. If today's closing price is higher than yesterday's closing price, 
then: Current OBV = Previous OBV + today's volume

2. If today's closing price is lower than yesterday's closing price, 
then: Current OBV = Previous OBV - today's volume

3. If today's closing price equals yesterday's closing price, 
then: Current OBV = Previous OBV

On-Balance Volume Example Calculation
Below is a list of 10 days' worth of a hypothetical stock's closing price and volume:

Day one: closing price equals $10, volume equals 25,200 shares
Day two: closing price equals $10.15, volume equals 30,000 shares
Day three: closing price equals $10.17, volume equals 25,600 shares
Day four: closing price equals $10.13, volume equals 32,000 shares
Day five: closing price equals $10.11, volume equals 23,000 shares
Day six: closing price equals $10.15, volume equals 40,000 shares
Day seven: closing price equals $10.20, volume equals 36,000 shares
Day eight: closing price equals $10.20, volume equals 20,500 shares
Day nine: closing price equals $10.22, volume equals 23,000 shares
Day 10: closing price equals $10.21, volume equals 27,500 shares

As can be seen, days two, three, six, seven and nine are up days, so these trading 
volumes are added to the OBV. Days four, five and 10 are down days, so these 
trading volumes are subtracted from the OBV. On day eight, no changes are made 
to the OBV since the closing price did not change. Given the days, the OBV for 
each of the 10 days is:

Day one OBV = 0
Day two OBV = 0 + 30,000 = 30,000
Day three OBV = 30,000 + 25,600 = 55,600
Day four OBV = 55,600 - 32,000 = 23,600
Day five OBV = 23,600 - 23,000 = 600
Day six OBV = 600 + 40,000 = 46,600
Day seven OBV = 46,600 + 36,000 = 76,600
Day eight OBV = 76,600
Day nine OBV = 76,600 + 23,000 = 99,600
Day 10 OBV = 99,600 - 27,500 = 72,100

'''
from tda_api_library import format_tda_datetime

from technical_analysis_utilities import increment_sample_counts

TIME_FRAME = 5
DIVERGENCE = 0.3

def calc_pct_change(df_data, idx, TIME_FRAME):

    if idx-TIME_FRAME >= 0:
        if not df_data.at[idx-TIME_FRAME, "OBV"] == 0:
            if df_data.at[idx, "OBV"] > 0:
                if df_data.at[idx-TIME_FRAME, "OBV"] > 0:
                    if df_data.at[idx, "OBV"] > df_data.at[idx-TIME_FRAME, "OBV"]:
                        #OBV +ve and increasing
                        obv_change = (df_data.at[idx, "OBV"] - df_data.at[idx-TIME_FRAME, "OBV"]) /  df_data.at[idx-TIME_FRAME, "OBV"]
                    else:
                        #OBV +ve and decreasing
                        obv_change = (df_data.at[idx-TIME_FRAME, "OBV"] - df_data.at[idx, "OBV"]) /  df_data.at[idx-TIME_FRAME, "OBV"]
                else:
                    #OBV transitioned from -ve to +ve
                    obv_change = 1.0
            else:
                if df_data.at[idx-TIME_FRAME, "OBV"] > 0:
                    #OBV transitioned from +ve to -ve
                    obv_change = 1.0
                else:
                    if df_data.at[idx, "OBV"] > df_data.at[idx-TIME_FRAME, "OBV"]:
                        #OBV -ve and increasing
                        obv_change = (df_data.at[idx-TIME_FRAME, "OBV"] - df_data.at[idx, "OBV"]) /  df_data.at[idx-TIME_FRAME, "OBV"]
                    else:
                        #OBV -ve and decreasing
                        obv_change = (df_data.at[idx, "OBV"] - df_data.at[idx, "OBV"]) /  df_data.at[idx-TIME_FRAME, "OBV"]
        else:
            obv_change = 0.0

        if not df_data.at[idx-TIME_FRAME, "Close"] == 0:
            close_change = (df_data.at[idx, "Close"] - df_data.at[idx-TIME_FRAME, "Close"]) /  df_data.at[idx-TIME_FRAME, "Close"]
        else:
            close_change = 0.0
    else:
        obv_change = 0.0
        close_change = 0.0
        
    return obv_change, close_change

def trade_on_obv(guidance, symbol, df_data):
    trigger_status = ""
    trade = False
    trigger_date = ""
    idx = df_data.shape[0] - 1

    obv_change, close_change = calc_pct_change(df_data, idx, TIME_FRAME)
    if obv_change == 0 and close_change > 0:
        if close_change > DIVERGENCE:
            trade = True
            trigger_status = "OBV5"
    elif obv_change == 0 and close_change < 0:
        if close_change + DIVERGENCE < 0:
            trade = True
            trigger_status = "OBV6"

    if trade:
        trigger_date = format_tda_datetime(df_data.at[df_data.shape[0]-1, 'DateTime'])
        guidance = guidance.append([[trade, symbol, 'OBV', trigger_date, trigger_status, df_data.at[idx, "Close"]]])
    return guidance

def eval_on_balance_volume(symbol, df_data, eval_results):
    obv1_index = ["OBV", "OBV1"]
    obv2_index = ["OBV", "OBV2"]
    obv3_index = ["OBV", "OBV3"]
    obv4_index = ["OBV", "OBV4"]
    obv5_index = ["OBV", "OBV5"]
    obv6_index = ["OBV", "OBV6"]
    obv7_index = ["OBV", "OBV7"]
    obv8_index = ["OBV", "OBV8"]
    obv9_index = ["OBV", "OBV9"]
    obv10_index = ["OBV", "OBV10"]
        
    idx = TIME_FRAME + 1
    while idx < len(df_data):
        obv_change, close_change = calc_pct_change(df_data, idx, TIME_FRAME)

        if obv_change > 0 and close_change > 0:
            if close_change + DIVERGENCE > obv_change:
                eval_results = increment_sample_counts(symbol, eval_results, obv1_index, df_data.iloc[idx, :]) 
            elif obv_change + DIVERGENCE > close_change:
                eval_results = increment_sample_counts(symbol, eval_results, obv2_index, df_data.iloc[idx, :]) 
        elif obv_change > 0 and close_change == 0:
            if close_change > DIVERGENCE:
                eval_results = increment_sample_counts(symbol, eval_results, obv3_index, df_data.iloc[idx, :]) 
        elif obv_change > 0 and close_change < 0:
            if close_change + DIVERGENCE < obv_change:
                eval_results = increment_sample_counts(symbol, eval_results, obv4_index, df_data.iloc[idx, :]) 
        elif obv_change == 0 and close_change > 0:
            if close_change > DIVERGENCE:
                eval_results = increment_sample_counts(symbol, eval_results, obv5_index, df_data.iloc[idx, :]) 
        elif obv_change == 0 and close_change < 0:
            if close_change + DIVERGENCE < 0:
                eval_results = increment_sample_counts(symbol, eval_results, obv6_index, df_data.iloc[idx, :]) 
        elif obv_change < 0 and close_change > 0:
            if obv_change + DIVERGENCE < close_change:
                eval_results = increment_sample_counts(symbol, eval_results, obv7_index, df_data.iloc[idx, :]) 
        elif obv_change < 0 and close_change == 0:
            if obv_change + DIVERGENCE < 0:
                eval_results = increment_sample_counts(symbol, eval_results, obv8_index, df_data.iloc[idx, :]) 
        elif obv_change < 0 and close_change < 0:
            if close_change + DIVERGENCE < obv_change:
                eval_results = increment_sample_counts(symbol, eval_results, obv9_index, df_data.iloc[idx, :]) 
            elif obv_change + DIVERGENCE < close_change:
                eval_results = increment_sample_counts(symbol, eval_results, obv10_index, df_data.iloc[idx, :]) 
        
        idx += 1
        
    return eval_results

def on_balance_volume(df_data=None, value_label=None, volume_lable=None):
    obv_colnum = 0
    df_data.insert(loc=obv_colnum, column='OBV', value=int(0))
    
    i_ndx = 0
    for col in df_data.columns:
        if col == 'Close':            
            value_col = i_ndx
        if col == 'Volume':
            vol_col = i_ndx
        i_ndx += 1
    
    idx = 1
    while idx < len(df_data):
        if df_data.iat[idx, value_col] == df_data.iat[idx-1, value_col]:
            # no price change
            df_data.iat[idx, obv_colnum] = int(df_data.iat[idx-1, obv_colnum])
        elif df_data.iat[idx, value_col] > df_data.iat[idx-1, value_col]:
            # price increase
            df_data.iat[idx, obv_colnum] = int(df_data.iat[idx-1, obv_colnum] + df_data.iat[idx, vol_col])
        else:
            # price decrease
            df_data.iat[idx, obv_colnum] = int(df_data.iat[idx-1, obv_colnum] - df_data.iat[idx, vol_col])
            
        idx += 1
    
    return df_data
