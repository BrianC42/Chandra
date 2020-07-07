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
def add_obv_fields(df_data, colnum=0):
    '''
    Only one additional field is added to the digest
        OBV field: 
            OBV
        Other fields
            N/A 
    '''
    df_data.insert(loc=colnum, column='OBV', value=int(0))

    return (df_data)

def on_balance_volume(df_data=None, value_label=None, volume_lable=None):
    obv_colnum = 0
    add_obv_fields(df_data, obv_colnum)
    print("columns %s" % df_data.columns)
    
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
