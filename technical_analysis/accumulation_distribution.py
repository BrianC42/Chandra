'''
Created on Jan 31, 2018

@author: Brian

DEFINITION of 'Accumulation/Distribution'
http://www.investopedia.com/terms/a/accumulationdistribution.asp-0

An indicator that tracks the relationship between volume and price. It is often 
considered a leading indicator because it shows when a stock is being accumulated 
or distributed, foreshadowing major price moves.

When the Accumulation/Distribution line is moving in the same direction as the price trend, 
it confirms the trend. When the Accumulation/Distribution line is moving in the opposite 
direction of the price trend, it indicates the price trend may not be sustainable.

The indicator has a three step calculation:
  1. Money Flow Multiplier = [(close  -  low) - (high - close)] /(high - low) 
  2. Money Flow Volume = Money Flow Multiplier x volume for the period
  3. Accumulation/Distribution= previous Accumulation/Distribution + current period's Money Flow Volume
 
BREAKING DOWN 'Accumulation/Distribution'
The indicator is primarily used to confirm price trends, or spot potential trend 
changes in price based on divergence with the Accumulation/Distribution line. 
The indicator looks at each period individually, and whether the price closes in 
the upper or lower portion of the period's (day's) range. This means not all 
divergences will result in a price trend reversal. Occasionally anomalies will 
occur where the price is trending lower (higher) but the indicator is rising (falling), 
because even though the price trend is down, each day the price is finishing in the 
upper portion of its daily range. High volume days can accentuate this characteristic. 

Therefore, the indicator is different from other cumulative volume indicators, such as On-Balance Volume (OBV). 

'''
def add_acc_dist_fields(df_data):
    '''
    Additional fields added to the digest are
        accumulation distribution fields: 
            AccumulationDistribution
    '''
    df_data.insert(loc=0, column='AccumulationDistribution', value=0.0)

    return (df_data)

def accumulation_distribution(df_data=None):
    add_acc_dist_fields(df_data)
    
    '''
    idx = 1
    while idx < len(df_data):
        # 1. Money Flow Multiplier = [(close  -  low) - (high - close)] /(high - low)
        p_range = df_data.ix[idx, 'adj_high'] - df_data.ix[idx, 'adj_low']
        if p_range == 0.0:
            mfm = 0.0
        else:
            mfm = ((df_data.ix[idx, 'adj_close'] - df_data.ix[idx, 'adj_low']) - \
                   (df_data.ix[idx, 'adj_high'] - df_data.ix[idx, 'adj_close'])) / \
                   (p_range)
        # 2. Money Flow Volume = Money Flow Multiplier x volume for the period
        mfv = mfm * df_data.ix[idx, 'adj_volume']
        # 3. Accumulation/Distribution= previous Accumulation/Distribution + current period's Money Flow Volume
        df_data.ix[idx, 'AccumulationDistribution'] = df_data.ix[idx-1, 'AccumulationDistribution'] + mfv
        
        idx += 1
    '''

    return df_data