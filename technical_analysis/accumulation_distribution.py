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
from technical_analysis_utilities import add_results_index
from technical_analysis_utilities import find_sample_index

def eval_accumulation_distribution(df_data, eval_results):
    result_index = 'Accumulation Distribution, AD>0, 10 day'
    if not result_index in eval_results.index:
        combo_results = add_results_index(eval_results, result_index)
        eval_results = eval_results.append(combo_results)
        
    rows = df_data.iterrows()
    for nrow in rows:
        if nrow[1]['AccumulationDistribution'] > 0:
            cat_str = find_sample_index(eval_results, nrow[1]['10 day change'])
            eval_results.at[result_index, cat_str] += 1
    return eval_results

def add_acc_dist_fields(df_data):
    df_data.insert(loc=0, column='AccumulationDistribution', value=0.0)

    return (df_data)

def accumulation_distribution(df_data=None):
    add_acc_dist_fields(df_data)
    
    data_ndx = 1
    while data_ndx < len(df_data):
        # 1. Money Flow Multiplier = [(close  -  low) - (high - close)] /(high - low)
        p_range = df_data.at[data_ndx, 'High'] - df_data.at[data_ndx, 'Low']
        if p_range == 0.0:
            mfm = 0.0
        else:
            mfm = ((df_data.at[data_ndx, 'Close'] - df_data.at[data_ndx, 'Low']) - \
                   (df_data.at[data_ndx, 'High'] - df_data.at[data_ndx, 'Close'])) / \
                   (p_range)
        # 2. Money Flow Volume = Money Flow Multiplier x volume for the period
        mfv = mfm * df_data.at[data_ndx, 'Volume']
        # 3. Accumulation/Distribution= previous Accumulation/Distribution + current period's Money Flow Volume
        df_data.at[data_ndx, 'AccumulationDistribution'] = df_data.at[data_ndx-1, 'AccumulationDistribution'] + mfv
        data_ndx += 1

    return df_data