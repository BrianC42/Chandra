'''
Created on Jan 31, 2018

@author: Brian

What is the 'Aroon Indicator'
The Aroon indicator is a technical indicator used for identifying trends in an underlying 
security and the likelihood that the trends will reverse. It is made up of two lines: 
one line is called "Aroon up", which measures the strength of the uptrend, and 
the other line is called "Aroon down", which measures the downtrend. 
The indicator reports the time it is taking for the price to reach, from a starting point, 
the highest and lowest points over a given time period, each reported as a percentage of total time.

BREAKING DOWN 'Aroon Indicator'
The Aroon indicator was developed by Tushar Chande in 1995. Both the Aroon up and the Aroon down 
fluctuate between zero and 100, with values close to 100 indicating a strong trend, and zero 
indicating a weak trend. The lower the Aroon up, the weaker the uptrend and the stronger the 
downtrend, and vice versa. The main assumption underlying this indicator is that a stock's 
price will close at record highs in an uptrend, and record lows in a downtrend.

This indicator is very similar to the directional movement index (DMI) that was developed by 
Welles Wilder, which is also a very popular indicator used to measure the strength of a given trend.

'''
from technical_analysis_utilities import add_results_index
from technical_analysis_utilities import find_sample_index

def eval_aroon_indicator(df_data, eval_results):
    result_index = 'Aroon Indicator, TBD'
    if not result_index in eval_results.index:
        combo_results = add_results_index(eval_results, result_index)
        eval_results = eval_results.append(combo_results)
        
    rows = df_data.iterrows()
    for nrow in rows:
        if False:
            cat_str = find_sample_index(eval_results, nrow[1]['20 day change'])
            eval_results.at[result_index, cat_str] += 1
    return eval_results

def add_aroon_fields(df_data=None):
    df_data.insert(loc=0, column='Aroon Indicator', value=0.0)
    return df_data

def aroon_indicator(df_data=None):
    add_aroon_fields(df_data)

    return df_data
