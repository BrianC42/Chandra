'''
Created on Jan 31, 2018

@author: Brian

http://www.investopedia.com/terms/a/adx.asp?ad=dirN&qo=investopediaSiteSearch&qsrc=0&o=40186

What is the 'Average Directional Index - ADX'
The average directional index (ADX) is an indicator used in technical analysis as an objective 
value for the strength of a trend. ADX is non-directional, so it quantifies a trend's 
strength regardless of whether it is up or down. It is usually plotted in a chart window 
along with two lines known as the DMI (Directional Movement Indicators).

BREAKING DOWN 'Average Directional Index - ADX'
Analysis of ADX is a method of evaluating trends and can help traders choose the strongest trends.
There are two forms of stock analysis: fundamental and technical. Fundamental analysis 
selects stocks based on business performance, whereas technical analysis selects stocks 
based on price movement. When technicians look for patterns over time to confirm a continuation 
or trend reversal, they often use the average directional index as an indicator. Developed by 
Welles Wilder for use with commodities and daily prices, the indicator is also used for 
stock selection. Wilder used the plus directional movement (+DM) and the minus direction 
movement (-DM) to determine the ADX.

The ADX
Traders believe that the trend is your friend. As a result, there are numerous trading 
indicators that are meant to confirm a trend. Once the trend is identified, however, the 
challenge is determining the best time to enter and exit a trade. The ADX is used to measure 
the strength or weakness of a trend and is therefore used alongside the +DM and -DM to 
determine the best course of action. Put together, Wilder created a framework for trading 
out of the ADX, DI+ and DI- trio of lines.

ADX Strategies
Traders start by using the ADX to determine if there is a trend. A strong trend is occurring 
when the ADX is over 25; likewise, there is no trend when the ADX falls below 20. When 
the +DI line is greater than the -DI line, the bulls have the directional edge. However, 
when the -DI line is above the +DI line, the bears have the directional edge. As with all 
technical trends, traders use several indicators to confirm a movement. One option is to sell 
when -DI is up and the major trend is down. Another option is to buy when +DI is higher 
than -DI, but only when the larger trend is also moving up. In other words, it is possible 
to use the ADX as a way to time an entry on a market that is already confirmed to be trading 
in a particular direction.

'''
from technical_analysis_utilities import add_results_index
from technical_analysis_utilities import find_sample_index

def eval_average_directional_index(df_data, eval_results):
    result_index = 'Average Directional Index, TBD'
    if not result_index in eval_results.index:
        combo_results = add_results_index(eval_results, result_index)
        eval_results = eval_results.append(combo_results)
        
    rows = df_data.iterrows()
    for nrow in rows:
        if True:
            cat_str = find_sample_index(eval_results, nrow[1]['10 day change'])
            eval_results.at[result_index, cat_str] += 1
    return eval_results

def add_adi_fields(df_data):
    df_data.insert(loc=0, column='Average Directional Index', value=0.0)
    return df_data

def average_directional_index(df_data=None):
    add_adi_fields(df_data)
    
    return df_data
