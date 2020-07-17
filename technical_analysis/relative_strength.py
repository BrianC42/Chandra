'''
Created on Jan 31, 2018

@author: Brian

http://www.investopedia.com/ask/answers/06/relativestrength.asp?ad=dirN&qo=investopediaSiteSearch&qsrc=0&o=40186
What is relative strength?

Relative strength is a measure of the price trend of a stock or other financial instrument 
compared to another stock, instrument or industry. It is calculated by taking the price 
of one asset and dividing it by another.

For example, if the price of Ford shares is $7 and the price of GM shares is $25, the 
relative strength of Ford to GM is 0.28 ($7/25). This number is given context when it is 
compared to the previous levels of relative strength. If, for example, the relative 
strength of Ford to GM ranges between 0.5 and 1 historically, the current level of 0.28 
suggests that Ford is undervalued or GM is overvalued, or a mix of both. The reason we 
know this is because the only way for this ratio to increase back to its normal historical 
range is for the numerator (number on the top of the ratio, in this case the price of Ford) 
to increase, or the denominator (number on the bottom of the ratio, in our case the price of GM) 
to decrease. It should also be noted that the ratio can also increase by combining an 
upward price move of Ford with a downward price move of GM. For example, if Ford shares 
rose to $14 and GM shares fell to $20, the relative strength would be 0.7, which is near 
the middle of the historic trading range.

It is by comparing the relative strengths of two companies that a trading opportunity, 
known as pairs trading, is realized. Pairs trading is a strategy in which a trader 
matches long and short positions of two stocks that are perceived to have a strong correlation 
to each other and are currently trading outside of their historical relative strength range. 
For example, in the case of the Ford/GM relative strength at 0.28, a pairs trader would enter 
a long position in Ford and short GM if he or she felt the pair would move back toward its historical range.

'''
import os
import pandas as pd
from numpy import NaN

from moving_average import simple_moving_average
from technical_analysis_utilities import add_results_index
from technical_analysis_utilities import find_sample_index

def eval_relative_strength(df_data, eval_results):
    result_index = 'Relative Strength, RS<RS SMA20, 10 day'
    rs_mean_index = 'Relative Strength, RS<RS mean, 10 day'
    if not result_index in eval_results.index:
        combo_results = add_results_index(eval_results, result_index)
        eval_results = eval_results.append(combo_results)
        combo_results = add_results_index(eval_results, rs_mean_index)
        eval_results = eval_results.append(combo_results)

    rs_mean = df_data.get('Relative Strength').mean()
    rows = df_data.iterrows()
    for nrow in rows:
        if nrow[1]['Relative Strength'] < nrow[1]['RS SMA20']:
            cat_str = find_sample_index(eval_results, nrow[1]['10 day change'])
            eval_results.at[result_index, cat_str] += 1
        if nrow[1]['Relative Strength'] < rs_mean:
            cat_str = find_sample_index(eval_results, nrow[1]['10 day change'])
            eval_results.at[rs_mean_index, cat_str] += 1
    return eval_results

def add_relative_strength(df_data=None):
    df_data.insert(loc=0, column='Relative Strength', value=0.0)
    df_data.insert(loc=0, column='RS SMA20', value=NaN)
    return df_data

def relative_strength(df_data=None, value_label=None, relative_to=None):
    add_relative_strength(df_data)
    
    if os.path.isfile(relative_to):
        df_comp = pd.read_csv(relative_to)
        data_ndx = 1
        comp_ndx = 1
        while data_ndx < len(df_data):
            matched = False
            if comp_ndx < len(df_comp):
                if df_data.at[data_ndx, 'DateTime'] == df_comp.at[comp_ndx + 1, 'DateTime']:
                    comp_ndx += 1
                    matched = True
            if not matched:
                comp_ndx = 1
                while comp_ndx < len(df_comp):
                    if df_data.at[data_ndx, 'DateTime'] == df_comp.at[comp_ndx, 'DateTime']:
                        matched = True
                        break
                    comp_ndx += 1
            if matched:
                df_data.at[data_ndx, 'Relative Strength'] = df_data.at[data_ndx, value_label] / df_comp.at[comp_ndx, value_label]
            '''
            print("data date: %s value %s\ncomp date: %s value: %s\nRelative strength: %s" % \
                  (df_data.at[data_ndx, 'DateTime'], df_data.at[data_ndx, value_label], \
                   df_comp.at[comp_ndx, 'DateTime'], df_comp.at[comp_ndx, value_label], \
                   df_data.at[data_ndx, value_label] / df_comp.at[comp_ndx, value_label]))
            '''
            data_ndx += 1
    df_data = simple_moving_average(df_data[:], value_label="Relative Strength", avg_interval=20, SMA_data_label='RS SMA20')

    return df_data