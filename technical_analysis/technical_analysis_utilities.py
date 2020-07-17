'''
Created on Jul 16, 2020

@author: Brian
'''
import pandas as pd
import numpy as np

def present_evaluation(eval_results):
    print("Evaluation results of technical analysis\n%s" % eval_results.to_string())
    return

def find_sample_index(eval_results, data_point):
    cat_range = 'Neutral'
    if not np.isnan(data_point):
        for cat_range in eval_results:
            cat_max = eval_results.at['Range Max', cat_range]
            cat_min = eval_results.at['Range Min', cat_range]
            if data_point > cat_min and data_point < cat_max:
                break
    return cat_range

def add_results_index(df_results, ndx_label):
    zero_data = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    result_cols = df_results.columns
    expanded_results = pd.DataFrame(data = zero_data, index=[ndx_label], columns=result_cols)
    return expanded_results

def initialize_eval_results():
    eval_results = pd.DataFrame([[-1000.0, -1.0, -0.5, -0.2, -0.1,  -0.05, -0.01, 0.01, 0.05, 0.1, 0.2, 0.5,    1.0], \
                                 [-   1.0, -0.5, -0.2, -0.1, -0.05, -0.01,  0.01, 0.05, 0.1,  0.2, 0.5, 1.0, 1000.0]], \
                                index=['Range Min', 'Range Max'], \
                                columns=['Neg', '-5', '-4', '-3', '-2', '-1', 'Neutral', '1', '2', '3', '4', '5', 'Pos'])
    
    return eval_results

def add_evaluation_results(cumulative_evaluation, incremental_evaluation):
    dont_accumulate = initialize_eval_results()
    dont_acc_ndx = dont_accumulate.index
    cum_indices = cumulative_evaluation.index
    inc_indices = incremental_evaluation.index
    for ndx in inc_indices:
        if not ndx in dont_acc_ndx:
            if not ndx in cum_indices:
                cumulative_evaluation = cumulative_evaluation.append(add_results_index(cumulative_evaluation, ndx))
            for col_ndx in incremental_evaluation.columns:
                cumulative_evaluation.at[ndx, col_ndx] = cumulative_evaluation.at[ndx, col_ndx] + incremental_evaluation.at[ndx, col_ndx]
    return cumulative_evaluation

def sample_count(df_data, eval_results):
    r1_index = 'Baseline 1 day'
    r5_index = 'Baseline 5 day'
    r10_index = 'Baseline 10 day'
    r20_index = 'Baseline 20 day'
    if not r1_index in eval_results.index:
        eval_results = eval_results.append(add_results_index(eval_results, r1_index))
        eval_results = eval_results.append(add_results_index(eval_results, r5_index))
        eval_results = eval_results.append(add_results_index(eval_results, r10_index))
        eval_results = eval_results.append(add_results_index(eval_results, r20_index))
        
    rows = df_data.iterrows()
    for nrow in rows:
        if True:
            eval_results.at[r1_index, find_sample_index(eval_results, nrow[1]['1 day change'])] += 1
            eval_results.at[r5_index, find_sample_index(eval_results, nrow[1]['5 day change'])] += 1
            eval_results.at[r10_index, find_sample_index(eval_results, nrow[1]['10 day change'])] += 1
            eval_results.at[r20_index, find_sample_index(eval_results, nrow[1]['20 day change'])] += 1
    return eval_results

def eval_combinations(df_data, eval_results):
    result_index = 'combination1 MACD<0.25+Close>SMA20, 10 day'
    if not result_index in eval_results.index:
        combo_results = add_results_index(eval_results, result_index)
        eval_results = eval_results.append(combo_results)
        
    rows = df_data.iterrows()
    for nrow in rows:
        if nrow[1]['MACD_Buy'] == True:
            if not np.isnan(nrow[1]['MACD']) and \
                nrow[1]['MACD'] < 0.25 and \
                nrow[1]['Close'] > nrow[1]['SMA20']:
                cat_str = find_sample_index(eval_results, nrow[1]['10 day change'])
                eval_results.at[result_index, cat_str] += 1
    return eval_results
