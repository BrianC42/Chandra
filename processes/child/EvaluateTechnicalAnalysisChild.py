'''
Created on Jul 16, 2020

@author: Brian
'''
import os
import pandas as pd

from technical_analysis_utilities import initialize_eval_results
from technical_analysis_utilities import sample_count
from technical_analysis_utilities import eval_combinations
from macd import eval_macd_positive_cross
from bollinger_bands import eval_bollinger_bands
from on_balance_volume import eval_on_balance_volume
from relative_strength import eval_relative_strength
from stochastic_oscillator import eval_stochastic_oscillator
from average_directional_index import eval_average_directional_index
from accumulation_distribution import  eval_accumulation_distribution
from aroon_indicator import eval_aroon_indicator

def EvaluateTechnicalAnalysisChild(pipe_in):
    
    try:
        while True:
            '''================ Work on data as long as the manager has more
            ==============================================================='''
            p_list = pipe_in.recv()
            #print("Worker instructions: %s" % p_list)            
            symbol = p_list[1]
            
            '''================ Perform technical analysis ==================='''
            file_in = p_list[0] + '\\' + symbol + '.csv'
            if os.path.isfile(file_in):
                print("Evaluating %s" % file_in)
                df_data = pd.read_csv(file_in)
                eval_results = initialize_eval_results()
                eval_results = sample_count(symbol, df_data, eval_results)
                eval_results = eval_macd_positive_cross(symbol, df_data, eval_results)
                eval_results = eval_bollinger_bands(symbol, df_data, eval_results)
                eval_results = eval_on_balance_volume(symbol, df_data, eval_results)
                eval_results = eval_relative_strength(symbol, df_data, eval_results)
                eval_results = eval_stochastic_oscillator(symbol, df_data, eval_results)
                eval_results = eval_accumulation_distribution(symbol, df_data, eval_results)
                eval_results = eval_combinations(symbol, df_data, eval_results)
                eval_results = eval_average_directional_index(symbol, df_data, eval_results)
                eval_results = eval_aroon_indicator(symbol, df_data, eval_results)

            segmentation = p_list[2]
            if os.path.isfile(segmentation):
                #print("reading segmentation information from %s" % segmentation)
                df_segmentation = pd.read_csv(segmentation)
                df_segmentation = df_segmentation.set_index('Symbol')
                if symbol in df_segmentation.index:
                    classification = df_segmentation.loc[symbol, 'Classification']
                    segment = df_segmentation.loc[symbol, 'Segment']
                else:
                    classification = "TBD"
                    segment = "TBD"
                ndx = 0
                while ndx < eval_results.shape[0]:
                    if not eval_results.at[ndx, 'symbol'] == 'Baseline':
                        eval_results.at[ndx, 'segment'] = segment
                        eval_results.at[ndx, 'classification'] = classification
                    ndx += 1
                    
            '''================ Return processed data ====================='''
            pipe_in.send(eval_results)
            
    except EOFError:
        #print ("The Master has no more work for me ...")
        pass
        
    finally:
        #print ("Cleaning up after myself")
        pass

    #print ("Worker completing")

    return True