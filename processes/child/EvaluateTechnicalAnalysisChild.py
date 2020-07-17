'''
Created on Jul 16, 2020

@author: Brian
'''
import os
import pandas as pd

from technical_analysis_utilities import initialize_eval_results
from technical_analysis_utilities import add_results_index
from technical_analysis_utilities import find_sample_index
from technical_analysis_utilities import present_evaluation
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
            
            '''================ Perform technical analysis ==================='''
            file_in = p_list[0] + '\\' + p_list[1] + '.csv'
            if os.path.isfile(file_in):
                print("Evaluating %s" % file_in)
                df_data = pd.read_csv(file_in)
                eval_results = initialize_eval_results()
                eval_results = sample_count(df_data, eval_results)
                eval_results = eval_macd_positive_cross(df_data, eval_results)
                eval_results = eval_bollinger_bands(df_data, eval_results)
                eval_results = eval_on_balance_volume(df_data, eval_results)
                eval_results = eval_relative_strength(df_data, eval_results)
                eval_results = eval_stochastic_oscillator(df_data, eval_results)
                eval_results = eval_average_directional_index(df_data, eval_results)
                eval_results = eval_aroon_indicator(df_data, eval_results)
                eval_results = eval_accumulation_distribution(df_data, eval_results)
                eval_results = eval_combinations(df_data, eval_results)

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