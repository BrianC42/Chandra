'''
Created on Jan 31, 2018

@author: Brian
'''
from macd import macd
from accumulation_distribution import accumulation_distribution
from aroon_indicator import aroon_indicator
from average_directional_index import average_directional_index
from on_balance_volume import on_balance_volume
from relative_strength import relative_strength
from stochastic_oscillator import stochastic_oscillator
from bollinger_bands import bollinger_bands

def quandl_worker_pipe(mpipe_feed):
    print ("Worker starting")
    
    try:
        while True:
            '''================ Work on data as long as the manager has more
            ==============================================================='''
            df_data = mpipe_feed.recv()
            #print ("Data from feeder:")
            #print (data)
            
            '''================ Perform technical analysis ===================
            implemented
                macd
                accumulation_distribution
                on_balance_volume
                bollinger_bands                
            '''
            df_data = macd(df_data[:], value_label="adj_close", \
                           short_interval=12, short_data='EMA12',\
                           long_interval=26, long_data='EMA26', \
                           date_label="date", \
                           prediction_interval=30)
            df_data = on_balance_volume(df_data[:])
            df_data = accumulation_distribution(df_data[:])
            df_data = bollinger_bands(df_data[:], value_label="adj_close", \
                                      sma_interval=20, sma_label='SMA20')
            '''
            future development
                aroon_indicator
                average_directional_index
                relative_strength
                stochastic_oscillator
            ==============================================================='''
            df_data = aroon_indicator(df_data[:])
            df_data = average_directional_index(df_data[:])
            df_data = relative_strength(df_data[:])
            df_data = stochastic_oscillator(df_data[:])

            '''================ Return processed data ====================='''
            mpipe_feed.send(df_data)
            
    except EOFError:
        print ("The Master has no more work for me ...")
        
    finally:
        print ("Cleaning up after myself")

    print ("Worker completing")

    return