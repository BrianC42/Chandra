'''
Created on Jul 17, 2020

@author: Brian

********************************************************************
scan TD Ameritrade watch lists:
1. Assess symbols against actionable trade triggers
2. evaluate ML predictive models
3. Option strategies of interest

********************************************************************
'''
import os
import datetime as dt
import time
import logging
import pandas as pd

from configuration import get_ini_data
from configuration import read_config_json

from tda_api_library import tda_get_authentication_details
from tda_api_library import tda_read_watch_lists
from tda_api_library import tda_read_option_chain
from tda_api_library import format_tda_datetime
from tda_api_library import tda_manage_throttling
from tda_api_library import covered_call
from tda_api_library import cash_secured_put

from tradingIndicationsML import tradingIndications

from macd import trade_on_macd
from macd import macd_trade_analysis
from bollinger_bands import trade_on_bb
from stochastic_oscillator import trade_on_stochastic_oscillator
from on_balance_volume import trade_on_obv
from relative_strength import trade_on_relative_strength

def calendar_spread():
    return

def diagonal_spread():
    return

def iron_condor():
    '''
    Consider a liquid stock or ETF with a high daily trading volume. Because this trade benefits when the underlying stays in the range
    between the two short strikes, consider stocks that are sideways-trending or that you have a neutral outlook on. Additionally,
    the probability of an iron condor having at least some success improves when the implied volatility of the underlying falls.
    To potentially capture this, consider looking for stocks with high implied volatility
    for example, those that are trading in the top 50% of their 52-week implied volatility range.
    When it comes to selecting the options themselves, look for options with narrow bid/ask spreads, high trading volumes, and large open interest. 
    This can help ensure you get the price you want for your transaction, which can be particularly important in this strategy as it has four legs.
    Penny increment options can be ideal as they can make it easier to enter, and potentially exit, the trade, but they're not available on every underlying.
    Sample entry rules involve selecting expirations and strike prices. When it comes to selecting your expiration, options that
    expire 20 to 50 days in the future can help you capture time decay for the trade. 
    Be sure to look at the financial calendar of the underlying and try to avoid options that expire close to earnings announcements
    or other major events. These events tend to cause implied volatility to rise  which can hurt the trade's success.
    We'll start with the short options because these set up the iron condor's body or the area where you can expect to achieve max gain. 
    To choose your short strike prices, a delta between .20 and .30 can be a good place to start because these options generally have a 
    lower probability of expiring in the money than options with higher deltas. As always, selecting your strike prices will likely
    be a trade-off between premium and chances of success. Options with a higher probability of expiring worthless (those with smaller deltas)
    will have lower premiums.
    It can be helpful to check these strikes against the resistance and support levels for the underlying to make sure the short strikes are near, 
    or outside, of those levels. You might consider selling a put spread below resistance and a call spread above it.
    To select the long option strikes, you'll have another trade-off, this time between cost and risk, The further apart the strikes,
    the greater the potential risk. However, options that are further out of the money tend to cost less, so your max gain might be higher.
    '''
    return

def assess_options_chanins(symbol, df_data, df_options):
    logger.info('assess_options chains ----> %s' % symbol)
    covered_call(symbol, df_data, df_options)
    cash_secured_put(symbol, df_data, df_options)
    logger.info('<---- assess_options chains')
    return
    
def assess_trading_signals(symbol, df_data):
    logger.info('assess_trading_signals ----> %s' % symbol)
    guidance = pd.DataFrame()
    guidance = trade_on_macd(guidance, symbol, df_data[:])
    guidance = trade_on_bb(guidance, symbol, df_data[:])
    guidance = trade_on_stochastic_oscillator(guidance, symbol, df_data)
    guidance = trade_on_obv(guidance, symbol, df_data)
    guidance = trade_on_relative_strength(guidance, symbol, df_data)

    logger.info('<---- assess_trading_signals')
    return guidance

def search_for_trading_opportunities(f_out, authentication_parameters, analysis_dir, json_config):
    print("\nScanning for technical analysis trading opportunities")
    logger.info('searching for trading opportunities ---->')
    localDirs = get_ini_data("LOCALDIRS")
    aiwork = localDirs['aiwork']
    guidance = pd.DataFrame()
    df_potential_strategies = pd.DataFrame()
    '''
    Trading signals based on technical analysis
    '''
    json_authentication = tda_get_authentication_details(authentication_parameters)
    potential_option_trades = aiwork + '\\' + json_config['potentialoptionstrades']
    for symbol in tda_read_watch_lists(json_authentication):
        #print("Assessing: %s" % symbol)
        filename = analysis_dir + '\\' + symbol + '.csv'
        if os.path.isfile(filename):
            df_data = pd.read_csv(filename)
            guidance = assess_trading_signals(symbol, df_data)
            for trigger in guidance.itertuples():
                report = '{:s}, {:>8s}, {:s}, {:>8.2f}, {:s}'.format(trigger[4], trigger[2], trigger[3], trigger[6], trigger[5])
                print(report)
                f_out.write(report + "\n")
                macd_trade_analysis(trigger, symbol, df_data)
                '''
                Additional trading strategy identification goes here
                '''
    '''
    Machine learning models
    '''
    tradingIndications()

    '''
    Covered call options
    '''
    callCount=0
    periodStart = time.time()
    callCount, periodStart = tda_manage_throttling(callCount, periodStart)

    print("\nAnalyzing potential covered calls")
    for symbol in tda_read_watch_lists(json_authentication, watch_list='Combined Holding'):
        callCount, periodStart = tda_manage_throttling(callCount, periodStart)
        df_options, options_json = tda_read_option_chain(authentication_parameters, symbol)
        filename = analysis_dir + '\\' + symbol + '.csv'
        if os.path.isfile(filename):
            df_data = pd.read_csv(filename)
        df_covered_calls = covered_call(symbol, df_data, df_options)
        if df_covered_calls.shape[0] > 0:
            df_potential_strategies = pd.concat([df_potential_strategies, df_covered_calls])

    '''
    Cash secured put options
    '''
    print("\nAnalyzing potential cash secured puts")
    for symbol in tda_read_watch_lists(json_authentication, watch_list='Potential Buy'):
        callCount, periodStart = tda_manage_throttling(callCount, periodStart)
        df_options, options_json = tda_read_option_chain(authentication_parameters, symbol)
        filename = analysis_dir + '\\' + symbol + '.csv'
        if os.path.isfile(filename):
            df_data = pd.read_csv(filename)
        df_cash_secured_puts = cash_secured_put(symbol, df_data, df_options)
        if df_cash_secured_puts.shape[0] > 0:
            df_potential_strategies = pd.concat([df_potential_strategies, df_cash_secured_puts])
              
    df_potential_strategies.to_csv(potential_option_trades + "potential_option_trades.csv", index=False)
    logger.info('<---- searching for trading opportunities done')
    return

if __name__ == '__main__':
    print ("Affirmative, Dave. I read you\n")
    '''
    Prepare the run time environment
    '''
    start = time.time()
    now = dt.datetime.now()
    
    localDirs = get_ini_data("LOCALDIRS")
    gitdir = localDirs['git']
    aiwork = localDirs['aiwork']
    
    # Get external initialization details
    app_data = get_ini_data("TDAMERITRADE")
    json_config = read_config_json(app_data['config'])

    try:    
        log_file = aiwork + '\\' + json_config['logFile']
        if json_config['loggingLevel'] == "debug":
            logging.basicConfig(filename=log_file, level=logging.DEBUG, format=json_config['loggingFormat'])
        elif json_config['loggingLevel'] == "info":
            logging.basicConfig(filename=log_file, level=logging.INFO, format=json_config['loggingFormat'])
        else:
            logging.basicConfig(filename=log_file, level=logging.WARNING, format=json_config['loggingFormat'])
            
        output_file = aiwork + '\\' + json_config['outputFile']
        output_file = output_file + ' {:4d} {:0>2d} {:0>2d} {:0>2d} {:0>2d} {:0>2d}'.format(now.year, now.month, now.day, \
                                                                                       now.hour, now.minute, now.second) + '.txt'
        f_out = open(output_file, 'w')    
        
        # global parameters
        #logging.debug("Global parameters")
    
    except Exception:
        print("\nAn exception occurred - log file details are missing from json configuration")
        
    print ("Logging to", log_file)
    logger = logging.getLogger('chandra_logger')
    log_fmt = logging.Formatter('%(asctime)s - %(name)s - %levelname - %(messages)s')
    logger.info('Updating stock data')

    #update_tda_eod_data(app_data['authentication'])
    search_for_trading_opportunities(f_out, app_data['authentication'], app_data['market_analysis_data'], json_config)
    
    '''
    clean up and prepare to exit
    '''
    f_out.close()

    print ("\nDave, this conversation can serve no purpose anymore. Goodbye")
