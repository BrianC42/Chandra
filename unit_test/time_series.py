'''
Created on Jul 20, 2018

@author: Brian
'''

import logging
from pandas import DataFrame
from pandas import concat
from quandl_library import read_symbol_historical_data
from quandl_library import get_ini_data
from time_series_data import series_to_supervised
from lstm import prepare_ts_lstm

if __name__ == '__main__':
    pass

'''
Initialize logging
'''
lstm_config_data = get_ini_data("LSTM")
log_file = lstm_config_data['log']
logging.basicConfig(filename=log_file, level=logging.DEBUG, format='%(asctime)s: %(levelname)s: %(message)s')
print ("Logging to", log_file)
logger = logging.getLogger('lstm_logger')
log_fmt = logging.Formatter('%(asctime)s - %(name)s - %levelname - %(messages)s')

logger.info('')
logger.info('============================================')
logger.info('Time series data preparation for LSTM')
logger.info('============================================')

'''
logger.info('')
logger.info('============================================')
logger.info('Single variant')
logger.info('============================================')
values = [x for x in range(10)]
df_data = series_to_supervised(values, 3)
logger.info('')
logger.info('Number of lag observations=3')
logger.info('series_to_supervised(values, 3)')
logger.info(df_data)

values = [x for x in range(10)]
df_data = series_to_supervised(values, 2, 2)
logger.info('')
logger.info('Number of lag observations=2, Number of observations as output=2')
logger.info('series_to_supervised(values, 2, 2)')
logger.info(df_data)

logger.info('')
logger.info('============================================')
logger.info('Multi-variant')
logger.info('============================================')
raw = DataFrame()
raw['ob1'] = [x for x in range(10)]
raw['ob2'] = [x for x in range(50, 60)]
values = raw.values
df_data = series_to_supervised(values)
logger.info('')
logger.info('Number of lag observations=default(1), Number of observations as output=default(1)')
logger.info('series_to_supervised(values)')
logger.info(df_data)

raw = DataFrame()
raw['ob1'] = [x for x in range(10)]
raw['ob2'] = [x for x in range(50, 60)]
values = raw.values
df_data = series_to_supervised(values, 1, 2)
logger.info('')
logger.info('Number of lag observations=1, Number of observations as output=2')
logger.info('series_to_supervised(values, 1, 2)')
logger.info(df_data)
'''

logger.info('')
logger.info('============================================')
logger.info('Historical market data')
logger.info('============================================')
x_train, y_train, x_test, y_test = prepare_ts_lstm('ibm', 120, 30)

print('all done')

