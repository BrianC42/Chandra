'''
Created on Jan 31, 2018

@author: Brian
'''
import sys
sys.path.append("../Technical_Analysis/")
sys.path.append("../Utilities/")
import datetime

from technical_analysis import perform_technical_analysis
from quandl_library import read_historical_data
from quandl_library import save_enhanced_historical_data

print ("Affirmative, Dave. I read you\n")

'''
    Read historical data sources
''' 
df_data = read_historical_data()

'''
    Add elements required and perform technical analyses
'''
s_time = datetime.datetime.now()

df_tickers = df_data.drop_duplicates("ticker") 
print ("tickers discovered:", df_tickers.shape[0])

perform_technical_analysis(df_data, df_tickers)

e_time = datetime.datetime.now()
print (s_time, e_time)

'''
    Save enhanced data file
'''
save_enhanced_historical_data(df_data)

print ("\nDave, this conversation can serve no purpose anymore. Goodbye")
