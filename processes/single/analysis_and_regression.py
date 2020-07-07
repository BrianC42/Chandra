'''
Created on Jan 31, 2018

@author: Brian
'''
import pandas as pd
import sys
from configuration import get_ini_data

sys.path.append("../Technical_Analysis/")

from classification_analysis import classification_and_regression

print ("Affirmative, Dave. I read you\n")

#surface_data_dir = "c:\\users\\brian\\documents\\Quandl\\"
#NY_data_dir = "d:\\D2\\Financial Data\\Quandl\\Stock Price Historical Data\\"
#NJ_data_dir = "d:\\Brian\\AI Projects\\Quandl\\"
#data_file = NJ_data_dir + "Enhanced historical data.csv"
data_loc = get_ini_data("DEVDATA")
data_file = data_loc['dir'] + "\\Enhanced historical data.csv"

df_historical_data = pd.read_csv(data_file)
print ("\ndf_historical_data data head:\nShape:", df_historical_data.shape, "\n", df_historical_data.head(4))
print ("\ndf_historical_data data tail:\n", df_historical_data.tail(4))
#print "\nrow 999997:\n", df_historical_data.loc[0]

df_historical_data = classification_and_regression(df_historical_data)

print ("\nDave, this conversation can serve no purpose anymore. Goodbye")
