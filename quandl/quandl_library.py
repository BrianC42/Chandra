'''
Created on Jan 31, 2018

@author: Brian
'''
import os
import datetime
import configparser
import quandl
import pandas as pd
from numpy.distutils.fcompiler import none

def get_devdata_dir():
    devdata = get_ini_data('DEVDATA')
    devdata_dir = devdata['dir']
    return devdata_dir

def get_ini_data(csection):
    config_file = os.getenv('localappdata') + "\\AI Projects\\data.ini"
    config = configparser.ConfigParser()
    config.read(config_file)
    config.sections()
    ini_data = config[csection]
    return ini_data

def get_quandl_key():
    quandl_data = get_ini_data("QUANDL")
    quandl.ApiConfig.api_key = quandl_data['key']
    
    return (quandl)

def ticker_status_dict():
    StatusDict = dict(inactive=-1, unknown=0, active=1)
    
    return(StatusDict)

def quandl_data_dict():
    QColsDict = dict({'ticker' : 1, \
                'date' : 2, \
                'open' : 3, \
                'high' : 4, \
                'low' : 5, \
                'close' : 6, \
                'volume' : 7, \
                'ex-dividend' : 8, \
                'split_ratio' : 9, \
                'adj_open' : 10, \
                'adj_high' : 11, \
                'adj_low' : 12, \
                'adj_close' : 13, \
                'adj_volume' : 14 })
    
    return(QColsDict)

def update_eod_data():
    '''
    https://www.quandl.com/api/v3/datatables/WIKI/PRICES/delta.json?api_key=1wWyDnk64NgsEtxwg1Lb
    
    The latest_full_data corresponds to a full dump of the data in the table at the time specified. 
    In the above example this equals 2016-09-04T18h37m34.

    files is a list of the delta files. WIKI has a file for insertions, 
    updates and deletions created every day. One or more of the delta files can be empty 
    if no changes have been made to the table. 
    from gives the time when the previous set of delta files were created. 
    to corresponds to the time when the current delta files were created. 
    This means that the delta files contain the changes that happened to the table 
    between the from and to times.

    We currently store a history of up to seven delta files when available. 
    The latest_full_data file will always be present and can be used at any time to sync 
    your database if you have missed several delta updates.
    '''

    #print("Retrieving end of day market data")
    data = [[1,2,3,"A",5,6,7,8,9,10,11,12,13,14],[14,13,12,11,10,9,8,7,6,5,"Z",3,2,1]]
    df_data = pd.DataFrame(data)

    '''
    df_data = quandl.get_table("WIKI/PRICES",ticker=ticker, \
                               paginate=True, \
                               qopts={'columns':{field_of_interest}})
    '''
    #print ("New data\n%s\n%s\nhas a shape %s" % (df_data.head(2), df_data.tail(2),df_data.shape))

    return(df_data)

def fetch_timeseries_data(field_of_interest,symbol,source):  
    if source == "remote":
        quandl_data = get_ini_data("QUANDL")
        quandl.ApiConfig.api_key = quandl_data['key']
        df_data = quandl.get_table("WIKI/PRICES",ticker=symbol,qopts={'columns':{field_of_interest}})
    else:
        '''
        From locally stored file
        '''
        df_data = read_symbol_historical_data(symbol)
    data = df_data[field_of_interest].values
    #print ("Time series for %s, field(s) of interest: %s, time series has %s data points:\n%s ... %s" % 
    #       (symbol, field_of_interest, len(data), data[:3], data[-3:]))
    print ("Time series for %s, field(s) of interest: %s, time series has %s data points" % 
           (symbol, field_of_interest, len(data)))
    
    return (data)

def get_list_of_tickers():
    print ("Getting the list of tickers")
    import_file = get_devdata_dir() + "\\TickerMetadata.csv"
    df_tickers = pd.read_csv(import_file)
    df_tickers = df_tickers.set_index('ticker')
    #print (df_tickers)
    
    return(df_tickers)

def save_list_of_tickers(df_tickers):
    output_file = get_devdata_dir() + "\\TickerMetadata.csv"
    #print ("Creating a file containing the list of the tickers processed ...", output_file, df_tickers)
    df_tickers.to_csv(output_file)
    
    return

def quandl_data_last_updated():
    input_file = get_devdata_dir() + "\\Quandl_update.csv"
    f = open(input_file, 'r')
    date = f.read()
    f.closed
    #print("Quandl data last updated %s" % date)

    return(date)

def save_last_quandl_update():
    # Save in a file the date of the last download from Quandl
    date = datetime.datetime.today() # date and time
    #print("Saving %s as the date the data from Quandl was last added to the local data set" % date)
    output_file = get_devdata_dir() + "\\Quandl_update.csv"
    f = open(output_file, 'w')
    f.write(date.date().isoformat())
    f.closed

    return

def save_enhanced_historical_data(df_data):
    output_file = get_devdata_dir() + "\\Enhanced historical data.csv"
    print ("Creating enhanced historical data...", output_file)
    print ("Save_enhanced_historical_data() df_quandl head:\n", df_data.head(3))
    print ("Save_enhanced_historical_data() df_quandl tail:\n", df_data.tail(3))
    df_data.to_csv(output_file)
    
    return

def save_symbol_historical_data(ticker, df_data):
    output_file = get_devdata_dir() + "\\symbols\\" + ticker + ".csv"
    print ("Saving %s to file %s" % (ticker, output_file))
    #print ("Save_Symbol_Historical_Data() data head:\n", df_data.head(3))
    #print ("Save_Symbol_Historical_Data() data tail:\n", df_data.tail(3))
    df_data.to_csv(output_file)
    
    return

def read_symbol_historical_data(ticker):
    input_file = get_devdata_dir() + "\\symbols\\" + ticker + ".csv"
    #print ("Reading %s file %s" % (ticker, input_file))
    df_data = pd.read_csv(input_file)
    df_data = df_data.set_index('date')
    #print ("Read_Symbol_Historical_Data(): \n%s\n%s " % (df_data.head(1), df_data.tail(1)))
    
    return (df_data)

def read_historical_data(recs=none):  
    '''
    Open and process source data file
        Data fields in Quandl file are
            'volume', 'ex-dividend', 'adj_low', 'ticker', 'adj_open', 'adj_close', 
            'high', 'adj_volume', 'low', 'date', 'close', 'open', 'adj_high', 'split_ratio'
    '''
    import_file = get_devdata_dir() + "\\Quandl stock data.csv"
    #print ("Processing historical data...", import_file)
    if recs == none:
        df_data = pd.read_csv(import_file)
    else:
        df_data = pd.read_csv(import_file, nrows=recs)
    #print ("Historical data starts: \n\t", df_data.head(5))
    #print ("Historical data row 2 - ticker is:", df_data.loc[2,"ticker"])
    #print ("Historical data tail: \n\t", df_data.tail(2))
    #print ("historical data processed...", df_data.shape)

    return (df_data)

