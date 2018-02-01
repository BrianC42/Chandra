'''
Created on Jan 31, 2018

@author: Brian
'''


def mp_prep_quandl_data_files():
    from quandl_library import read_historical_data
    from quandl_library import save_symbol_historical_data
    from quandl_library import save_list_of_tickers
    from quandl_library import ticker_status_dict
    
    print ("Affirmative, Dave. I read you\n")
    '''
    Reading historical data and splitting symbol data into separate files
    '''
    df_data = read_historical_data() #recs=10000)
    df_ticker_list = df_data.drop_duplicates("ticker")
    #print ("Tickers:", df_ticker_list)
    
    df_ticker_metadata = df_ticker_list.set_index('ticker')
    del df_ticker_metadata['open']
    del df_ticker_metadata['high']
    del df_ticker_metadata['low']
    del df_ticker_metadata['close']
    del df_ticker_metadata['date']
    del df_ticker_metadata['volume']
    del df_ticker_metadata['ex-dividend']
    del df_ticker_metadata['split_ratio']
    del df_ticker_metadata['adj_open']
    del df_ticker_metadata['adj_high']
    del df_ticker_metadata['adj_low']
    del df_ticker_metadata['adj_close']
    del df_ticker_metadata['adj_volume']
    
    #Set default values
    df_ticker_metadata['First date'] = ""
    df_ticker_metadata['Last date'] = ""
    df_ticker_metadata['Status'] = 0

    idx = 0
    indices = df_ticker_list.index.get_values()
    #print ("Indices", indices)
    
    dictTickerStatus = ticker_status_dict()

    print ("Read {1} Rows of data and discovered {0} tickers to process:\n".format(df_ticker_list.shape[0], len(df_data)), indices)

    last_slice = False
    while idx < df_ticker_list.shape[0]:   
        '''
        ================= Process slices ================
        '''
        while last_slice == False:
            if idx < df_ticker_list.shape[0] - 1:
                slice_start = indices[idx]
                slice_end = indices[idx+1]
                #print ("ticker", df_ticker_list.ix[slice_start, 'ticker'], "Not last ticker. Slice:", slice_start, "to", slice_end)
            else:
                slice_start = indices[len(indices)-1]
                slice_end = len(df_data) - 1
                last_slice = True
                #print ("ticker", df_ticker_list.ix[slice_start, 'ticker'], "Last ticker. Slice", slice_start, "to", slice_end)

            #print ("Processing slice", idx, slice_start, slice_end)
            ticker = df_ticker_list.ix[slice_start, 'ticker']
        
            df_ticker_data = df_data[slice_start:slice_end]
            df_ticker_data = df_ticker_data.set_index('date') # index data by date string
            save_symbol_historical_data(ticker, df_ticker_data)

            '''
            Update metadata captured for each ticker
            '''
            num_rows = df_ticker_data.shape[0]
            first_date = df_ticker_data.iloc[0].name 
            last_date = df_ticker_data.iloc[num_rows - 1].name
            df_ticker_metadata.loc[ticker, 'First date'] = first_date
            df_ticker_metadata.loc[ticker, 'Last date'] = last_date
            df_ticker_metadata.loc[ticker, 'Status'] = dictTickerStatus['unknown']
            
            idx += 1
            
        #print ("List of symbols\n", df_ticker_metadata)
        save_list_of_tickers(df_ticker_metadata)

    print ("\nDave, this conversation can serve no purpose anymore. Goodbye")
 
if __name__ == '__main__':
    mp_prep_quandl_data_files()