'''
Created on Jan 31, 2018

@author: Brian
'''
import datetime

def update_stock_data():
    from quandl_library import get_list_of_tickers
    from quandl_library import save_list_of_tickers
    from quandl_library import update_eod_data
    from quandl_library import read_symbol_historical_data
    from quandl_library import save_last_quandl_update
    from quandl_library import quandl_data_last_updated
    
    df_new_data = update_eod_data() #get delta from Quandl
    print ("New data\n%s\n has a shape %s" % (df_new_data, df_new_data.shape))
    df_tickers = get_list_of_tickers()
    print("df_tickers\n%s ..." % df_tickers.head(3))
    dt_Q_date = datetime.datetime.strptime(quandl_data_last_updated(), '%Y-%m-%d')
    print("Quandl date was last updated %s" % dt_Q_date)
    dt_today = datetime.datetime.today()
    print("Today is %s" % dt_today)
    
    
    for ticker, row in df_tickers.iterrows():
        #print ("Ticker %s" % index)
        df_ticker_data = read_symbol_historical_data(ticker)
        #print ("Update_Stock_Data(): \n%s\n%s " % (df_ticker_data.head(1), df_ticker_data.tail(1)))
        first_date = df_ticker_data.iloc[0].name # data is indexed by date string
        row['First date'] = first_date

        num_rows = df_ticker_data.shape[0]
        last_date = df_ticker_data.iloc[num_rows - 1].name
        row['Last date'] = first_date
        
        dt_1st_date = datetime.datetime.strptime(first_date, '%Y-%m-%d')
        dt_last_date = datetime.datetime.strptime(last_date, '%Y-%m-%d')
        
        if dt_Q_date == dt_last_date:
            #symbol was actively traded when data was last updated
            #print("Ticker %s, oldest data %s, most recent data %s, Status %s" % (ticker, first_date, last_date, row['Status']))
            print ("Ticker %s, 1st %s, last %s, delta %s" % \
                   (ticker, dt_1st_date, dt_last_date, dt_Q_date - dt_last_date))
            row['Status'] = "active"
        else:
            #symbol was inactive when data last updated
            print("Ticker %s is no longer active" % ticker)
            row['Status'] = "inactive"
        
        '''
        print("Ticker meta-data:\n%s" % row)
        df_tickers[ticker] = row
        ndx += 1
        if ndx >= 10:
            break
        '''
                
    save_list_of_tickers(df_tickers)
    save_last_quandl_update() #Update timestamp of last update from Quandl

    return

if __name__ == '__main__':
    print ("Affirmative, Dave. I read you\n")
    update_stock_data()
    print ("\nDave, this conversation can serve no purpose anymore. Goodbye")
