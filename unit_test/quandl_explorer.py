'''
Created on Jan 31, 2018

@author: Brian

dict_quandl = ['volume', 'ex-dividend', 'adj_low', 'ticker', 'adj_open', 'adj_close', \
    'high', 'adj_volume', 'low', 'date', 'close', 'open', 'adj_high', 'split_ratio' ]
#print "dict_quandl: ", dict_quandl

dict_quandl += ['MACD_future_chg', 'EMA12', 'EMA26', 'MACD_Buy', 'MACD_Sell', 'momentum']
#print "dict_quandl: ", dict_quandl

'''
def quandl_explorer(df_quandl=None, df_dict=None):
    

    print ("\nQuandl_explorer entry..............")
    print ("\ndata frame input:\n", df_quandl)
    print ("\ndictionary input:\n", df_dict)
    
    ### Suck it and see ... not yet understood
    print ("\nExperimentation...")
    print ("\n.at", df_quandl.at("2"))
    print ("\n.iat(1)", df_quandl.iat(1))
    print ("\n.get_value")

    ### All rows, select columns
    print ("\nColumns of data...")
    print ("\ndata by column ['label'='EMA26'] =\n", df_quandl['EMA26']) #returns all rows with EMA26 values
    print ("\ndata by column ['label'='EMA26'] > 29\n", df_quandl['EMA26']>29) #return all rows with True/False values
    df_quandl.ix[1,'MACD_Buy'] = True
    df_quandl.ix[3,'MACD_Buy'] = True
    print ("\nbuy indicator points MACD_Buy==True:\n", df_quandl['MACD_Buy'] == True)
    
    ### data frame information
    print ("\ndata frame analysis...")
    
    ### Select rows
    print ("\nrows of data...")
    print ("\n.loc loc[2]\n", df_quandl.loc[2]) #returns a single row
    print ("\n.loc loc[2:3]\n", df_quandl.loc[2:3]) #returns multiple rows
    print ("\ndata rows [1:3] =\n", df_quandl[1:3]) #returns all rows with EMA26 values
    
    ### Specific cell
    print ("\nSpecific cells...")
    print ("\n.iloc[2,3] =\n", df_quandl.iloc[2,3])
    print ("\n.ix integer row and column label = 2,EMA12 =\n", df_quandl.ix[2,"EMA12"])
    
    print ("\nQuandl_explorer exit................")
    
    return