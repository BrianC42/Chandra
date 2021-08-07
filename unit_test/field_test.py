'''
Created on Jan 31, 2018

@author: Brian
'''
from quandl_library import read_historical_data
from bollinger_bands import bollinger_bands

if __name__ == '__main__':
    print ("\nBeginning technical analysis ...")
    df_data = read_historical_data(recs=10000)
    df_tickers = df_data.drop_duplicates("ticker") 
    indices = df_tickers.index.get_values()
    print ("Discovered {0} tickers:".format(df_tickers.shape[0]))
    print ("{0} indices with {1} Rows of data to analyze:\n".format(len(indices), len(df_data)), indices )
    
    loop = False
    if loop:
        '''
        Loop to test multiple calls
        '''
        idx = 0
        last_slice = False
        while idx < len(indices) and not last_slice:   
            slice_start = indices[idx]
            
            if idx < len(indices)-1:
                slice_start = indices[idx]
                slice_end = indices[idx+1]
            else:
                slice_start = indices[len(indices)-1]
                slice_end = len(df_data) - 1
                last_slice = True
            
            print ("\nticker", df_tickers.ix[slice_start, 'ticker'], slice_start, "to", slice_end, "\n")
            df_data = df_data[slice_start:slice_end] = bollinger_bands(df_data[slice_start:slice_end], \
                                                            value_label='adj_close', \
                                                            sma_interval=20, sma_label='SMA20')
            # enh_data = pd.concat([output_data, data_enh])

            idx += 1
    else:
        '''
        Code to test a single call
        '''
        idx = 3
        slice_start = indices[idx]
        slice_end = indices[idx+1]

        print ("\nticker", df_tickers.ix[slice_start, 'ticker'], slice_start, "to", slice_end, "\n")
        df_enh = bollinger_bands(df_data[slice_start:slice_end], \
                                 value_label='adj_close', \
                                 sma_interval=20, sma_label='SMA20')

        print ("\ndf_enh head\n", df_enh.head(3), \
               "\nand tail ...\n", df_enh.tail(3))
