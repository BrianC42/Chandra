'''
Created on Jan 31, 2018

@author: Brian
'''

def classification_and_regression(df_data_src=None):
    
    ### draw the scatterplot, with color-coded training and testing points
    import matplotlib.pyplot as plt

    print ("classification_and_regression begin ...")
    
    #print ("Enhanced data file head and tail\n", df_data_src.head(5))
    #print (df_data_src.tail(5))

    print ("\nBeginning regression analysis...")

    print ("\nMACD analysis")
    df_MACD_Buy = df_data_src.query('MACD_Buy == True')
    Num_Buy_Signals = len(df_MACD_Buy)
    print ("MACD buy signals found:", Num_Buy_Signals)
    #print ("MACD buy signals:\n", df_MACD_Buy.head(40))
    
    df_MACD_Buy_Train = df_MACD_Buy[:int(Num_Buy_Signals*0.9)]   #1st 90% of samples
    df_MACD_Buy_Test = df_MACD_Buy[int(Num_Buy_Signals*0.9):]    #last 10% of samples
    print ("Training data point:", len(df_MACD_Buy_Train), "blue")
    print ("Testing data points:", len(df_MACD_Buy_Test), "red")
    
    train_color = "b"
    test_color = "r"

    print ("Plot training data", len(df_MACD_Buy_Train))
    for feature, target in zip(df_MACD_Buy_Train["momentum"], df_MACD_Buy_Train["MACD_future_chg"]):
        plt.scatter( feature, target, color=train_color ) 

    print ("Plot test data", len(df_MACD_Buy_Test))
    for feature, target in zip(df_MACD_Buy_Test["momentum"], df_MACD_Buy_Test["MACD_future_chg"]):
        plt.scatter( feature, target, color=test_color )
         
    ### labels for the legend
    #plt.scatter(feature_test[0], target_test[0], color=test_color, label="test")
    #plt.scatter(feature_test[0], target_test[0], color=train_color, label="train")

    plt.xlabel("momentum")
    plt.ylabel("30 day change")
    plt.legend()
    plt.show()

    
    print ("MACD buy signal analysis complete\n")

    df_MACD_Sell = df_data_src.query('MACD_Sell == True')
    Num_Sell_Signals = len(df_MACD_Sell)
    print ("MACD sell signals found:", Num_Sell_Signals)
    df_MACD_Sell_Train = df_MACD_Sell.loc[:(Num_Sell_Signals*0.9)]
    df_MACD_Sell_Test = df_MACD_Sell.loc[(Num_Sell_Signals*0.9):]
    print ("MACD sell signal analysis complete\n")

    print ("classification_and_regression end ...")

    return (df_data_src)
