'''
Created on Jun 25, 2019

@author: Brian

deprecated: based on Quandl data
'''
import logging
import pickle
import time

from matplotlib._constrained_layout import do_constrained_layout
from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY, \
    num2date
from matplotlib.dates import date2num

from configuration_constants import ANALASIS_SAMPLE_LENGTH
from configuration_constants import FEATURE_TYPE
from configuration_constants import FORECAST_FEATURE
from configuration_constants import FORECAST_LENGTH
from configuration_constants import LOGGING_FORMAT
from configuration_constants import LOGGING_LEVEL
from configuration_constants import RESULT_DRIVERS
from configuration_constants import TICKERS
import datetime as dt
from lstm import normalize_data
from lstm import pickle_load_training_data
from lstm import prepare_3D_cube
import matplotlib.dates as md 
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ochl
from mpl_finance import candlestick_ohlc
import numpy as np
import pandas as pd
from quandl_library import fetch_timeseries_data
from quandl_library import get_devdata_dir
from quandl_library import get_ini_data
from time_series_data import series_to_supervised

RED = [1.0, 0.0, 0.0]
GREEN = [0.0, 1.0, 0.0]
BLUE = [0.0, 0.0, 1.0]


#===================================================================================================
#    **************************** Copied from stackoverflow and unused **************************************
def pandas_candlestick_ohlc(dat, stick="day", adj=False, otherseries=None):
    '''
    :param dat: pandas DataFrame object with datetime64 index, and float columns 
    "Open", "High", "Low", and "Close", likely created via DataReader from "yahoo"
    :param stick: A string or number indicating the period of time covered by a single candlestick. 
        Valid string inputs include "day", "week", "month", and "year", ("day" default), and any numeric 
        input indicates the number of trading days included in a period
    :param adj: A boolean indicating whether to use adjusted prices
    :param otherseries: An iterable that will be coerced into a list, containing the columns of dat that 
        hold other series to be plotted as lines

    This will show a Japanese candlestick plot for stock data stored in dat, also plotting other series if passed.
    '''
    mondays = WeekdayLocator(MONDAY)  # major ticks on the mondays
    alldays = DayLocator()  # minor ticks on the days
    dayFormatter = DateFormatter('%d')  # e.g., 12

    # Create a new DataFrame which includes OHLC data for each period specified by stick input
    fields = ["Open", "High", "Low", "Close"]
    if adj:
        fields = ["Adj. " + s for s in fields]
    transdat = dat.loc[:, fields]
    transdat.columns = pd.Index(["Open", "High", "Low", "Close"])
    if (type(stick) == str):
        if stick == "day":
            plotdat = transdat
            stick = 1  # Used for plotting
        elif stick in ["week", "month", "year"]:
            if stick == "week":
                transdat["week"] = pd.to_datetime(transdat.index).map(lambda x: x.isocalendar()[1])  # Identify weeks
        elif stick == "month":
            transdat["month"] = pd.to_datetime(transdat.index).map(lambda x: x.month)  # Identify months
        transdat["year"] = pd.to_datetime(transdat.index).map(lambda x: x.isocalendar()[0])  # Identify years
        grouped = transdat.groupby(list(set(["year", stick])))  # Group by year and other appropriate variable
        plotdat = pd.DataFrame({"Open": [], "High": [], "Low": [], "Close": []})  # Create empty data frame containing what will be plotted
        for name, group in grouped:
            plotdat = plotdat.append(pd.DataFrame({"Open": group.iloc[0, 0],
                                                   "High": max(group.High),
                                                   "Low": min(group.Low),
                                                   "Close": group.iloc[-1, 3]},
                                                   index=[group.index[0]]))
        if stick == "week": stick = 5
        elif stick == "month": stick = 30
        elif stick == "year": stick = 365

    elif (type(stick) == int and stick >= 1):
        transdat["stick"] = [np.floor(i / stick) for i in range(len(transdat.index))]
        
    grouped = transdat.groupby("stick")
    plotdat = pd.DataFrame({"Open": [], "High": [], "Low": [], "Close": []}) 
    # Create empty data frame containing what will be plotted
    for name, group in grouped:
        plotdat = plotdat.append(pd.DataFrame({"Open": group.iloc[0, 0],
                                               "High": max(group.High),
                                               "Low": min(group.Low),
                                               "Close": group.iloc[-1, 3]},
                                               index=[group.index[0]]))

    else:
        raise ValueError('Valid inputs to argument "stick" include the strings "day", "week", "month", "year", or a positive integer')

    # Set plot parameters, including the axis object ax used for plotting
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.2)
    if plotdat.index[-1] - plotdat.index[0] < pd.Timedelta('730 days'):
        weekFormatter = DateFormatter('%b %d')  # e.g., Jan 12
        ax.xaxis.set_major_locator(mondays)
        ax.xaxis.set_minor_locator(alldays)
    else:
        weekFormatter = DateFormatter('%b %d, %Y')
        ax.xaxis.set_major_formatter(weekFormatter)

    ax.grid(True)

    # Create the candelstick chart
    candlestick_ohlc(ax,
                     list(zip(list(date2num(plotdat.index.tolist())),
                     plotdat["Open"].tolist(),
                     plotdat["High"].tolist(),
                     plotdat["Low"].tolist(),
                     plotdat["Close"].tolist())),
                     colorup="green", colordown="red", width=stick * .4)

    # Plot other series (such as moving averages) as lines
    if otherseries != None:
        if type(otherseries) != list:
            otherseries = [otherseries]
    dat.loc[:, otherseries].plot(ax=ax, lw=1.3, grid=True)

    ax.xaxis_date()
    ax.autoscale_view()
    plt.setp(plt.gca().get_xticklabels(), rotation=45,
    horizontalalignment='right')

    plt.show()


    # pandas_candlestick_ohlc(apple, adj=True, stick="month")
#===================================================================================================
def verify_pickling_process (data2dump):
    
    ml_config = get_ini_data('DEVDATA')
    training_dir = ml_config['dir']
    print('Writing data files to %s', training_dir)
    
    dump_file = training_dir + "\\process_verify.pickle"    
    dump_out = open(dump_file, "wb")
    pickle.dump(data2dump, dump_out)
    
    ml_config = get_ini_data('DEVDATA')
    training_dir = ml_config['dir']
    logging.info('Reading data files from %s', training_dir)
    
    data_file = training_dir + "\\process_verify.pickle"    
    data_in = open(data_file, "rb")
    dumped_data = pickle.load(data_in)

    return dumped_data


def market_activity(axis, df_data, ndx_start):
    # list_x_train.append(x_train[:, :, :5])    
    data = []
    day = 1.0
    for ndx_ts in range (ndx_start, ndx_start + ANALASIS_SAMPLE_LENGTH + FORECAST_LENGTH):
        data.append([day, \
                     df_data[ndx_ts, 2], \
                     df_data[ndx_ts, 3], \
                     df_data[ndx_ts, 1], \
                     df_data[ndx_ts, 0] \
                    ])
        day += 1
    # candlestick plot
    bplot = candlestick_ochl(axis, data, colorup="green", colordown="red")
    axis.set_title('market_activity from csv')
    axis.set_xlabel('Time Series')
    axis.set_ylabel('ochl')
    axis.yaxis.grid(True)
    return


def market_volume(axis, df_data, ndx_start):
    vol = []
    x = []
    day = 1.0
    for ndx_ts in range (0, ANALASIS_SAMPLE_LENGTH + FORECAST_LENGTH):
        x.append(day)
        vol.append(df_data[ndx_start + ndx_ts, 4])
        day += 1

    axis.bar(x, vol)
    axis.set_title('market_volume')
    axis.set_xlabel('Time Series')
    axis.set_ylabel('shares traded')
    axis.yaxis.grid(True)
    return


def market_activity_flattened(axis, df_data, ndx_start):
    data = []
    day = 1.0
    features = len(RESULT_DRIVERS)
    np_data = df_data.to_numpy(copy=True)
    for ndx_ts in range (0, ANALASIS_SAMPLE_LENGTH + FORECAST_LENGTH):
        data.append([day, \
                     np_data[ndx_start, (ndx_ts * features) + 2], \
                     np_data[ndx_start, (ndx_ts * features) + 3], \
                     np_data[ndx_start, (ndx_ts * features) + 1], \
                     np_data[ndx_start, (ndx_ts * features) + 0] \
                    ])
        day += 1
    # candlestick plot
    candlestick_ochl(axis, data, colorup="green", colordown="red")
    axis.set_title('market_activity flattened')
    axis.set_xlabel('Time Series')
    axis.set_ylabel('ochl')
    axis.yaxis.grid(True)
    return


def market_volume_flattened(axis, df_data, ndx_start):
    vol = []
    x = []
    day = 1.0
    features = len(RESULT_DRIVERS)
    np_data = df_data.to_numpy(copy=True)
    for ndx_ts in range (0, ANALASIS_SAMPLE_LENGTH + FORECAST_LENGTH):
        x.append(day)
        vol.append(np_data[ndx_start, (ndx_ts * features) + 4])
        day += 1

    axis.bar(x, vol)
    axis.set_title('market_volume flattened')
    axis.set_xlabel('Time Series')
    axis.set_ylabel('shares traded')
    axis.yaxis.grid(True)
    return


def market_activity_numpy(axis, np_data, ndx_start):
    data = []
    day = 1.0
    for ndx_ts in range (0, ANALASIS_SAMPLE_LENGTH + FORECAST_LENGTH):
        data.append([day, \
                     np_data[ndx_start, ndx_ts, 2], \
                     np_data[ndx_start, ndx_ts, 3], \
                     np_data[ndx_start, ndx_ts, 1], \
                     np_data[ndx_start, ndx_ts, 0] \
                    ])
        day += 1

    # candlestick plot
    candlestick_ochl(axis, data, colorup="green", colordown="red")
    axis.set_title('market_activity 3D Numpy')
    axis.set_xlabel('Time Series')
    axis.set_ylabel('ochl')
    axis.yaxis.grid(True)
    return


def market_volume_numpy(axis, np_data, ndx_start):
    vol = []
    x = []
    day = 1.0
    for ndx_ts in range (0, ANALASIS_SAMPLE_LENGTH + FORECAST_LENGTH):
        x.append(day)
        vol.append(np_data[ndx_start, ndx_ts, 4])
        day += 1

    axis.bar(x, vol)
    axis.set_title('market_volume 3D numpy')
    axis.set_xlabel('Time Series')
    axis.set_ylabel('shares traded')
    axis.yaxis.grid(True)
    return


def market_activity_normalized(axis, np_data, ndx_start):
    data = []
    day = 1.0
    for ndx_ts in range (0, ANALASIS_SAMPLE_LENGTH + FORECAST_LENGTH):
        data.append([day, \
                     np_data[ndx_start, ndx_ts, 2], \
                     np_data[ndx_start, ndx_ts, 3], \
                     np_data[ndx_start, ndx_ts, 1], \
                     np_data[ndx_start, ndx_ts, 0] \
                    ])
        day += 1

    # candlestick plot
    candlestick_ochl(axis, data, colorup="green", colordown="red")
    axis.set_title('market_activity normalized')
    axis.set_xlabel('Time Series')
    axis.set_ylabel('ochl')
    axis.yaxis.grid(True)
    return


def market_volume_normalized(axis, np_data, ndx_start):
    vol = []
    x = []
    day = 1.0
    for ndx_ts in range (0, ANALASIS_SAMPLE_LENGTH + FORECAST_LENGTH):
        x.append(day)
        vol.append(np_data[ndx_start, ndx_ts, 4])
        day += 1

    axis.bar(x, vol)
    axis.set_title('market volume normalized')
    axis.set_xlabel('Time Series')
    axis.set_ylabel('shares traded')
    axis.yaxis.grid(True)
    return


def market_activity_unpickled(axis, np_data, ndx_start):
    data = []
    day = 1.0
    for ndx_ts in range (0, ANALASIS_SAMPLE_LENGTH + FORECAST_LENGTH):
        data.append([day, \
                     np_data[ndx_start, ndx_ts, 2], \
                     np_data[ndx_start, ndx_ts, 3], \
                     np_data[ndx_start, ndx_ts, 1], \
                     np_data[ndx_start, ndx_ts, 0] \
                    ])
        day += 1

    candlestick_ochl(axis, data, colorup="green", colordown="red")
    axis.set_title('market_activity unpickled')
    axis.set_xlabel('Time Series')
    axis.set_ylabel('ochl')
    axis.yaxis.grid(True)
    return


def market_volume_unpickled(axis, np_data, ndx_start):
    vol = []
    x = []
    day = 1.0
    for ndx_ts in range (0, ANALASIS_SAMPLE_LENGTH + FORECAST_LENGTH):
        x.append(day)
        vol.append(np_data[ndx_start, ndx_ts, 4])
        day += 1

    axis.bar(x, vol)
    axis.set_title('market volume unpickled')
    axis.set_xlabel('Time Series')
    axis.set_ylabel('shares traded')
    axis.yaxis.grid(True)
    return


def bollinger_bands(axis, x_train):
    # list_x_train.append(x_train[:, :, 5:7])
    dummy_data = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
    labels = ['x1', 'x2', 'x3', 'x1', 'x2', 'x3']
    colors = ['pink', 'lightblue', 'lightgreen', RED, GREEN, BLUE]

    bplot = axis.boxplot(dummy_data,
                        notch=True,  # notch shape
                        vert=True,  # vertical box alignment
                        patch_artist=True,  # fill with color
                        labels=labels)  # will be used to label x-ticks
    axis.set_title('bollinger_bands')

    axis.yaxis.grid(True)
    axis.set_xlabel('Time Series')
    axis.set_ylabel('Upper & Lower')

    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    return


def accumulation_distribution(axis, x_train):
    # list_x_train.append(x_train[:, :, 9:10])
    dummy_data = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
    labels = ['x1', 'x2', 'x3']
    bplot = axis.boxplot(dummy_data,
                             notch=True,  # notch shape
                             vert=True,  # vertical box alignment
                             patch_artist=True,  # fill with color
                             labels=labels)  # will be used to label x-ticks

    axis.yaxis.grid(True)
    axis.set_xlabel('Time Series')
    axis.set_ylabel('Feature values')

    axis.set_title('accumulation_distribution')
    return bplot

    
def MACD_Sell(axis, x_train):
    # list_x_train.append(x_train[:, :, 10:12])
    dummy_data = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
    labels = ['x1', 'x2', 'x3']
    bplot = axis.boxplot(dummy_data,
                             notch=True,  # notch shape
                             vert=True,  # vertical box alignment
                             patch_artist=True,  # fill with color
                             labels=labels)  # will be used to label x-ticks

    axis.yaxis.grid(True)
    axis.set_xlabel('Time Series')
    axis.set_ylabel('Feature values')

    axis.set_title('MACD_Sell')
    return bplot

    
def MACD_Buy(axis, x_train):
    # list_x_train.append(x_train[:, :, 12:13])
    dummy_data = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
    labels = ['x1', 'x2', 'x3']
    bplot = axis.boxplot(dummy_data,
                             notch=True,  # notch shape
                             vert=True,  # vertical box alignment
                             patch_artist=True,  # fill with color
                             labels=labels)  # will be used to label x-ticks

    axis.yaxis.grid(True)
    axis.set_xlabel('Time Series')
    axis.set_ylabel('Feature values')

    axis.set_title('MACD_Buy')
    return bplot


if __name__ == '__main__':
    print ("Good morning Dr. chandra. I am ready to verify that you have prepared my training data correctly.\n")
    print ("Using %s as example" % TICKERS)
    
    feature_count = len(RESULT_DRIVERS)
    forecast_feature = FORECAST_FEATURE
    time_steps = ANALASIS_SAMPLE_LENGTH
    forecast_steps = FORECAST_LENGTH
    print ("Result Driver: %s" % RESULT_DRIVERS)
    
    start = time.time()
    now = dt.datetime.now()
    
    lstm_config_data = get_ini_data("LSTM")
    log_file = lstm_config_data['log']

    logging.basicConfig(filename=log_file, level=LOGGING_LEVEL, format=LOGGING_FORMAT)
    print ("Logging to", log_file)
    logger = logging.getLogger('lstm_logger')
    log_fmt = logging.Formatter('%(asctime)s - %(name)s - %levelname - %(messages)s')
    logger.info('Plot data at all stages of preparation')
    
    ts_length = ANALASIS_SAMPLE_LENGTH + FORECAST_LENGTH
    
    # Load raw data from file and plot 1st and last sample
    df_data = fetch_timeseries_data(RESULT_DRIVERS, TICKERS[0], source='')
    samples = df_data.shape[0]
    print ("Samples: %s" % df_data.shape[0])  
      
    # Convert raw data to time series dataframe
    time_steps = ANALASIS_SAMPLE_LENGTH
    forecast_steps = FORECAST_LENGTH
    df_data_supervised = series_to_supervised(df_data, time_steps, forecast_steps)
    
    # Prepare 3D Numpy data cube from 2D dataframe    
    np_data_3D, np_prediction = prepare_3D_cube(df_data_supervised, feature_count, forecast_feature, time_steps, forecast_steps)

    fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(9, 4), constrained_layout=True)

    # Normalize the data as required for machine learning
    np_data_normalized = np.copy(np_data_3D[:,:,:5])
    feature_groups = [[0, 1, 2, 3], [4]]
    feature_count = 5
    np_data_normalized = normalize_data(np_data_normalized, len(np_data_normalized), feature_count, \
                                        FEATURE_TYPE[:5], RESULT_DRIVERS[:5], feature_groups, \
                                        time_steps, forecast_steps)
    # Pickle and unpickle the data to confirm those steps do not change the data    
    print ("Pickle and unpickle the data to confirm those steps do not change the data")
    np_unpickled_data = verify_pickling_process (np_data_normalized)
    
    market_activity(axes[0, 0], df_data, 0)
    market_activity(axes[1, 0], df_data, (samples - 1) - ts_length)    
    market_volume(axes[2, 0], df_data, 0)
    market_volume(axes[3, 0], df_data, (samples - 1) - ts_length)    
    
    market_activity_flattened(axes[0, 1], df_data_supervised, 0)
    market_activity_flattened(axes[1, 1], df_data_supervised, (samples - 1) - ts_length)    
    market_volume_flattened(axes[2, 1], df_data_supervised, 0)
    market_volume_flattened(axes[3, 1], df_data_supervised, (samples - 1) - ts_length)    

    market_activity_numpy(axes[0, 2], np_data_3D, 0)
    market_activity_numpy(axes[1, 2], np_data_3D, (samples - 1) - ts_length)
    market_volume_numpy(axes[2, 2], np_data_3D, 0)
    market_volume_numpy(axes[3, 2], np_data_3D, (samples - 1) - ts_length)
    
    market_activity_normalized(axes[0, 3], np_data_normalized, 0)
    market_activity_normalized(axes[1, 3], np_data_normalized, (samples - 1) - ts_length)
    market_volume_normalized(axes[2, 3], np_data_normalized, 0)
    market_volume_normalized(axes[3, 3], np_data_normalized, (samples - 1) - ts_length)

    market_activity_unpickled(axes[0, 4], np_unpickled_data, 0)
    market_activity_unpickled(axes[1, 4], np_unpickled_data, (samples - 1) - ts_length)
    market_volume_unpickled(axes[2, 4], np_unpickled_data, 0)
    market_volume_unpickled(axes[3, 4], np_unpickled_data, (samples - 1) - ts_length)
    
    plt.show()
    
    print ("I hope the data looks correct.\n")
