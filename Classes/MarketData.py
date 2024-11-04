'''
Created on Jul 23, 2024

@author: brian
'''
import os
import multiprocessing
import sys
import json
import datetime as dt
from datetime import date
import datetime
import time
import numpy as np
from numpy import NaN
import pandas as pd

from configuration import get_ini_data

from financialDataServices import financialDataServices

from moving_average import simple_moving_average
from moving_average import exponential_moving_average
from tda_api_library import format_tda_datetime

from macd import macd
from on_balance_volume import on_balance_volume
from bollinger_bands import bollinger_bands
from accumulation_distribution import accumulation_distribution
from aroon_indicator import aroon_indicator
from average_directional_index import average_directional_index
from stochastic_oscillator import stochastic_oscillator
from relative_strength import relative_strength

ARCHIVE_PERIOD_TYPE = "month"
ARCHIVE_FREQUENCY_TYPE = "daily"
ARCHIVE_FREQUENCY = "1"
MAX_HISTORY_PERIOD = "20"
MAX_HISTORY_TYPE = 'year'
BASIC_ARCHIVE_DIR = "basic"
ENHANCED_ARCHIVE_DIR = "enhanced"

''' base class for market data - start '''
class basicMarketData(object):
    '''     classdocs      '''
    '''     class data     '''

    '''         Constructor        '''
    def __init__(self, symbol):
        exc_txt = "An exception occurred creating a basicMarketData object for {}".format(symbol)
        try:
            #print("basicMarketData init {}".format(symbol))
            self.symbol = symbol
            self.financialDataServicesObj = financialDataServices()
            return

        except Exception:
            exc_info = sys.exc_info()
            exc_str = exc_info[1].args[0]
            exc_txt = exc_txt + "\n\t" + exc_str
            sys.exit(exc_txt)
    
    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= len(self.df_marketData):
            raise StopIteration

        else:
            candleDict = {'datetime' : self.df_marketData.iloc[self.index]['DateTime'], \
                          'open' : self.df_marketData.iloc[self.index]['Open'], \
                          'high' : self.df_marketData.iloc[self.index]['High'], \
                          'low' : self.df_marketData.iloc[self.index]['Low'], \
                          'close' : self.df_marketData.iloc[self.index]['Close'], \
                          'volume' : self.df_marketData.iloc[self.index]['Volume']}
            self.candle = candleDict
            self.index += 1
            return self.candle
   
    ''' ============ validate creation parameters are supported by data service - start ============= '''
    def validateRequestParams(self):
        exc_txt = "WIP ***************\n\tneed code to check MarketData constructor parameter values\n\tinvalid market data request parameter"
        try:
            if self.periodType == 'month':
                validPeriods = ["1","2","3","6"]
                if self.period in validPeriods:
                    if self.frequencyType == "daily":
                        if self.frequency == "1":
                            pass
                        else:
                            exc_txt = exc_txt + " - frequency"
                            raise ValueError                    
                    else:
                        exc_txt = exc_txt + " - frequency type"
                        raise ValueError                    
                else:
                    exc_txt = exc_txt + " - period"
                    raise ValueError
            elif self.periodType == 'day':
                validPeriods = ["1","2","3","4","5","10"]
                if self.period in validPeriods:
                    if self.frequencyType == "daily":
                        if self.frequency == "1":
                            pass
                        else:
                            exc_txt = exc_txt + " - frequency"
                            raise ValueError                    
                    else:
                        exc_txt = exc_txt + " - frequency type"
                        raise ValueError                    
                else:
                    exc_txt = exc_txt + " - period"
                    raise ValueError
            else:
                exc_txt = exc_txt + " - period type"
                raise ValueError
            return
            
        except ValueError:
            sys.exit(exc_txt)
    ''' ============ validate creation parameters are supported by data service - end ============= '''

    ''' ============  locate archive file - start ============= '''
    def locateArchive(self, archive):
        exc_txt = "\nAn exception occurred - unable to locate market data archive folder"
        try:
            localDirs = get_ini_data("LOCALDIRS")
            aiwork = localDirs['aiwork']
            
            if archive == "basic":
                archiveFolder = aiwork + localDirs['market_data'] + localDirs['basic_market_data']
            else:
                archiveFolder = aiwork + localDirs['market_data'] + localDirs['augmented_market_data']
            
            if not os.path.exists(archiveFolder):
                raise Exception("market data archive directory does not exist - {}".format(archiveFolder))

            return archiveFolder
            
        except Exception:
            exc_info = sys.exc_info()
            exc_str = exc_info[1].args[0]
            sys.exit(exc_txt + "\n\t" + exc_str)
    ''' ============  locate basic archives  - end ============= '''
            
            
    ''' ============ data service provider and market data controls - start ============= '''
    @property
    def symbol(self):
        return self._symbol
    
    @symbol.setter
    def symbol(self, symbol):
        self._symbol = symbol
        
    @property
    def periodType(self):
        return self._periodType
    
    @periodType.setter
    def periodType(self, periodType):
        self._periodType = periodType

    @property
    def period(self):
        return self._period
    
    @period.setter
    def period(self, period):
        self._period = period

    @property
    def frequencyType(self):
        return self._frequencyType
    
    @frequencyType.setter
    def frequencyType(self, frequencyType):
        self._frequencyType = frequencyType

    @property
    def frequency(self):
        return self._frequency
    
    @frequency.setter
    def frequency(self, frequency):
        self._frequency = frequency

    @property
    def financialDataServicesObj(self):
        return self._financialDataServicesObj

    @financialDataServicesObj.setter
    def financialDataServicesObj(self, financialDataServicesObj):
        self._financialDataServicesObj = financialDataServicesObj

    ''' ============ data service provider and market data controls - start ============= '''
    ''' ============ data service provider returned data - start ============= '''
    @property
    def candle(self):
        return self._candle
    
    @candle.setter
    def candle(self, candle):
        self._candle = MarketDataCandle(candle)

    @property
    def df_marketData(self):
        return self._df_marketData
    
    @df_marketData.setter
    def df_marketData(self, df_marketData):
        self._df_marketData = df_marketData

    @property
    def marketDataJson(self):
        return self._marketDataJson
    
    @marketDataJson.setter
    def marketDataJson(self, marketDataJson):
        self._marketDataJson = marketDataJson

    @property
    def marketDataReturn(self):
        return self._marketDataReturn
    
    @marketDataReturn.setter
    def marketDataReturn(self, marketDataReturn):
        self._marketDataReturn = marketDataReturn
    ''' ============ data service provider returned data - end ============= '''
''' base class for market data - end '''

''' class for a current market data - start '''
class MarketData(basicMarketData):
    '''     classdocs      '''
    '''     class data     '''

    '''         Constructor        '''
    def __init__(self, symbol, periodType=ARCHIVE_PERIOD_TYPE, period=MAX_HISTORY_PERIOD, \
                 frequencyType=ARCHIVE_FREQUENCY_TYPE, frequency=ARCHIVE_FREQUENCY):
        
        exc_txt = "An exception occurred creating a MarketData object for {}".format(symbol)
        try:
            super().__init__(symbol)
            #self.useArchive = False
            self.periodType = periodType
            self.period = period
            self.frequencyType = frequencyType
            self.frequency = frequency
            
            self.validateRequestParams()
            self.requestMarketData()

        except ValueError:
            sys.exit(exc_txt)
        
    ''' ============ request market data as indicated by creation params and do not include archived data - start ============= '''
    def requestMarketData(self):
        exc_txt = "\nAn exception occurred - unable to request market data"
        try:
            '''
            Full flexibility of period, type, frequency type and frequency
            '''
            response = self.financialDataServicesObj.requestMarketData(symbol=self.symbol, \
                                                                       periodType=self.periodType, period=self.period, 
                                                                       frequencyType=self.frequencyType, frequency=self.frequency)
            self.marketDataReturn = response.text
            self.marketDataJson = json.loads(self.marketDataReturn)
            df_new = pd.DataFrame(columns=['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume'])
            for ndx in range(len(self.marketDataJson["candles"])):
                self.candle = self.marketDataJson["candles"][ndx]
                
                # Create a new row to append
                new_row = {'DateTime' : self.candle.candleDateValue * 1000, \
                           'Open' : self.candle.candleOpen, \
                           'High' : self.candle.candleHigh, \
                           'Low' : self.candle.candleLow, \
                           'Close' : self.candle.candleClose, \
                           'Volume' : self.candle.candleVolume}
                
                df_new.loc[len(df_new)] = new_row
                self.df_marketData = df_new
            return self.df_marketData
            
        except ValueError:
            sys.exit(exc_txt)
    ''' ============ request market data as indicated by creation params and do not include archived data - end ============= '''

   
    ''' ============ return a specific single candle - start ============= '''
    def iloc(self, ndx):
        #print("Returning candle #{}".format(ndx))
        exc_txt = "\nAn exception occurred - unable to access requested candle"
        try:
            if ndx > len(self.df_marketData):
                raise ValueError
            else:
                candleDict = {'datetime' : self.df_marketData.iloc[ndx]['DateTime'], \
                              'open' : self.df_marketData.iloc[ndx]['Open'], \
                              'high' : self.df_marketData.iloc[ndx]['High'], \
                              'low' : self.df_marketData.iloc[ndx]['Low'], \
                              'close' : self.df_marketData.iloc[ndx]['Close'], \
                              'volume' : self.df_marketData.iloc[ndx]['Volume']}
            self.candle = candleDict
            return self.candle
            
        except ValueError:
            sys.exit(exc_txt)
    ''' ============ return a specific single candle - end ============= '''
   
    ''' ============ return the candles as a pandas data frame - start ============= '''
    def dataFrameCandles(self):
        #print("Returning candle count")
        exc_txt = "\nAn exception occurred - unable to return candles as a dataframe"
        try:
            return self.df_marketData
            
        except ValueError:
            sys.exit(exc_txt)
    ''' ============ return the candles as a pandas data frame - end ============= '''

    ''' ============ return the number of candles in the archived data - start ============= '''
    def candleCount(self):
        #print("Returning candle count")
        exc_txt = "\nAn exception occurred - unable to determine candle count"
        try:
            return len(self.df_marketData)
            
        except ValueError:
            sys.exit(exc_txt)
    ''' ============ return the number of candles in the archived data - end ============= '''


''' class for a current market data - start '''
 
''' class for a local archive of basic market data - start '''
class BasicMarketDataArchive(basicMarketData):
    '''     classdocs      '''
    '''     class data     '''

    '''         Constructor        '''
    def __init__(self, symbol):
        exc_txt = "An exception occurred creating a basicMarketData object for {}".format(symbol)
        try:
            #print("basicMarketData init {}".format(symbol))
            super().__init__(symbol)
            self.periodType = MAX_HISTORY_TYPE
            self.period = MAX_HISTORY_PERIOD
            self.frequencyType = ARCHIVE_FREQUENCY_TYPE
            self.frequency = ARCHIVE_FREQUENCY
            self.archiveFolder = super().locateArchive(BASIC_ARCHIVE_DIR)
            return

        except Exception:
            exc_info = sys.exc_info()
            exc_str = exc_info[1].args[0]
            exc_txt = exc_txt + "\n\t" + exc_str
            sys.exit(exc_txt)
    
    def updateLocalArchive(self):
        exc_txt = "An exception occurred creating a basicMarketData object for {}".format(self.symbol)
        try:
            '''
            response = self.financialDataServicesObj.requestMarketData(self.symbol, \
                                                                      periodType=self.periodType, period=self.period, \
                                                                      frequencyType=self.frequencyType, frequency=self.frequency)
            '''
            #self.updateArchiveFile()
            self.eod_file = self.archiveFolder + '\\' + self.symbol + '.csv'
            if os.path.isfile(self.eod_file):
                #print("Basic market data file for {} exists, {}".format(self.symbol, eod_file))
                self.df_eod = pd.read_csv(self.eod_file)
                self.df_eod = self.df_eod.drop_duplicates(subset='DateTime')
                last_row_datetime = int(self.df_eod.iloc[-1]['DateTime'])
                now = int(time.time() * 1000)
                # candles requested based on start and end dates
                self.response = self.financialDataServicesObj.requestMarketData(symbol=self.symbol, \
                                                                                periodType=self.periodType, period=self.period, 
                                                                                frequencyType=self.frequencyType, frequency=self.frequency, \
                                                                                startDate=last_row_datetime, endDate=now)
            else:
                #print("Basic market data file for {} does not exists".format(self.symbol))
                self.response = self.financialDataServicesObj.requestMarketData(symbol=self.symbol, \
                                                                                periodType=self.periodType, period=self.period, 
                                                                                frequencyType=self.frequencyType, frequency=self.frequency, \
                                                                                startDate=None, endDate=None)
                self.df_eod = pd.DataFrame(columns=['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume'])
                
            self.marketDataReturn = self.response.text
            self.marketDataJson = json.loads(self.marketDataReturn)
            #self.updateArchiveFile(self.response, self.df_eod, self.eod_file)
            
            colNames = ['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume']
            df_new = pd.DataFrame(columns=colNames)
            for ndx in range(len(self.marketDataJson["candles"])):
                self.candle = self.marketDataJson["candles"][ndx]
                
                # Create a new row to append
                new_row = {'DateTime' : self.candle.candleDateValue * 1000, \
                           'Open' : self.candle.candleOpen, \
                           'High' : self.candle.candleHigh, \
                           'Low' : self.candle.candleLow, \
                           'Close' : self.candle.candleClose, \
                           'Volume' : self.candle.candleVolume}
                
                df_new.loc[len(df_new)] = new_row                
                
            if len(self.df_eod) == 0:
                self.df_marketData = df_new
            else:
                self.df_marketData = pd.concat([self.df_eod, df_new], ignore_index=True)
            
            # Archive market data
            if 'index' in self.df_marketData:
                self.df_marketData = self.df_marketData.drop('index', axis=1)
            self.df_marketData = self.df_marketData.drop_duplicates(subset='DateTime')
            self.df_marketData.to_csv(self.eod_file, index=False)

            return

        except Exception:
            exc_info = sys.exc_info()
            exc_str = exc_info[1].args[0]
            exc_txt = exc_txt + "\n\t" + exc_str
            sys.exit(exc_txt)
    
    ''' ============  update archive file - start ============= '''
    def updateArchiveFile(self):
        exc_txt = "\nAn exception occurred - unable to update market data archive file"
        try:

            return 
            
        except Exception:
            exc_info = sys.exc_info()
            exc_str = exc_info[1].args[0]
            sys.exit(exc_txt + "\n\t" + exc_str)
    ''' ============  update archive file  - end ============= '''
''' class for a local archive of basic market data - end '''

''' class for a local archive of market data enriched by calculated fields - start '''
class EnrichedMarketDataArchive(basicMarketData):
    '''     classdocs      '''
    '''     class data     '''

    '''         Constructor        '''
    def __init__(self, symbolList):
        exc_txt = "An exception occurred creating the EnrichedMarketDataArchive object"
        try:
            #print("EnrichedMarketDataArchive init {}".format(symbol))
            self.symbolList = symbolList
            self.basicArchiveFolder = super().locateArchive(BASIC_ARCHIVE_DIR)
            self.enrichedArchiveFolder = super().locateArchive(ENHANCED_ARCHIVE_DIR)
            '''
            self.symbol = symbol
            self.df_basicMarketData = pd.read_csv(self.basicArchiveFolder + '\\' + self.symbol + '.csv')
            df_marketData = self.df_basicMarketData
            '''
            self.updateLocalArchive()

            return

        except Exception:
            exc_info = sys.exc_info()
            exc_str = exc_info[1].args[0]
            exc_txt = exc_txt + "\n\t" + exc_str
            sys.exit(exc_txt)
    
    ''' ============ retrieve market data and maintain a local archive of equity instruments market data history - start ============= '''
    def updateLocalArchive(self):
        #print("Archiving market data for {}".format(self.symbol))
        exc_txt = "\nAn exception occurred - unable to maintain market data archive"
        try:
            with multiprocessing.Pool(processes=os.cpu_count()) as pool:
                results = pool.map(self.localArchiveWorker, self.symbolList)
                #print("worker result - {}".format(results))
            return
            
        except Exception:
            exc_info = sys.exc_info()
            exc_str = exc_info[1].args[0]
            sys.exit(exc_txt + "\n\t" + exc_str)
    ''' ============ retrieve market data and maintain a local archive of equity instruments market data history - end ============= '''

    def localArchiveWorker(self, symbol):
        #print("local archive worker thread starting - symbol {}\n\tbasic {}\n\tenriched {}".format(symbol, self.basicArchiveFolder, self.enrichedArchiveFolder))
        df_basicMarketData = pd.read_csv(self.basicArchiveFolder + '\\' + symbol + '.csv')
        df_marketData = df_basicMarketData
        
        df_marketData = self.calculateEnrichedHistory(df_marketData)
        
        df_marketData.to_csv(self.enrichedArchiveFolder + '\\' + symbol + '.csv', index=False)
        print("enriched {}, {} data points".format(symbol, len(df_marketData)))
        return symbol
    
    def add_derived_data(self, df_marketData):
        df_marketData.insert(loc=0, column='1 day change', value=NaN)
        df_marketData.insert(loc=0, column='5 day change', value=NaN)
        df_marketData.insert(loc=0, column='10 day change', value=NaN)
        df_marketData.insert(loc=0, column='10 day max', value=NaN)
        df_marketData.insert(loc=0, column='10 day min', value=NaN)
        df_marketData.insert(loc=0, column='20 day change', value=NaN)
        df_marketData.insert(loc=0, column='20 day max', value=NaN)
        df_marketData.insert(loc=0, column='20 day min', value=NaN)
        df_marketData.insert(loc=0, column='40 day change', value=NaN)
        df_marketData.insert(loc=0, column='40 day max', value=NaN)
        df_marketData.insert(loc=0, column='40 day min', value=NaN)
        df_marketData.insert(loc=0, column='date', value="")
        #df_marketData.insert(loc=0, column='month', value="")
        df_marketData.insert(loc=0, column='day', value="")
        #df_marketData.insert(loc=0, column='weekday', value="")
        df_marketData.insert(loc=0, column='10day5pct', value=False)
        df_marketData.insert(loc=0, column='10day10pct', value=False)
        df_marketData.insert(loc=0, column='10day25pct', value=False)
        df_marketData.insert(loc=0, column='10day50pct', value=False)
        df_marketData.insert(loc=0, column='10day100pct', value=False)
        
        df_marketData.insert(loc=0, column='EMA12', value=NaN)
        df_marketData.insert(loc=0, column='EMA20', value=NaN)
        df_marketData.insert(loc=0, column='EMA26', value=NaN)
        df_marketData.insert(loc=0, column='SMA20', value=NaN)
    
        idx = 0
        while idx < len(df_marketData):
            df_marketData.at[idx, 'date'] = format_tda_datetime( df_marketData.at[idx, 'DateTime'] )
            idx += 1
        #print(df_marketData)
        df_marketData = df_marketData.drop_duplicates(subset=['date'], keep='last', inplace=False)
        #print(df_marketData)
        #df_marketData = df_marketData.reset_index()
        df_marketData = df_marketData.set_index(i for i in range(0, df_marketData.shape[0]))
        #print(df_marketData)
        df_marketData = exponential_moving_average(df_marketData[:], value_label="Close", interval=12, EMA_data_label='EMA12')
        df_marketData = exponential_moving_average(df_marketData[:], value_label="Close", interval=20, EMA_data_label='EMA20')
        df_marketData = exponential_moving_average(df_marketData[:], value_label="Close", interval=26, EMA_data_label='EMA26')
        df_marketData = simple_moving_average(df_marketData[:], value_label="Close", avg_interval=20, SMA_data_label='SMA20')
        
        idx = 0
        while idx < len(df_marketData):
            '''
            df_marketData.at[idx, 'month'] = date.month(df_marketData.at[idx, 'date'])
            df_marketData.at[idx, 'weekday'] = date.weekday(df_marketData.at[idx, 'date'])
            '''
            df_marketData.at[idx, "day"] = date.fromtimestamp(df_marketData.at[idx, "DateTime"]/1000).timetuple().tm_yday
            closing_price = df_marketData.at[idx, "Close"]
            try:
                if not closing_price == 0:
                    if idx < len(df_marketData) - 1:
                        df_marketData.loc[idx, '1 day change'] = (df_marketData.loc[idx + int(1), "Close"] - closing_price) / closing_price                    
                    if idx < len(df_marketData) - 5:
                        df_marketData.loc[idx, '5 day change'] = (df_marketData.loc[idx + int(5), "Close"] - closing_price) / closing_price
                    if idx < len(df_marketData) - 10:
                        df_marketData.loc[idx, '10 day change'] = (df_marketData.loc[idx + int(10), "Close"] - closing_price) / closing_price
                        df_marketData.loc[idx, '10 day max'] = df_marketData.iloc[idx:idx+10].get('High').max()
                        df_marketData.loc[idx, '10 day min'] = df_marketData.iloc[idx:idx+10].get('Low').min()                
                        if df_marketData.loc[idx, '10 day max'] > df_marketData.loc[idx, 'Close'] * 1.1:
                            df_marketData.loc[idx, '10day10pct'] = True
                            if df_marketData.loc[idx, '10 day max'] > df_marketData.loc[idx, 'Close'] * 1.25:
                                df_marketData.loc[idx, '10day25pct'] = True
                                if df_marketData.loc[idx, '10 day max'] > df_marketData.loc[idx, 'Close'] * 1.50:
                                    df_marketData.loc[idx, '10day50pct'] = True
                                    if df_marketData.loc[idx, '10 day max'] > df_marketData.loc[idx, 'Close'] * 2:
                                        df_marketData.loc[idx, '10day100pct'] = True
                    if idx < len(df_marketData) - 14:
                        df_marketData.loc[idx, '14 day max'] = df_marketData.iloc[idx:idx+14].get('High').max()
                        df_marketData.loc[idx, '14 day min'] = df_marketData.iloc[idx:idx+14].get('Low').min()
                    if idx < len(df_marketData) - 20:
                        df_marketData.loc[idx, '20 day change'] = (df_marketData.loc[idx + 20, "Close"] - closing_price) / closing_price
                        df_marketData.loc[idx, '20 day max'] = df_marketData.iloc[idx:idx + 20].get('High').max()
                        df_marketData.loc[idx, '20 day min'] = df_marketData.iloc[idx:idx + 20].get('Low').min()
                    if idx < len(df_marketData) - 40:
                        df_marketData.loc[idx, '40 day change'] = (df_marketData.loc[idx + 40, "Close"] - closing_price) / closing_price
                        df_marketData.loc[idx, '40 day max'] = df_marketData.iloc[idx:idx + 40].get('High').max()
                        df_marketData.loc[idx, '40 day min'] = df_marketData.iloc[idx:idx + 40].get('Low').min()
                pass
            except:
                print("error")
            
            idx += 1
    
        return df_marketData

    def calculateEnrichedHistory(self, df_marketData):
        try:
            exc_txt = "An exception occurred in calculateEnrichedHistory"
            #print("EOD data for %s\n%s" % (filename, df_marketData))
            #df_marketData = add_trending_data(df_marketData)
            #df_marketData = add_change_data(df_marketData)
            df_marketData = self.add_derived_data(df_marketData)
            df_marketData = macd(df_marketData[:], value_label="Close")
            df_marketData = on_balance_volume(df_marketData[:], value_label='Close', volume_lable='Volume')
            df_marketData = bollinger_bands(df_marketData[:], value_label="EMA20", ema_interval=20)
            df_marketData = accumulation_distribution(df_marketData)
            df_marketData = aroon_indicator(df_marketData)
            df_marketData = average_directional_index(df_marketData)
            df_marketData = stochastic_oscillator(df_marketData)
            df_marketData = relative_strength(df_marketData, value_label="Close", relative_to='d:\\brian\\AI-Projects\\tda\\market_data\\$spx.x.csv')
            
            return df_marketData
        
        except Exception:
            exc_info = sys.exc_info()
            exc_str = exc_info[1].args[0]
            sys.exit(exc_txt + "\n\t" + exc_str)
''' class for a local archive of market data enriched by calculated fields - end '''
            

''' ============ Individual market data candle class - start ============= '''
class MarketDataCandle(object):
    '''     classdocs      '''
    '''     class data     '''

    '''         Constructor        '''
    def __init__(self, candle):
        self.candleOpen = float(candle["open"])
        self.candleClose = float(candle["close"])
        self.candleHigh = float(candle["high"])
        self.candleLow = float(candle["low"])
        self.candleVolume = int(candle["volume"])
        
        self.candleDateValue = float(candle["datetime"]/1000)
        dt = datetime.datetime.fromtimestamp(self.candleDateValue)
        self.candleDateTimeStr = dt.strftime("%Y-%m-%d")

    ''' ============ candle storage and access - start ============= '''
    @property
    def candle(self):
        return self._candle
    
    @candle.setter
    def candle(self, candle):
        self._candle = candle

    @property
    def candleDateValue(self):
        return self._candleDateValue
    
    @candleDateValue.setter
    def candleDateValue(self, candleDateValue):
        self._candleDateValue = candleDateValue

    @property
    def candleOpen(self):
        return self._candleOpen
    
    @candleOpen.setter
    def candleOpen(self, candleOpen):
        self._candleOpen = candleOpen

    @property
    def candleClose(self):
        return self._candleClose
    
    @candleClose.setter
    def candleClose(self, candleClose):
        self._candleClose = candleClose

    @property
    def candleHigh(self):
        return self._candleHigh
    
    @candleHigh.setter
    def candleHigh(self, candleHigh):
        self._candleHigh = candleHigh

    @property
    def candleLow(self):
        return self._candleLow
    
    @candleLow.setter
    def candleLow(self, candleLow):
        self._candleLow = candleLow

    @property
    def candleVolume(self):
        return self._candleVolume
    
    @candleVolume.setter
    def candleVolume(self, candleVolume):
        self._candleVolume = candleVolume

    @property
    def candleDateTimeStr(self):
        return self._candleDateTimeStr
    
    @candleDateTimeStr.setter
    def candleDateTimeStr(self, candleDateTimeStr):
        self._candleDateTimeStr = candleDateTimeStr

''' ============ candle storage and access - end ============= '''
''' ============ Individual market data candle class - end ============= '''
