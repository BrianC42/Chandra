'''
Created on Jul 23, 2024

@author: brian
'''
import os
import sys
import json
import datetime
import time
import pandas as pd

from configuration import get_ini_data

from financialDataServices import financialDataServices

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
class MarketData(object):
    '''     classdocs      '''
    
    '''     class data     '''
    ARCHIVE_PERIOD_TYPE = "month"
    ARCHIVE_FREQUENCY_TYPE = "daily"
    ARCHIVE_FREQUENCY = "1"
    MAX_HISTORY_PERIOD = "2"
    MAX_HISTORY_TYPE = 'year'

    '''         Constructor        '''
    def __init__(self, symbol, useArchive=True, \
                 periodType=ARCHIVE_PERIOD_TYPE, period=MAX_HISTORY_PERIOD, \
                 frequencyType=ARCHIVE_FREQUENCY_TYPE, frequency=ARCHIVE_FREQUENCY):
        
        exc_txt = "An exception occurred creating a MarketData object for {}".format(symbol)
        try:

            self.symbol = symbol
            self.useArchive = useArchive
            self.periodType = periodType
            self.period = period
            self.frequencyType = frequencyType
            self.frequency = frequency
            
            self.financialDataServicesObj = financialDataServices()
            
            if self.useArchive:
                self.maintainMarketDataArchive()
            else:
                self.validateRequestParams()
                self.requestMarketData()
    
        except ValueError:
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
   
    ''' ============ retrieve market data and maintain a local archive of equity instruments market data history - start ============= '''
    def maintainMarketDataArchive(self):
        #print("Archiving market data for {}".format(self.symbol))
        exc_txt = "\nAn exception occurred - unable to maintain market data archive"
        try:
            localDirs = get_ini_data("LOCALDIRS")
            aiwork = localDirs['aiwork']
            
            basicMarketDataDir = aiwork + localDirs['market_data'] + localDirs['basic_market_data']
            augmentedMarketDataDir = aiwork + localDirs['market_data'] + localDirs['augmented_market_data']

            eod_file = basicMarketDataDir + '\\' + self.symbol + '.csv'
            if os.path.isfile(eod_file):
                #print("Basic market data file for {} exists, {}".format(self.symbol, eod_file))
                df_eod = pd.read_csv(eod_file)
                df_eod = df_eod.drop_duplicates(subset='DateTime')
                last_row_datetime = int(df_eod.iloc[-1]['DateTime'])
                now = int(time.time() * 1000)
                # candles requested based on start and end dates
                response = self.financialDataServicesObj.requestMarketData(symbol=self.symbol, \
                                                         periodType=self.periodType, period=self.period, 
                                                         frequencyType=self.ARCHIVE_FREQUENCY_TYPE, frequency=self.ARCHIVE_FREQUENCY, \
                                                         startDate=last_row_datetime, endDate=now)
            else:
                #print("Basic market data file for {} does not exists".format(self.symbol))
                response = self.financialDataServicesObj.requestMarketData(symbol=self.symbol, \
                                                         periodType=self.MAX_HISTORY_TYPE, period=self.MAX_HISTORY_PERIOD, 
                                                         frequencyType=self.ARCHIVE_FREQUENCY_TYPE, frequency=self.ARCHIVE_FREQUENCY, \
                                                         startDate=None, endDate=None)
                df_eod = pd.DataFrame(columns=['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume'])
                
            self.marketDataReturn = response.text
            self.marketDataJson = json.loads(self.marketDataReturn)
        
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
                
            if len(df_eod) == 0:
                self.df_marketData = df_new
            else:
                self.df_marketData = pd.concat([df_eod, df_new], ignore_index=True)
            
            # Archive market data
            if 'index' in self.df_marketData:
                self.df_marketData = self.df_marketData.drop('index', axis=1)
            self.df_marketData = self.df_marketData.drop_duplicates(subset='DateTime')
            self.df_marketData.to_csv(eod_file, index=False)
            return
            
        except Exception:
            exc_info = sys.exc_info()
            exc_str = exc_info[1].args[0]
            sys.exit(exc_txt + "\n\t" + exc_str)
    ''' ============ retrieve market data and maintain a local archive of equity instruments market data history - end ============= '''
   
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
    ''' ============ return a specific single candle - end ============= '''
   
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
