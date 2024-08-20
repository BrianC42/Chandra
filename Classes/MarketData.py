'''
Created on Jul 23, 2024

@author: brian
'''
import json
import datetime

from financialDataServices import financialDataServices

class MarketDataCandle(object):
    '''
    classdocs
    
    '''
    
    '''
    class data
    '''

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

class MarketData(object):
    '''
    classdocs
    
    '''
    
    '''
    class data
    '''

    '''         Constructor        '''
    def __init__(self, symbol, periodType="", period="", frequencyType="", frequency=""):
        self.symbol = symbol
        self.periodType = periodType
        self.period = period
        self.frequencyType = frequencyType
        self.frequency = frequency
        self.financialDataServicesObj = financialDataServices()
        response = self.financialDataServicesObj.requestMarketData(symbol=symbol, \
                                                 periodType=periodType, period=period, 
                                                 frequencyType=frequencyType, frequency=frequency)
        
        self.marketDataReturn = response.text
        self.marketDataJson = json.loads(self.marketDataReturn)
        
    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= len(self.marketDataJson["candles"]):
            raise StopIteration

        else:
            self.candle = self.marketDataJson["candles"][self.index]
            self.index += 1
            return self.candle
   
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
