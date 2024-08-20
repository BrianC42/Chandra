'''
Created on Aug 12, 2024

@author: brian
'''
import json

from financialDataServices import financialDataServices

''' ================================= OptionDetails - start ===========================
    Class to manage the option data for a single option
'''
class OptionDetails(object):
    '''
    self.symbol, expirationDate=exp_date, daysToExpiry=daysToExp, \
                                            strikePrice=strike_price, optionDetails=option
    classdocs
    
    '''
    
    '''
    class data
    '''

    '''         Constructor        '''
    def __init__(self, symbol, expirationDate="", daysToExpiry=0, strikePrice=0.0, optionDetails=""):
        self.symbol = symbol
        self.expirationDate = expirationDate
        self.daysToExpiry = daysToExpiry
        self.strikePrice = strikePrice
        self.optionDetails = optionDetails
        
        self.putCall = self.optionDetails["putCall"]
        self.description = self.optionDetails["description"]
        self.optionSymbol = self.optionDetails["symbol"]
        self.bidPrice = float(self.optionDetails["bid"])
        self.askPrice = float(self.optionDetails["ask"])
        self.lastPrice = float(self.optionDetails["last"])
        self.markPrice = float(self.optionDetails["mark"])
        self.bidSize = int(self.optionDetails["bidSize"])
        self.askSize = int(self.optionDetails["askSize"])

        '''
        print("symbol: {}, expiration: {}, strike price: {}".format(symbol, expirationDate, strikePrice))

        self.lastSize = int(option["lastSize"])
        self.highPrice = float(option["highPrice"])
        self.lowPrice = float(option["lowPrice"])
        self.openPrice = float(option["openPrice"])
        self.closePrice = float(option["closePrice"])
        self.totalVolume = int(option["totalVolume"])
        self.quoteTimeInLong = int(option["quoteTimeInLong"])
        self.tradeTimeInLong = int(option["tradeTimeInLong"])
        self.netChange = float(option["netChange"])
        self.volatility = float(option["volatility"])
        self.delta = float(option["delta"])
        self.gamma = float(option["gamma"])
        self.theta = float(option["theta"])
        self.vega = float(option["vega"])
        self.rho = float(option["rho"])
        self.timeValue = float(option["timeValue"])
        self.openInterest = int(option["openInterest"])
        self.inTheMoney = option["inTheMoney"]
        self.theoreticalOptionValue = float(option["theoreticalOptionValue"])
        self.theoreticalVolatility = float(option["theoreticalVolatility"])
        self.mini = option["mini"]
        self.nonStandard = option["nonStandard"]
        self.optonDeliverablesList = option["optionDeliverablesList"]
        self.strikePrice = float(option["strikePrice"])
        self.expirationDate = option["expirationDate"]
        self.daysToExpiration = int(option["daysToExpiration"])
        self.expirationType = option["expirationType"]
        self.lastTradingDay = float(option["lastTradingDay"])
        self.multiplier = option["multiplier"]
        self.settlementType = option["settlementType"]
        self.deliverableNote = option["deliverableNote"]
        self.percentChange = float(option["percentChange"])
        self.markChange = float(option["markChange"])
        self.markPercentChange = float(option["markPercentChange"])
        self.pennyPilot = option["pennyPilot"]
        self.intrinsicValue = float(option["intrinsicValue"])
        self.optionRoot = option["optionRoot"]
        '''
    @property
    def optionDetails(self):
        return self._optionDetails
    
    @optionDetails.setter
    def optionDetails(self, optionDetails):
        self._optionDetails = optionDetails
        
    @property
    def symbol(self):
        return self._symbol
    
    @symbol.setter
    def symbol(self, symbol):
        self._symbol = symbol

    @property
    def expirationDate(self):
        return self._expirationDate
    
    @expirationDate.setter
    def expirationDate(self, expirationDate):
        self._expirationDate = expirationDate

    @property
    def daysToExpiry(self):
        return self._daysToExpiry
    
    @daysToExpiry.setter
    def daysToExpiry(self, daysToExpiry):
        self._daysToExpiry = daysToExpiry

    @property
    def strikePrice(self):
        return self._strikePrice
    
    @strikePrice.setter
    def strikePrice(self, strikePrice):
        self._strikePrice = strikePrice

    ''' ============== option data items ================= '''                        
    @property
    def putCall(self):
        return self._putCall
    
    @putCall.setter
    def putCall(self, putCall):
        self._putCall = putCall
        
    @property
    def description(self):
        return self._description
    
    @description.setter
    def description(self, description):
        self._description = description
        
    @property
    def optionSymbol(self):
        return self._optionSymbol
    
    @optionSymbol.setter
    def optionSymbol(self, optionSymbol):
        self._optionSymbol = optionSymbol
        
    @property
    def bidPrice(self):
        return self._bidPrice
    
    @bidPrice.setter
    def bidPrice(self, bidPrice):
        self._bidPrice = bidPrice
        
    @property
    def askPrice(self):
        return self._askPrice
    
    @askPrice.setter
    def askPrice(self, askPrice):
        self._askPrice = askPrice
        
    @property
    def lastPrice(self):
        return self._lastPrice
    
    @lastPrice.setter
    def lastPrice(self, lastPrice):
        self._lastPrice = lastPrice
        
    @property
    def markPrice(self):
        return self._markPrice
    
    @markPrice.setter
    def markPrice(self, markPrice):
        self._markPrice = markPrice
        
    @property
    def bidSize(self):
        return self._bidSize
    
    @bidSize.setter
    def bidSize(self, bidSize):
        self._bidSize = bidSize
        
    @property
    def askSize(self):
        return self._askSize
    
    @askSize.setter
    def askSize(self, askSize):
        self._askSize = askSize
    '''        
    @property
    def xxx(self):
        return self._xxx
    
    @xxx.setter
    def xxx(self, xxx):
        self._xxx = xxx
    '''
        
'''
    Class to manage the option data for a single option
================================= OptionDetails - end =========================== '''
    
class OptionChain(object):
    '''      classdocs       '''
    
    ''' accessible data '''
    
    ''' accessible methods   '''
    
    def __init__(self, symbol, optionType="Both", strikeCount=5, strikeRange="OTM", daysToExpiration=60):
        '''
        Constructor
        '''
        self.optType = optionType
        self.symbol = symbol
        self.strikeCount = strikeCount
        self.strikeRange = strikeRange
        self.daysToExpiration = daysToExpiration
        
        self.financialDataServicesObj = financialDataServices()
        response = self.financialDataServicesObj.requestOptionChain(type=optionType, symbol=symbol, \
                                                                    strikeCount=strikeCount, range=strikeRange, \
                                                                    daysToExpiration=daysToExpiration)

        if response.status_code == 200:
            self.optionChainData = response.text
            self.optionChainJson = json.loads(self.optionChainData)

    ''' ===== iterator over the options in the option chain ===== '''
    def __iter__(self):
        for chain in ['putExpDateMap', 'callExpDateMap']:
            for exp_date, options in self.optionChainJson[chain].items():
                exp_date, daysToExp = exp_date.split(":")
                for strike_price, options_data in options.items():
                    for opt in options_data:
                        option = OptionDetails(self.symbol, expirationDate=exp_date, daysToExpiry=daysToExp, \
                                            strikePrice=strike_price, optionDetails=opt)
                        yield option

    ''' ============ data service provider and option chain controls - start ============= '''
    @property
    def symbol(self):
        return self._symbol
    
    @symbol.setter
    def symbol(self, symbol):
        self._symbol = symbol
        
    @property
    def optionType(self):
        return self._optionType
    
    @optionType.setter
    def optionType(self, optionType):
        self._optionType = optionType

    @property
    def strikeCount(self):
        return self._strikeCount
    
    @strikeCount.setter
    def strikeCount(self, strikeCount):
        self._strikeCount = strikeCount

    @property
    def strikeRange(self):
        return self._strikeRange
    
    @strikeRange.setter
    def strikeRange(self, strikeRange):
        self._strikeRange = strikeRange

    @property
    def daysToExpiration(self):
        return self._daysToExpiration
    
    @daysToExpiration.setter
    def daysToExpiration(self, daysToExpiration):
        self._daysToExpiration = daysToExpiration

    @property
    def financialDataServicesObj(self):
        return self._financialDataServicesObj
    
    @financialDataServicesObj.setter
    def financialDataServicesObj(self, financialDataServicesObj):
        self._financialDataServicesObj = financialDataServicesObj
    ''' ============ data service provider and option chain controls - end ============= '''

    ''' ============ data service provider returned data - start ============= '''
    @property
    def option(self):
        return self._option
    
    @option.setter
    def option(self, option):
        self._option = OptionDetails(option)

    @property
    def optionChainJson(self):
        return self._optionChainJson
    
    @optionChainJson.setter
    def optionChainJson(self, optionChainJson):
        self._optionChainJson = optionChainJson

    @property
    def optionChainData(self):
        return self._optionChainData
    
    @optionChainData.setter
    def optionChainData(self, optionChainData):
        self._optionChainData = optionChainData
    ''' ============ data service provider returned data - end ============= '''
