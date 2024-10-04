'''
Created on Aug 12, 2024

@author: brian
'''
import sys
import json
import pandas as pd

from configuration import get_ini_data

from financialDataServices import financialDataServices

''' ================================= OptionDetails - start ===========================
    Class to manage the option data for a single option
'''
class OptionDetails(object):
    '''
    classdocs
    
    '''
    
    '''
    class data
    '''

    '''  ====== OptionDetails Constructor ====== '''
    def __init__(self, symbol, strategy="", expirationDate="", daysToExpiry=0, strikePrice=0.0, optionDetails=""):
        try:
            exc_txt = "Unable to construct option object"
            
            self.symbol = symbol
            
            if strategy == "putExpDateMap":
                self.strategy = "put"
            elif strategy == "callExpDateMap":
                self.strategy = "call"
            else:
                raise ValueError
            
            self.expirationDate = expirationDate
            self.daysToExpiry = daysToExpiry
            self.strikePrice = strikePrice
            self.optionDetails = optionDetails
            
            self.putCall = self.optionDetails["putCall"]
            self.description = self.optionDetails["description"]
            self.optionSymbol = self.optionDetails["symbol"]
            self.inTheMoney = self.optionDetails["inTheMoney"]
            
            self.bidSize = int(self.optionDetails["bidSize"])
            self.askSize = int(self.optionDetails["askSize"])
    
            self.bidPrice = float(self.optionDetails["bid"])
            self.askPrice = float(self.optionDetails["ask"])
            self.lastPrice = float(self.optionDetails["last"])
            self.markPrice = float(self.optionDetails["mark"])
            self.closePrice = float(self.optionDetails["closePrice"])
            self.volatility = float(self.optionDetails["volatility"])
            self.delta = float(self.optionDetails["delta"])
            self.gamma = float(self.optionDetails["gamma"])
            self.theta = float(self.optionDetails["theta"])
            self.vega = float(self.optionDetails["vega"])
            self.rho = float(self.optionDetails["rho"])

        except (ValueError, Exception):
            exc_info = sys.exc_info()
            exc_str = exc_info[1].args[0]
            exc_txt = exc_txt + "\n\t" + exc_str
            sys.exit(exc_txt)

        '''
        symbol    
        strategy    (putCall)
        expiration    
        days To Expiration    
        strike Price    
        bid    
        ask    
        expiration Date    

        closePrice    self.closePrice = float(option["closePrice"])
        volatility    self.volatility = float(option["volatility"])
        delta    self.delta = float(option["delta"])
        gamma    self.gamma = float(option["gamma"])
        theta    self.theta = float(option["theta"])
        vega    self.vega = float(option["vega"])
        rho    self.rho = float(option["rho"])
        inTheMoney    self.inTheMoney = option["inTheMoney"]


                underlying Price    
                break even    
                OTM Probability    
                ADX    
                probability of loss    
                ROI    
                Max Loss    
                Preferred Outcome    
                Preferred Result    
                Unfavored Result        
                
                            Purchase $    
                            Earnings    
                            Dividend    
                            Current Holding    
                            Qty    
                            max gain APY    
                            Max Profit    
                            Risk Management    
                            Loss vs. Profit    
                            premium    
                            commission    
                            Earnings Date    
                            dividend Date    

        self.lastSize = int(option["lastSize"])
        self.highPrice = float(option["highPrice"])
        self.lowPrice = float(option["lowPrice"])
        self.openPrice = float(option["openPrice"])
        self.totalVolume = int(option["totalVolume"])
        self.quoteTimeInLong = int(option["quoteTimeInLong"])
        self.tradeTimeInLong = int(option["tradeTimeInLong"])
        self.netChange = float(option["netChange"])
        self.timeValue = float(option["timeValue"])
        self.openInterest = int(option["openInterest"])
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
    ''' Option formatting methods
        xxx - return as json
        xxx - return as dict
        xxx - return as pandas dataframe row
    '''
        
    ''' Option constructor data field access functions '''
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

    @property
    def closePrice(self):
        return self._closePrice
    
    @closePrice.setter
    def closePrice(self, closePrice):
        self._closePrice = closePrice

    @property
    def volatility(self):
        return self._volatility
    
    @volatility.setter
    def volatility(self, volatility):
        self._volatility = volatility

    @property
    def delta(self):
        return self._delta
    
    @delta.setter
    def delta(self, delta):
        self._delta = delta

    @property
    def gamma(self):
        return self._gamma
    
    @gamma.setter
    def gamma(self, gamma):
        self._gamma = gamma

    @property
    def theta(self):
        return self._theta
    
    @theta.setter
    def theta(self, theta):
        self._theta = theta

    @property
    def vega(self):
        return self._vega
    
    @vega.setter
    def vega(self, vega):
        self._vega = vega

    @property
    def rho(self):
        return self._rho
    
    @rho.setter
    def rho(self, rho):
        self._rho = rho

    @property
    def inTheMoney(self):
        return self._inTheMoney
    
    @inTheMoney.setter
    def inTheMoney(self, inTheMoney):
        self._inTheMoney = inTheMoney

    '''        
    @property
    def xxx(self):
        return self._xxx
    
    @xxx.setter
    def xxx(self, xxx):
        self._xxx = xxx
    '''
        
''' =============================================================================
    Class to manage the option data for a single option
================================= OptionDetails - end =========================== '''
    
class OptionChain(object):
    '''      classdocs       '''
    
    ''' accessible data '''
    
    ''' accessible methods   '''
    
    ''' ======= OptionChain Constructor ======== '''
    def __init__(self, symbol="", strategy="Both", strikeCount=5, strikeRange="OTM", daysToExpiration=60):
        exc_txt = "An exception occurred creating an OptionChain object for {}".format(symbol)
        try:
            localDirs = get_ini_data("LOCALDIRS")
            aiwork = localDirs['aiwork']
            #basicMarketDataDir = aiwork + localDirs['market_data'] + localDirs['basic_market_data']
            #augmentedMarketDataDir = aiwork + localDirs['market_data'] + localDirs['augmented_market_data']
            #financialInstrumentDetailsDir = aiwork + localDirs['market_data'] + localDirs['financial_instrument_details']
            optionChainDir = aiwork + localDirs['market_data'] + localDirs['option_chains']
                
            self.symbol = symbol
            self.strategy = strategy
            self.strikeCount = strikeCount
            self.strikeRange = strikeRange
            self.daysToExpiration = daysToExpiration
            
            self.financialDataServicesObj = financialDataServices()
            response = self.financialDataServicesObj.requestOptionChain(type=strategy, symbol=symbol, \
                                                                        strikeCount=strikeCount, range=strikeRange, \
                                                                        daysToExpiration=daysToExpiration)
    
            if response.status_code == 200:
                self.optionChainData = response.text
                self.optionChainJson = json.loads(self.optionChainData)
                
                optionChainRows = []
                for chain in ['putExpDateMap', 'callExpDateMap']:
                    for exp_date, options in self.optionChainJson[chain].items():
                        exp_date, daysToExp = exp_date.split(":")
                        for strike_price, options_data in options.items():
                            for opt in options_data:
                                option = OptionDetails(self.symbol, strategy=chain, expirationDate=exp_date, daysToExpiry=daysToExp, \
                                                    strikePrice=strike_price, optionDetails=opt)
                                
                                # Create a new row to append
                                new_row = {'symbol' : option.symbol, \
                                           'strategy' : option.strategy, \
                                           'expirationDate' : option.expirationDate, \
                                           'strike Price' : option.strikePrice, \
                                           'days To Expiration' : option.daysToExpiry, \
                                           'bid' : option.bidPrice, \
                                           'ask' : option.askPrice, \
                                           'closePrice' : option.closePrice, \
                                           'volatility' : option.volatility, \
                                           "delta" : option.delta, \
                                           "gamma" : option.gamma, \
                                           "theta" : option.theta, \
                                           "vega" : option.vega, \
                                           "rho" : option.rho, \
                                           "inTheMoney" : option.inTheMoney
                                           }
                                '''
                                "lastSize" : option. \
                                "highPrice" : option. \
                                "lowPrice" : option. \
                                "openPrice" : option. \
                                "totalVolume" : option. \
                                "quoteTimeInLong" : option. \
                                "tradeTimeInLong" : option. \
                                "netChange" : option. \
                                "timeValue" : option. \
                                "openInterest" : option. \
                                "theoreticalOptionValue" : option. \
                                "theoreticalVolatility" : option. \
                                "mini" : option. \
                                "nonStandard" : option. \
                                "optionDeliverablesList" : option. \
                                "expirationType" : option. \
                                "lastTradingDay" : option. \
                                "multiplier" : option. \
                                "settlementType" : option. \
                                "deliverableNote" : option. \
                                "percentChange" : option. \
                                "markChange" : option. \
                                "markPercentChange" : option. \
                                "pennyPilot" : option. \
                                "intrinsicValue" : option. \
                                "optionRoot" : option.
                                '''
                                optionChainRows.append(new_row)
                '''
                self.df_OptionChain = pd.DataFrame(index=['symbol', 'strategy', 'expirationDate', 'strike Price'], \
                                                   columns=['days To Expiration', 'bid', 'ask', 'closePrice', \
                                                            "volatility", "delta", "gamma", "theta", "vega", "rho", \
                                                            "inTheMoney", "lastSize", "highPrice", "lowPrice", "openPrice", \
                                                            "totalVolume", "quoteTimeInLong", "tradeTimeInLong", "netChange", \
                                                            "timeValue", "openInterest", "theoreticalOptionValue", "theoreticalVolatility", \
                                                            "mini", "nonStandard", "optionDeliverablesList", "strikePrice", \
                                                            "expirationDate", "daysToExpiration", "expirationType", "lastTradingDay", \
                                                            "multiplier", "settlementType", "deliverableNote", "percentChange", \
                                                            "markChange", "markPercentChange", "pennyPilot", "intrinsicValue", \
                                                            "optionRoot"])
                '''
                self.df_OptionChain = pd.DataFrame(optionChainRows)                
                self.df_OptionChain.set_index(['symbol', 'strategy', 'expirationDate', 'strike Price'], inplace=True)
                return
            else:
                exc_txt = 'Unable to obtain option chain for {}. Error code = {}'.format(self.symbol, response.status_code)
                raise Exception
                
        except (ValueError, Exception):
            exc_info = sys.exc_info()
            exc_str = exc_info[1].args[0]
            exc_txt = exc_txt + "\n\t" + exc_str
            sys.exit(exc_txt)

    ''' ===== iterator over the options in the option chain ===== '''
    def __iter__(self):
        for chain in ['putExpDateMap', 'callExpDateMap']:
            for exp_date, options in self.optionChainJson[chain].items():
                exp_date, daysToExp = exp_date.split(":")
                for strike_price, options_data in options.items():
                    for opt in options_data:
                        option = OptionDetails(self.symbol, strategy=chain, expirationDate=exp_date, daysToExpiry=daysToExp, \
                                            strikePrice=strike_price, optionDetails=opt)
                        yield option

    ''' ============ iterator over the options in the option chain - end ============= '''

    '''
    optionCount
    iloc
    dataframeOptionsChain
    filterChain
    '''
    ''' ===== xxx - start ===== '''
    def xxx(self):
        try:
            exc_txt = "An xxx exception occurred {}".format(self.symbol)
            
            return
    
        except ValueError:
            exc_info = sys.exc_info()
            exc_str = exc_info[1].args[0]
            exc_txt = exc_txt + "\n\t" + exc_str
            sys.exit(exc_txt)
    ''' ===== xxx - end ===== '''

    ''' ============ data service provider and option chain controls - start ============= '''
    @property
    def symbol(self):
        return self._symbol
    
    @symbol.setter
    def symbol(self, symbol):
        self._symbol = symbol
        
    @property
    def strategy(self):
        return self._strategy
    
    @strategy.setter
    def strategy(self, strategy):
        self._strategy = strategy

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
    def df_OptionChain(self):
        return self._df_OptionChain
    
    @df_OptionChain.setter
    def df_OptionChain(self, df_OptionChain):
        self._df_OptionChain = df_OptionChain

    @property
    def optionChainData(self):
        return self._optionChainData
    
    @optionChainData.setter
    def optionChainData(self, optionChainData):
        self._optionChainData = optionChainData
    ''' ============ data service provider returned data - end ============= '''
