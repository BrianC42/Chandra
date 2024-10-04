'''
Created on Sep 17, 2024

@author: brian
'''
import sys

import pandas as pd

''' Google workspace requirements start '''
'''
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
'''
''' Google workspace requirements end '''
from configuration import get_ini_data
from configuration import read_config_json

from GoogleSheets import googleSheet
from MarketData import MarketData
from OptionChain import OptionChain
from FinancialInstrument import FinancialInstrument
#from financialDataServices import financialDataServices

class workbooks(object):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        try:
            exc_txt = "exception developing google workbook class"
            #print("workbook class")
            
            ''' ================ Authenticate with Google workplace and establish a connection to Google Drive API ============== '''
            exc_txt = "\nAn exception occurred - unable to authenticate with Google"
            self.gSheets = googleSheet()
                        
            return
        
        except :
            print(exc_txt)
            exc_info = sys.exc_info()
            if len(exc_info) > 1:
                print(exc_info[1].args[0])
            sys.exit()
        

class investments(workbooks):
    '''
    classdocs
    
    instrumentCellValues.columns
    Index(['Symbol', 'Change %', 'Last Price', 'Dividend Yield', 'Dividend',
           'Dividend Ex-Date', 'P/E Ratio', '52 Week High', '52 Week Low',
           'Volume', 'Sector', 'Next Earnings Date', 'Morningstar', 'Market Cap',
           'Schwab Equity Rating', 'Argus Rating']
           
    Missing data elements
        'Change %',
        'Sector', 'Next Earnings Date', 
        'Morningstar', 'Schwab Equity Rating', 'Argus Rating'
    
    MarketData elements
        'Last Price' : close
        'Volume'
    
    FinancialInstrument data elements
        'symbol': 'AAPL' 'Symbol'
        'high52': 237.23,  '52 Week High'
        'low52': 164.075 '52 Week Low'
        'dividendYield': 0.44269, 'Dividend Yield'
        'dividendAmount': 1.0, 'Dividend'
        'dividendDate': '2024-08-12 00:00:00.0', 'Dividend Ex-Date'
        'peRatio': 34.49389, 'P/E Ratio'
        'marketCap': 3434462506930.0, 'Market Cap'
        
        {
        'symbol': 'AAPL', 
        'high52': 237.23, 
        'low52': 164.075, 
        'dividendAmount': 1.0, 
        'dividendYield': 0.44269, 
        'dividendDate': '2024-08-12 00:00:00.0', 
        'peRatio': 34.49389, 
        'pegRatio': 112.66734, 
        'pbRatio': 48.06189, 
        'prRatio': 8.43929, 
        'pcfRatio': 23.01545, 
        'grossMarginTTM': 45.962, 
        'grossMarginMRQ': 46.2571, 
        'netProfitMarginTTM': 26.4406, 
        'netProfitMarginMRQ': 25.0043, 
        'operatingMarginTTM': 26.4406, 
        'operatingMarginMRQ': 25.0043, 
        'returnOnEquity': 160.5833, 
        'returnOnAssets': 22.6119, 
        'returnOnInvestment': 50.98106, 
        'quickRatio': 0.79752, 
        'currentRatio': 0.95298, 
        'interestCoverage': 0.0, 
        'totalDebtToCapital': 51.3034, 
        'ltDebtToEquity': 151.8618, 
        'totalDebtToEquity': 129.2138, 
        'epsTTM': 6.56667, 
        'epsChangePercentTTM': 10.3155, 
        'epsChangeYear': 0.0, 
        'epsChange': 0.0, 
        'revChangeYear': -2.8005, 
        'revChangeTTM': 0.4349, 
        'revChangeIn': 0.0, 
        'sharesOutstanding': 15204137000.0, 
        'marketCapFloat': 0.0, 
        'marketCap': 3434462506930.0, 
        'bookValuePerShare': 4.38227, 
        'shortIntToFloat': 0.0, 
        'shortIntDayToCover': 0.0, 
        'divGrowthRate3Year': 0.0, 
        'dividendPayAmount': 0.25, 
        'dividendPayDate': '2024-08-15 00:00:00.0', 
        'beta': 1.24364, 
        'vol1DayAvg': 0.0, 
        'vol10DayAvg': 0.0, 
        'vol3MonthAvg': 0.0, 
        'avg10DaysVolume': 47812576, 
        'avg1DayVolume': 60229630, 
        'avg3MonthVolume': 64569166, 
        'declarationDate': '2024-08-01 00:00:00.0', 
        'dividendFreq': 4, 
        'eps': 6.13, 
        'dtnVolume': 40687813, 
        'nextDividendPayDate': '2024-11-15 00:00:00.0', 
        'nextDividendDate': '2024-11-12 00:00:00.0', 
        'fundLeverageFactor': 0.0
        }, 
    '''  

    def __init__(self):
        '''
        Constructor
        '''
        ''' Google drive file details '''
        localDirs = get_ini_data("LOCALDIRS")
        aiwork = localDirs['aiwork']
        
        exc_txt = "\nAn exception occurred - unable to access Google sheet"
        googleAuth = get_ini_data("GOOGLE")
        googleDriveFiles = read_config_json(aiwork + "\\" + googleAuth['fileIDs'])
        
        ''' 
        Use the connetion to Google Drive API to read sheet data
        Find file ID of file used for development 
        '''
        self.sheetID = googleDriveFiles["Google IDs"]["Market Data"]["Production"]
        '''
        print("file 1: {} - {}".format('development', googleDriveFiles["Google IDs"]["Market Data"]["Development"]))
        print("file 2: {} - {}".format('production', googleDriveFiles["Google IDs"]["Market Data"]["Production"]))
        '''
        
        ''' ================ Authenticate with Google workplace and establish a connection to Google Drive API ============== '''
        super().__init__()                    
        self.readAccountTab()
        self.readStockInformationTab()
                    
        return

    def markToMarket(self):
        print("mark to market is WIP ====================================")
        
        try:
            '''
            Steps:
            1. Clear the contents of the 'TD Import Inf' tab
            2. create the list of financial symbols by reading the 'Stock Information' tab
            3. create an empty dataframe
            4. for each symbol request the basic financial instrument and the latest market data from the data service provider
            5. load the dataframe information
            6. Prepare data for Google sheet update
            7. update the workbook's 'TD Import Inf' tab
            '''
            mtomTab = 'FI Inf MtoM'
            mtomCells = 'A1:K999'
            mtomCols = 'A1:K1'
            mtomData = 'A2:K999'
            
            ''' Step 1 - Clear the contents of the 'FI Inf MtoM' tab '''
            exc_txt = "\nAn exception occurred - unable to clear cells"
            self.gSheets.clearGoogleSheet(self.sheetID, mtomTab, mtomCells)
            
            ''' Step 2 - create the list of financial symbols '''
            exc_txt = "\nAn exception occurred - unable to look up symbols required"
            cellRange = 'Stock Information!A2:AF999'
            symbolInformation = self.gSheets.readGoogleSheet(self.sheetID, cellRange)
            print("Stock Information:\n{}".format(symbolInformation))
            
            ''' Step 3 - create a dataframe '''
            indexLabel = "Symbol"
            columnLabels = ["Symbol", "Last Price", "Dividend Yield", "Dividend", "Dividend Ex-Date", "P/E Ratio", \
                            "52 Week High", "52 Week Low", "Volume", "Shares Outstanding", "Market Cap" ]
                        
            self.df_marktomarket = pd.DataFrame(columns=columnLabels)
            self.df_marktomarket.set_index(indexLabel, drop=True, inplace=True)
            
            ''' Step 4 - request the basic financial instrument and the latest market data '''
            exc_txt = "\nAn exception occurred - unable to obtain market information on symbols"
            for rowNdx in range(len(symbolInformation)):
            #for rowNdx in range(15):
                symbol = symbolInformation.loc[rowNdx, 'Symbol']
                exc_txt = "\nAn exception occurred - with symbol: {}".format(symbol)
            
                instrumentDetails = FinancialInstrument(symbol)
                mktData = MarketData(symbol, useArchive=False, periodType="month", period="1", frequencyType="daily", frequency="1")
                candle = mktData.iloc(mktData.candleCount()-1)
                
                ''' step 5. load the dataframe information '''
                new_row = {'symbol' : symbol, \
                           'Last Price' : candle.candleClose, \
                           'Dividend Yield' : instrumentDetails.dividendYield, \
                           'Dividend' : instrumentDetails.dividendAmount, \
                           'Dividend Ex-Date' : instrumentDetails.nextDividendDate, \
                           "P/E Ratio" : instrumentDetails.peRatio, \
                           "52 Week High" : instrumentDetails.high52, \
                           "52 Week Low" : instrumentDetails.low52, \
                           "Volume" : candle.candleVolume, \
                           "Shares Outstanding" : instrumentDetails.sharesOutstanding, \
                           "Market Cap" : instrumentDetails.marketCap
                           }

                self.df_marktomarket.loc[symbol] = new_row
            #print("Mark to market dataframe\n{}".format(self.df_marktomarket))
            
            ''' step 6 - prepare data for Google sheet update '''
            df_update = self.df_marktomarket
            df_update.reset_index(inplace=True)
            
            ''' Step 7 - update the workbook's mark to market tab '''
            exc_txt = "\nAn exception occurred - unable to update Google sheet"
            self.gSheets.updateGoogleSheet(self.sheetID, mtomTab + '!' + mtomCols, df_update.columns)
            self.gSheets.updateGoogleSheet(self.sheetID, mtomTab + '!' + mtomData, df_update)
            
            return 
    
        except :
            print(exc_txt)
            exc_info = sys.exc_info()
            if len(exc_info) > 1:
                print(exc_info[1].args[0])
            sys.exit()

    def readAccountTab(self):
        try:
            exc_txt = "Exception occurred reading accounts tab"
            cellRange = 'Accounts!A2:AU174'
            accountsTab = self.gSheets.readGoogleSheet(self.sheetID, cellRange)
        
            self.accountsTab = pd.DataFrame(accountsTab)                
            self.accountsTab.set_index(['Account', 'Asset'], inplace=True)
            
            return
    
        except :
            print(exc_txt)
            exc_info = sys.exc_info()
            if len(exc_info) > 1:
                print(exc_info[1].args[0])
            sys.exit()
    
    def readStockInformationTab(self):
        try:
            exc_txt = "Exception occurred returning the Stock Information tab"
            cellRange = 'Stock Information!A2:AF999'
            symbolInformation = self.gSheets.readGoogleSheet(self.sheetID, cellRange)

            self.stockInformationTab = pd.DataFrame(symbolInformation)                
            self.stockInformationTab.set_index(['Action Category', 'Symbol'], inplace=True)

            return
    
        except :
            print(exc_txt)
            exc_info = sys.exc_info()
            if len(exc_info) > 1:
                print(exc_info[1].args[0])
            sys.exit()
        
    def symbolDetails(self, symbol):
        try:
            ''' Return details for a symbol '''
            print("symbolDetails is WIP {} =====================================".format(symbol))
            exc_txt = "Exception occurred collecting details on symbol {}".format(symbol)
            details = {"symbol":symbol, "b":1}
                    
            return details
    
        except :
            print(exc_txt)
            exc_info = sys.exc_info()
            if len(exc_info) > 1:
                print(exc_info[1].args[0])
            sys.exit()
        
    def holdings(self):
        try:
            ''' Return a list of asset symbols categorized as current financial holdings '''
            print("holdings is WIP ===================================== - check holding > 0")
            symbolList = []
            exc_txt = "Exception occurred returning holdings list"
            
            for rowNdx in range(len(self.stockInformationTab)):
                actionCategory = self.stockInformationTab.iloc[rowNdx].name[0]
                holding = True
                if (actionCategory == "1 - Holding" or actionCategory == "2 - Call Options") and \
                    holding:
                    symbol = self.stockInformationTab.iloc[rowNdx].name[1]
                    symbolList.append(symbol)
                    
            return symbolList
    
        except :
            print(exc_txt)
            exc_info = sys.exc_info()
            if len(exc_info) > 1:
                print(exc_info[1].args[0])
            sys.exit()
        
    def potentialBuys(self):
        try:
            ''' Return a list of asset symbols categorized as financial assets flagged as potential buys '''
            print("potentialBuys is WIP =====================================")
            symbolList = []
            exc_txt = "Exception occurred returning list of potential buys"
            
            for rowNdx in range(len(self.stockInformationTab)):
                actionCategory = self.stockInformationTab.iloc[rowNdx].name[0]
                if actionCategory == "4 - Buy" or actionCategory == "5 - Put Options":
                    symbol = self.stockInformationTab.iloc[rowNdx].name[1]
                    symbolList.append(symbol)
            
            return symbolList
    
        except :
            print(exc_txt)
            exc_info = sys.exc_info()
            if len(exc_info) > 1:
                print(exc_info[1].args[0])
            sys.exit()

    def potentialCalls(self):
        try:
            ''' Return a list of asset symbols identified as holdings to be sold '''
            symbolList = []
            exc_txt = "Exception occurred returning potential call options"

            for rowNdx in range(len(self.stockInformationTab)):
                actionCategory = self.stockInformationTab.iloc[rowNdx].name[0]
                if actionCategory == "2 - Call Options":
                    symbol = self.stockInformationTab.iloc[rowNdx].name[1]
                    symbolList.append(symbol)
                    
            return symbolList
    
        except :
            print(exc_txt)
            exc_info = sys.exc_info()
            if len(exc_info) > 1:
                print(exc_info[1].args[0])
            sys.exit()
        
    def potentialPuts(self):
        try:
            ''' Return a list of asset symbols categorized as financial assets flagged as potential buys '''
            symbolList = []
            exc_txt = "Exception occurred returning list of potential put options"
            
            for rowNdx in range(len(self.stockInformationTab)):
                actionCategory = self.stockInformationTab.iloc[rowNdx].name[0]
                if actionCategory == "5 - Put Options":
                    symbol = self.stockInformationTab.iloc[rowNdx].name[1]
                    symbolList.append(symbol)
            
            return symbolList
    
        except :
            print(exc_txt)
            exc_info = sys.exc_info()
            if len(exc_info) > 1:
                print(exc_info[1].args[0])
            sys.exit()

    def updateYahooFinanceData(self):
        print("To update Yahoo finance data, use the script in the workbook")
        
        return

    ''' Option constructor data field access functions '''
    @property
    def accountsTab(self):
        return self._accountsTab
    
    @accountsTab.setter
    def accountsTab(self, accountsTab):
        self._accountsTab = accountsTab
        
    @property
    def stockInformationTab(self):
        return self._stockInformationTab
    
    @stockInformationTab.setter
    def stockInformationTab(self, stockInformationTab):
        self._stockInformationTab = stockInformationTab
        

class optionTrades(workbooks):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        ''' Google drive file details '''
        localDirs = get_ini_data("LOCALDIRS")
        aiwork = localDirs['aiwork']
        
        exc_txt = "\nAn exception occurred - unable to access Google sheet"
        googleAuth = get_ini_data("GOOGLE")
        googleDriveFiles = read_config_json(aiwork + "\\" + googleAuth['fileIDs'])

        self.optionsSheetID = googleDriveFiles["Google IDs"]["Market Options"]["Production"]
        '''
        print("file 1: {} - {}".format('development', googleDriveFiles["Google IDs"]["Market Options"]["Development"]))
        print("file 2: {} - {}".format('production', googleDriveFiles["Google IDs"]["Market Options"]["Production"]))
        '''
        
        ''' ================ Authenticate with Google workplace and establish a connection to Google Drive API ============== '''
        super().__init__()
        
        self.investmentSheet = None
        self.symbolDetails = []
        self.potentialPutSymbols = []
        self.filteredPutSymbols = []
        self.potentialCallSymbols = []
        self.filteredCallSymbols = []

        self.potentialOptions = []
        self.filteredOptions = []

        return
        
    def outputPotentalTrades(self):
        print("outputPotentalTrades is WIP")
        try:
            ''' Output the potential option trades to Google sheet '''
            exc_txt = "Exception occurred writing the list of potential option trades"
            
            ''' create new tab '''
            
            ''' wrtie data to new tab '''

            return
    
        except :
            print(exc_txt)
            exc_info = sys.exc_info()
            if len(exc_info) > 1:
                print(exc_info[1].args[0])
            sys.exit()

    def scanOptionChains(self):
        chainList = []
        self.investmentSheet = investments()
        self.potentialCallSymbols = self.investmentSheet.potentialCalls()
        for ndx in range(len(self.potentialCallSymbols)):
            symbol = self.potentialCallSymbols[ndx]
            options = OptionChain(symbol=symbol, strategy="Call", strikeCount=5, strikeRange="OTM", daysToExpiration=60)
            print("Call option details: symbol - {}, strategy - {}".format(options.symbol, options.strategy))
            #self.potentialOptions.append(options)
            chainList.append(options)

        self.potentialPutSymbols = self.investmentSheet.potentialPuts()
        for ndx in range(len(self.potentialPutSymbols)):
            symbol = self.potentialPutSymbols[ndx]
            options = OptionChain(symbol=symbol, strategy="Put", strikeCount=5, strikeRange="OTM", daysToExpiration=60)
            print("Put option details: symbol - {}, strategy - {}".format(options.symbol, options.strategy))
            #self.potentialOptions.append(options)
            chainList.append(options)
            
        optionList = []
        for ndx in range(len(chainList)):
            for optndx in range(len(chainList[ndx].df_OptionChain)):
                optionList.append(chainList[ndx].df_OptionChain.iloc[optndx])
                
        self.potentialOptions=pd.DataFrame(optionList)
        #print("Option chain dataframe:\n{}".format(self.potentialOptions))
        
        return
    
    def filterDelta(self, threshold):
        remainingOptions = []

        for ndx in range (len(self.filteredOptions)):
            optKey = self.filteredOptions.iloc[ndx].name
            optDetails = self.filteredOptions.iloc[ndx]
            optOTMPct = 1-abs(optDetails["delta"])
                
            if optOTMPct <= threshold:
                #print("Eliminating {} - will expire OTM {}%".format(optKey, optOTMPct) )
                pass
            else:
                #print("Retaining {} - will expire OTM {}%".format(optKey, optOTMPct) )
                remainingOptions.append(self.filteredOptions.iloc[ndx])
                    
        self.filteredOptions = pd.DataFrame(remainingOptions)
        
        return
    
    def filterMaxCover(self, threshold):
        remainingOptions = []

        for ndx in range (len(self.filteredOptions)):
            optKey = self.filteredOptions.iloc[ndx].name
            #optDetails = self.filteredOptions.iloc[ndx]
            symbol = optKey[0]
            price = self.investmentSheet.stockInformationTab.xs(symbol, level='Symbol').iloc[0]['Price']
            price = float(price.replace('$', ''))
            
            if (symbol in self.potentialPutSymbols):
                self.symbolDetails.loc[symbol, "maxTradeQty"] = threshold // (price * 100)
                if (price * 100) > threshold:
                    #print("Eliminating {} - cash required to cover {} exceeds filter limit {}".format(optKey, (price * 100), threshold) )
                    pass
                else:
                    #print("Retaining {}".format(optKey) )
                    remainingOptions.append(self.filteredOptions.iloc[ndx])
            else:
                #print("Retaining {}".format(optKey) )
                remainingOptions.append(self.filteredOptions.iloc[ndx])
                
        self.filteredOptions = pd.DataFrame(remainingOptions)
        
        return
    
    def filterMinGainAPY(self, threshold):
        print("filterMinGainAPY is WIP")
        remainingOptions = []

        for ndx in range (len(self.filteredOptions)):
            optKey = self.filteredOptions.iloc[ndx].name
            optDetails = self.filteredOptions.iloc[ndx]
            filterFieldVal = int(optDetails["days To Expiration"])
                
            if filterFieldVal <= threshold:
                #print("Eliminating {} - {}".format(optKey, filterFieldVal) )
                pass
            else:
                #print("Retaining {} - days to expiration {}".format(optKey, filterFieldVal) )
                remainingOptions.append(self.filteredOptions.iloc[ndx])
                    
        self.filteredOptions = pd.DataFrame(remainingOptions)
        
        return
    
    def filterPotentialOptionTrades(self):
        print("filterPotentialOptionTrades is WIP =======================")
        self.filteredOptions = self.potentialOptions
        
        for fltndx in range(len(self.filterList)):
            #print("filter definition: {}".format(self.filterList[fltndx]))
            
            if  self.filterList[fltndx]["dataElement"] == "delta":
                if self.filterList[fltndx]["condition"] == "GT":
                    self.filterDelta(float(self.filterList[fltndx]["threshold"]))
                else:
                    print("delta filtering is only supported with the condition GT")
        
            if  self.filterList[fltndx]["dataElement"] == "max cover":
                if self.filterList[fltndx]["condition"] == "LT":
                    self.filterMaxCover(float(self.filterList[fltndx]["threshold"]))
                else:
                    print("max cover filtering is only supported with the condition LT")
        
            if  self.filterList[fltndx]["dataElement"] == "option quantity":
                if self.filterList[fltndx]["condition"] == "GT":
                    print("option quantity filtering is WIP")
                else:
                    print("option quantity filtering is only supported with the condition GT")
        
            if  self.filterList[fltndx]["dataElement"] == "min gain APY":
                if self.filterList[fltndx]["condition"] == "GT":
                    self.filterMinGainAPY(float(self.filterList[fltndx]["threshold"]))
                else:
                    print("min gain APY filtering is only supported with the condition GT")
        
            if  self.filterList[fltndx]["dataElement"] == "min gain $":
                if self.filterList[fltndx]["condition"] == "GT":
                    print("min gain $ filtering is WIP")
                else:
                    print("min gain $ filtering is only supported with the condition GT")
        
            if  self.filterList[fltndx]["dataElement"] == "dividend date":
                if self.filterList[fltndx]["condition"] == "LT":
                    print("dividend date filtering is WIP")
                else:
                    print("dividend date filtering is only supported with the condition LT")
        
            if  self.filterList[fltndx]["dataElement"] == "earnings date":
                if self.filterList[fltndx]["condition"] == "LT":
                    print("earnings date filtering is WIP")
                else:
                    print("earnings date filtering is only supported with the condition LT")
        
            print("Option chain after filter {}\n{}".format(self.filterList[fltndx], self.filteredOptions))

        return
    
    def calculateRelatedTradeDetails(self):
        print("calculate related trade details is WIP =======================")
        
        self.investmentSheet = investments()
        #self.accountsTabDetails = self.investmentSheet.readAccountTab()

        symbols = set(self.potentialOptions.index.get_level_values(level=0))
        symbols = list(symbols)
        holdings = self.investmentSheet.holdings()
        potentialBuys = self.investmentSheet.potentialBuys()
        
        accountDetails = []
        print("Calculating maximum trade quantity")
        print("Calculating Maximum gain APY")
        print("Determining next dividend date")
        print("Determining next earnings date")
        for symbol in symbols:
            price = self.investmentSheet.stockInformationTab.xs(symbol, level='Symbol').iloc[0]['Price']
            price = float(price.replace('$', ''))
            
            if symbol in holdings:
                shares = int(self.investmentSheet.accountsTab.xs(symbol, level='Asset').iloc[0]["Equity Position"])
                maxTradeQty = shares // 100
            elif symbol in potentialBuys:
                shares = 0
                maxTradeQty = 0
            
            maxGainAPY = 0.0
            maxProfit = 0.0
            nextDividendDate = "WIP TBD"
            nextEarningDate = "WIP TBD"
            maxCashtoCover = 0.0
            
            newRow = {"symbol":symbol, \
                      "price" : price, \
                      "holding" : shares,
                      "maxTradeQty" : maxTradeQty, \
                      "maxGainAPY" : maxGainAPY, \
                      "maxProfit" : maxProfit, \
                      "nextDividendDate" : nextDividendDate, \
                      "nextEarningDate" : nextEarningDate
                      }
            accountDetails.append(newRow)

        self.symbolDetails = pd.DataFrame(accountDetails)
        self.symbolDetails.set_index(['symbol'], inplace=True)
        print("Related filter values\n{}".format(self.symbolDetails))
        '''
        Trade quantity - depends on number of shares held (multiples of 100)
        Maximum gain APY
        Maximum profit potential
        Dividend pay date before expiration
        Earnings announcement before expiration
        '''
        return 
    
    def findPotentialOptionTrades(self, strikeCount=5, strikeRange="OTM", daysToExpiration=60, filterList=[]):
        #print("filterOptionsChains is WIP\n\t{}".format(filterList))
        
        self.strikeCount = strikeCount
        self.strikeRange = strikeRange
        self.daysToExpiration = daysToExpiration
        self.filterList = filterList
        
        self.scanOptionChains()
        self.calculateRelatedTradeDetails()
        self.filterPotentialOptionTrades()
        self.outputPotentalTrades()
        
        ''' tab column headers
        Symbol
        Strategy    
        Expiration    
        Days to Expiration    
        Share price    
        Closing Price    
        Strike Price    
        Break Even    
        Bid    
        Ask    
        OTM Probability        based on delta. 1-abs(delta) = OTM probability TBD
        volatility    
        ADX (Price trend est.)    
        Probability of Loss    
        Purchase Price    
        Earnings    
        Dividend    
        Current Holding    
        Qty    
        Max Gain APY    
        Max Profit    
        Risk Management    
        Loss vs. profit    
        premium    
        commission    
        Earnings Date    
        dividend Date    
        delta            Calls have positive deltas; puts have negative deltas
        gamma    
        theta    
        vega    
        rho    
        in The Money    
        expiration Date    
        ROI    
        Max Loss    
        Preferred outcome    
        Preferred Result    
        Unfavored Result
        '''

    ''' ======================== Accessible data elements ============================== '''
    ''' financial asset symbols '''
    @property
    def potentialPutSymbols(self):
        return self._potentialPutSymbols
    
    @potentialPutSymbols.setter
    def potentialPutSymbols(self, potentialPutSymbols):
        self._potentialPutSymbols = potentialPutSymbols

    @property
    def filteredPutSymbols(self):
        return self._filteredPutSymbols
    
    @filteredPutSymbols.setter
    def filteredPutSymbols(self, filteredPutSymbols):
        self._filteredPutSymbols = filteredPutSymbols

    @property
    def potentialCallSymbols(self):
        return self._potentialCallSymbols
    
    @potentialCallSymbols.setter
    def potentialCallSymbols(self, potentialCallSymbols):
        self._potentialCallSymbols = potentialCallSymbols

    @property
    def filteredCallSymbols(self):
        return self._filteredCallSymbols
    
    @filteredCallSymbols.setter
    def filteredCallSymbols(self, filteredCallSymbols):
        self._filteredCallSymbols = filteredCallSymbols

    ''' option chain characteristics '''
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

    ''' financial asset option chains '''
    @property
    def potentialOptions(self):
        return self._potentialOptions
    
    @potentialOptions.setter
    def potentialOptions(self, potentialOptions):
        self._potentialOptions = potentialOptions

    @property
    def filteredOptions(self):
        return self._filteredOptions
    
    @filteredOptions.setter
    def filteredOptions(self, filteredOptions):
        self._filteredOptions = filteredOptions

    @property
    def filterList(self):
        return self._filterList
    
    @filterList.setter
    def filterList(self, filterList):
        self._filterList = filterList
