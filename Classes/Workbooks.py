'''
Created on Sep 17, 2024

@author: brian
'''
import sys
import datetime as dt

import pandas as pd
import keras

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
    
    def stockInformationSymbols(self):
        try:
            exc_txt = "\nAn exception occurred - unable to look up symbols on Stock Information tab"
            symbolList = []
            cellRange = 'Stock Information!A2:AF999'
            symbolInformation = self.gSheets.readGoogleSheet(self.sheetID, cellRange, headerRows=1)
            print("Stock Information:\n{}".format(symbolInformation))
            for rowNdx in range(len(symbolInformation)):
                symbol = symbolInformation.loc[rowNdx, 'Symbol']
                symbolList.append(symbol)        
            return symbolList

        except :
            print(exc_txt)
            exc_info = sys.exc_info()
            if len(exc_info) > 1:
                print(exc_info[1].args[0])
            sys.exit()

    def markToMarket(self):
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
            symbolInformation = self.gSheets.readGoogleSheet(self.sheetID, cellRange, headerRows=1)
            print("Stock Information:\n{}".format(symbolInformation))
            
            ''' Step 3 - create a dataframe '''
            indexLabel = "Symbol"
            columnLabels = ["Symbol", "Last Price", "Dividend Yield", "Dividend", "Dividend Ex-Date", "P/E Ratio", \
                            "52 Week High", "52 Week Low", "Volume", "Shares Outstanding", "Market Cap" ]
                        
            self.df_marktomarket = pd.DataFrame(columns=columnLabels)
            self.df_marktomarket.set_index(indexLabel, drop=True, inplace=True)
            
            ''' Step 4 - request the basic financial instrument and the latest market data '''
            exc_txt = "\nAn exception occurred - unable to obtain market information on symbols"
            symbolCount = len(symbolInformation)
            tf_progbar = keras.utils.Progbar(symbolCount, width=50, verbose=1, interval=50, stateful_metrics=None, unit_name='symbol')
            for rowNdx in range(symbolCount):
                tf_progbar.update(rowNdx)
                symbol = symbolInformation.loc[rowNdx, 'Symbol']
                exc_txt = "\nAn exception occurred - with symbol: {}".format(symbol)
            
                instrumentDetails = FinancialInstrument(symbol)
                if instrumentDetails.detailsRetrieved:
                    mktData = MarketData(symbol, periodType="month", period="1", frequencyType="daily", frequency="1")
                    if mktData.candleCount() > 0:
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
                else:
                    print("Unable to retrieve financial and market data for {}".format(symbol))
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
            cellRange = 'Accounts!A2:AU175'
            accountsTab = self.gSheets.readGoogleSheet(self.sheetID, cellRange, headerRows=1)
        
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
            symbolInformation = self.gSheets.readGoogleSheet(self.sheetID, cellRange, headerRows=1)

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
            symbolList = []
            exc_txt = "Exception occurred returning holdings list"
            
            for rowNdx in range(len(self.stockInformationTab)):
                actionCategory = self.stockInformationTab.iloc[rowNdx].name[0]
                holding = True
                if (actionCategory == "10 - Holding" or actionCategory == "12 - Call Options") and \
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
            symbolList = []
            exc_txt = "Exception occurred returning list of potential buys"
            
            for rowNdx in range(len(self.stockInformationTab)):
                actionCategory = self.stockInformationTab.iloc[rowNdx].name[0]
                if actionCategory == "45 - Buy" or actionCategory == "41 - Put Options":
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
                if actionCategory == "12 - Call Options":
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
                if actionCategory == "41 - Put Options":
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
        
        self.OPTIONINDEX = ['Symbol', 'Strategy', 'Expiration Date', 'Strike Price']
        self.investmentSheet = None
        self.symbolDetails = []
        self.potentialPutSymbols = []
        self.filteredPutSymbols = []
        self.potentialCallSymbols = []
        self.filteredCallSymbols = []

        self.potentialOptions = []
        self.filteredOptions = []

        return
        
    def calcDiscountYield(self, settlement, maturity, currentValue, maturityValue):
        '''
        YIELDDISC(settlement, maturity, price, redemption, [day_count_convention])
        Args:
        settlement - The settlement date of the security, the date after issuance when the security is delivered to the buyer.
        maturity - The maturity or end date of the security, when it can be redeemed at face, or par value.
        price - The price at which the security is bought.
        redemption - The redemption value of the security.

        Returns:
        The discounted yield of the asset.
        '''
        try:
            exc_txt = "Exception occurred calculating discount yield"
            '''
            Discount yield = ( (Par Value - Purchase Price) / Par Value ) * (360 / Days to Maturity)
            '''
            period = maturity - settlement
            daysToMaturity = period.days
            if maturityValue > 0 and daysToMaturity > 0:
                discountYield = ( (maturityValue - currentValue) / maturityValue ) * (360 / daysToMaturity)
            else:
                discountYield = 0.0
            
            return discountYield

        except :
            print(exc_txt)
            exc_info = sys.exc_info()
            if len(exc_info) > 1:
                print(exc_info[1].args[0])
            sys.exit()

    def outputPotentalTrades(self):
        try:
            ''' Output the potential option trades to Google sheet '''
            exc_txt = "Exception occurred writing the list of potential option trades"
            if len(self.filteredOptions) > 0:
                print("Options for review\n{}".format(self.filteredOptions))
                
                ''' create new tab '''
                now = dt.datetime.now()
                sheetName = '{:4d}{:0>2d}{:0>2d}'.format(now.year, now.month, now.day)
                self.optionsSheetID
                self.gSheets.addGoogleSheet(self.optionsSheetID, sheetName)
                
                ''' write data to new tab '''
                self.filteredOptions.index.names = ['Symbol', 'Strategy', 'Expiration Date', 'Strike Price']
                self.filteredOptions = self.filteredOptions.reset_index()
                
                # select and reorder the columns
                outputColumns = ['Symbol', 'Strategy', 'Symbol Price', \
                                'Expiration Date', 'days To Expiration', 'Strike Price', \
                                'Max Profit', 'Max Gain APY', \
                                'bid', 'ask', 'close Price', \
                                'Option Qty', 'OTM Probability', \
                                'Premium', 'Commission']
                googleTab = self.filteredOptions[outputColumns]
                self.gSheets.updateGoogleSheet(self.optionsSheetID, sheetName + "!A1:Z999", googleTab.columns)
                self.gSheets.updateGoogleSheet(self.optionsSheetID, sheetName + "!A2:Z999", googleTab)
            else:
                print("No options to review")
            
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
            options = OptionChain(symbol=symbol, strategy="Call", \
                                  strikeCount=self.strikeCount, strikeRange=self.strikeRange, daysToExpiration=self.daysToExpiration)
            print("Call option details: symbol - {}, strategy - {}".format(options.symbol, options.strategy))
            #self.potentialOptions.append(options)
            chainList.append(options)

        self.potentialPutSymbols = self.investmentSheet.potentialPuts()
        for ndx in range(len(self.potentialPutSymbols)):
            symbol = self.potentialPutSymbols[ndx]
            options = OptionChain(symbol=symbol, strategy="Put", \
                                  strikeCount=self.strikeCount, strikeRange=self.strikeRange, daysToExpiration=self.daysToExpiration)
            print("Put option details: symbol - {}, strategy - {}".format(options.symbol, options.strategy))
            #self.potentialOptions.append(options)
            chainList.append(options)
            
        optionList = []
        for ndx in range(len(chainList)):
            for optndx in range(len(chainList[ndx].df_OptionChain)):
                optionList.append(chainList[ndx].df_OptionChain.iloc[optndx])
                
        self.potentialOptions=pd.DataFrame(optionList)
        self.potentialOptions.sort_index(inplace=True)
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
        remainingOptions = []

        for ndx in range (len(self.filteredOptions)):
            optKey = self.filteredOptions.iloc[ndx].name
            optDetails = self.filteredOptions.iloc[ndx]
            filterFieldVal = float(optDetails["Max Gain APY"])
                
            if filterFieldVal <= threshold:
                #print("Eliminating {} - {}".format(optKey, filterFieldVal) )
                pass
            else:
                #print("Retaining {} - days to expiration {}".format(optKey, filterFieldVal) )
                remainingOptions.append(self.filteredOptions.iloc[ndx])
                    
        self.filteredOptions = pd.DataFrame(remainingOptions)
        
        return
    
    def filterMinGain(self, threshold):
        remainingOptions = []

        for ndx in range (len(self.filteredOptions)):
            optKey = self.filteredOptions.iloc[ndx].name
            optDetails = self.filteredOptions.iloc[ndx]
            filterFieldVal = float(optDetails["Max Profit"])
                
            if filterFieldVal <= threshold:
                #print("Eliminating {} - {}".format(optKey, filterFieldVal) )
                pass
            else:
                #print("Retaining {} - days to expiration {}".format(optKey, filterFieldVal) )
                remainingOptions.append(self.filteredOptions.iloc[ndx])
                    
        self.filteredOptions = pd.DataFrame(remainingOptions)
        
        return
    
    def filterDividendDate(self, threshold):
        remainingOptions = []

        for ndx in range (len(self.filteredOptions)):
            optKey = self.filteredOptions.iloc[ndx].name
            symbol = optKey[0]
            expDate = optKey[2]
            #optDetails = self.filteredOptions.iloc[ndx]

            exDividendDate = self.investmentSheet.stockInformationTab.xs(symbol, level='Symbol').iloc[0]['Ex Div']
            if len(exDividendDate) > 0:
                thresholdExDividend = dt.datetime.strptime(exDividendDate, '%m/%d/%Y').date()
                expirationDate = dt.date.fromisoformat(expDate)
            
                if thresholdExDividend <= expirationDate:
                    #print("Eliminating {} - {}".format(optKey, filterFieldVal) )
                    pass
                else:
                    #print("Retaining {} - days to expiration {}".format(optKey, filterFieldVal) )
                    remainingOptions.append(self.filteredOptions.iloc[ndx])
            else:
                #print("Retaining {} - days to expiration {}".format(optKey, filterFieldVal) )
                remainingOptions.append(self.filteredOptions.iloc[ndx])
                    
        self.filteredOptions = pd.DataFrame(remainingOptions)
        
        return
    
    def filterLimitPrice(self, threshold):
        remainingOptions = []

        for ndx in range (len(self.filteredOptions)):
            optKey = self.filteredOptions.iloc[ndx].name
            symbol = optKey[0]
            strategy = optKey[1]
            strikePrice = float(optKey[3])
    
            limitPrice = self.investmentSheet.stockInformationTab.xs(symbol, level='Symbol').iloc[0]['Limit Price']
            limitPrice = float(limitPrice.replace('$', ''))
            if strategy == 'put':
                if strikePrice < limitPrice:
                    remainingOptions.append(self.filteredOptions.iloc[ndx])
            else:
                if strikePrice > limitPrice:
                    remainingOptions.append(self.filteredOptions.iloc[ndx])
                               
        self.filteredOptions = pd.DataFrame(remainingOptions)

        return
    
    def filterPotentialOptionTrades(self):
        self.filteredOptions = self.potentialOptions
        
        for fltndx in range(len(self.filterList)):
            #print("filter definition: {}".format(self.filterList[fltndx]))
            
            if  self.filterList[fltndx]["dataElement"] == "delta":
                if self.filterList[fltndx]["condition"] == "GT":
                    self.filterDelta(float(self.filterList[fltndx]["threshold"]))
                else:
                    print("delta filtering is only supported with the condition GT")
        
            elif  self.filterList[fltndx]["dataElement"] == "max cover":
                if self.filterList[fltndx]["condition"] == "LT":
                    self.filterMaxCover(float(self.filterList[fltndx]["threshold"]))
                else:
                    print("max cover filtering is only supported with the condition LT")
        
            elif  self.filterList[fltndx]["dataElement"] == "option quantity":
                if self.filterList[fltndx]["condition"] == "GT":
                    print("option quantity filtering is WIP")
                else:
                    print("option quantity filtering is only supported with the condition GT")
        
            elif  self.filterList[fltndx]["dataElement"] == "min gain APY":
                if self.filterList[fltndx]["condition"] == "GT":
                    self.filterMinGainAPY(float(self.filterList[fltndx]["threshold"]))
                else:
                    print("min gain APY filtering is only supported with the condition GT")
        
            elif  self.filterList[fltndx]["dataElement"] == "min gain $":
                if self.filterList[fltndx]["condition"] == "GT":
                    self.filterMinGain(float(self.filterList[fltndx]["threshold"]))
                else:
                    print("min gain $ filtering is only supported with the condition GT")
        
            elif  self.filterList[fltndx]["dataElement"] == "dividend date":
                if self.filterList[fltndx]["condition"] == "GT":
                    self.filterDividendDate(self.filterList[fltndx]["threshold"])
                else:
                    print("dividend date filtering is only supported with the condition expiration date earlier than dividend date")
        
            elif  self.filterList[fltndx]["dataElement"] == "earnings date":
                if self.filterList[fltndx]["condition"] == "GT":
                    print("earnings date filtering is not currently supported")
                else:
                    print("earnings date filtering is only supported with the condition GT")
                    
            elif  self.filterList[fltndx]["dataElement"] == "limit price":
                self.filterLimitPrice(self.filterList[fltndx]["threshold"])
        
            else:
                print("filter control -{}- is not supported".format(self.filterList[fltndx]["dataElement"]))
        
        #print("Options after filtering {}\n{}".format(self.filterList[fltndx], self.filteredOptions))

        return
    
    def calculateOptionSpecificTradeDetails(self):
        self.potentialOptions['Premium'] = ''
        self.potentialOptions['Max Gain APY'] = ''
        self.potentialOptions['Max Profit'] = ''
        self.potentialOptions['Risk Management'] = ''
        self.potentialOptions["OTM Probability"] = ""
        settlementDate = dt.date.today()
        
        for ndx in range (len(self.potentialOptions)):
            optKey = self.potentialOptions.iloc[ndx].name
            optDetails = self.potentialOptions.iloc[ndx]
            symbol = optKey[0]
            strategy = optKey[1]

            strikePrice = optKey[3]
            strikePrice = float(strikePrice.replace('$', ''))

            expirationDate = optKey[2]
            #date_format = "%y-%m-%d"
            expirationDate = dt.date.fromisoformat(expirationDate)
            
            bid = optDetails["bid"]
            ask = optDetails["ask"]
            commission = self.symbolDetails.loc[symbol, "commission"]
            
            self.potentialOptions.loc[optKey, "Option Qty"] = self.symbolDetails.loc[symbol, "maxTradeQty"]
            premium = self.symbolDetails.loc[symbol, "maxTradeQty"] * 100 * bid
            maxProfit = premium - commission

            self.potentialOptions.loc[optKey, "Symbol Price"] = self.symbolDetails.loc[symbol, "price"]
            self.potentialOptions.loc[optKey, "OTM Probability"] = 1-abs(optDetails["delta"])
            self.potentialOptions.loc[optKey, "Commission"] = commission
            self.potentialOptions.loc[optKey, "Premium"] = premium
            self.potentialOptions.loc[optKey, "Max Profit"] = maxProfit
            
            if strategy == "put":
                riskManagement = strikePrice * (self.symbolDetails.loc[symbol, "maxTradeQty"] * 100)
                currentValue = riskManagement
                valueAtExpiration = riskManagement + maxProfit
            else:
                riskManagement="Current holding"
                symbolInformation = self.investmentSheet.stockInformationTab.xs(symbol, level='Symbol')
                price = symbolInformation.iloc[0]['Price']
                #price = self.investmentSheet.stockInformationTab.xs(symbol, level='Symbol').iloc[0]['Price']
                price = (price.replace('$', ''))
                price = (price.replace(',', ''))
                price = float(price)
                #riskManagementAmount = self.symbolDetails.loc[symbol, "holdingValue"]
                riskManagementAmount = price * (self.symbolDetails.loc[symbol, "maxTradeQty"] * 100)
                currentValue = riskManagementAmount
                valueAtExpiration = riskManagementAmount + maxProfit
                
            self.potentialOptions.loc[optKey, "Risk Management"] = riskManagement
            maxGainAPY = self.calcDiscountYield(settlementDate, expirationDate, currentValue, valueAtExpiration)
            self.potentialOptions.loc[optKey, "Max Gain APY"] = maxGainAPY
            
        return 
    
    def calculateSymbolRelatedTradeDetails(self):
        self.investmentSheet = investments()
        #self.accountsTabDetails = self.investmentSheet.readAccountTab()

        symbols = set(self.potentialOptions.index.get_level_values(level=0))
        symbols = list(symbols)
        holdings = self.investmentSheet.holdings()
        potentialBuys = self.investmentSheet.potentialBuys()
        
        accountDetails = []
        for symbol in symbols:
            symbolInformation = self.investmentSheet.stockInformationTab.xs(symbol, level='Symbol')
            price = symbolInformation.iloc[0]['Price']
            #price = self.investmentSheet.stockInformationTab.xs(symbol, level='Symbol').iloc[0]['Price']
            price = (price.replace('$', ''))
            price = (price.replace(',', ''))
            price = float(price)
            
            if symbol in holdings:
                symbolAccountInformation = self.investmentSheet.accountsTab.xs(symbol, level='Asset')
                if len(symbolAccountInformation) > 1:
                    print("Note: {} is held in {} accounts".format(symbol, len(symbolAccountInformation)))
                shares = symbolAccountInformation.iloc[0]["Equity Position"]
                shares = (shares.replace(',', ''))
                shares = float(shares)
                #shares = int(self.investmentSheet.accountsTab.xs(symbol, level='Asset').iloc[0]["Equity Position"])
                holdingValue = shares * price
            elif symbol in potentialBuys:
                maxInvestment = 30000
                for fltndx in range(len(self.filterList)):
                    if self.filterList[fltndx]["dataElement"] == "max cover":
                        maxInvestment = float(self.filterList[fltndx]["threshold"])
                    
                shares = maxInvestment / price
                holdingValue = 0.0
                
            maxTradeQty = shares // 100
            commission = maxTradeQty * 0.65
            nextDividendDate = symbolInformation.iloc[0]['Ex Div']
            
            newRow = {"symbol" : symbol, \
                      "shares" : shares, \
                      "price" : price, \
                      "holdingValue" : holdingValue, \
                      "maxTradeQty" : maxTradeQty, \
                      "commission" : commission, \
                      "nextDividendDate" : nextDividendDate
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
        self.calculateSymbolRelatedTradeDetails()
        self.calculateOptionSpecificTradeDetails()
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


class mlEvaluations(workbooks):
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
        self.googleDriveFiles = read_config_json(aiwork + "\\" + googleAuth['fileIDs'])

        print("file 1: {} - {}".format('development', self.googleDriveFiles["Google IDs"]["ML Model Predictions"]["Development"]))
        print("file 2: {} - {}".format('production', self.googleDriveFiles["Google IDs"]["ML Model Predictions"]["Production"]))
        '''
        '''
        
        ''' ================ Authenticate with Google workplace and establish a connection to Google Drive API ============== '''
        super().__init__()
        
        self.SIGNALINDEX = ['SignalID', 'Symbol', 'Date']

        return
    
    def archiveSignals(self, signals, outputs):
        exc_txt = "\nAn exception occurred - unable to archive signals to Google sheet"
        try:
            print("archiving {} signals".format(len(signals)))
            now = dt.datetime.now()
            evaluationDate = dt.date.today()
            signalsDate = '{:0>2d}/{:0>2d}/{:4d}'.format(now.month, now.day, now.year)
            
            print("signals assessed on {}".format(signalsDate))
            for signal in signals:
                print("signal: {}".format(signal))
            
            if self.sheetID == 'Development':
                self.sheetID = self.googleDriveFiles["Google IDs"]["ML Model Predictions"]["Development"]
            else:
                self.sheetID = self.googleDriveFiles["Google IDs"]["ML Model Predictions"]["Production"]

            ''' load previous signal archive '''
            self.signalsCols = self.gSheets.readGoogleSheet(self.sheetID, self.headerRange, headerRows=1)
            self.signalsRows = self.gSheets.readGoogleSheet(self.sheetID, self.dataRange, headerRows=0)
            self.signalsRows.columns = self.signalsCols.columns
            
            headers = ["model", "symbol", "date"]
            headers.extend(outputs)
            self.signalsCols = headers
            
            print("Column headers:\n{}".format(self.signalsCols))
            print("Existing data rows:\n{}".format(self.signalsRows))
            print("WIP ===============\n\tadd a parameter to readGoogleSheet to indicate whether a header row is included or not")
            
            for signal in signals:
                newRow = []
                newRow.append(signal["name"])
                newRow.append(signal['symbol'])
                newRow.append(signalsDate)

                for prediction in signal["prediction"]:
                    newRow.append(prediction)

                self.signalsRows.loc[len(self.signalsRows)] = newRow
                
            ''' save updated archive '''
            self.signalsCols = pd.DataFrame(self.signalsCols)
            self.gSheets.updateGoogleSheet(self.sheetID, self.headerRange, self.signalsCols[0])
            self.gSheets.updateGoogleSheet(self.sheetID, self.dataRange, self.signalsRows)
            
            return
    
        except :
            print(exc_txt)
            exc_info = sys.exc_info()
            if len(exc_info) > 1:
                print(exc_info[1].args[0])
            sys.exit()

    @property
    def processName(self):
        return self._processName
    
    @processName.setter
    def processName(self, processName):
        self._processName = processName


    @property
    def sheetID(self):
        return self._sheetID
    
    @sheetID.setter
    def sheetID(self, sheetID):
        self._sheetID = sheetID

    @property
    def headerRange(self):
        return self._headerRange
    
    @headerRange.setter
    def headerRange(self, headerRange):
        self._headerRange = headerRange

    @property
    def dataRange(self):
        return self._dataRange
    
    @dataRange.setter
    def dataRange(self, dataRange):
        self._dataRange = dataRange

    @property
    def features(self):
        return self._features
    
    @features.setter
    def features(self, features):
        self._features = features

