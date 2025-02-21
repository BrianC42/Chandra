'''
Created on Jul 13, 2020

@author: Brian
'''
import re
import sys
import datetime as dt
from datetime import date
from datetime import timedelta

import numpy as np
from numpy import NaN
import pandas as pd

from configuration import get_ini_data
from moving_average import simple_moving_average
from moving_average import exponential_moving_average
from tda_api_library import format_tda_datetime

import GoogleSheets as gs


def marketOprionsPopulate(row, col, dfHoldings, dfMarketData):
    '''
        mktOptions['Purchase $'] = mktOptions.apply(marketOprionsPopulate, axis=1, args=('Purchase $', dfHoldings, dfMarketData))
        mktOptions['Earnings Date'] = mktOptions.apply(marketOprionsPopulate, axis=1, args=('Earnings Date', dfHoldings, dfMarketData))
        mktOptions['Dividend Date'] = mktOptions.apply(marketOprionsPopulate, axis=1, args=('Dividend Date', dfHoldings, dfMarketData))
        mktOptions['Current Holding'] = mktOptions.apply(marketOprionsPopulate, axis=1, args=('Current Holding', dfHoldings, dfMarketData))
    '''
    exc_txt = "Unable to determine purchase price"
    
    try:
        symbol = row['symbol']
        if symbol in dfHoldings.index:
            if col == 'Purchase $':
                value = dfHoldings.loc[symbol]['Purchase $']
            elif col == 'Current Holding':
                value = dfHoldings.loc[symbol]['Current Holding']
            elif col == 'Earnings Date':
                value = dfMarketData.loc[symbol]['Next Earnings Date']
            elif col == 'Dividend Date':
                value = dfMarketData.loc[symbol]['Dividend Ex-Date']
        else:
            if col == 'Purchase $' or col == 'Current Holding':
                value = 0.0
            if col == 'Earnings Date' or col == 'Dividend Date':
                yyyy = dt.MINYEAR
                mm = 1
                dd = 1
                sheetDate = date(yyyy, mm, dd)
                value = sheetDate

    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        sys.exit(exc_txt + "\n\t" + exc_str)

    return  value

def dividendRisk(row):
    exc_txt = "Unable to determine risk due to dividend date"
    
    try:
        now = date.today()
        if now > row['Dividend Date']:
            riskStr = "2 - Div Past"
        elif row['expiration'] > row['Dividend Date']:            
            riskStr = "9 - Dividend"
        else:
            riskStr = "3 - TBD"

    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        sys.exit(exc_txt + "\n\t" + exc_str)

    return  riskStr

def earningsRisk(row):
    exc_txt = "Unable to determine risk due to dividend date"
    
    try:
        now = date.today()
        if now > row['Earnings Date']:
            riskStr = "1 - Earnings Past"
        elif row['expiration'] <= row['Earnings Date']:            
            riskStr = "2 - No earnings"
        elif row['expiration'] > row['Earnings Date']:            
            riskStr = "9 - Earnings"
        else:
            riskStr = "3 - Unknown"

    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        sys.exit(exc_txt + "\n\t" + exc_str)

    return  riskStr

def optionQty(row):
    exc_txt = "Unable to determine qty of options to trade"
    
    try:
        if row['strategy'] == "Covered call":
            optQty = int(row['Current Holding']) / 100
        elif row['strategy'] == "Cash Secured Put":
            optQty = int(25000 / (row['underlying Price'] * 100))
        else:
            optQty = 0

    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        sys.exit(exc_txt + "\n\t" + exc_str)

    return  optQty

def annual_yield(discount, face_value, days_to_maturity):
    """
      Calculates the annual yield of a discount security.
    
      Args:
        discount: The discount on the security.
        face_value: The face value of the security.
        days_to_maturity: The number of days until the security matures.
    
      Returns:
        The annual yield of the security.
    """
    
    annual_yield = (discount / face_value) * (365 / days_to_maturity) * 100
    
    return annual_yield

def maxGainApy(row):
    exc_txt = "Unable to determine maximum gain APY of options if traded"
    try:
        now = date.today()
        exp = row['expiration']
        dtm = timedelta()
        dtm = exp - now
        gain = row['Max Profit']
        if row['Risk Management'] == "Current holding":
            ''' yielddisc(today(),C{},((S{}*100)*E{}),((S{}*100)*E{})+U{})
            YIELDDISC(
                settlement,      now
                maturity,        exp
                price,           s * e * 100
                redemption,      (s * e * 100) + u
                [day_count_convention])
            YIELDDISC(DATE(2010,01,02),DATE(2010,12,31),98.45,100)
            col c = row['expiration']
            col s = row['Qty']
            col e = row['underlying Price']
            col u = row['Max Profit']
            '''
            futureVal = (row['Qty'] * 100 * row['underlying Price']) + row['Max Profit']
        else:
            ''' yielddisc(today(),C{},V{},V{}+U{}))
            col c = row['expiration']
            col v = row['Risk Management']
            col u = row['Max Profit']
            '''
            futureVal = row['Risk Management'] + row['Max Profit']

        if gain == 0.0 or futureVal == 0.0 or dtm.days == 0:
            maxGainApy = 0.0
        else:
            maxGainApy = annual_yield(gain, futureVal, dtm.days)

    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        sys.exit(exc_txt + "\n\t" + exc_str)

    return  maxGainApy

def riskManagement(row):
    exc_txt = "Unable to determine risk management of options if traded"
    
    try:
        if row['strategy'] == "Covered call":
            riskMgmt = "Current holding"
        elif row['strategy'] == "Cash Secured Put":            
            #riskMgmt = "${}".format(row['Qty'] * 100 * row['strike Price'])
            riskMgmt = row['Qty'] * 100 * row['strike Price']
        else:
            riskMgmt = "TBD"

    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        sys.exit(exc_txt + "\n\t" + exc_str)

    return  riskMgmt

def maxProfit(row):
    exc_txt = "Unable to calculate max profit"    
    try:
        profit = float(row['premium']) - float(row['commission'])

    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        sys.exit(exc_txt + "\n\t" + exc_str)

    return  profit

def lossVProfit(row):
    exc_txt = "Unable to calculate loss vs. profit"
    try:
        lvp = row['OTM Probability'] - row['probability of loss']

    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        sys.exit(exc_txt + "\n\t" + exc_str)

    return  lvp

def premium(row):
    exc_txt = "Unable to calculate premium"
    try:
        prem = row['Qty'] * 100 *  row['bid']

    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        sys.exit(exc_txt + "\n\t" + exc_str)

    return  prem

def commission(row):
    exc_txt = "Unable to calculate commission"
    try:
        com = row['Qty'] * 0.65

    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        sys.exit(exc_txt + "\n\t" + exc_str)

    return  com

def seperateSymbol(value):
    try:
        subStrs = re.split(',', value)
        return subStrs[0]

    except ValueError:
        return np.nan
    
def loadHoldings(sheet, sheetID, readRange):
    
    result = sheet.googleSheet.values().get(spreadsheetId=sheetID, range=readRange).execute()
    values = result.get('values', [])
    if not values:
        print('\tNo data found.')

    dfHoldings = pd.DataFrame(data=values[1:], columns=values[0])
    
    ''' convert strings to appropriate data types '''
    dfHoldings['Current Holding'] = dfHoldings['Current Holding'].apply(sheet.gCellStrToInt)
    dfHoldings['Purchase $'] = dfHoldings['Purchase $'].apply(sheet.gCellDollarStrToFloat)
    dfHoldings['Optioned'] = dfHoldings['Optioned'].apply(sheet.gCellStrToBool)
    
    ''' use the symbols as the index '''
    dfHoldings = dfHoldings.set_index('Symbol')
    
    return dfHoldings

def loadMarketDetails(sheet, sheetID, readRange):
    '''  '''
    result = sheet.googleSheet.values().get(spreadsheetId=sheetID, range=readRange).execute()
    values = result.get('values', [])
    if not values:
        print('\tNo data found.')

    dfMarketDetails = pd.DataFrame(data=values[1:], columns=values[0])

    ''' convert strings to appropriate data types '''
    dfMarketDetails['Symbol'] = dfMarketDetails['Symbol'].apply(seperateSymbol)
    dfMarketDetails['Day Change %'] = dfMarketDetails['Day Change %'].apply(sheet.gCellPctStrToFloat)
    dfMarketDetails['Last Price'] = dfMarketDetails['Last Price'].apply(sheet.gCellDollarStrToFloat)
    dfMarketDetails['Dividend Yield %'] = dfMarketDetails['Dividend Yield %'].apply(sheet.gCellPctStrToFloat)
    dfMarketDetails['Dividend $'] = dfMarketDetails['Dividend $'].apply(sheet.gCellDollarStrToFloat)
    dfMarketDetails['Dividend Ex-Date'] = dfMarketDetails['Dividend Ex-Date'].apply(sheet.gCellDateStrToDate, args=('MMDDYYYY', np.NaN))
    dfMarketDetails['P/E Ratio'] = dfMarketDetails['P/E Ratio'].apply(sheet.gCellDollarStrToFloat)
    dfMarketDetails['52 Week High'] = dfMarketDetails['52 Week High'].apply(sheet.gCellDollarStrToFloat)
    dfMarketDetails['52 Week Low'] = dfMarketDetails['52 Week Low'].apply(sheet.gCellDollarStrToFloat)
    dfMarketDetails['Volume'] = dfMarketDetails['Volume'].apply(sheet.gCellStrToInt)
    # Sectore remains a string
    dfMarketDetails['Next Earnings Date'] = dfMarketDetails['Next Earnings Date'].apply(sheet.gCellDateStrToDate, args=('MMDDYYYY', np.NaN))

    ''' use the symbols as the index '''
    dfMarketDetails = dfMarketDetails.set_index('Symbol')
    
    return dfMarketDetails
    
def calculateFields(mktOptions, dfHoldings, dfMarketData):
    exc_txt = "Unable to determine calculated field values"
    
    try:
        ''' The order of these updates is important as the later ones use the values calculated earlier '''
        mktOptions['Purchase $'] = mktOptions.apply(marketOprionsPopulate, axis=1, args=('Purchase $', dfHoldings, dfMarketData))
        mktOptions['Earnings Date'] = mktOptions.apply(marketOprionsPopulate, axis=1, args=('Earnings Date', dfHoldings, dfMarketData))
        mktOptions['Dividend Date'] = mktOptions.apply(marketOprionsPopulate, axis=1, args=('Dividend Date', dfHoldings, dfMarketData))
        mktOptions['Current Holding'] = mktOptions.apply(marketOprionsPopulate, axis=1, args=('Current Holding', dfHoldings, dfMarketData))
        mktOptions['Dividend'] = mktOptions.apply(dividendRisk, axis=1)
        mktOptions['Earnings'] = mktOptions.apply(earningsRisk, axis=1)
        mktOptions['Qty'] = mktOptions.apply(optionQty, axis=1)
        mktOptions['Risk Management'] = mktOptions.apply(riskManagement, axis=1)
        mktOptions['premium'] = mktOptions.apply(premium, axis=1)
        mktOptions['commission'] = mktOptions.apply(commission, axis=1)
        mktOptions['Max Profit'] = mktOptions.apply(maxProfit, axis=1)
        mktOptions['Loss vs. Profit'] = mktOptions.apply(lossVProfit, axis=1)
        mktOptions['max gain APY'] = mktOptions.apply(maxGainApy, axis=1)

    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        sys.exit(exc_txt + "\n\t" + exc_str)

    return mktOptions

def eliminateLowReturnOptions(processCtrl, putTab="", callTab=""):
    '''
    Access put and call options from Google sheets passed as parameters, eliminate options with high risk
    and low return, write the remaining options to a new tab for human consideration
    '''
        
    exc_txt = "An error occurred eliminating options"

    try:
        #print("eliminateLowReturnOptions\n\tworkbook: {}\n\tputs: {}\n\tcalls: {}".format(workbookID, putSheetName, callSheetName))

        ''' find Google sheet ID '''
        googleLocal = get_ini_data("GOOGLE")
        ''' run time control parameters from json and UI '''
        ''' text '''
        fileName = processCtrl["gsheet"]["entry"].get()
        fileID = googleLocal[fileName]
        #holdingRange = processCtrl["current holdings data range"]["entry"].get()
        #marketDataRange = processCtrl["market data range"]["entry"].get()
        optionsHeader = processCtrl["options header range"]["entry"].get()
        optionsDataRange = processCtrl["options data range"]["entry"].get()
        ''' numerical '''
        testForGainAPY = float(processCtrl['minimum max gain APY']["entry"].get())
        testForProfit = float(processCtrl['minimum max profit']["entry"].get())
        testForOTMThreshold = float(processCtrl['out of the money threshold']["entry"].get())
        ''' controls with multiple values '''
    
        gSheet = gs.googleSheet()

        if len(putTab) > 0:
            ''' load put option '''
            readRange = putTab + '!A1:CZ'
            puts = gSheet.readGoogleSheet(fileID, readRange)
            
        if len(callTab) > 0:
            ''' load call option '''
            readRange = callTab + '!A1:CZ'
            calls = gSheet.readGoogleSheet(fileID, readRange)
        
        if len(putTab) > 0 and len(callTab):
            toConsider = pd.concat([puts, calls])
        elif len(putTab) > 0:
            toConsider = puts
        elif len(callTab) > 0:
            toConsider = calls
        
        ''' Need to set index to avoid duplicates '''
        toConsider=toConsider.reset_index(drop=True)

        ''' remove Qty==0 rows '''
        print("removing 0 Qty option trades from {} rows".format(len(toConsider)))
        testColumn = 'Qty'
        testFor = '0'
        rowsToRemove = toConsider.loc[toConsider[testColumn] == testFor]
        toConsider = toConsider.drop(rowsToRemove.index)
            
        ''' remove in The Money==0 rows '''
        print("removing In The Money options from {} rows".format(len(toConsider)))
        testColumn = 'in The Money'
        testFor = 'TRUE'
        rowsToRemove = toConsider.loc[toConsider[testColumn] == testFor]
        toConsider = toConsider.drop(rowsToRemove.index)
        
        ''' remove Earnings risk rows '''
        print("removing Earnings risk rows rows from {} rows".format(len(toConsider)))
        testColumn = 'Earnings'
        regexPattern = '9 - '
        earningRiskRows = toConsider.loc[toConsider[testColumn].str.match(regexPattern)]
        toConsider = toConsider.drop(earningRiskRows.index)
        
        ''' remove Dividend risk rows '''
        print("removing Dividend risk rows from {} rows".format(len(toConsider)))
        testColumn = 'Dividend'
        regexPattern = '9 - '
        earningRiskRows = toConsider.loc[toConsider[testColumn].str.match(regexPattern)]
        toConsider = toConsider.drop(earningRiskRows.index)
        
        ''' remove max gain APY < 15% rows '''
        testColumn = 'max gain APY'
        testFor = testForGainAPY
        print("removing max gain APY < {}% rows from {} rows".format(testFor, len(toConsider)))
        toConsider[testColumn] = toConsider[testColumn].apply(gSheet.gCellStrToFloat)
        rowsToRemove = toConsider.loc[toConsider[testColumn] < testFor]
        toConsider = toConsider.drop(rowsToRemove.index)
        toConsider[testColumn] = toConsider[testColumn].apply(str)
        
        ''' remove Max Profit < 500 rows '''
        testColumn = 'Max Profit'
        testFor = testForProfit
        print("removing Max Profit < ${} rows from {} rows".format(testFor, len(toConsider)))
        toConsider[testColumn] = toConsider[testColumn].apply(gSheet.gCellStrToFloat)
        rowsToRemove = toConsider.loc[toConsider[testColumn] < testFor]
        toConsider = toConsider.drop(rowsToRemove.index)
        toConsider[testColumn] = toConsider[testColumn].apply(str)
        
        ''' remove OTM Probability < 0.8 '''
        testColumn = 'OTM Probability'
        testFor = testForOTMThreshold
        print("removing OTM (option expires worthless) Probability < {} rows from {} rows".format(testFor, len(toConsider)))
        toConsider[testColumn] = toConsider[testColumn].apply(gSheet.gCellStrToFloat)
        rowsToRemove = toConsider.loc[toConsider[testColumn] < testFor]
        toConsider = toConsider.drop(rowsToRemove.index)
        toConsider[testColumn] = toConsider[testColumn].apply(str)
        
        ''' Sort remaining options on potential yield (descending) '''
        toConsider = toConsider.sort_values(by=['max gain APY'], ascending=False)
        
        if len(toConsider) > 0:
            exc_txt = "An error occurred writing potential option trades to Google workbook"
            now = dt.datetime.now()
            timeStamp = ' {:4d}{:0>2d}{:0>2d} {:0>2d}{:0>2d}{:0>2d}'.format(now.year, now.month, now.day, \
                                                                            now.hour, now.minute, now.second)        
            sheetName = "consider" + timeStamp
            
            gSheet.addGoogleSheet(fileID, sheetName)
            gSheet.updateGoogleSheet(fileID, sheetName + optionsHeader, toConsider.columns)
            gSheet.updateGoogleSheet(fileID, sheetName + optionsDataRange, toConsider)
            print("There are {} options with acceptable risk / reward to consider in {}".format(len(toConsider), sheetName))
    
        else:
            print("There are no options with acceptable risk / reward to consider")

    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        sys.exit(exc_txt + "\n\t" + exc_str)

    return

def prepMktOptionsForSheets(mktOptions):
    ''' convert data elements to formats suitable for writing to Google sheet cells '''
    sheetMktOption = mktOptions
    
    sheetMktOption['Dividend Date'] = sheetMktOption['Dividend Date'].apply(date.isoformat)
    sheetMktOption['Earnings Date'] = sheetMktOption['Earnings Date'].apply(date.isoformat)
    sheetMktOption['expiration'] = sheetMktOption['expiration'].apply(date.isoformat)
        
    return sheetMktOption

def prepMktOptionsForAnalysis(mktOptions):
    ''' convert data elements to formats suitable for analysis '''
    analysisMktOption = mktOptions

    analysisMktOption = analysisMktOption.fillna(0)
    analysisMktOption['expiration'] = analysisMktOption['expiration'].apply(date.fromisoformat)
        
    return analysisMktOption
