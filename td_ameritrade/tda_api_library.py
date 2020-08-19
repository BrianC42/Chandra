'''
Created on Jan 31, 2018

@author: Brian

'''
import os
import logging
from datetime import date
import time
import requests
import json
import pandas as pd

def format_tda_datetime(tda_datetime):
    str_dt = date.fromtimestamp(tda_datetime/1000).strftime("%Y-%m-%d")
    return str_dt

def tda_get_authentication_details(auth_file):
    logging.debug('tda_get_authentication_details ---->')

    json_f = open(auth_file, "rb")
    json_auth = json.load(json_f)
    json_f.close
    
    logging.debug('<---- tda_get_authentication_details')
    return json_auth

def tda_update_authentication_details(json_authentication):
    logging.debug('tda_update_authentication_details ---->\n')
    
    json_f = open('d:\\brian\\AI Projects\\tda_local.json', "w")
    json.dump(json_authentication, json_f, indent=0)
    json_f.close
    
    logging.debug('<---- tda_update_authentication_details')
    return

def tda_get_access_token(json_authentication):
    logging.debug('tda_get_access_token ---->\n')

    if time.time() > (json_authentication['tokenObtained'] + json_authentication['expiresIn']):
        url = 'https://api.tdameritrade.com/v1/oauth2/token'
        params = {'grant_type' : 'refresh_token', 'refresh_token' : json_authentication['refreshToken'], 'access_type' : '', 'code' : '', 'client_id' : json_authentication['apikey'], 'redirect_uri' : json_authentication['redirectUri']}
        response = requests.post(url, data=params)       
        if response.ok:
            Authorization_details = json.loads(response.text)
            json_authentication["currentToken"] = Authorization_details["access_token"]
            json_authentication["scope"] = Authorization_details["scope"]
            json_authentication["tokenObtained"] = time.time()
            json_authentication["expiresIn"] = Authorization_details["expires_in"]
            json_authentication["token_type"] = Authorization_details["token_type"]
            tda_update_authentication_details(json_authentication)
        else:
            logging.info("Authorization request response not OK, response code=%s, reason %s, %s" % (response.status_code, response.reason, response.text))
            json_authentication["currentToken"] = ""
            json_authentication["scope"] = ""
            json_authentication["tokenObtained"] = 0.0
            json_authentication["expiresIn"] = 0.0
            json_authentication["token_type"] = ""
            
    logging.debug('<---- tda_get_access_token')
    return json_authentication["currentToken"]
        
def df_tda_option_chain(rcount):
    df = pd.DataFrame(index=[i for i in range(0, rcount)], \
                      columns = ["underlying symbol", "strategy", "underlyingPrice", "numberOfContracts", \
                                 "expiration", "strike", \
                                 'putCall','option symbol', 'description', 'exchangeName', 'bid', \
                                 'ask', 'last', 'mark', 'bidSize', 'askSize', 'lastSize', \
                                 'highPrice', 'lowPrice', 'openPrice', 'closePrice', 'totalVolume', 'quoteTimeInLong', \
                                 'tradeTimeInLong', 'netChange', 'volatility', \
                                 'delta', 'gamma', 'theta', 'vega', 'rho', \
                                 'timeValue', 'openInterest', 'inTheMoney', 'theoreticalOptionValue', 'theoreticalVolatility', \
                                 'mini', 'nonStandard', 'strikePrice', 'expirationDate', 'daysToExpiration', 'expirationType', \
                                 'multiplier', 'settlementType', 'deliverableNote', 'isIndexOption', 'percentChange', \
                                 'markChange', 'markPercentChange'])
    
    return df

def tda_read_option_chain(authentication_parameters, p_symbol):
    TSA_DATE_MAPS = ['callExpDateMap', 'putExpDateMap']
    df_tda_options = df_tda_option_chain(0)
    json_authentication = tda_get_authentication_details(authentication_parameters)
    currentToken = tda_get_access_token(json_authentication)    
    url = 'https://api.tdameritrade.com/v1/marketdata/chains'
    apikey = 'Bearer ' + currentToken
    symbol = p_symbol
    headers = {'Authorization' : apikey}
    params = {'symbol' : symbol}
    response = requests.get(url, headers=headers, params=params)
    if response.ok:
        tda_option_chain_json = json.loads(response.text)
        for dateMap in TSA_DATE_MAPS:
            tda_ExpDateMap = tda_option_chain_json[dateMap]
            ndx = 0
            for tda_expDate in tda_ExpDateMap:
                for tda_option in tda_ExpDateMap[tda_expDate]:
                    for tda_strike in tda_ExpDateMap[tda_expDate][tda_option]:
                        ndx += 1
            df_tda = df_tda_option_chain(ndx)
            ndx = 0
            for tda_expDate in tda_ExpDateMap:
                for tda_option in tda_ExpDateMap[tda_expDate]:
                    for tda_strike in tda_ExpDateMap[tda_expDate][tda_option]:
                        df_tda.at[ndx, "underlying symbol"] = tda_option_chain_json["symbol"]
                        df_tda.at[ndx, "strategy"] = tda_option_chain_json["strategy"]
                        df_tda.at[ndx, "underlyingPrice"] = tda_option_chain_json["underlyingPrice"]
                        df_tda.at[ndx, "numberOfContracts"] =  tda_option_chain_json["numberOfContracts"]
                        df_tda.at[ndx, "putCall"] = tda_strike['putCall']
                        df_tda.at[ndx, "option symbol"] = tda_strike["symbol"]
                        df_tda.at[ndx, "description"] = tda_strike["description"]
                        df_tda.at[ndx, "exchangeName"] = tda_strike["exchangeName"]
                        df_tda.at[ndx, "bid"] = tda_strike["bid"]
                        df_tda.at[ndx, "ask"] = tda_strike["ask"]
                        df_tda.at[ndx, "last"] = tda_strike["last"]
                        df_tda.at[ndx, "mark"] = tda_strike["mark"]
                        df_tda.at[ndx, "bidSize"] = tda_strike["bidSize"]
                        df_tda.at[ndx, "askSize"] = tda_strike["askSize"]
                        df_tda.at[ndx, "lastSize"] = tda_strike["lastSize"]
                        df_tda.at[ndx, "highPrice"] = tda_strike["highPrice"]
                        df_tda.at[ndx, "lowPrice"] = tda_strike["lowPrice"]
                        df_tda.at[ndx, "openPrice"] = tda_strike["openPrice"]
                        df_tda.at[ndx, "closePrice"] = tda_strike["closePrice"]
                        df_tda.at[ndx, "totalVolume"] = tda_strike["totalVolume"]
                        df_tda.at[ndx, "quoteTimeInLong"] = tda_strike["quoteTimeInLong"]
                        df_tda.at[ndx, "tradeTimeInLong"] = tda_strike["tradeTimeInLong"]
                        df_tda.at[ndx, "netChange"] = tda_strike["netChange"]
                        df_tda.at[ndx, "volatility"] = tda_strike["volatility"]
                        df_tda.at[ndx, "delta"] = tda_strike["delta"]
                        df_tda.at[ndx, "gamma"] = tda_strike["gamma"]
                        df_tda.at[ndx, "theta"] = tda_strike["theta"]
                        df_tda.at[ndx, "vega"] = tda_strike["vega"]
                        df_tda.at[ndx, "rho"] = tda_strike["rho"]
                        df_tda.at[ndx, "timeValue"] = tda_strike["timeValue"]
                        df_tda.at[ndx, "openInterest"] = tda_strike["openInterest"]
                        df_tda.at[ndx, "inTheMoney"] = tda_strike["inTheMoney"]
                        df_tda.at[ndx, "theoreticalOptionValue"] = tda_strike["theoreticalOptionValue"]
                        df_tda.at[ndx, "theoreticalVolatility"] = tda_strike["theoreticalVolatility"]
                        df_tda.at[ndx, "mini"] = tda_strike["mini"]
                        df_tda.at[ndx, "nonStandard"] = tda_strike["nonStandard"]
                        df_tda.at[ndx, "strikePrice"] = tda_strike["strikePrice"]
                        df_tda.at[ndx, "expirationDate"] = tda_strike["expirationDate"]
                        df_tda.at[ndx, "daysToExpiration"] = tda_strike["daysToExpiration"]
                        df_tda.at[ndx, "expirationType"] = tda_strike["expirationType"]
                        df_tda.at[ndx, "multiplier"] = tda_strike["multiplier"]
                        df_tda.at[ndx, "settlementType"] = tda_strike["settlementType"]
                        df_tda.at[ndx, "deliverableNote"] = tda_strike["deliverableNote"]
                        df_tda.at[ndx, "isIndexOption"] = tda_strike["isIndexOption"]
                        df_tda.at[ndx, "percentChange"] = tda_strike["percentChange"]
                        df_tda.at[ndx, "markChange"] = tda_strike["markChange"]
                        df_tda.at[ndx, "markPercentChange"] = tda_strike["markPercentChange"]
                    
                        '''
                        df_tda[ndx, "optionDeliverablesList"] = tda_strike[  "optionDeliverablesList": [
                        df_tda[ndx, ""] = tda_strike[    {
                        df_tda[ndx, ""] = tda_strike[      "symbol": "string",
                        df_tda[ndx, ""] = tda_strike[      "assetType": "string",
                        df_tda[ndx, ""] = tda_strike[      "deliverableUnits": "string",
                        df_tda[ndx, ""] = tda_strike[      "currencyType": "string"
                        df_tda[ndx, ""] = tda_strike[    }
                        df_tda[ndx, ""] = tda_strike[  ],
                        '''
                        ndx += 1
        df_tda_options = df_tda_options.append(df_tda, ignore_index=True)
        '''
        tda_putExpDateMap = tda_option_chain_json["putExpDateMap"]
        ndx = 0
        for tda_expDate in tda_putExpDateMap:
            for tda_option in tda_putExpDateMap[tda_expDate]:
                for tda_strike in tda_putExpDateMap[tda_expDate][tda_option]:
                    ndx += 1
        df_tda_puts = df_tda_option_chain(ndx)
        ndx = 0
        for tda_expDate in tda_putExpDateMap:
            for tda_option in tda_putExpDateMap[tda_expDate]:
                for tda_strike in tda_putExpDateMap[tda_expDate][tda_option]:
                    df_tda_puts.at[ndx, "underlying symbol"] = tda_option_chain_json["symbol"]
                    df_tda_puts.at[ndx, "strategy"] = tda_option_chain_json["strategy"]
                    df_tda_puts.at[ndx, "underlyingPrice"] = tda_option_chain_json["underlyingPrice"]
                    df_tda_puts.at[ndx, "numberOfContracts"] =  tda_option_chain_json["numberOfContracts"]
                    df_tda_puts.at[ndx, "putCall"] = tda_strike['putCall']
                    df_tda_puts.at[ndx, "option symbol"] = tda_strike["symbol"]
                    df_tda_puts.at[ndx, "description"] = tda_strike["description"]
                    df_tda_puts.at[ndx, "exchangeName"] = tda_strike["exchangeName"]
                    df_tda_puts.at[ndx, "bid"] = tda_strike["bid"]
                    df_tda_puts.at[ndx, "ask"] = tda_strike["ask"]
                    df_tda_puts.at[ndx, "last"] = tda_strike["last"]
                    df_tda_puts.at[ndx, "mark"] = tda_strike["mark"]
                    df_tda_puts.at[ndx, "bidSize"] = tda_strike["bidSize"]
                    df_tda_puts.at[ndx, "askSize"] = tda_strike["askSize"]
                    df_tda_puts.at[ndx, "lastSize"] = tda_strike["lastSize"]
                    df_tda_puts.at[ndx, "highPrice"] = tda_strike["highPrice"]
                    df_tda_puts.at[ndx, "lowPrice"] = tda_strike["lowPrice"]
                    df_tda_puts.at[ndx, "openPrice"] = tda_strike["openPrice"]
                    df_tda_puts.at[ndx, "closePrice"] = tda_strike["closePrice"]
                    df_tda_puts.at[ndx, "totalVolume"] = tda_strike["totalVolume"]
                    df_tda_puts.at[ndx, "quoteTimeInLong"] = tda_strike["quoteTimeInLong"]
                    df_tda_puts.at[ndx, "tradeTimeInLong"] = tda_strike["tradeTimeInLong"]
                    df_tda_puts.at[ndx, "netChange"] = tda_strike["netChange"]
                    df_tda_puts.at[ndx, "volatility"] = tda_strike["volatility"]
                    df_tda_puts.at[ndx, "delta"] = tda_strike["delta"]
                    df_tda_puts.at[ndx, "gamma"] = tda_strike["gamma"]
                    df_tda_puts.at[ndx, "theta"] = tda_strike["theta"]
                    df_tda_puts.at[ndx, "vega"] = tda_strike["vega"]
                    df_tda_puts.at[ndx, "rho"] = tda_strike["rho"]
                    df_tda_puts.at[ndx, "timeValue"] = tda_strike["timeValue"]
                    df_tda_puts.at[ndx, "openInterest"] = tda_strike["openInterest"]
                    df_tda_puts.at[ndx, "inTheMoney"] = tda_strike["inTheMoney"]
                    df_tda_puts.at[ndx, "theoreticalOptionValue"] = tda_strike["theoreticalOptionValue"]
                    df_tda_puts.at[ndx, "theoreticalVolatility"] = tda_strike["theoreticalVolatility"]
                    df_tda_puts.at[ndx, "mini"] = tda_strike["mini"]
                    df_tda_puts.at[ndx, "nonStandard"] = tda_strike["nonStandard"]
                    df_tda_puts.at[ndx, "strikePrice"] = tda_strike["strikePrice"]
                    df_tda_puts.at[ndx, "expirationDate"] = tda_strike["expirationDate"]
                    df_tda_puts.at[ndx, "expirationType"] = tda_strike["expirationType"]
                    df_tda_puts.at[ndx, "multiplier"] = tda_strike["multiplier"]
                    df_tda_puts.at[ndx, "settlementType"] = tda_strike["settlementType"]
                    df_tda_puts.at[ndx, "deliverableNote"] = tda_strike["deliverableNote"]
                    df_tda_puts.at[ndx, "isIndexOption"] = tda_strike["isIndexOption"]
                    df_tda_puts.at[ndx, "percentChange"] = tda_strike["percentChange"]
                    df_tda_puts.at[ndx, "markChange"] = tda_strike["markChange"]
                    df_tda_puts.at[ndx, "markPercentChange"] = tda_strike["markPercentChange"]
        '''
        '''
                    df_tda_puts[ndx, "optionDeliverablesList"] = tda_strike[  "optionDeliverablesList": [
                    df_tda_puts[ndx, ""] = tda_strike[    {
                    df_tda_puts[ndx, ""] = tda_strike[      "symbol": "string",
                    df_tda_puts[ndx, ""] = tda_strike[      "assetType": "string",
                    df_tda_puts[ndx, ""] = tda_strike[      "deliverableUnits": "string",
                    df_tda_puts[ndx, ""] = tda_strike[      "currencyType": "string"
                    df_tda_puts[ndx, ""] = tda_strike[    }
                    df_tda_puts[ndx, ""] = tda_strike[  ],
                    ndx += 1
        '''
    else:
        print("Unable to get option chains, response code=%s, reason %s, %s" % (response.status_code, response.reason, response.text))
        logging.info("Unable to get option chains, response code=%s, reason %s, %s" % (response.status_code, response.reason, response.text))
        
    return df_tda_options, response.text
        
def tda_read_watch_lists(json_authentication):
    logging.debug('tda_read_watch_lists ---->\n %s')
    
    #"watchLists" : ["Combined Holding","Stock Information","Stock Information 2"]
    currentToken = tda_get_access_token(json_authentication)    
    url = 'https://api.tdameritrade.com/v1/accounts/' + json_authentication["account"] + '/watchlists'
    apikey = 'Bearer ' + currentToken
    headers = {'Authorization' : apikey}
    response = requests.get(url, headers=headers)
    if response.ok:
        print("Watchlists retrieved")
        symbol_list = []
        tda_watch_lists_json = json.loads(response.text)
        for list_details in tda_watch_lists_json:
            list_name = list_details["name"]
            listItems = list_details["watchlistItems"]
            for itemDetails in listItems:
                instrument = itemDetails["instrument"]
                symbol = instrument["symbol"]
                if list_name in json_authentication["watchLists"]:
                    symbol_list.append(symbol)
        logging.info("Symbols, %s" % symbol_list)
    else:
        print("Unable to get watch list data, response code=%s, reason %s, %s" % (response.status_code, response.reason, response.text))
        logging.info("Unable to get watch list data, response code=%s, reason %s, %s" % (response.status_code, response.reason, response.text))
            
    logging.debug('<---- tda_read_watch_lists')
    return symbol_list

def update_tda_eod_data(authentication_parameters):
    logging.debug('update_tda_eod_data ---->')
    
    eod_data_dir = 'd:\\brian\\AI Projects\\tda\\market_data\\'
    json_authentication = tda_get_authentication_details(authentication_parameters)
    tda_throttle_time = time.time()
    tda_throttle_count = 0
    print("Now - %s, %s" % (time.ctime(time.time()), '{:.0f}'.format(time.time()*1000)))
    for symbol in tda_read_watch_lists(json_authentication):
        if tda_throttle_count < 110:
            tda_throttle_count += 1
        else:
            now = time.time()
            if now - tda_throttle_time < 60:
                time.sleep(now - tda_throttle_time)
            tda_throttle_count = 0
            tda_throttle_time = time.time()
        tda_get_access_token(json_authentication)    
        eod_file = eod_data_dir + symbol + '.csv'
        url = 'https://api.tdameritrade.com/v1/marketdata/' + symbol + '/pricehistory'
        if os.path.isfile(eod_file):
            df_eod = pd.read_csv(eod_data_dir + symbol + '.csv')
            i_ndx = 0
            for col in df_eod.columns:
                if col == 'DateTime':            
                    dtime_col = i_ndx
                i_ndx += 1
            eod_rows = df_eod.shape[0]
            if eod_rows > 0:
                f_last_date = float(df_eod.iat[eod_rows-1,dtime_col])
                now = time.time()
                eod_count = df_eod.shape[0]
                headers = {'Authorization' : 'Bearer ' + json_authentication["currentToken"]}
                params = {'apikey' : json_authentication["apikey"], \
                          'periodType' : 'month', 'frequencyType' : 'daily', 'frequency' : '1', \
                          'endDate' : '{:.0f}'.format(now*1000), 'startDate' : '{:.0f}'.format(f_last_date)}
                retry_count = 0
                retry = True
                while retry == True and retry_count < 5:
                    try:
                        response = requests.get(url, headers=headers, params=params)
                        if response.ok:
                            print("%s - Last datetime - %s, %s" % (symbol, time.ctime(time.time()), '{:.0f}'.format(time.time()*1000)))
                            price_history = json.loads(response.text)
                            tda_symbol = price_history["symbol"]
                            if not price_history["empty"]:
                                candles = price_history["candles"]
                                for candle in candles:
                                    df_eod.loc[eod_count] = [candle["datetime"], candle["open"], candle["high"], candle["low"], candle["close"], candle["volume"]]
                                    eod_count += 1
                                    #df_eod.drop_duplicates(subset=['DateTime'], keep='last', inplace=True)
                            else:
                                print("Incremental EOD data for %s was empty" % tda_symbol)
                                logging.info("Data for %s was empty" % tda_symbol)
                            retry = False
                    except:
                        retry_count += 1
            else:
                print("Unable to get incremental EOD data for %s, response code=%s" % (symbol, response.status_code))
                logging.info("Unable to get incremental EOD data for %s, response code=%s" % (symbol, response.status_code))            
        else:
            df_eod = pd.DataFrame(columns=['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume'])
            eod_count = 0
            params = {'apikey' : json_authentication["apikey"], 'periodType' : 'year', 'period' : '20', 'frequencyType' : 'daily', 'frequency' : '1'}
            response = requests.get(url, params=params)
            if response.ok:
                price_history = json.loads(response.text)
                tda_symbol = price_history["symbol"]
                if not price_history["empty"]:
                    candles = price_history["candles"]
                    for candle in candles:
                        df_eod.loc[eod_count] = [candle["datetime"], candle["open"], candle["high"], candle["low"], candle["close"], candle["volume"]]
                        eod_count += 1
                else:
                    print("Data for %s was empty" % tda_symbol)
                    logging.info("Data for %s was empty" % tda_symbol)
            else:
                print("Unable to get EOD data for %s, response code=%s" % (symbol, response.status_code))
                logging.info("Unable to get EOD data for %s, response code=%s" % (symbol, response.status_code))            
        df_eod.to_csv(eod_file, index=False)
        #print ("\nEOD data for %s\n%s" % (symbol, df_eod))
        logging.info("nEOD data for %s\n%s" % (symbol, df_eod))

    logging.debug('<---- update_tda_eod_data')
    return