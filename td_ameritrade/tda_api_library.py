'''
Created on Jan 31, 2018

@author: Brian
'''
import os
import logging
import time
import requests
import json
import pandas as pd

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
    for symbol in tda_read_watch_lists(json_authentication):
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
                print("Now - %s, %s, %s" % (now, time.ctime(now), '{:.0f}'.format(now*1000)))
                print("Last datetime - %s, %s, %s" % (f_last_date, time.ctime(f_last_date/1000), '{:.0f}'.format(f_last_date)))
                eod_count = df_eod.shape[0]
                headers = {'Authorization' : 'Bearer ' + json_authentication["currentToken"]}
                params = {'apikey' : json_authentication["apikey"], \
                          'periodType' : 'month', 'frequencyType' : 'daily', 'frequency' : '1', \
                          'endDate' : '{:.0f}'.format(now*1000), 'startDate' : '{:.0f}'.format(f_last_date)}
                response = requests.get(url, headers=headers, params=params)
                if response.ok:
                    print("Incremental price history received")
                    price_history = json.loads(response.text)
                    tda_symbol = price_history["symbol"]
                    if not price_history["empty"]:
                        candles = price_history["candles"]
                        for candle in candles:
                            df_eod.loc[eod_count] = [candle["datetime"], candle["open"], candle["high"], candle["low"], candle["close"], candle["volume"]]
                            eod_count += 1
                            df_eod.drop_duplicates(subset=['DateTime'], keep='last', inplace=True)
                    else:
                        print("Incremental EOD data for %s was empty" % tda_symbol)
                        logging.info("Data for %s was empty" % tda_symbol)
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
        print ("\nEOD data for %s\n%s" % (symbol, df_eod))
        logging.info("nEOD data for %s\n%s" % (symbol, df_eod))

    logging.debug('<---- update_tda_eod_data')
    return