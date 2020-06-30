'''
Created on Jan 31, 2018

@author: Brian
'''
import logging
import time
import requests
import json


def tda_get_access_token():
    logging.debug('tda_get_access_token ---->\n %s')
    token = "9NaYTaDGU0q24g1Sq5X8I3p583O71CvimYee6omMN8CmpmF3oWq2ARf1zxulmBDbDMF/IOL+cbTL467NKQ1ZN4feqKLmRYXKjymSsI98z2M4VZSL+QH1xgQg2Cg07Qm2taZypAKbBO35RpUNe8hIQu4fRrgnNekHeP19pGFUbVGUKv9YFRYbFMvZeswNWTIvH+sU7wBxBUvpfAIaUMnu9HjW+XoHCcRfImW0iLGMvTJO0uDKM5QDlULuZY4WgC+5kT79iCwGTAvcGAsr+2Cfpt5DfEgtoIywlNxyEOlEsxCSTLvGMxjfgA5YhAqVMP0Rp3D5D685gj1yASuXvTA1Icgk573FrZ0jXJYhYgM1qlhZZ1llKDBbTbdFISQPUOfUjN/Oi5n640Dy7ofDB/gFJkqlek3eX1ch/rF/hXmf/oSYQ1LJnPouQ5EpqR4g03v5zv3snoNSYrQ/AUmNcFFjmJbCaZkgkDgVSq3MBXpUM+tVD9erzTWUnsVv4t+vkD20bqoOMZadejdN3X8KsODSYntuok0MRDqruWIjU4RDN+xlq2W9FScg2T2AZScmreq/bgSM6tm1O8N3irjKPyHvqvfBA/zr6EaMNK6Sa100MQuG4LYrgoVi/JHHvlkkFLqzyLThlrZvrbMFa6SLGNFnEVjcEwHtjnTQnLEua/EvOispCvXTlf35oMCHOcSLzQ0/V/r2ywpTsHQTlyaMbEHARDBr05vWoJ3V4UUCJDrcMpNY0534Dg4qBKlLguEGKmhdr/MErO/KiKAdwuaUsLJssXjxMTS6gzAW1XiKe0XePoBJC5a4ZJn4km0UNySR+cVU0ZhliIcKjU3+M6S9BO5P8VadCW2vcnuiapPIZf3NpOggLbsGWRysUUsHZIG2btoFBIQ8AefArf727CGxrigWQJ1cH+TpPNkkO+yafJLePVltk4OMC70rZsAO0EKsiPntjO6UtRSmxBrZd32H1lsdk9HN/3ObUhrBxXhHYp2y3xIfWJLS19oVaAE6xabH6jhI2+OyvcfmR8FXB3uNXwYvvFDcslJQmv7T+uDhHSxJvVFtW/EqBF64IN+nDA8H1Ui15VJAlYyRBlMUnX74gZfw2yTGlu9PyIR0NLb389JWQQ7zNjpEMnwoWHyHXoLN2vXlpiEKxZjxPDWz+NSqw5kFcuI34aLH97LUPCcNcc3wMy8dMKlU7OhcDia7JFOkrnORW/7FKlZXkIfWR4fkvX6GnVzS8kfZylc+212FD3x19z9sWBHDJACbC00B75E"
    logging.debug('<---- tda_get_access_token')
    return token
        
def tda_symbol_lookup(match):
    logging.debug('tda_symbol_lookup ---->\n %s')
    found = True
    symbol = "C"
    logging.debug('<---- tda_symbol_lookup')
    return found, symbol

def tda_read_watch_list():
    return

def     update_tda_eod_data(df_symbols, tda_api_key):
    logging.debug('update_tda_eod_data ---->\n %s' % df_symbols)
    print ("\nPrice history update for:", df_symbols)
    
    '''
    TDA tokens are valid for 30 minutes
    '''
    tda_token_valid = False
    
    for symbol in df_symbols:
        if not tda_token_valid:
            print("need a new token")
            tda_token = tda_get_access_token()
            tda_token_valid = True
            tda_token_time = time.time()
            
        url = 'https://api.tdameritrade.com/v1/marketdata/AAPL/pricehistory'
        apikey = tda_api_key
        periodType = 'month'
        period = '1'
        frequencyType = 'daily'
        frequency = '1'
        protocolVersion = 'HTTP/1.1'
        params = {'apikey' : apikey,
                  'periodType' : periodType,
                  'period' : period,
                  'frequencyType' : frequencyType,
                  'frequency' : frequency
                  }
        response = requests.get(url, params=params)
           
        if response.ok:
            price_history_data = response.content
            price_history = json.loads(response.text)
            empty = price_history["empty"]
            tda_symbol = price_history["symbol"]
            if not empty:
                candles = price_history["candles"]
                for candle in candles:
                    open = candle["open"]
                    close = candle["close"]
                    high = candle["high"]
                    low = candle["low"]
                    volume = candle["volume"]
                    datetime = candle["datetime"]
            else:
                print("Data for %s was empty" % tda_symbol)
                logging.info("Data for %s was empty" % tda_symbol)
        else:
            print("Unable to get EOD data for %s, response code=%s" % symbol, response.status_code)
            logging.info("Unable to get EOD data for %s, response code=%s" % symbol, response.status_code)
            
        if time.time() - tda_token_time > 25*60 :
            tda_token_valid = False
    
    logging.debug('<---- update_tda_eod_data')
    return