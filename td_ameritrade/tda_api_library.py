from quandl_library import get_ini_data
from tda_price_history import tda_price_history_data
import urllib.request

def tda_data(key):
    devdata = get_ini_data('TDA_API')
    key_data = devdata[key]
    return key_data

def tda_login():
    print ("\nLogging into TD Ameritrade")
    
    source = tda_data("source")
    version = tda_data("version")
    url="https://apis.tdameritrade.com/apps/300/LogIn?source=" + source + "&version=" + version
    print ("URL:", url)
    
    user = tda_data("userid")
    pwd = tda_data("password")
    values={'userid':user,'password':pwd, 'source':source, 'version':version, '':'0%0A%0A%0A%0A'}
    print ("payload:", values)
    
    data = urllib.parse.urlencode(values)
    data = data.encode('utf-8')
    
    headers={}
    headers['Content-type'] = 'application/x-www-form-urlencoded'
    
    req = urllib.request.Request(url, data=data, headers=headers)
    HTTP_resp = urllib.request.urlopen(req)
    print ('login response', HTTP_resp.status, HTTP_resp.reason, "\n", HTTP_resp.read())
    
    keep_alive = 55
    
    return keep_alive 

def tda_logout():
    print ("\nLogging out from TD Ameritrade")
    
    source = tda_data("source")
    url="https://apis.tdameritrade.com/apps/100/LogOut?source=" + source
    HTTP_resp = urllib.request.urlopen(url)
    print ('logout response', HTTP_resp.read())

def tda_symbol_lookup(match):
    '''
    ===========================================================================================
    https://apis.tdameritrade.com/apps/100/SymbolLookup?source=<#sourceID#>&matchstring=some search string
    
    <amtd>
        <result>FAIL</result> 
        <error>Source parameter is required.</error> 
    </amtd>

    <amtd>
        <result>FAIL</result> 
        <error>Error -- No Symbol entered.</error> 
    </amtd>
    ===========================================================================================
    '''
    print ("\nSymbol lookup for:", match)
    
    source = tda_data("source")
    url="https://apis.tdameritrade.com/apps/100/SymbolLookup?source=" + source + "&matchstring=" + match
    print ("URL:", url)
    HTTP_resp = urllib.request.urlopen(url)
    print ('logout response', HTTP_resp.read())
    
    found = True
    symbol = "C"

    return found, symbol

def tda_price_history(symbol, intervaltype="DAILY", periodtype="MONTH", intervalduration=1, idtype="SYMBOL"):    
    '''
    ===========================================================================================
    https://apis.tdameritrade.com/apps/100/PriceHistory?source=XXXX&requestvalue=X&intervaltype=DAILY&
        periodtype=MONTH&intervalduration=1&requestidentifiertype=SYMBOL
        
    Reurns binary data
    
    PriceHistory Response
    Field               Type        Length         Description
                                    (8 bit bytes)
    Symbol Count        Integer     4              Number of symbols for which data is being returned. The subsequent sections are repeated this many times
 
    REPEATING SYMBOL DATA
    Symbol Length       Short       2              Length of the Symbol field
    Symbol              String      Variable       The symbol for which the historical data is returned
    Error Code          Byte        1              0=OK, 1=ERROR
    Error Length        Short       2              Only returned if Error Code=1. Length of the Error string
    Error Text          String      variable       Only returned if Error Code=1. The string describing the error
    Bar Count           Integer     4              # of chart bars; only if error code=0
    REPEATING PRICE DATA
    close               Float       4
    high                Float       4
    low                 Float       4
    open                Float       4
    volume              Float       4              in 100's
    timestamp Long                  8              time in milliseconds from 00:00:00 UTC on January 1, 1970
    END OF REPEATING PRICE DATA
    Terminator          Bytes       2              0xFF, 0XFF
    END OF REPEATING SYMBOL DATA
 
    ===========================================================================================
    '''
    print ("\nPrice history lookup for:", symbol)
    
    source = tda_data("source")
    url="https://apis.tdameritrade.com/apps/100/PriceHistory?source=" + source + "&requestvalue=" + symbol + "&intervaltype=DAILY&periodtype=MONTH&intervalduration=1&requestidentifiertype=SYMBOL"
    print ("URL:", url)
    HTTP_resp = urllib.request.urlopen(url)
    data = bytearray(HTTP_resp.read())
    #data = HTTP_resp.read()
    print ('Price history lookup response\n', data)
    
    history = tda_price_history_data()
    history.unpack(data)
    
    return history