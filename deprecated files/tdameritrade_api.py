'''
Created on Jan 31, 2018

@author: Brian
'''
from tda_api_library import tda_data
from tda_api_library import tda_login
from tda_api_library import tda_logout
from tda_api_library import tda_price_history
from tda_api_library import tda_symbol_lookup


if __name__ == '__main__':
    print ("Good morning Dr. chandra\n")
    
    print ("tda source", tda_data("source"))
    print ("tda version", tda_data("version"))
    print ("tda password", tda_data("password"))
    print ("tda user", tda_data("userid"))
    
    keep_alive = tda_login()
    
    found, symbol = tda_symbol_lookup("citigroup")
    print ("symbol lookup result", found, symbol)
    tda_price_history("C")
    
    tda_logout()
    
    print ("\nDave, this conversations can serve no further purpose")
    pass