'''
Created on Jan 31, 2018

@author: Brian

Class to handle TD Ameritrade price history data. Retrieved as binary data unpacked into more usable structures

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

'''

class tda_price_history_data:
    '''
    classdocs
    '''


    #def __init__(self, params):
    def __init__(self):
        '''
        Constructor
        '''
        print ("\nConstructing a tda price history object")
        
    def unpack(self, data):
        
        print ("data", len(data), "\n", data)
        symbol_count = 0
        symbol_count += int(data[3]) * 1
        symbol_count += int(data[2]) * (16*16)
        symbol_count += int(data[1]) * (16*16*16*16)
        symbol_count += int(data[0]) * (16*16*16*16*16*16)
        
        print ("1st 4 bytes", data[0:4], symbol_count)

        ndx = 0
        while ndx < len(data):
            ndx += 1
        
