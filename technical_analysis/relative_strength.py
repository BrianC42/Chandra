'''
Created on Jan 31, 2018

@author: Brian

http://www.investopedia.com/ask/answers/06/relativestrength.asp?ad=dirN&qo=investopediaSiteSearch&qsrc=0&o=40186
What is relative strength?

Relative strength is a measure of the price trend of a stock or other financial instrument 
compared to another stock, instrument or industry. It is calculated by taking the price 
of one asset and dividing it by another.

For example, if the price of Ford shares is $7 and the price of GM shares is $25, the 
relative strength of Ford to GM is 0.28 ($7/25). This number is given context when it is 
compared to the previous levels of relative strength. If, for example, the relative 
strength of Ford to GM ranges between 0.5 and 1 historically, the current level of 0.28 
suggests that Ford is undervalued or GM is overvalued, or a mix of both. The reason we 
know this is because the only way for this ratio to increase back to its normal historical 
range is for the numerator (number on the top of the ratio, in this case the price of Ford) 
to increase, or the denominator (number on the bottom of the ratio, in our case the price of GM) 
to decrease. It should also be noted that the ratio can also increase by combining an 
upward price move of Ford with a downward price move of GM. For example, if Ford shares 
rose to $14 and GM shares fell to $20, the relative strength would be 0.7, which is near 
the middle of the historic trading range.

It is by comparing the relative strengths of two companies that a trading opportunity, 
known as pairs trading, is realized. Pairs trading is a strategy in which a trader 
matches long and short positions of two stocks that are perceived to have a strong correlation 
to each other and are currently trading outside of their historical relative strength range. 
For example, in the case of the Ford/GM relative strength at 0.28, a pairs trader would enter 
a long position in Ford and short GM if he or she felt the pair would move back toward its historical range.

'''

def relative_strength(df_src=None):
    # print ("relative_strength")
    return df_src
