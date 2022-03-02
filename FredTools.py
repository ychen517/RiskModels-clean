""" See http://api.stlouisfed.org/docs/fred/series_observations.html
observation_start, observation_end, realtime_start, realtime_end: yyyy-mm-dd

units:'lin', 'chg', 'ch1', 'pch', 'pc1', 'pca', 'cch', 'cca', 'log'  
lin = Levels (No transformation)
chg = Change
ch1 = Change from Year Ago
pch = Percent Change
pc1 = Percent Change from Year Ago
pca = Compounded Annual Rate of Change
cch = Continuously Compounded Rate of Change
cca = Continuously Compounded Annual Rate of Change
log = Natural Log

frequency:  'd', 'w', 'bw', 'm', 'q', 'sa', 'a', 'wef', 'weth', 
            'wew', 'wetu', 'wem', 'wesu', 'wesa', 'bwew', 'bwem' 

aggregation_method: 'avg', 'sum', 'eop' 

output_type: An integer that indicates an output type.

    integer, optional, default: 1
    One of the following values: '1', '2', '3', '4'
    1 = Observations by Real-Time Period
    2 = Observations by Vintage Date, All Observations
    3 = Observations by Vintage Date, New and Revised Observations Only
    4 = Observations, Initial Release Only

vintage_dates: comma separated string of YYYY-MM-DD formatted dates in history
"""

import numpy as np
import pandas
import datetime
import dateutil #TODO: replace this, may not be installed.
from optparse import OptionParser
import sys


try:
    import fred
except:
    assert False, 'Please install package: fred.  http://pypi.python.org/pypi/fred'



def _parseFloat_orNaN(valueStr):
    try:
        value=float(valueStr)
    except:
        value=np.nan
    return value

#NOTE: This class is depricated.  It does define the ids of some useful series.  
class _Old_FredDB:
    """ This class fetches data from the Fed/FRED service."""
    US_ZCB_IDS=dict(
        DGS1MO='US ZCB  1m',
        DGS3MO='US ZCB  3m',
        DGS6MO='US ZCB  6m',
        DGS1=  'US ZCB 01y',
        DGS2='US ZCB 02y',
        DGS3='US ZCB 03y',
        DGS5='US ZCB 05y',
        DGS7='US ZCB 07y',
        DGS10='US ZCB 10y',
        DGS20='US ZCB 20y',
        DGS30='US ZCB 30y')

    US_LIBOR_IDS=dict(
        USDONTD156N='US LIBOR Overnight',
        USD1WKD156N='US LIBOR  1w',
        USD2WKD156N='US LIBOR  2w',
        USD1MTD156N='US LIBOR 01m',
        USD2MTD156N='US LIBOR 02m',
        USD3MTD156N='US LIBOR 03m',
        USD4MTD156N='US LIBOR 04m',
        USD5MTD156N='US LIBOR 05m',
        USD6MTD156N='US LIBOR 06m',
        USD7MTD156N='US LIBOR 07m',
        USD8MTD156N='US LIBOR 08m',
        USD9MTD156N='US LIBOR 09m',
        USD10MD156N='US LIBOR 10m',
        USD11MD156N='US LIBOR 11m',
        USD12MD156N='US LIBOR 12m')

    US_TIPS_IDS={'DFII5':'US TIPS 05y CMT Fred',
                 'DFII7':'US TIPS 07y CMT Fred',
                 'DFII10':'US TIPS 10y CMT Fred',
                 'DFII20':'US TIPS 20y CMT Fred ',
                 'DFII30':'US TIPS 30y CMT Fred',
                 'DLTIIT':'US TIPS Long Ave Fred',
                 'DLTA':'US TIPS Vintage Long Ave Fred',
                 'DTP30A28':'US TIPS Vintage 2028 30y 3-5/8',
                 'DTP30A29':'US TIPS Vintage 2029 30y 3-5/8',
                 'DTP3HA32':'US TIPS Vintage 2032 30y 3-3/8',
                 'DTP10L18':'US TIPS Vintage 2007 10y 1-3/8',
                 'DTP10J08':'US TIPS Vintage 2008 10y 3-5/8',
                 'DTP10J09':'US TIPS Vintage 2009 10y 3-7/8',
                 'DTP10J10':'US TIPS Vintage 2010 10y 4-1/4',
                 'DTP10J11':'US TIPS Vintage 2010 10y 3-1/2',
                 'DTP5L02':'US TIPS Vintage 2002 05y'
                 }

    US_SWAP_IDS={'DSWP1':'US Swap Rate 01y',
                 'DSWP2':'US Swap Rate 02y',
                 'DSWP3':'US Swap Rate 03y',
                 'DSWP5':'US Swap Rate 05y',
                 'DSWP7':'US Swap Rate 07y',
                 'DSWP10':'US Swap Rate 10y',
                 'DSWP30':'US Swap Rate 30y'}

    #Commercial Paper
    US_CP_IDS={'DCPF1M':'AA Fiancial CP 1m Rate',
               'DCPF2M':'AA Fiancial CP 2m Rate',
               'DCPF3M':'AA Fiancial CP 3m Rate',
               'DCPN30':'AA Non-Fiancial CP 1m Rate',
               'DCPN2M':'AA Non-Fiancial CP 2m Rate',
               'DCPN3M':'AA Non-Fiancial CP 3m Rate'}

    #THESE dates are weekly.
    US_MORTGAGE_IDS={'MORTGAGE30US':'US 30y Fixed Rate Mortgage',
                     'MORTGAGE15US':'US 15y Fixed Rate Mortgage'}

    #NOTE.  We can also get BAML option adjusted spreads
    US_CORPORATE_YIELD_IDS=dict(
        DAAA='Moodys AAA',
        DBAA='Moodys BAA',
        BAMLC0A0CMEY='BAML Corp MI EY',
        BAMLH0A0HYM2EY='BAML Corp MII EY',
        BAMLC1A0C13YEY='BAML US 1-3y EY',
        BAMLC2A0C35YEY='BAML US 3-5y EY',
        BAMLC3A0C57YEY='BAML US 5-7y EY',
        BAMLC7A0C1015YEY='BAML US 10-15y EY',
        BAMLC0A1CAAAEY='BAML US AAA EY',
        BAMLC0A4CBBBEY='BAML US BBB EY',
        BAMLH0A3HYCEY='BAML US CCC EY',
        BAMLH0A1HYBBEY='BAML US BB EY',
        BAMLH0A2HYBEY='BAML US B EY',
        BAMLC0A2CAAEY='BAML US AA EY',
        BAMLC0A3CAEY='BAML US A EY',
        BAMLEMHBHYCRPIEY='BAML HY-EM+SI EY')

    US_CORPORATE_TR_IDS={
        'BAMLCC0A0CMTRIV': 'BAML US Corp Master TR IdxVal',
        'BAMLCC0A1AAATRIV': 'BAML US Corp AAA TR IdxVal',
        'BAMLCC0A2AATRIV': 'BAML US Corp AA TR IdxVal',
        'BAMLCC0A3ATRIV': 'BAML US Corp A TR IdxVal',
        'BAMLCC0A4BBBTRIV': 'BAML US Corp BBB TR IdxVal',
        'BAMLCC1A013YTRIV': 'BAML US Corp 1-3yr TR IdxVal',
        'BAMLCC2A035YTRIV': 'BAML US Corp 3-5yr TR IdxVal',
        'BAMLCC3A057YTRIV': 'BAML US Corp 5-7yr TR IdxVal',
        'BAMLCC4A0710YTRIV': 'BAML US Corp 7-10yr TR IdxVal',
        'BAMLCC7A01015YTRIV': 'BAML US Corp 10-15yr TR IdxVal',
        'BAMLCC8A015PYTRIV': 'BAML US Corp 15+yr TR IdxVal',
        'BAMLEMCBPITRIV': 'BAML EM Corporate Plus Index TR IdxVal',
        'BAMLEMCLLCRPIUSTRIV': 'BAML US EM Liquid Corporate Plus Index TR IdxVal ',
        'BAMLEMEBCRPIETRIV': 'BAML Euro EM Corp+SI TR IdxVal',
        'BAMLEMELLCRPIEMEAUSTRIV':
            'BAML (EuroMidEastAfrica) US EM Liquid Corp+SI TR IdxVal',
        'BAMLEMFLFLCRPIUSTRIV': 'BAML Financial US EM Liquid Corp+SI TR IdxVal',
        'BAMLEMFSFCRPITRIV': 'BAML Financial EM Corp+SI TR IdxVal',
        'BAMLEMHBHYCRPITRIV': 'BAML High Yield EM Corp+SI TR IdxVal',
        'BAMLEMHGHGLCRPIUSTRIV': 'BAML High Grade US EM Liquid Corp+SI TR IdxVal',
        'BAMLEMHYHYLCRPIUSTRIV': 'BAML High Yield US EM Liquid Corp+SI TR IdxVal',
        'BAMLEMIBHGCRPITRIV': 'BAML High Grade EM Corp+SI TR IdxVal',
        'BAMLEMLLLCRPILAUSTRIV': 'BAML Latin America US EM Liquid Corp+SI TR IdxVal',
        'BAMLEMNFNFLCRPIUSTRIV': 'BAML Non-Financial US EM Liquid Corp+SI TR IdxVal',
        'BAMLEMNSNFCRPITRIV': 'BAML Non-Financial EM Corp+SI TR IdxVal',
        'BAMLEMPBPUBSICRPITRIV': 'BAML Public Sector Issuers EM Corp+SI TR IdxVal',
        'BAMLEMPTPRVICRPITRIV': 'BAML Prvt Sec Issuers EM Corp+SI TR IdxVal',
        'BAMLEMPUPUBSLCRPIUSTRIV': 'BAML Publ Sec Issuers US EM Liquid Corp+SI TR IdxVal',
        'BAMLEMPVPRIVSLCRPIUSTRIV': 'BAML Prvt Sec Issuers US EM Liquid Corp+SI TR IdxVal',
        'BAMLEMRACRPIASIATRIV': 'BAML Asia EM Corp+SI TR IdxVal',
        'BAMLEMRECRPIEMEATRIV': 'BAML (EuroMidEastAfrica) EM Corp+SI TR IdxVal',
        'BAMLEMRLCRPILATRIV': 'BAML Latin America EM Corp+SI TR IdxVal',
        'BAMLEMUBCRPIUSTRIV': 'BAML US EM Corp+SI TR IdxVal',
        'BAMLHYH0A0HYM2TRIV': 'BAML US High Yield Master II TR IdxVal',
        'BAMLHYH0A1BBTRIV': 'BAML US High Yield BB TR IdxVal',
        'BAMLHYH0A2BTRIV': 'BAML US High Yield B TR IdxVal',
        'BAMLHYH0A3CMTRIV': 'BAML US High Yield CCC or Below TR IdxVal'}

    US_ALL_CORPORATE_IDS={
        'BAMLC0A0CM': 'BAML US Corporate Master OA Sprd',
        'BAMLC0A0CMEY': 'BAML US Corporate Master EY',
        'BAMLC0A1CAAA': 'BAML US Corporate AAA OA Sprd',
        'BAMLC0A1CAAAEY': 'BAML US Corporate AAA EY',
        'BAMLC0A2CAA': 'BAML US Corporate AA OA Sprd',
        'BAMLC0A2CAAEY': 'BAML US Corporate AA EY',
        'BAMLC0A3CA': 'BAML US Corporate A OA Sprd',
        'BAMLC0A3CAEY': 'BAML US Corporate A EY',
        'BAMLC0A4CBBB': 'BAML US Corporate BBB OA Sprd',
        'BAMLC0A4CBBBEY': 'BAML US Corporate BBB EY',
        'BAMLC1A0C13Y': 'BAML US Corporate 1-3 Year OA Sprd',
        'BAMLC1A0C13YEY': 'BAML US Corporate 1-3 Year EY',
        'BAMLC2A0C35Y': 'BAML US Corporate 3-5 Year OA Sprd',
        'BAMLC2A0C35YEY': 'BAML US Corporate 3-5 Year EY',
        'BAMLC3A0C57Y': 'BAML US Corporate 5-7 Year OA Sprd',
        'BAMLC3A0C57YEY': 'BAML US Corporate 5-7 Year EY',
        'BAMLC4A0C710Y': 'BAML US Corporate 7-10 Year OA Sprd',
        'BAMLC4A0C710YEY': 'BAML US Corporate 7-10 Year EY',
        'BAMLC7A0C1015Y': 'BAML US Corporate 10-15 Year OA Sprd',
        'BAMLC7A0C1015YEY': 'BAML US Corporate 10-15 Year EY',
        'BAMLC8A0C15PY': 'BAML US Corporate 15+ Year OA Sprd',
        'BAMLC8A0C15PYEY': 'BAML US Corporate 15+ Year EY',
        'BAMLCC0A0CMTRIV': 'BAML US Corp Master TR IdxVal',
        'BAMLCC0A1AAATRIV': 'BAML US Corp AAA TR IdxVal',
        'BAMLCC0A2AATRIV': 'BAML US Corp AA TR IdxVal',
        'BAMLCC0A3ATRIV': 'BAML US Corp A TR IdxVal',
        'BAMLCC0A4BBBTRIV': 'BAML US Corp BBB TR IdxVal',
        'BAMLCC1A013YTRIV': 'BAML US Corp 1-3yr TR IdxVal',
        'BAMLCC2A035YTRIV': 'BAML US Corp 3-5yr TR IdxVal',
        'BAMLCC3A057YTRIV': 'BAML US Corp 5-7yr TR IdxVal',
        'BAMLCC4A0710YTRIV': 'BAML US Corp 7-10yr TR IdxVal',
        'BAMLCC7A01015YTRIV': 'BAML US Corp 10-15yr TR IdxVal',
        'BAMLCC8A015PYTRIV': 'BAML US Corp 15+yr TR IdxVal',
        'BAMLEM1BRRAAA2ACRPIEY': 'BAML AAA-A EM Corp+SI EY',
        'BAMLEM1BRRAAA2ACRPIOAS': 'BAML AAA-A EM Corp+SI OA Sprd',
        'BAMLEM1BRRAAA2ACRPITRIV': 'BAML AAA-A EM Corp+SI TR IdxVal',
        'BAMLEM1RAAA2ALCRPIUSEY': 'BAML AAA-A US EM Liquid Corp+SI EY',
        'BAMLEM1RAAA2ALCRPIUSOAS': 'BAML AAA-A US EM Liquid Corp+SI OA Sprd',
        'BAMLEM1RAAA2ALCRPIUSTRIV': 'BAML AAA-A US EM Liquid Corp+SI TR IdxVal',
        'BAMLEM2BRRBBBCRPIEY': 'BAML BBB EM Corp+SI EY',
        'BAMLEM2BRRBBBCRPIOAS': 'BAML BBB EM Corp+SI OA Sprd',
        'BAMLEM2BRRBBBCRPITRIV': 'BAML BBB EM Corp+SI TR IdxVal',
        'BAMLEM2RBBBLCRPIUSEY': 'BAML BBB US EM Liquid Corp+SI EY',
        'BAMLEM2RBBBLCRPIUSOAS': 'BAML BBB US EM Liquid Corp+SI OA Sprd',
        'BAMLEM2RBBBLCRPIUSTRIV': 'BAML BBB US EM Liquid Corp+SI TR IdxVal',
        'BAMLEM3BRRBBCRPIEY': 'BAML BB EM Corp+SI EY',
        'BAMLEM3BRRBBCRPIOAS': 'BAML BB EM Corp+SI OA Sprd',
        'BAMLEM3BRRBBCRPITRIV': 'BAML BB EM Corp+SI TR IdxVal',
        'BAMLEM3RBBLCRPIUSEY': 'BAML BB US EM Liquid Corp+SI EY',
        'BAMLEM3RBBLCRPIUSOAS': 'BAML BB US EM Liquid Corp+SI OA Sprd',
        'BAMLEM3RBBLCRPIUSTRIV': 'BAML BB US EM Liquid Corp+SI TR IdxVal',
        'BAMLEM4BRRBLCRPIEY': 'BAML B and Lower EM Corp+SI EY',
        'BAMLEM4BRRBLCRPIOAS': 'BAML B and Lower EM Corp+SI OA Sprd',
        'BAMLEM4BRRBLCRPITRIV': 'BAML B and Lower EM Corp+SI TR IdxVal',
        'BAMLEM4RBLLCRPIUSEY': 'BAML B & Lower US EM Liquid Corp+SI EY',
        'BAMLEM4RBLLCRPIUSOAS': 'BAML B & Lower US EM Liquid Corp+SI OA Sprd',
        'BAMLEM4BRRBLCRPIOAS': 'BAML B and Lower EM Corp+SI OA Sprd',
        'BAMLEM4BRRBLCRPITRIV': 'BAML B and Lower EM Corp+SI TR IdxVal',
        'BAMLEM4RBLLCRPIUSEY': 'BAML B & Lower US EM Liquid Corp+SI EY',
        'BAMLEM4RBLLCRPIUSOAS': 'BAML B & Lower US EM Liquid Corp+SI OA Sprd',
        'BAMLEM4RBLLCRPIUSTRIV': 'BAML B & Lower US EM Liquid Corp+SI TR IdxVal',
        'BAMLEM5BCOCRPIEY': 'BAML Crossover EM Corp+SI EY',
        'BAMLEM5BCOCRPIOAS': 'BAML Crossover EM Corp+SI OA Sprd',
        'BAMLEM5BCOCRPITRIV': 'BAML Crossover EM Corp+SI TR IdxVal',
        'BAMLEMALLCRPIASIAUSEY': 'BAML Asia US EM Liquid Corp+SI EY',
        'BAMLEMALLCRPIASIAUSOAS': 'BAML Asia US EM Liquid Corp+SI OA Sprd',
        'BAMLEMALLCRPIASIAUSTRIV': 'BAML Asia US EM Liquid Corp+SI TR IdxVal',
        'BAMLEMCBPIEY': 'BAML EM Corporate Plus Index EY',
        'BAMLEMCBPIOAS': 'BAML EM Corporate Plus Index OA Sprd',
        'BAMLEMCBPITRIV': 'BAML EM Corporate Plus Index TR IdxVal',
        'BAMLEMCLLCRPIUSEY': 'BAML US EM Liquid Corporate Plus Index EY',
        'BAMLEMCLLCRPIUSOAS': 'BAML US EM Liquid Corporate Plus Index OA Sprd',
        'BAMLEMCLLCRPIUSTRIV': 'BAML US EM Liquid Corporate Plus Index TR IdxVal ',
        'BAMLEMEBCRPIEEY': 'BAML Euro EM Corp+SI EY',
        'BAMLEMEBCRPIEOAS': 'BAML Euro EM Corp+SI OA Sprd',
        'BAMLEMEBCRPIETRIV': 'BAML Euro EM Corp+SI TR IdxVal',
        'BAMLEMELLCRPIEMEAUSEY':'BAML (EuroMidEastAfrica) US EM Liquid Corp+SI EY',
        'BAMLEMELLCRPIEMEAUSOAS':'BAML (EuroMidEastAfrica)) US EM Liquid Corp+SI OA Sprd',
        'BAMLEMELLCRPIEMEAUSTRIV':'BAML (EuroMidEastAfrica) US EM Liquid Corp+SI TR IdxVal',
        'BAMLEMFLFLCRPIUSEY': 'BAML Financial US EM Liquid Corp+SI EY',
        'BAMLEMFLFLCRPIUSOAS': 'BAML Financial US EM Liquid Corp+SI OA Sprd',
        'BAMLEMFLFLCRPIUSTRIV': 'BAML Financial US EM Liquid Corp+SI TR IdxVal',
        'BAMLEMFSFCRPIEY': 'BAML Financial EM Corp+SI EY',
        'BAMLEMFSFCRPIOAS': 'BAML Financial EM Corp+SI OA Sprd',
        'BAMLEMFSFCRPITRIV': 'BAML Financial EM Corp+SI TR IdxVal',
        'BAMLEMHBHYCRPIEY': 'BAML High Yield EM Corp+SI EY',
        'BAMLEMHBHYCRPIOAS': 'BAML High Yield EM Corp+SI OA Sprd',
        'BAMLEMHBHYCRPITRIV': 'BAML High Yield EM Corp+SI TR IdxVal',
        'BAMLEMHGHGLCRPIUSEY': 'BAML High Grade US EM Liquid Corp+SI EY',
        'BAMLEMHGHGLCRPIUSOAS': 'BAML High Grade US EM Liquid Corp+SI OA Sprd',
        'BAMLEMHGHGLCRPIUSTRIV': 'BAML High Grade US EM Liquid Corp+SI TR IdxVal',
        'BAMLEMHYHYLCRPIUSEY': 'BAML High Yield US EM Liquid Corp+SI EY',
        'BAMLEMHYHYLCRPIUSOAS': 'BAML High Yield US EM Liquid Corp+SI OA Sprd',
        'BAMLEMHYHYLCRPIUSTRIV': 'BAML High Yield US EM Liquid Corp+SI TR IdxVal',
        'BAMLEMIBHGCRPIEY': 'BAML High Grade EM Corp+SI EY',
        'BAMLEMIBHGCRPIOAS': 'BAML High Grade EM Corp+SI OA Sprd',
        'BAMLEMIBHGCRPITRIV': 'BAML High Grade EM Corp+SI TR IdxVal',
        'BAMLEMLLLCRPILAUSEY': 'BAML Latin America US EM Liquid Corp+SI EY',
        'BAMLEMLLLCRPILAUSOAS': 'BAML Latin America US EM Liquid Corp+SI OA Sprd',
        'BAMLEMLLLCRPILAUSTRIV': 'BAML Latin America US EM Liquid Corp+SI TR IdxVal',
        'BAMLEMNFNFLCRPIUSEY': 'BAML Non-Financial US EM Liquid Corp+SI EY',
        'BAMLEMNFNFLCRPIUSOAS': 'BAML Non-Financial US EM Liquid Corp+SI OA Sprd',
        'BAMLEMNFNFLCRPIUSTRIV': 'BAML Non-Financial US EM Liquid Corp+SI TR IdxVal',
        'BAMLEMNSNFCRPIEY': 'BAML Non-Financial EM Corp+SI EY',
        'BAMLEMNSNFCRPIOAS': 'BAML Non-Financial EM Corp+SI OA Sprd',
        'BAMLEMNSNFCRPITRIV': 'BAML Non-Financial EM Corp+SI TR IdxVal',
        'BAMLEMPBPUBSICRPIEY': 'BAML Public Sector Issuers EM Corp+SI EY',
        'BAMLEMPBPUBSICRPIOAS': 'BAML Public Sector Issuers EM Corp+SI OA Sprd',
        'BAMLEMPBPUBSICRPITRIV': 'BAML Public Sector Issuers EM Corp+SI TR IdxVal',
        'BAMLEMPTPRVICRPIEY': 'BAML Prvt Sec Issuers EM Corp+SI EY',
        'BAMLEMPTPRVICRPIOAS': 'BAML Prvt Sec Issuers EM Corp+SI OA Sprd',
        'BAMLEMPTPRVICRPITRIV': 'BAML Prvt Sec Issuers EM Corp+SI TR IdxVal',
        'BAMLEMPUPUBSLCRPIUSEY': 'BAML Pub Sec Issuers US EM Liquid Corp+SI EY',
        'BAMLEMPUPUBSLCRPIUSOAS': 'BAML Pub Sec Issuers US EM Liquid Corp+SI OA Sprd',
        'BAMLEMPTPRVICRPITRIV': 'BAML Prvt Sec Issuers EM Corp+SI TR IdxVal',
        'BAMLEMPUPUBSLCRPIUSEY': 'BAML Pub Sec Issuers US EM Liquid Corp+SI EY',
        'BAMLEMPUPUBSLCRPIUSOAS': 'BAML Pub Sec Issuers US EM Liquid Corp+SI OA Sprd',
        'BAMLEMPUPUBSLCRPIUSTRIV': 'BAML Publ Sec Issuers US EM Liquid Corp+SI TR IdxVal',
        'BAMLEMPVPRIVSLCRPIUSEY': 'BAML Prvt Sec Issuers US EM Liquid Corp+SI EY',
        'BAMLEMPVPRIVSLCRPIUSOAS': 'BAML Prvt Sec Issuers US EM Liquid Corp+SI OA Sprd',
        'BAMLEMPVPRIVSLCRPIUSTRIV': 'BAML Prvt Sec Issuers US EM Liquid Corp+SI TR IdxVal',
        'BAMLEMRACRPIASIAEY': 'BAML Asia EM Corp+SI EY',
        'BAMLEMRACRPIASIAOAS': 'BAML Asia EM Corp+SI OA Sprd',
        'BAMLEMRACRPIASIATRIV': 'BAML Asia EM Corp+SI TR IdxVal',
        'BAMLEMRECRPIEMEAEY': 'BAML (EuroMidEastAfrica) EM Corp+SI EY',
        'BAMLEMRECRPIEMEAOAS': 'BAML (EuroMidEastAfrica) EM Corp+SI OA Sprd',
        'BAMLEMRECRPIEMEATRIV': 'BAML (EuroMidEastAfrica) EM Corp+SI TR IdxVal',
        'BAMLEMRLCRPILAEY': 'BAML Latin America EM Corp+SI EY',
        'BAMLEMRLCRPILAOAS': 'BAML Latin America EM Corp+SI OA Sprd',
        'BAMLEMRLCRPILATRIV': 'BAML Latin America EM Corp+SI TR IdxVal',
        'BAMLEMUBCRPIUSEY': 'BAML US EM Corp+SI EY',
        'BAMLEMUBCRPIUSOAS': 'BAML US EM Corp+SI OA Sprd',
        'BAMLEMUBCRPIUSTRIV': 'BAML US EM Corp+SI TR IdxVal',
        'BAMLEMXOCOLCRPIUSEY': 'BAML Crossover US EM Liquid Corp+SI EY',
        'BAMLEMXOCOLCRPIUSOAS': 'BAML Crossover US EM Liquid Corp+SI OA Sprd',
        'BAMLEMXOCOLCRPIUSTRIV': 'BAML Crossover US EM Liquid Corp+SI TR IdxVal',
        'BAMLH0A0HYM2': 'BAML US High Yield Master II OA Sprd',
        'BAMLH0A0HYM2EY': 'BAML US High Yield Master II EY',
        'BAMLH0A1HYBB': 'BAML US High Yield BB OA Sprd',
        'BAMLH0A1HYBBEY': 'BAML US High Yield BB EY',
        'BAMLH0A2HYB': 'BAML US High Yield B OA Sprd',
        'BAMLH0A2HYBEY': 'BAML US High Yield B EY',
        'BAMLH0A3HYC': 'BAML US High Yield CCC or Below OA Sprd',
        'BAMLH0A3HYCEY': 'BAML US High Yield CCC or Below EY',
        'BAMLHYH0A0HYM2TRIV': 'BAML US High Yield Master II TR IdxVal',
        'BAMLHYH0A1BBTRIV': 'BAML US High Yield BB TR IdxVal',
        'BAMLHYH0A2BTRIV': 'BAML US High Yield B TR IdxVal',
        'BAMLHYH0A3CMTRIV': 'BAML US High Yield CCC or Below TR IdxVal'}




    def __init__(self):
        """ """
        import fred
        #self.FRED_API_KEY=fred.os.environ['FRED_API_KEY']
        #TODO: this is a personal key.  
        #WARNING: avoid legal issues before using this for anything serious
        self.FRED_API_KEY='03e9ac6bfe07e3c67725de2d8fb0a887'


    def getFredSeries(self,seriesName,use_title=False,units='lin'):
        """ This is the main method.
        units must be:  cap, cca, cch, ch1, chg, lin, log, pc1, pca, pch """
        import fred
        from  dateutil.parser import parse
        fred.key(self.FRED_API_KEY)

        s=fred.observations(seriesName,units=units)
        ss=s['observations']
        sss=ss['observation']
        sssDict=dict( (parse(i['date']),_parseFloat_orNaN(i['value'])) for i in sss)
        series=pandas.Series(sssDict)
        if use_title:
            series.name=fred.series(seriesName)['title']
        else:
            series.name=seriesName
        return series

    def getFredObj(self):
        import fred
        from  dateutil.parser import parse
        return fred.api.Fred(api_key=self.FRED_API_KEY)

    def findSeries(self,search_text):
        ff=self.getFredObj()
        df=pandas.DataFrame(
            ff.series('search',
                      search_text=search_text)['seriess']['series'])

        return df.rename(index=(df['id'].to_dict()))



    def getSeriesTitles(self,seriesNames):
        import fred
        from  dateutil.parser import parse
        fred.key(self.FRED_API_KEY)
        return dict( [(sn,fred.series(sn)['title']) for sn in seriesNames ] )

    def getSeriesInfo(self,seriesName):
        import fred
        from  dateutil.parser import parse
        fred.key(self.FRED_API_KEY)
        return fred.series(seriesName)


    def getFredDF(self,seriesNames):
        n2s={}

        for n in seriesNames:
            try:
                n2s[n]=self.getFredSeries(n)
            except:
                print('bad series name: ', end=' ')

        df=pandas.DataFrame(n2s).rename(index=lambda d: d.date())

        return df




class FredDB(object):

    def __init__(self):
        """ """
        #self.FRED_API_KEY=fred.os.environ['FRED_API_KEY']
        #TODO: this is a personal key.  
        #WARNING: avoid legal issues before using this for anything serious
        self.FRED_API_KEY='03e9ac6bfe07e3c67725de2d8fb0a887'
        fred.key(self.FRED_API_KEY)

    def getVintageDates(self,series_id):
        vin=fred.vintage(series_id)
        return sorted([dateutil.parser.parse(d).date() for d in vin['vintage_dates']['vintage_date']])



    def _processRawSeriesRow(self,rowDict):
        return dict(date=dateutil.parser.parse(rowDict['date']).date(),
                    realtime_start=dateutil.parser.parse(rowDict['realtime_start']).date(),
                    realtime_end=dateutil.parser.parse(rowDict['realtime_end']).date(),
                    value=_parseFloat_orNaN(rowDict['value']))

    def getRawSeries(self,series_id,**kwargs):
        """This a thin wrapper around the fred interface."""
        raw_output=fred.observations(series_id,**kwargs)

        df=pandas.DataFrame([self._processRawSeriesRow(rowDict) 
            for rowDict in raw_output['observations']['observation']]).rename(index=lambda i: str(i))
        
        return df
        
    
    def getSimpleSeries(self,series_id,units='lin',
                        observation_start=datetime.date(1970,1,1),
                        get_latest=True):
        """A simplified interface.
        units:'lin', 'chg', 'ch1', 'pch', 'pc1', 'pca', 'cch', 'cca', 'log'  
        lin = Levels (No transformation)
        chg = Change
        ch1 = Change from Year Ago
        pch = Percent Change
        pc1 = Percent Change from Year Ago
        pca = Compounded Annual Rate of Change
        cch = Continuously Compounded Rate of Change
        cca = Continuously Compounded Annual Rate of Change
        log = Natural Log

        get_latest=True/False ==> most recent revision/initial release

        """
        if(get_latest):
            valuesIdx=-1
        else:
            valuesIdx=0
        if units!='lin':
            df=self.getRawSeries(series_id,units=units,observation_start=str(observation_start))
        else:
            df=self.getRawSeries(series_id,observation_start=str(observation_start))
            print('no units')
        gb=df[['date','value']].groupby('date')
        series=pandas.Series( dict( (i[0],i[1]['value'].values[valuesIdx]) for i in  gb ) )
        series.name=series_id
        return series
                        
    def getSeriesTitles(self,seriesNames):
        from  dateutil.parser import parse
        return dict( [(sn,fred.series(sn)['seriess']['series']['title']) for sn in seriesNames ] )

    def getSeriesInfo(self,seriesName):
        return fred.series(seriesName)

def _old_doTest():
    #print 'Test not implemented'
    seriesIds="unrate payems bopgstb houst nmfinsi mich umcsent napm indpro".split()
    
    kwargsDefault=dict(observation_start='2010-01-01',units='lin',realtime_start='1900-01-01')

    fdb=FredDB()

    seriesDict={}
    dfDict={}
    for units in ['lin', 'chg', 'ch1', 'pch', 'pc1', 'cca']:  
        seriesDict[units]={}
        for series_id in seriesIds:
            try:
                seriesDict[units][series_id]=fdb.getSimpleSeries(series_id,units=units)
            except:
                print('no %s for series %s' % (units,series_id))
        df=pandas.DataFrame(seriesDict[units])
        dfDict[units]=df

    dfTitles=fdb.getSeriesTitles(df.columns)

"""
Note: this is a useful way to find series by name.

results=fred.Fred().series('search',search_text="Recession")['seriess']['series']
tt=dict( (k['id'],k['title']) for k in results if k['frequency'].find('Daily')>=0)
pandas.Series(tt).to_csv('../csv/fredDailyRecessionTitles.csv')
"""

if __name__=='__main__':

    #if argv == None:
    argv = sys.argv

    usage = 'usage: %prog [options] series_id1[:units] series_id2[:units] ...\n'
    parser = OptionParser(usage=usage)
    parser.add_option("--start", dest="observationStart", default='1970-01-01',
                      help="Date YYYY-MM-DD (default: %default))")
    parser.add_option("--output_csv", '-o', dest="OutputCSV", default='fedTS.csv',
                      help="Comma separated list series_ids (default: %default)")
    parser.add_option("--get_latest", dest="getLatest", default=1, type='int',
                      help="Use most recent revision or earliest release (default: %default)")
    parser.add_option("--units", dest="units", default='lin',
                      help="One of:lin,chg,ch1,pch,pc1,pca,cch,cca,log (default: %default)")
    (cmdoptions, args) = parser.parse_args(argv)
    
    #print cmdoptions
    #print args

    assert len(args)>1, 'need at least one fed/fred series_id. ' \
        'See http://research.stlouisfed.org/fred2/'

    seriesIds = args[1:]
    output_csv=cmdoptions.OutputCSV
    observation_start=cmdoptions.observationStart
    get_latest=cmdoptions.getLatest
    
    assert get_latest==1 or get_latest==0, 'bad value for get_latest: %s' %get_latest
    

    seriesIdsDict={}
    for sid in seriesIds:
        kv = sid.split(':')
        k=kv[0]
        if len(kv)<2:
            v=cmdoptions.units
        else:
            v=kv[1]
        seriesIdsDict[k]=v
            
    
    #TODO: add series description, frequency, and unit somewhere.  
    fdb=FredDB()
    seriesDict={}
    for series_id,units in seriesIdsDict.items():
        try:
            seriesDict[series_id]=fdb.getSimpleSeries(series_id,units=units,
                                                      observation_start=observation_start)
        except:
            print('Warning: series %s failed with units %s ' % (series_id, units))
    df=pandas.DataFrame(seriesDict)
    df.to_csv(output_csv)
    dfTitles=fdb.getSeriesTitles(df.columns)
    print("#Information about downloaded time series:")
    for k,v in dfTitles.items():
        print(k,'\n    ', v)
        

#A note on conventions:
"""
Note that because FRED uses levels and rounded data as published by the source, calculations of percentage changes and/or growth rates in some series may not be identical to those in the original releases.
The following formulas are used:

Change
x(t) - x(t-1)

Change from Year Ago
x(t) - x(t-n_obs_per_yr)

Percent Change
((x(t)/x(t-1)) - 1) * 100

Percent Change from Year Ago
((x(t)/x(t-n_obs_per_yr)) - 1) * 100

Compounded Annual Rate of Change
(((x(t)/x(t-1)) ** (n_obs_per_yr)) - 1) * 100

Continuously Compounded Rate of Change
(ln(x(t)) - ln(x(t-1))) * 100

Continuously Compounded Annual Rate of Change
((ln(x(t)) - ln(x(t-1))) * 100) * n_obs_per_yr

Natural Log
ln(x(t))

Notes:

'x(t)' is the value of series x at time period t.

'n_obs_per_yr' is the number of observations per year. The number of observations per year differs by frequency:

Daily, 260 (no values on weekends)
Annual, 1
Monthly, 12
Quarterly, 4
Bi-Weekly, 26
Weekly,52

"""
