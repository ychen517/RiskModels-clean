
import logging
import pprint
import datetime
import optparse
import subprocess
import os
import csv
from copy import deepcopy
from subprocess import Popen, PIPE
import time
from riskmodels import Utilities
from marketdb import MarketDB
from riskmodels import ModelDB

pp = pprint.PrettyPrinter(indent=2)
csv.register_dialect('pipe', delimiter='|', quoting=csv.QUOTE_NONE)

class Series(): 
    headers = {}
    types = {}
    units = {}
    '''#Columns: future_family_id    asset_type    asset_class    asset_sub_class    name    ticker    currency    trading_country    stock_exchange    expiration_periodicity    index_official_name    distributed_index_name    composite_axioma_id    contract_size    five_day_adv    twenty_day_adv    five_day_aoi    twenty_day_aoi
    '''
    '''
    #Type: ID: Attribute    Attribute    Attribute    Attribute    Attribute    Attribute    Attribute    Attribute    Attribute    Attribute    Attribute    Attribute    Attribute    Attribute    Attribute    Attribute    Attribute
    '''
    '''
    #Unit: ID    Text    Text    Text    Text    Text    Text    Text    Text    Text    Text    Text    Text    Number    Number    Number    Number    Number
    '''
    headers['seriesAttributes'] = ['future_family_id', 'asset_type', 'asset_class', 'asset_sub_class', 'name', 'ticker', 'currency', 'trading_country', 'stock_exchange', 'expiration_periodicity', 'index_official_name', 'distributed_index_name', 'composite_axioma_id', 'contract_size', 'five_day_adv', 'twenty_day_adv', 'five_day_aoi', 'twenty_day_aoi']
    types['seriesAttributes'] = ['ID', 'Attribute', 'Attribute', 'Attribute', 'Attribute', 'Attribute', 'Attribute', 'Attribute', 'Attribute', 'Attribute', 'Attribute', 'Attribute', 'Attribute', 'Attribute', 'Attribute', 'Attribute', 'Attribute', 'Attribute']
    units['seriesAttributes'] = ['ID', 'Text', 'Text', 'Text', 'Text', 'Text', 'Text', 'Text', 'Text', 'Text', 'Text', 'Text', 'Text', 'Number', 'Number', 'Number', 'Number', 'Number']

    '''#Columns: contract_id    future_family_id    asset_type    asset_class    asset_sub_class    ticker    sedol    cusip    isin    name    last_trading_date    price    price_roll_over_flag    return    volume    open_interest
    '''
    '''
    #Type: ID    ID    Attribute    Attribute    Attribute    Attribute    Attribute    Attribute    Attribute    Attribute    Attribute    Attribute    Set    Attribute    Attribute    Attribute
    '''
    '''
    #Unit: ID    ID    Text    Text    Text    Text    Text    Text    Text    Text    Date    CurrencyPerShare    NA    Number    Number    Number
    '''
    headers['contractAttributes'] = ['contract_id', 'future_family_id', 'asset_type', 'asset_class', 'asset_sub_class', 'ticker', 'sedol', 'cusip', 'isin', 'name', 'last_trading_date', 'price', 'price_roll_over_flag', 'return', 'volume', 'open_interest']
    types['contractAttributes'] = ['ID', 'ID', 'Attribute', 'Attribute', 'Attribute', 'Attribute', 'Attribute', 'Attribute', 'Attribute', 'Attribute', 'Attribute', 'Attribute', 'Set', 'Attribute', 'Attribute', 'Attribute']
    units['contractAttributes'] = ['ID', 'ID', 'Text', 'Text', 'Text', 'Text', 'Text', 'Text', 'Text', 'Text', 'Date', 'CurrencyPerShare', 'NA', 'Number', 'Number', 'Number']

    def __init__(self, axiomaId, extractor):
        self.new = {}  
        self.axiomaId = axiomaId
        self.extractor = extractor
        self.name = None
        self.externalTicker = None
        self.currencyId = None
        self.currencyCode = None
        self.exchangeName = None
        self.tradingCountry = None
        
        self.future_family_id = self.axiomaId        
        self.asset_type = None           
        self.asset_class = None            
        self.asset_sub_class = None          
        self.name = None          
        self.ticker = None       
        self.currency = None           
        self.trading_country = None          
        self.stock_exchange = None         
        self.expiration_periodicity = None        
        self.index_official_name = None         
        self.distributed_index_name = None           
        self.composite_axioma_id = None        
        self.contract_size = None         
        self.mic_code = None
        self.bbg_ticker = None
        self.index_type = None
        self.index_currency = None

        self.contracts = {}
        self.rows = {}
        for fileName in extractor.fileNames.keys():
            self.rows[fileName] = []
        self.classificationNames = {}

    def __str__(self):
        res =  'Series(axiomaId:%s, name:%s, externalTicker:%s, currencyId:%s, tradingCountry:%s)' % (
            self.axiomaId, self.name, self.externalTicker, self.currencyId, self.tradingCountry)
        return res
    def __repr__(self):
        return self.__str__()

       
    def getClassificationNames(self):
        q = '''
            select id,name From classification_Ref where revision_ID =  17 and id in (5357,5355,5356,5359)
        '''
        self.extractor.dbCursor.execute(q)
        res = self.extractor.dbCursor.fetchall()
        for each in res:
            self.classificationNames[each[0]] = each[1]
        
    def getCurrencyCode(self):
        q = '''
            select Code from currency_ref where id = :id_arg 
            and from_dt <= :dt_arg
            AND thru_dt > :dt_arg
        '''
        self.extractor.dbCursor.execute(q, id_arg=self.currencyId, dt_arg=self.extractor.date)
        res = self.extractor.dbCursor.fetchall()
        for each in res:
            self.currencyCode = each[0]

    def getExchangeName(self):
        q = '''
            select name from classification_ref where id = :id_arg 
        '''
        self.extractor.dbCursor.execute(q, id_arg=self.exchange)
        res = self.extractor.dbCursor.fetchall()
        for each in res:
            self.exchangeName = each[0]

    def getContracts(self):
        qold = '''
            select FUTURE_AXIOMA_ID, LAST_TRADING_DATE from future_linkage_active
            where FUTURE_FAMILY_ID = :id_arg and FUTURE_AXIOMA_ID in 
            (select MARKETDB_ID from modeldb_global.future_issue_map where from_DT <= :dt_arg and thru_DT > :dt_arg and DISTRIBUTE = 'Y')
        '''
        q = '''
            select FUTURE_AXIOMA_ID, LAST_TRADING_DATE, im.MODELDB_ID from future_linkage_active, modeldb_global.future_issue_map im
            where FUTURE_FAMILY_ID = :id_arg and FUTURE_AXIOMA_ID = MARKETDB_ID
            and im.from_DT <= :dt_arg and im.thru_DT > :dt_arg and im.DISTRIBUTE = 'Y'
        '''
        self.extractor.dbCursor.execute(q, id_arg=self.axiomaId, dt_arg=self.extractor.date)
        res = self.extractor.dbCursor.fetchall()
        for each in res:
            c = Contract(self, each[0], each[1])
            c.modeldb_id = each[2][1:]
            self.contracts[c.axiomaId] = c

    def getTransferContracts(self):
        q = '''
            select FUTURE_AXIOMA_ID, LAST_TRADING_DATE from future_linkage_active
            where FUTURE_FAMILY_ID = :id_arg
        '''
        self.extractor.dbCursor.execute(q, id_arg=self.axiomaId)
        res = self.extractor.dbCursor.fetchall()
#        print res
        for each in res:
            c = Contract(self, each[0], each[1])
            self.contracts[c.axiomaId] = c
            
    def getContractTickers(self):
        q = '''
            select AXIOMA_ID, ID from future_dim_ticker_active_int
            where AXIOMA_ID in (select FUTURE_AXIOMA_ID from future_linkage_active
            where FUTURE_FAMILY_ID = :id_arg and FUTURE_AXIOMA_ID in 
            (select MARKETDB_ID from modeldb_global.future_issue_map where from_DT <= :dt_arg and thru_DT > :dt_arg and DISTRIBUTE = 'Y'))
            and from_DT <= :dt_arg and thru_DT > :dt_arg
        '''
        self.extractor.dbCursor.execute(q, id_arg=self.axiomaId, dt_arg=self.extractor.date)
        res = self.extractor.dbCursor.fetchall()
        for each in res:
            c = self.contracts[each[0]]
            c.ticker = each[1]
            c.sedol = c.ticker
            c.cusip = c.ticker
            c.isin = c.ticker

    def getContractNames(self):
        q = '''
            select AXIOMA_ID, ID from future_dim_name_active_int
            where AXIOMA_ID in (select FUTURE_AXIOMA_ID from future_linkage_active
            where FUTURE_FAMILY_ID = :id_arg and FUTURE_AXIOMA_ID in 
            (select MARKETDB_ID from modeldb_global.future_issue_map where from_DT <= :dt_arg and thru_DT > :dt_arg and DISTRIBUTE = 'Y'))
            and from_DT <= :dt_arg and thru_DT > :dt_arg
        '''
        self.extractor.dbCursor.execute(q, id_arg=self.axiomaId, dt_arg=self.extractor.date)
        res = self.extractor.dbCursor.fetchall()
        for each in res:
            self.contracts[each[0]].name = each[1]

    def getContractBloombergTickers(self):
        q = '''
            select AXIOMA_ID, ID || ' Index' from future_dim_bb_ticker_act_int
            where AXIOMA_ID in (select FUTURE_AXIOMA_ID from future_linkage_active
            where FUTURE_FAMILY_ID = :id_arg and FUTURE_AXIOMA_ID in 
            (select MARKETDB_ID from modeldb_global.future_issue_map where from_DT <= :dt_arg and thru_DT > :dt_arg and DISTRIBUTE = 'Y'))
            and from_DT <= :dt_arg and thru_DT > :dt_arg
        '''
        self.extractor.dbCursor.execute(q, id_arg=self.axiomaId, dt_arg=self.extractor.date)
        res = self.extractor.dbCursor.fetchall()
        for each in res:
            self.contracts[each[0]].bbg_ticker = each[1]

    def getContractPrices(self):
 
        q = '''
            select VALUE, DT, PRICE_MARKER from future_dim_ucp_active
            where AXIOMA_ID = :id_arg and 
            DT = :dt_arg
        '''
        qPrev = '''
            select VALUE, DT from future_dim_ucp_active
            where AXIOMA_ID = :id_arg and 
            dt in (select maX(dt) from future_dim_ucp_active where  axioma_id = :id_arg and value > 0 and dt < :dt_arg)
        '''

        for k,c in self.contracts.items():
            self.extractor.dbCursor.execute(q, id_arg=k, dt_arg=self.extractor.extractDate)
            res = self.extractor.dbCursor.fetchall()
            c.price = None
            for each in res:
                if each[0] is not None and each[0] > 0:
                    c.price = each[0]
                    c.price_roll_over_flag = ''
                    if each[2] == 3:
                        c.price_roll_over_flag = '*'
                    else:
                        c.price_roll_over_flag = ''
#                    c.last_trading_date = each[1]
            if c.price is None:
                self.extractor.dbCursor.execute(qPrev, id_arg=k, dt_arg=self.extractor.extractDate)
                res = self.extractor.dbCursor.fetchall()
                for each in res:
                    if each[0] is not None and each[0] > 0:                        
                            c.price = each[0]
                            c.price_roll_over_flag = '*'
                            c.price_return = '0'
#                            c.last_trading_date = each[1]
 
    def getContractVolumes(self):
 
        q = '''
            select VALUE, DT from future_dim_tdv_active
            where AXIOMA_ID = :id_arg and 
            DT = :dt_arg
        '''
        qPrev = '''
            select VALUE, DT from future_dim_tdv_active
            where AXIOMA_ID = :id_arg and 
            dt in (select maX(dt) from future_dim_ucp_active where  axioma_id = :id_arg and dt < :dt_arg)
        '''
        for k,c in self.contracts.items():
            self.extractor.dbCursor.execute(q, id_arg=k, dt_arg=self.extractor.extractDate)
            res = self.extractor.dbCursor.fetchall()
            c.volume = None
            for each in res:
                if each[0] is not None:
                    c.volume = each[0]
            self.extractor.dbCursor.execute(qPrev, id_arg=k, dt_arg=self.extractor.extractDate)
            res = self.extractor.dbCursor.fetchall()
            for each in res:
                if each[0] is not None:
                    c.previous_volume = each[0]

    def getContractOpenInterests(self):
 
        q = '''
            select VALUE, DT, OI_MARKER  from future_dim_oi_active
            where AXIOMA_ID = :id_arg and 
            DT = :dt_arg
        '''
        qPrev = '''
            select VALUE, DT, OI_MARKER  from future_dim_oi_active
            where AXIOMA_ID = :id_arg and 
            dt in (select maX(dt) from future_dim_oi_active where  axioma_id = :id_arg and dt < :dt_arg)
        '''

        for k,c in self.contracts.items():
            self.extractor.dbCursor.execute(q, id_arg=k, dt_arg=self.extractor.extractDate)
            res = self.extractor.dbCursor.fetchall()
            c.open_interest = None
            for each in res:
                if each[0] is not None:
                    c.open_interest = each[0]
                    if each[2] == 3:
                        c.open_interest_roll_over_flag = '*'
                    else:
                        c.open_interest_roll_over_flag = ''
            self.extractor.dbCursor.execute(qPrev, id_arg=k, dt_arg=self.extractor.extractDate)
            res = self.extractor.dbCursor.fetchall()
            for each in res:
                if each[0] is not None:
                    if c.open_interest is None:
                        c.open_interest = each[0]
                        c.open_interest_roll_over_flag = '*'
                    c.previous_open_interest = each[0]
                    if each[2] == 3:
                        c.previous_open_interest_roll_over_flag = '*'
                    else:
                        c.previous_open_interest_roll_over_flag = ''

    def getContractReturns(self):
 
        q = '''
            select VALUE, DT from future_dim_return_active
            where AXIOMA_ID = :id_arg and 
            DT = :dt_arg
        '''
        for k,c in self.contracts.items():
            if c.price_return is not None:
                continue
            self.extractor.dbCursor.execute(q, id_arg=k, dt_arg=self.extractor.extractDate)
            res = self.extractor.dbCursor.fetchall()
            for each in res:
                if each[0] is not None:
                    c.price_return = each[0]

    def writeAttributes(self):
        if self.name:
            self.rows['seriesAttributes'] = [[self.future_family_id, self.asset_type, self.asset_class, self.asset_sub_class, self.name, 
                                              self.ticker, self.currency, self.trading_country, self.stock_exchange, self.expiration_periodicity, 
                                              self.index_official_name, self.distributed_index_name, self.composite_axioma_id, self.contract_size,
                                              self.fiveDayAdv, self.twentyDayAdv, self.fiveDayOi, self.twentyDayOi]]

#    def writeFutureLinks(self):
#        for c in self.contracts.values():
#            self.rows['futureLinks'].append([self.axiomaId, c.axiomaId])

    def writeContractAttributes(self):
        """
        headers['contractAttributes'] = ['contract_id', 'future_family_id', 'asset_type', 'asset_class', 'asset_sub_class', 
        'ticker', 'sedol', 'cusip', 'isin', 'name', 'last_trading_date', 'price', 'price_roll_over_flag', 'return', 'volume', 'open_interest']

        """
        dt = self.extractor.extractDate
        for c in self.contracts.values():
#            price = c.prices.get(dt, '')
#            volume = c.volumes.get(dt, '')
#            oi = c.openInterests.get(dt, '')
#            ret = c.returns.get(dt, '')
            self.rows['contractAttributes'].append([c.modeldb_id, self.future_family_id, self.asset_type, self.asset_class, self.classificationNames[5359], c.ticker,
                                                    c.sedol, c.cusip, c.isin,
                                                    c.name, str(c.last_trading_date).split()[0], c.price, c.price_roll_over_flag, c.price_return,
                                                    c.volume, c.open_interest])

    def writeDayContractMarketData(self):
        for c in self.contracts.values():
            dt = self.extractor.extractDate
            price = c.prices[dt]
            volume = ''
            if dt in c.volumes:
                volume = c.volumes[dt]
            oi = ''
            if dt in c.openInterests:
                oi = c.openInterests[dt]
            ret = ''
            if dt in c.returns:
                ret = c.returns[dt]
            self.rows['contractMarketData'].append([c.axiomaId, dt, price, volume, oi, ret])

    def writeContractMarketData(self):
        for c in self.contracts.values():
            for dt in sorted(c.prices.keys()):
                price = c.prices[dt]
                volume = ''
                if dt in c.volumes:
                    volume = c.volumes[dt]
                oi = ''
                if dt in c.openInterests:
                    oi = c.openInterests[dt]
                ret = ''
                if dt in c.returns:
                    ret = c.returns[dt]
                self.rows['contractMarketData'].append([c.axiomaId, dt, price, volume, oi, ret])

class Series40(Series):
    # new columns to be added in the Series for this version of the AFF include Bloomberg Ticker, MIC, Index Type and Index currency
    headers=deepcopy(Series.headers)
    headers['seriesAttributes'] += ['mic', 'bloomberg_ticker_prefix', 'index_type' ,'index_currency']
    #headers['seriesAttributes'] = ['future_family_id', 'asset_type', 'asset_class', 'asset_sub_class', 'name', 'ticker', 'currency', 'trading_country', 'stock_exchange', 'expiration_periodicity', 'index_official_name', 'distributed_index_name', 'composite_axioma_id', 'contract_size', 'five_day_adv', 'twenty_day_adv', 'five_day_aoi', 'twenty_day_aoi']
    headers['seriesAttributesFF'] = ['Axioma FutureFamilyID', 'Description', 'Ticker', 'Bloomberg Ticker Prefix', 
                                   'Trading Country', 'Trading Currency', 'Exchange', 'MIC',
                                   'Index Name', 'Index Type', 'Index Currency',
                                   'Composite AxiomaID', 'Contract Size', 'Expiration Periodicity', 
                                   'Asset Type', 'Asset Class', 'Asset SubClass']

    types=deepcopy(Series.types)
    types['seriesAttributes'] += ['Attribute', 'Attribute','Attribute','Attribute']
    types['seriesAttributesFF'] = ['ID', 'Attribute', 'Attribute', 'Attribute', 'Attribute', 'Attribute', 'Attribute', 'Attribute', 'Attribute', 'Attribute', 'Attribute', 'Attribute', 'Attribute', 'Attribute', 'Attribute', 'Attribute', 'Attribute' ]
    units=deepcopy(Series.units)
    units['seriesAttributes'] += ['Text', 'Text', 'Text', 'Text']
    units['seriesAttributesFF'] = ['ID', 'Text', 'Text', 'Text', 'Text', 'Text', 'Text', 'Text', 'Text', 'Text', 'Text', 'Text', 'Number', 'Text', 'Text', 'Text', 'Text']

    # now for the contract data - first add the new headers to the AFF 
    headers['contractAttributes'] += ['bloomberg_ticker', 'closest_to_expiration_flag']
    types['contractAttributes'] += ['Attribute', 'Set']
    units['contractAttributes'] += ['Text','NA']

    headers['contractAttributesFF'] = ['Axioma ContractID', 'Axioma FutureFamilyID', 'Description', 'Ticker', 'Bloomberg Ticker', 'Last Trading Date', 'Closest To Expiration Flag', 'Asset Type', 'Asset Class', 'Asset SubClass']
    types['contractAttributesFF'] = ['ID', 'ID', 'Attribute', 'Attribute', 'Attribute', 'Attribute', 'Set', 'Attribute','Attribute', 'Attribute']
    units['contractAttributesFF'] = ['ID', 'ID', 'Text', 'Text', 'Text', 'Date', 'NA','Text', 'Text', 'Text']

    INDEX_TYPE_DICT={'PR':'Price Return', 'GR':'Gross Return', 'NR':'Net Return', 'TR':'Total'}

    def __init__(self, axiomaId, extractor):
        Series. __init__(self, axiomaId, extractor)

    def writeAttributes(self):
        if self.name:
            self.rows['seriesAttributes'] = [[self.future_family_id, self.asset_type, self.asset_class, self.asset_sub_class, self.name, 
                                              self.ticker, self.currency, self.trading_country, self.stock_exchange, self.expiration_periodicity, 
                                              self.index_official_name, self.distributed_index_name, self.composite_axioma_id, self.contract_size,
                                              self.fiveDayAdv, self.twentyDayAdv, self.fiveDayOi, self.twentyDayOi, self.mic_code, self.bbg_ticker,
                                              Series40.INDEX_TYPE_DICT.get(self.index_type, self.index_type), self.index_currency]]
            self.rows['seriesAttributesFF'] = [[self.future_family_id, self.name, self.ticker, self.bbg_ticker,
                                              self.trading_country, self.currency, self.stock_exchange, self.mic_code,
                                              self.index_official_name, Series40.INDEX_TYPE_DICT.get(self.index_type, self.index_type), self.index_currency,
                                              self.composite_axioma_id, self.contract_size, self.expiration_periodicity,
                                              self.asset_type, self.asset_class, self.asset_sub_class]]
    def writeContractAttributes(self):
        dt = self.extractor.extractDate
        closest= min([c.last_trading_date for c in self.contracts.values()])
        for c in self.contracts.values():
            if closest == c.last_trading_date:
                c.closest_to_expiration='*'
            else:
                c.closest_to_expiration=''
            self.rows['contractAttributes'].append([c.modeldb_id, self.future_family_id, self.asset_type, self.asset_class, self.classificationNames[5359], 
                                                    c.ticker, c.sedol, c.cusip, c.isin,
                                                    c.name, str(c.last_trading_date).split()[0], c.price, c.price_roll_over_flag, c.price_return,
                                                    c.volume, c.open_interest, c.bbg_ticker, c.closest_to_expiration])
            self.rows['contractAttributesFF'].append([c.modeldb_id, self.future_family_id,  c.name, c.ticker, c.bbg_ticker, str(c.last_trading_date).split()[0],
                                                    c.closest_to_expiration,self.asset_type, self.asset_class, self.classificationNames[5359]])


class Contract():   
    def __init__(self, series, axiomaId, expirationDate):
        self.series = series
        self.axiomaId = axiomaId
        self.expirationDate = expirationDate
        self.last_trading_date = expirationDate
        self.ticker = None
        self.sedol = None
        self.cusip = None
        self.isin = None
        self.name = None
        self.price_return = None
        self.previous_price = None
        self.price_roll_over_flag = None
        self.previous_volume = None
        self.open_interest_roll_over_flag = None
        self.previous_open_interest_roll_over_flag = ''
        self.previous_open_interest = None
        self.prices = {}
        self.volumes = {}
        self.openInterests = {}
        self.returns = {}
          
    def __str__(self):
        res =  '  Contract(axiomaId:%s)' % (
            self.axiomaId)
        return res
    def __repr__(self):
        return self.__str__()

class EifExtractor:
    def __init__(self, marketDb=None, modelDb=None):
        self.log = logging.getLogger('EifExtractor')
        self.version=3.3
        self.testOnly = False
        if not marketDb:
            connectParameters = {}
#            connectParameters['sid'] = 'glsdg'
            connectParameters['sid'] = 'glprod'
            connectParameters['user'] = 'marketdb_global'
            connectParameters['passwd'] = 'marketdb_global'
            self.marketDB = MarketDB.MarketDB(sid=connectParameters['sid'], user=connectParameters['user'], passwd=connectParameters['passwd'])
        else:
            self.marketDB = marketDb
        if not modelDb:
            connectParameters = {}
#            connectParameters['sid'] = 'glsdg'
            connectParameters['sid'] = 'glprod'
            connectParameters['user'] = 'modeldb_global'
            connectParameters['passwd'] = 'modeldb_global'
            self.modelDB = ModelDB.ModelDB(sid=connectParameters['sid'], user=connectParameters['user'], passwd=connectParameters['passwd'])
        else:
            self.modelDB = modelDb
        self.dbConnection = self.marketDB.dbConnection
        self.dbCursor = self.dbConnection.cursor()
        self.modeldbConnection = self.modelDB.dbConnection
        self.modeldbCursor = self.modeldbConnection.cursor()
        self.fileNames = {}
#        self.fileNames['seriesAttributes'] = '/tmp/seriesAttributes.txt'
##        self.fileNames['futureLinks'] = '/tmp/futureLinks.txt'
#        self.fileNames['contractAttributes'] = '/tmp/contractAttributes.txt'
##        self.fileNames['contractMarketData'] = '/tmp/contractMarketData.txt'

        self.modelSeriesMap = {}
        self.modelSeriesMap['EU'] = """
select * from FUTURE_FAMILY_ATTR_ACTIVE_INT where SECTYPE_ID <> 5365 and distribute = 'Y' and FROM_DT <= :dt_arg and THRU_DT > :dt_arg and UNDERLYING_GEOGRAPHY in (
select 'EU' from dual union 
select rmg.MNEMONIC  from modeldb_global.RMG_MODEL_MAP rmm, modeldb_global.RISK_MODEL_SERIE rms, modeldb_global.RISK_MODEL rm, modeldb_global.risk_model_group rmg
where rm.MNEMONIC = 'AXEU21-MH' and rm.MODEL_ID = rms.RM_ID and rmm.RMS_ID = rms.SERIAL_ID
and rmg.RMG_ID = rmm.RMG_ID and rmm.thru_DT > :dt_arg and rms.thru_DT > :dt_arg)
"""
        self.modelSeriesMap['EM'] = """
select * from FUTURE_FAMILY_ATTR_ACTIVE_INT where SECTYPE_ID <> 5365 and distribute = 'Y' and FROM_DT <= :dt_arg and THRU_DT > :dt_arg  and UNDERLYING_GEOGRAPHY in (
select 'EM' from dual union 
select rmg.MNEMONIC from modeldb_global.RMG_MODEL_MAP rmm, modeldb_global.RISK_MODEL_SERIE rms, modeldb_global.RISK_MODEL rm, modeldb_global.risk_model_group rmg
where rm.MNEMONIC = 'AXEM21-MH' and rm.MODEL_ID = rms.RM_ID and rmm.RMS_ID = rms.SERIAL_ID
and rmg.RMG_ID = rmm.RMG_ID and rmm.thru_DT > :dt_arg and rms.thru_DT > :dt_arg)
"""
        self.modelSeriesMap['AP'] = """
select * from FUTURE_FAMILY_ATTR_ACTIVE_INT where SECTYPE_ID <> 5365 and distribute = 'Y' and FROM_DT <= :dt_arg and THRU_DT > :dt_arg   and UNDERLYING_GEOGRAPHY in (
select 'AP' from dual union 
select 'APxJP' from dual union 
select rmg.MNEMONIC from modeldb_global.RMG_MODEL_MAP rmm, modeldb_global.RISK_MODEL_SERIE rms, modeldb_global.RISK_MODEL rm, modeldb_global.risk_model_group rmg
where rm.MNEMONIC ='AXAP21-MH' and rm.MODEL_ID = rms.RM_ID and rmm.RMS_ID = rms.SERIAL_ID
and rmg.RMG_ID = rmm.RMG_ID and rmm.thru_DT > :dt_arg and rms.thru_DT > :dt_arg)
"""
        self.modelSeriesMap['APxJP'] = """
select * from FUTURE_FAMILY_ATTR_ACTIVE_INT where SECTYPE_ID <> 5365 and distribute = 'Y' and FROM_DT <= :dt_arg and THRU_DT > :dt_arg   and UNDERLYING_GEOGRAPHY in (
select 'APxJP' from dual union 
select rmg.MNEMONIC from modeldb_global.RMG_MODEL_MAP rmm, modeldb_global.RISK_MODEL_SERIE rms, modeldb_global.RISK_MODEL rm, modeldb_global.risk_model_group rmg
where rm.MNEMONIC = 'AXAPxJP21-MH' and rm.MODEL_ID = rms.RM_ID and rmm.RMS_ID = rms.SERIAL_ID
and rmg.RMG_ID = rmm.RMG_ID and rmm.thru_DT > :dt_arg and rms.thru_DT > :dt_arg)
"""
        self.modelSeriesMap['NA'] = """
select ID from FUTURE_FAMILY_ATTR_ACTIVE_INT where SECTYPE_ID <> 5365 and distribute = 'Y' and FROM_DT <= :dt_arg and THRU_DT > :dt_arg   and UNDERLYING_GEOGRAPHY in (
select 'NA' from dual union 
select rmg.MNEMONIC from modeldb_global.RMG_MODEL_MAP rmm, modeldb_global.RISK_MODEL_SERIE rms, modeldb_global.RISK_MODEL rm, modeldb_global.risk_model_group rmg
where rm.MNEMONIC = 'AXNA21-MH' and rm.MODEL_ID = rms.RM_ID and rmm.RMS_ID = rms.SERIAL_ID
and rmg.RMG_ID = rmm.RMG_ID and rmm.thru_DT > :dt_arg and rms.thru_DT > :dt_arg)
"""
        self.modelSeriesMap['NA'] = """
select ID from FUTURE_FAMILY_ATTR_ACTIVE_INT where SECTYPE_ID <> 5365 and distribute = 'Y' and FROM_DT <= :dt_arg and THRU_DT > :dt_arg   and UNDERLYING_GEOGRAPHY in (
select 'NA' from dual union 
select rmg.MNEMONIC from modeldb_global.RMG_MODEL_MAP rmm, modeldb_global.RISK_MODEL_SERIE rms, modeldb_global.RISK_MODEL rm, modeldb_global.risk_model_group rmg
where rm.MNEMONIC = 'AXNA21-MH' and rm.MODEL_ID = rms.RM_ID and rmm.RMS_ID = rms.SERIAL_ID
and rmg.RMG_ID = rmm.RMG_ID and rmm.thru_DT > :dt_arg and rms.thru_DT > :dt_arg)
"""
        self.modelSeriesMap['WWxUS'] = """
select ID from FUTURE_FAMILY_ATTR_ACTIVE_INT where SECTYPE_ID <> 5365 and distribute = 'Y' and FROM_DT <= :dt_arg and THRU_DT > :dt_arg   and UNDERLYING_GEOGRAPHY in (
select 'EM' from dual union 
select 'EU' from dual union 
select 'AP' from dual union 
select 'APxJP' from dual union 
select 'WWxUS' from dual union 
select rmg.MNEMONIC from modeldb_global.RMG_MODEL_MAP rmm, modeldb_global.RISK_MODEL_SERIE rms, modeldb_global.RISK_MODEL rm, modeldb_global.risk_model_group rmg
where rm.MNEMONIC = 'AXWWxUS21-MH' and rm.MODEL_ID = rms.RM_ID and rmm.RMS_ID = rms.SERIAL_ID
and rmg.RMG_ID = rmm.RMG_ID and rmm.thru_DT > :dt_arg and rms.thru_DT > :dt_arg)
"""
        self.modelSeriesMap['WW'] = """
select ID from FUTURE_FAMILY_ATTR_ACTIVE_INT where SECTYPE_ID <> 5365 and distribute = 'Y' and FROM_DT <= :dt_arg and THRU_DT > :dt_arg
"""

    def populateSeriesAttr(self, axiomaIdList, seriesAttr):
        q = """
with cte (iso_ctry_code,from_dt,row_rank)
as
(
   select /*+ inline */  
          iso_ctry_code,dt,row_rank
   from 
   (
    select /*+ first_rows(4000) index(mtc1, IDX2_META_TRADING_CALENDAR) */
           mtc1.iso_ctry_code, mtc1.dt, rank() over (partition by mtc1.iso_ctry_code order by mtc1.dt desc) as row_rank
    from meta_trading_calendar mtc1
    where mtc1.rev_del_flag = 'N'
      and mtc1.rev_dt = 
         (   
          select /*+ first_rows(1) index(mtc, IDX1_META_TRADING_CALENDAR) */
                 mtc.rev_dt
          from meta_trading_calendar mtc
          where --mtc1.rev_del_flag = 'N'
                mtc.iso_ctry_code  = mtc1.iso_ctry_code
            and mtc.dt             = mtc1.dt
            and rownum            <= 1
         )
      and mtc1.dt between add_months(:date_arg,-3) and :date_arg
   ) r
   where row_rank in (5,20)
)
select   
         ffa.id

        ,(select name from CLASSIFICATION_REF where id = 5355) as asset_type
        ,(select name from CLASSIFICATION_REF where id = 5356) as asset_class 
        ,(select name from CLASSIFICATION_REF where id = 5357) as asset_sub_class 

        ,ffa.NAME 
        ,ffa.EXTERNAL_TICKER as ticker
        ,cur.CODE as currency
        ,ffa.TRADING_COUNTRY
        ,cr.DESCRIPTION as stock_exchange 
        ,mc.NAME as expiration_periodicity

        ,(select  ion.OFFICIAL_NAME 
          from index_official_name ion  
          where ffa.INDEX_MEMBER_ID =  ion.INDEX_ID 
            and ion.FROM_DT         <= :date_arg 
            and ion.thru_dt         >  :date_arg
         ) as index_official_name 

        ,(select im.name 
          from index_member im , 
               index_family ifa 
          where im.id           =  ffa.INDEX_MEMBER_ID 
            and im.FAMILY_ID    =  ifa.ID 
            and ifa.DISTRIBUTE  =  'Y' 
            and im.DIST_FROM_DT <= :date_arg 
            and im.DIST_THRU_DT >  :date_arg ) as distributed_index_name
        
        ,(select substr(MODELDB_ID,2,9) 
            from modeldb_global.issue_map im , INDEX_BEST_COMPOSITE_MAP   map 
            where 
            ffa.index_member_id = map.index_member_id and
            map.FROM_DT      <= :date_arg 
            and map.THRU_DT      >  :date_arg 
            and map.ETF_AXIOMA_ID     =  im.MARKETDB_ID 
            and im.FROM_DT       <= :date_arg  
            and im.thru_dt       >  :date_arg 
         ) as composite_axioma_ID


        ,CONTRACT_SIZE

        ,(select avg(nvl(tdv.VALUE,0)) 
          from FUTURE_DIM_TDV_ACTIVE tdv, 
               future_linkage_active fl, 
               future_dim_ucp_active ucp 
          where fl.FUTURE_FAMILY_ID =  ffa.ID 
            and fl.FUTURE_AXIOMA_ID =  ucp.AXIOMA_ID 
            and ucp.axioma_ID       =  tdv.axioma_ID (+)
            and ucp.DT              <= :date_arg
            and ucp.dt              >= (select /*+ first_rows(1) */ cte.from_dt from cte where cte.iso_ctry_code = ffa.trading_country and cte.row_rank = 5)
            and ucp.dt = tdv.dt (+)
         ) as Five_day_ADV

        ,(select avg(nvl(tdv.VALUE,0)) 
          from FUTURE_DIM_TDV_ACTIVE tdv, 
               future_linkage_active fl, 
               future_dim_ucp_active ucp 
          where fl.FUTURE_FAMILY_ID = ffa.ID and fl.FUTURE_AXIOMA_ID = ucp.AXIOMA_ID and ucp.axioma_ID = tdv.axioma_ID (+)
            and ucp.DT <= :date_arg  
            and ucp.dt >= (select /*+ first_rows(1) */ cte.from_dt from cte where cte.iso_ctry_code = ffa.trading_country and cte.row_rank = 20)
            and ucp.dt = tdv.dt (+)
        ) as Twenty_day_ADV

       ,(select avg(nvl(oi.VALUE,0)) 
         from FUTURE_DIM_OI_ACTIVE  oi, 
              future_linkage_active fl, 
              future_dim_ucp_active ucp 
         where fl.FUTURE_FAMILY_ID = ffa.ID and fl.FUTURE_AXIOMA_ID = ucp.AXIOMA_ID and ucp.axioma_ID = oi.axioma_ID (+)
           and ucp.DT <= :date_arg 
           and ucp.dt >= (select /*+ first_rows(1) */ cte.from_dt from cte where cte.iso_ctry_code = ffa.trading_country and cte.row_rank = 5)
           and ucp.dt = oi.dt (+)
         ) as Five_day_AOI

        ,(select avg(nvl(oi.VALUE,0)) 
          from FUTURE_DIM_OI_ACTIVE  oi, 
               future_linkage_active fl, 
               future_dim_ucp_active ucp 
          where fl.FUTURE_FAMILY_ID = ffa.ID
            and fl.FUTURE_AXIOMA_ID = ucp.AXIOMA_ID 
            and ucp.axioma_ID       = oi.axioma_ID (+)
            and ucp.DT <= :date_arg  
            and ucp.dt >= (select /*+ first_rows(1) */ cte.from_dt from cte where cte.iso_ctry_code = ffa.trading_country and cte.row_rank = 20)
            and ucp.dt = oi.dt (+)
         ) as Twenty_day_AOI
        , (select OPERATING_MIC from CLASS_MIC_MAP_ACTIVE_INT mic where mic.classification_id=cr.id
                and mic.from_dt <= :date_arg and :date_arg < mic.thru_dt) mic_code
        , ffa.bloomberg_ticker_preface 
        , (select cc.code from currency_ref cc, index_variant iv where ffa.index_variant_id = iv.id and iv.from_dt <= :date_arg and :date_arg < iv.thru_dt and cc.id=iv.currency_id) index_currency 
        , (select iv.return_type from index_variant iv where ffa.index_variant_id = iv.id and iv.from_dt <= :date_arg and :date_arg < iv.thru_dt) index_return_type 

        from FUTURE_FAMILY_ATTR_ACTIVE_INT ffa,
             CURRENCY_REF                  cur, 
             CLASSIFICATION_REF            cr , 
             meta_codes                    mc
       
        where sectype_id      =  5360  
          and ffa.distribute  =  'Y' 
          and ffa.from_Dt     <= :date_arg 
          and ffa.thru_Dt     >  :date_arg 
          and ffa.CURRENCY_ID =  cur.ID 
          and cr.ID           =  ffa.STOCK_EXCHANGE_ID 
        -- to make sure at least one contract exists for this series
          and exists (select 1 
                      from modeldb_global.future_issue_map fim, 
                           future_linkage_active           fla
                      where fla.FUTURE_FAMILY_ID =  ffa.id 
                        and fim.distribute       =  'Y' 
                        and fim.MARKETDB_ID      =  fla.FUTURE_AXIOMA_ID
                        and fim.from_DT          <= :date_arg
        -- look up expiration periodicity from meta_codes table
                     ) 
          and ffa.EXPIRATION_PERIODICITY = mc.ID 
          and mc.CODE_TYPE               = 'future_family_attribute:expiration_periodicity'
        and ffa.id in ('%s')"""%"','".join(axiomaIdList)

#        print q
#        print self.axiomaId, self.extractor.date
        t0 = time.time()
        self.dbCursor.execute(q, date_arg=self.date)
        res = self.dbCursor.fetchall()
        for each in res:
            s = seriesAttr[each[0]]            
            s.asset_type = each[1]      
            s.asset_class = each[2]          
            s.asset_sub_class = each[3]         
            s.name = each[4]            
            s.ticker = each[5]            
            s.currency = each[6]            
            s.trading_country = each[7]            
            s.stock_exchange = each[8].upper()
            s.expiration_periodicity = each[9]            
            s.index_official_name = each[10]            
            s.distributed_index_name = each[11]            
            s.composite_axioma_id = each[12]            
            s.contract_size = each[13] 
            s.fiveDayAdv = each[14]           
            s.twentyDayAdv = each[15]           
            s.fiveDayOi = each[16]           
            s.twentyDayOi = each[17]           
            s.mic_code = each[18]
            s.bbg_ticker = each[19]
            s.index_currency = each[20]
            s.index_type = each[21]
        self.log.info('fetching query results took %s sec', time.time() - t0)

    def writeHeader(self):
        extractDate=self.extractDate
        gmtime = time.gmtime(time.mktime(self.modelDB.revDateTime.timetuple()))
        utctime = datetime.datetime(year=gmtime.tm_year,
                                    month=gmtime.tm_mon,
                                    day=gmtime.tm_mday,
                                    hour=gmtime.tm_hour,
                                    minute=gmtime.tm_min,
                                    second=gmtime.tm_sec)
        
        for fileName, filePath in self.fileNames.items():
            with open(filePath, 'a') as f:
                f.write('#DataDate: %s\n' % extractDate)
                f.write('#CreationTimestamp: %sZ\n' %
                      utctime.strftime('%Y-%m-%d %H:%M:%S'))
                f.write('#FlatFileVersion: %s\n' % self.version)
                if self.version==4.0:
                    header =  '|'.join(Series40.headers[fileName])
                    types =  '|'.join(Series40.types[fileName])
                    units =  '|'.join(Series40.units[fileName])
                else:
                    header =  '|'.join(Series.headers[fileName])
                    types =  '|'.join(Series.types[fileName])
                    units =  '|'.join(Series.units[fileName])
                f.write('#Columns: %s\n' % header)
                f.write('#Type: %s\n' % types)
                f.write('#Unit: %s\n' % units)
#                f.write('#Precision: ....\n')


    def processSeries(self, axiomaIdList, extractDate):
        self.extractDate = extractDate
        for f in self.fileNames.values():
            try:
                os.remove(f)
            except:
                pass
        self.writeHeader()

        seriesAttr = {}
        for axiomaId in axiomaIdList:
            if self.version == 4.0:
                s = Series40(axiomaId, self)
            else:
                s = Series(axiomaId, self)
            seriesAttr[s.axiomaId] = s
        self.populateSeriesAttr(axiomaIdList, seriesAttr)
        logging.info('Writing %s', ','.join(list(self.fileNames.values())))
        for s in seriesAttr.values():
            s.getContracts()
            if len(list(s.contracts.keys())) == 0:
                continue
#            print 'processing', axiomaId
            s.getClassificationNames()
#            s.getAttributes()
            s.writeAttributes()
            t0 = time.time()
            s.getContractTickers()
            s.getContractBloombergTickers()
            s.getContractNames()
            s.getContractPrices()
            s.getContractVolumes()
            s.getContractOpenInterests()
            s.getContractReturns()
            s.writeContractAttributes()
            self.log.debug('getting contract attributes for %s took %s sec', s.axiomaId, time.time() - t0)
       
            for fileName, filePath in self.fileNames.items():
                with open(filePath, 'a') as f:
                    writer = csv.writer(f, 'pipe')
                    for row in s.rows[fileName]:
                        writer.writerow(row)
                    
        #for f in self.fileNames.values():
        #    print open(f, 'rt').read()

    def findSeries(self, modelName):
        seriesList = []
        #try by country first
        q = "select * from FUTURE_FAMILY_ATTR_ACTIVE_INT where SECTYPE_ID <> 5365 and distribute = 'Y' and UNDERLYING_GEOGRAPHY = '%s' and FROM_DT <= :dt_arg and THRU_DT > :dt_arg" % modelName
        q = "%s and id in (select id from future_family_ref where dist_from_DT <= :dt_arg and dist_thru_DT > :dt_arg)" % q
        self.dbCursor.execute(q, dt_arg=self.date)
        res = self.dbCursor.fetchall()
        for each in res:
            seriesList.append(each[0])

        #now try ny region
        q = self.modelSeriesMap.get(modelName, None)
        if q is not None:
            q = "%s and id in (select id from future_family_ref where dist_from_DT <= :dt_arg and dist_thru_DT > :dt_arg)" % q
#            print q
            self.dbCursor.execute(q, dt_arg=self.date)
            res = self.dbCursor.fetchall()
            for each in res:
                seriesList.append(each[0])
#        print seriesList
        return sorted(list(set(seriesList)))
        
        
    def extract(self, folderName, encryptedFolderName,modelName, extractDate):
        
        #self.fileNames = {}
        date = ''.join(extractDate.split('-'))
        self.fileNames['seriesAttributes'] = '%s/Composite-EIF-%s-%s.%s.att' % (folderName, modelName, 'SERIES', date)
        self.fileNames['contractAttributes'] = '%s/Composite-EIF-%s-%s.%s.att' % (folderName, modelName, 'CONTRACTS', date)
        
        self.date = Utilities.parseISODate(extractDate)
        seriesList = self.findSeries(modelName)
        if len(seriesList) > 0:
            self.processSeries(seriesList, extractDate)
            for filetype in ['seriesAttributes','contractAttributes']:
                logging.info('Clear %s', self.fileNames[filetype])
                Utilities.encryptFile(self.date, self.fileNames[filetype], encryptedFolderName)
                
            return 1
        else:
            return 0

    def transferAllSeries(self):
        dateRange = '2008-01-01:2012-12-13'
        q = 'select distinct id from future_family_ref'
        self.dbCursor.execute(q)
        res = self.dbCursor.fetchall()
        for each in res:
            print('running transfer for', each[0], dateRange)
            self.transferSeries([each[0]], dateRange)
            
    def transferSeries(self, axiomaIdList, dateRange):
#        sections = ['Return']
        sections = ['EIF-Price', 'EIF-Volume', 'EIF-OpenInterest', 'EIF-Return', 'SubIssueData', 'SubIssueReturn', 'SubIssueCumulativeReturn']
        for axiomaId in axiomaIdList:
            s = Series(axiomaId, self)
            s.getTransferContracts()
            axiomaIdsString = ','.join(list(s.contracts.keys()))
            print('!!!', axiomaIdsString)

            for section in sections:
                transferParams = """production.config MarketDB:sid=glsdg AxiomaDB:sid=glsdg dates=%s
                axioma-ids=%s 
                sections=%s""" % (dateRange, axiomaIdsString, section)
                cmd = "cd MarketDB;%s/MarketDB/runTransfers '%s'" % (subprocess.getstatusoutput('pwd')[1], transferParams)
                print(cmd)
                p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
                out, err = p.communicate() 
                print(out)                  
                    
#                    cmd = 'ls'
#                    p1 = subprocess.Popen([cmd], stdout=subprocess.PIPE)
#                    print  p1.communicate()[0]
#                res = commands.getstatusoutput(cmd)
#                pp.pprint(res)

    def convertDateRangeToList(self, dateRange):
        dateList = []
        d1 = dateRange.split(':')[0]
        y1 = int(d1.split('-')[0])
        m1 = int(d1.split('-')[1])
        d1 = int(d1.split('-')[2])
        fromDate = datetime.date(year=y1, month=m1, day=d1)
        d2 = dateRange.split(':')[1]
        y2 = int(d2.split('-')[0])
        m2 = int(d2.split('-')[1])
        d2 = int(d2.split('-')[2])
        toDate = datetime.date(year=y2, month=m2, day=d2)
        dt = fromDate
        dateList.append(str(dt))
        while dt < toDate:
            dt += datetime.timedelta(days = 1)
            dateList.append(str(dt))
        return dateList
            

class EifExtractorV40(EifExtractor):
    def __init__(self, marketDb, modelDb):
        EifExtractor.__init__(self, marketDb, modelDb)
        self.version=4.0

    def extract(self, folderName, encryptedFolderName,modelName, extractDate, flatFileDir):
        """ call the parent extractor to take care of the old style stuff.
            and set up the call to the new flat files in the new style """
        date = ''.join(extractDate.split('-'))
        self.fileNames['seriesAttributesFF'] = '%s/Composite-EIF-%s-%s.%s.idm' % (flatFileDir, modelName, 'SERIES', date)
        self.fileNames['contractAttributesFF'] = '%s/Composite-EIF-%s-%s.%s.idm' % (flatFileDir, modelName, 'CONTRACTS', date)
        return(EifExtractor.extract(self, folderName, encryptedFolderName, modelName, extractDate))

if __name__ == '__main__':
    usage = "usage: %prog [options]"
    cmdlineParser = optparse.OptionParser(usage=usage)

    Utilities.addDefaultCommandLine(cmdlineParser)
    (options, args) = cmdlineParser.parse_args()
    
    Utilities.processDefaultCommandLine(options, cmdlineParser)
    
    extractor = EifExtractor()
#    extractor.transferAllSeries()
#    extractor.processSeries(['FUT_000110'], '20130109')

#    extractor.extract('/tmp/a', '/tmp/b','WWxUS', '2013-02-15')
#    extractor.extract('/tmp/a', '/tmp/b','WW', '2008-01-02')
#    extractor.extract('/tmp/a', '/tmp/b','EU', '2008-01-02')
#    extractor.extract('/tmp/a', '/tmp/b','US', '2008-01-02')
#    extractor.extract('/tmp/a', '/tmp/b','JP', '2008-01-02')
#
#    extractor.extract('/tmp/a', '/tmp/b','WW', '2012-01-23')
#    extractor.extract('/tmp/a', '/tmp/b','AP', '2012-01-23')
#    extractor.extract('/tmp/a', '/tmp/b','APxJP', '2012-01-23')
#
#    extractor.extract('/tmp/a', '/tmp/b','JP', '2013-01-02')

#    extractor.extract('/tmp/a', '/tmp/b','EU', '2013-02-11')
#    extractor.extract('/tmp/a', '/tmp/b','AP', '2013-02-11')
#    extractor.extract('/tmp/a', '/tmp/b','JP', '2013-02-11')
#    extractor.extract('/tmp/a', '/tmp/b','APxJP', '2013-02-11')
#

    startTime = time.time()
#    dateList = extractor.convertDateRangeToList('2008-06-23:2008-07-22')
#    dateList = extractor.convertDateRangeToList('2008-06-23:2008-06-23')
    dateList = extractor.convertDateRangeToList('2013-12-04:2013-12-04')
#    dateList = extractor.convertDateRangeToList('2012-09-12:2013-03-11')
    for dt in dateList:
        extractor.log.info('started %s', dt)
        extractor.extract('/tmp/a', '/tmp/b','WW', dt)
        extractor.log.info('finished %s', dt)
    extractor.log.info('extract took %s sec', time.time() - startTime)
#    extractor.extract('/tmp/a', '/tmp/b','WWxUS', '2013-02-14')
    
#    extractor.processSeries(['FUT_000077', 'FUT_000016', 'FUT_000089'])
#    extractor.processSeries(['FUT_999997', 'FUT_999998'])

#    extractor.transferSeries(['FUT_100001'], '2012-01-01:2012-11-28')
#    extractor.transferSeries(['FUT_100034'], '2002-10-01:2002-10-02')
    
#    extractor.transferSeries(['FUT_000048'], '2007-12-15:2013-02-15 ')
#    extractor.transferSeries(['FUT_000015'], '2007-12-15:2013-02-15 ')


