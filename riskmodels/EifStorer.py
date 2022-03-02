
import logging
import pprint
import datetime
import optparse
import calendar
import subprocess
from riskmodels import Utilities
from marketdb import MarketDB
from marketdb.EIFHelper import expirationDate, getDatastreamLTD, getFTIDLTD
from riskmodels import ModelDB

pp = pprint.PrettyPrinter(indent=2)

COMMODITY_SECTYPE = 5365
# this maps from AxiomaSecType classification IDs to Usage classification IDs
USAGE_MAP = {
    5365: 6303,
    5360: 6304
}

class Series(): 
    '''
    inserts into:
    asset_ref,
    future_linkage,
    future_dim_datastream,
    future_dim_ftid
    classification_constituent
    future_dim_trading_currency
    future_dim_name
    future_dim_ticker
    modeldb.issue
    modeldb.future_issue_map
    '''
    months = {}
    months['01']   = 'F'
    months['02']   = 'G'
    months['03']   = 'H'
    months['04']   = 'J'
    months['05']   = 'K'
    months['06']   = 'M'
    months['07']   = 'N'
    months['08']   = 'Q'
    months['09']   = 'U'
    months['10']   = 'V'
    months['11']   = 'X'
    months['12']   = 'Z'

    def __init__(self, axiomaId, storer):
        self.new = {}  
        self.axiomaId = axiomaId
        self.storer = storer
        self.name = None
        self.externalTicker = None
        self.extendedTicker = None
        self.bbTicker = None
        self.bbSwitchDt = None
        self.currencyId = None
        self.currencyCode = None
        self.tradingCountry = None
        self.classificationId = None
        self.expirationPattern = None
        self.underlyingCountry = None
        self.contracts = []
        self.hasAttributes = False
        self.getAttributes()
        self.getCurrencyCode()
        self.insertValues = []
        self.srcId = 907
        self.dsSrcId = 1000
        self.ftidSrcId = 1100
        self.familySrcId = 5000
        self.userName = 'system'
        self.initDate = datetime.date(1950, 1, 1)
        self.logger = logging.getLogger('EifStorer')

    def __str__(self):
        res =  'Series(axiomaId:%s, name:%s, externalTicker:%s, currencyId:%s, tradingCountry:%s)' % (
            self.axiomaId, self.name, self.externalTicker, self.currencyId, self.tradingCountry)
        return res
    
    def __repr__(self):
        return self.__str__()
        
    def getAttributes(self, asOfDate=None):
        if not asOfDate:
            asOfDate = self.storer.date
        q = '''
            select NAME, EXTERNAL_TICKER, CURRENCY_ID, TRADING_COUNTRY, STOCK_EXCHANGE_ID, LAST_TRADING_DATE_PATTERN, DISTRIBUTE, UNDERLYING_GEOGRAPHY, SECTYPE_ID, EXTENDED_TICKER, bloomberg_ticker_preface
            from future_family_attr_active_int 
            where id = :id_arg and from_dt <= :date_arg AND thru_dt > :date_arg
        '''
        self.storer.dbCursor.execute(q, id_arg=self.axiomaId, date_arg=asOfDate)
        res = self.storer.dbCursor.fetchall()
        if not res:
            q = '''
                select NAME, EXTERNAL_TICKER, CURRENCY_ID, TRADING_COUNTRY, STOCK_EXCHANGE_ID, LAST_TRADING_DATE_PATTERN, DISTRIBUTE, UNDERLYING_GEOGRAPHY, SECTYPE_ID, EXTENDED_TICKER, bloomberg_ticker_preface
                from future_family_attr_active_int 
                where id = :id_arg  AND thru_dt > :date_arg
            '''
            self.storer.dbCursor.execute(q, id_arg=self.axiomaId, date_arg=asOfDate)
            res = self.storer.dbCursor.fetchall()
            
            if not res:
                raise ValueError("No data for %s on %s" % (self.axiomaId, asOfDate))
        for each in res:
            self.name = each[0]
            self.externalTicker = each[1]
            self.currencyId = each[2]
            self.tradingCountry = each[3]
            self.classificationId = each[4]
            self.expirationPattern = each[5]
            self.distribute = each[6]
            self.underlyingCountry = each[7]
            self.secTypeId = each[8]
            self.extendedTicker = each[9]
            self.bbTicker = each[10]
            self.hasAttributes = True
        q = 'SELECT BBG_TICKER_PATTERN_SWITCH_DT from FUTURE_BB_TICKER_SWITCH ts WHERE FUTURE_FAMILY_ID=:id_arg'
        self.storer.dbCursor.execute(q, id_arg=self.axiomaId)
        res = self.storer.dbCursor.fetchall()
        self.bbSwitchDt = res and res[0][0].date() or None

    def getCurrencyCode(self, asOfDate=None):
        if not asOfDate:
            asOfDate = self.storer.date
        q = '''
            select Code from currency_ref where id = :id_arg 
            and from_dt <= :date_arg
            AND thru_dt > :date_arg
        '''
        self.storer.dbCursor.execute(q, id_arg=self.currencyId, date_arg=asOfDate)
        res = self.storer.dbCursor.fetchall()
        for each in res:
            self.currencyCode = each[0]

    def insertMany(self, insertQuery, insertValues):
        self.logger.info('Inserting %d records' % len(insertValues))
        self.logger.debug(insertQuery)
        self.logger.debug(insertValues)
        noValues = True
        nonEmptyValues = []
        for each in insertValues:
            if len(each) > 0:
                noValues = False
                nonEmptyValues.append(each)
        if noValues:
            self.logger.warning('Nothing to insert')
        else:
            self.storer.dbCursor.executemany(insertQuery, nonEmptyValues)

    def insertManyModelDB(self, insertQuery, insertValues):
        self.logger.info('Inserting %d records' % len(insertValues))
        self.logger.debug(insertQuery)
        self.logger.debug(insertValues)
        noValues = True
        nonEmptyValues = []
        for each in insertValues:
            if len(each) > 0:
                noValues = False
                nonEmptyValues.append(each)
        if noValues:
            self.logger.warning('Nothing to insert')
        else:
            self.storer.modeldbCursor.executemany(insertQuery, nonEmptyValues)
    
    def insertAssetRef(self):
        insertValues = []
        for c in self.contracts:
            if c.insertNeeded:
                insertValues.append(c.insertAssetRefValues())
#        pp.pprint(insertValues)
        insertQuery = """INSERT INTO ASSET_REF
        (AXIOMA_ID, FROM_DT, THRU_DT, SRC_ID, REF, ADD_DT, TRADING_COUNTRY)
        VALUES (:axiomaId, :fromDt, :thruDt, :srcId,
        :ref, :addDt, :ctry)"""
#        print insertQuery
        self.insertMany(insertQuery, insertValues)
                
    def insertIssueMap(self):
        insertValues = []
        for c in self.contracts:
            if c.insertNeeded:
                insertValues.append(c.insertIssueValues())
        insertQuery = """INSERT INTO ISSUE
        (ISSUE_ID, FROM_DT, THRU_DT)
        VALUES (:modeldb_id, :fromDt, :thruDt)"""
        self.insertManyModelDB(insertQuery, insertValues)
#        print 'inserted issue'
        insertValues = []
        for c in self.contracts:
            if c.insertNeeded:
                insertValues.append(c.insertIssueMapValues())
        insertQuery = """INSERT INTO FUTURE_ISSUE_MAP
        (MARKETDB_ID, MODELDB_ID, FROM_DT, THRU_DT, DISTRIBUTE)
        VALUES (:marketdb_id, :modeldb_id, :fromDt, :thruDt, :distribute)"""
        self.insertManyModelDB(insertQuery, insertValues)
        print('inserted future_issue_map')
    
    def insertSubIssueMap(self):
        insertQuery = """INSERT INTO sub_issue
        (issue_id, from_dt, thru_dt, sub_id, rmg_id)
        VALUES
        (:issue_id, :fromDt, :thruDt, :sub_id, :rmg)"""
        insertValues = []
        for c in self.contracts:
            if c.series.secTypeId == COMMODITY_SECTYPE and c.insertNeeded:
                # commodity series need sub_issue_data as well
                insertValues.append(c.insertSubIssueValues())
        self.insertManyModelDB(insertQuery, insertValues)
        # print 'inserted %d subissue records' % len(insertValues)
    
    def insertFutureLinkage(self):
        insertQuery = """INSERT INTO FUTURE_LINKAGE
        (FUTURE_FAMILY_ID, FUTURE_AXIOMA_ID, EXPIRATION_MONTH, last_trading_date, REF, REV_DT, REV_DEL_FLAG )
        VALUES (:familyId, :axiomaId, :expiryMonth, :lastTradeDate, :ref,
        :revDt, :revDelFlag)"""
        for delFlag in [False]:
            insertValues = []
            for c in self.contracts:
                if c.insertNeeded:
                    insertValues.append(c.insertFutureLinkageValues(delFlag))
            self.insertMany(insertQuery, insertValues)

    def insertFutureDimDatastream(self):
        insertQuery = """INSERT INTO FUTURE_DIM_DATASTREAM
        (AXIOMA_ID, CONTR_CODE, CLS_CODE, FUT_CODE, CHANGE_DT, CHANGE_DEL_FLAG, SRC_ID, REF, REV_DT, REV_DEL_FLAG  )
        VALUES (:axiomaId, :contrCode, :clsCode, :futCode, :changeDt, :changeDelFlag, :srcId, :ref, :revDt, :revDelFlag)
        """
        for delFlag in [False, True]:
            insertValues = []
            for c in self.contracts:
                if c.insertNeeded:
                    insertValues.append(c.insertFutureDimDatastreamValues(delFlag))
            self.insertMany(insertQuery, insertValues)

    def insertFutureDimFtid(self):
        insertQuery = """INSERT INTO FUTURE_DIM_FTID
        (AXIOMA_ID, FTID_INFOCODE, FTID_CODE, FTID_SERIES_CODE, CONTRACT_DATE, CHANGE_DT, CHANGE_DEL_FLAG, SRC_ID, REF, REV_DT, REV_DEL_FLAG)
        VALUES (:axiomaId, :ftidInfoCode, :ftidCode, :ftidSeriesCode, :contractDt, :changeDt, :changeDelFlag, :srcId, :ref, :revDt, :revDelFlag)
        """
        for delFlag in [False, True]:
            insertValues = []
            for c in self.contracts:
                if c.insertNeeded:
                    insertValues.append(c.insertFutureDimFtidValues(delFlag))
            self.insertMany(insertQuery, insertValues)

    def insertAssetType(self):
        insertQuery = """INSERT INTO classification_constituent
        (CLASSIFICATION_ID, AXIOMA_ID, WEIGHT, CHANGE_DT, CHANGE_DEL_FLAG, SRC_ID, REF, REV_DT, REV_DEL_FLAG)
        VALUES (:classificationId, :axiomaId, :weight, :changeDt, :changeDelFlag, :srcId, :ref, :revDt, :revDelFlag)
        """
        for delFlag in [False, True]:
            insertValues = []
            for c in self.contracts:
                if c.insertNeeded:
                    insertValues.append(c.insertAssetTypeValues(delFlag))
            self.insertMany(insertQuery, insertValues)
    
    def insertClassificationConstituent(self):
        insertQuery = """INSERT INTO classification_constituent
        (CLASSIFICATION_ID, AXIOMA_ID, WEIGHT, CHANGE_DT, CHANGE_DEL_FLAG, SRC_ID, REF, REV_DT, REV_DEL_FLAG)
        VALUES (:classificationId, :axiomaId, :weight, :changeDt, :changeDelFlag, :srcId, :ref, :revDt, :revDelFlag)
        """
        for delFlag in [False, True]:
            insertValues = []
            for c in self.contracts:
                if c.insertNeeded:
                    insertValues.append(c.insertClassificationConstituentValues(delFlag))
            self.insertMany(insertQuery, insertValues)

    def insertClassificationConstituentCorrection(self):
        insertQuery = """INSERT INTO axiomadb.classification_constituent
        (CLASSIFICATION_ID, AXIOMA_ID, WEIGHT, CHANGE_DT, CHANGE_DEL_FLAG, SRC_ID, REF, REV_DT, REV_DEL_FLAG)
        VALUES (:classificationId, :axiomaId, :weight, :changeDt, :changeDelFlag, :srcId, :ref, :revDt, :revDelFlag)
        """
        for delFlag in [False, True]:
            insertValues = []
            for c in self.contracts:
                if c.insertNeeded:
                    insertValues.append(c.insertClassificationConstituentValues(delFlag))
            self.insertMany(insertQuery, insertValues)

    def insertTradingCurrency(self):
        insertQuery = """INSERT INTO future_dim_trading_currency
        (AXIOMA_ID, ID, CHANGE_DT, CHANGE_DEL_FLAG, SRC_ID, REF, REV_DT, REV_DEL_FLAG)
        VALUES (:axiomaId, :currencyId, :changeDt, :changeDelFlag, :srcId, :ref, :revDt, :revDelFlag)
        """
        for delFlag in [False, True]:
            insertValues = []
            for c in self.contracts:
                if c.insertNeeded:
                    insertValues.append(c.insertTradingCurrencyValues(delFlag))
            self.insertMany(insertQuery, insertValues)

    def insertName(self):
        insertQuery = """INSERT INTO future_dim_name
        (AXIOMA_ID, ID, CHANGE_DT, CHANGE_DEL_FLAG, SRC_ID, REF, REV_DT, REV_DEL_FLAG)
        VALUES (:axiomaId, :name, :changeDt, :changeDelFlag, :srcId, :ref, :revDt, :revDelFlag)
        """
        for delFlag in [False, True]:
            insertValues = []
            for c in self.contracts:
                if c.insertNeeded:
                    insertValues.append(c.insertNameValues(delFlag))
            self.insertMany(insertQuery, insertValues)

    def insertTicker(self):
        insertQuery = """INSERT INTO future_dim_ticker
        (AXIOMA_ID, ID, CHANGE_DT, CHANGE_DEL_FLAG, SRC_ID, REF, REV_DT, REV_DEL_FLAG)
        VALUES (:axiomaId, :ticker, :changeDt, :changeDelFlag, :srcId, :ref, :revDt, :revDelFlag)
        """
        for delFlag in [False, True]:
            insertValues = []
            for c in self.contracts:
                if c.insertNeeded:
                    insertValues.append(c.insertTickerValues(delFlag))
            self.insertMany(insertQuery, insertValues)

    def insertBBTicker(self):
        insertQuery = """INSERT INTO future_dim_Bloomberg_ticker
        (AXIOMA_ID, ID, CHANGE_DT, CHANGE_DEL_FLAG, SRC_ID, REF, REV_DT, REV_DEL_FLAG)
        VALUES (:axiomaId, :ticker, :changeDt, :changeDelFlag, :srcId, :ref, :revDt, :revDelFlag)
        """
        for delFlag in [False, True]:
            insertValues = []
            for c in self.contracts:
                if c.insertNeeded:
                    insertValues.append(c.insertBBTickerValues(delFlag))
            self.insertMany(insertQuery, insertValues)

class Contract():
#    EIF_CONTRACT_CLASSIFICATION_ID = 5360 # asset type = EIF Contract
    
    def __init__(self, axiomaId, series, contrCode=None, clsCode=None, futCode=None, 
                 ftidInfoCode=None, contrDate=None, ftidCode=None, ftidSeriesCode=None, 
                 expiryMonth=None, startDate=None, ref=None):
        self.axiomaId = axiomaId
        self.series = series
        self.contrCode = contrCode
        self.clsCode = clsCode
        self.futCode = futCode
        self.infoCode = ftidInfoCode
        self.contrDate = contrDate
        self.ftidCode = ftidCode
        self.ftidSeriesCode = ftidSeriesCode
        if hasattr(startDate, 'date'):
            self.startDate = startDate.date()
        else:
            self.startDate = startDate
        # self.startDate = max(datetime.date(2008,1,2),startDate.date())
        self.expiryMonth = expiryMonth
        if self.axiomaId:
            self.insertNeeded = False
            self.getAttributes()
        else:
            self.insertNeeded = True
        self.processed = False
        self.generateExpiryDate()
        self.series.getAttributes(asOfDate=self.expiryDate)
        self.generateTickers()
        self.generateName()
        if self.insertNeeded:
            self.generateAxiomaId()
            self.generateModelId()
            self.storeGeneratedValues()
        else:
            self.findModelIds()
        self.ref = ref or 'test bulk addition'
        
    def getAttributes(self):
        # look up all attributes for existing contract
        cursor = self.series.storer.marketDB.dbCursor
        if not self.startDate:
            cursor.execute("SELECT from_dt FROM asset_ref WHERE axioma_id=:axid", axid=self.axiomaId)
            r = cursor.fetchall()
            if len(r) == 1:
                self.startDate = r[0][0].date()
            else:
                raise ValueError("Can't find asset_ref for %s\n%s" % (self.axiomaId, r))
        if not self.contrCode:
            cursor.execute("""SELECT contr_code, cls_code, fut_code FROM future_dim_ds_active_int
                           WHERE axioma_id=:axid order by from_dt""", axid=self.axiomaId)
            r = cursor.fetchall()
            if len(r) == 1:
                r = r[0]
                self.contrCode = r[0]
                self.clsCode = r[1]
                self.futCode = r[2]
            elif len(r) > 1:
                logging.warning('More than one Datastream mapping for axioma ID %s: %s', self.axiomaId, r)
                self.contrCode = r[-1][0]
                self.clsCode = r[-1][1]
                self.futCode = r[-1][2]
            else:
                logging.warning('No Datastream mapping for axioma ID %s', self.axiomaId)
                self.contrCode = None
                self.clsCode = None
                self.futCode = None
        if not self.infoCode:
            cursor.execute("""SELECT ftid_infocode, ftid_code, ftid_series_code FROM
                           future_dim_ftid_active_int WHERE axioma_id=:axid order by from_dt""", axid=self.axiomaId)
            r = cursor.fetchall()
            if len(r) == 1:
                r = r[0]
                self.infoCode = r[0]
                self.ftidCode = r[1]
                self.ftidSeriesCode = r[2]
            elif len(r) > 1:
                logging.warning('More than one FTID mapping for axioma ID %s: %s', self.axiomaId, r)
                self.infoCode = r[-1][0]
                self.ftidCode = r[-1][1]
                self.ftidSeriesCode = r[-1][2]
            else:
                logging.warning('No FTID mapping for axioma ID %s', self.axiomaId)
                self.infoCode = None
                self.ftidCode = None
                self.ftidSeriesCode = None
        if not self.contrDate:
            cursor.execute("""SELECT expiration_month, last_trading_date, rev_dt FROM
                           future_linkage WHERE future_axioma_id=:axid
                           ORDER BY rev_dt DESC""", axid=self.axiomaId)
            r = cursor.fetchall()
            if len(r) > 0:
                self.expiryMonth = r[0][0].date()
                self.contrDate = self.expiryMonth.strftime('%m%y')
            else:
                raise ValueError("Can't determine expiry month for %s\n%s" % (self.axiomaId, r))

    def generateTickers(self):
        if self.series.secTypeId == COMMODITY_SECTYPE:
            self.ticker = '%s%s%s' % (self.series.extendedTicker, Series.months[self.contrDate[:2]], self.contrDate[2:])
        else:
            self.ticker = '%s-%s-%s' % (self.series.externalTicker, Series.months[self.contrDate[:2]], self.contrDate[2:])
        if not self.series.bbSwitchDt or self.expiryDate < self.series.bbSwitchDt:
            self.bbTicker = "%s%s%s" % (self.series.bbTicker, Series.months[self.contrDate[:2]], self.contrDate[3:])
        else:
            self.bbTicker = "%s%s%s" % (self.series.bbTicker, Series.months[self.contrDate[:2]], self.contrDate[2:])

    def generateName(self):
        self.name = '%s - %s' % (self.series.name.upper(), self.expiryMonth.strftime('%b %Y').upper())
        
    def generateAxiomaId(self):
        marketIds = self.series.storer.marketDB.createNewMarketIDs(1)
        marketId = marketIds[0]
        marketId.string = 'F' + marketId.string[1:]
#        print marketId.string
        self.axiomaId =  marketId.getIDString()

    def generateModelId(self):
        modelIds = self.series.storer.modelDB.createNewModelIDs(1)
        modelId = modelIds[0]
#        print modelId.string
        self.modelId =  modelId.getIDString()
        if self.series.secTypeId == COMMODITY_SECTYPE:
            self.subIssueId = self.modelId+"11"
        else:
            self.subIssueId = None
    
    def findModelIds(self):
        # NOTE: assumes only one market -> model mapping per contract
        c = self.series.storer.modelDB.dbCursor
        c.execute("""SELECT modeldb_id, sub_id FROM
        future_issue_map fim LEFT OUTER JOIN sub_issue si ON fim.modeldb_id=si.issue_id
        WHERE fim.marketdb_id = :axid""", axid=self.axiomaId)
        r = c.fetchall()
        if r:
            (self.modelId, self.subIssueId) = r[0]
        else:
            self.modelId = self.subIssueId = None
    
    def generateExpiryDate(self):
        y = int('20' + self.contrDate[2:])
        m = int(self.contrDate[:2])
        self.series.getAttributes(asOfDate=datetime.date(y, m, 1))
        self.expiryDate = None
        pattern = self.series.expirationPattern.strip()
        if pattern == 'DS_LTD':
            self.expiryDate = getDatastreamLTD(self)
            if self.expiryDate == None:
                self.expiryDate = getFTIDLTD(self)
            if self.expiryDate == None:
                pattern = 'LAST'
        if self.expiryDate == None:
            self.expiryDate = expirationDate(y, m, pattern, 
                                             self.series.tradingCountry,
                                             self.series.underlyingCountry,
                                             self.series.storer.marketDB)
      
    def storeGeneratedValues(self, asOfDate=None):
        if not asOfDate:
            asOfDate = self.series.storer.date
        if hasattr(asOfDate,'date'):
            asOfDate = asOfDate.date()
        futCode = '= %s' % self.futCode
        if not self.futCode:
            futCode = 'is null'
        infoCode = '= %s' % self.infoCode
        if not self.infoCode:
            infoCode = 'is null'
        sortableContrDate = '%s%s' % (self.contrDate[2:], self.contrDate[:2])
            
        q = '''
            update daily_new_futures set axioma_id = '%s', ticker = '%s' , name = '%s', sortable_contr_date = '%s' where FUTURE_FAMILY_ID = '%s' and FUT_CODE %s and FTID_INFOCODE %s
            and CONTR_DATE = '%s' and dt = :dt_arg
        ''' % (self.axiomaId, self.ticker, self.name, sortableContrDate, self.series.axiomaId, futCode, infoCode, self.contrDate)
        self.series.storer.dbCursor.execute(q, dt_arg=asOfDate)
#        self.series.storer.dbConnection.commit()

    def insertAssetRefValues(self):  
        valueDict = {}
        valueDict['axiomaId'] = self.axiomaId
        valueDict['fromDt'] = self.startDate
        # add one to actual expiry date since our thru dates are t+1
        valueDict['thruDt'] = self.expiryDate + datetime.timedelta(1)
        valueDict['srcId'] = self.series.srcId
        valueDict['ref'] = 'added via new EIFs by %s, Series: %s' % (self.series.userName, self.series.axiomaId)
        valueDict['addDt'] = self.series.storer.date
        valueDict['ctry'] = self.series.tradingCountry
        return valueDict

    def insertIssueValues(self):  
        valueDict = {}
        valueDict['modeldb_id'] = self.modelId
        valueDict['fromDt'] = max(datetime.date(2008,1,2), self.startDate)
        # add one to actual expiry date since our thru dates are t+1
        valueDict['thruDt'] = self.expiryDate + datetime.timedelta(1)
        return valueDict
    
    def insertIssueMapValues(self):  
        valueDict = {}
        valueDict['marketdb_id'] = self.axiomaId
        valueDict['modeldb_id'] = self.modelId
        valueDict['fromDt'] = max(datetime.date(2008,1,2), self.startDate)
        # add one to actual expiry date since our thru dates are t+1
        valueDict['thruDt'] = self.expiryDate + datetime.timedelta(1)
        valueDict['distribute'] = self.series.distribute
        return valueDict
    
    def insertSubIssueValues(self):
        valueDict = {}
        valueDict['issue_id'] = self.modelId
        valueDict['sub_id'] = self.subIssueId
        valueDict['fromDt'] = max(datetime.date(2008,1,2), self.startDate)
        # add one to actual expiry date since our thru dates are t+1
        valueDict['thruDt'] = self.expiryDate + datetime.timedelta(1)
        query = """SELECT rmg.rmg_id, rmg.mnemonic, rmg.description FROM RISK_MODEL_GROUP rmg
                where rmg.mnemonic=:abbr"""
        c = self.series.storer.modelDB.dbCursor
        c.execute(query, abbr=self.series.tradingCountry)
        r = c.fetchall()
        rmg = r[0][0]
        valueDict['rmg'] = rmg
        return valueDict  
    
    def insertFutureLinkageValues(self, delFlag):  
        valueDict = {}
        valueDict['familyId'] = self.series.axiomaId
        valueDict['axiomaId'] = self.axiomaId
        valueDict['expiryMonth'] = self.expiryMonth
        valueDict['lastTradeDate'] = self.expiryDate
        valueDict['ref'] = self.ref
        valueDict['revDt'] = self.series.storer.date
        valueDict['revDelFlag'] = 'N'
        return valueDict

    def insertFutureDimDatastreamValues(self, delFlag):  
        valueDict = {}
        if not self.futCode:
            return valueDict
        valueDict['axiomaId'] = self.axiomaId
        valueDict['contrCode'] = self.contrCode
        valueDict['clsCode'] = self.clsCode
        valueDict['futCode'] = self.futCode
        if delFlag:
            valueDict['changeDt'] = self.expiryDate + datetime.timedelta(1)
            valueDict['changeDelFlag'] = 'Y'
        else:
            valueDict['changeDt'] = self.dsStartDate or self.startDate
            valueDict['changeDelFlag'] = 'N'
        valueDict['srcId'] = self.series.dsSrcId
        valueDict['ref'] = 'Series: %s ; DsFutContrInfo, contrcode = %s' % (self.series.axiomaId, self.contrCode)
        valueDict['revDt'] = self.series.storer.date
        valueDict['revDelFlag'] = 'N'
        return valueDict

    def insertFutureDimFtidValues(self, delFlag):  
        valueDict = {}
        if not self.infoCode:
            return valueDict
        valueDict['axiomaId'] = self.axiomaId
        valueDict['ftidInfoCode'] = self.infoCode
        valueDict['ftidCode'] = self.ftidCode
        valueDict['ftidSeriesCode'] = self.ftidSeriesCode
        valueDict['contractDt'] = datetime.date(int('20' + self.contrDate[2:]), int(self.contrDate[:2]), 1)
        if delFlag:
            valueDict['changeDt'] = self.expiryDate + datetime.timedelta(1)
            valueDict['changeDelFlag'] = 'Y'
        else:
            valueDict['changeDt'] = self.idcStartDate or self.startDate
            valueDict['changeDelFlag'] = 'N'
        valueDict['srcId'] = self.series.ftidSrcId
        valueDict['ref'] = 'Series: %s ; CmInfo, infocode = %s' % (self.series.axiomaId, self.infoCode)
        valueDict['revDt'] = self.series.storer.date
        valueDict['revDelFlag'] = 'N'
        return valueDict

    def insertAssetTypeValues(self, delFlag):
        valueDict = {}
#        if not self.infoCode:
#            return valueDict
        valueDict['classificationId'] = self.series.secTypeId
        valueDict['axiomaId'] = self.axiomaId
        valueDict['weight'] = 1
        if delFlag:
            valueDict['changeDt'] = self.expiryDate + datetime.timedelta(1)
            valueDict['changeDelFlag'] = 'Y'
        else:
            valueDict['changeDt'] = self.startDate
            valueDict['changeDelFlag'] = 'N'
        valueDict['srcId'] = self.series.familySrcId
        valueDict['ref'] = 'Series: %s ; future_family_attribute.sectype_ID' % self.series.axiomaId
        valueDict['revDt'] = self.series.storer.date
        valueDict['revDelFlag'] = 'N'
        return valueDict

    def insertUsageValues(self, delFlag):
        valueDict = {}
        #        if not self.infoCode:
        #            return valueDict
        valueDict['classificationId'] = USAGE_MAP[self.series.secTypeId]
        valueDict['axiomaId'] = self.axiomaId
        valueDict['weight'] = 1
        if delFlag:
            valueDict['changeDt'] = self.expiryDate + datetime.timedelta(1)
            valueDict['changeDelFlag'] = 'Y'
        else:
            valueDict['changeDt'] = self.startDate
            valueDict['changeDelFlag'] = 'N'
        valueDict['srcId'] = self.series.familySrcId
        valueDict['ref'] = 'Series: %s ; future_family_attribute.sectype_ID' % self.series.axiomaId
        valueDict['revDt'] = self.series.storer.date
        valueDict['revDelFlag'] = 'N'
        return valueDict

    def insertClassificationConstituentValues(self, delFlag):  
        valueDict = {}
#        if not self.infoCode:
#            return valueDict
        valueDict['classificationId'] = self.series.classificationId
        valueDict['axiomaId'] = self.axiomaId
        valueDict['weight'] = 1
        if delFlag:
            valueDict['changeDt'] = self.expiryDate + datetime.timedelta(1)
            valueDict['changeDelFlag'] = 'Y'
        else:
            valueDict['changeDt'] = self.startDate
            valueDict['changeDelFlag'] = 'N'
        valueDict['srcId'] = self.series.familySrcId
        valueDict['ref'] = 'Series: %s ; future_family_attribute.stock_exchange_ID' % self.series.axiomaId
        valueDict['revDt'] = self.series.storer.date
        valueDict['revDelFlag'] = 'N'
        return valueDict

    def insertTradingCurrencyValues(self, delFlag):  
        valueDict = {}
#        if not self.infoCode:
#            return valueDict
        valueDict['axiomaId'] = self.axiomaId
        valueDict['currencyId'] = self.series.currencyId
        if delFlag:
            valueDict['changeDt'] = self.expiryDate + datetime.timedelta(1)
            valueDict['changeDelFlag'] = 'Y'
        else:
            valueDict['changeDt'] = self.startDate
            valueDict['changeDelFlag'] = 'N'
        valueDict['srcId'] = self.series.familySrcId
        valueDict['ref'] = 'Series: %s ; future_family_attribute.currency_id' % self.series.axiomaId
        valueDict['revDt'] = self.series.storer.date
        valueDict['revDelFlag'] = 'N'
        return valueDict

    def insertNameValues(self, delFlag):  
        valueDict = {}
#        if not self.infoCode:
#            return valueDict
        valueDict['axiomaId'] = self.axiomaId
        valueDict['name'] = self.name
        if delFlag:
            valueDict['changeDt'] = self.expiryDate + datetime.timedelta(1)
            valueDict['changeDelFlag'] = 'Y'
        else:
            valueDict['changeDt'] = self.startDate
            valueDict['changeDelFlag'] = 'N'
        valueDict['srcId'] = self.series.familySrcId
        valueDict['ref'] = 'Series: %s ; future_family_attribute.name' % self.series.axiomaId
        valueDict['revDt'] = self.series.storer.date
        valueDict['revDelFlag'] = 'N'
        return valueDict
  
    def insertTickerValues(self, delFlag):  
        valueDict = {}
#        if not self.infoCode:
#            return valueDict
        valueDict['axiomaId'] = self.axiomaId
        valueDict['ticker'] = self.ticker
        if delFlag:
            valueDict['changeDt'] = self.expiryDate + datetime.timedelta(1)
            valueDict['changeDelFlag'] = 'Y'
        else:
            valueDict['changeDt'] = self.startDate
            valueDict['changeDelFlag'] = 'N'
        valueDict['srcId'] = self.series.familySrcId
        valueDict['ref'] = 'Series: %s ; future_family_attribute.external_ticker' % self.series.axiomaId
        valueDict['revDt'] = self.series.storer.date
        valueDict['revDelFlag'] = 'N'
        return valueDict
  
    def insertBBTickerValues(self, delFlag):  
        valueDict = {}
        valueDict['axiomaId'] = self.axiomaId
        valueDict['ticker'] = self.bbTicker
        if delFlag:
            valueDict['changeDt'] = self.expiryDate + datetime.timedelta(1)
            valueDict['changeDelFlag'] = 'Y'
        else:
            valueDict['changeDt'] = self.startDate
            valueDict['changeDelFlag'] = 'N'
        valueDict['srcId'] = self.series.familySrcId
        valueDict['ref'] = 'Series: %s ; future_family_attribute.bloomberg_ticker' % self.series.axiomaId
        valueDict['revDt'] = self.series.storer.date
        valueDict['revDelFlag'] = 'N'
        return valueDict
  
    def __str__(self):
        res =  '  Contract(axiomaId:%s, contrCode:%s, clsCode:%s, futCode:%s, infoCode:%s, contrDate:%s, ticker:%s, name:%s, axiomaId:%s)' % (
            self.axiomaId, self.contrCode, self.clsCode, self.futCode, self.infoCode, self.contrDate, self.ticker, self.name, self.axiomaId)
        return res
    def __repr__(self):
        return self.__str__()

class EifStorer:
    def __init__(self, marketDb=None, modelDb=None, date=None):
        self.log = logging.getLogger('EifStorer')
        self.date = date
        if self.date == None:
            self.date = datetime.datetime.now()
        self.testOnly = False
        if not marketDb:
            connectParameters = {}
            connectParameters['sid'] = 'glsdg'
#            connectParameters['sid'] = 'glprod'
            connectParameters['user'] = 'marketdb_global'
            connectParameters['passwd'] = 'marketdb_global'
            self.marketDB = MarketDB.MarketDB(sid=connectParameters['sid'], user=connectParameters['user'], passwd=connectParameters['passwd'])
        else:
            self.marketDB = marketDb
        if not modelDb:
            connectParameters = {}
            connectParameters['sid'] = 'glsdg'
#            connectParameters['sid'] = 'glprod'
            connectParameters['user'] = 'modeldb_global'
            connectParameters['passwd'] = 'modeldb_global'
            self.modelDB = ModelDB.ModelDB(sid=connectParameters['sid'], user=connectParameters['user'], passwd=connectParameters['passwd'])
        else:
            self.modelDB = modelDb
        self.dbConnection = self.marketDB.dbConnection
        self.dbCursor = self.dbConnection.cursor()
        self.modeldbConnection = self.modelDB.dbConnection
        self.modeldbCursor = self.modeldbConnection.cursor()
        self.series = {}
    
#    def deleteSeries(self, id):
#        queries = []
#        q = '''delete from future_dim_datastream where AXIOMA_ID in (select FUTURE_AXIOMA_ID from future_linkage_active
#            where FUTURE_FAMILY_ID = :id_arg)
#        '''
#        queries.append(q)
#        q = '''delete from future_dim_ftid where AXIOMA_ID in (select FUTURE_AXIOMA_ID from future_linkage_active
#            where FUTURE_FAMILY_ID = :id_arg)
#        '''
#        queries.append(q)
#        q = '''delete from classification_constituent where AXIOMA_ID in (select FUTURE_AXIOMA_ID from future_linkage_active
#            where FUTURE_FAMILY_ID = :id_arg)
#        '''
#        queries.append(q)
#        q = '''delete from axiomadb.classification_constituent where AXIOMA_ID in (select FUTURE_AXIOMA_ID from future_linkage_active
#            where FUTURE_FAMILY_ID = :id_arg)
#        '''
#        queries.append(q)
#        q = '''delete from future_dim_name where AXIOMA_ID in (select FUTURE_AXIOMA_ID from future_linkage_active
#            where FUTURE_FAMILY_ID = :id_arg)
#        '''
#        queries.append(q)
#        q = '''delete from axiomadb.future_dim_name where AXIOMA_ID in (select FUTURE_AXIOMA_ID from future_linkage_active
#            where FUTURE_FAMILY_ID = :id_arg)
#        '''
#        queries.append(q)
#        q = '''delete from future_dim_ticker where AXIOMA_ID in (select FUTURE_AXIOMA_ID from future_linkage_active
#            where FUTURE_FAMILY_ID = :id_arg)
#        '''
#        queries.append(q)
#        q = '''delete from axiomadb.future_dim_ticker where AXIOMA_ID in (select FUTURE_AXIOMA_ID from future_linkage_active
#            where FUTURE_FAMILY_ID = :id_arg)
#        '''
#        queries.append(q)
#        q = '''delete from future_dim_trading_currency where AXIOMA_ID in (select FUTURE_AXIOMA_ID from future_linkage_active
#            where FUTURE_FAMILY_ID = :id_arg)
#        '''
#        queries.append(q)
#        q = '''delete from axiomadb.future_dim_trading_currency where AXIOMA_ID in (select FUTURE_AXIOMA_ID from future_linkage_active
#            where FUTURE_FAMILY_ID = :id_arg)
#        '''
#        queries.append(q)
#        q = '''delete from future_dim_ucp where AXIOMA_ID in (select FUTURE_AXIOMA_ID from future_linkage_active
#            where FUTURE_FAMILY_ID = :id_arg)
#        '''
#        queries.append(q)
#        q = '''delete from axiomadb.future_dim_ucp where AXIOMA_ID in (select FUTURE_AXIOMA_ID from future_linkage_active
#            where FUTURE_FAMILY_ID = :id_arg)
#        '''
#        queries.append(q)
#        q = '''delete from future_dim_open_interest where AXIOMA_ID in (select FUTURE_AXIOMA_ID from future_linkage_active
#            where FUTURE_FAMILY_ID = :id_arg)
#        '''
#        queries.append(q)
#        q = '''delete from axiomadb.future_dim_open_interest where AXIOMA_ID in (select FUTURE_AXIOMA_ID from future_linkage_active
#            where FUTURE_FAMILY_ID = :id_arg)
#        '''
#        queries.append(q)
#        q = '''delete from future_dim_return where AXIOMA_ID in (select FUTURE_AXIOMA_ID from future_linkage_active
#            where FUTURE_FAMILY_ID = :id_arg)
#        '''
#        queries.append(q)
#        q = '''delete from future_dim_tdv where AXIOMA_ID in (select FUTURE_AXIOMA_ID from future_linkage_active
#            where FUTURE_FAMILY_ID = :id_arg)
#        '''
#        queries.append(q)
#        q = '''delete from axiomadb.future_dim_tdv where AXIOMA_ID in (select FUTURE_AXIOMA_ID from future_linkage_active
#            where FUTURE_FAMILY_ID = :id_arg)
#        '''
#        queries.append(q)
#        q = '''delete from asset_ref where AXIOMA_ID in (select FUTURE_AXIOMA_ID from future_linkage_active
#            where FUTURE_FAMILY_ID = :id_arg)
#        '''
#        queries.append(q)
#        q = '''delete from modeldb_global.sub_issue_cumulative_return where SUB_ISSUE_ID in (select SUB_ID from modeldb_global.sub_issue where ISSUE_ID in (select MODELDB_ID from modeldb_global.future_issue_map where MARKETDB_ID  in (select FUTURE_AXIOMA_ID from future_linkage_active
#            where FUTURE_FAMILY_ID = :id_arg))
#        '''
#        queries.append(q)
#        q = '''delete from modeldb_global.sub_issue_return where SUB_ISSUE_ID in (select SUB_ID from modeldb_global.sub_issue where ISSUE_ID in (select MODELDB_ID from modeldb_global.future_issue_map where MARKETDB_ID  in (select FUTURE_AXIOMA_ID from future_linkage_active
#            where FUTURE_FAMILY_ID = :id_arg))
#        '''
#        queries.append(q)
#        q = '''delete from modeldb_global.sub_issue_data where SUB_ISSUE_ID in (select SUB_ID from modeldb_global.sub_issue where ISSUE_ID in (select MODELDB_ID from modeldb_global.future_issue_map where MARKETDB_ID  in (select FUTURE_AXIOMA_ID from future_linkage_active
#            where FUTURE_FAMILY_ID = :id_arg))
#        '''
#        queries.append(q)
#        q = '''delete from modeldb_global.rms_issue where ISSUE_ID in (select MODELDB_ID from modeldb_global.future_issue_map where MARKETDB_ID  in (select FUTURE_AXIOMA_ID from future_linkage_active
#            where FUTURE_FAMILY_ID = :id_arg))
#        '''
#        queries.append(q)
#        q = '''delete from modeldb_global.issue where ISSUE_ID in (select MODELDB_ID from modeldb_global.future_issue_map where MARKETDB_ID  in (select FUTURE_AXIOMA_ID from future_linkage_active
#            where FUTURE_FAMILY_ID = :id_arg))
#        '''
#        queries.append(q)
#        q = '''delete from modeldb_global.sub_issue where ISSUE_ID in (select MODELDB_ID from modeldb_global.future_issue_map where MARKETDB_ID  in (select FUTURE_AXIOMA_ID from future_linkage_active
#            where FUTURE_FAMILY_ID = :id_arg))
#        '''
#        queries.append(q)
#        q = '''delete from modeldb_global.future_issue_map where MARKETDB_ID  in (select FUTURE_AXIOMA_ID from future_linkage_active
#            where FUTURE_FAMILY_ID = :id_arg)
#        '''
#        queries.append(q)
#        q = '''delete from future_linkage where FUTURE_FAMILY_ID = :id_arg
#        '''
#        queries.append(q)
#        q = '''delete from axiomadb.future_linkage where FUTURE_FAMILY_ID = :id_arg
#        '''
#        queries.append(q)
#
#        for q in queries:
#            print q, id
#            try:
#                self.dbCursor.execute(q, id_arg=id)
#                self.dbConnection.commit()
##                self.dbConnection.rollback()
#            except:
#                continue

    def deleteContractByAxiomaId(self, axiomaId, commit=False):
        return self.deleteContract("'%s'"%axiomaId, commit)
    
    def deleteContract(self, id, commit=True):
        axiomaIds = """(
 %s  
        )""" % id
        queries = []
        q = '''delete from future_dim_datastream where AXIOMA_ID in %s
        ''' % axiomaIds
        queries.append(q)
        q = '''delete from future_dim_ftid where AXIOMA_ID in %s
        ''' % axiomaIds
        queries.append(q)
        q = '''delete from classification_constituent where AXIOMA_ID in %s
        ''' % axiomaIds
        queries.append(q)
        q = '''delete from axiomadb.classification_constituent where AXIOMA_ID in %s
        ''' % axiomaIds
        queries.append(q)
        q = '''delete from future_dim_name where AXIOMA_ID in %s
        ''' % axiomaIds
        queries.append(q)
        q = '''delete from axiomadb.future_dim_name where AXIOMA_ID in %s
        ''' % axiomaIds
        queries.append(q)
        q = '''delete from future_dim_ticker where axioma_id in %s
        ''' % axiomaIds
        queries.append(q)
        q = '''delete from axiomadb.future_dim_ticker where axioma_id in %s
        ''' % axiomaIds
        queries.append(q)
        q = '''delete from future_dim_bloomberg_ticker where axioma_id in %s
        ''' % axiomaIds
        queries.append(q)
        q = '''delete from axiomadb.future_dim_bloomberg_ticker where axioma_id in %s
        ''' % axiomaIds
        queries.append(q)
        q = '''delete from future_dim_trading_currency where AXIOMA_ID in %s
        ''' % axiomaIds
        queries.append(q)
        q = '''delete from axiomadb.future_dim_trading_currency where AXIOMA_ID in %s
        ''' % axiomaIds
        queries.append(q)
        q = '''delete from future_dim_ucp where AXIOMA_ID in %s
        ''' % axiomaIds
        queries.append(q)
        q = '''delete from axiomadb.future_dim_ucp where AXIOMA_ID in %s
        ''' % axiomaIds
        queries.append(q)
        q = '''delete from future_dim_open_interest where AXIOMA_ID in %s
        ''' % axiomaIds
        queries.append(q)
        q = '''delete from axiomadb.future_dim_open_interest where AXIOMA_ID in %s
        ''' % axiomaIds
        queries.append(q)
        q = '''delete from future_dim_return where AXIOMA_ID in %s
        ''' % axiomaIds
        queries.append(q)
        q = '''delete from future_dim_tdv where AXIOMA_ID in %s
        ''' % axiomaIds
        queries.append(q)
        q = '''delete from axiomadb.future_dim_tdv where AXIOMA_ID in %s
        ''' % axiomaIds
        queries.append(q)
        q = '''delete from asset_ref where AXIOMA_ID in %s
        ''' % axiomaIds
        queries.append(q)

        q = '''delete from modeldb_global.sub_issue_cumulative_return where 
                    SUB_ISSUE_ID in (select SUB_ID from modeldb_global.sub_issue where 
                    ISSUE_ID in (select MODELDB_ID from modeldb_global.future_issue_map where MARKETDB_ID  in %s))
        ''' % axiomaIds
        queries.append(q)
        q = '''delete from modeldb_global.sub_issue_return where 
                    SUB_ISSUE_ID in (select SUB_ID from modeldb_global.sub_issue where 
                    ISSUE_ID in (select MODELDB_ID from modeldb_global.future_issue_map where MARKETDB_ID  in %s))
        ''' % axiomaIds
        queries.append(q)
        q = '''delete from modeldb_global.sub_issue_data where 
                    SUB_ISSUE_ID in (select SUB_ID from modeldb_global.sub_issue where 
                    ISSUE_ID in (select MODELDB_ID from modeldb_global.future_issue_map where MARKETDB_ID  in %s))
        ''' % axiomaIds
        queries.append(q)
        q = '''delete from modeldb_global.rms_issue where ISSUE_ID in (select MODELDB_ID from modeldb_global.future_issue_map where MARKETDB_ID in %s)
        ''' % axiomaIds
        queries.append(q)
        
        q = '''delete from modeldb_global.issue where ISSUE_ID in (select MODELDB_ID from modeldb_global.future_issue_map where MARKETDB_ID  in %s)
        ''' % axiomaIds
        queries.append(q)
        
        q = '''delete from modeldb_global.sub_issue where ISSUE_ID in (select MODELDB_ID from modeldb_global.future_issue_map where MARKETDB_ID  in  %s)
        ''' % axiomaIds
        queries.append(q)
                
        q = '''delete from modeldb_global.future_issue_map where MARKETDB_ID  in %s
        ''' % axiomaIds
        queries.append(q)
        q = '''delete from future_linkage where FUTURE_AXIOMA_ID in %s
        ''' % axiomaIds
        queries.append(q)
        q = '''delete from axiomadb.future_linkage where FUTURE_AXIOMA_ID in %s
        ''' % axiomaIds
        queries.append(q)

        for q in queries:
#            print q
            self.log.debug('query to execute %s' % q)
            try:
                self.dbCursor.execute(q)
#                if commit:
#                    self.dbConnection.commit()
##                self.dbConnection.rollback()
            except Exception as e:
                self.log.debug('an error occured when executing sql %s %s' % (q, e))
                #raise e
                continue
        self.log.debug('all %d queries executed'%len(queries))
        
        if commit:
            self.dbConnection.commit()
#                self.dbConnection.rollback()
        
#    def deleteUnlinked(self):
#        queries = []
#        q = '''delete from future_dim_datastream where AXIOMA_ID in (select distinct axioma_id from asset_Ref where axioma_ID like 'F%' and axioma_ID not in (
#select future_axioma_ID from future_linkage_active))
#        ''' 
#        queries.append(q)
#        q = '''delete from future_dim_ftid where AXIOMA_ID in (select distinct axioma_id from asset_Ref where axioma_ID like 'F%' and axioma_ID not in (
#select future_axioma_ID from future_linkage_active))
#        ''' 
#        queries.append(q)
#        q = '''delete from classification_constituent where AXIOMA_ID in (select distinct axioma_id from asset_Ref where axioma_ID like 'F%' and axioma_ID not in (
#select future_axioma_ID from future_linkage_active))
#        ''' 
#        queries.append(q)
#        q = '''delete from axiomadb.classification_constituent where AXIOMA_ID in (select distinct axioma_id from asset_Ref where axioma_ID like 'F%' and axioma_ID not in (
#select future_axioma_ID from future_linkage_active))
#        ''' 
#        queries.append(q)
#        q = '''delete from future_dim_name where AXIOMA_ID in (select distinct axioma_id from asset_Ref where axioma_ID like 'F%' and axioma_ID not in (
#select future_axioma_ID from future_linkage_active))
#        ''' 
#        queries.append(q)
#        q = '''delete from future_dim_ticker where AXIOMA_ID in (select distinct axioma_id from asset_Ref where axioma_ID like 'F%' and axioma_ID not in (
#select future_axioma_ID from future_linkage_active))
#        ''' 
#        queries.append(q)
#        q = '''delete from future_dim_trading_currency where AXIOMA_ID in (select distinct axioma_id from asset_Ref where axioma_ID like 'F%' and axioma_ID not in (
#select future_axioma_ID from future_linkage_active))
#        ''' 
#        queries.append(q)
#        q = '''delete from future_dim_ucp where AXIOMA_ID in (select distinct axioma_id from asset_Ref where axioma_ID like 'F%' and axioma_ID not in (
#select future_axioma_ID from future_linkage_active))
#        ''' 
#        q = '''delete from axiomadb.future_dim_ucp where AXIOMA_ID in (select distinct axioma_id from asset_Ref where axioma_ID like 'F%' and axioma_ID not in (
#select future_axioma_ID from future_linkage_active))
#        ''' 
#        queries.append(q)
#        q = '''delete from future_dim_open_interest where AXIOMA_ID in (select distinct axioma_id from asset_Ref where axioma_ID like 'F%' and axioma_ID not in (
#select future_axioma_ID from future_linkage_active))
#        ''' 
#        queries.append(q)
#        q = '''delete from axiomadb.future_dim_open_interest where AXIOMA_ID in (select distinct axioma_id from asset_Ref where axioma_ID like 'F%' and axioma_ID not in (
#select future_axioma_ID from future_linkage_active))
#        ''' 
#        queries.append(q)
#        q = '''delete from future_dim_return where AXIOMA_ID in (select distinct axioma_id from asset_Ref where axioma_ID like 'F%' and axioma_ID not in (
#select future_axioma_ID from future_linkage_active))
#        ''' 
#        queries.append(q)
#        q = '''delete from future_dim_tdv where AXIOMA_ID in (select distinct axioma_id from asset_Ref where axioma_ID like 'F%' and axioma_ID not in (
#select future_axioma_ID from future_linkage_active))
#        ''' 
#        queries.append(q)
#        q = '''delete from axiomadb.future_dim_tdv where AXIOMA_ID in (select distinct axioma_id from asset_Ref where axioma_ID like 'F%' and axioma_ID not in (
#select future_axioma_ID from future_linkage_active))
#        ''' 
#        queries.append(q)
#        q = '''delete from asset_ref where AXIOMA_ID in (select distinct axioma_id from asset_Ref where axioma_ID like 'F%' and axioma_ID not in (
#select future_axioma_ID from future_linkage_active))
#        ''' 
#        queries.append(q)
#        q = '''delete from modeldb_global.issue where ISSUE_ID in (select MODELDB_ID from modeldb_global.future_issue_map where MARKETDB_ID  in (select distinct axioma_id from asset_Ref where axioma_ID like 'F%' and axioma_ID not in (
#select future_axioma_ID from future_linkage_active)))
#        ''' 
#        queries.append(q)
#        q = '''delete from modeldb_global.future_issue_map where MARKETDB_ID  in (select distinct axioma_id from asset_Ref where axioma_ID like 'F%' and axioma_ID not in (
#select future_axioma_ID from future_linkage_active))
#        ''' 
#        queries.append(q)
#
#        for q in queries:
#            print q
#            try:
#                self.dbCursor.execute(q)
#                self.dbConnection.commit()
##                self.dbConnection.rollback()
#            except:
#                continue
            
    def deleteAllSeries(self):
        q = 'select distinct id from future_family_ref'
        self.dbCursor.execute(q)
        res = self.dbCursor.fetchall()
        for each in res:
            self.deleteSeries(each[0])

    def deleteSeries(self, id):
        q = '''select distinct  FUTURE_AXIOMA_ID from future_linkage_active
           where FUTURE_FAMILY_ID = '%s'
           ''' % id
        self.dbCursor.execute(q)
        res = self.dbCursor.fetchall()
        for each in res:
            self.log.info('Deleting contract %s' % each[0])
            self.deleteContract("'%s'" % each[0])

    def deleteUnlinked(self):
        q = '''select distinct axioma_id from asset_Ref where axioma_ID like'F%' and AXIOMA_ID not in(
                select future_axioma_ID from future_linkage)
        '''
        self.dbCursor.execute(q)
        res = self.dbCursor.fetchall()
        for each in res:
            self.log.info('Deleting contract %s' % each[0])
            self.deleteContract("'%s'" % each[0])

    def processSeries(self, date, rerun = False):
        self.date = date
        q = '''
            select FUTURE_FAMILY_ID, AXIOMA_ID, CONTR_CODE, CLS_CODE, FUT_CODE, FTID_INFOCODE, CONTR_DATE, FTID_CODE, FTID_SERIES_CODE, EXPIRATION_MONTH, START_DATE from daily_new_futures where dt = :dt_arg and axioma_id
        '''
        if rerun:
            q += ' is not null'
        else:
            q += ' is null'
#        q = '''
#            select FUTURE_FAMILY_ID, AXIOMA_ID, CONTR_CODE, CLS_CODE, FUT_CODE, FTID_INFOCODE, CONTR_DATE, FTID_CODE, FTID_SERIES_CODE from daily_new_futures where dt = :dt_arg and rownum = 1
#        '''
        self.dbCursor.execute(q, dt_arg=date)
        res = self.dbCursor.fetchall()
        for each in res:
            id = each[0]
            if id in ['FUT_999999']:
                continue
#            print each[6]
            if not each[6][:2].isdigit(): #some non-numeric commodity contract dates - to be ignored
                continue
            if id not in self.series:
                print('getting contracts for', id)
                self.series[id] = Series(id, self)
            s = self.series[id]
            c = Contract(each[1], s, each[2],each[3],each[4],each[5], each[6], each[7], each[8], each[9], each[10])
            s.contracts.append(c)
        for s in self.series.values():
            if not s.hasAttributes:
                print('skipping %s, no attributes' % s)
                continue
            print('Processing series %s' % s)
            pp.pprint(s.contracts)
            s.insertAssetRef()
            s.insertFutureLinkage()
            s.insertFutureDimDatastream()
            s.insertFutureDimFtid()
            s.insertAssetType()
            s.insertClassificationConstituent()
            s.insertTradingCurrency()
            s.insertName()
            s.insertTicker()
            s.insertIssueMap()
            s.insertSubIssueMap()
        if self.testOnly:
            self.marketDB.revertChanges()
            self.modelDB.revertChanges()
        else:
            self.marketDB.commitChanges()
            self.modelDB.commitChanges()

    def runTransfers(self):
#        os.environ["PYTHONSTARTUP"] = '%s/set_oracle.py' % commands.getstatusoutput('pwd')[1]
#        print os.environ["PYTHONSTARTUP"]
        
#        cmd = "%s/runTransfers '-n production.config MarketDB:sid=glsdg AxiomaDB:sid=glsdg dates=2012-05-29 VendorDB:sid=glsdg CompuStatDB:sid=glsdg TejDB:sid=glsdg axioma-ids=F00O0MD12M sections=EIF-OpenInterest'" % commands.getstatusoutput('pwd')[1]
        dateRange = '2012-01-01:2012-11-23'
        axiomaIds = []
        for s in self.series.values():
            for c in s.contracts:
                axiomaIds.append(c.axiomaId)
        axiomaIdsString = ','.join(axiomaIds)
#        print axiomaIdsString
        
#        sections = ['EIF-Price', 'EIF-Volume']
#        sections = ['Return']
        sections = ['EIF-Price', 'EIF-Volume', 'EIF-OpenInterest', 'Return']
        for section in sections:
            transferParams = """production.config MarketDB:sid=glsdg AxiomaDB:sid=glsdg dates=%s
            VendorDB:sid=glsdg CompuStatDB:sid=glsdg TejDB:sid=glsdg axioma-ids=%s 
            sections=%s""" % (dateRange, axiomaIdsString, section)
            cmd = "%s/runTransfers '%s'" % (subprocess.getstatusoutput('pwd')[1], transferParams)
            print(cmd)
            res = subprocess.getstatusoutput(cmd)
            for ln in res:
                print(ln)     
            
    def runTransfersOld(self):
#        os.environ["PYTHONSTARTUP"] = '%s/set_oracle.py' % commands.getstatusoutput('pwd')[1]
#        print os.environ["PYTHONSTARTUP"]
        
#        cmd = "%s/runTransfers '-n production.config MarketDB:sid=glsdg AxiomaDB:sid=glsdg dates=2012-05-29 VendorDB:sid=glsdg CompuStatDB:sid=glsdg TejDB:sid=glsdg axioma-ids=F00O0MD12M sections=EIF-OpenInterest'" % commands.getstatusoutput('pwd')[1]
        dateRange = '2008-01-01:2012-10-23'
        for s in self.series.values():
            for c in s.contracts:
                axiomaIdsString = c.axiomaId
                print(axiomaIdsString)
                
                transferParams = """-n production.config MarketDB:sid=glsdg AxiomaDB:sid=glsdg dates=%s
                VendorDB:sid=glsdg CompuStatDB:sid=glsdg TejDB:sid=glsdg axioma-ids=%s 
                sections=EIF-Price""" % (dateRange, axiomaIdsString)
                cmd = "%s/runTransfers '%s'" % (subprocess.getstatusoutput('pwd')[1], transferParams)
                print(cmd)
                res = subprocess.getstatusoutput(cmd)
                for ln in res:
                    print(ln)     

def testExpiryDates():
    # ID          LAST_TRADING_DATE_PATTERN    
    # ----------  --------------------- 
    # FUT_000093  FR-3B-BH|AT,RU,GB     
    # FUT_000105  LAST-1                
    # FUT_000017  TH_1E_BU              
    # FUT_000097  TH_2B_BH           
    # FUT_000095  WE_C15_AH          
    # FUT_000016  TH_1E_BH              
    # FUT_000091  TH_3B_BH           
    # FUT_000088  LAST-1                
    # FUT_000084  LAST-1
    # FUT_100022  DS_LTD

    print('\t'.join(['AXIOMA_ID','TR','UN','EXPIRATION','CONDT','EXPDT']))
    storer = EifStorer()
    series = Series('FUT_000093', storer)
    contract = Contract('FTEST12345', series, 'FUT_000093', 1, 2, 3, '0308', 4, 5, datetime.date(2008,3,1), datetime.datetime.now(), 'testing')
    #FUT_000093    AT    RU    FR_3B_BH|AT,RU,GB    0308    2008-03-20
    print('\t'.join([contract.series.axiomaId, contract.series.tradingCountry, contract.series.underlyingCountry, contract.series.expirationPattern, contract.contrDate, str(contract.expiryDate)]))
    assert str(contract.expiryDate)=='2008-03-20'
    series = Series('FUT_000093', storer)
    contract = Contract('FTEST12345', series, 'FUT_000093', 1, 2, 3, '0808', 4, 5, datetime.date(2008,8,1), datetime.datetime.now(), 'testing')
    #FUT_000093    AT    RU    FR_3B_BH|AT,RU,GB    0808    2008-08-14
    print('\t'.join([contract.series.axiomaId, contract.series.tradingCountry, contract.series.underlyingCountry, contract.series.expirationPattern, contract.contrDate, str(contract.expiryDate)]))
    assert str(contract.expiryDate)=='2008-08-14'
    series = Series('FUT_000105', storer)
    contract = Contract('FTEST23456', series, 'FUT_000105', 1, 2, 3, '0507', 4, 5, datetime.date(2007,5,1), datetime.datetime.now(), 'testing')
    #FUT_000105    TH    TH    LAST-1    0608    2007-05-29
    print('\t'.join([contract.series.axiomaId, contract.series.tradingCountry, contract.series.underlyingCountry, contract.series.expirationPattern, contract.contrDate, str(contract.expiryDate)]))
    assert str(contract.expiryDate)=='2007-05-29'
    series = Series('FUT_000017', storer)
    contract = Contract('FTEST34567', series, 'FUT_000017', 1, 2, 3, '1008', 4, 5, datetime.date(2008,10,1), datetime.datetime.now(), 'testing')
    #FUT_000017    SG    IN    TH_1E_BU    1008    2008-10-29
    print('\t'.join([contract.series.axiomaId, contract.series.tradingCountry, contract.series.underlyingCountry, contract.series.expirationPattern, contract.contrDate, str(contract.expiryDate)]))
    assert str(contract.expiryDate)=='2008-10-29'
    series = Series('FUT_000097', storer)
    contract = Contract('FTEST45678', series, 'FUT_000097', 1, 2, 3, '0608', 4, 5, datetime.date(2008,6,1), datetime.datetime.now(), 'testing')
    #FUT_000097    KR    KR    TH_2B_BH    0608    2008-06-12
    print('\t'.join([contract.series.axiomaId, contract.series.tradingCountry, contract.series.underlyingCountry, contract.series.expirationPattern, contract.contrDate, str(contract.expiryDate)]))
    assert str(contract.expiryDate)=='2008-06-12'
    series = Series('FUT_000095', storer)
    contract = Contract('FTEST56789', series, 'FUT_000095', 1, 2, 3, '1106', 4, 5, datetime.date(2008,6,1), datetime.datetime.now(), 'testing')
    #FUT_000095    BR    BR    WE_C15_AH    1106    2006-11-16
    print('\t'.join([contract.series.axiomaId, contract.series.tradingCountry, contract.series.underlyingCountry, contract.series.expirationPattern, contract.contrDate, str(contract.expiryDate)]))
    assert str(contract.expiryDate)=='2006-11-16'
    series = Series('FUT_000016', storer)
    contract = Contract('FTEST67890', series, 'FUT_000016', 1, 2, 3, '0608', 4, 5, datetime.date(2008,6,1), datetime.datetime.now(), 'testing')
    #FUT_000016    IN    IN    TH_1E_BH    0608    2008-06-26
    print('\t'.join([contract.series.axiomaId, contract.series.tradingCountry, contract.series.underlyingCountry, contract.series.expirationPattern, contract.contrDate, str(contract.expiryDate)]))
    assert str(contract.expiryDate)=='2008-06-26'
    series = Series('FUT_000091', storer)
    contract = Contract('FTEST78901', series, 'FUT_000091', 1, 2, 3, '0504', 4, 5, datetime.date(2004,5,1), datetime.datetime.now(), 'testing')
    #FUT_000091    DE    CH    TH_3B_BH    0504    2004-05-21
    print('\t'.join([contract.series.axiomaId, contract.series.tradingCountry, contract.series.underlyingCountry, contract.series.expirationPattern, contract.contrDate, str(contract.expiryDate)]))
    assert str(contract.expiryDate)=='2004-05-21'
    series = Series('FUT_000091', storer)
    contract = Contract('FTEST78901', series, 'FUT_000091', 1, 2, 3, '0403', 4, 5, datetime.date(2003,4,1), datetime.datetime.now(), 'testing')
    #FUT_000091    DE    CH    TH_3B_BH    0403    2003-04-17
    print('\t'.join([contract.series.axiomaId, contract.series.tradingCountry, contract.series.underlyingCountry, contract.series.expirationPattern, contract.contrDate, str(contract.expiryDate)]))
    assert str(contract.expiryDate)=='2003-04-17'
    series = Series('FUT_000003', storer)
    contract = Contract('FTEST03901', series, 'FUT_000003', 1, 2, 3, '0308', 4, 5, datetime.date(2010,5,1), datetime.datetime.now(), 'testing')
    #FUT_000003    US    US    FR_3B_-1_BH    0510    2008-03-20
    print('\t'.join([contract.series.axiomaId, contract.series.tradingCountry, contract.series.underlyingCountry, contract.series.expirationPattern, contract.contrDate, str(contract.expiryDate)]))
    assert str(contract.expiryDate)=='2008-03-20'
    series = Series('FUT_000003', storer)
    contract = Contract('FTEST03901', series, 'FUT_000003', 1, 2, 3, '1111', 4, 5, datetime.date(2010,5,1), datetime.datetime.now(), 'testing')
    #FUT_000003    US    US    FR_3B_-1_BH    0510    2011-11-17
    print('\t'.join([contract.series.axiomaId, contract.series.tradingCountry, contract.series.underlyingCountry, contract.series.expirationPattern, contract.contrDate, str(contract.expiryDate)]))
    assert str(contract.expiryDate)=='2011-11-17'
    series = Series('FUT_000013', storer)
    contract = Contract('FTEST13901', series, 'FUT_000013', 1, 2, 3, '0400', 4, 5, datetime.date(2010,5,1), datetime.datetime.now(), 'testing')
    #FUT_000013    CA    CA    FR_3B_BH_-1    0510    2000-04-19
    print('\t'.join([contract.series.axiomaId, contract.series.tradingCountry, contract.series.underlyingCountry, contract.series.expirationPattern, contract.contrDate, str(contract.expiryDate)]))
    assert str(contract.expiryDate)=='2000-04-19'
    series = Series('FUT_000013', storer)
    contract = Contract('FTEST13901', series, 'FUT_000013', 1, 2, 3, '0401', 4, 5, datetime.date(2010,5,1), datetime.datetime.now(), 'testing')
    #FUT_000013    CA    CA    FR_3B_BH_-1    0510    2001-04-19
    print('\t'.join([contract.series.axiomaId, contract.series.tradingCountry, contract.series.underlyingCountry, contract.series.expirationPattern, contract.contrDate, str(contract.expiryDate)]))
    assert str(contract.expiryDate)=='2001-04-19'
    series = Series('FUT_000102', storer)
    contract = Contract('FTEST10901', series, 'FUT_000102', 1, 2, 3, '0210', 4, 5, datetime.date(2010,5,1), datetime.datetime.now(), 'testing')
    #FUT_000102    JP    JP    FR_2B_-1_BH    0210    2010-02-10
    print('\t'.join([contract.series.axiomaId, contract.series.tradingCountry, contract.series.underlyingCountry, contract.series.expirationPattern, contract.contrDate, str(contract.expiryDate)]))
    assert str(contract.expiryDate)=='2010-02-10'
    series = Series('FUT_000078', storer)
    contract = Contract('FTEST78901', series, 'FUT_000078', 1, 2, 3, '0510', 4, 5, datetime.date(2010,5,1), datetime.datetime.now(), 'testing')
    #FUT_000078    SG    SG    LAST-1    0510    2010-05-27
    print('\t'.join([contract.series.axiomaId, contract.series.tradingCountry, contract.series.underlyingCountry, contract.series.expirationPattern, contract.contrDate, str(contract.expiryDate)]))
    assert str(contract.expiryDate)=='2010-05-27'
    series = Series('FUT_100022', storer)
    contract = Contract('FTEST89012', series, 'FUT_100022', 1986, 226983, None, '1108', 4, 5, datetime.date(2010,5,1), datetime.datetime.now(), 'testing')
    #FUT_100022    SG    SG    DS_LTD    1014    2008-11-20
    print('\t'.join([contract.series.axiomaId, contract.series.tradingCountry, contract.series.underlyingCountry, contract.series.expirationPattern, contract.contrDate, str(contract.expiryDate)]))
    assert str(contract.expiryDate)=='2008-11-20'
    
    
if __name__ == '__main__':
    usage = "usage: %prog [options]"
    cmdlineParser = optparse.OptionParser(usage=usage)

    Utilities.addDefaultCommandLine(cmdlineParser)
    (options, args) = cmdlineParser.parse_args()
    
#    Utilities.processDefaultCommandLine(options, cmdlineParser)
#    testExpiryDates()
#    storer.deleteSeries('FUT_100082')
#    storer.deleteSeries('FUT_000016')
#    storer.deleteSeries('FUT_000089')
#    storer.deleteSeries('FUT_000022')
#    storer.deleteAllSeries()
    storer = EifStorer()
#    storer.deleteAllSeries()
#    storer.processSeries(datetime.date.today(), False)
#    testExpiryDates()
#    storer.deleteContract("'FHYM89GW73', 'F9MQDNTKH0'") 
#    storer.deleteSeries('FUT_199998') 
#    storer.deleteUnlinked()  

#    for each in ["FUT_100001","FUT_100006","FUT_100023","FUT_100026","FUT_100004","FUT_100020","FUT_100003","FUT_100022","FUT_100024","FUT_100009","FUT_100016","FUT_100028","FUT_100029",
#                 "FUT_100019","FUT_100018","FUT_100007","FUT_100012","FUT_100017","FUT_100025","FUT_100021","FUT_100027","FUT_100005","FUT_100002","FUT_100014"]:
#        storer.deleteSeries(each)
#    storer.deleteSeries('FUT_000020')
#    storer.deleteSeries('FUT_000006')
#    storer.deleteSeries('FUT_000007')
#    storer.deleteSeries('FUT_000009')
#    storer.deleteSeries('FUT_000013')
#    storer.deleteSeries('FUT_000078')
#    storer.deleteSeries('FUT_000100')
#    storer.deleteSeries('FUT_000101')
#    storer.deleteSeries('FUT_000102')
#    storer.deleteSeries('FUT_000103')
#    storer.deleteSeries('FUT_000104')
#    storer.deleteSeries('FUT_000108')
#    storer.deleteSeries('FUT_000048')
#    storer.runTransfers()
