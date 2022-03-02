import datetime
import numpy
import numpy.ma as ma
import numpy.linalg as linalg
import logging
import optparse
import os
import sys
import time
import math
from marketdb import  MarketDB
from marketdb import MarketID
from marketdb.Utilities import listChunkIterator
import riskmodels
from riskmodels.Matrices import ExposureMatrix
from riskmodels import ModelDB, modelNameMap
from riskmodels import MFM
from riskmodels import RiskModels
from riskmodels import Utilities
from riskmodels import ModelID
from riskmodels import PhoenixModels
BDI_COUNTRY_CODES = set(['AI','AG','BS','BB','BZ','BM',
                         'VG','KY','CK','FO','GI','IM',
                         'LR','MH','AN','PA','TC','JE','GG'])

MODEL_INDEX_MAP = {'AXGB': ['FTSE 100', 
                            'FTSE 250', 
                            'FTSE 350', 
                            'FTSE SmallCap', 
                            'FTSE Fledgling', 
                            'FTSE AIM 100',
                            'FTSE All-Share'],
                   'AXCA': ['S&P/TSX Equity 60 Index',
                            'S&P/TSX MidCap Index',
                            'S&P/TSX SmallCap Index',
                            'S&P/TSX Income Trust Index',
                            'S&P/TSX Capped Energy Trust Index',
                            'S&P/TSX Capped Diversified Mining Index',
                            'S&P/TSX Composite Index'],
                   'AXAU': ['S&P ASX 100',
                            'S&P ASX MC 50',
                            'S&P ASX 200',
                            'S&P ASX 300',
                            'S&P ASX Small Ordinaries',
                            'S&P ASX All Ordinaries'],
                   'AXJP': ['Russell Japan Large Cap',
                            'Russell Japan Mid Cap',
                            'Russell Japan Small Cap',
                            'Russell Japan'],
                   'AXUS': ['S and P 500',
                            'S and P 400',
                            'S and P 600',
                            'S and P 1500',
                            'RUSSELL TOP200',
                            'RUSSELL MIDCAP',
                            'RUSSELL 1000',
                            'RUSSELL 2000',
                            'RUSSELL 3000',
                            'RUSSELL SMALLCAP',
                            'RUSSELL MICROCAP',
                            'RUSSELL 1000 VALUE',
                            'RUSSELL 2000 VALUE',
                            'RUSSELL 3000 VALUE', 
                            'RUSSELL 1000 GROWTH',
                            'RUSSELL 2000 GROWTH',
                            'RUSSELL 3000 GROWTH'],
                   'AXEM': ['FTSE Advanced Emerging',
                            'FTSE Secondary Emerging',
                            'FTSE All-World BRIC',
                            'FTSE Emerging',
                            'FTSE Europe',
                            'FTSE Asia Pacific',
                            'FTSE Latin America',
                            'FTSE Middle East & Africa'],
                   'AXEU': ['FTSE Eurobloc',
                            'FTSE Nordic',
                            'FTSE Developed Europe',
                            'FTSE Emerging Europe All Cap',
                            'FTSE Europe',
                            'FTSE 100',
                            'FTSE France All Cap',
                            'FTSE Germany All Cap',
                            'FTSE Russia All Cap'],
                   'AXWW': ['FTSE Asia Pacific',
                            'FTSE Europe',
                            'FTSE All-World BRIC',
                            'FTSE Developed Europe-Asia Pacific',
                            'FTSE Emerging',
                            'FTSE Greater China',
                            'FTSE Eurobloc',
                            'FTSE Developed Europe',
                            'FTSE Developed Asia Pacific',
                            'FTSE Developed',
                            'FTSE All-World'],
                   'AXTW': ['FTSE-TSEC 50',
                            'FTSE-TSEC Mid-Cap 100',
                            'FTSE-TSEC Technology',
                            'TSE Electronics',
                            'TSE Electric and Machinery',
                            'OTC Index',
                            'OTC Electronics',
                            'TSE TAIEX'],
                   'AXCN': ['FTSE/Xinhua A 50',
                            'FTSE/Xinhua A 200',
                            'FTSE/Xinhua A 400',
                            'FTSE/Xinhua A 600',
                            'FTSE/Xinhua A Small Cap',
                            'FTSE/Xinhua A Allshare',
                            'FTSE/Xinhua B 35'],
                   'AXAP': ['FTSE Asia Pacific',
                            'FTSE Asean',
                            'FTSE Greater China',
                            'FTSE Developed Asia Pacific',
                            'FTSE Emerging'],
                   'AXNA': ['RUSSELL 1000',
                            'RUSSELL 2000',
                            'RUSSELL 3000',
                            'RUSSELL MICROCAP',
                            'S&P/TSX Composite Index',
                            'S&P/TSX Equity 60 Index',
                            'S&P/TSX MidCap Index',
                            'S&P/TSX SmallCap Index',
                            'FTSE North America'],
                }

def formatFactorName(f, fTypeMap):
    if fTypeMap[f.name]==ExposureMatrix.CurrencyFactor:
        return '%s (%s)' % (f.name, f.description)
    else:
        return f.description

class ModelReport:
    def __init__(self, modelSelector, marketDB, modelAssetType):
        self.modelSelector = modelSelector
        self.buffer = ''
        self.maxRecords = 50
        self.marketDB = marketDB
        self.modelAssetType = modelAssetType
    
    def emailReport(self, date, recipientList):
        import smtplib
        import email.utils
        import email.mime.text
        smtp_server = 'mail.axiomainc.com'
        #smtp_user = 'rmail'
        #smtp_pass = 'wtSL0wGOP'
        session = smtplib.SMTP(smtp_server)
        #session.login(smtp_user, smtp_pass)
        
        sender = 'ops-rm@axiomainc.com'
        user = os.environ['USER']
        assert(len(recipientList) > 0)
        message = email.mime.text.MIMEText(self.buffer)
        message.set_charset('utf-8')
        message['From'] = email.utils.formataddr(('Risk Model Operations', sender))
        message['Subject'] = '%04d-%02d-%02d %s Risk Model QA Report run by %s' % \
                    (date.year, date.month, date.day, self.modelSelector.mnemonic, user)
        message['To'] = ', '.join(recipientList)
        try:
            session.sendmail(sender, recipientList, message.as_string())
        finally:
            session.quit()
    
    def writeReport(self, filename):
        outfile = open(filename, 'w')
        outfile.write(self.buffer)
        outfile.close()
    
    def getTopRecordsIndices(self, capMap, assets, showTop=None):
        rank = numpy.argsort([-capMap.get(n, 0.0) for n in assets])
        if showTop:
            rank = rank[:showTop]
        elif len(assets) > self.maxRecords:
            rank = rank[:self.maxRecords]
        return rank

    def displayMktAssetListDetails(self, ids, date, rowCount,
                                   dataDict):
        tickers =  dict(self.marketDB.loadIdentifierInformation(
            'asset_dim_ticker', date, ids, cache=None))
        names =  dict(self.marketDB.loadIdentifierInformation(
            'asset_dim_name', date, ids, cache=None))
        sedols =  dict(self.marketDB.loadIdentifierInformation(
            'asset_dim_sedol', date, ids, cache=None))
        cusips =  dict(self.marketDB.loadIdentifierInformation(
            'asset_dim_cusip', date, ids, cache=None))

        for (i,j) in enumerate(ids):
            self.buffer += '%2d. %-10.10s' % \
                        (rowCount + i +1, '')
            self.buffer += ' %-10.10s' % j.getIDString()
            name = names.get(j)
            id = sedols.get(j)
            if id is None:
                id = cusips.get(j)
            if name is None or id is None:
                # Lookup error most likely due to MarketDB ID tables changes
                self.buffer += ' %-31.31s' % ' -- UNKNOWN ERROR --'
            else:
                self.buffer += ' %-20.20s %-10.10s' % (name, id)
            ticker = tickers.get(j)
            if ticker is not None:
                self.buffer += ' %-8.8s' % ticker
            else:
                self.buffer += ' %-8.8s' % '--'
            #RMG/HQ
            self.buffer += ' -- '
            self.buffer += '%4s' % ' '
            #CAP
            self.buffer += '%12.1s' % '-'
            #INDEX
            if j in dataDict:
                value = dataDict[j]
                if value is not ma.masked:
                    try:
                        self.buffer += ' %8.2f' % ','.join(i for i in value)
                    except:
                        if isinstance(value, str):
                            self.buffer += ' %-20.20s' % ','.join(i for i in value)
                        else:
                            self.buffer += ' %8s' % ','.join(i for i in value)
                else:
                    self.buffer += ' %8s' % '--'
            else:
                self.buffer += ' %8s' % '--'
            self.newline()
        self.newline()

    def displayAssetListDetails(self, assets, basicData, showTop=None, 
                                extraFields=None, *otherDataDicts):
        # Set up basic header
        header = '    MODELID    MARKETID   NAME                 SEDOL/CUS. TICKER   RMG    '
        if len(basicData.modelSelector.rmg) > 1:
            header += 'HQ  '
        header += 'CAP (%s m)   ' % self.modelSelector.numeraire.currency_code
        
        if self.modelAssetType == 'Future':
            header = '    MODELID    MARKETID   NAME                                                                                RIC          TICKER   RMG    '
        if extraFields:
            header += extraFields
        header += '\r\n'
        header += self.divider(len(header))

        rank = self.getTopRecordsIndices(basicData.assetCapDict, assets, showTop)
        self.buffer += header
        
        # Loop through the asset list
        for (i,j) in enumerate(rank):
            sid = assets[j]
            self.buffer += '%2d. %-10.10s' % \
                        (i+1, sid.getModelID().getIDString())
            axid = basicData.getSubIssueMarketID(sid)
            if axid is None:
                self.buffer += ' %-10.10s' % ' -- N/A --'
            else:
                self.buffer += ' %-10.10s' % axid
            name = basicData.getSubIssueName(sid)
            id = basicData.getSubIssueSEDOLCusip(sid)
            if self.modelAssetType == 'Future':
                id = basicData.getSubIssueRICQuote(sid)
            if name is None or id is None :
                # Lookup error most likely due to MarketDB ID tables changes
                self.buffer += ' %-31.31s' % ' -- UNKNOWN ERROR --'
            else:
                self.buffer += ' %-20.20s %-10.10s' % (name, id)
            ticker = basicData.getSubIssueTicker(sid)
            if ticker is not None:
                self.buffer += ' %-8.8s' % ticker
            else:
                self.buffer += ' %-8.8s' % '--'
            countryISOs = basicData.getSubIssueCountries(sid)
            quoteCountry = ''
            if countryISOs is not None:
                quoteCountry = countryISOs[0]
                self.buffer += ' %2s' % quoteCountry
                if countryISOs[1] is not None and countryISOs[1] != quoteCountry:
                    self.buffer += '/%2s ' % countryISOs[1]
                else:
                    self.buffer += '%4s' % ' '
            else:
                self.buffer += ' --    '
            if len(basicData.modelSelector.rmg) > 1 and modelAssetType =='Equity':
                isin = basicData.getSubIssueISIN(sid)
                if isin is not None:
                    if isin[:2] != quoteCountry and \
                            isin[:2] not in BDI_COUNTRY_CODES:
                        self.buffer += ' %-2.2s ' % isin[:2]
                    else:
                        self.buffer += '%4s' % ' '
                else:
                    self.buffer += ' -- '
            cap = basicData.getSubIssueMarketCap(sid)
            if cap != 0.0:
                self.buffer += '%12.1f' % (cap / 1e6)
            else:
                self.buffer += '%12.1s' % '-'

            for dataDict in otherDataDicts:
                if dataDict is None:
                    continue
                elif sid in dataDict:
                    value = dataDict[sid]
                    if value is not ma.masked:
                        try:
                            self.buffer += ' %8.2f' % value
                        except:
                            if isinstance(value, str):
                                self.buffer += ' %-20.20s' % value
                            else:
                                self.buffer += ' %8s' % value
                    else:
                        self.buffer += ' %8s' % '--'
                else:
                    self.buffer += ' %8s' % '--'
            self.newline()
        # Only show the top X records
        if not showTop and len(assets) > self.maxRecords:
            self.buffer += '...too many records (%d) to display, only the top %d shown\r\n' % \
                    (len(assets), self.maxRecords)
        self.newline()
    
    def displayLeaversAndJoiners(self, current, previous, setName, currData, prevData, 
                    fromDateDict=None, thruDateDict=None, 
                    industryDict=None, estuQualifyDict=None):
        self.buffer += 'Assets covered: %d' % len(current)
        pd = prevData.rmi.date
        self.buffer += ', previous: %d (%04d-%02d-%02d)\r\n' % \
                     (len(previous), pd.year, pd.month, pd.day)
        joiners = list(set(current).difference(previous))
        self.newline()
        self.buffer += '%d Assets joining %s:\r\n' % (len(joiners), setName)
        if fromDateDict is not None:
            if industryDict is None:
                fields = ' FROM_DT'
            else:
                fields = ' FROM_DT INDUSTRY'
            self.displayAssetListDetails(joiners, currData, None, fields, 
                                         fromDateDict, industryDict)
        else:
            self.displayAssetListDetails(joiners, currData)
        leavers = list(set(previous).difference(current))
        self.buffer += '%d Assets leaving %s:\r\n' % (len(leavers), setName)
        if thruDateDict is not None and estuQualifyDict is not None:
            self.displayAssetListDetails(leavers, prevData, None, ' THRU_DT  QUALIFY', 
                                         thruDateDict, estuQualifyDict)
        elif thruDateDict is not None:
            self.displayAssetListDetails(leavers, prevData, None, ' THRU_DT', thruDateDict)
        else:
            self.displayAssetListDeails(leavers, prevData)
            
    def displayFactorListDetails(self, factors, currRet, cumRet, 
                                 fTypeMap, regStats, factorMap):
        header = '    FACTOR                           TYPE            RETURN   YTD RET    T-STAT\r\n'
        self.buffer += header
        self.buffer += self.divider(len(header))
        for (i,f) in enumerate(factors):
            fIdx = factorMap[f]
            self.buffer += '%2d. %-32.32s %-12.12s %8.2f%% %8.2f%%' % \
                         (i+1, formatFactorName(f, fTypeMap), 
                          fTypeMap[f.name].name, currRet[fIdx]*100.0, cumRet[fIdx]*100.0)
            if not regStats[fIdx,1] is ma.masked:
                self.buffer += ' %8.2f\r\n' % regStats[fIdx,1]
            else:
                self.buffer += ' %8s\r\n' % ' --'
        self.newline()
    
    def divider(self, width, marker='-', lb=True):
        str = ''.join([marker for i in range(width)])
        if lb:
            str += '\r\n'
        return str
    
    def makeTitleBanner(self, title, width=80):
        pad0 = (width - len(title) - 4) // 2
        pad1 = width - len(title) - 4 - pad0
        self.buffer += self.divider(width, marker='*')
        self.buffer += self.divider(pad0, marker='*', lb=False)
        self.buffer += '  %s  ' % title
        self.buffer += self.divider(pad1, marker='*')
        self.buffer += self.divider(width, marker='*')
    
    def newline(self):
        self.buffer += '\r\n'
    
    def write(self, str):
        self.buffer += str

class RiskModelDataBundle:
    def __init__(self, modelSelector, date, modelDB, marketDB, modelAssetType=None):
        self.modelDB = modelDB
        self.marketDB = marketDB
        self.modelSelector = modelSelector
        self.modelSelector.setFactorsForDate(date, modelDB)
        self.rmi = modelDB.getRiskModelInstance(
                        modelSelector.rms_id, date)
        if self.rmi is None:
            logging.error('No risk model instance for %04d-%02d-%02d!' % \
                        (date.year, date.month, date.day))
            sys.exit()
        if (self.rmi.has_exposures * self.rmi.has_returns * self.rmi.has_risks)==0:
            logging.error('Risk model instance for %04d-%02d-%02d is incomplete!' % \
                        (date.year, date.month, date.day))
            sys.exit()
        
        # Get risk model universe
        # pass False to get universe without considering subissue dates
        self.univ = modelDB.getRiskModelInstanceUniverse(self.rmi, False)
        
        # Map assets to (trading country, home country) pairs
        clsFamily = marketDB.getClassificationFamily('REGIONS')
        assert(clsFamily is not None)
        clsMembers = dict((i.name, i) for i in self.marketDB.\
                            getClassificationFamilyMembers(clsFamily))
        clsMember = clsMembers.get('HomeCountry')
        assert(clsMember is not None)
        clsRevision = marketDB.\
                getClassificationMemberRevision(clsMember, date)
        clsData = modelDB.getMktAssetClassifications(clsRevision, 
                        self.univ, date, marketDB)
        self.sidRMGMap = dict()
        for r in modelSelector.rmg:
            self.sidRMGMap.update(dict((sid, (r.mnemonic, None)) \
                    for sid in modelDB.getActiveSubIssues(r, date)))
        missingSubIssues = list(set(self.univ).difference(list(self.sidRMGMap.keys())))
        missingData = self.getSubIssueFromThruDates(missingSubIssues)
        for r in missingData:
            self.sidRMGMap[r[0]] = (r[3], None)
        for (sid,cls) in clsData.items():
            if sid in self.sidRMGMap:
                self.sidRMGMap[sid] = (self.sidRMGMap[sid][0], 
                                       cls.classification.code)
        
        # Check for possible secondary home country
        modelRMGCodes = set(r.mnemonic for r in modelSelector.rmg)
        missingSubIssues = [sid for (sid,cty) in self.sidRMGMap.items()
                            if cty[1] is None
                            or cty[1] not in modelRMGCodes]
        if len(missingSubIssues) > 0:
            clsMember2 = clsMembers.get('HomeCountry2')
            assert(clsMember2 is not None)
            clsRevision2 = marketDB.\
                    getClassificationMemberRevision(clsMember2, date)
            clsData2 = modelDB.getMktAssetClassifications(clsRevision2, 
                            missingSubIssues, date, marketDB)
            for (sid,cls) in clsData2.items():
                if sid in self.sidRMGMap:
                    self.sidRMGMap[sid] = (self.sidRMGMap[sid][0], 
                                           cls.classification.code)
        
        # ...and market caps
        mcapDates = modelDB.getDates(modelSelector.rmg, date, 19)
        mcap = modelDB.getAverageMarketCaps(mcapDates, list(self.sidRMGMap.keys()),
                modelSelector.numeraire.currency_id, marketDB)
        self.assetCapDict = dict(zip(list(self.sidRMGMap.keys()), mcap.filled(0.0)))
        
        # Get MarketDB IDs and other identifiers
        self.modelMarketMap = dict(modelDB.getIssueMapPairs(date))
        ids = list(self.modelMarketMap.keys())

        self.nameMap = modelDB.getIssueNames(date, ids, marketDB)
        self.nameMap.update(modelDB.getFutureNames(date, ids, marketDB))

        self.tickerMap = modelDB.getIssueTickers(date, ids, marketDB)
        self.tickerMap.update(modelDB.getFutureTickers(date, ids, marketDB))
        
        self.cusipMap = modelDB.getIssueCUSIPs(date, ids, marketDB)

        self.sedolMap = modelDB.getIssueSEDOLs(date, ids, marketDB)

        self.isinMap = modelDB.getIssueISINs(date, ids, marketDB)

        if not modelAssetType =='Equity':    
            self.ricMap = modelDB.getIssueRICQuote(date, ids, marketDB)
        
        self.sedCusMap = dict()
        self.sedCusMap.update(self.cusipMap)
        self.sedCusMap.update(self.sedolMap)
        
        # Get ESTU
        self.estu = modelSelector.loadEstimationUniverse(self.rmi, modelDB)
        
        # Get exposure matrix
        if self.rmi.has_exposures == 1:
            self.expM = modelSelector.loadExposureMatrix(self.rmi, modelDB)
        
        # Get factor covariance matrix
        (cov, factors) = modelSelector.loadFactorCovarianceMatrix(self.rmi, modelDB)
        self.covMatrix = cov
        
        # Specific risks
        (self.srDict, self.scDict) = modelSelector.loadSpecificRisks(self.rmi, modelDB)
    
    def getAllSubIssues(self):
        return self.univ
    
    def getESTU(self):
        return self.estu
    
    def getExposureMatrix(self):
        return self.expM
    
    def getFactorCov(self):
        return self.covMatrix
    
    def getSpecificRisks(self):
        return (self.srDict, self.scDict)
    
    def getSubIssueMarketID(self, sid):
        axid = self.modelMarketMap.get(sid.getModelID())
        if axid is None:
            try:
                [(id, fromDt, thruDt, rmg)] = self.getSubIssueFromThruDates([sid])
                lastDt = thruDt - datetime.timedelta(1)
                axidMap = dict(self.modelDB.getIssueMapPairs(lastDt))
                return axidMap.get(sid.getModelID()).getIDString()
            except:
                return None
        else:
            return axid.getIDString()
    
    def getSubIssueName(self, sid):
        name = self.nameMap.get(sid.getModelID())
        if name is None:
            try:
                [(id, fromDt, thruDt, rmg)] = self.getSubIssueFromThruDates([sid])
                lastDt = thruDt - datetime.timedelta(1)
                nameMap = self.modelDB.getIssueNames(
                                lastDt, [sid.getModelID()], self.marketDB)
                return nameMap.get(sid.getModelID())
            except:
                return None
        else:
            return name
    
    def getSubIssueSEDOLCusip(self, sid):
        id = self.sedCusMap.get(sid.getModelID())
        if id is None:
            try:
                [(id, fromDt, thruDt, rmg)] = self.getSubIssueFromThruDates([sid])
                lastDt = thruDt - datetime.timedelta(1)
                idMap = self.modelDB.getIssueCUSIPs(
                                lastDt, [sid.getModelID()], self.marketDB)
                if sid.getModelID() in idMap:
                    return idMap.get(sid.getModelID())
                idMap.update(self.modelDB.getIssueSEDOLs(
                                lastDt, [sid.getModelID()], self.marketDB))
                return idMap.get(sid.getModelID())
            except:
                return None
        else:
            return id 
    
    def getSubIssueMarketCap(self, sid):
        return self.assetCapDict.get(sid, 0.0)
    
    def getSubIssueCountries(self, sid):
        countryISOs = self.sidRMGMap.get(sid)
        if countryISOs is None:
            try:
                [(id, fromDt, thruDt, tradingCty)] = self.getSubIssueFromThruDates([sid])
                countryISOs = (tradingCty, None)
            except:
                return None
        return countryISOs
    
    def getSubIssueISIN(self, sid):
        id = self.isinMap.get(sid.getModelID())
        if id is None:
            try:
                [(id, fromDt, thruDt, rmg)] = self.getSubIssueFromThruDates([sid])
                lastDt = thruDt - datetime.timedelta(1)
                isinMap = self.modelDB.getIssueISIN(
                                lastDt, [sid.getModelID()], self.marketDB)
                id = isinMap.get(sid.getModelID())
            except:
                id = None
        if id is None:
            return None
        elif id.strip() == '':
            return None
        return id

    def getSubIssueRICQuote(self, sid):
        id = self.ricMap.get(sid.getModelID())
        if id is None:
            try:
                [(id, fromDt, thruDt, rmg)] = self.getSubIssueFromThruDates([sid])
                lastDt = thruDt - datetime.timedelta(1)
                ricMap = self.modelDB.getIssueRICQuote(
                                lastDt, [sid.getModelID()], self.marketDB)
                id = ricMap.get(sid.getModelID())
            except:
                id = None
        if id is None:
            return None
        elif id.strip() == '':
            return None
        return id
    
    def getSubIssueTicker(self, sid):
        id = self.tickerMap.get(sid.getModelID())
        if id is None:
            try:
                [(id, fromDt, thruDt, rmg)] = self.getSubIssueFromThruDates([sid])
                lastDt = thruDt - datetime.timedelta(1)
                tickerMap = self.modelDB.getIssueTickers(
                                lastDt, [sid.getModelID()], self.marketDB)
                id = tickerMap.get(sid.getModelID())
            except:
                id = None
        if id is None:
            return None
        elif id.strip() == '':
            return None
        return id
    
    def getSubIssueFromThruDates(self, sidList):
        if len(sidList) == 0:
            return list()
        INCR = 100
        sidArgList = [('sid%d' % i) for i in range(INCR)]
        query = """SELECT s.sub_id, s.from_dt, s.thru_dt, g.mnemonic
                   FROM sub_issue s, risk_model_group g
                   WHERE s.sub_id IN (%(sids)s) AND s.rmg_id = g.rmg_id""" % {
                'sids': ','.join([(':%s' % i) for i in sidArgList])}
        sidStrs = [sid.getSubIDString() for sid in sidList]
        defaultDict = dict((i, None) for i in sidArgList)
        sidInfo = list()
        for sidChunk in listChunkIterator(sidStrs, INCR):
            myDict = dict(defaultDict)
            myDict.update(dict(zip(sidArgList, sidChunk)))
            self.modelDB.dbCursor.execute(query, myDict)
            result = self.modelDB.dbCursor.fetchall()
            sidInfo.extend([(ModelDB.SubIssue(r[0]), 
                             r[1].date(), r[2].date(), r[3]) for r in result])
        return sidInfo 

class RiskModelValidator:
    def __init__(self, modelSelector, priorModelSelector, date, 
                 modelDB, marketDB, modelAssetType):
        self.modelSelector = modelSelector
        self.priorModelSelector = priorModelSelector
        self.modelAssetType = modelAssetType
        self.report = ModelReport(modelSelector, marketDB, modelAssetType)
        
        # Get today's risk model instance data
        self.currModelData = RiskModelDataBundle(self.modelSelector, 
                                        date, modelDB, marketDB, modelAssetType)
        
        # Try and get previous risk model instance data
        dateList = modelDB.getDates(self.priorModelSelector.rmg, date, 1, True)
        if len(dateList) > 1:
            prevDate = dateList[0]
            self.prevModelData = RiskModelDataBundle(self.priorModelSelector, 
                                            prevDate, modelDB, marketDB, modelAssetType)
        
        # Get mapping of RMG IDs to ISO code, description, etc
        query = """SELECT g.rmg_id FROM risk_model_group g
                   WHERE g.rmg_id IN (
                        SELECT DISTINCT m.rmg_id 
                        FROM rmg_model_map m WHERE m.rms_id > 0)"""
        modelDB.dbCursor.execute(query)
        allRiskModelGroups = [modelDB.getRiskModelGroup(r[0]) \
                            for r in modelDB.dbCursor.fetchall()]
        self.rmgInfoMap = dict((rmg.mnemonic, rmg) \
                            for rmg in allRiskModelGroups)
        for rmg in self.rmgInfoMap.values():
            rmg.setRMGInfoForDate(date)
        
        self.modelSelector.setFactorsForDate(date, modelDB)
    
    def displayModelStructure(self, modelDB):
        d = self.currModelData.rmi.date
        pd = self.prevModelData.rmi.date
        rm = self.modelSelector
        
        self.report.write('Results for %s (%s) on %04d-%02d-%02d\r\n' % \
                (rm.description, rm.mnemonic, d.year, d.month, d.day))
        self.report.write('Risk model %d, serie %d, numeraire currency %s (%d)\r\n' % \
                (rm.rm_id, rm.rms_id, rm.numeraire.currency_code, 
                 rm.numeraire.currency_id))
        self.report.newline()
        
        # If we don't even have exposures, skip and move on
        if self.currModelData.rmi.has_exposures != 1:
            self.report.write('Risk model instance on %04d-%02d-%02d has no exposures, skipping\r\n' % (d.year, d.month, d.day))
            return False
      
        # Report on factor structure
        rm.setFactorsForDate(d, modelDB)
        expM = self.currModelData.getExposureMatrix()
        self.report.write('Factors: %d\r\n' % len(rm.factors))
        factorTypeMap = dict()
        for fType in expM.factorIdxMap_.keys():
            count = len(list(expM.factorIdxMap_[fType].keys()))
            if count > 0:
                self.report.write('%s: %d  ' % (fType.name, count))
            for fName in expM.factorIdxMap_[fType].keys():
                factorTypeMap[fName] = fType
        self.report.newline()
        
        # Check for factor additions/removals
        currFactors = [f.description for f in rm.factors]
        rm.setFactorsForDate(pd, modelDB)
        prevFactors = [f.description for f in rm.factors]
        rm.setFactorsForDate(d, modelDB)
        add = set(currFactors).difference(prevFactors)
        rem = set(prevFactors).difference(currFactors)
        if len(add) > 0 or len(rem) > 0:
            self.report.write('Factor structure has changed since %04d-%02d-%02d!\n' \
                    % (pd.year, pd.month, pd.day))
        if len(add) > 0:
            self.report.write('NEW factors: %s\n' % (', '.join(add)))
        if len(rem) > 0:
            self.report.write('REMOVED factors: %s\n' % (', '.join(rem)))
        self.report.newline()
    
    def displayModelCoverage(self, modelDB, marketDB):
        univ = self.currModelData.getAllSubIssues()
        puniv = self.prevModelData.getAllSubIssues()
        
        if set(univ) == set(puniv):
            fromDateDict = dict()
            thruDateDict = dict()
            sidClsMap = dict()
        else:
            new = list(set(univ).difference(puniv))
            gone = list(set(puniv).difference(univ))
            sidArgList = [('sid%d' % i) for i in range(len(new+gone))]
            query = """SELECT sub_id, from_dt, thru_dt
                       FROM sub_issue
                       WHERE sub_id IN (%s)""" % ','.join([(':%s' % i) for i in sidArgList])
            valueDict = dict(zip(sidArgList, [sid.getSubIDString() for sid in new+gone]))
            modelDB.dbCursor.execute(query, valueDict)
            results = modelDB.dbCursor.fetchall()
            # Get from_dt to differeniate IPOs from 'resurrected' assets
            fromDateDict = dict((ModelDB.SubIssue(r[0]), r[1].date()) for r in results)
            
            # Fetch thru_dt records to check for proper asset termination
            thruDateDict = dict((ModelDB.SubIssue(r[0]), r[2].date()) for r in results)
        if self.modelAssetType =='Equity':    
            # Get industry classification
            sidClsMap = dict((sid, cls.classification.description) \
                    for (sid, cls) in self.modelSelector.industryClassification.\
                        getAssetConstituents(modelDB, list(set(univ).difference(puniv)),
                        self.currModelData.rmi.date).items())

            self.report.displayLeaversAndJoiners(
                            univ, puniv, 'model universe',
                            self.currModelData, self.prevModelData, 
                            fromDateDict, thruDateDict, industryDict=sidClsMap)
        else:
            self.report.displayLeaversAndJoiners(
                            univ, puniv, 'model universe',
                            self.currModelData, self.prevModelData, 
                            fromDateDict, thruDateDict)
      
            
        totalCap = numpy.sum(list(self.currModelData.assetCapDict.values())) / 1e9
        self.report.write('Total model universe market cap: %.2f bn %s (may include double-counting)\r\n' % \
                    (totalCap, self.modelSelector.numeraire.currency_code))
        self.report.newline()
        
        # Find any benchmark assets not in model
        irQuery = """(SELECT * from marketdb_global.index_revision_active WHERE 
                      dt = (SELECT max(r.dt) FROM marketdb_global.index_revision_active r
                      WHERE r.dt <= to_date('%s', 'YYYY-MM-DD'))) ir,"""%self.currModelData.rmi.date.isoformat()
        query = """
            SELECT im.sub_id, ifm.name
            FROM """+\
            irQuery+\
            """ marketdb_global.index_member imm, 
                marketdb_global.index_family ifm,
                marketdb_global.index_constituent ic
                left join (SELECT marketdb_id, sub_id, rmg_id FROM issue_map im, sub_issue s WHERE
                          im.modeldb_id = s.issue_id
                          AND s.from_dt <= to_date('%(dt_arg)s','YYYY-MM-DD') 
                          AND s.thru_dt > to_date('%(dt_arg)s','YYYY-MM-DD')
                          AND im.from_dt <= to_date('%(dt_arg)s','YYYY-MM-DD') 
                          AND im.thru_dt > to_date('%(dt_arg)s','YYYY-MM-DD')) im on im.marketdb_id = ic.axioma_id
            WHERE ic.revision_id = ir.id
            AND imm.id = ir.index_id
            AND ifm.id = imm.family_id
            AND NOT EXISTS (SELECT * FROM rmi_universe u
                WHERE u.sub_issue_id = im.sub_id
                AND u.rms_id = %(rms_arg)s AND u.dt = to_date('%(dt_arg)s','YYYY-MM-DD'))
            AND EXISTS (SELECT * FROM rmg_model_map rmm 
                WHERE rmm.rms_id = %(rms_arg)s AND rmm.rmg_id = im.rmg_id
                AND rmm.from_dt <= to_date('%(dt_arg)s ', 'YYYY-MM-DD')
                AND rmm.thru_dt > to_date('%(dt_arg)s','YYYY-MM-DD'))"""%\
                {'rms_arg':self.modelSelector.rms_id,
                 'dt_arg':self.currModelData.rmi.date}
        
        modelDB.dbCursor.execute(query)
        assetIndexFamilyMap = dict()
        mktIdIndexFamilyMap = dict()
        for row in modelDB.dbCursor.fetchall():
            if row[0] is not None:
                sid = ModelDB.SubIssue(string=row[0])
                if sid not in assetIndexFamilyMap:
                    assetIndexFamilyMap[sid] = set()
                assetIndexFamilyMap[sid].add(row[1])
        for (sid, families) in assetIndexFamilyMap.items():
            assetIndexFamilyMap[sid] = ','.join(families)
        if self.modelAssetType =='Equity':    
            self.report.write('Benchmark assets missing from risk model universe:\r\n')
            self.report.displayAssetListDetails(list(assetIndexFamilyMap.keys()), 
                        self.currModelData, len(assetIndexFamilyMap), 
                        ' INDEX', assetIndexFamilyMap)

            #Flag those without a issue_id
            query = """
                SELECT ic.axioma_id, ifm.name
                FROM """+\
                irQuery+\
                """ marketdb_global.index_member imm, 
                    marketdb_global.index_family ifm,
                    marketdb_global.index_constituent ic
                WHERE ic.revision_id = ir.id
                AND imm.id = ir.index_id
                AND ifm.id = imm.family_id
                AND imm.name not like '%%NEXT DAY OPEN'
                AND NOT EXISTS (SELECT * FROM issue_map im, sub_issue s WHERE
                              im.modeldb_id = s.issue_id
                              AND s.from_dt <= to_date('%(dt_arg)s','YYYY-MM-DD') 
                              AND s.thru_dt > to_date('%(dt_arg)s','YYYY-MM-DD')
                              AND im.from_dt <= to_date('%(dt_arg)s','YYYY-MM-DD') 
                              AND im.thru_dt > to_date('%(dt_arg)s','YYYY-MM-DD') AND im.marketdb_id = ic.axioma_id)"""%\
                    {'dt_arg':self.currModelData.rmi.date}

            modelDB.dbCursor.execute(query)
            for row in modelDB.dbCursor.fetchall():
                mktId = MarketID.MarketID(string=row[0])
                if mktId not in mktIdIndexFamilyMap:
                    mktIdIndexFamilyMap[mktId] = set()
                mktIdIndexFamilyMap[mktId].add(row[1])
            if len(list(mktIdIndexFamilyMap.keys()))> 0:
                self.report.displayMktAssetListDetails(list(mktIdIndexFamilyMap.keys()), 
                                                       self.currModelData.rmi.date,
                                                       len(assetIndexFamilyMap),
                                                       mktIdIndexFamilyMap)

        # Report exchange/section/board changes
        clsFamily = marketDB.getClassificationFamily('REGIONS')
        assert(clsFamily is not None)
        clsMembers = dict((i.name, i) for i in marketDB.\
                            getClassificationFamilyMembers(clsFamily))
        clsMember = clsMembers.get('Market')
        assert(clsMember is not None)
        clsRevision = marketDB.getClassificationMemberRevision(
                    clsMember, self.currModelData.rmi.date)
        clsData = dict((sid, cls.classification.name) \
                    for (sid,cls) in \
                    modelDB.getMktAssetClassifications(clsRevision, 
                    univ, self.currModelData.rmi.date, marketDB).items())
        prevClsRevision = marketDB.getClassificationMemberRevision(
                    clsMember, self.prevModelData.rmi.date)
        pClsData = dict((sid, cls.classification.name) \
                    for (sid,cls) in \
                    modelDB.getMktAssetClassifications(prevClsRevision, 
                    puniv, self.prevModelData.rmi.date, marketDB).items())
        
        changedAssets = list()
        for (sid,cls) in clsData.items():
            prevCls = pClsData.get(sid)
            if prevCls is not None:
                if cls != prevCls:
                    changedAssets.append(sid)
        self.report.write('Assets with reported section/exchange changes:\r\n')
        self.report.displayAssetListDetails(changedAssets,
                self.currModelData, None, 'CURR               PREV', clsData, pClsData)
    
    def displayModelESTUDetails(self, modelDB):
        # Load current and previous estimation universes
        estu = self.currModelData.getESTU()
        p_estu = self.prevModelData.getESTU()
        
        assets = set(estu).symmetric_difference(p_estu)
        
        fromDateDict = dict()
        thruDateDict = dict()
        qualifyDict = dict()
        INCR = 500
        sidArgList = [('sid%d' % i) for i in range(INCR)]
        defaultDict = dict([(arg, None) for arg in sidArgList])
        query = """SELECT sub_id, from_dt, thru_dt
        FROM sub_issue
        WHERE sub_id IN (%s)""" % ','.join([(':%s' % i) for i in sidArgList])
        for assetChunk in listChunkIterator(list(assets), INCR):
            valueDict = defaultDict
            valueDict.update(dict(zip(sidArgList, [sid.getSubIDString() for sid in assetChunk])))
            modelDB.dbCursor.execute(query, valueDict)
            results = modelDB.dbCursor.fetchall()
            # Fetch from_dt and thru_dt records
            thruDateDict.update(dict((ModelDB.SubIssue(r[0]), r[2].date()) for r in results))
            fromDateDict.update(dict((ModelDB.SubIssue(r[0]), r[1].date()) for r in results))
            
            qualifyInfo = modelDB.loadESTUQualifyHistory(
                    self.modelSelector.rms_id, assetChunk, [self.prevModelData.rmi.date])
            qualifyDict.update(dict((sid, True) for (i,sid) in \
                        enumerate(assetChunk) if qualifyInfo.data.filled(0.0)[i,0] != 0.0))
        
        # Show leavers and joiners
        self.report.displayLeaversAndJoiners(estu, p_estu, 
                'estimation universe',self.currModelData, self.prevModelData, 
                fromDateDict, thruDateDict, estuQualifyDict=qualifyDict)
        estuCaps = numpy.array([self.currModelData.getSubIssueMarketCap(s) \
                            for s in estu])
        estuWeights = [1/float(len(estu)) for s in estu]
        # Show asset composition of estu
        if self.modelAssetType == 'Equity':
            estuWeights = estuCaps / numpy.sum(estuCaps)
            self.report.write('Top 20 assets in estimation universe (approx regression % weights):\r\n')
            estuWeightDict = dict(zip(estu, estuWeights*100.0))
            currTop = numpy.array([self.currModelData.getSubIssueMarketCap(s) \
                                   for s in estu])
            currRank = numpy.argsort(-currTop)[:20]
            prevTop = numpy.array([self.prevModelData.getSubIssueMarketCap(s) \
                                   for s in p_estu])
            prevRank = numpy.argsort(-prevTop)[:20]
            diff = set(estu[i] for i in currRank).\
                        difference([p_estu[i] for i in prevRank])
            nonESTUDict = dict((s, True) for s in diff)
            
            # Compute MCTRs
            MCTR = Utilities.compute_MCTR(self.currModelData.getExposureMatrix(),
                                          self.currModelData.getFactorCov(),
                                          self.currModelData.getSpecificRisks()[0],
                                          list(zip(estu, estuWeights)))
            MCTRDict = dict(zip(self.currModelData.getExposureMatrix().getAssets(), MCTR))
            p_estuWeights = numpy.array([self.prevModelData.\
                                getSubIssueMarketCap(s) for s in p_estu])
            p_estuWeights /= numpy.sum(p_estuWeights)
            prev_MCTR = Utilities.compute_MCTR(self.prevModelData.getExposureMatrix(),
                                               self.prevModelData.getFactorCov(),
                                               self.prevModelData.getSpecificRisks()[0],
                                               list(zip(p_estu, p_estuWeights)))
            prevMCTRDict = dict(zip(self.prevModelData.getExposureMatrix().getAssets(), prev_MCTR))

            self.report.displayAssetListDetails(estu, self.currModelData, 20, 
                            '  WGT %    NEW?     MCTR   P.MCTR', 
                            estuWeightDict, nonESTUDict, MCTRDict, prevMCTRDict)
            
        # If multi-country model, report on composition by country
        if len(self.modelSelector.rmg) > 1:
            rmgWeightMap = dict()
            for r in self.modelSelector.rmg:
                rmgWeightMap[r.mnemonic] = 0.0
            for i in range(len(estu)):
                countryISOs = self.currModelData.getSubIssueCountries(estu[i])
                if countryISOs is None:
                    continue
                try:
                    rmgWeightMap[countryISOs[1]] += estuWeights[i]
                except:
                    rmgWeightMap[countryISOs[0]] += estuWeights[i]
            rank = numpy.argsort(list(rmgWeightMap.values())).tolist()
            rank.reverse()
            header = 'Top countries in estimation universe:\r\n'
            self.report.write(header)
            self.report.write(self.report.divider(len(header)))
            for (i,j) in enumerate(rank[:10]):
                rmgISO = list(rmgWeightMap.keys())[j]
                self.report.write('%2d. %-20.20s %5.2f%%\r\n' % \
                        (i+1, self.rmgInfoMap[rmgISO].description, 
                         rmgWeightMap[rmgISO]*100.0))
            self.report.newline()
        
        self.report.write('Total ESTU market cap: %.2f bn %s\r\n' % \
                    (numpy.sum(estuCaps) / 1e9, self.modelSelector.numeraire.currency_code))
        
        tr = modelDB.loadTotalReturnsHistory(self.modelSelector.rmg, 
                                    self.currModelData.rmi.date, estu, 0)
        tr = tr.data[:,0].filled(0.0)
        wgt = estuCaps / numpy.sum(estuCaps)
        estuReturn = ma.inner(tr, wgt) * 100.0
        self.report.write('ESTU return: %5.2f%% (cap-weighted)\r\n' % (estuReturn))
        self.report.newline()
    
    def displayRegressionStatistics(self, modelDB):
        r2 = modelDB.getRMSStatistics(self.modelSelector.rms_id,
                                      self.currModelData.rmi.date)
        if r2 is None:
            r2 = 1.0
        self.report.write('Adjusted R-squared: %.2f\r\n' % r2)
        dateList = modelDB.getDates(self.modelSelector.rmg, 
                                    self.currModelData.rmi.date, 252, True)
        dateList = [d for d in dateList if d.year == \
                    self.currModelData.rmi.date.year]
        r2List = list()
        for d in dateList:
            r2 = modelDB.getRMSStatistics(self.modelSelector.rms_id, d)
            if r2 is not None:
                r2List.append(r2)
        if len(r2List) > 0:
            avg_r2 = numpy.average(r2List)
            self.report.write('Year-to-date average adjusted R-squared: %.2f\r\n' % avg_r2)
        self.report.newline()
    
    def displayFactorReturnsChecks(self, modelDB):
        if not isinstance(self.modelSelector, MFM.StatisticalFactorModel):
            # Preliminaries
            self.modelSelector.setFactorsForDate(self.currModelData.rmi.date, modelDB)
            expM = self.currModelData.getExposureMatrix()
            factorTypeMap = dict()
            for fType in expM.factorIdxMap_.keys():
                for fName in expM.factorIdxMap_[fType].keys():
                    factorTypeMap[fName] = fType
            
            # Fetch current and previous factor returns
            startDate = datetime.date(self.currModelData.rmi.date.year,1,1)
            dateListYTD = modelDB.getDateRange(self.modelSelector.rmg, 
                            startDate, self.currModelData.rmi.date)
            subFactors = modelDB.getSubFactorsForDate(
                            self.currModelData.rmi.date, self.modelSelector.factors)
            factorReturns = modelDB.loadFactorReturnsHistory(
                            self.modelSelector.rms_id, subFactors, dateListYTD)
            factorReturns.data = factorReturns.data.filled(0.0)
            fr = factorReturns.data[:,-1]
            cfr = numpy.cumproduct(1.0 + factorReturns.data, axis=1)[:,-1] - 1.0
            
            # Get factor regression statistics
            (regStats, factors, r2) = self.modelSelector.\
                    loadRegressionStatistics(self.currModelData.rmi.date, modelDB)
            factorIdxMap = dict((s.factor,i) for (i,s) in enumerate(subFactors))
            
            # Show large positive factor returns
            rank = numpy.argsort(-fr)
            rank = [i for i in rank if factorTypeMap[factors[i].name] \
                        != ExposureMatrix.StatisticalFactor]
            self.report.write('Highest 5 factor returns:\r\n')
            factorList = [factors[i] for i in rank[:5]]
            self.report.displayFactorListDetails(
                    factorList, fr, cfr, factorTypeMap, regStats, factorIdxMap)
            
            # ...and large negative factor returns
            self.report.write('Lowest 5 factor returns:\r\n')
            factorList = [factors[i] for i in rank[-5:]]
            factorList.reverse()
            self.report.displayFactorListDetails(
                    factorList, fr, cfr, factorTypeMap, regStats, factorIdxMap)
            
            # Show factors with largest (most significant) and smallest t-stats
            tStats = numpy.array([abs(t) for t in \
                                  regStats[:,1].filled(0.0)])
            rank = numpy.argsort(-tStats)
            rank = [i for i in rank if factorTypeMap[factors[i].name] \
                        != ExposureMatrix.StatisticalFactor]
            factorList = [factors[i] for i in rank if \
                          regStats[i,1] is not ma.masked][:5]
            self.report.write('Strongest 5 factors:\r\n')
            self.report.displayFactorListDetails(
                    factorList, fr, cfr, factorTypeMap, regStats, factorIdxMap)
            
            factorList = [factors[i] for i in rank if \
                          regStats[i,1] is not ma.masked][-5:]
            factorList.reverse()
            self.report.write('Weakest 5 factors:\r\n')
            self.report.displayFactorListDetails(
                    factorList, fr, cfr, factorTypeMap, regStats, factorIdxMap)
            
        # Report any factors dropped from regression (empty, non-trading, etc.)
        if not isinstance(self.modelSelector, MFM.StatisticalFactorModel) and \
                not isinstance(self.modelSelector, MFM.RegionalStatisticalFactorModel) \
                and (not self.modelSelector.isLinkedModel()):
            if isinstance(riskModel_, PhoenixModels.CommodityFutureRegionalModel):
                #If it is a commodities model, look up missing factor return
                droppedFactorsIdx = numpy.flatnonzero(ma.getmaskarray(fr))
            else:
                #If not commodities model, look up missing regression stat
                droppedFactorsIdx = numpy.flatnonzero(
                    ma.getmaskarray(regStats[:,1]))
            currenciesIdx = [i for (i,f) in enumerate(factors) \
                    if expM.checkFactorType(f.name, ExposureMatrix.CurrencyFactor)]
            droppedFactorsIdx = [i for i in droppedFactorsIdx \
                                 if i not in currenciesIdx]
            self.report.write('%d Factors (non-currency) omitted from regression:\r\n' % len(droppedFactorsIdx))
            header = '    FACTOR                    TYPE      REASON\r\n'
            self.report.write(header)
            self.report.write(self.report.divider(len(header)))
            descRMGMap = dict((r.description, r) \
                                for r in self.modelSelector.rmg)
            for (i,j) in enumerate(droppedFactorsIdx):
                factorName = factors[j].name
                self.report.write('%2d. %-25.25s %-10.10s' % \
                             (i+1, factorName, factorTypeMap[factorName].name))
                # If it is a country factor, check to see if it's a non-trading day
                if factorName in descRMGMap or factorName == 'Domestic China':
                    if factorName == 'Domestic China':
                        rmg = descRMGMap['China']
                    else:
                        rmg = descRMGMap[factorName]
                    lastTradeDate = modelDB.getDates([rmg], d, 0)
                    if lastTradeDate[0] != d:
                        self.report.write('Non-trading day')
                    else:
                        self.report.write('-- UNKNOWN --')
                else:
                    self.report.write('-- UNKNOWN --')
                self.report.newline()
            self.report.newline()
    
    def displayDodgySpecificReturns(self, modelDB):
        if self.modelSelector.mnemonic[:4] == 'AXCN':
            threshold = 0.04
        elif self.modelSelector.mnemonic[:4] == 'AXTW':
            threshold = 0.03
        else:
            threshold = 0.15
        univ = self.currModelData.getAllSubIssues()
        estu = self.currModelData.getESTU()
        sr = modelDB.loadSpecificReturnsHistory(self.modelSelector.rms_id, 
                    univ, [self.currModelData.rmi.date])
        sr.data = ma.masked_where(sr.data <= -1.0, sr.data)
        sr = sr.data[:,0].filled(0.0)
        dodgyIdx = numpy.flatnonzero(ma.getmaskarray(ma.masked_where(
                                     abs(numpy.log10(1.0+sr)) > threshold, sr)))
        # Flag estu assets
        estuDict = dict((s, True) for s in estu)
        
        if len(dodgyIdx) > 0:
            bounds = [10.0**(threshold*x) - 1.0 for x in (-1.0, 1.0)]
            sr_assets = [univ[i] for i in dodgyIdx]
            srDict = dict(zip(sr_assets, [sr[i]*100.0 for i in dodgyIdx]))
            # Display their corresponding daily total returns too
            top_indices = self.report.getTopRecordsIndices(
                                self.currModelData.assetCapDict, sr_assets)
            sr_assets = [sr_assets[i] for i in top_indices]
            tr = modelDB.loadTotalReturnsHistory(
                                self.modelSelector.rmg, d, sr_assets, 0)
            tr.data = tr.data.filled(0.0)
            trDict = dict(zip(sr_assets, 
                          [val * 100.0 for val in tr.data[:,0]]))
            self.report.write('Assets with specific returns beyond [%.2f%%, %.2f%%]:\r\n' \
                        % (100.0 * bounds[0], 100.0 * bounds[1]))
            self.report.displayAssetListDetails(sr_assets, self.currModelData, None, 
                        ' SR %     TR %      ESTU', srDict, trDict, estuDict)
    
    def displayLargeExposureChanges(self, modelDB, threshold=2.0):
        if not isinstance(self.modelSelector, MFM.StatisticalFactorModel):
            expM = self.currModelData.getExposureMatrix()
            expMatrix = expM.getMatrix()
            # Re-shuffle around prev expM to match current asset positions
            p_expM = self.prevModelData.getExposureMatrix()
            p_expMatrix = p_expM.getMatrix()
            assetIdxMap = dict(zip(p_expM.getAssets(), 
                                range(len(p_expM.getAssets()))))
            
            estuDict = dict((s, True) for s in self.currModelData.getESTU())
            
            # Compare exposures for each asset, for all factors
            # First loop around all factors
            for i in range(expMatrix.shape[0]):
                factorName = expM.getFactorNames()[i]
                if expM.checkFactorType(factorName, 
                            ExposureMatrix.StatisticalFactor) is True:
                    continue
                # Make sure factor exists in previous model instance
                if factorName in p_expM.getFactorNames():
                    pidx = p_expM.getFactorIndex(factorName)
                    assetList = list()
                    currExpDict = dict()
                    prevExpDict = dict()
                    returnsDict = dict()
                    # Change threshold for different factor types
                    if factorName not in expM.getFactorNames(
                                    fType=ExposureMatrix.StyleFactor) or \
                        Utilities.is_binary_data(expMatrix[i,:]):
                        z = 1.0
                    else:
                        z = threshold
                    # Loop around assets 
                    for j in range(expMatrix.shape[1]):
                        sid = expM.getAssets()[j]
                        # Only process assets alive in both periods
                        if sid not in assetIdxMap:
                            continue
                        else:
                            pos = assetIdxMap[sid]
                        prevExp = p_expMatrix[pidx,pos]
                        currExp = expMatrix[i,j]
                        # Flag assets with big changes
                        if currExp is not ma.masked and prevExp is not ma.masked:
                            if abs(currExp - prevExp) >= z:
                                assetList.append(sid)
                                currExpDict[sid] = currExp
                                prevExpDict[sid] = prevExp
                        elif currExp is ma.masked and prevExp is ma.masked:
                            continue
                        elif currExp is ma.masked:
                            assetList.append(sid)
                            currExpDict[sid] = ma.masked
                            prevExpDict[sid] = prevExp
                        elif prevExp is ma.masked:
                            assetList.append(sid)
                            currExpDict[sid] = currExp
                            prevExpDict[sid] = ma.masked
                     
                    if len(assetList) > 0:
                        top_indices = self.report.getTopRecordsIndices(
                                    self.currModelData.assetCapDict, assetList)
                        assetList = [assetList[i] for i in top_indices]
                        tr = modelDB.loadTotalReturnsHistory(self.modelSelector.rmg, 
                                        self.currModelData.rmi.date, assetList, 0)
                        returnsDict = dict((sid, 100.0 * tr.data[i,0]) \
                                    for (i,sid) in enumerate(assetList))
                     
                        if factorName in ('Value', 'Growth', 'Leverage'):
                            sidArgs = ['sid%d' % i for i in range(len(assetList))]
                            query = """SELECT sub_issue_id FROM sub_issue_fund_currency_active
                                       WHERE sub_issue_id IN (%(sids)s)
                                       AND eff_dt >= :dt_arg""" % {
                                    'sids': ','.join([':%s' % arg for arg in sidArgs])}
                            dataDict = dict()
                            dataDict['dt_arg'] = self.currModelData.rmi.date - datetime.timedelta(5)
                            dataDict.update(dict(zip(sidArgs, 
                                [sid.getSubIDString() for sid in assetList])))
                            modelDB.dbCursor.execute(query, dataDict)
                            newFundDataIds = set(r[0] for r in modelDB.dbCursor.fetchall())
                            newFundDict = dict((sid, True) for sid in assetList \
                                    if sid.getSubIDString() in newFundDataIds)
                            header = '  CURR     PREV     TR %   FUND UPD?  ESTU'
                        else:
                            header = '  CURR     PREV     TR %     ESTU'
                            newFundDict = None

                        self.report.write('Assets with exposure change for %s > %.1f:\r\n' % (factorName, z))
                        self.report.displayAssetListDetails(assetList, 
                                self.currModelData, None, header,
                                currExpDict, prevExpDict, returnsDict, newFundDict, estuDict)
                else:
                    continue
    
    def displayCovarianceMatrixSpecs(self, modelDB, min_eig=1e-10):
        cov = self.currModelData.getFactorCov()
        p_cov = self.prevModelData.getFactorCov()
        (eigval, eigvec) = linalg.eigh(cov)
        (p_eigval, p_eigvec) = linalg.eigh(p_cov)
        bad = numpy.flatnonzero(ma.getmaskarray(ma.masked_where(
                                    eigval < min_eig, eigval)))
        self.report.write('%d eigenvalues in covariance matrix < %s:\r\n' % \
                (len(bad), str(min_eig)))
        for i in bad:
            self.report.write('%s\r\n' % str(eigval[i]))
        frobNorm = 1000.0 * numpy.sqrt(numpy.sum(eigval * eigval))
        p_frobNorm = 1000.0 * numpy.sqrt(numpy.sum(p_eigval * p_eigval))
        pd = self.prevModelData.rmi.date
        self.report.write('Frobenius norm times 1000.0: %f, previous: %f (%04d-%02d-%02d)\r\n' \
                        % (frobNorm, p_frobNorm, pd.year, pd.month, pd.day))
        self.report.newline()
    
    def displayFactorVolChecks(self, modelDB, threshold=0.005):
        if not isinstance(self.modelSelector, MFM.StatisticalFactorModel):
            expM = self.currModelData.getExposureMatrix()
            factorTypeMap = dict()
            for fType in expM.factorIdxMap_.keys():
                for fName in expM.factorIdxMap_[fType].keys():
                    factorTypeMap[fName] = fType
            self.modelSelector.setFactorsForDate(self.prevModelData.rmi.date, modelDB)
            pfactors = self.modelSelector.factors
            p_factorIdxMap = dict((pfactors[i].factorID, i) \
                                    for i in range(len(pfactors)))
            self.modelSelector.setFactorsForDate(self.currModelData.rmi.date, modelDB)
            factors = self.modelSelector.factors
            
            factorVols = numpy.sqrt(numpy.diag(
                                    self.currModelData.getFactorCov()))
            p_factorVols = numpy.sqrt(numpy.diag(
                                    self.prevModelData.getFactorCov()))
            rank = numpy.argsort(-factorVols)
            rank = [i for i in rank if factorTypeMap[factors[i].name] \
                        != ExposureMatrix.StatisticalFactor]
            
            self.report.write('5 MOST risky factors:\r\n')
            header = '    FACTOR                           TYPE         RISK (%)\r\n'
            self.report.write(header)
            self.report.write(self.report.divider(len(header)))
            for (i,j) in enumerate(rank[:5]):
                self.report.write('%2d. %-32.32s %-12.12s %.2f%%\r\n' % \
                        (i+1, formatFactorName(factors[j], factorTypeMap), 
                         factorTypeMap[factors[j].name].name, factorVols[j]*100.0))
            self.report.newline()
            
            self.report.write('5 LEAST risky factors:\r\n')
            self.report.write(header)
            self.report.write(self.report.divider(len(header)))
            displayOrder = [(5-i,j) for (i,j) in enumerate(rank[-5:])]
            displayOrder.reverse()
            for (i,j) in displayOrder:
                self.report.write('%2d. %-32.32s %-12.12s %.2f%%\r\n' % \
                        (i, formatFactorName(factors[j], factorTypeMap), 
                         factorTypeMap[factors[j].name].name, factorVols[j]*100.0))
            self.report.newline()
            
            self.report.write('Large changes in factor risk > %.1f%%:\r\n' % (100.0 * threshold))
            header = '    FACTOR                           TYPE            RISK (%)  PREV (%)\r\n'
            self.report.write(header)
            self.report.write(self.report.divider(len(header)))
            counter = 1
            for i in range(len(factors)):
                if factorTypeMap[factors[i].name]==ExposureMatrix.StatisticalFactor:
                    continue
                factorID = factors[i].factorID
                pidx = p_factorIdxMap.get(factorID)
                if pidx is not None:
                    diff = abs(factorVols[i] - p_factorVols[pidx])
                    if diff > threshold:
                        self.report.write('%2d. %-32.32s %-12.12s %8.2f%% %8.2f%%\r\n' % \
                            (counter, formatFactorName(factors[i], factorTypeMap), 
                             factorTypeMap[factors[i].name].name, 
                             factorVols[i]*100.0, p_factorVols[i]*100.0))
                        counter += 1
            self.report.newline()
    
    def displayFactorCorrChecks(self, modelDB):
        if not isinstance(self.modelSelector, MFM.StatisticalFactorModel):
            self.modelSelector.setFactorsForDate(self.currModelData.rmi.date, modelDB)
            factors = self.modelSelector.factors
            expM = self.currModelData.getExposureMatrix()
            factorTypeMap = dict()
            for fType in expM.factorIdxMap_.keys():
                for fName in expM.factorIdxMap_[fType].keys():
                    factorTypeMap[fName] = fType
            factorCov = self.currModelData.getFactorCov()
            factorVols = numpy.sqrt(numpy.diag(factorCov))
            factorCorr = factorCov / ma.outer(factorVols, factorVols)
            factorCorr = factorCorr.filled(0.0)
            allFactorCorrs = [(factorCorr[i,j], i, j) for i in range(
                            len(factors)) for j in range(len(factors)) if i > j]
            notAllowed = [ExposureMatrix.CurrencyFactor, 
                          ExposureMatrix.StatisticalFactor]
            allFactorCorrs = [n for n in allFactorCorrs if
                      factorTypeMap[factors[n[1]].name] not in notAllowed and \
                      factorTypeMap[factors[n[2]].name] not in notAllowed]
            rank = numpy.argsort([(n[0]) for n in allFactorCorrs])
            displayOrderPos = [(10-i,j) for (i,j) in enumerate(rank[-10:])]
            displayOrderPos.reverse()
            displayOrderNeg = [(i+1,j) for (i,j) in enumerate(rank[:10])]
            
            for (n, title)  in [(displayOrderPos, 'positively'), 
                                (displayOrderNeg, 'negatively')]:
                self.report.write('10 most %s correlated (non-currency) factors:\r\n' % title)
                header = '    FACTOR                           TYPE         FACTOR                           TYPE           CORR\r\n'
                self.report.write(header)
                self.report.write(self.report.divider(len(header)))
                for (i,j) in n:
                    fIdx0 = allFactorCorrs[j][1]
                    fIdx1 = allFactorCorrs[j][2]
                    self.report.write('%2d. %-32.32s %-12.12s %-32.32s %-12.12s %6.2f\r\n' % (i, 
                             formatFactorName(factors[fIdx0], factorTypeMap), 
                             factorTypeMap[factors[fIdx0].name].name, 
                             formatFactorName(factors[fIdx1], factorTypeMap), 
                             factorTypeMap[factors[fIdx1].name].name, 
                             allFactorCorrs[j][0]))
                self.report.newline()
                
    def displayLargeRiskChanges(self, modelDB, threshold=0.10, total=False):
        if not total:
            riskDict = self.currModelData.getSpecificRisks()[0]
            p_riskDict = self.prevModelData.getSpecificRisks()[0]
            riskType = 'specific'
        else:
            risks = list()
            for modeldata in [self.currModelData, self.prevModelData]:
                dat = Utilities.Struct()
                dat.factorCovariance = modeldata.getFactorCov()
                dat.exposureMatrix = modeldata.getExposureMatrix()
                dat.exposureMatrix.fill(0.0)
                dat.specificRisk = modeldata.getSpecificRisks()[0]
                risks.append(modeldata.modelSelector.computeTotalRisk(dat, modelDB))
            riskDict = dict(risks[0])
            p_riskDict = dict(risks[1])
            riskType = 'total'
        commonAssets = list(set(riskDict.keys()).intersection(list(p_riskDict.keys())))
        diff = numpy.array([riskDict[sid] - p_riskDict[sid] for sid in commonAssets])
        bigChange = numpy.flatnonzero(ma.getmaskarray(
                        ma.masked_where(abs(diff) > threshold, diff)))
        assetList = [commonAssets[i] for i in bigChange]
        tr = modelDB.loadTotalReturnsHistory(self.modelSelector.rmg, 
                        self.currModelData.rmi.date, assetList, 0)
        returnsDict = dict(zip(assetList, 100.0 * tr.data[:,0]))
        estuDict = dict((s, True) for s in self.currModelData.getESTU())
        riskDict = dict((s, v*100.0) for (s,v) in riskDict.items())
        p_riskDict = dict((s, v*100.0) for (s,v) in p_riskDict.items())
        self.report.write('Assets with %s risk change > %.1f%%:\r\n' % (riskType, 100.0 * threshold))
        self.report.displayAssetListDetails(assetList, self.currModelData, None,
            'RISK %     PREV    TR %      ESTU',
            riskDict, p_riskDict, returnsDict, estuDict)
    
    def displayLowSpecificCorrelations(self, modelDB):
        (specificRiskMap, specificCovMap) = self.currModelData.getSpecificRisks()
        runningTally = list()
        for (sid0, sidCovMap) in specificCovMap.items():
            for (sid1, cov) in sidCovMap.items():
                corr = cov / (specificRiskMap[sid0] * specificRiskMap[sid1])
                runningTally.append((corr, sid0, sid1))
        runningTally.sort()
        if len(specificCovMap)==0:
            return
        self.report.write('Asset pairs with lowest specific correlations:\r\n')
        (corrList, sid0List, sid1List) = zip(*runningTally[:10])
        corrDict = dict(zip(sid0List, corrList))
        sidLinkageDict = dict((sid0List[i], '%s %s' % ( 
                                sid1List[i].getModelID().getIDString(),
                                self.currModelData.getSubIssueName(sid1List[i])[:9])) 
                            for i in range(len(sid0List)))
        self.report.displayAssetListDetails(sid0List, self.currModelData, None,
                '  CORR LINKED-ID', corrDict, sidLinkageDict)
    
    def displayPortfolioCorrelations(self, modelDB, marketDB):
        self.report.newline()
        univ = self.currModelData.getAllSubIssues()
        mdl2sub = dict((n.getModelID(),n) for n in univ)
        folios = list()
        folioList = MODEL_INDEX_MAP.get(self.currModelData.modelSelector.mnemonic[:4])
        if folioList is None:
            self.report.write('No benchmarks specified for %s\r\n' 
                    % self.currModelData.modelSelector.mnemonic)
            self.report.newline()
            return False
        imp = modelDB.getIssueMapPairs(self.currModelData.rmi.date)
        for f in folioList:
            port = modelDB.getIndexConstituents(f, self.currModelData.rmi.date, 
                            marketDB, rollBack=20, issueMapPairs=imp)
            if len(port) > 0 :
                (assets, weights) = zip(*[(mdl2sub[a], w) for (a,w) \
                                        in port if a in mdl2sub])
                weights = numpy.array(weights)
                weights /= numpy.sum(weights)
                folios.append(list(zip(assets, weights)))
            else:
                folios.append([(univ[0],0.0)])
        estu = self.currModelData.getESTU()
        w = [self.currModelData.getSubIssueMarketCap(s) for s in estu]
        w /= numpy.sum(w)
        folios.append(list(zip(estu, w)))
        folioList.append('%s ESTU' % self.modelSelector.mnemonic)
        
        (vols, corr) = Utilities.compute_correlation_portfolios(
                        self.currModelData.getExposureMatrix(), 
                        self.currModelData.getFactorCov(), 
                        self.currModelData.getSpecificRisks()[0], folios, 
                        self.currModelData.getSpecificRisks()[1])
        vols = vols.filled(0.0)
        corr = corr.filled(0.0)
        
        header = '    BENCHMARK            %s\r\n' % ('\t'.join(['(%d)' % (n+1) for n in range(len(folioList))]))
        self.report.write(header)
        self.report.write(self.report.divider(len(header.replace('\t','    '))))
        for (i,f) in enumerate(folioList):
            line = '%2d. %-19.19s\t' % (i+1, f)
            for j in range(len(folioList)):
                line += '%5.2f\t' % (corr[i,j])
            self.report.write(line + '\r\n')
        self.report.newline()
        
        header = '    BENCHMARK            NAMES     RISK\r\n'
        self.report.write(header)
        self.report.write(self.report.divider(len(header)))
        for (i,f) in enumerate(folioList):
            line = '%2d. %-19.19s  %5d   %5.2f%%' \
                    % (i+1, f, len(folios[i]), vols[i] * 100.0) 
            self.report.write(line + '\r\n')
        self.report.newline()
        self.report.newline()
        
    def displayPortfolioCorrelationsAndExposures(self, modelDB, marketDB): 
        self.report.newline()
        univ = self.currModelData.getAllSubIssues()
        mdl2sub = dict((n.getModelID(),n) for n in univ)
        folios = list()
        folioList = MODEL_INDEX_MAP.get(self.currModelData.modelSelector.mnemonic[:4])
        
        #get information for benchmark exposures
        expM = self.currModelData.getExposureMatrix()
        assetIds = [s.getModelID().getIDString() for s in expM.getAssets()]
        assetIdxMap = dict((j,i) for (i,j) in enumerate(assetIds))
        styleFactors = self.currModelData.modelSelector.styles \
            + self.currModelData.modelSelector.macro_core \
            + self.currModelData.modelSelector.macro_market_traded \
            + self.currModelData.modelSelector.macro_equity \
            + self.currModelData.modelSelector.macro_sectors
        fIdx = [expM.getFactorIndex(f.name) for f in styleFactors]
        exposureMatrix = ma.take(ma.asarray(expM.getMatrix()), fIdx, axis=0).filled(0.0)
        
        if folioList is None:
            self.report.write('No benchmarks specified for %s\r\n' 
                    % self.currModelData.modelSelector.mnemonic)
            self.report.newline()
            return False
        imp = modelDB.getIssueMapPairs(self.currModelData.rmi.date)
        
        # create dictionary to store benchmark exposures
        exposuresDict={}
        
        for f in folioList:
            port = modelDB.getIndexConstituents(f, self.currModelData.rmi.date, 
                            marketDB, rollBack=20, issueMapPairs=imp)
            
            if len(port) > 0 :
                #Calculate benchmark factor exposures
                IdsWghtMap={}
                (assets, weights) = zip(*[(mdl2sub[a], w) for (a,w) \
                                        in port if a in mdl2sub])
                weights = numpy.array(weights)
                weights /= numpy.sum(weights)
                folios.append(list(zip(assets, weights)))
                
                bmIdx = folioList.index(f)
                
                # make dictionary with str modelIDs of benchmark assets as keys, benchmark weights as values
                for (sid, wt) in folios[bmIdx]:
                    key = sid.getModelID().getIDString()
                    IdsWghtMap[key] = wt
                
                # calculate benchmark exposures
                validIds = [n for n in IdsWghtMap.keys() if n in assetIdxMap]
                wgts = [IdsWghtMap[n] for n in validIds]
                indices = [assetIdxMap[n] for n in validIds]
                if len(indices) > 0:
                    wgts = wgts / numpy.sum(wgts)
                    expMat = ma.take(exposureMatrix, indices, axis=1)
                    expMat *= wgts
                    # store benchmark exposures in dict, with benchmark name as key
                    exposuresDict[f] = numpy.sum(expMat, axis=1)
                
            else:
                folios.append([(univ[0],0.0)])
        
        # estu names and market caps
        estu = self.currModelData.getESTU()
        w = [self.currModelData.getSubIssueMarketCap(s) for s in estu]
        w /= numpy.sum(w)
        
        # estu names and weights in estu
        estuWeightsMap = modelDB.getRMIESTUWeights(self.currModelData.rmi)
        # change keys from subissue ids to string model ids
        estuWgtsMap = dict((i.getModelID().getIDString(), v)
                            for (i, v) in estuWeightsMap.items())
        
        # calculate estu exposure
        validIds = [n for n in estuWgtsMap.keys() if n in assetIdxMap]
        wgts = [estuWgtsMap[n] for n in validIds]
        indices = [assetIdxMap[n] for n in validIds]
        modelESTUName = '%s ESTU' % self.modelSelector.mnemonic
        if len(indices) > 0:
            wgts = wgts / numpy.sum(wgts)
            expMat = ma.take(exposureMatrix, indices, axis=1)
            expMat *= wgts
            exposuresDict[modelESTUName] = numpy.sum(expMat, axis=1)
            folios.append(list(zip(estu, w)))
            folioList.append(modelESTUName)
         
        # calculate volatility and correlation
        (vols, corr) = Utilities.compute_correlation_portfolios(
                        self.currModelData.getExposureMatrix(), 
                        self.currModelData.getFactorCov(), 
                        self.currModelData.getSpecificRisks()[0], folios, 
                        self.currModelData.getSpecificRisks()[1])
        vols = vols.filled(0.0)
        corr = corr.filled(0.0)
        
        # style factor names
        styleFactorNames=[f.name for f in styleFactors]
        
        # write tables
        header = '    BENCHMARK            %s\r\n' % ('\t'.join(['(%d)' % (n+1) for n in range(len(folioList))]))
        self.report.write(header)
        self.report.write(self.report.divider(len(header.replace('\t','    '))))
        for (i,f) in enumerate(folioList):
            line = '%2d. %-19.19s\t' % (i+1, f)
            for j in range(len(folioList)):
                line += '%5.2f\t' % (corr[i,j])
            self.report.write(line + '\r\n')
        self.report.newline()

        header = '    BENCHMARK            NAMES     RISK '
        header = header + '%s\r\n' % ('\t'.join(['%-13.13s' % (' '*((13-len(n))//2) + n.upper()) for n in styleFactorNames]))
        self.report.write(header)
        self.report.write(self.report.divider(len(header.replace('\t','   '))))
        for (i,f) in enumerate(folioList):
            line = '%2d. %-19.19s  %5d   %5.2f%% ' % (i+1, f, len(folios[i]), vols[i] * 100.0)
            line += '\t'.join(['% #1.9f' % n for n in exposuresDict[f]])
            self.report.write(line + '\r\n')
        self.report.newline()
        self.report.newline()    
        
    def displayBetaDiscrepancies(self, modelDB, threshold=0.25, hist=False):
        univ = self.currModelData.getAllSubIssues()
        if not hasattr(self.modelSelector, 'modelHack'):
            self.modelSelector.modelHack = Utilities.Struct()
            self.modelSelector.modelHack.nonCheatHistoricBetas = True
        if not hist:
            betas = modelDB.getRMIPredictedBeta(self.currModelData.rmi)
            p_betas = modelDB.getRMIPredictedBeta(self.prevModelData.rmi)
            betaType = 'predicted'
        else:
            betas = dict(); p_betas = dict()
            if self.modelSelector.modelHack.nonCheatHistoricBetas:
                betas = modelDB.getHistoricBeta(
                            self.currModelData.rmi.date, univ)
                p_betas = modelDB.getHistoricBeta(
                            self.prevModelData.rmi.date, univ)
            else:
                betas = modelDB.getHistoricBetaOld(
                            self.currModelData.rmi.date, univ)
                p_betas = modelDB.getHistoricBetaOld(
                            self.prevModelData.rmi.date, univ)
            betaType = 'historical'
        commonAssets = list(set(betas.keys()).intersection(univ).\
                            intersection(list(p_betas.keys())))
        diff = numpy.array([betas[sid] - p_betas[sid] \
                            for sid in commonAssets])
        bigChange = numpy.flatnonzero(ma.getmaskarray(
                            ma.masked_where(abs(diff) > threshold, diff)))
        assetList = [commonAssets[i] for i in bigChange]
        if not hist:
            sourceDict = self.decomposePredictedBeta(assetList, modelDB, betas, p_betas)
        tr = modelDB.loadTotalReturnsHistory(self.modelSelector.rmg, 
                        self.currModelData.rmi.date, assetList, 0)
        returnsDict = dict(zip(assetList, 100.0 * tr.data[:,0]))
        estuDict = dict((s, True) for s in self.currModelData.getESTU())
        self.report.write('Assets with %s beta change > %.2f:\r\n' % (betaType, threshold))
        if hist:
            self.report.displayAssetListDetails(assetList, self.currModelData, None,
                                                '  BETA     PREV    TR %     ESTU', 
                                                betas, p_betas, returnsDict, estuDict)
        else:
            self.report.displayAssetListDetails(assetList, self.currModelData, None,
                                                '  BETA     PREV    TR %     ESTU     FACTOR', 
                                                betas, p_betas, returnsDict, estuDict, sourceDict)
    
    def decomposePredictedBeta(self, sidList, modelDB, betas, p_betas):
        result = dict()
        rmgSidDict = dict()

        isoRMGMap = dict([(rmg.mnemonic, rmg) for rmg in modelDB.getAllRiskModelGroups()])
        for sid in sidList:
            #rmgs is in form of (tradingCtry, Hc)
            processed = False
            rmgs =self.currModelData.getSubIssueCountries(sid)
            if rmgs is not None:
                try:
                    tradingCtryIso, hcIso  = rmgs
                    tradingRmg = isoRMGMap[tradingCtryIso]
                    hcRmg = isoRMGMap[hcIso]

                    if hcRmg in self.currModelData.modelSelector.rmg:
                        hc = hcRmg
                        if tradingCtryIso == 'CN' and hcIso == 'CN':
                            #Special treatment for China-A
                            hc = Utilities.Struct()
                            hc.mnemonic = 'XC'
                            hc.rmg_id = -2
                    else:
                        hc = tradingRmg
                except:
                    hc = isoRMGMap.get(rmgs[0])

                if hc is not None:
                    rmgSidDict.setdefault(hc, list()).append(sid)
                    processed = True
            if not processed:
                #If we cannot find neither the hc or tradingCountry, map it to None
                result[sid] = None
        
        #Now pull the rmg_market_portfolio and do the beta decomposition
        for (rmg, rmgSids) in rmgSidDict.items():
            marketPort = modelDB.getRMGMarketPortfolio(rmg, self.currModelData.rmi.date)
            decom = self.getDecomposition(rmgSids, 
                                        self.currModelData.getExposureMatrix(),
                                        self.currModelData.getFactorCov(), 
                                        self.currModelData.getSpecificRisks()[0],
                                        marketPort, rmg)
            p_marketPort = modelDB.getRMGMarketPortfolio(rmg, self.prevModelData.rmi.date)
            p_decom = self.getDecomposition(rmgSids, 
                                        self.prevModelData.getExposureMatrix(),
                                        self.prevModelData.getFactorCov(), 
                                        self.prevModelData.getSpecificRisks()[0],
                                        p_marketPort, rmg)
            if decom.shape[1] != p_decom.shape[1]:
                #If the cov matrix structure has changed, skip the pBeta decomposition
                return result
            delta = decom - p_decom
            for sIdx, sid in enumerate(rmgSids):
                #Make sure we are decomposing correctly
                try:
                    assert(abs(betas[sid] - numpy.sum(decom[sIdx, :])) < 0.001)
                    assert(abs(p_betas[sid] - numpy.sum(p_decom[sIdx, :])) < 0.001)
                except:
                    logging.error('Database PBeta: %.4f, Decomposition PBeta: %.4f'%\
                                      (betas[sid], numpy.sum(decom[sIdx, :])))
                    logging.error('Database Previous PBeta: %.4f, Decomposition Previous PBeta: %.4f'%\
                                      (p_betas[sid], numpy.sum(p_decom[sIdx, :])))

                diff = numpy.array([abs(k) for k in delta[sIdx, :]])
                idxDict = dict([(k, idx) for (idx, k) in enumerate(diff)])
                pos = idxDict[numpy.amax(diff)]
                if pos == len(self.currModelData.getExposureMatrix().getFactorNames()):
                    fName = "Specific Risk"
                else:
                    fName = self.currModelData.getExposureMatrix().getFactorNames()[pos]
                result[sid] = fName
        return result

    def getDecomposition(self, assets, expMatrix, factorCov,
                         srDict, marketPortfolio, rmg, forceRun=False):
        """Return a factor decomposition of the predicted beta. Sum of each row will be 
        the predicted beta of each asset"""

        logging.debug('getDecomposition: begin')
        # Make sure market portfolio is covered by model
        exposureAssets = expMatrix.getAssets()
        assetIdxMap = dict([(j,i) for (i,j) in enumerate(exposureAssets)])
        marketPortfolio = [(a,w) for (a,w) in marketPortfolio if a in assetIdxMap]
        if len(marketPortfolio) == 0 and forceRun:
            logging.warning('Empty market portfolio')
            return [0] * len(assets)

        mktIndices, weights = zip(*[(assetIdxMap[a], w) for (a,w) in marketPortfolio])
        market_ids = [exposureAssets[i] for i in mktIndices]
        market_id_map = dict([(exposureAssets[j],i) for (i,j) in enumerate(mktIndices)])
        # Compute market portfolio specific variance
        univ_sr = numpy.array([srDict[asset] for asset in market_ids])
        univ_sv = univ_sr * univ_sr
        market_sv = numpy.inner(weights, univ_sv * weights)

        # Compute market portfolio common factor variances
        expM_Idx = [assetIdxMap[a] for a in market_ids]
        market_exp = numpy.dot(ma.take(expMatrix.getMatrix(), expM_Idx, axis=1), weights)
        market_cv_exp = numpy.dot(factorCov, market_exp)
        market_var = numpy.inner(market_exp, market_cv_exp) + market_sv
        logging.info('Market risk for %s is %.2f%%'% (rmg.mnemonic,
                                                      100.0 * math.sqrt(market_var)))
                                                      

        # Compute asset predicted betas
        beta = ma.zeros((len(assets), expMatrix.getMatrix().shape[0] + 1), float)
        for i in range(len(assets)):
            asset = assets[i]
            if asset.isCashAsset():
                continue
            else:
                idx = assetIdxMap[asset]
                # Compute asset factor covariance with market
                fv = (expMatrix.getMatrix()[:,idx] * market_cv_exp)/ market_var
                sv = 0.0
                # Add specific component
                if asset in market_id_map:
                    sv = (weights[market_id_map[asset]] * srDict[asset] * srDict[asset])/ market_var
            fIdx = range(expMatrix.getMatrix().shape[0])
            ma.put(beta[i, :], fIdx, fv)
            ma.put(beta[i, :], -1, sv)
        logging.debug('getDecomposition: end')
        return beta

    def displayDailyThought(self):
        infile = open('koans.txt')
        blurbCollection = []
        new_blurb = True
        for line in infile:
            if new_blurb:
                blurb = ''
                new_blurb = False
            if line=='\n' or line=='\r':
                blurbCollection.append(blurb)
                new_blurb = True
                continue
            blurb += line
        modelDate = self.currModelData.rmi.date
        i = int(time.mktime(modelDate.timetuple()) / (3600*24)) % len(blurbCollection)
        self.report.write(blurbCollection[i])

if __name__ == '__main__':
    usage = "usage: %prog [options] <YYYY-MM-DD> [<YYYY-MM-DD>]"
    cmdlineParser = optparse.OptionParser(usage=usage)
    cmdlineParser.add_option('--email', action='store',
                             default=None, dest='emailRecipients',
                             help='email report')
    cmdlineParser.add_option('--store-report', action='store_true',
                             default=False, dest='storeReport',
                             help='write the report to a text file')
    cmdlineParser.add_option("--report-file", action="store",
                             default=None, dest="reportFile",
                             help="report file name")
    cmdlineParser.add_option("--prior-model", action="store",
                             default=None, dest="priorModelName",
                             help="model name for previous day")
    cmdlineParser.add_option("--max-records", action="store",
                             default=50, dest="maxRecords",
                             help="max records per report section")
        
    Utilities.addModelAndDefaultCommandLine(cmdlineParser)
    (options, args) = cmdlineParser.parse_args()
    if options.emailRecipients is not None:
        sendTo = options.emailRecipients.split(',')
    else:
        sendTo=[]
    if len(args) < 1 or len(args) > 2:
        cmdlineParser.error("Incorrect number of arguments")
    riskModel_Class = Utilities.processModelAndDefaultCommandLine(
        options, cmdlineParser)
    if options.priorModelName is not None:
        try:
            priorRiskModel_Class = riskmodels.getModelByName(
                options.priorModelName)
        except KeyError:
            print('Unknown risk model "%s"' % options.priorModelName)
            names = sorted(modelNameMap.keys())
            print('Possible values:', ', '.join(names))
            sys.exit(1)
    else:
        priorRiskModel_Class = riskModel_Class

    modelDB = ModelDB.ModelDB(sid=options.modelDBSID, 
                              user=options.modelDBUser, 
                              passwd=options.modelDBPasswd)
    marketDB = MarketDB.MarketDB(sid=options.marketDBSID, 
                                 user=options.marketDBUser, 
                                 passwd=options.marketDBPasswd)
    riskModel_ = riskModel_Class(modelDB, marketDB)
    priorRiskModel_ = priorRiskModel_Class(modelDB, marketDB)

    if isinstance(riskModel_, PhoenixModels.CommodityFutureRegionalModel):
        modelAssetType = 'Future'
    else:
        modelAssetType = 'Equity'


    startDate = Utilities.parseISODate(args[0])
    if len(args) == 1:
        endDate = startDate
    else:
        endDate = Utilities.parseISODate(args[1])

    dates = modelDB.getDateRange(riskModel_.rmg, startDate, endDate, True)
    modelDB.totalReturnCache = None
    modelDB.specReturnCache = None
    
    status = 0
    for d in dates:
        try:
            logging.info('Processing %s' % d)
            checker_ = RiskModelValidator(riskModel_, priorRiskModel_, d,
                                              modelDB, marketDB, modelAssetType)
            checker_.report.maxRecords = int(options.maxRecords)
            checker_.report.makeTitleBanner('MODEL STRUCTURE')
            checker_.displayModelStructure(modelDB)
            checker_.report.makeTitleBanner('MODEL UNIVERSE')
            checker_.displayModelCoverage(modelDB, marketDB)
            checker_.report.makeTitleBanner('ESTIMATION UNIVERSE')
            checker_.displayModelESTUDetails(modelDB)
            checker_.report.makeTitleBanner('REGRESSION DIAGNOSTICS')
            checker_.displayRegressionStatistics(modelDB)
            checker_.displayFactorReturnsChecks(modelDB)
            checker_.report.makeTitleBanner('ASSET SPECIFIC RETURNS & FACTOR EXPOSURES')
            checker_.displayDodgySpecificReturns(modelDB)
            checker_.displayLargeExposureChanges(modelDB)
            checker_.report.makeTitleBanner('FACTOR RISKS & COVARIANCES')
            checker_.displayCovarianceMatrixSpecs(modelDB)
            checker_.displayFactorVolChecks(modelDB)
            checker_.displayFactorCorrChecks(modelDB)
            checker_.report.makeTitleBanner('ASSET PREDICTED RISKS & BETAS')
            checker_.displayLargeRiskChanges(modelDB)
            checker_.displayLargeRiskChanges(modelDB, threshold=0.075, total=True)
            checker_.displayLowSpecificCorrelations(modelDB)
            checker_.displayBetaDiscrepancies(modelDB)
            checker_.displayBetaDiscrepancies(modelDB, hist=True)
            if riskModel_.mnemonic[2:4] == 'US' and riskModel_.mnemonic[-2:] != '-S':
                checker_.report.makeTitleBanner('PORTFOLIO CORRELATIONS, RISKS & EXPOSURES')
                checker_.displayPortfolioCorrelationsAndExposures(modelDB, marketDB)
            else:    
                checker_.report.makeTitleBanner('PORTFOLIO RISKS & CORRELATIONS')
                checker_.displayPortfolioCorrelations(modelDB, marketDB)
            checker_.displayDailyThought()
            
            report_ = checker_.report
            if len(sendTo) > 0:
                report_.emailReport(d, sendTo)
            if options.storeReport:
                filename=options.reportFile
                if filename == None:
                    filename = '%s.report.%04d%02d%02d' % (riskModel_.name, d.year, d.month, d.day)
                i = 0
                if os.path.isfile(filename + ".v" + str(i)):
                    logging.info(filename  + ".v" + str(i) + " already exists")
                    i = 1
                    while os.path.isfile(filename + ".v" + str(i)):
                        logging.info(filename + ".v" + str(i) + " already exists")
                        i = i + 1
                
                filename = filename + ".v" + str(i)
                path = os.path.dirname(filename)
                if not os.path.exists(path) and path != '':
                    os.makedirs(path)
                logging.info("Writing report to " + filename)
                report_.writeReport(filename)
        except Exception as ex:
            logging.error('Exception caught during processing', exc_info=True)
            modelDB.revertChanges()
            status = 1
    marketDB.finalize()
    modelDB.finalize()
    sys.exit(status)
