"""Classes to retrieve data from MarketDB for transfer into ModelDB."""
import cx_Oracle
import datetime
import logging
import numpy
import numpy.ma as ma
import operator
import copy
import pandas
from collections import defaultdict
from string import Template
from marketdb import MarketID
from marketdb import  MarketDB
from riskmodels import MarketIndex
from riskmodels import Matrices
from riskmodels import ModelDB
from riskmodels import ModelID
from riskmodels import RetSync
from riskmodels import Utilities
from marketdb.Utilities import listChunkIterator, OneToManyDictionary
from riskmodels import FactorReturns
from riskmodels import AssetProcessor_V4
from riskmodels import ProcessReturns
from riskmodels import TimeSeriesRegression
from riskmodels import Outliers
from marketdb import VendorUtils
from marketdb.Utilities import Tensor
import marketdb.Utilities as MktDBUtils
from marketdb.Utilities import PrimaryKey
from marketdb.Utilities import MATRIX_DATA_TYPE as MX_DT
from marketdb.resources import DateRange, AssetIDHistory, AssetIDInfo
from riskmodels.StyleExposures import computeAnnualDividends, computeAnnualDividendsNew

DATABASE_TYPE = MktDBUtils.enum(MARKETDB='MarketDB', MODELDB='ModelDB')
#--- ALL SubIssueIDs: SELECT distinct sub_id FROM sub_issue where sub_id not like 'DCSH_%'


class ModelDBHelper:
    def __init__(self, connections):
        self.modelDB = connections.modelDB
        self.marketDB = connections.marketDB
        self.log = self.modelDB.log
        self.idMapCache = ModelDB.FromThruCache()
        self.subidMapCache = ModelDB.FromThruCache(useModelID=False)

    def buildList(self, dateList, sidList, mapTableName):
        """Build the list of all Axioma IDs that correspond to the sub-issues in mapTableName
        on the given dates.
        Returns a list of MarketIDs and a dictonary that maps each date
        to a dictionary mapping sub-issues to the index of their corresponding
        MarketID in the MarketID list.
        """
        sidDtMap = dict()
        axids = set()
        issueIDs = self.modelDB.loadFromThruTable(
            'sub_issue', dateList, sidList,
            ['issue_id'], keyName='sub_id', cache=self.subidMapCache)
        for (dIdx, date) in enumerate(dateList):
            sidIdList = [(sid, ModelID.ModelID(string=i.issue_id)) for (sid, i) in
                      zip(sidList, issueIDs[dIdx]) if i is not None]
            marketIDs = self.modelDB.loadFromThruTable(
                mapTableName, [date], [i[1] for i in sidIdList],
                ['marketdb_id'], keyName='modeldb_id', cache=self.idMapCache)
            sidAxidMap = dict([(sidId[0], MarketID.MarketID(string=v.marketdb_id))
                               for (sidId, v) in zip(sidIdList, marketIDs[0])
                               if v is not None])
            axids |= set(sidAxidMap.values())
            sidDtMap[date] = sidAxidMap
        axids = list(axids)
        axidIdxMap = dict(zip(axids, range(len(axids))))
        for sidMap in sidDtMap.values():
            for (sid, axid) in sidMap.items():
                sidMap[sid] = axidIdxMap[axid]
        #self.log.debug('%d Axioma IDs', len(axids))
        return (axids, sidDtMap)
    
    def buildAxiomaIdList(self, dateList, sidList):
        """Build the list of all Axioma IDs that correspond to the sub-issues
        on the given dates.
        Returns a list of MarketIDs and a dictonary that maps each date
        to a dictionary mapping sub-issues to the index of their corresponding
        MarketID in the MarketID list.
        """
        return self.buildList(dateList, sidList, 'issue_map')
    
    def buildMktFuturesList(self, dateList, sidList):
        """Build the list of all Axioma IDs that correspond to the futures sub-issues
        on the given dates.
        Returns a list of MarketIDs and a dictionary that maps each date
        to a dictionary mapping sub-issues to the index of their corresponding
        MarketID in the MarketID list.
        """
        return self.buildList(dateList, sidList, 'future_issue_map')

    def buildHistory(self, sidList, mapTableName):
        """Build the list of all Axioma IDs that correspond to the sub-issues
        in mapTableName.
        Returns a list of Axioma IDs and a dictionary mapping Axioma IDs
        to their history of mappings to sub-issues.
        """
        axids = set()
        # Load the table for today to get the cache filled in
        self.modelDB.loadFromThruTable(
            'sub_issue', [datetime.date.today()], sidList,
            ['issue_id'], keyName='sub_id', cache=self.subidMapCache)
        issues = set()
        for sid in sidList:
            issues.update([ModelID.ModelID(string=i.issue_id)
                           for i in self.subidMapCache.getAssetHistory(sid)])
        marketIDs = self.modelDB.loadFromThruTable(
            mapTableName, [datetime.date.today()], sorted(issues),
            ['marketdb_id'], keyName='modeldb_id', cache=self.idMapCache)
        axids = set()
        axidSubIssueMap = dict()
        sidAxidMap = dict()
        for sid in sidList:
            issueHistory = self.subidMapCache.getAssetHistory(sid)
            for issueInterval in issueHistory:
                axidHistory = self.idMapCache.getAssetHistory(
                    ModelID.ModelID(string=issueInterval.issue_id))
                for axidInterval in axidHistory:
                    if issueInterval.fromDt < axidInterval.thruDt \
                       and axidInterval.fromDt < issueInterval.thruDt:
                        axids.add(MarketID.MarketID(
                            string=axidInterval.marketdb_id))
                        sidInterval = Utilities.Struct()
                        sidInterval.sub_issue = sid
                        sidInterval.fromDt = max(axidInterval.fromDt,
                                                 issueInterval.fromDt)
                        sidInterval.thruDt = min(axidInterval.thruDt,
                                                 issueInterval.thruDt)
                        axidSubIssueMap.setdefault(
                            axidInterval.marketdb_id, list()).append(
                            sidInterval)
                        sidAxidMap.setdefault(sid, list()).append(axidInterval)
        axids = list(axids)
        #self.log.debug('%d Axioma IDs', len(axids))
        return (axids, axidSubIssueMap, sidAxidMap)

    def buildAxiomaIdHistory(self, sidList):
        """Build the list of all Axioma IDs that correspond to the sub-issues.
        Returns a list of Axioma IDs and a dictionary mapping Axioma IDs
        to their history of mappings to sub-issues.
        """
        return self.buildHistory(sidList, 'issue_map')
    
    def buildMktFuturesHistory(self, sidList):
        """Build the list of all Axioma IDs that correspond to the future sub-issues.
        Returns a list of Axioma IDs and a dictionary mapping Axioma IDs
        to their history of mappings to sub-issues.
        """
        return self.buildHistory(sidList, 'future_issue_map')

class AxiomaDBSource:
    def __init__(self, connections):
        self.marketDB = connections.marketDB
        self.modelDB = connections.modelDB
        self.axiomaDB = connections.axiomaDB
        self.log = self.marketDB.log
        self.modelDBHelper = ModelDBHelper(connections)

class MarketDBSource:
    def __init__(self, connections):
        self.marketDB = connections.marketDB
        self.modelDB = connections.modelDB
        self.log = self.marketDB.log
        self.modelDBHelper = ModelDBHelper(connections)

class ModelDBSource:
    def __init__(self, connections):
        self.modelDB = connections.modelDB
        self.log = self.modelDB.log
        self.modelDBHelper = ModelDBHelper(connections)

class OldModelDB:
    """Provides read access to the old (production) ModelDB databases.
    """
    def __init__(self, maxTransferDt, transDateTime=datetime.datetime.now(),
                 **connectParameters):
        """Create connection to the database.
        """
        self.log = logging.getLogger('OldModelDB')
        self.dbConnection = cx_Oracle.connect(
            connectParameters['user'], connectParameters['passwd'],
            connectParameters['sid'])
        self.dbCursor = self.dbConnection.cursor()
        self.dbCursor.execute(
            'alter session set nls_date_format="YYYY-MM-DD HH24:MI:SS"')
        self.dbConnection.commit()
        self.dbCursor.arraysize = 20000
        self.maxTransferDt = maxTransferDt
        self.transDateTime = transDateTime
        self.transDateTimeStr = self.transDateTime.strftime('%Y-%m-%d %H:%M:%S')
    
    def finalize(self):
        """Close connection to the database.
        """
        self.dbCursor.close()
        self.dbConnection.close()

class ClassificationBase(MarketDBSource):
    """Base class for classification source from MarketDB."""
    def __init__(self, connections,
                 srcFamilyName, srcMemberName, srcClsDateStr,
                 tgtFamilyName, tgtMemberName, tgtClsDateStr,
                 tableName, tableConnection):
        MarketDBSource.__init__(self, connections)
        self.INCR = 200
        self.argList = [(':aid%d' % i) for i in range(self.INCR)]
        self.query = """SELECT axioma_id, classification_id, weight,
          src_id, ref, change_dt, change_del_flag
          FROM %(tableName)s WHERE axioma_id IN (%(args)s)
          AND revision_id=:rev_id""" % {
            'tableName': tableName,
            'args': ','.join(['%s' % arg for arg in self.argList]) }
        self.defaultDict = dict([(arg, None) for arg in self.argList])
        self.tableConnection = tableConnection
        family = self.marketDB.getClassificationFamily(srcFamilyName)
        members = self.marketDB.getClassificationFamilyMembers(family)
        member = [i for i in members if i.name==srcMemberName][0]
        clsDate = Utilities.parseISODate(srcClsDateStr)
        self.mktClsRev = self.marketDB.getClassificationMemberRevision(
            member, clsDate)
        self.defaultDict['rev_id'] = self.mktClsRev.id
        mktClsLeaves = self.marketDB.getClassificationRevisionLeaves(
            self.mktClsRev)
        mktClsLeafIds = set([i.id for i in mktClsLeaves])
        self.clsIdToName = {i.id: i.name for i in mktClsLeaves}
        tgtFamily = self.modelDB.getMdlClassificationFamily(tgtFamilyName)
        tgtMembers = self.modelDB.getMdlClassificationFamilyMembers(tgtFamily)
        tgtMember = [i for i in tgtMembers if i.name==tgtMemberName][0]
        tgtClsDate = Utilities.parseISODate(tgtClsDateStr)
        tgtClsLeaves = self.modelDB.getMdlClassificationMemberLeaves(
            tgtMember, tgtClsDate)
        tgtClsLeafIds = set([i.id for i in tgtClsLeaves])
        self.modelDB.dbCursor.execute("""SELECT market_ref_id, model_ref_id,
          flag_as_guessed FROM classification_market_map""")
        self.mktMdlMap = dict()
        for (mktRef, mdlRef, isGuess) in self.modelDB.dbCursor.fetchall():
            if mktRef in mktClsLeafIds and mdlRef in tgtClsLeafIds:
                self.mktMdlMap[mktRef] = (mdlRef, isGuess)
        if len(self.mktMdlMap) != len(mktClsLeafIds):
            unmapped = [i for i in mktClsLeaves if i.id not in self.mktMdlMap]
            self.log.info('%d unmapped classification leaves for %s %s %s',
                          len(unmapped), srcFamilyName, srcMemberName,
                          srcClsDateStr)
            for unmp in unmapped:
                self.log.info('Unmapped classification leaf: %s', unmp)
        
    def processCodes(self, cur, axidClsDict):
        for (axid, clsID, weight, srcID, ref, changeDate, changeFlag) \
                in cur.fetchall():
            rval = Utilities.Struct()
            changeDate = changeDate.date()
            if clsID not in self.mktMdlMap:
                rval.classificationIdAndWeight = list()
                rval.isGuess = False
            else:
                (mdlID, isGuess) = self.mktMdlMap[clsID]
                rval.classificationIdAndWeight = [(mdlID, weight)]
                rval.isGuess = (isGuess == 'Y')
                rval.origClassification = self.clsIdToName[clsID]
                rval.sourceRevision = self.mktClsRev
            rval.src_id = self.getSourceID(srcID)
            rval.ref = 'marketdb.classification_constituent, aid %s' \
                       ', change_dt %s' % (axid, changeDate)
            rval.changeDate = changeDate
            rval.changeFlag = changeFlag
            if axid not in axidClsDict:
                axidClsDict[axid] = list()
            axidClsDict[axid].append(rval)
    
    def sortHistories(self, axidClsDict):
        for axid in list(axidClsDict.keys()):
            history = axidClsDict[axid]
            if len(history) <= 1:
                continue
            # sort by timeDate
            history.sort(key=operator.attrgetter('changeDate'))
    
    def getBulkData(self, dateList, iidList):
        self.log.debug('ClassificationBase.getBulkData from src_id %s', self.getSourceID(None))
        retvals = numpy.empty((len(dateList), len(iidList)), dtype=object)
        iidDtMap = dict()
        axids = set()
        marketIDs = self.modelDB.loadFromThruTable(
            'issue_map', dateList, iidList,
            ['marketdb_id'], keyName='modeldb_id',
            cache=self.modelDBHelper.idMapCache)
        for (dIdx, date) in enumerate(dateList):
            idPairs = zip(iidList, marketIDs[dIdx])
            iidMap = dict([(i, v.marketdb_id) for (i,v) in idPairs
                           if v is not None])
            axids |= set(iidMap.values())
            iidDtMap[date] = iidMap
        self.log.debug('%d Axioma IDs', len(axids))
        cur = self.tableConnection.dbCursor
        axidClsDict = dict()
        for axidChunk in listChunkIterator(list(axids), self.INCR):
            myDict = dict(self.defaultDict)
            myDict.update(dict(zip(self.argList, axidChunk)))
            cur.execute(self.query, myDict)
            self.processCodes(cur, axidClsDict)
        self.sortHistories(axidClsDict)
        self.log.debug('%d Axioma ID are classified', len(axidClsDict))
        for (dIdx, date) in enumerate(dateList):
            iidMap = iidDtMap[date]
            for (aIdx, iid) in enumerate(iidList):
                retvals[dIdx, aIdx] = self.findCode(iidMap.get(iid), date,
                                                    axidClsDict)
        return retvals

class Classification(ClassificationBase):
    """Class to retrieve asset classifications from MarketDB for a given
    ModelDB classification. The mapping is done via classification_market_map.
    """
    def __init__(self, connections,
                 srcFamilyName, srcMemberName, srcClsDateStr,
                 tgtFamilyName, tgtMemberName, tgtClsDateStr):
        ClassificationBase.__init__(
            self, connections,
            srcFamilyName, srcMemberName, srcClsDateStr,
            tgtFamilyName, tgtMemberName, tgtClsDateStr,
            'classification_const_active', connections.marketDB)
    
    def getSourceID(self, srcID):
        return srcID
    
    def findCode(self, axid, date, axidClsDict):
        if axid is None or axid not in axidClsDict:
            return None
        if date < self.mktClsRev.from_dt or date >= self.mktClsRev.thru_dt:
            return None
        rval = [i for i in axidClsDict[axid] if i.changeDate <= date]
        if len(rval) == 0 or rval[-1].classificationIdAndWeight is None \
           or rval[-1].changeFlag == 'Y':
            return None
        return rval[-1]
        

class FutureClassification(ClassificationBase):
    """Class to retrieve future asset classifications from MarketDB for a given
    ModelDB classification. The mapping is done via classification_market_map.
    This will return the next classification that an asset will change to.
    """
    PREF_GRACE_DAYS = datetime.timedelta(6)
    def __init__(self, connections,
                 srcFamilyName, srcMemberName, srcClsDateStr,
                 tgtFamilyName, tgtMemberName, tgtClsDateStr):
        ClassificationBase.__init__(
            self, connections,
            srcFamilyName, srcMemberName, srcClsDateStr,
            tgtFamilyName, tgtMemberName, tgtClsDateStr,
            'classification_const_active', connections.marketDB)
        self.SRC_ID = 5
        
    def getSourceID(self, srcID):
        return self.SRC_ID
    
    def findCode(self, axid, date, axidClsDict):
        if axid is None or axid not in axidClsDict:
            return None
        if date < self.mktClsRev.from_dt + self.PREF_GRACE_DAYS:
            # If we have classifications right at the beginning, 
            # return the classification after a few days so switches
            # to preferred industries are more likely to be seen
            rval = [i for i in axidClsDict[axid] if i.changeFlag == 'N'
                    and i.changeDate < self.mktClsRev.from_dt
                    + self.PREF_GRACE_DAYS]
            if len(rval) >= 1:
                return rval[-1]
        
        rval = [i for i in axidClsDict[axid] if i.changeDate > date
                and i.changeFlag == 'N']
        if len(rval) == 0 or rval[0].classificationIdAndWeight is None:
            return None
        return rval[0]

class PastClassification(ClassificationBase):
    """Class to retrieve past asset classifications from MarketDB for a given
    ModelDB classification. The mapping is done via classification_market_map.
    This will return the previous classification of an asset.
    """
    def __init__(self, connections,
                 srcFamilyName, srcMemberName, srcClsDateStr,
                 tgtFamilyName, tgtMemberName, tgtClsDateStr):
        ClassificationBase.__init__(
            self, connections,
            srcFamilyName, srcMemberName, srcClsDateStr,
            tgtFamilyName, tgtMemberName, tgtClsDateStr,
            'classification_const_active', connections.marketDB)
        self.SRC_ID = 6
    
    def getSourceID(self, srcID):
        return self.SRC_ID
    
    def findCode(self, axid, date, axidClsDict):
        if axid is None or axid not in axidClsDict:
            return None
        rval = [i for i in axidClsDict[axid] if i.changeDate < date
                and i.changeFlag == 'N']
        if len(rval) == 0 or rval[-1].classificationIdAndWeight is None:
            return None
        return rval[-1]

class SubIssueData(MarketDBSource):
    """Class to retrieve asset information from MarketDB that is
    required for the sub_issue_data table.
    """
    def __init__(self, connections):
        MarketDBSource.__init__(self, connections)
        self.INCR = 200
    
    def getBulkData(self, dateList, sidList):
        self.log.debug('SubIssueData.getBulkData')
        retvals = numpy.empty((len(dateList), len(sidList)), dtype=object)
        (axids, sidAIdxDtMap) = self.modelDBHelper.buildAxiomaIdList(
            dateList, sidList)
        
        ucp = self.marketDB.loadTimeCurrencySeriesRaw(dateList, axids,
                                                      'asset_dim_ucp')
        tdv = self.marketDB.loadTimeSeries(dateList, axids,
                                           'asset_dim_tdv').data
        tso = self.marketDB.getSharesOutstanding(dateList, axids).data
        for (dIdx, date) in enumerate(dateList):
            sidAIdxMap = sidAIdxDtMap[date]
            for (sIdx, sid) in enumerate(sidList):
                if sid not in sidAIdxMap:
                    continue
                aIdx = sidAIdxMap[sid]
                if ucp[aIdx, dIdx] is None:
                    continue
                rval = Utilities.Struct()
                rval.ucp = ucp[aIdx, dIdx].value
                rval.price_marker = ucp[aIdx, dIdx].price_marker
                rval.currency_id = ucp[aIdx, dIdx].currency_id
                if tdv[aIdx, dIdx] is not ma.masked:
                    rval.tdv = tdv[aIdx, dIdx]
                else:
                    rval.tdv = None
                if tso[aIdx, dIdx] is not ma.masked:
                    rval.tso = tso[aIdx, dIdx]
                else:
                    rval.tso = None
                retvals[dIdx, sIdx] = rval
        return retvals

class FutureSubIssueData(MarketDBSource):
    """Class to retrieve asset information from MarketDB for futures that is
    required for the sub_issue_data table.
    """
    def __init__(self, connections):
        MarketDBSource.__init__(self, connections)
        self.INCR = 200
    
    def getBulkData(self, dateList, sidList):
        self.log.debug('FutureSubIssueData.getBulkData')
        retvals = numpy.empty((len(dateList), len(sidList)), dtype=object)
        (axids, sidAIdxDtMap) = self.modelDBHelper.buildMktFuturesList(
            dateList, sidList)
        
        ucp = self.marketDB.loadTimeCurrencySeriesRaw(dateList, axids,
                                                      'future_dim_ucp')
        tdv = self.marketDB.loadTimeSeries(dateList, axids,
                                           'future_dim_tdv').data
        tso = self.marketDB.loadTimeSeries(dateList, axids, 'future_dim_open_interest').data
        for (dIdx, date) in enumerate(dateList):
            sidAIdxMap = sidAIdxDtMap[date]
            for (sIdx, sid) in enumerate(sidList):
                if sid not in sidAIdxMap:
                    continue
                aIdx = sidAIdxMap[sid]
                if ucp[aIdx, dIdx] is None:
                    continue
                rval = Utilities.Struct()
                rval.ucp = ucp[aIdx, dIdx].value
                rval.price_marker = ucp[aIdx, dIdx].price_marker
                rval.currency_id = ucp[aIdx, dIdx].currency_id
                if tdv[aIdx, dIdx] is not ma.masked:
                    rval.tdv = tdv[aIdx, dIdx]
                else:
                    rval.tdv = None
                if tso[aIdx, dIdx] is not ma.masked:
                    rval.tso = tso[aIdx, dIdx]
                else:
                    rval.tso = None
                retvals[dIdx, sIdx] = rval
        return retvals

class CashSubIssueData(MarketDBSource):
    """Class to insert price information for cash assets.
    """
    def __init__(self, connections):
        MarketDBSource.__init__(self, connections)
        self.currencyCodeMap = dict()
        self.sidRanges = connections.sidRanges
    
    def getCurrencyID(self, isoCode, date):
        if isoCode not in self.currencyCodeMap:
            cid = self.marketDB.getCurrencyID(isoCode, date)
            self.currencyCodeMap[isoCode] = cid
        return self.currencyCodeMap.get(isoCode)
    
    def getBulkData(self, dateList, sidList):
        self.log.debug('CashSubIssueData.getBulkData')
        retvals = numpy.empty((len(dateList), len(sidList)), dtype=object)
        cashIDs = [(idx, sid) for (idx, sid) in enumerate(sidList)
                   if sid.isCashAsset()]
        for (idx, sid) in cashIDs:
            sidRange = self.sidRanges[sid]
            for (dIdx, date) in enumerate(dateList):
                if date < sidRange[0] or date >= sidRange[1]:
                    # Skip assets outside its life-time
                    continue
                rval = Utilities.Struct()
                rval.ucp = 1.0
                rval.tdv = None
                rval.tso = None
                rval.price_marker = 0
                rval.currency_id = self.getCurrencyID(sid.getCashCurrency(),
                                                      date)
                if rval.currency_id is not None:
                    retvals[dIdx,idx] = rval
                else:
                    self.log.warning('No currency ID for %s on %s',
                                     sid.getCashCurrency(), date)
        return retvals

class AxiomaDBReturn(AxiomaDBSource):
    """Class to retrieve asset return from AxiomaDB.asset_dim_return.
    """
    def __init__(self, connections):
        AxiomaDBSource.__init__(self, connections)
        self.DATEINCR = 30
        self.INCR = 200
        self.argList = [('axid%d' % i) for i in range(self.INCR)]
        self.dateArgList = [('date%d' % i) for i in range(self.DATEINCR)]
        self.query = """SELECT axioma_id, dt, value
        FROM asset_dim_return_active where
        dt in (%(dateargs)s) and axioma_id in (%(axidargs)s)""" % {
            'dateargs': ','.join([':%s' % darg for darg
                                  in self.dateArgList]),
            'axidargs': ','.join([':%s' % arg for arg in self.argList])}
        self.defaultDict = dict([(arg, None) for arg
                                 in self.argList + self.dateArgList])
        
    def getBulkData(self, dateList, sidList):
        self.log.debug('AxiomaDBReturn.getBulkData')
        retvals = numpy.empty((len(dateList), len(sidList)), dtype=object)
        (axids, sidAIdxDtMap) = self.modelDBHelper.buildAxiomaIdList(
            dateList, sidList)
        axidStrs = [axid.getIDString() for axid in axids]
        axidDateValues = dict()
        for dateChunk in listChunkIterator(dateList, self.DATEINCR):
            updateDateDict = dict(zip(self.dateArgList, dateChunk))
            for axidChunk in listChunkIterator(axidStrs, self.INCR):
                updateDict = dict(zip(self.argList, axidChunk))
                myDict = self.defaultDict.copy()
                myDict.update(updateDateDict)
                myDict.update(updateDict)
                self.axiomaDB.dbCursor.execute(self.query, myDict)
                for (axid, date, ret) \
                        in self.axiomaDB.dbCursor.fetchall():
                    date = date.date()
                    if ret is not None:
                        ret = float(ret)
                    axidDateValues[(axid, date)] = ret
        for (dIdx, date) in enumerate(dateList):
            sidAIdxMap = sidAIdxDtMap[date]
            for (sIdx, sid) in enumerate(sidList):
                if sid not in sidAIdxMap:
                    continue
                axid = axidStrs[sidAIdxMap[sid]]
                if (axid,date) not in axidDateValues:
                    continue
                val =  axidDateValues[(axid, date)]
                rval = Utilities.Struct()
                rval.tr = val
                retvals[dIdx, sIdx] = rval
        return retvals

class MarketDBReturn(MarketDBSource):
    """Class to retrieve asset return from MarketDB.asset_dim_return.
    """
    def __init__(self, connections):
        MarketDBSource.__init__(self, connections)
        self.INCR = 200
        
    def getBulkData(self, dateList, sidList):
        self.log.debug('MarketDBReturn.getBulkData')
        retvals = numpy.empty((len(dateList), len(sidList)), dtype=object)
        (axids, sidAIdxDtMap) = self.modelDBHelper.buildAxiomaIdList(
            dateList, sidList)
        
        ret = self.marketDB.getTotalReturns(dateList, axids).data
        for (dIdx, date) in enumerate(dateList):
            sidAIdxMap = sidAIdxDtMap[date]
            for (sIdx, sid) in enumerate(sidList):
                if sid not in sidAIdxMap:
                    continue
                aIdx = sidAIdxMap[sid]
                if ret[aIdx, dIdx] is ma.masked:
                    continue
                rval = Utilities.Struct()
                rval.tr = ret[aIdx, dIdx]
                retvals[dIdx, sIdx] = rval
        return retvals

class MarketDBVendorReturn(MarketDBSource):
    """Class to retrieve asset return from MarketDB.asset_dim_vendor_return.
    """
    def __init__(self, connections):
        MarketDBSource.__init__(self, connections)
        
    def getBulkData(self, dateList, sidList):
        self.log.debug('MarketDBVendorReturn.getBulkData')
        retvals = numpy.empty((len(dateList), len(sidList)), dtype=object)
        (axids, sidAIdxDtMap) = self.modelDBHelper.buildAxiomaIdList(dateList, sidList)
        
        ret = self.marketDB.getTotalVendorReturns(dateList, axids).data
        for (dIdx, date) in enumerate(dateList):
            sidAIdxMap = sidAIdxDtMap[date]
            for (sIdx, sid) in enumerate(sidList):
                if sid not in sidAIdxMap:
                    continue
                aIdx = sidAIdxMap[sid]
                if ret[aIdx, dIdx] is ma.masked:
                    continue
                rval = Utilities.Struct()
                rval.tr = ret[aIdx, dIdx]
                retvals[dIdx, sIdx] = rval
        return retvals

class CashReturn(ModelDBSource):
    """Class to compute the return for cash assets
    """
    def __init__(self, connections):
        ModelDBSource.__init__(self, connections)
        self.marketDB = connections.marketDB
        self.sidRMGs = dict([(sid, rmgId) for (sid, (fromDt, thruDt, rmgId))
                            in connections.sidRanges.items()])
        self.tradingRMGs = connections.tradingRMGs
        # tradingRMGs contains the trading day information for the whole
        # interval that contains all days we will process. Use it to
        # build a map from (date, rmg_id) to the previous trading day
        self.prevTradingDay = dict()
        tradingDays = set()
        allRMGs = set()
        for (date, rmgs) in self.tradingRMGs.items():
            tradingDays |= set([(date, rmg) for rmg in rmgs])
            allRMGs |= rmgs
        for rmg in allRMGs:
            myTradingDays = sorted(date for (date, trmg) in tradingDays
                                   if rmg == trmg)
            if len(myTradingDays) == 0:
                continue
            # Add in trading day prior to first one
            rmgObj = Utilities.Struct()
            rmgObj.rmg_id = rmg
            prevDay = self.modelDB.getPreviousTradingDay(
                rmgObj, myTradingDays[0])
            myTradingDays.insert(0, prevDay)
            prevTradingDay = dict(zip(myTradingDays[1:], myTradingDays[0:-1]))
            self.prevTradingDay.update(dict(
                [((date, rmg), prevTradingDay[date]) for date
                 in myTradingDays if date in prevTradingDay]))
        self.sidRanges = connections.sidRanges
        
    def computeCashReturn(self, rmg, cashSid, date, prevDate):
        """Compute the return of the cash asset on the given date.
        """
        if prevDate is None:
            self.log.error(
                'No previous trading day for %s on %s. Skipping',
                cashSid.getSubIDString(), date)
            return None
        isoCode = cashSid.getCashCurrency()
        numDays = (date - prevDate).days
        # get the currency risk free rate of the previous trading day
        rate = self.modelDB.getRiskFreeRateHistory([isoCode], [prevDate], 
                        self.marketDB, annualize=True).data[0,0]
        #print rate, date, prevDate, isoCode, numDays
        if rate is ma.masked:
            self.log.warning('No risk free rate on previous trading day'
                         ' in cash return computation: %s, %s'
                         % (isoCode, prevDate))
            return None
        else:
            # previous trading days interest rate converted to daily
            # by dividing by 360 and converted to period return by
            # multiplying by the number calendar
            # days from the last trading day.
            cashReturn = (rate / 360) * numDays
        return cashReturn
        
    def getBulkData(self, dateList, sidList):
        self.log.debug('CashReturn.getBulkData')
        retvals = numpy.empty((len(dateList), len(sidList)), dtype=object)
        cashIDs = [(idx, sid) for (idx, sid) in enumerate(sidList)
                   if sid.isCashAsset()]
        for (idx, sid) in cashIDs:
            sidRMG = self.sidRMGs[sid]
            sidRange = self.sidRanges[sid]
            for (dIdx, date) in enumerate(dateList):
                # if sidRMG not in self.tradingRMGs[date] \
                #         or date < sidRange[0] or date >= sidRange[1]:
                #     # Skip non-trading days
                #     continue
                # if sidRMG in (1,2,3):
                #     prevDate = self.prevTradingDay[(date, sidRMG)]
                # else:
                prevDate = date - datetime.timedelta(days=1)
                cashReturn = self.computeCashReturn(sidRMG, sid, date,
                                                    prevDate)
                if cashReturn is not None:
                    rval = Utilities.Struct()
                    rval.tr = cashReturn
                    retvals[dIdx,idx] = rval
        return retvals

class ModelDBReturn(MarketDBSource):
    """Class to compute the sub-issue returns.
    """
    mapTable = 'issue_map'
    
    def __init__(self, connections):
        MarketDBSource.__init__(self, connections)
        self.sidRMGs = dict((sid, rmgId) for (sid, (fromDt, thruDt, rmgId))
                            in connections.sidRanges.items())
        self.lookBackDays = 30
        # get currency converters
        self.currencyProvider = MarketDB.CurrencyProvider(self.marketDB, self.lookBackDays, None)
        self.idSidMap = ModelDB.FromThruCache(useModelID=True)
        
    def getBulkData(self, dateList, sidList):
        self.log.debug('ModelDBReturn.getBulkData')
        retvals = numpy.empty((len(dateList), len(sidList)), dtype=object)
        sidIdx = dict((sid, idx) for (idx, sid) in enumerate(sidList))
        rmgs = set()
        rmgSids = dict()
        for sid in sidList:
            rmg = self.sidRMGs[sid]
            rmgs.add(rmg)
            if rmg not in rmgSids:
                rmgSids[rmg] = list()
            rmgSids[rmg].append(sid)
        rmgs = [self.modelDB.getRiskModelGroup(rmgID) for rmgID in rmgs]
        # Get trading days per country for the given date range plus lookback
        minDt = min(dateList)
        maxDt = max(dateList)
        rmgTrading = dict()
        allTradingDays = set()
        cur = self.modelDB.dbCursor
        for rmg in rmgs:
            cur.execute("""SELECT dt FROM rmg_calendar c1
              WHERE rmg_id=:rmg AND dt <= :maxDt
              AND sequence >= (SELECT NVL(MIN(sequence), 0) FROM rmg_calendar c2
                WHERE c1.rmg_id=c2.rmg_id
                AND dt BETWEEN (SELECT MAX(dt)
                  FROM rmg_calendar c3
                  WHERE c3.rmg_id=c1.rmg_id AND dt <= :minDt)
                AND :maxDt) - :lookBack""",
                        rmg=rmg.rmg_id, minDt=minDt, maxDt=maxDt,
                        lookBack=self.lookBackDays)
            rmgTrading[rmg] = set(d[0].date() for d in cur.fetchall())
            allTradingDays |= rmgTrading[rmg]
        if len(allTradingDays) == 0:
            self.log.info('No trading days for assets in countries %s'
                          ' in given range.', ','.join([rmg.mnemonic for rmg in rmgs]))
            return retvals
        allDays = [min(allTradingDays) + datetime.timedelta(i) for i in
                    range((maxDt - min(allTradingDays)).days + 1)]
        allDaysIdxMap = dict((j,i) for (i,j) in enumerate(allDays))
        # get corporate actions
        issues = [s.getModelID() for s in sidList]
        marketIDs = self.modelDB.loadFromThruTable(
            self.mapTable, allDays, issues,
            ['marketdb_id'], keyName='modeldb_id',
            cache=self.modelDBHelper.idMapCache)
        modelMarketDtMap = dict() # Maps (issue ID, dt) pairs to market IDs
        allMarketIDs = set()
        for (dIdx, date) in enumerate(allDays):
            for (iIdx, issue) in enumerate(issues):
                if marketIDs[dIdx, iIdx] is not None:
                    axid = marketIDs[dIdx, iIdx].marketdb_id
                    modelMarketDtMap[(issue, date)] = axid
                    allMarketIDs.add(axid)
        allMarketIDs = sorted(allMarketIDs)
        
        self.splitsAndDivs = self.buildSplitAndDividendMatrix(
            allMarketIDs, allDays)
        self.mergersAndSpinOffs = self.buildMergerAndSpinOffMatrix(
            issues, rmgs, allDays)
        
        # Get all additional sub-issue IDs involved in corporate actions
        corpActionSIDs  = self.getCorpActionSubIssues(allDays, allDaysIdxMap)
        newSIDs = list(corpActionSIDs - set(sidList))
        # Load prices for all required days and assets
        allSIDs = sidList + newSIDs
        prices = self.modelDB.loadRawUCPHistory(allDays, allSIDs)
        allSidIdxMap = dict((j,i) for (i,j) in enumerate(allSIDs))
        priceMaps = dict()
        for day in allDays:
            dIdx = allDaysIdxMap[day]
            priceMaps[day] = dict()
            dayMap = priceMaps[day]
            for (sid, sIdx) in allSidIdxMap.items():
                price = prices[sIdx, dIdx]
                if price is not None:
                    dayMap[sid.getModelID()] = price
        for (dIdx, date) in enumerate(dateList):
            for rmg in rmgs:
                if date not in rmgTrading[rmg]:
                    self.log.debug('%s is not a trading day for %s. Skipping return calculation',
                                   date, rmg.description)
                    continue
                mySids = rmgSids[rmg.rmg_id]
                vals = self.getDataByDateAndRMG(
                    date, mySids, rmgTrading[rmg], rmg,
                    priceMaps, modelMarketDtMap)
                for (sid, val) in zip(mySids, vals):
                    retvals[dIdx, sidIdx[sid]] = val
        return retvals
    
    def getCorpActionSubIssues(self, allDays, allDaysIdxMap):
        corpActionSIDs = set()
        if len(self.mergersAndSpinOffs) > 0:
            corpActionModelIDs = set()
            for caList in self.mergersAndSpinOffs.values():
                for ca in caList:
                    corpActionModelIDs.update(ca.getInvolvedModelIDs())
            corpActionModelIDs = list(corpActionModelIDs)
            issueSIDMatrix = self.modelDB.loadFromThruTable(
                'sub_issue', allDays, corpActionModelIDs,
                ['sub_id'], keyName='issue_id', cache=self.idSidMap)
            midIdxMap = dict((j,i) for (i,j) in enumerate(corpActionModelIDs))
            for ((mid, d), caList) in self.mergersAndSpinOffs.items():
                dIdx = allDaysIdxMap[d]
                for ca in caList:
                    for cMid in ca.getInvolvedModelIDs():
                        mIdx = midIdxMap[cMid]
                        val = issueSIDMatrix[dIdx, mIdx]
                        if val is not None:
                            corpActionSIDs.add(ModelDB.SubIssue(val.sub_id))
        return corpActionSIDs
    
    def getDataByDateAndRMG(self, date, subIssues, tradingDays, rmg, priceMaps,
                            modelMarketDtMap):
        """Compute the total return for each asset for which we have
        a price on the given day.
        The return is computed between now and the last day for which
        and asset has a price in the last five trading days.
        If no price exists in the last five days, no return is reported.
        """
        retvals = numpy.empty((len(subIssues,)), dtype=object)
        tradingDaysList = sorted([d for d in tradingDays if d <= date])
        if len(tradingDaysList) < 2:
            self.log.info('No previous trading day.'
                          ' Skipping return calculation')
            return retvals
        # get price history
        tradingDaysList = tradingDaysList[-(self.lookBackDays+1):]
        currentPrices = priceMaps[date]
        self.log.info('%d assets for risk model group %s on %s',
                      len(subIssues), rmg.description, date)
        dateList = [tradingDaysList[0] + datetime.timedelta(i) for i in
                    range((tradingDaysList[-1] - tradingDaysList[0]).days + 1)]
        dateList.reverse()
        for (idx, sid) in enumerate(subIssues):
            issue = sid.getModelID()
            marketIssue = modelMarketDtMap.get((issue, date))
            if issue not in currentPrices or marketIssue is None:
                continue
            try:
                # XXX Need to handle return computation crossing currency
                # changes, like Turkish Lira to Turkish New Lira
                curPrice = currentPrices[issue]
                for day in range(len(tradingDaysList)-1):
                    oldDate = tradingDaysList[-2-day]
                    oldPrice = priceMaps[oldDate].get(issue)
                    if oldPrice is not None:
                        oldDate = tradingDaysList[-2-day]
                        days = dateList[0:dateList.index(oldDate)]
                        oldConv = self.currencyProvider.getCurrencyConverter(
                            oldDate)
                        ret = None
                        try:
                            oldRate = oldConv.getRate(
                                oldPrice.currency_id, curPrice.currency_id)
                            myOldPrice = oldPrice.ucp * oldRate
                            ret = self.computeReturn(
                                issue, marketIssue, myOldPrice, curPrice.ucp,
                                oldDate, days, curPrice.currency_id, priceMaps)
                        except KeyError:
                            oldCode = self.marketDB.getCurrencyISOCode(
                                oldPrice.currency_id, oldDate)
                            curCode = self.marketDB.getCurrencyISOCode(
                                curPrice.currency_id, date)
                            self.log.error("Can't convert currency %d/%s to %d/%s"
                                           " on %s for %s",
                                           oldPrice.currency_id, oldCode,
                                           curPrice.currency_id, curCode,
                                           oldDate, sid)
                        if ret is not None:
                            rval = Utilities.Struct()
                            rval.tr = ret
                            retvals[idx] = rval
                        break
            except KeyError:
                self.log.error('Error during return computation for'
                               ' %s on %s', sid.getModelID(), date, exc_info=True)
        return retvals
        
    def buildSplitAndDividendMatrix(self, marketIssues, dateList):
        """Build a dictionary that maps (MarketDB ID, date)
        -> list of split and dividend actions.
        """
        caMap = dict()
        for d in dateList:
            splits = self.marketDB.getStockSplits(d)
            splitMap = dict()
            for s in splits:
                splitMap.setdefault(s.asset, list()).append(s)
            dividends = self.marketDB.getCashDividends(d)
            divMap = dict()
            for s in dividends:
                divMap.setdefault(s.asset, list()).append(s)
            for i in marketIssues:
                i = MarketID.MarketID(string=i)
                key = (i,d)
                if i in splitMap:
                    caMap.setdefault(key, list()).extend(splitMap[i])
                if i in divMap:
                    caMap.setdefault(key, list()).extend(divMap[i])
        self.log.debug('%d splits/dividends' % len(caMap))
        return caMap
    
    def buildMergerAndSpinOffMatrix(self, issues, rmgs, dateList):
        """Build a dictionary that maps (ModelID, date) -> list of mergers
        and spin-off actions.
        """
        caMap = dict()
        for d in dateList:
            mergerMap = dict()
            mergers = self.modelDB.getCAMergerSurvivors(d)
            for i in mergers:
                mergerMap.setdefault(i.modeldb_id, list()).append(i)
            spinoffMap = dict()
            spinoffs = self.modelDB.getCASpinOffs(d)
            for i in spinoffs:
                spinoffMap.setdefault(i.modeldb_id, list()).append(i)
            for i in issues:
                key = (i,d)
                if i in mergerMap:
                    caMap.setdefault(key, list()).extend(mergerMap[i])
                if i in spinoffMap:
                    caMap.setdefault(key, list()).extend(spinoffMap[i])
        self.log.debug('%d mergers/spin-offs' % len(caMap))
        return caMap
    
    def computeReturn(self, issue, marketIssue, oldPrice, newPrice, oldDate,
                      days, baseCurrency, priceMaps):
        curPrice = newPrice
        hasActions = False
        for d in days:
            currencyConverter = self.currencyProvider.getCurrencyConverter(d)
            actions = self.splitsAndDivs.get((MarketID.MarketID(string=marketIssue), d), []) + self.mergersAndSpinOffs.get((issue, d), [])
            
            tmp = sorted((a.sequenceNumber, a) for a in actions)
            seqSet = dict(tmp)
            if len(seqSet) != len(tmp):
                self.log.error('Sequence numbers are not unique: asset %s/%s,'
                              ' date %s', issue, marketIssue, str(d))
                self.log.error(','.join([str(i[1]) for i in tmp]))
            actions = [i[1] for i in tmp]
            if len(actions) > 0:
                hasActions = True
                self.log.info('Corporate actions for %s on %s: %s',
                              issue, d, ','.join([str(a) for a in actions]))
            while len(actions) > 0:
                c = actions.pop()
                try:
                    (oldMarketIssue, curPrice) = c.backwardAdjustPrice(
                        marketIssue, curPrice, currency=baseCurrency,
                        currencyConverters=[currencyConverter],
                        priceMap=priceMaps.get(d, dict()))
                except:
                    self.log.error("Can't apply corporate action for %s"
                                   " on %s: %s", issue, d, c, exc_info=True)
                    return None
                if oldMarketIssue != marketIssue:
                    # Rebuild actions with update marketIssue
                    marketIssue = oldMarketIssue
                    actions = self.splitsAndDivs.get((MarketID.MarketID(string=marketIssue), d), []) \
                              + self.mergersAndSpinOffs.get((issue, d), [])
                    tmp = sorted((a.sequenceNumber, a) for a in actions)
                    seqSet = dict(tmp)
                    if len(seqSet) != len(tmp):
                        self.log.error('Sequence numbers are not unique:'
                                       ' asset %s/%s, date %s',
                                       issue, marketIssue, str(d))
                        self.log.error(','.join([str(i[1]) for i in tmp]))
                    actions = [i[1] for i in tmp]
                    actions = actions[0:actions.index(c)]
                    self.log.info('Corporate actions for %s on %s: %s',
                                  issue, d, ','.join([str(a) for a in actions]))
        ret = (curPrice - oldPrice) / oldPrice
        if hasActions or (oldPrice >= 1.0 and (ret > 0.8 or ret < -0.4)):
            if oldPrice >= 1.0 and (ret > 0.8 or ret < -0.4):
                self.log.warning(
                    'Suspicious return for %s/%s on %s: %g'
                    ', prev. price %g, curr. price %g, adjusted price %g',
                    issue, marketIssue, days[0], ret, oldPrice,
                    newPrice, curPrice)
            else:
                self.log.info(
                    'Return for asset %s/%s on %s with corp. actions: %g'
                    ', prev. price %g, curr. price %g, adjusted price %g',
                    issue, marketIssue, days[0], ret, oldPrice,
                    newPrice, curPrice)
        return ret
    
    def getMarketIssues(self, dateList, issues):
        """Get the MarketDB IDs corresponding to the issues on the given days.
        """
        marketIDs = self.modelDB.loadFromThruTable(
            self.mapTable, dateList, issues,
            ['marketdb_id'], keyName='modeldb_id',
            cache=self.modelDBHelper.idMapCache)
        mktStrs = set([i.marketdb_id for i in  marketIDs.flat
                       if i is not None])
        return list(mktStrs)
    
class FutureModelDBReturn(ModelDBReturn):
    mapTable = 'future_issue_map'

class CumulativeReturn(ModelDBSource):
    """Class to compute the cumulative return.
    """
    def __init__(self, connections):
        ModelDBSource.__init__(self, connections)
        self.sidRanges = connections.sidRanges
        self.INCR = 500
        self.DATEINCR = 1
        self.argList = [('sid%d' % i) for i in range(self.INCR)]
        self.dateArgList  = [('date%d' % i) for i in range(self.DATEINCR)]
        self.prevCumRetQuery = """SELECT sub_issue_id, dt, value
          FROM sub_issue_cum_return_active cr
          WHERE cr.sub_issue_id IN (%(sids)s)
          AND dt IN (%(dates)s)""" % {
            'sids': ','.join([':%s' % i for i in self.argList]),
            'dates': ','.join([':%s' % i for i in self.dateArgList])}
        self.defaultDict = dict([(arg, None) for arg
                                 in self.argList + self.dateArgList])
        self.oneDay = datetime.timedelta(days=1)
    
    def getBulkData(self, dateList, sidList):
        self.log.debug('CumulativeReturn.getBulkData')
        retvals = numpy.empty((len(dateList), len(sidList)), dtype=object)
        cur = self.modelDB.dbCursor
        sidStrings = [sid.getSubIDString() for sid in sidList]
        cumReturns = Matrices.allMasked((len(dateList), len(sidList)))
        prevDateList = [d - self.oneDay for d in dateList]
        prevDateIdx = dict((d,i) for (i,d) in enumerate(prevDateList))
        dateIdx = dict((d,i) for (i,d) in enumerate(dateList))
        sidStrIdx = dict((s,i) for (i,s) in enumerate(sidStrings))
        # Get cumulative return values of previous day
        for dateChunk in listChunkIterator(prevDateList, self.DATEINCR):
            updateDateDict = dict(zip(self.dateArgList, dateChunk))
            for sidChunk in listChunkIterator(sidStrings, self.INCR):
                updateSidDict = dict(zip(self.argList, sidChunk))
                myDict = self.defaultDict.copy()
                myDict.update(updateDateDict)
                myDict.update(updateSidDict)
                cur.execute(self.prevCumRetQuery, myDict)
                for (sid, dt, cumRet) in cur.fetchall():
                    dIdx = prevDateIdx[dt.date()]
                    sIdx = sidStrIdx[sid]
                    if cumRet is not None:
                        cumReturns[dIdx, sIdx] = cumRet
        cumReturns = cumReturns.filled(1.0)
        # Get returns for required days
        totReturns = self.modelDB.loadSubIssueData(
            dateList, sidList, 'sub_issue_return', 'tr',
            cache=None, withCurrency=False)
        totReturns = totReturns.data.filled(0.0)
        for (sIdx, sid) in enumerate(sidList):
            for (dIdx, date) in enumerate(dateList):
                totRet = totReturns[sIdx, dIdx]
                if self.sidRanges[sid][0] == date:
                    # sub-issue starts on this day, use active value if present
                    activeCumRet = self.modelDB.loadCumulativeReturnsHistory(
                        [date], [sid]).data.filled(1.0)
                    myCumRet = activeCumRet[0,0]
                else:
                    prevCumRet = cumReturns[dIdx,sIdx]
                    myCumRet = prevCumRet * (1.0 + totRet)
                nextDay = date + self.oneDay
                nextDIdx = dateIdx.get(nextDay)
                if nextDIdx is not None:
                    cumReturns[nextDIdx, sIdx] = myCumRet
                rval = Utilities.Struct()
                rval.value = myCumRet
                rval.tr = totRet
                retvals[dIdx, sIdx] = rval
        return retvals

class RMGHistoricBetaV3(ModelDBSource):
    """Class to compute the historic beta for risk model groups.
    """
    def __init__(self, connections, gp=None):
        ModelDBSource.__init__(self, connections)
        self.marketDB = connections.marketDB
        self.tradingRMGs = gp.tradingRMGs
        self.override = hasattr(gp, 'override') and gp.override
        self.cleanup = hasattr(gp, 'cleanup') and gp.cleanup
        self.nuke = hasattr(gp, 'nuke') and gp.nuke
        self.debugOutput = hasattr(gp, 'verbose') and gp.verbose
        self.expand = hasattr(gp, 'expand') and gp.expand
        self.notInRiskModels = hasattr(gp, 'notInRiskModels') and gp.notInRiskModels
        self.modelDB.setMarketCapCache(60)
        clsFamily = self.marketDB.getClassificationFamily('REGIONS')
        assert(clsFamily is not None)
        clsMembers = dict([(i.name, i) for i in self.marketDB.\
                            getClassificationFamilyMembers(clsFamily)])
        clsMember = clsMembers.get('HomeCountry', None)
        assert(clsMember is not None)
        self.homeCountryCls = clsMember
        rmgs = self.modelDB.getAllRiskModelGroups(inModels=False)
        self.rmgMap = dict((rmg.rmg_id, rmg) for rmg in rmgs)
        rmgs = self.modelDB.getAllRiskModelGroups(inModels=True)
        validRMGQuery = """
            SELECT a.mnemonic, MIN(a.from_dt) FROM (
                SELECT g.mnemonic, GREATEST(m.from_dt, s.from_dt) AS from_dt
                FROM risk_model_group g, rmg_model_map m, risk_model_serie s
                WHERE g.rmg_id > 0 AND g.rmg_id = m.rmg_id 
                AND m.rms_id = s.serial_id
                AND s.serial_id > 0 AND s.distribute = 1) a
            GROUP BY a.mnemonic"""
        self.modelDB.dbCursor.execute(validRMGQuery)
        self.modelDB.createCurrencyCache(self.marketDB)
        self.allowedRMGData = dict([(r[0], r[1].date()) \
                for r in self.modelDB.dbCursor.fetchall()])
    
    def getBulkData(self, dateList, rmgIdList):
        self.log.debug('RMGHistoricBeta.getBulkData')
        retvals = numpy.empty((len(dateList), len(rmgIdList)), dtype=object)

        # If doing a grand cleanup of everything, do it once, then reset the flag
        if self.nuke:
            check = input('About to delete all records. Are you sure? ').lower()
            if check == 'yes' or check == 'y':
                self.nuke = False
                self.modelDB.dbCursor.execute("""SELECT dt FROM rmg_historic_beta_v3""")
                ret = self.modelDB.dbCursor.fetchall()
                nukeDates = sorted(set(r[0] for r in ret))
                for dt in nukeDates:
                    logging.info('Deleting all records from %s', dt)
                    self.modelDB.dbCursor.execute(
                    """DELETE FROM rmg_historic_beta_v3
                    WHERE dt=:dt_arg""", dt_arg=dt)
                self.modelDB.commitChanges()

        for (rIdx, rmg) in enumerate(rmgIdList):
            rmgObj = self.rmgMap[rmg]
            for (dIdx, date) in enumerate(dateList):
                if not rmgObj.setRMGInfoForDate(date):
                    raise Exception('Cannot determine currency for %s risk model group (%d) on %s'\
                            % (rmgObj.description, rmgObj.rmg_id, str(date)))
                rval = Utilities.Struct()
                rval.valuePairs = self.computeRMGHistoricBetas(rmgObj, date, debugOutput=self.debugOutput)
                if rval.valuePairs is not None:
                    retvals[dIdx, rIdx] = rval
        return retvals
    
    def getAssetMktInfo(self, date, subIssues):
        # Get home market classifications for list of assets
        # Not sure whether this override flag is still needed
        if self.override:
            allowedRMGCodes = [rmgCode for (rmgCode, dt) in self.allowedRMGData.items()]
        else:
            allowedRMGCodes = [rmgCode for (rmgCode, dt) in self.allowedRMGData.items() if date >= dt]

        # Load market classification dict
        clsRevision = self.marketDB.getClassificationMemberRevision(self.homeCountryCls, date)
        assetMarketDict =  self.modelDB.getMktAssetClassifications(clsRevision, subIssues, date, self.marketDB)

        # Load trading market data
        rmgs = self.modelDB.getAllRiskModelGroups(inModels=False)
        tradingRMGMap = Utilities.flip_dict_of_lists(AssetProcessor_V4.loadRiskModelGroupAssetMap(\
                date, subIssues, self.modelDB, rmgs, quiet=True))

        return assetMarketDict, tradingRMGMap, allowedRMGCodes

    def getRMGExpandedUniverse(self, rmg, currDate):
        """Returns a tuple containing two lists of non-cash sub-issues.
        The first is all the assets trading in the given risk model
        group, and the second contains the first, plus all sub-issues 
        mapped to the risk model group as its home country 1, minus those 
        in the risk model group but assigned to a foreign home country.
        So, the first list is based on country of quotation while the
        second is based on home country.
        """
        # Load all sub-issues trading on the RMG
        rmgTradingSubIssues = self.modelDB.getActiveSubIssues(rmg, currDate)

        # Get home market information
        allSubIssues = self.modelDB.getAllActiveSubIssues(currDate)
        self.assetTypeDict = AssetProcessor_V4.get_asset_info(currDate, allSubIssues,
                self.modelDB, self.marketDB, 'ASSET TYPES', 'Axioma Asset Type')
        allSubIssues = [sid for sid in allSubIssues if self.assetTypeDict.get(sid, None) not in AssetProcessor_V4.noHBetaList]
        (assetMarketDict, tradingRMGMap, allowedRMGCodes) = self.getAssetMktInfo(currDate, allSubIssues)

        # Remove undesirables
        rmgTradingSubIssues = set([sid for sid in rmgTradingSubIssues \
                if self.assetTypeDict.get(sid, None) not in AssetProcessor_V4.noHBetaList])

        # Subset of assets traded elsewhere, but with home country set as current RMG
        listedAbroad = [sid for (sid,cls) in assetMarketDict.items() \
                        if sid not in rmgTradingSubIssues and cls.classification.code==rmg.mnemonic]

        # Report on these assets traded elsewhere
        self.log.debug('Adding %d %s assets trading outside of %s', 
                    len(listedAbroad), rmg.description, rmg.mnemonic)
        self.log.info('%d RiskModelGroups for valid/covered home country assignment',
                    len(allowedRMGCodes))

        # Get the list of assets traded on RMG but with home country elsewhere
        foreign = [sid for sid in rmgTradingSubIssues.intersection(list(assetMarketDict.keys())) \
                   if assetMarketDict[sid].classification.code != rmg.mnemonic \
                   and assetMarketDict[sid].classification.code in allowedRMGCodes]
        self.log.info('Removing %d %s assets with non-%s home country',
                    len(foreign), rmg.description, rmg.mnemonic)

        # Get the set of assets traded in the RMG minus cash assets
        expandedUniv = set([s for s in rmgTradingSubIssues if not s.isCashAsset()])
        # Add those assets listed abroad but with home country as RMG
        expandedUniv.update(listedAbroad)
        # Remove those with a different home country
        expandedUniv.difference_update(foreign)
        # Return the two sets: trading assets, home country RMG assets
        return (rmgTradingSubIssues, expandedUniv, assetMarketDict, tradingRMGMap)
    
    def computeRMGHistoricBetas(self, rmg, currDate, updateList=[], debugOutput=False):
        # Do some checking of dates and subissues
        if self.notInRiskModels:
            if rmg.rmg_id not in self.rmgMap.keys():
                self.log.info("RMG %s not recognised", rmg.mnemonic)
                return None
            if rmg.mnemonic not in self.allowedRMGData:
                # If not yet in a model, but forcing it to run, create a pseudo from-date
                self.allowedRMGData[rmg.mnemonic] = currDate - datetime.timedelta(365)
        if rmg.mnemonic not in self.allowedRMGData:
            self.log.info("Skipping %s since it is in no model", rmg.mnemonic)
            return None
        if currDate < self.allowedRMGData[rmg.mnemonic] and not self.override:
            self.log.info("Skipping %s since it doesn't join a model until %s", rmg.mnemonic,
                          self.allowedRMGData[rmg.mnemonic])
            return None
        self.log.info('Processing historic betas (V3) for %s on %s',
                      rmg.description, currDate)
        if currDate.isoweekday() > 5:
            self.log.info('Saturday/Sunday, skipping.')
            return None

        # Get lists of assets - those trading on the RMG market,
        # and those with RMG market set as their home, wherever they trade
        (rmgTradingSubIssues, rmgHomeSubIssues, fullAssetMarketDict, fullTradingRMGMap) = \
                self.getRMGExpandedUniverse(rmg, currDate)

        if len(updateList) > 0:
            # If we're doing secondary run of RMG traded assets with
            # a different home, then nuke the trading RMG list
            rmgTradingSubIssues = set()
        rmgHomeSubIssues = list(rmgHomeSubIssues)
        rmgTradingSubIssues = list(rmgTradingSubIssues)

        if len(rmgHomeSubIssues) == 0:
            self.log.info('No active sub-issues, skipping.')
            return None

        prevDate = self.modelDB.getDates([rmg], currDate, 1)
        if len(prevDate) < 1:
            self.log.info('No previous date for %s on %s', rmg.description, currDate)
            prevDate = currDate
        else:
            prevDate = prevDate[0]

        # For China, compute betas for A-Shares versus domestic market,
        # and use investible China market portfolio for all other stocks
        if rmg.description == 'China':
            # Divide Chinese stocks into their various classes
            otherSubIssues = [sid for sid in rmgHomeSubIssues if self.assetTypeDict.get(sid, None) not in ['AShares', 'BShares']]
            domesticSubIssues = [sid for sid in rmgHomeSubIssues if self.assetTypeDict.get(sid, None) in ['AShares', 'BShares']]
             
            # Compute betas for International China
            (ret_val, nMissMap) = self.computeRMGHistoricBetasInner(
                    rmg, otherSubIssues, currDate, prevDate, home=True, updateList=updateList)
             
            # Compute betas for Domestic China
            domesticChinaRMG = self.modelDB.getRiskModelGroupByISO('XC')
            (dc_ret_val, dc_nMissMap) = self.computeRMGHistoricBetasInner(
                    domesticChinaRMG, domesticSubIssues, currDate, prevDate,
                    home=True, returnsRMG=rmg, updateList=updateList)
            ret_val.extend(dc_ret_val)
            nMissMap = pandas.concat([nMissMap, dc_nMissMap], axis=0)
        else:
            # Other markets are straightforward
            (ret_val, nMissMap) = self.computeRMGHistoricBetasInner(
                            rmg, rmgHomeSubIssues, currDate, prevDate, home=True, updateList=updateList)

        # Now do trading beta for non-local assets traded on the current market
        mktList = [rmg.mnemonic] * len(ret_val)
        tmpDict=dict( [(r,1) for r in rmgHomeSubIssues])
        foreignIssues = [sid for sid in rmgTradingSubIssues if sid not in tmpDict]
        if len(foreignIssues) > 0:
            logging.info('Found subissues for %d non-local %s-traded assets',
                    len(foreignIssues), rmg.description)
            (ret_val1, nMissMap1) = self.computeRMGHistoricBetasInner(
                            rmg, foreignIssues, currDate, prevDate, home=False)
            ret_val.extend(ret_val1)
            nMissMap = pandas.concat([nMissMap, nMissMap1], axis=0)
            mktList.extend([rmg.mnemonic] * len(ret_val1))

            # Compute betas for trading RMG with alternative home market
            # Get market classification data
            (assetMarketDict, tradingRMGMap, allowedRMGCodes) = self.getAssetMktInfo(currDate, foreignIssues)
            rmgAll = self.modelDB.getAllRiskModelGroups(inModels=False)
            # Get list of other home RMGs
            otherRMGCodes = [cls.classification.code for cls in assetMarketDict.values()\
                    if cls.classification.code in allowedRMGCodes]
            otherRMGs = [r for r in rmgAll if r.mnemonic in otherRMGCodes]

            if self.expand and (len(otherRMGs) > 0):
                logging.info('%d RMGs have home assets traded on %s', len(otherRMGs), rmg.mnemonic)
                # Loop round other home markets - compute betas for subset of assets traded on RMG
                for r in otherRMGs:
                    otherRMGSubIssues = [sid for sid in assetMarketDict.keys() \
                            if assetMarketDict[sid].classification.code == r.mnemonic]
                    if len(otherRMGSubIssues) > 0:
                        # Warning - recursion here
                        logging.info('Computing betas for %d assets with %s home traded on %s',
                                len(otherRMGSubIssues), r.mnemonic, rmg.mnemonic)
                        other_ret_val = self.computeRMGHistoricBetas(r, currDate,
                                updateList=list(otherRMGSubIssues))
                        ret_val.extend(other_ret_val)
                        mktList.extend([r.mnemonic] * len(other_ret_val))

        if debugOutput:
            outArray = Matrices.allMasked((len(ret_val), 5))
            colNames = [',Home', 'Beta', 'P-Value', 'n-Rets', 'Missing']
            sidList = []
            idx = 0
            universe = [sid for (sid, home, beta, pval, nRets) in ret_val]
            exSpac = AssetProcessor_V4.sort_spac_assets(currDate, universe, self.modelDB, self.marketDB, returnExSpac=True)

            for (sid, home, beta, pval, nRets) in ret_val:
                outArray[idx, 0] = home
                outArray[idx, 1] = beta
                outArray[idx, 2] = pval
                outArray[idx, 3] = nRets
                if sid in nMissMap:
                    outArray[idx, 4] = nMissMap[sid]
                if sid in fullAssetMarketDict:
                    hMarket = fullAssetMarketDict[sid].classification.code
                else:
                    hMarket = mktList[idx]
                if sid in exSpac:
                    exSp = 1
                else:
                    exSp = 0
                sidList.append('Type:%s|Trade:%s|Home:%s|ExSPAC:%s|%s' % \
                        (self.assetTypeDict[sid], fullTradingRMGMap[sid].mnemonic, hMarket, exSp, sid.getSubIDString()))
                idx += 1
            outfile = 'tmp/hbeta-%s-%s.csv' % (rmg.mnemonic, currDate)
            Utilities.writeToCSV(outArray, outfile, rowNames=sidList, columnNames=colNames)

        return ret_val

    def computeRMGHistoricBetasInner(
            self, rmg, rmgHomeSubIssues, currDate, prevDate, home=True, returnsRMG=None, updateList=[]):
        """Inner processing loop for historic betas
        """
        # Processing of assets to be updated
        if len(rmgHomeSubIssues) < 1:
            return [], pandas.Series([])
        if len(updateList) < 1:
            updateList = sorted(set(rmgHomeSubIssues))
        else:
            updateList = sorted(set(updateList).intersection(set(rmgHomeSubIssues)))
        logging.info('Computing betas for %d %s assets', len(updateList), rmg.description)
        if len(updateList) < 1:
            return [], pandas.Series([])

        if returnsRMG is None:
            returnsRMG = rmg

        # Delete existing records
        if self.cleanup:
            deleteDicts = [dict([
                ('sub_issue_id', sid.getSubIDString()),
                ('dt', currDate)
                ]) for sid in rmgHomeSubIssues]
            self.modelDB.dbCursor.executemany("""DELETE FROM rmg_historic_beta_v3
                    WHERE sub_issue_id=:sub_issue_id AND dt=:dt""", deleteDicts)
            self.log.info('Deleting %d records', len(deleteDicts))

        # Load previous betas if they exist
        prev_betas_old = self.modelDB.getHistoricBetaDataV3(
                currDate, rmgHomeSubIssues, field='value', home=home, rollOverData=True)
        prev_betas_pvl = self.modelDB.getHistoricBetaDataV3(
                currDate, rmgHomeSubIssues, field='p_value', home=home, rollOverData=True)
        prev_betas_obs = self.modelDB.getHistoricBetaDataV3(
                currDate, rmgHomeSubIssues, field='nobs', home=home, rollOverData=True)

        # Get list of dates needed
        twoYearWeeks = 104
        twoYearsDays = (twoYearWeeks + 2) * 7
        startDate = currDate - (twoYearsDays * datetime.timedelta(1))
        tradingDaysList = self.modelDB.getDateRange(None, startDate, currDate, excludeWeekend=True)
        logging.debug('%d weekdays, spanning %s to %s',
                len(tradingDaysList), tradingDaysList[0], tradingDaysList[-1])

        # Only recompute betas at the start of each new period
        # In this case, on Mondays (or first trading day of the week)
        if prevDate.weekday() < currDate.weekday() and len(prev_betas_old) > 0 \
                and (currDate - prevDate < datetime.timedelta(7)) \
                and currDate != self.allowedRMGData[returnsRMG.mnemonic]:
            # If not start of new period, roll over last period's values
            logging.info('Rolling previous values from %s for %s', prevDate, currDate)
            return [], pandas.Series([])

        # Get the weekly dates (Mon-Mon, Tue-Tue etc.)
        period_dates = Utilities.change_date_frequency(tradingDaysList, frequency='weekly')
        logging.debug('%d end of week dates, spanning %s to %s',
                len(period_dates), period_dates[0], period_dates[-1])
            
        # Retrieve asset returns from the DB
        currencyCode = returnsRMG.getCurrencyCode(currDate)
        currencyID = self.modelDB.currencyCache.getCurrencyID(currencyCode, currDate)
        returns = self.modelDB.loadTotalReturnsHistoryV3(
                [returnsRMG], currDate, rmgHomeSubIssues, len(tradingDaysList), dateList=tradingDaysList,
                assetConvMap=currencyID, excludeWeekend=False, compound=False)
        if returns.data is None:
            return [], pandas.Series([])

        # Save proportion of missing returns (not pre-IPO)
        returnsDates = returns.dates
        notTradedFlag = pandas.DataFrame(returns.notTradedFlag, index=returns.assets, columns=returnsDates)
        nMissingReturns = notTradedFlag.sum(axis=1)
        nMissMap = nMissingReturns / float(nMissingReturns.max(axis=None))
        returns = pandas.DataFrame(returns.data, index=returns.assets, columns=returnsDates)

        # Find pre-IPO dates (taking account of SPACs)
        issueFromDates = Utilities.load_ipo_dates(
                currDate, rmgHomeSubIssues, self.modelDB, self.marketDB, exSpacAdjust=True, returnList=True)
        tmpReturns = Matrices.fillAndMaskByDate(returns, issueFromDates, returnsDates)
        preIPOFlag = pandas.isnull(tmpReturns)

        # Get risk-free rates and compute excess returns
        rfRates = self.modelDB.getRiskFreeRateHistory(\
                [currencyCode], returnsDates, self.marketDB, returnDF=True).loc[currencyCode].fillna(0.0)
        returns = returns.fillna(0.0).subtract(rfRates.values, axis=1)
        returns = returns.mask(preIPOFlag)

        # Load market return from the DB
        marketReturns = self.modelDB.loadRMGMarketReturnHistory(returnsDates, [rmg], returnDF=True).fillna(0.0)
        marketReturns = marketReturns.subtract(rfRates.values, axis=1)

        # Compound daily asset returns to weekly
        (assetPeriodReturns, period_dates1) = ProcessReturns.compute_compound_returns_v3(
            Utilities.df2ma(returns), returnsDates, period_dates, matchDates=True)
        (marketReturns, period_dates2) = ProcessReturns.compute_compound_returns_v3(
                Utilities.df2ma(marketReturns), returnsDates, period_dates, matchDates=True)
        period_dates = period_dates[:-1]
        assetPeriodReturns = pandas.DataFrame(assetPeriodReturns[:, :-1], index=returns.index, columns=period_dates)
        marketReturns = pandas.DataFrame(marketReturns[:, :-1], index=[rmg], columns=period_dates)

        # Trim arrays if we have too many dates
        if len(period_dates) > twoYearWeeks:
            period_dates = period_dates[-twoYearWeeks:]
            assetPeriodReturns = assetPeriodReturns.loc[:, period_dates]
            marketReturns = marketReturns.loc[:, period_dates]
            logging.debug('Trimmed to %d end of week dates, spanning %s to %s',
                    len(period_dates), period_dates[0], period_dates[-1])

        # Clip extreme asset returns
        opms = dict()
        opms['nBounds'] = [8.0, 8.0, 3.0, 3.0]
        outlierClass = Outliers.Outliers(opms)
        assetPeriodReturnsClipped = outlierClass.twodMAD(Utilities.df2ma(assetPeriodReturns), suppressOutput=True)
        assetPeriodReturnsClipped = pandas.DataFrame(\
                assetPeriodReturnsClipped, index=returns.index, columns=assetPeriodReturns.columns)

        # Debugging info
        if self.debugOutput:
            dateStr = [str(d) for d in period_dates]
            idList = [s.getSubIDString() for s in rmgHomeSubIssues]
            marketReturns.to_csv('tmp/mktret-%s-%s-%s.csv' % (rmg.mnemonic, home, dateStr[-1]))
            assetPeriodReturns.to_csv('tmp/assret-%s-%s-%s.csv' % (rmg.mnemonic, home, dateStr[-1]))
            assetPeriodReturnsClipped.to_csv('tmp/assretClip-%s-%s-%s.csv' % (rmg.mnemonic, home, dateStr[-1]))

        # Isolate assets with insufficient history and sort into good and bad
        T = len(assetPeriodReturns.columns)
        numPreIPOReturns = pandas.isnull(assetPeriodReturns).sum(axis=1)
        badList = set(numPreIPOReturns[numPreIPOReturns > 0.5*T].index).intersection(updateList)
        okList = set(updateList).difference(badList)
        nRetMap = float(T) - numPreIPOReturns

        # Special treatment for SPACs
        spacList = AssetProcessor_V4.sort_spac_assets(currDate, rmgHomeSubIssues, self.modelDB, self.marketDB)
        if len(spacList) > 0:
            assetPeriodReturnsClipped.loc[spacList, :] = assetPeriodReturnsClipped.loc[spacList, :].fillna(0.0)
             
        # Do the good assets first
        betas = dict()
        p_vals = dict()
        for t in sorted(set(numPreIPOReturns[okList].values)):
            subList = list(numPreIPOReturns[numPreIPOReturns==t].index)
            sub_dates = period_dates[t:]
            logging.info('Computing betas for %s assets with %s periods history', len(subList), T-t)
            (subset_betas, subset_pvals) = self.computeBetas(
                    rmg, currDate, subList, sub_dates,
                    assetPeriodReturnsClipped.loc[subList, sub_dates],
                    marketReturns.loc[:, sub_dates],
                    debugOutput=(self.debugOutput*(t==0)))
            betas.update(subset_betas)
            p_vals.update(subset_pvals)
            
        # Now do the set of bad assets
        if len(badList) > 0:
            t = int(T / 2.0)
            logging.info('Computing betas for %s assets with less than half the required history', len(badList))
            sub_dates = period_dates[t:]
            (subset_betas, subset_pvals) = self.computeBetas(
                    rmg, currDate, badList, sub_dates,
                    assetPeriodReturnsClipped.loc[badList, sub_dates],
                    marketReturns.loc[:, sub_dates])
            betas.update(subset_betas)
            p_vals.update(subset_pvals)
         
        # Dynamically trim extreme betas
        sidList = list(betas.keys())
        hbeta = numpy.array([betas[sid] for sid in sidList])
        hbeta = list(numpy.clip(hbeta, -1.5, 4.5))
        betas = dict(zip(sidList, hbeta))
        value_pairs = [(a, int(home), betas.get(a), p_vals.get(a), nRetMap.get(a)) for a in updateList if a in betas]
         
        # Output median for debugging
        if len(value_pairs) > 2:
            medianBeta = ma.median([bet for (sid, home, bet, pval, nobs) \
                    in value_pairs if bet is not None], axis=None)
            if home:
                logging.info('%s Median beta value: %.4f', rmg.mnemonic, medianBeta)
            else:
                logging.info('%s Median foreign beta value: %.4f', rmg.mnemonic, medianBeta)
        else:
            logging.info('Only %s beta value: %s', len(value_pairs),
                    [bet for (sid, home, bet, pval, nobs) in value_pairs])
        return value_pairs, nMissMap

    def computeBetas(self, rmg, currDate, subIssues, period_dates, clippedReturns, marketReturns, debugOutput=False):
        """Compute historic betas against the true return series of the market portfolio.
        """

        # Construct TimeSeriesMatrix of period asset returns
        ret = Matrices.TimeSeriesMatrix(subIssues, period_dates)
        ret.data = Utilities.df2ma(clippedReturns)
        mktRet = Matrices.TimeSeriesMatrix([rmg.rmg_id], period_dates)
        mktRet.data = Utilities.df2ma(marketReturns)

        # Fill missing returns with market
        maskedReturns = numpy.array(ma.getmaskarray(ret.data), dtype='float')
        for ii in range(mktRet.data.shape[1]):
            maskedReturns[:, ii] *= mktRet.data[0, ii]
        ret.data = ma.filled(ret.data, 0.0)
        ret.data += maskedReturns

        # Set up regression parameters
        params = dict()
        params['robust'] = False
        params['inclIntercept'] = True

        # Initialise time-series regression class
        TSR = TimeSeriesRegression.TimeSeriesRegression(
                self.modelDB, self.marketDB,
                TSRParameters=params,
                fillWithMarket=False,
                debugOutput=debugOutput,
                getRegStats=True)

        # Compute beta from time-series regression
        mm = TSR.TSR_inner(ret, mktRet)

        return (dict(zip(subIssues, mm.params[0,:])), dict(zip(subIssues, mm.pvals[0,:])))

class RMGCurrencyBeta(ModelDBSource):
    """Class to compute the currency betas for assets traded in a country that
    is not the home country
    """
    def __init__(self, connections, gp=None):
        ModelDBSource.__init__(self, connections)
        self.marketDB = connections.marketDB
        self.tradingRMGs = gp.tradingRMGs
        self.override = hasattr(gp, 'override') and gp.override
        self.cleanup = hasattr(gp, 'cleanup') and gp.cleanup
        self.nuke = hasattr(gp, 'nuke') and gp.nuke
        self.debugOutput = hasattr(gp, 'verbose') and gp.verbose
        self.expand = hasattr(gp, 'expand') and gp.expand
        self.modelDB.setMarketCapCache(60)
        clsFamily = self.marketDB.getClassificationFamily('REGIONS')
        assert(clsFamily is not None)
        clsMembers = dict([(i.name, i) for i in self.marketDB.\
                            getClassificationFamilyMembers(clsFamily)])
        clsMember = clsMembers.get('HomeCountry', None)
        assert(clsMember is not None)
        self.homeCountryCls = clsMember
        rmgs = self.modelDB.getAllRiskModelGroups(inModels=False)
        self.rmgMap = dict((rmg.rmg_id, rmg) for rmg in rmgs)
        rmgs = self.modelDB.getAllRiskModelGroups(inModels=True)
        validRMGQuery = """
            SELECT a.mnemonic, MIN(a.from_dt) FROM (
                SELECT g.mnemonic, GREATEST(m.from_dt, s.from_dt) AS from_dt
                FROM risk_model_group g, rmg_model_map m, risk_model_serie s
                WHERE g.rmg_id > 0 AND g.rmg_id = m.rmg_id
                AND m.rms_id = s.serial_id
                AND s.serial_id > 0 AND s.distribute = 1) a
            GROUP BY a.mnemonic"""
        self.modelDB.dbCursor.execute(validRMGQuery)
        self.modelDB.createCurrencyCache(self.marketDB)
        self.allowedRMGData = dict([(r[0], r[1].date()) \
                for r in self.modelDB.dbCursor.fetchall()])

    def getBulkData(self, dateList, rmgIdList):
        self.log.debug('RMGCurrencyBeta.getBulkData')
        retvals = numpy.empty((len(dateList), len(rmgIdList)), dtype=object)

        # If doing a grand cleanup of everything, do it once, then reset the flag
        if self.nuke:
            check = input('About to delete all records. Are you sure? ').lower()
            if check == 'yes' or check == 'y':
                self.nuke = False
                self.modelDB.dbCursor.execute("""SELECT dt FROM rmg_currency_beta""")
                ret = self.modelDB.dbCursor.fetchall()
                nukeDates = sorted(set(r[0] for r in ret))
                for dt in nukeDates:
                    logging.info('Deleting all records from %s', dt)
                    self.modelDB.dbCursor.execute(
                    """DELETE FROM rmg_currency_beta
                    WHERE dt=:dt_arg""", dt_arg=dt)
                self.modelDB.commitChanges()

        for (rIdx, rmg) in enumerate(rmgIdList):
            rmgObj = self.rmgMap[rmg]
            for (dIdx, date) in enumerate(dateList):
                if not rmgObj.setRMGInfoForDate(date):
                    raise Exception('Cannot determine currency for %s risk model group (%d) on %s'\
                            % (rmgObj.description, rmgObj.rmg_id, str(date)))
                rval = Utilities.Struct()
                rval.valuePairs = self.computeCurrencyBetas(rmgObj, date, debugOutput=self.debugOutput)
                if rval.valuePairs is not None:
                    retvals[dIdx, rIdx] = rval
        return retvals

    def getAssetMktInfo(self, date, subIssues):
        # Get home market classifications for list of assets
        # Not sure whether this override flag is still needed
        if self.override:
            allowedRMGCodes = [rmgCode for (rmgCode, dt) in self.allowedRMGData.items()]
        else:
            allowedRMGCodes = [rmgCode for (rmgCode, dt) in self.allowedRMGData.items()
                    if date >= dt]

        # Load market classification dict
        clsRevision = self.marketDB.getClassificationMemberRevision(self.homeCountryCls, date)
        assetMarketDict =  self.modelDB.getMktAssetClassifications(clsRevision, subIssues, date, self.marketDB)
        return assetMarketDict, allowedRMGCodes

    def getForeignTraded(self, rmg, currDate):
        """Loads assets whose home country is RMG, but trade elswhere
        Returns a mapping from each trading RMG to a list of the assets trading there
        """
        # Load all sub-issues trading on the RMG
        rmgTradingSubIssues = self.modelDB.getActiveSubIssues(rmg, currDate)
        rmgTradingSubIssues = set(rmgTradingSubIssues)

        # Get home market information
        allSubIssues = self.modelDB.getAllActiveSubIssues(currDate)
        (assetMarketDict, allowedRMGCodes) = self.getAssetMktInfo(currDate, allSubIssues)

        # Subset of assets traded elsewhere, but with home country set as current RMG
        listedAbroad = [sid for (sid,cls) in assetMarketDict.items() \
                        if sid not in rmgTradingSubIssues and \
                        cls.classification.code==rmg.mnemonic]
        assetTypeMap = AssetProcessor_V4.get_asset_info(currDate, listedAbroad,
                self.modelDB, self.marketDB, 'ASSET TYPES', 'Axioma Asset Type')
        marketMap = AssetProcessor_V4.get_asset_info(currDate, listedAbroad,
                self.modelDB, self.marketDB, 'REGIONS', 'Market')

        # Get mapping from subissue to trading RMG
        tradingCodeAssetMap = defaultdict(list)
        for (sid, tradeRMG) in  self.modelDB.getSubIssueRiskModelGroupPairs(currDate, restrict=listedAbroad):
            tradingCurrencyCode = tradeRMG.getCurrencyCode(currDate)
            tradingCodeAssetMap[tradingCurrencyCode].append(sid)

        # Report on these assets traded elsewhere
        self.log.debug('Adding %d %s assets trading outside of %s',
                    len(listedAbroad), rmg.description, rmg.mnemonic)
        self.log.info('%d Trading currencies for valid/covered home country assignment',
                    len(list(tradingCodeAssetMap.keys())))

        return tradingCodeAssetMap, assetTypeMap, marketMap

    def computeCurrencyBetas(self, rmg, currDate, debugOutput=False):
        debugOutput=True
        # Do some checking of dates and subissues
        if rmg.mnemonic not in self.allowedRMGData:
            self.log.info("Skipping %s since it is in no model", rmg.mnemonic)
            return None
        if currDate < self.allowedRMGData[rmg.mnemonic] and not self.override:
            self.log.info("Skipping %s since it doesn't join a model until %s", rmg.mnemonic,
                          self.allowedRMGData[rmg.mnemonic])
            return None
        self.log.info('Processing currency betas for %s on %s', rmg.description, currDate)
        if currDate.isoweekday() > 5:
            self.log.info('Saturday/Sunday, skipping.')
            return None
        prevDate = self.modelDB.getDates([rmg], currDate, 1)[0]

        # Get list of assets with RMG as home but trading on another market,
        tradingCodeAssetMap, assetTypeMap, marketMap = self.getForeignTraded(rmg, currDate)
        if len(tradingCodeAssetMap) == 0:
            self.log.info('No active sub-issues, skipping.')
            return None

        # Initialise
        total_ret_val = list()
        total_nMiss = dict()
        sidTradingRMGMap = dict()
        outRMG = dict()
        outRMG[rmg.rmg_id] = rmg

        # For China, compute betas for both international and domestic market
        if rmg.description == 'China':

            # Compute betas against Domestic China market and local->trading currency return
            domesticChinaRMG = self.modelDB.getRiskModelGroupByISO('XC')
            outRMG[domesticChinaRMG.rmg_id] = domesticChinaRMG
            for (tradingCode, sidList) in tradingCodeAssetMap.items():
                if len(sidList) > 0:
                    (ret_val, nMissMap) = self.computeCurrencyBetasInner(rmg, domesticChinaRMG, tradingCode, sidList, currDate, prevDate)
                    if len(ret_val) > 0:
                        ret_val = [(sid, rmg_id, iso, str(assetTypeMap[sid]), str(marketMap[sid]), beta, p_val, nobs) \
                                for (sid, rmg_id, iso, beta, p_val, nobs) in ret_val]
                        total_ret_val.extend(ret_val)
                        total_nMiss.update(nMissMap)

        # Compute betas against local market and local->trading currency return
        for (tradingCode, sidList) in tradingCodeAssetMap.items():
            print(rmg, len(sidList))
            if len(sidList) > 0:
                (ret_val, nMissMap) = self.computeCurrencyBetasInner(rmg, rmg, tradingCode, sidList, currDate, prevDate)
                if len(ret_val) > 0:
                    ret_val = [(sid, rmg_id, iso, str(assetTypeMap[sid]), str(marketMap[sid]), beta, p_val, nobs) \
                            for (sid, rmg_id, iso, beta, p_val, nobs) in ret_val]
                    total_ret_val.extend(ret_val)
                    total_nMiss.update(nMissMap)
            for sid in sidList:
                sidTradingRMGMap[sid] = tradingCode

        if debugOutput:
            colNames = ['Beta', 'P-Value', 'n-Rets', 'Missing%']
            outArray = Matrices.allMasked((len(total_ret_val), len(colNames)))
            sidList = []
            idx = 0
            for (sid, rmg_id, iso, typ, mkt, beta, pval, nRets) in total_ret_val:
                outArray[idx, 0] = beta
                outArray[idx, 1] = pval
                outArray[idx, 2] = nRets
                if sid in nMissMap:
                    outArray[idx, 3] = total_nMiss[sid]
                sidList.append('%s|%s|%s|%s|%s|%s' % (currDate, outRMG[rmg_id].mnemonic,
                    sidTradingRMGMap[sid], sid.getSubIDString(), assetTypeMap[sid], marketMap[sid]))
                idx += 1
            outfile = 'tmp/cbeta-%s-%s.csv' % (rmg.mnemonic, currDate)
            Utilities.writeToCSV(outArray, outfile, rowNames=sidList, columnNames=colNames)
        return total_ret_val

    def computeCurrencyBetasInner(
            self, rmg, marketRMG, tradingCode, rmgSubIssues, currDate, prevDate):
        """Inner processing loop for currency betas
        """
        # Processing of assets to be updated
        if len(rmgSubIssues) < 1:
            return [], dict()
        subissueIdxMap = dict(zip(rmgSubIssues, list(range(len(rmgSubIssues)))))
        logging.info('Computing betas for %d %s assets', len(rmgSubIssues), rmg.description)

        # Currency info
        currencyCode = rmg.getCurrencyCode(currDate)
        currencyID = self.modelDB.currencyCache.getCurrencyID(currencyCode, currDate)
        tradingCurrencyID = self.modelDB.currencyCache.getCurrencyID(tradingCode, currDate)
        logging.info('... Trade currency: %s, Home currency: %s', tradingCode, currencyCode)

        # Get list of dates needed
        twoYearWeeks = 104
        twoYearsDays = (twoYearWeeks + 2) * 7
        startDate = currDate - (twoYearsDays * datetime.timedelta(1))
        tradingDaysList = self.modelDB.getDateRange(None, startDate, currDate, excludeWeekend=True)
        logging.debug('%d weekdays, spanning %s to %s',
                len(tradingDaysList), tradingDaysList[0], tradingDaysList[-1])

        # Only recompute betas at the start of each new period
        # In this case, on Mondays (or first trading day of the week)
        if prevDate.weekday() < currDate.weekday() \
                and (currDate - prevDate < datetime.timedelta(7)) \
                and currDate != self.allowedRMGData[rmg.mnemonic]:
            # If not start of new period, roll over last period's values
            logging.info('Rolling previous values from %s for %s', prevDate, currDate)
            return [], dict()

        # Get the weekly dates (Mon-Mon, Tue-Tue etc.)
        period_dates = Utilities.change_date_frequency(tradingDaysList, frequency='weekly')
        logging.debug('%d end of week dates, spanning %s to %s',
                len(period_dates), period_dates[0], period_dates[-1])

        # Retrieve asset returns from the DB
        returns = self.modelDB.loadTotalReturnsHistoryV3(
                [rmg], currDate, rmgSubIssues,
                len(tradingDaysList), dateList=tradingDaysList,
                assetConvMap=tradingCurrencyID, excludeWeekend=True)
        if returns.data is None:
            return [], dict()

        # Do some housekeeping
        returns.data = ma.masked_where(returns.preIPOFlag, returns.data)
        nMissingReturns = ma.sum(returns.notTradedFlag==0, axis=1)
        nMissingReturns = nMissingReturns / float(max(nMissingReturns))
        nMissMap = dict(zip(rmgSubIssues, nMissingReturns))

        # Load currency returns from the DB
        fxReturns = self.modelDB.loadCurrencyReturnsHistory(None, None, 0,
                [currencyID], tradingCurrencyID, dateList=returns.dates, idlookup=False)
        fxReturns.data = ma.filled(fxReturns.data, 0.0)

        # Load market return from the DB
        marketReturns = self.modelDB.loadRMGMarketReturnHistory(returns.dates, [marketRMG])
        marketReturns.data = ma.filled(marketReturns.data, 0.0)

        # Compound daily asset returns to weekly
        (assetPeriodReturns, period_dates1) = ProcessReturns.compute_compound_returns_v3(
            returns.data, returns.dates, period_dates, matchDates=True)
        (marketReturns, period_dates2) = ProcessReturns.compute_compound_returns_v3(
                marketReturns.data, marketReturns.dates, period_dates, matchDates=True)
        (fxReturns, period_dates3) = ProcessReturns.compute_compound_returns_v3(
                fxReturns.data, fxReturns.dates, period_dates, matchDates=True)
        period_dates = period_dates[:-1]
        assetPeriodReturns = assetPeriodReturns[:, :-1]
        marketReturns = marketReturns[0, :-1]
        fxReturns = fxReturns[0, :-1]

        if len(period_dates) > twoYearWeeks:
            period_dates = period_dates[-twoYearWeeks:]
            assetPeriodReturns = assetPeriodReturns[:, -twoYearWeeks:]
            marketReturns = marketReturns[-twoYearWeeks:]
            fxReturns = fxReturns[-twoYearWeeks:]
            logging.debug('Trimmed to %d end of week dates, spanning %s to %s',
                    len(period_dates), period_dates[0], period_dates[-1])
        opms = dict()
        opms['nBounds'] = [8.0, 8.0, 3.0, 3.0]
        outlierClass = Outliers.Outliers(opms)
        assetPeriodReturnsClipped = outlierClass.twodMAD(assetPeriodReturns, suppressOutput=True)

        # Debugging info
        if self.debugOutput:
            dateStr = [str(d) for d in period_dates]
            idList = [s.getSubIDString() for s in rmgSubIssues]
            Utilities.writeToCSV(marketReturns[numpy.newaxis,:], 'tmp/mktret-%s-%s.csv' % \
                    (marketRMG.mnemonic, dateStr[-1]), columnNames=dateStr, rowNames=[rmg.mnemonic])
            Utilities.writeToCSV(assetPeriodReturns, 'tmp/assret-%s-%s-%s.csv' % \
                    (rmg.mnemonic, tradingCode, dateStr[-1]), columnNames=dateStr, rowNames=idList)
            Utilities.writeToCSV(fxReturns[numpy.newaxis,:], 'tmp/curRet-%s-%s-%s.csv' % \
                    (rmg.mnemonic, tradingCode, dateStr[-1]), columnNames=dateStr, rowNames=[tradingCode])

        # Isolate assets with insufficient history
        T = assetPeriodReturns.shape[1]
        numPreIPOReturns = ma.sum(ma.getmaskarray(assetPeriodReturns), axis=1)
        nRets = float(T) - numPreIPOReturns
        nRetMap = dict(zip(rmgSubIssues, nRets))

        # Sort assets into groups according to length of available history
        badAssetsIdx = numpy.flatnonzero(numPreIPOReturns > 0.5*T)
        okAssetsIdx = numpy.flatnonzero(numPreIPOReturns <= 0.5*T)

        # Do the good assets first
        betas = dict()
        p_vals = dict()
        nmReturns = numpy.take(numPreIPOReturns, okAssetsIdx, axis=0)
        idxNumMap = dict(zip(okAssetsIdx, nmReturns))
        for t in set(nmReturns):
            idxList = [idx for idx in okAssetsIdx if idxNumMap[idx] == t]
            logging.debug('Computing betas for %s assets with %s periods history', len(idxList), T-t)
            subsetIssues = numpy.take(rmgSubIssues, idxList)
            subsetReturns = ma.take(assetPeriodReturns, idxList, axis=0)
            subsetReturnsClipped = ma.take(assetPeriodReturnsClipped, idxList, axis=0)

            # Do the actual regression
            x = numpy.transpose(numpy.array([numpy.ones(T-t, float), marketReturns[t:], fxReturns[t:]]))
            y = numpy.transpose(subsetReturnsClipped[:,t:])
            res = Utilities.robustLinearSolver(y, x, robust=False, computeStats=True)
            subset_betas = dict(zip(subsetIssues, res.params[-1,:]))
            subset_pvals = dict(zip(subsetIssues, res.pvals[-1,:]))
            betas.update(subset_betas)
            p_vals.update(subset_pvals)

        # Dynamically trim extreme betas
        sidList = list(betas.keys())
        hbeta = numpy.array([betas[sid] for sid in sidList])
        betas = dict(zip(sidList, hbeta))
        value_pairs = [(a, marketRMG.rmg_id, tradingCode, betas.get(a), p_vals.get(a), nRetMap.get(a)) for a in betas]

        # Output median for debugging
        if len(value_pairs) > 2:
            medianBeta = ma.median([bet for (sid, rmg_id, iso, bet, pval, nobs) \
                    in value_pairs if bet is not None], axis=None)
            logging.info('... %s/%s Median currency beta value: %.4f', rmg.mnemonic, tradingCode, medianBeta)
        else:
            logging.info('... Only %s beta value: %s', len(value_pairs),
                    [bet for (sid, rmg_id, iso, bet, pval, nobs) in value_pairs])
        return value_pairs, nMissMap

class RMGMarketReturn(ModelDBSource):
    """Class to create the market portfolio return for risk model groups.
    """
    def __init__(self, connections, robust=False, gp=None):
        ModelDBSource.__init__(self, connections)
        self.rmgs = self.modelDB.getAllRiskModelGroups(inModels=False)
        self.rmgMap = dict((rmg.rmg_id, rmg) for rmg in self.rmgs)
        self.marketDB = connections.marketDB
        self.clsData = None # cache for home country data
        clsFamily = self.marketDB.getClassificationFamily('REGIONS')
        assert(clsFamily is not None)
        clsMembers = dict([(i.name, i) for i in self.marketDB.\
                            getClassificationFamilyMembers(clsFamily)])
        clsMember = clsMembers.get('HomeCountry', None)
        assert(clsMember is not None)
        self.homeCountryCls = clsMember
        self.cleanup = hasattr(gp, 'cleanup') and gp.cleanup
        self.notInRiskModels = hasattr(gp, 'notInRiskModels') and gp.notInRiskModels
        self.robustRegression = robust
        self.tradingRMGs = gp.tradingRMGs
        self.modelDB.createCurrencyCache(connections.marketDB)
        self.forexCache = self.modelDB.currencyCache
        self.override = hasattr(gp, 'override') and gp.override
        self.debugOutput = hasattr(gp, 'verbose') and gp.verbose

    
    def getBulkData(self, dateList, rmgIdList):
        self.log.debug('RMGMarketReturn.getBulkData')
        retvals = numpy.empty((len(dateList), len(rmgIdList)), dtype=object)
        dataArray = numpy.zeros(retvals.shape, float)
         
        # Loop round dates
        for (dIdx, date) in enumerate(dateList):
             
            # Remove existing data from DB if required
            if self.cleanup:
                deleteDicts = [dict([
                    ('rmg_id', rmg_id),
                    ('dt', date)
                    ]) for rmg_id in rmgIdList]
                if not self.robustRegression:
                    self.modelDB.dbCursor.executemany("""DELETE FROM rmg_market_return
                            WHERE rmg_id=:rmg_id AND dt=:dt""", deleteDicts)
                else:
                    self.modelDB.dbCursor.executemany("""DELETE FROM rmg_market_return_v3
                            WHERE rmg_id=:rmg_id AND dt=:dt""", deleteDicts)
                self.log.info('Deleting %d records', len(deleteDicts))

            # Loop round markets
            for (rIdx, rmg) in enumerate(rmgIdList):
                # Perform computation on weekdays for all markets
                # and weekend dates for those markets that trade on such
                if date.isoweekday() > 5 and rmg not in self.tradingRMGs[date]:
                    # skip non-trading weekends
                    continue
                rval = Utilities.Struct()
                # Compute market return and basic proxy returns
                rval.value = self.computeMarketReturn(self.rmgMap[rmg], date)
                if rval.value is not None:
                    retvals[dIdx, rIdx] = rval
                    dataArray[dIdx, rIdx] = rval.value

        if self.debugOutput:
            dtList = [str(d) for d in dateList]
            rmgList = [self.rmgMap[rmg].description.replace(',','') for rmg in rmgIdList]
            Utilities.writeToCSV(dataArray, 'tmp/mkt-ret-%s.csv' % dateList[-1],
                    columnNames=rmgList, rowNames=dtList)

        return retvals

    def computeMarketReturn(self, rmg, currDate):

        # Initialise
        mktRet = 0.0
        allSubIssues = self.modelDB.getAllActiveSubIssues(currDate)

        if self.robustRegression:
            self.log.info('Processing robust market return for %s on %s',
                    rmg.description, currDate)
        else:
            self.log.info('Processing market return for %s on %s',
                    rmg.description, currDate)
         
        # Do some fiddling about to deal with China and US Smallcap
        if rmg.description == 'Domestic China':
            rmgList = self.modelDB.getAllRiskModelGroups()
            realRMG = [r for r in rmgList if r.description=='China'][0]
        elif rmg.description == 'United States Small Cap':
            rmgList = self.modelDB.getAllRiskModelGroups()
            realRMG = [r for r in rmgList if r.description=='United States'][0]
        else:
            realRMG = rmg

        # Find previous trading day
        prevDate = self.modelDB.getDates([realRMG], currDate, 1, excludeWeekend=False)
        if len(prevDate) < 1:
            self.log.info('No previous trading day for %s, skipping.', rmg.mnemonic)
            return None
        if currDate in prevDate:
            prevDate = prevDate[0]
        else:
            prevDate = prevDate[-1]

        # Load market portfolio
        #amp = self.modelDB.convertLMP2AMP(rmg, prevDate)
        amp=None
        marketRaw = self.modelDB.getRMGMarketPortfolio(rmg, prevDate, amp=amp)
        market = [(sid, wt) for (sid, wt) in marketRaw if sid in allSubIssues]
        if len(market) < len(marketRaw):
            logging.warning('%d market portfolio assets not in universe', len(marketRaw)-len(market))
        if len(market) == 0:
            self.log.info('Empty %s market portfolio for %s, skipping.', realRMG.mnemonic, prevDate)
            return mktRet

        # Find set of trading markets spanned by benchmark subissues
        timeZoneCheck = False
        if timeZoneCheck:

            sidWtMap = dict(market)
            mktSubIssues = []
            wts = []

            tradingRMGMap = AssetProcessor_V4.loadRiskModelGroupAssetMap(
                    currDate, list(sidWtMap.keys()), self.modelDB, self.rmgs, quiet=True)

            # Filter out anything traded too far ahead of the benchmark's market
            outputList = []
            for trad_rmg_id in tradingRMGMap.keys():
                trad_sids = tradingRMGMap[trad_rmg_id]
                if len(trad_sids) < 1:
                    continue
                wtList = [sidWtMap[sid] for sid in trad_sids]
                trad_gmt_offset = self.rmgMap[trad_rmg_id].gmt_offset
                if trad_rmg_id == rmg.rmg_id:
                    mktSubIssues.extend(trad_sids)
                    wts.extend(wtList)
                else:
                    if trad_gmt_offset - rmg.gmt_offset > -4:
                        mktSubIssues.extend(trad_sids)
                        wts.extend(wtList)
                    else:
                        logging.warning('Will drop %d assets for %s',
                                len(trad_sids), self.rmgMap[trad_rmg_id].mnemonic)
                    rmgWt = str(round(100.0 * numpy.sum(wtList, axis=None), 1))
                    outputList.append(self.rmgMap[trad_rmg_id].mnemonic)
                    outputList.append(rmgWt)

            if self.debugOutput and len(outputList) > 0:
                logging.info('Initial trading market breakdown for (%s,%s): %s', rmg.mnemonic, currDate, ','.join(outputList))
        else:
            # Get subissues and weights
            mktSubIssues, wts = zip(*market)
            wts = numpy.array(wts, dtype=float)

        # Load current date's asset returns
        currencyCode = rmg.getCurrencyCode(currDate)
        currencyID = self.forexCache.getCurrencyID(currencyCode, currDate)
        assetReturns = self.modelDB.loadTotalReturnsHistoryV3(
                [realRMG], currDate, mktSubIssues, 1, assetConvMap=currencyID,
                allRMGDates=True, excludeWeekend=False)

        # Check that returns actually exist for the current date
        if currDate not in assetReturns.dates:
            self.log.info('Date %s not in available returns dates, skipping', currDate)
            self.log.info('Market return for %s on %s: %.4f', rmg.description, currDate, mktRet)
            return mktRet
        if assetReturns.data is None:
            self.log.warning('No %s returns found for %s', realRMG, currDate)
            return mktRet

        assetMktRets = assetReturns.data[:,-1]

        # Testing/debugging output
        if self.debugOutput:
            wts = numpy.array(wts, float)
            dataArray = numpy.concatenate([\
                    wts[:, numpy.newaxis], ma.filled(assetMktRets, 0.0)[:,numpy.newaxis]], axis=1)
            sidList = [sid.getSubIDString() for sid in mktSubIssues]
            outFile = 'tmp/asst-ret-%s-%s.csv' % (rmg.mnemonic, currDate)
            Utilities.writeToCSV(dataArray, outFile, columnNames=['wt', 'ret'], rowNames=sidList)

        # Compute weighted average of asset returns
        mktRet = ma.average(ma.filled(assetMktRets, 0.0), weights=wts, axis=None)
        if self.robustRegression:
            self.log.info('Market return (robust) for %s on %s: %.4f', rmg.description, currDate, mktRet)
        else:
            self.log.info('Market return for %s on %s: %.4f', rmg.description, currDate, mktRet)
        return float(mktRet)

class AMPIndustryReturn(ModelDBSource):
    """Class to create the market portfolio return for risk model groups.
    """
    def __init__(self, connections, robust=False, gp=None):
        ModelDBSource.__init__(self, connections)
        self.gicsCls = gp.gicsCls
        self.revision_id = gp.revision_id
        self.tradingRMGs = gp.tradingRMGs
        self.ampRmgMap = gp.ampRmgMap
        self.industryList = gp.industryList
        self.useRmgMarketPortfolio = gp.useRmgMarketPortfolio
        self.modelDB.createCurrencyCache(connections.marketDB)
        self.forexCache = self.modelDB.currencyCache
        self.marketDB = connections.marketDB
        self.cleanup = hasattr(gp, 'cleanup') and gp.cleanup
        self.notInRiskModels = hasattr(gp, 'notInRiskModels') and gp.notInRiskModels
        self.override = hasattr(gp, 'override') and gp.override
        self.debugOutput = hasattr(gp, 'verbose') and gp.verbose
    
    def getBulkData(self, amp, date):
        self.log.debug('AMPIndustryReturn.getBulkData')

        # get currency code in which returns will be computed
        assert len(self.ampRmgMap[amp.id]) == 1 # hack: assumes a single country amp
        rmg = list(self.ampRmgMap[amp.id].keys())[0]
        currencyCode = rmg.getCurrencyCode(date)
        currencyID = self.forexCache.getCurrencyID(currencyCode, date)

        # get previous trading day
        prevDate = self.modelDB.getDates(list(self.ampRmgMap[amp.id].keys()), 
                date, 1, excludeWeekend=False)
        if len(prevDate) < 1:
            self.log.info('No previous trading day for %s, skipping.', 
                    amp.name)
            return numpy.full((len(self.industryList),), numpy.nan)
        if date in prevDate:
            prevDate = prevDate[0]
        else:
            prevDate = prevDate[-1]

        # load portfolio constituents
        if not self.useRmgMarketPortfolio:
            wgts = pandas.Series(dict(self.modelDB.getModelPortfolioConstituents(prevDate, amp.id)))
            if wgts.empty:
                assert len(self.ampRmgMap[amp.id]) == 1 # to do: implement support for multiple rmgs
                rmg = list(self.ampRmgMap[amp.id].keys())[0]
                wgts = pandas.Series(dict(self.modelDB.getRMGMarketPortfolio(rmg, prevDate)))
        else:
            assert len(self.ampRmgMap[amp.id]) == 1 # to do: implement support for multiple rmgs
            rmg = list(self.ampRmgMap[amp.id].keys())[0]
            wgts = pandas.Series(dict(self.modelDB.getRMGMarketPortfolio(rmg, prevDate)))
     
        # refine asset list
        allSubIssues = self.modelDB.getAllActiveSubIssues(date) 
        assets = [a for a in wgts.index if a in allSubIssues]
        if len(assets) == 0:
            return numpy.zeros((len(self.industryList),))

        # get classificaiton
        exposures = self.gicsCls.getExposures(prevDate, assets, self.industryList, self.modelDB)
        indExp = pandas.DataFrame(exposures, index=self.industryList, columns=assets).T.fillna(0.0)

        # get asset returns
        assetReturns = self.modelDB.loadTotalReturnsHistoryV3(
                self.ampRmgMap[amp.id], date, assets, 1, assetConvMap=currencyID,
                allRMGDates=True, excludeWeekend=False)
        if date not in assetReturns.dates:
            self.log.info('Date %s not in available returns dates, skipping', date)
            return numpy.zeros((len(self.industryList),))
        if assetReturns.data is None:
            self.log.warning('No %s returns found for %s', amp.name, date)
            return numpy.zeros((len(self.industryList),))
        assetReturns = assetReturns.toDataFrame()[date].fillna(0.)

        # compute industry returns
        assets = list(set(assetReturns.index).intersection(set(assets)))
        wgts = wgts.reindex(index=assets)
        wgts = wgts/wgts.sum()
        assetReturns = assetReturns.reindex(index=assets)
        indExp = indExp.reindex(index=assets)

        wgtIndExp = indExp.multiply(wgts, axis='rows')
        indPorts = wgtIndExp.div(wgtIndExp.sum())
        indRets = indPorts.T.dot(assetReturns)
        return indRets.values

class RegionReturn(ModelDBSource):
    """Class to create the regional portfolio return.
    """
    def __init__(self, connections, robust=False, gp=None):
        ModelDBSource.__init__(self, connections)
        rmgs = self.modelDB.getAllRiskModelGroups(inModels=False)
        self.rmgMap = dict((rmg.rmg_id, rmg) for rmg in rmgs)
        self.marketDB = connections.marketDB
        self.clsData = None # cache for home country data
        clsFamily = self.marketDB.getClassificationFamily('REGIONS')
        assert(clsFamily is not None)
        clsMembers = dict([(i.name, i) for i in self.marketDB.\
                            getClassificationFamilyMembers(clsFamily)])
        clsMember = clsMembers.get('HomeCountry', None)
        assert(clsMember is not None)
        self.homeCountryCls = clsMember
        self.modelDB.createCurrencyCache(connections.marketDB)
        self.forexCache = self.modelDB.currencyCache
        self.override = hasattr(gp, 'override') and gp.override
        self.debugOutput = hasattr(gp, 'verbose') and gp.verbose

    def getBulkData(self, dateList, regIdList):
        self.log.debug('RegionReturn.getBulkData')
        retvals = numpy.empty((len(dateList), len(regIdList)), dtype=object)

        # Loop round dates
        allRMG = self.modelDB.getAllRiskModelGroups() # used for getting all trading dates - returnDateList
        for (dIdx, date) in enumerate(dateList):

            if date.isoweekday() > 5:
                self.log.info('Saturday/Sunday, skipping.')
                continue

            # figure out return date list
            returnDateList = self.modelDB.getAllRMDates(allRMG, date, numDays = 2)[1]
            returnDateList = sorted(dt for dt in returnDateList if dt>=datetime.date(1980,1,1))
            if date not in returnDateList:
                logging.info('Date %s is not a valid trading day', date)
                continue

            # Loop round regions
            for (rIdx, regId) in enumerate(regIdList):
                wts = []
                rets = []
                # Pick up list of rmgs for particular region
                rmgIds = self.modelDB.getRMGIdsForRegion(regId, date)
                if len(rmgIds) == 0:
                    continue
                # Retrieve the details for the region
                region = self.modelDB.getRiskModelRegion(regId)

                # Loop round RMGs and get weights and returns in relevant currency
                for rmgId in rmgIds:
                    weights, returns = self.getRMGAssetDetails(\
                            self.rmgMap[rmgId], date, numeraire=region.currency_code,
                            returnDateList=returnDateList)
                    wts.extend(weights)
                    rets.extend(returns)

                # Compute region return
                rets = Utilities.screen_data(ma.array(rets, float), fill=True)
                wts = Utilities.screen_data(ma.array(wts, float), fill=True)
                if len(wts) > 2:
                    regionReturn = Utilities.screen_data(
                        Utilities.robustAverage(rets, wts, k=15.0)[0], fill=True)
                    regionReturn = float(regionReturn)
                    self.log.info('Region return (robust) for %s on %s: %.6f %s',
                                region.name, date, regionReturn, region.currency_code)
                    rval = Utilities.Struct()
                    rval.value = regionReturn
                    retvals[dIdx, rIdx] = rval
                else:
                    self.log.warning('No return for %s on %s', region.name, date)

        return retvals

    def getRMGAssetDetails(self, rmg, currDate, numeraire='USD', returnDateList=None):
        self.log.debug('Retrieving market details for %s', rmg.description)
        retFail = ([], [])
 
        # Domestic China is a pseudo-rmg, so we need to link it to the real China RMG
        if rmg.description == 'Domestic China':
            rmgList = self.modelDB.getAllRiskModelGroups()
            realRMG = [r for r in rmgList if r.description=='China'][0]
        else:
            realRMG = rmg

        # Find previous trading day
        prevDate = self.modelDB.getDates([realRMG], currDate, 1, excludeWeekend=True)
        if len(prevDate) < 1:
            self.log.info('No previous trading day for %s, skipping.', rmg.mnemonic)
            return retFail
        if currDate in prevDate:
            prevDate = prevDate[0]
        else:
            prevDate = prevDate[-1]

        # Load up market assets
        amp = self.modelDB.convertLMP2AMP(rmg, prevDate)
        market = self.modelDB.getRMGMarketPortfolio(rmg, prevDate, amp=amp)
        if len(market) == 0:
            self.log.info('Empty market portfolio for %s, skipping.', rmg.mnemonic)
            return retFail
        mktSubIssues, dummy = zip(*market)

        # Pick out weights and returns for the most liquid assets
        currencyID = self.forexCache.getCurrencyID(numeraire, prevDate)
        wts = AssetProcessor_V4.robustLoadMCaps(
                prevDate, mktSubIssues, currencyID, self.modelDB, self.marketDB)
        wts = Utilities.screen_data(Utilities.df2ma(wts), fill=True)

        # Load asset history for last daysBack days
        daysBack = 2
        currencyID = self.forexCache.getCurrencyID(numeraire, currDate)
        assetReturns = self.modelDB.loadTotalReturnsHistoryV3(
                [realRMG], currDate, mktSubIssues, daysBack, allRMGDates=True,
                dateList=returnDateList, assetConvMap=currencyID,)
        if (assetReturns.data is None) or (currDate not in assetReturns.dates):
            self.log.warning('No returns found  for %s for %s', currDate, rmg.mnemonic)
            return retFail
        assetMktRets = Utilities.screen_data(assetReturns.data[:,-1], fill=True)
        return wts, assetMktRets

class RMGMarketVolatility(ModelDBSource):
    """Class to create the market portfolio volatility for risk model groups.
    """
    def __init__(self, connections, gp=None):
        ModelDBSource.__init__(self, connections)
        rmgs = self.modelDB.getAllRiskModelGroups(inModels=False)
        self.clsData = None # cache for home country data
        self.marketDB = connections.marketDB
        clsFamily = self.marketDB.getClassificationFamily('REGIONS')
        assert(clsFamily is not None)
        clsMembers = dict([(i.name, i) for i in self.marketDB.\
                            getClassificationFamilyMembers(clsFamily)])
        clsMember = clsMembers.get('HomeCountry', None)
        assert(clsMember is not None)
        self.homeCountryCls = clsMember
        self.rmgMap = dict((rmg.rmg_id, rmg) for rmg in rmgs)
        self.cleanup = gp.cleanup
        self.tradingRMGs = gp.tradingRMGs
        self.modelDB.createCurrencyCache(connections.marketDB)
        self.forexCache = self.modelDB.currencyCache
        self.override = gp.override
        self.debugOutput = gp.verbose

    def getBulkData(self, dateList, rmgIdList):
        self.log.debug('RMGMarketVolatility.getBulkData')
        retvals = numpy.empty((len(dateList), len(rmgIdList)), dtype=object)

        # Loop round dates
        for (dIdx, date) in enumerate(dateList):
            # Remove existing data from DB if required
            if self.cleanup:
                deleteDicts = [dict([
                    ('rmg_id', rmg_id),
                    ('dt', date)
                    ]) for rmg_id in rmgIdList]
                self.modelDB.dbCursor.executemany("""DELETE FROM rmg_market_volatility
                        WHERE rmg_id=:rmg_id AND dt=:dt""", deleteDicts)
                self.log.info('Deleting %d records', len(deleteDicts))

            # Loop round markets
            for (rIdx, rmg) in enumerate(rmgIdList):
                if date.isoweekday() > 5 and rmg not in self.tradingRMGs[date]:
                    # skip non-trading weekends
                    continue
                rval = Utilities.Struct()
                # Compute market return and basic proxy returns
                rval.value = self.computeMarketVolatility(self.rmgMap[rmg], date)
                if rval.value is not None:
                    retvals[dIdx, rIdx] = rval
        return retvals

    def computeMarketVolatility(self, rmg, currDate):
        """ Routine to compute the cross-sectional volatility of asset
        returns on a given date for a given RMG
        """
        self.log.info('Processing market volatility for %s on %s',
                      rmg.description, currDate)

        # Initialise things
        daysBack = 1
        currencyCode = rmg.getCurrencyCode(currDate)
        currencyID = self.forexCache.getCurrencyID(currencyCode, currDate)

        # Load the market portfolio
        amp = self.modelDB.convertLMP2AMP(rmg, currDate)
        market = self.modelDB.getRMGMarketPortfolio(rmg, currDate, amp=amp)
        if len(market) == 0:
            self.log.info('No market portfolio, skipping.')
            return None

        # Get "real" RMG if necessary
        if rmg.description == 'United States Small Cap':
            rmgList = self.modelDB.getAllRiskModelGroups()
            retRMG = [r for r in rmgList if r.description=='United States'][0]
        elif rmg.description == 'Domestic China':
            rmgList = self.modelDB.getAllRiskModelGroups()
            retRMG = [r for r in rmgList if r.description=='China'][0]
        else:
            retRMG = rmg

        # Set up subissue information
        rmgSubIssues, wts = zip(*market)
        rmgAssetMap = {retRMG.rmg_id:rmgSubIssues}
        tradingRmgAssetMap = {retRMG.rmg_id:rmgSubIssues}
        assetTypeDict = AssetProcessor_V4.get_asset_info(currDate, rmgSubIssues,
                self.modelDB, self.marketDB, 'ASSET TYPES', 'Axioma Asset Type')

        # Load in the day's returns
        returnsProcessor = ProcessReturns.assetReturnsProcessor(
            [retRMG], rmgSubIssues, rmgAssetMap, tradingRmgAssetMap,
            assetTypeDict, None, numeraire_id=currencyID,
            tradingCurrency_id=currencyID)
        returns = returnsProcessor.process_returns_history(
            currDate, daysBack, self.modelDB, self.marketDB,
            loadOnly=True, applyRT=False, trimData=False)
        if currDate not in returns.dates:
            self.log.info('Date %s not in available returns dates, skipping', currDate)
            return None

        # If too few assets have non-zero and non-missing returns don't return the CSV
        bad = ma.masked_where(returns.data==0.0, returns.data)
        io = numpy.sum(ma.getmaskarray(bad[:,-1]), axis=None)
        if io > 0.9 * returns.data.shape[0]:
            self.log.info('Too many zero or missing returns (%d of %s)',
                    int(io), returns.data.shape[0])
            return None

        # Compute the cross-sectional volatility
        outlierClass = Outliers.Outliers()
        clippedReturns = ma.filled(outlierClass.twodMAD(returns.data[:,-1]), 0.0)
        crossSectionalStdDev = Utilities.mlab_std(clippedReturns, axis=0)
        crossSectionalStdDev = ma.masked_where(crossSectionalStdDev==0.0, crossSectionalStdDev)

        self.log.info('Market volatility for %s on %s: %.8f',
                rmg.description, currDate, crossSectionalStdDev)
        return float(crossSectionalStdDev)

class RMGMarketPortfolio(ModelDBSource):
    """Class to create the market portfolio for risk model groups.
    """
    def __init__(self, connections, gp=None):
        ModelDBSource.__init__(self, connections)
        self.marketDB = connections.marketDB
        self.tradingRMGs = gp.tradingRMGs
        self.notInRiskModels = hasattr(gp, 'notInRiskModels') and gp.notInRiskModels
        self.override = hasattr(gp, 'override') and gp.override
        self.debugOutput = hasattr(gp, 'verbose') and gp.verbose
        self.modelDB.setMarketCapCache(40)
        self.modelDB.createCurrencyCache(connections.marketDB)
        self.indexSelector = MarketIndex.MarketIndexSelector(
                                        self.modelDB, self.marketDB,
                                        forceRun=self.override)
        clsFamily = self.marketDB.getClassificationFamily('REGIONS')
        assert(clsFamily is not None)
        clsMembers = dict([(i.name, i) for i in self.marketDB.\
                            getClassificationFamilyMembers(clsFamily)])
        clsMember = clsMembers.get('HomeCountry', None)
        assert(clsMember is not None)
        self.homeCountryCls = clsMember
        self.clsData = None # cache for home country data
        rmgs = self.modelDB.getAllRiskModelGroups(inModels=False)
        self.rmgMap = dict((rmg.rmg_id, rmg) for rmg in rmgs)
        self.debugOutput = hasattr(gp, 'verbose') and gp.verbose
    
    def getBulkData(self, dateList, rmgIdList):
        self.log.debug('RMGMarketPortfolio.getBulkData')
        retvals = numpy.empty((len(dateList), len(rmgIdList)), dtype=object)
        for (dIdx, date) in enumerate(dateList):
            for (rIdx, rmg) in enumerate(rmgIdList):
                if date.isoweekday() > 5 and rmg not in self.tradingRMGs[date]:
                    # skip non-trading weekends
                    continue
                rmgObj = self.rmgMap[rmg]
                rval = Utilities.Struct()
                rval.valuePairs = self.computeRMGMarketPortfolio(rmgObj, date)
                if rval.valuePairs is not None:
                    retvals[dIdx, rIdx] = rval
        return retvals
    
    def getRMGExpandedUniverse(self, rmg, currDate, allowExpansion=False):
        """Returns a list of all non-cash sub-issues active in the
        given risk model group plus all sub-issues mapped to it
        as its home country 1.
        """
        rmgSubIssues = self.modelDB.getActiveSubIssues(
                                rmg, currDate, inModels=(self.notInRiskModels==False))
        if (self.clsData is None or self.clsData[0] != currDate) and allowExpansion:
            rmgSubIssues = set(rmgSubIssues)
            allSubIssues = self.modelDB.getAllActiveSubIssues(currDate)
            clsRevision = self.marketDB.\
                getClassificationMemberRevision(self.homeCountryCls, currDate)
            clsData = self.modelDB.getMktAssetClassifications(
                clsRevision, allSubIssues, currDate, self.marketDB)
            self.clsData = (currDate, clsData)
            listedAbroad = [sid for (sid,cls) in self.clsData[1].items() \
                            if sid not in rmgSubIssues and \
                            cls.classification.code==rmg.mnemonic]
            self.log.debug('Adding %d %s assets trading outside of %s', 
                        len(listedAbroad), rmg.description, rmg.mnemonic)
            rmgSubIssues.update(listedAbroad)
        return [s for s in rmgSubIssues if not s.isCashAsset()]
    
    def computeRMGMarketPortfolio(self, rmg, currDate):
        self.log.info('Processing market portfolio for %s on %s',
                      rmg.description, currDate)
        if rmg.description == 'China':
            subIssues = self.getRMGExpandedUniverse(rmg, currDate, True)
            sidRMGPairs = self.modelDB.getSubIssueRiskModelGroupPairs(
                                currDate, restrict=subIssues)
            subIssues = [sid for (sid,r) in sidRMGPairs \
                                if r.mnemonic in('HK','CN')]
            assetCurrMap = self.modelDB.getTradingCurrency(currDate, 
                                subIssues, self.marketDB, returnType='code')
            subIssues = [sid for sid in subIssues \
                                if assetCurrMap.get(sid) in ('HKD','USD')]
        elif rmg.description == 'Domestic China':
            rmgList = self.modelDB.getAllRiskModelGroups()
            chinaRMG = [r for r in rmgList if r.description=='China'][0]
            subIssues = self.getRMGExpandedUniverse(chinaRMG, currDate)
        elif rmg.description == 'United States Small Cap':
            rmgList = self.modelDB.getAllRiskModelGroups()
            usRMG = [r for r in rmgList if r.description=='United States'][0]
            subIssues = self.getRMGExpandedUniverse(usRMG, currDate)
        else:
            subIssues = self.getRMGExpandedUniverse(rmg, currDate)
        if len(subIssues) == 0:
            self.log.info('No active sub-issues, skipping.')
            return None
        marketPortfolio = self.indexSelector.createMarketIndex(
            rmg.mnemonic, currDate, self.modelDB, self.marketDB, subIssues)
        assert(marketPortfolio is not None)
        if self.debugOutput:
            for (sid, wt) in marketPortfolio.data:
                logging.info('%s BM Data: %s, %s', rmg.mnemonic, sid, wt)
        return marketPortfolio.data

class ReturnsTimingAdjustments(ModelDBSource):
    """Class to create the market portfolio for risk model groups.
    """
    # Mapping timing ID to syncMarkets and markets
    supportedTimings = {1: (['US'], [
                'AE','AR','AT','AU','BE','BG','BH','BR','BW','CA','CH','CL',
                'CN','CO','CY','CZ','DE','DK','EE','EG','ES','FI','FR','GB',
                'GR','HK','HR','HU','ID','IE','IL','IN','IS','IT','JO','JP',
                'KR','KW','LK','LT','LU','LV','MA','MU','MX','MY','NL','NO',
                'NZ','OM','PE','PH','PK','PL','PT','QA','RO','RU','SE','SG',
                'SI','SK','TH','TR','TW','US','VE','ZA',
                'KZ','KE','LB','NG','SA','TN','UA','VN','MT','ZM','RS','EC',
                'BD','NA','GH','JM','TT','MK','BA','ME','CI','MW','PS','TZ',
                'ZW','UG',
                ])}

    def __init__(self, connections, timingId, gp=None):
        timingId = int(timingId)
        if timingId not in self.supportedTimings:
            raise KeyError('Unsupported timing ID: %d' % timingId)
        ModelDBSource.__init__(self, connections)
        self.timingId = timingId
        self.notInRiskModels = hasattr(gp, 'notInRiskModels') and gp.notInRiskModels
        rmgs = self.modelDB.getAllRiskModelGroups(inModels=(self.notInRiskModels==False))
        rmgMap = dict([(rmg.mnemonic, rmg) for rmg in rmgs])
        self.syncMarkets = [rmgMap[mnemonic] for mnemonic
                    in self.supportedTimings[self.timingId][0]
                    if mnemonic in rmgMap]
        self.rmgs = [rmgMap[mnemonic] for mnemonic
                     in self.supportedTimings[self.timingId][1]
                     if mnemonic in rmgMap]
        self.rmgMap = dict((rmg.rmg_id, rmg) for rmg in self.rmgs)
        self.override = hasattr(gp, 'override') and gp.override
        self.debugOutput = hasattr(gp, 'verbose') and gp.verbose

    def getBulkData(self, dateList, rmgIdList):
        self.log.debug('ReturnsTimingAdjustments.getBulkData')
        retvals = numpy.empty((len(dateList), len(rmgIdList)), dtype=object)
        for (dIdx, date) in enumerate(dateList):
            if date.isoweekday() > 5:
                self.log.info('Saturday/Sunday, skipping.')
                continue
            proxyFill, rmgTimings = ProcessReturns.compute_returns_timing_adjustments(
                    self.rmgs, date, self.modelDB, self.syncMarkets, debugReporting=self.debugOutput)

            for (rIdx, rmg) in enumerate(rmgIdList):
                rmgObj = self.rmgMap.get(rmg)
                if rmgObj is not None and ((rmgObj in rmgTimings) or \
                        (rmgObj in proxyFill)):
                    rval = Utilities.Struct()
                    rval.rmg_id = rmg
                    rval.value = rmgTimings.get(rmgObj, None)
                    rval.proxy = proxyFill.get(rmgObj, None)
                    retvals[dIdx, rIdx] = rval
        return retvals

class ReturnsTimingAdjustmentsLegacy(ModelDBSource):
    """Class to create the market portfolio for risk model groups.
    """
    # Mapping timing ID to syncMarkets and markets
    supportedTimings = {1: (['US'], [
                'AE','AR','AT','AU','BE','BG','BH','BR','BW','CA','CH','CL',
                'CN','CO','CY','CZ','DE','DK','EE','EG','ES','FI','FR','GB',
                'GR','HK','HR','HU','ID','IE','IL','IN','IS','IT','JO','JP',
                'KR','KW','LK','LT','LU','LV','MA','MU','MX','MY','NL','NO',
                'NZ','OM','PE','PH','PK','PL','PT','QA','RO','RU','SE','SG',
                'SI','SK','TH','TR','TW','US','VE','ZA',
                'KZ','KE','LB','NG','SA','TN','UA','VN','MT','ZM','RS','EC',
                'BD','NA','GH','JM','TT',
                'MK','BA','ME','CI','MW','PS','TZ','ZW','UG',
                ])}

    def __init__(self, connections, timingId, gp=None):
        timingId = int(timingId)
        if timingId not in self.supportedTimings:
            raise KeyError('Unsupported timing ID: %d' % timingId)
        ModelDBSource.__init__(self, connections)
        self.timingId = timingId
        self.notInRiskModels = hasattr(gp, 'notInRiskModels') and gp.notInRiskModels
        rmgs = self.modelDB.getAllRiskModelGroups(inModels=(self.notInRiskModels==False))
        rmgMap = dict([(rmg.mnemonic, rmg) for rmg in rmgs])
        self.syncMarkets = [rmgMap[mnemonic] for mnemonic
                            in self.supportedTimings[self.timingId][0]
                            if mnemonic in rmgMap]
        self.rmgs = [rmgMap[mnemonic] for mnemonic
                     in self.supportedTimings[self.timingId][1]
                     if mnemonic in rmgMap]
        self.rmgMap = dict((rmg.rmg_id, rmg) for rmg in self.rmgs)
        self.override = hasattr(gp, 'override') and gp.override
        self.debugOutput = hasattr(gp, 'verbose') and gp.verbose

    def getBulkData(self, dateList, rmgIdList):
        self.log.debug('ReturnsTimingAdjustments.getBulkData')
        retvals = numpy.empty((len(dateList), len(rmgIdList)), dtype=object)
        for (dIdx, date) in enumerate(dateList):
            if date.isoweekday() > 5:
                self.log.info('Saturday/Sunday, skipping.')
                continue
            rmgTimings = RetSync.synchronise_returns(
                    self.rmgs, date, self.modelDB, self.syncMarkets, debugReporting=self.debugOutput)

            for (rIdx, rmg) in enumerate(rmgIdList):
                rmgObj = self.rmgMap.get(rmg)
                if rmgObj is not None and rmgObj in rmgTimings:
                    rval = Utilities.Struct()
                    rval.rmg_id = rmg
                    rval.value = rmgTimings[rmgObj]
                    retvals[dIdx, rIdx] = rval
        return retvals

class MdlFundamentalDataItem(ModelDBSource):
    """Class to retrieve the values for a fundamental data item from
    MarketDB. The item names are resolved to codes
    via the meta_codes table.
    """
    def __init__(self, connections, tableName):
        ModelDBSource.__init__(self, connections)
        self.tableName = tableName
        self.DATEINCR = 30
        self.INCR = 200
        self.argList = [('sid%d' % i) for i in range(self.INCR)]
        self.dateArgList = [('date%d' % i) for i in range(self.DATEINCR)]
        self.updatedRecordsQuery = """
          SELECT t.sub_issue_id, t.dt, t.eff_dt
          FROM %(tablename)s t
          WHERE t.sub_issue_id IN (%(sids)s)
          AND t.eff_dt IN (%(dateargs)s) AND t.item_code=:item_code""" % {
            'dateargs': ','.join([':%s' % darg for darg
                                  in self.dateArgList]),
            'sids': ','.join([':%s' % sidarg for sidarg in self.argList]),
            'tablename': self.tableName + '_active'}
        self.defaultDict = dict([(arg, None) for arg
                                 in self.argList + self.dateArgList])
        marketDB = connections.marketDB
        mktTableName = {
            'sub_issue_fund_currency': 'asset_dim_fund_currency',
            'sub_issue_fund_number': 'asset_dim_fund_number'
            }[self.tableName]
        marketDB.dbCursor.execute("""SELECT name, id, code_type
           FROM meta_codes
           WHERE code_type in ('%(tablename)s:item_code')""" % {
            'tablename': mktTableName })
        self.itemMap = dict([(name, (id, type)) for (name, id, type)
                             in marketDB.dbCursor.fetchall()])
        
    def getBulkData(self, updateTuples, item):
        self.log.debug('MdlFundamentalDataItem.getBulkData')
        retvals = numpy.empty((len(updateTuples),), dtype=object)
        return retvals
    
    def findUpdates(self, dateList, sidList, item):
        """Returns a list of (sub-issue, Axioma ID string, dt, eff_dt)
        tuples which contains all the items that are currently in the
        ModelDB with the given eff_dt so that they can be updated.
        """
        self.log.debug('MdlFundamentalDataItem.findUpdates')
        (axids, axidSubIssueMap, sidAxidMap) = self.modelDBHelper.buildAxiomaIdHistory(sidList)
        sidStrs = [sid.getSubIDString() for sid in sidList]
        itemCode = self.itemMap[item][0]
        updates = set()
        # Find all item/date combinations that have eff_dt in the
        # given list
        for dateChunk in listChunkIterator(dateList, self.DATEINCR):
            updateDateDict = dict(zip(self.dateArgList, dateChunk))
            for sidStrChunk in listChunkIterator(sidStrs, self.INCR):
                updateDict = dict(zip(self.argList, sidStrChunk))
                myDict = dict(self.defaultDict)
                myDict['item_code'] = itemCode
                myDict.update(updateDict)
                myDict.update(updateDateDict)
                self.modelDB.dbCursor.execute(
                    self.updatedRecordsQuery, myDict)
                for (sid, dt, effDt) \
                        in self.modelDB.dbCursor.fetchall():
                    sid = ModelDB.SubIssue(string=sid)
                    dt = dt.date()
                    effDt = effDt.date()
                    updates.add((sid, dt, effDt))
        return updates

class MdlFundamentalCurrencyItem(MdlFundamentalDataItem):
    def __init__(self, connections):
        MdlFundamentalDataItem.__init__(
            self, connections, 'sub_issue_fund_currency')

    
''' Generic data item representing Fundamental or Estimate data '''
class DataItem(MarketDBSource):
    """Class to retrieve the values for a fundamental data item from
    MarketDB. The item names are resolved to codes
    via the meta_codes table.
    """
    INCR = 200
    DATEINCR = 30
    
    def __init__(self, connections, tableName, dataFields, itemCodeFieldID = 'item_code'):
        MarketDBSource.__init__(self, connections)
        self.tableName = tableName
        #self.DATEINCR = 30
        #self.INCR = 200
        
        self.itemCodeFieldID = itemCodeFieldID
        self.sidRanges = connections.sidRanges
        self.argList = [('aid%d' % i) for i in range(self.INCR)]
        self.dateArgList = [('date%d' % i) for i in range(self.DATEINCR)]
        self.query = """SELECT %(dataFields)s FROM %(tablename)s t
          WHERE t.axioma_id=:axid AND t.item_code=:item_code
          AND eff_del_flag = 'N'
          AND t.dt=:dt AND t.eff_dt=(SELECT max(eff_dt) FROM %(tablename)s t2
             WHERE t2.axioma_id=t.axioma_id AND t.item_code=t2.item_code
             AND t2.dt=t.dt AND eff_dt <= :effDt)""" % {
            'tablename': self.tableName + '_active',
            'dataFields': dataFields}
        self.updatedRecordsQuery = """
          SELECT t.axioma_id, t.dt, t.eff_dt
          FROM %(tablename)s t
          WHERE t.axioma_id IN (%(axId)s)
          AND t.eff_dt IN (%(dateargs)s) AND t.item_code=:item_code""" % {
            'dateargs': ','.join([':%s' % darg for darg
                                  in self.dateArgList]),
            'axId': ','.join([':%s' % axIdarg for axIdarg
                              in self.argList]),
            'tablename': self.tableName + '_active'}
        self.historyUpdateQuery = """
          SELECT t.axioma_id, t.dt
          FROM %(tablename)s t
          WHERE t.axioma_id = :axid
          AND t.eff_dt <= :dt AND t.item_code=:item_code""" % {
            'tablename': self.tableName + '_active'}
        self.futureUpdateQuery = """
          SELECT t.axioma_id, t.dt
          FROM %(tablename)s t
          WHERE t.axioma_id = :axid and dt >= :dt
          AND t.eff_dt <= :dt AND t.item_code=:item_code""" % {
            'tablename': self.tableName + '_active'}
        self.defaultDict = dict([(arg, None) for arg
                                 in self.argList + self.dateArgList])
        
        #  Point to correct table: Estimate item_codes mapped to ASSET_DIM_ESTI_CURRENCY table only!
        itemsTableName = 'asset_dim_esti_currency' if '_ESTI' in self.tableName.upper()\
                          else self.tableName            
        self.marketDB.dbCursor.execute("""SELECT name, id, code_type
           FROM meta_codes
           WHERE code_type in ('%(tablename)s:item_code')""" % {'tablename': itemsTableName})
        self.itemMap = dict([(name, (id, type)) for (name, id, type)
                             in self.marketDB.dbCursor.fetchall()])
    
    def getBulkData(self, updateTuples, item):      
        self.log.debug('DataItem.getBulkData')
        retvals = numpy.empty((len(updateTuples),), dtype=object)
        itemCode = self.itemMap[item][0] if item in self.itemMap else self.itemMap[item.upper()][0]
        sidList = list(set([sid for (sid, dt, effDt) in updateTuples]))
        (axids, axidSubIssueMap, sidAxidMap) = \
                self.modelDBHelper.buildAxiomaIdHistory(sidList)
        for (tIdx, (sid, dt, effDt)) in enumerate(updateTuples):
            axidHistory = sidAxidMap.get(sid, list())
            isActive = False
            for axidRecord in axidHistory:
                if axidRecord.fromDt <= dt and axidRecord.thruDt > dt:
                    # Exact match
                    isActive = True
                    axidStr = axidRecord.marketdb_id
                    break
                if axidRecord.thruDt > dt:
                    sidFromDt = self.sidRanges[sid][0]
                    if sidFromDt == axidRecord.fromDt:
                        # First Axioma ID mapped to sub-issue, so pretend it also mapped earlier
                        isActive = True
                        axidStr = axidRecord.marketdb_id
                        break
            if isActive:
                valDict = { 'axid' : axidStr, 'dt' : dt, 'effDT' : effDt, 'item_code' :itemCode }
                self.marketDB.dbCursor.execute(self.getBulkDataQuery(), valDict)
                r = self.marketDB.dbCursor.fetchall()
                assert(len(r) <= 1)
                if len(r) > 0:
                    retvals[tIdx] = self._createValueStruct(r[0])
        return retvals
    
    #@do_line_profile
    def findUpdates(self, dateList, sidList, item):
        """Returns a list of (sub-issue, Axioma ID string, dt, eff_dt)
        tuples which contains all the items that were updated on
        the dates in the dateList.
        """
        self.log.debug('DataItem.findUpdates')
        (axids, axidSubIssueMap, sidAxidMap) = self.modelDBHelper.buildAxiomaIdHistory(
            sidList)
        aidStrs = [aid.getIDString() for aid in axids]
        sidSet = set(sidList)
        itemCode = self.itemMap[item][0] if item in self.itemMap else self.itemMap[item.upper()][0]
        valueDict = dict()
        updates = set()
        # For sub-issues that start a new axioma_id on dt, add all item/date
        # combinations with eff_dt <= dt to the update list.
        # The changes will be effective in the sub-issue as of the from date.
        #
        # For sub-issues that end an axioma_id on dt, add all item/date
        # combinations with eff_dt <= dt and item dt >= dt to the update
        # list so that those future records can be removed/updated.
        # The changes will be effective in the sub-issue as of the thru date.
        dateSet = set(dateList)
        for (axid, subHistory) in axidSubIssueMap.items():
            for sidVal in subHistory:
                if sidVal.fromDt in dateSet:
                    fromDt = sidVal.fromDt
                    sidAxids = sidAxidMap[sidVal.sub_issue]
                    firstAxid = sorted(
                        sidAxids, key=operator.attrgetter('fromDt'))[0].marketdb_id
                    valueDict = { 'axid': axid, 'dt': fromDt, 'item_code' : itemCode }
                    self.marketDB.dbCursor.execute(self.getHistoryUpdateQuery(), valueDict)
                    for (axid, dt) in self.marketDB.dbCursor.fetchall():
                        dt = dt.date()
                        if dt < sidVal.thruDt and (dt >= sidVal.fromDt
                                                   or axid == firstAxid):
                            updates.add((sidVal.sub_issue, dt, fromDt))
                elif sidVal.thruDt in dateSet:
                    thruDt = sidVal.thruDt
                    valueDict = { 'axid': axid, 'dt': thruDt, 'item_code': itemCode }
                    self.marketDB.dbCursor.execute(self.getFutureUpdateQuery(), valueDict)
                    for (axid, dt) in self.marketDB.dbCursor.fetchall():
                        dt = dt.date()
                        updates.add((sidVal.sub_issue, dt, thruDt))
        
        # Find all item/date combinations that have eff_dt in the given list
        for dateChunk in listChunkIterator(dateList, self.DATEINCR):
            updateDateDict = dict(zip(self.dateArgList, dateChunk))
            for aidStrChunk in listChunkIterator(aidStrs, self.INCR):
                updateDict = dict(zip(self.argList, aidStrChunk))
                myDict = dict(self.defaultDict)
                myDict['item_code'] = itemCode
                myDict.update(updateDict)
                myDict.update(updateDateDict)
                self.marketDB.dbCursor.execute(self.getUpdatedRecorsQuery(), myDict)
                for (axid, dt, effDt) in self.marketDB.dbCursor.fetchall():
                    dt = dt.date()
                    effDt = effDt.date()
                    sidHistory = axidSubIssueMap[axid]
                    for sidRecord in sidHistory:
                        if sidRecord.sub_issue not in sidSet:
                            print('Skipping sub-issue not in list', sidRecord, axid)
                            continue
                        if sidRecord.fromDt <= dt and sidRecord.thruDt > dt:
                            # Update matches active axid->sid mapping
                            updates.add((sidRecord.sub_issue, dt, effDt))
                        elif sidRecord.thruDt > dt:
                            sidAxids = sidAxidMap[sidRecord.sub_issue]
                            firstAxid = sorted(
                                sidAxids, key=operator.attrgetter('fromDt'))[0].marketdb_id
                            if axid == firstAxid:
                                # Update is before active axid->sid mapping
                                # but this is the first axid for this sid
                                # so we assume it also mapped before
                                updates.add((sidRecord.sub_issue, dt, effDt))
        return updates

    def getBulkDataQuery(self):
        return self.query
    
    def getUpdatedRecorsQuery(self):
        return self.updatedRecordsQuery
           
    def getHistoryUpdateQuery(self):
        return self.historyUpdateQuery
    
    def getFutureUpdateQuery(self):
        return self.futureUpdateQuery

class CurrencyDataItem(DataItem):
    def __init__(self, connections, tableName, dataFields):
        DataItem.__init__(
            self, connections, tableName, dataFields)
    
    def _createValueStruct(self, values):
        rval = Utilities.Struct()
        (value, ccyId) = values
        rval.value = float(value)
        rval.currency_id = ccyId
        return rval
    
class FundamentalCurrencyItem(CurrencyDataItem):
    def __init__(self, connections):
        CurrencyDataItem.__init__(
            self, connections, 'asset_dim_fund_currency', 'value, currency_id')

class FundamentalDataItem(DataItem):
    SELECT_XDATA_FIELDS = 'VALUE, CURRENCY_ID, FISCAL_YEAR_END'
    DATA_TABLE_ID = 'ASSET_DIM_FUNDAMENTAL_DATA'
    ITEM_CODE_TABLE = ''
    
    def __init__(self, connections):
        DataItem.__init__(self, connections, FundamentalDataItem.DATA_TABLE_ID, 
                          FundamentalDataItem.SELECT_XDATA_FIELDS, 'item_code_id')
        self.dataFields = FundamentalDataItem.SELECT_XDATA_FIELDS
        self.tableName = FundamentalDataItem.DATA_TABLE_ID
        self.sidRanges = connections.sidRanges
        self.DATEINCR = 30
        self.INCR = 200
        
        self.argList = [('aid%d' % i) for i in range(self.INCR)]
        self.dateArgList = [('date%d' % i) for i in range(self.DATEINCR)]
        self.query = """SELECT %(dataFields)s FROM %(tablename)s t
          WHERE t.axioma_id=:axid AND t.item_code_id=:item_code
          AND eff_del_flag = 'N'
          AND t.dt=:dt AND t.eff_dt=(SELECT max(eff_dt) FROM %(tablename)s t2
             WHERE t2.axioma_id=t.axioma_id AND t.item_code_id=t2.item_code_id
             AND t2.dt=t.dt AND eff_dt <= :effDt)""" % {
            'tablename': self.tableName + '_ACT' , 'dataFields': self.dataFields}
        
        self.updatedRecordsQuery = """
          SELECT t.axioma_id, t.dt, t.eff_dt
          FROM %(tablename)s t
          WHERE t.axioma_id IN (%(axId)s)
          AND t.eff_dt IN (%(dateargs)s) AND t.item_code_id=:item_code""" % {
            'dateargs': ','.join([':%s' % darg for darg
                                  in self.dateArgList]),
            'axId': ','.join([':%s' % axIdarg for axIdarg
                              in self.argList]),
            'tablename': self.tableName + '_ACT'}
        
        self.historyUpdateQuery = """
          SELECT t.axioma_id, t.dt
          FROM %(tablename)s t
          WHERE t.axioma_id = :axid
          AND t.eff_dt <= :dt AND t.item_code_id=:item_code""" % {
            'tablename': self.tableName + '_ACT'}
          
        self.futureUpdateQuery = """
          SELECT t.axioma_id, t.dt
          FROM %(tablename)s t
          WHERE t.axioma_id = :axid and dt >= :dt
          AND t.eff_dt <= :dt AND t.item_code_id=:item_code""" % {
            'tablename': self.tableName + '_ACT'}
          
        itemsInfoQuery = '''SELECT AXIOMA_CODE_NAME, CODE_ID, XPSFEED_MEASURE 
                            from XPSFEED_MEASURE_AXIOMA_CODE'''
        self.defaultDict = dict([(arg, None) for arg
                                 in self.argList + self.dateArgList])
        
        #--- Populate Dictionary of ItemCode ID to Measure Names         
        self.marketDB.dbCursor.execute(itemsInfoQuery)
        self.itemMap = dict([(name, (codeId, xpsMeasure)) for (name, codeId, xpsMeasure) 
                                 in self.marketDB.dbCursor.fetchall()])
        
    def _createValueStruct(self, values):
        rval = Utilities.Struct()
        (value, ccyId, fye) = values
        rval.value = float(value)
        rval.currency_id = ccyId
        rval.fiscal_year_end = fye
        return rval

class ExpressoDataItem(DataItem):
    SELECT_XDATA_FIELDS = 'VALUE, CURRENCY_ID, FISCAL_YEAR_END'
    DATA_TABLE_ID = 'ASSET_DIM_EXPRESSO_DATA'
    ITEM_CODE_TABLE = ''
    
    def __init__(self, connections):
        DataItem.__init__(self, connections, ExpressoDataItem.DATA_TABLE_ID, 
                          FundamentalDataItem.SELECT_XDATA_FIELDS, 'item_code_id')
        self.dataFields = FundamentalDataItem.SELECT_XDATA_FIELDS
        self.tableName = FundamentalDataItem.DATA_TABLE_ID
        self.sidRanges = connections.sidRanges
        self.DATEINCR = 30
        self.INCR = 200
                   
        itemsInfoQuery = '''SELECT AXIOMA_CODE_NAME, CODE_ID, XPSFEED_MEASURE 
                            from XPSFEED_MEASURE_AXIOMA_CODE'''
        self.defaultDict = dict([(arg, None) for arg
                                 in self.argList + self.dateArgList])
        
        #--- Populate Dictionary of ItemCode ID to Measure Names         
        self.marketDB.dbCursor.execute(itemsInfoQuery)
        self.itemMap = dict([(name, (codeId, xpsMeasure)) for (name, codeId, xpsMeasure) 
                                 in self.marketDB.dbCursor.fetchall()])
    
    def _createValueStruct(self, values):
        rval = Utilities.Struct()
        (value, ccyId, fye) = values
        rval.value = float(value)
        rval.currency_id = ccyId
        rval.fiscal_year_end = fye
        return rval
    
    #---------------------------------#
    #      Override Query Methods     |
    #---------------------------------#
    def getBulkDataQuery(self):
        return """SELECT %(dataFields)s FROM %(tablename)s t
                  WHERE t.axioma_id=:axid AND t.ITEM_CODE_ID = :item_code AND eff_del_flag = 'N'
                       AND t.dt=:dt AND t.eff_dt=(SELECT max(eff_dt) FROM %(tablename)s t2
                           WHERE t2.axioma_id = t.axioma_id AND t.ITEM_CODE_ID = t2.ITEM_CODE_ID AND t2.dt = t.dt AND eff_dt <= :effDt)""" % \
                           { 'tablename': ExpressoDataItem.DATA_TABLE_ID + '_ACTIVE', 'dataFields': FundamentalDataItem.SELECT_XDATA_FIELDS }
    
    def getUpdatedRecorsQuery(self):
        return """SELECT t.axioma_id, t.dt, t.eff_dt FROM %(tablename)s t
                        WHERE t.axioma_id IN (%(axId)s) AND t.eff_dt IN (%(dateargs)s) AND t.ITEM_CODE_ID = :item_code""" % \
                        { 'dateargs': ','.join([':%s' % darg for darg in self.dateArgList]),
                          'axId': ','.join([':%s' % axIdarg for axIdarg in self.argList]),
                          'tablename': ExpressoDataItem.DATA_TABLE_ID  + '_ACTIVE'}
    
    def getHistoryUpdateQuery(self):
        return """ SELECT t.axioma_id, t.dt FROM %(tablename)s t
                   WHERE t.axioma_id = :axid AND t.eff_dt <= :dt AND t.ITEM_CODE_ID = :item_code""" % \
                { 'tablename': ExpressoDataItem.DATA_TABLE_ID  + '_ACTIVE' }
       
    def getFutureUpdateQuery(self):
        return """SELECT t.axioma_id, t.dt FROM %(tablename)s t
                  WHERE t.axioma_id = :axid and dt >= :dt AND t.eff_dt <= :dt AND t.ITEM_CODE_ID = :item_code""" % \
                  {'tablename': ExpressoDataItem.DATA_TABLE_ID  + '_ACTIVE' }

class XpressfeedFundamentalDataItem(CurrencyDataItem):
    def __init__(self, connections):
        CurrencyDataItem.__init__(
            self, connections, 'asset_dim_fund_xpsfeed', 'value, currency_id')
        
class EstimateCurrencyItem(CurrencyDataItem):
    def __init__(self, connections):
        CurrencyDataItem.__init__(
            self, connections, 'asset_dim_esti_currency', 'value, currency_id')

'''New: Handles Unadjusted IBES Estimate Data'''
class EstimateDataItem(MarketDBSource):
    MAX_ASSET_ID_COUNT = 300
    
    class DBDataProvider(VendorUtils.DefaultDBDataProvider):
        pass
    
    """Class to retrieve the values for a fundamental data item from
       MarketDB. The item names are resolved to codes via the meta_codes table.
    """
    def __init__(self, connections):
        MarketDBSource.__init__(self, connections)
        self.tableName = 'asset_dim_estimate_data'   # TODO: Revisit hard-coded value
        self.dataFields = 'value, currency_id'       # TODO: Revisit hard-coded value
        self.cachedValuesInfo_ = OneToManyDictionary()
        self.sidRanges = connections.sidRanges
        self.connections_ = connections
        self.cachedValues_ = {}
        self.DATEINCR = 30
        self.INCR = 500
    
        # 1-A Load AxiomaDB/MarketDB-ID to ModelDB-ID Mapping 
        self.axID2ModelDbIDHistoryMap_ = None
        self.modelID2AxIDHistoryMap_ = None
        
        # 1-B Sub-Issue Mapping
        self.subIssueID2ModelIDHistoryMap_ = None
        self.modelID2SubIssueIDHistoryMap_ = None
        
        # 2. Point to correct table: Estimate item_codes mapped to ASSET_DIM_ESTI_CURRENCY table only!
        itemsTableName = 'asset_dim_esti_currency' if '_ESTI' in self.tableName.upper() else self.tableName            
        self.marketDB.dbCursor.execute("""SELECT name, id, code_type FROM meta_codes
                                          WHERE code_type in ('%(tablename)s:item_code')""" %
                                          {'tablename': itemsTableName})
        self.itemMap = dict([(name, (id, type)) for (name, id, type)
                             in self.marketDB.dbCursor.fetchall()])
    
    def getBulkData(self, updateTuples, item):
        self.log.debug('EstimateDataItem.getBulkData')
        retvals = numpy.empty((len(updateTuples),), dtype=object)
        itemCode = self.itemMap[item][0]
        #sidList = list(set([sid for (sid, dataDT, effDT) in updateTuples]))
        #(axids, axidSubIssueMap, sidAxidMap) = self.modelDBHelper.buildAxiomaIdHistory(sidList)
        
        for (tIdx, (sid, dataDT, effDT)) in enumerate(updateTuples):
            # 1. See if data is contained in cache.
            estDataItem = self.cachedValues_.get((sid, itemCode, dataDT, effDT))
            
            if estDataItem is not None:
                values = (estDataItem.value_, estDataItem.currencyID_)
                retvals[tIdx] = self._createValueStruct(values)
            else:
                # N.B. This can occur and it's not an issue: Ex. K9JDN7CAZ5 / DZZLRFTLR211 => EffDT = 19-MAY-2016
                #      Contains valid values for CPS, but asymetrically 3 (dataDTs) for 203 and 2 for 223:
                #      The missing dataDT (2018-12-31) for 223 is tagged as an error.
                logging.debug("Skipping invalid value not found in cache. ")
                logging.debug("SubIssue:%s DataDT:%s EffDT:%s ItemCode:%s"%(sid, dataDT, effDT, itemCode))
        
        return retvals
    
    def findUpdates(self, effDateList, sidList, item):
        """Returns a list of (sub-issue, Axioma ID string, dt, eff_dt)
           tuples which contains all the items that were updated on
           the dates in the dateList.
        """
        self.log.debug('EstimateDataItem.findUpdates')
        maxEDT = effDateList[-1]     # TODO if max == min, decrease min by 30 days
        minEDT = effDateList[0] if len(effDateList) > 1 else maxEDT - (30 * MktDBUtils.ONE_DAY) # Lookback 30 days
        assert(minEDT <= maxEDT)
        
        #1. Populate Cache (if needed)
        self._populateCache(sidList, item, minEDT, maxEDT)
        
        #2. Retrieve updates from cache:
        return self._getUpdatesFromCache(sidList, item, effDateList)
     
    #------------------------------#
    #    Private Utility Methods   #
    #------------------------------#
    def _populateCache(self, sidList, item, minEDT, maxEDT):
        # 1. Do only if not already processed: 
        itemCode = self.itemMap[item][0]
        if self._doCaching(itemCode, minEDT, maxEDT):
            # diffs = self._getDatabaseDiffs(itemCode, minEDT, maxEDT)
            diffs = self._getDatabaseDifferences(itemCode, minEDT, maxEDT, sidList)
            
            if len(diffs) > 0:
                logging.info("Investigating %s differences for measure:%s. Updates may be present."%(len(diffs), itemCode))
                # Flush Cache: For Performance enhancement
                self.cachedValues_ = {}
            
            for record in diffs:
                axiomaID = record[0]
                dataDT = record[2]
                value = record[3]
                currencyID = record[4]
                effDT = record[5]
                REF = record[8]
            
                # Get Corresponding SubIssue instance(s): Need ModelDBID (from AxiomaID) to SubIssueID (from ModelDBID)
                aidHistoryMap = self._getAxiomaToModelDbIDHistoryMap().get(axiomaID)
                modelDbID = aidHistoryMap.getVendorIDForDate(effDT) if aidHistoryMap is not None else None
                #subIssueID = "%s11"%modelDbID   # N.B. Generally, this is true
                aidHistoryMapSID = self._getModelDbIDToSubIssueIDHistoryMap().get(modelDbID)
                subIssueID = aidHistoryMapSID.getVendorIDForDate(effDT) if aidHistoryMapSID is not None else None
                if modelDbID is not None:
                    # RSK_3643: SubIssueID can be None (e.g. Estimate effDT is before Axioma's Start Date for sub-issue)
                    subIssueInstance = ModelDB.SubIssue(subIssueID) if subIssueID is not None else ModelDB.SubIssue('%s11'%modelDbID)
                    estimateDataItem = EstimateDataItem.EstimateRecord(subIssueInstance, itemCode, dataDT, value, currencyID, effDT, REF)
                    self.cachedValues_.update({ (subIssueInstance, itemCode, dataDT, effDT) : estimateDataItem })
                 
                # 5. Update cached values info: To determine whether a measure + effDTs have already been processed
                self.cachedValuesInfo_.append(itemCode, (minEDT, maxEDT))
    
    def _doCaching(self, itemCode, minEDT, maxEDT):
        return (itemCode not in self.cachedValuesInfo_) or \
               ((minEDT, maxEDT) not in self.cachedValuesInfo_.get(itemCode))
        
    def _getUpdatesFromCache(self, sidList, item, effDateList):
        #sidListStr = ["%s"%x.getSubIDString() for x in sidList] # Get Corresponding String IDs
        self.log.debug('EstimateDataItem._getUpdatesFromCache')
        
        updates = set()        
        # 1. Check first if there's any relevant data in cache; we might have done this already
        itemCode = self.itemMap[item][0]
        if len(self.cachedValues_) > 0 and (itemCode in self.cachedValuesInfo_):
            for (subIssue, itemCode, dataDT, effDT) in self.cachedValues_.keys():
                if (effDT.date() in effDateList and subIssue in sidList):    # N.B. This update applicable only if effDT is in effDTList
                    updates.add((subIssue, dataDT, effDT))
        
        return updates       
    
    # 1. Load AxiomaDB/MarketDB-ID to ModelDB-ID Mapping 
    def _getAxiomaToModelDbIDHistoryMap(self):
        if (self.axID2ModelDbIDHistoryMap_ == None):
            modelDbIDDP = VendorUtils.ModelDbIDHistoryMap(self.connections_)
            self.axID2ModelDbIDHistoryMap_ = modelDbIDDP.getAxiomaIDToModelDbIDHistoryMap()
        
        return self.axID2ModelDbIDHistoryMap_
    
    def _getModelDbIDToAxiomaIDHistoryMap(self):
        if (self.modelID2AxIDHistoryMap_ == None):
            modelDbIDDP = VendorUtils.ModelDbIDHistoryMap(self.connections_)
            self.modelID2AxIDHistoryMap_ = modelDbIDDP.getInverseAssetIDHistoryMap()
        
        return self.modelID2AxIDHistoryMap_
       
    def _getModelDbIDToSubIssueIDHistoryMap(self):
        if (self.modelID2SubIssueIDHistoryMap_ == None):
            subIssueIDDP = VendorUtils.SubIssueIDHistoryMap(self.connections_)
            #self.subIssueID2ModelIDHistoryMap_ = subIssueIDDP.getModelDbIDToSubIssueIDHistoryMap()
            self.modelID2SubIssueIDHistoryMap_ = subIssueIDDP.getInverseAssetIDHistoryMap()
        
        return self.modelID2SubIssueIDHistoryMap_
    
    def _getSubIssueIDsToAxiomaIDMap(self, subIssueList, dateIn):
        subIssueIDMap = {}
        for subIssue in subIssueList:
            subIssueID = subIssue.getSubIDString()
            logging.debug("Processing SubIssue: %s for Date:%s with ModelID:%s"%(subIssueID, dateIn.date(), subIssue.getModelID()))
            axiomaID = self._getCorrespondingAxiomaID(subIssueID, dateIn)
            if axiomaID is not None:
                subIssueIDMap.update({ axiomaID : subIssueID})
        return subIssueIDMap
        
    def _getFormattedValueForDB(self, dbValue, dType): 
        ORACLE_DATE_FORMAT = '%d-%b-%Y'
        if dType == MX_DT.DATE: 
            return MktDBUtils.getDateInstance(dbValue).strftime(ORACLE_DATE_FORMAT)
        else:
            return str(dbValue)
        
    def _getRecordEntry(self, dbType, secID, measureCode, dataDT, value, currencyID, effDT, REF):
        axiomaID = self._getAxiomaID(dbType, secID, effDT)          
        return ((axiomaID, measureCode, dataDT, effDT), EstimateDataItem.EstimateRecord(secID, measureCode, dataDT, 
                                                                                    value, currencyID, effDT, REF))
    def _getAxiomaID(self, dbType, secID, dateIn):
        return secID if (dbType == DATABASE_TYPE.MARKETDB) else self._getCorrespondingAxiomaID(secID, dateIn)
            
    def _getCorrespondingAxiomaID(self, subIssueID, dateIn):
        modelDbID = subIssueID[:10]
        modelID2AxIDHistMap = self._getModelDbIDToAxiomaIDHistoryMap()
        modelDBID2AxIDHistory = modelID2AxIDHistMap.get(modelDbID)
        return modelDBID2AxIDHistory.getVendorIDForDate(dateIn) if modelDBID2AxIDHistory  is not None else None
    
    def _getDatabaseDifferences(self, itemCode, minEDT, maxEDT, sidList):
        mktDBIDClause = ""
        mdlDBIDClause = ""
        
        if len(sidList) < EstimateDataItem.MAX_ASSET_ID_COUNT:
            logging.debug("Searching for sub-issues subset differences...")
            subIDStrList = ','.join('\'%s\''%sid.getSubIDString() for sid in sidList)
            mdlDBIDClause = ' AND SUB_ISSUE_ID in (%s)'%subIDStrList
            logging.debug("ModelDB ID List: %s"%subIDStrList)
            
            # 1. Convert SubIssueIDs to MarketDBIDs
            # 2. String mktDBIDs together
            # 3. String mdlDBIDs together
            mktDBIDStrList = ','.join('\'%s\''%self._getEffMarketDBID(sid, minEDT, maxEDT) for sid in sidList)
            mktDBIDClause = ' AND AXIOMA_ID in (%s)'%mktDBIDStrList
            logging.debug("MktDB-ID List: %s"%mktDBIDStrList)
        
        mktDBQuery = '''SELECT AXIOMA_ID, DT, EFF_DT, VALUE, CURRENCY_ID, REF, EFF_DEL_FLAG 
                             from ASSET_DIM_ESTIMATE_DATA_ACTIVE 
                             where EFF_DT >= :minEffDt and EFF_DT <= :maxEffDt and ITEM_CODE = :itemCode 
                             %s
                             ORDER BY AXIOMA_ID, DT, EFF_DT '''%(mktDBIDClause)
        
        mdlDBQuery = '''SELECT SUBSTR(SUB_ISSUE_ID, 1, 10) MODELDB_ID, DT, EFF_DT, VALUE, CURRENCY_ID, EFF_DEL_FLAG  
                             from MODELDB_GLOBAL.SUB_ISSUE_ESTIMATE_DATA_ACTIVE
                             where EFF_DT >= :minEffDt and EFF_DT <= :maxEffDt and ITEM_CODE = :itemCode 
                             %s
                             ORDER BY MODELDB_ID, DT, EFF_DT'''%(mdlDBIDClause)
        
        logging.debug("Executing Query: %s", mktDBQuery)
        dbCursor = self.marketDB.dbCursor
        dbQueryBindingValueMap = { 'minEffDt': minEDT, 'maxEffDt': maxEDT, 'itemCode': itemCode }
       
        #--- MarketDB Info ----#
        dbDP =  EstimateDataItem.DBDataProvider()
        mktDmx = dbDP.getDatabaseInfo(dbCursor, mktDBQuery, dbQueryBindingValueMap)
    
        #--- ModelDB Info ---#
        dmxPK = PrimaryKey('MODELDB_ID', 'DT', 'EFF_DT')
        dmx = dbDP.getDatabaseInfo(dbCursor, mdlDBQuery, dbQueryBindingValueMap)
        tensorDMX = Tensor(dmx.getColumnIDs(), dmxPK, dmx.getDataInfo()) if dmx.getRowCount() > 0 else dmx
        
        return self._gatherDiffs(itemCode, mktDmx, tensorDMX) if mktDmx.getRowCount() > 0 else []
    
    def _gatherDiffs(self, itemCode, mktDmx, tensorDMX):
        diffs = []
        dmxPK = tensorDMX.getPrimaryKey()
        for rowValues in mktDmx.getDataInfo():
            # 1. Identify records in MarketDB not transferred to ModelDB or that have chanced
            currencyID = rowValues[4]
            mktValue = rowValues[3]
            axiomaID = rowValues[0]
            dataDT = rowValues[1]
            effDT = rowValues[2]
            REF = rowValues[5]
            effDelFlag = 'N'
                
            # 2. Get Corresponding: ModelDBID from AxiomaID 
            aidHistoryMap = self._getAxiomaToModelDbIDHistoryMap().get(axiomaID)
            modelDbID = aidHistoryMap.getVendorIDForDate(effDT) if aidHistoryMap is not None else None
            modelDbID = aidHistoryMap.getVendorIDForDate(dataDT) if (aidHistoryMap is not None and modelDbID is None) else modelDbID
            
            if modelDbID is not None:
                #subIssueID = '%s11'%modelDbID
                #subIssueIDInstance = ModelDB.SubIssue(subIssueID)
                       
                pkValueInst = PrimaryKey.InstanceValue(dmxPK, modelDbID, dataDT, effDT) if tensorDMX.getRowCount() else None
                mdlValue = None if pkValueInst is None else tensorDMX.getColumnPKValue(pkValueInst, 'VALUE')
                
                # N.B. Consider a difference if value is None or if value is different 
                if mdlValue is None or abs(mdlValue - mktValue) > 0.0000001:        
                    diffs.append((axiomaID, itemCode, dataDT, mktValue, currencyID, effDT, effDelFlag, 1075, REF))
                    msg = "AxiomaID: %s ModelDbID: %s DataDT: %s EffDT: %s MktVal: %s MldVal: %s Item: %s " %\
                            (axiomaID, modelDbID, dataDT.strftime(MktDBUtils.SHORT_DATE_FORMAT), 
                             effDT.strftime(MktDBUtils.SHORT_DATE_FORMAT), mktValue, mdlValue, itemCode)
                    logging.debug("Found value difference in ModelDB. %s" %msg)
        
        return diffs
    
    def _getEffMarketDBID(self, subIssueID, fromEDT, thruEDT):
        logging.debug("Searching for MarketDBID for SID:%s"%subIssueID)
        axiomaID = self._getCorrespondingAxiomaID(subIssueID.getSubIDString(), fromEDT)  # TODO: Consider thruEDT?
        return axiomaID
    
    # RSK-4101: Binding to AxiomaID alone is not enough (some updates are missed); use EFF_DT also.
    # TODO:This query is very slow!
    def _getDatabaseDiffs(self, itemCode, minEDT, maxEDT):
#         query='''SELECT * from ASSET_DIM_ESTIMATE_DATA 
#                     where EFF_DT >= :minEffDt and EFF_DT <= :maxEffDt 
#                         and ITEM_CODE = :itemCode and AXIOMA_ID NOT IN (
#                           SELECT MARKETDB_ID from MODELDB_GLOBAL.ISSUE_MAP 
#                                 where MODELDB_ID in (
#                                     SELECT SUBSTR(SUB_ISSUE_ID, 1, 10) from MODELDB_GLOBAL.SUB_ISSUE_ESTIMATE_DATA
#                                         where EFF_DT >= :minEffDt and EFF_DT <= :maxEffDt and ITEM_CODE = :itemCode
#                                 )
#                         )                        
#               '''
        query='''SELECT DISTINCT * from ASSET_DIM_ESTIMATE_DATA 
                    where EFF_DT >= :minEffDt and EFF_DT <= :maxEffDt 
                        and ITEM_CODE = :itemCode AND (
                              AXIOMA_ID NOT IN (
                                 SELECT MARKETDB_ID from MODELDB_GLOBAL.ISSUE_MAP 
                                    where MODELDB_ID in (
                                         SELECT SUBSTR(SUB_ISSUE_ID, 1, 10) from MODELDB_GLOBAL.SUB_ISSUE_ESTIMATE_DATA
                                             where EFF_DT >= :minEffDt and EFF_DT <= :maxEffDt and ITEM_CODE = :itemCode
                                  )
                        
                             )
                             OR
                             EFF_DT NOT IN (
                                SELECT EFF_DT from MODELDB_GLOBAL.SUB_ISSUE_ESTIMATE_DATA
                                              where EFF_DT >= :minEffDt and EFF_DT <= :maxEffDt and ITEM_CODE = :itemCode
                             )
                        )                        
              '''
        cursor = self.marketDB.dbCursor   # TODO: This needs Revision -- One level higher!
        logging.debug("Executing Query:%s"%query)
        cursor.execute(query, {'minEffDt': minEDT, 'maxEffDt': maxEDT, 'itemCode': itemCode})
        return cursor.fetchall()
    
    def _createValueStruct(self, values):
        rval = Utilities.Struct()
        (value, ccyId) = values
        rval.value = float(value)
        rval.currency_id = ccyId
        return rval
    
    class EstimateRecord(object):
        def __init__(self, secID, measureCode, dataDT, value, currencyID, effDT, REF):
            # TODO: Input Validation 
            self.secID_ = secID
            self.value_ = value
            self.effDate_ = effDT
            self.dataDate_ = dataDT
            self.measureID_ = measureCode
            self.currencyID_ = currencyID
            self.REF_ = REF
            
        def getIBESCode(self):   # For marketDB records only
            return self.REF_[self.REF_.find(":"):] if self.REF_ is not None else self.REF_

'''TODO: Document me. '''
class FundamentalNumberItem(DataItem):
    def __init__(self, connections):
        DataItem.__init__(
            self, connections, 'asset_dim_fund_number', 'value')
    
    def _createValueStruct(self, values):
        rval = Utilities.Struct()
        (value,) = values
        rval.value = float(value)
        return rval

class MetaInterestRates(MarketDBSource):
    """Class to retrieve the risk-free rate of a currency from the
    meta_interest_rates table.  The mapping between currencies and
    instruments is given in the connections.instruments variable as a
    list of tuples: (currency, name, from_dt, thru_dt). The instrument
    names are resolved to codes via the meta_instrument_ref table.
    """
    def __init__(self, connections):
        MarketDBSource.__init__(self, connections)
        self.INCR = 100
        self.argList = [('inst%d' % i) for i in range(self.INCR)]
        self.currencyInstCode = dict()
        self.liborCurrencies = set()
        for val in connections.instruments:
            (currency, instName, fromDt, thruDt) = val
            self.marketDB.dbCursor.execute("""SELECT id, from_dt, thru_dt
               FROM meta_interest_ref WHERE
               name=:name_arg AND from_dt<:thru_dt
               AND thru_dt>:from_dt""", name_arg=instName,
                                           from_dt=fromDt, thru_dt=thruDt)
            r = self.marketDB.dbCursor.fetchall()
            if len(r) == 0:
                self.log.debug('No code defined for %s', instName)
            else:
                for rec in r:
                    self.currencyInstCode.setdefault(currency, list()).append(
                        (rec[0], max(rec[1].date(), fromDt), min(rec[2].date(), thruDt)))
        self.mappedCurrencies = set(self.currencyInstCode.keys())
        self.query = """SELECT dt, instrument_code, value
           FROM meta_interest_rates_active ir1
           WHERE instrument_code in (%(args)s)
           AND dt=(SELECT MAX(dt) FROM meta_interest_rates_active ir2
              WHERE ir1.instrument_code=ir2.instrument_code
              AND ir2.dt between :dt-:rollOverLimit AND :dt)""" % {
            'args': ','.join([':%s' % arg for arg
                                  in self.argList])}
        self.defaultDict = dict([(arg, None) for arg in self.argList])
        self.rollOverLimit = 32
    
    def getBulkData(self, dateList, currencyList):
        self.log.debug('MetaInterestRates.getBulkData')
        retvals = numpy.empty((len(dateList), len(currencyList)), dtype=object)
        for currency in [c for c in currencyList
                         if c[0] not in self.mappedCurrencies
                         and c[0] not in self.liborCurrencies]:
            if min(dateList) < currency[2] and max(dateList) >= currency[1]:
                self.log.debug('No instrument defined for %s', currency[0])
        valueDict = dict()
        instList = set()
        for vList in self.currencyInstCode.values():
            instList.update([i[0] for i in vList])
        instList = sorted(instList)
        for dt in dateList:
            for instChunk in listChunkIterator(instList, self.INCR):
                updateDict = dict(zip(self.argList, instChunk))
                myDict = self.defaultDict.copy()
                myDict.update(updateDict)
                myDict['dt'] = dt
                myDict['rollOverLimit'] = self.rollOverLimit
                self.marketDB.dbCursor.execute(self.query, myDict)
                for (valDate, instCode, value) \
                        in self.marketDB.dbCursor.fetchall():
                    key = (instCode, dt)
                    valueDict[key] = float(value) / 100.0
        for (dIdx, date) in enumerate(dateList):
            for (cIdx, currency) in enumerate(currencyList):
                val = Utilities.Struct()
                if currency[0] not in self.currencyInstCode:
                    continue
                instList = self.currencyInstCode[currency[0]]
                instList = [v for v in instList
                            if v[1] <= date and v[2] > date]
                assert(len(instList) <= 1)
                if len(instList) == 0:
                    continue
                instCode = instList[0][0]
                val.value = valueDict.get((instCode, date))
                if val.value is not None:
                    retvals[dIdx, cIdx] = val
        return retvals

class CumulativeRiskFreeRate(ModelDBSource):
    """Class to compute the cumulative risk-free rates.
    """
    def __init__(self, connections):
        ModelDBSource.__init__(self, connections)
        self.marketDB = connections.marketDB
        self.INCR = 100
        self.DATEINCR = 10
        self.argList = [('ccy%d' % i) for i in range(self.INCR)]
        self.dateArgList  = [('date%d' % i) for i in range(self.DATEINCR)]
        self.prevCumRetQuery = """SELECT currency_code, dt, cumulative
          FROM currency_risk_free_rate
          WHERE currency_code IN (%(ccys)s) AND dt IN (%(dates)s)""" % {
            'ccys': ','.join([':%s' % i for i in self.argList]),
            'dates': ','.join([':%s' % i for i in self.dateArgList])}
        self.defaultDict = dict([(arg, None) for arg
                                 in self.argList + self.dateArgList])
        self.oneDay = datetime.timedelta(days=1)
    
    def getBulkData(self, dateList, ccyInfoList):
        self.log.debug('CumulativeRiskFreeRate.getBulkData')
        retvals = numpy.empty((len(dateList), len(ccyInfoList)), dtype=object)
        cur = self.modelDB.dbCursor
        cumRates = Matrices.allMasked((len(dateList), len(ccyInfoList)))
        ccyList = [ccy for (ccy, fromDt, thruDt) in ccyInfoList]
        prevDateList = [d - self.oneDay for d in dateList]
        prevDateIdx = dict((d,i) for (i,d) in enumerate(prevDateList))
        dateIdx = dict((d,i) for (i,d) in enumerate(dateList))
        ccyIdx = dict((c,i) for (i,c) in enumerate(ccyList))
        # Get cumulative return values of previous day
        for dateChunk in listChunkIterator(prevDateList, self.DATEINCR):
            updateDateDict = dict(zip(self.dateArgList, dateChunk))
            for ccyChunk in listChunkIterator(ccyList, self.INCR):
                updateCcyDict = dict(zip(self.argList, ccyChunk))
                myDict = self.defaultDict.copy()
                myDict.update(updateDateDict)
                myDict.update(updateCcyDict)
                cur.execute(self.prevCumRetQuery, myDict)
                for (ccy, dt, cumVal) in cur.fetchall():
                    dIdx = prevDateIdx[dt.date()]
                    cIdx = ccyIdx[ccy]
                    if cumVal is not None:
                        cumRates[dIdx, cIdx] = float(cumVal)
        cumRates = cumRates.filled(1.0)
        # Get risk-free rates for required days
        rates = self.getAnnualizedRates(ccyList, dateList, ccyInfoList)
        for (cIdx, (ccy, fromDt, thruDt)) in enumerate(ccyInfoList):
            for (dIdx, date) in enumerate(dateList):
                if date < fromDt or date >= thruDt:
                    continue
                prevCumRet = cumRates[dIdx,cIdx]
                annualRate = rates.data[cIdx, dIdx]
                if annualRate is ma.masked:
                    continue
                dailyRate = pow(1.0 + annualRate, 1.0 / 365.0)
                myCumRet = prevCumRet * dailyRate
                nextDay = date + self.oneDay
                nextDIdx = dateIdx.get(nextDay)
                if nextDIdx is not None:
                    cumRates[nextDIdx, cIdx] = myCumRet
                rval = Utilities.Struct()
                rval.cumulative = myCumRet
                rval.dailyRate = dailyRate
                retvals[dIdx, cIdx] = rval
        return retvals
    
    def getAnnualizedRates(self, ccyList, dateList, ccyInfoList):
        rates = Matrices.TimeSeriesMatrix(ccyList, dateList)
        if numpy.size(rates.data) == 0:
            return rates
        rates.data = Matrices.allMasked(rates.data.shape)
        dateIdxMap = dict((d, dIdx) for (dIdx, d) in enumerate(dateList))
        ccyIdxMap = dict((ccy, cIdx) for (cIdx, ccy) in enumerate(ccyList))
        dates = sorted(dateList)
        dates.reverse()
        # Process dates in chunks where no currency dies or starts
        while len(dates) > 0:
            dateChunk = list()
            d = dates.pop()
            dateChunk.append(d)
            activeCurrencies = self.getActiveCurrencies(ccyInfoList, d)
            # Add dates until active Currencies list chages
            while len(dates) > 0:
                candidateDate = dates[-1]
                if activeCurrencies == self.getActiveCurrencies(ccyInfoList, candidateDate):
                    dateChunk.append(candidateDate)
                    dates.pop()
                else:
                    break
            activeCurrencies = list(activeCurrencies)
            rateChunk = self.modelDB.getRiskFreeRateHistory(
                list(activeCurrencies), dateChunk, self.marketDB, annualize=True)
            rates.data[numpy.ix_([ccyIdxMap[ccy] for ccy in activeCurrencies],
                                 [dateIdxMap[date] for date in dateChunk])] = rateChunk.data
        return rates
    
    def getActiveCurrencies(self, ccyInfoList, d):
        return set(ccy for (ccy, fromDt, thruDt) in ccyInfoList
                   if fromDt <= d and d < thruDt)


class AxiomaDBClassificationOverride(ClassificationBase):
    """Pull Axioma defined industry classifications from AxiomaDB
    override table."""
    def __init__(self, connections,
                 srcFamilyName, srcMemberName, srcClsDateStr,
                 tgtFamilyName, tgtMemberName, tgtClsDateStr):
        ClassificationBase.__init__(
            self, connections,
            srcFamilyName, srcMemberName, srcClsDateStr,
            tgtFamilyName, tgtMemberName, tgtClsDateStr,
            'ind_class_const_active', connections.axiomaDB)
        self.SRC_ID = 1900
    
    def getSourceID(self, srcID):
        return self.SRC_ID
    
    def findCode(self, axid, date, axidClsDict):
        if axid is None or axid not in axidClsDict:
            return None
        rval = [i for i in axidClsDict[axid] if i.changeDate <= date]
        if len(rval) == 0 or rval[-1].classificationIdAndWeight is None \
               or rval[-1].changeFlag == 'Y':
            return None
        rval[-1].isGuess = True
        return rval[-1]


class AxiomaDBClassification(ClassificationBase):
    """Pull Axioma defined industry classifications from AxiomaDB."""
    def __init__(self, connections,
                 srcFamilyName, srcMemberName, srcClsDateStr,
                 tgtFamilyName, tgtMemberName, tgtClsDateStr):
        ClassificationBase.__init__(
            self, connections,
            srcFamilyName, srcMemberName, srcClsDateStr,
            tgtFamilyName, tgtMemberName, tgtClsDateStr,
            'mdl_class_const_active', connections.axiomaDB)
        self.SRC_ID = 1900
        self.AXIOMADB_SRC = 1900
    
    def getSourceID(self, srcID):
        return self.SRC_ID
    
    def findCode(self, axid, date, axidClsDict):
        if axid is None or axid not in axidClsDict:
            return None
        rval = [i for i in axidClsDict[axid] if i.changeDate <= date]
        if len(rval) == 0 or rval[-1].classificationIdAndWeight is None \
               or rval[-1].changeFlag == 'Y':
            return None
        rval[-1].isGuess = True
        return rval[-1]

class AxiomaDBFutureClassification(ClassificationBase):
    """Pull Axioma defined industry classifications from AxiomaDB."""
    def __init__(self, connections,
                 srcFamilyName, srcMemberName, srcClsDateStr,
                 tgtFamilyName, tgtMemberName, tgtClsDateStr):
        ClassificationBase.__init__(
            self, connections,
            srcFamilyName, srcMemberName, srcClsDateStr,
            tgtFamilyName, tgtMemberName, tgtClsDateStr,
            'mdl_class_const_active', connections.axiomaDB)
        self.SRC_ID = 1900
        self.AXIOMADB_SRC = 1900
    
    def getSourceID(self, srcID):
        return self.SRC_ID
    
    def findCode(self, axid, date, axidClsDict):
        if axid is None or axid not in axidClsDict:
            return None
        rval = [i for i in axidClsDict[axid] if i.changeDate > date
                and i.changeFlag == 'N']
        if len(rval) == 0 or rval[0].classificationIdAndWeight is None:
            return None
        rval[0].isGuess = True
        return rval[0]

class AxiomaDBPastClassification(ClassificationBase):
    """Pull Axioma defined industry classifications from AxiomaDB."""
    def __init__(self, connections,
                 srcFamilyName, srcMemberName, srcClsDateStr,
                 tgtFamilyName, tgtMemberName, tgtClsDateStr):
        ClassificationBase.__init__(
            self, connections,
            srcFamilyName, srcMemberName, srcClsDateStr,
            tgtFamilyName, tgtMemberName, tgtClsDateStr,
            'mdl_class_const_active', connections.axiomaDB)
        self.SRC_ID = 1900
        self.AXIOMADB_SRC = 1900
    
    def getSourceID(self, srcID):
        return self.SRC_ID
    
    def findCode(self, axid, date, axidClsDict):
        if axid is None or axid not in axidClsDict:
            return None
        rval = [i for i in axidClsDict[axid] if i.changeDate < date
                and i.changeFlag == 'N']
        if len(rval) == 0 or rval[-1].classificationIdAndWeight is None:
            return None
        rval[-1].isGuess = True
        return rval[-1]

class OldModelDBSource:
    """Base class for access to the old (production) ModelDBs.
    """
    def __init__(self, connections, dbConnection):
        if dbConnection == 'US':
            self.oldModelDB = connections.oldUSModelDB
            self.MODELDB_SRC = 1500
            self.OldBenchmark_SRC = 1700
        if dbConnection == 'UK':
            self.oldModelDB = connections.oldUKModelDB
            self.MODELDB_SRC = 1600
            self.OldBenchmark_SRC = 1800
        if dbConnection == 'TW':
            self.oldModelDB = connections.oldTWModelDB
            self.MODELDB_SRC = 2500
        self.dbMnemonic = dbConnection
        self.modelDB = connections.modelDB
        self.maxTransferDt = self.oldModelDB.maxTransferDt
        self.log = self.oldModelDB.log
        self.oldModelIDs = None
        self.oldSubIssueIDs = None
    
    def createOldModelIDs(self):
        """Create set of issue IDs that are part of this old ModelDB.
        """
        if self.oldModelIDs is None:
            self.oldModelDB.dbCursor.execute(
                'SELECT issue_id FROM issue')
            self.oldModelIDs = set([ModelID.ModelID(string=i[0]) for i
                                     in self.oldModelDB.dbCursor.fetchall()])
    
    def createOldSubIssueIDs(self):
        """Create set of sub-issue IDs that are part of this old ModelDB.
        """
        if self.oldSubIssueIDs is None:
            self.oldModelDB.dbCursor.execute(
                'SELECT sub_id FROM sub_issue')
            self.oldSubIssueIDs = set([ModelDB.SubIssue(string=i[0]) for i
                                       in self.oldModelDB.dbCursor.fetchall()])
    
    def getBulkData(self, dateList, axidList):
        """Returns the values corresponding to each date/Axioma ID
        pair defined by the two lists.
        The return value is an (date, ID) array with of structs containing
        id, src_id, and ref fields. The entry is None if no data is available.
        """
        retvals = numpy.empty((len(dateList), len(axidList)), dtype=object)
        for (dIdx, d) in enumerate(dateList):
            for (aIdx, axid) in enumerate(axidList):
                retvals[dIdx, aIdx] = self.getData(d, axid)
        return retvals

class OldModelDBReturn(OldModelDBSource):
    def __init__(self, connections, dbConnection):
        OldModelDBSource.__init__(self, connections, dbConnection)
        self.DATEINCR = 30
        self.INCR = 200
        self.argList = [('sid%d' % i) for i in range(self.INCR)]
        self.dateArgList = [('date%d' % i) for i in range(self.DATEINCR)]
        self.query = """SELECT sub_issue_id, dt, tr
          FROM sub_issue_data
          WHERE dt IN (%(dateargs)s)
          AND sub_issue_id in (%(sid)s) AND tr IS NOT NULL
          AND trans_from_dt <= :transDt and trans_thru_dt > :transDt""" % {
            'dateargs': ','.join([':%s' % darg for darg
                                  in self.dateArgList]),
            'sid': ','.join([':%s' % axIdarg for axIdarg
                             in self.argList])}
        self.defaultDict = dict([(arg, None) for arg
                                 in self.argList + self.dateArgList])
        self.defaultDict['transDt'] = self.oldModelDB.transDateTime
        self.createOldSubIssueIDs()
    
    def getBulkData(self, dateList, sidList):
        self.log.debug('OldModelDBReturn.getBulkData:%s', self.dbMnemonic)
        sidStrs = [i.getSubIDString() for i in sidList]
        retvals = numpy.empty((len(dateList), len(sidList)), dtype=object)
        # Nothing to do if the minimum date in dateList is > self.maxTransferDt
        if min(dateList) > self.maxTransferDt:
            return retvals
        valueDict = dict()
        for dateChunk in listChunkIterator(dateList, self.DATEINCR):
            updateDateDict = dict(zip(self.dateArgList, dateChunk))
            for codeChunk in listChunkIterator(sidStrs, self.INCR):
                updateDict = dict(zip(self.argList, codeChunk))
                myDict = dict(self.defaultDict)
                myDict.update(updateDateDict)
                myDict.update(updateDict)
                self.oldModelDB.dbCursor.execute(self.query, myDict)
                for (axid, date, tRet) \
                        in self.oldModelDB.dbCursor.fetchall():
                    date = date.date()
                    rval = Utilities.Struct()
                    rval.tr = float(tRet)
                    valueDict[(axid, date)] = rval
        noValue = Utilities.Struct()
        noValue.tr = None
        for (dIdx, date) in enumerate(dateList):
            for (sIdx, (sid, sidObj)) in enumerate(zip(sidStrs, sidList)):
                if date <= self.maxTransferDt:
                    val = valueDict.get((sid, date))
                    if val is None and sidObj in self.oldSubIssueIDs:
                        val = noValue                    
                    retvals[dIdx, sIdx] = val
                else:
                    retvals[dIdx, sIdx] = None
        return retvals

class OldModelDBCumulativeRFR(OldModelDBSource):
    def __init__(self, connections, dbConnection):
        OldModelDBSource.__init__(self, connections, dbConnection)
        self.INCR = 100
        self.argList = [('ccy%d' % i) for i in range(self.INCR)]
        self.query = """SELECT currency_code, dt, cumulative
          FROM currency_risk_free_rate cr1
          WHERE dt = (SELECT MAX(dt) FROM currency_risk_free_rate cr2
             WHERE cr1.currency_code=cr2.currency_code
             AND cr2.dt between :dt - 10 AND :dt)
          AND currency_code in (%(ccys)s)""" % {
            'ccys': ','.join([':%s' % axIdarg for axIdarg
                              in self.argList])}
        self.defaultDict = dict([(arg, None) for arg in self.argList])
    
    def getBulkData(self, dateList, ccyInfoList):
        self.log.debug('OldModelDBCumulativeRFRReturn.getBulkData')
        retvals = numpy.empty((len(dateList), len(ccyInfoList)), dtype=object)
        cur = self.oldModelDB.dbCursor
        ccyList = [ccy for (ccy, fromDt, thruDt) in ccyInfoList]
        ccyIdx = dict((c,i) for (i,c) in enumerate(ccyList))
        for (dIdx, dt) in enumerate(dateList):
            for ccyChunk in listChunkIterator(ccyList, self.INCR):
                updateCcyDict = dict(zip(self.argList, ccyChunk))
                myDict = self.defaultDict.copy()
                myDict['dt'] = dt
                myDict.update(updateCcyDict)
                cur.execute(self.query, myDict)
                for (ccy, dt, cumVal) in cur.fetchall():
                    cIdx = ccyIdx[ccy]
                    if cumVal is not None:
                        rval = Utilities.Struct()
                        rval.cumulative = float(cumVal)
                        retvals[dIdx, cIdx] = rval
        return retvals


class SubIssueDivYield(MarketDBSource):
    """Class to retrieve dividend information from MarketDB that is
    required for the sub_issue_divyield table.  
    Uses the StyleExposures.computeAnnualDividends method so if something changes there, please coordinate
    """
    def __init__(self, connections):
        MarketDBSource.__init__(self, connections)
        self.INCR = 200
    
    def getBulkData(self, dateList, sidList):
        self.log.debug('SubIssueDivYield.getBulkData')
        model=Utilities.Struct()
        model.numeraire=Utilities.Struct()
        model.numeraire.currency_id=1 # hard-code USD for now
        
        retvals = numpy.empty((len(dateList), len(sidList)), dtype=object)
        if not self.modelDB.currencyCache:
            self.modelDB.createCurrencyCache(self.marketDB)
        
        for (dIdx, date) in enumerate(dateList):
            # don't do anything for weekends and Jan 1st - Axioma holidays
            if date.isoweekday() > 5 or (date.month==1 and date.day==1):
                self.log.info('Saturday/Sunday, skipping.')
                continue
            divs = computeAnnualDividendsNew(date, sidList, model, self.modelDB, self.marketDB, includeSpecial=False,includeStock=False)
            mcaps = self.modelDB.getAverageMarketCaps([date], sidList, currencyID=1)
            divYield = divs/mcaps
            for (sIdx, sid) in enumerate(sidList):
                if divYield[sIdx] is ma.masked:
                    retvals[dIdx, sIdx] = None
                else:
                    rval = Utilities.Struct()
                    rval.value = divYield[sIdx]
                    retvals[dIdx, sIdx] = rval
        return retvals
