
import datetime
import logging
import optparse
import configparser
import copy
import datetime
import numpy
import multiprocessing
import functools
import os
from marketdb import MarketDB
import marketdb.Utilities as Utilities
from riskmodels import Connections
from riskmodels import Classification
from riskmodels import Matrices
from riskmodels import ModelDB
from riskmodels import ModelID
from riskmodels import transfer

class TimeSeriesMatrix:
    """asset by date) matrix of integers.
    """
    def __init__(self, assets, dates, masked=False):
        self.assets = assets
        self.dates = dates
        if masked:
            self.data = numpy.ma.array(numpy.zeros((len(assets), len(dates)), dtype=int),
                                 mask=numpy.array([True]))
        else:
            self.data = numpy.ma.zeros((len(assets), len(dates)), int)

class MissingGICSChecker:
    """Class to check assets with missing GICS and the corresponding time period
    """
    def __init__(self, GICSProcessor):
        self.GICSProcessor = GICSProcessor
        self.modelDB = self.GICSProcessor.modelDB
        self.revisionId = self.GICSProcessor.revisionId
        self.INCR = 200
        self.argList = [('iid%d' % i) for i in range(self.INCR)]
        self.defaultDict = dict([(arg, None) for arg in self.argList])
        
    def getIssueLife(self, iids):
        """Return a dictionary mapping for each iid to their 
           time span. The earliest date is 1980-01-02
        """
        logging.info('Getting Issue Life for %d assets', len(iids))
        RangeDict = dict()
        idStrs = [i.getIDString() for i in iids]
        query = """select b.MODELDB_ID, case when min(b.from_dt) < to_date('02jan1980', 'ddmmyyyy') 
                   then to_date('02jan1980', 'ddmmyyyy')
                   when a.from_dt < min(b.from_dt) then min(b.from_dt)
                   else a.from_dt end from_dt, case when a.thru_dt > max(b.thru_dt)
                   then max(b.thru_dt) else a.thru_dt end thru_dt
                   from marketdb_global.asset_ref a, modeldb_global.issue_map b where
                   a.axioma_id=b.marketdb_id and b.MODELDB_ID in (%(ids)s)
                   group by MODELDB_ID, a.from_dt, a.thru_dt
               """%{'ids':','.join([':%s'%i for i in self.argList])}
        for idChunk in Utilities.listChunkIterator(idStrs, self.INCR):
            myDict = dict(self.defaultDict)
            myDict.update(dict(zip(self.argList, idChunk)))
            self.modelDB.dbCursor.execute(query, myDict)
            for (iid, fromDt, thruDt) in self.modelDB.dbCursor.fetchall(): 
                RangeDict[ModelID.ModelID(string=iid)] = (fromDt, thruDt)
        return RangeDict

    def getClassificationHistory(self, iids):
        """Returns a dictionary mapping for each iid to their entire classification
           history given revision ID.
        """
        INCR=self.GICSProcessor.INCR
        keyList = ['key%d' % i for i in range(INCR)]
        defaultDict = dict([(a, None) for a in keyList])
        issueClsDict = dict()
        issueStr = [i.getIDString() for i in iids]
        query = """SELECT ca.classification_id, ca.weight, ca.change_dt,
                   ca.change_del_flag, ca.src_id, ca.ref, ca.issue_id
                   FROM classification_const_active ca
                   WHERE ca.issue_id IN (%(keys)s) AND ca.revision_id=:revision_id
                   ORDER BY change_dt ASC
                """% {'keys': ','.join([':%s' % i for i in keyList])}
        logging.info('Loading Classifcation History for %d assets', len(iids))
        for idChunk in Utilities.listChunkIterator(issueStr, INCR):
            valueDict = dict(defaultDict)
            valueDict['revision_id']=self.GICSProcessor.revisionId
            valueDict.update(dict(zip(keyList, idChunk)))
            self.modelDB.dbCursor.execute(query, valueDict)
            r = self.modelDB.dbCursor.fetchmany()
            while len(r) > 0:
                for (clsId, weight, changeDt, changeFlag, srcId, ref, iid) in r:
                    info = Utilities.Struct()
                    info.classification_id = clsId
                    info.weight = weight
                    info.change_dt = changeDt.date()
                    info.src = srcId
                    info.ref = ref
                    issueClsDict.setdefault(ModelID.ModelID(string=iid), list()).append(
                        (changeDt.date(), changeFlag, info))
                r = self.modelDB.dbCursor.fetchmany()
        return issueClsDict

    def getGICSMat(self, iids, iidRange):
        """Return a masked time series matrix of GICS. The shape of matrix is no. of date between 
           oldest from_dt among iids and no. of iids. Entry outside assets' lifespan is masked.
           Missing GICS is flagged as 0 whereas valid GICS is flagged as 1.
        """
        startDt = min(zip(*list(iidRange.values()))[0]).date()
        endDt = datetime.date.today()
        AllDtList = [startDt + datetime.timedelta(i) for i in
                     range((endDt - startDt).days + 1)]
        dtIdxMap = dict([(j,i) for (i,j) in enumerate(AllDtList)])
        iidIdxMap = dict([(j,i) for (i,j) in enumerate(iids)]) 
        GICSMat = TimeSeriesMatrix(iids, AllDtList, masked=True) 
        GICSMat.dtIdxMap = dtIdxMap
        GICSMat.iidIdxMap = iidIdxMap
        GICSHistory = self.getClassificationHistory(iids)
        logging.info('Processing Classifcation TimeSeries Matrix for %d assets', len(iids))
        itemsSoFar = 0
        lastReport = 0
        for iid in iids:
            itemsSoFar += 1
            if itemsSoFar - lastReport > 0.05 * len(iids):
                lastReport = itemsSoFar
                logging.info('...%g%%', 100.0 * float(itemsSoFar) / len(iids))
            iidLife = iidRange.get(iid, None)
            if iidLife is None:
                logging.error('%s is a dummy asset. Please check', iid.getIDString())
                exit(1)
            else:
                fromDt = iidLife[0].date()
                thruDt = iidLife[1].date()
            if thruDt > datetime.date.today():
                thruDt = datetime.date.today()
            dtList = [fromDt + datetime.timedelta(i) for i in
                      range((thruDt - fromDt).days)]
            #Initiate GICS Exposure for active asset
            for dt in dtList:
                GICSMat.data[iidIdxMap[iid], dtIdxMap[dt]] = 0
            #Populate GICS Exposure
            iidGICSHistory = GICSHistory.get(iid, None)
            if iidGICSHistory is None:
                logging.warning('%s has no GICS history. Please check', iid.getIDString())
            else:
                for dt in dtList:
                    for (changeDt, changeFlag, info) in iidGICSHistory:
                        if dt >= changeDt and changeFlag != 'Y':
                            GICSMat.data[iidIdxMap[iid], dtIdxMap[dt]] = 1
                        elif dt >= changeDt and changeFlag == 'Y':
                            GICSMat.data[iidIdxMap[iid], dtIdxMap[dt]] = 0
                            
        logging.info('Finished processing Classifcation TimeSeries Matrix for %d assets', len(iids))
        return GICSMat

    def getMissingGICSiids(self, iids):
        """Return dictionary mapping for each iid to missing GICS dates given GICS times 
           series matrix
        """
        iidRange = self.getIssueLife(iids)
        GICSExp = self.getGICSMat(iids, iidRange)
        (iidIdx, missingdtIdx) = numpy.nonzero(GICSExp.data==0)
        MissingDict = dict()
        for iid, dt in zip(iidIdx, missingdtIdx):
            MissingDict.setdefault(GICSExp.assets[iid], list()).append(GICSExp.dates[dt])
        for k, v in MissingDict.items():
            MissingDict[k] = list(getDtRanges(v))
        logging.info('Finished getting missing Classifcation')
        return MissingDict
            
def getDtRanges(dates):        
    """Generator of continous date
    """
    while dates:
        end = 1
        try:
            while dates[end] - dates[end - 1] == datetime.timedelta(days=1):
                end += 1
        except IndexError:
            pass
        yield (dates[0], dates[end - 1])
        dates = dates[end:]

def FilterNonEquityAssets(modelDB, axidList):
    mdlCur = modelDB.dbCursor
    nonEqSet = set()
    query = """SELECT DISTINCT im.MODELDB_ID FROM marketdb_global.classification_active_int cat
               left join modeldb_global.ISSUE_MAP im on im.MARKETDB_ID=cat.AXIOMA_ID
               WHERE classification_type_name='Axioma Asset Type'
               AND cat.name
               IN ('Derivatives-T', 'Futures-C', 'EIF Series-S', 'EIF Series-SS', 
                   'Commodity Futures Contract-SS', 'EIF Contract-S', 'EIF Contract-SS', 
                   'Commodity Futures Contract-S', 'Commodity Futures Series-SS',
                   'Commodity Futures Series-S', 'ETF-T', 'ETF-C', 'ETF (Stat)-S', 
                   'ETF (Stat)-SS', 'ETF (Composite)-S', 'ETF (Composite)-SS')
            """  
    mdlCur.execute(query)
    for i in mdlCur.fetchall(): 
        if i[0] is not None:
            nonEqSet.add(ModelID.ModelID(string=i[0]))

    query = """SELECT i.ISSUE_ID FROM modeldb_global.ISSUE i
               where not exists 
               (select ar.* from marketdb_global.asset_ref ar, modeldb_global.ISSUE_MAP im
               where im.MODELDB_ID=i.ISSUE_ID and im.MARKETDB_ID=ar.AXIOMA_ID)
               and not exists 
               (select * from modeldb_global.FUTURE_ISSUE_MAP fim where fim.MODELDB_ID=i.ISSUE_ID)
               and not exists 
               (select ar.* from marketdb_global.asset_ref ar, modeldb_global.ISSUE_MAP im 
               where im.MODELDB_ID=i.ISSUE_ID and im.MARKETDB_ID=ar.AXIOMA_ID)
            """ 
    mdlCur.execute(query)
    for i in mdlCur.fetchall():
        if i[0] is not None:
            nonEqSet.add(ModelID.ModelID(string=i[0]))

    logging.debug('Found %s ETFs/EIFs/Dummies...', len(nonEqSet))
    axidList = list(set(axidList)-nonEqSet)

    query = """SELECT issue_id, sub_id FROM sub_issue si JOIN
               issue_map im ON im.modeldb_id=si.issue_id and im.marketdb_id not like 'O%'
               and im.marketdb_id not like 'CSH%'
            """  
    mdlCur.execute(query)
    validIdsSet = set(ModelID.ModelID(string=i) for (i, s) in mdlCur.fetchall())
    axidList = list(set(axidList).intersection(validIdsSet))
    return axidList

def setlogger(idName):
    formatter = logging.Formatter('%(asctime)s %(levelname)s '+idName+' - %(message)s')
    log = logging.getLogger()
    for handler in log.handlers:
        handler.setFormatter(formatter)

def bucketizeList(List, bucketsize):
    total = len(List)
    remainder = total % bucketsize
    nbucket = total//bucketsize if remainder == 0 else (total//bucketsize) + 1
    remainder = total%nbucket
    seperateIdx = total/nbucket if remainder==0 else (total/nbucket)+1
    buckets = [List[x:x+seperateIdx] for x in range(0, len(List), seperateIdx)]
    return buckets
    
def runMain(config_, axidList):
    idName = multiprocessing.current_process().name
    setlogger(idName)
    connections_ = Connections.createConnections(config_)
    section = config_.get('DEFAULT', 'section').strip()
    logging.info('Processing section: %s', section)
    targetFields = config_.get(section, 'target').split(':')
    targetType = targetFields[0]
    targetArgs = targetFields[1:]
    GICSProcessor = eval('%s(connections_, *targetArgs)'% targetType)
    GICSChecker = MissingGICSChecker(GICSProcessor)
    axidList = FilterNonEquityAssets(connections_.modelDB, axidList)
    result = GICSChecker.getMissingGICSiids(axidList)
    Total_progress += Total_progress+len(axidList)
    return result

if __name__ == '__main__':
    usage = "usage: %prog config-file issue-id section [option=value...]"
    cmdlineParser = optparse.OptionParser(usage=usage) 
    Utilities.addDefaultCommandLine(cmdlineParser)
    cmdlineParser.add_option("--filename", action="store",
                             default='missingGICS', dest="reportFile",
                             help="report file name")
    cmdlineParser.add_option("--bucketsize", action="store",
                             default=5000, dest="bucket",
                             help="Specify a size of bucket to run multiprocessing")
    cmdlineParser.add_option("--processors", action="store",
                             default=5, dest="processors",
                             help="Specify a size of bucket to run multiprocessing")
    (options_, args_) = cmdlineParser.parse_args()
    Utilities.processDefaultCommandLine(options_, cmdlineParser)
    
    configFile_ = open(args_[0])
    config_ = configparser.ConfigParser()
    config_.read_file(configFile_)
    configFile_.close()

    section = 'DEFAULT'
    for arg in args_[1:]:
        fields = arg.split('=')
        if len(fields) != 2:
            logging.fatal('Incorrect command-line assignment "%s"'
                          ', exactly one "=" is required', arg)
        (option, value) = fields
        config_.set(section, option, value)

    if len(config_.get('DEFAULT', 'section').split(',')) > 1:
        logging.info('only 1 GICS section can be check at a time')
        exit(1)   
    if config_.get('DEFAULT', 'section').find('GICS') == -1:
        logging.info('No GICS section to check')
        exit(1)
        
    if config_.has_option(section,'issue-ids'):
        ids=config_.get(section,'issue-ids')
        if len(ids)==2:
            if section.find('GICSCustom')==0:
                ids=ids.upper()
                gicsctry=section[len('GICSCustom')+section.find('GICSCustom'):][:2].upper()
                if (gicsctry in ['AU','CN','CA','JP','GB','TW','US'] and gicsctry != ids):
                    logging.info('Can ignore this section %s for %s', section, ids)
                    exit(1)

    bucketsize = int(options_.bucket)
    processors = int(options_.processors)
    connections_ = Connections.createConnections(config_)

    if ids:
        (axidList, axidRanges) = transfer.createIssueIDList(ids, connections_)
    else:
        logging.info('No issue-ids to process')
        exit(1)

    if len(axidList) <= bucketsize:
        result = runMain(config_, axidList)
    else:
        logging.info('Creating pool with %d processes for checking %d issue-ids' 
                     % (processors, len(axidList)))
        pool = multiprocessing.Pool(processors)
        idsbuckets = bucketizeList(axidList, bucketsize)
        try:
            main = functools.partial(runMain, config_)
            result = pool.map_async(main, idsbuckets).get(9999999)
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()

    filename=options_.reportFile
    i = 0
    if os.path.isfile(filename + "_v" + str(i) + ".csv"):
        logging.info(filename + "_v" + str(i) + ".csv" + " already exists")
        i = 1
        while os.path.isfile(filename + "_v" + str(i) + ".csv"):
            logging.info(filename + "_v" + str(i) + ".csv"  + " already exists")
            i = i + 1
    filename = filename + "_v" + str(i) + ".csv"
    csvfile = open(filename, "w")
    logging.info("Writing csv file to " + filename)
    for missingDict in result:
        for iid, dtList in missingDict.items():
            for (fromdt, enddt) in dtList:
                csvfile.write(','.join([iid.getIDString(), str(fromdt), str(enddt)])+'\n')
    csvfile.close()
    logging.info('Fninished')
