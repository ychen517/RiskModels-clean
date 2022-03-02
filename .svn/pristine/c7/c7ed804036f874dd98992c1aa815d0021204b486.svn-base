
import sys
import optparse
import configparser
import logging
import datetime
import numpy
import numpy.ma as ma

from riskmodels import creatermsid
from marketdb import MarketDB
from marketdb.Utilities import listChunkIterator
from riskmodels import Connections
from riskmodels import ModelDB
from riskmodels import Utilities
from riskmodels import transfer

def getModelDBId(modelDB, aid, date):
    query = """SELECT DISTINCT modeldb_id, sub_id FROM issue_map im, sub_issue si
               WHERE marketdb_id = :aid_arg
               AND im.modeldb_id = si.issue_id"""
    modelDB.dbCursor.execute(query, aid_arg = aid)
    r = modelDB.dbCursor.fetchall()
    if len(r) == 1:
        return r[0]
    else:
        logging.info('Failed to find issue_id and sub_issue_id for %s'%aid)
        return (None,None)

def updateTable(modelDB, idDtMap, tableName, idCol):
    updateDicts = []
    for (iid, (newFromDt, oldFromDt, sid)) in idDtMap.items():
        updateDict = {'aid_arg':iid, 'old_arg':oldFromDt, 'new_arg':newFromDt}
        updateDicts.append(updateDict)
    modelDB.dbCursor.executemany("""UPDATE %s SET from_dt = :new_arg
                                    WHERE %s = :aid_arg AND 
                                    from_dt <= :old_arg AND thru_dt > :old_arg"""%\
                                     (tableName, idCol),
                                 updateDicts)

def insertCumRecord(modelDB, sid, af, date, rmg_id):
    modelDB.dbCursor.execute("""INSERT INTO SUB_ISSUE_CUMULATIVE_RETURN
                       VALUES(:sid_arg, :date_arg, sysdate, 'N', :value_arg, null, :rmg_arg)""",\
                        sid_arg = sid, date_arg = date, value_arg = af, rmg_arg = rmg_id)

def getResyncDict(modelDB, idList):
    resultDict = dict()
    mismatchDict = dict()
    for sid in idList:
        if len(sid) == 10:
            if sid[0] != 'D':
                sidStr = getModelDBId(modelDB, sid, None) + "11"
            else:
                sidStr = sid + "11"
        elif len(sid) == 12:
            sidStr = sid
        else:
            sidStr = "D" + sid + "11"

        trQuery = """SELECT dt, tr FROM sub_issue_return_active sira, sub_issue si
                     WHERE sira.dt >= si.from_dt AND sira.dt < si.thru_dt
                     AND sira.sub_issue_id = :sid_arg AND sira.sub_issue_id = si.sub_id
                     ORDER BY dt"""

        modelDB.dbCursor.execute(trQuery, sid_arg = sidStr)
        result = modelDB.dbCursor.fetchall()
        if len(result) > 0:
            retDateList, trList = zip(*result)
            retDateIdxMap = dict([(j,i) for (i,j) in enumerate(retDateList)])
            retMat = ma.zeros((2, len(retDateList)), float)
            retMat[0, :] = trList


            cumRetQuery = """SELECT dt, value FROM sub_issue_cum_return_active sicra,
                             sub_issue si WHERE sicra.dt >= si.from_dt AND sicra.dt < si.thru_dt
                             AND sicra.sub_issue_id = si.sub_id AND sub_id = :sid_arg
                             ORDER BY dt"""
            modelDB.dbCursor.execute(cumRetQuery, sid_arg = sidStr)
            cumResult = modelDB.dbCursor.fetchall()
            
            if len(cumResult) > 0:
                cumDateList, cumRetList = zip(*cumResult)
            else:
                cumDateList = list()
                cumRetList = list()
            
            cumMat = ma.zeros((1, len(cumDateList)), float)
            cumMat[0, :] = cumRetList
            derivedRet = (cumMat[0, 1:]/ cumMat[0, :-1]) - 1
            dateIdxMap = dict([(j,i) for (i,j) in enumerate(cumDateList[1:])])
            
            dateIdx = [dateIdxMap[dt] for dt in cumDateList[1:] if dt in retDateIdxMap]
            retDateIdx = [retDateIdxMap[dt] for dt in cumDateList[1:] if dt in retDateIdxMap]
            ma.put(retMat[1,:], retDateIdx,  ma.take(derivedRet, dateIdx))

            diff = retMat[0,:] - retMat[1,:]
            diff = ma.masked_where(abs(diff) > 1e-8, diff)
            maskedIdx = numpy.flatnonzero(ma.getmaskarray(diff))
            
            if len(maskedIdx) > 0:
                mismatchDict[sidStr] = retDateList[maskedIdx[-1]].date()
    
    if len(list(mismatchDict.keys())) == 0:
        return resultDict 

    #Get the min date here
    INCR = 200
    argList = ['sid%d' %i for i in range(INCR)]
    query = """SELECT sub_issue_id, rmg_id, min(dt) 
               FROM sub_issue_cum_return_active 
               WHERE sub_issue_id in (%(sids)s) group by sub_issue_id, rmg_id"""%{\
        'sids':','.join(':%s'%i for i in argList)}
    sidList = list(mismatchDict.keys())
    sidArgList = ['sid%d' % i for i in range(INCR)]
    sidDefaultDict = dict([(i,None) for i in sidArgList])

    for idChunk in listChunkIterator(sidList, INCR):
        myDict = sidDefaultDict.copy()
        myDict.update(dict(zip(sidArgList, idChunk)))
        modelDB.dbCursor.execute(query, myDict)
        r = modelDB.dbCursor.fetchall()
        if len(r)>0:
            for sid, rmgId, minDt in r:
                resyncDate = mismatchDict[sid]
                resultDict[sid] = (minDt, resyncDate, rmgId)
    return resultDict

def getCumulativeReturn(modelDB, sid, dt):
    modelDB.dbCursor.execute("""SELECT value FROM sub_issue_cum_return_active WHERE sub_issue_id = :sid_arg and dt = :date_arg""",\
                                 sid_arg = sid, date_arg = dt)
    return modelDB.dbCursor.fetchall()

def computeResync(modelDB, sid, minDt, mismatchDt):
#    dateList = transfer.createDateList('%s:%s'%(minDt, mismatchDt))
#    totReturns = modelDB.loadSubIssueData(dateList, [sid], 'sub_issue_return','tr', cache=None, withCurrency=False)
#     totReturns = totReturns.data.filled(0.0)
    modelDB.dbCursor.execute("""SELECT (exp(sum(ln(tr+1)))) FROM sub_issue_return_active sira 
                                WHERE sira.sub_issue_id=:sid_arg AND sira.dt BETWEEN :min_arg 
                                AND :mismatch_arg GROUP BY sira.sub_issue_id""",\
                                 sid_arg = sid, min_arg = minDt, mismatch_arg = mismatchDt)
    return modelDB.dbCursor.fetchall()

def getSubIssueID(modelDB, id, dt):
    assert(len(id) == 10)
    if id[0] == 'G':
        idCol = 'marketdb_id'
    elif id[0] == 'D':
        idCol = 'issue_id'

    query = """SELECT issue_id, sub_id FROM issue_map im, sub_issue si
               WHERE im.modeldb_id = si.issue_id and im.from_dt<=:date_arg
               AND im.thru_dt>:date_arg AND %(idCol)s = :id_arg
            """%{'idCol':idCol}
    modelDB.dbCursor.execute(query, date_arg = dt, id_arg = id)
    r = modelDB.dbCursor.fetchall()
    
    if len(r)>0:
        return r[0]
    else:
        return (None,None)

def getDateDict(fileName): 
    matrix = numpy.genfromtxt(fileName,dtype='str',delimiter=',')
    singleEntry=False 
    if len(list(matrix.shape))==1: 
        minfromDt = str(matrix[1].strip())
        maxthruDt = str(matrix[2].strip())
        idList = [matrix[0].strip()]
        singleEntry = True 
    else: 
        fromDt = matrix[:,1]
        thruDt = matrix[:,2]
        minfromDt = min(fromDt)
        maxthruDt = max(thruDt) 
        idList = matrix[:,0] 

    List = minfromDt.strip().split('-')
    minfromDt = datetime.date(int(List[0]),int(List[1]),int(List[2]))
    List = maxthruDt.strip().split('-')
    maxthruDt = datetime.date(int(List[0]),int(List[1]),int(List[2]))
    
    dateList = [] 
    fromDt = minfromDt
    thruDt = maxthruDt
    while fromDt <= thruDt: 
        dateList.append(fromDt)
        fromDt += datetime.timedelta(1)


    sidList = []
    for idx,id in enumerate(idList):
        id = id.strip()
        if id[0]=='D' and len(id)==12:
            sid = id
        elif id[0]=='D' and len(id) ==10:
            sid = id+"11"
        else:
            if singleEntry: 
                (iid,sid) = getSubIssueID(modelDB,id,(matrix[2]).strip())
            else: 
                (iid,sid) = getSubIssueID(modelDB,id,(matrix[idx][2]).strip())
        sidList.append(sid)

    if len(sidList)==1: 
        dtidList =[]
        for dt in dateList:
            dtidList.append((dt,[sidList[0]]))

    else: 
        dtIdxMap = dict([(j,i) for (i,j) in enumerate(dateList)])
        axidIdxMap = dict([(j,i) for (i,j) in enumerate(sidList)])

        dtidMatrix = ma.zeros((len(dateList),len(sidList)))
        for idx,axid in enumerate(sidList): 
            fromDtList =matrix[idx,1].strip().split('-')
            fromDt = datetime.date(int(fromDtList[0]),int(fromDtList[1]),int(fromDtList[2]))
            thruDtList =matrix[idx,2].strip().split('-')
            thruDt = datetime.date(int(thruDtList[0]),int(thruDtList[1]),int(thruDtList[2]))

            idxList = list(range(dtIdxMap[fromDt], dtIdxMap[thruDt]+1))
            ma.put(dtidMatrix[:, idx], idxList, 1)

        dtidMatrix = ma.masked_where(dtidMatrix==1,dtidMatrix)


        dtidList=[]
        for idx, dt in  enumerate(dateList):
            maskedIdx = ma.getmaskarray(dtidMatrix[idx:])
            maskedIdx = numpy.flatnonzero(ma.getmaskarray(dtidMatrix[idx,:]))
            if len(maskedIdx)!=0: 
                dtidList.append((dt, list(sidList[idx] for idx in maskedIdx)))


    return dtidList

if __name__=="__main__":
    usage = "usage: %prog [options] configfile cmdline-file"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addDefaultCommandLine(cmdlineParser)
    cmdlineParser.add_option("-u",action="store_true",
                             default=False,dest="updateDB",
                             help="change the database")
    cmdlineParser.add_option("-o",action="store_true",
                             default=False,dest="openID",
                             help="reset the fromDate")
    cmdlineParser.add_option("-r","--resync",action="store_true",
                             default=False,dest="resync",
                             help="resync the cumulative return")
    cmdlineParser.add_option("-t","--transfer",action="store_true",
                             default=False,dest="transfer",
                             help="transfer the data and return")

    (options_, args_) = cmdlineParser.parse_args()

    if len(args_)>2:
        cmdlineParser.error("Missing cmdline-file")
   
    configFile_ = open(args_[0])
    config_ = configparser.ConfigParser()
    config_.read_file(configFile_)
    configFile_.close()
    connections = Connections.createConnections(config_)
    modelDB = connections.modelDB
    Utilities.processDefaultCommandLine(options_, cmdlineParser)
    
    if not options_.updateDB:
        options_.testOnly=True
    if options_.updateDB:
        options_.testOnly=False

    cmdfile_ = open(args_[1],'r')

    recs=[]
    cum_return = {}
    iidDtMap = dict()

    dateDict = getDateDict(cmdfile_)


    for line in cmdfile_.readlines():
        (id, newFromdt, oldFromdt) = line.split(',')
        id = id.strip()
        oldDt = oldFromdt.strip()
        newDt = newFromdt.strip()
        if id[0] == 'D' and len(id) == 12:
            sid = id 
            iid = id[0:10]
        elif id[0] == 'D' and len(id) == 10:
            sid = id + "11"
            iid = id
        else:
            (iid, sid) = getSubIssueID(modelDB, id, oldFromdt)
            if iid is None or sid is None:
                logging.error('Cannot find sub_issue_id for axioma_id %s'%id)
                continue
        iidDtMap[iid] = (newDt, oldDt, sid)
        
    logging.info('Workng for %d issueID/subIssueID pairs'%len(list(iidDtMap.keys())))

    if options_.openID:
        updateTable(modelDB, iidDtMap, 'issue_map','modeldb_id')
        updateTable(modelDB, iidDtMap, 'issue','issue_id')
        updateTable(modelDB, iidDtMap, 'sub_issue','issue_id')

        if options_.testOnly:
            modelDB.revertChanges()
            logging.info('Reverting changes')
        else:
            modelDB.commitChanges()
            logging.info('Committing changes')

        #Use the createRMSID to do the update
        rmsGenerator = creatermsid.CreateRMSID(connections, None, None)
        modelID = ','.join('%s'%i for i in iidDtMap.keys())
        idList = transfer.createIssueIDList(modelID, connections)[0]

        rmsInfo = rmsGenerator.getRMSInfo(idList)
        options_.targetRM = None
        options_.addRecords = True
        options_.removeRecords = False
        options_.removeESTURecords = False
        options_.updateRecords = True
        options_.updateESTURecords = True
        options_.writeOutFile = False

        rmsInsertUpdate = creatermsid.InsertUpdateRMSID(connections, options_)
        rmsInsertUpdate.process(rmsInfo)
        logging.info('Finished updating rms_issue table')

        if options_.testOnly:
            modelDB.revertChanges()
            logging.info('Reverting changes')
        else:
            modelDB.commitChanges()
            logging.info('Committing changes')

    thisStatus = True       
    if options_.transfer:
        config_.set('DEFAULT','sections','FundamentalCurrencyData')
        config_.set('DEFAULT','sections','FundamentalNumberData')
        config_.set('DEFAULT','transfer','transferFundamentalData')

        config_.set('DEFAULT','sections','SubIssueData')
        config_.set('DEFAULT','sections','SubIssueReturn')
        config_.set('DEFAULT','sections','SubIssueCumulativeReturn')
        config_.set('DEFAULT','transfer','transferSubIssueData')

        config_.set('DEFAULT','sections','IndustryGroupGICS')
        config_.set('DEFAULT','sections','IndustryGICS-2006')
        config_.set('DEFAULT','sections','IndustryGICS-2008')
        config_.set('DEFAULT','sections','IndustryGICS-2014')

        config_.set('DEFAULT','sections','GICSCustomAU')
        config_.set('DEFAULT','sections','GICSCustomCA')
        config_.set('DEFAULT','sections','GICSCustomGB')
        config_.set('DEFAULT','sections','GICSCustomJP')
        config_.set('DEFAULT','sections','GICSCustomNA')
        config_.set('DEFAULT','sections','GICSCustomTW')
        config_.set('DEFAULT','sections','GICSCustomJP-2008')
        config_.set('DEFAULT','sections','IndustryGroupGICS-2008')
        config_.set('DEFAULT','transfer','transferIssueData')
        
        # Get Transfer by Date Dict 
        for item  in dateDict: 
#        for (iid, (newFromDt, oldFromDt, sid)) in iidDtMap.iteritems():
            dt,idList = item 
#            config_.set('DEFAULT','dates','%s:%s'%(newFromDt, oldFromDt))
            iosdt = '%04d-%02d-%02d'%(dt.year,dt.month,dt.day)
            config_.set('DEFAULT','dates','%s:%s'%(iosdt,iosdt))
            sidStr = ','.join (idList)
            config_.set('DEFAULT','sub-issue-ids','%s'%sidStr)
            # transfer.transferFundamentalData(config_, 'FundamentalCurrencyData', connections, options_)
            # transfer.transferFundamentalData(config_, 'FundamentalNumberData', connections, options_)
            # transfer.transferSubIssueData(config_,'SubIssueData',connections,options_)
            transfer.transferSubIssueData(config_,'SubIssueReturn',connections,options_)
            transfer.transferSubIssueData(config_,'SubIssueCumulativeReturn',connections,options_)

            #config_.set('DEFAULT','issue-ids',iid)
            # transfer.transferIssueData(config_,'IndustryGroupGICS',connections,options_)
            # transfer.transferIssueData(config_,'IndustryGICS-2006',connections,options_)
            #transfer.transferIssueData(config_,'IndustryGICS-2008',connections,options_)
            # transfer.transferIssueData(config_,'GICSCustomAU',connections,options_)
            # transfer.transferIssueData(config_,'GICSCustomCA',connections,options_)
            # transfer.transferIssueData(config_,'GICSCustomGB',connections,options_)
            # transfer.transferIssueData(config_,'GICSCustomJP',connections,options_)
            # transfer.transferIssueData(config_,'GICSCustomNA',connections,options_)
            # transfer.transferIssueData(config_,'GICSCustomJP-2008',connections,options_)
            # transfer.transferIssueData(config_,'IndustryGroupGICS-2008',connections,options_)
#        logging.info('Done data transfer for %s'%iid)
            logging.info('Done data transfer for %s'%dt)

        if options_.testOnly:
            modelDB.revertChanges()
            logging.info('Reverting changes')
        else:
            modelDB.commitChanges()
            logging.info('Committing changes')

    if options_.resync:
        resyncDict = getResyncDict(modelDB, list(iidDtMap.keys()))
        logging.info('Found %s sub_issue_ids need to perform resync'%len(list(resyncDict.keys())))
        for sub_issue_id in sorted(resyncDict.keys()):
            (min_dt, resyncDt, rmgId) = resyncDict[sub_issue_id]
            logging.info('Details - sub_issue_id: %s, resyncDate: %s, rmg_id: %s'%(sub_issue_id, resyncDt, rmgId))
            original_v = getCumulativeReturn(modelDB, sub_issue_id, min_dt)
            o_end_v = getCumulativeReturn(modelDB, sub_issue_id, resyncDt)
            if o_end_v == []:
                logging.info("Nothing found on the cum_return_table for %s on %s"%(sub_issue_id, resyncDt))
                continue
            else:
                o_end_v = o_end_v[0][0]
            adjFactor = computeResync(modelDB, sub_issue_id, min_dt + datetime.timedelta(1), resyncDt)
            if len(adjFactor) == 0:
                logging.error("No adjFactor can be found. Skipping %s"%sub_issue_id)
                continue
            else:
                adjFactor = adjFactor[0][0]
            if abs(adjFactor) <0.00001:
                logging.info("No need to resync. Skipping sub_issue_id %s"%sub_issue_id)
                continue
            insertCumRecord(modelDB, sub_issue_id, o_end_v/adjFactor, min_dt, rmgId)
            logging.info("Inserted records into cum_return record for sub_issue_id %s on date %04d-%02d-%02d"%(sub_issue_id, min_dt.year, min_dt.month,min_dt.day))
            startDt = min_dt + datetime.timedelta(1)
            startDt = str(startDt.isoformat()[0:10])
            logging.info("Now will proceed to transfer cumulative return for sub_issue_id %s between %s and %s"%(sub_issue_id, startDt, resyncDt))
            print(startDt, resyncDt)
            config_.set('DEFAULT','dates','%s:%s'%(startDt,resyncDt))
            config_.set('DEFAULT','sub-issue-ids',sub_issue_id)                                        
            transfer.transferSubIssueData(config_,'SubIssueCumulativeReturn',connections,options_)

            end_v = getCumulativeReturn(modelDB, sub_issue_id, resyncDt)
            if abs((end_v[0][0]/o_end_v)-1)>1e-8:
                logging.error("Error found after cum_resync. The original_value is %10f and the resync value is %10f. Please check sub_issue_id %s"%(o_end_v, end_v[0][0],sub_issue_id))
                thisStatus=False
                break
            else:
                logging.info("Resync done. Pre value =%f, post value=%f"%(o_end_v, end_v[0][0]))
                    
    if not options_.testOnly and thisStatus:
        modelDB.commitChanges()
        logging.info("Committing changes")
    else:
        modelDB.revertChanges()
        logging.info("Reverting changes")


    
