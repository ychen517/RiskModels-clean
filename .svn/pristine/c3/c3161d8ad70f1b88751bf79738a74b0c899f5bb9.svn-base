
import datetime
import optparse
import logging
from marketdb import MarketDB
from riskmodels import ModelDB
from riskmodels import Utilities

def mapMarketToModel(modelDB, marketIssues):
    modelDB.dbCursor.execute("SELECT marketdb_id, modeldb_id FROM issue_map")
    pairs = modelDB.dbCursor.fetchall()
    modelIssues = set()
    for (mktID, mdlID) in pairs:
        if mktID in marketIssues:
            modelIssues.add(mdlID)
    return modelIssues

def getAllBenchmarkAssets(marketDB):
    marketDB.dbCursor.execute("""SELECT DISTINCT axioma_id FROM index_constituent""")
    constituents = set()
    r = marketDB.dbCursor.fetchmany()
    while len(r) > 0:
        for i in r:
            constituents.add(i[0])
        r = marketDB.dbCursor.fetchmany()
    return constituents

def getAllConstituents(marketDB, leafClass):
    assert(leafClass.isLeaf)
    marketDB.dbCursor.execute("""SELECT axioma_id
    FROM classification_constituent
    WHERE trans_from_dt <= :trans_arg AND trans_thru_dt > :trans_arg
    AND classification_id = :id_arg""",
                          trans_arg=marketDB.transDateTimeStr,
                          id_arg=leafClass.id)
    constituents = set()
    r = marketDB.dbCursor.fetchmany()
    while len(r) > 0:
        for i in r:
            constituents.add(i[0])
        r = marketDB.dbCursor.fetchmany()
    return constituents

def updateMdlSharesIntoRMSIssue(modelDB, allMdlShares, rms_id):
    modelDB.dbCursor.execute("SELECT issue_id FROM rms_issue WHERE rms_id=:rms_arg", rms_arg=rms_id)
    currentRMSIssues = set()
    for i in modelDB.dbCursor.fetchall():
        currentRMSIssues.add(i[0])
    print('%d asset present in rms_issue' % len(currentRMSIssues))
    modelDB.dbCursor.execute("SELECT issue_id, from_dt, thru_dt FROM issue")
    issueList = modelDB.dbCursor.fetchall()
    updateDicts = list()
    for (mdlId, from_dt, thru_dt) in issueList:
        if mdlId in allMdlShares:
            if mdlId in currentRMSIssues:
                logging.debug('asset %s already present' % mdlId)
            else:
                updateDicts.append(dict([('rms_arg', rms_id),
                                         ('id_arg', mdlId),
                                         ('from_arg', from_dt),
                                         ('thru_arg', thru_dt)]))
        elif mdlId in currentRMSIssues:
            logging.warning('asset %s currently present but not in update'
                         % mdlId)
    if len(updateDicts) > 0:
        modelDB.dbCursor.executemany("""INSERT INTO rms_issue (rms_id,
        issue_id, from_dt, thru_dt) VALUES(:rms_arg, :id_arg, :from_arg,
        :thru_arg)""", updateDicts)
    print('inserted %d issues' % len(updateDicts))

if __name__ == '__main__':
    usage = "usage: %prog [options]"
    date = datetime.date(2006, 4, 29)
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addModelAndDefaultCommandLine(cmdlineParser)
    cmdlineParser.add_option("-n", action="store_true",
                             default=False, dest="testOnly",
                             help="don't change the database")
    (options, args) = cmdlineParser.parse_args()
    if len(args) != 0:
        cmdlineParser.error("Incorrect number of arguments")
    
    modelClass = Utilities.processModelAndDefaultCommandLine(
        options, cmdlineParser)
    modelDB = ModelDB.ModelDB(sid=options.modelDBSID, user=options.modelDBUser,
                              passwd=options.modelDBPasswd)
    marketDB = MarketDB.MarketDB(sid=options.marketDBSID, user=options.marketDBUser, passwd=options.marketDBPasswd)
    riskModel = modelClass(modelDB, marketDB)
    print('RMS: %d' % riskModel.rms_id)
    assetTypeFamily = marketDB.getClassificationFamily('ASSET TYPES')
    assert(assetTypeFamily != None)
    assetTypeMembers = marketDB.getClassificationFamilyMembers(assetTypeFamily)
    memberDict = dict([(i.name,i) for i in assetTypeMembers])
    compuStatMember = memberDict['Compustat Asset Type']
    telekursMember = memberDict['Telekurs Asset Type']
    compuStatRoot = marketDB.getClassificationMemberRoot(compuStatMember, date)
    telekursRoot = marketDB.getClassificationMemberRoot(telekursMember, date)
    compuStatChildren = marketDB.getClassificationChildren(compuStatRoot)
    telekursChildren = dict(
        [(i.description, i) for i
         in marketDB.getClassificationChildren(telekursRoot)])
    compuStatChildren = dict(
        [(i.description, i) for i
         in marketDB.getClassificationChildren(compuStatRoot)])

    sharesClass = telekursChildren['Shares']
    assert(sharesClass.isLeaf)
    unitClass = telekursChildren['Trust-Shares']
    assert(unitClass.isLeaf)
    commonClass = compuStatChildren['Common or ordinary']
    assert(commonClass.isLeaf)
    prefClass = compuStatChildren['Preferred or preference']
    assert(prefClass.isLeaf)
    cprefClass = compuStatChildren['Convertible preferred']
    assert(cprefClass.isLeaf)
    adrClass = compuStatChildren['American Depository Receipt (ADR)']
    assert(adrClass.isLeaf)

    shareSet = getAllConstituents(marketDB, sharesClass)
    print('%d assets classified as Share' % len(shareSet))
    unitSet = getAllConstituents(marketDB, unitClass)
    print('%d assets classified as Unit' % len(unitSet))
    commonSet = getAllConstituents(marketDB, commonClass)
    print('%d assets classified as Common Share' % len(commonSet))
    prefSet = getAllConstituents(marketDB, prefClass)
    print('%d assets classified as Preferred' % len(prefSet))
    cprefSet = getAllConstituents(marketDB, cprefClass)
    print('%d assets classified as Convertible Preferred' % len(cprefSet))
    adrSet = getAllConstituents(marketDB, adrClass)
    print('%d assets classified as ADR' % len(adrSet))
    benchmarkSet = getAllBenchmarkAssets(marketDB)
    print('%d assets classified as benchmark assets' % len(benchmarkSet))

    allShares = shareSet | unitSet | commonSet | adrSet
    print('%d asset in all share classes combined' % len(allShares))
    allShares = (allShares - (prefSet | cprefSet)) | benchmarkSet
    print('%d assets available for risk models' % len(allShares))

    allMdlShares = mapMarketToModel(modelDB, allShares)
    print(len(allMdlShares))
    updateMdlSharesIntoRMSIssue(modelDB, allMdlShares, riskModel.rms_id)
    if options.testOnly:
        logging.info('Reverting changes')
        modelDB.revertChanges()
    else:
        modelDB.commitChanges()
    marketDB.finalize()
    modelDB.finalize()
