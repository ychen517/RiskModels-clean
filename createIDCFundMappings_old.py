
import optparse
import configparser
import logging
import sys
import pymssql
import datetime
from riskmodels import Connections
from riskmodels import Utilities
from riskmodels import ModelDB
from riskmodels.ModelID import ModelID

def getCountryMinMaxDate(cur, assetid, assetsrc):
    
    country = None
    if assetsrc == 'NA':
        query = """select min([Pricing Date]), max([Pricing Date]) from IDC.dbo.NAPrices where cusip in (
                    select cusip from IDC.dbo.NAAssets where AssetId=%(assetid)d)  and
                    [Exchange Code] in (select [Exchange Code] from IDC.dbo.NAAssets where AssetId=%(assetid)d)"""
        logging.debug(query)
        cur.execute(query % {'assetid': assetid})
        r = cur.fetchall()
        if len(r) == 1 and r[0][0]:
            mindt, maxdt = r[0][0], r[0][1]
            mindt, maxdt = datetime.datetime.strptime(mindt, "%Y-%m-%d").date(), datetime.datetime.strptime(maxdt, "%Y-%m-%d").date()
        else:
            if len(r) > 1:
                logging.error("ERROR: Too many price records found for %d/%s" % (assetid, assetsrc))
            elif len(r) == 0:
                logging.error("ERROR: No price records found for %d/%s" % (assetid, assetsrc))
            mindt = datetime.date(1950, 1, 1)
            maxdt = datetime.date(9999, 12, 31)

        query = """select country, min(FromDate), max(ToDate) from IDC.dbo.NAAssets where AssetId=%(assetid)d group by country"""
        #logging.debug(query)
        cur.execute(query % {'assetid': assetid})
        r = cur.fetchall()
        if len(r) == 1 and r[0][0]:
            country, miniddt, maxiddt = r[0][0], r[0][1], r[0][2]
            miniddt, maxiddt = datetime.datetime.strptime(miniddt, "%Y-%m-%d").date(), datetime.datetime.strptime(maxiddt, "%Y-%m-%d").date()
        else:
            if len(r) > 1:
                logging.error("ERROR: Too many price records found for %d/%s" % (assetid, assetsrc))
            elif len(r) == 0:
                logging.error("ERROR: No price records found for %d/%s" % (assetid, assetsrc))
            miniddt = datetime.date(1950, 1, 1)
            maxiddt = datetime.date(9999, 12, 31)
    else:
        query = """select c.[ISO_CountryCode], min([Pricing Date]), max([Pricing Date]) 
                   from IDC.dbo.NonNAPrices a, IDC.dbo.NonNAExchangeCodes b,
                   IDC.dbo.Country c where sedol in (
                     select sedol from IDC.dbo.NonNAAssets where assetid=%(assetid)d) 
                   and substring(a.[Exchange Code], 1, 2)=b.[Country Code] 
                   and c.[Code]=b.[Country Code]
                   group by c.[ISO_CountryCode]"""
        #logging.debug(query)
        cur.execute(query % {'assetid': assetid})
        r = cur.fetchall()
        if len(r) == 1 and r[0][0]:
            country, mindt, maxdt = r[0][0], r[0][1], r[0][2]
            mindt, maxdt = datetime.datetime.strptime(mindt, "%Y-%m-%d").date(), datetime.datetime.strptime(maxdt, "%Y-%m-%d").date()
        else:
            if len(r) > 1:
                logging.error("ERROR: Too many price records found for %d/%s" % (assetid, assetsrc))
            elif len(r) == 0:
                logging.error("ERROR: No price records found for %d/%s" % (assetid, assetsrc))
            mindt = datetime.date(1950, 1, 1)
            maxdt = datetime.date(9999, 12, 31)

        query = """select min(FromDate), max(ToDate) from IDC.dbo.NonNAAssets where AssetId=%(assetid)d"""
        #logging.debug(query)
        cur.execute(query % {'assetid': assetid})
        r = cur.fetchall()
        if len(r) == 1 and r[0][0]:
            miniddt, maxiddt = r[0][0], r[0][1]
            miniddt, maxiddt = datetime.datetime.strptime(miniddt, "%Y-%m-%d").date(), datetime.datetime.strptime(maxiddt, "%Y-%m-%d").date()
        else:
            if len(r) > 1:
                logging.error("ERROR: Too many price records found for %d/%s" % (assetid, assetsrc))
            elif len(r) == 0:
                logging.error("ERROR: No price records found for %d/%s" % (assetid, assetsrc))
            miniddt = datetime.date(1950, 1, 1)
            maxiddt = datetime.date(9999, 12, 31)

    if miniddt > mindt:
        mindt = miniddt
    if maxdt < maxiddt:
        maxdt = maxiddt

    if maxdt == datetime.date(2099, 12, 31):
        maxdt = datetime.date(9999, 12, 31)
    return country, mindt, maxdt

def getCurrentMappings(idccur, loc='devtest-mac-db-ny'):
    sconn = pymssql.connect(user='MarketDataLoader',
                            password='mdl1234',
                            server='devtest-mac-db-ny',
                            database='MarketData')
    sconn.autocommit(True)
    cur = sconn.cursor()
    cur.execute("select distinct AssetId, replace(AssetSource, 'IDCAsset_', ''), AxiomaDataId from IDCAssetsMapping where ApprovalFlg='Y' and AxiomaDataId is not null")
    r = cur.fetchall()
    mappings = [(i[0], i[1], i[2]) for i in r]

    mappingswithpricedts = []
    for assetid, assetsrc, axiomadataid in mappings:
        country, mindt, maxdt = getCountryMinMaxDate(idccur, assetid, assetsrc)
        #logging.debug('%d, %s, %s, %s, %s' % (assetid, assetsrc, mindt, maxdt, country))
        mappingswithpricedts.append((assetid, assetsrc, mindt, maxdt, country, axiomadataid))
    sconn.close()
    return mappingswithpricedts

def getAssetIds(idccur, idList, idType, historical=False):
    mappings = []
    if historical:
        historicalStr=''
    else:
        historicalStr = " and a.[Pricing Date]>='1-Jan-2016'"
    queryna = """select distinct assetid, a.cusip, a.%s from IDC.dbo.NAPrices a, IDC.dbo.NAAssets b where a.%s='%s' and a.cusip=b.cusip and a.[Exchange Code] = b.[Exchange Code]"""+historicalStr
    querynonna = """select distinct assetid, a.sedol, a.%s from IDC.dbo.NonNAPrices a, IDC.dbo.NonNAAssets b where a.%s='%s' and a.sedol=b.sedol"""+historicalStr

    idsFound = []
    for idval in idList:
        logging.debug("query to run: "+queryna % (idType, idType, idval))
        idccur.execute(queryna % (idType, idType, idval))
        r = idccur.fetchall()
        if len(r) == 1 and r[0][0]:
            assetid, cusip, vendorid = r[0][0], r[0][1], r[0][2]
            mappings.append((assetid, 'NA', None,idval))
            idsFound.append(idval)
            logging.info("Found mapping for %s/%s/%s in NA" % (idType, idval, assetid))
        elif len(r) > 1:
            logging.error("Too many records for %s/%s in NA" % (idType, idval))

    if idType.lower() != 'cusip':
        for idval in idList:
            idccur.execute(querynonna % (idType, idType, idval))
            r = idccur.fetchall()
            if len(r) == 1 and r[0][0]:
                assetid, sedol, vendorid = r[0][0], r[0][1], r[0][2]
                mappings.append((assetid, 'NonNA', None, idval))
                idsFound.append(idval)
                logging.info("Found mapping for %s/%s/%s in NonNA" % (idType, idval, assetid))
            elif len(r) > 1:
                logging.error("Too many records for %s/%s in NonNA" % (idType, idval))

    for idval in idList:
        if idval not in idsFound:
            logging.error("%s/%s not found in either NA or NonNA with prices after Jan 01 2016" % (idType, idval))

    mappingswithpricedts = []
    for assetid, assetsrc, axiomadataid, idval in mappings:
        country, mindt, maxdt = getCountryMinMaxDate(idccur, assetid, assetsrc)
        #logging.debug('%d, %s, %s, %s, %s' % (assetid, assetsrc, mindt, maxdt, country))
        mappingswithpricedts.append((assetid, assetsrc, mindt, maxdt, country, axiomadataid, idval))
    return mappingswithpricedts

def checkForMappingInMktDb(conn, currMappings):
    notInMktDb = []
    query = """select distinct axioma_id from asset_dim_idc_funds_active 
               where asset_id=:asset_id and asset_src=:asset_src"""
    conn.dbCursor.prepare(query)
    for i in currMappings:
        conn.dbCursor.execute(None, {'asset_id': i[0], 'asset_src': i[1]})
        r = conn.dbCursor.fetchall()
        if len(r) == 0 or r[0][0] is None:
            notInMktDb.append(i)
        else:
            logging.info("%s/%s already in MarketDB. Will not add" % (i[0], i[1]))
    return notInMktDb

def checkForMappingInMdlDb(conn, currMappings):
    notInMdlDb = []
    query = """select distinct modeldb_id from asset_dim_idc_funds_active a, modeldb_global.issue_map b 
               where asset_id=:asset_id and asset_src=:asset_src and a.axioma_id=b.marketdb_id"""
    conn.dbCursor.prepare(query)
    for i in currMappings:
        conn.dbCursor.execute(None, {'asset_id': i[0], 'asset_src': i[1]})
        r = conn.dbCursor.fetchall()
        if len(r) == 0 or r[0][0] is None:
            notInMdlDb.append(i)
        else:
            logging.info("%s/%s already in ModelDB. Will not add" % (i[0], i[1]))

    return notInMdlDb

def createNewModelDBRecords(marketDB, conn, newmdldbmappings, axdataidmap, newmdlids):
    myReturnMap = {}
    assetidMap = {}
    query = """select distinct axioma_id, asset_src from marketdb_global.asset_dim_idc_funds where 
               asset_id=:asset_id and asset_src=:asset_src"""
    marketDB.dbCursor.prepare(query)
    for i in newmdldbmappings:
        marketDB.dbCursor.execute(None, {'asset_id': i[0], 'asset_src': i[1]})
        r = marketDB.dbCursor.fetchall()
        if len(r) > 0 and r[0][0] is not None:
            assetidMap[(i[0], i[1])] = r[0][0]
    
    query = """select rmg_id, mnemonic from risk_model_group"""
    conn.dbCursor.execute(query)
    r = conn.dbCursor.fetchall()
    rmgMap = dict([(i[1], i[0]) for i in r])
    
    imquery = """insert into issue_map (marketdb_id, modeldb_id, from_dt, thru_dt)
                 values (:marketdb_id, :modeldb_id, :from_dt, :thru_dt)"""
    issquery = """insert into issue (issue_id, from_dt, thru_dt)
                    values (:issue_id, :from_dt, :thru_dt)"""
    subissquery = """insert into sub_issue (issue_id, from_dt, thru_dt, sub_id, rmg_id)
                       values (:issue_id, :from_dt, :thru_dt, :sub_id, :rmg_id)"""

    imdictList = []
    issdictList = []
    subissdictList = []
    usedct = 0
    msg=[]
    for (assetid, assetsrc, mindt, maxdt, country, axiomadataid, idValue) in newmdldbmappings:
        if maxdt == datetime.date(9999, 12, 31):
            maxdt = datetime.date(2999, 12, 31)
        mktdbid = assetidMap.get((assetid, assetsrc), None)
        mdldbid = axdataidmap.get(axiomadataid, None)
        if mktdbid is None or rmgMap[country] is None:
            s1="Could not find mktdb id for %d/%s or rmg for %s" % (assetid, assetsrc, country)
            msg.append(s1)
            logging.error(s1)
        else:
            if mdldbid is None:
                mdldbid = newmdlids[usedct]
                usedct = usedct + 1
            imdict = {'marketdb_id': mktdbid,
                      'modeldb_id': mdldbid,
                      'from_dt': mindt,
                      'thru_dt': maxdt}
            imdictList.append(imdict)            
            myReturnMap[idValue]=imdict
            
            issdict = {'issue_id': mdldbid,
                       'from_dt': mindt,
                       'thru_dt': maxdt}
            issdictList.append(issdict)
            
            subissdict = {'issue_id': mdldbid,
                          'from_dt': mindt,
                          'thru_dt': maxdt,
                          'sub_id': mdldbid+'11',
                          'rmg_id': rmgMap[country]}
            subissdictList.append(subissdict)
            
    if len(imdictList) > 0:
        #print imdictList
        conn.dbCursor.executemany(imquery, imdictList)
    if len(issdictList) > 0:
        #print issdictList
        conn.dbCursor.executemany(issquery, issdictList)
    if len(subissdictList) > 0:
        #print subissdictList
        conn.dbCursor.executemany(subissquery, subissdictList)
    
    return (imdictList,issdictList,subissdictList, msg, myReturnMap)
        
def createMappings(connections,idList, idType, srcId, addModelDb, useAxiomaDataId, historical=False):
    
    marketDB = connections.marketDB
    modelDB = connections.modelDB
    qaDirect = connections.qaDirect
    
    #print idType, idList
    if idType is None:
        currMappings = getCurrentMappings(qaDirect.dbCursor)
    else:
        currMappings = getAssetIds(qaDirect.dbCursor, idList, idType, historical)

    logging.debug('mappings for %s: %s', idType,currMappings)
    numids = len(currMappings)
    if numids <1:
        return [[],[],[],[],[]]
    newMarketDbMappings = checkForMappingInMktDb(marketDB, currMappings)
    #logging.debug(newMarketDbMappings)
    newModelDbMappings = checkForMappingInMdlDb(marketDB, currMappings)
    logging.debug('newMarketDbMappings: %d', len(newMarketDbMappings))
    logging.debug('newModelDbMappings: %s', len(newModelDbMappings))
#    logging.debug('newMarketDbMappings: %s', newMarketDbMappings)
#    logging.debug('newModelDbMappings: %s', newModelDbMappings)
    
    arValueDicts = []
    valueDicts = []
    
    returnVal=dict()
    for asset in currMappings:
        returnVal[str(asset[6])]={"region":asset[1]}
        
    
    arValueDictsToReturn = []
    
    newmdlids = []
    res =[]
    createdMappings = None
    
    if len(newMarketDbMappings) > 0:
        ref = 'Ported from Devtest-IDCAssetsMapping'
        if idType is not None:
            ref = 'Created from IDC Funds data directly'
        ret=marketDB.createNewMarketIDs(numids)
        axids = ['O'+r.getIDString()[1:] for r in ret]
        logging.debug(len(axids))
        logging.debug(axids)
        assetrefInsertQuery = """INSERT INTO asset_ref 
                                 (axioma_id, from_dt, thru_dt, src_id, ref, add_dt, trading_country)
                                 VALUES (:axioma_id, :from_dt, :thru_dt, :src_id, :ref, :add_dt, :trading_country)"""
        

        insertQuery = """INSERT INTO asset_dim_idc_funds
            (axioma_id, asset_id, asset_src, change_dt, change_del_flag, rev_dt, rev_del_flag, src_id, ref)
            VALUES (:axioma_id, :asset_id, :asset_src, :change_dt, :change_del_flag, :rev_dt, :rev_del_flag,
            :src_id, :ref)"""

        
        i = 0
        changedt = datetime.date(1950, 1, 1)
        revdt = datetime.datetime.now()
        for assetid, assetsrc, mindt, maxdt, country, axiomadataid, idVal in currMappings:
            logging.info("New asset to be added: assetid %s, assetsrc %s, mindt %s, maxdt %s, country %s, axiomadataid %s" % (assetid, assetsrc, mindt, maxdt, country, axiomadataid))
            if country is None:
                logging.error("ERROR: Not creating any records for %d/%s" % (assetid, assetsrc))
            else:
                arValueDict = {'axioma_id': axids[i],
                               'from_dt': mindt,
                               'thru_dt': maxdt,
                               'src_id': srcId,
                               'ref': ref,
                               'add_dt': revdt,
                               'trading_country': country
                               }
                arValueDicts.append(arValueDict)
                retDict = {'id_value': idVal}
                retDict.update(arValueDict)
                arValueDictsToReturn.append(retDict)

                valueDict = {'axioma_id': axids[i],
                             'asset_id': assetid,
                             'asset_src': assetsrc,
                             'change_dt': changedt,
                             'change_del_flag': 'N',
                             'rev_dt': revdt,
                             'rev_del_flag': 'N',
                             'src_id': srcId,
                             'ref': ref}
                valueDicts.append(valueDict)
            i = i + 1
            
            returnVal[str(idVal)]["asset_ref"] = arValueDict
            returnVal[str(idVal)]["asset_dim_idc_funds"] = valueDicts
            

        marketDB.dbCursor.executemany(assetrefInsertQuery, arValueDicts)
        marketDB.dbCursor.executemany(insertQuery, valueDicts)

    if len(newModelDbMappings) > 0 and addModelDb:
#         if options.update:
#             logging.info("Committing Changes to MarketDB")
#             marketDB.commitChanges()

        new = 0
        mdlidindexes = []
        if useAxiomaDataId:
            for i in newModelDbMappings:
                if i[5] is not None:
                    mdlidindexes.append(i[5])
                else:
                    new = new + 1
        else:
            new = len(newModelDbMappings)

        axdataidMap = {}
        if len(mdlidindexes) > 0:
            for i in mdlidindexes:
                mdldbid = ModelID.ModelID(index=i).getIDString()
                axdataidMap[i] = mdldbid

        newmdlids=modelDB.createNewModelIDs(new)
        newmdlids = [r.getIDString() for r in newmdlids]
        logging.debug("New modeldbids(%d): %s",len(newmdlids),newmdlids)
        res = createNewModelDBRecords(marketDB, modelDB, newModelDbMappings, axdataidMap, newmdlids)
        for key, value in res[4].items():
            returnVal[key]["issue_map"] = value
        createdMappings=res[4]
#     return [[x['axioma_id'] for x in arValueDicts],[x['axioma_id'] for x in valueDicts],newmdlids,res]
    return [arValueDictsToReturn,valueDicts,newmdlids,res, [returnVal]]
                
if __name__=='__main__':
    usage = "usage: %prog [--update-database] config-file"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addDefaultCommandLine(cmdlineParser)
    cmdlineParser.add_option("--update-database", action="store_true",
                             default=False, dest="update",
                             help="change the database")
    cmdlineParser.add_option("--add-modeldb", action="store_true",
                             default=False, dest="addmodeldb",
                             help="set up the asset in modeldb")
    cmdlineParser.add_option("--use-axiomadataid", action="store_true",
                             default=False, dest="useaxiomadataid",
                             help="use axiomadataid from sql server to create modeldbid")
    cmdlineParser.add_option("--src-id", action="store", type="int",
                             default=901, dest="srcid",
                             help="integer value of src_id from meta_sources")
    (options, args) = cmdlineParser.parse_args()

    idType = None
    idList = []
    if len(args) > 2:
        cmdlineParser.error("incorrect usage")
    elif len(args) == 2:
        idTypeVals = args[1]
        if idTypeVals.find('=') > 0:
            idType, idVals = idTypeVals.split("=")
            idList = list(set([i.strip() for i in idVals.split(",")]))
        else:
            cmdlineParser.error("Second argument must be idtype=idval1,idval2,...")

    Utilities.processDefaultCommandLine(options, cmdlineParser, disable_existing_loggers=False)
    configFile_ = open(args[0])
    config_ = configparser.ConfigParser()
    config_.read_file(configFile_)
    configFile_.close()
    
    connections_ =  None
    try:
        connections_ = Connections.createConnections(config_)
        results = createMappings(connections_, idList, idType, options.srcid, options.addmodeldb, options.useaxiomadataid)
        logging.info("Mappings created; results: %s", results)
        if options.update:
            logging.info("Committing Changes")
            connections_.marketDB.commitChanges()
            connections_.modelDB.commitChanges()
        else:
            logging.info("Reverting Changes; use --update-database to commit")
            connections_.revertAll()
    #         marketDB.revertChanges()
    #         modelDB.revertChanges()
#         Connections.finalizeConnections(connections_)
        sys.exit(0)
    except Exception as e:
        logging.exception('An Error occurred: %s', e)
#         Connections.finalizeConnections(connections_)
        sys.exit(1)
    finally:    
        Connections.finalizeConnections(connections_)
    
