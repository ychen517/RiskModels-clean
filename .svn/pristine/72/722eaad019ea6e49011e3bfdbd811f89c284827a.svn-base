import configparser
import datetime
import logging
import optparse
from riskmodels import Connections
from riskmodels import ModelDB
from riskmodels import ModelID
from riskmodels import Utilities

def createModelIDs(connections, lastDataDate, updateRMS, axidList):
    logging.root.setLevel(logging.INFO)
    mdl = connections.modelDB
    mdl_dc = mdl.dbCursor
    mkt_db = connections.marketDB
    mkt_dc = mkt_db.dbCursor

    # Get basic info on MarketDB IDs we're interested in
    logging.info('Searching for MarketDB IDs not yet in ModelDB')
    try:
        mkt_dc.execute('DROP TABLE tmp_new_ids')
    except:
        True

    if axidList is None:
        mkt_dc.execute("""
            CREATE TABLE tmp_new_ids AS
            SELECT a.axioma_id, a.from_dt, a.thru_dt
            FROM asset_ref a 
            WHERE NOT EXISTS (SELECT * FROM modeldb_global.issue_map im
                WHERE im.marketdb_id = a.axioma_id)""")
    elif axidList.lower().find("select") < 0:
        axids = [axid.strip() for axid in axidList.split(",")]
        mkt_dc.execute("""
            CREATE TABLE tmp_new_ids AS
            SELECT a.axioma_id, a.from_dt, a.thru_dt
            FROM asset_ref a 
            WHERE axioma_id in ('%s')""" % "','".join(axids))
    elif axidList.lower().find("select") >= 0:
        axids = [axid.strip() for axid in axidList.split(",")]
        mkt_dc.execute("""
            CREATE TABLE tmp_new_ids AS
            SELECT a.axioma_id, a.from_dt, a.thru_dt
            FROM asset_ref a 
            WHERE axioma_id in (%s)""" % axidList)
        
    # Get these assets' from/thru dates
    # This looks inefficient but it's in fact the fastest way to do this
    logging.info('Determining from/thru dates for new IDs')
    mkt_dc.execute("""
        UPDATE tmp_new_ids k
        SET k.from_dt = (SELECT NVL(MIN(dt), k.from_dt) FROM asset_dim_ucp_active u
            WHERE u.axioma_id = k.axioma_id),
            k.thru_dt = (SELECT NVL(MAX(dt)+1,k.thru_dt) FROM asset_dim_ucp_active u
            WHERE u.axioma_id = k.axioma_id)""")

    # Get list of MarketDB IDs from asset_ref that we're interested in
    mkt_dc.execute("""
        SELECT a.axioma_id, k.from_dt, k.thru_dt
        FROM asset_ref a, tmp_new_ids k
        WHERE k.axioma_id = a.axioma_id""")
    r = sorted(mkt_dc.fetchall())
    marketdb_ids = [row[0] for row in r]
    assetDatesDict = dict(zip(marketdb_ids, [(row[1],row[2]) for row in r]))
    marketdb_ids = set(marketdb_ids)
    num_ids = len(marketdb_ids)
    logging.info('%d new MarketDB IDs found', num_ids)
    mkt_dc.execute("DROP TABLE tmp_new_ids")

    # Fix up some thru_dt's b/c MarketDB thru_dt's are 9999
    n = 0
    for k in assetDatesDict.keys():
        asset = assetDatesDict[k]
        # End-date some junk that haven't traded for a while
        if asset[1].date() < lastDataDate - datetime.timedelta(365):
            assetDatesDict[k] = (asset[0].date(), asset[1].date())
        else:
            assetDatesDict[k] = (asset[0].date(), datetime.date(2999,12,31))
            n += 1
    logging.info('Fixed %d thru_dt records', n)

    # Assign new ModelDB IDs
    mdlIDDict = dict(zip(marketdb_ids, mdl.createNewModelIDs(num_ids, dummy=1)))

    # Update ISSUE_MAP based on these start/end dates
    imDataDict = [dict([('mdl_id_arg', mdlIDDict[id].getIDString()),
                      ('mkt_id_arg', id),
                      ('from_arg', assetDatesDict[id][0]),
                      ('thru_arg', assetDatesDict[id][1])])
                for id in assetDatesDict.keys()]
    mdl_dc.executemany("""INSERT INTO issue_map VALUES(
            :mdl_id_arg, :mkt_id_arg, :from_arg, :thru_arg)""", imDataDict)
    logging.info('Inserted %d rows into ISSUE_MAP', len(imDataDict))

    # Update ISSUE table
    # Remember, issue id's may span multiple marketdb IDs
    mdl_dc.execute("""
        SELECT modeldb_id, MIN(from_dt), MAX(thru_dt), MIN(marketdb_id)
        FROM issue_map 
        WHERE modeldb_id NOT IN 
            (SELECT issue_id FROM sub_issue)
        GROUP by modeldb_id""")
    rr = mdl_dc.fetchall()
    dataDict = [dict([('mdl_id_arg', row[0]),
                      ('from_arg', row[1]),
                      ('thru_arg', row[2])])
                for row in rr if row[3] in marketdb_ids]
    mdl_dc.executemany("""INSERT INTO issue VALUES(
            :mdl_id_arg, :from_arg, :thru_arg)""", dataDict)
    logging.info('Inserted %d rows into ISSUE', len(dataDict))
    
    # Update SUB_ISSUE table
    # Determine RMG based on market classification in MarketDB
    regFamily = mkt_db.getClassificationFamily('REGIONS')
    mktMember = [m for m in mkt_db.getClassificationFamilyMembers(regFamily)
                 if m.name == 'Market'][0]
    today = datetime.date.today()
    mktRoot = mkt_db.getClassificationMemberRoot(mktMember, today)
    countries = mkt_db.getClassificationChildren(mktRoot)
    mktIdCountryMap = dict()
    for country in countries:
        axids = set()
        crefs = [country]
        while len(crefs) > 0:
            cref = crefs.pop()
            if cref.isLeaf:
                mkt_dc.execute("""SELECT axioma_id FROM classification_const_active
                  WHERE change_del_flag='N' and classification_id=:cid""",
                               cid=cref.id)
                axids.update([i[0] for i in mkt_dc.fetchall()])
            else:
                crefs.extend(mkt_db.getClassificationChildren(cref))
        mktIdCountryMap.update(dict([(m, country) for m in axids]))
    mdl_dc.execute("""SELECT mnemonic, rmg_id FROM risk_model_group""")
    countryRMGMap = dict(mdl_dc.fetchall())
    assetMarketDict = dict([(axid, countryRMGMap[ctry.code])
                            for (axid, ctry) in mktIdCountryMap.items()
                            if ctry.code in countryRMGMap])
    missing = [a for a in marketdb_ids if a not in assetMarketDict]
    if len(missing) > 0:
        logging.info('Cannot find RMG info for %d assets: %s',
                     len(missing), missing)

    dataDict = [dict([('sub_id_arg', row[0]+'11'),
                      ('mdl_id_arg', row[0]),
                      ('from_arg', row[1]),
                      ('thru_arg', row[2]),
                      ('rmg_arg', assetMarketDict[row[3]])])
                for row in rr if row[3] in assetMarketDict
                and row[3] in marketdb_ids]
    mdl_dc.executemany("""INSERT INTO sub_issue VALUES(
            :mdl_id_arg, :from_arg, :thru_arg, :sub_id_arg, :rmg_arg)""", dataDict)
    logging.info('Inserted %d rows into SUB_ISSUE', len(dataDict))

    # Add to risk models based on rmg_model_map
    if updateRMS:
        mdl_dc.execute("""SELECT rmg.rmg_id, rmm.rms_id, rmm.from_dt, rmm.thru_dt, rm.name
           FROM risk_model_group rmg
           JOIN rmg_model_map rmm on rmm.rmg_id=rmg.rmg_id
           JOIN risk_model_serie rms ON rmm.rms_id=rms.serial_id
           JOIN risk_model rm ON rm.model_id=rms.rm_id
           WHERE rm.name NOT LIKE 'FX%'""")
        rmgRmsMap = dict()
        modelDataDicts = list()
        for (rmg, rms, fromDt, thruDt, rmName) in mdl_dc.fetchall():
            rmgRmsMap.setdefault(rmg, list()).append(
                (rms, fromDt.date(), thruDt.date(), rmName))
        for data in dataDict:
            from_dt = data['from_arg'].date()
            thru_dt = data['thru_arg'].date()
            mid = data['mdl_id_arg']
            rmgId = data['rmg_arg']
            if rmgId not in rmgRmsMap:
                continue
            for rmsId, mapFromDt, mapThruDt, rmName in rmgRmsMap[rmgId]:
                if from_dt < mapThruDt and mapFromDt < thru_dt:
                    modelDataDicts.append({
                            'rms_arg': rmsId,
                            'from_arg': max(from_dt, mapFromDt),
                            'thru_arg': min(thru_dt, mapThruDt),
                            'issue_arg': mid})
        if len(modelDataDicts) > 0:
            mdl_dc.executemany("""INSERT INTO rms_issue
               (rms_id, issue_id, from_dt, thru_dt)
               VALUES(:rms_arg, :issue_arg, :from_arg, :thru_arg)""",
                               modelDataDicts)
            logging.info('Inserted %d rows into RMS_ISSUE', len(modelDataDicts))

    outfile = open('new-ids.csv', 'w')
    mdl_dc.execute("""SELECT rmg_id, description FROM risk_model_group""")
    rmgMarketMap = dict(mdl_dc.fetchall())
    tmp = sorted((data['mdl_id_arg'], data) for data in imDataDict)
    dataDict = [i[1] for i in tmp]
    for data in dataDict:
        from_dt = data['from_arg']
        thru_dt = data['thru_arg']
        mktId = data['mkt_id_arg']
        rmg_id = assetMarketDict.get(mktId, None)
        if rmg_id is None:
            continue
        rmg = rmgMarketMap[rmg_id]
        str = '%s, %s, %s, %s, %s' % (data['mdl_id_arg'], mktId, rmg, from_dt, thru_dt)
        outfile.write('%s\n' % str)
        logging.info('Created %s', str)
    outfile.close()

if __name__ == '__main__':
    usage = "usage: %prog [options] config-file"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addDefaultCommandLine(cmdlineParser)
    cmdlineParser.add_option("--update-database", action="store_false",
                             default=True, dest="testOnly",
                             help="update the database")
    cmdlineParser.add_option("--last-data-date", action="store",
                             default=True, dest="lastDataDate",
                             help="last data date in database")
    cmdlineParser.add_option("--axioma-ids", action="store",
                             default=None, dest="axids",
                             help="List of axioma ids")
    cmdlineParser.add_option("--update-rms", action="store_true",
                             default=False, dest="updateRMS",
                             help="insert records into RMS_ISSUE")
    (options_, args_) = cmdlineParser.parse_args()
    if len(args_) != 1:
        cmdlineParser.error("Incorrect number of arguments")
    Utilities.processDefaultCommandLine(options_, cmdlineParser)
    configFile_ = open(args_[0])
    config_ = configparser.ConfigParser()
    config_.read_file(configFile_)
    configFile_.close()
    connections_ = Connections.createConnections(config_)
    
    try:
        lastDataDate_ = Utilities.parseISODate(options_.lastDataDate)
    except Exception as ex:
        lastDataDate_ = datetime.date.today()
    logging.info('Last data date: %s, terminating thru_dt for assets with last trade prior to %s' 
            % (lastDataDate_, lastDataDate_ - datetime.timedelta(365)))
    success_ = False
    try:
        createModelIDs(connections_, lastDataDate_, options_.updateRMS, options_.axids)
        success_ = True
    except Exception as ex:
        logging.fatal('Oh Crap', exc_info=True)
        
    if options_.testOnly or not success_:
        logging.info('Reverting changes')
        connections_.modelDB.revertChanges()
        connections_.marketDB.revertChanges()
    else:
        logging.info('Committing changes')
        connections_.modelDB.commitChanges()
        connections_.marketDB.commitChanges()
    Connections.finalizeConnections(connections_)
