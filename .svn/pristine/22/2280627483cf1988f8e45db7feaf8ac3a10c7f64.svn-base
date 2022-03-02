#------------------------------------------------------------------------------
# This script will terminate assets in glprod that were terminated in the legacy database
# on the specified date

import logging
import optparse
import sys
import datetime
import configparser
from marketdb import Utilities
from marketdb import MarketDB
from marketdb import MarketID
from riskmodels import ModelDB
from riskmodels.Connections import createConnections,finalizeConnections

#------------------------------------------------------------------------------
# Get legacy marketdb ids terminated on the given date but not terminated in
# glprod
#
def getTerminatedAxIds(mktdb, date, remoteconn):
    query = """select b.axioma_id, b.thru_dt from asset_ref a, asset_ref@%s b
    where a.axioma_id=b.axioma_id and b.thru_dt=:date_arg and
    a.thru_dt > :date_arg""" % remoteconn
    mktdb.dbCursor.execute(query, date_arg=date)
    r = mktdb.dbCursor.fetchall()
    if len(r) > 0:
        return [(i[0], i[1].date()) for i in r]
    else:
        return []
    
#------------------------------------------------------------------------------
# Check to make sure it is not involved in a merger-survivor in legacy
# today
#
def hasMergerSurvivor(mktdb, date, axid, remoteconn, modeldb):
    query = """select distinct modeldb_id from %s.issue_map@%s a where
    modeldb_id in (select modeldb_id from %s.issue_map@%s where
    marketdb_id=:axid_arg) and not exists (select * from %s.issue_map@%s b
    where b.modeldb_id=a.modeldb_id and b.marketdb_id != :axid_arg and
    b.thru_dt>:date_arg)""" % (modeldb, remoteconn,
    modeldb, remoteconn, modeldb, remoteconn)
    mktdb.dbCursor.execute(query, date_arg=date, axid_arg=axid)
    r = mktdb.dbCursor.fetchall()
    if len(r) == 1:
        return False
    elif len(r) == 0:
        return True
    else:
        logging.error("Multiple records from issue_map for %s/%s" % (axid, date))
        return True
    
#------------------------------------------------------------------------------
# Get the sub issue id, rmg id and all the rms ids axid belongs to
#
def getSubIssueIdRMSIds(mdldb, axid, date):
    query = """select distinct sub_id, rmg_id, rms_id from 
    issue_map a, rms_issue b, modeldb_global.sub_issue c where 
    a.marketdb_id=:axid_arg and a.modeldb_id=b.issue_id and
    b.thru_dt > :date_arg and a.thru_dt > :date_arg
    and c.issue_id=a.modeldb_id and c.thru_dt > :date_arg"""

    mdldb.dbCursor.execute(query, axid_arg=axid, date_arg=date)
    r = mdldb.dbCursor.fetchall()
    rmsids = []
    subid = None
    rmgid = None
    for (sid, rid, rmsid) in r:
        if subid is not None and sid != subid:
            logging.error("Multiple sub issue ids for %s: %s/%s" % (axid, subid, sid))
            return (None, None, None)

        if subid is None:
            subid = sid
            rmgid = rid
        rmsids.append(rmsid)

    return (subid, rmgid, rmsids)

#------------------------------------------------------------------------------
if __name__ == '__main__':
    usage = "usage: %prog [options] <config file> <country> <date>."
    
    cmdlineParser = optparse.OptionParser(usage=usage)
    cmdlineParser.add_option("-u", "--update-database", action="store_true",
                             default=False, dest="updateDB",
                             help="run queries and commit")
    Utilities.addDefaultCommandLine(cmdlineParser)
    (options, args) = cmdlineParser.parse_args()
    
    Utilities.processDefaultCommandLine(options, cmdlineParser)
    if len(args) != 3:
        cmdlineParser.error("Incorrect number of arguments")
    
    configFile_ = open(args[0])

    config_ = configparser.ConfigParser()
    config_.read_file(configFile_)
    configFile_.close()
    connections_ = createConnections(config_)
    marketdb=connections_.marketDB
    modeldb=connections_.modelDB

    legacymktdb = None
    legacyremoteconn = 'mktdbus'
    legacymodeldb = 'modeldb'
    
    if args[1] == 'TW':
        legacyremoteconn = 'mktdbtw'
        legacymodeldb = 'modeldb_tw'
        
    if args[1] not in ['US', 'TW']:
        logging.error("This script is not applicable only to US and TW not %s" % args[1])
        sys.exit(0)
    date = Utilities.parseISODate(args[2])
    axids = getTerminatedAxIds(marketdb, date, legacyremoteconn)
    status = True
    for (axid, thrudt) in axids:
        logging.info("Processing %s/%s" % (axid, thrudt))
        # Now we are ready to terminate marketdb
        marketdb.deactivateAssets([MarketID.MarketID(string=axid)], thrudt)
        logging.info("Deactivated %s on %s" % (axid, thrudt))
        # Terminate modeldb only if it is not involved in a merger-survivor
        if not hasMergerSurvivor(marketdb, date, axid, legacyremoteconn, legacymodeldb):        
            subissueid, rmgid, rmsids = getSubIssueIdRMSIds(modeldb, axid, thrudt)
            sis = [ModelDB.SubIssue(string=subissueid)]
            issues = [si.getModelID() for si in sis]
            modeldb.deactivateIssues(issues, thrudt)
            logging.info("Deactivated issues %s on %s" % (",".join([i.getIDString() for i in issues]), thrudt))
            rmgid1 = ModelDB.RiskModelGroup(rmgid, "", "", "", {})
            modeldb.deactivateSubIssues(sis, rmgid1, thrudt)
            logging.info("Deactivated subissues %s on %s" % (",".join([i.getSubIDString() for i in sis]), thrudt))

            for rmsid in rmsids:
                try:
                    rmsdead = ModelDB.RMSDeadIssue(thrudt, issues[0].getIDString(), rmsid, 'Terminated in legacy')
                    rmsdead.apply(modeldb)
                    logging.info("Terminated rms_issue record %s/%d on %s" % (issues[0].getIDString(), rmsid, thrudt))
                except Exception as e:
                    logging.error("Exception: " + str(e))
                    logging.error(e)
                    status = False

    if status and options.updateDB:
        logging.info("Commiting all changes")
        marketdb.commitChanges()
        modeldb.commitChanges()
    else:
        logging.info("Reverting all changes")
        marketdb.revertChanges()
        modeldb.revertChanges()


    marketdb.finalize()
    modeldb.finalize()

    
            
            
