
import configparser
import datetime
import logging
import optparse
from marketdb import MarketDB
from riskmodels import ModelDB
from riskmodels import Connections
from riskmodels import Utilities

SRC_ID=900
REF_VAL='syncFromDates.py'

def getAxiomaIDs(db):
    db.dbCursor.execute("""SELECT axioma_id FROM asset_ref""")
    return set([i[0] for i in db.dbCursor.fetchall()])

def getAxiomaIDsFromList(idList, mdlDB):
    axids = set()
    for axid in idList:
        if len(axid) == 10:
            axids.add(axid)
        elif len(axid) == 2:
            # Treat as country code
            mdlDB.dbCursor.execute("""SELECT marketdb_id FROM issue_map im
               JOIN sub_issue si ON si.issue_id=im.modeldb_id
               JOIN risk_model_group rmg ON rmg.rmg_id=si.rmg_id
               WHERE rmg.mnemonic=:ctry""", ctry=axid)
            ctryIDs = [i[0] for i in mdlDB.dbCursor.fetchall()]
            logging.info('%d Axioma IDs loaded for %s', len(ctryIDs), axid)
            axids.update(ctryIDs)
    return axids

def getUCPStartDates(axids, mkt, ucpDtTable):
    """Returns a dictionary mapping Axioma IDs to their UCP start date.
    """
    mkt.dbCursor.execute("""SELECT axioma_id, min_dt from %(table)s"""
                         % {'table': ucpDtTable})
    datePairs = mkt.dbCursor.fetchall()
    axids = set(axids)
    ucpMap = dict([(axid, dt.date()) for (axid, dt) in datePairs
                   if axid in axids and dt is not None])
    #print ucpMap
    return ucpMap

def getIssueMapDates(axids, mdl):
    """Returns a dictionary mapping Axioma IDs to their earliest from_dt
    in issue_map and the corresponding Model ID.
    """
    mdl.dbCursor.execute("""SELECT marketdb_id, from_dt, thru_dt, modeldb_id
      FROM issue_map""")
    issueMapData = mdl.dbCursor.fetchall()
    axids = set(axids)
    issueMap = dict()
    for (axid, fromDt, thruDt, mdlID) in issueMapData:
        if axid in axids:
            if axid not in issueMap or fromDt.date() < issueMap[axid][0]:
                issueMap[axid] = (fromDt.date(), thruDt.date(), mdlID)
    #print issueMap
    return issueMap

def backDateIssue(axid, ucpDt, issueMapData, mdl):
    """Move from_dt for modeldb_id given in issueMapData to ucpDate
    which is earlier.
    """
    (issueMapFromDt, issueMapThruDt, mdlID) = issueMapData
    # Make sure this is the first Axioma ID mapped to the model ID
    mdl.dbCursor.execute("""SELECT min(from_dt) FROM issue_map
      WHERE modeldb_id=:mdlID""", mdlID=mdlID)
    minDt = mdl.dbCursor.fetchone()[0].date()
    #print minDt, issueMapFromDt, ucpDt
    if minDt < issueMapFromDt:
        logging.error('%s is not first Axioma ID mapped to %s. '
                      'issue-map from_dt: %s, ucp min dt: %s. Skipping',
                      axid, mdlID, issueMapFromDt, minDt)
        return
    assert(minDt == issueMapFromDt)
    # Get country mnemonic associated with the Model ID
    mdl.dbCursor.execute("""SELECT distinct mnemonic FROM risk_model_group rmg
      JOIN sub_issue si ON si.rmg_id=rmg.rmg_id
      WHERE si.issue_id=:mid""", mid=mdlID)
    countries = [i[0] for i in mdl.dbCursor.fetchall()]
    logging.info('Updating start date for %s/%s from %s to %s, rmgs: %s',
                 mdlID, axid, issueMapFromDt, ucpDt, ','.join(countries))
    # Update issue
    mdl.dbCursor.execute("""UPDATE issue SET from_dt=:dt
      WHERE issue_id=:mid""", dt=ucpDt, mid=mdlID)
    # Update issue_map
    mdl.dbCursor.execute("""UPDATE issue_map SET from_dt=:dt
      WHERE modeldb_id=:mid AND marketdb_id=:axid""",
                         dt=ucpDt, mid=mdlID, axid=axid)
    # Update sub_issue
    mdl.dbCursor.execute("""UPDATE sub_issue SET from_dt=:dt
      WHERE issue_id=:mid AND from_dt=:curDt""",
                         dt=ucpDt, mid=mdlID, curDt=issueMapFromDt)
    # Update rms_issue
    mdl.dbCursor.execute("""SELECT ri.rms_id, ri.from_dt, rms.from_dt,
      rmm.from_dt
      FROM rms_issue ri
      JOIN risk_model_serie rms ON rms.serial_id=ri.rms_id
      JOIN rmg_model_map rmm ON rmm.rms_id=ri.rms_id
      JOIN sub_issue si ON si.issue_id=ri.issue_id AND si.rmg_id=rmm.rmg_id
      WHERE ri.issue_id = :mid ORDER BY rms_id""", mid=mdlID)
    rmsIssueData = mdl.dbCursor.fetchall()
    for (rmsID, riDt, rmsDt, rmmDt) in rmsIssueData:
        newDt = max(ucpDt, rmmDt.date())
        if newDt != riDt.date():
            logging.debug('Updating rms_issue for %d, %s from %s to %s',
                          rmsID, mdlID, riDt.date(), newDt)
            mdl.dbCursor.execute("""UPDATE rms_issue SET from_dt=:dt
               WHERE rms_id=:rms AND issue_id=:mid AND from_dt=:curDt""",
                                 dt=newDt, rms=rmsID, mid=mdlID,
                                 curDt=riDt.date())
    # Update rms_estu_exclude
    mdl.dbCursor.execute("""SELECT re.rms_id, re.from_dt, ri.from_dt
      FROM rms_estu_excl_active_int re
      JOIN rms_issue ri ON ri.issue_id=re.issue_id AND ri.rms_id=re.rms_id
      WHERE re.issue_id = :mid ORDER BY re.rms_id""", mid=mdlID)
    reIssueData = mdl.dbCursor.fetchall()
    for (rmsID, reDt, riDt) in reIssueData:
        newDt = riDt.date()
        if reDt.date() == minDt:
            logging.debug('Updating rms_estu_excluded for %d, %s'
                          ' from %s to %s',
                          rmsID, mdlID, reDt.date(), newDt)
            mdl.dbCursor.execute("""INSERT INTO rms_estu_excluded
                SELECT rms_id, change_dt, change_del_flag, :src, :ref, sysdate, 'Y'
                FROM rms_estu_excl_active WHERE rms_id=:rms AND issue_id=:mid AND
                change_dt=:curDt""",
                                 rms=rmsID, mid=mdlID, curDt=reDt.date(),
                                 src=SRC_ID, ref=REF_VAL)
            mdl.dbCursor.execute("""INSERT INTO rms_estu_excluded
                (rms_id, issue_id, change_dt, change_del_flag, src_id, ref, rev_dt, rev_del_flag)
                VALUES(:rms, :mid, :dt, 'N', :src, :ref, sysdate, 'N')""",
                                 dt=newDt, rms=rmsID, mid=mdlID, src=SRC_ID, ref=REF_VAL)
        else:
            logging.warning('ESTU exclusion for %d, %s does not match issue'
                         ' life-time. Skipping', rmsID, mdlID)
def forwardDateIssue(axid, ucpDt, issueMapData, mdl):
    """Move from_dt for modeldb_id given in issueMapData to ucpDate
    which is later.
    """
    (issueMapFromDt, issueMapThruDt, mdlID) = issueMapData
    if issueMapThruDt <= ucpDt:
        logging.error('Updating from_dt for %s/%s to %s would create an'
                      'empty date range: thru_dt is %s',
                      mdlID, axid, ucpDt, issueMapThruDt)
        return
    # Make sure this is the first Axioma ID mapped to the model ID
    mdl.dbCursor.execute("""SELECT min(from_dt) FROM issue_map
      WHERE modeldb_id=:mdlID""", mdlID=mdlID)
    minDt = mdl.dbCursor.fetchone()[0].date()
    #print minDt, issueMapFromDt, ucpDt
    if minDt < issueMapFromDt:
        logging.error('%s is not first Axioma ID mapped to %s. '
                      'issue-map from_dt: %s, ucp min dt: %s. Skipping',
                      axid, mdlID, issueMapFromDt, minDt)
        return
    assert(minDt == issueMapFromDt)
    # Get country mnemonic associated with the Model ID
    mdl.dbCursor.execute("""SELECT distinct mnemonic FROM risk_model_group rmg
      JOIN sub_issue si ON si.rmg_id=rmg.rmg_id
      WHERE si.issue_id=:mid""", mid=mdlID)
    countries = [i[0] for i in mdl.dbCursor.fetchall()]
    logging.info('Updating start date for %s/%s from %s to %s, rmgs: %s',
                 mdlID, axid, issueMapFromDt, ucpDt, ','.join(countries))
    # Update issue
    mdl.dbCursor.execute("""UPDATE issue SET from_dt=:dt
      WHERE issue_id=:mid""", dt=ucpDt, mid=mdlID)
    # Update issue_map
    mdl.dbCursor.execute("""UPDATE issue_map SET from_dt=:dt
      WHERE modeldb_id=:mid AND marketdb_id=:axid""",
                         dt=ucpDt, mid=mdlID, axid=axid)
    # Update sub_issue
    mdl.dbCursor.execute("""UPDATE sub_issue SET from_dt=:dt
      WHERE issue_id=:mid AND from_dt=:curDt""",
                         dt=ucpDt, mid=mdlID, curDt=issueMapFromDt)
    # Update rms_issue
    mdl.dbCursor.execute("""SELECT ri.rms_id, ri.from_dt, rms.from_dt,
      rmm.from_dt
      FROM rms_issue ri
      JOIN risk_model_serie rms ON rms.serial_id=ri.rms_id
      JOIN rmg_model_map rmm ON rmm.rms_id=ri.rms_id
      JOIN sub_issue si ON si.issue_id=ri.issue_id AND si.rmg_id=rmm.rmg_id
      WHERE ri.issue_id = :mid ORDER BY rms_id""", mid=mdlID)
    rmsIssueData = mdl.dbCursor.fetchall()
    for (rmsID, riDt, rmsDt, rmmDt) in rmsIssueData:
        newDt = max(ucpDt, rmmDt.date())
        if newDt != riDt.date():
            logging.debug('Updating rms_issue for %d, %s from %s to %s',
                          rmsID, mdlID, riDt.date(), newDt)
            mdl.dbCursor.execute("""UPDATE rms_issue SET from_dt=:dt
               WHERE rms_id=:rms AND issue_id=:mid AND from_dt=:curDt""",
                                 dt=newDt, rms=rmsID, mid=mdlID,
                                 curDt=riDt.date())
    # Update rms_estu_exclude
    mdl.dbCursor.execute("""SELECT re.rms_id, re.from_dt, ri.from_dt
      FROM rms_estu_excl_active_int re
      JOIN rms_issue ri ON ri.issue_id=re.issue_id AND ri.rms_id=re.rms_id
      WHERE re.issue_id = :mid ORDER BY re.rms_id""", mid=mdlID)
    reIssueData = mdl.dbCursor.fetchall()
    for (rmsID, reDt, riDt) in reIssueData:
        newDt = riDt.date()
        if reDt.date() == minDt:
            logging.debug('Updating rms_estu_excluded for %d, %s'
                          ' from %s to %s',
                          rmsID, mdlID, reDt.date(), newDt)
            mdl.dbCursor.execute("""INSERT INTO rms_estu_excluded
                SELECT rms_id, change_dt, change_del_flag, :src, :ref, sysdate, 'Y'
                FROM rms_estu_excl_active WHERE rms_id=:rms AND issue_id=:mid AND
                change_dt=:curDt""",
                                 rms=rmsID, mid=mdlID, curDt=reDt.date(),
                                 src=SRC_ID, ref=REF_VAL)
            mdl.dbCursor.execute("""INSERT INTO rms_estu_excluded
                (rms_id, issue_id, change_dt, change_del_flag, src_id, ref, rev_dt, rev_del_flag)
                VALUES(:rms, :mid, :dt, 'N', :src, :ref, sysdate, 'N')""",
                                 dt=newDt, rms=rmsID, mid=mdlID, src=SRC_ID, ref=REF_VAL)
        else:
            logging.warning('ESTU exclusion for %d, %s does not match issue'
                         ' life-time. Skipping', rmsID, mdlID)

def main():
    usage = "usage: %prog config-file [options]"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addDefaultCommandLine(cmdlineParser)
    cmdlineParser.add_option("--update-database", action="store_false",
                             default=True, dest="testOnly",
                             help="commit changes the research database")
    cmdlineParser.add_option("--exclude", action="store",
                             default='', dest="excludeList",
                             help="comma-separated list of issues to exclude")
    cmdlineParser.add_option("--restrict", action="store",
                             default='', dest="restrictList",
                             help="comma-separated list of issues to process")
    cmdlineParser.add_option("--ucp-dt-table", action="store",
                             default='tmp_ucp_min_max_dt', dest="ucpDtTable",
                             help="table that store the UCP min dates")
    cmdlineParser.add_option("--no-forward-date", action="store_false",
                             default=True, dest="forwardDate",
                             help="forward-date issue from_dt based on UCP")
    cmdlineParser.add_option("--no-back-date", action="store_false",
                             default=True, dest="backDate",
                             help="back-date issue from_dt based on UCP")
    (options_, args_) = cmdlineParser.parse_args()
    if len(args_) != 1:
        cmdlineParser.error("Incorrect number of arguments")
    Utilities.processDefaultCommandLine(options_, cmdlineParser)
    
    configFile_ = open(args_[0])
    config_ = configparser.ConfigParser()
    config_.read_file(configFile_)
    configFile_.close()

    connections = Connections.createConnections(config_)
    mkt = connections.marketDB
    mdl = connections.modelDB

    if len(options_.excludeList) > 0:
        options_.excludeList = getAxiomaIDsFromList(
            options_.excludeList.split(','), mdl)
        logging.info('Excluding %d Axioma IDs from consideration',
                     len(options_.excludeList))
    else:
        options_.excludeList = set()
    
    if len(options_.restrictList) > 0:
        options_.restrictList = getAxiomaIDsFromList(
            options_.restrictList.split(','), mdl)
        logging.info('Restricting to %d Axioma IDs',
                     len(options_.restrictList))
    else:
        options_.restrictList = None
    
    axids = getAxiomaIDs(mkt)
    logging.info('%d Axioma IDs in marketdb', len(axids))
    if options_.restrictList is not None:
        axids &= options_.restrictList
    axids = axids - options_.excludeList
    if len(axids) > 0:
        logging.info('%d Axioma IDs after restictions and exclusions',
                     len(axids))
    else:
        logging.info('Nothing to do!')
        return
    
    axids = sorted(axids)
    ucpStartDates = getUCPStartDates(axids, mkt, options_.ucpDtTable)
    issueMapDates = getIssueMapDates(axids, mdl)
    for axid in axids:
        if axid not in ucpStartDates:
            logging.debug('No UCP start date for %s. Skipping', axid)
            continue
        if axid not in issueMapDates:
            logging.error('No issue_map date for %s. Skipping', axid)
            continue
        if ucpStartDates[axid] == issueMapDates[axid][0]:
            logging.debug('UCP and issue_map start dates match for %s', axid)
        elif ucpStartDates[axid] < issueMapDates[axid][0]:
            if options_.backDate:
                backDateIssue(axid, ucpStartDates[axid], issueMapDates[axid],
                              mdl)
        elif options_.forwardDate:
            forwardDateIssue(axid, ucpStartDates[axid], issueMapDates[axid],
                             mdl)
    
    if options_.testOnly:
        logging.info('Reverting changes')
        mkt.revertChanges()
        mdl.revertChanges()
    else:
        logging.info('Committing changes')
        mkt.commitChanges()
        mdl.commitChanges()
    mkt.finalize()
    mdl.finalize()

if __name__ == '__main__':
    main()
    
