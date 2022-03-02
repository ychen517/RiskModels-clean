
import datetime
import logging
import optparse
from riskmodels import ModelDB
from riskmodels import Utilities

SRC_ID=900
REF_VAL='syncIssueMetaData.py'

def getIssues(db):
    db.dbCursor.execute("""SELECT issue_id FROM issue""")
    return set([i[0] for i in db.dbCursor.fetchall()])

def getIssueMetaData(mid, db):
    # Get issue data
    db.execute("""SELECT from_dt, thru_dt FROM issue
      WHERE issue_id=:mid""", mid=mid)
    r = db.fetchone()
    rval = Utilities.Struct()
    issue = Utilities.Struct()
    rval.issue = issue
    issue.issue_id = mid
    issue.from_dt = r[0].date()
    issue.thru_dt = r[1].date()
    # Get issue_map
    db.execute("""SELECT marketdb_id, from_dt, thru_dt
      FROM issue_map WHERE modeldb_id=:mid
      ORDER BY from_dt ASC""", mid=mid)
    issueMap = list()
    for r in db.fetchall():
        val = Utilities.Struct()
        val.marketdb_id = r[0]
        val.from_dt = r[1].date()
        val.thru_dt = r[2].date()
        issueMap.append(val)
    rval.issueMap = issueMap
    # Get sub_issue
    db.execute("""SELECT rmg_id, from_dt, thru_dt, sub_id
      FROM sub_issue WHERE issue_id=:mid
      ORDER BY from_dt ASC""", mid=mid)
    subIssue = list()
    for r in db.fetchall():
        val = Utilities.Struct()
        val.rmg_id = int(r[0])
        val.from_dt = r[1].date()
        val.thru_dt = r[2].date()
        val.sub_id = r[3]
        subIssue.append(val)
    rval.subIssue = subIssue
    # Get rms_issue
    db.execute("""SELECT rms_id, from_dt, thru_dt
      FROM rms_issue WHERE issue_id=:mid AND rms_id > 0
      ORDER BY rms_id, from_dt ASC""", mid=mid)
    rmsIssue = list()
    for r in db.fetchall():
        val = Utilities.Struct()
        val.rms_id = int(r[0])
        val.from_dt = r[1].date()
        val.thru_dt = r[2].date()
        rmsIssue.append(val)
    rval.rmsIssue = rmsIssue
    # Get rms_estu_excluded
    db.execute("""SELECT rms_id, change_dt, change_del_flag, src_id, ref
      FROM rms_estu_excl_active WHERE issue_id=:mid AND rms_id > 0
      ORDER BY rms_id, change_dt ASC""", mid=mid)
    rmsESTUExcluded = list()
    for r in db.fetchall():
        val = Utilities.Struct()
        val.rms_id = int(r[0])
        val.change_dt = r[1].date()
        val.change_del_flag = r[2]
        val.src_id = r[3]
        val.ref = r[4]
        rmsESTUExcluded.append(val)
    rval.rmsESTUExcluded = rmsESTUExcluded
    # Get ca_spin_off
    db.execute("""SELECT dt, ca_sequence, child_id, share_ratio,
      implied_div, currency_id, ref, rmg_id
      FROM ca_spin_off WHERE parent_id=:mid
      ORDER BY dt, ca_sequence ASC""", mid=mid)
    spinOff = list()
    for r in db.fetchall():
        val = Utilities.Struct()
        val.dt = r[0].date()
        val.ca_sequence = int(r[1])
        val.child_id = r[2]
        share_ratio = r[3]
        if share_ratio is not None:
            share_ratio = float(r[3])
        val.share_ratio = share_ratio
        implied_div = r[4]
        if implied_div is not None:
            implied_div = float(implied_div)
        val.implied_div = implied_div
        currency_id = r[5]
        if currency_id is not None:
            currency_id = int(currency_id)
        val.currency_id = currency_id
        val.ref = r[6]
        val.rmg_id = int(r[7])
        spinOff.append(val)
    rval.spinOff = spinOff
    # Get ca_merger_survivor
    db.execute("""SELECT dt, ca_sequence, new_marketdb_id, old_marketdb_id,
      share_ratio, cash_payment, currency_id, ref, rmg_id
      FROM ca_merger_survivor WHERE modeldb_id=:mid
      ORDER BY dt, ca_sequence ASC""", mid=mid)
    merger = list()
    for r in db.fetchall():
        val = Utilities.Struct()
        val.dt = r[0].date()
        val.ca_sequence = int(r[1])
        val.new_marketdb_id = r[2]
        val.old_marketdb_id = r[3]
        val.share_ratio = float(r[4])
        cash_payment = r[5]
        if cash_payment is not None:
            cash_payment = float(cash_payment)
        val.cash_payment = cash_payment
        currency_id = r[6]
        if currency_id is not None:
            currency_id = int(currency_id)
        val.currency_id = currency_id
        val.ref = r[7]
        val.rmg_id = int(r[8])
        merger.append(val)
    rval.mergerSurvivor = merger
    return rval

def updateMetaData(mid, metaSrc, metaTgt, tgtDb):
    syncTable(mid, [metaSrc.issue], [metaTgt.issue], 'issue_id',
              ('issue_id',), tgtDb, 'issue')
    syncTable(mid, metaSrc.issueMap, metaTgt.issueMap, 'modeldb_id',
              ('from_dt',), tgtDb, 'issue_map')
    syncTable(mid, metaSrc.subIssue, metaTgt.subIssue, 'issue_id',
              ('sub_id',), tgtDb, 'sub_issue')
    syncTable(mid, metaSrc.rmsIssue, metaTgt.rmsIssue, 'issue_id',
              ('rms_id', 'from_dt',), tgtDb, 'rms_issue')
    syncEstuExcl(mid, metaSrc.rmsESTUExcluded, metaTgt.rmsESTUExcluded,
               'issue_id',
               ('rms_id', 'change_dt'), tgtDb)
    syncTable(mid, metaSrc.spinOff, metaTgt.spinOff, 'parent_id',
              ('dt', 'child_id',), tgtDb, 'ca_spin_off')
    syncTable(mid, metaSrc.mergerSurvivor, metaTgt.mergerSurvivor,
              'modeldb_id',
              ('dt', 'ca_sequence',), tgtDb, 'ca_merger_survivor')

def syncTable(mid, srcVals, tgtVals, issueField, keyFields, tgtDb,
              tableName):
    srcDict = buildValueDicts(srcVals, keyFields)
    tgtDict = buildValueDicts(tgtVals, keyFields)
    for dt in (set(tgtDict.keys()) - set(srcDict.keys())):
        # terminate records in tgt that are not in src
        tgtVal = tgtDict[dt]
        query = """DELETE FROM %(tableName)s
          WHERE %(issue)s=:issue AND %(keyList)s""" % {
            'tableName': tableName, 'issue': issueField,
            'keyList': ' AND '.join(['%s = :%s' % (f,f)
                                 for f in keyFields])}
        delDict = { 'issue': mid }
        for f in keyFields:
            delDict[f] = tgtVal.getField(f)
        logging.info('Deleting record in %s: %s', tableName, delDict)
        tgtDb.dbCursor.execute(query, delDict)
        if tgtDb.dbCursor.rowcount != 1:
            raise ValueError('Deleted wrong number (%d) of records' \
                  % tgtDb.dbCursor.rowcount)
    for dt in (set(tgtDict.keys()) & set(srcDict.keys())):
        # Compare and update records both in src and tgt
        srcVal = srcDict[dt]
        tgtVal = tgtDict[dt]
        updateDict = dict()
        for f in srcVal.getFieldNames():
            if srcVal.getField(f) != tgtVal.getField(f):
                updateDict[f] = srcVal.getField(f)
        if len(updateDict) > 0:
            query = """UPDATE %(tableName)s
              SET %(setList)s WHERE %(issue)s=:issue AND %(keyList)s""" % {
                'tableName': tableName,
                'issue': issueField,
                'setList': ','.join(['%s = :%s' % (f,f)
                                     for f in updateDict.keys()
                                     if f not in keyFields]),
                'keyList': ' AND '.join(['%s = :%s' % (f,f)
                                     for f in keyFields])}
            updateDict['issue'] = mid
            for f in keyFields:
                updateDict[f] = tgtVal.getField(f)
            logging.info('Updating record in %s: %s', tableName, updateDict)
            tgtDb.dbCursor.execute(query, updateDict)
            if tgtDb.dbCursor.rowcount != 1:
                raise ValueError('Updated wrong number (%d) of records' \
                      % tgtDb.dbCursor.rowcount)
    for k in (set(srcDict.keys()) - set(tgtDict.keys())):
        # Insert new records into tgt
        srcVal = srcDict[k]
        query = """INSERT INTO %(tableName)s
          (%(issue)s, %(fieldList)s)
          VALUES(:mid, %(fieldValues)s)""" % {
            'tableName': tableName,
            'issue': issueField,
            'fieldList': ', '.join(srcVal.getFieldNames()),
            'fieldValues': ', '.join([':%s' % f for f
                                     in srcVal.getFieldNames()])}
        valDict = dict([(f, srcVal.getField(f))
                        for f in srcVal.getFieldNames()])
        valDict['mid'] = mid
        logging.info('Inserting record into %s: %s', tableName, valDict)
        tgtDb.dbCursor.execute(query, valDict)

def syncEstuExcl(mid, srcVals, tgtVals, issueField, keyFields, tgtDb):
    srcDict = buildValueDicts(srcVals, keyFields)
    tgtDict = buildValueDicts(tgtVals, keyFields)
    for dt in (set(tgtDict.keys()) - set(srcDict.keys())):
        # terminate records in tgt that are not in src
        tgtVal = tgtDict[dt]
        query = """INSERT INTO rms_estu_excluded
           SELECT rms_id, issue_id, change_dt, change_del_flag, %(src)d, '%(ref)s', sysdate, 'Y'
           FROM rms_estu_excl_active WHERE rms_id=:rms_id AND issue_id=:mid AND
           change_dt=:change_dt""" % dict(src=SRC_ID, ref=REF_VAL)
        delDict = { 'mid': mid }
        for f in ['change_dt', 'rms_id']:
            delDict[f] = tgtVal.getField(f)
        logging.info('Deleting record in rms_estu_excluded: %s', delDict)
        tgtDb.dbCursor.execute(query, delDict)
        if tgtDb.dbCursor.rowcount != 1:
            raise ValueError('Deleted wrong number (%d) of records' \
                  % tgtDb.dbCursor.rowcount)
    for dt in (set(tgtDict.keys()) & set(srcDict.keys())):
        # Compare and update records both in src and tgt
        srcVal = srcDict[dt]
        tgtVal = tgtDict[dt]
        update = False
        for f in srcVal.getFieldNames():
            if srcVal.getField(f) != tgtVal.getField(f):
                update = True
        if update:
            valDict = dict()
            valDict['mid'] = mid
            # add required new records
            for f in srcVal.getFieldNames():
                valDict[f] = srcVal.getField(f)
            query = """INSERT INTO rms_estu_excluded
               (rms_id, issue_id, change_dt, change_del_flag, src_id, ref, rev_dt, rev_del_flag)
               VALUES(:rms_id, :mid, :change_dt, :change_del_flag, :src_id, :ref, sysdate, 'N')"""
            logging.info('Updating record in rms_estu_excluded: %s', valDict)
            tgtDb.dbCursor.execute(query, valDict)
    for k in (set(srcDict.keys()) - set(tgtDict.keys())):
        # Insert new records into tgt
        srcVal = srcDict[k]
        query = """INSERT INTO rms_estu_excluded
           (rms_id, issue_id, change_dt, change_del_flag, src_id, ref, rev_dt, rev_del_flag)
           VALUES(:rms_id, :mid, :change_dt, :change_del_flag, :src_id, :ref, sysdate, 'N')"""
        valDict = dict()
        valDict['mid'] = mid
        for f in srcVal.getFieldNames():
            valDict[f] = srcVal.getField(f)
        logging.info('Inserting record into rms_estu_excluded: %s', valDict)
        tgtDb.dbCursor.execute(query, valDict)

def buildValueDicts(valList, keyFields):
    d = dict()
    for v in valList:
        key = tuple([v.getField(f) for f in keyFields])
        d[key] = v
    return d

def addIssue(mid, metaData, tgtDb):
    issue = metaData.issue
    valDict = { 'mid': mid, 'fromDt': issue.from_dt, 'thruDt': issue.thru_dt }
    tgtDb.dbCursor.execute("""INSERT INTO issue
      (issue_id, from_dt, thru_dt)
      VALUES (:mid, :fromDt, :thruDt)""", valDict)
    logging.info('Inserting record into issue: %s', valDict)

def addIssueID(mid, srcDb, tgtDb):
    metaData = getIssueMetaData(mid, srcDb.dbCursor)
    addIssue(mid, metaData, tgtDb)
    metaDataTgt = getIssueMetaData(mid, tgtDb.dbCursor)
    updateMetaData(mid, metaData, metaDataTgt, tgtDb)

def syncMetaData(mid, srcDb, tgtDb):
    metaData = getIssueMetaData(mid, srcDb.dbCursor)
    metaDataTgt = getIssueMetaData(mid, tgtDb.dbCursor)
    #print mid, metaData
    updateMetaData(mid, metaData, metaDataTgt, tgtDb)

if __name__ == '__main__':
    CHUNK_SIZE=100
    usage = "usage: %prog [options]"
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
    cmdlineParser.add_option("--ignore-existing", action="store_false",
                             default=True, dest="syncExisting",
                             help="don't check if Axioma IDs already in both databases differ")
    cmdlineParser.add_option("--ignore-new", action="store_false",
                             default=True, dest="syncNew",
                             help="don't process Axioma IDs missing in target database")
    cmdlineParser.add_option("--range", action="store",
                             default=None, dest="syncRange",
                             help="a:b range of existing Axioma IDs to sync")
    (options_, args_) = cmdlineParser.parse_args()
    if len(args_) != 0:
        cmdlineParser.error("Incorrect number of arguments")
    Utilities.processDefaultCommandLine(options_, cmdlineParser)
    
    if len(options_.excludeList) > 0:
        options_.excludeList = set(options_.excludeList.split(','))
        logging.info('Excluding %d issues from consideration',
                     len(options_.excludeList))
    else:
        options_.excludeList = set()
    
    if len(options_.restrictList) > 0:
        options_.restrictList = set(options_.restrictList.split(','))
        logging.info('Restricting to %d issues',
                     len(options_.restrictList))
    else:
        options_.restrictList = None
    
# For testing
#    glprod = ModelDB.ModelDB(
#        user='modeldb_global', passwd='modeldb_global', sid='research')
#    research = ModelDB.ModelDB(
#        user='modeldb_global', passwd='modeldb_global', sid='XE')
    glprod = ModelDB.ModelDB(
        user='modeldb_global', passwd='modeldb_global', sid='glprodsb')
    research = ModelDB.ModelDB(
        user='modeldb_global', passwd='modeldb_global', sid='research')
    prodIssues = getIssues(glprod)
    researchIssues = getIssues(research)
    logging.info('%d issues on glprod, %d on research',
                 len(prodIssues), len(researchIssues))
    logging.info('%d common issues', len(prodIssues & researchIssues))
    newResearchIssues = researchIssues - prodIssues - options_.excludeList
    if options_.restrictList is not None:
        newResearchIssues &= options_.restrictList
    if len(newResearchIssues) > 0:
        logging.info('%d Issue IDs only on research: %s',
                     len(newResearchIssues), newResearchIssues)
    if options_.syncNew:
        newProdIssues = prodIssues - researchIssues - options_.excludeList
        if options_.restrictList is not None:
            newProdIssues &= options_.restrictList
        logging.info('%d new Issue IDs will be added to research',
                     len(newProdIssues))
        for (count, axid) in enumerate(sorted(newProdIssues)):
            addIssueID(axid, glprod, research)
            if (count + 1) % CHUNK_SIZE == 0:
                logging.info('Processed %d/%d', count+1, len(newProdIssues))
                if not options_.testOnly:
                    logging.info('Committing changes')
                    research.commitChanges()
    
    if options_.syncExisting:
        sameIssues = (prodIssues & researchIssues) - options_.excludeList
        if options_.restrictList is not None:
            sameIssues &= options_.restrictList
        if options_.syncRange is not None:
            (start, end) = options_.syncRange.split(':')
            start = int(start)
            end = int(end)
        else:
            start = 0
            end = len(sameIssues)
        for (count, mid) in enumerate(sorted(sameIssues)):
            if count < start or count >= end:
                continue
            syncMetaData(mid, glprod, research)
            if (count + 1) % CHUNK_SIZE == 0:
                logging.info('Processed %d/%d', count+1, len(sameIssues))
                if not options_.testOnly:
                    logging.info('Committing changes')
                    research.commitChanges()
    
    glprod.revertChanges()
    glprod.finalize()
    if options_.testOnly:
        logging.info('Reverting changes')
        research.revertChanges()
    else:
        logging.info('Committing changes')
        research.commitChanges()
    research.finalize()
