
from marketdb import MarketDB
import logging
import optparse
from riskmodels import ModelDB
from riskmodels import Utilities
import datetime
import sys

def getAssetPriceRange(marketDB):
    """Return list of (asset, first price date, last price date) tuples
    for all assets.
    """
    marketDB.dbCursor.execute("""SELECT axioma_id, min(dt), max(dt)
    FROM asset_dim_ucp GROUP BY axioma_id""")
    r = [(axiomaID, marketDB.oracleToDate(minDt), marketDB.oracleToDate(maxDt))
         for (axiomaID, minDt, maxDt) in marketDB.dbCursor.fetchall()]
    return r

def filterAssets(marketDB, modelDB, priceRange, cutoff):
    """Return a list of assets where their asset_ref.thru_dt
    is more than cutoff days after the last price.
    """
    now = datetime.date.today()
    oneDay = datetime.timedelta(days=1)
    priceRange = [(axiomaID, firstPrice, lastPrice)
                  for (axiomaID, firstPrice, lastPrice) in priceRange
                  if lastPrice + cutoff < now]
    candidates = list()
    counter = 0
    for (axiomaID, firstPrice, lastPrice) in priceRange:
        counter += 1
        if counter % 10 == 0:
            sys.stderr.write('%d/%d\n' % (counter, len(priceRange)))
        marketDB.dbCursor.execute("""SELECT thru_dt FROM
        asset_ref WHERE axioma_id=:aid_arg""", aid_arg=axiomaID)
        r = marketDB.dbCursor.fetchall()
        if len(r) != 1:
            logging.error('No asset_ref for %s: %s' % (axiomaID, r))
            continue
        thruDt = marketDB.oracleToDate(r[0][0])
        if thruDt <= lastPrice + oneDay:
            continue
        marketDB.dbCursor.execute("""SELECT irev.index_id,
        (SELECT name FROM index_member imem WHERE imem.id=irev.index_id),
        max(irev.dt)
        FROM index_constituent icon JOIN index_revision irev
        ON icon.revision_id=irev.id
        WHERE icon.axioma_id=:aid_arg
        GROUP BY irev.index_id""", aid_arg=axiomaID)
        benchDates = [(indexID, name, marketDB.oracleToDate(dt))
                      for (indexID, name, dt)
                      in marketDB.dbCursor.fetchall()]
        # Find convergence records in marketDB and modelDB
        marketDB.dbCursor.execute("""SELECT convergence_id, dt
        FROM asset_mod_convergence WHERE parent_axioma_id=:aid_arg""",
                                  aid_arg=axiomaID)
        convDates = [(convID, marketDB.oracleToDate(dt))
                     for (convID, dt) in marketDB.dbCursor.fetchall()]
        modelDB.dbCursor.execute("""SELECT modeldb_id, dt
        FROM ca_merger_survivor WHERE old_marketdb_id=:aid_arg""",
                                 aid_arg=axiomaID)
        convDates.extend([(mdlID, modelDB.oracleToDate(dt))
                          for (mdlID, dt) in modelDB.dbCursor.fetchall()])
        
        candidates.append((axiomaID, lastPrice, thruDt, benchDates, convDates))
    return candidates

def updateIDTable(marketDB, axiomaID, newThruDt, table):
    marketDB.dbCursor.execute("""SELECT id, from_dt, thru_dt
    FROM asset_dim_%s WHERE axioma_id=:aid_arg
    AND trans_from_dt <= :trans_arg AND trans_thru_dt > :trans_arg
    ORDER BY from_dt""" % table,
                              aid_arg=axiomaID,
                              trans_arg=marketDB.transDateTimeStr)
    r = marketDB.dbCursor.fetchall()
    logging.info('Current %s history' % table)
    for (val, fromDt, thruDt) in r:
        logging.info('%s|%s|%s' % (val, marketDB.oracleToDate(fromDt),
                                   marketDB.oracleToDate(thruDt)))
    marketDB.dbCursor.execute("""DELETE FROM asset_dim_%s
    WHERE axioma_id=:aid_arg AND from_dt>=:thru_arg
    AND trans_from_dt <= :trans_arg AND trans_thru_dt > :trans_arg""" % table,
                              aid_arg=axiomaID,
                              thru_arg=str(newThruDt),
                              trans_arg=marketDB.transDateTimeStr)
    marketDB.dbCursor.execute("""UPDATE asset_dim_%s
    SET thru_dt=:thru_arg WHERE axioma_id=:aid_arg
    AND thru_dt>:thru_arg
    AND trans_from_dt <= :trans_arg AND trans_thru_dt > :trans_arg""" % table,
                              aid_arg=axiomaID,
                              thru_arg=str(newThruDt),
                              trans_arg=marketDB.transDateTimeStr)
    marketDB.dbCursor.execute("""SELECT id, from_dt, thru_dt
    FROM asset_dim_%s WHERE axioma_id=:aid_arg
    AND trans_from_dt <= :trans_arg AND trans_thru_dt > :trans_arg
    ORDER BY from_dt""" % table,
                              aid_arg=axiomaID,
                              trans_arg=marketDB.transDateTimeStr)
    r = marketDB.dbCursor.fetchall()
    logging.info('New %s history' % table)
    for (val, fromDt, thruDt) in r:
        logging.info('%s|%s|%s' % (val, marketDB.oracleToDate(fromDt),
                                   marketDB.oracleToDate(thruDt)))
    
def updateIssueTable(modelDB, issueID, newThruDt, table):
    modelDB.dbCursor.execute("""SELECT from_dt, thru_dt
    FROM %s WHERE issue_id=:mid_arg ORDER BY from_dt""" % table,
                              mid_arg=issueID)
    r = modelDB.dbCursor.fetchall()
    logging.info('Current %s history' % table)
    for (fromDt, thruDt) in r:
        logging.info('%s|%s' % (modelDB.oracleToDate(fromDt),
                                modelDB.oracleToDate(thruDt)))
    modelDB.dbCursor.execute("""DELETE FROM %s
    WHERE issue_id=:mid_arg AND from_dt>=:thru_arg""" % table,
                              mid_arg=issueID,
                              thru_arg=str(newThruDt))
    modelDB.dbCursor.execute("""UPDATE %s
    SET thru_dt=:thru_arg WHERE issue_id=:mid_arg
    AND thru_dt>:thru_arg""" % table,
                             mid_arg=issueID,
                             thru_arg=str(newThruDt))
    modelDB.dbCursor.execute("""SELECT from_dt, thru_dt
    FROM %s WHERE issue_id=:mid_arg ORDER BY from_dt""" % table,
                              mid_arg=issueID)
    r = modelDB.dbCursor.fetchall()
    logging.info('New %s history' % table)
    for (fromDt, thruDt) in r:
        logging.info('%s|%s' % (modelDB.oracleToDate(fromDt),
                                modelDB.oracleToDate(thruDt)))
    
def updateIssueMapTable(modelDB, issueID, newThruDt):
    modelDB.dbCursor.execute("""SELECT marketdb_id, from_dt, thru_dt
    FROM issue_map WHERE modeldb_id=:mid_arg ORDER BY from_dt""",
                              mid_arg=issueID)
    r = modelDB.dbCursor.fetchall()
    logging.info('Current issue_map history')
    for (val, fromDt, thruDt) in r:
        logging.info('%s|%s|%s' % (val, modelDB.oracleToDate(fromDt),
                                   modelDB.oracleToDate(thruDt)))
    modelDB.dbCursor.execute("""DELETE FROM issue_map
    WHERE modeldb_id=:mid_arg AND from_dt>=:thru_arg""",
                              mid_arg=issueID,
                              thru_arg=str(newThruDt))
    modelDB.dbCursor.execute("""UPDATE issue_map
    SET thru_dt=:thru_arg WHERE modeldb_id=:mid_arg
    AND thru_dt>:thru_arg""", mid_arg=issueID,
                             thru_arg=str(newThruDt))
    modelDB.dbCursor.execute("""SELECT marketdb_id, from_dt, thru_dt
    FROM issue_map WHERE modeldb_id=:mid_arg ORDER BY from_dt""",
                              mid_arg=issueID)
    r = modelDB.dbCursor.fetchall()
    logging.info('New issue_map history')
    for (val, fromDt, thruDt) in r:
        logging.info('%s|%s|%s' % (val, modelDB.oracleToDate(fromDt),
                                   modelDB.oracleToDate(thruDt)))
    
def updateRefTable(marketDB, axiomaID, newThruDt):
    marketDB.dbCursor.execute("""SELECT from_dt, thru_dt
    FROM asset_ref WHERE axioma_id=:aid_arg""",
                              aid_arg=axiomaID)
    r = marketDB.dbCursor.fetchall()
    logging.info('Current asset_ref')
    for (fromDt, thruDt) in r:
        logging.info('%s|%s' % (marketDB.oracleToDate(fromDt),
                                marketDB.oracleToDate(thruDt)))
    marketDB.dbCursor.execute("""UPDATE asset_ref
    SET thru_dt=:thru_arg WHERE axioma_id=:aid_arg
    AND thru_dt>:thru_arg""", aid_arg=axiomaID,
                              thru_arg=str(newThruDt))
    marketDB.dbCursor.execute("""SELECT from_dt, thru_dt
    FROM asset_ref WHERE axioma_id=:aid_arg""",
                              aid_arg=axiomaID)
    r = marketDB.dbCursor.fetchall()
    logging.info('New asset_ref')
    for (fromDt, thruDt) in r:
        logging.info('%s|%s' % (marketDB.oracleToDate(fromDt),
                                marketDB.oracleToDate(thruDt)))
    
def updateValueTable(marketDB, axiomaID, newThruDt, table):
    marketDB.dbCursor.execute("""SELECT value, from_dt, thru_dt
    FROM asset_dim_%s WHERE axioma_id=:aid_arg
    AND trans_from_dt <= :trans_arg AND trans_thru_dt > :trans_arg
    ORDER BY from_dt""" % table,
                              aid_arg=axiomaID,
                              trans_arg=marketDB.transDateTimeStr)
    r = marketDB.dbCursor.fetchall()
    logging.info('Current %s history' % table)
    for (val, fromDt, thruDt) in r:
        logging.info('%s|%s|%s' % (val, marketDB.oracleToDate(fromDt),
                                   marketDB.oracleToDate(thruDt)))
    marketDB.dbCursor.execute("""DELETE FROM asset_dim_%s
    WHERE axioma_id=:aid_arg AND from_dt>=:thru_arg
    AND trans_from_dt <= :trans_arg AND trans_thru_dt > :trans_arg""" % table,
                              aid_arg=axiomaID,
                              thru_arg=str(newThruDt),
                              trans_arg=marketDB.transDateTimeStr)
    marketDB.dbCursor.execute("""UPDATE asset_dim_%s
    SET thru_dt=:thru_arg WHERE axioma_id=:aid_arg
    AND thru_dt>:thru_arg
    AND trans_from_dt <= :trans_arg AND trans_thru_dt > :trans_arg""" % table,
                              aid_arg=axiomaID,
                              thru_arg=str(newThruDt),
                              trans_arg=marketDB.transDateTimeStr)
    marketDB.dbCursor.execute("""SELECT value, from_dt, thru_dt
    FROM asset_dim_%s WHERE axioma_id=:aid_arg
    AND trans_from_dt <= :trans_arg AND trans_thru_dt > :trans_arg
    ORDER BY from_dt""" % table,
                              aid_arg=axiomaID,
                              trans_arg=marketDB.transDateTimeStr)
    r = marketDB.dbCursor.fetchall()
    logging.info('New %s history' % table)
    for (val, fromDt, thruDt) in r:
        logging.info('%s|%s|%s' % (val, marketDB.oracleToDate(fromDt),
                                   marketDB.oracleToDate(thruDt)))
    
def updateThruDate(marketDB, modelDB, axiomaID, newThruDt):
    """Update the thru_dt in asset_ref, asset_dim_ticker, _cusip, _isin,
    _sedol, _issuer, and _tso.
    """
    logging.info('%s, %s' % (axiomaID, newThruDt))
    updateRefTable(marketDB, axiomaID, newThruDt)
    updateValueTable(marketDB, axiomaID, newThruDt, 'tso')
    for table in ['ticker', 'cusip', 'sedol', 'isin', 'issuer']:
        updateIDTable(marketDB, axiomaID, newThruDt, table)
    # find corresponding modeldb issue
    modelDB.dbCursor.execute("""SELECT modeldb_id, from_dt, thru_dt
    FROM issue_map WHERE marketdb_id=:aid_arg and thru_dt > :thru_arg""",
                             aid_arg=axiomaID, thru_arg=str(newThruDt))
    r = modelDB.dbCursor.fetchall()
    if len(r) > 1:
        logging.error('More than one ModelDB match for %s after %s.'
                      ' Skipping update' % (axiomaID, newThruDt))
    elif len(r) == 1:
        issueID = r[0][0]
        curFromDt = modelDB.oracleToDate(r[0][1])
        curThruDt = modelDB.oracleToDate(r[0][2])
        logging.info('ModelDB ID for %s is %s,%s:%s'
                     % (axiomaID, issueID, curFromDt, curThruDt))
        if curFromDt >= newThruDt:
            logging.error('ModelDB ID starts after new thru_dt and should'
                          ' be removed')
            return
        # Check that issue ID is not used after thru_dt
        modelDB.dbCursor.execute("""SELECT marketdb_id, from_dt, thru_dt
        FROM issue_map WHERE modeldb_id=:mdl_arg AND thru_dt > :thru_arg""",
                                 mdl_arg=issueID,
                                 thru_arg=str(curThruDt))
        r = modelDB.dbCursor.fetchall()
        if len(r) > 0:
            logging.error('ModelDB ID %s is in use after %s: %s' % (
                issueID, curThruDt, ';'.join(['%s,%s,%s' % i for i in r])))
        else:
            updateIssueTable(modelDB, issueID, newThruDt, 'issue')
            updateIssueMapTable(modelDB, issueID, newThruDt)
            updateIssueTable(modelDB, issueID, newThruDt, 'sub_issue')
            updateIssueTable(modelDB, issueID, newThruDt, 'rms_issue')
            pass
    else:
        logging.warning('No matching ModelDB ID for %s' % axiomaID)
        
def updateThruDates(marketDB, modelDB, candidates):
    """Update the thru_dt for all assets to the last price or last
    benchmark membership plus 1 whichever is larger.
    """
    oneDay = datetime.timedelta(days=1)
    for (axiomaID, lastPrice, thruDt, benchmarks, convergences) in candidates:
        dates = [lastPrice] + [dt for (indexID, name, dt) in benchmarks] \
                + [dt - oneDay for (convID, dt) in convergences]
        updateThruDate(marketDB, modelDB, axiomaID, max(dates) + oneDay)
    
def reportCandidates(candidates):
    good = list()
    bad = list()
    for (axiomaID, lastPrice, thruDt, benchmarks) in candidates:
        benchmarksAfterLastPrice = [(name, dt) for (indexID, name, dt)
                                    in benchmarks if dt > lastPrice]
        if len(benchmarksAfterLastPrice) > 0:
            bad.append((axiomaID, lastPrice, thruDt, benchmarks))
        else:
            good.append((axiomaID, lastPrice, thruDt, benchmarks))
    logging.info('Assets in benchmark after last known price')
    logging.info('AxiomaID|last price|thru date|benchmarks')
    for (axiomaID, lastPrice, thruDt, benchmarks) in bad:
        logging.info('%s|%s|%s|%s' % (axiomaID, lastPrice, thruDt,';'.join(
            ['%s,%s' % (name, dt) for (indexID, name, dt) in benchmarks])))

    logging.info('\nAssets not in benchmarks after last known price')
    logging.info('AxiomaID|last price|thru date|benchmarks')
    for (axiomaID, lastPrice, thruDt, benchmarks) in good:
        logging.info('%s|%s|%s|%s' % (axiomaID, lastPrice, thruDt,';'.join(
            ['%s,%s' % (name, dt) for (indexID, name, dt) in benchmarks])))

if __name__ == '__main__':
    usage = "usage: %prog [options]"
    cmdlineParser = optparse.OptionParser(usage=usage)
    cmdlineParser.add_option("-n", action="store_true",
                             default=False, dest="testOnly",
                             help="don't change the database")
    Utilities.addDefaultCommandLine(cmdlineParser)
    (options_, args_) = cmdlineParser.parse_args()
    if len(args_) != 0:
        cmdlineParser.error("Incorrect number of arguments")
    Utilities.processDefaultCommandLine(options_, cmdlineParser)
    
    marketDB_ = MarketDB.MarketDB(sid=options_.marketDBSID,
                                  user=options_.marketDBUser,
                                  passwd=options_.marketDBPasswd)
    modelDB_ = ModelDB.ModelDB(sid=options_.modelDBSID,
                                  user=options_.modelDBUser,
                                  passwd=options_.modelDBPasswd)

    priceRange = getAssetPriceRange(marketDB_)
    candidateAssets = filterAssets(marketDB_, modelDB_, priceRange,
                                   datetime.timedelta(days=6*30))
    #reportCandidates(candidateAssets)
    updateThruDates(marketDB_, modelDB_, candidateAssets)
    if options_.testOnly:
        logging.info('Reverting changes')
        marketDB_.revertChanges()
        modelDB_.revertChanges()
    else:
        logging.info('Committing changes')
        marketDB_.commitChanges()
        modelDB_.commitChanges()

    marketDB_.finalize()
    modelDB_.finalize()
