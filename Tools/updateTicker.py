
import logging
import optparse
import datetime
import sys
from marketdb import MarketDB
from riskmodels import ModelDB
from riskmodels import Utilities

def getCurrentTickers(marketDB, axiomaID, fromDt, thruDt):
    """Return the current ticker history in chronological order.
    """
    marketDB.dbCursor.execute("""SELECT id, from_dt, thru_dt, src_id, ref
    FROM asset_dim_ticker WHERE axioma_id=:id_arg
    AND trans_thru_dt > :trans_arg AND trans_from_dt <= :trans_arg
    ORDER BY from_dt""", id_arg=axiomaID,
                              trans_arg=marketDB.transDateTimeStr)
    r = marketDB.dbCursor.fetchall()
    tickerHistory = list()
    for (ticker, fromDt, thruDt, src, ref) in r:
        entry = Utilities.Struct()
        entry.axiomaID =  axiomaID
        entry.id = ticker
        entry.fromDt = marketDB.oracleToDate(fromDt)
        entry.thruDt = marketDB.oracleToDate(thruDt)
        entry.src = src
        entry.ref = ref
        tickerHistory.append(entry)
    return tickerHistory

def report(prefix, axiomaID, tickerHistory):
    """Print a report from the result of getCurrentTickers.
    """
    print('%s Ticker history for %s' % (prefix, axiomaID))
    print('Axioma ID|From|Thru|Source|Ref|')
    for entry in tickerHistory:
        print('%s|%s|%s|%s|%s' % (
            entry.id, entry.fromDt, entry.thruDt, entry.src, entry.ref))

def updateTicker(marketDB, axiomaID, fromDt, thruDt, ticker, tickerHistory):
    """Change the ticker history for axiomaID to the given ticker
    in the given interval.
    """
    # ticker report has inclusive ranges, convert to database convention
    thruDt = thruDt + datetime.timedelta(days=1)
    if (fromDt >= thruDt):
        logging.error('date interval [%s,%s) is empty. Skipping.'
                      % (fromDt, thruDt))
        return
    # check that new ticker won't create a conflict
    marketDB.dbCursor.execute("""SELECT axioma_id, from_dt, thru_dt
    FROM asset_dim_ticker
    WHERE axioma_id<>:aid_arg AND id=:id_arg
    AND from_dt<:thru_arg AND thru_dt>:from_arg""",
                              aid_arg=axiomaID, id_arg=ticker,
                              from_arg=str(fromDt), thru_arg=str(thruDt))
    conflicts = [(i[0], marketDB.oracleToDate(i[1]),
                  marketDB.oracleToDate(i[2]))
                  for i in marketDB.dbCursor.fetchall()]
    if len(conflicts) > 0:
        return conflicts

    newHistory = list()
    oldHistory = list(tickerHistory)
    newEntry = Utilities.Struct()
    newEntry.axiomaID =  axiomaID
    newEntry.id = ticker
    newEntry.fromDt = fromDt
    newEntry.thruDt = thruDt
    newEntry.src = 900
    newEntry.ref = 'Manual ticker insert'
    twoDays = datetime.timedelta(days=2)
    while len(oldHistory) > 0:
        e = oldHistory.pop(0)
        if e.id == ticker and e.thruDt >= newEntry.fromDt - twoDays\
           and e.fromDt <= newEntry.thruDt + twoDays:
            # Same ticker, overlapping or adjacent intervals: merge
            newEntry.fromDt = min(newEntry.fromDt, e.fromDt)
            newEntry.thruDt = max(newEntry.thruDt, e.thruDt)
            newEntry.src = e.src
            newEntry.ref = e.ref
        elif e.thruDt < newEntry.fromDt or e.fromDt > newEntry.thruDt:
            newHistory.append(e)
        elif e.thruDt > newEntry.thruDt and e.fromDt < newEntry.fromDt:
            # Old interval contains new interval, split it in two
            e2 = Utilities.Struct()
            e2.axiomaID = e.axiomaID
            e2.id = e.id
            e2.fromDt = e.fromDt
            e2.thruDt = newEntry.fromDt
            e2.src = e.src
            e2.ref = e.ref
            e.fromDt = newEntry.thruDt
            newHistory.append(e2)
            newHistory.append(e)
            newHistory.extend(oldHistory)
            oldHistory = list()
        elif e.thruDt <= newEntry.thruDt and e.fromDt >= newEntry.fromDt:
            # Completely contained in the new interval
            pass
        elif e.fromDt < newEntry.fromDt:
            e.thruDt = newEntry.fromDt
            newHistory.append(e)
        else:
            e.fromDt = newEntry.thruDt
            newHistory.append(e)
    newHistory.append(newEntry)

    # Replace ticker history with the new one
    marketDB.dbCursor.execute("""DELETE FROM asset_dim_ticker
    WHERE axioma_id=:id_arg""", id_arg=axiomaID)
    marketDB.dbCursor.executemany("""INSERT INTO asset_dim_ticker
    (axioma_id, id, from_dt, thru_dt, src_id, ref, trans_from_dt,
    trans_thru_dt)
    VALUES(:aid_arg, :id_arg, :from_arg, :thru_arg, :src_arg, :ref_arg,
    :tfrom_arg, :tthru_arg)""", [{
        'aid_arg': e.axiomaID, 'id_arg': e.id, 'from_arg': str(e.fromDt),
        'thru_arg': str(e.thruDt), 'src_arg': e.src, 'ref_arg': e.ref,
        'tfrom_arg': '1950-01-01', 'tthru_arg': str(marketDB.futureDate)}
                                 for e in newHistory])
    return list()

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

    prevAxiomaId_ = None
    for line in sys.stdin:
        fields = line.split('|')
        (axiomaId_, fromDt_, thruDt_) = fields[0:3]
        if axiomaId_ is None or len(axiomaId_.strip()) == 0:
            axiomaId_ = prevAxiomaId_
        prevAxiomaId_ = axiomaId_
        ticker_ = fields[4]
        fromDt_ = Utilities.parseISODate(fromDt_)
        if len(thruDt_.strip()) > 0:
            thruDt_ = Utilities.parseISODate(thruDt_)
        else:
            thruDt_ = fromDt_

        history_ = getCurrentTickers(marketDB_, axiomaId_, fromDt_, thruDt_)
        report("\n--------------------\nCurrent", axiomaId_, history_)
        conflicts = updateTicker(marketDB_, axiomaId_, fromDt_, thruDt_,
                                 ticker_, history_)
        print('')
        if len(conflicts) > 0:
            print('Change would create conflict with:')
            for (cid_, cfrom_, cthru_) in conflicts:
                print('%s|%s|%s' % (cid_, cfrom_, cthru_))
        else:
            history_ = getCurrentTickers(marketDB_, axiomaId_,
                                         fromDt_, thruDt_)
            report("New", axiomaId_, history_)
        
    if options_.testOnly:
        logging.info('Reverting changes')
        marketDB_.revertChanges()
    else:
        marketDB_.commitChanges()
    
    marketDB_.finalize()
