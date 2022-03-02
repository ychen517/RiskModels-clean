
import datetime
import logging
import optparse
import sys
import riskmodels
from marketdb import MarketDB
from riskmodels import ModelDB
from riskmodels import Utilities

if __name__ == '__main__':
    usage = "usage: %prog [options] <sub-factor-id> <YYYY-MM-DD> [<YYYY-MM-DD>]"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addModelAndDefaultCommandLine(cmdlineParser)
    cmdlineParser.add_option("-n", action="store_true",
                             default=False, dest="testOnly",
                             help="don't change the database")
    cmdlineParser.add_option("--set-trans-date", action="store_true",
                             default=False, dest="setTransDate",
                             help="Use date + 1 day 6 hours as trans-date")
    (options, args) = cmdlineParser.parse_args()
    if len(args) < 2 or len(args) > 3:
        cmdlineParser.error("Incorrect number of arguments")
    riskModelClass = Utilities.processModelAndDefaultCommandLine(
        options, cmdlineParser)

    modelDB = ModelDB.ModelDB(sid=options.modelDBSID, user=options.modelDBUser,
                              passwd=options.modelDBPasswd)
    marketDB = MarketDB.MarketDB(
        sid=options.marketDBSID, user=options.marketDBUser,
        passwd=options.marketDBPasswd)
    startDate = Utilities.parseISODate(args[1])
    if len(args) == 1:
        endDate = startDate
    else:
        endDate = Utilities.parseISODate(args[2])
    riskModel = riskModelClass(modelDB, marketDB)
    dates = modelDB.getDateRange(riskModel.rmg, startDate, endDate, True)

    status = 0
    subFactorID = int(args[0])
    prevExposures = None
    prevDate = None

    for d in dates:
        try:
            logging.info('Processing %s' % d)
            if options.setTransDate:
                transDateTime = datetime.datetime.combine(d, datetime.time()) \
                                + datetime.timedelta(days=1, hours=6)
                logging.info('setting trans date to %s' % transDateTime)
                modelDB.setTransDateTime(transDateTime)
                #marketDB.setTransDateTime(transDateTime)
            riskModel.setFactorsForDate(d, modelDB)
            rmi = riskModel.getRiskModelInstance(d, modelDB)
            univ = modelDB.getRiskModelInstanceUniverse(rmi)
            masterSet = set([sid.getSubIDString() for sid in univ])
            query = """SELECT sub_issue_id, value FROM rmi_factor_exposure
                       WHERE rms_id=:rms_arg AND sub_factor_id=:sub_arg
                       AND dt=:dt_arg"""
            modelDB.dbCursor.execute(query, rms_arg=riskModel.rms_id,
                                     dt_arg=d, sub_arg=subFactorID)
            r = modelDB.dbCursor.fetchall()
            if len(r)==0:
                logging.debug('No exposures for sub factor %d on %s' % (subFactorID, d))
                if prevExposures is not None:
                    query = """INSERT INTO rmi_factor_exposure
                                    (rms_id, dt, sub_factor_id, sub_issue_id, value)
                               VALUES (:rms_arg, :dt_arg, :sub_arg, :issue_arg, :val_arg)"""
                    valueDicts = [{'issue_arg': row[0],
                                   'val_arg': row[1],
                                   'sub_arg': subFactorID,
                                   'rms_arg': riskModel.rms_id,
                                   'dt_arg': d} for row in prevExposures if row[0] in masterSet]
                    if len(valueDicts) > 0:
                        modelDB.dbCursor.executemany(query, valueDicts)
                        logging.debug('Rolled over %d values from %s for %s' % (len(valueDicts), prevDate, d))
            else:
                logging.debug('Found %d exposures for sub factor %d on %s' % (len(r), subFactorID, d))
                prevExposures = r
                prevDate = d

            if options.testOnly:
                logging.info('Reverting changes')
                modelDB.revertChanges()
            else:
                modelDB.commitChanges()
            logging.info('Finished processing exposures %s' % d)
        except Exception as ex:
            logging.error('Exception caught during processing', exc_info=True)
            modelDB.revertChanges()
            status = 1
            break

    marketDB.finalize()
    modelDB.finalize()
    sys.exit(status)
