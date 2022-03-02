# Create period returns based on the cumulative returns table.
# The input is a file containing a list of dates in ISO format, one per line.
# The script creates files called return-d1-d2.csv for each consecutive
# pair in the input file.
#

import numpy
import numpy.ma as ma
import optparse
import logging
from marketdb import MarketDB
from riskmodels import ModelDB
from riskmodels import Utilities

if __name__ == '__main__':
    usage = "usage: %prog [options] <period-file>"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addModelAndDefaultCommandLine(cmdlineParser)
    cmdlineParser.add_option("-d", "--directory", action="store",
                             default='.', dest="targetDir",
                             help="directory for output files")
    cmdlineParser.add_option("--use-numeraire", action="store_true",
                             default=False, dest="useNumeraire",
                             help="compute returns in base currency")
    (options, args) = cmdlineParser.parse_args()
    if len(args) != 1:
        cmdlineParser.error("Incorrect number of arguments")
    modelClass = Utilities.processModelAndDefaultCommandLine(
                                            options, cmdlineParser)
    modelDB = ModelDB.ModelDB(sid=options.modelDBSID, 
                              user=options.modelDBUser,
                              passwd=options.modelDBPasswd)
    marketDB = MarketDB.MarketDB(sid=options.marketDBSID, 
                                 user=options.marketDBUser, 
                                 passwd=options.marketDBPasswd)
    riskModel = modelClass(modelDB, marketDB)
    if options.useNumeraire:
        modelDB.createCurrencyCache(marketDB)
    periodFile = open(args[0], 'r')

    startDate = None
    for l in periodFile:
        try:
            nextDate = Utilities.parseISODate(l.strip())
        except:
            continue
        logging.info('%s', nextDate)
        riskModel.setFactorsForDate(nextDate, modelDB)
        modelDB.dbCursor.execute("""
            SELECT r.sub_issue_id, r.value 
                FROM sub_issue_cum_return_active r, rmi_universe u,
                     sub_issue s
                WHERE r.dt=:date_arg AND u.dt = r.dt
                AND u.sub_issue_id = r.sub_issue_id
                AND u.sub_issue_id = s.sub_id
                AND s.from_dt <= :date_arg AND s.thru_dt > :date_arg
                AND u.rms_id = :rms_arg""", 
            date_arg=str(nextDate), rms_arg=riskModel.rms_id)
        nextCumReturns = dict(modelDB.dbCursor.fetchall())
        if startDate != None:
            baseCurrencyID = riskModel.numeraire.currency_id
            # If computing period returns in model numeraire...
            if options.useNumeraire:
                # First determine currency ID of each asset
                rmi = modelDB.getRiskModelInstance(riskModel.rms_id, startDate)
                if rmi is None:
                    startDate = nextDate
                    startCumReturns = nextCumReturns
                    continue
                assets = modelDB.getRiskModelInstanceUniverse(rmi)
                currencies = modelDB.loadSubIssueData(
                        [startDate], assets, 'sub_issue_data', 'currency_id',
                        cache=None, withCurrency=False)
                currencies = currencies.data[:,0]
                subIssueCurrencyMap = dict(zip(assets, currencies))

                # Fetch daily currency returns for period
                currencyIds = numpy.unique(numpy.array(
                                    currencies.filled(baseCurrencyID), int))
                daysBack = len(modelDB.getDateRange(
                                riskModel.rmg, startDate, nextDate)) - 2
                currencyReturns = modelDB.loadCurrencyReturnsHistory(
                        riskModel.rmg, nextDate, daysBack, 
                        currencyIds, baseCurrencyID, idlookup=False)
                currencyIdxMap = dict(zip(currencyIds, range(len(currencyIds))))

                # Approximate currency period return
                periodCurrencyReturns = ma.product(
                        currencyReturns.data.filled(0.0) + 1.0, axis=1) - 1.0
            outFile = open('%s/return-%04d%02d%02d_%04d%02d%02d.csv' 
                    % (options.targetDir, startDate.year, startDate.month,
                    startDate.day, nextDate.year, nextDate.month, nextDate.day), 'w')
            for (a, v) in nextCumReturns.items():
                if a in startCumReturns:
                    subIssue = ModelDB.SubIssue(a)
                    if startCumReturns[a] <= 0.0:
                        periodReturn = 0.0
                    else:
                        periodReturn = v / startCumReturns[a] - 1.0
                    if options.useNumeraire:
                        cid = subIssueCurrencyMap.get(subIssue, baseCurrencyID)
                        if cid is not ma.masked and cid != baseCurrencyID:
                            periodReturn = (1.0 + periodReturn)*(1.0 + \
                                    periodCurrencyReturns[currencyIdxMap[cid]]) - 1.0
                    outFile.write("%s,%.8g\n" % (
                            subIssue.getModelID().getPublicID(), periodReturn))
            outFile.close()
        startDate = nextDate
        startCumReturns = nextCumReturns

    modelDB.finalize()
