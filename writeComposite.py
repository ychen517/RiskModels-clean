
import datetime
import logging
import optparse
import os
from marketdb import MarketDB
from riskmodels import ModelDB
from riskmodels import Utilities

def writeETFCSV(ETFcomp, outFile):
    """Write a CSV ETF Composite file to outFile.
    """
    for (asset, weight) in ETFcomp:
        outFile.write("%s, %.8g\n" % (asset.getPublicID(), weight))


if __name__ == '__main__':
    usage = "usage: %prog [options] <ETF name> <YYYY-MM-DD> <YYYY-MM-DD>" 
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addDefaultCommandLine(cmdlineParser)
    cmdlineParser.add_option("-d", "--directory", action="store",
                             default='.', dest="targetDir",
                             help="directory for output files")
    cmdlineParser.add_option("-p", "--prefix", action="store",
                             default="", dest="prefix",
                             help="Prefix to file name")
    cmdlineParser.add_option("-s", "--suffix", action="store",
                             default="", dest="suffix",
                             help="suffix to file name")
    cmdlineParser.add_option("-o", "--outname", action="store",
                             default=None, dest="outname",
                             help="Name of output files, defaults to ETF name")
    cmdlineParser.add_option("--num-days-forward", action="store",
                             default=5, dest="numDaysForward",
                             help="Next day (day > specified day) ETF")
    cmdlineParser.add_option("-m", "--monthend", action="store_true",
                             default=False, dest="monthend",
                             help="The frequency of writing, defaults to daily")
    cmdlineParser.add_option("--target-sub-dirs", action="store_true",
                                 default=False, dest="appendDateDirs",
                                 help="Append yyyy/mm to end of output directory path")
    cmdlineParser.add_option("--histbeta-new", action="store_true",
                             default=False, dest="histBetaNew",
                             help="process historic beta new way or legacy way")

    (options, args) = cmdlineParser.parse_args()
    if len(args) < 2 or len(args) > 3:
        cmdlineParser.error("Incorrect number of arguments")

    Utilities.processDefaultCommandLine(options, cmdlineParser)
    modelDB = ModelDB.ModelDB(sid=options.modelDBSID, user=options.modelDBUser,
                              passwd=options.modelDBPasswd)
    marketDB = MarketDB.MarketDB(sid=options.marketDBSID, user=options.marketDBUser, passwd=options.marketDBPasswd)
    ETFName = args[0]

    startDate = Utilities.parseISODate(args[1])
    if len(args) == 2:
        endDate = startDate
    else:
        endDate = Utilities.parseISODate(args[2])
    d = startDate
    dates = []
    dayInc = datetime.timedelta(1)

    while d <= endDate:
        if d.isoweekday() in [6,7]:
           d += dayInc
           continue
        else:
           dates.append(d)
           d += dayInc

    if options.monthend:
        dates = [prev for (prev, next) in zip(dates[:-1], dates[1:]) \
                if (next.month > prev.month or next.year > prev.year)]

    for d in dates:
        logging.info('Processing %s' % str(d))
        ETFNames=[ETFName]
        target=options.targetDir
        if options.appendDateDirs:
             target = os.path.join(target, '%04d' % d.year, '%02d' % d.month)
             try:
                os.makedirs(target)
             except OSError as e:
                if e.errno != 17:
                   raise
                else:
                   pass
        
        for ETFName in ETFNames:
            if len(ETFNames) == 1 and options.outname:
                outFileName = '%s/%s-%d%02d%02d.csv' % (
                    target, options.outname, d.year, d.month, d.day)
            else:
                outFileName='%s/%s%s-%02d%02d%02d%s.csv' % (target, options.prefix,ETFName,d.year,d.month,d.day, options.suffix)
            logging.info("Processing %s", ETFName)
            (day,etfComp )= modelDB.getCompositeConstituents(ETFName,d,marketDB)
            if len(etfComp) > 0:
                if not os.path.exists(os.path.dirname(outFileName)):
                    os.makedirs(os.path.dirname(outFileName))
                outFile = open(outFileName, 'w')
                writeETFCSV(etfComp, outFile)
                outFile.close()
            else:
                logging.info('No data for %s on %s, skipping', ETFName, str(d))
        d += dayInc
    marketDB.finalize()
    modelDB.finalize()
