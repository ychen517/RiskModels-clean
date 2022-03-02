
import datetime
import logging
import optparse

from marketdb import MarketDB
from riskmodels import ModelDB
from riskmodels import Utilities
from riskmodels import writeFlatFiles

def buildDateList(args):
    if len(args) == 1:
        dateRanges = [i.strip() for i in args[0].split(',')]
        dates = set()
        for dRange in dateRanges:
            if dRange.find(':') == -1:
                dates.add(Utilities.parseISODate(dRange))
            else:
                (startDate, endDate) = [i.strip() for i in dRange.split(':')]
                startDate = Utilities.parseISODate(startDate)
                endDate = Utilities.parseISODate(endDate)
                dates.update([startDate + datetime.timedelta(i)
                              for i in range((endDate-startDate).days + 1)])
    else:
        startDate = Utilities.parseISODate(args[0])
        endDate = Utilities.parseISODate(args[1])
        dates = [startDate + datetime.timedelta(i)
                 for i in range((endDate-startDate).days + 1)]
    dates = sorted(dates,reverse=True)
    return dates

def main():
    usage = "usage: %prog [options] <startdate or datelist> <end-date>"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addDefaultCommandLine(cmdlineParser)
    cmdlineParser.add_option("-d", "--directory", action="store",
                             default='.', dest="targetDir",
                             help="directory for output files")
    cmdlineParser.add_option("--target-sub-dirs", action="store_true",
                             default=False, dest="appendDateDirs",
                             help="Append yyyy/mm to end of output directory path")
    cmdlineParser.add_option("--file-format-version", action="store", type="float",
                             default=3.2, dest="fileFormatVersion",
                             help="version of flat file format to create")
    (options, args) = cmdlineParser.parse_args()
    if len(args) < 1 or len(args) > 2:
        cmdlineParser.error("Incorrect number of arguments")
    
    Utilities.processDefaultCommandLine(options, cmdlineParser)
    modelDB = ModelDB.ModelDB(sid=options.modelDBSID, user=options.modelDBUser,
                              passwd=options.modelDBPasswd)
    marketDB = MarketDB.MarketDB(sid=options.marketDBSID, user=options.marketDBUser, passwd=options.marketDBPasswd)
    dates = buildDateList(args)
    fileFormat = writeFlatFiles.FlatFilesV3()
    options.newRiskFields = True
    
    for d in dates:
        if d.isoweekday() in (6,7):
           logging.info('Ignoring weekend date %s', str(d))
           continue

        logging.info('Processing %s', d)
        fileFormat.writeCountryInfoDay(options, d, modelDB, marketDB)
        logging.info("Done writing %s", d)
    modelDB.finalize()
    marketDB.finalize()

if __name__ == '__main__':
    main()
