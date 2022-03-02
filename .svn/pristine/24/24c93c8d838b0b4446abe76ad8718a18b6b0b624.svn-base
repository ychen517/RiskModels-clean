
import datetime
import logging
import optparse
import os

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
    cmdlineParser.add_option("-c", "--composite-families", action="store",
                             default='', dest="compositeFamilies",
                             help="comma-separated list of composite families")
    cmdlineParser.add_option("-d", "--directory", action="store",
                             default='.', dest="targetDir",
                             help="directory for output files")
    cmdlineParser.add_option("--no-cst", action="store_false",
                             default=True, dest="writeCstFile",
                             help="don't create .cst file")
    cmdlineParser.add_option("--no-idm", action="store_false",
                             default=True, dest="writeIdmFile",
                             help="don't create .idm file")
    cmdlineParser.add_option("--no-att", action="store_false",
                             default=True, dest="writeAttFile",
                             help="don't create model.att file")
    cmdlineParser.add_option("--target-sub-dirs", action="store_true",
                             default=False, dest="appendDateDirs",
                             help="Append yyyy/mm to end of output directory path")
    cmdlineParser.add_option("--new-rsk-fields", action="store_true",
                             default=False, dest="newRiskFields",
                             help="Include new fields in .rsk files")
    cmdlineParser.add_option("--warn-not-crash", action="store_true",
                             default=False, dest="notCrash",
                             help="Output warning rather than crashing when some fields are missing")
    cmdlineParser.add_option("--vendordb-user", action="store",
                         default=os.environ.get('VENDORDB_USER'), dest="vendorDBUser",
                         help="Vendor DB User")
    cmdlineParser.add_option("--vendordb-passwd", action="store",
                         default=os.environ.get('VENDORDB_PASSWD'), dest="vendorDBPasswd",
                         help="Vendor DB Password")
    cmdlineParser.add_option("--vendordb-sid", action="store",
                         default=os.environ.get('VENDORDB_SID'), dest="vendorDBSID",
                         help="Vendor DB SID")
    cmdlineParser.add_option("--file-format-version", action="store", type="float",
                             default=3.2, dest="fileFormatVersion",
                             help="version of flat file format to create")
    cmdlineParser.add_option("--histbeta-new", action="store_true",
                             default=False, dest="histBetaNew",
                             help="process historic beta new way or legacy way")

    
    (options, args) = cmdlineParser.parse_args()
    if len(args) < 1 or len(args) > 2:
        cmdlineParser.error("Incorrect number of arguments")
    
    Utilities.processDefaultCommandLine(options, cmdlineParser)
    modelDB = ModelDB.ModelDB(sid=options.modelDBSID, user=options.modelDBUser,
                              passwd=options.modelDBPasswd)
    
    marketDB = MarketDB.MarketDB(sid=options.marketDBSID, user=options.marketDBUser, passwd=options.marketDBPasswd)
    vendorDB = MarketDB.MarketDB(sid=options.vendorDBSID, user=options.vendorDBUser,
                              passwd=options.vendorDBPasswd)
    modelDB.setTotalReturnCache(150)
    modelDB.setMarketCapCache(150)
    modelDB.setVolumeCache(150)
    modelDB.setHistBetaCache(30)
    modelDB.cumReturnCache = None
    # just use today's date to get the USD currency ID - surely that will not change, will it??
    modelDB.createCurrencyCache(marketDB)
    dates = buildDateList(args)
    compositeFamilies = options.compositeFamilies.split(',')
    compositeFamilies = [c for c in compositeFamilies if len(c) > 0]
    fileFormat_ = writeFlatFiles.FlatFilesV3()
    
    for d in dates:
        if d.isoweekday() > 5 or (d.day == 1 and d.month == 1):
            logging.info('Skipping weekends and Jan-1, %s', d)
            continue
        logging.info('Processing %s', d)
        for cf in compositeFamilies:
            fileFormat_.writeCompositeDay(options, d, cf, modelDB, marketDB, vendorDB)
        logging.info("Done writing %s", d)
    modelDB.finalize()
    marketDB.finalize()

if __name__ == '__main__':
    main()
