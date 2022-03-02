
import logging
import numpy
import os
import numpy.ma as ma
import optparse
import sys
import pandas as pd
import pandas.io.sql as sql
import configparser
import time
import datetime
import io

from marketdb import MarketDB
import marketdb.Utilities as mktUtilities
from marketdb.FactSetFundAttributes import FundDataMap
from riskmodels import Connections
from riskmodels import ModelDB
from riskmodels import Utilities
from riskmodels import transfer
from riskmodels.writeFlatFiles import TempFile


#df = pd.read_sql(qry, conn.dbConnection)

def writeHeader(outFile, date, codeMap):
    outFile.write('#DataDate: %s\n' % date)
    # write createDate in UTC
    gmtime = time.gmtime(time.mktime(datetime.datetime.now().timetuple()))
    utctime = datetime.datetime(year=gmtime.tm_year,
                                    month=gmtime.tm_mon,
                                    day=gmtime.tm_mday,
                                    hour=gmtime.tm_hour,
                                    minute=gmtime.tm_min,
                                    second=gmtime.tm_sec)
    outFile.write('#CreationTimestamp: %sZ\n' % utctime.strftime('%Y-%m-%d %H:%M:%S'))
    outFile.write("#FlatFileVersion: 4.0\n")
    outFile.write('#AXIOMA_ID|DT|CURRENCY')
    for (idx,k) in enumerate(sorted(codeMap.keys())):
        outFile.write('|%s' % codeMap[k])
    outFile.write('\n')

def writeData(outFile, date,retvals, codeMap):
    dropped=0
    for k in sorted(retvals.keys()):
        empty=True
        # stash all the values in line
        line = io.StringIO()        
        for (idx,code) in enumerate(sorted(codeMap.keys())):
            if len(retvals[k][code].getFields())==0:
                value='' 
            else:
                value='%.16g' % retvals[k][code].value
                curr=retvals[k][code].currency
                if curr is None: value = ''
                empty=False
            line.write('|%s' % (value))
        line.write('\n') 
        if empty:
            logging.debug('Dropping asset %s since there is no FS fund data', k) 
            dropped=dropped+1
        else:
            outFile.write('%s|%s|%s' % (k[1:], date, curr))
            outFile.write(line.getvalue()) 
    logging.info('Dropped %d assets since they had no FS fund data', dropped)
 
def main():

    usage = "usage: %prog [options] config-file <YYYY-MM-DD>"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addModelAndDefaultCommandLine(cmdlineParser)
    cmdlineParser.add_option("-d", "--directory", action="store",
                         default='.', dest="targetDir",
                         help="directory for output files")
    cmdlineParser.add_option("--encrypt", action="store_true",
                         default=False, dest="encrypt",
                         help="directory for output files")
    cmdlineParser.add_option("--corrections", action="store_true",
                         default=False, dest="corrections",
                         help="directory for output files")
    cmdlineParser.add_option("--clear-dir-name", action="store",
                         default='.', dest="clearDirName",
                         help="directory for output files")
    cmdlineParser.add_option("--prelim", action="store_true",
                         default=False, dest="prelim",
                         help="prelim version or not")
    cmdlineParser.add_option("-s", "--sub-issue-ids", action="store",
                         default=None, dest="subIDs",
                         help="sub-issue-ids only to use")
    (options, args) = cmdlineParser.parse_args()
    if len(args) < 2:
        cmdlineParser.error("Incorrect number of arguments")

    Utilities.processDefaultCommandLine(options, cmdlineParser)
    logging.info('Start')
    configFile_ = open(args[0])
    config_ = configparser.ConfigParser()
    config_.read_file(configFile_)
    configFile_.close()
    connections = Connections.createConnections(config_)
    date = Utilities.parseISODate(args[1])
    
    modelDB = connections.modelDB
    if len(args) > 2:
        enddate = args[2] 
    else:
        enddate = date

    dateList=mktUtilities.createDateList('%s:%s' %(args[1], enddate))

    dirName=options.targetDir
    if dirName and not os.path.exists(dirName):
        try:
            os.makedirs(dirName)
        except OSError:
            excstr=str(sys.exc_info()[1])
            if excstr.find('File exists') >= 0 and excstr.find(dirName) >= 0:
                logging.info('Error can be ignored - %s' % excstr)
            else:
                raise

    clearDirName=options.clearDirName
    if clearDirName and not os.path.exists(clearDirName):
        try:
            os.makedirs(clearDirName)
        except OSError:
            excstr=str(sys.exc_info()[1])
            if excstr.find('File exists') >= 0 and excstr.find(clearDirName) >= 0:
                logging.info('Error can be ignored - %s' % excstr)
            else:
                raise

    if options.prelim:
        prelim='-PRELIM'
    else:
        prelim=''

    INCR=1000

    if options.corrections:
        corr='-corr'
        filedt=str(enddate)[:10].replace('-','')
    else:
        corr=''
        filedt=None

    # have to a bunch of stuff for each date - unfortunate, but unavoidable
    for dIdx, dt in enumerate(dateList):
        date=dt
        if date.isoweekday() in [6,7]:
            logging.info('Ignoring %s since it is a weekend', date)
            continue 

        logging.info('Processing data for %s', date)

        if options.subIDs:
            subIssues, ignore = transfer.createSubIssueIDList(options.subIDs, connections) 
            mdlMktMap = dict([(sid.getModelID(), None) for sid in subIssues])
            # use the first date for now
            mktMdlMap = dict([(mktId, mid) for (mid, mktId) in modelDB.getIssueMapPairs(date) if mid in mdlMktMap])
        else:
            subIssues = modelDB.getAllActiveSubIssues(date, inModels=True)
            excludes = modelDB.getProductExcludedSubIssues(date)
            subIssues = list(set(subIssues) - set(excludes))
            logging.info('Loaded up %d subissues', len(subIssues))
            mdlMktMap = dict([(sid.getModelID(), None) for sid in subIssues])
            mktMdlMap = dict([(mktId, mid) for (mid, mktId) in modelDB.getIssueMapPairs(date) if mid in mdlMktMap])

        # fill in the reverse map for mdl to mkt
        for mktId,mid in mktMdlMap.items():
            mdlMktMap[mid] = mktId

        logging.info('Loaded up %d issues', len(mktMdlMap))

        fs=FundDataMap(connections, mktMdlMap)
        mdlidList=[k for k in sorted(mdlMktMap.keys()) if mdlMktMap[k]]
        logging.info('%d issues to consider', len(mdlidList))

        shortdate=str(dt)[:10].replace('-','')
        # create the file name only for the regular case.
        # if it is a correction, we need just one file
        if (options.corrections and dIdx==0) or not options.corrections:
            destFileName='AXFSFund%s.%s.txt' % (corr,filedt or shortdate)
            clearbasename='AXFSFund%s-clear.%s.txt' % (corr,filedt or shortdate)
            clearFileName='%s/%s' % (options.clearDirName.rstrip('/'), clearbasename)
            outputFile = TempFile(clearFileName, shortdate)
            outFile=outputFile.getFile()
            writeHeader(outFile, date, fs.codeMap)

        retvals={}
        for idx,mdlidChunk in enumerate(mktUtilities.listChunkIterator(mdlidList, INCR)):
            retvals.update(fs.getData([dt], mdlidChunk))
            # dump out each chunk as it comes along, since we do it one date at a time, the index is always 0
            writeData(outFile,dt, retvals[0], fs.codeMap)

        #close and move the files and encrypt if it is a non-correction case.  For corrections, do it only at the end
        if not options.corrections or (options.corrections and dIdx == len(dateList)-1):
            outputFile.closeAndMove()

            if options.encrypt:
                Utilities.encryptFile(date, clearFileName, dirName, destFileName=destFileName)

    modelDB.finalize()
    logging.info('Done')


if __name__ == '__main__':
    main()

