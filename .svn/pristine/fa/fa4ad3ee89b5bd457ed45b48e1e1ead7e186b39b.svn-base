
import datetime
import logging
import optparse
import os
import time
import sys
import shutil
from marketdb import MarketDB
from riskmodels import ModelDB
from riskmodels import Utilities


def writeISINFile(options, modelList, modelFamilyName, d, numeraire):
    target=options.targetDir
    if options.appendDateDirs:
         target = os.path.join(target, '%04d' % d.year, '%02d' % d.month)
         inDir  = os.path.join(options.flatFileDir, '%04d' % d.year, '%02d' % d.month)
         try:
            os.makedirs(target)
         except OSError as e:
            if e.errno != 17:
               raise
            else:
               pass
    else:
        inDir = options.flatFileDir.rstrip('/')
    dtstr = str(d).replace('-','')

    target2=options.targetDir2
    if target2 and options.appendDateDirs:
         target2 = os.path.join(target2, '%04d' % d.year, '%02d' % d.month)
         try:
            os.makedirs(target2)
         except OSError as e:
            if e.errno != 17:
               raise
            else:
               pass

    errorCase=False
 
    # check to see if there is any file for the family in here.
    flatFileList = ['%s/%s.%s.idm' % (inDir, m, dtstr) for m in modelList if  not m.endswith('-FL') and not  m.endswith('-EL')]
    for flatFile in flatFileList:
        if not os.path.exists(flatFile):
            logging.error('File %s does not exist!', flatFile)
            errorCase=True
    if errorCase:
        return 1

    # for the flat files use the shortened family name instead
    familyName = 'AX' + modelFamilyName.replace('Axioma','')
    isinFileName = '%s/%s.%04d%02d%02d.isin' % (target, familyName, d.year, d.month, d.day)
    isinFile = open(isinFileName,'w')

    isinColNumber=None
    axidColNumber=None

    axidDict={}
    linecounts=[]
    for fileNumber, flatFile in enumerate(flatFileList):
        logging.info('Working on %s to %s', flatFile, isinFileName)
        for linecount, line in enumerate(open(flatFile)):
            if line.startswith('#Columns:'):
                # if it is not the first file ignore it
                if fileNumber > 0:
                    continue
                results=[idx for (idx,l) in enumerate(line.split('|')) if l.replace('#Columns: ','')=='ISIN']
                if len(results):
                    isinColNumber = results[0]
                results=[idx for (idx,l) in enumerate(line.split('|')) if l.replace('#Columns: ','')=='AxiomaID']
                if len(results):
                    axidColNumber = results[0]
                logging.info('AXID=%d ISIN column=%d', axidColNumber, isinColNumber)
            elif line.startswith('#'):
               continue
            else:
               if axidColNumber is not None and isinColNumber is not None:
                   linelist=line.split('|')
                   axid=linelist[axidColNumber]
                   isin=linelist[isinColNumber]
                   if isin and axid not in axidDict:
                       axidDict[axid] = isin
        linecounts.append(linecount) 
    # write out the headers for the new file
    isinFile.write('#DataDate: %s\n' % str(d)[:10])
    gmtime = time.gmtime(time.mktime(time.gmtime()))
    utctime = datetime.datetime(year=gmtime.tm_year, month=gmtime.tm_mon, day=gmtime.tm_mday,
                                hour=gmtime.tm_hour, minute=gmtime.tm_min, second=gmtime.tm_sec)
    isinFile.write('#CreationTimestamp: %sZ\n' % utctime.strftime('%Y-%m-%d %H:%M:%S'))
    if options.family:
        isinFile.write('#ModelFamily: %s\n' % modelFamilyName)
        isinFile.write('#ModelNumeraire: %s\n' % numeraire)
    else:
        isinFile.write('#ModelName: %s\n' % modelName)

    isinFile.write('#FlatFileVersion: 4.0\n')
    isinFile.write('#Columns: AxiomaID|ISIN\n')


    for axid, isin in sorted(axidDict.items()): 
        isinFile.write('%s|%s\n' % (axid, isin))
    logging.info('Line counts=%s axid count=%d', linecounts, len(axidDict))
    isinFile.close() 
    if target2:
        isinFileName2 = '%s/%s.%04d%02d%02d.isin' % (target2, familyName, d.year, d.month, d.day)
        shutil.copyfile(isinFileName, isinFileName2)
        logging.info('Copied file %s to %s', isinFileName, isinFileName2)

    return 0


if __name__ == '__main__':
    usage = "usage: %prog [options] <YYYY-MM-DD> [<YYYY-MM-DD>]" 
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addModelAndDefaultCommandLine(cmdlineParser)

    cmdlineParser.add_option("-d", "--directory", action="store",
                             default='.', dest="targetDir",
                             help="directory for output files")
    cmdlineParser.add_option("--directory2", action="store",
                             default=None, dest="targetDir2",
                             help="2nd directory for output files")
    cmdlineParser.add_option("--flat-file-dir", action="store",
                             default='.', dest="flatFileDir",
                             help="directory where input files are stored")
    cmdlineParser.add_option("--target-sub-dirs", action="store_true",
                                 default=False, dest="appendDateDirs",
                                 help="Append yyyy/mm to end of output and input directory path")
    cmdlineParser.add_option("--family", action="store_true",
                             default=False, dest="family",
                             help="write ISIN file for family")


    (options, args) = cmdlineParser.parse_args()
    if len(args) < 1 or len(args) > 2:
        cmdlineParser.error("Incorrect number of arguments")

    modelClass = Utilities.processModelAndDefaultCommandLine(options, cmdlineParser)
    modelDB = ModelDB.ModelDB(sid=options.modelDBSID, user=options.modelDBUser, passwd=options.modelDBPasswd)
    marketDB = MarketDB.MarketDB(sid=options.marketDBSID, user=options.marketDBUser, passwd=options.marketDBPasswd)

    modelFamilyMap=modelDB.getModelFamilies()

    riskModel = modelClass(modelDB, marketDB)
    familyToModelMap = {}
    for k, v in modelFamilyMap.items():
        if v not in familyToModelMap:
            familyToModelMap[v] = [k]
        else:
            familyToModelMap[v].append(k)
             
    startDate = Utilities.parseISODate(args[0])
    if len(args) == 1:
        endDate = startDate
    else:
        endDate = Utilities.parseISODate(args[1])
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

    errorCase=False
    for d in dates:
        logging.info('Processing %s' % str(d))
        if options.family:
            modelList = familyToModelMap[modelFamilyMap[riskModel.mnemonic]] 
            familyName = modelFamilyMap[riskModel.mnemonic]
        else:
            modelList =  [riskModel.mnemonic]
            familyName = riskModel.mnemonic
        status=writeISINFile(options, modelList, familyName, d, riskModel.numeraire.currency_code)
        if status:
            errorCase=True
        d += dayInc
    marketDB.finalize()
    modelDB.finalize()
    if errorCase:
        logging.error('Exiting with errors')
        sys.exit(1)
    else:
        sys.exit(0)
