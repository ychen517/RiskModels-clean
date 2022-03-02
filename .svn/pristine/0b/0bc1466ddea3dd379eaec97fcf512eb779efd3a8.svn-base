
import datetime
import logging
import numpy.ma as ma
import numpy
import optparse
import sys
import time
import os

from marketdb import MarketDB
import riskmodels
from riskmodels import Matrices
from riskmodels import ModelDB
from riskmodels import Utilities
from riskmodels.Matrices import ExposureMatrix
from riskmodels import writeFlatFiles

NULL = ''
def numOrNull(val, fmt, threshhold=None, fmt2=None):
    """Formats a number with the specified format or returns NULL
    if the value is masked.
    """
    if val is ma.masked or val is None:
        return NULL
    if threshhold and fmt2 and val<threshhold:
        return fmt2 % val
    else:
        return fmt % val

def getSubFactors(modelDB, rms_id, allFactors=False):
    """get list of subfactor ids and names returned for the various factor names
    """
    # first find the list of subfactors that are in our master table
    if rms_id < 0:
        tblName = 'rms_m%d_fmp' % abs(rms_id)
    else:
        tblName = 'rms_%d_fmp' % rms_id
    query="select * from %s where 0=1" % tblName
    modelDB.dbCursor.execute(query)
    cols=[i[0].split('_')[1] for i in modelDB.dbCursor.description if i[0].upper().find('SF_')==0]

    # throw away the junk results
    junk=modelDB.dbCursor.fetchall()
    if allFactors:
        addQuery = ""
    else:
        addQuery = " and t.name = 'Style' "
    query="""select s.sub_id, f.name, t.name from
        sub_factor s, factor f, factor_type t where f.factor_id=s.factor_id and f.factor_type_id = t.factor_type_id
        and s.sub_id in (%s)  
        and f.name not in ('Domestic China')
        and t.name not in ('Currency')
        %s """ % (','.join(['%s' % c for c in cols]), addQuery)
    modelDB.dbCursor.execute(query)
    results= [r for r in modelDB.dbCursor.fetchall()]
    return results

def getFMP (modelDB, subfactors, rms_id, date):
    """get FMP data back for given rms_id and date and return it 
    """
    if rms_id < 0:
        tblName = 'rms_m%d_fmp' % abs(rms_id)
    else:
        tblName = 'rms_%d_fmp' % rms_id
    colList=','.join(['SF_%s' % i[0] for i in subfactors])
    query=""" select sub_issue_id, %s from %s where dt=:dt
    """ % (colList, tblName)
    modelDB.dbCursor.execute(query, dt=date)
    return dict([(r[0],r[1:]) for r in modelDB.dbCursor.fetchall()])
        
def runLoop(riskModel, dates, modelDB, marketDB, options):
    status = 0
    factors = sorted(modelDB.getRiskModelSerieFactors(riskModel.rms_id), key=lambda x: x.name)
    subfactors=getSubFactors(modelDB, riskModel.rms_id, allFactors=options.allFactors)
    # sort the subfactors by type, but only for the .pfp files and before getting the FMP out
    if options.allFactors:
        SDICT={'Style':0, 'Market':1, 'Local':2,'Industry':100, 'Country':105}
        subfactors=[s+ (SDICT.get(s[2],50),) for s in subfactors]
        subfactors.sort(key=lambda tup: (tup[3],tup[1]))

    for d in dates:
        logging.info('Processing %s', d)
        try:
            riskModel.setFactorsForDate(d, modelDB)
            rmi = modelDB.getRiskModelInstance(riskModel.rms_id, d)
            #assets = modelDB.getRiskModelInstanceUniverse(rmi)
            resDict = getFMP(modelDB, subfactors, riskModel.rms_id, d)

            # Let's publish it now and be damned
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

            options.fileFormatVersion = 4.0
            if options.allFactors:
                ext = 'pfp'
            else:
                ext = 'fmp'
            fileName = '%s/%s.%04d%02d%02d.%s' % \
                    (target,riskModel.mnemonic, d.year, d.month, d.day, ext)

            outFile = open(fileName, 'w')
            columnNames = '#Columns: AxiomaID|%s' % '|'.join([s[1] for s in subfactors])
            # Write the header but using the FlatFilesv3 class
            fileFormat = writeFlatFiles.FlatFilesV3()
            fileFormat.dataDate_ = d
            fileFormat.createDate_ = modelDB.revDateTime
            fileFormat.writeDateHeader(options, outFile, riskModel) 
            outFile.write('%s\n' % columnNames.rstrip('|'))
            # And other header stuff
            typeList = '#Type: ID|'
            for iLen in range(columnNames.count('|')):
                typeList += '%s|' % subfactors[iLen][2]
            outFile.write('%s\n' % typeList.rstrip('|'))

            # Loop through the IDs and write the data
            dropped=0
            for sid in sorted(resDict.keys()):
                if not any(resDict[sid]):
                    logging.debug("Dropping asset %s", sid)
                    dropped+=1
                    continue
                outFile.write('%s|' % sid[1:-2])
                # Write all the FMP data for this asset
                if options.allFactors:
                    outFile.write('|'.join(['%.8g' % r if r else '' for r in resDict[sid]]))
                else:
                    outFile.write('|'.join(['%.8f' % r if r else '' for r in resDict[sid]]))
                outFile.write('\n')
            outFile.close()
            logging.info('Dropped %d assets and processed %d', dropped, len(resDict)-dropped)
        except Exception:
            logging.error('Exception caught during processing', exc_info=True)
            modelDB.revertChanges()
            if not riskModel.forceRun:
                status = 1
                break
    return status

if __name__ == '__main__':
    usage = "usage: %prog [options] <startdate or datelist> [<end-date>]"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addModelAndDefaultCommandLine(cmdlineParser)
    
    # Typical model generation steps
    cmdlineParser.add_option("-n", action="store_true",
                             default=False, dest="testOnly",
                             help="don't change the database")
    cmdlineParser.add_option("--verbose", "-v", action="store_true",
                             default=False, dest="verbose",
                             help="perform a lot of debugging diagnostics")
    cmdlineParser.add_option("--force", "-f", action="store_true",
                             default=False, dest="force",
                             help="override certain constraints")
    cmdlineParser.add_option("-d", "--directory", action="store",
                             default='.', dest="targetDir",
                             help="directory for output files")
    cmdlineParser.add_option("--target-sub-dirs", action="store_true",
                             default=False, dest="appendDateDirs",
                             help="Append yyyy/mm to end of output directory path")
    cmdlineParser.add_option("--all-factors", action="store_true",
                             default=False, dest="allFactors",
                             help="all factors, not just style")
 
    (options, args) = cmdlineParser.parse_args()
    if len(args) < 1 or len(args) > 2:
        cmdlineParser.error("Incorrect number of arguments")
    Utilities.processDefaultCommandLine(options, cmdlineParser)

    modelDB = ModelDB.ModelDB(sid=options.modelDBSID, 
                              user=options.modelDBUser,
                              passwd=options.modelDBPasswd)
    marketDB = MarketDB.MarketDB(sid=options.marketDBSID, 
                                 user=options.marketDBUser,
                                 passwd=options.marketDBPasswd)

    # Determine the model family and define the 4 model variants
    modelName = options.modelName
    riskModelClass = riskmodels.getModelByName(modelName)
    riskModel = riskModelClass(modelDB, marketDB)
    if options.verbose:
        riskModel.debuggingReporting = True
    if options.force:
        riskModel.forceRun = True
                 
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
        startDate = min(dates)
        endDate = max(dates)
        modelDates = modelDB.getDateRange(
                riskModel.rmg, startDate, endDate, excludeWeekend=True)
        dates = sorted(dates & set(modelDates))
    else:
        startDate = Utilities.parseISODate(args[0])
        endDate = Utilities.parseISODate(args[1])
        dates = modelDB.getDateRange(
                riskModel.rmg, startDate, endDate, excludeWeekend=True)
    
    status = runLoop(riskModel, dates, modelDB, marketDB, options)
    marketDB.finalize()
    modelDB.finalize()
    sys.exit(status)
