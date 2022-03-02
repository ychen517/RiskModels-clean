
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
from riskmodels import AssetProcessor
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

def writeDateHeader(date, outFile):
    outFile.write('#DataDate: %s\n' % date)
    createDate_ = modelDB.revDateTime
    # write createDate in UTC
    gmtime = time.gmtime(time.mktime(createDate_.timetuple()))
    utctime = datetime.datetime(year=gmtime.tm_year,
            month=gmtime.tm_mon,
            day=gmtime.tm_mday,
            hour=gmtime.tm_hour,
            minute=gmtime.tm_min,
            second=gmtime.tm_sec)
    outFile.write('#CreationTimestamp: %sZ\n' %
            utctime.strftime('%Y-%m-%d %H:%M:%S'))

def runLoop(riskModel, dates, modelDB, marketDB, options):
    status = 0
    for d in dates:
        logging.info('Processing betas for %s', d)
        try:
            if len(riskModel.rmg) > 1:
                SCM = False
            else:
                SCM = True

            SCMModel=SCM
            if riskModel.mnemonic.startswith('AXCN4'):
                SCMModel=False

            riskModel.setFactorsForDate(d, modelDB)
            rmi = modelDB.getRiskModelInstance(riskModel.rms_id, d)
            assets = modelDB.getRiskModelInstanceUniverse(rmi)
            excludes = modelDB.getProductExcludedSubIssues(d)
            excludeDict = dict([e,1] for e in excludes)

            # Determine home country info and flag DR-like instruments
            data = AssetProcessor.process_asset_information(
                    d, assets, riskModel.rmg, modelDB, marketDB,
                    checkHomeCountry=(SCMModel==0),
                    numeraire_id=riskModel.numeraire.currency_id,
                    legacyDates=riskModel.legacyMCapDates,
                    forceRun=riskModel.forceRun)

            # Load in the historic betas. These are model-independent
            legacyBeta = modelDB.getPreviousHistoricBetaFixed(d, data.universe)
            historicBeta_home = modelDB.getHistoricBetaDataV3(
                    d, data.universe, field='value', home=1, rollOverData=True)
            nobs_home = modelDB.getHistoricBetaDataV3(
                    d, data.universe, field='nobs', home=1, rollOverData=True)
            historicBeta_trad = modelDB.getHistoricBetaDataV3(
                    d, data.universe, field='value', home=0, rollOverData=True)

            # Load the assorted V3 beta fields 
            localBetaWithCurrency = modelDB.getRMIPredictedBetaV3(rmi, field='local_num_beta')
            localBeta = modelDB.getRMIPredictedBetaV3(rmi, field='local_beta')
            globalBeta = modelDB.getRMIPredictedBetaV3(rmi, field='global_beta')

            # Let's publish it now and be damned
            idx = riskModel.mnemonic.find('-')
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
            #fileName = '%s/%s.F%2d.%04d%02d%02d.bet' % \
            #        (target,riskModel.mnemonic, options.fileFormatVersion * 10,d.year, d.month, d.day)
            fileName = '%s/%s.%04d%02d%02d.bet' % \
                    (target,riskModel.mnemonic, d.year, d.month, d.day)

            outFile = open(fileName, 'w')
            columnNames = '#Columns: AxiomaID|'
            columnNames += 'Historical Beta|Historical Beta (Trade)|Historical Beta (Home)|'
            columnNames += 'Number of Returns|'
            if SCM:
                columnNames += 'Predicted Beta (Local, Hedged)'
            else:
                # Note that the same beta type is defined as hedged for an SCM, but unhedged 
                # for a regional model. This is correct and due to legacy definitions
                columnNames += 'Predicted Beta (Local, Hedged)|'
                columnNames += 'Predicted Beta (Local, Unhedged)|'
                columnNames += 'Predicted Beta (Global, Unhedged)'
            # Write the header but using the FlatFilesv3 class
            fileFormat = writeFlatFiles.FlatFilesV3()
            fileFormat.dataDate_ = d
            fileFormat.createDate_ = modelDB.revDateTime
            fileFormat.writeDateHeader(options, outFile, riskModel) 
            outFile.write('%s\n' % columnNames.rstrip('|'))
            # And all the other header crap
            typeList = '#Type: ID|'
            unitList = '#Unit: ID|'
            for iLen in range(columnNames.count('|')):
                typeList += 'Attribute|'
                unitList += 'Number|'
            outFile.write('%s\n' % typeList.rstrip('|'))
            outFile.write('%s\n' % unitList.rstrip('|'))

            if riskModel.debuggingReporting:
                db_legacyHistoricBeta = ma.masked_all((len(data.universe)), float)
                db_tradingHistoricBeta = ma.masked_all((len(data.universe)), float)
                db_homeHistoricBeta = ma.masked_all((len(data.universe)), float)
                db_predictedBetaWCurrency = ma.masked_all((len(data.universe)), float)
                db_predictedBetaWOCurrency = ma.masked_all((len(data.universe)), float)

            # Loop through the IDs and write the data
            for (idx, sid) in enumerate(sorted(data.universe)):
                if sid in excludeDict:
                    logging.info('Excluding %s', sid)
                    continue
                outFile.write('%s|' % sid.getModelID().getPublicID())
                # Write historical beta data

                if options.histBetaNew:
                    if SCM:
                        hbetaVal = historicBeta_trad.get(sid, historicBeta_home.get(sid,None))
                    else:
                        hbetaVal = historicBeta_home.get(sid, None)
                else:
                    hbetaVal = legacyBeta.get(sid, None)
 
                outFile.write('%s|%s|%s|%s|' % \
                        (numOrNull(hbetaVal, '%.4f'),
                         numOrNull(historicBeta_trad.get(sid, None) or historicBeta_home.get(sid, None), '%.4f'),
                         numOrNull(historicBeta_home.get(sid, None), '%.4f'),
                         numOrNull(nobs_home.get(sid, None), '%d'),))
                if SCM:
                    outFile.write('%s' %  (numOrNull(localBetaWithCurrency.get(sid, None), '%.4f')))
                else:
                    # Write local betas without currency component
                    outFile.write('%s|' % (numOrNull(localBeta.get(sid, None), '%.4f')))
                    # Write model numeraire-perspective local betas
                    outFile.write('%s|' %  (numOrNull(localBetaWithCurrency.get(sid, None), '%.4f')))
                    # Write model "global" betas
                    outFile.write('%s' %  (numOrNull(globalBeta.get(sid, None), '%.4f')))
                outFile.write('\n')

                if riskModel.debuggingReporting:
                    #db_legacyHistoricBeta[idx] = legacyBeta.get(sid, ma.masked)
                    db_legacyHistoricBeta[idx] = hbetaVal #legacyBeta.get(sid, ma.masked)
                    db_tradingHistoricBeta[idx] = historicBeta_trad.get(sid, ma.masked)
                    db_homeHistoricBeta[idx] = historicBeta_home.get(sid, ma.masked)
                    db_predictedBetaWCurrency[idx] = localBetaWithCurrency.get(sid, ma.masked)
                    if not SCM:
                        db_predictedBetaWOCurrency[idx] = localBeta.get(sid, ma.masked)

            outFile.close()

            # Some testing output
            if riskModel.debuggingReporting:
                homeCorrel = ma.corrcoef(db_legacyHistoricBeta, db_homeHistoricBeta)[0,1]
                trdCorrel = ma.corrcoef(db_legacyHistoricBeta, db_tradingHistoricBeta)[0,1]
                predCorrel = ma.corrcoef(db_predictedBetaWCurrency, db_homeHistoricBeta)[0,1]
                predCorrel2 = ma.corrcoef(db_predictedBetaWCurrency, db_tradingHistoricBeta)[0,1]
                logging.info('Correlation with legacy betas: %.4f (Home), %.4f (trade)',
                        homeCorrel, trdCorrel)
                logging.info('Correlation with predicted legacy betas: %.4f (Home), %.4f (trade)',
                        predCorrel, predCorrel2)
                if not SCM:
                    predCorrel = ma.corrcoef(db_predictedBetaWOCurrency, db_homeHistoricBeta)[0,1]
                    predCorrel2 = ma.corrcoef(db_predictedBetaWOCurrency, db_tradingHistoricBeta)[0,1]
                    logging.info('Correlation with predicted local betas: %.4f (Home), %.4f (trade)',
                            predCorrel, predCorrel2)

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
    cmdlineParser.add_option("--hist-beta-new", action="store_true",
                             default=False, dest="histBetaNew",
                             help="Apply new treatment of hist beta")
 
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
