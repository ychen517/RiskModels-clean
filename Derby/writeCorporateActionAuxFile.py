import datetime
import logging
import optparse

from marketdb import MarketDB
from riskmodels import ModelID
from riskmodels import ModelDB
from riskmodels import Utilities
import writeCorporateActionXML as ca

def getActiveModels(modelList, d, modelDB, allowPartialModels):
    """Since corporate actions can fall on non-trading days, look
    back up to 30 days to find an instance.
    """
    dateRange = [d - datetime.timedelta(days=i) for i in range(30)]
    activeModels = list()
    for riskModel in modelList:
        modelInstances = modelDB.getRiskModelInstances(riskModel.rms_id, dateRange)
        if len(modelInstances) > 0:
            modelDate = modelInstances[0].date
            if modelDate != d:
                print('%s is not a trading day for %s' % (d, riskModel.mnemonic))
            rmi = riskModel.getRiskModelInstance(modelDate, modelDB)
            if rmi != None and (rmi.has_risks or allowPartialModels):
                riskModel.setFactorsForDate(modelDate, modelDB)
                activeModels.append(riskModel)
    return activeModels

def processModelIdDateArgs(args, options):
    """Split argument by comma into model ID-date pairs and then by colon
    into the components.
    """
    dateModelIDMap = dict()
    pairs = list()
    if len(args) == 1:
        pairs.extend(args[0].split(','))
    if options.inputFile is not None:
        with open(options.inputFile, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if len(line) > 0 and ':' in line:
                    pairs.append(line)
    for modelID, date in [i.split(":") for i in pairs]:
        date = Utilities.parseISODate(date)
        modelID = ModelID.ModelID(string=modelID)
        dateModelIDMap.setdefault(date, list()).append(modelID)
    return dateModelIDMap

def moveRequestsToTradingDays(dateAssetMap, assetRmgs, modelDB):
    movers = list()
    for date, assets in dateAssetMap.items():
        for asset in assets:
            if asset.isCashAsset():
                # cash assets trade every day, irrespective of the RMG
                continue
            # query next trading day
            modelDB.dbCursor.execute("""SELECT dt FROM rmg_calendar rcal
               WHERE rcal.rmg_id = :rmg_arg AND rcal.sequence = (SELECT MIN(sequence) FROM rmg_calendar subCal
                   WHERE subCal.rmg_id = rcal.rmg_id AND subCal.dt >= :dt_arg)""",
                                     rmg_arg=assetRmgs[asset][1], dt_arg=date)
            nextTradingDay = modelDB.dbCursor.fetchone()[0].date()
            if date != nextTradingDay:
                logging.info('%s is not a trading day for %s/%s, extracting for %s instead',
                             date, asset.getIDString(), assetRmgs[asset][0], nextTradingDay)
                movers.append((asset, date, nextTradingDay))
    for (asset, currentDate, tradingDay) in movers:
        dateAssetMap[currentDate].remove(asset)
        if len(dateAssetMap[currentDate]) == 0:
            del dateAssetMap[currentDate]
        dateAssetMap.setdefault(tradingDay, list()).append(asset)

def main():
    usage = "usage: %prog [options] filedate modelid:date,..."
    cmdlineParser = optparse.OptionParser(usage=usage)
    cmdlineParser.add_option("--no-encryption", action="store_false",
                             dest="encryptActions", default=True,
                             help="Don't encrypt action values")
    cmdlineParser.add_option("--model-families", action="store",
                             default='', dest="modelFamilies",
                             help="comma-separated list of risk model families")
    cmdlineParser.add_option("--model-names", action="store",
                             default='', dest="modelNames",
                             help="comma-separated list of risk model names")
    cmdlineParser.add_option("--allow-partial-models", action="store_true",
                         default=False, dest="allowPartialModels",
                         help="extract even if the risk model is incomplete")
    cmdlineParser.add_option("--file", action="store",
                         default=None, dest="inputFile",
                         help="read modelid:date pairs from file, one per line")
    cmdlineParser.add_option("--with-currency-convergences", action="store_true",
                         default=False, dest="withCurrencyConvergences",
                         help="include currency convergences as corporate actions")
    
    Utilities.addDefaultCommandLine(cmdlineParser)
    
    (options, args) = cmdlineParser.parse_args()
    if len(args) > 2 or len(args) < 1:
        cmdlineParser.error("Incorrect number of arguments")
    
    Utilities.processDefaultCommandLine(options, cmdlineParser)
    modelDB = ModelDB.ModelDB(sid=options.modelDBSID, user=options.modelDBUser, passwd=options.modelDBPasswd)
    marketDB = MarketDB.MarketDB(sid=options.marketDBSID, user=options.marketDBUser, passwd=options.marketDBPasswd)
    modelDict=modelDB.getModelFamilies()
    
    fileDate = Utilities.parseISODate(args[0])
    dateAssetMap = processModelIdDateArgs(args[1:], options)
    if len(dateAssetMap) == 0:
        cmdlineParser.error("No modelid:date pairs specified")
    
    if options.modelFamilies:
        modelFamilies = [i.strip() for i in options.modelFamilies.split(',') if len(i) > 0]
    else:
        modelNames = [i.strip() for i in options.modelNames.split(',') if len(i) > 0]
        modelNames=[i for i in modelNames if (i=='AXTW-MH-S' or i[-3:]=='-MH')]
        logging.info("Working on model names %s", ','.join(modelNames))
        # make sure to only do this for -MH models other than TW where we allow MH-S model
        modelFamilies=list(set([modelDict[i] for i in modelNames if i in modelDict]))
    logging.info("Working on families %s", ','.join(modelFamilies))    
    if not ca.validModelFamilies(modelFamilies, modelDB):
        return
    
    familyModelMap = ca.getRiskModelFamilyToModelMap(modelFamilies, modelDB, marketDB)
    assetRmgs = ca.getAssetRMGMapping(modelDB)
    moveRequestsToTradingDays(dateAssetMap, assetRmgs, modelDB)
    currencyMap = marketDB.getCurrencyISOCodeMap()
    assetTypeMapper = ca.AssetTypeMapper(marketDB, modelDB)
    adjTypes = ca.getAdjustmentFactorTypes(marketDB)
    familyXMLMap = dict((modelFamily, ca.createCADocument())
                        for modelFamily in familyModelMap.keys())
    logging.info('%d dates will be processed', len(dateAssetMap))
    for d, modelIDList in sorted(dateAssetMap.items()):
        logging.info('Processing %s, %s', d, modelIDList)
        rmgCalendars = ca.determineRMGCalendars(d, datetime.timedelta(days=1), modelIDList, assetRmgs, modelDB)
        calendarDays = set()
        for rmgTradingDays, rmgCalendarDays in rmgCalendars.values():
            calendarDays.update(rmgCalendarDays)
        corporateActions, assetsWithCAs = ca.gatherCorporateActions(
            calendarDays, currencyMap, options.withCurrencyConvergences, assetRmgs, marketDB, modelDB)
        ca.moveCorporateActionsToTradingDays(corporateActions, set(modelIDList), assetsWithCAs,
                                             assetRmgs, rmgCalendars, [d])
        for (modelFamily, models) in familyModelMap.items():
            activeModels = getActiveModels(models, d, modelDB, options.allowPartialModels)
            if len(activeModels) > 0:
                (activeAssets, lookBack) = ca.processModelFamilyDay(activeModels, d, modelDB, marketDB,
                                                                    options, False)
                activeAssets &= set(modelIDList)
                if len(activeAssets) > 0:
                    (xmlDoc, xmlRoot) = familyXMLMap[modelFamily]
                    ca.addCorporateActions(xmlDoc, xmlRoot, corporateActions[d], currencyMap,
                                           adjTypes, d, options.encryptActions,
                                           assetTypeMapper, activeAssets, activeAssets)
            else:
                logging.info('Skipping %s for %s model family. No active models', d, modelFamily)
    # Write out XML files
    for (modelFamily, (xmlDoc, xmlRoot)) in familyXMLMap.items():
        outName = "CorpActions-%s-%s.aux" % (modelFamily, fileDate.strftime('%Y%m%d'))
        with open(outName, 'w') as outFile:
            xmlDoc.writexml(outFile, '  ', '  ', newl='\n', encoding='UTF-8')
    
    modelDB.finalize()
    marketDB.finalize()

if __name__ == '__main__':
    main()
