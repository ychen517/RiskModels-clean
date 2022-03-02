import logging.config
import optparse
import sys
import datetime
import numpy
import numpy.ma as ma
from riskmodels import ModelDB
from marketdb import MarketDB
from riskmodels  import LegacyUtilities as Utilities
from riskmodels import AssetProcessor
from riskmodels import LegacyProcessReturns as ProcessReturns

def computeMarketPortfolio(rmg, realRMG, currDate, rmgSubIssues, forexCache, daysBack, modelDB):
    logging.info('Generating market portfolio for %s on %s', rmg.description, currDate)
    
    # Load subset of market IDs
    market = modelDB.getRMGMarketPortfolio(rmg, currDate)
    market = [(sid, wt) for (sid, wt) in market if sid in rmgSubIssues]
    assetIdxMap = dict(zip(rmgSubIssues, list(range(len(rmgSubIssues)))))
    if len(market) == 0:
        mktSubIssues = list(rmgSubIssues)
        wts = numpy.zeros((len(mktSubIssues)), float)
    else:
        mktSubIssues, wts = zip(*market)
        wts = numpy.array(wts, dtype=float)
    if len(mktSubIssues) == 0:
        logging.info('No market portfolio, skipping.')
        return []
    currencyCode = rmg.getCurrencyCode(currDate)
    currencyID = forexCache.getCurrencyID(currencyCode, currDate)
    
    # Load asset history for last few months
    assetReturns = modelDB.loadTotalReturnsHistoryV3(
            [realRMG], currDate, rmgSubIssues, 60, assetConvMap=currencyID)
    if currDate not in assetReturns.dates:
        logging.info('Date %s not in available returns dates, skipping', currDate)
        return []
    rollOverFlag = ma.array(ma.filled(assetReturns.notTradedFlag, 1.0), bool)
    
    # Mask asset returns with True roll-over flag
    assetReturns.data = ma.masked_where(rollOverFlag, assetReturns.data)
    assetReturns.data = ma.masked_where(assetReturns.data==0.0, assetReturns.data)
    
    # Take only assets with >95% of returns non-missing in latest month
    # for market portfolio (if possible)
    mktIdx = [assetIdxMap[sid] for sid in mktSubIssues]
    assetMktRets = ma.take(assetReturns.data, mktIdx, axis=0)
    nMissing = numpy.sum(ma.getmaskarray(assetMktRets), axis=1) / float(assetMktRets.shape[1])
    bound = max(0.05, min(nMissing)+0.05)
    nMissing = ma.masked_where(nMissing > bound, nMissing)
    okIdx = numpy.flatnonzero(~ma.getmaskarray(nMissing))
    logging.info('Dropping %s out of %s illiquid/new assets from %s market',
            len(mktSubIssues)-len(okIdx), len(mktSubIssues), realRMG.mnemonic)
    mktSubIssues = numpy.take(mktSubIssues, okIdx, axis=0)
    
    # Set asset info
    mktIdx = [assetIdxMap[sid] for sid in mktSubIssues]
    return mktIdx

def computeProxyReturns(rmg, currDate, rmgSubIssues, saveIDs,
                        modelDB, marketDB, debuggingReporting, forexCache, daysBack,
                        options_, backFilledIssues=None):
    logging.info('Processing return proxies for %s on %s',
                  rmg.description, currDate)
    # Do some fiddling about to deal with China
    if rmg.description == 'Domestic China':
        rmgList = modelDB.getAllRiskModelGroups()
        realRMG = [r for r in rmgList if r.description=='China'][0]
    else:
        realRMG = rmg
    currencyCode = rmg.getCurrencyCode(currDate)
    currencyID = forexCache.getCurrencyID(currencyCode, currDate)
    data = AssetProcessor.process_asset_information(
            currDate, rmgSubIssues, [realRMG], modelDB, marketDB,
            numeraire_id=currencyID, legacyDates=True, forceRun=True)
    mktIdx = computeMarketPortfolio(rmg, realRMG, currDate, data.universe, forexCache,
            daysBack, modelDB)
    # Set up proxy returns estimation
    returnsProcessor = ProcessReturns.assetReturnsProcessor(
            [realRMG], data.universe, data.rmgAssetMap, data.tradingRmgAssetMap,
            data.assetTypeDict, data.marketTypeDict, dr2UnderDict=data.dr2Underlying,
            debuggingReporting=debuggingReporting,
            numeraire_id=currencyID, returnsTimingID=None,
            estu=mktIdx, testOnly=options_.testOnly, dontWrite=options_.dontWrite, saveIDs=saveIDs)
    assetReturnMatrix = returnsProcessor.process_returns_history(
        currDate, daysBack, modelDB, marketDB, qadLoad=options_.qadFlag, backFilledIssues=backFilledIssues)
    return

def main():
    usage = "usage: %prog [options] <YYYY-MM-DD> [<YYYY-MM-DD>]"
    cmdlineParser = optparse.OptionParser(usage=usage)
    cmdlineParser.add_option("--verbose","-v", action="store_true",
                             default=False, dest="debuggingReporting",
                             help="extra debugging output")
    cmdlineParser.add_option("-n", action="store_true",
                             default=False, dest="testOnly",
                             help="revert all changes to the DB")
    cmdlineParser.add_option("--dw", action="store_true",
                             default=False, dest="dontWrite",
                             help="don't even attempt to write to the database")
    cmdlineParser.add_option("--rmgs", action="store",
                             default='all', dest="rmgs",
                             help="list of rmg IDs/mnemonics for processing")
    cmdlineParser.add_option("--clean", "-c", action="store_true",
                             default=False, dest="clean",
                             help="delete current records prior to beginning")
    cmdlineParser.add_option("--nuke", action="store_true",
                             default=False, dest="nukem",
                             help="delete all records")
    cmdlineParser.add_option("--qad", action="store_true",
                             default=False, dest="qadFlag",
                             help="Quick and Dirty returns proxy for current day only")
    cmdlineParser.add_option("--ids", action="store",
                             default=None, dest="saveIDs",
                             help="write results for subset of IDs only")
    cmdlineParser.add_option("--skip-drs", action="store_true",
                             default=False, dest="skipDRStep",
                             help="Skip the DR backfill step")
    Utilities.addDefaultCommandLine(cmdlineParser)
    (options_, args_) = cmdlineParser.parse_args()
    if len(args_) < 1:
        cmdlineParser.error("Incorrect number of arguments")

    # Set up DB stuff
    modelDB = ModelDB.ModelDB(sid=options_.modelDBSID,
                               user=options_.modelDBUser,
                               passwd=options_.modelDBPasswd)
    marketDB = MarketDB.MarketDB(sid=options_.marketDBSID,
                                 user=options_.marketDBUser,
                                 passwd=options_.marketDBPasswd)
    Utilities.processDefaultCommandLine(options_, cmdlineParser)
    
    # Date manipulation
    startDate_ = Utilities.parseISODate(args_[0])
    if len(args_) > 1:
        endDate_ = Utilities.parseISODate(args_[1])
    else:
        endDate_ = startDate_
    dates_ = modelDB.getDateRange(None, startDate_, endDate_)

    # Get list of specific assets to be updated if relevant
    saveIDs = []
    if options_.saveIDs is not None:
        saveIDs = [i.strip() for i in options_.saveIDs.split(',')]

    # A few safeguards
    if options_.testOnly or options_.dontWrite:
        options_.nukem = False
        options_.clean = False
    
    # Process list of RMGs input
    allRMGs = modelDB.getAllRiskModelGroups(inModels=False)
    rmgIdMap = dict((rmg.rmg_id, rmg) for rmg in allRMGs)
    rmgIdMnMap = dict((rmg.mnemonic, rmg.rmg_id) for rmg in allRMGs)
    rmgStrings = [i.strip() for i in options_.rmgs.split(',')]
    rmgIds = set()
    mdlCur = modelDB.dbCursor
    if options_.skipDRStep:
        linkedAssets = False
    else:
        linkedAssets = True
    for rmgString in rmgStrings:
        if rmgString == 'all':
            rmgIds |= set([int(i) for i in rmgIdMap.keys()])
            linkedAssets = True
        elif rmgString == 'DR':
            linkedAssets = True
        elif rmgString.isdigit():
            rmgIds.add(int(rmgString))
        elif len(rmgString) == 2:
            rmgIds |= set([rmgIdMnMap[rmgString]])
        else:
            logging.error('Incorrect format for RMG: %s. Skipping'
                          % rmgString)
    rmgIds = sorted(rmgIds)
    logging.info('Loaded %d risk model group(s)', len(rmgIds))
    
    # Set cache info
    modelDB.createCurrencyCache(marketDB)
    forexCache = modelDB.currencyCache
    daysBack = 250
    modelDB.setTotalReturnCache(2*daysBack+1)
    modelDB.setVolumeCache(2*daysBack+1)
    modelDB.setMarketCapCache(2*daysBack+1)
    modelDB.proxyReturnCache = None # We're generating/updating proxy returns, so don't cache them
    
    # Get dictionary of RMGs trading on each day
    minDt = min(dates_)
    maxDt = max(dates_)
    tradingRMGs = dict([(date, set()) for date in dates_])
    xcRMG = [r for r in allRMGs if r.description=='Domestic China'][0]
    chinaRMG = [r for r in allRMGs if r.description=='China'][0]
    for rmgId in rmgIds:
        rmg = Utilities.Struct()
        rmg.rmg_id = rmgId
        if rmgId == xcRMG.rmg_id:
            # For XC (Domestic China) use the CN trading calendar
            rmg.rmg_id = chinaRMG.rmg_id
        tradingDays = modelDB.getDateRange([rmg], minDt, maxDt)
        for td in tradingDays:
            if td in tradingRMGs:
                tradingRMGs[td].add(rmgId)
    
    # Loop round dates
    for (dIdx, date) in enumerate(dates_):
        
        # Remove existing data from DB if required
        if options_.nukem:
            check = input('About to delete all records. Are you sure? ').lower()
            if check == 'yes' or check == 'y':
                options_.nukem = False
                modelDB.dbCursor.execute("""SELECT dt FROM rmg_proxy_return""")
                ret = modelDB.dbCursor.fetchall()
                nukeDates = sorted(set(r[0] for r in ret))
                for dt in nukeDates:
                    modelDB.deleteProxyReturns([], dt)
                    modelDB.commitChanges()
        
        # Get list of linked assets
        entireUniverse = modelDB.getAllActiveSubIssues(date)
        subIssueGroups = modelDB.getIssueCompanyGroups(date, entireUniverse, marketDB)
        linkedIssues = [sid for sidList in subIssueGroups.values() for sid in sidList]
        saveIDs = [sid for sid in entireUniverse if sid.getSubIDString() in saveIDs]
        fromDates = modelDB.loadIssueFromDates([date], entireUniverse)
        newIPOs = [sid for (idx, sid) in enumerate(entireUniverse) if fromDates[idx]==date]
        oneYearBack = date - datetime.timedelta(500)
        ipoDeleteDates = modelDB.getDateRange(None, oneYearBack, date)

        if linkedAssets:

            if options_.clean:
                logging.info('Clearing all records for %d assets on %s',
                        len(linkedIssues), date)
                modelDB.deleteProxyReturns(linkedIssues, date)
                subIPOs = [sid for sid in newIPOs if sid in linkedIssues]
                if len(subIPOs) > 0:
                    logging.info('Clearing back-history for %d assets', len(subIPOs))
                    modelDB.deleteProxyReturns(subIPOs, ipoDeleteDates)

            else:
                # Determine home country info and flag DR-like instruments
                rmgList = list(rmgIdMap.values())
                data = AssetProcessor.process_asset_information(
                        date, linkedIssues, rmgList, modelDB, marketDB,
                        checkHomeCountry=True, legacyDates=True, forceRun=True)
        
                # Compute proxy returns for linked assets
                checkDates = modelDB.getDates(rmgList, date, 1, excludeWeekend=True)
                if date in checkDates:
                    returnsProcessor = ProcessReturns.assetReturnsProcessor(
                            rmgList, data.universe, data.rmgAssetMap, data.tradingRmgAssetMap,
                            data.assetTypeDict, data.marketTypeDict, dr2UnderDict=data.dr2Underlying,
                            debuggingReporting=options_.debuggingReporting,
                            testOnly=options_.testOnly, dontWrite=options_.dontWrite, saveIDs=saveIDs)
                    assetReturnMatrix = \
                            returnsProcessor.process_returns_history(
                                    date, int(daysBack), modelDB, marketDB,
                                    drCurrMap=data.drCurrData, backFill=True,
                                    qadLoad=options_.qadFlag)
                else:
                    logging.info('%s not a valid date. Skipping linked asset proxy', date)
        
        # Loop round markets
        for rmgId in rmgIds:
            rmg = rmgIdMap[rmgId]
            rmgSubIssues = modelDB.getActiveSubIssues(rmg, date)
            checkDates = modelDB.getDates([rmg], date, 1, excludeWeekend=True)

            if options_.clean:
                cleanIssues = [sid for sid in rmgSubIssues if sid not in linkedIssues]
                logging.info('Clearing all records for %d assets on %s',
                        len(cleanIssues), date)
                subIPOs = [sid for sid in newIPOs if sid in cleanIssues]
                if len(subIPOs) > 0:
                    logging.info('Clearing back-history for %d assets', len(subIPOs))
                    modelDB.deleteProxyReturns(subIPOs, ipoDeleteDates)
                modelDB.deleteProxyReturns(cleanIssues, date)

            else:
                # Compute proxy returns for assets belonging to market
                if date in checkDates:
                    computeProxyReturns(rmg, date, rmgSubIssues, saveIDs,
                            modelDB, marketDB, options_.debuggingReporting,
                            forexCache, daysBack, options_,
                            backFilledIssues=linkedIssues)
                else:
                    logging.info('%s not a valid date. Skipping %s asset proxy',
                            date, rmg.description)
                 
        if not options_.dontWrite:
            if options_.testOnly:
                logging.info('Reverting changes')
                modelDB.revertChanges()
            else:
                modelDB.commitChanges()
        logging.info('Finished returns proxy processing for %s', date)
    modelDB.finalize()

if __name__ == '__main__':
    main()
