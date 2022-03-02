import datetime
import logging
try:
    import numpy.ma as ma
except:
    import numpy.core.ma as ma
import numpy
import optparse
import sys
import scipy.cluster.hierarchy as cluster
from marketdb import MarketDB
from riskmodels import Matrices
from riskmodels import ModelDB
from riskmodels import Classification
from riskmodels import MarketIndex
from riskmodels import AssetProcessor
from riskmodels import Utilities
from riskmodels import EstimationUniverse
from riskmodels import Outliers

def generateSubIndustryReturns(date, dIdx, rm, modelDB, marketDB, indStruct):
    # Load model info for date
    logging.info('Loading model data for %s', date)
    outlierClass = Outliers.Outliers()

    # Load model universe, or, if not possible, all live IDs
    try:
        rmi = modelDB.getRiskModelInstance(rm.rms_id, date)
        universe = modelDB.getRiskModelInstanceUniverse(rmi)
    except:
        universe = modelDB.getActiveSubIssues(rm.rmg[0],date)
    logging.info('Loaded %d subissues for date: %s', len(universe), date)

    # Exclude undesirables
    buildEstu = EstimationUniverse.ConstructEstimationUniverse(universe, rm, modelDB, marketDB)
    # Remove assets from the exclusion table
    (estuIdx, nonest) = buildEstu.apply_exclusion_list(date)

    # Keep only common stocks and REITs
    data = Utilities.Struct()
    allowedTypes = AssetProcessor.commonStockTypes + AssetProcessor.otherAllowedStockTypes \
            + AssetProcessor.localChineseAssetTypes + AssetProcessor.intlChineseAssetTypes
    data.assetTypeDict = AssetProcessor.get_asset_info(date, universe, modelDB, marketDB,
            'ASSET TYPES', 'Axioma Asset Type')
    (estuIdx, nonest) = buildEstu.exclude_by_asset_type(date, data, includeFields=allowedTypes, baseEstu=estuIdx)

    # Process ID data
    universe = numpy.take(universe, estuIdx, axis=0)
    data = AssetProcessor.process_asset_information(
            date, universe, rm.rmg, modelDB, marketDB,
            checkHomeCountry=(rm.SCM==0),
            numeraire_id=rm.numeraire.currency_id,
            forceRun=rm.forceRun)
    universe = data.universe

    # Load market caps
    rootCaps = ma.filled(ma.sqrt(data.marketCaps), 0.0)

    # Load asset returns
    assetReturnMatrix = modelDB.loadTotalReturnsHistoryV3(
            rm.rmg, date, universe, 1, None)
    assetReturns = Utilities.screen_data(assetReturnMatrix.data[:,0])
    assetReturns = outlierClass.twodMAD(assetReturns)

    # Remove market return
    estu = universe
    estu_idx = list(range(len(universe)))
    weights = numpy.array(rootCaps)
    nAssets = len(estu)
    C = min(100, int(round(nAssets*0.05)))
    sortindex = ma.argsort(weights)
    ma.put(weights, sortindex[nAssets-C:nAssets],
            weights[sortindex[nAssets-C]])
    if len(assetReturns) > 0:
        marketReturn = ma.average(assetReturns, weights=weights)
        assetReturns = assetReturns - marketReturn
    logging.info('Total of %d eligible assets', len(estu))

    # Build exposure matrix
    rootCaps = numpy.take(rootCaps, estu_idx, axis=0)
    assetReturns = numpy.take(assetReturns, estu_idx, axis=0)
    exposures = indStruct.cls.getExposures(
            date, estu, indStruct.factorList, modelDB)
    Utilities.writeToCSV(exposures, 'tmp/%s-exp.csv' % indStruct.typ,
                    columnNames=estu, rowNames=indStruct.factorList)
    logging.info('Exposure matrix, %d by %d', exposures.shape[0], exposures.shape[1])

    # Get industry statistics
    indIdxMap = dict([(j,i) for (i,j) in enumerate(indStruct.factorList)])
    for ind in indStruct.factorList:
        sIdx = indIdxMap[ind]
        idxList = numpy.flatnonzero(ma.getmaskarray(exposures[sIdx])==0)
        if len(idxList) > 0:
            # Number of assets within industry
            indStruct.count[dIdx,sIdx] = len(idxList)
            subRootCap = numpy.take(rootCaps, idxList, axis=0)
            wgt = ma.sum(subRootCap, axis=None)
            indStruct.Weights[dIdx,sIdx] = wgt
            if wgt > 0.0:
                wgt = subRootCap / wgt
                indStruct.herf[dIdx,sIdx] = 1.0 / numpy.inner(wgt ,wgt)
                # Return of sub-industry
                subReturns = numpy.take(assetReturns, idxList, axis=0)
                indStruct.Returns[dIdx,sIdx] = numpy.average(subReturns, weights=subRootCap)
     
    # Make note of IDs missing industry
    missingClass = numpy.flatnonzero(ma.getmaskarray(
                        ma.sum(exposures, axis=0)))
    missingClassIds = numpy.take(estu, missingClass, axis=0)
    missingClassIds = [s.getSubIDString() for s in missingClassIds]
    for sid in missingClassIds:
        if sid in indStruct.missingClass:
            indStruct.missingClass[sid].append(date)
        else:
            indStruct.missingClass[sid] = [date]
    return indStruct

def runLoop(riskModel, dates, modelDB, marketDB, options):

    # Load industry classification
    if options.indType == 'S':
        cls = Classification.GICSSectors(datetime.date(2018,9,29))
        factorList = [f.description for f in cls.getLeafNodes(modelDB).values()]
    elif options.indType == 'IG':
        cls = Classification.GICSIndustryGroups(datetime.date(2018,9,29))
        factorList = [f.description for f in cls.getLeafNodes(modelDB).values()]
    elif options.indType == 'SI':
        # Note this may not work
        cls = Classification.GICSCustomSubIndustries(datetime.date(2018,9,29))
        factorList = [f.description for f in cls.getLeafNodes(modelDB).values()]
        factorIds = sorted(cls.codeToIndMap.keys())
        factorList = [cls.codeToIndMap[k] for k in factorIds if cls.codeToIndMap[k] in factorList]
    else:
        cls = Classification.GICSIndustries(datetime.date(2018,9,29))
        factorList = [f.description for f in cls.getLeafNodes(modelDB).values()]
    logging.info('Loaded %d industries (level %s)', len(factorList), options.indType)

    # Initialise results structure
    indStruct = Utilities.Struct()
    indStruct.typ = options.indType
    indStruct.herf = numpy.zeros((len(dates),len(factorList)), float)
    indStruct.count = numpy.zeros((len(dates),len(factorList)), float)
    indStruct.Returns = numpy.zeros((len(dates),len(factorList)), float)
    indStruct.Weights = numpy.zeros((len(dates),len(factorList)), float)

    # Map various levels from sectors down
    sectorSubIndMap = {}
    sectorIGMap = {}
    industryGroupIndustryMap = {}
    industrySubIndMap = {}
    cls2 = Classification.GICSIndustries(datetime.date(2018,9,29))
    root = [r for r in cls2.getClassificationRoots(modelDB) if r.name == 'Sectors']
    sectors = cls2.getClassificationChildren(root[0], modelDB)
    for sector in sectors:
        subInds = []
        # Industry group level
        indGroups = cls2.getClassificationChildren(sector, modelDB)
        sectorIGMap[sector] = indGroups
        # Industry level
        for indGroup in indGroups:
            industries = cls2.getClassificationChildren(indGroup, modelDB)
            industryGroupIndustryMap[indGroup] = industries
            # Sub-industry level
            for industry in industries:
                subIndustries = cls2.getClassificationChildren(industry, modelDB)
                #logging.info('SUBIND: %s', subIndustries)
                subInds.extend(subIndustries)
        sectorSubIndMap[sector] = subInds

    indStruct.cls = cls
    indStruct.factorList = factorList
    indStruct.sectorSubIndMap = sectorSubIndMap
    indStruct.sectorIGMap = sectorIGMap
    indStruct.industryGroupIndustryMap = industryGroupIndustryMap
    indStruct.industrySubIndMap = industrySubIndMap
    indStruct.missingClass = dict()

    # Loop round dates and collect data
    status = 0
    for (idx, d) in enumerate(dates):
        try:
            riskModel.setFactorsForDate(d, modelDB)
            generateSubIndustryReturns(d, idx, riskModel, modelDB, marketDB, indStruct)
            logging.info('Finished %s processing for %s', options.modelName, d)
        except Exception as ex:
            logging.error('Exception caught during processing', exc_info=True)
            modelDB.revertChanges()
            status = 1
            break
    
    # Trim missing industry information to meaningful level
    for sid in indStruct.missingClass.keys():
        dateList = sorted(indStruct.missingClass[sid])
        indStruct.missingClass[sid] = [dateList[0], dateList[-1], len(dateList)]
     
    # Aggregate data to weekly level
    dateDict = dict([(d,i) for (i,d) in enumerate(dates)])
    periodDateList = [nxt for (nxt, prev) in \
            zip(dates[1:], dates[:-1])
            if nxt.weekday() < prev.weekday()]
    dateLen = len(periodDateList)-1
    weeklyReturns = Matrices.allMasked((
        dateLen, indStruct.Returns.shape[1]), float)
    weeklyCount = Matrices.allMasked((
        dateLen, indStruct.Returns.shape[1]), float)
    weeklyHerf = Matrices.allMasked((
        dateLen, indStruct.Returns.shape[1]), float)
    weeklyWeight = Matrices.allMasked((
        dateLen, indStruct.Returns.shape[1]), float)
    i0 = -1
    dailyReturns = indStruct.Returns + 1.0
    for (j,d) in enumerate(periodDateList):
        i1 = dateDict[d]
        if i0 >= 0:
            tmp = numpy.product(dailyReturns[i0:i1,:], axis=0)
            weeklyReturns[j-1,:] = tmp - 1.0
            weeklyCount[j-1,:] = numpy.average(\
                    indStruct.count[i0:i1,:], axis=0)
            weeklyHerf[j-1,:] = numpy.average(\
                    indStruct.herf[i0:i1,:], axis=0)
            weeklyWeight[j-1,:] = numpy.average(\
                    indStruct.Weights[i0:i1,:], axis=0)
        i0 = i1
    dates = periodDateList[:-1]

    # Compute correlations over time
    correl = Utilities.compute_covariance(ma.filled(weeklyReturns, 0.0), corrOnly=True)

    # Save the raw data
    dateList = [str(d) for d in dates]
    mnem = riskModel.mnemonic
    yr = dates[-1].year
    outFile = 'tmp/%s-%s-Corrs-%d.csv' % (mnem, options.indType, yr)
    Utilities.writeToCSV(correl, outFile, columnNames=indStruct.factorList, rowNames=indStruct.factorList)
    outFile = 'tmp/%s-%s-Rets-%d.csv' % (mnem, options.indType, yr)
    Utilities.writeToCSV(weeklyReturns, outFile, columnNames=indStruct.factorList, rowNames=dateList)
    outFile = 'tmp/%s-%s-Count-%d.csv' % (mnem, options.indType, yr)
    Utilities.writeToCSV(weeklyCount, outFile, columnNames=indStruct.factorList, rowNames=dateList)
    outFile = 'tmp/%s-%s-Herf-%d.csv' % (mnem, options.indType, yr)
    Utilities.writeToCSV(weeklyHerf, outFile, columnNames=indStruct.factorList, rowNames=dateList)
    outFile = 'tmp/%s-%s-Weight-%d.csv' % (mnem, options.indType, yr)
    Utilities.writeToCSV(weeklyWeight, outFile, columnNames=indStruct.factorList, rowNames=dateList)
    outfile = open('tmp/%s-%s-Missing-%d.csv' % (mnem, options.indType, yr), 'w')
    outfile.write('SID,FIRST DATE,LAST DATE,N\n')
    for sid in indStruct.missingClass.keys():
        row = indStruct.missingClass[sid]
        outfile.write('%s,%s,%s,%d\n' % (sid, row[0], row[1], row[2]))
    outfile.close()
    return status

if __name__ == '__main__':
    usage = "usage: %prog [options] <YYYY-MM-DD> [<YYYY-MM-DD>]"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addModelAndDefaultCommandLine(cmdlineParser)
    
    # Typical model generation steps
    cmdlineParser.add_option("--type", action="store",
                             default='I', dest="indType",
                             help="I=Industry,SI=Subindustry,IG=IndustryGroup,S=Sectors")
    # Other options
    cmdlineParser.add_option("-n", action="store_true",
                             default=False, dest="testOnly",
                             help="don't change the database")
    cmdlineParser.add_option("--verbose", "-v", action="store_true",
                             default=False, dest="verbose",
                             help="perform a lot of debugging diagnostics")
    cmdlineParser.add_option("--force", "-f", action="store_true",
                             default=False, dest="force",
                             help="override certain constraints")
    
    (options, args) = cmdlineParser.parse_args()
    if len(args) < 1 or len(args) > 2:
        cmdlineParser.error("Incorrect number of arguments")

    # Process model and DB stuff
    riskModelClass = Utilities.processModelAndDefaultCommandLine(
        options, cmdlineParser)
    modelDB = ModelDB.ModelDB(sid=options.modelDBSID, 
                              user=options.modelDBUser,
                              passwd=options.modelDBPasswd)
    marketDB = MarketDB.MarketDB(sid=options.marketDBSID, 
                                 user=options.marketDBUser,
                                 passwd=options.marketDBPasswd)
    riskModel = riskModelClass(modelDB, marketDB)

    # Process dates
    startDate = Utilities.parseISODate(args[0])
    if len(args) == 1:
        endDate = startDate
    else:
        endDate = Utilities.parseISODate(args[1])
    dates = modelDB.getDateRange(riskModel.rmg, startDate, endDate, 
                                 excludeWeekend=True)

    # Set some more options
    if options.verbose:
        riskModel.debuggingReporting = True
    if options.force:
        riskModel.forceRun = True
    if len(riskModel.rmg) > 1:
        riskModel.SCM = False
    else:
        riskModel.SCM = True
    
    # Run the thing
    status = runLoop(riskModel, dates, modelDB, marketDB, options)
    marketDB.finalize()
    modelDB.finalize()
    sys.exit(status)
