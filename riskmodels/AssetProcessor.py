import logging
import numpy.ma as ma
import numpy.linalg as linalg
import numpy
import scipy.stats as stats
import copy
import pandas
import datetime
from collections import defaultdict
from itertools import chain
from riskmodels import Matrices
from riskmodels.Matrices import ExposureMatrix
from riskmodels import Utilities

# Asset type meta-groups
drAssetTypes = ['NVDR', 'GlobalDR', 'TDR', 'AmerDR', 'FStock', 'CDI', 'DR']
commonStockTypes = ['Com', 'ComClsA', 'ComClsB', 'ComClsC', 'ComClsD', 'ComClsE', 'ComClsL']
otherAllowedStockTypes = ['REIT', 'StapSec']
preferredStockTypes = ['Pref', 'PrefClsA', 'PrefClsB', 'PrefClsC', 'PrefClsD', 'PrefClsE']
fundAssetTypes = ['CEFund', 'InvT', 'Misc', 'UnitT']
localChineseAssetTypes = ['AShares', 'BShares']
intlChineseAssetTypes = ['HShares', 'RCPlus']
etfAssetTypes = ['NonEqETF', 'ComETF', 'StatETF', 'ETFnoRM']
otherAssetTypes = ['LLC', 'LP', 'ComWI', 'UnCls', 'CFCont-SS', 'ComSPAC']
noTransferList = etfAssetTypes + ['EIFCont-SS', 'CFCont-SS', 'MF', 'MACSPAC']
noHBetaList = [None, 'MF', 'MACSPAC']
allAssetTypes = drAssetTypes + commonStockTypes + otherAllowedStockTypes + preferredStockTypes + fundAssetTypes +\
        localChineseAssetTypes + intlChineseAssetTypes + etfAssetTypes + otherAssetTypes + noTransferList
allAssetTypes = list(set(allAssetTypes))

# Exchange meta-groups
connectExchanges = ['HKC','HKS','SHC','SZC']

# Linked companies (for the purposes of ISC)
extraLinkedList = []
#extraLinkedList.append(['CIBLPQH42', 'CILZJ9V64']) # FNF/FAF
#extraLinkedList.append(['CISACFVZ1', 'CI9YXN3V0']) # MFA/AGNC
#extraLinkedList.append(['CITU9FWV8', 'CI7V9JVV3']) # ETFC/AMTD
#extraLinkedList.append(['CIAURG827', 'CIU8X83S3']) # XEL/LNT
#extraLinkedList.append(['CIJ81QSY1', 'CIC4ZA5X2']) # VST/NRG
#extraLinkedList.append(['CIYHYPSX2', 'CIULXYF39']) # WEC/CMS
#extraLinkedList.append(['CIDNTYC73', 'CID1AM625']) # Tencent/Naspers
#extraLinkedList.append(['CIL531U65', 'CIJWFS854']) # Jardine M/Jardine S
#extraLinkedList.append(['CIX6SY9G3', 'CIJFGS1S7']) # Lowes/Home Depot

def loadRiskModelGroupAssetMap(modelDate, base_universe, rmgList, modelDB, marketDB, addNonModelRMGs, quiet=False):
    """Returns a dict mapping risk model group IDs to a set
    of assets trading in that country/market.  base_universe is the
    list of assets that will be considered for processing.
    If addNonModelRMGs is True then the risk model groups for
    assets trading outside the model are added to the map.
    Otherwise those assets won't be mapped.
    """
    # XXX - Todo: something similar for currencies
    logging.debug('loadRiskModelGroupAssetMap: begin')
    rmgAssetMap = dict()
    base_universe = set(base_universe)
    runningTotal = 0

    for rmg in rmgList:
        rmg_assets = set(modelDB.getActiveSubIssues(rmg, modelDate))
        rmg_assets = rmg_assets.intersection(base_universe)
        rmgAssetMap[rmg.rmg_id] = rmg_assets
        runningTotal += len(rmg_assets)
        if len(rmg_assets)==0:
            if not quiet:
                logging.warning('No assets in %s', rmg.description)
        else:
            logging.debug('%d assets in %s (RiskModelGroup %d, %s)',
                    len(rmg_assets), rmg.description, rmg.rmg_id, rmg.mnemonic)
    logging.debug('%d assets in %d RMGs', runningTotal, len(rmgList))

    nRMGAssets = sum([len(assets) for (rmg, assets) in rmgAssetMap.items()])
    if len(base_universe) != nRMGAssets:
        if not quiet:
            logging.warning('%d assets not assigned to a model RMG', len(base_universe)-nRMGAssets)
        if addNonModelRMGs:
            missing = set(base_universe)
            for assets in rmgAssetMap.values():
                missing -= assets
            sidRMGMap = dict(modelDB.getSubIssueRiskModelGroupPairs(modelDate, missing))
            for sid in missing:
                rmg_id = sidRMGMap[sid].rmg_id
                if rmg_id not in rmgAssetMap:
                    rmg = modelDB.getRiskModelGroup(rmg_id)
                    logging.debug('Adding %s (%s) to rmgAssetMap', rmg.description, rmg.mnemonic)
                rmgAssetMap.setdefault(rmg_id, set()).add(sid)
    logging.debug('loadRiskModelGroupAssetMap: end')
    return rmgAssetMap

def get_asset_info(date, universe, modelDB, marketDB, clsFamily, clsMemberStr):
    """Gets axioma asset classification from the DB
    Returns a dict of subids to asset types
    """
    clsFamily = marketDB.getClassificationFamily(clsFamily)
    assert(clsFamily is not None)
    clsMembers = dict([(i.name, i) for i in marketDB.\
            getClassificationFamilyMembers(clsFamily)])
    clsMember = clsMembers.get(clsMemberStr, None)
    assert(clsMember is not None)
    clsRevision = marketDB.\
            getClassificationMemberRevision(clsMember, date)
    clsData = modelDB.getMktAssetClassifications(
            clsRevision, universe, date, marketDB)
    retDict = {sid: (item.classification.code if item.classification is not None else None) for sid, item in clsData.items()}
    return retDict

def get_home_country(universe, date, modelDB, marketDB, clsType='HomeCountry'):
    # Load home country classification data
    clsFamily = marketDB.getClassificationFamily('REGIONS')
    assert(clsFamily is not None)
    clsMembers = dict([(i.name, i) for i in marketDB.getClassificationFamilyMembers(clsFamily)])
    clsMember = clsMembers.get(clsType, None)
    assert(clsMember is not None)
    clsRevision = marketDB.getClassificationMemberRevision(clsMember, date)
    return modelDB.getMktAssetClassifications(clsRevision, universe, date, marketDB)

def dumpAssetListForDebugging(fileName, assetList, dataColumn=None):
    assetIdxMap = dict(zip(assetList, list(range(len(assetList)))))
    assetList.sort()
    outFile = open(fileName, 'w')
    for sid in assetList:
        if hasattr(sid, 'getSubIdString'):
            outFile.write('%s,' % sid.getSubIdString())
        else:
            outFile.write('%s,' % sid)
        if dataColumn is not None:
            outFile.write('%s,' % dataColumn[assetIdxMap[sid]])
        outFile.write('\n')
    outFile.close()

def getModelAssetMaster(rm, date, modelDB, marketDB, legacyDates=True):
    """Determine the universe for the current model instance.
    The list of asset IDs is sorted.
    """
    logging.info('Risk model numeraire is %s (ID %d)',
                rm.numeraire.currency_code, rm.numeraire.currency_id)

    # Get all securities marked for this model at the current date
    assets = modelDB.getMetaEntity(marketDB, date, rm.rms_id, rm.primaryIDList)
    universe_list = list(set(assets))
    logging.info('Initial universe list: %d assets', len(universe_list))

    # Get asset types
    assetTypeDict = get_asset_info(date, universe_list, modelDB, marketDB,
            'ASSET TYPES', 'Axioma Asset Type')
    marketTypeDict = get_asset_info(date, universe_list, modelDB, marketDB,
            'REGIONS', 'Market')

    if rm.debuggingReporting:
        typeList = [assetTypeDict.get(sid,None) for sid in universe_list]
        dumpAssetListForDebugging('tmp/AssetTypeMap-%s.csv' % rm.mnemonic,
                universe_list, dataColumn=typeList)
        typeList = list(set(typeList))
        dumpAssetListForDebugging('tmp/AssetTypes-%s.csv' % rm.mnemonic, typeList)
        typeList = [marketTypeDict.get(sid,None) for sid in universe_list]
        dumpAssetListForDebugging('tmp/MarketTypeMap-%s.csv' % rm.mnemonic,
                universe_list, dataColumn=typeList)

    # Load market caps, convert to model numeraire
    if legacyDates:
        mcapDates = modelDB.getDates(rm.rmg, date, 20, excludeWeekend=True, fitNum=True)
        marketCaps = modelDB.getAverageMarketCaps(mcapDates, universe_list, rm.numeraire.currency_id, marketDB)
    else:
        marketCaps, mcapDates = robustLoadMCaps(
                date, universe_list, rm.numeraire.currency_id, modelDB, marketDB)

    # Remove assets with missing market cap
    missingCapIdx = numpy.flatnonzero(ma.getmaskarray(marketCaps))
    if len(missingCapIdx) > 0:
        missingCap = [universe_list[i] for i in missingCapIdx]
        logging.warning('%d assets dropped due to missing avg %d-day cap',
                len(missingCap), len(mcapDates))
        new_universe = list(set(universe_list).difference(set(missingCap)))
        if rm.debuggingReporting:
            dumpAssetListForDebugging('tmp/MissingCap.csv', missingCap)
        missingCap = [sid.getSubIDString() for sid in missingCap]
        logging.info('Dropped assets: %s', ','.join(missingCap))
    else:
        new_universe = list(universe_list)

    # Remove assets with missing asset type
    missingAT = [sid for sid in new_universe if sid not in assetTypeDict]
    if len(missingAT) > 0:
        new_universe = list(set(new_universe).difference(set(missingAT)))
        logging.warning('%d assets dropped due to missing asset Type', len(missingAT))
        if rm.debuggingReporting:
            dumpAssetListForDebugging('tmp/MissingAssetType-%s.csv' % date, missingAT)

    # Identify possible ETFs
    etf = [sid for sid in new_universe if assetTypeDict.get(sid,None) in ['NonEqETF', 'StatETF']]
    if len(etf) > 0:
        if rm.allowETFs:
            logging.info('Allowing %d non-equity asset ETFs into universe', len(etf))
        else:
            logging.info('Excluding %d non-equity asset ETFs from universe', len(etf))
        if rm.debuggingReporting:
            dumpAssetListForDebugging('tmp/NonEqETF.csv', etf)
    etf2 = [sid for sid in new_universe if assetTypeDict.get(sid,None)=='ComETF']
    if len(etf2) > 0:
        if rm.allowETFs:
            logging.info('Allowing %d composite ETFs into universe', len(etf2))
        else:
            logging.info('Excluding %d composite ETFs from universe', len(etf))
        if rm.debuggingReporting:
            dumpAssetListForDebugging('tmp/ComETF.csv', etf2)
    etf = set(etf).union(set(etf2))

    # Remove assets w/o industry membership (excluding ETFs)
    if rm.industryClassification is not None:
        assert(len(list(rm.industryClassification.getLeafNodes(modelDB).values())) > 0)
        leaves = rm.industryClassification.getLeafNodes(modelDB)
        factorList = [i.description for i in leaves.values()]
        exposures = rm.industryClassification.getExposures(date, new_universe, factorList, modelDB)
        indSum = ma.sum(exposures, axis=0).filled(0.0)
        nonMissingInd = numpy.flatnonzero(indSum)
        new_universe2 = [new_universe[i] for i in nonMissingInd]

        # Make special case for ETFs in stat models
        if rm.allowETFs:
            new_universe2 = list(set(new_universe2).union(etf))

        missingInd = set(new_universe).difference(set(new_universe2))
        if len(missingInd) > 0:
            logging.warning('%d assets dropped due to missing %s classification',
                    len(missingInd), rm.industryClassification.name)
            if rm.debuggingReporting:
                dumpAssetListForDebugging('tmp/missingInd.csv', list(missingInd))
    else:
        new_universe2 = list(new_universe)

    if len(new_universe2) > 0:
        logging.info('%d assets in the universe', len(new_universe2))
        return sorted(new_universe2)
    else:
        raise Exception('No assets in the universe!')

def robustLoadMCaps(date, subIssues, currency_id, modelDB, marketDB, freefloat_flag=False, days_avg=30):
    # Load initial set of mcaps
    mcapDates = modelDB.getAllRMGDateRange(date, days_avg)
    marketCaps = modelDB.getAverageMarketCaps(mcapDates, subIssues, currency_id, marketDB, loadAllDates=True)
    
    # Check for assets with missing market cap
    # Look back an extra month 
    missingCapIdx = numpy.flatnonzero(ma.getmaskarray(marketCaps))
    if len(missingCapIdx) > 0:
        missingCap = [subIssues[i] for i in missingCapIdx]
        logging.warning('%d assets have missing avg %d-day cap', len(missingCap), len(mcapDates))
        logging.warning('Looking back further...')
        extraDates = modelDB.getAllRMGDateRange(date-datetime.timedelta(days_avg), days_avg)
        extraCaps = modelDB.getAverageMarketCaps(
                extraDates, missingCap, currency_id, marketDB, loadAllDates=True)
        ma.put(marketCaps, missingCapIdx, extraCaps)
    if freefloat_flag:
        # considering free floating adjustment for marketcap
        marketCaps = modelDB.applyFreeFloatOnMCap(marketCaps, mcapDates, subIssues, marketDB)
    return marketCaps, mcapDates

def process_asset_information(date, universe, rmgList, modelDB, marketDB,
        checkHomeCountry=True, legacyDates=True, numeraire_id=1,
        forceRun=False, nurseryRMGList=[], includeByModel=None, rmgOverride=dict(),
        trackList=None, quiet=False):
    """Outer Loop to process asset data
    """
    stepNo = 1
    (data, missing) = process_asset_information_inner(
            date, universe, rmgList, modelDB, marketDB,
            checkHomeCountry, legacyDates, numeraire_id, forceRun,
            nurseryRMGList, includeByModel, stepNo, rmgOverride, trackList, quiet)

    # If anything classed as missing, reset everything
    while len(missing) > 0:
        stepNo += 1
        universe = [sid for sid in data.universe if sid not in missing]
        (data, missing) = process_asset_information_inner(date, universe, rmgList, modelDB, marketDB,
                checkHomeCountry, legacyDates, numeraire_id, forceRun,
                nurseryRMGList, includeByModel, stepNo, rmgOverride, trackList, quiet)

    # Load average market caps
    if legacyDates:
        mcapDates = modelDB.getDates(rmgList, date, 20, excludeWeekend=True, fitNum=True)
        data.marketCaps = modelDB.getAverageMarketCaps(mcapDates, data.universe, numeraire_id, marketDB, loadAllDates=True)
    else:
        data.marketCaps, mcapDates = robustLoadMCaps(date, data.universe, numeraire_id, modelDB, marketDB)
    data.freeFloat_marketCaps, tmp_mcapDates = robustLoadMCaps(\
            date, data.universe, numeraire_id, modelDB, marketDB, freefloat_flag=True)

    # Report on any zero/missing mcaps (there really shouldn't be any)
    missingIdx = numpy.flatnonzero(ma.getmaskarray(ma.masked_where(data.marketCaps==0.0, data.marketCaps)))
    if len(missingIdx) > 0:
        missingAssets = [data.universe[idx] for idx in missingIdx]
        missingAssets = [sid for sid in missingAssets if not sid.isCashAsset()]
        if len(missingAssets) > 0:
            logging.error('Missing mcap for %s', ','.join([sid.getSubIdString() for sid in missingAssets]))
    if not forceRun:
        assert(numpy.sum(ma.getmaskarray(data.marketCaps))==0)
        
    # Downweight some countries if required
    for r in [r for r in rmgList if (hasattr(r, 'downWeight') and (r.downWeight < 1.0))]:
        logging.info('Downweighting %s market caps by factor of %.2f%%', r.mnemonic, r.downWeight*100)
        for sid in data.rmgAssetMap[r.rmg_id]:
            if sid in data.assetIdxMap:
                data.marketCaps[data.assetIdxMap[sid]] *= r.downWeight

    # Pull out all assets with valid mcaps (for total mcap)
    allSidCompanyMap = modelDB.getCompanySubIssues(date, set(data.cid2sidMap.keys()), marketDB)
    okSids = set([sid for (sid, vals) in allSidCompanyMap.items() if not vals[1]])
    mCaps = pandas.Series(data.marketCaps, index=data.universe)

    # Find market cap per trading RMG
    rmgCap = dict()
    totalMCap = 0.0
    for rmg in rmgList:
        overlap = okSids.intersection(set(data.tradingRmgAssetMap[rmg.rmg_id]))
        mktCap = mCaps[overlap].sum(axis=None)
        if hasattr(rmg, 'mktCapCap') and (rmg.mktCapCap != 1.0):
            rmgCap[rmg] = mktCap
        else:
            totalMCap += mktCap

    # Downweight anything on the "naughty list" - use trading RMG rather than assigned home market
    data.mktCapDownWeight = dict()
    for rmg in rmgCap.keys():
        ratio = rmgCap[rmg] / totalMCap
        if ratio > rmg.mktCapCap:
            mktCapDownWeight = totalMCap * rmg.mktCapCap / rmgCap[rmg]
            data.mktCapDownWeight[rmg] = (1.0 - rmg.blendFactor) + (rmg.blendFactor * mktCapDownWeight)
            logging.info('RMG %s has fraction %.6f of total market cap against target of %.6f',
                            rmg.mnemonic, ratio, rmg.mktCapCap)
            logging.info('Scaling by factor of %.4f', data.mktCapDownWeight[rmg])
            for sid in data.tradingRmgAssetMap[rmg.rmg_id]:
                if sid in data.assetIdxMap:
                    data.marketCaps[data.assetIdxMap[sid]] *= data.mktCapDownWeight[rmg]

    # Get a mapping from asset to currency
    data.assetCurrencyMap = dict()
    for rmg in rmgList:
        isoCode = rmg.getCurrencyCode(date)
        for sid in data.rmgAssetMap[rmg.rmg_id]:
            data.assetCurrencyMap[sid] = isoCode

    # Construct mapping from currency to assets
    data.currencyAssetMap = dict()
    for sid in data.universe:
        if sid in data.assetCurrencyMap:
            cur = data.assetCurrencyMap[sid]
            if cur in data.currencyAssetMap:
                data.currencyAssetMap[cur].append(sid)
            else:
                data.currencyAssetMap[cur] = [sid]
        else:
            logging.warning('Subissue %s (RMG: %s) has no currency',
                    sid.getSubIDString(), data.assetRMGMap[sid])

    data.marketCaps = numpy.array(ma.filled(data.marketCaps, 0.0))
    logging.info('Universe total mcap: %.4f tn, Assets: %d',
            ma.sum(data.marketCaps, axis=None) / 1.0e12, len(data.universe))

    # Get subissue to ISIN and name maps
    mdIDs = [sid.getModelID() for sid in data.universe]
    mdIDtoISINMap = modelDB.getIssueISINs(date, mdIDs, marketDB)
    mdIDtoSEDOLMap = modelDB.getIssueSEDOLs(date, mdIDs, marketDB)
    mdIDtoCUSIPMap = modelDB.getIssueCUSIPs(date, mdIDs, marketDB)
    mdIDtoTickerMap = modelDB.getIssueTickers(date, mdIDs, marketDB)
    mdIDtoNameMap = modelDB.getIssueNames(date, mdIDs, marketDB)
    data.assetISINMap = dict()
    data.assetSEDOLMap = dict()
    data.assetCUSIPMap = dict()
    data.assetTickerMap = dict()
    data.assetNameMap = dict()
    for (mid, sid) in zip(mdIDs, data.universe):
        if mid in mdIDtoISINMap:
            data.assetISINMap[sid] = mdIDtoISINMap[mid]
        if mid in mdIDtoSEDOLMap:
            data.assetSEDOLMap[sid] = mdIDtoSEDOLMap[mid]
        if mid in mdIDtoCUSIPMap:
            data.assetCUSIPMap[sid] = mdIDtoCUSIPMap[mid]
        if mid in mdIDtoTickerMap:
            data.assetTickerMap[sid] = mdIDtoTickerMap[mid]
        if mid in mdIDtoNameMap:
            data.assetNameMap[sid] = mdIDtoNameMap[mid]

    data.assetData_DF = get_asset_DF(data)
    return data

def flip_dict(in_dict):
    '''
        flip the dictionary. for dict(itemA,list of itemB)
    :param in_dict: dict(itemA,list of itemB)
    :return: dict(itemB,itemA)
    '''
    out_dict = {y: x for x, z in in_dict.items() for y in z}
    return out_dict

def get_asset_DF(data):
    '''
        convert data(generated from process_asset_information) into dataframe
    :param data: 
    :return: 
    '''
    # ['hardCloneMap', #dict(sub_id,sub_id)
    #  'homeCtyRMG_Original_dict', # dict(sub_id,str)
    #  'nurseryUniverse', # a list of sub_ids
    #  'assetTradingCurrencyMap',# dict(sub_id,str)
    #  'subIssueGroups', # dict(company_id,list of sub_id)  (to ignore, redundant???)
    #  'assetCUSIPMap',  # dict(sub_id,str)
    #  'assetNameMap',   # dict(sub_id,str)
    #  'currencyAssetMap', # dict(currency,list of sub_id) (to ignore, redundant???)
    #  'assetTickerMap',   # dict(sub_id,str)
    #  'universe',         # list of sub_id, can be ignored
    #  'rmgAssetMap',      # dict(rmg_id,list of sub_ids) ???
    #  'assetSEDOLMap',    # dict(sub_id,str)
    #  'marketTypeDict',   # dict(sub_id,str)
    #  'rmcAssetMap',      # dict(rmc_id,list of sub_ids) ???
    #  'drCurrData',       # dict(sub_id,str)
    #  'assetISINMap',     # dict(sub_id,str)
    #  'assetData_DF',     # can be ignored
    #  'assetTypeDict',    # dict(sub_id,str)
    #  'originalNurseryUniverse', # a list of sub ids
    #  'forceCointMap',    # empty ???
    #  'foreign',          # a list of sub ids
    #  'sidToCIDMap',      # dict(sub_id,company_id)
    #  'assetCurrencyMap', # dict(sub_id,str)
    #  'marketCaps',       # np.array
    #  'noCointMap',       # empty ???
    #  'dr2Underlying',    # dict(sub_id,sub_id)
    #  'assetIdxMap',      # dict(sub_id,number)
    #  'tradingRmgAssetMap'] # dict(rmg_id,list of sub_id) ???
    nurseryUniverse_dict = dict(zip(data.nurseryUniverse, [True]*len(data.nurseryUniverse)))
    originalNurseryUniverse_dict = dict(zip(data.originalNurseryUniverse, [True]*len(data.originalNurseryUniverse)))
    foreign_dict = dict(zip(data.foreign, [True]*len(data.foreign)))
    marketCap_dict = dict(zip(data.universe, data.marketCaps))

    asset_rmg_map_dict = flip_dict(data.rmgAssetMap)
    asset_rmc_map_dict = flip_dict(data.rmcAssetMap)
    asset_tradingRmg_map_dict = flip_dict(data.tradingRmgAssetMap)

    hardCloneMap_copy = dict((x,y.getSubIDString()) if y is not None else (x,y) for x,y in data.hardCloneMap.items())
    dr2Underlying_copy = dict((x,y.getSubIDString()) if y is not None else (x,y) for x,y in data.dr2Underlying.items())

    bigDic = {  'hard_Clone': hardCloneMap_copy,
                'Orig_HomeCountry': data.homeCtyRMG_Original_dict,
                'nursery_universe': nurseryUniverse_dict,
                'trading_Currency': data.assetTradingCurrencyMap,
                'CUSIP': data.assetCUSIPMap,
                'SEDOL': data.assetSEDOLMap,
                'ISIN':  data.assetISINMap,
                'Name': data.assetNameMap,
                'Ticker': data.assetTickerMap,
                'Rmg': asset_rmg_map_dict,
                'Rmc': asset_rmc_map_dict,
                'tradingRmg': asset_tradingRmg_map_dict,
                'market_Type': data.marketTypeDict,
                'asset_Type': data.assetTypeDict,
                'DR_Currency': data.drCurrData,
                'Orig_NurseryUniverse': originalNurseryUniverse_dict,
                'Foreign': foreign_dict,
                'sidToCIDMap': data.sidToCIDMap,
                'asset_Currency': data.assetCurrencyMap,
                'marketCap': marketCap_dict,
                'dr2Underlying': dr2Underlying_copy,
                'asset_Idx': data.assetIdxMap
                }
    output = pandas.DataFrame.from_dict(bigDic)
    output.index = [x.getSubIDString() for x in output.index]
    return output

def process_asset_information_inner(date, universe, rmgList, modelDB, marketDB,
                checkHomeCountry, legacyDates, numeraire_id,
                forceRun, nurseryRMGList, includeByModel, stepNo, rmgOverride,
                trackList, quiet):
    """Creates map of RiskModelGroup ID to list of its SubIssues,
    and checks for assets belonging to non-model geographies.
    Assets will also be checked for home country classification,
    and removed from the intial list of SubIssues (univ) and mcaps
    if no valid classification is found.
    """
    logging.debug('process_asset_information: begin')
    logging.info('Iteration %d: Initial model universe: %d assets', stepNo, len(universe))

    # Initialise
    debugOutput = False
    if trackList is None:
        trackList = []

    if includeByModel is not None:
        # If we want to restrict coverage to those assets in certain model types,
        # we filter here
        fundamentalModelIds = modelDB.getRMSIDsByModelType(modelType=includeByModel)
        validModelIDs = modelDB.getActiveIssuesInModelList(date, rmsList=fundamentalModelIds)
        logging.info('%d valid sub-issues across %d fundamental models', len(validModelIDs), len(fundamentalModelIds))
        univModelIDs = [s.getModelID() for s in universe]
        univSubIDMap = dict(zip(univModelIDs, universe))

        # find assets common to existing universe and those in a valid model
        dumped = set(univModelIDs).difference(set(validModelIDs))
        univModelIDs = set(univModelIDs).intersection(set(validModelIDs))
        old_len = len(universe)
        universe = sorted(univSubIDMap[m_id] for m_id in univModelIDs)
        if len(universe) < old_len:
            diff = old_len-len(universe)
            logging.info('Excluding %d subissues not in any fundamental model', diff)
            logging.info('New universe is now %d assets', len(universe))

        if debugOutput:
            dumped = [univSubIDMap[m_id] for m_id in dumped]
            dmpTypeDict = get_asset_info(date, dumped, modelDB, marketDB, 'ASSET TYPES', 'Axioma Asset Type')
            typeList = [dmpTypeDict.get(sid, None) for sid in dumped]
            sidList = [sid.getSubIDString() for sid in dumped]
            outName = 'tmp/dmpList-%s.csv' % date
            dumpAssetListForDebugging(outName, sidList, typeList)

    # Initialise return variables
    data = Utilities.Struct()
    data.universe = universe
    data.foreign = []
    data.drCurrData = None

    # Pick up dict of assets to be cloned from others
    data.hardCloneMap = modelDB.getClonedMap(date, data.universe, linkageType=1)
    data.forceCointMap = modelDB.getClonedMap(date, data.universe, linkageType=2)
    data.noCointMap = modelDB.getClonedMap(date, data.universe, linkageType=3)
    data.homeCtyRMG_Original_dict = dict() # original home country dictionary
    data.dr2Underlying = dict()

    # Get issuer map
    data.subIssueGroups = modelDB.getIssueCompanyGroups(date, data.universe, marketDB)
    data.cid2sidMap = modelDB.getIssueCompanyGroups(date, data.universe, marketDB, mapAllIssues=True)
    data.sidToCIDMap = modelDB.getIssueCompanies(date, data.universe, marketDB, keepUnmapped=True)

    # Get linked company map
    data.companyIDGroups = []
    # Create lists of linked companies (e.g. DLCs)
    linkedCIDMap = list(extraLinkedList)
    linkedCIDMap += modelDB.getDLCs(date, marketDB)

    # Sift out entities not in the model
    allCIDs = set(modelDB.getIssueCompanies(date, data.universe, marketDB).values())
    tmpMap = []
    for itm in linkedCIDMap:
        subList = [cid for cid in itm if cid in allCIDs]
        if len(subList) > 1:
            tmpMap.append(subList)

    # Combine sub-lists with common CIDs
    allCids = set(chain.from_iterable(tmpMap))
    for cid in allCids:
        components = [x for x in tmpMap if cid in x]
        for j in components:
            tmpMap.remove(j)
        tmpMap += [sorted(set(chain.from_iterable(components)))]
    data.companyIDGroups = tmpMap

    # Check assets' country of quotation
    data.assetIdxMap = dict(zip(data.universe, range(len(data.universe))))
    data.rmgAssetMap = loadRiskModelGroupAssetMap(
                    date, data.universe, rmgList, modelDB, marketDB, False, quiet=quiet)
    data.rmcAssetMap = copy.deepcopy(data.rmgAssetMap)
    data.assetRMGMap = dict([(sid, rmg_id) for (rmg_id, ids) in \
            data.rmgAssetMap.items() for sid in ids])
     
    # Remember the asset to trading-country map
    data.tradingRmgAssetMap = defaultdict(list)
    for (sid, rmg) in modelDB.getSubIssueRiskModelGroupPairs(date, restrict=list(data.universe)):
            data.tradingRmgAssetMap[rmg.rmg_id].append(sid)

    # Make a note of assets in "nursery" markets
    nurseryUniverse = []
    for rmg in nurseryRMGList:
        nurseryUniverse.extend(data.rmgAssetMap[rmg.rmg_id])
    if len(nurseryUniverse) > 0:
        logging.info('%d assets belong to nursery markets', len(nurseryUniverse))
    data.nurseryUniverse = nurseryUniverse
    data.originalNurseryUniverse = list(data.nurseryUniverse)

    # Get a dict of asset types
    data.assetTypeDict = get_asset_info(date, data.universe, modelDB, marketDB,
            'ASSET TYPES', 'Axioma Asset Type')
    data.marketTypeDict = get_asset_info(date, data.universe, modelDB, marketDB,
            'REGIONS', 'Market')

    # Get dict of asset ages
    issueFromDates = modelDB.loadIssueFromDates([date], data.universe)
    age = [int((date-dt).days) for dt in issueFromDates]
    data.ageDict = dict(zip(data.universe, age))

    # Get a mapping to trading currency
    data.assetTradingCurrencyMap = dict()
    fullTradingCurrencyMap = modelDB.getTradingCurrency(date, data.universe, marketDB)
    runningRMGList = []
    tradingCountryCurrencyMismatch = []
    for rmg in rmgList:
        isoCode = rmg.getCurrencyCode(date)
        runningRMGList.extend(data.rmgAssetMap[rmg.rmg_id])
        for sid in data.rmgAssetMap[rmg.rmg_id]:
            data.assetTradingCurrencyMap[sid] = fullTradingCurrencyMap.get(sid, isoCode)
            if data.assetTradingCurrencyMap[sid] != isoCode:
                tradingCountryCurrencyMismatch.append(sid)
    if len(tradingCountryCurrencyMismatch) > 0:
        logging.debug('%d assets trade in a different currency to their trading country',
                len(tradingCountryCurrencyMismatch))

    if not checkHomeCountry:
        missing = set(data.universe).difference(set(runningRMGList))
        return data, list(missing)

    # Get dict of DR to underlying assets
    # Set up the default mapping - every DR-type mapped to None, except Thai F-Stocks,
    # which are not true DRs
    drList = [sid for sid in data.universe if data.assetTypeDict.get(sid, None) in drAssetTypes]
    drList = [sid for sid in drList if data.assetTypeDict.get(sid, None) != 'FStock']
    data.dr2Underlying = dict((sid, None) for sid in drList)
    # Pull out the DR to underlying mapping from the DB and update the default mapping
    dr2Underlying = modelDB.getDRToUnderlying(date, data.universe, marketDB)
    notDR = [sid for sid in dr2Underlying.keys() if data.assetTypeDict.get(sid, None) not in drAssetTypes]
    if len(notDR) > 0:
        # Remove any non-DRs that have sneaked in
        logging.debug('Removing %d assets in dr2Underlying that are not actually DRs', len(notDR))
        for sid in notDR:
            del dr2Underlying[sid]
    data.dr2Underlying.update(dr2Underlying)

    # Check for 3-way mappings and unpick any that exist
    underSet = set(data.dr2Underlying.values())
    underSet.discard(None)
    drSet = set([sid for sid in data.dr2Underlying.keys() if data.dr2Underlying[sid] is not None])
    threeWays = drSet.intersection(underSet)
    for sid in threeWays:
        mappedTo = data.dr2Underlying[sid]
        underFor = [dr for dr in drSet if (data.dr2Underlying[dr] == sid)]
        logging.debug('DR: %s (%s) mapped to underlying: %s (%s)',
                sid.getSubIDString(), data.assetTypeDict.get(sid, None),
                mappedTo.getSubIDString(), data.assetTypeDict.get(mappedTo, None))
        for sid2 in underFor:
            logging.debug('... and is itself underlying for DR: %s, (%s)',
                    sid2.getSubIDString(), data.assetTypeDict.get(sid2, None))
            logging.debug('... remapped to %s (%s)',
                    mappedTo.getSubIDString(), data.assetTypeDict.get(mappedTo, None))
            data.dr2Underlying[sid2] = mappedTo

    drNoUnderlying = [sid for sid in data.dr2Underlying if data.dr2Underlying[sid] is None]
    if len(drNoUnderlying) > 0:
        logging.debug('%d DRs with underlying asset of None', len(drNoUnderlying))
    logging.debug('%d DRs mapped to underlying or None', len(data.dr2Underlying))

    # Check for non-model geography assets
    nonModelAssets = set(data.universe)
    for ids in data.rmgAssetMap.values():
        nonModelAssets = nonModelAssets.difference(ids)
    if len(nonModelAssets) > 0:
        logging.info('%d assets from non-model geographies', len(nonModelAssets))
    data.assetRMGMap = dict([(sid, rmg_id) for (rmg_id, ids) in \
                    data.rmgAssetMap.items() for sid in ids])
    if debugOutput:
        typeList = [data.assetRMGMap.get(sid, None) for sid in data.universe]
        sidList = [sid.getSubIDString() for sid in data.universe]
        outName = 'tmp/assetRMG-%s.csv' % date
        dumpAssetListForDebugging(outName, sidList, typeList)

    # Load home country classification data
    clsData = get_home_country(data.universe, date, modelDB, marketDB, clsType='HomeCountry')

    # Load secondary home country classification data
    clsData2 = get_home_country(data.universe, date, modelDB, marketDB, clsType='HomeCountry2')

    # Initialise important lists and dicts
    missing = nonModelAssets.difference(list(clsData.keys()))
    validRiskModelGroups = set([r.mnemonic for r in rmgList if r not in nurseryRMGList])
    nurseryRMGs = set([r.mnemonic for r in nurseryRMGList])
    rmgMnemonicIDMap = dict([(r.mnemonic, r.rmg_id) for r in rmgList])
    rmgIDMnemonicMap = dict([(r.rmg_id, r.mnemonic) for r in rmgList])
    rmgIDObjMap = dict([(r.rmg_id, r) for r in rmgList])
    foreign = list()

    # Loop round assets and home countries - do some manipulation where necessary
    homeCtyRMG_Original_dict = {}
    for (sid, homeCty) in clsData.items():
        homeCtyRMG = homeCty.classification.code
        homeCtyRMG_Original = homeCtyRMG
        homeCtyRMG_Original_dict[sid] = homeCtyRMG_Original
        inNursery = False
        tradingCtyID = data.assetRMGMap.get(sid, None)
        tradingCtyRMG = rmgIDMnemonicMap.get(tradingCtyID, None)

        if homeCtyRMG not in validRiskModelGroups:
            # Get secondary home country information
            homeCty2 = clsData2.get(sid, homeCty)
            homeCty2RMG = homeCty2.classification.code
            # If secondary home country is legit, assign it there
            if homeCty2RMG in validRiskModelGroups:
                homeCtyRMG = homeCty2RMG
                checkAsset(trackList, sid, 'Home2', homeCtyRMG, tradingCtyRMG, homeCtyRMG==tradingCtyRMG)
            # If trading country is legit, assign it there
            elif tradingCtyRMG in validRiskModelGroups:
                foreign.append(sid)
                checkAsset(trackList, sid, 'Trading', homeCtyRMG, tradingCtyRMG, homeCtyRMG==tradingCtyRMG)
                homeCtyRMG = tradingCtyRMG
            # Otherwise check whether one of the markets is a nursery market
            elif homeCtyRMG in nurseryRMGs:
                inNursery = True
                checkAsset(trackList, sid, 'Nursery', homeCtyRMG, tradingCtyRMG, homeCtyRMG==tradingCtyRMG)
            elif homeCty2RMG in nurseryRMGs:
                inNursery = True
                homeCtyRMG = homeCty2RMG
                checkAsset(trackList, sid, 'Nursery2', homeCtyRMG, tradingCtyRMG, homeCtyRMG==tradingCtyRMG)
            elif tradingCtyRMG in nurseryRMGs:
                inNursery = True
                foreign.append(sid)
                checkAsset(trackList, sid, 'Trading2', homeCtyRMG, tradingCtyRMG, homeCtyRMG==tradingCtyRMG)
                homeCtyRMG = tradingCtyRMG
            # Drop if neither quotation nor secondary country is valid or nursery market
            elif sid in nonModelAssets:
                missing.add(sid)
                checkAsset(trackList, sid, 'Missing', homeCtyRMG, tradingCtyRMG, homeCtyRMG==tradingCtyRMG)
                continue
            # Keep in quotation country otherwise
            else:
                # Nothing should actually get this far - check sometime
                foreign.append(sid)
                checkAsset(trackList, sid, 'Unclassified', homeCtyRMG, tradingCtyRMG, homeCtyRMG==tradingCtyRMG)
                assert homeCtyRMG==tradingCtyRMG
                homeCtyRMG = tradingCtyRMG
                continue
        else:
            checkAsset(trackList, sid, 'Home', homeCtyRMG, tradingCtyRMG, homeCtyRMG==tradingCtyRMG)

        if sid in data.nurseryUniverse and not inNursery:
            logging.info('Moved nursery asset %s from %s to non-nursery %s',
                    sid.getSubIDString(), tradingCtyRMG, homeCtyRMG)
            data.nurseryUniverse.remove(sid)

        if inNursery and sid not in data.nurseryUniverse:
            logging.info('Asset %s classed as nursery, but not in nursery universe - fixing this',
                    sid.getSubIDString())
            data.nurseryUniverse.append(sid)

        # Manually reassign country and currency exposures if required
        tweakData = None
        changeExp = False
        if sid.getSubIdString() in rmgOverride:
            tweakData = rmgOverride[sid.getSubIdString()]
        elif data.sidToCIDMap[sid] in rmgOverride:
            tweakData = rmgOverride[data.sidToCIDMap[sid]]
        if tweakData is not None:
            if tweakData[0] <= date < tweakData[1]:
                # Change country exposure if necessary
                if homeCtyRMG != tweakData[2]:
                    if tradingCtyID is not None:
                        data.rmgAssetMap[tradingCtyID].remove(sid)
                    logging.debug('Changing RMG exposure for (%s, %s) from %s to %s',
                            data.sidToCIDMap.get(sid, None), sid.getSubIdString(), homeCtyRMG, tweakData[2])
                    data.rmgAssetMap[rmgMnemonicIDMap[tweakData[2]]].add(sid)
                    foreign.append(sid)
                    changeExp = True
                # Change currency exposure if necessary
                if homeCtyRMG != tweakData[3]:
                    if tradingCtyID is not None:
                        data.rmcAssetMap[tradingCtyID].remove(sid)
                    logging.debug('Changing RMG currency exposure for (%s, %s) from %s to %s',
                            data.sidToCIDMap.get(sid, None), sid.getSubIdString(), homeCtyRMG, tweakData[3])
                    data.rmcAssetMap[rmgMnemonicIDMap[tweakData[3]]].add(sid)
                    changeExp = True

        # If home country differs from RiskModelGroup, reassign
        # In time, data.rmcAssetMap should be independent of data.rmgAssetMap
        if (not changeExp) and (homeCtyRMG != tradingCtyRMG):
            if tradingCtyID is not None:
                data.rmgAssetMap[tradingCtyID].remove(sid)
                data.rmcAssetMap[tradingCtyID].remove(sid)
            data.rmgAssetMap[rmgMnemonicIDMap[homeCtyRMG]].add(sid)
            data.rmcAssetMap[rmgMnemonicIDMap[homeCtyRMG]].add(sid)
            foreign.append(sid)
            if inNursery:
                logging.info('Foreign nursery asset %s reassigned from %s to %s',
                        sid.getSubIDString(), tradingCtyRMG, homeCtyRMG)

        checkAsset(trackList, sid, 'Final', homeCtyRMG, tradingCtyRMG, homeCtyRMG==tradingCtyRMG)
    data.homeCtyRMG_Original_dict = homeCtyRMG_Original_dict
    if len(foreign) > 0:
        logging.debug('%d DR-like instruments and/or foreign assets', len(foreign))
    reassigned = set(nonModelAssets).intersection(set(foreign))
    if len(reassigned) > 0:
        logging.info('%d unassigned assets reassigned to RMG successfully', len(reassigned))
    if (len(missing) > 0):
        logging.warning('%d assets dropped due to missing/invalid home country',
                      len(missing))
        if len(missing) < 5:
            logging.info('missing Axioma IDs: %s', ', '.join([
                    sid.getSubIDString() for sid in missing]))
    data.assetRMGMap = dict([(sid, rmg_id) for (rmg_id, ids) in data.rmgAssetMap.items() for sid in ids])

    #  Mapping "foreign" subIssues to home currency
    if len(foreign) > 0:
        data.drCurrData = dict()
        for rmg in rmgList:
            isoCode = rmg.getCurrencyCode(date)
            currency_id = marketDB.getCurrencyID(isoCode, date)
            rmg_assets = set(data.rmgAssetMap[rmg.rmg_id]).intersection(set(foreign))
            for sid in rmg_assets:
                data.drCurrData[sid] = currency_id
    data.foreign = foreign

    # Catch any remaining assets whose trading currency and country differ
    if (len(tradingCountryCurrencyMismatch) > 0) and (data.drCurrData is not None):
        if data.drCurrData is None:
            data.drCurrData = dict()
        mismatchIDs = set(tradingCountryCurrencyMismatch).difference(set(data.foreign))
        if len(mismatchIDs) > 0:
            logging.debug('Mapping %d non-foreign listings with foreign trading currencies to their home currency',
                    len(mismatchIDs))
        for sid in mismatchIDs:
            rmg_id = data.assetRMGMap[sid]
            isoCode = rmgIDObjMap[rmg_id].getCurrencyCode(date)
            currency_id = marketDB.getCurrencyID(isoCode, date)
            data.drCurrData[sid] = currency_id

    debugOutput = False
    if debugOutput:
        univSorted = sorted(data.universe)
        typeList = [rmgIDObjMap.get(data.assetRMGMap.get(sid, None), None) for sid in univSorted]
        sidList = [sid.getSubIDString() for sid in univSorted]
        outName = 'tmp/assetRMG2-%s.csv' % date
        dumpAssetListForDebugging(outName, sidList, typeList)

    # Report on assets that share a company ID but have more than one home market
    # between them
    countryCheck = False
    if countryCheck:
        subSids = []
        subRMGs = []
        for (groupID, sidList) in data.subIssueGroups.items():
            sidList = [sid for sid in sidList if (data.assetTypeDict[sid] != 'AmerDR') \
                    and (data.assetTypeDict[sid] != 'GlobalDR')]
            rmgList = [modelDB.getRiskModelGroup(data.assetRMGMap[sid]).mnemonic for sid in sidList]
            tickerMap = modelDB.getIssueTickers(date, sidList, marketDB)
            mltplRMGs = list(set(rmgList))
            if len(mltplRMGs) > 1:
                logging.info('Multiple country assignment for company %s: %s',
                        groupID, ','.join(list(mltplRMGs)))
                if len(sidList) == 2:
                    sid1 = sidList[0]
                    subSids.append('TCK-%s-%s:%s' % \
                            (tickerMap[sid1], rmgList[0], sid1.getModelID().getPublicID()))
                    sid2 = sidList[1]
                    subRMGs.append('TCK-%s-%s:%s' % \
                            (tickerMap[sid2], rmgList[1], sid2.getModelID().getPublicID()))
                else:
                    for (sid, rmg) in zip(sidList, rmgList):
                        subSids.append('%s,%s-%s,%s' % \
                                (groupID, tickerMap[sid], sid.getModelID().getPublicID(), data.assetTypeDict[sid]))
                        subRMGs.append(rmg)
        if len(subSids) > 0:
            outName = 'tmp/mltplRMGs-%s.csv' % date
            dumpAssetListForDebugging(outName, subSids, subRMGs)

    logging.debug('process_asset_information: end')
    return data, missing

def checkAsset(trackList, sid, step, homeRMG, tradeRMG, noChange):
    if sid in trackList:
        logging.info('Asset %s classed as %s, Home: %s, Trading market: %s, Foreign: %s',
                sid.getSubIDString(), step, homeRMG, tradeRMG, noChange is False)
    return

def computeTotalIssuerMarketCaps(
        data, date, numeraire, modelDB, marketDB, sumAcrossRMG=False, debugReport=False, days_avg=30):
    """Returns an array of issuer-level market caps, using
    the asset_dim_company table in marketdb_global.
    data should be a Struct containing universe, marketCaps, and
    assetIdxMap attributes.
    """
    logging.debug('computeTotalIssuerMarketCaps: begin')
    sidList = data.universe

    # Find all company IDs for the assets
    sidCompanyMap = modelDB.getIssueCompanies(date, sidList, marketDB)
    companies = set(sidCompanyMap.values())    

    # Get lists of other companies associated with DLCs
    DLCList = list()
    extraCompanies = list()
    logging.info('DLC handling enabled')
    DLCList = modelDB.getDLCs(date, marketDB)
    DLCList = [dlc for dlc in DLCList if len(dlc) > 1]
    extraCompanies = set([comp for sublist in DLCList for comp in sublist])
    companies = companies.union(extraCompanies)

    # Find all subIssues for full list of companies
    allSidCompanyMap = modelDB.getCompanySubIssues(date, companies, marketDB)
    companySidMap = dict()
    for sid in allSidCompanyMap.keys():
        (company, excludeFromMcap, ctyISO) = allSidCompanyMap[sid]
        companySidMap.setdefault(company, list()).append((sid, excludeFromMcap, ctyISO))

    # Load market caps for sub-issues not present in initial universe
    extraSids = list(set(allSidCompanyMap.keys()) - set(sidList))
    logging.info('%d share lines reside outside model universe', len(extraSids))
    otherIdxMap = dict((j, i) for (i, j) in enumerate(extraSids))
    otherMarketCaps = numpy.zeros((len(extraSids),))
    if len(extraSids) > 0:
        otherMarketCaps, mCapDates = robustLoadMCaps(date, extraSids, numeraire.currency_id, modelDB, marketDB, days_avg=days_avg)
    
    # Initialise various types of market cap
    totalIssuerMarketCaps = numpy.array(data.marketCaps)
    issuerMarketCapDict = dict()

    # Loop through companies
    for (company, sidVals) in companySidMap.items():

        # Skip companies where only one SubIssue is alive
        if len(sidVals) < 1:
            continue
        if len(sidVals) == 1:
            sid = sidVals[0][0]
            if sid in data.assetIdxMap:
                issuerMarketCapDict[company] = data.marketCaps[data.assetIdxMap[sid]]
            else:
                issuerMarketCapDict[company] = otherMarketCaps[otherIdxMap[sid]]
            continue

        # Skip companies where all lines are excluded
        excludedSids = len([i for i in sidVals if i[1]])
        if excludedSids == len(sidVals):
            logging.debug('All assets for company %s are excluded', company)
            maxCap = 0.0
            # Find the maximum mcap of all these assets
            for (sid, excludeFromMCap, ctyISO) in sidVals:
                if sid in data.assetIdxMap:
                    maxCap = max(maxCap, data.marketCaps[data.assetIdxMap[sid]])
                else:
                    maxCap = max(maxCap, otherMarketCaps[otherIdxMap[sid]])
            # Set the total issuer mcap equal to the maximum
            issuerMarketCapDict[company] = maxCap
            for (sid, excludeFromMCap, ctyISO) in sidVals:
                if sid in data.assetIdxMap:
                    totalIssuerMarketCaps[data.assetIdxMap[sid]] = issuerMarketCapDict[company]
            continue

        # Group linked SubIssues by RiskModelGroup
        rmgSidMap = dict()
        if sumAcrossRMG:
            # At present, sumAcrossRMG should always be False - it should
            # only be otherwise if the excludeFromMCap field is fully fixed
            for (sid, excludeFromMcap, ctyISO) in sidVals:
                rmgSidMap.setdefault('ALL', list()).append(
                        (sid, excludeFromMcap, 'ALL'))
        else:
            for (sid, excludeFromMcap, ctyISO) in sidVals:
                rmgSidMap.setdefault(ctyISO, list()).append(
                                    (sid, excludeFromMcap, ctyISO))

        # Deal with Hong Kong / China
        if ('CN' in rmgSidMap) and ('HK' in rmgSidMap):
            rmgSidMap['CN-HK'] = rmgSidMap['CN'] + rmgSidMap['HK']
            rmgSidMap['CN'] = []
            rmgSidMap['HK'] = []

        # Initialise flavours of total mcap
        runningMarketCap = 0.0

        # Loop through RiskModelGroups, sum up market caps
        for (rmg, sidValList) in rmgSidMap.items():
            if len(sidValList) < 1:
                continue

            # Repeat earlier logic for single-issue cases
            if len(sidValList) == 1:
                sid = sidValList[0][0]
                if sid in data.assetIdxMap:
                    mcap = data.marketCaps[data.assetIdxMap[sid]]
                else:
                    mcap = otherMarketCaps[otherIdxMap[sid]]
                if mcap > runningMarketCap:
                    runningMarketCap = mcap
                continue

            # Skip companies where all lines are excluded
            # At this point there must be at least one valid issue
            # associated with at least one RMG
            excludedSids = len([i for i in sidValList if i[1]])
            if excludedSids == len(sidValList):
                logging.debug('All assets for company %s in %s are excluded', company, rmg)
                continue

            issuerMarketCap = 0.0
            # Sum mcaps over valid issues
            for (sid, excludeFromMCap, ctyISO) in sidValList:
                if not excludeFromMCap:
                    if sid in data.assetIdxMap:
                        issuerMarketCap += data.marketCaps[data.assetIdxMap[sid]]
                    else:
                        issuerMarketCap += otherMarketCaps[otherIdxMap[sid]]

            # If, by some chance, there is still no issuerMCap, we ignore the 
            # exclude from mcap flag as an emergency override
            if issuerMarketCap == 0.0:
                logging.warning('Company %s (%d sub-issues in %s) has zero issuer market cap', 
                            company, len(sidValList), rmg)
                logging.info('Ignoring excludeFromMCap flag for %s', company)
                for (sid, excludeFromMCap, ctyISO) in sidValList:
                    if sid in data.assetIdxMap:
                        issuerMarketCap += data.marketCaps[data.assetIdxMap[sid]]
                    else:
                        issuerMarketCap += otherMarketCaps[otherIdxMap[sid]]

            # Keep track of the largest total issuer mcap recorded
            if issuerMarketCap > runningMarketCap:
                runningMarketCap = issuerMarketCap
        
        # Loop through all assets within a company, and take the largest total issuer mcap
        # to be the value for each asset within the company
        # This ensures that the figure matches across RMGs
        issuerMarketCapDict[company] = runningMarketCap
        for (sid, excludeFromMCap, ctyISO) in sidVals:
            if sid in data.assetIdxMap:
                totalIssuerMarketCaps[data.assetIdxMap[sid]] = issuerMarketCapDict[company]

    # Get lists of DLC companies
    data.DLCMarketCap = numpy.array(totalIssuerMarketCaps, copy=True)
    for (i_DLC, companyList) in enumerate(DLCList):

        # Compute Market Cap across all companies in DLC entity
        DLCCap = 0.0
        for company in companyList:
            if company in issuerMarketCapDict:
                DLCCap += issuerMarketCapDict[company]

        # Map this to the relevant subIssues
        for company in companyList:
            for sidVals in companySidMap.get(company, []):
                sid = sidVals[0]
                if sid in data.assetIdxMap:
                    data.DLCMarketCap[data.assetIdxMap[sid]] = DLCCap
                    if debugReport:
                        logging.info('DLC %d, CID %s, SID: %s, MCap %.8f, DLC Cap %.8f',
                                i_DLC, company, sid, 
                                totalIssuerMarketCaps[data.assetIdxMap[sid]]/1.0e6, DLCCap/1.0e6)

    if debugReport:
        cidList = [allSidCompanyMap.get(sid, [''])[0] for sid in data.universe]
        typeList = [data.assetTypeDict.get(sid, 'None') for sid in data.universe]
        allCaps = numpy.zeros((len(data.universe), 3), float)
        allCaps[:,0] = numpy.ravel(ma.filled(data.marketCaps, 0.0))
        allCaps[:,1] = numpy.ravel(ma.filled(totalIssuerMarketCaps, 0.0))
        allCaps[:,2] = numpy.ravel(ma.filled(data.DLCMarketCap, 0.0))
        sidList = []
        for (cid, sid, atype) in zip(cidList, data.universe, typeList):
            sidList.append('%s|%s|%s' % (cid, sid.getSubIDString(), atype))
        outName = 'tmp/mcaps-%s.csv' % date
        Utilities.writeToCSV(allCaps, outName, rowNames=sidList, columnNames=['cap','totalCap','DLCCap'], dp=4)

    logging.debug('computeTotalIssuerMarketCaps: end')
    data.issuerTotalMarketCaps = totalIssuerMarketCaps
    return
