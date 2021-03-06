import logging
import numpy.ma as ma
import numpy
import copy
import pandas
import datetime
from collections import defaultdict
from itertools import chain
from riskmodels import Utilities
oldPD = Utilities.oldPandasVersion()

# Asset type meta-groups
drAssetTypes = ['NVDR', 'GlobalDR', 'TDR', 'AmerDR', 'FStock', 'CDI', 'DR']
commonStockTypes = ['Com', 'ComClsA', 'ComClsB', 'ComClsC', 'ComClsD', 'ComClsE', 'ComClsL']
commonShareTypes = commonStockTypes + ['REIT', 'InvT']
otherAllowedStockTypes = ['REIT', 'StapSec']
preferredStockTypes = ['Pref', 'PrefClsA', 'PrefClsB', 'PrefClsC', 'PrefClsD', 'PrefClsE']
fundAssetTypes = ['CEFund', 'InvT', 'Misc', 'UnitT']
localChineseAssetTypes = ['AShares', 'BShares']
intlChineseAssetTypes = ['HShares', 'RCPlus']
etfAssetTypes = ['NonEqETF', 'ComETF', 'StatETF', 'ETFnoRM']
otherAssetTypes = ['LLC', 'LP', 'ComWI', 'UnCls', 'CFCont-SS', 'ComSPAC']
spacTypes = ['ComSPAC', 'MACSPAC']
noTransferList = etfAssetTypes + ['EIFCont-SS', 'CFCont-SS', 'MF', 'MACSPAC']
noHBetaList = [None, 'MF', 'MACSPAC', 'EIFCont-SS', 'CFCont-SS']
noProxyTypes = etfAssetTypes + spacTypes

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

# Useful general utilities
def get_asset_info(date, universe, modelDB, marketDB, clsFamily, clsMemberStr):
    """Gets axioma asset classification from the DB
    Returns a dict of subids to asset types
    """
    clsFamily = marketDB.getClassificationFamily(clsFamily)
    assert(clsFamily is not None)
    clsMembers = dict([(i.name, i) for i in marketDB.getClassificationFamilyMembers(clsFamily)])
    clsMember = clsMembers.get(clsMemberStr, None)
    assert(clsMember is not None)
    clsRevision = marketDB.getClassificationMemberRevision(clsMember, date)
    clsData = modelDB.getMktAssetClassifications(clsRevision, list(universe), date, marketDB)
    retDict = {sid: (item.classification.code if item.classification is not None else None) for sid, item in clsData.items()}
    return retDict

def get_asset_info_range(date, universe, modelDB, marketDB, clsFamily, clsMemberStr):
    """Gets axioma asset classification history from the DB
    Returns a nested dict of subissues and dates to asset type
    """
    clsFamily = marketDB.getClassificationFamily(clsFamily)
    assert(clsFamily is not None)
    clsMembers = dict([(i.name, i) for i in marketDB.getClassificationFamilyMembers(clsFamily)])
    clsMember = clsMembers.get(clsMemberStr, None)
    assert(clsMember is not None)
    clsRevision = marketDB.getClassificationMemberRevision(clsMember, date)
    clsData = modelDB.getMktAssetClassifications(clsRevision, list(universe), date, marketDB, dtRange=True)
    retDict = defaultdict(dict)
    for sid, typeHist in clsData.items():
        for dt, item in typeHist.items():
            retDict[sid][dt] = item.classification.code if item.classification is not None else None
    return retDict

def sort_spac_assets(date, subIssues, modelDB, marketDB, returnExSpac=False):
    # Sort SPACs into pre- and post-announcement groups
    # Get assets currently classed as SPAC
    assetTypeDict = get_asset_info(date, subIssues, modelDB, marketDB, 'ASSET TYPES', 'Axioma Asset Type')
    allSPACs = set([sid for sid in subIssues if assetTypeDict.get(sid, None) in spacTypes])

    # Determine assets that were formerly SPACs
    fromDates1 = Utilities.load_ipo_dates(date, subIssues, modelDB, marketDB, exSpacAdjust=False)
    fromDates2 = Utilities.load_ipo_dates(date, subIssues, modelDB, marketDB, exSpacAdjust=True)
    exSpac = set(fromDates2[fromDates1!=fromDates2].index)

    # Determine which ex-SPACs are true ex-SPACs and which have merely passed the announcement date
    annSpac = set([sid for sid in exSpac if assetTypeDict.get(sid, None) in spacTypes])
    preSPACs = allSPACs.difference(exSpac)

    # Report and return
    logging.info('%d assets classed as %s divided into %d pre- and %d post-announcement',
            len(allSPACs), ','.join(spacTypes), len(preSPACs), len(annSpac))
    logging.info('%d assets classed as former SPACs', len(exSpac))
    if returnExSpac:
        return exSpac
    return preSPACs

def get_home_country(universe, date, modelDB, marketDB, clsType='HomeCountry'):

    # Load home country classification data
    clsFamily = marketDB.getClassificationFamily('REGIONS')
    assert(clsFamily is not None)
    clsMembers = dict([(i.name, i) for i in marketDB.getClassificationFamilyMembers(clsFamily)])
    clsMember = clsMembers.get(clsType, None)
    assert(clsMember is not None)
    clsRevision = marketDB.getClassificationMemberRevision(clsMember, date)
    return modelDB.getMktAssetClassifications(clsRevision, list(universe), date, marketDB)

def list_to_csv(fileName, assetList):
    aList = sorted(assetList)
    outFile = open(fileName, 'w')
    for sid in aList:
        if hasattr(sid, 'getSubIdString'):
            outFile.write('%s,' % sid.getSubIdString())
        else:
            outFile.write('%s,' % sid)
        outFile.write('\n')
    outFile.close()
    return

def get_all_markets(universe, date, modelDB, marketDB, tradeDict=None):
    # Get all Home and Trading markets
    home1 = get_home_country(universe, date, modelDB, marketDB, clsType='HomeCountry')
    home1 = [home1[sid].classification.code if sid in home1 else None for sid in universe]
    home2 = get_home_country(universe, date, modelDB, marketDB, clsType='HomeCountry2')
    home2 = [home2[sid].classification.code if sid in home2 else None for sid in universe]
    if tradeDict is None:
        trade = get_home_country(universe, date, modelDB, marketDB, clsType='Market')
        trade = [trade[sid].classification.code if sid in trade else None for sid in universe]
    else:
        asset_trad_dict = Utilities.flip_dict_of_lists(tradeDict)
        trade = [asset_trad_dict[sid].mnemonic if sid in asset_trad_dict else None for sid in universe]
    mkt_df = pandas.DataFrame(list(zip(home1, home2, trade)), index=universe, columns=['Home1', 'Home2', 'Trade'])
    return mkt_df

def dict_to_csv(fileName, assetDict):
    if type(assetDict) == pandas.Series:
        assetDict = assetDict.to_dict()
    aList = sorted(assetDict.keys())
    outFile = open(fileName, 'w')
    for sid in aList:
        if hasattr(sid, 'getSubIdString'):
            outFile.write('%s,' % sid.getSubIdString())
        else:
            outFile.write('%s,' % sid)
        outFile.write('%s,\n' % assetDict.get(sid, ''))
    outFile.close()
    return

def robustLoadMCaps(date, assets, currency_id, modelDB, marketDB, days_avg=30, lookBack=True):
    """ Compute average market cap over the past calendar month
    For assets with missing values, we look back a further month
    """

    if len(assets) < 1:
        return None

    # Load average market caps
    mcapDates = modelDB.getAllRMGDateRange(date, days_avg)
    marketCaps = modelDB.getAverageMarketCaps(\
            mcapDates, assets, currency_id, marketDB, loadAllDates=True)
    marketCaps = pandas.Series(marketCaps, index=assets)

    # Look back an extra month for any assets with missing market cap
    marketCaps = marketCaps.replace(0.0, numpy.nan)
    missingCapIds = list(marketCaps[marketCaps.isnull()].index)
    if lookBack and (len(missingCapIds) > 0):
        logging.warning('%d assets have missing/zero avg %d-day cap', len(missingCapIds), len(mcapDates))
        logging.warning('Looking back further...')
        extraDates = modelDB.getAllRMGDateRange(date-datetime.timedelta(days_avg), days_avg)
        extraCaps = modelDB.getAverageMarketCaps(
                extraDates, missingCapIds, currency_id, marketDB, loadAllDates=True)
        extraCaps = pandas.Series(extraCaps, index=missingCapIds).replace(0.0, numpy.nan)
        marketCaps.update(extraCaps)

    # Report on assets that still have missing cap
    missingCapIdx = list(marketCaps[marketCaps.isnull()].index)
    if len(missingCapIdx) > 0:
        logging.warning('%d assets still have missing/zero avg %d-day cap', len(missingCapIdx), len(mcapDates))
        #list_to_csv('tmp/MissingCap.csv', missingCapIdx)

    return marketCaps

def computeTotalIssuerMarketCaps(
        date, marketCaps, numeraire, modelDB, marketDB, debugReport=False):
    """Returns an array of issuer-level market caps, using
    the asset_dim_company table in marketdb_global.
    data should be a Struct containing universe, marketCaps, and
    assetIdxMap attributes.
    """
    logging.debug('computeTotalIssuerMarketCaps: begin')

    # Initialise
    mcaps = marketCaps.copy(deep=True)
    universe = list(mcaps.index)

    # Find all company IDs for the assets
    sidCompanyMap = modelDB.getIssueCompanies(date, universe, marketDB)
    companies = set(sidCompanyMap.values())

    # Get lists of other companies associated with DLCs
    DLCList = modelDB.getDLCs(date, marketDB, keepSingles=False)
    extraCompanies = set([cid for sublist in DLCList for cid in sublist])
    companies = companies.union(extraCompanies)

    # Get mapping from each CID to its full set of issues
    allSidCompanyMap = modelDB.getCompanySubIssues(date, companies, marketDB)
    companySidMap = dict()
    for sid in allSidCompanyMap.keys():
        (company, excludeFromMcap, ctyISO) = allSidCompanyMap[sid]
        companySidMap.setdefault(company, list()).append((sid, excludeFromMcap, ctyISO))

    # Load market caps for sub-issues not present in initial universe
    extraSids = set(allSidCompanyMap.keys()).difference(set(universe))
    if len(extraSids) > 0:
        logging.debug('%d share lines reside outside model universe', len(extraSids))
        otherMarketCaps = robustLoadMCaps(
                    date, extraSids, numeraire.currency_id, modelDB, marketDB)
        mcaps = pandas.concat([mcaps, otherMarketCaps])
        universe = list(mcaps.index)

    # Get asset type data
    assetTypeDict = get_asset_info(date, universe, modelDB, marketDB, 'ASSET TYPES', 'Axioma Asset Type')

    # Initialise various types of market cap
    totalIssuerMarketCaps = mcaps.copy(deep=True)
    issuerMarketCapDict = dict()

    # Loop through companies
    for (company, sidInfo) in companySidMap.items():

        # Treat companies with one or fewer SubIssues
        if len(sidInfo) < 1:
            continue
        if len(sidInfo) == 1:
            issuerMarketCapDict[company] = mcaps[sidInfo[0][0]]
            continue

        # Skip companies where all lines are excluded
        excludedSids = [sid for (sid, excl, iso) in sidInfo if excl]
        if len(excludedSids) == len(sidInfo):
            logging.debug('All assets for company %s are excluded', company)
            # Set the total issuer mcap equal to the maximum mcap of all these assets
            issuerMarketCapDict[company] = max(mcaps[excludedSids].values)
            continue

        # Group linked SubIssues by RiskModelGroup
        rmgSidMap = dict()
        for (sid, excl, iso) in sidInfo:
            rmgSidMap.setdefault(iso, list()).append((sid, excl))

        # Deal with Hong Kong / China
        if ('CN' in rmgSidMap) and ('HK' in rmgSidMap):
            rmgSidMap['CN-HK'] = rmgSidMap['CN'] + rmgSidMap['HK']
            rmgSidMap['CN'] = []
            rmgSidMap['HK'] = []

        # Initialise flavours of total mcap
        runningMarketCap = 0.0

        # Loop through RiskModelGroups; sum up company's eligible market cap within each 
        for (rmg, sidValList) in rmgSidMap.items():

            # Skip empty cases
            if len(sidValList) < 1:
                continue

            # Repeat earlier logic for single-issue cases
            if len(sidValList) == 1:
                runningMarketCap = max(runningMarketCap, mcaps[sidValList[0][0]])
                continue

            # Skip companies where all lines are excluded
            # At this point there must be at least one valid issue associated with at least one RMG
            okSids = [sid for (sid, excl) in sidValList if not excl]
            if len(okSids) < 1:
                logging.debug('All assets for company %s in %s are excluded', company, rmg)
                continue

            # Check to see if there are both DR-types and other issues
            # Exclude DRs if this is true
            drs = [sid for sid in okSids if assetTypeDict[sid] in drAssetTypes]
            if len(drs) < len(okSids):
                okSids = [sid for sid in okSids if sid not in drs]

            # Sum mcaps over valid issues
            issuerMarketCap = mcaps[okSids].fillna(0.0).sum()

            # If, by some chance, there is still no issuerMCap, ignore the exclude from mcap flag
            if issuerMarketCap == 0.0:
                logging.warning('Company %s (%d sub-issues in %s) has zero issuer market cap', company, len(sidValList), rmg)
                allSids = [sid for (sid, excl) in sidValList]
                issuerMarketCap = mcaps[allSids].fillna(0.0).sum()

            # Keep track of the largest total issuer mcap recorded
            runningMarketCap = max(runningMarketCap, issuerMarketCap)

        # Set the largest total mcap across each RMG to be the total mcap for the entire company
        issuerMarketCapDict[company] = runningMarketCap

    # Loop through all assets within a company, and assign total issuer market cap to each
    for cid, tcap in issuerMarketCapDict.items():
        sidList = [sid for (sid, excl, iso) in companySidMap[cid]]
        totalIssuerMarketCaps[sidList] = issuerMarketCapDict[cid]

    # Loop round list of DLC companies
    DLCMarketCap = totalIssuerMarketCaps.copy(deep=True)
    for (i_DLC, cidList) in enumerate(DLCList):

        # Compute Market Cap across all companies in DLC entity
        dlcCap = numpy.sum([issuerMarketCapDict.get(cid, 0.0) for cid in cidList], axis=None)

        # Map this to the relevant subIssues
        for cid in cidList:
            sidList = [sid for (sid, excl, iso) in companySidMap.get(cid, [])]
            DLCMarketCap[sidList] = dlcCap

    # Create market cap dataframe
    universe = sorted(marketCaps.index)
    mcapDF = pandas.concat([mcaps, totalIssuerMarketCaps, DLCMarketCap], axis=1).loc[universe]
    mcapDF.columns=['marketCap', 'totalCap', 'dlcCap']

    # Output some debugging info
    if debugReport:
        cidList = [allSidCompanyMap.get(sid, [''])[0] for sid in universe]
        assetTypeDict = get_asset_info(
            date, universe, modelDB, marketDB, 'ASSET TYPES', 'Axioma Asset Type')
        typeList = [assetTypeDict.get(sid, 'None') for sid in universe]
        allCaps = numpy.zeros((len(universe), 3), float)
        allCaps[:,0] = numpy.ravel(ma.filled(mcaps[universe], 0.0))
        allCaps[:,1] = numpy.ravel(ma.filled(totalIssuerMarketCaps[universe], 0.0))
        allCaps[:,2] = numpy.ravel(ma.filled(DLCMarketCap[universe], 0.0))
        sidList = []
        for (cid, sid, atype) in zip(cidList, universe, typeList):
            (company, excludeFromMcap, ctyISO) = allSidCompanyMap.get(sid, [None,None,None])
            xcl = 'Include'
            if excludeFromMcap:
                xcl = 'Exclude'
            sidList.append('%s|%s|%s|%s|%s' % (cid, sid.getSubIDString(), atype, xcl, ctyISO))
        outName = 'tmp/mcaps-%s.csv' % date
        Utilities.writeToCSV(allCaps, outName, rowNames=sidList, columnNames=['cap','totalCap','DLCCap'], dp=4)

    logging.debug('computeTotalIssuerMarketCaps: end')
    return mcapDF

def getAssetCompanyMapping(date, universe, modelDB, marketDB, includeSingles=False):
    """Returns a dataframe which has the following columns:
    SubIssue, CompanyID, excludeFromMCap, ctyISO, AssetType, more can be added later on if required
    """

    # Get set of companies
    sidCompanyMap = modelDB.getIssueCompanies(date, universe, marketDB)
    companies = set(sidCompanyMap.values())

    # Get lists of other companies associated with DLCs
    # Ignore DLCs for now
    #DLCList = modelDB.getDLCs(date, marketDB, keepSingles=False)
    #extraCompanies = set([cid for sublist in DLCList for cid in sublist])
    #companies = companies.union(extraCompanies) # add extra companies to the list

    # get all the subissues under each company ID
    allSidCompanyMap = modelDB.getCompanySubIssues(date, companies, marketDB)

    # convert to dataframe and format
    assetInfo = pandas.DataFrame.from_dict(allSidCompanyMap, orient='index')
    assetInfo.columns = ['CompanyID', 'excludeFromMCap', 'ctyISO']
    assetInfo.index.name='SubIssue'
    all_subissues = assetInfo.index

    # get asset type and added it to the df
    assetTypeDict = get_asset_info(date, all_subissues, modelDB, marketDB, 'ASSET TYPES', 'Axioma Asset Type')
    assetTypeSeries = pandas.Series(assetTypeDict)
    assetInfo['AssetType'] = assetTypeSeries

    # Get market type also
    marketTypeDict = get_asset_info(date, all_subissues, modelDB, marketDB,  'REGIONS', 'Market')
    marketTypeSeries = pandas.Series(marketTypeDict)
    assetInfo['MarketType'] = marketTypeSeries

    return assetInfo

def getTSOByType(date, subids, modelDB, marketDB):

    logging.info('Getting asset data for %s', date)
    # add domestic CN and HK share type to the list of common asset type and other types like REIT and InvT
    commonStockTypes_incl_CN_HK = commonShareTypes + localChineseAssetTypes + intlChineseAssetTypes     

    # Load asset info data
    assetInfo = getAssetCompanyMapping(date, subids, modelDB, marketDB)
        
    # Drop companies with only one issue
    multiListings = assetInfo[assetInfo.duplicated(subset=['CompanyID'], keep=False)].index
    assetInfo = assetInfo.loc[multiListings, :]

    # filter by excludeFromMCap
    assetInfo_filter = assetInfo.query("excludeFromMCap==False").copy()

    # deal with CN, HK by combining them into one ctyISO as CN_HK
    assetInfo_filter['ctyISO'].replace({'CN': 'CN_HK', 'HK': 'CN_HK'}, inplace=True)

    # First deal with common shares type of assets
    assetInfo_filter = assetInfo_filter.query("AssetType in @commonStockTypes")

    # Remove stock connect assets
    assetInfo_filter = assetInfo_filter.query("MarketType not in @connectExchanges")

    return assetInfo_filter

def getValidSiblings(date, subids, modelDB, marketDB):

    logging.info('Getting sibling data for %s', date)
    fixList = ['CIMPBBUP7'] # For now just fix Google until we've done more investigation
    fixList = None

    # Load asset info data
    assetInfo = getAssetCompanyMapping(date, subids, modelDB, marketDB)

    # deal with CN, HK by combining them into one ctyISO as CN_HK
    assetInfo['ctyISO'].replace({'CN': 'CN_HK', 'HK': 'CN_HK'}, inplace=True)

    # Drop companies with only one issue
    multiListings = set(assetInfo[assetInfo.duplicated(subset=['CompanyID'], keep=False)].index)

    # Get DR to underlying map
    valid_pool = set(assetInfo.query("excludeFromMCap==False").copy().index)
    valid_pool = set(assetInfo.loc[valid_pool, :].query("MarketType not in @connectExchanges").index)
    valid_pool = valid_pool.intersection(multiListings)
    validAssetInfo = assetInfo.loc[valid_pool, :]
    drSet = set(validAssetInfo.query("AssetType in @drAssetTypes").index)

    sid2sib = dict(zip(subids, [[sid] for sid in subids]))
    for sid in multiListings:

        # Find siblings - same CID and same market
        cid = assetInfo.loc[sid, 'CompanyID']
        if (fixList is not None) and (cid not in fixList):
            continue
        mkt = assetInfo.loc[sid, 'ctyISO']
        siblings = set(validAssetInfo.query("CompanyID==@cid").index)
        siblings = set(validAssetInfo.loc[siblings, :].query("ctyISO==@mkt").index)

        # If DRs and other asset types, drop the DRs
        nonDrs = siblings.difference(drSet)
        if len(nonDrs) > 0:
            siblings = nonDrs

        # If any valid siblings left, assign them to the asset in question
        if len(siblings) > 0:
            sid2sib[sid] = list(siblings)

    return sid2sib

def loadRiskModelGroupAssetMap(date, base_universe, modelDB, rmgList=None, quiet=False):
    """Returns a dict mapping risk model group IDs to a set of assets trading in that country/market.
    base_universe is the list of assets that will be considered for processing.
    If addNonModelRMGs is True then the risk model groups for assets trading outside rmgList are added to the map.
    Otherwise those assets won't be mapped.
    """
    # XXX - Todo: something similar for currencies
    # Initialise
    logging.debug('loadRiskModelGroupAssetMap: begin')
    rmgAssetMap = dict()
    universe = set(base_universe)
    if rmgList is None:
        rmgList = modelDB.getAllRiskModelGroups()

    # Loop through RMGs and assign assets to each
    for rmg in rmgList:
        rmg_assets = set(modelDB.getActiveSubIssues(rmg, date)).intersection(universe)
        rmgAssetMap[rmg] = rmg_assets
        logging.debug('%d assets in %s (RiskModelGroup %s, %s)',
                len(rmg_assets), rmg.description, str(rmg.rmg_id), rmg.mnemonic)

    # Check that all assets from universe have been mapped
    nRMGAssets = sum([len(assets) for (rmg, assets) in rmgAssetMap.items()])
    logging.debug('%d assets in %d RMGs', nRMGAssets, len(rmgList))
    if len(universe) != nRMGAssets:
        if not quiet:
            logging.info('%d assets not assigned to a model RMG', len(universe)-nRMGAssets)

    logging.debug('loadRiskModelGroupAssetMap: end')
    return rmgAssetMap

class AssetProcessor:
    """ Class for processing important asset information
    """
    def __init__(self, date, modelDB, marketDB, apParameters=dict()):
        # Important run-time parameters
        self.apParameters = apParameters
        self.date = date
        # DB info
        self.modelDB = modelDB
        self.marketDB = marketDB
        # Basic data
        self.universe = None
        self.marketCaps = None
        self.nurseryUniverse = set()
        self.originalNurseryUniverse = set()
        # Asset/RMG data
        self.rmgAssetMap = None
        self.tradingRmgAssetMap = None
        # Foreign listing data
        self.foreign = set()
        self.drCurrData = None

    def outputDebuggingInfo(self):
        return self.apParameters.get('debugOutput', False)

    def forceRun(self):
        return self.apParameters.get('forceRun', False)

    def useLegacyDates(self):
        return self.apParameters.get('legacyMCapDates', False)

    def getNumeraireID(self, dt):
        return self.marketDB.getCurrencyID(self.getNumeraireISO(), dt)

    def getNumeraireISO(self):
        return self.apParameters.get('numeraire_iso', 'USD')

    def getTrackList(self):
        return self.apParameters.get('trackList', [])

    def runQuietly(self):
        return self.apParameters.get('quiet', False)

    def getRMGOverride(self):

        # Create mapping from subissue to new RMG
        sid2newRMGDict = dict()
        overDict = self.apParameters.get('rmgOverride', None)

        if overDict is not None:
            # Loop round company/sub-issue
            for (cid, (dt0, dt1, iso1, iso2)) in overDict.items():
                if dt0 <= self.date < dt1:
                    rmg = self.modelDB.getRiskModelGroupByISO(iso1)
                    sid2cidMap = self.modelDB.getCompanySubIssues(self.date, [cid], self.marketDB)     
                    if len(sid2cidMap) > 0:
                        # If a company, assign ISO to all its sub-issues
                        for sid in sid2cidMap.keys():
                            sid2newRMGDict[sid] = rmg.mnemonic
                    else:
                        # Otherwise assume it's a single sub-issue
                        sid = ModelDB.SubIssue(string=cid)
                        sid2newRMGDict[sid] = rmg.mnemonic
            
        return sid2newRMGDict

    def getNurseryRMGs(self):
        return self.apParameters.get('nurseryRMGs', [])

    def checkHomeCountry(self):
        return self.apParameters.get('checkHomeCountry', False)

    def getModelAssetMaster(self, rm):
        """Determine the universe for the current model instance.
        The list of asset IDs is sorted.
        """
        # Initialise
        modelDB = self.modelDB
        marketDB = self.marketDB

        # Set up numeraire codes
        if hasattr(rm, 'numeraire'):
            numeraire_iso = rm.numeraire.currency_code
            numeraire_id = rm.numeraire.currency_id
        else:
            numeraire_iso = self.getNumeraireISO()
            numeraire_id = self.getNumeraireID(self.date)
        logging.info('Risk model numeraire is %s (ID %d)', numeraire_iso, numeraire_id)

        # Get all securities marked for this model at the current date
        universe = set(modelDB.getMetaEntity(marketDB, self.date, rm.rms_id, rm.primaryIDList))
        logging.info('Initial universe list: %d assets', len(universe))

        # Get asset types
        assetTypeDict = get_asset_info(self.date, universe, modelDB, marketDB, 'ASSET TYPES', 'Axioma Asset Type')

        # Load market caps, convert to model numeraire
        self.setMarketCaps(rm.rmg, subIssues=list(universe))

        # Remove assets with missing market cap
        missingCapIdx = self.marketCaps[self.marketCaps.isnull()].index
        if len(missingCapIdx) > 0:
            logging.warning('%d assets dropped due to missing market cap', len(missingCapIdx))
            universe = universe.difference(set(missingCapIdx))

        # Remove assets with missing asset type
        missingAT = [sid for sid in universe if assetTypeDict.get(sid, None) is None]
        if len(missingAT) > 0:
            universe = universe.difference(set(missingAT))
            logging.warning('%d assets dropped due to missing asset Type', len(missingAT))

        # Identify non-equity ETFs
        etf = set([sid for sid in universe if assetTypeDict.get(sid,None) in ['NonEqETF', 'StatETF']])
        if len(etf) > 0:
            logging.info('Excluding %d non-equity asset ETFs from universe', len(etf))
            universe = universe.difference(etf)

        # Identify composite EFTs
        etf2 = set([sid for sid in universe if assetTypeDict.get(sid,None) == 'ComETF'])
        if len(etf2) > 0:
            logging.info('Excluding %d composite ETFs from universe', len(etf))
            universe = universe.difference(etf2)
            etf = etf.union(etf2)

        # Remove assets w/o industry membership (excluding ETFs)
        missingInd = set()
        if rm.industryClassification is not None:
            assert(len(list(rm.industryClassification.getLeafNodes(modelDB).values())) > 0)
            leaves = rm.industryClassification.getLeafNodes(modelDB)
            factorList = [i.description for i in leaves.values()]
            exposures = rm.industryClassification.getExposures(
                        self.date, list(universe), factorList, modelDB, returnDF=True)
            if oldPD:
                indSum = exposures.replace(0.0, numpy.nan).sum(axis=1)
            else:
                indSum = exposures.replace(0.0, numpy.nan).sum(axis=1, min_count=1)
            missingInd = set(indSum[indSum.isnull()].index)

            # Report on assets (other than ETFs) with missing industries
            if len(missingInd) > 0:
                universe = universe.difference(missingInd)
                logging.warning('%d assets dropped due to missing %s classification',
                        len(missingInd), rm.industryClassification.name)

        # Make special case for ETFs in stat models
        if rm.allowETFs:
            logging.info('Adding %d non-equity asset ETFs back into universe', len(etf))
            universe = universe.union(etf)

        if self.outputDebuggingInfo():
            dict_to_csv('tmp/AssetTypeMap-%s.csv' % rm.mnemonic, assetTypeDict)
            list_to_csv('tmp/AssetTypes-%s.csv' % rm.mnemonic, set(assetTypeDict.values()))
            list_to_csv('tmp/MissingAssetType-%s.csv' % self.date, missingAT)
            list_to_csv('tmp/NonEqETF.csv', etf)
            list_to_csv('tmp/ComETF.csv', etf2)                    
            list_to_csv('tmp/missingInd.csv', missingInd)

        # Return sorted list of sub-issues
        base_universe = sorted(universe)
        self.marketCaps = self.getMarketCaps(base_universe)
        if len(base_universe) < 1:   
            raise Exception('No assets in the universe')
        logging.info('%d assets in the universe', len(base_universe))
        return base_universe

    def setMarketCaps(self, rmgList, subIssues=None):
        """ Function to retrieve market caps by one means or another
        """
        # Initialise
        if subIssues is None:
            subIssues = list(self.universe)

        # Load average market caps
        if self.useLegacyDates():
            mcapDates = self.modelDB.getDates(rmgList, self.date, 20, excludeWeekend=True, fitNum=True)
            self.marketCaps = self.modelDB.getAverageMarketCaps(\
                    mcapDates, subIssues, self.getNumeraireID(self.date), self.marketDB, returnDF=True)
        else:
            currency_id = self.getNumeraireID(self.date)
            self.marketCaps = robustLoadMCaps(self.date, subIssues, currency_id, self.modelDB, self.marketDB)
        return

    def getMarketCaps(self, subIssues=None):
        """ Function that returns the market caps for a given set of subissues
        or the existing universe if not specified.
        """
        # Initialise
        if subIssues is None:
            subIssues = list(self.universe)
        return self.marketCaps.reindex(index=subIssues)

    def downWeightMarketCaps(self, rmgList, rmgAssetMap=None):
        """ Downweights asset market caps for selected markets
        """
        if hasattr(self, 'alreadyDownWeighted') and self.alreadyDownWeighted:
            return
        if rmgAssetMap is None:
            rmgAssetMap = self.rmgAssetMap

        # Loop through markets and downweight some
        for rmg in [r for r in rmgList if (hasattr(r, 'downWeight') and (r.downWeight < 1.0))]:
            logging.info('Downweighting %s market caps by factor of %.2f%%', rmg.mnemonic, rmg.downWeight*100)
            self.marketCaps.loc[rmgAssetMap[rmg]] *= rmg.downWeight

        # Pull out all assets with valid mcaps (for total mcap)
        allSidCompanyMap = self.modelDB.getCompanySubIssues(\
                self.date, set(self.getCid2SubIssueMapping().keys()), self.marketDB)
        okSids = set([sid for (sid, vals) in allSidCompanyMap.items() if not vals[1]])

        # Find market cap per trading RMG
        rmgCap = dict()
        totalMCap = 0.0
        for rmg in rmgList:
            mktCap = 0.0
            overLap = okSids.intersection(set(self.tradingRmgAssetMap[rmg]))
            if len(overLap) > 0:
                mktCap = self.marketCaps[overLap].sum(axis=None)
            if hasattr(rmg, 'mktCapCap') and (rmg.mktCapCap != 1.0):
                rmgCap[rmg] = mktCap
            else:
                totalMCap += mktCap

        # Downweight anything on the "naughty list" - use trading RMG rather than assigned home market
        self.mktCapDownWeight = dict()
        for rmg in rmgCap.keys():
            ratio = rmgCap[rmg] / totalMCap
            if ratio > rmg.mktCapCap:
                sidList = list(set(self.tradingRmgAssetMap[rmg]))
                if len(sidList) > 0:
                    mktCapDownWeight = totalMCap * rmg.mktCapCap / rmgCap[rmg]
                    self.mktCapDownWeight[rmg] = (1.0 - rmg.blendFactor) + (rmg.blendFactor * mktCapDownWeight)
                    logging.info('RMG %s has fraction %.6f of total market cap against target of %.6f',
                                    rmg.mnemonic, ratio, rmg.mktCapCap)
                    logging.info('Scaling by factor of %.4f', self.mktCapDownWeight[rmg])
                    self.marketCaps[self.tradingRmgAssetMap[rmg]] = self.mktCapDownWeight[rmg] * self.marketCaps[sidList].values
        self.alreadyDownWeighted = True
        return

    def applyFreeFloatToMCap(self, marketCaps, days_avg=30):
        """Applies free-float adjustment to series of market caps
        """
        mcapDates = self.modelDB.getAllRMGDateRange(self.date, days_avg)
        ff_marketCaps = self.modelDB.applyFreeFloatOnMCap(\
                self.marketCaps.values, mcapDates, list(self.marketCaps.index), self.marketDB)
        return pandas.Series(ff_marketCaps, index=marketCaps.index)

    def getFreeFloatMarketCap(self, marketCaps=None):
        """ Compute free-float adjusted market caps
        """
        if marketCaps is None:
            marketCaps = self.marketCaps
        if hasattr(self,'freeFloat_marketCaps') and len(self.freeFloat_marketCaps) > 0:
            return self.freeFloat_marketCaps
        self.freeFloat_marketCaps = self.applyFreeFloatToMCap(self.date, marketCaps)
        return self.freeFloat_marketCaps

    def getISINMap(self, universe=None):
        """Mapping from sub-issue to ISIN
        """
        # Initialise
        if hasattr(self, 'sid2IsinMap') and len(self.sid2IsinMap) > 0:
            return self.sid2IsinMap
        if universe is None:
            universe = list(self.universe)

        # Load mapping
        modelIds = [sid.getModelID() for sid in universe]
        modelId2ItemMap = self.modelDB.getIssueISINs(self.date, modelIds, self.marketDB)
        self.sid2IsinMap = dict()
        for (mid, sid) in zip(modelIds, universe):
            self.sid2IsinMap[sid] = modelId2ItemMap.get(mid, None)
        return self.sid2IsinMap

    def getSEDOLMap(self, universe=None):
        """Mapping from sub-issue to SEDOL
        """
        # Initialise
        if hasattr(self, 'sid2SedolMap') and len(self.sid2SedolMap) > 0:
            return self.sid2SedolMap
        if universe is None:
            universe = list(self.universe)

        # Load mapping
        modelIds = [sid.getModelID() for sid in universe]
        modelId2ItemMap = self.modelDB.getIssueSEDOLs(self.date, modelIds, self.marketDB)
        self.sid2SedolMap = dict()
        for (mid, sid) in zip(modelIds, universe):
            self.sid2SedolMap[sid] = modelId2ItemMap.get(mid, None)
        return self.sid2SedolMap

    def getCUSIPMap(self, universe=None):
        """Mapping from sub-issue to CUSIP
        """
        # Initialise
        if hasattr(self, 'sid2CusipMap') and len(self.sid2CusipMap) > 0:
            return self.sid2CusipMap
        if universe is None:
            universe = list(self.universe)

        # Load mapping
        modelIds = [sid.getModelID() for sid in universe]
        modelId2ItemMap = self.modelDB.getIssueCUSIPs(self.date, modelIds, self.marketDB)
        self.sid2CusipMap = dict()
        for (mid, sid) in zip(modelIds, universe):
            self.sid2CusipMap[sid] = modelId2ItemMap.get(mid, None)
        return self.sid2CusipMap

    def getTickerMap(self, universe=None):
        """Mapping from sub-issue to Ticker
        """
        # Initialise
        if hasattr(self, 'sid2TickerMap') and len(self.sid2TickerMap) > 0:
            return self.sid2TickerMap
        if universe is None:
            universe = list(self.universe)

        # Load mapping
        modelIds = [sid.getModelID() for sid in universe]
        modelId2ItemMap = self.modelDB.getIssueTickers(self.date, modelIds, self.marketDB)
        self.sid2TickerMap = dict()
        for (mid, sid) in zip(modelIds, universe):
            self.sid2TickerMap[sid] = modelId2ItemMap.get(mid, None)
        return self.sid2TickerMap

    def getNameMap(self, universe=None):
        """Mapping from sub-issue to Name
        """
        # Initialise
        if hasattr(self, 'sid2NameMap') and len(self.sid2NameMap) > 0:
            return self.sid2NameMap
        if universe is None:
            universe = list(self.universe)

        # Load mapping
        modelIds = [sid.getModelID() for sid in universe]
        modelId2ItemMap = self.modelDB.getIssueNames(self.date, modelIds, self.marketDB)
        self.sid2NameMap = dict()
        for (mid, sid) in zip(modelIds, universe):
            self.sid2NameMap[sid] = modelId2ItemMap.get(mid, '')
        return self.sid2NameMap

    def getCloneMap(self, universe=None, cloneType='hard'):
        """ Pick up dict of assets to be cloned from others
        """
        # Initialise
        type2IDMap = {'hard': 1, 'coint': 2, 'soft': 3}
        if hasattr(self, 'cloneMap'):
            if (cloneType in self.cloneMap) and (len(self.cloneMap[cloneType]) > 0):
                return self.cloneMap[cloneType]
        else:
            self.cloneMap = dict()
        if universe is None:
            universe = list(self.universe)

        # Load mapping
        type2IDMap = {'hard': 1, 'coint': 2, 'soft': 3} 
        self.cloneMap[cloneType] = self.modelDB.getClonedMap(
                self.date, universe, linkageType=type2IDMap.get(cloneType.lower(), 'hard'))

        # Check for suspect mappings
        clMap = self.cloneMap[cloneType]
        overlap = set(clMap.keys()).intersection(set(clMap.values()))
        for sid1 in overlap:
            master1 = clMap[sid1]
            clone2 = [cln for cln in clMap if clMap[cln]==sid1]
            logging.error('Clone: %s mapped to master: %s',
                    sid1 if isinstance(sid1, str) else sid1.getSubIDString(), master1 if isinstance(master1, str) else master1.getSubIDString())
            for sid2 in clone2:
                logging.error('... and %s is mapped to master: %s',
                        master1 if isinstance(master1, str) else master1.getSubIDString(), sid2 if isinstance(sid2, str) else sid2.getSubIDString())
        assert len(overlap) < 1

        return self.cloneMap[cloneType]

    def getSubIssueGroups(self, universe=None):
        """ Get mapping of CID to its sub-issue children (if there are more than one)
        """
        if hasattr(self, 'subIssueGroups') and len(self.subIssueGroups) > 0:
            return self.subIssueGroups
        if universe is None:
            universe = list(self.universe)
        self.subIssueGroups = self.modelDB.getIssueCompanyGroups(self.date, universe, self.marketDB)
        return self.subIssueGroups

    def getCid2SubIssueMapping(self, universe=None):
        """ Get mapping from each CID to its sub-issue child or children
        """
        if hasattr(self, 'cid2sidMap') and len(self.cid2sidMap) > 0:
            return self.cid2sidMap
        if universe is None:
            universe = list(self.universe)
        self.cid2sidMap = self.modelDB.getIssueCompanyGroups(self.date, universe, self.marketDB, mapAllIssues=True)
        return self.cid2sidMap

    def getSubIssue2CidMapping(self, universe=None):
        """ Get mapping from each subissue to its CID
        """
        if hasattr(self, 'sid2cidMap') and len(self.sid2cidMap) > 0:
            return self.sid2cidMap
        if universe is None:
            universe = list(self.universe)
        self.sid2cidMap = self.modelDB.getIssueCompanies(self.date, universe, self.marketDB, keepUnmapped=True)
        return self.sid2cidMap

    def getLinkedCompanyGroups(self, universe=None):
        """ Get list of lists of linked company IDs
        """
        # Initialise
        if hasattr(self, 'companyIDGroups') and len(self.companyIDGroups) > 0:
            return self.companyIDGroups
        if universe is None:
            universe = list(self.universe)

        # Create lists of linked companies (e.g. DLCs)
        linkedCIDMap = list(extraLinkedList)
        linkedCIDMap += self.modelDB.getDLCs(self.date, self.marketDB)

        # Sift out entities not in the model
        allCIDs = set(self.getSubIssue2CidMapping().values())
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
        
        self.companyIDGroups = tmpMap
        return self.companyIDGroups

    def getAssetType(self, universe=None):
        """ Get mapping from each subissue to its asset type
        """
        if universe is None:
            universe = list(self.universe)
        if hasattr(self, 'assetTypeDict') and len(self.assetTypeDict) > 0:
            return self.assetTypeDict
        self.assetTypeDict = get_asset_info(
                self.date, universe, self.modelDB, self.marketDB, 'ASSET TYPES', 'Axioma Asset Type')
        if not self.forceRun():
            assert(len(universe)==len(self.assetTypeDict.keys()))
        return self.assetTypeDict

    def getTypeToAsset(self, universe=None):
        """ Get mapping from asset type to set of sub-issues
        """
        if universe is None:
            universe = list(self.universe)
        if hasattr(self, 'type2AssetDict') and len(self.type2AssetDict) > 0:
            return self.type2AssetDict
        assetTypeDict = self.getAssetType(universe)
        self.type2AssetDict = defaultdict(set)
        for sid, aType in assetTypeDict.items():
            self.type2AssetDict[aType].add(sid)
        return self.type2AssetDict

    def getSPACs(self, universe=None):
        """ Get mapping from each subissue to its asset type
        """
        if universe is None:
            universe = list(self.universe)
        if hasattr(self, 'spacList') and len(self.spacList) > 0:
            return self.spacList
        self.spacList = sort_spac_assets(
                self.date, universe, self.modelDB, self.marketDB)
        return self.spacList

    def getMarketType(self, universe=None):
        """ Get mapping from each subissue to its market code
        """
        if universe is None:
            universe = list(self.universe)
        if hasattr(self, 'marketTypeDict') and len(self.marketTypeDict) > 0:
            return self.marketTypeDict
        self.marketTypeDict = get_asset_info(
                self.date, universe, self.modelDB, self.marketDB, 'REGIONS', 'Market')
        missing = set(universe).difference(set(self.marketTypeDict.keys()))
        if len(missing) > 0:
            if self.forceRun():
                logging.warn('The following assets have no market type: %s',\
                        ','.join([sid.getSubIDString() for sid in missing]))
            else:
                logging.error('The following assets have no market type: %s',\
                        ','.join([sid.getSubIDString() for sid in missing]))
        if not self.forceRun():
            assert(len(universe)==len(self.marketTypeDict.keys()))
        return self.marketTypeDict

    def getMktTypeToAsset(self, universe=None):
        """ Get mapping from market type to set of sub-issues
        """
        if universe is None:
            universe = list(self.universe)
        if hasattr(self, 'mkt2AssetDict') and len(self.mkt2AssetDict) > 0:
            return self.mkt2AssetDict
        mktTypeDict = self.getMarketType(universe)
        self.mkt2AssetDict = defaultdict(set)
        for sid, mType in mktTypeDict.items():
            self.mkt2AssetDict[mType].add(sid)
        return self.mkt2AssetDict

    def getAssetAge(self, universe=None):
        """ Get mapping from each subissue to its age in days
        """
        if hasattr(self, 'ageDict') and len(self.ageDict) > 0:
            return self.ageDict

        if universe is None:
            universe = list(self.universe)

        # Load asset from dates
        issueFromDates = Utilities.load_ipo_dates(\
                self.date, universe, self.modelDB, self.marketDB, returnList=True, exSpacAdjust=True)
        age = [int((self.date-dt).days) for dt in issueFromDates]
        self.ageDict = dict(zip(universe, age))
        return self.ageDict

    def getAssetFromDt(self, universe=None):
        """ Get mapping from each subissue to its age in days
        """
        if hasattr(self, 'ipoDict') and len(self.ipoDict) > 0:
            return self.ipoDict

        if universe is None:
            universe = list(self.universe)

        # Load asset from dates
        issueFromDates = Utilities.load_ipo_dates(\
                self.date, universe, self.modelDB, self.marketDB, returnList=True, exSpacAdjust=True)
        self.ipoDict = dict(zip(universe, issueFromDates))
        return self.ipoDict

    def getCurrencyAssetMap(self, rmgAssetMap=None):
        """ Get mapping from each currency to its associated subissues
        """
        if hasattr(self, 'currencyAssetMap') and len(self.currencyAssetMap) > 0:
            return self.currencyAssetMap

        if rmgAssetMap is None:
            rmgAssetMap = self.rmgAssetMap

        # Get a mapping from asset to currency
        currencyAssetMap = defaultdict(set)
        for (rmg, sidList) in rmgAssetMap.items():
            isoCode = rmg.getCurrencyCode(self.date)
            currencyAssetMap[isoCode] = currencyAssetMap[isoCode].union(sidList)
        self.currencyAssetMap = currencyAssetMap

        return self.currencyAssetMap

    def getAssetInfoDF(self):
        # Set up DataFrame of important information
        nursUniv_dict = {sid: True for sid in self.nurseryUniverse}
        originalNUniv_dict = {sid: True for sid in self.originalNurseryUniverse}
        foreign_dict = {sid: True for sid in self.foreign}
        asset_rmg_map_dict = Utilities.flip_dict_of_lists(self.rmgAssetMap)
        asset_tradRmg_map_dict = Utilities.flip_dict_of_lists(self.tradingRmgAssetMap)
        assetTradingCurrencyMap = self.modelDB.getTradingCurrency(self.date, self.universe, self.marketDB)
        mkt_df = get_all_markets(self.universe, self.date, self.modelDB, self.marketDB).to_dict()
        fromDt_dict = self.getAssetFromDt()

        masterDic = {
                'nursery_universe': nursUniv_dict,
                'trading_Currency': assetTradingCurrencyMap,
                'Rmg': asset_rmg_map_dict,
                'tradingRmg': asset_tradRmg_map_dict,
                'DR_Currency': self.drCurrData,
                'Orig_NurseryUniverse': originalNUniv_dict,
                'Foreign': foreign_dict,
                'from_dt': fromDt_dict,
                'marketCap': self.marketCaps,
                'Home': mkt_df['Home1'],
                'Home2': mkt_df['Home2'],
                'Trade': mkt_df['Trade']
                }
        output = pandas.DataFrame.from_dict(masterDic)
        output.index = [x.getSubIDString() for x in output.index]
        return output

    def getDr2UnderMap(self, universe=None):
        """ Build mapping from each DR-type asset to its underlying
        """
        if not self.checkHomeCountry():
            return dict()
        if hasattr(self, 'dr2Underlying') and len(self.dr2Underlying) > 0:
            return self.dr2Underlying
        if universe is None:
            universe = list(self.universe)
        assetTypeDict = self.getAssetType(universe)

        # Set up default mapping - every DR-type mapped to None, except Thai F-Stocks, which are not true DRs
        drList = [sid for (sid, atype) in assetTypeDict.items() if atype in drAssetTypes]
        drNoFStock = [sid for sid in drList if assetTypeDict.get(sid, None) != 'FStock']
        dr2Under = dict((sid, None) for sid in drNoFStock)

        # Pull out the DR to underlying mapping from the DB and update the default mapping
        dr2UnderDB = self.modelDB.getDRToUnderlying(self.date, universe, self.marketDB)
        notDR = set(dr2UnderDB.keys()).difference(set(drList))
        if len(notDR) > 0:
            # Remove any non-DRs that have sneaked in
            logging.info('Removing %d assets in dr2Underlying that are not actually DRs', len(notDR))
            for sid in notDR:
                del dr2UnderDB[sid]
        dr2Under.update(dr2UnderDB)

        # Check for 3-way mappings and unpick any that exist
        underSet = set(dr2Under.values())
        underSet.discard(None)
        drSet = set([dr for (dr, under) in dr2Under.items() if under is not None])
        threeWays = drSet.intersection(underSet)

        # Loop round assets that occur as both 'dr' and 'under'
        for sid in threeWays:
            mappedTo = dr2Under[sid]
            underFor = [dr for dr in drSet if (dr2Under[dr] == sid)]
            logging.debug('DR: %s (%s) mapped to underlying: %s (%s)',
                    sid if isinstance(sid, str) else sid.getSubIDString(), assetTypeDict.get(sid, None),
                    mappedTo if isinstance(mappedTo, str) else mappedTo.getSubIDString(), assetTypeDict.get(mappedTo, None))
            for sid2 in underFor:
                logging.debug('... and is itself underlying for DR: %s, (%s)',
                        sid2 if isinstance(sid2, str) else sid2.getSubIDString(), assetTypeDict.get(sid2, None))
                logging.debug('... remapped to %s (%s)',
                        mappedTo if isinstance(mappedTo, str) else mappedTo.getSubIDString(), assetTypeDict.get(mappedTo, None))
                dr2Under[sid2] = mappedTo

        # Remove any instances where an asset is mapped to itself
        dropList = []
        for (dr, under) in dr2Under.items():
            if (dr is not None) and (under is not None) and (dr == under):
                dropList.append(dr)
        for dr in dropList:
            dr2Under.pop(dr)

        # Report on overall numbers
        drNoUnder = [dr for (dr, under) in dr2Under.items() if under is None]
        if len(drNoUnder) > 0:
            logging.info('%d DRs with underlying asset of None', len(drNoUnder))
        logging.info('%d DRs mapped to underlying or None', len(dr2Under))

        self.dr2Underlying = dr2Under
        return self.dr2Underlying

    def process_asset_information(self, rmgList, universe=None):
        """Outer Loop to process imporant asset data
        Builds mappings to market cap and RGMs/currencies
        """
        # Initialise
        if universe is None:
            universe = self.universe
        if len(universe) < 1:
            logging.warn('Empty universe supplied, exiting')
            return

        # Load market caps if we don't already have them
        if self.marketCaps is None:
            self.setMarketCaps(rmgList, universe)
        else:
            # If we have them, remap to the universe in case it's changed
            self.marketCaps = self.getMarketCaps(subIssues=universe)

        # Attempt to map subissues to RMGs
        missing = self.process_home_markets(rmgList, universe=universe)
        if len(missing) > 0:
            # If anything classed as missing, it is removed from the model
            logging.warning('%d assets dropped due to missing/invalid home country', len(missing))

        # Report on any zero/missing mcaps (there really shouldn't be any)
        self.marketCaps = self.getMarketCaps(subIssues=self.universe).replace(0.0, numpy.nan)
        missingIdx = list(self.marketCaps[self.marketCaps.isnull()].index)
        if len(missingIdx) > 0:
            missingIdx = [sid.getSubIDString() for sid in missingIdx]
            logging.error('Missing mcap for %s', ','.join(missingIdx))
            if not self.forceRun():
                assert (len(missingIdx) == 0)
        
        # Downweight market caps for select countries
        self.downWeightMarketCaps(rmgList)

        # Report on universe mcap
        logging.info('Universe total mcap: %.4f tn, Assets: %d',
                self.marketCaps.sum(axis=None) / 1.0e12, len(self.universe))

        return

    def process_home_markets(self, rmgList, universe=None):
        """Creates map of RiskModelGroup ID to list of its SubIssues
        Checks for assets belonging to non-model geographies.
        Assets will also be checked for home country classification,
        and removed from the universe if no valid classification is found.
        """
        logging.debug('process_home_markets: begin')
        logging.info('Initial model universe: %d assets', len(universe))

        # Initialise
        if universe is None:
            universe = set(self.universe)
        else:
            universe = set(universe)
        nurseryUniverse = set()
        originalNurseryUniverse = set()
        nurseryRMGs = self.getNurseryRMGs()
        foreign = set()

        # Set up mappings to RMG
        rmgAssetMap = loadRiskModelGroupAssetMap(self.date, universe, self.modelDB, rmgList=rmgList)
        assetRMGMap = Utilities.flip_dict_of_lists(rmgAssetMap)

        # Check for assets with no RMG
        mappedAssets = {sid for sidList in rmgAssetMap.values() for sid in sidList}
        unmappedAssets = universe.difference(mappedAssets)
        self.mappedTo = dict.fromkeys(mappedAssets, 0)

        # Stop here if not a regional model
        if not self.checkHomeCountry():
            self.tradingRmgAssetMap = loadRiskModelGroupAssetMap(self.date, mappedAssets, self.modelDB)
            self.rmgAssetMap = dict()
            for rmg in rmgAssetMap:
                self.rmgAssetMap[rmg] = set(rmgAssetMap[rmg]).intersection(mappedAssets)
            self.universe = sorted(mappedAssets)
            if self.outputDebuggingInfo():
                dict_to_csv('tmp/assetRMG-%s.csv' % self.date, assetRMGMap)
            return unmappedAssets

        # Make a note of assets in "nursery" markets
        validRMGs = set([r for r in rmgList if r not in nurseryRMGs])
        for rmg in nurseryRMGs:
            nurseryUniverse = nurseryUniverse.union(rmgAssetMap[rmg])
        if len(nurseryUniverse) > 0:
            logging.info('%d assets belong to nursery markets', len(nurseryUniverse))
            originalNurseryUniverse = set(nurseryUniverse)

        # Check for non-model geography assets
        if len(unmappedAssets) > 0:
            logging.info('%d assets from non-model geographies', len(unmappedAssets))

        # Load primary and secondary home country classification data
        clsData = get_home_country(universe, self.date, self.modelDB, self.marketDB, clsType='HomeCountry')
        clsData2 = get_home_country(universe, self.date, self.modelDB, self.marketDB, clsType='HomeCountry2')

        # Load any overrides for home country
        rmgOverride = self.getRMGOverride()
        validRmgISOs = [rmg.mnemonic for rmg in validRMGs]
        nurseryRmgISOs = [rmg.mnemonic for rmg in nurseryRMGs]
        iso2rmgMap = dict(zip([r.mnemonic for r in rmgList], rmgList))

        # Loop round assets and home countries - do some manipulation where necessary
        for (sid, homeCtyCls) in clsData.items():

            inNursery = False
            tradingCtyRMG = assetRMGMap.get(sid, None)
            if tradingCtyRMG is None:
                tradingCtyISO = None
            else:
                tradingCtyISO = tradingCtyRMG.mnemonic
            homeCtyISO = rmgOverride.get(sid, homeCtyCls.classification.code)
            checkList = ('Home', homeCtyISO, 2)

            if homeCtyISO not in validRmgISOs:

                # If primary home country is not a valid market, attempt to reassign
                homeCty2Cls = clsData2.get(sid, homeCtyCls)
                homeCty2ISO = rmgOverride.get(sid, homeCty2Cls.classification.code)

                # If secondary home country is legit, assign it there
                if homeCty2ISO in validRmgISOs:
                    checkList = ('Home2', homeCty2ISO, 1)
                    homeCtyISO = homeCty2ISO
                # If trading country is legit, assign it there
                elif tradingCtyRMG in validRMGs:
                    checkList = ('Trading', homeCtyISO, 0)
                    homeCtyISO = tradingCtyRMG.mnemonic
                    foreign.add(sid)
                # Otherwise check whether one of the markets is a nursery market
                elif homeCtyISO in nurseryRmgISOs:
                    checkList = ('Nursery', homeCtyISO, -1)
                    inNursery = True
                elif homeCty2ISO in nurseryRmgISOs:
                    checkList = ('Nursery2', homeCty2ISO, -2)
                    homeCtyISO = homeCty2ISO
                    inNursery = True
                elif tradingCtyRMG in nurseryRMGs:
                    checkList = ('Trading2', homeCtyISO, -3)
                    homeCtyISO = tradingCtyISO
                    foreign.append(sid)
                    inNursery = True

                # Drop if neither quotation nor secondary country is valid or nursery market
                else:
                    if sid in self.getTrackList():
                        logging.info('Asset %s classed as Missing, Home: %s, Trading market: %s, Foreign: %s',
                                sid.getSubIDString(), homeCtyISO, tradingCtyISO, homeCtyISO!=tradingCtyISO)
                    continue
                self.mappedTo[sid] = checkList[2]

            # Report on any assets we're tracking
            if sid in self.getTrackList():
                logging.info('Asset %s classed as %s, Home: %s, Trading market: %s, Foreign: %s',
                        sid.getSubIDString(), checkList[0], checkList[1], tradingCtyISO, checkList[1]!=tradingCtyISO)

            # Resolve any inconsistencies with nursery assets
            if sid in nurseryUniverse and not inNursery:
                logging.info('Moved nursery asset %s from %s to non-nursery %s',
                    sid.getSubIDString(), tradingCtyISO, homeCtyISO)
                nurseryUniverse.remove(sid)

            if inNursery and sid not in nurseryUniverse:
                logging.info('Asset %s classed as nursery, but not in nursery universe - fixing this', sid.getSubIDString())
                nurseryUniverse.add(sid)

            # If home country differs from RiskModelGroup, reassign
            if (homeCtyISO != tradingCtyISO):
                homeCtyRMG = iso2rmgMap[homeCtyISO]
                if tradingCtyRMG is not None:
                    rmgAssetMap[tradingCtyRMG].remove(sid)
                rmgAssetMap[homeCtyRMG].add(sid)
                foreign.add(sid)
                if inNursery:
                    logging.info('Foreign nursery asset %s reassigned from %s to %s',
                            sid.getSubIDString(), tradingCtyISO, homeCtyISO)

            if sid in self.getTrackList():
                logging.info('Asset %s classed as Final, Home: %s, Trading market: %s, Foreign: %s',
                    sid.getSubIDString(), homeCtyISO, tradingCtyISO, sid in foreign)

        # Get a mapping to trading currency
        fullTradingCurrencyMap = self.modelDB.getTradingCurrency(self.date, universe, self.marketDB)
        drCurrData = dict()
        for rmg, sidList in rmgAssetMap.items():

            # Get RMG's trading currency
            rmgISO = rmg.getCurrencyCode(self.date)
            rmgCurrencyID = self.marketDB.getCurrencyID(rmgISO, self.date)

            # Map 'foreign' listings to home currency
            fgn_assets = sidList.intersection(foreign)
            for sid in fgn_assets:
                drCurrData[sid] = rmgCurrencyID

            # Check for assets trading in the 'wrong' currency
            mismatch = [sid for sid in sidList if rmgISO != fullTradingCurrencyMap.get(sid, rmgISO)]
            mismatch = set(mismatch).difference(fgn_assets)
            if len(mismatch) > 0 and self.outputDebuggingInfo():
                logging.info('Mapping %d non-foreign %s listings with foreign currency back to %s',
                        len(mismatch), rmg.mnemonic, rmgISO)

            # Map any such mismatches to the 'correct' currency
            for sid in mismatch:
                drCurrData[sid] = rmgCurrencyID

        # Report on what has and hasn't been mapped
        mappedAssets = {sid for sidList in rmgAssetMap.values() for sid in sidList}
        reassigned = unmappedAssets.intersection(mappedAssets)
        if len(reassigned) > 0:
            logging.info('%d unmapped assets reassigned to RMG successfully', len(reassigned))
        unmappedAssets = universe.difference(mappedAssets)

        # Set up final structures
        self.tradingRmgAssetMap = loadRiskModelGroupAssetMap(self.date, mappedAssets, self.modelDB)
        self.rmgAssetMap = dict()
        for rmg in rmgAssetMap:
            self.rmgAssetMap[rmg] = set(rmgAssetMap[rmg]).intersection(mappedAssets)
        self.foreign = foreign.intersection(mappedAssets)
        for sid in unmappedAssets:
            if sid in drCurrData:
                del drCurrData[sid]
        self.drCurrData = drCurrData
        self.originalNurseryUniverse = originalNurseryUniverse.intersection(mappedAssets)
        self.nurseryUniverse = nurseryUniverse.intersection(mappedAssets)
        self.universe = sorted(mappedAssets)

        if self.outputDebuggingInfo():
            assetRMGMap = Utilities.flip_dict_of_lists(rmgAssetMap)
            dict_to_csv('tmp/assetRMG2-%s.csv' % self.date, assetRMGMap)
            infoDict = self.getAssetInfoDF()
            infoDict.to_csv('tmp/asset-info-%s.csv' % self.date)

        logging.debug('process_home_markets: end')
        return unmappedAssets

if __name__ == '__main__':
    from riskmodels import ModelDB
    from marketdb import MarketDB
    mdl = ModelDB.ModelDB(user='modeldb_global', passwd='modeldb_global', sid='glsdg')
    mkt = MarketDB.MarketDB(user='marketdb_global', passwd='marketdb_global', sid='glsdg')
    ex = AssetProcessor_V4(pars, mdl, mkt)
