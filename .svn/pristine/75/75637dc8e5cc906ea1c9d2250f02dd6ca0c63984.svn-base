from riskmodels import MarketIndex
from riskmodels import Matrices
from riskmodels.Matrices import ExposureMatrix
import numpy.ma as ma
import numpy
import logging
from riskmodels import LegacyUtilities as Utilities

def generate_currency_exposures(modelDate, modelSelector, modelDB, marketDB, 
                                data, simple=True):
    """Generate binary currency exposures.  If simple=True,
    assets are assigned unit exposure to the currency belonging
    to their country of quotation.  Otherwise, assets are
    assigned exposures based on their trading currency.
    """
    logging.debug('generate_currency_exposures: begin')
    if hasattr(data, 'rmgAssetMap'):
        rmgAssetMap = data.rmgAssetMap
        rmcAssetMap = data.rmcAssetMap
    else:
        rmgAssetMap = modelSelector.rmgAssetMap
        rmcAssetMap = modelSelector.rmcAssetMap

    expMatrix = data.exposureMatrix
    # Set up currency exposures array
    currencyExposures = Matrices.allMasked(
                    (len(modelSelector.currencies), len(data.universe)))
    currencyIdxMap = dict([(c.name, i) for (i,c) \
                    in enumerate(modelSelector.currencies)])
     
    if simple:
        # Populate currency factors, one country at a time
        for rmg in modelSelector.rmg:
            rmg_assets = list(rmcAssetMap[rmg.rmg_id])
            asset_indices = [data.assetIdxMap[n] for n in rmg_assets]
            pos = currencyIdxMap[rmg.currency_code]
            values = currencyExposures[pos,:]
            if len(asset_indices) > 0:
                ma.put(values, asset_indices, 1.0)
                currencyExposures[pos,:] = values
    else:
        # TODO: not yet implemented
        assert(False)

    # Insert currency exposures into exposureMatrix
    expMatrix.addFactors([c.name for c in modelSelector.currencies], 
                        currencyExposures, ExposureMatrix.CurrencyFactor)
    currencies = sorted(currencyIdxMap.keys())
    logging.info('Computed currency exposures for %d currencies: %s',
                len(list(currencyIdxMap.keys())), ','.join(currencies))

    logging.debug('generate_currency_exposures: end')
    return expMatrix

def generate_beta_country_exposures(modelDate, modelSelector, 
                                    modelDB, marketDB, data):
    """Compute beta country exposures based on country of quotation.
    """
    logging.debug('generate_beta_country_exposures: begin')
    expMatrix = data.exposureMatrix
    assets = expMatrix.getAssets()
    returns = data.returns
    betaDays = 120

    # Compute market betas for each risk model group
    datesIdxMap = dict(zip(returns.dates, range(len(returns.dates))))
    if not hasattr(modelSelector, 'indexSelector'):
        self.indexSelector = MarketIndex.\
                        MarketIndexSelector(modelDB, marketDB)

    for rmg in modelSelector.rmg:
        rmg_assets = modelSelector.rmgAssetMap[rmg.rmg_id]
        asset_indices = [data.assetIdxMap[n] for n in rmg_assets]
        rmgDates = [d for d in data.rmgCalendarMap[rmg.rmg_id][-betaDays:] \
                            if d >= returns.dates[0]]
        dt_indices = [datesIdxMap[d] for d in rmgDates]

        values = Matrices.allMasked(len(assets))
        logging.info('Computing market betas %s, %d assets (%s), %d days',
                    rmg.description, len(asset_indices), 
                    rmg.currency_code, len(rmgDates))

        # Compute betas for this market
        if len(asset_indices) > 0:
            rmgReturns = Matrices.TimeSeriesMatrix(rmg_assets, rmgDates)
            ret = ma.take(returns.data, asset_indices, axis=0)
            ret = ma.take(ret, dt_indices, axis=1)
            rmgReturns.data = ret

            mm = Utilities.run_market_model(
                    rmgReturns, rmg, modelDB, marketDB, betaDays,
                    sw=True, clip=False, indexSelector=self.indexSelector)

            # Insert values into exposure matrix
            beta = numpy.clip(mm[0], -0.2, 2.0)
            ma.put(values, asset_indices, beta)
        expMatrix.addFactor(rmg.description, values, ExposureMatrix.CountryFactor)

    logging.debug('generate_beta_country_exposures: end')
    return expMatrix

def generate_binary_country_exposures(modelDate, modelSelector, 
            modelDB, marketDB, data, rmgs=[]):
    """Assign unit exposure to the country of quotation.
    """
    logging.debug('generate_binary_country_exposures: begin')
    if hasattr(data, 'rmgAssetMap'):
        rmgAssetMap = data.rmgAssetMap
    else:
        rmgAssetMap = modelSelector.rmgAssetMap

    if len(rmgs) > 0:
        rmgList = [rmg for rmg in modelSelector.rmg if rmg in rmgs]
    else:
        rmgList = modelSelector.rmg

    expMatrix = data.exposureMatrix
    countryExposures = Matrices.allMasked(
                    (len(rmgList), len(data.universe)))
    rmgIdxMap = dict([(j,i) for (i,j) in enumerate(rmgList)])

    for rmg in rmgList:
        # Determine list of assets for each market
        rmg_assets = rmgAssetMap[rmg.rmg_id]
        indices = [data.assetIdxMap[n] for n in rmg_assets]
        logging.debug('Computing market exposures %s, %d assets (%s)',
                    rmg.description, len(indices), rmg.currency_code)
        values = Matrices.allMasked(len(data.universe))
        if len(indices) > 0:
            ma.put(values, indices, 1.0)
            countryExposures[rmgIdxMap[rmg],:] = values
    expMatrix.addFactors([r.description for r in rmgList],
                         countryExposures, ExposureMatrix.CountryFactor)

    logging.debug('generate_binary_country_exposures: end')
    return expMatrix

def generate_region_sector_exposures(modelDate, modelSelector, sectorNames, 
                                     modelDB, marketDB, expMatrix):
    logging.debug('generate_region_sector_exposures: begin')
    # Get exposures to sectors
    sectorExposures = modelSelector.industryClassification.\
            getExposures(modelDate, expMatrix.getAssets(), 
                        sectorNames, modelDB, marketDB, -1)
    
    # Create exposure matrices for each region
    rmgRegionMap = dict(zip([r.rmg_id for r in modelSelector.rmg], modelSelector.regions))
    regionSectorMap = {}
    for region in set(modelSelector.regions):
        regionSectorMap[region] = Matrices.allMasked(sectorExposures.shape)
    
    # Loop through markets, assign sector exposures by region
    allExposures = expMatrix.getMatrix()
    for (idx, rmg) in enumerate(modelSelector.rmg):
        # Assuming country exposures already present...
        rmgExposures = ma.take(allExposures, 
                    [expMatrix.getFactorIndex(rmg.description)], axis=0)
        rmg_indices = numpy.flatnonzero(ma.getmaskarray(rmgExposures)==0)
        rmg_exp = ma.take(sectorExposures, rmg_indices, axis=1)
        
        # Update sector exposures for region
        region_exp = regionSectorMap[rmgRegionMap[rmg.rmg_id]]
        for j in range(sectorExposures.shape[0]):
            region_sector_exp = region_exp[j,:]
            rmgSectorIndices = numpy.flatnonzero(ma.getmaskarray(rmg_exp[j,:])==0)
            values = ma.take(rmg_exp[j,:], rmgSectorIndices, axis=0)
            ma.put(region_sector_exp, numpy.take(rmg_indices, rmgSectorIndices, axis=0), values)
            regionSectorMap[rmgRegionMap[rmg.rmg_id]][j,:] = region_sector_exp
    
    # Update ExposureMatrix
    sectorNames = []
    for node in modelSelector.industryClassification.getLeafNodes(modelDB).values():
        name = modelDB.getMdlClassificationParent(node).name
        if name not in sectorNames:
            sectorNames.append(name)
    for region in set(modelSelector.regions):
        regionSectorNames = ['Region %d ' % region + f for f in sectorNames]
        expMatrix.addFactors(regionSectorNames, regionSectorMap[region], 'other')
    
    logging.debug('generate_region_sector_exposures: end')
    return expMatrix

def generate_EM_exposures(expMatrix, modelSelector, invert=False):
    """Assign 1/0 to all assets, 1 if it belongs to an EM country,
    zero otherwise.
    """
    logging.debug('generate_EM_exposures: begin')

    allExposures = expMatrix.getMatrix()
    values = Matrices.allMasked((allExposures.shape[1]))
    for rmg in modelSelector.rmg:
        if not rmg.developed:
            # Assuming country exposures already already present...
            rmgExposures = ma.take(allExposures, 
                        [expMatrix.getFactorIndex(rmg.description)], axis=0)
            rmg_indices = numpy.flatnonzero(ma.getmaskarray(rmgExposures)==0)
            ma.put(values, rmg_indices, 1.0)
    if invert:
        values = ma.getmaskarray(values)
        values = ma.masked_where(values==0, values)

    logging.debug('generate_EM_exposures: end')
    return values

def generate_china_domestic_exposures(data, modelDate, modelDB, marketDB):
    """Assigns unit exposure to this factor for all China A
    and B shares.
    """
    (aShares, bShares, hShares, intl) = MarketIndex.process_china_share_classes(
                                data, modelDate, modelDB, marketDB)
    domesticIdx = aShares + bShares
    values = Matrices.allMasked(len(data.universe))
    if len(domesticIdx) > 0:
        ma.put(values, domesticIdx, 1.0)
    data.exposureMatrix.addFactor(
        'Domestic China', values, ExposureMatrix.LocalFactor)
    return data.exposureMatrix

def generate_CDI_exposures(data, modelDate, modelDB, marketDB):
    """Assigns unit exposure to this factor for all CDI assets"""
    cdiAssets = [i for (i,j) in data.assetTypeDict.items() if j.strip() == 'CDI']
    cdiIdx = [data.assetIdxMap[sid] for sid in cdiAssets]
    values = Matrices.allMasked(len(data.universe))
    if len(cdiIdx) > 0:
        ma.put(values, cdiIdx, 1.0)
        logging.info('Found %s flagged as CDI'%len(cdiIdx))
    data.exposureMatrix.addFactor(
        'CDI', values, ExposureMatrix.StyleFactor)
    return data.exposureMatrix

def identify_top_china_a_shares(modelSelector, date, expMatrix, 
                                modelDB, marketDB, top=300):
    """Returns the index positions of the top N China A shares by
    total market cap as well as their square-root capitalization weights.
    """
    logging.debug('identify_top_china_a_shares: begin')
    mat = expMatrix.getMatrix()
    fIdx = expMatrix.getFactorIndex('Domestic China')
    indices = numpy.flatnonzero(ma.getmaskarray(mat[fIdx,:])==0)
    if len(indices)==0:
        return (list(), list())
    subids = [expMatrix.getAssets()[i] for i in indices]
    assetCurrencyMap = modelDB.getTradingCurrency(date, 
                        subids, marketDB, returnType='code')
    aShares = [sid for (sid,cur) in assetCurrencyMap.items() \
               if cur == 'CNY']
    mcapDates = modelDB.getDates(modelSelector.rmg, date, 19)
    avgMarketCaps = modelDB.getAverageMarketCaps(
            mcapDates, aShares, modelSelector.numeraire.currency_id, marketDB)
    indices_A = ma.argsort(-avgMarketCaps.filled(0.0))[:min(top, len(aShares))]
    weights_A = ma.sqrt(ma.take(avgMarketCaps, indices_A, axis=0))
    logging.info('Flagged top %d China A-shares, %.2f bn USD cap', 
                    top, ma.sum(weights_A**2.0) / 1e9) 
    assetIdxMap = dict([(j,i) for (i,j) in enumerate(subids)])
    indices_A = [indices[assetIdxMap[aShares[i]]] for i in indices_A]
    logging.debug('identify_top_china_a_shares: end')
    return (indices_A, weights_A)

def identify_top_taiwan_otc_shares(modelSelector, date, expMatrix, 
                                modelDB, marketDB, top=300):
    """Returns the index positions of the top N Taiwan OTC assets by
    total market cap as well as their square-root capitalization weights.
    """
    logging.debug('identify_top_china_a_shares: begin')
    mat = expMatrix.getMatrix()
    fIdx = expMatrix.getFactorIndex('OTC Market')
    indices = numpy.flatnonzero(ma.getmaskarray(mat[fIdx,:])==0)
    assert(len(indices) > 0)
    subids = [expMatrix.getAssets()[i] for i in indices]
    mcapDates = modelDB.getDates(modelSelector.rmg, date, 19)
    avgMarketCaps = modelDB.getAverageMarketCaps(mcapDates, subids, 
                                    modelSelector.numeraire.currency_id, marketDB)
    indices_OTC = ma.argsort(-avgMarketCaps.filled(0.0))[:min(top, len(subids))]
    weights_OTC = ma.sqrt(ma.take(avgMarketCaps, indices_OTC, axis=0))
    logging.info('Flagged top %d Taiwan OTC assets, %.2f bn USD cap', 
                    top, ma.sum(weights_OTC**2.0) / 1e9) 
    assetIdxMap = dict([(j,i) for (i,j) in enumerate(subids)])
    indices_OTC = [indices[assetIdxMap[subids[i]]] for i in indices_OTC]
    logging.debug('identify_top_china_a_shares: end')
    return (indices_OTC, weights_OTC)

def identify_cross_listings(univ, date, modelDB, marketDB,
                            homeRMG=None, secRMG=None):
    """Given a list of assets, attempt to locate any cross-listed
    entities using ISIN look-up.  ADRs and their parent/underlying
    securities, therefore, cannot be identified via this function.
    """
    logging.debug('identify_cross_listings: begin')
    # Determine pool of global assets from which to find x-listings
    if secRMG is not None:
        foreignSubIssues = modelDB.getActiveSubIssues(secRMG, date)
        tmp1 = secRMG.description
    else:
        foreignSubIssues = modelDB.getAllActiveSubIssues(date)
        foreignSubIssues = list(set(foreignSubIssues).difference(univ))
        tmp1 = 'foreign'
    issues = [n.getModelID() for n in univ]
    foreignIssues = [n.getModelID() for n in foreignSubIssues]
    logging.debug('Finding possible cross-listings from %d %s assets',
                len(foreignSubIssues), tmp1)

    # Find assets with same ISIN identifiers to ones in model universe
    homeISINMap = modelDB.getIssueISINs(date, issues, marketDB)
    foreignISINMap = modelDB.getIssueISINs(date, foreignIssues, marketDB)
    if homeRMG is not None:
        foreignISINMap = dict([(i,j) for (i,j) in \
                foreignISINMap.items() if j[:2]==homeRMG.mnemonic])
        logging.debug('Found %d foreign assets with %s ISINs',
                len(foreignISINMap), homeRMG.description)
        tmp2 = homeRMG.description
    else:
        tmp2 = ''
    mdl2sub = dict([(sid.getModelID(), sid) for sid in foreignSubIssues])
    isinAssetMap = dict([(j,mdl2sub[i]) for (i,j) in foreignISINMap.items()])
    xListPairs = [(sid, isinAssetMap[homeISINMap[sid.getModelID()]]) \
                    for sid in univ if homeISINMap.get(sid.getModelID()) in isinAssetMap]
    logging.debug('Found %d %s assets with %s cross-listings', len(xListPairs), tmp2, tmp1)
    logging.debug('identify_cross_listings: end')
    return xListPairs

def identify_depository_receipts(data, date, modelDB, marketDB, 
                                 restrictCls=None):
    """Given a list of SubIssues in data.universe, returns the 
    index positions of assets that are classified as ADR or GDR 
    in either the FTID or DataStream asset type classifications.
    Other DR-like instruments, such as CDIs in Australia,
    cannot be identified via this method.  data is assumed to be
    a Struct containing a universe attribute.
    If the optional restrictCls argument is given, only that asset
    type classification will checked for DRs.
    """
    logging.debug('identify_depository_receipts: begin')
    if not hasattr(data, 'assetIdxMap'):
        data.assetIdxMap = dict([(j,i) for (i,j) in enumerate(data.universe)])
    clsFamily = marketDB.getClassificationFamily('ASSET TYPES')
    assert(clsFamily is not None)
    clsMembers = dict([(i.name, i) for i in marketDB.\
                        getClassificationFamilyMembers(clsFamily)])
    indices = set()
    if restrictCls is None:
        restrictCls = list()
    for (clsMember, codes) in [
                    ('DataStream2 Asset Type', ('ADR', 'GDR')),
                    ('TQA FTID Global Asset Type', (
                        'MktLbl_3', # ADR
                        'MktLbl_4', # ADS
                        'MktLbl_5', # European DR
                        'MktLbl_6', # GDR
                        'MktLbl_7', # GDS
                        )),
                    ('TQA FTID Domestic Asset Type', ('A',))
                    ]:
        if clsMember not in restrictCls and len(restrictCls) > 0:
            continue
        cm  = clsMembers.get(clsMember, None)
        assert(cm is not None)
        clsRevision = marketDB.\
                getClassificationMemberRevision(cm, date)
        homeClsData = modelDB.getMktAssetClassifications(
                        clsRevision, data.universe, date, marketDB)
        codes = set(codes)
        idx = [data.assetIdxMap[sid] for (sid, cls) in homeClsData.items() \
                       if cls.classification.code in codes]
        logging.info('Found %d out of %d assets classified as DR',
                        len(idx), len(list(homeClsData.keys())))
        indices.update(idx)
    logging.info('Found total of %d out of %d assets classified as ADR/GDR',
                        len(indices), len(data.universe))
    logging.debug('identify_depository_receipts: end')
    return list(indices)

def getAssetHomeCurrencyID(univ, indices, expMatrix, modelDate, modelDB):
    """Given a list of assets whose index positions are
    specified by indices, returns a dict mapping their
    corresponding SubIssues to their home currency (as
    determined by their currency exposure) ID.
    """
    logging.debug('getAssetHomeCurrencyID: begin')
    currencyFactorNames = expMatrix.getFactorNames(ExposureMatrix.CurrencyFactor)
    currencyFactorIndices = expMatrix.getFactorIndices(ExposureMatrix.CurrencyFactor)
    assert(len(currencyFactorNames) > 0)
    assert(hasattr(modelDB, 'currencyCache'))
    currencyExposures = ma.getmaskarray(ma.take(expMatrix.getMatrix(), 
                            currencyFactorIndices, axis=0))
    currCodeIDMap = dict([(c, modelDB.currencyCache.getCurrencyID(
                           c, modelDate)) for c in currencyFactorNames])
    assetCurrIDMap = dict([(univ[i], currCodeIDMap[
                        currencyFactorNames[numpy.flatnonzero(
                        currencyExposures[:,i]==0)[0]]]) for i in indices])
    logging.debug('getAssetHomeCurrencyID: end')
    return (assetCurrIDMap, currCodeIDMap)

def computeTotalIssuerMarketCaps(
        data, date, numeraire, modelDB, marketDB, debugReport=False):
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

    # Get lists of DLC companies
    extraCompanies = list()

    # Find all SubIssues for those company IDs
    allSidCompanyMap = modelDB.getCompanySubIssues(date, companies, marketDB)

    # Group sids by company identifier
    companySidMap = dict()
    for (sid, (company, excludeFromMcap, ctyISO)) in allSidCompanyMap.items():
        if (sid not in sidCompanyMap) and (ctyISO not in ('HK', 'CN')):
            continue
        companySidMap.setdefault(company, list()).append((sid, excludeFromMcap, ctyISO))
    logging.info('%d issuers with multiple share lines or DRs', len(companySidMap))

    # Load market caps for sub-issues not present in universe
    extraSids = list(set([i[0] for j in companySidMap.values() for i in j])- set(sidList))
    logging.info('%d share lines reside outside model universe', len(extraSids))
    otherIdxMap = dict((j, i) for (i, j) in enumerate(extraSids))
    otherMarketCaps = numpy.zeros((len(extraSids),))
    if len(extraSids) > 0:
        # Fetch market caps by RiskModelGroup
        rmgSidMap = dict()
        isoRMGMap = dict([(r.mnemonic, r) for r in modelDB.getAllRiskModelGroups()])
        for sid in extraSids:
            rmgSidMap.setdefault(allSidCompanyMap[sid][2], list()).append(sid)
        for (rmg, rmgSids) in rmgSidMap.items():
            logging.info('Adding %d sub-issues from %s to compute issuer market cap', len(rmgSids), rmg)
            rmgSids.sort()
            mcapDates = modelDB.getDates([isoRMGMap[rmg]], date, 19)
            rmgMarketCaps = modelDB.getAverageMarketCaps(
                    mcapDates,rmgSids, numeraire.currency_id, marketDB).filled(0.0)
            rmgIndices = [otherIdxMap[sid] for sid in rmgSids]
            otherMarketCaps[rmgIndices] = rmgMarketCaps
    
    # Loop through companies
    totalIssuerMarketCaps = numpy.array(data.marketCaps)
    issuerMarketCapDict = dict()
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
            continue

        # Group linked SubIssues by RiskModelGroup
        rmgSidMap = dict()
        for (sid, excludeFromMcap, ctyISO) in sidVals:
            rmgSidMap.setdefault(ctyISO, list()).append((sid, excludeFromMcap, ctyISO))

        # Deal with Hong Kong / China
        if ('CN' in rmgSidMap) and ('HK' in rmgSidMap):
            allSids = [i for j in rmgSidMap.values() for i in j]
            rmgSidMap = dict()
            rmgSidMap['CN-HK'] = allSids

        # Loop through RiskModelGroups, sum up market caps
        runningMarketCap = 0.0
        for (rmg, sidValList) in rmgSidMap.items():
            if len(sidValList) == 1:
                sid = sidValList[0][0]
                if sid in data.assetIdxMap:
                    mcap = data.marketCaps[data.assetIdxMap[sid]]
                else:
                    mcap = otherMarketCaps[otherIdxMap[sid]]
                if mcap > runningMarketCap:
                    runningMarketCap = mcap
                continue
            excludedSids = len([i for i in sidValList if i[1]])
            if excludedSids == len(sidValList):
                logging.debug('All assets for company %s in %s are excluded', 
                            company, rmg)
                continue
            issuerMarketCap = 0.0
            for (sid, excludeFromMCap, ctyISO) in sidValList:
                if not excludeFromMCap:
                    if sid in data.assetIdxMap:
                        issuerMarketCap += data.marketCaps[data.assetIdxMap[sid]]
                    else:
                        issuerMarketCap += otherMarketCaps[otherIdxMap[sid]]
            if issuerMarketCap == 0.0:
                logging.warning('Company %s (%d sub-issues in %s) has zero issuer market cap', 
                            company, len(sidValList), rmg)
            for (sid, excludeFromMCap, ctyISO) in sidValList:
                if sid in data.assetIdxMap:
                    totalIssuerMarketCaps[data.assetIdxMap[sid]] = issuerMarketCap
            if issuerMarketCap > runningMarketCap:
                runningMarketCap = issuerMarketCap
        
        issuerMarketCapDict[company] = runningMarketCap

    # Get lists of DLC companies
    data.DLCMarketCap = numpy.array(totalIssuerMarketCaps, copy=True)

    if debugReport:
        cidList = [allSidCompanyMap.get(sid, ['']) for sid in data.universe]
        allCaps = numpy.zeros((len(data.universe), 3), float)
        allCaps[:,0] = numpy.ravel(ma.filled(data.marketCaps, 0.0))
        allCaps[:,1] = numpy.ravel(ma.filled(totalIssuerMarketCaps, 0.0))
        allCaps[:,2] = numpy.ravel(ma.filled(data.DLCMarketCap, 0.0))
        sidList = []
        for (cid, sid) in zip(cidList, data.universe):
            sidList.append('%s|%s' % (cid[0], sid.getSubIDString()))
        outName = 'tmp/mcaps-%s.csv' % date
        Utilities.writeToCSV(allCaps, outName, rowNames=sidList, columnNames=['cap','totalCap','DLCCap'], dp=6)

    logging.debug('computeTotalIssuerMarketCaps: end')
    return totalIssuerMarketCaps

def computeTotalLocalIssuerMarketCaps(
    data, date, numeraire, modelDB, marketDB, countryISO):
    """Returns an array of local-issuer-level market caps, using
    the asset_dim_company table in marketdb_global.
    data should be a Struct containing universe, marketCaps, and
    assetIdxMap attributes.
    """
    logging.debug('computeTotalLocalIssuerMarketCaps: begin')
    sidList = data.universe
    # Find all company IDs for the assets
    sidCompanyMap = modelDB.getIssueCompanies(date, sidList, marketDB)
    companies = set(sidCompanyMap.values())    
    # Find all SubIssues for those company IDs
    allSidCompanyMap = modelDB.getCompanySubIssues(
        date, companies, marketDB)
    # Group sids by company identifier
    companySidMap = dict()
    for (sid, (company, excludeFromMcap, ctyISO)) in allSidCompanyMap.items():
        if ctyISO not in countryISO:
            continue
        companySidMap.setdefault(company, list()).append(
                    (sid, excludeFromMcap, ctyISO))
    logging.info('%d issuers with multiple share lines or DRs', 
                    len(companySidMap))

    # Loop through companies
    totalIssuerMarketCaps = numpy.array(data.marketCaps)
    for (company, sidVals) in companySidMap.items():
        # Skip companies where only one SubIssue is alive
        if len(sidVals) <= 1:
            continue
        # Skip companies where all lines are excluded
        excludedSids = len([i for i in sidVals if i[1]])
        if excludedSids == len(sidVals):
            logging.debug('All assets for company %s are excluded', company)
            continue
        # Sum up market caps
        issuerMarketCap = 0.0
        for (sid, excludeFromMCap, ctyISO) in sidVals:
            if not excludeFromMCap:
                try:
                    assert(sid in data.assetIdxMap)
                except AssertionError:
                    print('check this',(sid,excludeFromMCap,ctyISO))
                    continue
                issuerMarketCap += data.marketCaps[data.assetIdxMap[sid]]
        if issuerMarketCap == 0.0:
            logging.warning('Company %s (%d sub-issues) has zero issuer market cap', 
                        company, len(sidVals))
        for (sid, excludeFromMCap, ctyISO) in sidVals:
            if sid in data.assetIdxMap:
                totalIssuerMarketCaps[data.assetIdxMap[sid]] = issuerMarketCap
                    
    logging.debug('computeTotalLocalIssuerMarketCaps: end')
    return totalIssuerMarketCaps
