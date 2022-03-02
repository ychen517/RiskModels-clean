import logging
import numpy.ma as ma
import numpy
from riskmodels import EstimationUniverse
from riskmodels import Matrices
from riskmodels import ModelDB
from riskmodels import LegacyUtilities as Utilities
from riskmodels import AssetProcessor

class MarketIndex:
    """Class to contain data items relevant to a 
    local market benchmark index.
    """
    def __init__(self, rmg, numHiCap, numLoCap, indexName, exchangeList):
        self.rmg = rmg
        self.numHiCap = numHiCap
        self.numLoCap = numLoCap
        self.indexName = indexName
        self.exchangeList = exchangeList
        self.data = list()

class MarketIndexSelector:
    """Class that returns a market portfolio when given a 
    country, either by replication logic or using actual 
    index vendor data.  Most DM and large EM markets are
    covered.  For smaller markets, we simply assume the 
    market portfolio is the universe of all local assets.
    """
    def __init__(self, modelDB, marketDB, forceRun=False):
        self.useBroadMarket = True
        self.useReplication = True
        self.useRootCapWeights = False
        self.forceRun = forceRun
        self.targetMarketCapCoverage = None
        self.estuExcludeCache = (None, None)

        isoCodeDataMap = {
            # Developed Americas
            # Russell 1000, Russell 2000
            'US': (1000, 2000, 'RUSSELL 3000', ['NYS','NAS','IEX','EXI']),
            'XU': (1000, 2000, 'RUSSELL 3000', ['NYS','NAS','IEX','EXI']),
            # TSX Composite, TSX SmallCap
            'CA': (250, 150, 'S&P/TSX Composite', ['TSE']),
            # Emerging Americas
            'BR': (100, 50, 'Brazil Bovespa Index', ['BSP']),
            # IPC Large 20, Mid 20, Small 20 = IPC Comp
            'MX': (40, 20, 'INDICE DE PRECIOS Y COTIZACIONES', ['MEX']),
            'CL': (40, 20, 'IPSA', ['SGO']),
            'PE': (15, 25, 'ISP 15', ['LIM']),
            # Developed Europe
            # FTSE 100 + 250 = FTSE 350, FTSE SmallCap
            'GB': (350, 300, 'FTSE All-Share', ['LON']),
            # CAC 40 + Next 20 + Mid 100, Small 90
            'FR': (160, 90, 'SBF 250 Constituents', ['PAR']),
            # DAX 30 + MDAX 50 + TecDAX 30 = HDAX, SDAX 50
            'DE': (110, 50, 'XETRA HDAX Index Constituents', None),
            # SMI 20 + SMI Mid 30, SPI Middle
            'CH': (50, 50, 'SWISS MARKET INDEX', None),
            # Ibex 35 + Ibex Medium 20, Ibex Small 30
            'ES': (55, 30, 'IBEX 35 INDEX', None),
            'PT': (20, 0, 'PORTUGAL PSI-20', ['LIS']),
            # MIB 30 + FTSE Italia Mid 50
            'IT': (100, 50, 'MILAN MIB 30', ['MIL']),
            # BEL 20 + BEL Mid
            'BE': (50, 30, 'BEL 20', ['BRU']),
            # ATX 20, ATX Prime 50
            'AT': (20, 30, 'ATX Index', ['WBO']),
            # OMX S30 + OMX S30 Next = OMX 60
            'SE': (60, 40, 'OMX STOCKHOLM 30 (Out of SIX proces', ['OME']),
            # AEX 25, AMX 25, AMS 25
            'NL': (50, 25, 'AMSTERDAM AEX (EOE)', ['AMS']),
            'FI': (45, 60, 'OMX Helsinki 25 (OMXH25)', ['HEL']),
            'DK': (40, 20, 'OMX Copenhagen (OMXC20)', ['CSE']),
            # OBX 25
            'NO': (25, 30, 'OBX', ['OSL']),
            # ISEQ 20, ISEQ Small Cap
            'IE': (20, 0, 'ISEQ 20', ['DUB']),
            # ATHEX 20 + Mid 40, SmallCap 80
            'GR': (60, 80, 'FTSE/ATHEX 20', ['ATH']),
            # Emerging Europe
            # MICEX 30, RTS 50
            'RU': (30, 20, 'Russian RTS', None),
            # ISE 30, ISE 50, ISE 100
            'TR': (50, 50, 'ISE NATIONAL 100', ['IST']),
            # WIG 20, mWIG 40, sWIG 80
            'PL': (20, 0, 'Warsaw General 20', ['WAR']),
            # Developed Asia-Pacific
            # TOPIX 100 + MID 400 = TOPIX 500, TOPIX SMALL
            'JP': (500, 1200, 'TSE TOPIX', ['TKS-S1','OSE-S1']),
            # ASX 200, ASX 300
            'AU': (200, 100, 'S&P/ASX 300', ['ASX']),
            'NZ': (50, 0, 'NZX 50', ['NZE']),
            # Hang Seng Large 15 + Mid 35, Composite 200
            'HK': (50, 150, 'Hang Seng Index', ['HKG']),
            # STI 30 + ST Mid 50, ST Small Cap
            'SG': (80, 40, 'SINGAPORE STRAITS TIMES(NEW)', ['SES']),
            # Emerging Asia-Pacific
            # CSI 100 + 200 = CSI 300, CSI 500
            'XC': (300, 500, 'CSI 300', ['SHG','SHE']),
            # H-shares, B-shares, redchips
            'CN': (75, 75, 'HS_MAINLAND_CPST._INDEX', None),
            # TSEC 50 + Midcap 100
            'TW': (300, 200, 'TSE TAIEX', ['TAI']),
            'KR': (200, 300, 'Korea Composite', ['KRX']),
            # BSE 100, 200, 500
            'IN': (200, 300, 'Bombay 200 INDEX', ['BOM','NSE']),
            'ID': (45, 0, 'Jakarta LQ45 (Top 45)', ['JKT']),
            'TH': (50, 50, 'BANGKOK S.E.T. 50', ['BKK','BKF']),
            # KLCI + Mid 70 = Top 100
            'MY': (30, 70, 'KLSE COMPOSITE', ['KLS']),
            'PH': (30, 0, 'iPSE', ['PHS']),
            # Middle East / Africa
            # TA 25, TA 75, TA 100
            'IL': (25, 75, 'TA-25', ['TAE']),
            'ZA': (40, 60, 'FTSE/JSE TOP 40', ['JSE']),
            'QA': (20, 0, 'DSM 20', ['DSM']),
            'KW': (40, 0, 'FTSE Coast Kuwait 40', ['FOK']),
            'AE': (20, 0, 'FTSE/NASDAQ Dubai UAE 20', None),
            # EGX 30 + EGX 70 = EGX 100
            'EG': (40, 60, 'EGX 30', ['CAI']),
        }
        self.ctyIndexDataMap = dict()
        self.codeRMGMap = dict([(n.mnemonic, n) for n in \
                    modelDB.getAllRiskModelGroups(inModels=False)])
        self.regionNameIdMap = dict([(r.description, r.region_id) \
                    for r in modelDB.getAllRiskModelRegions()])
        for r in self.codeRMGMap.values():
            data = isoCodeDataMap.get(r.mnemonic, (50, 50, '', None))
            self.ctyIndexDataMap[r.mnemonic] = MarketIndex(
                                    r, data[0], data[1], data[2], data[3])
    
    def toggleDomesticChinaHack(self, rmg, restoreXCValues):
        assert(rmg.description == 'Domestic China')
        
        if restoreXCValues is True:
            rmg.rmg_id = self.xc_rmg_id
            rmg.mnemonic = self.xc_mnemonic
        else:
            self.xc_rmg_id = rmg.rmg_id
            self.xc_mnemonic = rmg.mnemonic
            rmg.rmg_id = self.codeRMGMap['CN'].rmg_id
            rmg.mnemonic = self.codeRMGMap['CN'].mnemonic

    def createMarketIndex(self, isoCode, date, modelDB, marketDB, baseUniverse):
        """Create index constituents and weights for the market
        corresponding to the given isoCode.  
        Return value is a MarketIndex object.
        """
        logging.debug('createMarketIndex(%s): begin', isoCode)
        lmi = self.ctyIndexDataMap.get(isoCode, None)
        if lmi is None:
            raise Exception('No local market index defined for %s' % isoCode)
        if isoCode=='XC':
            self.toggleDomesticChinaHack(lmi.rmg, False)
        if not self.useReplication:
            lmi.data = self.loadVendorIndexData(
                        lmi, date, baseUniverse, modelDB, marketDB)
        else:
            lmi.data = self.replicateMarketIndex(
                        lmi, date, baseUniverse, modelDB, marketDB)
        if isoCode=='XC':
            self.toggleDomesticChinaHack(lmi.rmg, True)
        logging.debug('createMarketIndex(%s): end', isoCode)
        return lmi

    def getMarketIndex(self, isoCode, date, modelDB, baseUniverse=None):
        """Get index constituents and weights for the market
        corresponding to the given isoCode from the database.
        The assets are restricted to the baseUniverse if present.
        Return value is a MarketIndex object.
        """
        logging.debug('getMarketIndex(%s): begin', isoCode)
        lmi = self.ctyIndexDataMap.get(isoCode, None)
        lmi.data = modelDB.getRMGMarketPortfolio(lmi.rmg, date)
        if baseUniverse is not None:
            baseUniverse = set(baseUniverse)
            lmi.data = [(sid, val) for (sid, val) in lmi.data
                        if sid in baseUniverse]
        logging.debug('getMarketIndex(%s): end', isoCode)
        return lmi

    def loadVendorIndexData(self, lmi, date, univ, modelDB, marketDB):
        """Get index constituents and weights from vendor source.
        """
        logging.debug('loadVendorIndexData: begin')
        mdl2sub = dict([(n.getModelID(),n) for n in univ])
        benchmark = modelDB.getIndexConstituents(
                        lmi.indexName, date, marketDB, 120)
        data = [(mdl2sub(a),w) for (a,w) in benchmark if a in mdl2sub]
        if len(data) != len(benchmark):
            logging.warning('%d benchmark assets not in base universe',
                    len(benchmark) - len(data))
            logging.debug('Missing asset(s): %s', ', '.join(
                ['%s (%.4f%%)' % (a.getIDString(), w*100.0) for (a,w) \
                 in benchmark if a not in mdl2sub]))
        logging.debug('loadVendorIndexData: end')
        return data

    def replicateMarketIndex(self, lmi, date, univ, modelDB, marketDB):
        """Form capitalization-weighted portfolio for the given market.
        """
        logging.debug('replicateMarketIndex(%s): begin', lmi.rmg.mnemonic)
        if isinstance(lmi.rmg, ModelDB.RiskModelGroup):
            lmi.rmg.setRMGInfoForDate(date)

        # Get eligible universe of assets
        indices = self.determineEligibleUniverse(lmi, date, univ, modelDB, marketDB)
        if len(indices)==0:
            logging.warning('No eligible assets for %s', lmi.rmg.description)
            return list()

        # Get market caps, make sure everything is in same currency
        # If not, use RMG currency as numeraire
        rmgList = modelDB.getAllRiskModelGroups(inModels=False)
        xuRMG = [r for r in rmgList if r.description=='United States Small Cap'][0]
        usRMG = [r for r in rmgList if r.description=='United States'][0]
        if lmi.rmg == xuRMG:
            dateList = modelDB.getDates([usRMG], date, 19)
        else:
            dateList = modelDB.getDates([lmi.rmg], date, 19)

        assets = [univ[i] for i in indices]
        assetCurrMap = modelDB.getTradingCurrency(date, assets, marketDB)
        if len(set(assetCurrMap.values())) > 1:
            numeraire = marketDB.getCurrencyID(lmi.rmg.currency_code, date)
        else:
            numeraire = None
        marketCaps = modelDB.getAverageMarketCaps(dateList, assets, numeraire, marketDB)
        marketCaps = Utilities.screen_data(marketCaps)
        if not self.forceRun:
            assert(numpy.sum(ma.getmaskarray(ma.masked_where(marketCaps==0.0, marketCaps))) < len(assets))
        marketCaps = ma.filled(marketCaps, 0.0)

        # Downweight any foreign stocks, poor-man's free-float!
        e = EstimationUniverse.ConstructEstimationUniverse(assets, None, modelDB, marketDB)
        if lmi.rmg == xuRMG:
            (local_idx, foreign_idx) = e.exclude_by_market_classification(
                            date, 'HomeCountry', 'REGIONS', [usRMG.mnemonic])
        else:
            (local_idx, foreign_idx) = e.exclude_by_market_classification(
                            date, 'HomeCountry', 'REGIONS', [lmi.rmg.mnemonic])
        if len(foreign_idx) > 0:
            logging.info('Downweighting %d foreign stocks', len(foreign_idx))
            mcaps = numpy.take(marketCaps, foreign_idx, axis=0) * 0.5
            numpy.put(marketCaps, foreign_idx, mcaps)

        # Replication method: target mcap percentage, or fixed # of assets
        order = numpy.argsort(-marketCaps)
        if self.targetMarketCapCoverage is not None:
            sortedCaps = numpy.take(marketCaps, order, axis=0)
            capCovered = numpy.cumsum(sortedCaps) / numpy.sum(marketCaps)
            hit = numpy.flatnonzero(capCovered >= self.targetMarketCapCoverage)
            assert(len(hit) > 0)
            eligibleIdx = order[:hit[0]+1]
        else:
            # Only allow broad-market replication for developed markets
            if lmi.rmg.developed is None:
                devel = False
                logging.warning('Flag lmi.rmg.developed undefined - you should probably fix this')
            else:
                devel = lmi.rmg.developed
            maxAssets = lmi.numHiCap + (devel*lmi.numLoCap*self.useBroadMarket)
            if maxAssets >= len(indices):
                eligibleIdx = range(len(indices))
            else:
                eligibleIdx = order[:maxAssets]
            if lmi.rmg == xuRMG:
                eligibleIdx = order[1000:maxAssets]
            
        # Take subset of portfolio weights
        mc = numpy.take(marketCaps, eligibleIdx)

        # Check total weight
        portfolioCap = numpy.sum(mc)
        if portfolioCap <= 0.0:
            logging.error('Market portfolio has non-positive total mcap: %s', portfolioCap)
            if not self.forceRun:
                assert(portfolioCap>0.0)
            return list()

        # Take root-cap if necessary and scale
        if self.useRootCapWeights:
            mc = numpy.sqrt(mc)
        weights = mc / numpy.sum(mc)

        # Do some reporting
        maxIdx = numpy.argsort(weights)[-1]
        totalMCap = numpy.sum(marketCaps)
        if weights[maxIdx] > 0.30:
            logging.warning('Potentially concentrated market portfolio: %s, %s, %s, %.2f%%',
                    lmi.rmg.mnemonic, date, assets[eligibleIdx[maxIdx]].getSubIDString(), 
                    weights[maxIdx] * 100.0)
        logging.info('%s market portfolio has %d/%d assets, approx %.2f%% of total mcap of %.2f Bn',
                    lmi.rmg.mnemonic, len(eligibleIdx), len(indices), 
                    portfolioCap / totalMCap*100.0, totalMCap / 1.0e9)

        logging.debug('replicateMarketIndex(%s): end', lmi.rmg.mnemonic)
        return list(zip([assets[i] for i in eligibleIdx], weights))

    def determineEligibleUniverse(self, lmi, date, univ, modelDB, marketDB):
        """Determine subset of asset universe enligible 
        for market portfolio membership.  This step is to 
        systematically exclude troublesome asset types, foreign
        issuers, etc.
        """
        e = EstimationUniverse.ConstructEstimationUniverse(univ, None, modelDB, marketDB)
        indices0 = list(range(len(univ)))

        # First deal with the big troublemakers
        if lmi.rmg.mnemonic in ('RU','TH','MY','XC'):
            data = Utilities.Struct()
            data.universe = univ
            data.exposureMatrix = Matrices.ExposureMatrix(univ)
            data.exposureMatrix.addFactor(lmi.rmg.description, 
                    numpy.ones(len(univ)), Matrices.ExposureMatrix.StyleFactor)
            data.assetIdxMap = dict([(j,i) for (i,j) in enumerate(univ)])
            dateList = modelDB.getDates([lmi.rmg], date, 19)
            if len(dateList) < 1:
                return []
            data.marketCaps = numpy.array(modelDB.getAverageMarketCaps(
                dateList, univ, marketDB.getCurrencyID(lmi.rmg.currency_code, date), marketDB))           
            # For Thailand, Malaysia, deal with local/foreign shares
            if lmi.rmg.mnemonic in ('TH', 'MY'):
                dummyFactor = 'Thailand'
                if lmi.rmg.mnemonic == 'TH':
                    dummyFactor = 'Malaysia'
                data.exposureMatrix.addFactor(dummyFactor, numpy.zeros(len(univ)), Matrices.ExposureMatrix.CountryFactor)
                exclIndices = process_southeast_asia_share_classes(data, date, modelDB, marketDB)
            # For Russia, sort out RTS/MICEX mess
            elif lmi.rmg.mnemonic == 'RU':
                exclIndices = process_russian_exchanges(data, date, modelDB, marketDB)
            # Domestic China
            else:
                data.exposureMatrix.addFactor('China', numpy.ones(
                    len(univ)), Matrices.ExposureMatrix.CountryFactor)
                (a_idx, b_idx, h_idx, other) = process_china_share_classes(data, date, modelDB, marketDB)
                exclIndices = b_idx + h_idx
            indices0 = list(set(indices0).difference(exclIndices))
            logging.info('Keeping %d assets after %s-specific rules filter', 
                        len(indices0), lmi.rmg.description)

        # Dutch Madness
        if lmi.rmg.mnemonic == 'NL':
            exclIndices = list()
            gvkeyMap = modelDB.getIssueGVKEY(date, univ, marketDB)
            gvkeySubIssuePairMap = dict()
            for (sid, gvkey) in gvkeyMap.items():
                if gvkey not in gvkeySubIssuePairMap:
                    gvkeySubIssuePairMap[gvkey] = list()
                gvkeySubIssuePairMap[gvkey].append(sid)
            dateList = modelDB.getDates([lmi.rmg], date, 20)
            assetIdxMap = dict([(j,i) for (i,j) in enumerate(univ)])
            for (gvkey, idList) in gvkeySubIssuePairMap.items():
                if len(idList) > 1:
                    avgVolume = numpy.average(modelDB.loadVolumeHistory(
                                    dateList, idList, 1).data.filled(0.0), axis=1)
                    exclIndices.extend([assetIdxMap[idList[i]] for i in \
                                    numpy.argsort(-avgVolume)[1:]])
            indices0 = list(set(indices0).difference(exclIndices))

        # Restrict to certain exchanges for certain locales
        exch = lmi.exchangeList
        if exch is not None:
            (indices0, tmp) = e.exclude_by_market_classification(
                        date, 'Market', 'REGIONS', exch, 
                        baseEstu=indices0)
            logging.info('Keeping %d assets after restricting to exchanges %s',
                    len(indices0), ','.join(exch))

        # Exclude assets that are part of *any* model's exclusion list
        if lmi.rmg.mnemonic not in ('HK','NL','PE','AR','ZA'):
            if date != self.estuExcludeCache[0]:
                self.estuExcludeCache = (date, modelDB.getAllESTUExcludedIssues(date))
            exclIssues = self.estuExcludeCache[1]
            if len(exclIssues) > 0:
                mdlIdxMap = dict([(j.getModelID(), i) for (i,j) in enumerate(univ)])
                exclIndices = [mdlIdxMap[id] for id in exclIssues \
                               if id in mdlIdxMap]
                indices0 = list(set(indices0).difference(exclIndices))
                logging.info('Keeping %d assets after applying exclusion list', 
                            len(indices0))
        
        # Exclude foreign issuers by ISIN; we don't do this for
        # Europe (too much integration), or HK/China (H-share/redchip mess)
        if (lmi.rmg.region_id != self.regionNameIdMap['Europe']) and (
                lmi.rmg.mnemonic not in ('HK','CN','PE','AR')):
            if lmi.rmg.mnemonic == 'XU':
                (indices0, tmp) = e.exclude_by_isin_country(['US'], date, baseEstu=indices0)
            else:
                (indices0, tmp) = e.exclude_by_isin_country([lmi.rmg.mnemonic], date, baseEstu=indices0)
            logging.info('Keeping %d assets after ISIN filter', len(indices0))

        # Additional safeguard -- filter by home country
        if lmi.rmg.mnemonic not in ('CN','PE','AR'):
            if lmi.rmg.mnemonic == 'HK':
                allowedCodes = ['HK', 'CN']
            elif lmi.rmg.mnemonic == 'SG':
                allowedCodes = ['HK', 'SG']
            elif lmi.rmg.mnemonic == 'XU':
                allowedCodes = ['US']
            else:
                allowedCodes = [lmi.rmg.mnemonic]
            (indices0, tmp) = e.exclude_by_market_classification(
                            date, 'HomeCountry', 
                            'REGIONS', allowedCodes, baseEstu=indices0)
            logging.info('Keeping %d assets after home country filter', len(indices0))

        # Keep primarily common equity instruments
        allowedCodes = list()
        if lmi.rmg.mnemonic in ['US','CA', 'XU']:
            clsName = 'TQA FTID Domestic Asset Type'
            allowedCodes.extend(['C','I'])  # Allow REITs for US and Canada
            if lmi.rmg.mnemonic == 'CA':
                allowedCodes.append('U')    # Allow investment trusts for Canada
        else:
            clsName = 'TQA FTID Global Asset Type'
            allowedCodes = ['09','10']
            # Allow REITs for Australia, UK, Singapore, France
            if lmi.rmg.mnemonic in ['AU','GB','SG','FR']:
                allowedCodes.append('54')
            # Allow domestic-only stocks from GCC countries
            if lmi.rmg.region_id == self.regionNameIdMap['Middle East']:
                allowedCodes.append('08')
            # Allow preferred shares for Brazil
            if lmi.rmg.mnemonic == 'BR':
                allowedCodes.append('34')
        if len(allowedCodes) > 0:
            (indices1, tmp) = e.exclude_by_market_classification(
                    date, clsName,
                    'ASSET TYPES', allowedCodes, baseEstu=indices0)
            logging.info('Keeping %d assets after FTID asset type filter', 
                        len(indices1))
        else:
            indices1 = indices0

        # Extra safeguard; use secondary classification to weed out 
        # ADRs, GDRs, and other undesirables
        # TODO: consider adding code CF to list, but unclear what that is
        lenElig = len(indices1)
        (dr, tmp) = e.exclude_by_market_classification(
                date, 'DataStream2 Asset Type',
                'ASSET TYPES', ['ADR','GDR','ET','ETF'], keepMissing=False)
        indices2 = list(set(indices1).difference(dr))
        if lenElig > len(indices2):
            logging.info('Keeping %d assets after asset type filter',
                    len(indices1))

        lenElig = len(indices2)
        data = Utilities.Struct()
        # Filter out CN Stock Connect assets
        data.marketTypeDict = AssetProcessor.get_asset_info(\
                date, univ, modelDB, marketDB, 'REGIONS', 'Market')
        (indices2, tmp) = e.exclude_by_market_type(
                date, data, includeFields=None,
                excludeFields=AssetProcessor.connectExchanges,
                baseEstu=indices2)
        if lenElig > len(indices2):
            logging.info('Keeping %d assets after exchange type filter',
                    len(indices1))

        logging.info('Final universe consists of %d assets', len(indices2))
        return indices2

    def useBroadMarketReplication(self, value):
        self.useBroadMarket = value

    def useIndexReplication(self, value):
        self.useReplication = value

    def useMarketCapTargeting(self, percentage):
        self.targetMarketCapCoverage = percentage

    def useRootCapWeighting(self, value):
        self.useRootCapWeights = value

def process_southeast_asia_share_classes(data, modelDate, 
                       modelDB, marketDB, dr_indices=None, daysBack=20):
    """Handles issuers with multiple share classes trading in
    Thailand (local shares, foreign shares, NVDRs, etc.) and
    Malaysia (local vs foreign shares).
    Relies in companyIDs to identify unique issuers, and ticker 
    formats ('-F' suffix, for example).
    Additionally, compares median daily trading volume between 
    local and foreign lines to determine ESTU eligibility.
    """
    logging.debug('process_southeast_asia_share_classes: begin')
    # Find all Thai and Malaysian assets
    try:
        fIdx = [data.exposureMatrix.getFactorIndex('Thailand'),
                data.exposureMatrix.getFactorIndex('Malaysia')]
    except:
        return list()
    if dr_indices is None:
        dr_indices = list()
    dr_indices = set(dr_indices)
    exposures = ma.take(data.exposureMatrix.getMatrix(), fIdx, axis=0)
    subids = [data.universe[i] for i in numpy.flatnonzero(
                ma.getmaskarray(ma.sum(exposures, axis=0))==0) 
                if i not in dr_indices]
    tickerMap = modelDB.getIssueTickers(modelDate, subids, marketDB)

    # Look up individual issuers via CompanyID
    companyIDMap = modelDB.getIssueCompanies(modelDate, subids, marketDB)
    companyIDSubIssuePairMap = dict()
    for (sid, cid) in companyIDMap.items():
        if cid not in companyIDSubIssuePairMap:
            companyIDSubIssuePairMap[cid] = list()
        companyIDSubIssuePairMap[cid].append(sid)

    # Identify pairs of local/foreign shares; ignore warrants, NVDRs, etc
    localForeignPairs = list()
    exclSubIssues = list()
    for (key, idList) in companyIDSubIssuePairMap.items():
        if len(idList) <= 1:
            continue
        else:
            tickerLen = ma.array([len(tickerMap.get(id, list())) \
                                        for id in idList])
            tickerLen = ma.masked_where(tickerLen==0, tickerLen)
            idx = ma.argsort(tickerLen)[0]
            baseTicker = tickerMap.get(idList[idx], None)
            if baseTicker is None:
                continue
            fIdx = None
            for i in range(len(idList)):
                if i==idx:
                    continue
                elif tickerMap.get(idList[i]) == baseTicker + '-F' or \
                     tickerMap.get(idList[i]) == baseTicker + 'F' or \
                     tickerMap.get(idList[i]) == baseTicker + '-O1':
                    fIdx = i
                else:
                    exclSubIssues.append(idList[i])
            if fIdx is not None:
                localForeignPairs.append((idList[idx], idList[fIdx]))
    logging.info('Found %d local/foreign ID pairs', len(localForeignPairs))

    # Compare median daily volume over past month
    subids = [i for j in localForeignPairs for i in j]
    idxMap = dict([(j,i) for (i,j) in enumerate(subids)])
    rmgList = [r for r in modelDB.getAllRiskModelGroups() \
               if r.description in ('Thailand', 'Malaysia')]
    dateList = modelDB.getDates(rmgList, modelDate, daysBack)
    volume = modelDB.loadVolumeHistory(dateList, subids, 1)
    try:
        medianVolume = ma.median(volume.data.filled(0.0), axis=1)
    except:
        medianVolume = Utilities.ma_median(volume.data.filled(0.0), axis=1)

    # Opt for foreign share for those deemed 'sufficiently' liquid
    exclLocalShares = list()
    for (locShare, forShare) in localForeignPairs:
        vol0 = medianVolume[idxMap[locShare]]
        vol1 = medianVolume[idxMap[forShare]]
        if vol1 >= vol0 * 0.10 and vol1 != 0.0:
            exclLocalShares.append(locShare)
        else:
            exclSubIssues.append(forShare)
    mcap = ma.sum(ma.take(data.marketCaps, 
            [data.assetIdxMap[sid] for sid in exclLocalShares], axis=0))
    logging.info('Found %d sufficiently liquid foreign shares, %.2f bn USD mcap',
            len(exclLocalShares), mcap / 1e9)
    exclSubIssues.extend(exclLocalShares)
    indices = [data.assetIdxMap[sid] for sid in exclSubIssues]
    logging.info('Excluding %d multi-class shares in total', len(indices))

    logging.debug('process_southeast_asia_share_classes: end')
    return indices

def process_russian_exchanges(data, modelDate, modelDB, marketDB):
    """Deal with the mess involving RTS/MICEX.  Check for Russian
    issuers with multiple security lines using ISIN lookup, then,
    if both MICEX and RTS lines exist, flag the latter for exclusion.
    """
    logging.debug('process_russian_exchanges: begin')
    expM = data.exposureMatrix
    try:
        fidx = [expM.getFactorIndex('Russian Federation')]
    except:
        return list()
    exposures = ma.take(expM.getMatrix(), fidx, axis=0)
    subids = [expM.getAssets()[i] for i in numpy.flatnonzero(
                ma.getmaskarray(ma.sum(exposures, axis=0))==0)]
    ids = [sid.getModelID() for sid in subids]

    # Look for issuers with multiple lines based on ISIN
    modelISINMap = modelDB.getIssueISINs(modelDate, ids, marketDB)
    isinIssuesMap = dict()
    for (id, isin) in modelISINMap.items():
        if isin not in isinIssuesMap:
            isinIssuesMap[isin] = list()
        isinIssuesMap[isin].append(id)

    # Check tickers, flag -R (RTS) line if -M (MICEX) line coexists
    modelTickerMap = modelDB.getIssueTickers(modelDate, ids, marketDB)
    rts = list()
    for (isin, idList) in isinIssuesMap.items():
        if len(idList)==1 or isin[:2] != 'RU':
            continue
        tickerSuffixes = [modelTickerMap.get(n, '')[-2:] for n in idList]
        if '-R' in tickerSuffixes:
            rts.append(idList[tickerSuffixes.index('-R')])

    mdlSubidMap = dict([(n.getModelID(),n) for n in subids])
    indices = [data.assetIdxMap[mdlSubidMap[n]] for n in rts]
    totalCap = ma.sum(ma.take(data.marketCaps, indices))
    logging.info('Found %d assets with RTS and MICEX lines (%.2f bn mcap)',
                 len(rts), totalCap / 1e9)
#    logging.debug('Axioma IDs: %s' % \
#            (', '.join([n.getIDString() for n in rts])))

    logging.debug('process_russian_exchanges: end')
    return indices

def process_china_share_classes(data, modelDate, modelDB, marketDB, 
                                factorName='China'):
    """Identify various Chinese share classes.  Return 
    value is a tuple containing 3 lists containing 
    the index positions of A-, B-, and H-shares/Red-Chips.
    Assumes all Chinese stocks are assigned exposure to factorName
    in data.exposureMatrix.
    """
    logging.debug('process_china_share_classes: begin')
    # Find all assets with China as home country
    try:
        if factorName != '':
            expM = data.exposureMatrix
            exposures = ma.take(expM.getMatrix(), 
                                [expM.getFactorIndex(factorName)], axis=0)
        else:
            exposures = numpy.ones(len(data.universe))
    except:
        return (list(), list(), list(), list())
    indices = numpy.flatnonzero(ma.getmaskarray(exposures)==0)
    assets = [data.universe[i] for i in indices]

    # New code to use asset type
    assetTypeDict = AssetProcessor.get_asset_info(modelDate, assets, modelDB, marketDB,
            'ASSET TYPES', 'Axioma Asset Type')
    aShares = list()
    bShares = list()
    hShares = list()    # Includes Red-Chips
    sidHaveMap = [sid for sid in assets if (sid in data.assetIdxMap) and (sid in assetTypeDict)]
    aShares = [data.assetIdxMap[sid] for sid in sidHaveMap if assetTypeDict[sid] == 'AShares']
    bShares = [data.assetIdxMap[sid] for sid in sidHaveMap if assetTypeDict[sid] == 'BShares']
    hShares = [data.assetIdxMap[sid] for sid in sidHaveMap if assetTypeDict[sid] in AssetProcessor.intlChineseAssetTypes]

    checked = aShares + bShares + hShares
    mcap = data.marketCaps
    logging.info('Found %d A-Shares (%.2f bn mcap)', 
                len(aShares), numpy.sum(numpy.take(mcap, aShares)) / 1e9)
    logging.info('Found %d B-Shares (%.2f bn mcap)', 
                len(bShares), numpy.sum(numpy.take(mcap, bShares)) / 1e9)
    logging.info('Found %d H-Shares/Red-Chips (%.2f bn mcap)', 
                len(hShares), numpy.sum(numpy.take(mcap, hShares)) / 1e9)
    if len(checked) != len(assets):
        intl = list(set(indices).difference(checked))
        logging.info('Additional %d China assets (%.2f bn mcap) trading outside China/HK',
                    len(indices) - len(checked), numpy.sum(numpy.take(mcap, intl)) / 1e9)
    else:
        intl = list()

    logging.debug('process_china_share_classes: end')
    return (aShares, bShares, hShares, intl)

