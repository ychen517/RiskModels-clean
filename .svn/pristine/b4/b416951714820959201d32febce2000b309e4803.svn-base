
import datetime
import logging
import numpy as np
import numpy.ma as ma
import numpy
import pandas
import itertools

import riskmodels
from riskmodels import Classification
from riskmodels import CurrencyRisk
from riskmodels import EstimationUniverse
from riskmodels import GlobalExposures
from riskmodels import Matrices
from riskmodels.Matrices import ExposureMatrix
from riskmodels import MFM
from riskmodels import EquityModel
from riskmodels.Factors import ModelFactor
from riskmodels import RegressionToolbox
from riskmodels import ReturnCalculator
from riskmodels import RiskCalculator
from riskmodels import RiskModels
from riskmodels import RiskModels_V3
from riskmodels import Standardization
from riskmodels import TOPIX
from riskmodels import MarketIndex
from riskmodels import LegacyModelParameters as ModelParameters
from riskmodels import ModelParameters2017
from riskmodels import FactorReturns

class JPResearchModel1(EquityModel.FundamentalModel):
    """ Same as Final JP research model (JPResearchModelFinalESTU), but Profitability is ROE only
            1. Split EY and Value
            2. Combine D/A and D/E in Leverage
            3. Include profitability factor
            4. Include Dividend Yield factor
            5. Enhanced ESTU coverage (based on inclusion of several exchanges)
            6. Updated definition of MidCap based on clear inclusion criteria - similar to that of Russell Japan Mid Cap Index 80-97.5%
            7. Updated definitions of Growth, Market sensitivity, MTM, Volatility, Exchange Rate sensitivity
            8. GICS2016 definitions
    """
    estu_parameters = {
                       'minNonZero':0.75,
                       'minNonMissing':0.95,
                       'maskZeroWithNoADV_Flag': True,
                       'returnLegacy_Flag': False,
                       'CapByNumber_Flag':False,
                       'CapByNumber_hiCapQuota':np.nan,
                       'CapByNumber_lowCapQuota':np.nan,
                       'market_lower_pctile':1 ,
                       'country_lower_pctile':5,
                       'industry_lower_pctile':5,
                       'dummyThreshold': 10,
                       'inflation_cutoff':0.01
                        }
    k = 5.0
    minNonZero = 0.75
    rm_id = -5
    revision = 1
    rms_id = -10
    inflation_cutoff = 0.01
    regionalDescriptorStructure = False
    twoRegressionStructure = True
    multiCountry = False

    # List of 13 style factors in the model
    styleList = [
                'Earnings Yield',
                'Value',
                'Leverage',
                'Growth',
                'Profitability',
                'Dividend Yield',
                'Size',
                'Liquidity',
                'Market Sensitivity',
                'Volatility',
                'Medium-Term Momentum',
                'MidCap',
                'Exchange Rate Sensitivity',
                ]

    DescriptorMap = {
            'Earnings Yield': ['Earnings_to_Price_Annual', 'Est_Earnings_to_Price_12MFL_Annual'],
            'Value': ['Book_to_Price_Annual'],
            'Leverage': ['Debt_to_Assets_Annual', 'Debt_to_Equity_Annual'],
            'Growth': ['Earnings_Growth_RPF_Annual', 'Sales_Growth_RPF_Annual'],
            'Dividend Yield': ['Dividend_Yield_Annual'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_60D'],
            'Market Sensitivity': ['Market_Sensitivity_250D'],
            'Volatility': ['Volatility_125D'],
            'Medium-Term Momentum': ['Momentum_250x20D'],
            'Exchange Rate Sensitivity': ['XRate_104W_USD'],
            'Profitability': ['Return_on_Equity_Annual'] }

    DescriptorWeights = {'Earnings Yield': [0.25, 0.75]}
    smallCapMap = {'MidCap': [80.0, 97.5],}
    noProxyList = ['Dividend Yield']
    fillMissingList = ['Earnings Yield', 'Value', 'Leverage', 'Growth', 'Profitability']
    fillWithZeroList = ['Dividend Yield']
    shrinkList = {'Liquidity': 60,
                  'Market Sensitivity': 250,
                  'Volatility': 125,
                  'Medium-Term Momentum': 250}
    orthogList = {'Volatility': [['Market Sensitivity'], True, 1.0]}

    # Setting up market intercept if relevant
    interceptFactor = 'Market Intercept'
    intercept = ModelFactor(interceptFactor, interceptFactor)
    industryClassification = Classification.GICSCustomJP(datetime.date(2016,9,1))

    estuAssetTypes = ['REIT', 'Com']

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.JPResearchModelFinalESTU')
        # Set up relevant styles to be created/used
        ModelParameters2017.defaultExposureParameters(self, self.styleList)
        self.styles = [s for s in self.totalStyles if s.name in self.styleList]
        EquityModel.FundamentalModel.__init__(self, ['SEDOL'], modelDB, marketDB)

        # Set up regression parameters
        dummyThreshold = 10

        # Run risk based on the square root of market cap weights  
        self.internalCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType='Industry Groups',
                dummyThreshold=dummyThreshold,
                marketRegression=False,
                kappa=5.0,
                useRealMCaps=True,
                regWeight = 'rootCap',
                )

        # Set up external regression parameters
        self.returnCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType='Industry Groups',
                dummyThreshold=dummyThreshold,
                marketRegression=False,
                kappa=None,
                useRealMCaps=True,
                regWeight='invSpecificVariance'
                )

        # This controls the FMP regression
        self.fmpCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType='Industry Groups',
                marketRegression=False,
                dummyThreshold=dummyThreshold,
                kappa=None)

        # Set up risk parameters
        ModelParameters2017.defaultFundamentalCovarianceParameters(
                self, nwLag=2,
                varDVAOnly=False, unboundedDVA=False,
                )

        # Set up standardization parameters
        gloScope = Standardization.GlobalRelativeScope([f.name for f in self.styles])
        self.exposureStandardization = Standardization.BucketizedStandardization([gloScope],
                fillWithZeroList=self.fillWithZeroList)

        # Set up descriptor standardization parameters
        descriptors = sorted(list(set([item for sublist
            in list(self.DescriptorMap.values()) for item in sublist])))
        exceptionNames = [self.DescriptorMap[sf] for sf in self.noProxyList]
        exceptionNames = list(itertools.chain.from_iterable(exceptionNames))
        self.descriptorStandardization = Standardization.BucketizedStandardization(
                    [Standardization.GlobalRelativeScope(descriptors)],
                    mad_bound=15.0, fancyMAD=self.fancyMAD, exceptionNames=exceptionNames)

        self.indexSelector = MarketIndex.MarketIndexSelector(modelDB, marketDB)
        # Set up TOPIX replication - FIXME
        self.topixReplicator = TOPIX.TOPIXReplicator(self, modelDB)
        modelDB.createCurrencyCache(marketDB)

    def generate_model_specific_exposures(self, modelDate, data, modelDB, marketDB):
        """Generate the non-default factors.
        """
        beta = numpy.zeros((len(data.universe)), float)
        # Cap-based style factors here
        if not hasattr(self, 'estuMap') or self.estuMap is None:
            return data.exposureMatrix

        # Small-cap factors
        dateList = modelDB.getDates(self.rmg, modelDate, 61, excludeWeekend=True)
        styleNames = [s.name for s in self.styles]
        scList = [sc for sc in self.estuMap.keys() if sc in styleNames]
        for sc in scList:
            beta = Matrices.allMasked((len(data.universe)), float)
            scAssets = [sid for sid in self.estuMap[sc].assets if sid in data.universe]
            if len(scAssets) < 1:
                logging.warning('No assets in %s universe', sc)
            else:
                qualifiedAssets = modelDB.loadESTUQualifyHistory(
                    self.rms_id, scAssets, dateList, estuInstance=self.estuMap[sc])
                qualifiedAssets = ma.filled(ma.sum(qualifiedAssets.data, axis=1), 0.0)
                if len(qualifiedAssets) < 1:
                    logging.warning('No assets qualified for %s factor', sc)
                else:
                    qualifiedAssets = qualifiedAssets / float(numpy.max(qualifiedAssets, axis=None))
                    for (i_c, idx) in enumerate([data.assetIdxMap[sid] for sid in scAssets]):
                        beta[idx] = qualifiedAssets[i_c]
            data.exposureMatrix.addFactor(sc, beta, ExposureMatrix.StyleFactor)
            if self.exposureStandardization.exceptionNames is None:
                self.exposureStandardization.exceptionNames = [sc]
            else:
                self.exposureStandardization.exceptionNames.append(sc)
        self.exposureStandardization.exceptionNames = list(set(self.exposureStandardization.exceptionNames))

        return data.exposureMatrix

    def generate_estimation_universe_old(self, modelDate, data, modelDB, marketDB, excludeFactors=None):
        """Estimation universe selection criteria for JP.
        """
        import pandas as pd
        #import ipdb;ipdb.set_trace()
        self.log.info('generate_estimation_universe for JP Model: begin')
        buildEstu = EstimationUniverse.ConstructEstimationUniverse(data.universe, self, modelDB, marketDB)

        # Set up various eligible and total universes
        universeIdx = list(range(len(buildEstu.assets)))
        originalEligibleUniverse = list(data.eligibleUniverse)
        originalEligibleUniverseIdx = [data.assetIdxMap[sid] for sid in originalEligibleUniverse]

        # Remove nursery market assets
        if len(data.nurseryUniverse) > 0:
            logging.info('Checking for assets from nursery markets')
            ns_indices = [data.assetIdxMap[sid] for sid in data.nurseryUniverse]
            (eligibleUniverseIdx, nonest) = buildEstu.exclude_specific_assets(
                    ns_indices, baseEstu=originalEligibleUniverseIdx)
            if n != len(eligibleUniverseIdx):
                n = len(eligibleUniverseIdx)
                logging.info('ESTU currently stands at %d stocks', n)
        else:
            eligibleUniverseIdx = originalEligibleUniverseIdx

        universe = pd.DataFrame(buildEstu.assets,columns=['SubID_Obj'])
        universeIdx = universe.index
        original_eligibleUniverse = pd.DataFrame(list(zip(data.eligibleUniverse,originalEligibleUniverseIdx)),columns=['SubID_Obj','originalEligibleUniverseIdx'])

        logging.info('ESTU currently stands at %d stocks based on original eligible universe', len(originalEligibleUniverse))

        # Report on thinly-traded assets over the entire universe
        logging.info('Looking for thinly-traded stocks')
        # (nonSparseIdx, sparse) = buildEstu.exclude_thinly_traded_assets(modelDate, data, baseEstu=universeIdx, minNonZero=0.75)
        # data.nonSparse = numpy.take(buildEstu.assets, nonSparseIdx, axis=0)
        (non_thin_idx, thin_idx) = buildEstu.exclude_thinly_traded_assets(modelDate, data, baseEstu=universe.index.values, minNonZero=0.75)
        # (non_thin_idx, thin_idx) = buildEstu.exclude_thinly_traded_assets(modelDate, data, baseEstu=universe.index.values, minNonZero=0.95)
        data.nonSparse = pd.DataFrame(universe,index=non_thin_idx)["SubID_Obj"].tolist()

        # Exclude thinly traded assets
        estu_withoutThin_idx = list(set(eligibleUniverseIdx).intersection(set(non_thin_idx)))
        logging.info('ESTU currently stands at %d stocks after Filtering thinly-traded assets', len(estu_withoutThin_idx))
        # (2) Filtering tiny-cap assets by market, country and industry
        # (a) Weed out tiny-cap assets by market
        lowerBound = 1
        logging.info('Filtering by top %d%% mcap on entire market', 100-lowerBound)
        (large_byMkt_idx, nonest1) = buildEstu.exclude_by_cap_ranking(data, modelDate, baseEstu=estu_withoutThin_idx,
                                                                     lower_pctile=lowerBound, method='percentage')
        # (2b) Weed out tiny-cap assets by country
        lowerBound = 5
        logging.info('Filtering by top %d%% mcap on country', 100-lowerBound)
        (large_byCntry_idx, nonest2) = buildEstu.exclude_by_cap_ranking(data, modelDate, baseEstu=estu_withoutThin_idx,
                                                                       byFactorType=ExposureMatrix.CountryFactor,
                                                                       lower_pctile=lowerBound, method='percentage',
                                                                       excludeFactors=excludeFactors)
        # (2c) Perform similar check by industry
        logging.info('Filtering by top %d%% mcap on industry', 100-lowerBound)
        (large_byIndtry_idx, nonest3) = buildEstu.exclude_by_cap_ranking(
               data, modelDate, baseEstu=estu_withoutThin_idx,
               byFactorType=ExposureMatrix.IndustryFactor,
               lower_pctile=lowerBound, method='percentage',
               excludeFactors=excludeFactors)
    
        estu_mktCap_idx = set(large_byMkt_idx).union(large_byCntry_idx).union(large_byIndtry_idx)
        estu_mktCap_idx = list(estu_mktCap_idx)
        tinyCap_idx = list(set(estu_withoutThin_idx).difference(estu_mktCap_idx))

        logging.info('ESTU currently stands at %d stocks after Filtering by Market Cap.', len(estu_mktCap_idx))

        # candid_univ_idx = eligibleUniverseIdx
        candid_univ_idx = estu_withoutThin_idx
        # Inflate any thin countries or industries - add 2*
        # minFactorWidth=2*self.returnCalculator.allParameters[0].getThinFactorInformation().dummyThreshold
        minFactorWidth=self.returnCalculator.allParameters[0].getThinFactorInformation().dummyThreshold
        logging.info('Inflating any thin factors')

        (estu_inflated_idx, nonest,herf_num_list) = buildEstu.pump_up_factors2(
                data, modelDate, currentEstu=estu_mktCap_idx,
                baseEstu=candid_univ_idx,
                byFactorType=[ExposureMatrix.CountryFactor, ExposureMatrix.IndustryFactor],
                minFactorWidth=minFactorWidth,
                cutOff = self.inflation_cutoff, 
                excludeFactors=excludeFactors)

        logging.info('ESTU currently stands at %d stocks', len(estu_inflated_idx))
        herf_num_list = pd.DataFrame(herf_num_list)
        herf_num_list.to_csv('herf_num_list.csv')
        # Apply grandfathering rules
        logging.info('Incorporating grandfathering')
        (estu_final_Idx, ESTUQualify, nonest) = buildEstu.grandfather(modelDate,
                estu_inflated_idx, baseEstu=candid_univ_idx,
                estuInstance=self.estuMap['main'])

        totalcap = ma.sum(ma.take(data.marketCaps, estu_final_Idx, axis=0), axis=0) / 1e9
        self.log.info('Final estu contains %d assets, %.2f bn (%s)',len(estu_final_Idx), totalcap, self.numeraire.currency_code)

        self.log.info('Final estu contains %d assets in thin-traded assets.',sum(pd.Series(estu_inflated_idx).isin(thin_idx)))
        self.log.info('Final estu contains %d assets in tiny cap assets.',sum(pd.Series(estu_inflated_idx).isin(tinyCap_idx)))
        self.log.debug('generate_estimation_universe: end')

        # If we have a family of estimation universes, populate the main estu accordingly
        self.estuMap['main'].assets = [buildEstu.assets[idx] for idx in estu_final_Idx]
        self.estuMap['main'].qualify = [buildEstu.assets[idx] for idx in ESTUQualify]

        return estu_final_Idx

    def generate_estimation_universe(self, modelDate, data, modelDB, marketDB, excludeFactors=None):
        estu1 = self.generate_estimation_universe_v2(modelDate, data, modelDB, marketDB, excludeFactors=None)
        # estu2 = self.generate_estimation_universe_old(modelDate, data, modelDB, marketDB, excludeFactors=None)
        # print modelDate, estu1, estu2
        return estu1

class JPResearchModel3(EquityModel.FundamentalModel):
    """ Same as Final JP research model (JPResearchModelFinalESTU), but Profitability is ROA only
            1. Split EY and Value
            2. Combine D/A and D/E in Leverage
            3. Include profitability factor
            4. Include Dividend Yield factor
            5. Enhanced ESTU coverage (based on inclusion of several exchanges)
            6. Updated definition of MidCap based on clear inclusion criteria - similar to that of Russell Japan Mid Cap Index 80-97.5%
            7. Updated definitions of Growth, Market sensitivity, MTM, Volatility, Exchange Rate sensitivity
            8. GICS2016 definitions
    """
    estu_parameters = {
                       'minNonZero':0.75,
                       'minNonMissing':0.95,
                       'maskZeroWithNoADV_Flag': True,
                       'returnLegacy_Flag': False,
                       'CapByNumber_Flag':False,
                       'CapByNumber_hiCapQuota':np.nan,
                       'CapByNumber_lowCapQuota':np.nan,
                       'market_lower_pctile':1 ,
                       'country_lower_pctile':5,
                       'industry_lower_pctile':5,
                       'dummyThreshold': 10,
                       'inflation_cutoff':0.01
                        }
    k = 5.0
    minNonZero = 0.75
    rm_id = -5
    revision = 3
    rms_id = -1001
    inflation_cutoff = 0.01
    regionalDescriptorStructure = False
    twoRegressionStructure = True
    multiCountry = False

    # List of 13 style factors in the model
    styleList = [
                'Earnings Yield',
                'Value',
                'Leverage',
                'Growth',
                'Profitability',
                'Dividend Yield',
                'Size',
                'Liquidity',
                'Market Sensitivity',
                'Volatility',
                'Medium-Term Momentum',
                'MidCap',
                'Exchange Rate Sensitivity',
                ]

    DescriptorMap = {
            'Earnings Yield': ['Earnings_to_Price_Annual', 'Est_Earnings_to_Price_12MFL_Annual'],
            'Value': ['Book_to_Price_Annual'],
            'Leverage': ['Debt_to_Assets_Annual', 'Debt_to_Equity_Annual'],
            'Growth': ['Earnings_Growth_RPF_Annual', 'Sales_Growth_RPF_Annual'],
            'Dividend Yield': ['Dividend_Yield_Annual'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_60D'],
            'Market Sensitivity': ['Market_Sensitivity_250D'],
            'Volatility': ['Volatility_125D'],
            'Medium-Term Momentum': ['Momentum_250x20D'],
            'Exchange Rate Sensitivity': ['XRate_104W_USD'],
            'Profitability': ['Return_on_Assets_Annual'] }

    DescriptorWeights = {'Earnings Yield': [0.25, 0.75]}
    smallCapMap = {'MidCap': [80.0, 97.5],}
    noProxyList = ['Dividend Yield']
    fillMissingList = ['Earnings Yield', 'Value', 'Leverage', 'Growth', 'Profitability']
    fillWithZeroList = ['Dividend Yield']
    shrinkList = {'Liquidity': 60,
                  'Market Sensitivity': 250,
                  'Volatility': 125,
                  'Medium-Term Momentum': 250}
    orthogList = {'Volatility': [['Market Sensitivity'], True, 1.0]}

    # Setting up market intercept if relevant
    interceptFactor = 'Market Intercept'
    intercept = ModelFactor(interceptFactor, interceptFactor)
    industryClassification = Classification.GICSCustomJP(datetime.date(2016,9,1))

    estuAssetTypes = ['REIT', 'Com']

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.JPResearchModelFinalESTU')
        # Set up relevant styles to be created/used
        ModelParameters2017.defaultExposureParameters(self, self.styleList)
        self.styles = [s for s in self.totalStyles if s.name in self.styleList]
        EquityModel.FundamentalModel.__init__(self, ['SEDOL'], modelDB, marketDB)

        # Set up regression parameters
        dummyThreshold = 10

        # Run risk based on the square root of market cap weights  
        self.internalCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType='Industry Groups',
                dummyThreshold=dummyThreshold,
                marketRegression=False,
                kappa=5.0,
                useRealMCaps=True,
                regWeight = 'rootCap',
                )

        # Set up external regression parameters
        self.returnCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType='Industry Groups',
                dummyThreshold=dummyThreshold,
                marketRegression=False,
                kappa=None,
                useRealMCaps=True,
                regWeight='invSpecificVariance'
                )

        # This controls the FMP regression
        self.fmpCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType='Industry Groups',
                marketRegression=False,
                dummyThreshold=dummyThreshold,
                kappa=None)

        # Set up risk parameters
        ModelParameters2017.defaultFundamentalCovarianceParameters(
                self, nwLag=2,
                varDVAOnly=False, unboundedDVA=False,
                )

        # Set up standardization parameters
        gloScope = Standardization.GlobalRelativeScope([f.name for f in self.styles])
        self.exposureStandardization = Standardization.BucketizedStandardization([gloScope],
                fillWithZeroList=self.fillWithZeroList)

        # Set up descriptor standardization parameters
        descriptors = sorted(list(set([item for sublist
            in list(self.DescriptorMap.values()) for item in sublist])))
        exceptionNames = [self.DescriptorMap[sf] for sf in self.noProxyList]
        exceptionNames = list(itertools.chain.from_iterable(exceptionNames))
        self.descriptorStandardization = Standardization.BucketizedStandardization(
                    [Standardization.GlobalRelativeScope(descriptors)],
                    mad_bound=15.0, fancyMAD=self.fancyMAD, exceptionNames=exceptionNames)

        self.indexSelector = MarketIndex.MarketIndexSelector(modelDB, marketDB)
        # Set up TOPIX replication - FIXME
        self.topixReplicator = TOPIX.TOPIXReplicator(self, modelDB)
        modelDB.createCurrencyCache(marketDB)

    def generate_model_specific_exposures(self, modelDate, data, modelDB, marketDB):
        """Generate the non-default factors.
        """
        beta = numpy.zeros((len(data.universe)), float)
        # Cap-based style factors here
        if not hasattr(self, 'estuMap') or self.estuMap is None:
            return data.exposureMatrix

        # Small-cap factors
        dateList = modelDB.getDates(self.rmg, modelDate, 61, excludeWeekend=True)
        styleNames = [s.name for s in self.styles]
        scList = [sc for sc in self.estuMap.keys() if sc in styleNames]
        for sc in scList:
            beta = Matrices.allMasked((len(data.universe)), float)
            scAssets = [sid for sid in self.estuMap[sc].assets if sid in data.universe]
            if len(scAssets) < 1:
                logging.warning('No assets in %s universe', sc)
            else:
                qualifiedAssets = modelDB.loadESTUQualifyHistory(
                    self.rms_id, scAssets, dateList, estuInstance=self.estuMap[sc])
                qualifiedAssets = ma.filled(ma.sum(qualifiedAssets.data, axis=1), 0.0)
                if len(qualifiedAssets) < 1:
                    logging.warning('No assets qualified for %s factor', sc)
                else:
                    qualifiedAssets = qualifiedAssets / float(numpy.max(qualifiedAssets, axis=None))
                    for (i_c, idx) in enumerate([data.assetIdxMap[sid] for sid in scAssets]):
                        beta[idx] = qualifiedAssets[i_c]
            data.exposureMatrix.addFactor(sc, beta, ExposureMatrix.StyleFactor)
            if self.exposureStandardization.exceptionNames is None:
                self.exposureStandardization.exceptionNames = [sc]
            else:
                self.exposureStandardization.exceptionNames.append(sc)
        self.exposureStandardization.exceptionNames = list(set(self.exposureStandardization.exceptionNames))

        return data.exposureMatrix

    def generate_estimation_universe_old(self, modelDate, data, modelDB, marketDB, excludeFactors=None):
        """Estimation universe selection criteria for JP.
        """
        import pandas as pd
        #import ipdb;ipdb.set_trace()
        self.log.info('generate_estimation_universe for JP Model: begin')
        buildEstu = EstimationUniverse.ConstructEstimationUniverse(data.universe, self, modelDB, marketDB)

        # Set up various eligible and total universes
        universeIdx = list(range(len(buildEstu.assets)))
        originalEligibleUniverse = list(data.eligibleUniverse)
        originalEligibleUniverseIdx = [data.assetIdxMap[sid] for sid in originalEligibleUniverse]

        # Remove nursery market assets
        if len(data.nurseryUniverse) > 0:
            logging.info('Checking for assets from nursery markets')
            ns_indices = [data.assetIdxMap[sid] for sid in data.nurseryUniverse]
            (eligibleUniverseIdx, nonest) = buildEstu.exclude_specific_assets(
                    ns_indices, baseEstu=originalEligibleUniverseIdx)
            if n != len(eligibleUniverseIdx):
                n = len(eligibleUniverseIdx)
                logging.info('ESTU currently stands at %d stocks', n)
        else:
            eligibleUniverseIdx = originalEligibleUniverseIdx

        universe = pd.DataFrame(buildEstu.assets,columns=['SubID_Obj'])
        universeIdx = universe.index
        original_eligibleUniverse = pd.DataFrame(list(zip(data.eligibleUniverse,originalEligibleUniverseIdx)),columns=['SubID_Obj','originalEligibleUniverseIdx'])

        logging.info('ESTU currently stands at %d stocks based on original eligible universe', len(originalEligibleUniverse))

        # Report on thinly-traded assets over the entire universe
        logging.info('Looking for thinly-traded stocks')
        # (nonSparseIdx, sparse) = buildEstu.exclude_thinly_traded_assets(modelDate, data, baseEstu=universeIdx, minNonZero=0.75)
        # data.nonSparse = numpy.take(buildEstu.assets, nonSparseIdx, axis=0)
        (non_thin_idx, thin_idx) = buildEstu.exclude_thinly_traded_assets(modelDate, data, baseEstu=universe.index.values, minNonZero=0.75)
        # (non_thin_idx, thin_idx) = buildEstu.exclude_thinly_traded_assets(modelDate, data, baseEstu=universe.index.values, minNonZero=0.95)
        data.nonSparse = pd.DataFrame(universe,index=non_thin_idx)["SubID_Obj"].tolist()

        # Exclude thinly traded assets
        estu_withoutThin_idx = list(set(eligibleUniverseIdx).intersection(set(non_thin_idx)))
        logging.info('ESTU currently stands at %d stocks after Filtering thinly-traded assets', len(estu_withoutThin_idx))
        # (2) Filtering tiny-cap assets by market, country and industry
        # (a) Weed out tiny-cap assets by market
        lowerBound = 1
        logging.info('Filtering by top %d%% mcap on entire market', 100-lowerBound)
        (large_byMkt_idx, nonest1) = buildEstu.exclude_by_cap_ranking(data, modelDate, baseEstu=estu_withoutThin_idx,
                                                                     lower_pctile=lowerBound, method='percentage')
        # (2b) Weed out tiny-cap assets by country
        lowerBound = 5
        logging.info('Filtering by top %d%% mcap on country', 100-lowerBound)
        (large_byCntry_idx, nonest2) = buildEstu.exclude_by_cap_ranking(data, modelDate, baseEstu=estu_withoutThin_idx,
                                                                       byFactorType=ExposureMatrix.CountryFactor,
                                                                       lower_pctile=lowerBound, method='percentage',
                                                                       excludeFactors=excludeFactors)
        # (2c) Perform similar check by industry
        logging.info('Filtering by top %d%% mcap on industry', 100-lowerBound)
        (large_byIndtry_idx, nonest3) = buildEstu.exclude_by_cap_ranking(
               data, modelDate, baseEstu=estu_withoutThin_idx,
               byFactorType=ExposureMatrix.IndustryFactor,
               lower_pctile=lowerBound, method='percentage',
               excludeFactors=excludeFactors)
    
        estu_mktCap_idx = set(large_byMkt_idx).union(large_byCntry_idx).union(large_byIndtry_idx)
        estu_mktCap_idx = list(estu_mktCap_idx)
        tinyCap_idx = list(set(estu_withoutThin_idx).difference(estu_mktCap_idx))

        logging.info('ESTU currently stands at %d stocks after Filtering by Market Cap.', len(estu_mktCap_idx))

        # candid_univ_idx = eligibleUniverseIdx
        candid_univ_idx = estu_withoutThin_idx
        # Inflate any thin countries or industries - add 2*
        # minFactorWidth=2*self.returnCalculator.allParameters[0].getThinFactorInformation().dummyThreshold
        minFactorWidth=self.returnCalculator.allParameters[0].getThinFactorInformation().dummyThreshold
        logging.info('Inflating any thin factors')

        (estu_inflated_idx, nonest,herf_num_list) = buildEstu.pump_up_factors2(
                data, modelDate, currentEstu=estu_mktCap_idx,
                baseEstu=candid_univ_idx,
                byFactorType=[ExposureMatrix.CountryFactor, ExposureMatrix.IndustryFactor],
                minFactorWidth=minFactorWidth,
                cutOff = self.inflation_cutoff, 
                excludeFactors=excludeFactors)

        logging.info('ESTU currently stands at %d stocks', len(estu_inflated_idx))
        herf_num_list = pd.DataFrame(herf_num_list)
        herf_num_list.to_csv('herf_num_list.csv')
        # Apply grandfathering rules
        logging.info('Incorporating grandfathering')
        (estu_final_Idx, ESTUQualify, nonest) = buildEstu.grandfather(modelDate,
                estu_inflated_idx, baseEstu=candid_univ_idx,
                estuInstance=self.estuMap['main'])

        totalcap = ma.sum(ma.take(data.marketCaps, estu_final_Idx, axis=0), axis=0) / 1e9
        self.log.info('Final estu contains %d assets, %.2f bn (%s)',len(estu_final_Idx), totalcap, self.numeraire.currency_code)

        self.log.info('Final estu contains %d assets in thin-traded assets.',sum(pd.Series(estu_inflated_idx).isin(thin_idx)))
        self.log.info('Final estu contains %d assets in tiny cap assets.',sum(pd.Series(estu_inflated_idx).isin(tinyCap_idx)))
        self.log.debug('generate_estimation_universe: end')

        # If we have a family of estimation universes, populate the main estu accordingly
        self.estuMap['main'].assets = [buildEstu.assets[idx] for idx in estu_final_Idx]
        self.estuMap['main'].qualify = [buildEstu.assets[idx] for idx in ESTUQualify]

        return estu_final_Idx

    def generate_estimation_universe(self, modelDate, data, modelDB, marketDB, excludeFactors=None):
        estu1 = self.generate_estimation_universe_v2(modelDate, data, modelDB, marketDB, excludeFactors=None)
        # estu2 = self.generate_estimation_universe_old(modelDate, data, modelDB, marketDB, excludeFactors=None)
        # print modelDate, estu1, estu2
        return estu1

class JPResearchModel2(RiskModels_V3.JPAxioma2009MH_S):
    """Japan research model 2
    """
    rm_id = -5
    revision = 2
    rms_id = -9
    numFactors = 15
    blind = [ModelFactor('Statistical Factor %d' % n,
        'Statistical Factor %d' % n)
        for n in range(1, numFactors+1)]
    returnHistory = 250
    industryClassification = Classification.GICSCustomJP(
                                        datetime.date(2008,8,30))

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.JPResearchModel2')
        RiskModels_V3.JPAxioma2009MH_S.__init__(self, modelDB, marketDB)
        self.returnCalculator = ReturnCalculator. \
                            AsymptoticPrincipalComponents2(self.numFactors)
        # Set up risk parameters
        varParameters = {'halfLife': 125, 'minObs': 250, 'maxObs': 250,
                         'NWLag': 1, 'DVAWindow': 125, 'DVAType': 'step'}
        corrParameters = {'halfLife': 250, 'minObs': 250, 'maxObs': 250,
                          'NWLag': 1, 'DVAWindow': 125, 'DVAType': 'step'}
        fullCovParameters = {}
        srParameters = {'halfLife': 125, 'minObs': 125, 'maxObs': 125,
                        'NWLag': 1, 'clipBounds': (-15.0,18.0)}

        self.vp = RiskCalculator.RiskParameters2009(varParameters)
        self.cp = RiskCalculator.RiskParameters2009(corrParameters)
        self.fp = RiskCalculator.RiskParameters2009(fullCovParameters)
        self.sp = RiskCalculator.RiskParameters2009(srParameters)

        self.covarianceCalculator = RiskCalculator.\
                        CompositeCovarianceMatrix2009(self.fp, self.vp, self.cp)
        self.specificRiskCalculator = RiskCalculator.\
                        BrilliantSpecificRisk2009(self.sp)
        self.jpModel = JPResearchModel1(modelDB, marketDB)

#class JPResearchModel3(JPResearchModel1):
#    """JP research model 3: Effect of Leverage definition - Debt/Assets instead of Debt/Assets and Debt/Equity
#    """
#    rm_id = -5
#    revision = 3
#    rms_id = -1001
#    standardizationStats = True
#    globalDescriptorModel = True        # Uses the global descriptor model
#    DLCEnabled = True                   # Combines mcap for DLCs
#    SCM = True
#
#    # List of style factors in the model
#    styleList = [
#                'Earnings Yield',
#                'Value',
#                'Leverage',
#                'Growth',
#                'Profitability',
#                'Dividend Yield',
#                'Size',
#                'Liquidity',
#                'Market Sensitivity',
#                'Volatility',
#                'Medium-Term Momentum',
#                'MidCap',
#                'Exchange Rate Sensitivity',
#                ]
#
#    QtrlyDescriptorMap = {
#            'Earnings Yield': ['Earnings_to_Price_Quarterly', 'Est_Earnings_to_Price_12MFL_Quarterly'],
#            'Value': ['Book_to_Price_Quarterly'],
#            'Leverage': ['Debt_to_Assets_Quarterly'],
#            'Growth': ['Earnings_Growth_RPF_AFQ', 'Sales_Growth_RPF_AFQ'],
#            'Dividend Yield': ['Dividend_Yield_Quarterly'],
#            'Size': ['LnIssuerCap'],
#            'Liquidity': ['LnTrading_Activity_60D'],
#            'Market Sensitivity': ['Market_Sensitivity_250D'],
#            'Volatility': ['Volatility_125D'],
#            'Medium-Term Momentum': ['Momentum_250x20D'],
#            'Exchange Rate Sensitivity': ['XRate_104W_XDR'],
#            'Profitability': ['Return_on_Equity_Quarterly', 'Return_on_Assets_Quarterly',
#                              'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
#                              'Sales_to_Assets_Quarterly', 'Gross_Margin_Quarterly'],
#            }
#
#    DescriptorMap = {
#            'Earnings Yield': ['Earnings_to_Price_Annual', 'Est_Earnings_to_Price_12MFL_Annual'],
#            'Value': ['Book_to_Price_Annual'],
#            'Leverage': ['Debt_to_Assets_Annual'],
#            'Growth': ['Earnings_Growth_RPF_Annual', 'Sales_Growth_RPF_Annual'],
#            'Dividend Yield': ['Dividend_Yield_Annual'],
#            'Size': ['LnIssuerCap'],
#            'Liquidity': ['LnTrading_Activity_60D'],
#            'Market Sensitivity': ['Market_Sensitivity_250D'],
#            'Volatility': ['Volatility_125D'],
#            'Medium-Term Momentum': ['Momentum_250x20D'],
#            'Exchange Rate Sensitivity': ['XRate_104W_XDR'],
#            'Profitability': ['Return_on_Equity_Annual', 'Return_on_Assets_Annual',
#                              'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
#                              'Sales_to_Assets_Annual', 'Gross_Margin_Annual'],
#            }
#
#    DescriptorWeights = {'Earnings Yield': [0.75, 0.25]}
#    smallCapMap = {'MidCap': [80.0, 90.0],}
#    noProxyList = ['Dividend Yield']
#    fillMissingList = ['Earnings Yield', 'Value', 'Leverage', 'Growth', 'Profitability']
#    fillWithZeroList = ['Dividend Yield']
#    shrinkList = {'Liquidity': 60,
#                  'Market Sensitivity': 250,
#                  'Volatility': 125,
#                  'Medium-Term Momentum': 250}
#    orthogList = {'Volatility': [['Market Sensitivity'], True, 1.0]}
#
#    # Setting up market intercept if relevant
#    interceptFactor = 'Market Intercept'
#    intercept = ModelFactor(interceptFactor, interceptFactor)
#    # industryClassification = Classification.GICSIndustries(datetime.date(2014,3,1))
#    industryClassification = Classification.GICSCustomJP(
#                                                datetime.date(2008,8,30))
#    estuAssetTypes = ['REIT', 'Com']
#
#    def __init__(self, modelDB, marketDB):
#        self.log = logging.getLogger('RiskModels.JPResearchModel3')
#        # Set up relevant styles to be created/used
#        ModelParameters.defaultExposureParametersV3(self, self.styleList)
#        self.styles = [s for s in self.totalStyles if s.name in self.styleList]
#        MFM.FundamentalModel.__init__(self, ['SEDOL'], modelDB, marketDB)

class JPResearchModel4(EquityModel.FundamentalModel):
    """ Same as Final JP research model (JPResearchModelFinalESTU), but Profitability is ROE and ROA only
            1. Split EY and Value
            2. Combine D/A and D/E in Leverage
            3. Include profitability factor
            4. Include Dividend Yield factor
            5. Enhanced ESTU coverage (based on inclusion of several exchanges)
            6. Updated definition of MidCap based on clear inclusion criteria - similar to that of Russell Japan Mid Cap Index 80-97.5%
            7. Updated definitions of Growth, Market sensitivity, MTM, Volatility, Exchange Rate sensitivity
            8. GICS2016 definitions
    """
    estu_parameters = {
                       'minNonZero':0.75,
                       'minNonMissing':0.95,
                       'maskZeroWithNoADV_Flag': True,
                       'returnLegacy_Flag': False,
                       'CapByNumber_Flag':False,
                       'CapByNumber_hiCapQuota':np.nan,
                       'CapByNumber_lowCapQuota':np.nan,
                       'market_lower_pctile':1 ,
                       'country_lower_pctile':5,
                       'industry_lower_pctile':5,
                       'dummyThreshold': 10,
                       'inflation_cutoff':0.01
                        }
    k = 5.0
    minNonZero = 0.75
    rm_id = -5
    revision = 4
    rms_id = -1002
    inflation_cutoff = 0.01
    regionalDescriptorStructure = False
    twoRegressionStructure = True
    multiCountry = False

    # List of 13 style factors in the model
    styleList = [
                'Earnings Yield',
                'Value',
                'Leverage',
                'Growth',
                'Profitability',
                'Dividend Yield',
                'Size',
                'Liquidity',
                'Market Sensitivity',
                'Volatility',
                'Medium-Term Momentum',
                'MidCap',
                'Exchange Rate Sensitivity',
                ]

    DescriptorMap = {
            'Earnings Yield': ['Earnings_to_Price_Annual', 'Est_Earnings_to_Price_12MFL_Annual'],
            'Value': ['Book_to_Price_Annual'],
            'Leverage': ['Debt_to_Assets_Annual', 'Debt_to_Equity_Annual'],
            'Growth': ['Earnings_Growth_RPF_Annual', 'Sales_Growth_RPF_Annual'],
            'Dividend Yield': ['Dividend_Yield_Annual'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_60D'],
            'Market Sensitivity': ['Market_Sensitivity_250D'],
            'Volatility': ['Volatility_125D'],
            'Medium-Term Momentum': ['Momentum_250x20D'],
            'Exchange Rate Sensitivity': ['XRate_104W_USD'],
            'Profitability': ['Return_on_Equity_Annual', 'Return_on_Assets_Annual'] }

    DescriptorWeights = {'Earnings Yield': [0.25, 0.75]}
    smallCapMap = {'MidCap': [80.0, 97.5],}
    noProxyList = ['Dividend Yield']
    fillMissingList = ['Earnings Yield', 'Value', 'Leverage', 'Growth', 'Profitability']
    fillWithZeroList = ['Dividend Yield']
    shrinkList = {'Liquidity': 60,
                  'Market Sensitivity': 250,
                  'Volatility': 125,
                  'Medium-Term Momentum': 250}
    orthogList = {'Volatility': [['Market Sensitivity'], True, 1.0]}

    # Setting up market intercept if relevant
    interceptFactor = 'Market Intercept'
    intercept = ModelFactor(interceptFactor, interceptFactor)
    industryClassification = Classification.GICSCustomJP(datetime.date(2016,9,1))

    estuAssetTypes = ['REIT', 'Com']

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.JPResearchModelFinalESTU')
        # Set up relevant styles to be created/used
        ModelParameters2017.defaultExposureParameters(self, self.styleList)
        self.styles = [s for s in self.totalStyles if s.name in self.styleList]
        EquityModel.FundamentalModel.__init__(self, ['SEDOL'], modelDB, marketDB)

        # Set up regression parameters
        dummyThreshold = 10

        # Run risk based on the square root of market cap weights  
        self.internalCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType='Industry Groups',
                dummyThreshold=dummyThreshold,
                marketRegression=False,
                kappa=5.0,
                useRealMCaps=True,
                regWeight = 'rootCap',
                )

        # Set up external regression parameters
        self.returnCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType='Industry Groups',
                dummyThreshold=dummyThreshold,
                marketRegression=False,
                kappa=None,
                useRealMCaps=True,
                regWeight='invSpecificVariance'
                )

        # This controls the FMP regression
        self.fmpCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType='Industry Groups',
                marketRegression=False,
                dummyThreshold=dummyThreshold,
                kappa=None)

        # Set up risk parameters
        ModelParameters2017.defaultFundamentalCovarianceParameters(
                self, nwLag=2,
                varDVAOnly=False, unboundedDVA=False,
                )

        # Set up standardization parameters
        gloScope = Standardization.GlobalRelativeScope([f.name for f in self.styles])
        self.exposureStandardization = Standardization.BucketizedStandardization([gloScope],
                fillWithZeroList=self.fillWithZeroList)

        # Set up descriptor standardization parameters
        descriptors = sorted(list(set([item for sublist
            in list(self.DescriptorMap.values()) for item in sublist])))
        exceptionNames = [self.DescriptorMap[sf] for sf in self.noProxyList]
        exceptionNames = list(itertools.chain.from_iterable(exceptionNames))
        self.descriptorStandardization = Standardization.BucketizedStandardization(
                    [Standardization.GlobalRelativeScope(descriptors)],
                    mad_bound=15.0, fancyMAD=self.fancyMAD, exceptionNames=exceptionNames)

        self.indexSelector = MarketIndex.MarketIndexSelector(modelDB, marketDB)
        # Set up TOPIX replication - FIXME
        self.topixReplicator = TOPIX.TOPIXReplicator(self, modelDB)
        modelDB.createCurrencyCache(marketDB)

    def generate_model_specific_exposures(self, modelDate, data, modelDB, marketDB):
        """Generate the non-default factors.
        """
        beta = numpy.zeros((len(data.universe)), float)
        # Cap-based style factors here
        if not hasattr(self, 'estuMap') or self.estuMap is None:
            return data.exposureMatrix

        # Small-cap factors
        dateList = modelDB.getDates(self.rmg, modelDate, 61, excludeWeekend=True)
        styleNames = [s.name for s in self.styles]
        scList = [sc for sc in self.estuMap.keys() if sc in styleNames]
        for sc in scList:
            beta = Matrices.allMasked((len(data.universe)), float)
            scAssets = [sid for sid in self.estuMap[sc].assets if sid in data.universe]
            if len(scAssets) < 1:
                logging.warning('No assets in %s universe', sc)
            else:
                qualifiedAssets = modelDB.loadESTUQualifyHistory(
                    self.rms_id, scAssets, dateList, estuInstance=self.estuMap[sc])
                qualifiedAssets = ma.filled(ma.sum(qualifiedAssets.data, axis=1), 0.0)
                if len(qualifiedAssets) < 1:
                    logging.warning('No assets qualified for %s factor', sc)
                else:
                    qualifiedAssets = qualifiedAssets / float(numpy.max(qualifiedAssets, axis=None))
                    for (i_c, idx) in enumerate([data.assetIdxMap[sid] for sid in scAssets]):
                        beta[idx] = qualifiedAssets[i_c]
            data.exposureMatrix.addFactor(sc, beta, ExposureMatrix.StyleFactor)
            if self.exposureStandardization.exceptionNames is None:
                self.exposureStandardization.exceptionNames = [sc]
            else:
                self.exposureStandardization.exceptionNames.append(sc)
        self.exposureStandardization.exceptionNames = list(set(self.exposureStandardization.exceptionNames))

        return data.exposureMatrix

    def generate_estimation_universe_old(self, modelDate, data, modelDB, marketDB, excludeFactors=None):
        """Estimation universe selection criteria for JP.
        """
        import pandas as pd
        #import ipdb;ipdb.set_trace()
        self.log.info('generate_estimation_universe for JP Model: begin')
        buildEstu = EstimationUniverse.ConstructEstimationUniverse(data.universe, self, modelDB, marketDB)

        # Set up various eligible and total universes
        universeIdx = list(range(len(buildEstu.assets)))
        originalEligibleUniverse = list(data.eligibleUniverse)
        originalEligibleUniverseIdx = [data.assetIdxMap[sid] for sid in originalEligibleUniverse]

        # Remove nursery market assets
        if len(data.nurseryUniverse) > 0:
            logging.info('Checking for assets from nursery markets')
            ns_indices = [data.assetIdxMap[sid] for sid in data.nurseryUniverse]
            (eligibleUniverseIdx, nonest) = buildEstu.exclude_specific_assets(
                    ns_indices, baseEstu=originalEligibleUniverseIdx)
            if n != len(eligibleUniverseIdx):
                n = len(eligibleUniverseIdx)
                logging.info('ESTU currently stands at %d stocks', n)
        else:
            eligibleUniverseIdx = originalEligibleUniverseIdx

        universe = pd.DataFrame(buildEstu.assets,columns=['SubID_Obj'])
        universeIdx = universe.index
        original_eligibleUniverse = pd.DataFrame(list(zip(data.eligibleUniverse,originalEligibleUniverseIdx)),columns=['SubID_Obj','originalEligibleUniverseIdx'])

        logging.info('ESTU currently stands at %d stocks based on original eligible universe', len(originalEligibleUniverse))

        # Report on thinly-traded assets over the entire universe
        logging.info('Looking for thinly-traded stocks')
        # (nonSparseIdx, sparse) = buildEstu.exclude_thinly_traded_assets(modelDate, data, baseEstu=universeIdx, minNonZero=0.75)
        # data.nonSparse = numpy.take(buildEstu.assets, nonSparseIdx, axis=0)
        (non_thin_idx, thin_idx) = buildEstu.exclude_thinly_traded_assets(modelDate, data, baseEstu=universe.index.values, minNonZero=0.75)
        # (non_thin_idx, thin_idx) = buildEstu.exclude_thinly_traded_assets(modelDate, data, baseEstu=universe.index.values, minNonZero=0.95)
        data.nonSparse = pd.DataFrame(universe,index=non_thin_idx)["SubID_Obj"].tolist()

        # Exclude thinly traded assets
        estu_withoutThin_idx = list(set(eligibleUniverseIdx).intersection(set(non_thin_idx)))
        logging.info('ESTU currently stands at %d stocks after Filtering thinly-traded assets', len(estu_withoutThin_idx))
        # (2) Filtering tiny-cap assets by market, country and industry
        # (a) Weed out tiny-cap assets by market
        lowerBound = 1
        logging.info('Filtering by top %d%% mcap on entire market', 100-lowerBound)
        (large_byMkt_idx, nonest1) = buildEstu.exclude_by_cap_ranking(data, modelDate, baseEstu=estu_withoutThin_idx,
                                                                     lower_pctile=lowerBound, method='percentage')
        # (2b) Weed out tiny-cap assets by country
        lowerBound = 5
        logging.info('Filtering by top %d%% mcap on country', 100-lowerBound)
        (large_byCntry_idx, nonest2) = buildEstu.exclude_by_cap_ranking(data, modelDate, baseEstu=estu_withoutThin_idx,
                                                                       byFactorType=ExposureMatrix.CountryFactor,
                                                                       lower_pctile=lowerBound, method='percentage',
                                                                       excludeFactors=excludeFactors)
        # (2c) Perform similar check by industry
        logging.info('Filtering by top %d%% mcap on industry', 100-lowerBound)
        (large_byIndtry_idx, nonest3) = buildEstu.exclude_by_cap_ranking(
               data, modelDate, baseEstu=estu_withoutThin_idx,
               byFactorType=ExposureMatrix.IndustryFactor,
               lower_pctile=lowerBound, method='percentage',
               excludeFactors=excludeFactors)
    
        estu_mktCap_idx = set(large_byMkt_idx).union(large_byCntry_idx).union(large_byIndtry_idx)
        estu_mktCap_idx = list(estu_mktCap_idx)
        tinyCap_idx = list(set(estu_withoutThin_idx).difference(estu_mktCap_idx))

        logging.info('ESTU currently stands at %d stocks after Filtering by Market Cap.', len(estu_mktCap_idx))

        # candid_univ_idx = eligibleUniverseIdx
        candid_univ_idx = estu_withoutThin_idx
        # Inflate any thin countries or industries - add 2*
        # minFactorWidth=2*self.returnCalculator.allParameters[0].getThinFactorInformation().dummyThreshold
        minFactorWidth=self.returnCalculator.allParameters[0].getThinFactorInformation().dummyThreshold
        logging.info('Inflating any thin factors')

        (estu_inflated_idx, nonest,herf_num_list) = buildEstu.pump_up_factors2(
                data, modelDate, currentEstu=estu_mktCap_idx,
                baseEstu=candid_univ_idx,
                byFactorType=[ExposureMatrix.CountryFactor, ExposureMatrix.IndustryFactor],
                minFactorWidth=minFactorWidth,
                cutOff = self.inflation_cutoff, 
                excludeFactors=excludeFactors)

        logging.info('ESTU currently stands at %d stocks', len(estu_inflated_idx))
        herf_num_list = pd.DataFrame(herf_num_list)
        herf_num_list.to_csv('herf_num_list.csv')
        # Apply grandfathering rules
        logging.info('Incorporating grandfathering')
        (estu_final_Idx, ESTUQualify, nonest) = buildEstu.grandfather(modelDate,
                estu_inflated_idx, baseEstu=candid_univ_idx,
                estuInstance=self.estuMap['main'])

        totalcap = ma.sum(ma.take(data.marketCaps, estu_final_Idx, axis=0), axis=0) / 1e9
        self.log.info('Final estu contains %d assets, %.2f bn (%s)',len(estu_final_Idx), totalcap, self.numeraire.currency_code)

        self.log.info('Final estu contains %d assets in thin-traded assets.',sum(pd.Series(estu_inflated_idx).isin(thin_idx)))
        self.log.info('Final estu contains %d assets in tiny cap assets.',sum(pd.Series(estu_inflated_idx).isin(tinyCap_idx)))
        self.log.debug('generate_estimation_universe: end')

        # If we have a family of estimation universes, populate the main estu accordingly
        self.estuMap['main'].assets = [buildEstu.assets[idx] for idx in estu_final_Idx]
        self.estuMap['main'].qualify = [buildEstu.assets[idx] for idx in ESTUQualify]

        return estu_final_Idx

    def generate_estimation_universe(self, modelDate, data, modelDB, marketDB, excludeFactors=None):
        estu1 = self.generate_estimation_universe_v2(modelDate, data, modelDB, marketDB, excludeFactors=None)
        # estu2 = self.generate_estimation_universe_old(modelDate, data, modelDB, marketDB, excludeFactors=None)
        # print modelDate, estu1, estu2
        return estu1

#class JPResearchModel4(JPResearchModel1):
#    """JP research model 4: Effect of Value (no EY)
#    """
#    rm_id = -5
#    revision = 4
#    rms_id = -1002
#    standardizationStats = True
#    globalDescriptorModel = True        # Uses the global descriptor model
#    DLCEnabled = True                   # Combines mcap for DLCs
#    SCM = True
#
#    # List of style factors in the model
#    styleList = [
#                # 'Earnings Yield',
#                'Value',
#                'Leverage',
#                'Growth',
#                'Profitability',
#                'Dividend Yield',
#                'Size',
#                'Liquidity',
#                'Market Sensitivity',
#                'Volatility',
#                'Medium-Term Momentum',
#                'MidCap',
#                'Exchange Rate Sensitivity',
#                ]
#
#    QtrlyDescriptorMap = {
#            'Value': ['Book_to_Price_Quarterly'],
#            'Leverage': ['Debt_to_Assets_Quarterly', 'Debt_to_Equity_Quarterly'],
#            'Growth': ['Earnings_Growth_RPF_AFQ', 'Sales_Growth_RPF_AFQ'],
#            'Dividend Yield': ['Dividend_Yield_Quarterly'],
#            'Size': ['LnIssuerCap'],
#            'Liquidity': ['LnTrading_Activity_60D'],
#            'Market Sensitivity': ['Market_Sensitivity_250D'],
#            'Volatility': ['Volatility_125D'],
#            'Medium-Term Momentum': ['Momentum_250x20D'],
#            'Exchange Rate Sensitivity': ['XRate_104W_XDR'],
#            'Profitability': ['Return_on_Equity_Quarterly', 'Return_on_Assets_Quarterly',
#                              'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
#                              'Sales_to_Assets_Quarterly', 'Gross_Margin_Quarterly'],
#            }
#
#    DescriptorMap = {
#            'Value': ['Book_to_Price_Annual', 'Earnings_to_Price_Annual', 'Est_Earnings_to_Price_12MFL_Annual'],
#            'Leverage': ['Debt_to_Assets_Annual', 'Debt_to_Equity_Annual'],
#            'Growth': ['Earnings_Growth_RPF_Annual', 'Sales_Growth_RPF_Annual'],
#            'Dividend Yield': ['Dividend_Yield_Annual'],
#            'Size': ['LnIssuerCap'],
#            'Liquidity': ['LnTrading_Activity_60D'],
#            'Market Sensitivity': ['Market_Sensitivity_250D'],
#            'Volatility': ['Volatility_125D'],
#            'Medium-Term Momentum': ['Momentum_250x20D'],
#            'Exchange Rate Sensitivity': ['XRate_104W_XDR'],
#            'Profitability': ['Return_on_Equity_Annual', 'Return_on_Assets_Annual',
#                              'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
#                              'Sales_to_Assets_Annual', 'Gross_Margin_Annual'],
#            }
#
#    DescriptorWeights = {'Value':[0.5, 0.75/2, 0.25/2]}
#    smallCapMap = {'MidCap': [80.0, 90.0],}
#    noProxyList = ['Dividend Yield']
#    fillMissingList = ['Value', 'Leverage', 'Growth', 'Profitability']
#    fillWithZeroList = ['Dividend Yield']
#    shrinkList = {'Liquidity': 60,
#                  'Market Sensitivity': 250,
#                  'Volatility': 125,
#                  'Medium-Term Momentum': 250}
#    orthogList = {'Volatility': [['Market Sensitivity'], True, 1.0]}
#
#    # Setting up market intercept if relevant
#    interceptFactor = 'Market Intercept'
#    intercept = ModelFactor(interceptFactor, interceptFactor)
#    industryClassification = Classification.GICSCustomJP(
#                                                datetime.date(2008,8,30))
#    estuAssetTypes = ['REIT', 'Com']
#
#    def __init__(self, modelDB, marketDB):
#        self.log = logging.getLogger('RiskModels.JPResearchModel4')
#        # Set up relevant styles to be created/used
#        ModelParameters.defaultExposureParametersV3(self, self.styleList)
#        self.styles = [s for s in self.totalStyles if s.name in self.styleList]
#        MFM.FundamentalModel.__init__(self, ['SEDOL'], modelDB, marketDB)
#
#        # Set up regression parameters
#        dummyThreshold = 10
#        self.returnCalculator = ModelParameters.defaultRegressionParameters(
#                self, modelDB,
#                dummyType='Industry Groups',
#                dummyThreshold=dummyThreshold,
#                marketRegression=False,
#                kappa=5.0,
#                )
#
#        # This controls the FMP regression
#        self.fmpCalculator = ModelParameters.defaultRegressionParameters(
#                self, modelDB,
#                dummyType='Industry Groups',
#                marketRegression=False,
#                dummyThreshold=dummyThreshold,
#                kappa=None)
#
#        # Set up risk parameters
#        ModelParameters.defaultFundamentalCovarianceParametersV3(
#                self, nwLag=2,
#                varDVAOnly=False, unboundedDVA=False,
#                )
#
#        # Set up standardization parameters
#        gloScope = Standardization.GlobalRelativeScope([f.name for f in self.styles])
#        self.exposureStandardization = Standardization.BucketizedStandardization([gloScope],
#                fillWithZeroList=self.fillWithZeroList)
#
#        # Set up descriptor standardization parameters
#        descriptors = sorted(list(set([item for sublist
#            in self.DescriptorMap.values() for item in sublist])))
#        exceptionNames = [self.DescriptorMap[sf] for sf in self.noProxyList]
#        exceptionNames = list(itertools.chain.from_iterable(exceptionNames))
#        if not self.SCM:
#            self.descriptorStandardization = Standardization.BucketizedStandardization(
#                    [Standardization.RegionRelativeScope(modelDB, descriptors)],
#                    mad_bound=15.0, fancyMAD=self.fancyMAD, exceptionNames=exceptionNames)
#        else:
#            self.descriptorStandardization = Standardization.BucketizedStandardization(
#                    [Standardization.GlobalRelativeScope(descriptors)],
#                    mad_bound=15.0, fancyMAD=self.fancyMAD, exceptionNames=exceptionNames)
#
#        self.indexSelector = MarketIndex.MarketIndexSelector(modelDB, marketDB)
#        # Set up TOPIX replication - FIXME
#        self.topixReplicator = TOPIX.TOPIXReplicator(self, modelDB)
#        modelDB.createCurrencyCache(marketDB)

class JPResearchModel5(EquityModel.FundamentalModel):
    """ Same as Final JP research model (JPResearchModelFinalESTU), but Profitability does not have ROE and ROA 
            1. Split EY and Value
            2. Combine D/A and D/E in Leverage
            3. Include profitability factor
            4. Include Dividend Yield factor
            5. Enhanced ESTU coverage (based on inclusion of several exchanges)
            6. Updated definition of MidCap based on clear inclusion criteria - similar to that of Russell Japan Mid Cap Index 80-97.5%
            7. Updated definitions of Growth, Market sensitivity, MTM, Volatility, Exchange Rate sensitivity
            8. GICS2016 definitions
    """
    estu_parameters = {
                       'minNonZero':0.75,
                       'minNonMissing':0.95,
                       'maskZeroWithNoADV_Flag': True,
                       'returnLegacy_Flag': False,
                       'CapByNumber_Flag':False,
                       'CapByNumber_hiCapQuota':np.nan,
                       'CapByNumber_lowCapQuota':np.nan,
                       'market_lower_pctile':1 ,
                       'country_lower_pctile':5,
                       'industry_lower_pctile':5,
                       'dummyThreshold': 10,
                       'inflation_cutoff':0.01
                        }
    k = 5.0
    minNonZero = 0.75
    rm_id = -5
    revision = 5
    rms_id = -1003
    inflation_cutoff = 0.01
    regionalDescriptorStructure = False
    twoRegressionStructure = True
    multiCountry = False

    # List of 13 style factors in the model
    styleList = [
                'Earnings Yield',
                'Value',
                'Leverage',
                'Growth',
                'Profitability',
                'Dividend Yield',
                'Size',
                'Liquidity',
                'Market Sensitivity',
                'Volatility',
                'Medium-Term Momentum',
                'MidCap',
                'Exchange Rate Sensitivity',
                ]

    DescriptorMap = {
            'Earnings Yield': ['Earnings_to_Price_Annual', 'Est_Earnings_to_Price_12MFL_Annual'],
            'Value': ['Book_to_Price_Annual'],
            'Leverage': ['Debt_to_Assets_Annual', 'Debt_to_Equity_Annual'],
            'Growth': ['Earnings_Growth_RPF_Annual', 'Sales_Growth_RPF_Annual'],
            'Dividend Yield': ['Dividend_Yield_Annual'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_60D'],
            'Market Sensitivity': ['Market_Sensitivity_250D'],
            'Volatility': ['Volatility_125D'],
            'Medium-Term Momentum': ['Momentum_250x20D'],
            'Exchange Rate Sensitivity': ['XRate_104W_USD'],
            'Profitability': [ 'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
                              'Sales_to_Assets_Annual', 'Gross_Margin_Annual'] }

    DescriptorWeights = {'Earnings Yield': [0.25, 0.75]}
    smallCapMap = {'MidCap': [80.0, 97.5],}
    noProxyList = ['Dividend Yield']
    fillMissingList = ['Earnings Yield', 'Value', 'Leverage', 'Growth', 'Profitability']
    fillWithZeroList = ['Dividend Yield']
    shrinkList = {'Liquidity': 60,
                  'Market Sensitivity': 250,
                  'Volatility': 125,
                  'Medium-Term Momentum': 250}
    orthogList = {'Volatility': [['Market Sensitivity'], True, 1.0]}

    # Setting up market intercept if relevant
    interceptFactor = 'Market Intercept'
    intercept = ModelFactor(interceptFactor, interceptFactor)
    industryClassification = Classification.GICSCustomJP(datetime.date(2016,9,1))

    estuAssetTypes = ['REIT', 'Com']

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.JPResearchModelFinalESTU')
        # Set up relevant styles to be created/used
        ModelParameters2017.defaultExposureParameters(self, self.styleList)
        self.styles = [s for s in self.totalStyles if s.name in self.styleList]
        EquityModel.FundamentalModel.__init__(self, ['SEDOL'], modelDB, marketDB)

        # Set up regression parameters
        dummyThreshold = 10

        # Run risk based on the square root of market cap weights  
        self.internalCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType='Industry Groups',
                dummyThreshold=dummyThreshold,
                marketRegression=False,
                kappa=5.0,
                useRealMCaps=True,
                regWeight = 'rootCap',
                )

        # Set up external regression parameters
        self.returnCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType='Industry Groups',
                dummyThreshold=dummyThreshold,
                marketRegression=False,
                kappa=None,
                useRealMCaps=True,
                regWeight='invSpecificVariance'
                )

        # This controls the FMP regression
        self.fmpCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType='Industry Groups',
                marketRegression=False,
                dummyThreshold=dummyThreshold,
                kappa=None)

        # Set up risk parameters
        ModelParameters2017.defaultFundamentalCovarianceParameters(
                self, nwLag=2,
                varDVAOnly=False, unboundedDVA=False,
                )

        # Set up standardization parameters
        gloScope = Standardization.GlobalRelativeScope([f.name for f in self.styles])
        self.exposureStandardization = Standardization.BucketizedStandardization([gloScope],
                fillWithZeroList=self.fillWithZeroList)

        # Set up descriptor standardization parameters
        descriptors = sorted(list(set([item for sublist
            in list(self.DescriptorMap.values()) for item in sublist])))
        exceptionNames = [self.DescriptorMap[sf] for sf in self.noProxyList]
        exceptionNames = list(itertools.chain.from_iterable(exceptionNames))
        self.descriptorStandardization = Standardization.BucketizedStandardization(
                    [Standardization.GlobalRelativeScope(descriptors)],
                    mad_bound=15.0, fancyMAD=self.fancyMAD, exceptionNames=exceptionNames)

        self.indexSelector = MarketIndex.MarketIndexSelector(modelDB, marketDB)
        # Set up TOPIX replication - FIXME
        self.topixReplicator = TOPIX.TOPIXReplicator(self, modelDB)
        modelDB.createCurrencyCache(marketDB)

    def generate_model_specific_exposures(self, modelDate, data, modelDB, marketDB):
        """Generate the non-default factors.
        """
        beta = numpy.zeros((len(data.universe)), float)
        # Cap-based style factors here
        if not hasattr(self, 'estuMap') or self.estuMap is None:
            return data.exposureMatrix

        # Small-cap factors
        dateList = modelDB.getDates(self.rmg, modelDate, 61, excludeWeekend=True)
        styleNames = [s.name for s in self.styles]
        scList = [sc for sc in self.estuMap.keys() if sc in styleNames]
        for sc in scList:
            beta = Matrices.allMasked((len(data.universe)), float)
            scAssets = [sid for sid in self.estuMap[sc].assets if sid in data.universe]
            if len(scAssets) < 1:
                logging.warning('No assets in %s universe', sc)
            else:
                qualifiedAssets = modelDB.loadESTUQualifyHistory(
                    self.rms_id, scAssets, dateList, estuInstance=self.estuMap[sc])
                qualifiedAssets = ma.filled(ma.sum(qualifiedAssets.data, axis=1), 0.0)
                if len(qualifiedAssets) < 1:
                    logging.warning('No assets qualified for %s factor', sc)
                else:
                    qualifiedAssets = qualifiedAssets / float(numpy.max(qualifiedAssets, axis=None))
                    for (i_c, idx) in enumerate([data.assetIdxMap[sid] for sid in scAssets]):
                        beta[idx] = qualifiedAssets[i_c]
            data.exposureMatrix.addFactor(sc, beta, ExposureMatrix.StyleFactor)
            if self.exposureStandardization.exceptionNames is None:
                self.exposureStandardization.exceptionNames = [sc]
            else:
                self.exposureStandardization.exceptionNames.append(sc)
        self.exposureStandardization.exceptionNames = list(set(self.exposureStandardization.exceptionNames))

        return data.exposureMatrix

    def generate_estimation_universe_old(self, modelDate, data, modelDB, marketDB, excludeFactors=None):
        """Estimation universe selection criteria for JP.
        """
        import pandas as pd
        #import ipdb;ipdb.set_trace()
        self.log.info('generate_estimation_universe for JP Model: begin')
        buildEstu = EstimationUniverse.ConstructEstimationUniverse(data.universe, self, modelDB, marketDB)

        # Set up various eligible and total universes
        universeIdx = list(range(len(buildEstu.assets)))
        originalEligibleUniverse = list(data.eligibleUniverse)
        originalEligibleUniverseIdx = [data.assetIdxMap[sid] for sid in originalEligibleUniverse]

        # Remove nursery market assets
        if len(data.nurseryUniverse) > 0:
            logging.info('Checking for assets from nursery markets')
            ns_indices = [data.assetIdxMap[sid] for sid in data.nurseryUniverse]
            (eligibleUniverseIdx, nonest) = buildEstu.exclude_specific_assets(
                    ns_indices, baseEstu=originalEligibleUniverseIdx)
            if n != len(eligibleUniverseIdx):
                n = len(eligibleUniverseIdx)
                logging.info('ESTU currently stands at %d stocks', n)
        else:
            eligibleUniverseIdx = originalEligibleUniverseIdx

        universe = pd.DataFrame(buildEstu.assets,columns=['SubID_Obj'])
        universeIdx = universe.index
        original_eligibleUniverse = pd.DataFrame(list(zip(data.eligibleUniverse,originalEligibleUniverseIdx)),columns=['SubID_Obj','originalEligibleUniverseIdx'])

        logging.info('ESTU currently stands at %d stocks based on original eligible universe', len(originalEligibleUniverse))

        # Report on thinly-traded assets over the entire universe
        logging.info('Looking for thinly-traded stocks')
        # (nonSparseIdx, sparse) = buildEstu.exclude_thinly_traded_assets(modelDate, data, baseEstu=universeIdx, minNonZero=0.75)
        # data.nonSparse = numpy.take(buildEstu.assets, nonSparseIdx, axis=0)
        (non_thin_idx, thin_idx) = buildEstu.exclude_thinly_traded_assets(modelDate, data, baseEstu=universe.index.values, minNonZero=0.75)
        # (non_thin_idx, thin_idx) = buildEstu.exclude_thinly_traded_assets(modelDate, data, baseEstu=universe.index.values, minNonZero=0.95)
        data.nonSparse = pd.DataFrame(universe,index=non_thin_idx)["SubID_Obj"].tolist()

        # Exclude thinly traded assets
        estu_withoutThin_idx = list(set(eligibleUniverseIdx).intersection(set(non_thin_idx)))
        logging.info('ESTU currently stands at %d stocks after Filtering thinly-traded assets', len(estu_withoutThin_idx))
        # (2) Filtering tiny-cap assets by market, country and industry
        # (a) Weed out tiny-cap assets by market
        lowerBound = 1
        logging.info('Filtering by top %d%% mcap on entire market', 100-lowerBound)
        (large_byMkt_idx, nonest1) = buildEstu.exclude_by_cap_ranking(data, modelDate, baseEstu=estu_withoutThin_idx,
                                                                     lower_pctile=lowerBound, method='percentage')
        # (2b) Weed out tiny-cap assets by country
        lowerBound = 5
        logging.info('Filtering by top %d%% mcap on country', 100-lowerBound)
        (large_byCntry_idx, nonest2) = buildEstu.exclude_by_cap_ranking(data, modelDate, baseEstu=estu_withoutThin_idx,
                                                                       byFactorType=ExposureMatrix.CountryFactor,
                                                                       lower_pctile=lowerBound, method='percentage',
                                                                       excludeFactors=excludeFactors)
        # (2c) Perform similar check by industry
        logging.info('Filtering by top %d%% mcap on industry', 100-lowerBound)
        (large_byIndtry_idx, nonest3) = buildEstu.exclude_by_cap_ranking(
               data, modelDate, baseEstu=estu_withoutThin_idx,
               byFactorType=ExposureMatrix.IndustryFactor,
               lower_pctile=lowerBound, method='percentage',
               excludeFactors=excludeFactors)
    
        estu_mktCap_idx = set(large_byMkt_idx).union(large_byCntry_idx).union(large_byIndtry_idx)
        estu_mktCap_idx = list(estu_mktCap_idx)
        tinyCap_idx = list(set(estu_withoutThin_idx).difference(estu_mktCap_idx))

        logging.info('ESTU currently stands at %d stocks after Filtering by Market Cap.', len(estu_mktCap_idx))

        # candid_univ_idx = eligibleUniverseIdx
        candid_univ_idx = estu_withoutThin_idx
        # Inflate any thin countries or industries - add 2*
        # minFactorWidth=2*self.returnCalculator.allParameters[0].getThinFactorInformation().dummyThreshold
        minFactorWidth=self.returnCalculator.allParameters[0].getThinFactorInformation().dummyThreshold
        logging.info('Inflating any thin factors')

        (estu_inflated_idx, nonest,herf_num_list) = buildEstu.pump_up_factors2(
                data, modelDate, currentEstu=estu_mktCap_idx,
                baseEstu=candid_univ_idx,
                byFactorType=[ExposureMatrix.CountryFactor, ExposureMatrix.IndustryFactor],
                minFactorWidth=minFactorWidth,
                cutOff = self.inflation_cutoff, 
                excludeFactors=excludeFactors)

        logging.info('ESTU currently stands at %d stocks', len(estu_inflated_idx))
        herf_num_list = pd.DataFrame(herf_num_list)
        herf_num_list.to_csv('herf_num_list.csv')
        # Apply grandfathering rules
        logging.info('Incorporating grandfathering')
        (estu_final_Idx, ESTUQualify, nonest) = buildEstu.grandfather(modelDate,
                estu_inflated_idx, baseEstu=candid_univ_idx,
                estuInstance=self.estuMap['main'])

        totalcap = ma.sum(ma.take(data.marketCaps, estu_final_Idx, axis=0), axis=0) / 1e9
        self.log.info('Final estu contains %d assets, %.2f bn (%s)',len(estu_final_Idx), totalcap, self.numeraire.currency_code)

        self.log.info('Final estu contains %d assets in thin-traded assets.',sum(pd.Series(estu_inflated_idx).isin(thin_idx)))
        self.log.info('Final estu contains %d assets in tiny cap assets.',sum(pd.Series(estu_inflated_idx).isin(tinyCap_idx)))
        self.log.debug('generate_estimation_universe: end')

        # If we have a family of estimation universes, populate the main estu accordingly
        self.estuMap['main'].assets = [buildEstu.assets[idx] for idx in estu_final_Idx]
        self.estuMap['main'].qualify = [buildEstu.assets[idx] for idx in ESTUQualify]

        return estu_final_Idx

    def generate_estimation_universe(self, modelDate, data, modelDB, marketDB, excludeFactors=None):
        estu1 = self.generate_estimation_universe_v2(modelDate, data, modelDB, marketDB, excludeFactors=None)
        # estu2 = self.generate_estimation_universe_old(modelDate, data, modelDB, marketDB, excludeFactors=None)
        # print modelDate, estu1, estu2
        return estu1

class JPResearchModelRandomFactor(JPResearchModel1):
    """JP research model random factor: Same as RM1, except addition of random factor
    """
    rm_id = -5
    revision = 6
    rms_id = -1004

    # List of style factors in the model
    styleList = [
                'Earnings Yield',
                'Value',
                'Leverage',
                'Growth',
                'Profitability',
                'Dividend Yield',
                'Size',
                'Liquidity',
                'Market Sensitivity',
                'Volatility',
                'Medium-Term Momentum',
                'MidCap',
                'Exchange Rate Sensitivity',
                'Random Factor'
                ]

    DescriptorMap = {
            'Earnings Yield': ['Earnings_to_Price_Annual', 'Est_Earnings_to_Price_12MFL_Annual'],
            'Value': ['Book_to_Price_Annual'],
            'Leverage': ['Debt_to_Assets_Annual', 'Debt_to_Equity_Annual'],
            'Growth': ['Earnings_Growth_RPF_Annual', 'Sales_Growth_RPF_Annual'],
            'Dividend Yield': ['Dividend_Yield_Annual'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_60D'],
            'Market Sensitivity': ['Market_Sensitivity_250D'],
            'Volatility': ['Volatility_125D'],
            'Medium-Term Momentum': ['Momentum_250x20D'],
            'Exchange Rate Sensitivity': ['XRate_104W_XDR'],
            'Profitability': ['Return_on_Equity_Annual', 'Return_on_Assets_Annual',
                              'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
                              'Sales_to_Assets_Annual', 'Gross_Margin_Annual'],
            'Random Factor': ['Random_Factor']
            }

class JPResearchModelXrateUSD(JPResearchModel1):
    """JP research model exchange rate sensitivity to USD: Same as RM1, except different factor definition of Exchange Rate Sensitivity
    """
    rm_id = -5
    revision = 7
    rms_id = -1005

    DescriptorMap = {
            'Earnings Yield': ['Earnings_to_Price_Annual', 'Est_Earnings_to_Price_12MFL_Annual'],
            'Value': ['Book_to_Price_Annual'],
            'Leverage': ['Debt_to_Assets_Annual', 'Debt_to_Equity_Annual'],
            'Growth': ['Earnings_Growth_RPF_Annual', 'Sales_Growth_RPF_Annual'],
            'Dividend Yield': ['Dividend_Yield_Annual'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_60D'],
            'Market Sensitivity': ['Market_Sensitivity_250D'],
            'Volatility': ['Volatility_125D'],
            'Medium-Term Momentum': ['Momentum_250x20D'],
            'Exchange Rate Sensitivity': ['XRate_104W_USD'],
            'Profitability': ['Return_on_Equity_Annual', 'Return_on_Assets_Annual',
                              'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
                              'Sales_to_Assets_Annual', 'Gross_Margin_Annual'],
            }

class JPResearchModelFinal(EquityModel.FundamentalModel):
    """ 
    Similar to Final JP research model except 1 stage regression (i.e. use sqrt mcap 
    weights) and k = None. This model is used to compare with production model 
    with k = None for apples to apples comparison.

    Changes: k = None, twoStageRegression = False, removed self.internalCalculator 
    (from __init__), self.returnCalculator: regWeight = 'rootCap'
    """

    k = None
    minNonZero = 0.75
    rm_id = -5
    revision = 8
    rms_id = -1006

    estu_parameters = {
                       'minNonZero':0.75,
                       'minNonMissing':0.95,
                       'maskZeroWithNoADV_Flag': True,
                       'returnLegacy_Flag': False,
                       'CapByNumber_Flag':False,
                       'CapByNumber_hiCapQuota':np.nan,
                       'CapByNumber_lowCapQuota':np.nan,
                       'market_lower_pctile':1 ,
                       'country_lower_pctile':5,
                       'industry_lower_pctile':5,
                       'dummyThreshold': 10,
                       'inflation_cutoff':0.01
                        }

    dummyThreshold          = estu_parameters['dummyThreshold']
    minNonZero              = estu_parameters['minNonZero']
    inflation_cutoff        = estu_parameters['inflation_cutoff']
    market_lower_pctile     = estu_parameters['market_lower_pctile']
    country_lower_pctile    = estu_parameters['country_lower_pctile']
    industry_lower_pctile   = estu_parameters['industry_lower_pctile']

    regionalDescriptorStructure = False
    twoRegressionStructure = False
    multiCountry = False

    # List of 13 style factors in the model
    styleList = [
                'Earnings Yield',
                'Value',
                'Leverage',
                'Growth',
                'Profitability',
                'Dividend Yield',
                'Size',
                'Liquidity',
                'Market Sensitivity',
                'Volatility',
                'Medium-Term Momentum',
                'MidCap',
                'Exchange Rate Sensitivity',
                ]

    DescriptorMap = {
            'Earnings Yield': ['Earnings_to_Price_Annual', 'Est_Earnings_to_Price_12MFL_Annual'],
            'Value': ['Book_to_Price_Annual'],
            'Leverage': ['Debt_to_Assets_Annual', 'Debt_to_Equity_Annual'],
            'Growth': ['Earnings_Growth_RPF_Annual', 'Sales_Growth_RPF_Annual'],
            'Dividend Yield': ['Dividend_Yield_Annual'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_60D'],
            'Market Sensitivity': ['Market_Sensitivity_250D'],
            'Volatility': ['Volatility_125D'],
            'Medium-Term Momentum': ['Momentum_250x20D'],
            'Exchange Rate Sensitivity': ['XRate_104W_USD'],
            'Profitability': ['Return_on_Equity_Annual', 'Return_on_Assets_Annual',
                              'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
                              'Sales_to_Assets_Annual', 'Gross_Margin_Annual']
            }

    DescriptorWeights = {'Earnings Yield': [0.25, 0.75]}
    smallCapMap = {'MidCap': [80.0, 97.5],}
    noProxyList = ['Dividend Yield']
    fillMissingList = ['Earnings Yield', 'Value', 'Leverage', 'Growth', 'Profitability']
    fillWithZeroList = ['Dividend Yield']
    shrinkList = {'Liquidity': 60,
                  'Market Sensitivity': 250,
                  'Volatility': 125,
                  'Medium-Term Momentum': 250}
    orthogList = {'Volatility': [['Market Sensitivity'], True, 1.0]}

    # Setting up market intercept if relevant
    interceptFactor = 'Market Intercept'
    intercept = ModelFactor(interceptFactor, interceptFactor)
    industryClassification = Classification.GICSCustomJP(datetime.date(2016,9,1))

    estuAssetTypes = ['REIT', 'Com']

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.JPAxioma2017MH')
        # Set up relevant styles to be created/used
        ModelParameters2017.defaultExposureParameters(self, self.styleList)
        self.styles = [s for s in self.totalStyles if s.name in self.styleList]
        EquityModel.FundamentalModel.__init__(self, ['SEDOL'], modelDB, marketDB)

        self.estu_parameters.update({'includedAssetTypes': self.commonStockTypes + ['REIT']})

        # Set up external regression parameters
        self.returnCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType='Industry Groups',
                dummyThreshold=self.dummyThreshold,
                marketRegression=False,
                kappa=self.k,
                useRealMCaps=True,
                regWeight='rootCap'
                )

        # This controls the FMP regression
        self.fmpCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType='Industry Groups',
                marketRegression=False,
                dummyThreshold=self.dummyThreshold,
                kappa=None)

        # Set up risk parameters
        ModelParameters2017.defaultFundamentalCovarianceParameters(
                self, nwLag=2,
                varDVAOnly=False, unboundedDVA=False,
                )

        # Set up standardization parameters
        gloScope = Standardization.GlobalRelativeScope([f.name for f in self.styles])
        self.exposureStandardization = Standardization.BucketizedStandardization([gloScope],
                fillWithZeroList=self.fillWithZeroList)

        # Set up descriptor standardization parameters
        descriptors = sorted(list(set([item for sublist
            in list(self.DescriptorMap.values()) for item in sublist])))
        exceptionNames = [self.DescriptorMap[sf] for sf in self.noProxyList]
        exceptionNames = list(itertools.chain.from_iterable(exceptionNames))
        self.descriptorStandardization = Standardization.BucketizedStandardization(
                    [Standardization.GlobalRelativeScope(descriptors)],
                    mad_bound=15.0, fancyMAD=self.fancyMAD, exceptionNames=exceptionNames)

        self.indexSelector = MarketIndex.MarketIndexSelector(modelDB, marketDB)
        # Set up TOPIX replication - FIXME
        self.topixReplicator = TOPIX.TOPIXReplicator(self, modelDB)
        modelDB.createCurrencyCache(marketDB)

    def generate_model_specific_exposures(self, modelDate, data, modelDB, marketDB):
        """Generate the non-default factors.
        """
        beta = numpy.zeros((len(data.universe)), float)
        # Cap-based style factors here
        if not hasattr(self, 'estuMap') or self.estuMap is None:
            return data.exposureMatrix

        # Small-cap factors
        dateList = modelDB.getDates(self.rmg, modelDate, 61, excludeWeekend=True)
        styleNames = [s.name for s in self.styles]
        scList = [sc for sc in self.estuMap.keys() if sc in styleNames]
        for sc in scList:
            beta = Matrices.allMasked((len(data.universe)), float)
            scAssets = [sid for sid in self.estuMap[sc].assets if sid in data.universe]
            if len(scAssets) < 1:
                logging.warning('No assets in %s universe', sc)
            else:
                qualifiedAssets = modelDB.loadESTUQualifyHistory(
                    self.rms_id, scAssets, dateList, estuInstance=self.estuMap[sc])
                qualifiedAssets = ma.filled(ma.sum(qualifiedAssets.data, axis=1), 0.0)
                if len(qualifiedAssets) < 1:
                    logging.warning('No assets qualified for %s factor', sc)
                else:
                    qualifiedAssets = qualifiedAssets / float(numpy.max(qualifiedAssets, axis=None))
                    for (i_c, idx) in enumerate([data.assetIdxMap[sid] for sid in scAssets]):
                        beta[idx] = qualifiedAssets[i_c]
            data.exposureMatrix.addFactor(sc, beta, ExposureMatrix.StyleFactor)
            if self.exposureStandardization.exceptionNames is None:
                self.exposureStandardization.exceptionNames = [sc]
            else:
                self.exposureStandardization.exceptionNames.append(sc)
        self.exposureStandardization.exceptionNames = list(set(self.exposureStandardization.exceptionNames))

        return data.exposureMatrix

    def generate_estimation_universe_old(self, modelDate, data, modelDB, marketDB, excludeFactors=None):
        """Estimation universe selection criteria for JP.
        """
        import pandas as pd
        self.log.info('generate_estimation_universe for JP Model: begin')
        buildEstu = EstimationUniverse.ConstructEstimationUniverse(data.universe, self, modelDB, marketDB)

        # Set up various eligible and total universes
        universeIdx = list(range(len(buildEstu.assets)))
        originalEligibleUniverse = list(data.eligibleUniverse)
        originalEligibleUniverseIdx = [data.assetIdxMap[sid] for sid in originalEligibleUniverse]

        # Remove nursery market assets
        if len(data.nurseryUniverse) > 0:
            logging.info('Checking for assets from nursery markets')
            ns_indices = [data.assetIdxMap[sid] for sid in data.nurseryUniverse]
            (eligibleUniverseIdx, nonest) = buildEstu.exclude_specific_assets(
                    ns_indices, baseEstu=originalEligibleUniverseIdx)
            if n != len(eligibleUniverseIdx):
                n = len(eligibleUniverseIdx)
                logging.info('ESTU currently stands at %d stocks', n)
        else:
            eligibleUniverseIdx = originalEligibleUniverseIdx

        universe = pd.DataFrame(buildEstu.assets,columns=['SubID_Obj'])
        universeIdx = universe.index
        original_eligibleUniverse = pd.DataFrame(list(zip(data.eligibleUniverse,originalEligibleUniverseIdx)),columns=['SubID_Obj','originalEligibleUniverseIdx'])

        logging.info('ESTU currently stands at %d stocks based on original eligible universe', len(originalEligibleUniverse))

        # (1) Filtering on thinly-traded assets over the entire universe
        logging.info('Looking for thinly-traded stocks')
        # (nonSparseIdx, sparse) = buildEstu.exclude_thinly_traded_assets(modelDate, data, baseEstu=universeIdx, minNonZero=0.75)
        # data.nonSparse = numpy.take(buildEstu.assets, nonSparseIdx, axis=0)
        (non_thin_idx, thin_idx) = buildEstu.exclude_thinly_traded_assets(modelDate, data, baseEstu=universe.index.values, minNonZero=self.minNonZero)
        data.nonSparse = pd.DataFrame(universe,index=non_thin_idx)["SubID_Obj"].tolist()

        # Exclude thinly traded assets
        estu_withoutThin_idx = list(set(eligibleUniverseIdx).intersection(set(non_thin_idx)))
        logging.info('ESTU currently stands at %d stocks after Filtering thinly-traded assets', len(estu_withoutThin_idx))
        # (2) Filtering tiny-cap assets by market, country and industry
        # (a) Weed out tiny-cap assets by market
        lowerBound = self.market_lower_pctile
        logging.info('Filtering by top %d%% mcap on entire market', 100-lowerBound)
        (large_byMkt_idx, nonest1) = buildEstu.exclude_by_cap_ranking(data, modelDate, baseEstu=estu_withoutThin_idx,
                                                                     lower_pctile=lowerBound, method='percentage')
        # (2b) Weed out tiny-cap assets by country
        lowerBound = self.country_lower_pctile
        logging.info('Filtering by top %d%% mcap on country', 100-lowerBound)
        (large_byCntry_idx, nonest2) = buildEstu.exclude_by_cap_ranking(data, modelDate, baseEstu=estu_withoutThin_idx,
                                                                       byFactorType=ExposureMatrix.CountryFactor,
                                                                       lower_pctile=lowerBound, method='percentage',
                                                                       excludeFactors=excludeFactors)
        # (2c) Perform similar check by industry
        lowerBound = self.industry_lower_pctile
        logging.info('Filtering by top %d%% mcap on industry', 100-lowerBound)
        (large_byIndtry_idx, nonest3) = buildEstu.exclude_by_cap_ranking(
               data, modelDate, baseEstu=estu_withoutThin_idx,
               byFactorType=ExposureMatrix.IndustryFactor,
               lower_pctile=lowerBound, method='percentage',
               excludeFactors=excludeFactors)

        estu_mktCap_idx = set(large_byMkt_idx).union(large_byCntry_idx).union(large_byIndtry_idx)
        estu_mktCap_idx = list(estu_mktCap_idx)
        tinyCap_idx = list(set(estu_withoutThin_idx).difference(estu_mktCap_idx))

        logging.info('ESTU currently stands at %d stocks after Filtering by Market Cap.', len(estu_mktCap_idx))

        # candid_univ_idx = eligibleUniverseIdx
        candid_univ_idx = estu_withoutThin_idx
        # Inflate any thin countries or industries - add 2*
        # minFactorWidth=2*self.returnCalculator.allParameters[0].getThinFactorInformation().dummyThreshold
        minFactorWidth=self.returnCalculator.allParameters[0].getThinFactorInformation().dummyThreshold
        logging.info('Inflating any thin factors')

        (estu_inflated_idx, nonest,herf_num_list) = buildEstu.pump_up_factors2(
                data, modelDate, currentEstu=estu_mktCap_idx,
                baseEstu=candid_univ_idx,
                byFactorType=[ExposureMatrix.CountryFactor, ExposureMatrix.IndustryFactor],
                minFactorWidth=minFactorWidth,
                cutOff = self.inflation_cutoff,
                excludeFactors=excludeFactors)

        logging.info('ESTU currently stands at %d stocks', len(estu_inflated_idx))
        herf_num_list = pd.DataFrame(herf_num_list)
        herf_num_list.to_csv('herf_num_list.csv')
        # Apply grandfathering rules
        logging.info('Incorporating grandfathering')
        (estu_final_Idx, ESTUQualify, nonest) = buildEstu.grandfather(modelDate,
                estu_inflated_idx, baseEstu=candid_univ_idx,
                estuInstance=self.estuMap['main'])

        totalcap = ma.sum(ma.take(data.marketCaps, estu_final_Idx, axis=0), axis=0) / 1e9
        self.log.info('Final estu contains %d assets, %.2f bn (%s)',len(estu_final_Idx), totalcap, self.numeraire.currency_code)

        self.log.info('Final estu contains %d assets in thin-traded assets.',sum(pd.Series(estu_inflated_idx).isin(thin_idx)))
        self.log.info('Final estu contains %d assets in tiny cap assets.',sum(pd.Series(estu_inflated_idx).isin(tinyCap_idx)))
        self.log.debug('generate_estimation_universe: end')

        # If we have a family of estimation universes, populate the main estu accordingly
        self.estuMap['main'].assets = [buildEstu.assets[idx] for idx in estu_final_Idx]
        self.estuMap['main'].qualify = [buildEstu.assets[idx] for idx in ESTUQualify]

        return estu_final_Idx

    def generate_estimation_universe(self, modelDate, data, modelDB, marketDB, excludeFactors=None):
        estu1 = self.generate_estimation_universe_v2(modelDate, data, modelDB, marketDB, excludeFactors=None)
        # estu2 = self.generate_estimation_universe_old(modelDate, data, modelDB, marketDB, excludeFactors=None)
        # print modelDate, estu1, estu2
        return estu1

class JPResearchModelFinalNoRobust(JPResearchModelFinal):
    """ Same as Final Research Model except there is no robust parameter:
    """
    rm_id = -5
    revision = 9
    rms_id = -1007

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.JPResearchModel1')
        # Set up relevant styles to be created/used
        ModelParameters.defaultExposureParametersV3(self, self.styleList)
        self.styles = [s for s in self.totalStyles if s.name in self.styleList]
        MFM.FundamentalModel.__init__(self, ['SEDOL'], modelDB, marketDB)

        # Set up regression parameters
        dummyThreshold = 10
        self.returnCalculator = ModelParameters.defaultRegressionParameters(
                self, modelDB,
                dummyType='Industry Groups',
                dummyThreshold=dummyThreshold,
                marketRegression=False,
                kappa=None,
                )

        # This controls the FMP regression
        self.fmpCalculator = ModelParameters.defaultRegressionParameters(
                self, modelDB,
                dummyType='Industry Groups',
                marketRegression=False,
                dummyThreshold=dummyThreshold,
                kappa=None)

        # Set up risk parameters
        ModelParameters.defaultFundamentalCovarianceParametersV3(
                self, nwLag=2,
                varDVAOnly=False, unboundedDVA=False,
                )

        # Set up standardization parameters
        gloScope = Standardization.GlobalRelativeScope([f.name for f in self.styles])
        self.exposureStandardization = Standardization.BucketizedStandardization([gloScope],
                fillWithZeroList=self.fillWithZeroList)

        # Set up descriptor standardization parameters
        descriptors = sorted(list(set([item for sublist
            in list(self.DescriptorMap.values()) for item in sublist])))
        exceptionNames = [self.DescriptorMap[sf] for sf in self.noProxyList]
        exceptionNames = list(itertools.chain.from_iterable(exceptionNames))
        if not self.SCM:
            self.descriptorStandardization = Standardization.BucketizedStandardization(
                    [Standardization.RegionRelativeScope(modelDB, descriptors)],
                    mad_bound=15.0, fancyMAD=self.fancyMAD, exceptionNames=exceptionNames)
        else:
            self.descriptorStandardization = Standardization.BucketizedStandardization(
                    [Standardization.GlobalRelativeScope(descriptors)],
                    mad_bound=15.0, fancyMAD=self.fancyMAD, exceptionNames=exceptionNames)

        self.indexSelector = MarketIndex.MarketIndexSelector(modelDB, marketDB)
        # Set up TOPIX replication - FIXME
        self.topixReplicator = TOPIX.TOPIXReplicator(self, modelDB)
        modelDB.createCurrencyCache(marketDB)

class JPResearchModelFinalInvSpVar(JPResearchModelFinal):
    """ Same as Final Research Model except factor returns are now overwritten based
    on regression using inverse specific variance of assets rather than square-root of market cap
    """
    rm_id = -5
    revision = 10
    rms_id = -1008

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.JPResearchModelInvSpVar')
        # Set up relevant styles to be created/used
        ModelParameters.defaultExposureParametersV3(self, self.styleList)
        self.styles = [s for s in self.totalStyles if s.name in self.styleList]
        MFM.FundamentalModel.__init__(self, ['SEDOL'], modelDB, marketDB)

        # Set up regression parameters
        dummyThreshold = 10

        ''' When generating the model first time, leave the regWeight rootCap'''
        '''
        self.returnCalculator = ModelParameters.defaultRegressionParameters(
                self, modelDB,
                dummyType='Industry Groups',
                dummyThreshold=dummyThreshold,
                marketRegression=False,
                kappa=5.0,
                regWeight='rootCap'
                )
        '''

        ''' When running the model second time to overwrite the factor returns, use regWeight='invSpecificVariance '''
        self.returnCalculator = ModelParameters.defaultRegressionParameters(
                self, modelDB,
                dummyType='Industry Groups',
                dummyThreshold=dummyThreshold,
                useRealMCaps=True,
                marketRegression=False,
                kappa=25.0,
                regWeight='invSpecificVariance'
                )

        # This controls the FMP regression
        self.fmpCalculator = ModelParameters.defaultRegressionParameters(
                self, modelDB,
                dummyType='Industry Groups',
                marketRegression=False,
                dummyThreshold=dummyThreshold,
                kappa=None)

        # Set up risk parameters
        ModelParameters.defaultFundamentalCovarianceParametersV3(
                self, nwLag=2,
                varDVAOnly=False, unboundedDVA=False,
                )

        # Set up standardization parameters
        gloScope = Standardization.GlobalRelativeScope([f.name for f in self.styles])
        self.exposureStandardization = Standardization.BucketizedStandardization([gloScope],
                fillWithZeroList=self.fillWithZeroList)

        # Set up descriptor standardization parameters
        descriptors = sorted(list(set([item for sublist
            in list(self.DescriptorMap.values()) for item in sublist])))
        exceptionNames = [self.DescriptorMap[sf] for sf in self.noProxyList]
        exceptionNames = list(itertools.chain.from_iterable(exceptionNames))
        if not self.SCM:
            self.descriptorStandardization = Standardization.BucketizedStandardization(
                    [Standardization.RegionRelativeScope(modelDB, descriptors)],
                    mad_bound=15.0, fancyMAD=self.fancyMAD, exceptionNames=exceptionNames)
        else:
            self.descriptorStandardization = Standardization.BucketizedStandardization(
                    [Standardization.GlobalRelativeScope(descriptors)],
                    mad_bound=15.0, fancyMAD=self.fancyMAD, exceptionNames=exceptionNames)

        self.indexSelector = MarketIndex.MarketIndexSelector(modelDB, marketDB)
        # Set up TOPIX replication - FIXME
        self.topixReplicator = TOPIX.TOPIXReplicator(self, modelDB)
        modelDB.createCurrencyCache(marketDB)

    def generate_model_specific_exposures(self, modelDate, data, modelDB, marketDB):
        data.exposureMatrix = self.generate_md_fundamental_exposures(
                modelDate, data, modelDB, marketDB)
        data.exposureMatrix = GlobalExposures.generate_china_domestic_exposures(
                data, modelDate, modelDB, marketDB)
        return data.exposureMatrix

class JPResearchModelFinalInvSpVar2(JPResearchModelFinal):
    """ Same as Final Research Model except factor returns, risks and betas are now overwritten based
    on regression using inverse specific variance of assets rather than square-root of market cap
    """
    rm_id = -5
    revision = 11
    rms_id = -1009

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.JPResearchModelInvSpVar')
        # Set up relevant styles to be created/used
        ModelParameters.defaultExposureParametersV3(self, self.styleList)
        self.styles = [s for s in self.totalStyles if s.name in self.styleList]
        MFM.FundamentalModel.__init__(self, ['SEDOL'], modelDB, marketDB)

        # Set up regression parameters
        dummyThreshold = 10

        ''' When generating the model first time, leave the regWeight rootCap'''
        '''
        self.returnCalculator = ModelParameters.defaultRegressionParameters(
                self, modelDB,
                dummyType='Industry Groups',
                dummyThreshold=dummyThreshold,
                marketRegression=False,
                kappa=5.0,
                regWeight='rootCap'
                )
        '''

        ''' When running the model second time to overwrite the factor returns, use regWeight='invSpecificVariance '''
        self.returnCalculator = ModelParameters.defaultRegressionParameters(
                self, modelDB,
                dummyType='Industry Groups',
                dummyThreshold=dummyThreshold,
                useRealMCaps=True,
                marketRegression=False,
                kappa=25.0,
                regWeight='invSpecificVariance'
                )

        # This controls the FMP regression
        self.fmpCalculator = ModelParameters.defaultRegressionParameters(
                self, modelDB,
                dummyType='Industry Groups',
                marketRegression=False,
                dummyThreshold=dummyThreshold,
                kappa=None)

        # Set up risk parameters
        ModelParameters.defaultFundamentalCovarianceParametersV3(
                self, nwLag=2,
                varDVAOnly=False, unboundedDVA=False,
                )

        # Set up standardization parameters
        gloScope = Standardization.GlobalRelativeScope([f.name for f in self.styles])
        self.exposureStandardization = Standardization.BucketizedStandardization([gloScope],
                fillWithZeroList=self.fillWithZeroList)

        # Set up descriptor standardization parameters
        descriptors = sorted(list(set([item for sublist
            in list(self.DescriptorMap.values()) for item in sublist])))
        exceptionNames = [self.DescriptorMap[sf] for sf in self.noProxyList]
        exceptionNames = list(itertools.chain.from_iterable(exceptionNames))
        if not self.SCM:
            self.descriptorStandardization = Standardization.BucketizedStandardization(
                    [Standardization.RegionRelativeScope(modelDB, descriptors)],
                    mad_bound=15.0, fancyMAD=self.fancyMAD, exceptionNames=exceptionNames)
        else:
            self.descriptorStandardization = Standardization.BucketizedStandardization(
                    [Standardization.GlobalRelativeScope(descriptors)],
                    mad_bound=15.0, fancyMAD=self.fancyMAD, exceptionNames=exceptionNames)

        self.indexSelector = MarketIndex.MarketIndexSelector(modelDB, marketDB)
        # Set up TOPIX replication - FIXME
        self.topixReplicator = TOPIX.TOPIXReplicator(self, modelDB)
        modelDB.createCurrencyCache(marketDB)

class JPResearchModelFinalESTU(EquityModel.FundamentalModel):
    """ Final JP research model incorporating the follwong changes EY (25/75):
            1. Split EY and Value
            2. Combine D/A and D/E in Leverage
            3. Include profitability factor
            4. Include Dividend Yield factor
            5. Enhanced ESTU coverage (based on inclusion of several exchanges)
            6. Updated definition of MidCap based on clear inclusion criteria - similar to that of Russell Japan Mid Cap Index 80-97.5%
            7. Updated definitions of Growth, Market sensitivity, MTM, Volatility, Exchange Rate sensitivity
            8. GICS2016 definitions
    """
    estu_parameters = {
                       'minNonZero':0.75,
                       'minNonMissing':0.95,
                       'maskZeroWithNoADV_Flag': True,
                       'returnLegacy_Flag': False,
                       'CapByNumber_Flag':False,
                       'CapByNumber_hiCapQuota':np.nan,
                       'CapByNumber_lowCapQuota':np.nan,
                       'market_lower_pctile':1 ,
                       'country_lower_pctile':5,
                       'industry_lower_pctile':5,
                       'dummyThreshold': 10,
                       'inflation_cutoff':0.01
                        }
    k = 5.0
    minNonZero = 0.75
    rm_id = -5
    revision = 12
    rms_id = -1010
    inflation_cutoff = 0.01
    regionalDescriptorStructure = False
    twoRegressionStructure = True
    multiCountry = False

    # List of 13 style factors in the model
    styleList = [
                'Earnings Yield',
                'Value',
                'Leverage',
                'Growth',
                'Profitability',
                'Dividend Yield',
                'Size',
                'Liquidity',
                'Market Sensitivity',
                'Volatility',
                'Medium-Term Momentum',
                'MidCap',
                'Exchange Rate Sensitivity',
                ]

    DescriptorMap = {
            'Earnings Yield': ['Earnings_to_Price_Annual', 'Est_Earnings_to_Price_12MFL_Annual'],
            'Value': ['Book_to_Price_Annual'],
            'Leverage': ['Debt_to_Assets_Annual', 'Debt_to_Equity_Annual'],
            'Growth': ['Earnings_Growth_RPF_Annual', 'Sales_Growth_RPF_Annual'],
            'Dividend Yield': ['Dividend_Yield_Annual'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_60D'],
            'Market Sensitivity': ['Market_Sensitivity_250D'],
            'Volatility': ['Volatility_125D'],
            'Medium-Term Momentum': ['Momentum_250x20D'],
            'Exchange Rate Sensitivity': ['XRate_104W_USD'],
            'Profitability': ['Return_on_Equity_Annual', 'Return_on_Assets_Annual',
                              'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
                              'Sales_to_Assets_Annual', 'Gross_Margin_Annual']
            }

    DescriptorWeights = {'Earnings Yield': [0.25, 0.75]}
    smallCapMap = {'MidCap': [80.0, 97.5],}
    noProxyList = ['Dividend Yield']
    fillMissingList = ['Earnings Yield', 'Value', 'Leverage', 'Growth', 'Profitability']
    fillWithZeroList = ['Dividend Yield']
    shrinkList = {'Liquidity': 60,
                  'Market Sensitivity': 250,
                  'Volatility': 125,
                  'Medium-Term Momentum': 250}
    orthogList = {'Volatility': [['Market Sensitivity'], True, 1.0]}

    # Setting up market intercept if relevant
    interceptFactor = 'Market Intercept'
    intercept = ModelFactor(interceptFactor, interceptFactor)
    industryClassification = Classification.GICSCustomJP(datetime.date(2016,9,1))

    estuAssetTypes = ['REIT', 'Com']

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.JPResearchModelFinalESTU')
        # Set up relevant styles to be created/used
        ModelParameters2017.defaultExposureParameters(self, self.styleList)
        self.styles = [s for s in self.totalStyles if s.name in self.styleList]
        EquityModel.FundamentalModel.__init__(self, ['SEDOL'], modelDB, marketDB)

        self.estu_parameters.update({'includedAssetTypes': self.commonStockTypes + ['REIT']})
        # Set up regression parameters
        dummyThreshold = 10

        # Run risk based on the square root of market cap weights  
        self.internalCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType='Industry Groups',
                dummyThreshold=dummyThreshold,
                marketRegression=False,
                kappa=5.0,
                useRealMCaps=True,
                regWeight = 'rootCap',
                )

        # Set up external regression parameters
        self.returnCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType='Industry Groups',
                dummyThreshold=dummyThreshold,
                marketRegression=False,
                kappa=None,
                useRealMCaps=True,
                regWeight='invSpecificVariance'
                )

        # This controls the FMP regression
        self.fmpCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType='Industry Groups',
                marketRegression=False,
                dummyThreshold=dummyThreshold,
                kappa=None)

        # Set up risk parameters
        ModelParameters2017.defaultFundamentalCovarianceParameters(
                self, nwLag=2,
                varDVAOnly=False, unboundedDVA=False,
                )

        # Set up standardization parameters
        gloScope = Standardization.GlobalRelativeScope([f.name for f in self.styles])
        self.exposureStandardization = Standardization.BucketizedStandardization([gloScope],
                fillWithZeroList=self.fillWithZeroList)

        # Set up descriptor standardization parameters
        descriptors = sorted(list(set([item for sublist
            in list(self.DescriptorMap.values()) for item in sublist])))
        exceptionNames = [self.DescriptorMap[sf] for sf in self.noProxyList]
        exceptionNames = list(itertools.chain.from_iterable(exceptionNames))
        self.descriptorStandardization = Standardization.BucketizedStandardization(
                    [Standardization.GlobalRelativeScope(descriptors)],
                    mad_bound=15.0, fancyMAD=self.fancyMAD, exceptionNames=exceptionNames)

        self.indexSelector = MarketIndex.MarketIndexSelector(modelDB, marketDB)
        # Set up TOPIX replication - FIXME
        self.topixReplicator = TOPIX.TOPIXReplicator(self, modelDB)
        modelDB.createCurrencyCache(marketDB)

    def generate_model_specific_exposures(self, modelDate, data, modelDB, marketDB):
        """Generate the non-default factors.
        """
        beta = numpy.zeros((len(data.universe)), float)
        # Cap-based style factors here
        if not hasattr(self, 'estuMap') or self.estuMap is None:
            return data.exposureMatrix

        # Small-cap factors
        dateList = modelDB.getDates(self.rmg, modelDate, 61, excludeWeekend=True)
        styleNames = [s.name for s in self.styles]
        scList = [sc for sc in self.estuMap.keys() if sc in styleNames]
        for sc in scList:
            beta = Matrices.allMasked((len(data.universe)), float)
            scAssets = [sid for sid in self.estuMap[sc].assets if sid in data.universe]
            if len(scAssets) < 1:
                logging.warning('No assets in %s universe', sc)
            else:
                qualifiedAssets = modelDB.loadESTUQualifyHistory(
                    self.rms_id, scAssets, dateList, estuInstance=self.estuMap[sc])
                qualifiedAssets = ma.filled(ma.sum(qualifiedAssets.data, axis=1), 0.0)
                if len(qualifiedAssets) < 1:
                    logging.warning('No assets qualified for %s factor', sc)
                else:
                    qualifiedAssets = qualifiedAssets / float(numpy.max(qualifiedAssets, axis=None))
                    for (i_c, idx) in enumerate([data.assetIdxMap[sid] for sid in scAssets]):
                        beta[idx] = qualifiedAssets[i_c]
            data.exposureMatrix.addFactor(sc, beta, ExposureMatrix.StyleFactor)
            if self.exposureStandardization.exceptionNames is None:
                self.exposureStandardization.exceptionNames = [sc]
            else:
                self.exposureStandardization.exceptionNames.append(sc)
        self.exposureStandardization.exceptionNames = list(set(self.exposureStandardization.exceptionNames))

        return data.exposureMatrix

    def generate_estimation_universe_old(self, modelDate, data, modelDB, marketDB, excludeFactors=None):
        """Estimation universe selection criteria for JP.
        """
        import pandas as pd
        #import ipdb;ipdb.set_trace()
        self.log.info('generate_estimation_universe for JP Model: begin')
        buildEstu = EstimationUniverse.ConstructEstimationUniverse(data.universe, self, modelDB, marketDB)

        # Set up various eligible and total universes
        universeIdx = list(range(len(buildEstu.assets)))
        originalEligibleUniverse = list(data.eligibleUniverse)
        originalEligibleUniverseIdx = [data.assetIdxMap[sid] for sid in originalEligibleUniverse]

        # Remove nursery market assets
        if len(data.nurseryUniverse) > 0:
            logging.info('Checking for assets from nursery markets')
            ns_indices = [data.assetIdxMap[sid] for sid in data.nurseryUniverse]
            (eligibleUniverseIdx, nonest) = buildEstu.exclude_specific_assets(
                    ns_indices, baseEstu=originalEligibleUniverseIdx)
            if n != len(eligibleUniverseIdx):
                n = len(eligibleUniverseIdx)
                logging.info('ESTU currently stands at %d stocks', n)
        else:
            eligibleUniverseIdx = originalEligibleUniverseIdx

        universe = pd.DataFrame(buildEstu.assets,columns=['SubID_Obj'])
        universeIdx = universe.index
        original_eligibleUniverse = pd.DataFrame(list(zip(data.eligibleUniverse,originalEligibleUniverseIdx)),columns=['SubID_Obj','originalEligibleUniverseIdx'])

        logging.info('ESTU currently stands at %d stocks based on original eligible universe', len(originalEligibleUniverse))

        # Report on thinly-traded assets over the entire universe
        logging.info('Looking for thinly-traded stocks')
        # (nonSparseIdx, sparse) = buildEstu.exclude_thinly_traded_assets(modelDate, data, baseEstu=universeIdx, minNonZero=0.75)
        # data.nonSparse = numpy.take(buildEstu.assets, nonSparseIdx, axis=0)
        (non_thin_idx, thin_idx) = buildEstu.exclude_thinly_traded_assets(modelDate, data, baseEstu=universe.index.values, minNonZero=0.75)
        # (non_thin_idx, thin_idx) = buildEstu.exclude_thinly_traded_assets(modelDate, data, baseEstu=universe.index.values, minNonZero=0.95)
        data.nonSparse = pd.DataFrame(universe,index=non_thin_idx)["SubID_Obj"].tolist()

        # Exclude thinly traded assets
        estu_withoutThin_idx = list(set(eligibleUniverseIdx).intersection(set(non_thin_idx)))
        logging.info('ESTU currently stands at %d stocks after Filtering thinly-traded assets', len(estu_withoutThin_idx))
        # (2) Filtering tiny-cap assets by market, country and industry
        # (a) Weed out tiny-cap assets by market
        lowerBound = 1
        logging.info('Filtering by top %d%% mcap on entire market', 100-lowerBound)
        (large_byMkt_idx, nonest1) = buildEstu.exclude_by_cap_ranking(data, modelDate, baseEstu=estu_withoutThin_idx,
                                                                     lower_pctile=lowerBound, method='percentage')
        # (2b) Weed out tiny-cap assets by country
        lowerBound = 5
        logging.info('Filtering by top %d%% mcap on country', 100-lowerBound)
        (large_byCntry_idx, nonest2) = buildEstu.exclude_by_cap_ranking(data, modelDate, baseEstu=estu_withoutThin_idx,
                                                                       byFactorType=ExposureMatrix.CountryFactor,
                                                                       lower_pctile=lowerBound, method='percentage',
                                                                       excludeFactors=excludeFactors)
        # (2c) Perform similar check by industry
        logging.info('Filtering by top %d%% mcap on industry', 100-lowerBound)
        (large_byIndtry_idx, nonest3) = buildEstu.exclude_by_cap_ranking(
               data, modelDate, baseEstu=estu_withoutThin_idx,
               byFactorType=ExposureMatrix.IndustryFactor,
               lower_pctile=lowerBound, method='percentage',
               excludeFactors=excludeFactors)
    
        estu_mktCap_idx = set(large_byMkt_idx).union(large_byCntry_idx).union(large_byIndtry_idx)
        estu_mktCap_idx = list(estu_mktCap_idx)
        tinyCap_idx = list(set(estu_withoutThin_idx).difference(estu_mktCap_idx))

        logging.info('ESTU currently stands at %d stocks after Filtering by Market Cap.', len(estu_mktCap_idx))

        # candid_univ_idx = eligibleUniverseIdx
        candid_univ_idx = estu_withoutThin_idx
        # Inflate any thin countries or industries - add 2*
        # minFactorWidth=2*self.returnCalculator.allParameters[0].getThinFactorInformation().dummyThreshold
        minFactorWidth=self.returnCalculator.allParameters[0].getThinFactorInformation().dummyThreshold
        logging.info('Inflating any thin factors')

        (estu_inflated_idx, nonest,herf_num_list) = buildEstu.pump_up_factors2(
                data, modelDate, currentEstu=estu_mktCap_idx,
                baseEstu=candid_univ_idx,
                byFactorType=[ExposureMatrix.CountryFactor, ExposureMatrix.IndustryFactor],
                minFactorWidth=minFactorWidth,
                cutOff = self.inflation_cutoff, 
                excludeFactors=excludeFactors)

        logging.info('ESTU currently stands at %d stocks', len(estu_inflated_idx))
        herf_num_list = pd.DataFrame(herf_num_list)
        herf_num_list.to_csv('herf_num_list.csv')
        # Apply grandfathering rules
        logging.info('Incorporating grandfathering')
        (estu_final_Idx, ESTUQualify, nonest) = buildEstu.grandfather(modelDate,
                estu_inflated_idx, baseEstu=candid_univ_idx,
                estuInstance=self.estuMap['main'])

        totalcap = ma.sum(ma.take(data.marketCaps, estu_final_Idx, axis=0), axis=0) / 1e9
        self.log.info('Final estu contains %d assets, %.2f bn (%s)',len(estu_final_Idx), totalcap, self.numeraire.currency_code)

        self.log.info('Final estu contains %d assets in thin-traded assets.',sum(pd.Series(estu_inflated_idx).isin(thin_idx)))
        self.log.info('Final estu contains %d assets in tiny cap assets.',sum(pd.Series(estu_inflated_idx).isin(tinyCap_idx)))
        self.log.debug('generate_estimation_universe: end')

        # If we have a family of estimation universes, populate the main estu accordingly
        self.estuMap['main'].assets = [buildEstu.assets[idx] for idx in estu_final_Idx]
        self.estuMap['main'].qualify = [buildEstu.assets[idx] for idx in ESTUQualify]

        return estu_final_Idx

    def generate_estimation_universe(self, modelDate, data, modelDB, marketDB, excludeFactors=None):
        estu1 = self.generate_estimation_universe_v2(modelDate, data, modelDB, marketDB, excludeFactors=None)
        # estu2 = self.generate_estimation_universe_old(modelDate, data, modelDB, marketDB, excludeFactors=None)
        # print modelDate, estu1, estu2
        return estu1

class JPResearchModelSH(JPResearchModelFinal):
    """JP4 fundamental short-horizon model.  """
    rm_id = -5
    revision = 13
    rms_id = -1011
    regionalDescriptorStructure = False

    # List of style factors in the model
    styleList = ['Earnings Yield',
                 'Value',
                 'Leverage',
                 'Growth',
                 'Profitability',
                 'Dividend Yield',
                 'Size',
                 'Liquidity',
                 'Market Sensitivity',
                 'Volatility',
                 'Medium-Term Momentum',
                 'Short-Term Momentum',
                 'MidCap',
                 'Exchange Rate Sensitivity',
                ]

    DescriptorMap = {
            'Earnings Yield': ['Earnings_to_Price_Annual', 'Est_Earnings_to_Price_12MFL_Annual'],
            'Value': ['Book_to_Price_Annual'],
            'Leverage': ['Debt_to_Assets_Annual', 'Debt_to_Equity_Annual'],
            'Growth': ['Earnings_Growth_RPF_Annual', 'Sales_Growth_RPF_Annual'],
            'Dividend Yield': ['Dividend_Yield_Annual'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_20D'],
            'Market Sensitivity': ['Market_Sensitivity_125D'],
            'Volatility': ['Volatility_125D'],
            'Medium-Term Momentum': ['Momentum_250x20D'],
            'Short-Term Momentum': ['Momentum_20D'],
            'Exchange Rate Sensitivity': ['XRate_52W_USD'],
            'Profitability': ['Return_on_Equity_Annual', 'Return_on_Assets_Annual',
                              'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
                              'Sales_to_Assets_Annual', 'Gross_Margin_Annual'],
            }

    DescriptorWeights = {'Earnings Yield': [0.75, 0.25]}
    smallCapMap = {'MidCap': [85.00, 97.5],}
    noProxyList = ['Dividend Yield']
    fillMissingList = ['Earnings Yield', 'Value', 'Leverage', 'Growth', 'Profitability']
    fillWithZeroList = ['Dividend Yield']
    shrinkList = {'Liquidity': 20,
                  'Market Sensitivity': 125,
                  'Volatility': 60,
                  'Short-Term Momentum': 20,
                  'Medium-Term Momentum': 250}
    orthogList = {'Volatility': [['Market Sensitivity'], True, 1.0]}

    # Setting up market intercept if relevant
    interceptFactor = 'Market Intercept'
    intercept = ModelFactor(interceptFactor, interceptFactor)
    industryClassification = Classification.GICSCustomJP(datetime.date(2008,8,30))
    estuAssetTypes = ['REIT', 'Com']

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.JPResearchModelSH')
        # Set up relevant styles to be created/used
        ModelParameters.defaultExposureParameters(self, self.styleList)
        self.styles = [s for s in self.totalStyles if s.name in self.styleList]
        MFM.FundamentalModel.__init__(self, ['SEDOL'], modelDB, marketDB)

        # Set up regression parameters
        dummyThreshold = 10
        self.returnCalculator = ModelParameters.defaultRegressionParameters(
                self, modelDB,
                dummyType='Industry Groups',
                dummyThreshold=dummyThreshold,
                marketRegression=False,
                kappa=5.0,
                )

        # This controls the FMP regression
        self.fmpCalculator = ModelParameters.defaultRegressionParameters(
                self, modelDB,
                dummyType='Industry Groups',
                marketRegression=False,
                dummyThreshold=dummyThreshold,
                kappa=None)

        # Set up risk parameters
        ModelParameters.defaultFundamentalCovarianceParametersV3( 
        # ModelParameters2017.defaultFundamentalCovarianceParameters(
                self, nwLag=2,
                modelHorizon='short',
                varDVAOnly=False, unboundedDVA=False,
                )

        # Set up standardization parameters
        gloScope = Standardization.GlobalRelativeScope([f.name for f in self.styles])
        self.exposureStandardization = Standardization.BucketizedStandardization([gloScope],
                fillWithZeroList=self.fillWithZeroList)

        # Set up descriptor standardization parameters
        descriptors = sorted(list(set([item for sublist
            in list(self.DescriptorMap.values()) for item in sublist])))
        exceptionNames = [self.DescriptorMap[sf] for sf in self.noProxyList]
        exceptionNames = list(itertools.chain.from_iterable(exceptionNames))
        self.descriptorStandardization = Standardization.BucketizedStandardization( 
                    [Standardization.GlobalRelativeScope(descriptors)],  
                    mad_bound=15.0, fancyMAD=self.fancyMAD, exceptionNames=exceptionNames)  

        self.indexSelector = MarketIndex.MarketIndexSelector(modelDB, marketDB)
        modelDB.createCurrencyCache(marketDB)

class JPResearchModelMH_S(MFM.StatisticalModel):
    """JP4 Medium Horizon Statistical Model
    """
    rm_id = -5
    revision = 14
    rms_id = -1012
    numFactors = 15
    blind = [ModelFactor('Statistical Factor %d' % n, 'Statistical Factor %d' % n)
                         for n in range(1, numFactors+1)]
    industryClassification = Classification.GICSIndustries(datetime.date(2008,8,30))
    allowETFs = True
    pcaHistory = 250

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.JPResearchModelMH_S')
        MFM.StatisticalModel.__init__(self, ['SEDOL'], modelDB, marketDB)  
        # So we can use the same ESTU method as the fundamental model
        self.baseModelDateMap = {datetime.date(1980,1,1): JPResearchModelFinal(modelDB, marketDB)}
        # Set up returns model
        self.returnCalculator = FactorReturns.AsymptoticPrincipalComponents(self.numFactors)
        # Set up risk parameters
        ModelParameters.defaultStatisticalCovarianceParametersV3(self)      
        self.indexSelector = MarketIndex.MarketIndexSelector(modelDB, marketDB)
    
class JPResearchModelSH_S(JPResearchModelMH_S):
    """JP4 Short Horizon Statistical Model
    """
    rm_id = -5
    revision = 15
    rms_id = -1013
    numFactors = 15
    blind = [ModelFactor('Statistical Factor %d' % n, 'Statistical Factor %d' % n)
                         for n in range(1, numFactors+1)]
    industryClassification = Classification.GICSIndustries(datetime.date(2014,3,1))
    allowETFs = True
    pcaHistory = 250

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.JPResearchModelSH_S')
        MFM.StatisticalModel.__init__(self, ['SEDOL'], modelDB, marketDB)
        # So we can use the same ESTU method as the fundamental model
        self.baseModelDateMap = {datetime.date(1980,1,1): JPResearchModelFinal(modelDB, marketDB)}
        # Set up returns model
        self.returnCalculator = FactorReturns.AsymptoticPrincipalComponents(self.numFactors)
        # Set up risk parameters
        ModelParameters.defaultStatisticalCovarianceParametersV3(self, modelHorizon='short')
        self.indexSelector = MarketIndex.MarketIndexSelector(modelDB, marketDB)

class JPResearchModelMidCapInvSpVar(JPResearchModelFinal):
    """ Same as Final Research Model except factor returns are now overwritten based
    on regression using inverse specific variance of assets rather than square-root of market cap
    and midcap factor is 80-95% instead of 85-97.5
    """
    rm_id = -5
    revision = 16
    rms_id = -1014

    smallCapMap = {'MidCap': [80.0, 95],}

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.JPResearchModelMidCapInvSpVar')
        # Set up relevant styles to be created/used
        ModelParameters.defaultExposureParametersV3(self, self.styleList)
        self.styles = [s for s in self.totalStyles if s.name in self.styleList]
        MFM.FundamentalModel.__init__(self, ['SEDOL'], modelDB, marketDB)

        # Set up regression parameters
        dummyThreshold = 10

        ''' When generating the model first time, leave the regWeight rootCap'''
        '''
        self.returnCalculator = ModelParameters.defaultRegressionParameters(
                self, modelDB,
                dummyType='Industry Groups',
                dummyThreshold=dummyThreshold,
                marketRegression=False,
                kappa=5.0,
                regWeight='rootCap'
                )
        '''

        ''' When running the model second time to overwrite the factor returns, use regWeight='invSpecificVariance '''
        self.returnCalculator = ModelParameters.defaultRegressionParameters(
                self, modelDB,
                dummyType='Industry Groups',
                dummyThreshold=dummyThreshold,
                useRealMCaps=True,
                marketRegression=False,
                kappa=25.0,
                regWeight='invSpecificVariance'
                )

        # This controls the FMP regression
        self.fmpCalculator = ModelParameters.defaultRegressionParameters(
                self, modelDB,
                dummyType='Industry Groups',
                marketRegression=False,
                dummyThreshold=dummyThreshold,
                kappa=None)

        # Set up risk parameters
        ModelParameters.defaultFundamentalCovarianceParametersV3(
                self, nwLag=2,
                varDVAOnly=False, unboundedDVA=False,
                )

        # Set up standardization parameters
        gloScope = Standardization.GlobalRelativeScope([f.name for f in self.styles])
        self.exposureStandardization = Standardization.BucketizedStandardization([gloScope],
                fillWithZeroList=self.fillWithZeroList)

        # Set up descriptor standardization parameters
        descriptors = sorted(list(set([item for sublist
            in list(self.DescriptorMap.values()) for item in sublist])))
        exceptionNames = [self.DescriptorMap[sf] for sf in self.noProxyList]
        exceptionNames = list(itertools.chain.from_iterable(exceptionNames))
        if not self.SCM:
            self.descriptorStandardization = Standardization.BucketizedStandardization(
                    [Standardization.RegionRelativeScope(modelDB, descriptors)],
                    mad_bound=15.0, fancyMAD=self.fancyMAD, exceptionNames=exceptionNames)
        else:
            self.descriptorStandardization = Standardization.BucketizedStandardization(
                    [Standardization.GlobalRelativeScope(descriptors)],
                    mad_bound=15.0, fancyMAD=self.fancyMAD, exceptionNames=exceptionNames)

        self.indexSelector = MarketIndex.MarketIndexSelector(modelDB, marketDB)
        # Set up TOPIX replication - FIXME
        self.topixReplicator = TOPIX.TOPIXReplicator(self, modelDB)
        modelDB.createCurrencyCache(marketDB)

class JPResearchModelGics2016Raw(MFM.FundamentalModel):
    """ Final JP research model incorporating the follwong changes:
            1. Split EY and Value
            2. Combine D/A and D/E in Leverage
            3. Include profitability factor
            4. Include Dividend Yield factor
            5. Enhanced ESTU coverage (based on inclusion of several exchanges)
            6. Updated definition of MidCap based on clear inclusion criteria - similar to that of Russell Japan Mid Cap Index
            7. Updated definitions of Growth, Market sensitivity, MTM, Volatility, Exchange Rate sensitivity
    """
    rm_id = -5
    revision = 17
    rms_id = -1015
    standardizationStats = True
    globalDescriptorModel = True        # Uses the global descriptor model
    DLCEnabled = True                   # Combines mcap for DLCs
    SCM = True

    # List of 13 style factors in the model
    styleList = [
                'Earnings Yield',
                'Value',
                'Leverage',
                'Growth',
                'Profitability',
                'Dividend Yield',
                'Size',
                'Liquidity',
                'Market Sensitivity',
                'Volatility',
                'Medium-Term Momentum',
                'MidCap',
                'Exchange Rate Sensitivity',
                ]

    DescriptorMap = {
            'Earnings Yield': ['Earnings_to_Price_Annual', 'Est_Earnings_to_Price_12MFL_Annual'],
            'Value': ['Book_to_Price_Annual'],
            'Leverage': ['Debt_to_Assets_Annual', 'Debt_to_Equity_Annual'],
            'Growth': ['Earnings_Growth_RPF_Annual', 'Sales_Growth_RPF_Annual'],
            'Dividend Yield': ['Dividend_Yield_Annual'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_60D'],
            'Market Sensitivity': ['Market_Sensitivity_250D'],
            'Volatility': ['Volatility_125D'],
            'Medium-Term Momentum': ['Momentum_250x20D'],
            'Exchange Rate Sensitivity': ['XRate_104W_USD'],
            'Profitability': ['Return_on_Equity_Annual', 'Return_on_Assets_Annual',
                              'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
                              'Sales_to_Assets_Annual', 'Gross_Margin_Annual'],
            }

    DescriptorWeights = {'Earnings Yield': [0.75, 0.25]}
    smallCapMap = {'MidCap': [85.0, 97.5],}
    noProxyList = ['Dividend Yield']
    fillMissingList = ['Earnings Yield', 'Value', 'Leverage', 'Growth', 'Profitability']
    fillWithZeroList = ['Dividend Yield']
    shrinkList = {'Liquidity': 60,
                  'Market Sensitivity': 250,
                  'Volatility': 125,
                  'Medium-Term Momentum': 250}
    orthogList = {'Volatility': [['Market Sensitivity'], True, 1.0]}

    # Setting up market intercept if relevant
    interceptFactor = 'Market Intercept'
    intercept = ModelFactor(interceptFactor, interceptFactor)
    industryClassification = Classification.GICSIndustries(datetime.date(2016,9,1))

    estuAssetTypes = ['REIT', 'Com']

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.JPResearchModel1')
        # Set up relevant styles to be created/used
        ModelParameters.defaultExposureParametersV3(self, self.styleList)
        self.styles = [s for s in self.totalStyles if s.name in self.styleList]
        MFM.FundamentalModel.__init__(self, ['SEDOL'], modelDB, marketDB)

        # Set up regression parameters
        dummyThreshold = 10
        self.returnCalculator = ModelParameters.defaultRegressionParameters(
                self, modelDB,
                dummyType='Industry Groups',
                dummyThreshold=dummyThreshold,
                marketRegression=False,
                kappa=5.0,
                )

        # This controls the FMP regression
        self.fmpCalculator = ModelParameters.defaultRegressionParameters(
                self, modelDB,
                dummyType='Industry Groups',
                marketRegression=False,
                dummyThreshold=dummyThreshold,
                kappa=None)

        # Set up risk parameters
        ModelParameters.defaultFundamentalCovarianceParametersV3(
                self, nwLag=2,
                varDVAOnly=False, unboundedDVA=False,
                )

        # Set up standardization parameters
        gloScope = Standardization.GlobalRelativeScope([f.name for f in self.styles])
        self.exposureStandardization = Standardization.BucketizedStandardization([gloScope],
                fillWithZeroList=self.fillWithZeroList)

        # Set up descriptor standardization parameters
        descriptors = sorted(list(set([item for sublist
            in list(self.DescriptorMap.values()) for item in sublist])))
        exceptionNames = [self.DescriptorMap[sf] for sf in self.noProxyList]
        exceptionNames = list(itertools.chain.from_iterable(exceptionNames))
        if not self.SCM:
            self.descriptorStandardization = Standardization.BucketizedStandardization(
                    [Standardization.RegionRelativeScope(modelDB, descriptors)],
                    mad_bound=15.0, fancyMAD=self.fancyMAD, exceptionNames=exceptionNames)
        else:
            self.descriptorStandardization = Standardization.BucketizedStandardization(
                    [Standardization.GlobalRelativeScope(descriptors)],
                    mad_bound=15.0, fancyMAD=self.fancyMAD, exceptionNames=exceptionNames)

        self.indexSelector = MarketIndex.MarketIndexSelector(modelDB, marketDB)
        # Set up TOPIX replication - FIXME
        self.topixReplicator = TOPIX.TOPIXReplicator(self, modelDB)
        modelDB.createCurrencyCache(marketDB)

    def generate_model_specific_exposures(self, modelDate, data, modelDB, marketDB):
        """Generate the non-default factors.
        """
        beta = numpy.zeros((len(data.universe)), float)
        # Cap-based style factors here
        if not hasattr(self, 'estuMap') or self.estuMap is None:
            return data.exposureMatrix

        # Small-cap factors
        dateList = modelDB.getDates(self.rmg, modelDate, 61, excludeWeekend=True)
        styleNames = [s.name for s in self.styles]
        scList = [sc for sc in self.estuMap.keys() if sc in styleNames]
        for sc in scList:
            beta = Matrices.allMasked((len(data.universe)), float)
            scAssets = [sid for sid in self.estuMap[sc].assets if sid in data.universe]
            if len(scAssets) < 1:
                logging.warning('No assets in %s universe', sc)
            else:
                qualifiedAssets = modelDB.loadESTUQualifyHistory(
                    self.rms_id, scAssets, dateList, estuInstance=self.estuMap[sc])
                qualifiedAssets = ma.filled(ma.sum(qualifiedAssets.data, axis=1), 0.0)
                if len(qualifiedAssets) < 1:
                    logging.warning('No assets qualified for %s factor', sc)
                else:
                    qualifiedAssets = qualifiedAssets / float(numpy.max(qualifiedAssets, axis=None))
                    for (i_c, idx) in enumerate([data.assetIdxMap[sid] for sid in scAssets]):
                        beta[idx] = qualifiedAssets[i_c]
            data.exposureMatrix.addFactor(sc, beta, ExposureMatrix.StyleFactor)
            if self.exposureStandardization.exceptionNames is None:
                self.exposureStandardization.exceptionNames = [sc]
            else:
                self.exposureStandardization.exceptionNames.append(sc)
        self.exposureStandardization.exceptionNames = list(set(self.exposureStandardization.exceptionNames))

        return data.exposureMatrix

## class JPResearchModel2016(EquityModel.FundamentalModel):
##     """ Final JP research model incorporating the follwong changes:
##             1. Split EY and Value
##             2. Combine D/A and D/E in Leverage
##             3. Include profitability factor
##             4. Include Dividend Yield factor
##             5. Enhanced ESTU coverage (based on inclusion of several exchanges)
##             6. Updated definition of MidCap based on clear inclusion criteria - similar to that of Russell Japan Mid Cap Index 80-97.5%
##             7. Updated definitions of Growth, Market sensitivity, MTM, Volatility, Exchange Rate sensitivity
##             8. GICS2016 definitions
##     """
##     estu_parameters = {
##                        'minNonZero':0.75,
##                        'minNonMissing':0.95,
##                        'maskZeroWithNoADV_Flag': True,
##                        'returnLegacy_Flag': False,
##                        'CapByNumber_Flag':False,
##                        'CapByNumber_hiCapQuota':np.nan,
##                        'CapByNumber_lowCapQuota':np.nan,
##                        'market_lower_pctile':1 ,
##                        'country_lower_pctile':5,
##                        'industry_lower_pctile':5,
##                        'dummyThreshold': 10,
##                        'inflation_cutoff':0.01
##                         }
## 
##     rm_id = -5
##     revision = 18
##     rms_id = -1016
##     inflation_cutoff = 0.01
##     regionalDescriptorStructure = False
##     twoRegressionStructure = True
##     multiCountry = False
## 
##     # List of 13 style factors in the model
##     styleList = [
##                 'Earnings Yield',
##                 'Value',
##                 'Leverage',
##                 'Growth',
##                 'Profitability',
##                 'Dividend Yield',
##                 'Size',
##                 'Liquidity',
##                 'Market Sensitivity',
##                 'Volatility',
##                 'Medium-Term Momentum',
##                 'MidCap',
##                 'Exchange Rate Sensitivity',
##                 ]
## 
##     DescriptorMap = {
##             'Earnings Yield': ['Earnings_to_Price_Annual', 'Est_Earnings_to_Price_12MFL_Annual'],
##             'Value': ['Book_to_Price_Annual'],
##             'Leverage': ['Debt_to_Assets_Annual', 'Debt_to_Equity_Annual'],
##             'Growth': ['Earnings_Growth_RPF_Annual', 'Sales_Growth_RPF_Annual'],
##             'Dividend Yield': ['Dividend_Yield_Annual'],
##             'Size': ['LnIssuerCap'],
##             'Liquidity': ['LnTrading_Activity_60D'],
##             'Market Sensitivity': ['Market_Sensitivity_250D'],
##             'Volatility': ['Volatility_125D'],
##             'Medium-Term Momentum': ['Momentum_250x20D'],
##             'Exchange Rate Sensitivity': ['XRate_104W_USD'],
##             'Profitability': ['Return_on_Equity_Annual', 'Return_on_Assets_Annual',
##                               'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
##                               'Sales_to_Assets_Annual', 'Gross_Margin_Annual'],
##             }
## 
##     DescriptorWeights = {'Earnings Yield': [0.75, 0.25]}
##     smallCapMap = {'MidCap': [80.0, 97.5],}
##     noProxyList = ['Dividend Yield']
##     fillMissingList = ['Earnings Yield', 'Value', 'Leverage', 'Growth', 'Profitability']
##     fillWithZeroList = ['Dividend Yield']
##     shrinkList = {'Liquidity': 60,
##                   'Market Sensitivity': 250,
##                   'Volatility': 125,
##                   'Medium-Term Momentum': 250}
##     orthogList = {'Volatility': [['Market Sensitivity'], True, 1.0]}
## 
##     # Setting up market intercept if relevant
##     interceptFactor = 'Market Intercept'
##     intercept = ModelFactor(interceptFactor, interceptFactor)
##     industryClassification = Classification.GICSCustomJP(datetime.date(2016,9,1))
## 
##     estuAssetTypes = ['REIT', 'Com']
## 
##     def __init__(self, modelDB, marketDB):
##         self.log = logging.getLogger('RiskModels.JPResearchModel1')
##         # Set up relevant styles to be created/used
##         ModelParameters2017.defaultExposureParameters(self, self.styleList)
##         self.styles = [s for s in self.totalStyles if s.name in self.styleList]
##         EquityModel.FundamentalModel.__init__(self, ['SEDOL'], modelDB, marketDB)
## 
##         # Set up regression parameters
##         dummyThreshold = 10
## 
##         # Run risk based on the square root of market cap weights  
##         self.internalCalculator = ModelParameters2017.defaultRegressionParameters(
##                 self, modelDB,
##                 dummyType='Industry Groups',
##                 dummyThreshold=dummyThreshold,
##                 marketRegression=False,
##                 kappa=5.0,
##                 useRealMCaps=True,
##                 regWeight = 'rootCap',
##                 )
## 
##         # Set up external regression parameters
##         self.returnCalculator = ModelParameters2017.defaultRegressionParameters(
##                 self, modelDB,
##                 dummyType='Industry Groups',
##                 dummyThreshold=dummyThreshold,
##                 marketRegression=False,
##                 kappa=None,
##                 useRealMCaps=True,
##                 regWeight='invSpecificVariance'
##                 )
## 
##         # This controls the FMP regression
##         self.fmpCalculator = ModelParameters2017.defaultRegressionParameters(
##                 self, modelDB,
##                 dummyType='Industry Groups',
##                 marketRegression=False,
##                 dummyThreshold=dummyThreshold,
##                 kappa=None)
## 
##         # Set up risk parameters
##         ModelParameters2017.defaultFundamentalCovarianceParameters(
##                 self, nwLag=2,
##                 varDVAOnly=False, unboundedDVA=False,
##                 )
## 
##         # Set up standardization parameters
##         gloScope = Standardization.GlobalRelativeScope([f.name for f in self.styles])
##         self.exposureStandardization = Standardization.BucketizedStandardization([gloScope],
##                 fillWithZeroList=self.fillWithZeroList)
## 
##         # Set up descriptor standardization parameters
##         descriptors = sorted(list(set([item for sublist
##             in self.DescriptorMap.values() for item in sublist])))
##         exceptionNames = [self.DescriptorMap[sf] for sf in self.noProxyList]
##         exceptionNames = list(itertools.chain.from_iterable(exceptionNames))
##         self.descriptorStandardization = Standardization.BucketizedStandardization(
##                     [Standardization.GlobalRelativeScope(descriptors)],
##                     mad_bound=15.0, fancyMAD=self.fancyMAD, exceptionNames=exceptionNames)
## 
##         self.indexSelector = MarketIndex.MarketIndexSelector(modelDB, marketDB)
##         # Set up TOPIX replication - FIXME
##         self.topixReplicator = TOPIX.TOPIXReplicator(self, modelDB)
##         modelDB.createCurrencyCache(marketDB)
## 
##     def generate_model_specific_exposures(self, modelDate, data, modelDB, marketDB):
##         """Generate the non-default factors.
##         """
##         beta = numpy.zeros((len(data.universe)), float)
##         # Cap-based style factors here
##         if not hasattr(self, 'estuMap') or self.estuMap is None:
##             return data.exposureMatrix
## 
##         # Small-cap factors
##         dateList = modelDB.getDates(self.rmg, modelDate, 61, excludeWeekend=True)
##         styleNames = [s.name for s in self.styles]
##         scList = [sc for sc in self.estuMap.keys() if sc in styleNames]
##         for sc in scList:
##             beta = Matrices.allMasked((len(data.universe)), float)
##             scAssets = [sid for sid in self.estuMap[sc].assets if sid in data.universe]
##             if len(scAssets) < 1:
##                 logging.warning('No assets in %s universe', sc)
##             else:
##                 qualifiedAssets = modelDB.loadESTUQualifyHistory(
##                     self.rms_id, scAssets, dateList, estuInstance=self.estuMap[sc])
##                 qualifiedAssets = ma.filled(ma.sum(qualifiedAssets.data, axis=1), 0.0)
##                 if len(qualifiedAssets) < 1:
##                     logging.warning('No assets qualified for %s factor', sc)
##                 else:
##                     qualifiedAssets = qualifiedAssets / float(numpy.max(qualifiedAssets, axis=None))
##                     for (i_c, idx) in enumerate([data.assetIdxMap[sid] for sid in scAssets]):
##                         beta[idx] = qualifiedAssets[i_c]
##             data.exposureMatrix.addFactor(sc, beta, ExposureMatrix.StyleFactor)
##             if self.exposureStandardization.exceptionNames is None:
##                 self.exposureStandardization.exceptionNames = [sc]
##             else:
##                 self.exposureStandardization.exceptionNames.append(sc)
##         self.exposureStandardization.exceptionNames = list(set(self.exposureStandardization.exceptionNames))
## 
##         return data.exposureMatrix
## 
##     def generate_estimation_universe_old(self, modelDate, data, modelDB, marketDB, excludeFactors=None):
##         """Estimation universe selection criteria for JP.
##         """
##         import pandas as pd
##         #import ipdb;ipdb.set_trace()
##         self.log.info('generate_estimation_universe for JP Model: begin')
##         buildEstu = EstimationUniverse.ConstructEstimationUniverse(data.universe, self, modelDB, marketDB)
## 
##         # Set up various eligible and total universes
##         universeIdx = range(len(buildEstu.assets))
##         originalEligibleUniverse = list(data.eligibleUniverse)
##         originalEligibleUniverseIdx = [data.assetIdxMap[sid] for sid in originalEligibleUniverse]
## 
##         # Remove nursery market assets
##         if len(data.nurseryUniverse) > 0:
##             logging.info('Checking for assets from nursery markets')
##             ns_indices = [data.assetIdxMap[sid] for sid in data.nurseryUniverse]
##             (eligibleUniverseIdx, nonest) = buildEstu.exclude_specific_assets(
##                     ns_indices, baseEstu=originalEligibleUniverseIdx)
##             if n != len(eligibleUniverseIdx):
##                 n = len(eligibleUniverseIdx)
##                 logging.info('ESTU currently stands at %d stocks', n)
##         else:
##             eligibleUniverseIdx = originalEligibleUniverseIdx
## 
##         universe = pd.DataFrame(buildEstu.assets,columns=['SubID_Obj'])
##         universeIdx = universe.index
##         original_eligibleUniverse = pd.DataFrame(zip(data.eligibleUniverse,originalEligibleUniverseIdx),columns=['SubID_Obj','originalEligibleUniverseIdx'])
## 
##         logging.info('ESTU currently stands at %d stocks based on original eligible universe', len(originalEligibleUniverse))
## 
##         # Report on thinly-traded assets over the entire universe
##         logging.info('Looking for thinly-traded stocks')
##         # (nonSparseIdx, sparse) = buildEstu.exclude_thinly_traded_assets(modelDate, data, baseEstu=universeIdx, minNonZero=0.75)
##         # data.nonSparse = numpy.take(buildEstu.assets, nonSparseIdx, axis=0)
##         (non_thin_idx, thin_idx) = buildEstu.exclude_thinly_traded_assets(modelDate, data, baseEstu=universe.index.values, minNonZero=0.75)
##         # (non_thin_idx, thin_idx) = buildEstu.exclude_thinly_traded_assets(modelDate, data, baseEstu=universe.index.values, minNonZero=0.95)
##         data.nonSparse = pd.DataFrame(universe,index=non_thin_idx)["SubID_Obj"].tolist()
## 
##         # Exclude thinly traded assets
##         estu_withoutThin_idx = list(set(eligibleUniverseIdx).intersection(set(non_thin_idx)))
##         logging.info('ESTU currently stands at %d stocks after Filtering thinly-traded assets', len(estu_withoutThin_idx))
##         # (2) Filtering tiny-cap assets by market, country and industry
##         # (a) Weed out tiny-cap assets by market
##         lowerBound = 1
##         logging.info('Filtering by top %d%% mcap on entire market', 100-lowerBound)
##         (large_byMkt_idx, nonest1) = buildEstu.exclude_by_cap_ranking(data, modelDate, baseEstu=estu_withoutThin_idx,
##                                                                      lower_pctile=lowerBound, method='percentage')
##         # (2b) Weed out tiny-cap assets by country
##         lowerBound = 5
##         logging.info('Filtering by top %d%% mcap on country', 100-lowerBound)
##         (large_byCntry_idx, nonest2) = buildEstu.exclude_by_cap_ranking(data, modelDate, baseEstu=estu_withoutThin_idx,
##                                                                        byFactorType=ExposureMatrix.CountryFactor,
##                                                                        lower_pctile=lowerBound, method='percentage',
##                                                                        excludeFactors=excludeFactors)
##         # (2c) Perform similar check by industry
##         logging.info('Filtering by top %d%% mcap on industry', 100-lowerBound)
##         (large_byIndtry_idx, nonest3) = buildEstu.exclude_by_cap_ranking(
##                data, modelDate, baseEstu=estu_withoutThin_idx,
##                byFactorType=ExposureMatrix.IndustryFactor,
##                lower_pctile=lowerBound, method='percentage',
##                excludeFactors=excludeFactors)
##     
##         estu_mktCap_idx = set(large_byMkt_idx).union(large_byCntry_idx).union(large_byIndtry_idx)
##         estu_mktCap_idx = list(estu_mktCap_idx)
##         tinyCap_idx = list(set(estu_withoutThin_idx).difference(estu_mktCap_idx))
## 
##         logging.info('ESTU currently stands at %d stocks after Filtering by Market Cap.', len(estu_mktCap_idx))
## 
##         # candid_univ_idx = eligibleUniverseIdx
##         candid_univ_idx = estu_withoutThin_idx
##         # Inflate any thin countries or industries - add 2*
##         # minFactorWidth=2*self.returnCalculator.allParameters[0].getThinFactorInformation().dummyThreshold
##         minFactorWidth=self.returnCalculator.allParameters[0].getThinFactorInformation().dummyThreshold
##         logging.info('Inflating any thin factors')
## 
##         (estu_inflated_idx, nonest,herf_num_list) = buildEstu.pump_up_factors2(
##                 data, modelDate, currentEstu=estu_mktCap_idx,
##                 baseEstu=candid_univ_idx,
##                 byFactorType=[ExposureMatrix.CountryFactor, ExposureMatrix.IndustryFactor],
##                 minFactorWidth=minFactorWidth,
##                 cutOff = self.inflation_cutoff, 
##                 excludeFactors=excludeFactors)
## 
##         logging.info('ESTU currently stands at %d stocks', len(estu_inflated_idx))
##         herf_num_list = pd.DataFrame(herf_num_list)
##         herf_num_list.to_csv('herf_num_list.csv')
##         # Apply grandfathering rules
##         logging.info('Incorporating grandfathering')
##         (estu_final_Idx, ESTUQualify, nonest) = buildEstu.grandfather(modelDate,
##                 estu_inflated_idx, baseEstu=candid_univ_idx,
##                 estuInstance=self.estuMap['main'])
## 
##         totalcap = ma.sum(ma.take(data.marketCaps, estu_final_Idx, axis=0), axis=0) / 1e9
##         self.log.info('Final estu contains %d assets, %.2f bn (%s)',len(estu_final_Idx), totalcap, self.numeraire.currency_code)
## 
##         self.log.info('Final estu contains %d assets in thin-traded assets.',sum(pd.Series(estu_inflated_idx).isin(thin_idx)))
##         self.log.info('Final estu contains %d assets in tiny cap assets.',sum(pd.Series(estu_inflated_idx).isin(tinyCap_idx)))
##         self.log.debug('generate_estimation_universe: end')
## 
##         # If we have a family of estimation universes, populate the main estu accordingly
##         self.estuMap['main'].assets = [buildEstu.assets[idx] for idx in estu_final_Idx]
##         self.estuMap['main'].qualify = [buildEstu.assets[idx] for idx in ESTUQualify]
## 
##         return estu_final_Idx
## 
##     def generate_estimation_universe(self, modelDate, data, modelDB, marketDB, excludeFactors=None):
##         estu1 = self.generate_estimation_universe_v2(modelDate, data, modelDB, marketDB, excludeFactors=None)
##         # estu2 = self.generate_estimation_universe_old(modelDate, data, modelDB, marketDB, excludeFactors=None)
##         # print modelDate, estu1, estu2
##         return estu1 # 

class JPResearchModel2016(RiskModels_V3.JPAxioma2009MH):
    """Production AX-JP2 Japan model, Kappa = None. 
    For comparison to JP4 single stage, kappa = None and sqrtmcap (default)
    """

    rm_id = -5
    revision = 18
    rms_id = -1016

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.JPAxioma2009MH')
        MFM.SingleCountryFundamentalModel.__init__(
                            self, ['SEDOL'], modelDB, marketDB)
        self.setCalculators(modelDB)
        modelDB.createCurrencyCache(marketDB)
        # Set up TOPIX replication
        self.topixReplicator = TOPIX.TOPIXReplicator(self, modelDB)
        self.indexSelector = MarketIndex.MarketIndexSelector(modelDB, marketDB)

    def setCalculators(self, modelDB, overrider = False):
        # Set up regression parameters
        self.returnCalculator = \
                riskmodels.defaultRegressionParameters(\
                self, modelDB, dummyThreshold=6.0, overrider=overrider, k_rlm=1000)
        # Set up risk parameters
        riskmodels.defaultFundamentalCovarianceParameters(self, nwLag=4, dva='slowStep', overrider=overrider)

class JPResearchModelFinalESTUInvSpVar(JPResearchModelFinalESTU):
    """ Same as Final Research Model except factor returns are now overwritten based
    on regression using inverse specific variance of assets rather than square-root of market cap
    """
    rm_id = -5
    revision = 19
    rms_id = -1017

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.JPResearchModelInvSpVar')
        # Set up relevant styles to be created/used
        ModelParameters.defaultExposureParametersV3(self, self.styleList)
        self.styles = [s for s in self.totalStyles if s.name in self.styleList]
        MFM.FundamentalModel.__init__(self, ['SEDOL'], modelDB, marketDB)

        # Set up regression parameters
        dummyThreshold = 10

        ''' When generating the model first time, leave the regWeight rootCap'''
        '''
        self.returnCalculator = ModelParameters.defaultRegressionParameters(
                self, modelDB,
                dummyType='Industry Groups',
                dummyThreshold=dummyThreshold,
                marketRegression=False,
                kappa=5.0,
                regWeight='rootCap'
                )
        '''

        ''' When running the model second time to overwrite the factor returns, use regWeight='invSpecificVariance '''
        self.returnCalculator = ModelParameters.defaultRegressionParameters(
                self, modelDB,
                dummyType='Industry Groups',
                dummyThreshold=dummyThreshold,
                useRealMCaps=True,
                marketRegression=False,
                kappa=25.0,
                regWeight='invSpecificVariance'
                )

        # This controls the FMP regression
        self.fmpCalculator = ModelParameters.defaultRegressionParameters(
                self, modelDB,
                dummyType='Industry Groups',
                marketRegression=False,
                dummyThreshold=dummyThreshold,
                kappa=None)

        # Set up risk parameters
        ModelParameters.defaultFundamentalCovarianceParametersV3(
                self, nwLag=2,
                varDVAOnly=False, unboundedDVA=False,
                )

        # Set up standardization parameters
        gloScope = Standardization.GlobalRelativeScope([f.name for f in self.styles])
        self.exposureStandardization = Standardization.BucketizedStandardization([gloScope],
                fillWithZeroList=self.fillWithZeroList)

        # Set up descriptor standardization parameters
        descriptors = sorted(list(set([item for sublist
            in list(self.DescriptorMap.values()) for item in sublist])))
        exceptionNames = [self.DescriptorMap[sf] for sf in self.noProxyList]
        exceptionNames = list(itertools.chain.from_iterable(exceptionNames))
        if not self.SCM:
            self.descriptorStandardization = Standardization.BucketizedStandardization(
                    [Standardization.RegionRelativeScope(modelDB, descriptors)],
                    mad_bound=15.0, fancyMAD=self.fancyMAD, exceptionNames=exceptionNames)
        else:
            self.descriptorStandardization = Standardization.BucketizedStandardization(
                    [Standardization.GlobalRelativeScope(descriptors)],
                    mad_bound=15.0, fancyMAD=self.fancyMAD, exceptionNames=exceptionNames)

        self.indexSelector = MarketIndex.MarketIndexSelector(modelDB, marketDB)
        # Set up TOPIX replication - FIXME
        self.topixReplicator = TOPIX.TOPIXReplicator(self, modelDB)
        modelDB.createCurrencyCache(marketDB)
