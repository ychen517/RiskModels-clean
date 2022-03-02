
import datetime
import logging
import numpy as np
import numpy.ma as ma
import numpy
import pandas
import itertools
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
from riskmodels import Standardization
from riskmodels import TOPIX
from riskmodels import MarketIndex
from riskmodels import LegacyModelParameters as ModelParameters
from riskmodels import ModelParameters2017
from riskmodels import FactorReturns

class USResearchModel1(RiskModels.USAxioma2016MH):
    """US research model, version 1.
    """
    rm_id = -18
    revision = 1
    rms_id = -110

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.USResearchModel1')
        self.DescriptorMap['Volatility'] = ['Historical_Residual_Volatility_125D']
        # Set important model parameters
        ModelParameters2017.defaultModelSettingsUS4(self)
        # Set up relevant styles to be created/used
        ModelParameters2017.defaultExposureParameters(self, self.styleList)
        self.styles = [s for s in self.totalStyles if s.name in self.styleList]
        RiskModels.USAxioma2016MH.__init__(self, modelDB, marketDB)

class USResearchModel2(RiskModels.USAxioma2016SH):
    """US research model, version 2.
    """
    rm_id = -18
    revision = 2
    rms_id = -111

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.USResearchModel2')
        # Set important model parameters
        ModelParameters2017.defaultModelSettingsUS4(self)
        # Set up relevant styles to be created/used
        ModelParameters2017.defaultExposureParameters(self, self.styleList)
        self.styles = [s for s in self.totalStyles if s.name in self.styleList]
        RiskModels.USAxioma2016SH.__init__(self, modelDB, marketDB)

class USResearchModel3(RiskModels.USAxioma2016MH_S):
    """US research model, version 3.
    """
    rm_id = -18
    revision = 3
    rms_id = -112

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.USResearchModel3')
        # Set important model parameters
        ModelParameters2017.defaultModelSettingsUS4(self, statModel=True)
        RiskModels.USAxioma2016MH_S.__init__(self, modelDB, marketDB)

class USResearchModel4(RiskModels.USAxioma2016SH_S):
    """US research model, version 4.
    """
    rm_id = -18
    revision = 4
    rms_id = -113

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.USResearchModel4')
        # Set important model parameters
        ModelParameters2017.defaultModelSettingsUS4(self, statModel=True)
        RiskModels.USAxioma2016SH_S.__init__(self, modelDB, marketDB)

class EstherUSResearchModel(RiskModels.USAxioma2016MH):
    """US4 fundamental medium-horizon small cap research model.
    """
    rm_id = -18
    revision = 8
    rms_id = -188

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
                 #'MidCap',
                 'Exchange Rate Sensitivity',
                ]

    DescriptorMap = {
            'Earnings Yield': ['Earnings_to_Price_Quarterly', 'Est_Earnings_to_Price_12MFL_Quarterly'],
            'Value': ['Book_to_Price_Quarterly'],
            'Leverage': ['Debt_to_Assets_Quarterly', 'Debt_to_Equity_Quarterly'],
            'Growth': ['Earnings_Growth_RPF_AFQ', 'Sales_Growth_RPF_AFQ'],
            'Dividend Yield': ['Dividend_Yield_Quarterly'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_60D'],
            'Market Sensitivity': ['Market_Sensitivity_250D'],
            'Volatility': ['Volatility_125D'],
            'Medium-Term Momentum': ['Momentum_250x20D'],
            'Exchange Rate Sensitivity': ['XRate_104W_XDR'],
            'Profitability': ['Return_on_Equity_Quarterly', 'Return_on_Assets_Quarterly',
                              'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
                              'Sales_to_Assets_Quarterly', 'Gross_Margin_Quarterly'],
            }

    DescriptorWeights = {'Earnings Yield': [0.75, 0.25]}
    #smallCapMap = {'MidCap': [66.67, 86.67],}
    noProxyList = ['Dividend Yield']
    fillMissingList = ['Earnings Yield', 'Value', 'Leverage', 'Growth', 'Profitability']
    fillWithZeroList = ['Dividend Yield']
    shrinkList = {'Liquidity': 60,
                  'Market Sensitivity': 250,
                  'Volatility': 125,
                  'Medium-Term Momentum': 250}
    orthogList = {'Volatility': [['Market Sensitivity'], True, 1.0]}
    regionalStndList = []

    # descriptor settings
    noStndDescriptor = []
    noCloneDescriptor = []
    for s in noProxyList:
        noStndDescriptor.extend(DescriptorMap[s])
        noCloneDescriptor.extend(DescriptorMap[s])

    # Setting up market intercept if relevant
    interceptFactor = 'Market Intercept'
    intercept = ModelFactor(interceptFactor, interceptFactor)
    industryClassification = Classification.GICSIndustries(datetime.date(2016,9,1))

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.USAxioma2016MH')
        # Set important model parameters
        ModelParameters2017.defaultModelSettingsUS4(self)
        self.twoRegressionStructure = True
        
        # Set up relevant styles to be created/used
        ModelParameters2017.defaultExposureParameters(self, self.styleList)
        self.styles = [s for s in self.totalStyles if s.name in self.styleList]
        EquityModel.FundamentalModel.__init__(self, ['CUSIP'], modelDB, marketDB)

        # Set up eligible universe parameters
        self.elig_parameters = {'assetTypes': self.commonStockTypes + ['REIT'],
                                'excludeTypes': None,
                                'HomeCountry_List': ['US'],
                                'use_isin_country_Flag': True}

        # Set up regression parameters
        dummyThreshold = 10
        self.internalCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType='Industry Groups',
                dummyThreshold=dummyThreshold,
                marketRegression=False,
                kappa=5.0,
                useRealMCaps=True,
                regWeight='rootCap',
                )
        self.returnCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType='Industry Groups',
                dummyThreshold=dummyThreshold,
                marketRegression=False,
                kappa=25.0,
                useRealMCaps=True,
                regWeight='rootCap',
                )

        # This controls the FMP regression
        self.fmpCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType='Industry Groups',
                marketRegression=False,
                dummyThreshold=dummyThreshold,
                useRealMCaps=True,
                kappa=None)

        # Set up risk parameters
        ModelParameters2017.defaultFundamentalCovarianceParameters(
                self, nwLag=2, varDVAOnly=False, unboundedDVA=False)

        # Set up standardization parameters
        gloScope = Standardization.GlobalRelativeScope([f.name for f in self.styles])
        self.exposureStandardization = Standardization.BucketizedStandardization(\
                [gloScope],fillWithZeroList=self.fillWithZeroList)

        # Set up descriptor standardization parameters
        descriptors = sorted(list(set([item for sublist in self.DescriptorMap.values() for item in sublist])))
        self.descriptorStandardization = Standardization.BucketizedStandardization(
                [Standardization.GlobalRelativeScope(descriptors)],
                mad_bound=15.0, fancyMAD=self.fancyMAD, exceptionNames=self.noStndDescriptor)

        self.indexSelector = MarketIndex.MarketIndexSelector(modelDB, marketDB)
        modelDB.createCurrencyCache(marketDB)

    def generate_estimation_universe(self, modelDate, data, modelDB, marketDB,
            excludeFactors=None, grandfatherRMS_ID=None):
        """Estimation universe selection criteria for AX-US.
        """
        self.log.debug('generate_estimation_universe: begin')

        buildEstu = EstimationUniverse.ConstructEstimationUniverse(
                data.exposureMatrix.getAssets(), self, modelDB, marketDB)

        # Set up various eligible and total universes
        universeIdx = list(range(len(buildEstu.assets)))
        if not hasattr(data, 'eligibleUniverse'):
            eligibleUniverse = list(buildEstu.assets)
            eligibleUniverseIdx = list(universeIdx)
        else:
            eligibleUniverse = list(data.eligibleUniverse)
            eligibleUniverseIdx = [data.assetIdxMap[sid] for sid in eligibleUniverse]
        logging.info('Estimation universe currently stands at %d stocks', len(eligibleUniverse))

        # Keep only issues trading on NYSE and NASDAQ
        fade_in_date = datetime.date(2009,1,1)
        fade_out_date = datetime.date(2010,1,1)
        logging.info('Excluding assets not on NAS and NYS')
        (eligibleExchangeIdx, nonest) = buildEstu.exclude_by_market_type(
                modelDate, data, includeFields=['NAS','NYS'], excludeFields=None,
                baseEstu=eligibleUniverseIdx)
        exchangeDownWeight = numpy.zeros((len(universeIdx)), float)
        numpy.put(exchangeDownWeight, eligibleExchangeIdx, 1.0)
        delta = 0.25
        if modelDate < fade_in_date:
            eligibleExchangeIdx = list(eligibleUniverseIdx)
        elif modelDate > fade_out_date:
            delta = 0.0
        else:
            eligibleExchangeIdx = list(eligibleUniverseIdx)
            delta = delta * float((fade_out_date - modelDate).days) / float((fade_out_date - fade_in_date).days)
        if delta > 0.0:
            logging.info('Downweighting factor for pre-%d OTCs: %.3f', fade_out_date.year, delta)
        numpy.put(exchangeDownWeight, nonest, delta)
        eligibleExchange = [buildEstu.assets[idx] for idx in eligibleExchangeIdx]
        logging.info('Estimation universe currently stands at %d stocks', len(eligibleExchangeIdx))

        # Report on thinly-traded assets over the entire universe
        logging.info('Looking for sparsely traded stocks')
        (nonSparse, sparse) = buildEstu.exclude_thinly_traded_assets(
                                modelDate, data, minNonZero=0.5,
                                baseEstu=eligibleUniverseIdx, legacy=False)
        nonSparse = [buildEstu.assets[idx] for idx in nonSparse]

        # Exclude thinly traded assets if required
        estuIdx = [data.assetIdxMap[sid] for sid in eligibleExchange if sid in nonSparse]
        exchangeDownWeightNS = numpy.take(exchangeDownWeight, estuIdx, axis=0)
        logging.info('Estimation universe currently stands at %d stocks', len(estuIdx))

        # Rank stuff by market cap and total volume over past year
        logging.info('Excluding assets based on cap and volume')
        (estuIdx, nonest) = buildEstu.filter_by_cap_and_volume(
                data, modelDate, baseEstu=estuIdx, 
                downWeights=exchangeDownWeightNS)
        logging.info('Estimation universe currently stands at %d stocks', len(estuIdx))

        # Exclude assets from estuIdx with mcaps < .8 mcap quanitle
        idxAssetMap = {v: k for k,v in data.assetIdxMap.items()}
        mcap = pandas.Series(data.marketCaps, index=data.universe)
        estuMcap = mcap.reindex(index=[idxAssetMap[a] for a in estuIdx])
        maxMcap = estuMcap[estuMcap<estuMcap.quantile(.8)].max()
        estuAssets = estuMcap[estuMcap<estuMcap.quantile(.8)].index
        estuIdx = [data.assetIdxMap[a] for a in estuAssets]

        eligMcap = mcap.reindex(index=[idxAssetMap[a] for a in eligibleExchangeIdx])
        eligAssets = eligMcap[eligMcap<=maxMcap].index
        eligibleExchangeIdx = [data.assetIdxMap[a] for a in eligAssets]

        # Inflate thin industry factors if possible
        logging.info('Inflating any thin factors')
        minFactorWidth=2*self.returnCalculator.allParameters[0].dummyThreshold
        (estuIdx, nonest) = buildEstu.pump_up_factors(
                data, modelDate, currentEstu=estuIdx, baseEstu=eligibleExchangeIdx,
                minFactorWidth=minFactorWidth, downWeights=exchangeDownWeight)
        logging.info('Estimation universe currently stands at %d stocks', len(estuIdx))

        # Apply grandfathering rules
        logging.info('Applying grandfathering')
        (estuIdx, ESTUQualify, nonest) = buildEstu.grandfather(
                modelDate, estuIdx, baseEstu=eligibleExchangeIdx, estuInstance=self.estuMap['main'],
                grandfatherRMS_ID=grandfatherRMS_ID)
        logging.info('Estimation universe currently stands at %d stocks', len(estuIdx))

        # Drop assets that are more than 1.5* the max mcap on the modelDate
        estuMcap = mcap.reindex(index=[idxAssetMap[a] for a in estuIdx])
        estuAssets = estuMcap[estuMcap<=1.5*maxMcap].index
        estuIdx = [data.assetIdxMap[a] for a in estuAssets]

        self.estuMap['main'].assets = [buildEstu.assets[idx] for idx in estuIdx]
        self.estuMap['main'].qualify = [buildEstu.assets[idx] for idx in ESTUQualify]

        self.log.debug('generate_estimation_universe: end')
        return estuIdx


class DieterUSResearchModel(EquityModel.FundamentalModel):
    """US4 fundamental medium-horizon research model.
    """
    rm_id = -18
    revision = 6
    rms_id = -186
    legacyMCapDates = True
    regionalDescriptorStructure = False
    twoRegressionStructure = False
    fancyMAD = False
    firstReturnDate = datetime.date(1980,1,1)
    # Back-compatibility for fill-in rules
    gicsDate = datetime.date(2014,3,1)

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
                 'MidCap',
                 'Exchange Rate Sensitivity',
                ]

    DescriptorMap = {
            'Earnings Yield': ['Earnings_to_Price_Quarterly', 'Est_Earnings_to_Price_12MFL_Quarterly'],
            'Value': ['Book_to_Price_Quarterly'],
            'Leverage': ['Debt_to_Assets_Quarterly', 'Debt_to_Equity_Quarterly'],
            'Growth': ['Earnings_Growth_RPF_AFQ', 'Sales_Growth_RPF_AFQ'],
            'Dividend Yield': ['Dividend_Yield_Quarterly'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_60D'],
            'Market Sensitivity': ['Market_Sensitivity_250D'],
            'Volatility': ['Volatility_125D'],
            'Medium-Term Momentum': ['Momentum_250x20D'],
            'Exchange Rate Sensitivity': ['XRate_104W_XDR'],
            'Profitability': ['Return_on_Equity_Quarterly', 'Return_on_Assets_Quarterly',
                              'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
                              'Sales_to_Assets_Quarterly', 'Gross_Margin_Quarterly'],
            }

    DescriptorWeights = {'Earnings Yield': [0.75, 0.25]}
    smallCapMap = {'MidCap': [66.67, 86.67],}
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

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.USAxioma2016MH')
        # Set up relevant styles to be created/used
        ModelParameters2017.defaultExposureParameters(self, self.styleList)
        self.styles = [s for s in self.totalStyles if s.name in self.styleList]
        EquityModel.FundamentalModel.__init__(self, ['CUSIP'], modelDB, marketDB)

        # Set up eligible universe parameters
        self.elig_parameters = {'assetTypes': self.commonStockTypes + ['REIT'],
                                'excludeTypes': None,
                                'HomeCountry_List': ['US'],
                                'use_isin_country_Flag': True}

        # Set up regression parameters
        dummyThreshold = 10
        self.returnCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType='Industry Groups',
                dummyThreshold=dummyThreshold,
                marketRegression=False,
                kappa=5.0,
                useRealMCaps=True,
                regWeight='rootCap',
                )

        # This controls the FMP regression
        self.fmpCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType='Industry Groups',
                marketRegression=False,
                dummyThreshold=dummyThreshold,
                useRealMCaps=True,
                kappa=None)

        # Set up risk parameters
        ModelParameters2017.defaultFundamentalCovarianceParameters(
                self, nwLag=2, varDVAOnly=False, unboundedDVA=False)

        # Set up standardization parameters
        gloScope = Standardization.GlobalRelativeScope([f.name for f in self.styles])
        self.exposureStandardization = Standardization.BucketizedStandardization(\
                [gloScope],fillWithZeroList=self.fillWithZeroList)

        # Set up descriptor standardization parameters
        descriptors = sorted(list(set([item for sublist in self.DescriptorMap.values() for item in sublist])))
        exceptionNames = [self.DescriptorMap[sf] for sf in self.noProxyList]
        exceptionNames = list(itertools.chain.from_iterable(exceptionNames))
        self.descriptorStandardization = Standardization.BucketizedStandardization(
                [Standardization.GlobalRelativeScope(descriptors)],
                mad_bound=15.0, fancyMAD=self.fancyMAD, exceptionNames=exceptionNames)

        self.indexSelector = MarketIndex.MarketIndexSelector(modelDB, marketDB)
        modelDB.createCurrencyCache(marketDB)

    def generate_estimation_universe(self, modelDate, data, modelDB, marketDB, excludeFactors=None):
        """Estimation universe selection criteria for AX-US.
        """
        self.log.debug('generate_estimation_universe: begin')

        buildEstu = EstimationUniverse.ConstructEstimationUniverse(
                data.exposureMatrix.getAssets(), self, modelDB, marketDB)

        # Set up various eligible and total universes
        universeIdx = list(range(len(buildEstu.assets)))
        if not hasattr(data, 'eligibleUniverse'):
            eligibleUniverse = list(buildEstu.assets)
            eligibleUniverseIdx = list(universeIdx)
        else:
            eligibleUniverse = list(data.eligibleUniverse)
            eligibleUniverseIdx = [data.assetIdxMap[sid] for sid in eligibleUniverse]
        logging.info('Estimation universe currently stands at %d stocks', len(eligibleUniverse))

        # Keep only issues trading on NYSE and NASDAQ
        fade_in_date = datetime.date(2009,1,1)
        fade_out_date = datetime.date(2010,1,1)
        logging.info('Excluding assets not on NAS and NYS')
        (eligibleExchangeIdx, nonest) = buildEstu.exclude_by_market_type(
                modelDate, data, includeFields=['NAS','NYS'], excludeFields=None,
                baseEstu=eligibleUniverseIdx)
        exchangeDownWeight = numpy.zeros((len(universeIdx)), float)
        numpy.put(exchangeDownWeight, eligibleExchangeIdx, 1.0)
        delta = 0.25
        if modelDate < fade_in_date:
            eligibleExchangeIdx = list(eligibleUniverseIdx)
        elif modelDate > fade_out_date:
            delta = 0.0
        else:
            eligibleExchangeIdx = list(eligibleUniverseIdx)
            delta = delta * float((fade_out_date - modelDate).days) / float((fade_out_date - fade_in_date).days)
        if delta > 0.0:
            logging.info('Downweighting factor for pre-%d OTCs: %.3f', fade_out_date.year, delta)
        numpy.put(exchangeDownWeight, nonest, delta)
        eligibleExchange = [buildEstu.assets[idx] for idx in eligibleExchangeIdx]
        logging.info('Estimation universe currently stands at %d stocks', len(eligibleExchangeIdx))

        # Report on thinly-traded assets over the entire universe
        logging.info('Looking for sparsely traded stocks')
        (nonSparse, sparse) = buildEstu.exclude_thinly_traded_assets(
                                modelDate, data, minNonZero=0.5,
                                baseEstu=eligibleUniverseIdx, legacy=False)
        nonSparse = [buildEstu.assets[idx] for idx in nonSparse]

        # Exclude thinly traded assets if required
        estuIdx = [data.assetIdxMap[sid] for sid in eligibleExchange if sid in nonSparse]
        exchangeDownWeightNS = numpy.take(exchangeDownWeight, estuIdx, axis=0)
        logging.info('Estimation universe currently stands at %d stocks', len(estuIdx))

        # Rank stuff by market cap and total volume over past year
        logging.info('Excluding assets based on cap and volume')
        (estuIdx, nonest) = buildEstu.filter_by_cap_and_volume(
                data, modelDate, baseEstu=estuIdx, downWeights=exchangeDownWeightNS)
        logging.info('Estimation universe currently stands at %d stocks', len(estuIdx))

        # Inflate thin industry factors if possible
        logging.info('Inflating any thin factors')
        minFactorWidth=2*self.returnCalculator.allParameters[0].getThinFactorInformation().dummyThreshold
        (estuIdx, nonest) = buildEstu.pump_up_factors(
                data, modelDate, currentEstu=estuIdx, baseEstu=eligibleExchangeIdx,
                minFactorWidth=minFactorWidth, downWeights=exchangeDownWeight)
        logging.info('Estimation universe currently stands at %d stocks', len(estuIdx))

        # Apply grandfathering rules
        logging.info('Applying grandfathering')
        (estuIdx, ESTUQualify, nonest) = buildEstu.grandfather(
                modelDate, estuIdx, baseEstu=eligibleExchangeIdx, estuInstance=self.estuMap['main'])
        logging.info('Estimation universe currently stands at %d stocks', len(estuIdx))

        self.estuMap['main'].assets = [buildEstu.assets[idx] for idx in estuIdx]
        self.estuMap['main'].qualify = [buildEstu.assets[idx] for idx in ESTUQualify]

        self.log.debug('generate_estimation_universe: end')
        return estuIdx

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
        if self.exposureStandardization.exceptionNames is not None:
            self.exposureStandardization.exceptionNames = list(set(self.exposureStandardization.exceptionNames))

        return data.exposureMatrix

class LeonUSResearchModel(RiskModels.USAxioma2016MH):
    """US4 fundamental medium-horizon small cap research model.
    """
    rm_id = -18
    revision = 9
    rms_id = -189

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
                 #'MidCap',
                 'Exchange Rate Sensitivity',
                ]

    DescriptorMap = {
            'Earnings Yield': ['Earnings_to_Price_Quarterly', 'Est_Earnings_to_Price_12MFL_Quarterly'],
            'Value': ['Book_to_Price_Quarterly'],
            'Leverage': ['Debt_to_Assets_Quarterly', 'Debt_to_Equity_Quarterly'],
            'Growth': ['Earnings_Growth_RPF_AFQ', 'Sales_Growth_RPF_AFQ'],
            'Dividend Yield': ['Dividend_Yield_Quarterly'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_60D'],
            'Market Sensitivity': ['Market_Sensitivity_USSC_250D'],
            'Volatility': ['Volatility_125D'],
            'Medium-Term Momentum': ['Momentum_250x20D'],
            'Exchange Rate Sensitivity': ['XRate_104W_XDR'],
            'Profitability': ['Return_on_Equity_Quarterly', 'Return_on_Assets_Quarterly',
                              'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
                              'Sales_to_Assets_Quarterly', 'Gross_Margin_Quarterly'],
            }

    DescriptorWeights = {'Earnings Yield': [0.75, 0.25]}
    #smallCapMap = {'MidCap': [66.67, 86.67],}
    noProxyList = ['Dividend Yield']
    fillMissingList = ['Earnings Yield', 'Value', 'Leverage', 'Growth', 'Profitability']
    fillWithZeroList = ['Dividend Yield']
    shrinkList = {'Liquidity': 60,
                  'Market Sensitivity': 250,
                  'Volatility': 125,
                  'Medium-Term Momentum': 250}
    orthogList = {'Volatility': [['Market Sensitivity'], True, 1.0]}
    regionalStndList = []

    # descriptor settings
    noStndDescriptor = []
    noCloneDescriptor = []
    for s in noProxyList:
        noStndDescriptor.extend(DescriptorMap[s])
        noCloneDescriptor.extend(DescriptorMap[s])

    # Setting up market intercept if relevant
    interceptFactor = 'Market Intercept'
    intercept = ModelFactor(interceptFactor, interceptFactor)
    industryClassification = Classification.GICSIndustries(datetime.date(2016,9,1))

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.USAxioma2016MH')
        # Set important model parameters
        ModelParameters2017.defaultModelSettingsUS4(self)
        self.twoRegressionStructure = True

        # Set up relevant styles to be created/used
        ModelParameters2017.defaultExposureParameters(self, self.styleList)
        self.styles = [s for s in self.totalStyles if s.name in self.styleList]
        EquityModel.FundamentalModel.__init__(self, ['CUSIP'], modelDB, marketDB)

        # Set up eligible universe parameters
        self.elig_parameters = {'assetTypes': self.commonStockTypes + ['REIT'],
                                'excludeTypes': None,
                                'HomeCountry_List': ['US'],
                                'use_isin_country_Flag': True}

        # Set up regression parameters
        dummyThreshold = 10
        self.internalCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType='Industry Groups',
                dummyThreshold=dummyThreshold,
                marketRegression=False,
                kappa=5.0,
                useRealMCaps=True,
                regWeight='rootCap',
                )
        self.returnCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType='Industry Groups',
                dummyThreshold=dummyThreshold,
                marketRegression=False,
                kappa=25.0,
                useRealMCaps=True,
                regWeight='rootCap',
                )

        # This controls the FMP regression
        self.fmpCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType='Industry Groups',
                marketRegression=False,
                dummyThreshold=dummyThreshold,
                useRealMCaps=True,
                kappa=None)

        # Set up risk parameters
        ModelParameters2017.defaultFundamentalCovarianceParameters(
                self, nwLag=2, varDVAOnly=False, unboundedDVA=False)

        # Set up standardization parameters
        gloScope = Standardization.GlobalRelativeScope([f.name for f in self.styles])
        self.exposureStandardization = Standardization.BucketizedStandardization(\
                [gloScope],fillWithZeroList=self.fillWithZeroList)

        # Set up descriptor standardization parameters
        descriptors = sorted(list(set([item for sublist in self.DescriptorMap.values() for item in sublist])))
        self.descriptorStandardization = Standardization.BucketizedStandardization(
                [Standardization.GlobalRelativeScope(descriptors)],
                mad_bound=15.0, fancyMAD=self.fancyMAD, exceptionNames=self.noStndDescriptor)

        self.indexSelector = MarketIndex.MarketIndexSelector(modelDB, marketDB)
        modelDB.createCurrencyCache(marketDB)

    def generate_estimation_universe(self, modelDate, data, modelDB, marketDB,
            excludeFactors=None, grandfatherRMS_ID=None):
        """Estimation universe selection criteria for AX-US.
        """
        self.log.debug('generate_estimation_universe: begin')

        buildEstu = EstimationUniverse.ConstructEstimationUniverse(
                data.exposureMatrix.getAssets(), self, modelDB, marketDB)

        # Set up various eligible and total universes
        universeIdx = list(range(len(buildEstu.assets)))
        if not hasattr(data, 'eligibleUniverse'):
            eligibleUniverse = list(buildEstu.assets)
            eligibleUniverseIdx = list(universeIdx)
        else:
            eligibleUniverse = list(data.eligibleUniverse)
            eligibleUniverseIdx = [data.assetIdxMap[sid] for sid in eligibleUniverse]
        logging.info('Estimation universe currently stands at %d stocks', len(eligibleUniverse))

        # Keep only issues trading on NYSE and NASDAQ
        fade_in_date = datetime.date(2009,1,1)
        fade_out_date = datetime.date(2010,1,1)
        logging.info('Excluding assets not on NAS and NYS')
        (eligibleExchangeIdx, nonest) = buildEstu.exclude_by_market_type(
                modelDate, data, includeFields=['NAS','NYS'], excludeFields=None,
                baseEstu=eligibleUniverseIdx)
        exchangeDownWeight = numpy.zeros((len(universeIdx)), float)
        numpy.put(exchangeDownWeight, eligibleExchangeIdx, 1.0)
        delta = 0.25
        if modelDate < fade_in_date:
            eligibleExchangeIdx = list(eligibleUniverseIdx)
        elif modelDate > fade_out_date:
            delta = 0.0
        else:
            eligibleExchangeIdx = list(eligibleUniverseIdx)
            delta = delta * float((fade_out_date - modelDate).days) / float((fade_out_date - fade_in_date).days)
        if delta > 0.0:
            logging.info('Downweighting factor for pre-%d OTCs: %.3f', fade_out_date.year, delta)
        numpy.put(exchangeDownWeight, nonest, delta)
        eligibleExchange = [buildEstu.assets[idx] for idx in eligibleExchangeIdx]
        logging.info('Estimation universe currently stands at %d stocks', len(eligibleExchangeIdx))

        # Report on thinly-traded assets over the entire universe
        logging.info('Looking for sparsely traded stocks')
        (nonSparse, sparse) = buildEstu.exclude_thinly_traded_assets(
                                modelDate, data, minNonZero=0.5,
                                baseEstu=eligibleUniverseIdx, legacy=False)
        nonSparse = [buildEstu.assets[idx] for idx in nonSparse]

        # Exclude thinly traded assets if required
        estuIdx = [data.assetIdxMap[sid] for sid in eligibleExchange if sid in nonSparse]
        exchangeDownWeightNS = numpy.take(exchangeDownWeight, estuIdx, axis=0)
        logging.info('Estimation universe currently stands at %d stocks', len(estuIdx))

        # Rank stuff by market cap and total volume over past year
        logging.info('Excluding assets based on cap and volume')
        (estuIdx, nonest) = buildEstu.filter_by_cap_and_volume(
                data, modelDate, baseEstu=estuIdx, 
                downWeights=exchangeDownWeightNS)
        logging.info('Estimation universe currently stands at %d stocks', len(estuIdx))

        # Exclude assets from estuIdx with mcaps < .8 mcap quanitle
        idxAssetMap = {v: k for k,v in data.assetIdxMap.items()}
        mcap = pandas.Series(data.marketCaps, index=data.universe)
        estuMcap = mcap.reindex(index=[idxAssetMap[a] for a in estuIdx])
        maxMcap = estuMcap[estuMcap<estuMcap.quantile(.8)].max()
        estuAssets = estuMcap[estuMcap<estuMcap.quantile(.8)].index
        estuIdx = [data.assetIdxMap[a] for a in estuAssets]

        eligMcap = mcap.reindex(index=[idxAssetMap[a] for a in eligibleExchangeIdx])
        eligAssets = eligMcap[eligMcap<=maxMcap].index
        eligibleExchangeIdx = [data.assetIdxMap[a] for a in eligAssets]

        # Inflate thin industry factors if possible
        logging.info('Inflating any thin factors')
        minFactorWidth=2*self.returnCalculator.allParameters[0].dummyThreshold
        (estuIdx, nonest) = buildEstu.pump_up_factors(
                data, modelDate, currentEstu=estuIdx, baseEstu=eligibleExchangeIdx,
                minFactorWidth=minFactorWidth, downWeights=exchangeDownWeight)
        logging.info('Estimation universe currently stands at %d stocks', len(estuIdx))

        # Apply grandfathering rules
        logging.info('Applying grandfathering')
        (estuIdx, ESTUQualify, nonest) = buildEstu.grandfather(
                modelDate, estuIdx, baseEstu=eligibleExchangeIdx, estuInstance=self.estuMap['main'],
                grandfatherRMS_ID=grandfatherRMS_ID)
        logging.info('Estimation universe currently stands at %d stocks', len(estuIdx))

        # Drop assets that are more than 1.5* the max mcap on the modelDate
        estuMcap = mcap.reindex(index=[idxAssetMap[a] for a in estuIdx])
        estuAssets = estuMcap[estuMcap<=1.5*maxMcap].index
        estuIdx = [data.assetIdxMap[a] for a in estuAssets]

        self.estuMap['main'].assets = [buildEstu.assets[idx] for idx in estuIdx]
        self.estuMap['main'].qualify = [buildEstu.assets[idx] for idx in ESTUQualify]

        self.log.debug('generate_estimation_universe: end')
        return estuIdx

class JackUSResearchModel(RiskModels.USAxioma2016MH):
    """US4 fundamental medium-horizon small cap research model.
    """
    rm_id = -18
    revision = 10
    rms_id = -190

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
                 #'MidCap',
                 'Exchange Rate Sensitivity',
                ]

    DescriptorMap = {
            'Earnings Yield': ['Earnings_to_Price_Quarterly', 'Est_Earnings_to_Price_12MFL_Quarterly'],
            'Value': ['Book_to_Price_Quarterly'],
            'Leverage': ['Debt_to_Assets_Quarterly', 'Debt_to_Equity_Quarterly'],
            'Growth': ['Earnings_Growth_RPF_AFQ', 'Sales_Growth_RPF_AFQ'],
            'Dividend Yield': ['Dividend_Yield_Quarterly'],
            'Size': ['LnIssuerCap'],
            #'Liquidity': ['LnTrading_Activity_60D'],
            'Liquidity': ['LnTrading_Activity_60D', 'Amihud_Liquidity_125D', 'ISC_Ret_Score'],
            'Market Sensitivity': ['Market_Sensitivity_250D'],
            'Volatility': ['Volatility_125D'],
            'Medium-Term Momentum': ['Momentum_250x20D'],
            'Exchange Rate Sensitivity': ['XRate_104W_XDR'],
            'Profitability': ['Return_on_Equity_Quarterly', 'Return_on_Assets_Quarterly',
                              'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
                              'Sales_to_Assets_Quarterly', 'Gross_Margin_Quarterly'],
            }

    DescriptorWeights = {'Earnings Yield': [0.75, 0.25]}
    #smallCapMap = {'MidCap': [66.67, 86.67],}
    noProxyList = ['Dividend Yield']
    fillMissingList = ['Earnings Yield', 'Value', 'Leverage', 'Growth', 'Profitability']
    fillWithZeroList = ['Dividend Yield']
    shrinkList = {'Liquidity': 60,
                  'Market Sensitivity': 250,
                  'Volatility': 125,
                  'Medium-Term Momentum': 250}
    orthogList = {'Volatility': [['Market Sensitivity'], True, 1.0]}
    regionalStndList = []

    # descriptor settings
    noStndDescriptor = []
    noCloneDescriptor = []
    for s in noProxyList:
        noStndDescriptor.extend(DescriptorMap[s])
        noCloneDescriptor.extend(DescriptorMap[s])

    # Setting up market intercept if relevant
    interceptFactor = 'Market Intercept'
    intercept = ModelFactor(interceptFactor, interceptFactor)
    industryClassification = Classification.GICSIndustries(datetime.date(2016,9,1))

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.USAxioma2016MH')
        # Set important model parameters
        ModelParameters2017.defaultModelSettingsUS4(self)
        self.twoRegressionStructure = True

        # Set up relevant styles to be created/used
        ModelParameters2017.defaultExposureParameters(self, self.styleList)
        self.styles = [s for s in self.totalStyles if s.name in self.styleList]
        EquityModel.FundamentalModel.__init__(self, ['CUSIP'], modelDB, marketDB)

        # Set up eligible universe parameters
        self.elig_parameters = {'assetTypes': self.commonStockTypes + ['REIT'],
                                'excludeTypes': None,
                                'HomeCountry_List': ['US'],
                                'use_isin_country_Flag': True}

        # Set up regression parameters
        dummyThreshold = 10
        self.internalCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType='Industry Groups',
                dummyThreshold=dummyThreshold,
                marketRegression=False,
                kappa=5.0,
                useRealMCaps=True,
                regWeight='rootCap',
                )
        self.returnCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType='Industry Groups',
                dummyThreshold=dummyThreshold,
                marketRegression=False,
                kappa=25.0,
                useRealMCaps=True,
                regWeight='rootCap',
                )

        # This controls the FMP regression
        self.fmpCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType='Industry Groups',
                marketRegression=False,
                dummyThreshold=dummyThreshold,
                useRealMCaps=True,
                kappa=None)

        # Set up risk parameters
        ModelParameters2017.defaultFundamentalCovarianceParameters(
                self, nwLag=2, varDVAOnly=False, unboundedDVA=False)

        # Set up standardization parameters
        gloScope = Standardization.GlobalRelativeScope([f.name for f in self.styles])
        self.exposureStandardization = Standardization.BucketizedStandardization(\
                [gloScope],fillWithZeroList=self.fillWithZeroList)

        # Set up descriptor standardization parameters
        descriptors = sorted(list(set([item for sublist in self.DescriptorMap.values() for item in sublist])))
        self.descriptorStandardization = Standardization.BucketizedStandardization(
                [Standardization.GlobalRelativeScope(descriptors)],
                mad_bound=15.0, fancyMAD=self.fancyMAD, exceptionNames=self.noStndDescriptor)

        self.indexSelector = MarketIndex.MarketIndexSelector(modelDB, marketDB)
        modelDB.createCurrencyCache(marketDB)

    def generate_estimation_universe(self, modelDate, data, modelDB, marketDB,
            excludeFactors=None, grandfatherRMS_ID=None):
        """Estimation universe selection criteria for AX-US.
        """
        self.log.debug('generate_estimation_universe: begin')

        buildEstu = EstimationUniverse.ConstructEstimationUniverse(
                data.exposureMatrix.getAssets(), self, modelDB, marketDB)

        # Set up various eligible and total universes
        universeIdx = list(range(len(buildEstu.assets)))
        if not hasattr(data, 'eligibleUniverse'):
            eligibleUniverse = list(buildEstu.assets)
            eligibleUniverseIdx = list(universeIdx)
        else:
            eligibleUniverse = list(data.eligibleUniverse)
            eligibleUniverseIdx = [data.assetIdxMap[sid] for sid in eligibleUniverse]
        logging.info('Estimation universe currently stands at %d stocks', len(eligibleUniverse))

        # Keep only issues trading on NYSE and NASDAQ
        fade_in_date = datetime.date(2009,1,1)
        fade_out_date = datetime.date(2010,1,1)
        logging.info('Excluding assets not on NAS and NYS')
        (eligibleExchangeIdx, nonest) = buildEstu.exclude_by_market_type(
                modelDate, data, includeFields=['NAS','NYS'], excludeFields=None,
                baseEstu=eligibleUniverseIdx)
        exchangeDownWeight = numpy.zeros((len(universeIdx)), float)
        numpy.put(exchangeDownWeight, eligibleExchangeIdx, 1.0)
        delta = 0.25
        if modelDate < fade_in_date:
            eligibleExchangeIdx = list(eligibleUniverseIdx)
        elif modelDate > fade_out_date:
            delta = 0.0
        else:
            eligibleExchangeIdx = list(eligibleUniverseIdx)
            delta = delta * float((fade_out_date - modelDate).days) / float((fade_out_date - fade_in_date).days)
        if delta > 0.0:
            logging.info('Downweighting factor for pre-%d OTCs: %.3f', fade_out_date.year, delta)
        numpy.put(exchangeDownWeight, nonest, delta)
        eligibleExchange = [buildEstu.assets[idx] for idx in eligibleExchangeIdx]
        logging.info('Estimation universe currently stands at %d stocks', len(eligibleExchangeIdx))

        # Report on thinly-traded assets over the entire universe
        logging.info('Looking for sparsely traded stocks')
        (nonSparse, sparse) = buildEstu.exclude_thinly_traded_assets(
                                modelDate, data, minNonZero=0.5,
                                baseEstu=eligibleUniverseIdx, legacy=False)
        nonSparse = [buildEstu.assets[idx] for idx in nonSparse]

        # Exclude thinly traded assets if required
        estuIdx = [data.assetIdxMap[sid] for sid in eligibleExchange if sid in nonSparse]
        exchangeDownWeightNS = numpy.take(exchangeDownWeight, estuIdx, axis=0)
        logging.info('Estimation universe currently stands at %d stocks', len(estuIdx))

        # Rank stuff by market cap and total volume over past year
        logging.info('Excluding assets based on cap and volume')
        (estuIdx, nonest) = buildEstu.filter_by_cap_and_volume(
                data, modelDate, baseEstu=estuIdx, 
                downWeights=exchangeDownWeightNS)
        logging.info('Estimation universe currently stands at %d stocks', len(estuIdx))

        # Exclude assets from estuIdx with mcaps < .8 mcap quanitle
        idxAssetMap = {v: k for k,v in data.assetIdxMap.items()}
        mcap = pandas.Series(data.marketCaps, index=data.universe)
        estuMcap = mcap.reindex(index=[idxAssetMap[a] for a in estuIdx])
        maxMcap = estuMcap[estuMcap<estuMcap.quantile(.8)].max()
        estuAssets = estuMcap[estuMcap<estuMcap.quantile(.8)].index
        estuIdx = [data.assetIdxMap[a] for a in estuAssets]

        eligMcap = mcap.reindex(index=[idxAssetMap[a] for a in eligibleExchangeIdx])
        eligAssets = eligMcap[eligMcap<=maxMcap].index
        eligibleExchangeIdx = [data.assetIdxMap[a] for a in eligAssets]

        # Inflate thin industry factors if possible
        logging.info('Inflating any thin factors')
        minFactorWidth=2*self.returnCalculator.allParameters[0].dummyThreshold
        (estuIdx, nonest) = buildEstu.pump_up_factors(
                data, modelDate, currentEstu=estuIdx, baseEstu=eligibleExchangeIdx,
                minFactorWidth=minFactorWidth, downWeights=exchangeDownWeight)
        logging.info('Estimation universe currently stands at %d stocks', len(estuIdx))

        # Apply grandfathering rules
        logging.info('Applying grandfathering')
        (estuIdx, ESTUQualify, nonest) = buildEstu.grandfather(
                modelDate, estuIdx, baseEstu=eligibleExchangeIdx, estuInstance=self.estuMap['main'],
                grandfatherRMS_ID=grandfatherRMS_ID)
        logging.info('Estimation universe currently stands at %d stocks', len(estuIdx))

        # Drop assets that are more than 1.5* the max mcap on the modelDate
        estuMcap = mcap.reindex(index=[idxAssetMap[a] for a in estuIdx])
        estuAssets = estuMcap[estuMcap<=1.5*maxMcap].index
        estuIdx = [data.assetIdxMap[a] for a in estuAssets]

        self.estuMap['main'].assets = [buildEstu.assets[idx] for idx in estuIdx]
        self.estuMap['main'].qualify = [buildEstu.assets[idx] for idx in ESTUQualify]

        self.log.debug('generate_estimation_universe: end')
        return estuIdx






class KipUSResearchModel(EquityModel.FundamentalModel):
    """US3 fundamental medium-horizon research model.
    """
    rm_id = -18
    revision = 11
    rms_id = -191

    legacyMCapDates = True
    regionalDescriptorStructure = True
    twoRegressionStructure = False
    fancyMAD = False
    firstReturnDate = datetime.date(1980,1,1)
    # Back-compatibility for fill-in rules
    gicsDate = datetime.date(2014,3,1)

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
                 'MidCap',
                 'Exchange Rate Sensitivity',
                 'Term Spread Sensitivity'
                ]

    DescriptorMap = {
            'Earnings Yield': ['Earnings_to_Price_Quarterly', 'Est_Earnings_to_Price_12MFL_Quarterly'],
            'Value': ['Book_to_Price_Quarterly'],
            'Leverage': ['Debt_to_Assets_Quarterly', 'Debt_to_Equity_Quarterly'],
            'Growth': ['Earnings_Growth_RPF_AFQ', 'Sales_Growth_RPF_AFQ'],
            'Dividend Yield': ['Dividend_Yield_Quarterly'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_60D'],
            'Market Sensitivity': ['Market_Sensitivity_250D'],
            'Volatility': ['Volatility_125D'],
            'Medium-Term Momentum': ['Momentum_250x20D'],
            'Exchange Rate Sensitivity': ['XRate_104W_XDR'],
            'Profitability': ['Return_on_Equity_Quarterly', 'Return_on_Assets_Quarterly',
                              'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
                              'Sales_to_Assets_Quarterly', 'Gross_Margin_Quarterly'],
            'Term Spread Sensitivity': ['TermSpread_Sensitivity_5']
            }

    DescriptorWeights = {'Earnings Yield': [0.75, 0.25]}
    smallCapMap = {'MidCap': [66.67, 86.67],}
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
    industryClassification = Classification.GICSIndustries(datetime.date(2014,3,1))

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.USAxioma2016MH')
        # Set up relevant styles to be created/used
        ModelParameters2017.defaultExposureParameters(self, self.styleList)
        self.styles = [s for s in self.totalStyles if s.name in self.styleList]
        EquityModel.FundamentalModel.__init__(self, ['CUSIP'], modelDB, marketDB)

        # Set up eligible universe parameters
        self.elig_parameters = {'assetTypes': self.commonStockTypes + ['REIT'],
                                'excludeTypes': None,
                                'HomeCountry_List': ['US'],
                                'use_isin_country_Flag': True}

        # Set up regression parameters
        dummyThreshold = 10
        self.returnCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType='Industry Groups',
                dummyThreshold=dummyThreshold,
                marketRegression=False,
                kappa=5.0,
                useRealMCaps=True,
                regWeight='rootCap',
                )

        # This controls the FMP regression
        self.fmpCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType='Industry Groups',
                marketRegression=False,
                dummyThreshold=dummyThreshold,
                useRealMCaps=True,
                kappa=None)

        # Set up risk parameters
        ModelParameters2017.defaultFundamentalCovarianceParameters(
                self, nwLag=2, varDVAOnly=False, unboundedDVA=False)

        # Set up standardization parameters
        gloScope = Standardization.GlobalRelativeScope([f.name for f in self.styles])
        self.exposureStandardization = Standardization.BucketizedStandardization(\
                [gloScope],fillWithZeroList=self.fillWithZeroList)

        # Set up descriptor standardization parameters
        descriptors = sorted(list(set([item for sublist in self.DescriptorMap.values() for item in sublist])))
        exceptionNames = [self.DescriptorMap[sf] for sf in self.noProxyList]
        exceptionNames = list(itertools.chain.from_iterable(exceptionNames))
        self.descriptorStandardization = Standardization.BucketizedStandardization(
                [Standardization.GlobalRelativeScope(descriptors)],
                mad_bound=15.0, fancyMAD=self.fancyMAD, exceptionNames=exceptionNames)

        self.indexSelector = MarketIndex.MarketIndexSelector(modelDB, marketDB)
        modelDB.createCurrencyCache(marketDB)

    def generate_estimation_universe(self, modelDate, data, modelDB, marketDB, excludeFactors=None):
        """Estimation universe selection criteria for AX-US.
        """
        self.log.debug('generate_estimation_universe: begin')

        buildEstu = EstimationUniverse.ConstructEstimationUniverse(
                data.exposureMatrix.getAssets(), self, modelDB, marketDB)

        # Set up various eligible and total universes
        universeIdx = list(range(len(buildEstu.assets)))
        if not hasattr(data, 'eligibleUniverse'):
            eligibleUniverse = list(buildEstu.assets)
            eligibleUniverseIdx = list(universeIdx)
        else:
            eligibleUniverse = list(data.eligibleUniverse)
            eligibleUniverseIdx = [data.assetIdxMap[sid] for sid in eligibleUniverse]
        logging.info('Estimation universe currently stands at %d stocks', len(eligibleUniverse))

        # Keep only issues trading on NYSE and NASDAQ
        fade_in_date = datetime.date(2009,1,1)
        fade_out_date = datetime.date(2010,1,1)
        logging.info('Excluding assets not on NAS and NYS')
        (eligibleExchangeIdx, nonest) = buildEstu.exclude_by_market_type(
                modelDate, data, includeFields=['NAS','NYS'], excludeFields=None,
                baseEstu=eligibleUniverseIdx)
        exchangeDownWeight = numpy.zeros((len(universeIdx)), float)
        numpy.put(exchangeDownWeight, eligibleExchangeIdx, 1.0)
        delta = 0.25
        if modelDate < fade_in_date:
            eligibleExchangeIdx = list(eligibleUniverseIdx)
        elif modelDate > fade_out_date:
            delta = 0.0
        else:
            eligibleExchangeIdx = list(eligibleUniverseIdx)
            delta = delta * float((fade_out_date - modelDate).days) / float((fade_out_date - fade_in_date).days)
        if delta > 0.0:
            logging.info('Downweighting factor for pre-%d OTCs: %.3f', fade_out_date.year, delta)
        numpy.put(exchangeDownWeight, nonest, delta)
        eligibleExchange = [buildEstu.assets[idx] for idx in eligibleExchangeIdx]
        logging.info('Estimation universe currently stands at %d stocks', len(eligibleExchangeIdx))

        # Report on thinly-traded assets over the entire universe
        logging.info('Looking for sparsely traded stocks')
        (nonSparse, sparse) = buildEstu.exclude_thinly_traded_assets(
                                modelDate, data, minNonZero=0.5,
                                baseEstu=eligibleUniverseIdx, legacy=False)
        nonSparse = [buildEstu.assets[idx] for idx in nonSparse]

        # Exclude thinly traded assets if required
        estuIdx = [data.assetIdxMap[sid] for sid in eligibleExchange if sid in nonSparse]
        exchangeDownWeightNS = numpy.take(exchangeDownWeight, estuIdx, axis=0)
        logging.info('Estimation universe currently stands at %d stocks', len(estuIdx))

        # Rank stuff by market cap and total volume over past year
        logging.info('Excluding assets based on cap and volume')
        (estuIdx, nonest) = buildEstu.filter_by_cap_and_volume(
                data, modelDate, baseEstu=estuIdx, downWeights=exchangeDownWeightNS)
        logging.info('Estimation universe currently stands at %d stocks', len(estuIdx))

        # Inflate thin industry factors if possible
        logging.info('Inflating any thin factors')
        minFactorWidth=2*self.returnCalculator.allParameters[0].getThinFactorInformation().dummyThreshold
        (estuIdx, nonest) = buildEstu.pump_up_factors(
                data, modelDate, currentEstu=estuIdx, baseEstu=eligibleExchangeIdx,
                minFactorWidth=minFactorWidth, downWeights=exchangeDownWeight)
        logging.info('Estimation universe currently stands at %d stocks', len(estuIdx))

        # Apply grandfathering rules
        logging.info('Applying grandfathering')
        (estuIdx, ESTUQualify, nonest) = buildEstu.grandfather(
                modelDate, estuIdx, baseEstu=eligibleExchangeIdx, estuInstance=self.estuMap['main'])
        logging.info('Estimation universe currently stands at %d stocks', len(estuIdx))

        self.estuMap['main'].assets = [buildEstu.assets[idx] for idx in estuIdx]
        self.estuMap['main'].qualify = [buildEstu.assets[idx] for idx in ESTUQualify]

        self.log.debug('generate_estimation_universe: end')
        return estuIdx

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
        if self.exposureStandardization.exceptionNames is not None:
            self.exposureStandardization.exceptionNames = list(set(self.exposureStandardization.exceptionNames))

        return data.exposureMatrix

class NitinUSResearchModel(MFM.FundamentalModel):
    """US3 fundamental medium-horizon research model.
    """
    rm_id = -18
    revision = 12
    rms_id = -192
    newExposureFormat = True
    standardizationStats = True
    globalDescriptorModel = True
    DLCEnabled = True
    SCM = True
    #useLogRets = True

    # List of style factors in the model
    styleList = ['Earnings Yield',
                 'Value',
                 'Leverage',
                 'Growth',
                 'Return-on-Equity',
                 'Dividend Yield',
                 'Size',
                 'Liquidity',
                 'Market Sensitivity',
                 'Volatility',
                 'Medium-Term Momentum',
                 'Exchange Rate Sensitivity',
            ]

    DescriptorMap = {
            'Earnings Yield': ['Earnings_to_Price', 'Est_Earnings_to_PriceV2'],
            'Value': ['Book_to_Price'],
            'Leverage': ['Debt_to_Assets'],
            'Growth': ['Earnings_Growth_RPF_Annual', 'Sales_Growth_RPF_Annual'],
            'Return-on-Equity': ['Return_on_Equity_Quarterly'],
            'Dividend Yield': ['Dividend_Yield'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_60D'],
            'Market Sensitivity': ['Market_Sensitivity_250D'],
            'Volatility': ['Volatility_125D'],
            'Medium-Term Momentum': ['Momentum_250x20D'],
            'Exchange Rate Sensitivity': ['XRate_104W_XDR'],
            }

    DescriptorWeights = {
            'Earnings Yield': [0.1, 0.9],
            }

    # Setting up market intercept if relevant
    interceptFactor = 'Market Intercept'
    intercept = ModelFactor(interceptFactor, interceptFactor)
    industryClassification = Classification.GICSIndustries(datetime.date(2008,8,30))
    quarterlyFundamentalData = True
    proxyDividendPayout = False
    fancyMAD = False
    legacyEstGrowth = False

    def __init__(self, modelDB, marketDB, expTreat=None):
        self.log = logging.getLogger('RiskModels.DieterUSResearchModel')
        # Set up relevant styles to be created/used
        ModelParameters.defaultExposureParameters(self, self.styleList)
        self.styles = [s for s in self.totalStyles if s.name in self.styleList]
        MFM.FundamentalModel.__init__(self, ['SEDOL'], modelDB, marketDB)

        # Model-specific exposure parameter stuff here
        dummyThreshold = 10
        constrainedReg = True
        nestedMktReg = False

        self.styleParameters['Volatility'].orthogCoef = 1.0
        self.styleParameters['Volatility'].sqrtWt = True
        self.styleParameters['Dividend Yield'].includeSpecial = False
        self.styleParameters['Liquidity'].legacy = False

        # Set up regression parameters
        ModelParameters.defaultRegressionParametersLegacy(
                self, modelDB,
                dummyType='Industry Groups',
                marketReg=nestedMktReg,
                constrainedReg=constrainedReg,
                scndRegs=False,
                #scndRegs=[[ModelFactor('SmallCap 1')]], # Here as an example
                #scndRegEstus=['SmallCap 1'],
                k_rlm=[5.0],
                #k_rlm=None,
                #k_rlm=[None, None],
                regWeight=['rootCap'],
                dummyThreshold=dummyThreshold)

        # Set up risk parameters
        ModelParameters.defaultFundamentalCovarianceParameters(
                self, nwLag=2, dva='spline',
                varDVAOnly=False, selectiveDeMean=True)

        # Set up standardization parameters
        gloScope = Standardization.GlobalRelativeScope([f.name for f in self.styles])
        self.exposureStandardization = Standardization.BucketizedStandardization([gloScope])

        self.indexSelector = MarketIndex.MarketIndexSelector(modelDB, marketDB)
        modelDB.createCurrencyCache(marketDB)

    def generate_eligible_universe(self, modelDate, data, modelDB, marketDB,
            buildEstu=None, assetTypes=None):
        """Creates subset of eligible assets for consideration
        in US estimation universes
        """
        self.log.debug('generate_eligible_universe: begin')

        if buildEstu is None:
            buildEstu = EstimationUniverse.ConstructEstimationUniverse(
                data.exposureMatrix.getAssets(), self, modelDB, marketDB)

        universe = buildEstu.assets

        # Remove assets from the exclusion table (BRK-A/B, etc.)
        (estuIdx, nonest) = buildEstu.apply_exclusion_list(modelDate)

        # Remove ADRs and foreign listings
        (estuIdx, nonest) = buildEstu.exclude_by_market_classification(
                modelDate, 'HomeCountry', 'REGIONS', ['US'], baseEstu=estuIdx)

        # Remove everything except common stocks and REITs
        (estuIdx, nonest) = buildEstu.exclude_by_asset_type(
                modelDate, data,
                includeFields=['All-Com', 'REIT'],
                excludeFields=['ComWI'],
                baseEstu = estuIdx)

        # Weed out foreign issuers by ISIN country prefix
        (estuIdx, nonest)  = buildEstu.exclude_by_isin_country(['US'],
                modelDate, baseEstu=estuIdx)

        estu = [universe[idx] for idx in estuIdx]
        logging.info('%d eligible US assets out of %d total',
                        len(estu), len(universe))
        return estu

    def generate_estimation_universe(self, modelDate, data,
            modelDB, marketDB, excludeFactors=[]):
        """Estimation universe selection criteria for AX-US.
        """
        self.log.debug('generate_estimation_universe: begin')

        buildEstu = EstimationUniverse.ConstructEstimationUniverse(
                data.exposureMatrix.getAssets(), self, modelDB, marketDB)

        # Set up various eligible and total universes
        universeIdx = list(range(len(buildEstu.assets)))
        if not hasattr(data, 'eligibleUniverse'):
            eligibleUniverse = list(buildEstu.assets)
            eligibleUniverseIdx = list(universeIdx)
        else:
            eligibleUniverse = list(data.eligibleUniverse)
            eligibleUniverseIdx = [data.assetIdxMap[sid] for sid in eligibleUniverse]

        # Keep only issues trading on NYSE and NASDAQ
        if modelDate.year >= 2008:
            (eligibleExchangeIdx, nonest) = buildEstu.exclude_by_market_classification(
                    modelDate, 'Market', 'REGIONS', ['NAS','NYS'], baseEstu=eligibleUniverseIdx)
        else:
            eligibleExchangeIdx = list(eligibleUniverseIdx)
        eligibleExchange = [buildEstu.assets[idx] for idx in eligibleExchangeIdx]

        # Report on thinly-traded assets over the entire universe
        (nonSparse, sparse) = buildEstu.exclude_thinly_traded_assets(
                                modelDate, data, baseEstu=universeIdx)
        nonSparse = [buildEstu.assets[idx] for idx in nonSparse]

        # Exclude thinly traded assets if required
        estuIdx = [data.assetIdxMap[sid] for sid in eligibleExchange if sid in nonSparse]

        # Rank stuff by market cap and total volume over past year
        (estuIdx, nonest) = buildEstu.filter_by_cap_and_volume(
                data, modelDate, baseEstu=estuIdx)

        # Inflate thin industry factors if possible
        (estuIdx, nonest) = buildEstu.pump_up_factors(
                data, modelDate, currentEstu=estuIdx, baseEstu=eligibleExchangeIdx,
                minFactorWidth=2*self.returnCalculator.\
                        allParameters[0].getThinFactorInformation().dummyThreshold)

        # Apply grandfathering rules
        (estuIdx, ESTUQualify, nonest) = buildEstu.grandfather(
                modelDate, estuIdx, baseEstu=eligibleExchangeIdx, estuInstance=self.estuMap['main'])

        self.estuMap['main'].assets = [buildEstu.assets[idx] for idx in estuIdx]
        self.estuMap['main'].qualify = [buildEstu.assets[idx] for idx in ESTUQualify]

        self.log.debug('generate_estimation_universe: end')
        return estuIdx

    def generate_model_specific_exposures(self, modelDate, data, modelDB, marketDB):
        """Generate the non-default factors.
        """
        return data.exposureMatrix

    def shrink_to_mean(self, modelDate, data, modelDB, marketDB,
            descriptorName, historyLength, values):
        return values

class NathanUSResearchModel(EquityModel.FundamentalModel):
    """US3 fundamental medium-horizon research model.
    """
    rm_id = -18
    revision = 13
    rms_id = -193

    legacyMCapDates = True
    regionalDescriptorStructure = True
    twoRegressionStructure = False
    fancyMAD = False
    firstReturnDate = datetime.date(1980,1,1)
    # Back-compatibility for fill-in rules
    gicsDate = datetime.date(2014,3,1)

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
                 'MidCap',
                 'Exchange Rate Sensitivity',
                 'Term Spread Sensitivity'
                ]

    DescriptorMap = {
            'Earnings Yield': ['Earnings_to_Price_Quarterly', 'Est_Earnings_to_Price_12MFL_Quarterly'],
            'Value': ['Book_to_Price_Quarterly'],
            'Leverage': ['Debt_to_Assets_Quarterly', 'Debt_to_Equity_Quarterly'],
            'Growth': ['Earnings_Growth_RPF_AFQ', 'Sales_Growth_RPF_AFQ'],
            'Dividend Yield': ['Dividend_Yield_Quarterly'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_60D'],
            'Market Sensitivity': ['Market_Sensitivity_250D'],
            'Volatility': ['Volatility_125D'],
            'Medium-Term Momentum': ['Momentum_250x20D'],
            'Exchange Rate Sensitivity': ['XRate_104W_XDR'],
            'Profitability': ['Return_on_Equity_Quarterly', 'Return_on_Assets_Quarterly',
                              'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
                              'Sales_to_Assets_Quarterly', 'Gross_Margin_Quarterly'],
            'Term Spread Sensitivity': ['TermSpread_Sensitivity']
            }

    DescriptorWeights = {'Earnings Yield': [0.75, 0.25]}
    smallCapMap = {'MidCap': [66.67, 86.67],}
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
    industryClassification = Classification.GICSIndustries(datetime.date(2014,3,1))

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.USAxioma2016MH')
        # Set up relevant styles to be created/used
        ModelParameters2017.defaultExposureParameters(self, self.styleList)
        self.styles = [s for s in self.totalStyles if s.name in self.styleList]
        EquityModel.FundamentalModel.__init__(self, ['CUSIP'], modelDB, marketDB)

        # Set up eligible universe parameters
        self.elig_parameters = {'assetTypes': self.commonStockTypes + ['REIT'],
                                'excludeTypes': None,
                                'HomeCountry_List': ['US'],
                                'use_isin_country_Flag': True}

        # Set up regression parameters
        dummyThreshold = 10
        self.returnCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType='Industry Groups',
                dummyThreshold=dummyThreshold,
                marketRegression=False,
                kappa=5.0,
                useRealMCaps=True,
                regWeight='rootCap',
                )

        # This controls the FMP regression
        self.fmpCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType='Industry Groups',
                marketRegression=False,
                dummyThreshold=dummyThreshold,
                useRealMCaps=True,
                kappa=None)

        # Set up risk parameters
        ModelParameters2017.defaultFundamentalCovarianceParameters(
                self, nwLag=2, varDVAOnly=False, unboundedDVA=False)

        # Set up standardization parameters
        gloScope = Standardization.GlobalRelativeScope([f.name for f in self.styles])
        self.exposureStandardization = Standardization.BucketizedStandardization(\
                [gloScope],fillWithZeroList=self.fillWithZeroList)

        # Set up descriptor standardization parameters
        descriptors = sorted(list(set([item for sublist in self.DescriptorMap.values() for item in sublist])))
        exceptionNames = [self.DescriptorMap[sf] for sf in self.noProxyList]
        exceptionNames = list(itertools.chain.from_iterable(exceptionNames))
        self.descriptorStandardization = Standardization.BucketizedStandardization(
                [Standardization.GlobalRelativeScope(descriptors)],
                mad_bound=15.0, fancyMAD=self.fancyMAD, exceptionNames=exceptionNames)

        self.indexSelector = MarketIndex.MarketIndexSelector(modelDB, marketDB)
        modelDB.createCurrencyCache(marketDB)

    def generate_estimation_universe(self, modelDate, data, modelDB, marketDB, excludeFactors=None):
        """Estimation universe selection criteria for AX-US.
        """
        self.log.debug('generate_estimation_universe: begin')

        buildEstu = EstimationUniverse.ConstructEstimationUniverse(
                data.exposureMatrix.getAssets(), self, modelDB, marketDB)

        # Set up various eligible and total universes
        universeIdx = list(range(len(buildEstu.assets)))
        if not hasattr(data, 'eligibleUniverse'):
            eligibleUniverse = list(buildEstu.assets)
            eligibleUniverseIdx = list(universeIdx)
        else:
            eligibleUniverse = list(data.eligibleUniverse)
            eligibleUniverseIdx = [data.assetIdxMap[sid] for sid in eligibleUniverse]
        logging.info('Estimation universe currently stands at %d stocks', len(eligibleUniverse))

        # Keep only issues trading on NYSE and NASDAQ
        fade_in_date = datetime.date(2009,1,1)
        fade_out_date = datetime.date(2010,1,1)
        logging.info('Excluding assets not on NAS and NYS')
        (eligibleExchangeIdx, nonest) = buildEstu.exclude_by_market_type(
                modelDate, data, includeFields=['NAS','NYS'], excludeFields=None,
                baseEstu=eligibleUniverseIdx)
        exchangeDownWeight = numpy.zeros((len(universeIdx)), float)
        numpy.put(exchangeDownWeight, eligibleExchangeIdx, 1.0)
        delta = 0.25
        if modelDate < fade_in_date:
            eligibleExchangeIdx = list(eligibleUniverseIdx)
        elif modelDate > fade_out_date:
            delta = 0.0
        else:
            eligibleExchangeIdx = list(eligibleUniverseIdx)
            delta = delta * float((fade_out_date - modelDate).days) / float((fade_out_date - fade_in_date).days)
        if delta > 0.0:
            logging.info('Downweighting factor for pre-%d OTCs: %.3f', fade_out_date.year, delta)
        numpy.put(exchangeDownWeight, nonest, delta)
        eligibleExchange = [buildEstu.assets[idx] for idx in eligibleExchangeIdx]
        logging.info('Estimation universe currently stands at %d stocks', len(eligibleExchangeIdx))

        # Report on thinly-traded assets over the entire universe
        logging.info('Looking for sparsely traded stocks')
        (nonSparse, sparse) = buildEstu.exclude_thinly_traded_assets(
                                modelDate, data, minNonZero=0.5,
                                baseEstu=eligibleUniverseIdx, legacy=False)
        nonSparse = [buildEstu.assets[idx] for idx in nonSparse]

        # Exclude thinly traded assets if required
        estuIdx = [data.assetIdxMap[sid] for sid in eligibleExchange if sid in nonSparse]
        exchangeDownWeightNS = numpy.take(exchangeDownWeight, estuIdx, axis=0)
        logging.info('Estimation universe currently stands at %d stocks', len(estuIdx))

        # Rank stuff by market cap and total volume over past year
        logging.info('Excluding assets based on cap and volume')
        (estuIdx, nonest) = buildEstu.filter_by_cap_and_volume(
                data, modelDate, baseEstu=estuIdx, downWeights=exchangeDownWeightNS)
        logging.info('Estimation universe currently stands at %d stocks', len(estuIdx))

        # Inflate thin industry factors if possible
        logging.info('Inflating any thin factors')
        minFactorWidth=2*self.returnCalculator.allParameters[0].getThinFactorInformation().dummyThreshold
        (estuIdx, nonest) = buildEstu.pump_up_factors(
                data, modelDate, currentEstu=estuIdx, baseEstu=eligibleExchangeIdx,
                minFactorWidth=minFactorWidth, downWeights=exchangeDownWeight)
        logging.info('Estimation universe currently stands at %d stocks', len(estuIdx))

        # Apply grandfathering rules
        logging.info('Applying grandfathering')
        (estuIdx, ESTUQualify, nonest) = buildEstu.grandfather(
                modelDate, estuIdx, baseEstu=eligibleExchangeIdx, estuInstance=self.estuMap['main'])
        logging.info('Estimation universe currently stands at %d stocks', len(estuIdx))

        self.estuMap['main'].assets = [buildEstu.assets[idx] for idx in estuIdx]
        self.estuMap['main'].qualify = [buildEstu.assets[idx] for idx in ESTUQualify]

        self.log.debug('generate_estimation_universe: end')
        return estuIdx

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
        if self.exposureStandardization.exceptionNames is not None:
            self.exposureStandardization.exceptionNames = list(set(self.exposureStandardization.exceptionNames))

        return data.exposureMatrix

class JasonUSResearchModel(EquityModel.FundamentalModel):
    """US3 fundamental medium-horizon research model.
    """
    rm_id = -18
    revision = 14
    rms_id = -194

    legacyMCapDates = True
    regionalDescriptorStructure = True
    twoRegressionStructure = False
    fancyMAD = False
    firstReturnDate = datetime.date(1980,1,1)
    # Back-compatibility for fill-in rules
    gicsDate = datetime.date(2014,3,1)

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
                 'MidCap',
                 'Exchange Rate Sensitivity',
                 'Term Spread Sensitivity'
                ]

    DescriptorMap = {
            'Earnings Yield': ['Earnings_to_Price_Quarterly', 'Est_Earnings_to_Price_12MFL_Quarterly'],
            'Value': ['Book_to_Price_Quarterly'],
            'Leverage': ['Debt_to_Assets_Quarterly', 'Debt_to_Equity_Quarterly'],
            'Growth': ['Earnings_Growth_RPF_AFQ', 'Sales_Growth_RPF_AFQ'],
            'Dividend Yield': ['Dividend_Yield_Quarterly'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_60D'],
            'Market Sensitivity': ['Market_Sensitivity_250D'],
            'Volatility': ['Volatility_125D'],
            'Medium-Term Momentum': ['Momentum_250x20D'],
            'Exchange Rate Sensitivity': ['XRate_104W_XDR'],
            'Profitability': ['Return_on_Equity_Quarterly', 'Return_on_Assets_Quarterly',
                              'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
                              'Sales_to_Assets_Quarterly', 'Gross_Margin_Quarterly'],
            'Term Spread Sensitivity': ['TermSpread_Sensitivity_3']
            }

    DescriptorWeights = {'Earnings Yield': [0.75, 0.25]}
    smallCapMap = {'MidCap': [66.67, 86.67],}
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
    industryClassification = Classification.GICSIndustries(datetime.date(2014,3,1))

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.USAxioma2016MH')
        # Set up relevant styles to be created/used
        ModelParameters2017.defaultExposureParameters(self, self.styleList)
        self.styles = [s for s in self.totalStyles if s.name in self.styleList]
        EquityModel.FundamentalModel.__init__(self, ['CUSIP'], modelDB, marketDB)

        # Set up eligible universe parameters
        self.elig_parameters = {'assetTypes': self.commonStockTypes + ['REIT'],
                                'excludeTypes': None,
                                'HomeCountry_List': ['US'],
                                'use_isin_country_Flag': True}

        # Set up regression parameters
        dummyThreshold = 10
        self.returnCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType='Industry Groups',
                dummyThreshold=dummyThreshold,
                marketRegression=False,
                kappa=5.0,
                useRealMCaps=True,
                regWeight='rootCap',
                )

        # This controls the FMP regression
        self.fmpCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType='Industry Groups',
                marketRegression=False,
                dummyThreshold=dummyThreshold,
                useRealMCaps=True,
                kappa=None)

        # Set up risk parameters
        ModelParameters2017.defaultFundamentalCovarianceParameters(
                self, nwLag=2, varDVAOnly=False, unboundedDVA=False)

        # Set up standardization parameters
        gloScope = Standardization.GlobalRelativeScope([f.name for f in self.styles])
        self.exposureStandardization = Standardization.BucketizedStandardization(\
                [gloScope],fillWithZeroList=self.fillWithZeroList)

        # Set up descriptor standardization parameters
        descriptors = sorted(list(set([item for sublist in self.DescriptorMap.values() for item in sublist])))
        exceptionNames = [self.DescriptorMap[sf] for sf in self.noProxyList]
        exceptionNames = list(itertools.chain.from_iterable(exceptionNames))
        self.descriptorStandardization = Standardization.BucketizedStandardization(
                [Standardization.GlobalRelativeScope(descriptors)],
                mad_bound=15.0, fancyMAD=self.fancyMAD, exceptionNames=exceptionNames)

        self.indexSelector = MarketIndex.MarketIndexSelector(modelDB, marketDB)
        modelDB.createCurrencyCache(marketDB)

    def generate_estimation_universe(self, modelDate, data, modelDB, marketDB, excludeFactors=None):
        """Estimation universe selection criteria for AX-US.
        """
        self.log.debug('generate_estimation_universe: begin')

        buildEstu = EstimationUniverse.ConstructEstimationUniverse(
                data.exposureMatrix.getAssets(), self, modelDB, marketDB)

        # Set up various eligible and total universes
        universeIdx = list(range(len(buildEstu.assets)))
        if not hasattr(data, 'eligibleUniverse'):
            eligibleUniverse = list(buildEstu.assets)
            eligibleUniverseIdx = list(universeIdx)
        else:
            eligibleUniverse = list(data.eligibleUniverse)
            eligibleUniverseIdx = [data.assetIdxMap[sid] for sid in eligibleUniverse]
        logging.info('Estimation universe currently stands at %d stocks', len(eligibleUniverse))

        # Keep only issues trading on NYSE and NASDAQ
        fade_in_date = datetime.date(2009,1,1)
        fade_out_date = datetime.date(2010,1,1)
        logging.info('Excluding assets not on NAS and NYS')
        (eligibleExchangeIdx, nonest) = buildEstu.exclude_by_market_type(
                modelDate, data, includeFields=['NAS','NYS'], excludeFields=None,
                baseEstu=eligibleUniverseIdx)
        exchangeDownWeight = numpy.zeros((len(universeIdx)), float)
        numpy.put(exchangeDownWeight, eligibleExchangeIdx, 1.0)
        delta = 0.25
        if modelDate < fade_in_date:
            eligibleExchangeIdx = list(eligibleUniverseIdx)
        elif modelDate > fade_out_date:
            delta = 0.0
        else:
            eligibleExchangeIdx = list(eligibleUniverseIdx)
            delta = delta * float((fade_out_date - modelDate).days) / float((fade_out_date - fade_in_date).days)
        if delta > 0.0:
            logging.info('Downweighting factor for pre-%d OTCs: %.3f', fade_out_date.year, delta)
        numpy.put(exchangeDownWeight, nonest, delta)
        eligibleExchange = [buildEstu.assets[idx] for idx in eligibleExchangeIdx]
        logging.info('Estimation universe currently stands at %d stocks', len(eligibleExchangeIdx))

        # Report on thinly-traded assets over the entire universe
        logging.info('Looking for sparsely traded stocks')
        (nonSparse, sparse) = buildEstu.exclude_thinly_traded_assets(
                                modelDate, data, minNonZero=0.5,
                                baseEstu=eligibleUniverseIdx, legacy=False)
        nonSparse = [buildEstu.assets[idx] for idx in nonSparse]

        # Exclude thinly traded assets if required
        estuIdx = [data.assetIdxMap[sid] for sid in eligibleExchange if sid in nonSparse]
        exchangeDownWeightNS = numpy.take(exchangeDownWeight, estuIdx, axis=0)
        logging.info('Estimation universe currently stands at %d stocks', len(estuIdx))

        # Rank stuff by market cap and total volume over past year
        logging.info('Excluding assets based on cap and volume')
        (estuIdx, nonest) = buildEstu.filter_by_cap_and_volume(
                data, modelDate, baseEstu=estuIdx, downWeights=exchangeDownWeightNS)
        logging.info('Estimation universe currently stands at %d stocks', len(estuIdx))

        # Inflate thin industry factors if possible
        logging.info('Inflating any thin factors')
        minFactorWidth=2*self.returnCalculator.allParameters[0].getThinFactorInformation().dummyThreshold
        (estuIdx, nonest) = buildEstu.pump_up_factors(
                data, modelDate, currentEstu=estuIdx, baseEstu=eligibleExchangeIdx,
                minFactorWidth=minFactorWidth, downWeights=exchangeDownWeight)
        logging.info('Estimation universe currently stands at %d stocks', len(estuIdx))

        # Apply grandfathering rules
        logging.info('Applying grandfathering')
        (estuIdx, ESTUQualify, nonest) = buildEstu.grandfather(
                modelDate, estuIdx, baseEstu=eligibleExchangeIdx, estuInstance=self.estuMap['main'])
        logging.info('Estimation universe currently stands at %d stocks', len(estuIdx))

        self.estuMap['main'].assets = [buildEstu.assets[idx] for idx in estuIdx]
        self.estuMap['main'].qualify = [buildEstu.assets[idx] for idx in ESTUQualify]

        self.log.debug('generate_estimation_universe: end')
        return estuIdx

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
        if self.exposureStandardization.exceptionNames is not None:
            self.exposureStandardization.exceptionNames = list(set(self.exposureStandardization.exceptionNames))

        return data.exposureMatrix
