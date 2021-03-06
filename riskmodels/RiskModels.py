import datetime
import itertools
import logging
import numpy.ma as ma
import numpy as np
import numpy
import os.path
import pandas
import copy
import riskmodels
from riskmodels import Matrices
from riskmodels.Matrices import ExposureMatrix
from riskmodels.EquityModel import ModelFactor
from riskmodels import ModelParameters2017
from riskmodels import Classification
from riskmodels import CurrencyRisk
from riskmodels import EstimationUniverse_V4
from riskmodels import RiskCalculator_V4
from riskmodels import Standardization_V4
from riskmodels import FactorReturns
from riskmodels import EquityModel
from riskmodels import Utilities

"""US models go here
"""
class USAxioma2016MH(EquityModel.FundamentalModel):
    """US4 fundamental medium-horizon model.
    """
    rm_id = 202
    revision = 2
    rms_id = 185

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
    regionalStndList = []

    # descriptor settings
    wideCloneList = list(styleList)
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
        # Set up relevant styles to be created/used
        ModelParameters2017.defaultExposureParameters(self, self.styleList, descriptorNumeraire='USD')
        EquityModel.FundamentalModel.__init__(self, ['CUSIP'], modelDB, marketDB)

        # Set up eligible universe parameters
        self.elig_parameters = {'assetTypes': self.commonStockTypes + ['REIT'],
                                'excludeTypes': None,
                                'HomeCountry_List': ['US'],
                                'use_isin_country_Flag': True}

        # Set up regression parameters
        dummyThreshold = 6
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
                dummyType=None,
                marketRegression=False,
                dummyThreshold=dummyThreshold,
                useRealMCaps=True,
                kappa=None)

        # And this is for raw exposure proxying
        self.expProxyCalculator = ModelParameters2017.defaultExposureProxyParameters(
                self, modelDB,
                dummyType='market',
                dummyThreshold=dummyThreshold,
                kappa=5.0,
                useRealMCaps=True,
                regWeight='rootCap',
                )

        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=2)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=2)
        ModelParameters2017.defaultSpecificVarianceParameters(self, maxHistoryOverride=250)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

        # Set up standardization parameters
        gloScope = Standardization_V4.GlobalRelativeScope([f.name for f in self.styles])
        self.exposureStandardization = Standardization_V4.BucketizedStandardization(\
                [gloScope], fillWithZeroList=self.fillWithZeroList)

        # Set up descriptor standardization parameters
        descriptors = sorted(list(set([item for sublist in self.DescriptorMap.values() for item in sublist])))
        self.descriptorStandardization = Standardization_V4.BucketizedStandardization(
                [Standardization_V4.GlobalRelativeScope(descriptors)],
                mad_bound=15.0, exceptionNames=self.noStndDescriptor)

        modelDB.createCurrencyCache(marketDB)

    def generate_estimation_universe(\
            self, date, assetData, exposureMatrix, modelDB, marketDB, excludeFactors=None, grandfatherRMS_ID=None):
        """Estimation universe selection criteria for AX-US.
        """
        self.log.debug('generate_estimation_universe: begin')

        estuCls = EstimationUniverse_V4.ConstructEstimationUniverse(\
                date, assetData.universe, self, modelDB, marketDB, debugOutput=self.debuggingReporting)

        # Set up various eligible and total universes
        eligibleUniverse = set(assetData.eligibleUniverse)
        logging.info('Estimation universe currently stands at %d stocks', len(eligibleUniverse))
        n = len(eligibleUniverse)

        # Keep only issues trading on NYSE and NASDAQ
        fade_in_date = Utilities.parseISODate('2009-01-01')
        fade_out_date = Utilities.parseISODate('2010-01-01')
        keepList = ['NAS','NYS','IEX','EXI']
        logging.info('Excluding assets not on %s', ','.join(keepList))
        eligibleExchange = estuCls.exclude_by_market_type(
                assetData, includeFields=keepList, excludeFields=None, baseEstu=eligibleUniverse)
        ineligExchange = eligibleUniverse.difference(eligibleExchange)

        # Downweight assets on other exchanges for pre-2010 dates rather than exclude
        exchangeDownWeight = pandas.Series(1.0, index=eligibleUniverse)
        delta = 0.25
        if date < fade_in_date:
            eligibleExchange = set(eligibleUniverse)
        elif date > fade_out_date:
            delta = 0.0
        else:
            eligibleExchange = set(eligibleUniverse)
            delta = delta * float((fade_out_date - date).days) / float((fade_out_date - fade_in_date).days)
        if delta > 0.0:
            logging.info('Downweighting factor for pre-%d OTCs: %.3f', fade_out_date.year, delta)
        exchangeDownWeight[ineligExchange] = delta
        n = estuCls.report_on_changes(n, eligibleExchange)

        # Load return score descriptors
        logging.info('Looking for sparsely traded stocks')
        descDict = dict(modelDB.getAllDescriptors())
        scoreTypes = ['ISC_Ret_Score', 'ISC_Zero_Score', 'ISC_ADV_Score']
        assetData.assetScoreDict, okDescriptorCoverageMap = self.loadDescriptors(
                scoreTypes, descDict, date, assetData.universe, modelDB,
                assetData.getCurrencyAssetMap(), rollOver=self.rollOverDescriptors)
        for typ in scoreTypes:
            exposureMatrix.addFactor(typ, assetData.assetScoreDict.loc[:, typ].fillna(0.0).values, ExposureMatrix.StyleFactor)

        # Report on thinly-traded assets over the entire universe
        nonSparse = estuCls.exclude_sparsely_traded_assets_legacy(assetData.assetScoreDict, baseEstu=eligibleUniverse)
        estu = eligibleExchange.intersection(nonSparse)
        n = estuCls.report_on_changes(n, estu)

        # Rank stuff by market cap and total volume over past year
        logging.info('Excluding assets based on cap and volume')
        estu = estuCls.filter_by_cap_and_volume(
                assetData, baseEstu=estu, downWeights=exchangeDownWeight[estu])
        n = estuCls.report_on_changes(n, estu)

        # Inflate thin industry factors if possible
        logging.info('Inflating any thin factors')
        estu, hasThinFactors = estuCls.pump_up_factors(
                assetData, exposureMatrix, currentEstu=estu, baseEstu=eligibleExchange,
                minFactorWidth=self.returnCalculator.allParameters[0].dummyThreshold,
                downWeights=exchangeDownWeight)
        n = estuCls.report_on_changes(n, estu)

        # Apply grandfathering rules
        logging.info('Applying grandfathering')
        estu, ESTUQualify = estuCls.grandfather(
                estu, baseEstu=eligibleExchange, estuInstance=self.estuMap['main'], grandfatherRMS_ID=grandfatherRMS_ID)
        n = estuCls.report_on_changes(n, estu)

        self.estuMap['main'].assets = estu
        self.estuMap['main'].qualify = ESTUQualify
        self.log.debug('generate_estimation_universe: end')
        return estu

    def generate_model_specific_exposures(self, modelDate, assetData, exposureMatrix, modelDB, marketDB):
        return self.generate_cap_bucket_exposures(modelDate, assetData, exposureMatrix, modelDB, marketDB)

class USAxioma2016SH(USAxioma2016MH):
    """US4 fundamental short-horizon model.
    """
    rm_id = 203
    revision = 2
    rms_id = 186

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
            'Earnings Yield': ['Earnings_to_Price_Quarterly', 'Est_Earnings_to_Price_12MFL_Quarterly'],
            'Value': ['Book_to_Price_Quarterly'],
            'Leverage': ['Debt_to_Assets_Quarterly', 'Debt_to_Equity_Quarterly'],
            'Growth': ['Earnings_Growth_RPF_AFQ', 'Sales_Growth_RPF_AFQ'],
            'Dividend Yield': ['Dividend_Yield_Quarterly'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_20D'],
            'Market Sensitivity': ['Market_Sensitivity_125D'],
            'Volatility': ['Volatility_60D'],
            'Medium-Term Momentum': ['Momentum_250x20D'],
            'Short-Term Momentum': ['Momentum_20D'],
            'Exchange Rate Sensitivity': ['XRate_52W_XDR'],
            'Profitability': ['Return_on_Equity_Quarterly', 'Return_on_Assets_Quarterly',
                              'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
                              'Sales_to_Assets_Quarterly', 'Gross_Margin_Quarterly'],
            }

    DescriptorWeights = {'Earnings Yield': [0.75, 0.25]}
    smallCapMap = {'MidCap': [66.67, 86.67],}
    noProxyList = ['Dividend Yield']
    fillMissingList = ['Earnings Yield', 'Value', 'Leverage', 'Growth', 'Profitability']
    fillWithZeroList = ['Dividend Yield']
    shrinkList = {'Liquidity': 20,
                  'Market Sensitivity': 125,
                  'Volatility': 60,
                  'Short-Term Momentum': 20,
                  'Medium-Term Momentum': 250}
    orthogList = {'Volatility': [['Market Sensitivity'], True, 1.0]}
    wideCloneList = list(styleList)

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
        self.log = logging.getLogger('RiskModels.USAxioma2016SH')
        # Set important model parameters
        ModelParameters2017.defaultModelSettingsUS4(self)
        # Set up relevant styles to be created/used
        ModelParameters2017.defaultExposureParameters(self, self.styleList, descriptorNumeraire='USD')
        EquityModel.FundamentalModel.__init__(self, ['CUSIP'], modelDB, marketDB)

        # Set up eligible universe parameters
        self.elig_parameters = {'assetTypes': self.commonStockTypes + ['REIT'],
                                'excludeTypes': None,
                                'HomeCountry_List': ['US'],
                                'use_isin_country_Flag': True}

        # Set up regression parameters
        dummyThreshold = 6
        self.returnCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType='Industry Groups',
                dummyThreshold=dummyThreshold,
                marketRegression=False,
                useRealMCaps=True,
                kappa=5.0,
                )

        # This controls the FMP regression
        self.fmpCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType=None,
                marketRegression=False,
                dummyThreshold=dummyThreshold,
                useRealMCaps=True,
                kappa=None)

        # And this is for raw exposure proxying
        self.expProxyCalculator = ModelParameters2017.defaultExposureProxyParameters(
                self, modelDB,
                dummyType='market',
                dummyThreshold=dummyThreshold,
                kappa=5.0,
                useRealMCaps=True,
                regWeight='rootCap',
                )

        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=2, horizon='short', dva_downweightEnds=False)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=2, horizon='short', dva_downweightEnds=False)
        ModelParameters2017.defaultSpecificVarianceParameters(self, horizon='short', maxHistoryOverride=250)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

        # Set up standardization parameters
        gloScope = Standardization_V4.GlobalRelativeScope([f.name for f in self.styles])
        self.exposureStandardization = Standardization_V4.BucketizedStandardization([gloScope],
                fillWithZeroList=self.fillWithZeroList)

        # Set up descriptor standardization parameters
        descriptors = sorted(list(set([item for sublist
            in list(self.DescriptorMap.values()) for item in sublist])))
        self.descriptorStandardization = Standardization_V4.BucketizedStandardization(
                [Standardization_V4.GlobalRelativeScope(descriptors)],
                mad_bound=15.0, exceptionNames=self.noStndDescriptor)

        modelDB.createCurrencyCache(marketDB)

class USAxioma2016MH_S(EquityModel.StatisticalModel):
    """US4 Medium Horizon Statistical Model
    """
    rm_id = 204
    revision = 2
    rms_id = 187
    numFactors = 15
    blind = [ModelFactor('Statistical Factor %d' % n, 'Statistical Factor %d' % n)
                         for n in range(1, numFactors+1)]
    industryClassification = Classification.GICSIndustries(datetime.date(2016,9,1))
    pcaHistory = 250
    wideCloneList = list([s.name for s in blind])
    descriptorNumeraire = 'USD'

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.USAxioma2016MH_S')
        # Set important model parameters
        ModelParameters2017.defaultModelSettingsUS4(self, statModel=True)
        EquityModel.StatisticalModel.__init__(self, ['CUSIP'], modelDB, marketDB)
        # Set up eligible universe parameters
        self.elig_parameters = {'assetTypes': self.commonStockTypes + ['REIT'],
                                'excludeTypes': None,
                                'HomeCountry_List': ['US'],
                                'use_isin_country_Flag': True}
        # So we can use the same ESTU method as the fundamental model
        self.baseModelDateMap = {datetime.date(1980,1,1): USAxioma2016MH(modelDB, marketDB)}
        # Set up returns model
        self.returnCalculator = FactorReturns.AsymptoticPrincipalComponents2017(
                self.numFactors, trimExtremeExposures=False, replaceReturns=False, flexible=False, TOL=None)
        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=1)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=1)
        ModelParameters2017.defaultSpecificVarianceParameters(self, nwLag=1, maxHistoryOverride=250)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

class USAxioma2016SH_S(USAxioma2016MH_S):
    """US4 Short Horizon Statistical Model
    """
    rm_id = 205
    revision = 2
    rms_id = 188
    numFactors = 15
    blind = [ModelFactor('Statistical Factor %d' % n, 'Statistical Factor %d' % n)
                         for n in range(1, numFactors+1)]
    industryClassification = Classification.GICSIndustries(datetime.date(2016,9,1))
    pcaHistory = 250
    # Back-compatibility for fill-in rules
    gicsDate = datetime.date(2014,3,1)

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.USAxioma2016SH_S')
        # Set important model parameters
        ModelParameters2017.defaultModelSettingsUS4(self, statModel=True)
        EquityModel.StatisticalModel.__init__(self, ['CUSIP'], modelDB, marketDB)
        # Set up eligible universe parameters
        self.elig_parameters = {'assetTypes': self.commonStockTypes + ['REIT'],
                                'excludeTypes': None,
                                'HomeCountry_List': ['US'],
                                'use_isin_country_Flag': True}
        # So we can use the same ESTU method as the fundamental model
        self.baseModelDateMap = {datetime.date(1980,1,1): USAxioma2016MH(modelDB, marketDB)}
        # Set up returns model
        self.returnCalculator = FactorReturns.AsymptoticPrincipalComponents2017(
                self.numFactors, trimExtremeExposures=False, replaceReturns=False, flexible=False, TOL=None)
        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=1, horizon='short')
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=1, horizon='short')
        ModelParameters2017.defaultSpecificVarianceParameters(self, nwLag=1, horizon='short', maxHistoryOverride=250)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

class USAxioma2016FL(USAxioma2016MH):
    """US4 factor library.
    """
    rm_id = 214
    revision = 1
    rms_id = 189

    DescriptorMap = {
            'Book to Price Quarterly': ['Book_to_Price_Quarterly'],
            'Earnings to Price Quarterly': ['Earnings_to_Price_Quarterly'],
            'Est Earnings to Price Quarterly': ['Est_Earnings_to_Price_12MFL_Quarterly'],
            'Debt to Assets Quarterly': ['Debt_to_Assets_Quarterly'],
            'Debt to Equity Quarterly': ['Debt_to_Equity_Quarterly'],
            'Earnings Growth Quarterly': ['Earnings_Growth_RPF_AFQ'],
            'Sales Growth Quarterly': ['Sales_Growth_RPF_AFQ'],
            'Dividend Yield Quarterly': ['Dividend_Yield_Quarterly'],
            'Log of Issuer Cap': ['LnIssuerCap'],
            'Log of 20-Day ADV to Issuer Cap': ['LnTrading_Activity_20D'],
            'Log of 60-Day ADV to Issuer Cap': ['LnTrading_Activity_60D'],
            'Market Sensitivity 125 Day': ['Market_Sensitivity_125D'],
            'Market Sensitivity 250 Day': ['Market_Sensitivity_250D'],
            'Volatility 60 Day': ['Volatility_60D'],
            'Volatility 125 Day': ['Volatility_125D'],
            'Annual Return Excl Prev Month': ['Momentum_250x20D'],
            'Monthly Return': ['Momentum_20D'],
            'Exchange Rate Sensitivity 52 Week XDR': ['XRate_52W_XDR'],
            'Exchange Rate Sensitivity 104 Week XDR': ['XRate_104W_XDR'],
            'Return on Equity Quarterly': ['Return_on_Equity_Quarterly'],
            'Return on Assets Quarterly': ['Return_on_Assets_Quarterly'],
            'Cash Flow to Assets Annual': ['CashFlow_to_Assets_Annual'],
            'Cash Flow to Income Annual': ['CashFlow_to_Income_Annual'],
            'Sales to Assets Quarterly': ['Sales_to_Assets_Quarterly'],
            'Gross Margin Quarterly': ['Gross_Margin_Quarterly'],
            }
    
    styleList = sorted(list(DescriptorMap.keys()) + ['MidCap'])    

    DescriptorWeights = {}
    smallCapMap = {'MidCap': [66.67, 86.67],}
    noProxyList = ['Dividend Yield Quarterly']
    fillMissingList = []
    fillWithZeroList = ['Dividend Yield Quarterly']
    shrinkList = {'Log of 20-Day ADV to Issuer Cap': 20,
                  'Log of 60-Day ADV to Issuer Cap': 60,
                  'Market Sensitivity 125 Day': 125,
                  'Market Sensitivity 250 Day': 250,
                  'Volatility 60 Day': 60,
                  'Volatility 125 Day': 125,
                  'Annual Return Excl Prev Month': 250,
                  'Monthly Return': 20}
    orthogList = {}

    # descriptor settings
    wideCloneList = list(styleList)
    noStndDescriptor = []
    noCloneDescriptor = []
    for s in noProxyList:
        noStndDescriptor.extend(DescriptorMap[s])
        noCloneDescriptor.extend(DescriptorMap[s])

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.USAxioma2016FL')
        USAxioma2016MH.__init__(self, modelDB, marketDB)

class USSCAxioma2017MH(USAxioma2016MH):
    """US4 small cap fundamental medium-horizon model.
    """
    rm_id = 215
    revision = 1
    rms_id = 190

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
                 'Exchange Rate Sensitivity',
                ]

    DescriptorMap = {
            'Earnings Yield': ['Earnings_to_Price_Quarterly', 'Est_Earnings_to_Price_12MFL_Quarterly'],
            'Value': ['Book_to_Price_Quarterly'],
            'Leverage': ['Debt_to_Assets_Quarterly', 'Debt_to_Equity_Quarterly'],
            'Growth': ['Earnings_Growth_RPF_AFQ', 'Sales_Growth_RPF_AFQ'],
            'Dividend Yield': ['Dividend_Yield_Quarterly'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_60D', 'Amihud_Liquidity_125D', 'ISC_Ret_Score'],
            'Market Sensitivity': ['Market_Sensitivity_USSC_250D'],
            'Volatility': ['Volatility_USSC_125D'],
            'Medium-Term Momentum': ['Momentum_250x20D'],
            'Exchange Rate Sensitivity': ['XRate_104W_XDR'],
            'Profitability': ['Return_on_Equity_Quarterly', 'Return_on_Assets_Quarterly',
                              'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
                              'Sales_to_Assets_Quarterly', 'Gross_Margin_Quarterly'],
            }

    DescriptorWeights = {'Earnings Yield': [0.75, 0.25]}
    noProxyList = ['Dividend Yield']
    wideCloneList = list(styleList)
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
        self.log = logging.getLogger('RiskModels.USSCAxioma2017MH')
        # Set important model parameters
        ModelParameters2017.defaultModelSettingsUS4(self)
        self.twoRegressionStructure = True

        # Set up relevant styles to be created/used
        ModelParameters2017.defaultExposureParameters(self, self.styleList, descriptorNumeraire='USD')
        EquityModel.FundamentalModel.__init__(self, ['CUSIP'], modelDB, marketDB)

        # Set up eligible universe parameters
        self.elig_parameters = {'assetTypes': self.commonStockTypes + ['REIT'],
                                'excludeTypes': None,
                                'HomeCountry_List': ['US'],
                                'use_isin_country_Flag': True}

        # Set up regression parameters
        dummyThreshold = 6
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
                dummyType=None,
                marketRegression=False,
                dummyThreshold=dummyThreshold,
                useRealMCaps=True,
                kappa=None)

        # And this is for raw exposure proxying
        self.expProxyCalculator = ModelParameters2017.defaultExposureProxyParameters(
                self, modelDB,
                dummyType='market',
                dummyThreshold=dummyThreshold,
                kappa=5.0,
                useRealMCaps=True,
                regWeight='rootCap',
                )

        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=2)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=2)
        ModelParameters2017.defaultSpecificVarianceParameters(self, maxHistoryOverride=250)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

        # Set up standardization parameters
        gloScope = Standardization_V4.GlobalRelativeScope([f.name for f in self.styles])
        self.exposureStandardization = Standardization_V4.BucketizedStandardization(\
                [gloScope],fillWithZeroList=self.fillWithZeroList)

        # Set up descriptor standardization parameters
        descriptors = sorted(list(set([item for sublist in self.DescriptorMap.values() for item in sublist])))
        self.descriptorStandardization = Standardization_V4.BucketizedStandardization(
                [Standardization_V4.GlobalRelativeScope(descriptors)],
                mad_bound=15.0, exceptionNames=self.noStndDescriptor)

        modelDB.createCurrencyCache(marketDB)

    def generate_estimation_universe(\
            self, date, assetData, exposureMatrix, modelDB, marketDB, excludeFactors=None, grandfatherRMS_ID=None):
        """Estimation universe selection criteria for AX-US-SC.
        """
        self.log.debug('generate_estimation_universe: begin')

        estuCls = EstimationUniverse_V4.ConstructEstimationUniverse(\
                date, assetData.universe, self, modelDB, marketDB, debugOutput=self.debuggingReporting)

        # Set up various eligible and total universes
        eligibleUniverse = set(assetData.eligibleUniverse)
        logging.info('Estimation universe currently stands at %d stocks', len(eligibleUniverse))
        n = len(eligibleUniverse)

        # Keep only issues trading on NYSE and NASDAQ
        fade_in_date = Utilities.parseISODate('2009-01-01')
        fade_out_date = Utilities.parseISODate('2010-01-01')
        keepList = ['NAS','NYS','IEX','EXI']
        logging.info('Excluding assets not on %s', ','.join(keepList))
        eligibleExchange = estuCls.exclude_by_market_type(
                assetData, includeFields=keepList, excludeFields=None, baseEstu=eligibleUniverse)
        ineligExchange = eligibleUniverse.difference(eligibleExchange)

        # Downweight assets on other exchanges for pre-2010 dates rather than exclude
        exchangeDownWeight = pandas.Series(1.0, index=eligibleUniverse)
        delta = 0.25
        if date < fade_in_date:
            eligibleExchange = set(eligibleUniverse)
        elif date > fade_out_date:
            delta = 0.0
        else:
            eligibleExchange = set(eligibleUniverse)
            delta = delta * float((fade_out_date - date).days) / float((fade_out_date - fade_in_date).days)
        if delta > 0.0:
            logging.info('Downweighting factor for pre-%d OTCs: %.3f', fade_out_date.year, delta)
        exchangeDownWeight[ineligExchange] = delta
        n = estuCls.report_on_changes(n, eligibleExchange)

        # Load return score descriptors
        logging.info('Looking for sparsely traded stocks')
        descDict = dict(modelDB.getAllDescriptors())
        scoreTypes = ['ISC_Ret_Score', 'ISC_Zero_Score', 'ISC_ADV_Score']
        assetData.assetScoreDict, okDescriptorCoverageMap = self.loadDescriptors(
                scoreTypes, descDict, date, assetData.universe, modelDB,
                assetData.getCurrencyAssetMap(), rollOver=self.rollOverDescriptors)
        for typ in scoreTypes:
            exposureMatrix.addFactor(typ, assetData.assetScoreDict.loc[:, typ].fillna(0.0).values, ExposureMatrix.StyleFactor)

        # Report on thinly-traded assets over the entire universe
        nonSparse = estuCls.exclude_sparsely_traded_assets_legacy(assetData.assetScoreDict, baseEstu=eligibleUniverse)
        estu = eligibleExchange.intersection(nonSparse)
        n = estuCls.report_on_changes(n, estu)

        # Rank stuff by market cap and total volume over past year
        logging.info('Excluding assets based on cap and volume')
        estu = estuCls.filter_by_cap_and_volume(
                assetData, baseEstu=estu, downWeights=exchangeDownWeight[estu])
        n = estuCls.report_on_changes(n, estu)

        # Exclude assets from estuIdx with mcaps < .8 mcap quanitle
        estuMcap = assetData.marketCaps[estu]
        maxMcap = estuMcap[estuMcap<estuMcap.quantile(.8)].max()
        estu = set(estuMcap[estuMcap<estuMcap.quantile(.8)].index)
        eligMcap = assetData.marketCaps[eligibleExchange]
        eligibleExchange = set(eligMcap[eligMcap<=maxMcap].index)

        # Inflate thin industry factors if possible
        logging.info('Inflating any thin factors')
        estu, hasThinFactors = estuCls.pump_up_factors(
                assetData, exposureMatrix, currentEstu=estu, baseEstu=eligibleExchange,
                minFactorWidth=self.returnCalculator.allParameters[0].dummyThreshold,
                downWeights=exchangeDownWeight)
        n = estuCls.report_on_changes(n, estu)

        # Apply grandfathering rules
        logging.info('Applying grandfathering')
        estu, ESTUQualify = estuCls.grandfather(
                estu, baseEstu=eligibleExchange, estuInstance=self.estuMap['main'], grandfatherRMS_ID=grandfatherRMS_ID)
        n = estuCls.report_on_changes(n, estu)

        # Drop assets that are more than 1.5* the max mcap on the modelDate
        estuMcap = assetData.marketCaps[estu]
        estu = set(estuMcap[estuMcap<=1.5*maxMcap].index)

        self.estuMap['main'].assets = estu
        self.estuMap['main'].qualify = ESTUQualify
        self.log.debug('generate_estimation_universe: end')
        return estu

class USSCAxioma2017SH(USSCAxioma2017MH):
    """US4 small cap fundamental short-horizon model.
    """
    rm_id = 216
    revision = 1
    rms_id = 191

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
                 'Exchange Rate Sensitivity',
                ]

    DescriptorMap = {
            'Earnings Yield': ['Earnings_to_Price_Quarterly', 'Est_Earnings_to_Price_12MFL_Quarterly'],
            'Value': ['Book_to_Price_Quarterly'],
            'Leverage': ['Debt_to_Assets_Quarterly', 'Debt_to_Equity_Quarterly'],
            'Growth': ['Earnings_Growth_RPF_AFQ', 'Sales_Growth_RPF_AFQ'],
            'Dividend Yield': ['Dividend_Yield_Quarterly'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_20D', 'Amihud_Liquidity_60D', 'ISC_Ret_Score'],
            'Market Sensitivity': ['Market_Sensitivity_USSC_125D'],
            'Volatility': ['Volatility_USSC_60D'],
            'Medium-Term Momentum': ['Momentum_250x20D'],
            'Short-Term Momentum': ['Momentum_20D'],
            'Exchange Rate Sensitivity': ['XRate_52W_XDR'],
            'Profitability': ['Return_on_Equity_Quarterly', 'Return_on_Assets_Quarterly',
                              'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
                              'Sales_to_Assets_Quarterly', 'Gross_Margin_Quarterly'],
            }

    DescriptorWeights = {'Earnings Yield': [0.75, 0.25]}
    noProxyList = ['Dividend Yield']
    fillMissingList = ['Earnings Yield', 'Value', 'Leverage', 'Growth', 'Profitability']
    fillWithZeroList = ['Dividend Yield']
    shrinkList = {'Liquidity': 20,
                  'Market Sensitivity': 125,
                  'Volatility': 60,
                  'Short-Term Momentum': 20,
                  'Medium-Term Momentum': 250}
    orthogList = {'Volatility': [['Market Sensitivity'], True, 1.0]}
    regionalStndList = []

    # descriptor settings
    wideCloneList = list(styleList)
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
        self.log = logging.getLogger('RiskModels.USSCAxioma2016SH')
        # Set important model parameters
        ModelParameters2017.defaultModelSettingsUS4(self)
        self.twoRegressionStructure = True
        # Set up relevant styles to be created/used
        ModelParameters2017.defaultExposureParameters(self, self.styleList, descriptorNumeraire='USD')
        EquityModel.FundamentalModel.__init__(self, ['CUSIP'], modelDB, marketDB)

        # Set up eligible universe parameters
        self.elig_parameters = {'assetTypes': self.commonStockTypes + ['REIT'],
                                'excludeTypes': None,
                                'HomeCountry_List': ['US'],
                                'use_isin_country_Flag': True}

        # Set up regression parameters
        dummyThreshold = 6
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
                dummyType=None,
                marketRegression=False,
                dummyThreshold=dummyThreshold,
                useRealMCaps=True,
                kappa=None)

        # And this is for raw exposure proxying
        self.expProxyCalculator = ModelParameters2017.defaultExposureProxyParameters(
                self, modelDB,
                dummyType='market',
                dummyThreshold=dummyThreshold,
                kappa=5.0,
                useRealMCaps=True,
                regWeight='rootCap',
                )

        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=2, horizon='short', dva_downweightEnds=False)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=2, horizon='short', dva_downweightEnds=False)
        ModelParameters2017.defaultSpecificVarianceParameters(self, horizon='short', maxHistoryOverride=250)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

        # Set up standardization parameters
        gloScope = Standardization_V4.GlobalRelativeScope([f.name for f in self.styles])
        self.exposureStandardization = Standardization_V4.BucketizedStandardization(\
                [gloScope],fillWithZeroList=self.fillWithZeroList)

        # Set up descriptor standardization parameters
        descriptors = sorted(list(set([item for sublist in self.DescriptorMap.values() for item in sublist])))
        self.descriptorStandardization = Standardization_V4.BucketizedStandardization(
                [Standardization_V4.GlobalRelativeScope(descriptors)],
                mad_bound=15.0, exceptionNames=self.noStndDescriptor)

        modelDB.createCurrencyCache(marketDB)

class USSCAxioma2017MH_S(EquityModel.StatisticalModel):
    """US4 Small Cap Medium Horizon Statistical Model
    """
    rm_id = 217
    revision = 1
    rms_id = 192
    numFactors = 15
    blind = [ModelFactor('Statistical Factor %d' % n, 'Statistical Factor %d' % n) for n in range(1, numFactors+1)]
    industryClassification = Classification.GICSIndustries(datetime.date(2016,9,1))
    pcaHistory = 250
    descriptorNumeraire = 'USD'

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.USSCAxioma2017MH_S')
        # Set important model parameters
        ModelParameters2017.defaultModelSettingsUS4(self, statModel=True)
        EquityModel.StatisticalModel.__init__(self, ['CUSIP'], modelDB, marketDB)
        # Set up eligible universe parameters
        self.elig_parameters = {'assetTypes': self.commonStockTypes + ['REIT'],
                                'excludeTypes': None,
                                'HomeCountry_List': ['US'],
                                'use_isin_country_Flag': True}
        # So we can use the same ESTU method as the fundamental model
        self.baseModelDateMap = {datetime.date(1980,1,1): USSCAxioma2017MH(modelDB, marketDB)}
        # Set up returns model
        self.returnCalculator = FactorReturns.AsymptoticPrincipalComponents2017(
                self.numFactors, trimExtremeExposures=False, replaceReturns=False, flexible=False, TOL=None)
        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=1)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=1)
        ModelParameters2017.defaultSpecificVarianceParameters(self, nwLag=1, maxHistoryOverride=250)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

class USSCAxioma2017SH_S(USSCAxioma2017MH_S):
    """US4 Short Horizon Statistical Model
    """
    rm_id = 218
    revision = 1
    rms_id = 193
    numFactors = 15
    blind = [ModelFactor('Statistical Factor %d' % n, 'Statistical Factor %d' % n)
                         for n in range(1, numFactors+1)]
    industryClassification = Classification.GICSIndustries(datetime.date(2016,9,1))
    pcaHistory = 250
    # Back-compatibility for fill-in rules
    gicsDate = datetime.date(2014,3,1)

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.USSCAxioma2017SH_S')
        # Set important model parameters
        ModelParameters2017.defaultModelSettingsUS4(self, statModel=True)
        EquityModel.StatisticalModel.__init__(self, ['CUSIP'], modelDB, marketDB)
        # Set up eligible universe parameters
        self.elig_parameters = {'assetTypes': self.commonStockTypes + ['REIT'],
                                'excludeTypes': None,
                                'HomeCountry_List': ['US'],
                                'use_isin_country_Flag': True}
        # So we can use the same ESTU method as the fundamental model
        self.baseModelDateMap = {datetime.date(1980,1,1): USSCAxioma2017MH(modelDB, marketDB)}
        # Set up returns model
        self.returnCalculator = FactorReturns.AsymptoticPrincipalComponents2017(
                self.numFactors, trimExtremeExposures=False, replaceReturns=False, flexible=False, TOL=None)
        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=1, horizon='short')
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=1, horizon='short')
        ModelParameters2017.defaultSpecificVarianceParameters(self, nwLag=1, horizon='short', maxHistoryOverride=250)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

""" US4.1 models here
"""
class USAxioma2021MH(EquityModel.FundamentalModel):
    """US4 fundamental medium-horizon model.
    """
    rm_id = 402
    revision = 1
    rms_id = 402

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

    intercept = ModelFactor('Market Intercept', 'Market Intercept')
    smallCapMap = {'MidCap': [66.67, 86.67],}
    gicsDate = datetime.date(2018,9,29)
    industryClassification = Classification.GICSIndustries(gicsDate)
    firstReturnDate = datetime.date(1980,1,1)

    DescriptorMap = {
            'Earnings Yield': ['Earnings_to_Price_Quarterly', 'Est_Earnings_to_Price_12MFL_Quarterly'],
            'Value': ['Book_to_Price_Quarterly'],
            'Leverage': ['Debt_to_Assets_Quarterly', 'Debt_to_Equity_Quarterly'],
            'Growth': ['Earnings_Growth_RPF_AFQ', 'Sales_Growth_RPF_AFQ'],
            'Dividend Yield': ['Dividend_Yield_Quarterly'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_60D', 'Amihud_Liquidity_125D', 'ISC_Ret_Score'],
            'Market Sensitivity': ['Market_Sensitivity_104W'],
            'Volatility': ['RMG_Volatility_125D'],
            'Medium-Term Momentum': ['Momentum_260x21D_Regional'],
            'Exchange Rate Sensitivity': ['XRate_104W_XDR'],
            'Profitability': ['Return_on_Equity_Quarterly', 'Return_on_Assets_Quarterly',
                              'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
                              'Sales_to_Assets_Quarterly', 'Gross_Margin_Quarterly'],
            }

    DescriptorWeights = {'Earnings Yield': [0.75, 0.25]}
    allowMixedFreqDescriptors = True
    orthogList = {'Volatility': [['Market Sensitivity'], True, 1.0]}
    exposureConfigFile = 'exposures-US-mh'

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.USAxioma2021MH')

        # Set important model parameters
        ModelParameters2017.defaultModelSettings(self, scm=True)

        # Set up relevant styles to be created/used
        ModelParameters2017.defaultExposureParameters(self, self.styleList, configFile=self.exposureConfigFile)

        # Initialise
        EquityModel.FundamentalModel.__init__(self, ['CUSIP'], modelDB, marketDB)

        # Set up eligible universe parameters
        self.elig_parameters = {'assetTypes': self.commonStockTypes + ['REIT'],
                                'excludeTypes': None,
                                'HomeCountry_List': ['US'],
                                'use_isin_country_Flag': False}

        # Set up RiskModel Calculators
        self.setCalculators(modelDB)

        # Set up descriptor standardization parameters
        descriptors = sorted(list(set([item for sublist in self.DescriptorMap.values() for item in sublist])))
        gloScope = Standardization_V4.GlobalRelativeScope(descriptors)
        self.descriptorStandardization = Standardization_V4.BucketizedStandardization(
                [gloScope], mad_bound=15.0,  exceptionNames=self.noStndDescriptor)

        # Set up standardization parameters
        gloScope = Standardization_V4.GlobalRelativeScope([f.name for f in self.styles])
        self.exposureStandardization = Standardization_V4.BucketizedStandardization(\
                [gloScope], fillWithZeroList=self.fillWithZeroList)

    def setCalculators(self, modelDB, overrider=False):
        # Set up regression parameters
        dummyThreshold = 6
        self.internalCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB, dummyType='Industry Groups', dummyThreshold=dummyThreshold,
                marketRegression=False, kappa=5.0, useRealMCaps=True, regWeight='rootCap')
        self.returnCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB, dummyType='Industry Groups', dummyThreshold = dummyThreshold,
                marketRegression=False, kappa=25.0, useRealMCaps=True, regWeight='rootCap')

        # This controls the FMP regression
        self.fmpCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB, dummyType=None, dummyThreshold=dummyThreshold,
                marketRegression=False, useRealMCaps=True, kappa=None)

        # And this is for raw exposure proxying
        self.expProxyCalculator = ModelParameters2017.defaultExposureProxyParameters(
                self, modelDB, dummyType='market', dummyThreshold=dummyThreshold,
                kappa=5.0, useRealMCaps=True, regWeight='rootCap')

        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=3)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=3)
        ModelParameters2017.defaultSpecificVarianceParameters(self)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

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
        keepList = ['NAS','NYS','IEX','EXI']
        logging.info('Excluding assets not on %s', ','.join(keepList))
        (eligibleExchangeIdx, nonest) = buildEstu.exclude_by_market_type(
                modelDate, data, includeFields=keepList, excludeFields=None,
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
        self.checkTrackedAssets(buildEstu.assets, data, eligibleExchangeIdx)

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
        self.checkTrackedAssets(buildEstu.assets, data, estuIdx)

        # Rank stuff by market cap and total volume over past year
        logging.info('Excluding assets based on cap and volume')
        (estuIdx, nonest) = buildEstu.filter_by_cap_and_volume(
                data, modelDate, baseEstu=estuIdx, downWeights=exchangeDownWeightNS)
        logging.info('Estimation universe currently stands at %d stocks', len(estuIdx))
        self.checkTrackedAssets(buildEstu.assets, data, estuIdx)

        # Inflate thin industry factors if possible
        logging.info('Inflating any thin factors')
        minFactorWidth=self.returnCalculator.allParameters[0].dummyThreshold
        (estuIdx, nonest) = buildEstu.pump_up_factors(
                data, modelDate, currentEstu=estuIdx, baseEstu=eligibleExchangeIdx,
                minFactorWidth=minFactorWidth, downWeights=exchangeDownWeight)
        logging.info('Estimation universe currently stands at %d stocks', len(estuIdx))
        self.checkTrackedAssets(buildEstu.assets, data, estuIdx)

        # Apply grandfathering rules
        logging.info('Applying grandfathering')
        (estuIdx, ESTUQualify, nonest) = buildEstu.grandfather(
                modelDate, estuIdx, baseEstu=eligibleExchangeIdx, estuInstance=self.estuMap['main'],
                grandfatherRMS_ID=grandfatherRMS_ID)
        logging.info('Estimation universe currently stands at %d stocks', len(estuIdx))
        self.checkTrackedAssets(buildEstu.assets, data, estuIdx)

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

class USAxioma2021SH(USAxioma2021MH):
    """US4 fundamental short-horizon model.
    """
    rm_id = 403
    revision = 1
    rms_id = 403

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
            'Earnings Yield': ['Earnings_to_Price_Quarterly', 'Est_Earnings_to_Price_12MFL_Quarterly'],
            'Value': ['Book_to_Price_Quarterly'],
            'Leverage': ['Debt_to_Assets_Quarterly', 'Debt_to_Equity_Quarterly'],
            'Growth': ['Earnings_Growth_RPF_AFQ', 'Sales_Growth_RPF_AFQ'],
            'Dividend Yield': ['Dividend_Yield_Quarterly'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_20D', 'Amihud_Liquidity_60D', 'ISC_Ret_Score'],
            'Market Sensitivity': ['Market_Sensitivity_52W'],
            'Volatility': ['RMG_Volatility_60D'],
            'Medium-Term Momentum': ['Momentum_260x21D_Regional'],
            'Short-Term Momentum': ['Momentum_21D'],
            'Exchange Rate Sensitivity': ['XRate_52W_XDR'],
            'Profitability': ['Return_on_Equity_Quarterly', 'Return_on_Assets_Quarterly',
                              'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
                              'Sales_to_Assets_Quarterly', 'Gross_Margin_Quarterly'],
            }

    exposureConfigFile = 'exposures-US-sh'

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.USAxioma2021SH')
        USAxioma2021MH.__init__(self, modelDB, marketDB)

        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=3, horizon='short')
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=3, horizon='short')
        ModelParameters2017.defaultSpecificVarianceParameters(self, horizon='short')
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

"""AU models go here
"""
class AUAxioma2016MH(EquityModel.FundamentalModel):
    """
        AU2 base model. - 2016 version
    """
    # Model Parameters:
    rm_id,revision,rms_id = [206,1,200]
    k  = 5.0
    elig_parameters = {'HomeCountry_List': ['AU','NZ'],
                       'use_isin_country_Flag': False,
                       'assetTypes':['All-Com', 'REIT','StapSec'], # assetTypes is updated at the init
                       'excludeTypes': None}
    estu_parameters = {
                       'minNonZero':0.1,
                       'minNonMissing':0.5,
                       'CapByNumber_Flag':True,
                       'CapByNumber_hiCapQuota':200,
                       'CapByNumber_lowCapQuota':100,
                       'market_lower_pctile': np.nan,
                       'country_lower_pctile': np.nan,
                       'industry_lower_pctile': np.nan,
                       'dummyThreshold': 6,
                       'inflation_cutoff':0.03
                        }
    # List of style factors in the model
    styleList = ['Value','Leverage','Growth','Profitability','Dividend Yield','Size',
                 'Liquidity','Market Sensitivity','Volatility','Medium-Term Momentum',
                 'Exchange Rate Sensitivity','EM Sensitivity'
                 ]
    DescriptorMap = {
            'Value': ['Book_to_Price_Annual','Earnings_to_Price_Annual', 'Est_Earnings_to_Price_Annual'],
            'Leverage': ['Debt_to_Assets_Annual', 'Debt_to_Equity_Annual'],
            'Growth': ['Earnings_Growth_RPF_Annual', 'Sales_Growth_RPF_Annual'],
            'Dividend Yield': ['Dividend_Yield_Annual'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_60D'],
            'Market Sensitivity': ['Market_Sensitivity_250D'],
            'Volatility': ['Volatility_125D'],
            'Medium-Term Momentum': ['Momentum_250x20D'],
            'Exchange Rate Sensitivity': ['XRate_104W_XDR'],
            'EM Sensitivity': ['Market_Sensitivity_EM_104W','Market_Sensitivity_CN_104W'],
            'Profitability': ['Return_on_Equity_Annual','Return_on_Assets_Annual', 'CashFlow_to_Assets_Annual',
                              'CashFlow_to_Income_Annual','Sales_to_Assets_Annual', 'Gross_Margin_Annual'],
            }
    DescriptorWeights = {'Value': [0.5, 0.125, 0.375], # more weigth for forecast EY
                        'EM Sensitivity': [0.5,0.5]}
    smallCapMap = {}
    noProxyList = ['Dividend Yield']
    fillMissingList = ['Value', 'Leverage', 'Growth', 'Profitability']
    fillWithZeroList = ['Dividend Yield']
    shrinkList = {'Liquidity': 60,
                  'Market Sensitivity': 250,
                  'Volatility': 125,
                  'Exchange Rate Sensitivity': 500,
                  'EM Sensitivity': 500,
                  'Medium-Term Momentum': 250}
    orthogList = {'Volatility': [['Market Sensitivity'], True, 1.0],
                  'EM Sensitivity': [['Exchange Rate Sensitivity'], True, 1.0]}
    regionalStndList = []

    # descriptor settings
    wideCloneList = list(styleList)
    noStndDescriptor = []
    noCloneDescriptor = []
    for s in noProxyList:
        noStndDescriptor.extend(DescriptorMap[s])
        noCloneDescriptor.extend(DescriptorMap[s])

    # Setting up market intercept if relevant
    interceptFactor = 'Market Intercept'
    intercept = ModelFactor(interceptFactor, interceptFactor)
    # industry Classification
    industryClassification = Classification.GICSCustomAU2(datetime.date(2016,9,1))

    def __init__(self, modelDB, marketDB, expTreat=None):
        self.log = logging.getLogger('RiskModels.AUAxioma2016MH')
        # update parameter using parent class value
        self.elig_parameters['assetTypes'] = self.elig_parameters['assetTypes'] + self.commonStockTypes
        # Set up relevant styles to be created/used
        ModelParameters2017.defaultExposureParameters(self, self.styleList, descriptorNumeraire='USD')
        EquityModel.FundamentalModel.__init__(self, ['SEDOL'], modelDB, marketDB)
        # Set up regression parameters
        self.setCalculators(modelDB)
        # Set up standardization parameters
        gloScope = Standardization_V4.GlobalRelativeScope([f.name for f in self.styles])
        self.exposureStandardization = Standardization_V4.BucketizedStandardization(
                [gloScope], fillWithZeroList=self.fillWithZeroList)
        modelDB.createCurrencyCache(marketDB)
        # Set up descriptor standardization parameters
        descriptors = sorted(list(set([item for sublist in self.DescriptorMap.values() for item in sublist])))
        self.descriptorStandardization = Standardization_V4.BucketizedStandardization(
                [Standardization_V4.GlobalRelativeScope(descriptors)],
                mad_bound=15.0, exceptionNames=self.noStndDescriptor)

    def setCalculators(self, modelDB, overrider=False):
        # Set important model parameters
        ModelParameters2017.defaultModelSettingsSCM(self)
        #internal factor return
        self.internalCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,dummyType='Sectors',
                dummyThreshold=self.estu_parameters['dummyThreshold'],
                marketRegression=False, kappa = self.k,
                useRealMCaps=True, regWeight='rootCap', 
                overrider=overrider)
        #external factor return
        self.returnCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,dummyType='Sectors',
                dummyThreshold=self.estu_parameters['dummyThreshold'],
                marketRegression=False, kappa = None,
                useRealMCaps=True, regWeight='rootCap',
                overrider=overrider)
        #external factor return (not used)
        self.returnCalculator_V2 = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,dummyType='Sectors',
                dummyThreshold=self.estu_parameters['dummyThreshold'],
                marketRegression=False, kappa = None,
                useRealMCaps=True, regWeight='invSpecificVariance',
                overrider=overrider)
        # This controls the FMP regression
        self.fmpCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,dummyType=None,
                dummyThreshold=self.estu_parameters['dummyThreshold'],
                marketRegression=False, kappa=None,
                useRealMCaps=True, regWeight='rootCap',
                overrider=overrider)
        # And this is for raw exposure proxying
        self.expProxyCalculator = ModelParameters2017.defaultExposureProxyParameters(
                self, modelDB,
                dummyType='market',
                dummyThreshold=6,
                kappa=5.0,
                useRealMCaps=True,
                regWeight='rootCap',
                overrider=overrider,
                )
        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=2, overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=2, overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, maxHistoryOverride=250, overrider=overrider)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

    def generate_model_specific_exposures(self, modelDate, assetData, exposureMatrix, modelDB, marketDB):
        return self.generate_cap_bucket_exposures(modelDate, assetData, exposureMatrix, modelDB, marketDB)

class AUAxioma2016SH(AUAxioma2016MH):
    """Production AX-AU-S Australia model - 2016 short term fundamential model
    """
    # Model Parameters:
    rm_id,revision,rms_id = [208,1,202]
    styleList = ['Value','Leverage','Growth','Profitability','Dividend Yield','Size',
                 'Liquidity','Market Sensitivity','Volatility','Medium-Term Momentum','Short-Term Momentum',
                 'Exchange Rate Sensitivity','EM Sensitivity'
                 ]
    DescriptorMap = {
            'Value': ['Book_to_Price_Annual','Earnings_to_Price_Annual', 'Est_Earnings_to_Price_Annual'],
            'Leverage': ['Debt_to_Assets_Annual', 'Debt_to_Equity_Annual'],
            'Growth': ['Earnings_Growth_RPF_Annual', 'Sales_Growth_RPF_Annual'],
            'Dividend Yield': ['Dividend_Yield_Annual'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_20D'],
            'Market Sensitivity': ['Market_Sensitivity_125D'],
            'Volatility': ['Volatility_60D'],
            'Medium-Term Momentum': ['Momentum_250x20D'],
            'Short-Term Momentum': ['Momentum_20D'],
            'Exchange Rate Sensitivity': ['XRate_52W_XDR'],
            'EM Sensitivity': ['Market_Sensitivity_EM_52W','Market_Sensitivity_CN_52W'], #short term version
            'Profitability': ['Return_on_Equity_Annual','Return_on_Assets_Annual', 'CashFlow_to_Assets_Annual',
                              'CashFlow_to_Income_Annual','Sales_to_Assets_Annual', 'Gross_Margin_Annual'],
            }
    shrinkList = {'Liquidity': 20,
              'Market Sensitivity': 125,
              'Volatility': 60,
              'Exchange Rate Sensitivity': 250,
              'EM Sensitivity': 250,
              'Short-Term Momentum': 20,
              'Medium-Term Momentum': 250}
    wideCloneList = list(styleList)

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.AUAxioma2016SH')
        self.setCalculators(modelDB)
        super(AUAxioma2016SH,self).__init__(modelDB, marketDB)
    
    def setCalculators(self, modelDB, overrider=False):
        # Set important model parameters
        super(AUAxioma2016SH, self).setCalculators(modelDB, overrider=overrider)
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=2, horizon='short', dva_downweightEnds=False, overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=2, horizon='short', dva_downweightEnds=False, overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, horizon='short', maxHistoryOverride=250, overrider=overrider)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

class AUAxioma2016MH_S(EquityModel.StatisticalModel):
    """AU2 statistical model - 2016"""

    # Model Parameters:
    rm_id,revision,rms_id = [207,1,201]

    numFactors = 15
    blind = [ModelFactor('Statistical Factor %d' % n,'Statistical Factor %d' % n) for n in range(1, numFactors+1)]
    pcaHistory = 250
    wideCloneList = list([s.name for s in blind])
    industryClassification = Classification.GICSCustomAU2(datetime.date(2016,9,1))
    elig_parameters = {'HomeCountry_List': ['AU','NZ'],
                       'use_isin_country_Flag': False,
                       'assetTypes':['All-Com', 'REIT','StapSec'], # assetTypes is updated at the init
                       'excludeTypes': None}
    descriptorNumeraire = 'USD'

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.AUAxioma2016MH_S')
        EquityModel.StatisticalModel.__init__(self, ['SEDOL'], modelDB, marketDB)
        # update parameter using parent class value
        self.elig_parameters['assetTypes'] = self.elig_parameters['assetTypes'] + self.commonStockTypes
        # So we can use the same ESTU method as the fundamental model
        self.baseModelDateMap = {datetime.date(1980,1,1): AUAxioma2016MH(modelDB, marketDB)}
        self.setCalculators(modelDB)

    def setCalculators(self, modelDB, overrider=False):
        # Set important model parameters
        ModelParameters2017.defaultModelSettingsSCM(self, statModel=True)
        # Set up returns model
        self.returnCalculator = FactorReturns.AsymptoticPrincipalComponents2017(
                self.numFactors, trimExtremeExposures=False, replaceReturns=False, flexible=False, TOL=None)
        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=1, overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=1, overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, nwLag=1, maxHistoryOverride=250, overrider=overrider)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

class AUAxioma2016SH_S(AUAxioma2016MH_S):
    """Production AX-AU-S Australia model - 2016 short term statistical model
    """
    # Model Parameters:
    rm_id,revision,rms_id = [209,1,203]
    numFactors = 15
    blind = [ModelFactor('Statistical Factor %d' % n,'Statistical Factor %d' % n) for n in range(1, numFactors+1)]
    returnHistory = 250
    industryClassification = Classification.GICSCustomAU2(datetime.date(2016,9,1))

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.AUAxioma2016SH_S')
        self.setCalculators(modelDB)
        super(AUAxioma2016SH_S,self).__init__(modelDB, marketDB)

    def setCalculators(self, modelDB, overrider=False):
        # Set important model parameters
        ModelParameters2017.defaultModelSettingsSCM(self, statModel=True)
        # Set up returns model
        self.returnCalculator = FactorReturns.AsymptoticPrincipalComponents2017(
                self.numFactors, trimExtremeExposures=False, replaceReturns=False, flexible=False, TOL=None)
        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=1, horizon='short', overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=1, horizon='short', overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, nwLag=1, horizon='short',
                    maxHistoryOverride=250, overrider=overrider)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

"""JP models go here
"""
class JPAxioma2017MH(EquityModel.FundamentalModel):
    """ JP4 fundamental medium-horizon model
    """
    # Model parameters:
    rm_id = 210
    revision = 1
    rms_id = 210
    k = 5.0

    elig_parameters = {'HomeCountry_List': ['JP'],
                       'use_isin_country_Flag': True,
                       'excludeTypes': None}

    estu_parameters = {
                       'minNonZero':0.75,
                       'minNonMissing':0.95,
                       'CapByNumber_Flag':False,
                       'CapByNumber_hiCapQuota':np.nan,
                       'CapByNumber_lowCapQuota':np.nan,
                       'market_lower_pctile':1 ,
                       'country_lower_pctile':5,
                       'industry_lower_pctile':5,
                       'dummyThreshold': 6,
                       'inflation_cutoff':0.01
                        }

    dummyThreshold          = estu_parameters['dummyThreshold']
    minNonZero              = estu_parameters['minNonZero']
    inflation_cutoff        = estu_parameters['inflation_cutoff']
    market_lower_pctile     = estu_parameters['market_lower_pctile']
    country_lower_pctile    = estu_parameters['country_lower_pctile']
    industry_lower_pctile   = estu_parameters['industry_lower_pctile']

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
                  'Exchange Rate Sensitivity': 500,
                  'Medium-Term Momentum': 250}
    orthogList = {'Volatility': [['Market Sensitivity'], True, 1.0]}
    regionalStndList = []

    # descriptor settings
    wideCloneList = list(styleList)
    noStndDescriptor = []
    noCloneDescriptor = []
    for s in noProxyList:
        noStndDescriptor.extend(DescriptorMap[s])
        noCloneDescriptor.extend(DescriptorMap[s])

    # Setting up market intercept if relevant
    interceptFactor = 'Market Intercept'
    intercept = ModelFactor(interceptFactor, interceptFactor)
    industryClassification = Classification.GICSCustomJP(datetime.date(2016,9,1))

    estuAssetTypes = ['REIT', 'Com']

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.JPAxioma2017MH')
        # Set up relevant styles to be created/used
        ModelParameters2017.defaultExposureParameters(self, self.styleList, descriptorNumeraire='USD')
        EquityModel.FundamentalModel.__init__(self, ['SEDOL'], modelDB, marketDB)
        self.elig_parameters.update({'assetTypes': self.commonStockTypes + ['REIT']}) # refering to parent class input!
        
        # Set Calulators
        self.setCalculators(modelDB)

        # Set up standardization parameters
        gloScope = Standardization_V4.GlobalRelativeScope([f.name for f in self.styles])
        self.exposureStandardization = Standardization_V4.BucketizedStandardization([gloScope],
                fillWithZeroList=self.fillWithZeroList)

        # Set up descriptor standardization parameters
        descriptors = sorted(list(set([item for sublist
            in list(self.DescriptorMap.values()) for item in sublist])))
        self.descriptorStandardization = Standardization_V4.BucketizedStandardization(
                    [Standardization_V4.GlobalRelativeScope(descriptors)],
                    mad_bound=15.0, exceptionNames=self.noStndDescriptor)

        modelDB.createCurrencyCache(marketDB)
    
    def setCalculators(self, modelDB, overrider=False):
        # Set important model parameters
        ModelParameters2017.defaultModelSettingsSCM(self)
        # Run risk based on the square root of market cap weights
        self.internalCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType='Industry Groups',
                dummyThreshold=self.dummyThreshold,
                marketRegression=False,
                kappa=self.k,
                useRealMCaps=True,
                regWeight = 'rootCap',
                overrider=overrider,
                )

        # Set up external regression parameters
        self.returnCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType='Industry Groups',
                dummyThreshold=self.dummyThreshold,
                marketRegression=False,
                kappa=None,
                useRealMCaps=True,
                regWeight='rootCap',
                overrider=overrider
                )

        # This controls the FMP regression
        self.fmpCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType=None,
                marketRegression=False,
                dummyThreshold=self.dummyThreshold,
                kappa=None,
                useRealMCaps=True,
                regWeight='rootCap',
                overrider=overrider,
                )

        # And this is for raw exposure proxying
        self.expProxyCalculator = ModelParameters2017.defaultExposureProxyParameters(
                self, modelDB,
                dummyType='market',
                dummyThreshold = self.dummyThreshold,
                kappa=5.0,
                useRealMCaps=True,
                regWeight='rootCap',
                overrider=overrider,
                )

        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=2, overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=2, overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, maxHistoryOverride=250, overrider=overrider)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

    def generate_model_specific_exposures(self, modelDate, assetData, exposureMatrix, modelDB, marketDB):
        return self.generate_cap_bucket_exposures(modelDate, assetData, exposureMatrix, modelDB, marketDB)

class JPAxioma2017MH_S(EquityModel.StatisticalModel):
    """JP4 statistical model - 2017"""

    # Model Parameters:
    rm_id = 211
    revision = 1
    rms_id = 211

    numFactors = 15
    blind = [ModelFactor('Statistical Factor %d' % n,'Statistical Factor %d' % n) for n in range(1, numFactors+1)]
    pcaHistory = 250
    wideCloneList = list([s.name for s in blind])
    industryClassification = Classification.GICSCustomJP(datetime.date(2016,9,1))
    descriptorNumeraire = 'USD'

    elig_parameters = {'HomeCountry_List': ['JP'],
                       'use_isin_country_Flag': True,
                       'excludeTypes': None}

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.JPAxioma2017MH_S')
        EquityModel.StatisticalModel.__init__(self, ['SEDOL'], modelDB, marketDB)
        self.elig_parameters.update({'assetTypes': self.commonStockTypes + ['REIT']}) # refering to parent class input!
        # So we can use the same ESTU method as the fundamental model
        self.baseModelDateMap = {datetime.date(1980,1,1): JPAxioma2017MH(modelDB, marketDB)}
        # Set Calculators
        self.setCalculators(modelDB)

    def setCalculators(self, modelDB, overrider=False):
        # Set important model parameters
        ModelParameters2017.defaultModelSettingsSCM(self, statModel=True)
        # Set up returns model
        self.returnCalculator = FactorReturns.AsymptoticPrincipalComponents2017(
                self.numFactors, trimExtremeExposures=False, replaceReturns=False, flexible=False, TOL=None)
        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=1, overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=1, overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, nwLag=1, maxHistoryOverride=250, overrider=overrider)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

class JPAxioma2017SH(JPAxioma2017MH):
    """JP4 - 2017 short term model """

    # Model Parameters:
    rm_id = 212
    revision = 1
    rms_id = 212

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
            'Volatility': ['Volatility_60D'],
            'Medium-Term Momentum': ['Momentum_250x20D'],
            'Short-Term Momentum': ['Momentum_20D'],
            'Exchange Rate Sensitivity': ['XRate_52W_USD'],
            'Profitability': ['Return_on_Equity_Annual', 'Return_on_Assets_Annual',
                              'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
                              'Sales_to_Assets_Annual', 'Gross_Margin_Annual']
            }
    shrinkList = {'Liquidity': 20,
              'Market Sensitivity': 125,
              'Volatility': 60,
              'Short-Term Momentum': 20,
              'Exchange Rate Sensitivity': 250,
              'Medium-Term Momentum': 250}
    wideCloneList = list(styleList)

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.JPAxioma2017SH')
        self.setCalculators(modelDB)
        super(JPAxioma2017SH,self).__init__(modelDB, marketDB)
    
    def setCalculators(self, modelDB, overrider=False):
        # Set important model parameters
        super(JPAxioma2017SH, self).setCalculators(modelDB, overrider=overrider)
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=2, horizon='short', dva_downweightEnds=False, overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=2, horizon='short', dva_downweightEnds=False, overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, horizon='short', maxHistoryOverride=250, overrider=overrider)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

class JPAxioma2017SH_S(JPAxioma2017MH_S):
    """JP4 - 2017 short term statistical model """

    # Model Parameters:
    rm_id = 213
    revision = 1
    rms_id = 213

    numFactors = 15
    blind = [ModelFactor('Statistical Factor %d' % n,'Statistical Factor %d' % n) for n in range(1, numFactors+1)]
    returnHistory = 250
    industryClassification = Classification.GICSCustomJP(datetime.date(2016,9,1))

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.JPAxioma2017SH_S')
        EquityModel.StatisticalModel.__init__(self, ['SEDOL'], modelDB, marketDB)
        self.elig_parameters.update({'assetTypes': self.commonStockTypes + ['REIT']}) # refering to parent class input!
        # So we can use the same ESTU method as the fundamental model
        self.baseModelDateMap = {datetime.date(1980,1,1): JPAxioma2017MH(modelDB, marketDB)}
        # Set Calculators
        self.setCalculators(modelDB)
    
    def setCalculators(self, modelDB, overrider=False):
        # Set important model parameters
        ModelParameters2017.defaultModelSettingsSCM(self, statModel=True)
        # Set up returns model
        self.returnCalculator = FactorReturns.AsymptoticPrincipalComponents2017(
                self.numFactors, trimExtremeExposures=False, replaceReturns=False, flexible=False, TOL=None)
        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=1, horizon='short', overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=1, horizon='short', overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, nwLag=1, horizon='short',
                    maxHistoryOverride=250, overrider=overrider)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

"""CN models go here
"""
class CNAxioma2018MH(EquityModel.FundamentalModel):
    """
        CN4 fundamental medium-horizon model
    """
    # Model Parameters:
    rm_id,revision,rms_id = [262,1,262]

    #market factor
    intercept = ModelFactor('Market Intercept', 'Market Intercept')
    localStructureFactors = [ModelFactor('OffShore China', 'OffShore China')]
    # Industry Structure
    # (1) consider GICS - first
    gicsDate = datetime.date(2018,9,29)
    industryClassification = Classification.GICSCustomCN2b(gicsDate)
    #style factor
    styleList = ['Value',
                 'Leverage',
                 'Growth',
                 'Profitability',
                 'Dividend Yield',
                 'Size',
                 'Liquidity',
                 'Market Sensitivity',
                 'Volatility',
                 'Medium-Term Momentum',
                 'Exchange Rate Sensitivity',
                ]
    DescriptorMap = {
            'Value': ['Book_to_Price_Annual','Earnings_to_Price_Annual', 'Est_Earnings_to_Price_12MFL_Annual'],
            'Leverage': ['Debt_to_Assets_Annual', 'Debt_to_Equity_Annual'],
            'Growth': ['Earnings_Growth_RPF_Annual', 'Sales_Growth_RPF_Annual'],
            'Dividend Yield': ['Dividend_Yield_Annual'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_60D','Amihud_Liquidity_Adj_125D', 'ISC_Ret_Score'],
            'Market Sensitivity': ['Market_Sensitivity_XC_104W'],
            'Volatility': ['Volatility_CN_125D'],
            'Medium-Term Momentum': ['Momentum_250x20D'],
            'Exchange Rate Sensitivity': ['XRate_104W_XDR'],# XRate_104W_USD or XRate_104W_XDR
            'Profitability': ['Return_on_Equity_Annual','Return_on_Assets_Annual', 'CashFlow_to_Assets_Annual',
                              'CashFlow_to_Income_Annual','Sales_to_Assets_Annual', 'Gross_Margin_Annual'],
            }
    DescriptorWeights = {'Value': [0.5, 0.375, 0.125]}
    allowMixedFreqDescriptors = False
    ################################################################################################
    # estu related
    elig_parameters = {'HomeCountry_List': ['CN'],
                       'use_isin_country_Flag': False,
                       'assetTypes':['AShares'], # assetTypes for main estu
                       'excludeTypes': None,
                       'remove_China_AB':False,
                       'addBack_H_DR':False}
    estu_parameters = {
                       'minNonMissing':0.5,
                       'CapByNumber_Flag':True,
                       'CapByNumber_hiCapQuota':300,
                       'CapByNumber_lowCapQuota':500,
                       'market_lower_pctile': np.nan,
                       'country_lower_pctile': np.nan,
                       'industry_lower_pctile': np.nan,
                       'dummyThreshold': 6,
                       'inflation_cutoff':0.03,
                       'statModelEstuTol': 0.85
                      }
    estu_parameters_ChinaOff = {
                               'minGoodReturns':0.85,
                               'cap_lower_pctile':5
                                }
    # exposure related
    exposureConfigFile = 'exposures-CNAxioma2018MH'
    orthogList = {'Volatility': [['Market Sensitivity'], True, 1.0],}

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.CNAxioma2018MH')
        # update parameter using parent class value
        self.elig_parameters['excludeTypes'] = [x for x in self.allAssetTypes if x != 'AShares']# all but Ashares for main estu
        # Set up relevant styles to be created/used
        ModelParameters2017.defaultExposureParameters(self, self.styleList, configFile=self.exposureConfigFile,descriptorNumeraire='USD') #USD for numeraire descriptors
        EquityModel.FundamentalModel.__init__(self, ['SEDOL'], modelDB, marketDB)
        # Set up setCalculators - regression parameters
        self.setCalculators(modelDB)
        # Set up standardization parameters
        gloScope = Standardization_V4.GlobalRelativeScope([f.name for f in self.styles])
        self.exposureStandardization = Standardization_V4.BucketizedStandardization([gloScope], fillWithZeroList=self.fillWithZeroList)
        modelDB.createCurrencyCache(marketDB)
        # Set up descriptor standardization parameters
        descriptors = sorted(list(set([item for sublist in self.DescriptorMap.values() for item in sublist])))
        self.descriptorStandardization = Standardization_V4.BucketizedStandardization(
                [Standardization_V4.GlobalRelativeScope(descriptors)],
                mad_bound=15.0, exceptionNames=self.noStndDescriptor)

    def setCalculators(self, modelDB, overrider=False):
        # Set important model parameters
        ModelParameters2017.defaultModelSettings(self,scm=False) # single country with regional framework
        # model customize setting
        self.coverageMultiCountry = True
        self.hasCountryFactor = False
        self.hasCurrencyFactor = False
        self.applyRT2US = False
        self.useFreeFloatRegWeight = True

        #internal factor return
        self.internalCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,dummyType='Sectors',
                dummyThreshold=self.estu_parameters['dummyThreshold'],
                marketRegression=False,
                scndRegList=[[ExposureMatrix.LocalFactor]],
                scndRegEstus=['ChinaOff'],
                kappa = 5.0,
                useRealMCaps=True, regWeight='rootCap', overrider=overrider)
        #external factor return
        self.returnCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,dummyType='Sectors',
                dummyThreshold=self.estu_parameters['dummyThreshold'],
                marketRegression=False,
                scndRegList=[[ExposureMatrix.LocalFactor]],
                scndRegEstus=['ChinaOff'],
                kappa = None,
                useRealMCaps=True, regWeight='rootCap', overrider=overrider)
        # This controls the FMP regression
        self.fmpCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,dummyType=None,
                dummyThreshold=self.estu_parameters['dummyThreshold'],
                marketRegression=False,
                scndRegList=[[ExposureMatrix.LocalFactor]],
                scndRegEstus=['ChinaOff'],
                kappa=None,
                useRealMCaps=True, regWeight='rootCap', overrider=overrider)
        # And this is for raw exposure proxying
        self.expProxyCalculator = ModelParameters2017.defaultExposureProxyParameters(
                self, modelDB,
                dummyType='market',
                dummyThreshold = self.estu_parameters['dummyThreshold'],
                kappa=5.0,
                useRealMCaps=True,
                regWeight='rootCap', overrider=overrider)
        self.setRiskParameters(overrider)
    
    def setRiskParameters(self, overrider):
        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=3,horizon='medium', overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=3,horizon='medium', overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self,horizon='medium', overrider=overrider)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

    def generate_model_specific_exposures(self, modelDate, assetData, exposureMatrix, modelDB, marketDB):
        """Generate OffShore China local factor.
        """
        logging.info('Building OffShore China Exposures')
        values = pandas.Series(numpy.nan, index=assetData.universe)

        estuCls = EstimationUniverse_V4.ConstructEstimationUniverse(\
                modelDate, assetData.universe, self, modelDB, marketDB, debugOutput=self.debuggingReporting)
        OffShoreShares = estuCls.exclude_by_asset_type(
                assetData, includeFields=None, excludeFields=self.localChineseAssetTypes)

        OffShoreShares = OffShoreShares.difference(assetData.getSPACs())
        if len(OffShoreShares) > 0:
            logging.info('Assigning OffShore China exposure to %d assets', len(OffShoreShares))
            values[OffShoreShares] = 1.0
        exposureMatrix.addFactor('OffShore China', values, ExposureMatrix.LocalFactor)

        return exposureMatrix

class CNAxioma2018SH(CNAxioma2018MH):
    """
        CN4 fundamental short-horizon model
    """
    # Model Parameters:
    rm_id,revision,rms_id = [263,1,263]

    #style factor
    styleList = ['Value',
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
                 'Exchange Rate Sensitivity',
                ]
    DescriptorMap = {
            'Value': ['Book_to_Price_Annual','Earnings_to_Price_Annual', 'Est_Earnings_to_Price_12MFL_Annual'],
            'Leverage': ['Debt_to_Assets_Annual', 'Debt_to_Equity_Annual'],
            'Growth': ['Earnings_Growth_RPF_Annual', 'Sales_Growth_RPF_Annual'],
            'Dividend Yield': ['Dividend_Yield_Annual'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_20D','Amihud_Liquidity_Adj_60D', 'ISC_Ret_Score'],
            'Market Sensitivity': ['Market_Sensitivity_XC_52W'],
            'Volatility': ['Volatility_CN_60D'],
            'Medium-Term Momentum': ['Momentum_250x20D'],
            'Short-Term Momentum': ['Momentum_20D'],
            'Exchange Rate Sensitivity': ['XRate_52W_XDR'],
            'Profitability': ['Return_on_Equity_Annual','Return_on_Assets_Annual', 'CashFlow_to_Assets_Annual',
                              'CashFlow_to_Income_Annual','Sales_to_Assets_Annual', 'Gross_Margin_Annual'],
            }
    exposureConfigFile = 'exposures-CNAxioma2018SH'
    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.CNAxioma2018SH')
        super(CNAxioma2018SH,self).__init__(modelDB, marketDB)

    def setRiskParameters(self, overrider):
        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=3,horizon='short', overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=3,horizon='short', overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self,horizon='short', overrider=overrider)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

class CNAxioma2018MH_S(EquityModel.StatisticalModel):
    """
        CN4 statistical medium-horizon model
    """
    # Model Parameters:
    rm_id,revision,rms_id = [264,1,264]

    numFactors = 15
    blind = [ModelFactor('Statistical Factor %d' % n, 'Statistical Factor %d' % n) for n in range(1, numFactors+1)]
    pcaHistory = 250
    gicsDate = datetime.date(2018,9,29)
    industryClassification = Classification.GICSCustomCN2b(gicsDate)
    descriptorNumeraire = 'USD'

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.CNAxioma2018MH_S')
        EquityModel.StatisticalModel.__init__(self, ['SEDOL', 'CUSIP'], modelDB, marketDB)
        # So we can use the same ESTU method as the fundamental model
        self.baseModelDateMap = {datetime.date(1980,1,1): CNAxioma2018MH(modelDB, marketDB)}
        # Set up estimation universe parameters
        self.elig_parameters = list(self.baseModelDateMap.values())[0].elig_parameters.copy()
        # Set Calculators
        self.setCalculators(modelDB)

    def setCalculators(self, modelDB, overrider=False):
        # Set important model parameters
        ModelParameters2017.defaultModelSettings(self, scm=False, statModel=True)
        # model customize setting
        self.coverageMultiCountry = True
        self.hasCountryFactor = False
        self.hasCurrencyFactor = False
        self.applyRT2US = False
        self.useFreeFloatRegWeight = True
        
        # Set up returns model
        self.returnCalculator = FactorReturns.AsymptoticPrincipalComponents2017(self.numFactors, trimExtremeExposures=True)
        self.olsReturnClass = ModelParameters2017.simpleRegressionParameters(self, modelDB, overrider=overrider)
        self.setRiskParameters(overrider)
    
    def setRiskParameters(self, overrider):
        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=1, overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=1, overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, nwLag=1, overrider=overrider)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

class CNAxioma2018SH_S(CNAxioma2018MH_S):
    """
        CN4 statistical short-horizon model
    """
    # Model Parameters:
    rm_id,revision,rms_id = [265,1,265]

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.CNAxioma2018SH_S')
        super(CNAxioma2018SH_S,self).__init__(modelDB, marketDB)
        self.setCalculators(modelDB)

    def setRiskParameters(self, overrider):
        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=1, horizon='short', overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=1, horizon='short', overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, nwLag=1,horizon='short', overrider=overrider)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

class CNAxioma2018FL(CNAxioma2018MH):
    """CN4 factor model
    """
    rm_id = 288
    revision = 1
    rms_id = 288

    DescriptorMap = {
            'Book to Price Annual': ['Book_to_Price_Annual'],
            'Earnings to Price Annual': ['Earnings_to_Price_Annual'],
            'Est Earnings to Price Annual': ['Est_Earnings_to_Price_12MFL_Annual'],
            'Debt to Assets Annual': ['Debt_to_Assets_Annual'],
            'Debt to Equity Annual': ['Debt_to_Equity_Annual'],
            'Earnings Growth Annual': ['Earnings_Growth_RPF_Annual'],
            'Sales Growth Annual': ['Sales_Growth_RPF_Annual'],
            'Dividend Yield Annual': ['Dividend_Yield_Annual'],
            'Log of Issuer Cap': ['LnIssuerCap'],
            'Log of 60-Day ADV to Issuer Cap': ['LnTrading_Activity_60D'],
            'Log of 20-Day ADV to Issuer Cap': ['LnTrading_Activity_20D'],
            'Amihud Liquidity 60 Day': ['Amihud_Liquidity_60D'],
            'Amihud Liquidity 125 Day': ['Amihud_Liquidity_125D'],
            'Returns to Trading Days': ['ISC_Ret_Score'],
            'Domestic China Market Sensitivity 104 Week': ['Market_Sensitivity_XC_104W'],
            'Domestic China Market Sensitivity 52 Week': ['Market_Sensitivity_XC_52W'],
            'Domestic China Volatility 60 Day': ['Volatility_CN_125D'],
            'Domestic China Volatility 125 Day': ['Volatility_CN_60D'],
            'Annual Return Excl Prev Month': ['Momentum_260x21D_Regional'],
            'Monthly Return': ['Momentum_21D'],
            'Exchange Rate Sensitivity 52 Week XDR': ['XRate_52W_XDR'],
            'Exchange Rate Sensitivity 104 Week XDR': ['XRate_104W_XDR'],
            'Return on Assets Annual': ['Return_on_Assets_Annual'],
            'Return on Equity Annual': ['Return_on_Equity_Annual'],
            'Cash Flow to Assets Annual': ['CashFlow_to_Assets_Annual'],
            'Cash Flow to Income Annual': ['CashFlow_to_Income_Annual'],
            'Sales to Assets Annual': ['Sales_to_Assets_Annual'],
            'Gross Margin Annual': ['Gross_Margin_Annual'],
            }

    styleList = sorted(DescriptorMap.keys())

    DescriptorWeights = {}
    orthogList = {}
    exposureConfigFile = 'exposures-CNAxioma2018FL'

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.CNAxioma2018FL')
        CNAxioma2018MH.__init__(self, modelDB, marketDB)
        self.noClip_List = ['ISC_Ret_Score','Returns to Trading Days']
        self.allowMixedFreqDescriptors = False

class UKAxioma2018MH(EquityModel.FundamentalModel):
    """Version 4 UK medium-horizon fundamental model
    """
    rm_id = 310
    revision = 1
    rms_id = 310

    styleList = ['Value',
                 'Leverage',
                 'Growth',
                 'Profitability',
                 'Earnings Yield',
                 'Dividend Yield',
                 'Size',
                 'MidCap',
                 'Liquidity',
                 'Market Sensitivity',
                 'Volatility',
                 'Medium-Term Momentum',
                 'Exchange Rate Sensitivity',
                 'Investment Trusts',
                 ]

    intercept = ModelFactor('Market Intercept', 'Market Intercept')
    smallCapMap = {'MidCap': [72.0, 86.0],}
    gicsDate = datetime.date(2018,9,29)
    industryClassification = Classification.GICSCustomGB4(gicsDate)

    DescriptorMap = {
            'Earnings Yield': ['Earnings_to_Price_Annual', 'Est_Earnings_to_Price_12MFL_Annual'],
            'Value': ['Book_to_Price_Annual'],
            'Leverage': ['Debt_to_Assets_Annual', 'Debt_to_Equity_Annual'],
            'Growth': ['Earnings_Growth_RPF_Annual', 'Sales_Growth_RPF_Annual'],
            'Dividend Yield': ['Dividend_Yield_Annual'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_60D', 'Amihud_Liquidity_125D', 'ISC_Ret_Score'],
            'Market Sensitivity': ['Market_Sensitivity_104W'],
            'Volatility': ['UK_Volatility_125D'],
            'Medium-Term Momentum': ['Momentum_260x21D_Regional'],
            'Exchange Rate Sensitivity': ['XRate_104W_XDR'],
            'Profitability': ['Return_on_Equity_Annual', 'Return_on_Assets_Annual',
                'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
                'Sales_to_Assets_Annual', 'Gross_Margin_Annual'],
            }

    DescriptorWeights = {'Earnings Yield': [0.75, 0.25],}
    allowMixedFreqDescriptors = False
    orthogList = {'Volatility': [['Market Sensitivity'], True, 1.0],}
    exposureConfigFile = 'exposures-UK-mh'

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.UKAxioma2017MH')

        # Set important model parameters
        ModelParameters2017.defaultModelSettings(self, scm=True)

        # Set up relevant styles to be created/used
        ModelParameters2017.defaultExposureParameters(
                self, self.styleList, configFile=self.exposureConfigFile, descriptorNumeraire='USD')

        # Initialise
        EquityModel.FundamentalModel.__init__(self, ['SEDOL', 'CUSIP'], modelDB, marketDB)

        # Set up estimation universe parameters
        self.elig_parameters = {'assetTypes': self.commonStockTypes + ['REIT'] + ['InvT'],
                                'excludeTypes': None,
                                'use_isin_country_Flag': False,
                                'remove_China_AB': True,
                                'addBack_H_DR': False}
        self.estu_parameters = {'minNonMissing': 0.5,
                                'ADV_percentile': [5, 100],
                                'CapByNumber_Flag': False,
                                'CapByNumber_hiCapQuota': np.nan,
                                'CapByNumber_lowCapQuota': np.nan,
                                'market_lower_pctile': 2.0,
                                'country_lower_pctile': 2.0,
                                'industry_lower_pctile': 2.0,
                                'dummyThreshold': 6,
                                'inflation_cutoff': 0.05,
                                }

        # Set up RiskModel Calculators
        self.setCalculators(modelDB)

        # Set up descriptor standardization parameters
        descriptors = sorted(list(set([item for sublist in self.DescriptorMap.values() for item in sublist])))
        gloScope = Standardization_V4.GlobalRelativeScope([d for d in descriptors])
        self.descriptorStandardization = Standardization_V4.BucketizedStandardization(
                [gloScope], mad_bound=15.0,  exceptionNames=self.noStndDescriptor)

        # Set up standardization parameters
        gloScope = Standardization_V4.GlobalRelativeScope([f.name for f in self.styles])
        self.exposureStandardization = Standardization_V4.BucketizedStandardization([gloScope], fillWithZeroList=self.fillWithZeroList)

    def generate_model_specific_exposures(self, modelDate, assetData, exposureMatrix, modelDB, marketDB):
        """Generate UK-specific exposures
        """
        logging.info('Building Investment Trust Exposures')
        estuCls = EstimationUniverse_V4.ConstructEstimationUniverse(
                modelDate, assetData.universe, self, modelDB, marketDB, debugOutput=self.debuggingReporting)
        invT = estuCls.exclude_by_asset_type(assetData, includeFields=['InvT'], excludeFields=None)
        values = pandas.Series(numpy.nan, index=assetData.universe)
        invT = invT.difference(assetData.getSPACs())
        if len(invT) > 0:
            logging.info('Assigning Investment Trust exposure to %d assets', len(invT))
            values[invT] = 1.0
        exposureMatrix.addFactor('Investment Trusts', values, ExposureMatrix.StyleFactor)

        # Add to non-standardisation list
        if self.exposureStandardization.exceptionNames is None:
            self.exposureStandardization.exceptionNames = ['Investment Trusts']
        else:
            self.exposureStandardization.exceptionNames.append('Investment Trusts')

        return self.generate_cap_bucket_exposures(modelDate, assetData, exposureMatrix, modelDB, marketDB)

    def setCalculators(self, modelDB, overrider=False):
        # Set up internal factor return regression parameters
        dummyThreshold = 6
        self.internalCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType='Sectors',
                dummyThreshold=dummyThreshold,
                marketRegression=False,
                kappa=5.0,
                useRealMCaps=True,
                regWeight='rootCap',
                thinWeightMultiplier='simple',
                overrider = overrider,
                )

        # Set up external regression parameters
        self.returnCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType='Sectors',
                dummyThreshold = dummyThreshold,
                marketRegression=False,
                kappa=25.0,
                useRealMCaps=True,
                regWeight='rootCap',
                thinWeightMultiplier='simple',
                overrider = overrider,
                )

        # This controls the FMP regression
        self.fmpCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType=None,
                dummyThreshold = dummyThreshold,
                marketRegression=False,
                kappa=None,
                useRealMCaps=True,
                regWeight='rootCap',
                overrider = overrider,
                )

        # And this is for raw exposure proxying
        self.expProxyCalculator = ModelParameters2017.defaultExposureProxyParameters(
                self, modelDB,
                dummyType='market',
                dummyThreshold = dummyThreshold,
                kappa=5.0,
                useRealMCaps=True,
                regWeight='rootCap',
                thinWeightMultiplier='simple',
                overrider = overrider,
                )

        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=3, overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=3, overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, overrider=overrider)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

class UKAxioma2018SH(UKAxioma2018MH):
    """
        UK4 fundamental short-horizon model
    """
    # Model Parameters:
    rm_id,revision,rms_id = [311,1,311]

    styleList = ['Value',
                 'Leverage',
                 'Growth',
                 'Profitability',
                 'Earnings Yield',
                 'Dividend Yield',
                 'Size',
                 'MidCap',
                 'Liquidity',
                 'Market Sensitivity',
                 'Volatility',
                 'Medium-Term Momentum',
                 'Short-Term Momentum',
                 'Exchange Rate Sensitivity',
                 'Investment Trusts',
                 ]

    DescriptorMap = {
            'Earnings Yield': ['Earnings_to_Price_Annual', 'Est_Earnings_to_Price_12MFL_Annual'],
            'Value': ['Book_to_Price_Annual'],
            'Leverage': ['Debt_to_Assets_Annual', 'Debt_to_Equity_Annual'],
            'Growth': ['Earnings_Growth_RPF_Annual', 'Sales_Growth_RPF_Annual'],
            'Dividend Yield': ['Dividend_Yield_Annual'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_20D', 'Amihud_Liquidity_60D', 'ISC_Ret_Score'],
            'Market Sensitivity': ['Market_Sensitivity_52W'],
            'Volatility': ['UK_Volatility_60D'],
            'Medium-Term Momentum': ['Momentum_260x21D_Regional'],
            'Short-Term Momentum': ['Momentum_21D'],
            'Exchange Rate Sensitivity': ['XRate_52W_XDR'],
            'Profitability': ['Return_on_Equity_Annual', 'Return_on_Assets_Annual',
                'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
                'Sales_to_Assets_Annual', 'Gross_Margin_Annual'],
            }

    exposureConfigFile = 'exposures-UK-sh'

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.UKAxioma2018SH')
        UKAxioma2018MH.__init__(self, modelDB, marketDB)

    def setCalculators(self, modelDB, overrider=False):
        UKAxioma2018MH.setCalculators(self, modelDB, overrider=overrider)
        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=3, horizon='short', overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=3, horizon='short', overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, horizon='short', overrider=overrider)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

class UKAxioma2018MH_S(EquityModel.StatisticalModel):
    """Version 4 UK medium-horizon statistical model with GICS 2018
    """
    rm_id = 312
    revision = 1
    rms_id = 312
    numFactors = 15
    blind = [ModelFactor('Statistical Factor %d' % n, 'Statistical Factor %d' % n) for n in range(1, numFactors+1)]
    pcaHistory = 250
    gicsDate = datetime.date(2018,9,29)
    industryClassification = Classification.GICSCustomGB4(gicsDate)
    descriptorNumeraire = 'USD'

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.UKAxioma2018MH_S')
        # Set important model parameters
        ModelParameters2017.defaultModelSettings(self, scm=True, statModel=True)
        EquityModel.StatisticalModel.__init__(self, ['SEDOL', 'CUSIP'], modelDB, marketDB)
        # So we can use the same ESTU method as the fundamental model
        self.baseModelDateMap = {datetime.date(1980,1,1): UKAxioma2018MH(modelDB, marketDB)}
        # Set up estimation universe parameters
        self.elig_parameters = {'assetTypes': self.commonStockTypes + ['REIT'] + ['InvT'],
                                'excludeTypes': None,
                                'use_isin_country_Flag': False,
                                'remove_China_AB': True,
                                'addBack_H_DR': False}
        # Set Calculators
        self.setCalculators(modelDB)

    def setCalculators(self, modelDB, overrider=False):
        # Set up returns model
        self.returnCalculator = FactorReturns.AsymptoticPrincipalComponents2017(self.numFactors, trimExtremeExposures=True)
        self.olsReturnClass = ModelParameters2017.simpleRegressionParameters(self, modelDB, overrider=overrider)
        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=1, overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=1, overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, nwLag=1, overrider=overrider)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

class UKAxioma2018SH_S(UKAxioma2018MH_S):
    """Version 4 UK short-horizon statistical model with GICS 2018
    """
    rm_id = 313
    revision = 1
    rms_id = 313

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.UKAxioma2018SH_S')
        # Set important model parameters
        UKAxioma2018MH_S.__init__(self, modelDB, marketDB)

    def setCalculators(self, modelDB, overrider=False):
        # Set up returns model
        self.returnCalculator = FactorReturns.AsymptoticPrincipalComponents2017(self.numFactors, trimExtremeExposures=True)
        self.olsReturnClass = ModelParameters2017.simpleRegressionParameters(self, modelDB, overrider=overrider)
        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=1, horizon='short', overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=1, horizon='short', overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, nwLag=1, horizon='short', overrider=overrider)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

class UKAxioma2018FL(UKAxioma2018MH):
    """UK4 factor library
    """
    rm_id = 314
    revision = 1
    rms_id = 314

    DescriptorMap = {
            'Amihud Liquidity 60 Day': ['Amihud_Liquidity_60D'],
            'Amihud Liquidity 125 Day': ['Amihud_Liquidity_125D'],
            'Book to Price Annual': ['Book_to_Price_Annual'],
            'Cash Flow to Assets Annual': ['CashFlow_to_Assets_Annual'],
            'Cash Flow to Income Annual': ['CashFlow_to_Income_Annual'],
            'Debt to Assets Annual': ['Debt_to_Assets_Annual'],
            'Debt to Equity Annual': ['Debt_to_Equity_Annual'],
            'Dividend Yield Annual': ['Dividend_Yield_Annual'],
            'Earnings to Price Annual': ['Earnings_to_Price_Annual'],
            'Est Earnings to Price Annual': ['Est_Earnings_to_Price_12MFL_Annual'],
            'Earnings Growth Annual': ['Earnings_Growth_RPF_Annual'],
            'Gross Margin Annual': ['Gross_Margin_Annual'],
            'Returns to Trading Days': ['ISC_Ret_Score'],
            'Log of Issuer Cap': ['LnIssuerCap'],
            'Log of 20-Day ADV to Issuer Cap': ['LnTrading_Activity_20D'],
            'Log of 60-Day ADV to Issuer Cap': ['LnTrading_Activity_60D'],
            'Annual Return Excl Prev Month': ['Momentum_260x21D_Regional'],
            'Monthly Return': ['Momentum_21D'],
            'Market Sensitivity 250 Day': ['Market_Sensitivity_52W'],
            'Market Sensitivity 500 Day': ['Market_Sensitivity_104W'],
            'Return on Assets Annual': ['Return_on_Assets_Annual'],
            'Return on Equity Annual': ['Return_on_Equity_Annual'],
            'Sales to Assets Annual': ['Sales_to_Assets_Annual'],
            'Sales Growth Annual': ['Sales_Growth_RPF_Annual'],
            'Volatility 60 Day': ['UK_Volatility_60D'],
            'Volatility 125 Day': ['UK_Volatility_125D'],
            'Exchange Rate Sensitivity 52 Week XDR': ['XRate_52W_XDR'],
            'Exchange Rate Sensitivity 104 Week XDR': ['XRate_104W_XDR'],
            }

    styleList = sorted(list(DescriptorMap.keys()) + ['MidCap', 'Investment Trusts'])

    DescriptorWeights = {}
    orthogList = {}
    exposureConfigFile = 'exposures-UK-fl'

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.UKAxioma2018FL')
        UKAxioma2018MH.__init__(self, modelDB, marketDB)
        self.noClip_List = ['ISC_Ret_Score','Returns to Trading Days']
        self.allowMixedFreqDescriptors = False

class CAAxioma2018MH(EquityModel.FundamentalModel):
    """Version 4 CA medium-horizon fundamental model
    """
    rm_id = 320
    revision = 1
    rms_id = 320

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
                 'Exchange Rate Sensitivity',
                 'Residual Gold Sensitivity',
                 'Residual Oil Sensitivity',
                ]

    interceptFactor = 'Market Intercept'
    intercept = ModelFactor(interceptFactor, interceptFactor)
   
    smallCapMap = {}
    
    gicsDate = datetime.date(2018,9,29)
    industryClassification = Classification.GICSCustomCA4(
            gicsDate)
    
    # Note that CAAxioma2018MH uses quarterly descriptor data, filling in missing
    # values with annual descriptor data. _Annual descriptor names
    # are referenced in the DescriptorMap dict below for historical reasons 
    # (the first pass of the mixed frequency code required _Annual names)
    DescriptorMap = {
            'Earnings Yield': ['Earnings_to_Price_Annual', 'Est_Earnings_to_Price_12MFL_Annual'],
            'Value': ['Book_to_Price_Annual'],
            'Leverage': ['Debt_to_Assets_Annual', 'Debt_to_Equity_Annual'],
            'Growth': ['Earnings_Growth_RPF_Annual', 'Sales_Growth_RPF_Annual'],
            'Dividend Yield': ['Dividend_Yield_Annual'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_60D', 'Amihud_Liquidity_125D', 'ISC_Ret_Score'],
            'Market Sensitivity': ['Market_Sensitivity_104W'],
            'Volatility': ['CA_Volatility_125D'],
            'Medium-Term Momentum': ['Momentum_260x21D_Regional'],
            'Exchange Rate Sensitivity': ['XRate_104W_XDR'],
            'Profitability': ['Return_on_Equity_Annual', 'Return_on_Assets_Annual',
                'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
                'Sales_to_Assets_Annual', 'Gross_Margin_Annual'],
            'Residual Gold Sensitivity': ['CAGold_Sensitivity_NetOfSecBeta_104W'],
            'Residual Oil Sensitivity': ['CAOil_Sensitivity_NetOfSecBeta_104W'],
            }

    DescriptorWeights = {'Earnings Yield': [0.75, 0.25]}
   
    orthogList = {'Volatility': [['Market Sensitivity'], True, 1.0]}
    
    exposureConfigFile = 'exposures-CA-mh'

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.CAAxioma2018MH')
        # Set important model parameters
        ModelParameters2017.defaultModelSettings(self, scm=True)

        # Set up relevant styles to be created/used
        ModelParameters2017.defaultExposureParameters(
                self, self.styleList, configFile=self.exposureConfigFile, descriptorNumeraire='USD')

        # Initialize
        EquityModel.FundamentalModel.__init__(self, ['SEDOL', 'CUSIP'], modelDB, marketDB)

        # Set up eligible and estu universe parameters
        self.fundAssetTypes = ['CEFund', 'InvT', 'Misc'] # overwrite default to keep UnitT
        self.elig_parameters = {'assetTypes': self.commonStockTypes + ['REIT'] + ['UnitT'],
                                'excludeTypes': None,
                                'use_isin_country_Flag': False,
                                'remove_China_AB': True,
                                'addBack_H_DR': False,
        }  

        self.estu_parameters = {'minNonMissing':0.5,
                                'CapByNumber_Flag': True,
                                'CapByNumber_hiCapQuota': 250,
                                'CapByNumber_lowCapQuota': 150,
                                'market_lower_pctile': np.nan,
                                'country_lower_pctile': np.nan,
                                'industry_lower_pctile': np.nan,
                                'dummyThreshold': 6,
                                'inflation_cutoff':0.01 
        }

        # Set up RiskModel Calculators
        self.setCalculators(modelDB)

        # Set up descriptor standardization parameters
        descriptors = sorted(list(set([item for sublist in self.DescriptorMap.values() for item in sublist])))
        gloScope = Standardization_V4.GlobalRelativeScope([d for d in descriptors])
        self.descriptorStandardization = Standardization_V4.BucketizedStandardization(
                [gloScope], mad_bound=15.0,  exceptionNames=self.noStndDescriptor)

        # Set up standardization parameters
        gloScope = Standardization_V4.GlobalRelativeScope([f.name for f in self.styles])
        self.exposureStandardization = Standardization_V4.BucketizedStandardization([gloScope], fillWithZeroList=self.fillWithZeroList)

    def setCalculators(self, modelDB, overrider=False):
        # Set up regression parameters
        dummyThreshold = 6  
        
        self.internalCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType='Sectors',
                dummyThreshold=dummyThreshold,
                marketRegression=False,
                kappa=5.0,
                useRealMCaps=True,
                regWeight='rootCap',
                overrider = overrider,
                )

        self.returnCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType='Sectors',
                dummyThreshold=dummyThreshold,
                marketRegression=False,
                kappa=25.0,
                useRealMCaps=True,
                regWeight='rootCap',
                overrider = overrider,
                )

        # This controls the FMP regression
        self.fmpCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType=None,
                marketRegression=False,
                dummyThreshold=dummyThreshold,
                useRealMCaps=True,
                kappa=None,
                regWeight='rootCap',
                overrider = overrider,
                )

        # And this is for raw exposure proxying
        self.expProxyCalculator = ModelParameters2017.defaultExposureProxyParameters(
                self, modelDB,
                dummyType='market',
                dummyThreshold=dummyThreshold,
                kappa=5.0,
                useRealMCaps=True,
                regWeight='rootCap',
                overrider = overrider,
                )

        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=3, overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=3, overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, overrider=overrider)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

class CAAxioma2018SH(CAAxioma2018MH):
    """
        CA4 fundamental short-horizon model
    """
    # Model Parameters:
    rm_id, revision, rms_id = [321, 1, 321]

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
                 'Exchange Rate Sensitivity',
                 'Residual Gold Sensitivity',
                 'Residual Oil Sensitivity',
                ]

    # Note that CAAxioma2018SH uses quarterly descriptor data, filling in missing
    # values with annual descriptor data. _Annual descriptor names
    # are referenced in the DescriptorMap dict below for historical reasons 
    # (the first pass of the mixed frequency code required _Annual names)
    DescriptorMap = {
            'Earnings Yield': ['Earnings_to_Price_Annual', 'Est_Earnings_to_Price_12MFL_Annual'],
            'Value': ['Book_to_Price_Annual'],
            'Leverage': ['Debt_to_Assets_Annual', 'Debt_to_Equity_Annual'],
            'Growth': ['Earnings_Growth_RPF_Annual', 'Sales_Growth_RPF_Annual'],
            'Dividend Yield': ['Dividend_Yield_Annual'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_20D', 'Amihud_Liquidity_60D', 'ISC_Ret_Score'],
            'Market Sensitivity': ['Market_Sensitivity_52W'],
            'Volatility': ['CA_Volatility_60D'],
            'Medium-Term Momentum': ['Momentum_260x21D_Regional'],
            'Short-Term Momentum': ['Momentum_21D'],
            'Exchange Rate Sensitivity': ['XRate_52W_XDR'],
            'Profitability': ['Return_on_Equity_Annual', 'Return_on_Assets_Annual',
                'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
                'Sales_to_Assets_Annual', 'Gross_Margin_Annual'],
            'Residual Gold Sensitivity': ['CAGold_Sensitivity_NetOfSecBeta_52W'],
            'Residual Oil Sensitivity': ['CAOil_Sensitivity_NetOfSecBeta_52W'],
            }

    exposureConfigFile = 'exposures-CA-sh'

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.CAAxioma2018SH')
        CAAxioma2018MH.__init__(self, modelDB, marketDB)

    def setCalculators(self, modelDB, overrider=False):
        CAAxioma2018MH.setCalculators(self, modelDB, overrider=overrider)
        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=3, horizon='short', overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=3, horizon='short', overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, horizon='short', overrider=overrider)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

class CAAxioma2018MH_S(EquityModel.StatisticalModel):
    """Version 4 CA medium-horizon statistical model with GICS 2018
    """
    rm_id = 322
    revision = 1
    rms_id = 322
    numFactors = 15
    blind = [ModelFactor('Statistical Factor %d' % n, 'Statistical Factor %d' % n) for n in range(1, numFactors+1)]
    pcaHistory = 250
    gicsDate = datetime.date(2018, 9, 29)
    industryClassification = Classification.GICSCustomCA4(gicsDate)
    descriptorNumeraire = 'USD'

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.CAAxioma2018MH_S')
        # Set important model parameters
        ModelParameters2017.defaultModelSettings(self, scm=True, statModel=True)
        EquityModel.StatisticalModel.__init__(self, ['SEDOL', 'CUSIP'], modelDB, marketDB)
        # So we can use the same ESTU method as the fundamental model
        self.baseModelDateMap = {datetime.date(1980,1,1): CAAxioma2018MH(modelDB, marketDB)}
        # Set up estimation universe parameters
        self.fundAssetTypes = ['CEFund', 'InvT', 'Misc'] # overwrite default to keep UnitT - DO I NEED THIS?
        self.elig_parameters = {'assetTypes': self.commonStockTypes + ['REIT'] + ['UnitT'],
                                'excludeTypes': None,
                                'use_isin_country_Flag': False,
                                'remove_China_AB': True,
                                'addBack_H_DR': False,
                                }  

        # Set Calculators
        self.setCalculators(modelDB)

    def setCalculators(self, modelDB, overrider=False):
        # Set up returns model
        self.returnCalculator = FactorReturns.AsymptoticPrincipalComponents2017(self.numFactors, trimExtremeExposures=True)
        self.olsReturnClass = ModelParameters2017.simpleRegressionParameters(self, modelDB, overrider=overrider)
        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=1, overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=1, overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, nwLag=1, overrider=overrider)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

class CAAxioma2018SH_S(CAAxioma2018MH_S):
    """Version 4 CA short-horizon statistical model with GICS 2018
    """
    rm_id = 323
    revision = 1
    rms_id = 323

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.CAAxioma2018SH_S')
        # Set important model parameters
        CAAxioma2018MH_S.__init__(self, modelDB, marketDB)

    def setCalculators(self, modelDB, overrider=False):
        # Set up returns model
        self.returnCalculator = FactorReturns.AsymptoticPrincipalComponents2017(self.numFactors, trimExtremeExposures=True)
        self.olsReturnClass = ModelParameters2017.simpleRegressionParameters(self, modelDB, overrider=overrider)
        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=1, horizon='short', overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=1, horizon='short', overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, nwLag=1, horizon='short', overrider=overrider)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

class CAAxioma2018FL(CAAxioma2018MH):
    """CA4 factor library
    """
    rm_id = 324
    revision = 1
    rms_id = 324
    
    # Note that CAAxioma2018FL uses quarterly descriptor data, filling in missing
    # values with annual descriptor data. _Annual descriptor names
    # are referenced in the DescriptorMap dict below for historical reasons 
    # (the first pass of the mixed frequency code required _Annual names)
    DescriptorMap = {
            'Amihud Liquidity 60 Day': ['Amihud_Liquidity_60D'],
            'Amihud Liquidity 125 Day': ['Amihud_Liquidity_125D'],
            'Book to Price Quarterly': ['Book_to_Price_Annual'],
            'Cash Flow to Assets Annual': ['CashFlow_to_Assets_Annual'],
            'Cash Flow to Income Annual': ['CashFlow_to_Income_Annual'],
            'Debt to Assets Quarterly': ['Debt_to_Assets_Annual'],
            'Debt to Equity Quarterly': ['Debt_to_Equity_Annual'],
            'Dividend Yield Quarterly': ['Dividend_Yield_Annual'],
            'Earnings to Price Quarterly': ['Earnings_to_Price_Annual'],
            'Est Earnings to Price Quarterly': ['Est_Earnings_to_Price_12MFL_Annual'],
            'Earnings Growth Quarterly': ['Earnings_Growth_RPF_Annual'],
            'Gross Margin Quarterly': ['Gross_Margin_Annual'],
            'Returns to Trading Days': ['ISC_Ret_Score'],
            'Log of Issuer Cap': ['LnIssuerCap'],
            'Log of 20-Day ADV to Issuer Cap': ['LnTrading_Activity_20D'],
            'Log of 60-Day ADV to Issuer Cap': ['LnTrading_Activity_60D'],
            'Annual Return Excl Prev Month': ['Momentum_260x21D_Regional'],
            'Monthly Return': ['Momentum_21D'],
            'Market Sensitivity 250 Day': ['Market_Sensitivity_52W'],
            'Market Sensitivity 500 Day': ['Market_Sensitivity_104W'],
            'Return on Assets Quarterly': ['Return_on_Assets_Annual'],
            'Return on Equity Quarterly': ['Return_on_Equity_Annual'],
            'Sales to Assets Quarterly': ['Sales_to_Assets_Annual'],
            'Sales Growth Quarterly': ['Sales_Growth_RPF_Annual'],
            'Volatility 60 Day': ['CA_Volatility_60D'],
            'Volatility 125 Day': ['CA_Volatility_125D'],
            'Exchange Rate Sensitivity 52 Week XDR': ['XRate_52W_XDR'],
            'Exchange Rate Sensitivity 104 Week XDR': ['XRate_104W_XDR'],
            'Residual Gold Sensitivity 104 Week': ['CAGold_Sensitivity_NetOfSecBeta_104W'],
            'Residual Oil Sensitivity 104 Week': ['CAOil_Sensitivity_NetOfSecBeta_104W'],
            'Residual Gold Sensitivity 52 Week': ['CAGold_Sensitivity_NetOfSecBeta_52W'],
            'Residual Oil Sensitivity 52 Week': ['CAOil_Sensitivity_NetOfSecBeta_52W'],            
            }

    styleList = sorted(DescriptorMap.keys())

    DescriptorWeights = {}
    orthogList = {}
    exposureConfigFile = 'exposures-CA-fl'

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.CAAxioma2018FL')
        CAAxioma2018MH.__init__(self, modelDB, marketDB)
        self.noClip_List = ['ISC_Ret_Score','Returns to Trading Days']


###############################################################################################################
"""Currency models go here
"""
class FXAxioma2017USD_MH(CurrencyRisk.CurrencyStatisticalFactorModel2017):
    """Statistical factor based currency risk model, USD numeraire
    """
    rm_id = 220
    revision = 1
    rms_id = 220
    numStatFactors = 12
    blind = [ModelFactor('Statistical Factor %d' % n,
                         'Statistical Factor %d' % n)
                         for n in range(1, numStatFactors+1)]
    returnHistory = 250
    newExposureFormat = True

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.FXAxioma2017USD_MH')
        CurrencyRisk.CurrencyStatisticalFactorModel2017.__init__(self, modelDB, marketDB)
        self.setCalculators(modelDB)

    def setCalculators(self, modelDB, overrider = False):
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=1, overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=1, overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, useBlend=False, computeISC=False, minVar=0.0, overrider=overrider)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

class FXAxioma2017USD_SH(CurrencyRisk.CurrencyStatisticalFactorModel2017):
    """Statistical factor based currency risk model, USD numeraire
    Short-horizon flavour
    """
    rm_id = 221
    revision = 1
    rms_id = 221
    numStatFactors = 12
    blind = [ModelFactor('Statistical Factor %d' % n,
                         'Statistical Factor %d' % n)
                         for n in range(1, numStatFactors+1)]
    returnHistory = 250
    newExposureFormat = True

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.FXAxioma2017USD_SH')
        CurrencyRisk.CurrencyStatisticalFactorModel2017.__init__(self, modelDB, marketDB)
        self.setCalculators(modelDB)

    def setCalculators(self, modelDB, overrider = False):
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=1, horizon='short', overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=1, horizon='short', overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, useBlend=False, computeISC=False, minVar=0.0, horizon='short', overrider=overrider)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

class FXAxioma2017EUR_MH(FXAxioma2017USD_MH):
    """Statistical factor based currency risk model, USD numeraire
    """
    rm_id = 244
    revision = 1
    rms_id = 244

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.FXAxioma2017EUR_MH')
        FXAxioma2017USD_MH.__init__(self, modelDB, marketDB)

class FXAxioma2017EUR_SH(FXAxioma2017USD_SH):
    """Statistical factor based currency risk model, USD numeraire
    """
    rm_id = 245
    revision = 1
    rms_id = 245

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.FXAxioma2017EUR_SH')
        FXAxioma2017USD_SH.__init__(self, modelDB, marketDB)

###############################################################################################################
"""Regional models go here
"""
class WWAxioma2017MH(EquityModel.FundamentalModel):
    """Version 4 Global medium-horizon fundamental model
    """
    rm_id = 230
    revision = 1
    rms_id = 230

    styleList = ['Value',
                 'Leverage',
                 'Growth',
                 'Profitability',
                 'Earnings Yield',
                 'Dividend Yield',
                 'Size',
                 'Liquidity',
                 'Market Sensitivity',
                 'Volatility',
                 'Medium-Term Momentum',
                 'Exchange Rate Sensitivity',
                 ]

    intercept = ModelFactor('Global Market', 'Global Market')
    localStructureFactors = [ModelFactor('Domestic China', 'Domestic China')]
    gicsDate = datetime.date(2016,9,1)
    industryClassification = Classification.GICSIndustries(gicsDate)

    DescriptorMap = {
            'Earnings Yield': ['Earnings_to_Price_Annual', 'Est_Earnings_to_Price_12MFL_Annual'],
            'Value': ['Book_to_Price_Annual'],
            'Leverage': ['Debt_to_Assets_Annual', 'Debt_to_Equity_Annual'],
            'Growth': ['Earnings_Growth_RPF_Annual', 'Sales_Growth_RPF_Annual'],
            'Dividend Yield': ['Dividend_Yield_Annual'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_60D', 'Amihud_Liquidity_125D', 'ISC_Ret_Score'],
            'Market Sensitivity': ['Regional_Market_Sensitivity_500D'],
            'Volatility': ['Volatility_125D'],
            'Medium-Term Momentum': ['Momentum_260x21D_Regional'],
            'Exchange Rate Sensitivity': ['XRate_104W_XDR'],
            'Profitability': ['Return_on_Equity_Annual', 'Return_on_Assets_Annual',
                'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
                'Sales_to_Assets_Annual', 'Gross_Margin_Annual'],
            }

    DescriptorWeights = {'Earnings Yield': [0.75, 0.25],}
    allowMixedFreqDescriptors = False
    orthogList = {'Volatility': [['Market Sensitivity'], True, 1.0],}
    exposureConfigFile = 'exposures-mh'

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.WWAxioma2017MH')

        # Set important model parameters
        ModelParameters2017.defaultModelSettings(self, scm=False)
        self.hasInternalSpecRets = False
        #self.useReturnsTimingV3 = True
        self.legacySAWeight = True

        # Set up relevant styles to be created/used
        ModelParameters2017.defaultExposureParameters(self, self.styleList, configFile=self.exposureConfigFile)

        # Initialise
        EquityModel.FundamentalModel.__init__(self, ['SEDOL', 'CUSIP'], modelDB, marketDB)
        self.currencyModel = riskmodels.getModelByName('FXAxioma2017USD_MH')(modelDB, marketDB)

        # Set up estimation universe parameters
        self.estu_parameters, self.elig_parameters = ModelParameters2017.defaultRegionalModelEstuParameters(self)
        
        # Set up RiskModel Calculators
        self.setCalculators(modelDB)

        # Set up descriptor standardization parameters
        descriptors = sorted(list(set([item for sublist
            in list(self.DescriptorMap.values()) for item in sublist])))
        regScope = Standardization_V4.RegionRelativeScope(modelDB, self.regionalStndDesc)
        gloScope = Standardization_V4.GlobalRelativeScope([d for d in descriptors if d not in self.regionalStndDesc])
        self.descriptorStandardization = Standardization_V4.BucketizedStandardization(
                [regScope, gloScope], mad_bound=15.0, exceptionNames=self.noStndDescriptor)

        # Set up standardization parameters
        regScope = Standardization_V4.RegionRelativeScope(modelDB, self.regionalStndList)
        gloScope = Standardization_V4.GlobalRelativeScope(
                [f.name for f in self.styles if f.name not in self.regionalStndList])
        self.exposureStandardization = Standardization_V4.BucketizedStandardization(
                [regScope, gloScope], fillWithZeroList=self.fillWithZeroList)

        # Manually reassign select assets to RMG and currency
        self.rmgOverride = dict()
        # Force all RDS issues to have GB exposure
        self.rmgOverride['CIJBKGWL8'] = [datetime.date(1980,1,1), datetime.date(2999,12,31), 'GB', 'GB']

    def generate_model_specific_exposures(self, modelDate, assetData, exposureMatrix, modelDB, marketDB):
        return self.generate_domestic_china_exposures(modelDate, assetData, exposureMatrix, modelDB, marketDB)

    def setCalculators(self, modelDB, overrider=False):
        # Set up internal factor return regression parameters
        dummyThreshold = 6
        self.internalCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType='market',
                dummyThreshold=dummyThreshold,
                marketRegression=False,
                scndRegList=[[ExposureMatrix.LocalFactor]],
                scndRegEstus=['ChinaA'],
                kappa=5.0,
                useRealMCaps=True,
                regWeight='rootCap',
                overrider = overrider,
                )

        # Set up external regression parameters
        self.returnCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType='market',
                dummyThreshold = dummyThreshold,
                marketRegression=False,
                scndRegList=[[ExposureMatrix.LocalFactor]],
                scndRegEstus=['ChinaA'],
                kappa=25.0,
                useRealMCaps=True,
                regWeight='rootCap',
                overrider = overrider,
                )

        # Set up weekly regression parameters
        self.weeklyCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType='market',
                dummyThreshold=dummyThreshold,
                marketRegression=False,
                scndRegList=[[ExposureMatrix.LocalFactor]],
                scndRegEstus=['ChinaA'],
                kappa=5.0,
                useRealMCaps=True,
                regWeight='rootCap',
                overrider = overrider,
                )

        # This controls the FMP regression
        self.fmpCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType=None,
                dummyThreshold=dummyThreshold,
                marketRegression=False,
                scndRegList=[[ExposureMatrix.LocalFactor]],
                scndRegEstus=['ChinaA'],
                kappa=None,
                useRealMCaps=True,
                regWeight='rootCap',
                overrider = overrider,
                )

        # And this is for raw exposure proxying
        self.expProxyCalculator = ModelParameters2017.defaultExposureProxyParameters(
                self, modelDB,
                dummyType='market',
                dummyThreshold=dummyThreshold,
                kappa=5.0,
                useRealMCaps=True,
                regWeight='rootCap',
                overrider = overrider,
                )

        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=3, overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=3, overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, overrider=overrider)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

class WWAxioma2017SH(WWAxioma2017MH):
    """Version 4 Global short-horizon fundamental model
    """
    rm_id = 231
    revision = 1
    rms_id = 231

    styleList = ['Value',
                 'Leverage',
                 'Growth',
                 'Profitability',
                 'Earnings Yield',
                 'Dividend Yield',
                 'Size',
                 'Liquidity',
                 'Market Sensitivity',
                 'Volatility',
                 'Medium-Term Momentum',
                 'Short-Term Momentum',
                 'Exchange Rate Sensitivity',
                 ]

    DescriptorMap = {
            'Earnings Yield': ['Earnings_to_Price_Annual', 'Est_Earnings_to_Price_12MFL_Annual'],
            'Value': ['Book_to_Price_Annual'],
            'Leverage': ['Debt_to_Assets_Annual', 'Debt_to_Equity_Annual'],
            'Growth': ['Earnings_Growth_RPF_Annual', 'Sales_Growth_RPF_Annual'],
            'Dividend Yield': ['Dividend_Yield_Annual'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_20D', 'Amihud_Liquidity_60D', 'ISC_Ret_Score'],
            'Market Sensitivity': ['Regional_Market_Sensitivity_250D'],
            'Volatility': ['Volatility_60D'],
            'Medium-Term Momentum': ['Momentum_260x21D_Regional'],
            'Short-Term Momentum': ['Momentum_21D'],
            'Exchange Rate Sensitivity': ['XRate_52W_XDR'],
            'Profitability': ['Return_on_Equity_Annual', 'Return_on_Assets_Annual',
                'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
                'Sales_to_Assets_Annual', 'Gross_Margin_Annual'],
            }
    exposureConfigFile = 'exposures-sh'

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.WWAxioma2017SH')
        WWAxioma2017MH.__init__(self, modelDB, marketDB)
        self.currencyModel = riskmodels.getModelByName('FXAxioma2017USD_SH')(modelDB, marketDB)

    def setCalculators(self, modelDB, overrider=False):
        WWAxioma2017MH.setCalculators(self, modelDB, overrider=overrider)
        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=3, horizon='short', overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=3, horizon='short', overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, horizon='short', overrider=overrider)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

class WWAxioma2017FL(WWAxioma2017MH):
    """Version 4 Global short-horizon fundamental model
    """
    rm_id = 234
    revision = 1
    rms_id = 234

    DescriptorMap = {
            'Amihud Liquidity 60 Day': ['Amihud_Liquidity_60D'],
            'Amihud Liquidity 125 Day': ['Amihud_Liquidity_125D'],
            'Book to Price Annual': ['Book_to_Price_Annual'],
            'Cash Flow to Assets Annual': ['CashFlow_to_Assets_Annual'],
            'Cash Flow to Income Annual': ['CashFlow_to_Income_Annual'],
            'Debt to Assets Annual': ['Debt_to_Assets_Annual'],
            'Debt to Equity Annual': ['Debt_to_Equity_Annual'],
            'Dividend Yield Annual': ['Dividend_Yield_Annual'],
            'Earnings to Price Annual': ['Earnings_to_Price_Annual'],
            'Est Earnings to Price Annual': ['Est_Earnings_to_Price_12MFL_Annual'],
            'Earnings Growth Annual': ['Earnings_Growth_RPF_Annual'],
            'Gross Margin Annual': ['Gross_Margin_Annual'],
            #'Returns to Trading Days SH': ['ISC_Ret_Score'],   
            #'Returns to Trading Days MH': ['ISC_Ret_Score'],   
            'Returns to Trading Days': ['ISC_Ret_Score'],   
            'Log of Issuer Cap': ['LnIssuerCap'], 
            'Log of 20-Day ADV to Issuer Cap': ['LnTrading_Activity_20D'],
            'Log of 60-Day ADV to Issuer Cap': ['LnTrading_Activity_60D'],
            'Annual Return Excl Prev Month': ['Momentum_260x21D_Regional'], 
            'Monthly Return': ['Momentum_21D'],  
            'Regional Market Sensitivity 250 Day': ['Regional_Market_Sensitivity_250D'],
            'Regional Market Sensitivity 500 Day': ['Regional_Market_Sensitivity_500D'],
            'Return on Assets Annual': ['Return_on_Assets_Annual'],
            'Return on Equity Annual': ['Return_on_Equity_Annual'],
            'Sales to Assets Annual': ['Sales_to_Assets_Annual'],
            'Sales Growth Annual': ['Sales_Growth_RPF_Annual'],
            'Volatility 60 Day': ['Volatility_60D'],
            'Volatility 125 Day': ['Volatility_125D'],
            'Exchange Rate Sensitivity 52 Week XDR': ['XRate_52W_XDR'],
            'Exchange Rate Sensitivity 104 Week XDR': ['XRate_104W_XDR'],
            }

    styleList = sorted(DescriptorMap.keys())

    DescriptorWeights = {}
    orthogList = {}
    exposureConfigFile = 'exposures-fl'

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.WWAxioma2017FL')
        WWAxioma2017MH.__init__(self, modelDB, marketDB)
        self.allowMixedFreqDescriptors = False

class WWPreAxioma2017MH(WWAxioma2017MH):
    """Version 4 Global medium-horizon fundamental model
    """
    rm_id = 235
    revision = 1
    rms_id = 235

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.WWPreAxioma2017MH')
        WWAxioma2017MH.__init__(self, modelDB, marketDB)

class WWAxioma2017MH_S(EquityModel.StatisticalModel):
    """Version 4 Global medium-horizon statistical model
    """
    rm_id = 232
    revision = 1
    rms_id = 232
    numFactors = 20
    blind = [ModelFactor('Statistical Factor %d' % n, 'Statistical Factor %d' % n)
                         for n in range(1, numFactors+1)]
    pcaHistory = 250
    gicsDate = datetime.date(2016,9,1)
    industryClassification = Classification.GICSIndustries(gicsDate)
    descriptorNumeraire = 'USD'
    
    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.WWAxioma2017MH_S')
        # Set important model parameters
        ModelParameters2017.defaultModelSettings(self, scm=False, statModel=True)
        self.legacySAWeight = True
        EquityModel.StatisticalModel.__init__(self, ['SEDOL', 'CUSIP'], modelDB, marketDB)
        # So we can use the same ESTU method as the fundamental model
        self.baseModelDateMap = {datetime.date(1980,1,1): WWAxioma2017MH(modelDB, marketDB)}
        # Set up estimation universe parameters
        self.elig_parameters = ModelParameters2017.defaultRegionalModelEstuParameters(self)[1]
        # Set Calculators
        self.setCalculators(modelDB)
        # Define currency model
        self.currencyModel = riskmodels.getModelByName('FXAxioma2017USD_MH')(modelDB, marketDB)

        # Manually reassign select assets to RMG and currency
        self.rmgOverride = dict()
        # Force all RDS issues to have GB exposure
        self.rmgOverride['CIJBKGWL8'] = [datetime.date(1980,1,1), datetime.date(2999,12,31), 'GB', 'GB']
    
    def setCalculators(self, modelDB, overrider=False):
        # Set up returns model
        self.returnCalculator = FactorReturns.AsymptoticPrincipalComponents2017(self.numFactors, trimExtremeExposures=True)
        self.olsReturnClass = ModelParameters2017.simpleRegressionParameters(self, modelDB, overrider=overrider)
        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=1, overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=1, overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, nwLag=1, overrider=overrider)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

class WWAxioma2017SH_S(EquityModel.StatisticalModel):
    """Version 4 Global short-horizon statistical model
    """
    rm_id = 233
    revision = 1
    rms_id = 233
    numFactors = 20
    blind = [ModelFactor('Statistical Factor %d' % n, 'Statistical Factor %d' % n)
                         for n in range(1, numFactors+1)]
    pcaHistory = 250
    gicsDate = datetime.date(2016,9,1)
    industryClassification = Classification.GICSIndustries(gicsDate)
    descriptorNumeraire = 'USD'
    
    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.WWAxioma2017SH_S')
        # Set important model parameters
        ModelParameters2017.defaultModelSettings(self, scm=False, statModel=True)
        self.legacySAWeight = True
        EquityModel.StatisticalModel.__init__(self, ['SEDOL', 'CUSIP'], modelDB, marketDB)
        # So we can use the same ESTU method as the fundamental model
        self.baseModelDateMap = {datetime.date(1980,1,1): WWAxioma2017MH(modelDB, marketDB)}
        # Set up estimation universe parameters
        self.elig_parameters = ModelParameters2017.defaultRegionalModelEstuParameters(self)[1]
        # Set Calculators
        self.setCalculators(modelDB)
        # Define currency model
        self.currencyModel = riskmodels.getModelByName('FXAxioma2017USD_SH')(modelDB, marketDB)

        # Manually reassign select assets to RMG and currency
        self.rmgOverride = dict()
        # Force all RDS issues to have GB exposure
        self.rmgOverride['CIJBKGWL8'] = [datetime.date(1980,1,1), datetime.date(2999,12,31), 'GB', 'GB']
    
    def setCalculators(self, modelDB, overrider=False):
        # Set up returns model
        self.returnCalculator = FactorReturns.AsymptoticPrincipalComponents2017(self.numFactors, trimExtremeExposures=True)
        self.olsReturnClass = ModelParameters2017.simpleRegressionParameters(self, modelDB, overrider=overrider)
        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=1, horizon='short', overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=1, horizon='short', overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, nwLag=1, horizon='short', overrider=overrider)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

class WWAxioma2018MH(WWAxioma2017MH):
    """Version 4 WW medium-horizon fundamental model with GICS 2018
    """
    rm_id = 290
    revision = 1
    rms_id = 290
    gicsDate = datetime.date(2018,9,29)
    industryClassification = Classification.GICSIndustries(gicsDate)
    allowMixedFreqDescriptors = True

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.WWAxioma2018MH')
        WWAxioma2017MH.__init__(self, modelDB, marketDB)
        self.hasInternalSpecRets = True
        self.legacySAWeight = False

    def setCalculators(self, modelDB, overrider=False):
        # Set up internal factor return regression parameters
        dummyThreshold = 6
        self.internalCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType='market',
                dummyThreshold=dummyThreshold,
                marketRegression=False,
                scndRegList=[[ExposureMatrix.LocalFactor]],
                scndRegEstus=['ChinaA'],
                kappa=5.0,
                useRealMCaps=True,
                regWeight='rootCap',
                overrider = overrider,
                )

        # Set up external regression parameters
        self.returnCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType='market',
                dummyThreshold = dummyThreshold,
                marketRegression=False,
                scndRegList=[[ExposureMatrix.LocalFactor]],
                scndRegEstus=['ChinaA'],
                kappa=25.0,
                useRealMCaps=True,
                regWeight='rootCap',
                overrider = overrider,
                )

        # Set up weekly regression parameters
        self.weeklyCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType='market',
                dummyThreshold=dummyThreshold,
                marketRegression=False,
                scndRegList=[[ExposureMatrix.LocalFactor]],
                scndRegEstus=['ChinaA'],
                kappa=5.0,
                useRealMCaps=True,
                regWeight='rootCap',
                overrider = overrider,
                )

        # This controls the FMP regression
        self.fmpCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType=None,
                dummyThreshold = dummyThreshold,
                marketRegression=False,
                scndRegList=[[ExposureMatrix.LocalFactor]],
                scndRegEstus=['ChinaA'],
                kappa=None,
                useRealMCaps=True,
                regWeight='rootCap',
                overrider = overrider,
                )

        # And this is for raw exposure proxying
        self.expProxyCalculator = ModelParameters2017.defaultExposureProxyParameters(
                self, modelDB,
                dummyType='market',
                dummyThreshold = dummyThreshold,
                kappa=5.0,
                useRealMCaps=True,
                regWeight='rootCap',
                overrider = overrider,
                )

        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=3, overrider=overrider)#, dateOverLap=5)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=3, overrider=overrider)#, dateOverLap=5)
        ModelParameters2017.defaultSpecificVarianceParameters(self, overrider=overrider)#, dateOverLap=5)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

class WWAxioma2018SH(WWAxioma2017SH):
    """Version 4 WW short-horizon fundamental model with GICS 2018
    """
    rm_id = 291
    revision = 1
    rms_id = 291
    gicsDate = datetime.date(2018,9,29)
    industryClassification = Classification.GICSIndustries(gicsDate)

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.WWAxioma2018SH')
        WWAxioma2017SH.__init__(self, modelDB, marketDB)

class WWAxioma2018MH_S(EquityModel.StatisticalModel):
    """Version 4 WW medium-horizon statistical model with GICS 2018
    """
    rm_id = 292
    revision = 1
    rms_id = 292
    numFactors = 20
    pcaHistory = 250
    descriptorNumeraire = 'USD'
    gicsDate = datetime.date(2018,9,29)
    industryClassification = Classification.GICSIndustries(gicsDate)
    blind = [ModelFactor('Statistical Factor %d' % n, 'Statistical Factor %d' % n)
                         for n in range(1, numFactors+1)]

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.WWAxioma2018MH_S')
        # Set important model parameters
        ModelParameters2017.defaultModelSettings(self, scm=False, statModel=True)
        EquityModel.StatisticalModel.__init__(self, ['SEDOL', 'CUSIP'], modelDB, marketDB)
        # So we can use the same ESTU method as the fundamental model
        self.baseModelDateMap = {datetime.date(1980,1,1): WWAxioma2018MH(modelDB, marketDB)}
        # Set up estimation universe parameters
        self.elig_parameters = ModelParameters2017.defaultRegionalModelEstuParameters(self)[1]
        # Set Calculators
        self.setCalculators(modelDB)
        # Define currency model
        self.currencyModel = riskmodels.getModelByName('FXAxioma2017USD_MH')(modelDB, marketDB)

        # Manually reassign select assets to RMG and currency
        self.rmgOverride = dict()
        # Force all RDS issues to have GB exposure
        self.rmgOverride['CIJBKGWL8'] = [datetime.date(1980,1,1), datetime.date(2999,12,31), 'GB', 'GB']

    def setCalculators(self, modelDB, overrider=False):
        # Set up returns model
        self.returnCalculator = FactorReturns.AsymptoticPrincipalComponents2017(self.numFactors, trimExtremeExposures=True)
        self.olsReturnClass = ModelParameters2017.simpleRegressionParameters(self, modelDB, overrider=overrider)
        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=1, overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=1, overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, nwLag=1, overrider=overrider)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

class WWAxioma2018SH_S(EquityModel.StatisticalModel):
    """Version 4 WW short-horizon statistical model with GICS 2018
    """
    rm_id = 293
    revision = 1
    rms_id = 293
    numFactors = 20
    pcaHistory = 250
    descriptorNumeraire = 'USD'
    gicsDate = datetime.date(2018,9,29)
    industryClassification = Classification.GICSIndustries(gicsDate)
    blind = [ModelFactor('Statistical Factor %d' % n, 'Statistical Factor %d' % n)
                         for n in range(1, numFactors+1)]

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.WWAxioma2018SH_S')
        # Set important model parameters
        ModelParameters2017.defaultModelSettings(self, scm=False, statModel=True)
        EquityModel.StatisticalModel.__init__(self, ['SEDOL', 'CUSIP'], modelDB, marketDB)
        # So we can use the same ESTU method as the fundamental model
        self.baseModelDateMap = {datetime.date(1980,1,1): WWAxioma2018MH(modelDB, marketDB)}
        # Set up estimation universe parameters
        self.elig_parameters = ModelParameters2017.defaultRegionalModelEstuParameters(self)[1]
        # Set Calculators
        self.setCalculators(modelDB)
        # Define currency model
        self.currencyModel = riskmodels.getModelByName('FXAxioma2017USD_SH')(modelDB, marketDB)

        # Manually reassign select assets to RMG and currency
        self.rmgOverride = dict()
        # Force all RDS issues to have GB exposure
        self.rmgOverride['CIJBKGWL8'] = [datetime.date(1980,1,1), datetime.date(2999,12,31), 'GB', 'GB']

    def setCalculators(self, modelDB, overrider=False):
        # Set up returns model
        self.returnCalculator = FactorReturns.AsymptoticPrincipalComponents2017(self.numFactors, trimExtremeExposures=True)
        self.olsReturnClass = ModelParameters2017.simpleRegressionParameters(self, modelDB, overrider=overrider)
        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=1, horizon='short', overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=1, horizon='short', overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, nwLag=1, horizon='short', overrider=overrider)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

class EUAxioma2017MH(EquityModel.FundamentalModel):
    """Version 4 Europe medium-horizon fundamental model
    """
    rm_id = 246
    revision = 1
    rms_id = 246

    styleList = ['Value',
                 'Leverage',
                 'Growth',
                 'Profitability',
                 'Earnings Yield',
                 'Dividend Yield',
                 'Size',
                 'Liquidity',
                 'Market Sensitivity',
                 'Volatility',
                 'Medium-Term Momentum',
                 'Exchange Rate Sensitivity',
                 ]

    intercept = ModelFactor('European Market', 'European Market')
    gicsDate = datetime.date(2016,9,1)
    industryClassification = Classification.GICSCustomEU(gicsDate)

    DescriptorMap = {
            'Earnings Yield': ['Earnings_to_Price_Annual', 'Est_Earnings_to_Price_12MFL_Annual'],
            'Value': ['Book_to_Price_Annual'],
            'Leverage': ['Debt_to_Assets_Annual', 'Debt_to_Equity_Annual'],
            'Growth': ['Earnings_Growth_RPF_Annual', 'Sales_Growth_RPF_Annual'],
            'Dividend Yield': ['Dividend_Yield_Annual'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_60D', 'Amihud_Liquidity_125D', 'ISC_Ret_Score'],
            'Market Sensitivity': ['Europe_Regional_Market_Sensitivity_500D'],
            'Volatility': ['Europe_Volatility_125D'],
            'Medium-Term Momentum': ['Momentum_260x21D_Regional'],
            'Exchange Rate Sensitivity': ['XRate_104W_XDR'],
            'Profitability': ['Return_on_Equity_Annual', 'Return_on_Assets_Annual',
                'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
                'Sales_to_Assets_Annual', 'Gross_Margin_Annual'],
            }

    DescriptorWeights = {'Earnings Yield': [0.75, 0.25],}
    allowMixedFreqDescriptors = False
    orthogList = {'Volatility': [['Market Sensitivity'], True, 1.0],}
    exposureConfigFile = 'exposures-mh'

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.EUAxioma2017MH')

        # Set important model parameters
        ModelParameters2017.defaultModelSettings(self, scm=False)
        self.hasInternalSpecRets = False

        # Set up relevant styles to be created/used
        ModelParameters2017.defaultExposureParameters(
                self, self.styleList, configFile=self.exposureConfigFile, descriptorNumeraire='USD')

        # Initialise
        EquityModel.FundamentalModel.__init__(self, ['SEDOL', 'CUSIP'], modelDB, marketDB)
        self.currencyModel = riskmodels.getModelByName('FXAxioma2017EUR_MH')(modelDB, marketDB)

        # Set up estimation universe parameters
        self.estu_parameters, self.elig_parameters = ModelParameters2017.defaultRegionalModelEstuParameters(self)

        # Set up RiskModel Calculators
        self.setCalculators(modelDB)

        # Set up descriptor standardization parameters
        descriptors = sorted(list(set([item for sublist
            in list(self.DescriptorMap.values()) for item in sublist])))
        regScope = Standardization_V4.RegionRelativeScope(
                modelDB, self.regionalStndDesc)
        gloScope = Standardization_V4.GlobalRelativeScope(
                [d for d in descriptors if d not in self.regionalStndDesc])
        self.descriptorStandardization = Standardization_V4.BucketizedStandardization(
                [regScope, gloScope], mad_bound=15.0, exceptionNames=self.noStndDescriptor)

        # Set up standardization parameters
        regScope = Standardization_V4.RegionRelativeScope(modelDB, self.regionalStndList)
        gloScope = Standardization_V4.GlobalRelativeScope(
                [f.name for f in self.styles if f.name not in self.regionalStndList])
        self.exposureStandardization = Standardization_V4.BucketizedStandardization(
                [regScope, gloScope], fillWithZeroList=self.fillWithZeroList)

        # Manually reassign select assets to RMG and currency
        self.rmgOverride = dict()
        # Force all RDS issues to have GB exposure
        self.rmgOverride['CIJBKGWL8'] = [datetime.date(1980,1,1), datetime.date(2999,12,31), 'GB', 'GB']

    def setCalculators(self, modelDB, overrider=False):
        # Set up internal factor return regression parameters
        dummyThreshold = 6
        self.internalCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType='market',
                dummyThreshold=dummyThreshold,
                marketRegression=False,
                kappa=5.0,
                useRealMCaps=True,
                regWeight='rootCap',
                overrider = overrider,
                )

        # Set up external regression parameters
        self.returnCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType='market',
                dummyThreshold=dummyThreshold,
                marketRegression=False,
                kappa=25.0,
                useRealMCaps=True,
                regWeight='rootCap',
                overrider = overrider,
                )

        # This controls the FMP regression
        self.fmpCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType=None,
                dummyThreshold=dummyThreshold,
                marketRegression=False,
                kappa=None,
                useRealMCaps=True,
                regWeight='rootCap',
                overrider = overrider,
                )

        # And this is for raw exposure proxying
        self.expProxyCalculator = ModelParameters2017.defaultExposureProxyParameters(
                self, modelDB,
                dummyType='market',
                dummyThreshold=dummyThreshold,
                kappa=5.0,
                useRealMCaps=True,
                regWeight='rootCap',
                overrider = overrider,
                )

        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=3, overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=3, overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, overrider=overrider)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

class EUAxioma2017SH(EUAxioma2017MH):
    """Version 4 Europe short-horizon fundamental model
    """
    rm_id = 247
    revision = 1
    rms_id = 247

    styleList = ['Value',
                 'Leverage',
                 'Growth',
                 'Profitability',
                 'Earnings Yield',
                 'Dividend Yield',
                 'Size',
                 'Liquidity',
                 'Market Sensitivity',
                 'Volatility',
                 'Medium-Term Momentum',
                 'Short-Term Momentum',
                 'Exchange Rate Sensitivity',
                 ]

    DescriptorMap = {
            'Earnings Yield': ['Earnings_to_Price_Annual', 'Est_Earnings_to_Price_12MFL_Annual'],
            'Value': ['Book_to_Price_Annual'],
            'Leverage': ['Debt_to_Assets_Annual', 'Debt_to_Equity_Annual'],
            'Growth': ['Earnings_Growth_RPF_Annual', 'Sales_Growth_RPF_Annual'],
            'Dividend Yield': ['Dividend_Yield_Annual'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_20D', 'Amihud_Liquidity_60D', 'ISC_Ret_Score'],
            'Market Sensitivity': ['Europe_Regional_Market_Sensitivity_250D'],
            'Volatility': ['Europe_Volatility_60D'],
            'Medium-Term Momentum': ['Momentum_260x21D_Regional'],
            'Short-Term Momentum': ['Momentum_21D'],
            'Exchange Rate Sensitivity': ['XRate_52W_XDR'],
            'Profitability': ['Return_on_Equity_Annual', 'Return_on_Assets_Annual',
                'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
                'Sales_to_Assets_Annual', 'Gross_Margin_Annual'],
            }
    exposureConfigFile = 'exposures-sh'

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.EUAxioma2017SH')
        EUAxioma2017MH.__init__(self, modelDB, marketDB)
        self.currencyModel = riskmodels.getModelByName('FXAxioma2017EUR_SH')(modelDB, marketDB)

    def setCalculators(self, modelDB, overrider=False):
        EUAxioma2017MH.setCalculators(self, modelDB, overrider=overrider)
        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=3, horizon='short', overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=3, horizon='short', overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, horizon='short', overrider=overrider)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

class EUAxioma2017MH_S(EquityModel.StatisticalModel):
    """Version 4 Europe medium-horizon statistical model
    """
    rm_id = 248
    revision = 1
    rms_id = 248
    numFactors = 20
    blind = [ModelFactor('Statistical Factor %d' % n, 'Statistical Factor %d' % n)
                         for n in range(1, numFactors+1)]
    pcaHistory = 250
    gicsDate = datetime.date(2016,9,1)
    industryClassification = Classification.GICSCustomEU(gicsDate)
    descriptorNumeraire = 'USD'

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.EUAxioma2017MH_S')
        # Set important model parameters
        ModelParameters2017.defaultModelSettings(self, scm=False, statModel=True)
        EquityModel.StatisticalModel.__init__(self, ['SEDOL', 'CUSIP'], modelDB, marketDB)
        # So we can use the same ESTU method as the fundamental model
        self.baseModelDateMap = {datetime.date(1980,1,1): EUAxioma2017MH(modelDB, marketDB)}
        # Set up estimation universe parameters
        self.elig_parameters = ModelParameters2017.defaultRegionalModelEstuParameters(self)[1]
        # Set Calculators
        self.setCalculators(modelDB)
        # Define currency model
        self.currencyModel = riskmodels.getModelByName('FXAxioma2017EUR_MH')(modelDB, marketDB)

        # Manually reassign select assets to RMG and currency
        self.rmgOverride = dict()
        # Force all RDS issues to have GB exposure
        self.rmgOverride['CIJBKGWL8'] = [datetime.date(1980,1,1), datetime.date(2999,12,31), 'GB', 'GB']

    def setCalculators(self, modelDB, overrider=False):
        # Set up returns model
        self.returnCalculator = FactorReturns.AsymptoticPrincipalComponents2017(self.numFactors, trimExtremeExposures=True)
        self.olsReturnClass = ModelParameters2017.simpleRegressionParameters(self, modelDB, overrider=overrider)
        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=1, overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=1, overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, nwLag=1, overrider=overrider)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

class EUAxioma2017SH_S(EquityModel.StatisticalModel):
    """Version 4 Europe short-horizon statistical model
    """
    rm_id = 249
    revision = 1
    rms_id = 249
    numFactors = 20
    blind = [ModelFactor('Statistical Factor %d' % n, 'Statistical Factor %d' % n)
                         for n in range(1, numFactors+1)]
    pcaHistory = 250
    gicsDate = datetime.date(2016,9,1)
    industryClassification = Classification.GICSCustomEU(gicsDate)
    descriptorNumeraire = 'USD'

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.EUAxioma2017SH_S')
        # Set important model parameters
        ModelParameters2017.defaultModelSettings(self, scm=False, statModel=True)
        EquityModel.StatisticalModel.__init__(self, ['SEDOL', 'CUSIP'], modelDB, marketDB)
        # So we can use the same ESTU method as the fundamental model
        self.baseModelDateMap = {datetime.date(1980,1,1): EUAxioma2017MH(modelDB, marketDB)}
        # Set up estimation universe parameters
        self.elig_parameters = ModelParameters2017.defaultRegionalModelEstuParameters(self)[1]
        # Set Calculators
        self.setCalculators(modelDB)
        # Define currency model
        self.currencyModel = riskmodels.getModelByName('FXAxioma2017EUR_SH')(modelDB, marketDB)

        # Manually reassign select assets to RMG and currency
        self.rmgOverride = dict()
        # Force all RDS issues to have GB exposure
        self.rmgOverride['CIJBKGWL8'] = [datetime.date(1980,1,1), datetime.date(2999,12,31), 'GB', 'GB']

    def setCalculators(self, modelDB, overrider=False):
        # Set up returns model
        self.returnCalculator = FactorReturns.AsymptoticPrincipalComponents2017(self.numFactors, trimExtremeExposures=True)
        self.olsReturnClass = ModelParameters2017.simpleRegressionParameters(self, modelDB, overrider=overrider)
        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=1, horizon='short', overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=1, horizon='short', overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, nwLag=1, horizon='short', overrider=overrider)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

class EUAxioma2017FL(EUAxioma2017MH):
    """EU4 factor model
    """
    rm_id = 270
    revision = 1
    rms_id = 270

    DescriptorMap = {
            'Amihud Liquidity 60 Day': ['Amihud_Liquidity_60D'],
            'Amihud Liquidity 125 Day': ['Amihud_Liquidity_125D'],
            'Book to Price Annual': ['Book_to_Price_Annual'],
            'Cash Flow to Assets Annual': ['CashFlow_to_Assets_Annual'],
            'Cash Flow to Income Annual': ['CashFlow_to_Income_Annual'],
            'Debt to Assets Annual': ['Debt_to_Assets_Annual'],
            'Debt to Equity Annual': ['Debt_to_Equity_Annual'],
            'Dividend Yield Annual': ['Dividend_Yield_Annual'],
            'Earnings to Price Annual': ['Earnings_to_Price_Annual'],
            'Est Earnings to Price Annual': ['Est_Earnings_to_Price_12MFL_Annual'],
            'Earnings Growth Annual': ['Earnings_Growth_RPF_Annual'],
            'Gross Margin Annual': ['Gross_Margin_Annual'],
            'Returns to Trading Days': ['ISC_Ret_Score'],   
            'Log of Issuer Cap': ['LnIssuerCap'], 
            'Log of 20-Day ADV to Issuer Cap': ['LnTrading_Activity_20D'],
            'Log of 60-Day ADV to Issuer Cap': ['LnTrading_Activity_60D'],
            'Annual Return Excl Prev Month': ['Momentum_260x21D_Regional'], 
            'Monthly Return': ['Momentum_21D'],  
            'Europe Regional Market Sensitivity 250 Day': ['Europe_Regional_Market_Sensitivity_250D'],
            'Europe Regional Market Sensitivity 500 Day': ['Europe_Regional_Market_Sensitivity_500D'],
            'Return on Assets Annual': ['Return_on_Assets_Annual'],
            'Return on Equity Annual': ['Return_on_Equity_Annual'],
            'Sales to Assets Annual': ['Sales_to_Assets_Annual'],
            'Sales Growth Annual': ['Sales_Growth_RPF_Annual'],
            'Volatility 60 Day': ['Europe_Volatility_60D'],
            'Volatility 125 Day': ['Europe_Volatility_125D'],
            'Exchange Rate Sensitivity 52 Week XDR': ['XRate_52W_XDR'],
            'Exchange Rate Sensitivity 104 Week XDR': ['XRate_104W_XDR'],
            }

    styleList = sorted(DescriptorMap.keys())

    DescriptorWeights = {}
    orthogList = {}
    exposureConfigFile = 'exposures-EUAxioma2017FL'

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.EUAxioma2017FL')
        EUAxioma2017MH.__init__(self, modelDB, marketDB)
        self.allowMixedFreqDescriptors = False

class EMAxioma2018MH(EquityModel.FundamentalModel):
    """Version 4 EM medium-horizon fundamental model with GICS 2018
    """
    rm_id = 266
    revision = 1
    rms_id = 266

    styleList = ['Value',
                 'Leverage',
                 'Growth',
                 'Profitability',
                 'Earnings Yield',
                 'Dividend Yield',
                 'Size',
                 'Liquidity',
                 'Market Sensitivity',
                 'Volatility',
                 'Medium-Term Momentum',
                 'Exchange Rate Sensitivity',
                 ]

    intercept = ModelFactor('Emerging Market', 'Emerging Market')
    localStructureFactors = [ModelFactor('Domestic China', 'Domestic China')]
    gicsDate = datetime.date(2018,9,29)
    industryClassification = Classification.GICSCustomEM2(gicsDate)

    DescriptorMap = {
            'Earnings Yield': ['Earnings_to_Price_Annual', 'Est_Earnings_to_Price_12MFL_Annual'],
            'Value': ['Book_to_Price_Annual'],
            'Leverage': ['Debt_to_Assets_Annual', 'Debt_to_Equity_Annual'],
            'Growth': ['Earnings_Growth_RPF_Annual', 'Sales_Growth_RPF_Annual'],
            'Dividend Yield': ['Dividend_Yield_Annual'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_60D', 'Amihud_Liquidity_125D', 'ISC_Ret_Score'],
            'Market Sensitivity': ['EM_Regional_Market_Sensitivity_500D'],
            'Volatility': ['EM_Volatility_125D'],
            'Medium-Term Momentum': ['Momentum_260x21D_Regional'],
            'Exchange Rate Sensitivity': ['XRate_104W_XDR'],
            'Profitability': ['Return_on_Equity_Annual', 'Return_on_Assets_Annual',
                'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
                'Sales_to_Assets_Annual', 'Gross_Margin_Annual'],
            }

    DescriptorWeights = {'Earnings Yield': [0.75, 0.25],}
    allowMixedFreqDescriptors = False
    orthogList = {'Volatility': [['Market Sensitivity'], True, 1.0],}
    exposureConfigFile = 'exposures-mh'

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.EMAxioma2018MH')

        # Set important model parameters
        ModelParameters2017.defaultModelSettings(self, scm=False)
        self.downWeightMissingReturns = True
        self.legacySAWeight = True

        # Set up relevant styles to be created/used
        ModelParameters2017.defaultExposureParameters(
                self, self.styleList, configFile=self.exposureConfigFile, descriptorNumeraire='USD')

        # Initialise
        EquityModel.FundamentalModel.__init__(self, ['SEDOL', 'CUSIP'], modelDB, marketDB)
        self.currencyModel = riskmodels.getModelByName('FXAxioma2017USD_MH')(modelDB, marketDB)

        # Set up estimation universe parameters
        self.estu_parameters, self.elig_parameters = ModelParameters2017.defaultRegionalModelEstuParameters(self)

        # Set up RiskModel Calculators
        self.setCalculators(modelDB)

        # Set up descriptor standardization parameters
        descriptors = sorted(list(set([item for sublist
            in list(self.DescriptorMap.values()) for item in sublist])))
        regScope = Standardization_V4.RegionRelativeScope(modelDB, self.regionalStndDesc)
        gloScope = Standardization_V4.GlobalRelativeScope([d for d in descriptors if d not in self.regionalStndDesc])
        self.descriptorStandardization = Standardization_V4.BucketizedStandardization(
                [regScope, gloScope], mad_bound=15.0, exceptionNames=self.noStndDescriptor)

        # Set up standardization parameters
        regScope = Standardization_V4.RegionRelativeScope(modelDB, self.regionalStndList)
        gloScope = Standardization_V4.GlobalRelativeScope(
                [f.name for f in self.styles if f.name not in self.regionalStndList])
        self.exposureStandardization = Standardization_V4.BucketizedStandardization(
                [regScope, gloScope], fillWithZeroList=self.fillWithZeroList)

    def generate_model_specific_exposures(self, modelDate, assetData, exposureMatrix, modelDB, marketDB):
        return self.generate_domestic_china_exposures(modelDate, assetData, exposureMatrix, modelDB, marketDB)

    def setCalculators(self, modelDB, overrider=False):
        # Set up internal factor return regression parameters
        dummyThreshold = 6
        self.internalCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType='market',
                dummyThreshold=dummyThreshold,
                marketRegression=False,
                scndRegList=[[ExposureMatrix.LocalFactor]],
                scndRegEstus=['ChinaA'],
                kappa=5.0,
                useRealMCaps=True,
                regWeight='rootCap',
                overrider = overrider,
                )

        # Set up external regression parameters
        self.returnCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType='market',
                dummyThreshold=dummyThreshold,
                marketRegression=False,
                scndRegList=[[ExposureMatrix.LocalFactor]],
                scndRegEstus=['ChinaA'],
                kappa=25.0,
                useRealMCaps=True,
                regWeight='rootCap',
                overrider = overrider,
                )

        # This controls the FMP regression
        self.fmpCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType=None,
                dummyThreshold=dummyThreshold,
                marketRegression=False,
                scndRegList=[[ExposureMatrix.LocalFactor]],
                scndRegEstus=['ChinaA'],
                kappa=None,
                useRealMCaps=True,
                regWeight='rootCap',
                overrider = overrider,
               )

        # And this is for raw exposure proxying
        self.expProxyCalculator = ModelParameters2017.defaultExposureProxyParameters(
                self, modelDB,
                dummyType='market',
                dummyThreshold=dummyThreshold,
                kappa=5.0,
                useRealMCaps=True,
                regWeight='rootCap',
                overrider = overrider,
                )

        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=3, overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=3, overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, overrider=overrider)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

class EMAxioma2018SH(EMAxioma2018MH):
    """Version 4 EM short-horizon fundamental model with GICS 2018
    """
    rm_id = 275
    revision = 1
    rms_id = 275

    styleList = ['Value',
                 'Leverage',
                 'Growth',
                 'Profitability',
                 'Earnings Yield',
                 'Dividend Yield',
                 'Size',
                 'Liquidity',
                 'Market Sensitivity',
                 'Volatility',
                 'Medium-Term Momentum',
                 'Short-Term Momentum',
                 'Exchange Rate Sensitivity',
                 ]

    DescriptorMap = {
            'Earnings Yield': ['Earnings_to_Price_Annual', 'Est_Earnings_to_Price_12MFL_Annual'],
            'Value': ['Book_to_Price_Annual'],
            'Leverage': ['Debt_to_Assets_Annual', 'Debt_to_Equity_Annual'],
            'Growth': ['Earnings_Growth_RPF_Annual', 'Sales_Growth_RPF_Annual'],
            'Dividend Yield': ['Dividend_Yield_Annual'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_20D', 'Amihud_Liquidity_60D', 'ISC_Ret_Score'],
            'Market Sensitivity': ['EM_Regional_Market_Sensitivity_250D'],
            'Volatility': ['EM_Volatility_60D'],
            'Medium-Term Momentum': ['Momentum_260x21D_Regional'],
            'Short-Term Momentum': ['Momentum_21D'],
            'Exchange Rate Sensitivity': ['XRate_52W_XDR'],
            'Profitability': ['Return_on_Equity_Annual', 'Return_on_Assets_Annual',
                'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
                'Sales_to_Assets_Annual', 'Gross_Margin_Annual'],
            }
    exposureConfigFile = 'exposures-sh'

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.EMAxioma2018SH')
        EMAxioma2018MH.__init__(self, modelDB, marketDB)
        self.currencyModel = riskmodels.getModelByName('FXAxioma2017USD_SH')(modelDB, marketDB)

    def setCalculators(self, modelDB, overrider=False):
        EMAxioma2018MH.setCalculators(self, modelDB, overrider=overrider)
        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=3, horizon='short', overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=3, horizon='short', overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, horizon='short', overrider=overrider)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

class EMAxioma2018MH_S(EquityModel.StatisticalModel):
    """Version 4 EM medium-horizon statistical model with GICS 2018
    """
    rm_id = 276
    revision = 1
    rms_id = 276
    numFactors = 20
    blind = [ModelFactor('Statistical Factor %d' % n, 'Statistical Factor %d' % n) for n in range(1, numFactors+1)]
    pcaHistory = 250
    gicsDate = datetime.date(2018,9,29)
    industryClassification = Classification.GICSCustomEM2(gicsDate)
    descriptorNumeraire = 'USD'

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.EMAxioma2018MH_S')
        # Set important model parameters
        ModelParameters2017.defaultModelSettings(self, scm=False, statModel=True)
        self.legacySAWeight = True
        EquityModel.StatisticalModel.__init__(self, ['SEDOL', 'CUSIP'], modelDB, marketDB)
        # So we can use the same ESTU method as the fundamental model
        self.baseModelDateMap = {datetime.date(1980,1,1): EMAxioma2018MH(modelDB, marketDB)}
        # Set up estimation universe parameters
        self.elig_parameters = ModelParameters2017.defaultRegionalModelEstuParameters(self)[1]
        # Set Calculators
        self.setCalculators(modelDB)
        # Define currency model
        self.currencyModel = riskmodels.getModelByName('FXAxioma2017USD_MH')(modelDB, marketDB)

    def setCalculators(self, modelDB, overrider=False):
        # Set up returns model
        self.returnCalculator = FactorReturns.AsymptoticPrincipalComponents2017(self.numFactors, trimExtremeExposures=True)
        self.olsReturnClass = ModelParameters2017.simpleRegressionParameters(self, modelDB, overrider=overrider)
        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=1, overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=1, overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, nwLag=1, overrider=overrider)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

class EMAxioma2018SH_S(EquityModel.StatisticalModel):
    """Version 4 EM short-horizon statistical model with GICS 2018
    """
    rm_id = 277
    revision = 1
    rms_id = 277
    numFactors = 20
    blind = [ModelFactor('Statistical Factor %d' % n, 'Statistical Factor %d' % n) for n in range(1, numFactors+1)]
    pcaHistory = 250
    gicsDate = datetime.date(2018,9,29)
    industryClassification = Classification.GICSCustomEM2(gicsDate)
    descriptorNumeraire = 'USD'

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.EMAxioma2018SH_S')
        # Set important model parameters
        ModelParameters2017.defaultModelSettings(self, scm=False, statModel=True)
        self.legacySAWeight = True
        EquityModel.StatisticalModel.__init__(self, ['SEDOL', 'CUSIP'], modelDB, marketDB)
        # So we can use the same ESTU method as the fundamental model
        self.baseModelDateMap = {datetime.date(1980,1,1): EMAxioma2018MH(modelDB, marketDB)}
        # Set up estimation universe parameters
        self.elig_parameters = ModelParameters2017.defaultRegionalModelEstuParameters(self)[1]

        # Set Calculators
        self.setCalculators(modelDB)
        # Define currency model
        self.currencyModel = riskmodels.getModelByName('FXAxioma2017USD_SH')(modelDB, marketDB)

    def setCalculators(self, modelDB, overrider=False):
        # Set up returns model
        self.returnCalculator = FactorReturns.AsymptoticPrincipalComponents2017(self.numFactors, trimExtremeExposures=True)
        self.olsReturnClass = ModelParameters2017.simpleRegressionParameters(self, modelDB, overrider=overrider)
        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=1, horizon='short', overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=1, horizon='short', overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, nwLag=1, horizon='short', overrider=overrider)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

class EMAxioma2018FL(EMAxioma2018MH):
    """EU4 factor model
    """
    rm_id = 285
    revision = 1
    rms_id = 285

    DescriptorMap = {
            'Earnings to Price Annual': ['Earnings_to_Price_Annual'],
            'Est Earnings to Price Annual': ['Est_Earnings_to_Price_12MFL_Annual'],
            'Book to Price Annual': ['Book_to_Price_Annual'],
            'Debt to Assets Annual': ['Debt_to_Assets_Annual'],
            'Debt to Equity Annual': ['Debt_to_Equity_Annual'],
            'Earnings Growth Annual': ['Earnings_Growth_RPF_Annual'],
            'Sales Growth Annual': ['Sales_Growth_RPF_Annual'],
            'Dividend Yield Annual': ['Dividend_Yield_Annual'],
            'Log of Issuer Cap': ['LnIssuerCap'],
            'Log of 60-Day ADV to Issuer Cap': ['LnTrading_Activity_60D'],
            'Log of 20-Day ADV to Issuer Cap': ['LnTrading_Activity_20D'],
            'Amihud Liquidity 60 Day': ['Amihud_Liquidity_60D'],
            'Amihud Liquidity 125 Day': ['Amihud_Liquidity_125D'],
            'Returns to Trading Days': ['ISC_Ret_Score'],
            'EM Regional Market Sensitivity 250 Day': ['EM_Regional_Market_Sensitivity_250D'],
            'EM Regional Market Sensitivity 500 Day': ['EM_Regional_Market_Sensitivity_500D'],
            'EM Volatility 60 Day': ['EM_Volatility_60D'],
            'EM Volatility 125 Day': ['EM_Volatility_125D'],
            'Annual Return Excl Prev Month': ['Momentum_260x21D_Regional'],
            'Monthly Return': ['Momentum_21D'],
            'Exchange Rate Sensitivity 52 Week XDR': ['XRate_52W_XDR'],
            'Exchange Rate Sensitivity 104 Week XDR': ['XRate_104W_XDR'],
            'Return on Assets Annual': ['Return_on_Assets_Annual'],
            'Return on Equity Annual': ['Return_on_Equity_Annual'],
            'Cash Flow to Assets Annual': ['CashFlow_to_Assets_Annual'],
            'Cash Flow to Income Annual': ['CashFlow_to_Income_Annual'],
            'Sales to Assets Annual': ['Sales_to_Assets_Annual'],
            'Gross Margin Annual': ['Gross_Margin_Annual'],
            }

    styleList = sorted(DescriptorMap.keys())

    DescriptorWeights = {}
    orthogList = {}
    exposureConfigFile = 'exposures-EMAxioma2018FL'

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.EMAxioma2018FL')
        EMAxioma2018MH.__init__(self, modelDB, marketDB)
        self.noClip_List = ['ISC_Ret_Score','Returns to Trading Days']
        self.allowMixedFreqDescriptors = False

class APAxioma2018MH(EquityModel.FundamentalModel):
    """Version 4 AP medium-horizon fundamental model with GICS 2018
    """
    rm_id = 267
    revision = 1
    rms_id = 267

    styleList = ['Value',
                 'Leverage',
                 'Growth',
                 'Profitability',
                 'Earnings Yield',
                 'Dividend Yield',
                 'Size',
                 'Liquidity',
                 'Market Sensitivity',
                 'Volatility',
                 'Medium-Term Momentum',
                 'Exchange Rate Sensitivity',
                 ]

    intercept = ModelFactor('Asian Market', 'Asian Market')
    localStructureFactors = [ModelFactor('Domestic China', 'Domestic China')]
    gicsDate = datetime.date(2018,9,29)
    industryClassification = Classification.GICSCustomNoMortgageREITs2018(gicsDate)

    DescriptorMap = {
            'Earnings Yield': ['Earnings_to_Price_Annual', 'Est_Earnings_to_Price_12MFL_Annual'],
            'Value': ['Book_to_Price_Annual'],
            'Leverage': ['Debt_to_Assets_Annual', 'Debt_to_Equity_Annual'],
            'Growth': ['Earnings_Growth_RPF_Annual', 'Sales_Growth_RPF_Annual'],
            'Dividend Yield': ['Dividend_Yield_Annual'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_60D', 'Amihud_Liquidity_125D', 'ISC_Ret_Score'],
            'Market Sensitivity': ['APAC_Regional_Market_Sensitivity_500D'],
            'Volatility': ['APAC_Volatility_125D'],
            'Medium-Term Momentum': ['Momentum_260x21D_Regional'],
            'Exchange Rate Sensitivity': ['XRate_104W_XDR'],
            'Profitability': ['Return_on_Equity_Annual', 'Return_on_Assets_Annual',
                'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
                'Sales_to_Assets_Annual', 'Gross_Margin_Annual'],
            }

    DescriptorWeights = {'Earnings Yield': [0.75, 0.25],}
    allowMixedFreqDescriptors = False
    orthogList = {'Volatility': [['Market Sensitivity'], True, 1.0],}
    exposureConfigFile = 'exposures-mh'

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.APAxioma2018MH')

        # Set up relevant styles to be created/used
        ModelParameters2017.defaultExposureParameters(
                self, self.styleList, configFile=self.exposureConfigFile, descriptorNumeraire='USD')

        # Initialise
        EquityModel.FundamentalModel.__init__(self, ['SEDOL', 'CUSIP'], modelDB, marketDB)
        self.currencyModel = riskmodels.getModelByName('FXAxioma2017USD_MH')(modelDB, marketDB)

        # Set up estimation universe parameters
        self.estu_parameters, self.elig_parameters = ModelParameters2017.defaultRegionalModelEstuParameters(self)

        # Set up RiskModel Calculators
        self.setCalculators(modelDB)

        # Set up descriptor standardization parameters
        descriptors = sorted(list(set([item for sublist
            in list(self.DescriptorMap.values()) for item in sublist])))
        regScope = Standardization_V4.RegionRelativeScope(modelDB, self.regionalStndDesc)
        gloScope = Standardization_V4.GlobalRelativeScope([d for d in descriptors if d not in self.regionalStndDesc])
        self.descriptorStandardization = Standardization_V4.BucketizedStandardization(
                [regScope, gloScope], mad_bound=15.0, exceptionNames=self.noStndDescriptor)

        # Set up standardization parameters
        regScope = Standardization_V4.RegionRelativeScope(modelDB, self.regionalStndList)
        gloScope = Standardization_V4.GlobalRelativeScope([f.name for f in self.styles if f.name not in self.regionalStndList])
        self.exposureStandardization = Standardization_V4.BucketizedStandardization(
                [regScope, gloScope], fillWithZeroList=self.fillWithZeroList)

    def generate_model_specific_exposures(self, modelDate, assetData, exposureMatrix, modelDB, marketDB):
        return self.generate_domestic_china_exposures(modelDate, assetData, exposureMatrix, modelDB, marketDB)

    def setCalculators(self, modelDB, overrider=False):
        # Set important model parameters
        ModelParameters2017.defaultModelSettings(self, scm=False)

        # Set up internal factor return regression parameters
        dummyThreshold = 6
        self.internalCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType='market',
                dummyThreshold=dummyThreshold,
                marketRegression=False,
                scndRegList=[[ExposureMatrix.LocalFactor]],
                scndRegEstus=['ChinaA'],
                kappa=5.0,
                useRealMCaps=True,
                regWeight='rootCap',
                overrider = overrider,
                )

        # Set up external regression parameters
        self.returnCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType='market',
                dummyThreshold=dummyThreshold,
                marketRegression=False,
                scndRegList=[[ExposureMatrix.LocalFactor]],
                scndRegEstus=['ChinaA'],
                kappa=25.0,
                useRealMCaps=True,
                regWeight='rootCap',
                overrider = overrider,
                )

        # This controls the FMP regression
        self.fmpCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType=None,
                dummyThreshold=dummyThreshold,
                marketRegression=False,
                scndRegList=[[ExposureMatrix.LocalFactor]],
                scndRegEstus=['ChinaA'],
                kappa=None,
                useRealMCaps=True,
                regWeight='rootCap',
                overrider = overrider,
                )

        # And this is for raw exposure proxying
        self.expProxyCalculator = ModelParameters2017.defaultExposureProxyParameters(
                self, modelDB,
                dummyType='market',
                dummyThreshold=dummyThreshold,
                kappa=5.0,
                useRealMCaps=True,
                regWeight='rootCap',
                overrider = overrider,
                )

        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=3, overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=3, overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, overrider=overrider)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

class APAxioma2018SH(APAxioma2018MH):
    """Version 4 AP medium-horizon fundamental model with GICS 2018
    """
    rm_id = 278
    revision = 1
    rms_id = 278

    styleList = ['Value',
                 'Leverage',
                 'Growth',
                 'Profitability',
                 'Earnings Yield',
                 'Dividend Yield',
                 'Size',
                 'Liquidity',
                 'Market Sensitivity',
                 'Volatility',
                 'Medium-Term Momentum',
                 'Short-Term Momentum',
                 'Exchange Rate Sensitivity',
                 ]

    DescriptorMap = {
            'Earnings Yield': ['Earnings_to_Price_Annual', 'Est_Earnings_to_Price_12MFL_Annual'],
            'Value': ['Book_to_Price_Annual'],
            'Leverage': ['Debt_to_Assets_Annual', 'Debt_to_Equity_Annual'],
            'Growth': ['Earnings_Growth_RPF_Annual', 'Sales_Growth_RPF_Annual'],
            'Dividend Yield': ['Dividend_Yield_Annual'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_20D', 'Amihud_Liquidity_60D', 'ISC_Ret_Score'],
            'Market Sensitivity': ['APAC_Regional_Market_Sensitivity_250D'],
            'Volatility': ['APAC_Volatility_60D'],
            'Medium-Term Momentum': ['Momentum_260x21D_Regional'],
            'Short-Term Momentum': ['Momentum_21D'],
            'Exchange Rate Sensitivity': ['XRate_52W_XDR'],
            'Profitability': ['Return_on_Equity_Annual', 'Return_on_Assets_Annual',
                'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
                'Sales_to_Assets_Annual', 'Gross_Margin_Annual'],
            }
    exposureConfigFile = 'exposures-sh'

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.APAxioma2018SH')
        APAxioma2018MH.__init__(self, modelDB, marketDB)
        self.currencyModel = riskmodels.getModelByName('FXAxioma2017USD_SH')(modelDB, marketDB)

    def setCalculators(self, modelDB, overrider=False):
        APAxioma2018MH.setCalculators(self, modelDB, overrider=overrider)
        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=3, horizon='short', overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=3, horizon='short', overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, horizon='short', overrider=overrider)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

class APAxioma2018MH_S(EquityModel.StatisticalModel):
    """Version 4 AP medium-horizon statistical model with GICS 2018
    """
    rm_id = 279
    revision = 1
    rms_id = 279
    numFactors = 20
    blind = [ModelFactor('Statistical Factor %d' % n, 'Statistical Factor %d' % n) for n in range(1, numFactors+1)]
    pcaHistory = 250
    gicsDate = datetime.date(2018,9,29)
    industryClassification = Classification.GICSCustomNoMortgageREITs2018(gicsDate)
    descriptorNumeraire = 'USD'

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.APAxioma2018MH_S')
        EquityModel.StatisticalModel.__init__(self, ['SEDOL', 'CUSIP'], modelDB, marketDB)
        # So we can use the same ESTU method as the fundamental model
        self.baseModelDateMap = {datetime.date(1980,1,1): APAxioma2018MH(modelDB, marketDB)}
        # Set up estimation universe parameters
        self.elig_parameters = ModelParameters2017.defaultRegionalModelEstuParameters(self)[1]
        # Set Calculators
        self.setCalculators(modelDB)
        # Define currency model
        self.currencyModel = riskmodels.getModelByName('FXAxioma2017USD_MH')(modelDB, marketDB)

    def setCalculators(self, modelDB, overrider=False):
        # Set important model parameters
        ModelParameters2017.defaultModelSettings(self, scm=False, statModel=True)
        # Set up returns model
        if self.statModel21Settings:
            self.returnCalculator = FactorReturns.AsymptoticPrincipalComponents2017(self.numFactors,
                    trimExtremeExposures=False, replaceReturns=False, flexible=False, TOL=None)
        else:
            self.returnCalculator = FactorReturns.AsymptoticPrincipalComponents2017(self.numFactors,
                    trimExtremeExposures=True)
        self.olsReturnClass = ModelParameters2017.simpleRegressionParameters(self, modelDB, overrider=overrider)
        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=1, overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=1, overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, nwLag=1, overrider=overrider)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

class APAxioma2018SH_S(EquityModel.StatisticalModel):
    """Version 4 AP short-horizon statistical model with GICS 2018
    """
    rm_id = 280
    revision = 1
    rms_id = 280
    numFactors = 20
    blind = [ModelFactor('Statistical Factor %d' % n, 'Statistical Factor %d' % n) for n in range(1, numFactors+1)]
    pcaHistory = 250
    gicsDate = datetime.date(2018,9,29)
    industryClassification = Classification.GICSCustomNoMortgageREITs2018(gicsDate)
    descriptorNumeraire = 'USD'

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.APAxioma2018SH_S')
        EquityModel.StatisticalModel.__init__(self, ['SEDOL', 'CUSIP'], modelDB, marketDB)
        # So we can use the same ESTU method as the fundamental model
        self.baseModelDateMap = {datetime.date(1980,1,1): APAxioma2018MH(modelDB, marketDB)}
        # Set up estimation universe parameters
        self.elig_parameters = ModelParameters2017.defaultRegionalModelEstuParameters(self)[1]
        # Set Calculators
        self.setCalculators(modelDB)
        # Define currency model
        self.currencyModel = riskmodels.getModelByName('FXAxioma2017USD_SH')(modelDB, marketDB)

    def setCalculators(self, modelDB, overrider=False):
        # Set important model parameters
        ModelParameters2017.defaultModelSettings(self, scm=False, statModel=True)
        # Set up returns model
        self.returnCalculator = FactorReturns.AsymptoticPrincipalComponents2017(self.numFactors, trimExtremeExposures=True)
        self.olsReturnClass = ModelParameters2017.simpleRegressionParameters(self, modelDB, overrider=overrider)
        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=1, horizon='short', overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=1, horizon='short', overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, nwLag=1, horizon='short', overrider=overrider)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

class APAxioma2018FL(APAxioma2018MH):
    """EU4 factor model
    """
    rm_id = 286
    revision = 1
    rms_id = 286

    DescriptorMap = {
            'Earnings to Price Annual': ['Earnings_to_Price_Annual'],
            'Est Earnings to Price Annual': ['Est_Earnings_to_Price_12MFL_Annual'],
            'Book to Price Annual': ['Book_to_Price_Annual'],
            'Debt to Assets Annual': ['Debt_to_Assets_Annual'],
            'Debt to Equity Annual': ['Debt_to_Equity_Annual'],
            'Earnings Growth Annual': ['Earnings_Growth_RPF_Annual'],
            'Sales Growth Annual': ['Sales_Growth_RPF_Annual'],
            'Dividend Yield Annual': ['Dividend_Yield_Annual'],
            'Log of Issuer Cap': ['LnIssuerCap'],
            'Log of 60-Day ADV to Issuer Cap': ['LnTrading_Activity_60D'],
            'Log of 20-Day ADV to Issuer Cap': ['LnTrading_Activity_20D'],
            'Amihud Liquidity 60 Day': ['Amihud_Liquidity_60D'],
            'Amihud Liquidity 125 Day': ['Amihud_Liquidity_125D'],
            'Returns to Trading Days': ['ISC_Ret_Score'],
            'APAC Regional Market Sensitivity 250 Day': ['APAC_Regional_Market_Sensitivity_250D'],
            'APAC Regional Market Sensitivity 500 Day': ['APAC_Regional_Market_Sensitivity_500D'],
            'APAC Volatility 60 Day': ['APAC_Volatility_60D'],
            'APAC Volatility 125 Day': ['APAC_Volatility_125D'],
            'Annual Return Excl Prev Month': ['Momentum_260x21D_Regional'],
            'Monthly Return': ['Momentum_21D'],
            'Exchange Rate Sensitivity 52 Week XDR': ['XRate_52W_XDR'],
            'Exchange Rate Sensitivity 104 Week XDR': ['XRate_104W_XDR'],
            'Return on Assets Annual': ['Return_on_Assets_Annual'],
            'Return on Equity Annual': ['Return_on_Equity_Annual'],
            'Cash Flow to Assets Annual': ['CashFlow_to_Assets_Annual'],
            'Cash Flow to Income Annual': ['CashFlow_to_Income_Annual'],
            'Sales to Assets Annual': ['Sales_to_Assets_Annual'],
            'Gross Margin Annual': ['Gross_Margin_Annual'],
            }

    styleList = sorted(DescriptorMap.keys())

    DescriptorWeights = {}
    orthogList = {}
    exposureConfigFile = 'exposures-APAxioma2018FL'

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.APAxioma2018FL')
        APAxioma2018MH.__init__(self, modelDB, marketDB)
        self.noClip_List = ['ISC_Ret_Score','Returns to Trading Days']
        self.allowMixedFreqDescriptors = False

class APxJPAxioma2018MH(APAxioma2018MH):
    """Version 4 AP ex-Japan medium-horizon fundamental model with GICS 2018
    """
    rm_id = 281
    revision = 1
    rms_id = 281

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.APxJPAxioma2018MH')
        APAxioma2018MH.__init__(self, modelDB, marketDB)

class APxJPAxioma2018SH(APAxioma2018SH):
    """Version 4 AP ex-Japan short-horizon fundamental model with GICS 2018
    """
    rm_id = 282
    revision = 1
    rms_id = 282

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.APxJPAxioma2018SH')
        APAxioma2018SH.__init__(self, modelDB, marketDB)

class APxJPAxioma2018MH_S(APAxioma2018MH_S):
    """Version 4 AP ex-Japan medium-horizon statistical model with GICS 2018
    """
    rm_id = 283
    revision = 1
    rms_id = 283

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.APxJPAxioma2018MH_S')
        APAxioma2018MH_S.__init__(self, modelDB, marketDB)

class APxJPAxioma2018SH_S(APAxioma2018SH_S):
    """Version 4 AP ex-Japan short-horizon statistical model with GICS 2018
    """
    rm_id = 284
    revision = 1
    rms_id = 284

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.APxJPAxioma2018SH_S')
        APAxioma2018SH_S.__init__(self, modelDB, marketDB)

class APxJPAxioma2018FL(APxJPAxioma2018MH):
    """EU4 factor model
    """
    rm_id = 287
    revision = 1
    rms_id = 287

    DescriptorMap = {
            'Earnings to Price Annual': ['Earnings_to_Price_Annual'],
            'Est Earnings to Price Annual': ['Est_Earnings_to_Price_12MFL_Annual'],
            'Book to Price Annual': ['Book_to_Price_Annual'],
            'Debt to Assets Annual': ['Debt_to_Assets_Annual'],
            'Debt to Equity Annual': ['Debt_to_Equity_Annual'],
            'Earnings Growth Annual': ['Earnings_Growth_RPF_Annual'],
            'Sales Growth Annual': ['Sales_Growth_RPF_Annual'],
            'Dividend Yield Annual': ['Dividend_Yield_Annual'],
            'Log of Issuer Cap': ['LnIssuerCap'],
            'Log of 60-Day ADV to Issuer Cap': ['LnTrading_Activity_60D'],
            'Log of 20-Day ADV to Issuer Cap': ['LnTrading_Activity_20D'],
            'Amihud Liquidity 60 Day': ['Amihud_Liquidity_60D'],
            'Amihud Liquidity 125 Day': ['Amihud_Liquidity_125D'],
            'Returns to Trading Days': ['ISC_Ret_Score'],
            'APxJP Regional Market Sensitivity 250 Day': ['APAC_Regional_Market_Sensitivity_250D'],
            'APxJP Regional Market Sensitivity 500 Day': ['APAC_Regional_Market_Sensitivity_500D'],
            'APxJP Volatility 60 Day': ['APAC_Volatility_60D'],
            'APxJP Volatility 125 Day': ['APAC_Volatility_125D'],
            'Annual Return Excl Prev Month': ['Momentum_260x21D_Regional'],
            'Monthly Return': ['Momentum_21D'],
            'Exchange Rate Sensitivity 52 Week XDR': ['XRate_52W_XDR'],
            'Exchange Rate Sensitivity 104 Week XDR': ['XRate_104W_XDR'],
            'Return on Assets Annual': ['Return_on_Assets_Annual'],
            'Return on Equity Annual': ['Return_on_Equity_Annual'],
            'Cash Flow to Assets Annual': ['CashFlow_to_Assets_Annual'],
            'Cash Flow to Income Annual': ['CashFlow_to_Income_Annual'],
            'Sales to Assets Annual': ['Sales_to_Assets_Annual'],
            'Gross Margin Annual': ['Gross_Margin_Annual'],
            }

    styleList = sorted(DescriptorMap.keys())

    DescriptorWeights = {}
    orthogList = {}
    exposureConfigFile = 'exposures-APxJPAxioma2018FL'

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.APxJPAxioma2018FL')
        APxJPAxioma2018MH.__init__(self, modelDB, marketDB)
        self.noClip_List = ['ISC_Ret_Score','Returns to Trading Days']
        self.allowMixedFreqDescriptors = False

class WWAxioma2018FL(WWAxioma2017MH):
    """Global Extended Factor Library
    """
    rm_id = 300
    revision = 1
    rms_id = 300
   
    DescriptorMap = {
            'US 10Y Interest Rate Sensitivity 104W (net of market)':  ['USLTRate_Sensitivity_104W'],
            'US 10Y Interest Rate Sensitivity 104W':                  ['USLTRate_Sensitivity_NoMkt_104W'],
            'US 6M Interest Rate Sensitivity 104W (net of market)':   ['USSTRate_Sensitivity_104W'],
            'US 6M Interest Rate Sensitivity 104W':                   ['USSTRate_Sensitivity_NoMkt_104W'],
            'US Term Spread Sensitivity 104W (net of market)':        ['USTermSpread_Sensitivity_104W'],
            'US Term Spread Sensitivity 104W':                        ['USTermSpread_Sensitivity_NoMkt_104W'],
            'US Credit Spread Sensitivity 104W (net of market)':      ['USCreditSpread_Sensitivity_104W'],
            'US Credit Spread Sensitivity 104W':                      ['USCreditSpread_Sensitivity_NoMkt_104W'],
            'Oil Sensitivity 104W (net of market)':                   ['Oil_Sensitivity_104W'],
            'Oil Sensitivity 104W':                                   ['Oil_Sensitivity_NoMkt_104W'],
            'Gold Sensitivity 104W (net of market)':                  ['Gold_Sensitivity_104W'],
            'Gold Sensitivity 104W':                                  ['Gold_Sensitivity_NoMkt_104W'],
            'Commodity Sensitivity 104W (net of market)':             ['Commodity_Sensitivity_104W'],
            'Commodity Sensitivity 104W':                             ['Commodity_Sensitivity_NoMkt_104W'],
            'US BEI Sensitivity 104W (net of market)':                ['USBEI_Sensitivity_104W'],
            'US BEI Sensitivity 104W':                                ['USBEI_Sensitivity_NoMkt_104W'],
            'GB 10Y Interest Rate Sensitivity 104W (net of market)':  ['GBLTRate_Sensitivity_104W'],
            'GB 10Y Interest Rate Sensitivity 104W':                  ['GBLTRate_Sensitivity_NoMkt_104W'],
            'JP 10Y Interest Rate Sensitivity 104W (net of market)':  ['JPLTRate_Sensitivity_104W'],
            'JP 10Y Interest Rate Sensitivity 104W':                  ['JPLTRate_Sensitivity_NoMkt_104W'],
            'EU 10Y Interest Rate Sensitivity 104W (net of market)':  ['EULTRate_Sensitivity_104W'],
            'EU 10Y Interest Rate Sensitivity 104W':                  ['EULTRate_Sensitivity_NoMkt_104W'],
            'GB 6M Interest Rate Sensitivity 104W (net of market)':   ['GBSTRate_Sensitivity_104W'],
            'GB 6M Interest Rate Sensitivity 104W':                   ['GBSTRate_Sensitivity_NoMkt_104W'],
            'JP 6M Interest Rate Sensitivity 104W (net of market)':   ['JPSTRate_Sensitivity_104W'],
            'JP 6M Interest Rate Sensitivity 104W':                   ['JPSTRate_Sensitivity_NoMkt_104W'],
            'EU 6M Interest Rate Sensitivity 104W (net of market)':   ['EUSTRate_Sensitivity_104W'],
            'EU 6M Interest Rate Sensitivity 104W':                   ['EUSTRate_Sensitivity_NoMkt_104W'],
            'GB Term Spread Sensitivity 104W (net of market)':        ['GBTermSpread_Sensitivity_104W'],
            'GB Term Spread Sensitivity 104W':                        ['GBTermSpread_Sensitivity_NoMkt_104W'],
            'JP Term Spread Sensitivity 104W (net of market)':        ['JPTermSpread_Sensitivity_104W'],
            'JP Term Spread Sensitivity 104W':                        ['JPTermSpread_Sensitivity_NoMkt_104W'],
            'EU Term Spread Sensitivity 104W (net of market)':        ['EUTermSpread_Sensitivity_104W'],
            'EU Term Spread Sensitivity 104W':                        ['EUTermSpread_Sensitivity_NoMkt_104W'],
            'GB Credit Spread Sensitivity 104W (net of market)':      ['GBCreditSpread_Sensitivity_104W'],
            'GB Credit Spread Sensitivity 104W ':                     ['GBCreditSpread_Sensitivity_NoMkt_104W'],
            'JP Credit Spread Sensitivity 104W (net of market)':      ['JPCreditSpread_Sensitivity_104W'],
            'JP Credit Spread Sensitivity 104W':                      ['JPCreditSpread_Sensitivity_NoMkt_104W'],
            'EU Credit Spread Sensitivity 104W (net of market)':      ['EUCreditSpread_Sensitivity_104W'],
            'EU Credit Spread Sensitivity 104W':                      ['EUCreditSpread_Sensitivity_NoMkt_104W'],
            #'US Credit Spread Sensitivity 36M (net of market)':       ['USCreditSpread_Sensitivity_36M'],
            #'US Credit Spread Sensitivity 36M':                       ['USCreditSpread_Sensitivity_NoMkt_36M'],
            #'GB Credit Spread Sensitivity 36M (net of market)':       ['GBCreditSpread_Sensitivity_36M'],
            #'GB Credit Spread Sensitivity 36M':                       ['GBCreditSpread_Sensitivity_NoMkt_36M'],
            #'JP Credit Spread Sensitivity 36M (net of market)':       ['JPCreditSpread_Sensitivity_36M'],
            #'JP Credit Spread Sensitivity 36M':                       ['JPCreditSpread_Sensitivity_NoMkt_36M'],
            #'EU Credit Spread Sensitivity 36M (net of market)':       ['EUCreditSpread_Sensitivity_36M'],
            #'EU Credit Spread Sensitivity 36M':                       ['EUCreditSpread_Sensitivity_NoMkt_36M'],
            }

    styleList = sorted(DescriptorMap.keys())

    DescriptorWeights = {}
    orthogList = {}
    exposureConfigFile = 'exposures-WWAxioma2018FL'

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.WWAxioma2018FL')
        WWAxioma2017MH.__init__(self, modelDB, marketDB)

        # Set up descriptor standardization parameters
        descriptors = sorted(list(set([item for sublist
            in list(self.DescriptorMap.values()) for item in sublist])))
        regScope = Standardization_V4.RegionRelativeScope(
                modelDB, self.regionalStndDesc)
        gloScope = Standardization_V4.GlobalRelativeScope(
                [d for d in descriptors if d not in self.regionalStndDesc])
        self.descriptorStandardization = Standardization_V4.BucketizedStandardization(
                [regScope, gloScope], mad_bound=8.0,
                exceptionNames=self.noStndDescriptor, forceMAD=True)
        self.allowMixedFreqDescriptors = False

        # Set up standardization parameters
        regScope = Standardization_V4.RegionRelativeScope(modelDB, self.regionalStndList)
        gloScope = Standardization_V4.GlobalRelativeScope([])
        self.exposureStandardization = Standardization_V4.BucketizedStandardization(
                [regScope, gloScope], fillWithZeroList=self.fillWithZeroList,
                exceptionNames=self.styleList)

class NAAxioma2019MH(EquityModel.FundamentalModel):
    """Version 4 AP medium-horizon fundamental model with GICS 2018
    """
    rm_id = 330
    revision = 1
    rms_id = 330

    styleList = ['Value',
                 'Leverage',
                 'Growth',
                 'Profitability',
                 'Earnings Yield',
                 'Dividend Yield',
                 'Size',
                 'Liquidity',
                 'Market Sensitivity',
                 'Volatility',
                 'Medium-Term Momentum',
                 'Exchange Rate Sensitivity',
                 ]

    interceptFactor = 'North American Market'
    intercept = ModelFactor(interceptFactor, interceptFactor)
    gicsDate = datetime.date(2018, 9, 29)
    industryClassification = Classification.GICSCustomNA4(gicsDate)
    
    DescriptorMap = {
            'Earnings Yield': ['Earnings_to_Price_Quarterly', 'Est_Earnings_to_Price_12MFL_Quarterly'],
            'Value': ['Book_to_Price_Quarterly'],
            'Leverage': ['Debt_to_Assets_Quarterly', 'Debt_to_Equity_Quarterly'],
            'Growth': ['Earnings_Growth_RPF_AFQ', 'Sales_Growth_RPF_AFQ'],
            'Dividend Yield': ['Dividend_Yield_Quarterly'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_60D', 'Amihud_Liquidity_125D', 'ISC_Ret_Score'],
            'Market Sensitivity': ['NA_Regional_Market_Sensitivity_500D'],
            'Volatility': ['NA_Volatility_125D'],
            'Medium-Term Momentum': ['Momentum_260x21D_Regional'], 
            'Exchange Rate Sensitivity': ['XRate_104W_XDR'],
            'Profitability': ['Return_on_Equity_Quarterly', 'Return_on_Assets_Quarterly',
                'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
                'Sales_to_Assets_Quarterly', 'Gross_Margin_Quarterly'],
            }


    DescriptorWeights = {'Earnings Yield': [0.75, 0.25],}
    orthogList = {'Volatility': [['Market Sensitivity'], True, 1.0],}
    exposureConfigFile = 'exposures-NAAxioma2019MH'

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.NAAxioma2019MH')

        # Set up relevant styles to be created/used
        ModelParameters2017.defaultExposureParameters(
                self, self.styleList, configFile=self.exposureConfigFile, descriptorNumeraire='USD')

        # Initialise
        EquityModel.FundamentalModel.__init__(self, ['SEDOL', 'CUSIP'], modelDB, marketDB)
        self.currencyModel = riskmodels.getModelByName('FXAxioma2017USD_MH')(modelDB, marketDB)

        self.elig_parameters = {'assetTypes': self.commonStockTypes + ['REIT'] + ['UnitT'], #None,
                               'excludeTypes': None, #self.etfAssetTypes + self.otherAssetTypes,
                               'use_isin_country_Flag': False,
                               'remove_China_AB': True,
                               'addBack_H_DR': False}

        # So we can use the same ESTU method as the single country fundamental model
        self.baseSCMs = [USAxioma2016MH(modelDB, marketDB),
                         CAAxioma2018MH(modelDB, marketDB)]

        # Set up RiskModel Calculators
        self.setCalculators(modelDB)

        # Set up descriptor standardization parameters
        descriptors = sorted(list(set([item for sublist
            in list(self.DescriptorMap.values()) for item in sublist])))
        gloScope = Standardization_V4.GlobalRelativeScope([d for d in descriptors if d not in self.regionalStndDesc])
        self.descriptorStandardization = Standardization_V4.BucketizedStandardization(
                [gloScope], mad_bound=15.0, exceptionNames=self.noStndDescriptor)

        # Set up standardization parameters
        gloScope = Standardization_V4.GlobalRelativeScope([f.name for f in self.styles if f.name not in self.regionalStndList])
        self.exposureStandardization = Standardization_V4.BucketizedStandardization(
                [gloScope], fillWithZeroList=self.fillWithZeroList)

    def setCalculators(self, modelDB, overrider=False):
        # Set important model parameters
        ModelParameters2017.defaultModelSettings(self, scm=False)

        # Set up internal factor return regression parameters
        dummyThreshold = 6
        self.internalCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType='Industry Groups',
                dummyThreshold=dummyThreshold,
                marketRegression=False,
                kappa=5.0,
                useRealMCaps=True,
                regWeight='rootCap',
                overrider = overrider,
                )

        # Set up external regression parameters
        self.returnCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType='Industry Groups',
                dummyThreshold=dummyThreshold,
                marketRegression=False,
                kappa=25.0,
                useRealMCaps=True,
                regWeight='rootCap',
                overrider = overrider,
                )

        # This controls the FMP regression
        self.fmpCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType=None,
                dummyThreshold=dummyThreshold,
                marketRegression=False,
                kappa=None,
                useRealMCaps=True,
                regWeight='rootCap',
                overrider = overrider,
                )

        # And this is for raw exposure proxying
        self.expProxyCalculator = ModelParameters2017.defaultExposureProxyParameters(
                self, modelDB,
                dummyType='market',
                dummyThreshold=dummyThreshold,
                kappa=5.0,
                useRealMCaps=True,
                regWeight='rootCap',
                overrider = overrider,
                )

        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=3, overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=3, overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, overrider=overrider)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

    def generate_estimation_universe(self, modelDate, data, exposureMatrix, modelDB, marketDB, excludeFactors=[], grandfatherRMS_ID=None):

         #################################### added section ####################################
        # This section is added for regional model where the estu is generated by combining the underlying
        # SCMs' estus together, for NA4, this is effective combining US and CA country models
        from riskmodels import AssetProcessor_V4
        if hasattr(self, 'baseSCMs'):
 
            estu_idx_list = set()

            # loop through all the given SCMs
            for n,scm in enumerate(self.baseSCMs):
                self.log.info('########## Processing estimation universe for scm: %s begin ##########', scm.name)
                 # generate risk model instance
                scm.setFactorsForDate(modelDate, modelDB)
                
                #data_temp = copy.deepcopy(data)
                data_temp = AssetProcessor_V4.AssetProcessor(modelDate, modelDB, marketDB, self.getDefaultAPParameters())
                data_temp.process_asset_information(self.rmg, universe=data.rmgAssetMap[scm.rmg[0]])
                scm_expM = Matrices.ExposureMatrix(data_temp.universe)
                scm_expM = scm.generate_industry_exposures(modelDate, modelDB, marketDB, scm_expM)

                # Generate universe of eligible assets for SCM
                scm_estuCls = EstimationUniverse_V4.ConstructEstimationUniverse(\
                        modelDate, data_temp.universe, scm, modelDB, marketDB, debugOutput=self.debuggingReporting)
                data_temp.eligibleUniverse = scm_estuCls.generate_eligible_universe(data_temp)

                # overwrite estimation universe parameteres if necessary
                if hasattr(self, 'estu_parameters'):
                    scm.estu_parameters = self.estu_parameters.copy()
                if grandfatherRMS_ID is None:
                    grandfatherRMS_ID = self.rms_id
                scm_estu = scm.generate_estimation_universe(
                    modelDate, data_temp, scm_expM, modelDB, marketDB, grandfatherRMS_ID=grandfatherRMS_ID)
                self.log.info('estu size for %s is %d', scm.name, len(scm_estu))

                estu_idx_list = estu_idx_list.union(scm_estu)
                self.log.info('########## Processing estimation universe for scm: %s end ##########', scm.name)

                # update NA4 model's estuMap dictionary by combining base scm's estuMap, 
                # note that the model need to be set to a base SCM first in order to have the 
                # fields: assets, qualify. Assets are list of subissue ids. 
                # This directly map to rmi_estu_v3 table in modeldb
                for key in list(scm.estuMap.keys()):
                    if key not in self.estuMap:
                        del scm.estuMap[key]
                    else:
                        if n==0:
                            self.estuMap[key] = scm.estuMap[key]
                        else:
                            self.estuMap[key].assets = self.estuMap[key].assets.union(scm.estuMap[key].assets)
                            self.estuMap[key].qualify = self.estuMap[key].qualify.union(scm.estuMap[key].qualify)

            # filter the list to make sure each element is unique
            for key in self.estuMap.keys():
                self.estuMap[key].assets = self.estuMap[key].assets.intersection(data.eligibleUniverse)
                self.estuMap[key].qualify = self.estuMap[key].qualify.intersection(data.eligibleUniverse)

        # The final estu index is the union of all the results with no duplicate and intersect with eligible universe
        return estu_idx_list.intersection(data.eligibleUniverse)

class NAAxioma2019SH(NAAxioma2019MH):
    """
        NA4 fundamental short-horizon model
    """
    # Model Parameters:
    rm_id,revision,rms_id = [331,1,331]

    styleList = ['Value',
                 'Leverage',
                 'Growth',
                 'Profitability',
                 'Earnings Yield',
                 'Dividend Yield',
                 'Size',
                 'Liquidity',
                 'Market Sensitivity',
                 'Volatility',
                 'Medium-Term Momentum',
                 'Short-Term Momentum',
                 'Exchange Rate Sensitivity',
                 ]
    DescriptorMap = {
            'Earnings Yield': ['Earnings_to_Price_Quarterly', 'Est_Earnings_to_Price_12MFL_Quarterly'],
            'Value': ['Book_to_Price_Quarterly'],
            'Leverage': ['Debt_to_Assets_Quarterly', 'Debt_to_Equity_Quarterly'],
            'Growth': ['Earnings_Growth_RPF_AFQ', 'Sales_Growth_RPF_AFQ'],
            'Dividend Yield': ['Dividend_Yield_Quarterly'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_20D', 'Amihud_Liquidity_60D', 'ISC_Ret_Score'],
            'Market Sensitivity': ['NA_Regional_Market_Sensitivity_250D'],
            'Volatility': ['NA_Volatility_60D'],
            'Medium-Term Momentum': ['Momentum_260x21D_Regional'], 
            'Short-Term Momentum': ['Momentum_21D'], 
            'Exchange Rate Sensitivity': ['XRate_52W_XDR'],
            'Profitability': ['Return_on_Equity_Quarterly', 'Return_on_Assets_Quarterly',
                'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
                'Sales_to_Assets_Quarterly', 'Gross_Margin_Quarterly'],
            }

    exposureConfigFile = 'exposures-NAAxioma2019SH'

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.NAAxioma2019SH')
        NAAxioma2019MH.__init__(self, modelDB, marketDB)
        self.currencyModel = riskmodels.getModelByName('FXAxioma2017USD_SH')(modelDB, marketDB)

    def setCalculators(self, modelDB, overrider=False):
        NAAxioma2019MH.setCalculators(self, modelDB, overrider=overrider)
        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=3, horizon='short', overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=3, horizon='short', overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, horizon='short', overrider=overrider)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

class NAAxioma2019MH_S(EquityModel.StatisticalModel):
    """Version 4 NA medium-horizon statistical model with GICS 2018
    """
    rm_id = 332
    revision = 1
    rms_id = 332
    numFactors = 20
    blind = [ModelFactor('Statistical Factor %d' % n, 'Statistical Factor %d' % n) for n in range(1, numFactors+1)]
    pcaHistory = 250
    gicsDate = datetime.date(2018,9,29)
    industryClassification = Classification.GICSCustomNA4(gicsDate)
    descriptorNumeraire = 'USD'

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.NAAxioma2019MH_S')
        EquityModel.StatisticalModel.__init__(self, ['SEDOL', 'CUSIP'], modelDB, marketDB)
        # So we can use the same ESTU method as the fundamental model
        self.baseModelDateMap = {datetime.date(1980,1,1): NAAxioma2019MH(modelDB, marketDB)}
        # Set up estimation universe parameters
        self.elig_parameters = {'assetTypes': self.commonStockTypes + ['REIT'] + ['UnitT'], #None,
                               'excludeTypes': None, #self.etfAssetTypes + self.otherAssetTypes,
                               'use_isin_country_Flag': False,
                               'remove_China_AB':True,
                               'addBack_H_DR':False}
        # Set Calculators
        self.setCalculators(modelDB)
        # Define currency model
        self.currencyModel = riskmodels.getModelByName('FXAxioma2017USD_MH')(modelDB, marketDB)

    def setCalculators(self, modelDB, overrider=False):
        # Set important model parameters
        ModelParameters2017.defaultModelSettings(self, scm=False, statModel=True)
        # Set up returns model
        self.returnCalculator = FactorReturns.AsymptoticPrincipalComponents2017(self.numFactors, trimExtremeExposures=True)
        self.olsReturnClass = ModelParameters2017.simpleRegressionParameters(self, modelDB, overrider=overrider)
        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=1, overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=1, overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, nwLag=1, overrider=overrider)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

class NAAxioma2019SH_S(NAAxioma2019MH_S):
    """Version 4 NA short-horizon statistical model with GICS 2018
    """
    rm_id = 333
    revision = 1
    rms_id = 333

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.NAAxioma2019SH_S')
        # Set important model parameters
        NAAxioma2019MH_S.__init__(self, modelDB, marketDB)
        # Define currency model
        self.currencyModel = riskmodels.getModelByName('FXAxioma2017USD_SH')(modelDB, marketDB)

    def setCalculators(self, modelDB, overrider=False):
        # Set important model parameters
        ModelParameters2017.defaultModelSettings(self, scm=False, statModel=True)
        # Set up returns model
        self.returnCalculator = FactorReturns.AsymptoticPrincipalComponents2017(self.numFactors, trimExtremeExposures=True)
        self.olsReturnClass = ModelParameters2017.simpleRegressionParameters(self, modelDB, overrider=overrider)
        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=1,  horizon='short', overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=1,  horizon='short', overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, nwLag=1,  horizon='short', overrider=overrider)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

class NAAxioma2019FL(NAAxioma2019MH):
    """NA4 factor model
    """
    rm_id = 334
    revision = 1
    rms_id = 334

    DescriptorMap = {
            'Amihud Liquidity 60 Day': ['Amihud_Liquidity_60D'],
            'Amihud Liquidity 125 Day': ['Amihud_Liquidity_125D'],
            'Book to Price Quarterly': ['Book_to_Price_Quarterly'],
            'Cash Flow to Assets Annual': ['CashFlow_to_Assets_Annual'],
            'Cash Flow to Income Annual': ['CashFlow_to_Income_Annual'],
            'Debt to Assets Quarterly': ['Debt_to_Assets_Quarterly'],
            'Debt to Equity Quarterly': ['Debt_to_Equity_Quarterly'],
            'Dividend Yield Quarterly': ['Dividend_Yield_Quarterly'],
            'Earnings to Price Quarterly': ['Earnings_to_Price_Quarterly'],
            'Est Earnings to Price Quarterly': ['Est_Earnings_to_Price_12MFL_Quarterly'],
            'Earnings Growth Quarterly': ['Earnings_Growth_RPF_AFQ'],
            'Gross Margin Quarterly': ['Gross_Margin_Quarterly'],
            'Returns to Trading Days': ['ISC_Ret_Score'],   
            'Log of Issuer Cap': ['LnIssuerCap'], 
            'Log of 20-Day ADV to Issuer Cap': ['LnTrading_Activity_20D'],
            'Log of 60-Day ADV to Issuer Cap': ['LnTrading_Activity_60D'],
            'Annual Return Excl Prev Month': ['Momentum_260x21D_Regional'], 
            'Monthly Return': ['Momentum_21D'],  
            'NA Regional Market Sensitivity 250 Day': ['NA_Regional_Market_Sensitivity_250D'],
            'NA Regional Market Sensitivity 500 Day': ['NA_Regional_Market_Sensitivity_500D'],
            'Return on Assets Quarterly': ['Return_on_Assets_Quarterly'],
            'Return on Equity Quarterly': ['Return_on_Equity_Quarterly'],
            'Sales to Assets Quarterly': ['Sales_to_Assets_Quarterly'],
            'Sales Growth Quarterly': ['Sales_Growth_RPF_AFQ'],
            'NA Volatility 60 Day': ['NA_Volatility_60D'],
            'NA Volatility 125 Day': ['NA_Volatility_125D'],
            'Exchange Rate Sensitivity 52 Week XDR': ['XRate_52W_XDR'],
            'Exchange Rate Sensitivity 104 Week XDR': ['XRate_104W_XDR'],
            }

    styleList = sorted(DescriptorMap.keys())

    DescriptorWeights = {}
    orthogList = {}
    exposureConfigFile = 'exposures-NAAxioma2019FL'

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.NAAxioma2019FL')
        NAAxioma2019MH.__init__(self, modelDB, marketDB)
        self.noClip_List = ['ISC_Ret_Score','Returns to Trading Days']
        self.allowMixedFreqDescriptors = True

class DMAxioma2020MH(EquityModel.FundamentalModel):
    """Version 4 EAFE medium-horizon fundamental model
    """
    rm_id = 335
    revision = 1
    rms_id = 335

    styleList = ['Value',
                 'Leverage',
                 'Growth',
                 'Profitability',
                 'Earnings Yield',
                 'Dividend Yield',
                 'Size',
                 'Liquidity',
                 'Market Sensitivity',
                 'Volatility',
                 'Medium-Term Momentum',
                 'Exchange Rate Sensitivity',
                 ]

    intercept = ModelFactor('Developed Market', 'Developed Market')
    gicsDate = datetime.date(2018,9,29)
    industryClassification = Classification.GICSCustomNoMortgageREITs2018(gicsDate)

    DescriptorMap = {
            'Earnings Yield': ['Earnings_to_Price_Annual', 'Est_Earnings_to_Price_12MFL_Annual'],
            'Value': ['Book_to_Price_Annual'],
            'Leverage': ['Debt_to_Assets_Annual', 'Debt_to_Equity_Annual'],
            'Growth': ['Earnings_Growth_RPF_Annual', 'Sales_Growth_RPF_Annual'],
            'Dividend Yield': ['Dividend_Yield_Annual'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_60D', 'Amihud_Liquidity_125D', 'ISC_Ret_Score'],
            'Market Sensitivity': ['DMxUS_Regional_Market_Sensitivity_500D'],
            'Volatility': ['DMxUS_Volatility_125D'],
            'Medium-Term Momentum': ['Momentum_260x21D_Regional'],
            'Exchange Rate Sensitivity': ['XRate_104W_XDR'],
            'Profitability': ['Return_on_Equity_Annual', 'Return_on_Assets_Annual',
                'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
                'Sales_to_Assets_Annual', 'Gross_Margin_Annual'],
            }

    DescriptorWeights = {'Earnings Yield': [0.75, 0.25],}
    orthogList = {'Volatility': [['Market Sensitivity'], True, 1.0],}
    exposureConfigFile = 'exposures-mh'

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.DMAxioma2020MH')

        # Set up relevant styles to be created/used
        ModelParameters2017.defaultExposureParameters(
                self, self.styleList, configFile=self.exposureConfigFile, descriptorNumeraire='USD')

        # Initialise
        EquityModel.FundamentalModel.__init__(self, ['SEDOL', 'CUSIP'], modelDB, marketDB)
        self.currencyModel = riskmodels.getModelByName('FXAxioma2017USD_MH')(modelDB, marketDB)

        # Set up estimation universe parameters
        self.estu_parameters, self.elig_parameters = ModelParameters2017.defaultRegionalModelEstuParameters(self)

        # Set up RiskModel Calculators
        self.setCalculators(modelDB)

        # Set up descriptor standardization parameters
        descriptors = sorted(list(set([item for sublist
            in list(self.DescriptorMap.values()) for item in sublist])))
        regScope = Standardization_V4.RegionRelativeScope(
                modelDB, self.regionalStndDesc)
        gloScope = Standardization_V4.GlobalRelativeScope(
                [d for d in descriptors if d not in self.regionalStndDesc])
        self.descriptorStandardization = Standardization_V4.BucketizedStandardization(
                [regScope, gloScope], mad_bound=15.0, exceptionNames=self.noStndDescriptor)

        # Set up standardization parameters
        regScope = Standardization_V4.RegionRelativeScope(modelDB, self.regionalStndList)
        gloScope = Standardization_V4.GlobalRelativeScope(
                [f.name for f in self.styles if f.name not in self.regionalStndList])
        self.exposureStandardization = Standardization_V4.BucketizedStandardization(
                [regScope, gloScope], fillWithZeroList=self.fillWithZeroList)

        # Manually reassign select assets to RMG and currency
        self.rmgOverride = dict()
        # Force all RDS issues to have GB exposure
        self.rmgOverride['CIJBKGWL8'] = [datetime.date(1980,1,1), datetime.date(2999,12,31), 'GB', 'GB']

    def setCalculators(self, modelDB, overrider=False):
        # Set important model parameters
        ModelParameters2017.defaultModelSettings(self, scm=False)

        # Set up internal factor return regression parameters
        dummyThreshold = 6
        self.internalCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType='market',
                dummyThreshold=dummyThreshold,
                marketRegression=False,
                kappa=5.0,
                useRealMCaps=True,
                regWeight='rootCap',
                thinWeightMultiplier='simple',
                overrider = overrider,
                )

        # Set up external regression parameters
        self.returnCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType='market',
                dummyThreshold=dummyThreshold,
                marketRegression=False,
                kappa=25.0,
                useRealMCaps=True,
                regWeight='rootCap',
                thinWeightMultiplier='simple',
                overrider = overrider,
                )

        # This controls the FMP regression
        self.fmpCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType=None,
                dummyThreshold=dummyThreshold,
                marketRegression=False,
                kappa=None,
                useRealMCaps=True,
                regWeight='rootCap',
                thinWeightMultiplier='simple',
                overrider = overrider,
                )

        # And this is for raw exposure proxying
        self.expProxyCalculator = ModelParameters2017.defaultExposureProxyParameters(
                self, modelDB,
                dummyType='market',
                dummyThreshold=dummyThreshold,
                kappa=5.0,
                useRealMCaps=True,
                regWeight='rootCap',
                thinWeightMultiplier='simple',
                overrider = overrider,
                )

        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=3, overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=3, overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, overrider=overrider)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

class DMAxioma2020SH(DMAxioma2020MH):
    """Version 4 EAFE short-horizon fundamental model
    """
    rm_id = 336
    revision = 1
    rms_id = 336

    styleList = ['Value',
                 'Leverage',
                 'Growth',
                 'Profitability',
                 'Earnings Yield',
                 'Dividend Yield',
                 'Size',
                 'Liquidity',
                 'Market Sensitivity',
                 'Volatility',
                 'Medium-Term Momentum',
                 'Short-Term Momentum',
                 'Exchange Rate Sensitivity',
                 ]

    DescriptorMap = {
            'Earnings Yield': ['Earnings_to_Price_Annual', 'Est_Earnings_to_Price_12MFL_Annual'],
            'Value': ['Book_to_Price_Annual'],
            'Leverage': ['Debt_to_Assets_Annual', 'Debt_to_Equity_Annual'],
            'Growth': ['Earnings_Growth_RPF_Annual', 'Sales_Growth_RPF_Annual'],
            'Dividend Yield': ['Dividend_Yield_Annual'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_20D', 'Amihud_Liquidity_60D', 'ISC_Ret_Score'],
            'Market Sensitivity': ['DMxUS_Regional_Market_Sensitivity_250D'],
            'Volatility': ['DMxUS_Volatility_60D'],
            'Medium-Term Momentum': ['Momentum_260x21D_Regional'],
            'Short-Term Momentum': ['Momentum_21D'],
            'Exchange Rate Sensitivity': ['XRate_52W_XDR'],
            'Profitability': ['Return_on_Equity_Annual', 'Return_on_Assets_Annual',
                'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
                'Sales_to_Assets_Annual', 'Gross_Margin_Annual'],
            }
    exposureConfigFile = 'exposures-sh'

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.DMAxioma2020SH')
        DMAxioma2020MH.__init__(self, modelDB, marketDB)
        self.currencyModel = riskmodels.getModelByName('FXAxioma2017USD_SH')(modelDB, marketDB)

    def setCalculators(self, modelDB, overrider=False):
        DMAxioma2020MH.setCalculators(self, modelDB, overrider=overrider)
        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=3, horizon='short', overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=3, horizon='short', overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, horizon='short', overrider=overrider)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

class DMAxioma2020MH_S(EquityModel.StatisticalModel):
    """Version 4 EAFE medium-horizon statistical model
    """
    rm_id = 337
    revision = 1
    rms_id = 337
    numFactors = 20
    blind = [ModelFactor('Statistical Factor %d' % n, 'Statistical Factor %d' % n)
                         for n in range(1, numFactors+1)]
    pcaHistory = 250
    gicsDate = datetime.date(2018,9,29)
    industryClassification = Classification.GICSCustomNoMortgageREITs2018(gicsDate)
    descriptorNumeraire = 'USD'

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.DMAxioma2020MH_S')
        EquityModel.StatisticalModel.__init__(self, ['SEDOL', 'CUSIP'], modelDB, marketDB)
        # So we can use the same ESTU method as the fundamental model
        self.baseModelDateMap = {datetime.date(1980,1,1): DMAxioma2020MH(modelDB, marketDB)}
        # Set up estimation universe parameters
        self.elig_parameters = ModelParameters2017.defaultRegionalModelEstuParameters(self)[1]
        # Set Calculators
        self.setCalculators(modelDB)
        # Define currency model
        self.currencyModel = riskmodels.getModelByName('FXAxioma2017USD_MH')(modelDB, marketDB)

        # Manually reassign select assets to RMG and currency
        self.rmgOverride = dict()
        # Force all RDS issues to have GB exposure
        self.rmgOverride['CIJBKGWL8'] = [datetime.date(1980,1,1), datetime.date(2999,12,31), 'GB', 'GB']

    def setCalculators(self, modelDB, overrider=False):
        # Set important model parameters
        ModelParameters2017.defaultModelSettings(self, scm=False, statModel=True)
        # Set up returns model
        self.returnCalculator = FactorReturns.AsymptoticPrincipalComponents2017(self.numFactors, trimExtremeExposures=True)
        self.olsReturnClass = ModelParameters2017.simpleRegressionParameters(self, modelDB, overrider=overrider)
        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=1, overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=1, overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, nwLag=1, overrider=overrider)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

class DMAxioma2020SH_S(EquityModel.StatisticalModel):
    """Version 4 EAFE short-horizon statistical model
    """
    rm_id = 338
    revision = 1
    rms_id = 338
    numFactors = 20
    blind = [ModelFactor('Statistical Factor %d' % n, 'Statistical Factor %d' % n)
                         for n in range(1, numFactors+1)]
    pcaHistory = 250
    gicsDate = datetime.date(2018,9,29)
    industryClassification = Classification.GICSCustomNoMortgageREITs2018(gicsDate)
    descriptorNumeraire = 'USD'

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.DMAxioma2020SH_S')
        EquityModel.StatisticalModel.__init__(self, ['SEDOL', 'CUSIP'], modelDB, marketDB)
        # So we can use the same ESTU method as the fundamental model
        self.baseModelDateMap = {datetime.date(1980,1,1): DMAxioma2020MH(modelDB, marketDB)}
        # Set up estimation universe parameters
        self.elig_parameters = ModelParameters2017.defaultRegionalModelEstuParameters(self)[1]
        # Set Calculators
        self.setCalculators(modelDB)
        # Define currency model
        self.currencyModel = riskmodels.getModelByName('FXAxioma2017USD_SH')(modelDB, marketDB)

        # Manually reassign select assets to RMG and currency
        self.rmgOverride = dict()
        # Force all RDS issues to have GB exposure
        self.rmgOverride['CIJBKGWL8'] = [datetime.date(1980,1,1), datetime.date(2999,12,31), 'GB', 'GB']

    def setCalculators(self, modelDB, overrider=False):
        # Set important model parameters
        ModelParameters2017.defaultModelSettings(self, scm=False, statModel=True)
        # Set up returns model
        self.returnCalculator = FactorReturns.AsymptoticPrincipalComponents2017(self.numFactors, trimExtremeExposures=True)
        self.olsReturnClass = ModelParameters2017.simpleRegressionParameters(self, modelDB, overrider=overrider)
        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=1, horizon='short', overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=1, horizon='short', overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, nwLag=1, horizon='short', overrider=overrider)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

class DMAxioma2020FL(DMAxioma2020MH):
    """DM factor model
    """
    rm_id = 339
    revision = 1
    rms_id = 339

    DescriptorMap = {
            'Amihud Liquidity 60 Day': ['Amihud_Liquidity_60D'],
            'Amihud Liquidity 125 Day': ['Amihud_Liquidity_125D'],
            'Book to Price Annual': ['Book_to_Price_Annual'],
            'Cash Flow to Assets Annual': ['CashFlow_to_Assets_Annual'],
            'Cash Flow to Income Annual': ['CashFlow_to_Income_Annual'],
            'Debt to Assets Annual': ['Debt_to_Assets_Annual'],
            'Debt to Equity Annual': ['Debt_to_Equity_Annual'],
            'Dividend Yield Annual': ['Dividend_Yield_Annual'],
            'Earnings to Price Annual': ['Earnings_to_Price_Annual'],
            'Est Earnings to Price Annual': ['Est_Earnings_to_Price_12MFL_Annual'],
            'Earnings Growth Annual': ['Earnings_Growth_RPF_Annual'],
            'Gross Margin Annual': ['Gross_Margin_Annual'],
            'Returns to Trading Days': ['ISC_Ret_Score'],
            'Log of Issuer Cap': ['LnIssuerCap'],
            'Log of 20-Day ADV to Issuer Cap': ['LnTrading_Activity_20D'],
            'Log of 60-Day ADV to Issuer Cap': ['LnTrading_Activity_60D'],
            'Annual Return Excl Prev Month': ['Momentum_260x21D_Regional'],
            'Monthly Return': ['Momentum_21D'],
            'DMxUS Regional Market Sensitivity 250 Day': ['DMxUS_Regional_Market_Sensitivity_250D'],
            'DMxUS Regional Market Sensitivity 500 Day': ['DMxUS_Regional_Market_Sensitivity_500D'],
            'Return on Assets Annual': ['Return_on_Assets_Annual'],
            'Return on Equity Annual': ['Return_on_Equity_Annual'],
            'Sales to Assets Annual': ['Sales_to_Assets_Annual'],
            'Sales Growth Annual': ['Sales_Growth_RPF_Annual'],
            'Volatility 60 Day': ['DMxUS_Volatility_60D'],
            'Volatility 125 Day': ['DMxUS_Volatility_125D'],
            'Exchange Rate Sensitivity 52 Week XDR': ['XRate_52W_XDR'],
            'Exchange Rate Sensitivity 104 Week XDR': ['XRate_104W_XDR'],
            }

    styleList = sorted(DescriptorMap.keys())
    allowMissingFactors = ['Amihud Liquidity 60 Day', 'Amihud Liquidity 125 Day']
    DescriptorWeights = {}
    orthogList = {}
    exposureConfigFile = 'exposures-DMAxioma2020FL'

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.DMAxioma2020FL')
        DMAxioma2020MH.__init__(self, modelDB, marketDB)
        self.noClip_List = ['ISC_Ret_Score','Returns to Trading Days']

################################## Linked Models ########################################################################

class WWLMAxioma2020MH(EquityModel.LinkedModel):
    """Version 1 Global linked model
    """
    rm_id = 400
    revision = 1
    rms_id = 400
    gicsDate = datetime.date(2018,9,29)
    exposureConfigFile = 'exposures-mh'
    legacyLMFactorStructure = True
    industryRename = {
            'Telecommunication Services-S': 'Communication Services-S',
            'Media-G': 'Media & Entertainment-G',
            }
    industryDropChild = {'Consumer Discretionary-S': ['Media-G', 'Media & Entertainment-G'],
                              'Software & Services-G': ['Internet Software & Services', 'US4 Internet Software & Services']}
    industryAddChild = {'Media & Entertainment-G': ['Internet Software & Services', 'US4 Internet Software & Services']}
    industryAddRootParent = {}

    def __init__(self, modelDB, marketDB):
        # Initialise
        self.log = logging.getLogger('RiskModels.WWLMAxioma2020MH')
        ModelParameters2017.defaultModelSettings(self, scm=False)
        ModelParameters2017.defaultExposureParameters(self, [], configFile=self.exposureConfigFile)
        self.twoRegressionStructure = True

        origLoggingLevel = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(logging.ERROR)
        self.linkedModelMap = [
                (riskmodels.getModelByName('USAxioma2016MH')(modelDB, marketDB), 'US4', 'United States'),
                (riskmodels.getModelByName('DMAxioma2020MH')(modelDB, marketDB), 'DM4', None),
                (riskmodels.getModelByName('EMAxioma2018MH')(modelDB, marketDB), 'EM4', None)]
        logging.getLogger().setLevel(origLoggingLevel)

        EquityModel.LinkedModel.__init__(self, ['SEDOL', 'CUSIP'], modelDB, marketDB)
        self.currencyModel = riskmodels.getModelByName('FXAxioma2017USD_MH')(modelDB, marketDB)
        self.industryClassification = EquityModel.LinkedModelClassification(self, marketDB, modelDB)

        # Set up RiskModel Calculators
        self.setCalculators(modelDB)

    def setCalculators(self, modelDB, overrider=False):
        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=0, dateOverLap=5, overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=0, dateOverLap=5, overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, overrider=overrider)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

class WWLMAxioma2020SH(WWLMAxioma2020MH):
    """Version 1 Global linked model - short horizon
    """
    rm_id = 401
    revision = 1
    rms_id = 401
    gicsDate = datetime.date(2018,9,29)
    exposureConfigFile = 'exposures-sh'

    def __init__(self, modelDB, marketDB):
        # Initialise
        self.log = logging.getLogger('RiskModels.WWLMAxioma2020SH')
        ModelParameters2017.defaultModelSettings(self, scm=False)
        ModelParameters2017.defaultExposureParameters(self, [], configFile=self.exposureConfigFile)
        self.twoRegressionStructure = True

        origLoggingLevel = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(logging.ERROR)
        self.linkedModelMap = [
                (riskmodels.getModelByName('USAxioma2016SH')(modelDB, marketDB), 'US4', 'United States'),
                (riskmodels.getModelByName('DMAxioma2020SH')(modelDB, marketDB), 'DM4', None),
                (riskmodels.getModelByName('EMAxioma2018SH')(modelDB, marketDB), 'EM4', None)]
        logging.getLogger().setLevel(origLoggingLevel)

        EquityModel.LinkedModel.__init__(self, ['SEDOL', 'CUSIP'], modelDB, marketDB)
        self.currencyModel = riskmodels.getModelByName('FXAxioma2017USD_SH')(modelDB, marketDB)
        self.industryClassification = EquityModel.LinkedModelClassification(self, marketDB, modelDB)

        # Set up RiskModel Calculators
        self.setCalculators(modelDB)

    def setCalculators(self, modelDB, overrider=False):
        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=0, dateOverLap=5, horizon='short', overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=0, dateOverLap=5, horizon='short', overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, horizon='short', overrider=overrider)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

class NALMAxioma2021MH(EquityModel.LinkedModel):
    """NA4 linked model
    """
    rm_id = 410
    revision = 1
    rms_id = 410
    gicsDate = datetime.date(2018,9,29)
    exposureConfigFile = 'exposures-mh'
    industryRename = {
            'Telecommunication Services-S': 'Communication Services-S',
            'Media-G': 'Media & Entertainment-G',
            }

    ca_ind_list = ['Banks', 'Capital Goods', 'Commercial & Professional Services', 'Consumer Discretionary',
                   'Diversified Financials', 'Energy Equipment & Services', 'Equity Real Estate Investment Trusts (REITs)',
                   'Food & Staples Products', 'Food & Staples Retailing', 'Gold', 'Health Care',
                   'Insurance', 'Materials ex Metals, Mining & Forestry', 'Media & Entertainment',
                   'Metals & Mining ex Gold', 'Oil, Gas & Consumable Fuels', 'Paper & Forest Products',
                   'Real Estate Management & Development', 'Software & Services', 'Technology Hardware',
                   'Telecommunication Services', 'Transportation', 'Utilities']
    ca_ind_list = ca_ind_list + ['CA4 %s' % ind for ind in ca_ind_list]

    industryDropChild = {'Software & Services-G': ['Internet Software & Services', 'US4 Internet Software & Services'],
                         'Consumer Discretionary-S': ['Media-G', 'Media & Entertainment-G',
                                                      'Consumer Discretionary', 'CA4 Consumer Discretionary'],
                         'Financials-S': ['Banks', 'Diversified Financials', 'Insurance',
                                          'CA4 Banks', 'CA4 Diversified Financials', 'CA4 Insurance'],
                         'Health Care-S': ['Health Care', 'CA4 Health Care'],
                         'Industrials-S': ['Capital Goods', 'Commercial & Professional Services', 'Transportation',
                                            'CA4 Capital Goods', 'CA4 Commercial & Professional Services', 'CA4 Transportation'],
                         'Energy-S': ['Energy Equipment & Services', 'Oil, Gas & Consumable Fuels',
                                      'CA4 Energy Equipment & Services', 'CA4 Oil, Gas & Consumable Fuels'],
                         'Real Estate-S': ['Equity Real Estate Investment Trusts (REITs)', 'Real Estate Management & Development',
                                           'CA4 Equity Real Estate Investment Trusts (REITs)', 'CA4 Real Estate Management & Development'],
                         'Consumer Staples-S': ['Food & Staples Retailing', 'CA4 Food & Staples Retailing',
                                                'Food & Staples Products', 'CA4 Food & Staples Products'],
                         'Materials-S': ['Gold', 'Materials ex Metals, Mining & Forestry', 'Metals & Mining ex Gold', 'Paper & Forest Products',
                                         'CA4 Gold', 'CA4 Materials ex Metals, Mining & Forestry', 'CA4 Metals & Mining ex Gold', 'CA4 Paper & Forest Products'],
                         'Communication Services-S': ['Media & Entertainment', 'Telecommunication Services',
                                                      'CA4 Media & Entertainment', 'CA4 Telecommunication Services'],
                         'Information Technology-S': ['Software & Services', 'CA4 Software & Services',
                                                      'Technology Hardware', 'CA4 Technology Hardware'],
                         'Utilities-S': ['Utilities', 'CA4 Utilities'],
                         'Industry Groups': ca_ind_list,
                         }

    industryAddChild = {'Communication Services-S': ['Media-G', 'Media & Entertainment-G'],
                        'Media & Entertainment-G': ['Internet Software & Services', 'US4 Internet Software & Services',
                                                    'Media & Entertainment', 'CA4 Media & Entertainment'],
                        'Consumer Discretionary-S': ['CA4 Consumer Discretionary-G'],
                        'CA4 Consumer Discretionary-G': ['Consumer Discretionary', 'CA4 Consumer Discretionary'],
                        'Consumer Staples-S': ['CA4 Food & Staples Products-G'],
                        'CA4 Food & Staples Products-G': ['Food & Staples Products', 'CA4 Food & Staples Products'],
                        'Banks-G': ['Banks', 'CA4 Banks'],
                        'Capital Goods-G': ['Capital Goods', 'CA4 Capital Goods'],
                        'Commercial & Professional Services-G': ['Commercial & Professional Services', 'CA4 Commercial & Professional Services'],
                        'Diversified Financials-G': ['Diversified Financials', 'CA4 Diversified Financials'],
                        'Health Care-S': ['CA4 Health Care-G'],
                        'CA4 Health Care-G': ['Health Care', 'CA4 Health Care'],
                        'Real Estate-G': ['Equity Real Estate Investment Trusts (REITs)', 'Real Estate Management & Development',
                                          'CA4 Equity Real Estate Investment Trusts (REITs)', 'CA4 Real Estate Management & Development'],
                        'Energy-G': ['Energy Equipment & Services', 'Oil, Gas & Consumable Fuels',
                                     'CA4 Energy Equipment & Services', 'CA4 Oil, Gas & Consumable Fuels'],
                        'Food & Staples Retailing-G': ['Food & Staples Retailing', 'CA4 Food & Staples Retailing'],
                        'Insurance-G': ['Insurance', 'CA4 Insurance'],
                        'Materials-G': ['Gold', 'Materials ex Metals, Mining & Forestry', 'Metals & Mining ex Gold', 'Paper & Forest Products',
                                        'CA4 Gold', 'CA4 Materials ex Metals, Mining & Forestry', 'CA4 Metals & Mining ex Gold', 'CA4 Paper & Forest Products'],
                        'Software & Services-G': ['Software & Services', 'CA4 Software & Services'],
                        'Information Technology-S': ['CA4 Technology Hardware-G'],
                        'CA4 Technology Hardware-G': ['Technology Hardware', 'CA4 Technology Hardware'],
                        'Telecommunication Services-G': ['Telecommunication Services', 'CA4 Telecommunication Services'],
                        'Transportation-G': ['Transportation', 'CA4 Transportation'],
                        'Utilities-G': ['Utilities', 'CA4 Utilities'],
                        'Industries': ca_ind_list,
                        'Industry Groups': ['CA4 Consumer Discretionary-G', 'CA4 Food & Staples Products-G', 'CA4 Health Care-G', 'CA4 Technology Hardware-G']
                        }

    industryAddNode = [('CA4 Consumer Discretionary-G', 10001, 'CA4'),
                       ('CA4 Food & Staples Products-G', 10002, 'CA4'),
                       ('CA4 Health Care-G', 10003, 'CA4'),
                       ('CA4 Technology Hardware-G', 10004, 'CA4')]
    industryAddRootParent = {'Media-G': 'Industry Groups',
                             'CA4 Consumer Discretionary-G': 'Industry Groups',
                             'CA4 Food & Staples Products-G': 'Industry Groups',
                             'CA4 Health Care-G': 'Industry Groups',
                             'CA4 Technology Hardware-G': 'Industry Groups',
                             }

    def __init__(self, modelDB, marketDB):
        # Initialise
        self.log = logging.getLogger('RiskModels.NALMAxioma2021MH')
        ModelParameters2017.defaultModelSettings(self, scm=False)
        ModelParameters2017.defaultExposureParameters(self, [], configFile=self.exposureConfigFile)
        self.twoRegressionStructure = True

        origLoggingLevel = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(logging.ERROR)
        self.linkedModelMap = [
                (riskmodels.getModelByName('USAxioma2016MH')(modelDB, marketDB), 'US4', 'United States'),
                (riskmodels.getModelByName('CAAxioma2018MH')(modelDB, marketDB), 'CA4', 'Canada')]
        logging.getLogger().setLevel(origLoggingLevel)

        EquityModel.LinkedModel.__init__(self, ['SEDOL', 'CUSIP'], modelDB, marketDB)
        self.currencyModel = riskmodels.getModelByName('FXAxioma2017USD_MH')(modelDB, marketDB)
        self.industryClassification = EquityModel.LinkedModelClassification(self, marketDB, modelDB)

        # Set up RiskModel Calculators
        self.setCalculators(modelDB)

    def setCalculators(self, modelDB, overrider=False):
        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=0, dateOverLap=5, overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=0, dateOverLap=5, overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, overrider=overrider)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

class NALMAxioma2021SH(NALMAxioma2021MH):
    """NA4 linked model
    """
    rm_id = 411
    revision = 1
    rms_id = 411
    gicsDate = datetime.date(2018,9,29)
    exposureConfigFile = 'exposures-sh'

    def __init__(self, modelDB, marketDB):
        # Initialise
        self.log = logging.getLogger('RiskModels.NALMAxioma2021SH')
        ModelParameters2017.defaultModelSettings(self, scm=False)
        ModelParameters2017.defaultExposureParameters(self, [], configFile=self.exposureConfigFile)
        self.twoRegressionStructure = True

        origLoggingLevel = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(logging.ERROR)
        self.linkedModelMap = [
                (riskmodels.getModelByName('USAxioma2016SH')(modelDB, marketDB), 'US4', 'United States'),
                (riskmodels.getModelByName('CAAxioma2018SH')(modelDB, marketDB), 'CA4', 'Canada')]
        logging.getLogger().setLevel(origLoggingLevel)

        EquityModel.LinkedModel.__init__(self, ['SEDOL', 'CUSIP'], modelDB, marketDB)
        self.currencyModel = riskmodels.getModelByName('FXAxioma2017USD_SH')(modelDB, marketDB)
        self.industryClassification = EquityModel.LinkedModelClassification(self, marketDB, modelDB)

        # Set up RiskModel Calculators
        self.setCalculators(modelDB)

    def setCalculators(self, modelDB, overrider=False):
        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=0, dateOverLap=5, horizon='short', overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=0, dateOverLap=5, horizon='short', overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, horizon='short', overrider=overrider)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

class WWMPAxioma2020MH(EquityModel.ProjectionModel):
    """Version 1 Projection model - WW4
    """
    rm_id = 501
    revision = 1
    rms_id = 501

    # list of macro factors
    macro_list = ['Commodity', 
                 'Gold', 
                 'Oil', 
                 'USD BBB Corp Spread', 
                 'EUR BBB Corp Spread', 
                 'GBP BBB Corp Spread', 
                 'JPY BBB Corp Spread',
                 'US Inflation',
                 'GB Inflation',
                 'EU Inflation',
                 'US Term Spread: 10Y6M',
                 'EU Term Spread: 10Y6M',
                 'GB Term Spread: 10Y6M',
                 'JP Term Spread: 10Y6M',
                 'Carbon Emission Price'
                ]
    # static macros, contains all the possible macro factors
    all_macros = [ModelFactor(i,i) for i in macro_list]

    # dynamic, track the active macros by setFactorForDate method, being updated by setFactorForDate
    macros = [ModelFactor(i,i) for i in macro_list]

    gicsDate = datetime.date(2016,9,1)
    industryClassification = Classification.GICSIndustries(gicsDate)

    exposureConfigFile = 'exposures-mh'

    def __init__(self, modelDB, marketDB):
        # Initialise
        self.log = logging.getLogger('RiskModels.WWMPAxioma2020MH')
        ModelParameters2017.defaultModelSettings(self, scm=False) # set it to false
        ModelParameters2017.defaultExposureParameters(self, [], configFile=self.exposureConfigFile)

        self.twoRegressionStructure = True
        origLoggingLevel = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(logging.ERROR)
        self.baseModel = (riskmodels.getModelByName('WWAxioma2017MH')(modelDB, marketDB), 'WW4')
        logging.getLogger().setLevel(origLoggingLevel)
        
        EquityModel.ProjectionModel.__init__(self, ['SEDOL', 'CUSIP'], modelDB, marketDB)
        self.currencyModel = riskmodels.getModelByName('FXAxioma2017USD_MH')(modelDB, marketDB)

        # Set up RiskModel Calculators
        self.setCalculators(modelDB)

    def setCalculators(self, modelDB, overrider=False):
        # Set up risk parameters
        # Use weekly overlapping data
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=0, dateOverLap=5, overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=0, dateOverLap=5, overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, overrider=overrider)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

################################## End of model classes #################################################################
