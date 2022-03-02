
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
from riskmodels import EquityModel
from riskmodels.Factors import ModelFactor
from riskmodels import RiskCalculator
from riskmodels import RiskCalculator_V4
from riskmodels import RiskModels
from riskmodels import Standardization
from riskmodels import MarketIndex
from riskmodels import ModelParameters2017
from riskmodels import FactorReturns

class WWResearchModelEM1(RiskModels.WWAxioma2017MH):
    """Global research model with
    """
    rm_id = -6
    revision = 6
    rms_id = -301

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.WWResearchModelEM1')
        RiskModels.WWAxioma2017MH.__init__(self, modelDB, marketDB)

class USResearchModelEM1(RiskModels.USAxioma2016MH):
    """Global research model with
    """
    rm_id = -400
    revision = 1
    rms_id = -400

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.USResearchModelEM1')
        RiskModels.USAxioma2016MH.__init__(self, modelDB, marketDB)


class WWResearchModel1(EquityModel.FundamentalModel):
    """Global research model
    """
    rm_id = -6
    revision = 1
    rms_id = -11

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
                 #'Short-Term Momentum',
                 'Exchange Rate Sensitivity',
                 #'Emerging Market Sensitivity'
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
            #'Liquidity': ['LnTrading_Activity_60D'],
            'Market Sensitivity': ['Regional_Market_Sensitivity_500D'],
            #'Emerging Market Sensitivity': ['EMxWW_Market_Sensitivity_500D'],
            'Volatility': ['Volatility_125D'],
            'Medium-Term Momentum': ['Momentum_250x20D'],
            'Exchange Rate Sensitivity': ['XRate_104W_XDR'],
            'Interest Rate Sensitivity': ['XRate_104W_IR'],
            'Profitability': ['Return_on_Equity_Annual', 'Return_on_Assets_Annual',
                'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
                'Sales_to_Assets_Annual', 'Gross_Margin_Annual'],
            #Payout': ['Net_Equity_Issuance', 'Net_Debt_Issuance', 'Net_Payout_Over_Profits'],
            }

    DescriptorWeights = {'Earnings Yield': [0.75, 0.25],}
                         #'Payout': [-1.0, -1.0, 1.0]}
    noProxyList = ['Dividend Yield']
    fillMissingList = ['Earnings Yield', 'Value', 'Leverage', 'Growth', 'Profitability',]# 'Payout']
    fillWithZeroList = ['Dividend Yield']
    shrinkList = {'Liquidity': 60,
                  'Market Sensitivity': 250,
                  'Volatility': 125,
                  'Medium-Term Momentum': 250,
                  'Exchange Rate Sensitivity': 500,
                  #'Emerging Market Sensitivity': 500,
                  'Interest Rate Sensitivity': 500}
    orthogList = {'Volatility': [['Market Sensitivity'], True, 1.0],}
                  #'Emerging Market Sensitivity': [['Market Sensitivity'], True, 1.0]}
    regionalStndList = ['Earnings Yield', 'Value', 'Leverage', 'Growth', 'Profitability', 'Dividend Yield',]# 'Payout']
    _regionalMapper = lambda dm, l: [dm[st] for st in l]
    regionalStndDesc = list(itertools.chain.from_iterable(_regionalMapper(DescriptorMap, regionalStndList)))
    del _regionalMapper

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.WWResearchModel1')
        ModelParameters2017.defaultModelSettings(self, scm=False)
        # Set up relevant styles to be created/used
        ModelParameters2017.defaultExposureParameters(self, self.styleList)
        self.styles = [s for s in self.totalStyles if s.name in self.styleList]
        EquityModel.FundamentalModel.__init__(self, ['SEDOL', 'CUSIP'], modelDB, marketDB)
        self.currencyModel = riskmodels.getModelByName('FXAxioma2017USD_MH')(modelDB, marketDB)

        # Set up estimation universe parameters
        self.estu_parameters, self.elig_parameters = ModelParameters2017.defaultRegionalModelEstuParameters(self)

        # Set up internal factor return regression parameters
        dummyThreshold = 10
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
                )

        # And this is for raw exposure proxying
        self.expProxyCalculator = ModelParameters2017.defaultExposureProxyParameters(
                self, modelDB,
                dummyType='market',
                dummyThreshold = dummyThreshold,
                kappa=5.0,
                useRealMCaps=True,
                regWeight='rootCap',
                )

        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=3, )
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=3, )
        ModelParameters2017.defaultSpecificVarianceParameters(self, )
        self.covarianceCalculator = RiskCalculator.CompositeCovarianceMatrix2017(\
                self.fvParameters, self.fcParameters)

        # Set up descriptor standardization parameters
        descriptors = sorted(list(set([item for sublist
            in list(self.DescriptorMap.values()) for item in sublist])))
        regScope = Standardization.RegionRelativeScope(
                modelDB, self.regionalStndDesc)
        gloScope = Standardization.GlobalRelativeScope(
                [d for d in descriptors if d not in self.regionalStndDesc])
        exceptionNames = [self.DescriptorMap[sf] for sf in self.noProxyList]
        exceptionNames = list(itertools.chain.from_iterable(exceptionNames))
        self.descriptorStandardization = Standardization.BucketizedStandardization(
                [regScope, gloScope], mad_bound=15.0, fancyMAD=self.fancyMAD,
                exceptionNames=exceptionNames)

        # Set up standardization parameters
        regScope = Standardization.RegionRelativeScope(
                modelDB, self.regionalStndList)
        gloScope = Standardization.GlobalRelativeScope(
                [f.name for f in self.styles if f.name not in self.regionalStndList])
        self.exposureStandardization = Standardization.BucketizedStandardization(
                [regScope, gloScope], fillWithZeroList=self.fillWithZeroList)

        # Manually reassign select assets to RMG and currency
        self.tweakDict = dict()
        self.tweakDict['CIJBKGWL8'] = [datetime.date(1980,1,1), datetime.date(2999,12,31), 'GB', 'GB']

    def generate_model_specific_exposures(self, modelDate, data, modelDB, marketDB):
        """Generate Domestic China local factor.
        """
        logging.info('Building Domestic China Exposures')

        buildEstu = EstimationUniverse.ConstructEstimationUniverse(
                data.exposureMatrix.getAssets(), self, modelDB, marketDB)
        exTypes = ['AShares', 'BShares']
        (aShares, nonest) = buildEstu.exclude_by_asset_type(
                modelDate, data, includeFields=exTypes, excludeFields=None)

        values = Matrices.allMasked(len(data.universe))
        if len(aShares) > 0:
            logging.info('Assigning Domestic China exposure to %d assets', len(aShares))
            ma.put(values, aShares, 1.0)
        data.exposureMatrix.addFactor('Domestic China', values, ExposureMatrix.LocalFactor)

        return data.exposureMatrix

    def clone_linked_asset_exposures(self, date, data, modelDB, marketDB, scoreDict):
        return self.clone_linked_asset_exposures_new(date, data, modelDB, marketDB, scoreDict,
                commonList=self.fillMissingList + ['Size'])

    def proxy_missing_exposures(self, modelDate, data, modelDB, marketDB,
            factorNames=['Value', 'Leverage', 'Growth'], clip=True, sizeVec=None, kappa=5.0):
        return self.proxy_missing_exposures_new(modelDate, data, modelDB, marketDB,
                factorNames=factorNames, clip=clip, sizeVec=sizeVec, kappa=kappa)

class WWResearchModel2(RiskModels.WWAxioma2017MH):
    """Global research model
    """
    rm_id = -6
    revision = 2
    rms_id = -12

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.WWResearchModel4')
        RiskModels.WWAxioma2017MH.__init__(self, modelDB, marketDB)
        self.estu_parameters, self.elig_parameters = ModelParameters2017.defaultRegionalModelEstuParameters(self)

    def generate_estimation_universe(self, modelDate, data, modelDB, marketDB,
            excludeFactors=[], grandfatherRMS_ID=None):
        return self.generate_estimation_universe_internal(modelDate, data, modelDB, marketDB,
                excludeFactors=excludeFactors, grandfatherRMS_ID=grandfatherRMS_ID)

class WWResearchModel3(EquityModel.StatisticalModel):
    """Global research model
    """
    rm_id = -6
    revision = 3
    rms_id = -13
    numFactors = 20
    blind = [ModelFactor('Statistical Factor %d' % n, 'Statistical Factor %d' % n)
                         for n in range(1, numFactors+1)]
    pcaHistory = 250
    gicsDate = datetime.date(2016,9,1)
    industryClassification = Classification.GICSIndustries(gicsDate)

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.WWResearchModel3')
        ModelParameters2017.defaultModelSettings(self, scm=False, statModel=True)
        EquityModel.StatisticalModel.__init__(self, ['SEDOL', 'CUSIP'], modelDB, marketDB)
        # So we can use the same ESTU method as the fundamental model
        self.baseModelDateMap = {datetime.date(1980,1,1): WWResearchModel1(modelDB, marketDB)}
        # Set up returns model
        self.returnCalculator = FactorReturns.AsymptoticPrincipalComponents2017(self.numFactors)
        self.olsReturnClass = ModelParameters2017.simpleRegressionParameters(self, modelDB)
        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=1)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=1)
        ModelParameters2017.defaultSpecificVarianceParameters(self, nwLag=1)
        self.covarianceCalculator = RiskCalculator.CompositeCovarianceMatrix2017(\
                self.fvParameters, self.fcParameters)
        # Define currency model
        self.currencyModel = riskmodels.getModelByName('FXAxioma2017USD_MH')(modelDB, marketDB)
        # Force all RDS issues to have GB exposure
        self.tweakDict['CIJBKGWL8'] = [datetime.date(1980,1,1), datetime.date(2999,12,31), 'GB', 'GB']

    def clone_linked_asset_exposures(self, date, data, modelDB, marketDB, scoreDict):
        return self.clone_linked_asset_exposures_new(date, data, modelDB, marketDB, scoreDict)

class WWResearchModel4(EquityModel.StatisticalModel):
    """Global research model
    """
    rm_id = -6
    revision = 4
    rms_id = -14

    numFactors = 55
    numGlobalFactors = 20
    regionFactorMap = {
            'North America': 5,
            'Latin America': 5,
            'Europe': 5,
            'Asia ex-Pacific': 5,
            'Pacific': 5,
            'Middle East': 5,
            'Africa': 5}
    blind = [ModelFactor('Statistical Factor %d' % n, 'Statistical Factor %d' % n) for n in range(1, numFactors+1)]

    pcaHistory = 250
    gicsDate = datetime.date(2016,9,1)
    industryClassification = Classification.GICSIndustries(gicsDate)
    descriptorNumeraire = 'USD'

    def __init__(self, modelDB, marketDB):
        # Set things up
        self.log = logging.getLogger('RiskModels.WWResearchModel4')
        ModelParameters2017.defaultModelSettings(self, scm=False, statModel=True)
        EquityModel.StatisticalModel.__init__(self, ['SEDOL', 'CUSIP'], modelDB, marketDB)

        # Use the same ESTU method as the fundamental model
        self.baseModelDateMap = {datetime.date(1980,1,1): RiskModels.WWxUSAxioma2017MH(modelDB, marketDB)}

        # Set up estimation universe parameters
        self.elig_parameters = ModelParameters2017.defaultRegionalModelEstuParameters(self)[1]

        # Set Calculators
        self.setCalculators(modelDB)

        # Define currency model
        self.currencyModel = riskmodels.getModelByName('FXAxioma2017USD_MH')(modelDB, marketDB)

        # Manually reassign select assets to RMG and currency
        self.tweakDict = dict()
        # Force all RDS issues to have GB exposure
        self.tweakDict['CIJBKGWL8'] = [datetime.date(1980,1,1), datetime.date(2999,12,31), 'GB', 'GB']

    def setCalculators(self, modelDB, overrider=False):
        # Set up returns model
        self.returnCalculator = FactorReturns.AsymptoticPrincipalComponents2017(self.numFactors, trimExtremeExposures=True)
        self.olsReturnClass = ModelParameters2017.simpleRegressionParameters(self, modelDB, overrider=overrider)
        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=1, overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=1, overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, nwLag=1, overrider=overrider)
        self.covarianceCalculator = RiskCalculator.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

class EU4_GICS_IndGroups(RiskModels.EUAxioma2017MH):
    """Version 4 Europe medium-horizon fundamental model with GICS Industry Groups
    """
    rm_id = -7
    revision = 1
    rms_id = -21

    gicsDate = datetime.date(2016,9,1)
    industryClassification = Classification.GICSIndustryGroups(gicsDate)

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('EU4_GICS_IndGroups')
        RiskModels.EUAxioma2017MH.__init__(self, modelDB, marketDB)
        self.currencyModel = riskmodels.getModelByName('FXAxioma2017EUR_MH')(modelDB, marketDB)

    def setCalculators(self, modelDB, overrider=False):
        RiskModels.EUAxioma2017MH.setCalculators(self, modelDB, overrider=overrider)
        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=3, overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=3, overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, overrider=overrider)
        self.covarianceCalculator = RiskCalculator.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

class EU4_GICS_Industry(RiskModels.EUAxioma2017MH):
    """Version 4 Europe medium-horizon fundamental model with GICS Industry Groups
    """
    rm_id = -7
    revision = 2
    rms_id = -22

    gicsDate = datetime.date(2016,9,1)
    industryClassification = Classification.GICSCustomNoMortgageREITs(gicsDate)

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('EU4_GICS_Industries')
        RiskModels.EUAxioma2017MH.__init__(self, modelDB, marketDB)
