
import datetime
import itertools
import logging
import numpy
import numpy as np
import numpy.ma as ma
import pandas
import pandas as pd
import statsmodels.api as sm

from riskmodels import Classification
from riskmodels import CurrencyRisk
from riskmodels import EstimationUniverse
from riskmodels import GlobalExposures
from riskmodels import Matrices
from riskmodels.Matrices import ExposureMatrix
from riskmodels import EquityModel
from riskmodels.Factors import ModelFactor
from riskmodels import AssetProcessor
from riskmodels import RiskCalculator
from riskmodels import RiskModels
from riskmodels import Standardization
from riskmodels import MarketIndex
from riskmodels import ModelParameters2017
from riskmodels import FactorReturns
from riskmodels import Outliers
#try:
#    import DC_YD as yd
#except:
#    import sys
#    sys.path.append('/home/ydai/cassandra/RMTtrunk/scripts/modules')
#    sys.path.append('/home/ydai/cassandra/RMTtrunk/scripts/RiskModels')
#    sys.path.append('/home/ydai/cassandra/RMTtrunk/scripts/DBTools')
#    import DC_YD as yd
######################################################################################################################################################
# fundamental model
######################################################################################################################################################

class CNAxioma2018MH_R1(EquityModel.FundamentalModel):
    """
        CN4 base model. - 2018 version
        CN4 base model. - change the ESTU with dummyThreshold 6,
        CN4 base model. - CN4MH_S
    """
    # Model Parameters:
    rm_id,revision,rms_id = [-24,1,-241]

    #market factor
    intercept = ModelFactor('Market Intercept', 'Market Intercept') # changed from Global Market to Market Intercept
    #style factor
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
    # Industry Structure
    # (1) consider GICS - first
    gicsDate = datetime.date(2016,9,1)
    industryClassification = Classification.GICSCustomCN2(gicsDate)

    ################################################################################################
    # estu related
    elig_parameters = {'HomeCountry_List': ['CN'],
                       'use_isin_country_Flag': False,
                       'assetTypes':['All-Com', 'REIT','AShares','BShares'], # assetTypes is updated at the init
                       'excludeTypes': None}
    # estu_parameters = {
    #                    'minNonZero':0.95,
    #                    'minNonMissing':0.95,
    #                    'maskZeroWithNoADV_Flag': True,
    #                    'returnLegacy_Flag': False,
    #                    'CapByNumber_Flag':False,
    #                    'CapByNumber_hiCapQuota':np.nan,
    #                    'CapByNumber_lowCapQuota':np.nan,
    #                    'market_lower_pctile': 5,
    #                    'country_lower_pctile': 5,
    #                    'industry_lower_pctile': 5,
    #                    'dummyThreshold': 0,
    #                    'inflation_cutoff':0.03
    #                     }
    estu_parameters = {
                       'minNonZero':0.5,
                       'minNonMissing':0.5,
                       'maskZeroWithNoADV_Flag': True,
                       'returnLegacy_Flag': False,
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

    ################################################################################################
   # exposure setting
   #  noProxyList = ['Dividend Yield']
   #  noStndDescriptor = []
   #  noCloneDescriptor = []
   #  for s in noProxyList:
   #      noStndDescriptor.extend(DescriptorMap[s])
   #      noCloneDescriptor.extend(DescriptorMap[s])
   #
   #  fillMissingList = ['Earnings Yield','Value', 'Leverage', 'Growth', 'Profitability']
   #  fillWithZeroList = ['Dividend Yield']
   #  shrinkList = {'Liquidity': 60,
   #                'Market Sensitivity': 250,
   #                'Volatility': 125,
   #                'Exchange Rate Sensitivity': 500,
   #                'Medium-Term Momentum': 250}
    DescriptorMap = {
        'Earnings Yield': ['Earnings_to_Price_Annual', 'Est_Earnings_to_Price_12MFL_Annual'],
        'Value': ['Book_to_Price_Annual'],
        'Leverage': ['Debt_to_Assets_Annual', 'Debt_to_Equity_Annual'],
        'Growth': ['Earnings_Growth_RPF_Annual', 'Sales_Growth_RPF_Annual'],
        'Dividend Yield': ['Dividend_Yield_Annual'],
        'Size': ['LnIssuerCap'],
        # 'Liquidity': ['LnTrading_Activity_60D','Amihud_Liquidity_125D', 'ISC_Ret_Score'],
        'Liquidity': ['LnTrading_Activity_60D','Amihud_Liquidity_Adj_125D', 'ISC_Ret_Score'],
        # 'Market Sensitivity': ['Market_Sensitivity_250D'],
        'Market Sensitivity': ['Market_Sensitivity_XC_104W'],
        # 'Volatility': ['Volatility_125D'],
        'Volatility': ['Volatility_CN_125D'],
        'Medium-Term Momentum': ['Momentum_250x20D'],
        'Exchange Rate Sensitivity': ['XRate_104W_XDR'],# XRate_104W_USD or XRate_104W_XDR
        'Profitability': ['Return_on_Equity_Annual','Return_on_Assets_Annual', 'CashFlow_to_Assets_Annual',
                          'CashFlow_to_Income_Annual','Sales_to_Assets_Annual', 'Gross_Margin_Annual'],
        }
    # more weigth for forecast EY
    DescriptorWeights = {'Earnings Yield': [0.75, 0.25]}
    orthogList = {'Volatility': [['Market Sensitivity'], True, 1.0],}
    exposureConfigFile = 'exposures-mh'

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.CNAxioma2018MH_R1')
        # from IPython import embed; embed(header='Debug: Model Start');import ipdb;ipdb.set_trace()
        # Set important model parameters
        # ModelParameters2017.defaultModelSettingsSCM(self,scm=True)
        ModelParameters2017.defaultModelSettings(self,scm=False) # single country with regional framework
        # model customize setting
        self.coverageMultiCountry = True
        self.hasCountryFactor = False
        self.hasCurrencyFactor = False
        self.applyRT2US = False

        self.Remove_China_AB_Flag = False
        self.Use_FreeFloat_RegWeight = True
        self.shrinkPara_useRegion = False # proxy based on all assets in the country # to follow up
        self.useLegacyISCScores = False

        self.regionalDescriptorStructure = True
        self.local_descriptor_numeraire = 'CNY'
        # to remove:
        # self.Cover_all_rmgs = True
        # self.usePrimaryRMG = True
        # self.expandSingleDescriptorStructure = True #using regionalDescriptorStructure, not expandSingleDescriptorStructure
        # update parameter using parent class value
        self.elig_parameters['assetTypes'] = self.elig_parameters['assetTypes'] + self.commonStockTypes
        # Set up relevant styles to be created/used
        ModelParameters2017.defaultExposureParameters(self, self.styleList, configFile=self.exposureConfigFile,descriptorNumeraire='USD') #USD for numeraire descriptors
        EquityModel.FundamentalModel.__init__(self, ['SEDOL'], modelDB, marketDB)
        # Set up setCalculators - regression parameters
        #internal factor return
        self.internalCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,dummyType='Sectors',
                dummyThreshold=self.estu_parameters['dummyThreshold'],
                marketRegression=False, kappa = 5.0,
                useRealMCaps=True, regWeight='rootCap')
        #external factor return
        self.returnCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,dummyType='Sectors',
                dummyThreshold=self.estu_parameters['dummyThreshold'],
                marketRegression=False, kappa = None,
                useRealMCaps=True, regWeight='rootCap')
        # This controls the FMP regression
        self.fmpCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,dummyType=None,
                dummyThreshold=self.estu_parameters['dummyThreshold'],
                marketRegression=False, kappa=None,
                useRealMCaps=True, regWeight='rootCap')
        # And this is for raw exposure proxying
        self.expProxyCalculator = ModelParameters2017.defaultExposureProxyParameters(
                self, modelDB,
                dummyType='market',
                dummyThreshold = self.estu_parameters['dummyThreshold'],
                kappa=5.0,
                useRealMCaps=True,
                regWeight='rootCap')
        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=3)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=3)
        ModelParameters2017.defaultSpecificVarianceParameters(self)
        self.covarianceCalculator = RiskCalculator.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

        # Set up standardization parameters
        gloScope = Standardization.GlobalRelativeScope([f.name for f in self.styles])
        self.exposureStandardization = Standardization.BucketizedStandardization([gloScope], fillWithZeroList=self.fillWithZeroList)
        self.indexSelector = MarketIndex.MarketIndexSelector(modelDB, marketDB)
        modelDB.createCurrencyCache(marketDB)
        # Set up descriptor standardization parameters
        descriptors = sorted(list(set([item for sublist in self.DescriptorMap.values() for item in sublist])))
        self.descriptorStandardization = Standardization.BucketizedStandardization(
                [Standardization.GlobalRelativeScope(descriptors)],
                mad_bound=15.0, fancyMAD=self.fancyMAD, exceptionNames=self.noStndDescriptor)

    def generate_model_specific_exposures(self, modelDate, data, modelDB, marketDB):
        """Generate the non-default factors.
        """
        # from IPython import embed; embed(header='Debug:generate_model_specific_exposures');import ipdb;ipdb.set_trace()
        beta = numpy.zeros((len(data.universe)), float)
        # Cap-based style factors here
        if not hasattr(self, 'estuMap') or self.estuMap is None:
            return data.exposureMatrix
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

    def clone_linked_asset_exposures(self, date, data, modelDB, marketDB, scoreDict):
        return self.clone_linked_asset_exposures_new(date, data, modelDB, marketDB, scoreDict,
                commonList=self.wideCloneList)
    def proxy_missing_exposures(self, modelDate, data, modelDB, marketDB,
            factorNames=['Value', 'Leverage', 'Growth'], clip=True, sizeVec=None, kappa=5.0):
        return self.proxy_missing_exposures_new(modelDate, data, modelDB, marketDB,
                factorNames=factorNames, clip=clip, sizeVec=sizeVec, kappa=kappa)

class CNAxioma2018MH_R2(CNAxioma2018MH_R1):
    """
        CN4 base model. - 2018 version 2 - merge EY and Value and Volatility without Orthogolizing to Market Sensitivity        
    """
    # Model Parameters:
    rm_id,revision,rms_id = [-24,2,-242]

    #market factor
    intercept = ModelFactor('Market Intercept', 'Market Intercept') # changed from Global Market to Market Intercept
    #style factor
    styleList = ['Value',
                 'Leverage',
                 'Growth',
                 'Profitability',
                 # 'Earnings Yield',
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
            # 'Market Sensitivity': ['Market_Sensitivity_104W'],
            'Volatility': ['Volatility_CN_125D'],
            'Medium-Term Momentum': ['Momentum_250x20D'],
            'Exchange Rate Sensitivity': ['XRate_104W_XDR'],# XRate_104W_USD or XRate_104W_XDR
            'Profitability': ['Return_on_Equity_Annual','Return_on_Assets_Annual', 'CashFlow_to_Assets_Annual',
                              'CashFlow_to_Income_Annual','Sales_to_Assets_Annual', 'Gross_Margin_Annual'],
            }
    # more weigth for forecast EY
    DescriptorWeights = {'Value': [0.5, 0.375, 0.125]}

    fillMissingList = ['Value', 'Leverage', 'Growth', 'Profitability']
    fillWithZeroList = ['Dividend Yield']
    shrinkList = {'Liquidity': 60,
                  'Market Sensitivity': 250,
                  'Volatility': 125,
                  'Exchange Rate Sensitivity': 500,
                  'Medium-Term Momentum': 250}
    orthogList = {}
    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.CNAxioma2018MH_R2')
        super(CNAxioma2018MH_R2,self).__init__(modelDB, marketDB)

class CNAxioma2018MH_R3(CNAxioma2018MH_R1):
    """
        CN4 base model. - 2018 version 3 - based on R2 add Offshore Factor         
    """
    # Model Parameters:
    rm_id,revision,rms_id = [-24,3,-244]

    #market factor
    intercept = ModelFactor('Market Intercept', 'Market Intercept') # changed from Global Market to Market Intercept
    localStructureFactors = [ModelFactor('OffShore China', 'OffShore China')]
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
            # 'Market Sensitivity': ['Market_Sensitivity_104W'],
            'Volatility': ['Volatility_CN_125D'],
            'Medium-Term Momentum': ['Momentum_250x20D'],
            'Exchange Rate Sensitivity': ['XRate_104W_XDR'],# XRate_104W_USD or XRate_104W_XDR
            'Profitability': ['Return_on_Equity_Annual','Return_on_Assets_Annual', 'CashFlow_to_Assets_Annual',
                              'CashFlow_to_Income_Annual','Sales_to_Assets_Annual', 'Gross_Margin_Annual'],
            }
    # more weigth for forecast EY
    DescriptorWeights = {'Value': [0.5, 0.375, 0.125]}

    fillMissingList = ['Value', 'Leverage', 'Growth', 'Profitability']
    fillWithZeroList = ['Dividend Yield']
    shrinkList = {'Liquidity': 60,
                  'Market Sensitivity': 250,
                  'Volatility': 125,
                  'Exchange Rate Sensitivity': 500,
                  'Medium-Term Momentum': 250}
    orthogList = {}
    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.CNAxioma2018MH_R3')
        super(CNAxioma2018MH_R3,self).__init__(modelDB, marketDB)
        #internal factor return
        self.internalCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,dummyType='Sectors',
                dummyThreshold=self.estu_parameters['dummyThreshold'],
                marketRegression=False,
                scndRegList=[[ExposureMatrix.LocalFactor]],
                scndRegEstus=['ChinaOff'],
                kappa = 5.0,
                useRealMCaps=True, regWeight='rootCap')
        #external factor return
        self.returnCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,dummyType='Sectors',
                dummyThreshold=self.estu_parameters['dummyThreshold'],
                marketRegression=False,
                scndRegList=[[ExposureMatrix.LocalFactor]],
                scndRegEstus=['ChinaOff'],
                kappa = None,
                useRealMCaps=True, regWeight='rootCap')
        # This controls the FMP regression
        self.fmpCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,dummyType=None,
                dummyThreshold=self.estu_parameters['dummyThreshold'],
                marketRegression=False,
                scndRegList=[[ExposureMatrix.LocalFactor]],
                scndRegEstus=['ChinaOff'],
                kappa=None,
                useRealMCaps=True, regWeight='rootCap')
        self.estu_parameters_ChinaOff = {
                                       'minGoodReturns':0.85,
                                       'cap_lower_pctile':5
                                        }

    def generate_model_specific_exposures(self, modelDate, data, modelDB, marketDB):
        """Generate OffShore China local factor.
        """
        logging.info('Building OffShore China Exposures')

        buildEstu = EstimationUniverse.ConstructEstimationUniverse(
                data.exposureMatrix.getAssets(), self, modelDB, marketDB)
        exTypes = ['AShares', 'BShares']
        (OffShoreShares, nonest) = buildEstu.exclude_by_asset_type(
                modelDate, data, includeFields=None, excludeFields=exTypes)

        values = Matrices.allMasked(len(data.universe))
        if len(OffShoreShares) > 0:
            logging.info('Assigning OffShore China exposure to %d assets', len(OffShoreShares))
            ma.put(values, OffShoreShares, 1.0)
        data.exposureMatrix.addFactor('OffShore China', values, ExposureMatrix.LocalFactor)

        return data.exposureMatrix

class CNAxioma2018MH_S_R1(EquityModel.StatisticalModel):
    """
        EM research statistical model - base model
        
    """
    rm_id = -25
    revision = 1
    rms_id = -243
    numFactors = 15
    blind = [ModelFactor('Statistical Factor %d' % n, 'Statistical Factor %d' % n) for n in range(1, numFactors+1)]
    pcaHistory = 250
    gicsDate = datetime.date(2016,9,1)
    industryClassification = Classification.GICSCustomCN2(gicsDate)
    descriptorNumeraire = 'USD'

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.CNAxioma2018MH_S_R1')
        # Set important model parameters
        ModelParameters2017.defaultModelSettings(self, scm=False, statModel=True)
        EquityModel.StatisticalModel.__init__(self, ['SEDOL', 'CUSIP'], modelDB, marketDB)
        # So we can use the same ESTU method as the fundamental model
        self.baseModelDateMap = {datetime.date(1980,1,1): CNAxioma2018MH_R1(modelDB, marketDB)}
        # Set up estimation universe parameters
        self.elig_parameters = list(self.baseModelDateMap.values())[0].elig_parameters.copy()
        # model customize setting
        self.coverageMultiCountry = True
        self.hasCountryFactor = False
        self.hasCurrencyFactor = False
        self.applyRT2US = False

        self.Remove_China_AB_Flag = False
        self.Use_FreeFloat_RegWeight = True
        self.shrinkPara_useRegion = False # proxy based on all assets in the country
        self.useLegacyISCScores = False

        self.regionalDescriptorStructure = True
        self.local_descriptor_numeraire = 'CNY'
        # Set Calculators
        self.setCalculators(modelDB)
        # Manually reassign select assets to RMG and currency
        self.tweakDict = dict()

    def setCalculators(self, modelDB, overrider=False):
        # Set up returns model
        self.returnCalculator = FactorReturns.AsymptoticPrincipalComponents2017(self.numFactors, trimExtremeExposures=True)
        self.olsReturnClass = ModelParameters2017.simpleRegressionParameters(self, modelDB, overrider=overrider)
        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=1, overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=1, overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, nwLag=1, overrider=overrider)
        self.covarianceCalculator = RiskCalculator.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

    def clone_linked_asset_exposures(self, date, data, modelDB, marketDB, scoreDict):
        return self.clone_linked_asset_exposures_new(date, data, modelDB, marketDB, scoreDict)
