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
from riskmodels import Standardization_V4
from riskmodels import MarketIndex
from riskmodels import ModelParameters2017
from riskmodels import FactorReturns

# Penultimate set
class CAResearchModelEM2(EquityModel.FundamentalModel):
    """CA4 cookie cutter
    """
    rm_id = -13
    revision = 2
    rms_id = -72

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

    interceptFactor = 'Market Intercept'
    intercept = ModelFactor(interceptFactor, interceptFactor)
    
    smallCapMap = {'MidCap': [50., 95.],}
    
    gicsDate = datetime.date(2018,9,29)
    industryClassification = Classification.GICSCustomCA2(
            gicsDate)
    
    DescriptorMap = {
            'Earnings Yield': ['Earnings_to_Price_Quarterly', 'Est_Earnings_to_Price_12MFL_Quarterly'],
            'Value': ['Book_to_Price_Quarterly'],
            'Leverage': ['Debt_to_Assets_Quarterly', 'Debt_to_Equity_Quarterly'],
            'Growth': ['Earnings_Growth_RPF_AFQ', 'Sales_Growth_RPF_AFQ'],
            'Dividend Yield': ['Dividend_Yield_Quarterly'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_60D', 'Amihud_Liquidity_125D', 'ISC_Ret_Score'],
            'Market Sensitivity': ['Market_Sensitivity_104W'],
            'Volatility': ['Volatility_125D'],
            'Medium-Term Momentum': ['Momentum_250x20D'],
            'Exchange Rate Sensitivity': ['XRate_104W_XDR'],
            'Profitability': ['Return_on_Equity_Quarterly', 'Return_on_Assets_Quarterly',
                              'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
                              'Sales_to_Assets_Quarterly', 'Gross_Margin_Quarterly'],
            }

    DescriptorWeights = {'Earnings Yield': [0.75, 0.25]}
   
    orthogList = {'Volatility': [['Market Sensitivity'], True, 1.0]}
    
    exposureConfigFile = 'exposures-CAResearchModelEM2'

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
        #self.fundAssetTypes = ['CEFund', 'InvT', 'Misc', 'UnitT'] # default
        self.fundAssetTypes = ['CEFund', 'InvT', 'Misc'] # overwrite default to keep UnitT
        self.elig_parameters = {'assetTypes': self.commonStockTypes + ['REIT'] + ['UnitT'],
                                'excludeTypes': None,
                                'use_isin_country_Flag': False,
                                'remove_China_AB': True,
                                'addBack_H_DR': False,
                                }  

        self.estu_parameters = {'minNonZero':0.1,
                                'minNonMissing':0.5,
                                #'ADV_percentile': [5, 100], # HOW IS THIS USED??
                                'maskZeroWithNoADV_Flag': True, 
                                'CapByNumber_Flag': True,
                                'CapByNumber_hiCapQuota': 250,
                                'CapByNumber_lowCapQuota': 150,
                                'market_lower_pctile': np.nan,
                                'country_lower_pctile': np.nan,
                                'industry_lower_pctile': np.nan,
                                'dummyThreshold': 6,
                                #'inflation_cutoff':0.03
                                'inflation_cutoff':0.01 # CAAxioma2009MH setting - change to .03?
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


        # Manually reassign select assets to RMG and currency
        self.tweakDict = dict()

    def setCalculators(self, modelDB, overrider=False):
        # Set up regression parameters
        #dummyThreshold = 10
        dummyThreshold = 6  # CAAxioma2009MH setting - change to 10?
        
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
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=1, overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=1, overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, overrider=overrider)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

    def generate_model_specific_exposures(self, modelDate, data, modelDB, marketDB):
        """Generate the non-default factors.
        """
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

class CAResearchModelEM1(EquityModel.FundamentalModel):
    """CA1 kappa 5 
       CA1 factors (except STM) plus market factor
       Includes internal and external factor returns
       GICS 2018
       Uses V4 Estu
    """
    rm_id = -13
    revision = 1
    rms_id = -71

    # List of style factors in the model
    styleList = [
                 'Value',
                 'Leverage',
                 'Growth',
                 'Size',
                 'Liquidity',
                 'Market Sensitivity',
                 'Volatility',
                 'Medium-Term Momentum',
                ]

    interceptFactor = 'Market Intercept'
    intercept = ModelFactor(interceptFactor, interceptFactor)
    
    #smallCapMap = {'MidCap': [50., 95.],}
    
    gicsDate = datetime.date(2018,9,29)
    industryClassification = Classification.GICSCustomCA2(
            gicsDate)
    
    DescriptorMap = {
            'Value': ['Book_to_Price_Quarterly'],
            'Leverage': ['Debt_to_Assets_Quarterly', 'Debt_to_Equity_Quarterly'],
            'Growth': ['Return_on_Equity_Quarterly'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_60D', 'Amihud_Liquidity_125D', 'ISC_Ret_Score'],
            'Market Sensitivity': ['Market_Sensitivity_104W'],
            'Volatility': ['Volatility_125D'],
            'Medium-Term Momentum': ['Momentum_250x20D'],
            }

    DescriptorWeights = {}
   
    orthogList = {'Volatility': [['Market Sensitivity'], True, 1.0]}
    
    exposureConfigFile = 'exposures-CAResearchModelEM1'

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
        #self.fundAssetTypes = ['CEFund', 'InvT', 'Misc', 'UnitT'] # default
        self.fundAssetTypes = ['CEFund', 'InvT', 'Misc'] # overwrite default to keep UnitT
        self.elig_parameters = {'assetTypes': self.commonStockTypes + ['REIT'] + ['UnitT'],
                                'excludeTypes': None,
                                'use_isin_country_Flag': False,
                                'remove_China_AB': True,
                                'addBack_H_DR': False,
                                }  

        self.estu_parameters = {'minNonZero':0.1,
                    'minNonMissing':0.5,
                    #'ADV_percentile': [5, 100], # HOW IS THIS USED??
                    'maskZeroWithNoADV_Flag': True, 
                    'CapByNumber_Flag': True,
                    'CapByNumber_hiCapQuota': 250,
                    'CapByNumber_lowCapQuota': 150,
                    'market_lower_pctile': np.nan,
                    'country_lower_pctile': np.nan,
                    'industry_lower_pctile': np.nan,
                    'dummyThreshold': 6,
                    #'inflation_cutoff':0.03
                    'inflation_cutoff':0.01 # CAAxioma2009MH setting - change to .03?
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


        # Manually reassign select assets to RMG and currency
        self.tweakDict = dict()

    def setCalculators(self, modelDB, overrider=False):
        # Set up regression parameters
        #dummyThreshold = 10
        dummyThreshold = 6  # CAAxioma2009MH setting - change to 10?
        
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
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=1, overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=1, overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, overrider=overrider)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

    def generate_model_specific_exposures(self, modelDate, data, modelDB, marketDB):
        """Generate the non-default factors.
        """
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

class CAResearchModelEM3(EquityModel.FundamentalModel):
    """CA4 cookie cutter
       Net of mkt gold/oil sensitivities
    """
    rm_id = -73
    revision = 1
    rms_id = -73

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
                 'Gold Sensitivity',
                 'Oil Sensitivity',
                ]

    interceptFactor = 'Market Intercept'
    intercept = ModelFactor(interceptFactor, interceptFactor)
    
    smallCapMap = {'MidCap': [50., 95.],}
    
    gicsDate = datetime.date(2018,9,29)
    industryClassification = Classification.GICSCustomCA2(
            gicsDate)
    
    DescriptorMap = {
            'Earnings Yield': ['Earnings_to_Price_Quarterly', 'Est_Earnings_to_Price_12MFL_Quarterly'],
            'Value': ['Book_to_Price_Quarterly'],
            'Leverage': ['Debt_to_Assets_Quarterly', 'Debt_to_Equity_Quarterly'],
            'Growth': ['Earnings_Growth_RPF_AFQ', 'Sales_Growth_RPF_AFQ'],
            'Dividend Yield': ['Dividend_Yield_Quarterly'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_60D', 'Amihud_Liquidity_125D', 'ISC_Ret_Score'],
            'Market Sensitivity': ['Market_Sensitivity_104W'],
            'Volatility': ['Volatility_125D'],
            'Medium-Term Momentum': ['Momentum_250x20D'],
            'Exchange Rate Sensitivity': ['XRate_104W_XDR'],
            'Profitability': ['Return_on_Equity_Quarterly', 'Return_on_Assets_Quarterly',
                              'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
                              'Sales_to_Assets_Quarterly', 'Gross_Margin_Quarterly'],
            'Gold Sensitivity': ['Gold_Sensitivity_104W'],
            'Oil Sensitivity': ['Oil_Sensitivity_104W'],
            }

    DescriptorWeights = {'Earnings Yield': [0.75, 0.25]}
   
    orthogList = {'Volatility': [['Market Sensitivity'], True, 1.0]}
    
    exposureConfigFile = 'exposures-CAResearchModelEM3'

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
        #self.fundAssetTypes = ['CEFund', 'InvT', 'Misc', 'UnitT'] # default
        self.fundAssetTypes = ['CEFund', 'InvT', 'Misc'] # overwrite default to keep UnitT
        self.elig_parameters = {'assetTypes': self.commonStockTypes + ['REIT'] + ['UnitT'],
                                'excludeTypes': None,
                                'use_isin_country_Flag': False,
                                'remove_China_AB': True,
                                'addBack_H_DR': False,
                                }  

        self.estu_parameters = {'minNonZero':0.1,
                    'minNonMissing':0.5,
                    #'ADV_percentile': [5, 100], # HOW IS THIS USED??
                    'maskZeroWithNoADV_Flag': True, 
                    'CapByNumber_Flag': True,
                    'CapByNumber_hiCapQuota': 250,
                    'CapByNumber_lowCapQuota': 150,
                    'market_lower_pctile': np.nan,
                    'country_lower_pctile': np.nan,
                    'industry_lower_pctile': np.nan,
                    'dummyThreshold': 6,
                    #'inflation_cutoff':0.03
                    'inflation_cutoff':0.01 # CAAxioma2009MH setting - change to .03?
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


        # Manually reassign select assets to RMG and currency
        self.tweakDict = dict()

    def setCalculators(self, modelDB, overrider=False):
        # Set up regression parameters
        #dummyThreshold = 10
        dummyThreshold = 6  # CAAxioma2009MH setting - change to 10?
        
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
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=1, overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=1, overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, overrider=overrider)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

    def generate_model_specific_exposures(self, modelDate, data, modelDB, marketDB):
        """Generate the non-default factors.
        """
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

class CAResearchModelEM4(EquityModel.FundamentalModel):
    """CA4 cookie cutter
       Net of sector gold/oil sensitivities
    """
    rm_id = -74
    revision = 1
    rms_id = -74

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
                 'Gold Sensitivity',
                 'Oil Sensitivity',
                ]

    interceptFactor = 'Market Intercept'
    intercept = ModelFactor(interceptFactor, interceptFactor)
    
    smallCapMap = {'MidCap': [50., 95.],}
    
    gicsDate = datetime.date(2018,9,29)
    industryClassification = Classification.GICSCustomCA2(
            gicsDate)
    
    DescriptorMap = {
            'Earnings Yield': ['Earnings_to_Price_Quarterly', 'Est_Earnings_to_Price_12MFL_Quarterly'],
            'Value': ['Book_to_Price_Quarterly'],
            'Leverage': ['Debt_to_Assets_Quarterly', 'Debt_to_Equity_Quarterly'],
            'Growth': ['Earnings_Growth_RPF_AFQ', 'Sales_Growth_RPF_AFQ'],
            'Dividend Yield': ['Dividend_Yield_Quarterly'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_60D', 'Amihud_Liquidity_125D', 'ISC_Ret_Score'],
            'Market Sensitivity': ['Market_Sensitivity_104W'],
            'Volatility': ['Volatility_125D'],
            'Medium-Term Momentum': ['Momentum_250x20D'],
            'Exchange Rate Sensitivity': ['XRate_104W_XDR'],
            'Profitability': ['Return_on_Equity_Quarterly', 'Return_on_Assets_Quarterly',
                              'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
                              'Sales_to_Assets_Quarterly', 'Gross_Margin_Quarterly'],
            'Gold Sensitivity': ['CAGold_Sensitivity_NetOfSec_104W'],
            'Oil Sensitivity': ['CAOil_Sensitivity_NetOfSec_104W'],
            }

    DescriptorWeights = {'Earnings Yield': [0.75, 0.25]}
   
    orthogList = {'Volatility': [['Market Sensitivity'], True, 1.0]}
    
    exposureConfigFile = 'exposures-CAResearchModelEM4'

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
        #self.fundAssetTypes = ['CEFund', 'InvT', 'Misc', 'UnitT'] # default
        self.fundAssetTypes = ['CEFund', 'InvT', 'Misc'] # overwrite default to keep UnitT
        self.elig_parameters = {'assetTypes': self.commonStockTypes + ['REIT'] + ['UnitT'],
                                'excludeTypes': None,
                                'use_isin_country_Flag': False,
                                'remove_China_AB': True,
                                'addBack_H_DR': False,
                                }  

        self.estu_parameters = {'minNonZero':0.1,
                    'minNonMissing':0.5,
                    #'ADV_percentile': [5, 100], # HOW IS THIS USED??
                    'maskZeroWithNoADV_Flag': True, 
                    'CapByNumber_Flag': True,
                    'CapByNumber_hiCapQuota': 250,
                    'CapByNumber_lowCapQuota': 150,
                    'market_lower_pctile': np.nan,
                    'country_lower_pctile': np.nan,
                    'industry_lower_pctile': np.nan,
                    'dummyThreshold': 6,
                    #'inflation_cutoff':0.03
                    'inflation_cutoff':0.01 # CAAxioma2009MH setting - change to .03?
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


        # Manually reassign select assets to RMG and currency
        self.tweakDict = dict()

    def setCalculators(self, modelDB, overrider=False):
        # Set up regression parameters
        #dummyThreshold = 10
        dummyThreshold = 6  # CAAxioma2009MH setting - change to 10?
        
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
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=1, overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=1, overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, overrider=overrider)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

    def generate_model_specific_exposures(self, modelDate, data, modelDB, marketDB):
        """Generate the non-default factors.
        """
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

class CAResearchModelEM5(EquityModel.FundamentalModel):
    """Barra-esque model
    """
    rm_id = -75
    revision = 1
    rms_id = -75

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
                 'Gold Sensitivity',
                 'Oil Sensitivity',
                ]

    interceptFactor = 'Market Intercept'
    intercept = ModelFactor(interceptFactor, interceptFactor)
    
    smallCapMap = {'MidCap': [50., 95.],}
    
    gicsDate = datetime.date(2018,9,29)
    industryClassification = Classification.GICSCustomCA3(
            gicsDate)
    
    DescriptorMap = {
            'Earnings Yield': ['Earnings_to_Price_Quarterly', 'Est_Earnings_to_Price_12MFL_Quarterly'],
            'Value': ['Book_to_Price_Quarterly'],
            'Leverage': ['Debt_to_Assets_Quarterly', 'Debt_to_Equity_Quarterly'],
            'Growth': ['Earnings_Growth_RPF_AFQ', 'Sales_Growth_RPF_AFQ'],
            'Dividend Yield': ['Dividend_Yield_Quarterly'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_60D', 'Amihud_Liquidity_125D', 'ISC_Ret_Score'],
            'Market Sensitivity': ['Market_Sensitivity_104W'],
            'Volatility': ['Volatility_125D'],
            'Medium-Term Momentum': ['Momentum_250x20D'],
            'Exchange Rate Sensitivity': ['XRate_104W_XDR'],
            'Profitability': ['Return_on_Equity_Quarterly', 'Return_on_Assets_Quarterly',
                              'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
                              'Sales_to_Assets_Quarterly', 'Gross_Margin_Quarterly'],
            'Gold Sensitivity': ['Gold_Sensitivity_NoMkt_104W'],
            'Oil Sensitivity': ['Oil_Sensitivity_NoMkt_104W'],
            }

    DescriptorWeights = {'Earnings Yield': [0.75, 0.25]}
   
    orthogList = {'Volatility': [['Market Sensitivity'], True, 1.0]}
    
    exposureConfigFile = 'exposures-CAResearchModelEM4'

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
        #self.fundAssetTypes = ['CEFund', 'InvT', 'Misc', 'UnitT'] # default
        self.fundAssetTypes = ['CEFund', 'InvT', 'Misc'] # overwrite default to keep UnitT
        self.elig_parameters = {'assetTypes': self.commonStockTypes + ['REIT'] + ['UnitT'],
                                'excludeTypes': None,
                                'use_isin_country_Flag': False,
                                'remove_China_AB': True,
                                'addBack_H_DR': False,
                                }  

        self.estu_parameters = {'minNonZero':0.1,
                    'minNonMissing':0.5,
                    #'ADV_percentile': [5, 100], # HOW IS THIS USED??
                    'maskZeroWithNoADV_Flag': True, 
                    'CapByNumber_Flag': True,
                    'CapByNumber_hiCapQuota': 250,
                    'CapByNumber_lowCapQuota': 150,
                    'market_lower_pctile': np.nan,
                    'country_lower_pctile': np.nan,
                    'industry_lower_pctile': np.nan,
                    'dummyThreshold': 6,
                    #'inflation_cutoff':0.03
                    'inflation_cutoff':0.01 # CAAxioma2009MH setting - change to .03?
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


        # Manually reassign select assets to RMG and currency
        self.tweakDict = dict()

    def setCalculators(self, modelDB, overrider=False):
        # Set up regression parameters
        #dummyThreshold = 10
        dummyThreshold = 6  # CAAxioma2009MH setting - change to 10?
        
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
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=1, overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=1, overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, overrider=overrider)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

    def generate_model_specific_exposures(self, modelDate, data, modelDB, marketDB):
        """Generate the non-default factors.
        """
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

class CAResearchModelEM6(EquityModel.FundamentalModel):
    """Model with full oil/gold sensitivities with CS orthog
    """
    rm_id = -76
    revision = 1
    rms_id = -76

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
                 'Gold Sensitivity',
                 'Oil Sensitivity',
                ]

    interceptFactor = 'Market Intercept'
    intercept = ModelFactor(interceptFactor, interceptFactor)
    
    smallCapMap = {'MidCap': [50., 95.],}
    
    gicsDate = datetime.date(2018,9,29)
    industryClassification = Classification.GICSCustomCA2(
            gicsDate)
    
    DescriptorMap = {
            'Earnings Yield': ['Earnings_to_Price_Quarterly', 'Est_Earnings_to_Price_12MFL_Quarterly'],
            'Value': ['Book_to_Price_Quarterly'],
            'Leverage': ['Debt_to_Assets_Quarterly', 'Debt_to_Equity_Quarterly'],
            'Growth': ['Earnings_Growth_RPF_AFQ', 'Sales_Growth_RPF_AFQ'],
            'Dividend Yield': ['Dividend_Yield_Quarterly'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_60D', 'Amihud_Liquidity_125D', 'ISC_Ret_Score'],
            'Market Sensitivity': ['Market_Sensitivity_104W'],
            'Volatility': ['Volatility_125D'],
            'Medium-Term Momentum': ['Momentum_250x20D'],
            'Exchange Rate Sensitivity': ['XRate_104W_XDR'],
            'Profitability': ['Return_on_Equity_Quarterly', 'Return_on_Assets_Quarterly',
                              'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
                              'Sales_to_Assets_Quarterly', 'Gross_Margin_Quarterly'],
            'Gold Sensitivity': ['Gold_Sensitivity_NoMkt_104W'],
            'Oil Sensitivity': ['Oil_Sensitivity_NoMkt_104W'],
            }

    DescriptorWeights = {'Earnings Yield': [0.75, 0.25]}
   
    orthogList = {
            'Volatility': [['Market Sensitivity'], True, 1.0],
            'Gold Sensitivity': [['Gold'], True, 1.0],
            'Oil Sensitivity': [['Energy'], True, 1.0]
            }
    
    exposureConfigFile = 'exposures-CAResearchModelEM6'

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
        #self.fundAssetTypes = ['CEFund', 'InvT', 'Misc', 'UnitT'] # default
        self.fundAssetTypes = ['CEFund', 'InvT', 'Misc'] # overwrite default to keep UnitT
        self.elig_parameters = {'assetTypes': self.commonStockTypes + ['REIT'] + ['UnitT'],
                                'excludeTypes': None,
                                'use_isin_country_Flag': False,
                                'remove_China_AB': True,
                                'addBack_H_DR': False,
                                }  

        self.estu_parameters = {'minNonZero':0.1,
                    'minNonMissing':0.5,
                    #'ADV_percentile': [5, 100], # HOW IS THIS USED??
                    'maskZeroWithNoADV_Flag': True, 
                    'CapByNumber_Flag': True,
                    'CapByNumber_hiCapQuota': 250,
                    'CapByNumber_lowCapQuota': 150,
                    'market_lower_pctile': np.nan,
                    'country_lower_pctile': np.nan,
                    'industry_lower_pctile': np.nan,
                    'dummyThreshold': 6,
                    #'inflation_cutoff':0.03
                    'inflation_cutoff':0.01 # CAAxioma2009MH setting - change to .03?
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


        # Manually reassign select assets to RMG and currency
        self.tweakDict = dict()

    def setCalculators(self, modelDB, overrider=False):
        # Set up regression parameters
        #dummyThreshold = 10
        dummyThreshold = 6  # CAAxioma2009MH setting - change to 10?
        
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
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=1, overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=1, overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, overrider=overrider)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

    def generate_model_specific_exposures(self, modelDate, data, modelDB, marketDB):
        """Generate the non-default factors.
        """
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


# Final set
class CAResearchModelEM7(EquityModel.FundamentalModel):
    """CA4 cookie cutter with factor returns generated from March 1995 
       (baseline model)
    """
    rm_id = -77
    revision = 1
    rms_id = -77

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

    interceptFactor = 'Market Intercept'
    intercept = ModelFactor(interceptFactor, interceptFactor)
   
    smallCapMap = {}
    
    gicsDate = datetime.date(2018,9,29)
    industryClassification = Classification.GICSCustomCA2(
            gicsDate)
    
    DescriptorMap = {
            'Earnings Yield': ['Earnings_to_Price_Quarterly', 'Est_Earnings_to_Price_12MFL_Quarterly'],
            'Value': ['Book_to_Price_Quarterly'],
            'Leverage': ['Debt_to_Assets_Quarterly', 'Debt_to_Equity_Quarterly'],
            'Growth': ['Earnings_Growth_RPF_AFQ', 'Sales_Growth_RPF_AFQ'],
            'Dividend Yield': ['Dividend_Yield_Quarterly'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_60D', 'Amihud_Liquidity_125D', 'ISC_Ret_Score'],
            'Market Sensitivity': ['Market_Sensitivity_104W'],
            'Volatility': ['Volatility_125D'],
            'Medium-Term Momentum': ['Momentum_260x21D_Regional'],
            'Exchange Rate Sensitivity': ['XRate_104W_XDR'],
            'Profitability': ['Return_on_Equity_Quarterly', 'Return_on_Assets_Quarterly',
                              'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
                              'Sales_to_Assets_Quarterly', 'Gross_Margin_Quarterly'],
            }

    DescriptorWeights = {'Earnings Yield': [0.75, 0.25]}
   
    orthogList = {'Volatility': [['Market Sensitivity'], True, 1.0]}
    
    exposureConfigFile = 'exposures-CAResearchModelEM7'

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
        #self.fundAssetTypes = ['CEFund', 'InvT', 'Misc', 'UnitT'] # default
        self.fundAssetTypes = ['CEFund', 'InvT', 'Misc'] # overwrite default to keep UnitT
        self.elig_parameters = {'assetTypes': self.commonStockTypes + ['REIT'] + ['UnitT'],
                                'excludeTypes': None,
                                'use_isin_country_Flag': False,
                                'remove_China_AB': True,
                                'addBack_H_DR': False,
                                }  

        self.estu_parameters = {'minNonZero':0.1,
                    'minNonMissing':0.5,
                    #'ADV_percentile': [5, 100], # HOW IS THIS USED??
                    'maskZeroWithNoADV_Flag': True, 
                    'CapByNumber_Flag': True,
                    'CapByNumber_hiCapQuota': 250,
                    'CapByNumber_lowCapQuota': 150,
                    'market_lower_pctile': np.nan,
                    'country_lower_pctile': np.nan,
                    'industry_lower_pctile': np.nan,
                    'dummyThreshold': 6,
                    #'inflation_cutoff':0.03
                    'inflation_cutoff':0.01 # CAAxioma2009MH setting - change to .03?
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


        # Manually reassign select assets to RMG and currency
        self.tweakDict = dict()

    def setCalculators(self, modelDB, overrider=False):
        # Set up regression parameters
        #dummyThreshold = 10
        dummyThreshold = 6  # CAAxioma2009MH setting - change to 10?
        
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

class CAResearchModelEM8(EquityModel.FundamentalModel):
    """CA4 cookie cutter with net of sec oil/gold sensitivities
    """
    rm_id = -78
    revision = 1
    rms_id = -78

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
                 'Gold Sensitivity',
                 'Oil Sensitivity',
                ]

    interceptFactor = 'Market Intercept'
    intercept = ModelFactor(interceptFactor, interceptFactor)
   
    smallCapMap = {}
    
    gicsDate = datetime.date(2018,9,29)
    industryClassification = Classification.GICSCustomCA2(
            gicsDate)
    
    DescriptorMap = {
            'Earnings Yield': ['Earnings_to_Price_Quarterly', 'Est_Earnings_to_Price_12MFL_Quarterly'],
            'Value': ['Book_to_Price_Quarterly'],
            'Leverage': ['Debt_to_Assets_Quarterly', 'Debt_to_Equity_Quarterly'],
            'Growth': ['Earnings_Growth_RPF_AFQ', 'Sales_Growth_RPF_AFQ'],
            'Dividend Yield': ['Dividend_Yield_Quarterly'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_60D', 'Amihud_Liquidity_125D', 'ISC_Ret_Score'],
            'Market Sensitivity': ['Market_Sensitivity_104W'],
            'Volatility': ['CA_Volatility_125D'],
            'Medium-Term Momentum': ['Momentum_260x21D_Regional'],
            'Exchange Rate Sensitivity': ['XRate_104W_XDR'],
            'Profitability': ['Return_on_Equity_Quarterly', 'Return_on_Assets_Quarterly',
                              'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
                              'Sales_to_Assets_Quarterly', 'Gross_Margin_Quarterly'],
            'Gold Sensitivity': ['CAGold_Sensitivity_NetOfSec_104W'],
            'Oil Sensitivity': ['CAOil_Sensitivity_NetOfSec_104W'],
            }

    DescriptorWeights = {'Earnings Yield': [0.75, 0.25]}
   
    orthogList = {'Volatility': [['Market Sensitivity'], True, 1.0]}
    
    exposureConfigFile = 'exposures-CAResearchModelEM8'

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
        #self.fundAssetTypes = ['CEFund', 'InvT', 'Misc', 'UnitT'] # default
        self.fundAssetTypes = ['CEFund', 'InvT', 'Misc'] # overwrite default to keep UnitT
        self.elig_parameters = {'assetTypes': self.commonStockTypes + ['REIT'] + ['UnitT'],
                                'excludeTypes': None,
                                'use_isin_country_Flag': False,
                                'remove_China_AB': True,
                                'addBack_H_DR': False,
                                }  

        self.estu_parameters = {'minNonZero':0.1,
                    'minNonMissing':0.5,
                    #'ADV_percentile': [5, 100], # HOW IS THIS USED??
                    'maskZeroWithNoADV_Flag': True, 
                    'CapByNumber_Flag': True,
                    'CapByNumber_hiCapQuota': 250,
                    'CapByNumber_lowCapQuota': 150,
                    'market_lower_pctile': np.nan,
                    'country_lower_pctile': np.nan,
                    'industry_lower_pctile': np.nan,
                    'dummyThreshold': 6,
                    #'inflation_cutoff':0.03
                    'inflation_cutoff':0.01 # CAAxioma2009MH setting - change to .03?
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


        # Manually reassign select assets to RMG and currency
        self.tweakDict = dict()

    def setCalculators(self, modelDB, overrider=False):
        # Set up regression parameters
        #dummyThreshold = 10
        dummyThreshold = 6  # CAAxioma2009MH setting - change to 10?
        
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

class CAResearchModelEM9(EquityModel.FundamentalModel):
    """CA4 cookie cutter with factor returns generated from Jan 1996 
       (baseline model)
    """
    rm_id = -79
    revision = 1
    rms_id = -79

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

    interceptFactor = 'Market Intercept'
    intercept = ModelFactor(interceptFactor, interceptFactor)
   
    smallCapMap = {}
    
    gicsDate = datetime.date(2018,9,29)
    industryClassification = Classification.GICSCustomCA2(
            gicsDate)
    
    DescriptorMap = {
            'Earnings Yield': ['Earnings_to_Price_Quarterly', 'Est_Earnings_to_Price_12MFL_Quarterly'],
            'Value': ['Book_to_Price_Quarterly'],
            'Leverage': ['Debt_to_Assets_Quarterly', 'Debt_to_Equity_Quarterly'],
            'Growth': ['Earnings_Growth_RPF_AFQ', 'Sales_Growth_RPF_AFQ'],
            'Dividend Yield': ['Dividend_Yield_Quarterly'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_60D', 'Amihud_Liquidity_125D', 'ISC_Ret_Score'],
            'Market Sensitivity': ['Market_Sensitivity_104W'],
            'Volatility': ['Volatility_125D'],
            'Medium-Term Momentum': ['Momentum_260x21D_Regional'],
            'Exchange Rate Sensitivity': ['XRate_104W_XDR'],
            'Profitability': ['Return_on_Equity_Quarterly', 'Return_on_Assets_Quarterly',
                              'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
                              'Sales_to_Assets_Quarterly', 'Gross_Margin_Quarterly'],
            }

    DescriptorWeights = {'Earnings Yield': [0.75, 0.25]}
   
    orthogList = {'Volatility': [['Market Sensitivity'], True, 1.0]}
    
    exposureConfigFile = 'exposures-CAResearchModelEM7'

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
        #self.fundAssetTypes = ['CEFund', 'InvT', 'Misc', 'UnitT'] # default
        self.fundAssetTypes = ['CEFund', 'InvT', 'Misc'] # overwrite default to keep UnitT
        self.elig_parameters = {'assetTypes': self.commonStockTypes + ['REIT'] + ['UnitT'],
                                'excludeTypes': None,
                                'use_isin_country_Flag': False,
                                'remove_China_AB': True,
                                'addBack_H_DR': False,
                                }  

        self.estu_parameters = {'minNonZero':0.1,
                    'minNonMissing':0.5,
                    #'ADV_percentile': [5, 100], # HOW IS THIS USED??
                    'maskZeroWithNoADV_Flag': True, 
                    'CapByNumber_Flag': True,
                    'CapByNumber_hiCapQuota': 250,
                    'CapByNumber_lowCapQuota': 150,
                    'market_lower_pctile': np.nan,
                    'country_lower_pctile': np.nan,
                    'industry_lower_pctile': np.nan,
                    'dummyThreshold': 6,
                    #'inflation_cutoff':0.03
                    'inflation_cutoff':0.01 # CAAxioma2009MH setting - change to .03?
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


        # Manually reassign select assets to RMG and currency
        self.tweakDict = dict()

    def setCalculators(self, modelDB, overrider=False):
        # Set up regression parameters
        #dummyThreshold = 10
        dummyThreshold = 6  # CAAxioma2009MH setting - change to 10?
        
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

class CAResearchModelEM10(EquityModel.FundamentalModel):
    """CA4 cookie cutter with factor returns generated from Jan 1996 
       (EY and BTP combined in one Value factor)
    """
    rm_id = -80
    revision = 1
    rms_id = -80

    # List of style factors in the model
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

    interceptFactor = 'Market Intercept'
    intercept = ModelFactor(interceptFactor, interceptFactor)
   
    smallCapMap = {}
    
    gicsDate = datetime.date(2018,9,29)
    industryClassification = Classification.GICSCustomCA2(
            gicsDate)
    
    DescriptorMap = {
            'Value': ['Book_to_Price_Quarterly', 'Earnings_to_Price_Quarterly', 'Est_Earnings_to_Price_12MFL_Quarterly'],
            'Leverage': ['Debt_to_Assets_Quarterly', 'Debt_to_Equity_Quarterly'],
            'Growth': ['Earnings_Growth_RPF_AFQ', 'Sales_Growth_RPF_AFQ'],
            'Dividend Yield': ['Dividend_Yield_Quarterly'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_60D', 'Amihud_Liquidity_125D', 'ISC_Ret_Score'],
            'Market Sensitivity': ['Market_Sensitivity_104W'],
            'Volatility': ['Volatility_125D'],
            'Medium-Term Momentum': ['Momentum_260x21D_Regional'],
            'Exchange Rate Sensitivity': ['XRate_104W_XDR'],
            'Profitability': ['Return_on_Equity_Quarterly', 'Return_on_Assets_Quarterly',
                              'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
                              'Sales_to_Assets_Quarterly', 'Gross_Margin_Quarterly'],
            }

    DescriptorWeights = {'Value': [0.5, 0.375, 0.125]}
   
    orthogList = {'Volatility': [['Market Sensitivity'], True, 1.0]}
    
    exposureConfigFile = 'exposures-CAResearchModelEM10'

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
        #self.fundAssetTypes = ['CEFund', 'InvT', 'Misc', 'UnitT'] # default
        self.fundAssetTypes = ['CEFund', 'InvT', 'Misc'] # overwrite default to keep UnitT
        self.elig_parameters = {'assetTypes': self.commonStockTypes + ['REIT'] + ['UnitT'],
                                'excludeTypes': None,
                                'use_isin_country_Flag': False,
                                'remove_China_AB': True,
                                'addBack_H_DR': False,
                                }  

        self.estu_parameters = {'minNonZero':0.1,
                    'minNonMissing':0.5,
                    #'ADV_percentile': [5, 100], # HOW IS THIS USED??
                    'maskZeroWithNoADV_Flag': True, 
                    'CapByNumber_Flag': True,
                    'CapByNumber_hiCapQuota': 250,
                    'CapByNumber_lowCapQuota': 150,
                    'market_lower_pctile': np.nan,
                    'country_lower_pctile': np.nan,
                    'industry_lower_pctile': np.nan,
                    'dummyThreshold': 6,
                    #'inflation_cutoff':0.03
                    'inflation_cutoff':0.01 # CAAxioma2009MH setting - change to .03?
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


        # Manually reassign select assets to RMG and currency
        self.tweakDict = dict()

    def setCalculators(self, modelDB, overrider=False):
        # Set up regression parameters
        #dummyThreshold = 10
        dummyThreshold = 6  # CAAxioma2009MH setting - change to 10?
        
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


class CAResearchModelEM13(EquityModel.FundamentalModel):
    """CA4 cookie cutter with factor returns generated from Jan 1996
       Dense industry scheme
    """
    rm_id = -83
    revision = 1
    rms_id = -83

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

    interceptFactor = 'Market Intercept'
    intercept = ModelFactor(interceptFactor, interceptFactor)
   
    smallCapMap = {}
    
    gicsDate = datetime.date(2018,9,29)
    industryClassification = Classification.GICSCustomCA4(
            gicsDate)
    
    DescriptorMap = {
            'Earnings Yield': ['Earnings_to_Price_Quarterly', 'Est_Earnings_to_Price_12MFL_Quarterly'],
            'Value': ['Book_to_Price_Quarterly'],
            'Leverage': ['Debt_to_Assets_Quarterly', 'Debt_to_Equity_Quarterly'],
            'Growth': ['Earnings_Growth_RPF_AFQ', 'Sales_Growth_RPF_AFQ'],
            'Dividend Yield': ['Dividend_Yield_Quarterly'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_60D', 'Amihud_Liquidity_125D', 'ISC_Ret_Score'],
            'Market Sensitivity': ['Market_Sensitivity_104W'],
            'Volatility': ['Volatility_125D'],
            'Medium-Term Momentum': ['Momentum_260x21D_Regional'],
            'Exchange Rate Sensitivity': ['XRate_104W_XDR'],
            'Profitability': ['Return_on_Equity_Quarterly', 'Return_on_Assets_Quarterly',
                              'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
                              'Sales_to_Assets_Quarterly', 'Gross_Margin_Quarterly'],
            }

    DescriptorWeights = {'Earnings Yield': [0.75, 0.25]}
   
    orthogList = {'Volatility': [['Market Sensitivity'], True, 1.0]}
    
    exposureConfigFile = 'exposures-CAResearchModelEM7'

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
        #self.fundAssetTypes = ['CEFund', 'InvT', 'Misc', 'UnitT'] # default
        self.fundAssetTypes = ['CEFund', 'InvT', 'Misc'] # overwrite default to keep UnitT
        self.elig_parameters = {'assetTypes': self.commonStockTypes + ['REIT'] + ['UnitT'],
                                'excludeTypes': None,
                                'use_isin_country_Flag': False,
                                'remove_China_AB': True,
                                'addBack_H_DR': False,
                                }  

        self.estu_parameters = {'minNonZero':0.1,
                    'minNonMissing':0.5,
                    #'ADV_percentile': [5, 100], # HOW IS THIS USED??
                    'maskZeroWithNoADV_Flag': True, 
                    'CapByNumber_Flag': True,
                    'CapByNumber_hiCapQuota': 250,
                    'CapByNumber_lowCapQuota': 150,
                    'market_lower_pctile': np.nan,
                    'country_lower_pctile': np.nan,
                    'industry_lower_pctile': np.nan,
                    'dummyThreshold': 6,
                    #'inflation_cutoff':0.03
                    'inflation_cutoff':0.01 # CAAxioma2009MH setting - change to .03?
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


        # Manually reassign select assets to RMG and currency
        self.tweakDict = dict()

    def setCalculators(self, modelDB, overrider=False):
        # Set up regression parameters
        #dummyThreshold = 10
        dummyThreshold = 6  # CAAxioma2009MH setting - change to 10?
        
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


class CAResearchModelEM14(EquityModel.FundamentalModel):
    """CA4 cookie cutter with net of sec oil/gold sensitivities
    """
    rm_id = -84
    revision = 1
    rms_id = -84

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
                 'Gold Sensitivity',
                 'Oil Sensitivity',
                ]

    interceptFactor = 'Market Intercept'
    intercept = ModelFactor(interceptFactor, interceptFactor)
   
    smallCapMap = {}
    
    gicsDate = datetime.date(2018,9,29)
    industryClassification = Classification.GICSCustomCA2(
            gicsDate)
    
    DescriptorMap = {
            'Earnings Yield': ['Earnings_to_Price_Quarterly', 'Est_Earnings_to_Price_12MFL_Quarterly'],
            'Value': ['Book_to_Price_Quarterly'],
            'Leverage': ['Debt_to_Assets_Quarterly', 'Debt_to_Equity_Quarterly'],
            'Growth': ['Earnings_Growth_RPF_AFQ', 'Sales_Growth_RPF_AFQ'],
            'Dividend Yield': ['Dividend_Yield_Quarterly'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_60D', 'Amihud_Liquidity_125D', 'ISC_Ret_Score'],
            'Market Sensitivity': ['Market_Sensitivity_104W'],
            'Volatility': ['CA_Volatility_125D'],
            'Medium-Term Momentum': ['Momentum_260x21D_Regional'],
            'Exchange Rate Sensitivity': ['XRate_104W_XDR'],
            'Profitability': ['Return_on_Equity_Quarterly', 'Return_on_Assets_Quarterly',
                              'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
                              'Sales_to_Assets_Quarterly', 'Gross_Margin_Quarterly'],
            'Gold Sensitivity': ['CAGold_Sensitivity_NetOfSecBeta_104W'],
            'Oil Sensitivity': ['CAOil_Sensitivity_NetOfSecBeta_104W'],
            }

    DescriptorWeights = {'Earnings Yield': [0.75, 0.25]}
   
    orthogList = {'Volatility': [['Market Sensitivity'], True, 1.0]}
    
    exposureConfigFile = 'exposures-CAResearchModelEM8'

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
        #self.fundAssetTypes = ['CEFund', 'InvT', 'Misc', 'UnitT'] # default
        self.fundAssetTypes = ['CEFund', 'InvT', 'Misc'] # overwrite default to keep UnitT
        self.elig_parameters = {'assetTypes': self.commonStockTypes + ['REIT'] + ['UnitT'],
                                'excludeTypes': None,
                                'use_isin_country_Flag': False,
                                'remove_China_AB': True,
                                'addBack_H_DR': False,
                                }  

        self.estu_parameters = {'minNonZero':0.1,
                    'minNonMissing':0.5,
                    #'ADV_percentile': [5, 100], # HOW IS THIS USED??
                    'maskZeroWithNoADV_Flag': True, 
                    'CapByNumber_Flag': True,
                    'CapByNumber_hiCapQuota': 250,
                    'CapByNumber_lowCapQuota': 150,
                    'market_lower_pctile': np.nan,
                    'country_lower_pctile': np.nan,
                    'industry_lower_pctile': np.nan,
                    'dummyThreshold': 6,
                    #'inflation_cutoff':0.03
                    'inflation_cutoff':0.01 # CAAxioma2009MH setting - change to .03?
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


        # Manually reassign select assets to RMG and currency
        self.tweakDict = dict()

    def setCalculators(self, modelDB, overrider=False):
        # Set up regression parameters
        #dummyThreshold = 10
        dummyThreshold = 6  # CAAxioma2009MH setting - change to 10?
        
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

class CAResearchModelEM15(EquityModel.FundamentalModel):
    """CA4 cookie cutter with net of sec oil/gold sensitivities
    """
    rm_id = -85
    revision = 1
    rms_id = -85

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
                 'Gold Sensitivity',
                 'Oil Sensitivity',
                ]

    interceptFactor = 'Market Intercept'
    intercept = ModelFactor(interceptFactor, interceptFactor)
   
    smallCapMap = {}
    
    gicsDate = datetime.date(2018,9,29)
    industryClassification = Classification.GICSCustomCA2(
            gicsDate)
    
    DescriptorMap = {
            'Earnings Yield': ['Earnings_to_Price_Quarterly', 'Est_Earnings_to_Price_12MFL_Quarterly'],
            'Value': ['Book_to_Price_Quarterly'],
            'Leverage': ['Debt_to_Assets_Quarterly', 'Debt_to_Equity_Quarterly'],
            'Growth': ['Earnings_Growth_RPF_AFQ', 'Sales_Growth_RPF_AFQ'],
            'Dividend Yield': ['Dividend_Yield_Quarterly'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_60D', 'Amihud_Liquidity_125D', 'ISC_Ret_Score'],
            'Market Sensitivity': ['Market_Sensitivity_104W'],
            'Volatility': ['CA_Volatility_125D'],
            'Medium-Term Momentum': ['Momentum_260x21D_Regional'],
            'Exchange Rate Sensitivity': ['XRate_104W_XDR'],
            'Profitability': ['Return_on_Equity_Quarterly', 'Return_on_Assets_Quarterly',
                              'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
                              'Sales_to_Assets_Quarterly', 'Gross_Margin_Quarterly'],
            'Gold Sensitivity': ['CAGold_Sensitivity_NetOfSecUnit_104W'],
            'Oil Sensitivity': ['CAOil_Sensitivity_NetOfSecUnit_104W'],
            }

    DescriptorWeights = {'Earnings Yield': [0.75, 0.25]}
   
    orthogList = {'Volatility': [['Market Sensitivity'], True, 1.0]}
    
    exposureConfigFile = 'exposures-CAResearchModelEM8'

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
        #self.fundAssetTypes = ['CEFund', 'InvT', 'Misc', 'UnitT'] # default
        self.fundAssetTypes = ['CEFund', 'InvT', 'Misc'] # overwrite default to keep UnitT
        self.elig_parameters = {'assetTypes': self.commonStockTypes + ['REIT'] + ['UnitT'],
                                'excludeTypes': None,
                                'use_isin_country_Flag': False,
                                'remove_China_AB': True,
                                'addBack_H_DR': False,
                                }  

        self.estu_parameters = {'minNonZero':0.1,
                    'minNonMissing':0.5,
                    #'ADV_percentile': [5, 100], # HOW IS THIS USED??
                    'maskZeroWithNoADV_Flag': True, 
                    'CapByNumber_Flag': True,
                    'CapByNumber_hiCapQuota': 250,
                    'CapByNumber_lowCapQuota': 150,
                    'market_lower_pctile': np.nan,
                    'country_lower_pctile': np.nan,
                    'industry_lower_pctile': np.nan,
                    'dummyThreshold': 6,
                    #'inflation_cutoff':0.03
                    'inflation_cutoff':0.01 # CAAxioma2009MH setting - change to .03?
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


        # Manually reassign select assets to RMG and currency
        self.tweakDict = dict()

    def setCalculators(self, modelDB, overrider=False):
        # Set up regression parameters
        #dummyThreshold = 10
        dummyThreshold = 6  # CAAxioma2009MH setting - change to 10?
        
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


class CAResearchModelEM16(EquityModel.FundamentalModel):
    """CA4 cookie cutter with factor returns generated from Jan 1996 
       (Value factor has BTP ONLY!)
    """
    rm_id = -86
    revision = 1
    rms_id = -86

    # List of style factors in the model
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

    interceptFactor = 'Market Intercept'
    intercept = ModelFactor(interceptFactor, interceptFactor)
   
    smallCapMap = {}
    
    gicsDate = datetime.date(2018,9,29)
    industryClassification = Classification.GICSCustomCA2(
            gicsDate)
    
    DescriptorMap = {
            'Value': ['Book_to_Price_Quarterly'],
            'Leverage': ['Debt_to_Assets_Quarterly', 'Debt_to_Equity_Quarterly'],
            'Growth': ['Earnings_Growth_RPF_AFQ', 'Sales_Growth_RPF_AFQ'],
            'Dividend Yield': ['Dividend_Yield_Quarterly'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_60D', 'Amihud_Liquidity_125D', 'ISC_Ret_Score'],
            'Market Sensitivity': ['Market_Sensitivity_104W'],
            'Volatility': ['Volatility_125D'],
            'Medium-Term Momentum': ['Momentum_260x21D_Regional'],
            'Exchange Rate Sensitivity': ['XRate_104W_XDR'],
            'Profitability': ['Return_on_Equity_Quarterly', 'Return_on_Assets_Quarterly',
                              'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
                              'Sales_to_Assets_Quarterly', 'Gross_Margin_Quarterly'],
            }

    DescriptorWeights = {}
   
    orthogList = {'Volatility': [['Market Sensitivity'], True, 1.0]}
    
    exposureConfigFile = 'exposures-CAResearchModelEM10'

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
        #self.fundAssetTypes = ['CEFund', 'InvT', 'Misc', 'UnitT'] # default
        self.fundAssetTypes = ['CEFund', 'InvT', 'Misc'] # overwrite default to keep UnitT
        self.elig_parameters = {'assetTypes': self.commonStockTypes + ['REIT'] + ['UnitT'],
                                'excludeTypes': None,
                                'use_isin_country_Flag': False,
                                'remove_China_AB': True,
                                'addBack_H_DR': False,
                                }  

        self.estu_parameters = {'minNonZero':0.1,
                    'minNonMissing':0.5,
                    #'ADV_percentile': [5, 100], # HOW IS THIS USED??
                    'maskZeroWithNoADV_Flag': True, 
                    'CapByNumber_Flag': True,
                    'CapByNumber_hiCapQuota': 250,
                    'CapByNumber_lowCapQuota': 150,
                    'market_lower_pctile': np.nan,
                    'country_lower_pctile': np.nan,
                    'industry_lower_pctile': np.nan,
                    'dummyThreshold': 6,
                    #'inflation_cutoff':0.03
                    'inflation_cutoff':0.01 # CAAxioma2009MH setting - change to .03?
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


        # Manually reassign select assets to RMG and currency
        self.tweakDict = dict()

    def setCalculators(self, modelDB, overrider=False):
        # Set up regression parameters
        #dummyThreshold = 10
        dummyThreshold = 6  # CAAxioma2009MH setting - change to 10?
        
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
