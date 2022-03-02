import datetime
import logging
import numpy as np
import numpy.ma as ma
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
from riskmodels import Standardization_V4
from riskmodels import AssetProcessor
import riskmodels
import numpy
import pandas
import itertools
import copy

class NA4ResearchModel1(EquityModel.FundamentalModel):
    """Version 4 AP medium-horizon fundamental model with GICS 2018
    """
    rm_id = -16
    revision = 3
    rms_id = -200

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

    interceptFactor = 'Global Market'
    intercept = ModelFactor(interceptFactor, interceptFactor)
    gicsDate = datetime.date(2018, 9, 29)
    industryClassification = Classification.GICSIndustries(gicsDate)
    
    DescriptorMap = {
            'Earnings Yield': ['Earnings_to_Price_Quarterly', 'Est_Earnings_to_Price_12MFL_Quarterly'],
            'Value': ['Book_to_Price_Quarterly'],
            'Leverage': ['Debt_to_Assets_Quarterly', 'Debt_to_Equity_Quarterly'],
            'Growth': ['Earnings_Growth_RPF_AFQ', 'Sales_Growth_RPF_AFQ'],
            'Dividend Yield': ['Dividend_Yield_Quarterly'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_60D', 'Amihud_Liquidity_125D', 'ISC_Ret_Score'],
            'Market Sensitivity': ['NA_Regional_Market_Sensitivity_500D'], # to be confirmed, US used "Market_Sensitivity_250D"
            'Volatility': ['NA_Volatility_125D'],
            'Medium-Term Momentum': ['Momentum_260x21D_Regional'], # to be confirmed.
            'Exchange Rate Sensitivity': ['XRate_104W_XDR'],
            'Profitability': ['Return_on_Equity_Quarterly', 'Return_on_Assets_Quarterly',
                'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
                'Sales_to_Assets_Quarterly', 'Gross_Margin_Quarterly'],
            }


    DescriptorWeights = {'Earnings Yield': [0.75, 0.25],}
    orthogList = {'Volatility': [['Market Sensitivity'], True, 1.0],}
    exposureConfigFile = 'exposures-NAAxaiom2019MH'

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.NAAxioma2018MH')

        # Set important model parameters
        ModelParameters2017.defaultModelSettings(self, scm=False)

        # Set up relevant styles to be created/used
        ModelParameters2017.defaultExposureParameters(
                self, self.styleList, configFile=self.exposureConfigFile, descriptorNumeraire='USD')

        # Initialise
        EquityModel.FundamentalModel.__init__(self, ['SEDOL', 'CUSIP'], modelDB, marketDB)
        self.currencyModel = riskmodels.getModelByName('FXAxioma2017USD_MH')(modelDB, marketDB)

        self.elig_parameters = {'assetTypes': self.commonStockTypes + ['REIT'] + ['UnitT'], #None,
                               'excludeTypes': None, #self.etfAssetTypes + self.otherAssetTypes,
                               'use_isin_country_Flag': False,
                               'remove_China_AB':True,
                               'addBack_H_DR':True
                               }

        # So we can use the same ESTU method as the single country fundamental model
        self.baseSCMs = [RiskModels.USAxioma2016MH(modelDB, marketDB),
                         RiskModels.CAAxioma2018MH(modelDB, marketDB)]

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

        # Manually reassign select assets to RMG and currency
        self.tweakDict = dict()

    def setCalculators(self, modelDB, overrider=False):
        # Set up internal factor return regression parameters
        dummyThreshold = 10
        self.internalCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType='Industry Groups',
                dummyThreshold=dummyThreshold,
                marketRegression=False,
                kappa=5.0,
                useRealMCaps=True,
                regWeight='rootCap',
                #dcWeight='cap',
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
                #dcWeight='cap',
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
                #dcWeight='cap',
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
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=3, overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=3, overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, overrider=overrider)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)


    # overload generate_estimation_universe method in the original class, note that this function calls base SCM's generate_estimation_universe method with necessary inputs,
    # no interaction to alter the sub function calculation.

    def generate_estimation_universe(self, modelDate, data, modelDB, marketDB, excludeFactors=[], grandfatherRMS_ID=None):

         #################################### added section ####################################
        # This section is added for regional model where the estu is generated by combining the underlying
        # SCMs' estus together, for NA4, this is effective combining US and CA country models
        if hasattr(self, 'baseSCMs'):
 
            estu_idx_list = []

            # loop through all the given SCMs
            for n,i in enumerate(self.baseSCMs):
                self.log.info('########## Processing estimation universe for scm: %s begin ##########', i.name)
                 # generate risk model instance
                i.setFactorsForDate(modelDate, modelDB)
                

                data_temp = copy.deepcopy(data)

                # # Generate universe of eligible assets for SCM
                # if hasattr(self, 'elig_parameters'):
                #     i.elig_parameters['use_isin_country_Flag'] = self.elig_parameters['use_isin_country_Flag']

                data_temp.eligibleUniverse = i.generate_eligible_universe(
                    modelDate, data_temp, modelDB, marketDB)

                # overwrite estimation universe parameteres if necessary
                if hasattr(self, 'estu_parameters'):
                    i.estu_parameters = self.estu_parameters.copy()

                idx_list = i.generate_estimation_universe(
                    modelDate, data_temp, modelDB, marketDB, grandfatherRMS_ID=self.rms_id)
                self.log.info('estu size for %s is %d', i.name, len(idx_list))

                estu_idx_list += idx_list
                self.log.info('########## Processing estimation universe for scm: %s end ##########', i.name)

                # update NA4 model's estuMap dictionary by combining base scm's estuMap, 
                # note that the model need to be set to a base SCM first in order to have the 
                # fields: assets, qualify. Assets are list of subissue ids. 
                # This directly map to rmi_estu_v3 table in modeldb
                for key in list(i.estuMap.keys()):
                    if key not in self.estuMap:
                        del i.estuMap[key]
                    else:
                        if n==0:
                            self.estuMap[key] = i.estuMap[key]
                        else:
                            self.estuMap[key].assets += i.estuMap[key].assets
                            self.estuMap[key].qualify += i.estuMap[key].qualify

            # filter the list to make sure each element is unique
            for key in self.estuMap.keys():
                self.estuMap[key].assets = list(set(self.estuMap[key].assets).intersection(set(data.eligibleUniverse)))
                self.estuMap[key].qualify = list(set(self.estuMap[key].qualify).intersection(set(data.eligibleUniverse)))

        # The final estu index is the union of all the results with no duplicate and intersect with eligible universe
        return list(set(estu_idx_list).intersection(set(data.eligibleUniverseIdx)))


class NA4ResearchModel2(EquityModel.FundamentalModel):
    """Version 4 AP medium-horizon fundamental model with GICS 2018
    """
    rm_id = -16
    revision = 5
    rms_id = -201

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

    interceptFactor = 'Global Market'
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
    exposureConfigFile = 'exposures-NAAxaiom2019MH'

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.NAAxioma2018MH')

        # Set important model parameters
        ModelParameters2017.defaultModelSettings(self, scm=False)

        # Set up relevant styles to be created/used
        ModelParameters2017.defaultExposureParameters(
                self, self.styleList, configFile=self.exposureConfigFile, descriptorNumeraire='USD')

        # Initialise
        EquityModel.FundamentalModel.__init__(self, ['SEDOL', 'CUSIP'], modelDB, marketDB)
        self.currencyModel = riskmodels.getModelByName('FXAxioma2017USD_MH')(modelDB, marketDB)

        self.elig_parameters = {'assetTypes': self.commonStockTypes + ['REIT'] + ['UnitT'], #None,
                               'excludeTypes': None, #self.etfAssetTypes + self.otherAssetTypes,
                               'use_isin_country_Flag': False,
                               'remove_China_AB':True,
                               'addBack_H_DR':True}

        # So we can use the same ESTU method as the single country fundamental model
        self.baseSCMs = [RiskModels.USAxioma2016MH(modelDB, marketDB),
                         RiskModels.CAAxioma2018MH(modelDB, marketDB)]

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

        # Manually reassign select assets to RMG and currency
        self.tweakDict = dict()

    def setCalculators(self, modelDB, overrider=False):
        # Set up internal factor return regression parameters
        dummyThreshold = 10
        self.internalCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType='Industry Groups',
                dummyThreshold=dummyThreshold,
                marketRegression=False,
                kappa=5.0,
                useRealMCaps=True,
                regWeight='rootCap',
                #dcWeight='cap',
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
                #dcWeight='cap',
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
                #dcWeight='cap',
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
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=3, overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=3, overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, overrider=overrider)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

    #####################################################################################################################################################
    ## The following code is another version which has not been fully tested (running the whole process again). I think this may be a cleaner version.
    ## With overloading generate_estimation_universe, no need for the generate_model_universe above.



    # overload generate_estimation_universe method in the original class, note that this function calls base SCM's generate_estimation_universe method with necessary inputs,
    # no interaction to alter the sub function calculation.

    def generate_estimation_universe(self, modelDate, data, modelDB, marketDB, excludeFactors=[], grandfatherRMS_ID=None):

         #################################### added section ####################################
        # This section is added for regional model where the estu is generated by combining the underlying
        # SCMs' estus together, for NA4, this is effective combining US and CA country models
        if hasattr(self, 'baseSCMs'):
 
            estu_idx_list = []

            # loop through all the given SCMs
            for n,i in enumerate(self.baseSCMs):
                self.log.info('########## Processing estimation universe for scm: %s begin ##########', i.name)
                 # generate risk model instance
                i.setFactorsForDate(modelDate, modelDB)
                

                data_temp = copy.deepcopy(data)

                # Generate universe of eligible assets for SCM
                data_temp.eligibleUniverse = i.generate_eligible_universe(
                    modelDate, data_temp, modelDB, marketDB)

                # overwrite estimation universe parameteres if necessary
                if hasattr(self, 'estu_parameters'):
                    i.estu_parameters = self.estu_parameters.copy()

                idx_list = i.generate_estimation_universe(
                    modelDate, data_temp, modelDB, marketDB, grandfatherRMS_ID=self.rms_id)
                self.log.info('estu size for %s is %d', i.name, len(idx_list))

                estu_idx_list += idx_list
                self.log.info('########## Processing estimation universe for scm: %s end ##########', i.name)

                # update NA4 model's estuMap dictionary by combining base scm's estuMap, 
                # note that the model need to be set to a base SCM first in order to have the 
                # fields: assets, qualify. Assets are list of subissue ids. 
                # This directly map to rmi_estu_v3 table in modeldb
                for key in list(i.estuMap.keys()):
                    if key not in self.estuMap:
                        del i.estuMap[key]
                    else:
                        if n==0:
                            self.estuMap[key] = i.estuMap[key]
                        else:
                            self.estuMap[key].assets += i.estuMap[key].assets
                            self.estuMap[key].qualify += i.estuMap[key].qualify

            # filter the list to make sure each element is unique
            for key in self.estuMap.keys():
                self.estuMap[key].assets = list(set(self.estuMap[key].assets).intersection(set(data.eligibleUniverse)))
                self.estuMap[key].qualify = list(set(self.estuMap[key].qualify).intersection(set(data.eligibleUniverse)))

        # The final estu index is the union of all the results with no duplicate and intersect with eligible universe
        return list(set(estu_idx_list).intersection(set(data.eligibleUniverseIdx)))

  
class NA21ResearchModelV4(EquityModel.FundamentalModel):
    """Version 4 AP medium-horizon fundamental model with GICS 2018
    """
    rm_id = -16
    revision = 6
    rms_id = -94

    styleList = ['Value',
                 'Leverage',
                 'Growth',
                 'Size',
                 'Liquidity',
                 'Volatility',
                 'Medium-Term Momentum',
                 'Exchange Rate Sensitivity',
                 ]

    interceptFactor = 'Global Market'
    intercept = ModelFactor(interceptFactor, interceptFactor)
    industryClassification = Classification.GICSCustomNA(
            datetime.date(2008,8,30))
    
    DescriptorMap = {
            'Value': ['Book_to_Price_Quarterly', 'Earnings_to_Price_Quarterly'],
            'Leverage': ['Debt_to_Assets_Quarterly'],
            'Growth': ['Earnings_Growth_RPF_AFQ', 'Sales_Growth_RPF_AFQ','Return_on_Equity_Quarterly'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_20D'],
            'Volatility': ['Volatility_60D'],
            'Medium-Term Momentum': ['Momentum_260x21D_Regional'],
            'Exchange Rate Sensitivity': ['XRate_52W_XDR'],
            }


    #DescriptorWeights = {'Value': [0.75, 0.25],}
    exposureConfigFile = 'exposures-NA21V4MH'

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.NA21V4MH')

        # Set important model parameters
        ModelParameters2017.defaultModelSettings(self, scm=False)

        # Set up relevant styles to be created/used
        ModelParameters2017.defaultExposureParameters(
                self, self.styleList, configFile=self.exposureConfigFile, descriptorNumeraire='USD')

        # Initialise
        EquityModel.FundamentalModel.__init__(self, ['SEDOL', 'CUSIP'], modelDB, marketDB)
        self.currencyModel = riskmodels.getModelByName('FXAxioma2010USD')(modelDB, marketDB)# riskmodels.getModelByName('FXAxioma2017USD_MH')(modelDB, marketDB)

        self.elig_parameters = {'assetTypes': self.commonStockTypes + ['REIT'] + ['UnitT'], #None,
                               'excludeTypes': None, #self.etfAssetTypes + self.otherAssetTypes,
                               'use_isin_country_Flag': False,
                               'remove_China_AB':True,
                               'addBack_H_DR':True}

        # So we can use the same ESTU method as the single country fundamental model
        self.baseSCMs = [RiskModels.USAxioma2016MH(modelDB, marketDB),
                         RiskModels.CAAxioma2018MH(modelDB, marketDB)]

        # self.estu_parameters, self.elig_parameters = ModelParameters2017.defaultRegionalModelEstuParameters(self)

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

        # Manually reassign select assets to RMG and currency
        self.tweakDict = dict()

    def setCalculators(self, modelDB, overrider=False):
        # Set up internal factor return regression parameters
        dummyThreshold = 10
        self.internalCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType='Industry Groups',
                dummyThreshold=dummyThreshold,
                marketRegression=False,
                kappa=5.0,
                useRealMCaps=True,
                regWeight='rootCap',
                #dcWeight='cap',
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
                #dcWeight='cap',
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
                #dcWeight='cap',
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
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=3, overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=3, overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, overrider=overrider)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)


     #####################################################################################################################################################
    ## The following code is another version which has not been fully tested (running the whole process again). I think this may be a cleaner version.
    ## With overloading generate_estimation_universe, no need for the generate_model_universe above.



    # overload generate_estimation_universe method in the original class, note that this function calls base SCM's generate_estimation_universe method with necessary inputs,
    # no interaction to alter the sub function calculation.

    def generate_estimation_universe(self, modelDate, data, modelDB, marketDB, excludeFactors=[], grandfatherRMS_ID=None):

         #################################### added section ####################################
        # This section is added for regional model where the estu is generated by combining the underlying
        # SCMs' estus together, for NA4, this is effective combining US and CA country models
        if hasattr(self, 'baseSCMs'):
 
            estu_idx_list = []

            # loop through all the given SCMs
            for n,i in enumerate(self.baseSCMs):
                self.log.info('########## Processing estimation universe for scm: %s begin ##########', i.name)
                 # generate risk model instance
                i.setFactorsForDate(modelDate, modelDB)
                

                data_temp = copy.deepcopy(data)

                # Generate universe of eligible assets for SCM
                data_temp.eligibleUniverse = i.generate_eligible_universe(
                    modelDate, data_temp, modelDB, marketDB)

                # overwrite estimation universe parameteres if necessary
                if hasattr(self, 'estu_parameters'):
                    i.estu_parameters = self.estu_parameters.copy()

                idx_list = i.generate_estimation_universe(
                    modelDate, data_temp, modelDB, marketDB, grandfatherRMS_ID=self.rms_id)
                self.log.info('estu size for %s is %d', i.name, len(idx_list))

                estu_idx_list += idx_list
                self.log.info('########## Processing estimation universe for scm: %s end ##########', i.name)

                # update NA4 model's estuMap dictionary by combining base scm's estuMap, 
                # note that the model need to be set to a base SCM first in order to have the 
                # fields: assets, qualify. Assets are list of subissue ids. 
                # This directly map to rmi_estu_v3 table in modeldb
                for key in list(i.estuMap.keys()):
                    if key not in self.estuMap:
                        del i.estuMap[key]
                    else:
                        if n==0:
                            self.estuMap[key] = i.estuMap[key]
                        else:
                            self.estuMap[key].assets += i.estuMap[key].assets
                            self.estuMap[key].qualify += i.estuMap[key].qualify

            # filter the list to make sure each element is unique
            for key in self.estuMap.keys():
                self.estuMap[key].assets = list(set(self.estuMap[key].assets).intersection(set(data.eligibleUniverse)))
                self.estuMap[key].qualify = list(set(self.estuMap[key].qualify).intersection(set(data.eligibleUniverse)))

        # The final estu index is the union of all the results with no duplicate and intersect with eligible universe
        return list(set(estu_idx_list).intersection(set(data.eligibleUniverseIdx)))    
    
