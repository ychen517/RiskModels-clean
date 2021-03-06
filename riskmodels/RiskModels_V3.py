import copy
import datetime
import itertools
import logging
import numpy.ma as ma
import numpy as np
import numpy
import os.path

from marketdb.ConfigManager import findConfig
from riskmodels import Classification
from riskmodels import LegacyCurrencyRisk as CurrencyRisk
from riskmodels import EstimationUniverse
from riskmodels import GlobalExposures
from riskmodels import MarketIndex
from riskmodels import Matrices
from riskmodels.Matrices import ExposureMatrix
from riskmodels import MFM
from riskmodels.Factors import ModelFactor
from riskmodels.Factors import CompositeFactor
from riskmodels import LegacyModelParameters as ModelParameters
from riskmodels import RegressionToolbox
from riskmodels import ReturnCalculator
from riskmodels import RiskCalculator
from riskmodels import Standardization
from riskmodels import Standardization_US3
from riskmodels import StyleExposures
from riskmodels import LegacyFactorReturns
from riskmodels import TOPIX
from riskmodels import LegacyUtilities as Utilities
from riskmodels import AssetProcessor
from riskmodels import LegacyProcessReturns
from riskmodels import EquityModel

def defaultFundamentalCovarianceParameters(rm, modelHorizon='medium', nwLag=1, dva='spline', overrider=False):

    # Fundamental model setup
    if modelHorizon == 'medium':
        varParameters = {'halfLife': 125, 'minObs': 250, 'maxObs': 500, 'NWLag': nwLag}
        corrParameters = {'halfLife': 250, 'minObs': 250, 'maxObs': 1000, 'NWLag': nwLag}
        srParameters = {'halfLife': 125, 'minObs': 125, 'maxObs': 125,
                'NWLag': 1, 'clipBounds': (-15.0,18.0)}

    elif modelHorizon == 'short':
        varParameters = {'halfLife': 60, 'minObs': 125, 'maxObs': 250, 'NWLag': nwLag}
        corrParameters = {'halfLife': 125, 'minObs': 250, 'maxObs': 500, 'NWLag': nwLag}
        srParameters = {'halfLife': 60, 'minObs': 60, 'maxObs': 60,
                'NWLag': 1, 'clipBounds': (-15.0,18.0)}

    fullCovParameters = corrParameters

    if overrider:
        overrider.overrideCovarianceParams(varParameters, corrParameters, srParameters, fullCovParameters)

    if dva is not None:
        varParameters['DVAWindow'] = varParameters['halfLife']
        varParameters['DVAType'] = dva
        corrParameters['DVAWindow'] = varParameters['halfLife']
        corrParameters['DVAType'] = dva
        if modelHorizon == 'short' and dva=='slowStep':
            varParameters['DVAUpperBound'] = 2.0
            varParameters['DVALowerBound'] = 0.5
            corrParameters['DVAUpperBound'] = 2.0
            corrParameters['DVALowerBound'] = 0.5

    fullCovParameters = corrParameters

    rm.vp = RiskCalculator.RiskParameters2009(varParameters)
    rm.cp = RiskCalculator.RiskParameters2009(corrParameters)
    rm.fp = RiskCalculator.RiskParameters2009(fullCovParameters)
    rm.sp = RiskCalculator.RiskParameters2009(srParameters)
    rm.covarianceCalculator = RiskCalculator.CompositeCovarianceMatrix2009(rm.fp, rm.vp, rm.cp)
    rm.specificRiskCalculator = RiskCalculator.SparseSpecificRisk2010(rm.sp)

def defaultStatisticalCovarianceParameters(rm, modelHorizon='medium', nwLag=1, dva='spline',
        historyLength=250, scaleMinObs=False, longSpecificReturnHistory=True, overrider = False):
    # Stat model setup
    if scaleMinObs != False:
        minObs = int(scaleMinObs*historyLength)
    else:
        minObs = historyLength
    maxObs = historyLength

    if modelHorizon == 'medium':
        varParameters = {'halfLife': 125, 'minObs': minObs, 'maxObs': maxObs, 'NWLag': nwLag}
        corrParameters = {'halfLife': 250, 'minObs': minObs, 'maxObs': maxObs, 'NWLag': nwLag}
        if longSpecificReturnHistory:
            srParameters = {'halfLife': 125, 'minObs': minObs, 'maxObs': maxObs, 'NWLag': 1,
                            'clipBounds': (-15.0,18.0)}
        else:
            srParameters = {'halfLife': 125, 'minObs': 125, 'maxObs': 125, 'NWLag': 1,
                            'clipBounds': (-15.0,18.0)}

    elif modelHorizon == 'short':
        varParameters = {'halfLife': 60, 'minObs': minObs, 'maxObs': maxObs, 'NWLag': nwLag}
        corrParameters = {'halfLife': 125, 'minObs': minObs, 'maxObs': maxObs, 'NWLag': nwLag}
        if longSpecificReturnHistory:
            srParameters = {'halfLife': 60, 'minObs': minObs, 'maxObs': maxObs, 'NWLag': 1,
                    'clipBounds': (-15.0,18.0)}
        else:
            srParameters = {'halfLife': 60, 'minObs': 60, 'maxObs': 60, 'NWLag': 1,
                    'clipBounds': (-15.0,18.0)}

    fullCovParameters = corrParameters

    if overrider:
        overrider.overrideCovarianceParams(varParameters, corrParameters, srParameters, fullCovParameters)

    if dva is not None:
        varParameters['DVAWindow'] = varParameters['halfLife']
        varParameters['DVAType'] = dva
        corrParameters['DVAWindow'] = varParameters['halfLife']
        corrParameters['DVAType'] = dva
        if modelHorizon == 'short' and dva=='slowStep':
            varParameters['DVAUpperBound'] = 2.0
            varParameters['DVALowerBound'] = 0.5
            corrParameters['DVAUpperBound'] = 2.0
            corrParameters['DVALowerBound'] = 0.5

    fullCovParameters = corrParameters

    rm.vp = RiskCalculator.RiskParameters2009(varParameters)
    rm.cp = RiskCalculator.RiskParameters2009(corrParameters)
    rm.fp = RiskCalculator.RiskParameters2009(fullCovParameters)
    rm.sp = RiskCalculator.RiskParameters2009(srParameters)
    rm.covarianceCalculator = RiskCalculator.CompositeCovarianceMatrix2009(rm.fp, rm.vp, rm.cp)
    rm.specificRiskCalculator = RiskCalculator.SparseSpecificRisk2010(rm.sp)

def defaultRegressionParameters(rm, modelDB, dummyType=None,
        dummyThreshold=10.0, scndRegs=None, k_rlm=1.345, weightedRLM=True, overrider=False):
    # Set up return parameters
    if dummyType == None:
        dr = RegressionToolbox.DummyMarketReturns()
    else:
        dr = RegressionToolbox.DummyClsParentReturns(
                rm.industrySchemeDict, dummyType, modelDB)
    if len(rm.rmg) > 1:
        constraintList = [[
                RegressionToolbox.ConstraintSumToZero(ExposureMatrix.IndustryFactor),
                RegressionToolbox.ConstraintSumToZero(ExposureMatrix.CountryFactor),
            ]]
        regressionList = [[
                ExposureMatrix.InterceptFactor, ExposureMatrix.StyleFactor,
                ExposureMatrix.IndustryFactor, ExposureMatrix.CountryFactor,
            ]]
        if scndRegs != None:
            for regNo in range(len(scndRegs)):
                constraintList.append([])
                regressionList.append(scndRegs[regNo])
        regParameters = {
                'fixThinFactors': True,
                'dummyReturns': dr,
                'dummyWeights': RegressionToolbox.AxiomaDummyWeights(dummyThreshold),
                'factorConstraints': constraintList,
                'regressionOrder': regressionList,
                'k_rlm': k_rlm,
                'weightedRLM': weightedRLM,
                'whiteStdErrors': False
            }
    else:
        regParameters = {
                'fixThinFactors': True,
                'dummyReturns': dr,
                'dummyWeights': RegressionToolbox.AxiomaDummyWeights(dummyThreshold),
                'k_rlm': k_rlm,
                'whiteStdErrors': False
            }

    if overrider:
        overrider.overrideRegressionParams(regParameters)

    returnCalculator = ReturnCalculator.RobustRegression2(
            RegressionToolbox.RegressionParameters(regParameters))
    return returnCalculator

# SCMs
class USAxioma2009MH(MFM.SingleCountryFundamentalModel):
    """Production AX-US (Medium-Term) model with 2008 GICS revision.
    """
    rm_id = 20
    revision = 2
    rms_id = 131
    styles = [
              ModelFactor('Value', 'Value'),
              ModelFactor('Leverage', 'Leverage'),
              ModelFactor('Growth', 'Growth'),
              ModelFactor('Size', 'Size'),
              ModelFactor('Market Sensitivity', 'Market Sensitivity'),
              ModelFactor('Liquidity','Liquidity'),
              ModelFactor('Short-Term Momentum', 'Short-Term Momentum'),
              ModelFactor('Medium-Term Momentum', 'Medium-Term Momentum'),
              ModelFactor('Exchange Rate Sensitivity', 'Exchange Rate Sensitivity'),
              ModelFactor('Volatility', 'Volatility'),
             ]
    industryClassification = Classification.GICSIndustries(
         datetime.date(2008,8,30))
    quarterlyFundamentalData = True
    proxyDividendPayout = False
    newExposureFormat = True

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.USAxioma2009MH')
        MFM.SingleCountryFundamentalModel.__init__(
                            self, ['CUSIP'], modelDB, marketDB)

        self.setCalculators(modelDB)
        modelDB.createCurrencyCache(marketDB)
        self.indexSelector = MarketIndex.MarketIndexSelector(modelDB, marketDB)

    def setCalculators(self, modelDB, overrider = False):
        # Set up regression parameters
        self.returnCalculator = \
                defaultRegressionParameters(\
                self, modelDB, dummyType='Industry Groups', dummyThreshold=6.0, overrider = overrider)
        # Set up risk parameters
        defaultFundamentalCovarianceParameters(self, nwLag=2, dva='slowStep', overrider=overrider)

    def generate_estimation_universe(self, modelDate, exposureData,
                                     modelDB, marketDB):
        """Estimation universe selection criteria for AX-US.
        """
        self.log.debug('generate_estimation_universe: begin')
        buildEstu = EstimationUniverse.ConstructEstimationUniverse(
                        exposureData.exposureMatrix.getAssets(), self, modelDB, marketDB)

        # Remove assets from the exclusion table (BRK-A/B, etc.)
        (estu, nonest) = buildEstu.apply_exclusion_list(modelDate)

        # Remove ADRs and foreign listings
        (estu, nonest) = buildEstu.exclude_by_market_classification(
                        modelDate, 'HomeCountry', 'REGIONS', ['US'], baseEstu=estu)

        # Remove some specific asset types
        (estu, nonest) = self.excludeAssetTypes(
                modelDate, exposureData, modelDB, marketDB, buildEstu, estu)

        # Keep only common stocks and REITs (no ADRs)
        (estu0, nonest) = buildEstu.exclude_by_market_classification(
                        modelDate, 'TQA FTID Domestic Asset Type', 'ASSET TYPES', ['C','I'],
                        baseEstu=estu)

        # One more safeguard to weed out remaining ADRs and ETFs
        (adr, nonadr) = buildEstu.exclude_by_market_classification(
                        modelDate, 'DataStream2 Asset Type', 'ASSET TYPES', ['ADR','GDR','ET','ETF'],
                        keepMissing=False)
        estu = list(set(estu0).difference(adr))
        self.log.info('Removed an additional %d ADRs and ETFs' % (len(estu0)-len(estu)))

        # Weed out foreign issuers by ISIN country prefix
        (estu, nonest)  = buildEstu.exclude_by_isin_country(['US'],
                            modelDate, baseEstu=estu)

        # Keep only issues trading on NYSE and NASDAQ
        if modelDate.year >= 2008:
            (estu0, nonest) = buildEstu.exclude_by_market_classification(
                            modelDate, 'Market', 'REGIONS', ['NAS','NYS','IEX','EXI'], baseEstu=estu)
        else:
            estu0 = estu

        # Rank stuff by market cap and total volume over past year
        (estu, nonest) = buildEstu.filter_by_cap_and_volume(
                        exposureData, modelDate, baseEstu=estu0)

        # Inflate thin industry factors if possible
        (estu, nonest) = buildEstu.pump_up_factors(
                        exposureData, modelDate, currentEstu=estu, baseEstu=estu0,
                        minFactorWidth=self.returnCalculator.\
                                parameters.getDummyWeights().minAssets)

        self.log.debug('generate_estimation_universe: end')
        return estu

    def generate_model_specific_exposures(self, modelDate, data, modelDB, marketDB):
        """Generate the Exchange Rate Sensitivity factor.
        """
        data.exposureMatrix.addFactor(
            'Exchange Rate Sensitivity', StyleExposures.generate_forex_sensitivity(
            data.returns, self, modelDB, 120, 'XDR'), ExposureMatrix.StyleFactor)
        return data.exposureMatrix

class USAxioma2009SH(USAxioma2009MH):
    """Production AX-US (Short-Term) model with 2008 GICS revision.
    """
    rm_id = 22
    revision = 2
    rms_id = 132

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.USAxioma2009SH')
        MFM.SingleCountryFundamentalModel.__init__(
                            self, ['CUSIP'], modelDB, marketDB)
        self.setCalculators(modelDB)

    def setCalculators(self, modelDB, overrider = False):
        # Set up regression parameters
        self.returnCalculator = \
                defaultRegressionParameters(\
                self, modelDB, dummyType='Industry Groups', dummyThreshold=6.0, overrider=overrider)
        # Set up risk parameters
        defaultFundamentalCovarianceParameters(\
                self, modelHorizon='short', nwLag=2, dva='slowStep', overrider=overrider)

class USAxioma2009MH_S(MFM.StatisticalFactorModel):
    """Production AX-US (Medium-Term) statistical model
    """
    rm_id = 21
    revision = 2
    rms_id = 133
    numFactors = 15
    blind = [ModelFactor('Statistical Factor %d' % n,
                         'Statistical Factor %d' % n)
             for n in range(1, numFactors+1)]
    returnHistory = 250
    industryClassification = Classification.GICSIndustries(
        datetime.date(2008,8,30))
    newExposureFormat = True

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.USAxioma2009MH_S')
        MFM.StatisticalFactorModel.__init__(
                            self, ['CUSIP'], modelDB, marketDB)
        self.setCalculators(modelDB)
        # So we can use the same ESTU method as USAxioma2009MH
        self.baseModel = USAxioma2009MH(modelDB, marketDB)

    def setCalculators(self, modelDB, overrider = False):
        self.returnCalculator = ReturnCalculator. \
                                AsymptoticPrincipalComponents2(self.numFactors)
        # Set up risk parameters
        defaultStatisticalCovarianceParameters(\
                self, dva='slowStep', longSpecificReturnHistory=False, overrider=overrider)

class USAxioma2009SH_S(USAxioma2009MH_S):
    """Production AX-US (Short-Term) statistical model
    """
    rm_id = 23
    revision = 2
    rms_id = 134

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.USAxioma2009SH_S')
        MFM.StatisticalFactorModel.__init__(
                            self, ['CUSIP'], modelDB, marketDB)
        self.setCalculators(modelDB)
        # So we can use the same ESTU method as USAxioma2009MH
        self.baseModel = USAxioma2009MH(modelDB, marketDB)

    def setCalculators(self, modelDB, overrider = False):
        self.returnCalculator = ReturnCalculator. \
                                AsymptoticPrincipalComponents2(self.numFactors)
        # Set up risk parameters
        defaultStatisticalCovarianceParameters(self, modelHorizon='short',
                dva='slowStep', longSpecificReturnHistory=False, overrider=overrider)

class USAxioma2013MH(MFM.FundamentalModel):
    """US3 fundamental medium-horizon model.
    """
    rm_id = 105
    revision = 1
    rms_id = 170
    newExposureFormat = True
    standardizationStats = True
    SCM = True
    noProxyType = []

    # List of style factors in the model
    styleList = ['Value',
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

    # Setting up market intercept if relevant
    addList = ['Dividend Yield']
    interceptFactor = 'Market Intercept'
    intercept = ModelFactor(interceptFactor, interceptFactor)
    industryClassification = Classification.GICSIndustries(
            datetime.date(2008,8,30))
    quarterlyFundamentalData = True
    proxyDividendPayout = False
    fancyMAD = False
    legacyEstGrowth = True

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.USAxioma2013MH')
        # Set up relevant styles to be created/used
        ModelParameters.defaultExposureParameters(self, self.styleList)
        self.styles = [s for s in self.totalStyles if s.name in self.styleList]
        MFM.FundamentalModel.__init__(self, ['CUSIP'], modelDB, marketDB)

        # Model-specific exposure parameter stuff here
        dummyThreshold = 10
        self.styleParameters['Volatility'].orthogCoef = 0.8
        self.styleParameters['Volatility'].sqrtWt = False
        self.styleParameters['Dividend Yield'].includeSpecial = True
        self.styleParameters['Liquidity'].legacy = True
        self.styleParameters['Value'].descriptors = ['Book-to-Price Nu','Est Earnings-to-Price']
        self.legacyISCSwitchDate=datetime.date(2015, 2, 28)

        # Set up regression parameters
        ModelParameters.defaultRegressionParametersLegacy(
                self, modelDB,
                dummyType='Industry Groups',
                marketReg=True,
                constrainedReg=False,
                scndRegs=False,
                k_rlm=[8.0, 1.345],
                dummyThreshold=dummyThreshold)

        self.fmpCalculator = ModelParameters.defaultRegressionParameters(
                self, modelDB,
                dummyType='Industry Groups',
                marketRegression=False,
                dummyThreshold=dummyThreshold,
                kappa=None)

        # Set up risk parameters
        ModelParameters.defaultFundamentalCovarianceParameters(self, nwLag=2, dwe=False)

        # Set up standardization parameters
        gloScope = Standardization_US3.GlobalRelativeScope([f.name for f in self.styles])
        self.exposureStandardization = Standardization_US3.BucketizedStandardization(
                [gloScope], fancyMAD=self.fancyMAD)

        self.indexSelector = MarketIndex.MarketIndexSelector(modelDB, marketDB)
        modelDB.createCurrencyCache(marketDB)

    def setFactorsForDate(self, date, modelDB):
        """Determine which country/currency factors are in the
        model for the given date.
        """
        # Set up estimation universe parameters
        self.estuMap = modelDB.getEstuMappingTable(self.rms_id)
        if self.estuMap is None:
            logging.info('No estimation universe mapping defined')

        # Determine risk model groups (countries) in the model
        self.setRiskModelGroupsForDate(date)
        if hasattr(self, 'baseModelDateMap'):
            self.setBaseModelForDate(date)
        else:
            self.baseModel = None
        self.SCM = True

        factors = modelDB.getRiskModelSerieFactors(self.rms_id)
        self.descFactorMap = dict([(i.description, i) for i in factors])
        self.nameFactorMap = dict([(i.name, i) for i in factors])
        self.allStyles = list(self.styles)

        # Assign to new industry scheme if necessary
        if hasattr(self, 'industrySchemeDict'):
            chngDates = sorted(d for d in self.industrySchemeDict.keys() if d <= date)
            self.industryClassification = self.industrySchemeDict[chngDates[-1]]
            self.log.debug('Using %s classification scheme, rev_dt: %s'%\
                          (self.industryClassification.name, chngDates[-1].isoformat()))

        # Create industry factors
        industries = list(self.industryClassification.getLeafNodes(modelDB).values())
        self.industries = [ModelFactor(None, f.description) for f in industries]

        countries = []
        currencies = []
        self.countryRMGMap = dict()

        # Get factor types in order
        regional = countries + currencies
        allFactors = self.allStyles + self.industries + regional
        if self.intercept is not None and self.intercept not in allFactors:
            allFactors = allFactors + [self.intercept]
        if len(self.localStructureFactors) > 0:
            allFactors = allFactors + self.localStructureFactors

        for f in allFactors:
            if f.name in self.nameFactorMap:
                dbFactor = self.nameFactorMap[f.name]
            else:
                dbFactor = self.descFactorMap[f.description]
            f.factorID = dbFactor.factorID
            f.from_dt = dbFactor.from_dt
            f.thru_dt = dbFactor.thru_dt
            if f.name is None:
                f.name = dbFactor.name
            if f.description is None:
                f.description = dbFactor.description

        # Drop dead styles and currencies
        dropped = [s for s in self.allStyles if not s.isLive(date)]
        if len(dropped) > 0:
            self.log.info('%d styles not live in current model: %s', len(dropped), dropped)
        self.styles = [s for s in self.allStyles if s.isLive(date)]
        self.currencies = [c for c in currencies if c.isLive(date)]
        self.countries = [f for f in countries if f.isLive(date)]
        self.nurseryCountries = [f for f in countries if f not in self.countries]
        self.hiddenCurrencies = [f for f in currencies if f not in self.currencies]
        self.nurseryRMGs = [self.CountryFactorRMGMap[f] for f in self.nurseryCountries]

        # Set up dicts
        self.factors = [f for f in allFactors if f.isLive(date)]
        self.factorIDMap = dict([(f.factorID, f) for f in allFactors])
        self.validateFactorStructure(date, warnOnly=self.variableStyles)
        self.allFactors = allFactors

    def generate_eligible_universe(self, modelDate, data, modelDB, marketDB,
            buildEstu=None, assetTypes=None):
        """Creates subset of eligible assets for consideration
        in US estimation universes
        """
        self.log.debug('generate_eligible_universe: begin')

        if buildEstu is None:
            buildEstu = EstimationUniverse.ConstructEstimationUniverse(
                data.universe, self, modelDB, marketDB)

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
        data.eligibleUniverseIdx = estuIdx
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
                    modelDate, 'Market', 'REGIONS', ['NAS','NYS','IEX','EXI'], baseEstu=eligibleUniverseIdx)
        else:
            eligibleExchangeIdx = list(eligibleUniverseIdx)
        eligibleExchange = [buildEstu.assets[idx] for idx in eligibleExchangeIdx]

        # Report on thinly-traded assets over the entire universe
        (nonSparse, sparse) = buildEstu.exclude_thinly_traded_assets(
                                modelDate, data, baseEstu=universeIdx, minNonZero=0.5,
                                maskZeroWithNoADV=True)
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
                modelDate, estuIdx, baseEstu=eligibleExchangeIdx)
        data.ESTUQualify = ESTUQualify

        if self.estuMap is None:
            return estuIdx

        self.estuMap['main'].assets = [buildEstu.assets[idx] for idx in estuIdx]
        self.estuMap['main'].qualify = [buildEstu.assets[idx] for idx in ESTUQualify]
        self.estuMap['eligible'].assets = eligibleUniverse

        self.log.debug('generate_estimation_universe: end')
        return estuIdx

    def generate_fundamental_exposures(self,
                    modelDate, data, modelDB, marketDB):
        """Compute multiple-descriptor fundamental style exposures
        for assets in data.universe for all CompositeFactors in self.factors.
        data should be a Struct() containing the ExposureMatrix
        object as well as any required market data, like market
        caps, asset universe, etc.
        Factor exposures are computed as the equal-weighted average
        of all the normalized descriptors associated with the factor.
        """
        self.log.debug('generate_fundamental_exposures: begin')
        compositeFactors = [f for f in self.styles
                if hasattr(self.styleParameters[f.name], 'descriptors')]
        if len(compositeFactors) == 0:
            self.log.warning('No CompositeFactors found!')
            return data.exposureMatrix

        descriptors = []
        for f in compositeFactors:
            descriptors.extend(self.styleParameters[f.name].descriptors)
        descriptors = sorted(set(descriptors))
        descriptorExposures = Matrices.ExposureMatrix(data.universe)
        for d in descriptors:
            params = None
            if d == 'Book-to-Price':
                values = StyleExposures.generate_book_to_price(
                            modelDate, data, self, modelDB, marketDB,
                            restrict=None, useQuarterlyData=self.quarterlyFundamentalData)
            elif d == 'Book-to-Price Nu':
                values = StyleExposures.generate_book_to_price(
                        modelDate, data, self, modelDB, marketDB,
                        restrict=None, useQuarterlyData=self.quarterlyFundamentalData)
                values = self.proxy_missing_descriptors(
                        modelDate, data, modelDB, marketDB, 'BTP', values, kappa=1.345)
            elif d == 'Earnings-to-Price':
                values = StyleExposures.generate_earnings_to_price(
                            modelDate, data, self, modelDB, marketDB,
                            restrict=None, useQuarterlyData=self.quarterlyFundamentalData,
                            legacy=self.modelHack.legacyETP)
            elif d == 'Earnings-to-Price Nu':
                values = StyleExposures.generate_earnings_to_price(
                        modelDate, data, self, modelDB, marketDB,
                        restrict=None, useQuarterlyData=self.quarterlyFundamentalData,
                        legacy=self.modelHack.legacyETP, maskNegative=False)
            elif d == 'Est Earnings-to-Price':
                params = Utilities.Struct()
                params.maskNegative = False
                params.winsoriseRaw = True
                # Compute est EPS
                est_eps = StyleExposures.generate_est_earnings_to_price(
                        modelDate, data, self, modelDB, marketDB, params,
                        restrict=None, useQuarterlyData=False)
                # Compute realised BTP and proxy missing values
                btp = StyleExposures.generate_book_to_price(
                        modelDate, data, self, modelDB, marketDB,
                        restrict=None, useQuarterlyData=self.quarterlyFundamentalData)
                btp = self.proxy_missing_descriptors(
                        modelDate, data, modelDB, marketDB, 'BTP', btp, kappa=1.345)
                # Compute realised EPS and proxy missing values
                eps = StyleExposures.generate_earnings_to_price(
                        modelDate, data, self, modelDB, marketDB,
                        restrict=None, useQuarterlyData=self.quarterlyFundamentalData,
                        legacy=self.modelHack.legacyETP, maskNegative=params.maskNegative)
                eps = self.proxy_missing_descriptors(
                        modelDate, data, modelDB, marketDB, 'EPS', eps, kappa=1.345)
                # Proxy missing est EPS values
                est_eps_ESTU = ma.take(est_eps, data.estimationUniverseIdx, axis=0)
                btp_ESTU = ma.array(ma.take(btp, data.estimationUniverseIdx, axis=0))
                eps_ESTU = ma.array(ma.take(eps, data.estimationUniverseIdx, axis=0))
                rhs = ma.transpose(ma.array([btp_ESTU, eps_ESTU]))
                # Regress est EPS againsts EPS and BTP
                betas = Utilities.robustLinearSolver(est_eps_ESTU, rhs, robust=True).params
                fullRHS = ma.transpose(ma.array([btp, eps]))
                proxy_values = ma.dot(fullRHS, betas)
                missingEPS = ma.getmaskarray(est_eps)
                missingProxy = ma.getmaskarray(proxy_values)
                proxy_values = ma.masked_where(missingEPS==0, proxy_values)
                # Fill in any missing values with proxy if available
                est_eps = ma.filled(est_eps, 0.0) + ma.filled(proxy_values, 0.0)
                stillMissing = missingEPS * missingProxy
                values = ma.masked_where(stillMissing, est_eps)
                # Take average of EPS and est. EPS
                values = ma.average(ma.array([values, eps]),axis=0)

            elif d == 'Est Earnings-to-Price Nu':
                # Compute est EPS
                est_eps = StyleExposures.generate_est_earnings_to_price_nu(
                        modelDate, data, self, modelDB, marketDB,
                        useQuarterlyData=self.quarterlyFundamentalData)
                # Compute realised BTP and proxy missing values
                btp = StyleExposures.generate_book_to_price(
                        modelDate, data, self, modelDB, marketDB,
                        restrict=None, useQuarterlyData=self.quarterlyFundamentalData)
                btp = self.proxy_missing_descriptors(
                        modelDate, data, modelDB, marketDB, 'BTP', btp, kappa=1.345)
                # Compute realised EPS and proxy missing values
                eps = StyleExposures.generate_earnings_to_price(
                        modelDate, data, self, modelDB, marketDB,
                        restrict=None, useQuarterlyData=self.quarterlyFundamentalData,
                        legacy=self.modelHack.legacyETP, maskNegative=False)
                eps = self.proxy_missing_descriptors(
                        modelDate, data, modelDB, marketDB, 'EPS', eps, kappa=1.345)
                # Proxy missing est EPS values
                est_eps_ESTU = ma.take(est_eps, data.estimationUniverseIdx, axis=0)
                btp_ESTU = ma.array(ma.take(btp, data.estimationUniverseIdx, axis=0))
                eps_ESTU = ma.array(ma.take(eps, data.estimationUniverseIdx, axis=0))
                rhs = ma.transpose(ma.array([btp_ESTU, eps_ESTU]))
                # Regress est EPS againsts EPS and BTP
                betas = Utilities.robustLinearSolver(est_eps_ESTU, rhs, robust=True).params
                fullRHS = ma.transpose(ma.array([btp, eps]))
                proxy_values = ma.dot(fullRHS, betas)
                missingEPS = ma.getmaskarray(est_eps)
                missingProxy = ma.getmaskarray(proxy_values)
                proxy_values = ma.masked_where(missingEPS==0, proxy_values)
                # Fill in any missing values with proxy if available
                est_eps = ma.filled(est_eps, 0.0) + ma.filled(proxy_values, 0.0)
                stillMissing = missingEPS * missingProxy
                values = ma.masked_where(stillMissing, est_eps)

            elif d == 'Est Revenue':
                values = StyleExposures.generate_estimate_revenue(
                            modelDate, data, self, modelDB, marketDB,
                            restrict=None,
                            useQuarterlyData=False)
            elif d == 'Sales-to-Price':
                values = StyleExposures.generate_sales_to_price(
                            modelDate, data, self, modelDB, marketDB,
                            restrict=None,
                            useQuarterlyData=self.quarterlyFundamentalData)
            elif d == 'Debt-to-Assets':
                values = StyleExposures.generate_debt_to_marketcap(
                            modelDate, data, self, modelDB, marketDB,
                            restrict=None,
                            useQuarterlyData=self.quarterlyFundamentalData,
                            useTotalAssets=True)
                values = ma.where(values < 0.0, 0.0, values)   # Negative debt -> 0.0
            elif d == 'Debt-to-MarketCap':
                values = StyleExposures.generate_debt_to_marketcap(
                            modelDate, data, self, modelDB, marketDB,
                            restrict=None,
                            useQuarterlyData=self.quarterlyFundamentalData)
            elif d == 'Plowback times ROE':
                roe = StyleExposures.generate_return_on_equity(
                            modelDate, data, self, modelDB, marketDB,
                            restrict=None,
                            useQuarterlyData=self.quarterlyFundamentalData)
                if not self.proxyDividendPayout:
                    divPayout = StyleExposures.generate_dividend_payout(
                            modelDate, data, self, modelDB, marketDB,
                            useQuarterlyData=self.quarterlyFundamentalData)
                else:
                    divPayout = StyleExposures.generate_proxied_dividend_payout(
                            modelDate, data, self, modelDB, marketDB,
                            restrict=None,
                            useQuarterlyData=self.quarterlyFundamentalData,
                            includeStock=self.allowStockDividends)
                values = (1.0 - divPayout) * roe
            elif d == 'Dividend Payout':
                values = StyleExposures.generate_dividend_payout(
                            modelDate, data, self, modelDB, marketDB,
                            useQuarterlyData=self.quarterlyFundamentalData,
                            maskZero=True)
            elif d == 'Proxied Dividend Payout':
                values = StyleExposures.generate_proxied_dividend_payout(
                            modelDate, data, self, modelDB, marketDB,
                            restrict=None, maskZero=True,
                            useQuarterlyData=self.quarterlyFundamentalData,
                            includeStock=self.allowStockDividends)
            elif d == 'Return-on-Equity':
                values = StyleExposures.generate_return_on_equity(
                            modelDate, data, self, modelDB, marketDB,
                            restrict=None,
                            useQuarterlyData=self.quarterlyFundamentalData)
            elif d == 'Return-on-Assets':
                values = StyleExposures.generate_return_on_assets(
                            modelDate, data, self, modelDB, marketDB,
                            restrict=None,
                            useQuarterlyData=self.quarterlyFundamentalData)
            elif d == 'Sales Growth':
                values = StyleExposures.generate_sales_growth(
                            modelDate, data, self, modelDB, marketDB,
                            restrict=None,
                            useQuarterlyData=self.quarterlyFundamentalData)
            elif d == 'Est Sales Growth':
                values = StyleExposures.generate_est_sales_growth(
                        modelDate, data, self, modelDB, marketDB,
                        restrict=None, winsoriseRaw=True,
                        useQuarterlyData=self.quarterlyFundamentalData,
                        legacy=self.legacyEstGrowth)
            elif d == 'Earnings Growth':
                values = StyleExposures.generate_earnings_growth(
                            modelDate, data, self, modelDB, marketDB,
                            restrict=None,
                            useQuarterlyData=self.quarterlyFundamentalData)
            elif d == 'Est Earnings Growth':
                values = StyleExposures.generate_est_earnings_growth(
                        modelDate, data, self, modelDB, marketDB,
                        restrict=None, winsoriseRaw=True,
                        useQuarterlyData=self.quarterlyFundamentalData,
                        legacy=self.legacyEstGrowth)
            elif d == 'Est EBITDA':
                values = StyleExposures.generate_estimate_EBITDA(
                            modelDate, data, self, modelDB, marketDB,
                            restrict=None,
                            useQuarterlyData=False)
            elif d == 'Est Enterprise Value':
                values = StyleExposures.generate_estimate_enterprise_value(
                            modelDate, data, self, modelDB, marketDB,
                            restrict=None,
                            useQuarterlyData=False)
            elif d == 'Est Cash-Flow-per-Share':
                values = StyleExposures.generate_estimate_cash_flow_per_share(
                            modelDate, data, self, modelDB, marketDB,
                            restrict=None,
                            useQuarterlyData=False)
            elif d == 'Est Avg Return-on-Equity':
                values = StyleExposures.generate_est_return_on_equity(
                            modelDate, data, self, modelDB, marketDB,
                            restrict=None,
                            useQuarterlyData=False)
            elif d == 'Est Avg Return-on-Assets':
                values = StyleExposures.generate_est_return_on_assets(
                            modelDate, data, self, modelDB, marketDB,
                            restrict=None,
                            useQuarterlyData=False)
            elif d == 'Dividend Yield':
                values = StyleExposures.generate_dividend_yield(
                            modelDate, data, self, modelDB, marketDB,
                            restrict=None,
                            params=self.styleParameters['Dividend Yield'])
            elif d == 'Market Sensitivity Descriptor':
                params = self.styleParameters['Market Sensitivity']
                mm = Utilities.run_market_model_v3(
                        self.rmg[0], data.returns, modelDB, marketDB, params,
                        debugOutput=self.debuggingReporting,
                        clippedReturns=data.clippedReturns)
                values = mm.beta
            elif d == 'Volatility Descriptor':
                params = self.styleParameters['Volatility']
                values = StyleExposures.generate_cross_sectional_volatility_v3(
                        data.returns, params, indices=data.estimationUniverseIdx,
                        clippedReturns=data.clippedReturns)
            elif d == 'Liquidity Descriptor':
                params = self.styleParameters['Liquidity']
                values = StyleExposures.generate_trading_volume_exposures_v3(
                        modelDate, data, self.rmg, modelDB,
                        params, self.numeraire.currency_id)
            elif d == 'Amihud Liquidity Descriptor':
                params = self.styleParameters['Amihud Liquidity']
                if params.legacy:
                    values = StyleExposures.generateAmihudLiquidityExposures(
                            modelDate, data.returns, data, self.rmg, modelDB,
                            params, self.numeraire.currency_id, scaleByTO=True)
                else:
                    values = StyleExposures.generateAmihudLiquidityExposures(
                            modelDate, data.returns, data, self.rmg, modelDB,
                            params, self.numeraire.currency_id, scaleByTO=True,
                            originalReturns=data.returns.originalData)
            elif d == 'Proportion Returns Traded':
                params = self.styleParameters['Liquidity']
                values = StyleExposures.generate_proportion_non_missing_returns(
                        data.returns, data, modelDB, marketDB, daysBack=params.daysBack)
            elif d == 'Share Buyback':
                values = StyleExposures.generate_share_buyback(
                         modelDate, data, self, modelDB, marketDB,
                         restrict = None, maskZero=True)
            elif d == 'Short Interest':
                values = StyleExposures.generate_short_interest(
                         modelDate, data, self, modelDB, marketDB,
                         restrict = None, maskZero=True)
            else:
                raise Exception('Undefined descriptor %s!' % d)

            descriptorExposures.addFactor(
                    d, values, ExposureMatrix.StyleFactor)

        # Add country factors to ExposureMatrix, needed for regional-relative standardization
        country_indices = data.exposureMatrix.\
                            getFactorIndices(ExposureMatrix.CountryFactor)
        if len(country_indices) > 0:
            countryExposures = ma.take(data.exposureMatrix.getMatrix(),
                                country_indices, axis=0)
            countryNames = data.exposureMatrix.getFactorNames(ExposureMatrix.CountryFactor)
            descriptorExposures.addFactors(countryNames, countryExposures,
                                ExposureMatrix.CountryFactor)

        if self.debuggingReporting:
            descriptorExposures.dumpToFile('tmp/raw-Desc-%s-%04d%02d%02d.csv'\
                    % (self.name, modelDate.year, modelDate.month, modelDate.day),
                    modelDB, marketDB, modelDate, estu=data.estimationUniverseIdx, dp=self.dplace)

        # Standardize raw descriptors for multi-descriptor factors
        mat = descriptorExposures.getMatrix()
        origStandardization = copy.copy(self.exposureStandardization)
        if not self.SCM:
            self.exposureStandardization = Standardization_US3.BucketizedStandardization(
                    [Standardization_US3.RegionRelativeScope(
                        modelDB, descriptors)], fancyMAD=self.fancyMAD)
        else:
            self.exposureStandardization = Standardization_US3.BucketizedStandardization(
                    [Standardization_US3.GlobalRelativeScope(descriptors)], fancyMAD=self.fancyMAD)
        self.standardizeExposures(descriptorExposures, data, modelDate, modelDB, marketDB)

        # Update exposure matrix standardisation stats
        if hasattr(data.exposureMatrix, 'meanDict'):
            descriptorExposures.meanDict.update(data.exposureMatrix.meanDict)
            descriptorExposures.stdDict.update(data.exposureMatrix.stdDict)
        data.exposureMatrix.meanDict = descriptorExposures.meanDict
        data.exposureMatrix.stdDict = descriptorExposures.stdDict

        # Form multi-descriptor CompositeFactors and add to ExposureMatrix
        for cf in compositeFactors:
            cfDescriptors = self.styleParameters[cf.name].descriptors
            self.log.info('Factor %s has %d descriptor(s): %s',
                    cf.description, len(cfDescriptors), cfDescriptors)
            valueList = [mat[descriptorExposures.getFactorIndex(d),:] for d in cfDescriptors]
            valueList = ma.array(valueList)

            #### Quick and dirty hack for Value factor
            for (idx, d) in enumerate(cfDescriptors):
                if d == 'Est Earnings-to-Price Nu':
                    valueList[idx,:] *= 0.25
                if d == 'Earnings-to-Price Nu':
                    valueList[idx,:] *= .75

            if len(valueList) > 1:
                e = ma.average(valueList, axis=0)
            else:
                e = valueList[0,:]
            data.exposureMatrix.addFactor(cf.description, e, ExposureMatrix.StyleFactor)

        # Proxy (standardized) exposures for assets missing data
        self.proxy_missing_exposures(modelDate, data, modelDB, marketDB,
                         factorNames=[cf.description for cf in compositeFactors], kappa=1.345,
                         legacy=True)
        self.exposureStandardization = origStandardization

        self.log.debug('generate_md_fundamental_exposures: end')
        return data.exposureMatrix

    def proxy_missing_descriptors(self, modelDate, data, modelDB, marketDB,
            descriptorName, values, sizeVec=None, kappa=5.0):
        self.log.debug('proxy_missing_descriptors: begin')
        expM = data.exposureMatrix

        origStandardization = copy.copy(self.exposureStandardization)
        if not self.SCM:
            self.exposureStandardization = Standardization_US3.BucketizedStandardization(
                    [Standardization_US3.RegionRelativeScope(
                        modelDB, descriptorName)], fancyMAD=self.fancyMAD)
        else:
            self.exposureStandardization = Standardization_US3.BucketizedStandardization(
                    [Standardization_US3.GlobalRelativeScope(descriptorName)], fancyMAD=self.fancyMAD)

        # Prepare sector exposures, will be used for missing data proxies
        roots = self.industryClassification.getClassificationRoots(modelDB)
        root = [r for r in roots if r.name=='Sectors'][0]
        sectorNames = [n.description for n in self.\
                industryClassification.getClassificationChildren(root, modelDB)]
        sectorExposures = Matrices.ExposureMatrix(data.universe)
        e = self.industryClassification.getExposures(modelDate,
                data.universe, sectorNames, modelDB, level=-(len(roots)-1))
        sectorExposures.addFactors(sectorNames, e, ExposureMatrix.IndustryFactor)

        # Set up regression to proxy raw exposures for assets missing data
        rp = {'dummyReturnType': 'market',
              'dummyThreshold': 10,
              'k_rlm': kappa}
        proxyParameters = ModelParameters.RegressionParameters2012(rp)
        returnCalculator = LegacyFactorReturns.RobustRegressionLegacy([proxyParameters])
        returnCalculator.regParameters = returnCalculator.allParameters[0]
        dp = returnCalculator.allParameters[0].getThinFactorInformation()
        mat = expM.getMatrix()

        # Determine which assets are missing data and require proxying
        missingIndices = numpy.flatnonzero(ma.getmaskarray(values))
        estuSet = set(data.estimationUniverseIdx)
        self.log.info('%d/%d assets missing %s fundamental data (%d/%d ESTU)',
                    len(missingIndices), len(values), descriptorName,
                    len(estuSet.intersection(missingIndices)), len(estuSet))
        if len(missingIndices)==0:
            self.exposureStandardization = origStandardization
            return values

        # Loop around regions
        for (regionName, asset_indices) in self.exposureStandardization.\
                factorScopes[0].getAssetIndices(expM, modelDate):
            missing_indices = list(set(asset_indices).intersection(missingIndices))
            good_indicesUniv = set(asset_indices).difference(missingIndices)
            good_indices = list(good_indicesUniv.intersection(estuSet))
            if len(good_indices) <= 10:
                good_indices = list(good_indicesUniv.intersection(set(\
                        self.estuMap['eligible'].assetIdx)))
                if len(good_indices) <= 10:
                    if len(missing_indices) > 0:
                        self.log.warning('Too few assets (%d) in %s with %s data present',
                                    len(good_indices), regionName, descriptorName)
                    continue

            good_data = ma.take(values, good_indices, axis=0)
            good_data = ma.masked_where(good_data==0, good_data)
            nonMissingData = numpy.flatnonzero(ma.getmaskarray(good_data)==0)
            if len(nonMissingData) == 0:
                self.log.warning('All non-missing values are zero, skipping')
                continue

            # Assemble regressand, regressor matrix and weights
            weights = numpy.take(data.marketCaps, good_indices, axis=0)**0.5
            regressand = ma.take(values, good_indices, axis=0)
            regressor = ma.zeros((len(good_indices), len(sectorNames) + 1))
            if sizeVec is None:
                regressor[:,0] = ma.take(mat[expM.getFactorIndex(
                                'Size'),:], good_indices, axis=0)
            else:
                regressor[:,0] = ma.take(sizeVec, good_indices, axis=0)
            regressor[:,1:] = ma.transpose(ma.take(
                    sectorExposures.getMatrix(), good_indices, axis=1).filled(0.0))
            regressor = ma.transpose(regressor)
            # Set up thin factor correction
            dp.factorIndices = list(range(regressor.shape[0]))
            dp.factorNames = ['Size'] + sectorNames
            dp.factorTypes = [ExposureMatrix.StyleFactor] + ([ExposureMatrix.IndustryFactor] * (len(dp.factorIndices)-1))
            dp.nonzeroExposuresIdx = [0]
            returnCalculator.computeDummyReturns(\
                    dp, ma.array(regressand), list(range(len(weights))), weights,
                    [data.universe[j] for j in good_indices], modelDate, data)
            returnCalculator.thinFactorParameters = dp

            # Run regression to get proxy values
            self.log.info('Running %s proxy regression for %s (%d assets)',
                    descriptorName, regionName, len(good_indices))
            coefs = returnCalculator.calc_Factor_Specific_Returns(
                    self, modelDate, list(range(regressor.shape[1])),
                    regressand, regressor, ['Size'] + sectorNames, weights, None,
                    returnCalculator.allParameters[0].getFactorConstraints()).factorReturns

            # Substitute proxies for missing values
            self.log.info('Proxying %d %s exposures for %s',
                    len(missing_indices), descriptorName, regionName)
            for i in range(len(sectorNames)):
                regSectorExposures = sectorExposures.getMatrix()[i,missing_indices]
                reg_sec_indices = numpy.flatnonzero(ma.getmaskarray(regSectorExposures)==0)
                if len(reg_sec_indices)==0:
                    continue
                reg_sec_indices = [missing_indices[j] for j in reg_sec_indices]
                sectorCoef = coefs[i+1]
                if sizeVec is None:
                    sizeExposures = mat[expM.getFactorIndex(
                                    'Size'), reg_sec_indices]
                else:
                    sizeExposures = ma.take(sizeVec, reg_sec_indices, axis=0)
                proxies = sizeExposures * coefs[0] + sectorCoef
                for (ii, idx) in enumerate(reg_sec_indices):
                    values[idx] = proxies[ii]

        self.log.debug('proxy_missing_descriptors: end')
        self.exposureStandardization = origStandardization
        return values

    def generate_market_exposures(self, modelDate, data, modelDB, marketDB):
        """Compute exposures for market factors for the assets
        in data.universe and add them to data.exposureMatrix.
        """
        self.log.debug('generate_market_exposures: begin')
        expM = data.exposureMatrix
        styleNames = [s.name for s in self.styles]

        # Determine how many calendar days of data we need
        requiredDays = self.returnHistory - 1

        # Get the daily returns for the last 250 trading days
        returnsProcessor = LegacyProcessReturns.assetReturnsProcessor(
                self.rmg, data.universe, data.rmgAssetMap, data.tradingRmgAssetMap,
                data.assetTypeDict, data.marketTypeDict, debuggingReporting=self.debuggingReporting,
                numeraire_id=self.numeraire.currency_id,
                returnsTimingID=self.returnsTimingId)
        returns = returnsProcessor.process_returns_history(
                modelDate, int(self.returnHistory), modelDB, marketDB,
                drCurrMap=data.drCurrData, loadOnly=True,
                applyRT=(self.SCM==False))

        if returns.data.shape[1] > self.returnHistory:
            returns.data = returns.data[:,-self.returnHistory:]
            returns.dates = returns.dates[-self.returnHistory:]

        # On the fly proxy to fill in any remaining missing returns
        logging.info('Using older proxy returns fill-in')
        returns = Utilities.proxyMissingAssetReturnsLegacy(
                self.rmg, modelDate, returns, data, modelDB)

        data.returns = returns
        data.clippedReturns = ma.filled(
                Utilities.twodMAD(returns.data, nDev=self.nDev,
                        estu=data.estimationUniverseIdx), 0.0)
        data.returns.data = ma.filled(data.returns.data, 0.0)

        # If regional model, we also require returns in numeraire
        if not self.SCM:
            numeraireReturns = returnsProcessor.process_returns_history(
                    modelDate, int(self.returnHistory), modelDB, marketDB,
                    drCurrMap=self.numeraire.currency_id, loadOnly=True)
            if numeraireReturns.data.shape[1] > self.returnHistory:
                numeraireReturns.data = numeraireReturns.data[:,-self.returnHistory:]
            data.clippedNumeraireReturns = ma.filled(
                    Utilities.twodMAD(numeraireReturns.data, nDev=self.nDev,
                        estu=data.estimationUniverseIdx), 0.0)

        if returns.data.shape[1] < requiredDays:
            raise LookupError('Not enough previous trading days (need %d, got %d)' \
                        % (requiredDays, returns.data.shape[1]))
        else:
            self.log.debug('Loaded %d days of returns (from %s) for exposures',
                           requiredDays + 1, returns.dates[0])

        # Create intercept factor
        if self.intercept is not None:
            if self.SCM:
                # Create simple intercept
                if self.intercept.name == 'Market Intercept':
                    beta = numpy.ones((len(data.universe)), float)
                # Or a beta sensitivity
                elif self.intercept.name == 'Market Beta':
                    params = self.styleParameters['Market Sensitivity']
                    mm = Utilities.run_market_model_v3(
                            self.rmg[0], data.returns, modelDB, marketDB, params,
                            debugOutput=self.debuggingReporting,
                            clippedReturns=data.clippedReturns)
                    beta = mm.beta
            else:
                beta = numpy.ones((len(data.universe)), float)
            expM.addFactor(self.intercept.name, beta, ExposureMatrix.InterceptFactor)

        # Momentum factors
        mmList = ['Short-Term Momentum',
                  'Medium-Term Momentum',
                  'Yearly Momentum',
                  'Medium-Term Momentum (6 Months)',
                  'Short-Term Momentum Tm2']
        for mom in mmList:
            if mom in styleNames:
                params = self.styleParameters[mom]
                stm = StyleExposures.generate_momentum(data.returns, params)
                expM.addFactor(mom, stm, ExposureMatrix.StyleFactor)


        mmList = ['Medium-Term Momentum (Clipped)',
                  'Short-Term Momentum (Clipped)']
        for mom in mmList:
            if mom in styleNames:
                tmpData = ma.array(data.returns.data, copy=True)
                data.returns.data = data.clippedReturns
                params = self.styleParameters[mom]
                stm = StyleExposures.generate_momentum(data.returns, params)
                data.returns.data = tmpData
                expM.addFactor(mom, stm, ExposureMatrix.StyleFactor)

        # Trading volume factors
        liqList = ['Liquidity', 'Liquidity 60', 'Liquidity (Median)',
                   'Liquidity SH', 'Liquidity (2_1)',
                   'Dollar Volume', 'Dollar Volume (Median)']
        for fac in liqList:
            if fac in styleNames:
                params = self.styleParameters[fac]
                liq = StyleExposures.generate_trading_volume_exposures_v3(
                        modelDate, data, self.rmg, modelDB,
                        params, self.numeraire.currency_id)
                expM.addFactor(fac, liq, ExposureMatrix.StyleFactor)

        if 'Amihud Liquidity' in styleNames:
            params = self.styleParameters['Amihud Liquidity']
            if params.legacy:
                iliq = StyleExposures.generateAmihudLiquidityExposures(
                        modelDate, data.returns, data, self.rmg, modelDB,
                        params, self.numeraire.currency_id, scaleByTO=True)
            else:
                iliq = StyleExposures.generateAmihudLiquidityExposures(
                        modelDate, data.returns, data, self.rmg, modelDB,
                        params, self.numeraire.currency_id, scaleByTO=True,
                        originalReturns=data.returns.originalData)
            expM.addFactor('Amihud Liquidity', iliq, ExposureMatrix.StyleFactor)

        if 'Returns Skewness' in styleNames:
            params = self.styleParameters['Returns Skewness']
            skew = StyleExposures.generate_returns_skewness(data.returns, params)
            expM.addFactor('Returns Skewness', skew, ExposureMatrix.StyleFactor)

        if 'Size' in styleNames:
            params = self.styleParameters['Size']
            size = StyleExposures.generate_size_exposures(data)
            expM.addFactor('Size', size, ExposureMatrix.StyleFactor)

        if 'Size Non-linear' in styleNames:
            params = self.styleParameters['Size Non-linear']
            size = StyleExposures.generate_size_nl_exposures(data)
            expM.addFactor('Size Non-linear', size, ExposureMatrix.StyleFactor)

        msList = ['Market Sensitivity', 'Market Sensitivity SH',
                  'Market Sensitivity (PartOg)', 'Market Sensitivity SH (PartOg)']
        for ms in msList:
            if ms in styleNames:
                params = self.styleParameters[ms]
                hl = int(len(data.returns.dates) * params.historyScale)
                if len(self.rmg) > 1:
                    region = modelDB.getRiskModelRegion(self.RegionPortfolioID)
                    mm = Utilities.run_market_model_v3(
                            self.rmg, data.returns, modelDB, marketDB, params,
                            debugOutput=self.debuggingReporting,
                            clippedReturns=data.clippedNumeraireReturns,
                            marketRegion=region, historyLength=hl)
                else:
                    mm = Utilities.run_market_model_v3(
                            self.rmg[0], data.returns, modelDB, marketDB, params,
                            debugOutput=self.debuggingReporting,
                            clippedReturns=data.clippedReturns,
                            historyLength=hl)
                expM.addFactor(ms, mm.beta, ExposureMatrix.StyleFactor)

        if 'Residual Volatility' in styleNames:
            params = self.styleParameters['Residual Volatility']
            mm = Utilities.run_market_model_v3(
                    self.rmg[0], data.returns, modelDB, marketDB, params,
                    debugOutput=self.debuggingReporting,
                    clippedReturns=data.clippedReturns)
            tmpReturns = ma.array(data.returns.data, copy=True)
            data.returns.data = mm.resid
            values = StyleExposures.generate_cross_sectional_volatility_v3(
                    data.returns, params, indices=data.estimationUniverseIdx,
                    clippedReturns=data.clippedReturns)
            data.returns.data = tmpReturns
            expM.addFactor('Residual Volatility', values, ExposureMatrix.StyleFactor)

        hrList = ['Historical Residual Volatility',
                  'Hist Resid Vol (PartOg)',
                  'Hist Resid Vol (PartOg Size)']
        for hr in hrList:
            if hr in styleNames:
                params = self.styleParameters[hr]
                mm = Utilities.run_market_model_v3(
                        self.rmg[0], data.returns, modelDB, marketDB, params,
                        debugOutput=self.debuggingReporting,
                        clippedReturns=data.clippedReturns)
                expM.addFactor(hr, mm.sigma, ExposureMatrix.StyleFactor)

        if 'Historical Residual Std SizeAdj' in styleNames:
            params = self.styleParameters['Historical Residual Std SizeAdj']
            mm = Utilities.run_market_model_v3(
                    self.rmg[0], data.returns, modelDB, marketDB, params,
                    debugOutput=self.debuggingReporting,
                    clippedReturns=data.clippedReturns)
            std = ma.sqrt(mm.sigma)
            lnMcap = ma.log(data.marketCaps)
            lnMcap4 = ma.power(lnMcap,[4]*len(lnMcap))
            val = ma.multiply(std,lnMcap4)
            expM.addFactor('Historical Residual Std SizeAdj', val, ExposureMatrix.StyleFactor)

        if 'Ln Historical Residual Volatility' in styleNames:
            params = self.styleParameters['Ln Historical Residual Volatility']
            mm = Utilities.run_market_model_v3(
                    self.rmg[0], data.returns, modelDB, marketDB, params,
                    debugOutput=self.debuggingReporting,
                    clippedReturns=data.clippedReturns)
            expM.addFactor('Ln Historical Residual Volatility', numpy.log(mm.sigma), ExposureMatrix.StyleFactor)

        if 'Historical Residual Volatility ex Size' in styleNames:
            params = self.styleParameters['Historical Residual Volatility ex Size']
            mm = Utilities.run_market_model_ff(
                    self.rmg[0], data.returns, modelDB, marketDB, params,
                    debugOutput=self.debuggingReporting,
                    clippedReturns=data.clippedReturns)
            expM.addFactor('Historical Residual Volatility ex Size', mm.sigma, ExposureMatrix.StyleFactor)

        volList = ['Volatility', 'Volatility SH', 'Volatility 250 Day (PartOg)', 'Volatility (Orthog)',
                   'Volatility 60 Day', 'Volatility 125 Day', 'Volatility 250 Day',]
        for volName in volList:
            if volName in styleNames:
                params = self.styleParameters[volName]
                values = StyleExposures.generate_cross_sectional_volatility_v3(
                        data.returns, params, indices=data.estimationUniverseIdx,
                        clippedReturns=data.clippedReturns)
                expM.addFactor(volName, values, ExposureMatrix.StyleFactor)

        if 'Historical Volatility' in styleNames:
            params = self.styleParameters['Historical Volatility']
            values = StyleExposures.generate_historic_volatility(data.returns, params)
            expM.addFactor('Historical Volatility', values, ExposureMatrix.StyleFactor)

        # XRT factors
        xrtList = ['Exchange Rate Sensitivity',
                   'Exchange Rate Sensitivity (GBP)',
                   'Exchange Rate Sensitivity (EUR)',
                   'Exchange Rate Sensitivity (JPY)',
                   'Exchange Rate Sensitivity (USD)']
        for xrt in xrtList:
            if xrt in styleNames:
                params = self.styleParameters[xrt]
                val = StyleExposures.generate_forex_sensitivity_v3(
                        data, self, modelDB, params,
                        clippedReturns=data.clippedReturns)
                expM.addFactor(xrt, val, ExposureMatrix.StyleFactor)

        self.log.debug('generate_market_exposures: end')

    def generate_model_specific_exposures(self, modelDate, data, modelDB, marketDB):
        """Generate the non-default factors.
        """
        return data.exposureMatrix

    def generateExposureMatrix(self, modelDate, modelDB, marketDB):
        """Generates and returns the exposure matrix for the given date.
        The exposures are not inserted into the database.
        Data is accessed through the modelDB and marketDB DAOs.
        The return is a structure containing the exposure matrix
        (exposureMatrix), the universe as a list of assets (universe),
        and a list of market capitalizations (marketCaps).
        """
        self.log.debug('generateExposureMatrix: begin')

        # Get risk model universe and market caps
        # Determine home country info and flag DR-like instruments
        rmi = modelDB.getRiskModelInstance(self.rms_id, modelDate)
        universe = modelDB.getRiskModelInstanceUniverse(rmi, returnExtra=True)
        data = AssetProcessor.process_asset_information(
                modelDate, universe, self.rmg, modelDB, marketDB,
                checkHomeCountry=(self.SCM==0),
                numeraire_id=self.numeraire.currency_id,
                legacyDates=True,
                forceRun=self.forceRun,
                nurseryRMGList=self.nurseryRMGs)
        data.exposureMatrix = Matrices.ExposureMatrix(data.universe)
        data.eligibleUniverse = self.generate_eligible_universe(
                modelDate, data, modelDB, marketDB)

        if self.SCM and not hasattr(self, 'indexSelector'):
            self.indexSelector = MarketIndex.\
                    MarketIndexSelector(modelDB, marketDB)
            self.log.info('Index Selector: %s', self.indexSelector)

        # Fetch trading calendars for all risk model groups
        # Start-date should depend on how long a history is required
        # for exposures computation
        data.rmgCalendarMap = dict()
        startDate = modelDate - datetime.timedelta(365*2)
        for rmg in self.rmg:
            data.rmgCalendarMap[rmg.rmg_id] = \
                    modelDB.getDateRange(rmg, startDate, modelDate)

        # Compute issuer-level market caps if required
        data.issuerMarketCaps = GlobalExposures.computeTotalIssuerMarketCaps(
                data, modelDate, self.numeraire, modelDB, marketDB,
                debugReport=self.debuggingReporting)
        data.issuerTotalMarketCaps = data.issuerMarketCaps.copy()

        if not self.SCM:
            data.exposureMatrix = GlobalExposures.generate_binary_country_exposures(
                            modelDate, self, modelDB, marketDB, data)
            data.exposureMatrix = GlobalExposures.generate_currency_exposures(
                            modelDate, self, modelDB, marketDB, data)

        # Generate 0/1 industry exposures
        self.generate_industry_exposures(
            modelDate, modelDB, marketDB, data.exposureMatrix)

        # Generate universe of eligible assets
        data.eligibleUniverse = self.generate_eligible_universe(
                modelDate, data, modelDB, marketDB, assetTypes=self.estuAssetTypes)

        # Load estimation universe
        estu = self.loadEstimationUniverse(rmi, modelDB, data)

        # Generate market exposures
        self.generate_market_exposures(modelDate, data, modelDB, marketDB)

        # Generate fundamental data exposures
        self.generate_fundamental_exposures(modelDate, data, modelDB, marketDB)

        # Generate other, model-specific factor exposures
        data.exposureMatrix = self.generate_model_specific_exposures(
            modelDate, data, modelDB, marketDB)

        # Clone DR and cross-listing exposures if required
        subIssueGroups = modelDB.getIssueCompanyGroups(
                modelDate, data.universe, marketDB)
        scores = self.score_linked_assets(
                modelDate, data.universe, modelDB, marketDB,
                subIssueGroups=data.subIssueGroups)
        data.exposureMatrix = self.clone_linked_asset_exposures(
                modelDate, data, modelDB, marketDB, scores,
                subIssueGroups=subIssueGroups)

        if self.debuggingReporting:
            dr_indices = [data.assetIdxMap[sid] for sid in data.foreign]
            data.exposureMatrix.dumpToFile('tmp/raw-expM-%s-%04d%02d%02d.csv'\
                    % (self.name, modelDate.year, modelDate.month, modelDate.day),
                    modelDB, marketDB, modelDate, estu=data.estimationUniverseIdx,
                    subIssueGroups=subIssueGroups, assetType=data.assetTypeDict, dp=self.dplace)

        tmpDebug = self.debuggingReporting
        self.debuggingReporting = False
        self.standardizeExposures(data.exposureMatrix, data, modelDate, modelDB, marketDB, subIssueGroups)

        # Orthogonalise where required
        orthogDict = dict()
        for st in self.styles:
            params = self.styleParameters[st.name]
            if hasattr(params, 'orthog'):
                if not hasattr(params, 'sqrtWt'):
                    params.sqrtWt = True
                if not hasattr(params, 'orthogCoef'):
                    params.orthogCoef = 1.0
                if params.orthog is not None and len(params.orthog) > 0:
                    orthogDict[st.name] = (params.orthog, params.orthogCoef, params.sqrtWt)

        if len(orthogDict) > 0:
            Utilities.partial_orthogonalisation(modelDate, data, modelDB, marketDB, orthogDict)
            tmpExcNames = list(self.exposureStandardization.exceptionNames)
            self.exposureStandardization.exceptionNames = [st.name for st in self.styles if st.name not in orthogDict]
            self.standardizeExposures(data.exposureMatrix, data, modelDate,
                        modelDB, marketDB, subIssueGroups)
            self.exposureStandardization.exceptionNames = tmpExcNames
        self.debuggingReporting = tmpDebug

        expMatrix = data.exposureMatrix.getMatrix()
        for st in self.styles:
            params = self.styleParameters[st.name]
            if hasattr(params, 'fillWithZero'):
                if (params.fillWithZero is True) and (st.name in data.exposureMatrix.meanDict):
                    fIdx = data.exposureMatrix.getFactorIndex(st.name)
                    values = expMatrix[fIdx,:]
                    nMissing = numpy.flatnonzero(ma.getmaskarray(values))
                    fillValue = (0.0 - data.exposureMatrix.meanDict[st.name] / \
                            data.exposureMatrix.stdDict[st.name])
                    expMatrix[fIdx,:] = ma.filled(values, fillValue)
                    logging.info('Filling %d missing values for %s with standardised zero: %.2f',
                            len(nMissing), st.name, fillValue)

        if self.debuggingReporting:
            dr_indices = [data.assetIdxMap[sid] for sid in data.foreign]
            data.exposureMatrix.dumpToFile('tmp/expM-%s-%04d%02d%02d.csv'\
                    % (self.name, modelDate.year, modelDate.month, modelDate.day),
                    modelDB, marketDB, modelDate, estu=data.estimationUniverseIdx,
                    subIssueGroups=subIssueGroups, assetType=data.assetTypeDict, dp=self.dplace)

        self.log.debug('generateExposureMatrix: end')
        return data

    def assetReturnLoader(self, data, modelDate, modelDB, marketDB, buildFMP=False):
        """ Function to load in returns for factor regression
        """
        daysBack = 21
        returnsProcessor = LegacyProcessReturns.assetReturnsProcessor(
                self.rmg, data.universe, data.rmgAssetMap,
                data.tradingRmgAssetMap, data.assetTypeDict,
                data.marketTypeDict, numeraire_id=self.numeraire.currency_id,
                returnsTimingID=self.returnsTimingId,
                debuggingReporting=self.debuggingReporting)
        assetReturnMatrix = returnsProcessor.process_returns_history(
                modelDate, daysBack, modelDB, marketDB,
                drCurrMap=data.drCurrData, loadOnly=True,
                applyProxy=(buildFMP==False), applyRT=(self.SCM==False))
        return assetReturnMatrix

    def insertLegacyEstimationUniverse(self, rmi, assets, estuIdx, modelDB, qualify):
        """Inserts the estimation universes into the database for the given
        risk model instance.
        """
        estuAssets = [assets[idx] for idx in estuIdx]
        logging.info('Writing to legacy estu table')
        modelDB.insertEstimationUniverse(rmi, estuAssets, qualify)

    def insertEstimationUniverseWeights(self, rmi, subidWeightPairs, modelDB, estuName='main'):
        """Inserts model estimation (regression) weights for estimation
        universe assets in the given risk model instance.
        """
        modelDB.insertEstimationUniverseWeights(rmi, subidWeightPairs)
        modelDB.insertEstimationUniverseWeightsV3(rmi, self.estuMap[estuName], estuName, subidWeightPairs)

class USAxioma2013FL(USAxioma2013MH):
    """Production AX-US Factor Library model
    """
    rm_id = 109
    revision = 1
    rms_id = 174

    styleList = [
            # Keep the base model factors, at least for now, as there
            # are some cross-factor dependencies in the code
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
            'Liquidity SH',
            'Volatility SH',
            'Market Sensitivity SH',
            'Short-Term Momentum',
            # Value
            'Value (2_1)',
            'Book-to-Price',
            'Earnings-to-Price',
            'Sales-to-Price',
            'Est Earnings-to-Price',
            # Cashback
            'Dividend Payout',
            'Proxied Dividend Payout',
            # Growth
            'Growth (2_1)',
            'Sustainable Growth Rate',
            'Sales Growth',
            'Est Sales Growth',
            'Earnings Growth',
            'Est Earnings Growth',
            # Leverage
            'Debt-to-MarketCap',
            # Quality
            'Return-on-Assets',
            # Liquidity
            'Liquidity (2_1)',
            'Dollar Volume',
            'Amihud Liquidity',
            # Volatility
            'Volatility 60 Day',
            'Volatility 125 Day',
            'Volatility 250 Day',
            'Volatility 250 Day (PartOg)',
            'Historical Residual Volatility',
            'Market Sensitivity (PartOg)',
            'Market Sensitivity SH (PartOg)',
            # Momentum
            'Medium-Term Momentum (6 Months)',
            'Yearly Momentum',
            # XRT
            'Exchange Rate Sensitivity (GBP)',
            'Exchange Rate Sensitivity (EUR)',
            'Exchange Rate Sensitivity (JPY)',
            ]
    newExposureFormat = True
    standardizationStats = True
    addList = ['Size']
    # Setting up market intercept if relevant
    interceptFactor = 'Market Intercept'
    intercept = ModelFactor(interceptFactor, interceptFactor)
    industryClassification = Classification.GICSIndustries(datetime.date(2008,8,30))
    quarterlyFundamentalData = True
    proxyDividendPayout = False

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.USAxioma2013FL')

        # Set up relevant styles to be created/used
        ModelParameters.defaultExposureParameters(self, self.styleList)
        self.styles = [s for s in self.totalStyles if s.name in self.styleList]
        MFM.FundamentalModel.__init__(self, ['CUSIP'], modelDB, marketDB)

        # Model-specific tweaks to default parameters
        self.styleParameters['Dividend Yield'].includeSpecial = True
        self.styleParameters['Liquidity'].legacy = True
        self.styleParameters['Liquidity SH'].legacy = True
        self.styleParameters['Amihud Liquidity'].legacy = True
        self.styleParameters['Liquidity (2_1)'].legacy = True
        self.styleParameters['Value'].descriptors = ['Book-to-Price Nu','Est Earnings-to-Price']
        resetList = ['Volatility 250 Day (PartOg)', 'Market Sensitivity (PartOg)',
                     'Market Sensitivity SH (PartOg)', 'Volatility', 'Volatility SH']
        for styleName in resetList:
            self.styleParameters[styleName].orthogCoef = 0.8
            self.styleParameters[styleName].sqrtWt = False
        self.legacyISCSwitchDate=datetime.date(2015, 2, 28)

        self.setCalculators(modelDB)
        self.indexSelector = MarketIndex.MarketIndexSelector(modelDB, marketDB)
        modelDB.createCurrencyCache(marketDB)

    def setCalculators(self, modelDB, overrider = False):
        # Set up regression parameters
        dummyThreshold = 10
        ModelParameters.defaultRegressionParametersLegacy(
                self, modelDB,
                dummyType='Industry Groups',
                marketReg=True,
                constrainedReg=False,
                scndRegs=False,
                k_rlm=[8.0, 1.345],
                dummyThreshold=dummyThreshold)

        # Set up risk parameters
        ModelParameters.defaultFundamentalCovarianceParameters(self, nwLag=2, dwe=False)

        # Set up standardization parameters
        gloScope = Standardization_US3.GlobalRelativeScope([f.name for f in self.styles])
        self.exposureStandardization = Standardization_US3.BucketizedStandardization([gloScope])

    def loadEstimationUniverse(self, rmi_id, modelDB, data=None):
        """Loads the estimation universe(s) of the given risk model instance.
        Returns a list of sub-issue IDs.
        """
        estu = modelDB.getRiskModelInstanceESTU(rmi_id)
        estu = [sid for sid in estu if sid in data.assetIdxMap]
        if data is not None:
            data.estimationUniverseIdx = [data.assetIdxMap[sid] for sid in estu]
        assert(len(estu) > 0)
        return estu

    def insertEstimationUniverse(self, rmi, assets, estuIdx, modelDB, qualify):
        """Inserts the estimation universes into the database for the given
        risk model instance.
        """
        estuAssets = [assets[idx] for idx in estuIdx]
        modelDB.insertEstimationUniverse(rmi, estuAssets, qualify)

    def insertLegacyEstimationUniverse(self, rmi, assets, estuIdx, modelDB, qualify):
        """Dummy function
        """
        return

    def insertEstimationUniverseWeights(self, rmi, subidWeightPairs, modelDB, estuName='main'):
        """Inserts model estimation (regression) weights for estimation
        universe assets in the given risk model instance.
        """
        modelDB.insertEstimationUniverseWeights(rmi, subidWeightPairs)

class USAxioma2013SH(USAxioma2013MH):
    """Production AX-US (Short-Term) model with 2008 GICS revision.
    """
    rm_id = 107
    revision = 1
    rms_id = 172
    noProxyType = []

    styleList = ['Value',
                 'Leverage',
                 'Growth',
                 'Return-on-Equity',
                 'Dividend Yield',
                 'Size',
                 'Liquidity SH',
                 'Volatility SH',
                 'Market Sensitivity SH',
                 'Short-Term Momentum',
                 'Medium-Term Momentum',
                 'Exchange Rate Sensitivity',
                 ]

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.USAxioma2013SH')
        # Set up relevant styles to be created/used
        ModelParameters.defaultExposureParameters(self, self.styleList)
        self.styles = [s for s in self.totalStyles if s.name in self.styleList]
        MFM.FundamentalModel.__init__(self, ['CUSIP'], modelDB, marketDB)

        # Model-specific tweaks to default parameters
        self.styleParameters['Volatility SH'].orthogCoef = 0.8
        self.styleParameters['Volatility SH'].sqrtWt = False
        self.styleParameters['Liquidity SH'].legacy = True
        self.styleParameters['Dividend Yield'].includeSpecial = True
        self.styleParameters['Value'].descriptors = ['Book-to-Price Nu','Est Earnings-to-Price']
        self.legacyISCSwitchDate=datetime.date(2015, 2, 28)

        self.setCalculators(modelDB)
        self.indexSelector = MarketIndex.MarketIndexSelector(modelDB, marketDB)
        modelDB.createCurrencyCache(marketDB)

    def setCalculators(self, modelDB, overrider = False):
        # Set up regression parameters
        dummyThreshold = 10
        ModelParameters.defaultRegressionParametersLegacy(
                self, modelDB,
                dummyType='Industry Groups',
                marketReg=True,
                constrainedReg=False,
                scndRegs=False,
                k_rlm=[8.0, 1.345],
                dummyThreshold=dummyThreshold)

        self.fmpCalculator = ModelParameters.defaultRegressionParameters(
                self, modelDB,
                dummyType='Industry Groups',
                marketRegression=False,
                dummyThreshold=dummyThreshold,
                kappa=None)

        # Set up risk parameters
        ModelParameters.defaultFundamentalCovarianceParameters(
                self, nwLag=2, modelHorizon='short', dwe=False)

        # Set up standardization parameters
        gloScope = Standardization_US3.GlobalRelativeScope([f.name for f in self.styles])
        self.exposureStandardization = Standardization_US3.BucketizedStandardization([gloScope])

class USAxioma2013MH_S(MFM.StatisticalModel):
    """US3 medium-horizon statistical model
    """
    rm_id = 106
    revision = 1
    rms_id = 171
    numFactors = 15
    blind = [ModelFactor('Statistical Factor %d' % n, 'Statistical Factor %d' % n)
        for n in range(1, numFactors+1)]
    pcaHistory = 250
    industryClassification = Classification.GICSIndustries(datetime.date(2008,8,30))
    newExposureFormat = True
    allowETFs = True
    legacyModel = True
    SCM = True
    noProxyType = []

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.USAxioma2013MH_S')
        MFM.StatisticalModel.__init__(
                self, ['CUSIP'], modelDB, marketDB)
        self.legacyISCSwitchDate=datetime.date(2015, 2, 28)

        # So we can use the same ESTU method as the fundamental model
        self.baseModelDateMap = {datetime.date(1980,1,1): USAxioma2013MH(modelDB, marketDB)}

        # Set up returns model
        self.returnCalculator = LegacyFactorReturns.AsymptoticPrincipalComponentsLegacy(self.numFactors)

        # Set up risk parameters
        ModelParameters.defaultStatisticalCovarianceParameters(\
                self, varDVAOnly=False)
        self.indexSelector = MarketIndex.MarketIndexSelector(modelDB, marketDB)

    def assetReturnHistoryLoader(self, data, returnHistory, modelDate, modelDB, marketDB):
        """ Function to load in returns for factor regression
        """
        # Load in history of returns
        returnsProcessor = LegacyProcessReturns.assetReturnsProcessor(
                self.rmg, data.universe, data.rmgAssetMap, data.tradingRmgAssetMap,
                data.assetTypeDict, data.marketTypeDict, numeraire_id=self.numeraire.currency_id,
                returnsTimingID=self.returnsTimingId)
        returnsHistory = returnsProcessor.process_returns_history(
                modelDate, int(returnHistory), modelDB, marketDB,
                drCurrMap=data.drCurrData, loadOnly=True, excludeWeekend=True,
                applyRT=(self.SCM==False))

        logging.info('Using older proxy returns fill-in')
        returnsHistory = Utilities.proxyMissingAssetReturnsLegacy(
                self.rmg, modelDate, returnsHistory, data, modelDB)
        return returnsHistory

    def insertLegacyEstimationUniverse(self, rmi, assets, estuIdx, modelDB, qualify):
        """Inserts the estimation universes into the database for the given
        risk model instance.
        """
        estuAssets = [assets[idx] for idx in estuIdx]
        logging.info('Writing to legacy estu table')
        modelDB.insertEstimationUniverse(rmi, estuAssets, qualify)

    def insertEstimationUniverseWeights(self, rmi, subidWeightPairs, modelDB, estuName='main'):
        """Inserts model estimation (regression) weights for estimation
        universe assets in the given risk model instance.
        """
        modelDB.insertEstimationUniverseWeights(rmi, subidWeightPairs)
        modelDB.insertEstimationUniverseWeightsV3(rmi, self.estuMap[estuName], estuName, subidWeightPairs)

class USAxioma2013SH_S(USAxioma2013MH_S):
    """US3 short-horizon statistical model
    """
    rm_id = 108
    revision = 1
    rms_id = 173
    numFactors = 15
    blind = [ModelFactor('Statistical Factor %d' % n, 'Statistical Factor %d' % n)
        for n in range(1, numFactors+1)]
    pcaHistory = 250
    industryClassification = Classification.GICSIndustries(datetime.date(2008,8,30))
    newExposureFormat = True
    legacyModel = True
    allowETFs = True
    SCM = True
    noProxyType = []

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.USAxioma2013SH_S')
        MFM.StatisticalModel.__init__(
                self, ['CUSIP'], modelDB, marketDB)
        self.legacyISCSwitchDate=datetime.date(2015, 2, 28)

        # So we can use the same ESTU method as the fundamental model
        self.baseModelDateMap = {datetime.date(1980,1,1): USAxioma2013MH(modelDB, marketDB)}

        # Set up returns model
        self.returnCalculator = LegacyFactorReturns.AsymptoticPrincipalComponentsLegacy(self.numFactors)

        # Set up risk parameters
        ModelParameters.defaultStatisticalCovarianceParameters(\
                self, varDVAOnly=False, modelHorizon='short')
        self.indexSelector = MarketIndex.MarketIndexSelector(modelDB, marketDB)

class USAxioma2013MH_M(MFM.MacroeconomicModel):
    """US3 medium-horizon macroeconomic model"""
    rm_id = 201
    revision = 1
    rms_id = 2001

    newExposureFormat = True
    allowETFs = True
    SCM=True
    noProxyType = []
    returnHistory=1000
    factorHistory=2000
    debugging=False
    industryClassification = Classification.GICSIndustries(
        datetime.date(2008,8,30))

    macro_core = [
        ModelFactor('Economic Growth', 'Economic Growth'),
        ModelFactor('Inflation', 'Inflation'),
        ModelFactor('Confidence', 'Confidence'),
        ]

    macro_market_traded = [
        ModelFactor('Oil', 'Oil'),
        ModelFactor('Gold', 'Gold'),
        ModelFactor('Commodity', 'Commodity'),
        ModelFactor('Credit Spread', 'Credit Spread'),
        ModelFactor('Term Spread', 'Term Spread'),
        ModelFactor('FX Basket', 'FX Basket'),
        ]

    macro_equity = [
        ModelFactor('Equity Market', 'Equity Market'),
        ModelFactor('Equity Size', 'Equity Size'),
        ModelFactor('Equity Value', 'Equity Value')
        ]

    macro_sectors = [
        ModelFactor('Consumer Discretionary', 'Consumer Discretionary'),
        ModelFactor('Consumer Staples', 'Consumer Staples'),
        ModelFactor('Energy', 'Energy'),
        ModelFactor('Financials', 'Financials'),
        ModelFactor('Health Care', 'Health Care'),
        ModelFactor('Industrials', 'Industrials'),
        ModelFactor('Information Technology', 'Information Technology'),
        ModelFactor('Materials', 'Materials'),
        ModelFactor('Telecommunication Services', 'Telecommunication Services'),
        ModelFactor('Utilities', 'Utilities')
        ]

    def __init__(self, modelDB, marketDB):
        from riskmodels import MacroCalculators
        from riskmodels import MacroDataUtils

        self.log = logging.getLogger('RiskModels.USAxioma2013MH_M')
        MFM.MacroeconomicModel.__init__(self, ['CUSIP'], modelDB, marketDB)

        self.fundamentalModel=USAxioma2013MH(modelDB,marketDB)
        self.statisticalModel=USAxioma2013MH_S(modelDB,marketDB)
        self.baseModel=self.fundamentalModel

        self.exposureCalculator=MacroCalculators.SectorExposureCalculator({'sector_residual': True})
        self.legacyISCSwitchDate=datetime.date(2015, 2, 28)

        self.macroMetaDataFile=findConfig('US3-MH-M-MetaData-bare.csv', os.path.join('riskmodels', 'macroconf'))
        self.frcParams=Utilities.Struct()
        self.frcParams.include_sectors=True
        self.frcParams.residualize_delta=False
        self.frcParams.orthogonalize=False
        self.frcParams.macroMetaCsvFile=self.macroMetaDataFile
        self.frcParams.factorsToKill=[]
        self.frcParams.f2i=dict(zip(['Economic Growth', 'Inflation', 'Confidence', ],
                                    ['ind_prod_total_idx', 'cpi_core_d2', 'cci', ]))
        self.frcParams.Nstatic=4
        self.frcParams.useExcessIndustries=True
        self.factorReturnsCalculator=MacroCalculators.MacroFactorReturnsCalculator_CoreSSM_bare(self.frcParams)

        self.dateFirstExposures=datetime.date(1995,1,1)
        self.dateFirstFactorReturns=datetime.date(1993,1,1)
        self.dateFloor=datetime.date(1988,1,1) #get macro time series data this far back
        self.dateCeiling=datetime.date(2999,12,31)
        self.datesForMacroCache=[self.dateFloor,datetime.date.today(),datetime.date.today()] #Probably too big.

        self._macroDataMaster=None #A struct to store things loaded only once.
        self._needsAssetReturns=False
        self._needsMacroData=False

        # Set up risk parameters
        ModelParameters.defaultFundamentalCovarianceParameters(
                self, nwLag=2, dva='spline', dwe=False)

        self.macroQAhelper=MacroDataUtils.US3MacroDataManager()

    def loadEstimationUniverse(self, rmi_id, modelDB, data=None):
        """Loads the estimation universe(s) of the given risk model instance.
        Returns a list of sub-issue IDs.
        """
        estu = modelDB.getRiskModelInstanceESTU(rmi_id)
        if data is not None:
            estu = [sid for sid in estu if sid in data.assetIdxMap.keys()]
            data.estimationUniverseIdx = [data.assetIdxMap[sid] for sid in estu]
        assert(len(estu) > 0)
        return estu

    def insertEstimationUniverse(self, rmi, assets, estuIdx, modelDB, qualify):
        """Inserts the estimation universes into the database for the given
        risk model instance.
        """
        estuAssets = [assets[idx] for idx in estuIdx]
        modelDB.insertEstimationUniverse(rmi, estuAssets, qualify)

    def insertEstimationUniverseWeights(self, rmi, subidWeightPairs, modelDB, estuName='main'):
        """Inserts model estimation (regression) weights for estimation
        universe assets in the given risk model instance.
        """
        modelDB.insertEstimationUniverseWeights(rmi, subidWeightPairs)

    def assetReturnHistoryLoader(self, data, returnHistory, modelDate, modelDB, marketDB):
        """ Function to load in returns for factor regression
        """
        # Load in history of returns
        returnsProcessor = LegacyProcessReturns.assetReturnsProcessor(
                self.rmg, data.universe, data.rmgAssetMap, data.tradingRmgAssetMap,
                data.assetTypeDict, data.marketTypeDict, numeraire_id=self.numeraire.currency_id,
                returnsTimingID=self.returnsTimingId)
        returnsHistory = returnsProcessor.process_returns_history(
                modelDate, int(returnHistory), modelDB, marketDB,
                drCurrMap=data.drCurrData, loadOnly=True, excludeWeekend=True,
                applyRT=(self.SCM==False))

        logging.info('Using older proxy returns fill-in')
        returnsHistory = Utilities.proxyMissingAssetReturnsLegacy(
                self.rmg, modelDate, returnsHistory, data, modelDB)
        return returnsHistory

class CAAxioma2009MH(MFM.SingleCountryFundamentalModel):
    """Production AX-CA Canada model
    """
    rm_id = 18
    revision = 3
    rms_id = 141
    styles = [
              ModelFactor('Value', 'Value'),
              ModelFactor('Leverage', 'Leverage'),
              ModelFactor('Growth', 'Growth'),
              ModelFactor('Size', 'Size'),
              ModelFactor('Market Sensitivity', 'Market Sensitivity'),
              ModelFactor('Liquidity','Liquidity'),
              ModelFactor('Short-Term Momentum', 'Short-Term Momentum'),
              ModelFactor('Medium-Term Momentum', 'Medium-Term Momentum'),
              ModelFactor('Volatility', 'Volatility'),
             ]
    industryClassification = Classification.GICSCustomCA(
            datetime.date(2008,8,30))
    quarterlyFundamentalData = True
    proxyDividendPayout = False
    newExposureFormat = True

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.CAAxioma2009MH')
        MFM.SingleCountryFundamentalModel.__init__(
                            self, ['CUSIP'], modelDB, marketDB)
        self.setCalculators(modelDB)
        modelDB.createCurrencyCache(marketDB)
        self.indexSelector = MarketIndex.MarketIndexSelector(modelDB, marketDB)

    def setCalculators(self, modelDB, overrider = False):
        # Set up regression parameters
        self.returnCalculator = \
                defaultRegressionParameters(\
                self, modelDB, dummyThreshold=6.0, overrider=overrider)
        # Set up risk parameters
        defaultFundamentalCovarianceParameters(self, dva='slowStep', overrider=overrider)

    def generate_estimation_universe(self, modelDate, exposureData,
                                     modelDB, marketDB):
        """Estimation universe selection criteria for AX-CA.
        """
        self.log.debug('generate_estimation_universe: begin')
        buildEstu = EstimationUniverse.ConstructEstimationUniverse(
                        exposureData.exposureMatrix.getAssets(), self, modelDB, marketDB)

        # Remove assets from the exclusion table
        (estu, nonest) = buildEstu.apply_exclusion_list(modelDate)

        # Remove some specific asset types
        (estu, nonest) = self.excludeAssetTypes(
                modelDate, exposureData, modelDB, marketDB, buildEstu, estu)

        # Remove foreign-domiciled issuers
        (estu, nonest) = buildEstu.exclude_by_market_classification(
                        modelDate, 'HomeCountry', 'REGIONS', ['CA'], baseEstu=estu)

        # Weed out foreign issuers by ISIN country prefix
        (estu, nonest)  = buildEstu.exclude_by_isin_country(['CA'],
                            modelDate, baseEstu=estu)

        # Keep only TSX listed stocks (excludes Venture board)
        (estu, nonest) = buildEstu.exclude_by_market_classification(
                        modelDate, 'Market', 'REGIONS', ['TSE'], baseEstu=estu)

        # Keep only common stocks, REITs, and possibly unit trusts
        (estu0, nonest) = buildEstu.exclude_by_market_classification(
                        modelDate, 'TQA FTID Domestic Asset Type', 'ASSET TYPES',
                        ['C','I','U'], baseEstu=estu)

        # Rank stuff by market cap and total volume over past year
        (estu, nonest) = buildEstu.filter_by_cap_and_volume(
                        exposureData, modelDate, baseEstu=estu0, hiCapQuota=250, loCapQuota=150,
                        bufferFactor=1.2)

        # Exclude assets with price below C$1 (S&P/TSX logic)
        (estu, nonest) = buildEstu.exclude_low_price_assets(
                        modelDate, baseEstu=estu, minPrice=1.0)

        # Inflate thin factors
        (estu, nonest) = buildEstu.pump_up_factors(
                        exposureData, modelDate, currentEstu=estu, baseEstu=estu0,
                        minFactorWidth=self.returnCalculator.\
                                parameters.getDummyWeights().minAssets)

        self.log.debug('generate_estimation_universe: end')
        return estu

class CAAxioma2009SH(CAAxioma2009MH):
    """Production AX-CA (Short-Term) model
    """
    rm_id = 49
    revision = 1
    rms_id = 70
    newExposureFormat = True

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.CAAxioma2009SH')
        MFM.SingleCountryFundamentalModel.__init__(
                            self, ['CUSIP'], modelDB, marketDB)
        self.setCalculators(modelDB)

    def setCalculators(self, modelDB, overrider = False):
        # Set up regression parameters
        self.returnCalculator = defaultRegressionParameters(
                                    self, modelDB, dummyThreshold=6.0, overrider=overrider)
        # Set up risk parameters
        defaultFundamentalCovarianceParameters(self, modelHorizon='short', dva='slowStep', overrider=overrider)

class CAAxioma2009MH_S(MFM.StatisticalFactorModel):
    """Production AX-CA-S Canada model
    """
    rm_id = 19
    revision = 3
    rms_id = 142
    numFactors = 15
    blind = [ModelFactor('Statistical Factor %d' % n,
        'Statistical Factor %d' % n)
        for n in range(1, numFactors+1)]
    returnHistory = 250
    industryClassification = Classification.GICSCustomCA(
            datetime.date(2008,8,30))
    newExposureFormat = True

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.CAAxioma2009MH_S')
        MFM.StatisticalFactorModel.__init__(
                            self, ['CUSIP'], modelDB, marketDB)
        self.setCalculators(modelDB)
        self.baseModel = CAAxioma2009MH(modelDB, marketDB)

    def setCalculators(self, modelDB, overrider = False):
        self.returnCalculator = ReturnCalculator. \
                            AsymptoticPrincipalComponents2(self.numFactors)
        # Set up risk parameters
        defaultStatisticalCovarianceParameters(self, dva='slowStep',
                longSpecificReturnHistory=False, overrider=overrider)

class CAAxioma2009SH_S(CAAxioma2009MH_S):
    """Production AX-CA-S (Short-Term) statistical model
    """
    rm_id = 50
    revision = 1
    rms_id = 71
    newExposureFormat = True

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.CAAxioma2009SH_S')
        MFM.StatisticalFactorModel.__init__(
                            self, ['CUSIP'], modelDB, marketDB)
        self.setCalculators(modelDB)
        self.baseModel = CAAxioma2009MH(modelDB, marketDB)

    def setCalculators(self, modelDB, overrider = False):
        self.returnCalculator = ReturnCalculator. \
                                AsymptoticPrincipalComponents2(self.numFactors)
        # Set up risk parameters
        defaultStatisticalCovarianceParameters(
                            self, modelHorizon='short',
                            dva='slowStep', longSpecificReturnHistory=False, overrider=overrider)

class AUAxioma2009MH(MFM.SingleCountryFundamentalModel):
    """Production AX-AU Australia model
    """
    rm_id = 24
    revision = 3
    rms_id = 139
    k  = 1.345
    styles = [
              ModelFactor('Value', 'Value'),
              ModelFactor('Leverage', 'Leverage'),
              ModelFactor('Growth', 'Growth'),
              ModelFactor('Size', 'Size'),
              ModelFactor('Market Sensitivity', 'Market Sensitivity'),
              ModelFactor('Liquidity','Liquidity'),
              ModelFactor('Short-Term Momentum', 'Short-Term Momentum'),
              ModelFactor('Medium-Term Momentum', 'Medium-Term Momentum'),
              ModelFactor('Volatility', 'Volatility'),
             ]
    industryClassification = Classification.GICSCustomAU(
            datetime.date(2008,8,30))
    newExposureFormat = True

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.AUAxioma2009MH')
        MFM.SingleCountryFundamentalModel.__init__(
                            self, ['SEDOL'], modelDB, marketDB)
        # from IPython import embed; embed(header='1');
        self.setCalculators(modelDB)
        modelDB.createCurrencyCache(marketDB)
        self.indexSelector = MarketIndex.MarketIndexSelector(modelDB, marketDB)
    def setCalculators(self, modelDB, overrider = False):
        # Set up regression parameters
        self.returnCalculator = \
                defaultRegressionParameters(\
                self, modelDB, dummyThreshold=6.0,k_rlm=self.k, overrider=overrider)
        # Set up risk parameters
        defaultFundamentalCovarianceParameters(self, nwLag=3, dva='slowStep', overrider=overrider)

    def generate_estimation_universe(self, modelDate, exposureData,
                                     modelDB, marketDB):
        """Estimation universe selection criteria for AX-AU.
        """
        self.log.debug('generate_estimation_universe: begin')
        buildEstu = EstimationUniverse.ConstructEstimationUniverse(
                        exposureData.exposureMatrix.getAssets(), self, modelDB, marketDB)

        # Remove assets from the exclusion table
        (estu, nonest) = buildEstu.apply_exclusion_list(modelDate)

        # Remove some specific asset types
        (estu, nonest) = self.excludeAssetTypes(
                modelDate, exposureData, modelDB, marketDB, buildEstu, estu)

        # Remove foreign-domiciled issuers
        (estu, nonest) = buildEstu.exclude_by_market_classification(
                        modelDate, 'HomeCountry', 'REGIONS', ['AU'], baseEstu=estu)

        # Weed out foreign issuers by ISIN country prefix
        (estu, nonest)  = buildEstu.exclude_by_isin_country(['AU'],
                            modelDate, baseEstu=estu)

        # Keep only equities
        (estu0, nonest) = buildEstu.exclude_by_market_classification(
                        modelDate, 'TQA FTID Global Asset Type', 'ASSET TYPES',
                        ['10','09'], baseEstu=estu)

        # Remove possible Investment Trusts, ETFs, various funds, etc.
        (funds, tmp) = buildEstu.exclude_by_market_classification(
                        modelDate, 'DataStream2 Asset Type', 'ASSET TYPES',
                        ['CF', 'CEF'], baseEstu=estu, keepMissing=False)
        estu0 = list(set(estu0).difference(funds))

        # Attempt to identify Infrastructure Funds and REITs
        (composites, tmp) = buildEstu.exclude_by_market_classification(
                        modelDate, 'TQA FTID Global Asset Type', 'ASSET TYPES',
                        ['54'], baseEstu=estu, keepMissing=False)
        expM = exposureData.exposureMatrix
        fIdx = [expM.getFactorIndex(f) for f in (
                'Real Estate Investment Trusts (REITs)',
                'Transportation', 'Utilities', 'Media')]
        exposures = ma.take(expM.getMatrix(), fIdx, axis=0)
        indices = numpy.flatnonzero(ma.getmaskarray(ma.sum(exposures, axis=0))==0)
        indices = list(set(composites).intersection(indices))
        self.log.info('Found %d possible Infrastructure Funds and REITs' \
                % len(indices))
        estu0.extend(indices)

        # Rank stuff by market cap and total volume over past year
        (estu, nonest) = buildEstu.filter_by_cap_and_volume(
                        exposureData, modelDate, baseEstu=estu0, hiCapQuota=200, loCapQuota=100,
                        bufferFactor=1.2)

        # Inflate thin factors
        (estu, nonest) = buildEstu.pump_up_factors(
                        exposureData, modelDate, currentEstu=estu, baseEstu=estu0,
                        minFactorWidth=self.returnCalculator.\
                                parameters.getDummyWeights().minAssets, cutOff=0.05)

        self.log.debug('generate_estimation_universe: end')
        return estu

class AUAxioma2009SH(AUAxioma2009MH):
    """Production AX-AU-S Australia model
    """
    rm_id = 51
    revision = 1
    rms_id = 72
    newExposureFormat = True

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.AUAxioma2009SH')
        MFM.SingleCountryFundamentalModel.__init__(
                            self, ['SEDOL'], modelDB, marketDB)
        self.setCalculators(modelDB)

    def setCalculators(self, modelDB, overrider = False):
        # Set up regression parameters
        self.returnCalculator = \
                defaultRegressionParameters(\
                self, modelDB, dummyThreshold=6.0, overrider=overrider)
        # Set up risk parameters
        defaultFundamentalCovarianceParameters(
                self, modelHorizon='short', nwLag=3, dva='slowStep', overrider=overrider)

class AUAxioma2009MH_S(MFM.StatisticalFactorModel):
    """Production AX-AU-S Australia model
    """
    rm_id = 25
    revision = 3
    rms_id = 140
    numFactors = 15
    blind = [ModelFactor('Statistical Factor %d' % n,
        'Statistical Factor %d' % n)
        for n in range(1, numFactors+1)]
    returnHistory = 250
    industryClassification = Classification.GICSCustomAU(
            datetime.date(2008,8,30))
    newExposureFormat = True

    def __init__(self, modelDB, marketDB):

        self.log = logging.getLogger('RiskModels.AUAxioma2009MH_S')
        MFM.StatisticalFactorModel.__init__(
                            self, ['SEDOL'], modelDB, marketDB)
        self.setCalculators(modelDB)
        self.baseModel = AUAxioma2009MH(modelDB, marketDB)

    def setCalculators(self, modelDB, overrider = False):
        self.returnCalculator = ReturnCalculator. \
                            AsymptoticPrincipalComponents2(self.numFactors)
        # Set up risk parameters
        defaultStatisticalCovarianceParameters(\
                self, dva='slowStep', longSpecificReturnHistory=False, overrider=overrider)

class AUAxioma2009SH_S(AUAxioma2009MH_S):
    """Production AX-AU-S (Short Horizon) Australia model
    """
    rm_id = 52
    revision = 1
    rms_id = 73
    newExposureFormat = True

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.AUAxioma2009SH_S')
        MFM.StatisticalFactorModel.__init__(
                            self, ['SEDOL'], modelDB, marketDB)
        self.setCalculators(modelDB)
        self.baseModel = AUAxioma2009MH(modelDB, marketDB)

    def setCalculators(self, modelDB, overrider = False):
        self.returnCalculator = ReturnCalculator. \
                            AsymptoticPrincipalComponents2(self.numFactors)
        # Set up risk parameters
        defaultStatisticalCovarianceParameters(\
                self, modelHorizon='short', dva='slowStep', longSpecificReturnHistory=False, overrider=overrider)

class GBAxioma2009MH(MFM.SingleCountryFundamentalModel):
    """Production AX-GB United Kingom model
    """
    rm_id = 28
    revision = 2
    rms_id = 137
    styles = [
              ModelFactor('Value', 'Value'),
              ModelFactor('Leverage', 'Leverage'),
              ModelFactor('Growth', 'Growth'),
              ModelFactor('Size', 'Size'),
              ModelFactor('Market Sensitivity', 'Market Sensitivity'),
              ModelFactor('Liquidity','Liquidity'),
              ModelFactor('Short-Term Momentum', 'Short-Term Momentum'),
              ModelFactor('Medium-Term Momentum', 'Medium-Term Momentum'),
              ModelFactor('Volatility', 'Volatility'),
              ModelFactor('Exchange Rate Sensitivity', 'Exchange Rate Sensitivity'),
              ModelFactor('Investment Trusts', 'Investment Trusts'),
             ]
    industryClassification = Classification.GICSCustomGB(
            datetime.date(2008,8,30))
    newExposureFormat = True

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.GBAxioma2009MH')
        MFM.SingleCountryFundamentalModel.__init__(
                            self, ['SEDOL'], modelDB, marketDB)
        self.setCalculators(modelDB)
        modelDB.createCurrencyCache(marketDB)
        self.indexSelector = MarketIndex.MarketIndexSelector(modelDB, marketDB)

    def setCalculators(self, modelDB, overrider = False):
        # Set up regression parameters
        self.returnCalculator = \
                defaultRegressionParameters(\
                self, modelDB, dummyThreshold=6.0, overrider=overrider)
        self.zeroExposureNames = ['Value', 'Leverage', 'Growth']
        self.zeroExposureTypes = ['Investment Trusts'] * 3
        # Set up risk parameters
        defaultFundamentalCovarianceParameters(self, nwLag=2, dva='slowStep', overrider=overrider)

    def generate_estimation_universe(self, modelDate, exposureData,
                                     modelDB, marketDB):
        """Estimation universe selection criteria for AX-GB.
        """
        self.log.debug('generate_estimation_universe: begin')
        buildEstu = EstimationUniverse.ConstructEstimationUniverse(
                        exposureData.exposureMatrix.getAssets(), self, modelDB, marketDB)

        # Remove assets from the exclusion table
        (estu, nonest) = buildEstu.apply_exclusion_list(modelDate)

        # Remove some specific asset types
        (estu, nonest) = self.excludeAssetTypes(
                modelDate, exposureData, modelDB, marketDB, buildEstu, estu)

        # Remove GDRs -- note the country code, GB not UK
        (estu, nonest) = buildEstu.exclude_by_market_classification(
                        modelDate, 'HomeCountry', 'REGIONS', ['GB'], baseEstu=estu)

        # Weed out foreign issuers by ISIN country prefix
        (estu, nonest)  = buildEstu.exclude_by_isin_country(['GB'],
                            modelDate, baseEstu=estu)

        # Keep only equities and investment trusts (exclude mutual funds, DRs)
        (estu, nonest) = buildEstu.exclude_by_market_classification(
                        modelDate, 'DataStream2 Asset Type', 'ASSET TYPES', ['EQ','INVT'],
                        baseEstu=estu)

        # Exclude assets with price below 0.0001 GBP (weird FTSE business logic)
        (estu0, nonest) = buildEstu.exclude_low_price_assets(
                        modelDate, baseEstu=estu, minPrice=0.0001)

        # Dynamically determine size of small-cap segment
        mcaps = ma.take(exposureData.marketCaps, estu0)
        rank = ma.argsort(-mcaps)
        small_prc = ma.sum(ma.take(mcaps, rank[350:650])) / ma.sum(mcaps)
        nAssets = round(0.9 * 1e4 * small_prc)
        nAssets = int(max(min(nAssets, 550), 275))

        # Rank stuff by market cap and total volume over past year
        (estu, nonest) = buildEstu.filter_by_cap_and_volume(
                        exposureData, modelDate, baseEstu=estu0, hiCapQuota=350, loCapQuota=nAssets,
                        bufferFactor=1.2)

        # Inflate thin factors
        (estu, nonest) = buildEstu.pump_up_factors(
                        exposureData, modelDate, currentEstu=estu, baseEstu=estu0,
                        minFactorWidth=self.returnCalculator.\
                                parameters.getDummyWeights().minAssets)

        self.log.debug('generate_estimation_universe: end')
        return estu

    def identify_investment_trusts(self, data, modelDate, modelDB, marketDB):
        """Returns array positions corresponding to assets that
        are investment trusts.
        """
        self.log.debug('identify_investment_trusts: begin')
        e = EstimationUniverse.ConstructEstimationUniverse(
                        data.universe, self, modelDB, marketDB)
        (indices, other) = e.exclude_by_market_classification(
                        modelDate, 'DataStream2 Asset Type', 'ASSET TYPES', ['INVT'],
                        keepMissing=False)
        mcap = ma.sum(numpy.take(data.marketCaps, indices, axis=0))
        values = Matrices.allMasked(len(data.universe))
        if len(indices) > 0:
            ma.put(values, indices, 1.0)
        self.log.info('Found %d Investment Trusts worth %.2f bn %s'
                % (len(indices), mcap / 1e9, self.numeraire.currency_code))
        self.log.debug('identify_investment_trusts: end')
        return values

    def generate_model_specific_exposures(self, modelDate, data, modelDB, marketDB):
        """Generate the Exchange Rate Sensitivity and
        Investment Trusts factors
        """
        data.exposureMatrix.addFactor(
            'Investment Trusts', self.identify_investment_trusts(
            data, modelDate, modelDB, marketDB), ExposureMatrix.StyleFactor)
        data.exposureMatrix.addFactor(
            'Exchange Rate Sensitivity', StyleExposures.generate_forex_sensitivity(
            data.returns, self, modelDB, 120, 'XDR'), ExposureMatrix.StyleFactor)
        return data.exposureMatrix

class GBAxioma2009SH(GBAxioma2009MH):
    """Production AX-GB-SH United Kingom model
    """
    rm_id = 53
    revision = 1
    rms_id = 74
    newExposureFormat = True

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.GBAxioma2009SH')
        MFM.SingleCountryFundamentalModel.__init__(
                            self, ['SEDOL'], modelDB, marketDB)
        self.setCalculators(modelDB)

    def setCalculators(self, modelDB, overrider = False):
        # Set up regression parameters
        self.returnCalculator = \
                defaultRegressionParameters(\
                self, modelDB, dummyThreshold=6.0, overrider=overrider)
        self.zeroExposureNames = ['Value', 'Leverage', 'Growth']
        self.zeroExposureTypes = ['Investment Trusts'] * 3
        # Set up risk parameters
        defaultFundamentalCovarianceParameters(
                    self, modelHorizon='short', nwLag=2, dva='slowStep', overrider=overrider)

class GBAxioma2009MH_S(MFM.StatisticalFactorModel):
    """Production AX-GB-S United Kingdom model
    """
    rm_id = 29
    revision = 2
    rms_id = 138
    numFactors = 15
    blind = [ModelFactor('Statistical Factor %d' % n,
        'Statistical Factor %d' % n)
        for n in range(1, numFactors+1)]
    returnHistory = 250
    industryClassification = Classification.GICSCustomGB(
            datetime.date(2008,8,30))
    newExposureFormat = True

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.GBAxioma2009MH_S')
        MFM.StatisticalFactorModel.__init__(
                            self, ['SEDOL'], modelDB, marketDB)
        self.setCalculators(modelDB)
        self.baseModel = GBAxioma2009MH(modelDB, marketDB)

    def setCalculators(self, modelDB, overrider = False):
        self.returnCalculator = ReturnCalculator. \
                            AsymptoticPrincipalComponents2(self.numFactors)
        # Set up risk parameters
        defaultStatisticalCovarianceParameters(\
                self, dva='slowStep', longSpecificReturnHistory=False, overrider=overrider)

class GBAxioma2009SH_S(GBAxioma2009MH_S):
    """Production AX-GB-SH-S United Kingdom model
    """
    rm_id = 54
    revision = 1
    rms_id = 75
    newExposureFormat = True

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.GBAxioma2009SH_S')
        MFM.StatisticalFactorModel.__init__(
                            self, ['SEDOL'], modelDB, marketDB)
        self.setCalculators(modelDB)
        self.baseModel = GBAxioma2009MH(modelDB, marketDB)

    def setCalculators(self, modelDB, overrider = False):
        self.returnCalculator = ReturnCalculator. \
                            AsymptoticPrincipalComponents2(self.numFactors)
        # Set up risk parameters
        defaultStatisticalCovarianceParameters(\
                self, modelHorizon='short', dva='slowStep', longSpecificReturnHistory=False, overrider=overrider)

class JPAxioma2009MH(MFM.SingleCountryFundamentalModel):
    """Production AX-JP2 Japan model
    """
    rm_id = 26
    revision = 2
    rms_id = 135
    styles = [
              ModelFactor('Value', 'Value'),
              ModelFactor('Leverage', 'Leverage'),
              ModelFactor('Growth', 'Growth'),
              ModelFactor('Size', 'Size'),
              ModelFactor('Market Sensitivity', 'Market Sensitivity'),
              ModelFactor('Liquidity','Liquidity'),
              ModelFactor('Short-Term Momentum', 'Short-Term Momentum'),
              ModelFactor('Medium-Term Momentum', 'Medium-Term Momentum'),
              ModelFactor('Volatility', 'Volatility'),
              ModelFactor('Exchange Rate Sensitivity', 'Exchange Rate Sensitivity'),
              ModelFactor('Mid Market', 'Mid Market'),
             ]
    industryClassification = Classification.GICSCustomJP(datetime.date(2008,8,30))
    newExposureFormat = True

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
                defaultRegressionParameters(\
                self, modelDB, dummyThreshold=6.0, overrider=overrider)
        # Set up risk parameters
        defaultFundamentalCovarianceParameters(self, nwLag=4, dva='slowStep', overrider=overrider)

    def generate_estimation_universe(self, modelDate, exposureData, modelDB, marketDB):
        """AX-JP-2 estimation universe selection.
        Attempts to replicate constituents of TOPIX, plus some REITs.
        """
        self.log.debug('generate_estimation_universe: begin')
        mcaps = exposureData.marketCaps
        estu = self.topixReplicator.generate_topix_universe(
                            exposureData.universe, mcaps, modelDate,
                            modelDB, marketDB)
        # Add REITs
        if modelDate >= datetime.date(2001,9,11):
            expM = exposureData.exposureMatrix
            fidx = expM.getFactorIndex('Real Estate Investment Trusts (REITs)')
            exposures = ma.take(expM.getMatrix(), [fidx], axis=0)
            reit_mcap = exposures * mcaps
            threshold = Utilities.prctile(mcaps, [75])[0]
            reit_mcap = ma.masked_where(reit_mcap < threshold, reit_mcap)
            indices = numpy.flatnonzero(ma.getmaskarray(reit_mcap)==0)
            if len(indices) > 0:
                estu.extend(indices.tolist())
            self.log.info('Added %d REITs to ESTU above %.2f bn %s'
                    % (len(indices), threshold / 1e9, self.numeraire.currency_code))

        self.log.debug("generate estimation universe: end")
        return estu

    def generate_model_specific_exposures(self, modelDate, data, modelDB, marketDB):
        """Generate the Exchange Rate Sensitivity and Mid Market factors
        """
        expM = data.exposureMatrix
        # Note that we use lagged FX data here, unlike other models
        expM.addFactor(
            'Exchange Rate Sensitivity', StyleExposures.generate_forex_sensitivity(
            data.returns, self, modelDB, 120, lag=1, swAdj=True),
            ExposureMatrix.StyleFactor)

        # Generate TOPIX 400 (midcap) membership proxy
        if modelDate >= datetime.date(2001,9,11):
            expM = data.exposureMatrix
            fidx = expM.getFactorIndex('Real Estate Investment Trusts (REITs)')
            reit_indices = numpy.flatnonzero(ma.getmaskarray(
                            expM.getMatrix()[fidx,:])==0)
            reit_indices = set(reit_indices)
        else:
            reit_indices = set()
        baseUniv = [data.universe[i] for i in data.estimationUniverseIdx \
                    if i not in reit_indices]
        self.topixReplicator.baseUniverse = baseUniv
        mid400 = self.topixReplicator.replicate_topix_subindex(
                        self.topixReplicator.TOPIXMid400,
                        modelDate, data.universe, data.marketCaps,
                        modelDB, marketDB)
        values = Matrices.allMasked(len(data.universe))
        ma.put(values, mid400, 1.0)
        expM.addFactor(
            'Mid Market', values, ExposureMatrix.StyleFactor)
        return expM

class JPAxioma2009SH(JPAxioma2009MH):
    """Production AX-JP2-SH Japan model
    """
    rm_id = 55
    revision = 1
    rms_id = 76
    newExposureFormat = True

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.JPAxioma2009SH')
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
                defaultRegressionParameters(\
                self, modelDB, dummyThreshold=6.0, overrider=overrider)
        # Set up risk parameters
        defaultFundamentalCovarianceParameters(self, modelHorizon='short', nwLag=4, dva='slowStep', overrider=overrider)

class JPAxioma2009MH_S(MFM.StatisticalFactorModel):
    """Production AX-JP2-S United Kingdom model
    """
    rm_id = 27
    revision = 2
    rms_id = 136
    numFactors = 15
    blind = [ModelFactor('Statistical Factor %d' % n,
        'Statistical Factor %d' % n)
        for n in range(1, numFactors+1)]
    returnHistory = 250
    industryClassification = Classification.GICSCustomJP(datetime.date(2008,8,30))
    newExposureFormat = True

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.JPAxioma2009MH_S')
        MFM.StatisticalFactorModel.__init__(
                            self, ['SEDOL'], modelDB, marketDB)
        self.setCalculators(modelDB)
        self.baseModel = JPAxioma2009MH(modelDB, marketDB)

    def setCalculators(self, modelDB, overrider = False):
        self.returnCalculator = ReturnCalculator. \
                            AsymptoticPrincipalComponents2(self.numFactors)
        # Set up risk parameters
        defaultStatisticalCovarianceParameters(\
                self, dva='slowStep', longSpecificReturnHistory=False, overrider=overrider)

class JPAxioma2009SH_S(JPAxioma2009MH_S):
    """Production AX-JP2-SH-S United Kingdom model
    """
    rm_id = 56
    revision = 1
    rms_id = 77
    newExposureFormat = True

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.JPAxioma2009SH_S')
        MFM.StatisticalFactorModel.__init__(
                            self, ['SEDOL'], modelDB, marketDB)
        self.setCalculators(modelDB)
        self.baseModel = JPAxioma2009MH(modelDB, marketDB)

    def setCalculators(self, modelDB, overrider = False):
        self.returnCalculator = ReturnCalculator. \
                            AsymptoticPrincipalComponents2(self.numFactors)
        # Set up risk parameters
        defaultStatisticalCovarianceParameters(\
                self, modelHorizon='short', dva='slowStep', longSpecificReturnHistory=False, overrider=overrider)

class TWAxioma2012MH(MFM.SingleCountryFundamentalModel):
    """Production AX-TW Taiwan model.
    """
    rm_id = 90
    revision = 1
    rms_id = 153
    newExposureFormat = True
    interceptFactor = None
    styles = [  CompositeFactor('Value', 'Value'),
                CompositeFactor('Leverage', 'Leverage'),
                CompositeFactor('Growth', 'Growth'),
                ModelFactor('Size', 'Size'),
                ModelFactor('Liquidity','Liquidity'),
                ModelFactor('Short-Term Momentum', 'Short-Term Momentum'),
                ModelFactor('Medium-Term Momentum', 'Medium-Term Momentum'),
                ModelFactor('Exchange Rate Sensitivity', 'Exchange Rate Sensitivity'),
                ModelFactor('Volatility', 'Volatility'),
                ]
    industryClassification = Classification.GICSCustomTW(
        datetime.date(2008,8,30))
    quarterlyFundamentalData = False

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.TWAxioma2012MH')
        MFM.SingleCountryFundamentalModel.__init__(
            self, ['SEDOL','Ticker'], modelDB, marketDB)
        self.setCalculators(modelDB)
        modelDB.createCurrencyCache(marketDB)
        self.indexSelector = MarketIndex.MarketIndexSelector(modelDB, marketDB)

    def setCalculators(self, modelDB, overrider = False):
        # Set up regression parameters
        self.returnCalculator = \
                defaultRegressionParameters(\
                self, modelDB, dummyThreshold=6.0,
                overrider=overrider)
        # Set up risk parameters
        defaultFundamentalCovarianceParameters(self, nwLag=3, overrider=overrider)

    def generate_estimation_universe(self, modelDate, exposureData,
                                     modelDB, marketDB):
        """Estimation universe selection criteria for AX-TW.
        """
        self.log.debug('generate_estimation_universe: begin')
        buildEstu = EstimationUniverse.ConstructEstimationUniverse(
                        exposureData.exposureMatrix.getAssets(), self, modelDB, marketDB)

        # Remove assets from the exclusion table
        (estu, nonest) = buildEstu.apply_exclusion_list(modelDate)

        # Remove some specific asset types
        (estu, nonest) = self.excludeAssetTypes(
                modelDate, exposureData, modelDB, marketDB, buildEstu, estu)

        # Remove foreign-domiciled issuers
        (estu, nonest) = buildEstu.exclude_by_market_classification(
                        modelDate, 'HomeCountry', 'REGIONS', ['TW'], baseEstu=estu)

        # Weed out foreign issuers by ISIN country prefix
        (estu, nonest)  = buildEstu.exclude_by_isin_country(['TW'],
                            modelDate, baseEstu=estu)

        # Throw out OTC Emerging stocks
        (otc_emerging, nonest) = buildEstu.exclude_by_market_classification(
                        modelDate, 'Market', 'REGIONS', ['REG'], baseEstu=estu,
                        keepMissing=False)

        # Flag and exclude ETFs and TDRs
        (etf_tdr0, nonest) = buildEstu.exclude_by_market_classification(
                        modelDate, 'TQA FTID Global Asset Type', 'ASSET TYPES',
                        ['MktLbl_6','6C','5G'], baseEstu=estu, keepMissing=False)

        # Flag and exclude more ETFs and TDRs
        (etf_tdr1, nonest) = buildEstu.exclude_by_market_classification(
                        modelDate, 'DataStream2 Asset Type', 'ASSET TYPES',
                        ['GDR','ET','ETF','CF','CEF'], baseEstu=estu, keepMissing=False)

        # Make use of the Axioma Sec Type
        (etf_tdr2, nonest) = buildEstu.exclude_by_market_classification(
                        modelDate, 'Axioma Asset Type', 'ASSET TYPES',
                        ['TDR','ComETF', 'NonEqETF'], baseEstu=estu, keepMissing=False)

        estu0 = list(set(estu).difference(etf_tdr0 + etf_tdr1 + etf_tdr2 + otc_emerging))

        # Rank stuff by market cap and total volume over past year
        (estu, nonest) = buildEstu.filter_by_cap_and_volume(
                        exposureData, modelDate, baseEstu=estu0, hiCapQuota=150, loCapQuota=250,
                        bufferFactor=1.2)

        # Inflate thin factors
        (estu, nonest) = buildEstu.pump_up_factors(
                        exposureData, modelDate, currentEstu=estu, baseEstu=estu0,
                        minFactorWidth=6.0, cutOff=0.05)

        self.log.debug('generate_estimation_universe: end')
        return estu

    def generate_model_specific_exposures(self, modelDate, data, modelDB, marketDB):
        """Generate the Exchange Rate Sensitivity factor.
        """
        data.exposureMatrix.addFactor(
            'Exchange Rate Sensitivity', StyleExposures.generate_forex_sensitivity_v2(
                data.returns, self, modelDB, 120, fixDate=self.modelHack.xrtFixDate),
                ExposureMatrix.StyleFactor)
        return data.exposureMatrix

class TWAxioma2012SH(TWAxioma2012MH):
    """Production AX-TW-SH Taiwan model
    """
    rm_id = 91
    revision = 1
    rms_id = 154

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.TWAxioma2012SH')
        MFM.SingleCountryFundamentalModel.__init__(
            self, ['SEDOL','Ticker'], modelDB, marketDB)
        self.setCalculators(modelDB)

    def setCalculators(self, modelDB, overrider = False):
        # Set up regression parameters
        self.returnCalculator = \
                defaultRegressionParameters(\
            self, modelDB, dummyThreshold=6.0, overrider=overrider)
        # Set up risk parameters
        defaultFundamentalCovarianceParameters(self, modelHorizon='short', nwLag=3, overrider=overrider)

class TWAxioma2012MH_S(MFM.StatisticalFactorModel):
    """Production AX-TW-S Taiwan model
    """
    rm_id = 42
    revision = 2
    rms_id = 155
    numFactors = 15
    blind = [ModelFactor('Statistical Factor %d' % n,
        'Statistical Factor %d' % n)
        for n in range(1, numFactors+1)]
    returnHistory = 250
    industryClassification = Classification.GICSCustomTW(
            datetime.date(2008,8,30))
    newExposureFormat = True

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.TWAxioma2012MH_S')
        MFM.StatisticalFactorModel.__init__(
                            self, ['SEDOL','Ticker'], modelDB, marketDB)
        self.setCalculators(modelDB)
        self.baseModel = TWAxioma2012MH(modelDB, marketDB)

    def setCalculators(self, modelDB, overrider = False):
        self.returnCalculator = ReturnCalculator. \
                            AsymptoticPrincipalComponents2(self.numFactors)
        # Set up risk parameters
        defaultStatisticalCovarianceParameters(self,
                scaleMinObs=0.9, longSpecificReturnHistory=False, overrider=overrider)

class TWAxioma2012SH_S(TWAxioma2012MH_S):
    """Production AXTW-SH-S Taiwan Model
    """
    rm_id = 92
    revision = 1
    rms_id = 156

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.TWAxioma2012SH_S')
        MFM.StatisticalFactorModel.__init__(
                            self, ['SEDOL','Ticker'], modelDB, marketDB)
        self.setCalculators(modelDB)
        self.baseModel = TWAxioma2012MH(modelDB, marketDB)

    def setCalculators(self, modelDB, overrider = False):
        self.returnCalculator = ReturnCalculator. \
                            AsymptoticPrincipalComponents2(self.numFactors)
        # Set up risk parameters
        defaultStatisticalCovarianceParameters(self, modelHorizon='short',
                scaleMinObs=0.9, longSpecificReturnHistory=False, overrider=overrider)

class CNAxioma2010MH(MFM.SingleCountryFundamentalModel):
    """Production AX-CN China model
    """
    rm_id = 40
    revision = 2
    rms_id = 143
    styles = [
              ModelFactor('Value', 'Value'),
              ModelFactor('Leverage', 'Leverage'),
              ModelFactor('Growth', 'Growth'),
              ModelFactor('Size', 'Size'),
              ModelFactor('Liquidity','Liquidity'),
              ModelFactor('Short-Term Momentum', 'Short-Term Momentum'),
              ModelFactor('Medium-Term Momentum', 'Medium-Term Momentum'),
              ModelFactor('Volatility', 'Volatility'),
              ModelFactor('Exchange Rate Sensitivity', 'Exchange Rate Sensitivity'),
              ModelFactor('B-Share Market', 'B-Share Market'),
             ]
    industryClassification = Classification.GICSCustomCN(datetime.date(2008,8,30))
    newExposureFormat = True

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.CNAxioma2010MH')
        MFM.SingleCountryFundamentalModel.__init__(
                            self, ['SEDOL'], modelDB, marketDB)
        self.setCalculators(modelDB)
        modelDB.createCurrencyCache(marketDB)
        self.indexSelector = MarketIndex.MarketIndexSelector(modelDB, marketDB)

    def setCalculators(self, modelDB, overrider = False):
        # Set up regression parameters
        self.returnCalculator = \
                defaultRegressionParameters(\
                self, modelDB, dummyThreshold=6.0, overrider=overrider)
        # Set up risk parameters
        defaultFundamentalCovarianceParameters(self, nwLag=3, overrider=overrider)

    def generate_estimation_universe(self, modelDate, exposureData,
                                     modelDB, marketDB):
        """Estimation universe selection criteria for AX-CN.
        """
        self.log.debug('generate_estimation_universe: begin')
        buildEstu = EstimationUniverse.ConstructEstimationUniverse(
                        exposureData.exposureMatrix.getAssets(), self, modelDB, marketDB)
        assetIdxMap = dict(zip(exposureData.universe, range(len(exposureData.universe))))

        # Remove assets from the exclusion table
        logging.info('Dropping assets in the exclusion table')
        (estu, nonest) = buildEstu.apply_exclusion_list(modelDate)
        self.checkTrackedAssets(exposureData.universe, assetIdxMap, estu)

        # Remove some specific asset types
        logging.info('Excluding particular asset types')
        (estu, nonest) = self.excludeAssetTypes(
                modelDate, exposureData, modelDB, marketDB, buildEstu, estu)
        self.checkTrackedAssets(exposureData.universe, assetIdxMap, estu)

        # Remove duplicates on connect exchanges
        exchangeCodes = ['SHC','SZC']
        logging.info('Excluding assets on exchanges %s', ','.join(exchangeCodes))
        (otc, tmp) = buildEstu.exclude_by_market_classification(
                modelDate, 'Market', 'REGIONS', exchangeCodes,
                baseEstu=list(estu), keepMissing=False)
        estu = list(set(estu).difference(otc))
        self.checkTrackedAssets(exposureData.universe, assetIdxMap, estu)

        # Remove foreign-domiciled issuers
        logging.info('Excluding foreign listings')
        (estu, nonest) = buildEstu.exclude_by_market_classification(
                        modelDate, 'HomeCountry', 'REGIONS', ['CN'], baseEstu=estu)
        self.checkTrackedAssets(exposureData.universe, assetIdxMap, estu)

        # One more safeguard to weed out ETFs
        logging.info('Excluding ETFs')
        (etf1, other) = buildEstu.exclude_by_market_classification(
                modelDate, 'DataStream2 Asset Type', 'ASSET TYPES', ['ET','ETF'], baseEstu=estu)
        (etf2, other) = buildEstu.exclude_by_market_classification(
                modelDate, 'TQA FTID Global Asset Type', 'ASSET TYPES', ['6C'], baseEstu=estu)
        estu = list(set(estu).difference(etf1 + etf2))
        self.log.info('Removed an additional %d ETFs', len(etf1 + etf2))
        self.checkTrackedAssets(exposureData.universe, assetIdxMap, estu)

        # Weed out foreign issuers by ISIN country prefix
        logging.info('Excluding foreign listings by ISIN')
        (estu0, nonest)  = buildEstu.exclude_by_isin_country(['CN'],
                            modelDate, baseEstu=estu)
        self.checkTrackedAssets(exposureData.universe, assetIdxMap, estu0)

        # Rank stuff by market cap and total volume over past year
        logging.info('Filtering by mcap and volume')
        (estu, nonest) = buildEstu.filter_by_cap_and_volume(
                        exposureData, modelDate, baseEstu=estu0, hiCapQuota=300, loCapQuota=500,
                        bufferFactor=1.2)
        self.checkTrackedAssets(exposureData.universe, assetIdxMap, estu)

        # Force in a few B-Shares, in case they are not present yet
        (aShares, bShares, hShares, other) = MarketIndex.\
            process_china_share_classes(exposureData, modelDate,
                                        modelDB, marketDB, factorName='')
        indices_b = numpy.argsort(-1.0 * numpy.take(
                        exposureData.marketCaps, bShares, axis=0))[:35]
        indices_b = [bShares[i] for i in indices_b]
        estu = list(set(estu).union(indices_b))
        logging.info('Added additional %d B-shares to ESTU',
                    len(estu) - (len(exposureData.universe) - len(nonest)))
        self.checkTrackedAssets(exposureData.universe, assetIdxMap, estu)

        # Reporting on B-share component
        bMCap = ma.take(exposureData.marketCaps, indices_b, axis=0)
        estuMCap = ma.take(exposureData.marketCaps, estu, axis=0)
        propn = ma.sum(bMCap, axis=None) / ma.sum(estuMCap, axis=None)
        logging.info('B-shares account for %.2f%% of total estimation universe mcap',
                100.0 * propn)

        # Inflate thin factors
        logging.info('Inflating thin factors')
        (estu, nonest) = buildEstu.pump_up_factors(
                        exposureData, modelDate, currentEstu=estu, baseEstu=estu0,
                        minFactorWidth=self.returnCalculator.\
                                parameters.getDummyWeights().minAssets, cutOff=0.05)
        self.checkTrackedAssets(exposureData.universe, assetIdxMap, estu)

        self.log.debug('generate_estimation_universe: end')
        return estu

    def generate_share_class_exposures(self, modelDate, data, modelDB, marketDB):
        """Compute factors based on A/B share class.
        """
        self.log.debug('generate_share_class_exposures: begin')
        expM = data.exposureMatrix

        # Differentiate between various China share classes
        (aShares, bShares, hShares, other) = MarketIndex.\
            process_china_share_classes(data, modelDate,
                                        modelDB, marketDB, factorName='')
        # Populate B-share indicator factor
        values = Matrices.allMasked(len(data.universe))
        ma.put(values, bShares, 1.0)
        expM.addFactor(
            'B-Share Market', values, ExposureMatrix.StyleFactor)

        self.log.debug('generate_share_class_exposures: end')
        return data.exposureMatrix

    def generate_model_specific_exposures(self, modelDate, data, modelDB, marketDB):
        """Generate the Exchange Rate Sensitivity factor and A/B
        share class dependent factors.
        """
        data.exposureMatrix.addFactor(
            'Exchange Rate Sensitivity', StyleExposures.generate_forex_sensitivity(
            data.returns, self, modelDB, 120, 'XDR'), ExposureMatrix.StyleFactor)
        data.exposureMatrix = self.generate_share_class_exposures(
                                modelDate, data, modelDB, marketDB)
        return data.exposureMatrix

    def score_linked_assets(self, date, universe, modelDB, marketDB, subIssueGroups=None):
        return dict()

    def clone_linked_asset_exposures(self, date, data, modelDB, marketDB, scoreDict,
                                        subIssueGroups=None):
        """Clones exposures for selected Chinese stocks from a parent asset
        """
        self.log.debug('clone_linked_CN_asset_exposures: begin')
        expM = data.exposureMatrix

        if not hasattr(data, 'hardCloneMap'):
            # Pick up dict of assets to be cloned from others
            data.hardCloneMap = modelDB.getClonedMap(date, data.universe, cloningOn=self.hardCloning)

        # Output some stats if required
        if self.debuggingReporting and hasattr(data, 'estimationUniverse'):
            expos_ESTU = ma.take(ma.filled(expM.getMatrix(), 0.0), exposureIdx, axis=0)
            expos_ESTU = ma.take(expos_ESTU, data.estimationUniverseIdx, axis=1)
            wt_ESTU = ma.take(data.marketCaps, data.estimationUniverseIdx, axis=0)
            averageExposureBefore = ma.average(expos_ESTU, axis=1, weights=wt_ESTU)

        # Loop round sets of linked assets and pull out exposures
        for (groupId, subIssueList) in subIssueGroups.items():
            cloneList = [n for n in subIssueList if n in data.hardCloneMap]
            for sid in cloneList:
                expM.getMatrix()[:, data.assetIdxMap[sid]] = \
                        expM.getMatrix()[:, data.assetIdxMap[data.hardCloneMap[sid]]]

        # Output some more stats if required
        if self.debuggingReporting and hasattr(data, 'estimationUniverse'):
            expos_ESTU = ma.take(ma.filled(expM.getMatrix(), 0.0), exposureIdx, axis=0)
            expos_ESTU = ma.take(expos_ESTU, data.estimationUniverseIdx, axis=1)
            averageExposureAfter = ma.average(expos_ESTU, axis=1, weights=wt_ESTU)
            for (idx, n) in enumerate(exposureNames):
                self.log.info('Date: %s, Factor: %s, Mean Before Cloning: %8.6f, After: %8.6f',
                        date, n, averageExposureBefore[idx], averageExposureAfter[idx])

        self.log.debug('clone_linked_asset_exposures: end')
        return data.exposureMatrix

class CNAxioma2010SH(CNAxioma2010MH):
    """Production AX-CN-SH China model
    """
    rm_id = 57
    revision = 1
    rms_id = 78

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.CNAxioma2010SH')
        MFM.SingleCountryFundamentalModel.__init__(
                            self, ['SEDOL'], modelDB, marketDB)
        self.setCalculators(modelDB)

    def setCalculators(self, modelDB, overrider = False):
        # Set up regression parameters
        self.returnCalculator = \
                defaultRegressionParameters(\
                self, modelDB, dummyThreshold=6.0, overrider=overrider)
        # Set up risk parameters
        defaultFundamentalCovarianceParameters(self, modelHorizon='short', nwLag=3, overrider=overrider)

class CNAxioma2010MH_S(MFM.StatisticalFactorModel):
    """Production AX-CN-S China model
    """
    rm_id = 41
    revision = 2
    rms_id = 144
    numFactors = 15
    blind = [ModelFactor('Statistical Factor %d' % n,
        'Statistical Factor %d' % n)
        for n in range(1, numFactors+1)]
    returnHistory = 250
    industryClassification = Classification.GICSCustomCN(
            datetime.date(2008,8,30))
    newExposureFormat = True

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.CNAxioma2010MH_S')
        MFM.StatisticalFactorModel.__init__(
                            self, ['SEDOL'], modelDB, marketDB)
        self.setCalculators(modelDB)
        self.baseModel = CNAxioma2010MH(modelDB, marketDB)

    def setCalculators(self, modelDB, overrider = False):
        self.returnCalculator = ReturnCalculator. \
                            AsymptoticPrincipalComponents2(self.numFactors)
        # Set up risk parameters
        defaultStatisticalCovarianceParameters(self,
                longSpecificReturnHistory=False, overrider=overrider)

    def clone_linked_asset_exposures(self, date, data, modelDB, marketDB, scoreDict,
            subIssueGroups=None):
        return self.baseModel.clone_linked_asset_exposures(\
                date, data, modelDB, marketDB, scoreDict, subIssueGroups=subIssueGroups)

class CNAxioma2010SH_S(CNAxioma2010MH_S):
    """Production AX-CN-SH-S China model
    """
    rm_id = 58
    revision = 1
    rms_id = 79

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.CNAxioma2010SH_S')
        MFM.StatisticalFactorModel.__init__(
                            self, ['SEDOL'], modelDB, marketDB)
        self.setCalculators(modelDB)
        self.baseModel = CNAxioma2010MH(modelDB, marketDB)

    def setCalculators(self, modelDB, overrider = False):
        self.returnCalculator = ReturnCalculator. \
                            AsymptoticPrincipalComponents2(self.numFactors)
        # Set up risk parameters
        defaultStatisticalCovarianceParameters(self,
                modelHorizon='short', longSpecificReturnHistory=False, overrider=overrider)

# Currency Models
class FXAxioma2010USD(CurrencyRisk.CurrencyStatisticalFactorModel):
    """Statistical factor based currency risk model, USD numeraire
    """
    rm_id = 36
    revision = 1
    rms_id = 56
    numStatFactors = 12
    blind = [ModelFactor('Statistical Factor %d' % n,
                         'Statistical Factor %d' % n)
                         for n in range(1, numStatFactors+1)]
    returnHistory = 250
    newExposureFormat = True

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.FXAxioma2010USD')
        CurrencyRisk.CurrencyStatisticalFactorModel.__init__(
                self, modelDB, marketDB)
        self.setCalculators(modelDB)

    def setCalculators(self, modelDB, overrider = False):
        varParameters = {'halfLife': 125, 'minObs': 250, 'maxObs': 500,
                         'NWLag': 1, 'DVAWindow': 125, 'DVAType': 'spline',}
        corrParameters = {'halfLife': 250, 'minObs': 250, 'maxObs': 500,
                          'NWLag': 1, 'DVAWindow': 125, 'DVAType': 'spline',}
        fullCovParameters = {}
        srParameters = {'halfLife': 125, 'minObs': 125, 'maxObs': 250,
                        'NWLag': 1, 'clipBounds': (-15.0,18.0),
                        'fillInFlag': False,}
        if overrider:
            overrider.overrideCovarianceParams(varParameters, corrParameters, srParameters, fullCovParameters)
        fullCovParameters = {}
        varParameters['DVAWindow'] = varParameters['halfLife']
        corrParameters['DVAWindow'] = varParameters['halfLife']

        self.vp = RiskCalculator.RiskParameters2009(varParameters)
        self.cp = RiskCalculator.RiskParameters2009(corrParameters)
        self.fp = RiskCalculator.RiskParameters2009(fullCovParameters)
        self.sp = RiskCalculator.RiskParameters2009(srParameters)
        self.covarianceCalculator = RiskCalculator.\
                CompositeCovarianceMatrix2009(self.fp, self.vp, self.cp)
        self.specificRiskCalculator = RiskCalculator.\
                BrilliantSpecificRisk2009(self.sp)

class FXAxioma2010USD_SH(CurrencyRisk.CurrencyStatisticalFactorModel):
    """Statistical factor based currency risk model, USD numeraire
    """
    rm_id = 66
    revision = 1
    rms_id = 93
    numStatFactors = 12
    blind = [ModelFactor('Statistical Factor %d' % n,
                         'Statistical Factor %d' % n)
                         for n in range(1, numStatFactors+1)]
    returnHistory = 250
    newExposureFormat = True

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.FXAxioma2010USD_SH')
        CurrencyRisk.CurrencyStatisticalFactorModel.__init__(
                self, modelDB, marketDB)
        self.setCalculators(modelDB)

    def setCalculators(self, modelDB, overrider = False):
        varParameters = {'halfLife': 60, 'minObs': 125, 'maxObs': 250,
                         'NWLag': 1, 'DVAWindow': 60, 'DVAType': 'spline',}
        corrParameters = {'halfLife': 125, 'minObs': 250, 'maxObs': 500,
                          'NWLag': 1, 'DVAWindow': 60, 'DVAType': 'spline',}
        fullCovParameters = {}
        srParameters = {'halfLife': 60, 'minObs': 60, 'maxObs': 125,
                        'NWLag': 1, 'clipBounds': (-15.0,18.0),
                        'fillInFlag': False,}
        if overrider:
            overrider.overrideCovarianceParams(varParameters, corrParameters, srParameters, fullCovParameters)
        fullCovParameters = {}
        varParameters['DVAWindow'] = varParameters['halfLife']
        corrParameters['DVAWindow'] = varParameters['halfLife']

        self.vp = RiskCalculator.RiskParameters2009(varParameters)
        self.cp = RiskCalculator.RiskParameters2009(corrParameters)
        self.fp = RiskCalculator.RiskParameters2009(fullCovParameters)
        self.sp = RiskCalculator.RiskParameters2009(srParameters)
        self.covarianceCalculator = RiskCalculator.\
                CompositeCovarianceMatrix2009(self.fp, self.vp, self.cp)
        self.specificRiskCalculator = RiskCalculator.\
                BrilliantSpecificRisk2009(self.sp)

class FXAxioma2010EUR(CurrencyRisk.CurrencyStatisticalFactorModel):
    """Statistical factor based currency risk model, EUR numeraire
    """
    rm_id = 37
    revision = 1
    rms_id = 57
    numStatFactors = 12
    blind = [ModelFactor('Statistical Factor %d' % n,
                         'Statistical Factor %d' % n)
                         for n in range(1, numStatFactors+1)]
    returnHistory = 250
    newExposureFormat = True

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.FXAxioma2010EUR')
        CurrencyRisk.CurrencyStatisticalFactorModel.__init__(
                self, modelDB, marketDB)
        self.setCalculators(modelDB)

    def setCalculators(self, modelDB, overrider = False):
        varParameters = {'halfLife': 125,
                         'minObs': 250, 'maxObs': 500,
                         'NWLag': 1,
                         'DVAWindow': 125, 'DVAType': 'spline',
                        }
        corrParameters = {'halfLife': 250,
                          'minObs': 250, 'maxObs': 500,
                          'NWLag': 1,
                          'DVAWindow': 125, 'DVAType': 'spline',
                         }
        fullCovParameters = {}
        srParameters = {'halfLife': 125, 'minObs': 125, 'maxObs': 250,
                        'NWLag': 1, 'clipBounds': (-15.0,18.0),
                        'fillInFlag': False,}
        if overrider:
            overrider.overrideCovarianceParams(varParameters, corrParameters, srParameters, fullCovParameters)
        fullCovParameters = {}
        varParameters['DVAWindow'] = varParameters['halfLife']
        corrParameters['DVAWindow'] = varParameters['halfLife']

        self.vp = RiskCalculator.RiskParameters2009(varParameters)
        self.cp = RiskCalculator.RiskParameters2009(corrParameters)
        self.fp = RiskCalculator.RiskParameters2009(fullCovParameters)
        self.sp = RiskCalculator.RiskParameters2009(srParameters)
        self.covarianceCalculator = RiskCalculator.\
                CompositeCovarianceMatrix2009(self.fp, self.vp, self.cp)
        self.specificRiskCalculator = RiskCalculator.\
                BrilliantSpecificRisk2009(self.sp)

class FXAxioma2010EUR_SH(CurrencyRisk.CurrencyStatisticalFactorModel):
    """Statistical factor based currency risk model, EUR numeraire
    """
    rm_id = 67
    revision = 1
    rms_id = 94
    numStatFactors = 12
    blind = [ModelFactor('Statistical Factor %d' % n,
                         'Statistical Factor %d' % n)
                         for n in range(1, numStatFactors+1)]
    returnHistory = 250
    newExposureFormat = True

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.FXAxioma2010EUR_SH')
        CurrencyRisk.CurrencyStatisticalFactorModel.__init__(
                self, modelDB, marketDB)
        self.setCalculators(modelDB)

    def setCalculators(self, modelDB, overrider = False):
        varParameters = {'halfLife': 60, 'minObs': 125, 'maxObs': 250,
                         'NWLag': 1, 'DVAWindow': 60, 'DVAType': 'spline',}
        corrParameters = {'halfLife': 125, 'minObs': 250, 'maxObs': 500,
                          'NWLag': 1, 'DVAWindow': 60, 'DVAType': 'spline',}
        fullCovParameters = {}
        srParameters = {'halfLife': 60, 'minObs': 60, 'maxObs': 125,
                        'NWLag': 1, 'clipBounds': (-15.0,18.0),
                        'fillInFlag': False,}
        if overrider:
            overrider.overrideCovarianceParams(varParameters, corrParameters, srParameters, fullCovParameters)
        fullCovParameters = {}
        varParameters['DVAWindow'] = varParameters['halfLife']
        corrParameters['DVAWindow'] = varParameters['halfLife']

        self.vp = RiskCalculator.RiskParameters2009(varParameters)
        self.cp = RiskCalculator.RiskParameters2009(corrParameters)
        self.fp = RiskCalculator.RiskParameters2009(fullCovParameters)
        self.sp = RiskCalculator.RiskParameters2009(srParameters)
        self.covarianceCalculator = RiskCalculator.\
                CompositeCovarianceMatrix2009(self.fp, self.vp, self.cp)
        self.specificRiskCalculator = RiskCalculator.\
                BrilliantSpecificRisk2009(self.sp)

# Regional Models
class WWAxioma2011MH(MFM.RegionalFundamentalModel):
    """Production AX-WW2.1 (Medium Horizon) Fundamental Model
    """
    rm_id = 76
    revision = 3
    rms_id = 109
    styles = [
              CompositeFactor('Value', 'Value'),
              CompositeFactor('Leverage', 'Leverage'),
              CompositeFactor('Growth', 'Growth'),
              ModelFactor('Size', 'Size'),
              ModelFactor('Short-Term Momentum', 'Short-Term Momentum'),
              ModelFactor('Medium-Term Momentum', 'Medium-Term Momentum'),
              ModelFactor('Volatility', 'Volatility'),
              ModelFactor('Liquidity','Liquidity'),
              ModelFactor('Exchange Rate Sensitivity', 'Exchange Rate Sensitivity'),
              ModelFactor('Domestic China', 'Domestic China'),
             ]
    localStructureFactors = [ModelFactor('Domestic China', 'Domestic China')]
    intercept = ModelFactor('Global Market', 'Global Market')
    sensitivityNumeraire = 'XDR'
    simpleIntercept = True
    industryClassification = Classification.GICSIndustries(
                                        datetime.date(2008,8,30))
    countryBetas = False
    newExposureFormat = True
    returnsTimingId = 1
    allCurrencies = True

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.WWAxioma2011MH')
        MFM.RegionalFundamentalModel.__init__(
                        self, ['SEDOL', 'CUSIP'], modelDB, marketDB)
        self.setCalculators(modelDB)
        modelDB.setTotalReturnCache(365*2)
        self.currencyModel = FXAxioma2010USD(modelDB, marketDB)
        scope = [Standardization.RegionRelativeScope(
                        modelDB, ['Value', 'Growth', 'Leverage']),
                 Standardization.GlobalRelativeScope(
                        ['Size', 'Short-Term Momentum', 'Medium-Term Momentum',
                         'Liquidity', 'Volatility', 'Exchange Rate Sensitivity',
                         'Domestic China'])]
        self.exposureStandardization = Standardization.\
                                    BucketizedStandardization(scope)

    def setCalculators(self, modelDB, overrider = False):
        # Set up regression parameters
        self.returnCalculator = \
                defaultRegressionParameters(self, modelDB,
                        scndRegs=[[ModelFactor('Domestic China', 'Domestic China')]],
                        k_rlm=5.0, weightedRLM=False, overrider=overrider)
        # Set up risk parameters
        defaultFundamentalCovarianceParameters(self, nwLag=2, overrider=overrider)

    def generate_model_specific_exposures(self, modelDate, data, modelDB, marketDB):
        """Generate descriptor based fundamental factors and
        Domestic China style factor.
        """
        data.exposureMatrix = self.generate_md_fundamental_exposures(
                modelDate, data, modelDB, marketDB)
        data.exposureMatrix = GlobalExposures.generate_china_domestic_exposures(
                data, modelDate, modelDB, marketDB)
        return data.exposureMatrix

    def generate_residual_regression_ESTU(self, indices_ESTU, weights_ESTU, iter,
                                          modelDate, expM, modelDB, marketDB):
        """Returns the estimation universe assets and weights
        to be used for the given loop of the nested model regression.
        The first loop uses the 'standard' ESTU, while the second
        uses the top 300 Chinese A-shares by total market cap.
        """
        self.log.debug('generate_residual_regression_ESTU: begin')
        if iter == 1:
            self.log.info('Using default ESTU for regression loop %d', iter)
        elif iter == 2:
            (indices_ESTU, weights_ESTU) = GlobalExposures.\
                    identify_top_china_a_shares(self, modelDate, expM,
                                             modelDB, marketDB, top=300)
        else:
            self.log.warning('Regression not defined for loop %d!', iter)
        self.log.debug('generate_residual_regression_ESTU: end')
        return (indices_ESTU, weights_ESTU)

    def generate_md_fundamental_exposures(self, modelDate, data, modelDB, marketDB):
        return self.generate_md_fundamental_exposures_v2(modelDate, data, modelDB, marketDB)

class WWAxioma2013MH(WWAxioma2011MH):
    """Production AX-WW2.2 (Medium Horizon) Fundamental Model
    """
    rm_id = 101
    revision = 1
    rms_id = 165

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.WWAxioma2013MH')
        WWAxioma2011MH.__init__(self, modelDB, marketDB)

class WWAxioma2011MH_Pre2009(WWAxioma2011MH):
    """Production AX-WW2.1 (Medium Horizon) Fundamental Model
    For pre-2009 dates, does not contain some Frontier markets.
    """
    rm_id = 76
    revision = 2
    rms_id = 108

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.WWAxioma2011MH_Pre2009')
        WWAxioma2011MH.__init__(self, modelDB, marketDB)

class WWAxioma2011MH_Pre2003(WWAxioma2011MH):
    """Production AX-WW2 (Medium Horizon) Fundamental Model
    For pre-2003 dates, does not contain Frontier markets.
    """
    rm_id = 76
    revision = 4
    rms_id = 145
    additionalCurrencyFactors = [ModelFactor('EUR', 'Euro')]

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.WWAxioma2011MH_Pre2003')
        WWAxioma2011MH.__init__(self, modelDB, marketDB)

class WWAxioma2011SH(WWAxioma2011MH):
    """Production AX-WW2.1 (Short Horizon) Fundamental Model
    """
    rm_id = 60
    revision = 3
    rms_id = 83
    newExposureFormat = True

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.WWAxioma2011SH')
        MFM.RegionalFundamentalModel.__init__(
                        self, ['SEDOL', 'CUSIP'], modelDB, marketDB)
        self.setCalculators(modelDB)
        modelDB.setTotalReturnCache(365*2)
        self.currencyModel = FXAxioma2010USD_SH(modelDB, marketDB)
        scope = [Standardization.RegionRelativeScope(
                        modelDB, ['Value', 'Growth', 'Leverage']),
                 Standardization.GlobalRelativeScope(
                        ['Size', 'Short-Term Momentum', 'Medium-Term Momentum',
                         'Liquidity', 'Volatility', 'Exchange Rate Sensitivity',
                         'Domestic China'])]
        self.exposureStandardization = Standardization.\
                                    BucketizedStandardization(scope)

    def setCalculators(self, modelDB, overrider = False):
        # Set up regression parameters
        self.returnCalculator = \
                defaultRegressionParameters(self, modelDB,
                        scndRegs=[[ModelFactor('Domestic China', 'Domestic China')]],
                        k_rlm=5.0, weightedRLM=False, overrider=overrider)
        # Set up risk parameters
        defaultFundamentalCovarianceParameters(self, nwLag=2, overrider=overrider, modelHorizon='short')

class WWAxioma2013SH(WWAxioma2011SH):
    """Production AX-WW2.2 (Short Horizon) Fundamental Model
    """
    rm_id = 103
    revision = 1
    rms_id = 167

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.WWAxioma2013SH')
        WWAxioma2011SH.__init__(self, modelDB, marketDB)


class WWAxioma2011SH_Pre2009(WWAxioma2011SH):
    """Production AX-WW2.1 (Short Horizon) Fundamental Model
    For pre-2009 dates, does not contain some Frontier markets.
    """
    rm_id = 60
    revision = 2
    rms_id = 82

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.WWAxioma2011SH_Pre2009')
        WWAxioma2011SH.__init__(self, modelDB, marketDB)

class WWAxioma2011SH_Pre2003(WWAxioma2011SH):
    """Production AX-WW2.1 (Short Horizon) Fundamental Model
    For pre-2003 dates, does not contain Frontier markets.
    """
    rm_id = 60
    revision = 4
    rms_id = 147
    additionalCurrencyFactors = [ModelFactor('EUR', 'Euro')]

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.WWAxioma2011SH_Pre2003')
        WWAxioma2011SH.__init__(self, modelDB, marketDB)

class WWAxioma2011MH_S(MFM.RegionalStatisticalFactorModel):
    """Production AX-WW2.1 (Medium Horizon) Stat Model
    """
    rm_id = 77
    revision = 1
    rms_id = 110
    numFactors = 20
    blind = [ModelFactor('Statistical Factor %d' % n,
                         'Statistical Factor %d' % n)
                         for n in range(1, numFactors+1)]
    returnHistory = 250
    industryClassification = Classification.GICSIndustries(
                                        datetime.date(2008,8,30))
    newExposureFormat = True
    returnsTimingId = 1
    allCurrencies = True

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.WWAxioma2011MH_S')
        MFM.RegionalStatisticalFactorModel.__init__(
                        self, ['SEDOL', 'CUSIP'], modelDB, marketDB)
        self.setCalculators(modelDB)
        modelDB.setTotalReturnCache(367)
        self.currencyModel = FXAxioma2010USD(modelDB, marketDB)
        l = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(logging.ERROR)
        self.baseModelDateMap = {
                datetime.date(2009,1,1): WWAxioma2011MH(modelDB, marketDB),
                datetime.date(2003,1,1): WWAxioma2011MH_Pre2009(modelDB, marketDB),
                datetime.date(1980,1,1): WWAxioma2011MH_Pre2003(modelDB, marketDB)
               }
        logging.getLogger().setLevel(l)

    def setCalculators(self, modelDB, overrider = False):
        # Set up risk parameters
        defaultStatisticalCovarianceParameters(self, scaleMinObs=0.9, overrider=overrider)
        self.returnCalculator = ReturnCalculator. \
                AsymptoticPrincipalComponents2(self.numFactors)

class WWAxioma2013MH_S(WWAxioma2011MH_S):
    """Production AX-WW2.2 (Medium Horizon) Fundamental Model
    """
    rm_id = 102
    revision = 1
    rms_id = 166
    additionalCurrencyFactors = [ModelFactor('EUR', 'Euro')]

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.WWAxioma2013MH_S')
        WWAxioma2011MH_S.__init__(self, modelDB, marketDB)

class WWAxioma2011SH_S(WWAxioma2011MH_S):
    """Production AX-WW2.1 (Short Horizon) Stat Model
    """
    rm_id = 61
    revision = 1
    rms_id = 84
    newExposureFormat = True

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.WWAxioma2011SH_S')
        MFM.RegionalStatisticalFactorModel.__init__(
                        self, ['SEDOL', 'CUSIP'], modelDB, marketDB)
        self.setCalculators(modelDB)
        modelDB.setTotalReturnCache(367)
        self.currencyModel = FXAxioma2010USD_SH(modelDB, marketDB)
        l = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(logging.ERROR)
        self.baseModelDateMap = {
                datetime.date(2009,1,1): WWAxioma2011MH(modelDB, marketDB),
                datetime.date(2003,1,1): WWAxioma2011MH_Pre2009(modelDB, marketDB),
                datetime.date(1980,1,1): WWAxioma2011MH_Pre2003(modelDB, marketDB)
               }
        logging.getLogger().setLevel(l)

    def setCalculators(self, modelDB, overrider = False):
        # Set up risk parameters
        defaultStatisticalCovarianceParameters(self, scaleMinObs=0.9,
                modelHorizon='short', overrider=overrider)
        self.returnCalculator = ReturnCalculator. \
                AsymptoticPrincipalComponents2(self.numFactors)

class WWAxioma2013SH_S(WWAxioma2011SH_S):
    """Production AX-WW2.2 (Short Horizon) Stat Model
    """
    rm_id = 104
    revision = 1
    rms_id = 168

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.WWAxioma2013SH_S')
        WWAxioma2011SH_S.__init__(self, modelDB, marketDB)

class EUAxioma2011MH(MFM.RegionalFundamentalModel):
    """Production AX-EU2 (Medium Horizon) Fundamental Model
    """
    rm_id = 78
    revision = 3
    rms_id = 113
    styles = [
              CompositeFactor('Value', 'Value'),
              CompositeFactor('Leverage', 'Leverage'),
              CompositeFactor('Growth', 'Growth'),
              ModelFactor('Size', 'Size'),
              ModelFactor('Short-Term Momentum', 'Short-Term Momentum'),
              ModelFactor('Medium-Term Momentum', 'Medium-Term Momentum'),
              ModelFactor('Volatility', 'Volatility'),
              ModelFactor('Liquidity','Liquidity'),
              ModelFactor('Exchange Rate Sensitivity', 'Exchange Rate Sensitivity'),
             ]
    intercept = ModelFactor('European Market', 'European Market')
    sensitivityNumeraire = 'XDR'
    simpleIntercept = True
    industryClassification = Classification.GICSIndustryGroups(
                                        datetime.date(2008,8,30))
    countryBetas = False
    newExposureFormat = True
    specReturnTimingId = 1
    allCurrencies = True

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.EUAxioma2011MH')
        MFM.RegionalFundamentalModel.__init__(
                        self, ['SEDOL', 'CUSIP'], modelDB, marketDB)
        self.setCalculators(modelDB)
        modelDB.setTotalReturnCache(260*2)
        self.currencyModel = FXAxioma2010EUR(modelDB, marketDB)
        scope = [Standardization.RegionRelativeScope(
                        modelDB, ['Value', 'Growth', 'Leverage']),
                 Standardization.GlobalRelativeScope(
                        ['Size', 'Short-Term Momentum', 'Medium-Term Momentum',
                         'Liquidity', 'Volatility', 'Exchange Rate Sensitivity'])]
        self.exposureStandardization = Standardization.\
                                    BucketizedStandardization(scope)

    def setCalculators(self, modelDB, overrider = False):
        # Set up regression parameters
        self.returnCalculator = \
                defaultRegressionParameters(self, modelDB,
                    k_rlm=5.0, weightedRLM=False, overrider=overrider)
        # Set up risk parameters
        defaultFundamentalCovarianceParameters(self, nwLag=3, overrider=overrider)

    def generate_model_specific_exposures(self, modelDate, data, modelDB, marketDB):
        """Generate descriptor based fundamental factors.
        """
        return self.generate_md_fundamental_exposures(
                modelDate, data, modelDB, marketDB)

    def generate_md_fundamental_exposures(self, modelDate, data, modelDB, marketDB):
        return self.generate_md_fundamental_exposures_v2(modelDate, data, modelDB, marketDB)

class EUAxioma2011MH_Pre2009(EUAxioma2011MH):
    """Production AX-EU2 (Medium Horizon) Fundamental Model
    For pre-2009 dates, does not contain some Frontier markets.
    """
    rm_id = 78
    revision = 2
    rms_id = 112

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.EUAxioma2011MH_Pre2009')
        EUAxioma2011MH.__init__(self, modelDB, marketDB)

class EUAxioma2011MH_Pre2003(EUAxioma2011MH):
    """Production AX-EU2 (Medium Horizon) Fundamental Model
    For pre-2003 dates, does not contain Frontier markets.
    """
    rm_id = 78
    revision = 4
    rms_id = 146

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.EUAxioma2011MH_Pre2003')
        EUAxioma2011MH.__init__(self, modelDB, marketDB)

class EUAxioma2011SH(EUAxioma2011MH):
    """Production AX-EU2 (Short Horizon) Fundamental Model
    """
    rm_id = 62
    revision = 3
    rms_id = 87
    newExposureFormat = True

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.EUAxioma2011SH')
        MFM.RegionalFundamentalModel.__init__(
                        self, ['SEDOL', 'CUSIP'], modelDB, marketDB)
        self.setCalculators(modelDB)
        modelDB.setTotalReturnCache(260*2)
        self.currencyModel = FXAxioma2010EUR_SH(modelDB, marketDB)
        scope = [Standardization.RegionRelativeScope(
                        modelDB, ['Value', 'Growth', 'Leverage']),
                 Standardization.GlobalRelativeScope(
                        ['Size', 'Short-Term Momentum', 'Medium-Term Momentum',
                         'Liquidity', 'Volatility', 'Exchange Rate Sensitivity'])]
        self.exposureStandardization = Standardization.\
                                    BucketizedStandardization(scope)

    def setCalculators(self, modelDB, overrider = False):
        # Set up regression parameters
        self.returnCalculator = \
                defaultRegressionParameters(self, modelDB,
                    k_rlm=5.0, weightedRLM=False, overrider=overrider)
        # Set up risk parameters
        defaultFundamentalCovarianceParameters(self, nwLag=3, modelHorizon='short', overrider=overrider)

class EUAxioma2011SH_Pre2009(EUAxioma2011SH):
    """Production AX-EU2 (Short Horizon) Fundamental Model
    For pre-2009 dates, does not contain some Frontier markets.
    """
    rm_id = 62
    revision = 2
    rms_id = 86

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.EUAxioma2011SH_Pre2009')
        EUAxioma2011SH.__init__(self, modelDB, marketDB)

class EUAxioma2011SH_Pre2003(EUAxioma2011SH):
    """Production AX-EU2 (Short Horizon) Fundamental Model
    For pre-2003 dates, does not contain Frontier markets.
    """
    rm_id = 62
    revision = 4
    rms_id = 148

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.EUAxioma2011SH_Pre2003')
        EUAxioma2011SH.__init__(self, modelDB, marketDB)

class EUAxioma2011MH_S(MFM.RegionalStatisticalFactorModel):
    """Production AX-EU2 (Medium Horizon) Stat Model
    """
    rm_id = 79
    revision = 1
    rms_id = 114
    numFactors = 20
    blind = [ModelFactor('Statistical Factor %d' % n,
                         'Statistical Factor %d' % n)
                         for n in range(1, numFactors+1)]
    returnHistory = 250
    industryClassification = Classification.GICSIndustryGroups(
                                        datetime.date(2008,8,30))
    newExposureFormat = True
    allCurrencies = True

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.EUAxioma2011MH_S')
        MFM.RegionalStatisticalFactorModel.__init__(
                        self, ['SEDOL', 'CUSIP'], modelDB, marketDB)
        self.setCalculators(modelDB)
        modelDB.setTotalReturnCache(252)
        self.currencyModel = FXAxioma2010EUR(modelDB, marketDB)
        l = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(logging.ERROR)
        self.baseModelDateMap = {
                datetime.date(2009,1,1): EUAxioma2011MH(modelDB, marketDB),
                datetime.date(2003,1,1): EUAxioma2011MH_Pre2009(modelDB, marketDB),
                datetime.date(1980,1,1): EUAxioma2011MH_Pre2003(modelDB, marketDB)
               }
        logging.getLogger().setLevel(l)

    def setCalculators(self, modelDB, overrider = False):
        # Set up risk parameters
        defaultStatisticalCovarianceParameters(self, scaleMinObs=0.9, overrider=overrider)
        self.returnCalculator = ReturnCalculator. \
                AsymptoticPrincipalComponents2(self.numFactors)

class EUAxioma2011SH_S(EUAxioma2011MH_S):
    """Production AX-EU2 (Short Horizon) Stat Model
    """
    rm_id = 63
    revision = 1
    rms_id = 88
    newExposureFormat = True

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.EUAxioma2011MH_S')
        MFM.RegionalStatisticalFactorModel.__init__(
                        self, ['SEDOL', 'CUSIP'], modelDB, marketDB)
        self.setCalculators(modelDB)
        modelDB.setTotalReturnCache(252)
        self.currencyModel = FXAxioma2010EUR_SH(modelDB, marketDB)
        l = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(logging.ERROR)
        self.baseModelDateMap = {
                datetime.date(2009,1,1): EUAxioma2011MH(modelDB, marketDB),
                datetime.date(2003,1,1): EUAxioma2011MH_Pre2009(modelDB, marketDB),
                datetime.date(1980,1,1): EUAxioma2011MH_Pre2003(modelDB, marketDB)
               }
        logging.getLogger().setLevel(l)

    def setCalculators(self, modelDB, overrider = False):
        # Set up risk parameters
        defaultStatisticalCovarianceParameters(self, scaleMinObs=0.9,
                modelHorizon='short', overrider=overrider)
        self.returnCalculator = ReturnCalculator. \
                AsymptoticPrincipalComponents2(self.numFactors)

class EMAxioma2011MH(MFM.RegionalFundamentalModel):
    """Production AX-EM2 (Medium Horizon) Fundamental Model
    """
    rm_id = 80
    revision = 5
    rms_id = 175
    styles = [
              CompositeFactor('Value', 'Value'),
              CompositeFactor('Leverage', 'Leverage'),
              CompositeFactor('Growth', 'Growth'),
              ModelFactor('Size', 'Size'),
              ModelFactor('Short-Term Momentum', 'Short-Term Momentum'),
              ModelFactor('Medium-Term Momentum', 'Medium-Term Momentum'),
              ModelFactor('Volatility', 'Volatility'),
              ModelFactor('Liquidity','Liquidity'),
              ModelFactor('Exchange Rate Sensitivity', 'Exchange Rate Sensitivity'),
              ModelFactor('Domestic China', 'Domestic China'),
             ]
    localStructureFactors = [ModelFactor('Domestic China', 'Domestic China')]
    intercept = ModelFactor('Global Market', 'Global Market')
    sensitivityNumeraire = 'XDR'
    simpleIntercept = True
    industryClassification = Classification.GICSIndustryGroups(
                                        datetime.date(2008,8,30))
    countryBetas = False
    newExposureFormat = True
    returnsTimingId = 1
    allCurrencies = True

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.EMAxioma2011MH')
        MFM.RegionalFundamentalModel.__init__(
                        self, ['SEDOL', 'CUSIP'], modelDB, marketDB)
        self.setCalculators(modelDB)
        modelDB.setTotalReturnCache(365*2)
        self.currencyModel = FXAxioma2010USD(modelDB, marketDB)
        scope = [Standardization.RegionRelativeScope(
                        modelDB, ['Value', 'Growth', 'Leverage']),
                 Standardization.GlobalRelativeScope(
                        ['Size', 'Short-Term Momentum', 'Medium-Term Momentum',
                         'Liquidity', 'Volatility', 'Exchange Rate Sensitivity',
                         'Domestic China'])]
        self.exposureStandardization = Standardization.\
                                    BucketizedStandardization(scope)

    def setCalculators(self, modelDB, overrider = False):
        # Set up regression parameters
        self.returnCalculator = \
                defaultRegressionParameters( self, modelDB,
                scndRegs=[[ModelFactor('Domestic China', 'Domestic China')]],
                    k_rlm=5.0, weightedRLM=False, overrider=overrider)
        # Set up risk parameters
        defaultFundamentalCovarianceParameters(self, nwLag=4, overrider=overrider)

    def generate_model_specific_exposures(self, modelDate, data, modelDB, marketDB):
        """Generate descriptor based fundamental factors and
        Domestic China style factor.
        """
        data.exposureMatrix = self.generate_md_fundamental_exposures(
                modelDate, data, modelDB, marketDB)
        data.exposureMatrix = GlobalExposures.generate_china_domestic_exposures(
                data, modelDate, modelDB, marketDB)
        return data.exposureMatrix

    def generate_residual_regression_ESTU(self, indices_ESTU, weights_ESTU, iter,
                                          modelDate, expM, modelDB, marketDB):
        """Returns the estimation universe assets and weights
        to be used for the given loop of the nested model regression.
        The first loop uses the 'standard' ESTU, while the second
        uses the top 300 Chinese A-shares by total market cap.
        """
        self.log.debug('generate_residual_regression_ESTU: begin')
        if iter == 1:
            self.log.info('Using default ESTU for regression loop %d', iter)
        elif iter == 2:
            (indices_ESTU, weights_ESTU) = GlobalExposures.\
                    identify_top_china_a_shares(self, modelDate, expM,
                                             modelDB, marketDB, top=300)
        else:
            self.log.warning('Regression not defined for loop %d!', iter)
        self.log.debug('generate_residual_regression_ESTU: end')
        return (indices_ESTU, weights_ESTU)

    def generate_md_fundamental_exposures(self, modelDate, data, modelDB, marketDB):
        return self.generate_md_fundamental_exposures_v2(modelDate, data, modelDB, marketDB)

class EMAxioma2011MH_Pre2013(EMAxioma2011MH):
    """Production AX-EM2 (Medium Horizon) Fundamental Model
    For pre-2013 dates; Greece-free
    """
    rm_id = 80
    revision = 3
    rms_id = 117

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.EMAxioma2011MH_Pre2013')
        EMAxioma2011MH.__init__(self, modelDB, marketDB)

class EMAxioma2011MH_Pre2009(EMAxioma2011MH):
    """Production AX-EM2 (Medium Horizon) Fundamental Model
    For pre-2009 dates, does not contain some Frontier markets.
    """
    rm_id = 80
    revision = 2
    rms_id = 116

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.EMAxioma2011MH_Pre2009')
        EMAxioma2011MH.__init__(self, modelDB, marketDB)

class EMAxioma2011MH_Pre2003(EMAxioma2011MH):
    """Production AX-EM2 (Medium Horizon) Fundamental Model
    For pre-2003 dates, does not contain Frontier markets.
    """
    rm_id = 80
    revision = 4
    rms_id = 149

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.EMAxioma2011MH_Pre2003')
        EMAxioma2011MH.__init__(self, modelDB, marketDB)

class EMAxioma2011SH(EMAxioma2011MH):
    """Production AX-EM2 (Short Horizon) Fundamental Model
    """
    rm_id = 64
    revision = 5
    rms_id = 177
    newExposureFormat = True

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.EMAxioma2011SH')
        MFM.RegionalFundamentalModel.__init__(
                        self, ['SEDOL', 'CUSIP'], modelDB, marketDB)
        self.setCalculators(modelDB)
        modelDB.setTotalReturnCache(365*2)
        self.currencyModel = FXAxioma2010USD_SH(modelDB, marketDB)
        scope = [Standardization.RegionRelativeScope(
                        modelDB, ['Value', 'Growth', 'Leverage']),
                 Standardization.GlobalRelativeScope(
                        ['Size', 'Short-Term Momentum', 'Medium-Term Momentum',
                         'Liquidity', 'Volatility', 'Exchange Rate Sensitivity',
                         'Domestic China'])]
        self.exposureStandardization = Standardization.\
                                    BucketizedStandardization(scope)

    def setCalculators(self, modelDB, overrider = False):
        # Set up regression parameters
        self.returnCalculator = \
                defaultRegressionParameters( self, modelDB,
                scndRegs=[[ModelFactor('Domestic China', 'Domestic China')]],
                    k_rlm=5.0, weightedRLM=False, overrider=overrider)
        # Set up risk parameters
        defaultFundamentalCovarianceParameters(self, nwLag=4, modelHorizon='short', overrider=overrider)

class EMAxioma2011SH_Pre2013(EMAxioma2011SH):
    """Production AX-EM2 (Short Horizon) Fundamental Model
    For pre-2013 dates; Greece-free
    """
    rm_id = 64
    revision = 3
    rms_id = 91

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.EMAxioma2011SH')
        EMAxioma2011SH.__init__(self, modelDB, marketDB)

class EMAxioma2011SH_Pre2009(EMAxioma2011SH):
    """Production AX-EM2 (Short Horizon) Fundamental Model
    For pre-2009 dates, does not contain some Frontier markets.
    """
    rm_id = 64
    revision = 2
    rms_id = 90

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.EMAxioma2011SH_Pre2009')
        EMAxioma2011SH.__init__(self, modelDB, marketDB)

class EMAxioma2011SH_Pre2003(EMAxioma2011SH):
    """Production AX-EM2 (Short Horizon) Fundamental Model
    For pre-2003 dates, does not contain Frontier markets.
    """
    rm_id = 64
    revision = 4
    rms_id = 150

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.EMAxioma2011SH_Pre2003')
        EMAxioma2011SH.__init__(self, modelDB, marketDB)

class EMAxioma2011MH_S(MFM.RegionalStatisticalFactorModel):
    """Production AX-EM2 (Medium Horizon) Stat Model
    """
    rm_id = 81
    revision = 2
    rms_id = 176
    numFactors = 20
    blind = [ModelFactor('Statistical Factor %d' % n,
                         'Statistical Factor %d' % n)
                         for n in range(1, numFactors+1)]
    returnHistory = 250
    industryClassification = Classification.GICSIndustryGroups(
                                        datetime.date(2008,8,30))
    newExposureFormat = True
    returnsTimingId = 1
    allCurrencies = True

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.EMAxioma2011MH_S')
        MFM.RegionalStatisticalFactorModel.__init__(
                        self, ['SEDOL', 'CUSIP'], modelDB, marketDB)
        self.setCalculators(modelDB)
        modelDB.setTotalReturnCache(367)
        self.currencyModel = FXAxioma2010USD(modelDB, marketDB)
        l = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(logging.ERROR)
        self.baseModelDateMap = {
                datetime.date(2013,6,28): EMAxioma2011MH(modelDB, marketDB),
                datetime.date(2009,1,1): EMAxioma2011MH_Pre2013(modelDB, marketDB),
                datetime.date(2003,1,1): EMAxioma2011MH_Pre2009(modelDB, marketDB),
                datetime.date(1980,1,1): EMAxioma2011MH_Pre2003(modelDB, marketDB)
               }
        logging.getLogger().setLevel(l)

    def setCalculators(self, modelDB, overrider = False):
        # Set up risk parameters
        defaultStatisticalCovarianceParameters(self, scaleMinObs=0.9, overrider=overrider)
        self.returnCalculator = ReturnCalculator. \
                AsymptoticPrincipalComponents2(self.numFactors)

class EMAxioma2011MH_S_Pre2013(EMAxioma2011MH_S):
    """Production AX-EM2 (Medium Horizon) Stat Model pre the addition of Greece
    """
    rm_id = 81
    revision = 1
    rms_id = 118

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.EMAxioma2011MH_S')
        EMAxioma2011MH_S.__init__( self, modelDB, marketDB)
        l = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(logging.ERROR)
        self.baseModelDateMap = {
                datetime.date(2009,1,1): EMAxioma2011MH_Pre2013(modelDB, marketDB),
                datetime.date(2003,1,1): EMAxioma2011MH_Pre2009(modelDB, marketDB),
                datetime.date(1980,1,1): EMAxioma2011MH_Pre2003(modelDB, marketDB)
               }
        logging.getLogger().setLevel(l)

class EMAxioma2011SH_S(EMAxioma2011MH_S):
    """Production AX-EM2 (Short Horizon) Stat Model + Greece
    """
    rm_id = 65
    revision = 2
    rms_id = 178
    newExposureFormat = True

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.EMAxioma2011SH_S')
        MFM.RegionalStatisticalFactorModel.__init__(
                        self, ['SEDOL', 'CUSIP'], modelDB, marketDB)
        self.setCalculators(modelDB)
        modelDB.setTotalReturnCache(367)
        self.currencyModel = FXAxioma2010USD_SH(modelDB, marketDB)
        l = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(logging.ERROR)
        self.baseModelDateMap = {
                datetime.date(2013,6,28): EMAxioma2011MH(modelDB, marketDB),
                datetime.date(2009,1,1): EMAxioma2011MH_Pre2013(modelDB, marketDB),
                datetime.date(2003,1,1): EMAxioma2011MH_Pre2009(modelDB, marketDB),
                datetime.date(1980,1,1): EMAxioma2011MH_Pre2003(modelDB, marketDB)
               }
        logging.getLogger().setLevel(l)

    def setCalculators(self, modelDB, overrider = False):
        # Set up risk parameters
        defaultStatisticalCovarianceParameters(self, scaleMinObs=0.9,
                modelHorizon='short', overrider=overrider)
        self.returnCalculator = ReturnCalculator. \
                AsymptoticPrincipalComponents2(self.numFactors)

class EMAxioma2011SH_S_Pre2013(EMAxioma2011SH_S):
    """Production AX-EM2 (Short Horizon) Stat Model pre the addition of Greece
    """
    rm_id = 65
    revision = 1
    rms_id = 92

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.EMAxioma2011SH_S')
        EMAxioma2011SH_S.__init__( self, modelDB, marketDB)
        l = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(logging.ERROR)
        self.baseModelDateMap = {
                datetime.date(2009,1,1): EMAxioma2011MH_Pre2013(modelDB, marketDB),
                datetime.date(2003,1,1): EMAxioma2011MH_Pre2009(modelDB, marketDB),
                datetime.date(1980,1,1): EMAxioma2011MH_Pre2003(modelDB, marketDB)
               }
        logging.getLogger().setLevel(l)

class APAxioma2011MH(MFM.RegionalFundamentalModel):
    """Production Asia-Pacific Fundamental Model
    """
    rm_id = 82
    revision = 2
    rms_id = 120
    styles = [
              CompositeFactor('Value', 'Value'),
              CompositeFactor('Leverage', 'Leverage'),
              CompositeFactor('Growth', 'Growth'),
              ModelFactor('Size', 'Size'),
              ModelFactor('Short-Term Momentum', 'Short-Term Momentum'),
              ModelFactor('Medium-Term Momentum', 'Medium-Term Momentum'),
              ModelFactor('Volatility', 'Volatility'),
              ModelFactor('Liquidity','Liquidity'),
              ModelFactor('Exchange Rate Sensitivity', 'Exchange Rate Sensitivity'),
              ModelFactor('Domestic China', 'Domestic China'),
             ]
    localStructureFactors = [ModelFactor('Domestic China', 'Domestic China')]
    intercept = ModelFactor('Asian Market', 'Asian Market')
    sensitivityNumeraire = 'XDR'
    simpleIntercept = True
    industryClassification = Classification.GICSIndustryGroups(
            datetime.date(2008,8,30))
    countryBetas = False
    newExposureFormat = True
    specReturnTimingId = 1
    allCurrencies = True

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.APAxioma2011MH')
        MFM.RegionalFundamentalModel.__init__(self, ['SEDOL'], modelDB, marketDB)
        self.setCalculators(modelDB)
        modelDB.setTotalReturnCache(260*2)
        self.currencyModel = FXAxioma2010USD(modelDB, marketDB)
        scope = [Standardization.RegionRelativeScope(
                        modelDB, ['Value', 'Growth', 'Leverage']),
                 Standardization.GlobalRelativeScope(
                        ['Size', 'Short-Term Momentum', 'Medium-Term Momentum',
                         'Liquidity', 'Volatility', 'Exchange Rate Sensitivity',
                         'Domestic China'])]
        self.exposureStandardization = Standardization.\
                                    BucketizedStandardization(scope)

    def setCalculators(self, modelDB, overrider = False):
        # Set up regression parameters
        self.returnCalculator = \
                defaultRegressionParameters(\
                self, modelDB, dummyType='Sectors',
                scndRegs=[[ModelFactor('Domestic China', 'Domestic China')]],
                    k_rlm=5.0, weightedRLM=False, overrider=overrider)
        # Set up risk parameters
        defaultFundamentalCovarianceParameters(self, nwLag=2, overrider=overrider)

    def generate_model_specific_exposures(self, modelDate, data, modelDB, marketDB):
        """Generate descriptor based fundamental factors and
        Domestic China style factor.
        """
        data.exposureMatrix = self.generate_md_fundamental_exposures(
                modelDate, data, modelDB, marketDB)
        data.exposureMatrix = GlobalExposures.generate_china_domestic_exposures(
                data, modelDate, modelDB, marketDB)
        return data.exposureMatrix

    def generate_residual_regression_ESTU(self, indices_ESTU, weights_ESTU, iter,
                                          modelDate, expM, modelDB, marketDB):
        """Returns the estimation universe assets and weights
        to be used for the given loop of the nested model regression.
        The first loop uses the 'standard' ESTU, while the second
        uses the top 300 Chinese A-shares by total market cap.
        """
        self.log.debug('generate_residual_regression_ESTU: begin')
        if iter == 1:
            self.log.info('Using default ESTU for regression loop %d', iter)
        elif iter == 2:
            (indices_ESTU, weights_ESTU) = GlobalExposures.\
                    identify_top_china_a_shares(self, modelDate, expM,
                                             modelDB, marketDB, top=300)
        else:
            self.log.warning('Regression not defined for loop %d!', iter)
        self.log.debug('generate_residual_regression_ESTU: end')
        return (indices_ESTU, weights_ESTU)

    def generate_md_fundamental_exposures(self, modelDate, data, modelDB, marketDB):
        return self.generate_md_fundamental_exposures_v2(modelDate, data, modelDB, marketDB)

class APAxioma2013MH(APAxioma2011MH):
    """Production AX-AP2.2 (Medium Horizon) Fundamental Model
    """
    rm_id = 97
    revision = 1
    rms_id = 161

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.APAxioma2013MH')
        APAxioma2011MH.__init__(self, modelDB, marketDB)

class APAxioma2011MH_Pre2009(APAxioma2011MH):
    """Production AX-AP2.1 (Medium Horizon) Fundamental Model
    For pre-2009 dates, does not contain some Frontier markets.
    """
    rm_id = 82
    revision = 1
    rms_id = 119

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.APAxioma2011MH_Pre2009')
        APAxioma2011MH.__init__(self, modelDB, marketDB)

class APAxioma2011SH(APAxioma2011MH):
    """Production Asia-Pacific Fundamental Model
    """
    rm_id = 68
    revision = 2
    rms_id = 96
    newExposureFormat = True

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.APAxioma2011MH')
        MFM.RegionalFundamentalModel.__init__(self, ['SEDOL'], modelDB, marketDB)
        self.setCalculators(modelDB)
        modelDB.setTotalReturnCache(260*2)
        self.currencyModel = FXAxioma2010USD_SH(modelDB, marketDB)
        scope = [Standardization.RegionRelativeScope(
                        modelDB, ['Value', 'Growth', 'Leverage']),
                 Standardization.GlobalRelativeScope(
                        ['Size', 'Short-Term Momentum', 'Medium-Term Momentum',
                         'Liquidity', 'Volatility', 'Exchange Rate Sensitivity',
                         'Domestic China'])]
        self.exposureStandardization = Standardization.\
                                    BucketizedStandardization(scope)

    def setCalculators(self, modelDB, overrider = False):
        # Set up regression parameters
        self.returnCalculator = \
                defaultRegressionParameters(\
                self, modelDB, dummyType='Sectors',
                scndRegs=[[ModelFactor('Domestic China', 'Domestic China')]],
                    k_rlm=5.0, weightedRLM=False, overrider=overrider)
        # Set up risk parameters
        defaultFundamentalCovarianceParameters(self, nwLag=2, modelHorizon='short', overrider=overrider)

class APAxioma2013SH(APAxioma2011SH):
    """Production AX-AP2.2 (Short Horizon) Fundamental Model
    """
    rm_id = 99
    revision = 1
    rms_id = 163

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.APAxioma2013SH')
        APAxioma2011SH.__init__(self, modelDB, marketDB)

class APAxioma2011SH_Pre2009(APAxioma2011SH):
    """Production AX-AP2.1 (Short Horizon) Fundamental Model
    For pre-2009 dates, does not contain some Frontier markets.
    """
    rm_id = 68
    revision = 1
    rms_id = 95

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.APAxioma2011SH_Pre2009')
        APAxioma2011SH.__init__(self, modelDB, marketDB)

class APAxioma2011MH_S(MFM.RegionalStatisticalFactorModel):
    """Production Asia-Pacific Statistical Model
    """
    rm_id = 83
    revision = 1
    rms_id = 121
    numFactors = 20
    blind = [ModelFactor('Statistical Factor %d' % n,
                         'Statistical Factor %d' % n)
             for n in range(1, numFactors+1)]
    returnHistory = 250
    industryClassification = Classification.GICSIndustryGroups(
                                        datetime.date(2008,8,30))
    newExposureFormat = True
    allCurrencies = True

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.APAxioma2011MH_S')
        MFM.RegionalStatisticalFactorModel.__init__(
            self, ['SEDOL'], modelDB, marketDB)
        self.setCalculators(modelDB)
        modelDB.setTotalReturnCache(367)
        modelDB.createCurrencyCache(marketDB)
        self.currencyModel = FXAxioma2010USD(modelDB, marketDB)
        l = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(logging.ERROR)
        self.baseModelDateMap = {
                datetime.date(2009,1,1): APAxioma2011MH(modelDB, marketDB),
                datetime.date(1980,1,1): APAxioma2011MH_Pre2009(modelDB, marketDB),
               }
        logging.getLogger().setLevel(l)

    def setCalculators(self, modelDB, overrider = False):
        # Set up risk parameters
        defaultStatisticalCovarianceParameters(self, scaleMinObs=0.9, overrider=overrider)
        self.returnCalculator = ReturnCalculator.AsymptoticPrincipalComponents2(self.numFactors)

class APAxioma2013MH_S(APAxioma2011MH_S):
    """Production Asia-Pacific Statistical Model
    """
    rm_id = 98
    revision = 1
    rms_id = 162

    def __init__(self, modelDB, marketDB):
        APAxioma2011MH_S.__init__(self, modelDB, marketDB)
        self.log = logging.getLogger('RiskModels.APAxioma2013MH_S')

class APAxioma2011SH_S(APAxioma2011MH_S):
    """Production Asia-Pacific Statistical Model
    """
    rm_id = 69
    revision = 1
    rms_id = 97
    newExposureFormat = True

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.APAxioma2011SH_S')
        MFM.RegionalStatisticalFactorModel.__init__(
            self, ['SEDOL'], modelDB, marketDB)
        self.setCalculators(modelDB)
        modelDB.setTotalReturnCache(367)
        modelDB.createCurrencyCache(marketDB)
        self.currencyModel = FXAxioma2010USD_SH(modelDB, marketDB)
        l = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(logging.ERROR)
        self.baseModelDateMap = {
                datetime.date(2009,1,1): APAxioma2011MH(modelDB, marketDB),
                datetime.date(1980,1,1): APAxioma2011MH_Pre2009(modelDB, marketDB),
               }
        logging.getLogger().setLevel(l)

    def setCalculators(self, modelDB, overrider = False):
        # Set up risk parameters
        defaultStatisticalCovarianceParameters(self, scaleMinObs=0.9, modelHorizon='short', overrider=overrider)
        self.returnCalculator = ReturnCalculator.AsymptoticPrincipalComponents2(self.numFactors)

class APAxioma2013SH_S(APAxioma2011SH_S):
    """Production Asia-Pacific ex-Japan Statistical Model
    """
    rm_id = 100
    revision = 1
    rms_id = 164

    def __init__(self, modelDB, marketDB):
        APAxioma2011SH_S.__init__(self, modelDB, marketDB)
        self.log = logging.getLogger('RiskModels.APAxioma2013SH_S')

class APxJPAxioma2011MH(APAxioma2011MH):
    """Production Asia-Pacific ex-Japan Fundamental Model
    """
    rm_id = 84
    revision = 2
    rms_id = 123

    def __init__(self, modelDB, marketDB):
        APAxioma2011MH.__init__(self, modelDB, marketDB)
        self.log = logging.getLogger('RiskModels.APxJPAxioma2011MH')

class APxJPAxioma2013MH(APAxioma2011MH):
    """Production Asia-Pacific ex-Japan Fundamental Model. Including the fix for A/H shares
    """
    rm_id = 93
    revision = 1
    rms_id = 157

    def __init__(self, modelDB, marketDB):
        APAxioma2011MH.__init__(self, modelDB, marketDB)
        self.log = logging.getLogger('RiskModels.APxJPAxioma2013MH')

class APxJPAxioma2011MH_Pre2009(APAxioma2011MH):
    """Production AX-APxJP2.1 (Medium Horizon) Fundamental Model
    For pre-2009 dates, does not contain some Frontier markets.
    """
    rm_id = 84
    revision = 1
    rms_id = 122

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.APxJPAxioma2011MH_Pre2009')
        APAxioma2011MH.__init__(self, modelDB, marketDB)

class APxJPAxioma2011SH(APAxioma2011SH):
    """Production Asia-Pacific ex-Japan Fundamental Model
    """
    rm_id = 70
    revision = 2
    rms_id = 99

    def __init__(self, modelDB, marketDB):
        APAxioma2011SH.__init__(self, modelDB, marketDB)
        self.log = logging.getLogger('RiskModels.APxJPAxioma2011SH')

class APxJPAxioma2013SH(APAxioma2011SH):
    """Production Asia-Pacific ex-Japan Fundamental Model. Including the fix for A/H shares
    """
    rm_id = 95
    revision = 1
    rms_id = 159

    def __init__(self, modelDB, marketDB):
        APAxioma2011SH.__init__(self, modelDB, marketDB)
        self.log = logging.getLogger('RiskModels.APxJPAxioma2013SH')

class APxJPAxioma2011SH_Pre2009(APAxioma2011SH):
    """Production AX-APxJP2.1 (Short Horizon) Fundamental Model
    For pre-2009 dates, does not contain some Frontier markets.
    """
    rm_id = 70
    revision = 1
    rms_id = 98

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.APxJPAxioma2011SH_Pre2009')
        APAxioma2011SH.__init__(self, modelDB, marketDB)

class APxJPAxioma2011MH_S(MFM.RegionalStatisticalFactorModel):
    """Production Asia-Pacific Statistical Model
    """
    rm_id = 85
    revision = 1
    rms_id = 124
    numFactors = 20
    blind = [ModelFactor('Statistical Factor %d' % n,
                         'Statistical Factor %d' % n)
             for n in range(1, numFactors+1)]
    returnHistory = 250
    industryClassification = Classification.GICSIndustryGroups(
                                        datetime.date(2008,8,30))
    newExposureFormat = True
    allCurrencies = True

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.APxJPAxioma2011MH_S')
        MFM.RegionalStatisticalFactorModel.__init__(
            self, ['SEDOL'], modelDB, marketDB)
        self.setCalculators(modelDB)
        modelDB.setTotalReturnCache(367)
        modelDB.createCurrencyCache(marketDB)
        self.currencyModel = FXAxioma2010USD(modelDB, marketDB)
        l = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(logging.ERROR)
        self.baseModelDateMap = {
                datetime.date(2009,1,1): APxJPAxioma2011MH(modelDB, marketDB),
                datetime.date(1980,1,1): APxJPAxioma2011MH_Pre2009(modelDB, marketDB),
               }
        logging.getLogger().setLevel(l)

    def setCalculators(self, modelDB, overrider = False):
        # Set up risk parameters
        defaultStatisticalCovarianceParameters(self, scaleMinObs=0.9, overrider=overrider)
        self.returnCalculator = ReturnCalculator.AsymptoticPrincipalComponents2(self.numFactors)

class APxJPAxioma2013MH_S(APxJPAxioma2011MH_S):
    """Production Asia-Pacific ex-Japan Statistical Model
    """
    rm_id = 94
    revision = 1
    rms_id = 158

    def __init__(self, modelDB, marketDB):
        APxJPAxioma2011MH_S.__init__(self, modelDB, marketDB)
        self.log = logging.getLogger('RiskModels.APxJPAxioma2013MH_S')

class APxJPAxioma2011SH_S(APxJPAxioma2011MH_S):
    """Production Asia-Pacific Statistical Model
    """
    rm_id = 71
    revision = 1
    rms_id = 100
    newExposureFormat = True

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.APxJPAxioma2011SH_S')
        MFM.RegionalStatisticalFactorModel.__init__(
            self, ['SEDOL'], modelDB, marketDB)
        self.setCalculators(modelDB)
        modelDB.setTotalReturnCache(367)
        modelDB.createCurrencyCache(marketDB)
        self.currencyModel = FXAxioma2010USD_SH(modelDB, marketDB)
        l = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(logging.ERROR)
        self.baseModelDateMap = {
                datetime.date(2009,1,1): APxJPAxioma2011MH(modelDB, marketDB),
                datetime.date(1980,1,1): APxJPAxioma2011MH_Pre2009(modelDB, marketDB),
               }
        logging.getLogger().setLevel(l)

    def setCalculators(self, modelDB, overrider = False):
        # Set up risk parameters
        defaultStatisticalCovarianceParameters(self, scaleMinObs=0.9, modelHorizon='short', overrider=overrider)
        self.returnCalculator = ReturnCalculator.AsymptoticPrincipalComponents2(self.numFactors)

class APxJPAxioma2013SH_S(APxJPAxioma2011SH_S):
    """Production Asia-Pacific ex-Japan Statistical Model
    """
    rm_id = 96
    revision = 1
    rms_id = 160

    def __init__(self, modelDB, marketDB):
        APxJPAxioma2011SH_S.__init__(self, modelDB, marketDB)
        self.log = logging.getLogger('RiskModels.APxJPAxioma2013SH_S')

class NAAxioma2011MH(MFM.RegionalFundamentalModel):
    """Production North America Fundamental Model
    """
    rm_id = 86
    revision = 1
    rms_id = 125
    styles = [
              CompositeFactor('Value', 'Value'),
              CompositeFactor('Leverage', 'Leverage'),
              CompositeFactor('Growth', 'Growth'),
              ModelFactor('Size', 'Size'),
              ModelFactor('Short-Term Momentum', 'Short-Term Momentum'),
              ModelFactor('Medium-Term Momentum', 'Medium-Term Momentum'),
              ModelFactor('Volatility', 'Volatility'),
              ModelFactor('Liquidity','Liquidity'),
              ModelFactor('Exchange Rate Sensitivity', 'Exchange Rate Sensitivity'),
             ]
    industryClassification = Classification.GICSCustomNA(
            datetime.date(2008,8,30))
    intercept = ModelFactor('North American Market', 'North American Market')
    simpleIntercept = True
    countryBetas = False
    specReturnTimingId = 1
    sensitivityNumeraire = 'XDR'
    newExposureFormat = True
    quarterlyFundamentalData = True
    allCurrencies = True

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.NAAxioma2011MH')
        MFM.RegionalFundamentalModel.__init__(self, ['SEDOL', 'CUSIP'], modelDB, marketDB)
        self.setCalculators(modelDB)
        modelDB.setTotalReturnCache(367*2)
        self.currencyModel = FXAxioma2010USD(modelDB, marketDB)
        scope = [Standardization.RegionRelativeScope(
                        modelDB, ['Value', 'Growth', 'Leverage']),
                 Standardization.GlobalRelativeScope(
                        ['Size', 'Short-Term Momentum', 'Medium-Term Momentum',
                         'Liquidity', 'Volatility', 'Exchange Rate Sensitivity'])]
        self.exposureStandardization = Standardization.\
                                    BucketizedStandardization(scope)

    def setCalculators(self, modelDB, overrider = False):
        # Set up regression parameters
        self.returnCalculator = \
                defaultRegressionParameters(self, modelDB, k_rlm=5.0, weightedRLM=False, overrider=overrider)
        # Set up risk parameters
        defaultFundamentalCovarianceParameters(self, nwLag=2, overrider=overrider)

    def generate_model_specific_exposures(self, modelDate, data, modelDB, marketDB):
        """Generate fundamental factors.
        """
        return self.generate_pd_fundamental_exposures(
                modelDate, data, modelDB, marketDB)

    def generate_md_fundamental_exposures(self, modelDate, data, modelDB, marketDB):
        return self.generate_md_fundamental_exposures_v2(modelDate, data, modelDB, marketDB)

class NAAxioma2011SH(NAAxioma2011MH):
    """Production North America Fundamental Model
    """
    rm_id = 72
    revision = 1
    rms_id = 101
    newExposureFormat = True

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.NAAxioma2011SH')
        MFM.RegionalFundamentalModel.__init__(self, ['SEDOL', 'CUSIP'], modelDB, marketDB)
        self.setCalculators(modelDB)
        modelDB.setTotalReturnCache(367*2)
        self.currencyModel = FXAxioma2010USD_SH(modelDB, marketDB)
        scope = [Standardization.RegionRelativeScope(
                        modelDB, ['Value', 'Growth', 'Leverage']),
                 Standardization.GlobalRelativeScope(
                        ['Size', 'Short-Term Momentum', 'Medium-Term Momentum',
                         'Liquidity', 'Volatility', 'Exchange Rate Sensitivity'])]
        self.exposureStandardization = Standardization.\
                                    BucketizedStandardization(scope)

    def setCalculators(self, modelDB, overrider = False):
        # Set up regression parameters
        self.returnCalculator = \
                defaultRegressionParameters(self, modelDB, k_rlm=5.0, weightedRLM=False, overrider=overrider)
        # Set up risk parameters
        defaultFundamentalCovarianceParameters(self, nwLag=2, modelHorizon='short', overrider=overrider)

class NAAxioma2011MH_S(MFM.RegionalStatisticalFactorModel):
    """Production North America Statistical Model
    """
    rm_id = 87
    revision = 1
    rms_id = 126
    numFactors = 20
    blind = [ModelFactor('Statistical Factor %d' % n,
                         'Statistical Factor %d' % n)
                         for n in range(1, numFactors+1)]
    returnHistory = 250
    newExposureFormat = True
    industryClassification = Classification.GICSCustomNA(
            datetime.date(2008,8,30))
    allCurrencies = True

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.NAAxioma2011MH_S')
        MFM.RegionalStatisticalFactorModel.__init__(
                self, ['SEDOL', 'CUSIP'], modelDB, marketDB)
        self.setCalculators(modelDB)
        modelDB.setTotalReturnCache(367)
        self.currencyModel = FXAxioma2010USD(modelDB, marketDB)
        l = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(logging.ERROR)
        self.baseModelDateMap = {
                datetime.date(1980,1,1): NAAxioma2011MH(modelDB, marketDB),
               }
        logging.getLogger().setLevel(l)

    def setCalculators(self, modelDB, overrider = False):
        # Set up risk parameters
        defaultStatisticalCovarianceParameters(self, scaleMinObs=0.9, overrider=overrider)
        self.returnCalculator = ReturnCalculator. \
                AsymptoticPrincipalComponents2(self.numFactors)

class NAAxioma2011SH_S(NAAxioma2011MH_S):
    """Production North America Statistical Model
    """
    rm_id = 73
    revision = 1
    rms_id = 102
    newExposureFormat = True

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.NAAxioma2011SH_S')
        MFM.RegionalStatisticalFactorModel.__init__(
                self, ['SEDOL', 'CUSIP'], modelDB, marketDB)
        self.setCalculators(modelDB)
        modelDB.setTotalReturnCache(367)
        self.currencyModel = FXAxioma2010USD_SH(modelDB, marketDB)
        l = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(logging.ERROR)
        self.baseModelDateMap = {
                datetime.date(1980,1,1): NAAxioma2011MH(modelDB, marketDB),
               }
        logging.getLogger().setLevel(l)

    def setCalculators(self, modelDB, overrider = False):
        # Set up risk parameters
        defaultStatisticalCovarianceParameters(self, scaleMinObs=0.9, modelHorizon='short', overrider=overrider)
        self.returnCalculator = ReturnCalculator. \
                AsymptoticPrincipalComponents2(self.numFactors)

class WWxUSAxioma2011MH(MFM.RegionalFundamentalModel):
    """Global ex US Fundamental Model
    """
    rm_id = 88
    revision = 3
    rms_id = 129
    styles = [
              CompositeFactor('Value', 'Value'),
              CompositeFactor('Leverage', 'Leverage'),
              CompositeFactor('Growth', 'Growth'),
              ModelFactor('Size', 'Size'),
              ModelFactor('Short-Term Momentum', 'Short-Term Momentum'),
              ModelFactor('Medium-Term Momentum', 'Medium-Term Momentum'),
              ModelFactor('Volatility', 'Volatility'),
              ModelFactor('Liquidity','Liquidity'),
              ModelFactor('Exchange Rate Sensitivity', 'Exchange Rate Sensitivity'),
              ModelFactor('Domestic China', 'Domestic China'),
             ]

    industryClassification = Classification.GICSIndustries(
            datetime.date(2008,8,30))
    localStructureFactors = [ModelFactor('Domestic China', 'Domestic China')]
    intercept = ModelFactor('Global Market', 'Global Market')
    allCurrencies = True
    simpleIntercept = True
    countryBetas = False
    returnsTimingId = 1
    sensitivityNumeraire = 'XDR'
    newExposureFormat = True

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.WWxUSAxioma2011MH')
        MFM.RegionalFundamentalModel.__init__(
                self, ['SEDOL', 'CUSIP'], modelDB, marketDB)
        self.setCalculators(modelDB)
        modelDB.setTotalReturnCache(367*2)
        self.currencyModel = FXAxioma2010USD(modelDB, marketDB)
        scope = [Standardization.RegionRelativeScope(
                        modelDB, ['Value', 'Growth', 'Leverage']),
                 Standardization.GlobalRelativeScope(
                        ['Size', 'Short-Term Momentum', 'Medium-Term Momentum',
                         'Liquidity', 'Volatility', 'Exchange Rate Sensitivity',
                         'Domestic China'])]
        self.exposureStandardization = Standardization.\
                                    BucketizedStandardization(scope)

    def setCalculators(self, modelDB, overrider = False):
        # Set up regression parameters
        self.returnCalculator = \
                defaultRegressionParameters(\
                self, modelDB,
                scndRegs=[[ModelFactor('Domestic China', 'Domestic China')]],
                    k_rlm=5.0, weightedRLM=False, overrider=overrider)
        # Set up risk parameters
        defaultFundamentalCovarianceParameters(self, nwLag=2, overrider=overrider)

    def generate_model_specific_exposures(self, modelDate, data, modelDB, marketDB):
        """Generate descriptor based fundamental factors and
        Domestic China style factor.
        """
        data.exposureMatrix = self.generate_md_fundamental_exposures(
                modelDate, data, modelDB, marketDB)
        data.exposureMatrix = GlobalExposures.generate_china_domestic_exposures(
                data, modelDate, modelDB, marketDB)
        return data.exposureMatrix

    def generate_residual_regression_ESTU(self, indices_ESTU, weights_ESTU, iter,
            modelDate, expM, modelDB, marketDB):
        """Returns the estimation universe assets and weights
        to be used for the given loop of the nested model regression.
        The first loop uses the 'standard' ESTU, while the second
        uses the top 300 Chinese A-shares by total market cap.
        """
        self.log.debug('generate_residual_regression_ESTU: begin')
        if iter == 1:
            self.log.info('Using default ESTU for regression loop %d', iter)
        elif iter == 2:
            (indices_ESTU, weights_ESTU) = GlobalExposures.\
                    identify_top_china_a_shares(self, modelDate, expM,
                            modelDB, marketDB, top=300)
        else:
            self.log.warning('Regression not defined for loop %d!', iter)
        self.log.debug('generate_residual_regression_ESTU: end')
        return (indices_ESTU, weights_ESTU)

    def generate_md_fundamental_exposures(self, modelDate, data, modelDB, marketDB):
        return self.generate_md_fundamental_exposures_v2(modelDate, data, modelDB, marketDB)

class WWxUSAxioma2011MH_Pre2009(WWxUSAxioma2011MH):
    """Global ex US Fundamental Model
    """
    rm_id = 88
    revision = 2
    rms_id = 128
    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.WWxUSAxioma2011MH_Pre2009')
        WWxUSAxioma2011MH.__init__(self, modelDB, marketDB)

class WWxUSAxioma2011MH_Pre2003(WWxUSAxioma2011MH):
    """Global ex US Fundamental Model
    """
    rm_id = 88
    revision = 4
    rms_id = 151
    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.WWxUSAxioma2011MH_Pre2003')
        WWxUSAxioma2011MH.__init__(self, modelDB, marketDB)

class WWxUSAxioma2011SH(WWxUSAxioma2011MH):
    """Global ex US Fundamental Model
    """
    rm_id = 74
    revision = 3
    rms_id = 105
    newExposureFormat = True

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.WWxUSAxioma2011SH')
        MFM.RegionalFundamentalModel.__init__(
                self, ['SEDOL', 'CUSIP'], modelDB, marketDB)
        self.setCalculators(modelDB)
        modelDB.setTotalReturnCache(367*2)
        self.currencyModel = FXAxioma2010USD_SH(modelDB, marketDB)
        scope = [Standardization.RegionRelativeScope(
                        modelDB, ['Value', 'Growth', 'Leverage']),
                 Standardization.GlobalRelativeScope(
                        ['Size', 'Short-Term Momentum', 'Medium-Term Momentum',
                         'Liquidity', 'Volatility', 'Exchange Rate Sensitivity',
                         'Domestic China'])]
        self.exposureStandardization = Standardization.\
                                    BucketizedStandardization(scope)

    def setCalculators(self, modelDB, overrider = False):
        # Set up regression parameters
        self.returnCalculator = \
                defaultRegressionParameters(\
                self, modelDB,
                scndRegs=[[ModelFactor('Domestic China', 'Domestic China')]],
                    k_rlm=5.0, weightedRLM=False, overrider=overrider)
        # Set up risk parameters
        defaultFundamentalCovarianceParameters(self, nwLag=2, modelHorizon='short', overrider=overrider)

class WWxUSAxioma2011SH_Pre2009(WWxUSAxioma2011SH):
    """Global ex US Fundamental Model
    """
    rm_id = 74
    revision = 2
    rms_id = 104
    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.WWxUSAxioma2011SH_Pre2009')
        WWxUSAxioma2011SH.__init__(self, modelDB, marketDB)

class WWxUSAxioma2011SH_Pre2003(WWxUSAxioma2011SH):
    """Global ex US Fundamental Model
    """
    rm_id = 74
    revision = 4
    rms_id = 152
    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.WWxUSAxioma2011SH_Pre2003')
        WWxUSAxioma2011SH.__init__(self, modelDB, marketDB)

class WWxUSAxioma2011MH_S(MFM.RegionalStatisticalFactorModel):
    """Production Global ex US Statistical Model
    """
    rm_id = 89
    revision = 1
    rms_id = 130
    numFactors = 20
    returnsTimingId = 1
    blind = [ModelFactor('Statistical Factor %d' % n,
                         'Statistical Factor %d' % n)
                         for n in range(1, numFactors+1)]
    returnHistory = 250
    newExposureFormat = True
    industryClassification = Classification.GICSIndustries(
            datetime.date(2008,8,30))
    allCurrencies = True

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.WWxUSAxioma2011MH_S')
        MFM.RegionalStatisticalFactorModel.__init__(
                self, ['SEDOL', 'CUSIP'], modelDB, marketDB)
        self.setCalculators(modelDB)
        modelDB.setTotalReturnCache(367)
        self.currencyModel = FXAxioma2010USD(modelDB, marketDB)
        l = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(logging.ERROR)
        self.baseModelDateMap = {
                datetime.date(2009,1,1): WWxUSAxioma2011MH(modelDB, marketDB),
                datetime.date(2003,1,1): WWxUSAxioma2011MH_Pre2009(modelDB, marketDB),
                datetime.date(1980,1,1): WWxUSAxioma2011MH_Pre2003(modelDB, marketDB)
               }
        logging.getLogger().setLevel(l)

    def setCalculators(self, modelDB, overrider = False):
        defaultStatisticalCovarianceParameters(self, scaleMinObs=0.9, overrider=overrider)
        self.returnCalculator = ReturnCalculator. \
                AsymptoticPrincipalComponents2(self.numFactors)

class WWxUSAxioma2011SH_S(WWxUSAxioma2011MH_S):
    """Production Global ex US Statistical Model
    """
    rm_id = 75
    revision = 1
    rms_id = 106
    newExposureFormat = True

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.WWxUSAxioma2011SH_S')
        MFM.RegionalStatisticalFactorModel.__init__(
                self, ['SEDOL', 'CUSIP'], modelDB, marketDB)
        self.setCalculators(modelDB)
        modelDB.setTotalReturnCache(367)
        self.currencyModel = FXAxioma2010USD_SH(modelDB, marketDB)
        l = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(logging.ERROR)
        self.baseModelDateMap = {
                datetime.date(2009,1,1): WWxUSAxioma2011MH(modelDB, marketDB),
                datetime.date(2003,1,1): WWxUSAxioma2011MH_Pre2009(modelDB, marketDB),
                datetime.date(1980,1,1): WWxUSAxioma2011MH_Pre2003(modelDB, marketDB)
               }
        logging.getLogger().setLevel(l)

    def setCalculators(self, modelDB, overrider = False):
        defaultStatisticalCovarianceParameters(self, scaleMinObs=0.9,
                modelHorizon='short', overrider=overrider)
        self.returnCalculator = ReturnCalculator. \
                AsymptoticPrincipalComponents2(self.numFactors)
