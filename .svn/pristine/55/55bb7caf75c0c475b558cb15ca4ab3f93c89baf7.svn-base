
import datetime
import logging
import numpy as np
import numpy
import pandas as pd
import pandas
import itertools
import numpy.ma as ma
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
from riskmodels import Utilities
from riskmodels import Outliers
######################################################################################################################################################
# fundamental model
######################################################################################################################################################
# for research only
# try:
#     import DC_YD as yd
# except:
#     import sys
#     sys.path.append('/home/ydai/cassandra/RMTtrunk/scripts/modules')
#     sys.path.append('/home/ydai/cassandra/RMTtrunk/scripts/RiskModels')
#     sys.path.append('/home/ydai/cassandra/RMTtrunk/scripts/DBTools')
#     import DC_YD as yd
class EM4_Research1(EquityModel.FundamentalModel):
    """
        EM research model - base model - added few style factors
        EM research model - base model - before release - deal with bad assets: do nothing        
    """
    rm_id,revision,rms_id = [-22,1,-220]
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

    intercept = ModelFactor('Market Intercept', 'Market Intercept') # changed from Global Market to Market Intercept
    localStructureFactors = [ModelFactor('Domestic China', 'Domestic China')]
    gicsDate = datetime.date(2016,9,1)
    # industryClassification = Classification.GICSIndustries(gicsDate)
    industryClassification = Classification.GICSCustomEM(gicsDate)

    # downweighting
    downweighting_flag = True
    # run_sectionA_flag = True # comment out section A
    # run_sectionB_flag = True # comment out section B
    # run_sectionC_flag = False # comment out section C
    # run_sectionD_flag = True # comment out section D

    DescriptorMap = {
            'Earnings Yield': ['Earnings_to_Price_Annual', 'Est_Earnings_to_Price_12MFL_Annual'],
            'Value': ['Book_to_Price_Annual'],
            'Leverage': ['Debt_to_Assets_Annual', 'Debt_to_Equity_Annual'],
            'Growth': ['Earnings_Growth_RPF_Annual', 'Sales_Growth_RPF_Annual'],
            'Dividend Yield': ['Dividend_Yield_Annual'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_60D', 'Amihud_Liquidity_125D', 'ISC_Ret_Score'],
            'Market Sensitivity': ['EM_Regional_Market_Sensitivity_500D_V0'],
            'Volatility': ['Volatility_125D'],
            'Medium-Term Momentum': ['Momentum_260x21D_Regional'],
            'Exchange Rate Sensitivity': ['XRate_104W_XDR'],
            'Profitability': ['Return_on_Equity_Annual', 'Return_on_Assets_Annual',
                'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
                'Sales_to_Assets_Annual', 'Gross_Margin_Annual'],
            }
    # EM_Regional_Market_Sensitivity_500D_V0 should be same as EM_Regional_Market_Sensitivity_500D.
    # EM_Regional_Market_Sensitivity_500D doesn't have values in research_vital yet
    DescriptorWeights = {'Earnings Yield': [0.75, 0.25],}
    exposureConfigFile = 'exposures-mh' #exposures-mh.config file
    orthogList = {'Volatility': [['Market Sensitivity'], True, 1.0],}

    # run_sectionA_flag = False # comment out section A
    # run_sectionB_flag = True # comment out section B
    # run_sectionC_flag = True # comment out section C
    # run_sectionD_flag = True # comment out section D
    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.EM4_Research1')
        ModelParameters2017.defaultModelSettings(self, scm=False)
        # Set up relevant styles to be created/used
        ModelParameters2017.defaultExposureParameters(self, self.styleList, configFile=self.exposureConfigFile, descriptorNumeraire='USD')

        EquityModel.FundamentalModel.__init__(self, ['SEDOL', 'CUSIP'], modelDB, marketDB)
        self.currencyModel = riskmodels.getModelByName('FXAxioma2017USD_MH')(modelDB, marketDB)

        # Set up estimation universe parameters
        self.estu_parameters, self.elig_parameters = ModelParameters2017.defaultRegionalModelEstuParameters(self)
        # from IPython import embed; embed(header='Debug');import ipdb;ipdb.set_trace()
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
        # default
        # ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=3)
        # # ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=3, selectDemean = False)
        # ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=3, )
        # ModelParameters2017.defaultSpecificVarianceParameters(self, )

        # new to match EM2
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=4)
        # ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=3, selectDemean = False)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=4, )
        # ModelParameters2017.defaultSpecificVarianceParameters(self,)
        ModelParameters2017.defaultSpecificVarianceParameters(self)
        self.covarianceCalculator = RiskCalculator.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

       # Set up descriptor standardization parameters
        descriptors = sorted(list(set([item for sublist
            in list(self.DescriptorMap.values()) for item in sublist])))
        regScope = Standardization.RegionRelativeScope(
                modelDB, self.regionalStndDesc)
        gloScope = Standardization.GlobalRelativeScope(
                [d for d in descriptors if d not in self.regionalStndDesc])
        self.descriptorStandardization = Standardization.BucketizedStandardization(
                [regScope, gloScope], mad_bound=15.0, fancyMAD=self.fancyMAD,
                exceptionNames=self.noStndDescriptor)

        # Set up standardization parameters
        regScope = Standardization.RegionRelativeScope(modelDB, self.regionalStndList)
        gloScope = Standardization.GlobalRelativeScope(
                [f.name for f in self.styles if f.name not in self.regionalStndList])
        self.exposureStandardization = Standardization.BucketizedStandardization(
                [regScope, gloScope], fillWithZeroList=self.fillWithZeroList)

        # Manually reassign select assets to RMG and currency
        self.tweakDict = dict()

    def generate_model_specific_exposures(self, modelDate, data, modelDB, marketDB):
        """
            Generate Domestic China local factor.
        """
        logging.info('Building Domestic China Exposures')

        buildEstu = EstimationUniverse.ConstructEstimationUniverse(data.exposureMatrix.getAssets(), self, modelDB, marketDB)
        exTypes = ['AShares', 'BShares']
        (aShares, nonest) = buildEstu.exclude_by_asset_type(modelDate, data, includeFields=exTypes, excludeFields=None)
        # (aShares1, nonest) = buildEstu.exclude_by_asset_type(modelDate, data, includeFields=['AShares'], excludeFields=None)
        # (aShares2, nonest) = buildEstu.exclude_by_asset_type(modelDate, data, includeFields=['BShares'], excludeFields=None)
        values = Matrices.allMasked(len(data.universe))
        if len(aShares) > 0:
            logging.info('Assigning Domestic China exposure to %d assets', len(aShares))
            ma.put(values, aShares, 1.0)
        data.exposureMatrix.addFactor('Domestic China', values, ExposureMatrix.LocalFactor)

        return data.exposureMatrix

    # def generateExposureMatrix(self, modelDate, modelDB, marketDB):
    #     """Generates and returns the exposure matrix for the given date.
    #     The exposures are not inserted into the database.
    #     Data is accessed through the modelDB and marketDB DAOs.
    #     The return is a structure containing the exposure matrix
    #     (exposureMatrix), the universe as a list of assets (universe),
    #     and a list of market capitalizations (marketCaps).
    #     """
    #     self.log.debug('generateExposureMatrix: begin')
    #
    #     # Get risk model universe and market caps
    #     # Determine home country info and flag DR-like instruments
    #     rmi = modelDB.getRiskModelInstance(self.rms_id, modelDate)
    #     universe = modelDB.getRiskModelInstanceUniverse(rmi, returnExtra=True)
    #     data = AssetProcessor.process_asset_information(
    #             modelDate, universe, self.rmg, modelDB, marketDB,
    #             checkHomeCountry=self.multiCountry,
    #             numeraire_id=self.numeraire.currency_id,
    #             legacyDates=self.legacyMCapDates,
    #             forceRun=self.forceRun,
    #             nurseryRMGList=self.nurseryRMGs,
    #             tweakDict=self.tweakDict)
    #     data.exposureMatrix = Matrices.ExposureMatrix(data.universe)
    #
    #     if (not self.multiCountry) and (not hasattr(self, 'indexSelector')):
    #         self.indexSelector = MarketIndex.MarketIndexSelector(modelDB, marketDB)
    #         self.log.info('Index Selector: %s', self.indexSelector)
    #
    #     # Generate eligible universe
    #     data.eligibleUniverse = self.generate_eligible_universe(
    #             modelDate, data, modelDB, marketDB)
    #
    #     # Fetch trading calendars for all risk model groups
    #     # Start-date should depend on how long a history is required
    #     # for exposures computation
    #     data.rmgCalendarMap = dict()
    #     startDate = modelDate - datetime.timedelta(365*2)
    #     for rmg in self.rmg:
    #         data.rmgCalendarMap[rmg.rmg_id] = \
    #                 modelDB.getDateRange(rmg, startDate, modelDate)
    #
    #     # Compute issuer-level market caps if required
    #     AssetProcessor.computeTotalIssuerMarketCaps(
    #             data, modelDate, self.numeraire, modelDB, marketDB,
    #             debugReport=self.debuggingReporting)
    #
    #     if self.multiCountry:
    #         self.generate_binary_country_exposures(modelDate, modelDB, marketDB, data)
    #         self.generate_currency_exposures(modelDate, modelDB, marketDB, data)
    #
    #     # Generate 0/1 industry exposures
    #     self.generate_industry_exposures(
    #         modelDate, modelDB, marketDB, data.exposureMatrix)
    #
    #     # Load estimation universe
    #     estu = self.loadEstimationUniverse(rmi, modelDB, data)
    #
    #     # Create intercept factor
    #     if self.intercept is not None:
    #         beta = numpy.ones((len(data.universe)), float)
    #         data.exposureMatrix.addFactor(self.intercept.name, beta, ExposureMatrix.InterceptFactor)
    #
    #     # Build all style exposures
    #     descriptorData = self.generate_style_exposures(modelDate, data, modelDB, marketDB)
    #
    #     # Shrink some values where there is insufficient history
    #     for st in self.styles:
    #         params = self.styleParameters.get(st.name, None)
    #         if (params is None) or (not  hasattr(params, 'shrinkValue')):
    #             continue
    #         fIdx = data.exposureMatrix.getFactorIndex(st.name)
    #         values = data.exposureMatrix.getMatrix()[fIdx]
    #         # Check and warn of missing values
    #         missingIdx = numpy.flatnonzero(ma.getmaskarray(values))
    #         if len(missingIdx) > 0:
    #             missingSIDs = numpy.take(data.universe, missingIdx, axis=0)
    #             missingSIDs_notnursery = list(set(missingSIDs).difference(data.nurseryUniverse))
    #             self.log.warning('%d assets have missing %s data', len(missingIdx), st.description)
    #             self.log.warning('%d non-nursery assets have missing %s data', len(missingSIDs_notnursery), st.description)
    #             self.log.info('Subissues: %s', missingSIDs)
    #             if (len(missingSIDs_notnursery) > 5) and not self.forceRun:
    #                 yd1 = yd.DC_YD()
    #                 fac_types = yd1.get_fac_type(data.exposureMatrix.factors_)
    #                 cntry_factors = [x for x,y in zip(data.exposureMatrix.factors_,fac_types) if y == 'Country']
    #                 cntry_expo = data.exposureMatrix.toDataFrame().reindex(missingSIDs_notnursery)[cntry_factors]
    #                 missingSIDs_cntry_expo = [row_val.T.dropna().index[0] for sub_obj,row_val in cntry_expo.iterrows()]
    #                 nurseryCountries = [x.name for x in self.nurseryCountries]
    #                 if modelDate > datetime.date(2011,4,1):
    #                     nurseryCountries = nurseryCountries + ['Israel']
    #                 real_missingSIDs_notnursery = [x for x in missingSIDs_cntry_expo if x not in nurseryCountries]
    #                 if (len(real_missingSIDs_notnursery) > 5):
    #                     print ('There are %d assets missing data.' % len(real_missingSIDs_notnursery))
    #                     # assert (len(real_missingSIDs_notnursery)==0)
    #                     # sub_meta2 = yd1.get_subissue_meta2([x.getSubIDString() for x in real_missingSIDs_notnursery],modelDate)
    #                     from IPython import embed; embed(header='Debug');import ipdb;ipdb.set_trace()
    #
    #         testNew = False
    #         if self.regionalDescriptorStructure and testNew:
    #             shrunkValues = self.shrink_to_mean(modelDate, data, modelDB, marketDB,
    #                     st.name, params.daysBack, values, missingIdx, onlyIPOs=False)
    #         else:
    #             shrunkValues = self.shrink_to_mean(modelDate, data, modelDB, marketDB,
    #                     st.name, params.daysBack, values, missingIdx)
    #         data.exposureMatrix.getMatrix()[fIdx] = shrunkValues
    #
    #     # Clone DR and cross-listing exposures if required
    #     scores = self.load_ISC_Scores(modelDate, data, modelDB, marketDB)
    #     self.group_linked_assets(modelDate, data, modelDB, marketDB)
    #     data.exposureMatrix = self.clone_linked_asset_exposures(
    #             modelDate, data, modelDB, marketDB, scores)
    #
    #     if self.debuggingReporting:
    #         dr_indices = [data.assetIdxMap[sid] for sid in data.foreign]
    #         data.exposureMatrix.dumpToFile('tmp/raw-expM-%s-%04d%02d%02d.csv'\
    #                 % (self.name, modelDate.year, modelDate.month, modelDate.day),
    #                 modelDB, marketDB, modelDate, estu=data.estimationUniverseIdx, assetData=data, dp=self.dplace)
    #
    #     tmpDebug = self.debuggingReporting
    #     self.debuggingReporting = False
    #     self.standardizeExposures(data.exposureMatrix, data, modelDate, modelDB, marketDB, data.subIssueGroups)
    #
    #     # Orthogonalise where required
    #     orthogDict = dict()
    #     for st in self.styles:
    #         params = self.styleParameters[st.name]
    #         if hasattr(params, 'orthog'):
    #             if not hasattr(params, 'sqrtWt'):
    #                 params.sqrtWt = True
    #             if not hasattr(params, 'orthogCoef'):
    #                 params.orthogCoef = 1.0
    #             if params.orthog is not None and len(params.orthog) > 0:
    #                 orthogDict[st.name] = (params.orthog, params.orthogCoef, params.sqrtWt)
    #
    #     if len(orthogDict) > 0:
    #         Utilities.partial_orthogonalisation(modelDate, data, modelDB, marketDB, orthogDict)
    #         tmpExcNames = list(self.exposureStandardization.exceptionNames)
    #         self.exposureStandardization.exceptionNames = [st.name for st in self.styles if st.name not in orthogDict]
    #         self.standardizeExposures(data.exposureMatrix, data, modelDate,
    #                     modelDB, marketDB, data.subIssueGroups)
    #         self.exposureStandardization.exceptionNames = tmpExcNames
    #     self.debuggingReporting = tmpDebug
    #
    #     expMatrix = data.exposureMatrix.getMatrix()
    #     fail = False
    #
    #     for st in self.styles:
    #         params = self.styleParameters[st.name]
    #         # Here we have two parameters that do essentially the same thing
    #         # 'fillWithZero' is intended to cover items like Dividend Yield, where a large number
    #         # of observations are genuinely missing
    #         # 'fillMissing' is a failsafe for exposures that shouldn't normally have any missing values,
    #         # but given the vagaries of global data, may have some from time to time
    #         if (hasattr(params, 'fillWithZero') and (params.fillWithZero is True)) or \
    #                 (hasattr(params, 'fillMissing') and (params.fillMissing is True)):
    #             fIdx = data.exposureMatrix.getFactorIndex(st.name)
    #             for scope in self.exposureStandardization.factorScopes:
    #                 if st.name in scope.factorNames:
    #                     for (bucket, assetIndices) in scope.getAssetIndices(data.exposureMatrix, modelDate):
    #                         values = expMatrix[fIdx, assetIndices]
    #                         nMissing = numpy.flatnonzero(ma.getmaskarray(values))
    #                         if len(nMissing) > 0:
    #                             denom = ma.filled(data.exposureMatrix.stdDict[bucket][st.name], 0.0)
    #                             if abs(denom) > 1.0e-6:
    #                                 fillValue = (0.0 - data.exposureMatrix.meanDict[bucket][st.name]) / denom
    #                                 expMatrix[fIdx,assetIndices] = ma.filled(values, fillValue)
    #                                 logging.info('Filling %d missing values for %s with standardised zero: %.2f for region %s',
    #                                         len(nMissing), st.name, fillValue, bucket)
    #                             else:
    #                                 logging.warning('Zero/missing standard deviation %s for %s for region %s',
    #                                     data.exposureMatrix.stdDict[bucket][st.name], st.name, bucket)
    #
    #     if self.debuggingReporting:
    #         dr_indices = [data.assetIdxMap[sid] for sid in data.foreign]
    #         data.exposureMatrix.dumpToFile('tmp/expM-%s-%04d%02d%02d.csv'\
    #                 % (self.name, modelDate.year, modelDate.month, modelDate.day),
    #                 modelDB, marketDB, modelDate, estu=data.estimationUniverseIdx, assetData=data, dp=self.dplace)
    #
    #     # Check for exposures with all missing values
    #     for st in self.styles:
    #         fIdx = data.exposureMatrix.getFactorIndex(st.name)
    #         values = Utilities.screen_data(expMatrix[fIdx,:])
    #         missingIdx = numpy.flatnonzero(ma.getmaskarray(values))
    #         if len(missingIdx) > 0:
    #             self.log.warning('Style factor %s has %d missing exposures', st.name, len(missingIdx))
    #         nonMissingIdx = numpy.flatnonzero(ma.getmaskarray(values)==0)
    #         if len(nonMissingIdx) < 1:
    #             self.log.error('All %s values are missing', st.description)
    #             if not self.forceRun:
    #                 assert(len(nonMissingIdx)>0)
    #
    #     self.log.debug('generateExposureMatrix: end')
    #     # from IPython import embed; embed(header='Debug: Exposure end');import ipdb;ipdb.set_trace()
    #     return [data, descriptorData]

    def clone_linked_asset_exposures(self, date, data, modelDB, marketDB, scoreDict):
        return self.clone_linked_asset_exposures_new(date, data, modelDB, marketDB, scoreDict,
                commonList=self.wideCloneList)

    def proxy_missing_exposures(self, modelDate, data, modelDB, marketDB,
            factorNames=['Value', 'Leverage', 'Growth'], clip=True, sizeVec=None, kappa=5.0):
        return self.proxy_missing_exposures_new(modelDate, data, modelDB, marketDB,
                factorNames=factorNames, clip=clip, sizeVec=sizeVec, kappa=kappa)

class EM4_Research2(EM4_Research1):
    """
        EM research model - FX sensitivity use XDR or USD (retired)
        EM research model - regional-based style momentum factor
        EM research model - for research different methods
        EM research model - keep 98% MarketCap
    """
    rm_id,revision,rms_id = [-22,2,-221]
    # old: to study regional-based style momentum factors
    # styleList = ['Value',
    #          'Leverage',
    #          'Growth',
    #          'Profitability',
    #          'Earnings Yield',
    #          'Dividend Yield',
    #          'Size',
    #          'Liquidity',
    #          'Market Sensitivity',
    #          'Volatility',
    #          'Medium-Term Momentum Europe',
    #          'Medium-Term Momentum Asia exPacific',
    #          'Medium-Term Momentum Africa',
    #          'Medium-Term Momentum Middle East',
    #          'Medium-Term Momentum Latin America',
    #          'Exchange Rate Sensitivity',
    #          ]
    # DescriptorMap = {
    #         'Earnings Yield': ['Earnings_to_Price_Annual', 'Est_Earnings_to_Price_12MFL_Annual'],
    #         'Value': ['Book_to_Price_Annual'],
    #         'Leverage': ['Debt_to_Assets_Annual', 'Debt_to_Equity_Annual'],
    #         'Growth': ['Earnings_Growth_RPF_Annual', 'Sales_Growth_RPF_Annual'],
    #         'Dividend Yield': ['Dividend_Yield_Annual'],
    #         'Size': ['LnIssuerCap'],
    #         'Liquidity': ['LnTrading_Activity_60D', 'Amihud_Liquidity_125D', 'ISC_Ret_Score'],
    #         #'Liquidity': ['LnTrading_Activity_60D'],
    #         'Market Sensitivity': ['EM_Regional_Market_Sensitivity_500D'],
    #         #'Emerging Market Sensitivity': ['EM_Market_Sensitivity_500D'],
    #         'Volatility': ['Volatility_125D'],
    #         'Medium-Term Momentum Europe': ['Momentum_250x20D'],
    #         'Medium-Term Momentum Asia exPacific': ['Momentum_250x20D'],
    #         'Medium-Term Momentum Africa': ['Momentum_250x20D'],
    #         'Medium-Term Momentum Middle East': ['Momentum_250x20D'],
    #         'Medium-Term Momentum Latin America': ['Momentum_250x20D'],
    #         'Exchange Rate Sensitivity': ['XRate_104W_XDR'], #'XRate_104W_USD',XRate_104W_XDR
    #         'Interest Rate Sensitivity': ['XRate_104W_IR'],
    #         'Profitability': ['Return_on_Equity_Annual', 'Return_on_Assets_Annual',
    #             'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
    #             'Sales_to_Assets_Annual', 'Gross_Margin_Annual'],
    #         #Payout': ['Net_Equity_Issuance', 'Net_Debt_Issuance', 'Net_Payout_Over_Profits'],
    #         }
    # regionalStndList = ['Earnings Yield', 'Value', 'Leverage', 'Growth', 'Profitability', 'Dividend Yield','Medium-Term Momentum Europe',
    #                     'Medium-Term Momentum Asia exPacific','Medium-Term Momentum Africa','Medium-Term Momentum Middle East','Medium-Term Momentum Latin America']# 'Payout']
    # regionalStndDesc = list(itertools.chain.from_iterable([DescriptorMap[st] for st in regionalStndList]))

    industryClassification = Classification.GICSCustomEM(datetime.date(2016,9,1))

    DescriptorMap = {
            'Earnings Yield': ['Earnings_to_Price_Annual', 'Est_Earnings_to_Price_12MFL_Annual'],
            'Value': ['Book_to_Price_Annual'],
            'Leverage': ['Debt_to_Assets_Annual', 'Debt_to_Equity_Annual'],
            'Growth': ['Earnings_Growth_RPF_Annual', 'Sales_Growth_RPF_Annual'],
            'Dividend Yield': ['Dividend_Yield_Annual'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_60D', 'Amihud_Liquidity_125D', 'ISC_Ret_Score'],
            'Market Sensitivity': ['EM_Regional_Market_Sensitivity_500D_V0'],
            'Volatility': ['Volatility_125D'],
            'Medium-Term Momentum': ['Momentum_260x21D_Regional'],
            'Exchange Rate Sensitivity': ['XRate_104W_XDR'],
            'Profitability': ['Return_on_Equity_Annual', 'Return_on_Assets_Annual',
                'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
                'Sales_to_Assets_Annual', 'Gross_Margin_Annual'],
            }
    ##############################################################################################
    YD_Version_ESTU = False
    YD_Version_EXP = False
    # exposureConfigFile = 'exposures-mh2.config'
    YD_Version_Factor = False
    YD_Version_Risk = False
    dropZeroNanAsset = False
    run_sectionA_flag = False # comment out section A
    run_sectionB_flag = False # comment out section B
    run_sectionC_flag = False # comment out section C
    run_sectionD_flag = False # comment out section D (it should be true, it was False before)
    ##############################################################################################
    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.EM4_Research2')
        super(EM4_Research2,self).__init__(modelDB, marketDB)
        ############################################ change ESTU Setting ############################################
        self.estu_parameters = {'minNonZero': [0.95, 0.05, 0.05],
                           'minNonMissing': [0.95, 0.05, 0.05],
                           'ChinaATol': 0.9,
                           'maskZeroWithNoADV_Flag': True,
                           'CapByNumber_Flag': False,
                           'CapByNumber_hiCapQuota': np.nan,
                           'CapByNumber_lowCapQuota': np.nan,
                           'market_lower_pctile': 2.0,
                           'country_lower_pctile': 2.0,
                           'industry_lower_pctile': 2.0,
                           'dummyThreshold': 10,
                           'inflation_cutoff': 0.05,
                          }
        ####################################################################################################################################
    def generate_model_universe_YD(self, modelDate, modelDB, marketDB):

        self.log.info('[YD]generate_model_universe from_BenchMark: begin')
        data = Utilities.Struct()
        universe = AssetProcessor.getModelAssetMaster(
                 self, modelDate, modelDB, marketDB, legacyDates=self.legacyMCapDates)
        # Throw out any assets with missing size descriptors
        descDict = dict(modelDB.getAllDescriptors())
        sizeVec = self.loadDescriptorData(['LnIssuerCap'], descDict, modelDate,
                        universe, modelDB, None, rollOver=False)[0]['LnIssuerCap']
        missingSizeIdx = numpy.flatnonzero(ma.getmaskarray(sizeVec))
        if len(missingSizeIdx) > 0:
            missingIds = numpy.take(universe, missingSizeIdx, axis=0)
            universe = list(set(universe).difference(set(missingIds)))
            logging.warning('%d missing LnIssuerCaps dropped from model', len(missingSizeIdx))
        data.universe = universe

        # load Benchmark info
        import DC_YD as yd
        yd1 = yd.DC_YD() # use research_vital
        bmname = 'Russell Emerging'
        if modelDate < datetime.date(2000,1,3):
            bm_dts = sorted(yd1.dc.getBenchmarkDates(bmname))
            if modelDate <= bm_dts[0]:
                bmname_dt = bm_dts[0]
            else:
                bmname_dt = modelDate
                for d0,d1 in zip(bm_dts[:-1],bm_dts[1:]):
                    if modelDate < d1:
                        bmname_dt = d0
                        break
        else:
            bmname_dt = modelDate
        bm_assets = yd1.dc.getBenchmark(bmname_dt,bmname)
        tmp_bmname_dt = bmname_dt
        s = 0
        while(len(bm_assets)==0):
            tmp_bmname_dt = tmp_bmname_dt - datetime.timedelta(1)
            s = s+1
            bm_assets = yd1.dc.getBenchmark(tmp_bmname_dt,bmname)
            if s>30:
                raise Exception('No Benchmarch data for Russell Emerging.')
        sub_ids = yd1.ID2SubID(bm_assets.index, bmname_dt, type='AxiomaID')
        sub_ids_objs = yd1.get_subissue_objs(sub_ids)
        estu1 = list(set(universe).intersection(sub_ids_objs))
        self.estuMap['main'].assets = estu1
        self.estuMap['main'].qualify = estu1
        # for China A
        bmname = 'CSI 300'
        if modelDate < datetime.date(2004,12,31):
            bmname_dt = datetime.date(2004,12,31)
        else:
            bmname_dt = modelDate
        bm_assets = yd1.dc.getBenchmark(bmname_dt,bmname)
        tmp_bmname_dt = bmname_dt
        s = 0
        while(len(bm_assets)==0):
            tmp_bmname_dt = tmp_bmname_dt - datetime.timedelta(1)
            s = s+1
            bm_assets = yd1.dc.getBenchmark(tmp_bmname_dt,bmname)
            if s>30:
                raise Exception('No Benchmarch data for CSI 300.')
        # from IPython import embed; embed(header='Debug');import ipdb;ipdb.set_trace()
        sub_ids = yd1.ID2SubID(bm_assets.index, bmname_dt, type='AxiomaID')
        sub_ids_objs = yd1.get_subissue_objs(sub_ids)
        self.log.info('[YD]there are %d ChinaA assets before intersection.' % len(sub_ids_objs))
        estu2 = list(set(universe).intersection(sub_ids_objs))
        self.estuMap['ChinaA'].assets = estu2
        self.estuMap['ChinaA'].qualify = estu2
        self.log.info('[YD]generate_model_universe: finish')
        self.log.info('[YD]generate_model_universe from_BenchMark: main ESTU: %d' % len(estu1))
        self.log.info('[YD]generate_model_universe from_BenchMark: ChinaA ESTU: %d' % len(estu2))
        # from IPython import embed; embed(header='Debug:[YD]ESTU END');import ipdb;ipdb.set_trace()
        return data

    def generate_model_universe(self, modelDate, modelDB, marketDB):
        if self.YD_Version_ESTU:
            data = self.generate_model_universe_YD(modelDate, modelDB, marketDB)
        else:
            data = super(EM4_Research2,self).generate_model_universe(modelDate,modelDB, marketDB)
        return data

    def generateExposureMatrix_RegExp(self, modelDate, modelDB, marketDB):
        print('EM4_Research2================================================================================')
        data = super(EM4_Research2,self).generateExposureMatrix(modelDate, modelDB, marketDB)
        # from IPython import embed; embed(header='Debug');import ipdb;ipdb.set_trace()
        reg_fac_dict = {'Europe':'Medium-Term Momentum Europe','Asia ex-Pacific':'Medium-Term Momentum Asia exPacific','Africa':'Medium-Term Momentum Africa','Middle East':'Medium-Term Momentum Middle East','Latin America':'Medium-Term Momentum Latin America'}
        tmpdata = data[0]
        reg_scopes = self.descriptorStandardization.factorScopes[0]
        for (bucketDesc, assetIndices) in reg_scopes.getAssetIndices(tmpdata.exposureMatrix, modelDate):
            if bucketDesc in reg_fac_dict.keys():
                fac_name = reg_fac_dict[bucketDesc]
                print(fac_name)
                fac_idx = tmpdata.exposureMatrix.getFactorIndex(fac_name)
                non_asset_idx = list(set(range(len(tmpdata.exposureMatrix.assets_))).difference(assetIndices))
                tmpdata.exposureMatrix.data_[fac_idx,non_asset_idx]=0
        data[0] = tmpdata
        return data

    def generateExposureMatrix(self, modelDate, modelDB, marketDB):
        if self.YD_Version_EXP:
            data = self.generateExposureMatrix_YD(modelDate, modelDB, marketDB)
        else:
            data = super(EM4_Research2,self).generateExposureMatrix(modelDate, modelDB, marketDB)
        # from IPython import embed; embed(header='Debug:generateExposureMatrix EM4_V2');import ipdb;ipdb.set_trace()
        # out1 = data[0].exposureMatrix.toDataFrame()
        # estu_out1 = out1.reindex(np.take(data[0].universe,data[0].estimationUniverseIdx))
        # estu_out1.index = [x.getSubIDString() for x in estu_out1.index]
        # yd1 = yd.DC_YD()
        # fac_dict = yd1.get_fac_type(estu_out1.columns,'dict')
        # estu_out_dict = {}
        # for key in fac_dict.keys():
        #     estu_out_dict[key] = estu_out1[fac_dict[key]]
        # output_filename = '/home/ydai/Project/2017_result_v4/EM4_R2/R2_Expsure_YD.xlsx'
        # yd1.output_result(estu_out_dict, output_filename)
        #
        # output_filename1 = '/home/ydai/Project/2017_result_v4/EM4_R2/R2_Expsure_Default.xlsx'
        # Default_expo = yd1.read_result(output_filename1)
        # YD_expo = estu_out_dict
        # comb_out_dict = {}
        # for key in fac_dict.keys():
        #     tmp_df = pd.concat([YD_expo[key],Default_expo[key]],axis=1,keys=['YD','Default'])
        #     comb_out_dict[key] = tmp_df.swaplevel(axis=1).sort_index(axis=1)
        # output_filename2 = '/home/ydai/Project/2017_result_v4/EM4_R2/R2_Expsure_Combo.xlsx'
        # yd1.output_result(comb_out_dict, output_filename2)
        return data

    def compute_stand_expo(self,factorScopes,expo_df,mkt_cap,cntry_expo,exp_bound,fill_fac_list=None,sel_factors=None):
        '''
        
        :param factorScopes: from either descriptorStandardization or exposureStandardization
        :param expo_df: exposure matrix
        :param cntry_expo_df2: country exposure matrix to identify region membership
        :param mkt_cap: as weight for Standardization, used for selecting estu and calculation std
        :return: 
        
        factorScopes = self.descriptorStandardization.factorScopes
        factor_scope = factorScopes[0]
        '''
        stand_expo_stats_dict = {}
        style_expo_df2 = expo_df.copy()
        estu_sub_ids = list(mkt_cap.index)
        for factor_scope in factorScopes:
            if factor_scope.description == 'Global-Relative':
                fac_names = factor_scope.factorNames
                if sel_factors is not None:
                    fac_names = list(set(sel_factors).intersection(sel_factors))
                if len(fac_names)==0:
                    continue
                sub_df = expo_df[fac_names].copy()
                mean_std_univ_weight = mkt_cap['mkt_cap']/ mkt_cap['mkt_cap'].sum()
                standed_sub_df,tmp_dict,weight_used = yd.stand_exposure(sub_df,mean_std_univ_weight,exp_bound,fill_fac_list)
                max_abs_mean = (standed_sub_df.reindex(weight_used.index)*(weight_used)).sum().abs().max()
                if max_abs_mean>(10**(-1)):
                    # from IPython import embed; embed(header='Debug:compute_stand_expo');import ipdb;ipdb.set_trace()
                    # raise Exception('result from standardization does not match up!')
                    print(('result from standardization does not match up! max_abs_mean = %f' % max_abs_mean))
                style_expo_df2.loc[:,fac_names] = standed_sub_df
                tmp_df = pd.DataFrame(tmp_dict,index=['Mean','Std']).T
                stand_expo_stats_dict['Universe'] = tmp_df
            elif factor_scope.description == 'Region-Relative':
                region_map = factor_scope.regionCountryMap
                fac_names = factor_scope.factorNames
                if sel_factors is not None:
                    fac_names = list(set(sel_factors).intersection(sel_factors))
                if len(fac_names)==0:
                    continue
                for one_region,rg_cntry_list in region_map.items():
                    # one_region = 'Latin America'
                    # rg_cntry_list=region_map[one_region]
                    sel_cntries = list(set(cntry_expo.columns).intersection(rg_cntry_list))
                    sel_assets = list(cntry_expo[sel_cntries].stack().index.get_level_values(0))
                    sub_df = style_expo_df2.loc[sel_assets,fac_names].copy()
                    if len(sub_df)>0:
                        mean_std_univ = list(set(sel_assets).intersection(estu_sub_ids))
                        mean_std_univ_weight = mkt_cap['mkt_cap'].reindex(mean_std_univ)
                        standed_sub_df,tmp_dict,weight_used = yd.stand_exposure(sub_df,mean_std_univ_weight,exp_bound,fill_fac_list)
                        # from IPython import embed; embed(header='Debug:compute_stand_expo');import ipdb;ipdb.set_trace()
                        # print(one_region)
                        max_abs_mean = (standed_sub_df.reindex(weight_used.index)*(weight_used)).sum().abs().max()
                        if max_abs_mean>(10**(-1)):
                            # from IPython import embed; embed(header='Debug:compute_stand_expo');import ipdb;ipdb.set_trace()
                            # raise Exception('result from standardization does not match up! max_abs_mean = %f' % max_abs_mean)
                            print(('result from standardization does not match up! max_abs_mean = %f' % max_abs_mean))
                        style_expo_df2.loc[sel_assets,fac_names] = standed_sub_df
                        tmp_df = pd.DataFrame(tmp_dict,index=['Mean','Std']).T
                        stand_expo_stats_dict[one_region] = tmp_df
            else:
                raise Exception('Unknown factor standardization scope')
            if (standed_sub_df.max().max() > exp_bound) or (standed_sub_df.min().min() < (-exp_bound)):
                from IPython import embed; embed(header='Debug:compute_stand_expo2');import ipdb;ipdb.set_trace()
        stand_expo_stats_df = pd.concat(list(stand_expo_stats_dict.values()),axis=1,keys=list(stand_expo_stats_dict.keys()))
        # clipped is done within yd.stand_exposure
        # style_expo_df2[style_expo_df2>exp_bound] = exp_bound
        # style_expo_df2[style_expo_df2<-exp_bound] = -exp_bound
        return style_expo_df2,stand_expo_stats_df

    def convert_exposure_df_to_expM(self,exposure_df,universe):
        '''
            covnert the exposure in dataframe to ExposureMatrix instance
        :param exposure_df: 
        :param universe: output order should be same as universe
        :return: 
        '''
        import DC_YD as yd
        yd1 = yd.DC_YD()
        fac_type_dict = yd1.get_fac_type(exposure_df.columns,'dict')
        fac_type_obj_dict = {'Market':ExposureMatrix.InterceptFactor, 'Local':ExposureMatrix.LocalFactor,
                             'Country':ExposureMatrix.CountryFactor, 'Industry':ExposureMatrix.IndustryFactor,
                             'Currency':ExposureMatrix.CurrencyFactor, 'Style':ExposureMatrix.StyleFactor}
        exposureMatrix = Matrices.ExposureMatrix(universe)
        mdl_universe = [x.getSubIDString() for x in universe]
        exposure_df = exposure_df.reindex(mdl_universe)
        # prepare data.exposureMatrix
        raw_list = ['Market', 'Local', 'Country', 'Industry', 'Currency', 'Style']
        type_list = [x for x in raw_list if x in fac_type_dict.keys()]
        for fac_type in type_list:
            factor_names = fac_type_dict[fac_type]
            values = np.ma.array(exposure_df[factor_names],mask=exposure_df[factor_names].isnull()).T
            fac_type_obj = fac_type_obj_dict[fac_type]
            exposureMatrix.addFactors(factor_names, values, fac_type_obj)
        return exposureMatrix

    def generateExposureMatrix_YD(self, modelDate, modelDB, marketDB):
        """Generates and returns the exposure matrix for the given date.
        The exposures are not inserted into the database.
        Data is accessed through the modelDB and marketDB DAOs.
        The return is a structure containing the exposure matrix
        (exposureMatrix), the universe as a list of assets (universe),
        and a list of market capitalizations (marketCaps).
        """
        import pandas as pd
        self.log.info('generateExposureMatrix: begin')
        # Get risk model universe and market caps
        # Determine home country info and flag DR-like instruments
        # from IPython import embed; embed(header='Debug:generateExposureMatrix_YD:Begin');import ipdb;ipdb.set_trace()
        rmi = modelDB.getRiskModelInstance(self.rms_id, modelDate)
        universe = modelDB.getRiskModelInstanceUniverse(rmi, returnExtra=True)
        data = AssetProcessor.process_asset_information(
                modelDate, universe, self.rmg, modelDB, marketDB,
                checkHomeCountry=self.multiCountry,
                numeraire_id=self.numeraire.currency_id,
                legacyDates=self.legacyMCapDates,
                forceRun=self.forceRun,
                nurseryRMGList=self.nurseryRMGs,
                tweakDict=self.tweakDict)
        data.exposureMatrix = Matrices.ExposureMatrix(data.universe)

        mdl_universe = [x.getSubIDString() for x in data.universe]
        # Fetch trading calendars for all risk model groups
        # Start-date should depend on how long a history is required
        # for exposures computation
        data.rmgCalendarMap = dict()
        startDate = modelDate - datetime.timedelta(365*2)
        for rmg in self.rmg:
            data.rmgCalendarMap[rmg.rmg_id] = modelDB.getDateRange(rmg, startDate, modelDate)

        # Compute issuer-level market caps if required
        AssetProcessor.computeTotalIssuerMarketCaps(
                data, modelDate, self.numeraire, modelDB, marketDB,
                debugReport=self.debuggingReporting)
        # issuer_totalMktCaps = pd.DataFrame(data.issuerTotalMarketCaps,index=mdl_universe,columns=['IssuerMktCap'])
        ##############################################################################################################################################
        # generate country exposure
        self.log.info('generate country exposures: begin')
        cntry_expo_list =[]
        for rmg_id, rmg_assets in data.rmgAssetMap.items():
            tmp_sub_ids = [x.getSubIDString() for x in list(rmg_assets)]
            tmp_df = pd.DataFrame([rmg_id]*len(tmp_sub_ids),index=tmp_sub_ids,columns=['RMG_ID'])
            cntry_expo_list.append(tmp_df)
        cntry_expo_df = pd.concat(cntry_expo_list,axis=0)
        cntry_expo_df.index.name = 'SubID'
        cntry_expo_df = cntry_expo_df.reset_index()
        cntry_expo_df['Value'] = 1.0
        cntry_expo_df2 = cntry_expo_df.pivot(index='SubID', columns='RMG_ID', values='Value')
        # change from rmg_id to rmg_name, which is country factor name
        rmg_id_dict = dict((x.rmg_id,x.description) for x in self.rmg)
        cntry_expo_df2.columns = [rmg_id_dict[x] for x in cntry_expo_df2.columns]
        missing_cntry_list = list(set(rmg_id_dict.values()).difference(cntry_expo_df2.columns))
        for missing_cntry in missing_cntry_list:
            cntry_expo_df2[missing_cntry] = np.nan
        cntry_expo_df2 = cntry_expo_df2.sort_index(axis=1)
        # from IPython import embed; embed(header='Debug:generateExposureMatrix_YD0');import ipdb;ipdb.set_trace()
        ##############################################################################################################################################
        # generate currency exposure
        self.log.info('generate currency exposures: begin')
        # self.generate_currency_exposures(modelDate, modelDB, marketDB, data)
        currency_expo_list =[]
        for rmg_id, rmc_assets in data.rmcAssetMap.items():
            tmp_sub_ids = [x.getSubIDString() for x in list(rmc_assets)]
            tmp_df = pd.DataFrame([rmg_id]*len(tmp_sub_ids),index=tmp_sub_ids,columns=['RMG_ID'])
            currency_expo_list.append(tmp_df)
        currency_expo_df = pd.concat(currency_expo_list,axis=0)
        currency_expo_df.index.name = 'SubID'
        currency_expo_df = currency_expo_df.reset_index()
        currency_expo_df['Value'] = 1.0
        currency_expo_df2 = currency_expo_df.pivot(index='SubID', columns='RMG_ID', values='Value')
        # change from rmg_id to rmg_currency_code, which is currency factor name
        rmg_id_currency_dict = dict((x.rmg_id,x.currency_code) for x in self.rmg)
        currency_expo_df2.columns = [rmg_id_currency_dict[x] for x in currency_expo_df2.columns]
        # get all currency factors: adding the hidden currency factors
        all_currency = [x.name for x in (self.currencies + self.hiddenCurrencies)]
        missing_currency = list(set(all_currency).difference(currency_expo_df2.columns))
        for each_cur in missing_currency:
            currency_expo_df2[each_cur] = np.nan
        currency_expo_df2 = currency_expo_df2[all_currency]
        # mutiple rmg_id can be mapped to one curreny:
        from collections import Counter
        cnt1 = Counter(currency_expo_df2.columns)
        muti_currency_list = [x for x,y in cnt1.items() if y>1]
        for sel_cur in muti_currency_list:
            tmp_df = currency_expo_df2[sel_cur].sum(axis=1)
            tmp_df[tmp_df==0] = np.nan
            currency_expo_df2 = currency_expo_df2.drop(sel_cur,axis=1)
            currency_expo_df2[sel_cur] = tmp_df
        currency_expo_df2 = currency_expo_df2.sort_index(axis=1)
        ##############################################################################################################################################
        # self.generate_industry_exposures(modelDate, modelDB, marketDB, data.exposureMatrix)
        self.log.info('generate industry exposures: begin')
        factor_list = [f.description for f in self.industryClassification.getLeafNodes(modelDB).values()]
        ind_exposures = self.industryClassification.getExposures(modelDate, data.universe, factor_list, modelDB)
        ind_exposure_df = pd.DataFrame(ind_exposures,columns=mdl_universe,index=factor_list).T
        # check cntry_expo, currency_expo and industry_expo:
        # print(cntry_expo_df2.sum().sort_values())
        # print(ind_exposure_df.sum().sort_values())
        # print(currency_expo_df2.sum().sort_values())
        ##############################################################################################################################################
        market_expo_df = pd.DataFrame([1]*len(mdl_universe),index=mdl_universe,columns=[self.intercept.name])
        # Build all style exposures
        data.eligibleUniverse = self.generate_eligible_universe(modelDate, data, modelDB, marketDB) # elig_idx is used for Standardizing raw descriptors
        estu = self.loadEstimationUniverse(rmi, modelDB, data) # estu_idx is used for Standardizing raw descriptors
        # descriptorData = self.generate_style_exposures(modelDate, data, modelDB, marketDB)
        # Compute style exposures
        self.log.info('generate style exposures: begin')
        # Get list of all descriptors needed
        descriptors = sorted([x for sublist in self.DescriptorMap.values() for x in sublist])
        # Map descriptor names to their IDs
        desc_dict = dict(modelDB.getAllDescriptors())

        # Pull out a proxy for size, isn't this same as issuer_mkt_cap? this is used for proxy missing assets
        # sizeVec = self.loadDescriptorData(['LnIssuerCap'], desc_dict, modelDate, data.universe, modelDB, data.currencyAssetMap, rollOver=False)[0]['LnIssuerCap']
        # size_df = pd.DataFrame(sizeVec,index=mdl_universe,columns=['LnIssuerCap'])
        # tmp = pd.concat([size_df,np.log(issuer_totalMktCaps)],axis=1)

        # Load the descriptor data
        descValueDict, okDescriptorCoverageMap = self.loadDescriptorData(
                        descriptors, desc_dict, modelDate, data.universe,
                        modelDB, data.currencyAssetMap,
                        rollOver=(self.regionalDescriptorStructure==False))
        desc_value_df = pd.DataFrame(list(descValueDict.values()),index = list(descValueDict.keys()),columns=mdl_universe).T
        for col,col_val in desc_value_df.items():
            col_val[col_val.apply(np.ma.is_masked)] = np.nan
            desc_value_df[col] = col_val

        for col,col_val in desc_value_df.items():
            desc_value_df[col] = pd.to_numeric(col_val)

        desc_cover_pcntg = desc_value_df.count() / len(desc_value_df)
        # Check that each exposure has at least one descriptor with decent coverage
        expo_cover_pcntg = []
        for key, value in self.DescriptorMap.items():
            expo_cover_pcntg.append([key,desc_cover_pcntg[value].max()])
        expo_cover_pcntg_df = pd.DataFrame(expo_cover_pcntg,columns=['Desc_Name','Cover_Pcntg'])
        print(expo_cover_pcntg_df)
        ##################################################
        # Clone DR and cross-listing exposures if required
        ISC_scores_dict = self.load_ISC_Scores(modelDate, data, modelDB, marketDB) #Issuer Specific Covariance
        clone_descs = [n for n in desc_value_df.columns if n not in self.noCloneDescriptor]
        # Loop round sets of linked assets and pull out exposures
        for (groupId, subIssueList) in data.subIssueGroups.items():
            tmp_sub_ids = [x.getSubIDString() for x in subIssueList]
            tmp_df = desc_value_df.reindex(tmp_sub_ids)[clone_descs]
            indices  = [data.assetIdxMap[n] for n in subIssueList]
            tmp_score = ISC_scores_dict[groupId]
            score = pd.DataFrame([tmp_score]*len(clone_descs),index=clone_descs,columns=tmp_sub_ids).T
            sel_main_datasource = ((1-tmp_df.isnull())*score).apply(np.argmax)
            tmp_df2 = tmp_df.copy()
            for col,col_val in tmp_df.items():
                # col_val[col_val.isnull()] = col_val[sel_main_datasource[col]]
                # col_val = col_val[sel_main_datasource[col]]
                tmp_df2[col] = col_val[sel_main_datasource[col]]
            desc_value_df.loc[tmp_sub_ids,clone_descs] = tmp_df2
        ##########################################################
        # Standardize raw descriptors for multi-descriptor factors
        # self.descriptorStandardization.standardize(descriptorExposures, data.estimationUniverseIdx,
        #         data.marketCaps, modelDate, writeStats=True, eligibleEstu=data.eligibleUniverseIdx)
        # clip the descriptor values
        mat_bound = self.descriptorStandardization.mad_bound
        zero_tolerance = self.descriptorStandardization.zero_tolerance
        OutlierParameters = {'nBounds':[mat_bound, mat_bound],'zeroTolerance':zero_tolerance}
        outlierClass = Outliers.Outliers(OutlierParameters)
        desc_value_arr = np.array(desc_value_df)
        desc_value_arr_clipped = outlierClass.twodMAD(desc_value_arr, axis=0, estu=None)
        desc_value_df_clipped = pd.DataFrame(desc_value_arr_clipped,index=desc_value_df.index,columns=desc_value_df.columns)

        estu_sub_ids = [x.getSubIDString() for x in estu]
        mdl_universe = [x.getSubIDString() for x in data.universe]
        # Standardized descriptors based on ESTU
        # estu_desc = desc_value_df_clipped.reindex(estu_sub_ids)
        # compute mkt_cap mean and equal-weighted std
        estu_mkt_cap = pd.DataFrame(data.marketCaps,index=mdl_universe,columns=['mkt_cap']).reindex(estu_sub_ids)

        # Standardized the desc values (by region or by global based on factorScopes)
        stand_desc_value,desc_stats_df = self.compute_stand_expo(self.descriptorStandardization.factorScopes,desc_value_df_clipped,estu_mkt_cap,cntry_expo_df2,mat_bound,fill_fac_list=None)
        # desc_stats_df.swaplevel(axis=1)['Std'].sort_index(axis=1)
        # from IPython import embed; embed(header='Debug:generateExposureMatrix_YD:Begin');import ipdb;ipdb.set_trace()
        # Save descriptor standardisation stats to the DB (not interested to save these data)
        ##############################################################################################################################################
        # Form multi-descriptor CompositeFactors and add to ExposureMatrix
        style_expo_mat = []
        for style in self.styleList:
            needed_descs = self.DescriptorMap[style]
            sel_desc_values = stand_desc_value[needed_descs]

            if hasattr(self.styleParameters[style], 'descriptorWeights'):
                desc_weights_list = self.styleParameters[style].descriptorWeights
            else:
                desc_weights_list = [1.0/len(needed_descs)]*len(needed_descs)
            # expo_arr = ma.average(np.array(sel_desc_values), axis=1, weights=desc_weights_list)
            # expo_df = pd.DataFrame(expo_arr,index=sel_desc_values.index,columns=[style]) # it takes the intersection of available values
            desc_weights_df = pd.DataFrame([desc_weights_list]*len(sel_desc_values),index = sel_desc_values.index,columns=sel_desc_values.columns)
            desc_weights_df[sel_desc_values.isnull()] = np.nan
            desc_weights_sum = pd.concat([desc_weights_df.sum(axis=1)]*desc_weights_df.shape[1],axis=1,keys=sel_desc_values.columns)
            desc_weights_df2 = desc_weights_df/desc_weights_sum
            expo_df = (sel_desc_values * desc_weights_df2).sum(axis=1)
            expo_df[sel_desc_values.isnull().all(axis=1)] = np.nan
            style_expo_mat.append(expo_df.to_frame(style))
        style_expo_df = pd.concat(style_expo_mat,axis=1)
        self.log.debug('generate_style_exposures: end')
        # Proxy raw style exposures for assets missing data (not sure how it will help the factor estimation)
        # self.proxy_missing_exposures(modelDate, data, modelDB, marketDB, factorNames=[cf.description for cf in self.styles], sizeVec=sizeVec)
        ##############################################################################################################################################
        # Generate other, model-specific factor exposures: China Factor
        # data.exposureMatrix = self.generate_model_specific_exposures(modelDate, data, modelDB, marketDB)
        logging.info('Building Domestic China Exposures')
        buildEstu = EstimationUniverse.ConstructEstimationUniverse(data.universe, self, modelDB, marketDB)
        (aShares_idx, nonest) = buildEstu.exclude_by_asset_type(modelDate, data, includeFields=['AShares', 'BShares'], excludeFields=None)
        sel_sud_ids = [x.getSubIDString() for x in np.take(data.universe,aShares_idx)]
        ChinaA_expo = pd.DataFrame([np.nan]*len(mdl_universe),index=mdl_universe,columns=['Domestic China'])
        ChinaA_expo.loc[sel_sud_ids,'Domestic China'] = 1
        self.log.debug('generate_ChinaA_exposures: end')
        ###################################################################################################################################
        #  Shrink some values where there is insufficient history - for example, IPO stock
        for st in self.styles:
            params = self.styleParameters.get(st.name, None)
            if (params is not None) and (hasattr(params, 'shrinkValue')):
                # shrunkValues = self.shrink_to_mean(modelDate, data, modelDB, marketDB,st.name, params.daysBack, values, missingIdx)
                if params.shrinkValue:
                    self.log.debug('shrink_to_mean for factor %s: begin' % st.name)
                    st_value = style_expo_df[st.name]
                    # Load up from dates and determine scaling factor based on how far it is from
                    logging.info('Shrinking based on IPO-date')
                    fromDates = modelDB.loadIssueFromDates([modelDate],  data.universe)
                    distance = ma.array([int((modelDate - dt).days) for dt in fromDates], int)
                    distance_sa = pd.Series(distance,index=mdl_universe)
                    distance_sa[distance_sa>params.daysBack] = params.daysBack
                    scale_factor_ipo = distance_sa/params.daysBack
                    needed_scale_factor_ipo = scale_factor_ipo[scale_factor_ipo<1.0]
                    # Get cap weights to be used for mean
                    # estuArray = numpy.zeros((len(data.universe)), float)
                    # numpy.put(estuArray, data.estimationUniverseIdx, 1.0)
                    # mcapsEstu = ma.filled(data.marketCaps, 0.0) * estuArray
                    # fillHist = ma.filled(ma.masked_where(scaleFactor<1.0, scaleFactor), 0.0)
                    # mcapsEstu *= fillHist

                    # Get sector/industry group exposures
                    level = 'Sectors' #'Industry Groups'
                    sectorExposures = Utilities.buildGICSExposures(data.universe, modelDate, modelDB, level=level, clsDate=self.gicsDate)
                    # industryClassification = Classification.GICSIndustries(self.gicsDate)
                    # parents = industryClassification.getClassificationParents(level, modelDB)
                    # factorList = [f.description for f in parents]
                    sectorExposures_df = pd.DataFrame(sectorExposures,index=mdl_universe)
                    # Bucket assets into regions/countries
                    # regionIDMap = dict()
                    # regionAssetMap = dict()
                    # for r in self.rmg:
                    #     rmg_assets = data.rmgAssetMap[r.rmg_id]
                    #     if r.region_id not in regionAssetMap:
                    #         regionAssetMap[r.region_id] = list()
                    #         regionIDMap[r.region_id] = 'Region %d' % r.region_id
                    #     regionAssetMap[r.region_id].extend(rmg_assets)
                    # Compute mean of entire estimation universe to be used if insufficient
                    # values in any region/sector bucket
                    estu_scale = scale_factor_ipo.reindex(estu_sub_ids)
                    sel_estu_scale = estu_scale[estu_scale==1.0]
                    sel_st_value = st_value.reindex(sel_estu_scale.index).dropna()
                    tmp_weight = estu_mkt_cap['mkt_cap'].reindex(sel_st_value.index)
                    tmp_weight = tmp_weight/tmp_weight.sum()
                    global_mean = sel_st_value.dot(tmp_weight)
                    # Loop round countries/regions and sector
                    sectorExposures_df2 = sectorExposures_df.stack()
                    sectorExposures_df2 = sectorExposures_df2.reset_index()
                    sectorExposures_df2.columns=['SubID','Sector_ID','Value']
                    sub_meta = pd.concat([cntry_expo_df.set_index('SubID'),sectorExposures_df2.set_index('SubID'),sel_estu_scale.to_frame('Sel_ESTU'),estu_mkt_cap,scale_factor_ipo.to_frame('scale_factor'),st_value],axis=1)
                    sub_meta2 = sub_meta.reindex(needed_scale_factor_ipo.index)
                    sub_meta3 = sub_meta2[['RMG_ID','Sector_ID']].drop_duplicates()
                    sub_meta_gps = sub_meta.dropna(subset=['RMG_ID','Sector_ID']).groupby(['RMG_ID','Sector_ID'])
                    gp_mean = {}
                    for row_id,row_val in sub_meta3.iterrows():
                        gp_name = (row_val['RMG_ID'],row_val['Sector_ID'])
                        if any(np.isnan(gp_name)):
                            gp_mean[gp_name] = global_mean
                        else:
                            sel_gp_val = sub_meta_gps.get_group(gp_name)
                            tmp_df = sel_gp_val[sel_gp_val['Sel_ESTU']==1]
                            if len(tmp_df)>=2:
                                tmp_weights = tmp_df['mkt_cap']/(tmp_df['mkt_cap'].sum())
                                gp_mean[gp_name] = tmp_df[st.name].dot(tmp_weights)
                            else:
                                gp_mean[gp_name] = global_mean
                    gp_mean_df = pd.DataFrame(list(gp_mean.values()),index=pd.MultiIndex.from_tuples(list(gp_mean.keys()))).reset_index()
                    gp_mean_df.columns=['RMG_ID','Sector_ID','Group_Mean']
                    sub_meta2 = sub_meta2.reset_index().merge(gp_mean_df,how='left',on=['RMG_ID','Sector_ID'])
                    sub_meta2 = sub_meta2.set_index('index')
                # Shrink relevant values
                logging.info('Shrinking %d values of %s', len(sub_meta2), st.name)
                new_values = sub_meta2[st.name]*sub_meta2['scale_factor'] + sub_meta2['Group_Mean']*(1-sub_meta2['scale_factor'])
                self.log.debug('shrink_to_mean for factor %s: end' % st.name)
                style_expo_df.loc[new_values.index,st.name] = new_values
        #######################################################
        # Clone DR and cross-listing exposures if required
        # self.group_linked_assets(modelDate, data, modelDB, marketDB)
        # data.exposureMatrix = self.clone_linked_asset_exposures(modelDate, data, modelDB, marketDB, ISC_scores_dict)
        # self.log.debug('group_linked_assets: begin')
        # sub_groups_list = []
        # for key,value in data.subIssueGroups.items():
        #     tmp_df = pd.DataFrame(value,columns=['SubID'])
        #     tmp_df['Comp_ID'] = key
        #     sub_groups_list.append(tmp_df)
        # sub_groups_df = pd.concat(sub_groups_list,axis=0)
        # sub_groups_df['Asset_Type'] = [data.assetTypeDict[x] for x in sub_groups_df['SubID']]
        # main_assetType_clusters = self.commonStockTypes + self.otherAllowedStockTypes +\
        #                           self.fundAssetTypes + self.intlChineseAssetTypes +\
        #                           self.otherAllowedStockTypes + self.drAssetTypes
        # sub_groups_df.index = [x.getSubIDString() for x in sub_groups_df['SubID']]
        # sub_groups_df = sub_groups_df.drop('SubID',axis=1)
        # sel_sub_groups_df = sub_groups_df[sub_groups_df['Asset_Type'].isin(main_assetType_clusters)]
        # s = 0
        # for gp_name,gp_val in sel_sub_groups_df.groupby('Comp_ID'):
        #     if s==100:
        #         break
        #     s = s+1
        #     style_expo_df.loc[gp_val.index]
        #######################################################################################################
        # self.standardizeExposures(data.exposureMatrix, data, modelDate, modelDB, marketDB, data.subIssueGroups)
        exp_bound = self.exposureStandardization.exp_bound
        # fill in missing factor list
        fill_fac_list = []
        for st in self.styles:
            params = self.styleParameters[st.name]
            if (hasattr(params, 'fillWithZero') and (params.fillWithZero is True)) or (hasattr(params, 'fillMissing') and (params.fillMissing is True)):
                fill_fac_list.append(st.name)
        logging.info('standardization factor exposures.')
        style_expo_df2,expo_stats_df = self.compute_stand_expo(self.exposureStandardization.factorScopes,style_expo_df,estu_mkt_cap,cntry_expo_df2,exp_bound,fill_fac_list)
        ###############################
        # check style_expo:
        # tmp_df = style_expo_df2.reindex(estu_sub_ids).sort_index(axis=0)
        # weight = estu_mkt_cap/estu_mkt_cap.sum()
        # print(tmp_df.T.dot(weight)) #there is readjustment in weight for each columns. it won't match up
        # expo_stats_df.swaplevel(axis=1)['Std'].sort_index(axis=1)
        # from IPython import embed; embed(header='Debug:generateExposureMatrix_YD:Begin');import ipdb;ipdb.set_trace()
        ##############################
        # for factor_scope in self.exposureStandardization.factorScopes:
        #     if factor_scope.description == 'Global-Relative':
        #         fac_names = factor_scope.factorNames
        #         sub_df = style_expo_df[fac_names].copy()
        #         mean_std_univ_weight = mkt_cap['mkt_cap']
        #         standed_sub_df,tmp_dict = yd.stand_exposure(sub_df,mean_std_univ_weight,exp_bound,fill_fac_list)
        #         style_expo_df2.loc[:,fac_names] = standed_sub_df
        #         stand_expo_dict.update(tmp_dict)
        #     elif factor_scope.description == 'Region-Relative':
        #         region_map = factor_scope.regionCountryMap
        #         fac_names = factor_scope.factorNames
        #         for one_region,rg_cntry_list in region_map.items():
        #             sel_cntries = list(set(cntry_expo_df2.columns).intersection(rg_cntry_list))
        #             sel_assets = list(cntry_expo_df2[sel_cntries].stack().index.get_level_values(0))
        #             sub_df = style_expo_df.loc[sel_assets,fac_names].copy()
        #             mean_std_univ = list(set(sel_assets).intersection(estu_sub_ids))
        #             mean_std_univ_weight = mkt_cap['mkt_cap'].reindex(mean_std_univ)
        #             standed_sub_df,tmp_dict = yd.stand_exposure(sub_df,mean_std_univ_weight,exp_bound,fill_fac_list)
        #             style_expo_df2.loc[sel_assets,fac_names] = standed_sub_df
        #             stand_expo_dict.update(tmp_dict)
        #     else:
        #         raise Exception('Unknown factor standardization scope')
        ########################################################################################################
        # Orthogonalise where required
        orthogDict = self.orthogList # params.orthog, params.sqrtWt, params.orthogCoef
        # if len(orthogDict) > 0:
        #     Utilities.partial_orthogonalisation(modelDate, data, modelDB, marketDB, orthogDict)
        #     tmpExcNames = list(self.exposureStandardization.exceptionNames)
        #     self.exposureStandardization.exceptionNames = [st.name for st in self.styles if st.name not in orthogDict]
        #     self.standardizeExposures(data.exposureMatrix, data, modelDate,modelDB, marketDB, data.subIssueGroups)
        #     self.exposureStandardization.exceptionNames = tmpExcNames
        import statsmodels.api as sm
        if len(orthogDict) > 0:
            for key,val in orthogDict.items():
                sel_Y = style_expo_df2[key].reindex(estu_sub_ids)
                sel_X = style_expo_df2[val[0]].reindex(estu_sub_ids)
                sel_Y = sel_Y.dropna()
                sel_X = sel_X.reindex(sel_Y.index)
                weight = np.sqrt(estu_mkt_cap).reindex(sel_Y.index)
                reg1 = sm.WLS(sel_Y, sel_X, weights=weight).fit()
                new_val = style_expo_df2[key] - style_expo_df2[val[0]].dot(reg1.params)
                if new_val.count() == 0:
                    raise Exception('After orthog, no valid number.')
                # standed_new_val, tmp_dict, weight_used = yd.stand_exposure(new_val.to_frame(key),estu_mkt_cap,exp_bound)
                # standed_new_val.reindex(weight_used.index).T.dot(weight_used)
                style_expo_df2[key] = new_val
            # standardizing orthogalized factors
            sel_facs = list(orthogDict.keys())
            style_expo_df3,expo_stats_df2 = self.compute_stand_expo(self.exposureStandardization.factorScopes,style_expo_df2,estu_mkt_cap,cntry_expo_df2,exp_bound,fill_fac_list,sel_facs)
            expo_stats_df.loc[expo_stats_df2.index,expo_stats_df2.columns] = expo_stats_df2
            # from IPython import embed; embed(header='Debug:orthogDict');import ipdb;ipdb.set_trace()
        ########################################################################################################
        exposure_df = pd.concat([market_expo_df,ChinaA_expo.fillna(0),cntry_expo_df2.fillna(0),ind_exposure_df.fillna(0),currency_expo_df2,style_expo_df3],axis=1)
        self.log.debug('generateExposureMatrix: end')
        ############################################
        # compute VIF for a given factor
        # yd1 = yd.DC_YD()
        # fac_types = yd1.get_fac_type(exposure_df.columns,'dict')
        # sel_Y = exposure_df['Volatility'].reindex(estu_sub_ids)
        # sel_X = exposure_df.drop(['Volatility']+fac_types['Currency'],axis=1).reindex(estu_sub_ids)
        # weight = np.sqrt(estu_mkt_cap).reindex(estu_sub_ids)
        # tmp_vif_result = yd1.compute_VIF(sel_Y,sel_X.fillna(0),weight)
        # check each factor
        # vif_result = []
        # for sel_cntry in fac_types['Country']:
        #     sel_X = exposure_df[fac_types['Style']+fac_types['Industry']+[sel_cntry]].drop(['Volatility'],axis=1).reindex(estu_sub_ids)
        #     weight = np.sqrt(estu_mkt_cap).reindex(estu_sub_ids)
        #     reg2 = sm.WLS(sel_Y, sel_X.fillna(0), weights=weight).fit()
        #     VIF = 1.0/(1.0-reg2.rsquared)
        #     vif_result.append([sel_cntry,VIF])
        # vif_result_df = pd.DataFrame(vif_result,columns=['Country','VIF']).sort_values('VIF')
        # tmp1 = exposure_df[['Volatility','Kuwait']].reindex(estu_sub_ids).dropna()
        # tmp_w = estu_mkt_cap.reindex(tmp1.index)
        # tmp2 = pd.concat([tmp1,tmp_w/tmp_w.sum()],axis=1)
        ############################################
        exposure_df_corr = exposure_df.corr()
        import DC_YD as yd
        yd1 = yd.DC_YD()
        fac_types = yd1.get_fac_type(exposure_df.columns,'dict')
        sel_corr = yd1.take_upper_matrix(exposure_df_corr.loc[fac_types['Style'],fac_types['Style']])
        sel_corr_df = yd1.cat_df_axis(sel_corr.stack().abs(),'_VS_')
        sel_corr_df.columns=['Abs(corr)']
        print((sel_corr_df.sort_values('Abs(corr)',ascending=False).head(10)))
        ########################################################################################################
        # prepare output
        # yd1 = yd.DC_YD()
        # fac_type_dict = yd1.get_fac_type(exposure_df.columns,'dict')
        # fac_type_obj_dict = {'Market':ExposureMatrix.InterceptFactor, 'Local':ExposureMatrix.LocalFactor,
        #                      'Country':ExposureMatrix.CountryFactor, 'Industry':ExposureMatrix.IndustryFactor,
        #                      'Currency':ExposureMatrix.CurrencyFactor, 'Style':ExposureMatrix.StyleFactor}
        # data.exposureMatrix = Matrices.ExposureMatrix(data.universe)
        # exposure_df = exposure_df.reindex(mdl_universe)
        # # prepare data.exposureMatrix
        # for fac_type in ['Market', 'Local', 'Country', 'Industry', 'Currency', 'Style']:
        #     factor_names = fac_type_dict[fac_type]
        #     values = np.ma.array(exposure_df[factor_names],mask=exposure_df[factor_names].isnull()).T
        #     fac_type_obj = fac_type_obj_dict[fac_type]
        #     data.exposureMatrix.addFactors(factor_names, values, fac_type_obj)
        data.exposureMatrix = self.convert_exposure_df_to_expM(exposure_df,data.universe)
        # prepare descriptor statistics data
        descriptorData = Utilities.Struct()
        descriptorData.descriptors = descriptors
        descriptorData.descDict = desc_dict
        descriptorData.meanDict = dict() # dict of dict
        descriptorData.stdDict = dict()
        descriptorData.DescriptorWeights = dict()

        for key,val in desc_stats_df.items():
            if key[1] == 'Mean':
                descriptorData.meanDict[key[0]] = val.dropna().to_dict()
            elif key[1] == 'Std':
                descriptorData.stdDict[key[0]] = val.dropna().to_dict()
            else:
                raise Exception('Unexcepted Statistics Name: %s' % key[1])
        from IPython import embed; embed(header='Debug:generateExposureMatrix_YD1');import ipdb;ipdb.set_trace()
        return [data, descriptorData]

    def generateFactorSpecificReturns(self, modelDB, marketDB, modelDate,buildFMPs=False, internalRun=False, cointTest=False,weeklyRun=False):
        if self.YD_Version_Factor:
            data = self.generateFactorSpecificReturns_YD(modelDB, marketDB,modelDate,buildFMPs,internalRun,cointTest,weeklyRun)
        #     data = super(EM4_Research2,self).generateFactorSpecificReturns(modelDB, marketDB, modelDate, buildFMPs,internalRun,cointTest,weeklyRun)
        else:
            data = super(EM4_Research2,self).generateFactorSpecificReturns(modelDB, marketDB, modelDate, buildFMPs,internalRun,cointTest,weeklyRun)
        return data

    def generateFactorSpecificReturns_YD(self, modelDB, marketDB, modelDate,buildFMPs=False, internalRun=False, cointTest=False,weeklyRun=False):
        """Generates the factor and specific returns for the given
        date.  Assumes that the factor exposures for the previous
        trading day exist as those will be used for the exposures.
        Returns a Struct with factorReturns, specificReturns, exposureMatrix,
        regressionStatistics, and adjRsquared.  The returns are 
        arrays matching the factor and assets in the exposure matrix.
        exposureMatrix is an ExposureMatrix object.  The regression
        coefficients is a two-dimensional masked array
        where the first dimension is the number of factors used in the
        regression, and the second is the number of statistics for
        each factor that are stored in the array.
        """
        # from IPython import embed; embed(header='Debug:generateFactorSpecificReturns_YD');import ipdb;ipdb.set_trace()
        # Testing parameters for FMPs
        # testFMPs = False
        # nextTradDate = None

        # Important regression parameters
        # Set up parameters for factor returns regression - previous trading date
        dateList = modelDB.getDates(self.rmg, modelDate, 1, excludeWeekend=True)
        if len(dateList) < 2:
            raise LookupError('No previous trading day for %s' %  str(modelDate))
        prevDate = dateList[0]

        if internalRun:
            logging.info('Generating internal factor returns')
            # Regression for internal factor returns
            if not hasattr(self, 'internalCalculator'):
                logging.warning('No internal factor return parameters set up, skipping')
                return None
            rcClass = self.internalCalculator # return calculator class
            if self.multiCountry:
                applyRT = True # apply return timing
            else:
                applyRT = False
        elif weeklyRun:
            logging.info('Generating weekly factor returns from %s to %s',
                    prevDate, modelDate)
            # Regression for weekly factor returns
            if not hasattr(self, 'weeklyCalculator'):
                logging.warning('No weekly factor return parameters set up, skipping')
                return None
            rcClass = self.weeklyCalculator # return calculator class
            applyRT = False
        else:
            # Regression for public factor returns
            logging.info('Generating external (public) factor returns')
            rcClass = self.returnCalculator # return calculator class
            applyRT = False

        #####################################################################################################
        # set factor date as previous date to get exposure and factor names
        self.setFactorsForDate(prevDate, modelDB)
        # Get exposure matrix for previous trading day
        rmi = modelDB.getRiskModelInstance(self.rms_id, prevDate)
        if rmi == None:
            raise LookupError('no risk model instance for %s' % str(prevDate))
        if not rmi.has_exposures:
            raise LookupError('no exposures in risk model instance for %s' % str(prevDate))

        # Determine home country info and flag DR-like instruments
        universe = modelDB.getRiskModelInstanceUniverse(rmi, returnExtra=True)
        data = AssetProcessor.process_asset_information(
                prevDate, universe, self.rmg, modelDB, marketDB,
                checkHomeCountry=self.multiCountry, numeraire_id=self.numeraire.currency_id,
                legacyDates=self.legacyMCapDates,
                forceRun=self.forceRun, nurseryRMGList=self.nurseryRMGs,
                tweakDict=self.tweakDict)

        # Load previous day's exposure matrix
        expM = self.loadExposureMatrix(rmi, modelDB, addExtraCountries=True, assetList=data.universe)
        prevFactors = self.factors + self.nurseryCountries
        prevSubFactors = modelDB.getSubFactorsForDate(prevDate, prevFactors)
        nameSubIDMap = dict([(s.factor.name, s.subFactorID) for s in prevSubFactors])

        # Get main estimation universe for previous day
        estu = self.loadEstimationUniverse(rmi, modelDB, data)
        if 'nursery' in self.estuMap:
            logging.info('Adding %d nursery assets to main estimation universe',len(self.estuMap['nursery'].assets))
            estu = estu + self.estuMap['nursery'].assets
            self.estuMap['main'].assets = estu
            logging.info('Main estimation universe now %d assets', len(estu))
        estuIdx = [data.assetIdxMap[sid] for sid in estu]
        self.estuMap['main'].assetIdx = estuIdx
        data.estimationUniverseIdx = estuIdx
        estuMap = self.estuMap
        #####################################################################################################
        # Get map of current day's factor IDs
        self.setFactorsForDate(modelDate, modelDB) #overwrite estuMap
        allFactors = self.factors + self.nurseryCountries
        subFactors = modelDB.getSubFactorsForDate(modelDate, allFactors)
        subFactorIDIdxMap = dict([(subFactors[i].subFactorID, i) for i in range(len(subFactors))])
        deadFactorNames = [s.factor.name for s in prevSubFactors if s not in subFactors]
        deadFactorIdx = [expM.getFactorIndex(n) for n in deadFactorNames]
        if len(deadFactorIdx) > 0:
            self.log.warning('Dropped factors %s on %s', deadFactorNames, modelDate)
        rmi = modelDB.getRiskModelInstance(self.rms_id, modelDate)

        # Load asset returns
        if internalRun:
            assetReturnMatrix = self.assetReturnHistoryLoader(data, 10, modelDate, modelDB, marketDB, loadOnly=True,applyRT=applyRT, fixNonTradingDays=True)
        else:
            assetReturnMatrix = self.assetReturnHistoryLoader(data, 2, modelDate, modelDB, marketDB, loadOnly=True,applyRT=applyRT, fixNonTradingDays=False)
        assetReturnMatrix.data = assetReturnMatrix.data[:,-1][:,numpy.newaxis]
        assetReturnMatrix.dates = [assetReturnMatrix.dates[-1]]

        missingReturnsMask = assetReturnMatrix.missingFlag
        zeroReturnsMask = assetReturnMatrix.zeroFlag
        missingReturnsMask = missingReturnsMask[:,-1]
        zeroReturnsMask = zeroReturnsMask[:,-1]
        # Do some checking on missing and zero returns
        # missingReturnsIdx = numpy.flatnonzero(missingReturnsMask)
        # logging.info('%d out of %d genuine missing returns in total', len(missingReturnsIdx), len(missingReturnsMask))
        # zeroReturnsIdx = numpy.flatnonzero(zeroReturnsMask)
        # logging.info('%d out of %d zero returns in total', len(zeroReturnsIdx), len(zeroReturnsMask))
        stillMissingReturnsIdx = numpy.flatnonzero(ma.getmaskarray(assetReturnMatrix.data))
        if len(stillMissingReturnsIdx) > 0:
            logging.info('%d missing total returns filled with zero', len(stillMissingReturnsIdx))
        asset_rets_df = assetReturnMatrix.toDataFrame()
        asset_rets_df.index = [x.getSubIDString() for x in asset_rets_df.index]
        ###############################################################################
        # deal with missing asset returns: drop or fillin 0
        # assetReturnMatrix.data = ma.filled(assetReturnMatrix.data, 0.0)
        # chinaA_estu = [x.getSubIDString() for x in estuMap['ChinaA'].assets]
        # asset_rets_df.reindex(chinaA_estu)
        drop_NAZero_Flag = True
        if drop_NAZero_Flag:
            asset_rets_df.iloc[np.flatnonzero(missingReturnsMask),:] = np.nan
            asset_rets_df.iloc[np.flatnonzero(zeroReturnsMask),:] = np.nan
            n_asset0 = len(asset_rets_df)
            asset_rets_df = asset_rets_df.dropna()
            n_asset1 = len(asset_rets_df)
            self.log.info('There are %d assets now , dropping from %d assets' % (n_asset1,n_asset0))
        else:
            asset_rets_df = asset_rets_df.fillna(0)
        # Do some checking on estimation universe returns
        # Report on missing returns for ESTU
        # missingESTURets = numpy.flatnonzero(ma.take(missingReturnsMask, estuIdx, axis=0))
        # badRetList = list()
        # badRetList.extend(numpy.take(data.universe, missingESTURets, axis=0))
        # propnBadRets = len(missingESTURets) / float(len(estuIdx))
        # self.log.info('%.1f%% of %d main ESTU original returns missing', 100.0*propnBadRets, len(estuIdx))
        # # Report on zero returns
        # zeroESTURets = numpy.flatnonzero(ma.take(zeroReturnsMask, estuIdx, axis=0))
        # badRetList.extend(numpy.take(data.universe, zeroESTURets, axis=0))
        # propnBadRets = len(zeroESTURets) / float(len(estuIdx))
        # self.log.info('%.1f%% of %d main ESTU final returns zero', 100.0*propnBadRets, len(estuIdx))
        # if self.run_sectionA_flag:
        #     if internalRun:
        #         self.badRetList = list(set(badRetList))
        #     else:
        #         self.badRetList = list()
        # # If too many of both, set t-stats to be nuked
        # propnBadRets = (len(zeroESTURets) + len(missingESTURets)) / float(len(estuIdx))
        # if propnBadRets > 0.5:
        #     logging.info('Regression statistics suspect and will be nuked')
        #     suspectDay = True
        # else:
        #     suspectDay = False
        suspectDay = False
        ###############################################################################
        # Compute excess returns
        (excessReturnMatrix, rfr) = self.computeExcessReturns(modelDate,assetReturnMatrix, modelDB, marketDB, data.drCurrData)
        excessReturns = excessReturnMatrix.data[:,0]

        allCurrencyISOs = list(set(data.assetCurrencyMap.values()))
        rfHistory = modelDB.getRiskFreeRateHistory(allCurrencyISOs, asset_rets_df.columns, marketDB)
        rf_df = rfHistory.toDataFrame().fillna(0)

        currency_dict = dict((x.getSubIDString(),y) for x,y in data.assetCurrencyMap.items())
        sub_iso = [currency_dict[x] for x in asset_rets_df.index]
        rf_df = rf_df.reindex(sub_iso)
        rf_df.index = asset_rets_df.index

        excessReturns_df = asset_rets_df - rf_df
        # for i in xrange(len(rfr.assets)):
        #     if rfr.data[i,0] is not ma.masked:
        #         self.log.debug('Using risk-free rate of %f%% for %s',rfr.data[i,0] * 100.0, rfr.assets[i])
        ##################################################################################################################
        # compute pcttrade:
        # Report on markets with non-trading day or all missing returns
        # Such will have their returns replaced either with zero or a proxy
        # nonTradingMarketsIdx = []
        nukeTStatList = []
        totalESTUMarketCaps = ma.sum(ma.take(data.marketCaps, data.estimationUniverseIdx, axis=0), axis=None)
        tradingCaps = 0.0
        # from IPython import embed; embed(header='Debug:compute pcttrade');import ipdb;ipdb.set_trace()
        try:
            for r in self.rmg:
                # Pull out assets for each RMG
                if r.rmg_id not in data.rmgAssetMap:
                    rmg_indices = []
                else:
                    rmg_indices = [data.assetIdxMap[n] for n in data.rmgAssetMap[r.rmg_id].intersection(estuMap['main'].assets)]
                rmg_returns = ma.take(excessReturns, rmg_indices, axis=0)

                # Get missing returns (before any proxying) and calendar dates
                noOriginalReturns = numpy.sum(ma.take(missingReturnsMask, rmg_indices, axis=0), axis=None)
                rmgCalendarList = modelDB.getDateRange(r, assetReturnMatrix.dates[0], assetReturnMatrix.dates[-1])

                if noOriginalReturns >= 0.95 * len(rmg_returns) or modelDate not in rmgCalendarList:
                    # Do some reporting and manipulating for NTD markets
                    nukeTStatList.append(r.description)
                    rmgMissingIdx = list(set(stillMissingReturnsIdx).intersection(set(rmg_indices)))
                    if len(rmgMissingIdx) > 0:
                        self.log.info('Non-trading day for %s, %d/%d returns missing',r.description, noOriginalReturns, len(rmg_returns))
                    else:
                        self.log.info('Non-trading day for %s, %d/%d returns imputed',r.description, noOriginalReturns, len(rmg_returns))
                else:
                    rmg_caps = ma.sum(ma.take(data.marketCaps, rmg_indices, axis=0), axis=None)
                    tradingCaps += rmg_caps
        except:
            from IPython import embed; embed(header='Debug:compute pcttrade');import ipdb;ipdb.set_trace()
        # # Report on % of market trading today
        pcttrade = tradingCaps / totalESTUMarketCaps
        logging.info('Proportion of total ESTU market trading: %.2f', pcttrade)

        # Get industry asset buckets
        # data.industryAssetMap = dict()
        # for idx in expM.getFactorIndices(ExposureMatrix.IndustryFactor):
        #     assetsIdx = numpy.flatnonzero(expM.getMatrix()[idx,:])
        #     data.industryAssetMap[idx] = numpy.take(data.universe, assetsIdx, axis=0)

        # Get indices of factors that we don't want in the regression
        # countryFactorsIdx = expM.getFactorIndices(ExposureMatrix.CountryFactor)
        currencyFactorsIdx = expM.getFactorIndices(ExposureMatrix.CurrencyFactor)
        excludeFactorsIdx1 = list(set(deadFactorIdx + currencyFactorsIdx))
        exclude_fac_names = np.take(expM.getFactorNames(),excludeFactorsIdx1)
        exp_mat = expM.toDataFrame().fillna(0)
        exp_mat.index = [x.getSubIDString() for x in exp_mat.index]
        exp_mat = exp_mat.drop(exclude_fac_names, axis=1).reindex(excessReturns_df.index)
        x_status = (exp_mat!=0)+0
        y_status = (excessReturns_df!=0)+0
        tmp_df = y_status.T.dot(x_status).T.iloc[:,0]
        if sum(tmp_df == 0)>0:
            extra_exclude_names = list(tmp_df[tmp_df == 0].index)
            exp_mat = exp_mat.drop(extra_exclude_names, axis=1)
        # Remove any remaining empty factors
        # excludeFactorsIdx = excludeFactorsIdx1
        # for idx in expM.getFactorIndices(ExposureMatrix.StyleFactor):
        #     assetsIdx = numpy.flatnonzero(expM.getMatrix()[idx,:])
        #     if len(assetsIdx) == 0:
        #         self.log.warning('100%% empty factor, excluded from all regressions: %s', expM.getFactorNames()[idx])
        #         excludeFactorsIdx.append(idx)
        #     else:
        #         propn = len(assetsIdx) / float(len(data.universe))
        #         if propn < 0.01:
        #             self.log.warning('%.1f%% exposures non-missing, excluded from all regressions: %s',
        #                     100*propn, expM.getFactorNames()[idx])
        #             excludeFactorsIdx.append(idx)
        ##############################################################################################################################################
        # Call nested regression routine
        # returnData = rcClass.run_factor_regressions(self, rcClass, prevDate, excessReturns, expM, estu, data,excludeFactorsIdx, modelDB, marketDB, applyRT=applyRT, fmpRun=buildFMPs)

        # Set up some data items to be used later
        # Factor info
        factorReturnsMap = dict()
        # allFactorNames = expMatrixCls.getFactorNames()
        # self.factorNameIdxMap = dict(zip(allFactorNames, range(len(allFactorNames))))
        # self.excludeFactorIdx = excludeFactorIdx
        # Returns info
        # regressionReturns = ma.array(excessReturns)
        # assetExposureMatrix = ma.array(expMatrixCls.getMatrix())
        # Regression stats info
        # robustWeightMap = dict()
        # ANOVA_data = list()
        # regressStatsMap = dict()
        # FMP Info
        # fmpMap = dict()
        # ccMap = dict()
        # ccXMap = dict()
        # DB Info
        # self.modelDB = modelDB
        # self.marketDB = marketDB
        # self.date = date
        # self.VIF = None

        # Report on returns
        # returns_ESTU = ma.take(excessReturns, data.estimationUniverseIdx, axis=0)
        # rtCap_ESTU = ma.take(data.marketCaps, data.estimationUniverseIdx, axis=0)
        # skewness = stats.skew(returns_ESTU * ma.sqrt(rtCap_ESTU), axis=0)
        # logging.info('Skewness of ESTU returns: %f', skewness)

        # Some models may require certain exposures to be set to zero
        # if _riskModel.zeroExposureNames != []:
        #     assetExposureMatrix = self.zeroCertainFactorExp(expMatrixCls, assetExposureMatrix, _riskModel)

        # Identify possible dummy style factors
        # dummyStyles = self.checkForStyleDummies(expMatrixCls, assetExposureMatrix, _riskModel)

        # Loop round however many regressions are required
        import DC_YD as yd
        yd1 = yd.DC_YD()
        fac_types = yd1.get_fac_type(exp_mat.columns,'dict')
        regKeys = sorted(rcClass.allParameters.keys()) # two regressions: (1) for main (2) for ChinaA
        # from IPython import embed; embed(header='Debug:generateFactorSpecificReturns_YD2');import ipdb;ipdb.set_trace()
        excess_rets_for_regs = excessReturns_df.copy()
        adjRsquared = {}
        for iReg in regKeys:
            # Get specific regression paramters
            regPar = rcClass.allParameters[iReg]
            self.log.info('Beginning nested regression, loop %d, ESTU: %s', iReg+1, regPar.estuName)
            self.log.info('Factors in loop: %s', ', '.join([f.name for f in regPar.regressionList]))
            # determin the estu
            estu = list(estuMap[regPar.estuName].assets)
            estu_sub_ids = [x.getSubIDString() for x in estu]
            estu_excess_rets = excess_rets_for_regs.reindex(estu_sub_ids)
            estu_exp_mat = exp_mat.reindex(estu_sub_ids)
            mktcap_df = pd.DataFrame(data.marketCaps,index=data.universe,columns=['mktcap'])
            mktcap_df.index = [x.getSubIDString() for x in mktcap_df.index]
            estu_mktcap_df = mktcap_df.reindex(estu_sub_ids)

            # Determine which factors will go into this regression loop
            # regPar.regFactorsIdx, regPar.regFactorNames = rcClass.processFactorsForRegression(regPar, expMatrixCls, _riskModel)
            # regMatrix = ma.take(assetExposureMatrix, regPar.regFactorsIdx, axis=0)

            try:
                sel_fac_types = [x.name for x in regPar.regressionList]
                sel_facs = []
                for x in sel_fac_types:
                    if x in fac_types.keys():
                        for item in fac_types[x]:
                            sel_facs.append(item)
                if len(sel_facs) >0:
                    sel_exp = estu_exp_mat[sel_facs]
                else:
                    adjRsquared[regPar.name] = 0.0
                    continue
            except:
                from IPython import embed; embed(header='Debug:fac_types');import ipdb;ipdb.set_trace()
            # Get estimation universe for this loop
            # regPar.reg_estu = list(_riskModel.estuMap[regPar.estuName].assets)
            # logging.info('Using %d assets from %s estimation universe', len(regPar.reg_estu), regPar.estuName)
            # regPar.regEstuIdx = [data.assetIdxMap[sid] for sid in regPar.reg_estu]

            # Get the regression weights
            # regPar.reg_weights = self.getRegressionWeights(regPar, data, _riskModel)
            # sqrt root of market cap with Top 5% clipped:
            reg_weight = np.sqrt(estu_mktcap_df)
            top5_w = np.percentile(reg_weight,95)
            reg_weight[reg_weight>top5_w] = top5_w

            # If ESTU assets have no returns, warn and skip
            bad_rets_cnt = (estu_excess_rets.abs() < 1e-12).sum()
            factorReturnsMap = dict() # one of output
            regressStatsMap = dict()
            if (bad_rets_cnt[0] >= 0.99 * len(estu_excess_rets)) or (len(estu_excess_rets) < 1):
                self.log.warning('No returns for nested regression loop %d ESTU, skipping', iReg + 1)
                specific_rets = estu_excess_rets.copy()
                for fName in sel_facs:
                    factorReturnsMap[fName] = 0.0
                    regressStatsMap[fName] = Matrices.allMasked(4)
                continue
            # Deal with thin factors - compute market returns
            # thinFacPar = self.processThinFactorsForRegression(_riskModel, regPar, expMatrixCls, regressionReturns, data, applyRT)
            import statsmodels.api as sm
            if regPar.estuName == 'main':
                raw_mdl = sm.WLS(estu_excess_rets,sel_exp['Market Intercept'],weights=reg_weight).fit()
            elif regPar.estuName == 'ChinaA':
                raw_mdl = sm.WLS(estu_excess_rets,sel_exp['Domestic China'],weights=reg_weight).fit()
            else:
                raise Exception('Unknown estu type: %s' % regPar.estuName)
            raw_mkta_rets = raw_mdl.params[0]

            # correct for thin country and industry factors
            from collections import defaultdict
            dummy_assets = defaultdict(list)
            dummy_size = regPar.dummyThreshold
            if regPar.estuName == 'main':
                check_fac_list = fac_types['Country'] + fac_types['Industry']
            else:
                check_fac_list = fac_types['Local']
            for each_fac in check_fac_list:
                tmp_exp = sel_exp[each_fac]
                tmp_w = reg_weight.reindex(tmp_exp[tmp_exp==1].index)
                herf_size = yd1.compute_herf0(tmp_w)
                if herf_size < dummy_size:
                    print(('%s is a thin factor with herf_size: %f' % (each_fac,herf_size)))
                    dummy_assetname = ['dummyAsset_%s' %each_fac]
                    dummy_weight = (dummy_size -1) * (dummy_size**4-herf_size**4)/(dummy_size**4-1)*tmp_w.sum()/herf_size
                    dummy_weight.index = dummy_assetname
                    dummy_return = pd.Series(raw_mkta_rets,index=dummy_assetname)
                    dummy_exposure = pd.DataFrame([1,1],columns=dummy_assetname,index=['Market Intercept',each_fac]).T
                    dummy_assets['weight'].append(dummy_weight)
                    dummy_assets['return'].append(dummy_return)
                    dummy_assets['exposure'].append(dummy_exposure)
            if len(dummy_assets['weight'])>0:
                dummy_assets['weight'] = pd.concat(dummy_assets['weight'])
                dummy_assets['return'] = pd.concat(dummy_assets['return'])
                dummy_assets['exposure'] = pd.concat(dummy_assets['exposure'])
                # insert dummy assets into regression input
                # add dummyAsset to correct thin factor for
                estu_excess_retsV2 = pd.concat([estu_excess_rets,dummy_assets['return'].to_frame(estu_excess_rets.columns[0])])
                sel_expV2 = pd.concat([sel_exp,dummy_assets['exposure']]).fillna(0)
                reg_weightV2 = pd.concat([reg_weight,dummy_assets['weight'].to_frame(reg_weight.columns[0])])
            else:
                estu_excess_retsV2 = estu_excess_rets.copy()
                sel_expV2 = sel_exp.copy()
                reg_weightV2 = reg_weight.copy()

            if regPar.estuName == 'main':
                # construct constraints
                cntry_w = sel_expV2[fac_types['Country']].T.dot(reg_weightV2)
                industry_w = sel_expV2[fac_types['Industry']].T.dot(reg_weightV2)
                constraint1 = cntry_w.reindex(sel_expV2.columns).fillna(0).T
                constraint1.index = ['constraint_country']
                constraint2 = industry_w.reindex(sel_expV2.columns).fillna(0).T
                constraint2.index = ['constraint_industry']
                constraints = pd.concat([constraint1,constraint2])
                constraints_rets = pd.DataFrame([0,0],index=['constraint_country','constraint_industry'],columns=estu_excess_retsV2.columns)
                constraints_weights = pd.DataFrame([cntry_w.sum(),industry_w.sum()],index=['constraint_country','constraint_industry'],columns=reg_weightV2.columns)
                # insert dummy assets into regression input
                estu_excess_retsV3 = pd.concat([estu_excess_retsV2,constraints_rets])
                sel_expV3 = pd.concat([sel_expV2,constraints])
                reg_weightV3 = pd.concat([reg_weightV2,constraints_weights])
            else:
                estu_excess_retsV3 = estu_excess_retsV2.copy()
                sel_expV3 = sel_expV2.copy()
                reg_weightV3 = reg_weightV2.copy()
            # sel_expV2[fac_types['Country']].sum(axis=1)
            ######################################################################################################################
            # Finally, run the regression
            # regPar.iReg = iReg + 1
            # regOut = self.calc_Factor_Specific_Returns(_riskModel, regPar, thinFacPar, data, regressionReturns, regMatrix,expMatrixCls, fmpRun)
            # reg_mdl1 = sm.OLS(estu_excess_retsV2,sel_expV2).fit()
            ######################################################################################################################
            # drop assets that are missing or not traded
            try:
                n_missing = estu_excess_retsV3.isnull().sum()[0]
                if n_missing>0:
                    logging.info('There are %d missing returns in ESTU.', n_missing)
                    logging.info('Drop Them in the regression')
                    estu_excess_retsV3 = estu_excess_retsV3.dropna()
                    sel_expV3 = sel_expV3.reindex(estu_excess_retsV3.index)
                    reg_weightV3 = reg_weightV3.reindex(estu_excess_retsV3.index)
            except:
                from IPython import embed; embed(header='Debug:n_missing');import ipdb;ipdb.set_trace()

            if regPar.estuName == 'main':
                # save regression weight
                # from IPython import embed; embed(header='Debug:estu_weight_pair');import ipdb;ipdb.set_trace()
                estu_weight_pair = []
                estu_ids = [x.getSubIDString() for x in estu]
                reg_weightV4 = reg_weightV3.reindex(estu_ids).fillna(0)
                for x in estu:
                    estu_weight_pair.append((x,np.float(reg_weightV4.loc[x.getSubIDString()])))
                # estu_weight_pair = dict(estu_weight_pair)
            if len(estu_excess_retsV3)>0:
                try:
                    reg_mdl2 = sm.WLS(estu_excess_retsV3,sel_expV3,weights=reg_weightV3).fit()
                except:
                    from IPython import embed; embed(header='Debug:WLS');import ipdb;ipdb.set_trace()
                # import datamodel.analytics as analytics
                # import numpy as np
                # reg_weightV2.columns = estu_excess_retsV2.columns
                # reg_mdl3 = analytics.ConstrainedLinearModel(estu_excess_retsV2, sel_expV2, C=constraints, weights=reg_weightV2)
                # fr = reg_mdl3.params
                # tmp_df = pd.concat([reg_mdl2.params,fr.iloc[:,0]],axis=1,keys=['Method1','Method2'])
                fac_rets = reg_mdl2.params
                fac_rets[fac_rets.abs()<1e-10] = 0.0
                # spec_rets = reg_mdl2.resid
                # use current residual returns for next round
                fitted_rets = exp_mat[fac_rets.index].dot(fac_rets)
                fitted_rets = fitted_rets.to_frame(excessReturns_df.columns[0])
                spec_rets = excessReturns_df - fitted_rets # residual returns
                excess_rets_for_regs = spec_rets.copy()
                ######################################################################################################################
                # if ExposureMatrix.StyleFactor in regPar.regressionList and regPar.computeVIF:
                #     # Assuming that 1st round of regression is  done on styles, industries, market etc.,
                #     # and second round regression is done on Domestic China etc.,
                #     self.sm = sm
                #     self.VIF = self.calculateVIF(ma.filled(expMatrixCls.getMatrix(), 0.0),
                #                     regOut.regressANOVA.weights_, regOut.regressANOVA.estU_, expMatrixCls.factorIdxMap_)
                # Pass residuals to next regression loop [???]
                # regressionReturns = regOut.specificReturns
                # from IPython import embed; embed(header='Debug:generateFactorSpecificReturns_YD3');import ipdb;ipdb.set_trace()
                # Keep record of factor returns and regression statistics
                # if regPar.estuName == 'main':
                #     ANOVA_data.append(regOut.regressANOVA)
                # for jdx in xrange(len(regPar.regFactorsIdx)):
                #
                #     # Save factor returns
                #     fName = regPar.regFactorNames[jdx]
                #     factorReturnsMap[fName] = regOut.factorReturns[jdx]
                #
                #     # Save FMP info
                #     if fName in regOut.fmpDict:
                #         fmpMap[fName] = dict(zip(regOut.sidList, regOut.fmpDict[fName].tolist()))
                #     if regOut.fmpConstrComponent is not None:
                #         if fName in regOut.fmpConstrComponent.ccDict:
                #             ccMap[fName] = regOut.fmpConstrComponent.ccDict[fName]
                #             ccXMap[fName] = regOut.fmpConstrComponent.ccXDict[fName]
                #
                #     # Save regression stats
                #     values = Matrices.allMasked(4)
                #     values[:3] = regOut.regressANOVA.regressStats_[jdx,:]
                #     if fName in regOut.constraintWeight:
                #         values[3] = regOut.constraintWeight[fName]
                #     regressStatsMap[fName] = values
                # organize output:
                factorReturnsMap.update(fac_rets.to_dict())
                adjRsquared[regPar.name] = reg_mdl2.rsquared_adj
            else:
                tmp_fac_rets_dict = {(x,0.0) for x in sel_expV3.columns}
                factorReturnsMap.update(tmp_fac_rets_dict)
                spec_rets = excessReturns_df.copy()
                adjRsquared[regPar.name] = 0.0
        # fill in None for missing asset returs

        all_sub_ids = [x.getSubIDString() for x in data.universe]
        # spec_rets = spec_rets.reindex(all_sub_ids)
        spec_rets = spec_rets.reindex(all_sub_ids).fillna(0)
        # from IPython import embed; embed(header='Debug:reshape spec_rets');import ipdb;ipdb.set_trace()
            # Add robust weights to running total
            # tmpWeights = ma.masked_where(regOut.rlmDownWeights==1.0, regOut.rlmDownWeights)
            # tmpWeightsIdx = numpy.flatnonzero(ma.getmaskarray(tmpWeights)==0)
            # tmpWeights = ma.take(tmpWeights, tmpWeightsIdx, axis=0)
            # tmpSIDs = numpy.take(regOut.sidList, tmpWeightsIdx, axis=0)
            # robustWeightMap[iReg] = dict(zip(tmpSIDs, tmpWeights))
        # Regression ANOVA: take average of all regressions using MAIN ESTU
        # self.log.info('%d regression loops all use same ESTU, computing ANOVA', len(ANOVA_data))
        # numFactors = sum([n.nvars_ for n in ANOVA_data])
        # regWeights = numpy.average(numpy.array([n.weights_ for n in ANOVA_data]), axis=0)
        # anova = RegressionANOVA(excessReturns, regOut.specificReturns, numFactors, ANOVA_data[0].estU_, regWeights)
        self.log.debug('run_factor_regressions: end')
        for fName in factorReturnsMap.keys():
            regressStatsMap[fName] = Matrices.allMasked(4)
        ######################################################################################################################
        returnData = Utilities.Struct()
        returnData.factorReturnsMap = factorReturnsMap
        returnData.specificReturns = np.ndarray.flatten(np.transpose(np.ma.array(spec_rets)))
        returnData.regStatsMap = regressStatsMap
        returnData.anova = None # It is a FactorReturns.RegressionANOVA instance
        returnData.robustWeightMap = {}
        returnData.fmpMap = {}
        returnData.ccMap = {}
        returnData.ccXMap = {}
        # Map specific returns for cloned assets
        # returnData.specificReturns = ma.masked_where(missingReturnsMask, returnData.specificReturns)
        # if len(data.hardCloneMap) > 0:
        #     cloneList = set(data.hardCloneMap.keys()).intersection(set(data.universe))
        #     for sid in cloneList:
        #         if data.hardCloneMap[sid] in data.universe:
        #             returnData.specificReturns[data.assetIdxMap[sid]] = returnData.specificReturns\
        #                     [data.assetIdxMap[data.hardCloneMap[sid]]]

        # Store regression results
        factorReturns = Matrices.allMasked((len(allFactors),))
        regressionStatistics = Matrices.allMasked((len(allFactors), 4))
        for (fName, ret) in returnData.factorReturnsMap.items():
            idx = subFactorIDIdxMap.get(nameSubIDMap[fName], None)
            if idx is not None:
                factorReturns[idx] = ret
                if (not suspectDay) and (fName not in nukeTStatList):
                    regressionStatistics[idx,:] = returnData.regStatsMap[fName]
                else:
                    regressionStatistics[idx,-1] = returnData.regStatsMap[fName][-1]
        if not internalRun and not buildFMPs:
            # Calculate Variance Inflation Factors for each style factor regressed on other style factors.
            # self.VIF = rcClass.VIF
            self.VIF = None

        result = Utilities.Struct()
        result.universe = data.universe
        result.factorReturns = factorReturns
        result.specificReturns = returnData.specificReturns
        result.exposureMatrix = self.convert_exposure_df_to_expM(exp_mat,data.universe)
        result.regressionStatistics = regressionStatistics
        result.adjRsquared = adjRsquared['main']
        result.pcttrade = pcttrade
        result.regression_ESTU = estu_weight_pair #subidWeightPairs
        result.VIF = None
        # from IPython import embed; embed(header='Debug:output');import ipdb;ipdb.set_trace()
        # Process robust weights
        newRWtMap = dict()
        # sid2StringMap = dict([(sid if isinstance(sid, basestring) else sid.getSubIDString(), sid) for sid in data.universe])
        # for (iReg, rWtMap) in returnData.robustWeightMap.items():
        #     tmpMap = dict()
        #     for (sidString, rWt) in rWtMap.items():
        #         if sidString in sid2StringMap:
        #             tmpMap[sid2StringMap[sidString]] = rWt
        #     newRWtMap[iReg] = tmpMap
        result.robustWeightMap = newRWtMap

        # Process FMPs
        # newFMPMap = dict()
        # for (fName, fmpMap) in returnData.fmpMap.items():
        #     tmpMap = dict()
        #     for (sidString, fmp) in fmpMap.items():
        #         if sidString in sid2StringMap:
        #             tmpMap[sid2StringMap[sidString]] = fmp
        #     newFMPMap[nameSubIDMap[fName]] = tmpMap
        # result.fmpMap = newFMPMap

        # Report non-trading markets and set factor return to zero
        # allFactorNames = expM.getFactorNames()
        # if len(nonTradingMarketsIdx) > 0:
        #     nonTradingMarketNames = ', '.join([allFactorNames[i] for i in nonTradingMarketsIdx])
        #     self.log.info('%d non-trading market(s): %s', len(nonTradingMarketsIdx), nonTradingMarketNames)
        #     for i in nonTradingMarketsIdx:
        #         idx = subFactorIDIdxMap[nameSubIDMap[allFactorNames[i]]]
        #         result.factorReturns[idx] = 0.0

        # Pull in currency factor returns from currency model
        if self.multiCountry:
            crmi = modelDB.getRiskModelInstance(self.currencyModel.rms_id, modelDate)
            assert (crmi is not None)
            if not weeklyRun:
                (currencyReturns, currencyFactors) = self.currencyModel.loadCurrencyReturns(crmi, modelDB)
            currSubFactors = modelDB.getSubFactorsForDate(modelDate, currencyFactors)
            currSubIDIdxMap = dict([(currSubFactors[i].subFactorID, i) \
                                    for i in range(len(currSubFactors))])
            self.log.info('loaded %d currencies from currency model', len(currencyFactors))

            # Lookup currency factor returns from currency model
            currencyFactors = set(self.currencies)
            for (i,j) in subFactorIDIdxMap.items():
                cidx = currSubIDIdxMap.get(i, None)
                if cidx is None:
                    if allFactors[j] in currencyFactors:
                        self.log.warning('Missing currency factor return for %s', allFactors[j].name)
                        value = 0.0
                    else:
                        continue
                else:
                    value = currencyReturns[cidx]
                result.factorReturns[j] = value

        # constrComp = Utilities.Struct()
        # constrComp.ccDict = returnData.ccMap
        # constrComp.ccXDict = returnData.ccXMap

        # if nextTradDate is not None:
        #     self.regressionReporting(futureExcessReturns, result, expM, nameSubIDMap, data.assetIdxMap,
        #                     modelDate, buildFMPs=buildFMPs, constrComp=constrComp, specificRets=result.specificReturns)
        # else:
        #     self.regressionReporting(excessReturns, result, expM, nameSubIDMap, data.assetIdxMap,
        #                     modelDate, buildFMPs=buildFMPs, constrComp=constrComp, specificRets=result.specificReturns)

        # if self.debuggingReporting:
        #     for (i,sid) in enumerate(data.universe):
        #         if abs(returnData.specificReturns[i]) > 1.5:
        #             self.log.info('Large specific return for: %s, ret: %s',
        #                     sid, returnData.specificReturns[i])
        return result

class EM4_Research3(EM4_Research2):
    """
        EM research model - base model - DM Sensitivity
        EM research model - base model - DM Sensitivity (use Oil Sensitivity descriptor)
        EM research model - base model - DM Sensitivity (use Oil Sensitivity descriptor without Orthogonal)
        EM research model - base model - before release - deal with bad assets: doEverything
        EM research model - base model - to study methodology systematically
        EM research model - base model - dropZeroNanAsset
    """
    rm_id,revision,rms_id = [-22,3,-222]
    ##############################################################################################
    YD_Version_ESTU = False
    YD_Version_EXP = False
    # exposureConfigFile = 'exposures-mh2.config'
    YD_Version_Factor = False
    YD_Version_Risk = False
    run_sectionA_flag = True # comment out section A
    run_sectionB_flag = True # comment out section B
    run_sectionC_flag = False # comment out section C
    run_sectionD_flag = True # comment out section D (it should be true, it was False before)
    ##############################################################################################

    # styleList = ['Value',
    #              'Leverage',
    #              'Growth',
    #              'Profitability',
    #              'Earnings Yield',
    #              'Dividend Yield',
    #              'Size',
    #              'Liquidity',
    #              'Market Sensitivity',
    #              'DM Sensitivity',
    #              'Volatility',
    #              'Medium-Term Momentum',
    #              'Exchange Rate Sensitivity',
    #              ]
    # DescriptorMap = {
    #         'Earnings Yield': ['Earnings_to_Price_Annual', 'Est_Earnings_to_Price_12MFL_Annual'],
    #         'Value': ['Book_to_Price_Annual'],
    #         'Leverage': ['Debt_to_Assets_Annual', 'Debt_to_Equity_Annual'],
    #         'Growth': ['Earnings_Growth_RPF_Annual', 'Sales_Growth_RPF_Annual'],
    #         'Dividend Yield': ['Dividend_Yield_Annual'],
    #         'Size': ['LnIssuerCap'],
    #         'Liquidity': ['LnTrading_Activity_60D', 'Amihud_Liquidity_125D', 'ISC_Ret_Score'],
    #         'Market Sensitivity': ['EM_Regional_Market_Sensitivity_500D'],
    #         # 'DM Sensitivity': ['DM_netEM_Regional_Market_Sensitivity_500D'],
    #         'DM Sensitivity': ['Oil_Sensitivity_4'],
    #         'Volatility': ['Volatility_125D'],
    #         'Medium-Term Momentum': ['Momentum_250x20D'],
    #         'Exchange Rate Sensitivity': ['XRate_104W_XDR'],
    #         'Interest Rate Sensitivity': ['XRate_104W_IR'],
    #         'Profitability': ['Return_on_Equity_Annual', 'Return_on_Assets_Annual',
    #             'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
    #             'Sales_to_Assets_Annual', 'Gross_Margin_Annual'],
    #         }

    #industryClassification = Classification.GICSCustomEM(datetime.date(2016,9,1))
    # orthogList = {'Volatility': [['Market Sensitivity'], True, 1.0],
    #               'DM Sensitivity': [['Market Sensitivity'], True, 1.0]}
    # orthogList = {'Volatility': [['Market Sensitivity'], True, 1.0]}
    dropZeroNanAsset = False
    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.EM4_Research3')
        super(EM4_Research3,self).__init__(modelDB, marketDB)
        ############################################ change ESTU Setting ############################################
        self.estu_parameters = {'minNonZero': [0.9, 0.05, 0.05],
                           'minNonMissing': [0.9, 0.05, 0.05],
                           'ChinaATol': 0.9,
                           'maskZeroWithNoADV_Flag': True,
                           'CapByNumber_Flag': False,
                           'CapByNumber_hiCapQuota': np.nan,
                           'CapByNumber_lowCapQuota': np.nan,
                           'market_lower_pctile': 5.0,
                           'country_lower_pctile': 5.0,
                           'industry_lower_pctile': 5.0,
                           'dummyThreshold': 10,
                           'inflation_cutoff': 0.05,
                          }
        ####################################################################################################################################

class EM4_Research4(EM4_Research1):
    """
        EM research model - base model - no regional standardization
        EM research model - base model - Customized Industry for industry factors
        EM research model - base model - overlapped 5 days for factorVolatility calculation
        EM research model - use weekly returns without NW adjustment and without dateOverLap
        EM research model - use external returns with NW adjustment and without dateOverLap
        EM research model - create the venilla version for derby files
        EM research model - with aggressive ESTU setting - 0.9
        EM research model - use Russell Emerging as ESTU - 2017-12-26
        EM research model - base model - before release - deal with bad assets: AssetOnly
        EM research model - base model - EM21 Estu + DropZeroNan
        EM research model - regional-based style momentum factor - 2018-02-28
    """
    rm_id,revision,rms_id = [-22,4,-223]
    ##############################################################################################
    YD_Version_ESTU = False
    YD_Version_EXP = False
    YD_Version_Factor = False
    YD_Version_Risk = False
    dropZeroNanAsset = False
    exposureConfigFile = 'exposures-mh_Reg'
    ##############################################################################################
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
             'Medium-Term Momentum Europe',
             'Medium-Term Momentum Asia exPacific',
             'Medium-Term Momentum Africa',
             'Medium-Term Momentum Middle East',
             'Medium-Term Momentum Latin America',
             'Exchange Rate Sensitivity',
             ]
    DescriptorMap = {
            'Earnings Yield': ['Earnings_to_Price_Annual', 'Est_Earnings_to_Price_12MFL_Annual'],
            'Value': ['Book_to_Price_Annual'],
            'Leverage': ['Debt_to_Assets_Annual', 'Debt_to_Equity_Annual'],
            'Growth': ['Earnings_Growth_RPF_Annual', 'Sales_Growth_RPF_Annual'],
            'Dividend Yield': ['Dividend_Yield_Annual'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_60D', 'Amihud_Liquidity_125D', 'ISC_Ret_Score'],
            'Market Sensitivity': ['EM_Regional_Market_Sensitivity_500D_V0'],
            'Volatility': ['Volatility_125D'],
            'Medium-Term Momentum Europe': ['Momentum_250x20D'],
            'Medium-Term Momentum Asia exPacific': ['Momentum_250x20D'],
            'Medium-Term Momentum Africa': ['Momentum_250x20D'],
            'Medium-Term Momentum Middle East': ['Momentum_250x20D'],
            'Medium-Term Momentum Latin America': ['Momentum_250x20D'],
            'Exchange Rate Sensitivity': ['XRate_104W_XDR'], #'XRate_104W_USD',XRate_104W_XDR
            'Interest Rate Sensitivity': ['XRate_104W_IR'],
            'Profitability': ['Return_on_Equity_Annual', 'Return_on_Assets_Annual',
                'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
                'Sales_to_Assets_Annual', 'Gross_Margin_Annual'],
            #Payout': ['Net_Equity_Issuance', 'Net_Debt_Issuance', 'Net_Payout_Over_Profits'],
            }
    regionalStndList = ['Earnings Yield', 'Value', 'Leverage', 'Growth', 'Profitability', 'Dividend Yield','Medium-Term Momentum Europe',
                        'Medium-Term Momentum Asia exPacific','Medium-Term Momentum Africa','Medium-Term Momentum Middle East','Medium-Term Momentum Latin America']# 'Payout']
    _regionalMapper = lambda dm, l: [dm[st] for st in l]
    regionalStndDesc = list(itertools.chain.from_iterable(_regionalMapper(DescriptorMap, regionalStndList)))
    del _regionalMapper
    
    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.EM4_Research4')
        super(EM4_Research4,self).__init__(modelDB, marketDB)

    # expusre for region-based style factor
    def generateExposureMatrix(self, modelDate, modelDB, marketDB):
        print('EM4_Research4================================================================================')
        data = super(EM4_Research4,self).generateExposureMatrix(modelDate, modelDB, marketDB)
        # from IPython import embed; embed(header='Debug:generateExposureMatrix - Region Style Model');import ipdb;ipdb.set_trace()
        reg_fac_dict = {'Europe':'Medium-Term Momentum Europe','Asia ex-Pacific':'Medium-Term Momentum Asia exPacific','Africa':'Medium-Term Momentum Africa','Middle East':'Medium-Term Momentum Middle East','Latin America':'Medium-Term Momentum Latin America'}
        tmpdata = data[0]
        reg_scopes = self.descriptorStandardization.factorScopes[0]
        for (bucketDesc, assetIndices) in reg_scopes.getAssetIndices(tmpdata.exposureMatrix, modelDate):
            if bucketDesc in reg_fac_dict.keys():
                fac_name = reg_fac_dict[bucketDesc]
                print(fac_name)
                fac_idx = tmpdata.exposureMatrix.getFactorIndex(fac_name)
                non_asset_idx = list(set(range(len(tmpdata.exposureMatrix.assets_))).difference(assetIndices))
                tmpdata.exposureMatrix.data_[fac_idx,non_asset_idx]=0
        data[0] = tmpdata
        return data

    def generate_model_universe_YD(self, modelDate, modelDB, marketDB):

        self.log.info('[YD]generate_model_universe from EM2: begin')
        data = Utilities.Struct()
        universe = AssetProcessor.getModelAssetMaster(
                 self, modelDate, modelDB, marketDB, legacyDates=self.legacyMCapDates)
        # Throw out any assets with missing size descriptors
        descDict = dict(modelDB.getAllDescriptors())
        sizeVec = self.loadDescriptorData(['LnIssuerCap'], descDict, modelDate,
                        universe, modelDB, None, rollOver=False)[0]['LnIssuerCap']
        missingSizeIdx = numpy.flatnonzero(ma.getmaskarray(sizeVec))
        if len(missingSizeIdx) > 0:
            missingIds = numpy.take(universe, missingSizeIdx, axis=0)
            universe = list(set(universe).difference(set(missingIds)))
            logging.warning('%d missing LnIssuerCaps dropped from model', len(missingSizeIdx))
        data.universe = universe

        import DC_YD as yd
        yd_sb = yd.DC_YD('glprodsb')
        dt = modelDate
        rm_name_em2 = 'EMAxioma2011MH'

        if dt <= datetime.date(2002,12,31):
            rm_name2_estu = yd_sb.get_model_ESTU('EMAxioma2011MH_Pre2003', dt)
        elif dt <= datetime.date(2008,12,31):
            rm_name2_estu = yd_sb.get_model_ESTU('EMAxioma2011MH_Pre2009', dt)
        elif dt <= datetime.date(2012,12,31):
            rm_name2_estu = yd_sb.get_model_ESTU('EMAxioma2011MH_Pre2013', dt)
        else:
            rm_name2_estu = yd_sb.get_model_ESTU(rm_name_em2, dt)

        rm_name2_estu = [x for x in rm_name2_estu if isinstance(x, str)]
        import DC_YD as yd
        yd1 = yd.DC_YD()
        rm_name4_nursery = yd1.get_model_ESTU('EM4_Research1', dt,estu_type='nursery')
        rm_name4_ChinaA = yd1.get_model_ESTU('EM4_Research1', dt,estu_type='ChinaA')

        rm_name2_estu = list(set(rm_name2_estu).difference(rm_name4_nursery).difference(rm_name4_ChinaA))
        rm_name2_estu = yd_sb.get_subissue_objs(rm_name2_estu)
        rm_name4_nursery = yd_sb.get_subissue_objs(rm_name4_nursery)
        rm_name4_ChinaA = yd_sb.get_subissue_objs(rm_name4_ChinaA)

        self.estuMap['main'].assets = rm_name2_estu
        self.estuMap['main'].qualify = rm_name2_estu
        self.estuMap['nursery'].assets = rm_name4_nursery
        self.estuMap['nursery'].qualify = rm_name4_nursery
        self.estuMap['ChinaA'].assets = rm_name4_ChinaA
        self.estuMap['ChinaA'].qualify = rm_name4_ChinaA
        self.log.info('[YD]generate_model_universe from EM2: end.')
        return data
        # from IPython import embed; embed(header='Debug:generate_model_universe_YD');import ipdb;ipdb.set_trace()
    #     # Set up risk parameters with dateOverLap=5
    #     # ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=3, dateOverLap=5)
    #     # self.covarianceCalculator = RiskCalculator.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)
    #     # case 1
    #     # self.YD_dateOverLap = None
    #     # self.YD_NWLag = 0
    #     # self.return_flag = 'weekly'
    #     # case 2
    #     # self.YD_dateOverLap = None
    #     # self.YD_NWLag = 3
    #     # self.return_flag = 'external'
    #     # case 3
    #     # self.YD_dateOverLap = None
    #     # self.YD_NWLag = 3
    #     # self.return_flag = 'internal'
    #
    # def generate_model_universe(self, modelDate, modelDB, marketDB):
    #
    #     self.log.info('[YD]generate_model_universe: begin')
    #     data = Utilities.Struct()
    #     universe = AssetProcessor.getModelAssetMaster(
    #              self, modelDate, modelDB, marketDB, legacyDates=self.legacyMCapDates)
    #     # Throw out any assets with missing size descriptors
    #     descDict = dict(modelDB.getAllDescriptors())
    #     sizeVec = self.loadDescriptorData(['LnIssuerCap'], descDict, modelDate,
    #                     universe, modelDB, None, rollOver=False)[0]['LnIssuerCap']
    #     missingSizeIdx = numpy.flatnonzero(ma.getmaskarray(sizeVec))
    #     if len(missingSizeIdx) > 0:
    #         missingIds = numpy.take(universe, missingSizeIdx, axis=0)
    #         universe = list(set(universe).difference(set(missingIds)))
    #         logging.warning('%d missing LnIssuerCaps dropped from model', len(missingSizeIdx))
    #     data.universe = universe
    #
    #     # load Benchmark info
    #     import DC_YD as yd
    #     yd1 = yd.DC_YD() # use research_vital
    #     bmname = 'Russell Emerging'
    #     if modelDate < datetime.date(2000,1,3):
    #         bm_dts = sorted(yd1.dc.getBenchmarkDates(bmname))
    #         if modelDate <= bm_dts[0]:
    #             bmname_dt = bm_dts[0]
    #         else:
    #             bmname_dt = modelDate
    #             for d0,d1 in zip(bm_dts[:-1],bm_dts[1:]):
    #                 if modelDate < d1:
    #                     bmname_dt = d0
    #                     break
    #     else:
    #         bmname_dt = modelDate
    #     bm_assets = yd1.dc.getBenchmark(bmname_dt,bmname)
    #     tmp_bmname_dt = bmname_dt
    #     s = 0
    #     while(len(bm_assets)==0):
    #         tmp_bmname_dt = tmp_bmname_dt - datetime.timedelta(1)
    #         s = s+1
    #         bm_assets = yd1.dc.getBenchmark(tmp_bmname_dt,bmname)
    #         if s>30:
    #             raise Exception('No Benchmarch data for Russell Emerging.')
    #     sub_ids = yd1.ID2SubID(bm_assets.index, bmname_dt, type='AxiomaID')
    #     sub_ids_objs = yd1.get_subissue_objs(sub_ids)
    #     estu1 = list(set(universe).intersection(sub_ids_objs))
    #     self.estuMap['main'].assets = estu1
    #     self.estuMap['main'].qualify = estu1
    #     # for China A
    #     bmname = 'CSI 300'
    #     if modelDate < datetime.date(2004,12,31):
    #         bmname_dt = datetime.date(2004,12,31)
    #     else:
    #         bmname_dt = modelDate
    #     bm_assets = yd1.dc.getBenchmark(bmname_dt,bmname)
    #     tmp_bmname_dt = bmname_dt
    #     s = 0
    #     while(len(bm_assets)==0):
    #         tmp_bmname_dt = tmp_bmname_dt - datetime.timedelta(1)
    #         s = s+1
    #         bm_assets = yd1.dc.getBenchmark(tmp_bmname_dt,bmname)
    #         if s>30:
    #             raise Exception('No Benchmarch data for CSI 300.')
    #     # from IPython import embed; embed(header='Debug');import ipdb;ipdb.set_trace()
    #     sub_ids = yd1.ID2SubID(bm_assets.index, bmname_dt, type='AxiomaID')
    #     sub_ids_objs = yd1.get_subissue_objs(sub_ids)
    #     self.log.info('[YD]there are %d ChinaA assets before intersection.' % len(sub_ids_objs))
    #     estu2 = list(set(universe).intersection(sub_ids_objs))
    #     self.estuMap['ChinaA'].assets = estu2
    #     self.estuMap['ChinaA'].qualify = estu2
    #     self.log.info('[YD]generate_model_universe: finish')
    #     self.log.info('[YD]generate_model_universe: main ESTU: %d' % len(estu1))
    #     self.log.info('[YD]generate_model_universe: ChinaA ESTU: %d' % len(estu2))
    #     # from IPython import embed; embed(header='Debug:[YD]ESTU END');import ipdb;ipdb.set_trace()
    #     return data
    #
    # def generateExposureMatrix(self, modelDate, modelDB, marketDB):
    #     """Generates and returns the exposure matrix for the given date.
    #     The exposures are not inserted into the database.
    #     Data is accessed through the modelDB and marketDB DAOs.
    #     The return is a structure containing the exposure matrix
    #     (exposureMatrix), the universe as a list of assets (universe),
    #     and a list of market capitalizations (marketCaps).
    #     """
    #     self.log.info('generateExposureMatrix: begin')
    #     # from IPython import embed; embed(header='Debug:generateExposureMatrix');import ipdb;ipdb.set_trace()
    #     # Get risk model universe and market caps
    #     # Determine home country info and flag DR-like instruments
    #     rmi = modelDB.getRiskModelInstance(self.rms_id, modelDate)
    #     universe = modelDB.getRiskModelInstanceUniverse(rmi, returnExtra=True)
    #     data = AssetProcessor.process_asset_information(
    #             modelDate, universe, self.rmg, modelDB, marketDB,
    #             checkHomeCountry=self.multiCountry,
    #             numeraire_id=self.numeraire.currency_id,
    #             legacyDates=self.legacyMCapDates,
    #             forceRun=self.forceRun,
    #             nurseryRMGList=self.nurseryRMGs,
    #             tweakDict=self.tweakDict)
    #     data.exposureMatrix = Matrices.ExposureMatrix(data.universe)
    #
    #     if (not self.multiCountry) and (not hasattr(self, 'indexSelector')):
    #         self.indexSelector = MarketIndex.MarketIndexSelector(modelDB, marketDB)
    #         self.log.info('Index Selector: %s', self.indexSelector)
    #
    #     # Generate eligible universe
    #     data.eligibleUniverse = self.generate_eligible_universe(modelDate, data, modelDB, marketDB)
    #
    #     # Fetch trading calendars for all risk model groups
    #     # Start-date should depend on how long a history is required
    #     # for exposures computation
    #     data.rmgCalendarMap = dict()
    #     startDate = modelDate - datetime.timedelta(365*2)
    #     for rmg in self.rmg:
    #         data.rmgCalendarMap[rmg.rmg_id] = modelDB.getDateRange(rmg, startDate, modelDate)
    #
    #     # Compute issuer-level market caps if required
    #     AssetProcessor.computeTotalIssuerMarketCaps(
    #             data, modelDate, self.numeraire, modelDB, marketDB,
    #             debugReport=self.debuggingReporting)
    #
    #     if self.multiCountry:
    #         self.generate_binary_country_exposures(modelDate, modelDB, marketDB, data)
    #         self.generate_currency_exposures(modelDate, modelDB, marketDB, data)
    #     # Generate 0/1 industry exposures
    #     self.generate_industry_exposures(modelDate, modelDB, marketDB, data.exposureMatrix)
    #     # check country,currency and industry exposure
    #     tmp_expo_df = data.exposureMatrix.toDataFrame()
    #     import DC_YD as yd
    #     yd1 = yd.DC_YD()
    #     fac_types = yd1.get_fac_type(tmp_expo_df.columns, output_type='dict')
    #     tmp_expo_df[fac_types['Country']].sum().sort_values()
    #     tmp_expo_df[fac_types['Industry']].sum().sort_values()
    #     tmp_expo_df[fac_types['Country']].sum().sort_values()
    #     # Load estimation universe
    #     estu = self.loadEstimationUniverse(rmi, modelDB, data)
    #
    #     # Create intercept factor
    #     if self.intercept is not None:
    #         beta = numpy.ones((len(data.universe)), float)
    #         data.exposureMatrix.addFactor(self.intercept.name, beta, ExposureMatrix.InterceptFactor)
    #
    #     # Build all style exposures
    #     descriptorData = self.generate_style_exposures(modelDate, data, modelDB, marketDB)
    #     # import pandas as pd
    #     # mean_desc_df = pd.DataFrame(descriptorData.meanDict)
    #     # std_desc_df = pd.DataFrame(descriptorData.stdDict)
    #     tmp_df = data.exposureMatrix.toDataFrame()
    #     fac_types = yd1.get_fac_type(tmp_df.columns,'dict')
    #     estu_sub_ids = [x.getSubIDString() for x in estu]
    #     tmp_df2 = tmp_df[fac_types['Style']].reindex(estu)
    #     tmp_df2.index = estu_sub_ids
    #
    #     mdl_universe =[x.getSubIDString() for x in data.universe]
    #     estu_mkt_cap = pd.DataFrame(data.marketCaps,index=mdl_universe,columns=['mkt_cap']).reindex(estu_sub_ids)
    #     # weight = estu_mkt_cap/estu_mkt_cap.sum()
    #     # tmp_df2.fillna(0).T.dot(weight)
    #     # np.sqrt((tmp_df2**2).sum()/(len(tmp_df2)-1))
    #     # Shrink some values where there is insufficient history
    #     for st in self.styles:
    #         params = self.styleParameters.get(st.name, None)
    #         if (params is None) or (not  hasattr(params, 'shrinkValue')):
    #             continue
    #         fIdx = data.exposureMatrix.getFactorIndex(st.name)
    #         values = data.exposureMatrix.getMatrix()[fIdx]
    #         # Check and warn of missing values
    #         missingIdx = numpy.flatnonzero(ma.getmaskarray(values))
    #         if len(missingIdx) > 0:
    #             missingSIDs = numpy.take(data.universe, missingIdx, axis=0)
    #             missingSIDs_notnursery = list(set(missingSIDs).difference(data.nurseryUniverse))
    #             self.log.warning('%d assets have missing %s data', len(missingIdx), st.description)
    #             self.log.warning('%d non-nursery assets have missing %s data', len(missingSIDs_notnursery), st.description)
    #             self.log.info('Subissues: %s', missingSIDs)
    #             if (len(missingSIDs_notnursery) > 5) and not self.forceRun:
    #                 import DC_YD as yd
    #                 yd1 = yd.DC_YD()
    #                 fac_types = yd1.get_fac_type(data.exposureMatrix.factors_)
    #                 cntry_factors = [x for x,y in zip(data.exposureMatrix.factors_,fac_types) if y == 'Country']
    #                 cntry_expo = data.exposureMatrix.toDataFrame().reindex(missingSIDs_notnursery)[cntry_factors]
    #                 missingSIDs_cntry_expo = [row_val.T.dropna().index[0] for sub_obj,row_val in cntry_expo.iterrows()]
    #                 nurseryCountries = [x.name for x in self.nurseryCountries]
    #                 real_missingSIDs_notnursery = [x for x in missingSIDs_cntry_expo if x not in nurseryCountries]
    #                 if (len(real_missingSIDs_notnursery) > 5):
    #                     assert (len(missingSIDs_notnursery)==0)
    #                     sub_meta2 = yd1.get_subissue_meta2([x.getSubIDString() for x in real_missingSIDs_notnursery],modelDate)
    #                     from IPython import embed; embed(header='Debug');import ipdb;ipdb.set_trace()
    #
    #         testNew = False
    #         if self.regionalDescriptorStructure and testNew:
    #             shrunkValues = self.shrink_to_mean(modelDate, data, modelDB, marketDB,
    #                     st.name, params.daysBack, values, missingIdx, onlyIPOs=False)
    #         else:
    #             shrunkValues = self.shrink_to_mean(modelDate, data, modelDB, marketDB,
    #                     st.name, params.daysBack, values, missingIdx)
    #         data.exposureMatrix.getMatrix()[fIdx] = shrunkValues
    #
    #     # Clone DR and cross-listing exposures if required
    #     scores = self.load_ISC_Scores(modelDate, data, modelDB, marketDB)
    #     self.group_linked_assets(modelDate, data, modelDB, marketDB)
    #     data.exposureMatrix = self.clone_linked_asset_exposures(
    #             modelDate, data, modelDB, marketDB, scores)
    #
    #     if self.debuggingReporting:
    #         dr_indices = [data.assetIdxMap[sid] for sid in data.foreign]
    #         data.exposureMatrix.dumpToFile('tmp/raw-expM-%s-%04d%02d%02d.csv'\
    #                 % (self.name, modelDate.year, modelDate.month, modelDate.day),
    #                 modelDB, marketDB, modelDate, estu=data.estimationUniverseIdx, assetData=data, dp=self.dplace)
    #
    #     tmpDebug = self.debuggingReporting
    #     self.debuggingReporting = False
    #     self.standardizeExposures(data.exposureMatrix, data, modelDate, modelDB, marketDB, data.subIssueGroups)
    #
    #     # Orthogonalise where required
    #     orthogDict = dict()
    #     for st in self.styles:
    #         params = self.styleParameters[st.name]
    #         if hasattr(params, 'orthog'):
    #             if not hasattr(params, 'sqrtWt'):
    #                 params.sqrtWt = True
    #             if not hasattr(params, 'orthogCoef'):
    #                 params.orthogCoef = 1.0
    #             if params.orthog is not None and len(params.orthog) > 0:
    #                 orthogDict[st.name] = (params.orthog, params.orthogCoef, params.sqrtWt)
    #
    #     if len(orthogDict) > 0:
    #         Utilities.partial_orthogonalisation(modelDate, data, modelDB, marketDB, orthogDict)
    #         tmpExcNames = list(self.exposureStandardization.exceptionNames)
    #         self.exposureStandardization.exceptionNames = [st.name for st in self.styles if st.name not in orthogDict]
    #         self.standardizeExposures(data.exposureMatrix, data, modelDate,
    #                     modelDB, marketDB, data.subIssueGroups)
    #         self.exposureStandardization.exceptionNames = tmpExcNames
    #     self.debuggingReporting = tmpDebug
    #
    #     expMatrix = data.exposureMatrix.getMatrix()
    #     fail = False
    #
    #     for st in self.styles:
    #         params = self.styleParameters[st.name]
    #         # Here we have two parameters that do essentially the same thing
    #         # 'fillWithZero' is intended to cover items like Dividend Yield, where a large number
    #         # of observations are genuinely missing
    #         # 'fillMissing' is a failsafe for exposures that shouldn't normally have any missing values,
    #         # but given the vagaries of global data, may have some from time to time
    #         if (hasattr(params, 'fillWithZero') and (params.fillWithZero is True)) or \
    #                 (hasattr(params, 'fillMissing') and (params.fillMissing is True)):
    #             fIdx = data.exposureMatrix.getFactorIndex(st.name)
    #             for scope in self.exposureStandardization.factorScopes:
    #                 if st.name in scope.factorNames:
    #                     for (bucket, assetIndices) in scope.getAssetIndices(data.exposureMatrix, modelDate):
    #                         values = expMatrix[fIdx, assetIndices]
    #                         nMissing = numpy.flatnonzero(ma.getmaskarray(values))
    #                         if len(nMissing) > 0:
    #                             denom = ma.filled(data.exposureMatrix.stdDict[bucket][st.name], 0.0)
    #                             if abs(denom) > 1.0e-6:
    #                                 fillValue = (0.0 - data.exposureMatrix.meanDict[bucket][st.name]) / denom
    #                                 expMatrix[fIdx,assetIndices] = ma.filled(values, fillValue)
    #                                 logging.info('Filling %d missing values for %s with standardised zero: %.2f for region %s',
    #                                         len(nMissing), st.name, fillValue, bucket)
    #                             else:
    #                                 logging.warning('Zero/missing standard deviation %s for %s for region %s',
    #                                     data.exposureMatrix.stdDict[bucket][st.name], st.name, bucket)
    #
    #     if self.debuggingReporting:
    #         dr_indices = [data.assetIdxMap[sid] for sid in data.foreign]
    #         data.exposureMatrix.dumpToFile('tmp/expM-%s-%04d%02d%02d.csv'\
    #                 % (self.name, modelDate.year, modelDate.month, modelDate.day),
    #                 modelDB, marketDB, modelDate, estu=data.estimationUniverseIdx, assetData=data, dp=self.dplace)
    #
    #     # Check for exposures with all missing values
    #     for st in self.styles:
    #         fIdx = data.exposureMatrix.getFactorIndex(st.name)
    #         values = Utilities.screen_data(expMatrix[fIdx,:])
    #         missingIdx = numpy.flatnonzero(ma.getmaskarray(values))
    #         if len(missingIdx) > 0:
    #             self.log.warning('Style factor %s has %d missing exposures', st.name, len(missingIdx))
    #         nonMissingIdx = numpy.flatnonzero(ma.getmaskarray(values)==0)
    #         if len(nonMissingIdx) < 1:
    #             self.log.error('All %s values are missing', st.description)
    #             if not self.forceRun:
    #                 assert(len(nonMissingIdx)>0)
    #     from IPython import embed; embed(header='Debug:generateExposureMatrix');import ipdb;ipdb.set_trace()
    #     exp_mat = data.exposureMatrix.toDataFrame()
    #     exp_mat.index = [x.getSubIDString() for x in exp_mat.index]
    #     exp_mat = exp_mat.reindex(estu_sub_ids)
    #     exp_mat[['Volatility','Kuwait']].dropna()
    #     # exp_mat.fillna(0).corr().loc['Volatility'].sort_values()
    #     exposure_df = exp_mat
    #     # compute VIF for a given factor
    #     yd1 = yd.DC_YD()
    #     fac_types = yd1.get_fac_type(exposure_df.columns,'dict')
    #
    #     sel_Y = exposure_df['Volatility'].reindex(estu_sub_ids)
    #     sel_X = exposure_df.drop(['Volatility']+fac_types['Currency'],axis=1).reindex(estu_sub_ids)
    #     weight = np.sqrt(estu_mkt_cap).reindex(estu_sub_ids)
    #     compute_VIF(Y_df,X_df,weight)
    #     reg2 = sm.WLS(sel_Y, sel_X.fillna(0), weights=weight).fit()
    #     VIF = 1.0/(1.0-reg2.rsquared)
    #     # vif_result = []
    #     # for sel_cntry in fac_types['Country']:
    #     #     sel_X = exposure_df[fac_types['Style']+fac_types['Industry']+[sel_cntry]].drop(['Volatility'],axis=1).reindex(estu_sub_ids)
    #     #     weight = np.sqrt(estu_mkt_cap).reindex(estu_sub_ids)
    #     #     reg2 = sm.WLS(sel_Y, sel_X.fillna(0), weights=weight).fit()
    #     #     VIF = 1.0/(1.0-reg2.rsquared)
    #     #     vif_result.append([sel_cntry,VIF])
    #     # vif_result_df = pd.DataFrame(vif_result,columns=['Country','VIF']).sort_values('VIF')
    #     # tmp1 = exposure_df[['Volatility','Kuwait']].reindex(estu_sub_ids).dropna()
    #     # tmp_w = estu_mkt_cap.reindex(tmp1.index)
    #     # tmp2 = pd.concat([tmp1,tmp_w/tmp_w.sum()],axis=1)
    #     self.log.debug('generateExposureMatrix: end')
    #     return [data, descriptorData]
    #
    # def generateFactorSpecificReturns(self, modelDB, marketDB, modelDate,
    #                                 buildFMPs=False, internalRun=False, cointTest=False,
    #                                 weeklyRun=False):
    #     """Generates the factor and specific returns for the given
    #     date.  Assumes that the factor exposures for the previous
    #     trading day exist as those will be used for the exposures.
    #     Returns a Struct with factorReturns, specificReturns, exposureMatrix,
    #     regressionStatistics, and adjRsquared.  The returns are
    #     arrays matching the factor and assets in the exposure matrix.
    #     exposureMatrix is an ExposureMatrix object.  The regression
    #     coefficients is a two-dimensional masked array
    #     where the first dimension is the number of factors used in the
    #     regression, and the second is the number of statistics for
    #     each factor that are stored in the array.
    #     """
    #     # Testing parameters for FMPs
    #     testFMPs = False
    #     nextTradDate = None
    #
    #     # Important regression parameters
    #     applyRT = False
    #     if buildFMPs:
    #         logging.info('Generating FMPs')
    #         # Set up parameters for FMP calculation
    #         prevDate = modelDate
    #         if testFMPs:
    #             futureDates = modelDB.getDateRange(self.rmg, modelDate,
    #                     modelDate+datetime.timedelta(20), excludeWeekend=True)
    #             nextTradDate = futureDates[1]
    #
    #         if not hasattr(self, 'fmpCalculator'):
    #             logging.warning('No FMP parameters set up, skipping')
    #             return None
    #         rcClass = self.fmpCalculator
    #     else:
    #         # Set up parameters for factor returns regression
    #         if weeklyRun:
    #             startDate = modelDate - datetime.timedelta(7)
    #             dateList = modelDB.getDateRange(
    #                     self.rmg, startDate, modelDate, excludeWeekend=True)
    #         else:
    #             dateList = modelDB.getDates(self.rmg, modelDate, 1, excludeWeekend=True)
    #         if len(dateList) < 2:
    #             raise LookupError, (
    #                 'No previous trading day for %s' %  str(modelDate))
    #         prevDate = dateList[0]
    #
    #         if internalRun:
    #             logging.info('Generating internal factor returns')
    #             # Regression for internal factor returns
    #             if not hasattr(self, 'internalCalculator'):
    #                 logging.warning('No internal factor return parameters set up, skipping')
    #                 return None
    #             rcClass = self.internalCalculator
    #             if self.multiCountry:
    #                 applyRT = True
    #         elif weeklyRun:
    #             logging.info('Generating weekly factor returns from %s to %s',
    #                     prevDate, modelDate)
    #             # Regression for weekly factor returns
    #             if not hasattr(self, 'weeklyCalculator'):
    #                 logging.warning('No weekly factor return parameters set up, skipping')
    #                 return None
    #             rcClass = self.weeklyCalculator
    #             applyRT = False
    #         else:
    #             # Regression for public factor returns
    #             logging.info('Generating external (public) factor returns')
    #             rcClass = self.returnCalculator
    #
    #     # Get exposure matrix for previous trading day
    #     rmi = modelDB.getRiskModelInstance(self.rms_id, prevDate)
    #     if rmi == None:
    #         raise LookupError, (
    #             'no risk model instance for %s' % str(prevDate))
    #     if not rmi.has_exposures:
    #         raise LookupError, (
    #             'no exposures in risk model instance for %s' % str(prevDate))
    #     self.setFactorsForDate(prevDate, modelDB)
    #
    #     # Determine home country info and flag DR-like instruments
    #     universe = modelDB.getRiskModelInstanceUniverse(rmi, returnExtra=True)
    #     data = AssetProcessor.process_asset_information(
    #             prevDate, universe, self.rmg, modelDB, marketDB,
    #             checkHomeCountry=self.multiCountry, numeraire_id=self.numeraire.currency_id,
    #             legacyDates=self.legacyMCapDates,
    #             forceRun=self.forceRun, nurseryRMGList=self.nurseryRMGs,
    #             tweakDict=self.tweakDict)
    #
    #     # Load previous day's exposure matrix
    #     expM = self.loadExposureMatrix(rmi, modelDB, addExtraCountries=True, assetList=data.universe)
    #     prevFactors = self.factors + self.nurseryCountries
    #     prevSubFactors = modelDB.getSubFactorsForDate(prevDate, prevFactors)
    #     nameSubIDMap = dict([(s.factor.name, s.subFactorID) for s in prevSubFactors])
    #
    #     # Get map of current day's factor IDs
    #     self.setFactorsForDate(modelDate, modelDB)
    #     allFactors = self.factors + self.nurseryCountries
    #     subFactors = modelDB.getSubFactorsForDate(modelDate, allFactors)
    #     subFactorIDIdxMap = dict([(subFactors[i].subFactorID, i) for i in xrange(len(subFactors))])
    #     deadFactorNames = [s.factor.name for s in prevSubFactors if s not in subFactors]
    #     deadFactorIdx = [expM.getFactorIndex(n) for n in deadFactorNames]
    #     if len(deadFactorIdx) > 0:
    #         self.log.warning('Dropped factors %s on %s', deadFactorNames, modelDate)
    #
    #     # Get main estimation universe for previous day
    #     estu = self.loadEstimationUniverse(rmi, modelDB, data)
    #     if 'nursery' in self.estuMap:
    #         logging.info('Adding %d nursery assets to main estimation universe',
    #                 len(self.estuMap['nursery'].assets))
    #         estu = estu + self.estuMap['nursery'].assets
    #         self.estuMap['main'].assets = estu
    #         logging.info('Main estimation universe now %d assets', len(estu))
    #     estuIdx = [data.assetIdxMap[sid] for sid in estu]
    #     self.estuMap['main'].assetIdx = estuIdx
    #     data.estimationUniverseIdx = estuIdx
    #     rmi = modelDB.getRiskModelInstance(self.rms_id, modelDate)
    #
    #     # Load asset returns
    #     if self.multiCountry:
    #         if cointTest:
    #             historyLength = self.cointHistory
    #             modelDB.notTradedIndCache = ModelDB.TimeSeriesCache(int(1.5*historyLength))
    #             assetReturnMatrix = self.assetReturnHistoryLoader(
    #                     data, historyLength, modelDate, modelDB, marketDB, cointTest=True)
    #             return
    #         elif internalRun:
    #             assetReturnMatrix = self.assetReturnHistoryLoader(
    #                     data, 10, modelDate, modelDB, marketDB, loadOnly=True,
    #                     applyRT=applyRT, fixNonTradingDays=True)
    #             missingReturnsMask = assetReturnMatrix.missingFlag
    #             zeroReturnsMask = assetReturnMatrix.zeroFlag
    #         elif weeklyRun:
    #
    #             # Load daily returns and convert to weekly
    #             self.returnHistory = 10
    #             data = self.build_excess_return_history(data, modelDate, modelDB, marketDB,
    #                     loadOnly=True, applyRT=False, fixNonTradingDays=False)
    #             weeklyExcessReturns = Utilities.compute_compound_returns_v4(
    #                     ma.filled(data.returns.data, 0.0), data.returns.dates, [prevDate, modelDate])[0]
    #             # Mask missing weekly returns
    #             nonMissingData = ma.getmaskarray(data.returns.data)==0
    #             weeklyDataMask = Utilities.compute_compound_returns_v4(
    #                     nonMissingData.astype(int), data.returns.dates, [prevDate, modelDate])[0]
    #             weeklyExcessReturns = ma.masked_where(weeklyDataMask==0, weeklyExcessReturns)
    #
    #             # Load currency returns in now rather than later
    #             crmi = modelDB.getRiskModelInstance(self.currencyModel.rms_id, modelDate)
    #             assert (crmi is not None)
    #             currencyFactors = self.currencyModel.getCurrencyFactors()
    #             currencySubFactors = modelDB.getSubFactorsForDate(crmi.date, currencyFactors)
    #             currencyReturnsHistory = modelDB.loadFactorReturnsHistory(
    #                     crmi.rms_id, currencySubFactors, data.returns.dates)
    #             # Convert to weekly returns
    #             currencyReturns = Utilities.compute_compound_returns_v4(
    #                     currencyReturnsHistory.data, currencyReturnsHistory.dates, [prevDate, modelDate])[0]
    #             # Mask missing weekly returns
    #             nonMissingData = ma.getmaskarray(currencyReturnsHistory.data)==0
    #             weeklyDataMask = Utilities.compute_compound_returns_v4(
    #                     nonMissingData.astype(int), data.returns.dates, [prevDate, modelDate])[0]
    #             currencyReturns  = ma.masked_where(weeklyDataMask==0, currencyReturns)
    #
    #             # Pull out latest weekly returns
    #             assetReturnMatrix = data.returns
    #             assetReturnMatrix.data = weeklyExcessReturns[:,-1][:,numpy.newaxis]
    #             assetReturnMatrix.dates = [modelDate]
    #             missingReturnsMask = ma.getmaskarray(assetReturnMatrix.data)
    #             currencyReturns = currencyReturns[:,-1]
    #         else:
    #             assetReturnMatrix = self.assetReturnHistoryLoader(
    #                     data, 2, modelDate, modelDB, marketDB, loadOnly=True,
    #                     applyRT=applyRT, fixNonTradingDays=False)
    #             missingReturnsMask = assetReturnMatrix.missingFlag
    #             zeroReturnsMask = assetReturnMatrix.zeroFlag
    #     else:
    #         assetReturnMatrix = self.assetReturnHistoryLoader(
    #                 data, 1, modelDate, modelDB, marketDB, loadOnly=True, applyRT=False)
    #         missingReturnsMask = assetReturnMatrix.missingFlag
    #         zeroReturnsMask = assetReturnMatrix.zeroFlag
    #     if not weeklyRun:
    #         assetReturnMatrix.data = assetReturnMatrix.data[:,-1][:,numpy.newaxis]
    #         assetReturnMatrix.dates = [assetReturnMatrix.dates[-1]]
    #         missingReturnsMask = missingReturnsMask[:,-1]
    #         zeroReturnsMask = zeroReturnsMask[:,-1]
    #
    #     # Do some checking on missing and zero returns
    #     missingReturnsIdx = numpy.flatnonzero(missingReturnsMask)
    #     logging.info('%d out of %d genuine missing returns in total', len(missingReturnsIdx), len(missingReturnsMask))
    #     zeroReturnsIdx = numpy.flatnonzero(zeroReturnsMask)
    #     logging.info('%d out of %d zero returns in total', len(zeroReturnsIdx), len(zeroReturnsMask))
    #     stillMissingReturnsIdx = numpy.flatnonzero(ma.getmaskarray(assetReturnMatrix.data))
    #     if len(stillMissingReturnsIdx) > 0:
    #         logging.info('%d missing total returns filled with zero', len(stillMissingReturnsIdx))
    #
    #     # Do some checking on estimation universe returns
    #     suspectDay = False
    #     badRetList = list()
    #     # Report on missing returns
    #     missingESTURets = numpy.flatnonzero(ma.take(missingReturnsMask, estuIdx, axis=0))
    #     badRetList.extend(numpy.take(data.universe, missingESTURets, axis=0))
    #     propnBadRets = len(missingESTURets) / float(len(estuIdx))
    #     self.log.info('%.1f%% of %d main ESTU original returns missing', 100.0*propnBadRets, len(estuIdx))
    #     # Report on zero returns
    #     zeroESTURets = numpy.flatnonzero(ma.take(zeroReturnsMask, estuIdx, axis=0))
    #     badRetList.extend(numpy.take(data.universe, zeroESTURets, axis=0))
    #     propnBadRets = len(zeroESTURets) / float(len(estuIdx))
    #     self.log.info('%.1f%% of %d main ESTU final returns zero', 100.0*propnBadRets, len(estuIdx))
    #
    #     # if internalRun:
    #     #     self.badRetList = list(set(badRetList))
    #
    #     # If too many of both, set t-stats to be nuked
    #     propnBadRets = (len(zeroESTURets) + len(missingESTURets)) / float(len(estuIdx))
    #     if propnBadRets > 0.5:
    #         logging.info('Regression statistics suspect and will be nuked')
    #         suspectDay = True
    #
    #     # Compute excess returns
    #     assetReturnMatrix.data = ma.filled(assetReturnMatrix.data, 0.0)
    #     if not weeklyRun:
    #         (assetReturnMatrix, rfr) = self.computeExcessReturns(modelDate,
    #                                 assetReturnMatrix, modelDB, marketDB, data.drCurrData)
    #         for i in xrange(len(rfr.assets)):
    #             if rfr.data[i,0] is not ma.masked:
    #                 self.log.debug('Using risk-free rate of %f%% for %s',
    #                         rfr.data[i,0] * 100.0, rfr.assets[i])
    #     excessReturns = assetReturnMatrix.data[:,0]
    #
    #     # FMP testing stuff
    #     if nextTradDate is not None:
    #         futureReturnMatrix = returnsProcessor.process_returns_history(
    #                 nextTradDate, 1, modelDB, marketDB,
    #                 drCurrMap=data.drCurrData, loadOnly=True,
    #                 applyRT=False, trimData=False)
    #         futureReturnMatrix.data = futureReturnMatrix.data[:,-1][:,numpy.newaxis]
    #         futureReturnMatrix.dates = [futureReturnMatrix.dates[-1]]
    #         futureReturnMatrix.data = ma.filled(futureReturnMatrix.data, 0.0)
    #
    #         (futureReturnMatrix, rfr) = self.computeExcessReturns(nextTradDate,
    #                 futureReturnMatrix, modelDB, marketDB, data.drCurrData)
    #         futureExcessReturns = futureReturnMatrix.data[:,0]
    #
    #     # Report on markets with non-trading day or all missing returns
    #     # Such will have their returns replaced either with zero or a proxy
    #     nonTradingMarketsIdx = []
    #     nukeTStatList = []
    #     totalESTUMarketCaps = ma.sum(ma.take(data.marketCaps, data.estimationUniverseIdx, axis=0), axis=None)
    #     tradingCaps = 0.0
    #     for r in self.rmg:
    #
    #         # Pull out assets for each RMG
    #         if r.rmg_id not in data.rmgAssetMap:
    #             rmg_indices = []
    #         else:
    #             rmg_indices = [data.assetIdxMap[n] for n in \
    #                             data.rmgAssetMap[r.rmg_id].intersection(self.estuMap['main'].assets)]
    #         rmg_returns = ma.take(excessReturns, rmg_indices, axis=0)
    #
    #         # Get missing returns (before any proxying) and calendar dates
    #         noOriginalReturns = numpy.sum(ma.take(missingReturnsMask, rmg_indices, axis=0), axis=None)
    #         rmgCalendarList = modelDB.getDateRange(r, assetReturnMatrix.dates[0], assetReturnMatrix.dates[-1])
    #
    #         if noOriginalReturns >= 0.95 * len(rmg_returns) or modelDate not in rmgCalendarList:
    #             # Do some reporting and manipulating for NTD markets
    #             nukeTStatList.append(r.description)
    #             rmgMissingIdx = list(set(stillMissingReturnsIdx).intersection(set(rmg_indices)))
    #             if len(rmgMissingIdx) > 0:
    #                 self.log.info('Non-trading day for %s, %d/%d returns missing',
    #                             r.description, noOriginalReturns, len(rmg_returns))
    #             else:
    #                 self.log.info('Non-trading day for %s, %d/%d returns imputed',
    #                         r.description, noOriginalReturns, len(rmg_returns))
    #         else:
    #             rmg_caps = ma.sum(ma.take(data.marketCaps, rmg_indices, axis=0), axis=None)
    #             tradingCaps += rmg_caps
    #
    #     # Report on % of market trading today
    #     pcttrade = tradingCaps / totalESTUMarketCaps
    #     logging.info('Proportion of total ESTU market trading: %.2f', pcttrade)
    #
    #     # Get industry asset buckets
    #     data.industryAssetMap = dict()
    #     for idx in expM.getFactorIndices(ExposureMatrix.IndustryFactor):
    #         assetsIdx = numpy.flatnonzero(expM.getMatrix()[idx,:])
    #         data.industryAssetMap[idx] = numpy.take(data.universe, assetsIdx, axis=0)
    #
    #     # Get indices of factors that we don't want in the regression
    #     if self.multiCountry:
    #         currencyFactorsIdx = expM.getFactorIndices(ExposureMatrix.CurrencyFactor)
    #         excludeFactorsIdx = list(set(deadFactorIdx + currencyFactorsIdx + nonTradingMarketsIdx))
    #     else:
    #         excludeFactorsIdx = deadFactorIdx
    #
    #     # Remove any remaining empty style factors
    #     for idx in expM.getFactorIndices(ExposureMatrix.StyleFactor):
    #         assetsIdx = numpy.flatnonzero(expM.getMatrix()[idx,:])
    #         if len(assetsIdx) == 0:
    #             self.log.warning('100%% empty factor, excluded from all regressions: %s', expM.getFactorNames()[idx])
    #             excludeFactorsIdx.append(idx)
    #         else:
    #             propn = len(assetsIdx) / float(len(data.universe))
    #             if propn < 0.01:
    #                 self.log.warning('%.1f%% exposures non-missing, excluded from all regressions: %s',
    #                         100*propn, expM.getFactorNames()[idx])
    #                 excludeFactorsIdx.append(idx)
    #
    #     # Call nested regression routine
    #     # from IPython import embed; embed(header='Debug:factor regression returnData0');import ipdb;ipdb.set_trace()
    #     returnData = rcClass.run_factor_regressions(
    #             self, rcClass, prevDate, excessReturns, expM, estu, data,
    #             excludeFactorsIdx, modelDB, marketDB, applyRT=applyRT, fmpRun=buildFMPs)
    #     ##############################################################################################################################################
    #     # import pandas as pd
    #     # fac_rets = pd.DataFrame(returnData.factorReturnsMap.items(),columns=['factor_name','factor_return'])
    #     # fac_rets2 = fac_rets.sort_values('factor_return')
    #     # fac_names = expM_df.columns
    #     #
    #     # expM_df = expM.toDataFrame()
    #     # asset_names = [x.getSubIDString() for x in expM_df.index]
    #     # expM_df.index = asset_names
    #     # estu_names = [x.getSubIDString() for x in estu]
    #     # estu_expM_df = expM_df.reindex(estu_names)
    #     # bah_assets = estu_expM_df['Bahrain'].dropna()
    #     #
    #     # rets_df = pd.Series(excessReturns,index=asset_names)
    #     # rets_df.reindex(bah_assets.index)
    #     #
    #     # [x for x,y in zip(excessReturns,asset_names) if y in ['D4UMZN7SY511','DFFPSVFZN811']]
    #     # from IPython import embed; embed(header='Debug:factor regression returnData');import ipdb;ipdb.set_trace()
    #     # Map specific returns for cloned assets
    #     returnData.specificReturns = ma.masked_where(missingReturnsMask, returnData.specificReturns)
    #     if len(data.hardCloneMap) > 0:
    #         cloneList = set(data.hardCloneMap.keys()).intersection(set(data.universe))
    #         for sid in cloneList:
    #             if data.hardCloneMap[sid] in data.universe:
    #                 returnData.specificReturns[data.assetIdxMap[sid]] = returnData.specificReturns\
    #                         [data.assetIdxMap[data.hardCloneMap[sid]]]
    #
    #     # Store regression results
    #     factorReturns = Matrices.allMasked((len(allFactors),))
    #     regressionStatistics = Matrices.allMasked((len(allFactors), 4))
    #     for (fName, ret) in returnData.factorReturnsMap.items():
    #         idx = subFactorIDIdxMap.get(nameSubIDMap[fName], None)
    #         if idx is not None:
    #             factorReturns[idx] = ret
    #             if (not suspectDay) and (fName not in nukeTStatList):
    #                 regressionStatistics[idx,:] = returnData.regStatsMap[fName]
    #             else:
    #                 regressionStatistics[idx,-1] = returnData.regStatsMap[fName][-1]
    #     if not internalRun and not buildFMPs:
    #         # Calculate Variance Inflation Factors for each style factor regressed on other style factors.
    #         self.VIF = rcClass.VIF
    #
    #     result = Utilities.Struct()
    #     result.universe = data.universe
    #     result.factorReturns = factorReturns
    #     result.specificReturns = returnData.specificReturns
    #     result.exposureMatrix = expM
    #     result.regressionStatistics = regressionStatistics
    #     result.adjRsquared = returnData.anova.calc_adj_rsquared()
    #     result.pcttrade = pcttrade
    #     result.regression_ESTU = zip([result.universe[i] for i in returnData.anova.estU_],
    #             returnData.anova.weights_ / numpy.sum(returnData.anova.weights_))
    #     result.VIF = self.VIF
    #
    #     # Process robust weights
    #     newRWtMap = dict()
    #     sid2StringMap = dict([(sid if isinstance(sid, basestring) else sid.getSubIDString(), sid) for sid in data.universe])
    #     for (iReg, rWtMap) in returnData.robustWeightMap.items():
    #         tmpMap = dict()
    #         for (sidString, rWt) in rWtMap.items():
    #             if sidString in sid2StringMap:
    #                 tmpMap[sid2StringMap[sidString]] = rWt
    #         newRWtMap[iReg] = tmpMap
    #     result.robustWeightMap = newRWtMap
    #
    #     # Process FMPs
    #     newFMPMap = dict()
    #     for (fName, fmpMap) in returnData.fmpMap.items():
    #         tmpMap = dict()
    #         for (sidString, fmp) in fmpMap.items():
    #             if sidString in sid2StringMap:
    #                 tmpMap[sid2StringMap[sidString]] = fmp
    #         newFMPMap[nameSubIDMap[fName]] = tmpMap
    #     result.fmpMap = newFMPMap
    #
    #     # Report non-trading markets and set factor return to zero
    #     allFactorNames = expM.getFactorNames()
    #     if len(nonTradingMarketsIdx) > 0:
    #         nonTradingMarketNames = ', '.join([allFactorNames[i] \
    #                 for i in nonTradingMarketsIdx])
    #         self.log.info('%d non-trading market(s): %s',
    #                 len(nonTradingMarketsIdx), nonTradingMarketNames)
    #         for i in nonTradingMarketsIdx:
    #             idx = subFactorIDIdxMap[nameSubIDMap[allFactorNames[i]]]
    #             result.factorReturns[idx] = 0.0
    #
    #     # Pull in currency factor returns from currency model
    #     if self.multiCountry:
    #         crmi = modelDB.getRiskModelInstance(self.currencyModel.rms_id, modelDate)
    #         assert (crmi is not None)
    #         if not weeklyRun:
    #             (currencyReturns, currencyFactors) = \
    #                     self.currencyModel.loadCurrencyReturns(crmi, modelDB)
    #         currSubFactors = modelDB.getSubFactorsForDate(modelDate, currencyFactors)
    #         currSubIDIdxMap = dict([(currSubFactors[i].subFactorID, i) \
    #                                 for i in xrange(len(currSubFactors))])
    #         self.log.info('loaded %d currencies from currency model', len(currencyFactors))
    #
    #         # Lookup currency factor returns from currency model
    #         currencyFactors = set(self.currencies)
    #         for (i,j) in subFactorIDIdxMap.items():
    #             cidx = currSubIDIdxMap.get(i, None)
    #             if cidx is None:
    #                 if allFactors[j] in currencyFactors:
    #                     self.log.warning('Missing currency factor return for %s', allFactors[j].name)
    #                     value = 0.0
    #                 else:
    #                     continue
    #             else:
    #                 value = currencyReturns[cidx]
    #             result.factorReturns[j] = value
    #
    #     constrComp = Utilities.Struct()
    #     constrComp.ccDict = returnData.ccMap
    #     constrComp.ccXDict = returnData.ccXMap
    #
    #     if nextTradDate is not None:
    #         self.regressionReporting(futureExcessReturns, result, expM, nameSubIDMap, data.assetIdxMap,
    #                         modelDate, buildFMPs=buildFMPs, constrComp=constrComp, specificRets=result.specificReturns)
    #     else:
    #         self.regressionReporting(excessReturns, result, expM, nameSubIDMap, data.assetIdxMap,
    #                         modelDate, buildFMPs=buildFMPs, constrComp=constrComp, specificRets=result.specificReturns)
    #
    #     if self.debuggingReporting:
    #         for (i,sid) in enumerate(data.universe):
    #             if abs(returnData.specificReturns[i]) > 1.5:
    #                 self.log.info('Large specific return for: %s, ret: %s',
    #                         sid, returnData.specificReturns[i])
    #     return result

class EM4_Research5(EM4_Research2):
    """
        EM research model - no regional standardization
        EM research model - GICS Industry Group for industry factors
        EM research model - GICS Industry Group for industry factors with weekly returns for risk forecast
        EM research model - base model - before release - deal with bad assets: prevDefault
        EM research model - same EM21 ESTU and Exposure and same factor strucutre as EM21  
    """
    rm_id,revision,rms_id = [-22,5,-224]

    styleList = ['Value',
                 'Leverage',
                 'Growth',
                 'Size',
                 'Liquidity',
                 'Volatility',
                 'Medium-Term Momentum',
                 'Short-Term Momentum',
                 'Exchange Rate Sensitivity',
                 ]
    DescriptorMap = {
            # 'Earnings Yield': ['Earnings_to_Price_Annual', 'Est_Earnings_to_Price_12MFL_Annual'],
            'Value': ['Book_to_Price_Annual'],
            'Leverage': ['Debt_to_Assets_Annual', 'Debt_to_Equity_Annual'],
            'Growth': ['Earnings_Growth_RPF_Annual', 'Sales_Growth_RPF_Annual'],
            # 'Dividend Yield': ['Dividend_Yield_Annual'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_60D', 'Amihud_Liquidity_125D', 'ISC_Ret_Score'],
            # 'Market Sensitivity': ['EM_Regional_Market_Sensitivity_500D_V0'],
            'Volatility': ['Volatility_125D'],
            'Short-Term Momentum': ['Momentum_21D'],
            'Medium-Term Momentum': ['Momentum_260x21D_Regional'],
            'Exchange Rate Sensitivity': ['XRate_104W_XDR'],
            # 'Profitability': ['Return_on_Equity_Annual', 'Return_on_Assets_Annual',
            #     'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
                # 'Sales_to_Assets_Annual', 'Gross_Margin_Annual'],
            }
    ##############################################################################################
    YD_Version_ESTU = True
    YD_Version_EXP = True
    exposureConfigFile = 'exposures-mhV2' # .config
    YD_Version_Factor = False
    YD_Version_Risk = False
    dropZeroNanAsset = True
    ##############################################################################################
    # industryClassification = Classification.GICSIndustryGroups(datetime.date(2016,9,1))
    industryClassification = Classification.GICSIndustryGroups(datetime.date(2008,8,30))

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.EM4_Research5')
        super(EM4_Research5,self).__init__(modelDB, marketDB)

    def generate_model_universe_YD(self, modelDate, modelDB, marketDB):

        self.log.info('[YD]generate_model_universe from EM2: begin')
        data = Utilities.Struct()
        universe = AssetProcessor.getModelAssetMaster(
                 self, modelDate, modelDB, marketDB, legacyDates=self.legacyMCapDates)
        # Throw out any assets with missing size descriptors
        descDict = dict(modelDB.getAllDescriptors())
        sizeVec = self.loadDescriptorData(['LnIssuerCap'], descDict, modelDate,
                        universe, modelDB, None, rollOver=False)[0]['LnIssuerCap']
        missingSizeIdx = numpy.flatnonzero(ma.getmaskarray(sizeVec))
        if len(missingSizeIdx) > 0:
            missingIds = numpy.take(universe, missingSizeIdx, axis=0)
            universe = list(set(universe).difference(set(missingIds)))
            logging.warning('%d missing LnIssuerCaps dropped from model', len(missingSizeIdx))
        data.universe = universe

        import DC_YD as yd
        yd_sb = yd.DC_YD('glprodsb')
        dt = modelDate
        rm_name_em2 = 'EMAxioma2011MH'

        if dt <= datetime.date(2002,12,31):
            rm_name2_estu = yd_sb.get_model_ESTU('EMAxioma2011MH_Pre2003', dt)
        elif dt <= datetime.date(2008,12,31):
            rm_name2_estu = yd_sb.get_model_ESTU('EMAxioma2011MH_Pre2009', dt)
        elif dt <= datetime.date(2012,12,31):
            rm_name2_estu = yd_sb.get_model_ESTU('EMAxioma2011MH_Pre2013', dt)
        else:
            rm_name2_estu = yd_sb.get_model_ESTU(rm_name_em2, dt)

        rm_name2_estu = [x for x in rm_name2_estu if isinstance(x, str)]
        import DC_YD as yd
        yd1 = yd.DC_YD()
        rm_name4_nursery = yd1.get_model_ESTU('EM4_Research1', dt,estu_type='nursery')
        rm_name4_ChinaA = yd1.get_model_ESTU('EM4_Research1', dt,estu_type='ChinaA')

        rm_name2_estu = list(set(rm_name2_estu).difference(rm_name4_nursery).difference(rm_name4_ChinaA))
        rm_name2_estu = yd_sb.get_subissue_objs(rm_name2_estu)
        rm_name4_nursery = yd_sb.get_subissue_objs(rm_name4_nursery)
        rm_name4_ChinaA = yd_sb.get_subissue_objs(rm_name4_ChinaA)

        self.estuMap['main'].assets = rm_name2_estu
        self.estuMap['main'].qualify = rm_name2_estu
        self.estuMap['nursery'].assets = rm_name4_nursery
        self.estuMap['nursery'].qualify = rm_name4_nursery
        self.estuMap['ChinaA'].assets = rm_name4_ChinaA
        self.estuMap['ChinaA'].qualify = rm_name4_ChinaA
        self.log.info('[YD]generate_model_universe from EM2: end.')
        return data

    def generateExposureMatrix_YD(self, modelDate, modelDB, marketDB):
        """Generates and returns the exposure matrix for the given date.
        The exposures are not inserted into the database.
        Data is accessed through the modelDB and marketDB DAOs.
        The return is a structure containing the exposure matrix
        (exposureMatrix), the universe as a list of assets (universe),
        and a list of market capitalizations (marketCaps).
        """
        import pandas as pd
        self.log.info('generateExposureMatrix: begin')
        # Get risk model universe and market caps
        # Determine home country info and flag DR-like instruments
        from IPython import embed; embed(header='Debug:generateExposureMatrix_YD:Begin');import ipdb;ipdb.set_trace()
        rmi = modelDB.getRiskModelInstance(self.rms_id, modelDate)
        universe = modelDB.getRiskModelInstanceUniverse(rmi, returnExtra=True)
        data = AssetProcessor.process_asset_information(
                modelDate, universe, self.rmg, modelDB, marketDB,
                checkHomeCountry=self.multiCountry,
                numeraire_id=self.numeraire.currency_id,
                legacyDates=self.legacyMCapDates,
                forceRun=self.forceRun,
                nurseryRMGList=self.nurseryRMGs,
                tweakDict=self.tweakDict)
        data.exposureMatrix = Matrices.ExposureMatrix(data.universe)

        mdl_universe = [x.getSubIDString() for x in data.universe]
        import DC_YD as yd
        yd1 = yd.DC_YD()
        yd_sb = yd.DC_YD('glprodsb')
        dt = modelDate
        if dt <= datetime.date(2002,12,31):
            rm_name_em2 = 'EMAxioma2011MH_Pre2003'
        elif dt <= datetime.date(2008,12,31):
            rm_name_em2 = 'EMAxioma2011MH_Pre2009'
        elif dt <= datetime.date(2012,12,31):
            rm_name_em2 = 'EMAxioma2011MH_Pre2013'
        else:
            rm_name_em2 = 'EMAxioma2011MH'

        # facs = yd_sb.get_model_factors(rm_name_em2,dateobj = dt)
        fac_expo_mat = yd_sb.get_fac_exposure_matrix(rm_name_em2, dt, convertAxiomaID=True)
        fac_expo_mat.loc['DCN2TB1NM511','Volatility']
        # em4_fac_dict = yd1.get_model_factors('EM4_Research5', dt, out_type='dict') # misuse of mutiple version of risk model
        # from IPython import embed; embed(header='Debug:generateExposureMatrix_YD0');import ipdb;ipdb.set_trace()
        if hasattr(self, 'nurseryCountries') and (len(self.nurseryCountries) > 0):
            factors = self.factors + self.nurseryCountries + self.hiddenCurrencies
        elif hasattr(self, 'hiddenCurrencies') and (len(self.hiddenCurrencies) > 0):
            factors = self.factors + self.hiddenCurrencies
        else:
            factors = self.factors

        rmi_obj = modelDB.getRiskModelInstance(self.rms_id,dt)
        subFactors = modelDB.getRiskModelInstanceSubFactors(rmi_obj, factors)
        fac_list = [x.factor.name for x in subFactors]

        # styles = [x.description for x in self.styles]
        # industries = [x.description for x in self.industries]
        # countries = [x.description for x in self.countries]
        # if not hasattr(self, 'currencies'):# fix old single model
        #     currencies = []
        # else:
        #     currencies = [x.name for x in self.currencies]
        #blind = [x.name for x in self.blind]
        # fac_list = ['Market Intercept'] + styles + countries + industries + currencies

        fac_expo_mat2 = fac_expo_mat.rename(columns={'Global Market':'Market Intercept'})
        fac_expo_mat3 = fac_expo_mat2.reindex(columns=fac_list).fillna(0)
        # fac_expo_mat4 = fac_expo_mat3.reindex([x.getSubIDString() for x in data.universe])
        missing_sub_ids = list(set([x.getSubIDString() for x in data.universe]).difference(fac_expo_mat3.index))

        # generate country exposure for missing sud ids
        self.log.info('generate country exposures: begin')
        cntry_expo_list =[]
        for rmg_id, rmg_assets in data.rmgAssetMap.items():
            tmp_sub_ids = [x.getSubIDString() for x in list(rmg_assets)]
            tmp_df = pd.DataFrame([rmg_id]*len(tmp_sub_ids),index=tmp_sub_ids,columns=['RMG_ID'])
            cntry_expo_list.append(tmp_df)
        cntry_expo_df = pd.concat(cntry_expo_list,axis=0)
        cntry_expo_df.index.name = 'SubID'
        cntry_expo_df = cntry_expo_df.reset_index()
        cntry_expo_df['Value'] = 1.0
        cntry_expo_df2 = cntry_expo_df.pivot(index='SubID', columns='RMG_ID', values='Value')
        # change from rmg_id to rmg_name, which is country factor name
        rmg_id_dict = dict((x.rmg_id,x.description) for x in self.rmg)
        cntry_expo_df2.columns = [rmg_id_dict[x] for x in cntry_expo_df2.columns]
        missing_cntry_list = list(set(rmg_id_dict.values()).difference(cntry_expo_df2.columns))
        for missing_cntry in missing_cntry_list:
            cntry_expo_df2[missing_cntry] = np.nan
        cntry_expo_df2 = cntry_expo_df2.sort_index(axis=1)
        ##############################################################################################################################################
        # generate currency exposure
        self.log.info('generate currency exposures: begin')
        # self.generate_currency_exposures(modelDate, modelDB, marketDB, data)
        currency_expo_list =[]
        for rmg_id, rmc_assets in data.rmcAssetMap.items():
            tmp_sub_ids = [x.getSubIDString() for x in list(rmc_assets)]
            tmp_df = pd.DataFrame([rmg_id]*len(tmp_sub_ids),index=tmp_sub_ids,columns=['RMG_ID'])
            currency_expo_list.append(tmp_df)
        currency_expo_df = pd.concat(currency_expo_list,axis=0)
        currency_expo_df.index.name = 'SubID'
        currency_expo_df = currency_expo_df.reset_index()
        currency_expo_df['Value'] = 1.0
        currency_expo_df2 = currency_expo_df.pivot(index='SubID', columns='RMG_ID', values='Value')
        # change from rmg_id to rmg_currency_code, which is currency factor name
        rmg_id_currency_dict = dict((x.rmg_id,x.currency_code) for x in self.rmg)
        currency_expo_df2.columns = [rmg_id_currency_dict[x] for x in currency_expo_df2.columns]
        # get all currency factors: adding the hidden currency factors
        all_currency = [x.name for x in (self.currencies + self.hiddenCurrencies)]
        missing_currency = list(set(all_currency).difference(currency_expo_df2.columns))
        for each_cur in missing_currency:
            currency_expo_df2[each_cur] = np.nan
        currency_expo_df2 = currency_expo_df2[all_currency]
        # mutiple rmg_id can be mapped to one curreny:
        from collections import Counter
        cnt1 = Counter(currency_expo_df2.columns)
        muti_currency_list = [x for x,y in cnt1.items() if y>1]
        for sel_cur in muti_currency_list:
            tmp_df = currency_expo_df2[sel_cur].sum(axis=1)
            tmp_df[tmp_df==0] = np.nan
            currency_expo_df2 = currency_expo_df2.drop(sel_cur,axis=1)
            currency_expo_df2[sel_cur] = tmp_df
        currency_expo_df2 = currency_expo_df2.sort_index(axis=1)
        ##############################################################################################################################################
        # self.generate_industry_exposures(modelDate, modelDB, marketDB, data.exposureMatrix)
        self.log.info('generate industry exposures: begin')
        factor_list = [f.description for f in self.industryClassification.getLeafNodes(modelDB).values()]
        ind_exposures = self.industryClassification.getExposures(modelDate, data.universe, factor_list, modelDB)
        ind_exposure_df = pd.DataFrame(ind_exposures,columns=mdl_universe,index=factor_list).T
        # check cntry_expo, currency_expo and industry_expo:
        # print(cntry_expo_df2.sum().sort_values())
        # print(ind_exposure_df.sum().sort_values())
        # print(currency_expo_df2.sum().sort_values())
        ##############################################################################################################################################

        missing_expo = pd.concat([cntry_expo_df2,ind_exposure_df,currency_expo_df2],axis=1).reindex(missing_sub_ids)
        fac_expo_mat4 = pd.concat([fac_expo_mat3,missing_expo],axis=0)
        fac_expo_mat4['Market Intercept'] = 1
        # from IPython import embed; embed(header='Debug:generateExposureMatrix_YD0');import ipdb;ipdb.set_trace()
        data.exposureMatrix = self.convert_exposure_df_to_expM(fac_expo_mat4,data.universe)
        # prepare fake descriptor statistics data for output
        descriptors = sorted([x for sublist in self.DescriptorMap.values() for x in sublist])
        desc_dict = dict(modelDB.getAllDescriptors())

        descriptorData = Utilities.Struct()
        descriptorData.descriptors = descriptors
        descriptorData.descDict = desc_dict
        descriptorData.meanDict = dict() # dict of dict
        descriptorData.stdDict = dict()
        descriptorData.DescriptorWeights = dict()
        desc_stats_rg_list = ['Europe','Asia ex-Pacific','Africa','Middle East','Universe','Latin America']
        for key in desc_stats_rg_list:
            descriptorData.meanDict[key] = dict([(x,0.0) for x in descriptors])
            descriptorData.stdDict[key] = dict([(x,1.0) for x in descriptors])
        # from IPython import embed; embed(header='Debug:generateExposureMatrix_YD');import ipdb;ipdb.set_trace()
        return [data, descriptorData]

class EM4_Research6(EM4_Research1):
    """
        EM research model - base model - remove some countries that traded on Saterday instead of Friday
        EM research model - base model + with aggressive ESTU setting - 0.9 + default GICS
        EM research model - regional-based style factors (for all style factors) - 2017-12-26 
    """
    rm_id,revision,rms_id = [-22,6,-225]
    industryClassification = Classification.GICSCustomEM(datetime.date(2016,9,1))

    styleList = ['Value Europe','Value Asia exPacific','Value Africa','Value Middle East','Value Latin America',
                 'Leverage Europe','Leverage Asia exPacific','Leverage Africa','Leverage Middle East','Leverage Latin America',
                 'Growth Europe','Growth Asia exPacific','Growth Africa','Growth Middle East','Growth Latin America',
                 'Profitability Europe','Profitability Asia exPacific','Profitability Africa','Profitability Middle East','Profitability Latin America',
                 'Earnings Yield Europe','Earnings Yield Asia exPacific','Earnings Yield Africa','Earnings Yield Middle East','Earnings Yield Latin America',
                 'Dividend Yield Europe','Dividend Yield Asia exPacific','Dividend Yield Africa','Dividend Yield Middle East','Dividend Yield Latin America',
                 'Size Europe','Size Asia exPacific','Size Africa','Size Middle East','Size Latin America',
                 'Liquidity Europe','Liquidity Asia exPacific','Liquidity Africa','Liquidity Middle East','Liquidity Latin America',
                 'Market Sensitivity Europe','Market Sensitivity Asia exPacific','Market Sensitivity Africa','Market Sensitivity Middle East','Market Sensitivity Latin America',
                 'Volatility Europe','Volatility Asia exPacific','Volatility Africa','Volatility Middle East','Volatility Latin America',
                 'Medium-Term Momentum Europe','Medium-Term Momentum Asia exPacific','Medium-Term Momentum Africa','Medium-Term Momentum Middle East','Medium-Term Momentum Latin America',
                 'Exchange Rate Sensitivity Europe','Exchange Rate Sensitivity Asia exPacific','Exchange Rate Sensitivity Africa','Exchange Rate Sensitivity Middle East','Exchange Rate Sensitivity Latin America',
                 ]

    DescriptorMap = {
            'Earnings Yield Europe': ['Earnings_to_Price_Annual', 'Est_Earnings_to_Price_12MFL_Annual'],
            'Earnings Yield Asia exPacific': ['Earnings_to_Price_Annual', 'Est_Earnings_to_Price_12MFL_Annual'],
            'Earnings Yield Africa': ['Earnings_to_Price_Annual', 'Est_Earnings_to_Price_12MFL_Annual'],
            'Earnings Yield Middle East': ['Earnings_to_Price_Annual', 'Est_Earnings_to_Price_12MFL_Annual'],
            'Earnings Yield Latin America': ['Earnings_to_Price_Annual', 'Est_Earnings_to_Price_12MFL_Annual'],
            'Value Europe': ['Book_to_Price_Annual'],
            'Value Asia exPacific': ['Book_to_Price_Annual'],
            'Value Africa': ['Book_to_Price_Annual'],
            'Value Middle East': ['Book_to_Price_Annual'],
            'Value Latin America': ['Book_to_Price_Annual'],
            'Leverage Europe': ['Debt_to_Assets_Annual', 'Debt_to_Equity_Annual'],
            'Leverage Asia exPacific': ['Debt_to_Assets_Annual', 'Debt_to_Equity_Annual'],
            'Leverage Africa': ['Debt_to_Assets_Annual', 'Debt_to_Equity_Annual'],
            'Leverage Middle East': ['Debt_to_Assets_Annual', 'Debt_to_Equity_Annual'],
            'Leverage Latin America': ['Debt_to_Assets_Annual', 'Debt_to_Equity_Annual'],
            'Growth Europe': ['Earnings_Growth_RPF_Annual', 'Sales_Growth_RPF_Annual'],
            'Growth Asia exPacific': ['Earnings_Growth_RPF_Annual', 'Sales_Growth_RPF_Annual'],
            'Growth Africa': ['Earnings_Growth_RPF_Annual', 'Sales_Growth_RPF_Annual'],
            'Growth Middle East': ['Earnings_Growth_RPF_Annual', 'Sales_Growth_RPF_Annual'],
            'Growth Latin America': ['Earnings_Growth_RPF_Annual', 'Sales_Growth_RPF_Annual'],            
            'Dividend Yield Europe': ['Dividend_Yield_Annual'],
            'Dividend Yield Asia exPacific': ['Dividend_Yield_Annual'],
            'Dividend Yield Africa': ['Dividend_Yield_Annual'],
            'Dividend Yield Middle East': ['Dividend_Yield_Annual'],
            'Dividend Yield Latin America': ['Dividend_Yield_Annual'],            
            'Size Europe': ['LnIssuerCap'],
            'Size Asia exPacific': ['LnIssuerCap'],
            'Size Africa': ['LnIssuerCap'],
            'Size Middle East': ['LnIssuerCap'],
            'Size Latin America': ['LnIssuerCap'],
            'Liquidity Europe': ['LnTrading_Activity_60D', 'Amihud_Liquidity_125D', 'ISC_Ret_Score'],
            'Liquidity Asia exPacific': ['LnTrading_Activity_60D', 'Amihud_Liquidity_125D', 'ISC_Ret_Score'],
            'Liquidity Africa': ['LnTrading_Activity_60D', 'Amihud_Liquidity_125D', 'ISC_Ret_Score'],
            'Liquidity Middle East': ['LnTrading_Activity_60D', 'Amihud_Liquidity_125D', 'ISC_Ret_Score'],
            'Liquidity Latin America': ['LnTrading_Activity_60D', 'Amihud_Liquidity_125D', 'ISC_Ret_Score'],            
            'Market Sensitivity Europe': ['EM_Regional_Market_Sensitivity_500D_V0'],
            'Market Sensitivity Asia exPacific': ['EM_Regional_Market_Sensitivity_500D_V0'],
            'Market Sensitivity Africa': ['EM_Regional_Market_Sensitivity_500D_V0'],
            'Market Sensitivity Middle East': ['EM_Regional_Market_Sensitivity_500D_V0'],
            'Market Sensitivity Latin America': ['EM_Regional_Market_Sensitivity_500D_V0'],
            'Volatility Europe': ['Volatility_125D'],
            'Volatility Asia exPacific': ['Volatility_125D'],
            'Volatility Africa': ['Volatility_125D'],
            'Volatility Middle East': ['Volatility_125D'],
            'Volatility Latin America': ['Volatility_125D'],            
            'Medium-Term Momentum Europe': ['Momentum_250x20D'],
            'Medium-Term Momentum Asia exPacific': ['Momentum_250x20D'],
            'Medium-Term Momentum Africa': ['Momentum_250x20D'],
            'Medium-Term Momentum Middle East': ['Momentum_250x20D'],
            'Medium-Term Momentum Latin America': ['Momentum_250x20D'],
            'Exchange Rate Sensitivity Europe': ['XRate_104W_XDR'], #'XRate_104W_USD',XRate_104W_XDR
            'Exchange Rate Sensitivity Asia exPacific': ['XRate_104W_XDR'],
            'Exchange Rate Sensitivity Africa': ['XRate_104W_XDR'],
            'Exchange Rate Sensitivity Middle East': ['XRate_104W_XDR'],
            'Exchange Rate Sensitivity Latin America': ['XRate_104W_XDR'],            
            'Profitability Europe': ['Return_on_Equity_Annual', 'Return_on_Assets_Annual',
                'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
                'Sales_to_Assets_Annual', 'Gross_Margin_Annual'],
            'Profitability Asia exPacific': ['Return_on_Equity_Annual', 'Return_on_Assets_Annual',
                'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
                'Sales_to_Assets_Annual', 'Gross_Margin_Annual'],
            'Profitability Africa': ['Return_on_Equity_Annual', 'Return_on_Assets_Annual',
                'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
                'Sales_to_Assets_Annual', 'Gross_Margin_Annual'],
            'Profitability Middle East': ['Return_on_Equity_Annual', 'Return_on_Assets_Annual',
                'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
                'Sales_to_Assets_Annual', 'Gross_Margin_Annual'],
            'Profitability Latin America': ['Return_on_Equity_Annual', 'Return_on_Assets_Annual',
                'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
                'Sales_to_Assets_Annual', 'Gross_Margin_Annual']                                                
            }
    exposureConfigFile = 'exposures-mh_YD.csv'
    regionalStndList = [ 'Value Europe','Value Asia exPacific','Value Africa','Value Middle East','Value Latin America',
                         'Leverage Europe','Leverage Asia exPacific','Leverage Africa','Leverage Middle East','Leverage Latin America',
                         'Growth Europe','Growth Asia exPacific','Growth Africa','Growth Middle East','Growth Latin America',
                         'Profitability Europe','Profitability Asia exPacific','Profitability Africa','Profitability Middle East','Profitability Latin America',
                         'Earnings Yield Europe','Earnings Yield Asia exPacific','Earnings Yield Africa','Earnings Yield Middle East','Earnings Yield Latin America',
                         'Dividend Yield Europe','Dividend Yield Asia exPacific','Dividend Yield Africa','Dividend Yield Middle East','Dividend Yield Latin America',
                         'Size Europe','Size Asia exPacific','Size Africa','Size Middle East','Size Latin America',
                         'Liquidity Europe','Liquidity Asia exPacific','Liquidity Africa','Liquidity Middle East','Liquidity Latin America',
                         'Market Sensitivity Europe','Market Sensitivity Asia exPacific','Market Sensitivity Africa','Market Sensitivity Middle East','Market Sensitivity Latin America',
                         'Volatility Europe','Volatility Asia exPacific','Volatility Africa','Volatility Middle East','Volatility Latin America',
                         'Medium-Term Momentum Europe','Medium-Term Momentum Asia exPacific','Medium-Term Momentum Africa','Medium-Term Momentum Middle East','Medium-Term Momentum Latin America',
                         'Exchange Rate Sensitivity Europe','Exchange Rate Sensitivity Asia exPacific','Exchange Rate Sensitivity Africa','Exchange Rate Sensitivity Middle East','Exchange Rate Sensitivity Latin America',
                         ]# 'Payout']

    _regionalMapper = lambda dm, l: [dm[st] for st in l]
    regionalStndDesc = list(itertools.chain.from_iterable(_regionalMapper(DescriptorMap, regionalStndList)))
    del _regionalMapper
    orthogList = {'Volatility Europe': [['Market Sensitivity Europe'], True, 1.0],
                  'Volatility Asia exPacific': [['Market Sensitivity Asia exPacific'], True, 1.0],
                  'Volatility Africa': [['Market Sensitivity Africa'], True, 1.0],
                  'Volatility Middle East': [['Market Sensitivity Middle East'], True, 1.0],
                  'Volatility Latin America': [['Market Sensitivity Latin America'], True, 1.0],}

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.EM4_Research6')
        super(EM4_Research6,self).__init__(modelDB, marketDB)
        # Set important model parameters
        # self.elig_parameters['exclude_HomeCountry_List'] = ['QA','BD','BH','SA','JO',
        #                                                     'KW','OM','PS','AE','EG']

    def generateExposureMatrix(self, modelDate, modelDB, marketDB):
        print('EM4_Research6================================================================================')
        # data = super(EM4_Research6,self).generateExposureMatrix(modelDate, modelDB, marketDB)
        data = self.generateExposureMatrixRaw(modelDate, modelDB, marketDB)
        # from IPython import embed; embed(header='Debug');import ipdb;ipdb.set_trace()
        reg_fac_dict = {
                        'Europe':['Value Europe','Leverage Europe','Growth Europe','Profitability Europe','Earnings Yield Europe','Dividend Yield Europe','Size Europe','Liquidity Europe','Market Sensitivity Europe','Volatility Europe','Medium-Term Momentum Europe','Exchange Rate Sensitivity Europe'],
                        'Asia ex-Pacific':['Value Asia exPacific','Leverage Asia exPacific','Growth Asia exPacific','Profitability Asia exPacific','Earnings Yield Asia exPacific','Dividend Yield Asia exPacific','Size Asia exPacific','Liquidity Asia exPacific','Market Sensitivity Asia exPacific','Volatility Asia exPacific','Medium-Term Momentum Asia exPacific','Exchange Rate Sensitivity Asia exPacific'],
                        'Africa':['Value Africa','Leverage Africa','Growth Africa','Profitability Africa','Earnings Yield Africa','Dividend Yield Africa','Size Africa','Liquidity Africa','Market Sensitivity Africa','Volatility Africa','Medium-Term Momentum Africa','Exchange Rate Sensitivity Africa'],
                        'Middle East':['Value Middle East','Leverage Middle East','Growth Middle East','Profitability Middle East','Earnings Yield Middle East','Dividend Yield Middle East','Size Middle East','Liquidity Middle East','Market Sensitivity Middle East','Volatility Middle East','Medium-Term Momentum Middle East','Exchange Rate Sensitivity Middle East'],
                        'Latin America':['Value Latin America','Leverage Latin America','Growth Latin America','Profitability Latin America','Earnings Yield Latin America','Dividend Yield Latin America','Size Latin America','Liquidity Latin America','Market Sensitivity Latin America','Volatility Latin America','Medium-Term Momentum Latin America','Exchange Rate Sensitivity Latin America']
                        }
        tmpdata = data[0]
        reg_scopes = self.descriptorStandardization.factorScopes[0]
        for (bucketDesc, assetIndices) in reg_scopes.getAssetIndices(tmpdata.exposureMatrix, modelDate):
            if bucketDesc in reg_fac_dict.keys():
                fac_name_list = reg_fac_dict[bucketDesc]
                for fac_name in fac_name_list:
                    print(('%s:%s' % (bucketDesc,fac_name)))
                    fac_idx = tmpdata.exposureMatrix.getFactorIndex(fac_name)
                    non_asset_idx = list(set(range(len(tmpdata.exposureMatrix.assets_))).difference(assetIndices))
                    tmpdata.exposureMatrix.data_[fac_idx,non_asset_idx]=0
        data[0] = tmpdata
        return data

    def generateExposureMatrixRaw(self, modelDate, modelDB, marketDB):
        """
            Generates and returns the exposure matrix for the given date.
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
                checkHomeCountry=self.multiCountry,
                numeraire_id=self.numeraire.currency_id,
                legacyDates=self.legacyMCapDates,
                forceRun=self.forceRun,
                nurseryRMGList=self.nurseryRMGs,
                tweakDict=self.tweakDict)
        data.exposureMatrix = Matrices.ExposureMatrix(data.universe)

        if (not self.multiCountry) and (not hasattr(self, 'indexSelector')):
            self.indexSelector = MarketIndex.MarketIndexSelector(modelDB, marketDB)
            self.log.info('Index Selector: %s', self.indexSelector)

        # Generate eligible universe
        data.eligibleUniverse = self.generate_eligible_universe(
                modelDate, data, modelDB, marketDB)

        # Fetch trading calendars for all risk model groups
        # Start-date should depend on how long a history is required
        # for exposures computation
        data.rmgCalendarMap = dict()
        startDate = modelDate - datetime.timedelta(365*2)
        for rmg in self.rmg:
            data.rmgCalendarMap[rmg.rmg_id] = \
                    modelDB.getDateRange(rmg, startDate, modelDate)

        # Compute issuer-level market caps if required
        AssetProcessor.computeTotalIssuerMarketCaps(
                data, modelDate, self.numeraire, modelDB, marketDB,
                debugReport=self.debuggingReporting)

        if self.multiCountry:
            self.generate_binary_country_exposures(modelDate, modelDB, marketDB, data)
            self.generate_currency_exposures(modelDate, modelDB, marketDB, data)

        # Generate 0/1 industry exposures
        self.generate_industry_exposures(
            modelDate, modelDB, marketDB, data.exposureMatrix)

        # Load estimation universe
        estu = self.loadEstimationUniverse(rmi, modelDB, data)

        # Create intercept factor
        if self.intercept is not None:
            beta = numpy.ones((len(data.universe)), float)
            data.exposureMatrix.addFactor(self.intercept.name, beta, ExposureMatrix.InterceptFactor)

        # Build all style exposures
        descriptorData = self.generate_style_exposures(modelDate, data, modelDB, marketDB)

        # Shrink some values where there is insufficient history
        for st in self.styles:
            params = self.styleParameters.get(st.name, None)
            if (params is None) or (not  hasattr(params, 'shrinkValue')):
                continue
            fIdx = data.exposureMatrix.getFactorIndex(st.name)
            values = data.exposureMatrix.getMatrix()[fIdx]
            # Check and warn of missing values
            missingIdx = numpy.flatnonzero(ma.getmaskarray(values))
            if len(missingIdx) > 0:
                missingSIDs = numpy.take(data.universe, missingIdx, axis=0)
                missingSIDs_notnursery = list(set(missingSIDs).difference(data.nurseryUniverse))
                self.log.warning('%d assets have missing %s data', len(missingIdx), st.description)
                self.log.warning('%d non-nursery assets have missing %s data', len(missingSIDs_notnursery), st.description)
                self.log.info('Subissues: %s', missingSIDs)
                if (len(missingSIDs_notnursery) > 5) and not self.forceRun:
                    import DC_YD as yd
                    yd1 = yd.DC_YD()
                    fac_types = yd1.get_fac_type(data.exposureMatrix.factors_)
                    cntry_factors = [x for x,y in zip(data.exposureMatrix.factors_,fac_types) if y == 'Country']
                    cntry_expo = data.exposureMatrix.toDataFrame().reindex(missingSIDs_notnursery)[cntry_factors]
                    missingSIDs_cntry_expo = [row_val.T.dropna().index[0] for sub_obj,row_val in cntry_expo.iterrows()]
                    nurseryCountries = [x.name for x in self.nurseryCountries]
                    real_missingSIDs_notnursery = [x for x in missingSIDs_cntry_expo if x not in nurseryCountries]
                    if (len(real_missingSIDs_notnursery) > 5):
                        assert (len(missingSIDs_notnursery)==0)
                        sub_meta2 = yd1.get_subissue_meta2([x.getSubIDString() for x in real_missingSIDs_notnursery],modelDate)
                        from IPython import embed; embed(header='Debug');import ipdb;ipdb.set_trace()

            testNew = False
            if self.regionalDescriptorStructure and testNew:
                shrunkValues = self.shrink_to_mean(modelDate, data, modelDB, marketDB,
                        st.name, params.daysBack, values, missingIdx, onlyIPOs=False)
            else:
                shrunkValues = self.shrink_to_mean(modelDate, data, modelDB, marketDB,
                        st.name, params.daysBack, values, missingIdx)
            data.exposureMatrix.getMatrix()[fIdx] = shrunkValues

        # Clone DR and cross-listing exposures if required
        scores = self.load_ISC_Scores(modelDate, data, modelDB, marketDB)
        self.group_linked_assets(modelDate, data, modelDB, marketDB)
        data.exposureMatrix = self.clone_linked_asset_exposures(
                modelDate, data, modelDB, marketDB, scores)

        if self.debuggingReporting:
            dr_indices = [data.assetIdxMap[sid] for sid in data.foreign]
            data.exposureMatrix.dumpToFile('tmp/raw-expM-%s-%04d%02d%02d.csv'\
                    % (self.name, modelDate.year, modelDate.month, modelDate.day),
                    modelDB, marketDB, modelDate, estu=data.estimationUniverseIdx, assetData=data, dp=self.dplace)

        tmpDebug = self.debuggingReporting
        self.debuggingReporting = False
        self.standardizeExposures(data.exposureMatrix, data, modelDate, modelDB, marketDB, data.subIssueGroups)

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
                        modelDB, marketDB, data.subIssueGroups)
            self.exposureStandardization.exceptionNames = tmpExcNames
        self.debuggingReporting = tmpDebug

        expMatrix = data.exposureMatrix.getMatrix()
        fail = False

        for st in self.styles:
            params = self.styleParameters[st.name]
            # Here we have two parameters that do essentially the same thing
            # 'fillWithZero' is intended to cover items like Dividend Yield, where a large number
            # of observations are genuinely missing
            # 'fillMissing' is a failsafe for exposures that shouldn't normally have any missing values,
            # but given the vagaries of global data, may have some from time to time
            if (hasattr(params, 'fillWithZero') and (params.fillWithZero is True)) or \
                    (hasattr(params, 'fillMissing') and (params.fillMissing is True)):
                fIdx = data.exposureMatrix.getFactorIndex(st.name)
                for scope in self.exposureStandardization.factorScopes:
                    if st.name in scope.factorNames:
                        for (bucket, assetIndices) in scope.getAssetIndices(data.exposureMatrix, modelDate):
                            values = expMatrix[fIdx, assetIndices]
                            nMissing = numpy.flatnonzero(ma.getmaskarray(values))
                            if len(nMissing) > 0:
                                denom = ma.filled(data.exposureMatrix.stdDict[bucket][st.name], 0.0)
                                if abs(denom) > 1.0e-6:
                                    fillValue = (0.0 - data.exposureMatrix.meanDict[bucket][st.name]) / denom
                                    expMatrix[fIdx,assetIndices] = ma.filled(values, fillValue)
                                    logging.info('Filling %d missing values for %s with standardised zero: %.2f for region %s',
                                            len(nMissing), st.name, fillValue, bucket)
                                else:
                                    logging.warning('Zero/missing standard deviation %s for %s for region %s',
                                        data.exposureMatrix.stdDict[bucket][st.name], st.name, bucket)

        if self.debuggingReporting:
            dr_indices = [data.assetIdxMap[sid] for sid in data.foreign]
            data.exposureMatrix.dumpToFile('tmp/expM-%s-%04d%02d%02d.csv'\
                    % (self.name, modelDate.year, modelDate.month, modelDate.day),
                    modelDB, marketDB, modelDate, estu=data.estimationUniverseIdx, assetData=data, dp=self.dplace)

        # Check for exposures with all missing values
        for st in self.styles:
            fIdx = data.exposureMatrix.getFactorIndex(st.name)
            values = Utilities.screen_data(expMatrix[fIdx,:])
            missingIdx = numpy.flatnonzero(ma.getmaskarray(values))
            if len(missingIdx) > 0:
                self.log.warning('Style factor %s has %d missing exposures', st.name, len(missingIdx))
            nonMissingIdx = numpy.flatnonzero(ma.getmaskarray(values)==0)
            if len(nonMissingIdx) < 1:
                self.log.error('All %s values are missing', st.description)
                if not self.forceRun:
                    assert(len(nonMissingIdx)>0)
        self.log.debug('generateExposureMatrix: end')

        return [data, descriptorData]

class EM4_Research7(EM4_Research1):
    """
        EM research model - base model - for research different methods - failed due to no space
    """
    rm_id,revision,rms_id = [-22,4,-227]
    industryClassification = Classification.GICSCustomEM(datetime.date(2016,9,1))

    DescriptorMap = {
            'Earnings Yield': ['Earnings_to_Price_Annual', 'Est_Earnings_to_Price_12MFL_Annual'],
            'Value': ['Book_to_Price_Annual'],
            'Leverage': ['Debt_to_Assets_Annual', 'Debt_to_Equity_Annual'],
            'Growth': ['Earnings_Growth_RPF_Annual', 'Sales_Growth_RPF_Annual'],
            'Dividend Yield': ['Dividend_Yield_Annual'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_60D', 'Amihud_Liquidity_125D', 'ISC_Ret_Score'],
            'Market Sensitivity': ['EM_Regional_Market_Sensitivity_500D_V0'],
            'Volatility': ['Volatility_125D'],
            'Medium-Term Momentum': ['Momentum_260x21D_Regional'],
            'Exchange Rate Sensitivity': ['XRate_104W_XDR'],
            'Profitability': ['Return_on_Equity_Annual', 'Return_on_Assets_Annual',
                'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
                'Sales_to_Assets_Annual', 'Gross_Margin_Annual'],
            }
    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.EM4_Research7')
        super(EM4_Research7,self).__init__(modelDB, marketDB)
######################################################################################################################################################
# statistical model
######################################################################################################################################################

class EM4_Stat_R1(EquityModel.StatisticalModel):
    """EM research statistical model - base model
    """
    rm_id = -23
    revision = 1
    rms_id = -226
    numFactors = 20
    blind = [ModelFactor('Statistical Factor %d' % n, 'Statistical Factor %d' % n) for n in range(1, numFactors+1)]
    pcaHistory = 250
    gicsDate = datetime.date(2016,9,1)
    industryClassification = Classification.GICSIndustries(gicsDate)
    descriptorNumeraire = 'USD'

    # estuTweakList = {'DTMNFLCQ6311': 'DUVNMNNZ5211', 'D7G6HUSH4511': 'DP4RXG9DN411',
    #                  'DTHKZKJTQ211': 'DJFFZGW94011', 'DFGC28XA2711': 'DZHZPNFFG311',
    #                  'DW4J9X3UJ211': 'DWVNNSF28011', 'DXXCJBQY4411': 'DCF5QYY4J611',
    #                  'DN7453TDT211': 'DPM44PTQ9611', 'DRFGJNZKU011': 'DXJ4JZNSL411',
    #                  'DXBGSLNKL111': 'DRY4ZTVBM711', 'DB6K55P6H911': 'DFBTF4J5X411',
    #                  'DGX665CTG511': 'DLD9C42AV711', 'DXRBBLQCW611': 'D5L4JVPTN411',
    #                  'DFTS3QX5U411': 'DTWFGXQLT911', 'DVT1GPK3A111': 'DFMV4MB11311',
    #                  'DQYD27H6X811': 'DJLY5G61A311', 'DH9YTVD16711': 'DVVDDC254611',
    #                  'DCCUQLHPA211': 'DV683DBMB311', 'DSN5UM9F2611': 'DUQ125NRV311',
    #                  'D7X94KQVH411': 'DJ7HG7ZKV811', 'DRPLRDH9F111': 'D12UCV16X211'}

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.EM4_Stat_R1')
        # Set important model parameters
        ModelParameters2017.defaultModelSettings(self, scm=False, statModel=True)
        EquityModel.StatisticalModel.__init__(self, ['SEDOL', 'CUSIP'], modelDB, marketDB)
        # So we can use the same ESTU method as the fundamental model
        self.baseModelDateMap = {datetime.date(1980,1,1): EM4_Research4(modelDB, marketDB)}
        # Set up estimation universe parameters
        self.elig_parameters = ModelParameters2017.defaultRegionalModelEstuParameters(self)[1]
        self.statModelEstuTol = 0.9
        # Set Calculators
        self.setCalculators(modelDB)
        # Define currency model
        self.currencyModel = riskmodels.getModelByName('FXAxioma2017USD_MH')(modelDB, marketDB)

        # Manually reassign select assets to RMG and currency
        self.tweakDict = dict()
        # Force all RDS issues to have GB exposure
        # self.tweakDict['CIJBKGWL8'] = [datetime.date(1980,1,1), datetime.date(2999,12,31), 'GB', 'GB']

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
