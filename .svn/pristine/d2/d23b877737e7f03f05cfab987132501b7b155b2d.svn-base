
import datetime
import logging
import numpy as np
import numpy.ma as ma
import numpy
import pandas
import itertools

import riskmodels
from riskmodels.RiskModels import AUAxioma2016MH
from riskmodels import Classification
from riskmodels import EstimationUniverse
from riskmodels.Matrices import ExposureMatrix
from riskmodels import CurrencyRisk
from riskmodels import GlobalExposures
from riskmodels import Matrices
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


class AU2Yilin(EquityModel.FundamentalModel):
    """AU2 base model.
    """
    rm_id = -20
    revision = 12
    rms_id = -126
    k  = 5.0
    inflation_cutoff = 0.05
    dummyThreshold = 10     # dummyThreshold = 6

    regionalDescriptorStructure = False
    twoRegressionStructure = False
    multiCountry = False

    # parameters used for MFM class - to remove
    # standardizationStats = True
    # globalDescriptorModel = True
    # DLCEnabled = True
    # SCM = True

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
            'Earnings Yield': ['Earnings_to_Price_Annual', 'Est_Earnings_to_Price_Annual'],
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
            'Profitability': ['Return_on_Equity_Annual','Return_on_Assets_Annual', 'CashFlow_to_Assets_Annual',
                              'CashFlow_to_Income_Annual','Sales_to_Assets_Annual', 'Gross_Margin_Annual'],
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
    industryClassification = Classification.GICSCustomAU(datetime.date(2008,8,30))
    estuAssetTypes = ['REIT', 'Com', 'StapSec']


    def __init__(self, modelDB, marketDB, expTreat=None):
        self.log = logging.getLogger('RiskModels.AU2Yilin')
        # Set up relevant styles to be created/used
        # ModelParameters.defaultExposureParameters(self, self.styleList)
        ModelParameters2017.defaultExposureParameters(self, self.styleList)
        # if expTreat == 'addNew':
        #     self.styles = [s for s in self.totalStyles if s.name in self.addList]
        #     self.variableStyles = True
        # else:
        #     self.styles = [s for s in self.totalStyles if s.name in self.styleList]
        self.styles = [s for s in self.totalStyles if s.name in self.styleList]
        EquityModel.FundamentalModel.__init__(self, ['SEDOL'], modelDB, marketDB)
        # MFM.FundamentalModel.__init__(self, ['CUSIP'], modelDB, marketDB) # it doesn't work!

        # Set up regression parameters

        self.returnCalculator = ModelParameters2017.defaultRegressionParameters(self, modelDB,dummyType='Sectors', dummyThreshold=self.dummyThreshold,marketRegression=False,kappa = self.k)

        # This controls the FMP regression
        self.fmpCalculator = ModelParameters2017.defaultRegressionParameters(self, modelDB,dummyType='Sectors',marketRegression=False,dummyThreshold=self.dummyThreshold,kappa=None)

        # Set up risk parameters
        ModelParameters2017.defaultFundamentalCovarianceParameters(self, nwLag=2,varDVAOnly=False, unboundedDVA=False)

        # Set up standardization parameters
        gloScope = Standardization.GlobalRelativeScope([f.name for f in self.styles])
        # self.exposureStandardization = Standardization.BucketizedStandardization([gloScope], fancyMAD=self.fancyMAD)
        self.exposureStandardization = Standardization.BucketizedStandardization([gloScope], fillWithZeroList=self.fillWithZeroList)

        self.indexSelector = MarketIndex.MarketIndexSelector(modelDB, marketDB)
        modelDB.createCurrencyCache(marketDB)

        # Set up descriptor standardization parameters
        descriptors = sorted(list(set([item for sublist in self.DescriptorMap.values() for item in sublist])))
        exceptionNames = [self.DescriptorMap[sf] for sf in self.noProxyList]
        exceptionNames = list(itertools.chain.from_iterable(exceptionNames))
        # from IPython import embed;  embed(header='')
        # if not self.SCM:
        #     self.descriptorStandardization = Standardization.BucketizedStandardization(
        #             [Standardization.RegionRelativeScope(modelDB, descriptors)],
        #             mad_bound=15.0, fancyMAD=self.fancyMAD, exceptionNames=exceptionNames)
        # else:
        self.descriptorStandardization = Standardization.BucketizedStandardization(
                [Standardization.GlobalRelativeScope(descriptors)],
                mad_bound=15.0, fancyMAD=self.fancyMAD, exceptionNames=exceptionNames)

    def generate_eligible_universe(self, modelDate, data, modelDB, marketDB, capBounds=None,assetTypes=['All-Com', 'REIT'],excludeTypes=['ComWI', 'UnCls', 'StatETF', 'LLC', 'LP', 'NonEqETF', 'ComETF']):
        """Creates subset of eligible assets for consideration
        in estimation universes
        """
        # import ipdb;ipdb.set_trace()
        self.log.info('generate_eligible_universe: begin')
        HomeCountry_List = ['AU','NZ']
        buildEstu = EstimationUniverse.ConstructEstimationUniverse(
                data.universe, self, modelDB, marketDB)
        universe = buildEstu.assets
        n = len(universe)
        stepNo = 0
        if assetTypes is None:
            assetTypes = []
        if excludeTypes is None:
            excludeTypes = []
        logging.info('Eligible Universe currently stands at %d stocks', n)

        # Remove assets from the exclusion table
        stepNo+=1
        logging.info('Step %d: Applying exclusion table', stepNo)
        (estuIdx, nonest) = buildEstu.apply_exclusion_list(modelDate)
        if n != len(estuIdx):
            logging.info('...Step %d: Eligible Universe down %d and currently stands at %d stocks',
                    stepNo, n-len(estuIdx), len(estuIdx))
            n = len(estuIdx)

        # Remove cloned assets
        if len(data.hardCloneMap) > 0:
            stepNo+=1
            logging.info('Step %d: Removing cloned assets', stepNo)
            cloneIdx = [data.assetIdxMap[sid] for sid in data.hardCloneMap.keys()]
            (estuIdx, nonest) = buildEstu.exclude_specific_assets(cloneIdx, baseEstu=estuIdx)
            if n != len(estuIdx):
                logging.info('...Step %d: Eligible Universe down %d and currently stands at %d stocks',
                        stepNo, n-len(estuIdx), len(estuIdx))
                n = len(estuIdx)

        # Include by asset type field
        stepNo+=1
        logging.info('Step %d: Include by asset types %s', stepNo, ','.join(assetTypes))
        (estuIdx, nonest) = buildEstu.exclude_by_asset_type(
                modelDate, data, includeFields=assetTypes, excludeFields=None,
                baseEstu = estuIdx)
        if n != len(estuIdx):
            logging.info('...Step %d: Eligible Universe down %d and currently stands at %d stocks',
                    stepNo, n-len(estuIdx), len(estuIdx))
            n = len(estuIdx)

        # Pull out Chinese H shares and redchips to save for later
        stepNo+=1
        assetTypes = ['HShares', 'RCPlus']
        logging.info('Step %d: Getting list of Chinese assets: %s', stepNo, ','.join(assetTypes))
        (chinese, nonest) = buildEstu.exclude_by_asset_type(
                modelDate, data, includeFields=assetTypes, excludeFields=None,
                baseEstu = estuIdx)
        if len(chinese) > 0:
            logging.info('...Step %d: Found %d Chinese H-shares and Redchips', stepNo, len(chinese))

        # Exclude by asset type field
        stepNo+=1
        logging.info('Step %d: Exclude default asset types %s', stepNo, ','.join(excludeTypes))
        (estuIdx, nonest) = buildEstu.exclude_by_asset_type(
                modelDate, data, includeFields=None, excludeFields=excludeTypes,
                baseEstu = estuIdx)
        if n != len(estuIdx):
            logging.info('...Step %d: Eligible Universe down %d and currently stands at %d stocks',
                    stepNo, n-len(estuIdx), len(estuIdx))
            n = len(estuIdx)

        # Remove various types of DRs and foreign listings
        stepNo+=1
        exTypes = ['NVDR', 'GlobalDR', 'DR', 'TDR', 'AmerDR', 'FStock', 'CDI']
        logging.info('Step %d: Exclude DR asset types %s', stepNo, ','.join(exTypes))
        (estuIdx, nonest) = buildEstu.exclude_by_asset_type(
                modelDate, data, includeFields=None, excludeFields=exTypes,
                baseEstu = estuIdx)
        if n != len(estuIdx):
            logging.info('...Step %d: Eligible Universe down %d and currently stands at %d stocks',
                    stepNo, n-len(estuIdx), len(estuIdx))
            n = len(estuIdx)

        # Remove trusts, funds and other odds and ends
        stepNo+=1
        exTypes = ['InvT', 'UnitT', 'CEFund', 'Misc']
        logging.info('Step %d: Exclude fund/trust asset types %s', stepNo, ','.join(exTypes))
        (estuIdx, nonest) = buildEstu.exclude_by_asset_type(
                modelDate, data, includeFields=None, excludeFields=exTypes,
                baseEstu = estuIdx)
        if n != len(estuIdx):
            logging.info('...Step %d: Eligible Universe down %d and currently stands at %d stocks',
                    stepNo, n-len(estuIdx), len(estuIdx))
            n = len(estuIdx)

        # Remove Chinese A and B shares
        stepNo+=1
        exTypes = ['AShares', 'BShares']
        logging.info('Step %d: Exclude Chinese asset types %s', stepNo, ','.join(exTypes))
        (estuIdx, nonest) = buildEstu.exclude_by_asset_type(
                modelDate, data, includeFields=None, excludeFields=exTypes,
                baseEstu = estuIdx)
        if n != len(estuIdx):
            logging.info('...Step %d: Eligible Universe down %d and currently stands at %d stocks',
                    stepNo, n-len(estuIdx), len(estuIdx))
            n = len(estuIdx)

        # Remove assets classed as foreign via home market classification
        dr_indices = []
        if len(data.foreign) > 0:
            stepNo+=1
            logging.info('Step %d: Excluding foreign listings by home country mapping', stepNo)
            dr_indices = [data.assetIdxMap[sid] for sid in data.foreign]
            (estuIdx, nonest) = buildEstu.exclude_specific_assets(dr_indices, baseEstu=estuIdx)
            if n != len(estuIdx):
                logging.info('...Step %d: Eligible Universe down %d and currently stands at %d stocks',
                        stepNo, n-len(estuIdx), len(estuIdx))
                n = len(estuIdx)

        # Remove assets classed as foreign via home market classification - part 2
        stepNo+=1
        # import ipdb;ipdb.set_trace()
        logging.info('Step %d: Excluding foreign listings by market classification', stepNo)
        (estuIdx2, nonest) = buildEstu.exclude_by_market_classification(modelDate, 'HomeCountry', 'REGIONS', HomeCountry_List, baseEstu=estuIdx)
        estuIdx = estuIdx2
        if n != len(estuIdx):
            logging.info('...Step %d: Eligible Universe down %d and currently stands at %d stocks',
                    stepNo, n-len(estuIdx), len(estuIdx))
            n = len(estuIdx)

        # # Weed out foreign issuers by ISIN country prefix
        # stepNo+=1
        # # import ipdb;ipdb.set_trace()
        # logging.info('Step %d: Excluding foreign listings by ISIN prefix', stepNo)
        # (estuIdx, nonest)  = buildEstu.exclude_by_isin_country(
        #         [r.mnemonic for r in self.rmg], modelDate, baseEstu=estuIdx)
        # if n != len(estuIdx):
        #     logging.info('...Step %d: Eligible Universe down %d and currently stands at %d stocks',
        #             stepNo, n-len(estuIdx), len(estuIdx))
        #     n = len(estuIdx)
        # Manually add H-shares and Red-Chips back into list of eligible assets
        stepNo+=1
        logging.info('Step %d: Adding back %d H-Shares and Redchips', stepNo, len(chinese))
        estuIdx = set(estuIdx).union(set(chinese))
        if n != len(estuIdx):
            logging.info('...Step %d: Eligible Universe up by %d and currently stands at %d stocks',
                    stepNo, len(estuIdx)-n, len(estuIdx))
            n = len(estuIdx)

        # Logic to selectively filter RTS/MICEX Russian stocks
        rmgID = [rmg.rmg_id for rmg in self.rmg if rmg.mnemonic=='RU']
        if len(rmgID) > 0:
            stepNo+=1
            logging.info('Step %d: Cleaning up Russian assets', stepNo)
            # Get lists of RTS and MICEX-quoted assets
            (rtsIdx, nonest) = buildEstu.exclude_by_market_type(
                    modelDate, data, includeFields='RTS', excludeFields=None,
                    baseEstu = estuIdx)
            (micIdx, nonest) = buildEstu.exclude_by_market_type(
                    modelDate, data, includeFields='MIC', excludeFields=None,
                    baseEstu = estuIdx)
            # Find companies with multiple listings which include lines
            # on both exchanges
            rtsDropList = []
            for (groupID, sidList) in data.subIssueGroups.items():
                if len(sidList) > 1:
                    subIdxList = [data.assetIdxMap[sid] for sid in sidList]
                    rtsOverlap = set(rtsIdx).intersection(subIdxList)
                    micOverLap = set(micIdx).intersection(subIdxList)
                    if len(micOverLap) > 0:
                        rtsDropList.extend(rtsOverlap)
            estuIdx = estuIdx.difference(rtsDropList)
            if n != len(estuIdx):
                logging.info('...Step %d: Eligible Universe down %d and currently stands at %d stocks',
                        stepNo, n-len(estuIdx), len(estuIdx))
                n = len(estuIdx)

        # Remove preferred stock except for selected markets
        prefMarketList = ['BR','CO']
        prefAssetBase = []
        mktsUsed = []
        for mkt in prefMarketList:
            # Find assets in our current estu that are in the relevant markets
            rmgID = [rmg.rmg_id for rmg in self.rmg if rmg.mnemonic==mkt]
            if len(rmgID) > 0:
                prefAssetBase.extend(data.rmgAssetMap[rmgID[0]])
                mktsUsed.append(mkt)
        baseEstuIdx = [data.assetIdxMap[sid] for sid in prefAssetBase]
        baseEstuIdx = list(set(baseEstuIdx).intersection(set(estuIdx)))
        if len(baseEstuIdx) > 0:
            # Find which of our subset of assets are preferred stock
            stepNo+=1
            logging.info('Step %d: Dropping preferred stocks NOT on markets: %s', stepNo, ','.join(mktsUsed))
            (prefIdx, nonest) = buildEstu.exclude_by_asset_type(
                    modelDate, data,
                    includeFields=['All-Pref'], excludeFields=None,
                    baseEstu=baseEstuIdx)
            # Get rid of preferred stock in general
            (estuIdx, nonest) = buildEstu.exclude_by_asset_type(
                    modelDate, data,
                    includeFields=None, excludeFields=['All-Pref'],
                    baseEstu=estuIdx)
            # Add back in allowed preferred stock
            estuIdx = list(set(estuIdx).union(prefIdx))
            if n != len(estuIdx):
                logging.info('...Step %d: Eligible Universe down %d and currently stands at %d stocks',
                        stepNo, n-len(estuIdx), len(estuIdx))
                n = len(estuIdx)

        # Get rid of some exchanges we don't want
        exchangeCodes = ['REG']
        if modelDate.year > 2009:
            exchangeCodes.extend(['XSQ','OFE','XLF']) # GB junk
        stepNo+=1
        logging.info('Step %d: Removing stocks on exchanges: %s', stepNo, ','.join(exchangeCodes))
        (estuIdx, nonest) = buildEstu.exclude_by_market_type(
                modelDate, data, includeFields=None, excludeFields=exchangeCodes,
                baseEstu = estuIdx)
        if n != len(estuIdx):
            logging.info('...Step %d: Eligible Universe down %d and currently stands at %d stocks',
                    stepNo, n-len(estuIdx), len(estuIdx))
            n = len(estuIdx)

        # Limit stocks to certain exchanges in select countries
        mktExchangeMap = {'CA': ['TSE'],
                          'JP': ['TKS-S1','TKS-S2','OSE-S1','OSE-S2','TKS-MSC','TKS-M','JAS']
                         }
        if modelDate.year > 2009:
            mktExchangeMap['US'] = ['NAS','NYS']
        for mkt in mktExchangeMap.keys():
            rmgID = [rmg.rmg_id for rmg in self.rmg if rmg.mnemonic==mkt]
            if len(rmgID) < 1:
                continue
            stepNo+=1
            logging.info('Step %d: Dropping %s stocks NOT on exchanges: %s', stepNo,
                    mkt, ','.join(mktExchangeMap[mkt]))

            # Pull out subset of assets on relevant market
            baseEstuIdx = [data.assetIdxMap[sid] for sid in data.rmgAssetMap[rmgID[0]]]
            baseEstuIdx = list(set(baseEstuIdx).intersection(set(estuIdx)))
            if len(baseEstuIdx) < 1:
                continue

            # Determine whether they are on permitted exchange
            (mktEstuIdx, nonest) = buildEstu.exclude_by_market_type(
                    modelDate, data, includeFields=mktExchangeMap[mkt], excludeFields=None,
                    baseEstu = baseEstuIdx)

            # Remove undesirables
            nonMktEstuIdx = set(baseEstuIdx).difference(set(mktEstuIdx))
            estuIdx = list(set(estuIdx).difference(nonMktEstuIdx))
            if n != len(estuIdx):
                logging.info('...Step %d: Eligible Universe down %d and currently stands at %d stocks',
                    stepNo, n-len(estuIdx), len(estuIdx))
                n = len(estuIdx)

        estu = [universe[idx] for idx in estuIdx]
        data.eligibleUniverseIdx = estuIdx
        logging.info('%d eligible assets out of %d total', len(estu), len(universe))
        self.log.info('generate_eligible_universe: end')
        return estu

    def generate_estimation_universe_old2(self, modelDate, data, modelDB, marketDB, excludeFactors=None):
        """Estimation universe selection criteria for AX-AU.
        """
        import pandas as pd
        import pickle
        # import ipdb;ipdb.set_trace()
        self.log.info('generate_estimation_universe for AU Model: begin')
        buildEstu = EstimationUniverse.ConstructEstimationUniverse(data.universe, self, modelDB, marketDB)

        # Set up various eligible and total universes
        universeIdx = list(range(len(buildEstu.assets)))
        originalEligibleUniverse = list(data.eligibleUniverse)
        originalEligibleUniverseIdx = [data.assetIdxMap[sid] for sid in originalEligibleUniverse]

        universe = pd.DataFrame(buildEstu.assets,columns=['SubID_Obj'])
        universeIdx = universe.index
        original_eligibleUniverse = pd.DataFrame(list(zip(data.eligibleUniverse,originalEligibleUniverseIdx)),columns=['SubID_Obj','originalEligibleUniverseIdx'])

        logging.info('ESTU currently stands at %d stocks based on original eligible universe', len(originalEligibleUniverse))

        # (1) Filtering thinly-traded assets over the entire universe
        logging.info('Looking for thinly-traded stocks')
        # (nonSparseIdx, sparse) = buildEstu.exclude_thinly_traded_assets(modelDate, data, baseEstu=universeIdx, minNonZero=0.75)
        # data.nonSparse = numpy.take(buildEstu.assets, nonSparseIdx, axis=0)
        (non_thin_idx, thin_idx,ret_quality_stat) = buildEstu.exclude_thinly_traded_assets(modelDate, data, baseEstu=universe.index.values, minNonZero=0.5)

        data.nonSparse = pd.DataFrame(universe,index=non_thin_idx)["SubID_Obj"].tolist()
        # Remove nursery market assets
        if len(data.nurseryUniverse) > 0:
            logging.info('Checking for assets from nursery markets')
            raise('ERROR: Ask for what is the nurseryUniverse for?')
        else:
            eligibleUniverseIdx = originalEligibleUniverseIdx

        # Exclude thinly traded assets
        estu_withoutThin_idx = list(set(eligibleUniverseIdx).intersection(set(non_thin_idx)))
        logging.info('ESTU currently stands at %d stocks after Filtering thinly-traded assets', len(estu_withoutThin_idx))
        # (2) Filtering tiny-cap assets by market, country and industry
        # (2a) Weed out tiny-cap assets by market
        lowerBound = 5
        logging.info('Filtering by top %d%% mcap on entire market', 100-lowerBound)
        (large_byMkt_idx, nonest1,factor_mktCap1) = buildEstu.exclude_by_cap_ranking(data, modelDate, baseEstu=estu_withoutThin_idx,
                                                                     lower_pctile=lowerBound, method='percentage',weight='mcap')  #rootCap
                #weight='rootCap')
        subissue_id_objs = data.universe
        mktcap = modelDB.getAverageMarketCaps([modelDate], subissue_id_objs, currencyID = None)
        mktcap = pd.DataFrame(mktcap)
        print(mktcap.ix[large_byMkt_idx].min(),len(large_byMkt_idx))
        # (2b) Weed out tiny-cap assets by country
        # lowerBound = 5
        # logging.info('Filtering by top %d%% mcap on country', 100-lowerBound)
        # (large_byCntry_idx, nonest2) = buildEstu.exclude_by_cap_ranking(data, modelDate, baseEstu=estu_withoutThin_idx,
        #                                                                byFactorType=ExposureMatrix.CountryFactor,
        #                                                                lower_pctile=lowerBound, method='percentage',
        #                                                                excludeFactors=excludeFactors)
        #                                                                #weight='rootCap')
        # (2c) Perform similar check by industry
        lowerBound = 10
        logging.info('Filtering by top %d%% mcap on industry', 100-lowerBound)
        (large_byIndtry_idx, nonest3,factor_mktCap2) = buildEstu.exclude_by_cap_ranking(data, modelDate, baseEstu=estu_withoutThin_idx,byFactorType=ExposureMatrix.IndustryFactor,lower_pctile=lowerBound, method='percentage',excludeFactors=excludeFactors,weight='mcap')  #rootCap
        print(mktcap.iloc[large_byIndtry_idx].min(),len(large_byIndtry_idx))

        # estu_mktCap_idx = set(large_byMkt_idx).union(large_byCntry_idx).union(large_byIndtry_idx)
        estu_mktCap_idx = set(large_byMkt_idx).intersection(large_byIndtry_idx)
        estu_mktCap_idx = list(estu_mktCap_idx)
        tinyCap_idx = list(set(estu_withoutThin_idx).difference(estu_mktCap_idx))
        logging.info('ESTU currently stands at %d stocks after Filtering by Market Cap.', len(estu_mktCap_idx))
        # candid_univ_idx = eligibleUniverseIdx
        # candid_univ_idx = estu_withoutThin_idx
        estu_withoutThin_idx2 = list(set(estu_withoutThin_idx).intersection(set(large_byMkt_idx).union(large_byIndtry_idx)))
        candid_univ_idx = estu_withoutThin_idx2
        # Inflate any thin countries or industries
        minFactorWidth=self.returnCalculator.allParameters[0].getThinFactorInformation().dummyThreshold
        logging.info('Inflating any thin factors')
        (estu_inflated_idx, nonest,herf_num_list) = buildEstu.pump_up_factors(data, modelDate,currentEstu=estu_mktCap_idx,
                                                                              baseEstu=candid_univ_idx,byFactorType=[ExposureMatrix.CountryFactor, ExposureMatrix.IndustryFactor],minFactorWidth=minFactorWidth,
                                                                              cutOff = self.inflation_cutoff, excludeFactors=excludeFactors)

        logging.info('ESTU currently stands at %d stocks', len(estu_inflated_idx))

        # herf_num_list.to_csv('herf_num_list.csv')
        # Apply grandfathering rules
        # import ipdb;ipdb.set_trace()
        logging.info('Incorporating grandfathering')
        (estu_final_Idx, ESTUQualify, nonest) = buildEstu.grandfather(modelDate, estu_inflated_idx, baseEstu=eligibleUniverseIdx,daysBack=63,
                                                                estuInstance=self.estuMap['main'])

        totalcap = ma.sum(ma.take(data.marketCaps, estu_final_Idx, axis=0), axis=0) / 1e9
        self.log.info('Final estu contains %d assets, %.2f bn (%s)',len(estu_final_Idx), totalcap, self.numeraire.currency_code)
        self.log.info('Grandfather Rule Brings in %d assets',len(estu_final_Idx)-len(estu_inflated_idx))
        self.log.info('Final estu contains %d assets in thin-traded assets.',sum(pd.Series(estu_final_Idx).isin(thin_idx)))
        self.log.info('Final estu contains %d assets in tiny cap assets.',sum(pd.Series(estu_final_Idx).isin(tinyCap_idx)))
        self.log.debug('generate_estimation_universe: end')

        # If we have a family of estimation universes, populate the main estu accordingly
        self.estuMap['main'].assets = [buildEstu.assets[idx] for idx in estu_final_Idx]
        self.estuMap['main'].qualify = [buildEstu.assets[idx] for idx in ESTUQualify]
        # import ipdb;ipdb.set_trace()
        print('final_est:', len(estu_final_Idx))

        estu_output = {'ret_quality_stat':ret_quality_stat,'factor_mktCap1':factor_mktCap1,'factor_mktCap2':factor_mktCap2,'herf_num_list':herf_num_list}
        with open('result_estu/'+ str(modelDate)+'.pickle', 'wb') as handle:
            pickle.dump(estu_output, handle)
        return estu_final_Idx

    def generate_estimation_universe(self, modelDate, data, modelDB, marketDB, excludeFactors=None):
        """Estimation universe selection criteria for AX-AU.
        """
        import pandas as pd
        # import ipdb;ipdb.set_trace()
        self.log.info('generate_estimation_universe for AU Model: begin')
        buildEstu = EstimationUniverse.ConstructEstimationUniverse(data.universe, self, modelDB, marketDB)

        # Set up various eligible and total universes
        universeIdx = list(range(len(buildEstu.assets)))
        originalEligibleUniverse = list(data.eligibleUniverse)
        originalEligibleUniverseIdx = [data.assetIdxMap[sid] for sid in originalEligibleUniverse]

        universe = pd.DataFrame(buildEstu.assets,columns=['SubID_Obj'])
        universeIdx = universe.index
        original_eligibleUniverse = pd.DataFrame(list(zip(data.eligibleUniverse,originalEligibleUniverseIdx)),columns=['SubID_Obj','originalEligibleUniverseIdx'])

        logging.info('ESTU currently stands at %d stocks based on original eligible universe', len(originalEligibleUniverse))

        # (1) Filtering thinly-traded assets over the entire universe
        logging.info('Looking for thinly-traded stocks')
        # (nonSparseIdx, sparse) = buildEstu.exclude_thinly_traded_assets(modelDate, data, baseEstu=universeIdx, minNonZero=0.75)
        # data.nonSparse = numpy.take(buildEstu.assets, nonSparseIdx, axis=0)
        (non_thin_idx, thin_idx) = buildEstu.exclude_thinly_traded_assets(modelDate, data, baseEstu=universe.index.values, minNonZero=0.3)

        data.nonSparse = pd.DataFrame(universe,index=non_thin_idx)["SubID_Obj"].tolist()
        # Remove nursery market assets
        if len(data.nurseryUniverse) > 0:
            logging.info('Checking for assets from nursery markets')
            raise('ERROR: Ask for what is the nurseryUniverse for?')
        else:
            eligibleUniverseIdx = originalEligibleUniverseIdx

        # Exclude thinly traded assets
        estu_withoutThin_idx = list(set(eligibleUniverseIdx).intersection(set(non_thin_idx)))
        logging.info('ESTU currently stands at %d stocks after Filtering thinly-traded assets', len(estu_withoutThin_idx))
        # (2) Filtering tiny-cap assets by market, country and industry
        # (2a) Weed out tiny-cap assets by market
        lowerBound = 5
        logging.info('Filtering by top %d%% mcap on entire market', 100-lowerBound)
        (large_byMkt_idx, nonest1) = buildEstu.exclude_by_cap_ranking(data, modelDate, baseEstu=estu_withoutThin_idx,
                                                                     lower_pctile=lowerBound, method='percentage')
                #weight='rootCap')
        subissue_id_objs = data.universe
        mktcap = modelDB.getAverageMarketCaps([modelDate], subissue_id_objs, currencyID = None)
        mktcap = pd.DataFrame(mktcap)
        print(mktcap.ix[large_byMkt_idx].min(),len(large_byMkt_idx))
        # (2b) Weed out tiny-cap assets by country
        # lowerBound = 5
        # logging.info('Filtering by top %d%% mcap on country', 100-lowerBound)
        # (large_byCntry_idx, nonest2) = buildEstu.exclude_by_cap_ranking(data, modelDate, baseEstu=estu_withoutThin_idx,
        #                                                                byFactorType=ExposureMatrix.CountryFactor,
        #                                                                lower_pctile=lowerBound, method='percentage',
        #                                                                excludeFactors=excludeFactors)
        #                                                                #weight='rootCap')
        # (2c) Perform similar check by industry
        lowerBound = 10
        logging.info('Filtering by top %d%% mcap on industry', 100-lowerBound)
        (large_byIndtry_idx, nonest3) = buildEstu.exclude_by_cap_ranking(data, modelDate, baseEstu=estu_withoutThin_idx,byFactorType=ExposureMatrix.IndustryFactor,lower_pctile=lowerBound, method='percentage',excludeFactors=excludeFactors)
                                                                        #weight='rootCap')
        print(mktcap.iloc[large_byIndtry_idx].min(),len(large_byIndtry_idx))

        # estu_mktCap_idx = set(large_byMkt_idx).union(large_byCntry_idx).union(large_byIndtry_idx)
        estu_mktCap_idx = set(large_byMkt_idx).intersection(large_byIndtry_idx)
        estu_mktCap_idx = list(estu_mktCap_idx)
        tinyCap_idx = list(set(estu_withoutThin_idx).difference(estu_mktCap_idx))
        logging.info('ESTU currently stands at %d stocks after Filtering by Market Cap.', len(estu_mktCap_idx))
        # candid_univ_idx = eligibleUniverseIdx
        # candid_univ_idx = estu_withoutThin_idx
        estu_withoutThin_idx2 = list(set(estu_withoutThin_idx).intersection(set(large_byMkt_idx).union(large_byIndtry_idx)))
        candid_univ_idx = estu_withoutThin_idx2
        # Inflate any thin countries or industries
        minFactorWidth=self.returnCalculator.allParameters[0].getThinFactorInformation().dummyThreshold
        logging.info('Inflating any thin factors')
        (estu_inflated_idx, nonest,herf_num_list) = buildEstu.pump_up_factors(data, modelDate,currentEstu=estu_mktCap_idx,
                                                                              baseEstu=candid_univ_idx,byFactorType=[ExposureMatrix.CountryFactor, ExposureMatrix.IndustryFactor],minFactorWidth=minFactorWidth,
                                                                              cutOff = self.inflation_cutoff, excludeFactors=excludeFactors)

        logging.info('ESTU currently stands at %d stocks', len(estu_inflated_idx))
        herf_num_list = pd.DataFrame(herf_num_list)
        # herf_num_list.to_csv('herf_num_list.csv')
        # Apply grandfathering rules
        # import ipdb;ipdb.set_trace()
        logging.info('Incorporating grandfathering')
        (estu_final_Idx, ESTUQualify, nonest) = buildEstu.grandfather(modelDate, estu_inflated_idx, baseEstu=eligibleUniverseIdx,daysBack=61,
                                                                estuInstance=self.estuMap['main'])

        totalcap = ma.sum(ma.take(data.marketCaps, estu_final_Idx, axis=0), axis=0) / 1e9
        self.log.info('Final estu contains %d assets, %.2f bn (%s)',len(estu_final_Idx), totalcap, self.numeraire.currency_code)
        self.log.info('Grandfather Rule Brings in %d assets',len(estu_final_Idx)-len(estu_inflated_idx))
        self.log.info('Final estu contains %d assets in thin-traded assets.',sum(pd.Series(estu_final_Idx).isin(thin_idx)))
        self.log.info('Final estu contains %d assets in tiny cap assets.',sum(pd.Series(estu_final_Idx).isin(tinyCap_idx)))
        self.log.debug('generate_estimation_universe: end')

        # If we have a family of estimation universes, populate the main estu accordingly
        self.estuMap['main'].assets = [buildEstu.assets[idx] for idx in estu_final_Idx]
        self.estuMap['main'].qualify = [buildEstu.assets[idx] for idx in ESTUQualify]
        # import ipdb;ipdb.set_trace()
        print('final_est:', len(estu_final_Idx))
        return estu_final_Idx

    def generate_estimation_universe_old(self, modelDate, data, modelDB, marketDB, excludeFactors=None):
        """Estimation universe selection criteria for AX-AU.
        """
        self.log.debug('generate_estimation_universe: begin')
        # import ipdb;ipdb.set_trace()
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

        # Report on thinly-traded assets over the entire universe
        (nonSparse, sparse) = buildEstu.exclude_thinly_traded_assets(
                                modelDate, data, baseEstu=universeIdx)
        nonSparse = [buildEstu.assets[idx] for idx in nonSparse]
        sparse = [buildEstu.assets[idx] for idx in sparse]

        # Exclude thinly traded assets if required
        estuIdx = [data.assetIdxMap[sid] for sid in eligibleUniverse if sid in nonSparse]
        buildEstu.report_estu_content(data.marketCaps, estuIdx, stepName='Exclude Sparse')

        # Rank stuff by market cap and total volume over past year
        (estuIdx, nonest) = buildEstu.filter_by_cap_and_volume(
                data, modelDate,
                baseEstu=estuIdx,hiCapQuota=200, loCapQuota=100,bufferFactor=1.2)

        buildEstu.report_estu_content(data.marketCaps, estuIdx, stepName='Size/Volume Filter')

        # Inflate thin industry factors if possible
        (estuIdx, nonest) = buildEstu.pump_up_factors(
            data, modelDate, currentEstu=estuIdx, baseEstu=eligibleUniverseIdx,
            minFactorWidth=self.returnCalculator.allParameters[0].getThinFactorInformation().dummyThreshold,cutOff = 0.05)
        buildEstu.report_estu_content(data.marketCaps, estuIdx, stepName='Factor Inflate')

        # Apply grandfathering rules
        (estuIdx, ESTUQualify, nonest) = buildEstu.grandfather(
                 modelDate, estuIdx, baseEstu=eligibleUniverseIdx, estuInstance=self.estuMap['main'])
        buildEstu.report_estu_content(data.marketCaps, estuIdx, stepName='Grandfather')
        # import ipdb;ipdb.set_trace()

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
        # import ipdb;ipdb.set_trace()
        # Mid/Small-cap factors
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

class AU2Yilin_Prod_K5(RiskModels_V3.AUAxioma2009MH):
    rm_id = -20
    revision = 14
    rms_id = -128
    k  = 5.0
    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('RiskModels.AU2Yilin_Prod_K5')
        MFM.SingleCountryFundamentalModel.__init__(self, ['SEDOL'], modelDB, marketDB)
        self.setCalculators(modelDB)
        modelDB.createCurrencyCache(marketDB)
        self.indexSelector = MarketIndex.MarketIndexSelector(modelDB, marketDB)

    def setCalculators(self, modelDB, overrider = False):
        # Set up regression parameters
        self.returnCalculator = riskmodels.defaultRegressionParameters(self, modelDB, dummyThreshold=6.0, k_rlm = self.k,overrider=overrider)
        # Set up risk parameters
        riskmodels.defaultFundamentalCovarianceParameters(self, nwLag=3, dva='slowStep', overrider=overrider)

class AU2Yilin_d18(EquityModel.FundamentalModel):
    """AU2 base model.
    """
    # Model Parameters:
    rm_id,revision,rms_id = [-20,32,-146]
    k  = 5.0
    elig_parameters = {'HomeCountry_List': ['AU','NZ'],
                       'use_isin_country_Flag': False}
    estu_parameters = {
                       'minNonZero':0.1,
                       'minNonMissing':0.5,
                       'maskZeroWithNoADV_Flag': True,
                       'returnLegacy_Flag': False,
                       'CapByNumber_Flag':True,
                       'CapByNumber_hiCapQuota':200,
                       'CapByNumber_lowCapQuota':100,
                       'market_lower_pctile': np.nan,
                       'country_lower_pctile': np.nan,
                       'industry_lower_pctile': np.nan,
                       'dummyThreshold': 6,
                       'inflation_cutoff':0.03
                        }
    dummyThreshold = 6     # dummyThreshold = 6
    inflation_cutoff = 0.03
    minNonZero = 0.1

    # Model Parameters, Workflow flag
    regionalDescriptorStructure = False
    twoRegressionStructure = False
    multiCountry = False
    # parameters used for MFM class - to remove
    # standardizationStats = True
    # globalDescriptorModel = True
    # DLCEnabled = True
    # SCM = True

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
            'EM Sensitivity': ['Market_Sensitivity_EM_W','Market_Sensitivity_CN_W'],
            'Profitability': ['Return_on_Equity_Annual','Return_on_Assets_Annual', 'CashFlow_to_Assets_Annual',
                              'CashFlow_to_Income_Annual','Sales_to_Assets_Annual', 'Gross_Margin_Annual'],
            }
    DescriptorWeights = {'Value': [0.5, 0.375,0.125],
                        'EM Sensitivity': [0.5,0.5]}
    smallCapMap = {}
    noProxyList = ['Dividend Yield']
    fillMissingList = ['Value', 'Leverage', 'Growth', 'Profitability']
    fillWithZeroList = ['Dividend Yield']
    shrinkList = {'Liquidity': 60,
                  'Market Sensitivity': 250,
                  'Volatility': 125,
                  'Medium-Term Momentum': 250}
    orthogList = {'Volatility': [['Market Sensitivity'], True, 1.0],'EM Sensitivity': [['Exchange Rate Sensitivity'], True, 1.0]}

    # Setting up market intercept if relevant
    interceptFactor = 'Market Intercept'
    intercept = ModelFactor(interceptFactor, interceptFactor)
    industryClassification = Classification.GICSCustomAU2(datetime.date(2016,9,1))
    estuAssetTypes = ['REIT', 'Com', 'StapSec']

    def __init__(self, modelDB, marketDB, expTreat=None):
        self.log = logging.getLogger('RiskModels.AU2Yilin')
        # Set up relevant styles to be created/used
        # ModelParameters.defaultExposureParameters(self, self.styleList)
        ModelParameters2017.defaultExposureParameters(self, self.styleList)
        # if expTreat == 'addNew':
        #     self.styles = [s for s in self.totalStyles if s.name in self.addList]
        #     self.variableStyles = True
        # else:
        #     self.styles = [s for s in self.totalStyles if s.name in self.styleList]
        self.styles = [s for s in self.totalStyles if s.name in self.styleList]
        EquityModel.FundamentalModel.__init__(self, ['SEDOL'], modelDB, marketDB)
        # MFM.FundamentalModel.__init__(self, ['CUSIP'], modelDB, marketDB) # it doesn't work!

        # Set up regression parameters

        self.returnCalculator = ModelParameters2017.defaultRegressionParameters(self, modelDB,dummyType='Sectors', dummyThreshold=self.dummyThreshold,marketRegression=False,kappa = self.k)

        # This controls the FMP regression
        self.fmpCalculator = ModelParameters2017.defaultRegressionParameters(self, modelDB,dummyType='Sectors',marketRegression=False,dummyThreshold=self.dummyThreshold,kappa=None)

        # Set up risk parameters
        ModelParameters2017.defaultFundamentalCovarianceParameters(self, nwLag=2,varDVAOnly=False, unboundedDVA=False)

        # Set up standardization parameters
        gloScope = Standardization.GlobalRelativeScope([f.name for f in self.styles])
        # self.exposureStandardization = Standardization.BucketizedStandardization([gloScope], fancyMAD=self.fancyMAD)
        self.exposureStandardization = Standardization.BucketizedStandardization([gloScope], fillWithZeroList=self.fillWithZeroList)

        self.indexSelector = MarketIndex.MarketIndexSelector(modelDB, marketDB)
        modelDB.createCurrencyCache(marketDB)

        # Set up descriptor standardization parameters
        descriptors = sorted(list(set([item for sublist in self.DescriptorMap.values() for item in sublist])))
        exceptionNames = [self.DescriptorMap[sf] for sf in self.noProxyList]
        exceptionNames = list(itertools.chain.from_iterable(exceptionNames))
        # from IPython import embed;  embed(header='')
        # if not self.SCM:
        #     self.descriptorStandardization = Standardization.BucketizedStandardization(
        #             [Standardization.RegionRelativeScope(modelDB, descriptors)],
        #             mad_bound=15.0, fancyMAD=self.fancyMAD, exceptionNames=exceptionNames)
        # else:
        self.descriptorStandardization = Standardization.BucketizedStandardization(
                [Standardization.GlobalRelativeScope(descriptors)],
                mad_bound=15.0, fancyMAD=self.fancyMAD, exceptionNames=exceptionNames)

    def generate_eligible_universe(self, modelDate, data, modelDB, marketDB, capBounds=None,assetTypes=['All-Com', 'REIT'],excludeTypes=['ComWI', 'UnCls', 'StatETF', 'LLC', 'LP', 'NonEqETF', 'ComETF']):
        """Creates subset of eligible assets for consideration
        in estimation universes
        """
        # import ipdb;ipdb.set_trace()
        self.log.info('generate_eligible_universe: begin')
        buildEstu = EstimationUniverse.ConstructEstimationUniverse(
                data.universe, self, modelDB, marketDB)
        universe = buildEstu.assets
        n = len(universe)
        stepNo = 0
        if assetTypes is None:
            assetTypes = []
        if excludeTypes is None:
            excludeTypes = []
        logging.info('Eligible Universe currently stands at %d stocks', n)

        # Remove assets from the exclusion table
        stepNo+=1
        logging.info('Step %d: Applying exclusion table', stepNo)
        (estuIdx, nonest) = buildEstu.apply_exclusion_list(modelDate)
        if n != len(estuIdx):
            logging.info('...Step %d: Eligible Universe down %d and currently stands at %d stocks',
                    stepNo, n-len(estuIdx), len(estuIdx))
            n = len(estuIdx)

        # Remove cloned assets
        if len(data.hardCloneMap) > 0:
            stepNo+=1
            logging.info('Step %d: Removing cloned assets', stepNo)
            cloneIdx = [data.assetIdxMap[sid] for sid in data.hardCloneMap.keys()]
            (estuIdx, nonest) = buildEstu.exclude_specific_assets(cloneIdx, baseEstu=estuIdx)
            if n != len(estuIdx):
                logging.info('...Step %d: Eligible Universe down %d and currently stands at %d stocks',
                        stepNo, n-len(estuIdx), len(estuIdx))
                n = len(estuIdx)

        # Include by asset type field
        stepNo+=1
        logging.info('Step %d: Include by asset types %s', stepNo, ','.join(assetTypes))
        (estuIdx, nonest) = buildEstu.exclude_by_asset_type(
                modelDate, data, includeFields=assetTypes, excludeFields=None,
                baseEstu = estuIdx)
        if n != len(estuIdx):
            logging.info('...Step %d: Eligible Universe down %d and currently stands at %d stocks',
                    stepNo, n-len(estuIdx), len(estuIdx))
            n = len(estuIdx)

        # Pull out Chinese H shares and redchips to save for later
        stepNo+=1
        assetTypes = ['HShares', 'RCPlus']
        logging.info('Step %d: Getting list of Chinese assets: %s', stepNo, ','.join(assetTypes))
        (chinese, nonest) = buildEstu.exclude_by_asset_type(
                modelDate, data, includeFields=assetTypes, excludeFields=None,
                baseEstu = estuIdx)
        if len(chinese) > 0:
            logging.info('...Step %d: Found %d Chinese H-shares and Redchips', stepNo, len(chinese))

        # Exclude by asset type field
        stepNo+=1
        logging.info('Step %d: Exclude default asset types %s', stepNo, ','.join(excludeTypes))
        (estuIdx, nonest) = buildEstu.exclude_by_asset_type(
                modelDate, data, includeFields=None, excludeFields=excludeTypes,
                baseEstu = estuIdx)
        if n != len(estuIdx):
            logging.info('...Step %d: Eligible Universe down %d and currently stands at %d stocks',
                    stepNo, n-len(estuIdx), len(estuIdx))
            n = len(estuIdx)

        # Remove various types of DRs and foreign listings
        stepNo+=1
        exTypes = ['NVDR', 'GlobalDR', 'DR', 'TDR', 'AmerDR', 'FStock', 'CDI']
        logging.info('Step %d: Exclude DR asset types %s', stepNo, ','.join(exTypes))
        (estuIdx, nonest) = buildEstu.exclude_by_asset_type(
                modelDate, data, includeFields=None, excludeFields=exTypes,
                baseEstu = estuIdx)
        if n != len(estuIdx):
            logging.info('...Step %d: Eligible Universe down %d and currently stands at %d stocks',
                    stepNo, n-len(estuIdx), len(estuIdx))
            n = len(estuIdx)

        # Remove trusts, funds and other odds and ends
        stepNo+=1
        exTypes = ['InvT', 'UnitT', 'CEFund', 'Misc']
        logging.info('Step %d: Exclude fund/trust asset types %s', stepNo, ','.join(exTypes))
        (estuIdx, nonest) = buildEstu.exclude_by_asset_type(
                modelDate, data, includeFields=None, excludeFields=exTypes,
                baseEstu = estuIdx)
        if n != len(estuIdx):
            logging.info('...Step %d: Eligible Universe down %d and currently stands at %d stocks',
                    stepNo, n-len(estuIdx), len(estuIdx))
            n = len(estuIdx)

        # Remove Chinese A and B shares
        stepNo+=1
        exTypes = ['AShares', 'BShares']
        logging.info('Step %d: Exclude Chinese asset types %s', stepNo, ','.join(exTypes))
        (estuIdx, nonest) = buildEstu.exclude_by_asset_type(
                modelDate, data, includeFields=None, excludeFields=exTypes,
                baseEstu = estuIdx)
        if n != len(estuIdx):
            logging.info('...Step %d: Eligible Universe down %d and currently stands at %d stocks',
                    stepNo, n-len(estuIdx), len(estuIdx))
            n = len(estuIdx)

        # Remove assets classed as foreign via home market classification
        dr_indices = []
        if len(data.foreign) > 0:
            stepNo+=1
            logging.info('Step %d: Excluding foreign listings by home country mapping', stepNo)
            dr_indices = [data.assetIdxMap[sid] for sid in data.foreign]
            (estuIdx, nonest) = buildEstu.exclude_specific_assets(dr_indices, baseEstu=estuIdx)
            if n != len(estuIdx):
                logging.info('...Step %d: Eligible Universe down %d and currently stands at %d stocks',
                        stepNo, n-len(estuIdx), len(estuIdx))
                n = len(estuIdx)

        # Remove assets classed as foreign via home market classification - part 2
        stepNo+=1
        # import ipdb;ipdb.set_trace()
        logging.info('Step %d: Excluding foreign listings by market classification', stepNo)
        (estuIdx2, nonest) = buildEstu.exclude_by_market_classification(modelDate, 'HomeCountry', 'REGIONS', self.elig_parameters['HomeCountry_List'], baseEstu=estuIdx)
        estuIdx = estuIdx2
        if n != len(estuIdx):
            logging.info('...Step %d: Eligible Universe down %d and currently stands at %d stocks',
                    stepNo, n-len(estuIdx), len(estuIdx))
            n = len(estuIdx)

        # # Weed out foreign issuers by ISIN country prefix
        if self.elig_parameters['use_isin_country_Flag']:
            stepNo+=1
            # import ipdb;ipdb.set_trace()
            logging.info('Step %d: Excluding foreign listings by ISIN prefix', stepNo)
            (estuIdx, nonest)  = buildEstu.exclude_by_isin_country(
                    [r.mnemonic for r in self.rmg], modelDate, baseEstu=estuIdx)
            if n != len(estuIdx):
                logging.info('...Step %d: Eligible Universe down %d and currently stands at %d stocks',
                        stepNo, n-len(estuIdx), len(estuIdx))
                n = len(estuIdx)
        # Manually add H-shares and Red-Chips back into list of eligible assets
        stepNo+=1
        logging.info('Step %d: Adding back %d H-Shares and Redchips', stepNo, len(chinese))
        estuIdx = set(estuIdx).union(set(chinese))
        if n != len(estuIdx):
            logging.info('...Step %d: Eligible Universe up by %d and currently stands at %d stocks',
                    stepNo, len(estuIdx)-n, len(estuIdx))
            n = len(estuIdx)

        # Logic to selectively filter RTS/MICEX Russian stocks
        rmgID = [rmg.rmg_id for rmg in self.rmg if rmg.mnemonic=='RU']
        if len(rmgID) > 0:
            stepNo+=1
            logging.info('Step %d: Cleaning up Russian assets', stepNo)
            # Get lists of RTS and MICEX-quoted assets
            (rtsIdx, nonest) = buildEstu.exclude_by_market_type(
                    modelDate, data, includeFields='RTS', excludeFields=None,
                    baseEstu = estuIdx)
            (micIdx, nonest) = buildEstu.exclude_by_market_type(
                    modelDate, data, includeFields='MIC', excludeFields=None,
                    baseEstu = estuIdx)
            # Find companies with multiple listings which include lines
            # on both exchanges
            rtsDropList = []
            for (groupID, sidList) in data.subIssueGroups.items():
                if len(sidList) > 1:
                    subIdxList = [data.assetIdxMap[sid] for sid in sidList]
                    rtsOverlap = set(rtsIdx).intersection(subIdxList)
                    micOverLap = set(micIdx).intersection(subIdxList)
                    if len(micOverLap) > 0:
                        rtsDropList.extend(rtsOverlap)
            estuIdx = estuIdx.difference(rtsDropList)
            if n != len(estuIdx):
                logging.info('...Step %d: Eligible Universe down %d and currently stands at %d stocks',
                        stepNo, n-len(estuIdx), len(estuIdx))
                n = len(estuIdx)

        # Remove preferred stock except for selected markets
        prefMarketList = ['BR','CO']
        prefAssetBase = []
        mktsUsed = []
        for mkt in prefMarketList:
            # Find assets in our current estu that are in the relevant markets
            rmgID = [rmg.rmg_id for rmg in self.rmg if rmg.mnemonic==mkt]
            if len(rmgID) > 0:
                prefAssetBase.extend(data.rmgAssetMap[rmgID[0]])
                mktsUsed.append(mkt)
        baseEstuIdx = [data.assetIdxMap[sid] for sid in prefAssetBase]
        baseEstuIdx = list(set(baseEstuIdx).intersection(set(estuIdx)))
        if len(baseEstuIdx) > 0:
            # Find which of our subset of assets are preferred stock
            stepNo+=1
            logging.info('Step %d: Dropping preferred stocks NOT on markets: %s', stepNo, ','.join(mktsUsed))
            (prefIdx, nonest) = buildEstu.exclude_by_asset_type(
                    modelDate, data,
                    includeFields=['All-Pref'], excludeFields=None,
                    baseEstu=baseEstuIdx)
            # Get rid of preferred stock in general
            (estuIdx, nonest) = buildEstu.exclude_by_asset_type(
                    modelDate, data,
                    includeFields=None, excludeFields=['All-Pref'],
                    baseEstu=estuIdx)
            # Add back in allowed preferred stock
            estuIdx = list(set(estuIdx).union(prefIdx))
            if n != len(estuIdx):
                logging.info('...Step %d: Eligible Universe down %d and currently stands at %d stocks',
                        stepNo, n-len(estuIdx), len(estuIdx))
                n = len(estuIdx)

        # Get rid of some exchanges we don't want
        exchangeCodes = ['REG']
        if modelDate.year > 2009:
            exchangeCodes.extend(['XSQ','OFE','XLF']) # GB junk
        stepNo+=1
        logging.info('Step %d: Removing stocks on exchanges: %s', stepNo, ','.join(exchangeCodes))
        (estuIdx, nonest) = buildEstu.exclude_by_market_type(
                modelDate, data, includeFields=None, excludeFields=exchangeCodes,
                baseEstu = estuIdx)
        if n != len(estuIdx):
            logging.info('...Step %d: Eligible Universe down %d and currently stands at %d stocks',
                    stepNo, n-len(estuIdx), len(estuIdx))
            n = len(estuIdx)

        # Limit stocks to certain exchanges in select countries
        mktExchangeMap = {'CA': ['TSE'],
                          'JP': ['TKS-S1','TKS-S2','OSE-S1','OSE-S2','TKS-MSC','TKS-M','JAS']
                         }
        if modelDate.year > 2009:
            mktExchangeMap['US'] = ['NAS','NYS']
        for mkt in mktExchangeMap.keys():
            rmgID = [rmg.rmg_id for rmg in self.rmg if rmg.mnemonic==mkt]
            if len(rmgID) < 1:
                continue
            stepNo+=1
            logging.info('Step %d: Dropping %s stocks NOT on exchanges: %s', stepNo,
                    mkt, ','.join(mktExchangeMap[mkt]))

            # Pull out subset of assets on relevant market
            baseEstuIdx = [data.assetIdxMap[sid] for sid in data.rmgAssetMap[rmgID[0]]]
            baseEstuIdx = list(set(baseEstuIdx).intersection(set(estuIdx)))
            if len(baseEstuIdx) < 1:
                continue

            # Determine whether they are on permitted exchange
            (mktEstuIdx, nonest) = buildEstu.exclude_by_market_type(
                    modelDate, data, includeFields=mktExchangeMap[mkt], excludeFields=None,
                    baseEstu = baseEstuIdx)

            # Remove undesirables
            nonMktEstuIdx = set(baseEstuIdx).difference(set(mktEstuIdx))
            estuIdx = list(set(estuIdx).difference(nonMktEstuIdx))
            if n != len(estuIdx):
                logging.info('...Step %d: Eligible Universe down %d and currently stands at %d stocks',
                    stepNo, n-len(estuIdx), len(estuIdx))
                n = len(estuIdx)

        estu = [universe[idx] for idx in estuIdx]
        data.eligibleUniverseIdx = estuIdx
        logging.info('%d eligible assets out of %d total', len(estu), len(universe))
        self.log.info('generate_eligible_universe: end')
        return estu

    def generate_estimation_universe_old(self, modelDate, data, modelDB, marketDB, excludeFactors=None):
        """Estimation universe selection criteria for AX-AU.
        """
        import pandas as pd
        # import ipdb;ipdb.set_trace()
        self.log.info('generate_estimation_universe for AU Model: begin')
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

        # (1) Filtering thinly-traded assets over the entire universe
        logging.info('Looking for thinly-traded stocks')
        # (nonSparseIdx, sparse) = buildEstu.exclude_thinly_traded_assets(modelDate, data, baseEstu=universeIdx, minNonZero=0.75)
        # data.nonSparse = numpy.take(buildEstu.assets, nonSparseIdx, axis=0)
        (non_thin_idx, thin_idx) = buildEstu.exclude_thinly_traded_assets(modelDate, data, baseEstu=universe.index.values, minNonZero=self.minNonZero)

        data.nonSparse = pd.DataFrame(universe,index=non_thin_idx)["SubID_Obj"].tolist()
        # Remove nursery market assets
        if len(data.nurseryUniverse) > 0:
            logging.info('Checking for assets from nursery markets')
            raise('ERROR: Ask for what is the nurseryUniverse for?')
        else:
            eligibleUniverseIdx = originalEligibleUniverseIdx

        # Exclude thinly traded assets
        estu_withoutThin_idx = list(set(eligibleUniverseIdx).intersection(set(non_thin_idx)))
        logging.info('ESTU currently stands at %d stocks after Filtering thinly-traded assets', len(estu_withoutThin_idx))
        # Rank stuff by market cap and total volume over past year
        # import ipdb;ipdb.set_trace()
        (estu_Base_idx, nonest) = buildEstu.filter_by_cap_and_volume(data, modelDate, baseEstu=estu_withoutThin_idx, hiCapQuota=200, loCapQuota=100, bufferFactor=1.2)
        # (2) Filtering tiny-cap assets by market, country and industry
        # (2a) Weed out tiny-cap assets by market
        # lowerBound = 5
        # logging.info('Filtering by top %d%% mcap on entire market', 100-lowerBound)
        # (large_byMkt_idx, nonest1) = buildEstu.exclude_by_cap_ranking(data, modelDate, baseEstu=estu_withoutThin_idx,
        #                                                              lower_pctile=lowerBound, method='percentage')
        #         #weight='rootCap')
        # subissue_id_objs = data.universe
        # mktcap = modelDB.getAverageMarketCaps([modelDate], subissue_id_objs, currencyID = None)
        # mktcap = pd.DataFrame(mktcap)
        # print mktcap.ix[large_byMkt_idx].min(),len(large_byMkt_idx)

        # (2c) Perform similar check by industry
        lowerBound = 5
        logging.info('Filtering by top %d%% mcap on industry', 100-lowerBound)
        (large_byIndtry_idx, nonest3) = buildEstu.exclude_by_cap_ranking(data, modelDate, baseEstu=estu_withoutThin_idx,byFactorType=ExposureMatrix.IndustryFactor,lower_pctile=lowerBound, method='percentage',excludeFactors=excludeFactors)
                                                                        #weight='rootCap')
        # print mktcap.iloc[large_byIndtry_idx].min(),len(large_byIndtry_idx)

        # estu_mktCap_idx = set(large_byMkt_idx).union(large_byCntry_idx).union(large_byIndtry_idx)
        # estu_mktCap_idx = set(large_byMkt_idx).intersection(large_byIndtry_idx)
        # estu_mktCap_idx = list(estu_mktCap_idx)
        # tinyCap_idx = list(set(estu_withoutThin_idx).difference(estu_mktCap_idx))
        # logging.info('ESTU currently stands at %d stocks after Filtering by Market Cap.', len(estu_mktCap_idx))
        # candid_univ_idx = eligibleUniverseIdx
        # candid_univ_idx = estu_withoutThin_idx
        # estu_withoutThin_idx2 = list(set(estu_withoutThin_idx).intersection(set(large_byMkt_idx).union(large_byIndtry_idx)))
        candid_univ_idx = list(set(estu_withoutThin_idx).intersection(set(large_byIndtry_idx)))
        # Inflate any thin countries or industries
        minFactorWidth=self.returnCalculator.allParameters[0].getThinFactorInformation().dummyThreshold
        logging.info('Inflating any thin factors')

        # import pandas as pd
        # estu_subids = [data.universe[x].getSubIDString() for x in estu_Base_idx]
        # estu_subids = pd.DataFrame(estu_subids,columns=['SubID'])
        # estu_subids.to_excel('research_estu.xlsx')

        # (estu_inflated_idx, nonest,herf_num_list) = buildEstu.pump_up_factors(data, modelDate,currentEstu=estu_Base_idx,        baseEstu=eligibleUniverseIdx,byFactorType=[ExposureMatrix.CountryFactor, ExposureMatrix.IndustryFactor],minFactorWidth=minFactorWidth,cutOff = self.inflation_cutoff, excludeFactors=excludeFactors)
        (estu_inflated_idx, nonest) = buildEstu.pump_up_factors(data, modelDate,currentEstu=estu_Base_idx,        baseEstu=eligibleUniverseIdx,byFactorType=[ExposureMatrix.CountryFactor, ExposureMatrix.IndustryFactor],minFactorWidth=minFactorWidth,cutOff = self.inflation_cutoff, excludeFactors=excludeFactors)
        logging.info('ESTU currently stands at %d stocks', len(estu_inflated_idx))
        # herf_num_list = pd.DataFrame(herf_num_list)
        # herf_num_list.to_csv('herf_num_list.csv')
        # Apply grandfathering rules
        # import ipdb;ipdb.set_trace()
        logging.info('Incorporating grandfathering')
        (estu_final_Idx, ESTUQualify, nonest) = buildEstu.grandfather(modelDate, estu_inflated_idx, baseEstu=eligibleUniverseIdx,
                                                                estuInstance=self.estuMap['main'])

        totalcap = ma.sum(ma.take(data.marketCaps, estu_final_Idx, axis=0), axis=0) / 1e9
        self.log.info('Final estu contains %d assets, %.2f bn (%s)',len(estu_final_Idx), totalcap, self.numeraire.currency_code)
        self.log.info('Grandfather Rule Brings in %d assets',len(estu_final_Idx)-len(estu_inflated_idx))
        self.log.info('Final estu contains %d assets in thin-traded assets.',sum(pd.Series(estu_final_Idx).isin(thin_idx)))
        # self.log.info('Final estu contains %d assets in tiny cap assets.',sum(pd.Series(estu_final_Idx).isin(tinyCap_idx)))
        self.log.debug('generate_estimation_universe: end')

        # If we have a family of estimation universes, populate the main estu accordingly
        self.estuMap['main'].assets = [buildEstu.assets[idx] for idx in estu_final_Idx]
        self.estuMap['main'].qualify = [buildEstu.assets[idx] for idx in ESTUQualify]
        # import ipdb;ipdb.set_trace()
        print('final_est:', len(estu_final_Idx))
        return estu_final_Idx

    def generate_estimation_universe(self, modelDate, data, modelDB, marketDB, excludeFactors=None):
        estu1 = self.generate_estimation_universe_v2(modelDate, data, modelDB, marketDB, excludeFactors=None)
        # estu2 = self.generate_estimation_universe_old(modelDate, data, modelDB, marketDB, excludeFactors=None)
        # print modelDate,len(estu1),len(estu2)
        return estu1

    def generate_model_specific_exposures(self, modelDate, data, modelDB, marketDB):
        """Generate the non-default factors.
        """
        beta = numpy.zeros((len(data.universe)), float)
        # Cap-based style factors here
        if not hasattr(self, 'estuMap') or self.estuMap is None:
            return data.exposureMatrix
        # import ipdb;ipdb.set_trace()
        # Mid/Small-cap factors
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

class AU2Yilin_d19(MFM.FundamentalModel):
    """AU2 base model.
    """
    # Model Parameters:
    rm_id, revision, rms_id = [-20, 33, -147]
    k  = 5.0
    dummyThreshold = 6     # dummyThreshold = 6
    inflation_cutoff = 0.03
    minNonZero = 0.1

    # Model Parameters, Workflow flag
    standardizationStats = True
    globalDescriptorModel = True
    DLCEnabled = True
    SCM = True

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
            'EM Sensitivity': ['Market_Sensitivity_EM_W','Market_Sensitivity_CN_W'],
            'Profitability': ['Return_on_Equity_Annual','Return_on_Assets_Annual', 'CashFlow_to_Assets_Annual',
                              'CashFlow_to_Income_Annual','Sales_to_Assets_Annual', 'Gross_Margin_Annual'],
            }
    DescriptorWeights = {'Value': [0.5, 0.375,0.125],
                        'EM Sensitivity': [0.5,0.5]}
    smallCapMap = {}
    noProxyList = ['Dividend Yield']
    fillMissingList = ['Value', 'Leverage', 'Growth', 'Profitability']
    fillWithZeroList = ['Dividend Yield']
    shrinkList = {'Liquidity': 60,
                  'Market Sensitivity': 250,
                  'Volatility': 125,
                  'Medium-Term Momentum': 250}
    orthogList = {'Volatility': [['Market Sensitivity'], True, 1.0],'EM Sensitivity': [['Exchange Rate Sensitivity'], True, 1.0]}

    # Setting up market intercept if relevant
    interceptFactor = 'Market Intercept'
    intercept = ModelFactor(interceptFactor, interceptFactor)
    industryClassification = Classification.GICSCustomAU2(datetime.date(2016,9,1))
    estuAssetTypes = ['REIT', 'Com', 'StapSec']

    def __init__(self, modelDB, marketDB, expTreat=None):
        self.log = logging.getLogger('RiskModels.AU2Yilin')
        # Set up relevant styles to be created/used
        # ModelParameters.defaultExposureParameters(self, self.styleList)
        ModelParameters.defaultExposureParametersV3(self, self.styleList)
        # if expTreat == 'addNew':
        #     self.styles = [s for s in self.totalStyles if s.name in self.addList]
        #     self.variableStyles = True
        # else:
        #     self.styles = [s for s in self.totalStyles if s.name in self.styleList]
        self.styles = [s for s in self.totalStyles if s.name in self.styleList]
        MFM.FundamentalModel.__init__(self, ['SEDOL'], modelDB, marketDB)
        # MFM.FundamentalModel.__init__(self, ['CUSIP'], modelDB, marketDB) # it doesn't work!

        # Set up regression parameters

        self.returnCalculator = ModelParameters.defaultRegressionParameters(self, modelDB,dummyType='Sectors', dummyThreshold=self.dummyThreshold,marketRegression=False,kappa = self.k)

        # This controls the FMP regression
        self.fmpCalculator = ModelParameters.defaultRegressionParameters(self, modelDB,dummyType='Sectors',marketRegression=False,dummyThreshold=self.dummyThreshold,kappa=None)

        # Set up risk parameters
        ModelParameters.defaultFundamentalCovarianceParametersV3(self, nwLag=2,varDVAOnly=False, unboundedDVA=False)

        # Set up standardization parameters
        gloScope = Standardization.GlobalRelativeScope([f.name for f in self.styles])
        # self.exposureStandardization = Standardization.BucketizedStandardization([gloScope], fancyMAD=self.fancyMAD)
        self.exposureStandardization = Standardization.BucketizedStandardization([gloScope], fillWithZeroList=self.fillWithZeroList)

        self.indexSelector = MarketIndex.MarketIndexSelector(modelDB, marketDB)
        modelDB.createCurrencyCache(marketDB)

        # Set up descriptor standardization parameters
        descriptors = sorted(list(set([item for sublist in self.DescriptorMap.values() for item in sublist])))
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

    def generate_eligible_universe(self, modelDate, data, modelDB, marketDB, capBounds=None,assetTypes=['All-Com', 'REIT'],excludeTypes=['ComWI', 'UnCls', 'StatETF', 'LLC', 'LP', 'NonEqETF', 'ComETF']):
        """Creates subset of eligible assets for consideration
        in estimation universes
        """
        # import ipdb;ipdb.set_trace()
        self.log.info('generate_eligible_universe: begin')
        HomeCountry_List = ['AU','NZ']
        buildEstu = EstimationUniverse.ConstructEstimationUniverse(
                data.universe, self, modelDB, marketDB)
        universe = buildEstu.assets
        n = len(universe)
        stepNo = 0
        if assetTypes is None:
            assetTypes = []
        if excludeTypes is None:
            excludeTypes = []
        logging.info('Eligible Universe currently stands at %d stocks', n)

        # Remove assets from the exclusion table
        stepNo+=1
        logging.info('Step %d: Applying exclusion table', stepNo)
        (estuIdx, nonest) = buildEstu.apply_exclusion_list(modelDate)
        if n != len(estuIdx):
            logging.info('...Step %d: Eligible Universe down %d and currently stands at %d stocks',
                    stepNo, n-len(estuIdx), len(estuIdx))
            n = len(estuIdx)

        # Remove cloned assets
        if len(data.hardCloneMap) > 0:
            stepNo+=1
            logging.info('Step %d: Removing cloned assets', stepNo)
            cloneIdx = [data.assetIdxMap[sid] for sid in data.hardCloneMap.keys()]
            (estuIdx, nonest) = buildEstu.exclude_specific_assets(cloneIdx, baseEstu=estuIdx)
            if n != len(estuIdx):
                logging.info('...Step %d: Eligible Universe down %d and currently stands at %d stocks',
                        stepNo, n-len(estuIdx), len(estuIdx))
                n = len(estuIdx)

        # Include by asset type field
        stepNo+=1
        logging.info('Step %d: Include by asset types %s', stepNo, ','.join(assetTypes))
        (estuIdx, nonest) = buildEstu.exclude_by_asset_type(
                modelDate, data, includeFields=assetTypes, excludeFields=None,
                baseEstu = estuIdx)
        if n != len(estuIdx):
            logging.info('...Step %d: Eligible Universe down %d and currently stands at %d stocks',
                    stepNo, n-len(estuIdx), len(estuIdx))
            n = len(estuIdx)

        # Pull out Chinese H shares and redchips to save for later
        stepNo+=1
        assetTypes = ['HShares', 'RCPlus']
        logging.info('Step %d: Getting list of Chinese assets: %s', stepNo, ','.join(assetTypes))
        (chinese, nonest) = buildEstu.exclude_by_asset_type(
                modelDate, data, includeFields=assetTypes, excludeFields=None,
                baseEstu = estuIdx)
        if len(chinese) > 0:
            logging.info('...Step %d: Found %d Chinese H-shares and Redchips', stepNo, len(chinese))

        # Exclude by asset type field
        stepNo+=1
        logging.info('Step %d: Exclude default asset types %s', stepNo, ','.join(excludeTypes))
        (estuIdx, nonest) = buildEstu.exclude_by_asset_type(
                modelDate, data, includeFields=None, excludeFields=excludeTypes,
                baseEstu = estuIdx)
        if n != len(estuIdx):
            logging.info('...Step %d: Eligible Universe down %d and currently stands at %d stocks',
                    stepNo, n-len(estuIdx), len(estuIdx))
            n = len(estuIdx)

        # Remove various types of DRs and foreign listings
        stepNo+=1
        exTypes = ['NVDR', 'GlobalDR', 'DR', 'TDR', 'AmerDR', 'FStock', 'CDI']
        logging.info('Step %d: Exclude DR asset types %s', stepNo, ','.join(exTypes))
        (estuIdx, nonest) = buildEstu.exclude_by_asset_type(
                modelDate, data, includeFields=None, excludeFields=exTypes,
                baseEstu = estuIdx)
        if n != len(estuIdx):
            logging.info('...Step %d: Eligible Universe down %d and currently stands at %d stocks',
                    stepNo, n-len(estuIdx), len(estuIdx))
            n = len(estuIdx)

        # Remove trusts, funds and other odds and ends
        stepNo+=1
        exTypes = ['InvT', 'UnitT', 'CEFund', 'Misc']
        logging.info('Step %d: Exclude fund/trust asset types %s', stepNo, ','.join(exTypes))
        (estuIdx, nonest) = buildEstu.exclude_by_asset_type(
                modelDate, data, includeFields=None, excludeFields=exTypes,
                baseEstu = estuIdx)
        if n != len(estuIdx):
            logging.info('...Step %d: Eligible Universe down %d and currently stands at %d stocks',
                    stepNo, n-len(estuIdx), len(estuIdx))
            n = len(estuIdx)

        # Remove Chinese A and B shares
        stepNo+=1
        exTypes = ['AShares', 'BShares']
        logging.info('Step %d: Exclude Chinese asset types %s', stepNo, ','.join(exTypes))
        (estuIdx, nonest) = buildEstu.exclude_by_asset_type(
                modelDate, data, includeFields=None, excludeFields=exTypes,
                baseEstu = estuIdx)
        if n != len(estuIdx):
            logging.info('...Step %d: Eligible Universe down %d and currently stands at %d stocks',
                    stepNo, n-len(estuIdx), len(estuIdx))
            n = len(estuIdx)

        # Remove assets classed as foreign via home market classification
        dr_indices = []
        if len(data.foreign) > 0:
            stepNo+=1
            logging.info('Step %d: Excluding foreign listings by home country mapping', stepNo)
            dr_indices = [data.assetIdxMap[sid] for sid in data.foreign]
            (estuIdx, nonest) = buildEstu.exclude_specific_assets(dr_indices, baseEstu=estuIdx)
            if n != len(estuIdx):
                logging.info('...Step %d: Eligible Universe down %d and currently stands at %d stocks',
                        stepNo, n-len(estuIdx), len(estuIdx))
                n = len(estuIdx)

        # Remove assets classed as foreign via home market classification - part 2
        stepNo+=1
        # import ipdb;ipdb.set_trace()
        logging.info('Step %d: Excluding foreign listings by market classification', stepNo)
        (estuIdx2, nonest) = buildEstu.exclude_by_market_classification(modelDate, 'HomeCountry', 'REGIONS', HomeCountry_List, baseEstu=estuIdx)
        estuIdx = estuIdx2
        if n != len(estuIdx):
            logging.info('...Step %d: Eligible Universe down %d and currently stands at %d stocks',
                    stepNo, n-len(estuIdx), len(estuIdx))
            n = len(estuIdx)

        # # Weed out foreign issuers by ISIN country prefix
        # stepNo+=1
        # # import ipdb;ipdb.set_trace()
        # logging.info('Step %d: Excluding foreign listings by ISIN prefix', stepNo)
        # (estuIdx, nonest)  = buildEstu.exclude_by_isin_country(
        #         [r.mnemonic for r in self.rmg], modelDate, baseEstu=estuIdx)
        # if n != len(estuIdx):
        #     logging.info('...Step %d: Eligible Universe down %d and currently stands at %d stocks',
        #             stepNo, n-len(estuIdx), len(estuIdx))
        #     n = len(estuIdx)
        # Manually add H-shares and Red-Chips back into list of eligible assets
        stepNo+=1
        logging.info('Step %d: Adding back %d H-Shares and Redchips', stepNo, len(chinese))
        estuIdx = set(estuIdx).union(set(chinese))
        if n != len(estuIdx):
            logging.info('...Step %d: Eligible Universe up by %d and currently stands at %d stocks',
                    stepNo, len(estuIdx)-n, len(estuIdx))
            n = len(estuIdx)

        # Logic to selectively filter RTS/MICEX Russian stocks
        rmgID = [rmg.rmg_id for rmg in self.rmg if rmg.mnemonic=='RU']
        if len(rmgID) > 0:
            stepNo+=1
            logging.info('Step %d: Cleaning up Russian assets', stepNo)
            # Get lists of RTS and MICEX-quoted assets
            (rtsIdx, nonest) = buildEstu.exclude_by_market_type(
                    modelDate, data, includeFields='RTS', excludeFields=None,
                    baseEstu = estuIdx)
            (micIdx, nonest) = buildEstu.exclude_by_market_type(
                    modelDate, data, includeFields='MIC', excludeFields=None,
                    baseEstu = estuIdx)
            # Find companies with multiple listings which include lines
            # on both exchanges
            rtsDropList = []
            for (groupID, sidList) in data.subIssueGroups.items():
                if len(sidList) > 1:
                    subIdxList = [data.assetIdxMap[sid] for sid in sidList]
                    rtsOverlap = set(rtsIdx).intersection(subIdxList)
                    micOverLap = set(micIdx).intersection(subIdxList)
                    if len(micOverLap) > 0:
                        rtsDropList.extend(rtsOverlap)
            estuIdx = estuIdx.difference(rtsDropList)
            if n != len(estuIdx):
                logging.info('...Step %d: Eligible Universe down %d and currently stands at %d stocks',
                        stepNo, n-len(estuIdx), len(estuIdx))
                n = len(estuIdx)

        # Remove preferred stock except for selected markets
        prefMarketList = ['BR','CO']
        prefAssetBase = []
        mktsUsed = []
        for mkt in prefMarketList:
            # Find assets in our current estu that are in the relevant markets
            rmgID = [rmg.rmg_id for rmg in self.rmg if rmg.mnemonic==mkt]
            if len(rmgID) > 0:
                prefAssetBase.extend(data.rmgAssetMap[rmgID[0]])
                mktsUsed.append(mkt)
        baseEstuIdx = [data.assetIdxMap[sid] for sid in prefAssetBase]
        baseEstuIdx = list(set(baseEstuIdx).intersection(set(estuIdx)))
        if len(baseEstuIdx) > 0:
            # Find which of our subset of assets are preferred stock
            stepNo+=1
            logging.info('Step %d: Dropping preferred stocks NOT on markets: %s', stepNo, ','.join(mktsUsed))
            (prefIdx, nonest) = buildEstu.exclude_by_asset_type(
                    modelDate, data,
                    includeFields=['All-Pref'], excludeFields=None,
                    baseEstu=baseEstuIdx)
            # Get rid of preferred stock in general
            (estuIdx, nonest) = buildEstu.exclude_by_asset_type(
                    modelDate, data,
                    includeFields=None, excludeFields=['All-Pref'],
                    baseEstu=estuIdx)
            # Add back in allowed preferred stock
            estuIdx = list(set(estuIdx).union(prefIdx))
            if n != len(estuIdx):
                logging.info('...Step %d: Eligible Universe down %d and currently stands at %d stocks',
                        stepNo, n-len(estuIdx), len(estuIdx))
                n = len(estuIdx)

        # Get rid of some exchanges we don't want
        exchangeCodes = ['REG']
        if modelDate.year > 2009:
            exchangeCodes.extend(['XSQ','OFE','XLF']) # GB junk
        stepNo+=1
        logging.info('Step %d: Removing stocks on exchanges: %s', stepNo, ','.join(exchangeCodes))
        (estuIdx, nonest) = buildEstu.exclude_by_market_type(
                modelDate, data, includeFields=None, excludeFields=exchangeCodes,
                baseEstu = estuIdx)
        if n != len(estuIdx):
            logging.info('...Step %d: Eligible Universe down %d and currently stands at %d stocks',
                    stepNo, n-len(estuIdx), len(estuIdx))
            n = len(estuIdx)

        # Limit stocks to certain exchanges in select countries
        mktExchangeMap = {'CA': ['TSE'],
                          'JP': ['TKS-S1','TKS-S2','OSE-S1','OSE-S2','TKS-MSC','TKS-M','JAS']
                         }
        if modelDate.year > 2009:
            mktExchangeMap['US'] = ['NAS','NYS']
        for mkt in mktExchangeMap.keys():
            rmgID = [rmg.rmg_id for rmg in self.rmg if rmg.mnemonic==mkt]
            if len(rmgID) < 1:
                continue
            stepNo+=1
            logging.info('Step %d: Dropping %s stocks NOT on exchanges: %s', stepNo,
                    mkt, ','.join(mktExchangeMap[mkt]))

            # Pull out subset of assets on relevant market
            baseEstuIdx = [data.assetIdxMap[sid] for sid in data.rmgAssetMap[rmgID[0]]]
            baseEstuIdx = list(set(baseEstuIdx).intersection(set(estuIdx)))
            if len(baseEstuIdx) < 1:
                continue

            # Determine whether they are on permitted exchange
            (mktEstuIdx, nonest) = buildEstu.exclude_by_market_type(
                    modelDate, data, includeFields=mktExchangeMap[mkt], excludeFields=None,
                    baseEstu = baseEstuIdx)

            # Remove undesirables
            nonMktEstuIdx = set(baseEstuIdx).difference(set(mktEstuIdx))
            estuIdx = list(set(estuIdx).difference(nonMktEstuIdx))
            if n != len(estuIdx):
                logging.info('...Step %d: Eligible Universe down %d and currently stands at %d stocks',
                    stepNo, n-len(estuIdx), len(estuIdx))
                n = len(estuIdx)

        estu = [universe[idx] for idx in estuIdx]
        data.eligibleUniverseIdx = estuIdx
        logging.info('%d eligible assets out of %d total', len(estu), len(universe))
        self.log.info('generate_eligible_universe: end')
        return estu

    def generate_estimation_universe(self, modelDate, data, modelDB, marketDB, excludeFactors=None):
        """Estimation universe selection criteria for AX-AU.
        """
        import pandas as pd
        # import ipdb;ipdb.set_trace()
        self.log.info('generate_estimation_universe for AU Model: begin')
        buildEstu = EstimationUniverse.ConstructEstimationUniverse(data.universe, self, modelDB, marketDB)

        # Set up various eligible and total universes
        universeIdx = list(range(len(buildEstu.assets)))
        originalEligibleUniverse = list(data.eligibleUniverse)
        originalEligibleUniverseIdx = [data.assetIdxMap[sid] for sid in originalEligibleUniverse]

        universe = pd.DataFrame(buildEstu.assets,columns=['SubID_Obj'])
        universeIdx = universe.index
        original_eligibleUniverse = pd.DataFrame(list(zip(data.eligibleUniverse,originalEligibleUniverseIdx)),columns=['SubID_Obj','originalEligibleUniverseIdx'])

        logging.info('ESTU currently stands at %d stocks based on original eligible universe', len(originalEligibleUniverse))

        # (1) Filtering thinly-traded assets over the entire universe
        logging.info('Looking for thinly-traded stocks')
        # (nonSparseIdx, sparse) = buildEstu.exclude_thinly_traded_assets(modelDate, data, baseEstu=universeIdx, minNonZero=0.75)
        # data.nonSparse = numpy.take(buildEstu.assets, nonSparseIdx, axis=0)
        (non_thin_idx, thin_idx) = buildEstu.exclude_thinly_traded_assets(modelDate, data, baseEstu=universe.index.values, minNonZero=self.minNonZero)

        data.nonSparse = pd.DataFrame(universe,index=non_thin_idx)["SubID_Obj"].tolist()
        # Remove nursery market assets
        if len(data.nurseryUniverse) > 0:
            logging.info('Checking for assets from nursery markets')
            raise('ERROR: Ask for what is the nurseryUniverse for?')
        else:
            eligibleUniverseIdx = originalEligibleUniverseIdx

        # Exclude thinly traded assets
        estu_withoutThin_idx = list(set(eligibleUniverseIdx).intersection(set(non_thin_idx)))
        logging.info('ESTU currently stands at %d stocks after Filtering thinly-traded assets', len(estu_withoutThin_idx))
        # Rank stuff by market cap and total volume over past year
        # import ipdb;ipdb.set_trace()
        (estu_Base_idx, nonest) = buildEstu.filter_by_cap_and_volume(data, modelDate, baseEstu=estu_withoutThin_idx, hiCapQuota=200, loCapQuota=100, bufferFactor=1.2)
        # (2) Filtering tiny-cap assets by market, country and industry
        # (2a) Weed out tiny-cap assets by market
        # lowerBound = 5
        # logging.info('Filtering by top %d%% mcap on entire market', 100-lowerBound)
        # (large_byMkt_idx, nonest1) = buildEstu.exclude_by_cap_ranking(data, modelDate, baseEstu=estu_withoutThin_idx,
        #                                                              lower_pctile=lowerBound, method='percentage')
        #         #weight='rootCap')
        # subissue_id_objs = data.universe
        # mktcap = modelDB.getAverageMarketCaps([modelDate], subissue_id_objs, currencyID = None)
        # mktcap = pd.DataFrame(mktcap)
        # print mktcap.ix[large_byMkt_idx].min(),len(large_byMkt_idx)

        # (2c) Perform similar check by industry
        lowerBound = 5
        logging.info('Filtering by top %d%% mcap on industry', 100-lowerBound)
        (large_byIndtry_idx, nonest3) = buildEstu.exclude_by_cap_ranking(data, modelDate, baseEstu=estu_withoutThin_idx,byFactorType=ExposureMatrix.IndustryFactor,lower_pctile=lowerBound, method='percentage',excludeFactors=excludeFactors)
                                                                        #weight='rootCap')
        # print mktcap.iloc[large_byIndtry_idx].min(),len(large_byIndtry_idx)

        # estu_mktCap_idx = set(large_byMkt_idx).union(large_byCntry_idx).union(large_byIndtry_idx)
        # estu_mktCap_idx = set(large_byMkt_idx).intersection(large_byIndtry_idx)
        # estu_mktCap_idx = list(estu_mktCap_idx)
        # tinyCap_idx = list(set(estu_withoutThin_idx).difference(estu_mktCap_idx))
        # logging.info('ESTU currently stands at %d stocks after Filtering by Market Cap.', len(estu_mktCap_idx))
        # candid_univ_idx = eligibleUniverseIdx
        # candid_univ_idx = estu_withoutThin_idx
        # estu_withoutThin_idx2 = list(set(estu_withoutThin_idx).intersection(set(large_byMkt_idx).union(large_byIndtry_idx)))
        candid_univ_idx = list(set(estu_withoutThin_idx).intersection(set(large_byIndtry_idx)))
        # Inflate any thin countries or industries
        minFactorWidth=self.returnCalculator.allParameters[0].getThinFactorInformation().dummyThreshold
        logging.info('Inflating any thin factors')

        # import pandas as pd
        # estu_subids = [data.universe[x].getSubIDString() for x in estu_Base_idx]
        # estu_subids = pd.DataFrame(estu_subids,columns=['SubID'])
        # estu_subids.to_excel('research_estu.xlsx')

        # (estu_inflated_idx, nonest,herf_num_list) = buildEstu.pump_up_factors(data, modelDate,currentEstu=estu_Base_idx,        baseEstu=eligibleUniverseIdx,byFactorType=[ExposureMatrix.CountryFactor, ExposureMatrix.IndustryFactor],minFactorWidth=minFactorWidth,cutOff = self.inflation_cutoff, excludeFactors=excludeFactors)
        (estu_inflated_idx, nonest) = buildEstu.pump_up_factors(data, modelDate,currentEstu=estu_Base_idx,        baseEstu=eligibleUniverseIdx,byFactorType=[ExposureMatrix.CountryFactor, ExposureMatrix.IndustryFactor],minFactorWidth=minFactorWidth,cutOff = self.inflation_cutoff, excludeFactors=excludeFactors)
        logging.info('ESTU currently stands at %d stocks', len(estu_inflated_idx))
        # herf_num_list = pd.DataFrame(herf_num_list)
        # herf_num_list.to_csv('herf_num_list.csv')
        # Apply grandfathering rules
        # import ipdb;ipdb.set_trace()
        logging.info('Incorporating grandfathering')
        (estu_final_Idx, ESTUQualify, nonest) = buildEstu.grandfather(modelDate, estu_inflated_idx, baseEstu=eligibleUniverseIdx,daysBack=61,
                                                                estuInstance=self.estuMap['main'])

        totalcap = ma.sum(ma.take(data.marketCaps, estu_final_Idx, axis=0), axis=0) / 1e9
        self.log.info('Final estu contains %d assets, %.2f bn (%s)',len(estu_final_Idx), totalcap, self.numeraire.currency_code)
        self.log.info('Grandfather Rule Brings in %d assets',len(estu_final_Idx)-len(estu_inflated_idx))
        self.log.info('Final estu contains %d assets in thin-traded assets.',sum(pd.Series(estu_final_Idx).isin(thin_idx)))
        # self.log.info('Final estu contains %d assets in tiny cap assets.',sum(pd.Series(estu_final_Idx).isin(tinyCap_idx)))
        self.log.debug('generate_estimation_universe: end')

        # If we have a family of estimation universes, populate the main estu accordingly
        self.estuMap['main'].assets = [buildEstu.assets[idx] for idx in estu_final_Idx]
        self.estuMap['main'].qualify = [buildEstu.assets[idx] for idx in ESTUQualify]
        # import ipdb;ipdb.set_trace()
        print('final_est:', len(estu_final_Idx))
        return estu_final_Idx

    def generate_model_specific_exposures(self, modelDate, data, modelDB, marketDB):
        """Generate the non-default factors.
        """
        beta = numpy.zeros((len(data.universe)), float)
        # Cap-based style factors here
        if not hasattr(self, 'estuMap') or self.estuMap is None:
            return data.exposureMatrix
        # import ipdb;ipdb.set_trace()
        # Mid/Small-cap factors
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

class AU2Yilin_d18_S(EquityModel.StatisticalModel):
    """AU2 statistical model"""

    # Model Parameters:
    rm_id,revision,rms_id = [-21,1,-201]

    numFactors = 15
    blind = [ModelFactor('Statistical Factor %d' % n,'Statistical Factor %d' % n) for n in range(1, numFactors+1)]
    pcaHistory = 250
    allowETFs = True
    # industryClassification = Classification.GICSCustomAU(datetime.date(2008,8,30))
    industryClassification = Classification.GICSCustomAU2(datetime.date(2016,9,1))
    newExposureFormat = True

    def __init__(self, modelDB, marketDB):
        # import ipdb;ipdb.set_trace()
        self.log = logging.getLogger('RiskModels.AU2Research-S')
        EquityModel.StatisticalModel.__init__(self, ['SEDOL'], modelDB, marketDB)
        # So we can use the same ESTU method as the fundamental model
        self.baseModelDateMap = {datetime.date(1980,1,1): AU2Yilin_d18(modelDB, marketDB)}
        # Set up returns model
        self.returnCalculator = FactorReturns.AsymptoticPrincipalComponents(self.numFactors)
        # Set up risk parameters
        ModelParameters.defaultStatisticalCovarianceParametersV3(self, shrinkFactor=None, modelHorizon='short')
        self.indexSelector = MarketIndex.MarketIndexSelector(modelDB, marketDB)

class AU2Yilin_RandomFactor1(AU2Yilin):
# base on AU2Yilin base model,
# - add one more random factor and check siginificant level

    rm_id = -20
    revision = 13
    rms_id = -127

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
                 'Random Factor'
                 ]
    DescriptorMap = {
            'Earnings Yield': ['Earnings_to_Price_Annual', 'Est_Earnings_to_Price_Annual'],
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
            'Profitability': ['Return_on_Equity_Annual','Return_on_Assets_Annual', 'CashFlow_to_Assets_Annual',
                              'CashFlow_to_Income_Annual','Sales_to_Assets_Annual', 'Gross_Margin_Annual'],
            'Random Factor': ['Random_Factor'],
            }

class AU2Yilin_d1(AU2Yilin_d18):
# base on AU2Yilin base model,
# - merge Earning Yield and Value as one Value factor
    rm_id,revision,rms_id = [-20,15,-129]
    # List of style factors in the model
    styleList = ['Value','Leverage','Growth','Profitability','Dividend Yield','Size',
                 'Liquidity','Market Sensitivity','Volatility','Medium-Term Momentum',
                 'MidCap','Exchange Rate Sensitivity'
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
            'Profitability': ['Return_on_Equity_Annual','Return_on_Assets_Annual', 'CashFlow_to_Assets_Annual',
                              'CashFlow_to_Income_Annual','Sales_to_Assets_Annual', 'Gross_Margin_Annual'],
            }
    DescriptorWeights = {'Value': [0.5, 0.375,0.125]}
    fillMissingList = ['Value', 'Leverage', 'Growth', 'Profitability']

class AU2Yilin_d2(AU2Yilin):
# base on AU2Yilin base model - remove midcap factor
    rm_id,revision,rms_id = [-20,16,-130]
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
    smallCapMap = {}

class AU2Yilin_d6(AU2Yilin):
# AU2Yilin, without MidCap, merge BP and EY
    rm_id,revision,rms_id = [-20,20,-134]
    # List of style factors in the model
    styleList = ['Value','Leverage','Growth','Profitability','Dividend Yield','Size',
                 'Liquidity','Market Sensitivity','Volatility','Medium-Term Momentum',
                 'Exchange Rate Sensitivity'
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
            'Profitability': ['Return_on_Equity_Annual','Return_on_Assets_Annual', 'CashFlow_to_Assets_Annual',
                              'CashFlow_to_Income_Annual','Sales_to_Assets_Annual', 'Gross_Margin_Annual'],
            }
    DescriptorWeights = {'Value': [0.5, 0.375,0.125]}
    fillMissingList = ['Value', 'Leverage', 'Growth', 'Profitability']
    smallCapMap = {}

class AU2Yilin_d7(AU2Yilin):
# AU2Yilin, without MidCap, merge BP and EY and DY
    rm_id,revision,rms_id = [-20,21,-135]
  # List of style factors in the model
    # List of style factors in the model
    styleList = ['Value','Leverage','Growth','Profitability','Size',
                 'Liquidity','Market Sensitivity','Volatility','Medium-Term Momentum',
                 'Exchange Rate Sensitivity'
                 ]
    DescriptorMap = {
            'Value': ['Book_to_Price_Annual','Dividend_Yield_Annual','Earnings_to_Price_Annual', 'Est_Earnings_to_Price_Annual'],
            'Leverage': ['Debt_to_Assets_Annual', 'Debt_to_Equity_Annual'],
            'Growth': ['Earnings_Growth_RPF_Annual', 'Sales_Growth_RPF_Annual'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_60D'],
            'Market Sensitivity': ['Market_Sensitivity_250D'],
            'Volatility': ['Volatility_125D'],
            'Medium-Term Momentum': ['Momentum_250x20D'],
            'Exchange Rate Sensitivity': ['XRate_104W_XDR'],
            'Profitability': ['Return_on_Equity_Annual','Return_on_Assets_Annual', 'CashFlow_to_Assets_Annual',
                              'CashFlow_to_Income_Annual','Sales_to_Assets_Annual', 'Gross_Margin_Annual'],
            }
    DescriptorWeights = {'Value': [0.34,0.33, 0.22,0.11]}
    fillMissingList = ['Value', 'Leverage', 'Growth', 'Profitability']
    smallCapMap = {}
    noProxyList = []
    fillWithZeroList = []

class AU2Yilin_d10(AU2Yilin):
# AU2Yilin, based on D6, based on D6, adding EM+CN Effect Factor
    rm_id,revision,rms_id = [-20,24,-138]
    # List of style factors in the model
    styleList = ['Value','Leverage','Growth','Profitability','Dividend Yield','Size',
                 'Liquidity','Market Sensitivity','Volatility','Medium-Term Momentum',
                 'Exchange Rate Sensitivity'
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
            'Exchange Rate Sensitivity': ['XRate_104W_XDR','Market_Sensitivity_EM','Market_Sensitivity_CN'],
            'Profitability': ['Return_on_Equity_Annual','Return_on_Assets_Annual', 'CashFlow_to_Assets_Annual',
                              'CashFlow_to_Income_Annual','Sales_to_Assets_Annual', 'Gross_Margin_Annual'],
            }
    DescriptorWeights = {'Value': [0.5, 0.375,0.125],
                         'Exchange Rate Sensitivity': [0.34,0.33,0.33]}
    fillMissingList = ['Value', 'Leverage', 'Growth', 'Profitability']
    smallCapMap = {}

class AU2Yilin_d11(AU2Yilin):
# AU2Yilin, based on D6, adding EM Effect Factor
    rm_id,revision,rms_id = [-20,25,-139]
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
            'EM Sensitivity': ['Market_Sensitivity_EM'],
            'Profitability': ['Return_on_Equity_Annual','Return_on_Assets_Annual', 'CashFlow_to_Assets_Annual',
                              'CashFlow_to_Income_Annual','Sales_to_Assets_Annual', 'Gross_Margin_Annual'],
            }
    DescriptorWeights = {'Value': [0.5, 0.375,0.125]}
    fillMissingList = ['Value', 'Leverage', 'Growth', 'Profitability']
    smallCapMap = {}
    industryClassification = Classification.GICSCustomAU2(datetime.date(2016,9,1))

class AU2Yilin_d12(AU2Yilin_d11):
    rm_id, revision, rms_id = [-20, 26, -140]

    DescriptorMap = {
        'Value': ['Book_to_Price_Annual', 'Earnings_to_Price_Annual', 'Est_Earnings_to_Price_Annual'],
        'Leverage': ['Debt_to_Assets_Annual', 'Debt_to_Equity_Annual'],
        'Growth': ['Earnings_Growth_RPF_Annual', 'Sales_Growth_RPF_Annual'],
        'Dividend Yield': ['Dividend_Yield_Annual'],
        'Size': ['LnIssuerCap'],
        'Liquidity': ['LnTrading_Activity_60D'],
        'Market Sensitivity': ['Market_Sensitivity_250D'],
        'Volatility': ['Volatility_125D'],
        'Medium-Term Momentum': ['Momentum_250x20D'],
        'Exchange Rate Sensitivity': ['XRate_104W_XDR'],
        'EM Sensitivity': ['Market_Sensitivity_CN'],
        'Profitability': ['Return_on_Equity_Annual', 'Return_on_Assets_Annual', 'CashFlow_to_Assets_Annual',
                          'CashFlow_to_Income_Annual', 'Sales_to_Assets_Annual', 'Gross_Margin_Annual'],
    }

class AU2Yilin_d15(AU2Yilin_d11):
    rm_id,revision,rms_id = [-20,29,-143]

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
            'EM Sensitivity': ['Market_Sensitivity_EM_W','Market_Sensitivity_CN_W'],
            'Profitability': ['Return_on_Equity_Annual','Return_on_Assets_Annual', 'CashFlow_to_Assets_Annual',
                              'CashFlow_to_Income_Annual','Sales_to_Assets_Annual', 'Gross_Margin_Annual'],
            }
    orthogList = {'Volatility': [['Market Sensitivity'], True, 1.0],'EM Sensitivity': [['Exchange Rate Sensitivity'], True, 1.0]}
    DescriptorWeights = {'Value': [0.5, 0.375,0.125],
                        'EM Sensitivity': [0.5,0.5]}
    def generate_estimation_universe(self, modelDate, data, modelDB, marketDB, excludeFactors=None):
        """Estimation universe selection criteria for AX-AU.
        """
        import pandas as pd
        # import ipdb;ipdb.set_trace()
        self.log.info('generate_estimation_universe for AU Model: begin')
        buildEstu = EstimationUniverse.ConstructEstimationUniverse(data.universe, self, modelDB, marketDB)

        # Set up various eligible and total universes
        universeIdx = list(range(len(buildEstu.assets)))
        originalEligibleUniverse = list(data.eligibleUniverse)
        originalEligibleUniverseIdx = [data.assetIdxMap[sid] for sid in originalEligibleUniverse]

        universe = pd.DataFrame(buildEstu.assets,columns=['SubID_Obj'])
        universeIdx = universe.index
        original_eligibleUniverse = pd.DataFrame(list(zip(data.eligibleUniverse,originalEligibleUniverseIdx)),columns=['SubID_Obj','originalEligibleUniverseIdx'])

        logging.info('ESTU currently stands at %d stocks based on original eligible universe', len(originalEligibleUniverse))

        # (1) Filtering thinly-traded assets over the entire universe
        logging.info('Looking for thinly-traded stocks')
        # (nonSparseIdx, sparse) = buildEstu.exclude_thinly_traded_assets(modelDate, data, baseEstu=universeIdx, minNonZero=0.75)
        # data.nonSparse = numpy.take(buildEstu.assets, nonSparseIdx, axis=0)
        (non_thin_idx, thin_idx) = buildEstu.exclude_thinly_traded_assets(modelDate, data, baseEstu=universe.index.values, minNonZero=0.1)

        data.nonSparse = pd.DataFrame(universe,index=non_thin_idx)["SubID_Obj"].tolist()
        # Remove nursery market assets
        if len(data.nurseryUniverse) > 0:
            logging.info('Checking for assets from nursery markets')
            raise('ERROR: Ask for what is the nurseryUniverse for?')
        else:
            eligibleUniverseIdx = originalEligibleUniverseIdx

        # Exclude thinly traded assets
        estu_withoutThin_idx = list(set(eligibleUniverseIdx).intersection(set(non_thin_idx)))
        logging.info('ESTU currently stands at %d stocks after Filtering thinly-traded assets', len(estu_withoutThin_idx))
        # Rank stuff by market cap and total volume over past year
        # import ipdb;ipdb.set_trace()
        (estu_Base_idx, nonest) = buildEstu.filter_by_cap_and_volume(data, modelDate, baseEstu=estu_withoutThin_idx, hiCapQuota=200, loCapQuota=100, bufferFactor=1.2)
        # (2) Filtering tiny-cap assets by market, country and industry
        # (2a) Weed out tiny-cap assets by market
        # lowerBound = 5
        # logging.info('Filtering by top %d%% mcap on entire market', 100-lowerBound)
        # (large_byMkt_idx, nonest1) = buildEstu.exclude_by_cap_ranking(data, modelDate, baseEstu=estu_withoutThin_idx,
        #                                                              lower_pctile=lowerBound, method='percentage')
        #         #weight='rootCap')
        # subissue_id_objs = data.universe
        # mktcap = modelDB.getAverageMarketCaps([modelDate], subissue_id_objs, currencyID = None)
        # mktcap = pd.DataFrame(mktcap)
        # print mktcap.ix[large_byMkt_idx].min(),len(large_byMkt_idx)

        # (2c) Perform similar check by industry
        lowerBound = 5
        logging.info('Filtering by top %d%% mcap on industry', 100-lowerBound)
        (large_byIndtry_idx, nonest3) = buildEstu.exclude_by_cap_ranking(data, modelDate, baseEstu=estu_withoutThin_idx,byFactorType=ExposureMatrix.IndustryFactor,lower_pctile=lowerBound, method='percentage',excludeFactors=excludeFactors)
                                                                        #weight='rootCap')
        # print mktcap.iloc[large_byIndtry_idx].min(),len(large_byIndtry_idx)

        # estu_mktCap_idx = set(large_byMkt_idx).union(large_byCntry_idx).union(large_byIndtry_idx)
        # estu_mktCap_idx = set(large_byMkt_idx).intersection(large_byIndtry_idx)
        # estu_mktCap_idx = list(estu_mktCap_idx)
        # tinyCap_idx = list(set(estu_withoutThin_idx).difference(estu_mktCap_idx))
        # logging.info('ESTU currently stands at %d stocks after Filtering by Market Cap.', len(estu_mktCap_idx))
        # candid_univ_idx = eligibleUniverseIdx
        # candid_univ_idx = estu_withoutThin_idx
        # estu_withoutThin_idx2 = list(set(estu_withoutThin_idx).intersection(set(large_byMkt_idx).union(large_byIndtry_idx)))
        candid_univ_idx = list(set(estu_withoutThin_idx).intersection(set(large_byIndtry_idx)))
        # Inflate any thin countries or industries
        minFactorWidth=self.returnCalculator.allParameters[0].getThinFactorInformation().dummyThreshold
        logging.info('Inflating any thin factors')

        # import pandas as pd
        # estu_subids = [data.universe[x].getSubIDString() for x in estu_Base_idx]
        # estu_subids = pd.DataFrame(estu_subids,columns=['SubID'])
        # estu_subids.to_excel('research_estu.xlsx')

        # (estu_inflated_idx, nonest,herf_num_list) = buildEstu.pump_up_factors(data, modelDate,currentEstu=estu_Base_idx,        baseEstu=eligibleUniverseIdx,byFactorType=[ExposureMatrix.CountryFactor, ExposureMatrix.IndustryFactor],minFactorWidth=minFactorWidth,cutOff = self.inflation_cutoff, excludeFactors=excludeFactors)
        (estu_inflated_idx, nonest) = buildEstu.pump_up_factors(data, modelDate,currentEstu=estu_Base_idx,baseEstu=eligibleUniverseIdx,byFactorType=[ExposureMatrix.CountryFactor, ExposureMatrix.IndustryFactor],minFactorWidth=minFactorWidth,cutOff = self.inflation_cutoff, excludeFactors=excludeFactors)
        logging.info('ESTU currently stands at %d stocks', len(estu_inflated_idx))
        # herf_num_list = pd.DataFrame(herf_num_list)
        # herf_num_list.to_csv('herf_num_list.csv')
        # Apply grandfathering rules
        # import ipdb;ipdb.set_trace()
        logging.info('Incorporating grandfathering')
        (estu_final_Idx, ESTUQualify, nonest) = buildEstu.grandfather(modelDate, estu_inflated_idx, baseEstu=eligibleUniverseIdx,daysBack=61,
                                                                estuInstance=self.estuMap['main'])

        totalcap = ma.sum(ma.take(data.marketCaps, estu_final_Idx, axis=0), axis=0) / 1e9
        self.log.info('Final estu contains %d assets, %.2f bn (%s)',len(estu_final_Idx), totalcap, self.numeraire.currency_code)
        self.log.info('Grandfather Rule Brings in %d assets',len(estu_final_Idx)-len(estu_inflated_idx))
        self.log.info('Final estu contains %d assets in thin-traded assets.',sum(pd.Series(estu_final_Idx).isin(thin_idx)))
        # self.log.info('Final estu contains %d assets in tiny cap assets.',sum(pd.Series(estu_final_Idx).isin(tinyCap_idx)))
        self.log.debug('generate_estimation_universe: end')

        # If we have a family of estimation universes, populate the main estu accordingly
        self.estuMap['main'].assets = [buildEstu.assets[idx] for idx in estu_final_Idx]
        self.estuMap['main'].qualify = [buildEstu.assets[idx] for idx in ESTUQualify]
        # import ipdb;ipdb.set_trace()
        print('final_est:', len(estu_final_Idx))
        return estu_final_Idx





# class AU2Yilin_d18(AU2Yilin_d15):
#     rm_id,revision,rms_id = [-20,32,-146]
#     dummyThreshold = 6     # dummyThreshold = 6
#     inflation_cutoff = 0.03
# class AU2Yilin_d19(AU2Yilin_d15):
#     rm_id, revision, rms_id = [-20, 33, -147]
#     dummyThreshold = 10     # dummyThreshold = 6
#     inflation_cutoff = 0.03
class AU2Yilin_d20(AU2Yilin_d18):
    rm_id, revision, rms_id = [-20, 34, -148]
    # (1) for volatility
    orthogList = {}
    # # (2) for leverage
    # DescriptorMap = {
    #     'Value': ['Book_to_Price_Annual','Earnings_to_Price_Annual', 'Est_Earnings_to_Price_Annual'],
    #     'Leverage': ['Debt_to_MarketCap_Annual'],
    #     'Growth': ['Earnings_Growth_RPF_Annual', 'Sales_Growth_RPF_Annual'],
    #     'Dividend Yield': ['Dividend_Yield_Annual'],
    #     'Size': ['LnIssuerCap'],
    #     'Liquidity': ['LnTrading_Activity_60D'],
    #     'Market Sensitivity': ['Market_Sensitivity_250D'],
    #     'Volatility': ['Volatility_125D'],
    #     'Medium-Term Momentum': ['Momentum_250x20D'],
    #     'Exchange Rate Sensitivity': ['XRate_104W_XDR'],
    #     'EM Sensitivity': ['Market_Sensitivity_EM_W','Market_Sensitivity_CN_W'],
    #     'Profitability': ['Return_on_Equity_Annual','Return_on_Assets_Annual', 'CashFlow_to_Assets_Annual',
    #                       'CashFlow_to_Income_Annual','Sales_to_Assets_Annual', 'Gross_Margin_Annual'],
    #     }
# Tried:
# 'Exchange Rate Sensitivity': ['XRate_104W_XDR_V2'],
# 'EM Sensitivity': ['Market_Sensitivity_EM_W_V2','Market_Sensitivity_CN_W_V2'],

class AU2Yilin_d21(AU2Yilin_d18):
    rm_id = -20
    revision = 35
    rms_id = -149

    # List of style factors in the model
    styleList = ['Value','Leverage','Growth','Profitability','Dividend Yield','Size',
             'Liquidity','Market Sensitivity','Volatility','Medium-Term Momentum',
             'Exchange Rate Sensitivity','EM Sensitivity','Random Factor'
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
        'EM Sensitivity': ['Market_Sensitivity_EM_W','Market_Sensitivity_CN_W'],
        'Profitability': ['Return_on_Equity_Annual','Return_on_Assets_Annual', 'CashFlow_to_Assets_Annual',
                          'CashFlow_to_Income_Annual','Sales_to_Assets_Annual', 'Gross_Margin_Annual'],
        'Random Factor': ['Random_Factor'],
        }


class AU2Yilin_d23(AU2Yilin_d18):
    # variant1 of profitability factor
    rm_id = -20
    revision = 37
    rms_id = -151

    # List of style factors in the model
    DescriptorMap = {
        'Earnings Yield': ['Earnings_to_Price_Annual', 'Est_Earnings_to_Price_Annual'],
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
        'EM Sensitivity': ['Market_Sensitivity_EM_W','Market_Sensitivity_CN_W'],
        'Profitability': ['Return_on_Equity_Annual','Return_on_Assets_Annual'],
        }

class AU2Yilin_d24(AU2Yilin_d18):
    # variant2 of profitability factor
    rm_id = -20
    revision = 38
    rms_id = -152

    # List of style factors in the model
    DescriptorMap = {
        'Earnings Yield': ['Earnings_to_Price_Annual', 'Est_Earnings_to_Price_Annual'],
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
        'EM Sensitivity': ['Market_Sensitivity_EM_W','Market_Sensitivity_CN_W'],
        'Profitability': ['Return_on_Equity_Annual','Return_on_Assets_Annual'],
        }

# Profitability
class AU2Yilin_d3(AU2Yilin_d18):
# base on AU2Yilin_d18, what if profit factor just use ROE?
    rm_id,revision,rms_id = [-20,17,-131]
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
            'EM Sensitivity': ['Market_Sensitivity_EM_W','Market_Sensitivity_CN_W'],
            'Profitability': ['Return_on_Equity_Annual'],
            }

class AU2Yilin_d4(AU2Yilin_d18):
# base on AU2Yilin_d18, what if profit factor just use ROA?
    rm_id,revision,rms_id = [-20,18,-132]
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
            'EM Sensitivity': ['Market_Sensitivity_EM_W','Market_Sensitivity_CN_W'],
            'Profitability': ['Return_on_Assets_Annual'],
            }

class AU2Yilin_d5(AU2Yilin_d18):
# base on AU2Yilin_d18, what if profit factor just use ROE and ROA?

    rm_id,revision,rms_id = [-20,19,-133]

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
            'EM Sensitivity': ['Market_Sensitivity_EM_W','Market_Sensitivity_CN_W'],
            'Profitability': ['Return_on_Equity_Annual','Return_on_Assets_Annual'],
            }

# Value
class AU2Yilin_d16(AU2Yilin_d18):
# seperte Value and EY - version 1
    rm_id,revision,rms_id = [-20,30,-144]
    # List of style factors in the model
    styleList = ['Earnings Yield','Value','Leverage','Growth','Profitability','Dividend Yield','Size',
             'Liquidity','Market Sensitivity','Volatility','Medium-Term Momentum',
             'Exchange Rate Sensitivity','EM Sensitivity']
    DescriptorMap = {
        'Earnings Yield': ['Earnings_to_Price_Annual', 'Est_Earnings_to_Price_Annual'],
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
        'EM Sensitivity': ['Market_Sensitivity_EM_W','Market_Sensitivity_CN_W'],
        'Profitability': ['Return_on_Equity_Annual','Return_on_Assets_Annual', 'CashFlow_to_Assets_Annual',
                          'CashFlow_to_Income_Annual','Sales_to_Assets_Annual', 'Gross_Margin_Annual'],
        }
    DescriptorWeights = {'Earnings Yield': [0.5, 0.5],
                         'EM Sensitivity': [0.5,0.5]}

class AU2Yilin_d17(AU2Yilin_d18):
# seperte Value and EY - version 2
    rm_id,revision,rms_id = [-20,31,-145]
    # List of style factors in the model
    styleList = ['Earnings Yield','Value','Leverage','Growth','Profitability','Dividend Yield','Size',
             'Liquidity','Market Sensitivity','Volatility','Medium-Term Momentum',
             'Exchange Rate Sensitivity','EM Sensitivity']
    DescriptorMap = {
        'Earnings Yield': ['Earnings_to_Price_Annual', 'Est_Earnings_to_Price_Annual'],
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
        'EM Sensitivity': ['Market_Sensitivity_EM_W','Market_Sensitivity_CN_W'],
        'Profitability': ['Return_on_Equity_Annual','Return_on_Assets_Annual', 'CashFlow_to_Assets_Annual',
                          'CashFlow_to_Income_Annual','Sales_to_Assets_Annual', 'Gross_Margin_Annual'],
        }
    DescriptorWeights = {'Earnings Yield': [0.75, 0.25],
                         'EM Sensitivity': [0.5,0.5]}

class AU2Yilin_d22(AU2Yilin_d18):
# seperte Value and EY - version 3
    rm_id,revision,rms_id = [-20,36,-150]
    # List of style factors in the model
    styleList = ['Earnings Yield','Value','Leverage','Growth','Profitability','Dividend Yield','Size',
             'Liquidity','Market Sensitivity','Volatility','Medium-Term Momentum',
             'Exchange Rate Sensitivity','EM Sensitivity']
    DescriptorMap = {
        'Earnings Yield': ['Earnings_to_Price_Annual', 'Est_Earnings_to_Price_Annual'],
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
        'EM Sensitivity': ['Market_Sensitivity_EM_W','Market_Sensitivity_CN_W'],
        'Profitability': ['Return_on_Equity_Annual','Return_on_Assets_Annual', 'CashFlow_to_Assets_Annual',
                          'CashFlow_to_Income_Annual','Sales_to_Assets_Annual', 'Gross_Margin_Annual'],
        }
    DescriptorWeights = {'Earnings Yield': [0.25, 0.75],
                         'EM Sensitivity': [0.5,0.5]}

# Em Factor
class AU2Yilin_d13(AU2Yilin_d18):
    rm_id,revision,rms_id = [-20,27,-141]
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
            'EM Sensitivity': ['Market_Sensitivity_EM_W'],
            'Profitability': ['Return_on_Equity_Annual','Return_on_Assets_Annual', 'CashFlow_to_Assets_Annual',
                              'CashFlow_to_Income_Annual','Sales_to_Assets_Annual', 'Gross_Margin_Annual'],
            }
    DescriptorWeights = {'Value': [0.5, 0.375,0.125]}

class AU2Yilin_d14(AU2Yilin_d18):
    rm_id,revision,rms_id = [-20,28,-142]
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
            'EM Sensitivity': ['Market_Sensitivity_CN_W'],
            'Profitability': ['Return_on_Equity_Annual','Return_on_Assets_Annual', 'CashFlow_to_Assets_Annual',
                              'CashFlow_to_Income_Annual','Sales_to_Assets_Annual', 'Gross_Margin_Annual'],
            }
    DescriptorWeights = {'Value': [0.5, 0.375,0.125]}

# # leverage Factor
# class AU2Yilin_d8(AU2Yilin_d18):
# # AU2Yilin, based on d18, study leverage factor
#     rm_id,revision,rms_id = [-20,22,-136]
#     DescriptorMap = {
#             'Value': ['Book_to_Price_Annual','Earnings_to_Price_Annual', 'Est_Earnings_to_Price_Annual'],
#             'Leverage': ['Debt_to_Assets_Annual'],
#             'Growth': ['Earnings_Growth_RPF_Annual', 'Sales_Growth_RPF_Annual'],
#             'Dividend Yield': ['Dividend_Yield_Annual'],
#             'Size': ['LnIssuerCap'],
#             'Liquidity': ['LnTrading_Activity_60D'],
#             'Market Sensitivity': ['Market_Sensitivity_250D'],
#             'Volatility': ['Volatility_125D'],
#             'Medium-Term Momentum': ['Momentum_250x20D'],
#             'Exchange Rate Sensitivity': ['XRate_104W_XDR'],
#             'EM Sensitivity': ['Market_Sensitivity_EM_W','Market_Sensitivity_CN_W'],
#             'Profitability': ['Return_on_Equity_Annual','Return_on_Assets_Annual', 'CashFlow_to_Assets_Annual',
#                               'CashFlow_to_Income_Annual','Sales_to_Assets_Annual', 'Gross_Margin_Annual'],
#             }
# class AU2Yilin_d9(AU2Yilin_d18):
# # AU2Yilin, based on d18, study leverage factor
#     rm_id,revision,rms_id = [-20,23,-137]
#     DescriptorMap = {
#             'Value': ['Book_to_Price_Annual','Earnings_to_Price_Annual', 'Est_Earnings_to_Price_Annual'],
#             'Leverage': ['Debt_to_MarketCap_Annual'],
#             'Growth': ['Earnings_Growth_RPF_Annual', 'Sales_Growth_RPF_Annual'],
#             'Dividend Yield': ['Dividend_Yield_Annual'],
#             'Size': ['LnIssuerCap'],
#             'Liquidity': ['LnTrading_Activity_60D'],
#             'Market Sensitivity': ['Market_Sensitivity_250D'],
#             'Volatility': ['Volatility_125D'],
#             'Medium-Term Momentum': ['Momentum_250x20D'],
#             'Exchange Rate Sensitivity': ['XRate_104W_XDR'],
#             'EM Sensitivity': ['Market_Sensitivity_EM_W','Market_Sensitivity_CN_W'],
#             'Profitability': ['Return_on_Equity_Annual','Return_on_Assets_Annual', 'CashFlow_to_Assets_Annual',
#                               'CashFlow_to_Income_Annual','Sales_to_Assets_Annual', 'Gross_Margin_Annual'],
#             }

# study the inconsistency
# class AU2Yilin_d8(AUAxioma2016MH):
# # investigating the difference caused by EM discriptor
#     rm_id,revision,rms_id = [-20,22,-136]
#     DescriptorMap = {
#         'Value': ['Book_to_Price_Annual','Earnings_to_Price_Annual', 'Est_Earnings_to_Price_Annual'],
#         'Leverage': ['Debt_to_Assets_Annual', 'Debt_to_Equity_Annual'],
#         'Growth': ['Earnings_Growth_RPF_Annual', 'Sales_Growth_RPF_Annual'],
#         'Dividend Yield': ['Dividend_Yield_Annual'],
#         'Size': ['LnIssuerCap'],
#         'Liquidity': ['LnTrading_Activity_60D'],
#         'Market Sensitivity': ['Market_Sensitivity_250D'],
#         'Volatility': ['Volatility_125D'],
#         'Medium-Term Momentum': ['Momentum_250x20D'],
#         'Exchange Rate Sensitivity': ['XRate_104W_XDR'],
#         'EM Sensitivity': ['Market_Sensitivity_EM_W_V2','Market_Sensitivity_CN_W_V2'],
#         'Profitability': ['Return_on_Equity_Annual','Return_on_Assets_Annual', 'CashFlow_to_Assets_Annual',
#                           'CashFlow_to_Income_Annual','Sales_to_Assets_Annual', 'Gross_Margin_Annual'],
#         }
# class AU2Yilin_d9(AUAxioma2016MH):
# # investigating the difference caused by ESTU
#     rm_id,revision,rms_id = [-20,23,-137]
#     def generate_estimation_universe(self, modelDate, data, modelDB, marketDB, excludeFactors=None):
#         estu1 = super(AUAxioma2016MH, self).generate_estimation_universe(modelDate, data, modelDB, marketDB, excludeFactors=None)
#         return estu1

# study the value factors
class AU2Yilin_d8(AUAxioma2016MH):
# study the value factors with a different weight
    rm_id,revision,rms_id = [-20,22,-136]
    DescriptorWeights = {'Value': [0.5, 0.25,0.25],
                        'EM Sensitivity': [0.5,0.5]}
class AU2Yilin_d9(AUAxioma2016MH):
# study the value factors with a different weight
    rm_id,revision,rms_id = [-20,23,-137]
    DescriptorWeights = {'Value': [0.5, 0.125,0.375],
                        'EM Sensitivity': [0.5,0.5]}

