import bisect
import copy
from calendar import monthrange
import datetime
import logging
import numpy
import numpy.ma as ma
import pandas
import re
import statsmodels.formula.api as sm
import sys
from marketdb import MarketID
import riskmodels
from riskmodels import AssetProcessor
from riskmodels import AssetProcessor_V4
from riskmodels import Classification
from riskmodels import DescriptorExposures
from riskmodels import DescriptorRatios
from riskmodels import FactorReturns
from riskmodels import Matrices
from riskmodels.Matrices import ExposureMatrix
from riskmodels import ModelDB
from riskmodels.ModelDB import SubIssue
from riskmodels import ProcessReturns
from riskmodels import TimeSeriesRegression
from riskmodels import LegacyTimeSeriesRegression
from riskmodels import Utilities
from riskmodels import Outliers
from riskmodels import CurrencyRisk

class DescriptorClass(object):
    """Base class for all the descriptor transfer class"""
    def __init__(self, connections, gp=None):
        self.descriptorType = 'SCM'
        self.marketDB = connections.marketDB
        self.modelDB = connections.modelDB
        self.mongoDB = connections.mongoDB
        self.log = self.modelDB.log
        self.cleanup = hasattr(gp, 'cleanup') and gp.cleanup
        self.debuggingReporting = gp.verbose
        self.gicsDate = gp.gicsDate
        self.forceRun = gp.override
        self.trackList = gp.trackList
        self.returnHistory = 500
        self.modelDB.setTotalReturnCache(2*self.returnHistory+1)
        self.modelDB.setVolumeCache(2*self.returnHistory+1)
        self.modelDB.setMarketCapCache(90)
        self.simpleProxyRetTol = 0.95
        self.useFixedFrequency = None

    def setNumeraire(self, numeraire):
        self.numeraire_ISO = numeraire
        self.modelDB.createCurrencyCache(self.marketDB, boT=datetime.date(1980,1,1))

    def setNumeraireID(self, numeraire_ISO, date):
        self.numeraire_id = self.marketDB.getCurrencyID(numeraire_ISO, date)

    def setLocalCurrency(self, currencyMap, date):
        self.localCurrencyISO = dict()
        self.localCurrency_id = dict()
        for dt in currencyMap.keys():
            self.localCurrencyISO[dt] = currencyMap[dt]
            self.localCurrency_id[dt] = self.marketDB.getCurrencyID(currencyMap[dt], dt)
        dtList = sorted(currencyMap.keys())
        self.latestDate = dtList[-1]
        self.latestLocalCurrencyISO = self.localCurrencyISO[dtList[-1]]
        self.latestLocalCurrency_id = self.localCurrency_id[dtList[-1]]
        self.returnsClippedNoProxy = None
        self.returnsNoProxy = None
        self.returnsClipped = None
        self.returns = None
        self.returnsClippedNoRT = None
        self.returnsNoRT = None
        self.returnsClippedNumeraire = None
        self.returnsNumeraire = None

    def setDateHistory(self, descriptorType, date):
        if (descriptorType == 'SCM'):
            return
        startDate = date - datetime.timedelta(self.returnHistory-1)
        self.returnDateList = self.modelDB.getDateRange(None, startDate, date, excludeWeekend=True)
        return

    def setReturnHistory(self, returnHistory):
        self.returnHistory = returnHistory
        
    def buildBasicData(self, date, assetData, rmg, allRMGList):
        self.dates = [date]
        self.allRMGList = allRMGList
        self.rmg = rmg
        self.getMarketCaps(assetData)
        self.storedResults = dict()

    def setUpReturnObject(self, subids, data):
        if type(data) is pandas.Series:
            data = Utilities.df2ma(data[subids])
        # Create return value structure
        retvals = numpy.empty((1, len(subids)), dtype=object)
        for (sIdx, sid) in enumerate(subids):
            rval = Utilities.Struct()
            if data[sIdx] is ma.masked:
                rval.value = None
            else:
                rval.value = data[sIdx]
            retvals[0, sIdx] = rval
        return retvals

    def loadReturnsArray(self, assetData, rootClass,
            daysBack=None, adjustForRT=False, clippedReturns=False, applyProxy=True, numeraire=None):
        # Load in variety of asset returns required

        p = Utilities.Struct()
        p.adjustForRT = adjustForRT
        p.clippedReturns = clippedReturns
        p.applyProxy = applyProxy
        if daysBack is None:
            daysBack = rootClass.returnHistory
        p.simpleProxyRetTol = rootClass.simpleProxyRetTol

        # Build relevant set of returns
        if numeraire is not None:
            # Set up base dict if it doesn't exist
            if (not hasattr(rootClass, 'returnsNumeraire')) or (rootClass.returnsNumeraire is None):
                rootClass.returnsNumeraire = dict()
            if (not hasattr(rootClass, 'returnsClippedNumeraire')) or (rootClass.returnsClippedNumeraire is None):
                rootClass.returnsClippedNumeraire = dict()

            # Returns in numeraire currency
            tmpClip = p.clippedReturns

            # Load unclipped returns if necessary
            if numeraire not in rootClass.returnsNumeraire:
                p.clippedReturns = False
                rootClass.returnsNumeraire[numeraire] = rootClass.loadProcessedReturn(p, assetData, convert2Numeraire=numeraire)
            p.clippedReturns = tmpClip

            # Return unclipped returns if required
            if not p.clippedReturns:
                returnsClass = rootClass.returnsNumeraire[numeraire]
                logging.info('Loading returns: rt-adjust: %s, clip: %s, proxy: %s, currency: %s', adjustForRT, clippedReturns, applyProxy, numeraire) 
            else:
                # Otherwise clip returns
                if numeraire not in rootClass.returnsClippedNumeraire:
                    rootClass.returnsClippedNumeraire[numeraire] = copy.deepcopy(rootClass.returnsNumeraire)
                    outlierClass0 = Outliers.Outliers()
                    rootClass.returnsClippedNumeraire[numeraire].data = outlierClass0.twodMAD(rootClass.returnsNumeraire[numeraire].data)
                returnsClass = rootClass.returnsClippedNumeraire[numeraire]
                logging.info('Loading returns: rt-adjust: %s, clip: %s, proxy: %s, currency: %s', adjustForRT, clippedReturns, applyProxy, numeraire)

        elif not p.applyProxy:
            # Returns with no proxy
            tmpClip = p.clippedReturns

            # Load unclipped returns if necessary
            if (not hasattr(rootClass, 'returnsNoProxy')) or (rootClass.returnsNoProxy is None):
                p.clippedReturns = False
                rootClass.returnsNoProxy = rootClass.loadProcessedReturn(p, assetData)
            p.clippedReturns = tmpClip

            # Return unclipped returns if required
            if not p.clippedReturns:
                returnsClass = rootClass.returnsNoProxy
                logging.info('Loading returns: rt-adjust: %s, clip: %s, proxy: %s, currency: %s', adjustForRT, clippedReturns, applyProxy, numeraire)
            else:
                # Otherwise clip returns
                if (not hasattr(rootClass, 'returnsClippedNoProxy')) or (rootClass.returnsClippedNoProxy is None):
                    rootClass.returnsClippedNoProxy = copy.deepcopy(rootClass.returnsNoProxy)
                    outlierClass0 = Outliers.Outliers()
                    rootClass.returnsClippedNoProxy.data = outlierClass0.twodMAD(rootClass.returnsNoProxy.data)
                returnsClass = rootClass.returnsClippedNoProxy
                logging.info('Loading returns: rt-adjust: %s, clip: %s, proxy: %s, currency: %s', adjustForRT, clippedReturns, applyProxy, numeraire)

        elif p.adjustForRT:
            # Returns with proxy and adjustment for returns-timing
            tmpClip = p.clippedReturns

            # Load unclipped returns if necessary
            if (not hasattr(rootClass, 'returns')) or (rootClass.returns is None):
                p.clippedReturns = False
                rootClass.returns = rootClass.loadProcessedReturn(p, assetData)
            p.clippedReturns = tmpClip

            # Return unclipped returns if required
            if not p.clippedReturns:
                returnsClass = rootClass.returns
                logging.info('Loading returns: rt-adjust: %s, clip: %s, proxy: %s, currency: %s', adjustForRT, clippedReturns, applyProxy, numeraire)
            else:
                # Otherwise clip returns
                if (not hasattr(rootClass, 'returnsClipped')) or (rootClass.returnsClipped is None):
                    rootClass.returnsClipped = copy.deepcopy(rootClass.returns)
                    outlierClass1 = Outliers.Outliers()
                    rootClass.returnsClipped.data = outlierClass1.twodMAD(rootClass.returns.data)
                returnsClass = rootClass.returnsClipped
                logging.info('Loading returns: rt-adjust: %s, clip: %s, proxy: %s, currency: %s', adjustForRT, clippedReturns, applyProxy, numeraire)
        else:
            # Returns with no returns-timing adjustment
            tmpClip = p.clippedReturns

            # Load unclipped returns if necessary
            if (not hasattr(rootClass, 'returnsNoRT')) or (rootClass.returnsNoRT is None):
                p.clippedReturns = False
                rootClass.returnsNoRT = rootClass.loadProcessedReturn(p, assetData)
            p.clippedReturns = tmpClip

            # Return unclipped returns if required
            if not p.clippedReturns:
                returnsClass = rootClass.returnsNoRT
                logging.info('Loading returns: rt-adjust: %s, clip: %s, proxy: %s, currency: %s', adjustForRT, clippedReturns, applyProxy, numeraire)
            else:
                # Otherwise clip returns
                if (not hasattr(rootClass, 'returnsClippedNoRT')) or (rootClass.returnsClippedNoRT is None):
                    rootClass.returnsClippedNoRT = copy.deepcopy(rootClass.returnsNoRT)
                    outlierClass2 = Outliers.Outliers()
                    rootClass.returnsClippedNoRT.data = outlierClass2.twodMAD(rootClass.returnsNoRT.data)
                returnsClass = rootClass.returnsClippedNoRT
                logging.info('Loading returns: rt-adjust: %s, clip: %s, proxy: %s, currency: %s', adjustForRT, clippedReturns, applyProxy, numeraire)

        tmpReturns = Matrices.TimeSeriesMatrix(assetData.universe, returnsClass.dates[-daysBack:])
        tmpReturns.data = returnsClass.data[:, -daysBack:]
        tmpReturns.rollOverFlag = returnsClass.rollOverFlag[:, -daysBack:]
        tmpReturns.missingFlag = returnsClass.missingFlag[:, -daysBack:]
        tmpReturns.preIPOFlag = returnsClass.preIPOFlag[:, -daysBack:]
        tmpReturns.ntdFlag = returnsClass.ntdFlag[:, -daysBack:]
        tmpReturns.zeroFlag = returnsClass.zeroFlag[:, -daysBack:]
        return tmpReturns

    def getMarketCaps(self, assetData):
        # Method to build issuerMarketCaps for the given rmg and universe
        # Compute issuer-level caps
        numer = Utilities.Struct()
        numer.currency_id = self.numeraire_id
        marketCaps = pandas.Series(assetData.marketCaps, index=assetData.universe)
        mcapDF = AssetProcessor_V4.computeTotalIssuerMarketCaps(
                self.dates[0], marketCaps, numer, self.modelDB, self.marketDB)
        assetData.issuerTotalMarketCaps = mcapDF.loc[assetData.universe, 'totalCap'].values
        assetData.DLCMarketCap = mcapDF.loc[assetData.universe, 'dlcCap'].values
        assetData.sid2sib = AssetProcessor_V4.getValidSiblings(\
                self.dates[0], assetData.universe, self.modelDB, self.marketDB)
        return

    def loadProcessedReturn(self, params, assetData, convert2Numeraire=None):
        # Wrapper to call returns loader
        retProcessor = ProcessReturns.assetReturnsProcessor(
                [self.rmg], assetData.universe, assetData.rmgAssetMap,
                assetData.tradingRmgAssetMap, assetData.assetTypeDict,
                numeraire_id=self.numeraire_id,
                tradingCurrency_id=self.localCurrency_id[self.latestDate],
                debuggingReporting=self.debuggingReporting, gicsDate=self.gicsDate,
                simpleProxyRetTol=params.simpleProxyRetTol)
        if convert2Numeraire is not None:
            drCurrMap = self.marketDB.getCurrencyID(convert2Numeraire, self.latestDate)
            logging.info('Converting returns to ID %s(%s)', drCurrMap, convert2Numeraire)
        else:
            drCurrMap = assetData.drCurrData

        # Sort out undesirables
        assetTypeMap = pandas.Series(assetData.assetTypeDict)
        noProxyReturnsList = set(assetTypeMap[assetTypeMap.isin(AssetProcessor_V4.noProxyTypes)].index)
        exSpacs = AssetProcessor_V4.sort_spac_assets(\
                self.dates[0], assetData.universe, self.modelDB, self.marketDB, returnExSpac=True)
        noProxyReturnsList = noProxyReturnsList.difference(exSpacs)
        if len(noProxyReturnsList) > 0:
            logging.info('%d assets of type %s excluded from proxying', len(noProxyReturnsList), AssetProcessor_V4.noProxyTypes)

        if hasattr(self, 'returnDateList'):
            returns = retProcessor.process_returns_history(self.returnDateList, None,
                    self.modelDB, self.marketDB, drCurrMap=drCurrMap,
                    loadOnly=(params.applyProxy==False), noProxyList=noProxyReturnsList,
                    applyRT=params.adjustForRT, trimData=params.clippedReturns)
        else:
            returns = retProcessor.process_returns_history(self.dates[0], self.returnHistory,
                    self.modelDB, self.marketDB, drCurrMap=drCurrMap,
                    loadOnly=(params.applyProxy==False), noProxyList=noProxyReturnsList,
                    applyRT=params.adjustForRT, trimData=params.clippedReturns)

        return returns

class NumeraireDescriptorClass(DescriptorClass):
    """Class for descriptors denoted in particular numeraire"""
    def __init__(self, connections, gp=None):
        self.descriptorType = 'numeraire'
        self.marketDB = connections.marketDB
        self.modelDB = connections.modelDB
        self.log = self.modelDB.log
        self.gicsDate = gp.gicsDate
        self.cleanup = hasattr(gp, 'cleanup') and gp.cleanup
        self.debuggingReporting = gp.verbose
        self.historyLength = 500
        self.trackList = gp.trackList
        self.modelDB.setVolumeCache(self.historyLength+1)
        self.modelDB.setMarketCapCache(90)
        self.returnHistory = 2*365 + 60
        self.modelDB.setTotalReturnCache(self.returnHistory)
        self.simpleProxyRetTol = 0.8
        self.useFixedFrequency = None
        if hasattr(gp, 'useFixedFrequency'):
            if gp.useFixedFrequency is None:
                self.useFixedFrequency = None
            elif gp.useFixedFrequency.lower() in ['annual', '_ann', 'ann']:
                self.useFixedFrequency = DescriptorRatios.DataFrequency('_ann')
            elif gp.useFixedFrequency.lower() in ['quarter', 'quarterly', 'qtr', '_qtr']:
                self.useFixedFrequency = DescriptorRatios.DataFrequency('_qtr')
            else:
                self.useFixedFrequency = None

class LocalDescriptorClass(DescriptorClass):
    """Class for descriptors denoted in local/trading currency"""
    def __init__(self, connections, gp=None):
        self.descriptorType = 'local'
        self.marketDB = connections.marketDB
        self.modelDB = connections.modelDB
        self.log = self.modelDB.log
        self.gicsDate = gp.gicsDate
        self.cleanup = hasattr(gp, 'cleanup') and gp.cleanup
        self.debuggingReporting = gp.verbose
        self.returnHistory = 2*365 + 60 
        self.trackList = gp.trackList
        self.modelDB.setTotalReturnCache(2*self.returnHistory+1)
        self.modelDB.setVolumeCache(2*self.returnHistory+1)
        self.modelDB.setMarketCapCache(90)
        self.simpleProxyRetTol = 0.8
        self.useFixedFrequency = None

# Momentum classes
class Momentum(DescriptorClass):
    """Class to compute returns momentum. 
    """
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)
        self.INCR = 200
        self.fromT = 250
        self.thruT = 20
        self.adjustForRT = False
        self.clippedReturns = False
        self.weights = None
        self.peak = 0
        self.peak2 = None
        self.useLogReturns = False
        
    def buildDescriptor(self, data, rootClass):
        self.log.debug('Momentum.buildDescriptor')
        # Build descriptor
        tmpReturns = self.loadReturnsArray(data,
                rootClass, adjustForRT=self.adjustForRT, clippedReturns=self.clippedReturns)
        mtm = DescriptorExposures.generate_momentum(
                tmpReturns, self.fromT, self.thruT, weights=self.weights,
                peak=self.peak, peak2=self.peak2, useLogRets=self.useLogReturns)

        # Create return value structure
        return self.setUpReturnObject(data.universe, mtm)

class Momentum_20D(Momentum):
    """Class to compute short-term momentum. 
    """
    def __init__(self, connections, gp=None):
        Momentum.__init__(self, connections, gp=gp)
        self.fromT = 20
        self.thruT = 0

class Momentum_21D(Momentum):
    """Class to compute short-term momentum.
    """
    def __init__(self, connections, gp=None):
        Momentum.__init__(self, connections, gp=gp)
        self.fromT = 21
        self.thruT = 0

class Momentum_250x20D(Momentum):
    """Class to compute medium-term momentum.
    """
    def __init__(self, connections, gp=None):
        Momentum.__init__(self, connections, gp=gp)
        self.fromT = 250
        self.thruT = 10
        self.weights = 'pyramid'
        self.peak = 10
        self.peak2 = 20

class Momentum_260x21D_Regional(Momentum):
    """Class to compute medium-term momentum.
    for regional models
    """
    def __init__(self, connections, gp=None):
        Momentum.__init__(self, connections, gp=gp)
        self.fromT = 260
        self.thruT = 11
        self.weights = 'pyramid'
        self.peak = 10
        self.peak2 = 20

# Linked asset score classes
class ISC_ADV_Score(DescriptorClass):
    """Class to compute ADV component for linked asset scores
    """
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)
        self.INCR = 200

    def buildDescriptor(self, data, rootClass):
        self.log.debug('ISC_ADV_Score.buildDescriptor')
        score = DescriptorExposures.generate_linked_asset_ADV_score(
                rootClass.dates[0], data.universe, rootClass.numeraire_id, self.modelDB)
        return self.setUpReturnObject(data.universe, score)

class ISC_Ret_Score(DescriptorClass):
    """Class to compute % of non-missing returns component for linked asset scores
    """
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)
        self.INCR = 200
        self.adjustForRT = False
        self.clippedReturns = False
        self.applyProxy = False
        self.checkForZeros = False
        self.daysBack = 260
        self.minDays = 22
        self.countPreIPODates = False

    def buildDescriptor(self, data, rootClass):
        self.log.debug('ISC_Ret_Score.buildDescriptor')

        tmpReturns = self.loadReturnsArray(data, rootClass, adjustForRT=self.adjustForRT,
                                clippedReturns=self.clippedReturns, applyProxy=self.applyProxy)
        score = DescriptorExposures.generate_linked_asset_ret_score(
                tmpReturns, self.daysBack, self.minDays, self.checkForZeros,
                countPreIPO=self.countPreIPODates)
        return self.setUpReturnObject(data.universe, score)

class ISC_Zero_Score(ISC_Ret_Score):
    def __init__(self, connections, gp=None):
        ISC_Ret_Score.__init__(self, connections, gp=gp)
        self.checkForZeros = True

class ISC_IPO_Score(DescriptorClass):
    """Class to compute IPO age component for linked asset scores
    """
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)
        self.INCR = 200

    def buildDescriptor(self, data, rootClass):
        self.log.debug('ISC_IPO_Score.buildDescriptor')
        score = DescriptorExposures.generate_linked_asset_IPO_score(
                rootClass.dates[0], data.universe, self.modelDB, self.marketDB)
        return self.setUpReturnObject(data.universe, score)

class ISC_Score_Legacy(DescriptorClass):
    """Class to compute legacy ISC scores
    """
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)
        self.INCR = 200

    def buildDescriptor(self, data, rootClass):
        self.log.debug('ISC_Score_Legacy.buildDescriptor')
        # Build descriptor
        score = DescriptorExposures.generate_linked_asset_scores_legacy(
                rootClass.dates[0], [rootClass.rmg], data.universe, self.modelDB, self.marketDB,
                data.subIssueGroups, rootClass.numeraire_id)
        return self.setUpReturnObject(data.universe, score)

# Volatility classes
class PACE_Volatility(DescriptorClass):
    """Class to compute asset level PACE volatility
    """
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)
        self.INCR = 200 
        self.daysBack = 125
        self.adjustForRT = False
        self.clippedReturns = True
        self.weights = None
        self.peak = 0
        self.peak2 = 0
        self.rmg = None # specify for scm descriptors that use specialized rmg
        self.regionID = None

    def buildDescriptor(self, data, rootClass):
        self.log.debug('PACE_Volatility.buildDescriptor')
        tmpReturns = self.loadReturnsArray(
                data, rootClass, daysBack=self.daysBack,
                adjustForRT=self.adjustForRT, clippedReturns=self.clippedReturns)

        if self.weights is None:
            weights = numpy.ones((tmpReturns.data.shape[1]), float)
        else:
            weights = Utilities.computePyramidWeights(self.peak, self.peak2, tmpReturns.data.shape[1])

        # Get list of live RMGs
        if rootClass.descriptorType == 'local':
            if self.rmg == 'RMG':
                allRMGs = [rootClass.rmg]
            elif self.rmg is not None:
                allRMGs = self.modelDB.getAllRiskModelGroups(False)
                spRMG = self.modelDB.getRiskModelGroup(self.rmg)
                assert(spRMG in allRMGs)
                allRMGs = [spRMG]
            else:
                allRMGs = self.modelDB.getAllRiskModelGroups()
            if self.regionID is not None:
                regionRmgIDs = self.modelDB.getRMGIdsForRegion(self.regionID, rootClass.dates[0])
                allRMGs = [rmg for rmg in allRMGs if rmg.rmg_id in regionRmgIDs]
            logging.info('there are %d RMGs to use.' % len(allRMGs))
            # Loop round RMGs and get market portfolio length
            marketLength = []
            for rmg in allRMGs:
                amp = self.modelDB.convertLMP2AMP(rmg,  rootClass.dates[0])
                market = self.modelDB.getRMGMarketPortfolio(rmg,  rootClass.dates[0], amp=amp)
                marketLength.append(float(len(market)))

            # Load in CSV for each RMG
            rmgVols = self.modelDB.loadRMGMarketVolatilityHistory(
                    tmpReturns.dates, allRMGs, rollOver=30)
        
            # Compute average weighted CSV
            rmgVols = ma.average(rmgVols.data*rmgVols.data, axis=0, weights=marketLength)
            rmgVols = ma.sqrt(rmgVols)
        else:
            if self.rmg is not None:
                allRMGs = self.modelDB.getAllRiskModelGroups(inModels=False)
                spRMG = self.modelDB.getRiskModelGroup(self.rmg)
                assert(spRMG in allRMGs)
                rmgVols = self.modelDB.loadRMGMarketVolatilityHistory(
                        tmpReturns.dates, [spRMG], rollOver=30)
            else:
                rmgVols = self.modelDB.loadRMGMarketVolatilityHistory(
                        tmpReturns.dates, [rootClass.rmg], rollOver=30)
            rmgVols = rmgVols.data[0, :]

        missingCSV = numpy.flatnonzero(ma.getmaskarray(rmgVols))
        if len(missingCSV) == len(rmgVols):
            logging.warning('No CSV data for %s: %s to %s',
                    rootClass.rmg.mnemonic, tmpReturns.dates[0], tmpReturns.dates[-1])
            rmgVols = None

        # Compute PACE volatility
        values = DescriptorExposures.generate_pace_volatility(
                tmpReturns.data, data.universe, self.daysBack, csvHistory=rmgVols, weights=weights, trackList=rootClass.trackList)

        return self.setUpReturnObject(data.universe, values)

class Volatility_125D(PACE_Volatility):
    """Class to compute asset level MH PACE volatility
    """
    def __init__(self, connections, gp=None):
        PACE_Volatility.__init__(self, connections, gp=gp)
        self.weights = 'pyramid'
        self.peak = 10
        self.peak2 = 10

class Volatility_USSC_125D(PACE_Volatility):
    """Class to compute asset level MH PACE volatility
    """
    def __init__(self, connections, gp=None):
        PACE_Volatility.__init__(self, connections, gp=gp)
        self.weights = 'pyramid'
        self.peak = 10
        self.peak2 = 10
        self.rmg = -3 # rmg for US small cap portfolio

class Volatility_CN_60D(PACE_Volatility):
    """Class to compute asset level MH PACE volatility used for CN4
    """
    def __init__(self, connections, gp=None):
        PACE_Volatility.__init__(self, connections, gp=gp)
        self.daysBack = 60
        self.weights = 'pyramid'
        self.peak = 5
        self.peak2 = 5
        self.rmg = -2 # rmg for Domestic CN

class Volatility_CN_125D(PACE_Volatility):
    """Class to compute asset level MH PACE volatility used for CN4
    """
    def __init__(self, connections, gp=None):
        PACE_Volatility.__init__(self, connections, gp=gp)
        self.daysBack = 125
        self.weights = 'pyramid'
        self.peak = 10
        self.peak2 = 10
        self.rmg = -2 # rmg for Domestic CN

class Volatility_60D(PACE_Volatility):
    """Class to compute asset level SH PACE volatility
    """
    def __init__(self, connections, gp=None):
        PACE_Volatility.__init__(self, connections, gp=gp)
        self.daysBack = 60
        self.weights = 'pyramid'
        self.peak = 5
        self.peak2 = 5

class APAC_Volatility_125D(PACE_Volatility):
    """Class to compute asset level MH PACE volatility for APAC region
    """
    def __init__(self, connections, gp=None):
        PACE_Volatility.__init__(self, connections, gp=gp)
        self.weights = 'pyramid'
        self.peak = 10
        self.peak2 = 10
        self.regionID = 106

class APAC_Volatility_60D(PACE_Volatility):
    """Class to compute asset level SH PACE volatility for APAC region
    """
    def __init__(self, connections, gp=None):
        PACE_Volatility.__init__(self, connections, gp=gp)
        self.daysBack = 60
        self.weights = 'pyramid'
        self.peak = 5
        self.peak2 = 5
        self.regionID = 106

class EM_Volatility_125D(PACE_Volatility):
    """Class to compute asset level MH PACE volatility for EM region
    """
    def __init__(self, connections, gp=None):
        PACE_Volatility.__init__(self, connections, gp=gp)
        self.weights = 'pyramid'
        self.peak = 10
        self.peak2 = 10
        self.regionID = 107

class EM_Volatility_60D(PACE_Volatility):
    """Class to compute asset level SH PACE volatility for EM region
    """
    def __init__(self, connections, gp=None):
        PACE_Volatility.__init__(self, connections, gp=gp)
        self.daysBack = 60
        self.weights = 'pyramid'
        self.peak = 5
        self.peak2 = 5
        self.regionID = 107

class Europe_Volatility_125D(PACE_Volatility):
    """Class to compute asset level MH PACE volatility for Europe region
    """
    def __init__(self, connections, gp=None):
        PACE_Volatility.__init__(self, connections, gp=gp)
        self.weights = 'pyramid'
        self.peak = 10
        self.peak2 = 10
        self.regionID = 100

class Europe_Volatility_60D(PACE_Volatility):
    """Class to compute asset level SH PACE volatility for Europe region
    """
    def __init__(self, connections, gp=None):
        PACE_Volatility.__init__(self, connections, gp=gp)
        self.daysBack = 60
        self.weights = 'pyramid'
        self.peak = 5
        self.peak2 = 5
        self.regionID = 100

class UK_Volatility_125D(PACE_Volatility):
    """Class to compute asset level MH PACE volatility for UK market
    """
    def __init__(self, connections, gp=None):
        PACE_Volatility.__init__(self, connections, gp=gp)
        self.weights = 'pyramid'
        self.peak = 10
        self.peak2 = 10
        self.rmg = 2

class UK_Volatility_60D(PACE_Volatility):
    """Class to compute asset level SH PACE volatility for UK market
    """
    def __init__(self, connections, gp=None):
        PACE_Volatility.__init__(self, connections, gp=gp)
        self.daysBack = 60
        self.weights = 'pyramid'
        self.peak = 5
        self.peak2 = 5
        self.rmg = 2

class RMG_Volatility_125D(PACE_Volatility):
    """Class to compute asset level MH PACE volatility for UK market
    """
    def __init__(self, connections, gp=None):
        PACE_Volatility.__init__(self, connections, gp=gp)
        self.weights = 'pyramid'
        self.peak = 10
        self.peak2 = 10
        self.rmg = 'RMG'

class RMG_Volatility_60D(PACE_Volatility):
    """Class to compute asset level SH PACE volatility for UK market
    """
    def __init__(self, connections, gp=None):
        PACE_Volatility.__init__(self, connections, gp=gp)
        self.daysBack = 60
        self.weights = 'pyramid'
        self.peak = 5
        self.peak2 = 5
        self.rmg = 'RMG'

class CA_Volatility_125D(PACE_Volatility):
    """Class to compute asset level MH PACE volatility for CA market
    """
    def __init__(self, connections, gp=None):
        PACE_Volatility.__init__(self, connections, gp=gp)
        self.weights = 'pyramid'
        self.peak = 10
        self.peak2 = 10
        self.rmg = 10

class CA_Volatility_60D(PACE_Volatility):
    """Class to compute asset level SH PACE volatility for CA market
    """
    def __init__(self, connections, gp=None):
        PACE_Volatility.__init__(self, connections, gp=gp)
        self.daysBack = 60
        self.weights = 'pyramid'
        self.peak = 5
        self.peak2 = 5
        self.rmg = 10

class NA_Volatility_125D(PACE_Volatility):
    """Class to compute asset level MH PACE volatility for NA region
    """
    def __init__(self, connections, gp=None):
        PACE_Volatility.__init__(self, connections, gp=gp)
        self.weights = 'pyramid'
        self.peak = 10
        self.peak2 = 10
        self.regionID = 108

class NA_Volatility_60D(PACE_Volatility):
    """Class to compute asset level SH PACE volatility for NA market
    """
    def __init__(self, connections, gp=None):
        PACE_Volatility.__init__(self, connections, gp=gp)
        self.daysBack = 60
        self.weights = 'pyramid'
        self.peak = 5
        self.peak2 = 5
        self.regionID = 108

class DMxUS_Volatility_125D(PACE_Volatility):
    """Class to compute asset level MH PACE volatility for DMxUS region
    """
    def __init__(self, connections, gp=None):
        PACE_Volatility.__init__(self, connections, gp=gp)
        self.weights = 'pyramid'
        self.peak = 10
        self.peak2 = 10
        self.regionID = 109

class DMxUS_Volatility_60D(PACE_Volatility):
    """Class to compute asset level SH PACE volatility for DMxUS market
    """
    def __init__(self, connections, gp=None):
        PACE_Volatility.__init__(self, connections, gp=gp)
        self.daysBack = 60
        self.weights = 'pyramid'
        self.peak = 5
        self.peak2 = 5
        self.regionID = 109

class Volatility_USSC_60D(PACE_Volatility):
    """Class to compute asset level MH PACE volatility
    """
    def __init__(self, connections, gp=None):
        PACE_Volatility.__init__(self, connections, gp=gp)
        self.daysBack = 60
        self.weights = 'pyramid'
        self.peak = 5
        self.peak2 = 5
        self.rmg = -3 # rmg for US small cap portfolio

class Volatility_250D(PACE_Volatility):
    """Class to compute asset level LH PACE volatility
    """
    def __init__(self, connections, gp=None):
        PACE_Volatility.__init__(self, connections, gp=gp)
        self.daysBack = 250
        self.weights = 'pyramid'
        self.peak = 21
        self.peak2 = 21

# Various "beta" descriptors
class Market_Sensitivity_Legacy(DescriptorClass):
    """Class to compute market sensitivity
    Superceded by Regional_Market_Sensitivity class
    Should only be used by US4 family of models now
    """
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)
        self.INCR = 200
        self.daysBack = 250
        self.adjustForRT = False
        self.clippedReturns = True
        self.robust = False
        self.kappa = 5.0
        self.swAdj = True
        self.lag = None
        self.weighting = 'pyramid'
        self.halflife = 21
        self.fadePeak = 21
        self.frequency = 'daily'
        self.fillWithMarket = True
        self.outputDateList = None
        self.marketReturns = None
        self.regionPortfolioID = None
        self.modelPortfolioName = None # specify to use returns
                                       # from index team
        self.rmg = None                # specify to use returns 
        self.useRobustReturn = False   # for specialized rmg
        self.readFromList = []

        # Parameters for (optionally) saving regression stats
        self.getRegStats = False
        self.collName = 'TS_RegressStats'

    def saveRegressStats(self, modelDate, rmg_id, data):
        coll = self.mongoDB[self.collName]

        # upsert regress stats
        modelDate = datetime.datetime(modelDate.year,
                                      modelDate.month,
                                      modelDate.day)
        baseDict = {'rmg_id': rmg_id,
                    'dt': modelDate,
                    'regression': self.name}
        dataDict = {}
        dataDict['sub_issue_ids'] = data.subids
        dataDict['tstats'] = data.tstatDict
        dataDict['pvals'] = data.pvalDict
        dataDict['rsquare'] = data.rsq
        #dataDict['vifs'] = data.vifs
        dataDict['dof'] = data.dof 
        try:
            res = coll.update_one(baseDict, {'$set': dataDict}, upsert=True)
        except:
            self.log.exception('Unexpected error: %s' % sys.exc_info()[0])

    def buildDescriptor(self, data, rootClass):
        self.log.debug('MarketSensitivity.buildDescriptor')

        # Load the returns used
        tmpReturns = self.loadReturnsArray(
                data, rootClass, daysBack=self.daysBack,
                adjustForRT=self.adjustForRT, clippedReturns=self.clippedReturns)

        # Determine region if relevant
        if self.regionPortfolioID is not None:
            region = modelDB.getRiskModelRegion(self.regionPortfolioID)
        else:
            region = None

        # Get market returns if relevant
        if self.modelPortfolioName is not None:
            portInfo = self.modelDB.getModelPortfolioByShortName(
                    self.modelPortfolioName, 
                    max(tmpReturns.dates))
            dates = self.modelDB.getDateRange(rootClass.rmg, 
                    min(tmpReturns.dates), max(tmpReturns.dates), 
                    excludeWeekend=False)
            self.marketReturns = self.modelDB.loadModelPortfolioReturns(dates,
                    [portInfo]) 
        elif self.rmg is not None:
            allRMGs = self.modelDB.getAllRiskModelGroups(inModels=False)
            spRMG = self.modelDB.getRiskModelGroup(self.rmg)
            assert(spRMG in allRMGs)
            dates = self.modelDB.getDateRange(rootClass.rmg, 
                    min(tmpReturns.dates), max(tmpReturns.dates), 
                    excludeWeekend=False)
            self.marketReturns = self.modelDB.loadRMGMarketReturnHistory(dates, [spRMG], robust=self.useRobustReturn)

        # Check to see whether we've already run the relevant regressions
        found = False
        for item in self.readFromList:
            if item in rootClass.storedResults.keys():
                mmDict = rootClass.storedResults[item]
                found = True
                continue
        if not found:
            mmDict = dict()

        # Set up regression parameters
        params = dict()
        params['swAdj'] = self.swAdj
        params['lag'] = self.lag
        params['robust'] = self.robust
        params['kappa'] = self.kappa
        params['weighting'] = self.weighting
        params['halflife'] = self.halflife
        params['fadePeak'] = self.fadePeak
        params['historyLength'] = self.daysBack

        # Get dates
        if self.frequency != 'daily':
            self.outputDateList = Utilities.change_date_frequency(tmpReturns.dates, frequency='weekly')

        TSR = LegacyTimeSeriesRegression.LegacyTimeSeriesRegression(
                TSRParameters = params,
                fillWithMarket = self.fillWithMarket,
                outputDateList = self.outputDateList,
                debugOutput = self.debuggingReporting,
                marketReturns = self.marketReturns,
                forceRun = self.forceRun,
                marketRegion = region,
                getRegStats = self.getRegStats)

        # Do the regression
        mm = TSR.TSR(rootClass.rmg, tmpReturns, self.modelDB, self.marketDB, swFix=True)
        mmDict[rootClass.rmg] = mm

        # Store regression statistics
        if self.getRegStats:
            res = Utilities.Struct()
            res.subids = [s.getSubIdString() for s in data.universe]
            tstatDict = {}
            pvalDict = {}
            if len(mm.tstat.shape) == 1:
                mm.tstat = numpy.reshape(mm.tstat, (1, len(mm.tstat)))
                mm.pvals = numpy.reshape(mm.pvals, (1, len(mm.pvals)))
            for f_n, factor in enumerate(mm.factors):
                tstatDict[factor] = list(mm.tstat[f_n, :])
                pvalDict[factor] = list(mm.pvals[f_n, :])
            res.tstatDict = tstatDict
            res.pvalDict = pvalDict
            res.rsq = [val for val in mm.rsquare]
            #res.vifs = mm.vifs
            res.dof = mm.dof
            self.saveRegressStats(max(tmpReturns.dates), rootClass.rmg.rmg_id, res)

        rootClass.storedResults[self.name] = mmDict
        return self.setUpReturnObject(data.universe, mm.beta)

class Market_Sensitivity_USSC_250D(Market_Sensitivity_Legacy):
    """Class to compute MH market sensitivity
    """
    def __init__(self, connections, gp=None):
        Market_Sensitivity_Legacy.__init__(self, connections, gp=gp)
        self.name = 'Market_Sensitivity_USSC_250D'
        #self.modelPortfolioName = 'US-S' # populate marketReturns with 
                                          # portfolio returns from index team
        self.rmg = -3 # used small cap rmg returns
        self.useRobustReturn = True
        self.readFromList = []
        # Parameters for (optionally) saving regression stats
        self.getRegStats = False
        self.collName = 'TS_RegressStats'

class Market_Sensitivity_250D(Market_Sensitivity_Legacy):
    """Class to compute MH market sensitivity
    """
    def __init__(self, connections, gp=None):
        Market_Sensitivity_Legacy.__init__(self, connections, gp=gp)
        self.name = 'Market_Sensitivity_250D'

class Market_Sensitivity_USSC_125D(Market_Sensitivity_Legacy):
    """Class to compute MH market sensitivity
    """
    def __init__(self, connections, gp=None):
        Market_Sensitivity_Legacy.__init__(self, connections, gp=gp)
        self.name = 'Market_Sensitivity_USSC_125D'
        #self.modelPortfolioName = 'US-S' # populate marketReturns with 
                                          # portfolio returns from index team
        self.rmg = -3 # used small cap rmg returns
        self.useRobustReturn = True
        self.daysBack = 125
        self.halflife = 10
        self.fadePeak = 10
        self.readFromList = []
        # Parameters for (optionally) saving regression stats
        self.getRegStats = False
        self.collName = 'TS_RegressStats'

class Market_Sensitivity_125D(Market_Sensitivity_Legacy):
    """Class to compute SH market sensitivity
    """
    def __init__(self, connections, gp=None):
        Market_Sensitivity_Legacy.__init__(self, connections, gp=gp)
        self.daysBack = 125
        self.halflife = 10
        self.fadePeak = 10
        self.name = 'Market_Sensitivity_125D'

class Regional_Market_Sensitivity(DescriptorClass):
    """Class to compute market sensitivity for regional models
    Should now be considered as the template for basic market sensitivity for all models
    """
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)
        self.INCR = 200

        # Returns-loading parameters
        self.historyInYears = 2
        self.adjustForRT = False
        self.clippedReturns = False
        self.returnsCurrency = 'USD'

        # Regression parameters
        self.robust = False
        self.kappa = 5.0
        self.lag = None
        self.inclIntercept = True
        self.getRegStats = False

        # Returns-processing parameters
        self.weighting = 'pyramid'
        self.frequency = 'weekly'
        self.fillWithMarket = True
        self.fillInputsWithZeros = True
        self.minnobs = None
        self.fuzzyScale = None
        self.lagReturns_flag = False

        # Identification parameters
        self.readFromList = []
        self.name = 'Regional_Market_Sensitivity'

    def buildDescriptor(self, data, rootClass):
        self.log.debug('Regional_Market_Sensitivity.buildDescriptor')

        # Load returns
        daysBack = int(self.historyInYears*260)
        tmpReturns = self.loadReturnsArray(
                data, rootClass, daysBack=daysBack,
                adjustForRT=self.adjustForRT, clippedReturns=self.clippedReturns,
                numeraire=self.returnsCurrency)

        # Remove most recent return if required
        if self.lagReturns_flag and (rootClass.dates[0] in tmpReturns.dates):
            self.log.info('Dropping %s from returns series', tmpReturns.dates[-1])
            tmpReturns.dates = tmpReturns.dates[:-1]
            tmpReturns.data = tmpReturns.data[:, :-1]

        # Check to see whether we've already run the relevant regressions
        found = False
        for item in self.readFromList:
            if item in rootClass.storedResults.keys():
                mmDict = rootClass.storedResults[item]
                found = True
                continue
        if not found:
            mmDict = dict()

        # Set up regression parameters
        if self.frequency == 'weekly':
            self.periodNobs = int(self.historyInYears*52)
            self.halflife = 4
            self.fadePeak = 4
        elif self.frequency == 'daily':
            self.periodNobs = int(self.historyInYears*260)
            self.halflife = 21
            self.fadePeak = 21
        elif self.frequency == 'monthly':
            self.periodNobs = int(self.historyInYears*12)
            self.halflife = 2
            self.fadePeak = 2

        params = dict()
        params['lag'] = self.lag
        params['robust'] = self.robust
        params['kappa'] = self.kappa
        params['weighting'] = self.weighting
        params['halflife'] = self.halflife
        params['fadePeak'] = self.fadePeak
        params['frequency'] = self.frequency
        params['nobs'] = self.periodNobs
        params['fuzzyScale'] = self.fuzzyScale
        params['minnobs'] = self.minnobs
        params['inclIntercept'] = self.inclIntercept

        # Initialise time-series regression class
        TSR = TimeSeriesRegression.TimeSeriesRegression(
                self.modelDB, self.marketDB, TSRParameters=params,
                fillWithMarket=self.fillWithMarket,
                debugOutput=self.debuggingReporting,
                fillInputsWithZeros=self.fillInputsWithZeros,
                localRMG=rootClass.rmg, localSector=None, getRegStats=self.getRegStats)

        # Do the regression
        if found:
            mm = mmDict[rootClass.rmg]
        else:
            mm = TSR.TSR(tmpReturns, self.regInputs)
            mmDict[rootClass.rmg] = mm

        rootClass.storedResults[self.name] = mmDict
        retName = self.regInputs[0]
        return self.setUpReturnObject(data.universe, mm.params[mm.factors.index(retName), :])

class Regional_Market_Sensitivity_250D(Regional_Market_Sensitivity):
    """Class to compute SH regional market sensitivity
    """
    def __init__(self, connections, gp=None):
        Regional_Market_Sensitivity.__init__(self, connections, gp=gp)
        self.historyInYears = 1
        self.regInputs = [('Region', 104)]
        self.name = 'Regional_Market_Sensitivity_250D'

class Regional_Market_Sensitivity_500D(Regional_Market_Sensitivity):
    """Class to compute MH regional market sensitivity
    """
    def __init__(self, connections, gp=None):
        Regional_Market_Sensitivity.__init__(self, connections, gp=gp)
        self.historyInYears = 2
        self.regInputs = [('Region', 104)]
        self.name = 'Regional_Market_Sensitivity_500D'

class Europe_Regional_Market_Sensitivity_250D(Regional_Market_Sensitivity):
    """Class to compute SH regional market sensitivity
    """
    def __init__(self, connections, gp=None):
        Regional_Market_Sensitivity.__init__(self, connections, gp=gp)
        self.historyInYears = 1
        self.regInputs = [('Region', 100)]
        self.name = 'Europe_Regional_Market_Sensitivity_250D'
        self.returnsCurrency = 'EUR'

class Europe_Regional_Market_Sensitivity_500D(Regional_Market_Sensitivity):
    """Class to compute MH regional market sensitivity
    """
    def __init__(self, connections, gp=None):
        Regional_Market_Sensitivity.__init__(self, connections, gp=gp)
        self.historyInYears = 2
        self.regInputs = [('Region', 100)]
        self.name = 'Europe_Regional_Market_Sensitivity_500D'
        self.returnsCurrency = 'EUR'

class APAC_Regional_Market_Sensitivity_250D(Regional_Market_Sensitivity):
    """Class to compute SH regional market sensitivity
    """
    def __init__(self, connections, gp=None):
        Regional_Market_Sensitivity.__init__(self, connections, gp=gp)
        self.historyInYears = 1
        self.regInputs = [('Region', 106)]
        self.name = 'APAC_Regional_Market_Sensitivity_250D'

class APAC_Regional_Market_Sensitivity_500D(Regional_Market_Sensitivity):
    """Class to compute MH regional market sensitivity
    """
    def __init__(self, connections, gp=None):
        Regional_Market_Sensitivity.__init__(self, connections, gp=gp)
        self.historyInYears = 2
        self.regInputs = [('Region', 106)]
        self.name = 'APAC_Regional_Market_Sensitivity_500D'

class EM_Regional_Market_Sensitivity_250D(Regional_Market_Sensitivity):
    """Class to compute SH regional market sensitivity - EM Region
    """
    def __init__(self, connections, gp=None):
        Regional_Market_Sensitivity.__init__(self, connections, gp=gp)
        self.historyInYears = 1
        self.regInputs = [('Region', 107)]
        self.name = 'EM_Regional_Market_Sensitivity_250D'

class EM_Regional_Market_Sensitivity_500D(Regional_Market_Sensitivity):
    """Class to compute MH regional market sensitivity - EM Region
    """
    def __init__(self, connections, gp=None):
        Regional_Market_Sensitivity.__init__(self, connections, gp=gp)
        self.historyInYears = 2
        self.regInputs = [('Region', 107)]
        self.name = 'EM_Regional_Market_Sensitivity_500D'

class NA_Regional_Market_Sensitivity_250D(Regional_Market_Sensitivity):
    """Class to compute SH regional market sensitivity - NA
    """
    def __init__(self, connections, gp=None):
        Regional_Market_Sensitivity.__init__(self, connections, gp=gp)
        self.historyInYears = 1
        self.regInputs = [('Region', 108)]
        self.name = 'NA_Regional_Market_Sensitivity_250D'
        
class NA_Regional_Market_Sensitivity_500D(Regional_Market_Sensitivity):
    """Class to compute MH regional market sensitivity - NA
    """
    def __init__(self, connections, gp=None):
        Regional_Market_Sensitivity.__init__(self, connections, gp=gp)
        self.historyInYears = 2
        self.regInputs = [('Region', 108)]
        self.name = 'NA_Regional_Market_Sensitivity_500D'

class DMxUS_Regional_Market_Sensitivity_250D(Regional_Market_Sensitivity):
    """Class to compute SH regional market sensitivity - DMxUS
    """
    def __init__(self, connections, gp=None):
        Regional_Market_Sensitivity.__init__(self, connections, gp=gp)
        self.historyInYears = 1
        self.regInputs = [('Region', 109)]
        self.name = 'DMxUS_Regional_Market_Sensitivity_250D'

class DMxUS_Regional_Market_Sensitivity_500D(Regional_Market_Sensitivity):
    """Class to compute MH regional market sensitivity - DMxUS
    """
    def __init__(self, connections, gp=None):
        Regional_Market_Sensitivity.__init__(self, connections, gp=gp)
        self.historyInYears = 2
        self.regInputs = [('Region', 109)]
        self.name = 'DMxUS_Regional_Market_Sensitivity_500D'

class Market_Sensitivity_104W(Regional_Market_Sensitivity):
    """Class to compute 2 year weekly beta
    """
    def __init__(self, connections, gp=None):
        Regional_Market_Sensitivity.__init__(self, connections, gp=gp)
        self.historyInYears = 2
        self.regInputs = [('Market', 'Local')]
        self.name = 'Market_Sensitivity_104W'
        self.returnsCurrency = None
        self.clippedReturns = True

class Market_Sensitivity_52W(Regional_Market_Sensitivity):
    """Class to compute 1 year weekly beta
    """
    def __init__(self, connections, gp=None):
        Regional_Market_Sensitivity.__init__(self, connections, gp=gp)
        self.historyInYears = 1
        self.regInputs = [('Market', 'Local')]
        self.name = 'Market_Sensitivity_52W'
        self.returnsCurrency = None
        self.clippedReturns = True

class Market_Sensitivity_XC_104W(Regional_Market_Sensitivity):
    """Class to compute 2 year weekly CN beta
    """
    def __init__(self, connections, gp=None):
        Regional_Market_Sensitivity.__init__(self, connections, gp=gp)
        self.historyInYears = 2
        self.regInputs = [('Market', -2)] # Domestic China
        self.name = 'Market_Sensitivity_XC_104W'
        self.returnsCurrency = None
        self.clippedReturns = True

class Market_Sensitivity_XC_52W(Regional_Market_Sensitivity):
    """Class to compute 1 year weekly CN beta
    """
    def __init__(self, connections, gp=None):
        Regional_Market_Sensitivity.__init__(self, connections, gp=gp)
        self.historyInYears = 1
        self.regInputs = [('Market', -2)] # Domestic China
        self.name = 'Market_Sensitivity_XC_52W'
        self.returnsCurrency = None
        self.clippedReturns = True

class Market_Sensitivity_EM_104W(Regional_Market_Sensitivity):
    '''
        Used for AU4 model
    '''
    def __init__(self, connections,gp=None):
        Regional_Market_Sensitivity.__init__(self, connections,gp=gp)
        self.historyInYears = 2
        self.regInputs = [('Region', 103), ('Market', 'Local')]
        self.lagReturns_flag  = True                # EM return data is not ready when generating AU descriptor
        self.returnsCurrency = None
        self.clippedReturns = True
        self.fuzzyScale = 0.94              # AU descriptors use DescriptorClass which doesn't load enough returns

class Market_Sensitivity_CN_104W(Regional_Market_Sensitivity):
    '''
        Used for AU4 model
    '''
    def __init__(self, connections,gp=None):
        Regional_Market_Sensitivity.__init__(self, connections,gp=gp)
        self.historyInYears = 2
        self.regInputs = [('Market', -2), ('Market', 'Local')]
        self.returnsCurrency = None
        self.clippedReturns = True
        self.fuzzyScale = 0.94              # AU descriptors use DescriptorClass which doesn't load enough returns

class Market_Sensitivity_EM_52W(Regional_Market_Sensitivity):
    '''
        Used for AU4 model
    '''
    def __init__(self, connections,gp=None):
        Regional_Market_Sensitivity.__init__(self, connections,gp=gp)
        self.historyInYears = 1
        self.regInputs = [('Region', 103), ('Market', 'Local')]
        self.lagReturns_flag  = True                # EM return data is not ready when generating AU descriptor
        self.returnsCurrency = None
        self.clippedReturns = True
        self.fuzzyScale = 0.94              # AU descriptors use DescriptorClass which doesn't load enough returns

class Market_Sensitivity_CN_52W(Regional_Market_Sensitivity):
    '''
        Used for AU4 model
    '''
    def __init__(self, connections,gp=None):
        Regional_Market_Sensitivity.__init__(self, connections,gp=gp)
        self.historyInYears = 1
        self.regInputs = [('Market', -2), ('Market', 'Local')]
        self.returnsCurrency = None
        self.clippedReturns = True
        self.fuzzyScale = 0.94              # AU descriptors use DescriptorClass which doesn't load enough returns

# Size and liquidity descriptors
class LnIssuerCap(DescriptorClass):
    """Class to compute ln of market cap for size exposure
    """

    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)
    
    def buildDescriptor(self, data, rootClass):
        self.log.debug('LnIssuerCap.buildDescriptor')
        size = DescriptorExposures.generate_size_exposures(data)
        return self.setUpReturnObject(data.universe, size)

class LnTrading_Activity(DescriptorClass):
    """Class to compute ln of trading activity
    """
    
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)
        self.simple = False
        self.lnComb = True
        self.median = False
        self.daysBack = 90
        self.weights = None
        self.peak = 0
        self.peak2 = 0
    
    def buildDescriptor(self, data, rootClass):
        self.log.debug('LnTrading_Activity.buildDescriptor')
        params = Utilities.Struct()
        params.simple = self.simple
        params.lnComb = self.lnComb
        params.daysBack = self.daysBack
        params.median = self.median
        params.weights = self.weights
        params.peak = self.peak
        params.peak2 = self.peak2

        liq = DescriptorExposures.generate_trading_volume_exposures(
            rootClass.dates[0], data, self.modelDB, self.marketDB,
            params, rootClass.numeraire_id)
        return self.setUpReturnObject(data.universe, liq)

class LnTrading_Activity_60D(LnTrading_Activity):
    """Class to compute 60 day (90 calendar-day) ln of trading activity
    """
    def __init__(self, connections, gp=None):
        LnTrading_Activity.__init__(self, connections, gp=gp)

class LnTrading_Activity_20D(LnTrading_Activity):
    """Class to compute 20 day (30 calendar-day) ln of trading activity
    """
    def __init__(self, connections, gp=None):
        LnTrading_Activity.__init__(self, connections, gp=gp)
        self.daysBack = 30

class LnTrading_Activity_125D(LnTrading_Activity):
    """Class to compute 125 day (180 calendar-day) ln of trading activity
    """
    def __init__(self, connections, gp=None):
        LnTrading_Activity.__init__(self, connections, gp=gp)
        self.daysBack = 180

class Amihud_Liquidity(DescriptorClass):
    """Class to compute Amihud liquidity measure
    """
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)
        self.daysBack = 60
        self.applyProxy = False
        self.useFreeFloatMktCap = False
    def buildDescriptor(self, data, rootClass):
        self.log.debug('Amihud_Liquidity.buildDescriptor')
        tmpReturns = self.loadReturnsArray(
                data, rootClass, daysBack=self.returnHistory, applyProxy=self.applyProxy)
        liq = DescriptorExposures.generateAmihudLiquidityExposures(
            rootClass.dates[0], tmpReturns, data, rootClass.rmg, self.modelDB, self.marketDB,
            self.daysBack, rootClass.numeraire_id,self.useFreeFloatMktCap)
        return self.setUpReturnObject(data.universe, liq)

class Amihud_Liquidity_60D(Amihud_Liquidity):
    """Class to compute 60-day Amihud liquidity measure
    """
    def __init__(self, connections, gp=None):
        Amihud_Liquidity.__init__(self, connections, gp=gp)

class Amihud_Liquidity_125D(Amihud_Liquidity):
    """Class to compute 125-day Amihud liquidity measure
    """
    def __init__(self, connections, gp=None):
        Amihud_Liquidity.__init__(self, connections, gp=gp)
        self.daysBack = 125

class Amihud_Liquidity_Adj_60D(Amihud_Liquidity):
    """Class to compute 60-day Amihud liquidity measure
    """
    def __init__(self, connections, gp=None):
        Amihud_Liquidity.__init__(self, connections, gp=gp)
        self.useFreeFloatMktCap = True

class Amihud_Liquidity_Adj_125D(Amihud_Liquidity):
    """Class to compute 125-day Amihud liquidity measure
    """
    def __init__(self, connections, gp=None):
        Amihud_Liquidity.__init__(self, connections, gp=gp)
        self.daysBack = 125
        self.useFreeFloatMktCap = True

# Balance sheet stuff
# Leverage items
class Debt_to_Assets_Quarterly(DescriptorClass):
    """Class to compute raw Debt to Asset scores"""
    
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)
        
    def buildDescriptor(self, data, rootClass):
        values, freq = DescriptorRatios.DebtToTotalAssets(
                self.modelDB, self.marketDB,
                trackList=rootClass.trackList,
                useFixedFrequency=rootClass.useFixedFrequency,
                sidRanges=data.sidRanges).\
                        getValues(rootClass.dates[0], data, rootClass.numeraire_id)
        return self.setUpReturnObject(data.universe, values)

class Debt_to_Assets_Annual(DescriptorClass):
    """Class to compute raw Debt to Asset scores"""
    
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)
        
    def buildDescriptor(self, data, rootClass):
        values, freq = DescriptorRatios.DebtToTotalAssets(
                self.modelDB, self.marketDB, 
                trackList=rootClass.trackList,
                useFixedFrequency=DescriptorRatios.DescriptorRatio.AnnualFrequency,
                sidRanges=data.sidRanges).\
                        getValues(rootClass.dates[0], data, rootClass.numeraire_id)
        return self.setUpReturnObject(data.universe, values)

class Debt_to_Equity_Quarterly(DescriptorClass):
    """Class to compute raw Debt to Equity scores"""

    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)

    def buildDescriptor(self, data, rootClass):
        values, freq = DescriptorRatios.DebtToEquity(
                self.modelDB, self.marketDB,
                trackList=rootClass.trackList,
                useFixedFrequency=rootClass.useFixedFrequency,
                denomProcess='average',
                sidRanges=data.sidRanges).\
                        getValues(rootClass.dates[0], data, rootClass.numeraire_id)
        return self.setUpReturnObject(data.universe, values)

class Debt_to_Equity_Annual(DescriptorClass):
    """Class to compute raw Debt to Equity scores"""

    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)

    def buildDescriptor(self, data, rootClass):
        values, freq = DescriptorRatios.DebtToEquity(
                self.modelDB, self.marketDB, 
                trackList=rootClass.trackList,
                useFixedFrequency=DescriptorRatios.DescriptorRatio.AnnualFrequency,
                denomProcess='average',
                denomDaysBack=3*365,
                sidRanges=data.sidRanges).\
                        getValues(rootClass.dates[0], data, rootClass.numeraire_id)
        return self.setUpReturnObject(data.universe, values)

# Growth descriptors
class Sales_Growth_RPF_Annual(DescriptorClass):
    """Class to compute raw Est.Sales Growth scores"""
    
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)

    def buildDescriptor(self, assetData, rootClass):
        values = DescriptorExposures.generate_growth_rate_annual(
                'sale', assetData, rootClass.dates[0], 
                rootClass.numeraire_id, self.modelDB, 
                self.marketDB, forecastItem='rev_median_ann', 
                forecastItemScaleByTSO=False,
                trackList=rootClass.trackList)
        return self.setUpReturnObject(assetData.universe, values)

class Sales_Growth_RPF_AFQ(DescriptorClass):
    """Class to compute raw Est.Earnings Growth scores"""
    
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)

    def buildDescriptor(self, assetData, rootClass):
        if rootClass.descriptorType == 'SCM':
            values = DescriptorExposures.generate_growth_rateAFQ_mix_version(
                    'sale', assetData, rootClass.dates[0], rootClass.numeraire_id, self.modelDB, self.marketDB,
                    forecastItem='rev_median_ann', forecastItemScaleByTSO=False,
                    trackList=rootClass.trackList, requireConsecQtrData=16)
        else:
            values = DescriptorExposures.generate_growth_rateAFQ(
                    'sale', assetData, rootClass.dates[0], rootClass.numeraire_id, self.modelDB, self.marketDB,
                    forecastItem='rev_median_ann', forecastItemScaleByTSO=False,
                    trackList=rootClass.trackList, requireConsecQtrData=16)

        return self.setUpReturnObject(assetData.universe, values)

class Earnings_Growth_RPF_Annual(DescriptorClass):
    """Class to compute raw Est.Earnings Growth scores"""
    
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)

    def buildDescriptor(self, assetData, rootClass):
        values = DescriptorExposures.generate_growth_rate_annual(
                'ibei', assetData, rootClass.dates[0], 
                rootClass.numeraire_id, self.modelDB, 
                self.marketDB, forecastItem='eps_median_ann', 
                forecastItemScaleByTSO=True,
                trackList=rootClass.trackList)
        return self.setUpReturnObject(assetData.universe, values)

class Earnings_Growth_RPF_AFQ(DescriptorClass):
    """Class to compute raw Est.Earnings Growth scores"""
    
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)

    def buildDescriptor(self, assetData, rootClass):
        if rootClass.descriptorType == 'SCM':
            values = DescriptorExposures.generate_growth_rateAFQ_mix_version(
                'ibei', assetData, rootClass.dates[0], rootClass.numeraire_id, self.modelDB, self.marketDB,
                forecastItem='eps_median_ann', forecastItemScaleByTSO=True,
                trackList=rootClass.trackList, requireConsecQtrData=16)
        else:
            values = DescriptorExposures.generate_growth_rateAFQ(
                'ibei', assetData, rootClass.dates[0], rootClass.numeraire_id, self.modelDB, self.marketDB,
                forecastItem='eps_median_ann', forecastItemScaleByTSO=True,
                trackList=rootClass.trackList, requireConsecQtrData=16)

        return self.setUpReturnObject(assetData.universe, values)

# Value items
class Book_to_Price_Quarterly(DescriptorClass):
    """Class to compute book-to-price scores"""
    
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)

    def buildDescriptor(self, data, rootClass):
        values, freq = DescriptorRatios.BookToPrice(
                self.modelDB, self.marketDB,
                useFixedFrequency=rootClass.useFixedFrequency,
                sidRanges=data.sidRanges).\
                        getValues(rootClass.dates[0], data, rootClass.numeraire_id)
        return self.setUpReturnObject(data.universe, values)

class Book_to_Price_Annual(DescriptorClass):
    """Class to compute book-to-price scores"""
    
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)

    def buildDescriptor(self, data, rootClass):
        values, freq = DescriptorRatios.BookToPrice(
                self.modelDB, self.marketDB,
                useFixedFrequency=DescriptorRatios.DescriptorRatio.AnnualFrequency,
                sidRanges=data.sidRanges).\
                        getValues(rootClass.dates[0], data, rootClass.numeraire_id)
        return self.setUpReturnObject(data.universe, values)

class Earnings_to_Price_Quarterly(DescriptorClass):
    """Class to compute earnings-to-price scores"""
    
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)

    def buildDescriptor(self, data, rootClass):
        values, freq = DescriptorRatios.EarningsToPrice(
                self.modelDB, self.marketDB,
                useFixedFrequency=rootClass.useFixedFrequency,
                numeratorProcess='annualize',
                numeratorNegativeTreatment=None,
                trackList=rootClass.trackList,
                sidRanges=data.sidRanges).\
                        getValues(rootClass.dates[0], data, rootClass.numeraire_id)
        return self.setUpReturnObject(data.universe, values)

class Earnings_to_Price_Annual(DescriptorClass):
    """Class to compute earnings-to-price scores"""
    
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)

    def buildDescriptor(self, data, rootClass):
        values, freq = DescriptorRatios.EarningsToPrice(
                self.modelDB, self.marketDB, 
                useFixedFrequency=DescriptorRatios.DescriptorRatio.AnnualFrequency,
                numeratorProcess='extractlatest',
                numeratorNegativeTreatment=None,
                trackList=rootClass.trackList,
                sidRanges=data.sidRanges).\
                        getValues(rootClass.dates[0], data, rootClass.numeraire_id)
        return self.setUpReturnObject(data.universe, values)

class Est_Earnings_to_Price_Annual(DescriptorClass):
    """Class to compute Est Earnings-to-Price scores"""
    # Used by AU4 models only
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)

    def buildDescriptor(self, data, rootClass):
        params = Utilities.Struct()
        params.maskNegative = False

        values = DescriptorExposures.generate_est_earnings_to_price_12MFL(
                rootClass.dates[0], data, self.modelDB, self.marketDB, params,
                rootClass.numeraire_id, useFixedFrequency=DescriptorRatios.DescriptorRatio.AnnualFrequency,
                trackList=rootClass.trackList)
        return self.setUpReturnObject(data.universe, values)

class Est_Earnings_to_Price_12MFL_Quarterly(DescriptorClass):
    """Class to compute Est Earnings-to-Price scores"""

    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)

    def buildDescriptor(self, data, rootClass):
        params = Utilities.Struct()
        params.maskNegative = False

        values = DescriptorExposures.generate_est_earnings_to_price_12MFL(
            rootClass.dates[0], data, self.modelDB, self.marketDB, params,
            rootClass.numeraire_id, useFixedFrequency=rootClass.useFixedFrequency,
            trackList=rootClass.trackList)
        return self.setUpReturnObject(data.universe, values)

class Est_Earnings_to_Price_12MFL_Annual(DescriptorClass):
    """Class to compute Est Earnings-to-Price scores"""

    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)

    def buildDescriptor(self, data, rootClass):
        params = Utilities.Struct()
        params.maskNegative = False

        values = DescriptorExposures.generate_est_earnings_to_price_12MFL(
            rootClass.dates[0], data, self.modelDB, self.marketDB, params,
            rootClass.numeraire_id, useFixedFrequency=DescriptorRatios.DescriptorRatio.AnnualFrequency,
            trackList=rootClass.trackList)
        return self.setUpReturnObject(data.universe, values)

# Dividend descriptors
class Dividend_Yield_Quarterly(DescriptorClass):
    """Class to compute raw dividend yield
    Note that DescriptorRatios.DividendYield does not depend on the data frequency,
    so this class and Dividend_Yield_Annual yield the same results
    """
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)

    def buildDescriptor(self, data, rootClass):
        values, freq = DescriptorRatios.DividendYield(
                self.modelDB, self.marketDB,
                trackList=rootClass.trackList,
                useFixedFrequency=rootClass.useFixedFrequency,
                sidRanges=data.sidRanges).\
                        getValues(rootClass.dates[0], data, rootClass.numeraire_id)
        return self.setUpReturnObject(data.universe, values)

class Dividend_Yield_Annual(DescriptorClass):
    """Class to compute raw dividend yield"""
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)

    def buildDescriptor(self, data, rootClass):
        values, freq = DescriptorRatios.DividendYield(
                self.modelDB, self.marketDB,
                trackList=rootClass.trackList,
                useFixedFrequency=DescriptorRatios.DescriptorRatio.AnnualFrequency, 
                sidRanges=data.sidRanges).\
                        getValues(rootClass.dates[0], data, rootClass.numeraire_id)
        return self.setUpReturnObject(data.universe, values)

# QMJ Profitability descriptors
class Return_on_Equity_Quarterly(DescriptorClass):
    """Class to compute raw return-on-equity scores"""
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)

    def buildDescriptor(self, data, rootClass):
        values, freq = DescriptorRatios.ReturnOnEquity(
                self.modelDB, self.marketDB,
                useFixedFrequency=rootClass.useFixedFrequency,
                numeratorProcess='annualize',
                denomProcess='average',
                sidRanges=data.sidRanges).\
                        getValues(rootClass.dates[0], data, rootClass.numeraire_id)
        return self.setUpReturnObject(data.universe, values)

class Return_on_Equity_Annual(DescriptorClass):
    """Class to compute raw return-on-equity scores"""

    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)

    def buildDescriptor(self, data, rootClass):
        values, freq = DescriptorRatios.ReturnOnEquity(
                self.modelDB, self.marketDB,
                useFixedFrequency=DescriptorRatios.DescriptorRatio.AnnualFrequency,
                numeratorProcess='extractLatest',
                denomProcess='average',
                denomDaysBack=3*365,
                sidRanges=data.sidRanges).\
                        getValues(rootClass.dates[0], data, rootClass.numeraire_id)
        return self.setUpReturnObject(data.universe, values)

class Return_on_Assets_Quarterly(DescriptorClass):
    """Class to compute raw return-on-assets scores"""

    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)

    def buildDescriptor(self, data, rootClass):
        values, freq = DescriptorRatios.ReturnOnAssets(
                self.modelDB, self.marketDB,
                useFixedFrequency=rootClass.useFixedFrequency,
                numeratorProcess='annualize',
                denomProcess='average',
                sidRanges=data.sidRanges).\
                        getValues(rootClass.dates[0], data, rootClass.numeraire_id)
        return self.setUpReturnObject(data.universe, values)

class Return_on_Assets_Annual(DescriptorClass):
    """Class to compute raw return-on-assets scores"""

    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)

    def buildDescriptor(self, data, rootClass):
        values, freq = DescriptorRatios.ReturnOnAssets(
                self.modelDB, self.marketDB,
                useFixedFrequency=DescriptorRatios.DescriptorRatio.AnnualFrequency,
                denomProcess='average',
                denomDaysBack=3*365,
                sidRanges=data.sidRanges).getValues(rootClass.dates[0], data, rootClass.numeraire_id)
        return self.setUpReturnObject(data.universe, values)

class Gross_Margin_Quarterly(DescriptorClass):
    """Class to compute raw asset turnover
    """
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)

    def buildDescriptor(self, data, rootClass):
        values, freq = DescriptorRatios.GrossMargin(
                self.modelDB, self.marketDB,
                useFixedFrequency=rootClass.useFixedFrequency,
                numeratorProcess='annualize',
                denomProcess='annualize',
                sidRanges=data.sidRanges).\
                        getValues(rootClass.dates[0], data, rootClass.numeraire_id)
        return self.setUpReturnObject(data.universe, values)

class Gross_Margin_Annual(DescriptorClass):
    """Class to compute raw asset turnover
    """
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)

    def buildDescriptor(self, data, rootClass):
        values, freq = DescriptorRatios.GrossMargin(
                self.modelDB, self.marketDB, 
                useFixedFrequency=DescriptorRatios.DescriptorRatio.AnnualFrequency,
                sidRanges=data.sidRanges).\
                        getValues(rootClass.dates[0], data, rootClass.numeraire_id)
        return self.setUpReturnObject(data.universe, values)

class Sales_to_Assets_Quarterly(DescriptorClass):
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)

    def buildDescriptor(self, data, rootClass):
        values, freq = DescriptorRatios.SalesToAssets(
                self.modelDB, self.marketDB,
                useFixedFrequency=rootClass.useFixedFrequency,
                numeratorProcess='annualize', 
                denomProcess='average',
                sidRanges=data.sidRanges).\
                        getValues(rootClass.dates[0], data, rootClass.numeraire_id)
        return self.setUpReturnObject(data.universe, values)

class Sales_to_Assets_Annual(DescriptorClass):
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)

    def buildDescriptor(self, data, rootClass):
        values, freq = DescriptorRatios.SalesToAssets(
                self.modelDB, self.marketDB, 
                useFixedFrequency=DescriptorRatios.DescriptorRatio.AnnualFrequency,
                numeratorProcess='extractLatest',
                denomProcess='average',
                denomDaysBack=3*365,
                sidRanges=data.sidRanges).\
                        getValues(rootClass.dates[0], data, rootClass.numeraire_id)
        return self.setUpReturnObject(data.universe, values)

class CashFlow_to_Assets_Annual(DescriptorClass):
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)

    def buildDescriptor(self, data, rootClass):
        values, freq = DescriptorRatios.CashFlowToAssets(
                self.modelDB, self.marketDB, 
                useFixedFrequency=DescriptorRatios.DescriptorRatio.AnnualFrequency,
                numeratorProcess='extractlatest',
                denomProcess='average',
                denomDaysBack=3*365,
                sidRanges=data.sidRanges).\
                        getValues(rootClass.dates[0], data, rootClass.numeraire_id)
        return self.setUpReturnObject(data.universe, values)

class CashFlow_to_Income_Annual(DescriptorClass):
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)

    def buildDescriptor(self, data, rootClass):
        values, freq = DescriptorRatios.CashFlowToIncome(
                self.modelDB, self.marketDB,
                useFixedFrequency=DescriptorRatios.DescriptorRatio.AnnualFrequency,
                numeratorProcess='average', denomProcess='average',
                numeratorDaysBack=3*365, denomDaysBack=3*365,
                sidRanges=data.sidRanges).\
                        getValues(rootClass.dates[0], data, rootClass.numeraire_id)
        return self.setUpReturnObject(data.universe, values)
    
class Random_Factor(DescriptorClass):
    """Class to compute raw asset turnover
    """
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)

    def buildDescriptor(self, data, rootClass):
        # values, freq = DescriptorRatios.XXXRatio(self.modelDB, self.marketDB).\
        #             getValues(rootClass.dates[0], data, rootClass.numeraire_id)
        import random
        import pandas as pd
        sigma = 0.5
        w = 0.975
        sigma_e = 0.02
        desc_id = 229 # Random Factor
        curr = rootClass.numeraire_ISO
        AnnualFrequency = DescriptorRatios.DataFrequency('_ann')

        data_date = rootClass.dates[0]
        au_rmg = rootClass.rmg
        # (1) get previous trading date
        date_range = self.modelDB.getDateRange([au_rmg], data_date - datetime.timedelta(days=10), data_date - datetime.timedelta(days=1))
        prev_date = date_range[-1]
        # (2) get current coverage universe
        # act_issues = self.modelDB.getActiveSubIssues(au_rmg, prev_date) $previous coverage universe
        sub_issue  = data.universe
        # (3) get descriptor value of coverage universe on previous trading date
        a = self.modelDB.loadDescriptorData(prev_date, sub_issue, curr, desc_id, rollOverData=False)
        a = a.toDataFrame()
        tf_hasValue = pd.notnull(a)
        print('---------------------------------------------------There are ' + str(tf_hasValue.sum()[0]) + ' issues existing before.')
        values = Matrices.allMasked(len(data.universe), dtype=float)
        freq = Matrices.allMasked(len(data.universe), dtype=object)

        for (i, sid) in enumerate(sub_issue):
            if tf_hasValue.iloc[i,0]: #if has value
                values[i] = w * a.iloc[i,0] + random.gauss(0, sigma_e)
            else:
                values[i] = random.gauss(0,sigma)

            if values[i] is not ma.masked:
                freq[i] = AnnualFrequency

        return self.setUpReturnObject(data.universe, values)

########## Time series sensitivities ##########
class TS_Sensitivity(DescriptorClass):
    """Class to compute time series sensitivies
    """
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)
      
        # History and weighting
        self.historyInYears = 2
        self.minHistoryInYears = None # specify to use variable number of observations in regression
        self.weighting = 'pyramid' # exponential, triangle, pyramid, or None (equal) 
        self.frequency = 'weekly' # weekly, daily, monthly
       
        # Parameters for retrieving DAILY asset returns 
        self.applyProxy = True # Proxy missing returns 
        self.adjustForRT = False # Adjust returns for returns timing (set to True if using daily returns) 
        self.clippedReturns = True # Generally recommended
        self.fuzzyScale = None

        # Parameters for filling missing (compounded) returns
        self.fillWithMarket = True # fill missing (compounded) returns with market returns;
        self.fillInputsWithZeros = True # fill missing (compounded) inputs with zeros
        self.maskShortHistory = True

        # Regression settings
        self.lag = None
        self.robust = False
        self.kappa = 5.0
        self.inclIntercept = True

        # Parameters for saving regression data
        self.getRegStats = False
        self.returnName = None

    def setHistoryAndRegressionParameters(self):
        """Set parameters used to retrieve and 
        compound returns and inputs"""

        self.daysBack = int(self.historyInYears*252) + 40 # number of daily asset returns used
        self.minnobs = None
        if self.frequency == 'daily':
            self.nobs = int(self.historyInYears*252)
            if self.minHistoryInYears is not None:
                self.minnobs = int(self.minHistoryInYears*252)
            self.halflife = 21
            self.fadePeak = 21
        elif self.frequency == 'weekly':
            self.nobs = int(self.historyInYears*52)
            if self.minHistoryInYears is not None:
                self.minnobs = int(self.minHistoryInYears*52)
            self.halflife = 4
            self.fadePeak = 4
        elif self.frequency == 'monthly':
            self.nobs = int(self.historyInYears*12)
            if self.minHistoryInYears is not None:
                self.minnobs = int(self.minHistoryInYears*12)
            self.halflife = 2
            self.fadePeak = 2

        # Regression parameters
        self.params = {
                'lag': self.lag,
                'robust': self.robust,
                'kappa': self.kappa,
                'weighting': self.weighting,
                'halflife': self.halflife,
                'fadePeak': self.fadePeak,
                'frequency': self.frequency,
                'fuzzyScale': self.fuzzyScale,
                'nobs': self.nobs,
                'minnobs': self.minnobs,
                'inclIntercept': self.inclIntercept
        }

    def buildDescriptor(self, data, rootClass):
        self.log.debug('%s.buildDescriptor' % self.name)
        
        # Tweak for SCM forex descriptors, which sometimes have insufficent returns history
        if rootClass.descriptorType == 'SCM':
            self.fuzzyScale = 0.94

        # Set history and regression parameters
        self.setHistoryAndRegressionParameters()

        # Special treatment for FX sensitivities: skip pegged currencies
        if self.factorSensitivity.lower() == 'forex':
            pegged = CurrencyRisk.getPeggedCurrencies(self.baseCurrency, rootClass.dates[0])
            if rootClass.rmg.currency_code in pegged:
                return self.setUpReturnObject(data.universe, Matrices.allMasked(len(data.universe), dtype=float))

        # Get output data (daily asset returns)
        assetReturns = self.loadReturnsArray(
                data, rootClass, daysBack=self.daysBack, adjustForRT=self.adjustForRT, 
                clippedReturns=self.clippedReturns, applyProxy=self.applyProxy)
        modelDate = max(assetReturns.dates)
        if rootClass.descriptorType != 'SCM':
            assert(assetReturns.data.shape[1] >= self.daysBack)

        # Set up regression parameters
        TSR = TimeSeriesRegression.TimeSeriesRegression(
                self.modelDB, self.marketDB, TSRParameters=self.params,
                debugOutput=self.debuggingReporting, fillWithMarket=self.fillWithMarket,
                fillInputsWithZeros=self.fillInputsWithZeros,
                localRMG=rootClass.rmg, getRegStats=self.getRegStats)

        # Run regression(s)
        if self.reg1inputs is not None:
            (mm1, mm) = TSR.TSR(assetReturns, self.reg1inputs, self.reg2inputs)
        else:
            mm = TSR.TSR(assetReturns, self.reg2inputs)

        # Mask data for assets that have insufficient return histories
        if self.maskShortHistory:
            tDelta = datetime.timedelta(int(0.5 * self.historyInYears * 365))
            invalidIdx = [i for i, a in enumerate(data.universe) if data.sidRanges[a][0] > modelDate - tDelta]
            validIdx = [i for i, a in enumerate(data.universe) if i not in invalidIdx]
            validAssets = [a for i, a in enumerate(data.universe) if i in validIdx]
            mm.params[:, invalidIdx] = ma.masked

        # Process regression stats
        if self.getRegStats:
            res = Utilities.Struct()
            res.subids = [s.getSubIdString() for s in data.universe if s in validAssets]
            tstatDict = {}
            pvalDict = {}
            for f_n, factor in enumerate(mm.factors):
                tstatDict[factor] = list(mm.tstat[f_n, validIdx])
                pvalDict[factor] = list(mm.pvals[f_n, validIdx])
            res.tstatDict = tstatDict
            res.pvalDict = pvalDict
            res.rsq = [val for val in mm.rsquarecentered[validIdx]]
            res.vifs = mm.vifs
            res.dof = mm.dof
            self.saveRegressStats(modelDate, rootClass.rmg.rmg_id, res)

        # Assume desired beta is the first item in final regression unless told otherwise
        if self.returnName is None:
            values = mm.params[mm.factors.index(self.reg2inputs[0]), :]
        else:
            values = mm.params[mm.factors.index(self.returnName), :]
        
        return self.setUpReturnObject(data.universe, values)

    def saveRegressStats(self, modelDate, rmg_id, data):
        """Save regression stats to MongoDB"""
        coll = self.mongoDB[self.collName]
        # upsert regress stats
        modelDate = datetime.datetime(modelDate.year,
                                      modelDate.month,
                                      modelDate.day)
        baseDict = {
            'rmg_id': rmg_id,
            'dt': modelDate,
            'regression': self.name
        }
        dataDict = {}
        dataDict['sub_issue_ids'] = data.subids
        dataDict['tstats'] = data.tstatDict
        dataDict['pvals'] = data.pvalDict
        dataDict['rsquare'] = data.rsq
        dataDict['vifs'] = data.vifs
        dataDict['dof'] = data.dof 
        try:
            res = coll.update_one(baseDict, {'$set': dataDict}, upsert=True)
        except:
            self.log.exception('Unexpected error: %s' % sys.exc_info()[0])

class XRate_104W_XDR(TS_Sensitivity):
    """Class to compute SDR exchange rate sensitivity
    """
    def __init__(self, connections, gp=None):
        TS_Sensitivity.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.maskShortHistory = False
        self.factorSensitivity = 'Forex' # name of factor sensitivity
        self.baseCurrency = 'XDR'
        self.reg1inputs = None
        self.reg2inputs = [(self.factorSensitivity, self.baseCurrency), ('Market', 'Local')]

class XRate_52W_XDR(XRate_104W_XDR):
    """Class to compute SDR exchange rate sensitivity
    """
    def __init__(self, connections, gp=None):
        XRate_104W_XDR.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.historyInYears = 1
        self.weighting = None

class XRate_104W_USD(XRate_104W_XDR):
    """Class to compute USD exchange rate sensitivity
    """
    def __init__(self, connections, gp=None):
        XRate_104W_XDR.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.baseCurrency = 'USD'
        self.reg2inputs = [(self.factorSensitivity, self.baseCurrency), ('Market', 'Local')]

class XRate_52W_USD(XRate_104W_USD):
    """Class to compute USD exchange rate sensitivity
    """
    def __init__(self, connections, gp=None):
        XRate_104W_USD.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.historyInYears = 1
        self.weighting = None

class USLTRate_Sensitivity_104W(TS_Sensitivity):
    def __init__(self, connections, gp=None):
        TS_Sensitivity.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.factorSensitivity = 'Long Term Rate' # name of factor sensitivity
        self.reg1inputs = [('Market', 'Local')]
        self.reg2inputs = [(self.factorSensitivity, 'US')]

class USLTRate_Sensitivity_NoMkt_104W(USLTRate_Sensitivity_104W):
    def __init__(self, connections, gp=None):
        USLTRate_Sensitivity_104W.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.reg1inputs = None

class GBLTRate_Sensitivity_104W(TS_Sensitivity):
    def __init__(self, connections, gp=None):
        TS_Sensitivity.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.factorSensitivity = 'Long Term Rate' # name of factor sensitivity
        self.reg1inputs = [('Market', 'Local')]
        self.reg2inputs = [(self.factorSensitivity, 'GB')]

class GBLTRate_Sensitivity_NoMkt_104W(GBLTRate_Sensitivity_104W):
    def __init__(self, connections, gp=None):
        GBLTRate_Sensitivity_104W.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.reg1inputs = None

class JPLTRate_Sensitivity_104W(TS_Sensitivity):
    def __init__(self, connections, gp=None):
        TS_Sensitivity.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.factorSensitivity = 'Long Term Rate' # name of factor sensitivity
        self.reg1inputs = [('Market', 'Local')]
        self.reg2inputs = [(self.factorSensitivity, 'JP')]

class JPLTRate_Sensitivity_NoMkt_104W(JPLTRate_Sensitivity_104W):
    def __init__(self, connections, gp=None):
        JPLTRate_Sensitivity_104W.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.reg1inputs = None

class EULTRate_Sensitivity_104W(TS_Sensitivity):
    def __init__(self, connections, gp=None):
        TS_Sensitivity.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.factorSensitivity = 'Long Term Rate' # name of factor sensitivity
        self.reg1inputs = [('Market', 'Local')]
        self.reg2inputs = [(self.factorSensitivity, 'EP')]

class EULTRate_Sensitivity_NoMkt_104W(EULTRate_Sensitivity_104W):
    def __init__(self, connections, gp=None):
        EULTRate_Sensitivity_104W.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.reg1inputs = None

class USSTRate_Sensitivity_104W(TS_Sensitivity):
    def __init__(self, connections, gp=None):
        TS_Sensitivity.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.factorSensitivity = 'Short Term Rate' # name of factor sensitivity
        self.reg1inputs = [('Market', 'Local')]
        self.reg2inputs = [(self.factorSensitivity, 'US')]

class USSTRate_Sensitivity_NoMkt_104W(USSTRate_Sensitivity_104W):
    def __init__(self, connections, gp=None):
        USSTRate_Sensitivity_104W.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.reg1inputs = None

class GBSTRate_Sensitivity_104W(TS_Sensitivity):
    def __init__(self, connections, gp=None):
        TS_Sensitivity.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.factorSensitivity = 'Short Term Rate' # name of factor sensitivity
        self.reg1inputs = [('Market', 'Local')]
        self.reg2inputs = [(self.factorSensitivity, 'GB')]

class GBSTRate_Sensitivity_NoMkt_104W(GBSTRate_Sensitivity_104W):
    def __init__(self, connections, gp=None):
        GBSTRate_Sensitivity_104W.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.reg1inputs = None

class JPSTRate_Sensitivity_104W(TS_Sensitivity):
    def __init__(self, connections, gp=None):
        TS_Sensitivity.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.factorSensitivity = 'Short Term Rate' # name of factor sensitivity
        self.reg1inputs = [('Market', 'Local')]
        self.reg2inputs = [(self.factorSensitivity, 'JP')]

class JPSTRate_Sensitivity_NoMkt_104W(JPSTRate_Sensitivity_104W):
    def __init__(self, connections, gp=None):
        JPSTRate_Sensitivity_104W.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.reg1inputs = None

class EUSTRate_Sensitivity_104W(TS_Sensitivity):
    def __init__(self, connections, gp=None):
        TS_Sensitivity.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.factorSensitivity = 'Short Term Rate' # name of factor sensitivity
        self.reg1inputs = [('Market', 'Local')]
        self.reg2inputs = [(self.factorSensitivity, 'EP')]

class EUSTRate_Sensitivity_NoMkt_104W(EUSTRate_Sensitivity_104W):
    def __init__(self, connections, gp=None):
        EUSTRate_Sensitivity_104W.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.reg1inputs = None

class USTermSpread_Sensitivity_104W(TS_Sensitivity):
    def __init__(self, connections, gp=None):
        TS_Sensitivity.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.factorSensitivity = 'Term Spread' # name of factor sensitivity
        self.reg1inputs = [('Market', 'Local')]
        self.reg2inputs = [(self.factorSensitivity, 'US')]

class USTermSpread_Sensitivity_NoMkt_104W(USTermSpread_Sensitivity_104W):
    def __init__(self, connections, gp=None):
        USTermSpread_Sensitivity_104W.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.reg1inputs = None

class GBTermSpread_Sensitivity_104W(TS_Sensitivity):
    def __init__(self, connections, gp=None):
        TS_Sensitivity.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.factorSensitivity = 'Term Spread' # name of factor sensitivity
        self.reg1inputs = [('Market', 'Local')]
        self.reg2inputs = [(self.factorSensitivity, 'GB')]

class GBTermSpread_Sensitivity_NoMkt_104W(GBTermSpread_Sensitivity_104W):
    def __init__(self, connections, gp=None):
        GBTermSpread_Sensitivity_104W.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.reg1inputs = None

class JPTermSpread_Sensitivity_104W(TS_Sensitivity):
    def __init__(self, connections, gp=None):
        TS_Sensitivity.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.factorSensitivity = 'Term Spread' # name of factor sensitivity
        self.reg1inputs = [('Market', 'Local')]
        self.reg2inputs = [(self.factorSensitivity, 'JP')]

class JPTermSpread_Sensitivity_NoMkt_104W(JPTermSpread_Sensitivity_104W):
    def __init__(self, connections, gp=None):
        JPTermSpread_Sensitivity_104W.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.reg1inputs = None

class EUTermSpread_Sensitivity_104W(TS_Sensitivity):
    def __init__(self, connections, gp=None):
        TS_Sensitivity.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.factorSensitivity = 'Term Spread' # name of factor sensitivity
        self.reg1inputs = [('Market', 'Local')]
        self.reg2inputs = [(self.factorSensitivity, 'EP')]

class EUTermSpread_Sensitivity_NoMkt_104W(EUTermSpread_Sensitivity_104W):
    def __init__(self, connections, gp=None):
        EUTermSpread_Sensitivity_104W.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.reg1inputs = None

class USCreditSpread_Sensitivity_104W(TS_Sensitivity):
    def __init__(self, connections, gp=None):
        TS_Sensitivity.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.factorSensitivity = 'Credit Spread' # name of factor sensitivity
        self.reg1inputs = [('Market', 'Local')]
        self.reg2inputs = [(self.factorSensitivity, 'US')]

class USCreditSpread_Sensitivity_NoMkt_104W(USCreditSpread_Sensitivity_104W):
    def __init__(self, connections, gp=None):
        USCreditSpread_Sensitivity_104W.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.reg1inputs = None

class USCreditSpread_Sensitivity_36M(TS_Sensitivity):
    def __init__(self, connections, gp=None):
        TS_Sensitivity.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.factorSensitivity = 'Credit Spread' # name of factor sensitivity
        self.frequency = 'monthly' # weekly, daily, monthly
        self.historyInYears = 3
        self.reg1inputs = [('Market', 'Local')]
        self.reg2inputs = [(self.factorSensitivity, 'US')]

class USCreditSpread_Sensitivity_NoMkt_36M(USCreditSpread_Sensitivity_36M):
    def __init__(self, connections, gp=None):
        USCreditSpread_Sensitivity_36M.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.reg1inputs = None

class GBCreditSpread_Sensitivity_104W(TS_Sensitivity):
    def __init__(self, connections, gp=None):
        TS_Sensitivity.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.factorSensitivity = 'Credit Spread' # name of factor sensitivity
        self.reg1inputs = [('Market', 'Local')]
        self.reg2inputs = [(self.factorSensitivity, 'GB')]

class GBCreditSpread_Sensitivity_NoMkt_104W(GBCreditSpread_Sensitivity_104W):
    def __init__(self, connections, gp=None):
        GBCreditSpread_Sensitivity_104W.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.reg1inputs = None

class JPCreditSpread_Sensitivity_104W(TS_Sensitivity):
    def __init__(self, connections, gp=None):
        TS_Sensitivity.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.factorSensitivity = 'Credit Spread' # name of factor sensitivity
        self.reg1inputs = [('Market', 'Local')]
        self.reg2inputs = [(self.factorSensitivity, 'JP')]

class JPCreditSpread_Sensitivity_NoMkt_104W(JPCreditSpread_Sensitivity_104W):
    def __init__(self, connections, gp=None):
        JPCreditSpread_Sensitivity_104W.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.reg1inputs = None

class EUCreditSpread_Sensitivity_104W(TS_Sensitivity):
    def __init__(self, connections, gp=None):
        TS_Sensitivity.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.factorSensitivity = 'Credit Spread' # name of factor sensitivity
        self.reg1inputs = [('Market', 'Local')]
        self.reg2inputs = [(self.factorSensitivity, 'EP')]

class EUCreditSpread_Sensitivity_NoMkt_104W(EUCreditSpread_Sensitivity_104W):
    def __init__(self, connections, gp=None):
        EUCreditSpread_Sensitivity_104W.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.reg1inputs = None

class Oil_Sensitivity_104W(TS_Sensitivity):
    def __init__(self, connections, gp=None):
        TS_Sensitivity.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.factorSensitivity = 'Oil' # name of factor sensitivity
        self.reg1inputs = [('Market', 'Local')]
        self.reg2inputs = [(self.factorSensitivity, 'Oil')]

class Oil_Sensitivity_NoMkt_104W(Oil_Sensitivity_104W):
    def __init__(self, connections, gp=None):
        Oil_Sensitivity_104W.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.reg1inputs = None

class Gold_Sensitivity_104W(TS_Sensitivity):
    def __init__(self, connections, gp=None):
        TS_Sensitivity.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.factorSensitivity = 'Gold' # name of factor sensitivity
        self.reg1inputs = [('Market', 'Local')]
        self.reg2inputs = [(self.factorSensitivity, 'Gold')]

class Gold_Sensitivity_NoMkt_104W(Gold_Sensitivity_104W):
    def __init__(self, connections, gp=None):
        Gold_Sensitivity_104W.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.reg1inputs = None

class Commodity_Sensitivity_104W(TS_Sensitivity):
    def __init__(self, connections, gp=None):
        TS_Sensitivity.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.factorSensitivity = 'Commodity' # name of factor sensitivity
        self.reg1inputs = [('Market', 'Local')]
        self.reg2inputs = [(self.factorSensitivity, 'Commodity')]

class Commodity_Sensitivity_NoMkt_104W(Commodity_Sensitivity_104W):
    def __init__(self, connections, gp=None):
        Commodity_Sensitivity_104W.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.reg1inputs = None

class USBEI_Sensitivity_104W(TS_Sensitivity):
    def __init__(self, connections, gp=None):
        TS_Sensitivity.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.factorSensitivity = 'BEI Rate' # name of factor sensitivity
        self.reg1inputs = [('Market', 'Local')]
        self.reg2inputs = [(self.factorSensitivity, 'US')]

class USBEI_Sensitivity_NoMkt_104W(USBEI_Sensitivity_104W):
    def __init__(self, connections, gp=None):
        USBEI_Sensitivity_104W.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.reg1inputs = None

class GBCreditSpread_Sensitivity_36M(TS_Sensitivity):
    def __init__(self, connections, gp=None):
        TS_Sensitivity.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.factorSensitivity = 'Credit Spread' # name of factor sensitivity
        self.frequency = 'monthly' # weekly, daily, monthly
        self.historyInYears = 3
        self.reg1inputs = [('Market', 'Local')]
        self.reg2inputs = [(self.factorSensitivity, 'GB')]

class GBCreditSpread_Sensitivity_NoMkt_36M(GBCreditSpread_Sensitivity_36M):
    def __init__(self, connections, gp=None):
        GBCreditSpread_Sensitivity_36M.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.reg1inputs = None

class JPCreditSpread_Sensitivity_36M(TS_Sensitivity):
    def __init__(self, connections, gp=None):
        TS_Sensitivity.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.factorSensitivity = 'Credit Spread' # name of factor sensitivity
        self.frequency = 'monthly' # weekly, daily, monthly
        self.historyInYears = 3
        self.reg1inputs = [('Market', 'Local')]
        self.reg2inputs = [(self.factorSensitivity, 'JP')]

class JPCreditSpread_Sensitivity_NoMkt_36M(JPCreditSpread_Sensitivity_36M):
    def __init__(self, connections, gp=None):
        JPCreditSpread_Sensitivity_36M.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.reg1inputs = None

class EUCreditSpread_Sensitivity_36M(TS_Sensitivity):
    def __init__(self, connections, gp=None):
        TS_Sensitivity.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.factorSensitivity = 'Credit Spread' # name of factor sensitivity
        self.frequency = 'monthly' # weekly, daily, monthly
        self.historyInYears = 3
        self.reg1inputs = [('Market', 'Local')]
        self.reg2inputs = [(self.factorSensitivity, 'EP')]

class EUCreditSpread_Sensitivity_NoMkt_36M(EUCreditSpread_Sensitivity_36M):
    def __init__(self, connections, gp=None):
        EUCreditSpread_Sensitivity_36M.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.reg1inputs = None

# Net of sector time series sensitivities 
class CAOil_Sensitivity_NetOfSecBeta_104W(TS_Sensitivity):
    def __init__(self, connections, gp=None):
        TS_Sensitivity.__init__(self, connections, gp=gp)

        # Important parameters
        self.name = self.__class__.__name__
        self.nestedReg = True
        self.historyInYears = 2
        self.minHistoryInYears = 5./12
        self.getRegStats = False

        # classification data for amp industry returns
        self.gicsDate = datetime.date(2018, 9, 29)
        self.gicsCls = Classification.GICSCustomCA4(self.gicsDate)
        clsFamily = self.modelDB.getMdlClassificationFamily('INDUSTRIES')
        clsMembers = self.modelDB.getMdlClassificationFamilyMembers(clsFamily)
        clsMember = [i for i in clsMembers if i.name==self.gicsCls.name][0]
        clsRevision = self.modelDB.getMdlClassificationMemberRevision(clsMember, self.gicsDate)

        # Set up the default regression(s)
        self.amp_id = 89
        self.factorSensitivity = 'Oil'
        self.reg1inputs = [('AMP', self.amp_id, clsRevision.id)]
        self.reg2inputs = [(self.factorSensitivity, self.factorSensitivity, self.factorSensitivity)]

    def buildDescriptor(self, data, rootClass):
        self.log.debug('%s.buildDescriptor' % self.name)
        
        # Set history and regression parameters
        self.setHistoryAndRegressionParameters()

        # Get output data (daily asset returns)
        assetReturns = self.loadReturnsArray(data, rootClass, daysBack=self.daysBack,
                adjustForRT=self.adjustForRT, clippedReturns=self.clippedReturns, applyProxy=self.applyProxy)
        assetReturnsDF = assetReturns.toDataFrame()
        assert(assetReturns.data.shape[1] >= self.daysBack)

        # List of params to return
        returnParams = ma.masked_all((len(data.universe),), dtype=float) 

        # note that ADRs will be masked since we don't have exposures
        if self.getRegStats:
            rsq1 = ma.masked_all((len(data.universe),), dtype=float) 
            rsq = ma.masked_all((len(data.universe),), dtype=float) 

        # Get industry exposures
        modelDate = max(assetReturns.dates)
        industryList = [f.description for f in self.gicsCls.getLeafNodes(self.modelDB).values()]
        exposures = self.gicsCls.getExposures(modelDate, data.universe, industryList, self.modelDB)
        expM = pandas.DataFrame(exposures, index=industryList, columns=data.universe).T.fillna(0.)

        # Iterate through industries and run regression 
        for ind in industryList:
            indSids = expM[expM[ind] == 1].index
            indAssetReturns = Matrices.TimeSeriesMatrix.fromDataFrame(assetReturnsDF.reindex(index=indSids))
            
            # Set up regression parameters
            TSR = TimeSeriesRegression.TimeSeriesRegression(
                    self.modelDB, self.marketDB, TSRParameters=self.params,
                    debugOutput=self.debuggingReporting, fillWithMarket=self.fillWithMarket,
                    fillInputsWithZeros=self.fillInputsWithZeros,
                    localRMG=rootClass.rmg, localSector=ind, getRegStats=self.getRegStats)
            
            # Run regression(s)
            if self.reg1inputs is not None:
                (mm1, mm) = TSR.TSR(indAssetReturns, self.reg1inputs, self.reg2inputs)
            else:
                mm = TSR.TSR(indAssetReturns, self.reg2inputs)

            retName = self.reg2inputs[0]
            for i, sid in enumerate(indSids):
                returnParams[data.assetIdxMap[sid]] = mm.params[mm.factors.index(retName), i]
                if self.getRegStats:
                    if self.nestedReg:
                        rsq1[data.assetIdxMap[sid]] = mm1.rsquarecentered[i]
                    rsq[data.assetIdxMap[sid]] = mm.rsquarecentered[i]

            #if self.debuggingReporting:
            #    fname = 'tmp/params-%s-%s-%s.csv' % \
            #            (re.sub('[ ()&]', '', ind), self.name, modelDate)
            #    pandas.Series(returnParams, index=data.universe).dropna().to_csv(fname)

        if self.getRegStats:
            pandas.DataFrame({
                'Sensitivity': pandas.Series(returnParams, index=data.universe),
                'Rsq1': pandas.Series(rsq1, index=data.universe),
                'RsqFinal': pandas.Series(rsq, index=data.universe)
            }).to_csv('tmp/AllParamsWithRsq-%s-%s.csv' % (self.name, modelDate))

        return self.setUpReturnObject(data.universe, returnParams)

class CAOil_Sensitivity_NetOfSecBeta_52W(CAOil_Sensitivity_NetOfSecBeta_104W):
    def __init__(self, connections, gp=None):
        CAOil_Sensitivity_NetOfSecBeta_104W.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.historyInYears = 1

class CAGold_Sensitivity_NetOfSecBeta_104W(CAOil_Sensitivity_NetOfSecBeta_104W):
    def __init__(self, connections, gp=None):
        CAOil_Sensitivity_NetOfSecBeta_104W.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.factorSensitivity = 'Gold'
        self.reg2inputs = [(self.factorSensitivity, self.factorSensitivity, self.factorSensitivity)]

class CAGold_Sensitivity_NetOfSecBeta_52W(CAGold_Sensitivity_NetOfSecBeta_104W):
    def __init__(self, connections, gp=None):
        CAGold_Sensitivity_NetOfSecBeta_104W.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.historyInYears = 1
