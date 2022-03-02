
import os
import sys
import logging
import datetime
import optparse
import configparser
import copy
import numpy
import numpy.ma as ma
import bisect
from operator import itemgetter

from marketdb import MarketDB
import riskmodels
from riskmodels import Connections
from riskmodels import Matrices
from riskmodels import MFM
from riskmodels import RiskModels
from riskmodels import Utilities
from riskmodels import LegacyUtilities
from riskmodels import ModelDB
from riskmodels import AssetProcessor
from riskmodels import DescriptorSources
from riskmodels import DescriptorRatios
from riskmodels import DescriptorExposures
from riskmodels import VisualTool

class DescRatioComposition:
    """A class decomposes descriptors ratios into the corresponding fundamental data.
       paramsDict and instanceDict are copied from DescriptorSources.py in order to
       instantiate the root class
    """
    instanceDict = {'Earnings_to_Price_Quarterly': "DescriptorRatios.EarningsToPrice",
                    'Earnings_to_Price_Annual': "DescriptorRatios.EarningsToPrice",
                    'Book_to_Price_Annual': "DescriptorRatios.BookToPrice",
                    'Book_to_Price_Quarterly': "DescriptorRatios.BookToPrice",
                    'Debt_to_Assets_Quarterly': "DescriptorRatios.DebtToTotalAssets",
                    'Debt_to_Assets_Annual': "DescriptorRatios.DebtToTotalAssets",
                    'Debt_to_Equity_Quarterly': "DescriptorRatios.DebtToEquity",
                    'Debt_to_Equity_Annual': "DescriptorRatios.DebtToEquity",
                    'Dividend_Yield_Quarterly': "DescriptorRatios.DividendYield", 
                    'Dividend_Yield_Annual': "DescriptorRatios.DividendYield",
                    'Return_on_Equity_Quarterly': "DescriptorRatios.ReturnOnEquity",
                    'Return_on_Equity_Annual': "DescriptorRatios.ReturnOnEquity",
                    'Return_on_Assets_Quarterly': "DescriptorRatios.ReturnOnAssets",
                    'Return_on_Assets_Annual': "DescriptorRatios.ReturnOnAssets",
                    'CashFlow_to_Assets_Annual': "DescriptorRatios.CashFlowToAssets",
                    'CashFlow_to_Income_Annual': "DescriptorRatios.CashFlowToIncome",
                    'Sales_to_Assets_Quarterly': "DescriptorRatios.SalesToAssets",
                    'Sales_to_Assets_Annual': "DescriptorRatios.SalesToAssets",
                    'Gross_Margin_Annual': "DescriptorRatios.GrossMargin"}
    
    
    paramsDict = {'Earnings_to_Price_Quarterly': "numeratorProcess='annualize'"\
                      ", numeratorNegativeTreatment=None",
                  'Earnings_to_Price_Annual': "DescriptorRatios.DescriptorRatio.AnnualFrequency,"\
                      "numeratorProcess='extractlatest',numeratorNegativeTreatment=None",
                  'Book_to_Price_Quarterly': "",
                  'Book_to_Price_Annual': "DescriptorRatios.DescriptorRatio.AnnualFrequency",
                  'Debt_to_Assets_Quarterly': "",
                  'Debt_to_Assets_Annual': "DescriptorRatios.DescriptorRatio.AnnualFrequency",
                  'Debt_to_Equity_Quarterly': "denomProcess='average'",
                  'Debt_to_Equity_Annual': "DescriptorRatios.DescriptorRatio.AnnualFrequency,"\
                      "denomProcess='average',denomDaysBack=3*365",
                  'Dividend_Yield_Quarterly': "",
                  'Dividend_Yield_Annual': "DescriptorRatios.DescriptorRatio.AnnualFrequency",
                  'Return_on_Equity_Quarterly': "numeratorProcess='annualize',"\
                      " denomProcess='average'",
                  'Return_on_Equity_Annual': "DescriptorRatios.DescriptorRatio.AnnualFrequency,"\
                      "numeratorProcess='extractLatest',denomProcess='average',denomDaysBack=3*365",
                  'Return_on_Assets_Quarterly': "numeratorProcess='annualize',"\
                      " denomProcess='average'",
                  'Return_on_Assets_Annual': "DescriptorRatios.DescriptorRatio.AnnualFrequency,"\
                      "denomProcess='average',denomDaysBack=3*365",
                  'CashFlow_to_Assets_Annual': "DescriptorRatios.DescriptorRatio.AnnualFrequency,"\
                      "numeratorProcess='extractlatest', denomProcess='average', denomDaysBack=3*365",
                  'CashFlow_to_Income_Annual': "DescriptorRatios.DescriptorRatio.AnnualFrequency,"\
                      "numeratorProcess='average', denomProcess='average'",
                  'Sales_to_Assets_Quarterly': "numeratorProcess='annualize',"\
                      " denomProcess='average'",
                  'Sales_to_Assets_Annual': "DescriptorRatios.DescriptorRatio.AnnualFrequency,"\
                      "numeratorProcess='extractLatest',denomProcess='average',denomDaysBack=3*365",
                  'Gross_Margin_Annual': "DescriptorRatios.DescriptorRatio.AnnualFrequency"}
    
    def __init__(self, connections, ds, assetData, numeraire):
        self.connections = connections
        self.marketDB = connections.marketDB
        self.modelDB = connections.modelDB
        self.ds = ds
        self.assetData = assetData
        self.numeraire = numeraire
        self.dsInstance = self.getdsInstance()
        if ds not in self.instanceDict.keys():
            logging.warning('%s is not included in DescriptorRatio decomposition', ds)
            return

    def getdsInstance(self):
        instance = self.instanceDict[self.ds]
        params = "(self.modelDB, self.marketDB, " + self.paramsDict[self.ds] + ")"
        dsInstance = eval(instance+params)
        return dsInstance

    def getNumerators(self, modelDate):
        startDt = modelDate - datetime.timedelta(self.dsInstance.numDaysBack)
        endDt = modelDate - datetime.timedelta(self.dsInstance.numEndDate)
        numerators = self.dsInstance.loadRawNumeratorData(self.dsInstance.numItem, modelDate, 
                                                          startDt, endDt, self.assetData, 
                                                          self.assetData.universe, 
                                                          self.numeraire.currency_id)
        if self.dsInstance.numItem == 'totalDebt':
            # if numItem is totalDebt, map to dicl for now. just for matching the fund code
            numItem = 'dicl'
        else:
            numItem = self.dsInstance.numItem
        return (numItem, (startDt, endDt), numerators)

    def getDenomerators(self, modelDate):
        startDt = modelDate - datetime.timedelta(self.dsInstance.denomDaysBack)
        endDt = modelDate - datetime.timedelta(self.dsInstance.denomEndDate)
        denomerators = self.dsInstance.loadRawDenominatorData(self.dsInstance.denomItem, modelDate, 
                                                              startDt, endDt, self.assetData, 
                                                              self.assetData.universe, 
                                                              self.numeraire.currency_id)
        return (self.dsInstance.denomItem, (startDt, endDt), denomerators)

    def getnumProcess(self):
        return self.dsInstance.numProcess

    def getdenomProcess(self):
        return self.dsInstance.denomProcess

class DescExpComposition:
    """A class decomposes descriptors exposures into the corresponding fundamental data.
       Decomposition is according to DescriptorExposures.py and only some of exposures
       are included in this class. 
       Currently, this class only supports descriptors exposures which consist of 1
       fund data and 1 estimate data at most.
       4 Dicts need to be set up manually in order to 
       decompose descriptors exposures, ie. ibesfundPairDict, compDict, daysBackDict, daysFWDict.
    """
    ibesfundPairDict = {'eps_median_ann': 'ibei', 'rev_median_ann': 'sale'}
    compDict = {'Est_Earnings_to_Price_12MFL_Quarterly': ['eps_median_ann', 'market_cap'],
                'Est_Earnings_to_Price_12MFL_Annual': ['eps_median_ann', 'market_cap'],
                'Earnings_Growth_RPF_AFQ': ['eps_median_ann', 'ibei'],
                'Sales_Growth_RPF_AFQ': ['rev_median_ann', 'sale'],
                'Est_Earnings_to_Price_Annual': ['eps_median_ann', 'market_cap'],
                'Earnings_Growth_RPF_Annual': ['eps_median_ann', 'ibei'],
                'Sales_Growth_RPF_Annual': ['rev_median_ann', 'sale']}
    
    daysBackDict = {'Est_Earnings_to_Price_12MFL_Quarterly': (2*366),
                    'Est_Earnings_to_Price_12MFL_Annual': (2*366),
                    'Earnings_Growth_RPF_AFQ': (5*366)+94,
                    'Sales_Growth_RPF_AFQ': (5*366)+94,
                    'Est_Earnings_to_Price_Annual': (2*366),
                    'Earnings_Growth_RPF_Annual': (5*366),
                    'Sales_Growth_RPF_Annual': (5*366)}
    daysFWDict = {'Est_Earnings_to_Price_12MFL_Quarterly': (3*366),
                  'Est_Earnings_to_Price_12MFL_Annual': (3*366),
                  'Earnings_Growth_RPF_AFQ': (2*366),
                  'Sales_Growth_RPF_AFQ': (2*366),
                  'Est_Earnings_to_Price_Annual': 366,
                  'Earnings_Growth_RPF_Annual': 366,
                  'Sales_Growth_RPF_Annual': 366}

    def __init__(self, connections, ds, assetData, numeraire):
        self.connections = connections
        self.marketDB = connections.marketDB
        self.modelDB = connections.modelDB
        self.ds = ds
        self.assetData = assetData
        self.numeraire = numeraire
        if self.ds not in self.compDict.keys():
            logging.warning('%s is not included in DescriptorExposure decomposition', ds)

    def getComposition(self, modelDate):
        comp = self.compDict[self.ds]
        startDt = modelDate - datetime.timedelta(self.daysBackDict[self.ds])
        endDt = modelDate + datetime.timedelta(self.daysFWDict[self.ds])
        result = list()
        fcItems = self.modelDB.getFundamentalItemCodes('sub_issue_esti_currency', self.marketDB)
        fundItems = self.modelDB.getFundamentalItemCodes('sub_issue_fund_currency', self.marketDB)
        fundItemsList = list()
        for i in fundItems.keys():
            fundItemsList.append(i[:-4])
        for i in comp:
            if i in fcItems:
                fcValues = self.getIBESdata(modelDate, startDt, endDt, i)
                result.append((fcValues, (startDt, endDt)))
            elif i in fundItemsList:
                requireConsecQtrData = 16
                values, valueFreq = self.modelDB.getMixFundamentalCurrencyItem(
                    i, startDt, modelDate, self.assetData.universe, modelDate,
                    self.marketDB, convertTo=self.numeraire.currency_id, splitAdjust=None, 
                    requireConsecQtrData=requireConsecQtrData)
                result.append((values, valueFreq, (startDt, modelDate)))
            elif i != 'market_cap':
                logging.error('No fundamental/estimate item in modelDB for %s', i)
                exit(1)
        return result

    def getIBESdata(self, modelDate, startDt, endDt, itemcode):
        """Get forcast values from IBES according to different descriptor exposures
        """
        if 'Annual' in self.ds:
            useFixedFrequency = DescriptorRatios.DescriptorRatio.AnnualFrequency
        else:
            useFixedFrequency = None
        fcValues = DescriptorExposures.process_IBES_data_all(
            modelDate, self.assetData.universe, startDt, endDt, self.modelDB,
            self.marketDB, itemcode, self.numeraire.currency_id, self.ibesfundPairDict[itemcode],
            dropOldEstimates=True, scaleByTSO=False, useFixedFrequency=useFixedFrequency)
        IBESdata = list()
        if (self.ds == 'Est_Earnings_to_Price_12MFL_Quarterly') or (
            self.ds == 'Est_Earnings_to_Price_12MFL_Annual'):
            for idx, est in enumerate(fcValues):
                eVals = [ev for ev in est if ev[0] > modelDate]
                currFYidx = None
                nextFYidx = None
                if len(eVals) >= 2:
                    if (eVals[0][0] - modelDate).days <= 366:
                        currFYidx = 0
                        dtDiff = numpy.array([(eVals[j][0] - eVals[currFYidx][0]).days for j in 
                                              range(1, len(eVals))])
                        index = numpy.where(numpy.logical_and(dtDiff >= 330, dtDiff <= 400))
                        if len(index) > 0:
                            if len(index[0]) > 0:
                                nextFYidx = index[0][0] + 1
                                if eVals[currFYidx][-1] <= eVals[nextFYidx][-1]:
                                    IBESdata.append([eVals[currFYidx], eVals[nextFYidx]])
                                else:
                                    nextFYidx = None
                elif len(eVals) == 1:
                    if (eVals[0][0] - modelDate).days <= 366:
                        currFYidx = 0
                if currFYidx is not None and nextFYidx is None:
                        IBESdata.append([eVals[currFYidx]])
                elif currFYidx is None and nextFYidx is None:
                    IBESdata.append([])
        elif self.ds == 'Est_Earnings_to_Price_Annual':
            fcValues = Utilities.extractLatestValuesAndDate(fcValues)
            for idx, est in enumerate(fcValues):
                if (est is None) or (est is ma.masked) or (len(est)<1):
                    IBESdata.append([])
                IBESdata.append([est])
        elif (self.ds == 'Earnings_Growth_RPF_Annual') or (self.ds == 'Sales_Growth_RPF_Annual'):
            values = self.modelDB.getFundamentalCurrencyItem(
                self.ibesfundPairDict[itemcode] + '_ann', startDt, modelDate,self.assetData.universe,
                modelDate, self.marketDB, convertTo=self.numeraire.currency_id)
            valueFreq = list()
            for idx in range(len(values)):
                valueFreq.append(DescriptorRatios.DataFrequency('_ann'))
            for idx, est in enumerate(fcValues):
                if (len(est) > 0) and (len(values[idx]) > 0):
                    latestDate = values[idx][-1][0]
                    targetDate = latestDate + datetime.timedelta(366)
                    fcIdx = bisect.bisect_left([n[0] for n in est], targetDate)
                    if fcIdx == len(est):
                        fcIdx -= 1
                    fcDate = est[fcIdx][0]
                    val = est[fcIdx][1]
                    if (fcDate > latestDate) and (fcDate < (latestDate + datetime.timedelta(400))):
                        IBESdata.append([est[fcIdx]])
                    else:
                        IBESdata.append([])
                else:
                    IBESdata.append([])
        elif (self.ds == 'Earnings_Growth_RPF_AFQ') or (self.ds == 'Sales_Growth_RPF_AFQ'):
            requireConsecQtrData = 16
            values, valueFreq = self.modelDB.getMixFundamentalCurrencyItem(
                self.ibesfundPairDict[itemcode], startDt, modelDate, self.assetData.universe,
                modelDate, self.marketDB, convertTo=self.numeraire.currency_id, splitAdjust=None,
                requireConsecQtrData=requireConsecQtrData)
            for idx, est in enumerate(fcValues):
                if (len(est) > 0) and (len(values[idx]) > 0):
                    latestDate = values[idx][-1][0]
                    targetDate = latestDate + datetime.timedelta(366)
                    if len(est) >= 2:
                        if 355 <= (est[0][0] - latestDate).days <= 366:
                            IBESdata.append([est[0]])
                        elif (est[0][0] > latestDate) and ((est[0][0] - latestDate).days < 355) and \
                                (est[1][0] > targetDate) and (355 <= (est[1][0] - est[0][0]).days \
                                                                  <= 366):
                            IBESdata.append([est[0], est[1]])
                        else:
                            IBESdata.append([])
                    elif len(est) == 1:
                        if 300 <= (est[0][0] - latestDate).days <= 400:
                            IBESdata.append([est[0]])
                        else:
                            IBESdata.append([])
                else:
                    IBESdata.append([])
        else:
            logging.warning('Cannot get IBES Data for %s', self.ds)
            for i in range(len(fcValues)):
                IBESdata.append([])
        return IBESdata
        
class DescriptorReport:
    """A class defines descriptors QA report
    """
    country_model_map = {'AU': 'AUAxioma2016MH',
                         'CA': 'CAAxioma2009MH',
                         'CN': 'CNAxioma2010MH',
                         'GB': 'GBAxioma2009MH',
                         'JP': 'JPAxioma2017MH',
                         'TW': 'TWAxioma2012MH',
                         'US': 'USAxioma2016MH'}
    DescriptorsMatDtMap = dict()
    reportContents = list()
    checkingUniverseDtMap = dict()
    issueFromDatesDtMap =dict()
    
    def __init__(self, connections, dt, rmg):
        logging.info('Initializng DescriptorReport for %s', rmg.description)
        self.connections = connections
        self.marketDB = connections.marketDB
        self.modelDB = connections.modelDB
        self.dt = dt
        if rmg.mnemonic not in self.country_model_map:
            logging.error('%s has not been set up. Report will not be generated.', rmg.mnemonic)
            exit(1)
        self.rmg= rmg
        self.prevdt = modelDB.getDates([rmg], dt, 30, excludeWeekend=True)[-2]
        self.riskModel = self.getRiskModelbyRMG(rmg)
        rmInfo = self.modelDB.getRiskModelInfo(self.riskModel.rm_id, self.riskModel.revision)
        if rmInfo.distribute == 0:
            logging.info('%s is not distributed. Report will not be generated.', 
                         self.riskModel.__class__.__name__)
            sys.exit(0)
        self.fundCodeMap = self.modelDB.getFundamentalItemCodes(
            'sub_issue_fund_currency', self.marketDB)
        self.estCodeMap = self.modelDB.getFundamentalItemCodes(
            'sub_issue_esti_currency', self.marketDB)
        self.TDescriptorKeywords = ['Trading_Activity','Momentum','Volatility','Market_Sensitivity',
                                    'Cap','XRate']
        if not hasattr(self.riskModel, 'DescriptorMap'):
            logging.error('%s has no descriptors. Report will not be generated.', 
                          self.riskModel.__class__.__name__)
            exit(1)
        
    def _loadCheckingUniverseDtMap(self, dtList):
        """Load checking universe into a date map cache given a date list
           If model universe is not available in rmi_universe, it will be extracted from rms_issue
           and active rmg assets. Otherwise, rmi_universe will be used.
           If dt is already in the cache, universe will not be loaded.
        """
        rmgUniverse = self.modelDB.getActiveSubIssues(self.rmg, self.dt, inModels=True)
        for dt in dtList:
            if dt in self.checkingUniverseDtMap:
                continue
            rmi = self.modelDB.getRiskModelInstance(self.riskModel.rms_id, dt)
            if rmi is not None:
                logging.info('Loading checking universe by rmi_universe for %s', str(dt))
                modelUniverse = self.modelDB.getRiskModelInstanceUniverse(rmi)
            else:
                logging.info('Loading checking universe by rms-issue for %s', str(dt))
                modelUniverse = AssetProcessor.getModelAssetMaster(self.riskModel, dt, 
                                                                   self.modelDB, self.marketDB)
            checkingUniverse = list(set(rmgUniverse).intersection(set(modelUniverse)))
            self.checkingUniverseDtMap[dt] = checkingUniverse

    def _loadPrevEstu(self):
        """Load the latest Estu checking universe, either report date or the day before.
           Estu checking universe consists intersection of active assets in the rmg and 
           assets in Estu in risk models from rmg-model-map 
        """
        self.riskModel.estuMap = self.modelDB.getEstuMappingTable(self.riskModel.rms_id)
        rmi = self.modelDB.getRiskModelInstance(self.riskModel.rms_id, self.dt)
        if rmi is None:
            rmi = self.modelDB.getRiskModelInstance(self.riskModel.rms_id, self.prevdt)
            if rmi is None:
                logging.error('No Estimation universe is available either on date or date - 1')
                exit(1)
        self.estu = [sid for sid in self.riskModel.loadEstimationUniverse(rmi, self.modelDB) if
                     sid in self.checkingUniverseDtMap[self.dt]]

    def _loadDescriptorsMatDtMap(self, dtList):
        """Load descriptors matrix into a date map cache given a date list.
           If dt is already in the cache, descriptors matrix will not be loaded.
        """
        for dt in dtList:
            if dt in self.DescriptorsMatDtMap:
                continue
            else:
                logging.info('Processing descriptors data matrix for %s', str(dt))
                universe = self.checkingUniverseDtMap[dt]
                descriptorMat = self._getDescriptorsMat(dt, universe)
                self.DescriptorsMatDtMap[dt] = descriptorMat
    
    def _getDescriptorsMat(self, dt, universe, rollOver=False):
        descriptors = []
        descriptorMat = Matrices.ExposureMatrix(universe)
        for dsList in self.riskModel.DescriptorMap.values():
            descriptors.extend(dsList)
        descriptors = list(set(descriptors))
        # ### Development code
        # descriptors = ['Est_Earnings_to_Price_12MFL_Quarterly']
        # ### Development code
        isoCode = self.rmg.getCurrencyCode(dt)
        currencyAssetMap = {isoCode: universe}
        descDict = dict(self.modelDB.getAllDescriptors())
        descValueDict, okDescriptorCoverageMap = self.riskModel.loadDescriptors(
            descriptors, descDict, dt, universe, self.modelDB,
            currencyAssetMap, rollOver=rollOver)
        for ds in descriptors:
            if ds in descValueDict:
                descriptorMat.addFactor(ds, descValueDict[ds],
                                        Matrices.ExposureMatrix.StyleFactor)
        return descriptorMat

    def loadDefaultCache(self):
        dtList = self.modelDB.getDates([self.rmg], self.dt, 19, excludeWeekend=True)
        self._loadCheckingUniverseDtMap(dtList)
#        self.issueFromDates = self.modelDB.loadIssueFromDates(dtList, self.universe)
        self._loadDescriptorsMatDtMap(dtList)
        self._loadPrevEstu()
        
    def populateReportContent(self, reportSection, reportName, colheader=None, rowheader=None,
                              content=None, description=None, reportType=None, position=None):
        data = Utilities.Struct()
        data.reportSection = reportSection
        data.reportName = reportName
        if description is not None:
            data.description = description
        if reportType is not None:
            data.reportType = reportType
        if content is not None:
            data.content = numpy.c_[rowheader, content]
            data.header = colheader
        if position is not None:
            data.position = position
        self.reportContents.append(data)
                
    def getRiskModelbyRMG(self, rmg, defaultModel='WWAxioma2011MH'):
        """Get risk model using rmg. RMG which have SCM will return its SCM.
           Other RMGs will return WW21MH by default
        """
        if rmg.mnemonic in self.country_model_map:
            riskModelClass = riskmodels.getModelByName(self.country_model_map[rmg.mnemonic])
        else:
            riskModelClass = riskmodels.getModelByName(defaultModel)
        riskModel = riskModelClass(self.modelDB, self.marketDB)
        return riskModel

    def getDualHistogram(self, data1, data2, binNo=100):
        """Given two data array and number of bins
           Return two histograms with the same bin sizes
        """
        minval = min(numpy.append(data1, data2))
        maxval = max(numpy.append(data1, data2))
        increment = (maxval - minval)/binNo
        bins = numpy.arange(minval-increment, maxval+increment, increment)
        hist1, bins1 = numpy.histogram(data1, bins=bins)
        hist2, bins2 =  numpy.histogram(data2, bins=bins)
        return (hist1, hist2, bins)

    def convertDtMapToDescMap(self, DescriptorsMatDtMap, restrict=None):
        """Convert descriptors matrix DtMap to assets time series matrix Descriptor Map 
        """
        logging.info('Converting descriptors values date map to assets TS matrix descriptors map')
        dtList = sorted(DescriptorsMatDtMap.keys()) 
        dtIdxMap = dict([(j, i) for (i, j) in enumerate(dtList)])
        fullUniverse = list()
        descriptorList = list()
        for i in DescriptorsMatDtMap.values():
            fullUniverse += i.getAssets()
        if restrict is None:
            universe = list(set(fullUniverse))
        else:
            universe = list(set(fullUniverse).intersection(set(restrict)))
        assetIdxMap = dict([(j, i) for (i, j) in enumerate(universe)])
        for dt in dtList:
            descriptorsMat = DescriptorsMatDtMap[dt]
            descriptors = descriptorsMat.getFactorNames()
            for ds in descriptors:
                if ds not in descriptorList:
                    descriptorList.append(ds)
        assetsTSMatDescMap = dict()
        for ds in descriptorList:
            assetsTSMat = Matrices.TimeSeriesMatrix(universe, dtList)
            for dt in dtList:
                descriptorsMat = DescriptorsMatDtMap[dt]
                dsIdx = descriptorsMat.getFactorIndex(ds)
                data = descriptorsMat.getMatrix()
                assets = descriptorsMat.getAssets()
                NoDSasset = list(set(universe)-set(assets))
                for idx, i in enumerate(NoDSasset):
                    # Mask any assets without Descriptor
                    assetIdx = assetIdxMap[i]
                    dtIdx = dtIdxMap[dt]
                    assetsTSMat.data[assetIdx, dtIdx] = ma.masked
                for idx, i in enumerate(assets):
                    if i not in assetIdxMap:
                        continue
                    value = data[dsIdx, idx]
                    assetIdx = assetIdxMap[i]
                    dtIdx = dtIdxMap[dt]
                    assetsTSMat.data[assetIdx, dtIdx] = value
            assetsTSMatDescMap[ds] = assetsTSMat
        return assetsTSMatDescMap

    def decomposeDescChange(self, ds, sids):
        """Decompose descriptor values change into the corresponding fundamental data change.
           The latest change in fundamental data for the past 7 days will be caught
        """
        logging.info('Decomposing %s change(s) for %d assets', ds, len(sids))
        nonRatiods = ['Est_Earnings_to_Price_12MFL_Quarterly', 'Earnings_Growth_RPF_AFQ',
                      'Sales_Growth_RPF_AFQ', 'Est_Earnings_to_Price_Annual',
                      'Earnings_Growth_RPF_Annual', 'Sales_Growth_RPF_Annual',
                      'Est_Earnings_to_Price_12MFL_Annual']
        if ds not in nonRatiods:
            self.decomposeDescRatioChange(ds, sids)
        else:
            self.decomposeDescExpChange(ds, sids)
            
    def decomposeDescRatioChange(self, ds, sids):
        # prevdt is the last 2 day which accounts for 1 day fund data delay
        prevdt = self.prevdt - datetime.timedelta(1)
        assetData = self.getbasicAssetData(self.dt, sids)
        dc = DescRatioComposition(self.connections, ds, assetData, self.riskModel.numeraire)
        PrevassetData = self.getbasicAssetData(prevdt, sids)
        Prevdc = DescRatioComposition(self.connections, ds, PrevassetData, self.riskModel.numeraire)
        # To Do: need enhancement here for divs_ann/qtr and totalDebt_ann/qtr
        (numcode, numperiod, nums) = dc.getNumerators(self.dt)
        (denomcode, denomperiod, denoms) = dc.getDenomerators(self.dt)
        (prevnumcode, prevnumperiod, prevnums) = Prevdc.getNumerators(prevdt)
        (prevdenomcode, prevdenomperiod, prevdenoms) = Prevdc.getDenomerators(prevdt)
        fundChgDict = self.getFundamentalChange(sids)

        content = list()
        fundChgcontent = list()
        for idx, sid in enumerate(sids):
            # Populate fundamental data changes in Descriptor computation
            modelID = sid.getModelID()
            num = nums[idx]
            prevnum = prevnums[idx]
            if prevnum.value != 0:
                numchange = str(round((num.value - prevnum.value)/abs(prevnum.value)*100, 2))
            else:
                numchange = '--'
            denom = denoms[idx]
            prevdenom = prevdenoms[idx]
            if prevdenom.value != 0:
                denomchange = str(round((denom.value - prevdenom.value)/abs(prevdenom.value)*100, 2))
            else:
                denomchange = '--'
            comp1 = numcode + num.frequency.suffix + ': ' + str(round(num.value, 1))
            prevcomp1 = numcode + prevnum.frequency.suffix + ': ' + str(round(prevnum.value, 1))
            if denomcode == 'market_cap':
                comp2 = denomcode + ': ' + str(round(denom.value/1000000, 2))
                prevcomp2 = denomcode + ': ' + str(round(prevdenom.value/1000000, 2))
                denomIsCap = True
            else:
                comp2 = denomcode + denom.frequency.suffix + ': ' + str(round(denom.value, 1))
                prevcomp2 = denomcode + prevdenom.frequency.suffix + ': ' + str(
                    round(prevdenom.value,1 ))
                denomIsCap = False
            content.append([modelID.getIDString(), dc.getnumProcess(), comp1, prevcomp1, numchange,
                            dc.getdenomProcess(), comp2, prevcomp2, denomchange])
            
            # Populate recent fundamental data change
            comp1chg = numcode + num.frequency.suffix + ': '
            sidStr = sid.getSubIDString()
            if sidStr not in fundChgDict:
                comp1chg += '--'
                diff1 = '--'
            elif (numcode + num.frequency.suffix) not in fundChgDict[sidStr]:
                comp1chg += '--'
                diff1 = '--'
            else:
                v1 = fundChgDict[sidStr][numcode + num.frequency.suffix][0]
                comp1chg += str(round(v1, 2))
                last = self.findHisFund(numcode + num.frequency.suffix, sidStr)
                if last[0] is not None:
                    comp1chg += ' / ' + str(last[0]) + ' (' + str(last[1].date()) + ')'
                    diff1 = str(round(((v1-last[0])/abs(last[0]))*100, 1))
                else:
                    comp1chg += ' / --'
                    diff1 = '--'

            if denomIsCap is False:
                comp2chg = denomcode + denom.frequency.suffix + ': ' 
                if sidStr not in fundChgDict:
                    comp2chg += '--'
                    diff2 = '--'
                elif (denomcode + denom.frequency.suffix) not in fundChgDict[sidStr]:
                    comp2chg += '--'
                    diff2 = '--'
                else:
                    v2 = fundChgDict[sidStr][denomcode + denom.frequency.suffix][0]
                    comp2chg += str(round(v2, 2))
                    last = self.findHisFund(denomcode + denom.frequency.suffix, sidStr)
                    if last[0] is not None:
                        comp2chg += ' / ' + str(last[0]) + ' (' + str(last[1].date()) + ')'
                        diff2 = str(round(((v2-last[0])/abs(last[0]))*100, 1))
                    else:
                        comp2chg += ' / --' 
                        diff2 = '--'
                fundChgcontent.append([modelID.getIDString(), comp1chg, diff1, comp2chg, diff2])
            else:
                fundChgcontent.append([modelID.getIDString(), comp1chg, diff1])
                                      
        if denomIsCap:
            denomheader = 'Comp 2'
            prevdenomheader = 'Prev Comp 2'
            fundChgheader = ['#', 'ModelID', 'Comp1', '%Change1']
        else:
            denomheader='Comp 2 (' + str(denomperiod[0])+' - '+str(denomperiod[1])+')'
            prevdenomheader='Prev Comp 2 ('+str(prevdenomperiod[0])+' - '+str(prevdenomperiod[1])+')'
            fundChgheader = ['#', 'ModelID', 'Comp1', '%Change1', 'Comp2', '%Change2']

        reportSection = "Major EstU assets with large change in fundamental descriptors"
        reportName = "Explaining Descriptor Jump for %s" % ds
        description = reportName
        colheader = ['#', 'ModelID', 'Comp 1 method',
                     'Comp 1 ('+str(numperiod[0])+' - '+str(numperiod[1])+')',
                     'Prev Comp 1 (' + str(prevnumperiod[0])+' - '+str(prevnumperiod[1])+')',
                     '%Change1', 'Comp 2 method', denomheader, prevdenomheader, '%Change2']
        rowheader = numpy.array([i+1 for i in range(len(sids))]).transpose()
        content = numpy.matrix(content)
        self.populateReportContent(reportSection, reportName, colheader, rowheader, content,
                                   description)
        reportName = "Recent Fundamental data change for %s" % ds
        description = reportName
        content = numpy.matrix(fundChgcontent)
        self.populateReportContent(reportSection, reportName, fundChgheader, rowheader,
                                   fundChgcontent, description)

    def decomposeDescExpChange(self, ds, sids):
        # prevdt is the last 2 day which accounts for 1 day fund data delay
        prevdt = self.prevdt - datetime.timedelta(1)
        assetData = self.getbasicAssetData(self.dt, sids)
        dc = DescExpComposition(self.connections, ds, assetData, self.riskModel.numeraire)
        PrevassetData = self.getbasicAssetData(prevdt, sids)
        Prevdc = DescExpComposition(self.connections, ds, PrevassetData, self.riskModel.numeraire)
        comps = dc.compDict[ds]
        compositionsData = dc.getComposition(self.dt)
        prevcompositionsData = Prevdc.getComposition(prevdt)
        values = None
        if len(compositionsData) == 1:
            fcValues, fcPeriod = compositionsData[0]
            prevfcValues, prevfcPeriod = prevcompositionsData[0]
        elif (len(compositionsData) == 2) and ('market_cap' in comps):
            fcValues, fcPeriod = compositionsData[0]
            prevfcValues, prevfcPeriod = prevcompositionsData[0]
        else:
            fcValues, fcPeriod = compositionsData[0]
            (values, valueFreq, valuesPeriod) = compositionsData[1]
            prevfcValues, prevfcPeriod = prevcompositionsData[0]
            (prevvalues, prevvalueFreq, prevvaluesPeriod) = prevcompositionsData[1]
        fundChgDict = self.getFundamentalChange(sids)
        estfundChgDict = self.getFundamentalChange(sids, includeEst=True)
            
        content = list()
        fundChgcontent = list()
        for idx, sid in enumerate(sids):
            # Populate fundamental data changes in Descriptor computation
            modelID = sid.getModelID()
            fcValue = fcValues[idx]
            prevfcValue = prevfcValues[idx]
            fcValueStr = comps[0] + ': '
            prevfcValueStr = comps[0] + ': '
            seperator = ''
            prevseperator = ''
            if len(fcValue) == 0:
                fcValueStr += '--'
            else:
                for i in fcValue:
                    fcValueStr += seperator + str(round(i[1], 2)) + ' (' + str(i[0]) + ')'
                    seperator = ' / '
            if len(prevfcValue) == 0:
                prevfcValueStr += '--'
            else:
                for i in prevfcValue:
                    prevfcValueStr += prevseperator + str(round(i[1], 2)) + ' (' + str(i[0]) + ')'
                    prevseperator = ' / '
            if values is None:
                mcap = 'market_cap: ' + str(round(assetData.issuerTotalMarketCaps[idx]/1000000, 2))
                prevmcap = 'market_cap: ' + str(round(
                        PrevassetData.issuerTotalMarketCaps[idx]/1000000, 2))
                content.append([modelID.getIDString(), fcValueStr, prevfcValueStr,
                                mcap, prevmcap])
            else:
                value = values[idx]
                prevvalue = prevvalues[idx]
                valueStr = comps[1] + valueFreq[idx].suffix + ': '
                prevvalueStr = comps[1] + prevvalueFreq[idx].suffix + ': ' 
                seperator = ''
                prevseperator = ''
                if len(value) == 0:
                    valueStr += '--'
                else:
                    for i in value[-2:]:
                        valueStr += seperator + str(i[1]) + ' (' + str(i[0]) + ')'
                        seperator = ' / '
                if len(prevvalue) == 0:
                    prevvalueStr += '--'
                else:
                    for i in prevvalue[-2:]:
                        prevvalueStr += prevseperator + str(i[1]) + ' (' + str(i[0]) + ')'
                        prevseperator = ' / '
                content.append([modelID.getIDString(), fcValueStr, prevfcValueStr,
                                valueStr, prevvalueStr])

        # Populate recent fundamental data change
            comp1chg = comps[0] + ': '
            sidStr = sid.getSubIDString()
            if sidStr not in estfundChgDict:
                comp1chg += '--'
                diff1 = "--"
            elif comps[0] not in estfundChgDict[sidStr]:
                comp1chg += '--'
                diff1 = "--"
            else:
                v1 = estfundChgDict[sidStr][comps[0]][0]
                comp1chg += str(round(v1, 2))
                last = self.findHisFund(comps[0], sidStr, EstMode=True)
                if last[0] is not None:
                    comp1chg += ' / ' + str(round(last[0], 2)) + ' (' + str(last[1].date()) + ')'
                    diff1 = str(round(((v1-last[0])/abs(last[0]))*100, 2))
                else:
                    comp1chg += ' / --'
                    diff1 = "--"
            if values is not None:
                comp2chg = comps[1] + valueFreq[idx].suffix + ': '
                if sidStr not in fundChgDict:
                    comp2chg += '--'
                    diff2 = "--"
                elif (comps[1] + valueFreq[idx].suffix) not in fundChgDict[sidStr]:
                    comp2chg += ' --'
                    diff2 = "/"
                else:
                    v2 = fundChgDict[sidStr][comps[1] + valueFreq[idx].suffix][0]
                    comp2chg += str(round(v2, 2))
                    last = self.findHisFund(comps[1] + valueFreq[idx].suffix, sidStr)
                    if last[0] is not None:
                        comp2chg += ' / ' + str(round(last[0], 2)) + ' (' + str(last[1].date()) + ')'
                        diff2 = str(round(((v2-last[0])/abs(last[0]))*100, 2))
                    else:
                        comp2chg += ' / --'
                        diff2 = "/"
                fundChgcontent.append([modelID.getIDString(), comp1chg, diff1, comp2chg, diff2])
            else:
                fundChgcontent.append([modelID.getIDString(), comp1chg, diff1])

        header1 = 'Comp 1 ('+str(fcPeriod[0])+' - '+str(fcPeriod[1])+')'
        header2 = 'Prev Comp 1 (' + str(prevfcPeriod[0])+' - '+str(prevfcPeriod[1])+')'
        if values is None:
            header3 = 'Comp 2'
            header4 = 'Prev Comp 2'
            fundChgheader = ['#', 'ModelID', 'Comp1', '%Change1']
        else:
            header3 = 'Comp 2 ('+str(valuesPeriod[0])+' - '+str(valuesPeriod[1])+')'
            header4 = 'Prev Comp 2 (' + str(prevvaluesPeriod[0])+' - '+str(prevvaluesPeriod[1])+')'
            fundChgheader = ['#', 'ModelID', 'Comp1', '%Change1', 'Comp2', '%Change2']

        reportSection = "Major EstU assets with large change in fundamental descriptors"
        reportName = "Explaining Descriptor Jump for %s" % ds
        description = reportName
        colheader = ['#', 'ModelID', header1, header2, header3, header4]
        rowheader = numpy.array([i+1 for i in range(len(sids))]).transpose()
        content = numpy.matrix(content)
        self.populateReportContent(reportSection, reportName, colheader, rowheader, content,
                                   description)
        reportName = "Recent Fundamental data change for %s" % ds
        description = reportName
        content = numpy.matrix(fundChgcontent)
        self.populateReportContent(reportSection, reportName, fundChgheader, rowheader,
                                   fundChgcontent, description)

    def getFundamentalChange(self, sids, includeEst=False):
        sidArgs = ['sid%d' % i for i in range(len(sids))]
        basequery = """
                    select SUB_ISSUE_ID, NAME, VALUE, CURRENCY_ID, EFF_DT,REV_DT
                    from marketdb_global.META_CODES code, SUB_ISSUE_FUND_CURRENCY fund
                    where code.CODE_TYPE = 'asset_dim_fund_currency:item_code'
                    and SUB_ISSUE_ID IN (%(sids)s)
                    and fund.ITEM_CODE = code.ID
                    and fund.EFF_DT <= fund.REV_DT
                    and fund.EFF_DT <=:date_arg supp
                    and fund.REV_DT >:date_arg0  and REV_DT <:date_arg2
                    order by SUB_ISSUE_ID,ITEM_CODE,DT , EFF_DT
                    """%{'sids':','.join([':%s' % arg for arg in sidArgs])}
        query = copy.deepcopy(basequery)
        query = query.replace('supp', '')
        dataDict = dict()
        dataDict['date_arg'] = self.dt
        dataDict['date_arg2'] = self.dt + datetime.timedelta(1)
        dataDict['date_arg0'] = self.prevdt
        dataDict.update(dict(zip(sidArgs, [i.getSubIDString() for i in sids])))
        self.modelDB.dbCursor.execute(query, dataDict)
        result = self.modelDB.dbCursor.fetchall()
        fundDict = dict()
        for r in result:
            fundDict.setdefault(r[0],dict()).update(dict([(r[1],[item for item in r[2:]])]))
        if includeEst:
            query = copy.deepcopy(basequery)
            dataDict['date_arg3'] = self.dt + datetime.timedelta(365)
            supp = " and dt <= :date_arg3"
            query = query.replace('asset_dim_fund_currency','asset_dim_esti_currency')
            query = query.replace('SUB_ISSUE_FUND_CURRENCY','SUB_ISSUE_ESTIMATE_DATA')
            query = query.replace('supp', supp)
            self.modelDB.dbCursor.execute(query, dataDict)
            result = self.modelDB.dbCursor.fetchall()
            for r in result:
                fundDict.setdefault(r[0],dict()).update(dict([(r[1],[item for item in r[2:]])]))
        return fundDict

    def findHisFund(self, item, sid, EstMode=False):
        if EstMode is True:
            table = 'SUB_ISSUE_ESTIMATE_DATA'
            itemCode = self.estCodeMap[item]
            # only get 1 year estimate
            dtConstraint = self.dt + datetime.timedelta(365)
            supp = " and dt < :dt_arg"
            myDict = dict([('code_arg', itemCode), ('id_arg', sid), ('dt_arg', dtConstraint)])
        else:
            table = 'sub_issue_fund_currency'
            itemCode = self.fundCodeMap[item]
            supp = ''
            myDict = dict([('code_arg', itemCode), ('id_arg', sid)])
        query = """ select eff_dt,value,dt from %s
                    where item_code =:code_arg
                    and sub_issue_id =:id_arg
                    and rev_del_flag = 'N' %s
                    order by eff_dt desc,dt desc""" %(table, supp)
        self.modelDB.dbCursor.execute(query, myDict)
        result = self.modelDB.dbCursor.fetchall()
        for idx, row in enumerate(result):
            if idx == 0:
                today = row[0]
            else:
                if EstMode is True:
                    if row[0]!= today:
                        last = [row[1], row[0]]
                        return last
                else:
                    if row[0]!= today and row[2].date()<= self.prevdt:
                        last = [row[1], row[0]]
                        return last
        return (None,None)

    def getbasicAssetData(self, dt, sids):
        assetData = AssetProcessor.process_asset_information(
            dt, sids, [self.rmg], self.modelDB, self.marketDB,
            checkHomeCountry=False, numeraire_id=self.riskModel.numeraire.currency_id,
            forceRun=True, legacyDates=False)
        numer = Utilities.Struct()
        numer.currency_id = self.riskModel.numeraire.currency_id
        AssetProcessor.computeTotalIssuerMarketCaps( 
            assetData, dt, numer, self.modelDB, self.marketDB)
        return assetData

    def filter_asset_by_cap(self, sids, lower_pctile=25, upper_pctile=100):
        data = self.getbasicAssetData(self.dt, sids)
        mcap = ma.array(data.marketCaps)
        sortedIdx = ma.argsort(-mcap)
        targetCapRatio = (upper_pctile - lower_pctile) / 100.0
        runningCapRatio = numpy.cumsum(ma.take(mcap, sortedIdx), axis=0)
        runningCapRatio /= ma.sum(ma.take(mcap, sortedIdx))
        reachedTarget = list(runningCapRatio >= targetCapRatio)
        m = min(reachedTarget.index(True)+1, len(sortedIdx))
        eligibleIdx = sortedIdx[:m]
        return [sids[i] for i in eligibleIdx]
        
    def getDescriptorCoverageStatByAsset(self, lookback=19):
        """Get historical descriptor coverage statistics by no. of assets
        """
        logging.info('Processing Coverage section: Descriptor Coverage by no. of assets')
        dtList = self.modelDB.getDates([self.rmg], self.dt, lookback, excludeWeekend=True)
        for dt in dtList:
            if dt not in self.checkingUniverseDtMap:
                self._loadCheckingUniverseDtMap([dt])
            if dt not in self.DescriptorsMatDtMap:
                self._loadDescriptorsMatDtMap([dt])

        resultList = list()
        coveragestat = list()
        descriptorList = list()
        for dt in dtList:
            coverageList = list()
            TDescriptors = list()
            universe = self.checkingUniverseDtMap[dt]
            descriptorsMat = self.DescriptorsMatDtMap[dt]
            descriptors = descriptorsMat.getFactorNames()
            descriptorList.append(descriptors)
            data = descriptorsMat.getMatrix()
            for keyword in self.TDescriptorKeywords:
                for d in descriptors:
                    if (keyword in d) and (d not in TDescriptors):
                        TDescriptors.append(d)
            descriptors = list(set(descriptors) - set(TDescriptors))
            for ds in descriptors:
                dsIdx = descriptorsMat.getFactorIndex(ds)
                dsValue =  data[[dsIdx],:]
                coverage = float(dsValue.count())/float(len(universe))*100
                coverageList.append(coverage)
            resultList.append(coverageList)

        assert len([list(i) for i in set(tuple(i) for i in descriptorList)])==1

        reportSection = "Coverage"
        # Coverage line chart
        reportName = "Descriptor Coverage in % by no. of assets for the last 20 days"
        description = reportName
        reportType = 'LineChart'
        colheader = ['Date'] + descriptors
        rowheader = numpy.array([dt.strftime('%b%d') for dt in dtList]).transpose()
        content = numpy.matrix(resultList)
        self.populateReportContent(reportSection, reportName, colheader, rowheader, content,
                                   description, reportType)
        # Coverage table
        reportName = "Daily change in Descriptor Coverage by no. of assets for the last 2 days"
        description = reportName
        colheader = ['Descriptor'] + [dt.strftime('%b%d') for dt in dtList[-2:]] + ['Difference in %']
        rowheader = numpy.array(descriptors).transpose()
        content = numpy.matrix(resultList).transpose()[:,-2:]
        lastChg = numpy.diff(content)[:,-1]
        content = numpy.c_[content, lastChg]
        self.populateReportContent(reportSection, reportName, colheader, rowheader, content,
                                   description)
        
    def getDescriptorCoverageStatByMcap(self, lookback=19):
        """Get historical descriptor coverage statistics by Mcap
        """
        logging.info('Processing Coverage section: Descriptor Coverage by Mcaps')
        dtList = self.modelDB.getDates([self.rmg], self.dt, lookback, excludeWeekend=True)
        for dt in dtList:
            if dt not in self.checkingUniverseDtMap:
                self._loadCheckingUniverseDtMap([dt])
            if dt not in self.DescriptorsMatDtMap:
                self._loadDescriptorsMatDtMap([dt])

        resultList = list()
        coveragestat = list()
        descriptorList = list()
        for dt in dtList:
            coverageList = list()
            TDescriptors = list()
            universe = self.checkingUniverseDtMap[dt]
            McapDts = self.modelDB.getDates([self.rmg], dt, 19, excludeWeekend=True)
            Mcaps = self.modelDB.getAverageMarketCaps(McapDts, universe, 
                                                      self.riskModel.numeraire.currency_id,
                                                      self.marketDB)
            descriptorsMat = self.DescriptorsMatDtMap[dt]
            descriptors = descriptorsMat.getFactorNames()
            descriptorList.append(descriptors)
            data = descriptorsMat.getMatrix()
            for keyword in self.TDescriptorKeywords:
                for d in descriptors:
                    if (keyword in d) and (d not in TDescriptors):
                        TDescriptors.append(d)
            descriptors = list(set(descriptors)- set(TDescriptors))
            for ds in descriptors:
                dsIdx = descriptorsMat.getFactorIndex(ds)
                dsValue =  data[[dsIdx],:]
                coveredAssetsIdx = numpy.flatnonzero(dsValue)
                coveredMcaps = [Mcaps[idx] for idx in coveredAssetsIdx]
                coverage = float(sum(coveredMcaps))/float(sum(Mcaps))*100
                coverageList.append(coverage)
            resultList.append(coverageList)

        assert len([list(i) for i in set(tuple(i) for i in descriptorList)])==1
        
        reportSection = "Coverage"
        # Coverage line chart
        reportName = "Descriptor Coverage in % by Mcap for the last 20 days"
        description = reportName
        reportType = 'LineChart'
        colheader = ['Date'] + descriptors
        rowheader = numpy.array([dt.strftime('%b%d') for dt in dtList]).transpose()
        content = numpy.matrix(resultList)
        self.populateReportContent(reportSection, reportName, colheader, rowheader, content,
                                   description, reportType)
        # Coverage table
        reportName = "Daily change in Descriptor Coverage by Mcap for the last 2 days"
        description = reportName
        colheader = ['Descriptor']+[dt.strftime('%b%d') for dt in dtList[-2:]]+['Difference in %']
        rowheader = numpy.array(descriptors).transpose()
        content = numpy.matrix(resultList).transpose()[:,-2:]
        lastChg = numpy.diff(content)[:,-1]
        content = numpy.c_[content, lastChg]
        self.populateReportContent(reportSection, reportName, colheader, rowheader, content,
                                   description)

    def getMissingDescriptor(self):
        """Get assets which have no technical descriptors populated at the report date
        """
        logging.info('Processing Coverage section: Missing Technical Descriptors')
        TDescriptorKeywords = ['Trading_Activity','Momentum','Volatility','Market_Sensitivity',
                               'Cap','XRate']
        TDescriptors = list()
        missingList = list()
        universe = self.checkingUniverseDtMap[self.dt]
        assetIdxMap = dict([(j, i) for (i, j) in enumerate(universe)])
        descriptorsMat = self.DescriptorsMatDtMap[self.dt]
        data = descriptorsMat.getMatrix()
        descriptors = descriptorsMat.getFactorNames()

        for keyword in TDescriptorKeywords:
            for d in descriptors:
                if keyword in d:
                    TDescriptors.append(d)
        TDescriptors= list(set(TDescriptors))
        
        for td in TDescriptors:
            tdIdx = descriptorsMat.getFactorIndex(td)
            tdValue =  data[[tdIdx],:]
            missingIdx = numpy.flatnonzero(ma.getmaskarray(tdValue))
            for idx in missingIdx:
                missingList.append([td, universe[idx].getModelID().getIDString()])

        reportSection = "Coverage"
        reportName = "Missing technical descriptors"
        if len(missingList) == 0:
            description = "No assets with missing technical descriptors"
            self.populateReportContent(reportSection, reportName, description=description)
        else:
            description = "%d asset(s) with missing technical descriptors" % (len(missingList)) 
            colheader = ['Descriptor', 'ModelID']
            rowheader = numpy.matrix(missingList, dtype=str)[:,0]
            content = numpy.matrix(missingList)[:,1]
            self.populateReportContent(reportSection, reportName, colheader, rowheader, content,
                                       description)

    def getMissingEstuDescriptor(self):
        """Get Estu assets which have no descriptors populated at the report date
        """
        logging.info('Processing Coverage section: Missing Estu Descriptors')
        missingList = list()
        universe = self.checkingUniverseDtMap[self.dt]
        assetIdxMap = dict([(j, i) for (i, j) in enumerate(universe)])
        estuIdx = [assetIdxMap[i] for i in self.estu if i in assetIdxMap]
        descriptorsMat = self.DescriptorsMatDtMap[self.dt]
        data = descriptorsMat.getMatrix()
        estuData = data[:, estuIdx]
        descriptors = descriptorsMat.getFactorNames()
        
        for ds in descriptors:
            dsIdx = descriptorsMat.getFactorIndex(ds)
            dsValue =  estuData[[dsIdx],:]
            missingIdx = numpy.flatnonzero(ma.getmaskarray(dsValue))
            for idx in missingIdx:
                missingList.append([ds, universe[idx].getModelID().getIDString()])
                
        reportSection = "Coverage"
        reportName = "Estu assets with missing descriptors"
        if len(missingList) == 0:
            description = "No Estu assets with missing descriptors"
            self.populateReportContent(reportSection, reportName, description=description)
        else:
            description = "%d Estu asset(s) with missing descriptors" % (len(missingList))
            colheader = ['Descriptor', 'ModelID']
            rowheader = numpy.matrix(missingList, dtype=str)[:,0]
            content = numpy.matrix(missingList)[:,1]
            self.populateReportContent(reportSection, reportName, colheader, rowheader, content,
                                       description)

    def getDescriptorDistribution(self):
        logging.info('Processing Distribtion section: Descriptors Distribution')
        descriptorsMat1 = self.DescriptorsMatDtMap[self.dt]
        descriptorsMat2 = self.DescriptorsMatDtMap[self.prevdt]
        data1 = descriptorsMat1.getMatrix()
        data2 = descriptorsMat2.getMatrix()
        descriptors = descriptorsMat1.getFactorNames()
        assert (descriptorsMat1.getFactorNames())==(descriptorsMat2.getFactorNames())
        position = '1_2'
        prevpostion = '2_2'

        for ds in descriptors:
            dsIdx = descriptorsMat1.getFactorIndex(ds)
            dsValue1 = data1[[dsIdx],:]
            dsValue2 = data2[[dsIdx],:]
            logging.info('Filtering %s for distribution', ds)
            (dsValue1, bounds1) = LegacyUtilities.mad_dataset(dsValue1, -8, 8, axis=1, treat='mask')
            (dsValue2, bounds2) = LegacyUtilities.mad_dataset(dsValue1, -8, 8, axis=1, treat='mask')
            goodIdx1 = numpy.flatnonzero(ma.getmaskarray(dsValue1)==0)
            goodIdx2 = numpy.flatnonzero(ma.getmaskarray(dsValue2)==0)
            goodVal1 = dsValue1[0, goodIdx1]
            goodVal2 = dsValue2[0, goodIdx2]
            (hist1, hist2, bins) = self.getDualHistogram(goodVal1, goodVal2, 100)
            reportSection = "Distribution"
            reportName = "Descriptor Distribution: %s" %(ds)
            description = reportName
            reportType = 'AreaChart'
            if prevpostion == '1_2':
                position = '2_2'
                prevpostion = '2_2'
            else:
                position = '1_2'
                prevpostion = '1_2'
            colheader = [ds, str(self.dt), str(self.prevdt)]
            rowheader = list()
            for idx, i in enumerate(bins[1:]):
                rowheader.append((bins[idx] + i)/2)
            content = numpy.row_stack((hist1, hist2)).transpose()
            self.populateReportContent(reportSection, reportName, colheader, rowheader, content,
                                       description, reportType, position)

    def getDescriptorCorr(self, lookback=19):
        """Get descriptor churn statistics by estu 
        """
        logging.info('Processing Distribtion section: Descriptors Churn Stat')
        dtList = self.modelDB.getDates([self.rmg], self.dt, lookback, excludeWeekend=True)
        for dt in dtList:
            if dt not in self.checkingUniverseDtMap:
                self._loadCheckingUniverseDtMap([dt])
            if dt not in self.DescriptorsMatDtMap:
                self._loadDescriptorsMatDtMap([dt])
        
        resultList = list()
        descriptorList = list()
        for dtIdx, dt in enumerate(dtList[1:]):
            universe = self.checkingUniverseDtMap[dt]
            prevuniverse = self.checkingUniverseDtMap[dtList[dtIdx]]
            assetIdxMap = dict([(j, i) for (i, j) in enumerate(universe)])
            prevassetIdxMap =  dict([(j, i) for (i, j) in enumerate(prevuniverse)])
            commomestu = [i for i in self.estu if (i in assetIdxMap) and (i in prevassetIdxMap)]
            estuIdx = [assetIdxMap[i] for i in commomestu]
            prevestuIdx = [prevassetIdxMap[i] for i in commomestu]
            descriptorsMat = self.DescriptorsMatDtMap[dt]
            prevdescriptorsMat = self.DescriptorsMatDtMap[dtList[dtIdx]]
            data = descriptorsMat.getMatrix()
            prevdata = prevdescriptorsMat.getMatrix()
            estuData = data[:, estuIdx]
            prevestuData = prevdata[:, prevestuIdx]
            descriptors = descriptorsMat.getFactorNames()
            descriptorList.append(descriptors)
            corrList = list()
            for ds in descriptors:
                dsIdx = descriptorsMat.getFactorIndex(ds)
                dsValue1 = estuData[[dsIdx],:]
                dsValue2 = prevestuData[[dsIdx],:]
                mask1 = ma.getmaskarray(dsValue1)
                mask2 = ma.getmaskarray(dsValue2)
                mask = mask1 | mask2
                goodIdx = numpy.flatnonzero(mask==0)
                goodVal1 = dsValue1[0, goodIdx]
                goodVal2 = dsValue2[0, goodIdx]
                corrMat = Utilities.compute_covariance(numpy.vstack((goodVal1, goodVal2)), axis=1, 
                                                    corrOnly=True)
                corrList.append(corrMat[0,1])
            resultList.append(corrList)

        assert len([list(i) for i in set(tuple(i) for i in descriptorList)])==1

        reportSection = "Descriptor Stability"
        reportName = "Descriptor stability for Estu assets for the last 20 days"
        description = reportName
        reportType = 'LineChart'
        colheader = ['Date'] + descriptorList[0]
        rowheader = numpy.array([dt.strftime('%b%d') for dt in dtList[1:]]).transpose()
        content = numpy.matrix(resultList)
        self.populateReportContent(reportSection, reportName, colheader, rowheader, content,
                                   description, reportType)

    def getDescriptorJump(self, lookback=60):
        """Get descriptor jump. Residual beyond 30MAD AND descriptors jump over 100% will be catched
        """
        logging.info('Processing Coverage section: Descriptors jump')
        dtList = self.modelDB.getDates([self.rmg], self.dt, lookback, excludeWeekend=True)
        for dt in dtList:
            if dt not in self.DescriptorsMatDtMap:
                self._loadCheckingUniverseDtMap([dt])
                self._loadDescriptorsMatDtMap([dt])
        descriptorsMat = self.DescriptorsMatDtMap[self.dt]
        descriptors = descriptorsMat.getFactorNames()
        tmpdescriptors = list(descriptors)
        for ds in descriptors:
            for keyword in self.TDescriptorKeywords:
                if keyword in ds:
                    tmpdescriptors.remove(ds)
        descriptors = tmpdescriptors
        # only look into those large cap assets
        FilteredSids = self.filter_asset_by_cap(self.estu, lower_pctile=20, upper_pctile=100)
        assetsTSMatDescMap = self.convertDtMapToDescMap(self.DescriptorsMatDtMap, 
                                                        restrict=FilteredSids)

        # Set some threshold here
        returnthreshold = 1
        diffthreshold = 0.01
        MADbound = 30
        for ds in descriptors:
            resultList = list()
            residualList = list()
            outlierIdx = list()
            returnJumpIdx = list()
            largeDiffIdx = list()
            assetsTSMat = assetsTSMatDescMap[ds]
            assetIdxMap = dict([(j,i) for (i,j) in enumerate(assetsTSMat.assets)])
            rpy.r.library('stats')
            logging.info('Running time series decomposition for descriptor: %s', ds)
            for idx, i in enumerate(assetsTSMat.assets):
                assetTS = assetsTSMat.data[idx,:]
                if (assetTS[-1] is not ma.masked) and (assetTS[-2] is not ma.masked) and (
                    abs(assetTS[-1]-assetTS[-2]) >= diffthreshold):
                    largeDiffIdx.append(idx)
                if assetTS[-2] == 0:
                    returnJumpIdx.append(idx)
                elif (assetTS[-1] is not ma.masked) and (assetTS[-2] is not ma.masked) and (
                    abs((assetTS[-1]-assetTS[-2])/assetTS[-2]) >= returnthreshold):
                    returnJumpIdx.append(idx)
                if float(len(numpy.flatnonzero(ma.getmaskarray(assetTS)==0)))/len(dtList) < 0.8:
                    residualList.append([float('nan')]*len(dtList))
                else:
                    rpy.set_default_mode(rpy.NO_CONVERSION)
                    ts = rpy.r.ts(data=assetTS, frequency=2)
                    res = rpy.r.decompose(ts, type="additive")
                    rpy.set_default_mode(rpy.BASIC_CONVERSION)
                    residualList.append(res.as_py()['random'])
            residualTSMat = numpy.matrix(residualList)
            residualTSMat = ma.masked_invalid(residualTSMat)
            latestresidual = ma.masked_invalid(numpy.array(residualTSMat[:,-2]))
            invalidIdx = numpy.flatnonzero(ma.getmaskarray(latestresidual))
            residualList = list()
            logging.info('MAD Filtering residuals for descriptor: %s', ds)
            for h in logging.root.handlers:
                h.setLevel(logging.WARNING)
            for idx, i in enumerate(assetsTSMat.assets):
                residualTS = ma.masked_invalid(numpy.array(residualTSMat[idx,:]))
                (residualTS, bounds) = LegacyUtilities.mad_dataset(residualTS, -MADbound, MADbound, axis=1,
                                                             treat='mask')
                if (residualTS.mask[-2]) and (idx not in invalidIdx):
                    outlierIdx.append(idx)
            for h in logging.root.handlers:
                h.setLevel(logging.INFO)

            outlierIdx = list(set(largeDiffIdx) & set(outlierIdx) & set(returnJumpIdx))
            # ### Development code
            # print len(returnJumpIdx)
            # outlierIdx = returnJumpIdx
            # #outlierIdx =[0]
            # ### Development code
            content = list()

            for idx in outlierIdx:
                sid = assetsTSMat.assets[idx]
                modelID = sid.getModelID()
                sedol = str(self.modelDB.getIssueSEDOLs(self.dt,[modelID],self.marketDB)[modelID])
                ticker = str(self.modelDB.getIssueTickers(self.dt,[modelID],self.marketDB)[modelID])
                name = self.modelDB.getIssueNames(self.dt, [modelID], self.marketDB)[modelID]
                mcap = round(self.getlastestMcap([assetsTSMat.assets[idx]])[0]/1000000,2)
                curr = assetsTSMat.data[idx,-1]
                prev = assetsTSMat.data[idx,-2]
                if prev != 0:
                    change = round((curr-prev)/abs(prev)*100, 2)
                else:
                    change = '--'
                self.getFundamentalChange([sid], includeEst=True)
                if sid.getSubIDString() in self.getFundamentalChange([sid], includeEst=True):
                    FundUpdate = 1
                else:
                    FundUpdate = 0
                content.append([modelID.getIDString(), sedol, ticker, name, mcap, 
                                curr, prev, change, FundUpdate])
                
            if len(outlierIdx) > 0:
                reportSection = "Major EstU assets with large change in fundamental descriptors"
                reportName = "%d EstU asset(s) have large descriptor change in %s " % (
                    len(outlierIdx), ds)
                description = reportName
                colheader = ['#', 'ModelID', 'Sedol', 'Ticker', 'Name', 'MCap m', 
                             'Curr' ,'Prev', '% Change', 'FUND UPD?']
                rowheader = numpy.array([i+1 for i in range(len(outlierIdx))]).transpose()
                content = sorted(content, key=itemgetter(4), reverse=True)
                decomposeAssets = [ModelDB.SubIssue(string=i[0]+'11') for i in content]
                content = numpy.matrix(content)
                self.populateReportContent(reportSection, reportName, colheader, rowheader, content,
                                           description)
                self.decomposeDescChange(ds, decomposeAssets)

if __name__ == '__main__':
    usage = "usage: %prog <config-file> <Country> <YYYY-MM-DD> [<YYYY-MM-DD]> [options]"
    cmdlineParser = optparse.OptionParser(usage=usage)
    cmdlineParser.add_option("--report-file", action="store",
                             default=None, dest='reportFile',
                             help='report file name')
    Utilities.addModelAndDefaultCommandLine(cmdlineParser)
    (options, args) = cmdlineParser.parse_args()
    Utilities.processDefaultCommandLine(options, cmdlineParser)
    fileName=options.reportFile

    if len(args)<3 and len(args)>4:
        cmdlineParser.error("Incorrect number of arguments")

    filename=options.reportFile
    configFile = open(args[0])
    config = configparser.ConfigParser()
    config.read_file(configFile)
    configFile.close()

    connections = Connections.createConnections(config)
    marketDB = connections.marketDB
    modelDB = connections.modelDB

    country = args[1]
    rmg = modelDB.getRiskModelGroupByISO(country)
    
    startDate = Utilities.parseISODate(args[2])
    if len(args) == 3:
        endDate = startDate
    else:
        endDate = Utilities.parseISODate(args[3])
    dates = modelDB.getDateRange([rmg], startDate, endDate, excludeWeekend=True)

    for dt in dates:
        if filename == None:
            filename = '%sDescriptorReport.%04d%02d%02d' % (
                rmg.mnemonic, dt.year, dt.month, dt.day)
        i = 0
        path = os.path.dirname(filename)
        if not os.path.exists(path) and path != '':
            os.makedirs(path)
        if os.path.isfile(filename + '.html.v' + str(i) ):
            logging.info(filename  + '.html.v' + str(i) + " already exists")
            i = 1
            while os.path.isfile(filename + '.html.v' + str(i) ):
                logging.info(filename + '.html.v' + str(i) + " already exists")
                i = i + 1
        htmlName = filename + '.html.v' + str(i) 

        reporter = DescriptorReport(connections, dt, rmg)
        reporter.loadDefaultCache()
        reporter.getDescriptorCoverageStatByAsset()
        reporter.getDescriptorCoverageStatByMcap()
        reporter.getMissingDescriptor()
#        reporter.getMissingEstuDescriptor()
        reporter.getDescriptorDistribution()
        reporter.getDescriptorCorr()

        header = 'Descriptor QA Report'
        visualizer = VisualTool.visualizer(htmlName, reporter.reportContents, dt, 
                                           reportHeader = header, displayThought=False)
        visualizer.visualize()

    marketDB.finalize()
    modelDB.finalize()
    sys.exit(0)
