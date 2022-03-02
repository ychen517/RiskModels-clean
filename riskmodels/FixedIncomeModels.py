
import pandas
import logging
import numpy
import numpy.ma as ma
import riskmodels
from riskmodels import LegacyUtilities as Utilities
from riskmodels import MFM
from riskmodels import RiskCalculator
from riskmodels.Factors import ModelFactor
from riskmodels.ModelDB import ModelSubFactor
from riskmodels.CurrencyRisk import ModelCurrency
from riskmodels.ModelDB import SubIssue

class FIAxioma2014MH(MFM.RegionalFundamentalModel):
    rm_id = 2000
    revision = 1
    rms_id = 20000
    xrBnds = [-15.0, 15.0]


    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('FIModels.FIAxioma2014MH')
        self.specific = False
        self.modelDB = modelDB
        self.marketDB = marketDB
        self.factors = []
        self.factorNames = []
        self.subFactors = {}
        self.rmInfo = modelDB.getRiskModelInfo(self.rm_id, self.revision)
        self.rms_id = self.rmInfo.serial_id
        self.name = self.rmInfo.name
        self.description = self.rmInfo.description
        self.mnemonic = self.rmInfo.mnemonic
        self.rmg = [modelDB.getRiskModelGroup(rmg.rmg_id) \
                    for rmg in self.rmInfo.rmgTimeLine]
        self.rmgTimeLine = self.rmInfo.rmgTimeLine
        self.specific = False
        self.currencyModel = riskmodels.getModelByName('FXAxioma2010USD')(modelDB, marketDB)
        self.log.info('Initializing %s (%s)', self.description, self.rmInfo.numeraire)
        
    def getRiskModelInstance(self, date, modelDB):
        
        rmi =  modelDB.getRiskModelInstance(self.rms_id, date)
        print('rmi', rmi)
        return rmi
    
    def setFactorsForDate(self, date, modelDB):
        self.factors = []
        self.factorNames = []
        q = """SELECT f.name, f.description
        FROM rms_factor rf JOIN factor f ON f.factor_id=rf.factor_id
            JOIN sub_factor sf on f.factor_id = sf.factor_id
        WHERE rf.rms_id=:rms_id
        and sf.from_dt <= :date_arg AND sf.thru_dt > :date_arg
        ORDER BY f.name
        """
        modelDB.dbCursor.execute(q, rms_id=self.rms_id, date_arg=date)
        for r in modelDB.dbCursor.fetchall():
            self.factors.append(ModelFactor(r[0], r[1]))
            self.factorNames.append(r[0])
        dbFactors = modelDB.getRiskModelSerieFactors(self.rms_id)
        self.nameFactorMap = dict([(f.name, f) for f in dbFactors])
        for f in self.factors:
            f.factorID = self.nameFactorMap[f.name].factorID
            
        self.setRiskModelGroupsForDate(date)
        
        # Create country and currency factors
        self.countries = [ModelFactor(r.description, None)
                           for r in self.rmg]
        self.currencies = [ModelFactor(f, None)
                           for f in set([r.currency_code for r in self.rmg])]

#        self.validateFactorStructure(date, warnOnly = True)

    def setCalculators(self):
        # Set up risk parameters
#         defaultStatisticalCovarianceParameters(self, nwLag=0, dva=None)
#         defaultFundamentalCovarianceParameters(self, nwLag=0, dva=None)
        riskmodels.defaultFundamentalCovarianceParameters(self, nwLag=2, overrider=False)

    def transferReturnsforAxIDList(self, date, rmi, modelDB, marketDB, axiomaIDList):
        self.setFactorsForDate(date, modelDB)
        self.log.debug('getting returns from marketDB for %s' % str(date))          
        returns = marketDB.getFIReturnsforAxIDList(date, axiomaIDList)
        self.subFactors = modelDB.getRiskModelInstanceSubFactors( rmi, self.factors)
        sfList = []
        returnList = []
        for sf in self.subFactors:
            if sf.factor.name not in returns:
                self.log.debug('no returns for %s on %s, skipping' % (sf.factor.name, str(date)))
                continue
            sfList .append(sf)
            returnList.append(returns[sf.factor.name])            

        if len(sfList) > 0:
            modelDB.deleteFactorReturns(rmi, sfList)
            self.log.info('done deleting %d returns', len(sfList))
            modelDB.insertFactorReturns(self.rms_id, date, sfList, returnList)
            self.log.info('done inserting returns for %s' % str(date))   

                
    def transferReturns(self, date, rmi, options, modelDB, marketDB):
        self.setFactorsForDate(date, self.modelDB)
        self.log.info('deleting returns for %s' % str(date))        
        self.modelDB.deleteFactorReturns(rmi)
        self.log.info('done deleting returns for %s' % str(date))        
        if options.testOnly:
            self.log.info('Reverting changes')
            #self.modelDB.revertChanges()
        else:
            self.modelDB.commitChanges()   
        self.log.info('getting returns from marketDB for %s' % str(date))          
        returns = self.marketDB.getFIReturns(date, self.specific)
        self.log.info('done getting returns from marketDB for %s' % str(date)) 
        if len(self.subFactors) == 0: 
            self.log.info('getting subFactors')   
            self.subFactors = self.modelDB.getRiskModelInstanceSubFactors(
                                                                     rmi, self.factors)
            self.log.info('done getting subFactors')   
        sfList = []
        returnList = []
        for sf in self.subFactors:
            if sf.factor.name not in returns:
                self.log.info('no returns for %s on %s, skipping' % (sf.factor.name, str(date)))
                continue
            sfList .append(sf)
            returnList.append(returns[sf.factor.name])            
                
#        subFactorMap = dict(zip([f.name for f in self.factors], subFactors))
        self.log.info('inserting returns for %s' % str(date))       
        self.modelDB.insertFactorReturns(self.rms_id, date, sfList, returnList)
        rmi.setHasReturns(True, self.modelDB)
        self.log.info('done inserting returns for %s' % str(date))   

    def computeCov(self, date, modelDB, marketDB):
        """Compute the covariance matrix.
        """
        self.log.debug('computeCov: begin')
        self.numeraire = ModelCurrency(self.rmInfo.numeraire)
#         self.numeraire.currency_id = \
#                     self.marketDB.getCurrencyID(rmInfo.numeraire, date)        
#         self.modelDB.createCurrencyCache(marketDB)
        rmi=modelDB.getRiskModelInstance(self.rms_id,date)
        if rmi == None:
            raise LookupError('No risk model instance for %s' % str(date))
        if not rmi.has_returns:
            raise LookupError('Returns are missing in risk model instance on %s' % str(date))
        self.setFactorsForDate(date, modelDB)
        if len(self.subFactors) == 0: 
            self.log.info('getting subFactors')  
#             self.subFactors = self.modelDB.getSubFactorsForDate(date, self.factors) 
            self.subFactors = modelDB.getRiskModelInstanceSubFactors(
                                                                     rmi, self.factors)
            self.log.info('done getting subFactors')   
        # Get some basic risk parameters
        self.setCalculators()
        if isinstance(self.covarianceCalculator,
                RiskCalculator.CompositeCovarianceMatrix2009):
            (minVarObs, maxVarObs) = self.vp.getCovarianceSampleSize()
            (minCorrObs, maxCorrObs) = self.cp.getCovarianceSampleSize()
            (minOmegaObs, maxOmegaObs) = (max(minVarObs, minCorrObs), max(maxVarObs, maxCorrObs))
            (minDeltaObs, maxDeltaObs) = self.sp.getCovarianceSampleSize()
        else:
            (minOmegaObs, maxOmegaObs) = self.rp.getCovarianceSampleSize()
            (minDeltaObs, maxDeltaObs) = self.rp.getSpecificRiskSampleSize()
#         if isinstance(self.covarianceCalculator,
#                 RiskCalculator.CompositeCovarianceMatrix2009):
# #                 RiskCalculator.CompositeCovarianceMatrix2012):
#             (minOmegaObs, maxOmegaObs) = self.cp.getCovarianceSampleSize()
#         else:
# #             (minOmegaObs, maxOmegaObs) = self.rp.getCovarianceSampleSize()
#             (minOmegaObs, maxOmegaObs) = self.cp.getCovarianceSampleSize()        
# #        (minOmegaObs, maxOmegaObs) = (250, 250)
#         
#         # Deal with weekend dates (well, ignore them)
        dateList = modelDB.getDates(self.rmg, date, maxOmegaObs-1, 
                                    excludeWeekend=True)
        dateList.reverse()
        dateList = dateList[:maxOmegaObs]
        
        # Sanity check -- make sure we have all required data
        rmiList = modelDB.getRiskModelInstances(self.rms_id, dateList)
        okDays = [i.date == j and i.has_returns for (i,j)
                in zip(rmiList, dateList)]
        okDays.append(False)
        firstBadDay = okDays.index(False)
        dateList = dateList[:firstBadDay]
        if len(dateList) < minOmegaObs:
            self.log.warning('%d missing risk model instances for required days',
                    minOmegaObs - len(dateList))
            raise LookupError(
                    '%d missing risk model instances for required days'
                    % (minOmegaObs - len(dateList)))
        if len(dateList) < maxOmegaObs:
            self.log.info('Using only %d of %d days of factor return history',
                    len(dateList), maxOmegaObs)
        
#        if len(self.subFactors) == 0: 
#            self.log.info('getting subFactors')   
#            self.subFactors = self.modelDB.getRiskModelInstanceSubFactors(
#                                                                     rmi, self.factors)
#            self.log.info('done getting subFactors')   

        # Remove dates for which many markets are non-trading
        minMarkets = int(0.5 * len(self.countries))
        datesAndMarkets = modelDB.getActiveMarketsForDates(
                                self.rmg, dateList[-1], dateList[0])
        datesAndMarkets = [(d,n) for (d,n) in datesAndMarkets \
                           if d.weekday() <= 4]
        

        # Remember, datesAndMarkets is in chron. order whereas dateList is reversed
        badDatesIdx = [len(dateList)-i-1 for (i,n) in \
                        enumerate(datesAndMarkets) if n[1] <= minMarkets]
        badDates = numpy.take(dateList, badDatesIdx)
        self.log.info('Removing dates with < %d markets trading: %s',
                      minMarkets, ','.join([str(d) for d in badDates]))
        goodDatesIdx = [i for i in range(len(dateList)) if i not in badDatesIdx]
        dateList = numpy.take(dateList, goodDatesIdx)
       
        cr = modelDB.loadFactorReturnsHistory(
                                self.rms_id, self.subFactors, dateList)
#         print cr
        
#         # Remove dates with lots of missing returns (eg. non-trading days)
#         io = ma.sum(ma.getmaskarray(cr.data), axis=0)
#         goodDatesIdx = numpy.flatnonzero(io < 0.5 * len(self.subFactors))
#         badDatesIdx = [i for i in range(len(dateList)) \
#                        if i not in goodDatesIdx]
#         if len(badDatesIdx) > 0:
#             badDates = numpy.take(dateList, badDatesIdx)
#             self.log.info('Removing dates with very little trading: %s',
#                             ','.join([str(d) for d in badDates]))
#             cr.dates = numpy.take(dateList, goodDatesIdx)
#             cr.data = ma.take(cr.data, goodDatesIdx, axis=1)
#         
#         # Back-compatibility point
#         if self.rms_id in (14, 15):
#             # Legacy trimming
#             crFlat = ma.ravel(cr.data)
#             (ret_mad, bounds) = Utilities.mad_dataset(crFlat, -25, 25,
#                                                       axis=0, treat='zero')
#             cr.data = ma.where(cr.data<bounds[0], 0.0, cr.data)
#             cr.data = ma.where(cr.data>bounds[1], 0.0, cr.data)
#         else:
#             print cr.data
#             (cr.data, bounds) = Utilities.mad_dataset(cr.data,
#                                                       self.xrBnds[0], self.xrBnds[1],
#                                                       axis=0, treat='clip')
        # Just do it
        self.log.debug('computing cov')
#         self.countries = []
#         self.currencies = []
#         cov = self.generateFactorSpecificRisk(date, self.modelDB, self.marketDB).factorCov
        cov = self.covarianceCalculator.computeFactorCovarianceMatrix(cr)
#         cov = self.generateFactorSpecificRisk(date, self.modelDB, self.marketDB)
        self.log.debug('done computing cov')
        ret = Utilities.Struct()
        ret.subFactors = self.subFactors
        ret.factorCov = cov
        return ret

    def loadFactorCovarianceMatrix(self, rmi, modelDB):
        """Loads the factor-factor covariance matrix of the given risk
        model instance.
        Returns a (cov, factors) pair where cov is an m by m array
        containing the factor covariances and factors is a list of the
        m factor names.
        """
#         self.setFactorsForDate(date, modelDB)
        statSubFactors = modelDB.getRiskModelInstanceSubFactors(rmi, self.factors)
        cov = modelDB.getFactorCovariances(rmi, statSubFactors)
        return (cov, self.factors)
    
    def getAllClassifications(self, modelDB):
        return list()
    
    def loadFactorReturns(self, date, modelDB):
        """Loads the factor returns of the given dates.
        Returns a pair of lists with factor returns and names.
        """
        self.setFactorsForDate(date, modelDB)
        subFactors = modelDB.getSubFactorsForDate(date, self.factors)
        factorReturns = modelDB.loadFactorReturnsHistory(
            self.rms_id, subFactors, [date]).data[:,0]
        return (factorReturns, self.factors)
    
    def loadCumulativeFactorReturns(self, date, modelDB):
        """Loads the cumulative factor returns of the given dates.
        Returns a pair of lists with factor returns and names.
        """
        self.setFactorsForDate(date, modelDB)
        subFactors = modelDB.getSubFactorsForDate(date, self.factors)
        cumReturns = modelDB.loadCumulativeFactorReturnsHistory(
            self.rms_id, subFactors, [date]).data[:,0]
        return (cumReturns, self.factors)

class FIAxioma2014MH1(FIAxioma2014MH):
    rm_id = 2500
    revision = 1
    rms_id = 25000
    xrBnds = [-15.0, 15.0]


    def __init__(self, modelDB, marketDB):
        FIAxioma2014MH.__init__(self, modelDB, marketDB)
        self.log = logging.getLogger('FIModels.FIAxioma2014MH1')
        self.specific = True
        self.currencyModel = riskmodels.getModelByName('FXAxioma2010USD')(modelDB, marketDB)
                
class Univ10AxiomaMH(FIAxioma2014MH):
    rm_id = 3000
    revision = 1
    rms_id = 30000
    xrBnds = [-15.0, 15.0]
    returnsTimingId = 1
    intercept = ModelFactor('Global Market', 'Global Market')

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('FIModels.Univ10AxiomaMH')
        MFM.RegionalFundamentalModel.__init__(
                        self, ['SEDOL', 'CUSIP'], modelDB, marketDB)
        self.currencyModel = riskmodels.getModelByName('FXAxioma2010USD')(modelDB, marketDB)
        self.setCalculators(modelDB)
        self.equityModel = riskmodels.getModelByName('WWAxioma2011MH')(modelDB, marketDB)
        self.submodelSeriesIDs = (20000, 10001, 109)


    def setFactorsForDate(self, date, modelDB):
        self.setRiskModelGroupsForDate(date)
        self.equityModel.setFactorsForDate(date, modelDB)

        self.factors = []
        for f in self.descFactorMap:
            dbFactor = self.descFactorMap[f]
            if date >= dbFactor.from_dt and date < dbFactor.thru_dt:
                self.factors.append(dbFactor)

        currencies = set(f.factorID for f in self.equityModel.currencies)
        self.currencies = [f for f in self.factors if f.factorID in currencies]
        countries = set(f.factorID for f in self.equityModel.countries)
        self.countries = [f for f in self.factors if f.factorID in countries]


    def getSubmodelReturns(self, date, modelDB):
        returns = {}
        q = """
                SELECT distinct 
                    sub_factor_id,
                    VALUE 
                FROM
                    rms_factor_return 
                WHERE
                    dt = :date_arg                   
                AND
                    rms_id in %s

        """ % str(self.submodelSeriesIDs)
        modelDB.dbCursor.execute(q, date_arg=date)
        factor = ModelFactor('someName', 'someDesc')
        for r in modelDB.dbCursor.fetchall():
            sf = ModelSubFactor(factor, r[0], date)
            returns[sf] = r[1]
        return returns
            
    def transferReturns(self, date, rmi, options, modelDB, marketDB):
        self.setFactorsForDate(date, modelDB)
        self.log.info('deleting returns for %s' % str(date))        
        modelDB.deleteFactorReturns(rmi)
        self.log.info('done deleting returns for %s' % str(date))        
        if options.testOnly:
            self.log.info('Reverting changes')
            #modelDB.revertChanges()
        else:
            modelDB.commitChanges()   
        self.log.info('getting submodel returns from for %s' % str(date))          
        returns = self.getSubmodelReturns(date, modelDB)
        #print returns
        self.log.info('done getting submodel returns from for %s' % str(date)) 
        self.log.info('inserting returns for %s' % str(date))       
        modelDB.insertFactorReturns(self.rms_id, date, list(returns.keys()), list(returns.values()))
        rmi.setHasReturns(True, modelDB)
        self.log.info('done inserting returns for %s' % str(date))  

    def setCalculators(self,modelDB,overrider = False):
        riskmodels.defaultFundamentalCovarianceParameters(self,nwLag=2,overrider=overrider)

    def computeCov(self, date, modelDB, marketDB):
        """Compute the factor-factor covariance matrix.
        This function is based on some of the code in the MFM.py RegionalFundamentalModel method generateFactorSpecificRisks starting at line 4963.
        """
        self.log.debug('computeCov: begin')

        if isinstance(self.covarianceCalculator,
                RiskCalculator.CompositeCovarianceMatrix2009):
            (minVarObs, maxVarObs) = self.vp.getCovarianceSampleSize()
            (minCorrObs, maxCorrObs) = self.cp.getCovarianceSampleSize()
            (minOmegaObs, maxOmegaObs) = (max(minVarObs, minCorrObs), max(maxVarObs, maxCorrObs))
            (minDeltaObs, maxDeltaObs) = self.sp.getCovarianceSampleSize()
        else:
            (minOmegaObs, maxOmegaObs) = self.rp.getCovarianceSampleSize()
            (minDeltaObs, maxDeltaObs) = self.rp.getSpecificRiskSampleSize()


        subFactors = modelDB.getSubFactorsForDate(date, self.factors)

        rmi=modelDB.getRiskModelInstance(self.rms_id,date)
        if rmi == None:
            raise LookupError('No risk model instance for %s' % str(date))
        if not rmi.has_returns:
            raise LookupError('Returns are missing in risk model instance on %s' % str(date))

        # Check if any member RMGs have weekend trading
        # If so, need to adjust list of dates fetched
        omegaDateList = modelDB.getDates(self.rmg, date, maxOmegaObs-1, excludeWeekend=True)
        omegaDateList.reverse()
        deltaDateList = modelDB.getDates(self.rmg, date, maxDeltaObs-1, excludeWeekend=True)
        deltaDateList.reverse()
        # Check that enough consecutive days have returns, try to get
        # the maximum number of observations
        if len(omegaDateList) > len(deltaDateList):
           dateList = omegaDateList
        else:
           dateList = deltaDateList

        rmiList = modelDB.getRiskModelInstances(self.rms_id, dateList)
        okDays = [i.date == j and i.has_returns for (i,j) in zip(rmiList, dateList)]
        okDays.append(False)
        firstBadDay = okDays.index(False)
        dateList = dateList[:firstBadDay]
        if len(dateList) < max(minOmegaObs, minDeltaObs):
            required = max(minOmegaObs, minDeltaObs)
            self.log.warning('%d missing risk model instances for required days', required - len(dateList))
            raise LookupError('%d missing risk model instances for required days' % (required - len(dateList)))
        if len(dateList) < maxOmegaObs:
            self.log.info('Using only %d of %d days of factor return history', len(dateList), maxOmegaObs)
        if len(dateList) < maxDeltaObs:
            self.log.info('Using only %d of %d days of specific return history', len(dateList), maxDeltaObs)

        # Remove dates for which many markets are non-trading
        minMarkets = int(0.5 * len(self.countries))
        datesAndMarkets = modelDB.getActiveMarketsForDates(self.rmg, dateList[-1], dateList[0])
        datesAndMarkets = [(d,n) for (d,n) in datesAndMarkets if d.weekday() <= 4]

        # Remember, datesAndMarkets is in chron. order whereas dateList is reversed
        badDatesIdx = [len(dateList)-i-1 for (i,n) in enumerate(datesAndMarkets) if n[1] <= minMarkets]
        badDates = numpy.take(dateList, badDatesIdx)
        self.log.info('Removing dates with < %d markets trading: %s',minMarkets, ','.join([str(d) for d in badDates]))
        goodDatesIdx = [i for i in range(len(dateList)) if i not in badDatesIdx]
        dateList = numpy.take(dateList, goodDatesIdx)

 
        self.log.debug('building time-series matrices: begin')

        ret = Utilities.Struct()

        # Load returns timing adjustment factors if required
        nonCurrencySubFactors = [f for f in subFactors if f.factor not in self.currencies]
        if self.usesReturnsTimingAdjustment(): 
            rmgList = modelDB.getAllRiskModelGroups()
            adjustments = modelDB.loadReturnsTimingAdjustmentsHistory(self.returnsTimingId, rmgList, dateList[:max(maxOmegaObs, maxDeltaObs)])
            adjustments.data = adjustments.data.filled(0.0)
            if self.debuggingReporting:
                outData = adjustments.data
                mktNames = [r.mnemonic for r in rmgList]
                dtStr = [str(dt) for dt in adjustments.dates]
                sretFile = 'tmp/%s-RetTimAdj.csv' % self.name
                Utilities.writeToCSV(outData, sretFile, rowNames=mktNames, columnNames=dtStr)
        else:
            adjustments = None

        # Factor covariance matrix next
        ret.subFactors = subFactors
        if isinstance(self.covarianceCalculator, RiskCalculator.CompositeCovarianceMatrix2009) or isinstance(self.covarianceCalculator, RiskCalculator.CompositeCovarianceMatrix):
            # Load up non-currency factor returns
            nonCurrencyFactorReturns = modelDB.loadFactorReturnsHistory(self.rms_id, nonCurrencySubFactors, dateList[:maxOmegaObs])

            if self.debuggingReporting:
                outData = numpy.transpose(nonCurrencyFactorReturns.data.filled(0.0))
                dateStr = [str(d) for d in dateList[:maxOmegaObs]]
                assetNames = [s.factor.name for s in nonCurrencySubFactors]
                sretFile = 'tmp/%s-fret.csv' % self.name
                Utilities.writeToCSV(outData, sretFile, columnNames=assetNames,rowNames=dateStr)

            # Adjust factor returns for returns-timing, if applicable.  Need to pass in the rms_id of the WW fundamental model to make sure the returns timing adjustments are
            # applied correctly
            nonCurrencyFactorReturns = self.adjustFactorReturnsForTiming(date, nonCurrencyFactorReturns, adjustments, modelDB, marketDB, rms_id=self.equityModel.rms_id)

            if self.debuggingReporting and hasattr(nonCurrencyFactorReturns, 'adjust'):
                outData = nonCurrencyFactorReturns.data.filled(0.0) + nonCurrencyFactorReturns.adjust
                outData = numpy.transpose(outData)
                sretFile = 'tmp/%s-fretAdj.csv' % self.name
                Utilities.writeToCSV(outData, sretFile, columnNames=assetNames, rowNames=dateStr)

            # Pull up currency subfactors and returns
            currencySubFactors = [f for f in subFactors if f.factor in self.currencies]
            currencyFactorReturns = modelDB.loadFactorReturnsHistory(self.currencyModel.rms_id, currencySubFactors, dateList[:maxOmegaObs])
            crmi = modelDB.getRiskModelInstance(self.currencyModel.rms_id, date)
            assert(crmi is not None)

            # Post-process the array data a bit more, then compute cov
            self.equityModel.setFactorsForDate(date,modelDB) # perhaps we should call this when we call it for the Univ10 model? or overwrite the Univ model's setFactorsForDate() function to include this line of code?
            self.covarianceCalculator.fullCovParameters.setSubFactorsForDVA(modelDB.getSubFactorsForDate(date,self.equityModel.factors))
            self.covarianceCalculator.varParameters.setSubFactorsForDVA(modelDB.getSubFactorsForDate(date,self.equityModel.factors))
            self.covarianceCalculator.corrParameters.setSubFactorsForDVA(modelDB.getSubFactorsForDate(date,self.equityModel.factors))
            ret.factorCov = self.build_regional_covariance_matrix(date, dateList[:maxOmegaObs],currencyFactorReturns, nonCurrencyFactorReturns, crmi, modelDB, marketDB)
            ret.subFactors = nonCurrencySubFactors + currencySubFactors
        # Do it the old simple way, if composite cov not used
        else:
            factorReturns = modelDB.loadFactorReturnsHistory(self.rms_id, subFactors, dateList[:maxOmegaObs])
            (factorReturns.data, bounds) = Utilities.mad_dataset(factorReturns.data, self.xrBnds[0], self.xrBnds[1], treat='clip')
            ret.factorCov = self.covarianceCalculator.computeFactorCovarianceMatrix(factorReturns)
            ret.subFactors = subFactors

        # #### Transform factor correlation/covariance matrices to ensure that the block corresponding to the WW model matches the WW model
        
        # Get initial (full) correlation matrix for the Universal model
        (stdVector, factorCorr) = Utilities.cov2corr(ret.factorCov)
        fullStdVector = pandas.Series(stdVector, index=[f.factor.name for f in ret.subFactors]).fillna(0.0)
        fullCorr = pandas.DataFrame(factorCorr, index=[f.factor.name for f in ret.subFactors], columns=[f.factor.name for f in ret.subFactors]).fillna(0.0)

        # Get target equity model correlation matrix
        equity_rmi = modelDB.getRiskModelInstance(self.equityModel.rms_id, date)
        (equityCov, equityFactors) = self.equityModel.loadFactorCovarianceMatrix(equity_rmi, modelDB)
        (equityStdVector, equityCorr) = Utilities.cov2corr(equityCov)
        equityCorr = pandas.DataFrame(equityCorr,index=[f.name for f in equityFactors],columns=[f.name for f in equityFactors]).fillna(0.0)
        equityStdVec = pandas.Series(equityStdVector, index=[f.name for f in equityFactors]).fillna(0.0)

        if not set(fullStdVector.index).issuperset(set(equityStdVec.index)):
            raise Exception('Equity factors are not a subset of Universal model factors')

        # Reorder initial (full) correlation matrix according to desired block structure
        index = equityCorr.index.tolist() + [f for f in fullCorr.index if f not in equityCorr.index]

        initCorr = fullCorr.reindex(index=index, columns=index)
        initStdVec = fullStdVector.reindex(index=index)

        # Perform orthogonal Procrustean analysis to transform initCorr so that the relevant block in initCorr is as close as possible to equityCorr
        finalCorr = Utilities.procrustes_transform(initCorr,equityCorr)
        finalStdVec = initStdVec.copy()
        finalStdVec[equityStdVec.index] = equityStdVec

        # Transform finalCorr to Cov
        finalCov = finalCorr.mul(finalStdVec, axis=0).mul(finalStdVec, axis=1)

        fname2subf = dict((f.factor.name,f) for f in ret.subFactors)
        ret.factorCov = finalCov.values
        ret.subFactors = [fname2subf[f] for f in finalCov.index]

        # Debugging: write initial and final correlation matrices and the final covariance matrix to pickle
        if self.debuggingReporting:
            equityCovdf=pandas.DataFrame(equityCov,index=equityCorr.index, columns=equityCorr.columns)
            if hasattr(equityCorr, 'save'):
                equityCorr.save('equityCorr-%s-%s.pick' % (self.equityModel.mnemonic, date.strftime('%Y%m%d')))
                equityCovdf.save('equityCov-%s-%s.pick' % (self.equityModel.mnemonic,date.strftime('%Y%m%d')))
                initCorr.save('initCorr-%s-%s.pick' % (self.mnemonic, date.strftime('%Y%m%d')))
                finalCorr.save('finalCorr-%s-%s.pick' % (self.mnemonic, date.strftime('%Y%m%d')))
                finalCov.save('finalCov-%s-%s.pick' % (self.mnemonic, date.strftime('%Y%m%d')))
            else:
                equityCorr.to_pickle('equityCorr-%s-%s.pick' % (self.equityModel.mnemonic, date.strftime('%Y%m%d')))
                equityCovdf.to_pickle('equityCov-%s-%s.pick' % (self.equityModel.mnemonic,date.strftime('%Y%m%d')))
                initCorr.to_pickle('initCorr-%s-%s.pick' % (self.mnemonic, date.strftime('%Y%m%d')))
                finalCorr.to_pickle('finalCorr-%s-%s.pick' % (self.mnemonic, date.strftime('%Y%m%d')))
                finalCov.to_pickle('finalCov-%s-%s.pick' % (self.mnemonic, date.strftime('%Y%m%d')))


        # Report day-on-day correlation matrix changes
        self.reportCorrelationMatrixChanges(date, ret.factorCov, rmiList[1], modelDB)
        if self.debuggingReporting:
            # Write correlation matrix to flatfile
            factorNames = [f.factor.name for f in ret.subFactors]
            (d, corrMatrix) = Utilities.cov2corr(ret.factorCov, fill=True)
            corroutfile = 'tmp/%s-corrFinal-%s.csv' % (self.name, dateList[0])
            Utilities.writeToCSV(corrMatrix, corroutfile, columnNames=factorNames, rowNames=factorNames)
            var = numpy.diag(ret.factorCov)[:,numpy.newaxis]
            varoutfile = 'tmp/%s-varFinal-%s.csv' % (self.name, dateList[0])
            Utilities.writeToCSV(var, varoutfile, rowNames=factorNames)
            covOutFile = 'tmp/%s-covFinal-%s.csv' % (self.name, dateList[0])
            Utilities.writeToCSV(ret.factorCov, covOutFile, columnNames=factorNames, rowNames=factorNames)

        self.log.info('Sum of composite cov matrix elements: %f', ma.sum(ret.factorCov, axis=None))
        self.log.debug('computed factor covariances')
        return ret

class Univ10AxiomaMH_Pre2009(Univ10AxiomaMH):
    rm_id = 3000
    revision = 2
    rms_id = 30001
    
    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('FIModels.Univ10AxiomaMH_Pre2009')
        MFM.RegionalFundamentalModel.__init__(
                        self, ['SEDOL', 'CUSIP'], modelDB, marketDB)
        self.currencyModel = riskmodels.getModelByName('FXAxioma2010USD')(modelDB, marketDB)
        self.setCalculators(modelDB)
        self.equityModel = riskmodels.getModelByName('WWAxioma2011MH_Pre2009')(modelDB, marketDB)
        self.submodelSeriesIDs = (20000, 10001, 108)
