
import logging
import numpy
import numpy.ma as ma
from marketdb import MarketDB
from riskmodels import ModelDB
from riskmodels import MFM
from riskmodels import RiskCalculator
from riskmodels import LegacyUtilities as Utilities
from riskmodels.Factors import ModelFactor

class SubModel():
    def __init__(self, modelDB, marketDB):    
        self.name = None
        self.modelName = None
        self.rm_id = None
        self.revision = None
        self.rms_id = None
        self.factors = []
        self.factorNames = []
        self.subFactors = {}
        rmInfo = modelDB.getRiskModelInfo(self.rm_id, self.revision)
        self.rms_id = rmInfo.serial_id
        self.name = rmInfo.name
        self.description = rmInfo.description
        self.mnemonic = rmInfo.mnemonic
        self.rmg = [modelDB.getRiskModelGroup(rmg.rmg_id) \
                    for rmg in rmInfo.rmgTimeLine]
        self.log.info('Initializing %s (%s)', self.description, rmInfo.numeraire)


class Univ10AxiomaMH():
    rm_id = 3000
    revision = 1
    rms_id = 30000
    xrBnds = [-15.0, 15.0]


    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('UnivModels.Univ10AxiomaMH')
        self.modelDB = modelDB
        self.marketDB = marketDB
        self.subModels = {}
        self.factors = []
        self.factorNames = []  
        self.subModelList = ['FIAxioma2014MH', 'WW21AxiomaMH', 'CommodityModelMH']
        self.subModels = self.addSubmodels()
        self.subFactors = {}
        rmInfo = modelDB.getRiskModelInfo(self.rm_id, self.revision)
        self.rms_id = rmInfo.serial_id
        self.name = rmInfo.name
        self.description = rmInfo.description
        self.mnemonic = rmInfo.mnemonic
        self.rmg = [modelDB.getRiskModelGroup(rmg.rmg_id) \
                    for rmg in rmInfo.rmgTimeLine]
        self.log.info('Initializing %s (%s)', self.description, rmInfo.numeraire)
    
    def addSubModels(self): 
        subModels = {}
        sm = SubModel(self.modelDB, self.marketDB)
        sm.name = 'FIAxioma2014MH'
        sm.rm_id = 2000
        sm.rms_id = 20000
        sm.revision = 1
        subModels['FIAxioma2014MH'] = sm
        
        sm = SubModel(self.modelDB, self.marketDB)
        sm.name = 'WW21AxiomaMH'
        sm.rm_id = 76
        sm.rms_id = 109
        sm.revision = 3
        subModels['WW21AxiomaMH'] = sm

        sm = SubModel(self.modelDB, self.marketDB)
        sm.name = 'CommodityModelMH'
        sm.rm_id = 1000
        sm.rms_id = 10001
        sm.revision = 2
        subModels['FIAxioma2014MH'] = sm
                 
    def getRiskModelInstance(self, date, modelDB):
        
        rmi =  modelDB.getRiskModelInstance(self.rms_id, date)
        print('rmi', rmi)
        return rmi
    
    def setFactorsForDate(self, date, modelDB):
        self.factors = []
        self.factorNames = []
        q = """SELECT f.name, f.description
        FROM rms_factor rf JOIN factor f ON f.factor_id=rf.factor_id
        WHERE rf.rms_id=:rms_id
        ORDER BY f.name
        """
        modelDB.dbCursor.execute(q, rms_id=self.rms_id)
        for r in modelDB.dbCursor.fetchall():
            self.factors.append(ModelFactor(r[0], r[1]))
            self.factorNames.append(r[0])
        dbFactors = modelDB.getRiskModelSerieFactors(self.rms_id)
        self.nameFactorMap = dict([(f.name, f) for f in dbFactors])
        for f in self.factors:
            f.factorID = self.nameFactorMap[f.name].factorID


#        self.validateFactorStructure(date, warnOnly = True)

    def setCalculators(self):
        from RiskModels_V3 import defaultFundamentalCovarianceParameters, defaultStatisticalCovarianceParameters
        # Set up risk parameters
        defaultStatisticalCovarianceParameters(self, nwLag=0, dva=None)
        
    def transferReturns(self, date, rmi, options):
        self.setFactorsForDate(date, self.modelDB)
        self.log.info('deleting returns for %s' % str(date))        
        self.modelDB.deleteFactorReturns(rmi)
        self.log.info('done deleting returns for %s' % str(date))        
        if options.testOnly:
            self.log.info('Reverting changes')
            self.modelDB.revertChanges()
        else:
            self.modelDB.commitChanges()   
        self.log.info('getting returns from marketDB for %s' % str(date))          
        returns = self.marketDB.getFIReturns(date)
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
        self.log.info('done inserting returns for %s' % str(date))   

    def computeCov(self, date, rmi, options):
        """Compute the covariance matrix.
        """
        self.log.debug('computeCov: begin')
        self.setFactorsForDate(date, self.modelDB)
        if len(self.subFactors) == 0: 
            self.log.info('getting subFactors')  
            self.subFactors = self.modelDB.getSubFactorsForDate(date, self.factors) 
#            self.subFactors = self.modelDB.getRiskModelInstanceSubFactors(
#                                                                     rmi, self.factors)
            self.log.info('done getting subFactors')   
        # Get some basic risk parameters
        self.setCalculators()
        if isinstance(self.covarianceCalculator,
                RiskCalculator.CompositeCovarianceMatrix2009):
            (minOmegaObs, maxOmegaObs) = self.cp.getCovarianceSampleSize()
        else:
            (minOmegaObs, maxOmegaObs) = self.rp.getCovarianceSampleSize()
        
#        (minOmegaObs, maxOmegaObs) = (250, 250)
        
        # Deal with weekend dates (well, ignore them)
        dateList = self.modelDB.getDates([self.rmg], date, maxOmegaObs-1, 
                                    excludeWeekend=True)
        dateList.reverse()
        dateList = dateList[:maxOmegaObs]
        
        # Sanity check -- make sure we have all required data
        rmiList = self.modelDB.getRiskModelInstances(self.rms_id, dateList)
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
        cr = self.modelDB.loadFactorReturnsHistory(
                                self.rms_id, self.subFactors, dateList)
        
        # Remove dates with lots of missing returns (eg. non-trading days)
        io = ma.sum(ma.getmaskarray(cr.data), axis=0)
        goodDatesIdx = numpy.flatnonzero(io < 0.5 * len(self.subFactors))
        badDatesIdx = [i for i in range(len(dateList)) \
                       if i not in goodDatesIdx]
        if len(badDatesIdx) > 0:
            badDates = numpy.take(dateList, badDatesIdx)
            self.log.info('Removing dates with very little trading: %s',
                            ','.join([str(d) for d in badDates]))
            cr.dates = numpy.take(dateList, goodDatesIdx)
            cr.data = ma.take(cr.data, goodDatesIdx, axis=1)
        
        # Back-compatibility point
        if self.rms_id in (14, 15):
            # Legacy trimming
            crFlat = ma.ravel(cr.data)
            (ret_mad, bounds) = Utilities.mad_dataset(crFlat, -25, 25,
                                                      axis=0, treat='zero')
            cr.data = ma.where(cr.data<bounds[0], 0.0, cr.data)
            cr.data = ma.where(cr.data>bounds[1], 0.0, cr.data)
        else:
            (cr.data, bounds) = Utilities.mad_dataset(cr.data,
                                                      self.xrBnds[0], self.xrBnds[1],
                                                      axis=0, treat='clip')
        # Just do it
        self.log.debug('computing cov')
        cov = self.covarianceCalculator.computeFactorCovarianceMatrix(cr)
        self.log.debug('done computing cov')
        self.log.info('deleting covMatrix in modeldb')
        self.modelDB.deleteRMIFactorCovMatrix(rmi)
        self.log.debug('inserting cov')
        self.modelDB.insertFactorCovariances(rmi, self.subFactors, cov)
        self.log.debug('done inserting cov')
        self.log.debug('computeCov: end')
        
    def loadFactorCovarianceMatrix(self, rmi, modelDB):
        """Loads the factor-factor covariance matrix of the given risk
        model instance.
        Returns a (cov, factors) pair where cov is an m by m array
        containing the factor covariances and factors is a list of the
        m factor names.
        """
        statSubFactors = modelDB.getRiskModelInstanceSubFactors(rmi, self.factors)
        cov = modelDB.getFactorCovariances(rmi, statSubFactors)
        return (cov, self.factors)
    
    def loadFactorReturns(self, date, modelDB):
        """Loads the factor returns of the given dates.
        Returns a pair of lists with factor returns and names.
        """
        subFactors = modelDB.getSubFactorsForDate(date, self.factors)
        factorReturns = modelDB.loadFactorReturnsHistory(
            self.rms_id, subFactors, [date]).data[:,0]
        return (factorReturns, self.factors)
    
    def loadCumulativeFactorReturns(self, date, modelDB):
        """Loads the cumulative factor returns of the given dates.
        Returns a pair of lists with factor returns and names.
        """
        subFactors = modelDB.getSubFactorsForDate(date, self.factors)
        cumReturns = modelDB.loadCumulativeFactorReturnsHistory(
            self.rms_id, subFactors, [date]).data[:,0]
        return (cumReturns, self.factors)
