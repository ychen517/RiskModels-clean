
import datetime
import logging
import numpy
import numpy.ma as ma
import numpy.linalg as linalg
from riskmodels.Factors import ModelFactor
from riskmodels import Matrices
from riskmodels import ModelDB
from riskmodels import RiskCalculator
from riskmodels import StyleExposures
from riskmodels import LegacyUtilities as Utilities
from riskmodels import Outliers

class ModelCurrency:
    """A currency in a currency risk model.
    Contains its currency_code, currency_id (MarketDB), and
    description.
    """
    def __init__(self, code):
        self.currency_code = code
        self.currency_id = None
        self.description = None
    def __repr__(self):
        return 'ModelCurrency(%s)' % self.currency_code
    def __eq__(self, other):
        return self.currency_code == other.currency_code
    def __ne__(self, other):
        return self.currency_code != other.currency_code
    def __lt__(self, other):
        return self.currency_code < other.currency_code
    def __le__(self, other):
        return self.currency_code <= other.currency_code
    def __gt__(self, other):
        return self.currency_code > other.currency_code
    def __ge__(self, other):
        return self.currency_code >= other.currency_code
    def __hash__(self):
        return self.currency_code.__hash__()

class BaseCurrencyModel:
    # Instances should have rm_id, revision
    currencyFactorModel = False
    debuggingReporting = False
    xrBnds = [-15.0, 15.0]
    newExposureFormat = False
    intercept = None
    forceRun = False
    styles = []
    industries = []
    currencies = []
    macro_core = []
    macro_market_traded = []
    macro_equity = []
    macro_sectors = []
    
    def __init__(self, modelDB, marketDB):
        rmInfo = modelDB.getRiskModelInfo(self.rm_id, self.revision)
        self.rms_id = rmInfo.serial_id
        self.name = rmInfo.name
        self.description = rmInfo.description
        self.mnemonic = rmInfo.mnemonic
        self.rmg = [modelDB.getRiskModelGroup(rmg.rmg_id) \
                    for rmg in rmInfo.rmgTimeLine]
        self.rmgTimeLine = rmInfo.rmgTimeLine
        self.log.info('Initializing %s (%s)', self.description, rmInfo.numeraire)
        
        # Get currency info from MarketDB
        self.currencyInfoMap = self.loadCurrencyInfo(marketDB)
        dbFactors = modelDB.getRiskModelSerieFactors(self.rms_id)
        self.nameFactorMap = dict([(f.name, f) for f in dbFactors])
        
        # Create ModelCurrency object for numeraire
        numeraire = rmInfo.numeraire
        assert(numeraire in self.currencyInfoMap)
        nc = ModelCurrency(numeraire)
        nc.currency_id = self.currencyInfoMap[numeraire]['currency_id']
        nc.description = self.currencyInfoMap[numeraire]['description']
        self.numeraire = nc
        
        # Initialize ModelDB currency cache
        modelDB.createCurrencyCache(marketDB)
    
    def createInstance(self, date, modelDB):
        """Creates a new risk model instance for this risk model serie
        on the given date.
        """
        return modelDB.createRiskModelInstance(self.rms_id, date)

    def getRiskModelInstance(self, date, modelDB):
        """Creates a new risk model instance for this risk model serie
        on the given date.
        """
        return modelDB.getRiskModelInstance(self.rms_id, date)
    
    def isCurrencyModel(self):
        return True

    def isStatModel(self):
        return False

    def isRegionalModel(self):
        return False

    def isFactorModel(self):
        return True

    def isProjectionModel(self):
        return False

    def computeCurrencyReturns(self, date, modelDB, marketDB):
        """Compute currency returns.  Returns an array (masked where
        appropriate) of currency returns.
        """
        self.log.debug('computeCurrencyReturns: begin')
        currencyIds = [c.currency_id for c in self.currencies]
        fxReturnsMatrix = modelDB.loadCurrencyReturnsHistory(
                self.rmg, date, 0, currencyIds, self.numeraire.currency_id, 
                idlookup=False)
        fxReturnsMatrix.data = fxReturnsMatrix.data.filled(0.0)
        
        codes = [c.currency_code for c in self.currencies]
        currencyIdxMap = dict([(j,i) for (i,j) in enumerate(codes)])
        rfRates = modelDB.getRiskFreeRateHistory(
                            codes, [date], marketDB).data[:,0].filled(0.0)
        r0 = rfRates[currencyIdxMap[self.numeraire.currency_code]]
        currencyReturns = fxReturnsMatrix.data[:,0] + rfRates - r0
        if self.debuggingReporting:
            for (code, c_id, raw, fac, rfr) in zip(codes, currencyIds, fxReturnsMatrix.data, currencyReturns, rfRates):
                logging.info('Currency return: %s, (%s, %s), Raw, %s, Factor, %s, rfRate, %s, rfRate0, %s',
                        date, code, c_id, raw[0], fac, rfr, r0)
        
        self.log.debug('computeCurrencyReturns: end')
        return currencyReturns
    
    def deleteInstance(self, date, modelDB):
        """Deletes the risk model instance for the given date if it exists.
        """
        rmi = modelDB.getRiskModelInstance(self.rms_id, date)
        if rmi != None:
            modelDB.deleteRiskModelInstance(rmi, self.newExposureFormat)
    
    
    def getCurrencyFactors(self):
        """Returns a list of ModelFactors corresponding to each 
        ModelCurrency in self.currencies.
        """
        factors = [ModelFactor(c.currency_code, c.currency_code) for c in self.currencies]
        for f in factors:
            f.factorID = self.nameFactorMap[f.name].factorID
        return factors
    
    def insertCurrencyReturns(self, date, currencyReturns, modelDB):
        """Inserts the currency returns into the database for the given date.
        currencyReturns is an array of the return values.
        """
        currencyFactors = self.getCurrencyFactors()
        subFactors = modelDB.getSubFactorsForDate(date, currencyFactors)
        assert(len(subFactors)==len(self.currencies))
        modelDB.insertFactorReturns(self.rms_id, date, subFactors, currencyReturns)
    
    def loadCurrencyInfo(self, marketDB):
        """Creates mapping of currency code to currency name, ID, 
        and valid from/thru dates.
        """
        query = """SELECT code, id, description, from_dt, thru_dt
                   FROM currency_ref"""
        marketDB.dbCursor.execute(query)
        results = marketDB.dbCursor.fetchall()
        currencyInfoMap = dict([(r[0], dict([
                                ('currency_id', r[1]),
                                ('description', r[2]),
                                ('from_dt', r[3].date()),
                                ('thru_dt', r[4].date())])) for r in results])
        return currencyInfoMap
    
    def loadCurrencyFactorReturns(self, rmi, modelDB):
        """Loads the factor returns from the given risk model instance.
        Returns a pair of lists with factor returns and names.
        """
        currencyFactors = self.getCurrencyFactors()
        subFactors = modelDB.getSubFactorsForDate(rmi.date, currencyFactors)
        factorReturns = modelDB.loadFactorReturnsHistory(rmi.rms_id, subFactors, [rmi.date]).data[:,0]
        return (factorReturns, currencyFactors)

    def loadCurrencyFactorReturnsHistory(self, subFactors, dateList, modelDB):
        """
        Create a timeseries matrix of the currency returns for this model
        """
        return modelDB.loadFactorReturnsHistory(self.rms_id, subFactors, dateList)

    def setFactorsForDate(self, *args):
        pass

    def RMGDateLogic(self, r, date):
        """Logic for determining where in the timeline
        each RMG is
        """

        # Set defaults
        r.rmg.downWeight = 1.0

        # Fade-out date
        if r.fade_dt <= date and (abs(r.full_dt-date) > (date-r.fade_dt)):
            fadePeriod = (r.thru_dt - r.fade_dt).days
            # Exponential downweighting function
            expWeights = Utilities.computeExponentialWeights(
                    60, fadePeriod, normalize=False)
            r.rmg.downWeight = expWeights[(date - r.fade_dt).days]

        # Fade-in date
        if r.full_dt > date and ((r.full_dt-date) <= abs(date-r.fade_dt)):
            fadePeriod = (r.full_dt - r.from_dt).days
            halfLife = 30
            expWeights = Utilities.computeExponentialWeights(
                    halfLife, fadePeriod, normalize=False)
            iLoc = min((r.full_dt - date).days, int(16*halfLife))
            iLoc = min(iLoc, len(expWeights)-1)
            r.rmg.downWeight = expWeights[iLoc]

        # Report on weighting
        if r.rmg.downWeight < 1.0:
            self.log.debug('%s (RiskModelGroup %d, %s) down-weighted to %.3f%%',
                    r.rmg.description, r.rmg.rmg_id, r.rmg.mnemonic,
                    r.rmg.downWeight * 100.0)
        return

    def setRiskModelGroupsForDate(self, date):
        """Determines the currencies correspoding to the
        member risk model groups for the given date -- to
        account for currency redenominations, etc.
        """
        if self.numeraire.currency_code in ('EUR', 'XEU'):
            # WARNING WARNING - hard-coding of dates
            if date < datetime.date(1995,1,1):
                nc = ModelCurrency('XEU')
            else:
                nc = ModelCurrency('EUR')
            nc.currency_id = self.currencyInfoMap[nc.currency_code]['currency_id']
            nc.description = self.currencyInfoMap[nc.currency_code]['description']
            self.numeraire = nc
        self.currencies = []
        self.rmg = []
        for r in self.rmgTimeLine:
            if r.from_dt <= date and r.thru_dt > date:
                self.RMGDateLogic(r, date)
                self.rmg.append(r.rmg)

        newRMGList = []
        for rmg in self.rmg:
            # Extra check in case RMG does not exist at all for current date
            inModel = False
            for tvd in rmg.timeVariantDicts:
                if tvd['from_dt'] <= date and tvd['thru_dt'] > date:
                    inModel = True
                 
            if inModel:
                newRMGList.append(rmg)
                if not rmg.setRMGInfoForDate(date):
                    raise Exception('Cannot determine details for %s risk model group (%d) on %s' % \
                            (rmg.description, rmg.rmg_id, str(date)))
            
                currInfo = self.currencyInfoMap.get(rmg.currency_code, None)
                assert(currInfo != None)
                if currInfo is None:
                    raise Exception('No records for currency code %s' % rmg.currency_code)
                if not (currInfo['from_dt'] <= date and currInfo['thru_dt'] > date):
                    raise Exception('Date inconsistencies for %s, %s (from: %s, thru: %s) on %s' %\
                            (rmg.description, rmg.currency_code, currInfo['from_dt'], currInfo['thru_dt'], date))
                mc = ModelCurrency(rmg.currency_code)
                if mc not in self.currencies:
                    mc.currency_id = currInfo['currency_id']
                    mc.description = currInfo['description']
                    self.currencies.append(mc)
            else:
                self.log.warning('RMG: %s does not exist for %s', rmg, date)
        self.currencyAssets = [ModelDB.SubIssue(string='DCSH_%s__11' % f.currency_code) for f in self.currencies]
        self.rmg = newRMGList

class CurrencyStatisticalFactorModel(BaseCurrencyModel):
    """Base class for currency models that derive the currency
    covariance matrix using PCA.
    The setup is an abuse of the standard model tables as follows:
      - the factors of the model (per rms_factor) are the currency factors
      - rms_factor_return stores the currency factor returns. The stat factor
        returns are not stored in the database.
      - the covariance table in the database contains the covariance
        of the statistical factors.
      - in the exposure table the "factors" are the statistical factors
        and the currency factors are proxied by their cash assets.
      - the specific return and specific variance tables also use the cash
        assets to proxy the currency factors.
    """
    def __init__(self, modelDB, marketDB):
        BaseCurrencyModel.__init__(self, modelDB, marketDB)
        for (factor, factorID) in zip(
            self.blind, modelDB.getFactors(
                [f.description for f in self.blind])):
            factor.factorID = factorID
        self.factors = self.blind

    def createCurrencyAssets(self):
        return [ModelDB.SubIssue(string='DCSH_%s__11' % f.currency_code) for f in self.currencies]

    def isCurrencyModel(self):
        return True

    def deleteStatisticalModel(self, rmi, modelDB):
        """Delete all statistical model information of the given
        risk model instance.
        """
        if self.newExposureFormat:
            modelDB.deleteRMIExposureMatrixNew(rmi)
        else:
            modelDB.deleteRMIExposureMatrix(rmi)
        modelDB.deleteRiskModelUniverse(rmi)
        modelDB.deleteSpecificReturns(rmi)
        modelDB.deleteEstimationUniverse(rmi)
        modelDB.deleteRMIFactorSpecificRisk(rmi)

    def generate_estimation_universe(self, date, data, subFactors, modelDB, marketDB):
        """ Simple estu function for currency stat model
        """
        self.log.debug('generate_estimation_universe: begin')

        # WARNING WARNING - hard-coding of dates
        if date < datetime.date(1995,1,1):
            estuISOList = ['USD', 'CAD', 'AUD', 'GBP', 'XEU', 'CHF', 'JPY']
        else:
            estuISOList = ['USD', 'CAD', 'AUD', 'GBP', 'EUR', 'CHF', 'JPY']
        estuISOList.extend(['BRL', 'MXN', 'SGD', 'KRW', 'ZAR', 'PLN'])
        estu = []
        pegList = []
        # Set to zero pegged currency returns
        ignoreList = StyleExposures.getPeggedCurrencies(\
                self.numeraire.currency_code, date)
        for (i,f) in enumerate(subFactors):
            if f.factor.name in ignoreList:
                pegList.append(i)
            elif f.factor.name in estuISOList and \
                    f.factor.name != self.numeraire.currency_code:
                estu.append(i)

        # Sift out currencies with really extreme behaviour
        medianValue = ma.median(abs(data), axis=1)
        medianValue = ma.masked_where(medianValue <= 0.0, medianValue)
        maxValue = ma.max(ma.masked_where(data==0.0, data), axis=1)
        crappiness = maxValue / medianValue
        crappy = numpy.flatnonzero(ma.getmaskarray(crappiness))
        if self.debuggingReporting:
            crapList = [f.factor.name for (i,f) in enumerate(subFactors) \
                    if i in crappy]
            self.log.info('These currencies are crappy: %s', crapList)

        # Sort out estu, nonest and the "nuked" set - these are
        # currencies which are just too extreme in behaviour
        # and will have their exposures set to zero
        universe = list(range(data.shape[0]))
        estu = set([i for i in estu if i not in crappy])
        nukeList = set(pegList).union(set(crappy))
        nonest = set(universe).difference(estu)

        self.log.debug('generate_estimation_universe: end')
        return (list(estu), list(nonest), list(nukeList))

    def generateStatisticalModel(self, date, modelDB, marketDB):
        """Calculate exposures and returns for statistical factors.
        """
        dateList = self.getDateList(date, modelDB)

        # Retrieve currency returns (in reverse chronological order)
        currencyFactors = self.getCurrencyFactors()
        subFactors = modelDB.getSubFactorsForDate(\
                date, currencyFactors)
        cr = self.loadCurrencyFactorReturnsHistory(subFactors, dateList, modelDB)

        # Create assets to represent currencies
        currencyAssets = self.createCurrencyAssets()

        # Remove dates with lots of missing returns (eg. non-trading days)
        io = ma.sum(ma.getmaskarray(cr.data), axis=0)
        goodDatesIdx = numpy.flatnonzero(io < 0.5 * len(self.currencies))
        badDatesIdx = [i for i in range(len(dateList)) \
                if i not in goodDatesIdx]
        if len(badDatesIdx) > 0:
            badDates = numpy.take(dateList, badDatesIdx)
            self.log.info('Removing dates with very little trading: %s',
                    ','.join([str(d) for d in badDates]))
            cr.dates = numpy.take(dateList, goodDatesIdx)
            cr.data = ma.take(cr.data, goodDatesIdx, axis=1)

        # Get estu
        (estu, nonest, nukeList) = self.generate_estimation_universe(\
                date, cr.data, subFactors, modelDB, marketDB)

        if self.debuggingReporting:
            srNames = [s.factor.name for s in subFactors]
            fName = 'tmp/cret-%s.csv' % date.year
            dates = [str(dt) for dt in cr.dates]
            Utilities.writeToCSV(
                    ma.transpose(cr.data), fName,
                    columnNames=srNames, rowNames = dates)

            tmp = Utilities.multi_mad_data(cr.data,
                    restrictOneAxis=list(range(self.cp.getCovarianceHalfLife())))
            fName = 'tmp/cretSMAD-%s.csv' % date.year
            Utilities.writeToCSV(
                    ma.transpose(tmp), fName,
                    columnNames=srNames, rowNames=dates)

            tmp = Utilities.mad_dataset(cr.data, -15.0, 18.0)[0]
            fName = 'tmp/cretMAD-%s.csv' % date.year
            Utilities.writeToCSV(
                    numpy.transpose(ma.filled(tmp, 0.0)), fName,
                    columnNames=srNames, rowNames=dates)

        # Trim excessive values
        crCopy = ma.filled(Utilities.multi_mad_data(cr.data,
                restrictOneAxis=list(range(self.cp.getCovarianceHalfLife()))), 0.0)

        # Compute svd of returns history
        crSubset = numpy.array(crCopy[:,:self.returnHistory])
        cr_estu = numpy.take(crSubset, estu, axis=0)
        (u, d, v) = linalg.svd(cr_estu, full_matrices=False)
        d = d / numpy.sqrt(crSubset.shape[1])
        order = numpy.argsort(-d)
        u = numpy.take(u, order, axis=1)
        d = numpy.take(d, order, axis=0)

        # Create estu exposure matrix
        expMatrixEstu = numpy.dot(u[:,:self.numStatFactors],\
                numpy.diag(d[:self.numStatFactors]))
        # Compute factor returns
        (currencyFR, dummy) = Utilities.ordinaryLeastSquares(\
                numpy.take(crCopy, estu, axis=0), expMatrixEstu)
        # Back out nonestu exposures
        (expMatrixNonest, dummy) = Utilities.ordinaryLeastSquares(\
                numpy.transpose(numpy.take(crCopy[:,:self.returnHistory],\
                nonest, axis=0)), numpy.transpose(\
                currencyFR[:,:self.returnHistory]))
        expMatrixNonest = numpy.transpose(expMatrixNonest)

        # In case we have fewer estu currencies than there are factors
        # do some padding with zeros
        if len(estu) < self.numStatFactors:
            zDim = self.numStatFactors - len(estu)
            currencyFR = numpy.concatenate((currencyFR, \
                    numpy.zeros((zDim, currencyFR.shape[1]), float)), axis=0)

        # Build complete exposure matrix
        expMatrix = numpy.zeros((crSubset.shape[0], self.numStatFactors),
                                float)
        jDim = min(self.numStatFactors, len(estu))
        for j in range(jDim):
            expCol = expMatrix[:,j]
            numpy.put(expCol, estu, expMatrixEstu[:,j])
            numpy.put(expCol, nonest, expMatrixNonest[:,j])
            numpy.put(expCol, nukeList, 0.0)
            expMatrix[:,j] = expCol

        # And compute specific returns using original untreated asset returns
        currencySR = cr.data.filled(0.0) - numpy.dot(expMatrix, currencyFR)
        currencySR = ma.masked_where(ma.getmaskarray(cr.data), currencySR)
        if self.debuggingReporting:
            dtList = [str(dt) for dt in cr.dates]
            outfile = 'tmp/%s-facretHist-%s.csv' % (date, self.mnemonic)
            retNames = ['Factor %d' % (i_f+1) for i_f in range(currencyFR.shape[0])]
            Utilities.writeToCSV(currencyFR, outfile, rowNames=retNames,
                    columnNames=dtList)

        # Compute currency stat factor covariance matrix
        cr.data = numpy.array(currencyFR)
        smallCov = self.covarianceCalculator.\
                computeFactorCovarianceMatrix(cr)

        # Compute specific variances
        srMaxObs = self.sp.getCovarianceSampleSize()[1]
        srMatrix = Matrices.TimeSeriesMatrix(
            currencyAssets, dateList[:srMaxObs])
        srMatrix.data = currencySR[:,:srMaxObs]

        specificVars = self.specificRiskCalculator.\
                computeSpecificRisks(None, None, srMatrix,
                        estu=nonest, clipVars=False, multiMAD=True)

        d = d*d
        prc = numpy.sum(d[0:self.numStatFactors], axis=0) \
                / numpy.sum(d, axis=0) * 100
        self.log.info('%d factors explain %f%% of variance', self.numStatFactors, prc)

        if self.debuggingReporting:
            # Sort out current and past factor names
            curNames = [s.factor.name.replace(',','') for s in subFactors]
            allNames = list(self.currencyInfoMap.keys())
            currencyNamesMap = dict([(j,i) for (i,j) in enumerate(allNames)])
            curNamesIdx = [currencyNamesMap[n] for n in curNames]
            dtList = [str(date)]
            # Output exposure matrix

        if self.debuggingReporting:
            # Sort out current and past factor names
            curNames = [s.factor.name.replace(',','') for s in subFactors]
            allNames = list(self.currencyInfoMap.keys())
            currencyNamesMap = dict([(j,i) for (i,j) in enumerate(allNames)])
            curNamesIdx = [currencyNamesMap[n] for n in curNames]
            dtList = [str(date)]
            # Output exposure matrix
            outfile = 'tmp/cfexp-%s-%s.csv' % (date, self.mnemonic)
            retNames = ['Factor %d' % (i_f+1) for i_f in range(expMatrix.shape[1])]
            Utilities.writeToCSV(expMatrix, outfile, columnNames=retNames, rowNames=allNames)
            # Output common factor risk
            cfriskFull = numpy.zeros((1,len(allNames)), float)
            covMatrix = ma.dot(expMatrix, ma.dot(smallCov, ma.transpose(expMatrix)))
            cfrisk = ma.sqrt(ma.diag(covMatrix) / 252.0)
            for (ii, idx) in enumerate(curNamesIdx):
                cfriskFull[0,idx] = cfrisk[ii]
            outfile = 'tmp/cfrisk-%s-%s.csv' % (date, self.mnemonic)
            dtList = [str(date)]
            Utilities.writeToCSV(cfriskFull, outfile, rowNames=dtList, columnNames=allNames)
            # Output specific risk
            sriskFull = numpy.zeros((1,len(allNames)), float)
            srisk = ma.sqrt(specificVars / 252.0)
            for (ii, idx) in enumerate(curNamesIdx):
                sriskFull[0,idx] = srisk[ii]
            outfile = 'tmp/srisk-%s-%s.csv' % (date, self.mnemonic)
            Utilities.writeToCSV(sriskFull, outfile, rowNames=dtList,
                    columnNames=allNames)
            # Output full cov matrix
            for ii in range(len(specificVars)):
                covMatrix[ii,ii] += specificVars[ii]
            outfile = 'tmp/cov-%s-%s.csv' % (date, self.mnemonic)
            Utilities.writeToCSV(covMatrix, outfile, rowNames=curNames, columnNames=curNames)

        data = Utilities.Struct()
        data.universe = currencyAssets
        data.estimationUniverseIdx = estu
        data.exposureMatrix = Matrices.ExposureMatrix(currencyAssets)
        data.exposureMatrix.addFactors(
            [f.name for f in self.blind],
            expMatrix.transpose(),
            Matrices.ExposureMatrix.StatisticalFactor)
        data.srMatrix = srMatrix
        data.factorCov = smallCov
        data.specificVars = specificVars
        self.log.debug('computeCurrencyRisk: end')
        return data

    def getDateList(self, date, modelDB):
        """Function that determines the list of dates for which
           to retrieve data to compute the currency model
        """
        (minOmegaObs, maxOmegaObs) = self.cp.getCovarianceSampleSize()
        # Deal with weekend dates (well, ignore them)
        dateList = modelDB.getDates(self.rmg, date, maxOmegaObs-1,
                excludeWeekend=True)
        dateList.reverse()
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
        return dateList

    def insertFactorCovariances(self, rmi, cov, modelDB):
        """Inserts the stat factor covariances into the database for the given
        risk model instance.
        cov is a stat factor-stat factor array of the covariances.
        """
        statSubFactors = modelDB.getSubFactorsForDate(rmi.date, self.blind)
        modelDB.insertFactorCovariances(rmi, statSubFactors, cov)

    def insertEstimationUniverse(self, rmi, universe, estU, modelDB):
        """Inserts the estimation universe into the database for the given
        risk model instance.
        universe is a list of sub-issues, estU contains the indices into
        universe of sub-issues in the estimation universe.
        """
        estUniv = [universe[i] for i in estU]
        modelDB.insertEstimationUniverse(rmi, estUniv, None)

    def insertExposures(self, rmi, data, modelDB, marketDB):
        """Insert the exposure matrix into the database for the given
        risk model instance.
        The exposure matrix is stored in data as returned by
        generateExposureMatrix().
        """
        statSubFactors = modelDB.getRiskModelInstanceSubFactors(
            rmi, self.blind)
        expMat = data.exposureMatrix
        expMat.data_ = ma.masked_where(expMat.data_==0.0, expMat.data_)
        assert(len(expMat.getFactorNames()) == len(statSubFactors))
        subFactorMap = dict(zip([f.name for f in self.blind], statSubFactors))
        if self.newExposureFormat:
            modelDB.insertFactorExposureMatrixNew(
                rmi, expMat, subFactorMap, update=False)
        else:
            modelDB.insertFactorExposureMatrix(rmi, expMat, subFactorMap)

    def insertSpecificReturns(self, date, specificReturns, subIssues,
                              modelDB):
        """Inserts the specific returns into the database for the given date.
        specificReturns is a masked array of the return values.
        subIssues is an array containing the corresponding sub-issue IDs.
        """
        assert(len(specificReturns.shape) == 1)
        assert(specificReturns.shape[0] == len(subIssues))
        indices = numpy.flatnonzero(ma.getmaskarray(specificReturns) == 0)
        subIssues = numpy.take(numpy.array(
            subIssues, dtype=object), indices, axis=0)
        specificReturns = ma.take(specificReturns, indices, axis=0)
        modelDB.insertSpecificReturns(self.rms_id, date, subIssues,
                                      specificReturns)

    def insertSpecificRisks(self, rmi_id, specificVariance, subIssues,
                            modelDB):
        """Inserts the specific risk into the database for the given date.
        specificVariance is a masked array of the specific variances.
        subIssues is an array containing the corresponding sub-issue IDs.
        """
        assert(len(specificVariance.shape) == 1)
        assert(specificVariance.shape[0] == len(subIssues))
        indices = numpy.flatnonzero(ma.getmaskarray(specificVariance) == 0)
        subIssues = numpy.take(numpy.array(
            subIssues, dtype=object), indices, axis=0)
        specificRisks = ma.sqrt(ma.take(specificVariance, indices, axis=0))
        modelDB.insertSpecificRisks(rmi_id, subIssues, specificRisks)

    def isFactorModel(self):
        return True

    def loadFactorCovarianceMatrix(self, rmi, modelDB):
        """Loads the factor-factor covariance matrix of the given risk
        model instance.
        Returns a (cov, factors) pair where cov is an m by m array
        containing the factor covariances and factors is a list of the
        m factor names.
        """
        statSubFactors = modelDB.getRiskModelInstanceSubFactors(rmi, self.blind)
        cov = modelDB.getFactorCovariances(rmi, statSubFactors)
        return (cov, self.blind)

    def loadExposureMatrix(self, rmi, modelDB):
        """Loads the exposure matrix of the given risk model instance.
        Returns an ExposureMatrix object.
        """
        statSubFactors = modelDB.getRiskModelInstanceSubFactors(rmi, self.blind)
        statSubFactorMap = dict([(s.factor.name, s) for s in statSubFactors])
        assets = self.createCurrencyAssets()
        # Set up an empty exposure matrix
        factorList = [(f.name, Matrices.ExposureMatrix.StatisticalFactor) for f in self.blind]
        expM = Matrices.ExposureMatrix(assets, factorList)
        if self.newExposureFormat:
            modelDB.getFactorExposureMatrixNew(rmi, expM, statSubFactorMap)
        else:
            modelDB.getFactorExposureMatrix(rmi, expM, statSubFactorMap)
        return expM

    def loadSpecificRisks(self, rmi, modelDB):
        """Loads the specific risks of the given risk model instance.
        Returns a dictionary mapping sub-issue IDs to their specific risks.
        """
        return modelDB.getSpecificRisks(rmi, restrictDates=False)

    def loadEstimationUniverse(self, rmi, modelDB):
        """Loads the specific risks of the given risk model instance.
        Returns a list of sub-issue IDs.
        """
        return modelDB.getRiskModelInstanceESTU(rmi)

class CurrencyRiskModel(BaseCurrencyModel):
    """A currency risk model.  This is a 'dense', asset-by-asset
    risk model where 'assets' are currencies.
    """
    # Instances should have rm_id, revision
    currencyFactorModel = False
    xrBnds = [-15.0, 15.0]
    newExposureFormat = False

    def __init__(self, modelDB, marketDB):
        BaseCurrencyModel.__init__(self, modelDB, marketDB)

    def insertCurrencyCovariances(self, rmi, cov, modelDB):
        """Inserts the factor covariances into the database for the given
        risk model instance.
        cov is a currency-currency array of the covariances.
        """
        currencyFactors = self.getCurrencyFactors()
        subFactors = modelDB.getSubFactorsForDate(rmi.date, currencyFactors)
        modelDB.insertFactorCovariances(rmi, subFactors, cov)

    def isFactorModel(self):
        return False

    def isCurrencyModel(self):
        return True

    def loadCurrencyCovarianceMatrix(self, rmi, modelDB, rebase=None):
        """Loads the covariance matrix of the given risk model instance.
        Returns a (cov, factors) pair where cov is an m by m array
        containing the covariances and factors is a list of the
        m ModelFactors.
        """
        currencyFactors = self.getCurrencyFactors()
        subFactors = modelDB.getSubFactorsForDate(rmi.date, currencyFactors)
        cov = modelDB.getFactorCovariances(rmi, subFactors)
        if rebase is not None and rebase != self.numeraire:
            currencyIdxMap = dict(zip(currencyFactors,
                                  range(len(currencyFactors))))
            idx = currencyIdxMap.get(rebase, None)
            assert(idx!=None)
            newCov = ma.zeros(cov.shape, numpy.float)
            for i in range(len(currencyFactors)):
                for j in range(i+1):
                    newCov[i,j] = cov[i,j] - cov[i,idx] \
                                - cov[j,idx] + cov[idx,idx]
                    newCov[j,i] = newCov[i,j]
            cov = newCov
        return (cov, currencyFactors)

    def computeCurrencyRisk(self, date, modelDB, marketDB):
        """Compute the currency covariance matrix.
        """
        self.log.debug('computeCurrencyRisk: begin')
        # Get some basic risk parameters
        if isinstance(self.covarianceCalculator,
                RiskCalculator.CompositeCovarianceMatrix2009):
            (minOmegaObs, maxOmegaObs) = self.cp.getCovarianceSampleSize()
        else:
            (minOmegaObs, maxOmegaObs) = self.rp.getCovarianceSampleSize()

        # Deal with weekend dates (well, ignore them)
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

        # Retrieve currency returns (in reverse chronological order)
        currencyFactors = self.getCurrencyFactors()
        subFactors = modelDB.getSubFactorsForDate(date, currencyFactors)
        cr = modelDB.loadFactorReturnsHistory(self.rms_id, subFactors, dateList)

        # Remove dates with lots of missing returns (eg. non-trading days)
        io = ma.sum(ma.getmaskarray(cr.data), axis=0)
        goodDatesIdx = numpy.flatnonzero(io < 0.5 * len(self.currencies))
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
        cov = self.covarianceCalculator.computeFactorCovarianceMatrix(cr)
        self.log.debug('computeCurrencyRisk: end')
        return cov
