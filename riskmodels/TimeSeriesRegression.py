import logging
import numpy.ma as ma
import numpy
import pandas
import datetime
import statsmodels.stats.outliers_influence as smout
from riskmodels import Matrices
from riskmodels import Utilities
from riskmodels import ProcessReturns
from riskmodels import ModelDB
from marketdb import MarketID

# Fixed Income Series used in TS Library
FIAXID_10Y = {'US': ('FI00001167', datetime.date(2000, 1, 4)),
              'GB': ('FI00000616', datetime.date(2000, 1, 4)),
              'EP': ('FI00000407', datetime.date(2000, 1, 4)),
              'JP': ('FI00000828', datetime.date(2000, 1, 5))}
FIAXID_6M = {'US': ('FI00001172', datetime.date(2000, 1, 4)),
             'GB': ('FI00000621', datetime.date(2000, 1, 4)),
             'EP': ('FI00000412', datetime.date(2000, 1, 4)),
             'JP': ('FI00000833', datetime.date(2000, 1, 5))}

FIAXID_BBB = {'EP': ('FI00000443', datetime.date(2004, 6, 22)),
              'JP': ('FI00000843', datetime.date(2006, 7, 14)),
              'US': ('FI00001185', datetime.date(2004, 1, 21)),
              'GB': ('FI00000631', datetime.date(2004, 6, 22))}

class TimeSeriesRegression:

    def __init__(self,
                 modelDB,
                 marketDB,
                 TSRParameters=dict(),
                 debugOutput=False,
                 fillWithMarket=True,
                 fillInputsWithZeros=True,
                 localRMG=None,
                 localSector=None,
                 forceRun=False,
                 getRegStats=False):
        """ Generic class for time-series regression
        """
        self.TSRParameters = TSRParameters      # Important regression parameters
        self.debugOutput = debugOutput          # Extra reporting
        self.localRMG = localRMG                # Asset's local RMG object
        self.localSector = localSector          # Assets' industry/sector where relevant
        self.forceRun = forceRun                # Override a few failsafes
        self.getRegStats = getRegStats
        self.modelDB = modelDB
        self.marketDB = marketDB
        self.fillWithMarket = fillWithMarket
        self.fillInputsWithZeros = fillInputsWithZeros

        # Dict of methods for retrieving inputs
        self.methods = {
                'Term Spread': self.getTermSpreadFD,
                'Long Term Rate': self.getLTRateFD,
                'Short Term Rate': self.getSTRateFD,
                'Credit Spread': self.getCreditSpreadFD,
                'Oil': self.getOilReturn,
                'Gold': self.getGoldReturn,
                'Commodity': self.getCommodityReturn,
                'BEI Rate': self.getBEIRateFD
        }

    def getRegressionType(self):
        return (self.TSRParameters.get('robust', True),
                self.TSRParameters.get('kappa', 5.0),
                self.TSRParameters.get('maxiter', 10))

    def getWeightingScheme(self):
        return (self.TSRParameters.get('weighting', None),
                self.TSRParameters.get('halflife', None),
                self.TSRParameters.get('fadePeak', None))

    def getHistoryLength(self):
        return self.TSRParameters.get('historyLength', None)

    def getReturnsParameters(self):
        return (self.TSRParameters.get('frequency', 'weekly'),
                self.TSRParameters.get('nobs', None),
                self.TSRParameters.get('minnobs', None),
                self.TSRParameters.get('fuzzyScale', None))

    def setUpReturnStructure(self, n, t, F, factors):
        # Create default return value
        retval = Utilities.Struct()

        retval.factors = factors
        retval.vifs = {}
        retval.params = numpy.zeros((F, n), float)
        retval.tstat = numpy.zeros((F, n), float)
        retval.pvals = numpy.ones((F, n), float)
        retval.sigma = numpy.zeros((n), float)
        retval.error = numpy.zeros((t, n), float)
        retval.rsquare = numpy.zeros((n), float)
        return retval

    def getWeights(self, t):
        # Set up weighting scheme for regression
        # Weights are in chronological order
        if self.weightingScheme is None:
            return None
        
        if self.weightingScheme.lower() == 'exponential':
            weights = Utilities.computeExponentialWeights(self.halflife, t)
            weights.reverse
        elif self.weightingScheme.lower() == 'triangle':
            weights = Utilities.computeTriangleWeights(self.halflife, t)
        else:
            weights = Utilities.computePyramidWeights(self.halflife, self.fadePeak, t)
        return weights

    def computeVIFs(self, X, factors):
        vifs = {}
        for (idx, factor) in enumerate(factors):
            if factor != 'Intercept':
                vifs[factor] = smout.variance_inflation_factor(X, 1)
        return vifs

    def getIndexReturn(self, axid, startDate, endDate):
        """Retrieve index returns"""

        level = self.marketDB.getMacroTs([axid], startDate-datetime.timedelta(10))
        level = pandas.Series(level[axid]).sort_index()
        ret = level / level.shift(1) - 1.
        if isinstance(ret.index, pandas.DatetimeIndex):
            ret.index = ret.index.date
        idx = [d for d in ret.index if d >= startDate and d <= endDate]
        return ret.reindex(index=idx)

    def getOilReturn(self, startDate, endDate, dummy):
        """Retrieve oil price returns in USD using
        Crude Oil-West Texas Intermediate Spot Cushing
        measured in USD Per Barrel"""

        minDate = datetime.date(1983, 1, 10)
        assert(startDate >= minDate)
        series = 'M000100022'
        return self.getIndexReturn(series, startDate, endDate)

    def getGoldReturn(self, startDate, endDate, dummy):
        """Retrieve gsci gold spot index returns"""

        minDate = datetime.date(1978, 1, 6)
        assert(startDate >= minDate)
        series = 'M000100093'
        return self.getIndexReturn(series, startDate, endDate)

    def getCommodityReturn(self, startDate, endDate, dummy):
        """Retrieve gsci non-energy spot index returns"""

        minDate = datetime.date(1969, 12, 31)
        assert(startDate >= minDate)
        series = 'M000100117'
        return self.getIndexReturn(series, startDate, endDate)

    def getBEIRateFD(self, startDate, endDate, dummy):
        """Retrieve changes in the US break even inflation rate
        as measured by the 30Y tenor of the Axioma US.USD.BEI.ZC curve"""

        axid = MarketID.MarketID(string='FI00001161')
        minDate = datetime.date(2001, 10, 12)
        assert(startDate >= minDate)
        level = self.marketDB.getFITs([axid])
        level = pandas.Series(level[axid.string]).sort_index()

        # compute first differences
        fd = level.diff().dropna()
        if isinstance(fd.index, pandas.DatetimeIndex):
            fd.index = fd.index.date
        idx = [d for d in fd.index if d >= startDate and d <= endDate]

        return fd.reindex(index=idx)

    def getCreditSpreadFD(self, startDate, endDate, factorCountry):

        axid = MarketID.MarketID(string=FIAXID_BBB[factorCountry][0])
        minDate = FIAXID_BBB[factorCountry][1]
        assert(startDate >= minDate)
        level = self.marketDB.getFITs([axid])
        level = pandas.Series(level[axid.string]).sort_index()

        fd = level.diff().dropna()
        if isinstance(fd.index, pandas.DatetimeIndex):
            fd.index = fd.index.date
        elif isinstance(fd.index[0], str):  # remove when we have marketdb data!
            fd.index = [d.date() for d in pandas.to_datetime(fd.index)]
        idx = [d for d in fd.index if d >= startDate and d <= endDate]

        return fd.reindex(index=idx)

    def getLTRateFD(self, startDate, endDate, factorCountry):
        """Retrieve changes in the yields of 10-year
        sovereign bonds with constant maturity"""

        axid = MarketID.MarketID(string=FIAXID_10Y[factorCountry][0])
        minDate = FIAXID_10Y[factorCountry][1]
        assert(startDate >= minDate)
        level = self.marketDB.getFITs([axid])
        level = pandas.Series(level[axid.string]).sort_index()

        # compute first differences
        fd = level.diff().dropna()
        if isinstance(fd.index, pandas.DatetimeIndex):
            fd.index = fd.index.date
        idx = [d for d in fd.index if d >= startDate and d <= endDate]

        return fd.reindex(index=idx)

    def getSTRateFD(self, startDate, endDate, factorCountry):
        """Retrieve changes in the yields of 6-month
        sovereign bonds with constant maturity"""

        axid = MarketID.MarketID(string=FIAXID_6M[factorCountry][0])
        minDate = FIAXID_6M[factorCountry][1]
        assert(startDate >= minDate)
        level = self.marketDB.getFITs([axid])
        level = pandas.Series(level[axid.string]).sort_index()

        # compute first differences
        fd = level.diff().dropna()
        if isinstance(fd.index, pandas.DatetimeIndex):
            fd.index = fd.index.date
        idx = [d for d in fd.index if d >= startDate and d <= endDate]

        return fd.reindex(index=idx)

    def getTermSpreadFD(self, startDate, endDate, factorCountry):
        """Retrieve changes in the spread between the yields
        of 10-year and 6-month sovereign bonds with constant
        maturity"""

        axids = [MarketID.MarketID(string=FIAXID_10Y[factorCountry][0]),
                 MarketID.MarketID(string=FIAXID_6M[factorCountry][0])]
        assert (startDate > FIAXID_10Y[factorCountry][1] and \
                startDate > FIAXID_6M[factorCountry][1])
        df = pandas.DataFrame.from_dict(self.marketDB.getFITs(axids)).sort_index(axis=0)
        level = df[FIAXID_10Y[factorCountry][0]] - \
                df[FIAXID_6M[factorCountry][0]]

        # compute first differences
        fd = level.diff().dropna()
        if isinstance(fd.index, pandas.DatetimeIndex):
            fd.index = fd.index.date
        idx = [d for d in fd.index if d >= startDate and d <= endDate]

        return fd.reindex(index=idx)

    def getRMG(self, rmgInput):
        if type(rmgInput) is int:
            return self.modelDB.getRiskModelGroup(rmgInput)
        elif type(rmgInput) is str:
            return self.modelDB.getRiskModelGroupByISO(rmgInput)
        else:
            return rmgInput

    def getRegion(self, regInput):
        if type(regInput) is int:
            return self.modelDB.getRiskModelRegion(regInput)
        elif type(regInput) is str:
            return self.modelDB.getRiskModelRegionByName(regInput)
        else:
            return regInput

    def getMarketReturn(self, startDate, endDate, rmg):
        """Retrieve RMG market returns"""
        mktDateList = self.modelDB.getDateRange(None, startDate, endDate)
        marketReturns = ma.filled(self.modelDB.loadRMGMarketReturnHistory(
                mktDateList, rmg, robust=False).data[0,:], 0.0)
        return pandas.Series(marketReturns, index=mktDateList)

    def getRegionReturn(self, startDate, endDate, reg):
        """Retrieve regional market returns"""
        mktDateList = self.modelDB.getDateRange(None, startDate, endDate)
        marketReturns = ma.filled(self.modelDB.loadRegionReturnHistory(\
                mktDateList, [reg]).data[0,:], 0.0)
        return pandas.Series(marketReturns, index=mktDateList)

    def getAMPReturn(self, startDate, endDate, amp_id, revision_id):
        frets = self.modelDB.loadAMPIndustryReturnHistory(
                amp_id, revision_id, startDate=startDate, endDate=endDate)
        frets = frets.toDataFrame().T
        if self.localSector is not None:
            frets = frets[self.localSector]
        if isinstance(frets.index, pandas.DatetimeIndex):
            frets.index = frets.index.date
        idx = [d for d in frets.index if d >= startDate and d <= endDate]
        return frets.reindex(index=idx)

    def getForexReturn(self, startDate, endDate, localISO, numeraire):
        dateList = self.modelDB.getDateRange(None, startDate, endDate)
        frets = self.modelDB.loadCurrencyReturnsHistory(\
                None, None, None, [localISO], numeraire, dateList=dateList, returnDF=True)
        return frets.loc[localISO, :].fillna(0.0)

    def processReturns(self, assetReturns, factorReturns, rmg):
        """Get commonDates in assetReturns and factorReturns and
        compound returns to match commonDates. Then get dates
        associated with desired frequency and compound returns
        to match that frequency. Optionally fill missing
        assetReturns with market return.
        Returns time series matrices: assetReturns, factorReturns"""

        (frequency, nobs, minnobs, fuzzyScale) = self.getReturnsParameters()
        # Get common dates and compound returns for those date
        fRets = factorReturns.toDataFrame().T.sort_index()
        allDateSets = [set(assetReturns.dates)]
        for factor in factorReturns.assets:
            allDateSets.append(set(fRets[factor].dropna().index))
        commonDates = sorted(set.intersection(*allDateSets))
        if len(commonDates) == 0:
            logging.error('Insufficient overlapping data are available to run the regression')
        assert(len(commonDates) > 0)

        # Compound returns so dates match across each set
        if assetReturns.dates != commonDates:
            assetReturns.data = ProcessReturns.compute_compound_returns_v4(
                    assetReturns.data[:, assetReturns.dates.index(min(commonDates)):],
                    assetReturns.dates[assetReturns.dates.index(min(commonDates)):],
                    commonDates,
                    fillWithZeros=False)[0]
        if factorReturns.dates != commonDates:
            factorReturns.data = ProcessReturns.compute_compound_returns_v4(
                    factorReturns.data[:, factorReturns.dates.index(min(commonDates)):],
                    factorReturns.dates[factorReturns.dates.index(min(commonDates)):],
                    commonDates,
                    fillWithZeros=self.fillInputsWithZeros)[0]

        # Get date list for specified frequency
        if (frequency == 'weekly') or (frequency == 'monthly'):
            periodDateList = Utilities.get_period_dates(self.modelDB, commonDates, frequency=frequency)
        else:
            periodDateList = list(commonDates)
        periodDateList.sort()

        # Compound returns to match specified frequency
        if commonDates != periodDateList:
            assetReturns.data = ProcessReturns.compute_compound_returns_v4(
                    assetReturns.data, commonDates, periodDateList, fillWithZeros=False)[0]
            factorReturns.data = ProcessReturns.compute_compound_returns_v4(
                    factorReturns.data, commonDates, periodDateList, fillWithZeros=self.fillInputsWithZeros)[0]

        # Take last nobs observations
        if minnobs is not None:
            nobs = max(factorReturns.data.shape[1], minnobs)
        if fuzzyScale is not None:
            # Some legacy descriptors have insufficient returns - so we need to lower our standards
            newNobs = int(fuzzyScale * nobs)
            logging.info('Changing minimum obs for legacy descriptors from %d to %d', nobs, newNobs)
            assert(factorReturns.data.shape[1] >= newNobs and \
                    assetReturns.data.shape[1] >= newNobs and
                    len(periodDateList) >= newNobs)
        else:
            assert(factorReturns.data.shape[1] >= nobs and \
                    assetReturns.data.shape[1] >= nobs and
                    len(periodDateList) >= nobs)
        factorReturns.data = factorReturns.data[:, -nobs:]
        assetReturns.data = assetReturns.data[:, -nobs:]
        finalDateList = periodDateList[-nobs:]

        # Report on number of missing return observations per asset
        #tmp = pandas.DataFrame(assetReturns.data, index=assetReturns.assets)
        #propNullC = tmp.isnull().sum(axis=1)/tmp.shape[1]
        #self.propNull = propNullC
        #if self.debugOutput:
        #    propNullC.to_csv('tmp/%s-propNull%sReturns-%s.csv' % \
        #            (rmg.mnemonic, frequency, max(assetReturns.dates).isoformat()))

        # Fill missing asset returns (ensuring each has a complete history)
        if self.fillWithMarket:
            if ('Market', 'Local') in factorReturns.assets:
                mktVals = factorReturns.data[factorReturns.assets.index(('Market', 'Local')), :]
            else:
                # Load relevant market returns if we don't already have them
                mktRets = self.getMarketReturn(min(commonDates), max(commonDates), [self.localRMG])
                mktVals = mktRets.values[numpy.newaxis, :]
                mktDates = list(mktRets.index)
                # Repeat all the date processing
                if mktDates != commonDates:
                    mktVals = ProcessReturns.compute_compound_returns_v4(
                            mktVals[:, mktDates.index(min(commonDates)):],
                            mktDates[mktDates.index(min(commonDates)):],
                            commonDates, fillWithZeros=self.fillInputsWithZeros)[0]

                if commonDates != periodDateList:
                    mktVals = ProcessReturns.compute_compound_returns_v4(
                            mktVals, commonDates, periodDateList)[0]
                mktVals = mktVals[0, -nobs:]
                assert(len(mktVals) == assetReturns.data.shape[1])
            # Use to fill missing asset returns
            maskedReturns = numpy.array(ma.getmaskarray(assetReturns.data), dtype='float')
            for ii in range(len(mktVals)):
                maskedReturns[:, ii] *= mktVals[ii]
            assetReturns.data = ma.filled(assetReturns.data, 0.0)
            assetReturns.data += maskedReturns
        else:
            assetReturns.data = ma.filled(assetReturns.data, 0.0)
        assert (assetReturns.data.shape[1] == factorReturns.data.shape[1])

        # Return reformatted time series matrices
        retFactorReturns = Matrices.TimeSeriesMatrix(factorReturns.assets, finalDateList)
        retFactorReturns.data = factorReturns.data
        retAssetReturns = Matrices.TimeSeriesMatrix(assetReturns.assets, finalDateList)
        retAssetReturns.data = assetReturns.data
        return (retAssetReturns, retFactorReturns)

    def TSR(self, assetReturns, *regInputs):
        """ Internal organs for time-series regression
        """
        logging.debug('TSR: begin')
        startDate, endDate = (min(assetReturns.dates), max(assetReturns.dates))
        # Load the various right-hand-side returns
        inputs = {}
        for inputSet in regInputs:
            for itm in inputSet:
                logging.debug('Adding factor: (%s)', itm)
                if itm[0].lower() == 'market':
                    if (type(itm[1]) is str) and (itm[1].lower() == 'local'):
                        inputs[itm] = self.getMarketReturn(startDate, endDate, self.localRMG)
                    else:
                        rmgObj = self.getRMG(itm[1])
                        inputs[itm] = self.getMarketReturn(startDate, endDate, rmgObj)
                elif itm[0].lower() == 'region':
                    regObj = self.getRegion(itm[1])
                    inputs[itm] = self.getRegionReturn(startDate, endDate, regObj)
                elif itm[0].lower() == 'amp':
                    inputs[itm] = self.getAMPReturn(startDate, endDate, itm[1], itm[2])   
                elif itm[0].lower() == 'forex':
                    localISO = self.localRMG.currency_code
                    inputs[itm] = self.getForexReturn(startDate, endDate, localISO, itm[1])
                else:
                    inputs[itm] = self.methods[itm[0]](startDate, endDate, itm[1])
                logging.debug('Loaded %d values from %s to %s', len(inputs[itm]), startDate, endDate)

        # Build total set of factor returns
        df = pandas.DataFrame.from_dict(inputs, orient='index').sort_index(axis=1)
        df = df[sorted(list(df.columns))]
        factorReturns = Matrices.TimeSeriesMatrix.fromDataFrame(df)

        # Process returns to rationalise dates and fill missing values where appropriate
        (assetReturns, factorReturns) = self.processReturns(assetReturns, factorReturns, self.localRMG)

        # Perform time-series regression for each subset of factors
        retValSet = []
        assetReturnsCopy = Matrices.TimeSeriesMatrix(assetReturns.assets, assetReturns.dates)
        assetReturnsCopy.data = ma.array(assetReturns.data, copy=True)
        for (iterNo, inputSet) in enumerate(regInputs):
            logging.debug('Regression %d, factors: %s', (iterNo+1), inputSet)
            subFactorReturns = Matrices.TimeSeriesMatrix.fromDataFrame(factorReturns.toDataFrame().reindex(index=inputSet))
            retVal = self.TSR_inner(assetReturnsCopy, subFactorReturns, iterNo=iterNo)
            assetReturnsCopy.data = ma.transpose(retVal.resid)
            retValSet.append(retVal)

        logging.debug('TSR: end')
        if len(retValSet) < 2:
            return retValSet[0]
        return retValSet

    def TSR_inner(self, assetReturns, factorReturns, iterNo=0):
        # Perform the time-series regression itself

        # Format input data
        T = assetReturns.data.shape[1]
        if self.TSRParameters.get('inclIntercept', True):
            X = numpy.transpose(ma.append(factorReturns.data, ma.ones((1, T), float), 0))
            factors = factorReturns.assets + ['Intercept']
        else:
            X = numpy.transpose(factorReturns.data)
            factors = factorReturns.assets
        y = numpy.transpose(assetReturns.data)
        assets = assetReturns.assets
        periodDateList = assetReturns.dates
        assert(y.shape[0] == X.shape[0]) and (len(factors) == X.shape[1])
      
        # Output regression matrices for debugging
        if self.debugOutput:
            pandas.DataFrame(y, index=periodDateList[-y.shape[0]:], 
                columns=[a.getSubIdString() for a in assets]).to_csv('tmp/y-%s.csv' % iterNo)
            pandas.DataFrame(X, index=periodDateList[-X.shape[0]:], 
                columns=factors).to_csv('tmp/X-%s.csv' % iterNo)

        # Compute VIFs
        if self.getRegStats:
            if len(factors) > 1:
                vifs = self.computeVIFs(X, factors)
            else:
                vifs = None

        # Get model parameters
        self.historyLength = self.getHistoryLength()
        (self.robustBeta, self.kappa, self.maxiter) = self.getRegressionType()
        (self.weightingScheme, self.halflife, self.fadePeak) = self.getWeightingScheme()

        # Set up dimensions
        (t, n) = y.shape
        F = X.shape[1] # number of factors
        retval = self.setUpReturnStructure(n, t, F, factors)

        # Get weights if any
        weights = self.getWeights(t)
        if self.debugOutput and weights is not None:
             pandas.Series(weights, index=periodDateList[-len(weights):]).to_csv('tmp/weights-%s.csv' % iterNo, header=False)
        
        # Perform main regression
        if weights is not None:
            weights = [w*w for w in weights]
        res = Utilities.robustLinearSolver(y, X,
                        robust=self.robustBeta, k=self.kappa, 
                        maxiter=self.maxiter, weights=weights,
                        computeStats=self.getRegStats)
        
        params = Utilities.screen_data(res.params)
        if self.getRegStats:
            dof = {'noObs': t, 'noFactors': F, 'dof': t-F}

        # Compute historical sigma (residual variance)
        sigma = Matrices.allMasked(n)
        for j in range(len(sigma)):
            sigma[j] = ma.inner(res.error[:,j], res.error[:,j]) / (t - 1.) 

        # Set return values
        retval.weights = weights
        retval.sigma = sigma
        retval.params = params
        if self.getRegStats:
            retval.tstat = res.tstat
            retval.pvals = res.pvals
            retval.vifs = vifs
            retval.dof = dof
            retval.factors = factors
            retval.rsquare = res.rsquare
            retval.rsquarecentered = res.rsquarecentered
            retval.ser = numpy.sqrt(numpy.diag((res.error.T).dot(res.error))/(t-F))
            retval.X = pandas.DataFrame(X, index=periodDateList[-X.shape[0]:], columns=factors)
            retval.y = pandas.DataFrame(y, index=periodDateList[-y.shape[0]:], columns=[a.getSubIdString() for a in assets])
            if weights is None:
                retval.w = None
            else:
                retval.w = pandas.Series(weights, index=periodDateList[-len(weights):])
        retval.resid = res.error
        retval.residAssets = assets
        retval.residDates = periodDateList[-y.shape[0]:]

        return retval
