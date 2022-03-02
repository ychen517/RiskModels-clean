from __future__ import absolute_import
import logging
import numpy.ma as ma
import numpy
from riskmodels import Matrices
from riskmodels import Utilities
from riskmodels import ProcessReturns
from riskmodels import ModelDB

class LegacyTimeSeriesRegression:

    def __init__(self,
                 TSRParameters=dict(),
                 fillWithMarket=False,
                 outputDateList=None,
                 marketReturns=None,
                 debugOutput=False,
                 forceRun=False,
                 marketRegion=None,
                 marketBaseRegion=None,
                 getRegStats=False):
        """ Generic class for time-series regression
        """
        self.TSRParameters = TSRParameters      # Important regression parameters
        self.fillWithMarket = fillWithMarket    # Fill missing asset returns with market
        self.outputDateList = outputDateList    # Set of dates we wish to match
        self.marketReturns = marketReturns      # Pre-defined market returns
        self.debugOutput = debugOutput          # Extra reporting
        self.forceRun = forceRun                # Override a few failsafes
        self.marketRegion = marketRegion        # Regional market returns for regional models
        self.marketBaseRegion = marketBaseRegion
        self.getRegStats = getRegStats          # Compute regression stats (slower)

    def getSWAdj(self):
        return self.TSRParameters.get('swAdj', False)

    def getLag(self):
        if 'lag' in self.TSRParameters:
            if (self.TSRParameters['lag'] is None) or \
                    (self.TSRParameters['lag'] == 0):
                return False
            return self.TSRParameters['lag']
        else:
            return False

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

    def setUpReturnStructure(self, n):
       # Create default return value
        retval = Utilities.Struct()
        if self.lag != False:
            retval.beta = numpy.zeros((n, len(self.lag)), float)
            retval.tstat = numpy.zeros((n, len(self.lag)), float)
            retval.pvals = numpy.ones((n, len(self.lag)), float)
        else:
            retval.beta = numpy.zeros((n), float)
            retval.tstat = numpy.zeros((n), float)
            retval.pvals = numpy.ones((n), float)
        retval.alpha = numpy.zeros((n), float)
        retval.sigma = numpy.zeros((n), float)
        retval.resid = numpy.zeros((n), float)
        return retval

    def getMarketReturn(self, modelDB, initialDateList):
        # Load market returns
        if self.marketReturns == None:
            mktDateList = modelDB.getDateRange(None, min(initialDateList), max(initialDateList))
            if self.marketRegion is not None:
                marketReturns = ma.filled(modelDB.loadRegionReturnHistory(
                    mktDateList, [self.marketRegion]).data[0,:], 0.0)
                if self.marketBaseRegion is not None:
                    baseMarketReturns = ma.filled(modelDB.loadRegionReturnHistory(
                        mktDateList, [self.marketBaseRegion]).data[0,:], 0.0)
                    marketReturns = marketReturns - baseMarketReturns
            else:
                marketReturns = ma.filled(modelDB.loadRMGMarketReturnHistory(
                                    mktDateList, self.rmg, robust=False).data[0,:], 0.0)
        else:
            mktDateList = self.marketReturns.dates
            marketReturns = ma.filled(self.marketReturns.data, 0.0)
            if len(marketReturns.shape) > 1:
                marketReturns = marketReturns[0,:]
            if self.historyLength is not None:
                marketReturns = marketReturns[-self.historyLength:]

        return marketReturns, mktDateList

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

    def performScholesWilliamsAdjustment(self, x, y, b0, assetReturns, weights, univ):
        """ Do the regressions on lead/lag data and perform
        the adjustment according to Scholes Williams
        """

        # Market lagging asset returns...
        x_lag = x[:-1,:]
        y_lag = y[1:,:]
        if weights is None:
            w_lag = None
        else:
            w_lag = weights[1:]
        b_lag = Utilities.robustLinearSolver(y_lag, x_lag,
                    robust=self.robustBeta, k=self.kappa, maxiter=self.maxiter,
                    weights=w_lag, computeStats=False).params
        b_lag = b_lag[1:,:]

        # Market leading asset returns...
        x_lead = x[1:,:]
        y_lead = y[:-1,:]
        if weights is None:
            w_lead = None
        else:
            w_lead = weights[:-1]
        b_lead = Utilities.robustLinearSolver(y_lead, x_lead,
                    robust=self.robustBeta, k=self.kappa, maxiter=self.maxiter,
                    weights=w_lead, computeStats=False).params
        b_lead = b_lead[1:,:]

        # Put it all together...
        xCombined = ma.transpose(ma.array([x_lag[:,1], x_lead[:,1]]))
        corr = numpy.corrcoef(xCombined, rowvar=False)[0,1]
        k = 1.0 + 2.0 * corr

        # Set up shrinkage parameters
        if self.scholes_williams_adj_fix:
            lag_mean = numpy.average(b_lag, axis=1)[:,numpy.newaxis]
            lead_mean = numpy.average(b_lead, axis=1)[:,numpy.newaxis]
            logging.info('SW Statistics BEFORE, Kappa, %s, Mean_lag, %s, STD_lag, %s, Mean_lead, %s, STD_lead, %s',
                    k, lag_mean, numpy.std(b_lag, axis=None), lead_mean, numpy.std(b_lead, axis=None))

            shrink_fac = 0.6666667
            target_k = 1.0
            shrinkToZero = True
            if shrinkToZero:
                target_lag = 0.0
                target_lead = 0.0
            else:
                target_lag = lag_mean
                target_lead = lead_mean
            b_old = (b_lag + b0[1:,:] + b_lead) / k

            # Shrink the k value towards one
            k = (shrink_fac * k) + ((1.0 - shrink_fac) * target_k)

            # Shrink lead and lag betas
            b_lag = (shrink_fac * b_lag) + ((1.0 - shrink_fac) * target_lag)
            b_lead = (shrink_fac * b_lead) + ((1.0 - shrink_fac) * target_lead)

            lag_mean = numpy.average(b_lag, axis=1)[:,numpy.newaxis]
            lead_mean = numpy.average(b_lead, axis=1)[:,numpy.newaxis]
            logging.info('SW Statistics AFTER, Kappa, %s, Mean_lag, %s, STD_lag, %s, Mean_lead, %s, STD_lead, %s',
                    k, lag_mean, numpy.std(b_lag, axis=None), lead_mean, numpy.std(b_lead, axis=None))

        # Quick fix for flipping autocorrelation
        k = abs(k)

        # Perform the actual adjustment
        t = y.shape[0]
        if k != 0.0:

            beta = (b_lag + b0[1:,:] + b_lead) / k
            # New alphas
            meanReturns = ma.sum(y[1:-1,:], axis=0) / (t-2.0)
            meanMktReturn = ma.sum(x[1:-1,1], axis=0) / (t-2.0)
            alpha = (meanReturns - beta * meanMktReturn)[0,:]

            # Update residuals
            bb = numpy.zeros(b0.shape)
            bb[0,:] = alpha
            bb[1:,:] = beta
            resid = assetReturns - numpy.transpose(numpy.dot(x, bb))
        
        if self.scholes_williams_adj_fix:
            logging.info('BETA Statistics, BEFORE, mean, %s, STD, %s, AFTER, mean, %s, STD, %s',
                    numpy.average(b_old, axis=1), numpy.std(b_old, axis=None),
                    numpy.average(beta, axis=1), numpy.std(beta, axis=None))

        return alpha, beta, resid

    def TSR(self, rmg, returns, modelDB, marketDB, swFix=False):
        """ Internal organs for time-series regression
        Computes parameters for the Market Model using the given
        returns, the given RiskModelGroup's market returns, assets
        in returns.assets, and the time range given by dateList.
        The dates in the TimeSeriesMatrix returns do not have to be
        perfectly aligned with dateList, as long as every date in
        dateList is covered.
        The Scholes-Williams (1977) adjustment with a lead/lag of one
        will be applied if self.scholes_williams_adj=True.

        Returns (beta, alpha, sigma, residuals) for all assets.
        residuals is an asset-by-time array, sigma is the estimated
        variance of the residuals, and alpha is the intercept term.
        If n by t array marketReturns is specified, this will be used for
        beta computation, otherwise the routine will load the appropriate
        set.
        If lag is specfied, it should be a list of lags to be used,
        the routine computes a VAR model regressing current returns
        against a set of lagged market returns
        and will return a list of betas to these lags
        """
        logging.debug('TSR: begin')

        # Determine whether it's a single market or not
        if type(rmg) is not list:
            rmg = [rmg]
        if len(rmg) > 1:
            mnem = self.marketRegion.name
        else:
            mnem = rmg[0].mnemonic
        self.rmg = rmg
        self.mnem = mnem

        # Get model parameters
        self.scholes_williams_adj = self.getSWAdj()
        self.scholes_williams_adj_fix = swFix
        self.lag = self.getLag()
        self.historyLength = self.getHistoryLength()
        (self.robustBeta, self.kappa, self.maxiter) = self.getRegressionType()
        (self.weightingScheme, self.halflife, self.fadePeak) = self.getWeightingScheme()

        # Set up dimensions and dates
        assets = list(returns.assets)
        n = len(assets)
        if self.historyLength is None:
            initialDateList = list(returns.dates)
        else:
            initialDateList = returns.dates[-self.historyLength:]

        # Create default return value
        retval = self.setUpReturnStructure(n)

        # Debugging info
        if self.debugOutput:
            dateStr = [str(d) for d in initialDateList]
            idList = [s.getSubIDString() for s in assets]
            if self.historyLength is not None:
                initialReturns = returns.data[:, -self.historyLength:]
            else:
                initialReturns = returns.data
            Utilities.writeToCSV(initialReturns, 'tmp/initial-assret-%s-%s.csv' % \
                    (mnem, dateStr[-1]), columnNames=dateStr, rowNames=idList)

        # Copy asset returns to temporary array
        assetReturns = ma.array(returns.data, copy=True)

        # Do some checking of history lengths
        if n < 1:
            logging.warning('Too few assets (%d) in RiskModelGroup', n)
            return retval
        if n < 2 and len(assetReturns.shape) < 2:
            assetReturns = assetReturns[numpy.newaxis, :]

        if len(initialDateList) != assetReturns.shape[1]:
            logging.warning('Number of dates (%d) not equal to length of returns history (%d)',
                    len(initialDateList), assetReturns.shape[1])
        if self.historyLength is not None:
            assetReturns = assetReturns[:, -self.historyLength:]
        assert(len(initialDateList) == assetReturns.shape[1])

        # Load the given RiskModelGroup's daily market returns if not already given
        (marketReturns, mktDateList) = self.getMarketReturn(modelDB, initialDateList)

        # Compound returns for invalid dates into the following trading-day
        commonDates = list(set(mktDateList).intersection(set(initialDateList)))
        commonDates.sort()
        if initialDateList != commonDates:
            assetReturns = ProcessReturns.compute_compound_returns_v3(
                    assetReturns, initialDateList, commonDates, keepFirst=True)[0]
        if mktDateList != commonDates:
            marketReturns = ProcessReturns.compute_compound_returns_v3(
                    marketReturns, mktDateList, commonDates, keepFirst=True)[0]

        # Report on market returns used
        if len(self.rmg) > 1:
            logging.debug('Market model for %s (n: %d, t: %d)',
                    self.marketRegion.name, n, len(commonDates))
        else:
            logging.debug('Market model for %s (RiskModelGroup %d, %s) (n: %d, t: %d)',
                    self.rmg[0].description, self.rmg[0].rmg_id, self.rmg[0].mnemonic, n, len(commonDates))

        # Compound asset and market returns to match frequency of outputDateList
        if self.outputDateList is not None:
            self.outputDateList.sort()
            if commonDates != self.outputDateList:
                marketReturns = ProcessReturns.compute_compound_returns_v3(
                        marketReturns, commonDates, self.outputDateList)[0]
                assetReturns = ProcessReturns.compute_compound_returns_v3(
                        assetReturns, commonDates, self.outputDateList)[0]
        else:
            self.outputDateList = commonDates

        # Fix for assets with deficient histories
        if self.fillWithMarket:
            # Ensure that each asset has a complete history
            # by filling missing values with the market return
            maskedReturns = numpy.array(ma.getmaskarray(assetReturns), dtype='float')
            for ii in range(len(marketReturns)):
                maskedReturns[:,ii] *= marketReturns[ii]
            assetReturns = ma.filled(assetReturns, 0.0)
            assetReturns += maskedReturns
        else:
            assetReturns = ma.filled(assetReturns, 0.0)

        # Debugging info
        if self.debugOutput:
            dateStr = [str(d) for d in self.outputDateList][:len(marketReturns)]
            idList = [s.getSubIDString() for s in assets]
            Utilities.writeToCSV(marketReturns[numpy.newaxis,:], 'tmp/mktret-%s-%s.csv' % \
                    (self.mnem, dateStr[-1]), columnNames=dateStr, rowNames=[self.mnem])
            Utilities.writeToCSV(assetReturns, 'tmp/assret-%s-%s.csv' % \
                    (self.mnem, dateStr[-1]), columnNames=dateStr, rowNames=idList)

        # Check on dimensions
        t = assetReturns.shape[1]
        assert(len(marketReturns) == t)

        # Checks on missing returns
        if self.forceRun:
            if numpy.sum(ma.getmaskarray(ma.masked_where(
                marketReturns==0.0, marketReturns))) == t:
                logging.warning('All market returns missing or zero')
                retval.beta = ma.masked_where(retval.beta==0.0, retval.beta)
                return retval
        else:
            assert(numpy.sum(ma.getmaskarray(ma.masked_where(
                marketReturns==0.0, marketReturns))) < t)

        # If we're doing a lagged regression:
        if self.lag != False:
            # Construct a lags-by-t RHS matrix
            marketReturns_matrix = numpy.zeros((len(self.lag)+1,t), float)
            marketReturns_matrix[0,:] = 1.0
            for (i,l) in enumerate(self.lag):
                marketReturns_matrix[i+1,l:] = marketReturns[:-l]
            assetReturns = assetReturns[:,l:]
            marketReturns_matrix = marketReturns_matrix[:,l:]
            t -= l
            x = numpy.transpose(marketReturns_matrix)
        else:
            x = numpy.transpose(ma.array(
                [ma.ones(t, float), marketReturns]))

        # Get weights if any
        weights = self.getWeights(t)

        # Perform main regression
        y = numpy.transpose(assetReturns)
        res = Utilities.robustLinearSolver(y, x,
                        robust=self.robustBeta, k=self.kappa, maxiter=self.maxiter,
                        weights=weights, computeStats=self.getRegStats)

        dof = {'noObs': x.shape[0], 
               'noFactors': x.shape[1], 
               'dof': x.shape[0]-x.shape[1]}
        factors = ['alpha', 'beta']
        if self.lag != False:
            factors.extend(['beta_' + str(l+1) for l in range(self.lag)])
        
        # Apply Scholes-Williams adjustment if required
        if self.scholes_williams_adj:
            alpha, beta, resid = self.performScholesWilliamsAdjustment(x, y, res.params, assetReturns, weights, assets)
        else:
            alpha = res.params[0,:]
            beta = res.params[1:,:]
            resid = ma.transpose(res.error)
        beta = Utilities.screen_data(beta)

        # Compute historical sigma
        sigma = Matrices.allMasked(assetReturns.shape[0])
        for j in range(len(sigma)):
            sigma[j] = ma.inner(resid[j,:],resid[j,:]) / (t - 1.0)

        # Set return values
        retval.alpha = alpha
        retval.sigma = sigma
        retval.resid = resid
        if self.lag is not False:
            retval.beta = beta
            if self.getRegStats:
                retval.tstat = res.tstat[1:,:]
                retval.pvals = res.pvals[1:,:]
                retval.rsquare = res.rsquare
                retval.dof = dof
                retval.factors = factors[1:]

        else:
            retval.beta = beta[0,:]
            if self.getRegStats:
                retval.tstat = res.tstat[1,:]
                retval.pvals = res.pvals[1,:]
                retval.rsquare = res.rsquare
                retval.dof = dof
                retval.factors = [factors[1]]

        logging.debug('TSR: end')
        return retval
