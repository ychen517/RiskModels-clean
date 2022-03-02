
import logging
import numpy.ma as ma
import numpy.linalg as linalg
import numpy
import scipy.stats as stats
import os
import datetime
from riskmodels import Matrices
from riskmodels.Matrices import ExposureMatrix
from riskmodels import LegacyUtilities as Utilities

def calc_DailyInterestRate(annRate):
    dailyRate = pow(1.0 + annRate, 1.0 / 252.0) - 1.0
    return dailyRate

def calc_Weights(rmg, modelDB, marketDB, modelDate, universe, baseCurrencyID=None):
    if len(rmg) > 1:
        dateList = modelDB.getDates(rmg, modelDate, 250)
        nonWeekDays = len([d for d in dateList if d.weekday() > 4])
        goodRatio = 1.0 - float(nonWeekDays) / 250
        if goodRatio > 0.9:
            goodRatio = 1.0
        needDays = int(20 / goodRatio)
    else:
        needDays = 20
    mcapDates = modelDB.getDates(rmg, modelDate, needDays-1)
    nAssets = len(universe)
    avgMktCaps = modelDB.getAverageMarketCaps(mcapDates, universe, baseCurrencyID, marketDB)
    weights = ma.sqrt(avgMktCaps)
    C = min(100, int(round(nAssets*0.05)))
    sortindex = ma.argsort(weights)
    ma.put(weights, sortindex[nAssets-C:nAssets],
                weights[sortindex[nAssets-C]])
    return weights

class RegressionANOVA:
    def __init__(self, data, resid, nvars, estu, weights=None, deMean=False):
        self.log = logging.getLogger('ReturnsCalculator.RegressionANOVA')
        self.data_ = data
        self.resid_ = resid
        self.nvars_ = nvars
        if weights is None:
            self.weights_ = numpy.ones(len(estu), float)
        else:
            assert(len(weights)==len(estu))
            self.weights_ = weights
        self.estU_ = estu
        self.regressStats_ = Matrices.allMasked((nvars, 3))

        # Standard ANOVA stuff
        y = ma.take(self.data_, self.estU_, axis=0)
        # Note: SAS does not de-mean
        if deMean:
            ybar = ma.inner(self.weights_, y) / ma.sum(self.weights_, axis=0)
            y = y - ybar 
        y = y * ma.sqrt(self.weights_)
        self.sst_ = float(ma.inner(y, y))
        e = ma.take(self.resid_, self.estU_, axis=0)
        e = e * ma.sqrt(self.weights_)
        self.sse_ = float(ma.inner(e, e))
        self.ssr_ = self.sst_ - self.sse_

    def getANOVA(self):
        """Returns values from a typical linear regression ANOVA.
        Value is a dictionary with 3 keys - 'regression', 'residual',
        and 'total', each containing a tuple with 2 entries:
        sum of squares, degrees of freedom.
        """
        retval = {}
        df_ssr = self.nvars_
        df_sse = len(self.estU_) - self.nvars_
        df_sst = len(self.estU_)
        retval['regression'] = (self.ssr_, df_ssr)
        retval['residual'] = (self.sse_, df_sse)
        retval['total'] = (self.sst_, df_sst)
        fStat = (self.ssr_ / df_ssr) / (self.sse_ / df_sse)
        logging.info('F-statistic=%f, degrees of freedom=(%d, %d)', fStat, df_ssr, df_sse)
        return retval

    def calc_adj_rsquared(self):
        """Computes adjusted R-squared.
        """
        r = 0.0
        adjr = 0.0
        if self.sst_ > 0.0:
            r = 1.0 - self.sse_ / self.sst_
        logging.info('Unadjusted R-Squared=%f', r)
        n = len(self.weights_)
        if n > self.nvars_:
            adjr = max(1.0 - (1.0-r)*(n-1)/(n-self.nvars_), 0.0)
        logging.info('Adjusted R-Squared=%f', adjr)
        return adjr

    def calc_regression_statistics(self, estimates, exposureMatrix,
                                   constraintMatrix=None, white=False):
        """Computes regression statistics manually if they 
        have not been loaded already. eg. for a statistical model.
        """
        e = ma.take(self.resid_, self.estU_, axis=0)
        df = len(self.data_) - self.nvars_
        assert(exposureMatrix.shape[0]==len(self.weights_))
        self.regressStats_[:,0] = Utilities.computeRegressionStdError(
                    e, exposureMatrix, self.weights_, 
                    constraintMatrix, white=white)
        self.regressStats_[:,1] = estimates / self.regressStats_[:,0]
        self.regressStats_[:,2] = 1.0 - \
                    numpy.array(stats.t.cdf(abs(self.regressStats_[:,1]), df))
        return self.regressStats_

    def store_regression_statistics(self, stdErr, tStats, pVals):
        """Stores regression statistics into regressStats_.
        Assumes these are computed externally, eg. from R.
        """
        assert(len(stdErr)==self.nvars_ and len(tStats)==self.nvars_ \
               and len(pVals)==self.nvars_)
        self.regressStats_[:,0] = stdErr
        self.regressStats_[:,1] = tStats
        self.regressStats_[:,2] = pVals

class StatsmodelsRegression(object):
    def __init__(self, model, result, sm):
        self.model = model
        self.result = result
        self.sm = sm
    
    def getWeights(self):
        return numpy.array(self.result.weights)

    def getKappa(self):
        #TODO: not clear this is correct.  Doesn't match output from R
        return numpy.sqrt(numpy.linalg.cond(self.model.normalized_cov_params))

    def getCoefficients(self):
        return numpy.array(self.result.params)

    def getRegressionStats(self):
        params = self.result.params
        bse = self.result.bse
        regressStats = numpy.zeros((params.shape[0],2))
        regressStats[:,0] = bse
        regressStats[:,1] = params/bse
        return regressStats

class RobustRegression2:
    def __init__(self, params):
        self.log = logging.getLogger('ReturnCalculator.RobustRegression2')
        self.parameters = params

    def __regressWithStatsmodels(self, rd):
        model = self.sm.OLS(numpy.array(rd.regressionLHS),numpy.transpose(numpy.array(rd.regressionRHS)))
        result = model.fit()
        return StatsmodelsRegression(model, result, self.sm)

    def __robustRegressWithStatsmodels(self, rd):
        k = self.parameters.getRlmKParameter()
        model = self.sm.RLM(numpy.array(rd.regressionLHS), \
                numpy.transpose(numpy.array(rd.regressionRHS)),
                M=self.sm.robust.norms.HuberT(t=k))
        result = model.fit(maxiter=500)
        return StatsmodelsRegression(model, result, self.sm)

    def __initStatsmodels(self):
        try:
            import statsmodels.api as sm
        except ImportError:
            import scikits.statsmodels.api as sm
        except:
            import scikits.statsmodels as sm
        self.sm = sm

    def processFactorIndices(self, rd, factorNames, factorReturns, constraints, expM,
                                removeNTDsFromRegression=True):
        """Routine to sort out the thin and empty from the big and fat
        Also sets up constraints for singular regressions
        """
        factorIndices = [i for i in range(len(factorReturns))]
        # Invoke thin factor correction mechanism if required
        if self.parameters.getThinFactorCorrection():
            (rd.regressorMatrix, rd.excessReturns_ESTU, rd.weights_ESTU,
             factorReturns, thinFactorIndices, emptyFactorIndices, rd.dummyRetWeights) = \
                    self.thinFactorAdjustment.insertDummyAssets(
                        rd.regressorMatrix, rd.weights_ESTU,
                        rd.excessReturns_ESTU, factorReturns, dummyRetWeights=rd.dummyRetWeights)
            factorIndices = [i for i in factorIndices if i not in emptyFactorIndices]
        else:
            thinFactorIndices = []
            emptyFactorIndices = []
        if len(thinFactorIndices)==0:
            self.log.info('No thin factors!')

        # Remove any remaining empty factors
        for idx in factorIndices:
            assetsIdx = numpy.flatnonzero(rd.regressorMatrix[idx,:])
            if len(assetsIdx) == 0:
                factorReturns[idx] = 0.0
                self.log.warning('Empty factor: %s', factorNames[idx])
                emptyFactorIndices.append(idx)
        factorIndices = [i for i in factorIndices if i not in emptyFactorIndices]

        # Remove factors with all exposures equal to 1 unless this is
        # the market intercept regression
        interceptFactorIndices = [i for i in factorIndices if \
                (sum(rd.regressorMatrix[i,:])==len(rd.excessReturns_ESTU))]
        if len(factorIndices) > 1:
            if expM is None:
                officialInterceptFactorList = list()
            else:
                officialInterceptFactorList = expM.getFactorNames(ExposureMatrix.InterceptFactor)
            factorIndices = [i for i in factorIndices if (\
                    (i not in interceptFactorIndices) or \
                    (factorNames[i] in officialInterceptFactorList))]
            for i in interceptFactorIndices:
                if factorNames[i] not in officialInterceptFactorList:
                    self.log.warning('Factor %s has become an intercept and will be removed',
                                    factorNames[i])

        # Add constraint(s) to deal with deliberately singular regression
        numConstraints = 0
        totalDummies = len(thinFactorIndices)
        for rc in constraints:
            rc.factorWeights = None
            factorIdx = [i for i in factorIndices if factorNames[i] in \
                         expM.getFactorNames(rc.factorType)]
            if len(factorIdx) > 0:
                (rd.regressorMatrix, rd.excessReturns_ESTU, rd.weights_ESTU) = \
                        rc.createDummyAsset(factorIdx, rd.excessReturns_ESTU,
                                            rd.regressorMatrix, rd.weights_ESTU, totalDummies,
                                            removeNTDsFromRegression=removeNTDsFromRegression)
                totalDummies += 1
                numConstraints += 1

        self.log.info('%i Factors, %i intercept, %i thin, %i empty, %i used',
                len(factorReturns), len(interceptFactorIndices), len(thinFactorIndices),
                len(emptyFactorIndices), len(factorIndices))
        # Pick out final regressor matrix
        rd.regressorMatrix = ma.take(rd.regressorMatrix, factorIndices, axis=0)
        # And return everything
        rd.factorIndices = factorIndices
        rd.thinFactorIndices = thinFactorIndices
        rd.emptyFactorIndices = emptyFactorIndices
        rd.numConstraints = numConstraints
        return (rd, factorReturns)

    def check_regression_constraints(self, rd, factorNames, factorReturns,
                            constraints, maxTolerance=1.0e-4):
        """Performs check on whether regression constraints are being met
        increases weight if not
        """
        constraintWeight = dict()
        constraintsOK = True
         
        # If no constraints, bail out
        if rd.numConstraints==0:
            return (constraintsOK, constraintWeight, rd)
         
        # Loop round each constraint and check it is being met
        for (k,rc) in enumerate(constraints):
            if rc.factorWeights is None:
                continue
            for idx in rd.factorIndices:
                if factorNames[idx] in constraintWeight:
                    constraintWeight[factorNames[idx]] += rc.factorWeights[idx]
                else:
                    constraintWeight[factorNames[idx]] = rc.factorWeights[idx]
            constraintValue = numpy.inner(rc.factorWeights, factorReturns)
            diff = abs(constraintValue - rc.sumToValue)
            self.log.info('%s constraint, target: %f, actual: %f, diff: %f',
                    rc.factorType.name, rc.sumToValue, constraintValue, diff)
            # If constraint not satisfied, double its weight
            if diff > maxTolerance:
                self.log.warning('%s constraint not satisfied. Re-running regression',
                        rc.factorType.name)
                pos = rd.regressionRHS.shape[1] - rd.numConstraints + k - 1
                rd.regressionRHS[:,pos] = 2.0 * rd.regressionRHS[:,pos]
                rd.regressionLHS[pos] = 2.0 * rd.regressionLHS[pos]
                rd.weights_ESTU[pos] = 2.0 * rd.weights_ESTU[pos]
                constraintsOK = False
        return (constraintsOK, constraintWeight, rd)

    def calculateVIF(self, exposureMatrix, weights, regIdx, factorIdxMap_):
        """Calculate Variance Inflation Factors for each style factor regressed on other style factors.

        Parameters
        ----------
        exposureMatrix : Numpy array of exposures for all factors (styles,industries, etc.,). Dimension is factors * assets
        weights : Array of weights corresponding to entries in regIdx.
        regIdx : Indices corresponding to assets to be used in VIF regressions.
        factorIdxMap_ : Map used to identify style factor indices in exposure matrix.

        Returns
        -------
        VIF : Dictionary mapping style factor names to its VIF.
        """
        self.log.debug('Calculating VIF for the style factors.')
        styleIdxMap = {}
        for fType in factorIdxMap_:
            if fType.name=='Style':
                styleIdxMap = factorIdxMap_[fType]
        if len(styleIdxMap)==0:
            logging.info('Could not find style factors to calculate VIF.')
            return
        exposureMatrix = numpy.take(exposureMatrix, list(regIdx), axis=1)
        # Get the indices for styles that are constant.
        const_style_indices = numpy.where(numpy.var(exposureMatrix,1)==0)[0]
        style_indices = [x for x in styleIdxMap.values() if x not in const_style_indices]
        VIF = {}
        for style, idx in styleIdxMap.items():
            if not idx in style_indices:
                continue
            rd = Utilities.Struct()
            rd.regressionLHS = numpy.take(exposureMatrix, [idx], axis=0)
            rd.regressionRHS = numpy.take(exposureMatrix, [x for x in style_indices if x!=idx], axis=0)
            #  Adding a constant term, so that it kind of represents the sum of exposures over all industries.
            rd.regressionRHS = numpy.transpose(self.sm.add_constant(numpy.transpose(rd.regressionRHS), prepend=True))
            # Run the regression
            model = self.sm.WLS(numpy.array(rd.regressionLHS),numpy.transpose(numpy.array(rd.regressionRHS)),weights)
            result = model.fit()
            VIF[style] = 1.0/(1-result.rsquared)
        return VIF
    
    def calc_Factor_Specific_Returns(self, rm, estimationUniverseIdx, excessReturns, 
                         exposureMatrix, factorNames, weights, expM, constraints=None,
                         force=False, robustRegression=True, removeNTDsFromRegression=True):
        
        # Initialise
        exposureMatrix = ma.filled(exposureMatrix, 0.0)
        factorReturns = numpy.zeros(exposureMatrix.shape[0])
        self.removeNTDsFromRegression = removeNTDsFromRegression

        # Report on returns extremities
        tmp = ma.take(excessReturns, estimationUniverseIdx, axis=0)
        logging.info('Excess returns bounds (estu): [%.3f, %.3f]' % \
                (ma.min(tmp, axis=None), ma.max(tmp, axis=None)))
        logging.info('Excess returns bounds (all): [%.3f, %.3f]' % \
                (ma.min(excessReturns, axis=None),
                    ma.max(excessReturns, axis=None)))

        # Get estimation universe
        nonMissingReturnsIdx = numpy.flatnonzero(ma.getmaskarray(
            ma.take(excessReturns, estimationUniverseIdx, axis=0))==0)
        indices_ESTU = numpy.take(estimationUniverseIdx, nonMissingReturnsIdx, axis=0)
        self.log.info('%d assets in actual estimation universe (%d original)',
                len(indices_ESTU), len(estimationUniverseIdx))
        rd = Utilities.Struct()
        self.__initStatsmodels()

        rd.dummyRetWeights = []
        rd.subIssues = None
        if expM is not None:
            rd.subIssues = numpy.take(expM.getAssets(), indices_ESTU, axis=0)
            rd.subIssues = [sid.getSubIDString() for sid in rd.subIssues]
         
        if robustRegression:
            # Initialise regressor matrix
            rd.excessReturns_ESTU = ma.take(excessReturns, indices_ESTU, axis=0)
            rd.regressorMatrix = numpy.take(exposureMatrix, indices_ESTU, axis=1)
            if not self.parameters.useWeightedRLM():
                rd.weights_ESTU = 1000.0 * numpy.ones((len(nonMissingReturnsIdx)), float)
            else:
                rd.weights_ESTU = ma.take(weights, nonMissingReturnsIdx, axis=0)
        
            (rd, factorReturns) = self.processFactorIndices(
                    rd, factorNames, factorReturns, constraints, expM,
                        removeNTDsFromRegression=self.removeNTDsFromRegression)

            # Output regressor matrix (useful for debugging)
            if rm.debuggingReporting and (rd.subIssues is not None): 
                outfile = open('tmp/regressorMatrix%d.csv' % len(rd.factorIndices), 'w')
                outfile.write('ID,Return,Weight')
                for i in rd.factorIndices:
                    outfile.write(',%s' % factorNames[i].replace(',',''))
                outfile.write('\n')
                for j in range(rd.regressorMatrix.shape[1]):
                    if (j > len(rd.subIssues)-1):
                        outfile.write('ID_%d,%f,%f' % (j+1, rd.excessReturns_ESTU[j], rd.weights_ESTU[j]))
                    else:
                        outfile.write('%s,%f,%f' % (rd.subIssues[j], rd.excessReturns_ESTU[j], rd.weights_ESTU[j]))
                    for m in range(rd.regressorMatrix.shape[0]):
                        outfile.write(',%f' % rd.regressorMatrix[m,j])
                    outfile.write('\n')
                outfile.close()
        
            # Set up regression data
            rd.regressionRHS = rd.regressorMatrix * numpy.sqrt(rd.weights_ESTU)
            rd.regressionLHS = rd.excessReturns_ESTU * ma.sqrt(rd.weights_ESTU)

            # Report on final state of the regressor matrix
            xwx = numpy.dot(rd.regressionRHS, numpy.transpose(rd.regressionRHS))
            (eigval, eigvec) = linalg.eigh(xwx)
            conditionNumber = max(eigval) / min(eigval)
            self.log.info('Weighted regressor (%i by %i) has condition number %f',
                    xwx.shape[0], xwx.shape[1], conditionNumber)
        
            # Perform the robust regression loop
            maxRuns = 10
            for i in range(maxRuns):
                if force:
                    try:
                        rlmResult = self.__robustRegressWithStatsmodels(rd)
                    except:
                        self.log.warning('Regression failed: defective inputs')
                        rlmDownWeights = numpy.ones((len(rd.excessReturns_ESTU)), float)
                        constraintWeight = dict()
                        rlmResult = None
                        allZeroFR = True
                        break
                else:
                    rlmResult = self.__robustRegressWithStatsmodels(rd)
                
                # Pull out final regression weights
                rlmDownWeights = rlmResult.getWeights()
                kappa = rlmResult.getKappa()
                self.log.info('Kappa=%f', kappa)

                # Grab factor returns from R
                regressionResults = rlmResult.getCoefficients()
                allZeroFR = True
                if len(factorReturns) == 1:
                    factorReturns[0] = regressionResults[0]
                    allZeroFR = False
                else:
                    for (i,j) in enumerate(rd.factorIndices):
                        factorReturns[j] = regressionResults[i]
                        if abs(factorReturns[j]) > 0.0:
                            allZeroFR = False
            
                # Check that linear constraints (if any) are being satisfied
                (constraintsOK, constraintWeight, rd) = self.check_regression_constraints(
                            rd, factorNames, factorReturns, constraints)
                if constraintsOK:
                    break
        
            # Report on effect of robust regression
            tmp = rd.excessReturns_ESTU * numpy.sqrt(rlmDownWeights)
            downWeightFactor = numpy.sum(abs(tmp)) / numpy.sum(abs(rd.excessReturns_ESTU))
            logging.info('Downweight factor: %6.4f', downWeightFactor)
            logging.info('Excess returns bounds after downweighting (estu): [%.3f, %.3f]' % \
                    (ma.min(tmp, axis=None), ma.max(tmp, axis=None)))

            if rm.debuggingReporting and (rd.subIssues is not None):
                outfile = open('tmp/downWeights%d.csv' % len(rd.factorIndices), 'w')
                outfile.write('ID,Weight\n')
                for (sid, wt) in zip(rd.subIssues, rlmDownWeights):
                    outfile.write('%s,%f\n' % (sid, wt))
                outfile.close()
        else:
            rlmDownWeights = numpy.ones((len(indices_ESTU)), float)

        # Perform additional weighted OLS step if required
        if not self.parameters.useWeightedRLM() or not robustRegression:
            logging.info('Performing final weighted OLS')
             
            # Reset things first
            rd.dummyRetWeights = []
            rd.excessReturns_ESTU = ma.take(excessReturns, indices_ESTU, axis=0)
            rd.weights_ESTU = ma.take(weights, nonMissingReturnsIdx, axis=0)
            rd.regressorMatrix = numpy.take(exposureMatrix, indices_ESTU, axis=1)
            factorReturns = numpy.zeros(exposureMatrix.shape[0])
             
            # Redo all the thin/empty industry and constraint stuff
            (rd, factorReturns) = self.processFactorIndices(
                    rd, factorNames, factorReturns, constraints, expM,
                    removeNTDsFromRegression=self.removeNTDsFromRegression)
             
            # Combine rlm downweights with regular weights
            robustWeights = numpy.ones((len(rd.weights_ESTU)), float)
            off = len(nonMissingReturnsIdx)
            robustWeights[0:off] = rlmDownWeights[0:off]
            rd.weights_ESTU = robustWeights * rd.weights_ESTU
             
            # Output regressor matrix (useful for debugging)
            if rm.debuggingReporting and (rd.subIssues is not None):
                outfile = open('tmp/regressorMatrix-nonrob-%d.csv' % len(rd.factorIndices), 'w')
                outfile.write('ID,Return,Weight')
                for i in rd.factorIndices:
                    outfile.write(',%s' % factorNames[i].replace(',',''))
                outfile.write('\n')
                for j in range(rd.regressorMatrix.shape[1]):
                    if (j > len(rd.subIssues)-1):
                        outfile.write('ID_%d,%f,%f' % (j+1, rd.excessReturns_ESTU[j], rd.weights_ESTU[j]))
                    else:
                        outfile.write('%s,%f,%f' % (rd.subIssues[j], rd.excessReturns_ESTU[j], rd.weights_ESTU[j]))
                    for m in range(rd.regressorMatrix.shape[0]):
                        outfile.write(',%f' % rd.regressorMatrix[m,j])
                    outfile.write('\n')
                outfile.close()

            # Now do the regression
            maxRuns = 10
            rd.regressionRHS = rd.regressorMatrix * numpy.sqrt(rd.weights_ESTU)
            rd.regressionLHS = rd.excessReturns_ESTU * ma.sqrt(rd.weights_ESTU)

            # Report on final state of the regressor matrix
            xwx = numpy.dot(rd.regressionRHS, numpy.transpose(rd.regressionRHS))
            (eigval, eigvec) = linalg.eigh(xwx)
            conditionNumber = max(eigval) / min(eigval)
            self.log.info('Weighted regressor (%i by %i) has condition number %f',
                    xwx.shape[0], xwx.shape[1], conditionNumber)

            for i in range(maxRuns):
                lmResult = self.__regressWithStatsmodels(rd)
                 
                # Grab factor returns from R
                regressionResults = lmResult.getCoefficients()
                allZeroFR = True
                if len(factorReturns) == 1:
                    factorReturns[0] = regressionResults[0]
                    allZeroFR = False
                else:
                    for (i,j) in enumerate(rd.factorIndices):
                        factorReturns[j] = regressionResults[i]
                        if abs(factorReturns[j]) > 0.0:
                            allZeroFR = False

                # Check that linear constraints (if any) are being satisfied
                (constraintsOK, constraintWeight, rd) = self.check_regression_constraints(
                        rd, factorNames, factorReturns, constraints)
                if constraintsOK:
                    break
            robustWeights = rd.weights_ESTU
        else:
            robustWeights = rlmDownWeights * rd.weights_ESTU
            lmResult = rlmResult

        fmpDict = dict()
        constrComponent = None
        if rd.subIssues is not None:
            fmpDict, constrComponent = Utilities.buildFMPMapping(
                    rd, robustWeights, len(indices_ESTU), factorNames, nonMissingReturnsIdx)

        # Get rid of some residue from the dummies
        if len(rd.thinFactorIndices) + rd.numConstraints > 0:
            rd.weights_ESTU = rd.weights_ESTU[:-len(rd.thinFactorIndices) - rd.numConstraints]
            robustWeights = robustWeights[:-len(rd.thinFactorIndices) - rd.numConstraints]
         
        # Compute residuals
        specificReturns = excessReturns - numpy.dot(
            numpy.transpose(exposureMatrix), factorReturns)
        spec_return_ESTU = ma.take(specificReturns, indices_ESTU, axis=0)
        cap_wts = rd.weights_ESTU * rd.weights_ESTU
        spec_return = ma.average(spec_return_ESTU, weights=cap_wts)
        self.log.info('ESTU specific return (cap wt): %12.8f', spec_return)
        
        # Calculate regression statistics
        regressANOVA = RegressionANOVA(excessReturns, specificReturns,
                exposureMatrix.shape[0], indices_ESTU, robustWeights)
        regressionStatistics = numpy.zeros((exposureMatrix.shape[0], 3))
        white = self.parameters.getWhiteStdErrors()
        # Temporary, replace 999 with 0 to enable.
        if white or rd.numConstraints > 999:
            if rd.numConstraints > 0:
                expMatrix = rd.regressorMatrix[:,:-len(rd.thinFactorIndices)-rd.numConstraints]
                constrMatrix = numpy.transpose(rd.regressorMatrix[:,-rd.numConstraints:])
            else:
                expMatrix = rd.regressorMatrix 
                constrMatrix = None
            regressionStats = regressANOVA.calc_regression_statistics(
                    factorReturns, numpy.transpose(expMatrix), constrMatrix, white=white)
        elif not allZeroFR:
            regressionStats = lmResult.getRegressionStats() 
            # Compute p-values for t-stats since rlm doesn't provide them
            pvals = [(1.0 - stats.t.cdf(abs(t), rd.regressorMatrix.shape[1] - \
                    len(rd.factorIndices))) for t in regressionStats[:,1]]
            regressionStats = numpy.concatenate([regressionStats, 
                    numpy.transpose(numpy.array(pvals)[numpy.newaxis])], axis=1)
        else:
            logging.warning('All factor returns are zero - better investigate')
            regressionStats = regressionStatistics
        
        # Put it all together (to account for factors omitted from regression)
        for (i,j) in enumerate(rd.factorIndices):
            regressionStatistics[j,:] = regressionStats[i,:]

        regressANOVA.store_regression_statistics(regressionStatistics[:,0],
                regressionStatistics[:,1], regressionStatistics[:,2])
        
        return (numpy.transpose(exposureMatrix), factorReturns,
                specificReturns, regressANOVA, constraintWeight, fmpDict, rd.subIssues,
                constrComponent)

class PrincipalComponents:
    def __init__(self, numFactors):
        self.log = logging.getLogger('ReturnCalculator.PrincipalComponents')
        self.numFactors = numFactors

    def calc_ExposuresAndReturns(self, returns, date, estu=None):
        """Principal Components Analysis
        """
        self.log.debug('calc_ExposuresAndReturns: begin')
        # If no ESTU specified, use all assets
        if estu is None:
            estu = list(range(returns.data.shape[0]))
        if returns.data.shape[1] >= self.numFactors:
            if len(estu) >= returns.data.shape[1]:
                self.log.warning('ESTU assets (%d) > Returns history (%d), unreliable estimates',
                              len(estu), returns.data.shape[1])
            returnsESTU = ma.take(returns.data.filled(0.0), estu, axis=0)

            # Watch out, right eigenvectors returned in rows, not columns
            (u, d, v) = linalg.svd(returnsESTU, full_matrices=False)
            d = d**2 / returnsESTU.shape[0]
            order = numpy.argsort(-d)
            if not len(estu) >= returns.data.shape[1]:
                u = numpy.take(u, order, axis=0)
            u = u[:,0:self.numFactors]
            d = numpy.take(d, order, axis=0)
            v = numpy.take(v, order, axis=0)[0:self.numFactors,:]

            # Calculate exposures for ESTU assets
            exposuresESTU = numpy.dot(u, numpy.diag(d[0:self.numFactors]**0.5))

            # Calculate factor returns via OLS regression
            # Weighted LS not used here, see Johnson & Wichern (sec 9.5)
            (factorReturns, specificReturnsESTU)  = Utilities.\
                    ordinaryLeastSquares(returnsESTU, exposuresESTU)

#            # Alternatively, compute factor returns via weighted LS
#            specificVarsESTU = numpy.zeros(returnsESTU.shape[0])
#            for n in range(returnsESTU.shape[0]):
#                specificVarsESTU[n] = numpy.inner(returnsESTU[n], returnsESTU[n]) - \
#                        numpy.inner(u[n], u[n]) * d[0:self.numFactors] 
#            factorReturns = Utilities.generalizedLeastSquares(
#                returnsESTU, exposuresESTU, numpy.diag(1.0 / specificVarsESTU))
#            specificReturnsESTU = returnsESTU - numpy.dot(exposuresESTU, factorReturns)

            # Back out exposures for non-ESTU assets 
            (exposureMatrix, specificReturns) = Utilities.ordinaryLeastSquares(
                    numpy.transpose(returns.data.filled(0.0)), numpy.transpose(factorReturns))

            # Merge with exposures and specific returns from ESTU assets
            # The put() logic a little funky to avoid issues with non-contiguous arrays
            for i in range(len(estu)):
                numpy.put(exposureMatrix, numpy.arange(i, exposureMatrix.shape[0] \
                    * exposureMatrix.shape[1], exposureMatrix.shape[1]), exposuresESTU[i])
                numpy.put(specificReturns, numpy.arange(i, specificReturns.shape[0] \
                    * specificReturns.shape[1], specificReturns.shape[1]), specificReturnsESTU[i])
            exposureMatrix = numpy.transpose(exposureMatrix)
            specificReturns = ma.masked_where(ma.getmaskarray(returns.data), 
                                                numpy.transpose(specificReturns))

            # Finally, calculate specific returns
#            specificReturns = returns.data - numpy.dot(exposureMatrix, factorReturns)

            # Calculate regression statistics
            regressANOVA = RegressionANOVA(returns.data[:,-1], 
                            specificReturns[:,-1], self.numFactors, estu)
        else:
            raise LookupError('PCA requires returns history (%d) > ESTU assets (%d) > factors (%d)' % (returns.data.shape[1], len(estu), self.numFactors))
        self.log.debug('calc_ExposuresAndReturns: end')
        return (exposureMatrix, factorReturns, specificReturns, regressANOVA)

class AsymptoticPrincipalComponents:
    def __init__(self, numFactors):
        self.log = logging.getLogger('ReturnCalculator.AsymptoticPrincipalComponents')
        self.numFactors = numFactors

    def calc_ExposuresAndReturns(self, returns, date, estu=None, mapReturns=None):
        """Asymptotic Principal Components Analysis (Connor, Korajczyk 1986)
        Uses APCA to determine factor returns, and exposures for non-ESTU
        are backed out via time-series regression.
        Input is a TimeSeriesMatrix of returns and ESTU indices.
        Returns exposures, factor returns, and specific returns as arrays
        """
        self.log.debug('calc_ExposuresAndReturns: begin')

        # If no ESTU specified, use all assets
        if estu is None:
            estu = list(range(returns.data.shape[0]))

        if returns.data.shape[1] >= self.numFactors:
            if returns.data.shape[1] >= len(estu):
                self.log.warning('Returns history (%d) > ESTU assets (%d), unreliable estimates',
                              returns.data.shape[1], len(estu))
            returnsESTU = ma.take(returns.data.filled(0.0), estu, axis=0)
            # Using SVD, more efficient than eigendecomp on huge matrix
            try:
                (d, v) = linalg.svd(returnsESTU, full_matrices=False)[1:]
                d = d**2 / returnsESTU.shape[0]
            except:
                logging.warning('SVD routine failed... computing eigendecomposition instead')
                tr = numpy.dot(numpy.transpose(returnsESTU), returnsESTU)
                (d, v) = linalg.eigh(tr)
                d = d / returnsESTU.shape[0]
                v = numpy.transpose(v)
            order = numpy.argsort(-d)
            d = numpy.take(d, order, axis=0)
            v = numpy.take(v, order, axis=0)[0:self.numFactors,:]

            # Take factor returns to be right singular vectors
            factorReturns = v 

            # Back out exposures (time-series regression yields same result)
            if mapReturns is None:
                mapReturns = returns
            exposureMatrix = numpy.dot(
                        mapReturns.data.filled(0.0), numpy.transpose(v))

            # Finally, calculate specific returns
            specificReturns = mapReturns.data - \
                    numpy.dot(exposureMatrix, factorReturns)

            # Calculate regression statistics
            regressANOVA = RegressionANOVA(mapReturns.data[:,-1], 
                            specificReturns[:,-1], self.numFactors, estu)
            
            # Not sure if this really applies to PCA of TxT matrix...
#            for n in range(1,self.numFactors+1):
#                prc = numpy.sum(d[0:n], axis=0) / numpy.sum(d, axis=0) * 100
#                self.log.info('%d factors explains %f%% of variance, eigenvalue=%f',
#                            n, prc, d[n-1])
            prc = numpy.sum(d[0:self.numFactors], axis=0) \
                  / numpy.sum(d, axis=0) * 100
            self.log.info('%d factors explains %f%% of variance',
                           self.numFactors, prc)
        else:
            raise LookupError('Asymptotic PCA requires ESTU assets (%d) > returns history (%d) > factors (%d)' % (len(estu), returns.data.shape[1], self.numFactors))
        self.log.debug('calc_ExposuresAndReturns: end')
        return (exposureMatrix, factorReturns, specificReturns, regressANOVA)

class AsymptoticPrincipalComponents2:
    def __init__(self, numFactors, min_sigma=0.02, max_sigma=1.0):
        self.log = logging.getLogger('ReturnCalculator.AsymptoticPrincipalComponents2')
        self.numFactors = numFactors
        self.min_sv = min_sigma**2.0
        self.max_sv = max_sigma**2.0

    def calc_ExposuresAndReturns(self, returns, date, estu=None):
        loader = AsymptoticPrincipalComponents(self.numFactors)

        # First iteration: APCA to obtain specific returns
        (tmpExpM, tmpFacRet, tmpSpecRet, tmpRegANOVA) = loader.\
                            calc_ExposuresAndReturns(returns, date, estu)

        # Apply residual variance adjustment to returns
        adjReturns = Matrices.TimeSeriesMatrix(returns.assets, returns.dates)
        adjReturns.data = ma.array(
                self.residualVarianceAdjustment(returns.data, tmpSpecRet))

        # Second iteration: APCA on adjusted returns
        (exposureMatrix, factorReturns, specificReturns, regressANOVA) = loader.\
                            calc_ExposuresAndReturns(adjReturns, date, estu, returns)
        return (exposureMatrix, factorReturns, specificReturns, regressANOVA)
        
    def residualVarianceAdjustment(self, returns, specificReturns):
        """Residual variance adjustment (Connor, Korajczyk 1988)
        Converges quicker than regular PCA in determining the 
        realized factor returns. 
        Inputs are asset and specific returns as arrays.
        Returns scaled/adjusted returns as an array.
        """
        self.log.debug('residualVarianceAdjustment: begin')
        specificVars = [ma.inner(specificReturns[n], specificReturns[n])
                        for n in range(returns.shape[0])]
        specificVars = ma.masked_where(specificVars==0.0, specificVars)
        # Prevent massively over-weighting some small specific-risk assets
        # and under-weighting of large specific-risk ones
        n = min(200, int(len(specificVars)*0.01))
        rank = ma.argsort(specificVars)
        min_threshold = max(self.min_sv, specificVars[rank[n]])
        max_threshold = min(self.max_sv, specificVars[rank[-n]])
        self.log.info('Residual std dev bounds: (%.4f, %.4f)',
                        min_threshold**0.5, max_threshold**0.5)
        specificVars = ma.where(specificVars < min_threshold,
                                min_threshold, specificVars)
#        specificVars = ma.where(specificVars > max_threshold,
#                                max_threshold, specificVars)
        invSpecificRisk = 1.0 / (specificVars ** 0.5)
        adjReturns = ma.transpose(invSpecificRisk * ma.transpose(returns))
        self.log.debug('residualVarianceAdjustment: end')
        return adjReturns

#####################################
# Run an example when called as main.
if __name__ == '__main__':
    import logging.config
    import optparse
    import MarketDB.MarketDB as MarketDB
    import ModelDB

    usage = "usage: %prog [options]"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addModelAndDefaultCommandLine(cmdlineParser)
    cmdlineParser.add_option("-d", "--date", action="store", type="string",
                             default="2003-10-03", dest="modelDate",
                             help="use exposure matrix for specified date")
    (options, args) = cmdlineParser.parse_args()
    
    modelClass = Utilities.processModelAndDefaultCommandLine(
        options, cmdlineParser)
    modelDB = ModelDB.ModelDB(sid=options.modelDBSID, user=options.modelDBUser,
                              passwd=options.modelDBPasswd)
    modelDB.totalReturnCache = None
    marketDB = MarketDB.MarketDB(sid=options.marketDBSID, user=options.marketDBUser, passwd=options.marketDBPasswd)
    riskModel = modelClass(modelDB, marketDB)
    
    print(options.modelDate)
    modelDate = Utilities.parseISODate(options.modelDate)
    (factorReturns, specificReturns, exposureMatrix, regressStats, adjRsquared) = \
                    riskModel.generateFactorSpecificReturns(
        modelDB, marketDB, modelDate)
#    riskModel.insertFactorReturns(modelDate, factorReturns, modelDB)
#    riskModel.insertSpecificReturns(modelDate, specificReturns,
#                                  exposureMatrix.getAssets(), modelDB)
    marketDB.finalize()
    modelDB.finalize()


# vim: set softtabstop=4 shiftwidth=4:
