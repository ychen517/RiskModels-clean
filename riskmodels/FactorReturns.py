import logging
import numpy.ma as ma
import numpy.linalg as linalg
import pandas
import numpy
import scipy.stats as stats
import datetime
try:
    import statsmodels.api as sm
except ImportError:
    import scikits.statsmodels.api as sm
except:
    import scikits.statsmodels as sm
from riskmodels import AssetProcessor
from riskmodels import Matrices
from riskmodels.Matrices import ExposureMatrix
from riskmodels import Utilities
from riskmodels import Outliers

def calc_Weights(rmg, modelDB, marketDB, modelDate, universe,
        baseCurrencyID=None, clip=True, freefloat_flag=False):

    # Load average mcaps
    nAssets = len(universe)
    avgMktCaps = AssetProcessor.robustLoadMCaps(modelDate, universe, 
            baseCurrencyID, modelDB, marketDB, freefloat_flag)[0]

    weights = ma.sqrt(avgMktCaps)

    # Trim at 95th percentile
    C = min(100, int(round(nAssets*0.05)))
    sortindex = ma.argsort(weights)
    ma.put(weights, sortindex[nAssets-C:nAssets], weights[sortindex[nAssets-C]])
    return weights

class RegressionANOVA:
    def __init__(self, data, resid, nvars, estu, weights=None, deMean=False):
        self.log = logging.getLogger('FactorReturns.RegressionANOVA')
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

    def calc_regression_statistics(self, estimates, exposureMatrix, constraintMatrix=None, white=False):
        """Computes regression statistics manually if they 
        have not been loaded already. eg. for a statistical model.
        """
        e = ma.take(self.resid_, self.estU_, axis=0)
        df = len(self.data_) - self.nvars_
        assert(exposureMatrix.shape[0]==len(self.weights_))
        self.regressStats_[:,0] = Utilities.computeRegressionStdError(
                    e, exposureMatrix, self.weights_, constraintMatrix, white=white)
        self.regressStats_[:,1] = estimates / self.regressStats_[:,0]
        self.regressStats_[:,2] = 1.0 - numpy.array(stats.t.cdf(abs(self.regressStats_[:,1]), df))
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

class RobustRegression2017:
    # New class for factor regression 
    # To be tidied and streamlined over time
    def __init__(self, params):
        self.log = logging.getLogger('FactorReturns.RobustRegression2017')
        self.allParameters = params
        self.badRetDownWeight = 1.0e-8

    def insertDummyAssets(self, thinFacPar, regData, regPar):
        """Examines the factors in positions (idxToCheck) for thinness.
        Inserts dummy assets into the regressor matrix where necessary,
        returns a new copy of the regressor matrix, returns, and weights,
        as well as an array of factor returns and lists of thin and empty
        factor positions.
        """
        thinFactorIdx = []
        emptyFactorIdx = []
        dummyThreshold = thinFacPar.dummyThreshold
        thinWeightMultiplier = regPar.thinWeightMultiplier
        factorNameIdxMap = dict(zip(regPar.regFactorNames, list(range(len(regPar.regFactorNames)))))

        # Loop round factors to test for thinness
        for (factorType, factorName) in zip(thinFacPar.factorTypes, thinFacPar.factorNames):

            # Get thin factor info for particular factor
            fIdx = factorNameIdxMap[factorName]
            assetsIdx = numpy.flatnonzero(regData.regressorMatrix[fIdx,:])
            if factorName not in thinFacPar.dummyReturns:
                raise Exception('Dummy returns have not been specified for %s' % factorName)
            dummyRet = thinFacPar.dummyReturns[factorName]
            dummyRetWeights = thinFacPar.dummyRetWeights[factorName]

            # Compute regression weight of factor
            factorWeights = ma.take(regData.weights_ESTU, assetsIdx, axis=0) * \
                    ma.take(regData.regressorMatrix[fIdx], assetsIdx, axis=0)
            totalWgt = numpy.sum(ma.filled(abs(factorWeights), 0.0), axis=0)

            if totalWgt <= 0.0:
                # Empty factor, keep track of these and omit from reg later
                dummyRet = 0.0
                self.log.warning('Empty factor: %s, ret %f', factorName, dummyRet)
                regData.factorReturns[fIdx] = dummyRet
                emptyFactorIdx.append(fIdx)
                continue

            # Compute Herfindahl effective number
            wgt = factorWeights / totalWgt
            score = 1.0 / ma.inner(wgt, wgt)

            # If this factor is thin...
            if score < dummyThreshold:

                if factorType == ExposureMatrix.StyleFactor:
                    # Thin style factor - treat as empty
                    self.log.warning('Thin style factor: %s', factorName)
                    regData.factorReturns[fIdx] = 0.0
                    emptyFactorIdx.append(fIdx)
                    continue

                # Compute weight and exposure of dummy asset
                if thinWeightMultiplier == 'OE':
                    partial1 = score**4.0
                    partial2 = dummyThreshold**4.0
                    dummyWgt = (dummyThreshold - 1.0) * (partial1 - partial2) / (1.0 - partial2)
                    dummyWgt *= totalWgt / score
                else:
                    dummyWgt = (dummyThreshold - score) * totalWgt / score
                dummyExp = numpy.zeros((regData.regressorMatrix.shape[0], 1))
                dummyExp[fIdx] = 1.0
                nTrueAssets = len(regPar.reg_estu)

                # Assign style factor exposures to dummy, if applicable
                for nzName in thinFacPar.nonzeroExposures:
                    nzIdx = factorNameIdxMap[nzName]
                    if Utilities.is_binary_data(regData.regressorMatrix[nzIdx, :nTrueAssets+1]):
                        val = ma.median(regData.regressorMatrix[nzIdx, :nTrueAssets+1], axis=0)
                    else:
                        mywgts = ma.take(regData.weights_ESTU, assetsIdx, axis=0) / ma.sum(ma.take(regData.weights_ESTU, assetsIdx, axis=0))
                        myexps = ma.take(regData.regressorMatrix[nzIdx], assetsIdx, axis=0)
                        val = ma.inner(mywgts, myexps)
                    dummyExp[nzIdx] = val

                # Append dummy exposures, weight, and return to estu
                thinFactorIdx.append(fIdx)
                regData.regressorMatrix = numpy.concatenate([regData.regressorMatrix, dummyExp], axis=1)
                regData.excessReturns_ESTU = ma.concatenate([regData.excessReturns_ESTU, ma.array([dummyRet])], axis=0)
                regData.weights_ESTU = ma.concatenate([regData.weights_ESTU, ma.array([dummyWgt])], axis=0)

                # Save these for FMP calculation
                if not hasattr(regData, 'dummyRetWeights'):
                    regData.dummyRetWeights = numpy.array(dummyRetWeights, float)[numpy.newaxis,:]
                else:
                    regData.dummyRetWeights = ma.concatenate(
                            [regData.dummyRetWeights, numpy.array(dummyRetWeights, float)[numpy.newaxis,:]], axis=0)

                # Report on dummy asset data
                if regData.sidList is not None:
                    regData.sidList.append('%s-Dummy' % factorName.replace(',','').replace(' ',''))
                self.log.info('Thin factor: %s (Multiplier: %s, N: %.2f/%d, Dummy wgt: %.1f%%, ret: %.1f%%)',
                        factorName, thinWeightMultiplier, score, len(assetsIdx),
                        100.0*dummyWgt/(dummyWgt+totalWgt), 100.0*dummyRet)

        regData.thinFactorIdx = thinFactorIdx
        regData.emptyFactorIdx = emptyFactorIdx
        return

    def processFactorIndices(self, rm, assetData, regData, regPar, thinFacPar, expMatrixCls):
        """Routine to sort out the thin and empty from the big and fat
        Also sets up constraints for singular regressions
        """
        # Initialise set of allowed factor indices
        okFactorIdx = list(range(len(regData.factorReturns)))

        # Remove factors with all exposures equal to 1 except for the market intercept
        interceptFactorIdx = []
        if expMatrixCls is not None:
            officialInterceptName = [f for f in regPar.regFactorNames if \
                    expMatrixCls.getFactorType(f)==ExposureMatrix.InterceptFactor]
            interceptFactorIdx = [idx for idx in okFactorIdx if \
                    (sum(regData.regressorMatrix[idx,:])==len(regData.excessReturns_ESTU))]
        if (len(okFactorIdx) > 1) and (len(interceptFactorIdx) > 1):
            okFactorIdx = [idx for idx in okFactorIdx if ((idx not in interceptFactorIdx) or \
                    (regPar.regFactorNames[idx] in officialInterceptName))]
            for idx in interceptFactorIdx:
                if regPar.regFactorNames[idx] not in officialInterceptName:
                    self.log.warning('Factor %s has become an intercept and will be removed',
                                    regPar.regFactorNames[idx])

        # Invoke thin factor correction mechanism if required
        if regPar.fixThinFactors and thinFacPar is not None:
            self.insertDummyAssets(thinFacPar, regData, regPar)
            okFactorIdx = [i for i in okFactorIdx if i not in regData.emptyFactorIdx]
        else:
            regData.thinFactorIdx = []
            regData.emptyFactorIdx = []
        if len(regData.thinFactorIdx)==0:
            self.log.info('No thin factors!')

        # Remove any remaining empty factors
        for idx in okFactorIdx:
            assetsIdx = numpy.flatnonzero(regData.regressorMatrix[idx,:])
            if len(assetsIdx) == 0:
                regData.factorReturns[idx] = 0.0
                self.log.warning('Empty factor: %s', regPar.regFactorNames[idx])
                regData.emptyFactorIdx.append(idx)
        okFactorIdx = [i for i in okFactorIdx if i not in regData.emptyFactorIdx]

        # Add constraint(s) to deal with deliberately singular regression
        numConstraints = 0
        numDummies = len(regData.thinFactorIdx)
        regData.constraintWeights = dict()
        for constrType in regPar.constraintTypes:
            regData.constraintWeights[constrType] = None
            if (constrType is ExposureMatrix.IndustryFactor) or \
               (constrType is ExposureMatrix.CountryFactor):
                conFactorIdx = [idx for idx in okFactorIdx if \
                    regPar.regFactorNames[idx] in expMatrixCls.getFactorNames(constrType)]
            else:
                conFactorIdx = []
            if len(conFactorIdx) > 0:
                self.createConstraintAsset(constrType, conFactorIdx, rm, assetData, regData, regPar, numDummies)
                regData.sidList.append('%s-Constraint' % constrType)
                numDummies += 1
                numConstraints += 1

        self.log.info('%i Factors, %i intercept, %i thin, %i empty, %i used',
                len(regData.factorReturns), len(interceptFactorIdx), len(regData.thinFactorIdx),
                len(regData.emptyFactorIdx), len(okFactorIdx))

        # And return everything
        regData.okFactorIdx = okFactorIdx
        regData.numConstraints = numConstraints
        return

    def createConstraintAsset(self, constrType, conFactorIdx, rm, assetData, regData, regPar, numDummies):
        """Creates dummy asset by appending an entry to the regressor
        matrix and weight and return arrays.
        """
        self.log.info('Applying constraint to %s factors', constrType.name)

        # Add new column to regressor matrix
        expCol = numpy.zeros((regData.regressorMatrix.shape[0]))[numpy.newaxis]
        regData.regressorMatrix = ma.concatenate([regData.regressorMatrix, ma.transpose(expCol)], axis=1)

        # Load market caps for constraint
        if regPar.useRealMCapsForConstraints:
            logging.info('Using actual 30-day mcaps for constraint')
            cWeight = AssetProcessor.robustLoadMCaps(self.date, regPar.reg_estu, 
                    rm.numeraire.currency_id, self.modelDB, self.marketDB)[0]

            # Downweight some countries if required
            assetIdxMap_ESTU = dict([(j,i) for (i,j) in enumerate(regPar.reg_estu)])
            for r in [r for r in rm.rmg if r.downWeight < 1.0]:
                if r.rmg_id in assetData.rmgAssetMap:
                    if regPar.regWeight == 'rootCap':
                        if rm.legacySAWeight:
                            for sid in assetData.rmgAssetMap[r.rmg_id].intersection(regPar.reg_estu):
                                cWeight[assetIdxMap_ESTU[sid]] *= (r.downWeight * r.downWeight)
                        else:
                            for sid in assetData.rmgAssetMap[r.rmg_id].intersection(regPar.reg_estu):
                                cWeight[assetIdxMap_ESTU[sid]] *= r.downWeight
                    else:
                        # Not consistent yet
                        for sid in assetData.rmgAssetMap[r.rmg_id].intersection(regPar.reg_estu):
                            cWeight[assetIdxMap_ESTU[sid]] *= r.downWeight
                elif r in assetData.rmgAssetMap:
                    if regPar.regWeight == 'rootCap':
                        if rm.legacySAWeight:
                            for sid in assetData.rmgAssetMap[r].intersection(regPar.reg_estu):
                                cWeight[assetIdxMap_ESTU[sid]] *= (r.downWeight * r.downWeight)
                        else:
                            for sid in assetData.rmgAssetMap[r].intersection(regPar.reg_estu):
                                cWeight[assetIdxMap_ESTU[sid]] *= r.downWeight
                    else:
                        # Not consistent yet
                        for sid in assetData.rmgAssetMap[r].intersection(regPar.reg_estu):
                            cWeight[assetIdxMap_ESTU[sid]] *= r.downWeight

            # Downweight anything on the basket-case list
            if hasattr(assetData, 'mktCapDownWeight'):
                for (rmg, dnWt) in assetData.mktCapDownWeight.items():
                    if rmg.rmg_id in assetData.tradingRmgAssetMap:
                        if regPar.regWeight == 'rootCap':
                            for sid in set(assetData.tradingRmgAssetMap[rmg.rmg_id]).intersection(regPar.reg_estu):
                                cWeight[assetIdxMap_ESTU[sid]] *= dnWt
                        else:
                            for sid in set(assetData.tradingRmgAssetMap[rmg.rmg_id]).intersection(regPar.reg_estu):
                                cWeight[assetIdxMap_ESTU[sid]] *= dnWt

            # This code will downweight bad assets for the constraint
            if hasattr(rm, 'badRetList'):
                if regPar.regWeight == 'rootCap':
                    for sid in set(rm.badRetList).intersection(set(regPar.reg_estu)):
                        cWeight[assetIdxMap_ESTU[sid]] *= (self.badRetDownWeight * self.badRetDownWeight)
                else:
                    for sid in set(rm.badRetList).intersection(set(regPar.reg_estu)):
                        cWeight[assetIdxMap_ESTU[sid]] *= self.badRetDownWeight
        else:
            # Note - need to add extra logic for different regression weights here
            cWeight = regData.weights_ESTU[:len(regData.weights_ESTU)-numDummies]**2

        # Compute weights on factors
        factorTotalCap = 0.0
        for idx in conFactorIdx:
            indices = numpy.flatnonzero(regData.regressorMatrix[idx,:-(numDummies+1)])
            self.log.debug('Computing weight for %s', regPar.regFactorNames[idx])
            if len(indices) > 0:
                # NOTE: adding cap weights here but weight vector is root-cap
                factorMCap = ma.sum(ma.take(cWeight, indices, axis=0), axis=None)
                regData.regressorMatrix[idx,-1] = factorMCap
                factorTotalCap += factorMCap

        # Scale so exposures sum to one
        if factorTotalCap > 0.0:
            regData.regressorMatrix[:,-1] /= factorTotalCap
        regData.constraintWeights[constrType] = regData.regressorMatrix[:,-1]

        # Assign return - add other options here as needed
        ret = 0.0

        # Update return and weight arrays
        factorTotalCap = factorTotalCap / float(len(regPar.reg_estu))
        regData.excessReturns_ESTU = ma.concatenate([regData.excessReturns_ESTU, ma.array([ret])], axis=0)
        regData.weights_ESTU = ma.concatenate([regData.weights_ESTU, ma.array([factorTotalCap])], axis=0)

        return

    def check_regression_constraints(self, regData, regPar, maxTolerance=1.0e-4):
        """Performs check on whether regression constraints are being met
        increases weight if not
        """
        constraintsOK = True
        constaintSumTo = 0.0
        constrFactorNameMap = dict()
         
        # If no constraints, bail out
        if regData.numConstraints == 0:
            return constraintsOK, constrFactorNameMap
         
        # Loop round each constraint and check it is being met
        for (k, constrType) in enumerate(regPar.constraintTypes):
            if regData.constraintWeights[constrType] is None:
                continue

            # Save total weights to a mapping for later
            for idx in regData.okFactorIdx:
                if regPar.regFactorNames[idx] in constrFactorNameMap:
                    constrFactorNameMap[regPar.regFactorNames[idx]] += regData.constraintWeights[constrType][idx]
                else:
                    constrFactorNameMap[regPar.regFactorNames[idx]] = regData.constraintWeights[constrType][idx]

            # Compute constraint weighted factor return contribution
            constraintValue = numpy.inner(regData.constraintWeights[constrType], regData.factorReturns)
            diff = abs(constraintValue - constaintSumTo)
            self.log.info('%s constraint, target: %f, actual: %.12f, diff: %.12f',
                    constrType.name, constaintSumTo, constraintValue, diff)

            # If constraint not satisfied, double its weight
            if diff > maxTolerance:
                self.log.warning('%s constraint not satisfied. Re-running regression', constrType.name)
                pos = regData.regressionRHS.shape[1] - regData.numConstraints + k - 1
                regData.regressionRHS[:,pos] = 2.0 * regData.regressionRHS[:,pos]
                regData.regressionLHS[pos] = 2.0 * regData.regressionLHS[pos]
                regData.weights_ESTU[pos] = 2.0 * regData.weights_ESTU[pos]
                constraintsOK = False

        return constraintsOK, constrFactorNameMap

    def calculateVIF(self, exposureMatrix, weights, regIdx, factorIdxMap_):
        """Calculate Variance Inflation Factors for each style factor regressed on other style factors.

        Parameters
        ----------
        exposureMatrix : Numpy array of exposures for all factors (styles,industries, etc.,).
        Dimension is factors * assets
        weights : Array of weights corresponding to entries in regIdx.
        regIdx : Indices corresponding to assets to be used in VIF regressions.
        factorIdxMap_ : Map used to identify style factor indices in exposure matrix.

        Returns
        -------
        VIF : Dictionary mapping style factor names to its VIF.
        """
        self.log.debug('Calculating VIF for the style factors.')
        styleIdxMap = {}
        industryIdxMap = {}
        countryIdxMap = {}
        marketIdxMap = {}
        for fType in factorIdxMap_:
            if fType.name== ExposureMatrix.StyleFactor.name:
                styleIdxMap = factorIdxMap_[fType]
            elif fType.name== ExposureMatrix.IndustryFactor.name:
                industryIdxMap = factorIdxMap_[fType]
            elif fType.name== ExposureMatrix.CountryFactor.name:
                countryIdxMap = factorIdxMap_[fType]
            elif fType.name== ExposureMatrix.InterceptFactor.name:
                marketIdxMap = factorIdxMap_[fType]
        if len(styleIdxMap)==0:
            logging.info('Could not find style factors to calculate VIF.')
            return
        exposureMatrix = numpy.take(exposureMatrix, list(regIdx), axis=1)

        # Get the indices for styles that are constant.
        const_indices = numpy.where(numpy.var(exposureMatrix,1)==0)[0]
        style_indices = [x for x in styleIdxMap.values() if x not in const_indices]
        industry_indices = [x for x in industryIdxMap.values() if x not in const_indices]
        country_indices = [x for x in countryIdxMap.values() if x not in const_indices]
        market_index = marketIdxMap.values()

        VIF = {}

        for style, idx in styleIdxMap.items():
            if not idx in style_indices:
                continue
            rd = Utilities.Struct()
            rd.regressionLHS = numpy.take(exposureMatrix, [idx], axis=0).flatten()
            rd.regressionRHS = numpy.take(exposureMatrix, [x for x in list(style_indices)+list(industry_indices)+list(country_indices)+list(market_index) if x!=idx], axis=0)

            # Run the regression
            model = self.sm.WLS(numpy.array(rd.regressionLHS),numpy.transpose(numpy.array(rd.regressionRHS)),weights)
            result = model.fit()
            VIF[style] = 1.0/(1-result.rsquared)

        return VIF
    
    def computeRegressionStatistics(self, rm, rlmResult, regData, regPar, regMatrix, excessReturns, specificReturns):
        """Compute and store the regression stats from the robust regression
        """

        # Initialise
        regressANOVA = RegressionANOVA(excessReturns, specificReturns,
                regMatrix.shape[0], regPar.regEstuIdx, regData.robustWeights)
        regressionStatistics = numpy.zeros((regMatrix.shape[0], 3))
        numZeroFRs = numpy.flatnonzero(ma.getmaskarray(ma.masked_where(
                        regData.factorReturns==0.0, regData.factorReturns)))

        # Our own computation for stdError - to be tested someday
        if regPar.whiteStdErrors:
            if regData.numConstraints > 0:
                expMatrix = regData.regressorMatrix[:,:-len(regData.thinFactorIdx)-regData.numConstraints]
                constrMatrix = numpy.transpose(regData.regressorMatrix[:,-regData.numConstraints:])
            else:
                expMatrix = regData.regressorMatrix
                constrMatrix = None
            regStats = regressANOVA.calc_regression_statistics(
                    regData.factorReturns, numpy.transpose(expMatrix), constrMatrix, white=True)

        # Otherwise use the output from the regression itself (the default for now)
        elif len(numZeroFRs) < len(regData.factorReturns):
            params = numpy.array(rlmResult.params)
            bse = rlmResult.bse
            regStats = numpy.zeros((params.shape[0],2))
            regStats[:,0] = bse
            bse = ma.masked_where(abs(bse)<1.0e-12, bse)
            regStats[:,1] = ma.filled(params/bse, 0.0)

            # Compute p-values for t-stats since rlm doesn't provide them
            pvals = [(1.0 - stats.t.cdf(abs(t), regData.regressorMatrix.shape[1] - \
                    len(regData.okFactorIdx))) for t in regStats[:,1]]
            regStats = numpy.concatenate([regStats,
                    numpy.transpose(numpy.array(pvals)[numpy.newaxis])], axis=1)
        else:
            regStats = regressionStatistics

        # Put it all together (to account for factors omitted from regression)
        for (idx, jdx) in enumerate(regData.okFactorIdx):
            regressionStatistics[jdx,:] = regStats[idx,:]

        regressANOVA.store_regression_statistics(regressionStatistics[:,0],
                regressionStatistics[:,1], regressionStatistics[:,2])
        return regressANOVA

    def calc_Factor_Specific_Returns(
            self, rm, regPar, thinFacPar, assetData, excessReturns, regMatrix, expMatrixCls, fmpRun):
        
        # Report on returns extremities
        if rm.debuggingReporting:
            tmp = ma.take(excessReturns, regPar.regEstuIdx, axis=0)
            logging.debug('Excess returns bounds (estu): [%.3f, %.3f]' % (ma.min(tmp, axis=None), ma.max(tmp, axis=None)))
            logging.debug('Excess returns bounds (all): [%.3f, %.3f]' % \
                    (ma.min(excessReturns, axis=None), ma.max(excessReturns, axis=None)))

        # Set up important regression dict
        regData = Utilities.Struct()
        regData.excessReturns_ESTU = ma.filled(ma.take(excessReturns, regPar.regEstuIdx, axis=0), 0.0)
        regData.weights_ESTU = numpy.array(ma.filled(regPar.reg_weights, 0.0), copy=True)
        regData.regressorMatrix = ma.filled(ma.take(regMatrix, regPar.regEstuIdx, axis=1), 0.0)
        regData.factorReturns = numpy.zeros(regMatrix.shape[0])
        regData.sidList = [sid if isinstance(sid, str) else sid.getSubIDString() for sid in regPar.reg_estu]

        # Determine robustness of regression
        if regPar.kappa == None:
            opms = dict()
            opms['nBounds'] = [25.0, 25.0]
            outlierClass = Outliers.Outliers(opms)
            regData.excessReturns_ESTU = outlierClass.twodMAD(regData.excessReturns_ESTU, suppressOutput=True)
         
        # Some models may require certain exposures to be set to zero
        if rm.zeroExposureFactorNames != []:
            regData.regressorMatrix = self.zeroCertainFactorExp(expMatrixCls, assetData, regPar.reg_estu, regData, regPar, rm)

        # Process thin industries and constraints
        self.processFactorIndices(rm, assetData, regData, regPar, thinFacPar, expMatrixCls)
        regData.regressorMatrix = ma.take(regData.regressorMatrix, regData.okFactorIdx, axis=0)

        # Output regressor matrix (useful for debugging)
        if rm.debuggingReporting and regData.sidList is not None: 
            outfile = open('tmp/regressorMatrix-%d.csv' % regPar.iReg, 'w')
            outfile.write(',Return,Weight')
            for idx in regData.okFactorIdx:
                outfile.write(',%s' % regPar.regFactorNames[idx].replace(',',''))
            outfile.write('\n')
            for j in range(regData.regressorMatrix.shape[1]):
                outfile.write('%s,%.16f,%.16f' % (regData.sidList[j], regData.excessReturns_ESTU[j], regData.weights_ESTU[j]))
                for m in range(regData.regressorMatrix.shape[0]):
                    outfile.write(',%.16f' % regData.regressorMatrix[m,j])
                outfile.write('\n')
            outfile.close()
        
        # Set up regression data
        regData.regressionRHS = regData.regressorMatrix * numpy.sqrt(regData.weights_ESTU)
        regData.regressionLHS = regData.excessReturns_ESTU * numpy.sqrt(regData.weights_ESTU)

        # Report on final state of the regressor matrix
        xwx = numpy.dot(regData.regressionRHS, numpy.transpose(regData.regressionRHS))
        (eigval, eigvec) = linalg.eigh(xwx)
        conditionNumber = max(eigval) / min(eigval)
        self.log.info('Weighted regressor (%i by %i) has condition number %.12f',
                xwx.shape[0], xwx.shape[1], conditionNumber)

        # Perform the robust regression loop
        maxRuns = 10
        for i_run in range(maxRuns):
            if regData.numConstraints > 0:
                try:
                    C = pandas.DataFrame(regData.regressorMatrix[:,-regData.numConstraints:],
                            index=[regPar.regFactorNames[idx] for idx in regData.okFactorIdx],
                            columns=regData.sidList[-regData.numConstraints:]).T
                    B = pandas.DataFrame(regData.regressorMatrix[:,:-regData.numConstraints],
                            index=[regPar.regFactorNames[idx] for idx in regData.okFactorIdx],
                            columns=regData.sidList[:-regData.numConstraints]).T
                    w = pandas.Series(regData.weights_ESTU[:-regData.numConstraints],
                            index=regData.sidList[:-regData.numConstraints])
                    r = pandas.Series(regData.excessReturns_ESTU[:-regData.numConstraints],
                           index=regData.sidList[:-regData.numConstraints])
                    model = Utilities.ConstrainedLinearModel(r, B, C=C, weights=w, huberT=regPar.kappa)
                    rlmResult = Utilities.Struct()
                    rlmResult.weights = numpy.ones(regData.regressionLHS.shape[0])
                    rlmResult.weights[:-regData.numConstraints] = model.robustweights.values
                    rlmResult.params = model.params.values
                    rlmResult.bse = model.bse.values
                except:
                    logging.error('Null space method failed for solving constrained regression')
                    logging.error('Switching to dummy asset approach')
                    if regPar.kappa == None:
                        logging.info('Running OLS regression')
                        model = sm.OLS(numpy.array(regData.regressionLHS), \
                                numpy.transpose(numpy.array(regData.regressionRHS)))
                    else:
                        model = sm.RLM(numpy.array(regData.regressionLHS), \
                                numpy.transpose(numpy.array(regData.regressionRHS)),
                                M = sm.robust.norms.HuberT(t=regPar.kappa))
                    rlmResult = model.fit(maxiter=500)
            else:
                if regPar.kappa == None:
                    logging.info('Running OLS regression')
                    model = sm.OLS(numpy.array(regData.regressionLHS), \
                            numpy.transpose(numpy.array(regData.regressionRHS)))
                else:
                    model = sm.RLM(numpy.array(regData.regressionLHS), \
                            numpy.transpose(numpy.array(regData.regressionRHS)),
                            M = sm.robust.norms.HuberT(t=regPar.kappa))
                rlmResult = model.fit(maxiter=500)
            
            # Pull out final regression weights
            if regPar.kappa == None:
                rlmDownWeights = numpy.ones((len(regData.regressionLHS)), float)
            else:
                rlmDownWeights = numpy.array(rlmResult.weights)

            # Grab factor returns from regression output
            regressionResults = numpy.array(rlmResult.params)
            allZeroFR = True
            if len(regData.factorReturns) == 1:
                regData.factorReturns[0] = regressionResults[0]
                allZeroFR = False
            else:
                for (ix,jx) in enumerate(regData.okFactorIdx):
                    regData.factorReturns[jx] = regressionResults[ix]
                    if abs(regData.factorReturns[jx]) > 0.0:
                        allZeroFR = False
            
            # Check that linear constraints (if any) are being satisfied
            constraintsOK, constrFactorNameMap = self.check_regression_constraints(regData, regPar)
            if constraintsOK:
                break
        
        # Report on effect of robust regression
        tmp = regData.excessReturns_ESTU * numpy.sqrt(rlmDownWeights)
        downWeightFactor = numpy.sum(abs(tmp)) / numpy.sum(abs(regData.excessReturns_ESTU))
        logging.info('Regression %s Downweight factor: %6.4f', regPar.iReg, downWeightFactor)
        logging.debug('Excess returns bounds after downweighting (estu): [%.3f, %.3f]' % \
                (ma.min(tmp, axis=None), ma.max(tmp, axis=None)))

        # Output regression weights (useful for debugging)
        if rm.debuggingReporting and (regData.sidList is not None):
            outfile = open('tmp/regression-%s-Weights-%s.csv' % (regPar.iReg, self.date), 'w')
            outfile.write('ID,InitialWeight,RobustWeight,\n')
            for (sid, wt, rWt) in zip(regData.sidList, regData.weights_ESTU, rlmDownWeights):
                outfile.write('%s,%f,%f,\n' % (sid, wt, rWt))
            outfile.close()

        # Compute FMP matrix
        regData.robustWeights = rlmDownWeights * regData.weights_ESTU
        fmpDict = dict()
        fmpConstrComponent = None
        if fmpRun:
            fmpDict, fmpConstrComponent = self.buildFMPMapping(regPar, regData)

        # Get rid of some residue from the dummies
        if len(regData.thinFactorIdx) + regData.numConstraints > 0:
            regData.weights_ESTU = regData.weights_ESTU[:-len(regData.thinFactorIdx) - regData.numConstraints]
            regData.robustWeights = regData.robustWeights[:-len(regData.thinFactorIdx) - regData.numConstraints]
         
        # Compute residuals
        specificReturns = excessReturns - numpy.dot(numpy.transpose(ma.filled(regMatrix, 0.0)), regData.factorReturns)
        spec_return_ESTU = ma.take(specificReturns, regPar.regEstuIdx, axis=0)
        spec_return = ma.average(spec_return_ESTU, weights=(regData.robustWeights))
        self.log.info('Regression %s final ESTU specific return (reg wt): %.8f', regPar.iReg, spec_return)
        
        # Calculate regression statistics
        regressANOVA = self.computeRegressionStatistics(
                rm, rlmResult, regData, regPar, ma.filled(regMatrix, 0.0), excessReturns, specificReturns)
        
        # Set up return structure
        regOutput = Utilities.Struct()
        regOutput.factorReturns = regData.factorReturns
        regOutput.specificReturns = specificReturns
        regOutput.regressANOVA = regressANOVA
        regOutput.constraintWeight = constrFactorNameMap
        regOutput.rlmDownWeights = rlmDownWeights
        regOutput.sidList = regData.sidList
        regOutput.fmpDict = fmpDict
        regOutput.fmpConstrComponent = fmpConstrComponent

        return regOutput

    def run_factor_regressions(self, _riskModel, dummy1, date, excessReturns, expMatrixCls,
                        dummy2, assetData, excludeFactorIdx, modelDB, marketDB, robustWeightList=None,
                        applyRT=False, fmpRun=False, rmgHolidayList=None):
        """Performs the dirty work of shuffling around arrays and
        such that is required for the nested regressions.  Also
        computes 'aggregate' regression statistics for the model, such
        as r-square, t-stats, etc.  Returns a tuple containing a
        map of factor names to factor return values, a specific returns
        array, a map of factor names to regression statistics, and
        a ReturnCalculator.RegressionANOVA object housing the results.
        """
        self.log.debug('run_factor_regressions: begin')
        self.log.info('**************** Running NEW regression code ****************')

        # Set up some data items to be used later
        # Factor info
        factorReturnsMap = dict()
        allFactorNames = expMatrixCls.getFactorNames()
        self.factorNameIdxMap = dict(zip(allFactorNames, list(range(len(allFactorNames)))))
        self.excludeFactorIdx = excludeFactorIdx
        # Returns info
        regressionReturns = ma.array(excessReturns)
        assetExposureMatrix = ma.array(expMatrixCls.getMatrix())
        # Regression stats info
        robustWeightMap = dict()
        ANOVA_data = list()
        regressStatsMap = dict()
        # FMP Info
        fmpMap = dict()
        ccMap = dict()
        ccXMap = dict()
        # DB Info
        self.modelDB = modelDB
        self.marketDB = marketDB
        self.date = date
        self.VIF = None

        assetIdxMap = dict(zip(assetData.universe, range(len(assetData.universe))))
        estimationUniverseIdx = [assetIdxMap[sid] for sid in assetData.estimationUniverse]
        # Report on returns
        if type(assetData.marketCaps) is pandas.Series:
            rtCap_ESTU = assetData.marketCaps[assetData.estimationUniverse].values
        else:
            rtCap_ESTU = ma.take(assetData.marketCaps, estimationUniverseIdx, axis=0)
        returns_ESTU = ma.take(excessReturns, estimationUniverseIdx, axis=0)
        skewness = stats.skew(returns_ESTU * ma.sqrt(rtCap_ESTU), axis=0)
        logging.debug('Skewness of ESTU returns: %f', skewness)

        # Identify possible dummy style factors
        dummyStyles = self.checkForStyleDummies(expMatrixCls, assetExposureMatrix, _riskModel)
        if hasattr(assetData, 'assetIdxMap'):
            assetIdxMap = assetData.assetIdxMap
        else:
            assetIdxMap = dict(zip(assetData.universe, range(len(assetData.universe))))

        # Loop round however many regressions are required
        regKeys = sorted(self.allParameters.keys())
        for iReg in regKeys:
             
            # Get specific regression paramters
            regPar = self.allParameters[iReg]
            self.log.info('Beginning nested regression, loop %d, ESTU: %s', iReg+1, regPar.estuName)
            self.log.debug('Factors in loop: %s', ', '.join([f.name for f in regPar.regressionList]))

            # Determine which factors will go into this regression loop
            regPar.regFactorsIdx, regPar.regFactorNames = self.processFactorsForRegression(
                                                regPar, expMatrixCls, _riskModel)
            regMatrix = ma.take(assetExposureMatrix, regPar.regFactorsIdx, axis=0)

            # Get estimation universe for this loop
            regPar.reg_estu = list(_riskModel.estuMap[regPar.estuName].assets)
            logging.debug('Using %d assets from %s estimation universe', len(regPar.reg_estu), regPar.estuName)
            regPar.regEstuIdx = [assetIdxMap[sid] for sid in regPar.reg_estu]
            if len(regPar.regEstuIdx) < 10:
                 continue

            # Get the regression weights
            regPar.reg_weights = self.getRegressionWeights(regPar, assetData, _riskModel)

            # If ESTU assets have no returns, warn and skip
            checkReturns = ma.take(excessReturns, regPar.regEstuIdx, axis=0)
            checkReturns = ma.masked_where(abs(checkReturns) < 1e-12, checkReturns)
            badReturns = numpy.sum(ma.getmaskarray(checkReturns))
            if (badReturns >= 0.99 * len(regPar.regEstuIdx)) or (len(regPar.regFactorsIdx)  < 1):
                self.log.warning('No returns for nested regression loop %d ESTU, skipping', iReg + 1)
                specificReturns = ma.array(regressionReturns, copy=True)
                for fName in regPar.regFactorNames:
                    factorReturnsMap[fName] = 0.0
                    regressStatsMap[fName] = Matrices.allMasked(4)
                continue

            # Deal with thin factors (industry, country, dummy styles)
            thinFacPar = self.processThinFactorsForRegression(
                    _riskModel, regPar, expMatrixCls, regressionReturns, assetData, applyRT, rmgHolidayList=rmgHolidayList)

            # Finally, run the regression
            regPar.iReg = iReg + 1
            regOut = self.calc_Factor_Specific_Returns(
                            _riskModel, regPar, thinFacPar, assetData, regressionReturns, regMatrix, expMatrixCls, fmpRun)

            if ExposureMatrix.StyleFactor in regPar.regressionList and regPar.computeVIF:
                # Assuming that 1st round of regression is  done on styles, industries, market etc.,
                # and second round regression is done on Domestic China etc.,
                self.sm = sm
                factorsInRegression = {fType: expMatrixCls.factorIdxMap_[fType] for fType in regPar.regressionList}
                self.VIF = self.calculateVIF(ma.filled(expMatrixCls.getMatrix(), 0.0),
                                regOut.regressANOVA.weights_, regOut.regressANOVA.estU_, factorsInRegression)

            # Pass residuals to next regression loop
            regressionReturns = regOut.specificReturns
            specificReturns = ma.array(regOut.specificReturns, copy=True)

            # Keep record of factor returns and regression statistics
            if regPar.estuName == 'main' or regPar.name.startswith('main'):
                ANOVA_data.append(regOut.regressANOVA)
            for jdx in range(len(regPar.regFactorsIdx)):

                # Save factor returns
                fName = regPar.regFactorNames[jdx]
                factorReturnsMap[fName] = regOut.factorReturns[jdx]

                # Save FMP info
                if fName in regOut.fmpDict:
                    fmpMap[fName] = dict(zip(regOut.sidList, regOut.fmpDict[fName].tolist()))
                if regOut.fmpConstrComponent is not None:
                    if fName in regOut.fmpConstrComponent.ccDict:
                        ccMap[fName] = regOut.fmpConstrComponent.ccDict[fName]
                        ccXMap[fName] = regOut.fmpConstrComponent.ccXDict[fName]

                # Save regression stats
                values = Matrices.allMasked(4)
                values[:3] = regOut.regressANOVA.regressStats_[jdx,:]
                if fName in regOut.constraintWeight:
                    values[3] = regOut.constraintWeight[fName]
                regressStatsMap[fName] = values

            # Add robust weights to running total
            tmpWeights = ma.masked_where(regOut.rlmDownWeights==1.0, regOut.rlmDownWeights)
            tmpWeightsIdx = numpy.flatnonzero(ma.getmaskarray(tmpWeights)==0)
            tmpWeights = ma.take(tmpWeights, tmpWeightsIdx, axis=0)
            tmpSIDs = numpy.take(regOut.sidList, tmpWeightsIdx, axis=0)
            robustWeightMap[iReg] = dict(zip(tmpSIDs, tmpWeights))

        # Regression ANOVA: take average of all regressions using MAIN ESTU
        if len(ANOVA_data) > 0:
            self.log.info('%d regression loops all use same ESTU, computing ANOVA', len(ANOVA_data))
            numFactors = sum([n.nvars_ for n in ANOVA_data])
            regWeights = numpy.average(numpy.array([n.weights_ for n in ANOVA_data]), axis=0)
            anova = RegressionANOVA(excessReturns, specificReturns, numFactors, ANOVA_data[0].estU_, regWeights)
        else:
            self.log.info('No valid regressions, hence no regression statistics')
            anova = None

        self.log.debug('run_factor_regressions: end')
        retVal = Utilities.Struct()
        retVal.factorReturnsMap = factorReturnsMap
        retVal.specificReturns = specificReturns
        retVal.regStatsMap = regressStatsMap
        retVal.anova = anova
        retVal.robustWeightMap = robustWeightMap
        retVal.fmpMap = fmpMap
        retVal.ccMap = ccMap
        retVal.ccXMap = ccXMap

        return retVal   

    def zeroCertainFactorExp(self, expM, assetData, sidList, regData, regPar, rm): 
        """Set some exposures to zero for certain combinations of factors
        """
        assetIdxMap = dict(zip(sidList, list(range(len(sidList)))))
        factorIdxMap = dict(zip(regPar.regFactorNames, list(range(len(regPar.regFactorNames)))))
        if hasattr(assetData, 'assetTypeDict'):
            assetTypeDict = assetData.assetTypeDict
        else:
            assetTypeDict = assetData.getAssetType()
        # Loop round the factors
        for (assetType, factorName) in zip(rm.zeroExposureAssetTypes, rm.zeroExposureFactorNames):

            # First pick out the factor to be treated by type
            fIdx = factorIdxMap.get(factorName, None)

            if fIdx is not None:
                # Now set the relevant exposure values to zero by asset type
                zeroExpAssetIdx = [assetIdxMap[sid] for sid in sidList if assetTypeDict.get(sid, None) == assetType]
                factorExp = regData.regressorMatrix[fIdx,:]
                ma.put(factorExp, zeroExpAssetIdx, 0.0)
                regData.regressorMatrix[fIdx,:] = factorExp
                self.log.info('Zeroing %d %s exposures for %s factor', len(zeroExpAssetIdx), assetType, factorName)

        return regData.regressorMatrix

    def checkForStyleDummies(self, expM, data, rm):
        """Test for dummy style factors in exposure matrix
        """
        allFactorNames = expM.getFactorNames()
        dummyStyles = set()
        for idx in expM.getFactorIndices(ExposureMatrix.StyleFactor):
            if Utilities.is_binary_data(data[idx,:]):
                vals = list(set(ma.filled(data[idx,:], 0.0)))
                dummyStyles.add(allFactorNames[idx])
                logging.debug('Dummy style factor: %s, values: %s', allFactorNames[idx], vals)
        return dummyStyles

    def processFactorsForRegression(self, regPar, expM, rm):
        """Get the definitive list of factors that will go into the regression
        and report on exclusions
        """
        regFactorNames = list()

        # Look for permanently or temporarily excluded factors
        excludeFactorNames = set([f.name for f in regPar.excludeFactors \
                                if isinstance(f, EquityModel.ModelFactor)])
        if len(excludeFactorNames) > 0:
            self.log.info('Excluding these factors: %s',
                    ', '.join([f for f in excludeFactorNames]))
        droppedFactorNames = [f for f in self.factorNameIdxMap.keys() \
                    if self.factorNameIdxMap[f] in self.excludeFactorIdx]

        # Determine which factors will go into this regression loop
        for obj in regPar.regressionList:
            try:
                fList = expM.getFactorNames(obj)
            except:
                fList = [obj.name]
            if len(fList) == 0:
                logging.info('No factors in model belonging to %s', obj.name)
            for f in fList:
                if f in droppedFactorNames:
                    logging.info('Excluding %s from this regression', f)
                elif f not in self.factorNameIdxMap:
                    logging.warning('Factor %s not in available list', f)
                elif f in excludeFactorNames:
                    logging.warning('Factor %s in exclusion list', f)
                else:
                    regFactorNames.append(f)
        regFactorsIdx = [self.factorNameIdxMap[f] for f in regFactorNames]
        return regFactorsIdx, regFactorNames

    def getRegressionWeights(self, regPar, assetData, rm):
        """Get the correct weights for the particular regression
        """
        # Get the regression weights
        if regPar.estuName in rm.estuMap \
                and hasattr(rm.estuMap[regPar.estuName], 'weights'):
            # Regression weights pre-computed
            reg_weights = rm.estuMap[regPar.estuName].weights
            wgtDict = dict(zip(rm.estuMap[regPar.estuName].assets, reg_weights))
            reg_weights = [wgtDict[sid] for sid in regPar.reg_estu]
            logging.debug('Using pre-defined regression weights, kappa=%s', regPar.kappa)
            return reg_weights
        
        # Compute basic sqrt-cap weights
        if rm.useFreeFloatRegWeight:
            reg_weights = calc_Weights(
                    rm.rmg, self.modelDB, self.marketDB, self.date, regPar.reg_estu,
                    rm.numeraire.currency_id, clip=regPar.clipWeights, freefloat_flag=True)
        else:
            reg_weights = calc_Weights(
                        rm.rmg, self.modelDB, self.marketDB, self.date, regPar.reg_estu,
                        rm.numeraire.currency_id, clip=regPar.clipWeights)

        if regPar.regWeight == 'cap':
            # Change to cap-weights if required
            reg_weights = reg_weights * reg_weights
            logging.debug('Cap-weighted regression, clipped Wts=%s, kappa=%s',
                        regPar.clipWeights, regPar.kappa)
        elif regPar.regWeight == 'equal':
            # Or equal-weights
            reg_weights = 1000.0 * numpy.ones((len(reg_weights)), float)
            logging.debug('Using unweighted RLM regression, kappa=%s', regPar.kappa)

        elif regPar.regWeight == 'invSpecificVariance':
            # Or invSpecVar weights
            rmi = rm.getRiskModelInstance(self.date, self.modelDB)
            hasSR = True
            if not rmi.has_risks:
                olderRMIs = self.modelDB.getRiskModelInstances(rm.rms_id)
                rmiDates = [rmi.date for rmi in olderRMIs if rmi.has_risks and \
                        rmi.date < self.date and rmi.date > (self.date-datetime.timedelta(30))]
                if len(rmiDates) < 1:
                    hasSR = False
                    logging.warning('No specific risk for %s or earlier, using sqrt(cap) instead', str(self.date))
                    logging.debug('Root-cap-weighted regression, clipped Wts=%s, kappa=%s',
                            regPar.clipWeights, regPar.kappa)
                else:
                    rmiDates.sort()
                    rmi = [rmi for rmi in olderRMIs if rmi.date==rmiDates[-1]][0]
                    logging.warning('No specific risk for %s, using data from %s instead', str(self.date), str(rmi.date))

            if hasSR:
                # Load model specific risks
                reg_weights = self.modelDB.getSpecificRisks(rmi)
                reg_weights = pandas.Series(reg_weights)
                reg_weights = reg_weights[reg_weights.index.isin(regPar.reg_estu)]
                reg_weights = reg_weights.reindex(regPar.reg_estu)
                reg_weights[reg_weights == 0.0] = numpy.nan
                reg_weights = ma.array(reg_weights.values, mask = pandas.isnull(reg_weights.values))

                # Form regression weights as inverse of specific variance
                reg_weights = reg_weights * reg_weights
                reg_weights = Utilities.clip_extrema(reg_weights, 0.025)
                reg_weights = 1.0 / reg_weights
                logging.debug('Using inverse of specific variance, kappa=%s', regPar.kappa)
        else:
            logging.debug('Root-cap-weighted regression, clipped Wts=%s, kappa=%s',
                        regPar.clipWeights, regPar.kappa)

        # Downweight bad assets - i.e. missing/zero returns
        assetIdxMap_ESTU = dict([(j,i) for (i,j) in enumerate(regPar.reg_estu)])
        # Using this code will downweight bad assets and related dummy assets in the regression
        if hasattr(rm, 'badRetList'):
            dnWtSet = set(rm.badRetList).intersection(regPar.reg_estu)
            logging.debug('Downweighting %d bad assets', len(dnWtSet))
            for sid in dnWtSet:
                reg_weights[assetIdxMap_ESTU[sid]] *= self.badRetDownWeight

        # Downweight some countries if required
        for r in [r for r in rm.rmg if r.downWeight < 1.0]:
            if r.rmg_id in assetData.rmgAssetMap:
                if regPar.regWeight == 'rootCap':
                    if rm.legacySAWeight:
                        for sid in assetData.rmgAssetMap[r.rmg_id].intersection(regPar.reg_estu):
                            reg_weights[assetIdxMap_ESTU[sid]] *= r.downWeight
                    else:
                        for sid in assetData.rmgAssetMap[r.rmg_id].intersection(regPar.reg_estu):
                            reg_weights[assetIdxMap_ESTU[sid]] *= numpy.sqrt(r.downWeight)
                else:
                    for sid in assetData.rmgAssetMap[r.rmg_id].intersection(regPar.reg_estu):
                        reg_weights[assetIdxMap_ESTU[sid]] *= r.downWeight
            elif r in assetData.rmgAssetMap:
                if regPar.regWeight == 'rootCap':
                    if rm.legacySAWeight:
                        for sid in assetData.rmgAssetMap[r].intersection(regPar.reg_estu):
                            reg_weights[assetIdxMap_ESTU[sid]] *= r.downWeight
                    else:
                        for sid in assetData.rmgAssetMap[r].intersection(regPar.reg_estu):
                            reg_weights[assetIdxMap_ESTU[sid]] *= numpy.sqrt(r.downWeight)
                else:
                    for sid in assetData.rmgAssetMap[r].intersection(regPar.reg_estu):
                        reg_weights[assetIdxMap_ESTU[sid]] *= r.downWeight

        # Downweight anything on the basket-case list
        if hasattr(assetData, 'mktCapDownWeight'):
            for (rmg, dnWt) in assetData.mktCapDownWeight.items():
                if rmg.rmg_id in assetData.tradingRmgAssetMap:
                    if regPar.regWeight == 'rootCap':
                        for sid in set(assetData.tradingRmgAssetMap[rmg.rmg_id]).intersection(regPar.reg_estu):
                            reg_weights[assetIdxMap_ESTU[sid]] *= numpy.sqrt(dnWt)
                    else:
                        for sid in set(data.tradingRmgAssetMap[rmg.rmg_id]).intersection(regPar.reg_estu):
                            reg_weights[assetIdxMap_ESTU[sid]] *= dnWt

        return reg_weights

    def processThinFactorsForRegression(self, rm, regPar, expMatrixCls, regressionReturns,
            assetData, applyRT, rmgHolidayList=None, factorTypes=None, nonZeroNames=None):
        """Process list of factors and decide which are to be tested for thinness
        Also compute the various dummy returns required
        """
        thinFacPar = None
        thinTestNames = list()
        thinTestTypes = list()

        # Get the types of factor to test for thinness
        if regPar.fixThinFactors:
            if factorTypes is None:
                for fName in regPar.regFactorNames:
                    fType = expMatrixCls.getFactorType(fName)
                    if fType in (ExposureMatrix.IndustryFactor,
                                ExposureMatrix.CountryFactor,
                                ExposureMatrix.StyleFactor):
                        thinTestNames.append(fName)
                        thinTestTypes.append(fType)
            else:
                thinTestNames = list(regPar.regFactorNames)
                thinTestTypes = list(factorTypes)

        if len(thinTestNames) > 0:

            # Set up thin factor parameter structure
            thinFacPar = Utilities.Struct()
            thinFacPar.dummyReturnType = regPar.dummyType
            thinFacPar.dummyThreshold = regPar.dummyThreshold
            thinFacPar.factorNames = thinTestNames
            thinFacPar.factorTypes = thinTestTypes
            thinFacPar.applyRT = applyRT
            thinFacPar.nonzeroExposures = []
            thinFacPar.dummyReturns = dict()
            thinFacPar.dummyRetWeights = dict()

            # Allow dummy assets to have nonzero style and intercept exposures
            if nonZeroNames is None:
                for f in regPar.regressionList:
                    try:
                        fType = expMatrixCls.getFactorType(f.name)
                    except:
                        fType = f
                    if fType in (ExposureMatrix.StyleFactor, ExposureMatrix.InterceptFactor):
                        thinFacPar.nonzeroExposures = [f for f in regPar.regFactorNames if \
                                expMatrixCls.getFactorType(f) in (ExposureMatrix.StyleFactor,
                                                              ExposureMatrix.InterceptFactor)]
                        break
            else:
                thinFacPar.nonzeroExposures = list(nonZeroNames)

            # Get relevant returns
            returns_ESTU = ma.take(regressionReturns, regPar.regEstuIdx, axis=0)

            opms = dict()
            opms['nBounds'] = [15.0, 15.0]
            outlierClass = Outliers.Outliers(opms)
            returns_ESTU = outlierClass.twodMAD(returns_ESTU, axis=0, suppressOutput=True)

            # Compute dummy market return as a default
            self.computeDummyMarketReturns(thinFacPar, rm, regPar, assetData, returns_ESTU, rmgHolidayList)

            # If required, compute other dummy returns
            if thinFacPar.dummyReturnType != 'market':
                self.computeDummySectorReturns(thinFacPar, rm, regPar, assetData, returns_ESTU)

        return thinFacPar

    def computeDummyMarketReturns(self, thinFacPar, rm, regPar, assetData, returns, rmgHolidayList):
        """Compute dummy market return
        """
        # Compute market return for dummy asset
        self.log.info('Assigning market return to dummy assets')
        values = Matrices.allMasked(len(thinFacPar.factorNames))

        # Compute simple market return
        ret = ma.average(returns, weights=regPar.reg_weights)
        sclWt = ma.filled(regPar.reg_weights / ma.sum(regPar.reg_weights, axis=None), 0.0)
        for name in thinFacPar.factorNames:
            thinFacPar.dummyReturns[name] = ret
            thinFacPar.dummyRetWeights[name] = sclWt

        # For country factors we need to compute a market return excluding
        # later-traded assets to avoid lookahead bias
        if (ExposureMatrix.CountryFactor in thinFacPar.factorTypes) and not thinFacPar.applyRT:
            rmgZoneMap = dict((rmg, rmg.gmt_offset) for rmg in rm.rmg)
            estuIdxMap = dict(zip(regPar.reg_estu, list(range(len(regPar.reg_estu)))))
            for rmg in rm.rmg:
                if rmg.description not in thinFacPar.factorNames:
                    continue

                # Find all markets traded at same time or earlier
                validRMGIds = [r for r in rmgZoneMap.keys() if rmgZoneMap[r] >= rmgZoneMap[rmg]-1]
                validSubIds = list()
                for r_id in validRMGIds:
                    rmg_assets = Utilities.readMap(r_id, assetData.rmgAssetMap, set())
                    validSubIds.extend(list(rmg_assets))
                validSubIds = list(set(validSubIds).intersection(set(regPar.reg_estu)))

                # Compute regional market return
                ret = 0.0
                sclWgts = numpy.zeros((len(regPar.reg_weights)), float)
                subIdx = list()
                if len(validSubIds) > 0:
                    subIdx = [estuIdxMap[sid] for sid in validSubIds]
                    subRets = ma.take(returns, subIdx, axis=0)
                    subWts = ma.take(regPar.reg_weights, subIdx, axis=0)
                    ret = ma.filled(ma.average(subRets, weights=subWts), 0.0)

                # Assign returns and weights to relevant country factor
                for wtIdx in subIdx:
                    sclWgts[wtIdx] = regPar.reg_weights[wtIdx]
                sclWgts = sclWgts / ma.sum(sclWgts, axis=None)
                if (rmgHolidayList is not None) and (rmg in rmgHolidayList):
                    thinFacPar.dummyReturns[rmg.description] = 0.0
                else:
                    thinFacPar.dummyReturns[rmg.description] = ret
                thinFacPar.dummyRetWeights[rmg.description] = sclWgts
        return

    def computeDummySectorReturns(self, thinFacPar, rm, regPar, assetData, returns):
        """Compute industry/sector-based dummy returns corresponding to the dummy's parent
        classification. (eg. SuperSectors in the case of ICB Sectors)
        Applies only to industry factor dummies.
        """

        # Create relevant classification mappings
        classification = rm.industryClassification
        clsParentLevel = -1
        self.log.info('Assigning %s %s returns to dummy assets',
                        classification.name, thinFacPar.dummyReturnType)
        parents = classification.getClassificationParents(thinFacPar.dummyReturnType, self.modelDB)
        parentNames = [i.description for i in parents]
        childrenMap = {}
        for parent in parents:
            children = classification.getClassificationChildren(parent, self.modelDB)
            childrenMap[parent] = children

        # Build exposure matrix
        values = numpy.zeros((len(thinFacPar.factorNames)), float)
        parentExpM = ma.filled(classification.getExposures(self.date, regPar.reg_estu, parentNames,
                            self.modelDB, clsParentLevel), 0.0)

        for (pIdx, parent) in enumerate(parents):

            # Compute classification return
            assetsIdx = numpy.flatnonzero(parentExpM[pIdx])
            nonSectorIdx = numpy.flatnonzero(parentExpM[pIdx]==0)
            if len(assetsIdx) > 0:
                assetReturns = ma.take(returns, assetsIdx, axis=0)
                wgts = ma.take(regPar.reg_weights, assetsIdx, axis=0)
                parentReturn = ma.average(assetReturns, weights=wgts)
            else:
                parentReturn = 0.0

            # Compute weights
            sclWgts = numpy.array(regPar.reg_weights, copy=True)
            numpy.put(sclWgts, nonSectorIdx, 0.0)
            sclWgts = sclWgts / ma.sum(sclWgts, axis=None)

            # Map to relevant factor
            for child in childrenMap[parent]:
                if child.description in thinFacPar.factorNames:
                    thinFacPar.dummyReturns[child.description] = parentReturn
                    thinFacPar.dummyRetWeights[child.description] = sclWgts
                else:
                    logging.warning('No factor: %s for thin industry correction', child.description)
        return

    def buildFMPMapping(self, regPar, regData):
        """"Compute set of FMPs using output from the model factor regression
        """
        fmpDict = dict()
        nESTU = len(regPar.regEstuIdx)
        estuIdx = list(range(nESTU))
        sidIdxMap = dict(zip(regData.sidList, list(range(len(regData.sidList)))))

        # Set up regressor matrix components
        regM = regData.regressorMatrix * numpy.sqrt(regData.robustWeights)
        xwx = numpy.dot(regM, numpy.transpose(regM))
        xtw = regM * numpy.sqrt(regData.robustWeights)
        tfMatrix = numpy.eye(nESTU, dtype=float)

        # Incorporate thin factor component, if relevant
        if hasattr(regData, 'dummyRetWeights') and (type(regData.dummyRetWeights) is not list):
            tfMatrix = numpy.eye(nESTU, dtype=float)
            dummyRetWeights = numpy.take(regData.dummyRetWeights, estuIdx, axis=1)
            sumWgts = ma.sum(dummyRetWeights, axis=1)
            for idx in range(len(sumWgts)):
                dummyRetWeights[idx,:] = dummyRetWeights[idx,:] / sumWgts[idx]
            tfMatrix = ma.concatenate([tfMatrix, dummyRetWeights], axis=0)

        # Incorporate constraints, if relevant
        if regData.numConstraints > 0:
            cnRows = numpy.zeros((regData.numConstraints, nESTU), dtype=float)
            cnCols = numpy.zeros((tfMatrix.shape[0], regData.numConstraints), dtype=float)
            smallEye = numpy.eye(regData.numConstraints, dtype=float)
            cnCols = ma.concatenate([cnCols, smallEye], axis=0)
            tfMatrix = ma.concatenate([tfMatrix, cnRows], axis=0)
            tfMatrix = ma.concatenate([tfMatrix, cnCols], axis=1)
        xtw = numpy.dot(xtw, tfMatrix)

        # Compute the FMPs
        fmpMatrix = Utilities.robustLinearSolver(xtw, xwx, computeStats=False).params
        if regData.numConstraints > 0:
            ccMatrix = fmpMatrix[:, -regData.numConstraints:]
            fmpMatrix = fmpMatrix[:, :-regData.numConstraints]
        for (iLoc, idx) in enumerate(regData.okFactorIdx):
            fmpDict[regPar.regFactorNames[idx]] = fmpMatrix[iLoc, :]

        # Compute the leftover part corresponding to the constraints
        constraintComponent = None
        if regData.numConstraints > 0:
            constraintComponent = Utilities.Struct()
            cc_X = regData.regressorMatrix[:, -regData.numConstraints:]
            ccDict = dict()
            ccXDict = dict()
            for (iLoc, idx) in enumerate(regData.okFactorIdx):
                ccDict[regPar.regFactorNames[idx]] = ccMatrix[iLoc, :]
                ccXDict[regPar.regFactorNames[idx]] = cc_X[iLoc, :]
            constraintComponent.ccDict = ccDict
            constraintComponent.ccXDict = ccXDict

        return fmpDict, constraintComponent

class AsymptoticPrincipalComponents2017:
    def __init__(self, numFactors, flexible=False, trimExtremeExposures=True,
            replaceReturns=True, applyJones=False, TOL=1.0e-10):
        self.log = logging.getLogger('FactorReturns.PrincipalComponents')
        self.numFactors = numFactors
        self.trimExtremeExposures = trimExtremeExposures
        self.flexible = flexible
        self.TOL = TOL
        self.applyJones = applyJones
        self.replaceReturns = replaceReturns

    def calc_ExposuresAndReturns_Inner(
                    self, returns, T, scaleVector=None, estu=None, useAPCA=True, trimExtremeExposures=True):
        """Asymptotic Principal Components Analysis (Connor, Korajczyk 1986)
        Uses APCA to determine factor returns, and exposures for non-ESTU
        are backed out via time-series regression.
        Input is a TimeSeriesMatrix of returns and ESTU indices.
        Returns exposures, factor returns, and specific returns as arrays
        """
        self.log.debug('calc_ExposuresAndReturns: begin')

        nonest = list(set(range(returns.data.shape[0])).difference(set(estu)))
        returnsESTU = Utilities.screen_data(returns.data)
        returnsESTU = numpy.take(ma.filled(returnsESTU, 0.0), estu, axis=0)

        if useAPCA:
            # Incorporate weighting if required
            if scaleVector is not None:
                weights = 1.0 / (ma.take(scaleVector, estu) ** 0.5)
                scaledReturns = ma.filled(ma.transpose(weights * ma.transpose(returnsESTU[:,-T:])), 0.0)
            else:
                scaledReturns = returnsESTU[:,-T:]

            # Using SVD, more efficient than eigendecomp on huge matrix
            try:
                (d, v) = linalg.svd(scaledReturns, full_matrices=False)[1:]
                d = d**2 / returnsESTU.shape[0]
            except:
                logging.warning('SVD routine failed... computing eigendecomposition instead')
                tr = numpy.dot(numpy.transpose(scaledReturns), scaledReturns)
                (d, v) = linalg.eigh(tr)
                d = d / returnsESTU.shape[0]
                v = numpy.transpose(v)
            order = numpy.argsort(-d)
            d = numpy.take(d, order, axis=0)
            v = numpy.take(v, order, axis=0)[0:self.numFactors,:]

            # Take factor returns to be right singular vectors
            factorReturns = v
        else:
            # If fewer assets than t-values, switch to regular PCA
            # Compute svd of returns history
            try:
                (u, d, v) = linalg.svd(returnsESTU[:,-T:], full_matrices=False)
            except:
                logging.warning('SVD routine failed... computing eigendecomposition instead')
                tr = numpy.dot(returnsESTU[:,-T:], numpy.transpose(returnsESTU[:,-T:]))
                (d, u) = linalg.eigh(tr)
                d = ma.filled(ma.sqrt(d), 0.0)
            d = d / numpy.sqrt(T)
            order = numpy.argsort(-d)
            u = numpy.take(u, order, axis=1)
            d = numpy.take(d, order, axis=0)
            # Create estu exposure matrix
            expMatrixEstu = numpy.dot(u[:,:self.numFactors],numpy.diag(d[:self.numFactors]))
            # Compute factor returns
            factorReturns = Utilities.robustLinearSolver(\
                    returnsESTU, expMatrixEstu, computeStats=False).params

        # Back out exposures
        if useAPCA:
            exposureMatrix = numpy.dot(ma.filled(returns.data[:,-T:], 0.0), numpy.transpose(v))
            if T < returnsESTU.shape[1]:
                expMatrixEstu = ma.take(exposureMatrix, estu, axis=0)
                returnsESTU = ma.take(ma.filled(returns.data, 0.0), estu, axis=0)
                fullFactorReturns = Utilities.robustLinearSolver(returnsESTU, expMatrixEstu).params
                factorReturns = numpy.array(fullFactorReturns, copy=True)
        else:
            exposureMatrix = Utilities.robustLinearSolver(\
                    numpy.transpose(ma.filled(returns.data[:,-T:], 0.0)),
                    numpy.transpose(factorReturns[:,-T:]), computeStats=False).params
            exposureMatrix = numpy.transpose(exposureMatrix)

        # Checks on data for pathological values
        exposureMatrix = Utilities.screen_data(exposureMatrix, fill=True)
        factorReturns = Utilities.screen_data(factorReturns, fill=True)
        if trimExtremeExposures:
            outlierClass = Outliers.Outliers()
            exposureMatrix = outlierClass.twodMAD(exposureMatrix, axis=0, suppressOutput=True)

        # Calculate specific returns
        specificReturns = returns.data - numpy.dot(exposureMatrix, factorReturns)

        # Calculate regression statistics
        regressANOVA = RegressionANOVA(returns.data[:,-1],
                        specificReturns[:,-1], self.numFactors, estu)

        # Not sure if this really applies to PCA of TxT matrix...
        prc = numpy.sum(d[0:self.numFactors], axis=0) / numpy.sum(d, axis=0)
        self.log.info('%d factors explains %f%% of variance', self.numFactors, 100*prc)
        self.log.debug('calc_ExposuresAndReturns: end')

        return (exposureMatrix, factorReturns, specificReturns, regressANOVA, prc)

    def calc_JonesAdjustment(self, returns, T, initialF, estu=None):
        """Asymptotic Principal Components Analysis (Connor, Korajczyk 1986)
        Uses APCA to determine factor returns, and exposures for non-ESTU
        are backed out via time-series regression.
        """
        self.log.debug('calc_ExposuresAndReturns: begin')

        nonest = list(set(range(returns.data.shape[0])).difference(set(estu)))
        returnsESTU = Utilities.screen_data(returns.data)
        returnsESTU = numpy.take(ma.filled(returnsESTU, 0.0), estu, axis=0)

        # Compute initial matrices
        initialF = ma.filled(initialF[:,-T:], 0.0)
        C = numpy.dot(numpy.transpose(returnsESTU[:,-T:]), returnsESTU[:,-T:]) / float(returnsESTU.shape[0])
        D = C - numpy.dot(numpy.transpose(initialF), initialF)
        D = numpy.sqrt(numpy.diag(D))
        scaledReturns = numpy.dot(returnsESTU[:,-T:], numpy.diag(1.0 / D))

        # Using SVD, more efficient than eigendecomp on huge matrix
        try:
            # SVD factorisation is A=USV
            (sigma, v) = linalg.svd(scaledReturns, full_matrices=False)[1:]
            sigma = sigma * sigma / returnsESTU.shape[0]
        except:
            logging.warning('SVD routine failed... computing eigendecomposition instead')
            tr = numpy.dot(numpy.transpose(scaledReturns), scaledReturns)
            # eigh factorisation is A=VSV'
            (sigma, v) = linalg.eigh(tr)
            sigma = sigma / returnsESTU.shape[0]
            v = numpy.transpose(v)

        # Order and take the relevant subset of eigenvectors/eigenvalues
        order = numpy.argsort(-sigma)
        sigma = numpy.take(sigma, order, axis=0)
        v = numpy.take(v, order, axis=0)[0:self.numFactors,:]

        # Compute the factor returns
        p1 = numpy.dot(numpy.diag(D), numpy.transpose(v))
        p2 = numpy.diag(numpy.sqrt(sigma[0:self.numFactors] - 1.0))
        factorReturns = numpy.transpose(numpy.dot(p1, p2))

        # Back out full set of factor returns
        exposureMatrix = Utilities.robustLinearSolver(
                numpy.transpose(ma.filled(returns.data[:,-T:], 0.0)),
                numpy.transpose(factorReturns)).params
        exposureMatrix = numpy.transpose(exposureMatrix)
        if T < returnsESTU.shape[1]:
            expMatrixEstu = ma.take(exposureMatrix, estu, axis=0)
            returnsESTU = ma.take(ma.filled(returns.data, 0.0), estu, axis=0)
            fullFactorReturns = Utilities.robustLinearSolver(returnsESTU, expMatrixEstu).params
            factorReturns = numpy.array(fullFactorReturns, copy=True)

        # Checks on data for pathological values
        exposureMatrix = Utilities.screen_data(exposureMatrix, fill=True)
        factorReturns = Utilities.screen_data(factorReturns, fill=True)
        outlierClass = Outliers.Outliers()
        exposureMatrix = outlierClass.twodMAD(exposureMatrix, axis=0, suppressOutput=True)

        # Calculate specific returns
        specificReturns = returns.data - numpy.dot(exposureMatrix, factorReturns)

        # Calculate regression statistics
        regressANOVA = RegressionANOVA(returns.data[:,-1],
                        specificReturns[:,-1], self.numFactors, estu)

        # Not sure if this really applies to PCA of TxT matrix...
        prc = numpy.sum(sigma[0:self.numFactors], axis=0) / numpy.sum(sigma, axis=0)
        self.log.info('%d factors explains %f%% of variance', self.numFactors, 100*prc)
        self.log.debug('calc_ExposuresAndReturns: end')

        return (exposureMatrix, factorReturns, specificReturns, regressANOVA, prc)


    def calc_ExposuresAndReturns(self, returns, estu=None, T=None, flexibleOverride=None):
        if T == None:
            T = returns.data.shape[1]

        # If no ESTU specified, use all assets
        if estu is None:
            estu = list(range(returns.data.shape[0]))
        if T > returns.data.shape[1]:
            T = returns.data.shape[1]
        originalReturns = ma.array(returns.data, copy=True)

        if flexibleOverride is not None:
            flexible = flexibleOverride
        else:
            flexible = self.flexible

        APCA = True
        if len(estu) < 1:
            raise LookupError('Empty estimation universe')
        if T >= self.numFactors:
            if T >= len(estu):
                self.log.warning('Returns history (%d) > ESTU assets (%d), unreliable estimates',
                              T, len(estu))
                if flexible:
                    self.log.warning('Switching to vanilla PCA')
                    APCA = False
            self.log.info('Returns history: %d periods, by %d assets', T, len(estu))
        else:
            raise LookupError('Asymptotic PCA requires ESTU assets (%d) > returns history (%d) > factors (%d)' %\
                    (len(estu),   returns.data.shape[1], self.numFactors))

        if APCA:
            # First iteration: APCA to obtain specific returns
            logging.info('********************** Computing APCA iteration 1')
            (exposureMatrix_0, factorReturns_0, specificReturns, RegANOVA_0, pct_0) = \
                    self.calc_ExposuresAndReturns_Inner(returns, T, estu=estu,
                            trimExtremeExposures=self.trimExtremeExposures)
            if self.replaceReturns:
                self.replaceMissingReturns(returns, originalReturns, exposureMatrix_0, factorReturns_0)

            if self.TOL is not None:
                i_count = 1
                max_count = 10
                diff = 9999.9
                residVar0 = numpy.zeros((returns.data.shape[0]), float)

                while (i_count < max_count) and (diff > self.TOL):
                    # Apply residual variance adjustment to returns
                    logging.info('Computing residual variance vector')
                    residVar = self.initialResidualVariance(specificReturns[:,-T:])
                    diff = residVar - residVar0
                    diff = ma.filled(ma.average(diff * diff, axis=None), 0.0)
                    i_count += 1
                    residVar0 = ma.array(residVar, copy=True)
                    logging.info('Residual norm: %.4e', diff)

                    # Second iteration: APCA on adjusted returns
                    logging.info('********************** Computing APCA iteration %d', i_count)
                    (exposureMatrix, factorReturns, specificReturns, regressANOVA, pct) = \
                            self.calc_ExposuresAndReturns_Inner(
                                    returns, T, scaleVector=residVar, estu=estu,
                                    trimExtremeExposures=self.trimExtremeExposures)
            else:
                # Just do simple two-stage APCA
                # Apply residual variance adjustment to returns
                logging.info('Computing residual variance vector')
                residVar = self.initialResidualVariance(specificReturns[:,-T:])

                # Second iteration: APCA on adjusted returns
                logging.info('********************** Computing APCA iteration 2')
                (exposureMatrix, factorReturns, specificReturns, regressANOVA, pct) = \
                        self.calc_ExposuresAndReturns_Inner(
                                returns, T, scaleVector=residVar, estu=estu,
                                trimExtremeExposures=self.trimExtremeExposures)

            if self.applyJones:
                # If we're applying the Jones adaptation of APCA, then previous steps
                # were just preliminary input

                # Initial values
                i_count = 1
                max_count = 10
                diff = 9999.9
                residVar0 = numpy.zeros((returns.data.shape[0]), float)

                while (i_count < max_count) and (diff > self.TOL):
                    logging.info('********************** Computing JPCA iteration %d', i_count)
                    (exposureMatrix, factorReturns, specificReturns, regressANOVA, pct) = \
                            self.calc_JonesAdjustment(returns, T, initialF=factorReturns, estu=estu)

                    # Check for convergence
                    residVar = self.initialResidualVariance(specificReturns[:,-T:])
                    diff = residVar - residVar0
                    diff = ma.filled(ma.average(diff * diff, axis=None), 0.0)
                    i_count += 1
                    residVar0 = ma.array(residVar, copy=True)
                    logging.info('Residual norm: %.4e', diff)
        else:
            # Vanilla PCA 
            logging.info('Computing regular PCA')
            (exposureMatrix, factorReturns, specificReturns, regressANOVA, pct) = \
                                self.calc_ExposuresAndReturns_Inner(
                                        returns, T, estu=estu, useAPCA=False,
                                        trimExtremeExposures=self.trimExtremeExposures)

        return (exposureMatrix, factorReturns, specificReturns, regressANOVA, pct)
        
    def replaceMissingReturns(self, returns, returnsData, exposures, factorReturns):
        from riskmodels import ProcessReturns
        # Use estimated returns as a new set of proxy returns
        proxyReturns = numpy.dot(ma.filled(exposures, 0.0), factorReturns)
        proxyReturns = numpy.clip(proxyReturns, -0.75, 2.0)
        returnsData = ma.masked_where(returns.missingFlag, returnsData)
        missingDataMask = numpy.array(ma.getmaskarray(returnsData), copy=True)
        returns.data = ProcessReturns.fill_and_smooth_returns(
                returnsData, proxyReturns, mask=missingDataMask, preIPOFlag=returns.preIPOFlag)[0]
        return

    def initialResidualVariance(self, specificReturns):
        """Compute a simple residual variance estimate For input to the Jones method for adjusted APCA
        """
        self.log.debug('initialResidualVariance: begin')
        specificVars = ma.array([ma.inner(specificReturns[n], specificReturns[n])
                        for n in range(specificReturns.shape[0])], float)
        specificVars = ma.masked_where(specificVars==0.0, specificVars)
        # Trim outliers to prevent any whacky numbers
        opms = dict()
        opms['nBounds'] = [5.0, 5.0]
        outlierClass = Outliers.Outliers(opms)
        specificVars = outlierClass.twodMAD(specificVars, suppressOutput=True) / float(specificReturns.shape[1])
        # Output stats and return
        stdVec = 100.0 * ma.sqrt(250* specificVars)
        self.log.info('Residual (annualised) stdev bounds: (%.4f%%, %.4f%%)', ma.min(stdVec), ma.max(stdVec))
        self.log.debug('initialResidualVarianceA: end')
        return specificVars
