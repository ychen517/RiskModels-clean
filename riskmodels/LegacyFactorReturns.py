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
from riskmodels import Matrices
from riskmodels.Matrices import ExposureMatrix
from riskmodels import LegacyUtilities as Utilities
from riskmodels.Factors import ModelFactor
from riskmodels import Outliers

def calc_Weights(rmg, modelDB, marketDB, modelDate, assetData, universe,
        baseCurrencyID=None, clip=True, clipByIndustry=False):

    # Get list of dates
    if len(rmg) > 1:
        mcapDates = modelDB.getDateRange(rmg, None, modelDate, excludeWeekend=True, calendarDaysBack=30)
    else:
        mcapDates = modelDB.getDates(rmg, modelDate, 19)

    # Load average mcaps
    nAssets = len(universe)
    avgMktCaps = modelDB.getAverageMarketCaps(
                mcapDates, universe, baseCurrencyID, marketDB)
    weights = ma.sqrt(avgMktCaps)

    # Trim at 95th percentile
    if clipByIndustry and hasattr(assetData, 'industryAssetMap'):
        univIdxMap = dict(zip(universe, list(range(len(universe)))))
        for indSubIDs in assetData.industryAssetMap.values():
            idxList = [univIdxMap[sid] for sid in indSubIDs if sid in universe]
            subWts = numpy.take(weights, idxList, axis=0)
            nSub = len(subWts)
            if nSub > 1:
                upperBound = Utilities.prctile(subWts, [90.0])[0]
                subWts = ma.where(subWts>upperBound, upperBound, subWts)
                ma.put(weights, idxList, subWts)
        upperBound = Utilities.prctile(weights, [99.0])[0]
        weights = ma.where(weights>upperBound, upperBound, weights)
    elif clip:
        C = min(100, int(round(nAssets*0.05)))
        sortindex = ma.argsort(weights)
        ma.put(weights, sortindex[nAssets-C:nAssets],
                weights[sortindex[nAssets-C]])
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

class RobustRegressionLegacy:
    def __init__(self, params):
        self.log = logging.getLogger('FactorReturns.RobustRegression')
        self.allParameters = params
        self.regParameters = None

    def computeDummyReturns(self, parameters, excessReturns, estu,
                            weights, universe, date, data=None):

        # Initialise dict of dummy returns if necessary
        if not hasattr(parameters, 'dummyReturns'):
            parameters.dummyReturns = dict()
            parameters.dummyRetWeights = dict()
         
        # Compute market return for dummy asset
        if parameters.dummyReturnType == 'market':
            self.log.info('Assigning market return to dummy assets')
            values = Matrices.allMasked(len(parameters.factorNames))
            excessReturns_ESTU = ma.take(excessReturns, estu)
            # Remove any really large returns
            (excessReturns_ESTU, mad_bounds) = Utilities.mad_dataset(
                    excessReturns_ESTU, -15, 15, axis=0)
            # Compute simple market return
            ret = ma.average(excessReturns_ESTU, weights=weights)
            sclWt = ma.filled(weights / ma.sum(weights, axis=None), 0.0)
            for name in parameters.factorNames:
                parameters.dummyReturns[name] = ret
                parameters.dummyRetWeights[name] = sclWt

            # For country factors we need to compute a market return excluding
            # later-traded assets to avoid lookahead bias
            if (ExposureMatrix.CountryFactor in parameters.factorTypes) and not parameters.applyRT:
                rmgZoneMap = dict((rmg.rmg_id, rmg.gmt_offset) for rmg in parameters.rmgList)
                estuSids = [universe[idx] for idx in estu]
                estuIdxMap = dict(zip(estuSids, list(range(len(estuSids)))))
                for rmg in parameters.rmgList:
                    if rmg.description not in parameters.factorNames:
                        continue

                    # Find all markets traded at same time or earlier
                    validRMGIds = [rmg_id for rmg_id in rmgZoneMap.keys() if \
                            rmgZoneMap[rmg_id] >= rmgZoneMap[rmg.rmg_id]-1]
                    validSubIds = list()
                    for r_id in validRMGIds:
                        if r_id in parameters.data.rmgAssetMap:
                            validSubIds.extend(parameters.data.rmgAssetMap[r_id])
                    validSubIds = list(set(validSubIds).intersection(set(estuSids)))

                    # Compute regional market return
                    ret = 0.0
                    sclWgts = numpy.zeros((len(weights)), float)
                    if len(validSubIds) > 0:
                        subIdx = [estuIdxMap[sid] for sid in validSubIds]
                        subRets = ma.take(excessReturns_ESTU, subIdx, axis=0)
                        subWts = ma.take(weights, subIdx, axis=0)
                        ret = ma.filled(ma.average(subRets, weights=subWts), 0.0)

                    # Assign returns and weights to relevant country factor
                    for wtIdx in subIdx:
                        sclWgts[wtIdx] = weights[wtIdx]
                    sclWgts = sclWgts / ma.sum(sclWgts, axis=None)
                    parameters.dummyReturns[rmg.description] = ret
                    parameters.dummyRetWeights[rmg.description] = sclWgts
            return
        
        # Otherwise assign the return corresponding to the dummy's parent
        # classification. (eg. SuperSectors in the case of ICB Sectors)
        # Applies only to industry factor dummies.
         
        # Initialise parameters
        classification = parameters.dummyCls
        clsParentLevel = -1
        modelDB = parameters.modelDB
        # Create relevant classification mappings
        self.log.info('Assigning %s %s returns to dummy assets',
                        classification.name, parameters.dummyReturnType)
        parents = classification.getClassificationParents(parameters.dummyReturnType, modelDB)
        parentNames = [i.description for i in parents]
        childrenMap = {}
        for parent in parents:
            children = classification.getClassificationChildren(parent, modelDB)
            childrenMap[parent] = children
         
        # Build exposure matrix
        values = numpy.zeros((len(parameters.factorNames)), float)
        ids_ESTU = [universe[n] for n in estu]
        factorIdxMap = dict([(parameters.factorNames[i], i) for i
                             in parameters.factorIndices])
        returns_ESTU = ma.take(excessReturns, estu, axis=0)
        parentExpM = classification.getExposures(
            date, ids_ESTU, parentNames, modelDB,
            clsParentLevel).filled(0.0)
         
        # Remove any really large returns
        (returns_ESTU, mad_bounds) = Utilities.mad_dataset(
                returns_ESTU, -25, 25, axis=0)
         
        # Map returns to sectors/industries etc.
        for (pIdx, parent) in enumerate(parents):
            assetsIdx = numpy.flatnonzero(parentExpM[pIdx])
            nonSectorIdx = numpy.flatnonzero(parentExpM[pIdx]==0)
            if len(assetsIdx) > 0:
                assetReturns = ma.take(returns_ESTU, assetsIdx, axis=0)
                wgts = ma.take(weights, assetsIdx) / \
                                ma.sum(ma.take(weights, assetsIdx))
                parentReturn = ma.average(assetReturns, weights=wgts)
            else:
                parentReturn = 0.0
            sclWgts = numpy.array(weights, copy=True)
            numpy.put(sclWgts, nonSectorIdx, 0.0)
            sclWgts = sclWgts / ma.sum(sclWgts, axis=None)
            for child in childrenMap[parent]:
                if child.description in factorIdxMap:
                    idx = factorIdxMap[child.description]
                    if idx in parameters.factorIndices:
                        parameters.dummyReturns[child.description] = parentReturn
                        parameters.dummyRetWeights[child.description] = sclWgts
                else:
                    logging.warning('No factor: %s for thin industry correction', child.description)
        return

    def insertDummyAssets(self, parameters, rd, factorReturns):
        """Examines the factors in positions (idxToCheck) for thinness.
        Inserts dummy assets into the regressor matrix where necessary,
        returns a new copy of the regressor matrix, returns, and weights,
        as well as an array of factor returns and lists of thin and empty
        factor positions.
        """
        thinFactorIndices = []
        emptyFactorIndices = []

        # Make sure dummy returns are present
        missing = len(numpy.flatnonzero(
                            ma.getmaskarray(parameters.dummyReturns)))
        if len(parameters.factorIndices) > 0 and missing==len(parameters.dummyReturns):
            raise Exception('Returns have not been specified for dummy assets!')

        # Check specified factors for thin-ness
        for (i, idx) in enumerate(parameters.factorIndices):
            assetsIdx = numpy.flatnonzero(rd.regressorMatrix[idx,:])
            factorName = parameters.factorNames[idx]
            factorType = parameters.factorTypes[i]
            dummyRet = parameters.dummyReturns[factorName]
            dummyRetWeights = parameters.dummyRetWeights[factorName]
            if len(assetsIdx) == 0:
                # Empty factor, keep track of these and omit from reg later
                dummyRet = 0.0
                self.log.warning('Empty factor: %s, ret %f', factorName, dummyRet)
                factorReturns[idx] = dummyRet
                emptyFactorIndices.append(idx)
            else:
                # Herfindahl
                factorWeights = ma.take(rd.weights_ESTU, assetsIdx, axis=0) * \
                        ma.take(rd.regressorMatrix[idx], assetsIdx, axis=0)
                factorWeights = abs(factorWeights)
                totalWgt = ma.sum(factorWeights, axis=0)
                if totalWgt <= 0.0:
                    # Empty factor, keep track of these and omit from reg later
                    dummyRet = 0.0
                    self.log.warning('Empty factor: %s, ret %f', factorName, dummyRet)
                    factorReturns[idx] = dummyRet
                    emptyFactorIndices.append(idx)
                else:
                    wgt = factorWeights / totalWgt
                    score = 1.0 / ma.inner(wgt, wgt)

                    # This factor is thin...
                    if score < parameters.dummyThreshold:
                        if factorType == ExposureMatrix.StyleFactor:
                            # Thin style factor - treat as empty
                            self.log.warning('Thin style factor: %s', factorName)
                            factorReturns[idx] = 0.0
                            emptyFactorIndices.append(idx)
                        else:
                            partial1 = score**4.0
                            partial2 = parameters.dummyThreshold**4.0
                            dummyWgt = (parameters.dummyThreshold - 1.0) * (partial1 - partial2) / (1.0 - partial2)
                            dummyWgt *= ma.sum(factorWeights, axis=0) / score
                            dummyExp = numpy.zeros((rd.regressorMatrix.shape[0], 1))
                            dummyExp[idx] = 1.0
                            nTrueAssets = len(rd.weights_ESTU) - len(thinFactorIndices)
                            # Assign style factor exposures to dummy, if applicable
                            for ii in parameters.nonzeroExposuresIdx:
                                if Utilities.is_binary_data(rd.regressorMatrix[ii,:nTrueAssets+1]):
                                    val = ma.median(rd.regressorMatrix[ii,:nTrueAssets+1], axis=0)
                                else:
                                    val = ma.inner(rd.weights_ESTU[:nTrueAssets+1], rd.regressorMatrix[ii,:nTrueAssets+1])
                                    val /= ma.sum(rd.weights_ESTU[:nTrueAssets+1])
                                dummyExp[ii] = val

                            # Append dummy exposures, weight, and return to estu
                            rd.regressorMatrix = numpy.concatenate(
                                            [rd.regressorMatrix, dummyExp], axis=1)
                            rd.excessReturns_ESTU = ma.concatenate(
                                    [rd.excessReturns_ESTU, ma.array([dummyRet])], axis=0)
                            rd.weights_ESTU = ma.concatenate(
                                    [rd.weights_ESTU, ma.array([dummyWgt])], axis=0)
                            thinFactorIndices.append(idx)
                            self.log.info('Thin factor: %s (N: %.2f/%d, Dummy wgt: %.1f%%, ret: %.1f%%)',
                                    factorName, score, len(assetsIdx),
                                    100.0*dummyWgt/(dummyWgt+totalWgt), 100.0*dummyRet)
                            if rd.subIssues is not None:
                                rd.subIssues.append('%s-Dummy' % factorName.replace(',','').replace(' ',''))
                            if not hasattr(rd, 'dummyRetWeights'):
                                rd.dummyRetWeights = numpy.array(dummyRetWeights, float)[numpy.newaxis,:]
                            else:
                                rd.dummyRetWeights = ma.concatenate(
                                        [rd.dummyRetWeights, numpy.array(dummyRetWeights, float)[numpy.newaxis,:]], axis=0)

        return (rd, factorReturns, thinFactorIndices, emptyFactorIndices)

    def processFactorIndices(self, rm, rd, factorNames, factorReturns, constraints, expM,
                            robustWeights=None):
        """Routine to sort out the thin and empty from the big and fat
        Also sets up constraints for singular regressions
        """
        factorIndices = [i for i in range(len(factorReturns))]
        # Invoke thin factor correction mechanism if required
        if self.regParameters.getThinFactorCorrection() and self.thinFactorParameters is not None:
            (rd, factorReturns, thinFactorIndices, emptyFactorIndices) = \
                    self.insertDummyAssets(
                        self.thinFactorParameters, rd, factorReturns)
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
            if (rc.factorType is ExposureMatrix.IndustryFactor) or (rc.factorType is ExposureMatrix.CountryFactor):
                factorIdx = [i for i in factorIndices if factorNames[i] in \
                        expM.getFactorNames(rc.factorType)]
            else:
                factorIdx = []
            if len(factorIdx) > 0:
                rd = rc.createDummyAsset(factorIdx, rm, rd, totalDummies,
                                    expM, robustWeights=robustWeights)
                if rd.subIssues is not None:
                    rd.subIssues.append('%s-Constraint' % rc.factorType)
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
    
    def calc_Factor_Specific_Returns(
            self, rm, date, estimationUniverseIdx, excessReturns, exposureMatrix,
            factorNames, weights, expM, constraints, robustWeights=None):
        
        if hasattr(rm, 'iReg'):
            iReg = rm.iReg + 1
        else:
            iReg = ''

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
        rd = Utilities.Struct()
        rd.excessReturns_ESTU = ma.take(excessReturns, indices_ESTU, axis=0)
        rd.weights_ESTU = ma.take(weights, nonMissingReturnsIdx, axis=0)
        rd.indices_ESTU = indices_ESTU
        k = self.regParameters.getRlmKParameter()
        if k == None:
            opms = dict()
            opms['nBounds'] = [25.0, 25.0]
            outlierClass = Outliers.Outliers(opms)
            rd.excessReturns_ESTU = outlierClass.twodMAD(rd.excessReturns_ESTU)
         
        # Initialise regressor matrix
        exposureMatrix = ma.filled(exposureMatrix, 0.0)
        rd.regressorMatrix = numpy.take(exposureMatrix, indices_ESTU, axis=1)
        rd.subIssues = None
        if expM is not None:
            rd.subIssues = numpy.take(expM.getAssets(), indices_ESTU, axis=0)
            rd.subIssues = [sid.getSubIDString() for sid in rd.subIssues]
        factorReturns = numpy.zeros(exposureMatrix.shape[0])
        self.log.info('%d assets in actual estimation universe', len(indices_ESTU))
        
        # Process thin industries and constraints
        rd.useRealCaps = self.regParameters.realWeightsForConstraints()
        if rd.useRealCaps:
            rd.modelDB = self.modelDB
            rd.marketDB = self.marketDB
            rd.date = date
        (rd, factorReturns) = self.processFactorIndices(
                rm, rd, factorNames, factorReturns, constraints, expM,
                robustWeights=robustWeights)

        # Output regressor matrix (useful for debugging)
        if rm.debuggingReporting and rd.subIssues is not None: 
            outfile = open('tmp/regressorMatrix-%d.csv' % iReg, 'w')
            outfile.write(',Return,Weight')
            for i in rd.factorIndices:
                outfile.write(',%s' % factorNames[i].replace(',',''))
            outfile.write('\n')
            for j in range(rd.regressorMatrix.shape[1]):
                outfile.write('%s,%f,%f' % (rd.subIssues[j], rd.excessReturns_ESTU[j], rd.weights_ESTU[j]))
                for m in range(rd.regressorMatrix.shape[0]):
                    outfile.write(',%f' % rd.regressorMatrix[m,j])
                outfile.write('\n')
            outfile.close()
        
        # Set up regression data
        self.sm = sm
        rd.regressionRHS = rd.regressorMatrix * numpy.sqrt(rd.weights_ESTU)
        rd.regressionLHS = rd.excessReturns_ESTU * ma.sqrt(rd.weights_ESTU)

        k = self.regParameters.getRlmKParameter()
        if robustWeights is not None:
            k = None
            rd.regressionRHS = rd.regressionRHS * numpy.sqrt(robustWeights)
            rd.regressionLHS = rd.regressionLHS * numpy.sqrt(robustWeights)
        
        # Report on final state of the regressor matrix
        xwx = numpy.dot(rd.regressionRHS, numpy.transpose(rd.regressionRHS))
        (eigval, eigvec) = linalg.eigh(xwx)
        conditionNumber = max(eigval) / min(eigval)
        self.log.info('Weighted regressor (%i by %i) has condition number %f',
                xwx.shape[0], xwx.shape[1], conditionNumber)

        # Perform the robust regression loop
        maxRuns = 10
        for i_run in range(maxRuns):
            if k == None:
                logging.info('Running OLS regression')
                model = sm.OLS(numpy.array(rd.regressionLHS), \
                        numpy.transpose(numpy.array(rd.regressionRHS)))
            else:
                model = sm.RLM(numpy.array(rd.regressionLHS), \
                        numpy.transpose(numpy.array(rd.regressionRHS)),
                        M = sm.robust.norms.HuberT(t=k))
            if rm.forceRun:
                try:
                    rlmResult = model.fit(maxiter=500)
                except:
                    self.log.warning('Regression failed: defective inputs')
                    rlmDownWeights = numpy.ones((len(rd.excessReturns_ESTU)), float)
                    constraintWeight = dict()
                    rlmResult = None
                    allZeroFR = True
                    break
            else:
                rlmResult = model.fit(maxiter=500)
                
            # Pull out final regression weights
            if k == None:
                if robustWeights is None:
                    rlmDownWeights = numpy.ones((len(rd.regressionLHS)), float)
                else:
                    rlmDownWeights = robustWeights
            else:
                rlmDownWeights = numpy.array(rlmResult.weights)

            # Grab factor returns from R
            regressionResults = numpy.array(rlmResult.params)
            allZeroFR = True
            if len(factorReturns) == 1:
                factorReturns[0] = regressionResults[0]
                allZeroFR = False
            else:
                for (ix,jx) in enumerate(rd.factorIndices):
                    factorReturns[jx] = regressionResults[ix]
                    if abs(factorReturns[jx]) > 0.0:
                        allZeroFR = False
            
            # Check that linear constraints (if any) are being satisfied
            (constraintsOK, constraintWeight, rd) = self.check_regression_constraints(
                        rd, factorNames, factorReturns, constraints)
            if constraintsOK:
                break
        
        # Report on effect of robust regression
        tmp = rd.excessReturns_ESTU * numpy.sqrt(rlmDownWeights)
        downWeightFactor = numpy.sum(abs(tmp)) / numpy.sum(abs(rd.excessReturns_ESTU))
        logging.info('Regression %s Downweight factor: %6.4f', iReg, downWeightFactor)
        logging.info('Excess returns bounds after downweighting (estu): [%.3f, %.3f]' % \
                (ma.min(tmp, axis=None), ma.max(tmp, axis=None)))

        # Output regression weights (useful for debugging)
        if rm.debuggingReporting and rd.subIssues is not None:
            outfile = open('tmp/regression-%s-Weights-%s.csv' % (iReg, date), 'w')
            outfile.write('ID,InitialWeight,RobustWeight,\n')
            for (sid, wt, rWt) in zip(rd.subIssues, rd.weights_ESTU, rlmDownWeights):
                outfile.write('%s,%f,%f,\n' % (sid, wt, rWt))
            outfile.close()

        # Compute FMP matrix
        robustWeights = rlmDownWeights * rd.weights_ESTU
        fmpDict = dict()
        constrComp = None
        if rd.subIssues is not None:
            fmpDict, constrComp = Utilities.buildFMPMapping(
                    rd, robustWeights, len(indices_ESTU), factorNames, nonMissingReturnsIdx)

        # Compute residuals from actual regression
        specificReturns_Reg = rd.excessReturns_ESTU - numpy.dot(
                numpy.transpose(rd.regressorMatrix), regressionResults)
        spec_return = ma.average(specificReturns_Reg, weights=rd.weights_ESTU)
        self.log.info('Regression %s actual ESTU specific return (reg wt): %.8f', iReg, spec_return)

        # Get rid of some residue from the dummies
        if len(rd.thinFactorIndices) + rd.numConstraints > 0:
            rd.weights_ESTU = rd.weights_ESTU[:-len(rd.thinFactorIndices) - rd.numConstraints]
            robustWeights = robustWeights[:-len(rd.thinFactorIndices) - rd.numConstraints]
         
        # Compute residuals
        specificReturns = excessReturns - numpy.dot(
            numpy.transpose(exposureMatrix), factorReturns)
        spec_return_ESTU = ma.take(specificReturns, indices_ESTU, axis=0)
        spec_return = ma.average(spec_return_ESTU, weights=rd.weights_ESTU)
        self.log.info('Regression %s final ESTU specific return (reg wt): %.8f', iReg, spec_return)
        if expM is not None:
            subIssues = numpy.take(expM.getAssets(), indices_ESTU, axis=0)
            cWeight = self.modelDB.getAverageMarketCaps(
                    [date], subIssues, rm.numeraire.currency_id, self.marketDB)
            spec_return = ma.average(spec_return_ESTU, weights=cWeight)
            self.log.info('Regression %s final ESTU specific return (cap wt): %.8f', iReg, spec_return)
        
        # Calculate regression statistics
        regressANOVA = RegressionANOVA(excessReturns, specificReturns,
                exposureMatrix.shape[0], indices_ESTU, robustWeights)
        regressionStatistics = numpy.zeros((exposureMatrix.shape[0], 3))
        white = self.regParameters.getWhiteStdErrors()
        if white or rd.numConstraints > 999: # Temporary, replace 999 with 0 to enable
            if rd.numConstraints > 0:
                expMatrix = rd.regressorMatrix[:,:-len(rd.thinFactorIndices)-rd.numConstraints]
                constrMatrix = numpy.transpose(rd.regressorMatrix[:,-rd.numConstraints:])
            else:
                expMatrix = rd.regressorMatrix 
                constrMatrix = None
            regressionStats = regressANOVA.calc_regression_statistics(
                    factorReturns, numpy.transpose(expMatrix), constrMatrix, white=white)
        elif not allZeroFR:
            params = numpy.array(rlmResult.params)
            bse = rlmResult.bse
            regressionStats = numpy.zeros((params.shape[0],2))
            regressionStats[:,0] = bse
            bse = ma.masked_where(abs(bse)<1.0e-12, bse)
            regressionStats[:,1] = params/bse

            # Compute p-values for t-stats since rlm doesn't provide them
            pvals = [(1.0 - stats.t.cdf(abs(t), rd.regressorMatrix.shape[1] - \
                    len(rd.factorIndices))) for t in regressionStats[:,1]]
            regressionStats = numpy.concatenate([regressionStats, 
                    numpy.transpose(numpy.array(pvals)[numpy.newaxis])], axis=1)
        else:
            regressionStats = regressionStatistics
        
        # Put it all together (to account for factors omitted from regression)
        for (i,j) in enumerate(rd.factorIndices):
            regressionStatistics[j,:] = regressionStats[i,:]

        regressANOVA.store_regression_statistics(regressionStatistics[:,0],
                regressionStatistics[:,1], regressionStatistics[:,2])
        
        # Set up return structure
        regOutput = Utilities.Struct()
        regOutput.factorReturns = factorReturns
        regOutput.specificReturns = specificReturns
        regOutput.regressANOVA = regressANOVA
        regOutput.constraintWeight = constraintWeight
        regOutput.rlmDownWeights = rlmDownWeights
        regOutput.subIssues = rd.subIssues
        regOutput.fmpDict = fmpDict
        regOutput.constrComp = constrComp

        return regOutput

    def run_factor_regressions(self, rm, rcClass, date, excessReturns, expMatrix,
                             mainEstu, data, excludeIndices, modelDB, marketDB,
                               robustWeightList=None, applyRT=False, fmpRun=False):
        """Performs the dirty work of shuffling around arrays and
        such that is required for the nested regressions.  Also
        computes 'aggregate' regression statistics for the model, such
        as r-square, t-stats, etc.  Returns a tuple containing a
        map of factor names to factor return values, a specific returns
        array, a map of factor names to regression statistics, and
        a ReturnCalculator.RegressionANOVA object housing the results.
        """
        self.log.debug('run_factor_regressions: begin')
        self.log.info('Running old regression code')

        # Set up some data items to be used later
        allFactorNames = expMatrix.getFactorNames()
        factorReturnsMap = dict()
        regressStatsMap = dict()
        regressionReturns = ma.array(excessReturns)
        regressorMatrix = expMatrix.getMatrix()
        ANOVA_data = list()
        estuNameList = []
        robustWeightMap = dict()
        fmpMap = dict()
        ccMap = dict()
        ccXMap = dict()
        numRegs = len(rcClass.allParameters)
        #rcClass.date = date
        rcClass.modelDB = modelDB
        rcClass.marketDB = marketDB
        if robustWeightList is None: #fix for Mutable Default Argument
            robustWeightList = []
        if len(robustWeightList) == 0:
            rlmRun = True
        else:
            rlmRun = False

        returns_ESTU = ma.take(excessReturns, data.estimationUniverseIdx, axis=0)
        rtCap_ESTU = ma.take(data.marketCaps, data.estimationUniverseIdx, axis=0)
        skewness = stats.skew(returns_ESTU * ma.sqrt(rtCap_ESTU), axis=0)
        logging.info('Skewness of ESTU returns: %f', skewness)

        # Some models may require certain exposures to be set to zero
        if rm.zeroExposureNames != []:
            # Loop round the factors
            for (i, factor) in enumerate(rm.zeroExposureNames):
                # First determine the type of asset (e.g. Investment Trust)
                # for which we wish to set exposures to zero
                # If not defined, or set to None, all assets are included
                zeroExpAssetIdx = []
                if rm.zeroExposureTypes[i] != [] and rm.zeroExposureTypes[i] != None:
                    zeroExpFactorIdx = regressorMatrix[expMatrix.getFactorIndex(\
                            rm.zeroExposureTypes[i]), :]
                    zeroExpAssetIdx = numpy.flatnonzero(ma.getmaskarray(\
                            zeroExpFactorIdx)==0.0)
                # Now pick out the factor and set the relevant
                # exposures to zero
                idx = expMatrix.getFactorIndex(factor)
                if zeroExpAssetIdx == []:
                    regressorMatrix[idx,:] = 0.0
                    self.log.info('Zeroing all exposures for %s factor', factor)
                else:
                    factorExp = regressorMatrix[idx,:]
                    ma.put(factorExp, zeroExpAssetIdx, 0.0)
                    regressorMatrix[idx,:] = factorExp
                    self.log.info('Zeroing all %s exposures for %s factor',
                            rm.zeroExposureTypes[i], factor)

        # Create list of all factors going into any regression
        factorNameIdxMap = dict([(f,i) for (i,f) in enumerate(
                            allFactorNames) if i not in excludeIndices])
        droppedFactorNames = [f for (i,f) in enumerate(allFactorNames) if i in excludeIndices]

        # Identify possible dummy style factors
        dummyStyles = set()
        for i in expMatrix.getFactorIndices(ExposureMatrix.StyleFactor):
            if Utilities.is_binary_data(regressorMatrix[i,:]):
                vals = list(set(ma.filled(regressorMatrix[i,:], 0.0)))
                dummyStyles.add(allFactorNames[i])
                logging.info('Dummy style factor: %s, values: %s',
                        allFactorNames[i], vals)

        # Loop round however many regressions are required
        for (iReg, rp) in enumerate(self.allParameters):
             
            # Get specific regression paramters
            tfPar = rp.getThinFactorInformation()
            self.regParameters = rp
            regressionOrder = rp.getRegressionOrder()
            excludeList = rp.getExcludeFactors()
            estuName = rp.getEstuName()
            estuNameList.append(estuName)

            self.log.info('Beginning nested regression, loop %d, ESTU: %s',
                    iReg+1, estuName)
            self.log.info('Factors in loop: %s',
                        ', '.join([f.name for f in regressionOrder]))

            # Determine which factors will go into this regression loop
            regFactorNames = list()
            excludeFactorNames = set([f.name for f in excludeList \
                                    if isinstance(f, ModelFactor)])
            if len(excludeFactorNames) > 0:
                self.log.info('Excluding these factors: %s',
                            ', '.join([f for f in excludeFactorNames]))
            for obj in regressionOrder:
                try:
                    fList = expMatrix.getFactorNames(obj)
                except:
                    fList = [obj.name]
                if len(fList) == 0:
                    logging.info('No factors in model belonging to %s', obj.name)
                for f in fList:
                    if f in droppedFactorNames:
                        logging.info('Excluding %s from regression', f)
                    elif f not in factorNameIdxMap:
                        logging.warning('Factor %s not in available list', f)
                    elif f in excludeFactorNames:
                        logging.warning('Factor %s in exclusion list', f)
                    else:
                        regFactorNames.append(f)
            regFactorsIdx = [factorNameIdxMap[f] for f in regFactorNames]
            regMatrix = ma.take(regressorMatrix, regFactorsIdx, axis=0)

            # Get estimation universe and weights for this loop
            if (rm.estuMap is not None) and (estuName in rm.estuMap):
                reg_estu = list(rm.estuMap[estuName].assets)
                if estuName != 'main':
                    if 'eligible' in rm.estuMap:
                        reg_estu = set(reg_estu).intersection(set(rm.estuMap['eligible'].assets))
                        reg_estu = list(reg_estu)
                logging.info('Using %d assets from %s estimation universe', len(reg_estu), estuName)
            else:
                reg_estu = list(mainEstu)
                logging.info('Using main estimation universe')
            regEstuIdx = [data.assetIdxMap[sid] for sid in reg_estu]

            kappa = self.regParameters.getRlmKParameter()
            if (rm.estuMap is not None) and (estuName in rm.estuMap) \
                    and hasattr(rm.estuMap[estuName], 'weights'):
                reg_weights = rm.estuMap[estuName].weights
                wgtDict = dict(zip(rm.estuMap[estuName].assets, reg_weights))
                reg_weights = [wgtDict[sid] for sid in reg_estu]
                logging.info('Using pre-defined regression weights, kappa=%s', kappa)
            else:
                reg_weights = calc_Weights(
                        rm.rmg, modelDB, marketDB, date, data, reg_estu,
                        rm.numeraire.currency_id, clip=rp.getClipWeights())

                if rp.getRegWeights() == 'cap':
                    reg_weights = reg_weights * reg_weights
                    logging.info('Cap-weighted regression, clipped Wts=%s, kappa=%s',
                                rp.getClipWeights(), kappa)
                elif rp.getRegWeights() == 'equal':
                    reg_weights = 1000.0 * numpy.ones((len(reg_weights)), float)
                    logging.info('Using unweighted RLM regression, kappa=%s', kappa)

                elif rp.getRegWeights() == 'invSpecificVariance':
                    # This is to be run after the entire model has been run.
                    # Inverse specific risks are obtained from the full model run ''' 
                    rmi = rm.getRiskModelInstance(date, modelDB)
                    hasSR = True
                    if not rmi.has_risks:
                        olderRMIs = modelDB.getRiskModelInstances(rm.rms_id)
                        rmiDates = [rmi.date for rmi in olderRMIs if rmi.has_risks and \
                                rmi.date < date and rmi.date > (date-datetime.timedelta(30))]
                        if len(rmiDates) < 1:
                            hasSR = False
                            logging.warning('No specific risk for %s or earlier, using sqrt(cap) instead', str(date))
                            logging.info('Root-cap-weighted regression, clipped Wts=%s, kappa=%s',
                                    rp.getClipWeights(), kappa)
                        else:
                            rmiDates.sort()
                            rmi = [rmi for rmi in olderRMIs if rmi.date==rmiDates[-1]][0]
                            logging.warning('No specific risk for %s, using data from %s instead',
                                str(date), str(rmi.date))

                    if hasSR:
                        # Load model specific risks
                        reg_weights = modelDB.getSpecificRisks(rmi)
                        reg_weights = pandas.Series(reg_weights)
                        reg_weights = reg_weights[reg_weights.index.isin(reg_estu)] 
                        reg_weights = reg_weights.reindex(reg_estu)
                        reg_weights[reg_weights == 0.0] = numpy.nan
                        reg_weights = ma.array(reg_weights.values, mask = pandas.isnull(reg_weights.values))

                        # Form regression weights as inverse of specific variance
                        reg_weights = reg_weights * reg_weights
                        reg_weights = Utilities.clip_extrema(reg_weights, 0.025)
                        reg_weights = 1.0 / reg_weights 
                        logging.info('Using inverse of specific variance, kappa=%s', kappa)
                else:
                    logging.info('Root-cap-weighted regression, clipped Wts=%s, kappa=%s',
                                rp.getClipWeights(), kappa)

            # Downweight some countries if required
            assetIdxMap_ESTU = dict([(j,i) for (i,j) in enumerate(reg_estu)])
            for r in [r for r in rm.rmg if r.downWeight < 1.0]:
                if r.rmg_id in data.rmgAssetMap:
                    for sid in data.rmgAssetMap[r.rmg_id].intersection(reg_estu):
                        reg_weights[assetIdxMap_ESTU[sid]] *= r.downWeight

            # If ESTU assets have no returns, warn and skip
            checkReturns = ma.take(excessReturns, regEstuIdx, axis=0)
            checkReturns = ma.masked_where(abs(checkReturns) < 1e-12, checkReturns)
            badReturns = numpy.sum(ma.getmaskarray(checkReturns))
            if (badReturns >= 0.99 * len(regEstuIdx)) or (len(regFactorsIdx)  < 1):
                self.log.warning('No returns for nested regression loop %d ESTU, skipping',
                                iReg + 1)
                specificReturns = ma.array(regressionReturns, copy=True)
                for fName in regFactorNames:
                    factorReturnsMap[fName] = 0.0
                    regressStatsMap[fName] = Matrices.allMasked(4)
                continue

            # Deal with thin factors (industry, country, dummy styles)
            thinTestIdx = list()
            thinTestType = list()
            if rp.getThinFactorCorrection():
                factorTypesForThin = list()
                for (i,f) in enumerate(regFactorNames):
                    fType = expMatrix.getFactorType(f)
                    if fType in (ExposureMatrix.IndustryFactor,
                                 ExposureMatrix.CountryFactor,
                                 ExposureMatrix.StyleFactor):
                        thinTestIdx.append(i)
                        thinTestType.append(fType)
                factorTypesForThin = list(set(thinTestType))

            # Identify thin factors and compute dummy asset returns
            if len(thinTestIdx) > 0:
                tfPar.nonzeroExposuresIdx = []
                # Allow dummy assets to have nonzero style exposures
                for f in regressionOrder:
                    try:
                        fType = expMatrix.getFactorType(f.name)
                    except:
                        fType = f
                    if fType in (ExposureMatrix.StyleFactor, ExposureMatrix.InterceptFactor):
                        styleFactorsIdx = [i for i in range(len(regFactorsIdx)) if \
                            expMatrix.getFactorType(regFactorNames[i]) in
                            (ExposureMatrix.StyleFactor, ExposureMatrix.InterceptFactor)]
                        tfPar.nonzeroExposuresIdx = styleFactorsIdx
                        break

                # If this regression doesn't involve industry factors, use market return
                tfPar.factorTypes = thinTestType
                tfPar.factorIndices = thinTestIdx
                tfPar.factorNames = regFactorNames
                tfPar.rmgList = rm.rmg
                tfPar.data = data
                tfPar.applyRT = applyRT
                tfPar.dummyCls = rm.industryClassification
                if ExposureMatrix.IndustryFactor not in factorTypesForThin:
                    tfPar.dummyReturnType = 'market'
                 
                # If it involves multiple factors, including industries, use
                # market for non-industry factors, and classification-based
                # returns for industries
                elif tfPar.dummyReturnType != 'market' and len(factorTypesForThin) > 1:
                    tmpType = tfPar.dummyReturnType
                    tfPar.dummyReturnType = 'market'
                    self.computeDummyReturns(\
                            tfPar, regressionReturns, regEstuIdx, reg_weights, data.universe, date)
                    tfPar.dummyReturnType = tmpType
                self.computeDummyReturns(
                        tfPar, regressionReturns, regEstuIdx, reg_weights, data.universe, date)
            else:
                tfPar = None
            self.thinFactorParameters = tfPar

            # Finally, run the regression

            rm.iReg = iReg
            if rlmRun:
                ro = rcClass.calc_Factor_Specific_Returns(
                                rm, date, regEstuIdx, regressionReturns, regMatrix, regFactorNames,
                                reg_weights, expMatrix, rp.getFactorConstraints(), robustWeights=None)
                robustWeightList.append(ro.rlmDownWeights)
            else:
                ro = rcClass.calc_Factor_Specific_Returns(
                                rm, date, regEstuIdx, regressionReturns, regMatrix, regFactorNames,
                                reg_weights, expMatrix, rp.getFactorConstraints(), robustWeights=robustWeightList[iReg])

            if iReg==0 and rp.getCalcVIF():
                # Assuming that 1st round of regression is  done on styles, industries, market etc.,
                # and second round regression is done on Domestic China etc.,
                exposureMatrix = ma.filled(expMatrix.getMatrix(), 0.0)
                self.VIF = rcClass.calculateVIF(\
                        exposureMatrix, ro.regressANOVA.weights_, ro.regressANOVA.estU_, expMatrix.factorIdxMap_)

            # Pass residuals to next regression loop
            regressionReturns = ro.specificReturns

            # Keep record of factor returns and regression statistics
            if estuName == 'main':
                ANOVA_data.append(ro.regressANOVA)
            for j in range(len(regFactorsIdx)):
                fName = regFactorNames[j]
                factorReturnsMap[fName] = ro.factorReturns[j]
                if fName in ro.fmpDict:
                    fmpMap[fName] = dict(zip(ro.subIssues, ro.fmpDict[fName].tolist()))
                if ro.constrComp is not None:
                    if fName in ro.constrComp.ccDict:
                        ccMap[fName] = ro.constrComp.ccDict[fName]
                        ccXMap[fName] = ro.constrComp.ccXDict[fName]
                values = Matrices.allMasked(4)
                values[:3] = ro.regressANOVA.regressStats_[j,:]
                if fName in ro.constraintWeight:
                    values[3] = ro.constraintWeight[fName]
                regressStatsMap[fName] = values

            # Add robust weights to running total
            tmpWeights = ma.masked_where(ro.rlmDownWeights==1.0, ro.rlmDownWeights)
            tmpWeightsIdx = numpy.flatnonzero(ma.getmaskarray(tmpWeights)==0)
            tmpWeights = ma.take(tmpWeights, tmpWeightsIdx, axis=0)
            tmpSIDs = numpy.take(ro.subIssues, tmpWeightsIdx, axis=0)
            robustWeightMap[iReg] = dict(zip(tmpSIDs, tmpWeights))

        # Regression ANOVA: take average of all regressions using MAIN ESTU
        self.log.info('%d regression loops all use same ESTU, computing ANOVA', len(ANOVA_data))
        numFactors = sum([n.nvars_ for n in ANOVA_data])
        regWeights = numpy.average(numpy.array([n.weights_ for n in ANOVA_data]), axis=0)
        anova = RegressionANOVA(excessReturns, ro.specificReturns, numFactors, ANOVA_data[0].estU_, regWeights)

        self.log.debug('run_factor_regressions: end')
        retVal = Utilities.Struct()
        retVal.factorReturnsMap = factorReturnsMap
        retVal.specificReturns = ro.specificReturns
        retVal.regStatsMap = regressStatsMap
        retVal.anova = anova
        retVal.robustWeightMap = robustWeightMap
        retVal.fmpMap = fmpMap
        retVal.ccMap = ccMap
        retVal.ccXMap = ccXMap

        return retVal

class AsymptoticPrincipalComponentsLegacy:
    def __init__(self, numFactors, min_sigma=0.02, max_sigma=1.0, flexible=False, trimExtremeExposures=False):
        self.log = logging.getLogger('FactorReturns.PrincipalComponents')
        self.numFactors = numFactors
        self.trimExtremeExposures = trimExtremeExposures
        self.flexible = flexible
        self.min_sv = min_sigma**2.0
        self.max_sv = max_sigma**2.0

    def calc_ExposuresAndReturns_Inner(
                    self, returns, T, estu=None, originalReturns=None, useAPCA=True,
                    trimExtremeExposures=False):
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
 
        # Using SVD, more efficient than eigendecomp on huge matrix
        if useAPCA:
            try:
                (d, v) = linalg.svd(returnsESTU[:,-T:], full_matrices=False)[1:]
                d = d**2 / returnsESTU.shape[0]
            except:
                logging.warning('SVD routine failed... computing eigendecomposition instead')
                tr = numpy.dot(numpy.transpose(returnsESTU[:,-T:]), returnsESTU[:,-T:])
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
            expMatrixEstu = numpy.dot(u[:,:self.numFactors],\
                    numpy.diag(d[:self.numFactors]))
            # Compute factor returns
            factorReturns = Utilities.robustLinearSolver(\
                    returnsESTU, expMatrixEstu, computeStats=False).params

        # Back out exposures
        if originalReturns is None:
            originalReturns = returns
        if useAPCA:
            exposureMatrix = numpy.dot(ma.filled(originalReturns.data[:,-T:], 0.0), numpy.transpose(v))
            if T < returnsESTU.shape[1]:
                expMatrixEstu = ma.take(exposureMatrix, estu, axis=0)
                returnsESTU = ma.take(ma.filled(originalReturns.data, 0.0), estu, axis=0)
                fullFactorReturns = Utilities.robustLinearSolver(\
                        returnsESTU, expMatrixEstu).params
                factorReturns = numpy.array(fullFactorReturns, copy=True)
        else:
            exposureMatrix = Utilities.robustLinearSolver(\
                    numpy.transpose(ma.filled(originalReturns.data[:,-T:], 0.0)),
                    numpy.transpose(factorReturns[:,-T:]), computeStats=False).params
            exposureMatrix = numpy.transpose(exposureMatrix)

        exposureMatrix = Utilities.screen_data(exposureMatrix, fill=True)
        factorReturns = Utilities.screen_data(factorReturns, fill=True)
        if trimExtremeExposures:
            outlierClass = Outliers.Outliers()
            exposureMatrix = outlierClass.twodMAD(exposureMatrix, axis=0)

        # Finally, calculate specific returns
        specificReturns = originalReturns.data - \
                numpy.dot(exposureMatrix, factorReturns)

        # Calculate regression statistics
        regressANOVA = RegressionANOVA(originalReturns.data[:,-1],
                        specificReturns[:,-1], self.numFactors, estu)

        # Not sure if this really applies to PCA of TxT matrix...
        prc = numpy.sum(d[0:self.numFactors], axis=0) / numpy.sum(d, axis=0)
        self.log.info('%d factors explains %f%% of variance', self.numFactors, 100*prc)
        self.log.debug('calc_ExposuresAndReturns: end')
        return (exposureMatrix, factorReturns, specificReturns, regressANOVA, prc)

    def calc_ExposuresAndReturns(self, returns, estu=None, T=None):
        if T == None:
            T = returns.data.shape[1]

        # If no ESTU specified, use all assets
        if estu is None:
            estu = list(range(returns.data.shape[0]))
        if T > returns.data.shape[1]:
            T = returns.data.shape[1]

        APCA = True
        if len(estu) < 1:
            raise LookupError('Empty estimation universe')
        if T >= self.numFactors:
            if T >= len(estu):
                self.log.warning('Returns history (%d) > ESTU assets (%d), unreliable estimates',
                              T, len(estu))
                if self.flexible:
                    self.log.warning('Switching to vanilla PCA')
                    APCA = False
            self.log.info('Returns history: %d periods, by %d assets', T, len(estu))
        else:
            raise LookupError('Asymptotic PCA requires ESTU assets (%d) > returns history (%d) > factors (%d)' %\
                    (len(estu),   returns.data.shape[1], self.numFactors))

        if APCA:
            # First iteration: APCA to obtain specific returns
            logging.info('Computing APCA iteration one')
            (tmpExpM, tmpFacRet, tmpSpecRet, tmpRegANOVA, pct) = \
                                self.calc_ExposuresAndReturns_Inner(returns, T, estu,
                                        trimExtremeExposures=self.trimExtremeExposures)

            # Apply residual variance adjustment to returns
            logging.info('Computing APCA iteration two')
            adjReturns = Matrices.TimeSeriesMatrix(returns.assets, returns.dates)
            adjReturns.data = ma.array(
                    self.residualVarianceAdjustment(returns.data, tmpSpecRet[:,-T:]))

            # Second iteration: APCA on adjusted returns
            (exposureMatrix, factorReturns, specificReturns, regressANOVA, pct) = \
                                self.calc_ExposuresAndReturns_Inner(
                                        adjReturns, T, estu, originalReturns=returns,
                                        trimExtremeExposures=self.trimExtremeExposures)
        else:
            # Vanilla PCA 
            logging.info('Computing regular PCA')
            (exposureMatrix, factorReturns, specificReturns, regressANOVA, pct) = \
                                self.calc_ExposuresAndReturns_Inner(
                                        returns, T, estu, useAPCA=False)

        return (exposureMatrix, factorReturns, specificReturns, regressANOVA, pct)
        
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
        invSpecificRisk = 1.0 / (specificVars ** 0.5)
        adjReturns = ma.transpose(invSpecificRisk * ma.transpose(returns))
        self.log.debug('residualVarianceAdjustment: end')
        return adjReturns

class RegressionConstraint:
    """Represents a linear regression constraint in the form:
    weighted sum of a subset of factors equals some value.
    Implemented by introducing a dummy asset into the regression.
    """
    def __init__(self, factorType):
        self.log = logging.getLogger('FactorReturns.RegressionConstraint')
        self.factorType = factorType
        self.sumToValue = 0.0
        self.factorWeights = None

    def createDummyAsset(self, factorIndices, rm, rd, numDummies, expM, robustWeights=None):
        """Creates dummy asset by appending an entry to the regressor
        matrix and weight and return arrays.
        """
        self.log.info('Applying constraint to %s factors', self.factorType.name)
        # Add new column to regressor matrix
        expCol = numpy.zeros((rd.regressorMatrix.shape[0]))[numpy.newaxis]
        rd.regressorMatrix = ma.concatenate([rd.regressorMatrix, ma.transpose(expCol)], axis=1)
        
        if rd.useRealCaps:
            logging.info('Using actual 30-day mcaps for constraint')
            subIssues = numpy.take(expM.getAssets(), rd.indices_ESTU, axis=0)
            mcapDates = rd.modelDB.getAllRMGDateRange(rd.date, 30)
            cWeight = rd.modelDB.getAverageMarketCaps(
                    mcapDates, subIssues, rm.numeraire.currency_id, rd.marketDB, loadAllDates=True)
        else:
            cWeight = rd.weights_ESTU[:len(rd.weights_ESTU)-numDummies]**2
        if robustWeights is not None:
            cWeight = cWeight * robustWeights[:len(rd.weights_ESTU)-numDummies]

        # Compute weights on factors
        factorTotalCap = 0.0
        for (i, idx) in enumerate(factorIndices):
            indices = numpy.flatnonzero(rd.regressorMatrix[idx,:-(numDummies+1)])
            if len(indices) > 0:
                # NOTE: adding cap weights here but weight vector is root-cap
                factorMCap = ma.sum(ma.take(cWeight, indices, axis=0), axis=None)
                rd.regressorMatrix[idx,-1] = factorMCap
                factorTotalCap += factorMCap
        
        # Scale so exposures sum to one
        if factorTotalCap > 0.0:
            rd.regressorMatrix[:,-1] /= factorTotalCap
        self.factorWeights = rd.regressorMatrix[:,-1]
        
        # Assign return
        ret = self.computeConstrainedValue(
                rd.excessReturns_ESTU[:-numDummies], rd.weights_ESTU[:-numDummies])
        
        # Update return and weight arrays
        factorTotalCap = factorTotalCap / float(len(rd.indices_ESTU))
        rd.excessReturns_ESTU = ma.concatenate([rd.excessReturns_ESTU, ma.array([ret])], axis=0)
        rd.weights_ESTU = ma.concatenate([rd.weights_ESTU, ma.array([factorTotalCap])], axis=0)
        
        return rd

class ConstraintSumToZero(RegressionConstraint):
    def __init__(self, factorType):
        RegressionConstraint.__init__(self, factorType)
    def computeConstrainedValue(self, *args):
        self.sumToValue = 0.0
        return self.sumToValue

class ConstraintSumToMarket(RegressionConstraint):
    def __init__(self, factorType):
        RegressionConstraint.__init__(self, factorType)
    def computeConstrainedValue(self, excessReturns, weights):
        self.sumToValue = ma.average(excessReturns, weights=weights)
        return self.sumToValue
