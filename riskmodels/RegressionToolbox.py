import logging
import numpy
import numpy.ma as ma
from riskmodels import Matrices
from riskmodels import LegacyUtilities as Utilities

class RegressionParameters:
    """
    """
    def __init__(self, paramsDict):
        self.regressionParameters = paramsDict
    
    def getDummyReturns(self):
        return self.regressionParameters.get('dummyReturns', DummyMarketReturns())
    
    def getDummyWeights(self):
        return self.regressionParameters.get('dummyWeights', AxiomaDummyWeights(6.0))
    
    def getFactorConstraints(self):
        return self.regressionParameters.get('factorConstraints', list())
    
    def getRegressionOrder(self):
        return self.regressionParameters.get('regressionOrder', None)
    
    def getThinFactorCorrection(self):
        return self.regressionParameters.get('fixThinFactors', True)

    def getCalcVIF(self):
        return self.regressionParameters.get('calcVIF', False)
    
    def getWhiteStdErrors(self):
        return self.regressionParameters.get('whiteStdErrors', False)

    def getRlmKParameter(self):
        return self.regressionParameters.get('k_rlm', 1.345)

    def useWeightedRLM(self):
        return self.regressionParameters.get('weightedRLM', True)

class DummyAssetWeights:
    def __init__(self, minAssets=6.0):
        self.log = logging.getLogger('RegressionToolbox.DummyAssetWeights')
        self.minAssets = minAssets

class AxiomaDummyWeights(DummyAssetWeights):
    """New weight function, increases the weight of dummies
    with Herfindahl scores very near the threshold.
    """
    def __init__(self, minAssets):
        DummyAssetWeights.__init__(self, minAssets)
    def computeWeight(self, score, factorWeights):
        partial1 = score**4.0
        partial2 = self.minAssets**4.0
        weight = (self.minAssets - 1.0) * (partial1 - partial2) / (1.0 - partial2)
        weight *= ma.sum(factorWeights, axis=0) / score
        return weight

class BarraDummyWeights(DummyAssetWeights):
    """Weight function used by most Barra models.
    """
    def __init__(self, minAssets):
        DummyAssetWeights.__init__(self, minAssets)
    def computeWeight(self, score, factorWeights):
        weight = (self.minAssets / score - 1.0) * ma.sum(factorWeights, axis=0)
        return weight

class DummyAssetReturns:
    def __init__(self):
        self.log = logging.getLogger('RegressionToolbox.DummyMarketReturns')
        self.factorIndices = None
        self.factorNames = None
        self.dummyRetWeights = dict()

class DummyMarketReturns(DummyAssetReturns):
    """Assigns the market excess return to all dummy assets.
    Safe, easy, and appropriate for most cases.  Since we
    are unable to accurately determine an industry's performance,
    assume it's behaving just like the market.
    """
    def __init__(self):
        DummyAssetReturns.__init__(self)
    def computeReturns(self, excessReturns, estu, weights_ESTU, universe, date):
        self.log.info('Assigning market return to dummy assets')
        values = Matrices.allMasked(len(self.factorNames))
        excessReturns_ESTU = ma.take(excessReturns, estu)
        # Remove any really large returns
        (excessReturns_ESTU, mad_bounds) = Utilities.mad_dataset(
                excessReturns_ESTU, -15, 15, axis=0)
        ret = ma.average(excessReturns_ESTU, weights=weights_ESTU)
        sclWt = ma.filled(weights_ESTU / ma.sum(weights_ESTU, axis=None), 0.0)
        for i in self.factorIndices:
            values[i] = ret
            self.dummyRetWeights[self.factorNames[i]] = sclWt
        return values

class DummyClsParentReturns(DummyAssetReturns):
    """Assigns the return corresponding to the dummy's parent
    classification. (eg. SuperSectors in the case of ICB Sectors)
    Applies only to industry factor dummies.
    """
    def __init__(self, cls, parentName, modelDB, parentLevel=-1):
        DummyAssetReturns.__init__(self)
        self.clsSchemeDict = cls
        self.clsParentLevel = parentLevel
        self.clsParentName = parentName
        self.modelDB = modelDB
    
    def setClassification(self, date):
        chngDates = sorted(d for d in self.clsSchemeDict.keys() if d <= date)
        self.classification = self.clsSchemeDict[chngDates[-1]]
        self.log.info('Using %s classification scheme', self.classification.name)
        self.parents = self.classification.getClassificationParents(
            self.clsParentName, self.modelDB)
        self.parentNames = [i.description for i in self.parents]
        self.childrenMap = {}
        for parent in self.parents:
            children = self.classification.getClassificationChildren(
                parent, self.modelDB)
            self.childrenMap[parent] = children

    def computeReturns(self, excessReturns, estu, weights, universe, date):
        self.setClassification(date)
        self.log.info('Assigning %s %s returns to dummy assets',
                     self.classification.name, self.clsParentName)
        values = Matrices.allMasked(len(self.factorNames))
        ids_ESTU = [universe[n] for n in estu]
        factorIdxMap = dict([(self.factorNames[i], i) for i
                             in self.factorIndices])
        returns_ESTU = ma.take(excessReturns, estu, axis=0)
        parentExpM = self.classification.getExposures(
            date, ids_ESTU, self.parentNames, self.modelDB,
            self.clsParentLevel).filled(0.0)
        # Remove any really large returns
        (returns_ESTU, mad_bounds) = Utilities.mad_dataset(
                returns_ESTU, -25, 25, axis=0)
        for (pIdx, parent) in enumerate(self.parents):
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
            for child in self.childrenMap[parent]:
                idx = factorIdxMap[child.description]
                if idx in self.factorIndices:
                    values[idx] = parentReturn
                    self.dummyRetWeights[self.factorNames[idx]] = sclWgts
        return values 

class DummyAssetHandler:
    def __init__(self, idxToCheck, factorNames, regParams):
        self.log = logging.getLogger('RegressionToolbox.DummyAssetHandler')
        self.idxToCheck = idxToCheck
        self.factorNames = factorNames
        self.parameters = regParams
        self.dummyReturns = Matrices.allMasked(len(factorNames))
        self.dummyRetWeights = dict()
    
    def setDummyReturns(self, values):
        """values should be a masked array containing return values
        in all unmasked positions, such as one returned by calling
        computeReturns() on a DummyAssetReturn object.
        """
        assert(len(values)==len(self.dummyReturns))
        validIdx = numpy.flatnonzero(ma.getmaskarray(values)==0)
        for i in validIdx:
            self.dummyReturns[i] = values[i]
    
    def setDummyReturnWeights(self, retWgtDict):
        self.dummyRetWeights.update(retWgtDict)

    def insertDummyAssets(self, regressorMatrix, weights, excessReturns, factorReturns, 
            dummyRetWeights=None):
        """Examines the factors in positions (idxToCheck) for thinness.
        Inserts dummy assets into the regressor matrix where necessary, 
        returns a new copy of the regressor matrix, returns, and weights,
        as well as an array of factor returns and lists of thin and empty
        factor positions.
        """
        thinFactorIndices = []
        emptyFactorIndices = []
        
        # Prepare dummy weight calculation
        dw = self.parameters.getDummyWeights()
        minEffectiveAssets = dw.minAssets
        
        # Make sure dummy returns are present
        missing = len(numpy.flatnonzero(
                            ma.getmaskarray(self.dummyReturns)))
        if len(self.idxToCheck) > 0 and missing==len(self.dummyReturns):
            raise Exception('Returns have not been specified for dummy assets!')
        
        # Check specified factors for thin-ness
        for idx in self.idxToCheck:
            assetsIdx = numpy.flatnonzero(regressorMatrix[idx,:])
            dummyRet = self.dummyReturns[idx]
            factorName = self.factorNames[idx]
            if len(assetsIdx) == 0:
                # Empty factor, keep track of these and omit from reg later
                self.log.warning('Empty factor: %s, ret %f', self.factorNames[idx], dummyRet)
                factorReturns[idx] = dummyRet
                emptyFactorIndices.append(idx)
            else:
                # Herfindahl
                factorWeights = ma.take(weights, assetsIdx, axis=0) * \
                        ma.take(regressorMatrix[idx], assetsIdx, axis=0)
                factorWeights = abs(factorWeights)
                wgt = ma.sum(factorWeights, axis=0)
                if wgt <= 0.0:
                    # Empty factor, keep track of these and omit from reg later
                    self.log.warning('Empty factor: %s, ret %f', self.factorNames[idx], dummyRet)
                    factorReturns[idx] = dummyRet
                    emptyFactorIndices.append(idx)
                else:
                    wgt = factorWeights / wgt
                    score = 1.0 / ma.inner(wgt, wgt)
                
                    # This factor is thin...
                    if score < minEffectiveAssets:
                        dummyWgt = dw.computeWeight(score, factorWeights)
                        dummyExp = numpy.zeros((regressorMatrix.shape[0], 1))
                        dummyExp[idx] = 1.0
                        # Assign style factor exposures to dummy, if applicable
                        for i in self.nonzeroExposuresIdx:
                            if Utilities.is_binary_data(regressorMatrix[i,:]):
                                val = ma.median(regressorMatrix[i,:], axis=0)
                            else:
                                val = ma.inner(weights, regressorMatrix[i,:])
                                val /= ma.sum(weights)
                            dummyExp[i] = val
                    
                        # Append dummy exposures, weight, and return to estu
                        regressorMatrix = numpy.concatenate(
                                        [regressorMatrix, dummyExp], axis=1)
                        excessReturns = ma.concatenate(
                                [excessReturns, ma.array([dummyRet])], axis=0)
                        weights = ma.concatenate(
                                [weights, ma.array([dummyWgt])], axis=0)
                        self.log.info('Thin factor: %s (N: %.2f/%d, Dummy wgt: %.3f, ret: %.3f)',
                                self.factorNames[idx], score, len(assetsIdx), dummyWgt, dummyRet)
                        thinFactorIndices.append(idx)
                        if dummyRetWeights is not None:
                            if len(dummyRetWeights) > 0:
                                dummyRetWeights = ma.concatenate(
                                        [dummyRetWeights, numpy.array(self.dummyRetWeights[factorName], float)[numpy.newaxis,:]], axis=0)
                            else:
                                dummyRetWeights = numpy.array(self.dummyRetWeights[factorName], float)[numpy.newaxis,:]
        if dummyRetWeights is not None:
            return (regressorMatrix, excessReturns, weights,
                    factorReturns, thinFactorIndices, emptyFactorIndices, dummyRetWeights)
        return (regressorMatrix, excessReturns, weights,
                factorReturns, thinFactorIndices, emptyFactorIndices)

class RegressionConstraint:
    """Represents a linear regression constraint in the form:
    weighted sum of a subset of factors equals some value.
    Implemented by introducing a dummy asset into the regression.
    """
    def __init__(self, factorType):
        self.log = logging.getLogger('RegressionToolbox.RegressionConstraint')
        self.factorType = factorType
        self.sumToValue = 0.0
        self.factorWeights = None
    
    def createDummyAsset(self, factorIndices, excessReturns, 
                         regressorMatrix, weights, numDummies,
                         removeNTDsFromRegression=True):
        """Creates dummy asset by appending an entry to the regressor
        matrix and weight and return arrays.
        """
        self.log.info('Applying constraint to %s factors', self.factorType.name)
        
        # Add new column to regressor matrix
        expCol = numpy.zeros((regressorMatrix.shape[0]))[numpy.newaxis]
        regressorMatrix = ma.concatenate([regressorMatrix, ma.transpose(expCol)], axis=1)
        
        # Compute weights on factors
        factorTotalCap = 0.0
        for i in factorIndices:
            indices = numpy.flatnonzero(regressorMatrix[i,:-(numDummies+1)])
            if len(indices) > 0:
                # NOTE: adding cap weights here but weight vector is root-cap
                factorMCap = ma.sum(ma.take(weights[:len(weights)-numDummies]**2, 
                                    indices, axis=0))
                regressorMatrix[i,-1] = factorMCap
                factorTotalCap += factorMCap
        
        # Scale so exposures sum to one
        if factorTotalCap > 0.0:
            regressorMatrix[:,-1] /= factorTotalCap
        self.factorWeights = regressorMatrix[:,-1]
        
        # Assign return
        ret = self.computeConstrainedValue(
                        excessReturns[:-numDummies], weights[:-numDummies])
        
        # Update return and weight arrays
        excessReturns = ma.concatenate([excessReturns, ma.array([ret])], axis=0)
        if not removeNTDsFromRegression:
            factorTotalCap = factorTotalCap / float(len(excessReturns))
        weights = ma.concatenate([weights, ma.array([factorTotalCap])], axis=0)
        
        return (regressorMatrix, excessReturns, weights)

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
