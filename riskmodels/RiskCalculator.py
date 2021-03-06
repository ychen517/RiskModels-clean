
from collections import defaultdict
import datetime
import numpy.ma as ma
import numpy.linalg as linalg
import numpy
import logging
from riskmodels import LegacyUtilities as Utilities
from riskmodels import Matrices
from riskmodels.Matrices import ExposureMatrix
import riskmodels.Outliers

class RiskParameters:
    """Stores parameters (half-life, etc.) for all risk-related
    computation procedures.
    Instantiate using 2 dictionaries containing the parameter
    names and values for covariance matrix and specific risk
    computation.
    """
    def __init__(self, covParamsDict, srParamsDict):
        self.log = logging.getLogger('RiskCalculator.RiskParameters')
        self.covarianceParameters = covParamsDict
        self.specificRiskParameters = srParamsDict

    def getCovarianceHalfLives(self):
        return (self.covarianceParameters.get('varHalfLife', 125),
                self.covarianceParameters.get('corrHalfLife', 125))

    def getCovarianceSampleSize(self):
        hl = self.getCovarianceHalfLives()[1]
        return (self.covarianceParameters.get('minOmegaObs', hl),
                self.covarianceParameters.get('maxOmegaObs', hl*2))

    def getCovarianceNeweyWestLag(self):
        return (self.covarianceParameters.get('useNWAutoLag', False),
                self.covarianceParameters.get('useAxiomaAutoLag', False),
                self.covarianceParameters.get('preWhiten', False),
                self.covarianceParameters.get('omegaLag', 0),
                self.covarianceParameters.get('autoLagWeights', None),
                self.covarianceParameters.get('autoLagHistory', None))

    def getCovarianceClipping(self):
        return (self.covarianceParameters.get('omegaClipMethod', 'median'),
                self.covarianceParameters.get('omegaClipHistory', None),
                self.covarianceParameters.get('omegaClipBounds', None))

    def getCovarianceDVAOptions(self):
        return self.covarianceParameters.get('omegaDVAWindow', None)

    def getCovarianceComposition(self):
        return (self.covarianceParameters.get('useTransformMatrix', False),
                self.covarianceParameters.get('offDiagScaleFactor', 1.0))

    def useDeMeanedCovariance(self):
        return self.covarianceParameters.get('omegaDeMeanFlag', False)

    def useEqualWeightedCovariance(self):
        return self.covarianceParameters.get('omegaEqualWeightFlag', False)

    def getSpecificRiskHalfLife(self):
        return self.specificRiskParameters.get('deltaHalfLife', 125)

    def getSpecificRiskSampleSize(self):
        hl = self.getSpecificRiskHalfLife()
        return (self.specificRiskParameters.get('minDeltaObs', hl),
                self.specificRiskParameters.get('maxDeltaObs', hl*2))

    def getSpecificRiskNeweyWestLag(self):
        return self.specificRiskParameters.get('deltaLag', 0)

    def getSpecificRiskClipping(self):
        return (self.specificRiskParameters.get('deltaClipMethod', 'median'),
                self.specificRiskParameters.get('deltaClipHistory', None),
                self.specificRiskParameters.get('deltaClipBounds', None))

    def getSpecificRiskDVAOptions(self):
        return self.specificRiskParameters.get('deltaDVAWindow', None)

    def useDeMeanedSpecificRisk(self):
        return self.specificRiskParameters.get('deltaDeMeanFlag', False)

    def useEqualWeightedSpecificRisk(self):
        return self.specificRiskParameters.get('deltaEqualWeightFlag', False)

class StandardCovarianceMatrix:
    """Computes factor variances and covariances using a 
    user-specified weighting scheme.  Supports different 
    half-lives for factor variances and correlations, and 
    Newey-West autocorrelation adjustment.
    """
    def __init__(self, covParameters):
        self.log = logging.getLogger('RiskCalculator.StandardCovarianceMatrix')
        # Basic parameters
        (self.varHalfLife, self.corrHalfLife) = \
                           covParameters.getCovarianceHalfLives()
        (self.minOmegaObs, self.maxOmegaObs) = \
                           covParameters.getCovarianceSampleSize()
        self.deMeanFlag = covParameters.useDeMeanedCovariance()
        self.equalWeightFlag = covParameters.useEqualWeightedCovariance()
        self.vanilla = False

        # Enhancements: autocorrelation lag
        (self.nwAutoLag, self.axAutoLag, self.preWhiten, 
         self.omegaLag, self.autoLagWeights, self.autoLagHistory) = \
                              covParameters.getCovarianceNeweyWestLag()
        if self.nwAutoLag and self.axAutoLag:
            raise Exception('Unable to use both Newey-West and Axioma auto-lag together')

        # Enhancements: factor returns clipping
        (self.useMedian, self.clipSampleObs, bounds) = \
                         covParameters.getCovarianceClipping()
        if self.clipSampleObs is not None and bounds is None:
            raise Exception('MAD-clipping enabled but no clipping bounds given')
        if bounds is not None:
            (self.clipLowerBound, self.clipUpperBound) = bounds

        # Enhancements: Dynamic Volatility Adjustment (DVA)
        self.DVAWindow = covParameters.getCovarianceDVAOptions()

        if self.clipSampleObs is not None and \
                    self.DVAWindow is not None:
            self.log.warning('Both clipping and DVA enabled, clipping will be disabled')
            self.clipSampleObs = None

    def toggleAdvancedOptions(self):
        """Switches self.vanilla to True/False.  When True, over-rides and 
        disables all the bells and whistles (eg. autolag, DVA/clipping, etc.)
        Also resets (temporarily) the Newey-West lag parameter to 1.0
        """
        self.vanilla = not self.vanilla
        if self.vanilla:
            self.log.info('Disabling advanced RiskCalculator controls')
        else:
            self.log.info('Enabling advanced RiskCalculator controls')

    def cov_main(self, fr):
        """Core routine for computing a factor covariance matrix.
        """
        # Quick sanity check on the data
        if fr.shape[1] < self.minOmegaObs:
            raise LookupError('Number of time periods, %d, is less than required min number of observations, %d' % (fr.shape[1], self.minOmegaObs))
            
        if fr.shape[1] > self.maxOmegaObs:
            raise LookupError('Number of time periods, %d, is greater than max number of observations, %d' % (fr.shape[1], self.maxOmegaObs))

        nPeriods = fr.shape[1]
        numLags = self.omegaLag
        self.log.info('Using %d factors by %d periods', fr.shape[0], nPeriods)

        # Calculate weights
        varWeights = Utilities.computeExponentialWeights(
                self.varHalfLife, nPeriods, self.equalWeightFlag)
        corrWeights = Utilities.computeExponentialWeights(
                self.corrHalfLife, nPeriods, self.equalWeightFlag)

        # Pre-process (clip, DVA, etc) factor returns
        frFull = fr
        if self.clipSampleObs is not None and not self.vanilla:
            self.log.debug('MAD-clipping factor feturns')
            (fr_mad, bounds) = Utilities.mad_dataset(
                    fr, self.clipLowerBound, self.clipUpperBound,
                    restrict=list(range(self.clipSampleObs)), axis=1, method=self.useMedian)
            frFull = fr_mad
        elif self.DVAWindow is not None and not self.vanilla:
            self.log.debug('Applying Dynamic Volatility Adjustment (DVA)')
            frFull = Utilities.dynamic_volatility_adjustment(frFull, self.DVAWindow)

        # Check for factor returns consisting entirely of zeros
        frFullMasked = ma.masked_where(abs(frFull*corrWeights) < 1e-10, frFull)
        frNonZeros = (ma.getmaskarray(frFullMasked)==0.0)
        sumFactors = ma.sum(frNonZeros, axis=1).astype(numpy.float) / fr.shape[1]
        sumFactors = ma.where(sumFactors < 0.01, 0.0, sumFactors)
        goodFactorsIdx = numpy.flatnonzero(sumFactors)

        frData = ma.take(frFull, goodFactorsIdx, axis=0).filled(0.0)
        if len(goodFactorsIdx) != frFull.shape[0]:
            self.log.info('%d out of %d factors are effectively zero',
                    frFull.shape[0]-len(goodFactorsIdx), frFull.shape[0])
            self.log.info('Dimension of cov matrix reduced to %d for calculation',
                    len(goodFactorsIdx))

        # If Newey-West auto-lag selection enabled
        if self.nwAutoLag and not self.vanilla:
            if not self.autoLagHistory:
                self.autoLagHistory = fr.shape[1]
            if self.autoLagWeights:
                autoLagWeights = corrWeights
            else:
                autoLagWeights = None
            (tmpData, auto_lag, transformMatrix) = \
                    self.computeAutoLag(frData, autoLagWeights)
            numLags = auto_lag
            self.log.info('Number of autocorrelation lags (Newey-West): %d', numLags)
            # Uncomment the following to enable 'full' pre-whitening
#            if self.preWhiten:
#                frData = tmpData
#                varWeights = varWeights[:-1]
#                corrWeights = corrWeights[:-1]

        # Calculate sample variance, pluck out diagonals
        factorVar = Utilities.compute_covariance(frData, 
            weights=varWeights, deMean=self.deMeanFlag, axis=1, varsOnly=True)

        # Extract correlations matrix
        covMatrix = Utilities.compute_covariance(frData,
                    weights=corrWeights, deMean=self.deMeanFlag, axis=1)
        (d, corrMatrix) = Utilities.cov2corr(covMatrix, fill=True)
        
        # Sandwich everything together
        vols = ma.masked_where(factorVar <= 0.0, factorVar)
        vols = ma.sqrt(vols)
        factorCov = ma.transpose(ma.transpose(corrMatrix * vols) * vols)
        factorCov = factorCov.filled(0.0)

        # Our in-house auto-lag algorithm, alternative to Newey-West
        if self.axAutoLag and not self.vanilla:
            (eigval, eigvec) = linalg.eigh(factorCov)
            normLag0 = numpy.sqrt(numpy.sum(eigval * eigval))
            SNRatio = []
            for lag in range(1,10):
                lagCov = Utilities.compute_covariance(frData,
                        weights=corrWeights, deMean=self.deMeanFlag, axis=1, lag=lag)
                # Compute signal to noise ratio
                (eigval, eigvec) = linalg.eigh(lagCov)
                normLagN = numpy.sqrt(numpy.sum(eigval * eigval))
                SNRatio.append(normLagN / normLag0)
            SNRatio = numpy.array(SNRatio)
            medianSNR = ma.median(SNRatio, axis=0)
            okSNRatio = numpy.flatnonzero(ma.masked_where(SNRatio>medianSNR, SNRatio))
            numLags = okSNRatio[0]
            self.log.info('Number of autocorrelation lags (Axioma): %d', numLags)
        elif not self.nwAutoLag:
            if self.vanilla:
                numLags = 1
            self.log.info('Number of autocorrelation lags (Static): %d', numLags)

        # Perform Newey-West using given number of lags
        ARadj = numpy.zeros(factorCov.shape, float)
        for lag in range(1,numLags+1):
            lagCov = Utilities.compute_covariance(frData,
                    weights=corrWeights, deMean=self.deMeanFlag, axis=1, lag=lag)
            ARadj += (1.0 - float(lag)/(float(numLags)+1.0)) * \
                    (lagCov + numpy.transpose(lagCov))

        ARadj = ARadj / ma.outer(d,d)
        ARadj = ma.transpose(ma.transpose(ARadj * vols) * vols)
        factorCov += ARadj.filled(0.0)

        # Uncomment the following to enable 'full' pre-whitening
#        if self.preWhiten and self.nwAutoLag and not self.vanilla:
#            factorCov = numpy.dot(transformMatrix, numpy.dot( \
#                    factorCov, numpy.transpose(transformMatrix)))

        # In case python's numerical accuracy falls short
        factorCov = (factorCov + numpy.transpose(factorCov)) / 2.0

        # If we've computed a reduced covariance matrix, paste it
        # back into the full-sized variant
        if frData.shape[0] < fr.shape[0]:
            fullFactorCov = numpy.zeros((fr.shape[0], fr.shape[0]), float)
            for (i, id) in enumerate(goodFactorsIdx):
                for (j, jd) in enumerate(goodFactorsIdx):
                    fullFactorCov[id,jd] = factorCov[i,j]
            return fullFactorCov
        else:
            return factorCov

    def computeFactorCovarianceMatrix(self, frMatrix):
        """Compute factor covariance matrix.  Input is a 
        TimeSeriesMatrix of factor returns.  The first column 
        of observations should be the most recent.
        """
        self.log.debug('computeFactorCovarianceMatrix: begin')
        
        # Call core covariance matrix routine
        factorCov = self.cov_main(frMatrix.data)
        
        # Clip small eigenvalues and reconstitute the cov matrix if necessary
        factorCov = Utilities.forcePositiveSemiDefiniteMatrix(factorCov,
                    min_eigenvalue=0.0)

        # Annualize
        factorCov *= 252.0
        
        self.log.info('Sum of cov matrix elements: %f', ma.sum(factorCov, axis=None))
        self.log.debug('computeFactorCovarianceMatrix: end')
        return factorCov

    def computeAutoLag(self, dataMatrix, weights=None):
        """Routine that calculates the number of lags used
        for Newey-West autocorrelation correction based
        on the data properties. This is a first draft
        and will probably be subject to a degree of change.
        """

        logging.debug('computeAutoLag: begin')
        k = dataMatrix.shape[0]
        if not self.autoLagHistory:
            t = dataMatrix.shape[1]
        else:
            t = min(self.autoLagHistory, dataMatrix.shape[1])
        transformMatrix = 1.0

        # Weight the factor returns if required
        if weights is None:
            weights = numpy.ones((t), float)
        weightedData = dataMatrix[:,:t] * weights[:t]

        # Calculate bound of truncated series
        n = int(numpy.ceil(4.0 * (t / 100.0)**(2.0/9.0)))
        n = max(n, 6)
        n = min(n, 15)

        # Pre-whiten the factor returns if required
        if self.preWhiten != False:
            # Take the first lag 
            laggedData1 = weightedData[:, 1:]
            laggedData = weightedData[:, :-1]
            # Calculate regressor matrix
            a_1 = numpy.dot(laggedData, numpy.transpose(laggedData1))
            a_2 = numpy.dot(laggedData1, numpy.transpose(laggedData1))
            A = linalg.solve(numpy.transpose(a_2), numpy.transpose(a_1))
            A = numpy.transpose(A)
            # Create matrix for un-whitening (dirtying?)
            transformMatrix = linalg.inv(numpy.eye(k) - A)
            # Create pre-whitened factor returns for output
            dataMatrix = dataMatrix[:, :-1] - numpy.dot(A, dataMatrix[:, 1:])
            weightedData = dataMatrix * weights[:-1]

        t_red = min(t, weightedData.shape[1])
        weightedData = weightedData[:, :t_red]

        sigma = numpy.zeros(n+2, float)
        jSigma = numpy.zeros(n+2, float)
        gamma = numpy.zeros(n+2, float)

        for j in range(n+2):
            laggedData1 = weightedData[:, j:]
            laggedData = weightedData[:, :t_red-j]
            crossProd = numpy.dot(laggedData, numpy.transpose(laggedData1))
            sigma[j] = sum(sum(crossProd))
            jSigma[j] = j * sigma[j]
            s_1 = 2.0 * sum(jSigma)
            s_0 = sigma[0] + 2.0 * sum(sigma[1:])
            gamma[j] = 1.1447 * (s_1*s_1/(s_0*s_0))**(1.0/3.0)

        gamma = numpy.average(gamma[-3:])
        lag = int(numpy.ceil(gamma * (t**(1.0/3.0))))
        lag = min(lag, 20)

        logging.info('Lag parameters (n: %d, gamma: %f, m: %d)', n, gamma, lag)
        logging.debug('computeAutoLag: end')
        return (dataMatrix, lag, transformMatrix)

class BlockDiagonalCovarianceMatrix(StandardCovarianceMatrix):
    """Block-diagonal covariance matrix estimator.
    """
    def __init__(self, covParameters):
        self.log = logging.getLogger('RiskCalculator.BlockDiagonalCovarianceMatrix')
        StandardCovarianceMatrix.__init__(self, covParameters)

    def computeFactorCovarianceMatrix(self, *frMatrix):
        """Each input arg is a TimeSeriesMatrix of factor returns
        corresponding to each block of the full covariance matrix.
        Minimal flexibility for now: assumes all blocks use
        same half-life, min/max obs, etc.
        """
        self.log.debug('computeFactorCovarianceMatrix: begin')
        nFactors = sum([f.data.shape[0] for f in frMatrix])
        fullFactorCov = numpy.zeros((nFactors, nFactors), float)
        idx = 0
        
        for fm in frMatrix:
            factorCov = self.cov_main(fm.data)

            # Insert block into full covariance matrix
            dim = fr.shape[0]
            fullFactorCov[idx:idx+dim,idx:idx+dim] = factorCov
            idx = dim

        # Clip small eigenvalues and reconstitute the cov matrix if necessary
        fullFactorCov = Utilities.forcePositiveSemiDefiniteMatrix(fullFactorCov)

        # Annualize
        fullFactorCov *= 252.0
        
        self.log.debug('computeFactorCovarianceMatrix: end')
        return fullFactorCov

class StambaughCovarianceMatrix(StandardCovarianceMatrix):
    """Covariance matrix estimator based on Stambaugh (1996).
    Allows for two sets of factor returns, each with a different
    history length.  Can be used, for example, with hybrid 
    factor models where the stat factors are likely to have
    a shorter history.
    """
    def __init__(self, covParameters):
        self.log = logging.getLogger('RiskCalculator.StambaughCovarianceMatrix')
        StandardCovarianceMatrix.__init__(self, covParameters)

    def computeFactorCovarianceMatrix(self, frMatrixLong, frMatrixShort):
        """Inputs are two TimeSeriesMatrix objects; the
        second should be the one containing the shorter
        history factor returns.
        """
        self.log.debug('computeFactorCovarianceMatrix: begin')
        if frMatrixShort.data.shape[1] < frMatrixLong.data.shape[0]:
            raise LookupError('Shorter history (%d) needs to be >= factors in long history (%d)' % (frMatrixShort.data.shape[1], frMatrixLong.data.shape[0]))

        longFactors = frMatrixLong.data.shape[0]
        shortFactors = frMatrixShort.data.shape[0]
        nFactors = longFactors + shortFactors
        fullFactorCov = numpy.zeros((nFactors, nFactors), float)

        shortPeriods = frMatrixShort.data.shape[1]

        # Regression of shorter-history factor returns on those with longer history
        frMatrixLong_clipped = ma.take(frMatrixLong.data, list(range(shortPeriods)), axis=1)
        regMatrix = numpy.ones((shortPeriods, longFactors+1), float)
        regMatrix[:,1:] = numpy.transpose(frMatrixLong_clipped.filled(0.0))
        (betas, resid) = Utilities.ordinaryLeastSquares(
                numpy.transpose(frMatrixShort.data.filled(0.0)), regMatrix)
        residCov = numpy.dot(numpy.transpose(resid), resid)
        coef = betas[1:,:]

        # Compute covariance matrix of longer history factors
        covLong_full = self.computeSimpleCov(frMatrixLong.data)

        # Piece everything together
        fullFactorCov[:longFactors,:longFactors] = covLong_full
        off_diag = numpy.dot(covLong_full, coef)
        fullFactorCov[:longFactors,longFactors:] = off_diag
        fullFactorCov[longFactors:,longFactors:] = residCov + \
                numpy.dot(numpy.transpose(coef), off_diag)
        fullFactorCov[longFactors:,:longFactors] = numpy.transpose(off_diag)

        # Clip small eigenvalues and reconstitute the cov matrix if necessary
        fullFactorCov = Utilities.forcePositiveSemiDefiniteMatrix(fullFactorCov)

        # Annualize
        fullFactorCov *= 252.0

        self.log.debug('computeFactorCovarianceMatrix: end')
        return fullFactorCov

class CompositeCovarianceMatrix(StandardCovarianceMatrix):
    """Composite covariance matrix.
    """
    def __init__(self, covParameters):
        self.log = logging.getLogger('RiskCalculator.CompositeCovarianceMatrix')
        StandardCovarianceMatrix.__init__(self, covParameters)
        self.subCovariances = dict()

        # Enhancements: aggregating multiple covariance matrices
        (self.useTransformMatrix, self.offDiagScaleFactor) = \
                        covParameters.getCovarianceComposition()

    def configureSubCovarianceMatrix(self, position, factorCov=None, rp=None):
        """Fine-tune the covariance matrix for the 'block' 
        corresponding to the given position. (starts at zero) 
        eg, position=0 refers to the first factor returns series 
        provided to computeFactorCovarianceMatrix().  
        If factorCov is specified, this pre-computed covariance 
        matrix will be used.  If a RiskParameters object rp is provided,
        it will be used to compute this factor covariance matrix.
        """
        self.subCovariances[position] = dict()
        self.subCovariances[position]['fcov'] = factorCov
        self.subCovariances[position]['props'] = rp
        return True

    def computeFactorCovarianceMatrix(self, *frMatrix):
        """Compute a 'composite' factor covariance matrix.  Each 
        input is a TimeSeriesMatrix of factor returns belonging to
        the corresponding block of the covariance matrix.  Even if
        the covariances for a particular block have been pre-computed
        (eg. currencies), its factor returns series must still be 
        provided.  The first column in each TimeSeriesMatrix of 
        observations should be the most recent.
        """
        self.log.debug('computeFactorCovarianceMatrix: begin')
        # Aggregate all factor returns
        firstTime = True
        for fr in frMatrix:
            if firstTime:
                factorReturns = fr.data
                firstTime = False
            else:
                assert(fr.data.shape[1]==prevSize)
                factorReturns = ma.concatenate(
                        [factorReturns, fr.data], axis=0)
            prevSize = fr.data.shape[1]

        # Compute full-size covariance matrix
        self.log.debug('Computing full size covariance matrix')
        self.toggleAdvancedOptions()
        fullCovarianceMatrix = self.cov_main(factorReturns)
        self.toggleAdvancedOptions()

        # Extract volatilities and correlations
        (fullStdVector, fullCorrelationMatrix) = Utilities.cov2corr(
                fullCovarianceMatrix, fill=True)

        # Initialise the transformation matrix
        correlationTransformMatrix = numpy.eye(
                        factorReturns.shape[0], factorReturns.shape[0])
        overlayMatrix = numpy.zeros(
                        (factorReturns.shape[0], factorReturns.shape[0]))

        # Loop through all the factor 'blocks'
        idx = 0
        for i in range(len(frMatrix)):
            self.log.info('Computing sub covariance matrix, block %d', i)
            fr = frMatrix[i]
            n = idx + fr.data.shape[0]

            #absFR = numpy.median(numpy.transpose(abs(fr.data)), axis=0)
            #absFR = absFR / sum(absFR)
            #badFR = ma.masked_where(absFR<1.0e-3, absFR)
            #badFRIdx = numpy.flatnonzero(ma.getmaskarray(badFR))

            # Check if using a pre-computed cov
            sub = self.subCovariances.get(i, None)
            if sub is not None:
                subCovarianceBlock = sub.get('fcov', None)
                if subCovarianceBlock is None:
                    subCovarianceProps = sub.get('props', None)
            else:
                subCovarianceBlock = None
                subCovarianceProps = None
            # If not, compute covariance matrix for this block
            if subCovarianceBlock is None:
                if subCovarianceProps is None:
                    subCovarianceBlock = self.cov_main(fr.data)
                else:
                    c = StandardCovarianceMatrix(subCovarianceProps)
                    subCovarianceBlock = c.computeFactorCovarianceMatrix(fr)
                    subCovarianceBlock /= 252.0

            # Blank out rows and columns corresponding to bad factor returns
            #for id in badFRIdx:
            #    fullCovarianceMatrix[id,:] = 0.0
            #    fullCovarianceMatrix[:,id] = 0.0
            #    subCovarianceBlock[id,:] = 0.0
            #    subCovarianceBlock[:,id] = 0.0

            overlayMatrix[idx:n,idx:n] = subCovarianceBlock
            (subStdVector, subCorrelationBlock) = Utilities.cov2corr(
                                                subCovarianceBlock, fill=True)
            overlayMatrix[idx:n,idx:n] = subCovarianceBlock
            fullStdVector[idx:n] = subStdVector

            if self.useTransformMatrix:
                # Create part of the transformation matrix
                (s,x) = linalg.eigh(subCorrelationBlock)
                s = ma.sqrt(ma.masked_where(s <= 1.0e-6, s))
                t1 = x*s

                # Create second part of the transformation matrix
                (s,x) = linalg.eigh(fullCorrelationMatrix[idx:n,idx:n])
                s = ma.sqrt(ma.masked_where(s <= 1.0e-6, s))
                t2 = ma.transpose(x*(1.0/s))

                # Insert transform block into the full-size matrix
                correlationTransformMatrix[idx:n,idx:n] = ma.dot(t1, t2).filled(0.0)
            idx += fr.data.shape[0]

        # Compute composite covariance matrix
        if self.useTransformMatrix:
            factorCov = numpy.dot(correlationTransformMatrix, 
                            numpy.dot(fullCorrelationMatrix, 
                            numpy.transpose(correlationTransformMatrix)))
        else:
            factorCov = fullCorrelationMatrix * self.offDiagScaleFactor
        fullStdVector = numpy.diag(fullStdVector)
        factorCov = numpy.dot(fullStdVector, numpy.dot(factorCov, fullStdVector))

        # Directly paste the diagonal blocks into the covariance matrix
        # to eliminate differences due to machine error or masking
        idx = 0
        for i in range(len(frMatrix)):
            fr = frMatrix[i]
            n = idx + fr.data.shape[0]
            factorCov[idx:n,idx:n] = overlayMatrix[idx:n,idx:n]
            idx += fr.data.shape[0]
        
        # Clip small eigenvalues and reconstitute the cov matrix if necessary
        factorCov = Utilities.forcePositiveSemiDefiniteMatrix(
                                    factorCov, min_eigenvalue=0.0)

        # Annualize
        factorCov *= 252.0

        self.log.info('Sum of cov matrix elements: %f', ma.sum(factorCov, axis=None))
        self.log.debug('computeFactorCovarianceMatrix: end')
        return factorCov

class PACESpecificRisk:
    """The original PACE-derived specific risk methodology.
    The ExponentialWeightedVariance class in earlier revision.
    Assets with insufficient specific return histories are
    assigned specific risk values according to a simple proxy/
    replacement rule.
    """
    def __init__(self, srParameters):
        self.log = logging.getLogger('RiskCalculator.PACESpecificRisk')
        self.deltaHalfLife = srParameters.getSpecificRiskHalfLife()
        self.minDeltaObs = minDeltaObs
        self.maxDeltaObs = maxDeltaObs

    def computeSpecificRisks(self, srMatrix):
        """Compute the specific risk values for each asset given a
        TimeSeriesMatrix of specific returns, srMatrix.
        """
        self.log.debug('computeSpecificRisk: begin')
        if srMatrix.data.shape[1] < self.minDeltaObs:
            raise LookupError('Number of time periods, %d, is less than required min number of observations, %d' % (srMatrix.data.shape[1], self.minDeltaObs))

        if srMatrix.data.shape[1] > self.maxDeltaObs:
            raise LookupError('Number of time periods, %d, is greater than max number of observations, %d' % (srMatrix.data.shape[1], self.maxDeltaObs))

        # Winsorize specific return data
        srData = srMatrix.data
        srData = ma.where(ma.greater(srData, 2.5), 2.5, srData)
        srData = ma.where(ma.less(srData, -0.75), -0.75, srData)

        # Get Assets with sufficient history
        condition1 = numpy.sum(ma.getmaskarray(
                    srData[:,0:self.minDeltaObs])==0, axis=1) > (self.minDeltaObs*0.9)
        condition2 = numpy.sum(ma.getmaskarray(
                    srData[:,0:10])==0, axis=1) == 10
        goodAssets = numpy.flatnonzero(condition1 & condition2)
        badAssets = numpy.flatnonzero((condition1 & condition2) == 0)
        self.log.info('Computing specific risk for %d out of %d assets',
                        len(goodAssets), srData.shape[0])
        self.log.info('Using average specific risk for %d out of %d assets',
                        len(badAssets), srData.shape[0])

        # Compute variances of specific returns
        weights = Utilities.computeExponentialWeights(
                            self.deltaHalfLife, srData.shape[1])
        io = (ma.getmaskarray(srData) == 0)
        weightSums = numpy.sum(io.astype(numpy.float) * weights, axis=1)
        goodWeightSums = numpy.take(weightSums, goodAssets, axis=0)
        goodWeightedData = numpy.take(srData.filled(0.0), goodAssets, axis=0) * numpy.sqrt(weights)
        goodSpecificVars = numpy.array([numpy.inner(
                    goodWeightedData[i], goodWeightedData[i])/goodWeightSums[i] \
                    for i in range(len(goodAssets))])

        # Shrink large values
        scaledSpecVars = numpy.log(numpy.sqrt(goodSpecificVars))
        avgScaledSpecVars = numpy.average(scaledSpecVars)
        proxySpecVar = (numpy.exp(avgScaledSpecVars))**2
        volatileAssets = numpy.flatnonzero(scaledSpecVars > avgScaledSpecVars)
        modifiedSpecVars = numpy.take(scaledSpecVars, volatileAssets, axis=0)
        shrink = 0.4
        modifiedSpecVars = modifiedSpecVars * (1.0-shrink) \
                           + avgScaledSpecVars * shrink
        modifiedSpecVars = (numpy.exp(modifiedSpecVars))**2
        numpy.put(goodSpecificVars, volatileAssets, modifiedSpecVars)

        specificVars = Matrices.allMasked(srData.shape[0])
        ma.put(specificVars, goodAssets, goodSpecificVars)
        ma.put(specificVars, badAssets, proxySpecVar)

        # Multiply values by number of annual periods
        specificVars = specificVars * 252.0

        self.log.debug('computeSpecificRisk: end')
        return specificVars

class EnhancedSpecificRisk:
    """Uses regression-based model to proxy specific risk for
    assets with insufficient specific returns history.
    More dynamic (less hard-coding) criteria for determining
    which assets have adequate history.
    Same as the AxiomaVariance2 class in earlier revisions.
    """
    def __init__(self, srParameters):
        self.log = logging.getLogger('RiskCalculator.EnhancedSpecificRisk')
        (minDeltaObs, maxDeltaObs) = srParameters.getSpecificRiskSampleSize()
        self.deltaHalfLife = srParameters.getSpecificRiskHalfLife()
        self.minDeltaObs = minDeltaObs
        self.maxDeltaObs = maxDeltaObs

    def proxySpecificRisk(self, expM, mktcap, specRisk, assetsIdx):
        """Estimate specific risk for assets with insufficient 
        specific return history by regressing specific risks for
        assets with adequate histories against ln(marketCap) and
        industry exposure (if available).
        Returns an array of predicted specific variances for assets
        corresponding to assetsIdx.
        """
        self.log.debug('proxySpecificRisk: begin')

        # Build the regressor matrix
        industryIdx = expM.getFactorIndices(ExposureMatrix.IndustryFactor)
        regressorMatrixFull = numpy.log(mktcap.filled(1.0))[numpy.newaxis]

        # If industry factors exist (ie not stat model) add them
        if len(industryIdx) > 0:
            regM = numpy.take(expM.getMatrix().filled(0.0),
                                            industryIdx, axis=0)
            # Remove any empty industries
            regM = numpy.take(regM, numpy.flatnonzero(numpy.sum(
                    numpy.take(regM, assetsIdx, axis=1), axis=1)), axis=0)
            regressorMatrixFull = numpy.concatenate(
                                [regM, regressorMatrixFull], axis=0)

        # Keep only the assets we're interested in
        regressorMatrix = numpy.take(regressorMatrixFull, assetsIdx, axis=1)

        # Regress specific risk against mcap and (if applicable) industries
        (sr_model_coefs, resid) = Utilities.ordinaryLeastSquares(
                                        specRisk, numpy.transpose(regressorMatrix))

        # Predict specific variance via the regression coefficients
        predictedSpecRisk = numpy.dot(
                numpy.transpose(regressorMatrixFull), sr_model_coefs)
        predictedSpecVars = predictedSpecRisk**2

        self.log.debug('proxySpecificRisk: end')
        return predictedSpecVars

    def computeSpecificRisks(self, exposureMatrix, mktcap, srMatrix):
        """Compute the specific risk values for each asset given a
        TimeSeriesMatrix of specific returns, srMatrix.
        """
        self.log.debug('computeSpecificRisk: begin')
        if srMatrix.data.shape[1] < self.minDeltaObs:
            raise LookupError('Number of time periods, %d, is less than required min number of observations, %d' % (srMatrix.data.shape[1], self.minDeltaObs))

        if srMatrix.data.shape[1] > self.maxDeltaObs:
            raise LookupError('Number of time periods, %d, is greater than max number of observations, %d' % (srMatrix.data.shape[1], self.maxDeltaObs))

        # Winsorize specific return data
        srData = srMatrix.data
        srData = ma.where(ma.greater(srData, 2.5), 2.5, srData)
        srData = ma.where(ma.less(srData, -0.75), -0.75, srData)

        # Get Assets with sufficient history
        weights = Utilities.computeExponentialWeights(
                            self.deltaHalfLife, srData.shape[1])
        io = (ma.getmaskarray(srData[:,0:self.maxDeltaObs]) == 0)
        weightSums = numpy.sum(io.astype(numpy.float) * weights, axis=1)
        reallyGoodThreshold = 0.8
        goodThreshold = 0.2
        condition0 = weightSums > reallyGoodThreshold
        condition1 = weightSums > goodThreshold
        reallyGoodAssets = numpy.flatnonzero(condition0)
        goodAssets = numpy.flatnonzero(condition1)
        badAssets = numpy.flatnonzero(numpy.logical_not(condition1))
        self.log.info('Using only specific risk estimate for %d out of %d assets',
                    len(reallyGoodAssets), srData.shape[0])
        self.log.info('Using combination of estimate and proxy for %d out of %d assets',
                    len(goodAssets)-len(reallyGoodAssets), srData.shape[0])
        self.log.info('Using only specific risk model for %d out of %d assets',
                    len(badAssets), srData.shape[0])

        # Compute variances of specific returns
        goodWeightSums = numpy.take(weightSums, goodAssets, axis=0)
        goodData = numpy.take(srData.filled(0.0), goodAssets, axis=0)
        goodSpecificVars = Utilities.compute_covariance(goodData, 
                weights=weights, deMean=False, axis=1, varsOnly=True)
        goodSpecificVars = goodSpecificVars / goodWeightSums

        # Shrink values greater than average 
        scaledSpecVars = numpy.log(numpy.sqrt(goodSpecificVars))
        avgScaledSpecVars = numpy.average(scaledSpecVars)
        volatileAssets = numpy.flatnonzero(scaledSpecVars > avgScaledSpecVars)
        modifiedSpecVars = numpy.take(scaledSpecVars, volatileAssets, axis=0)
        shrink = 0.4
        modifiedSpecVars = modifiedSpecVars * (1.0-shrink) \
                           + avgScaledSpecVars * shrink
        modifiedSpecVars = (numpy.exp(modifiedSpecVars))**2
        numpy.put(goodSpecificVars, volatileAssets, modifiedSpecVars)

        specificVars = Matrices.allMasked(srData.shape[0])
        ma.put(specificVars, goodAssets, goodSpecificVars)

        reallyGoodSpecVars = numpy.take(specificVars.filled(0.0),
                                          reallyGoodAssets, axis=0)
        reallyGoodSpecRisk = numpy.sqrt(reallyGoodSpecVars)

        # If insufficient assets with specific return history,
        # Compute specific-risk replacement model
        if len(reallyGoodAssets) != srData.shape[0]: 

            predictedSpecVars = self.proxySpecificRisk(exposureMatrix, 
                        mktcap, reallyGoodSpecRisk, reallyGoodAssets)
            
            # Compute final specific risks as combination of estimated and
            # model-predicted values
            goodMultiplier = (goodWeightSums - goodThreshold)\
                         / (reallyGoodThreshold - goodThreshold)
            goodSpecificVars = goodMultiplier*goodSpecificVars \
                               + (1.0-goodMultiplier) * \
                               ma.take(predictedSpecVars, goodAssets, axis=0)
            ma.put(specificVars, goodAssets, goodSpecificVars)
            ma.put(specificVars, badAssets, ma.take(predictedSpecVars, badAssets, axis=0))
        ma.put(specificVars, reallyGoodAssets, reallyGoodSpecVars)
        
        # Multiply values by number of annual periods
        specificVars = specificVars * 252.0

        self.log.debug('computeSpecificRisk: end')
        return specificVars

class BetterSpecificRisk(EnhancedSpecificRisk):
    """Same regression-based method to proxy specific risks
    for assets with insufficient history.  Median Absolute 
    Deviation (MAD) based clipping of specific returns 
    prior to sample variance estimation to reduce the effect 
    of noisy outliers.  Also supports Newey-West autocorrelation 
    adjustment, alternate weighting scheme, and de-meaned 
    variance computation.
    Previously known as AxiomaVariance3.
    """
    def __init__(self, srParameters):
        self.log = logging.getLogger('RiskCalculator.BetterSpecificRisk')
        EnhancedSpecificRisk.__init__(self, srParameters)
        self.deltaLag = srParameters.getSpecificRiskNeweyWestLag()
        self.deMeanFlag = srParameters.useDeMeanedSpecificRisk()
        self.equalWeightFlag = srParameters.useEqualWeightedSpecificRisk()
        (tmp1, tmp2, bounds) = srParameters.getSpecificRiskClipping()
        if bounds is not None:
            (self.lowerMADBound, self.upperMADBound) = bounds
        else:
            raise Exception('No specific return clip bounds given')

    def computeSpecificRisks(self, exposureMatrix, mktcap, srMatrix):
        """Compute the specific risk values for each asset given a
        TimeSeriesMatrix of specific returns, srMatrix.
        """
        self.log.debug('computeSpecificRisk: begin')
        if srMatrix.data.shape[1] < self.minDeltaObs:
            raise LookupError('Number of time periods, %d, is less than required min number of observations, %d' % (srMatrix.data.shape[1], self.minDeltaObs))
        if srMatrix.data.shape[1] > self.maxDeltaObs:
            raise LookupError('Number of time periods, %d, is greater than max number of observations, %d' % (srMatrix.data.shape[1], self.maxDeltaObs))
            
        # Get Assets with sufficient history
        srData = srMatrix.data
        weights = Utilities.computeExponentialWeights(
                        self.deltaHalfLife, srData.shape[1], self.equalWeightFlag)
        io = (ma.getmaskarray(srData[:,0:self.maxDeltaObs]) == 0)
        weightSums = numpy.sum(io.astype(numpy.float) * weights[0:self.maxDeltaObs], axis=1)
        reallyGoodThreshold = 0.8
        goodThreshold = 0.5
        condition0 = weightSums > reallyGoodThreshold
        condition1 = weightSums > goodThreshold
        reallyGoodAssets = numpy.flatnonzero(condition0)
        goodAssets = numpy.flatnonzero(condition1)
        badAssets = numpy.flatnonzero(numpy.logical_not(condition1))
        self.log.info('Using only specific risk estimate for %d out of %d assets',
                len(reallyGoodAssets), srData.shape[0])
        self.log.info('Using combination of estimate and proxy for %d out of %d assets',
                len(goodAssets)-len(reallyGoodAssets), srData.shape[0])
        self.log.info('Using only fill-in algorithm for %d out of %d assets',
                len(badAssets), srData.shape[0])

        tmp = srData.filled(0.0)
        self.log.info('Specific return bounds: [%.3f, %.3f]',
                      numpy.min(tmp, axis=None), numpy.max(tmp, axis=None))

        # Clip extreme specific returns using MAD
        (srData, mad_bounds) = Utilities.mad_dataset(srData,
                        self.lowerMADBound, self.upperMADBound,
                        restrict=reallyGoodAssets, axis=0)
        
        tmp = srData.filled(0.0)
        self.log.info('Specific return bounds after MADing: [%.3f, %.3f]',
                      numpy.min(tmp, axis=None), numpy.max(tmp, axis=None))

        # Compute variances of specific returns
        goodWeightSums = numpy.take(weightSums, goodAssets, axis=0)
        goodData = numpy.take(srData.filled(0.0), goodAssets, axis=0)

        goodSpecificVars = Utilities.compute_covariance(goodData, 
                weights=weights, deMean=self.deMeanFlag, axis=1, varsOnly=True)

        # Perform Newey-West using given number of lags
        ARadj = numpy.zeros(goodSpecificVars.shape, float)
        for lag in range(1,self.deltaLag+1):
            lagVar = Utilities.compute_covariance(goodData, weights=weights, 
                    deMean=self.deMeanFlag, axis=1, varsOnly=True, lag=lag)
            ARadj += (1.0 - float(lag)/(float(self.deltaLag)+1.0)) * (2.0 * lagVar)
        goodSpecificVars += ARadj

        # Adjust variances for variable histories
        goodSpecificVars = goodSpecificVars / goodWeightSums

        # Sort out the really good from the merely good forecasts
        specificVars = Matrices.allMasked(srData.shape[0])
        ma.put(specificVars, goodAssets, goodSpecificVars)
        reallyGoodSpecVars = numpy.take(specificVars.filled(0.0),
                reallyGoodAssets, axis=0)
        reallyGoodSpecRisk = numpy.sqrt(reallyGoodSpecVars)

        # If insufficient assets with specific return history,
        # Compute specific-risk replacement model
        if len(reallyGoodAssets) != srData.shape[0]:

            predictedSpecVars = self.proxySpecificRisk(exposureMatrix, 
                        mktcap, reallyGoodSpecRisk, reallyGoodAssets)

            # Compute final specific risks as combination of estimated and
            # model-predicted values
            goodMultiplier = (goodWeightSums - goodThreshold)\
                    / (reallyGoodThreshold - goodThreshold)
            goodSpecificVars = goodMultiplier*goodSpecificVars \
                    + (1.0-goodMultiplier) * \
            ma.take(predictedSpecVars, goodAssets, axis=0)
            ma.put(specificVars, goodAssets, goodSpecificVars)
            ma.put(specificVars, badAssets, ma.take(predictedSpecVars, badAssets, axis=0))

        ma.put(specificVars, reallyGoodAssets, reallyGoodSpecVars)
        
        # Multiply values by number of annual periods
        specificVars = specificVars * 252.0
        
        # If, after all this, there are still really huge or really tiny values,
        # do a little simple clipping
        specificVars = numpy.clip(specificVars, 0.0004, 4.0)
        specificRisks = ma.sqrt(specificVars)
        self.log.info('Specific Risk bounds: [%.3f, %.3f]', 
                min(specificRisks), max(specificRisks))
        
        self.log.debug('computeSpecificRisk: end')
        return specificVars

class BrilliantSpecificRisk:
    """Instead of proxying specific risk values, we generate a 
    proxy/pseudo specific return values whenever an asset is
    missing a specific return.  In addition, supports 
    clipping/transformation of specific returns based on recent
    MAD behavior to improve forecast responsiveness.
    Usual bells n' whistles: Newey-West autocorrelation adjustment, 
    alternate weighting scheme, and de-meaned variance computation.
    """
    def __init__(self, srParameters):
        self.log = logging.getLogger('RiskCalculator.BrilliantSpecificRisk')
        (self.minDeltaObs, self.maxDeltaObs) = \
                           srParameters.getSpecificRiskSampleSize()
        self.deltaHalfLife = srParameters.getSpecificRiskHalfLife()
        self.deltaLag = srParameters.getSpecificRiskNeweyWestLag()
        self.deMeanFlag = srParameters.useDeMeanedSpecificRisk()
        self.equalWeightFlag = srParameters.useEqualWeightedSpecificRisk()

        # Enhancements: specific return proxy
        self.buckets = None

        # Enhancements: specific return clipping
        (self.useMedian, self.clipSampleObs, bounds) = \
                         srParameters.getSpecificRiskClipping()
        if self.clipSampleObs is not None and bounds is None:
            raise Exception('MAD-clipping enabled but no clipping bounds given')
        if bounds is not None:
            (self.clipLowerBound, self.clipUpperBound) = bounds

        # Enhancements: Dynamic Volatility Adjustment (DVA)
        self.DVAWindow = srParameters.getSpecificRiskDVAOptions()

        if self.clipSampleObs is not None and \
                    self.DVAWindow is not None:
            self.log.warning('Both clipping and DVA enabled, clipping will be disabled')
            self.clipSampleObs = None

    def computeSpecificRisks(self, exposureMatrix, marketCaps, srMatrix, estu=None):
        """Compute the specific risk values for each asset given a
        TimeSeriesMatrix of specific returns, srMatrix.
        The exposureMatrix variable is not used; just to keep
        list of input args consistent with other specific risk
        classes.
        """
        self.log.debug('computeSpecificRisk: begin')
        if srMatrix.data.shape[1] < self.minDeltaObs:
            raise LookupError('Number of time periods, %d, is less than required min number of observations, %d' % (srMatrix.data.shape[1], self.minDeltaObs))
        if srMatrix.data.shape[1] > self.maxDeltaObs:
            raise LookupError('Number of time periods, %d, is greater than max number of observations, %d' % (srMatrix.data.shape[1], self.maxDeltaObs))
            
        srData = srMatrix.data
        tmp = srData.filled(0.0)
        self.log.info('Specific return bounds: [%.3f, %.3f]',
                      numpy.min(tmp, axis=None), numpy.max(tmp, axis=None))

        # Identify and mask large values via MeanAD
        srData_flat = ma.ravel(srData)[numpy.newaxis,...]
        (srData_flat, mad_bounds) = Utilities.mad_dataset(srData_flat,
                        10.0*self.clipLowerBound, 10.0*self.clipUpperBound,
                        restrict=None, axis=1, method='mean')
        ma.put(ma.array(srData), 
                list(range(srData.shape[0] * srData.shape[1])), srData_flat)
        srData = ma.masked_where(srData <= mad_bounds[0], srData)
        srData = ma.masked_where(srData >= mad_bounds[1], srData)

        tmp = srData.filled(0.0)
        self.log.info('Specific return bounds: [%.3f, %.3f]',
                      numpy.min(tmp, axis=None), numpy.max(tmp, axis=None))

        # Clip or mask extreme values
        if self.clipSampleObs is not None:
            (srData, bounds) = Utilities.mad_dataset(srData,
                    self.clipLowerBound, self.clipUpperBound,
                    restrict=list(range(self.clipSampleObs)), axis=1,
                    method=self.useMedian)
        else:
            (sr_mad, bounds) = Utilities.mad_dataset(srData.filled(0.0),
                    self.clipLowerBound, self.clipUpperBound,
                    restrict=None, axis=0, treat='mask', method='mean')
            srData = ma.masked_where(srData <= bounds[0], srData)
            srData = ma.masked_where(srData >= bounds[1], srData)

        # Fill in missing specific returns with pseudo values
        if self.buckets is None:
            buckets = Utilities.generate_marketcap_buckets(
                        marketCaps, (0.80, 0.15))
        else:
            buckets = self.buckets
        srData = Utilities.compute_asset_value_proxies(srData, buckets)
        self.log.info('Specific return bounds: [%.3f, %.3f]',
                numpy.min(srData, axis=None), numpy.max(srData, axis=None))

        # Dampen specific returns
        if self.DVAWindow:
            self.log.debug('Applying Dynamic Volatility Adjustment (DVA)')
            srData = Utilities.dynamic_volatility_adjustment(srData, self.DVAWindow)

        weights = Utilities.computeExponentialWeights(
                            self.deltaHalfLife, srData.shape[1], self.equalWeightFlag)
        specificVars = Utilities.compute_covariance(numpy.array(srData),
                weights=weights, deMean=self.deMeanFlag, axis=1, varsOnly=True)

        # Perform Newey-West using given number of lags
        for lag in range(1,self.deltaLag+1):
            lagVar = Utilities.compute_covariance(numpy.array(srData),
                    weights=weights, deMean=self.deMeanFlag, 
                    axis=1, varsOnly=True, lag=lag)
            specificVars += (1.0 - float(lag)/(float(self.deltaLag)+1.0)) * (2.0 * lagVar)

        # Multiply values by number of annual periods
        specificVars = specificVars * 252.0

        # Trunate any really huge or really tiny values which remain
        specificVars = numpy.clip(specificVars, 0.0025, 2.25)
            
        specificRisks = ma.sqrt(specificVars)
        self.log.info('Specific Risk bounds: [%.3f, %.3f], Mean: %.3f', 
            min(specificRisks), max(specificRisks), ma.average(specificRisks))
        
        self.log.debug('computeSpecificRisk: end')
        return specificVars


class RiskParameters2009:
    """Stores parameters (half-life, etc.) for all risk-related
    computation procedures.
    Instantiate using 2 dictionaries containing the parameter
    names and values for covariance matrix and specific risk
    computation.
    """
    def __init__(self, covParamsDict):
        self.log = logging.getLogger('RiskCalculator.RiskParameters')
        self.covarianceParameters = covParamsDict

    def getCovarianceHalfLife(self):
        return (self.covarianceParameters.get('halfLife', 125))

    def getCovarianceSampleSize(self):
        hl = self.getCovarianceHalfLife()
        return (self.covarianceParameters.get('minObs', hl),
                self.covarianceParameters.get('maxObs', hl*2))

    def getCovarianceClipping(self):
        return (self.covarianceParameters.get('clipMethod', 'median'),
                self.covarianceParameters.get('clipHistory', None),
                self.covarianceParameters.get('clipBounds', None))

    def getCovarianceNeweyWestLag(self):
        return ( self.covarianceParameters.get('NWLag', 0))

    def getCovarianceDVAOptions(self):
        return (self.covarianceParameters.get('DVAWindow', None),
                self.covarianceParameters.get('DVAType', 'step'),
                self.covarianceParameters.get('DVAUpperBound', 1.25),
                self.covarianceParameters.get('DVALowerBound', 0.80))

    def getSplineDVAScaleRatios(self):
        return (self.covarianceParameters.get('DVAUpperRatio', 0.05),
                self.covarianceParameters.get('DVALowerRatio', -0.1))

    def getCovarianceComposition(self):
        return (self.covarianceParameters.get('useTransformMatrix', False),
                self.covarianceParameters.get('offDiagScaleFactor', 1.0))

    def useDeMeanedCovariance(self):
        return self.covarianceParameters.get('deMeanFlag', False)

    def getSelectiveDeMeanParameters(self):
        return (self.covarianceParameters.get('selectiveDeMean', False),
                self.covarianceParameters.get('deMeanFactorTypes', None),
                self.covarianceParameters.get('deMeanHalfLife', None),
                self.covarianceParameters.get('deMeanHistoryLength', None))

    def useEqualWeightedCovariance(self):
        return self.covarianceParameters.get('equalWeightFlag', False)

    def useProxyReturnFillIn(self):
        return self.covarianceParameters.get('fillInFlag', True)

    def getSubFactorsForDVA(self):
        return self.covarianceParameters.get('subFactorsForDVA',None)

    def setSubFactorsForDVA(self,subFactorsForDVA):
        self.covarianceParameters['subFactorsForDVA']=subFactorsForDVA

class CompositeCovarianceMatrix2009:
    """Composite covariance matrix.
    """
    def __init__(self, fullCovParameters, varParameters, corrParameters):
        self.log = logging.getLogger('RiskCalculator.CompositeCovarianceMatrix')
        self.subCovariances = dict()
        self.fullCovParameters = fullCovParameters
        self.varParameters = varParameters
        self.corrParameters = corrParameters

        # Enhancements: aggregating multiple covariance matrices
        (self.useTransformMatrix, self.offDiagScaleFactor) = \
                        self.fullCovParameters.getCovarianceComposition()

    def configureSubCovarianceMatrix(self, position, factorCov=None, rp=None):
        """Fine-tune the covariance matrix for the 'block' 
        corresponding to the given position. (starts at zero) 
        eg, position=0 refers to the first factor returns series 
        provided to computeFactorCovarianceMatrix().  
        If factorCov is specified, this pre-computed covariance 
        matrix will be used.  If a RiskParameters object rp is provided,
        it will be used to compute this factor covariance matrix.
        """
        self.subCovariances[position] = dict()
        self.subCovariances[position]['fcov'] = factorCov
        return True

    def cov_main(self, fr, covParameters, matrixType='cov', factorIndices=None):
        """Core routine for computing a factor covariance matrix.
           factorIndices contains a list of indices corresponding to fr to which dva is to be applied.
        """
        # Set up parameters
        self.halfLife = covParameters.getCovarianceHalfLife()
        (self.minObs, self.maxObs) = covParameters.getCovarianceSampleSize()
        self.deMeanFlag = covParameters.useDeMeanedCovariance()
        self.equalWeightFlag = covParameters.useEqualWeightedCovariance()

        # Enhancements: autocorrelation lag
        self.NWLag = covParameters.getCovarianceNeweyWestLag()

        # Enhancements: Dynamic Volatility Adjustment (DVA)
        (self.DVAWindow, self.DVAType, self.DVAUpperBound, self.DVALowerBound) =\
                covParameters.getCovarianceDVAOptions()

        if fr.shape[1] > self.maxObs:
            fr = fr[:, 0:self.maxObs]
        nPeriods = fr.shape[1]
        numLags = self.NWLag
        self.log.info('Using %d factors by %d periods', fr.shape[0], nPeriods)

        # Calculate weights
        factorReturnWeights = Utilities.computeExponentialWeights(
                self.halfLife, nPeriods, self.equalWeightFlag)

        # Pre-process (clip, DVA, etc) factor returns
        if self.DVAWindow is not None:
            self.log.debug('Applying Dynamic Volatility Adjustment (DVA)')
            if self.DVAType == 'spline':
                (self.DVAUpperBound, self.DVALowerBound) = \
                        covParameters.getSplineDVAScaleRatios()
                frFull = Utilities.spline_dva(fr, self.DVAWindow, 
                        upperBound=self.DVAUpperBound, lowerBound=self.DVALowerBound, factorIndices=factorIndices)
            else:
                frFull = Utilities.dynamic_volatility_adjustment(fr, self.DVAWindow,
                        scaleType=self.DVAType, upperBound=self.DVAUpperBound, lowerBound=self.DVALowerBound)
        else:
            frFull = fr

        # Check for factor returns consisting entirely of zeros
        frFullMasked = ma.masked_where(abs(frFull*factorReturnWeights) < 1e-10, frFull)
        frNonZeros = (ma.getmaskarray(frFullMasked)==0.0)
        sumFactors = ma.sum(frNonZeros, axis=1).astype(numpy.float) / fr.shape[1]
        sumFactors = ma.where(sumFactors < 0.01, 0.0, sumFactors)
        goodFactorsIdx = numpy.flatnonzero(sumFactors)

        frData = ma.take(frFull, goodFactorsIdx, axis=0).filled(0.0)
        if len(goodFactorsIdx) != frFull.shape[0]:
            self.log.info('%d out of %d factors are effectively zero',
                    frFull.shape[0]-len(goodFactorsIdx), frFull.shape[0])
            self.log.info('Dimension of cov matrix reduced to %d for calculation',
                    len(goodFactorsIdx))

        # Calculate lag zero covariance matrix
        factorCov = Utilities.compute_covariance(frData,
                weights=factorReturnWeights, deMean=self.deMeanFlag, axis=1)

        # Perform Newey-West using given number of lags
        self.log.info('Number of autocorrelation lags (Static): %d', numLags)
        ARadj = numpy.zeros(factorCov.shape, float)
        for lag in range(1,numLags+1):
            lagCov = Utilities.compute_covariance(frData,
                    weights=factorReturnWeights, deMean=self.deMeanFlag, axis=1, lag=lag)
            ARadj += (1.0 - float(lag)/(float(numLags)+1.0)) * \
                    (lagCov + numpy.transpose(lagCov))
        factorCov += ARadj

        # In case python's numerical accuracy falls short
        factorCov = (factorCov + numpy.transpose(factorCov)) / 2.0

        # If we've computed a reduced covariance matrix, paste it
        # back into the full-sized variant
        if frData.shape[0] < fr.shape[0]:
            fullFactorCov = numpy.zeros((fr.shape[0], fr.shape[0]), float)
            for (i, id) in enumerate(goodFactorsIdx):
                for (j, jd) in enumerate(goodFactorsIdx):
                    fullFactorCov[id,jd] = factorCov[i,j]
            factorCov = fullFactorCov

        # Return the relevant object
        if matrixType=='std' or matrixType=='cor':
            (stdVector, factorCor) = Utilities.cov2corr(factorCov)
            if matrixType=='std':
                return stdVector
            else:
                return factorCor
        else:
            (stdVector, factorCov) = Utilities.cov2corr(
                    factorCov, returnCov=True)
            return factorCov

    def computeFactorCovarianceMatrix(self, *frMatrix):
        """Compute a 'composite' factor covariance matrix.  Each 
        input is a TimeSeriesMatrix of factor returns belonging to
        the corresponding block of the covariance matrix.  Even if
        the covariances for a particular block have been pre-computed
        (eg. currencies), its factor returns series must still be 
        provided.  The first column in each TimeSeriesMatrix of 
        observations should be the most recent.
        """
        self.log.debug('computeFactorCovarianceMatrix: begin')
        # Aggregate all factor returns
        firstTime = True
        factorReturnsIdx=[]
        for fr in frMatrix:
            factorReturnsIdx.extend(fr.assets)
            # Returns timing adjustment
            if hasattr(fr, 'adjust'):
                self.log.info('Adjusting returns')
                frData = fr.data + fr.adjust
            # Or not
            else:
                frData = ma.array(fr.data, copy=True)

            # Long mean adjustment
            if hasattr(fr, 'mean'):
                frData = numpy.transpose(numpy.transpose(frData) - fr.mean)

            if firstTime:
                factorReturns = ma.array(frData, copy=True)
                firstTime = False
            else:
                assert(fr.data.shape[1]==prevSize)
                factorReturns = ma.concatenate(
                        [factorReturns, frData], axis=0)
            prevSize = frData.shape[1]

        # Compute full-size covariance matrix
        if len(frMatrix) > 1:
            self.log.debug('Computing full size correlation matrix')
            if self.useTransformMatrix:
                self.log.info('Using matrix transform')
            else:
                self.log.info('Using matrix cut-and-paste')
            if self.fullCovParameters.getSubFactorsForDVA() is not None:
                factorIndices = [i for i,x in enumerate(factorReturnsIdx) if x in self.fullCovParameters.getSubFactorsForDVA()]              
            else:
                factorIndices = None

            fullCorrelationMatrix = self.cov_main(factorReturns,
                    self.fullCovParameters, matrixType='cor', factorIndices=factorIndices)
        else:
            self.useTransformMatrix = False
            fullCorrelationMatrix = ma.zeros(
                    (factorReturns.shape[0], factorReturns.shape[0]), float)
        fullStdVector = ma.zeros((factorReturns.shape[0]), float)

        # Initialise the transformation matrix
        covarianceTransformMatrix = ma.zeros(
                        (factorReturns.shape[0], factorReturns.shape[0]), float)
        overlayMatrix = ma.zeros(
                        (factorReturns.shape[0], factorReturns.shape[0]), float)

        # Loop through all the factor 'blocks'
        idx = 0
        for i in range(len(frMatrix)):
            self.log.info('Computing sub covariance matrix, block %d', i)
            fr = frMatrix[i]
            # Returns timing adjustment
            if hasattr(fr, 'adjust'):
                self.log.info('Adjusting returns')
                frData = fr.data + fr.adjust
            # Or not
            else:
                frData = ma.array(fr.data, copy=True)
            n = idx + fr.data.shape[0]

            # Check if using a pre-computed cov
            sub = self.subCovariances.get(i, None)
            if sub is not None:
                self.log.info('Loading pre-computed block')
                subCovarianceBlock = sub.get('fcov', None)
                (subStdVector, subCorrelationBlock) = Utilities.cov2corr(
                                                                subCovarianceBlock)
            else:
                # Compute correlations
                self.log.info('Computing correlation sub-block')
                if self.corrParameters.getSubFactorsForDVA() is not None:
                    factorIndices = [i for i,x in enumerate(fr.assets) if x in self.corrParameters.getSubFactorsForDVA()]              
                else:
                    factorIndices = None
                subCorrelationBlock = self.cov_main(frData, self.corrParameters,
                                                    matrixType='cor', factorIndices=factorIndices)
            
                # Compute variances
                self.log.info('Computing variance sub-block')
                subStdVector = self.cov_main(frData, self.varParameters,
                                                    matrixType='std', factorIndices=factorIndices)

            # Fill in relevant parts of full-sized matrices
            overlayMatrix[idx:n,idx:n] = subCorrelationBlock
            fullStdVector[idx:n] = subStdVector

            if self.useTransformMatrix:
                # Create part of the transformation matrix
                (s,x) = linalg.eigh(subCorrelationBlock.filled(0.0))
                s = ma.sqrt(ma.masked_where(s <= 1.0e-12, s))
                t1 = x*s

                # Create second part of the transformation matrix
                (s,x) = linalg.eigh(fullCorrelationMatrix[idx:n,idx:n].filled(0.0))
                s = ma.sqrt(ma.masked_where(s <= 1.0e-12, s))
                t2 = ma.transpose(x*(1.0/s))

                # Insert transform block into the full-size matrix
                covarianceTransformMatrix[idx:n,idx:n] = ma.dot(t1, t2)
            idx += fr.data.shape[0]

        # Compute composite correlation matrix
        if self.useTransformMatrix:
            # The "proper" transformed correlation matrix
            fullCorrelationMatrix = ma.dot(covarianceTransformMatrix,\
                    ma.dot(fullCorrelationMatrix,\
                    ma.transpose(covarianceTransformMatrix)))

        # Even if using the "proper" transform, directly paste the diagonal blocks
        # into the correlation matrix to eliminate differences due to machine error
        # or masking
        idx = 0
        for i in range(len(frMatrix)):
            fr = frMatrix[i]
            n = idx + fr.data.shape[0]
            fullCorrelationMatrix[idx:n,idx:n] = overlayMatrix[idx:n,idx:n]
            idx += fr.data.shape[0]

        # Combine variances with correlations
        factorCov = ma.transpose(ma.transpose(
            fullCorrelationMatrix*fullStdVector)*fullStdVector)

        # Clip small eigenvalues and reconstitute the cov matrix if necessary
        factorCov = factorCov.filled(0.0)
        if len(frMatrix) < 2:
            factorCov = Utilities.forcePositiveSemiDefiniteMatrix(
                    factorCov, min_eigenvalue=0.0)

        # Annualize
        factorCov *= 252.0

        self.log.info('Sum of cov matrix elements: %f', ma.sum(factorCov, axis=None))
        self.log.debug('computeFactorCovarianceMatrix: end')
        return factorCov

class BrilliantSpecificRisk2009:
    """Instead of proxying specific risk values, we generate a 
    proxy/pseudo specific return values whenever an asset is
    missing a specific return.  In addition, supports 
    clipping/transformation of specific returns based on recent
    MAD behavior to improve forecast responsiveness.
    Usual bells n' whistles: Newey-West autocorrelation adjustment, 
    alternate weighting scheme, and de-meaned variance computation.
    """
    def __init__(self, srParameters):
        self.log = logging.getLogger('RiskCalculator.BrilliantSpecificRisk')
        (self.minDeltaObs, self.maxDeltaObs) = \
                           srParameters.getCovarianceSampleSize()
        self.deltaHalfLife = srParameters.getCovarianceHalfLife()
        self.deltaLag = srParameters.getCovarianceNeweyWestLag()
        self.deMeanFlag = srParameters.useDeMeanedCovariance()
        self.equalWeightFlag = srParameters.useEqualWeightedCovariance()
        
        # Enhancements: specific return proxy
        self.buckets = None
        self.returnsFillIn = srParameters.useProxyReturnFillIn()
        
        # Enhancements: specific return clipping
        (self.useMedian, self.clipSampleObs, bounds) = \
                         srParameters.getCovarianceClipping()
        if self.clipSampleObs is not None and bounds is None:
            raise Exception('MAD-clipping enabled but no clipping bounds given')
        if bounds is not None:
            (self.clipLowerBound, self.clipUpperBound) = bounds
        
        # Enhancements: Dynamic Volatility Adjustment (DVA)
        (self.DVAWindow, self.DVAType, self.DVAUpperBound, self.DVALowerBound) =\
                srParameters.getCovarianceDVAOptions()
        
        if self.clipSampleObs is not None and \
                    self.DVAWindow is not None:
            self.log.warning('Both clipping and DVA enabled, clipping will be disabled')
            self.clipSampleObs = None
    
    def preProcessSpecificReturns(self, srMatrix, marketCaps, estu, multiMAD):
        if srMatrix.data.shape[1] > self.maxDeltaObs:
            raise LookupError('Number of time periods, %d, is greater than max number of observations, %d' % (srMatrix.data.shape[1], self.maxDeltaObs))
            
        srData = srMatrix.data
        self.log.info('Specific return bounds: [%.3f, %.3f]',
                      numpy.min(srData, axis=None), numpy.max(srData, axis=None))
        weights = Utilities.computeExponentialWeights(
                self.deltaHalfLife, srData.shape[1], self.equalWeightFlag)
        
        # Clip or mask extreme values
        if self.clipSampleObs is not None:
            (srData, bounds) = Utilities.mad_dataset(srData,
                    self.clipLowerBound, self.clipUpperBound,
                    restrict=list(range(self.clipSampleObs)), axis=1,
                    method=self.useMedian)
        elif multiMAD:
            srData = Utilities.multi_mad_data(srData, restrictZeroAxis=estu)
            logging.info('Using old test MultiMAD code')
        else:
            (srData, bounds) = Utilities.mad_dataset(srData,
                    self.clipLowerBound, self.clipUpperBound,
                    restrict=estu, axis=0, method=self.useMedian,
                    treat='clip')
        
        # Fill in missing specific returns with pseudo values
        if self.returnsFillIn:
            if self.buckets is None:
                buckets = Utilities.generate_marketcap_buckets(
                            marketCaps, (0.80, 0.15), restrict=estu)
            else:
                buckets = self.buckets
            srData = Utilities.compute_asset_value_proxies(\
                    srData, buckets, restrict=estu)
            weightSums = None
        else:
            io = (ma.getmaskarray(srData[:,0:self.maxDeltaObs]) == 0)
            weightSums = numpy.sum(io.astype(numpy.float) \
                    * weights[0:self.maxDeltaObs], axis=1)
        srData = srData.filled(0.0)
        
        self.log.info('Specific return bounds: [%.3f, %.3f]',
                numpy.min(srData, axis=None), numpy.max(srData, axis=None))
        
        # Dampen specific returns
        if self.DVAWindow is not None:
            self.log.debug('Applying Dynamic Volatility Adjustment (DVA)')
            if self.DVAType == 'spline':
                (self.DVAUpperBound, self.DVALowerBound) = \
                        srParameters.getSplineDVAScaleRatios()
                srData = Utilities.spline_dva(srData, self.DVAWindow,
                        upperBound=self.DVAUpperBound, lowerBound=self.DVALowerBound)
            else:
                srData = Utilities.dynamic_volatility_adjustment(srData, self.DVAWindow,
                        scaleType=self.DVAType, upperBound=self.DVAUpperBound, lowerBound=self.DVALowerBound)
        
        return (srData, weights, weightSums)
    
    def computeSpecificRiskInternal(self, srData, weights, weightSums, clipVars):
        specificVars = Utilities.compute_covariance(numpy.array(srData),
                weights=weights, deMean=self.deMeanFlag, axis=1, varsOnly=True)
        
        # Perform Newey-West using given number of lags
        for lag in range(1,self.deltaLag+1):
            lagVar = Utilities.compute_covariance(numpy.array(srData),
                    weights=weights, deMean=self.deMeanFlag, 
                    axis=1, varsOnly=True, lag=lag)
            specificVars += (1.0 - float(lag)/(float(self.deltaLag)+1.0)) * (2.0 * lagVar)
        
        if not self.returnsFillIn:
            specificVars = specificVars / weightSums
        
        # Multiply values by number of annual periods
        specificVars = specificVars * 252.0
        
        # Truncate any really huge or really tiny values which remain
        minSpecificVar = 0.00000025
        maxSpecificVar = 4.0
        if clipVars:
            specificVars = numpy.clip(specificVars, minSpecificVar, maxSpecificVar)
            
        specificRisks = ma.sqrt(specificVars)
        self.log.info('Specific Risk bounds: [%.3f, %.3f], Mean: %.3f',
            min(specificRisks), max(specificRisks), ma.average(specificRisks))
        return specificVars
        
    def computeSpecificRisks(self, exposureMatrix, marketCaps, srMatrix, estu=None,
                             clipVars=True, multiMAD=False):
        """Compute the specific risk values for each asset given a
        TimeSeriesMatrix of specific returns, srMatrix.
        The exposureMatrix variable is not used; just to keep
        list of input args consistent with other specific risk
        classes.
        """
        self.log.debug('computeSpecificRisk: begin')
        (srData, weights, weightSums) = self.preProcessSpecificReturns(
            srMatrix, marketCaps, estu, multiMAD)
        specificVars = self.computeSpecificRiskInternal(
            srData, weights, weightSums, clipVars)
        self.log.debug('computeSpecificRisk: end')
        return specificVars

class SparseSpecificRisk2010(BrilliantSpecificRisk2009):
    """Identical to BrilliantSpecificRisk2009 but computes specific
    covariances between 'linked' SubIssues.
    """
    def __init__(self, srParameters):
        BrilliantSpecificRisk2009.__init__(self, srParameters)
    
    def computeCointegratedCovariances(self, assets, srData,
            assetGroupingsMap, specificVars, scoreDict, weights, exclList,
            hardCloneMap):
        """Compute ISC information using cointegration-based
        techniques
        """
        self.log.info("Computing robust covariances for %d blocks of assets",
                len(assetGroupingsMap))
        # Initialise
        assetIdxMap = dict([(j,i) for (i,j) in enumerate(assets)])
        specificRisks = numpy.sqrt(specificVars)
        specificCovMap = dict()
        # Create history of pseudo-prices
        prices = numpy.log(1.0 + srData)
        prices[:,-1] = prices[:,-1] + 1.0
        n = prices.shape[1]
        for k in range(n-1,0,-1):
            prices[:,k-1] = prices[:,k-1] + prices[:,k]
        # Loop round blocks of linked assets
        for (groupId, subIssueList) in assetGroupingsMap.items():
            # Assign scores to each asset and order from highest to lowest-scoring
            score = scoreDict[groupId]
            indices  = [assetIdxMap[n] for n in subIssueList]
            cloneList = [n for n in subIssueList if n in hardCloneMap]
            sortedIndices = [indices[j] for j in numpy.argsort(-score)]
            sortedIdxMap = dict(zip(sortedIndices, list(range(len(sortedIndices)))))
            subCovMatrix = numpy.zeros((len(sortedIndices), len(sortedIndices)), float)

            # Loop round each combination of linked assets
            for (i,idx) in enumerate(sortedIndices):
                subid1 = assets[idx]
                subCovMatrix[i,i] = specificVars[idx]

                # Only need perform computation on half the matrix
                for (j,jdx) in enumerate(sortedIndices[i+1:]):
                    subid2 = assets[jdx]

                    # Compute price-beta
                    x = prices[idx,:]
                    y = prices[jdx,:]
                    coeff = numpy.dot(x,y) / numpy.dot(x,x)
                    coeff = numpy.clip(coeff, 0.8, 1.25)

                    # Compute error variance
                    error = y - (coeff*x)
                    eps = error[:-1] - error[1:]
                    epsVar = numpy.var(eps)
                     
                    # Compute new variance for linked asset
                    if i == 0 and subid2 not in exclList:
                        specificVars[jdx] = (coeff * coeff * specificVars[idx]) + epsVar
                        specificRisks[jdx] = numpy.sqrt(specificVars[jdx])
                         
                    # Compute pseudo-correlation between assets
                    correl = coeff * specificRisks[idx] / specificRisks[jdx]
                    correl = min(correl, 0.99995)
                    correl = max(correl, 0.7)
                     
                    # Compute specific correlation and fill in the transposed element too
                    if subid2 not in exclList and subid1 not in exclList:
                        # Only calculate those not in the exclusion list
                        subCovMatrix[i,j+1+i] = correl  * specificRisks[idx] * specificRisks[jdx]
                        subCovMatrix[j+1+i,i] = subCovMatrix[i,j+i+1]
                    else:
                        # For those excluded, only calculate the specific covariance
                        targetList = [subid1, subid2]
                        srDataChunk = numpy.take(srData, [idx, jdx], axis=0)
                        specificCov = Utilities.compute_covariance(srDataChunk,
                                                                   weights=weights, deMean=self.deMeanFlag, axis=1)
                        for lag in range(1,self.deltaLag+1):
                            lagCov = Utilities.compute_covariance(srDataChunk,
                                                                  weights=weights, deMean=self.deMeanFlag, axis=1, lag=lag)
                            specificCov += (1.0 - float(lag)/(float(self.deltaLag)+1.0)) * \
                                (lagCov + numpy.transpose(lagCov))
                        specificCov *= 252.0
                        origLoggingLevel = logging.getLogger().getEffectiveLevel()
                        logging.getLogger().setLevel(logging.ERROR)

                        # Final checks
                        specificCov = Utilities.forcePositiveSemiDefiniteMatrix(
                                specificCov, min_eigenvalue=1e-8, quiet=True)
                        logging.getLogger().setLevel(origLoggingLevel)
                        (d, specificCorr) = Utilities.cov2corr(specificCov, fill=True)
                        specificCorr = (2.0 / 3.0) * specificCorr + (1.0 / 3.0)
                        # The specificCorr is a 2x2 Matrix
                        subCovMatrix[i,j+1+i] = specificCorr[0, 1] * specificRisks[idx] * specificRisks[jdx]
                        subCovMatrix[j+1+i,i] = subCovMatrix[i,j+i+1]
                
            # Have to back out specific vars again for consistency
            (specRisk, corrMatrix) = Utilities.cov2corr(subCovMatrix, fill=True)

            # Now force direct cloning of some pairs where required
            for slv in cloneList:
                mst = hardCloneMap[slv]
                if (slv in assetIdxMap) and (mst in assetIdxMap):
                    i_loc = sortedIdxMap[assetIdxMap[slv]]
                    j_loc = sortedIdxMap[assetIdxMap[mst]]
                    specRisk[i_loc] = specRisk[j_loc]
                    corrMatrix[i_loc, j_loc] = 0.99995
                    corrMatrix[j_loc, i_loc] = 0.99995
                    # Match correlations between clone and other assets
                    # with those for the master
                    for sid in subIssueList:
                        if (sid != slv) and (sid != mst):
                            k_loc = sortedIdxMap[assetIdxMap[sid]]
                            corrMatrix[i_loc, k_loc] = corrMatrix[k_loc, j_loc]
                            corrMatrix[k_loc, i_loc] = corrMatrix[i_loc, k_loc]

            # Force positive semi-definiteness of the correlation matrix
            corrMatrix = Utilities.forcePositiveSemiDefiniteMatrix(corrMatrix,
                    min_eigenvalue=1e-8, quiet=True)
            (dummy, corrMatrix) = Utilities.cov2corr(corrMatrix, fill=True)

            # Go back round and fill in the covariance matrix dictionary entries
            for (i,idx) in enumerate(sortedIndices):
                subid1 = assets[idx]
                specificVars[idx] = specRisk[i] * specRisk[i]
                if subid1 not in specificCovMap:
                    specificCovMap[subid1] = dict()
                for (j,jdx) in enumerate(sortedIndices[i+1:]):
                    subid2 = assets[jdx]
                    if subid1 not in specificCovMap:
                        specificCovMap[subid1] = dict()
                    specificCovMap[subid1][subid2] = corrMatrix[i,j+i+1] * specRisk[i] * specRisk[j+i+1]
                    if subid2 not in specificCovMap:
                        specificCovMap[subid2] = dict()
                    specificCovMap[subid2][subid1] = specificCovMap[subid1][subid2]

        return (specificVars, specificCovMap)

    def computeSpecificCovariances(self, assets, srData, weights,
                        assetGroupingsMap, specificRisks, hardCloneMap):
        # Compute specific covariances
        assetIdxMap = dict([(j,i) for (i,j) in enumerate(assets)])
        specificCovMap = dict()
        self.log.info("Computing correlations for %d blocks of assets", 
                        len(assetGroupingsMap))
        for (groupId, subIssueList) in assetGroupingsMap.items():

            # Get cloning info
            cloneList = [n for n in subIssueList if n in hardCloneMap]
            subIdx = [assetIdxMap[n] for n in subIssueList]
            subIdxMap = dict(zip(subIdx, list(range(len(subIdx)))))

            # Build initial covariance matrix
            srDataChunk = numpy.take(srData, subIdx, axis=0)
            specificCov = Utilities.compute_covariance(srDataChunk,
                        weights=weights, deMean=self.deMeanFlag, axis=1)
            for lag in range(1,self.deltaLag+1):
                lagCov = Utilities.compute_covariance(srDataChunk,
                        weights=weights, deMean=self.deMeanFlag, axis=1, lag=lag)
                specificCov += (1.0 - float(lag)/(float(self.deltaLag)+1.0)) * \
                        (lagCov + numpy.transpose(lagCov))
            specificCov *= 252.0
            origLoggingLevel = logging.getLogger().getEffectiveLevel()
            logging.getLogger().setLevel(logging.ERROR)

            # Check for non-PSD
            specificCov = Utilities.forcePositiveSemiDefiniteMatrix(
                            specificCov, min_eigenvalue=1e-8, quiet=True)
            logging.getLogger().setLevel(origLoggingLevel)
            (d, specificCorr) = Utilities.cov2corr(specificCov, fill=True)

            # "Bayesian" adjustment
            specificCorr = (2.0 / 3.0) * specificCorr + (1.0 / 3.0)
            
            # Now force direct cloning of some pairs where required
            for slv in cloneList:
                mst = hardCloneMap[slv]
                if (slv in assetIdxMap) and (mst in assetIdxMap):
                    i_loc = subIdxMap[assetIdxMap[slv]]
                    j_loc = subIdxMap[assetIdxMap[mst]]
                    specificRisks[assetIdxMap[slv]] = specificRisks[assetIdxMap[mst]]
                    specificCorr[i_loc, j_loc] = 0.99995
                    specificCorr[j_loc, i_loc] = 0.99995
                    # Match correlations between clone and other assets
                    # with those for the master
                    for sid in subIssueList:
                        if (sid != slv) and (sid != mst):
                            k_loc = subIdxMap[assetIdxMap[sid]]
                            specificCorr[i_loc, k_loc] = specificCorr[k_loc, j_loc]
                            specificCorr[k_loc, i_loc] = specificCorr[i_loc, k_loc]

            specificCorr = Utilities.forcePositiveSemiDefiniteMatrix(
                    specificCorr, min_eigenvalue=1e-8, quiet=True)
            (dummy, specificCorr) = Utilities.cov2corr(specificCorr, fill=True)

            for (j, subid1) in enumerate(subIssueList):
                for (k, subid2) in enumerate(subIssueList):
                    if j == k:
                        continue
                    if subid1 not in specificCovMap:
                        specificCovMap[subid1] = dict()
                    specificCovMap[subid1][subid2] = specificCorr[j,k] * \
                            specificRisks[assetIdxMap[subid1]] * specificRisks[assetIdxMap[subid2]]
        specificVars = specificRisks * specificRisks
        return (specificVars, specificCovMap)

    def computeSpecificRisks(self, srMatrix, marketCaps, assetGroupingsMap,
                             restrict=None, scores=None, specialDRTreatment=False, multiMAD=False,
                             debuggingReporting=False, nOkRets=None, coinExclList=list(),
                             hardCloneMap=[]):
        """Compute the specific risk values for each asset given a
        TimeSeriesMatrix of specific returns, an array of market caps,
        and a list of lists containing SubIssues that are 'linked'.
        Assumes the same parameters (history, half-life, etc) that are
        used for specific risk computation are also used for specific
        covariance computation.
        CoinExclList is a list of sids that are part of the 
        assetGroupingsMap but we do not want them to do the cointegration
        specific risk calculation. Yet, they should exist in the specific 
        risk covariance.
        Returns an array of specific risks for each SubIssue and a 
        dictionary of dictionaries, mapping SubIssues to maps of
        linked SubIssues to their specific covariances.
        """
        self.log.debug('computeSpecificRisk: begin')

        # Compute specific variances
        (srData, weights, weightSums) = self.preProcessSpecificReturns(
            srMatrix, marketCaps, restrict, multiMAD)
        specificVars = BrilliantSpecificRisk2009.computeSpecificRiskInternal(
            self, srData, weights, weightSums, True)

        # Compute specific covariances - either using cointegration...
        if specialDRTreatment:
            (specificVars, specificCovMap) = self.computeCointegratedCovariances(
                    srMatrix.assets, srData, assetGroupingsMap, specificVars, scores,
                    weights, coinExclList, hardCloneMap)
        # ... or directly
        else:
            specificRisks = ma.sqrt(specificVars)
            (specificVars, specificCovMap) = self.computeSpecificCovariances(
                    srMatrix.assets, srData, numpy.array(weights), assetGroupingsMap,
                    specificRisks, hardCloneMap)

        # Output ISC info for debugging
        if debuggingReporting:
            outfile = 'tmp/isc-%s.csv' % srMatrix.dates[0]
            outfile = open(outfile, 'w')
            outfile.write('GID,SID,Name,Score,Risk,SID,Name,Score,Risk,Corr,TE,\n')
            assetIdxMap = dict([(j,i) for (i,j) in enumerate(srMatrix.assets)])
            # Build correlation matrix and compute tracking error between pairs
            for (groupId, subIssueList) in assetGroupingsMap.items():
                if scores is not None:
                    score = scores[groupId]
                else:
                    score = [0] * len(subIssueList)
                for (j, subid1) in enumerate(subIssueList):
                    for (k, subid2) in enumerate(subIssueList):
                        if j >= k:
                            continue
                        corr = specificCovMap[subid1][subid2] / \
                                (numpy.sqrt(specificVars[assetIdxMap[subid1]] * specificVars[assetIdxMap[subid2]]))
                        TE = numpy.sqrt((specificVars[assetIdxMap[subid1]]) + \
                                (specificVars[assetIdxMap[subid2]]) - \
                                2.0 * specificCovMap[subid1][subid2])
                        if (subid1 in srMatrix.nameDict) and (subid2 in srMatrix.nameDict):
                            outfile.write('%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,\n' % (groupId, \
                                    subid1.getSubIDString(), srMatrix.nameDict[subid1].replace(',',''),
                                    score[j], numpy.sqrt(specificVars[assetIdxMap[subid1]]),
                                    subid2.getSubIDString(), srMatrix.nameDict[subid2].replace(',',''),
                                    score[k], numpy.sqrt(specificVars[assetIdxMap[subid2]]), corr, TE))
            outfile.close()
        specificRisks = ma.sqrt(specificVars)
        self.log.info('Specific Risk bounds after ISC: [%.3f, %.3f], Mean: %.3f',
                min(specificRisks), max(specificRisks), ma.average(specificRisks))
        if debuggingReporting:
            sidList = [sid.getSubIDString() for sid in srMatrix.assets]
            fname = 'tmp/specificRisks.csv'
            Utilities.writeToCSV(specificRisks, fname, rowNames=sidList)
        self.log.debug('computeSpecificRisk: end')
        return (specificVars, specificCovMap)

class CompositeCovarianceMatrix2012:
    """Composite covariance matrix.
    """
    # XXX New stuff marker
    def __init__(self, fullCovParameters, varParameters, corrParameters):
        self.log = logging.getLogger('RiskCalculator.CompositeCovarianceMatrix')
        self.subCovariances = dict()
        self.fullCovParameters = fullCovParameters
        self.varParameters = varParameters
        self.corrParameters = corrParameters

        # Enhancements: aggregating multiple covariance matrices
        (self.useTransformMatrix, self.offDiagScaleFactor) = \
                        self.fullCovParameters.getCovarianceComposition()

    def configureSubCovarianceMatrix(self, position, factorCov=None, rp=None):
        """Fine-tune the covariance matrix for the 'block' 
        corresponding to the given position. (starts at zero) 
        eg, position=0 refers to the first factor returns series 
        provided to computeFactorCovarianceMatrix().  
        If factorCov is specified, this pre-computed covariance 
        matrix will be used.  If a RiskParameters object rp is provided,
        it will be used to compute this factor covariance matrix.
        """
        self.subCovariances[position] = dict()
        self.subCovariances[position]['fcov'] = factorCov
        return True

    def cov_main(self, fr, covParameters, matrixType='cov'):
        """Core routine for computing a factor covariance matrix.
        """
        # Set up parameters
        self.halfLife = covParameters.getCovarianceHalfLife()
        (self.minObs, self.maxObs) = covParameters.getCovarianceSampleSize()
        self.deMeanFlag = covParameters.useDeMeanedCovariance()
        self.equalWeightFlag = covParameters.useEqualWeightedCovariance()

        # Enhancements: autocorrelation lag
        self.NWLag = covParameters.getCovarianceNeweyWestLag()

        # Enhancements: Dynamic Volatility Adjustment (DVA)
        (self.DVAWindow, self.DVAType, self.DVAUpperBound, self.DVALowerBound,
                self.downweightEnds) = covParameters.getCovarianceDVAOptions()

        if fr.shape[1] > self.maxObs:
            fr = fr[:, 0:self.maxObs]
        nPeriods = fr.shape[1]
        numLags = self.NWLag
        self.log.info('Using %d factors by %d periods', fr.shape[0], nPeriods)

        # Calculate weights
        factorReturnWeights = Utilities.computeExponentialWeights(
                self.halfLife, nPeriods, self.equalWeightFlag)

        # Pre-process (clip, DVA, etc) factor returns
        frFull = numpy.array(fr, copy=True)
        if self.DVAWindow is not None:
            self.log.debug('Applying Dynamic Volatility Adjustment (DVA)')
            if self.DVAType == 'spline':
                frFull = Utilities.spline_dva(fr, self.DVAWindow,
                        upperBound=self.DVAUpperBound, lowerBound=self.DVALowerBound,
                        downWeightEnds=self.downweightEnds)

        # Check for factor returns consisting entirely of zeros
        frFullMasked = ma.masked_where(abs(frFull*factorReturnWeights) < 1e-10, frFull)
        frNonZeros = (ma.getmaskarray(frFullMasked)==0.0)
        sumFactors = ma.sum(frNonZeros, axis=1).astype(numpy.float) / fr.shape[1]
        sumFactors = ma.where(sumFactors < 0.01, 0.0, sumFactors)
        goodFactorsIdx = numpy.flatnonzero(sumFactors)

        frData = ma.take(frFull, goodFactorsIdx, axis=0).filled(0.0)
        if len(goodFactorsIdx) != frFull.shape[0]:
            self.log.info('%d out of %d factors are effectively zero',
                    frFull.shape[0]-len(goodFactorsIdx), frFull.shape[0])
            self.log.info('Dimension of cov matrix reduced to %d for calculation',
                    len(goodFactorsIdx))

        # Calculate lag zero covariance matrix
        factorCov = Utilities.compute_covariance(frData,
                weights=factorReturnWeights, deMean=self.deMeanFlag, axis=1)

        # Perform Newey-West using given number of lags
        self.log.info('Number of autocorrelation lags (Static): %d', numLags)
        ARadj = numpy.zeros(factorCov.shape, float)
        for lag in range(1,numLags+1):
            lagCov = Utilities.compute_covariance(frData,
                    weights=factorReturnWeights, deMean=self.deMeanFlag, axis=1, lag=lag)
            ARadj += (1.0 - float(lag)/(float(numLags)+1.0)) * \
                    (lagCov + numpy.transpose(lagCov))
        factorCov += ARadj

        # In case python's numerical accuracy falls short
        factorCov = (factorCov + numpy.transpose(factorCov)) / 2.0

        # If we've computed a reduced covariance matrix, paste it
        # back into the full-sized variant
        if frData.shape[0] < fr.shape[0]:
            fullFactorCov = numpy.zeros((fr.shape[0], fr.shape[0]), float)
            for (i, id) in enumerate(goodFactorsIdx):
                for (j, jd) in enumerate(goodFactorsIdx):
                    fullFactorCov[id,jd] = factorCov[i,j]
            factorCov = fullFactorCov

        # Return the relevant object
        if matrixType=='std' or matrixType=='cor':
            (stdVector, factorCor) = Utilities.cov2corr(factorCov)
            if matrixType=='std':
                return stdVector
            else:
                return factorCor
        else:
            (stdVector, factorCov) = Utilities.cov2corr(
                    factorCov, returnCov=True)
            return factorCov

    def computeFactorCovarianceMatrix(self, *frMatrix):
        """Compute a 'composite' factor covariance matrix.  Each 
        input is a TimeSeriesMatrix of factor returns belonging to
        the corresponding block of the covariance matrix.  Even if
        the covariances for a particular block have been pre-computed
        (eg. currencies), its factor returns series must still be 
        provided.  The first column in each TimeSeriesMatrix of 
        observations should be the most recent.
        """
        self.log.debug('computeFactorCovarianceMatrix: begin')
        #self.useTransformMatrix = True
        # Aggregate all factor returns
        firstTime = True
        for fr in frMatrix:
            # Returns timing adjustment
            if hasattr(fr, 'adjust'):
                self.log.info('Adjusting returns')
                frData = fr.data + fr.adjust
            # Or not
            else:
                frData = ma.array(fr.data, copy=True)

            # Long mean adjustment
            if hasattr(fr, 'mean'):
                frData = numpy.transpose(numpy.transpose(frData) - fr.mean)

            if firstTime:
                factorReturns = ma.array(frData, copy=True)
                firstTime = False
            else:
                assert(fr.data.shape[1]==prevSize)
                factorReturns = ma.concatenate(
                        [factorReturns, frData], axis=0)
            prevSize = frData.shape[1]

        # Compute full-size covariance matrix
        if len(frMatrix) > 1:
            self.log.debug('Computing full size correlation matrix')
            if self.useTransformMatrix:
                self.log.info('Using matrix transform')
            else:
                self.log.info('Using matrix cut-and-paste')

            fullCorrelationMatrix = self.cov_main(factorReturns,
                    self.fullCovParameters, matrixType='cor')
        else:
            self.useTransformMatrix = False
            fullCorrelationMatrix = ma.zeros(
                    (factorReturns.shape[0], factorReturns.shape[0]), float)
        fullStdVector = ma.zeros((factorReturns.shape[0]), float)

        # Initialise the transformation matrix
        covarianceTransformMatrix = ma.zeros(
                        (factorReturns.shape[0], factorReturns.shape[0]), float)
        overlayMatrix = ma.zeros(
                        (factorReturns.shape[0], factorReturns.shape[0]), float)

        # Loop through all the factor 'blocks'
        idx = 0
        for i in range(len(frMatrix)):
            self.log.info('Computing sub covariance matrix, block %d', i)
            fr = frMatrix[i]
            # Returns timing adjustment
            if hasattr(fr, 'adjust'):
                self.log.info('Adjusting returns')
                frData = fr.data + fr.adjust
            # Or not
            else:
                frData = ma.array(fr.data, copy=True)
            # Long mean adjustment
            if hasattr(fr, 'mean'):
                frData = numpy.transpose(numpy.transpose(frData) - fr.mean)

            n = idx + fr.data.shape[0]

            # Check if using a pre-computed cov
            sub = self.subCovariances.get(i, None)
            if sub is not None:
                self.log.info('Loading pre-computed block')
                subCovarianceBlock = sub.get('fcov', None)
                (subStdVector, subCorrelationBlock) = Utilities.cov2corr(
                                                                subCovarianceBlock)
            else:
                # Compute correlations
                self.log.info('Computing correlation sub-block')
                subCorrelationBlock = self.cov_main(frData, self.corrParameters,
                                                    matrixType='cor')
                # Compute variances
                self.log.info('Computing variance sub-block')
                subStdVector = self.cov_main(frData, self.varParameters,
                                                    matrixType='std')

                if hasattr(self, 'shrinkMatrix'):
                    self.log.info('Shrinking correlation matrix by %.2f', self.shrinkFactor)
                    subCorrelationBlock = (self.shrinkFactor * self.shrinkMatrix) + \
                            ((1.0 - self.shrinkFactor) * subCorrelationBlock)

            # Fill in relevant parts of full-sized matrices
            overlayMatrix[idx:n,idx:n] = subCorrelationBlock
            fullStdVector[idx:n] = subStdVector

            if self.useTransformMatrix:
                # Create part of the transformation matrix
                (s,x) = linalg.eigh(ma.filled(subCorrelationBlock, 0.0))
                t1 = ma.dot(ma.dot(x, ma.diag(ma.sqrt(s))), ma.transpose(x))

                # Create second part of the transformation matrix
                (s,x) = linalg.eigh(ma.filled(fullCorrelationMatrix[idx:n,idx:n], 0.0))
                t2 = ma.dot(ma.dot(x, ma.diag(ma.sqrt(s))), ma.transpose(x))

                # Find Procrustes rotation matrix
                m = ma.dot(ma.transpose(t1), t2)
                try:
                    (u, d, v) = linalg.svd(ma.filled(m, 0.0), full_matrices=True)
                    Q = numpy.dot(u, v)
                except:
                    logging.warning('Linalg SVD routine failed for procrustes transform component')
                    logging.warning('Switching to more involved route')
                    mmT = ma.dot(m, ma.transpose(m))
                    mmT = 0.5 * (mmT + ma.transpose(mmT))
                    mTm = ma.dot(ma.transpose(m), m)
                    mTm = 0.5 * (mTm + ma.transpose(mTm))
                    (s, u) = linalg.eigh(ma.filled(mmT, 0.0))
                    evOrder = numpy.argsort(-s)
                    u = ma.take(u, evOrder, axis=1)
                    (s, v) = linalg.eigh(ma.filled(mTm, 0.0))
                    evOrder = numpy.argsort(-s)
                    v = ma.take(v, evOrder, axis=1)
                    Q = numpy.dot(u, ma.transpose(v))
                t1 = numpy.dot(t1, Q)

                # Compute block of transform matrix
                t = Utilities.robustLinearSolver(ma.transpose(t1), ma.transpose(t2),
                        computeStats=False)

                # Insert transform block into the full-size matrix
                covarianceTransformMatrix[idx:n,idx:n] = ma.transpose(t.params)
            idx += fr.data.shape[0]

        # Compute composite correlation matrix
        if self.useTransformMatrix:
            # The "proper" transformed correlation matrix
            fullCorrelationMatrix = ma.dot(covarianceTransformMatrix,\
                    ma.dot(fullCorrelationMatrix,\
                    ma.transpose(covarianceTransformMatrix)))
            fullCorrelationMatrix = (fullCorrelationMatrix + ma.transpose(fullCorrelationMatrix)) / 2.0

        # Even if using the "proper" transform, directly paste the diagonal blocks
        # into the correlation matrix to eliminate differences due to machine error
        # or masking
        idx = 0
        for i in range(len(frMatrix)):
            fr = frMatrix[i]
            n = idx + fr.data.shape[0]
            fullCorrelationMatrix[idx:n,idx:n] = overlayMatrix[idx:n,idx:n]
            idx += fr.data.shape[0]

        # Combine variances with correlations
        factorCov = ma.transpose(ma.transpose(
            fullCorrelationMatrix*fullStdVector)*fullStdVector)

        # Clip small eigenvalues and reconstitute the cov matrix if necessary
        factorCov = factorCov.filled(0.0)
        if len(frMatrix) < 2:
            factorCov = Utilities.forcePositiveSemiDefiniteMatrix(
                    factorCov, min_eigenvalue=0.0)

        # Annualize
        factorCov *= 252.0

        self.log.info('Sum of cov matrix elements: %f', ma.sum(factorCov, axis=None))
        self.log.debug('computeFactorCovarianceMatrix: end')
        return factorCov

class ComputeSpecificRisk2012:
    """Specific risk computation used by the US3 models
    Exponential weighting, Newey West and DVA can be applied.
    Assets with short histories are shrunk towards a cross-sectional average.
    Linked assets have specific risk and correlation computed using either
    cointegration or the naive approach, depending on their level of cointegration
    Also contains some unused test code:
        1. Compute specific risk as average + asset specific components
        2. Test ISC using combination of cointegration and direct approach
    """
    def __init__(self, srParameters):
        self.log = logging.getLogger('RiskCalculator.BrilliantSpecificRisk')
        (self.minDeltaObs, self.maxDeltaObs) = \
                           srParameters.getCovarianceSampleSize()
        self.deltaHalfLife = srParameters.getCovarianceHalfLife()
        self.deltaLag = srParameters.getCovarianceNeweyWestLag()
        self.deMeanFlag = srParameters.useDeMeanedCovariance()
        self.equalWeightFlag = srParameters.useEqualWeightedCovariance()
        self.averageComponent = srParameters.useStructuredModel()
        self.clipBounds = srParameters.getClipBounds()

        # Enhancements: Dynamic Volatility Adjustment (DVA)
        (self.DVAWindow, self.DVAType, self.DVAUpperBound, self.DVALowerBound, self.downweightEnds) =\
                srParameters.getCovarianceDVAOptions()

    def computeSpecificVariance(self, srMatrix, estu, data,
                                nOkRets, rmgList, modelDB, clipVars=True):
        minSpecificVar = 0.0025 / 252.0
        maxSpecificVar = 4.0 / 252.0
        if srMatrix.data.shape[1] > self.maxDeltaObs:
            raise LookupError('Number of time periods, %d, is greater than max number of observations, %d' % \
                    (srMatrix.data.shape[1], self.maxDeltaObs))

        self.log.info('Specific return bounds: [%.3f, %.3f]',
                      ma.min(srMatrix.data, axis=None),
                      ma.max(srMatrix.data, axis=None))
        srData = ma.filled(srMatrix.data, 0.0)

        if self.averageComponent:
            # Divide assets into market cap buckets and compute each average bucket return
            lncaps = numpy.log(data.marketCaps)
            pctiles = (0.1, 0.1, 0.1, 0.1, 0.1)
            buckets = Utilities.generate_marketcap_buckets(lncaps, pctiles)
            avgSpecRet = numpy.zeros((len(buckets), srData.shape[1]), float)

            # Loop round mcap buckets
            self.log.info('Computing average specific return for %d buckets', len(buckets))
            for (i_bkt, idxList) in enumerate(buckets):
                if len(idxList) < 1:
                    continue
                # Pick out returns
                estu_wgt_vec = 0.01 * numpy.ones((len(data.marketCaps)), float)
                estu_idx = list(set(idxList).intersection(set(estu)))
                bucketRets = ma.take(srData, idxList, axis=0)
                bucketRets = Utilities.twodMAD(bucketRets, axis=0, nDev=self.clipBounds)
                # Compute weights used
                numpy.put(estu_wgt_vec, estu_idx, 1.0)
                wts = data.marketCaps * estu_wgt_vec
                wts = ma.take(wts, idxList, axis=0)
                # Compute weighted average return for bucket
                for (i_day, d) in enumerate(srMatrix.dates):
                    dayRet = bucketRets[:, i_day]
                    avgSpecRet[i_bkt, i_day] = numpy.average(dayRet, axis=0, weights=wts)
                if self.averageComponent:
                    for idx in idxList:
                        srData[idx, :] = srData[idx, :] - avgSpecRet[i_bkt, :]

            # Apply DVA to specific returns
            if self.DVAWindow is not None:
                self.log.debug('Applying Dynamic Volatility Adjustment (DVA)')
                if self.DVAType == 'spline':
                    avgSpecRet = Utilities.spline_dva(avgSpecRet, self.DVAWindow,
                            upperBound=self.DVAUpperBound, lowerBound=self.DVALowerBound)

        # Compute asset specific variance
        srDataClipped = Utilities.twodMAD(srData, axis=1, nDev=self.clipBounds)
        expWeights = Utilities.computeExponentialWeights(
                self.deltaHalfLife, srData.shape[1], self.equalWeightFlag)
        specificVars = Utilities.compute_covariance(numpy.array(srDataClipped),
                weights=expWeights, deMean=self.deMeanFlag, axis=1, varsOnly=True)

        # Compute variance of average component
        if self.averageComponent:
            # Do one NW lag correction for "residual" specific risk
            lag = 1
            lagVar = Utilities.compute_covariance(
                    srDataClipped, weights=expWeights, deMean=self.deMeanFlag,
                    axis=1, varsOnly=True, lag=lag)
            specificVars += (1.0 - float(lag)/(float(lag)+1.0)) * (2.0 * lagVar)

            # Compute daily variance of average specific risk
            self.log.info('Computing average specific variance')
            expWeights = Utilities.computeExponentialWeights(
                    self.deltaHalfLife, avgSpecRet.shape[1], self.equalWeightFlag)
            avSpecVar = Utilities.compute_covariance(numpy.array(avgSpecRet),
                    weights=expWeights, deMean=self.deMeanFlag, axis=1, varsOnly=True)

            # Perform Newey-West using given number of lags on average specific risk
            for lag in range(1,self.deltaLag+1):
                lagVar = Utilities.compute_covariance(numpy.array(avgSpecRet),
                        weights=expWeights, deMean=self.deMeanFlag,
                        axis=1, varsOnly=True, lag=lag)
                avSpecVar += (1.0 - float(lag)/(float(self.deltaLag)+1.0)) * (2.0 * lagVar)
        else:
            # Perform Newey-West using given number of lags
            for lag in range(1,self.deltaLag+1):
                lagVar = Utilities.compute_covariance(srDataClipped,
                        weights=expWeights, deMean=self.deMeanFlag,
                        axis=1, varsOnly=True, lag=lag)
                specificVars += (1.0 - float(lag)/(float(lag)+1.0)) * (2.0 * lagVar)

            avSpecVar = Utilities.Struct()
            avSpecVar.data = numpy.array(specificVars)
            avSpecVar.data = ma.masked_where(nOkRets<0.75, avSpecVar.data)
            avSpecVar.data = avSpecVar.data[:,numpy.newaxis]
            avSpecVar = Utilities.proxyMissingAssetReturnsV3(
                    rmgList, srMatrix.dates[0], avSpecVar, data, modelDB, robust=True,
                    gicsDate=datetime.date(2014,3,1))
            avSpecVar = avSpecVar.estimates[:,0]
            avSpecVar = numpy.clip(avSpecVar, minSpecificVar, maxSpecificVar)

        # Combine risk components
        hl = float(self.deltaHalfLife) / 2.0
        if self.averageComponent:
            for (i_bkt, idxList) in enumerate(buckets):
                for idx in idxList:
                    wt = 2.0**( -(nOkRets[idx]*srData.shape[1]) / hl)
                    svar = (1.0-wt) * specificVars[idx]
                    specificVars[idx] = svar + avSpecVar[i_bkt]
        else:
            for idx in range(len(specificVars)):
                wt = 2.0**( -(nOkRets[idx]*srData.shape[1]) / hl)
                specificVars[idx] = (1.0-wt) * specificVars[idx] + (wt * avSpecVar[idx])

        # Multiply values by number of annual periods
        # Truncate any really huge or really tiny values which remain
        if clipVars:
            etfList = []
            etfRsk = []
            for (idx, sid) in enumerate(srMatrix.assets):
                if data.assetTypeDict.get(sid, '') == 'NonEqETF':
                    etfList.append(idx)
                    etfRsk.append(specificVars[idx])
            specificVars = numpy.clip(specificVars, minSpecificVar, maxSpecificVar)
            if len(etfList) > 0:
                for (iLoc, idx) in enumerate(etfList):
                    specificVars[idx] = etfRsk[iLoc]

        # Multiply values by number of annual periods
        specificVars = specificVars * 252.0
        specificRisks = ma.sqrt(specificVars)
        self.log.info('Specific Risk bounds: [%.3f, %.3f], Mean: %.3f',
            min(specificRisks), max(specificRisks), ma.average(specificRisks))
        return specificVars

    def computeCointegratedCovariancesLegacy(self, assets, srData,
            assetGroupingsMap, specificVars, scoreDict):
        """Compute ISC information using cointegration-based
        techniques
        """
        self.log.info("Computing robust covariances for %d blocks of assets",
                len(assetGroupingsMap))
        # Initialise
        assetIdxMap = dict([(j,i) for (i,j) in enumerate(assets)])
        specificRisks = Utilities.screen_data(numpy.sqrt(specificVars), fill=True)
        specificCovMap = dict()
        # Get cointegration stats
        coefDict, dfPDict, eVarDict = Utilities.compute_cointegration_parameters_legacy(
                numpy.fliplr(srData), assetGroupingsMap, assets)
        # Loop round blocks of linked assets
        for (groupId, subIssueList) in assetGroupingsMap.items():
            # Assign scores to each asset and order from highest to lowest-scoring
            score = scoreDict[groupId]
            indices  = [assetIdxMap[n] for n in subIssueList]
            sortedIndices = [indices[j] for j in numpy.argsort(-score)]
            subCovMatrix = numpy.zeros((len(sortedIndices), len(sortedIndices)), float)
            # Loop round each combination of linked assets
            for (i,idx) in enumerate(sortedIndices):
                subid1 = assets[idx]
                subCovMatrix[i,i] = specificVars[idx]
                # Only need perform computation on half the matrix
                for (j,jdx) in enumerate(sortedIndices[i+1:]):
                    subid2 = assets[jdx]
                    coeff = coefDict[subid1][subid2]
                    coeff = numpy.clip(coeff, 0.8, 1.25)
                    epsVar = eVarDict[subid1][subid2]
                    # Compute new variance for linked asset
                    if i == 0:
                        specificVars[jdx] = (coeff * coeff * specificVars[idx]) + epsVar
                        specificRisks[jdx] = numpy.sqrt(specificVars[jdx])
                    # Compute pseudo-correlation between assets
                    correl = coeff * specificRisks[idx] / specificRisks[jdx]
                    correl = Utilities.screen_data(correl, fill=True)
                    correl = min(correl, 0.99995)
                    correl = max(correl, 0.7)
                    # Compute specific correlation and fill in the transposed element too
                    subCovMatrix[i,j+1+i] = correl  * specificRisks[idx] * specificRisks[jdx]
                    subCovMatrix[j+1+i,i] = subCovMatrix[i,j+i+1]

            # Force positive semi-definiteness of the correlation matrix
            subCovMatrix = Utilities.screen_data(subCovMatrix, fill=True)
            subCovMatrix = Utilities.forcePositiveSemiDefiniteMatrix(subCovMatrix,
                    min_eigenvalue=1e-8, quiet=True)
            # Have to back out specific vars again for consistency
            (specRisk, corrMatrix) = Utilities.cov2corr(subCovMatrix, fill=True)
            # Go back round and fill in the covariance matrix dictionary entries
            for (i,idx) in enumerate(sortedIndices):
                subid1 = assets[idx]
                specificVars[idx] = specRisk[i] * specRisk[i]
                if subid1 not in specificCovMap:
                    specificCovMap[subid1] = dict()
                for (j,jdx) in enumerate(sortedIndices[i+1:]):
                    subid2 = assets[jdx]
                    if subid1 not in specificCovMap:
                        specificCovMap[subid1] = dict()
                    specificCovMap[subid1][subid2] = corrMatrix[i,j+i+1] * specRisk[i] * specRisk[j+i+1]
                    if subid2 not in specificCovMap:
                        specificCovMap[subid2] = dict()
                    specificCovMap[subid2][subid1] = specificCovMap[subid1][subid2]
        return (specificVars, specificCovMap)


    def computeDirectCovariances(self, assets, srData, weights,
                                    assetGroupingsMap, specificRisks):
        # Compute specific covariances
        assetIdxMap = dict([(j,i) for (i,j) in enumerate(assets)])
        specificCovMap = dict()
        self.log.info("Computing correlations for %d blocks of assets",
                        len(assetGroupingsMap))
        for (groupId, subIssueList) in assetGroupingsMap.items():
            srDataChunk = numpy.take(srData,
                        [assetIdxMap[n] for n in subIssueList], axis=0)
            specificCov = Utilities.compute_covariance(srDataChunk,
                        weights=weights, deMean=self.deMeanFlag, axis=1)
            for lag in range(1,self.deltaLag+1):
                lagCov = Utilities.compute_covariance(srDataChunk,
                        weights=weights, deMean=self.deMeanFlag, axis=1, lag=lag)
                specificCov += (1.0 - float(lag)/(float(self.deltaLag)+1.0)) * \
                        (lagCov + numpy.transpose(lagCov))
            specificCov *= 252.0
            origLoggingLevel = logging.getLogger().getEffectiveLevel()
            logging.getLogger().setLevel(logging.ERROR)
            specificCov = Utilities.forcePositiveSemiDefiniteMatrix(
                            specificCov, min_eigenvalue=1e-8, quiet=True)
            logging.getLogger().setLevel(origLoggingLevel)
            (d, specificCorr) = Utilities.cov2corr(specificCov, fill=True)

            for (j, subid1) in enumerate(subIssueList):
                for (k, subid2) in enumerate(subIssueList):
                    if j == k:
                        continue
                    if subid1 not in specificCovMap:
                        specificCovMap[subid1] = dict()
                    specificCovMap[subid1][subid2] = specificCorr[j,k] * \
                            specificRisks[assetIdxMap[subid1]] * specificRisks[assetIdxMap[subid2]]
        return specificCovMap

    def outputISCData(self, srMatrix, assetGroupingsMap, scores, specificCovMap, specificVars):
        """Output ISC info for debugging
        """
        outfile = 'tmp/isc-%s.csv' % srMatrix.dates[0]
        outfile = open(outfile, 'w')
        outfile.write('GID,SID,Name,Score,Risk,SID,Name,Score,Risk,Corr,TE,\n')
        assetIdxMap = dict([(j,i) for (i,j) in enumerate(srMatrix.assets)])
        specificRisks = numpy.sqrt(specificVars)
        # Build correlation matrix and compute tracking error between pairs
        for (groupId, subIssueList) in assetGroupingsMap.items():
            if scores is not None:
                score = scores[groupId]
            else:
                score = [0] * len(subIssueList)
            for (j, subid1) in enumerate(subIssueList):
                jdx = assetIdxMap[subid1]
                for (k, subid2) in enumerate(subIssueList):
                    kdx = assetIdxMap[subid2]
                    if j >= k:
                        continue
                    corr = specificCovMap[subid1][subid2] / \
                            (specificRisks[jdx] * specificRisks[kdx])
                    TE = numpy.sqrt(specificVars[jdx] + specificVars[kdx] - \
                                (2.0 * specificCovMap[subid1][subid2]))
                    if (subid1 in srMatrix.nameDict) and (subid2 in srMatrix.nameDict):
                        outfile.write('%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,\n' % ( \
                                groupId, subid1.getSubIDString(),
                                srMatrix.nameDict[subid1].replace(',',''),
                                score[j], specificRisks[jdx],
                                subid2.getSubIDString(),
                                srMatrix.nameDict[subid2].replace(',',''),
                                score[k], specificRisks[kdx],
                                corr, TE))
        outfile.close()
        return

    def computeSpecificRisks(self, srMatrix, data, assetGroupingsMap,
                             rmgList, modelDB, nOkRets=None, restrict=None, scores=None,
                             debuggingReporting=False, hardCloneMap=None, excludeTypes=[],
                             gicsDate=None):
        """Compute the specific risk values for each asset given a
        TimeSeriesMatrix of specific returns, an array of market caps,
        and a list of lists containing SubIssues that are 'linked'.
        Assumes the same parameters (history, half-life, etc) that are
        used for specific risk computation are also used for specific
        covariance computation.
        Returns an array of specific risks for each SubIssue and a
        dictionary of dictionaries, mapping SubIssues to maps of
        linked SubIssues to their specific covariances.
        """
        self.log.debug('computeSpecificRisk: begin')
        if nOkRets is None:
            nOkRets = numpy.ones((srMatrix.data.shape[0]), float)

        # Compute specific variances
        specificVars = self.computeSpecificVariance(
            srMatrix, restrict, data, nOkRets, rmgList, modelDB)
        specificRisks = ma.sqrt(specificVars)

        (specificVars, specificCovMap) = self.computeCointegratedCovariancesLegacy(
                srMatrix.assets, ma.filled(srMatrix.data, 0.0), assetGroupingsMap, specificVars, scores)

        # Output ISC info for debugging
        if debuggingReporting:
            self.outputISCData(srMatrix, assetGroupingsMap, scores, specificCovMap, specificVars)

        specificRisks = ma.sqrt(specificVars)
        self.log.info('Specific Risk bounds after ISC: [%.3f, %.3f], Mean: %.3f',
                min(specificRisks), max(specificRisks), ma.average(specificRisks))
        if debuggingReporting:
            sidList = [sid.getSubIDString() for sid in srMatrix.assets]
            fname = 'tmp/specificRisks.csv'
            Utilities.writeToCSV(specificRisks, fname, rowNames=sidList)
        self.log.debug('computeSpecificRisk: end')
        return (specificVars, specificCovMap)

class ComputeSpecificRisk2015:
    """Specific risk code for US4 model variants.
    Mostly a cleaned up version of ComputeSpecificRisk2012, but contains
    some extra logic to exclude non-equity ETFs from being
    shrunk towards a cross-sectional average, as this can inflate their risks
    Otherwise...
    Computes specific risk from time-series of specific returns.
    Exponential weighting, Newey West and DVA can be applied.
    Assets with short histories are shrunk towards a cross-sectional average.
    Linked assets have specific risk and correlation computed using either
    cointegration or the naive approach, depending on their level of cointegration
    """
    def __init__(self, srParameters):
        self.log = logging.getLogger('RiskCalculator.BrilliantSpecificRisk')
        (self.minDeltaObs, self.maxDeltaObs) = \
                           srParameters.getCovarianceSampleSize()
        self.deltaHalfLife = srParameters.getCovarianceHalfLife()
        self.deltaLag = srParameters.getCovarianceNeweyWestLag()
        self.deMeanFlag = srParameters.useDeMeanedCovariance()
        self.equalWeightFlag = srParameters.useEqualWeightedCovariance()
        self.clipBounds = srParameters.getClipBounds()
        
        # Enhancements: Dynamic Volatility Adjustment (DVA)
        (self.DVAWindow, self.DVAType, self.DVAUpperBound, self.DVALowerBound, self.downweightEnds) =\
                srParameters.getCovarianceDVAOptions()

    def computeSpecificVariance(self, srMatrix, data,
                                nOkRets, rmgList, modelDB, gicsDate, clipVars=True, excludeTypes=[]):
        logging.info('Computing specific risk with %d observations for %d assets with %d day halflife',
                srMatrix.data.shape[1], srMatrix.data.shape[0], self.deltaHalfLife)
        minSpecificVar = 0.0025 / 252.0
        maxSpecificVar = 4.0 / 252.0
        if srMatrix.data.shape[1] > self.maxDeltaObs:
            raise LookupError('Number of time periods, %d, is greater than max number of observations, %d' % \
                    (srMatrix.data.shape[1], self.maxDeltaObs))
            
        self.log.info('Specific return bounds: [%.3f, %.3f]',
                      ma.min(srMatrix.data, axis=None),
                      ma.max(srMatrix.data, axis=None))
        srData = ma.filled(srMatrix.data, 0.0)
        
        # Compute asset specific variance
        srDataClipped = Utilities.twodMAD(srData, axis=1, nDev=self.clipBounds)
        expWeights = Utilities.computeExponentialWeights(
                self.deltaHalfLife, srData.shape[1], self.equalWeightFlag)
        specificVars = Utilities.compute_covariance(numpy.array(srDataClipped),
                weights=expWeights, deMean=self.deMeanFlag, axis=1, varsOnly=True)

        # Perform Newey-West using given number of lags
        for lag in range(1,self.deltaLag+1):
            lagVar = Utilities.compute_covariance(srDataClipped,
                    weights=expWeights, deMean=self.deMeanFlag,
                    axis=1, varsOnly=True, lag=lag)
            specificVars += (1.0 - float(lag)/(float(lag)+1.0)) * (2.0 * lagVar)

        avSpecVar = Utilities.Struct()
        avSpecVar.data = numpy.array(specificVars)
        avSpecVar.data = ma.masked_where(nOkRets<0.75, avSpecVar.data)
        avSpecVar.data = avSpecVar.data[:,numpy.newaxis]
        avSpecVar = Utilities.proxyMissingAssetReturnsV3(
                rmgList, srMatrix.dates[0], avSpecVar, data, modelDB, robust=True,
                gicsDate=gicsDate)
        avSpecVar = avSpecVar.estimates[:,0]
        avSpecVar = numpy.clip(avSpecVar, minSpecificVar, maxSpecificVar)
        
        # Combine risk components
        hl = float(self.deltaHalfLife) / 2.0
        nExclude = 0
        for idx in range(len(specificVars)):
            sid = srMatrix.assets[idx]
            if data.assetTypeDict.get(sid, '').lower() not in excludeTypes:
                wt = 2.0**( -(nOkRets[idx]*srData.shape[1]) / hl)
                specificVars[idx] = (1.0-wt) * specificVars[idx] + (wt * avSpecVar[idx])
            else:
                nExclude += 1
        if nExclude > 0:
            logging.info('%d assets of type %s excluded from specific risk proxying',
                    nExclude, excludeTypes)

        # Multiply values by number of annual periods
        # Truncate any really huge or really tiny values which remain
        if clipVars:
            etfList = []
            etfRsk = []
            for (idx, sid) in enumerate(srMatrix.assets):
                if data.assetTypeDict.get(sid, '').lower() in excludeTypes:
                    etfList.append(idx)
                    etfRsk.append(specificVars[idx])
            specificVars = numpy.clip(specificVars, minSpecificVar, maxSpecificVar)
            if len(etfList) > 0:
                for (iLoc, idx) in enumerate(etfList):
                    specificVars[idx] = etfRsk[iLoc]
            
        # Multiply values by number of annual periods
        specificVars = specificVars * 252.0
        specificRisks = ma.sqrt(specificVars)
        self.log.info('Specific Risk bounds: [%.3f, %.3f], Mean: %.3f',
            min(specificRisks), max(specificRisks), ma.average(specificRisks))
        return specificVars
        
    def computeCointegratedCovariancesLegacy(self, assets, srData,
            assetGroupingsMap, specificVars, scoreDict):
        """Compute ISC information using cointegration-based
        techniques
        """
        self.log.info("Computing robust covariances for %d blocks of assets",
                len(assetGroupingsMap))
        # Initialise
        assetIdxMap = dict([(j,i) for (i,j) in enumerate(assets)])
        specificRisks = Utilities.screen_data(numpy.sqrt(specificVars), fill=True)
        specificCovMap = dict()
        # Get cointegration stats
        coefDict, dfPDict, eVarDict = Utilities.compute_cointegration_parameters_legacy(
                numpy.fliplr(srData), assetGroupingsMap, assets)
        # Loop round blocks of linked assets
        for (groupId, subIssueList) in assetGroupingsMap.items():
            # Assign scores to each asset and order from highest to lowest-scoring
            score = scoreDict[groupId]
            indices  = [assetIdxMap[n] for n in subIssueList]
            sortedIndices = [indices[j] for j in numpy.argsort(-score)]
            subCovMatrix = numpy.zeros((len(sortedIndices), len(sortedIndices)), float)
            # Loop round each combination of linked assets
            for (i,idx) in enumerate(sortedIndices):
                subid1 = assets[idx]
                subCovMatrix[i,i] = specificVars[idx]
                # Only need perform computation on half the matrix
                for (j,jdx) in enumerate(sortedIndices[i+1:]):
                    subid2 = assets[jdx]
                    coeff = coefDict[subid1][subid2]
                    coeff = numpy.clip(coeff, 0.8, 1.25)
                    epsVar = eVarDict[subid1][subid2]
                    # Compute new variance for linked asset
                    if i == 0:
                        specificVars[jdx] = (coeff * coeff * specificVars[idx]) + epsVar
                        specificRisks[jdx] = numpy.sqrt(specificVars[jdx])
                    # Compute pseudo-correlation between assets
                    correl = coeff * specificRisks[idx] / specificRisks[jdx]
                    correl = Utilities.screen_data(correl, fill=True)
                    correl = min(correl, 0.99995)
                    correl = max(correl, 0.7)
                    # Compute specific correlation and fill in the transposed element too
                    subCovMatrix[i,j+1+i] = correl  * specificRisks[idx] * specificRisks[jdx]
                    subCovMatrix[j+1+i,i] = subCovMatrix[i,j+i+1]

            # Force positive semi-definiteness of the correlation matrix
            subCovMatrix = Utilities.screen_data(subCovMatrix, fill=True)
            subCovMatrix = Utilities.forcePositiveSemiDefiniteMatrix(subCovMatrix,
                    min_eigenvalue=1e-8, quiet=True)
            # Have to back out specific vars again for consistency
            (specRisk, corrMatrix) = Utilities.cov2corr(subCovMatrix, fill=True)
            # Go back round and fill in the covariance matrix dictionary entries
            for (i,idx) in enumerate(sortedIndices):
                subid1 = assets[idx]
                specificVars[idx] = specRisk[i] * specRisk[i]
                if subid1 not in specificCovMap:
                    specificCovMap[subid1] = dict()
                for (j,jdx) in enumerate(sortedIndices[i+1:]):
                    subid2 = assets[jdx]
                    if subid1 not in specificCovMap:
                        specificCovMap[subid1] = dict()
                    specificCovMap[subid1][subid2] = corrMatrix[i,j+i+1] * specRisk[i] * specRisk[j+i+1]
                    if subid2 not in specificCovMap:
                        specificCovMap[subid2] = dict()
                    specificCovMap[subid2][subid1] = specificCovMap[subid1][subid2]
        return (specificVars, specificCovMap)

    def outputISCData(self, srMatrix, assetGroupingsMap, scores, specificCovMap, specificVars):
        """Output ISC info for debugging
        """
        outfile = 'tmp/isc-%s.csv' % srMatrix.dates[0]
        outfile = open(outfile, 'w')
        outfile.write('GID,SID,Name,Score,Risk,SID,Name,Score,Risk,Corr,TE,\n')
        assetIdxMap = dict([(j,i) for (i,j) in enumerate(srMatrix.assets)])
        specificRisks = numpy.sqrt(specificVars)
        if not hasattr(srMatrix, 'nameDict'):
            srMatrix.nameDict = dict()
        # Build correlation matrix and compute tracking error between pairs
        for (groupId, subIssueList) in assetGroupingsMap.items():
            if scores is not None:
                score = scores[groupId]
            else:
                score = [0] * len(subIssueList)
            for (j, subid1) in enumerate(subIssueList):
                jdx = assetIdxMap[subid1]
                for (k, subid2) in enumerate(subIssueList):
                    kdx = assetIdxMap[subid2]
                    if j >= k:
                        continue
                    corr = specificCovMap[subid1][subid2] / \
                            (specificRisks[jdx] * specificRisks[kdx])
                    TE = numpy.sqrt(specificVars[jdx] + specificVars[kdx] - \
                                (2.0 * specificCovMap[subid1][subid2]))
                    outfile.write('%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,\n' % ( \
                            groupId, subid1.getSubIDString() if not isinstance(subid1, str) else subid1,
                            srMatrix.nameDict.get(subid1, '').replace(',',''),
                            score[j], specificRisks[jdx],
                            subid2.getSubIDString() if not isinstance(subid2, str) else subid2,
                            srMatrix.nameDict.get(subid2, '').replace(',',''),
                            score[k], specificRisks[kdx],
                            corr, TE))
        outfile.close()
        return

    def computeSpecificRisks(self, srMatrix, data, assetGroupingsMap,
                             rmgList, modelDB, nOkRets=None, scores=None,
                             debuggingReporting=False, hardCloneMap=None, excludeTypes=[],
                             gicsDate=datetime.date(2014,3,1)):
        """Compute the specific risk values for each asset given a
        TimeSeriesMatrix of specific returns, an array of market caps,
        and a list of lists containing SubIssues that are 'linked'.
        Assumes the same parameters (history, half-life, etc) that are
        used for specific risk computation are also used for specific
        covariance computation.
        Returns an array of specific risks for each SubIssue and a 
        dictionary of dictionaries, mapping SubIssues to maps of
        linked SubIssues to their specific covariances.
        """
        self.log.debug('computeSpecificRisk: begin')
        if nOkRets is None:
            nOkRets = numpy.ones((srMatrix.data.shape[0]), float)
         
        # Compute specific variances
        specificVars = self.computeSpecificVariance(
            srMatrix, data, nOkRets, rmgList, modelDB, gicsDate, excludeTypes=excludeTypes)
        specificRisks = ma.sqrt(specificVars)
         
        # Compute specific covariances
        (specificVars, specificCovMap) = self.computeCointegratedCovariancesLegacy(
                srMatrix.assets, ma.filled(srMatrix.data, 0.0), assetGroupingsMap, specificVars, scores)
             
        # Output ISC info for debugging
        if debuggingReporting:
            self.outputISCData(srMatrix, assetGroupingsMap, scores, specificCovMap, specificVars)
         
        specificRisks = ma.sqrt(specificVars)
        self.log.info('Specific Risk bounds after ISC: [%.3f, %.3f], Mean: %.3f',
                min(specificRisks), max(specificRisks), ma.average(specificRisks))
        if debuggingReporting:
            sidList = [sid.getSubIDString() if not isinstance(sid, str) else sid for sid in srMatrix.assets]
            fname = 'tmp/specificRisks.csv'
            Utilities.writeToCSV(specificRisks, fname, rowNames=sidList)
        self.log.debug('computeSpecificRisk: end')
        return (specificVars, specificCovMap)
