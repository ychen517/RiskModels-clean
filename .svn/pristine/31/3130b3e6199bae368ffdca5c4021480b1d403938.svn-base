
import logging
import numpy.ma as ma
import numpy
import scipy.stats as stats
from scipy.stats import norm as Gaussian
import time
import datetime
import pickle as pickle
import warnings
try:
    import  bottleneck as bn
    hasbn = True
except ImportError:
    hasbn = False
    pass
from riskmodels import Matrices
from riskmodels import Utilities
from riskmodels import AssetProcessor
from riskmodels import Classification

class Outliers:

    def __init__(self,
                 OutlierParameters=dict(),
                 debugOutput=False,
                 forceRun=False,
                 downweightEnds=True,
                 industryClassificationDate=datetime.date(2014,3,1)):

        """ Generic class for time-series regression
        """
        self.OutlierParameters = OutlierParameters      # Important regression parameters
        self.debugOutput = debugOutput                  # Extra reporting
        self.forceRun = forceRun                        # Override a few failsafes
        self.downweightEnds = downweightEnds
        self.gicsDate = industryClassificationDate
        self.mad_const = Gaussian.ppf(3/4.)
        self.TOL = 1.0e-15

    def getNBounds(self):
        return self.OutlierParameters.get('nBounds', [8.0, 8.0, 3.0, 3.0])

    def getMethod(self):
        return self.OutlierParameters.get('method', 'MAD')

    def getZeroTolerance(self):
        return self.OutlierParameters.get('zeroTolerance', 0.0)

    def getShrinkage(self):
        return self.OutlierParameters.get('shrink', False)

    def outputStats(self, data, bf='BEFORE', estu=None, suppressOutput=False):
        # Output new limits
        if estu is not None:
            skewness = stats.skew(ma.take(data, estu, axis=0), axis=None)
        else:
            skewness = stats.skew(data, axis=None)
        if suppressOutput:
            logging.debug('Data bounds %s 2D MADing: (Min, Mean, Max, Skew) [%.3f, %.3f, %.3f, %.3f]' % \
                    (bf, ma.min(data, axis=None), ma.average(data, axis=None),
                        ma.max(data, axis=None), skewness))
        else:
            logging.info('Data bounds %s 2D MADing: (Min, Mean, Max, Skew) [%.3f, %.3f, %.3f, %.3f]' % \
                    (bf, ma.min(data, axis=None), ma.average(data, axis=None),
                        ma.max(data, axis=None), skewness))
        return

    def outputReturnsForDebugging(self, outfileName, returns):
        if self.debugOutput:
            retOutFile = 'tmp/%s' % outfileName
            Utilities.writeToCSV(returns, retOutFile)
        return

    def getBounds(self, data, method, nDev, weights=None):
        # Compute upper and lower bounds
        if method.lower() == 'iqr':
            bounds = Utilities.prctile(data, [25.0, 75.0])
            iqr = bounds[1] - bounds[0]
            lo = bounds[0] - (nDev[0] * iqr)
            up = bounds[1] + (nDev[1] * iqr)
            return lo, up
        if method.lower() == 'idr':
            bounds = Utilities.prctile(data, [10.0, 90.0])
            idr = bounds[1] - bounds[0]
            lo = bounds[0] - (nDev[0] * idr)
            up = bounds[1] + (nDev[1] * idr)
            return lo, up
        raise Exception('Unsupported method')

    def getMADBounds(self, data):
        if not hasbn:
            # To do - code for weighted median
            loc = ma.median(data, axis=0)
            scale = ma.median(ma.fabs(data - loc)/self.mad_const, axis=0)
            return loc, scale
        else:
            data2 = ma.filled(data, numpy.nan)
            loc2 = bn.nanmedian(data2, axis=0)
            scale2 = bn.nanmedian(numpy.fabs(data2 - loc2)/self.mad_const, axis=0)
            loc2 = ma.masked_array(loc2, mask=(loc2 == numpy.nan))
            scale2 = ma.masked_array(scale2, mask=(scale2 == numpy.nan))
            return loc2, scale2

    def computeOutlierBoundsInnerIQR(self, data, idxList, estuArray, weights):
        # Initialise parameters
        lower = []
        upper = []
        goodIdx = []
        
        if self.method.lower() not in ('iqr', 'idr'):
            raise Exception('Unsupported methods')

        for j in idxList:

            # Pick out column of data for MADing
            col = data[:,j]
            col = ma.masked_where(estuArray==False, col)
            missingIdx = list(numpy.flatnonzero(ma.getmaskarray(col)))

            # Check for overabundance of zeros
            zero_idx = ma.where(col==0.0)[0]
            zeroCol = numpy.zeros((len(col)), int)
            if len(zero_idx) > self.zeroTolerance * (len(col)-len(missingIdx)):
                # If too many zeros, mask some
                nZero = int(self.zeroTolerance * (len(col)-len(missingIdx)))
                zeroMaskIdx = zero_idx[nZero:]
                numpy.put(zeroCol, zeroMaskIdx, 1)
                col = ma.masked_where(zeroCol, col)
                missingIdx.extend(zeroMaskIdx)

            if len(missingIdx) < len(col)-1:
                goodIdx.append(j)
                okIdx = numpy.flatnonzero(ma.getmaskarray(col)==0)
                okData = ma.filled(ma.take(col, okIdx, axis=0), 0.0)
                wts = numpy.take(weights, okIdx, axis=0)

                # Compute upper and lower bounds
                (lo, up) = self.getBounds(okData, self.method, self.nBounds)
                lower.append(lo)
                upper.append(up)

        return goodIdx, lower, upper

    def computeOutlierBoundsInnerMAD(self, dataIn, idxList=None, estuArray=None, weights=None):
        self.method = self.getMethod()
        self.zeroTolerance = self.getZeroTolerance()
        data = ma.array(dataIn, copy=True)
        # Python 3 fix here
        data = ma.where(abs(data)<self.TOL, 0.0, data)
        if idxList is None:
            idxList = list(range(data.shape[1]))
        if estuArray is None:
            estuArray = numpy.array([True] * data.shape[0])
        if weights is None:
            weights = numpy.ones((data.shape[0]), float)
        if self.method.lower() in ('iqr', 'idr'):
            raise Exception('Unsupported methods')

        if self.zeroTolerance == 0.0:
            subdata = data[:,idxList]
            tmpEstu = numpy.repeat(estuArray[:,numpy.newaxis], subdata.shape[1],1) #Repeat estuArray as columns
            subdata[tmpEstu == False] = ma.masked
            mask = ma.getmaskarray(subdata)
            missingcnt = numpy.sum(mask, axis=0)
            zeroflag = (subdata == 0.0)
            # zerocnt = ma.sum        (zeroflag, axis = 0)
            subdata[zeroflag] = ma.masked
            missingcnt = ma.sum(ma.getmaskarray(subdata), axis=0)

            subdata_idx = ma.where(missingcnt < subdata.shape[0] - 1)[0]
            if len(subdata_idx) < 1:
                return [],[],[]
            subdata = subdata[:,subdata_idx]
            goodIdx = ma.array(idxList)[subdata_idx]
            dataCentre, dataScale = self.getMADBounds(subdata)
            return goodIdx, dataCentre, dataScale
        else:
            if len(idxList)>50:
                warnings.warn("This section code is slow. we may want to rewrite it !!!")
            dataCentre = []
            dataScale = []
            goodIdx = []
            for j in idxList:

                # Pick out column of data for MADing
                col = data[:,j]
                col = ma.masked_where(estuArray==False, col)
                missingIdx = list(numpy.flatnonzero(ma.getmaskarray(col)))

                # Check for overabundance of zeros
                zero_idx = ma.where(col==0.0)[0]
                zeroCol = numpy.zeros((len(col)), int)
                if len(zero_idx) > self.zeroTolerance * (len(col)-len(missingIdx)):
                    # If too many zeros, mask some
                    nZero = int(self.zeroTolerance * (len(col)-len(missingIdx)))
                    zeroMaskIdx = zero_idx[nZero:]
                    numpy.put(zeroCol, zeroMaskIdx, 1)
                    col = ma.masked_where(zeroCol, col)
                    missingIdx.extend(zeroMaskIdx)

                if len(missingIdx) < len(col)-1:
                    goodIdx.append(j)
                    okIdx = numpy.flatnonzero(ma.getmaskarray(col)==0)
                    okData = ma.filled(ma.take(col, okIdx, axis=0), 0.0)
                    # Compute upper and lower bounds
                    (loc, scale) = self.getMADBounds(okData)
                    dataCentre.append(loc)
                    dataScale.append(scale)
            return goodIdx, dataCentre, dataScale

    def computeOutlierBounds(self, data, idxList=None, estu=None, downweightEnds=False):
        """ Applies either IQR or MAD bounds method to a data array per column
        Returns a matrix of scaling values for the data
        """
        if idxList is None:
            idxList = list(range(data.shape[1]))

        # Sort out estu assets
        if estu is None:
            estu = list(range(data.shape[0]))
        estuArray = Matrices.allMasked((data.shape[0],), bool)
        ma.put(estuArray, estu, True)

        # Downweight ends of data if necessary
        if downweightEnds:
            weights = Utilities.computePyramidWeights(20, 20, data.shape[0])
        else:
            weights = numpy.ones((data.shape[0]), float)
        logging.debug('Outlier method: %s, bounds: %s', self.method, self.nBounds)

        if self.method.lower() in ('iqr', 'idr'):
            goodIdx, lower, upper = self.computeOutlierBoundsInnerIQR(data, idxList, estuArray, weights)
        else:
            goodIdx, dataCentre, dataScale = self.computeOutlierBoundsInnerMAD(data, idxList, estuArray, weights) 

        # Initialise return data
        retVal = Utilities.Struct()
        retVal.outlierArray = Matrices.allMasked(data.shape)
        retVal.lowerBounds = Matrices.allMasked((data.shape[1]))
        retVal.upperBounds = Matrices.allMasked((data.shape[1]))

        # If no decent data at all, bail
        if len(goodIdx) < 1:
            return retVal

        if (self.method.lower() == 'mad'):
            dataCentre = Utilities.screen_data(numpy.array(dataCentre, float), fill=True)
            dataScale = Utilities.screen_data(numpy.array(dataScale, float))
            dataScale = ma.masked_where(dataScale==0.0, dataScale)
            lower = dataCentre - (self.nBounds[0] * dataScale)
            upper = dataCentre + (self.nBounds[1] * dataScale)
        else:
            lower = numpy.array(lower, float)
            upper = numpy.array(upper, float)

        badLowerIdx = set(numpy.flatnonzero(ma.getmaskarray(lower)))
        badUpperIdx = set(numpy.flatnonzero(ma.getmaskarray(upper)))

        bad_idx = list(badLowerIdx.union(badUpperIdx))
        if len(bad_idx) < len(lower): # there are cases all idx are bad
            lower2 = numpy.delete(lower, bad_idx, axis=0)
            upper2 = numpy.delete(upper, bad_idx, axis=0)
            goodIdx2 = numpy.delete(goodIdx, bad_idx, axis=0)
            good_data = data[:,goodIdx2]
            isOutlier_bools = ma.logical_or(good_data < lower2, good_data > upper2)
            
            outlierArray_copy = retVal.outlierArray[:,goodIdx2].copy()
            outlierArray_copy[isOutlier_bools] = 1.0
            retVal.outlierArray[:,goodIdx2] = outlierArray_copy
            retVal.lowerBounds[goodIdx2] = lower2
            retVal.upperBounds[goodIdx2] = upper2
        return retVal

    def twodMADInner(self, dataArray, axis, nVector, estu, suppressOutput):
        """Inner workings of 2D MAD routine
        """
        data = Utilities.screen_data(dataArray)
        # Special treatment for 1-D arrays
        vector = False
        if len(data.shape) == 1:
            vector = True
            if nVector:
                data = data[:,numpy.newaxis]
            else:
                data = data[numpy.newaxis,:]

        # Initialise stuff
        dataMsk = ma.getmaskarray(data)
        clippedLowerValues = Matrices.allMasked(data.shape)
        clippedUpperValues = Matrices.allMasked(data.shape)
        N = data.shape[0]
        T = data.shape[1]
        saveBounds = list(self.nBounds)
        if len(self.nBounds) == 2:
            tmpBounds = self.nBounds + self.nBounds
        else:
            tmpBounds = list(self.nBounds)

        # Check whether we're using estu to determine bounds
        if estu is None:
            if N > 1:
                estu = list(range(N))
        elif len(estu) < 2:
            logging.warning('Estimation universe length: %d; will not use', len(estu))
            if N > 1:
                estu = list(range(N))
            else:
                estu = None

        # Output data limits
        self.outputStats(data, bf='BEFORE', estu=estu, suppressOutput=suppressOutput)

        # First get bounds per column of data
        if N > 1 and (axis==None or axis==0):
            logging.debug('MADing along all %s columns', T)
            self.nBounds = tmpBounds[:2]
            retVal1 = self.computeOutlierBounds(data, estu=estu)
            colOutliers = retVal1.outlierArray
            outliers = ma.getmaskarray(colOutliers)==0
            outlierTimes = numpy.flatnonzero(ma.sum(outliers, axis=0))
        else:
            outliers = ma.getmaskarray(data)==0
            outlierTimes = list(range(T))
            colOutliers = None

        # Next get MAD bounds per row of data
        outlierAssets = numpy.flatnonzero(ma.sum(outliers, axis=1))
        if len(outlierAssets) > 0 and T > 1 and (axis==None or axis==1):

            # Loop round all observations identified as outliers along previous dimension
            logging.debug('MADing along %d of %d data rows',
                    len(outlierAssets), N)
            self.nBounds = tmpBounds[2:]
            retVal2 = self.computeOutlierBounds(ma.transpose(data), idxList=outlierAssets,
                    downweightEnds=self.downweightEnds)
            rowOutliers = ma.transpose(retVal2.outlierArray)
            outliers = outliers * (ma.getmaskarray(rowOutliers)==0)
        
            clippedLowerValues[outlierAssets,:] = retVal2.lowerBounds[outlierAssets][:,numpy.newaxis] # broadcasting along colaxis
            clippedUpperValues[outlierAssets,:] = retVal2.upperBounds[outlierAssets][:,numpy.newaxis] # broadcasting along colaxis
            if colOutliers is not None:
                # Lower set of bound
                lowerCols = clippedLowerValues[:, outlierTimes]
                tmp_retVal1_lowerBounds = retVal1.lowerBounds[outlierTimes][numpy.newaxis,:].repeat(lowerCols.shape[0],axis=0)
                lowerCols = ma.where(lowerCols > tmp_retVal1_lowerBounds,tmp_retVal1_lowerBounds, lowerCols)
                clippedLowerValues[:, outlierTimes] = lowerCols
                # Upper set of bound
                upperCols = clippedUpperValues[:, outlierTimes]
                tmp_retVal1_upperBounds = retVal1.upperBounds[outlierTimes][numpy.newaxis,:].repeat(upperCols.shape[0],axis=0)
                upperCols = ma.where(upperCols < tmp_retVal1_upperBounds,tmp_retVal1_upperBounds, upperCols)
                clippedUpperValues[:, outlierTimes] = upperCols
            
        elif colOutliers is not None:
            clippedLowerValues[:,outlierTimes] = retVal1.lowerBounds[outlierTimes][numpy.newaxis,:] # broadcasting along rowaxis
            clippedUpperValues[:,outlierTimes] = retVal1.upperBounds[outlierTimes][numpy.newaxis,:] # broadcasting along rowaxis
        # Mask anything that isn't an outlier
        clippedLowerValuesMsk = ma.masked_where(outliers==0, clippedLowerValues)
        clippedUpperValuesMsk = ma.masked_where(outliers==0, clippedUpperValues)

        # Finally clip outliers
        tmpData = ma.masked_where(outliers==0, data)
        tmpData = ma.clip(tmpData, clippedLowerValuesMsk, clippedUpperValuesMsk)
        tmpData2 = ma.masked_where(outliers, data)
        tmpData = ma.filled(tmpData, 0.0) + ma.filled(tmpData2, 0.0)

        # Remask formerly masked values and output some stats
        tmpData = ma.masked_where(dataMsk, tmpData)
        diffs = ma.masked_where(data == tmpData, tmpData)
        num = len(numpy.flatnonzero(diffs))
        data = tmpData
        self.nBounds = list(saveBounds)
        if suppressOutput:
            logging.debug('%d out of %d values identified as outliers (%.2f %%)',
                        num, N*T, 100.0*num / (N*T))
        else:
            logging.info('%d out of %d values identified as outliers (%.2f %%)',
                        num, N*T, 100.0*num / (N*T))

        if vector:
            data = ma.ravel(data)
        # Output new data limits
        self.outputStats(data, bf='AFTER', estu=estu, suppressOutput=suppressOutput)

        return data, outliers


    def twodMAD(self, dataArray, axis=None, nVector=True, estu=None, nBounds=None,
                    suppressOutput=False, hideData=None):
        """ 2D outlier detection. Uses multiples of IQR to detect outliers in
        both axes of the data, and then treats as outliers all observations
        which qualify as outliers in both dimensions.
        Assumes data is an array of dimensions N by T,
        where T is arranged in chronological order
        If the data is a 1-D vector, then it assumes that it is a vector
        of N assets. If it is a T-vector, then nVector should be set to False
        zero_pct is the proportion (between 0 and 1) of zeros that are allowed
        per row or column of data
        """
        # Get model parameters
        if nBounds is None:
            self.nBounds = self.getNBounds()
        else:
            self.nBounds = nBounds
        self.method = self.getMethod()
        self.zeroTolerance = self.getZeroTolerance()
        self.shrink = self.getShrinkage()

        self.outputReturnsForDebugging('raw-returns.csv', dataArray)

        # Sort out masking of data
        maskedData = ma.getmaskarray(dataArray)
        if hideData is not None:
            hiddenData = ma.masked_where(hideData==0, dataArray)
            dataCopy = ma.masked_where(hideData, dataArray)
        else:
            dataCopy = ma.array(dataArray, copy=True)

        # Simple case - trim all data using given criteria
        if not self.shrink:

            clippedData, outliers = self.twodMADInner(dataCopy, axis, nVector, estu, suppressOutput)
            if hideData is not None:
                clippedData = ma.masked_where(hideData, clippedData)
                clippedData = ma.filled(clippedData, 0.0) + ma.filled(hiddenData, 0.0)
                clippedData = ma.masked_where(maskedData, clippedData)

            self.outputReturnsForDebugging('clipped-returns.csv', clippedData)
            return clippedData

        # Get the inner band for trimming
        self.outputStats(dataArray, bf='BEFORE', estu=estu, suppressOutput=suppressOutput)
        clippedData, outliers = self.twodMADInner(dataCopy, axis, nVector, estu, True)
        clippedData = ma.masked_where(outliers==0, clippedData)
        dataMsk = ma.getmaskarray(dataCopy)
        tmpBounds = list(self.nBounds)

        # Now trim only the most egregious outliers
        self.nBounds = 2.0 * tmpBounds
        clippedDataX, outliersX = self.twodMADInner(dataCopy, axis, nVector, estu, True)
        clippedDataX = ma.masked_where(outliers==0, clippedDataX)

        # Now find the outer bound for clipped data
        self.nBounds = 1.0 + tmpBounds
        clippedData1, outliersOuter  = self.twodMADInner(dataCopy, axis, nVector, estu, True)
        clippedData1 = ma.masked_where(outliers==0, clippedData1)
        
        # Now shrink the upper outliers
        upperBound = ma.masked_where(clippedDataX <= clippedData, clippedData)
        upperBound1 = ma.masked_where(clippedDataX <= clippedData, clippedData1)
        upperBoundX = ma.masked_where(clippedDataX <= clippedData, clippedDataX)
        ratio = upperBoundX - upperBound
        ratio = ma.masked_where(ratio==0.0, ratio)
        ratio = (upperBound1 - upperBound) / ratio
        tmpDataU = upperBound + (upperBoundX - upperBound) * ratio

        # And do the lower outliers
        lowerBound = ma.masked_where(clippedDataX >= clippedData, clippedData)
        lowerBound1 = ma.masked_where(clippedDataX >= clippedData, clippedData1)
        lowerBoundX = ma.masked_where(clippedDataX >= clippedData, clippedDataX)
        ratio = lowerBound - lowerBoundX
        ratio = ma.masked_where(ratio==0.0, ratio)
        ratio = (lowerBound - lowerBound1) / ratio
        tmpDataL = lowerBound + (lowerBound - lowerBoundX) * ratio
        
        # Recombine
        data = ma.masked_where(outliers, dataCopy)
        clippedData = ma.filled(data, 0.0) + ma.filled(tmpDataU, 0.0) + ma.filled(tmpDataL, 0.0)

        # Remask formerly masked values and output some stats
        clippedData = ma.masked_where(dataMsk, clippedData)
        if hideData is not None:
            clippedData = ma.masked_where(hideData, clippedData)
            clippedData = ma.filled(clippedData, 0.0) + ma.filled(hiddenData, 0.0)
            clippedData = ma.masked_where(maskedData, clippedData)
        diffs = ma.masked_where(dataArray == clippedData, clippedData)
        num = len(numpy.flatnonzero(diffs))

        # Report on number of outliers detected
        if len(clippedData.shape) > 1:
            NT = clippedData.shape[1] * clippedData.shape[0]
        else:
            NT = clippedData.shape[0]
        if suppressOutput:
            logging.debug('%d out of %d values identified as outliers (%.2f %%)',
                        num, NT, 100.0*num / NT)
        else:
            logging.info('%d out of %d values identified as outliers (%.2f %%)',
                        num, NT, 100.0*num / NT)

        # Output new data limits
        self.outputStats(clippedData, bf='AFTER', estu=estu, suppressOutput=suppressOutput)
        self.outputReturnsForDebugging('clipped-returns.csv', clippedData)
        return clippedData

    def bucketedMAD(self, rmgList, date, returnsIn, assetData, modelDB,
                        axis=0, industryGroupFactor=False):
        """Given an array of asset returns, sorts them into sector/region
        buckets and MADs each separately
        """
        logging.debug('bucketedMAD: begin')

        # Get model parameters
        self.nBounds = self.getNBounds()
        self.method = self.getMethod()
        self.zeroTolerance = self.getZeroTolerance()
        self.shrink = self.getShrinkage()
        if type(rmgList) is not list:
            rmgList = [rmgList]

        assetIdxMap = dict(zip(assetData.universe, range(len(assetData.universe))))
        returns = ma.array(returnsIn, copy=True)
        vector = False
        if len(returns.shape) < 2:
            returns = returns[:, numpy.newaxis]
            vector = True

        self.outputStats(returns, bf='BEFORE')

        # Get sector-level buckets
        if industryGroupFactor:
            parentName = 'Industry Groups'
            level = -1
        else:
            parentName = 'Sectors'
            level = -2

        # Bucket assets into regions
        regionAssetMap = dict()
        for r in rmgList:
            rmg_assets = Utilities.readMap(r, assetData.rmgAssetMap)
            if r.region_id not in regionAssetMap:
                regionAssetMap[r.region_id] = list()
            regionAssetMap[r.region_id].extend(rmg_assets)

        # Go through assets by region
        for reg in regionAssetMap.keys():
            subSetIds = regionAssetMap[reg]
            logging.debug('MADing over %d assets for region %s',
                    len(subSetIds), reg)
            remainingIds = set(subSetIds)
            exposures, factorList = modelDB.getGICSExposures(date, subSetIds, level=parentName, clsDate=self.gicsDate)
            exposures = ma.transpose(ma.masked_where(exposures==0.0, exposures))

            # Loop round sectors and pull out sector/region assets
            for isec in range(exposures.shape[0]):
                sectorAssetIdx = numpy.flatnonzero(ma.getmaskarray(exposures[isec,:])==0)
                if len(sectorAssetIdx) < 10:
                    logging.warning('Too few assets (%d) for sector %s', len(sectorAssetIdx), factorList[isec])
                    sectorAssets = []
                else:
                    sectorAssets = numpy.take(subSetIds, sectorAssetIdx, axis=0)
                    logging.debug('MADing over %d assets for sector %s', len(sectorAssets), factorList[isec])
                    returnsIdx = [assetIdxMap[sid] for sid in sectorAssets]
                    subSetReturns = ma.take(returns, returnsIdx, axis=0)
                    clippedReturns = self.twodMAD(subSetReturns, axis=axis, suppressOutput=True)
                    for (idim, idx) in enumerate(returnsIdx):
                        returns[idx,:] = clippedReturns[idim,:]
                remainingIds = remainingIds.difference(set(sectorAssets))

            # Deal with anything not mapped
            if len(remainingIds) > 0:
                logging.warning('%s assets not mapped to sector/IG', len(remainingIds))
                if len(remainingIds) < 6:
                    logging.warning('Too few assets (%d) for missing sector/IG', len(remainingIds))
                else:
                    sectorAssets = list(remainingIds)
                    returnsIdx = [assetIdxMap[sid] for sid in sectorAssets]
                    subSetReturns = ma.take(returns, returnsIdx, axis=0)
                    clippedReturns = self.twodMAD(subSetReturns, axis=axis, suppressOutput=True)
                    for (idim, idx) in enumerate(returnsIdx):
                        returns[idx,:] = clippedReturns[idim,:]

        self.outputStats(returns, bf='AFTER')
        if vector:
            returns = returns[:,0]
        logging.debug('bucketedMAD: end')
        return returns

