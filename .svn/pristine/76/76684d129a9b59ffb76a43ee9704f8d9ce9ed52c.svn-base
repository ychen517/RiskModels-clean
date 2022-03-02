import numpy.ma as ma
import numpy.linalg as linalg
import math
import numpy
import datetime
import logging.config
import os
import sys
import string
import scipy.interpolate as spline
import scipy.stats as stats
from collections import defaultdict
import numpy as np
import pandas
import gzip
from riskmodels import Matrices
from riskmodels import Classification
from riskmodels import Outliers

# Old legacy Utilities code, full of superceded functions

class Struct:
    def __init__(self, copy=None):
        if copy is not None:
            if isinstance(copy, dict):
                self.__dict__ = copy.copy()
            else:
                self.__dict__ = dict(copy.__dict__)
    def getFields(self): return list(self.__dict__.values())
    def getFieldNames(self): return list(self.__dict__.keys())
    def setField(self, name, val): self.__dict__[name] = val
    def getField(self, name): return self.__dict__[name]
    def __str__(self):
        return '(%s)' % ', '.join(['%s: %s' % (i, j) for (i,j)
                                  in self.__dict__.items()])
    def __repr__(self):
        return self.__str__()

    def to_dict(self):
        return self.__dict__.copy()

    def update(self,other):
        assert isinstance(other,dict) or isinstance(other,Struct)
        if isinstance(other,dict):
            for k,v in other.items():
                self.__dict__[k]=v
        if isinstance(other,Struct):
            for k in other.getFieldNames():
                self.__dict__[k]=other.getField(k)

class NeweyWestStandardErrors(object):
    def __init__(self,Y,X,B,nw_lags,nw_overlap=True):
        """
        Computes Newey West standard errors for the regressions 
        Y = X.dot( B.T ) + error
        assuming
        Y,X are pandas time series DataFrames (with dates in rows)
        B is a Ny x Nx DataFrame
        nw_lags is an integer >= 0.

        pandas.ols conventions are followed.  

        """
        from numpy.linalg import pinv
        self._keep_beta_cov=False
        self.Y=Y
        self.X=X
        self.B=B
        self.nw_lags=nw_lags
        self.nw_overlap=nw_overlap

        self.xx=np.dot(X.T, X)
        self.xx_inv=pinv( self.xx )
        self.resid=Y - np.dot(X, B.T)

    def compute(self):
        """ Returns standard errors with newey_west adjustment."""
        from pandas.stats.math import newey_west,  rank
        X=self.X
        Y=self.Y

        df_raw=rank(X.values)
        beta_cov_map={}
        std_err_map={}
        for a in Y.columns:
            m=(self.X.T * self.resid[a]).T.copy().values.copy()
            nobs=len( set(Y[a].dropna().index).intersection(X.dropna().index))
            xeps=newey_west(m,self.nw_lags, nobs, df_raw, self.nw_overlap)
            beta_cov_map[a]=pandas.DataFrame( np.dot( self.xx_inv, np.dot(xeps, self.xx_inv)), X.columns, X.columns)
            std_err_map[a]=pandas.Series( np.sqrt( np.diag(beta_cov_map[a].values) ), X.columns)
        self.std_err=pandas.DataFrame(std_err_map).T.copy()

        if self._keep_beta_cov:
            self.beta_cov=pandas.Panel(beta_cov_map)

        return self.std_err 

def oldPandasVersion():
    pdVer = float('.'.join(pandas.__version__.split('.')[:2]))
    if pdVer < 0.23:
        print ('WARNING - Using old version of pandas:', pdVer)
        return True
    return False

def getEOMDates(dates,drop_if_before_day=0):
    """ The last date may be dropped if drop_if_before_day>0. """
    months = defaultdict(list)
    for d in dates:
        months[(d.year,d.month)].append(d)
    tmp=sorted(max(values) for values in months.values())
    return [d for d in tmp if d.day>drop_if_before_day]


def getFOMDates(dates,drop_if_after_day=32):
    """ The first date may be dropped if drop_if_after_day<32. """
    months = defaultdict(list)
    for d in dates:
        months[(d.year,d.month)].append(d)
    tmp=sorted(min(values) for values in months.values())
    return [d for d in tmp if d.day<drop_if_after_day]


def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    if len(l) > 0 and n == 0:
        raise Exception('Trying to chunk a nonempty list with chunks of size 0')
    if n != 0:
        for i in range(0, len(l), n):
            yield l[i:i+n]

def noneToVal(x, val):
    """Returns val if x is None and x otherwise.
    """
    if x == None:
        return val
    return x

def reverseMap(inMap):
    """ Given a dictionary mapping, outputs the reverse map
        i.e. from dict(a,b) to dict(b,a)
    """
    return dict((j,i) for (i,j) in inMap.items())

def noneToValTuple(x, val):
    """Returns val if x is None and x otherwise.
    """
    if x == (None,None):
        return val
    return x

def stringToDate(dateStr):
    """Convert YYYY-MM-DD string to datetime.date object.
    """
    return datetime.date(int(dateStr[0:4]), int(dateStr[5:7]), int(dateStr[8:10]))

def matchLatestDate(list1, list2):
    """Returns for each entry the latest value common to both lists.
    The inputs are lists of lists. Each entry is a list of
    (date, value) tuples sorted in chronological order.
    The result returns a tuple of two lists, each containing
    (date, value) tuples which for each index has the latest date
    that exists in both lists and the corresponding value in list1
    and list2 respectively.
    If no such date exists, the result entries are (None, None).
    """
    assert(len(list1) == len(list2))
    r1 = []
    r2 = []
    for (l1, l2) in zip(list1, list2):
        d1 = dict(l1)
        l2.reverse()
        v1 = (None, None)
        v2 = (None, None)
        for t in l2:
            if t[0] in d1:
                v1 = (t[0], d1[t[0]])
                v2 = t
                break
        r1.append(v1)
        r2.append(v2)
    return (r1, r2)

def extractLatestValue(values):
    """Extracts the latest value from a list of lists of (date, value)
    tuples.
    Returns an array of the latest values.
    If no tuple was present for an entry, the corresponding return value
    is masked.
    """
    # extract last value, None if list is empty
    latest = [([(None, None)] + i)[-1][1] for i in values]
    # convert list into array, masking the None values
    return ma.masked_where([i == None for i in latest],
                           [noneToVal(i, 0.0) for i in latest])

def extractLatestValueAndDate(values):
    """Extracts the latest value and corresponding date from a list
    of lists of (date, value) tuples.
    Returns an array of the latest values and dates.
    If no tuple was present for an entry, the corresponding return value
    is masked.
    """
    # extract last value, None if list is empty
    latest = [([(None, None)] + i)[-1][:2] for i in values]
    latest.append((None,None))
    # convert list into array, masking the None values
    retVal = ma.masked_where([i == (None,None) for i in latest],
           [noneToValTuple(i, 0.0) for i in latest])
    return retVal[:-1]

def extractLatestValuesAndDate(values):
    """Extracts the latest tuple of values and corresponding date from a list
    of lists of (date, value1, value2, ...) tuples.
    Returns an array of the latest values and dates.
    If no tuple was present for an entry, the corresponding return value
    is masked.
    """
    # extract last value, None if list is empty
    latest = [([None] + i)[-1] for i in values]
    # convert list into array, masking the None values
    isNone = [i == None for i in latest]
    values = [noneToValTuple(i, 0.0) for i in latest]
    isNone.append(True)
    values.append(np.nan)
    out = ma.masked_where(isNone,values)
    out2 = out[:-1]
    return out2

def prctile(x, percentiles):
    """Computes the given percentiles of x similar to how
    the prctile computes them.
    percentiles is a list of percentages, from 0 to 100.
    For an N element vector x, prctile computes percentiles as follows:
      1) the sorted values of X are taken as 100*(0.5/N), 100*(1.5/N),
         ..., 100*((N-0.5)/N) percentiles
      2) linear interpolation is used to compute percentiles for percent
         values between 100*(0.5/N) and 100*((N-0.5)/N)
      3) the minimum or maximum values in X are assigned to percentiles
         for percent values outside that range.
    prctile assumes that x is a one-dimensional masked array
    and ignores masked values.
    """
    y = ma.take(x, numpy.flatnonzero(ma.getmaskarray(x) == 0), axis=0)
    sortorder = ma.argsort(y)
    prct = numpy.array(percentiles, float)
    idx = ((prct * y.shape[0]) / 100) - 0.5
    idx = numpy.clip(idx, 0.0, len(y) - 1)
    low = [int(math.floor(i)) for i in idx]
    high = [int(math.floor(i+1)) for i in idx]
    sortorder = list(sortorder)
    sortorder.append(sortorder[-1])
    return [(h - i) * y[sortorder[l]] + (i - l) * y[sortorder[h]]
            for (l, h, i) in zip(low, high, idx)]

def compute_total_risk_portfolio(portfolio, expMatrix, factorCov, 
                                 srDict, scDict=None, factorTypes=None):
    """Compute the total risk of a given portfolio.  portfolio 
    should be a list of (asset,weight) values.
    factorTypes is an optional argument and should be a list of 
    ExposureMatrix.FactorType objects.  If specified, the total 
    common factor risk from those factors are returned instead.
    """
    expM_Map = dict([(expMatrix.getAssets()[i], i) \
                    for i in range(len(expMatrix.getAssets()))])
    # Discard any portfolio assets not covered in model
    indices = [i for (i,j) in enumerate(portfolio) \
               if (j[0] in expM_Map) and (j[0] in srDict)]
    port = [portfolio[i] for i in indices]
    (assets, weights) = zip(*port)

    expM = expMatrix.getMatrix().filled(0.0)
    expM = ma.take(expM, [expM_Map[a] for (a,w) in port], axis=1)
    if factorTypes is not None:
        f_indices = []
        for fType in factorTypes:
            fIdx = expMatrix.getFactorIndices(fType)
            f_indices.extend(fIdx)
        expM = ma.take(expM, f_indices, axis=0)
        factorCov = numpy.take(factorCov, f_indices, axis=0)
        factorCov = numpy.take(factorCov, f_indices, axis=1)

    # Compute total risk
    assetExp = numpy.dot(expM, weights)
    totalVar = numpy.dot(assetExp, numpy.dot(factorCov, assetExp))
    if factorTypes is None:
        totalVar += numpy.sum([(w * srDict[a])**2 for (a,w) in port])
    
    if scDict is None:
        return totalVar**0.5

    # Incorporate linked specific risk
    assetSet = set(assets)
    assetIdxMap = dict([(j,i) for (i,j) in enumerate(assets)])
    for sid0 in assets:
        if sid0 in scDict:
            for (sid1, cov) in scDict[sid0].items():
                if sid1 not in assetSet:
                    continue
                weight0 = weightMatrix[assetIdxMap[sid0]] 
                weight1 = weightMatrix[assetIdxMap[sid1]] 
                totalVar += 2.0 * weight0 * weight1 * cov
    return totalVar**0.5

def get_indices_from_list(mainList, subList):
    """Returns a set of indices for elements in subList
    as they appear in mainList
    """
    mainIdxMap = dict([(j,i) for (i,j) in enumerate(mainList)])
    subList = [n for n in subList if n in mainIdxMap]
    return [mainIdxMap[n] for n in subList]

def get_subset_by_id(data, universe, subset, axis=0):
    """Picks of subset of values from an array of data
    The parameter universe is a list of IDs - we assume
    that the array data is sorted by these along one
    axis (the 0 axis by default). subset is a smaller set of
    IDs, not necessarily all in universe, for which we
    wish to pick corresponding values from data
    """
    vector = False
    if len(data.shape) < 2:
        axis = 0
        vector = True
     
    # Sort out array indices
    mainIdxMap = dict([(j,i) for (i,j) in enumerate(universe)])
    subIdxMap = dict([(j,i) for (i,j) in enumerate(subset)])
    okIdList = list(set(subset).intersection(set(universe)))
     
    # Set up subset array
    if vector:
        outArray = Matrices.allMasked((len(subset)))
    elif axis == 0:
        outArray = Matrices.allMasked((len(subset), data.shape[1]))
    else:
        outArray = Matrices.allMasked((data.shape[0], len(subset)))

    # Pick out relevant entries
    subIdx = [mainIdxMap[sid] for sid in okIdList]
    subData = ma.take(data, subIdx, axis=axis)

    # Assign them to the correct place
    subIdx = [subIdxMap[sid] for sid in okIdList]
    ma.put(outArray, subIdx, axis=axis)
    return outArray

def change_date_frequency(dailyDateList, frequency='weekly'):
    """Converts a list of daily dates to either weekly (default)
    or monthly
    """
    if frequency == 'weekly':
        periodDateList = [prv for (prv, nxt) in \
                zip(dailyDateList[:-1], dailyDateList[1:])
                if nxt.weekday() < prv.weekday()]
    elif frequency == 'monthly':
        periodDateList = [prv for (prv, nxt) in \
                zip(dailyDateList[:-1], dailyDateList[1:])
                if nxt.month > prv.month or nxt.year > prv.year]
    elif frequency == 'rolling':
        periodDateList = [d for d in dailyDateList \
                if d.weekday() == dailyDateList[-1].weekday()]
    else:
        return dailyDateList
    return periodDateList

def compute_compound_returns(data, highFreqDateList, lowFreqDateList,
        average=False):
    """Converts a set of higher frequency returns (e.g. daily)
    into lower frequency (weekly, monthly)
    data is an n*t array where t is the number of dates in highFreqDateList
    """
    vector = False
    if len(data.shape) == 1:
        data = data[numpy.newaxis,:]
        vector = True
    # Set up date info
    dateDict = dict([(d,i) for (i,d) in enumerate(highFreqDateList)])
    lowFreqHistLen = len(lowFreqDateList)-1
    # Initialise low frequency data array
    lowFreqData = Matrices.allMasked((\
            data.shape[0],lowFreqHistLen), float)
    # Loop round low frequency dates
    i0 = -1
    for (j,d) in enumerate(lowFreqDateList):
        i1 = dateDict[d]
        if i0 >= 0:
            dataChunk = data[:,i0:i1]
            if average:
                lowFreqData[:,j-1] = ma.average(dataChunk, axis=1)
            else:
                lowFreqData[:,j-1] = ma.product(dataChunk+1, axis=1) - 1.0
        i0 = i1

    if vector:
        lowFreqData = lowFreqData[0,:]
    return lowFreqData

def compute_compound_returns_v3(dataIn, highFreqDateListIn, lowFreqDateListIn,
                                keepFirst=False, matchDates=False,
                                sumVals=False, mean=False):
    """Converts a set of higher frequency returns (e.g. daily)
    into lower frequency (weekly, monthly)
    data is an n*t array where t is the number of dates in highFreqDateList
    """
    data = screen_data(dataIn)
    highFreqDateList = list(highFreqDateListIn)
    lowFreqDateList = list(lowFreqDateListIn)
    maskArray = numpy.array(ma.getmaskarray(data)==0, copy=True)
    data = ma.filled(data, 0.0)

    # Dimensional rearranging if input is a vector
    vector = False
    if len(data.shape) == 1:
        data = data[numpy.newaxis,:]
        maskArray = maskArray[numpy.newaxis,:]
        vector = True

    logging.debug('Compounding %s returns from %s dates to %s',
            data.shape[0], len(highFreqDateList), len(lowFreqDateList))
    # Sift out dates with no returns that are not in the output date list
    sumNonMissing = ma.sum(maskArray, axis=0)
    sumNonMissing = ma.masked_where(sumNonMissing>0, sumNonMissing)
    allMissingIdx = numpy.flatnonzero(ma.getmaskarray(sumNonMissing)==0)
    allMissingIdx = [idx for idx in allMissingIdx \
            if highFreqDateList[idx] not in lowFreqDateList]
    if len(allMissingIdx) > 0:
        logging.debug('Removing %d dates where all returns missing',
                len(allMissingIdx))
        okIdx = [idx for idx in range(data.shape[1]) if idx not in allMissingIdx]
        highFreqDateList = [highFreqDateList[idx] for idx in okIdx]
        data = numpy.take(data, okIdx, axis=1)
        maskArray = numpy.take(maskArray, okIdx, axis=1)

    if sorted(highFreqDateList) == sorted(lowFreqDateList):
        logging.debug('Lists of dates are identical: skipping')
        data = ma.masked_where(maskArray==0, data)
        if vector:
            data = data[0,:]
        return (data, highFreqDateList)

    origLFDates = list(lowFreqDateList)
    if len(lowFreqDateList) < 1:
        logging.warning('Problem with dates: zero length. Bailing...')
        return (dataIn, highFreqDateList)

    # Set up date info
    lowFreqDateList = [d for d in lowFreqDateList \
            if d >= highFreqDateList[0] and d <= highFreqDateList[-1]]
    dateIdxMap = dict([(d,i) for (i,d) in enumerate(highFreqDateList)])
    dt = highFreqDateList[0]
    idx = None
    while dt < highFreqDateList[-1]:
        if dt not in dateIdxMap:
            dateIdxMap[dt] = idx
        idx = dateIdxMap[dt]
        dt += datetime.timedelta(1)
    dIds = [dateIdxMap[d] for d in lowFreqDateList]
     
    # Work out compound frequency
    freq = int(len(highFreqDateList) / float(len(lowFreqDateList)))
    if freq < 2:
        keepFirst = True
    else:
        if dateIdxMap[highFreqDateList[-1]] - dateIdxMap[lowFreqDateList[-1]] >= freq:
            dIds.append(dateIdxMap[highFreqDateList[-1]])
     
    # Compute compound returns
    freqList = []
    if not sumVals:
        lowFreqData = numpy.zeros((data.shape[0],len(dIds)-1), float)
        id0 = dIds[0]
        for (idx,id1) in enumerate(dIds[1:]):
            if id1 == id0:
                lowFreqData[:,idx] = data[:,id0]
                freqList.append(1.0)
            else:
                lowFreqData[:,idx] = numpy.product(data[:,id0+1:id1+1]+1.0, axis=1) - 1.0
                freqList.append(id1-id0)
            id0 = id1
    else:
        lowFreqData = numpy.zeros((data.shape[0],len(dIds)-1), float)
        id0 = dIds[0]
        for (idx,id1) in enumerate(dIds[1:]):
            if id1 == id0:
                lowFreqData[:,idx] = data[:,id0]
                freqList.append(1.0)
            else:
                lowFreqData[:,idx] = numpy.sum(data[:,id0+1:id1+1], axis=1)
                freqList.append(id1-id0)
            id0 = id1
    cumMask = numpy.cumsum(maskArray, axis=1)
    cumMaskSample = numpy.take(cumMask, dIds, axis=1)
    lowFreqMask = cumMaskSample[:,1:] - cumMaskSample[:,:-1]
    lowFreqMask = numpy.array(lowFreqMask, dtype='bool')
    if keepFirst:
        # Optional retaining of first return if we're merely removing
        # a few rogue dates rather than truly compounding
        lowFreqData = numpy.concatenate((
            data[:,dIds[0]][:,numpy.newaxis], lowFreqData), axis=1)
        lowFreqMask = numpy.concatenate((
            cumMaskSample[:,0][:,numpy.newaxis], lowFreqMask), axis=1)
        freqList = [1.0] + freqList
    dateList = lowFreqDateList[:lowFreqData.shape[1]]
    if mean:
        if not sumVals:
            for idx in range(lowFreqData.shape[1]):
                lowFreqData[:,idx] = lowFreqData[:,idx] ** (1.0/float(freqList[idx]))
        else:
            for idx in range(lowFreqData.shape[1]):
                lowFreqData[:,idx] = lowFreqData[:,idx] / float(freqList[idx])

    # Replace masks where appropriate
    lowFreqData = ma.masked_where(lowFreqMask==0, lowFreqData)
     
    # If required, map returns to array of original size/dates
    if matchDates:
        tmpData = Matrices.allMasked((data.shape[0], len(origLFDates)))
        dateIdxMap = dict([(d,i) for (i,d) in enumerate(dateList)])
        for (ii, d) in enumerate(origLFDates):
            if d in dateIdxMap:
                idx = dateIdxMap[d]
                tmpData[:,ii] = lowFreqData[:,idx]
        lowFreqData = tmpData
        dateList = origLFDates

    if vector:
        lowFreqData = lowFreqData[0,:]
    return (lowFreqData, dateList)

def compute_compound_returns_v4(dataIn, highFreqDateListIn, lowFreqDateListIn, fillWithZeros=True):
    """Converts a set of higher frequency returns (e.g., daily)
    into lower frequency returns (e.g., weekly, monthly)
    -- data is an n*t array of returns corresponding to the t dates in
    highFreqDateList, where the t dates are the end-of-period dates for 
    the t time periods
    -- note that the dates in lowFreqDateListIn are assumed to be the 
    end-of-period dates for the periods to which returns are compounded
    -- the date list that is returned from this function corresponds to the
    end-of-period dates for the lower frequency returns
    -- if fillWithZeros is True, low frequency returns for which ALL
    corresponding high frequency returns are null will be set to zeros (and
    will not be masked), otherwise they will be set to np.nan (and masked)
    """

    # get mask identifying nans, infs, and masked values
    if isinstance(dataIn, np.ma.MaskedArray):
        if not isinstance(dataIn.mask, np.bool_):
            dataMask = np.isnan(dataIn.data) + np.isinf(dataIn.data) + dataIn.mask
        else:
            dataMask = np.isnan(dataIn.data) + np.isinf(dataIn.data) 
    else:
        dataMask = np.isnan(dataIn) + np.isinf(dataIn) 

    data = ma.array(dataIn, copy=True) # convert ndarray or ma to ma
    data = ma.filled(data, 0.0) # create ndarray from ma, filling masked values with zeros
    data[dataMask] = 0.0
    vector = False
    if len(data.shape) == 1:
        vector = True
        data = data[numpy.newaxis, :]
        dataMask = dataMask[numpy.newaxis, :]

    highFreqDateList = list(highFreqDateListIn)
    lowFreqDateList = list(lowFreqDateListIn)
    assert(data.shape[1] == len(highFreqDateListIn))
    if sorted(highFreqDateList) == sorted(lowFreqDateList):
        if fillWithZeros:
            return (data, highFreqDateList)
        else:
            return (dataIn, highFreqDateList)

    # compound returns
    lowFreqDateList = sorted([d for d in lowFreqDateList \
            if d >= highFreqDateList[0] and d <= highFreqDateList[-1]])
    dateIdxMap = dict([(d,i) for (i,d) in enumerate(highFreqDateList)])
    lowFreqData = numpy.empty((data.shape[0], len(lowFreqDateList)), float)
    if fillWithZeros:
        lowFreqData[:] = 0.
    else:
        lowFreqData[:] = np.nan
    mask = np.ones(lowFreqData.shape[0], dtype=np.bool)
    for idx, dt in enumerate(lowFreqDateList):
        if idx == 0:
           retIdx = [dateIdxMap[d] for d in dateIdxMap.keys() if d <= dt]
        else:
           prevDt = lowFreqDateList[idx-1]
           retIdx = [dateIdxMap[d] for d in dateIdxMap.keys() if d > prevDt and d <= dt]
        prets = np.take(data, retIdx, axis=1)
        cumrets = np.product(1. + prets, axis=1) - 1.
        if not fillWithZeros:
            if prets.shape[1] == 1:
                mask = ~np.take(dataMask, retIdx, axis=1)[:, 0] 
            else:
                # only replace values if at least one pret is not masked/null
                mask = np.take(dataMask, retIdx, axis=1).sum(axis=1) < len(retIdx)
        lowFreqData[mask, idx]  = cumrets[mask] 

    if vector:
        lowFreqData = lowFreqData.flatten()
    
    return (ma.masked_invalid(lowFreqData), lowFreqDateList)

def compute_correlation_portfolios(expMatrix, factorCov, 
                                   srDict, folioList, scDict=None):
    """Compute correlation matrix between two 
    or more portfolios.
    Returns a tuple containing the volatilities (risks)
    of the portfolios and their correlation matrix.
    scDict is an optional dict-of-dicts containing 
    specific correlations.
    """
    assert(len(folioList) > 1)
    expM_Map = dict([(expMatrix.getAssets()[i], i) \
                    for i in range(len(expMatrix.getAssets()))])

    # Only keep track of assets covered in the model
    assets = set([a for f in folioList for (a,w) in f])
    assets = list(assets.intersection(expMatrix.getAssets()))
    weightMatrix = numpy.zeros((len(assets), len(folioList)))
    assetIdxMap = dict([(j,i) for (i,j) in enumerate(assets)])

    # Populate matrix of portfolio weights
    for (i,f) in enumerate(folioList):
        for (a,w) in f:
            pos = assetIdxMap.get(a, None)
            if pos is None:
                continue
            weightMatrix[pos,i] = w
                
    # Reshuffle exposure matrix
    expM = expMatrix.getMatrix()
    expM = ma.filled(ma.take(expM, [expM_Map[a] for a in assets], axis=1), 0.0)

    # Multiply things out
    portExp = numpy.dot(expM, weightMatrix)
    portCov = numpy.dot(numpy.transpose(portExp), 
                        numpy.dot(factorCov, portExp))

    # Add specific risks
    specVars = [srDict.get(a, 0.0)**2.0 for a in assets]
    for i in range(len(folioList)):
        for j in range(len(folioList)):
            if i >= j:
                sv = numpy.inner(weightMatrix[:,i] * weightMatrix[:,j], 
                                 specVars)
                portCov[i,j] += sv
                if i != j:
                    portCov[j,i] += sv

    # Return correlation matrix
    if scDict is None:
        return cov2corr(portCov)

    # Add specific covariances
    assetSet = set(assets)
    for sid0 in assets:
        if sid0 in scDict:
            for (sid1, cov) in scDict[sid0].items():
                if sid1 not in assetSet:
                    continue
                weights0 = weightMatrix[assetIdxMap[sid0],:] 
                weights1 = weightMatrix[assetIdxMap[sid1],:] 
                crossWeights = numpy.outer(weights0, weights1)
                crossWeights = crossWeights + numpy.transpose(crossWeights)
                portCov += crossWeights * cov
    return cov2corr(portCov)

def compute_MCTR(expMatrix, factorCov, srDict, portfolio):
    """Compute MCTR for all assets relative to a given
    portfolio
    """
    expM_Map = dict([(expMatrix.getAssets()[i], i) \
            for i in range(len(expMatrix.getAssets()))])
    # Discard any portfolio assets not covered in model
    port_indices = [i for (i,j) in enumerate(portfolio) \
            if (j[0] in expM_Map) and (j[0] in srDict)]
    # Set up weights
    port = [portfolio[i] for i in port_indices]
    weights = [w for (a,w) in port]
    #weights /= numpy.sum(weights)
    # Get exp matrix
    expM = expMatrix.getMatrix().filled(0.0)
    # Build full weights vector
    h = numpy.zeros((len(expMatrix.getAssets())), float)
    w_indices = [expM_Map[a] for (a,w) in port]
    numpy.put(h, w_indices, weights)
    # Compute numerator
    assetExp = numpy.dot(expM, h)[:,numpy.newaxis]
    Vh = numpy.dot(numpy.transpose(expM),\
                numpy.dot(factorCov, assetExp))
    # Compute total risk
    factorVar = numpy.dot(h, Vh)
    totalVar = factorVar + numpy.sum([(w * srDict[a])**2 for (a,w) in port])

    MCTR = Vh / math.sqrt(totalVar)
    return MCTR

def compute_total_risk_assets(assetList, expMatrix, factorCov, srDict):
    """Compute the total risk of each asset in the given assetList as
    predicted by the given risk model.  A list of total risks is
    returned giving the total risk of each asset in the order they
    appear in assetList.
    """
    expM_Map = dict([(expMatrix.getAssets()[i], i) for i in range(len(expMatrix.getAssets()))])
    fc_times_exp = numpy.dot(factorCov, expMatrix.getMatrix())
    totalrisks = []
    for i in range(len(assetList)):
        asset = assetList[i]
        idx = expM_Map[asset]
        fv = numpy.inner(expMatrix.getMatrix()[:,idx],
                                  fc_times_exp[:,idx])
        sr = srDict[asset]
        totalrisks.append(math.sqrt(fv + sr*sr))
    return totalrisks

def isCashAsset(asset):
    if hasattr(asset, 'isCashAsset'):
        return asset.isCashAsset()
    elif isinstance(asset, str):
        return asset.startswith('CSH_') and asset.endswith('__')
    return False

def generate_predicted_betas(assets, expMatrix, factorCov,
                            srDict, marketPortfolio, forceRun=False,
                            nonCurrencyIdx=None, debugging=False):
    """Compute predicted beta of each SubIssue in assets using
    the given risk model.  A list of market (SubIssue, weight) assets
    is given by marketPortfolio.  A list of betas is
    returns in the same order as the assets in assets.
    """
    logging.debug('generate_predicted_betas: begin')

    # Make sure market portfolio is covered by model
    exposureAssets = expMatrix.getAssets()
    assetIdxMap = dict([(j,i) for (i,j) in enumerate(exposureAssets)])
    len0 = len(marketPortfolio)
    marketPortfolio = [(a,w) for (a,w) in marketPortfolio if a in assetIdxMap]
    if len(marketPortfolio) < len0:
        logging.warning('%d assets in market portfolio not in model', len0-len(marketPortfolio))
    if len(marketPortfolio) == 0:
        logging.warning('Empty market portfolio')
        if forceRun:
            return [0] * len(assets)
    mktIndices, weights = zip(*[(assetIdxMap[a], w) for (a,w) in marketPortfolio])
    market_ids = [exposureAssets[i] for i in mktIndices]
    market_id_map = dict([(exposureAssets[j],i) for (i,j) in enumerate(mktIndices)])

    # Compute market portfolio specific variance
    univ_sr = numpy.array([srDict[asset] for asset in market_ids])
    univ_sv = univ_sr * univ_sr
    market_sv = numpy.inner(weights, univ_sv * weights)

    if nonCurrencyIdx is not None:
        exposures = numpy.take(expMatrix.getMatrix(), nonCurrencyIdx, axis=0)
        factorCov = numpy.take(factorCov, nonCurrencyIdx, axis=0)
        factorCov = numpy.take(factorCov, nonCurrencyIdx, axis=1)
    else:
        exposures = expMatrix.getMatrix()
     
    # Compute market portfolio common factor variances
    expM_Idx = [assetIdxMap[a] for a in market_ids]
    market_exp = numpy.dot(ma.take(exposures, expM_Idx, axis=1), weights)
    market_cv_exp = numpy.dot(factorCov, market_exp)
    market_var = numpy.inner(market_exp, market_cv_exp) + market_sv
    if debugging:
        logging.info('Market risk is %.2f%%', 100.0 * math.sqrt(market_var))
        logging.info('...Market portfolio weights range: [%.6f, %.6f,  %.6f]',
                min(weights), ma.mean(weights), max(weights))

    # Compute asset predicted betas
    beta = []
    cfList = []
    svList = []
    for i in range(len(assets)):
        asset = assets[i]
        if isCashAsset(asset):
            beta.append(0.0)
        else:
            idx = assetIdxMap[asset]
            # Compute asset factor covariance with market
            fv = numpy.inner(exposures[:,idx], market_cv_exp)
            sv = 0.0
            # Add specific component
            if asset in market_id_map:
                sv = weights[market_id_map[asset]] * srDict[asset] * srDict[asset]
            beta.append((fv + sv)/market_var)
            cfList.append(fv)
            svList.append(sv)

    if debugging:
        logging.info('...Market var: %.6f, Market specific var: %.6f', market_var, market_sv)
        logging.info('...CF range: [%.6f, %.6f], Sp range: [%.6f, %.6f]',
                min(cfList), max(cfList), min(svList), max(svList))
        logging.info('...Beta range: [%.6f, %.6f,  %.6f]', min(beta), ma.mean(beta), max(beta))
    logging.debug('generate_predicted_betas: end')
    return beta

def addLogConfigCommandLine(optionParser):
    """Add options to select the log configuration file
    """
    optionParser.add_option("-l", "--log-config", action="store",
                            default='log.config', dest="logConfigFile",
                            help="logging configuration file")

def addDefaultCommandLine(optionParser):
    """Add options to select the log configuration file and
    the ModelDB and MarketDB connection parameters and provide defaults.
    """
    addLogConfigCommandLine(optionParser)
    optionParser.add_option("--modeldb-passwd", action="store",
                            default=os.environ.get('MODELDB_PASSWD'),
                            dest="modelDBPasswd",
                            help="Password for ModelDB access")
    optionParser.add_option("--modeldb-sid", action="store",
                            default=os.environ.get('MODELDB_SID'),
                            dest="modelDBSID",
                            help="Oracle SID for ModelDB access")
    optionParser.add_option("--modeldb-user", action="store",
                            default=os.environ.get('MODELDB_USER'),
                            dest="modelDBUser",
                            help="user for ModelDB access")
    optionParser.add_option("--marketdb-passwd", action="store",
                            default=os.environ.get('MARKETDB_PASSWD'),
                            dest="marketDBPasswd",
                            help="Password for MarketDB access")
    optionParser.add_option("--marketdb-sid", action="store",
                            default=os.environ.get('MARKETDB_SID'),
                            dest="marketDBSID",
                            help="Oracle SID for MarketDB access")
    optionParser.add_option("--marketdb-user", action="store",
                            default=os.environ.get('MARKETDB_USER'),
                            dest="marketDBUser",
                            help="user for MarketDB access")
    optionParser.add_option("--macdb_host", action="store",
                            default=None,
                            dest="macdb_host",
                            help="MAC DB host name")
    optionParser.add_option("--macdb_port", action="store",
                            default='1433',
                            dest="macdb_port",
                            help="MAC DB port, default to 1433")
    optionParser.add_option("--macdb_name", action="store",
                            default=os.environ.get('MACB_NAME'),
                            dest="macdb_name",
                            help="MAC DB name, default to None")
    optionParser.add_option("--macdb_user", action="store",
                            default=os.environ.get('MACB_USER'),
                            dest="macdb_user",
                            help="MAC DB username, default to None")
    optionParser.add_option("--macdb_pwd", action="store",
                            default=os.environ.get('MACB_PWD'),
                            dest="macdb_pwd",
                            help="MAC DB password, default to None")

def addModelAndDefaultCommandLine(optionParser):
    """Add the -m, -i, and -r options to the command line that
    select the risk model class and the default options as well.
    """
    optionParser.add_option("-m", "--model-name", action="store",
                            default=None, dest="modelName",
                            help="risk model name")
    optionParser.add_option("-i", "--model-id", action="store",
                            default=None, dest="modelID",
                            help="risk model ID")
    optionParser.add_option("-r", "--model-revision", action="store",
                            default=None, dest="modelRev",
                            help="risk model revision")
    addDefaultCommandLine(optionParser)

def processDefaultCommandLine(options, optionsParser,disable_existing_loggers=True):#disable_existing_loggers=True is python's default
    """Configure the log system.
    """
    import os
    import sys
    try:
        logging.config.fileConfig(options.logConfigFile,disable_existing_loggers=disable_existing_loggers)
        if hasattr(os, 'ttyname'): #ttyname doesn't exist on windows
           if sys.stderr.isatty() and sys.stdout.isatty() and \
              os.ttyname(sys.stdout.fileno()) == os.ttyname(sys.stderr.fileno()):
               for h in logging.root.handlers:
                   if hasattr(h, 'stream') and h.stream == sys.stderr:
                       logging.root.removeHandler(h)
                       break
    except Exception as e:
        print(e)
        optionsParser.error('Error parsing log configuration "%s"'
                            % options.logConfigFile)

def processModelAndDefaultCommandLine(options, optionsParser):
    """Configure the log system and returns the risk model class
    corresponding to the modelName or modelID/modelRev given in options.
    If the risk model does not exists, a list of possible values is printed
    and the sys.exit() is called.
    """
    import riskmodels
    processDefaultCommandLine(options, optionsParser)
    if options.modelName != None:
        try:
            riskModelClass = riskmodels.getModelByName(options.modelName)
        except KeyError:
            print('Unknown risk model "%s"' % options.modelName)
            names = sorted(riskmodels.modelNameMap.keys())
            print('Possible values:', ', '.join(names))
            sys.exit(1)
    elif options.modelID != None and options.modelRev != None:
        rm_id = int(options.modelID)
        rev = int(options.modelRev)
        try:
            riskModelClass = riskmodels.getModelByVersion(rm_id, rev)
        except KeyError:
            print('Unknown risk model %d/%d' % (rm_id, rev))
            revs = list(riskmodels.modelRevMap.keys())
            revs = sorted('/'.join((str(i[0]), str(i[1]))) for i in revs)
            print('Possible values:', ', '.join(revs))
            sys.exit(1)
    else:
        optionsParser.error(
            "Either model name or ID and revision must be given")
    return riskModelClass

def parseISODate(dateStr):
    """Parse a string in YYYY-MM-DD format and return the corresponding
    date object.
    """
    if len(dateStr) != 10:
        print('Strange format for date: %s' % dateStr)
    assert(len(dateStr) == 10)
    assert(dateStr[4] == '-' and dateStr[7] == '-')
    return datetime.date(int(dateStr[0:4]), int(dateStr[5:7]),
                         int(dateStr[8:10]))

def parseDate(datestr):
    """Parse date in YYYY-MM-DD or YYYYMMDD format
    """
    if len(datestr) == 10:
        assert(datestr[4] == '-' and datestr[7] == '-')
        date = datestr.split('-')
        return datetime.date(int(date[0]), int(date[1]), int(date[2]))
    elif len(datestr) == 8:
        return datetime.date(int(datestr[0:4]),int(datestr[4:6]),int(datestr[6:8]))
    else:
        raise Exception('Unable to create date object from %s' % datestr)

def robustLinearSolver(yIn, XIn, robust=False,
        k=1.345, maxiter=50, tol=1.0e-6, weights=None,
        computeStats=True):
    """ Estimates y = Xb + e via either least-squares regression,
    or robust regression
    If robust, assumes robust regression is performed one column
    of y at a time
    """
    testMode = False
    # Useful to test this code every now and then
    # Basic solution is [0.24324324 -0.72972973]
    # Weighted solution is [ 0.4, -0.8]
    if testMode:
        XIn = numpy.array(((1.0, 0.0), (2.0, 1.0), (3.0, 0.0),
                           (4.0, 1.0), (5.0, 0.0)))
        yIn = numpy.array((1.0, 0.0, 1.0, 0.0, 1.0))
        #weights = numpy.array((1.0, 1.0, 1.0, 0.0, 0.0))
        k = 5.0
        robust = False

    # Initialise variables
    y = ma.array(yIn, copy=True)
    X = ma.array(XIn, copy=True)
    X = screen_data(X, fill=True)
    y = screen_data(y, fill=True)
    vector = False

    # Some manipulation of array dimensions
    if len(y.shape) < 2:
        y = y[:,numpy.newaxis]
        vector = True
    if len(X.shape) < 2:
        X = X[:,numpy.newaxis]
    if X.shape[0] != y.shape[0]:
        if X.shape[1] == y.shape[0]:
            X = numpy.transpose(X)
    assert(y.shape[0]==X.shape[0])
    ts = numpy.zeros((X.shape[1], y.shape[1]), float)
    pvals = numpy.zeros((X.shape[1], y.shape[1]), float)
    rsquare = numpy.zeros((y.shape[1]), float)
    if weights is None:
        wt = numpy.ones((y.shape[0]), float)
    else:
        wt = numpy.sqrt(weights)

    # Solve implicitly via lstsq routine
    if not robust:
        X_wt = numpy.transpose(wt * numpy.transpose(X))
        y_wt = numpy.transpose(wt * numpy.transpose(y))
        b = numpy.linalg.lstsq(X_wt, y_wt, rcond=-1)[0]
        if len(b.shape) < 2:
            if X.shape[1] < 2:
                b = b[numpy.newaxis,:]
            else:
                b = b[:,numpy.newaxis]
        err1 = y - ma.dot(X, b)
        # Loop round columns and compute t-stats
        # Note that for now, they're computed unweighted
        yMasked = ma.masked_where(y==0.0, y)
        if computeStats:
            for j in range(y.shape[1]):
                stdErr = computeRegressionStdError(err1[:,j], X, weights=wt*wt)
                ts[:,j] = b[:,j] / stdErr
                pvals[:,j] = 2.0 - (2.0 * stats.t.cdf(abs(ts[:,j]), X.shape[0]-b.shape[0]))
                err = err1[:,j] * wt
                yyy = yMasked[:,j] * wt
                r2 = float(ma.inner(err, err)) / ma.inner(yyy, yyy)
                rsquare[j] = 1.0 - r2
    
    # Or using robust regression
    else:
        logging.debug('Using robust statsmodels regression for solver')
        try:
            import statsmodels.api as sm
        except ImportError:
            import scikits.statsmodels.api as sm
        except:
            import scikits.statsmodels as sm

        b = numpy.zeros((X.shape[1], y.shape[1]), float)
        xMask = ma.masked_where(X==0.0, X)

        # Do some jiggling about in case of zero/missing columns
        nCols = numpy.sum(xMask, axis=0)
        okColIdx = numpy.flatnonzero(nCols)
        nFailed = 0

        if len(okColIdx) > 0:
            X_reg = ma.take(X, okColIdx, axis=1)
            X_wt = numpy.transpose(wt * numpy.transpose(X_reg))

            # Loop round one column at a time
            for jDim in range(y.shape[1]):
                yCol = y[:,jDim]
                y_wt = numpy.ravel(yCol * wt)
                # Test for completely zero y vector
                yMask = ma.masked_where(y_wt==0.0, y_wt)
                nRows = len(numpy.flatnonzero(ma.getmaskarray(yMask)==0))
                if nRows >= len(okColIdx):
                    try:
                        model = sm.RLM(y_wt, X_wt, M = sm.robust.norms.HuberT(t=k))
                        results = model.fit(maxiter=maxiter)
                        params = numpy.array(results.params)
                        bse = results.bse
                        tvalues = params/bse
                        err = yCol - numpy.dot(X_reg, params)
                        for (ii, idx) in enumerate(okColIdx):
                            b[idx, jDim] = params[ii]
                            ts[idx, jDim] = tvalues[ii]
                        if computeStats:
                            pvals[:,jDim] = 2.0 - (2.0 * stats.t.cdf(abs(ts[:,jDim]), X_reg.shape[0]-b.shape[0]))
                            eWt = err * numpy.sqrt(results.weights) * wt
                            yWt = yMask * numpy.sqrt(results.weights) * wt
                            r2 = float(ma.inner(eWt, eWt)) / ma.inner(yWt, yWt)
                            rsquare[jDim] = 1.0 - r2
                    except:
                        logging.warning('%s Robust Regression failed', jDim)
                        nFailed += 1
            if nFailed == len(okColIdx):
                logging.warning('Every robust regression failed. Something is wrong')

    # Compute residual
    b = screen_data(b, fill=True)
    y_hat = numpy.dot(X, b)
    e = y - y_hat
    results = Struct()
    # Return results
    if vector:
        results.params = b[:,0]
        results.error = e[:,0]
        results.y_hat = y_hat[:,0]
        if computeStats:
            results.tstat = ts[:,0]
            results.pvals = pvals[:,0]
            results.rsquare = rsquare
    else:
        results.params = b
        results.error = e
        results.y_hat = y_hat
        if computeStats:
            results.tstat = ts
            results.pvals = pvals
            results.rsquare = rsquare

    if testMode:
        logging.info('Solution: %s', results.params)
        logging.info('Error: %s', results.error)
        if computeStats:
            logging.info('Average R-Square: %s', ma.average(results.rsquare, axis=None))
        exit(0)

    return results

def generalizedLeastSquares(y, x, omegaInv, deMean=False, allowPseudo=False, \
        implicit=False):
    """Estimates y = Xb + e via Generalized Least Squares
    Appropriate where e is assumed to violate Gauss-Markov conditions
    omega is an estimator for ee', that is, a symmetric, positive semidefinite
    matrix proportional to the covariance matrix of the residuals.
    The parameter omegaInv is the inverse of omega.
    Inputs are assumed to be matrices/arrays without masked/missing values.
    x'x and omegaInv should both be full rank.
    Setting allowPseudo to be True enables use of pseudo-inverse for rank-deficient
    X matrices. Setting implicit to be True solves the system
    via the lstsq option, thus avoiding the cardinal sin of computing a matrix
    inverse directly
    """
    # Initialise variables
    if implicit:
        allowPseudo = False
    # Demean, if required 
    if deMean: 
        ybar = numpy.average(y,axis=0)
        xbar = numpy.average(x,axis=0)
        y = y - ybar
        x = x - xbar
    x = ma.filled(x, 0.0)
    y = ma.filled(y, 0.0)
    vector = False
    if len(y.shape) < 2:
        y = y[:,numpy.newaxis]
        if len(x.shape) < 2:
            x = x[:,numpy.newaxis]
        vector = True
    # Solve implicitly via lstsq routine
    if implicit:
        xv = numpy.dot(numpy.sqrt(omegaInv), x)
        yv = numpy.dot(numpy.sqrt(omegaInv), y)
        b = numpy.linalg.lstsq(xv, yv, rcond=-1)[0]
        if len(b.shape) < 2:
            b = b[:,numpy.newaxis]
    # Or directly, via explicit inverse
    elif allowPseudo:
        xv = numpy.dot(x.transpose(), omegaInv)
        try:
            xvxinv = linalg.inv(numpy.dot(xv, x))
        except:
            # Use pseudo-inverse if regressor singular
            logging.warning('Singular matrix in least-squares. Computing pseudo-inverse', exc_info=True)
            xvxinv = linalg.pinv(numpy.dot(xv, x))
        b = numpy.dot(xvxinv, numpy.dot(xv, y))
    else:
        xv = numpy.dot(x.transpose(), omegaInv)
        xvxinv = linalg.inv(numpy.dot(xv, x))
        b = numpy.dot(xvxinv, numpy.dot(xv, y))
    # Compute residual
    if deMean:
        x = x + xbar
        y = y + ybar
    e = y - numpy.dot(x, b)
    # Return results
    if vector:
        return (b[:,0], e[:,0])
    else:
        return (b, e)

def ordinaryLeastSquares(y, x, deMean=False, allowPseudo=False, implicit=False):
    """Estimates y = bX + e via equal-weighted OLS
    Assumes input arays/matrices do not have missing/masked values.
    """
    (b, e) = generalizedLeastSquares(y, x, numpy.eye(x.shape[0]),deMean, allowPseudo,\
            implicit)
    return (b, e)

def restrictedLeastSquares(y, x, r, g, w=None):
    """Estimates y = bX + e via constrained regression, satisfying
    a set of restrictions rb = g.
    Additionally, an optional vector of weights w can be supplied.
    Assumes input arays/matrices do not have missing/masked values.
    """
    assert(r.shape[1]==x.shape[1] and r.shape[0]==g.shape[0])
    if w is None:
        w = numpy.ones(x.shape[0])
    w = numpy.diag(w)
    (b0, e0) = generalizedLeastSquares(y, x, w)
    xw = numpy.dot(numpy.transpose(x), w)
    xwxinvr = numpy.dot(linalg.inv(numpy.dot(xw, x)), numpy.transpose(r))
    blob = linalg.inv(numpy.dot(r, xwxinvr))
    adj = numpy.dot(numpy.dot(xwxinvr, blob), numpy.dot(r, b0) - g)
    b1 = b0 - adj
    e1 = y - numpy.dot(x, b1)
    return (b1, e1)

def robustAverage(v, wt=[], k=1.345, maxiter=500):
    """Computes a (possibly) weighted average using scikits.statsmodels rlm to
    reduce the impact of outliers
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        import scikits.statsmodels.api as sm
    except:
        import scikits.statsmodels as sm
    v_part = ma.ravel(v)
    n = len(v_part)
    if len(wt) != n:
        wt = numpy.ones((n), float)
     
    # Flag missing returns and weights
    badRetIdx = numpy.flatnonzero(ma.getmaskarray(v_part))
    zeroWgt = ma.masked_where(wt <= 0.0, wt)
    badWgtIdx = numpy.flatnonzero(ma.getmaskarray(zeroWgt))
    badIdx = set(list(badRetIdx) + list(badWgtIdx))
    # Take subset of observations corresponding to non-missing returns
    # and non-zero weights
    okIdx = list(set(range(n)).difference(badIdx))
    wt_part = numpy.take(ma.filled(wt, 0.0), okIdx, axis=0)
    v_part = numpy.take(ma.filled(v_part, 0.0), okIdx, axis=0)
    rlmDownWeights = numpy.ones((n), float)

    # Perform weighted average
    if len(v_part) < 2:
        avVal = 0.0
    else:
        # Set up regressor and regressand
        v_part = (v_part * numpy.sqrt(ma.filled(wt_part, 0.0)))
        inter = numpy.array(numpy.sqrt(ma.filled(wt_part, 0.0)))
        inter = inter.reshape((inter.shape[0],1))
        #logging.info('V: %s', v_part)
        #logging.info('INTER: %s', inter)

        # Do the regression
        model = sm.RLM(v_part, inter, M = sm.robust.norms.HuberT(t=k))
        results = model.fit(maxiter=maxiter)
        if hasattr(results, 'weights'):
            downWeights = numpy.array(results.weights)
            numpy.put(rlmDownWeights, okIdx, downWeights)
        avVal = results.params[0]
    avVal = screen_data(avVal)
    return (avVal, rlmDownWeights)

def cubic_spline(x, y, lower_boundary=None, upper_boundary=None):
    """Given a function y, sampled at points x, and optional
    first derivative boundary conditions, returns a set of
    cubic spline coefficients
    """
    n = len(x)
    u = numpy.zeros((n), float)
    y2 = numpy.zeros((n), float)
    un = 0.0
    qn = 0.0
    # Set up lower boundary condition
    if lower_boundary != None:
        y2[0] = -0.5
        u[0] = (3.0 / (x[1]-x[0])) * (((y[1]-y[0]) / (x[1]-x[0])) - lower_boundary)

    # Tridiagonal decomposition loop
    for i in range(1, n-1):
        sig = (x[i]-x[i-1])/(x[i+1]-x[i-1])
        p = sig * y2[i-1] + 2.0
        y2[i] = (sig - 1.0) / p
        u[i] = ((y[i+1]-y[i])/(x[i+1]-x[i])) - ((y[i]-y[i-1])/(x[i]-x[i-1]))
        u[i] = (6.0 * u[i]/(x[i+1]-x[i-1]) - (sig*u[i-1])) / p

    # Set up upper boundary condition
    if upper_boundary != None:
        qn = 0.5
        un = (3.0/(x[n-1]-x[n-2])) * (upper_boundary - \
                (y[n-1]-y[n-2]) / (x[n-1]-x[n-2]))

    # Back-substitution loop
    y2[n-1] = (un-qn*u[n-2]) / (qn*y2[n-2]+1.0)
    for i in range(n-2, -1, -1):
        y2[i] = (y2[i]*y2[i+1]) + u[i]

    return y2

def spline_interpolate(x_orig, y_orig, spl, x_new):
    """ Given the original sampled function y_orig(x_orig)
    and the spline nodal values, returns interpolated
    estimates of y at sample points x_new
    """
    m = len(x_new)
    n = len(x_orig)
    y_new = numpy.zeros((m), float)
    # Find interval in x_orig that contains the first value of x_new
    lowerIndex = 0
    upperIndex = n-1
    while (upperIndex-lowerIndex > 1):
        k = int((lowerIndex + upperIndex) / 2.0)
        if x_orig[k] > x_new[0]:
            upperIndex = k
        else:
            lowerIndex = k

    # Loop round vector x_new and calculated interpolated values
    for i in range(m):
        # Reset the bounding indices if necessary
        while (x_new[i] > x_orig[upperIndex] and upperIndex < n):
            upperIndex += 1
        lowerIndex = upperIndex - 1
        while (x_new[i] < x_orig[lowerIndex] and lowerIndex >= 0):
            lowerIndex -= 1
        # Compute spline polynomial value
        h = x_orig[upperIndex] - x_orig[lowerIndex]
        if h > 0.0:
            a = (x_orig[upperIndex] - x_new[i]) / h
            b = (x_new[i] - x_orig[lowerIndex]) / h
            y_new[i] = (a*y_orig[lowerIndex]) + (b*y_orig[upperIndex])\
                    + ((a*a*a-a) * spl[lowerIndex] + \
                       (b*b*b-b) * spl[upperIndex]) * (h*h)/6.0

    return y_new
                    
def dynamic_volatility_adjustment(originalData, sampleLength, scaleType='step',
        upperBound=1.25, lowerBound=0.8):
    """Applies 'secret sauce/supertrick/DVA' to a returns series.
    data should be an asset-by-time array with no masked values
    where the most recent observations at position 0.
    """
    data = ma.array(originalData, copy=True)
    sampleLength = int(sampleLength)
    # Set up parameters
    # T is the length of each sample used to compute statistics
    # t is the current start point of the chunk to be scaled
    # Except for the first chunk of data, overlapping chunks
    # are used of length T, centred on point t
    # The very first chunk (that which is thereafter used as a 
    # reference) runs from t=0 to t=T
    T = sampleLength
    if scaleType == 'step':
        T = int(numpy.ceil(T / 2.0))
    t = 0
    tMax = data.shape[1]
    tMinusT = 0

    logging.info('DVA Reference sample Length: %d, Type=%s',
                sampleLength, scaleType)

    # Fill-in rule to deal with troublesome values
    # Don't like this bit. Needs more effort
    averageDataVal = ma.median(abs(data), axis=1)
    dataCopy = ma.masked_where(data == 0.0, data)
    for iRow in range(dataCopy.shape[0]):
        dataCopy[iRow,:] = dataCopy[iRow,:].filled(averageDataVal[iRow])

    # Loop through the chunks of returns
    while tMinusT < tMax:
        # Pick out our sample of returns
        tPlusT = min(t+T, tMax)
        dataSample = abs(ma.take(dataCopy, list(range(tMinusT, tPlusT)), axis=1)).filled(0.0)

        # Derive statistics on the returns sample
        tMAD = ma.average(dataSample, axis=1)[:,numpy.newaxis]
        tMAD = ma.masked_where(tMAD<=0.0, tMAD)

        if t > 0:
            # Calculate scaling factor for current chunk
            baseMAD = ma.masked_where(baseMAD<=0.0, baseMAD)
            Lambda = (baseMAD / tMAD)

            # Three types of DVA: step, piecewise linear or spline
            if scaleType=='step':
                # Trim large values of scale factor
                Lambda = ma.where(Lambda > upperBound, upperBound, Lambda)
                Lambda = ma.where(Lambda < lowerBound, lowerBound, Lambda)
                # Scale chunk of data
                if t < tMax:
                    data[:,list(range(t, tPlusT))] *= Lambda.filled(1.0)

            elif scaleType=='slowStep':
                # Trim large values of scale factor
                Lambda = ma.where(Lambda > upperBound, upperBound, Lambda)
                Lambda = ma.where(Lambda < lowerBound, lowerBound, Lambda)
                # If remaining data is of short length, throw it
                # into the mix
                if (tMax - tPlusT) < 21:
                    tPlusT = tMax
                # Scale chunk of data
                if t < tMax:
                    data[:,list(range(t, tPlusT))] *= Lambda.filled(1.0)

            elif scaleType=='pwlinear':
                # Trim large values
                (Lambda, bounds) = mad_dataset(Lambda, -2.0, 1.0)
                # Get slope gradient
                if t < tMax:
                    h = (Lambda.filled(1.0) - prevLambda.filled(1.0)) / T
                else:
                    T = dataSample.shape[1]
                    h = -prevLambda.filled(1.0) / T
                # Fill in interpolated values over length of chunk
                scaleMatrix = ma.zeros((data.shape[0],T), float)
                for j in range(T):
                    scaleColumn = prevLambda + (j * h)
                    ma.put(scaleMatrix[:,j],
                            list(range(data.shape[0])),
                            scaleColumn)
                # Scale chunk of data
                data[:,list(range(tMinusT, min(t, tMax)))] *= scaleMatrix
                prevLambda = ma.array(Lambda, copy=True)

            logging.debug('DVA Scaling: (Min, Med, Max): (%.3f, %.3f, %.3f)',
                        numpy.min(Lambda.filled(999.0)),
                        ma_median(Lambda), 
                        numpy.max(Lambda.filled(-999.0)))
        else:
            # For the initial chunk, set some parameters
            baseMAD = ma.array(tMAD, copy=True)
            prevLambda = ma.ones((data.shape[0],1),float)

        # If at any time, a chunk has empty values for a factor
        # ensure all consequent values are zerod
        check = numpy.flatnonzero(ma.getmaskarray(tMAD))
        ma.put(baseMAD, check, 0.0)
        # Reset various stepping parameters
        if t == 0 and scaleType != 'step':
            t += T
            T = int(numpy.ceil(T / 2.0))
        else:
            t += T
        # A few safeguards, to prevent overreaching 
        # array bounds or tiny chunks
        tMinusT = max(t-T, 0)
        if t > tMax-20 and t < tMax:
            t = tMax

    outputValues = False
    if outputValues:
        for ii in range(data.shape[1]):
            logging.info('Scaled Data, T, %s, DATA, %s' %\
                    (ii, data[0,ii]))
    return data

def computeRegressionStdError(resid, exposureMatrix, weights=None,
                                   constraintMatrix=None, white=False):
    """Given a set of residuals, a N-by-K exposure matrix and an 
    optional set of weights, computes standard errors for 
    estimators from a linear regression process.
    Can adjust for heteroskedasticity using White (1980).  The M-by-K
    matrix of linear constraint coefficients should also be provided
    in the case of constrained/restricted regression."""

    X = ma.filled(exposureMatrix, 0.0)
    n = X.shape[0]
    if weights is None:
        weights = numpy.ones([n], float)
    elif len(weights) != n:
        logging.warning('Dimension of weights vector inconsistent with data matrix')

    if len(X.shape) == 1:
        X = X[:,numpy.newaxis]
    k = X.shape[1]
    wx = (weights**0.5) * numpy.transpose(X)
    # Invert carefully, exposure matrix may be rank deficient
    # in the case of constrained regression
    xwx = numpy.dot(wx, numpy.transpose(wx))
    xwx_good = forcePositiveSemiDefiniteMatrix(xwx, min_eigenvalue=1e-8, quiet=True)
    try:
        xwxinv = linalg.inv(xwx_good)
    except:
        # Use pseudo-inverse if regressor singular
        logging.warning('Singular matrix in least-squares. Computing pseudo-inverse', exc_info=True)
        xwxinv = linalg.pinv(xwx_good)

    # Calculate variance of the residual times the exposure term
    if not white:
        resid = resid * numpy.sqrt(weights)
        sse = ma.inner(resid, resid)
        s = sse / (n-k)
        omega = s * xwxinv
    else:
        s = numpy.transpose(X) * (weights**0.5 * resid)
        s = numpy.dot(s, numpy.transpose(s))
        omega = numpy.dot(xwxinv, numpy.dot(s, xwxinv))

    # If using constrained regression...
    if constraintMatrix is None:
        stdError = numpy.sqrt(numpy.diag(omega))
    else:
        blob = numpy.dot(constraintMatrix, numpy.dot(
                         xwxinv, numpy.transpose(constraintMatrix)))
        blob = linalg.inv(blob)
        bigBlob = numpy.dot(numpy.transpose(constraintMatrix), 
                            numpy.dot(blob, constraintMatrix))
        biggerBlob = numpy.dot(bigBlob, xwxinv)
        if not white:
            var = (s * xwxinv) - \
                  (s * numpy.dot(xwxinv, biggerBlob))
        else:
            var = omega - numpy.dot(omega, biggerBlob) - \
                  numpy.dot(numpy.transpose(biggerBlob), omega) + \
                  numpy.dot(numpy.transpose(biggerBlob), 
                            numpy.dot(omega, biggerBlob))
        stdError = numpy.sqrt(numpy.diag(var))

    return stdError

def forcePositiveSemiDefiniteMatrix(matrix, min_eigenvalue=1e-10, quiet=False):
    """Checks whether a given matrix is positive semidefinite.
    If not, issue a warning and tweak it accordingly.
    """
    (eigval, eigvec) = linalg.eigh(matrix)
    check = ma.getmaskarray(ma.masked_where(eigval < min_eigenvalue, eigval))
    if min(eigval) < 0.0:
        if not quiet:
            logging.info('Zero eigenvalue - infinite condition number')
    else:
        condNum = max(eigval) / min(eigval)
        if not quiet:
            logging.info('Covariance matrix condition number: %.2f', condNum)
    if numpy.sum(check) > 0:
        violations = numpy.flatnonzero(check)
        if not quiet:
            logging.warning('Matrix not positive semidefinite, %d eigenvalues < %.2e: %s',
                    len(violations), min_eigenvalue, ma.take(eigval, violations, axis=0))
        numpy.put(eigval, violations, min_eigenvalue)
        matrix = numpy.dot(numpy.dot(
                    eigvec, numpy.diag(eigval)), numpy.transpose(eigvec))
    elif not quiet:
        logging.info('Matrix positive semidefinite, 0 eigenvalues < %.2e',
                min_eigenvalue)
    if not quiet:
        frobeniusNorm = 1000.0 * numpy.sqrt(numpy.sum(eigval * eigval))
        logging.info('Frobenius norm of cov matrix (*1000): %f', frobeniusNorm)
    return matrix

def computeWeightedMedian(dataMatrix, weights=None):
    """ Given a data matrix of N variables by T observations
    and a set of T weights, returns an N-vector of medians
    The weights are taken to mean frequencies
    """

    if len(dataMatrix.shape) == 2:
        T = dataMatrix.shape[1]
    else:
        T = dataMatrix.shape[0]

    if weights == None:
        median = ma.median(numpy.transpose(dataMatrix), axis=0)
        return median
    elif len(weights) != T:
        logging.warning('Dimension of weights vector inconsistent with data matrix')
        
    wtMask = ma.masked_where(weights <= 0.0, weights)
    wtMask = numpy.flatnonzero(ma.getmaskarray(wtMask)==0)
    minWeight = min(ma.take(weights, wtMask, axis=0))
    weights = weights / minWeight
    wghts = weights.astype(numpy.int)
    dataMatrix = numpy.repeat(numpy.transpose(dataMatrix), wghts, axis=0)
    median = ma.median(dataMatrix, axis=0)
    return median


def computeMAD(dataMatrix, weights = None, deMean = True, 
        centreMean = True, lag = 0):
    """ Given a data matrix of N variables by T observations
    returns a vector of N mean absolute deviations. Observations
    may be weighted if required
     deMean - if set to false, a mean of zero is assumed
     centreMean - if set to false, the median absolute deviation
            is calculated. weights are interpreted as frequencies
     lag - if non-zero, the square root of the data is multiplied
            into the square root of the lagged data
    """
    
    if len(dataMatrix.shape) == 2:
        T = dataMatrix.shape[1]
    else:
        T = dataMatrix.shape[0]
    
    if weights == None:
        weights = numpy.ones([T], float)
    elif len(weights) != T:
        logging.warning('Dimension of weights vector inconsistent with data matrix')

    weights = T * weights / numpy.sum(weights, axis=0)

    # Demean the observations
    if deMean == True:
        if centreMean == True:
            mean = dataMatrix * weights
            mean = numpy.sum(numpy.transpose(mean), axis=0) / T
        else:
            mean = computeWeightedMedian(dataMatrix, weights=weights)

        weightedData = numpy.transpose(numpy.transpose(dataMatrix) - mean)
    else:
        weightedData = dataMatrix
        
    if centreMean == True:
        weightedData = weightedData * weights
        
    # Lag the data if required
    if lag > 0:
        if len(weightedData.shape) > 1:
            weightedData1 = weightedData[:, lag:T]
            weightedData = weightedData[:, 0:T-lag]
        else:
            weightedData1 = weightedData[lag:T]
            weightedData = weightedData[0:T-lag]
        weights = weights[0:T-lag]
    else:
        weightedData1 = weightedData

    weightedData = weightedData * weightedData1
    weightedData = numpy.sign(weightedData) * numpy.sqrt(abs(weightedData))

    # And compute the mean absolute deviation
    if centreMean == True:
        meanAbsDev = numpy.sum(numpy.transpose(weightedData), axis=0)
        if deMean == True:
            meanAbsDev = meanAbsDev / (T-1)
        else:
            meanAbsDev = meanAbsDev / T
    else:
        meanAbsDev = computeWeightedMedian(weightedData, weights=weights)

    return meanAbsDev
        
def cov2corr(covMatrix, fill=False, returnCov=False, 
             tol=1.0e-12, conditioning=False):
    """Trivial routine to convert a covariance matrix to 
    a correlation matrix, but one which is used quite a bit
    so this will hopefully standardise treatment
    Also returns original cov matrix with the problem
    rows and columns masked
    """
    covMatrix = screen_data(covMatrix)
    if conditioning:
        # Perform some conditioning of the covariance matrix
        (d, v) = linalg.eigh(covMatrix)
        condNo = d / max(d)
        d = ma.masked_where(condNo<tol, d)
        covMatrix = ma.dot((v*d), ma.transpose(v)).filled(0.0)

    varVector = numpy.diag(covMatrix)
    # Deal with zero or negative variances
    varVector = ma.masked_where(abs(varVector)<=tol, varVector)
    stdVector = ma.sqrt(varVector)
    # Create correlation matrix with problem rows and columns masked
    corMatrix = (covMatrix / ma.outer(stdVector, stdVector))
    # Fill masked values if required
    for i in range(corMatrix.shape[0]):
        corMatrix[i,i] = 1.0
    if fill:
        corMatrix = corMatrix.filled(0.0)
        stdVector = stdVector.filled(0.0)
    # Reconstitute covariance matrix if required
    if returnCov:
        corMatrix = ma.transpose(ma.transpose(
                        corMatrix * stdVector) * stdVector)
    return (stdVector, corMatrix)

def compute_NWAdj_covariance(data, nwLag, weights=None, deMean=True,
                    axis=0, corrOnly=False, varsOnly=False):
    """Outer wrapper for compute_covariance that has the NW adjustment
    baked in
    """
    covMatrix = compute_covariance(data, weights=weights, deMean=deMean,
            axis=axis, corrOnly=False, varsOnly=False)
    for lag in range(1,nwLag+1):
        adj = compute_covariance(data, lag=lag, weights=weights, deMean=deMean,
                axis=axis, corrOnly=False, varsOnly=False)
        covMatrix += (1.0 - float(lag)/(float(nwLag)+1.0)) * 2.0 * adj

    if corrOnly:
        return cov2corr(covMatrix, fill=True)[1]
    if varsOnly:
        varVect = cov2corr(covMatrix, fill=True)[0]
        return varVect*varVect

    return covMatrix

def compute_covariance(data, weights=None, deMean=True, lag=0, 
                       axis=0, corrOnly=False, varsOnly=False):
    """A generic multivariate variance/covariance/correlation 
    computation routine.
    Supports weighted and/or lagged data, computation of only
    the variances or correlations.
    axis=0 means each row in the input array is an observation.
    """
    if corrOnly:
        varsOnly = False
    elif varsOnly:
        corrOnly = False

    data = ma.filled(data, 0.0)
    # If univariate data series, make into 2-D array first
    if len(data.shape) == 1:
        data = data[numpy.newaxis]
        axis = 1
    if axis == 1:
        data = numpy.transpose(data)

    obs = data.shape[0]
    if weights is None:
        weights = numpy.ones([obs], float)
    assert(len(weights) == obs)

    # This bit of jiggery-pokery ensures a degree of consistency
    # with different scaling of the weights
    weights = obs * weights / numpy.sum(weights, axis=0)

    # Demean the observations if required
    if deMean:
        mean = numpy.transpose(data) * weights
        mean = numpy.sum(mean, axis=1) / obs
        divisor = obs - 1.0 
    else:
        mean = 0.0
        divisor = obs
    weightedData = numpy.transpose(data - mean)

    # Weight the observations
    weightedData = weightedData * numpy.sqrt(weights)

    # Lag the data if required
    if lag > 0:
        weightedData1 = weightedData[:,lag:obs]
        weightedData = weightedData[:,:obs-lag]
    else:
        weightedData1 = weightedData

    # Compute the covariance matrix
    if varsOnly:
        cov = numpy.array([numpy.inner(weightedData[i,:], weightedData1[i,:]) \
               for i in range(weightedData.shape[0])])
    else:
        cov = numpy.dot(weightedData, numpy.transpose(weightedData1))
    cov /= divisor

    # Convert to a correlation matrix if required
    # Need a little care in case of zeros on the diagonal
    if corrOnly:
        (diagCov, cov) = cov2corr(cov, fill=True)
    return cov

def multi_mad_data(data, restrictZeroAxis=None, restrictOneAxis=None,
        lowerMADBound=-5.0, upperMADBound=5.0, axisZeroTol=0.1, axisOneTol=0.1,
        zero_tolerance=1.0, method='median', treat='clip', loopRound=1, minPerc=0.95):
    """ More sophisticated MADing technique - MADs along
    both axes of the data, and then treats as outliers all observations
    which are outliers in both dimensions. Also allows for flexible
    number of MADs - the number of bounds is increased one by one
    until the change in number of outliers falls below a given tolerance
    """

    # This is an older test version, used only by the currency models

    # Special treatment for 1-D arrays
    vector = False
    if len(data.shape) == 1:
        vector = True
        data = data[numpy.newaxis]

    # Set flexible MAD bound parameters
    flexiBound = Struct()
    # Upper and lower intial MAD bounds
    flexiBound.lower = lowerMADBound
    flexiBound.upper = upperMADBound
    # Cut-off tolerance
    flexiBound.tol = axisZeroTol
    # Minimum proportion of observations that are treated as inliers
    flexiBound.minPerc = minPerc

    # First get MAD bounds per column of data
    if not vector:
        dataCopy1 = ma.array(data, copy=True)
        (dataCopy1, zeroAxisBounds) = mad_dataset_legacy(dataCopy1,
                flexiBound.lower, flexiBound.upper, restrict=restrictZeroAxis,
                axis=0, zero_tolerance=zero_tolerance, method=method,
                treat=treat, loopRound=loopRound, flexiBound=flexiBound)

    # Next get MAD bounds per row of data
    flexiBound.tol = axisOneTol
    dataCopy2 = ma.array(data, copy=True)
    (dataCopy2, oneAxisBounds) = mad_dataset_legacy(dataCopy2,
            flexiBound.lower, flexiBound.upper, restrict=restrictOneAxis,
            axis=1, zero_tolerance=zero_tolerance, method=method,
            treat=treat, loopRound=loopRound, flexiBound=flexiBound)

    # Loop through the data array and treat all elements that
    # are outliers along both dimensions
    dataMsk = ma.getmaskarray(data)
    dataCopy = data.filled(0.0)
    lowerBound = numpy.zeros((data.shape[0], data.shape[1]), float)
    upperBound = numpy.zeros((data.shape[0], data.shape[1]), float)
    if vector:
        for i in range(data.shape[1]):
            lowerBound[0,i] = oneAxisBounds[0][0]
            upperBound[0,i] = oneAxisBounds[1][0]
    else:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                lowerBound[i,j] = min(zeroAxisBounds[0][j], oneAxisBounds[0][i])
                upperBound[i,j] = max(zeroAxisBounds[1][j], oneAxisBounds[1][i])

    dataCopy = numpy.clip(dataCopy, lowerBound, upperBound)
    diffs = ma.masked_where(data == dataCopy, dataCopy)
    data = ma.masked_where(dataMsk, dataCopy)
    num = len(numpy.flatnonzero(diffs))

    logging.info('%d out of %d values identified as outliers (%.2f %%)',
                num, data.shape[1]*data.shape[0], 100.0*num / \
                (data.shape[1]*data.shape[0]))

    if vector:
        data = data[0,:]
    return data

def mad_dataset_legacy(data, lowerMADBound, upperMADBound, restrict=None,
            axis=0, zero_tolerance=0.25, method='median', treat='clip',
            loopRound=1, flexiBound=False):
    """Truncates items in a data set beyond a given number of MADs.
    MADs are computed along the specified axis. axis=0 means one
    MAD per column of data.
    zero_tolerance specifies the maximum fraction of allowable zero
    entries.  Additional zero entries beyond this will be removed for
    MAD computation.
    restrict is an optional list of index posiions (like an estU)
    specifiying the elements of the dataset to use for MAD computation.
    treat can be 'clip', in which case the data is clipped to the MAD
    bounds, 'zero', in which case values exceeding the bounds are set
    to zero, or 'mask' in which case the offending values are masked
    loopRound refers to the number of iterations round the data
    Returns a masked array containing the truncated data and a list
    containing the lower- and upper-bounds from the MAD'ing.
    'method' can be either 'median', 'mad' or 'std' and refers
    to the measure of dispersion used
    """
    logging.debug('mad_dataset_legacy: begin')
    # This is an older test version, used only by the currency models
    vector = False
    # Output data limits
    logging.info('Data bounds before MADing: [%.3f, %.3f, %.3f]' % \
            (ma.min(data, axis=None), ma.average(data, axis=None),
                ma.max(data, axis=None)))
    if len(data.shape) == 1:
        vector = True
        data = data[numpy.newaxis]
        axis = 1
    if axis == 1:
        data = ma.transpose(data)
    medians = numpy.zeros(data.shape[1])
    mads = numpy.zeros(data.shape[1])

    # Loop round requisite number of iterations
    for iLoop in range(loopRound):

        # Subset the data if required
        if restrict is not None:
            data_restr = ma.take(data, restrict, axis=0)
        else:
            data_restr = ma.array(data, copy=True)

        # Loop thru data fields
        for i in range(data_restr.shape[1]):
            # Discard masked entries
            sample = ma.take(data_restr[:,i], numpy.flatnonzero(
                ma.getmaskarray( data_restr[:,i])==0), axis=0 )

            medians[i] = 0.0
            mads[i] = 0.0

            if len(sample) > 0:
                # Check for overabundance of zeros
                maskzero = ma.getmaskarray(ma.masked_where(sample==0.0, sample))
                nonzero_idx = numpy.flatnonzero(maskzero==0)
                zero_idx = numpy.flatnonzero(maskzero)
                # If too many zeros, remove some
                n = float(len(zero_idx)) / float(len(sample))
                if n > zero_tolerance:
                    logging.debug('High fraction (%.3f) of zeros in data field %d', n, i)
                    pad = int(numpy.ceil(len(nonzero_idx) * zero_tolerance \
                            / (1.0 - zero_tolerance)))      # max number of zero entries
                    sample = numpy.take(sample, nonzero_idx, axis=0)
                    new_zeros = numpy.zeros(pad)
                    sample = numpy.concatenate([sample, new_zeros], axis=0)

                # Compute MAD
                if len(sample) > 0:
                    if method == 'median':
                        medians[i] = ma.median(sample, axis=0)
                    else:
                        medians[i] = ma.average(sample, axis=0)
                    abs_dev = abs(sample - medians[i])
                    if method == 'mean':
                        mads[i] = ma.average(abs_dev, axis=0)
                    elif method == 'median':
                        mads[i] = ma.median(abs_dev, axis=0)
                    else:
                        mads[i] = numpy.std(sample)

        if flexiBound == False:
            # Determine bounds
            lowerBounds = medians + lowerMADBound * mads
            upperBounds = medians + upperMADBound * mads

            masked1 = numpy.sum(ma.getmaskarray(data), axis=None)
            mask2 = ma.masked_where(data < lowerBounds, data)
            mask2 = ma.masked_where(mask2 > upperBounds, mask2)
            masked2 = numpy.sum(ma.getmaskarray(mask2), axis=None)
            logging.debug('%d out of %d values beyond [%d,%d] MADs (%.2f %%)',
                        masked2-masked1, data.shape[1]*data.shape[0],
                        lowerMADBound, upperMADBound,
                        100.0 * (masked2-masked1) / (data.shape[1]*data.shape[0]))
        else:
            # New - flexible MAD bound loop
            # Get parameters from flexiBound structure
            lowerMADBound = flexiBound.lower
            upperMADBound = flexiBound.upper
            tol = flexiBound.tol
            minPerc = flexiBound.minPerc
            # Initialise loop parameters
            lowStop = False
            upStop = False
            maxIter = 50
            iter = 1
            alreadyMasked = numpy.sum(ma.getmaskarray(data), axis=None)
            totalObs = data.shape[1] * data.shape[0] - alreadyMasked

            while (lowStop == False) or (upStop == False):
                # Check data below lower bound
                if lowStop == False:
                    lowerBounds = medians + lowerMADBound * mads
                    outsideLower = ma.masked_where(data < lowerBounds, data)
                    lowSum = numpy.sum(ma.getmaskarray(outsideLower), axis=None) - alreadyMasked
                    lowPerc = 1.0 - (float(lowSum) / float(totalObs))
                    if lowSum == 0:
                        # If no outliers below this bound, freeze the bound
                        lowStop = True
                    elif iter > 1:
                        # Find change in proportion of outliers
                        lowDecrease = (lowPerc / prevLowPerc) - 1.0
                        if lowDecrease < tol:
                            # If change is low enough, freeze the bound
                            lowStop = True

                # Check data above upper bound
                if upStop == False:
                    upperBounds = medians + upperMADBound * mads
                    outsideUpper = ma.masked_where(data > upperBounds, data)
                    upSum = numpy.sum(ma.getmaskarray(outsideUpper), axis=None) - alreadyMasked
                    upPerc = 1.0 - (float(upSum) / float(totalObs))
                    if upSum == 0:
                        # If no outliers below this bound, freeze the bound
                        upStop = True
                    elif iter > 1:
                        # Find change in proportion of outliers
                        upDecrease = (upPerc / prevUpPerc) - 1.0
                        if upDecrease < tol:
                            # If change is low enough, freeze the bound
                            upStop = True

                # Check total number of data outside bounds
                totalOutside = lowSum + upSum
                percInside = 1.0 - (float(totalOutside) / float(totalObs))
                if percInside < minPerc:
                    # If we haven't attained the minimum proportion of
                    # inliers, unfreeze the bound parameters if it
                    # makes sense to do so
                    if lowSum > 0:
                        lowStop = False
                    if upSum > 0:
                        upStop = False

                logging.debug('%d out of %d values beyond [%d,%d] MADs (%.2f %%)',
                            totalOutside, totalObs, lowerMADBound, upperMADBound,
                            100.0 * (1.0 - percInside))

                # Initialise for the next round, if relevant
                iter += 1
                if iter > maxIter:
                    lowStop = True
                    upStop = True
                # Increment MAD bounds by 1
                if lowStop == False:
                    prevLowPerc = lowPerc
                    lowerMADBound -= 1
                if upStop == False:
                    prevUpPerc = upPerc
                    upperMADBound += 1

        # Treat the data via the relevant method
        if treat == 'zero':
            lowerFill = 0.0
            upperFill = 0.0
        elif treat == 'clip':
            lowerFill = lowerBounds
            upperFill = upperBounds
        if treat == 'mask':
            data = ma.masked_where(data < lowerBounds, data)
            data = ma.masked_where(data > upperBounds, data)
        else:
            data = ma.where(data <= lowerBounds, lowerFill, data)
            data = ma.where(data >= upperBounds, upperFill, data)

    if axis == 1:
        data = ma.transpose(data)
    if vector:
        data = data[0,:]

    logging.info('Data bounds after MADing: [%.3f, %.3f, %.3f]' % \
            (ma.min(data, axis=None), ma.average(data, axis=None),
                ma.max(data, axis=None)))
    logging.debug('mad_dataset_legacy: end')
    return (data, [lowerBounds, upperBounds])

def symmetric_clip(data, upperBound=1000.0):
    upperBound = numpy.log(1.0 + upperBound) - 1.0
    lowerBound = -upperBound
    tmpData = numpy.log(1.0 + data) - 1.0
    tmpData = numpy.clip(tmpData, -upperBound, upperBound)
    tmpData = numpy.exp(tmpData + 1.0) - 1.0
    return tmpData

def mad_dataset(data, lowerMADBound, upperMADBound, restrict=None,
        axis=0, zero_tolerance=0.25, method='median', treat='clip'):
    """Truncates items in a data set beyond a given number of MADs.
    MADs are computed along the specified axis. axis=0 means one
    MAD per column of data.
    zero_tolerance specifies the maximum fraction of allowable zero
    entries.  Additional zero entries beyond this will be removed for
    MAD computation.
    restrict is an optional list of index posiions (like an estU)
    specifiying the elements of the dataset to use for MAD computation.
    treat can be 'clip', in which case the data is clipped to the MAD
    bounds, 'zero', in which case values exceeding the bounds are set
    to zero, or 'mask' in which case the offending values are masked
    loopRound refers to the number of iterations round the data
    Returns a masked array containing the truncated data and a list
    containing the lower- and upper-bounds from the MAD'ing.
    'method' can be either 'median', 'mad' or 'std' and refers
    to the measure of dispersion used
    """
    logging.debug('mad_dataset: begin')
    vector = False
    TOL = 1.0e-15
    # Output data limits
    logging.info('Data bounds before MADing: [%.3f, %.3f, %.3f]' % \
            (ma.min(data, axis=None), ma.average(data, axis=None),
                ma.max(data, axis=None)))
    if len(data.shape) == 1:
        vector = True
        data = data[numpy.newaxis]
        axis = 1
    if axis == 1:
        data = ma.transpose(data)
    medians = numpy.zeros(data.shape[1])
    mads = numpy.zeros(data.shape[1])

    # Subset the data if required
    if restrict is not None:
        data_restr = ma.take(data, restrict, axis=0)
    else:
        data_restr = ma.array(data, copy=True)
 
    # Loop thru data fields
    for i in range(data_restr.shape[1]):
        # Discard masked entries
        sample = ma.take(data_restr[:,i], numpy.flatnonzero(
            ma.getmaskarray( data_restr[:,i])==0), axis=0 )

        medians[i] = 0.0
        mads[i] = 1.0

        if len(sample) > 0:
            mads[i] = max(abs(ma.filled(sample, 0.0)))
            # Check for overabundance of zeros
            # Python 3 fix here
            maskzero = ma.getmaskarray(ma.masked_where(abs(sample)<TOL, sample))
            nonzero_idx = numpy.flatnonzero(maskzero==0)
            zero_idx = numpy.flatnonzero(maskzero)
            # If too many zeros, remove some
            n = float(len(zero_idx)) / float(len(sample))
            if n > zero_tolerance:
                logging.debug('High fraction (%.3f) of zeros in data field %d', n, i)
                pad = int(numpy.ceil(len(nonzero_idx) * zero_tolerance \
                        / (1.0 - zero_tolerance)))      # max number of zero entries
                sample = numpy.take(sample, nonzero_idx, axis=0)
                new_zeros = numpy.zeros(pad)
                logging.debug('Trimmed sample has %d zeros, %d non-zeros', pad, len(sample))
                sample = numpy.concatenate([sample, new_zeros], axis=0)

            # Compute MAD
            if len(sample) > 0:
                if method == 'median':
                    medians[i] = ma.median(sample, axis=0)
                else:
                    medians[i] = ma.average(sample, axis=0)
                abs_dev = abs(sample - medians[i])
                if method == 'mean':
                    mads[i] = ma.average(abs_dev, axis=0)
                elif method == 'median':
                    mads[i] = ma.median(abs_dev, axis=0)
                else:
                    mads[i] = numpy.std(sample)

    # Determine bounds
    lowerBounds = medians + lowerMADBound * mads
    upperBounds = medians + upperMADBound * mads

    masked1 = numpy.sum(ma.getmaskarray(data), axis=None)
    mask2 = ma.masked_where(data < lowerBounds, data)
    mask2 = ma.masked_where(mask2 > upperBounds, mask2)
    masked2 = numpy.sum(ma.getmaskarray(mask2), axis=None)
             
    # Treat the data via the relevant method
    if treat == 'zero':
        lowerFill = 0.0
        upperFill = 0.0
    elif treat == 'clip':
        lowerFill = lowerBounds
        upperFill = upperBounds
    if treat == 'mask':
        data = ma.masked_where(data < lowerBounds, data)
        data = ma.masked_where(data > upperBounds, data)
    else:
        data = ma.where(data <= lowerBounds, lowerFill, data)
        data = ma.where(data >= upperBounds, upperFill, data)

    if axis == 1:
        data = ma.transpose(data)

    logging.info('Data bounds after MADing: [%.3f, %.3f, %.3f]' % \
            (ma.min(data, axis=None), ma.average(data, axis=None),
                ma.max(data, axis=None)))
    logging.info('%d out of %d values beyond [%.1f,%.1f] MADs (%.2f %%)',
            masked2-masked1, data.shape[1]*data.shape[0],
            lowerMADBound, upperMADBound,
            100.0 * (masked2-masked1) / (data.shape[1]*data.shape[0]))
    if vector:
        data = data[0,:]
    logging.debug('mad_dataset: end')
    return (data, [lowerBounds, upperBounds])

def twodMAD(dataArray, nDev=[8.0, 8.0], method='huber', k=1.345, axis=None,
        nVector=True, zero_pct=0.0, estu=None, shrink=False,
        suppressOutput=False):
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
    data = screen_data(dataArray)

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
    clippedLowerValues0 = Matrices.allMasked(data.shape)
    clippedUpperValues0 = Matrices.allMasked(data.shape)
    clippedLowerValues1 = Matrices.allMasked(data.shape)
    clippedUpperValues1 = Matrices.allMasked(data.shape)
    clippedLowerValues2 = Matrices.allMasked(data.shape)
    clippedUpperValues2 = Matrices.allMasked(data.shape)

    N = data.shape[0]

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
    if estu is not None:
        skewness = stats.skew(ma.take(data, estu, axis=0), axis=None)
    else:
        skewness = stats.skew(data, axis=None)
    if suppressOutput:
        logging.debug('Data bounds before 2D MADing: (Min, Mean, Max, Skew) [%.3f, %.3f, %.3f, %.3f]' % \
                (ma.min(data, axis=None), ma.average(data, axis=None),
                    ma.max(data, axis=None), skewness))
    else:
        logging.info('Data bounds before 2D MADing: (Min, Mean, Max, Skew) [%.3f, %.3f, %.3f, %.3f]' % \
                (ma.min(data, axis=None), ma.average(data, axis=None),
                    ma.max(data, axis=None), skewness))

    # First get bounds per column of data
    if data.shape[0] > 1 and (axis==None or axis==0):
        logging.debug('MADing along all %s columns', data.shape[1])
        colOutliers, lowColBnd, upColBnd = computeOutlierBounds(
                data, zero_pct, nDev=nDev, method=method, k=k, estu=estu)
        outliers = ma.getmaskarray(colOutliers)==0
        outlierTimes = numpy.flatnonzero(ma.sum(outliers, axis=0))
    else:
        outliers = ma.getmaskarray(data)==0
        outlierTimes = list(range(data.shape[1]))
        colOutliers = None

    # Next get MAD bounds per row of data
    outlierAssets = numpy.flatnonzero(ma.sum(outliers, axis=1))
    if len(outlierAssets) > 0 and data.shape[1] > 1 and (axis==None or axis==1):
        # Loop round all observations identified as outliers along previous dimension
        logging.debug('MADing along %d of %d data rows',
                len(outlierAssets), data.shape[0])
        rowOutliers, lowRowBnd, upRowBnd = computeOutlierBounds(
                            ma.transpose(data), zero_pct, idxList=outlierAssets,
                            nDev=nDev, method=method, k=k,
                            includeInStats=estu)
        rowOutliers = ma.transpose(rowOutliers)
        outliers = outliers * (ma.getmaskarray(rowOutliers)==0)

        # Merge the two sets of outliers - row and column
        for idx in outlierAssets:
            clippedLowerValues0[idx,:] = lowRowBnd[idx,0]
            clippedUpperValues0[idx,:] = upRowBnd[idx,0]
            clippedLowerValues1[idx,:] = lowRowBnd[idx,1]
            clippedUpperValues1[idx,:] = upRowBnd[idx,1]
            clippedLowerValues2[idx,:] = lowRowBnd[idx,2]
            clippedUpperValues2[idx,:] = upRowBnd[idx,2]

        if colOutliers is not None:
            for jdx in outlierTimes:
                # Lower set of bounds
                lowerCol = clippedLowerValues0[:,jdx]
                lowerCol = ma.where(lowerCol>lowColBnd[jdx,0], lowColBnd[jdx,0], lowerCol)
                clippedLowerValues0[:,jdx] = lowerCol
                lowerCol = clippedLowerValues1[:,jdx]
                lowerCol = ma.where(lowerCol>lowColBnd[jdx,1], lowColBnd[jdx,1], lowerCol)
                clippedLowerValues1[:,jdx] = lowerCol
                lowerCol = clippedLowerValues2[:,jdx]
                lowerCol = ma.where(lowerCol>lowColBnd[jdx,2], lowColBnd[jdx,2], lowerCol)
                clippedLowerValues2[:,jdx] = lowerCol
                # Upper set of bounds
                upperCol = clippedUpperValues0[:,jdx]
                upperCol = ma.where(upperCol<upColBnd[jdx,0], upColBnd[jdx,0], upperCol)
                clippedUpperValues0[:,jdx] = upperCol
                upperCol = clippedUpperValues1[:,jdx]
                upperCol = ma.where(upperCol<upColBnd[jdx,1], upColBnd[jdx,1], upperCol)
                clippedUpperValues1[:,jdx] = upperCol
                upperCol = clippedUpperValues2[:,jdx]
                upperCol = ma.where(upperCol<upColBnd[jdx,2], upColBnd[jdx,2], upperCol)
                clippedUpperValues2[:,jdx] = upperCol

    elif colOutliers is not None:
        for jdx in outlierTimes:
            clippedLowerValues0[:,jdx] = lowColBnd[jdx,0]
            clippedUpperValues0[:,jdx] = upColBnd[jdx,0]
            clippedLowerValues1[:,jdx] = lowColBnd[jdx,1]
            clippedUpperValues1[:,jdx] = upColBnd[jdx,1]
            clippedLowerValues2[:,jdx] = lowColBnd[jdx,2]
            clippedUpperValues2[:,jdx] = upColBnd[jdx,2]

    # Mask anything that isn't an outlier
    clippedLowerValues0 = ma.masked_where(outliers==0, clippedLowerValues0)
    clippedUpperValues0 = ma.masked_where(outliers==0, clippedUpperValues0)

    # Finally clip outliers
    tmpData = ma.masked_where(outliers==0, data)
    if shrink:
        # Clip the very largest outliers first
        tmpData = ma.clip(tmpData, clippedLowerValues2, clippedUpperValues2)
        # Now shrink the upper outliers
        tmpDataU = ma.masked_where(tmpData<clippedUpperValues0, tmpData)
        ratio = clippedUpperValues2 - clippedUpperValues0
        ratio = ma.masked_where(ratio==0.0, ratio)
        ratio = (clippedUpperValues1 - clippedUpperValues0) / ratio
        tmpDataU = clippedUpperValues0 + (tmpDataU - clippedUpperValues0) * ratio
        # And do the lower outliers
        tmpDataL = ma.masked_where(tmpData>clippedLowerValues0, tmpData)
        ratio = clippedLowerValues0 - clippedLowerValues2
        ratio = ma.masked_where(ratio==0.0, ratio)
        ratio = (clippedLowerValues0 - clippedLowerValues1) / ratio
        tmpDataL = clippedLowerValues0 - (clippedLowerValues0 - tmpDataL) * ratio
        # Recombine
        tmpData = ma.masked_where(outliers, data)
        tmpData = ma.filled(tmpData, 0.0) + ma.filled(tmpDataU, 0.0) + ma.filled(tmpDataL, 0.0)
    else:
        tmpData = ma.clip(tmpData, clippedLowerValues0, clippedUpperValues0)
        tmpData2 = ma.masked_where(outliers, data)
        tmpData = ma.filled(tmpData, 0.0) + ma.filled(tmpData2, 0.0)

    # Remask formerly masked values and output some stats
    tmpData = ma.masked_where(dataMsk, tmpData)
    diffs = ma.masked_where(data == tmpData, tmpData)
    num = len(numpy.flatnonzero(diffs))
    data = tmpData
    if suppressOutput:
        logging.debug('%d out of %d values identified as outliers (%.2f %%)',
                    num, data.shape[1]*data.shape[0], 100.0*num / \
                            (data.shape[1]*data.shape[0]))
    else:
        logging.info('%d out of %d values identified as outliers (%.2f %%)',
                    num, data.shape[1]*data.shape[0], 100.0*num / \
                            (data.shape[1]*data.shape[0]))

    if vector:
        data = ma.ravel(data)
    # Output new data limits
    if estu is not None:
        skewness = stats.skew(ma.take(data, estu, axis=0), axis=None)
    else:
        skewness = stats.skew(data, axis=None)
    if suppressOutput:
        logging.debug('Data bounds after 2D MADing: (Min, Mean, Max, Skew) [%.3f, %.3f, %.3f, %.3f]' % \
                (ma.min(data, axis=None), ma.average(data, axis=None),
                    ma.max(data, axis=None), skewness))
    else:
        logging.info('Data bounds after 2D MADing: (Min, Mean, Max, Skew) [%.3f, %.3f, %.3f, %.3f]' % \
                (ma.min(data, axis=None), ma.average(data, axis=None),
                    ma.max(data, axis=None), skewness))
    return data

def computeOutlierBounds(data, zero_pct, nDev=[3.0, 3.0], idxList=None,
        method='huber', k=1.345, estu=None, includeInStats=None, tol=1.0e-6,
        shrink=False):
    """ Applies either IQR/Huber or MAD bounds method to a data array per column
    Returns a matrix of scaling values for the data
    """
    downWeightArray = Matrices.allMasked(data.shape)
    try:
        import statsmodels.api as sm
    except ImportError:
        import scikits.statsmodels.api as sm
    except:
        import scikits.statsmodels as sm
    if idxList is None:
        idxList = list(range(data.shape[1]))
         
    # Sort out estu assets
    if estu is None:
        estu = list(range(data.shape[0]))
    estuArray = Matrices.allMasked((data.shape[0],), bool)
    ma.put(estuArray, estu, True)
    if includeInStats is None:
        includeInStats = list(range(data.shape[1]))
    inclArray = Matrices.allMasked((data.shape[1],), bool)
    ma.put(inclArray, includeInStats, True)
     
    # Initialise parameters
    medRatio = numpy.zeros((data.shape[1]), float)
    lower = []
    upper = []
    locList = []
    scaList = []
    actualIdx = []
      
    for j in idxList:
        # Pick out column of data for MADing
        col = data[:,j]
        col = ma.masked_where(estuArray==False, col)
        col = screen_data(col)
        missingIdx = list(numpy.flatnonzero(ma.getmaskarray(col)))
 
        # Check for overabundance of zeros
        zero_idx = ma.where(col==0.0)[0]
        zeroCol = numpy.zeros((len(col)), int)
        if len(zero_idx) > zero_pct * (len(col)-len(missingIdx)):
            # If too many zeros, mask some
            nZero = int(zero_pct * (len(col)-len(missingIdx)))
            zeroMaskIdx = zero_idx[nZero:]
            numpy.put(zeroCol, zeroMaskIdx, 1)
            col = ma.masked_where(zeroCol, col)
            missingIdx.extend(zeroMaskIdx)
       
        if len(missingIdx) < len(col)-1:
            actualIdx.append(j)
            okIdx = numpy.flatnonzero(ma.getmaskarray(col)==0)
            okData = ma.filled(ma.take(col, okIdx, axis=0), 0.0)
            # Compute upper and lower bounds
            if method.lower() == 'iqr':
                bounds = prctile(okData, [25.0,75.0])
                iqr = bounds[1] - bounds[0]
                lo = bounds[0] - (nDev[0] * iqr)
                up = bounds[1] + (nDev[1] * iqr)
                lower.append(lo)
                upper.append(up)
            elif method.lower() == 'idr':
                bounds = prctile(okData, [10.0,90.0])
                idr = bounds[1] - bounds[0]
                lo = bounds[0] - (nDev[0] * idr)
                up = bounds[1] + (nDev[1] * idr)
                lower.append(lo)
                upper.append(up)
            else:
                if method.lower() == 'huber':
                    huberScale = sm.robust.scale.Huber(c=k,norm=sm.robust.norms.HuberT(t=k))
                    try:
                        (loc, scale) = huberScale(okData)
                    except:
                        loc = ma.median(ma.asanyarray(okData), axis=None)
                        scale = sm.robust.scale.mad(okData)
                elif method.lower() == 'mad':
                    loc = ma.median(ma.asanyarray(okData), axis=None)
                    scale = sm.robust.scale.mad(okData)
                ratio = ma.sqrt(ma.inner(okData-loc, okData-loc) / (len(okData) - 1.0)) / scale
                medRatio[j] = ratio
                locList.append(loc)
                scaList.append(scale)
     
    outlierArray = Matrices.allMasked(data.shape)
    lowerBounds = Matrices.allMasked((data.shape[1],3))
    upperBounds = Matrices.allMasked((data.shape[1],3))

    if len(actualIdx) < 1:
        return outlierArray, lowerBounds, upperBounds

    if (method.lower() == 'huber') or (method.lower() == 'mad'):
        locList = screen_data(numpy.array(locList, float))
        scaList = screen_data(numpy.array(scaList, float))
        lower = locList - (nDev[0] * scaList)
        upper = locList + (nDev[1] * scaList)
        lower1 = locList - ((nDev[0]+1.0) * scaList)
        upper1 = locList + ((nDev[1]+1.0) * scaList)
        lower2 = locList - ((2.0*nDev[0]) * scaList)
        upper2 = locList + ((2.0*nDev[1]) * scaList)
    else:
        lower = numpy.array(lower, float)
        upper = numpy.array(upper, float)
        lower1 = lower
        lower2 = lower
        upper1 = upper
        upper2 = upper
     
    for (i,j) in enumerate(actualIdx):
        col = data[:,j]
        downWeightCol = Matrices.allMasked(len(col))
        # Determine those data points outside the bounds
        lowerIdx = ma.where(col < lower[i])[0]
        upperIdx = ma.where(col > upper[i])[0]
        ma.put(downWeightCol, lowerIdx, 1.0)
        ma.put(downWeightCol, upperIdx, 1.0)
        outlierArray[:,j] = downWeightCol
        lowerBounds[j,0] = lower[i]
        upperBounds[j,0] = upper[i]
        lowerBounds[j,1] = lower1[i]
        upperBounds[j,1] = upper1[i]
        lowerBounds[j,2] = lower2[i]
        upperBounds[j,2] = upper2[i]

    return outlierArray, lowerBounds, upperBounds

def computeExponentialWeights(halfLife, length, equalWeightFlag=False, normalize=True):
    """Return an array of exponential weights given a half-life of
    the given value, halfLife, and that sum to 1.
    The most recent observation is assumed to be first and given
    the greatest weight.
    If equalWeightFlag is enabled, returns a vector of equal weights.
    """
    if equalWeightFlag:
        w = numpy.ones(length, float)
    else:
        w = 2.**(numpy.arange(0,-length,-1)/float(halfLife))
    if normalize:
        w /= sum(w)
    return w

def computeTriangleWeights(peak, length, normalise=False, chrono=True):
    """Computes a piecewise linear function rising from near zero to a peak 
    at the given index (first argument), then dropping to near zero
    over time. Weights are in chronological order (i.e. oldest
    to most recent) unless specified otherwise
    """
    if peak == 0:
        weights = list(range(length))
    else:
        peak = min(peak, length)
        # Upweighting from most recent value
        wt1 = [(idx+1) / float(peak) for idx in range(peak)]
        wt1.reverse()
        # Upweighting from oldest value
        wt2 = [(idx+1) / float(length-peak+1) for idx in range(length-peak)]
        weights = wt2 + wt1
    if normalise:
        weights = weights / ma.sum(weights, axis=None)
    if not chrono:
        weights.reverse()
    return weights

def computePyramidWeights(inPeak, outPeak, length, normalise=False, chrono=True):
    """Computes a piecewise linear function rising from zero to a peak
    at the given index (first argument), remainding constant for a period, 
    then dropping to zero. Weights are in chronological order (i.e. oldest
    to most recent) unless specified otherwise
    """
    inPeak = min(inPeak, length)
    outPeak = min(outPeak, length-inPeak)

    # Upweighting from most recent value
    if inPeak > 0:
        wt1 = [(idx+1) / float(inPeak) for idx in range(inPeak)]
        wt1.reverse()
    else:
        wt1 = []

    # Downweighting to oldest value
    wt2 = [min(1.0, (idx+1) / float(outPeak+1)) for idx in range(length-inPeak)]
    weights = wt2 + wt1

    if normalise:
        weights = weights / ma.sum(weights, axis=None)
    if not chrono:
        weights.reverse()
    return weights

def zeroPad(val, length):
    if val == '' or val is None:
        return val
    if len(val) >= length:
        return val[:length]
    zeros = length*'0'
    return zeros[:length - len(val)] + val

def createIDMapping(subIssues, date, modelDB, marketDB):
    """Creates mappings from sedol,cusip etc. to subissue
    in one convenient place
    """
    IDMaps = Struct()

    # Load in the various subissue to ID maps and invert them
    cusipMap = reverseMap(modelDB.getIssueCUSIPs(date, subIssues, marketDB))
    tickerMap = reverseMap(modelDB.getIssueTickers(date, subIssues, marketDB))
    
    # Create two sedol maps - 6 and 7 digit
    tmpMap = reverseMap(modelDB.getIssueSEDOLs(date, subIssues, marketDB))
    sedolMap = dict()
    sedol6Map = dict()
    for (sedol, sid) in tmpMap.items():
        sedol = zeroPad(sedol, 7)
        sedol6 = sedol[:-1]
        sedolMap[sedol] = sid
        sedol6Map[sedol6] = sid

    # Add to mapping struct
    IDMaps.cusipMap = cusipMap
    IDMaps.tickerMap = tickerMap
    IDMaps.sedolMap = sedolMap
    IDMaps.sedol6Map = sedol6Map
    return IDMaps

def run_market_model(returns, rmg, modelDB, marketDB, periodsBack, 
                     sw=False, clip=True, indexSelector=None,
                     forceRun=False, debugOutput=False):
    """Computes parameters for the Market Model using the given
    returns and the given risk model group's market portfolio.  
    If sw=True, betas are adjusted using the Scholes-Williams (1977) 
    adjustment with a lead/lag of one.  
    Returns (beta, alpha, sigma, residuals) for each asset in 
    returns.assets.  residuals is an asset-by-time array.
    Sigma is the estimated variance of the residuals from the
    regression and alpha is the intercept term.
    NOTE: returns.data should not contain masked entries.
    """
    from riskmodels import MarketIndex
    logging.debug('run_market_model: begin')
    n = len(returns.assets)
    failedReturn = (numpy.ones(n), numpy.zeros(n), numpy.zeros(n), numpy.zeros(n))
    if n==1:
        return failedReturn

    # Remove days where many returns are missing
    io = numpy.sum(ma.getmaskarray(returns.data), axis=0)
    goodDatesIdx = numpy.flatnonzero(io < 0.7 * len(returns.assets))
    goodDates = numpy.take(returns.dates, goodDatesIdx, axis=0)
    # Restrict history
    periodsBack = min(periodsBack, len(goodDates))
    goodDates = goodDates[-periodsBack:]
    assetReturns = ma.filled(ma.take(returns.data, 
                                     goodDatesIdx, axis=1), 0.0)
    assetReturns = assetReturns[:,-periodsBack:]

    # Load the given risk model group's market returns
    logging.info('Computing market model for %s (%d assets, %d periods)',
            rmg.description, n, len(goodDates))
    if indexSelector is None:
        indexSelector = MarketIndex.MarketIndexSelector(modelDB, marketDB)
    assetIdxMap = dict([(j,i) for (i,j) in enumerate(returns.assets)])
    marketPortfolio = indexSelector.getMarketIndex(
            rmg.mnemonic, goodDates[-1], modelDB, returns.assets)
    if len(marketPortfolio.data) == 0 and forceRun:
        logging.warning('No market index data available')
        return failedReturn
    ids, weights = zip(*marketPortfolio.data)
    ret = ma.take(assetReturns, [assetIdxMap[sid] for sid in ids], axis=0)
    market_index = ma.dot(numpy.array(weights), ret)
    
    t = assetReturns.shape[1]
    assert(market_index.shape[0] == t)
    assert(numpy.sum(ma.getmaskarray(ma.masked_where(
                            market_index==0.0, market_index))) < t)

    # Debugging info
    if debugOutput:
        dateStr = [str(d) for d in goodDates]
        idList = [s.getSubIDString() for s in returns.assets]
        writeToCSV(market_index[numpy.newaxis,:], 'tmp/mktret-%s-%s.csv' % \
                (rmg.mnemonic, dateStr[-1]), columnNames=dateStr, rowNames=[rmg.mnemonic])
        writeToCSV(assetReturns, 'tmp/assret-%s-%s.csv' % \
                (rmg.mnemonic, dateStr[-1]), columnNames=dateStr, rowNames=idList)

    # Compute alpha, beta, sigma
    x = numpy.transpose(ma.array(
                [ma.ones(t, float), market_index]))
    y = numpy.transpose(assetReturns)
    (b0, resid) = ordinaryLeastSquares(y, x)
    beta = b0[1,:]
    alpha = b0[0,:]
    sigma = Matrices.allMasked(assetReturns.shape[0])
    for j in range(len(sigma)):
        sigma[j] = ma.inner(resid[:,j],resid[:,j]) / (t - 1.0)

    # Apply Scholes-Williams adjustment if required
    if sw:
        # Market lagging asset returns...
        x_lag = x[:-1,:]
        y_lag = y[1:,:]
        (b_lag, e_lag) = ordinaryLeastSquares(y_lag, x_lag)
        # Market leading asset returns...
        x_lead = x[1:,:]
        y_lead = y[:-1,:]
        (b_lead, e_lead) = ordinaryLeastSquares(y_lead, x_lead)
        
        # Put it all together...
        xCombined = ma.transpose(ma.array([x_lag[:,1], x_lead[:,1]]))
        corr = numpy.corrcoef(xCombined, rowvar=False)[0,1]
        k = 1.0 + 2.0 * corr
        logging.info('KAPPA REPORT: %s', k)
        k = abs(k)
        if k != 0.0:
            beta = (b_lag[1,:] + b0[1,:] + b_lead[1,:]) / k

            # New alphas
            meanReturns = ma.sum(y[1:-1,:], axis=0) / (t-2.0)
            meanMktReturn = ma.sum(x[1:-1,1], axis=0) / (t-2.0)
            alpha = meanReturns - beta * meanMktReturn

            # Update residuals
            bb = numpy.zeros(b0.shape)
            bb[0,:] = alpha
            bb[1,:] = beta
            resid = assetReturns - numpy.transpose(numpy.dot(x, bb))

    # Truncate betas is so desired
    if clip:
        beta = numpy.clip(beta, -0.5, 3.5)

    logging.debug('run_market_model: end')
    return (beta, alpha, sigma, resid)

def run_market_model2(rmg, returns, dateList, modelDB, marketDB,
                      indexSelector, scholes_williams_adj=False,
                      clip=True, period_returns=False, fillWithMarket=False,
                      forceRun=False, debugOutput=False):
    """Computes parameters for the Market Model using the given
    returns, the given RiskModelGroup's market portfolio, assets
    in returns.assets, and the time range given by dateList.
    The dates in the TimeSeriesMatrix returns do not have to be
    perfectly aligned with dateList, as long as every date in
    dateList is covered.
    If non-daily returns are used, set period_returns=True.
    The Scholes-Williams (1977) adjustment with a lead/lag of one
    will be applied if scholes_williams_adj=True.
    Returns (beta, alpha, sigma, residuals) for all assets.
    residuals is an asset-by-time array, sigma is the estimated
    variance of the residuals, and alpha is the intercept term.
    """
    logging.debug('run_market_model: begin')
    n = len(returns.assets)
    failedReturn = (numpy.ones(n), numpy.zeros(n), numpy.zeros(n), numpy.zeros(n))
    if n < 1:
        logging.warning('Too few assets (%d) in RiskModelGroup %s', n, rmg.description)
        return failedReturn

    # Restrict history to specified dates
    threshold = 0.7
    datesIdxMap = dict([(d,i) for (i,d) in enumerate(returns.dates)])
    date_indices = [datesIdxMap[d] for d in dateList]
    assetReturns = ma.take(returns.data, date_indices, axis=1)
    io = numpy.sum(ma.getmaskarray(assetReturns), axis=0)
    if fillWithMarket:
        goodDatesIdx = list(range(assetReturns.shape[1]))
    else:
        goodDatesIdx = numpy.flatnonzero(io < threshold * n)
    if len(goodDatesIdx) < 2:
        logging.warning('Too few dates (%d) with at least %d%% of stocks trading',
                        len(goodDatesIdx), threshold * 100.0)
        for sid in returns.assets:
            logging.info('Missing at least %d%% of returns: %s',
                    threshold * 100.0, sid.getSubIDString())
        return failedReturn
    else:
        assetReturns = ma.take(assetReturns, goodDatesIdx, axis=1)

    # Load the given RiskModelGroup's daily market returns
    logging.info('Market model for %s (RiskModelGroup %d, %s) (n: %d, t: %d)',
                    rmg.description, rmg.rmg_id, rmg.mnemonic, n, len(goodDatesIdx))

    if period_returns:
        if rmg.mnemonic == 'XC':
            rmgList = modelDB.getAllRiskModelGroups()
            realRMG = [r for r in rmgList if r.mnemonic == 'CN'][0]
        else:
            realRMG = rmg
        fullDateList = modelDB.getDateRange([realRMG], min(dateList), max(dateList))
    else:
        fullDateList = dateList
    marketReturns = modelDB.loadRMGMarketReturnHistory(
                        fullDateList, [rmg], useAMPs=False).data[0,:].filled(0.0)

    # Aggregate daily returns up to period returns if required
    if period_returns:
        cumMarketReturns = numpy.cumproduct(marketReturns + 1.0, axis=0)
        datesIdxMap = dict([(d,i) for (i,d) in enumerate(fullDateList)])
        cumMarketReturns = numpy.take(cumMarketReturns,
                            [datesIdxMap[d] for d in dateList], axis=0)
        market_index = cumMarketReturns[1:] / cumMarketReturns[:-1] - 1.0
        assetReturns = assetReturns[:,1:]
        goodDatesIdx = [i-1 for i in goodDatesIdx[1:]]
    else:
        market_index = marketReturns
    market_index = numpy.take(market_index, goodDatesIdx, axis=0)

    # Fix for assets with deficient histories
    if fillWithMarket:
        # Ensure that each asset has a complete history
        # by filling missing values with the market return
        maskedReturns = numpy.array(\
                ma.getmaskarray(assetReturns), dtype='float')
        for ii in range(len(market_index)):
            maskedReturns[:,ii] *= market_index[ii]
        assetReturns = ma.filled(assetReturns, 0.0)
        assetReturns += maskedReturns
    else:
        assetReturns = ma.filled(assetReturns, 0.0)
         
    if debugOutput:
        dateList = numpy.take(dateList, goodDatesIdx, axis=0)
        dateList = [str(d) for d in dateList]
        writeToCSV(market_index[numpy.newaxis,:], 'tmp/mktret-%s-%s.csv' % \
                (rmg.mnemonic, dateList[-1]), columnNames=dateList, rowNames=[rmg.mnemonic])
         
    t = assetReturns.shape[1]
    assert(len(market_index) == t)
    if forceRun:
        if numpy.sum(ma.getmaskarray(ma.masked_where(
                    market_index==0.0, market_index))) == t:
            logging.warning('All market returns missing or zero')
            return failedReturn
    else:
        assert(numpy.sum(ma.getmaskarray(ma.masked_where(
                    market_index==0.0, market_index))) < t)

    # Compute alpha, beta, sigma
    x = numpy.transpose(ma.array(
                [ma.ones(t, float), market_index]))
    y = numpy.transpose(assetReturns)
    (b0, resid) = ordinaryLeastSquares(y, x)
    beta = b0[1,:]
    alpha = b0[0,:]
    sigma = Matrices.allMasked(assetReturns.shape[0])
    for j in range(len(sigma)):
        sigma[j] = ma.inner(resid[:,j],resid[:,j]) / (t - 1.0)

    # Apply Scholes-Williams adjustment if required
    if scholes_williams_adj:
        # Market lagging asset returns...
        x_lag = x[:-1,:]
        y_lag = y[1:,:]
        (b_lag, e_lag) = ordinaryLeastSquares(y_lag, x_lag)
        # Market leading asset returns...
        x_lead = x[1:,:]
        y_lead = y[:-1,:]
        (b_lead, e_lead) = ordinaryLeastSquares(y_lead, x_lead)

        # Put it all together...
        xCombined = ma.transpose(ma.array([x_lag[:,1], x_lead[:,1]]))
        corr = numpy.corrcoef(xCombined, rowvar=False)[0,1]
        k = 1.0 + 2.0 * corr
        logging.info('KAPPA REPORT: %s', k)
        k = abs(k)
        if k != 0.0:
            beta = (b_lag[1,:] + b0[1,:] + b_lead[1,:]) / k

            # New alphas
            meanReturns = ma.sum(y[1:-1,:], axis=0) / (t-2.0)
            meanMktReturn = ma.sum(x[1:-1,1], axis=0) / (t-2.0)
            alpha = meanReturns - beta * meanMktReturn

            # Update residuals
            bb = numpy.zeros(b0.shape)
            bb[0,:] = alpha
            bb[1,:] = beta
            resid = assetReturns - numpy.transpose(numpy.dot(x, bb))

    # Truncate betas if so desired
    if clip:
        beta = numpy.clip(beta, -0.5, 3.5)

    logging.debug('run_market_model: end')
    return (beta, alpha, sigma, resid)

def run_market_model_v3(rmg, returns, modelDB, marketDB, params,
                      clip=False, fillWithMarket=False, outputDateList=None,
                      marketReturns=None, debugOutput=False, forceRun=False,
                      clippedReturns=None, historyLength=None,
                      marketRegion=None):
    """More general version of run_market_model2.
    Computes parameters for the Market Model using the given
    returns, the given RiskModelGroup's market portfolio, assets 
    in returns.assets, and the time range given by dateList.
    The dates in the TimeSeriesMatrix returns do not have to be
    perfectly aligned with dateList, as long as every date in 
    dateList is covered.
    The Scholes-Williams (1977) adjustment with a lead/lag of one
    will be applied if scholes_williams_adj=True.
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
    logging.debug('run_market_model: begin')
     
    if type(rmg) is not list:
        rmg = [rmg]
    if len(rmg) > 1:
        mnem = marketRegion.name
    else:
        mnem = rmg[0].mnemonic

    # Get model parameters
    if hasattr(params, 'swAdj'):
        scholes_williams_adj = params.swAdj
    else:
        scholes_williams_adj = False
    if hasattr(params, 'lag'):
        lag = params.lag
    else:
        lag = False
    if hasattr(params, 'robust'):
        robustBeta = params.robust
    else:
        robustBeta = False
    if hasattr(params, 'k_val'):
        kappa = params.k_val
    else:
        kappa = 1.345

    # Set up dimensions and dates
    assets = list(returns.assets)
    n = len(assets)
    if historyLength is None:
        initialDateList = list(returns.dates)
    else:
        initialDateList = returns.dates[-historyLength:]
     
    # Create default return value
    retval = Struct()
    retval.beta = numpy.zeros((n), float)
    retval.alpha = numpy.zeros((n), float)
    retval.sigma = numpy.zeros((n), float)
    retval.resid = numpy.zeros((n), float)
    retval.tstat = numpy.zeros((n), float)
    retval.pvals = numpy.ones((n), float)

    # Debugging info
    if debugOutput:
        dateStr = [str(d) for d in initialDateList]
        idList = [s.getSubIDString() for s in assets]
        if historyLength is not None:
            initialReturns = returns.data[:, -historyLength:]
        else:
            initialReturns = returns.data
        writeToCSV(initialReturns, 'tmp/initial-assret-%s-%s.csv' % \
                (mnem, dateStr[-1]), columnNames=dateStr, rowNames=idList)

    if robustBeta or (clippedReturns is None):
        assetReturns = ma.array(returns.data, copy=True)
    else:
        assetReturns = ma.array(clippedReturns, copy=True)
    if lag != False:
        retval.beta = numpy.zeros((n, len(lag)), float)
        retval.tstat = numpy.zeros((n, len(lag)), float)
        retval.pvals = numpy.ones((n, len(lag)), float)

    if n < 1:
        logging.warning('Too few assets (%d) in RiskModelGroup', n)
        return retval
    if n < 2 and len(assetReturns.shape) < 2:
        assetReturns = assetReturns[numpy.newaxis, :]
    if len(initialDateList) != assetReturns.shape[1]:
        logging.warning('Number of dates (%d) not equal to length of returns history (%d)',
                len(initialDateList), assetReturns.shape[1])
    if historyLength is not None:
        assetReturns = assetReturns[:, -historyLength:]
    assert(len(initialDateList) == assetReturns.shape[1])

    # Load the given RiskModelGroup's daily market returns if not already given
    if marketReturns == None:
        mktDateList = modelDB.getDateRange(None, min(initialDateList), max(initialDateList))
        if marketRegion is not None:
            marketReturns = ma.filled(modelDB.loadRegionReturnHistory(
                mktDateList, [marketRegion]).data[0,:], 0.0)
        else:
            marketReturns = ma.filled(modelDB.loadRMGMarketReturnHistory(
                                mktDateList, rmg, robust=True, useAMPs=False).data[0,:], 0.0)
    else:
        mktDateList = marketReturns.dates
        marketReturns = ma.filled(marketReturns.data, 0.0)
        if len(marketReturns.shape) > 1:
            marketReturns = marketReturns[0,:]
        if historyLength is not None:
            marketReturns = marketReturns[-historyLength:]
         
    # Compound returns for invalid dates into the following trading-day
    commonDates = sorted(set(mktDateList).intersection(set(initialDateList)))
    if initialDateList != commonDates:
        assetReturns = compute_compound_returns_v3(
                assetReturns, initialDateList, commonDates, keepFirst=True)[0]
    if mktDateList != commonDates:
        marketReturns = compute_compound_returns_v3(
                marketReturns, mktDateList, commonDates, keepFirst=True)[0]
    if len(rmg) > 1:
        logging.debug('Market model for %s (n: %d, t: %d)',
                marketRegion.name, n, len(commonDates))
    else:
        logging.debug('Market model for %s (RiskModelGroup %d, %s) (n: %d, t: %d)',
                rmg[0].description, rmg[0].rmg_id, rmg[0].mnemonic, n, len(commonDates))
         
    # Compound asset and market returns to match frequency of outputDateList
    if outputDateList is not None:
        outputDateList.sort()
        if commonDates != outputDateList:
            marketReturns = compute_compound_returns_v3(
                    marketReturns, commonDates, outputDateList)[0]
            assetReturns = compute_compound_returns_v3(
                    assetReturns, commonDates, outputDateList)[0]
    else:
        outputDateList = commonDates
     
    # Fix for assets with deficient histories
    if fillWithMarket:
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
    if debugOutput:
        dateStr = [str(d) for d in outputDateList]
        idList = [s.getSubIDString() for s in assets]
        writeToCSV(marketReturns[numpy.newaxis,:], 'tmp/mktret-%s-%s.csv' % \
                (mnem, dateStr[-1]), columnNames=dateStr, rowNames=[mnem])
        writeToCSV(assetReturns, 'tmp/assret-%s-%s.csv' % \
                (mnem, dateStr[-1]), columnNames=dateStr, rowNames=idList)

    t = assetReturns.shape[1]
    assert(len(marketReturns) == t)
    if forceRun:
        if numpy.sum(ma.getmaskarray(ma.masked_where(
            marketReturns==0.0, marketReturns))) == t:
            logging.warning('All market returns missing or zero')
            return retval
    else:
        assert(numpy.sum(ma.getmaskarray(ma.masked_where(
            marketReturns==0.0, marketReturns))) < t)

    # If we're doing a lagged regression:
    if lag != False:
        # Construct a lags-by-t RHS matrix
        marketReturns_matrix = numpy.zeros((len(lag)+1,t), float)
        marketReturns_matrix[0,:] = 1.0
        for (i,l) in enumerate(lag):
            marketReturns_matrix[i+1,l:] = marketReturns[:-l]
        assetReturns = assetReturns[:,l:]
        marketReturns_matrix = marketReturns_matrix[:,l:]
        t -= l
        x = numpy.transpose(marketReturns_matrix)
    else:
        x = numpy.transpose(ma.array(
            [ma.ones(t, float), marketReturns]))
         
    # Compute alpha, beta, sigma
    y = numpy.transpose(assetReturns)
    res = robustLinearSolver(y, x, robust=robustBeta, k=kappa)
    resid = ma.transpose(res.error)
    b0 = res.params
    tval = res.tstat
    pval = res.pvals
    beta = b0[1:,:]
    alpha = b0[0,:]
    t_beta = tval[1:,:]
    t_alph = tval[0,:]
    p_beta = pval[1:,:]

    # Apply Scholes-Williams adjustment if required
    if scholes_williams_adj:
        # Market lagging asset returns...
        x_lag = x[:-1,:]
        y_lag = y[1:,:]
        b_lag = robustLinearSolver(y_lag, x_lag, robust=robustBeta, k=kappa).params
        # Market leading asset returns...
        x_lead = x[1:,:]
        y_lead = y[:-1,:]
        b_lead = robustLinearSolver(y_lead, x_lead, robust=robustBeta, k=kappa).params
        
        # Put it all together...
        xCombined = ma.transpose(ma.array([x_lag[:,1], x_lead[:,1]]))
        corr = numpy.corrcoef(xCombined, rowvar=False)[0,1]
        k = 1.0 + 2.0 * corr
        logging.info('KAPPA REPORT: %s', k)
        k = abs(k)
        if k != 0.0:
            beta = (b_lag[1:,:] + b0[1:,:] + b_lead[1:,:]) / k

            # New alphas
            meanReturns = ma.sum(y[1:-1,:], axis=0) / (t-2.0)
            meanMktReturn = ma.sum(x[1:-1,1], axis=0) / (t-2.0)
            alpha = (meanReturns - beta * meanMktReturn)[0,:]

            # Update residuals
            bb = numpy.zeros(b0.shape)
            bb[0,:] = alpha
            bb[1:,:] = beta
            resid = assetReturns - numpy.transpose(numpy.dot(x, bb))
             
    sigma = Matrices.allMasked(assetReturns.shape[0])
    for j in range(len(sigma)):
        sigma[j] = ma.inner(resid[j,:],resid[j,:]) / (t - 1.0)

    beta = screen_data(beta)
    # Truncate betas if so desired
    if clip:
        beta = numpy.clip(beta, -0.5, 3.5)
    if not lag:
        beta = beta[0,:]
        t_beta = t_beta[0,:]
        p_beta = p_beta[0,:]

    logging.debug('run_market_model: end')
    retval.beta = beta
    retval.alpha = alpha
    retval.sigma = sigma
    retval.resid = resid
    retval.tstat = t_beta
    retval.pvals = p_beta
    return retval

def run_market_model_ff(rmg, returns, modelDB, marketDB, params,
                      clip=False, fillWithMarket=False, outputDateList=None,
                      marketReturns=None, debugOutput=False, forceRun=False,
                      clippedReturns=None):
    """Specific version of run_market_model_v3. Uses fama-french market
    and size returns.
    Computes parameters for the Market Model using the given
    returns, the given RiskModelGroup's market portfolio, assets 
    in returns.assets, and the time range given by dateList.
    The dates in the TimeSeriesMatrix returns do not have to be
    perfectly aligned with dateList, as long as every date in 
    dateList is covered.
    The Scholes-Williams (1977) adjustment with a lead/lag of one
    will be applied if scholes_williams_adj=True.
    Returns (beta, alpha, sigma, residuals) for all assets.
    residuals is an asset-by-time array, sigma is the estimated 
    variance of the residuals, and alpha is the intercept term.
    If n by t array marketReturns is specified, this will be used for
    beta computation, otherwise the routine will load the appropriate
    set.
    """
    logging.debug('run_market_model_ff: begin')
     
    # Get model parameters
    if hasattr(params, 'swAdj'):
        scholes_williams_adj = params.swAdj
    else:
        scholes_williams_adj = False
    if hasattr(params, 'robust'):
        robustBeta = params.robust
    else:
        robustBeta = False
    if hasattr(params, 'k_val'):
        k = params.k_val
    else:
        k = 1.345

    # Set up dimensions and dates
    assets = list(returns.assets)
    n = len(assets)
    initialDateList = list(returns.dates)
     
    # Create default return value
    retval = Struct()
    retval.beta = numpy.zeros((n), float)
    retval.alpha = numpy.zeros((n), float)
    retval.sigma = numpy.zeros((n), float)
    retval.resid = numpy.zeros((n), float)
    retval.tstat = numpy.zeros((n), float)
    retval.pvals = numpy.ones((n), float)

    if robustBeta or (clippedReturns is None):
        assetReturns = ma.array(returns.data, copy=True)
    else:
        assetReturns = ma.array(clippedReturns, copy=True)

    if n < 1:
        logging.warning('Too few assets in RiskModelGroup %s', n, rmg.description)
        return retval
    if n < 2 and len(assetReturns.shape) < 2:
        assetReturns = assetReturns[numpy.newaxis, :]

    #load ff date from file
    inFile = open('FFFactors.csv', 'r')
    ffData = inFile.readlines()
    inFile.close()
    
    header = ffData[0]
    ffDat = ffData[1:]
    
    ffDatesTmp = [i.split(',')[0] for i in ffDat]
    ffDates = [datetime.date(int(i[:4]),int(i[4:6]),int(i[6:8])) for i in ffDatesTmp]
    
    datesMktList = [(datetime.date(int(i.split(',')[0][:4]),
                            int(i.split(',')[0][4:6]),
                            int(i.split(',')[0][6:8]))
                     , float(i.split(',')[1])*.01) for i in ffDat]
    ffMktDict = dict(datesMktList)
    
    datesSizeList = [(datetime.date(int(i.split(',')[0][:4]),
                            int(i.split(',')[0][4:6]),
                            int(i.split(',')[0][6:8]))
                      , float(i.split(',')[2])*.01) for i in ffDat]
    ffSizeDict = dict(datesSizeList)

    endDate = max(initialDateList)
    endDateInd = ffDates.index(endDate)
    mktDateList = ffDates[(endDateInd-249):(endDateInd+1)]
    assert len(mktDateList)==250
    
    marketReturns = ma.array([ffMktDict[i] for i in mktDateList])
         
    # Compound returns for invalid dates into the following trading-day
    assert mktDateList == initialDateList
    commonDates = sorted(set(mktDateList).intersection(set(initialDateList)))
    if initialDateList != commonDates:
        assetReturns = compute_compound_returns_v3(
                assetReturns, initialDateList, commonDates, keepFirst=True)[0]
    if mktDateList != commonDates:
        marketReturns = compute_compound_returns_v3(
                marketReturns, mktDateList, commonDates, keepFirst=True)[0]
    logging.info('Market model for %s (RiskModelGroup %d, %s) (n: %d, t: %d)',
            rmg.description, rmg.rmg_id, rmg.mnemonic, n, len(commonDates))
         
    # Compound asset and market returns to match frequency of outputDateList
    if outputDateList is not None:
        outputDateList.sort()
        if commonDates != outputDateList:
            marketReturns = compute_compound_returns_v3(
                    marketReturns, commonDates, outputDateList)[0]
            assetReturns = compute_compound_returns_v3(
                    returns, commonDates, outputDateList)[0]
    else:
        outputDateList = commonDates
     
    # Fix for assets with deficient histories
    if fillWithMarket:
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
    if debugOutput:
        dateStr = [str(d) for d in outputDateList]
        idList = [s.getSubIDString() for s in assets]
        writeToCSV(marketReturns[numpy.newaxis,:], 'tmp/mktret-%s-%s.csv' % \
                (rmg.mnemonic, dateStr[-1]), columnNames=dateStr, rowNames=[rmg.mnemonic])
        writeToCSV(assetReturns, 'tmp/assret-%s-%s.csv' % \
                (rmg.mnemonic, dateStr[-1]), columnNames=dateStr, rowNames=idList)

    t = assetReturns.shape[1]
    assert(len(marketReturns) == t)
    if forceRun:
        if numpy.sum(ma.getmaskarray(ma.masked_where(
            marketReturns==0.0, marketReturns))) == t:
            logging.warning('All market returns missing or zero')
            return retval
    else:
        assert(numpy.sum(ma.getmaskarray(ma.masked_where(
            marketReturns==0.0, marketReturns))) < t)

    sizeReturns = [ffSizeDict[i] for i in outputDateList]
    x = numpy.transpose(ma.array(
        [ma.ones(t, float), sizeReturns, marketReturns]))
         
    # Compute alpha, beta, sigma
    y = numpy.transpose(assetReturns)
    res = robustLinearSolver(y, x, robust=robustBeta, k=k)
    resid = ma.transpose(res.error)
    b0 = res.params
    tval = res.tstat
    pval = res.pvals
    beta = b0[1:,:]
    alpha = b0[0,:]
    t_beta = tval[1:,:]
    t_alph = tval[0,:]
    p_beta = pval[1:,:]

    # Apply Scholes-Williams adjustment if required
    if scholes_williams_adj:
        # Market lagging asset returns...
        x_lag = x[:-1,:]
        y_lag = y[1:,:]
        b_lag = robustLinearSolver(y_lag, x_lag, robust=robustBeta, k=k).params
        # Market leading asset returns...
        x_lead = x[1:,:]
        y_lead = y[:-1,:]
        b_lead = robustLinearSolver(y_lead, x_lead, robust=robustBeta, k=k).params
        
        # Put it all together...
        xCombined = ma.transpose(ma.array([x_lag[:,1], x_lead[:,1]]))
        corr = numpy.corrcoef(xCombined, rowvar=False)[0,1]
        k = 1.0 + 2.0 * corr
        logging.info('KAPPA REPORT: %s', k)
        k = abs(k)
        if k != 0.0:
            beta = (b_lag[1:,:] + b0[1:,:] + b_lead[1:,:]) / k

            # New alphas
            meanReturns = ma.sum(y[1:-1,:], axis=0) / (t-2.0)
            meanMktReturn = ma.sum(x[1:-1,1], axis=0) / (t-2.0)
            alpha = (meanReturns - beta * meanMktReturn)[0,:]

            # Update residuals
            bb = numpy.zeros(b0.shape)
            bb[0,:] = alpha
            bb[1:,:] = beta
            resid = assetReturns - numpy.transpose(numpy.dot(x, bb))
             
    sigma = Matrices.allMasked(assetReturns.shape[0])
    for j in range(len(sigma)):
        sigma[j] = ma.inner(resid[j,:],resid[j,:]) / (t - 1.0)

    beta = screen_data(beta)
    # Truncate betas if so desired
    if clip:
        beta = numpy.clip(beta, -0.5, 3.5)
    
    beta = beta[0,:]
    t_beta = t_beta[0,:]
    p_beta = p_beta[0,:]

    logging.debug('run_market_model_ff: end')
    retval.beta = beta
    retval.alpha = alpha
    retval.sigma = sigma
    retval.resid = resid
    retval.tstat = t_beta
    retval.pvals = p_beta
    return retval

def mlab_std(x, axis=0):
    """Legacy method to replicates MLab.std() functionality 
    using numpy methods.
    numpy.std() divides by N rather than N-1 to give biased
    estimates (lower MSE).
    """
    obs = x.shape[axis]
    var = numpy.var(x, axis=axis) * obs / (obs-1.0)
    std = var**0.5
    return std

def symmetric_pencil(a, b):
    """Computes simultaneous eigendecomposition of a pair
    of matrics. Matrix A is assumed symmetric
    whilst matrix B must be positive definite also.
    Returns transform matrix X such that
    X'AX = D_A
    X'BX = I
    Algorithm is due to Stewart
    """
    # Compute symmetric eigendecomposition of B
    (d, y) = linalg.eigh(b)
    # Assemble transform matrix Z^-1
    d_inv = numpy.diag(1.0/d)
    z_inv = numpy.dot(y, numpy.sqrt(d_inv))
    # Form transformed A matrix
    a = numpy.dot(numpy.transpose(z_inv),numpy.dot(a,z_inv))
    # Compute symmetric eigendecomposition of new A
    (da, u) = linalg.eigh(a)
    # Form ultimate transform matrix X
    x = numpy.dot(z_inv, u)
    # Back out generalised eigenvalues
    eig_a = numpy.diag(numpy.dot(numpy.transpose(x),numpy.dot(a,x)))
    eig_b = numpy.diag(numpy.dot(numpy.transpose(x),numpy.dot(b,x)))
    return (eig_a, eig_b, x)

def is_binary_data(data):
    """Checks if an array (not matrix) of data is binary.  
    ie, contains only 2 unique values, besides masked entries.
    """
    assert(len(data.shape)==1)
    goodData = ma.take(data, numpy.flatnonzero(
                       ma.getmaskarray(data)==0), axis=0)
    freq = numpy.unique(numpy.array(goodData))
    if len(freq) < 3:
        return True
    else:
        return False
    
class BufferingSMTPHandler(logging.handlers.BufferingHandler):
     def __init__(self, mailhost, fromaddr, toaddrs, subject, capacity=100000):
         logging.handlers.BufferingHandler.__init__(self, capacity)
         self.mailhost = mailhost
         self.mailport = None
         self.fromaddr = fromaddr
         self.toaddrs = toaddrs
         
         # try to parse out the model name from the -m option and the model id using the -i option
         # from sys.argv.  If nothing exists, then just use the stock subject passed in
         s=''
         dt=''
         found=False
         el=None
         for el in sys.argv:
             if found:
                 s = "%s Model=%s" % (s,el)
                 found=False
                 continue
             
             if len(el) >= 2 and el[0:2] in ['-m', '-i']:
                 if el in ['-m', '-i']:
                     found=True
                     continue
                 if el[2]=='=':
                     s= "%s Model=%s" % (s,el[3:])
                 else:
                     s= "%s Model=%s" % (s,el[2:])
         # last element is the date
         if s=='':
             self.subject = subject
             
         else:
             # get rid of the caller's directory and the .py and just basename of the python file
             if el is not None:
                 dt='Date=%s'% (el)
             self.subject = "%s %s %s" % (os.path.basename(sys.argv[0]).replace('.py',''), s, dt)
         self.setFormatter(logging.Formatter("%(asctime)s %(levelname)-5s %(message)s"))
         
     def flush(self):
         if len(self.buffer) > 0:
             try:
                 import smtplib
                 port = self.mailport
                 if not port:
                     port = smtplib.SMTP_PORT
                 smtp = smtplib.SMTP(self.mailhost, port)
                 msg = "From: %s\r\nTo: %s\r\nSubject: %s\r\n\r\n" % (self.fromaddr, string.join(self.toaddrs, ","), self.subject)
                 for record in self.buffer:
                     s = self.format(record)
                     msg = msg + s + "\r\n"
                 smtp.sendmail(self.fromaddr, self.toaddrs, msg)
                 smtp.quit()
             except:
                 self.handleError(None)  # no record failed in particular
             self.buffer = []
 
logging.handlers.BufferingSMTPHandler = BufferingSMTPHandler

def generate_marketcap_buckets(mcaps, breakPoints, 
                               restrict=None, trimMegaCaps=True):
    """Sorts, then bucketizes assets based on the given market
    caps and breakPoints.  breakPoints contains the fraction of
    total cap to be represented in each bucket.  For example,
    (0.80, 0.15) will return 3 (not 2!) buckets -- the largest 
    assets which constitute 80% of total cap, then the next 15%,
    and the remaining 5%.
    Returns a list of lists, each containing the array
    positions corresponding to assets in that bucket.
    """
    logging.debug('generate_marketcap_buckets: begin')
    assert(sum(breakPoints) < 1.0 and len(breakPoints) > 0)
    if restrict is None:
        restrict = list(range(len(mcaps)))

    # Flatten the very top of the market to avoid tiny bands
    mcaps = ma.filled(mcaps, 0.0)
    rank = numpy.argsort(mcaps).tolist()
    if trimMegaCaps:
        numMegaCaps = min(100, int(round(len(mcaps)*0.025)))
        megaCapsIdx = rank[-numMegaCaps:]
        numpy.put(mcaps, megaCapsIdx, mcaps[rank[-numMegaCaps]])
    totalCap = numpy.sum(numpy.take(mcaps, restrict), axis=0)
    restrict = set(restrict)

    # Populate capitalization buckets
    buckets = list()
    for i in range(len(breakPoints)):
        capBucket = list()
        bucketTotalCap = 0.0
        ratio = 0.0
        while ratio < breakPoints[i] and len(rank) > 0:
            idx = rank.pop()
            capBucket.append(idx)
            if idx in restrict:
                bucketTotalCap += mcaps[idx]
            ratio = bucketTotalCap / totalCap 
        buckets.append(capBucket)
        logging.info('Bucket %d contains %d (%d) assets, %.2f%% of total cap',
            i+1, len(capBucket), len(restrict.intersection(capBucket)), ratio*100.0)
    buckets.append(rank)
    logging.info('Bucket %d contains %d assets, %.2f%% of total cap',
            len(buckets), len(rank), numpy.sum(numpy.take(mcaps, rank))/totalCap*100.0)

    logging.debug('generate_marketcap_buckets: end')
    return buckets
    
def compute_asset_value_proxies(values, buckets, restrict=None):
    """Given an asset-by-time array of data (eg. returns),
    missing (masked) values are filled-in using a proxy algorithm
    which, for each point in time, computes a proxy value
    based on the given asset buckets.  buckets should be a list
    of lists, with each nested list containing array positions
    of assets belonging to that bucket.
    Proxy values can be based on a subset of assets whose
    positions are specified by the optional restrict argument.
    """
    logging.debug('compute_asset_value_proxies: begin')
    if numpy.sum(ma.getmaskarray(values))==0:
        logging.info('No missing values! Skipping proxy computation')
        return values
    if restrict is None:
        restrict = list(range(values.shape[0]))
    restrict = set(restrict)
    # Make sure buckets cover all assets
#    assert(set([i for j in buckets for i in j]) \
#                == set(range(values.shape[0])))

    # Loop round dates and asset buckets
    logging.info('Using %d asset buckets, processing %d periods',
                    len(buckets), values.shape[1])
    n = 0
    for indices in buckets:
        if len(indices)==0:
            continue
        oldValues = ma.take(values, indices, axis=0)
        mask = ma.getmaskarray(oldValues)
        # If no missing values in this bucket, skip
        if numpy.sum(ma.getmaskarray(oldValues))==0:
            continue
        goodAssetsIdx = list(restrict.intersection(indices))
        # If insufficient assets to act as proxy, fill with zero
        if len(goodAssetsIdx)==0:
            values[indices,:] = values[indices,:].filled(0.0)
            continue
        goodValues = ma.take(values, goodAssetsIdx, axis=0)
        # For time periods with too many missing values, proxy with zero
        singularIdx = numpy.flatnonzero(numpy.sum(
                            ma.getmaskarray(goodValues)==0, axis=0) < 2)
        # Otherwise, use fill-in rule
        signs = numpy.sign(ma.median(goodValues, axis=0).filled(0.0))
        proxyValues = signs * numpy.mean(abs(goodValues), 
                                        axis=0).filled(0.0)
        if len(singularIdx) > 0:
            numpy.put(proxyValues, singularIdx, 0.0)
        newValues = ma.where(mask, proxyValues, oldValues)

        values[indices,:] = newValues
        n += numpy.sum(mask)

    shape = values.shape
    logging.info('Used proxy fill-in values for %d out of %d obs (%.2f %%)',
                n, shape[0]*shape[1],  100.0 * n / (shape[0]*shape[1]))

    # Make sure all missing values have been proxied
#    assert(numpy.sum(ma.getmaskarray(values), axis=None)==0)

    logging.debug('compute_asset_value_proxies: end')
    return values

def bucketedMAD(rmgList, date, returnsIn, data, modelDB,
        nDev=[3.0, 3.0], MADaxis=None, method='Huber', industryGroupFactor=False,
        gicsDate=datetime.date(2016,9,1)):
    """Given an array of asset returns, sorts them into sector/region
    buckets and MADs each separately
    """
    logging.debug('bucketedMAD: begin')
    returns = ma.array(returnsIn, copy=True)
    vector = False
    if len(returns.shape) < 2:
        returns = returns[:, numpy.newaxis]
        vector = True

    logging.info('Data bounds before 2D MADing: [%.3f, %.3f, %.3f]...' % \
            (ma.min(returns, axis=None), ma.average(returns, axis=None),
                ma.max(returns, axis=None)))
             
    # Get sector-level buckets
    industryClassification = Classification.GICSIndustries(gicsDate)
    if industryGroupFactor:
        parentName = 'Industry Groups'
        level = -1
    else:
        parentName = 'Sectors'
        level = -2
    parents = industryClassification.getClassificationParents(parentName, modelDB)
    factorList = [f.description for f in parents]

    # Bucket assets into regions
    regionAssetMap = dict()
    for r in rmgList:
        rmg_assets = data.rmgAssetMap[r.rmg_id]
        if r.region_id not in regionAssetMap:
            regionAssetMap[r.region_id] = list()
        regionAssetMap[r.region_id].extend(rmg_assets)

    # Go through assets by region
    for reg in regionAssetMap.keys():
        subSetIds = regionAssetMap[reg]
        logging.debug('MADing over %d assets for region %s', 
                len(subSetIds), reg)
        remainingIds = set(subSetIds)
        exposures = industryClassification.getExposures(
                date, subSetIds, factorList, modelDB, level=level)
        exposures = ma.masked_where(exposures==0.0, exposures)

        # Loop round sectors and pull out sector/region assets
        for isec in range(exposures.shape[0]):
            sectorAssetIdx = numpy.flatnonzero(ma.getmaskarray(exposures[isec,:])==0)
            if len(sectorAssetIdx) < 10:
                logging.warning('Too few assets (%d) for sector %s',
                        len(sectorAssetIdx), factorList[isec])
                sectorAssets = []
            else:
                sectorAssets = numpy.take(subSetIds, sectorAssetIdx, axis=0)
                logging.debug('MADing over %d assets for sector %s',
                        len(sectorAssets), factorList[isec])
                returnsIdx = [data.assetIdxMap[sid] for sid in sectorAssets]
                subSetReturns = ma.take(returns, returnsIdx, axis=0)
                clippedReturns = twodMAD(subSetReturns, nDev=nDev, axis=MADaxis,
                        method=method, suppressOutput=True)
                for (idim, idx) in enumerate(returnsIdx):
                    returns[idx,:] = clippedReturns[idim,:]
            remainingIds = remainingIds.difference(set(sectorAssets))

        # Deal with anything not mapped
        if len(remainingIds) > 0:
            logging.warning('%s assets not mapped to sector/IG', len(remainingIds))
            sectorAssets = list(remainingIds)
            returnsIdx = [data.assetIdxMap[sid] for sid in sectorAssets]
            subSetReturns = ma.take(returns, returnsIdx, axis=0)
            clippedReturns = twodMAD(subSetReturns, nDev=nDev, axis=MADaxis,
                    method=method, suppressOutput=True)
            for (idim, idx) in enumerate(returnsIdx):
                returns[idx,:] = clippedReturns[idim,:]

    logging.info('...data bounds after 2D MADing: [%.3f, %.3f, %.3f]' % \
            (ma.min(returns, axis=None), ma.average(returns, axis=None),
                ma.max(returns, axis=None)))
    if vector:
        returns = returns[:,0]
    logging.debug('bucketedMAD: end')
    return returns

def proxyMissingAssetReturnsLegacy(rmgList, date, returns, data, modelDB,
                            MADClipping=False, robust=False,
                            countryFactor=True, industryGroupFactor=True,
                            gicsDate=datetime.date(2014,3,1)):
    """Given a TimeSeriesMatrix of asset returns with masked
    values, fill in missing values using proxies.
    Assets are modelled using countries/regions, GICS Sectors/industry groups
    and log(cap) as factors.
    """
    logging.debug('proxyMissingAssetReturnsLegacy: begin')
    returnsData = ma.array(returns.data, copy=True)

    # Use Sector-level classification for exposures
    industryClassification = Classification.GICSIndustries(gicsDate)
    if industryGroupFactor:
        parentName = 'Industry Groups'
        level = -1
    else:
        parentName = 'Sectors'
        level = -2
    parents = industryClassification.getClassificationParents(parentName, modelDB)
    factorList = [f.description for f in parents]

    exposures = industryClassification.getExposures(
                date, data.universe, factorList, modelDB, level=level)
    exposures = numpy.transpose(ma.filled(exposures, 0.0))

    # Bucket assets into regions
    regionAssetMap = dict()
    for r in rmgList:
        rmg_assets = data.rmgAssetMap[r.rmg_id]
        if r.region_id not in regionAssetMap:
            regionAssetMap[r.region_id] = list()
        regionAssetMap[r.region_id].extend(rmg_assets)

    # Create country/region exposures
    if len(rmgList) > 1:
        if countryFactor:
            for rmg in rmgList:
                newColumn = numpy.zeros((len(data.universe)), float)
                rmgAssetsIdx = [data.assetIdxMap[sid] for sid in data.rmgAssetMap[rmg]]
                numpy.put(newColumn, rmgAssetsIdx, 1.0)
                exposures = numpy.concatenate((
                    exposures, newColumn[:,numpy.newaxis]), axis=1)
        else:
            for reg in regionAssetMap.keys():
                newColumn = numpy.zeros((len(data.universe)), float)
                rmgAssetsIdx = [data.assetIdxMap[sid] for sid in regionAssetMap[reg]]
                numpy.put(newColumn, rmgAssetsIdx, 1.0)
                exposures = numpy.concatenate((
                    exposures, newColumn[:,numpy.newaxis]), axis=1)

    # Create lncap column
    N = returns.data.shape[0]
    marketCaps = ma.filled(data.marketCaps, 0.0)
    C = int(round(N*0.05))
    C = numpy.clip(C, 1, 100)
    sortindex = ma.argsort(marketCaps)
    upperBound = marketCaps[sortindex[N-C]]
    lowerBound = marketCaps[sortindex[C]]
    marketCaps = numpy.clip(marketCaps, lowerBound, upperBound)
    lnCap = numpy.log(marketCaps+1.0)
    lnCap = lnCap / numpy.max(lnCap, axis=None)
    exposures = numpy.concatenate((
        exposures, lnCap[:,numpy.newaxis]), axis=1)

    # Sort returns into good and bad
    maskedReturns = ma.getmaskarray(returnsData)
    numberOkRets = ma.sum(maskedReturns==0, axis=1)
    numberOkRets = numberOkRets / float(numpy.max(numberOkRets, axis=None))
    numberOkRets = ma.masked_where(numberOkRets>0.95, numberOkRets)
    goodRetsIdx = numpy.flatnonzero(ma.getmaskarray(numberOkRets))
    badRetsIdx = numpy.flatnonzero(ma.getmaskarray(numberOkRets)==0)

    # Compute regression coefficients
    tmpRets = ma.array(returnsData, copy=True)
    if MADClipping:
        tmpRets = bucketedMAD(rmgList, date, tmpRets, data, modelDB)
    else:
        tmpRets = clip_extrema_old(tmpRets, 0.01)

    if len(goodRetsIdx) < 1:
        logging.warning('No assets with good enough returns histories')
        return returns
    goodRets = ma.filled(ma.take(tmpRets, goodRetsIdx, axis=0), 0.0)
    goodExposures = numpy.take(exposures, goodRetsIdx, axis=0)
    factorReturns = robustLinearSolver(goodRets, goodExposures, robust=robust).params

    # Warn on assets with all missing or zero exposures
    noExp = numpy.sum(exposures, axis=1)
    noExp = ma.masked_where(noExp==0.0, noExp)
    noExpIdx = numpy.flatnonzero(ma.getmaskarray(noExp))
    if len(noExpIdx) > 0:
        sidList = [data.universe[idx].getSubIDString() for idx in noExpIdx]
        logging.warning('%d assets with no or zero exposures in proxy: %s', len(noExpIdx), sidList)

    # Replace missing returns with proxied values
    estimatedReturns = numpy.dot(exposures, factorReturns)
    estimatedReturns = clip_extrema_old(estimatedReturns, 0.01)
    estimatedReturnsFill = ma.filled(ma.masked_where(
                    maskedReturns==0, estimatedReturns), 0.0)
    returns.data = ma.filled(returnsData, 0.0) + estimatedReturnsFill
    returns.estimates = estimatedReturns
    logging.info('Filling %s missing returns on the fly', numpy.size(numpy.flatnonzero(maskedReturns)))
    logging.debug('proxyMissingAssetReturnsLegacy: end')
    return returns

def clip_extrema_old(data, pct=0.05):
    linData = numpy.ravel(ma.filled(data, 0.0))
    sortindex = ma.argsort(linData)
    N = len(linData)
    C = int(round(N*pct))
    C = numpy.clip(C, 1, 100)
    upperBound = linData[sortindex[N-C]]
    lowerBound = linData[sortindex[C]]
    return ma.clip(data, lowerBound, upperBound)

def proxyMissingAssetReturnsV3(rmgList, date, returns, data, modelDB,
                            MADClipping=False, robust=False,
                            countryFactor=True, industryGroupFactor=True,
                            debugging=False, gicsDate=datetime.date(2016,9,1)):
    """Given a TimeSeriesMatrix of asset returns with masked
    values, fill in missing values using proxies.
    Assets are modelled using countries/regions, GICS Sectors/industry groups
    and log(cap) as factors.
    """
    logging.debug('proxyMissingAssetReturnsV3: begin')
    returnsData = ma.filled(returns.data, 0.0)
    returnsData = ma.masked_where(ma.getmaskarray(returns.data), returnsData)
    returns.data = ma.filled(returns.data, 0.0)
    returns.estimates = numpy.zeros((returns.data.shape), float)

    # Use Sector-level classification for exposures
    if industryGroupFactor:
        level = 'Industry Groups'
    else:
        level = 'Sectors'

    logging.info('Building industry classification, level %s, version date: %s', level, gicsDate)
    allExposures = buildGICSExposures(data.universe, date, modelDB, level=level, clsDate=gicsDate)
    allExposures = ma.filled(allExposures, 0.0)

    # Create lncap column
    N = returns.data.shape[0]
    marketCaps = ma.filled(data.marketCaps, 0.0)
    if MADClipping:
        marketCaps = twodMAD(marketCaps, nDev=[8.0, 8.0])
    else:
        C = int(round(N*0.05))
        C = numpy.clip(C, 1, 100)
        sortindex = ma.argsort(marketCaps)
        upperBound = marketCaps[sortindex[N-C-1]]
        lowerBound = marketCaps[sortindex[C]]
        marketCaps = numpy.clip(marketCaps, lowerBound, upperBound)
    lnCap = numpy.log(marketCaps+1.0)
    lnCap = lnCap / mlab_std(lnCap)
    allExposures = numpy.concatenate((allExposures, lnCap[:,numpy.newaxis]), axis=1)

    # Clip returns as required
    clipRets = ma.array(returns.data, copy=True)
    if MADClipping:
        clipRets = bucketedMAD(rmgList, date, clipRets, data, modelDB)
    else:
        clipRets = clip_extrema(clipRets, 0.01)

    # Loop round countries
    for rmg in rmgList:
        if rmg.rmg_id not in data.rmgAssetMap:
            logging.warning('No data in rmgAssetMap for %s', rmg.mnemonic)
            continue
        rmgAssets = [sid for sid in data.rmgAssetMap[rmg.rmg_id]]
        if len(rmgAssets) < 1:
            logging.warning('No assets for %s', rmg.mnemonic)
            continue
        rmgAssetsIdx = [data.assetIdxMap[sid] for sid in rmgAssets]
        exposures = numpy.take(allExposures, rmgAssetsIdx, axis=0)
        rmgReturnsClip = ma.take(clipRets, rmgAssetsIdx, axis=0)
        rmgReturns = ma.take(returnsData, rmgAssetsIdx, axis=0)

        # Sort returns into good and bad
        maskedReturns = ma.getmaskarray(rmgReturns)
        numberOkRets = ma.sum(maskedReturns==0, axis=1)
        numberOkRets = numberOkRets / float(numpy.max(numberOkRets, axis=None))
        numberOkRets = ma.masked_where(numberOkRets>0.95, numberOkRets)
        goodRetsIdx = numpy.flatnonzero(ma.getmaskarray(numberOkRets))
        badRetsIdx = numpy.flatnonzero(ma.getmaskarray(numberOkRets)==0)
        logging.info('Filling %s %s missing returns on the fly',
                numpy.size(numpy.flatnonzero(maskedReturns)), rmg.mnemonic)

        # Compute regression coefficients
        goodRets = ma.filled(ma.take(rmgReturnsClip, goodRetsIdx, axis=0), 0.0)
        goodExposures = numpy.take(exposures, goodRetsIdx, axis=0)
        okExpIdx = nonEmptyColumns(goodExposures)
        if len(okExpIdx) < 1:
            logging.warning('No non-zero exposures for %s', rmg.mnemonic)
            continue
        goodExposures = numpy.take(goodExposures, okExpIdx, axis=1)

        # Make sure that the regression is well-enough conditioned
        xwx = numpy.dot(numpy.transpose(goodExposures), goodExposures)
        (eigval, eigvec) = linalg.eigh(xwx)
        conditionNumber = max(eigval) / min(eigval)
        logging.debug('Regressor (%i by %i) has condition number %f',
                xwx.shape[0], xwx.shape[1], conditionNumber)

        if goodExposures.shape[0] < 2*goodExposures.shape[1] \
                or abs(conditionNumber) > 1.0e6:
            factorReturns = robustLinearSolver(goodRets, goodExposures, computeStats=False).params
        else:
            factorReturns = robustLinearSolver(goodRets, goodExposures, computeStats=False, robust=robust).params

        # Warn on assets with all missing or zero exposures
        noExpIdx = nonEmptyColumns(exposures, byColumn=False, findEmpty=True)
        if len(noExpIdx) > 0:
            sidList = [rmgAssets[idx].getSubIDString() for idx in noExpIdx]
            logging.warning('%s: %d assets with no or zero exposures in proxy: %s',
                    rmg.mnemonic, len(noExpIdx), sidList)

        # Create full array of estimated returns
        exposures = numpy.take(exposures, okExpIdx, axis=1)
        estimatedReturns = numpy.dot(exposures, factorReturns)
        if MADClipping:
            estimatedReturns = twodMAD(estimatedReturns, nDev=[1.5, 1.5], axis=1)
        else:
            estimatedReturns = clip_extrema(estimatedReturns, 0.01)

        # Replace missing returns with proxied values
        estimatedReturnsFilled = ma.filled(ma.masked_where(\
                maskedReturns==0, estimatedReturns), 0.0)
        rmgReturns = ma.filled(rmgReturns, 0.0) + estimatedReturnsFilled
        for (ii, idx) in enumerate(rmgAssetsIdx):
            returns.data[idx,:] = rmgReturns[ii, :]
            returns.estimates[idx,:] = estimatedReturns[ii, :]

    # Output for debugging
    error = ma.filled((returns.data-returns.estimates), 0.0)
    fNorm = numpy.max(numpy.sum(abs(error), axis=1))
    logging.info('Residual norm of proxied data: %.3f', fNorm)

    if debugging:
        idList = [sid.getSubIDString() for sid in data.universe]
        dtList = ['Obs-%d' % i for i in range(returns.data.shape[1])]
        dFile = 'tmp/data-raw-%s.csv' % date
        writeToCSV(returns.data, dFile, rowNames=idList, columnNames=dtList)
        dFile = 'tmp/data-est-%s.csv' % date
        writeToCSV(estimatedReturns, dFile, rowNames=idList, columnNames=dtList)

    logging.debug('proxyMissingAssetReturnsV3: end')
    return returns

def buildGICSExposures(subIssues, date, modelDB, level='Sectors',
        clsDate=datetime.date(2016,9,1)):
    """Build exposure matrix of GICS industry groups or sectors
    Note - returns transposed (i.e. correct) matrix of N assets by P factors
    """
    return modelDB.getGICSExposures(date, subIssues, level, clsDate)[0]

def proxyMissingAssetData(rmgList, date, dataArray, data, modelDB, outlierClass,
                          estu=None, countryFactor=True, industryGroupFactor=False,
                          debugging=False, robust=False, gicsDate=datetime.date(2016,9,1),
                          minGoodAssets=0.1, pctGoodReturns=0.95, forceRun=False):
    """Given a TimeSeriesMatrix of asset data with masked
    values, fill in missing values using proxies.
    Assets are modelled using countries/regions, GICS Sectors/industry groups
    and log(cap) as factors.
    """
    logging.debug('proxyMissingAssetData: begin')
    vector = False
    if len(dataArray.shape) < 2:
        dataArrayCopy = ma.array(dataArray[:,numpy.newaxis], copy=True)
        vector = True
    else:
        dataArrayCopy = ma.array(dataArray, copy=True)
    dataEstimates = numpy.zeros((dataArrayCopy.shape), float)

    # Get sector/industry group exposures
    if industryGroupFactor:
        level = 'Industry Groups'
    else:
        level = 'Sectors'
    allExposures = buildGICSExposures(data.universe, date, modelDB, level=level, clsDate=gicsDate)
    allExposures = ma.filled(allExposures, 0.0)

    # Bucket assets into regions/countries
    regionIDMap = dict()
    regionAssetMap = dict()
    if countryFactor or (len(rmgList) < 2):
        for rmg in rmgList:
            regionIDMap[rmg.rmg_id] = rmg.mnemonic
            regionAssetMap[rmg.rmg_id] = data.rmgAssetMap[rmg.rmg_id]
    else:
        for r in rmgList:
            rmg_assets = data.rmgAssetMap[r.rmg_id]
            if r.region_id not in regionAssetMap:
                regionAssetMap[r.region_id] = list()
                regionIDMap[r.region_id] = 'Region %s' % str(r.region_id)
            regionAssetMap[r.region_id].extend(rmg_assets)

    # Create lncap column
    marketCaps = ma.filled(data.marketCaps, 0.0)
    bounds = prctile(marketCaps, [5.0, 95.0])
    marketCaps = numpy.clip(marketCaps, bounds[0], bounds[1])
    lnCap = numpy.log(marketCaps+1.0)

    # Create regression weights
    rootCap = ma.sqrt(marketCaps)
    if estu is None:
        regWeights = numpy.array(numpy.sqrt(marketCaps), float)
    else:
        regWeights = numpy.zeros((len(marketCaps)), float)
        estuIdx = [data.assetIdxMap[sid] for sid in estu]
        for idx in estuIdx:
            regWeights[idx] = ma.sqrt(marketCaps[idx])

    # Loop round countries/regions
    for regID in regionIDMap.keys():

        # Get relevant assets and data
        rmgAssets = [sid for sid in regionAssetMap[regID]]
        if len(rmgAssets) < 1:
            logging.warning('No assets for %s', regionIDMap[regID])
            continue
        rmgAssetsIdx = [data.assetIdxMap[sid] for sid in rmgAssets]
        exposures = numpy.take(allExposures, rmgAssetsIdx, axis=0)
        rmgData = ma.take(dataArrayCopy, rmgAssetsIdx, axis=0)
        estuWeights = numpy.take(regWeights, rmgAssetsIdx, axis=0)

        # Quick and dirty standardisation of lncap column
        lnCapReg = numpy.take(lnCap, rmgAssetsIdx, axis=0)
        lnCapEstu = numpy.take(lnCapReg, numpy.flatnonzero(estuWeights), axis=0)
        if len(lnCapEstu) < 1:
            logging.warning('No non-missing estu weights')
            continue
        minBound = numpy.min(lnCapEstu, axis=None)
        maxBound = numpy.max(lnCapEstu, axis=None)
        lnCapReg = numpy.clip(lnCapReg, minBound, maxBound)
        capWtEstu = estuWeights * estuWeights
        meanWt = numpy.average(lnCapReg, axis=0, weights=capWtEstu)
        lnCapReg = (lnCapReg - meanWt) / numpy.std(lnCapEstu)
        exposures = numpy.concatenate((exposures, lnCapReg[:,numpy.newaxis]), axis=1)

        # Sort data into good and bad
        maskedData = ma.getmaskarray(rmgData)
        numberOkData = ma.sum(maskedData==0, axis=1)
        maxNonMissingReturns = float(numpy.max(numberOkData, axis=None))
        numberOkData = numberOkData / maxNonMissingReturns
        logging.info('Filling %s %s missing values on the fly',
                numpy.size(numpy.flatnonzero(maskedData)), regionIDMap[regID])

        # Check that a reasonable proportion of assets have enough returns
        enoughData = False
        newTol = pctGoodReturns
        while (enoughData is False) and (newTol > 0.01):
            goodData = ma.masked_where(numberOkData>newTol, numberOkData)
            goodDataIdx = numpy.flatnonzero(ma.getmaskarray(goodData))
            if len(goodDataIdx) < minGoodAssets * len(rmgAssets):
                logging.error('Not enough assets (%d/%d) with good enough data to fit model',
                        len(goodDataIdx), len(rmgAssets))
                logging.info('Lowering tolerance from %.3f to %.3f', newTol, 0.9*newTol)
                newTol = 0.9 * newTol
            else:
                enoughData = True

        if not forceRun:
            assert(len(goodDataIdx)>=minGoodAssets*len(rmgAssets))

        # Create exposure matrix
        goodData = ma.filled(ma.take(rmgData, goodDataIdx, axis=0), 0.0)
        goodExposures = numpy.take(exposures, goodDataIdx, axis=0)
        weights = numpy.take(estuWeights, goodDataIdx, axis=0)
        if len(weights) < minGoodAssets * len(rmgAssets):
            logging.warning('No assets with good enough data and non-missing weights')
            continue
        sumWeight = ma.sum(weights, axis=None)
        if sumWeight <= 0.0:
            logging.warning('No non-zero weights for %s', regionIDMap[regID])
            continue
        weights /= sumWeight
        okExpIdx = nonEmptyColumns(goodExposures)
        if len(okExpIdx) < 1:
            logging.warning('No non-zero exposures for %s', regionIDMap[regID])
            continue
        goodExposures = numpy.take(goodExposures, okExpIdx, axis=1)

        # Make sure that the regression is well-enough conditioned
        wt_Exp = numpy.transpose(weights * numpy.transpose(goodExposures))
        xwx = numpy.dot(numpy.transpose(wt_Exp), wt_Exp)
        (eigval, eigvec) = linalg.eigh(xwx)
        if min(eigval) == 0.0:
            conditionNumber = 1.0e16
        else:
            conditionNumber = max(eigval) / min(eigval)
        logging.info('Regressor (%i by %i) has condition number %f',
                xwx.shape[0], xwx.shape[1], conditionNumber)
        if abs(conditionNumber) > 1.0e6:
            logging.warning('Condition number too high: aborting proxy computation')
            continue

        # Perform the regression
        if outlierClass is not None:
            tmpRets = outlierClass.twodMAD(goodData)
        else:
            tmpRets = goodData
        if goodExposures.shape[0] < 2*goodExposures.shape[1]:
            factorReturns = robustLinearSolver(tmpRets, goodExposures,
                    weights=weights, computeStats=False).params
        else:
            logging.info('Solving linear model, dimensions: %s, robust %s',
                    goodExposures.shape, robust)
            factorReturns = robustLinearSolver(tmpRets, goodExposures,
                    weights=weights, computeStats=False, robust=robust).params

        # Warn on assets with all missing or zero exposures
        noExpIdx = nonEmptyColumns(exposures, byColumn=False, findEmpty=True)
        if len(noExpIdx) > 0:
            sidList = [rmgAssets[idx].getSubIDString() for idx in noExpIdx]
            logging.warning('%s: %d assets with no or zero exposures in proxy: %s',
                    regionIDMap[regID], len(noExpIdx), sidList)

        # Replace missing observations with proxied values
        exposures = numpy.take(exposures, okExpIdx, axis=1)
        estimatedData = clip_extrema(numpy.dot(exposures, factorReturns), 0.01)
        estimatedDataFilled = ma.filled(ma.masked_where(maskedData==0, estimatedData), 0.0)
        rmgData = ma.filled(rmgData, 0.0) + estimatedDataFilled
        for (ii, idx) in enumerate(rmgAssetsIdx):
            dataArrayCopy[idx,:] = rmgData[ii, :]
            dataEstimates[idx,:] = estimatedData[ii, :]

    # Output for debugging
    error = ma.filled((dataArrayCopy-dataEstimates), 0.0)
    fNorm = numpy.max(numpy.sum(abs(error), axis=1))
    logging.info('Residual norm of proxied data: %.3f', fNorm)
 
    if debugging:
        idList = [sid.getSubIDString() for sid in data.universe]
        dtList = ['Obs-%d' % i for i in range(dataArray.shape[1])]
        dFile = 'tmp/data-raw-%s.csv' % date
        writeToCSV(dataArray, dFile, rowNames=idList, columnNames=dtList, dp=8)
        dFile = 'tmp/data-estm-%s.csv' % date
        writeToCSV(dataEstimates, dFile, rowNames=idList, columnNames=dtList, dp=8)

    logging.debug('proxyMissingAssetData: end')
    if vector:
        return dataArrayCopy[:,0], dataEstimates[:,0]
    else:
        return dataArrayCopy, dataEstimates

def clip_extrema(data, pct=0.05):
    linData = numpy.ravel(ma.filled(data, 0.0))
    sortindex = ma.argsort(linData)
    N = len(linData)
    C = min(100, int(round(N*pct)))
    upperBound = linData[sortindex[N-C-1]]
    lowerBound = linData[sortindex[C]]
    return ma.clip(data, lowerBound, upperBound)

def nonEmptyColumns(data, byColumn=True, zerosAreMissing=True, findEmpty=False):

    # By column or by row
    if byColumn:
        ax = 0
    else:
        ax = 1

    # Whether or not to treat zeros as missing
    if zerosAreMissing:
        missingDataMask = ma.getmaskarray(ma.masked_where(data==0.0, data))
    else:
        missingDataMask = ma.getmaskarray(data)
    columnSum = ma.sum(missingDataMask==0, axis=ax)
    columnSum = ma.masked_where(columnSum==0, columnSum)

    if findEmpty:
        # If we are actually looking for the empty rows/columns
        return numpy.flatnonzero(ma.getmaskarray(columnSum))
    return numpy.flatnonzero(ma.getmaskarray(columnSum)==0)

def ma_median(data, axis=None):
    """Computes median for an array containing masked values.
    Also supports computing along a specified axis, whereas
    numpy.median() is implicitly axis=0.
    If a median cannot be computed (ie, no non-masked values)
    the resulting value itself is masked.
    """
    if axis is None or len(data.shape)==1:
        data_flat = ma.ravel(data)
        data_flat = [data_flat[i] for i in numpy.flatnonzero(
                     ma.getmaskarray(data_flat)==0)]
        if len(data_flat) > 0:
            return ma.median(data_flat, axis=0)
        else:
            return ma.masked
    if axis==1:
        data = ma.transpose(data)
    mask = ma.getmaskarray(data)
    med = Matrices.allMasked(data.shape[1])
    for j in range(data.shape[1]):
        data_good = [data[i,j] for i in numpy.flatnonzero(mask[:,j]==0)]
        if len(data_good) > 0:
            med[j] = ma.median(data_good, axis=0)
    return med

def eps_median(data, buckets=10, axis=0):
    """ Robust median routine that divides data into blocks,
    computes the median of each block, then returns the
    final median for the entire data as that block
    median which is closest to the simple median (that
    computed along the entire data history
    """
    # Change number of buckets if too few observations
    if (data.shape[axis] / buckets) < 20.0:
        buckets = int(data.shape[axis] / 20.0)
    delta = int(data.shape[axis] / buckets)
    # Compute simple median
    fullMedian = ma.median(data, axis=axis)
    #logging.info('FULL MEDIAN: %s' % fullMedian)
    i = 0
    # Loop round buckets of data
    for b in range(buckets):
        ib = min(i+delta, data.shape[axis])
        idx = list(range(i,ib))
        # Compute bucket median
        bucketMedian = ma.median(numpy.take(\
                data, idx, axis=axis), axis=axis)
        #logging.info('BUCKET %d MEDIAN: %s' % (b, bucketMedian))
        diff = abs(bucketMedian - fullMedian)
        # Overwrite current "best" median if new estimate
        # is closer
        if i == 0:
            runningMedian = bucketMedian
            minDiff = diff
        else:
            if len(data.shape)>1:
                for j in range(len(runningMedian)):
                    if diff[j] < minDiff[j]:
                        runningMedian[j] = bucketMedian[j]
                        minDiff[j] = diff[j]
            else:
                if diff < minDiff:
                    runningMedian = bucketMedian
                    minDiff = diff
        i += delta
    #logging.info('FINAL MEDIAN: %s' % runningMedian)
    return runningMedian

def spline_dva(originalData, sampleLength, upperBound=0.1, lowerBound=-0.1,
            factorIndices=None, downWeightEnds=False):
    """Applies 'secret sauce/supertrick/DVA' to a returns series.
    data should be an asset-by-time array with no masked values
    where the most recent observation is at position 0.
    If factorIndices is not None, apply dva to only those factors in factorIndices.
    If factorIndices is None, apply dva to all factors in originalData.
    """

    logging.debug('spline_dva: begin')
    if factorIndices is None:
        factorIndices = list(range(0,originalData.shape[0]))

    if len(factorIndices) == 0:
        return originalData

    data = ma.array(originalData, copy=True)
    sampleLength = int(sampleLength)
    logging.debug('DVA upper bound: %s, lower bound: %s', upperBound, lowerBound)
    # Set up parameters
    # T is the length of each sample used to compute statistics
    # t is the current start point of the chunk to be scaled
    # Except for the first chunk of data, overlapping chunks
    # are used of length T, centred on point t
    # The very first chunk (that which is thereafter used as a 
    # reference) runs from t=0 to t=T
    T = sampleLength
    t = 0
    tMax = data.shape[1]
    tMinusT = 0

    # Initialise some extra arrays for spline interpolation
    m = int(2 * tMax / T) + 2
    tSample = numpy.zeros((m), float)
    wtSample = ma.ones((data.shape[0], m), float)
    tFull = numpy.array((list(range(tMax))), float)

    # Setting this flag to 1 will ensure that the first half of the initial
    # chunk is unscaled, set to 0 otherwise
    initialScaleIndex = 0

    # Index of values of t in vector tSample to keep track of length of vector
    loopCount = initialScaleIndex
    if initialScaleIndex != 0:
        tSample[1] = int(T/2.0)

    logging.info('DVA Reference sample Length: %d, Type=spline', sampleLength)
    dataCopy = ma.take(data, factorIndices, axis=0)

    # Loop through the chunks of returns
    while tMinusT < tMax: # loop through time series
        # Pick out our sample of returns
        tPlusT = min(t+T, tMax)
        dataSample = abs(ma.take(dataCopy, list(range(tMinusT, tPlusT)), axis=1))
        if downWeightEnds:
            endWts = computePyramidWeights(20, 20, dataSample.shape[1])
            dataSample *= endWts

        # Derive statistics on the returns sample
        tMAD = ma.average(dataSample, axis=1)[:,numpy.newaxis] # 232 x 1 vector
        tMAD = ma.masked_where(tMAD<=0.0, tMAD)

        if t > 0:

            # Calculate scaling factor for current chunk
            Lambda = (baseMAD / tMAD).filled(1.0)
            if (upperBound is None) and (lowerBound is None):
                outlierClass = Outliers.Outliers()
                Lambda = outlierClass.twodMAD(Lambda, suppressOutput=True, nBounds=[3.0, 3.0])

            # Trim large values
            lRatio = (Lambda / prevLambda) - 1.0
            if upperBound is not None:
                Lambda = ma.where(lRatio > upperBound, (1.0+upperBound)*prevLambda, Lambda)
            if lowerBound is not None:
                Lambda = ma.where(lRatio < lowerBound, (1.0+lowerBound)*prevLambda, Lambda)

            # Add current t value to vector of sample t values
            tSample[loopCount] = min(t, tMax)

            # Add current scale factors to vector of sample scale factors
            ma.put(wtSample[:,loopCount], factorIndices, Lambda)
            prevLambda = numpy.array(Lambda, copy=1)

            logging.debug('DVA Scaling: (Min, Med, Max): (%.3f, %.3f, %.3f)',
                    numpy.min(Lambda), ma_median(Lambda), numpy.max(Lambda))
        else:
            # For the initial chunk, set some parameters
            baseMAD = ma.array(tMAD, copy=True)
            prevLambda = ma.ones((baseMAD.shape[0],1),float)

        # Reset various stepping parameters
        if t == 0:
            t += T
            T = int(numpy.ceil(T / 2.0))
        else:
            t += T
        # A few safeguards, to prevent overreaching array bounds or tiny chunks
        tMinusT = max(t-T, 0)
        if t > tMax-20 and t < tMax:
            t = tMax
        loopCount += 1

    # Extra stuff for spline-fitting
    tSample = tSample[initialScaleIndex:loopCount]
    wtSample = wtSample[:,initialScaleIndex:loopCount]
   
    #Handle case where spline.splrep will throw exception
    #and skip adjustment
    if tSample.shape[0] <= 3: 
        #This should only happen because of silly inputs 
        #in the RiskParameters
        return data

    for i in factorIndices: # loop through factors, get interpolated weighted over full history
        j = int(tSample[0])
        # Compute spline coefficients over sample
        spline_coef = spline.splrep(tSample, wtSample[i,:]) # get B-spline representation of 1-D curve given by (x=tSample, y=wtSample[i,:])
        # Compute interpolated values over full history
        wtFull = spline.splev(tFull[j:], spline_coef)
        # And scale the data by the spline-fitted interpolants
        data[i,j:] *= wtFull

    logging.debug('spline_dva: end')
    return data

def non_negative_least_square(A, b):
    """Routine which solves b=Ax via least-squares fit, but where
    the solution x is constrained to be non-negative
    A is n by k 
    b is n by 1
    """
    
    # Initialise variables
    n = A.shape[0]
    k = A.shape[1]
    b = numpy.ravel(b)
    x = numpy.zeros((k), float)
    P = set()
    Z = set(range(k))
    iter = 0
    itmax = 3*n

    # Main loop
    while len(P) <= k and iter <= itmax:

        # Compute dual vector w=A'(b-Ax)
        w = b - numpy.sum(A*x, axis=1)
        w = numpy.sum(numpy.transpose(A)*w, axis=1)

        # Find maximum element of w whose index is in Z
        wMax = 0.0
        for idx in Z:
            if w[idx] > wMax:
                wMax = w[idx]
                wMaxIdx = idx

        # If convergence attained, break out of here
        if wMax <= 0.0 or len(Z) == 0:
            return x

        # Shift index of max(W) from Z to P
        P.add(wMaxIdx)
        Z = Z.difference(P)
        feasible = False

        while (feasible == False):
            # Solve reduced ols system
            y = numpy.zeros((k), float)
            Ap = numpy.take(A, list(P), axis=1)
            lhs = numpy.dot(numpy.transpose(Ap), Ap)
            rhs = numpy.dot(numpy.transpose(Ap), b[:,numpy.newaxis])
            try:
                yp = linalg.solve(lhs, rhs)[:,0]
            except linalg.linalg.LinAlgError as err:
                if 'Singular matrix' in str(err):
                    logging.warning('Singular matrix in linalg.solve')
                    yp = numpy.zeros((lhs.shape[0]), float)
                    break
                else:
                    raise
            numpy.put(y, list(P), yp)

            # Check whether y is a suitable test solution
            yMin = min(yp)
            if yMin <= 0:
                logging.debug('Infeasible coefficient')
                # if not, compute alpha
                alpha = 2.0
                for j in P:
                    if abs(x[j] - y[j])< 1.0e-12:
                        t = 0.0
                    else:
                        t = x[j] / (x[j] - y[j])
                    if alpha > t and y[j] <= 0.0:
                        alpha = t
                        alphaMinIdx = j
                # Move index of min alpha to Z and return to inner loop
                Z.add(alphaMinIdx)
                P = P.difference(Z)
            else:
                feasible = True
                x = y
            iter += 1
            if iter > itmax:
                logging.warning('Maximum iterations exceeded in NNLS')
                feasible = True
                x = y
    return x

def screen_data(dataIn, fill=False, fillValue=0.0):
    data = ma.array(dataIn, copy=True)
    maskedData = numpy.array(ma.getmaskarray(data), copy=True)
    data = ma.filled(data, fillValue)
    badData = numpy.isfinite(data)==0
    data = ma.where(badData, fillValue, data)
    if not fill:
        maskedData = ((maskedData==0) * (badData==0))==0
        data = ma.masked_where(maskedData, data)
    return data

def output_stats(dataIn, name):
    logger = logging.getLogger()
    if logger.isEnabledFor(logging.DEBUG):
        data = screen_data(dataIn)
        mask = ma.getmaskarray(data)
        zeros = ma.where(ma.ravel(data)==0.0)[0]
        logging.debug('%s data: dim(%s) Bounds: [%.3f, %.3f, %.3f], Missing: %d, Zero: %d',
                name, data.shape, ma.min(data, axis=None),
                ma.average(data, axis=None), ma.max(data, axis=None),
                len(numpy.flatnonzero(mask)), len(numpy.ravel(zeros)))

def blend_values(oldDataArray, newDataArray, date, start_date, end_date):
    """Given two arrays of data, returns a weighted average of the two
    based on the date relative to two reference dates
    The closer the date to start_date, the greater the weight given to oldDataArray
    """
    if start_date > end_date:
        return None
    if date < start_date:
        return oldDataArray
    if date > end_date:
        return newDataArray

    # Deal with missing data - fill in missing values in one array with values from the other
    oldMask = ma.getmaskarray(oldDataArray)
    oldDataArray = ma.where(oldMask, newDataArray, oldDataArray)
    newMask = ma.getmaskarray(newDataArray)
    newDataArray = ma.where(newMask, oldDataArray, newDataArray)

    # Return blended mix if within phase-in dates
    ratio = 1.0 - float((date - start_date).days) / float((end_date - start_date).days)
    blendData = (oldDataArray * ratio) + ((1.0-ratio) * newDataArray)
    logging.info('Blending data, weight on new array: %s', 1.0-ratio)

    oldData_tmp = ma.filled(oldDataArray, 0.0)
    newData_tmp = ma.filled(newDataArray, 0.0)
    blndData_tmp = ma.filled(blendData, 0.0)
    if len(blendData.shape) > 1:
        for jdx in range(blendData.shape[1]):
            logging.info('Correlation between column %d blended data and old: %.6f, new: %.6f',
                    jdx, numpy.corrcoef(oldData_tmp[:,jdx], blndData_tmp[:,jdx])[0,1],
                    numpy.corrcoef(newData_tmp[:,jdx], blndData_tmp[:,jdx])[0,1])
    else:
        logging.info('Correlation between blended data and old: %.6f, new: %.6f',
                numpy.corrcoef(oldData_tmp, blndData_tmp)[0,1], numpy.corrcoef(newData_tmp, blndData_tmp)[0,1])
    return blendData

def writeToCSV(data, filepath, columnNames=None, rowNames=None, delim=',', dp=12):
    """ Writes array of data to a csv file
    """
    if len(data.shape) == 1:
        data = data[:,numpy.newaxis]
    data = p2_round(data, dp)
    outfile = open(filepath, 'w')
    mask = ma.getmaskarray(data)
    # Write column names if required
    if columnNames != None:
        if rowNames != None:
            outfile.write('%s' % delim)
        for f in columnNames:
            outfile.write('%s%s' % (str(f).replace(delim,''), delim))
        outfile.write('\n')
    # Write data values
    for i in range(data.shape[0]):
        if rowNames != None:
            outfile.write('%s%s' % (str(rowNames[i]).replace(delim,''), delim))
        for j in range(data.shape[1]):
            if not mask[i,j]:
                outfile.write('{0:.{1}f}'.format(data[i,j], dp))
            outfile.write('%s' % delim)
        outfile.write('\n')
    outfile.close()
    return

def readFromCSV(filepath, columnNames=None, rowNames=None,
                matchNames=False, zipped=False):
    """ Reads array of data from a csv file
    """
    # Flag to determine at which column data begins
    if rowNames == None:
        incr = 0
    else:
        incr = 1
    # Check whether file exists
    try:
        if zipped:
            infile = gzip.open(filepath, 'r')
        else:
            infile = open(filepath, 'r')
    except:
        logging.info('File %s does not exist', filepath)
        return []
    # Initialise parameters
    iDim = 0
    jDim = 0
    firstRow = True
    dataList = []
    missList = []
    rowNameList = []
    colNameList = []
    # Loop round rows
    for inline in infile:
        fields = inline.split(',')
        if firstRow:
            # Determine number of columns of data
            jDim = len(fields) - incr
            if fields[-1] == '\n':
                jDim-=1
            # Read in first row - either names or data
            if columnNames != None:
                for j in range(jDim):
                    colNameList.append(fields[j+incr])
            else:
                if rowNames != None:
                    rowNameList.append(fields[0])
                for j in range(jDim):
                    try:
                        elt = float(fields[j+incr])
                        missList.append(False)
                    except:
                        elt = 0.0
                        missList.append(True)
                    dataList.append(elt)
                iDim+=1
            firstRow = False
        else:
            # Read remaining lines
            if rowNames != None:
                rowNameList.append(fields[0])
            for j in range(jDim):
                try:
                    elt = float(fields[j+incr])
                    missList.append(False)
                except:
                    elt = 0.0
                    missList.append(True)
                dataList.append(elt)
            iDim+=1
    infile.close()
    # Arrange data into array of correct dimension
    data = numpy.array(dataList, copy=1)
    data = numpy.reshape(data, (iDim, jDim))
    missing = numpy.array(missList, copy=1)
    missing = numpy.reshape(missing, (iDim, jDim))
    data = ma.masked_where(missing, data)
    # Re-order data if required
    if matchNames == True:
        # First get factors in correct order
        if rowNames != None:
            rowNames = [f.replace(',','') for f in rowNames]
            rowIdxMap = dict(zip([n for n in rowNameList],
                range(len(rowNameList))))
            rowOrder = [rowIdxMap[n] for n in rowNames if n in rowIdxMap]
            data = numpy.take(data, rowOrder, axis=0)
        if columnNames != None:
            columnNames = [f.replace(',','') for f in columnNames]
            colIdxMap = dict(zip([n for n in colNameList],
                range(len(colNameList))))
            colOrder = [colIdxMap[n] for n in columnNames if n in colIdxMap]
            data = numpy.take(data, colOrder, axis=1)
        # Secondly resize data matrix to correct dimensions
        if rowNames != None:
            data1 = Matrices.allMasked((len(rowNames), data.shape[1]))
            rowIds = [i for (i,n) in enumerate(rowNames) \
                    if n in rowNameList]
            for (ii, iLoc) in enumerate(rowIds):
                data1[iLoc, :] = data[ii, :]
            data = ma.array(data1, copy=True)
        if columnNames != None:
            data1 = Matrices.allMasked((data.shape[0], len(columnNames)))
            colIds = [i for (i,n) in enumerate(columnNames) \
                    if n in colNameList]
            for (jj, jLoc) in enumerate(colIds):
                data1[:, jLoc] = data[:, jj]
            data = ma.array(data1, copy=True)
    return data

def compute_cointegration_parameters(data, subIssueGroups, subIssues,
                    rmgAssetMap=None, TOL=1.0e-6, skipDifferentMarkets=True):
    """Given a returns series and a dataset of linked issues,
    computes the cointegration beta between each pair of
    linked assets, along with the p-value from the 
    Dickey-Fuller test  as a measure of their degree of cointegration
    """
    # Initialise important stuff
    dataCopy = ma.masked_where(data<=-1.0, data)
    maskedData = ma.getmaskarray(dataCopy)
    dataCopy = ma.filled(dataCopy, 0.0)
    assetIdxMap = dict([(j,i) for (i,j) in enumerate(subIssues)])
    coefDict = dict()
    dfCValueDict = dict()
    errorVarDict = dict()
    dfStatDict = dict()
    pValueDict = dict()
    nobsDict = dict()

    # Get mapping of asset to RMGs
    if rmgAssetMap == None:
        assetRMGMap = dict(zip(subIssues, [1]*len(subIssues)))
    else:
        assetRMGMap = dict([(sid, rmg_id) for (rmg_id, ids) in \
                rmgAssetMap.items() for sid in ids])

    import statsmodels.tsa.stattools as ts
    import statsmodels.api as sm
     
    # Create history of pseudo-log-prices
    prices = numpy.log(1.0 + dataCopy)
    prices = screen_data(prices)
    prices[:,0] = 1.0
    n = prices.shape[1]
    for k in range(1,n):
        prices[:,k] += prices[:,k-1]
    prices = ma.masked_where(maskedData, prices)
    results = Struct()

    # Loop round sets of linked assets
    for (groupId, subIssueList) in subIssueGroups.items():
        indices  = [assetIdxMap[n] for n in subIssueList]
        nExact = 0

        # Loop round first of each pair
        for sid1 in subIssueList:
            coefDict[sid1] = dict()
            dfCValueDict[sid1] = dict()
            errorVarDict[sid1] = dict()
            dfStatDict[sid1] = dict()
            nobsDict[sid1] = dict()
            pValueDict[sid1] = dict()
            idx1 = assetIdxMap[sid1]
            price1 = prices[idx1,:]
            mask1 = ma.getmaskarray(price1)

            # Loop round second of the pair
            for sid2 in subIssueList:
                if sid1 == sid2:
                    continue
                # If assets are assigned to two different markets, impose zero cointegration and skip
                if skipDifferentMarkets and (assetRMGMap[sid1] != assetRMGMap[sid2]):
                    continue
                idx2 = assetIdxMap[sid2]
                price2 = prices[idx2,:]
                mask2 = ma.getmaskarray(price2)
                 
                # Locate missing values
                nonMissingIdx = numpy.flatnonzero((mask1==0) * (mask2==0))
                if len(nonMissingIdx) > 4:

                    # Sift out observations where both prices are missing
                    tmpPrice1 = ma.take(price1, nonMissingIdx, axis=0)
                    tmpPrice2 = ma.take(price2, nonMissingIdx, axis=0)

                    # Perform OLS regression on log prices
                    tmpPrice1 = sm.add_constant(tmpPrice1)
                    ols_result = sm.OLS(tmpPrice2, tmpPrice1).fit()
                    coef = ols_result.params[-1]
                    error = ols_result.resid
                    eps = error[1:] - error[:-1]
                    errorVar = numpy.var(eps)
                    coefDict[sid1][sid2] = coef
                    errorVarDict[sid1][sid2] = errorVar
                    nobsDict[sid1][sid2] = len(error)

                    # Look for cases of all or most of the errors being zero
                    # i.e. two exact price series. The ADF routine can fail in
                    # such situations
                    nZeroError = len(numpy.flatnonzero(ma.getmaskarray(ma.masked_where(
                        abs(error) < TOL, error))))
                    if nZeroError > 0.75 * len(error):
                        nExact += 1
                        coefDict[sid1][sid2] = 1.0
                        pValueDict[sid1][sid2] = 0.0
                        dfCValueDict[sid1][sid2] = 1.0
                        dfStatDict[sid1][sid2] = 1.0
                    else:
                        try:
                            # Dickey-Fuller test
                            adf = ts.adfuller(error, maxlag=2, autolag=None)
                            adfstat = adf[0]
                            cvalue = adf[4]['5%']
                            pvalue = adf[1]
                            adfstat = screen_data(ma.array([adfstat]), fill=True)[0]

                            # Save the results
                            dfCValueDict[sid1][sid2] = cvalue
                            dfStatDict[sid1][sid2] = adfstat
                            pValueDict[sid1][sid2] = pvalue

                        except:
                            logging.warning('Dickey Fuller test failed for assets (%s, %s) %d obs, Beta: %s',
                                    sid1.getSubIDString(), sid2.getSubIDString(), len(nonMissingIdx), coef)
                            pValue[sid1][sid2] = 1.0
                            coefDict[sid1][sid2] = 1.0
                            dfCValueDict[sid1][sid2] = 1.0
                            dfStatDict[sid1][sid2] = 1.0

    results.coefDict = coefDict
    results.errorVarDict = errorVarDict
    results.dfCValueDict = dfCValueDict
    results.nobsDict = nobsDict
    results.dfStatDict = dfStatDict
    results.pValueDict = pValueDict
    return results 

def compute_cointegration_parameters_legacy(data, subIssueGroups, subIssues,
                    rmgAssetMap=None, TOL=1.0e-4):
    """Given a returns series and a dataset of linked issues,
    computes the cointegration beta between each pair of
    linked assets, along with the p-value from the
    Dickey-Fuller test as a measure of their degree of cointegration
    """
    # Initialise important stuff
    dataCopy = ma.masked_where(data<=-1.0, data)
    maskedData = ma.getmaskarray(dataCopy)
    dataCopy = ma.filled(dataCopy, 0.0)
    assetIdxMap = dict([(j,i) for (i,j) in enumerate(subIssues)])
    coefDict = dict()
    dfPValueDict = dict()
    errorVarDict = dict()

    # Get mapping of asset to RMGs
    if rmgAssetMap == None:
        assetRMGMap = dict(zip(subIssues, [1]*len(subIssues)))
    else:
        assetRMGMap = dict([(sid, rmg_id) for (rmg_id, ids) in \
                rmgAssetMap.items() for sid in ids])

    import statsmodels.tsa.stattools as ts

    # Create history of pseudo-log-prices
    prices = numpy.log(1.0 + dataCopy)
    prices[:,0] = 1.0
    n = prices.shape[1]
    for k in range(1,n):
        prices[:,k] += prices[:,k-1]
    prices = ma.masked_where(maskedData, prices)

    # Loop round sets of linked assets
    for (groupId, subIssueList) in subIssueGroups.items():
        indices  = [assetIdxMap[n] for n in subIssueList]

        # Loop round first of each pair
        for sid1 in subIssueList:
            coefDict[sid1] = dict()
            dfPValueDict[sid1] = dict()
            errorVarDict[sid1] = dict()
            idx1 = assetIdxMap[sid1]
            price1 = prices[idx1,:]
            mask1 = ma.getmaskarray(price1)

            # Loop round second of the pair
            for sid2 in subIssueList:
                if sid1 == sid2:
                    continue
                # If assets are assigned to two different markets, impose zero
                # cointegration and skip
                if assetRMGMap[sid1] != assetRMGMap[sid2]:
                    pvalue = 1.0
                    coef = 0.0
                    errorVar = 0.0
                    continue
                idx2 = assetIdxMap[sid2]
                price2 = prices[idx2,:]
                mask2 = ma.getmaskarray(price2)
                pvalue = 1.0
                errorVar = 0.0

                # Locate missing values
                nonMissingIdx = numpy.flatnonzero((mask1==0) * (mask2==0))
                if len(nonMissingIdx) > 4:

                    # Sift out observations where both prices are missing
                    tmpPrice1 = ma.take(price1, nonMissingIdx, axis=0)
                    tmpPrice2 = ma.take(price2, nonMissingIdx, axis=0)

                    # Perform OLS regression on log prices
                    coef, error = ordinaryLeastSquares(tmpPrice2, tmpPrice1, implicit=True)
                    coef = screen_data(coef, fill=True)[0]
                    error = numpy.ravel(error)
                    eps = error[1:] - error[:-1]

                    # Look for cases of all or most of the errors being zero
                    # i.e. two exact price series. The ADF routine can fail in
                    # such situations
                    nZeroError = len(numpy.flatnonzero(ma.getmaskarray(ma.masked_where(
                        abs(error) < TOL, error))))

                    # Skip if most or all errors are zero
                    if nZeroError < 0.95 * len(error):
                        try:
                            # Dickey-Fuller test
                            adf = ts.adfuller(error, maxlag=2, autolag=None)
                            pvalue = adf[1]
                            errorVar = numpy.var(eps)
                        except:
                            logging.warning('Dickey Fuller test failed for assets (%s, %s) %d obs, Beta: %s',
                                    sid1.getSubIDString(), sid2.getSubIDString(), len(nonMissingIdx), coef)
                            pvalue = 1.0
                    pvalue = screen_data(ma.array([pvalue]), fill=True)[0]
                else:
                    # If insufficent observations we assume perfect
                    # relationship but with 100% uncertainty
                    coef = 1.0
                    pvalue = 1.0

                # Save the results
                coefDict[sid1][sid2] = coef
                errorVarDict[sid1][sid2] = errorVar
                # If p-value is tiny, might as well call it zero
                if pvalue < TOL:
                    pvalue = 0.0
                dfPValueDict[sid1][sid2] = pvalue

    return coefDict, dfPValueDict, errorVarDict

def fill_and_smooth_returns(data, fillData, mask=None,
        indices=None, preIPOFlag=None):
    """ Given a returns matrix with missing values,
    together with an array of the same size containing proxies, this
    routine back-fills the history, and smooths out the
    non-missing values, to preserve the long-term return
    """
    logging.debug('fill_and_smooth_returns: begin')
    returns = screen_data(data)
    if mask is None:
        maskArray = numpy.array(ma.getmaskarray(returns), copy=True)
    else:
        maskArray = mask.copy()
    if preIPOFlag is None:
        preIPOFlag = numpy.zeros((returns.shape), bool)

    returns = ma.filled(returns, 0.0)

    # Manipulation of axes in case of 1-D arrays
    vector = False
    if len(data.shape) < 2:
        logging.info('Data is 1-D: assuming row vector')
        returns = returns[numpy.newaxis,:]
        fillData = fillData[numpy.newaxis,:]
        maskArray = maskArray[numpy.newaxis,:]
        vector = True
    if indices is None:
        indices = list(range(data.shape[0]))
   
    try:
        import nantools.utils as nanutils 
        fill_and_smooth_returns_fast(returns, fillData, maskArray, indices, preIPOFlag)
    except ImportError:
        raise
        fill_and_smooth_returns_slow(returns, fillData, maskArray, indices, preIPOFlag)

    returns = ma.array(returns, mask=maskArray)
    if vector:
        returns = ma.ravel(returns)
        maskArray = ma.ravel(maskArray)
    logging.debug('fill_and_smooth_returns: end')
    return (returns, maskArray)

def fill_and_smooth_returns_fast(returns, fillData, maskArray,
        indices, preIPOFlag):
    import nantools.utils as nanutils 
    fillData = ma.filled(fillData, numpy.nan)
    preIPOFlag = preIPOFlag.view(dtype=numpy.int8)
    maskArray = maskArray.view(dtype=numpy.int8)
    indices = numpy.array(indices,dtype=numpy.intp)
    nanutils.fill_and_smooth(returns, fillData, maskArray, preIPOFlag, indices)

def fill_and_smooth_returns_slow(returns, fillData, maskArray,
        indices, preIPOFlag):
    proxyMask = ma.getmaskarray(fillData)

    # Loop round indices to be filled by the relevant proxy
    for idx in indices:
        maskedRetIdx = numpy.flatnonzero(maskArray[idx,:])
        if len(maskedRetIdx) > 0:
            t0 = maskedRetIdx[0]
            cumret = 1.0
            for t in maskedRetIdx:
                tmin1 = max(t-1,0)
                # Fill in particular masked return
                if not proxyMask[idx,t]:
                    returns[idx, t] = fillData[idx,t]
                    maskArray[idx, t] = False
                # If consecutive missing returns, compound replacement returns
                if t-t0 < 2:
                    if not maskArray[idx,t] and not preIPOFlag[idx,tmin1]:
                        cumret = (1.0 + returns[idx,t]) * cumret
                # Else scale non-missing return by previous compounded return
                else:
                    if not maskArray[idx,t0+1]:
                        returns[idx,t0+1] = -1.0 + (1.0 + returns[idx,t0+1]) / cumret
                    else:
                        returns[idx,t0+1] = -1.0 + (1.0 / cumret)
                        maskArray[idx,t0+1] = False
                    if not maskArray[idx,t] and not preIPOFlag[idx,tmin1]:
                        cumret = 1.0 + returns[idx,t]
                    else:
                        cumret = 1.0
                t0 = t
                 
            # Deal with last in list
            if t < returns.shape[1]-1:
                if not maskArray[idx,t+1]:
                    returns[idx,t+1] = -1.0 + (1.0 + returns[idx,t+1]) / cumret
                else:
                    returns[idx,t+1] = -1.0 + (1.0 / cumret)
                    maskArray[idx,t+1] = False

def encryptFile(date, inputFileName, targetDir, delim='|', destFileName=None):
    """
        given a file, it creates an encrypted version of the same file in targetDir
    """
    import tempfile
    from wombat import scrambleString
    import base64
    import shutil
    tmpfile = tempfile.mkstemp(suffix='enc', prefix='tmp',dir=targetDir)
    
    os.close(tmpfile[0])
    tmpfilename=tmpfile[1]
    outFile=open(tmpfilename,'w')
    fileName=os.path.basename(inputFileName)
    
    # write out the encrypted data into outFile
    for line in open(inputFileName):
        line=line.strip()
        if not line:
            continue
        if line[0]=='#':
            outFile.write('%s\n' % line)
        else:
            idx=line.find(delim)
            if idx < 0:
                continue
            assetid=line[:idx]
            dataString=line[idx+1:]
            #outFile.write('%s|%s|%d\n' % (axid, wombat3(dataString),wombat(wombat3(dataString))))
            encVal = scrambleString(dataString, 25, assetid, date, '')
            base64val = base64.b64encode(encVal)
            outFile.write('%s%s%s\n' % (assetid,delim, base64val.decode('utf8')))
    outFile.close()
    if destFileName:
        outFileName='%s/%s' % (targetDir.rstrip('/'),destFileName)
    else:
        outFileName='%s/%s' % (targetDir.rstrip('/'),fileName)
    shutil.move(tmpfilename, outFileName)
    logging.info("Encrypted file %s to %s", inputFileName,outFileName)
    os.chmod(outFileName,0o644)

def sort_chinese_market(name, date, modelDB, marketDB, universe):
    """ Sort out Chinese stocks into the relevant types
    """
    from riskmodels import MarketIndex
    data = Struct()
    data.universe = universe
    data.marketCaps = numpy.zeros(len(data.universe))
    data.exposureMatrix = Matrices.ExposureMatrix(data.universe)
    data.exposureMatrix.addFactor(name, numpy.ones(
        len(data.universe)), Matrices.ExposureMatrix.CountryFactor)
    data.assetIdxMap = dict([(j,i) for (i,j) in enumerate(data.universe)])
    return MarketIndex.process_china_share_classes(data, date, modelDB, marketDB)

def procrustes_transform(source, target):
    """ Performs orthogonal Procrustean analysis to transform the source to be as close as possible to the target.
        The source and target are specified as pandas DataFrames.
        Returns a pandas dataframe with the transformed source.
    """

    if numpy.any(target.index != target.columns) or numpy.any(source.index != source.columns):
        raise Exception('Unable to perform procrustes transformation: source and/or target dataframes do not appear to be symmetric; verify that index and columns match.')

    if not set(target.index).issubset(set(source.index)):
        raise Exception('Unable to perform procrustes transformation: target.index is not a subset of source.index')

    # initialize transform matrix
    transform = pandas.DataFrame(numpy.eye(len(source.index)),index=source.index,columns=source.columns)

    idx = target.index
    sourceBlock = source.loc[idx,idx]
    targetBlock = target

    (s,x) = linalg.eigh(sourceBlock.values)
    s[s<0.0]=0.0
    sb =  (x * numpy.sqrt(s)).dot(x.T)

    sinv = 1.0/s
    sinv[numpy.isinf(sinv)]=0.0
    sb_inv = (x * numpy.sqrt(sinv)).dot(x.T)

    (s,x) = linalg.eigh(targetBlock.values)
    s[s<0.0]=0.0
    tb = (x * numpy.sqrt(s)).dot(x.T)

    m = numpy.dot(tb.T, sb)
    (u,d,v) = linalg.svd(m, full_matrices=True)
    Q = numpy.dot(u,v)

    transform.loc[idx,idx] = tb.dot(Q).dot(sb_inv)

    # compute final matrix
    finalMatrix = transform.dot(source).dot(transform.T)

    return finalMatrix

def findIndustryParent(indCls, modelDB, node, ancestorNames):
    """Beginning from an industry classification node, recursively goes up
    a level until it finds the relevant parent from the given list
    """
    # Check to see whether it's in current level
    parents = indCls.getAllParents(node, modelDB)
    if parents is None:
        return None
    parentNames = [p.name for p in parents if p.name in ancestorNames]
    if len(parentNames) == 1:
        return parentNames[0]

    # If not, go up a level
    for parent in parents:
        grandparent = findIndustryParent(indCls, modelDB, parent, ancestorNames)
        if grandparent in ancestorNames:
            return grandparent
    return None

def buildFMPMapping(rd, rWeights, nESTU, factorNames, nonMissingReturnsIdx):
    # Compute set of FMPs using output from the model factor regression
    fmpDict = dict()
    sidIdxMap = dict(zip(rd.subIssues, list(range(len(rd.subIssues)))))

    # Set up regressor matrix components
    regM = rd.regressorMatrix * numpy.sqrt(rWeights)
    xwx = numpy.dot(regM, numpy.transpose(regM))
    xtw = regM * numpy.sqrt(rWeights)
    tfMatrix = numpy.eye(nESTU, dtype=float)

    # Incorporate thin factor component, if relevant
    if hasattr(rd, 'dummyRetWeights') and (type(rd.dummyRetWeights) is not list):
        tfMatrix = numpy.eye(nESTU, dtype=float)
        dummyRetWeights = numpy.take(rd.dummyRetWeights, nonMissingReturnsIdx, axis=1)
        sumWgts = ma.sum(dummyRetWeights, axis=1)
        for idx in range(len(sumWgts)):
            dummyRetWeights[idx,:] = dummyRetWeights[idx,:] / sumWgts[idx]
        tfMatrix = ma.concatenate([tfMatrix, dummyRetWeights], axis=0)

    # Incorporate constraints, if relevant
    if rd.numConstraints > 0:
        cnRows = numpy.zeros((rd.numConstraints, nESTU), dtype=float)
        cnCols = numpy.zeros((tfMatrix.shape[0], rd.numConstraints), dtype=float)
        smallEye = numpy.eye(rd.numConstraints, dtype=float)
        cnCols = ma.concatenate([cnCols, smallEye], axis=0)
        tfMatrix = ma.concatenate([tfMatrix, cnRows], axis=0)
        tfMatrix = ma.concatenate([tfMatrix, cnCols], axis=1)
    xtw = numpy.dot(xtw, tfMatrix)

    # Compute the FMPs
    fmpMatrix = robustLinearSolver(xtw, xwx, computeStats=False).params
    if rd.numConstraints > 0:
        ccMatrix = fmpMatrix[:, -rd.numConstraints:]
        fmpMatrix = fmpMatrix[:, :-rd.numConstraints]
    for (iLoc, idx) in enumerate(rd.factorIndices):
        fmpDict[factorNames[idx]] = fmpMatrix[iLoc, :]

    # Compute the leftover part corresponding to the constraints
    constraintComponent = None
    if rd.numConstraints > 0:
        constraintComponent = Struct()
        cc_X = rd.regressorMatrix[:, -rd.numConstraints:]
        ccDict = dict()
        ccXDict = dict()
        for (iLoc, idx) in enumerate(rd.factorIndices):
            ccDict[factorNames[idx]] = ccMatrix[iLoc, :]
            ccXDict[factorNames[idx]] = cc_X[iLoc, :]
        constraintComponent.ccDict = ccDict
        constraintComponent.ccXDict = ccXDict

    return fmpDict, constraintComponent

def partial_orthogonalisation(
        modelDate, data, modelDB, marketDB, orthogDict, intercept=True):
    """Does partial orthogonalisation of exposures to reduce colinearity issues
    """
    logging.debug('partial_orthogonalisation: begin')

    # Initialise necessary data
    mcaps = ma.filled(data.marketCaps, 0.0)
    estu = data.estimationUniverseIdx
    expMatrix = data.exposureMatrix.getMatrix()
    expMCopy = ma.array(expMatrix, copy=True)
    maskedData = ma.getmaskarray(expMatrix)
    X = ma.filled(ma.array(expMatrix, copy=True), 0.0)

    for (style, oList) in orthogDict.items():

        # Get particular weighting scheme
        regWgts = numpy.sqrt(mcaps)
        if oList[2]:
            regWgts = numpy.sqrt(regWgts)
        X_wgt = X * regWgts

        # Pick out relevant columns from exposure matrix
        lhsIdx = data.exposureMatrix.getFactorIndex(style)
        rhsIdx = [data.exposureMatrix.getFactorIndex(st) for st in oList[0]]
        rhs1 = numpy.transpose(numpy.take(X_wgt, rhsIdx, axis=0))

        # Regress one set of exposures against the others
        if intercept:
            rhs = numpy.concatenate((regWgts[:,numpy.newaxis], rhs1), axis=1)
        else:
            rhs = rhs1
        lhs = ma.filled(X_wgt[lhsIdx], 0.0)
        rhs_estu = numpy.take(rhs, estu, axis=0)
        lhs_estu = numpy.take(lhs, estu, axis=0)
        beta = robustLinearSolver(lhs_estu, rhs_estu, computeStats=False).params

        # Compute correlation before orthogonalisation
        if intercept:
            rhs_estu = rhs_estu[:,1:]
        tmpMat = numpy.array((lhs_estu.flatten(), rhs_estu.flatten()))
        corrBefore = numpy.corrcoef(tmpMat)

        # Reconstitute exposures
        logging.info('Orthogonalising %s %d%% relative to %s and mkt cap, Beta: %s',
                style, int(100*oList[1]), oList[0], beta)
        if intercept:
            beta[1:] = beta[1:] * oList[1]
        else:
            beta = beta * oList[1]
        lhs = ma.filled(X[lhsIdx], 0.0)
        rhs = numpy.transpose(numpy.take(X, rhsIdx, axis=0))
        if intercept:
            inter = numpy.ones((len(regWgts)), float)
            rhs = numpy.concatenate((inter[:,numpy.newaxis], rhs), axis=1)

        resid = numpy.ravel(lhs - numpy.dot(rhs, beta))

        # Quick and dirty re-scaling
        e_estu = numpy.take(resid, estu, axis=0)
        stdev = ma.sqrt(ma.inner(e_estu, e_estu) / (len(e_estu) - 1.0))

        # Compute correlation after orthogonalisation
        lhs_tmp = resid / stdev
        lhs_tmp = lhs_tmp * regWgts
        lhs_estu = numpy.take(lhs_tmp, estu, axis=0)
        tmpMat = numpy.array((lhs_estu.flatten(), rhs_estu.flatten()))
        corrAfter = numpy.corrcoef(tmpMat)
        logging.info('%s correlation before and after: %s, %s',
                style, corrBefore[0,1:], corrAfter[0,1:])

        expMatrix[lhsIdx, :] = resid / stdev
    # Remask missing values
    expMatrix = ma.masked_where(maskedData, expMatrix)
    logging.debug('partial_orthogonalisation: end')
    return

def getNextValue(data):
    if type(data) is not list:
        return data
    if len(data) < 2:
        return data[0]
    else:
        return data.pop(0)

def test_for_cointegration(rm, date, data, modelDB, marketDB, returns):
    # Get cointegration results
    cointResults = compute_cointegration_parameters(
            returns.data, data.subIssueGroups, returns.assets, data.rmgAssetMap,
            skipDifferentMarkets=False)

    # Initialise
    ktDict = dict()
    avStatsDict = dict()
    dr2Underlying = dict()
    metaType = dict()
    for typ in rm.allAssetTypes:
        #if typ in rm.drAssetTypes and (typ != 'DR'):
        #    metaType[typ] = 'AllDR'
        if typ in rm.commonStockTypes:
            metaType[typ] = 'COM'
        elif typ in rm.preferredStockTypes:
            metaType[typ] = 'PREF'
        else:
            metaType[typ] = typ
        avStatsDict[metaType[typ]] = defaultdict(list)

    for sid in data.dr2Underlying.keys():
        if data.dr2Underlying[sid] is not None:
            dr2Underlying[sid] = data.dr2Underlying[sid]

    # Output cointegration results
    outfile = 'tmp/coint-%s.csv' % date
    outfile = open(outfile, 'w')
    outfile.write('GID,SID1,Type1,Mkt1,Parent1,SID2,Type2,Mkt2,Parent2,DF-Stat,C-Value,N,\n')

    for (groupId, subIssueList) in data.subIssueGroups.items():
        for (idx1, sid1) in enumerate(subIssueList):
            type1 = data.assetTypeDict.get(sid1, None)
            mkt1 = data.marketTypeDict.get(sid1, None)
            for (idx2, sid2) in enumerate(subIssueList):
                type2 = data.assetTypeDict.get(sid2, None)
                mkt2 = data.marketTypeDict.get(sid2, None)
                # Sort out what we're comparing
                pnt1 = 0
                pnt2 = 0
                if (sid1 in dr2Underlying) and dr2Underlying[sid1] == sid2:
                    pnt2 = 1
                if (sid2 in dr2Underlying) and dr2Underlying[sid2] == sid1:
                    pnt1 = 1
                if (sid1 in data.hardCloneMap) and data.hardCloneMap[sid1] == sid2:
                    pnt2 = 1
                if (sid2 in data.hardCloneMap) and data.hardCloneMap[sid2] == sid1:
                    pnt1 = 1
                if sid1 == sid2:
                    continue
                cKey1 = '%s-%s' % (sid1.getSubIDString(), sid2.getSubIDString())
                cKey2 = '%s-%s' % (sid2.getSubIDString(), sid1.getSubIDString())
                try:
                    # Output results if not already done
                    adfStat = cointResults.dfStatDict[sid1][sid2]
                    pvalue = cointResults.dfCValueDict[sid1][sid2]
                    nobs = cointResults.nobsDict[sid1][sid2]
                    if (adfStat < 0.0) and (pvalue < 0.0):
                        if (pnt1 == 0) and (pnt2 == 0):
                            ratio = adfStat / pvalue
                            avStatsDict[metaType[type1]][metaType[type2]].append(ratio)
                            avStatsDict[metaType[type2]][metaType[type1]].append(ratio)
                    if (cKey1 not in ktDict) and (cKey2 not in ktDict):
                        outfile.write('%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%d,\n' % (groupId, \
                                sid1.getSubIDString(), type1, mkt1, pnt1,
                                sid2.getSubIDString(), type2, mkt2, pnt2,
                                adfStat, pvalue, nobs))
                        ktDict[cKey1] = True
                        ktDict[cKey2] = True
                except KeyError:
                    continue
    outfile.close()

    # Output summary stats
    metaVals = list(set(metaType.values()))
    outfile = 'tmp/coint-summ-%s.csv' % date
    outfile2 = 'tmp/coint-n-%s.csv' % date
    outfile = open(outfile, 'w')
    outfile2 = open(outfile2, 'w')
    outfile.write(',')
    outfile2.write(',')
    for typ in metaVals:
        outfile.write(',%s' % typ)
        outfile2.write(',%s' % typ)
    outfile.write(',\n')
    outfile2.write(',\n')
    for (idx, aType) in enumerate(metaVals):
        if aType in avStatsDict:
            outfile.write('%s,%s,' % (date.year, aType))
            outfile2.write('%s,%s,' % (date.year, aType))
            subDict = avStatsDict[aType]
            for bType in metaVals:
                if bType in subDict:
                    ln = len(subDict[bType])
                    summary = ma.median(subDict[bType], axis=None)
                    outfile.write('%.4f,' % summary)
                    outfile2.write('%d,' % ln)
                else:
                    outfile.write(',')
                    outfile2.write(',')
            outfile.write('\n')
            outfile2.write('\n')
    outfile.close()
    outfile2.close()

    # Output DR cointegration stats
    outfile = 'tmp/coint-dr-%s.csv' % date
    outfile = open(outfile, 'w')
    outfile.write('Type,N,MedianRatio,MeanRatio,\n')
    drStatDict = defaultdict(list)

    for (groupId, subIssueList) in data.subIssueGroups.items():
        for sid1 in subIssueList:
            if sid1 in dr2Underlying:
                sid2 = dr2Underlying[sid1]
                type_ = data.assetTypeDict.get(sid2, None)
                if type_ in metaType:
                    found = False
                    if sid1 in cointResults.dfStatDict:
                        subDict = cointResults.dfStatDict[sid1]
                        if sid2 in subDict:
                            adfStat = cointResults.dfStatDict[sid1][sid2]
                            pvalue = cointResults.dfCValueDict[sid1][sid2]
                            found = True
                    elif sid2 in cointResults.dfStatDict:
                        subDict = cointResults.dfStatDict[sid2]
                        if sid1 in subDict:
                            adfStat = cointResults.dfStatDict[sid2][sid1]
                            pvalue = cointResults.dfCValueDict[sid2][sid1]
                            found = True
                    if found:
                        ratio = adfStat / pvalue
                    else:
                        ratio = 0.0
                    drStatDict[metaType[type_]].append(ratio)
    for (idx, aType) in enumerate(metaVals):
        if aType in drStatDict:
            ln = len(drStatDict[aType])
            summary = ma.median(drStatDict[aType], axis=None)
            summ1 = ma.average(drStatDict[aType], axis=None)
            outfile.write('%s,%d,%.4f,%.4f,\n' % (aType, ln, summary, summ1))
    outfile.close()

    return

def p2_rnd_int(x, p):
    if x is ma.masked:
        return x
    if x > 0.0:
        x = float(math.floor((x * p) + 0.5))/p
    else:
        x = float(math.ceil((x * p) - 0.5))/p
    return x

def p2_round(x, dp=0):
    # Python 2 rounding - mostly to be deleted later
    p = 10**dp
    if numpy.ndim(x) > 1:
        x = ma.where(abs(x)<10.0/p, 0.0, x)
        # Scrap this in time
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if x[i,j] is not ma.masked:
                    x[i,j] = p2_rnd_int(x[i,j], p)
        x = ma.where(abs(x)<10.0/p, 0.0, x)
    elif numpy.ndim(x) == 1:
        x = ma.where(abs(x)<10.0/p, 0.0, x)
        for i in range(x.shape[0]):
            if x[i] is not ma.masked:
                x[i] = p2_rnd_int(x[i], p)
        x = ma.where(abs(x)<10.0/p, 0.0, x)
    else:
        if abs(x)<10.0/p:
            x = 0.0
        else:
            x = p2_rnd_int(x, p)
    return x
