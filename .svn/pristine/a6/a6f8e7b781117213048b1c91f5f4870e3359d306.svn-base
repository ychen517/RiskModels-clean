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
from calendar import monthrange
import numpy as np
import pandas
import gzip
import statsmodels.api as sm

def oldPandasVersion():
    pdVer = float('.'.join(pandas.__version__.split('.')[:2]))
    if pdVer < 0.23:
        print ('WARNING - Using old version of pandas:', pdVer)
        return True
    return False

def nullspace(A, atol=1e-13, rtol=0):
    """Compute an approximate basis for the nullspace of A.

    The algorithm used by this function is based on the singular value
    decomposition of `A`.

    A : ndarray
        A should be at most 2-D.  A 1-D array with length k will be treated
        as a 2-D with shape (1, k)
    atol : float
        The absolute tolerance for a zero singular value.  Singular values
        smaller than `atol` are considered to be zero.
    rtol : float
        The relative tolerance.  Singular values less than rtol*smax are
        considered to be zero, where smax is the largest singular value.

    If both `atol` and `rtol` are positive, the combined tolerance is the
    maximum of the two; that is::
        tol = max(atol, rtol * smax)
    Singular values smaller than `tol` are considered to be zero.

    ns : ndarray
        If `A` is an array with shape (m, k), then `ns` will be an array
        with shape (k, n), where n is the estimated dimension of the
        nullspace of `A`.  The columns of `ns` are a basis for the
        nullspace; each element in numpy.dot(A, ns) will be approximately
        zero.
    """

    A = np.atleast_2d(A)
    u, s, vh = np.linalg.svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns

def readMap(key, rmgMap, fill=None):
    if key in rmgMap:
        return rmgMap[key]
    elif hasattr(key, 'rmg_id'):
        return rmgMap.get(key.rmg_id, fill)
    return fill

class ConstrainedLinearModel(object):
    """ y = X f + e
         C f = 0
    """
    def __init__(self, Y, X, C=None, weights=None, huberT=None, criteria=np.sqrt):
        Y = Y.reindex(index=X.index, copy=False)
        self.Y = Y
        self.X = X
        self.huberT = huberT
        self._criteria = criteria
        
        if np.any(np.any(pandas.isnull(self.Y),axis=0),axis=0):
            raise Exception('Y variable contains missing values')
        if np.any(np.any(pandas.isnull(self.X),axis=0),axis=0):
            raise Exception('X variable contains missing values')
        self.C = C
        if self.C is not None:
            self.C = C.reindex(columns=X.columns, copy=False)
        self.weights = pandas.Series(1.0, index=X.index)
        if weights is not None:
            self.weights = weights.reindex(index=X.index)
        self.weights /= self.weights.sum()
        if np.any(pandas.isnull(self.weights)):
            raise Exception('Weights are missing values')
        self._result = None
        
    def __compute(self):
        if isinstance(self.Y, pandas.DataFrame):
            raise Exception('matrix left-hand side currently not supported')
        y = self.Y.copy()
        if self.C is not None:
            N = pandas.DataFrame(nullspace(self.C.values),index=self.X.columns)
            Xbar = self.X.dot(N)
        else:
            N = pandas.DataFrame(np.eye(self.X.shape[1]), index=self.X.columns, columns=self.X.columns)
            Xbar = self.X.copy()
        self.adjweights = self.weights.copy()
        wy = y * self._criteria(self.weights)
        if self.huberT is not None:
            wXbar = Xbar.mul(self._criteria(self.weights), axis=0)
            tmpresult = sm.RLM(wy, wXbar, sm.robust.norms.HuberT(t=self.huberT), missing='drop').fit(maxiter=200, tol=1e-10)
            self.adjweights = self.adjweights*tmpresult.weights
            self._robustweights = tmpresult.weights
        else:
            self._robustweights = pandas.Series(1., index=self.weights.index)
        model = sm.WLS(y, Xbar, weights=self.adjweights, missing='drop')
        result = model.fit(maxiter=500, tol=1e-10)
        fcovbar = result.cov_params() 
        self._params = N.dot(result.params)
        self._result = result
        self._model = model
        self._fcov = N.dot(fcovbar).dot(N.T)
        #self._bse = np.sqrt(np.diag(self._fcov))
        self._bse = pandas.Series(np.sqrt(np.diag(self._fcov)), index=self._fcov.index)
        self._tvalues = self._params/self._bse 
        
    @property
    def robustweights(self):
        if self._result is None:
            self.__compute()
        return self._robustweights

    @property
    def bse(self):
        if self._result is None:
            self.__compute()
        return self._bse

    @property
    def params(self):
        if self._result is None:
            self.__compute()
        return self._params

    @property
    def tvalues(self):
        if self._result is None:
            self.__compute()
        return self._tvalues
    
    @property
    def rsquared(self):
        if self._result is None:
            self.__compute()
        if self._result.k_constant:
            return 1 - (self._ssr/self._centered_tss)
        else:
            return 1 - (self._ssr/self._uncentered_tss)
    
    @property
    def rsquared_adj(self):
        if self._result is None:
            self.__compute()
        return 1 - np.divide(self._result.nobs - self._result.k_constant, self._result.df_resid) * (1 - self.rsquared)

    @property
    def fmps(self):
        if self._result is None:
            self.__compute()
        return self._fmp

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

def df2ma(df):
    return ma.array(df.values.copy(), mask=pandas.isnull(df).values)

def dfDtConv(df, columns=True):
    if type(df) is pandas.Series:
        return [x.to_pydatetime().date() for x in pandas.to_datetime(df.index)]
    if columns:
        return [x.to_pydatetime().date() for x in pandas.to_datetime(df.columns)]
    return [x.to_pydatetime().date() for x in pandas.to_datetime(df.index)]

def flip_dict_of_lists(in_dict):
    '''
        flip the dictionary. for dict(itemA,list of itemB)
    :param in_dict: dict(itemA,list of itemB)
    :return: dict(itemB,itemA)
    '''
    return {y: x for x, z in in_dict.items() for y in z}

def flip_dict_of_lists_m2m(in_dict):
    '''
    '''
    retDict = defaultdict(list)
    for x, z in in_dict.items():
        for y in z:
            retDict[y].append(x)
    return retDict

def noneToVal(x, val):
    """Returns val if x is None and x otherwise.
    """
    if x == None:
        return val
    return x

def noneToValTuple(x, val):
    """Returns val if x is None and x otherwise.
    """
    if x == (None,None):
        return val
    return x

def writeToCSV(dataIn, filepath, columnNames=None, rowNames=None, delim=',', dp=12):
    """ Writes array of data to a csv file
    """
    data = screen_data(ma.array(dataIn, copy=True))
    if len(data.shape) == 1:
        data = data[:,numpy.newaxis]
    data = p2_round(data, dp)
    writeHeader = False
    if columnNames is not None:
        columnNames = [str(cn).replace(delim,'') for cn in columnNames]
        writeHeader = True
    outDF = pandas.DataFrame(data, index=rowNames, columns=columnNames)
    outDF.to_csv(filepath, sep=delim, columns=columnNames, float_format='%%.%df' % dp,
            header=writeHeader, line_terminator=',\n')
    return

def is_binary_data(data):
    """Checks if an array (not matrix) of data is binary.
    ie, contains only 2 unique values, besides masked entries.
    """
    assert(len(data.shape)==1)
    goodData = ma.take(data, numpy.flatnonzero(ma.getmaskarray(data)==0), axis=0)
    freq = numpy.unique(numpy.array(goodData))
    if len(freq) < 3:
        return True
    else:
        return False

def encryptFile(date, inputFileName, targetDir, delim='|', destFileName=None):
    """
        given a file, it creates an encrypted version of the same file in targetDir
    """
    import tempfile
    from riskmodels.wombat import scrambleString
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

def isCashAsset(asset):
    if hasattr(asset, 'isCashAsset'):
        return asset.isCashAsset()
    elif isinstance(asset, str):
        return asset.startswith('CSH_') and asset.endswith('__')
    return False

def change_date_frequency(dailyDateList, frequency='weekly'):
    """Converts a list of daily dates to either weekly (default) or monthly
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

def get_period_dates(modelDB, dailyDateList, frequency='weekly'):
    """Converts a list of daily dates to either weekly (default) or monthly
    Uses the same date convention as daily returns - i.e. ret(T) = price(T)/price(T-1)-1
    rather than the function above which assumes ret(T-1) = price(T)/price(T-1)-1
    """
    if frequency == 'weekly':
        fullDateList = modelDB.getDateRange(None, min(dailyDateList), max(dailyDateList), excludeWeekend=True)
        periodDateList = [d for d in fullDateList if d.weekday() == 4]
    elif frequency == 'monthly':
        fullDateList = modelDB.getDateRange(None, min(dailyDateList), max(dailyDateList))
        periodDateList = sorted(list(set([
            datetime.date(d.year, d.month, monthrange(d.year, d.month)[1]) for d in fullDateList
            if datetime.date(d.year, d.month, monthrange(d.year, d.month)[1]) <= max(dailyDateList)])))
    elif frequency == 'rolling':
        fullDateList = modelDB.getDateRange(None, min(dailyDateList), max(dailyDateList), excludeWeekend=True)
        periodDateList = [d for d in fullDateList if d.weekday() == fullDateList[-1].weekday()]
    else:
        return dailyDateList
    periodDateList.sort()
    return periodDateList

def screen_data(dataIn, fill=False, fillValue=0.0):
    # Tool to screen a data array for undesirable types
    # such as NaN, inf etc.
    if (type(dataIn) is pandas.DataFrame) or (type(dataIn) is pandas.Series):
        data = df2ma(dataIn)
    else:
        data = ma.array(dataIn, copy=True)

    # Take note of masked data
    maskedData = numpy.array(ma.getmaskarray(data), copy=True)

    # Now look for NaNs/Infs and similar
    data = ma.filled(data, fillValue)
    badData = numpy.isfinite(data)==0
    data = ma.where(badData, fillValue, data)

    if not fill:
        # Mask bad and missing data again
        maskedData = ((maskedData==0) * (badData==0))==0
        data = ma.masked_where(maskedData, data)

    if type(dataIn) is pandas.DataFrame:
        return pandas.DataFrame(data, index=dataIn.index, columns=dataIn.columns)
    if type(dataIn) is pandas.Series:
        return pandas.Series(data, index=dataIn.index)
    return data

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

def clip_extrema(data, pct=0.05):
    linData = numpy.ravel(ma.filled(data, 0.0))
    sortindex = ma.argsort(linData)
    N = len(linData)
    C = min(100, int(round(N*pct)))
    upperBound = linData[sortindex[N-C-1]]
    lowerBound = linData[sortindex[C]]
    return ma.clip(data, lowerBound, upperBound)

def clip_extremaDF(data, pct=0.05):
    linData = numpy.ravel(data.fillna(0.0).values)
    sortindex = numpy.argsort(linData)
    N = len(linData)
    C = min(100, int(round(N*pct)))
    upperBound = linData[sortindex[N-C-1]]
    lowerBound = linData[sortindex[C]]
    return data.clip(lowerBound, upperBound)

def symmetric_clip(data, upperBound=1000.0):
    upperBound = numpy.log(1.0 + upperBound) - 1.0
    lowerBound = -upperBound
    tmpData = numpy.log(1.0 + data) - 1.0
    tmpData = numpy.clip(tmpData, -upperBound, upperBound)
    tmpData = numpy.exp(tmpData + 1.0) - 1.0
    return tmpData

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

def load_ipo_dates(date, universe, modelDB, marketDB, returnList=False, exSpacAdjust=False):
    """Return a series of IPO (from) dates for the given subissues
    If exSpacAdjust is true, then for assets that once were SPACs,
    the ipo date is changed to the SPAC's eff-date (i.e. after merging was effective).
    """

    # Load the regular from dates
    fromDates = modelDB.loadIssueFromDates([date],  universe)
    ipoDates = pandas.Series(fromDates, index=universe)
    if not exSpacAdjust:
        if returnList:
            return fromDates
        return ipoDates

    # Find dates where SPACs converted to regular assets
    # Load in the history of asset type mappings
    spacHist = defaultdict(dict)
    import riskmodels.AssetProcessor_V4 as ap
    assetTypeRange = ap.get_asset_info_range(\
            date, universe, modelDB, marketDB, 'ASSET TYPES', 'Axioma Asset Type')

    # Load round sub-issues and determine SPAC-status or not
    for sid in set(assetTypeRange.keys()).intersection(set(universe)):
        assetTypes = set(assetTypeRange[sid].values())
        spacs = assetTypes.intersection(ap.spacTypes)

        # Check whether asset is or ever has been a SPAC
        if len(spacs) > 0:

            # Identify assets that have changed from SPAC to non-SPAC
            nonSpacs = assetTypes.difference(ap.spacTypes)
            if len(nonSpacs) > 0:
                spacDts = []
                nonSpacDts = []

                # Separate SPAC and non-SPAC dates
                for dt, itm in assetTypeRange[sid].items():
                    if dt > date:
                        continue
                    if itm in ap.spacTypes:
                        spacDts.append(dt)
                    else:
                        nonSpacDts.append(dt)

                # Find most recent SPAC date
                if len(spacDts) > 0:
                    maxSpacDt = max(spacDts)

                    # Find earliest non-SPAC date that is more recent than the latest SPAC date
                    nonSpacDts = [dt for dt in nonSpacDts if dt>maxSpacDt]
                    if len(nonSpacDts) > 0:
                        spacHist[sid][min(nonSpacDts)] = 1

    # Convert dict to a dataframe
    if len(spacHist) > 0:
        spacHist = pandas.DataFrame(spacHist).T
        spacIDs = sorted(spacHist.index)
        spacDates = sorted(spacHist.columns)
        spacHist = spacHist.loc[spacIDs, spacDates]

        # Drop anything that is still a SPAC as of current date
        notSPACNow = spacHist.fillna(method='ffill', axis=1).loc[:, spacDates[-1]].dropna().index
        spacHist = spacHist.loc[notSPACNow, :]
        #spacHist.to_csv('tmp/spacHist.csv')

        # If ex-SPAC date is later than IPO date, replace the latter with the former
        changes = 0
        for sid in spacHist.index:
            nonSpacDt = max(spacHist.loc[sid, :].dropna().index)
            if nonSpacDt > ipoDates[sid]:
                logging.debug('Changing IPO date of %s from %s to %s due to change from SPAC',
                        sid if isinstance(sid, str) else sid.getSubIDString(), ipoDates[sid], nonSpacDt)
                ipoDates[sid] = nonSpacDt
                changes += 1
        logging.info('Changing IPO date of %d assets due to change from SPAC', changes)

    # Finally, load announcement dates and overwrite the IPO dates with these
    changes = 0
    annDateMap = modelDB.getSPACAnnounceDate(date, universe, marketDB)
    for (sid, dt) in annDateMap.items():
        if ipoDates[sid] != dt:
            logging.debug('Changing IPO date of %s from %s to %s due to SPAC announcement',
                    sid if isinstance(sid, str) else sid.getSubIDString(), ipoDates[sid], dt)
            ipoDates[sid] = dt
            changes += 1
    logging.info('Changing IPO date of %d assets due to SPAC announcement', changes)

    if returnList:
        return list(ipoDates.values)
    return ipoDates

def computeExponentialWeights(halfLife, length, equalWeightFlag=False, normalize=True):
    """Return an array of exponential weights given a half-life of the given value,
    halfLife, and that sum to 1.
    The most recent observation is assumed to be first and given the greatest weight.
    If equalWeightFlag is enabled, returns a vector of equal weights.
    """
    if equalWeightFlag:
        w = numpy.ones(length, float)
    else:
        w = 2.**(numpy.arange(0,-length,-1)/float(halfLife))
    if normalize:
        w /= sum(w)
    return w

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

def compute_NWAdj_covariance(dataIn, nwLag, weights=None, deMean=True,
                    axis=0, corrOnly=False, varsOnly=False):
    """Outer wrapper for compute_covariance that has the NW adjustment
    baked in
    """
    if type(dataIn) is pandas.DataFrame:
        data = df2ma(dataIn);
    else:
        data = ma.array(dataIn, copy=True)

    covMatrix = compute_covariance(data, weights=weights, deMean=deMean,
            axis=axis, corrOnly=False, varsOnly=varsOnly)

    for lag in range(1,nwLag+1):
        adj = compute_covariance(data, lag=lag, weights=weights, deMean=deMean,
                axis=axis, corrOnly=False, varsOnly=varsOnly)
        if varsOnly:
            covMatrix += (1.0 - float(lag)/(float(nwLag)+1.0)) * (2.0 * adj)
        else:
            covMatrix += (1.0 - float(lag)/(float(nwLag)+1.0)) * (adj + numpy.transpose(adj))

    if corrOnly:
        return cov2corr(covMatrix, fill=True)[1]
    return covMatrix

def cov2corr(covMatrix, fill=False, returnCov=False, tol=1.0e-12, conditioning=False):
    """Trivial routine to convert a covariance matrix to a correlation matrix,
    but one which is used quite a bit so this will hopefully standardise treatment
    Also returns original cov matrix with the problem rows and columns masked
    """
    covMatrix = screen_data(covMatrix)
    if conditioning:
        # Perform some conditioning of the covariance matrix
        (d, v) = linalg.eigh(covMatrix)
        condNo = d / max(d)
        d = ma.masked_where(condNo<tol, d)
        covMatrix = ma.dot((v*d), ma.transpose(v)).filled(0.0)

    varVector = ma.diag(covMatrix)
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
        corMatrix = ma.transpose(ma.transpose(corMatrix * stdVector) * stdVector)
    return (stdVector, corMatrix)

def forcePositiveSemiDefiniteMatrix(matrix, min_eigenvalue=1e-10, quiet=False):
    """Checks whether a given matrix is positive semidefinite.
    If not, issue a warning and tweak it accordingly.
    """
    returnDF = False
    if type(matrix) is pandas.DataFrame:
        returnDF = True
        idx = matrix.index
        cols = matrix.columns
        matrix = matrix.fillna(0.0).values

    # Process matrix first
    matrix = ma.filled(matrix, 0.0)
    matrix = (matrix + numpy.transpose(matrix)) / 2.0

    (eigval, eigvec) = linalg.eigh(matrix)
    check = ma.getmaskarray(ma.masked_where(eigval < min_eigenvalue, eigval))
    if min(eigval) <= 0.0:
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
    if returnDF:
        return pandas.DataFrame(matrix, index=idx, columns=cols)
    return matrix

def generalizedLeastSquares(y, x, omegaInv, deMean=False, allowPseudo=False, implicit=False):
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

def robustLinearSolver(yIn, XIn, robust=False, k=1.345, maxiter=50,
        tol=1.0e-6, weights=None, computeStats=True):
    """ Estimates y = Xb + e via either least-squares regression, or robust regression
    If robust, assumes robust regression is performed one column of y at a time
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
    rsquarecentered = numpy.zeros((y.shape[1]), float)
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
                denom = ma.inner(yyy, yyy)
                if denom > 0.0:
                    r2 = float(ma.inner(err, err)) / denom
                    rsquare[j] = 1.0 - r2
                else:
                    rsquare[j] = 0.0
                yyycentered = yyy -  ma.mean(yyy)
                denom = ma.inner(yyycentered, yyycentered)
                if denom > 0.0:
                    r2centered = float(ma.inner(err, err)) /  denom
                    rsquarecentered[j] = 1.0 - r2centered
                else:
                    rsquarecentered[j] = 0.0
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
                            yWtcentered = yWt - ma.mean(yWt)
                            r2centered = float(ma.inner(eWt, eWt)) / ma.inner(yWtcentered, yWtcentered) 
                            rsquarecentered[jDim] = 1.0 - r2centered
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
            results.rsquarecentered = rsquarecentered
    else:
        results.params = b
        results.error = e
        results.y_hat = y_hat
        if computeStats:
            results.tstat = ts
            results.pvals = pvals
            results.rsquare = rsquare
            results.rsquarecentered = rsquarecentered

    if testMode:
        logging.info('Solution: %s', results.params)
        logging.info('Error: %s', results.error)
        if computeStats:
            logging.info('Average R-Square: %s', ma.average(results.rsquare, axis=None))
            logging.info('Average R-Square Centered: %s', ma.average(results.rsquarecentered, axis=None))
        exit(0)

    return results

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

def robustAverage(v, wtList=[], k=1.345, maxiter=500):
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
    if len(wtList) != n:
        wt = numpy.ones((n), float)
    else:
        wt = numpy.array(wtList, float)
     
    # Flag missing returns and weights
    badRetIdx = numpy.flatnonzero(ma.getmaskarray(v_part))
    zeroWgt = ma.masked_where(wt <= 0.0, wt)
    badWgtIdx = numpy.flatnonzero(ma.getmaskarray(zeroWgt))
    badIdx = set(list(badRetIdx) + list(badWgtIdx))
    # Take subset of observations corresponding to non-missing returns and non-zero weights
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

        # Do the regression
        model = sm.RLM(v_part, inter, M = sm.robust.norms.HuberT(t=k))
        results = model.fit(maxiter=maxiter)
        if hasattr(results, 'weights'):
            downWeights = numpy.array(results.weights)
            numpy.put(rlmDownWeights, okIdx, downWeights)
        avVal = results.params[0]
    avVal = screen_data(avVal)
    return (avVal, rlmDownWeights)

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

def computeRegressionStdError(resid, exposureMatrix, weights=None, constraintMatrix=None, white=False):
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
        blob = numpy.dot(constraintMatrix, numpy.dot(xwxinv, numpy.transpose(constraintMatrix)))
        blob = linalg.inv(blob)
        bigBlob = numpy.dot(numpy.transpose(constraintMatrix), 
                            numpy.dot(blob, constraintMatrix))
        biggerBlob = numpy.dot(bigBlob, xwxinv)
        if not white:
            var = (s * xwxinv) - (s * numpy.dot(xwxinv, biggerBlob))
        else:
            var = omega - numpy.dot(omega, biggerBlob) - \
                  numpy.dot(numpy.transpose(biggerBlob), omega) + \
                  numpy.dot(numpy.transpose(biggerBlob), 
                            numpy.dot(omega, biggerBlob))
        stdError = numpy.sqrt(numpy.diag(var))

    return stdError

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

def partial_orthogonalisation(
        modelDate, assetData, expM, modelDB, marketDB, orthogDict, intercept=True):
    """Does partial orthogonalisation of exposures to reduce colinearity issues
    """
    logging.debug('partial_orthogonalisation: begin')

    # Initialise necessary data
    estu = assetData.estimationUniverse
    expM_DF = expM.toDataFrame()
    mask = expM_DF.isnull()
    X = expM.toDataFrame().copy(deep=True).fillna(0.0)

    for (style, oList) in orthogDict.items():

        # Unpack parameters
        styleList = oList[0]
        dummy = oList[1]
        sqrtFlag = oList[2]

        # Weight exposures
        regWgts = numpy.sqrt(assetData.marketCaps.fillna(0.0))
        regWgts.name = 'intercept'
        if sqrtFlag:
            regWgts = numpy.sqrt(regWgts)
        X_wgt = X.multiply(regWgts, axis=0)

        # Set up regression matrices
        rhs = X_wgt.loc[:, styleList]
        if intercept:
            regWgts.name = 'intercept'
            rhs = pandas.concat([regWgts, rhs], axis=1)
        lhs = X_wgt.loc[:, style].fillna(0.0)

        # Compute correlation before orthogonalisation - only makes sense if doing a 1D orthogonalisation
        tmpMat = numpy.array((lhs[estu].values.flatten(), rhs.loc[estu, styleList].values.flatten()))
        corrBefore = numpy.corrcoef(tmpMat)

        # Regress one set of exposures against the others
        beta = robustLinearSolver(lhs[estu].values, rhs.loc[estu, :].values, computeStats=False).params
        beta = pandas.Series(beta, index=rhs.columns)

        # Reconstitute exposures
        logging.info('Orthogonalising %s relative to mkt cap and %s, Beta: %s', style, styleList, beta.values)
        rhs = X.loc[:, styleList]
        if intercept:
            inter = pandas.Series(1.0, index=X.index, name='intercept')
            rhs = pandas.concat([inter, rhs], axis=1)
        resid = X.loc[:, style] - rhs.dot(beta)

        # Quick and dirty re-scaling
        stdev = ma.sqrt(resid[estu].dot(resid[estu]) / (len(estu) - 1.0))
        resid = resid / stdev

        # Compute correlation after orthogonalisation
        lhs = resid * regWgts
        tmpMat = numpy.array((lhs[estu].values.flatten(), rhs.loc[estu, styleList].values.flatten()))
        corrAfter = numpy.corrcoef(tmpMat)
        logging.info('%s correlation before and after: %s, %s', style, corrBefore[0,1:], corrAfter[0,1:])

        # Overwrite exposures
        expM_DF.loc[:, style] = resid

    # Remask missing values
    expM_DF = expM_DF.mask(mask)
    expM.data_ = df2ma(expM_DF.T)
    logging.debug('partial_orthogonalisation: end')

    return expM

def buildGICSExposures(subIssues, date, modelDB, level='Sectors', clsDate=datetime.date(2016,9,1),
                returnNames=False, returnDF=False):
    """Build exposure matrix of GICS industry groups or sectors
    Note - returns transposed (i.e. correct) matrix of N assets by P factors
    """
    expMatrix, gicsNames = modelDB.getGICSExposures(date, subIssues, level, clsDate)
    if returnNames:
        return expMatrix, gicsNames
    if returnDF:
        return pandas.DataFrame(expMatrix, index=subIssues, columns=gicsNames)
    return expMatrix

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
    debugging = False
    vector = False
    if len(dataArray.shape) < 2:
        dataArrayCopy = ma.array(dataArray[:,numpy.newaxis], copy=True)
        vector = True
    else:
        dataArrayCopy = ma.array(dataArray, copy=True)
    dataEstimates = numpy.zeros((dataArrayCopy.shape), float)
    if hasattr(data, 'assetIdxMap'):
        assetIdxMap = data.assetIdxMap
    else:
        assetIdxMap = dict(zip(data.universe, range(len(data.universe))))

    # Get sector/industry group exposures
    if industryGroupFactor:
        level = 'Industry Groups'
    else:
        level = 'Sectors'
    allExposures, expNames = buildGICSExposures(
            data.universe, date, modelDB, level=level, clsDate=gicsDate, returnNames=True)
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
        estuIdx = [assetIdxMap[sid] for sid in estu]
        for idx in estuIdx:
            regWeights[idx] = ma.sqrt(marketCaps[idx])

    # Loop round countries/regions
    for regID in regionIDMap.keys():

        # Get relevant assets and data
        rmgAssets = [sid for sid in regionAssetMap[regID]]
        if len(rmgAssets) < 1:
            logging.warning('No assets for %s', regionIDMap[regID])
            continue
        rmgAssetsIdx = [assetIdxMap[sid] for sid in rmgAssets]
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
        if numpy.all(numpy.isfinite(lnCapReg)):
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

        if not enoughData:
            logging.error('Completely unable to find enough good data for region: %s, skipping...', regionIDMap[regID])
            continue

        if debugging:
            idList = [sid.getSubIDString() for sid in rmgAssets]
            estuVec = numpy.zeros((len(idList)), float)
            numpy.put(estuVec, goodDataIdx, 1.0)
            tmpExpos = numpy.concatenate((exposures, estuVec[:,numpy.newaxis]), axis=1)
            facNames = expNames + ['lnCap', 'estu']
            dFile = 'tmp/proxy-expos-%s-%s.csv' % (regID, date)
            writeToCSV(tmpExpos, dFile, rowNames=idList, columnNames=facNames, dp=8)

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

        if debugging:
            idList = [sid.getSubIDString() for sid in rmgAssets]
            facNames = expNames + ['lnCap']
            dFile = 'tmp/proxy-frets-%s-%s.csv' % (regID, date)
            writeToCSV(factorReturns, dFile, rowNames=[facNames[idx] for idx in okExpIdx], dp=8)

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
        dFile = 'tmp/proxy-data-raw-%s.csv' % date
        writeToCSV(dataArray, dFile, rowNames=idList, columnNames=dtList, dp=8)
        dFile = 'tmp/proxy-data-estm-%s.csv' % date
        writeToCSV(dataEstimates, dFile, rowNames=idList, columnNames=dtList, dp=8)

    logging.debug('proxyMissingAssetData: end')
    if vector:
        return dataArrayCopy[:,0], dataEstimates[:,0]
    else:
        return dataArrayCopy, dataEstimates

def proxyMissingAssetDataV4(rmgList, data, assetData, modelDB, outlierClass,
                            estu=None, countryFactor=True, industryGroupFactor=False,
                            debugging=False, robust=False, gicsDate=datetime.date(2018,9,29),
                            minGoodAssets=0.1, pctGoodReturns=0.95, forceRun=False):
    """Given a dataframe of N by T asset data with masked values, fill in missing values using proxies.
    Assets are modelled using countries/regions, GICS Sectors/industry groups and log(cap) as factors.
    """
    logging.debug('proxyMissingAssetData: begin')

    # Initialise
    extraDebugging = False
    dataCopy = data.copy(deep=True)
    dataEstimates = pandas.DataFrame(0.0, index=dataCopy.index, columns=dataCopy.columns)
    date = max(dataCopy.columns)
    if estu is None:
        estu = set(assetData.universe)
    level = 'Sectors'
    if industryGroupFactor:
        level = 'Industry Groups'

    # Build GICS exposure matrix
    gicsExposures = buildGICSExposures(
            assetData.universe, date, modelDB, level=level, clsDate=gicsDate, returnDF=True)

    # Bucket assets into regions/countries
    regionAssetMap = defaultdict(set)
    for rmg in rmgList:
        rmg_assets = set(assetData.universe).intersection(assetData.rmgAssetMap.get(rmg, set()))
        if countryFactor or (len(rmgList) < 2):
            regionAssetMap[rmg] = rmg_assets
        else:
            region = modelDB.getRiskModelRegion(rmg.region_id)
            regionAssetMap[region] = regionAssetMap[region].union(rmg_assets)

    # Create lncap column and regression weights
    marketCaps = assetData.marketCaps.fillna(0.0)
    bounds = prctile(marketCaps.values, [5.0, 95.0])
    marketCaps = numpy.clip(marketCaps, bounds[0], bounds[1])
    lnCap = numpy.log(marketCaps+1.0).rename('lnCap')
    regWeights = numpy.sqrt(marketCaps[estu]).reindex(assetData.universe)

    # Loop round countries/regions
    for region in regionAssetMap.keys():

        rmgAssets = sorted(regionAssetMap[region])
        # Skip if empty
        if len(rmgAssets) < 1:
            logging.warning('No assets for %s', region.description)
            continue
        rmgEstu = sorted(estu.intersection(set(rmgAssets)))
        if len(rmgEstu) < 1:
            logging.warning('No estu assets for %s', region.description)
            continue

        # Initialise
        rmgExposures = gicsExposures.loc[rmgAssets, :]
        rmgData = dataCopy.loc[rmgAssets, :]
        rmgWeights = regWeights[rmgAssets]

        # Quick and dirty standardisation of lncap column
        minBound = numpy.min(lnCap[rmgEstu], axis=None)
        maxBound = numpy.max(lnCap[rmgEstu], axis=None)
        rmgLnCap = numpy.clip(lnCap[rmgAssets], minBound, maxBound)
        meanWt = numpy.average(rmgLnCap, axis=None, weights=(rmgWeights*rmgWeights))
        rmgLnCap = (rmgLnCap - meanWt) / numpy.std(lnCap[rmgEstu])
        if numpy.all(numpy.isfinite(rmgLnCap)):
            rmgExposures = pandas.concat([rmgExposures, rmgLnCap], axis=1)

        # Sort data into good and bad
        numOkData = numpy.isfinite(rmgData).sum(axis=1)
        numOkData = numOkData / float(numpy.max(numOkData, axis=None))
        logging.info('Filling %s %s missing values on the fly',
                rmgData.isnull().values.sum(axis=None), region.description)

        # Check that a reasonable proportion of assets have enough returns
        notEnoughData = True
        runningTol = pctGoodReturns
        while notEnoughData and (runningTol > 0.01):
            goodDataIdx = numOkData[numOkData.mask(numOkData>runningTol).isnull()].index
            if len(goodDataIdx) < minGoodAssets * len(rmgAssets):
                logging.error('Not enough assets (%d/%d) with good enough data to fit model',
                        len(goodDataIdx), len(rmgAssets))
                logging.info('Lowering tolerance from %.3f to %.3f', runningTol, 0.9*runningTol)
                runningTol *= 0.9
            else:
                notEnoughData = False
        # Skip if insufficient "good" data
        if notEnoughData:
            logging.error('Completely unable to find enough good data for region: %s, skipping...', region.description)
            continue

        # Do some checks on weights
        weights = rmgWeights[goodDataIdx]
        if numpy.sum(numpy.isfinite(weights), axis=None) < minGoodAssets * len(rmgAssets):
            logging.warning('No assets with good enough data and non-missing weights')
            continue
        sumWeight = weights.fillna(0.0).sum(axis=None)
        if sumWeight <= 0.0:
            logging.warning('No non-zero weights for %s', region.description)
            continue
        weights /= sumWeight

        # Check on columns with all missing exposures
        goodExposures = rmgExposures.loc[goodDataIdx, :]
        okExpCols = sorted(goodExposures.mask(goodExposures==0.0).dropna(how='all', axis=1).columns)
        if len(okExpCols) < 1:
            logging.warning('No non-zero exposures for %s', region.description)
            continue
        goodExposures = goodExposures.loc[: ,okExpCols].fillna(0.0)

        # Make sure that the regression is well-enough conditioned
        xwx = goodExposures.mul(weights, axis=0).T.dot(goodExposures.mul(weights, axis=0))
        (eigval, eigvec) = linalg.eigh(xwx)
        condNum = numpy.inf
        if min(eigval) > 0.0:
            condNum = max(eigval) / min(eigval)
        if debugging:
            logging.info('Regressor (%i by %i) has condition number %f', xwx.shape[0], xwx.shape[1], condNum)
        if abs(condNum) > 1.0e6:
            logging.warning('Condition number %f too high: aborting for %s', condNum, region.description)
            continue

        # Trim returns if required
        goodData = rmgData.loc[goodDataIdx, :].fillna(0.0)
        if outlierClass is not None:
            trimmedData = outlierClass.twodMAD(goodData.values)
            goodData = pandas.DataFrame(trimmedData, index=goodData.index, columns=goodData.columns)

        # Perform the regression
        if goodExposures.shape[0] < 2*goodExposures.shape[1]:
            factorReturns = robustLinearSolver(\
                    goodData.values, goodExposures.values, weights=list(weights.values), computeStats=False).params
        else:
            logging.debug('Solving linear model, dimensions: %s, robust %s', goodExposures.shape, robust)
            factorReturns = robustLinearSolver(\
                    goodData.values, goodExposures.values, weights=list(weights.values), computeStats=False, robust=robust).params

        # Set up solution objects
        factorReturns = pandas.DataFrame(factorReturns, index=goodExposures.columns, columns=goodData.columns).fillna(0.0)
        rmgExposures = rmgExposures.loc[:, okExpCols]

        # Warn on assets with all missing or zero exposures
        hasExpAssets = set(rmgExposures.mask(rmgExposures==0.0).dropna(how='all', axis=0).index)
        noExpAssets = set(rmgAssets).difference(hasExpAssets)
        if len(noExpAssets) > 0:
            sidList = [sid.getSubIDString() for sid in noExpAssets]
            logging.warning('%s: %d assets with no or zero exposures in proxy: %s',
                    region.description, len(noExpAssets), sidList)

        # Replace missing observations with proxied values
        estimatedData = clip_extremaDF(rmgExposures.fillna(0.0).dot(factorReturns), 0.01)
        rmgData = rmgData.fillna(0.0) + estimatedData.mask(numpy.isfinite(rmgData)).fillna(0.0)

        # Save to full size arrays
        dataCopy.loc[rmgAssets, :] = rmgData.loc[rmgAssets, :]
        dataEstimates.loc[rmgAssets, :] = estimatedData.loc[rmgAssets, :]

        # Some debugging output
        if extraDebugging:
            dFile = 'tmp/proxy-frets-%s-%s.csv' % (region.region_id, date)
            writeToCSV(factorReturns, dFile, rowNames=list(goodExposures.columns), dp=8)

            estuVec = pandas.Series(1.0, index=rmgEstu, name='estu').reindex(rmgAssets).fillna(0.0)
            tmpExpos = pandas.concat([rmgExposures, estuVec], axis=1)
            dFile = 'tmp/proxy-expos-%s-%s.csv' % (region.region_id, date)
            writeToCSV(tmpExpos.values, dFile, rowNames=[sid.getSubIDString() for sid in rmgAssets],
                        columnNames=list(tmpExpos.columns), dp=8)

    # Output for debugging
    errorNorm = (dataCopy-dataEstimates).fillna(0.0).sum(axis=1).max(axis=None)
    logging.info('Residual norm of proxied data: %.3f', errorNorm)

    if debugging:
        idList = [sid.getSubIDString() for sid in assetData.universe]
        dFile = 'tmp/proxy-data-raw-%s.csv' % date
        writeToCSV(data.values, dFile, rowNames=idList, columnNames=list(data.columns), dp=8)
        dFile = 'tmp/proxy-data-estm-%s.csv' % date
        writeToCSV(dataEstimates.values, dFile, rowNames=idList, columnNames=list(dataEstimates.columns), dp=8)

    logging.debug('proxyMissingAssetData: end')
    return dataCopy, dataEstimates

def procrustes_transform(source, target, returnTransform=False):
    """ Performs orthogonal Procrustean analysis to transform the source to be as close as possible to the target.
        The source and target are specified as pandas DataFrames.
        Returns a pandas dataframe with the transformed source.
    """
    if not isinstance(target, list):
        target = [target]

    if any([numpy.any(t.index != t.columns) for t in target]) or numpy.any(source.index != source.columns):
        raise Exception('Unable to perform procrustes transformation: source and/or target dataframes do not appear to be symmetric; verify that index and columns match.')

    if not all([set(t.index).issubset(set(source.index)) for t in target]):
        raise Exception('Unable to perform procrustes transformation: target.index is not a subset of source.index')

    def getTransFormMatrixFragment(targetBlock):
        idx = targetBlock.index
        sourceBlock = source.loc[idx,idx]

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

        return tb.dot(Q).dot(sb_inv)

    # initialize transform matrix
    transform = pandas.DataFrame(numpy.eye(len(source.index)),index=source.index,columns=source.columns)
    for targetBlock in target:
        idx = targetBlock.index
        transform.loc[idx,idx] = getTransFormMatrixFragment(targetBlock)
    
    # compute final matrix
    finalMatrix = transform.dot(source).dot(transform.T)
    if returnTransform:
        return transform

    return finalMatrix

def inverse_herfindahl(weights):
    """Computes the inverse Herfindahl index for a given set of weights
    """
    herf = 0.0
    sumWgt = ma.sum(weights, axis=None)
    if sumWgt > 0:
        scaledWt = weights / sumWgt
        herf = 1.0 / ma.inner(scaledWt, scaledWt)
    return herf

def optimal_shrinkage(covMatrix, nObs):
    """Performs optimal shrinkage on covariance matrix via adjustment of eigenvalues
    """
    # Initialise
    (eigval, eigvec) = linalg.eigh(covMatrix)
    T = nObs
    N = eigval.shape[0]
    q = float(N) / T
    rootQ = numpy.sqrt(q)
    lambda_min = numpy.clip(numpy.min(eigval, axis=None), 0.0, None)
    lambda_plus = lambda_min * ((1+rootQ) * (1+rootQ) / ((1-rootQ) * (1-rootQ)))
    sigma2 = lambda_min / ((1-rootQ) * (1-rootQ))

    # Loop round eigenvalues
    lambda_new = []
    for idx in range(N):
        lambda_old = eigval[idx]
        z = complex(lambda_old, -1.0 / numpy.sqrt(float(N)))
        s = 0.0
        for jdx in range(N):
            if jdx != idx:
                s += 1.0 / (z - eigval[jdx])
        s = s / float(N)

        if sigma2 == 0.0:
            gamma = 0.0
        else:
            g = z + (sigma2 * (q-1)) - (numpy.sqrt(z-lambda_min) * numpy.sqrt(z-lambda_plus))
            g = g / (2.0*q*z*sigma2)
            gamma = abs(1.0 - q + (q*z*g))
            gamma = sigma2 * gamma * gamma / lambda_old

        # Adjust eigenvalue
        new_val = lambda_old / abs(1.0-q + (q*z*s))**2.0
        if gamma > 1.0:
            new_val = gamma * new_val
        lambda_new.append(new_val)

    # Reconstitute covariance matrix with adjusted eigenvalues
    lambda_new = numpy.array(lambda_new)
    factorCov = numpy.dot(eigvec, numpy.dot(numpy.diag(lambda_new), numpy.transpose(eigvec)))
    return factorCov

def compute_beta_from_cov(cov, to, fro):
    """ Compute beta based on covariance matrix
    """
    return cov.loc[to, fro].dot(pandas.DataFrame(numpy.linalg.pinv(cov.loc[fro, fro]), index=fro, columns=fro))
