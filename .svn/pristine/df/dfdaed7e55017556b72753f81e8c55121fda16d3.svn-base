import numpy.ma as ma
import numpy
import pandas
from riskmodels import Utilities
from riskmodels import AssetProcessor_V4

def allMasked(shape, dtype=float):
    return ma.masked_all(shape, dtype)
     
def fillAndMaskByDate(array, issueDates, arrayDates, maskIPODate=False):
    """Array is assumed to be a masked asset-by-date array.
    arrayDates is a list of the dates corresponding to the columns in array.
    issueDates is a list with a date for each asset.
    The returned array has all masked entries in array filled with 0.0
    and then all entries (i,j) masked for which arrayDates[j] < issueDates[i].
    """
    if type(array) is pandas.DataFrame:
        rowDates = numpy.array(issueDates)[:, numpy.newaxis]
        colDates = numpy.array(array.columns)[numpy.newaxis, :]
        if maskIPODate:
            return array.fillna(0.0).mask(colDates<=rowDates)
        return array.fillna(0.0).mask(colDates<rowDates)
    issueDates = numpy.array(issueDates)
    issueDates = issueDates[:, numpy.newaxis]
    arrayDates = numpy.array(arrayDates)
    arrayDates = arrayDates[numpy.newaxis, :]
    array = ma.filled(array, 0.0)
    if maskIPODate:
        return ma.masked_where(arrayDates<=issueDates, array)
    return ma.masked_where(arrayDates<issueDates, array)

def maskByDate(array, fromDates, maskIPODate=False):
    """The returned array has all entries (i,j) masked for which arrayDates[j] < issueDates[i].
    """
    rowDates = numpy.array(fromDates)[:, numpy.newaxis]
    colDates = numpy.array(array.columns)[numpy.newaxis, :]
    if maskIPODate:
        return array.mask(colDates<=rowDates)
    return array.mask(colDates<rowDates)

class TimeSeriesMatrix:
    """(asset by date) matrix of values.
    """
    def __init__(self, assets, dates):
        self.assets = assets
        self.dates = dates
        self.data = ma.zeros((len(assets), len(dates)), float)

    def toDataFrame(self):
        import pandas
        return pandas.DataFrame(self.data, index=self.assets, columns=self.dates)

    @classmethod
    def fromDataFrame(cls, df):
        import pandas
        if len(df.index) <= 0 or len(df.columns) <= 0:
            raise ValueError('DataFrame cannot be empty')
        dates = [x.to_pydatetime().date() for x in pandas.to_datetime(df.columns)]
        ts = cls(df.index.tolist(), dates)
        ts.data = ma.array(df.values.copy(), mask=pandas.isnull(df).values) 
        return ts

class FactorType:
    """Class to represent different factor type in the exposure matrix.
    """
    def __init__(self, name, description):
        self.name = name
        self.description = description
    def __eq__(self, other):
        if other is None:
            return False
        assert(isinstance(other, FactorType))
        return self.name == other.name
    def __ne__(self, other):
        if other is None:
            return True
        assert(isinstance(other, FactorType))
        return self.name != other.name
    def __lt__(self, other):
        if other is None:
            return False
        assert(isinstance(other, FactorType))
        return self.name < other.name
    def __le__(self, other):
        if other is None:
            return False
        assert(isinstance(other, FactorType))
        return self.name <= other.name
    def __gt__(self, other):
        if other is None:
            return True
        assert(isinstance(other, FactorType))
        return self.name > other.name
    def __ge__(self, other):
        if other is None:
            return True
        assert(isinstance(other, FactorType))
        return self.name >= other.name
    def __hash__(self):
        return self.name.__hash__()
    def __repr__(self):
        return 'FactorType(%s)' % self.name

class ExposureMatrix:
    """(factor by asset) matrix of (masked) values.
    """
    InterceptFactor = FactorType('Market', 'Market factor')
    RegionalIntercept = FactorType('Regional Intercept', 'Linked model regional intercept')
    StyleFactor = FactorType('Style', 'Style and fundamental factors')
    IndustryFactor = FactorType('Industry', 'Industry classification')
    StatisticalFactor = FactorType('Statistical', 'Statistical factors')
    CountryFactor = FactorType('Country', 'Country classification')
    CurrencyFactor = FactorType('Currency', 'Currency classification')
    LocalFactor = FactorType('Local', 'Local market factors')
    MacroCoreFactor = FactorType('Core Macro','Core macroeconomic time-series factors')
    MacroMarketTradedFactor = FactorType('Market-Traded Macro','Market-traded macroeconomic time-series factors')
    MacroEquityFactor = FactorType('Equity','Equity time-series factors')
    MacroSectorFactor = FactorType('Sector','Sector sensitivity factors')
    MacroFactor = FactorType('Macro','Macroeconomic time series factors')

    
    def __init__(self, assetList, factorList=None):
        """Create an all masked exposure matrix over the assets in the
        asset list and the given factors.
        Each entry in factorList is a pair of (factorName, factorType)
        """
        self.assets_ = assetList
        self.factors_ = list()
        
        self.factorTypes_ = (self.InterceptFactor, self.StyleFactor,
                             self.IndustryFactor, self.RegionalIntercept,
                             self.StatisticalFactor, self.CountryFactor,
                             self.CurrencyFactor, self.LocalFactor,
                             self.MacroCoreFactor,self.MacroMarketTradedFactor,self.MacroEquityFactor,
                             self.MacroSectorFactor, self.MacroFactor)
        self.factorIdxMap_ = dict()
        for f in self.factorTypes_:
            self.factorIdxMap_[f] = dict()
        if factorList is not None:
            for (fIdx, (fName, fType)) in enumerate(factorList):
                self.factors_.append(fName)
                self.factorIdxMap_[fType][fName] = fIdx
        self.data_ = allMasked((len(self.factors_), len(self.assets_)))
        
    def addFactor(self, factorName, factorExposures, fType):
        assert(fType in self.factorTypes_)
        if type(factorExposures) is pandas.Series:
            factorExposures = Utilities.df2ma(factorExposures)
        assert(len(factorExposures.shape)==1)
        if factorName in self.factors_:
            expMIdx = self.factorIdxMap_[fType][factorName]
            self.data_[expMIdx,:] = factorExposures
        else:
            self.data_ = ma.concatenate([self.data_, factorExposures[numpy.newaxis]])
            self.factorIdxMap_[fType][factorName] = len(self.factors_)
            self.factors_.append(factorName)
    
    def addFactors(self, factorNames, factorExposures, fType):
        if type(factorExposures) is pandas.DataFrame:
            factorExposures = Utilities.df2ma(factorExposures)
        assert(len(factorNames)==factorExposures.shape[0])
        assert(fType in self.factorTypes_)
        assert(len(factorExposures.shape)==2)
        # Overwrite any factors if they already exist
        owFactorIdx = [idx for (idx, f) in enumerate(factorNames) if f in self.factors_]
        if len(owFactorIdx) > 0:
            keepFactorIdx = [idx for idx in range(len(factorNames)) if idx not in owFactorIdx]
            for idx in owFactorIdx:
                expMIdx = self.factorIdxMap_[fType][factorNames[idx]]
                self.data_[expMIdx,:] = factorExposures[idx,:]
            factorNames = [f for (idx,f) in enumerate(factorNames) \
                    if idx not in owFactorIdx]
            factorExposures = ma.take(factorExposures, keepFactorIdx, axis=0)
        if len(factorNames) > 0:
            self.data_ = ma.concatenate([self.data_, factorExposures])
            for f in factorNames:
                self.factorIdxMap_[fType][f] = len(self.factors_)
                self.factors_.append(f)
         
    def fill(self, val):
        self.data_ = ma.filled(self.data_, val)
    
    def getMatrix(self):
        return self.data_
    
    def getAssets(self):
        return self.assets_
   
    def setAssets(self, assets):
        """Set assets for this exposure matrix to given list,
           keeping the factor structure in place
           Caution: this will erase all the existing data
        """
        self.assets_ = list(assets)
        self.data_ = allMasked((len(self.factors_), len(self.assets_)))


    def getFactorIndex(self, factorName):
        for t in self.factorTypes_:
            if self.checkFactorType(factorName, t):
                return self.factorIdxMap_[t][factorName]
                break
        raise LookupError('Unknown factor: %s' % factorName)
    
    def getFactorNames(self, fType=None):
        if fType:
            assert(fType in self.factorTypes_)
            return list(self.factorIdxMap_[fType].keys())
        else:
            return self.factors_
    
    def getFactorIndices(self, fType=None):
        if fType:
            assert(fType in self.factorTypes_)
            return list(self.factorIdxMap_[fType].values())
        else:
            return [self.getFactorIndex(f) for f in self.factors_]

    def getFactorType(self, factorName):
        for fType in self.factorTypes_:
            if self.checkFactorType(factorName, fType):
                return fType
        raise KeyError('no factor called "%s" in exposure matrix' % factorName)
    
    def checkFactorType(self, factorName, fType):
        assert(fType in self.factorTypes_)
        return factorName in self.factorIdxMap_[fType]
    
    def dumpToFile(self, filepath, modelDB, marketDB, date, estu=None,
            subIssueGroups=None, orderFactors=True, compact=True,
            fillMissing=False, assetType=None, dp=12, assetData=None):
        sidList = [sid if isinstance(sid, str) else sid.getModelID().getIDString() for sid in self.assets_]
        assetGroupMap = dict(zip(self.assets_, sidList))

        # Sort out asset data mappings
        if assetData is not None:
            estuFlag = pandas.Series(0, index=self.assets_)
            if hasattr(assetData, 'estimationUniverse'):
                estuFlag[assetData.estimationUniverse] = 1
            estuFlag = estuFlag.values
            assetType = assetData.assetTypeDict if hasattr(assetData, 'assetTypeDict') else assetData.getAssetType()
            assetGroupMap = assetData.sidToCIDMap if hasattr(assetData, 'sidToCIDMap') else assetData.getSubIssue2CidMapping()
            assetISINMap = assetData.assetISINMap if hasattr(assetData, 'assetISINMap') else assetData.getISINMap()
            marketTypeMap = assetData.marketTypeDict if hasattr(assetData, 'marketTypeDict') else assetData.getMarketType()
            if type(assetData.marketCaps) is pandas.Series:
                marketCapMap = assetData.marketCaps.fillna(0.0).to_dict()
            else:
                marketCapMap = dict(zip(assetData.universe, ma.filled(assetData.marketCaps, 0.0)))
        else:
            estuFlag = numpy.zeros((self.data_.shape[1]), int)
            if estu != None:
                numpy.put(estuFlag, estu, 1)
            from riskmodels import AssetProcessor
            assetType = AssetProcessor.get_asset_info(date, self.assets_, modelDB, marketDB,
                    'ASSET TYPES', 'Axioma Asset Type')
            marketTypeMap = AssetProcessor.get_asset_info(date, self.assets_, modelDB, marketDB,
                    'REGIONS', 'Market')
            assetGroupMap = modelDB.getIssueCompanies(date, self.assets_, marketDB)
            mdIDs = [sid.getModelID() for sid in self.assets_]
            mdIDtoISINMap = modelDB.getIssueISINs(date, mdIDs, marketDB)
            marketCapMap = dict()
            assetISINMap = dict()
            for (mid, sid) in zip(mdIDs, self.assets_):
                if mid in mdIDtoISINMap:
                    assetISINMap[sid] = mdIDtoISINMap[mid]
        exSpac = AssetProcessor_V4.sort_spac_assets(date, self.assets_, modelDB, marketDB, returnExSpac=True)

        if assetType is None:
            assetType = dict()
        outfile = open(filepath, 'w')
        outfile.write(',,,')
        fTypes = [self.getFactorType(f) for f in self.factors_]
        compactTypes = []
        if compact:
            compactTypes = [self.IndustryFactor, self.CountryFactor, self.CurrencyFactor]
            compactTypes = [t for t in compactTypes if t in fTypes]
        if orderFactors:
            fNameOrder = numpy.array(self.factors_).argsort()
        else:
            fNameOrder = list(range(len(self.factors_)))
        for fid in fNameOrder:
            if fTypes[fid] not in compactTypes:
                outfile.write(',%s' % self.factors_[fid].replace(',',''))
        for t in compactTypes:
            outfile.write(',%s' % t.name)
        outfile.write(',estu,type,exSPAC,market,mcap\n')
        data = self.data_
        data = Utilities.screen_data(data)
        data = Utilities.p2_round(data, dp)
        data.mask = ma.getmaskarray(data)
        ids = [self.assets_[i] if isinstance(self.assets_[i], str) else self.assets_[i].getModelID() for i in range(self.data_.shape[1])]
        idxOrder = numpy.array(ids).argsort()
        nameMap = modelDB.getIssueNames(date, ids, marketDB)
        for i in idxOrder:
            exSpacFlag = False
            if self.assets_[i] in exSpac:
                exSpacFlag = True
            outfile.write('%s' % sidList[i])
            outfile.write(',%s' % assetGroupMap.get(self.assets_[i], sidList[i]))
            outfile.write(',%s' % nameMap.get(ids[i], '').replace(',',''))
            outfile.write(',%s' % assetISINMap.get(self.assets_[i], None))
            cmpData = [None] * len(compactTypes)
            for j in fNameOrder:
                if fTypes[j] in compactTypes:
                    if data.mask[j,i] == False and data[j,i] > 0.0:
                        idx = compactTypes.index(fTypes[j])
                        cmpData[idx] = self.factors_[j].replace(',','')
                else:
                    if data.mask[j,i] == False:
                        outfile.write(',{0:.{1}f}'.format(data[j,i], dp))
                    else:
                        if fillMissing:
                            outfile.write(',0.0')
                        else:
                            outfile.write(',')
            for (j,t) in enumerate(compactTypes):
                outfile.write(',%s' % cmpData[j])
            outfile.write(',%d,%s,%s,%s,%s\n' % (estuFlag[i],
                assetType.get(self.assets_[i],None),exSpacFlag,
                marketTypeMap.get(self.assets_[i],None),
                marketCapMap.get(self.assets_[i],None)))
        outfile.close()

    def attrDumpToFile(self, filepath, attrSuffix, modelDB, marketDB, date, estu=None,
            subIssueGroups=None, orderFactors=True, compact=True, fillMissing=True):
        estuFlag = numpy.zeros((self.data_.shape[1]), int)
        if estu != None:
            numpy.put(estuFlag, estu, 1)
        sidList = [sid.getModelID().getIDString() for sid in self.assets_]
        assetGroupMap = dict(zip(self.assets_, sidList))
        if subIssueGroups != None:
            for (groupId, subIssueList) in subIssueGroups.items():
                for sid in subIssueList:
                    assetGroupMap[sid] = groupId
        outfile = open(filepath, 'w')
        fTypes = [self.getFactorType(f) for f in self.factors_]
        compactTypes = []
        if compact:
            compactTypes = [self.IndustryFactor, self.CountryFactor, self.CurrencyFactor]
            compactTypes = [t for t in compactTypes if t in fTypes]
        if orderFactors:
            fNameOrder = numpy.array(self.factors_).argsort()
        else:
            fNameOrder = list(range(len(self.factors_)))

        for fid in fNameOrder:
            if fTypes[fid] not in compactTypes:
                outfile.write(',ALPHA')
        for t in compactTypes:
            outfile.write(',ALPHA')
        outfile.write('\n')

        outfile.write('NAME')
        for fid in fNameOrder:
            if fTypes[fid] not in compactTypes:
                outfile.write(',%s-%s' % (self.factors_[fid].replace(',','').replace(' ',''), attrSuffix))
        for t in compactTypes:
            outfile.write(',%s' % t.name)
        outfile.write('\n')

        outfile.write('UNIT')
        for fid in fNameOrder:
            if fTypes[fid] not in compactTypes:
                outfile.write(',NUMBER')
        for t in compactTypes:
            outfile.write(',NUMBER')
        outfile.write('\n\n')

        data = self.data_
        ids = [self.assets_[i].getModelID() for i in range(data.shape[1])]
        idxOrder = numpy.array(ids).argsort()
        nameMap = modelDB.getIssueNames(date, ids, marketDB)
        for i in idxOrder:
            outfile.write('%s' % sidList[i][1:])
            cmpData = list(compactTypes)
            for j in fNameOrder:
                if fTypes[j] in compactTypes:
                    if data.mask[j,i] == False and data[j,i] > 0.0:
                        idx = compactTypes.index(fTypes[j])
                        cmpData[idx] = self.factors_[j].replace(',','')
                else:
                    if data.mask[j,i] == False:
                        outfile.write(',%.12f' % data[j,i])
                    else:
                        if fillMissing:
                            outfile.write(',0.0')
                        else:
                            outfile.write(',')
            for (j,t) in enumerate(compactTypes):
                outfile.write(',%s' % cmpData[j])
            outfile.write('\n')
        outfile.close()

    def bucketize(self, fType, enforce_coverage=False):
        """Bucketizes assets by industry/country membership.
        Returns a list of lists, with each nested list representing
        array positions of assets in that industry/country.
        Assumes the given factor type 'partitions' the asset 
        universe -- that is, each asset only has exposure value
        to one industry/country.
        """
        # Check if factors form partition
        exposures = ma.take(self.data_,
                        self.getFactorIndices(fType), axis=0)
        if enforce_coverage:
            assert(numpy.sum(ma.getmaskarray(exposures)==0, axis=0).all())
        exposures = ma.masked_where(exposures==0.0, exposures)
        buckets = [numpy.flatnonzero(ma.getmaskarray(exposures[i,:])==0) \
                        for i in range(exposures.shape[0])]
        return buckets
    
    def toDataFrame(self):
        import pandas
        return pandas.DataFrame(self.data_.T, index=self.assets_, columns=self.factors_)

