
import os
import optparse
import collections
import cx_Oracle
import datetime
import logging
import numpy
import numpy.ma as ma
import pandas
import itertools
import math
import pandas.io.sql as sql
from collections import defaultdict
from riskmodels import ModelID
from marketdb import MarketID
from marketdb import MarketDB
from marketdb.Utilities import listChunkIterator
from marketdb.MarketDB import BaseProcessor
from riskmodels import Classification
from riskmodels import Factors
from riskmodels import Matrices
from riskmodels import ProcessReturns
from riskmodels import Utilities

pandas_version = int(pandas.__version__.split('.')[1])
if pandas_version >= 15:
    sql.read_frame = sql.read_sql


from string import Template
from marketdb.Utilities import MATRIX_DATA_TYPE as MX_DT
import marketdb.Utilities as MktDBUtils
MINCR = 500

#from MarketDB.Utilities import do_line_profile

class MergerSurvivor:
    """Corporate action denoting which company was declared the survivor
    of a merger/acquisition and what terms apply to it.
    """
    def __init__(self, ca_sequence, modeldb_id, new_marketdb_id,
                 old_marketdb_id, share_ratio, cash_payment,
                 currency_id, ref):
        self.sequenceNumber = ca_sequence
        self.modeldb_id = modeldb_id
        self.new_marketdb_id = new_marketdb_id
        self.old_marketdb_id = old_marketdb_id
        self.share_ratio = share_ratio
        self.cash_payment = cash_payment
        self.currency_id = currency_id
        self.ref = ref
    def __str__(self):
        if self.cash_payment is None or self.cash_payment == 0.0:
            return '(%d, %s, MergerSurvivor %s-> %s %s)' % (
                self.sequenceNumber, self.modeldb_id, self.old_marketdb_id,
                self.share_ratio, self.new_marketdb_id)
        else:
            return '(%d, %s, MergerSurvivor %s-> %s %s + %s currency %d)' % (
                self.sequenceNumber, self.modeldb_id, self.old_marketdb_id,
                self.share_ratio, self.new_marketdb_id, self.cash_payment,
                self.currency_id)
    def __repr__(self):
        return self.__str__()
    
    def backwardAdjustPrice(self, marketIssue, price, currency,
                            currencyConverters, **other):
        assert(marketIssue == self.new_marketdb_id)
        if self.cash_payment is None or self.cash_payment == 0.0:
            return (self.old_marketdb_id, price * self.share_ratio)
        exchangeRate = None
        for conv in currencyConverters:
            if conv.hasRate(self.currency_id, currency):
                exchangeRate = conv.getRate(self.currency_id, currency)
                break
        if exchangeRate is None:
            raise KeyError('No exchange rate from %s to %s' % (
                self.currency_id, currency))
        return (self.old_marketdb_id,
                price * self.share_ratio + self.cash_payment * exchangeRate)
    
    def getInvolvedModelIDs(self):
        return [self.modeldb_id]

class SpinOff:
    """Corporate action denoting a company handing out shares of a different
    company.
    """
    def __init__(self, ca_sequence, parent_id, child_id, share_ratio,
                 implied_dividend, currency_id, ref):
        self.sequenceNumber = ca_sequence
        self.modeldb_id = parent_id
        self.child_id = child_id
        self.share_ratio = share_ratio
        self.implied_dividend = implied_dividend
        self.currency_id = currency_id
        self.ref = ref
    def __str__(self):
        if self.implied_dividend is None:
            return '(%d, %s, Spin-off %s share of %s)' % (
                self.sequenceNumber, self.modeldb_id,
                self.share_ratio, self.child_id)
        else:
            return '(%d, %s, Spin-off (%s) implied dividend %f, currency %d)' \
                   % (self.sequenceNumber, self.modeldb_id,
                      self.child_id, self.implied_dividend,
                      self.currency_id)
    def __repr__(self):
        return self.__str__()
    def backwardAdjustPrice(self, marketIssue, price, priceMap,
                            currency, currencyConverters, **other):
        if self.implied_dividend is None:
            priceVal = 0.0
            if self.child_id in priceMap:
                priceVal = priceMap[self.child_id]
                div_currency_id = priceVal.currency_id
                dividend = priceVal.ucp * self.share_ratio
            else:
                dividend = 0.0
                div_currency_id = 1
                logging.error("Child id %s does not have price. Ignoring spin-off." % self.child_id)
        else:
            dividend = self.implied_dividend
            div_currency_id = self.currency_id
        exchangeRate = None
        for conv in currencyConverters:
            if conv.hasRate(div_currency_id, currency):
                exchangeRate = conv.getRate(div_currency_id, currency)
                break
        if exchangeRate is None:
            raise KeyError('No exchange rate from %s to %s' % (
                div_currency_id, currency))
        return (marketIssue, price + dividend * exchangeRate)
    
    def getInvolvedModelIDs(self):
        return [self.modeldb_id, self.child_id]

class CANewIssue:
    def __init__(self, modeldb_id, marketdb_id, ref):
        self.modeldb_id = ModelID.ModelID(string=modeldb_id)
        self.marketdb_id = marketdb_id
        self.ref = ref

class RMSDeadIssue:
    def __init__(self, date, modeldb_id, rms_id, ref):
        self.date = date
        self.modeldb_id = ModelID.ModelID(string=modeldb_id)
        self.rms_id = rms_id
        self.ref = ref
        
    def apply(self, modelDB):
        """Update the ModelDB according to this record.
        """
        modelDB.dbCursor.execute("""SELECT from_dt, thru_dt FROM rms_issue
        WHERE rms_id=:rms_arg AND issue_id=:issue_arg
        AND thru_dt > :date_arg""",
                                 rms_arg=self.rms_id,
                                 issue_arg=self.modeldb_id.getIDString(),
                                 date_arg = self.date)
        r = modelDB.dbCursor.fetchall()
        if len(r) == 1:
            old = r[0]
            modelDB.dbCursor.execute("""UPDATE rms_issue
            SET thru_dt=:date_arg
            WHERE rms_id=:rms_arg AND issue_id=:issue_arg AND 
            thru_dt=:thru_arg""",
                                     rms_arg=self.rms_id,
                                     issue_arg=self.modeldb_id.getIDString(),
                                     date_arg = self.date,
                                     thru_arg=old[1])
        elif len(r) == 0:
            logging.warning('No active rms_issue record for %s on %s; nothing to delete',
                         self.modeldb_id.getIDString(), self.date)
        else:
            raise Exception('Multiple active future rms_issue records for %s on %s' % (self.modeldb_id.getIDString(), self.date))

class RiskModelInstance:
    """An instance of a risk model on a given day.
    An instance is identified by its series and date.
    It has boolean variables that indicate if exposures, returns, and
    risks are present.
    """
    def __init__(self, rms_id, date, has_exposures=False, has_returns=False, 
                 has_risks=False, is_final=True):
        self.rms_id = rms_id
        self.date = date
        self.has_exposures = has_exposures
        self.has_returns = has_returns
        self.has_risks = has_risks
        self.is_final = is_final

    def __str__(self):
        return 'Risk model instance: RMS_ID: {rms_id}, date {date!s}, exposures? {exp} returns? {rtn} risks? {rsk} final? {final}'.format(
            rms_id=self.rms_id, date=self.date, exp=self.has_exposures, rtn=self.has_returns, rsk=self.has_risks, final=self.is_final
            )
    
    def setHasExposures(self, val, modelDB):
        """Set has_exposures to val and update it in the database.
        """
        if val:
            dbVal = 1
            self.has_exposures = True
        else:
            dbVal = 0
            self.has_exposures = False
        modelDB.dbCursor.execute("""UPDATE risk_model_instance
        SET has_exposures = :has_exp_arg, update_dt = :dt_arg
        WHERE rms_id = :rms_arg
        AND dt = :date_arg""", has_exp_arg=dbVal, dt_arg=datetime.datetime.now(),
                               rms_arg=self.rms_id, date_arg=self.date)
        
    def setHasReturns(self, val, modelDB):
        """Set has_returns to val and update it in the database.
        """
        if val:
            dbVal = 1
            self.has_returns = True
        else:
            dbVal = 0
            self.has_returns = False
        modelDB.dbCursor.execute("""UPDATE risk_model_instance
        SET has_returns = :has_ret_arg, update_dt = :dt_arg
        WHERE rms_id = :rms_arg 
        AND dt = :date_arg""", has_ret_arg=dbVal, dt_arg=datetime.datetime.now(),
                               rms_arg=self.rms_id, date_arg=self.date)
        
    def setHasRisks(self, val, modelDB):
        """Set has_risks to val and update it in the database.
        """
        if val:
            dbVal = 1
            self.has_risks = True
        else:
            dbVal = 0
            self.has_risks = False
        modelDB.dbCursor.execute("""UPDATE risk_model_instance
        SET has_risks = :has_risk_arg, update_dt = :dt_arg
        WHERE rms_id = :rms_arg
        AND dt = :date_arg""", has_risk_arg=dbVal, dt_arg=datetime.datetime.now(),
                               rms_arg=self.rms_id, date_arg=self.date)

    def setIsFinal(self, val, modelDB):
        """Set is_final to val and update it in the database.
        """
        if val:
            dbVal = 1
            self.is_final = True
        else:
            dbVal = 0
            self.is_final = False
        modelDB.dbCursor.execute("""UPDATE risk_model_instance
        SET is_final = :is_final_arg, update_dt = :dt_arg
        WHERE rms_id = :rms_arg
        AND dt = :date_arg""", is_final_arg=dbVal, dt_arg=datetime.datetime.now(),
                               rms_arg=self.rms_id, date_arg=self.date)

class RiskModelGroup:
    """A RiskModelGroup stores information about a particular
    market and how it changes over time.
    """
    def __init__(self, rmg_id, desc, 
                 mnemonic, region, gmt_offset, iso_code, timeVariantDicts):
        self.rmg_id = rmg_id
        self.description = desc
        self.mnemonic = mnemonic
        self.region_id = region
        self.gmt_offset = gmt_offset
        self.iso_code = iso_code
        self.timeVariantDicts = timeVariantDicts
        self.currency_code = None
        self.developed = None
    def __eq__(self, other):
        if hasattr(self, 'rmg_id') and hasattr(other, 'rmg_id'):
            return self.rmg_id == other.rmg_id
        return False
    def __ne__(self, other):
        return self.rmg_id != other.rmg_id
    def __lt__(self, other):
        return self.rmg_id < other.rmg_id
    def __le__(self, other):
        return self.rmg_id <= other.rmg_id
    def __gt__(self, other):
        return self.rmg_id > other.rmg_id
    def __ge__(self, other):
        return self.rmg_id >= other.rmg_id
    def __hash__(self):
        return self.rmg_id.__hash__()
    def __str__(self):
        return 'RiskModelGroup(%s, %s)' % (str(self.rmg_id), self.description)
    def __repr__(self):
        return self.__str__()
    
    def getCurrencyCode(self, date):
        """Returns the currency_code of this RMG on the given date.
        """
        for d in self.timeVariantDicts:
            if d['from_dt'] <= date and d['thru_dt'] > date:
                return d['currency_code']
        return None
    
    def setRMGInfoForDate(self, date):
        """Determines the risk model group's currency and
        market developed status for the given date and 
        sets the class attributes.
        """
        for d in self.timeVariantDicts:
            if d['from_dt'] <= date and d['thru_dt'] > date:
                self.currency_code = d['currency_code']
                self.developed = d['developed']
                self.emerging = d['emerging']
                self.frontier = d['frontier']
                return True
        return False

class RiskModelRegion:
    """A RiskModelRegion represents a geography containing
    constituent countries (RiskModelGroups).
    """
    def __init__(self, id, name, desc, ccode=None):
        self.region_id = id 
        self.name = name
        self.description = desc
        self.currency_code = ccode
    def __eq__(self, other):
        return self.region_id == other.region_id
    def __ne__(self, other):
        return self.region_id != other.region_id
    def __lt__(self, other):
        return self.region_id < other.region_id
    def __le__(self, other):
        return self.region_id <= other.region_id
    def __gt__(self, other):
        return self.region_id > other.region_id
    def __ge__(self, other):
        return self.region_id >= other.region_id
    def __hash__(self):
        return self.region_id.__hash__()
    def __str__(self):
        return 'RiskModelRegion(%d, %s)' % (self.region_id, self.description)
    def __repr__(self):
        return self.__str__()

class SubIssue:
    def __init__(self, string):
        assert(len(string) == 12)
        self.string = string
        self.modelID = None
    def __eq__(self, other):
        return self.string == other.string
    def __ne__(self, other):
        return self.string != other.string
    def __lt__(self, other):
        return self.string < other.string
    def __le__(self, other):
        return self.string <= other.string
    def __gt__(self, other):
        return self.string > other.string
    def __ge__(self, other):
        return self.string >= other.string
    def __hash__(self):
        return self.string.__hash__()
    def __str__(self):
        return 'SubIssue(%s)' % self.string
    def __repr__(self):
        return self.__str__()
    def getSubIDString(self):
        return self.string
    def getSubIdString(self):
        return self.getSubIDString()
    def getModelID(self):
        if self.modelID is None:
            self.modelID = ModelID.ModelID(string=self.string[0:-2])
        return self.modelID
    def isCashAsset(self):
        return self.string[1:5] == 'CSH_'
    def getCashCurrency(self):
        assert(self.isCashAsset())
        return self.string[5:8]

class ModelSubFactor:
    """A sub-factor in a risk model.
    A sub-factor contains its factor, the sub-factor ID, and the date
    on which the sub-factor ID is valid.
    """
    def __init__(self, factor, subFactorID, date):
        self.factor = factor
        self.subFactorID = subFactorID
        self.date = date
    
    def __eq__(self, other):
        """Let two ModelSubFactors be equal if they have the same subFactorID.
        """
        return self.subFactorID == other.subFactorID
    
    def __ne__(self, other):
        return self.subFactorID != other.subFactorID
    
    def __lt__(self, other):
        return self.subFactorID < other.subFactorID
    
    def __le__(self, other):
        return self.subFactorID <= other.subFactorID
    
    def __gt__(self, other):
        return self.subFactorID > other.subFactorID
    
    def __ge__(self, other):
        return self.subFactorID >= other.subFactorID
    
    def __hash__(self):
        """Hash ModelSubFactors based on their subFactorID.
        """
        return self.subFactorID.__hash__()
    
    def __repr__(self):
        return 'ModelSubFactor(%s, %s, %s)' % (self.factor, str(self.subFactorID), self.date)

def maskToNone(val):
    """Returns None if val is masked and val otherwise."""
    if val is ma.masked:
        return None
    return val

class ForexCache:
    """Cache mechanism to expedite currency conversions 
    involving large datasets.
    'boT' is 'beginning of time' and allows use to specify a minimum look-back
    date, beyond which the code will not seek - will speed things up
    when there is genuinely nothing there
    """
    def __init__(self, marketDB, maxDates):
        self.currencyProvider = MarketDB.CurrencyProvider(
                                marketDB, 20, 1, maxNumConverters=maxDates)
        self.lookBackDays = 20
    
    def getCurrencyID(self, currencyCode, date):
        """Fetch the MarketDB ID of the given ISO currency code.
        """
        id = self.currencyProvider.getCurrencyID(currencyCode, date)
        self.currencyProvider.currencyIds.clear()
        return id
    
    def getRate(self, date, fromID, toID, tweakForNYD=False):
        """Retrieve the exchange rate for the given currency IDs.
        If no rate exists in the CurrencyConverter for the 
        specified date, search back self.lookBackDays number of days.
        """
        if fromID == toID:
            return 1.0
        for back in range(self.lookBackDays):
            d = date - datetime.timedelta(back)
            if tweakForNYD:
                if (d.day == 1) and (d.month == 1):
                    d = d - datetime.timedelta(1)
            try:
                cc = self.currencyProvider.getCurrencyConverter(d)
            except LookupError:
                cc = None
            if cc is not None:
                if cc.hasRate(fromID, toID):
                    if back > 0:
                        logging.warning(
                            'Using older exchange rate from %s for %s, IDs (%s, %s)' % \
                                (d, date, fromID, toID))
                    return cc.getRate(fromID, toID)
        logging.error('Unable to fetch currency rate for (%s, %s) %s', fromID, toID, date)
        return None
    
    def fillInMissingDates(self, dateList):
        """Populate the cache, adding CurrencyConverters if required.
        """
        pass

    def to_pickle(self, fname):
        import pickle as pkl
        with open(fname, 'wb') as f:
            obj = {'converters': self.currencyProvider.currencyConverters, 'maxNumConverters': self.currencyProvider.maxNumConverters}
            pkl.dump(obj, f, protocol=pkl.HIGHEST_PROTOCOL)

    @classmethod
    def from_pickle(cls, fname, marketDB):
        import pickle as pkl
        result = None
        with open(fname, 'rb') as fh:
            result = pkl.load(fh)
        cache = ForexCache(marketDB, result['maxNumConverters'])
        cache.currencyProvider.currencyConverters = result['converters']
        return cache

class ClassificationCache:
    def __init__(self, db, revision, isModelDB):
        self.historyDict = dict()
        # Get classifications and their hierarchy
        self.classificationDict = dict()
        if isModelDB:
            clsRoot = db.getMdlClassificationRevisionRoot(revision)
        else:
            clsRoot = db.getClassificationRevisionRoot(revision)
        clsRoot.level = 0
        clsRoot.levelParent = dict()
        self.classificationDict[clsRoot.id] = clsRoot
        clsRefs = [clsRoot]
        while len(clsRefs) > 0:
            cls = clsRefs.pop()
            if isModelDB:
                children = db.getMdlClassificationChildren(cls)
            else:
                children = db.getClassificationChildren(cls)
            parents = cls.levelParent.copy()
            parents[cls.level] = cls
            for c in children:
                self.classificationDict[c.id] = c
                c.level = cls.level + 1
                c.levelParent = parents
            clsRefs.extend(children)
        
    def addChange(self, key, changeDt, changeFlag, value):
        self.historyDict.setdefault(key, list()).append(
            (changeDt, changeFlag, value))
    
    def addKey(self, key):
        self.historyDict.setdefault(key, list())
        
    def getClassifications(self, idKeyList, date, level):
        """Returns a dictionary mapping each id in idKeyList of (id, key)
        pairs to their classification on date. The classification is
        looked up by the corresponding key.
        If the asset has no classification, it is omitted from the dictionary.
        The dictionary values are structures with classification_id,
        classification (as returned by getClassificationByID),
        weight, src, and ref.
        """
        retval = dict()
        for (idVal, key) in idKeyList:
            history = self.historyDict.get(key, list())
            history = [h for h in history if h[0] <= date]
            if len(history) > 0 and history[-1][1] == 'N':
                leaf = history[-1][2]
                if level is None:
                    leaf.classification = self.classificationDict.get(leaf.classification_id, None)
                    retval[idVal] = leaf
                else:
                    cls = self.classificationDict[leaf.classification_id]
                    if cls.level == level:
                        parent = cls
                    elif level >= 0:
                        parent = cls.levelParent[level]
                    else:
                        parent = cls.levelParent[cls.level + level]
                    val = Utilities.Struct(copy=leaf)
                    val.classification = parent
                    val.classification_id = parent.id
                    retval[idVal] = val
        return retval
                
    def getClassificationRange(self, idKeyList, date, level):
        """Returns a dictionary mapping each id in idKeyList of (id, key)
        pairs to their classification on date. The classification is
        looked up by the corresponding key.
        If the asset has no classification, it is omitted from the dictionary.
        The dictionary values are structures with classification_id,
        classification (as returned by getClassificationByID),
        weight, src, and ref.
        """
        retval = defaultdict(dict)
        for (idVal, key) in idKeyList:
            history = self.historyDict.get(key, list())
            history = [h for h in history if h[0] <= date]
            for hist in history:
                if hist[1] == 'N':
                    leaf = hist[2]
                    dt = hist[0]
                    if level is None:
                        leaf.classification = self.classificationDict.get(leaf.classification_id, None)
                        retval[idVal][dt] = leaf
                    else:
                        cls = self.classificationDict[leaf.classification_id]
                        if cls.level == level:
                            parent = cls
                        elif level >= 0:
                            parent = cls.levelParent[level]
                        else:
                            parent = cls.levelParent[cls.level + level]
                        val = Utilities.Struct(copy=leaf)
                        val.classification = parent
                        val.classification_id = parent.id
                        retval[idVal][dt] = val
        return retval

    def getMissingKeys(self, keyList):
        """Returns the list of keys that are not present in the cache.
        """
        return [key for key in keyList if key not in self.historyDict]

    def asFromThruCache(self, assets, level, idField='name'):
        """Convert the classification history for the given level into from/thru
        format and return it as a FromThruCache object.
        """
        cache = FromThruCache()
        for asset in assets:
            history = self.historyDict.get(asset, list())
            cache.addAssetValues(asset, self.convertHistoryToFromThru(history, level, idField))
        return cache

    def convertHistoryToFromThru(self, history, level, idField='name'):
        EOT = datetime.date(2999, 12, 31)
        newHistory = list()
        for (v1, v2) in zip(history, history[1:] + [(EOT, 'Y')]):
            if v1[1] == 'Y':
                continue
            fromDt = v1[0]
            thruDt = v2[0]
            # extract value based on level
            leaf = v1[2]
            if level is None:
                value = self.classificationDict[leaf.classification_id]
            else:
                cls = self.classificationDict[leaf.classification_id]
                if cls.level == level:
                    parent = cls
                elif level >= 0:
                    parent = cls.levelParent[level]
                else:
                    parent = cls.levelParent[cls.level + level]
                value = parent
            val = Utilities.Struct()
            val.fromDt = fromDt
            val.thruDt = thruDt
            # Classification IDs are unicode but other FromThruCache users use utf-8 for ID
            if idField == 'name':
                val.id = value.name
            else:
                val.id = getattr(value, idField)
            newHistory.append(val)
        return newHistory

class ValueHistoryCache:
    def __init__(self):
        self.historyDict = dict()
        
    def addChange(self, key, dt, value):
        self.historyDict.setdefault(key, list()).append((dt, value))
        
    def getHistories(self, idKeyList, startDate, endDate):
        """Returns a dictionary mapping each id in idKeyList of (id, key)
        pairs to their history truncated to the range
        from startDate (inclusive) to endDate (exclusive).
        The history is looked up by the corresponding key.
        The dictionary values are lists containing (dt, value) pairs.
        """
        retval = dict()
        for (idVal, key) in idKeyList:
            history = self.historyDict.get(key, list())
            history = [h for h in history
                       if h[0] >= startDate and h[0] < endDate]
            retval[idVal] = history
        return retval
                
    def getMissingKeys(self, keyList):
        """Returns the list of keys that are not present in the cache.
        """
        return [key for key in keyList if key not in self.historyDict]
    def sortHistory(self, key):
        hist = sorted(self.historyDict.setdefault(key, list()))

class TimeSeriesCache:
    def __init__(self, maxDates):
        self.IDDict = dict()
        self.IDList = []
        self.dateDict = dict() # maps dates to a [value, age] list
        self.maxDates = maxDates
    
    def addMissingIDs(self, tsm):
        """tsm is a TimeSeriesMatrix containing the data for the missing
        IDs on the existing dates.
        """
        assert(len(tsm.dates) == len(self.dateDict))
        update = dict(zip(tsm.assets, list(range(len(self.IDDict), len(self.IDDict) + len(tsm.assets)))))
        self.IDDict.update(update)
        self.IDList.extend(tsm.assets)
        for (i, d) in enumerate(tsm.dates):
            value = self.dateDict[d]
            value[0] = ma.concatenate((value[0], tsm.data[:,i]), axis=None)
            assert(value[0].shape[0] == len(self.IDDict))
    
    def addMissingDates(self, tsm, currencyCache):
        assert(len(tsm.assets) == len(self.IDDict))
        for (i, d) in enumerate(tsm.dates):
            assert(d not in self.dateDict)
            self.dateDict[d] = [ma.array(tsm.data[:, i]), 0]
    
    def removeDate(self, date):
        if date in self.dateDict:
            del self.dateDict[date]
    def findMissingDates(self, dateList):
        return [d for d in dateList if d not in self.dateDict]
    
    def findMissingIDs(self, IDList):
        return [a for a in IDList if a not in self.IDDict]
    
    def getDateList(self):
        return list(self.dateDict.keys())
    
    def getIDList(self):
        return self.IDList
    
    def getSubMatrix(self, dateList, ids):
        tsm = Matrices.TimeSeriesMatrix(ids, dateList)
        if numpy.size(tsm.data) == 0:
            return tsm
        tsm.data = Matrices.allMasked(tsm.data.shape)
        IDIndices = [self.IDDict[a] for a in ids]
        subMatrices = []
        for v in self.dateDict.values():
            v[1] += 1
        for dt in dateList:
            mat = self.dateDict[dt][0]
            self.dateDict[dt][1] = 0
            subMat = ma.take(mat, IDIndices, axis=0)
            subMat = ma.reshape(subMat, (subMat.shape[0], 1))
            subMatrices.append(subMat)
        tsm.data = ma.concatenate(subMatrices, axis=1)
        if len(self.dateDict) > self.maxDates:
            # Remove oldest dates
            ageList = sorted((-v[1], d) for (d, v) in self.dateDict.items())
            oldDates = ageList[:-(self.maxDates)]
            for (age, d) in oldDates:
                del self.dateDict[d]
            
        assert(len(tsm.data.shape) == 2)
        assert(tsm.data.shape[0] == len(tsm.assets))
        assert(tsm.data.shape[1] == len(tsm.dates))
        return tsm

    def size(self):
        return len(self.IDList), len(self.dateDict)

    def to_pickle(self, fname):
        import pickle as pkl
        ids = self.IDList
        dates = sorted(self.dateDict.keys())
        tsm = self.getSubMatrix(dates, ids)
        maindf = tsm.toDataFrame()
        with open(fname, 'wb') as f:
            obj = {'main': maindf, 'maxDates': self.maxDates}
            pkl.dump(obj, f, protocol=pkl.HIGHEST_PROTOCOL)

    @classmethod
    def from_pickle(cls, fname, currencyCache):
        import pickle as pkl
        result = None
        with open(fname, 'rb') as fh:
            result = pkl.load(fh)
        tsm = Matrices.TimeSeriesMatrix.fromDataFrame(result['main'])
        tsc = cls(result['maxDates'])
        emptytsm = Matrices.TimeSeriesMatrix([],tsm.dates)
        tsc.addMissingDates(emptytsm, currencyCache)
        tsc.addMissingIDs(tsm)
        return tsc

class TimeSeriesCurrencyCache:
    def __init__(self, maxDates):
        self.IDDict = dict()
        self.idCurrency = dict()
        self.IDList = []
        self.dateDict = dict() # maps dates to a [value, ccy, age] list
        self.maxDates = maxDates
        self.VAL_IDX = 0
        self.CCY_IDX = 1
        self.CCY_RATE_IDX = 2
        self.AGE_IDX = 3
    
    def addMissingIDs(self, tsm):
        """tsm is a TimeSeriesMatrix containing the data for the missing
        IDs on the existing dates.
        """
        assert(len(tsm.dates) == len(self.dateDict))
        update = dict(zip(tsm.assets, list(range(len(self.IDDict), len(self.IDDict) + len(tsm.assets)))))
        self.IDDict.update(update)
        self.IDList.extend(tsm.assets)
        for (i, d) in enumerate(tsm.dates):
            value = self.dateDict[d]
            #logging.info('XXX: %s, %s', value[self.VAL_IDX].shape, tsm.data.shape)
            value[self.VAL_IDX] = ma.concatenate(
                (value[self.VAL_IDX], tsm.data[:,i]), axis=None)
            value[self.CCY_IDX] = ma.concatenate(
                (value[self.CCY_IDX], tsm.ccy[:,i]), axis=None)
            value[self.CCY_RATE_IDX] = numpy.concatenate(
                (value[self.CCY_RATE_IDX],
                 numpy.ones(tsm.data[:,i].shape, dtype=float)),
                 axis=None)
            assert(value[self.VAL_IDX].shape[0] == len(self.IDDict))
            assert(value[self.CCY_IDX].shape[0] == len(self.IDDict))
            assert(value[self.CCY_RATE_IDX].shape[0] == len(self.IDDict))
    
    def addMissingDates(self, tsm, currencyCache):
        assert(len(tsm.assets) == len(self.IDDict))
        for (i, d) in enumerate(tsm.dates):
            assert(d not in self.dateDict)
            if len(tsm.assets) == 1:
                # subsetting an array that results in a single masked
                # element is silently converted to a tuple so we take
                # a more complicated route
                tsdata = ma.take(tsm.data, [i], axis=1)
                tsdata = ma.resize(tsdata, (1,))
                tsccy = ma.take(tsm.ccy, [i], axis=1)
                tsccy = ma.resize(tsccy, (1,))
                self.dateDict[d] = [
                    tsdata, tsccy,
                    numpy.ones(tsm.data.shape[0], dtype=float),
                    0]
            else:
                self.dateDict[d] = [
                    ma.array(tsm.data[:, i]), ma.array(tsm.ccy[:, i]),
                    numpy.ones(tsm.data[:,i].shape, dtype=float),
                    0]
            value = self.dateDict[d]
            assert(value[self.VAL_IDX].shape[0] == len(self.IDDict))
            assert(value[self.CCY_IDX].shape[0] == len(self.IDDict))
            assert(value[self.CCY_RATE_IDX].shape[0] == len(self.IDDict))
        # Update the currency exchange rates for the newly added dates
        # to match the currency of the rest (stored in self.idCurrency)
        for (sid, ccy) in self.idCurrency.items():
            if ccy is not None:
                self._updateCurrencyRates(sid, tsm.dates, ccy,
                                          currencyCache)
            
    def _getExchangeRate(self, dt, ccy, targetCurrency, currencyCache):
        rate = 1.0
        if ccy is not ma.masked and targetCurrency is not None and ccy != targetCurrency:
            rate = currencyCache.getRate(dt, ccy, targetCurrency)
            if rate is None:
                rate = 1.0
                logging.error('No exchange rate (%s,%s) on %s',
                              ccy, targetCurrency, dt)
        return rate
        
    def _updateCurrencyRates(self, sid, dates, targetCurrency, currencyCache):
        """Compute the exchange rate from local currency to targetCurrency
        for the given identifier and the list of dates.
        If dates is None, all dates are updated.
        """
        sidIdx = self.IDDict[sid]
        if dates is None:
            dates = list(self.dateDict.keys())
        for dt in dates:
            values = self.dateDict[dt]
            ccy = values[self.CCY_IDX][sidIdx]
            rate = self._getExchangeRate(dt, ccy, targetCurrency, currencyCache)
            values[self.CCY_RATE_IDX][sidIdx] = rate
    
    def _updateAllCurrencyRates(self, subids, currencyCache, convertTo):
        """Loop through all given subids and update their conversion
           rate
        """
        # Update currency conversions to convertTo
        for sid in subids:
            if not isinstance(convertTo, dict):
                targetCurrency = convertTo
            else:
                targetCurrency = convertTo[sid]
            currentCurrency = self.idCurrency.get(sid)
            if currentCurrency is not targetCurrency:
                self._updateCurrencyRates(sid, None, targetCurrency,
                                          currencyCache)
                self.idCurrency[sid] = targetCurrency

    def _setAllCurrencyRates(self, currencyCache, targetCurrency):
        """Faster code path that updates all assets/dates to
           the given targetCurrency (which must be a single currency)
        """
        assert(not isinstance(targetCurrency, dict))
        for subid in self.IDList:
            self.idCurrency[subid] = targetCurrency
        CCY_IDX = self.CCY_IDX
        CCY_RATE_IDX = self.CCY_RATE_IDX
        
        for dt, values in self.dateDict.items():
            currs = ma.unique(values[CCY_IDX]) 
            rates = values[CCY_RATE_IDX]
            for ccy in currs:
                if ccy is ma.masked:
                    continue
                rate = self._getExchangeRate(dt, ccy, targetCurrency, currencyCache)
                rates[values[CCY_IDX] == ccy] = rate

    def findMissingDates(self, dateList):
        return [d for d in dateList if d not in self.dateDict]
    
    def findMissingIDs(self, IDList):
        return [a for a in IDList if a not in self.IDDict]
    
    def getDateList(self):
        return list(self.dateDict.keys())
    
    def getIDList(self):
        return self.IDList
    
    def getSubMatrix(self, dateList, ids):
        tsm = Matrices.TimeSeriesMatrix(ids, dateList)
        if numpy.size(tsm.data) == 0:
            tsm.ccy = Matrices.allMasked(tsm.data.shape, dtype=int)
            return tsm
        tsm.data = Matrices.allMasked(tsm.data.shape)
        IDIndices = [self.IDDict[a] for a in ids]
        subMatrices = []
        subCcyMatrices = []
        for v in self.dateDict.values():
            v[self.AGE_IDX] += 1
        for dt in dateList:
            mat = self.dateDict[dt][self.VAL_IDX]
            ccy = self.dateDict[dt][self.CCY_IDX]
            self.dateDict[dt][self.AGE_IDX] = 0
            subMat = ma.take(mat, IDIndices, axis=0)
            subMat = ma.reshape(subMat, (subMat.shape[0], 1))
            subMatrices.append(subMat)
            subCcyMat = ma.take(ccy, IDIndices, axis=0)
            subCcyMat = ma.reshape(subCcyMat, (subCcyMat.shape[0], 1))
            subCcyMatrices.append(subCcyMat)
        tsm.data = ma.concatenate(subMatrices, axis=1)
        tsm.ccy = ma.concatenate(subCcyMatrices, axis=1)
        if len(self.dateDict) > self.maxDates:
            # Remove oldest dates
            ageList = sorted((-v[self.AGE_IDX], d) for (d, v) in self.dateDict.items())
            oldDates = ageList[:-(self.maxDates)]
            for (age, d) in oldDates:
                del self.dateDict[d]
            
        assert(len(tsm.data.shape) == 2)
        assert(len(tsm.ccy.shape) == 2)
        assert(tsm.data.shape[0] == len(tsm.assets))
        assert(tsm.data.shape[1] == len(tsm.dates))
        assert(tsm.ccy.shape[0] == len(tsm.assets))
        assert(tsm.ccy.shape[1] == len(tsm.dates))
        return tsm
    
    def getCurrencyConversions(self, dateList, subids,
                               currencyCache, convertTo):
        logging.debug('getCurrencyConversions: begin')
        if len(self.idCurrency) > 0 or isinstance(convertTo, dict): 
            self._updateAllCurrencyRates(subids, currencyCache, convertTo)
        else:
            self._setAllCurrencyRates(currencyCache, convertTo)

        subCcyMatrices = []
        IDIndices = [self.IDDict[a] for a in subids]
        for dt in dateList:
            ccyRates = self.dateDict[dt][self.CCY_RATE_IDX]
            subCcyMat = ma.take(ccyRates, IDIndices, axis=0)
            subCcyMat = ma.reshape(subCcyMat, (subCcyMat.shape[0], 1))
            subCcyMatrices.append(subCcyMat)
        ccyRates = ma.concatenate(subCcyMatrices, axis=1)
        logging.debug('getCurrencyConversions: end')
        return ccyRates

    def size(self):
        return len(self.IDList), len(self.dateDict)

    def to_pickle(self, fname):
        import pickle as pkl
        ids = self.IDList
        dates = sorted(self.dateDict.keys())
        tsm = self.getSubMatrix(dates, ids)
        maindf = tsm.toDataFrame()
        ccydf = pandas.DataFrame(tsm.ccy, index=maindf.index, columns=maindf.columns) 
        with open(fname, 'wb') as f:
            obj = {'main': maindf, 'ccy': ccydf, 'maxDates': self.maxDates}
            pkl.dump(obj, f, protocol=pkl.HIGHEST_PROTOCOL)
            
    @classmethod
    def from_pickle(cls, fname, currencyCache):
        import pickle as pkl
        result = None
        with open(fname, 'rb') as fh:
            result = pkl.load(fh)
        tsm = Matrices.TimeSeriesMatrix.fromDataFrame(result['main'])
        ccy = Matrices.TimeSeriesMatrix.fromDataFrame(result['ccy'])
        tsm.ccy = ccy.data.astype('int64')
        tsc = cls(result['maxDates'])
        emptytsm = Matrices.TimeSeriesMatrix([],tsm.dates)
        emptytsm.ccy = ma.masked_all(emptytsm.data.shape,dtype='int64')
        tsc.addMissingDates(emptytsm, currencyCache)
        tsc.addMissingIDs(tsm)
        return tsc

class FromThruCache:
    def __init__(self, useModelID=True):
        self.assetValueMap = dict()
        self.useModelID = useModelID
    
    def getAssetValue(self, axid, date):
        if self.useModelID and isinstance(axid, SubIssue):
            axid = axid.getModelID()
        changeValues = self.assetValueMap[axid]
        value = [val for val in changeValues
                 if val.fromDt <= date and val.thruDt > date]
        if len(value) == 0:
            value = None
        elif len(value) == 1:
            value = value[0]
        else:
            logging.fatal('%s, %s, %s', axid, date, changeValues)
            raise ValueError('more than one active record for %s on %s' % (
                axid, date))
        return value
    
    def getAssetHistory(self, axid):
        if self.useModelID and isinstance(axid, SubIssue):
            axid = axid.getModelID()
        return self.assetValueMap[axid]
    
    def addAssetValues(self, axid, valueList):
        self.assetValueMap[axid] = valueList
    def getMissingIds(self, axidList):
        return [a for a in axidList if a not in self.assetValueMap]

class FundamentalDataCache:
    def __init__(self, startDate, endDate, currencyCache):
        self.assetValueMap = dict()
        self.startDate = startDate
        self.endDate = endDate
        self.currencyCache = currencyCache
        self.log = logging.getLogger('ModelDB')
    
    def _convert(self, dt, val, srcCcy, tgtCcy, forceRun=False):
        if tgtCcy is None or srcCcy == tgtCcy:
            return (dt, val, srcCcy)
        rate = self.currencyCache.getRate(dt, srcCcy, tgtCcy)
        if rate is not None:
            val = val * rate
            return (dt, val, tgtCcy)
        self.log.error('No exchange rate (%d, %d) on %s', srcCcy, tgtCcy, dt)
        if not forceRun:
            assert rate is not None
        return None

    def getAssetHistory(self, sid, startDate, endDate, effDate, tgtCcy, forceRun=False):
        sidDtList = self.assetValueMap.get(sid)
        sidValues = []
        for (dt, values) in sidDtList:
            if dt < startDate or dt >  endDate:
                continue
            effectiveValues = [val for val in values if val[0] <= effDate]
            if len(effectiveValues) > 0 and effectiveValues[-1][1] == 'N':
                (val, ccyId) = effectiveValues[-1][2:4]
                convertVal = self._convert(dt, val, ccyId, tgtCcy, forceRun=forceRun)
                if convertVal is not None:
                    sidValues.append(convertVal)
        return sidValues
    
    def addAssetValues(self, axid, dataDateDict):
        self.assetValueMap[axid] = dataDateDict
    
    def getMissingIds(self, axidList):
        return [a for a in axidList if a not in self.assetValueMap]

class FundamentalDataCacheFY(FundamentalDataCache):
    """Same as FundamentalDataCache but also stores the 
    Fiscal Year End (month) corresponding to each data item.
    """
    def getAssetHistory(self, sid, startDate, endDate, effDate, tgtCcy, forceRun=False):
        sidDtList = self.assetValueMap.get(sid)
        sidValues = []
        for (dt, values) in sidDtList:
            if dt < startDate or dt >  endDate:
                continue
            effectiveValues = [val for val in values if val[0] <= effDate]
            if len(effectiveValues) > 0 and effectiveValues[-1][1] == 'N':
                (val, ccyId) = effectiveValues[-1][2:4]
                yearEnd = effectiveValues[-1][4]
                convertVal = self._convert(dt, val, ccyId, tgtCcy, forceRun=forceRun)
                if convertVal is not None:
                    sidValues.append(convertVal + (yearEnd,))
        return sorted(sidValues, key=lambda x: x[0])

class IBESDataCache:
    def __init__(self, startDate, endDate, currencyCache):
        self.assetValueMap = dict()
        self.startDate = startDate
        self.endDate = endDate
        self.currencyCache = currencyCache
        self.log = logging.getLogger('ModelDB')

    def _convert(self, dt, date, val, srcCcy, tgtCcy, effDate, forceRun=False):
        if tgtCcy is None or srcCcy == tgtCcy:
            return (dt, val, srcCcy, effDate)
        rate = self.currencyCache.getRate(date, srcCcy, tgtCcy)
        if rate is not None:
            val = val * rate
            return (dt, val, tgtCcy, effDate)
        self.log.error('No exchange rate (%d, %d) on %s', srcCcy, tgtCcy, date)
        if not forceRun:
            assert rate is not None
        return None

    def getAssetHistory(self, sid, startDate, endDate, date, tgtCcy, getEarliest=False, forceRun=False, namedTuple=None):
        sidDtList = self.assetValueMap.get(sid, list())
        sidValues = []
        for (dt, values) in sidDtList:
            #For estimate date, there can be cases that startDate, i.e. risk modelDate, 
            #is later than the dt, i.e. fiscal cutoff date
            #Here take the value even if the fiscal cutoff date is one year
            #later than the risk model date. Because as on that risk model date,
            #we may still be interested in forecast data that predicts the result
            #of past fiscal year
            if dt < startDate - datetime.timedelta(366) or dt >  endDate:
                continue
            effectiveValues = [val for val in values if val[0] <= date]
            if len(effectiveValues) > 0 and effectiveValues[-1][1] == 'N':
                if getEarliest:
                    effDate = effectiveValues[0][0]
                    (val, ccyId) = effectiveValues[0][2:4]
                else:
                    effDate = effectiveValues[-1][0]
                    (val, ccyId) = effectiveValues[-1][2:4]
                convertVal = self._convert(dt, date, val, ccyId, tgtCcy, effDate, forceRun=forceRun)
                if convertVal is not None:
                    if namedTuple is not None:
                        convertVal = namedTuple(*convertVal)
                    sidValues.append(convertVal)
        return sidValues

    def addAssetValues(self, axid, dataDateDict):
        self.assetValueMap[axid] = dataDateDict
    
    def getMissingIds(self, axidList):
        return [a for a in axidList if a not in self.assetValueMap]

class ModelDB:
    """Provides read/write access to the ModelDB database.
    """
    def __init__(self, revDateTime=None, **connectParameters):
        """Create connection to the database.
        """
        self.log = logging.getLogger('ModelDB')
        self.log.debug('connecting to ModelDB sid %s, user %s, password %s', connectParameters['sid'],connectParameters['user'],connectParameters['passwd'])
        self.log.debug('connectParameters %s', connectParameters)
        self.dbConnection = cx_Oracle.connect(
            connectParameters['user'], connectParameters['passwd'],
            connectParameters['sid'])
        self.dbCursor = self.dbConnection.cursor()
        self.dbCursor.execute(
            'alter session set nls_date_format="YYYY-MM-DD HH24:MI:SS"')
        # Uncomment this to enable session tracing
        #self.dbCursor.execute(
        #   "alter session set events='10046 trace name context forever, level 12'")
        self.dbConnection.commit()
        self.dbCursor.arraysize = 20000
        self.futureDate = '2999-12-31'
        self.flushCaches(revDateTime)
        self.forceRun = False
    
    def cursor(self):#for compatability with existing code
        return self.dbCursor
    def rollback(self):#for compatability with existing code
        self.dbConnection.rollback()
    
    def flushCaches(self, revDateTime=None):
        if revDateTime:
            self.revDateTime = revDateTime
        else:
            self.revDateTime = datetime.datetime.now()
        # Asset data caches
        self.estuQualifyCache = collections.defaultdict(lambda : TimeSeriesCache(90))
        self.marketCapCache = TimeSeriesCurrencyCache(90)
        self.ISCScoreCache = TimeSeriesCache(365)
        self.volumeCache = TimeSeriesCurrencyCache(302)
        self.ucpCache = TimeSeriesCurrencyCache(150)
        self.tsoCache = TimeSeriesCache(3400)
        self.robustWeightCache = TimeSeriesCache(20)
        # Factor caches
        self.factorReturnCache = collections.defaultdict(lambda : TimeSeriesCache(3000))
        # Asset market data caches here
        historyLength = 1505
        self.returnsTimingAdjustmentCache = collections.defaultdict(lambda : TimeSeriesCache(historyLength))
        self.returnsTimingAdjustmentV2Cache = collections.defaultdict(lambda : TimeSeriesCache(historyLength))
        self.returnsTimingProxyCache = collections.defaultdict(lambda : TimeSeriesCache(historyLength))
        self.specReturnCache = collections.defaultdict(lambda : TimeSeriesCache(historyLength))
        self.riskFreeRateCache = TimeSeriesCache(historyLength)
        self.totalReturnCache = TimeSeriesCache(historyLength)
        self.notTradedIndCache = TimeSeriesCache(historyLength)
        self.proxyReturnCache = TimeSeriesCache(historyLength)
        self.totalLocalReturnCache = TimeSeriesCache(302)
        self.cumReturnCache = TimeSeriesCache(120)
        # RMG-level caches
        self.marketReturnCache = TimeSeriesCache(1502)
        self.marketReturnV3Cache = TimeSeriesCache(1502)
        self.marketVolatilityCache = TimeSeriesCache(1502)
        self.regionReturnCache = TimeSeriesCache(1502)
        self.totalRMGReturnCache = TimeSeriesCache(302)
        self.cumRMGReturnCache = TimeSeriesCache(120)
        # Others
        self.divyieldCache = None
        self.histBetaCacheOld = None
        self.histBetaCache = None
        self.histBetaCacheFixed = None
        self.histBetaVersn3Cache = None
        self.histRobustBetaCache = None
        self.histBetaPValueCache = None
        self.histBetaNObsCache = None
        self.descriptorCache = dict()
        self.shareAFCache = ValueHistoryCache()
        self.cdivCache = ValueHistoryCache()
        self.currencyCache = None
        self.fmpCache = None
        self.modelClassificationCaches = {}
        self.marketClassificationCaches = {}
        self.ibesUSCache = FromThruCache()
        self.gvkeyUSCache = FromThruCache()
        self.cusipCache = FromThruCache()
        self.sedolCache = FromThruCache()
        self.isinCache = FromThruCache()
        self.ricCache = FromThruCache()
        self.nameCache = FromThruCache()
        self.gvkeyCache = FromThruCache()
        self.tickerCache = FromThruCache()
        self.tradeCcyCache = FromThruCache()
        self.companyCache = FromThruCache()
        self.issueFromDateCache = FromThruCache(useModelID=False)
        self.freeFloatCache = FromThruCache()
        self.tradingStatusCache = FromThruCache()
        self.stockConnectBuySellCache = FromThruCache()
        self.stockConnectSellOnlyCache = FromThruCache()
        self.issueMapCache = dict()
        self.fundamentalDataCache = dict()
        self.ibesDataCache = dict()
        self.allRMGCache = dict() # cache used for getAllRiskModelGroups
        self.DRtoUnderlyingCache = dict()
        self.mdlPortReturnCache = TimeSeriesCache(2500)
   
    def setDescriptorCache(self, name, days):

        if not hasattr(self, 'descCacheMap'):
            self.descCacheMap = dict()
        self.descCacheMap[name] = TimeSeriesCache(days)
        return

    def saveCaches(self, dirname='modeldb_pickles'):
        caches = ['totalReturnCache', 'volumeCache', 'marketCapCache', 'notTradedIndCache', 'proxyReturnCache']
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        for cache_name in caches:
            cache = getattr(self, cache_name, None)
            if cache is not None:
                fname = os.path.join(dirname, cache_name + '.pickle')
                cache.to_pickle(fname)

    def loadCaches(self, dirname='modeldb_pickles'):
        caches = ['totalReturnCache', 'volumeCache', 'marketCapCache', 'notTradedIndCache', 'proxyReturnCache']
        if self.currencyCache is None:
            raise Exception('CurrencyCache should not be None')
        for cache_name in caches:
            cache = getattr(self, cache_name, None)
            if cache is not None:
                fname = os.path.join(dirname, cache_name + '.pickle')
                try:
                    newcache = cache.from_pickle(fname, self.currencyCache)
                except:
                    newcache = cache
                setattr(self, cache_name, newcache)

    def setESTUQualifyCache(self, days):
        self.estuQualifyCache = collections.defaultdict(lambda : TimeSeriesCache(days))
    
    def setFactorReturnCache(self, days):
        self.factorReturnCache = collections.defaultdict(lambda : TimeSeriesCache(days))
    
    def setFactorReturnAdjustmentCache(self, days):
        self.factorReturnAdjustmentCache = TimeSeriesCache(days)
    
    def setSpecReturnCache(self, days):
        self.specReturnCache = collections.defaultdict(
            lambda : TimeSeriesCache(days))
    
    def setMarketCapCache(self, days, shrink=False):
        if not shrink and self.marketCapCache is not None\
               and self.marketCapCache.maxDates >= days:
            self.log.debug('Leaving cache unchanged: current cache size %d,'
                           ' requested %d', self.marketCapCache.maxDates,
                           days)
            return
        self.marketCapCache = TimeSeriesCurrencyCache(days)

    def setRiskFreeRateCache(self, days):
        self.riskFreeRateCache = TimeSeriesCache(days)
    
    def setTotalReturnCache(self, days, shrink=False):
        if not shrink and self.totalReturnCache is not None\
               and self.totalReturnCache.maxDates >= days:
            self.log.debug('Leaving cache unchanged: current cache size %d,'
                           ' requested %d', self.totalReturnCache.maxDates,
                           days)
            return
        self.totalReturnCache = TimeSeriesCache(days)

    def setNotTradedIndCache(self, days, shrink=False):
        if not shrink and self.notTradedIndCache is not None\
                and self.notTradedIndCache.maxDates >= days:
            self.log.debug('Leaving cache unchanged: current cache size %d,'
                    ' requested %d', self.notTradedIndCache .maxDates, days)
            return
        self.notTradedIndCache = TimeSeriesCache(days)
    
    def setProxyReturnCache(self, days, shrink=False):
        if not shrink and self.proxyReturnCache is not None\
               and self.proxyReturnCache.maxDates >= days:
            self.log.debug('Leaving cache unchanged: current cache size %d,'
                           ' requested %d', self.proxyReturnCache.maxDates,
                           days)
            return
        self.proxyReturnCache = TimeSeriesCache(days)

    def setISCScoreCache(self, days, shrink=False):
        if not shrink and self.ISCScoreCache is not None\
               and self.ISCScoreCache.maxDates >= days:
            self.log.debug('Leaving cache unchanged: current cache size %d,'
                           ' requested %d', self.ISCScoreCache.maxDates,
                           days)
            return
        self.ISCScoreCache = TimeSeriesCache(days)

    def setHistBetaCache(self, days, shrink=False):
        if not shrink and self.histBetaCache is not None\
               and self.histBetaCache.maxDates >= days:
            self.log.debug('Leaving cache unchanged: current cache size %d,'
                           ' requested %d', self.histBetaCache.maxDates,
                           days)
        else:
            self.histBetaCache = TimeSeriesCache(days)
        if not shrink and self.histBetaCacheOld is not None\
               and self.histBetaCacheOld.maxDates >= days:
            self.log.debug('Leaving cache unchanged: current cache size %d,'
                           ' requested %d', self.histBetaCacheOld.maxDates,
                           days)
        else:
            self.histBetaCacheOld = TimeSeriesCache(days)
        if not shrink and self.histBetaCacheFixed is not None\
                and self.histBetaCacheFixed.maxDates >= days:
            self.log.debug('Leaving cache unchanged: current cache size %d,'
                           ' requested %d', self.histBetaCacheFixed.maxDates,
                           days)
        else:
            self.histBetaCacheFixed = TimeSeriesCache(days)
    
    def setDividendYieldCache(self, days, shrink=False):
        if not shrink and self.divyieldCache is not None\
               and self.divyieldCache.maxDates >= days:
            self.log.debug('Leaving cache unchanged: current cache size %d,'
                           ' requested %d', self.divyieldCache.maxDates,
                           days)
        else:
            self.divyieldCache = TimeSeriesCache(days)

    def setHistBetaV3Cache(self, days, shrink=False):
        cacheList = [self.histBetaVersn3Cache, self.histRobustBetaCache,
                     self.histBetaPValueCache, self.histBetaNObsCache]
        for betaCache in cacheList:
            if not shrink and betaCache is not None\
                    and betaCache.maxDates >= days:
                self.log.debug('Leaving cache unchanged: current cache size %d,'
                        ' requested %d', betaCache, days)
            else:
                betaCache = TimeSeriesCache(days)

    def setCumReturnCache(self, days, shrink=False):
        if not shrink and self.cumReturnCache is not None\
               and self.cumReturnCache.maxDates >= days:
            self.log.debug('Leaving cache unchanged: current cache size %d,'
                           ' requested %d', self.cumReturnCache.maxDates,
                           days)
            return
        self.cumReturnCache = TimeSeriesCache(days)
    
    def setVolumeCache(self, days, shrink=False):
        if not shrink and self.volumeCache is not None\
               and self.volumeCache.maxDates >= days:
            self.log.debug('Leaving cache unchanged: current cache size %d,'
                           ' requested %d', self.volumeCache.maxDates,
                           days)
            return
        self.volumeCache = TimeSeriesCurrencyCache(days)
    
    def createCurrencyCache(self, marketDB, numeraire=1, days=2002, lookBack=20, boT=None, shrink=False):
        if shrink or self.currencyCache is None or self.currencyCache.currencyProvider.maxNumConverters < days:
            self.currencyCache = ForexCache(marketDB, days)
    
    def finalize(self):
        """Close connection to the database.
        """
        self.dbCursor.close()
        self.dbConnection.close()
    
    def annualizeQuarterlyValues(self, dateValues, adjustShortHistory=False, requireAnnualWindow=False):
        """For each list in dateValues, retrieve the most recent
        ~4 entries (verifying that the date associated with each
        entry is within one year of the last entry, thereby ensuring
        that these entries correspond to the "most recent year"), and 
        take the sum of these entries.
        The return value is a masked array of the averaged values.
        The entries in each list are assumed to be (date, value) tuples
        possibly with additional members like currency.
        Note that there is no logic for handling currencies at present.
        """
        annualized = ma.zeros(len(dateValues))
        for i in range(len(dateValues)):
            if len(dateValues[i]) == 0:
                annualized[i] = ma.masked
            else:
                if requireAnnualWindow:
                    vals = [dateValues[i][j][1]
                            for j in range(len(dateValues[i]))
                            if (dateValues[i][-1][0] - dateValues[i][j][0]).days < 365]
                    val = numpy.sum(vals)
                    if adjustShortHistory:
                        val *= 4.0 / len(vals)
                else:
                    val = dateValues[i][-1][1]
                    date = dateValues[i][-1][0]
                    prev3 = dateValues[i][-4:-1]
                    val0 = val
                    for pval in prev3:
                        (d, inc) = pval[:2]
                        if (date - d).days <= 15 * 31:
                            val += inc
                    if adjustShortHistory:
                        val *= 4.0 / (len(prev3) + 1)
                annualized[i] = val
        return annualized
    
    def averageQuarterlyValues(self, dateValues):
        """For each list in dateValues, retrieve the most recent
        ~4 entries (verifying that the date associated with each
        entry is within one year of the last entry, thereby ensuring
        that these entries correspond to the "most recent year"), and 
        take the average of these entries.
        The return value is a masked array of the averaged values.
        The entries in each list are assumed to be (date, value) tuples
        possibly with additional members like currency.
        Note that there is no logic for handling currencies at present.
        """
        averaged = ma.zeros(len(dateValues))
        for i in range(len(dateValues)):
            if len(dateValues[i]) == 0:
                averaged[i] = ma.masked
            else:
                averaged[i] = numpy.average([dateValues[i][j][1] 
                                             for j in range(len(dateValues[i])) 
                                             if (dateValues[i][-1][0] - dateValues[i][j][0]).days < 365])
        return averaged

    def averageAnnualValues(self, dateValues):
        """For each list in dateValues, retrieve the most recent
        ~2 entries (verifying that the date associated with each
        entry is within one year of the last entry, thereby ensuring
        that these entries correspond to the "most recent year"), and 
        take the average of these entries.
        The return value is a masked array of the averaged values.
        The entries in each list are assumed to be (date, value) tuples
        possibly with additional members like currency.
        Note that there is no logic for handling currencies at present.
        """
        averaged = ma.zeros(len(dateValues))
        for i in range(len(dateValues)):
            if len(dateValues[i]) == 0:
                averaged[i] = ma.masked
            else:
                averaged[i] = numpy.average([dateValues[i][j][1] 
                                             for j in range(len(dateValues[i])) 
                                             if (dateValues[i][-1][0] - dateValues[i][j][0]).days < 370])
        return averaged

    def commitChanges(self):
        """Commit pending changes to the database.
        """
        self.dbConnection.commit()
    
    def createNewIssue(self, newIssues, currentDate):
        """Create new Axioma issues and link them to the given MarketDB
        issues as of the current date.
        For each new issues, create a new entry in the issue
        and issue_map tables.
        The newIssues argument is a list of (ModelID, MarketDB ID) pairs.
        """
        if len(newIssues) == 0:
            return
        
        newIDDicts = [dict([('newID_arg', newID.modeldb_id.getIDString()),
                            ('from_arg', currentDate),
                            ('thru_arg', self.futureDate)])
                      for newID in newIssues]
        self.dbCursor.executemany("""INSERT INTO issue VALUES(:newID_arg,
        :from_arg, :thru_arg)""", newIDDicts)
        bothIDsDicts = [dict([('newID_arg', newID.modeldb_id.getIDString()),
                              ('from_arg', currentDate),
                              ('thru_arg', self.futureDate),
                              ('marketID_arg', newID.marketdb_id)])
                        for newID in newIssues]
        self.dbCursor.executemany("""INSERT INTO issue_map VALUES(:newID_arg,
        :marketID_arg, :from_arg, :thru_arg)""", bothIDsDicts)
    
    def createNewSubIssue(self, subIssues, rmg, currentDate):
        """Enter the given sub-issues into the sub_issue table for the given
        risk model group, starting at the given date.
        """
        if len(subIssues) == 0:
            return
        IDDicts = [dict([('issue_arg', i.getModelID().getIDString()),
                         ('from_arg', currentDate),
                         ('thru_arg', self.futureDate),
                         ('rmg_arg', rmg.rmg_id),
                         ('subissue_arg', i.getSubIDString())])
                   for i in subIssues]
        self.dbCursor.executemany("""INSERT INTO sub_issue
        VALUES(:issue_arg, :from_arg, :thru_arg, :subissue_arg, :rmg_arg)""",
                                  IDDicts)
    
    def createNewModelIDs(self, numIDs, dummy=0):
        """Create a list of numIDs new ModelIDs."""
        self.dbCursor.execute('LOCK TABLE last_issue IN EXCLUSIVE MODE')
        self.dbCursor.execute("""SELECT last_issue_counter FROM last_issue
        WHERE dummy=:dummy_arg""", dummy_arg=dummy)
        lastCounter = self.dbCursor.fetchall()[0][0]
        newIDs = [ModelID.ModelID(index=i + lastCounter)
                  for i in range(numIDs)]
        lastCounter += numIDs
        self.dbCursor.execute("""UPDATE last_issue
        SET last_issue_counter = :counter_arg WHERE dummy = :dummy_arg""",
                              counter_arg=lastCounter, dummy_arg=dummy)
        return newIDs
    
    def createRiskModelInstance(self, rms_id, date):
        """Create a new risk model instance for the given serial ID
        and date.
        Returns the instance ID.
        """
        self.dbCursor.execute("""INSERT INTO risk_model_instance
        (rms_id, dt, has_exposures, has_returns, has_risks, is_final, update_dt)
        VALUES (:rms_arg, :date_arg, 0, 0, 0, 1, :upd_dt_arg)""",
              rms_arg=rms_id, date_arg=date, upd_dt_arg=datetime.datetime.now())
        return self.getRiskModelInstance(rms_id, date)
    
    def createRiskModelSerie(self, rms_id, rm_id, revision):
        """Create a new risk model serie for the given serie, model,
        and revision.
        """
        self.dbCursor.execute("""INSERT INTO risk_model_serie
        (serial_id, rm_id, revision)
        VALUES (:serial_arg, :rm_arg, :rev_arg)""",
                              serial_arg=rms_id, rm_arg=rm_id,
                              rev_arg=revision)
    
    def deactivateIssues(self, deadIssues, currentDate):
        """Mark issues as deactivated by setting the thru_dt of the
        currently active entry to the given date.
        """
        query = """UPDATE issue SET thru_dt = :thru_arg
        WHERE issue_id = :issue_arg AND thru_dt > :thru_arg"""
        for issue in deadIssues:
            self.dbCursor.execute(query, issue_arg=issue.getIDString(),
                                  thru_arg=currentDate)
        query = """UPDATE issue_map SET thru_dt = :thru_arg
        WHERE modeldb_id = :issue_arg AND thru_dt > :thru_arg"""
        for issue in deadIssues:
            self.dbCursor.execute(query, issue_arg=issue.getIDString(),
                                  thru_arg=currentDate)
    
    def deactivateSubIssues(self, deadSubIssues, rmg, currentDate):
        """Mark sub-issues as deactivated by setting the thru_dt of the
        currently active entry to the given date.
        """
        query = """UPDATE sub_issue SET thru_dt=:thru_arg
        WHERE sub_id=:issue_arg AND thru_dt > :thru_arg AND rmg_id=:rmg_arg"""
        for subIssue in deadSubIssues:
            self.dbCursor.execute(query, issue_arg=subIssue.getSubIDString(),
                                  thru_arg=currentDate,
                                  rmg_arg=rmg.rmg_id)
    
    def deleteRMIExposureMatrix(self, rmi):
        """Deletes exposure data associated with the given risk model 
        instance from rmi_factor_exposure.
        """
        self.dbCursor.execute("""DELETE FROM rmi_factor_exposure
        WHERE rms_id = :rms_arg AND dt=:date_arg""", rms_arg=rmi.rms_id,
                              date_arg=rmi.date)
    
    def deleteProxyReturns(self, subIssues, dateList):
        """Routine to delete a set of subids from the proxy return table
        for a particular date
        """
        if not isinstance(dateList, list):
            dateList = [dateList]

        for date in dateList:
            if len(subIssues) > 0:
                deleteDicts = [dict([
                    ('sub_issue_id', sid.getSubIDString()), ('dt', date)
                    ]) for sid in subIssues]
                self.dbCursor.executemany("""DELETE FROM rmg_proxy_return
                        WHERE sub_issue_id=:sub_issue_id AND dt=:dt""", deleteDicts)
                self.log.info('Deleting %d records for %s', len(deleteDicts), date)
            else:
                self.dbCursor.execute("""DELETE FROM rmg_proxy_return
                        WHERE dt=:dt""", dt=date)
                self.log.info('Deleting ALL records for %s', date)

    def deleteISCScores(self, subIssues, date):
        """Routine to delete a set of subids from the ISC asset score table
        for a particular date
        """
        if len(subIssues) > 0:
            deleteDicts = [dict([
                ('sub_issue_id', sid.getSubIDString()), ('dt', date)
                ]) for sid in subIssues]
            self.dbCursor.executemany("""DELETE FROM rmg_isc_score
                    WHERE sub_issue_id=:sub_issue_id AND dt=:dt""", deleteDicts)
            self.log.info('Deleting %d records for %s', len(deleteDicts), date)
        else:
            self.dbCursor.execute("""DELETE FROM rmg_isc_score
                    WHERE dt=:dt""", dt=date)
            self.log.info('Deleting ALL records for %s', date)

    def deleteCointegrationParameters(self, date):
        """Deletes cointegration betas and p-values for all asset pairs
        on a particular date
        """
        self.dbCursor.execute("""DELETE FROM rmg_isc_beta
                WHERE dt=:date_arg""", date_arg=date)
        self.dbCursor.execute("""DELETE FROM rmg_isc_pvalue
                WHERE dt=:date_arg""", date_arg=date)

    def deleteRMIExposureMatrixNew(self, rmi, subIssues=[]):
        """Deletes exposure data associated with the given risk model 
        instance from rmi_factor_exposure.
        """
        tableName = self.getWideExposureTableName(rmi.rms_id)
        if len(subIssues) > 0:
            query = """DELETE FROM %(table)s
            WHERE sub_issue_id=:sub_issue_id AND dt=:date_arg""" % { 'table': tableName }
            argDict = [dict([
                ('sub_issue_id', sid.getSubIDString()), ('date_arg', rmi.date)
                ]) for sid in subIssues]
            logging.info('Deleting %d exposures', len(argDict))
            self.dbCursor.executemany(query, argDict)
        else:
            self.dbCursor.execute("""DELETE FROM %(table)s
            WHERE dt=:date_arg""" % { 'table': tableName },
                                date_arg=rmi.date)
            try:
                self.dbCursor.execute("""DELETE FROM rms_stnd_exp
                WHERE dt=:date_arg and rms_id=:rms_arg""",
                            date_arg=rmi.date, rms_arg=rmi.rms_id)
            except:
                self.log.warning('No standardisation stats for exposure matrix. Skipping')
            try:
                self.dbCursor.execute("""DELETE FROM rms_stnd_desc
                WHERE dt=:date_arg and rms_id=:rms_arg""",
                            date_arg=rmi.date, rms_arg=rmi.rms_id)
            except:
                self.log.warning('No standardisation stats for descriptor matrix. Skipping')
            try:
                self.dbCursor.execute("""DELETE FROM rms_stnd_mean 
                WHERE dt=:date_arg and rms_id=:rms_arg""",
                            date_arg=rmi.date, rms_arg=rmi.rms_id)
            except:
                self.log.warning('No standardisation stats for descriptor matrix. Skipping')
            try:
                self.dbCursor.execute("""DELETE FROM rms_stnd_stdev 
                WHERE dt=:date_arg and rms_id=:rms_arg""",
                            date_arg=rmi.date, rms_arg=rmi.rms_id)
            except:
                self.log.warning('No standardisation stats for descriptor matrix. Skipping')
    
    def getWideExposureTableName(self, rmsID):
        if rmsID >= 0:
            rmsTag = '%.2d' % rmsID
        else:
            rmsTag = 'M%.2d' % abs(rmsID)
        return 'rms_%s_factor_exposure' % rmsTag
    
    def getSubFactorColumnName(self, subFactor):
        return 'sf_%d' % ( subFactor.subFactorID)
    
    def deleteRMIFactorSpecificRisk(self, rmi):
        """Deletes factor and specific risk variance/covariance data associated
        with the given risk model instance.
        Data is removed from rmi_covariance and rmi_specific_risk.
        """
        self.dbCursor.execute("""DELETE FROM rmi_specific_risk
        WHERE rms_id = :rms_arg AND dt=:date_arg""", rms_arg=rmi.rms_id,
                              date_arg=rmi.date)
        self.dbCursor.execute("""DELETE FROM rmi_specific_covariance
        WHERE rms_id = :rms_arg AND dt=:date_arg""", rms_arg=rmi.rms_id,
                              date_arg=rmi.date)
        self.dbCursor.execute("""DELETE FROM rmi_covariance
        WHERE rms_id = :rms_arg AND dt=:date_arg""", rms_arg=rmi.rms_id,
                              date_arg=rmi.date)

    def deleteRMIFactorCovMatrix(self, rmi):
        """Deletes factor covariance data associated
        with the given risk model instance.
        Data is removed from rmi_covariance
        """
        self.dbCursor.execute("""DELETE FROM rmi_covariance
                WHERE rms_id = :rms_arg AND dt=:date_arg""", rms_arg=rmi.rms_id,
                date_arg=rmi.date)
    
    def deleteRiskModelUniverse(self, rmi, subIssues=[]):
        """Deletes asset records from rmi_universe for the given
        risk model instance.
        """
        if len(subIssues) > 0:
            valueDicts = [dict([('rms_arg', rmi.rms_id),
                                ('date_arg', rmi.date),
                                ('id_arg', sid.getSubIDString())])
                                for sid in subIssues]
            self.dbCursor.executemany("""DELETE FROM rmi_universe
            WHERE rms_id = :rms_arg AND dt=:date_arg
            AND sub_issue_id = :id_arg""", valueDicts)
            logging.info('Deleting %d records from model universe', len(valueDicts))
        else:
            self.dbCursor.execute("""DELETE FROM rmi_universe
            WHERE rms_id = :rms_arg AND dt=:date_arg""",
                                rms_arg=rmi.rms_id, date_arg=rmi.date)

    def flushRiskModelSerieData(self, rms_id, newExposureFormat,
                                deleteUniverse=True, 
                                deleteExposures=True, 
                                deleteFactors=True):
        """Bulk delete of all model data associated with 
        the given risk model serie.
        """
        if rms_id < 0:
            partition = 'RM%.2d' % (-1 * rms_id)
        else:
            partition = 'R%.2d' % rms_id
        delete_flag = False
        if deleteUniverse:
            self.log.info('Deleting risk model instances and universe/ESTU')
            # Risk model universe
            self.dbCursor.execute("""ALTER TABLE rmi_universe
                    TRUNCATE PARTITION p_univ_%s""" % partition)
            # Estimation universe
            self.dbCursor.execute("""ALTER TABLE rmi_estu
                    TRUNCATE PARTITION p_estu_%s""" % partition)
            try:
                self.dbCursor.execute("""ALTER TABLE rmi_estu_v3
                        TRUNCATE PARTITION p_estu_%s""" % partition)
            except:
                logging.warning('No partition p_estu_%s for rmi_estu_v3', partition)
                self.dbCursor.execute("""DELETE FROM rmi_estu_v3
                WHERE rms_id = :rms_arg""", rms_arg=rms_id)
            
            # Risk model instances
            self.dbCursor.execute("""DELETE FROM risk_model_instance
            WHERE rms_id = :rms_arg""", rms_arg=rms_id)
            delete_flag = True
        if deleteExposures or delete_flag:
            self.log.info('Deleting factor exposures')
            # Exposures
            if newExposureFormat:
                self.dbCursor.execute("""TRUNCATE TABLE %s"""
                                      % self.getWideExposureTableName(rms_id))
                try:
                    self.dbCursor.execute("""DELETE FROM rms_stnd_mean
                            WHERE rms_id = :rms_arg""", rms_arg=rms_id)
                    self.dbCursor.execute("""DELETE FROM rms_stnd_stdev
                            WHERE rms_id = :rms_arg""", rms_arg=rms_id)
                except:
                    self.log.warning('No standardisation stats for exposure matrix. Skipping')
                try:
                    self.dbCursor.execute("""DELETE FROM rms_stnd_desc
                            HERE rms_id = :rms_arg""", rms_arg=rms_id)
                except:
                    try:
                        self.dbCursor.execute("""DELETE FROM rms_stnd_stats
                            WHERE rms_id = :rms_arg""", rms_arg=rms_id)
                    except:
                        self.log.warning('No standardisation stats for descriptor matrix. Skipping')
            else:
                self.dbCursor.execute("""ALTER TABLE rmi_factor_exposure 
                    TRUNCATE PARTITION p_exposure_%s""" % partition)
            delete_flag = True
            # Risk model instances
            self.dbCursor.execute("""UPDATE risk_model_instance SET has_exposures = 0 WHERE rms_id = :rms_arg""",
                                  rms_arg=rms_id)
            delete_flag = True
        if deleteFactors or delete_flag:
            self.log.info('Deleting factor/specific returns and regression statistics')
            # Factor returns
            self.dbCursor.execute("""DELETE FROM rms_factor_return
                    WHERE rms_id = :rms_arg""", rms_arg=rms_id)
            try:
                self.dbCursor.execute("""DELETE FROM rms_factor_return_internal
                        WHERE rms_id = :rms_arg""", rms_arg=rms_id)
            except:
                logging.info('No factor return internal table')
            try:
                self.dbCursor.execute("""DELETE FROM rms_factor_return_weekly
                        WHERE rms_id = :rms_arg""", rms_arg=rms_id)
            except:
                logging.info('No factor return weekly table')
            try:
                if rms_id < 0:
                    self.dbCursor.execute("""TRUNCATE TABLE rms_m%.2d_stat_factor_return""" % abs(rms_id))
                else:
                    self.dbCursor.execute("""TRUNCATE TABLE rms_%s_stat_factor_return""" % rms_id)
            except:
                logging.info('No statistical factor return table')
            # Specific returns
            self.dbCursor.execute("""ALTER TABLE rms_specific_return 
                    TRUNCATE PARTITION p_specret_%s""" % partition)
            try:
                self.dbCursor.execute("""ALTER TABLE rms_specific_return_internal
                    TRUNCATE PARTITION p_sprtint_%s""" % partition)
            except:
                logging.info('No p_sprtint_%s partition in rms_specific_return_internal table', partition)
                self.dbCursor.execute("""DELETE FROM rms_specific_return_internal WHERE rms_id = :rms_arg""", rms_arg=rms_id)
            # R-squared
            self.dbCursor.execute("""DELETE FROM rms_statistics
                    WHERE rms_id = :rms_arg""", rms_arg=rms_id)
            self.dbCursor.execute("""DELETE FROM rms_statistics_internal
                    WHERE rms_id = :rms_arg""", rms_arg=rms_id)
            try:
                self.dbCursor.execute("""DELETE FROM rms_statistics_weekly
                        WHERE rms_id = :rms_arg""", rms_arg=rms_id)
            except:
                logging.info('No rms_statistics_weekly table')
            # T-stats
            self.dbCursor.execute("""DELETE FROM rms_factor_statistics
                    WHERE rms_id = :rms_arg""", rms_arg=rms_id)
            self.dbCursor.execute("""DELETE FROM rms_factor_statistics_internal
                    WHERE rms_id = :rms_arg""", rms_arg=rms_id)
            try:
                self.dbCursor.execute("""DELETE FROM rms_factor_statistics_weekly
                        WHERE rms_id = :rms_arg""", rms_arg=rms_id)
            except:
                logging.info('No rms_factor_statistics_weekly table')
            # Risk model instances
            self.dbCursor.execute("""UPDATE risk_model_instance SET has_returns = 0 WHERE rms_id = :rms_arg""",
                                  rms_arg=rms_id)
            delete_flag = True
        self.log.info('Deleting robust regression weights')
        try:
            self.dbCursor.execute("""ALTER TABLE rms_robust_weight
                    TRUNCATE PARTITION p_robustwt_%s""" % partition)
        except:
            self.dbCursor.execute("""DELETE FROM rms_robust_weight
                    WHERE rms_id = :rms_arg""", rms_arg=rms_id)
            self.log.warning('No %s partition in rms_robust_weight', partition)
        try:
            self.dbCursor.execute("""ALTER TABLE rms_robust_weight_internal
                    TRUNCATE PARTITION p_robwtint_%s""" % partition)
        except:
            self.dbCursor.execute("""DELETE FROM rms_robust_weight_internal
                    WHERE rms_id = :rms_arg""", rms_arg=rms_id)
            self.log.warning('No %s partition in rms_robust_weight_internal', partition)
        try:
            self.dbCursor.execute("""ALTER TABLE rms_robust_weight_weekly
                    TRUNCATE PARTITION p_robwtwkly_%s""" % partition)
        except:
            #self.dbCursor.execute("""DELETE FROM rms_robust_weight_weekly
            #        WHERE rms_id = :rms_arg""", rms_arg=rms_id)
            self.log.warning('No %s partition in rms_robust_weight_weekly', partition)

        self.log.info('Deleting FMPs')
        if rms_id < 0:
            table = 'rms_m%.2d_fmp' % abs(rms_id)
        else:
            table = 'rms_%s_fmp' % rms_id
        try:
            self.dbCursor.execute("""TRUNCATE TABLE %s""" % table)
        except:
            self.log.warning('No FMP table: %s', table)
        self.log.info('Deleting factor/specific risks and covariances')
        # Factor covariances
        self.dbCursor.execute("""ALTER TABLE rmi_covariance
                TRUNCATE PARTITION p_fcov_%s""" % partition)
        self.dbCursor.execute("""DELETE FROM rms_dva_statistics WHERE rms_id = :rms_arg""", rms_arg=rms_id)
        # Specific risk
        self.dbCursor.execute("""ALTER TABLE rmi_specific_risk
                TRUNCATE PARTITION p_specrisk_%s""" % partition)
        try:
            self.dbCursor.execute("""ALTER TABLE rmi_specific_covariance
                TRUNCATE PARTITION p_speccov_%s""" % partition)
        except:
            self.log.warning('Specific covariance partition does not exist. skipping')
        # Total risk
        self.dbCursor.execute("""ALTER TABLE rmi_total_risk
                TRUNCATE PARTITION p_totalrisk_%s""" % partition)
        # Predicted betas
        self.dbCursor.execute("""ALTER TABLE rmi_predicted_beta
                TRUNCATE PARTITION p_predbeta_%s""" % partition)
        try:
            # New V3 table
            self.dbCursor.execute("""ALTER TABLE rmi_predicted_beta_v3
                TRUNCATE PARTITION p_predbeta_%s""" % partition)
        except:
            pass # no V3 partitions
    
        # Risk model instances
        self.dbCursor.execute("""UPDATE risk_model_instance SET has_risks = 0, update_dt=:updateDt
                WHERE rms_id = :rms_arg""", rms_arg=rms_id, updateDt=datetime.datetime.now())

    def deleteEstimationUniverse(self, rmi):
        """Deletes the estimation of the given risk model instance
        from the database.
        """
        logging.info('Deleting estimation universe')
        self.dbCursor.execute("""DELETE FROM rmi_estu WHERE rms_id = :rms_arg AND dt = :date_arg""",
                rms_arg=rmi.rms_id, date_arg=rmi.date)
        estuMap = self.getEstuMappingTable(rmi.rms_id)
        if estuMap is not None:
            for estuName in estuMap.keys():
                self.dbCursor.execute("""DELETE FROM rmi_estu_v3
                        WHERE rms_id = :rms_arg AND dt = :date_arg AND nested_id = :id_arg""",
                        rms_arg=rmi.rms_id, date_arg=rmi.date, id_arg=estuMap[estuName].id)
    
    def deleteFactorReturns(self, rmi, subFactors=None, flag=None):
        """Deletes the internal (model-only) factor returns of the given risk model instance
        from the database. If subFactors is specified, only those factor
        returns are removed.
        """
        if flag is None:
            tableName = 'rms_factor_return'
        else:
            tableName = 'rms_factor_return_%s' % flag
        if subFactors is None:
            logging.info('Deleting all factor returns from %s for %s', tableName, rmi.date)
            self.dbCursor.execute("""DELETE FROM %(table_arg)s
                    WHERE rms_id = :rms_arg AND dt = :date_arg""" % {'table_arg': tableName},
                    rms_arg=rmi.rms_id, date_arg=rmi.date)
        elif len(subFactors) > 0:
            logging.debug('Deleting %d %s factor returns for %s', 
                    len(subFactors), flag, rmi.date)
            valueDicts = [dict([('rms_arg', rmi.rms_id),
                            ('date_arg', rmi.date),
                            ('id_arg', f.subFactorID)])
                        for f in subFactors]
            self.dbCursor.executemany("""DELETE FROM %(table_arg)s
                    WHERE rms_id = :rms_arg AND dt=:date_arg
                    AND sub_factor_id = :id_arg""" % {'table_arg': tableName}, valueDicts)
        else:
            logging.warning('Empty list of subfactors for deletion')

    def deleteRawFactorReturns(self, rmi, subFactors=None):
        """Deletes the raw factor returns of the given risk model instance
        from the database. If subFactors is specified, only those raw factor
        returns are removed.
        """
        if subFactors:
            valueDicts = [dict([('rms_arg', rmi.rms_id),
                                ('date_arg', rmi.date),
                                ('id_arg', f.subFactorID)])
                          for f in subFactors]
            self.dbCursor.executemany("""DELETE FROM rms_raw_factor_return
            WHERE rms_id = :rms_arg AND dt=:date_arg 
            AND sub_factor_id = :id_arg""", valueDicts)
        else:
            self.dbCursor.execute("""DELETE FROM rms_raw_factor_return
            WHERE rms_id = :rms_arg AND dt = :date_arg""", rms_arg=rmi.rms_id,
                                  date_arg=rmi.date)

    def deleteStatFactorReturns(self, rmi):
        """Deletes the factor returns of the given statistical risk model instance
        from the database.
        """

        if rmi.rms_id < 0:
            rms_id_str = 'm%.2d' % abs(rmi.rms_id)
        else:
            rms_id_str = rmi.rms_id

        try:
            query = """DELETE FROM rms_%(rms_id_str)s_stat_factor_return
                    WHERE exp_dt = :date_arg""" % {'rms_id_str': rms_id_str}
            argDict = {'date_arg': rmi.date}
            self.dbCursor.execute(query, argDict)
            return True
        except:
            logging.warning('Table rms_%s_stat_factor_return does not exist', rms_id_str)
            return False
    
    def deleteRiskModelInstance(self, rmi, newExposureFormat):
        """Deletes all data associated with the given risk model instance
        and the instance entry itself.
        """
        # Asset universe data
        self.deleteRiskModelUniverse(rmi)
        self.deleteEstimationUniverse(rmi)
        # Exposure data
        if newExposureFormat:
            self.deleteRMIExposureMatrixNew(rmi)
        else:
            self.deleteRMIExposureMatrix(rmi)
        # Returns data
        self.deleteFactorReturns(rmi)
        self.deleteRMSFactorStatistics(rmi)
        self.deleteRMSStatistics(rmi)
        self.deleteFactorReturns(rmi, flag='internal')
        self.deleteRMSFactorStatistics(rmi, flag='internal')
        self.deleteRMSStatistics(rmi, flag='internal')
        try:
            self.deleteFactorReturns(rmi, flag='weekly')
            self.deleteRMSFactorStatistics(rmi, flag='weekly')
            self.deleteRMSStatistics(rmi, flag='weekly')
        except:
            logging.info('No weekly factor returns tables')
        self.deleteSpecificReturns(rmi)
        try:
            self.deleteSpecificReturns(rmi, internal=True)
        except:
            logging.info('No internal specific returns table')
        # Risk data
        self.deleteRMIFactorSpecificRisk(rmi)
        self.deleteRMIPredictedBeta(rmi)
        self.deleteRMIPredictedBeta(rmi, v3=True)
        self.deleteRMITotalRisk(rmi)
        self.dbCursor.execute("""DELETE FROM risk_model_instance
        WHERE rms_id = :rms_arg AND dt=:date_arg""", rms_arg=rmi.rms_id,
                              date_arg=rmi.date)
    
    def deleteRiskModelSerie(self, rms_id, newExposureFormat):
        """Deletes all data associated with the given risk model serie
        and the instance entry itself.
        """
        # Delete all associated instances
        self.dbCursor.execute("""DELETE FROM rmi_specific_risk
        WHERE rms_id = :rms_arg""", rms_arg=rms_id)
        self.dbCursor.execute("""DELETE FROM rmi_specific_covariance
        WHERE rms_id = :rms_arg""", rms_arg=rms_id)
        
        if newExposureFormat:
            self.dbCursor.execute("""TRUNCATE TABLE %(tables)"""
                                  % self.getWideExposureTableName(rms_id))
        else:
            self.dbCursor.execute("""DELETE FROM rmi_factor_exposure
        WHERE rms_id = :rms_arg""", rms_arg=rms_id)
        
        self.dbCursor.execute("""DELETE FROM rmi_covariance
        WHERE rms_id = :rms_arg""", rms_arg=rms_id)
        
        self.dbCursor.execute("""DELETE FROM rmi_estu
        WHERE rms_id = :rms_arg""", rms_arg=rms_id)

        self.dbCursor.execute("""DELETE FROM rmi_estu_v3
        WHERE rms_id = :rms_arg""", rms_arg=rms_id)
        
        self.dbCursor.execute("""DELETE FROM rmi_universe
        WHERE rms_id = :rms_arg""", rms_arg=rms_id)
        
        self.dbCursor.execute("""DELETE FROM risk_model_instance
        WHERE rms_id = :rms_arg""", rms_arg=rms_id)
        
        # Delete all serie data
        self.dbCursor.execute("""DELETE FROM rms_specific_return
        WHERE rms_id = :rms_arg""", rms_arg=rms_id)
        try:
            self.dbCursor.execute("""DELETE FROM rms_specific_return_internal
            WHERE rms_id = :rms_arg""", rms_arg=rms_id)
        except:
            logging.info('No internal specific return table')
        self.dbCursor.execute("""DELETE FROM rms_dva_statistics
        WHERE rms_id = :rms_arg""", rms_arg=rms_id)
        self.dbCursor.execute("""DELETE FROM rms_factor_return
        WHERE rms_id = :rms_arg""", rms_arg=rms_id)
        self.dbCursor.execute("""DELETE FROM rms_factor_return_internal
        WHERE rms_id = :rms_arg""", rms_arg=rms_id)
        self.dbCursor.execute("""DELETE FROM rms_factor_statistics
        WHERE rms_id = :rms_arg""", rms_arg=rms_id)
        self.dbCursor.execute("""DELETE FROM rms_factor_statistics_internal
        WHERE rms_id = :rms_arg""", rms_arg=rms_id)
        self.dbCursor.execute("""DELETE FROM rms_statistics
        WHERE rms_id = :rms_arg""", rms_arg=rms_id)
        self.dbCursor.execute("""DELETE FROM rms_statistics_internal
        WHERE rms_id = :rms_arg""", rms_arg=rms_id)
        self.dbCursor.execute("""DELETE FROM rms_factor
        WHERE rms_id = :rms_arg""", rms_arg=rms_id)
        self.dbCursor.execute("""DELETE FROM rms_robust_weight
        WHERE rms_id = :rms_arg""", rms_arg=rms_id)
        self.dbCursor.execute("""DELETE FROM rms_robust_weight_internal
        WHERE rms_id = :rms_arg""", rms_arg=rms_id)
        self.dbCursor.execute("""DELETE FROM rms_issue
        WHERE rms_id = :rms_arg""", rms_arg=rms_id)
        try:
            self.dbCursor.execute("""DELETE FROM rms_factor_return_weekly
            WHERE rms_id = :rms_arg""", rms_arg=rms_id)
            self.dbCursor.execute("""DELETE FROM rms_factor_statistics_weekly
            WHERE rms_id = :rms_arg""", rms_arg=rms_id)
            self.dbCursor.execute("""DELETE FROM rms_statistics_weekly
            WHERE rms_id = :rms_arg""", rms_arg=rms_id)
            self.dbCursor.execute("""DELETE FROM rms_robust_weight_weekly
            WHERE rms_id = :rms_arg""", rms_arg=rms_id)
        except:
            logging.info('No weekly regression tables')

        if rms_id < 0:
            table = 'rms_m%.2d_fmp' % abs(rms_id)
        else:
            table = 'rms_%s_fmp' % rms_id
        try:
            self.dbCursor.execute("""TRUNCATE TABLE %s""" % table)
        except:
            self.log.warning('Table %s does not exist', table)
        
        # Delete serie itself
        self.dbCursor.execute("""DELETE FROM risk_model_serie
        WHERE serial_id = :rms_arg""", rms_arg=rms_id)
    
    def deleteRMIPredictedBeta(self, rmi, v3=False):
        """Deletes predicted beta associated with the given
        risk model instance.
        Data is removed from rmi_predicted_beta.
        """
        if not v3:
            self.dbCursor.execute("""DELETE FROM rmi_predicted_beta
            WHERE rms_id = :rms_arg AND dt=:date_arg""", rms_arg=rmi.rms_id,
                                date_arg=rmi.date)
        else:
            try:
                self.dbCursor.execute("""DELETE FROM rmi_predicted_beta_v3
                        WHERE rms_id = :rms_arg AND dt=:date_arg""", rms_arg=rmi.rms_id,
                        date_arg=rmi.date)
            except:
                logging.info('No V3 beta table')

    def deleteRMITotalRisk(self, rmi):
        """Deletes total risk values associated with the given
        risk model instance.
        Data is removed from rmi_total_risk.
        """
        self.dbCursor.execute("""DELETE FROM rmi_total_risk
        WHERE rms_id = :rms_arg AND dt=:date_arg""", rms_arg=rmi.rms_id,
                              date_arg=rmi.date)
    
    def deleteRMSFactorStatistics(self, rmi, subFactors=None, flag=None):
        """Deletes the internal factor statistics of the given risk model instance
        from the database.
        """
        if flag is None:
            tableName = 'rms_factor_statistics'
        else:
            tableName = 'rms_factor_statistics_%s' % flag
        self.log.info('Deleting from %s for %s', tableName, rmi.date)
        if subFactors:
            valueDicts = [dict([('rms_arg', rmi.rms_id),
                                ('date_arg', rmi.date),
                                ('id_arg', f.subFactorID)])
                          for f in subFactors]
            self.dbCursor.executemany("""DELETE FROM %(table_arg)s
            WHERE rms_id = :rms_arg AND dt=:date_arg
            AND sub_factor_id = :id_arg""" % {'table_arg': tableName}, valueDicts)
        else:
            self.dbCursor.execute("""DELETE FROM %(table_arg)s
            WHERE rms_id = :rms_arg AND dt = :date_arg"""% {'table_arg': tableName},
            rms_arg=rmi.rms_id, date_arg=rmi.date)

    def deleteRMSStatistics(self, rmi, flag=None):
        """Deletes the summary internal regression statistics of the given
        risk model instance from the database.
        """
        if flag is None:
            tableName = 'rms_statistics'
        else:
            tableName = 'rms_statistics_%s' % flag
        self.log.info('Deleting from %s for %s', tableName, rmi.date)
        self.dbCursor.execute("""DELETE FROM %(table_arg)s
        WHERE rms_id = :rms_arg AND dt = :date_arg""" % {'table_arg': tableName},
        rms_arg=rmi.rms_id, date_arg=rmi.date)

    def deleteDVAStatistics(self, rmi):
        """Deletes the DVA scale statistics of the given
        risk model instance from the database.
        """
        tableName = 'rms_dva_statistics'
        self.log.info('Deleting from %s for %s', tableName, rmi.date)
        self.dbCursor.execute("""DELETE FROM %(table_arg)s
                WHERE rms_id = :rms_arg AND dt = :date_arg""" % {'table_arg': tableName},
                rms_arg=rmi.rms_id, date_arg=rmi.date)
    
    def deleteSpecificReturns(self, rmi, subIssues=[], internal=False):
        """Deletes the specific returns of the given risk model instance
        from the database.
        """
        if internal:
            tableName = 'rms_specific_return_internal'
        else:
            tableName = 'rms_specific_return'
        if len(subIssues) > 0:
            deleteDicts = [dict([('rms_arg', rmi.rms_id),
                                 ('date_arg', rmi.date),
                                 ('id_arg', sid.getSubIDString())])
                            for sid in subIssues]
            self.dbCursor.executemany("""DELETE FROM %(table_arg)s
            WHERE rms_id = :rms_arg AND dt = :date_arg
            AND sub_issue_id = :id_arg""" % {'table_arg': tableName}, deleteDicts)
            logging.info('Deleting %d specific returns from %s', len(deleteDicts), tableName)
        else:
            self.dbCursor.execute("""DELETE FROM %(table_arg)s
            WHERE rms_id = :rms_arg AND dt = :date_arg""" % {'table_arg': tableName},
            rms_arg=rmi.rms_id, date_arg=rmi.date)
            logging.info('Deleting specific returns from %s', tableName)

    def deleteRobustWeights(self, rmi, flag=None):
        """Deletes the robust weights of the given risk model instance
        from the database.
        """
        if flag is None:
            tableName = 'rms_robust_weight'
        else:
            tableName = 'rms_robust_weight_%s' % flag
        self.log.info('Deleting  robust weights from %s', tableName)
        self.dbCursor.execute("""DELETE FROM %(table_arg)s
        WHERE rms_id = :rms_arg AND dt = :date_arg""" % {'table_arg': tableName},
        rms_arg=rmi.rms_id, date_arg=rmi.date)

    def deleteFMPs(self, rmi):
        """Deletes the FMPs of the given risk model instance
        from the database.
        """
        if rmi.rms_id < 0:
            table = 'rms_m%.2d_fmp' % abs(rmi.rms_id)
        else:
            table = 'rms_%s_fmp' % rmi.rms_id
        try:
            self.dbCursor.execute("""DELETE FROM %s
            WHERE dt = :date_arg""" % table, date_arg=rmi.date)
        except:
            self.log.warning('Table %s does not exist: cannot delete', table)
            return False
        return True
    
    def getActiveIssues(self, currentDate):
        """Returns a list of all issues that are active on the given date.
        """
        self.dbCursor.execute("""SELECT issue_id FROM issue
        WHERE from_dt <= :date_arg AND thru_dt > :date_arg""",
                              date_arg=currentDate)
        return [ModelID.ModelID(string=i[0]) for i in self.dbCursor.fetchall()]
    
    def getRMSIDsByModelType(self, modelType='Fundamental'):
        """Returns a list of RMS IDs whose description matches
        a particular type
        """
        query = """SELECT rms_id FROM rms_id_description WHERE description LIKE :desc_arg"""
        self.dbCursor.execute(query, desc_arg="%%%s%%" % modelType)
        return [i[0] for i in self.dbCursor.fetchall()]

    def getActiveIssuesInModelList(self, currentDate, rmsList=[]):
        """Returns a list of all sub-issues in the given list of
        risk models
        """
        query = """SELECT DISTINCT issue_id FROM rms_issue
                   WHERE from_dt <= :date_arg AND thru_dt > :date_arg
                   AND rms_id IN (%(ids)s)""" % \
                           {'ids': ','.join([('%s' % i) for i in rmsList])}
        self.dbCursor.execute(query, date_arg=currentDate)
        return [ModelID.ModelID(string=i[0]) for i in self.dbCursor.fetchall()]

    def getActiveSubIssues(self, rmg, currentDate, inModels=False):
        """Returns a list of all sub-issues in the given risk model group
        that are active on the given date.
        If inModels=True, returns only sub-issues alive in one or
        more production model series.
        """
        query = """SELECT s.sub_id FROM sub_issue s
        WHERE s.rmg_id = :rmg_arg AND s.from_dt <= :date_arg
        AND s.thru_dt > :date_arg """
        if inModels is True:
            query += """AND EXISTS (SELECT * FROM rms_issue r
                        WHERE r.issue_id = s.issue_id
                        AND r.from_dt <= :date_arg AND r.thru_dt > :date_arg
                        AND r.rms_id  > 0)"""
        self.dbCursor.execute(query, rmg_arg=rmg.rmg_id, date_arg=currentDate)
        return [SubIssue(i[0]) for i in self.dbCursor.fetchall()]
    
    def getAllActiveSubIssuesInRange(self, d0, d1):
        """Returns a list of all sub-issues that are active in the given date range.
        """
        query="""SELECT sub_id FROM sub_issue s
        WHERE s.from_dt <= :d1 AND s.thru_dt > :d0 """
        self.dbCursor.execute(query, d0 = d0, d1 = d1)
        return [SubIssue(string=i[0]) for i in self.dbCursor.fetchall()]
    
    def getAllActiveSubIssues(self, currentDate, inModels=False):
        """Returns a list of all sub-issues that are active on the given date.
        """
        query="""SELECT sub_id FROM sub_issue s
        WHERE s.from_dt <= :date_arg AND s.thru_dt > :date_arg """
        if inModels is True:
            query += """AND EXISTS (SELECT * FROM rms_issue r
                        WHERE r.issue_id = s.issue_id
                        AND r.from_dt <= :date_arg AND r.thru_dt > :date_arg
                        AND r.rms_id  > 0)"""
        self.dbCursor.execute(query, date_arg = currentDate)
        return [SubIssue(string=i[0]) for i in self.dbCursor.fetchall()]
    
    def getAllIssues(self):
        """Returns a list of all ModelIDs with their from- and thru-date.
        The return value is a list of (issue-id, from_dt, thru_dt) tuples.
        """
        self.dbCursor.execute("SELECT issue_id, from_dt, thru_dt FROM issue")
        return [(ModelID.ModelID(string=i[0]), i[1].date(), i[2].date())
                for i in self.dbCursor.fetchall()]
    
    def getIssuesByMarketDBType(self, mktDBType):
        """Returns a list of sub-issues adhering to a particular marketDB ID pattern
        For instance open-ended funds ('O%'), Cash ('CSH%')
        """
        query="""SELECT sub_id FROM sub_issue si JOIN
        issue_map im ON im.modeldb_id=si.issue_id and im.marketdb_id like :type_arg"""
        self.dbCursor.execute(query, type_arg=mktDBType)
        return [SubIssue(string=i[0]) for i in self.dbCursor.fetchall()]

    def getRiskFreeRateHistoryInternal(self, dateList, ids, marketDB):
        """Helper function for getRiskFreeRateHistory()
        """
        dateLen = len(dateList)
        idLen = len(ids)
        self.log.debug('loading risk-free rates for %d days and %d currencies',
                       dateLen, idLen)
        results = Matrices.TimeSeriesMatrix(ids, dateList)
        results.data = Matrices.allMasked(results.data.shape)
        if dateLen == 0 or idLen == 0:
            return results
        dateIdxMap = dict([(j,i) for (i,j) in enumerate(dateList)])
        currencyIdxMap = dict([(j,i) for (i,j) in enumerate(ids)])
        all_ids = list(ids)
        cidArgList = [('cid%d' % i) for i in range(len(all_ids))]
        query = """SELECT r.currency_code, r.dt, POWER(1.0 + r.value, 1.0 / 252.0) - 1.0
                   FROM currency_risk_free_rate r
                   WHERE r.currency_code IN (%(ids)s) 
                   AND r.dt = :dt""" % {
            'ids': ','.join([(':%s' % i) for i in cidArgList])}
        valueDict = dict(zip(cidArgList, all_ids))
        for (dIdx, date) in enumerate(dateList):
            valueDict['dt'] = date
            self.dbCursor.execute(query, valueDict)
            r = self.dbCursor.fetchmany()
            while len(r) > 0:
                for row in r:
                    if row[2] is None:
                        continue
                    value = float(row[2])
                    cIdx = currencyIdxMap[row[0]]
                    dIdx = dateIdxMap[row[1].date()]
                    results.data[cIdx, dIdx] = value
                r = self.dbCursor.fetchmany()
        return results
    
    def getRiskFreeRateHistory(self, isoCodeList, dateList, marketDB, annualize=False, returnDF=False):
        """Returns a TimeSeriesMatrix the risk-free rate associated 
        with the specified currencies and dates.  Missing records 
        are masked.  Values are stored as daily rates in the cache,
        but can be converted to annualized figures if required.
        """
        cache = self.riskFreeRateCache
        if cache.maxDates < len(dateList):
            self.log.warning('cache uses %d days, requesting %d', 
                            cache.maxDates, len(dateList))
        missingDates = cache.findMissingDates(dateList)
        missingIDs = cache.findMissingIDs(isoCodeList)
        
        if len(missingIDs) > 0:
            # Get data for missing currencies on existing dates
            missingIDData = self.getRiskFreeRateHistoryInternal(
                                cache.getDateList(), missingIDs, marketDB)
            cache.addMissingIDs(missingIDData)
        if len(missingDates) > 0:
            # Get data for all currencies on missing dates
            missingDateData = self.getRiskFreeRateHistoryInternal(
                                missingDates, cache.getIDList(), marketDB)
            cache.addMissingDates(missingDateData, None)
        # extract subset
        result = cache.getSubMatrix(dateList, isoCodeList)
        self.log.debug('received data')
        
        # Check for missing values
        for i in range(len(isoCodeList)):
            missing = numpy.sum(ma.getmaskarray(result.data[i,:]))
            missing = numpy.flatnonzero(ma.getmaskarray(result.data[i,:]))
            if len(missing) > 0:
                self.log.warning('Missing risk-free rate for %s on %d dates',
                        isoCodeList[i], len(missing))
                self.log.debug('Missing dates: %s', 
                        ','.join([str(dateList[j]) for j in missing]))
        
        # Annualize rates if required
        if annualize:
            result.data = ma.power(result.data + 1.0, 252.0) - 1.0
            self.log.debug('annualized data')
        if returnDF:
            result = result.toDataFrame()
        return result
    
    def getCAMergerSurvivors(self, date, rmg=None):
        """Return the list of merger survivors according to the
        corporate action table.
        The return value is a list of MergerSurvivor objects.
        """
        if rmg is not None:
            self.dbCursor.execute("""SELECT ca_sequence, modeldb_id,
            new_marketdb_id, old_marketdb_id, share_ratio, cash_payment,
            currency_id, ref
            FROM ca_merger_survivor
            WHERE dt = :date_arg and rmg_id=:rmg_arg""",
                                  date_arg=date, rmg_arg=rmg.rmg_id)
        else:
            self.dbCursor.execute("""SELECT ca_sequence, modeldb_id,
            new_marketdb_id, old_marketdb_id, share_ratio, cash_payment,
            currency_id, ref
            FROM ca_merger_survivor
            WHERE dt = :date_arg""", date_arg=date)
        return [MergerSurvivor(i[0], ModelID.ModelID(string=i[1]), i[2],
                               i[3], i[4], i[5], i[6], i[7])
                for i in self.dbCursor.fetchall()]
    
    def getCASpinOffs(self, date, rmg=None):
        """Return the list of spin-offs according to the
        corporate action table.
        The return value is a list of SpinOff objects.
        """
        if rmg is not None:
            self.dbCursor.execute("""SELECT ca_sequence, parent_id,
            child_id, share_ratio, implied_div, currency_id, ref
            FROM ca_spin_off
            WHERE dt = :date_arg and rmg_id=:rmg_arg""",
                                  date_arg=date, rmg_arg=rmg.rmg_id)
        else:
            self.dbCursor.execute("""SELECT ca_sequence, parent_id,
            child_id, share_ratio, implied_div, currency_id, ref
            FROM ca_spin_off
            WHERE dt = :date_arg""", date_arg=date)
        return [SpinOff(i[0], ModelID.ModelID(string=i[1]),
                        ModelID.ModelID(string=i[2]), i[3], i[4],
                        i[5], i[6])
                for i in self.dbCursor.fetchall()]
    
    def getCompositeConstituents(self, compositeName, date, marketDB, 
                                 rollBack=0, marketMap=None, compositeAxiomaID=None):
        """Returns the assets and their weight in the composite for the
        given date. If there is no data for the specified date, the most
        recent assets up to rollBack days ago will be returned.
        The return value is a pair of date and list of (ModelID, weight)
        pairs.
        The returned date is the date from which the constituents were
        taken which will be different from the requested date if
        rollBack > 0 and there is no active revision for the requested date.
        marketMap: an optional dictionary mapping Axioma ID strings (marketdb)
        to their corresponding ModelID object.
        """
        self.log.debug('getCompositeConstituents: begin')
        # if axioma_id is supplied look it up by that
        if compositeAxiomaID:
            composite = marketDB.getETFbyAxiomaID(compositeAxiomaID.getIDString(), date)
        else:
            composite = marketDB.getETFByName(compositeName, date)
        if composite is None:
            return (None, list())
        compositeRevInfo = marketDB.getETFRevision(composite, date, rollBack)
        if compositeRevInfo is None:
            return (None, list())
        assetWeightsMap = marketDB.getETFConstituents(compositeRevInfo[0])
        if marketMap is None:
            issueMapPairs = self.getIssueMapPairs(date)
            marketMap = dict([(i[1].getIDString(), i[0]) for i in issueMapPairs])
        constituents = [(marketMap[i], assetWeightsMap[i]) for i
                        in list(assetWeightsMap.keys()) if i in marketMap]
        notMatched = [i for i in assetWeightsMap.keys()
                      if i not in marketMap]
        if len(notMatched) > 0:
            self.log.info("%d unmapped assets in composite", len(notMatched))
            self.log.debug("Can't match assets in %s to ModelDB on %s: %s",
                           composite.name, date, ','.join(notMatched))
        if len(constituents) > 0:
            self.log.debug('%d assets in %s composite', len(constituents),
                          composite.name)
        else:
            self.log.warning('no assets in %s composite', composite.name)
        self.log.debug('getCompositeConstituents: end')
        return (compositeRevInfo[1], constituents)
        
    def getFutureFamilyModelMap(self):
        """Returns the model_id/future_family_id pairs
        from future_family_model_map.
        """
        self.dbCursor.execute("""SELECT model_id, future_family_id
           FROM future_family_model_map""")
        return list(self.dbCursor.fetchall())

    def getCompositeFamilyModelMap(self):
        """Returns the model_id/composite_family_id pairs
        from composite_family_model_map.
        """
        self.dbCursor.execute("""SELECT model_id, composite_family_id
           FROM composite_family_model_map""")
        return list(self.dbCursor.fetchall())
    
    def getDividendYield(self, date, subids):
        """Returns a dictionary mapping SubIssues to their dividend yields 
        for the given day.
        """
        self.log.debug('getDividendYield: begin')
        divyield = self.loadSubIssueData(
            [date], subids, 'sub_issue_divyield', 'data.value',
             cache=None, convertTo=None, withCurrency=False,
            withRevDate=False)
        self.log.debug('getDividendYield: end')
        return dict([(sid, val) for (sid, val) in zip(subids, divyield.data[:,0])
                     if val is not ma.masked])
    
    def getHistoricBetaOld(self, date, subids):
        """Returns a dictionary mapping SubIssues to their old (deprecated) historic beta
        for the given day.
        """
        self.log.debug('getHistoricBetaOld: begin')
        self.log.debug('getHistoricBetaOld: end')
        return self.getHistoricBetaFixed(date, subids)
    
    def getHistoricBeta(self, date, subids):
        """Returns a dictionary mapping SubIssues to their historic beta
        for the given day.
        """
        self.log.debug('getHistoricBeta: begin')
        self.log.debug('getHistoricBeta: end')
        return self.getHistoricBetaFixed(date, subids)
    
    def getHistoricBetaFixed(self, date, subids, legacy=False):
        """Returns a dictionary mapping SubIssues to their historic beta
        for the given day.
        """
        self.log.debug('getHistoricBetaFixed: begin')
        if legacy:
            hbeta = self.loadSubIssueData(
                    [date], subids, 'rmg_historic_beta', 'data.fixed_value',
                    cache=self.histBetaCacheFixed, convertTo=None, withCurrency=False,
                    withRevDate=False)
            self.log.debug('getHistoricBetaFixed: end')
            return dict([(sid, val) for (sid, val) in zip(subids, hbeta.data[:,0])
                if val is not ma.masked])
        hbeta = self.getHistoricBetaDataV3(date, subids, home=1)
        hbeta.update(self.getHistoricBetaDataV3(date, subids, home=0))
        self.log.debug('getHistoricBetaFixed: end')
        return hbeta

    def getHistoricBetaDataV3(self, dateList, subids, field='value', home=1,
                                rollOverData=False, average=False, returnDF=False):
        """Returns a dictionary mapping SubIssues to their historic beta fields
        """
        cacheDict = {'value': self.histBetaVersn3Cache,
                     'robust_value': self.histRobustBetaCache,
                     'p_value': self.histBetaPValueCache,
                     'nobs': self.histBetaNObsCache}

        self.log.debug('getHistoricBetaDataV3: begin')
        if not isinstance(dateList, list):
            if rollOverData:
                dateList = [dateList - datetime.timedelta(i) for i in range(31)]
                dateList.reverse()
            else:
                dateList = [dateList]

        hbeta = self.loadSubIssueData(
                dateList, subids, 'rmg_historic_beta_v3', 'data.%s' % field,
                cache=cacheDict[field], convertTo=None, withCurrency=False,
                withRevDate=False, condition='HOME=%f' % home)

        self.log.debug('getHistoricBetaDataV3: end')

        if rollOverData:
            hbetaDict = dict()
            for (sidIdx, sid) in enumerate(subids):
                for dIdx in range(hbeta.data.shape[1]):
                    if hbeta.data[sidIdx, dIdx] is not ma.masked:
                        hbetaDict[sid] = hbeta.data[sidIdx, dIdx]
            if returnDF:
                return pandas.Series(hbetaDict)
            return hbetaDict

        if average:
            hbeta.data = ma.median(hbeta.data, axis=1)
        else:
            hbeta.data = hbeta.data[:,-1]

        hbetaDict = dict([(sid, val) for (sid, val) in zip(subids, hbeta.data) if val is not ma.masked])
        if returnDF:
            return pandas.Series(hbetaDict)
        return hbetaDict

    def loadDescriptorData(self, dateList, subids, tableNameCurr, desc_id,
            tableName='descriptor_exposure', rollOverData=0, curr_field=None, returnDF=False):
        """Returns a TimeSeriesMatrix to their descriptor
        data.
        """
        self.log.debug('loadDescriptorData: begin')
        tableName = '%s_%s' % (tableName, tableNameCurr)
        self.log.debug('Loading descriptor data from %s', tableName)

        # If we're loading in a history, don't rollover data
        if not isinstance(dateList, list):
            dateList = [dateList]
        if len(dateList) > 1:
            rollOverData = 0

        # Get dates for which to roll-over data if required
        if rollOverData > 1:
            rollOverDateList = [dateList[0] - datetime.timedelta(i) for i in range(rollOverData)]
        elif rollOverData == 1:
            rollOverDateList = self.getDateRange(self.getAllRiskModelGroups(),
                            None, dateList[0], tradingDaysBack=2, excludeWeekend=True)
            rollOverDateList.reverse()

        # Initialise
        data = Matrices.TimeSeriesMatrix(subids, dateList)
        data.data = Matrices.allMasked(data.data.shape)
        if curr_field is None:
            curr_cond = None
        else:
            curr_cond = """curr_field='%s'""" % curr_field

        # Get cache and create it if it's missing or doesn't cover the right date range
        cache_key = (tableName, desc_id)
        cache = self.descriptorCache.get(cache_key, None)
        if (cache is None) or (cache.maxDates < len(dateList)):
            self.descriptorCache[cache_key] = TimeSeriesCache(len(dateList))

        # If we're not rolling data over, merely load in the data for the required date(s)
        if rollOverData == 0:
            data = self.loadSubIssueData(dateList, subids, tableName, 'ds_%d' % desc_id,
                    cache=self.descriptorCache[cache_key], convertTo=None, withCurrency=False,
                    withRevDate=False, condition=curr_cond)
            self.log.debug('loadDescriptorData: end')
            data.data = Utilities.screen_data(data.data)
            if returnDF:
                return data.toDataFrame()
            return data

        # Otherwise, loop round dates loading data until either everything is populated,
        # or we've reached the end of the road
        runningSidList = list(subids)
        subIdxMap = dict(zip(subids, list(range(len(subids)))))

        for dt in rollOverDateList:

            # Load data for one day and current set of assets
            subData = Matrices.TimeSeriesMatrix(runningSidList, [dt])
            subData.data = Matrices.allMasked(subData.data.shape)
            subData = self.loadSubIssueData([dt], runningSidList, tableName, 'ds_%d' % desc_id,
                    cache=self.descriptorCache[cache_key], convertTo=None, withCurrency=False,
                    withRevDate=False, condition=curr_cond)
            subData.data = Utilities.screen_data(subData.data)

            # Write non-missing data to main array
            nonMissingIdx = numpy.flatnonzero(ma.getmaskarray(subData.data[:,0])==0)
            nonMissingSids = numpy.take(runningSidList, nonMissingIdx, axis=0)
            nonMissingData = ma.take(subData.data, nonMissingIdx, axis=0)
            for (idx, sid) in enumerate(nonMissingSids):
                data.data[subIdxMap[sid],0] = nonMissingData[idx,0]

            # Get new list of assets still missing data
            missingIdx = numpy.flatnonzero(ma.getmaskarray(subData.data))
            if len(missingIdx) < 1:
                break
            runningSidList = numpy.take(runningSidList, missingIdx, axis=0)

        data.data = Utilities.screen_data(data.data)
        self.log.debug('loadDescriptorData: end')
        if returnDF:
            return data.toDataFrame()
        return data

    def loadLocalDescriptorData(self, date, subids, currencyMap, desc_id_list, rollOverData=0, returnDF=False):
        """Returns a matrix of subissues to descriptor items
        """
        self.log.debug('loadLocalDescriptorData: begin')
        tableName = 'descriptor_local_currency'
        self.log.debug('Loading descriptor data from %s', tableName)

       # Get dates for which to roll-over data if required
        if rollOverData > 1:
            rollOverDateList = [date - datetime.timedelta(i) for i in range(rollOverData)]
        elif rollOverData == 1:
            rollOverDateList = self.getDateRange(self.getAllRiskModelGroups(),
                            None, date, tradingDaysBack=2, excludeWeekend=True)
            rollOverDateList.reverse()

        # If we're not rolling data over, merely load in the data for the required date(s)
        data = self.loadLocalDescriptorArray(date, subids, currencyMap, desc_id_list, tableName)
        missingData = ma.getmaskarray(data)
        missingIdx = numpy.flatnonzero(ma.sum(missingData, axis=1))
        if (rollOverData == 0) or (len(missingIdx)<1):
            if returnDF:
                return pandas.DataFrame(data, index=subids, columns=desc_id_list)
            return data

        # Otherwise, for each descriptor,  loop round dates loading data until everything that can be is populated
        for (dIdx, desc_id) in enumerate(desc_id_list):
            for dt in rollOverDateList[1:]:
                missingIdx = numpy.flatnonzero(ma.getmaskarray(data[:, dIdx]))
                if len(missingIdx) > 0:
                    logging.info('Loading descriptor ID %s data on %s for %d missing values',
                            desc_id, dt, len(missingIdx))
                    # Load data for one day and set of assets missing data
                    subSidList = numpy.take(subids, missingIdx, axis=0)
                    subData = Matrices.allMasked((len(subSidList), 1))
                    subData = self.loadLocalDescriptorArray(dt, subSidList, currencyMap, [desc_id], tableName)

                    # Write non-missing data to main array
                    for (ii, idx) in enumerate(missingIdx):
                        data[idx, dIdx] = subData[ii, 0]
                else:
                    break

        self.log.debug('loadLocalDescriptorData: end')
        if returnDF:
            return pandas.DataFrame(data, index=subids, columns=desc_id_list)
        return data

    def loadLocalDescriptorArray(self, date, subids, currencyMap, desc_id_list, tableName):

        # Set up DB query
        desc_ids = ['ds_%d' % ds for ds in desc_id_list]
        query = """SELECT sub_issue_id, %(dsColumns)s FROM %(table)s
                   WHERE dt=:date_arg AND curr_field=:curr_arg""" % \
                   {'table': tableName, 'dsColumns': ','.join(desc_ids)}

        # Loop round set of currencies required
        data = Matrices.allMasked((len(subids), len(desc_id_list)))
        currencyList = set(currencyMap.keys())
        subIDStrings = [s.getSubIDString() for s in subids]
        assetIdxMap = dict(zip(subIDStrings, range(len(subids))))
        for curISO in currencyList:
            sidSubSet = currencyMap[curISO]
            tmpDict=dict( [(r.getSubIDString(),1) for r in sidSubSet])
            # Pull out descriptors for subset of subissues
            #logging.info('CUR: %s', curISO)
            self.dbCursor.execute(query, date_arg=date, curr_arg=curISO)
            r = self.dbCursor.fetchmany()
            while len(r) > 0:
                #logging.info('SUBS: %s', len(r))
                for sidData in r:
                    sidIdx = assetIdxMap.get(sidData[0], None)
                    if (sidIdx is not None) and (sidData[0] in tmpDict):
                        data[sidIdx, :] = ma.array(sidData[1:], float)
                r = self.dbCursor.fetchmany()
        data = Utilities.screen_data(data)
        return data

    def loadDescriptorDataHistory(self, desc_id, dateList, subids, table, currencyMap=None, curr_field=None):
        """Returns TimeSeriesMatrix (subids x dateList) of desc_id values
        
        Parameters
        ----------
        desc_id: int 
            Descriptor ID
        dateList: list of datetime.date objects
        subids: list of SubIssue objects
        table: string
            Name of descriptor table exluding ccy code: 
            descriptor_exposure, descriptor_local_currency, or
            descriptor_numeraire
        currencyMap: dict <ccy: list of SubIssue objects>
            Required for table descriptor_local_currency
        curr_field: string
            Currency code; required for descriptor_numeraire or 
            descriptor_exposure tables

        Returns
        -------
        data: TimeSeriesMatrix

        Examples
        --------
        # Momentum_250x20D
        res = modelDB.loadDescriptorDataHistory(184, dateList, sidList, 
            table='descriptor_local_currency', 
            currencyMap=modelData.currencyAssetMap) 

        # Book_to_Price_Annual
        res = modelDB.loadDescriptorDataHistory(217, dateList, sidList, 
            table='descriptor_numeraire', curr_field='USD')

        # Book_to_Price_Quarterly
        res = modelDB.loadDescriptorDataHistory(113, dateList, sidList, 
            table='descriptor_exposure', curr_field='USD')
        """
        self.log.debug('loadDescriptorDataHistory: begin')
        self.log.debug('Loading descriptor data history from %s', table)

        validTableNames = ('descriptor_local_currency', 'descriptor_numeraire',
            'descriptor_exposure')
        assert(table in validTableNames)
        desc_id_col = 'ds_%d' % desc_id

        if table == 'descriptor_local_currency':
            assert(currencyMap is not None)
            results = pandas.DataFrame()
            for curISO in currencyMap.keys():
                sidSubset = list(set(currencyMap[curISO]).intersection(set(subids)))
                res = self.loadSubIssueData(dateList, sidSubset, table, 
                    desc_id_col, withRevDate=False, withCurrency=False, 
                    condition=" curr_field='USD' ")
                results =  pandas.concat([results, res.toDataFrame()])
            data = Matrices.TimeSeriesMatrix.fromDataFrame(results)
        elif table == 'descriptor_numeraire':
            assert(curr_field is not None)
            table = table + '_' + curr_field
            res = self.loadSubIssueData(dateList, subids, table, 
                desc_id_col, withRevDate=False, withCurrency=False, 
                condition=" curr_field='%s' " % curr_field)
            data = res
        elif table == 'descriptor_exposure':
            assert(curr_field is not None)
            table = table + '_' + curr_field
            res = self.loadSubIssueData(dateList, subids, table, 
                desc_id_col, withRevDate=False, withCurrency=False)
            data = res

        self.log.debug('loadDescriptorDataHistory: end')
        return data

    def getPreviousHistoricBetaOld(self, date, subids):
        """Returns a dictionary mapping SubIssues to their most recent
        historic beta prior to the given day (looking back at most 30 days).
        This uses the old, deprecated beta definition.
        """
        self.log.debug('getPreviousHistoricBetaOld: begin')
        return self.getPreviousHistoricBetaFixed(date, subids)
    
    def getPreviousHistoricBeta(self, date, subids):
        """Returns a dictionary mapping SubIssues to their most recent
        historic beta prior to the given day (looking back at most 30 days).
        """
        self.log.debug('getPreviousHistoricBeta: begin')
        return self.getPreviousHistoricBetaFixed(date, subids)
    
    def getPreviousHistoricBetaFixed(self, date, subids, legacy=False):
        """Returns a dictionary mapping SubIssues to their most recent
        historic beta prior to the given day (looking back at most 30 days).
        """
        self.log.debug('getPreviousHistoricBetaFixed: begin')
        if legacy:
            dateList = [date - datetime.timedelta(i + 1) for i in range(30)]
            dateList.reverse()
            hbetaHist = self.loadSubIssueData(
                    dateList, subids, 'rmg_historic_beta', 'data.fixed_value',
                    cache=self.histBetaCacheFixed, convertTo=None, withCurrency=False,
                    withRevDate=False)
            hbeta = dict()
            for (sidIdx, sid) in enumerate(subids):
                for dIdx in range(hbetaHist.data.shape[1]):
                    if hbetaHist.data[sidIdx, dIdx] is not ma.masked:
                        hbeta[sid]  = hbetaHist.data[sidIdx, dIdx]
            return hbeta

        hbeta = self.getHistoricBetaDataV3(date, subids, home=1, rollOverData=True)
        hbeta.update(self.getHistoricBetaDataV3(date, subids, home=0, rollOverData=True))
        return hbeta
     
    def getMdlClassificationByName(self, name):
        """Returns a Struct with id, name, isRoot,
        and isLeaf entries for the classification with the given name,
        or None if it doesn't exist.
        """
        self.dbCursor.execute("""SELECT id
        FROM classification_ref WHERE name = :name_arg""", name_arg=name)
        r = self.dbCursor.fetchall()
        if len(r) == 0:
            return None
        elif len(r) > 1:
            self.log.error('More than one classification with name %s',
                           name)
            return None
        cls_id = r[0][0]
        return self.getMdlClassificationByID(cls_id)
    
    def getMdlClassificationByID(self, cls_id):
        """Returns a Struct with id, name, description isRoot,
        and isLeaf entries for the classification with the given ID,
        or None if it doesn't exist.
        """
        classification = Utilities.Struct()
        classification.id = cls_id
        self.dbCursor.execute("""SELECT is_root, is_leaf, name, description,
        revision_id
        FROM classification_ref WHERE id = :id_arg""", id_arg=cls_id)
        r = self.dbCursor.fetchall()
        if len(r) == 0:
            return None
        assert(len(r) == 1)
        classification.isRoot = (r[0][0] == 'Y')
        classification.isLeaf = (r[0][1] == 'Y')
        classification.name = r[0][2]
        classification.description = r[0][3]
        classification.revision_id = r[0][4]
        self.dbCursor.execute("""SELECT market_ref_id
        FROM classification_market_map WHERE model_ref_id=:mdl_ref""",
                              mdl_ref=cls_id)
        r = self.dbCursor.fetchall()
        classification.market_cls_ids = [i[0] for i in r]
        return classification
    
    def getMdlClassificationChildren(self, parentClass):
        """Returns a list of the children of the classification given
        by parentID in the classification hierarchy.
        Each entry in the list is a Struct of the same form as
        returned by getClassification with an additional weight entry.
        """
        self.dbCursor.execute("""SELECT child_classification_id, weight
        FROM classification_dim_hier
        WHERE parent_classification_id = :id_arg""", id_arg=parentClass.id)
        ret = self.dbCursor.fetchall()
        children = [self.getMdlClassificationByID(r[0]) for r in ret]
        for (c, r) in zip(children, ret):
            c.weight = r[1]
        return children
    
    def getMdlClassificationFamily(self, name):
        """Returns a struct with id, name, description entries for the
        given classification family.
        If the classification family doesn't exist, the return value is None.
        """
        self.dbCursor.execute("""SELECT id, description
        FROM classification_family WHERE name = :name_arg""", name_arg=name)
        r = self.dbCursor.fetchall()
        if len(r) == 0:
            return None
        elif len(r) > 1:
            self.log.error('Classification family name %s not unique' % name)
            return None
        
        family = Utilities.Struct()
        family.name = name
        family.id = r[0][0]
        family.description = r[0][1]
        return family
    
    def getMdlClassificationFamilyMembers(self, family):
        """Returns the classification members in the given family.
        Return value is a list of structs with id, name, description,
        family_id entries.
        If the classification family doesn't exist, the return value is None.
        """
        if family == None:
            return None
        self.dbCursor.execute("""SELECT id, name, description
        FROM classification_member WHERE family_id = :family_arg""",
                              family_arg=family.id)
        r = self.dbCursor.fetchall()
        members = []
        for i in r:
            member = Utilities.Struct()
            member.id = i[0]
            member.name = i[1]
            member.description = i[2]
            member.family_id = family.id
            members.append(member)
        return members
    
    def getMdlClassificationMemberLeaves(self, member, date):
        """Returns the leaf classifications for the given classification
        member on the given day.
        If no revision is present on that day, None is returned.
        """
        self.dbCursor.execute("""SELECT c.id
        FROM classification_ref c, classification_revision r
        WHERE c.is_leaf='Y' AND c.revision_id = r.id
        AND r.member_id=:member_arg
        AND r.from_dt <= :date_arg AND r.thru_dt > :date_arg""",
                              member_arg=member.id,
                              date_arg=date)
        r = self.dbCursor.fetchall()
        if len(r) == 0:
            return None
        return [self.getMdlClassificationByID(i[0]) for i in r]
            
    def getMdlClassificationMemberRevision(self, member, date):
        """Returns the active revision for the given classification
        member on the given day.
        If no revision is present on that day, None is returned.
        Otherwise a struct with id, member_id, from_dt, and thru_dt
        is returned.
        """
        self.dbCursor.execute("""SELECT id, from_dt, thru_dt
        FROM classification_revision
        WHERE member_id=:member_arg
        AND from_dt <= :date_arg AND thru_dt > :date_arg""",
                              member_arg=member.id,
                              date_arg=date)
        r = self.dbCursor.fetchall()
        if len(r) > 1:
            raise KeyError('More than one active revision')
        elif len(r) == 0:
            return None
        rev = Utilities.Struct()
        rev.id = r[0][0]
        rev.from_dt = self.oracleToDate(r[0][1])
        rev.thru_dt = self.oracleToDate(r[0][2])
        rev.member_id = member.id
        return rev
            
    def getMdlClassificationMemberRoots(self, member, date):
        """Returns the root classifications for the given classification
        member on the given day.
        If no revision is present on that day, None is returned.
        """
        self.dbCursor.execute("""SELECT c.id
        FROM classification_ref c, classification_revision r
        WHERE c.is_root='Y' AND c.revision_id = r.id
        AND r.member_id=:member_arg
        AND r.from_dt <= :date_arg AND r.thru_dt > :date_arg""",
                              member_arg=member.id,
                              date_arg=date)
        r = self.dbCursor.fetchall()
        if len(r) == 0:
            return None
        return [self.getMdlClassificationByID(i[0]) for i in r]
            
    def getMdlClassificationMembers(self, member, date):
        """Returns all non-root classifications for the given classification
        member on the given day.
        If no revision is present on that day, None is returned.
        """
        self.dbCursor.execute("""SELECT c.id
        FROM classification_ref c, classification_revision r
        WHERE c.is_root='N' AND c.revision_id = r.id
        AND r.member_id=:member_arg
        AND r.from_dt <= :date_arg AND r.thru_dt > :date_arg""",
                              member_arg=member.id,
                              date_arg=date)
        r = self.dbCursor.fetchall()
        if len(r) == 0:
            return None
        return [self.getMdlClassificationByID(i[0]) for i in r]
            
    def getMdlClassificationParent(self, childClass):
        """Returns the parent classification (if any) of the
        specified child classification according to the hierarchy.
        The return value is a Struct of the same form as 
        returned by getMdlClassificationByID().
        """
        self.dbCursor.execute("""SELECT cd.parent_classification_id
        FROM classification_dim_hier cd, classification_ref cr
        WHERE cd.child_classification_id = :child_id
        AND cd.parent_classification_id = cr.id
        AND cr.is_root = 'N'""",
                                child_id=childClass.id)
        r = self.dbCursor.fetchall()
        if len(r) == 0:
            return None
        return self.getMdlClassificationByID(r[0][0])
    
    def getMdlClassificationRevisionRoot(self, revision):
        """Returns the main root classification for the given classification
        revision which is the root classification with the lowest ID number.
        """
        self.dbCursor.execute("""SELECT MIN(c.id)
        FROM classification_ref c
        WHERE c.is_root='Y' AND c.revision_id = :rev_arg""",
                              rev_arg=revision.id)
        r = self.dbCursor.fetchall()
        if len(r) == 0:
            return None
        return self.getMdlClassificationByID(r[0][0])
            
    def getMdlClassificationAllParents(self, childClass):
        """Returns all parent classifications (if any) of the
        specified child classification according to the hierarchy.
        The return value is a list of Structs of the same form as 
        returned by getMdlClassificationByID().
        """
        self.dbCursor.execute("""SELECT cd.parent_classification_id
        FROM classification_dim_hier cd
        WHERE cd.child_classification_id = :child_id""",
                                child_id=childClass.id)
        r = self.dbCursor.fetchall()
        if len(r) == 0:
            return None
        return [self.getMdlClassificationByID(p[0]) for p in r]
    
    def getMdlAssetClassifications(self, revision, assetList, date,
                                   level=None):
        """Returns the classification of all the passed issues or sub-issues
        within a particular ModelDB classification revision
        on the supplied date.
        The return value is a dict mapping assets to structs with
        the information from the classification_constituent table:
        classification_id, weight, src, ref, and
        classification (as returned by getClassificationByID).
        """
        self.log.debug('loading Model Classification %d/%s for %d assets',
                       revision.member_id, revision.from_dt, len(assetList))
        if revision.id not in self.modelClassificationCaches:
            cache = ClassificationCache(self, revision, True)
            self.modelClassificationCaches[revision.id] = cache
        else:
            cache = self.modelClassificationCaches[revision.id]
        
        INCR=200
        keyList = ['key%d' % i for i in range(INCR)]
        defaultDict = dict([(a, None) for a in keyList])
        
        issueList = list()
        for a in assetList:
            if isinstance(a, SubIssue):
                issueList.append(a.getModelID())
            else:
                issueList.append(a)
        missingIds = cache.getMissingKeys(issueList)
        self.log.debug('adding %d assets to cache', len(missingIds))
        issueStr = [i.getIDString() for i in missingIds]
        # Get classification history for issue IDs
        issueClsDict = dict()
        query = """SELECT ca.classification_id, ca.weight, ca.change_dt,
          ca.change_del_flag, ca.src_id, ca.ref, ca.issue_id
        FROM classification_const_active ca
        WHERE ca.issue_id IN (%(keys)s) AND ca.revision_id=:revision_id
        ORDER BY change_dt ASC
        """ % dict(keys=','.join([':%s' % i for i in keyList]))
        for idChunk in listChunkIterator(issueStr, INCR):
            valueDict = dict(defaultDict)
            valueDict['revision_id']=revision.id
            valueDict.update(dict(zip(keyList, idChunk)))
            self.dbCursor.execute(query, valueDict)
            r = self.dbCursor.fetchmany()
            while len(r) > 0:
                for (clsId, weight, changeDt, changeFlag, srcId,
                     ref, iid) in r:
                    info = Utilities.Struct()
                    info.classification_id = clsId
                    info.weight = weight
                    info.change_dt = changeDt.date()
                    info.src = srcId
                    info.ref = ref
                    issueClsDict.setdefault(iid, list()).append(
                        (changeDt.date(), changeFlag, info))
                r = self.dbCursor.fetchmany()
        
        # Construct classification history for sub-issues
        for mid in missingIds:
            cache.addKey(mid)
            clsHistory = issueClsDict.get(mid.getIDString(), list())
            for this in clsHistory:
                cache.addChange(mid, this[0], this[1], this[2])
        retval = cache.getClassifications(list(zip(assetList, issueList)),
                                          date, level)
        return retval
           
    def getMktAssetClassifications(self, revision, assetList, date, marketDB, level=None, dtRange=False):
        """Returns the classification of all the passed issues or sub-issues
        within a particular MarketDB classification revision
        on the supplied date.
        The return value is a dict mapping assets to structs with
        the information from the classification_constituent table:
        classification_id, weight, src, ref, and
        classification (as returned by getClassificationByID).
        """
        self.log.debug('loading Market Classification %d/%s for %d assets',
                       revision.member_id, revision.from_dt, len(assetList))
        if revision.id not in self.marketClassificationCaches:
            cache = ClassificationCache(marketDB, revision, False)
            self.marketClassificationCaches[revision.id] = cache
        else:
            cache = self.marketClassificationCaches[revision.id]
        issueList = list()
        for a in assetList:
            if isinstance(a, SubIssue):
                issueList.append(a.getModelID())
            else:
                issueList.append(a)
        missingIds = cache.getMissingKeys(issueList)
        self.log.debug('adding %d assets to cache', len(missingIds))
        (axidStr, modelMarketMap) = self.getMarketDB_IDs(marketDB, missingIds)
        # Get classification history for MarketDB IDs
        axidClsDict = dict()
        INCR=200
        keyList = ['key%d' % i for i in range(INCR)]
        defaultDict = dict([(a, None) for a in keyList])       
        query = """SELECT ca.classification_id, ca.weight, ca.change_dt,
          ca.change_del_flag, ca.src_id, ca.ref, ca.axioma_id
        FROM classification_const_active ca
        WHERE ca.axioma_id IN (%(keys)s) AND ca.revision_id=:revision_id
        ORDER BY change_dt ASC
        """ % dict(keys=','.join([':%s' % i for i in keyList]))
        for idChunk in listChunkIterator(axidStr, INCR):
            valueDict = dict(defaultDict)
            valueDict['revision_id']=revision.id
            valueDict.update(dict(zip(keyList, idChunk)))
            marketDB.dbCursor.execute(query, valueDict)
            r = marketDB.dbCursor.fetchmany()
            while len(r) > 0:
                for (clsId, weight, changeDt, changeFlag, srcId,
                     ref, aid) in r:
                    info = Utilities.Struct()
                    info.classification_id = clsId
                    info.weight = weight
                    info.change_dt = changeDt.date()
                    info.src = srcId
                    info.ref = ref
                    axidClsDict.setdefault(aid, list()).append(
                        (changeDt.date(), changeFlag, info))
                r = marketDB.dbCursor.fetchmany()
        
        # Construct classification history for sub-issues
        for mid in missingIds:
            cache.addKey(mid)
            axidHistory = sorted(modelMarketMap.get(mid, list()))
            for (fromDt, thruDt, axid) in axidHistory:
                clsHistory = axidClsDict.get(axid.getIDString(), list())
                clsHistory.append((datetime.date(2999,12,31), 'Y', None))
                for (this, next) in zip(clsHistory[:-1], clsHistory[1:]):
                    if this[0] < thruDt and next[0] > fromDt:
                        cache.addChange(mid, max(this[0], fromDt), this[1], this[2])
        if dtRange:
            return cache.getClassificationRange(list(zip(assetList, issueList)), date, level)
        retval = cache.getClassifications(list(zip(assetList, issueList)), date, level)
        return retval
           
    def getExcludeTypes(self, date, marketDB, excludeLogic='ESTU_EXCLUDE'):
        """Returns the list of asset types to be excluded for a particular
        date and type of exclusion as given by the exclude_logic table in marketDB
        """
        marketDB.dbCursor.execute("""select cr.code from classification_exclude cl, exclude_logic lg, classification_ref cr
        where cl.EXCLUDE_LOGIC_ID=lg.ID and lg.exclude_logic=:excl_arg and cl.CLASSIFICATION_ID=cr.ID
        and from_dt<=:date_arg and thru_dt>:date_arg""", date_arg=date, excl_arg=excludeLogic)
        return [r[0] for r in marketDB.dbCursor.fetchall()]

    def getCumulativeRiskFreeRate(self, isoCode, date):
        """Returns the cumulative risk free rate for the given currency on
        the given date. The currency is specified by its ISO currency code.
        None is returned if no cumulative rate is available.
        """
        self.dbCursor.execute("""SELECT cumulative FROM currency_risk_free_rate
        WHERE currency_code = :iso_arg AND dt = :dt_arg""",
                              iso_arg=isoCode, dt_arg=date)
        r = self.dbCursor.fetchall()
        if len(r) == 0:
            return None
        elif len(r) == 1:
            return r[0][0]
        raise KeyError('More than one risk free rate for %s on %s'\
              % (isoCode, date))
    
    def getDateRange(self, rmgList, startDate, endDate, forceEndDate=False,
            excludeWeekend=False, tradingDaysBack=None, calendarDaysBack=None):
        """Returns a list of the trading days in the given closed interval.
        If tradingDaysBack is specified, then it returns a given history
        of trading days back from the end-date.
        If calendarDaysBack is specified, it returns a list of trading days
        between endDate and a endDate-calendarDaysBack
        The final list is chronologically sorted
        """
        # Pull up start date if we're given a number of days back
        if calendarDaysBack is not None:
            startDate = endDate - datetime.timedelta(calendarDaysBack-1)
            tradingDaysBack = None

        # If we don't care about dates relating to RMGs, return a simple date list
        if rmgList is None:
            dates = [startDate + datetime.timedelta(i) for i in
                    range((endDate - startDate).days + 1)]
            if excludeWeekend:
                dates = [d for d in dates if d.weekday() < 5]
            dates.sort()
            return dates

        if type(rmgList) is not list:
            rmgs = [rmgList]
        else:
            rmgs = list(rmgList)

        # Decide whether or not weekends are wanted
        if excludeWeekend:
            extra_arg = "AND TO_CHAR(dt, 'DY') NOT IN ('SAT', 'SUN')"
        else:
            extra_arg = ''

        # Pull up the dates
        if tradingDaysBack is None:
            # Query for pulling dates within a range
            query = """SELECT DISTINCT dt FROM (
                SELECT dt FROM rmg_calendar WHERE
                rmg_id IN (%(rmg_ids)s) 
                AND dt >= :start_date AND dt <= :end_date %(extra_arg)s)
                ORDER BY dt""" % {
                    'rmg_ids':  ','.join([str(n.rmg_id) for n in rmgs]),
                    'extra_arg': extra_arg}
            self.dbCursor.execute(query, start_date=startDate, end_date=endDate)
        else:
            # Query for pulling a certain number of dates
            query = """SELECT * FROM (SELECT DISTINCT dt FROM (
                SELECT dt FROM rmg_calendar WHERE
                rmg_id IN (%(rmg_ids)s) AND dt <= :date_arg %(extra_arg)s)
                ORDER BY dt DESC)
                WHERE ROWNUM <= :days_arg""" % {
                    'rmg_ids': ','.join([str(n.rmg_id) for n in rmgs]),
                    'extra_arg': extra_arg}
            self.dbCursor.execute(query, date_arg=endDate, days_arg=tradingDaysBack+1)

        dates = [self.oracleToDate(d[0]) for d in self.dbCursor.fetchall()]

        # Temporary fix to remove 1st of Jan
        if excludeWeekend:
            nyd = set([d for d in dates if (d.day==1) and (d.month==1)])
            dates = list(set(dates).difference(nyd))

        # Final processing of end-date and length
        dates.sort()
        if endDate not in dates and forceEndDate:
            dates.append(endDate)
        if (tradingDaysBack is not None) and (len(dates) > tradingDaysBack):
            dates = dates[-tradingDaysBack:]

        return dates
    
    def getAllRMGDateRange(self, date, daysBack, excludeWeekend=True):
        """Get list of dates for every live RMG
        """
        return self.getDateRange(self.getAllRiskModelGroups(),
                None, date, calendarDaysBack=daysBack, excludeWeekend=excludeWeekend)

    def getAllRMGDates(self, date, daysBack, excludeWeekend=True):
        """To be deleted when we are sure it's safe
        """
        return self.getAllRMGDateRange(date, daysBack, excludeWeekend=excludeWeekend)

    def getCountryDateRange(self, rmg, startDate, endDate, excludeWeekend=False):
        """To be deleted when we are sure it's safe
        """
        return self.getDateRange(rmg, startDate, endDate, excludeWeekend=excludeWeekend)

    def getEveryDateRange(self, startDate, endDate, excludeWeekend=False):
        """Returns a list of every single date between two values
        To be deleted when we are sure it's safe
        """
        return self.getDateRange(None, startDate, endDate, excludeWeekend=excludeWeekend)

    def getAllRMDates(self, rmgList, date, numDays, excludeWeekend=True, fitNum=False):
        """Returns a list of dates such that each model geography
        contains numDays trading dates. These are then combined such
        that the total list has every trading day from the earliest to
        the latest
        To be deleted when we are sure it's safe
        """
        allDays = set()
        rmgTradingDays = dict()
        for rmg in rmgList:
            dateList = set(self.getDates([rmg], date, numDays-1))
            allDays = allDays | dateList
            rmgTradingDays[rmg] = dateList
        allDays = sorted(allDays)
        if len(allDays) == 0:
            return (dict(), [])
        allDays = self.getDateRange(rmgList, allDays[0], allDays[-1], excludeWeekend=excludeWeekend)
        if fitNum and len(allDays) > numDays:
            allDays = allDays[-numDays:]
        return (rmgTradingDays, list(allDays))

    def getDates(self, rmgList, date, daysBack, excludeWeekend=False,
            forceLatestDate=False, fitNum=False):
        """Returns a list of the daysBack trading days for the given
        risk model groups preceeding date.
        date is included in the list, so the total length of the list
        is daysBack + 1.
        If fitNum is True, then the length of the list will be at most daysBack
        To be deleted when we are sure it's safe
        """
        if type(rmgList) is not list:
            rmgs = [rmgList]
        else:
            rmgs = list(rmgList)

        if excludeWeekend:
            extra_arg = "AND TO_CHAR(dt, 'DY') NOT IN ('SAT', 'SUN')"
        else:
            extra_arg = ''
        query = """SELECT * FROM (SELECT DISTINCT dt FROM (
            SELECT dt FROM rmg_calendar WHERE
            rmg_id IN (%(rmg_ids)s) AND dt <= :date_arg %(extra_arg)s) 
            ORDER BY dt DESC)
            WHERE ROWNUM <= :days_arg""" % {
                'rmg_ids': ','.join([str(n.rmg_id) for n in rmgs]),
                'extra_arg': extra_arg}
        self.dbCursor.execute(query, date_arg=date, days_arg=daysBack+1)
        dates = [self.oracleToDate(d[0]) for d in self.dbCursor.fetchall()]
        dates.reverse()
        if date not in dates and forceLatestDate:
            dates.append(date)
        if fitNum and len(dates) > daysBack:
            dates = dates[-daysBack:]
        return dates

    def getPreviousTradingDay(self, rmg, date):
        """Returns the most recent trading day in the risk model group rmg
        prior to date based on the trading calendar in rmg_calendar.
        If no previous trading day can be found, None is returned.
        """
        query = """SELECT * FROM (SELECT dt FROM rmg_calendar WHERE
        rmg_id = :rmg_arg AND dt < :date_arg ORDER BY sequence DESC)
        WHERE ROWNUM <= 1"""
        self.dbCursor.execute(query, rmg_arg=rmg.rmg_id, date_arg=date)
        r = self.dbCursor.fetchall()
        assert(len(r) <= 1)
        if len(r) == 0:
            return None
        return self.oracleToDate(r[0][0])
    
    def getActiveMarketsForDates(self, rmgList, startDate, endDate):
        """Returns a list of (date, number) where number corresponds
        to the number of markets within the given rmgList which have
        trading activity.
        Period covered is from startDate to endDate (inclusive).
        """
        query = """SELECT dt, count(rmg_id) FROM rmg_calendar
        WHERE rmg_id IN (%(rmg_ids)s) 
        AND dt <= :end_date AND dt >= :start_date 
        GROUP BY dt ORDER BY dt""" % {
                'rmg_ids': ','.join([str(n.rmg_id) for n in rmgList])}
        self.dbCursor.execute(query, 
                              start_date=startDate,
                              end_date=endDate)
        values = [(self.oracleToDate(r[0]), r[1])
                     for r in self.dbCursor.fetchall()]
        return values
    
    def getTradingMarkets(self, date):
        """Returns a set of rmg_id with the markets trading on the
        given date.
        """
        query = """SELECT rmg_id FROM rmg_calendar
        WHERE dt = :date_arg"""
        self.dbCursor.execute(query, date_arg=date)
        return set(r[0] for r in self.dbCursor.fetchall())
    
    def getFactorCovariances(self, rmi, subFactors, returnDF=False):
        """Returns the factor covariance matrix of the given factors for the
        risk model instance as an array.
        Covariances that are not present in the database are set to zero.
        The return value is an m by m array where m is the number of factors.
        """
        mat = numpy.zeros((len(subFactors), len(subFactors)), float)
        subIdxMap = dict(zip([s.subFactorID for s in subFactors],
                             list(range(len(subFactors)))))
        for subIdChunk in listChunkIterator(list(subIdxMap.keys()), 500):
            subIds = ','.join([str(id) for id in subIdChunk])
            query = """SELECT sub_factor1_id, sub_factor2_id, value
                       FROM rmi_covariance 
                       WHERE rms_id = :rms_arg AND dt=:date_arg
                       AND sub_factor1_id IN (%s)""" % subIds
            self.dbCursor.execute(query, rms_arg=rmi.rms_id,
                                  date_arg=rmi.date)
            r = self.dbCursor.fetchall()
            for (f1, f2, v) in r:
                if f2 not in subIdxMap.keys():
                    continue
                i1 = subIdxMap[f1]
                i2 = subIdxMap[f2]
                mat[i1, i2] = v
                mat[i2, i1] = v
        if returnDF:
            fnames = [sf.factor.name for sf in subFactors]
            return pandas.DataFrame(mat, index=fnames, columns=fnames)
        return mat
    
    def getFactorExposureMatrix(self, rmi, expM, subFactorMap):
        """Fills in the given exposure matrix from the database.
        rmi: RiskModelInfo structure which provides the rms_id and the date
        expM: ExposureMatrix with the requested sub-issues and factors
        subFactorMap: name->SubFactor dictionary
        """
        logging.debug('getFactorExposureMatrix: begin')
        subFactors = [subFactorMap[factorName] for factorName
                      in expM.getFactorNames()]
        subids = expM.getAssets()
        mat = self.getFactorExposures(rmi, subFactors, subids)
        assert(mat.shape == expM.data_.shape)
        expM.data_ = mat
        logging.debug('getFactorExposureMatrix: end')
        
    def getFactorExposures(self, rmi, subFactors, subids):
        """Returns the factor exposures of the given factors for the
        risk model instance as an array.
        Exposures that are not present in the database are masked.
        The return value is an m by n array where m is the number of
        factors and n the number of subids.
        """
        mat = Matrices.allMasked((len(subFactors), len(subids)))
        sidIdx = dict((j.getSubIDString(),i) for (i,j) in enumerate(subids))
        sfidIdx = dict((j.subFactorID,i) for (i,j) in enumerate(subFactors))
        for sfChunk in listChunkIterator(subFactors, 500):
            query = """SELECT sub_factor_id, sub_issue_id, fe.value
                       FROM rmi_factor_exposure fe
                       WHERE fe.rms_id = :rms_arg AND fe.dt = :date_arg
                       AND sub_factor_id IN (%(factors)s)""" % {
                'factors': ','.join(str(i.subFactorID) for i in sfChunk) }
            self.dbCursor.execute(query, rms_arg=rmi.rms_id,
                                  date_arg=rmi.date)
            r = self.dbCursor.fetchmany()
            numIds = len(subids)
            while len(r) > 0:
                indices = [sfidIdx[fIdx] * numIds + sidIdx[sid] for (fIdx, sid, val) in r if sid in sidIdx]
                values = [val for (fIdx, sid, val) in r if sid in sidIdx]
                ma.put(mat, indices, values)
                r = self.dbCursor.fetchmany()
        return mat
    
    def getFactorExposureMatrixNew(self, rmi, expM, subFactorMap):
        """Returns the factor exposures of the given factors for the
        risk model instance as an array.
        Exposures that are not present in the database are masked.
        The return value is an m by n array where m is the number of
        factors and n the number of subids.
        """
        logging.debug('getFactorExposureMatrixNew: begin')
        binaryColumns = set()
        columnNames = list()
        columnIndices = list()
        binaryIndexMap = dict()
        for (fIdx, factorName) in enumerate(expM.getFactorNames()):
            fType = expM.getFactorType(factorName)
            subFactor = subFactorMap[factorName]
            if fType == expM.IndustryFactor:
                binaryColumns.add('binary_industry')
                binaryIndexMap[subFactor.subFactorID] = fIdx
            elif fType == expM.CurrencyFactor:
                binaryColumns.add('binary_currency')
                binaryIndexMap[subFactor.subFactorID] = fIdx
            elif fType == expM.CountryFactor:
                binaryColumns.add('binary_country')
                binaryIndexMap[subFactor.subFactorID] = fIdx
            else:
                columnNames.append(self.getSubFactorColumnName(subFactorMap[factorName]))
                columnIndices.append(fIdx)
        query = """SELECT sub_issue_id, %(sfColumns)s
        FROM  %(table)s WHERE dt = :date_arg""" % {
            'table': self.getWideExposureTableName(rmi.rms_id),
            'sfColumns': ','.join(columnNames + list(binaryColumns))
            }
        self.dbCursor.execute(query, date_arg=rmi.date)
        sidIdx = dict([(j,i) for (i,j) in enumerate([
                        i.getSubIDString() for i in expM.getAssets()])])
        numIds = len(expM.getAssets())
        r = self.dbCursor.fetchmany()
        numFullColumns = len(columnNames)
        mat = expM.getMatrix()
        while len(r) > 0:
            indexValPairs = []
            for sidData in r:
                sid = sidData[0]
                myIdx = sidIdx.get(sid)
                if myIdx is None:
                    continue
                indexValPairs.extend([
                    (fIdx * numIds + myIdx, val) for (fIdx, val)
                    in zip(columnIndices, sidData[1:1+numFullColumns])
                    if val is not None])
                indexValPairs.extend([(binaryIndexMap[sf] * numIds + myIdx,
                                       1.0) for sf
                                      in sidData[1+numFullColumns:] if sf in binaryIndexMap])
            if indexValPairs:
                (indices, values) = zip(*indexValPairs)
                ma.put(mat, indices, values)
            r = self.dbCursor.fetchmany()
             
        meanDict = dict()
        stdevDict = dict()
        try:
            # Grab the standardization statistics if they exist in the DB
            query = """SELECT sub_factor_id, mean FROM rms_stnd_exp
            WHERE dt = :date_arg AND rms_id = :rms_arg"""
            self.dbCursor.execute(query, date_arg=rmi.date, rms_arg=rmi.rms_id)
            ret = self.dbCursor.fetchall()
            meanDict = dict((fid, val) for (fid, val) in ret)
            query = """SELECT sub_factor_id, stdev FROM rms_stnd_exp
            WHERE dt = :date_arg AND rms_id = :rms_arg"""
            self.dbCursor.execute(query, date_arg=rmi.date, rms_arg=rmi.rms_id)
            ret = self.dbCursor.fetchall()
            stdevDict = dict((fid, val) for (fid, val) in ret)
        except:
            logging.info('No standardization statistics in the DB')
        expM.meanDict = meanDict
        expM.stdevDict = stdevDict

        logging.debug('getFactorExposureMatrixNew: end')
    
    def getDescriptorStndStats(self, rmi, descriptorMap=None):
        """Pulls out the standardisation stats for the descriptors
        used in a particular model instance
        """
        query = """SELECT descriptor_id, mean, stdev FROM rms_stnd_desc
        WHERE dt = :date_arg AND rms_id = :rms_arg"""
        self.dbCursor.execute(query, date_arg=rmi.date, rms_arg=rmi.rms_id)
        ret = self.dbCursor.fetchall()

        # Return subset of values if required
        if descriptorMap is not None:
            descIDList = list(descriptorMap.values())
            meanDict = dict((d_id, mean) for (d_id, mean, stdev) in ret if d_id in descIDList)
            stdDict = dict((d_id, stdev) for (d_id, mean, stdev) in ret if d_id in descIDList)
        else:
            meanDict = dict((d_id, mean) for (d_id, mean, stdev) in ret)
            stdDict = dict((d_id, stdev) for (d_id, mean, stdev) in ret)

        logging.info('Retrieved standardisation statistics for %d descriptors', len(meanDict))
        return meanDict, stdDict

    def getFactors(self, factorDescr, upper=False):
        """Returns the factor IDs for the given factor descriptions.
        The return value is a list of factor IDs. The ID is None if
        a factor is not present in the database.
        """
        factors = []
        for fn in factorDescr:
            if not upper:
                self.dbCursor.execute("SELECT factor_id FROM factor WHERE description=:desc_arg", desc_arg=fn)
            else:
                self.dbCursor.execute("SELECT factor_id FROM factor WHERE UPPER(description)=:desc_arg and rownum <= 1", desc_arg=fn) 
            r = self.dbCursor.fetchall()
            assert(len(r) <= 1)
            if len(r) == 1:
                factors.append(r[0][0])
            else:
                factors.append(None)
        return factors

    def getGICSExposures(self, date, subIssues, level='Sectors', clsDate=datetime.date(2016,9,1)):
        """Build exposure matrix of GICS industry groups or sectors
        Note - returns transposed (i.e. correct) matrix of N assets by P factors
        """
        industryClassification = Classification.GICSIndustries(clsDate)
        logging.info('Building industry classification, level %s, version date: %s', level, clsDate)
        if level == 'Sectors':
            parentName = 'Sectors'
            level = -2
        else:
            parentName = 'Industry Groups'
            level = -1

        parents = industryClassification.getClassificationParents(parentName, self)
        factorList = [f.description for f in parents]
        exposures = industryClassification.getExposures(date, subIssues, factorList, self, level=level)
        return ma.transpose(exposures), factorList

    def getIBESCurrencyItem(self, item, startDate, endDate, subids, modelDate, marketDB, convertTo,
                            splitAdjust=None, bufferPeriod=2*366, namedTuple=None):
        """For each sub-issue, load the values for the given item
        as effective on 'modelDate' for filing dates between startDate
        and endDate (inclusive).
        convertTo is a dictionary that for each SubIssue in subids must
        contain the currency ID to which the values should be converted.
        The values are left in their local currency if the currencyID is None.
        If convertTo is None then no currency conversion will take place.
        If convertTo is an integer, then it is used as the currencyID for all
        assets.
        If splitAdjust is None then the values are returned as they are
        in the fundamental data table. Otherwise they are adjusted
        based on the share adjustments (asset_dim_af.shares_af) so that
        all value are based on shares at endDate. For per-share values
        splitAdjust should be set to 'divide', for share values like common
        shares outstanding to 'multiply'.
        The return value is a list with an entry for each asset in subids.
        Each entry is a list of (date, value, currency_id) tuples
        sorted in chronological order.
        """
        self.log.debug('start loading %s between %s and %s for %d assets, %s, %s',
                       item, startDate, endDate, len(subids), splitAdjust, convertTo)

        # If currency conversions required
        if (convertTo is not None) and (not self.currencyCache):
            raise ValueError('Currency conversion requires a ForexCache object present')

        # get cache and create it if it's missing or doesn't cover the right date range
        cache = self.ibesDataCache.get(item)
        bufferDates = datetime.timedelta(days=(bufferPeriod))
        # Here we have to be careful. Refresh the cache one year earlier than the 
        # cache end date because we are forward looking and cannot wait till it dies
        # same for cache start date, we should extend backward when instantiation
        if (cache is None) or (startDate < cache.startDate) or (endDate > cache.endDate-bufferDates):
            bufferDates = datetime.timedelta(days=(2*bufferPeriod))
            if cache is not None:
                cacheStart = min(cache.startDate, startDate-bufferDates)
                cacheEnd = max(cache.endDate, endDate+bufferDates)
            else:
                cacheStart = startDate - bufferDates
                cacheEnd = endDate + bufferDates
                
            cache = IBESDataCache(cacheStart, cacheEnd, self.currencyCache)
            self.log.debug('Created cache for IBES %s with range %s to %s',
                            item, cache.startDate, cache.endDate)
            self.ibesDataCache[item] = cache

        # Get the item_code for the item requested
        itemCodes = self.getFundamentalItemCodes('sub_issue_esti_currency', marketDB)
        code = itemCodes[item]

        missing = cache.getMissingIds(subids)
        self.getIBESCurrencyItemInternal(cache, missing, code, True)
        self.log.debug('Done loading IBES fundamental data')

        # Pull up history of adjustment factors
        if splitAdjust is not None:
            afs = self.getShareAdjustmentFactors(startDate, endDate, subids, marketDB)
            if splitAdjust == 'divide':
                splitDivide = True
            elif splitAdjust == 'multiply':
                splitDivide = False
            else:
                raise ValueError('Unsupported splitAdjust value %s' % splitAdjust)
            self.log.debug('Done get split adjustments')

        resultValues = list()
        if isinstance(convertTo, dict):
            ccyLookup = lambda x : convertTo[x]
        else:
            ccyLookup = lambda x : convertTo

        # Get effective values in date range in chronological order
        # converted to target currency and adjusted if necessary
        for sid in subids:
            sidValues = cache.getAssetHistory(sid, startDate, endDate, modelDate, ccyLookup(sid),
                            getEarliest=False, forceRun=self.forceRun, namedTuple=namedTuple)
            if splitAdjust is not None and len(sidValues) > 0 \
                   and len(afs[sid]) > 0:
                sidAFs = afs[sid]
                caf = 1.0
                aIdx = -1
                for vIdx in range(-1,-len(sidValues)-1,-1):
                    sidVal = sidValues[vIdx]
                    dt = sidVal[0]
                    while -aIdx <= len(sidAFs) and dt < sidAFs[aIdx][0]:
                        caf *= sidAFs[aIdx][1]
                        aIdx -= 1
                    if splitDivide:
                        convertVal = (dt, sidVal[1] / caf) + sidVal[2:]
                    else:
                        convertVal = (dt, sidVal[1] * caf) + sidVal[2:]
                    if namedTuple is not None:
                        convertVal = namedTuple(*convertVal)
                    sidValues[vIdx] = convertVal
            resultValues.append(sidValues)

        self.log.debug('Done processing %s', item)
        return resultValues

    def getIBESCurrencyItemLegacy(self, item, startDate, endDate,
                            subids, date, marketDB, convertTo,
                            splitAdjust=None, getEarliest=False,
                            maxEndDate=None, useNewTable=False):
        """For each sub-issue, load the values for the given item
        as effective on 'date' for filing dates between startDate
        and endDate (inclusive).
        marketDB is a MarketDB object which will be used to resolve
        the item name and to retrieve adjustment factors when the data
        is per share and hence needs to be adjusted to be comparable.
        convertTo is a dictionary that for each SubIssue in subids must
        contain the currency ID to which the values should be converted.
        The values are left in their local currency if the currencyID is None.
        If convertTo is None then no currency conversion will take place.
        If convertTo is an integer, then it is used as the currencyID for all
        assets.
        If splitAdjust is None then the values are returned as they are
        in the fundamental data table. Otherwise they are adjusted
        based on the share adjustments (asset_dim_af.shares_af) so that
        all value are based on shares at endDate. For per-share values
        splitAdjust should be set to 'divide', for share values like common
        shares outstanding to 'multiply'.
        If useNewTable is True, data is retrieved from sub_issue_estimate data,
        otherwise it is retrieved from sub_issue_esti_currency.
        The return value is a list with an entry for each asset in subids.
        Each entry is a list of (date, value, currency_id) tuples
        sorted in chronological order.
        """
        self.log.debug('start loading %s between %s and %s for %d assets, %s, %s',
                       item, startDate, endDate, len(subids), splitAdjust, convertTo)
        # If currency conversions required
        if convertTo is not None:
            if not self.currencyCache:
                raise ValueError('Currency conversion requires a ForexCache object present')
        # get cache and create it if it's missing or doesn't cover the right date range
        cache = self.ibesDataCache.get(item)
        # here we have to be careful. Refresh the cache one year earlier than the
        # cache end date because we are forward looking and cannot wait till it dies
        # same for cache start date, we should extend backward when instantiation

        if cache is None or cache.startDate > startDate or \
                cache.endDate - datetime.timedelta(days=366) < endDate:
            if cache is not None:
                cacheStart = min(cache.startDate,
                                 startDate - datetime.timedelta(days= (2*366)))
                cacheEnd = max(cache.endDate,
                               endDate + datetime.timedelta(days= (2*366)))
            else:
                cacheStart = startDate - datetime.timedelta(days = (2*366))
                cacheEnd = endDate + datetime.timedelta(days= (2*366))

            cache = IBESDataCache(cacheStart, cacheEnd, self.currencyCache)
            self.log.debug('Created cache for IBES %s with range %s to %s', item,
                          cache.startDate,
                          cache.endDate)
            self.ibesDataCache[item] = cache

        #To get the item_code for the item requested
        itemCodes = self.getFundamentalItemCodes('sub_issue_esti_currency', marketDB)
        code = itemCodes[item]
        missing = cache.getMissingIds(subids)
        self.getIBESCurrencyItemInternal(cache, missing, code, useNewTable)
        self.log.debug('done loading IBES fundamental data')
        if splitAdjust is not None:
            afs = self.getShareAdjustmentFactors(startDate, cache.endDate, subids,
                                                 marketDB)
            if splitAdjust == 'divide':
                splitDivide = True
            elif splitAdjust == 'multiply':
                splitDivide = False
            else:
                raise ValueError('Unsupport splitAdjust value %s'%splitAdjust)
        self.log.debug('done get split adjustments')

        resultValues = list()
        if isinstance(convertTo, dict):
            ccyLookup = lambda x : convertTo[x]
        else:
            ccyLookup = lambda x : convertTo

        if maxEndDate is None:
            cacheEndDate = cache.endDate
        else:
            cacheEndDate = maxEndDate

        for sid in subids:
            # get effective values in date range in chronological order
            # converted to target currency
            sidValues = cache.getAssetHistory(sid, startDate, cacheEndDate, date,
                                              ccyLookup(sid), getEarliest=getEarliest)
            if splitAdjust is not None and len(sidValues) > 0 \
                   and len(afs[sid]) > 0:
                sidAFs = afs[sid]
                caf = 1.0
                aIdx = -1
                for vIdx in range(-1,-len(sidValues)-1,-1):
                    sidVal = sidValues[vIdx]
                    dt = sidVal[0]
                    while -aIdx <= len(sidAFs) and dt < sidAFs[aIdx][0]:
                        caf *= sidAFs[aIdx][1]
                        aIdx -= 1
                    if splitDivide:
                        sidValues[vIdx] = (dt, sidVal[1] / caf, sidVal[2], sidVal[3])
                    else:
                        sidValues[vIdx] = (dt, sidVal[1] * caf, sidVal[2], sidVal[3])
            resultValues.append(sidValues)
        self.log.debug('done processing %s', item)
        return resultValues

    def getIBESCurrencyItemInternal(self, cache, missing, code, useNewTable):
        self.log.debug('Adding %d assets to cache (%d present)', len(missing),
                       len(cache.assetValueMap))
        if len(missing) == 0:
            return
        INCR = 10
        sidArgs = ['sid%d' % i for i in range(INCR)]
        defaultDict = dict([(arg, None) for arg in sidArgs])
        defaultDict['item'] = code
        defaultDict['startDt'] = cache.startDate
        defaultDict['endDt'] = cache.endDate
        query = """SELECT sub_issue_id, dt, value, currency_id,
        eff_dt, eff_del_flag
        FROM %(table)s a
        WHERE item_code=:item AND dt BETWEEN :startDt AND :endDt
        AND sub_issue_id IN (%(sids)s)
        AND rev_dt=(SELECT MAX(rev_dt) FROM %(table)s b
                    WHERE a.sub_issue_id = b.sub_issue_id
                    AND a.item_code = b.item_code
                    AND a.dt = b.dt
                    AND a.eff_dt = b.eff_dt)
        AND rev_del_flag = 'N' ORDER BY dt, eff_dt""" % {
        'sids': ','.join([':%s' % arg for arg in sidArgs]), 
        'table': 'sub_issue_estimate_data' if useNewTable else 'sub_issue_esti_currency'}
        
        sidStrs = sorted(sid.getSubIDString() for sid in missing)  
        itemDateMap = dict((sid, dict()) for sid in missing)
        for sidChunk in listChunkIterator(sidStrs, INCR):
            myDict = defaultDict.copy()
            myDict.update(dict(zip(sidArgs, sidChunk)))
            self.dbCursor.execute(query, myDict)
            for (sid, dt, val, ccyId, effDt, effDelFlag) \
                    in self.dbCursor.fetchall():
                sid = SubIssue(string=sid)
                dt = dt.date()
                effDt = effDt.date()
                val = float(val)
                ccyId = int(ccyId)
                itemDateMap[sid].setdefault(dt, []).append(
                    (effDt, effDelFlag, val, ccyId))
        for (sid, values) in itemDateMap.items():
            cache.addAssetValues(sid, sorted(values.items()))

    def getFundamentalCurrencyItem(self, item, startDate, endDate,
                                   subids, date, marketDB, convertTo,
                                   splitAdjust=None, useNewTable=False,
                                   requireConsecQtrData=None):
        """For each sub-issue, load the values for the given item
        as effective on 'date' for filing dates between startDate
        and endDate (inclusive).
        marketDB is a MarketDB object which will be used to resolve
        the item name and to retrieve adjustment factors when the data
        is per share and hence needs to be adjusted to be comparable.
        convertTo is a dictionary that for each SubIssue in subids must
        contain the currency ID to which the values should be converted.
        The values are left in their local currency if the currencyID is None.
        If convertTo is None then no currency conversion will take place.
        If convertTo is an integer, then it is used as the currencyID for all
        assets.
        If splitAdjust is None then the values are returned as they are
        in the fundamental data table. Otherwise they are adjusted
        based on the share adjustments (asset_dim_af.shares_af) so that
        all value are based on shares at endDate. For per-share values
        splitAdjust should be set to 'divide', for share values like common
        shares outstanding to 'multiply'.
        The return value is a list with an entry for each asset in subids.
        Each entry is a list of (date, value, currency_id) tuples
        sorted in chronological order.
        If useNewTable=True, will access data in sub_issue_fundamental_data
        instead of sub_issue_fund_currency.
        requireConsecQtrData should be set to None unless quarterly data are 
        being loaded and we require a given number of consecutive values per asset
        """
        self.log.debug('start loading %s between %s and %s for %d assets, %s, %s',
                       item, startDate, endDate, len(subids), splitAdjust, convertTo)
        if useNewTable:
            tableName = 'sub_issue_fundamental_data'
        else:
            tableName = 'sub_issue_fund_currency'
        # If currency conversions required
        if convertTo is not None:
            if not self.currencyCache:
                raise ValueError('Currency conversion requires a ForexCache object present')
        # get cache and create it if it's missing or doesn't cover the right date range
        cache = self.fundamentalDataCache.get(item)
        if cache is None or cache.startDate > startDate or cache.endDate < endDate:
            if cache is not None:
                cacheStart = min(cache.startDate, startDate)
                cacheEnd = max(cache.endDate, endDate + datetime.timedelta(days=366))
            else:
                cacheStart = startDate
                cacheEnd = endDate + datetime.timedelta(days=366)
            if useNewTable:
                cache = FundamentalDataCacheFY(cacheStart, cacheEnd, self.currencyCache)
            else:
                cache = FundamentalDataCache(cacheStart, cacheEnd, self.currencyCache)
            self.log.debug('Created cache for %s with range %s to %s', item, cache.startDate,
                          cache.endDate)
            self.fundamentalDataCache[item] = cache
            
        itemCodes = self.getFundamentalItemCodes(tableName, marketDB)
        code = itemCodes[item]
        logging.info('Loading item: %s, code: %s', item, code)
        missing = cache.getMissingIds(subids)
        self.getFundamentalCurrencyItemInternal(cache, missing, code, useNewTable)
        self.log.debug('done loading fundamental data')
        if splitAdjust is not None:
            afs = self.getShareAdjustmentFactors(startDate, endDate, subids,
                                                 marketDB)
            if splitAdjust == 'divide':
                splitDivide = True
            elif splitAdjust == 'multiply':
                splitDivide = False
            else:
                raise ValueError('Unsupport splitAdjust value %s').with_traceback(splitAdjust)
        self.log.debug('done get split adjustments')
        resultValues = list()
        if isinstance(convertTo, dict):
            ccyLookup = lambda x : convertTo[x]
        else:
            ccyLookup = lambda x : convertTo
        
        for sid in subids:
            # get effective values in date range in chronological order
            # converted to target currency
            sidValues = cache.getAssetHistory(sid, startDate, endDate, date,
                                              ccyLookup(sid), forceRun=self.forceRun)
            if splitAdjust is not None and len(sidValues) > 0 \
                   and len(afs[sid]) > 0:
                sidAFs = afs[sid]
                caf = 1.0
                aIdx = -1
                for vIdx in range(-1,-len(sidValues)-1,-1):
                    sidVal = sidValues[vIdx]
                    dt = sidVal[0]
                    while -aIdx <= len(sidAFs) and dt < sidAFs[aIdx][0]:
                        caf *= sidAFs[aIdx][1]
                        aIdx -= 1
                    if splitDivide:
                        if useNewTable:
                            sidValues[vIdx] = (dt, sidVal[1] / caf) + sidVal[2:]
                        else:
                            sidValues[vIdx] = (dt, sidVal[1] / caf, sidVal[2])
                    else:
                        if useNewTable:
                            sidValues[vIdx] = (dt, sidVal[1] * caf) + sidVal[2:]
                        else:
                            sidValues[vIdx] = (dt, sidVal[1] * caf, sidVal[2])
            resultValues.append(sidValues)

        if requireConsecQtrData is not None:
            consecData = [0]*len(resultValues)
            for i in range(len(resultValues)):
                if len(resultValues[i]) >= requireConsecQtrData:
                    qtrIntervals = [True if 85 <= (resultValues[i][j][0] - resultValues[i][j-1][0]).days <= 95 \
                        else False for j in range(1, len(resultValues[i]))]
                    if all(qtrIntervals):
                        consecData[i] = 1
            missing = set([idx for idx in range(len(resultValues)) if consecData[idx]==0])
            for qIdx in missing:
                resultValues[qIdx] = []

        self.log.debug('done processing %s', item)
        return resultValues
    
    def getFundamentalCurrencyItemInternal(self, cache, missing, code, useNewTable=False):
        self.log.debug('Adding %d assets to cache (%d present)', len(missing),
                       len(cache.assetValueMap))
        if len(missing) == 0:
            return
        INCR = 10
        sidArgs = ['sid%d' % i for i in range(INCR)]
        defaultDict = dict([(arg, None) for arg in sidArgs])
        defaultDict['item'] = code
        defaultDict['startDt'] = cache.startDate
        defaultDict['endDt'] = cache.endDate
        if useNewTable:
            tableName = 'sub_issue_fundamental_data'
            itemFieldName = 'item_code_id'
            extraFields = ', fiscal_year_end'
        else:
            tableName = 'sub_issue_fund_currency'
            itemFieldName = 'item_code'
            extraFields = ''
        query = """SELECT sub_issue_id, dt, value, currency_id,
          eff_dt, eff_del_flag %(extraFields)s
          FROM %(tableName)s a
          WHERE %(itemFieldName)s=:item AND dt BETWEEN :startDt AND :endDt
          AND sub_issue_id IN (%(sids)s)
          AND rev_dt=(SELECT MAX(rev_dt) from %(tableName)s b
             WHERE a.sub_issue_id=b.sub_issue_id AND a.%(itemFieldName)s=b.%(itemFieldName)s
             AND a.dt=b.dt and a.eff_dt=b.eff_dt)
          AND rev_del_flag='N' ORDER BY dt, eff_dt""" % {
            'sids': ','.join([':%s' % arg for arg in sidArgs]), 
            'tableName': tableName,
            'itemFieldName': itemFieldName,
            'extraFields': extraFields}
        sidStrs = sorted(sid.getSubIDString() for sid in missing)
        itemDateMap = dict((sid, dict()) for sid in missing)
        for sidChunk in listChunkIterator(sidStrs, INCR):
            myDict = defaultDict.copy()
            myDict.update(dict(zip(sidArgs, sidChunk)))
            self.dbCursor.execute(query, myDict)
            for row in self.dbCursor.fetchall():
                if useNewTable:
                    (sid, dt, val, ccyId, effDt, effDelFlag, yearEnd) = row
                    if yearEnd == 0:
                        yearEnd = dt.month
                else:
                    (sid, dt, val, ccyId, effDt, effDelFlag) = row
                sid = SubIssue(string=sid)
                dt = dt.date()
                effDt = effDt.date()
                val = float(val)
                ccyId = int(ccyId)
                values = (effDt, effDelFlag, val, ccyId)
                if useNewTable:
                    values += (yearEnd, )
                itemDateMap[sid].setdefault(dt, []).append(values)
        for (sid, values) in itemDateMap.items():
            cache.addAssetValues(sid, sorted(values.items()))

    def getFundamentalItemCodes(self, tableName, marketDB):
        """Returns a dictionary mapping item names to codes for the
        specified table.
        """
        if tableName == 'sub_issue_fund_currency':
            codeType = 'asset_dim_fund_currency:item_code'
        elif tableName == 'sub_issue_fund_number':
            codeType = 'asset_dim_fund_number:item_code'
        elif tableName == 'sub_issue_esti_currency':
            codeType = 'asset_dim_esti_currency:item_code'
        elif tableName == 'mdl_port_mcap_map':
            codeType = 'mdl_port_mcap_map:size_code'
        elif tableName == 'sub_issue_fundamental_data':
            codeType = 'asset_dim_xpsfeed_data:item_code'
        else:
            raise KeyError('Unsupported table %s' % tableName)
        marketDB.dbCursor.execute("""SELECT name, id FROM meta_codes
           WHERE code_type = :codeType""", codeType=codeType)
        return dict(marketDB.dbCursor.fetchall())

    def getMixFundamentalCurrencyItem(self, itemPrefix, startDate, endDate,
                                      subIssues, date, marketDB, convertTo, 
                                      splitAdjust=None, requireConsecQtrData=None):
        """For each sub-issue, load the values for the given item
        as effective on 'date' for filing dates between startDate
        and endDate (inclusive).
        marketDB is a MarketDB object which will be used to resolve
        the item name and to retrieve adjustment factors when the data
        is per share and hence needs to be adjusted to be comparable.
        convertTo is a dictionary that for each SubIssue in subids must
        contain the currency ID to which the values should be converted.
        The values are left in their local currency if the currencyID is None.
        If convertTo is None then no currency conversion will take place.
        If convertTo is an integer, then it is used as the currencyID for all
        assets.
        If splitAdjust is None then the values are returned as they are
        in the fundamental data table. Otherwise they are adjusted
        based on the share adjustments (asset_dim_af.shares_af) so that
        all value are based on shares at endDate. For per-share values
        splitAdjust should be set to 'divide', for share values like common
        shares outstanding to 'multiply'.

        If requireCondexQtrData is some number (e.g., 3), then quarterly 
        data is returned for a given subid only if there are values for at 
        least 3 consecutive quarters with the specified date range.

        Returns two lists, the first containing (dt, value, currency_id) 
        tuples and the second containing the frequency associated with the
        each data.
        """
        sidIdxMap = dict([(j,i) for (i,j) in enumerate(subIssues)])

        # First try loading quarterly data
        qtrData = self.getFundamentalCurrencyItem(itemPrefix + '_qtr',
                                                  startDate, endDate,
                                                  subIssues, date, marketDB,
                                                  convertTo=convertTo,
                                                  splitAdjust=splitAdjust)

        if requireConsecQtrData:
            consecData = [0]*len(qtrData)
            for i in range(len(qtrData)):
                if len(qtrData[i]) >= requireConsecQtrData:
                    qtrIntervals = [True if 85 <= (qtrData[i][j][0] - qtrData[i][j-1][0]).days <= 95 \
                            else False for j in range(1, len(qtrData[i]))]
                    if all(qtrIntervals):
                        consecData[i] = 1 
            missing = set([idx for idx in range(len(qtrData)) if consecData[idx]==0])
        else:
            missing = set([idx for idx in range(len(qtrData)) if len(qtrData[idx])==0])

        qtrIdx = [idx for idx in range(len(subIssues)) if idx not in missing]
        
        # Reporting
        annualSubIssues = [subIssues[k] for k in missing]
        pctMiss = 100.0 * len(missing) / float(len(subIssues))
        if len(annualSubIssues) > 0:
            if pctMiss > 50.0:
                logging.warning('Missing quarterly %s data for %.1f%% of %d assets, attempting annual',
                        itemPrefix, pctMiss, len(subIssues))
            else:
                logging.info('Missing quarterly %s data for %.1f%% of %d assets, attempting annual',
                        itemPrefix, pctMiss, len(subIssues))

        # Fall back to annual data
        annData = self.getFundamentalCurrencyItem(itemPrefix + '_ann',
                                             startDate, endDate,
                                             annualSubIssues, date, marketDB,
                                             convertTo=convertTo,
                                             splitAdjust=splitAdjust)

        from riskmodels.DescriptorRatios import DataFrequency
        AnnualFrequency = DataFrequency('_ann')
        QuarterlyFrequency = DataFrequency('_qtr')
        values = Matrices.allMasked(len(subIssues), dtype=object)
        frequencies = Matrices.allMasked(len(subIssues), dtype=object)

        # Combine annual and quarterly
        for qIdx in qtrIdx:
            values[qIdx] = qtrData[qIdx]
            frequencies[qIdx] = QuarterlyFrequency
        missingAnnual = []
        for (sIdx, sid) in enumerate(annualSubIssues):
            idx = sidIdxMap[sid]
            frequencies[idx] = AnnualFrequency
            values[idx] = annData[sIdx]
            if len(annData[sIdx]) < 1:
                missingAnnual.append(sid)

        # Reporting
        if len(missingAnnual) > 0:
            pctMiss = 100.0 * len(missingAnnual) / float(len(annualSubIssues))
            if pctMiss > 50.0:
                logging.warning('No annual data for %.1f%% of %d issues missing quarterly %s data',
                        pctMiss, len(annualSubIssues), itemPrefix)
            else:
                logging.info('No annual data for %.1f%% of %d issues missing quarterly %s data',
                        pctMiss, len(annualSubIssues), itemPrefix)
        logging.info('%s data Qtr/Ann: %.1f/%.1f %%',
                itemPrefix, 100*len(qtrIdx)/len(values),
                100*len(annualSubIssues)/len(values))
        return (values, frequencies)
    
    def getIndexConstituents(self, indexName, date, marketDB, rollBack=0, issueMapPairs=None):
        """Returns the assets and their weight in the index for the
        given date. If there is no data for the specified date, the most
        recent assets up to rollBack days ago will be returned.
        The return value is a list of (ModelID, weight) pairs.
        """
        self.log.debug('getIndexConstituents: begin')
        index = marketDB.getIndexByName(indexName, date)
        if index is None:
            return list()
        indexRev = marketDB.getIndexRevision(index, date, rollBack)
        #print 'index=%s, revision=%s' % (index,indexRev or 'None')
        assetWeightsMap = marketDB.getIndexConstituents(indexRev)
        if issueMapPairs == None:
            issueMapPairs = self.getIssueMapPairs(date)
        marketMap = dict([(i[1].getIDString(), i[0]) for i in issueMapPairs])
        bench = [(marketMap[i], assetWeightsMap[i]) for i
                 in list(assetWeightsMap.keys()) if i in marketMap]
        notMatched = [i for i in assetWeightsMap.keys() if i not in  marketMap]
        if len(notMatched) > 0:
            self.log.warning('%d unmapped assets in benchmark | %s | %s |  %s |%s|', len(notMatched), index.name, index.id, date, ','.join(notMatched))
            self.log.debug("Can't match assets in %s to ModelDB on %s: %s",
                           index.name, date, ','.join(notMatched))
        if len(bench) > 0:
            self.log.info('%d assets in %s benchmark', len(bench), index.name)
        else:
            self.log.debug('no assets in %s benchmark', index.name)
        self.log.debug('getIndexConstituents: end')
        return bench

    def getIssueIBESUSCodes(self, date, ids, marketDB):
        """Returns an issue to IBES US code map for the given issues valid on
        the given date.
        """
        self.loadMarketIdentifierHistoryCache(ids, marketDB, 'asset_dim_ibes_us',
                                              'code', cache=self.ibesUSCache)
        ibesCodes = [(mid, self.ibesUSCache.getAssetValue(mid, date))
                  for mid in ids]
        ibesCodes = dict([(i, j.id) for (i,j) in ibesCodes if j is not None])
        return ibesCodes

    def getIssueGVKEYs(self, date, ids, marketDB):
        """Returns an issue to IBES US code map for the given issues valid on
        the given date.
        """
        self.loadMarketIdentifierHistoryCache(ids, marketDB, 'asset_dim_gvkey',
                                              'id', cache=self.gvkeyUSCache)
        gvkeyCodes = [(mid, self.gvkeyUSCache.getAssetValue(mid, date))
                  for mid in ids]
        gvkeyCodes = dict([(i, j.id) for (i,j) in gvkeyCodes if j is not None])
        return gvkeyCodes
    
    def getIssueRICQuote(self, date, ids, marketDB):
        """Returns an issue to RIC_Quote map for the given issues valid on the given date"""
        INCR = 400
        resultDict = dict()
        bulkArgList = [('mid%d'%i) for i in range(INCR)]
        query ="""select c.modeldb_id, b.REUTERS_QUOTE from
                  MARKETDB_GLOBAL.FUTURE_LINKAGE_ACTIVE a, \
                  MARKETDB_GLOBAL.FUTURE_FAMILY_ATTR_ACTIVE_INT b, \
                  MODELDB_GLOBAL.FUTURE_ISSUE_MAP c where a.FUTURE_family_id = b.ID \
                  and c.modeldb_id IN(%(keys)s)
                  and a.future_axioma_id = c.marketdb_id
                  and c.from_dt<=:date_arg and c.thru_dt >:date_arg
                  and a.last_trading_date>=:date_arg
                  and b.from_dt<=:date_arg and b.thru_dt >:date_arg
                  """%{'keys':','.join([':%s'% i for i in bulkArgList])}
        defaultDict= dict([(arg, None) for arg in bulkArgList])
        defaultDict['date_arg']=date
        aidStr = [i.getIDString() for i in ids]
        for aidChunk in listChunkIterator(aidStr, INCR):
            valueDict= dict(defaultDict)
            valueDict.update(dict(zip(bulkArgList, aidChunk)))
            self.dbCursor.execute(query, valueDict)
            for  id, code  in self.dbCursor.fetchall():
                resultDict[MarketID.MarketID(string=id)] =  code
        return resultDict        


    def getIssueCUSIPs(self, date, ids, marketDB):
        """Returns an issue to CUSIP map for the given issues valid on
        the given date.
        """
        self.loadMarketIdentifierHistoryCache(ids, marketDB, 'asset_dim_cusip',
                                              'id', cache=self.cusipCache)
        cusips = [(mid, self.cusipCache.getAssetValue(mid, date))
                  for mid in ids]
        cusips = dict([(i, j.id) for (i,j) in cusips if j is not None])
        return cusips
    
    def getIssueSEDOLs(self, date, ids, marketDB):
        """Returns an issue to SEDOL map for the given issues valid on
        the given date.
        """
        self.loadMarketIdentifierHistoryCache(ids, marketDB, 'asset_dim_sedol',
                                              'id', cache=self.sedolCache)
        sedols = [(mid, self.sedolCache.getAssetValue(mid, date))
                  for mid in ids]
        sedols = dict([(i, j.id) for (i,j) in sedols if j is not None])
        return sedols
    
    def getIssueISINs(self, date, ids, marketDB):
        """Returns an issue to ISIN map for the given issues valid on
        the given date.
        """
        self.loadMarketIdentifierHistoryCache(ids, marketDB, 'asset_dim_isin',
                                              'id', cache=self.isinCache)
        isins = [(mid, self.isinCache.getAssetValue(mid, date)) for mid in ids]
        isins = dict([(i, j.id) for (i,j) in isins if j is not None])
        return isins
    
    def getIssueRICs(self, date, ids, marketDB):
        """Returns an issue to RIC map for the given issues valid on
        the given date.
        """
        self.loadMarketIdentifierHistoryCache(ids, marketDB, 'asset_dim_ric',
                                              'id', cache=self.ricCache)
        rics = [(mid, self.ricCache.getAssetValue(mid, date)) for mid in ids]
        rics = dict([(i, j.id) for (i,j) in rics if j is not None])
        return rics
    
    def getIssueNames(self, date, ids, marketDB):
        """Returns an issue to issue name map for the given issues valid on
        the given date.
        """
        self.loadMarketIdentifierHistoryCache(
            ids, marketDB, 'asset_dim_name', 'id', cache=self.nameCache)
        name = [(mid, self.nameCache.getAssetValue(mid, date)) for mid in ids]
        name = dict([(i, j.id) for (i,j) in name if j is not None])
        return name
    
    def getFutureNames(self, date, ids, marketDB):
        """Returns an issue to name map for the given issues valid on
        the given date.
        """
        futureNameCache = FromThruCache()
        self.loadMarketIdentifierHistoryCache(
            ids, marketDB, 'future_dim_name', 'id', cache=futureNameCache)
        name = [(mid, futureNameCache.getAssetValue(mid, date))
                  for mid in ids]
        name = dict([(i, j.id) for (i,j) in name if j is not None])
        return name

    def getIssueGVKEY(self, date, ids, marketDB):
        """Returns an issue to gvkey map for the given issues valid on
        the given date.
        """
        self.loadMarketIdentifierHistoryCache(
            ids, marketDB, 'asset_dim_gvkey', 'id', cache=self.gvkeyCache)
        gvkey = [(mid, self.gvkeyCache.getAssetValue(mid, date)) for mid in ids]
        gvkey = dict([(i, j.id) for (i,j) in gvkey if j is not None])
        return gvkey

    def getIssueRootIDs(self, date, ids, marketDB, issueMapPairs=None):
        """Returns an issue to issue parent/root map for the given
        issues valid on the given date.
        """
        if issueMapPairs == None:
            issueMapPairs = self.getIssueMapPairs(date)
        modelMarketMap = dict(issueMapPairs)
        marketModelMap = dict([(j,i) for (i,j) in issueMapPairs])
        marketIDs = [modelMarketMap[i] for i in ids]
        roots = marketDB.getAssetRootIDs(date, marketIDs)
        roots = dict([(marketModelMap[i], marketModelMap[j]) 
                    for (i,j) in roots.items() if j is not None
                    and i in marketModelMap and j in marketModelMap])
        return roots
    
    def getTradingCurrency(self, date, ids, marketDB, returnType='code'):
        """Returns a issue to trading currency map for the given issues
        valid on the given date.
        """
        self.loadMarketIdentifierHistoryCache(
            ids, marketDB, 'asset_dim_trading_currency', 'id',
            cache=self.tradeCcyCache)
        tccy = [(mid, self.tradeCcyCache.getAssetValue(mid, date))
                for mid in ids]
        if returnType == 'id':
            tccy = dict([(i, j.id) for (i,j) in tccy if j is not None])
        else:
            marketDB.dbCursor.execute("""SELECT id, code FROM currency_ref""")
            isoCodeMap = dict(marketDB.dbCursor.fetchall())
            tccy = dict([(i, isoCodeMap[j.id]) for (i,j) in tccy
                         if j is not None])
        return tccy
    
    def getFutureTradingCurrency(self, date, ids, marketDB, returnType='code'):
        """Returns a issue to trading currency map for the given issues
        valid on the given date.
        """
        myCache = FromThruCache()
        self.loadMarketIdentifierHistoryCache(
            ids, marketDB, 'future_dim_trading_currency', 'id',
            cache=myCache)
        tccy = [(mid, myCache.getAssetValue(mid, date))
                for mid in ids]
        if returnType == 'id':
            tccy = dict([(i, j.id) for (i,j) in tccy if j is not None])
        else:
            marketDB.dbCursor.execute("""SELECT id, code FROM currency_ref""")
            isoCodeMap = dict(marketDB.dbCursor.fetchall())
            tccy = dict([(i, isoCodeMap[j.id]) for (i,j) in tccy
                         if j is not None])
        return tccy
    
    def getIssueMapPairs(self, currentDate):
        """Returns (modelID, marketID) pairs for issues active on the
        given date"""
        self.dbCursor.execute("""SELECT modeldb_id, marketdb_id FROM issue_map
        WHERE from_dt <= :date_arg AND thru_dt > :date_arg 
        UNION 
        SELECT modeldb_id, marketdb_id FROM future_issue_map 
        WHERE from_dt<= :date_arg and thru_dt > :date_arg """,
                              date_arg=currentDate)

        return [(ModelID.ModelID(string=i), MarketID.MarketID(string=j))
                for (i,j) in self.dbCursor.fetchall()]
    
    def getPEIssueMapPairs(self, currentDate):
        """Returns (modelID, preqinFundId) pairs for issues active on the
        given date"""
        self.dbCursor.execute("""SELECT modeldb_id, preqin_fund_id FROM pe_issue_map
        WHERE from_dt <= :date_arg AND thru_dt > :date_arg""",
                              date_arg=currentDate)

        return [(ModelID.ModelID(string=i), j) for (i,j) in self.dbCursor.fetchall()]
    
    def getDLCs(self, date, marketDB, keepSingles=True):
        """Returns a list of lists of company IDs linked as DLCs
        for a given date
        """
        query= """SELECT company_id, axioma_id FROM company_mcap_linkage
        WHERE from_dt <= :date_arg AND thru_dt > :date_arg """
        marketDB.dbCursor.execute(query, date_arg=date)
        DLCCIds = list()
        DLCAxIds = list()

        # Sort into groups with common company ID or Axioma ID
        for (cid, axid) in marketDB.dbCursor.fetchall():
            found = False
            for i in range(len(DLCCIds)):
                if (cid in DLCCIds[i]) or (axid in DLCAxIds[i]):
                    DLCCIds[i].append(cid)
                    DLCAxIds[i].append(axid)
                    found = True
                    continue
            if not found:
                DLCCIds.append([cid])
                DLCAxIds.append([axid])

        # Strip out duplicates from list of lists of company IDs
        newList = list()
        for cidList in DLCCIds:
            cidList = list(set(cidList))
            if keepSingles or (len(cidList) > 1 ):
                newList.append(cidList)
        return newList

    def getDRToUnderlying(self, date, subIssues, marketDB):
        """Returns a mapping from DR to underlying asset
        for the given set of subissues and date
        """
        dr2UMap = dict()
        allSubIssues = self.getAllActiveSubIssues(date)
        allSubIDMap = dict((i.getModelID().getIDString(), i) for i in allSubIssues)

        query= """SELECT axioma_id, root_axioma_id FROM asset_dim_root_id_active_int
        WHERE from_dt <= :date_arg AND thru_dt > :date_arg """
        marketDB.dbCursor.execute(query, date_arg=date)

        # Get mappings from market ID to model ID
        ax2MdIDMap = self.getIssueMapPairs(date)
        ax2MdIDMap = dict((j.getIDString(),i.getIDString()) for (i,j) in ax2MdIDMap)
        subIDMap = dict((i.getModelID().getIDString(), i) for i in subIssues)
        
        for (dr_id, un_id) in marketDB.dbCursor.fetchall():
            if dr_id in ax2MdIDMap:
                dr_mdl_id = ax2MdIDMap[dr_id]
                # No good if we can't identify DR ID
                if dr_mdl_id in subIDMap:
                    drSid = subIDMap[dr_mdl_id]
                    if (un_id is None) or (un_id == dr_id):
                        dr2UMap[drSid] = None
                    elif un_id in ax2MdIDMap:
                        un_mdl_id = ax2MdIDMap[un_id]
                        if un_mdl_id in allSubIDMap:
                            dr2UMap[drSid] = allSubIDMap[un_mdl_id]
        logging.debug('%d DRs mapped to underlying assets', len(dr2UMap))

        return dr2UMap

    def getSPACAnnounceDate(self, date, subIssues, marketDB):
        """Returns a mapping from sub-issue to SPAC announcement for any SPACs in the list of sub-issues
        that have an announcement date in the DB
        """
        spacDateMap = dict()
        query= """SELECT axioma_id, announcement_dt FROM spac_announcement_info
        WHERE eff_from_dt <= :date_arg AND eff_thru_dt > :date_arg """
        marketDB.dbCursor.execute(query, date_arg=date)

        # Get mappings from market ID to model ID
        ax2MdIDMap = self.getIssueMapPairs(date)
        ax2MdIDMap = dict((j.getIDString(),i.getIDString()) for (i,j) in ax2MdIDMap)
        subIDMap = dict((i.getModelID().getIDString(), i) for i in subIssues)

        for (axid, dt) in marketDB.dbCursor.fetchall():
            mdl_id = ax2MdIDMap.get(axid, None)
            sid = subIDMap.get(mdl_id, None)
            if (sid is not None) and (sid in subIssues):
                if sid in spacDateMap:
                    if spacDateMap[sid] < dt.date():
                        spacDateMap[sid] = dt.date()
                else:
                    spacDateMap[sid] = dt.date()
        logging.debug('%d SPACs mapped to announcement dates', len(spacDateMap))
        return spacDateMap

    def loadRootIDLinkage(self, issues, marketDB):
       """Returns a mapping from issue to root_id issues for the full history
       """
       query="""select axioma_id, root_axioma_id, from_dt, thru_dt from asset_dim_root_id_active_int"""
       df=sql.read_frame(query,marketDB.dbConnection)
       query="""select marketdb_id , modeldb_id, from_dt as im_from_dt, thru_dt as im_thru_dt from issue_map"""
       map_df=sql.read_frame(query,self.dbConnection)

       ret=pandas.merge(pandas.merge(df, map_df, left_on='AXIOMA_ID', right_on='MARKETDB_ID' ), map_df, left_on='ROOT_AXIOMA_ID', right_on='MARKETDB_ID', suffixes=['_model', '_root'])
      
       rootIDmap={}
       for idx, row in ret.iterrows():
           issueid= row['MODELDB_ID_model']
           rootid= row['MODELDB_ID_root'] 
           fromdt=max([row['FROM_DT'], row['IM_FROM_DT_model'], row['IM_FROM_DT_root']]) 
           thrudt=min([row['THRU_DT'], row['IM_THRU_DT_model'], row['IM_THRU_DT_root']])

           if issueid in rootIDmap:
               rootIDmap[issueid].append([rootid, fromdt, thrudt])
           else:
               rootIDmap[issueid]= [[rootid,fromdt,thrudt]]

       # now populate a dictionary for return to the caller with the modelIDs populated with the right type of structure
       retValDict={}
       for i in issues:
          key = i.getIDString()
          retValDict[i] = []
          if key not in rootIDmap:
              continue
          for linkage in rootIDmap[key]:
              val = Utilities.Struct()
              val.id = linkage[0][1:]  # make sure to use the public value
              val.fromDt = linkage[1]
              val.thruDt = linkage[2]
              retValDict[i].append(val)

       return retValDict
      
    def loadIssueExposureLinkage(self, issues, linkage_type=1):
       """Returns a mapping from slave to master issues for the full history
       """
       query="""select slave_issue_id, master_issue_id, linkage_from_dt, linkage_thru_dt from issue_exposure_linkage where linkage_type=:linkage_arg""" 
       self.dbCursor.execute(query, linkage_arg=linkage_type)
       linkageMap={}
       for i in self.dbCursor.fetchall():
           slave_issue_id = i[0]
           if slave_issue_id in linkageMap:
               linkageMap[slave_issue_id].append([i[1][1:],i[2],i[3]])
           else:
               linkageMap[slave_issue_id]= [[i[1][1:],i[2],i[3]]]

       # now populate a dictionary for return to the caller with the modelIDs populated with the right type of structure
       retValDict={}
       for i in issues:
          key = i.getIDString()
          retValDict[i] = []
          if key not in linkageMap:
              continue
          for linkage in linkageMap[key]:
              val = Utilities.Struct()
              val.id = linkage[0]
              val.fromDt = linkage[1]
              val.thruDt = linkage[2]
              retValDict[i].append(val)

       return retValDict
      
    def getClonedMap(self, date, subIssues, cloningOn=True, linkageType=1):
        """Returns a mapping from slave to master subissue for the given date
        """
        cloneMap = dict()
        if not cloningOn:
            return cloneMap
        try:
            query = """SELECT slave_issue_id, master_issue_id FROM issue_exposure_linkage
            WHERE linkage_from_dt <= :date_arg AND linkage_thru_dt > :date_arg
            AND linkage_type = :linkage_arg"""
            self.dbCursor.execute(query, date_arg=date, linkage_arg=linkageType)
            subIDMap = dict((i.getModelID().getIDString(), i) for i in subIssues)
            for i in self.dbCursor.fetchall():
                if (i[0] in subIDMap) and (i[1] in subIDMap):
                    cloneMap[subIDMap[i[0]]] = subIDMap[i[1]]
            logging.debug('%d pairs of clones (linkage type %d)', len(cloneMap), linkageType)
        except:
            logging.warning('issue_exposure_linkage table does not exist...')
        return cloneMap

    def getIdentifierHistory(self, modelID):
        """Return the history of marketDB identifiers for the
        given ID.  Returns a list of (value, from_dt, thru_dt).
        """
        query = """SELECT marketdb_id, from_dt, thru_dt FROM issue_map
        WHERE modeldb_id = :id_arg
        ORDER BY from_dt"""
        self.dbCursor.execute(query,
                              id_arg=modelID)
        return [(id, fromDt.date(), thruDt.date()) for (id, fromDt, thruDt)
                in self.dbCursor.fetchall()]
    
    def getIssueTickers(self, date, ids, marketDB):
        """Returns an issue to ticker map for the given issues valid on
        the given date.
        """
        self.loadMarketIdentifierHistoryCache(
            ids, marketDB, 'asset_dim_ticker', 'id', cache=self.tickerCache)
        ticker = [(mid, self.tickerCache.getAssetValue(mid, date))
                  for mid in ids]
        ticker = dict([(i, j.id) for (i,j) in ticker if j is not None])
        return ticker
    
    def getFutureTickers(self, date, ids, marketDB):
        """Returns an issue to ticker map for the given issues valid on
        the given date.
        """
        futureTickerCache = FromThruCache()
        self.loadMarketIdentifierHistoryCache(
            ids, marketDB, 'future_dim_ticker', 'id', cache=futureTickerCache)
        ticker = [(mid, futureTickerCache.getAssetValue(mid, date))
                  for mid in ids]
        ticker = dict([(i, j.id) for (i,j) in ticker if j is not None])
        return ticker
    
    def getFutureSeries(self, date, assets, marketDB):
        """Returns a map of asset to (marketDB series axioma_id, last_trading_date)
        for each future in the list, valid on the given date
        """
        (mktids, modelMarketMap) = self.getMarketDB_IDs(marketDB, assets)
        return marketDB.getFutureFamilyID(MarketID.MarketID(string=m) for m in mktids)
    
    def getIssueCompanies(self, date, ids, marketDB, keepUnmapped=False):
        """Returns an issue to company ID map for the given issues
        valid on the given date.
        """
        self.loadMarketIdentifierHistoryCache(
            ids, marketDB, 'asset_dim_company', 'company_id',
            cache=self.companyCache)
        companies = [(mid, self.companyCache.getAssetValue(mid, date)) for mid in ids]
        if keepUnmapped:
            companies = dict((i, j.id) if j is not None else (i, None) for (i,j) in companies)
        else:
            companies = dict((i, j.id) for (i,j) in companies if j is not None)
        return companies
    
    def getIssueCompanyGroups(self, date, ids, marketDB, mapAllIssues=False):
        """Returns a mapping of Company IDs to lists of SubIssues belonging to
        that issuer.  Only Company IDs associated with two or more
        SubIssues are returned.
        """
        companyMap = self.getIssueCompanies(date, ids, marketDB)
        subIssueGroups = collections.defaultdict(list)
        for (sid, company) in companyMap.items():
            subIssueGroups[company].append(sid)
        if mapAllIssues:
            subIssueGroups = dict((company, vals) for company, vals in subIssueGroups.items())
        else:
            subIssueGroups = dict((company, vals) for company, vals in subIssueGroups.items() if len(vals) != 1)
        self.log.debug('%d groups of company ID linked assets', len(subIssueGroups))
        return subIssueGroups

    def getCompanySubIssues(self, date, companies, marketDB):
        """For the given company IDs, returns all sub-issues mapped
        to them on the given date.
        The return value is a dictionary mapping sub-issues to
        a (company-ID, exclude_from_mcap, country_iso) tuple.
        """
        axidCompanyMap = marketDB.getCompanyAssets(date, companies)
        mdlSubMap = dict([(sid.getModelID(), sid) for sid in self.getAllActiveSubIssues(date)])
        mktMdlMap = dict([(mktId, mid) for (mid, mktId) in self.getIssueMapPairs(date)])
        issueCompanyMap = dict()
        for (axid, val) in axidCompanyMap.items():
            sid = mdlSubMap.get(mktMdlMap.get(axid))
            if sid is not None:
                issueCompanyMap[sid] = val
        return issueCompanyMap

    def getSpecificIssueCompanyGroups(self, date, ids, marketDB, excludeList):
        """returns a mapping of Company IDs to lists of SubIssues belonging to 
        that issuer, except stocks trading in China. Make use of Axioma Sec Type.
        Exclude List should be a list of String specifying sectype you want to exclude.
        Only Company IDs associated with two or more SubIssues are returned.
        """
        clsFamily = marketDB.getClassificationFamily('ASSET TYPES')
        assert(clsFamily is not None)
        clsMembers = dict([(i.name, i) for i in marketDB.\
                               getClassificationFamilyMembers(clsFamily)])
        cm = clsMembers.get('Axioma Asset Type')
        assert(cm is not None)
        clsRevision = marketDB.\
                getClassificationMemberRevision(cm, date)
        homeClsData = self.getMktAssetClassifications(
                        clsRevision, ids, date, marketDB)
        secTypeDict = dict([(i, j.classification.code) for (i,j) in homeClsData.items()])

        companyMap = self.getIssueCompanies(date, ids, marketDB)
        subIssueGroups = dict()
        for (sid, company) in companyMap.items():
            secType = secTypeDict.get(sid)
            if secType is None:
                self.log.error('Missing Axioma Sec Type for %s'%sid.getSubIDString())
            elif secType in excludeList:
                continue

            if company not in subIssueGroups:
                subIssueGroups[company] = list()
            subIssueGroups[company].append(sid)
        for company in list(subIssueGroups.keys()):
            if len(subIssueGroups[company])==1:
                subIssueGroups.pop(company)
        self.log.debug('%d groups of company ID linked assets', len(subIssueGroups))
        return subIssueGroups

    def getLastCalendarEntry(self, rmg):
        """Returns the chronologically last calendar entry for the given
        risk model group.
        """
        self.dbCursor.execute("""SELECT * FROM (SELECT dt, sequence
        FROM rmg_calendar WHERE rmg_id=:rmg_arg ORDER BY sequence DESC)
        WHERE ROWNUM = 1""", rmg_arg=rmg.rmg_id)
        return [(self.oracleToDate(i[0]), i[1])
                for i in self.dbCursor.fetchall()]
    
    def getMarketDB_IDs(self, marketDB, mids):
        """Find the MarketDB IDs corresponding to the given Model IDs
        and return a tuple with the set of MarketDB IDs and
        a dictionary mapping Model ID to (from_dt, thru_dt, MarketDB ID)
        """
        self.log.debug('getMarketDB_IDs')
        if len(self.issueMapCache) == 0:
            for table in ['issue_map', 'future_issue_map']:
                query = """SELECT modeldb_id, marketdb_id, from_dt, thru_dt
                   FROM %s""" % table
                self.dbCursor.execute(query)
                tmp = dict()
                for (mdlid, mktid, fromDt, thruDt) in self.dbCursor.fetchall():
                    mdlid = ModelID.ModelID(string=mdlid)
                    tmp.setdefault(
                        mdlid, list()).append(
                        (fromDt.date(), thruDt.date(),
                         MarketID.MarketID(string=mktid)))
                for (key, value) in tmp.items():
                    self.issueMapCache[key] = (
                        value, [i[2].getIDString() for i in value])
                self.log.debug('loaded issue map info for %d Model IDs',
                               len(tmp))
        modelMarketMap = dict()
        axidStr = set()
        for mid in mids:
            (mappings, axids) = self.issueMapCache.get(mid, (None, None))
            if mappings is not None:
                modelMarketMap[mid] = mappings
                axidStr.update(axids)
        axidStr = sorted(axidStr)
        self.log.debug('done mapping model IDs: %d/%d', len(axidStr),
                       len(mids))
        return (axidStr, modelMarketMap)
    
    def getMetaEntity(self, marketDB, currentDate, rms_id, primaryIDList):
        """Returns all sub-issues active on the given date for the
        given risk model serie.
        The return value is a set of axioma sub-issue ID.
        Assets that don't have an identifier mapping are removed from
        the set.
        """
        self.log.debug('getMetaEntity: begin')
        subids = self.getRiskModelSerieSubIssues(currentDate, rms_id)
        ids = [i.getModelID() for i in subids]
        identifier = dict()
        for primaryID in primaryIDList:
            if primaryID == 'CUSIP':
                primaryIDs = self.getIssueCUSIPs(currentDate, ids, marketDB)
            elif primaryID == 'SEDOL':
                primaryIDs = self.getIssueSEDOLs(currentDate, ids, marketDB)
            elif primaryID == 'ISIN':
                primaryIDs = self.getIssueISINs(currentDate, ids, marketDB)
            elif primaryID == 'Ticker':
                primaryIDs = self.getIssueTickers(currentDate, ids, marketDB)
            else:
                raise ValueError('Unknown primary ID %s' % (primaryID))
            identifier.update(primaryIDs)
        
        complete = set([sid for sid in subids if sid.getModelID() in identifier])
        missing = set(subids) - complete
        missingID = [i.getModelID().getIDString() for i in missing]
        if len(missingID) > 0:
            self.log.warning('%d assets dropped due to missing %s',
                          len(missingID), primaryIDList)
            self.log.debug('dropped assets: %s', ','.join(missingID))
        self.log.debug('getMetaEntity: end')
        return complete
    
    def getPaidDividends(self, startDate, endDate, assetList, marketDB,
                         convertTo, includeSpecial=True, includeStock=True):
        """Return the total paid dividends in currency of the given subids over
        then period from startDate to endDate inclusively.
        The return value is a dictionary mapping assets to a list containing
        the dividends in the requested range chronological order.
        The dividends are reported in currency, either the currency of the
        dividend payment or the requested currency given by convertTo.
        Note that there could be multiple entries for the same day in case
        a company pays more than one dividend on that day.
        """
        self.log.debug('getPaidDividends: begin')
        # If currency conversions required
        if convertTo is not None:
            if not self.currencyCache:
                raise ValueError('Currency conversion requires a ForexCache object present')
        cache = self.cdivCache
        INCR=200
        keyList = ['key%d' % i for i in range(INCR)]
        defaultDict = dict([(a, None) for a in keyList])
        
        issueList = list()
        for a in assetList:
            if isinstance(a, SubIssue):
                issueList.append(a.getModelID())
            else:
                issueList.append(a)
        missingIds = cache.getMissingKeys(issueList)
        (axidStr, modelMarketMap) = self.getMarketDB_IDs(marketDB, missingIds)
        # Get paid dividend history for MarketDB IDs (cdiv plus current and prior tso)
        axidCDivDict = dict()
        query = """SELECT cdiv.axioma_id, ex_dt, ca_sequence, gross_value, currency_id,
           tso1.value, (SELECT tso2.value FROM  asset_dim_tso_active tso2
                        WHERE cdiv.axioma_id=tso2.axioma_id AND tso2.change_del_flag='N'
                        AND tso2.change_dt = (SELECT MAX(change_dt) FROM asset_dim_tso_active t2
                            WHERE t2.axioma_id=cdiv.axioma_id and t2.change_dt < cdiv.ex_dt))
        FROM asset_dim_cdiv_active cdiv
           JOIN asset_dim_tso_active tso1 ON cdiv.axioma_id=tso1.axioma_id
        WHERE cdiv.axioma_id IN (%(keys)s) AND tso1.change_del_flag='N'
        AND tso1.change_dt = (SELECT MAX(change_dt) FROM asset_dim_tso_active t1
            WHERE t1.axioma_id=cdiv.axioma_id and t1.change_dt <= cdiv.ex_dt)
        """ % {'keys': ','.join([':%s' % i for i in keyList])}

        if not includeSpecial:
            logging.info('Excluding special dividends')
            query += """ AND (pay_type is null or pay_type != 2)"""
        if not includeStock:
            logging.info('Excluding stock dividends')
            query += """ AND (pay_type is null or pay_type != 3)"""
        query += """ ORDER BY ex_dt ASC"""
        axidStr = sorted(axidStr)
        # Note for later: this bit really slow - try to speed up
        for idChunk in listChunkIterator(axidStr, INCR):
            valueDict = defaultDict.copy()
            valueDict.update(dict(zip(keyList, idChunk)))
            marketDB.dbCursor.execute(query, valueDict)
            r = marketDB.dbCursor.fetchmany()
            while len(r) > 0:
                for (aid, dt, caSeq, value, currencyID, currentTso, priorTso) in r:
                    axidCDivDict.setdefault(aid, list()).append(
                        (dt.date(), float(value), int(currencyID), currentTso, priorTso, caSeq))
                r = marketDB.dbCursor.fetchmany()
        # Get Axioma ID AF records
        axidAFDict = {}
        query = """SELECT axioma_id, dt, ca_sequence, shares_af FROM asset_dim_af_active
        WHERE axioma_id in (%(keys)s)
        """% { 'keys': ','.join(':%s' % i for i in keyList)}
        for idChunk in listChunkIterator(axidStr, INCR):
            valueDict = defaultDict.copy()
            valueDict.update(dict(zip(keyList, idChunk)))
            marketDB.dbCursor.execute(query, valueDict)
            r = marketDB.dbCursor.fetchmany()
            while len(r) > 0:
                for (aid, dt, caSeq, sharesAf) in r:
                    axidAFDict.setdefault((aid, dt.date()), list()).append((sharesAf, caSeq))
                r = marketDB.dbCursor.fetchmany()
        # Construct cdiv history for sub-issues
        for mid in missingIds:
            axidHistory = sorted(modelMarketMap.get(mid, list()))
            for (fromDt, thruDt, axid) in axidHistory:
                cdivHistory = axidCDivDict.get(axid.getIDString(), list())
                for (cdivDt, cdivVal, cdivCcy, currentTso, priorTso, caSeq) in cdivHistory:
                    if cdivDt < thruDt and cdivDt >= fromDt:
                        afs = axidAFDict.get((axid.getIDString(), cdivDt), list())
                        priorAFs = [af for af in afs if af[1] < caSeq]
                        laterAFs = [af for af in afs if af[1] > caSeq]
                        if len(laterAFs) == 0:
                            # No AF between dividend and current TSO, use that
                            tso = currentTso
                        elif len(priorAFs) == 0 and priorTso is not None:
                            # No AF between prior TSO and dividend, use that
                            tso = priorTso
                        else:
                            # Adjust current TSO by adjustments after dividend ca_sequence
                            tso = currentTso
                            for afValue, afSeq in laterAFs:
                                tso = tso / afValue
                        cache.addChange(mid, cdivDt, (cdivVal * tso, cdivCcy))
            cache.sortHistory(mid)
        retval = cache.getHistories(list(zip(assetList, issueList)), startDate,
                                    endDate + datetime.timedelta(days=1))
        # Convert to currency and apply currency conversion if necessary
        for asset in assetList:
            values = retval[asset]
            if convertTo is not None:
                if isinstance(convertTo, int):
                    targetCurrency = convertTo
                else:
                    targetCurrency = convertTo[asset]
            else:
                targetCurrency = None
            # Define helper method for currency conversion
            def getRate(dt, fromCcy, toCcy):
                if toCcy is None:
                    return 1.0
                rate = self.currencyCache.getRate(dt, fromCcy, toCcy)
                if rate is None:
                    self.log.error('No exchange rate (%d, %d) on'
                                   ' %s using zero', fromCcy, toCcy, dt)
                    rate = 0.0
                return rate
            newValues = [(dt, cdiv*getRate(dt, ccy, targetCurrency))
                         for (dt, (cdiv, ccy)) in values]
            retval[asset] = newValues
        return retval

    def getQuarterlyTotalDebt(self, startDate, endDate, subids, date,
                              marketDB, convertTo):
        """For each sub-issue, returns the total debt (in million currnecy)
        as effective on 'data' for filing dates between startDate and
        endDate (inclusive).
        Total debt is the sum of long-term debt and debt in current
        liabilities where both exist.
        The values are taken from the quarterly filings.
        Dates where one or both is missing are ignored.
        marketDB is a MarketDB object which will be used to resolve
        the item name.
        convertTo is a dictionary that for each SubIssue in subids must
        contain the currency ID to which the values should be converted.
        The values are left in their local currency if the currencyID is None.
        If convertTo is None then no currency conversion will take place. 
        The return value is a list with an entry for each asset in subids.
        Each entry is a list of (date, value, currency_id) tuples
        sorted in chronological order.
        """
        self.log.debug('getQuarterlyTotalDebt: begin')
        dicl = self.getFundamentalCurrencyItem(
            'dicl_qtr', startDate, endDate, subids, date, marketDB, convertTo)
        ltd = self.getFundamentalCurrencyItem(
            'ltd_qtr', startDate, endDate, subids, date, marketDB, convertTo)
        qdebt = list()

        # each loop is for one Sid
        for (diclList, ltdList) in zip(dicl, ltd):
            # this is a hack for CR-17271
            # instead of using iter, use dataframe and join to match dates
            dicl_df = pandas.DataFrame(diclList, columns=['date', 'dicl', 'ccy_id1'])
            ltd_df = pandas.DataFrame(ltdList, columns=['date', 'ltd', 'ccy_id2'])
            result_df = pandas.merge(left=dicl_df, right=ltd_df, left_on='date',right_on='date')
            size = len(result_df)

            if size > 1: # only one or none, proceed.

                # Roll if only all the following 4 conditions are met
                #      dicl  ltd   
                # T-1  <>0   <>0   
                # T     0    <>0     
                dicl_T_cond = (result_df.at[size-1, 'dicl'] == 0.0)
                dicl_T_1_cond = (result_df.at[size-2, 'dicl'] != 0.0)
                ltd_T_cond = (result_df.at[size-1, 'ltd'] != 0.0)
                ltd_T_1_cond = (result_df.at[size-2, 'ltd'] != 0.0)

                if (dicl_T_cond and dicl_T_1_cond and ltd_T_cond and ltd_T_1_cond):
                        result_df.at[size-1, 'dicl'] = result_df.at[size-2, 'dicl']
                        result_df.at[size-1, 'ltd'] = result_df.at[size-2, 'ltd']

            # calculate total debt 
            result_df['total_debt'] = result_df['dicl'] + result_df['ltd']

            # convert back to list of tuples to match output
            output_list = list(result_df[['date', 'total_debt', 'ccy_id1']].to_records(index=False))
            qdebt.append(output_list)

        assert(len(qdebt) == len(subids))
        self.log.debug('getQuarterlyTotalDebt: end')
        return qdebt
    
    def getAnnualTotalDebt(self, startDate, endDate, subids, date, marketDB, convertTo):
        """For each sub-issue, returns the total debt (in million currnecy)
        as effective on 'data' for filing dates between startDate and
        endDate (inclusive).
        Total debt is the sum of long-term debt and debt in current
        liabilities where both exist.
        The values are taken from the annual filings.
        Dates where one or both is missing are ignored.
        marketDB is a MarketDB object which will be used to resolve
        the item name.
        convertTo is a dictionary that for each SubIssue in subids must
        contain the currency ID to which the values should be converted.
        The values are left in their local currency if the currencyID is None.
        If convertTo is None then no currency conversion will take place.
        The return value is a list with an entry for each asset in subids.
        Each entry is a list of (date, value, currency_id) tuples
        sorted in chronological order.
        """
        self.log.debug('getAnnualTotalDebt: begin')
        dicl = self.getFundamentalCurrencyItem(
            'dicl_ann', startDate, endDate, subids, date, marketDB, convertTo)
        ltd = self.getFundamentalCurrencyItem(
            'ltd_ann', startDate, endDate, subids, date, marketDB, convertTo)
        adebt = list()
        for (diclList, ltdList) in zip(dicl, ltd):
            myList = list()
            adebt.append(myList)
            dIter = iter(diclList)
            lIter = iter(ltdList)
            try:
                dVal = next(dIter)
                lVal = next(lIter)
                while True:
                    if dVal[0] == lVal[0]:
                        # Match, report sum
                        myList.append((dVal[0], dVal[1] + lVal[1], dVal[2]))
                        dVal = next(dIter)
                        lVal = next(lIter)
                    elif dVal[0] < lVal[0]:
                        dVal = next(dIter)
                    else:
                        lVal = next(lIter)
            except StopIteration:
                pass
        assert(len(adebt) == len(subids))
        self.log.debug('getAnnualTotalDebt: end')
        return adebt

    def getModelSeries(self, rm_id):
        """Returns information about all the risk model series for the
           given rm_id
           Returns a list of structs containing rms_id, revision, from_dt,
           thru_dt, and distribute
           The risk model series are sorted in ascending order by thru_dt
        """
        self.dbCursor.execute('select serial_id, revision, from_dt, thru_dt, distribute from risk_model_serie where rm_id=:rmid_arg order by thru_dt', 
                              rmid_arg=rm_id)
        values = []
        for row in self.dbCursor:
            result = Utilities.Struct()
            result.rms_id = row[0]
            result.revision = row[1]
            result.from_dt = self.oracleToDate(row[2])
            result.thru_dt = self.oracleToDate(row[3])
            result.distribute = row[4]
            values.append(result)
        return values

    def getRiskModelInfo(self, rm_id, revision):
        """Returns the risk model information for the given risk model
        id and revision.
        The return value is a struct containing serial_id, name, description,
        mnemonic, group_id, and region_id.
        """
        self.dbCursor.execute("""
            SELECT rms.serial_id, rm.name, rm.description,
                rm.mnemonic, rmm.rmg_id, rmg.region_id, rm.numeraire,
                rmm.from_dt, rmm.thru_dt, rmm.fade_dt,
                rms.distribute, rm.model_region, rmm.full_dt
            FROM risk_model rm, risk_model_serie rms,
                risk_model_group rmg, rmg_model_map rmm
            WHERE rms.rm_id = :model_arg AND rms.revision = :rev_arg
            AND rm.model_id = :model_arg AND rms.serial_id = rmm.rms_id
            AND rmm.rmg_id = rmg.rmg_id """,
                model_arg=rm_id, rev_arg=revision)
            
        r = self.dbCursor.fetchall()
        if len(r) == 0:
            raise LookupError('Unkown risk model revision %d/%d'
                                % (rm_id, revision))
        rmInfo = Utilities.Struct()
        rmInfo.serial_id = r[0][0]
        rmInfo.rm_id = rm_id
        rmInfo.revision = revision
        rmInfo.name = r[0][1]
        rmInfo.description = r[0][2]
        rmInfo.mnemonic = r[0][3]
        rmInfo.numeraire = r[0][6]
        rmInfo.distribute = r[0][10] != 0
        rmInfo.rmgTimeLine = []
        rmInfo.region = r[0][11]
        for row in r:
            rmg = Utilities.Struct()
            rmg.rmg = self.getRiskModelGroup(row[4])
            rmg.rmg_id = rmg.rmg.rmg_id
            rmg.from_dt = row[7].date()
            rmg.thru_dt = row[8].date()
            rmg.fade_dt = row[9].date()
            if row[12] is None:
                rmg.full_dt = rmg.from_dt
            else:
                rmg.full_dt = row[12].date()
            rmInfo.rmgTimeLine.append(rmg)
        
        return rmInfo
    
    def getRiskModelsForExtraction(self, date):
        """Returns information about risk models in the database which
        should be included in an extraction.
        Uses FROM_DT, THRU_DT and DISTRIBUTE columns in the risk_model_serie
        table to determine if a model should be extracted for the passed date.
        """
        self.dbCursor.execute("""SELECT rm_id, revision
        FROM risk_model_serie rms
        WHERE from_dt <= :date_arg AND thru_dt > :date_arg
        AND distribute = 1""",
                              date_arg=date)
        r = self.dbCursor.fetchall()
        models = [self.getRiskModelInfo(i[0], i[1]) for i in r]
        return models
    
    def getRiskModelGroup(self, rmg_id):
        """Load the entry for the given risk model group ID from the
        risk_model_group table.
        Returns a struct with rmg_id, description, region, currency
        and mnemonic of the risk model group.
        Returns None, if no risk model group with the given ID exists.
        """
        self.dbCursor.execute(
            """SELECT rmg.description, rmg.region_id, rmg.mnemonic,
                    rmg.gmt_offset, rmg.iso_code
               FROM risk_model_group rmg
               WHERE rmg.rmg_id = :rmg_arg""",
               rmg_arg = rmg_id)
        r = self.dbCursor.fetchall()
        if len(r) != 1:
            return None
        (description, region, mnemonic, gmt_offset, iso_code) = r[0]

        query = """SELECT fx.currency_code, dev.developed, dev.emerging, dev.frontier,
          GREATEST(dev.from_dt, fx.from_dt), LEAST(dev.thru_dt, fx.thru_dt)
        FROM rmg_currency fx, rmg_dev_status dev
        WHERE fx.rmg_id = :rmg_arg AND dev.rmg_id = fx.rmg_id
          AND dev.thru_dt > fx.from_dt AND fx.thru_dt > dev.from_dt"""
        self.dbCursor.execute(query, rmg_arg = rmg_id)
        r = self.dbCursor.fetchall()
        assert(len(r) > 0)
        
        # Create RiskModelGroup object
        dataDicts = [dict([('currency_code', row[0]),
                           ('developed', (row[1]==1)),
                           ('emerging', (row[2]==1)),
                           ('frontier', (row[3]==1)),
                           ('from_dt', row[4].date()),
                           ('thru_dt', row[5].date())]) for row in r]
        rmg = RiskModelGroup(rmg_id, description, mnemonic,
                             region, gmt_offset, iso_code, dataDicts)
        return rmg

    def getRiskModelGroupByISO(self, mnemonic):
        """Returns the RiskModelGroup object for the given ISO/mnemonic.
        Returns None if an invalid mnemonic is given.
        """
        self.dbCursor.execute(
            """SELECT rmg.rmg_id
               FROM risk_model_group rmg
               WHERE rmg.mnemonic = :iso_arg""",
               iso_arg = mnemonic)
        r = self.dbCursor.fetchall()
        if len(r) == 1:
            return self.getRiskModelGroup(r[0][0])
        else:
            raise LookupError('RiskModelGroup for %s not found' % mnemonic)
        return None
    
    def getAllDescriptors(self, local=False):
        """Returns a list all descriptors in the database.
        """
        query = """SELECT name, descriptor_id
                   FROM descriptor
                   WHERE (legacy is null or legacy != 1)"""
        if local:
            query += """ AND local=1"""
        logging.debug('getAllDescriptors() query: %s', query)          
        self.dbCursor.execute(query)
        r = self.dbCursor.fetchall()
        result = list()
        for name, descId in r:
            # Remove all the special characters here
            name = name.replace('(','')
            name = name.replace(')','')
            name = name.replace('.','')
            name = name.replace('-','')
            name = name.replace(' ','')
            result.append((name, descId))
        return result

    def getAllRiskModelGroups(self, inModels=True):
        """Returns a list all risk model groups in the database.
        If optional argument inModels=True, only returns risk
        model groups that are part of one or more models.
        """
        if inModels in self.allRMGCache:
            return self.allRMGCache[inModels]
        else:
            query = "SELECT g.rmg_id FROM risk_model_group g"
            if inModels:
                query += """ WHERE EXISTS (
                SELECT * FROM rmg_model_map m WHERE
                m.rmg_id = g.rmg_id) AND g.rmg_id > 0"""
            self.dbCursor.execute(query)
            r = self.dbCursor.fetchall()
            result = [self.getRiskModelGroup(row[0]) for row in r]
            self.allRMGCache[inModels] = result
        return result

    def getRMGIdsForRegion(self, reg_id, date):
        """Returns a set of RMG Ids for a particular region ID
        """
        self.dbCursor.execute("""SELECT rmg_id from rmg_region_map
                WHERE id = :reg_id AND from_dt <= :dt AND thru_dt > :dt""",
                reg_id=reg_id, dt=date)
        ret = self.dbCursor.fetchall()
        return [r[0] for r in ret]

    def getRiskModelInstanceESTU(self, rmi, estu_idx=1, estu_name='main'):
        """Returns the subissues of the estimation universe
        for the given risk model instance.
        The return value is a list of SubIssues.
        """
        self.dbCursor.execute("""SELECT sub_issue_id FROM rmi_estu_v3
                WHERE rms_id = :rms_arg AND dt = :date_arg AND nested_id = :id_arg""",
                rms_arg=rmi.rms_id, date_arg=rmi.date, id_arg=estu_idx)
        r = [SubIssue(i[0]) for i in self.dbCursor.fetchall()]
        if (len(r) < 1) and (estu_idx==1):
            self.log.info('No estu in rmi_estu_v3, trying legacy table')
            self.dbCursor.execute("""SELECT sub_issue_id FROM rmi_estu
            WHERE rms_id = :rms_arg AND dt = :date_arg""",
                                rms_arg=rmi.rms_id, date_arg=rmi.date)
            r = [SubIssue(i[0]) for i in self.dbCursor.fetchall()]
        self.log.info('Loaded %d assets from %s estimation universe', len(r), estu_name)
        return r

    def getAllRiskModelInstanceESTUs(self, rms_id, startDate, endDate):
        """Returns a dict mapping dates to lists of SubIssues
        """
        logging.info('getAllRiskModelInstanceESTUs: begin')
        self.dbCursor.execute("""SELECT dt, sub_issue_id FROM rmi_estu
                WHERE rms_id = :rms_arg AND dt >= :start_date_arg AND dt <= :end_date_arg""",
                rms_arg=rms_id, start_date_arg=startDate, end_date_arg = endDate)
        result = collections.defaultdict(list)
        for dt, subid in self.dbCursor:
            result[dt].append(SubIssue(subid))
        result = dict((dt.date(), subids) for dt, subids in result.items())
        logging.info('getAllRiskModelInstanceESTUs: end')
        return result

    def getEstuMappingTable(self, rms_id):
        """ Returns a mapping of estus by name to an estimation universe
        object for the given risk model instance
        """
        # Load master mapping table of estus
        self.dbCursor.execute("""SELECT id, name FROM rmi_estu_nested_info
                WHERE rms_id = :rms_arg""",
                rms_arg=rms_id)
        estuMap = dict()
        for (i,j) in self.dbCursor.fetchall():
            estuMap[j] = Utilities.Struct()
            estuMap[j].id = i
            estuMap[j].name = j
        if len(estuMap) > 0:
            return estuMap
        else:
            return None

    def getRMIESTUWeights(self, rmi, estu_idx=1):
        """Returns the weights of the passed subissues in the
        estimation universe for the given risk model instance.  The
        return value is a dictionary mapping the passed subIssues
        to a float containing the weight for the issue in the
        ESTU for the risk model instance.
        """
        query = """SELECT sub_issue_id, weight FROM rmi_estu_v3
        WHERE rms_id = :rms_arg AND dt = :date_arg
        AND nested_id = :id_arg AND weight IS NOT NULL"""
        self.dbCursor.execute(query, rms_arg=rmi.rms_id, date_arg=rmi.date, id_arg=estu_idx)
        r = self.dbCursor.fetchall()
        if (len(r) < 1) and (estu_idx==1):
            self.log.info('No estu weights in rmi_estu_v3, trying legacy table')
            query = """SELECT sub_issue_id, weight FROM rmi_estu
            WHERE rms_id = :rms_arg AND dt = :date_arg
            AND weight IS NOT NULL"""
            self.dbCursor.execute(query, rms_arg=rmi.rms_id, date_arg=rmi.date)
            r = self.dbCursor.fetchall()
        returnDict = dict()
        for (i, j) in r:
            returnDict[SubIssue(i)] = j
        self.log.info('Loaded %d estimation universe weights', len(returnDict))
        return returnDict
    
    def getRiskModelInstance(self, rms_id, date):
        """Returns a RiskModelInstance object corresponding to the given series
        on the given date.
        """
        self.dbCursor.execute("""SELECT has_exposures, has_returns, has_risks, is_final
        FROM risk_model_instance WHERE rms_id = :rms_arg AND dt = :date_arg""",
                              rms_arg=rms_id, date_arg=date)
        r = self.dbCursor.fetchall()
        if len(r) == 1:
            return RiskModelInstance(
                rms_id, date, has_exposures=(r[0][0] != 0),
                has_returns=(r[0][1] != 0), has_risks=(r[0][2] != 0),
                is_final=(r[0][3] != 0)
            )
        else:
            return None
    
    def getRiskModelInstances(self, rms_id, dateList=None):
        """Returns a list of RiskModelInstance objects corresponding to the
        given series on the given dates.
        Dates for which no instance exists are omitted from the list.
        """
        if dateList is None:
            mindt = datetime.date(1900,1,1)
            maxdt = datetime.date(2999, 12, 31)
        elif len(dateList) == 0:
            return list()
        else:
            mindt = min(dateList)
            maxdt=max(dateList)
            
        self.dbCursor.execute("""SELECT rmi.dt, rmi.has_exposures,
        rmi.has_returns, rmi.has_risks, rmi.is_final
        FROM risk_model_instance rmi
        WHERE rmi.rms_id = :rms_arg AND rmi.dt between :mindt and :maxdt""",
                              rms_arg=rms_id, mindt=mindt,
                              maxdt=maxdt)
        result = dict((i[0].date(), i) for i in self.dbCursor.fetchall())
        if dateList is None:
            result = list(result.values())
        else:
            result = [result[d] for d in dateList if d in result]
        result = [RiskModelInstance(rms_id, r[0].date(),
                                    has_exposures=(r[1] != 0),
                                    has_returns=(r[2] != 0),
                                    has_risks=(r[3] != 0),
                                    is_final=(r[4] != 0))
                  for r in result]
        return result
    
    def getRiskModelInstanceSubFactors(self, rmi, factors):
        """Returns the sub-factors of the given factors
        for the given risk model instance.
        The return value is a list of sub-factors matching the factors.
        """
        return self.getSubFactorsForDate(rmi.date, factors)
    
    def getRiskModelInstanceUniverse(self, rmi, restrictDates=True, returnExtra=False):
        """Returns the subissues of the exposure universe
        for the given risk model instance.
        The return value is a list of SubIssues.
        If restrictDates is False, don't consider the dates in the sub_issue table.
        If returnExtra is True, nursery assets (those with qualify=0) are also returned.
        Otherwise, only officially 'live' assets will be returned
        """
        if restrictDates:
            query = """SELECT sub_issue_id, qualify FROM rmi_universe rmiu, sub_issue si
            WHERE rms_id = :rms_arg AND rmiu.dt = :date_arg
            AND si.sub_id=rmiu.sub_issue_id AND si.from_dt<=rmiu.dt AND si.thru_dt>rmiu.dt"""
        else:
            query = """SELECT sub_issue_id, qualify FROM rmi_universe rmiu
            WHERE rms_id = :rms_arg AND rmiu.dt = :date_arg"""
        self.dbCursor.execute(query, rms_arg=rmi.rms_id, date_arg=rmi.date)
        r = []
        for i in self.dbCursor.fetchall():
            if i[1] is None:
                r.append((SubIssue(i[0]), 1))
            else:
                r.append((SubIssue(i[0]), int(i[1])))
        if returnExtra:
            return [i for (i,j) in r]
        else:
            return [i for (i,j) in r if j != 0]
    
    def getRiskModelRegion(self, region_id):
        """Returns a RiskModelRegion corresponding to the region ID.
        """
        try:
            self.dbCursor.execute("""
                SELECT id, name, description, currency_code FROM region WHERE id = :id_arg""",
                id_arg = region_id)
        except:
            logging.warning('No CURRENCY_CODE field in region table')
            self.dbCursor.execute("""
                    SELECT id, name, description FROM region WHERE id = :id_arg""",
                    id_arg = region_id)
        r = self.dbCursor.fetchall()
        if len(r) != 1:
            return None
        elif len(r[0]) == 3:
            return RiskModelRegion(r[0][0], r[0][1], r[0][2])
        return RiskModelRegion(r[0][0], r[0][1], r[0][2], r[0][3])

    def getAllRiskModelRegions(self):
        """Return a list of all the regions.
        """
        try:
            self.dbCursor.execute("""SELECT id, name, description, currency_code FROM region""")
            r = self.dbCursor.fetchall()
            return [RiskModelRegion(i[0], i[1], i[2], i[3]) for i in r]
        except:
            logging.warning('No CURRENCY_CODE field in region table')
            self.dbCursor.execute("""SELECT id, name, description FROM region""")
            r = self.dbCursor.fetchall()
            return [RiskModelRegion(i[0], i[1], i[2]) for i in r]

    def getRiskModelRegionByName(self, region_name):
        """Returns a RiskModelRegion corresponding to the region name.
        """
        try:
            self.dbCursor.execute("""
                SELECT id, name, description, currency_code FROM region WHERE name = :name_arg""",
                name_arg = region_name)
        except:
            logging.warning('No CURRENCY_CODE field in region table')
            self.dbCursor.execute("""
                    SELECT id, name, description FROM region WHERE name = :name_arg""",
                    name_arg = name_id)
        r = self.dbCursor.fetchall()
        if len(r) != 1:
            return None
        elif len(r[0]) == 3:
            return RiskModelRegion(r[0][0], r[0][1], r[0][2])
        return RiskModelRegion(r[0][0], r[0][1], r[0][2], r[0][3])

    def getFactorTypeMap(self, factorList):
        """Returns a map of factors to their type names
        """
        self.dbCursor.execute("""SELECT factor_type_id, name FROM factor_type""")
        ret = self.dbCursor.fetchall()
        typeIDNameMap = dict([(r[0],r[1]) for r in ret])
        fTypeMap = dict()
        for fac in factorList:
            if fac.type_id in typeIDNameMap:
                fTypeMap[fac] = typeIDNameMap[fac.type_id]
            else:
                fTypeMap[fac] = None
        return fTypeMap

    def getRiskModelSerieFactors(self, rms_id):
        """Returns a list of ModelFactor objects belonging to the 
        given risk model serie.  All factors, including inactive
        ones, are returned.  Use the from_dt and thru_dt fields
        to determine the correct factor set for any given date.
        """
        self.dbCursor.execute("""SELECT rmsf.factor_id, f.name, 
            f.description, f.factor_type_id, rmsf.from_dt, rmsf.thru_dt, 
            d.descriptor_id, d.name, d.description, rmsd.scale
        FROM factor f, rms_factor rmsf
        LEFT OUTER JOIN rms_factor_descriptor rmsd
            ON rmsd.rms_id = rmsf.rms_id AND rmsd.factor_id = rmsf.factor_id
        LEFT OUTER JOIN descriptor d
            ON d.descriptor_id=rmsd.descriptor_id
        WHERE rmsf.rms_id = :rms_arg AND rmsf.factor_id = f.factor_id""",
                                rms_arg=rms_id)
        factors = []
        factorIdxMap = dict()
        for f in self.dbCursor.fetchall():
            if f[6] is None:
                factor = Factors.ModelFactor(f[1], f[2])
            else:
                if f[0] in factorIdxMap:
                    factor = factors[factorIdxMap[f[0]]]
                else:
                    factor = Factors.CompositeFactor(f[1], f[2])
                descr = Factors.FactorDescriptor(f[7], f[8])
                descr.descriptor_id = f[6]
                descr.scale = f[9]
                factor.descriptors.append(descr)
            factor.factorID = f[0]
            factor.from_dt = f[4].date()
            factor.thru_dt = f[5].date()
            factor.type_id = f[3]
            if f[0] not in factorIdxMap:
                factorIdxMap[factor.factorID] = len(factors)
                factors.append(factor)
        return factors
    
    def getRiskModelSerieIssues(self, currentDate, rms_id):
        """Returns a list of ModelIDs that are active in the
        given risk model serie over the entire history
        The currentDate argument appears to be very much superfluous
        """
        self.dbCursor.execute("""SELECT issue_id FROM rms_issue
        WHERE rms_id = :rms_id_arg""", rms_id_arg=rms_id)
        r = [ModelID.ModelID(string=i[0]) for i in self.dbCursor.fetchall()]
        return r

    def getRiskModelSerieSubIssues(self, currentDate, rms_id):
        """Returns a list of SubIssues that are active in the
        given risk model serie on the given date.
        """
        query = """SELECT si.sub_id 
        FROM rms_issue rmi
        JOIN sub_issue si ON si.issue_id=rmi.issue_id
        WHERE rmi.rms_id=:rms_id AND rmi.from_dt <= :date_arg
        AND rmi.thru_dt > :date_arg AND si.from_dt <= :date_arg
        AND si.thru_dt > :date_arg"""
        self.dbCursor.execute(query,
                              date_arg=currentDate,
                              rms_id=rms_id)
        r = [SubIssue(i[0]) for i in self.dbCursor.fetchall()]
        return r
    
    def getRMGMarketUniverse(self, date):
        """Returns a SubIssues-->RMG mapping if those SubIssues exist in the
        RMG Market Portfolio on the given date
        """
        query = """SELECT rmg_id, sub_issue_id
        FROM rmg_market_portfolio r WHERE dt = :dt_arg
        """
        rmgMarketDict = dict()
        rmgDict = dict([(rmg.rmg_id, rmg) for rmg in self.getAllRiskModelGroups()])
        self.dbCursor.execute(query, dt_arg = date)
        result = self.dbCursor.fetchall()
        for rmg_id, sid in result:
            if rmg_id in rmgDict:
                rmgMarketDict[SubIssue(sid)] = rmgDict[rmg_id]
        return rmgMarketDict

    def convertLMP2AMP(self, rmgList, date):
        """Converts list of RMGs (or menmonics or rmg_ids) to their corresponding
        AMP types
        """
        isList = True
        if type(rmgList) is not list:
            rmgList = [rmgList]
            isList = False
        ampList = []
        for rmg in rmgList:
            # First get the shortname (mnemonic) from the RMG item
            if type(rmg) is int:
                rmgObj = self.getRiskModelGroup(rmg)
                mnem = rmgObj.mnemonic
            elif type(rmg) is str:
                mnem = rmg
            else:
                mnem = rmg.mnemonic
            # Hard-coded exceptions here
            if mnem == 'XU':
                # US smallcap benchmark
                ampName = 'US-S'
            else:
                # Standard market benchmarks
                ampName = '%s-LMS' % mnem
            ampObj = self.getModelPortfolioByShortName(ampName, date)
            ampList.append(ampObj)
        if not isList:
            return ampList[0]
        return ampList

    def getRMGMarketPortfolio(self, rmg, date, amp=None, lookBack=30, fallBack=True, returnDF=False):
        """Returns a list of sub-issue/weight pairs corresponding
        to the market portfolio of the risk model group on the given day.
        """
        if amp is None:
            self.dbCursor.execute("""SELECT sub_issue_id, value
            FROM rmg_market_portfolio mp
            JOIN sub_issue si ON mp.sub_issue_id=si.sub_id
            WHERE si.from_dt <= :date_arg AND si.thru_dt > :date_arg
            AND mp.dt = :date_arg AND mp.rmg_id=:rmg_arg""",
                                rmg_arg=rmg.rmg_id, date_arg=date)
            legacyMP = [(SubIssue(i), j) for (i,j) in self.dbCursor.fetchall()]
            if returnDF:
                ids, wts = zip(*legacyMP)
                return pandas.Series(dict(zip(ids, wts)))
            return legacyMP

        # Now load in the whizzy new portfolio
        if type(amp) is str:
            amp = self.getModelPortfolioByShortName(amp, date)
        for back in range(lookBack):
            dt = date - datetime.timedelta(back)
            ampBack = self.getModelPortfolioByShortName(amp.short_name, dt)
            if ampBack is not None:
                ampConstituents = self.getModelPortfolioConstituents(dt, ampBack.id, quiet=True)
                if len(ampConstituents) > 0:
                    break
            else:
                logging.warning('%s AMP does not exist on %s', amp.short_name, dt)

        if len(ampConstituents) < 1:
            if fallBack:
                logging.warning('%s AMP is empty, loading legacy market portfolio', amp.short_name)
                ampConstituents = self.getRMGMarketPortfolio(rmg, date, amp=None)
            else:
                raise ValueError('%s AMP is empty', amp.short_name)

        if returnDF:
            ids, wts = zip(*ampConstituents)
            return pandas.Series(dict(zip(ids, wts)))
        return ampConstituents

    def getRMIPredictedBeta(self, rmi):
        """Returns a dictionary mapping SubIssues to their predicted beta
        in the given risk model instance.
        """
        self.dbCursor.execute("""SELECT sub_issue_id, value
        FROM rmi_predicted_beta WHERE rms_id = :rms_arg AND dt = :date_arg""",
                              rms_arg=rmi.rms_id, date_arg=rmi.date)
        return dict([(SubIssue(i), j) for (i,j) in self.dbCursor.fetchall()])

    def getRMIPredictedBetaV3(self, rmi, field='local_beta'):
        """Returns a dictionary mapping SubIssues to their predicted betas
        in the given risk model instance.
        """
        if rmi is None:
            return dict()
        self.dbCursor.execute("""SELECT sub_issue_id, %(field)s
            FROM rmi_predicted_beta_v3 WHERE rms_id = :rms_arg AND dt = :date_arg""" % {'field': field},
                              rms_arg=rmi.rms_id, date_arg=rmi.date)
        return dict([(SubIssue(i), j) for (i,j) in self.dbCursor.fetchall()])
    
    def getRMITotalRisk(self, rmi):
        """Returns a dictionary mapping SubIssues to their total risk
        in the given risk model instance.
        """
        self.dbCursor.execute("""SELECT sub_issue_id, value
        FROM rmi_total_risk WHERE rms_id = :rms_arg AND dt = :date_arg""",
                              rms_arg=rmi.rms_id, date_arg=rmi.date)
        return dict([(SubIssue(i), j) for (i,j) in self.dbCursor.fetchall()])
    
    def getRMSESTUExcludedIssues(self, rms_id, date):
        """Return the list of ModelIDs that should be excluded from
        the estimation universe of the model given by rms_id on the
        given date.
        The return value is a list of ModelID objects.
        """
        self.dbCursor.execute("""SELECT issue_id FROM rms_estu_excl_active_int
        WHERE rms_id = :rms_arg AND from_dt <= :date_arg
        AND thru_dt > :date_arg""",
                              rms_arg=rms_id, date_arg=date)
        return [ModelID.ModelID(string=i[0]) for i in self.dbCursor.fetchall()]

    def getAllESTUExcludedIssues(self, date):
        """Return the list of ModelIDs that are excluded from
        any model on the given date.
        The return value is a list of ModelID objects.
        """
        self.dbCursor.execute("""SELECT DISTINCT issue_id 
        FROM rms_estu_excl_active_int
        WHERE from_dt <= :date_arg AND thru_dt > :date_arg""", date_arg=date)
        return [ModelID.ModelID(string=i[0]) for i in self.dbCursor.fetchall()]

    def getDRSubIssues(self, rmi):
        """Return all DR subissues in the passed risk model instance on the date passed.
        DR assets are assets whose trading country is not mapped to the model.
        """
        query="""SELECT sub_id FROM rmi_universe ri JOIN sub_issue si ON si.sub_id=ri.sub_issue_id
        WHERE ri.rms_id=:rms_id AND ri.dt=:date_arg
        AND si.from_dt <= :date_arg AND si.thru_dt > :date_arg
        AND si.rmg_id NOT IN (SELECT rmm.rmg_id FROM rmg_model_map rmm WHERE rmm.rms_id=:rms_id
        AND rmm.from_dt <= :date_arg and rmm.thru_dt > :date_arg)
        """
        self.dbCursor.execute(query, rms_id=rmi.rms_id, date_arg=rmi.date)
        drs = [SubIssue(string=i[0]) for i in self.dbCursor.fetchall()]
        return drs
    
    def getProductExcludedSubIssues(self, date):
        """Return the list of subissues that should be excluded from
        extracted files as of the given date.
        The return value is a list of SubIssue objects.
        """
        self.dbCursor.execute("""SELECT si.sub_id FROM product_exclude pe, sub_issue si WHERE
        pe.from_dt <= :dt AND pe.thru_dt > :dt AND pe.issue_id=si.issue_id AND
        si.from_dt <= :dt AND si.thru_dt > :dt""", dt=date)
        excludes = [SubIssue(string=i[0]) for i in self.dbCursor.fetchall()]
        return excludes
    
    def getRMSFactorStatistics(self, rms_id, date, subFactors, flag=None):
        """Loads the factor statistics from the rms_factor_statistics
        table for risk model series rms_id on the given date.
        statistics that are not present or NULL are masked.
        """
        regressStats = Matrices.allMasked((len(subFactors), 4))
        if len(subFactors) == 0:
            return regressStats
        if flag is None:
            tableName = 'rms_factor_statistics'
        else:
            tableName = 'rms_factor_statistics_%s' % flag
        query = """SELECT sub_factor_id, std_error, t_value, t_probability, regr_constraint_weight
        FROM %(table_arg)s WHERE rms_id = :rms_arg and dt =:date_arg""" % {'table_arg': tableName}
        self.dbCursor.execute(query, rms_arg=rms_id, date_arg=date)
        factorIdxMap = dict(zip([f.subFactorID for f in subFactors],
                                list(range(len(subFactors)))))
        for (f, stdError, tVal, tProb, regrConstrWeight) in self.dbCursor.fetchall():
            if f not in factorIdxMap:
                continue
            fIdx = factorIdxMap[f]
            if stdError != None:
                regressStats[fIdx,0] = stdError
            if tVal != None:
                regressStats[fIdx,1] = tVal
            if tProb != None:
                regressStats[fIdx,2] = tProb
            if regrConstrWeight != None:
                regressStats[fIdx,3] = regrConstrWeight
        return regressStats
    
    def getRMSStatistics(self, rms_id, date, flag=None, fieldName='adjRsquared'):
        """Loads the summary statistics from the rms_statistics
        table for risk model series rms_id on the given date.
        Returns the adjusted R-squared value or None, if it isn't present
        in the datebase.
        """
        if flag is None:
            tableName = 'rms_statistics'
        else:
            tableName = 'rms_statistics_%s' % flag

        self.dbCursor.execute(
        """SELECT %(field_name)s FROM %(table_arg)s
        WHERE rms_id = :rms_arg AND dt = :date_arg""" % \
                {'field_name': fieldName, 'table_arg': tableName},
                rms_arg=rms_id, date_arg=date)
        r = self.dbCursor.fetchall()

        assert(len(r) <= 1)
        if len(r) == 0:
            return None
        return r[0][0]

    def getRMSStatisticsHistory(self, rms_id, flag=None, fieldName='adjRsquared'):
        """Loads the summary statistics from the rms_statistics
        table for risk model series rms_id.
        Returns a dictionary mapping datetime.date to the adjusted R-squared value for
        that date
        """
        if flag is None:
            tableName = 'rms_statistics'
        else:
            tableName = 'rms_statistics_%s' % flag

        self.dbCursor.execute(
        """SELECT dt, %(field_name)s FROM %(table_arg)s
        WHERE rms_id = :rms_arg""" % \
                {'field_name': fieldName, 'table_arg': tableName}, rms_arg=rms_id)
        return dict((self.oracleToDate(x[0]), x[1]) for x in self.dbCursor)
    
    def getShareAdjustmentFactors(self, startDate, endDate, assetList,
                                  marketDB):
        """Returns a dictionary mapping sub-issue/issue to their adjustment
        factors in the given date range [startDate; endDate].
        The adjustment factors are provided as a list of (date, value)
        pairs. The list is empty, if a sub-issue/issue has no adjustment
        in the given period. 
        """
        self.log.debug('getShareAdjustmentFactors: begin')
        cache = self.shareAFCache
        INCR=200
        keyList = ['key%d' % i for i in range(INCR)]
        defaultDict = dict([(a, None) for a in keyList])
        
        issueList = list()
        for a in assetList:
            if isinstance(a, SubIssue):
                issueList.append(a.getModelID())
            else:
                issueList.append(a)
        missingIds = cache.getMissingKeys(issueList)
        issueStr = [i.getIDString() for i in missingIds]
        # Get adjustment factors from ca_merger_survivor
        query = """SELECT modeldb_id, dt, share_ratio FROM ca_merger_survivor
          WHERE modeldb_id in (%(keys)s) ORDER BY dt ASC""" % {
            'keys': ','.join([':%s' % i for i in keyList])}
        for idChunk in listChunkIterator(issueStr, INCR):
            valueDict = defaultDict.copy()
            valueDict.update(dict(zip(keyList, idChunk)))
            self.dbCursor.execute(query, valueDict)
            for (mid, dt, share_af) in self.dbCursor.fetchall():
                cache.addChange(ModelID.ModelID(string=mid), dt.date(),
                                float(share_af))
        # Find MarketDB IDs that correspond to unprocessed sub-issues
        (axidStr, modelMarketMap) = self.getMarketDB_IDs(marketDB, missingIds)
        # Get adjustment factor history for MarketDB IDs
        axidAfDict = dict()
        query = """SELECT axioma_id, dt, shares_af
        FROM asset_dim_af_active
        WHERE axioma_id IN (%(keys)s)
        ORDER BY dt ASC""" % {
            'keys': ','.join([':%s' % i for i in keyList])}
        for idChunk in listChunkIterator(axidStr, INCR):
            valueDict = defaultDict.copy()
            valueDict.update(dict(zip(keyList, idChunk)))
            marketDB.dbCursor.execute(query, valueDict)
            r = marketDB.dbCursor.fetchmany()
            while len(r) > 0:
                for (aid, dt, share_af) in r:
                    axidAfDict.setdefault(aid, list()).append(
                        (dt.date(), float(share_af)))
                r = marketDB.dbCursor.fetchmany()
        # Construct af history for sub-issues
        for mid in missingIds:
            axidHistory = sorted(modelMarketMap.get(mid, list()))
            for (fromDt, thruDt, axid) in axidHistory:
                afHistory = axidAfDict.get(axid.getIDString(), list())
                for (afDt, afVal) in afHistory:
                    if afDt < thruDt and afDt >= fromDt:
                        cache.addChange(mid, afDt, afVal)
            cache.sortHistory(mid)
        retval = cache.getHistories(list(zip(assetList, issueList)), startDate,
                                endDate + datetime.timedelta(days=1))
        return retval
    
    def getSubFactorsForDate(self, date, factors):
        """Returns the sub-factors of the given factors for the given date.
        The return value is a list of sub-factors matching the factors.
        """
        query = """SELECT sub_id FROM sub_factor
        WHERE from_dt <= :date_arg AND thru_dt > :date_arg
        AND factor_id = :factor_arg"""
        subFactors = []
        for f in factors:
            self.dbCursor.execute(query, factor_arg=f.factorID, date_arg=date)
            ret = self.dbCursor.fetchall()
            subFactors.append(ModelSubFactor(f, ret[0][0], date))
        return subFactors
    
    def getISCPValues(self, date):
        """Returns a dictionary of dictionaries, mapping SubIssues to
        mappings of their 'linked' SubIssues to the corresponding Dickey-Fuller
        p-values.
        """
        self.dbCursor.execute("""
            SELECT dt, sub_issue1_id, sub_issue2_id, value
            FROM rmg_isc_pvalue sr, sub_issue s1, sub_issue s2
            WHERE sr.dt = :date_arg
            AND s1.sub_id = sr.sub_issue1_id
            AND s2.sub_id = sr.sub_issue2_id
            AND s1.from_dt <= sr.dt AND s1.thru_dt > sr.dt
            AND s2.from_dt <= sr.dt AND s2.thru_dt > sr.dt""",
                            date_arg=date)
        sidPValueMap = dict()
        for (sid1, sid2, pv) in self.dbCursor.fetchall():
            subid1 = SubIssue(sid1)
            subid2 = SubIssue(sid2)
            sidPValueMap.setdefault(subid1, dict())[subid2] = pv
        return sidPValueMap

    def getSpecificCovariances(self, rmi):
        """Returns a dictionary of dictionaries, mapping SubIssues to 
        mappings of their 'linked' SubIssues to the corresponding specific
        covariances.
        """
        self.dbCursor.execute("""
            SELECT sub_issue1_id, sub_issue2_id, value
            FROM rmi_specific_covariance sr, sub_issue s1, sub_issue s2
            WHERE sr.rms_id = :rms_arg AND sr.dt = :date_arg
            AND s1.sub_id = sr.sub_issue1_id 
            AND s2.sub_id = sr.sub_issue2_id 
            AND s1.from_dt <= sr.dt AND s1.thru_dt > sr.dt
            AND s2.from_dt <= sr.dt AND s2.thru_dt > sr.dt""",
                            rms_arg=rmi.rms_id, date_arg=rmi.date)
        sidCovMap = dict()
        for (sid1, sid2, cov) in self.dbCursor.fetchall():
            subid1 = SubIssue(sid1)
            subid2 = SubIssue(sid2)
            sidCovMap.setdefault(subid1, dict())[subid2] = float(cov)
        return sidCovMap
    
    def getSpecificRisks(self, rmi, restrictDates=True):
        """Returns a dictionary mapping SubIssues to specific risks
        containing the specific risks for the given risk model instance.
        """
        if restrictDates:
            self.dbCursor.execute("""
                SELECT sub_issue_id, value
                FROM rmi_specific_risk sr, sub_issue si
                WHERE sr.rms_id = :rms_arg AND sr.dt = :date_arg
                AND si.sub_id = sr.sub_issue_id 
                AND si.from_dt <= sr.dt AND si.thru_dt > sr.dt""",
                    rms_arg=rmi.rms_id, date_arg=rmi.date)
        else:
            self.dbCursor.execute("""
                SELECT sub_issue_id, value
                FROM rmi_specific_risk sr, sub_issue si
                WHERE sr.rms_id = :rms_arg AND sr.dt = :date_arg
                AND si.sub_id = sr.sub_issue_id""",
                    rms_arg=rmi.rms_id, date_arg=rmi.date)
        return dict([(SubIssue(i), float(j)) for (i,j) in self.dbCursor.fetchall()])
    
    def getSubIssueRiskModelGroupPairs(self, date, restrict=None):
        """Returns a list of (SubIssue, RiskModelGroup) tuples
        corresponding to the risk model group (quotation country).
        By default all active sub-issues are procesed but the optional 
        restrict argument can also be used.
        """
        validRiskModelGroups = self.getAllRiskModelGroups(inModels=False)
        for rmg in validRiskModelGroups:
            rmg.setRMGInfoForDate(date)
        rmgIdMap = dict([(r.rmg_id, r) for r in validRiskModelGroups])
        self.dbCursor.execute("""SELECT sub_id, rmg_id FROM sub_issue
        WHERE from_dt <= :date_arg AND thru_dt > :date_arg""",
                              date_arg=date)
        r = self.dbCursor.fetchall()
        if restrict is not None:
            sidStrSet = set([s.getSubIDString() for s in restrict])
            return [(SubIssue(string=i[0]), rmgIdMap[i[1]]) \
                        for i in r if i[0] in sidStrSet]
        else:
            return [(SubIssue(string=i[0]), rmgIdMap[i[1]]) for i in r]
    
    def insertCalendarEntry(self, rmg, currentDate, nextSequence):
        """Inserts the given calendar entry into the rmg_calendar table.
        """
        self.dbCursor.execute("""INSERT INTO rmg_calendar
        VALUES(:date_arg, :rmg_arg, :seq_arg)""",
                              date_arg=currentDate,
                              rmg_arg=rmg.rmg_id, seq_arg=nextSequence)
    
    def insertCASpinOff(self, spinOff, date):
        """Insert the given spin-off on the given date
        into the ca_spin_off table.
        """
        if spinOff.implied_dividend is None:
            self.dbCursor.execute("""INSERT INTO ca_spin_off (dt,
            ca_sequence, parent_id, child_id, share_ratio,ref)
            VALUES (:date_arg, :seq_arg, :parent_arg, :child_arg, :ratio_arg,
            :ref_arg)""",
                                  date_arg=date,
                                  seq_arg=spinOff.sequenceNumber,
                                  parent_arg=spinOff.modeldb_id.getIDString(),
                                  child_arg=spinOff.child_id.getIDString(),
                                  ratio_arg=spinOff.share_ratio,
                                  ref_arg=spinOff.ref)
        else:
            self.dbCursor.execute("""INSERT INTO ca_spin_off (dt,
            ca_sequence, parent_id, child_id, implied_div, currency_id, ref)
            VALUES (:date_arg, :seq_arg, :parent_arg, :child_arg, :div_arg,
            :currency_arg, :ref_arg)""",
                                  date_arg=date,
                                  seq_arg=spinOff.sequenceNumber,
                                  parent_arg=spinOff.modeldb_id.getIDString(),
                                  child_arg=spinOff.child_id,
                                  div_arg=spinOff.implied_dividend,
                                  currency_arg=spinOff.currency_id,
                                  ref_arg=spinOff.ref)
    
    def insertEstimationUniverse(self, rmi, subids, qualified_subids=None):
        """Inserts the given sub-issues as the estimation universe 
        of the risk model instance into the database.  
        If grandfathering is enabled, an optional argument needs to
        be provided, consisting of a list of sub-issues that 
        qualified for ESTU membersihp on the given date.
        """
        if qualified_subids is None:
            qualified_subids = subids
        qualifyMap = dict([(sid, 1.0) for sid in qualified_subids])
        query = """INSERT INTO rmi_estu
        (rms_id, dt, sub_issue_id, qualify)
        VALUES (:rms_arg, :date_arg, :issue_arg, :qualify_arg)"""
        valueDicts = [dict([('issue_arg', i.getSubIDString()),
                            ('rms_arg', rmi.rms_id),
                            ('date_arg', rmi.date),
                            ('qualify_arg', qualifyMap.get(i, 0.0))])
                      for i in subids]
        self.dbCursor.executemany(query, valueDicts)
        self.log.info('Inserted %d assets into estimation universe (%d qualified)',
                       len(valueDicts), len(qualified_subids))

    def insertEstimationUniverseV3(self, rmi, estuInstance, name):
        """Inserts the given sub-issues as an estimation universe
        of the risk model instance into the database.
        """
        if not hasattr(estuInstance, 'qualify'):
            qualified_subids = estuInstance.assets
        else:
            qualified_subids = estuInstance.qualify
        qualifyMap = dict([(sid, 1.0) for sid in qualified_subids])
        query = """INSERT INTO rmi_estu_v3
        (rms_id, nested_id, dt, sub_issue_id, qualify)
        VALUES (:rms_arg, :id_arg, :date_arg, :issue_arg, :qualify_arg)"""
        valueDicts = [dict([('issue_arg', i.getSubIDString()),
                            ('rms_arg', rmi.rms_id),
                            ('id_arg', estuInstance.id),
                            ('date_arg', rmi.date),
                            ('qualify_arg', qualifyMap.get(i, 0.0))])
                        for i in estuInstance.assets]
        self.dbCursor.executemany(query, valueDicts)
        self.log.info('Inserted %d assets into %s estimation universe (%d qualified)',
                len(valueDicts), name, len(qualified_subids))

    def insertEstimationUniverseWeights(self, rmi, subidWeightPairs):
        """Update estimation universe records of the given SubIssues
        with their corresponding regression weights for that risk model
        instance.  subidWeightPairs should be list of (SubIssue, weight)
        pairs.
        """
        query = """UPDATE rmi_estu SET weight = :weight_arg 
        WHERE sub_issue_id = :issue_arg
        AND rms_id = :rms_arg AND dt = :date_arg"""
        valueDicts = [dict([('issue_arg', i.getSubIDString()),
                            ('rms_arg', rmi.rms_id),
                            ('date_arg', rmi.date),
                            ('weight_arg', j)])
                      for (i,j) in subidWeightPairs]
        self.dbCursor.executemany(query, valueDicts)
        self.log.info('Updated %d weights for legacy estimation universe', len(valueDicts))

    def insertEstimationUniverseWeightsV3(self, rmi, estuInstance, name, subidWeightPairs):
        """Update estimation universe records of the given SubIssues
        with their corresponding regression weights for that risk model
        instance.  subidWeightPairs should be list of (SubIssue, weight)
        pairs.
        """
        query = """UPDATE rmi_estu_v3 SET weight = :weight_arg
        WHERE sub_issue_id = :issue_arg AND nested_id = :id_arg
        AND rms_id = :rms_arg AND dt = :date_arg"""
        valueDicts = [dict([('issue_arg', i.getSubIDString()),
                            ('id_arg', estuInstance.id),
                            ('rms_arg', rmi.rms_id),
                            ('date_arg', rmi.date),
                            ('weight_arg', j)])
                        for (i,j) in subidWeightPairs]
        self.dbCursor.executemany(query, valueDicts)
        self.log.info('Updated %d weights for estimation universe %s',
                len(valueDicts), name)
    
    def MapSubIDToSubIdx(self, subIssues, subSet):
        sidIdxMap = dict([(sid, sIdx) for (sIdx, sid) in enumerate(subIssues)])
        subIdxList = [sidIdxMap[sid] for sid in subSet if sid in sidIdxMap]
        return subIdxList

    def insertExposureUniverse(self, rmi, subids, excludeList=None):
        """Inserts the given sub-issues as the exposure matrix universe of the
        risk model instance into the database.
        """
        if excludeList is None:
            excludeList = []

        qualify = list()
        for (idx, sid) in enumerate(subids):
            if sid in excludeList:
                qualify.append(0)
            else:
                qualify.append(1)

        query = """INSERT INTO rmi_universe
        (rms_id, dt, sub_issue_id, qualify)
        VALUES (:rms_arg, :date_arg, :issue_arg, :qual_arg)"""
        valueDicts = [dict([('issue_arg', i.getSubIDString()),
                            ('rms_arg', rmi.rms_id),
                            ('date_arg', rmi.date),
                            ('qual_arg', j)])
                        for (i,j) in zip(subids, qualify)]

        self.dbCursor.executemany(query, valueDicts)
        self.log.info('Inserted %d assets into exposure universe',
                       len(valueDicts))
    
    def insertFactorCovariances(self, rmi, subFactors, factorCov):
        """Inserts the given factor-factor covariances into the
        table for risk model instance rmi.
        subFactors is a list of sub-factors.
        factorCov is a factor-factor array of covariances.
        Only the non-zeros of the lower triangle are stored in the database.
        """
        query = """INSERT INTO rmi_covariance
        (rms_id, dt, sub_factor1_id, sub_factor2_id, value)
        VALUES (:rms_arg, :date_arg, :factor1_arg, :factor2_arg, :value_arg)"""
        valueDicts = []
        for f1 in range(len(subFactors)):
            for f2 in range(f1 + 1):
                value = dict([('factor1_arg', subFactors[f1].subFactorID),
                              ('factor2_arg', subFactors[f2].subFactorID),
                              ('value_arg', factorCov[f1,f2]),
                              ('rms_arg', rmi.rms_id),
                              ('date_arg', rmi.date)])

                if abs(factorCov[f1,f2] - factorCov[f2,f1]) >= 1e-12:
                    logging.error('Covariance symmetry breached: Factors %s and %s',
                            subFactors[f1].factor.name, subFactors[f2].factor.name)
                    logging.error('... Values: (%s, %s), Diff: %s',
                            factorCov[f1, f2], factorCov[f2, f1], factorCov[f1,f2] - factorCov[f2,f1])
                assert(abs(factorCov[f1,f2] - factorCov[f2,f1]) < 1e-12)
                valueDicts.append(value)
        self.dbCursor.executemany(query, valueDicts)
        self.log.info('Inserted %d factor covariances', len(valueDicts))
    
    def insertFactorExposures(self, rmi, subFactor, subids, values, name):
        """Inserts the given exposures to factor subFactor in the exposure
        table for risk model instance rmi.
        issues and values are lists of SubIssues and numbers respectively.
        Only non-masked values are added to the table.
        """
        assert(len(subids) == len(values))
        query = """INSERT INTO rmi_factor_exposure
        (rms_id, dt, sub_factor_id, sub_issue_id, value)
        VALUES (:rms_arg, :date_arg, :factor_arg, :issue_arg,
        :value_arg)"""
        valueDicts = [{'issue_arg': i.getSubIDString(),
                       'value_arg': j,
                       'factor_arg': subFactor.subFactorID,
                       'rms_arg': rmi.rms_id,
                       'date_arg': rmi.date}
                      for (i,j) in zip(subids, values) if j is not ma.masked]
        if len(valueDicts) > 0:
            if len(valueDicts) <= 50:
                self.dbCursor.executemany(query, valueDicts)
                self.log.info('Inserted %d exposures: %s', len(valueDicts), name)
            else:
                for each in valueDicts:
                    self.dbCursor.executemany(query, [each])
                    self.log.info('Inserted %d exposures: %s', len(each), name)

        self.log.info('Inserted %d exposures: %s', len(valueDicts), name)
        
    def insertFactorExposureMatrix(self, rmi, expMat, subFactorMap):
        """Inserts the given exposure matrix into the exposure
        table for risk model instance rmi.
        Only non-masked values are added to the table.
        """
        for factorName in expMat.getFactorNames():
            subFactor = subFactorMap[factorName]
            factorIdx = expMat.getFactorIndex(factorName)
            values = expMat.getMatrix()[factorIdx]
            if numpy.sum(ma.getmaskarray(values)) == expMat.getMatrix().shape[1]:
                self.log.warning('No exposure values for %s' % factorName)
            self.insertFactorExposures(
                rmi, subFactor, expMat.getAssets(), values, factorName)
            
    def insertFactorExposureMatrixNew(self, rmi, expM, subFactorMap, update, setNull=None):
        """Inserts the given exposure matrix into the exposure
        table for risk model instance rmi.
        Only non-masked values are added to the table.
        If update is True, then the assumption is that all assets
        already have records in the database and SQL UPDATE should be
        used rather than SQL INSERT.
        """
        mat = expM.getMatrix()
        univ = expM.getAssets()
        numNonMissing = len(univ)
        if setNull is not None:
            numNonMissing -= len(setNull)

        binaryColumns = set()
        columnNames = list()
        inputSizes = dict()
        
        regularColumnNames = ( expM.getFactorNames(expM.InterceptFactor)
            + expM.getFactorNames(expM.StyleFactor) 
            + expM.getFactorNames(expM.StatisticalFactor)
            + expM.getFactorNames(expM.RegionalIntercept)
            + expM.getFactorNames(expM.LocalFactor)
            + expM.getFactorNames(expM.MacroCoreFactor) 
            + expM.getFactorNames(expM.MacroMarketTradedFactor) 
            + expM.getFactorNames(expM.MacroEquityFactor)
            + expM.getFactorNames(expM.MacroSectorFactor) 
            + expM.getFactorNames(expM.MacroFactor) )
        columnIndices = [expM.getFactorIndex(f) for f in regularColumnNames]
        columnSubFactors = [subFactorMap[f].subFactorID for f in expM.getFactorNames()]
        columnNames = [self.getSubFactorColumnName(subFactorMap[f]) for f in regularColumnNames]
        inputSizes.update(dict([(col, cx_Oracle.NATIVE_FLOAT) for col in columnNames]))
        binaryCountry = expM.getFactorIndices(expM.CountryFactor)
        binaryIndustry = expM.getFactorIndices(expM.IndustryFactor)
        binaryCurrency = expM.getFactorIndices(expM.CurrencyFactor)
        binaryColumnMap = dict()
        if len(binaryCountry) > 0:
            sidFactors = numpy.empty(len(univ), dtype=int)
            tmp = mat[binaryCountry, :].nonzero()
            if tmp[0].shape[0] != numNonMissing:
                raise ValueError('Missing or multiple country exposures')
            sidFactors.put(indices=tmp[1], values=tmp[0])
            binaryColumnMap['binary_country'] = (binaryCountry, sidFactors)
            inputSizes['binary_country'] = cx_Oracle.NUMBER
        if len(binaryCurrency) > 0:
            sidFactors = numpy.empty(len(univ), dtype=int)
            tmp = mat[binaryCurrency, :].nonzero()
            if tmp[0].shape[0] != len(univ):
                raise ValueError('Missing or multiple currency exposures')
            sidFactors.put(indices=tmp[1], values=tmp[0])
            binaryColumnMap['binary_currency'] = (binaryCurrency, sidFactors)
            inputSizes['binary_currency'] = cx_Oracle.NUMBER
        if len(binaryIndustry) > 0:
            sidFactors = numpy.empty(len(univ), dtype=int)
            tmp = mat[binaryIndustry, :].nonzero()
            if tmp[0].shape[0] != numNonMissing:
                raise ValueError('Missing or multiple industry exposures')
            sidFactors.put(indices=tmp[1], values=tmp[0])
            binaryColumnMap['binary_industry'] = (binaryIndustry, sidFactors)
            inputSizes['binary_industry'] = cx_Oracle.NUMBER
        binaryColumns = list(binaryColumnMap.keys())
        fullColumnArgs = [':%s' % c for c in columnNames]
        binaryColumnArgs = [':%s' % c for c in binaryColumns]
        if update:
            query = """UPDATE %(table)s SET %(setArgs)s
              WHERE sub_issue_id=:sid AND  dt=:dt""" % {
                'table': self.getWideExposureTableName(rmi.rms_id),
                'setArgs': ','.join(['%s = %s' % (col, arg) for (col,arg)
                                    in zip(columnNames + binaryColumns,
                                           fullColumnArgs + binaryColumnArgs)])
                }
        else:
            query = """INSERT INTO %(table)s (sub_issue_id, dt, %(sfColumns)s)
               VALUES(:sid, :dt, %(sfColumnArgs)s)""" % {
                'table': self.getWideExposureTableName(rmi.rms_id),
                'sfColumns': ','.join(columnNames + binaryColumns),
                'sfColumnArgs': ','.join(fullColumnArgs + binaryColumnArgs)
                }
        valueDicts = list()
        defaultDict = dict(zip(columnNames + binaryColumns,
                               [None] * len(columnNames + binaryColumns)))
        defaultDict['dt'] = rmi.date
        for (sidIdx, sid) in enumerate(univ):
            myDict = dict(defaultDict)
            fullColDict = dict([(arg, mat[colIdx, sidIdx]) for (arg, colIdx)
                                in zip(columnNames, columnIndices)
                                if mat[colIdx, sidIdx] is not ma.masked])
            for (field, (columnList, indexMat)) in binaryColumnMap.items():
                if (setNull is not None) and (sid in setNull):
                    if field == 'binary_currency':
                        myDict[field] = columnSubFactors[columnList[indexMat[sidIdx]]]
                else:
                    myDict[field] = columnSubFactors[columnList[indexMat[sidIdx]]]
            myDict['sid'] = sid.getSubIDString()
            myDict.update(fullColDict)
            valueDicts.append(myDict)
        if len(valueDicts) > 0:
            self.dbCursor.setinputsizes(**inputSizes)
            self.dbCursor.executemany(query, valueDicts)
            logging.info('Inserted exposures for %d sub-issues', len(valueDicts))

    def insertStandardisationStats(self, rms_id, date, descriptors,
                    descrMapping, meanDict, stndDict):
        """Inserts standardisation means and stdevs for risk model series
        rms_id on the given date for the descriptors which comprise the style factors.
        """
        query = """INSERT INTO rms_stnd_stats
                   (rms_id, descriptor_id, dt, mean, stdev)
                   VALUES (:rms_arg, :desc_arg, :date_arg, :mean_arg, :stdev_arg)"""
        valueDicts = [dict([('desc_arg', descrMapping[ds]),
            ('mean_arg', meanDict[ds]),
            ('stdev_arg', stndDict[ds]),
            ('rms_arg', rms_id),
            ('date_arg', date)])
            for ds in descriptors if (ds in meanDict and ds in stndDict\
                                    and meanDict[ds] is not ma.masked\
                                    and stndDict[ds] is not ma.masked)]
        self.dbCursor.executemany(query, valueDicts)
        self.log.info('Inserted %d descriptor standardisation means', len(valueDicts))

    def insertStandardisationDesc(self, rms_id, date, descList, 
            regionList, meanList, stdevList):
        """Inserts standardization means and stdevs for risk model series
        rms_id on the given date"""
        query = """INSERT INTO RMS_STND_DESC
                   (rms_id, descriptor_id, region_id, dt, mean, stdev)
                   VALUES (:rms_arg, :descriptor_arg, :region_arg, :date_arg, :mean_arg, :stdev_arg)"""
        valueDicts = [dict([('descriptor_arg', d),
            ('region_arg', r),
            ('mean_arg', m),
            ('stdev_arg', s),
            ('rms_arg', rms_id),
            ('date_arg', date)])
            for (d,r,m,s) in zip(descList, regionList, meanList, stdevList) 
            if m is not ma.masked and s is not ma.masked]
        self.dbCursor.executemany(query, valueDicts)
        self.log.info('Inserted %d standardization means', len(valueDicts))

    def insertStandardisationExp(self, rms_id, date, sfList, 
            regionList, meanList, stdevList):
        """Inserts standardization means and stdevs for risk model series
        rms_id on the given date"""
        query = """INSERT INTO RMS_STND_EXP
                   (rms_id, sub_factor_id, region_id, dt, mean, stdev)
                   VALUES (:rms_arg, :factor_arg, :region_arg, :date_arg, :mean_arg, :stdev_arg)"""
        valueDicts = [dict([('factor_arg', i.subFactorID),
            ('region_arg', r),
            ('mean_arg', m),
            ('stdev_arg', s),
            ('rms_arg', rms_id),
            ('date_arg', date)])
            for (i,r,m,s) in zip(sfList, regionList, meanList, stdevList) 
            if m is not ma.masked and s is not ma.masked]
        self.dbCursor.executemany(query, valueDicts)
        self.log.info('Inserted %d standardization means', len(valueDicts))

    def insertStandardizationMean(self, rms_id, date, subFactors,
            values, update=False):
        """Inserts standardization means for risk model series
        rms_id on the given date.
        subFactors and values are lists of sub-factors and numbers respectively.
        """
        if update:
            query = """UPDATE rms_stnd_mean
                       SET value = :value_arg
                       WHERE rms_id=:rms_arg AND sub_factor_id=:factor_arg
                       AND dt=:date_arg"""
            valueDicts = [dict([('factor_arg', i.subFactorID),
                ('value_arg', j),
                ('rms_arg', rms_id),
                ('date_arg', date)])
                for (i,j) in zip(subFactors, values) if j is not ma.masked]
        else:
            query = """INSERT INTO rms_stnd_mean
                       (rms_id, sub_factor_id, dt, value)
                       VALUES (:rms_arg, :factor_arg, :date_arg, :value_arg)"""
            valueDicts = [dict([('factor_arg', i.subFactorID),
                ('value_arg', j),
                ('rms_arg', rms_id),
                ('date_arg', date)])
                for (i,j) in zip(subFactors, values) if j is not ma.masked]
        self.dbCursor.executemany(query, valueDicts)
        self.log.info('Inserted %d standardization means', len(valueDicts))

    def insertStandardizationStDev(self, rms_id, date, subFactors,
            values, update=False):
        """Inserts standardization stdev for risk model series
        rms_id on the given date.
        subFactors and values are lists of sub-factors and numbers respectively.
        """
        if update:
            query = """UPDATE rms_stnd_stdev
            SET value = :value_arg
            WHERE rms_id=:rms_arg AND sub_factor_id=:factor_arg
            AND dt=:date_arg"""
            valueDicts = [dict([('factor_arg', i.subFactorID),
                ('value_arg', j),
                ('rms_arg', rms_id),
                ('date_arg', date)])
                for (i,j) in zip(subFactors, values) if j is not ma.masked]
        else:
            query = """INSERT INTO rms_stnd_stdev
            (rms_id, sub_factor_id, dt, value)
            VALUES (:rms_arg, :factor_arg, :date_arg, :value_arg)"""
            valueDicts = [dict([('factor_arg', i.subFactorID),
                ('value_arg', j),
                ('rms_arg', rms_id),
                ('date_arg', date)])
                for (i,j) in zip(subFactors, values) if j is not ma.masked]
        self.dbCursor.executemany(query, valueDicts)
        self.log.info('Inserted %d standardization stdevs', len(valueDicts))

    def insertFactorReturns(self, rms_id, date, subFactors, values, flag=None, addedTags=None):
        """Inserts the given internal factor returns for the subFactors in the factor return
        table for risk model series rms_id on the given date.
        subFactors and values are lists of sub-factors and numbers
        respectively.
        addedTags is used by RMM only and may be ignored here
        """
        if flag is None:
            tableName = 'rms_factor_return'
        else:
            tableName = 'rms_factor_return_%s' % flag
        query = """INSERT INTO %(table)s
        (rms_id, sub_factor_id, dt, value)
        VALUES (:rms_arg, :factor_arg, :date_arg, :value_arg)""" % { 'table': tableName }
        valueDicts = [dict([('factor_arg', i.subFactorID),
                            ('value_arg', j),
                            ('rms_arg', rms_id),
                            ('date_arg', date)])
                      for (i,j) in zip(subFactors, values) if j is not ma.masked]
        self.dbCursor.executemany(query, valueDicts)
        if (rms_id, tableName) in self.factorReturnCache:
            self.factorReturnCache[(rms_id, tableName)].removeDate(date)
        self.log.info('Inserted %d factor returns to %s', len(valueDicts), tableName)

    def insertRawFactorReturns(self, rms_id, date, subFactors, values):
        """Inserts the given raw returns for the subFactors in the raw factor return
        table for risk model series rms_id on the given date.
        subFactors and values are lists of sub-factors and numbers
        respectively.
        """
        query = """INSERT INTO rms_raw_factor_return
        (rms_id, sub_factor_id, dt, value)
        VALUES (:rms_arg, :factor_arg, :date_arg, :value_arg)"""
        valueDicts = [dict([('factor_arg', i.subFactorID),
                            ('value_arg', j),
                            ('rms_arg', rms_id),
                            ('date_arg', date)])
                      for (i,j) in zip(subFactors, values) if j is not ma.masked]
        self.dbCursor.executemany(query, valueDicts)
        self.log.info('Inserted %d raw factor returns', len(valueDicts))
        
    def insertStatFactorReturns(self, rms_id, date, exp_dt, subFactors, values):
        """Inserts the returns for given date for the subFactors in the factor return
        table for statistical risk model series rms_id on the given exposure date.
        subFactors and values are lists of sub-factors and numbers
        respectively.
        """
        if rms_id < 0:
            rms_id_str = 'm%.2d' % abs(rms_id)
        else:
            rms_id_str = rms_id

        query = """INSERT INTO rms_%(rms_id_str)s_stat_factor_return
        (rms_id, sub_factor_id, exp_dt, dt, value)
        VALUES (:rms_arg, :factor_arg, :exp_dt_arg, :date_arg, :value_arg)""" % {'rms_id_str': rms_id_str}
        valueDicts = [dict([('factor_arg', i.subFactorID),
                            ('value_arg', j),
                            ('rms_arg', rms_id),
                            ('exp_dt_arg', exp_dt),
                            ('date_arg', date)])
                      for (i,j) in zip(subFactors, values) if j is not ma.masked]

        self.dbCursor.executemany(query, valueDicts)
        self.log.debug('Inserted %d statistical factor returns for (%s,%s)',
                len(valueDicts), date, exp_dt)

    def insertRMIPredictedBeta(self, rmi, assetValuePairs):
        """Inserts the given predicted beta values for the given
        risk model instance.
        assetValuePairs is a list of (SubIssue, beta) pairs.
        """
        query = """INSERT INTO rmi_predicted_beta
        (rms_id, sub_issue_id, dt, value)
        VALUES (:rms_arg, :subid_arg, :date_arg, :value_arg)"""
        valueDicts = [dict([('subid_arg', i.getSubIDString()),
                            ('value_arg', j),
                            ('rms_arg', rmi.rms_id),
                            ('date_arg', rmi.date)])
                      for (i,j) in assetValuePairs]
        if len(valueDicts) > 0:
            self.dbCursor.executemany(query, valueDicts)
        self.log.info('Inserted %d predicted betas', len(valueDicts))

    def insertRMIPredictedBetaV3(self, rmi, assetValuePairs):
        """Inserts the given predicted beta values for the given
        risk model instance.
        assetValuePairs is a list of (SubIssue, beta1,..,betak) values
        """
        query = """INSERT INTO rmi_predicted_beta_v3
        (rms_id, sub_issue_id, dt, global_beta, local_num_beta, local_beta)
        VALUES (:rms_arg, :subid_arg, :date_arg, :global_arg, :legacy_arg, :local_arg)"""
        valueDicts = [dict([('subid_arg', i.getSubIDString()),
                            ('global_arg', maskToNone(b0)),
                            ('legacy_arg', maskToNone(b1)),
                            ('local_arg', maskToNone(b2)),
                            ('rms_arg', rmi.rms_id),
                            ('date_arg', rmi.date)])
                        for (i,b0,b1,b2) in assetValuePairs]
        if len(valueDicts) > 0:
            self.dbCursor.executemany(query, valueDicts)
        self.log.info('Inserted %d V3 predicted betas', len(valueDicts))

    def insertRMITotalRisk(self, rmi, assetValuePairs):
        """Inserts the given total risk values for the given
        risk model instance.
        assetValuePairs is a list of (SubIssue, beta) pairs.
        """
        query = """INSERT INTO rmi_total_risk
        (rms_id, sub_issue_id, dt, value)
        VALUES (:rms_arg, :subid_arg, :date_arg, :value_arg)"""
        valueDicts = [dict([('subid_arg', i.getSubIDString()),
                            ('value_arg', j),
                            ('rms_arg', rmi.rms_id),
                            ('date_arg', rmi.date)])
                      for (i,j) in assetValuePairs]
        if len(valueDicts) > 0:
            self.dbCursor.executemany(query, valueDicts)
        self.log.info('Inserted %d total risk values', len(valueDicts))
        
    def insertRMSFactors(self, rms_id, factorIDs):
        """Inserts the given factors into the rms_factor table for the
        given rms_id.
        """
        self.dbCursor.executemany("""INSERT INTO rms_factor (rms_id, factor_id)
        VALUES(:rms_arg, :factor_arg)""",
                                  [dict([('factor_arg',f),
                                         ('rms_arg', rms_id)])
                                   for f in factorIDs])
        
    def insertRMSFactorsForDates(self, rms_id, factorIDs, fromDt, thruDt):
        """Inserts the given factors into the rms_factor table for the
        given rms_id.
        """
        
        self.dbCursor.executemany("""INSERT INTO rms_factor (rms_id, factor_id, from_dt, thru_dt)
        VALUES(:rms_arg, :factor_arg, '%s', '%s')""" % (fromDt, thruDt),
                                  [dict([('factor_arg',f),
                                      ('rms_arg', rms_id)])
                                   for f in factorIDs])

    def insertRMSFactorStatistics(self, rms_id, date, subFactors,
                                  stdErr, tVal, tProb, regConstrWeight, VIF=None, flag=None, addedTags=None):
        """Inserts the given factor statistics in the rms_factor_statistics
        table for risk model series rms_id on the given date.
        Masked values are set to NULL. If all values for a factor
        are masked, then the whole entry is omitted.
        addedTags is used by RMM only and may be ignored here
        """
        if len(subFactors) == 0:
            return
        if flag is None:
            tableName = "rms_factor_statistics"
        else:
            tableName = "rms_factor_statistics_%s" % flag
        query = """INSERT INTO %(table_arg)s 
        (rms_id, dt, sub_factor_id, std_error, t_value, t_probability,
         regr_constraint_weight)
        VALUES(:rms_arg, :date_arg, :factor_arg, :stderr_arg, :tval_arg,
        :tprob_arg, :reg_weight_arg)""" % {'table_arg': tableName}
        valueDicts = []
        assert(len(stdErr.shape) == 1)
        assert(stdErr.shape[0] == len(subFactors))
        assert(len(tVal.shape) == 1)
        assert(tVal.shape[0] == len(subFactors))
        assert(len(tProb.shape) == 1)
        assert(tProb.shape[0] == len(subFactors))
        for i in range(len(subFactors)):
            values = {}
            values['rms_arg'] = rms_id
            values['date_arg'] = date
            values['factor_arg'] = subFactors[i].subFactorID
            values['stderr_arg'] = maskToNone(stdErr[i])
            values['tval_arg'] = maskToNone(tVal[i])
            values['tprob_arg'] = maskToNone(tProb[i])
            values['reg_weight_arg'] = maskToNone(regConstrWeight[i])
            allNone = False
            if values['stderr_arg'] == None and values['tval_arg'] == None \
                   and values['tprob_arg'] == None:
                allNone = True
            if not allNone:
                valueDicts.append(values)
        
        self.dbCursor.setinputsizes(
            factor_arg=cx_Oracle.NUMBER, stderr_arg=cx_Oracle.NUMBER,
            tval_arg=cx_Oracle.NUMBER, tprob_arg=cx_Oracle.NUMBER,
            reg_weight_arg=cx_Oracle.NUMBER)
        self.dbCursor.executemany(query, valueDicts)
        self.log.info('Inserted %s rows into %s', len(valueDicts), tableName)
        
    def insertRMSIssues(self, rms_id, issueList):
        """Inserts the given issues into the rms_issue table for the
        given rms_id.
        issueList is a list of (ModelID, from_dt, thru_dt) tuples.
        """
        self.dbCursor.executemany("""INSERT INTO rms_issue
        (rms_id, issue_id, from_dt, thru_dt)
        VALUES(:rms_arg, :issue_arg, :from_arg, :thru_arg)""",
                                  [dict([('issue_arg',i[0].getIDString()),
                                         ('rms_arg', rms_id),
                                         ('from_arg', i[1]),
                                         ('thru_arg', i[2])])
                                   for i in issueList])
       
    def insertRMSStatistics(self, rms_id, date, adjRsquared, pcttrade=None, pctgVar=None, flag=None):
        """Inserts the given summary statistics in the rms_statistics
        table for risk model series rms_id on the given date.
        """
        if flag is None:
            tableName = "rms_statistics"
        else:
            tableName = "rms_statistics_%s" % flag
        if pctgVar is None:
            if pcttrade is None:
                query = """INSERT INTO %(table_arg)s
                (rms_id, dt, adjRsquared)
                VALUES (:rms_arg, :date_arg, :adjRsq_arg)""" % {'table_arg': tableName}
                self.dbCursor.execute(query, adjRsq_arg=adjRsquared,
                        rms_arg=rms_id, date_arg=date)
            else:
                query = """INSERT INTO %(table_arg)s
                (rms_id, dt, adjRsquared, pcttrade)
                VALUES (:rms_arg, :date_arg, :adjRsq_arg, :pcttrade_arg)""" % {'table_arg': tableName}
                self.dbCursor.execute(query, adjRsq_arg=adjRsquared,
                        rms_arg=rms_id, date_arg=date, pcttrade_arg=pcttrade)
        else:
            if pcttrade is None:
                query = """INSERT INTO %(table_arg)s
                (rms_id, dt, adjRsquared, pctgVar)
                VALUES (:rms_arg, :date_arg, :adjRsq_arg, :pctVar_arg)""" % {'table_arg': tableName}
                self.dbCursor.execute(query, adjRsq_arg=adjRsquared,
                        rms_arg=rms_id, date_arg=date, pctVar_arg=pctgVar)
            else:
                query = """INSERT INTO %(table_arg)s
                (rms_id, dt, adjRsquared, pctgVar, pcttrade)
                VALUES (:rms_arg, :date_arg, :adjRsq_arg, :pctVar_arg, :pcttrade_arg)""" % {'table_arg': tableName}
                self.dbCursor.execute(query, adjRsq_arg=adjRsquared,
                        rms_arg=rms_id, date_arg=date, pctVar_arg=pctgVar, pcttrade_arg=pcttrade)
        self.log.info('Inserted summary statistics into %s', tableName)

    def insertDVAStatistics(self, rmi, dva_scale, nw_scale):
        """Inserts the given DVA scale statistics in the rms_dva_statistics
        table for risk model series rms_id on the given date.
        """
        tableName = "rms_dva_statistics"
        query = """INSERT INTO %(table_arg)s
        (rms_id, dt, dva_scale, nw_scale)
        VALUES (:rms_arg, :date_arg, :dvaScale_arg, :nwScale_arg)""" % {'table_arg': tableName}
        self.dbCursor.execute(query, dvaScale_arg=dva_scale, nwScale_arg=nw_scale, rms_arg=rmi.rms_id, date_arg=rmi.date)
        self.log.info('Inserted summary statistics into %s', tableName)

    def insertSpecificReturns(self, rms_id, date, subIssues, values, addedTags=None,
            estuWeights=None, internal=False):
        """Inserts the given returns for the sub-issues in the specific return
        table for risk model series rms_id on the given date.
        subIssues and values are lists of SubIssues and numbers
        respectively.
        addedTags and estuWeights are arguments that are used by RMMs ModelDB and 
        can be ignored by the oracle ModelDB
        """
        if internal:
            tableName = 'rms_specific_return_internal'
        else:
            tableName = 'rms_specific_return'

        query = """INSERT INTO %(table_arg)s
        (rms_id, sub_issue_id, dt, value)
        VALUES (:rms_arg, :issue_arg, :date_arg, :value_arg)""" % {'table_arg': tableName}
        valueDicts = [dict([('issue_arg', i.getSubIDString()),
                            ('value_arg', j),
                            ('rms_arg', rms_id),
                            ('date_arg', date)])
                      for (i,j) in zip(subIssues, values)]
        self.dbCursor.executemany(query, valueDicts)
        self.log.info('Inserted %d specific returns to %s', len(valueDicts), tableName)

    def insertRobustWeights(self, rms_id, date, robustWeightMap, flag=None):
        """Inserts the robust regression weights for the estimation universe
        assets used in the factor regression. 
        Only weights that differ from one are stored.
        """
        if flag is None:
            tableName = "rms_robust_weight"
        else:
            tableName = "rms_robust_weight_%s" % flag
        for (iReg, rWtMap) in robustWeightMap.items():
            query = """INSERT INTO %(table_arg)s
            (rms_id, dt, reg_id, sub_issue_id, value)
            VALUES (:rms_arg, :date_arg, :reg_arg, :issue_arg, :value_arg)""" % {'table_arg': tableName}
            valueDicts = [dict([('issue_arg', i.getSubIDString()),
                                ('value_arg', j),
                                ('rms_arg', rms_id),
                                ('date_arg', date),
                                ('reg_arg', iReg)])
                        for (i,j) in rWtMap.items() if j is not ma.masked]
            self.dbCursor.executemany(query, valueDicts)
            self.log.info('Inserted %d robust weights to %s for regression %d', len(valueDicts), tableName, iReg+1)

    def insertFMPs(self, rms_id, date, fmpMap, saveIDs=None):
        """Inserts the FMPs for the estimation universe assets used in the factor regression.
        """
        if rms_id < 0:
            tableName = 'rms_m%.2d_fmp' % abs(rms_id)
        else:
            tableName = 'rms_%s_fmp' % rms_id
        inputSizes = dict()

        # Get lists of sub-factors and subissues
        if saveIDs is None:
            sfIDList = list(fmpMap.keys())
        else:
            sfIDList = [sf_id for sf_id in fmpMap.keys() if sf_id in saveIDs]
        sidList = []
        for sf_id in sfIDList:
            sidList.extend(list(fmpMap[sf_id].keys()))
        sidList = list(set(sidList))
        sfStrList = ['sf_%s' % sf_id for sf_id in sfIDList]
        argList = [':sf_%s' % sf_id for sf_id in sfIDList]
        inputSizes.update(dict([(col, cx_Oracle.NATIVE_FLOAT) for col in sfStrList]))

        # Set up array of FMPs
        valueArray = Matrices.allMasked((len(sidList), len(sfIDList)))
        for (idx, sid) in enumerate(sidList):
            for (jdx, sf_id) in enumerate(sfIDList):
                if sid in fmpMap[sf_id]:
                    valueArray[idx, jdx] = fmpMap[sf_id][sid] 

        # Set up query
        insertQuery = """INSERT INTO %(table_arg)s \
                (sub_issue_id, dt, %(fields)s)
                VALUES(:sub_issue_id, :dt_arg, %(args)s)""" % \
                {'table_arg': tableName, 'fields': ','.join(sfStrList), 'args': ','.join(argList)}
        
        # Set up values to be passed to query
        valueDicts = list()
        for (idx, sid) in enumerate(sidList):
            sidstr = sid.getSubIDString()
            # Set up dict of items to be added/updated
            valueDict = dict()
            valueDict['sub_issue_id'] = sidstr
            valueDict['dt_arg'] = date
            for (jdx, sf_id) in enumerate(sfIDList):
                nm = 'sf_%s' % sf_id
                val = valueArray[idx, jdx]
                if val is ma.masked:
                    valueDict[nm] = None
                else:
                    valueDict[nm] = val
            valueDicts.append(valueDict)

        # Write FMP matrix to DB
        if len(valueDicts) > 0:
            self.dbCursor.setinputsizes(**inputSizes)
            self.dbCursor.executemany(insertQuery, valueDicts)
            self.log.info('Inserting %d FMP entries for %d factors into %s',
                len(valueDicts), len(sfIDList), tableName )

    def insertProxyReturns(self, date, subIssues, values):
        """Inserts the given proxy returns
        subIssues and values are lists of SubIssues and numbers
        respectively.
        """
        query = """INSERT INTO rmg_proxy_return
        (sub_issue_id, dt, value)
        VALUES (:issue_arg, :date_arg, :value_arg)"""
        valueDicts = [dict([('issue_arg', i.getSubIDString()),
                            ('value_arg', j),
                            ('date_arg', date)])
                      for (i,j) in zip(subIssues, values)]
        self.dbCursor.executemany(query, valueDicts)
        self.log.info('Inserted %d proxy returns', len(valueDicts))

    def insertISCScores(self, date, subIssues, values):
        """Inserts the given ISC asset scores
        subIssues and values are lists of SubIssues and numbers
        respectively.
        """
        query = """INSERT INTO rmg_isc_score
        (sub_issue_id, dt, value)
        VALUES (:issue_arg, :date_arg, :value_arg)"""
        valueDicts = [dict([('issue_arg', i.getSubIDString()),
                            ('value_arg', j),
                            ('date_arg', date)])
                      for (i,j) in zip(subIssues, values)]
        self.dbCursor.executemany(query, valueDicts)
        self.log.info('Inserted %d ISC scores', len(valueDicts))
 
    def insertISCBetaCoefficients(self, date, subIssueBetaMap):
        """Inserts cointegration beta coefficients for the given SubIssue pairs
        subIssuePairs and values are lists of 'linked' (SubIssue, SubIssue)
        pairs and numbers respectively.
        """
        query = """INSERT INTO rmg_isc_beta
        (dt, sub_issue1_id, sub_issue2_id, value)
        VALUES (:date_arg, :issue1_arg, :issue2_arg, :value_arg)"""
        valueDicts = list()
        processedPairs = set()
        for (sid1, betaMap) in subIssueBetaMap.items():
            for (sid2, beta) in betaMap.items():
                if (sid1, sid2) in processedPairs or (sid2, sid1) in processedPairs:
                    continue
                processedPairs.add((sid1, sid2))
                value = dict([('issue1_arg', sid1.getSubIDString()),
                              ('issue2_arg', sid2.getSubIDString()),
                              ('value_arg', beta),
                              ('date_arg', date)])
                valueDicts.append(value)
        self.dbCursor.executemany(query, valueDicts)
        self.log.info('Inserted %d cointegration betas', len(valueDicts))

    def insertISCPValues(self, date, subIssuePValueMap):
        """Inserts cointegration Dickey-Fuller p-values for given SubIssue pairs
        subIssuePairs and values are lists of 'linked' (SubIssue, SubIssue)
        pairs and numbers respectively.
        """
        query = """INSERT INTO rmg_isc_pvalue
        (dt, sub_issue1_id, sub_issue2_id, value)
        VALUES (:date_arg, :issue1_arg, :issue2_arg, :value_arg)"""
        valueDicts = list()
        processedPairs = set()
        for (sid1, pvalueMap) in subIssuePValueMap.items():
            for (sid2, pvalue) in pvalueMap.items():
                if (sid1, sid2) in processedPairs or (sid2, sid1) in processedPairs:
                    continue
                processedPairs.add((sid1, sid2))
                value = dict([('issue1_arg', sid1.getSubIDString()),
                              ('issue2_arg', sid2.getSubIDString()),
                              ('value_arg', pvalue),
                              ('date_arg', date)])
                valueDicts.append(value)
        self.dbCursor.executemany(query, valueDicts)
        self.log.info('Inserted %d cointegration p-values', len(valueDicts))

    def insertSpecificCovariances(self, rmi, subIssueCovMap):
        """Inserts specific covariances for the given SubIssue pairs
        in the specific covariance table for risk model instance rmi.
        subIssuePairs and values are lists of 'linked' (SubIssue, SubIssue) 
        pairs and numbers respectively.
        """
        query = """INSERT INTO rmi_specific_covariance
        (rms_id, dt, sub_issue1_id, sub_issue2_id, value)
        VALUES (:rms_arg, :date_arg, :issue1_arg, :issue2_arg, :value_arg)"""
        valueDicts = list()
        processedPairs = set()
        for (sid1, covMap) in subIssueCovMap.items():
            for (sid2, cov) in covMap.items():
                if sid1 == sid2:
                    continue
                if (sid1, sid2) in processedPairs or (sid2, sid1) in processedPairs:
                    continue
                if sid2 in subIssueCovMap:
                    assert(abs(subIssueCovMap[sid2].get(sid1, cov) - cov) < 1e-12)
                processedPairs.add((sid1, sid2))
                value = dict([('issue1_arg', sid1.getSubIDString()),
                              ('issue2_arg', sid2.getSubIDString()),
                              ('value_arg', cov),
                              ('rms_arg', rmi.rms_id),
                              ('date_arg', rmi.date)])
                valueDicts.append(value)
        self.dbCursor.executemany(query, valueDicts)
        self.log.info('Inserted %d specific covariances', len(valueDicts))
        
    def insertSpecificRisks(self, rmi, subIssues, values):
        """Inserts the given specific risks for the sub-issues in the
        specific risk table for risk model instance rmi.
        subIssues and values are lists of SubIssues and numbers
        respectively.
        """
        query = """INSERT INTO rmi_specific_risk
        (rms_id, dt, sub_issue_id, value)
        VALUES (:rms_arg, :date_arg, :issue_arg, :value_arg)"""
        valueDicts = [dict([('issue_arg', i.getSubIDString()),
                            ('value_arg', j),
                            ('rms_arg', rmi.rms_id),
                            ('date_arg', rmi.date)])
                      for (i,j) in zip(subIssues, values)]
        self.dbCursor.executemany(query, valueDicts)
        self.log.info('Inserted %d specific risks', len(valueDicts))
        
    def loadACPHistory(self, dateList, subids, marketDB, convertTo):
        """Returns the adjusted closing prices for the given days in dateList
        and sub-issues in the subids list.
        The return value is a TimeSeriesMatrix of ACPs.
        If an assets has no price recorded for a given day, the
        corresponding entry in the matrix is masked.
        Prices are converted to the currencies specified in convertTo.
        The adjustment is such that all prices are comparable with the
        price as of the latest date in the date list.
        """
        self.log.debug('loadACPHistory: begin')
        ucp = self.loadUCPHistory(dateList, subids, convertTo)
        startDate = min(dateList)
        endDate = max(dateList)
        afs = self.getShareAdjustmentFactors(startDate, endDate, subids,
                                             marketDB)
        sortedDates = sorted(dateList, reverse=True)
        dtIdxMap = dict([(j,i) for (i,j) in enumerate(dateList)])
        sortedDatesIdx = [(dt, dtIdxMap[dt]) for dt in sortedDates]
        # Adjust prices here
        acp = ucp
        for (sIdx, sid) in enumerate(subids):
            sidAFs = afs[sid]
            caf = 1.0
            aIdx = -1
            for (dt, dIdx) in sortedDatesIdx:
                while -aIdx <= len(sidAFs) and dt < sidAFs[aIdx][0]:
                    caf *= sidAFs[aIdx][1]
                    aIdx -= 1
                if acp.data[sIdx, dIdx] is not ma.masked:
                    acp.data[sIdx, dIdx] /= caf
        self.log.debug('loadACPHistory: end')
        return acp

    def loadMarketIdentifierHistory(self, assetList, marketDB,
                                    tableName, fieldName, cache=None):
        """Returns a dictionary mapping sub-issue/issue to their identifier
        history in the specified MarketDB table.
        """
        self.log.debug('loadMarketIdentifierHistory: begin %s', tableName)
        if cache is None:
            cache = FromThruCache()
        self.loadMarketIdentifierHistoryCache(
            assetList, marketDB, tableName, fieldName, cache)
        retval = dict()
        issueList = list()
        for a in assetList:
            if isinstance(a, SubIssue):
                issueList.append(a.getModelID())
            else:
                issueList.append(a)
        for (asset, key) in zip(assetList, issueList):
            retval[asset] = cache.getAssetHistory(key)
        self.log.debug('loadMarketIdentifierHistory: end')
        return retval
    
    def loadMarketIdentifierHistoryCache(self, assetList, marketDB,
                                         tableName, fieldName, cache):
        """Returns a dictionary mapping sub-issue/issue to their identifier
        history in the specified MarketDB table.
        """
        self.log.debug('loadMarketIdentifierHistoryCache')
        issueList = list()
        for a in assetList:
            if isinstance(a, SubIssue):
                issueList.append(a.getModelID())
            else:
                issueList.append(a)
        missingIds = cache.getMissingIds(issueList)
        (axidStr, modelMarketMap) = self.getMarketDB_IDs(marketDB, missingIds)
        # Get identifier history for MarketDB IDs
        if tableName.lower() == 'asset_dim_trading_currency':
            tableName = 'asset_dim_trading_curr'

        elif tableName.lower() == 'asset_dim_company_by_country':
            tableName = 'asset_dim_cmpbyctry'
        elif tableName.lower() == 'future_dim_trading_currency':
            tableName = 'future_dim_trading_curr'
        elif tableName.lower() == 'asset_dim_stoxx_country':
            tableName = 'asset_dim_stoxx_ctry'
        axidIDDict = dict()
        INCR=200
        keyList = ['key%d' % i for i in range(INCR)]
        defaultDict = dict([(a, None) for a in keyList])
        
        query = """SELECT axioma_id, change_dt, change_del_flag, %(field)s
        FROM %(table)s_active
        WHERE axioma_id IN (%(keys)s)
        ORDER BY change_dt ASC""" % {
            'field': fieldName, 'table': tableName,
            'keys': ','.join([':%s' % i for i in keyList])}
        for idChunk in listChunkIterator(axidStr, INCR):
            valueDict = defaultDict.copy()
            valueDict.update(dict(zip(keyList, idChunk)))
            marketDB.dbCursor.execute(query, valueDict)
            r = marketDB.dbCursor.fetchmany()
            while len(r) > 0:
                for (aid, changeDt, changeDelFlag, val) in r:
                    if changeDelFlag == 'Y':
                        val = None
                    axidIDDict.setdefault(aid, list()).append(
                        (changeDt.date(), val))
                r = marketDB.dbCursor.fetchmany()
        # Construct identifier history for sub-issues
        EOT = datetime.date(2999, 12, 31)
        for mid in missingIds:
            axidHistory = sorted(modelMarketMap.get(mid, list()))
            midHistory = list()
            for (fromDt, thruDt, axid) in axidHistory:
                afHistory = axidIDDict.get(axid.getIDString(), list())
                for ((idFromDt, idVal), (idThruDt, ignore)) in zip(
                    afHistory, afHistory[1:] + [(EOT, None)]):
                    if idFromDt < thruDt and idThruDt > fromDt:
                        if idVal is None:
                            continue
                        val = Utilities.Struct()
                        val.fromDt = max(idFromDt, fromDt)
                        val.thruDt = min(idThruDt, thruDt)
                        val.id = idVal
                        
                        midHistory.append(val)
            cache.addAssetValues(mid, midHistory)
    
    def loadCumulativeFactorReturnsHistory(self, rms_id, subFactors, dateList):
        """Returns the cumulative factor returns of the sub-factors
        in subFactorss for the given days and risk model series ID.
        The return value is a TimeSeriesMatrix of cumulative factor returns.
        If a factor has no cumulative return recorded for a given day, the
        corresponding entry in the matrix is masked.
        """
        self.log.debug('loadCumulativeFactorReturnsHistory: begin')
        results = self.loadSubFactorData(
            dateList, subFactors, 'rms_factor_return', 'data.cumulative',
            'data.rms_id = %d AND data.cumulative is not null' % rms_id, None)
        self.log.debug('loadCumulativeFactorReturnsHistory: end')
        return results
    
    def loadExchangeRateHistory(self, rmgList, date, daysBack, 
                                   currencies, base, dateList=None, idlookup=True):
        """Returns a TimeSeriesMatrix of exchange rates 
        for the currencies specified in currencyCodes, computed
        in the baseCurrencyID.  In other words, if base currency
        appreciates, the exchange rate decreases.
        If idlookup=True, the latter two arguments are assumed to 
        be ISO currency codes, not currency IDs.
        If dateList is not None, but rather a list of dates,
        we ignore date and daysBack, and compute returns for
        all dates in dateList
        """
        self.log.debug('loadExchangeRateHistory: begin')
        if not self.currencyCache:
            raise ValueError('Currency conversion requires a ForexCache object')
        if dateList is not None:
            date = dateList[-1]
        else:
            dateList = self.getDates(rmgList, date, daysBack+1)
        if idlookup:
            currencyIDs = [self.currencyCache.getCurrencyID(c, date) \
                                            for c in currencies]
            baseCurrencyID = self.currencyCache.getCurrencyID(
                                            base, date)
        else:
            currencyIDs = currencies
            baseCurrencyID = base
        results = Matrices.TimeSeriesMatrix(currencies, dateList)
        for i in range(len(currencyIDs)):
            if idlookup:
                self.log.debug('Calculating returns for currencies %s/%s', 
                                base, currencies[i])
            else:
                self.log.debug('Calculating returns for currency IDs %s/%s',
                                base, currencies[i])
            for j in range(len(dateList)):
                curr_rate = self.currencyCache.getRate(
                            dateList[j], currencyIDs[i], baseCurrencyID)
                if curr_rate:
                    results.data[i,j] = curr_rate
                else:
                    results.data[i,j] = ma.masked
        self.log.debug('loadExchangeRateHistory: end')
        return results

    def loadCurrencyReturnsHistory(self, rmgList, date, daysBack, 
                    currencies, base, dateList=None, idlookup=True, tweakForNYD=False, returnDF=False):
        """Returns a TimeSeriesMatrix of exchange rate returns
        for the currencies specified in currencyCodes, computed
        in the baseCurrencyID.  In other words, if base currency
        appreciates, the currency return is negative.
        If idlookup=True, the latter two arguments are assumed to 
        be ISO currency codes, not currency IDs.
        If dateList is not None, but rather a list of dates,
        we ignore date and daysBack, and compute returns for
        all dates in dateList
        """
        self.log.debug('loadCurrencyReturnsHistory: begin')
        if not self.currencyCache:
            raise ValueError('Currency conversion requires a ForexCache object')
        if dateList is not None:
            date = dateList[-1]
        else:
            dateList = self.getDates(rmgList, date, daysBack+1)
        if idlookup:
            currencyIDs = [self.currencyCache.getCurrencyID(c, date) for c in currencies]
            baseCurrencyID = self.currencyCache.getCurrencyID(base, date)
        else:
            currencyIDs = currencies
            baseCurrencyID = base
        results = Matrices.TimeSeriesMatrix(currencies, dateList[1:])
        for i in range(len(currencyIDs)):
            if idlookup:
                self.log.debug('Calculating returns for currencies %s/%s', 
                                base, currencies[i])
            else:
                self.log.debug('Calculating returns for currency IDs %s/%s', base, currencies[i])
            for j in range(1,len(dateList)):
                curr_rate = self.currencyCache.getRate(
                            dateList[j], currencyIDs[i], baseCurrencyID, tweakForNYD=tweakForNYD)
                prev_rate = self.currencyCache.getRate(
                            dateList[j-1], currencyIDs[i], baseCurrencyID, tweakForNYD=tweakForNYD)
                if curr_rate and prev_rate and prev_rate != 0.0:
                    results.data[i,j-1] = curr_rate / prev_rate - 1.0
                else:
                    results.data[i,j-1] = ma.masked
        self.log.debug('loadCurrencyReturnsHistory: end')
        if returnDF:
            return results.toDataFrame()
        return results

    def loadFactorReturnsHistory(self, rms_id, subFactors, dateList, flag=None, screen=False):
        """Returns the factor returns of the sub-factors in subFactors for the
        given days and risk model series ID.
        The return value is a TimeSeriesMatrix of factor returns.
        If a factor has no return recorded for a given day, the
        corresponding entry in the matrix is masked.
        """
        self.log.debug('loadFactorReturnsHistory: begin')
        if flag is None:
            tableName = 'rms_factor_return'
        else:
            tableName = 'rms_factor_return_%s' % flag
        logging.info('Loading factor returns from %s' % tableName)

        cache = self.factorReturnCache[(rms_id, tableName)]
        results = self.loadSubFactorData(dateList, subFactors, tableName, 'value', 'data.rms_id = %d' % rms_id, cache)
        if screen:
            results.data = Utilities.screen_data(results.data)
        self.log.debug('loadFactorReturnsHistory: end')
        return results

    def loadFactorVolatilityHistory(self, rms_id, subFactors, startDate=None, endDate=None):
        """Load the complete history of factor volatilities for the given
        rms_id and subFactors.  Returns a TimeSeriesMatrix
        """
        subIdxMap = dict((s.subFactorID,s) for s in subFactors)
        results = []
        if startDate is None:
            startDate = datetime.date(1900,1,1)
        if endDate is None:
            endDate = datetime.date.today()
        for subIdChunk in listChunkIterator(list(subIdxMap.keys()), 500):
            subIds = ','.join([str(id) for id in subIdChunk])
            query = """SELECT sub_factor1_id, dt, value
                       FROM rmi_covariance
                       WHERE rms_id = :rms_arg AND sub_factor1_id = sub_factor2_id
                       AND sub_factor1_id IN (%s) AND dt >= :start_date_arg
                       AND dt <= :end_date_arg""" % subIds
            self.dbCursor.execute(query, rms_arg=rms_id, start_date_arg=startDate, end_date_arg=endDate)
            results += self.dbCursor.fetchall()
            
        df = pandas.DataFrame(results, columns=['subfactor', 'dt', 'value']).pivot('dt', 'subfactor', 'value')
        df = numpy.sqrt(df.rename(columns=subIdxMap).reindex(columns=subFactors))
        return Matrices.TimeSeriesMatrix.fromDataFrame(df.T)

    def loadRawFactorReturnsHistory(self, rms_id, subFactors, dateList):
        """Returns the raw factor returns of the sub-factors in subFactors for the
        given days and risk model series ID.
        The return value is a TimeSeriesMatrix of raw factor returns.
        If a factor has no raw return recorded for a given day, the
        corresponding entry in the matrix is masked.
        """
        self.log.debug('loadRawFactorReturnsHistory: begin')
        cache = None
        results = self.loadSubFactorData(
            dateList, subFactors, 'rms_raw_factor_return', 'value',
            'data.rms_id = %d' % rms_id, cache)
        self.log.debug('loadRawFactorReturnsHistory: end')
        return results

    def loadStatFactorReturnsHistory(self, rms_id, subFactors, dateList):
        """Returns the factor returns of the sub-factors in subFactors for the
        given days and risk model series ID.
        The return value is a TimeSeriesMatrix of factor returns.
        If a factor has no return recorded for a given day, the
        corresponding entry in the matrix is masked.
        """
        dateList.sort()
        self.log.debug('loadStatFactorReturnsHistory: begin')
        d = dateList[-1]
        # No cache: we actually want to load a new history each day
        cache = None
        if rms_id < 0:
            table = 'rms_m%.2d_stat_factor_return' % abs(rms_id)
        else:
            table = 'rms_%s_stat_factor_return' % rms_id
        results = self.loadSubFactorData(
                dateList, subFactors, table, 'value',
                'data.rms_id=%d' % rms_id, cache, exp_dt=dateList[-1])
        self.log.debug('loadStatFactorReturnsHistory: end')
        return results
   
    def loadAMPIndustryReturnHistory(self, amp_id, revision_id, 
                                     startDate=None, endDate=None):
        """Load the complete history of AMP industry portfolio returns for the given
        amp_id and classification revision_id.  Returns a TimeSeriesMatrix
        """
        cur = self.dbCursor
        query = """
                SELECT id, name 
                FROM classification_ref 
                WHERE revision_id=:revision_id_arg AND
                      is_leaf='Y'
                """
        cur.execute(query, {'revision_id_arg': revision_id})
        industryIdNameMap = dict(cur.fetchall())

        results = []
        if startDate is None:
            startDate = datetime.date(1900,1,1)
        if endDate is None:
            endDate = datetime.date.today()
        query = """
           SELECT ref_id, dt, value
           FROM amp_industry_return
           WHERE 
              mdl_port_member_id=:mdl_port_member_id_arg AND
              revision_id=:revision_id_arg AND
              dt >= :start_date_arg AND
              dt <= :end_date_arg 
            ORDER BY dt
            """ 
        myDict = {'mdl_port_member_id_arg': amp_id, 'revision_id_arg': revision_id,
                'start_date_arg': startDate, 'end_date_arg': endDate}
        cur.execute(query, myDict)
        results += self.dbCursor.fetchall()
         
        df = pandas.DataFrame(results, columns=['ref_id', 'dt', 'value']).pivot('dt', 'ref_id', 'value')
        df.columns = [industryIdNameMap[col] for col in df.columns]
        return Matrices.TimeSeriesMatrix.fromDataFrame(df.T)

    def loadRMGMarketReturnHistory(self, dateList, rmgs, robust=False, useAMPs=True, returnDF=False, **kwargs):
        """Returns a time series of the market returns associated 
        with the specified risk model groups. dateList is assumed 
        to be in chronological order.
        The return value is a TimeSeriesMatrix of the table values.
        Missing entries are masked.
        """
        self.log.debug('loadRMGMarketReturnHistory: begin')
        if type(rmgs) is not list:
            rmgs = [rmgs]
        switchDate = Utilities.parseISODate('2020-01-26')
        dateList1 = list(dateList)
        dateList2 = []
        if useAMPs:
            dateList1 = [dt for dt in dateList if dt<switchDate]
            dateList2 = [dt for dt in dateList if dt>=switchDate]

        if len(dateList1) > 0:
            # Load legacy portfolio returns history
            if robust:
                field = 'rmg_market_return_v3'
                cache=self.marketReturnV3Cache
            else:
                field = 'rmg_market_return'
                cache=self.marketReturnCache
            results1 = self.loadRMGData(
                    dateList1, rmgs, field, 'value', withExtraKey=None, cache=cache)

            if len(dateList2) < 1:
                # If no AMP history to load, return legacy history
                if returnDF:
                    return results1.toDataFrame()
                return results1

        # Load AMP returns history
        ampList = self.convertLMP2AMP(rmgs, dateList2[-1])
        results2 = self.loadModelPortfolioReturns(dateList2, ampList)

        # Construct final object to be returned
        results = Matrices.TimeSeriesMatrix(rmgs, dateList)
        if len(dateList1) < 1:
            # If only AMP returns loaded, return those
            results.data = results2.data
        else:
            results.data = ma.concatenate((results1.data, results2.data), axis=1)

        self.log.debug('loadRMGMarketReturnHistory: end')
        if returnDF:
            return results.toDataFrame()
        return results

    def loadRMGMarketVolatilityHistory(self, dateList, rmgs, rollOver=0):
        """Returns a time series of the market volatility associated with
        the specified risk model groups. dateList is assumed to be in chronological order.
        The return value is a TimeSeriesMatrix of the table values.
        Missing entries are masked.
        """
        self.log.debug('loadRMGMarketVolatilityHistory: begin')
        field = 'rmg_market_volatility'
        cache=self.marketVolatilityCache

        if not isinstance(dateList, list):
            dateList = [dateList]
        numDates = len(dateList)
        if rollOver > 0:
            extraDateList = [dateList[0] - datetime.timedelta(i+1) for i in range(rollOver)]
            dateList.extend(extraDateList)
            dateList.sort()

        results = self.loadRMGData(dateList, rmgs, field, 'value',
            withExtraKey=None, cache=cache)
        results.data = Utilities.screen_data(results.data)

        if rollOver > 0:
            for idx in range(len(rmgs)):
                for dIdx in range(len(dateList)-1):
                    if results.data[idx, dIdx+1] is ma.masked:
                        results.data[idx, dIdx+1] = results.data[idx, dIdx]
            results.data = results.data[:, -numDates:]
            results.dates = results.dates[-numDates:]

        self.log.debug('loadRMGMarketVolatilityHistory: end')
        return results

    def loadRegionReturnHistory(self, dateList, regions):
        """Returns a time series of the region returns associated
        with the specified region ID. dateList is assumed
        to be in chronological order.
        The return value is a TimeSeriesMatrix of the table values.
        Missing entries are masked.
        """
        self.log.debug('loadRegionReturnHistory: begin')
        cache=self.regionReturnCache
        results = self.loadRegionData(
                dateList, regions, 'region_return', 'value',
                withExtraKey=None, cache=cache)
        self.log.debug('loadRegionReturnHistory: end')
        return results

    def loadRMSFactorStatisticHistory(self, rms_id, subFactors, dateList,
                                      fieldName):
        """Returns a time series of the factor statistic given by
        fieldName associated with the specified factors.
        dateList is assumed to be in chronological order.
        The return value is a TimeSeriesMatrix of the table values.
        Missing entries are masked.
        """
        self.log.debug('loadRMSFactorStatisticHistory: begin')
        results = self.loadSubFactorData(
            dateList, subFactors, 'rms_factor_statistics', fieldName,
            condition='rms_id=%d' % rms_id, cache=None)
        self.log.debug('loadRMSFactorStatisticHistory: end')
        return results
    
    def loadReturnsTimingAdjustmentsHistory(self, timingID, rmgs, inDateList,
                            loadProxy=False, loadAllDates=False, legacy=True):
        """Returns the market timing adjustments of the risk model groups
        in rmgs for the given days and risk model series ID.
        The return value is a TimeSeriesMatrix of market timing adjustments.
        If a risk model group has no adjustment recorded for a given day, the
        corresponding entry in the matrix is masked.
        """
        dateList = list(inDateList)
        reverseOrder = False
        if dateList[0] > dateList[-1]:
            reverseOrder = True
        dateList.sort()

        if legacy:
            tableName = 'returns_timing_adjustment'
        else:
            tableName = 'rmg_returns_timing_adj'

        self.log.debug('loadReturnsTimingAdjustmentsHistory: begin')
        if loadAllDates:
            allDates = self.getDateRange(None, dateList[0], dateList[-1])
        else:
            allDates = list(dateList)

        if loadProxy:
            if self.returnsTimingProxyCache is not None:
                cache = self.returnsTimingProxyCache[timingID]
            else:
                cache = None
            results = self.loadRMGData(
                    allDates, rmgs, tableName, 'proxy',
                    withExtraKey=('timing_id', timingID), cache=cache)
        else:
            if legacy:
                if self.returnsTimingAdjustmentCache is not None:
                    cache = self.returnsTimingAdjustmentCache[timingID]
                else:
                    cache = None
            else:
                if self.returnsTimingAdjustmentV2Cache is not None:
                    cache = self.returnsTimingAdjustmentV2Cache[timingID]
                else:
                    cache = None
            results = self.loadRMGData(
                allDates, rmgs, tableName, 'value',
                withExtraKey=('timing_id', timingID), cache=cache)

        results.data = Utilities.screen_data(results.data)
        if loadAllDates:
            results.data = ProcessReturns.compute_compound_returns_v3(
                    results.data, results.dates, dateList, keepFirst=True, matchDates=True)[0]

        if reverseOrder:
            results.dates.reverse()
            maskedData = ma.getmaskarray(results.data)
            results.data = numpy.fliplr(ma.filled(results.data, 0.0))
            results.data = ma.masked_where(numpy.fliplr(maskedData), results.data)

        self.log.debug('loadReturnsTimingAdjustmentsHistory: end')
        return results
    
    def getAverageTradingVolume(self, dates, subids, currencyID=None, loadAllDates=False):
        """Returns the average daily trading volume over the given dates
        of the sub-issues in the subids list.
        The result is a masked array of the ADVs.
        Volume is defined as currency amount traded, not share volume.
        """
        self.log.debug('getAverageTradingVolume: begin')
        if len(subids) < 1:
            return None
        volume = self.loadVolumeHistory(dates, subids, currencyID, loadAllDates=loadAllDates)
        self.log.debug('getAverageTradingVolume: end')
        return numpy.average(ma.filled(volume.data, 0.0), axis=1)
    
    def getAverageMarketCaps(self, dates, subids, currencyID=None,
            marketDB=None, loadAllDates=False, returnDF=False):
        """Returns the average market cap over the last daysBack days
        of the sub-issues in the subids list.
        The result is a masked array of the average market caps.
        The average for each asset is taken over the days where both
        price and shares outstanding are present. The entry of an asset
        is masked if the average could not be computed.
        Market caps are converted to the specified currency; if none
        is provided, the local currency of quotation is used.
        """
        self.log.debug('getAverageMarketCaps: begin')
        marketCap = self.loadMarketCapsHistory(dates, subids, currencyID, loadAllDates=loadAllDates)
        self.log.debug('getAverageMarketCaps: end')
        mcaps = ma.average(marketCap, axis=1)
        if returnDF:
            return pandas.Series(mcaps, index=subids)
        return mcaps
    
    def getVolumeWeightedAveragePrice(self, dates, subids, currencyID=None,
                                        loadAllDates=False):
        """Returns the volume-weighted average price over the given dates
        of the sub-issues in the subids list.
        The result is a masked array of the VWAPs.
        """
        self.log.debug('getVolumeWeightedAveragePrice: begin')
        volume = self.loadVolumeHistory(dates, subids, currencyID, loadAllDates=loadAllDates)
        prices = self.loadUCPHistory(dates, subids, currencyID)
        totalVolume = ma.sum(volume.data, axis=1)
        totalVolume = ma.masked_where(totalVolume==0.0, totalVolume)
        vwap = ma.sum(prices.data * volume.data, axis=1) / totalVolume
        self.log.debug('getVolumeWeightedAveragePrice: end')
        return vwap

    def loadCumulativeReturnsHistory(self, dateList, subids):
        """Returns the cumulative returns over the given days
        of the sub-issues in the subids list.
        The return value is a TimeSeriesMatrix of cumulative returns.
        If an assets has no cumulative return recorded for a given day, the
        corresponding entry in the matrix is masked.
        """
        self.log.debug('loadCumulativeReturnsHistory: begin')
        cret = self.loadSubIssueData(
            dateList, subids, 'sub_issue_cumulative_return', 'data.value',
             cache=self.cumReturnCache, convertTo=None, withCurrency=False)
        self.log.debug('loadCumulativeReturnsHistory: end')
        return cret
    
    def applyFreeFloatOnMCap(self,marketCaps,mcapDates, subIssues, marketDB, method='average'):
        '''
            apply free floating adjustment logics onto the marketCaps. it is currently used for liquitity and regression weight adjustment.
        :param marketCaps: 
        :param mcapDates: 
        :param subIssues: 
        :param marketDB: 
        :param method: average or lastest
        :return: marketCaps
        '''
        freefloat_df = self.loadFreeFloatHistory(mcapDates, subIssues, marketDB)
        freefloat_df2 = freefloat_df.toDataFrame().ffill(axis='columns').reindex(subIssues).fillna(1.0) # forward filling freefloating data within given days, otherwise fill with 1.0
        if method == 'average':
            freefloat_df3 = freefloat_df2.mean(axis=1)
        elif method == 'lastest':
            freefloat_df3 = freefloat_df2.iloc[:,-1]
        else:
            raise Exception('Unknown method in ModelDB.applyFreeFloatOnMCap:%s' % method)
        freefloat_ma = numpy.ndarray.flatten(freefloat_df3.values)
        marketCaps = marketCaps*freefloat_ma
        return marketCaps

    def loadFreeFloatHistory(self, dateList, ids, marketDB):
        """Returns the free float ratio in decimal over the given days
        of the sub-issuers in the subids list.
        The return value is a TimeSeriesmatrix of Free Float Pct.
        If an asset has no free float recorded for a given day, the
        corresponding entry in the matrix is masked.
        """
        self.log.debug('loadFreeFloatHistory: begin')
        self.loadMarketIdentifierHistoryCache(ids, marketDB, 'asset_dim_ff',
                                    'ff_value', cache=self.freeFloatCache)
        results = Matrices.TimeSeriesMatrix(ids, dateList)
        results.data = Matrices.allMasked(results.data.shape)
        dateIdx = dict([(date, dIdx) for (dIdx, date) in enumerate(dateList)])
        midIdx = dict([(mid, mIdx) for (mIdx, mid) in enumerate(ids)])

        tsmStartDate = min(dateList)
        tsmEndDate = max(dateList)

        for mid in ids:
            allValues = self.freeFloatCache.getAssetHistory(mid)
            if allValues is not None:
                # Filter active free float record
                activeValues = [[val.id, val.fromDt, val.thruDt] for val in allValues]
                activeValues = sorted(activeValues, key=lambda x:x[2], reverse=True)

                # Manually open up record, as vendor often does not start covering
                # assets as early as their IPO date
                if len(activeValues) > 0:
                    activeValues[-1][1] = datetime.date(1900,1,1)
                    activeValues[0][2] = datetime.date(2999,12,31)

                for (ffPct, fromDt, thruDt) in activeValues:
                    if fromDt > tsmEndDate:
                        continue
                    if thruDt <= tsmStartDate:
                        continue
                    dtIndicesToUpdate = [dateIdx[date] for date in dateList \
                                         if date >= fromDt and date < thruDt]
                    results.data[midIdx[mid], dtIndicesToUpdate] = float(ffPct) / 100.0

        results.data = Utilities.screen_data(results.data)
        results.data = ma.where(results.data > 1.0, 1.0, results.data)
        results.data = ma.where(results.data < 0.0, 0.0, results.data)
        self.log.debug('loadFreeFloatHistory: end')
        return results

    def loadMarketCapsHistory(self, dateList, subids, convertTo, loadAllDates=False):
        """Returns the market cap for the given  days of the
        sub-issues in the subids list.
        The result is an asset-by-time array of market caps.
        If an asset has no market cap recorded for a
        given day (or if it is non-positive), the corresponding entry
        in the matrix is masked.
        Market caps are converted to the specified currency; if none
        is provided, the local currency of quotation is used.
        """
        self.log.debug('loadMarketCapsHistory: begin')
        if loadAllDates:
            dateList.sort()
            allDates = self.getDateRange(None, dateList[0], dateList[-1])
            marketCap = self.loadSubIssueData(
                    allDates, subids, 'sub_issue_data', 'data.tso * data.ucp',
                    convertTo=convertTo, cache=self.marketCapCache)
        else:
            marketCap = self.loadSubIssueData(
                dateList, subids, 'sub_issue_data', 'data.tso * data.ucp',
                convertTo=convertTo, cache=self.marketCapCache)
        marketCap.data = Utilities.screen_data(marketCap.data)
        marketCap = ma.masked_where(marketCap.data <= 0.0, marketCap.data)
        self.log.debug('loadMarketCapsHistory: end')
        return marketCap
    
    def loadOpenInterestHistory(self, marketDB, dateList, subids):
        """Returns the open interest for the given  days of the
        (future) sub-issues in the subids list.
        """
        self.log.debug('loadOpenInterestHistory: begin')
        results = Matrices.TimeSeriesMatrix(subids, dateList)
        results.data = Matrices.allMasked(results.data.shape)
        issueList = list()
        for a in subids:
            if isinstance(a, SubIssue):
                issueList.append(a.getModelID())
            else:
                issueList.append(a)
        (axidStr, modelMarketMap) = self.getMarketDB_IDs(marketDB, issueList)
        mktList = [MarketID.MarketID(string=axid) for axid in axidStr]
        oi_values = marketDB.getFutureOpenInterest(dateList, mktList)
        aidMap = dict((axid, i) for (i, axid) in enumerate(axidStr))
        dtMap = dict((dt, i) for (i, dt) in enumerate(dateList))
        for (i, mid) in enumerate(issueList):
            axidHistory = sorted(modelMarketMap.get(mid, list()))
            for (fromDt, thruDt, axid) in axidHistory:
                for d in dateList:
                    results.data[i,dtMap[d]] = oi_values.data[aidMap[axid.getIDString()], dtMap[d]]
        return results 
    
    def loadRMGData(self, dateList, ids, tableName,
                    fieldExpression, withExtraKey=None, cache=None):
        """Returns a time series of the values selected by the field name
        for the risk model groups in ids on the dates specified in dateList.
        The dates in dateList are assumed to be in chronological order.
        The return value is a TimeSeriesMatrix of the table values.
        If a risk model group has no table entry recorded for a given day,
        the corresponding entry in the matrix is masked.
        """
        if cache == None:
            return self.loadRMGDataInternal(dateList, ids, tableName,
                                            fieldExpression, withExtraKey)
        if cache.maxDates < len(dateList):
            self.log.warning('cache uses %d days, requesting %d', cache.maxDates,
                          len(dateList))
        missingDates = cache.findMissingDates(dateList)
        missingIDs = cache.findMissingIDs(ids)
        
        if len(missingIDs) > 0:
            # Get data for missing IDs for existing dates
            missingIDData = self.loadRMGDataInternal(
                cache.getDateList(), missingIDs, tableName,
                fieldExpression, withExtraKey)
            cache.addMissingIDs(missingIDData)
        if len(missingDates) > 0:
            # Get data for all IDs for missing dates
            missingDateData = self.loadRMGDataInternal(
                missingDates, cache.getIDList(), tableName,
                fieldExpression, withExtraKey)
            cache.addMissingDates(missingDateData, None)
        # extract subset
        result = cache.getSubMatrix(dateList, ids)
        return result
    
    def loadRMGDataInternal(self, dateList, rmgs, tableName,
                            fieldExpression, withExtraKey):
        """Returns a time series of the values selected by the field name
        for the risk model groups in rmgs on the dates specified in dateList.
        The dates in dateList are assumed to be in chronological order.
        The return value is a TimeSeriesMatrix of the table values.
        If a risk model group has no table entry recorded for a given day,
        the corresponding entry in the matrix is masked.
        """
        dateLen = len(dateList)
        idLen = len(rmgs)
        self.log.debug('loading RMG data for %d days and %d groups',
                       dateLen, idLen)
        results = Matrices.TimeSeriesMatrix(rmgs, dateList)
        results.data = Matrices.allMasked(results.data.shape)
        if dateLen == 0 or idLen == 0:
            return results
        dateIdx = dict([(date, dIdx) for (dIdx, date) in enumerate(dateList)])
        rmgIdx = dict([(rmg.rmg_id, rIdx) for (rIdx, rmg) in enumerate(rmgs)])
        if dateLen >= 10:
            DINCR = 10
            RINCR = 100
        else:
            DINCR = 1
            RINCR = 200
        dateArgList = [('date%d' % i) for i in range(DINCR)]
        rmgArgList = [('rmg%d' % i) for i in range(RINCR)]
        defaultDict = dict([(i, None) for i in dateArgList + rmgArgList])
        query = 'SELECT rmg_id, dt, %(field)s' % {'field': fieldExpression}
        query += """
          FROM %(table)s data WHERE data.rmg_id IN (%(rmgs)s)
          AND dt IN (%(dates)s)""" % {
            'table': tableName,
            'rmgs': ','.join([(':%s' % i) for i in rmgArgList]),
            'dates': ','.join([(':%s' % i) for i in dateArgList])}
        if withExtraKey is not None:
            query += ' AND %s = :extra_arg' % withExtraKey[0]
            defaultDict['extra_arg'] = withExtraKey[1]
        rmgIDs = [rmg.rmg_id for rmg in rmgs]
        itemsTotal = len(rmgIDs) * len(dateList)
        itemsSoFar = 0
        lastReport = 0
        startTime = datetime.datetime.now()
        for dateChunk in listChunkIterator(dateList, DINCR):
            myDateDict = dict(zip(dateArgList, dateChunk))
            for rmgChunk in listChunkIterator(rmgIDs, RINCR):
                myRMGDict = dict(zip(rmgArgList, rmgChunk))
                valueDict = dict(defaultDict)
                valueDict.update(myDateDict)
                valueDict.update(myRMGDict)
                self.dbCursor.execute(query, valueDict)
                valueMap = dict()
                for row in self.dbCursor.fetchall():
                    if row[2] is None:
                        continue
                    dt = row[1].date()
                    rIdx = rmgIdx[row[0]]
                    dIdx = dateIdx[dt]
                    value = float(row[2])
                    valueMap[(rIdx, dIdx)] = value
                if len(valueMap) > 0:
                    rmgIndices, dtIndices = zip(*valueMap.keys())
                    values = list(valueMap.values())
                    results.data[rmgIndices, dtIndices] = values
                itemsSoFar += len(dateChunk) * len(rmgChunk)
                if itemsSoFar - lastReport > 0.05 * itemsTotal:
                    lastReport = itemsSoFar
                    logging.debug('%d/%d, %g%%, %s', itemsSoFar, itemsTotal,
                                  100.0 * float(itemsSoFar) / itemsTotal,
                                  datetime.datetime.now() - startTime)
        return results

    def loadRegionData(self, dateList, ids, tableName,
                    fieldExpression, withExtraKey=None, cache=None):
        """Returns a time series of the values selected by the field name
        for the risk model region ids on the dates specified in dateList.
        The dates in dateList are assumed to be in chronological order.
        The return value is a TimeSeriesMatrix of the table values.
        If a region has no table entry recorded for a given day,
        the corresponding entry in the matrix is masked.
        """
        if cache == None:
            return self.loadRegionDataInternal(dateList, ids, tableName,
                                            fieldExpression, withExtraKey)
        if cache.maxDates < len(dateList):
            self.log.warning('cache uses %d days, requesting %d', cache.maxDates,
                          len(dateList))
        missingDates = cache.findMissingDates(dateList)
        missingIDs = cache.findMissingIDs(ids)

        if len(missingIDs) > 0:
            # Get data for missing IDs for existing dates
            missingIDData = self.loadRegionDataInternal(
                cache.getDateList(), missingIDs, tableName,
                fieldExpression, withExtraKey)
            cache.addMissingIDs(missingIDData)
        if len(missingDates) > 0:
            # Get data for all IDs for missing dates
            missingDateData = self.loadRegionDataInternal(
                missingDates, cache.getIDList(), tableName,
                fieldExpression, withExtraKey)
            cache.addMissingDates(missingDateData, None)
        # extract subset
        result = cache.getSubMatrix(dateList, ids)
        return result

    def loadRegionDataInternal(self, dateList, regs, tableName,
                            fieldExpression, withExtraKey):
        """Returns a time series of the values selected by the field name
        for the risk model regions on the dates specified in dateList.
        The dates in dateList are assumed to be in chronological order.
        The return value is a TimeSeriesMatrix of the table values.
        If a risk model group has no table entry recorded for a given day,
        the corresponding entry in the matrix is masked.
        """
        dateLen = len(dateList)
        idLen = len(regs)
        self.log.debug('loading Region data for %d days and %d groups',
                       dateLen, idLen)
        results = Matrices.TimeSeriesMatrix(regs, dateList)
        results.data = Matrices.allMasked(results.data.shape)
        if dateLen == 0 or idLen == 0:
            return results
        dateIdx = dict([(date, dIdx) for (dIdx, date) in enumerate(dateList)])
        regIdx = dict([(reg.region_id, rIdx) for (rIdx, reg) in enumerate(regs)])
        if dateLen >= 10:
            DINCR = 10
            RINCR = 100
        else:
            DINCR = 1
            RINCR = 200
        dateArgList = [('date%d' % i) for i in range(DINCR)]
        regArgList = [('reg%d' % i) for i in range(RINCR)]
        defaultDict = dict([(i, None) for i in dateArgList + regArgList])
        query = 'SELECT id, dt, %(field)s' % {'field': fieldExpression}
        query += """
          FROM %(table)s data WHERE data.id IN (%(regs)s)
          AND dt IN (%(dates)s)""" % {
            'table': tableName,
            'regs': ','.join([(':%s' % i) for i in regArgList]),
            'dates': ','.join([(':%s' % i) for i in dateArgList])}
        if withExtraKey is not None:
            query += ' AND %s = :extra_arg' % withExtraKey[0]
            defaultDict['extra_arg'] = withExtraKey[1]
        regIDs = [reg.region_id for reg in regs]
        itemsTotal = len(regIDs) * len(dateList)
        itemsSoFar = 0
        lastReport = 0
        startTime = datetime.datetime.now()
        for dateChunk in listChunkIterator(dateList, DINCR):
            myDateDict = dict(zip(dateArgList, dateChunk))
            for regChunk in listChunkIterator(regIDs, RINCR):
                myRMGDict = dict(zip(regArgList, regChunk))
                valueDict = dict(defaultDict)
                valueDict.update(myDateDict)
                valueDict.update(myRMGDict)
                self.dbCursor.execute(query, valueDict)
                valueMap = dict()
                for row in self.dbCursor.fetchall():
                    if row[2] is None:
                        continue
                    dt = row[1].date()
                    rIdx = regIdx[row[0]]
                    dIdx = dateIdx[dt]
                    value = float(row[2])
                    valueMap[(rIdx, dIdx)] = value
                if len(valueMap) > 0:
                    regIndices, dtIndices = zip(*valueMap.keys())
                    values = list(valueMap.values())
                    results.data[regIndices, dtIndices] = values
                itemsSoFar += len(dateChunk) * len(regChunk)
                if itemsSoFar - lastReport > 0.05 * itemsTotal:
                    lastReport = itemsSoFar
                    logging.debug('%d/%d, %g%%, %s', itemsSoFar, itemsTotal,
                                  100.0 * float(itemsSoFar) / itemsTotal,
                                  datetime.datetime.now() - startTime)
        return results

    def loadSubFactorData(self, dateList, ids, tableName,
                          fieldExpression, condition, cache=None, exp_dt=None):
        """Returns a time series of the values selected by the field name
        for the sub-factors in ids on the dates specified in dateList.
        The dates in dateList are assumed to be in chronological order.
        The return value is a TimeSeriesMatrix of the table values.
        If a sub-factor has no table entry recorded for a given day,
        the corresponding entry in the matrix is masked.
        """
        if cache == None:
            return self.loadSubFactorDataInternal(dateList, ids, tableName,
                                                  fieldExpression, condition, exp_dt)
        if cache.maxDates < len(dateList):
            self.log.warning('cache uses %d days, requesting %d', cache.maxDates,
                          len(dateList))
        missingDates = cache.findMissingDates(dateList)
        missingIDs = cache.findMissingIDs(ids)
        
        if len(missingIDs) > 0:
            # Get data for missing IDs for existing dates
            missingIDData = self.loadSubFactorDataInternal(
                cache.getDateList(), missingIDs, tableName,
                fieldExpression, condition, exp_dt)
            cache.addMissingIDs(missingIDData)
        if len(missingDates) > 0:
            # Get data for all IDs for missing dates
            missingDateData = self.loadSubFactorDataInternal(
                missingDates, cache.getIDList(), tableName,
                fieldExpression, condition, exp_dt)
            cache.addMissingDates(missingDateData, None)
        # extract subset
        result = cache.getSubMatrix(dateList, ids)
        return result
    
    def loadSubFactorDataInternal(self, dateList, ids, tableName,
                                  fieldExpression, condition, exp_dt):
        """Returns a time series of the values selected by the field name
        for the sub-factors in ids on the dates specified in dateList.
        The dates in dateList are assumed to be in chronological order.
        The return value is a TimeSeriesMatrix of the table values.
        If a sub-factor has no table entry recorded for a given day,
        the corresponding entry in the matrix is masked.
        """
        dateLen = len(dateList)
        idLen = len(ids)
        self.log.debug('loading sub-factor data for %d days and %d factors',
                       dateLen, idLen)
        results = Matrices.TimeSeriesMatrix(ids, dateList)
        if dateLen == 0 or idLen == 0:
            return results
        
        results.data = Matrices.allMasked(results.data.shape)
        
        dateIdx = dict([(date, dIdx) for (dIdx, date) in enumerate(dateList)])
        sfIdxMap = dict([(sf.subFactorID, sfIdx) for (sfIdx, sf)
                         in enumerate(ids)])
        if dateLen >= 10:
            DINCR = 10
            SINCR = 100
        else:
            DINCR = 1
            SINCR = 100
        dateArgList = [('date%d' % i) for i in range(DINCR)]
        sfArgList = [('sf%d' % i) for i in range(SINCR)]
        defaultDict = dict([(i, None) for i in dateArgList + sfArgList])
        query = """SELECT sub_factor_id, dt, %(field)s
          FROM %(table)s data WHERE data.sub_factor_id IN (%(sfs)s)
          AND dt IN (%(dates)s) AND %(condition)s""" % {
            'table': tableName,
            'field': fieldExpression,
            'condition': condition,
            'sfs': ','.join([(':%s' % i) for i in sfArgList]),
            'dates': ','.join([(':%s' % i) for i in dateArgList])}
        if exp_dt is not None:
            query += ' AND data.exp_dt=:dt_arg'
            defaultDict['dt_arg'] = exp_dt
    
        sfIDs = [sf.subFactorID for sf in ids]
        itemsTotal = len(sfIDs) * len(dateList)
        itemsSoFar = 0
        lastReport = 0
        startTime = datetime.datetime.now()
        for sfChunk in listChunkIterator(sfIDs, SINCR):
            mySFDict = dict(zip(sfArgList, sfChunk))
            for dateChunk in listChunkIterator(dateList, DINCR):
                myDateDict = dict(zip(dateArgList, dateChunk))
                valueDict = dict(defaultDict)
                valueDict.update(myDateDict)
                valueDict.update(mySFDict)
                self.dbCursor.execute(query, valueDict)
                r = self.dbCursor.fetchall()
                valueMap = dict()
                for row in r:
                    if row[2] is None:
                        continue
                    dt = row[1].date()
                    sfIdx = sfIdxMap[row[0]]
                    dIdx = dateIdx[dt]
                    value = float(row[2])
                    valueMap[(sfIdx, dIdx)] = value
                if len(valueMap) > 0:
                    sfIndices, dtIndices = zip(*valueMap.keys())
                    values = list(valueMap.values())
                    results.data[sfIndices, dtIndices] = values
                itemsSoFar += len(dateChunk) * len(sfChunk)
                if itemsSoFar - lastReport > 0.05 * itemsTotal:
                    lastReport = itemsSoFar
                    logging.debug('%d/%d, %g%%, %s', itemsSoFar, itemsTotal,
                                  100.0 * float(itemsSoFar) / itemsTotal,
                                  datetime.datetime.now() - startTime)
        return results
    
    def loadTradingSuspensionHistory(self, dateList, ids, marketDB):
        """Returns a TimeSeriesmatrix containing 1.0 on dates for which
        the corresponding asset was suspended from trading.
        If an asset was not suspended, whether or not it actually traded, 
        as long as the given date was a valid exchange trading day, the 
        corresponding entry in the matrix is masked.
        """
        self.log.debug('loadTradingSuspensionHistory: begin')
        self.loadMarketIdentifierHistoryCache(ids, marketDB, 'asset_dim_trad_status',
                                    'value', cache=self.tradingStatusCache)
        results = Matrices.TimeSeriesMatrix(ids, dateList)
        results.data = Matrices.allMasked(results.data.shape)
        dateIdx = dict([(date, dIdx) for (dIdx, date) in enumerate(dateList)])
        midIdx = dict([(mid, mIdx) for (mIdx, mid) in enumerate(ids)])

        for mid in ids:
            allValues = self.tradingStatusCache.getAssetHistory(mid)
            # Ignore 'A' (active) and 'D' (delisted)
            for val in allValues:
                if val.id == 'S':
                    suspendedDatesIdx = [dateIdx[d] for d in dateList \
                                    if d >= val.fromDt and d < val.thruDt]
                    results.data[midIdx[mid], suspendedDatesIdx] = 1.0

        self.log.debug('loadTradingSuspensionHistory: end')
        return results

    def loadStockConnectEligibility(self, dateList, ids, marketDB):
        """Returns a TimeSeriesmatrix containing 1.0 on dates for which
        the corresponding asset was eligible for both buying and selling
        on the Shanghai/Shenzhen Hong Kong Stock Connect.
        If an asset was not eligible, the corresponding entry in the matrix is masked.
        """
        self.log.debug('loadStockConnectEligibility: begin')
        self.loadMarketIdentifierHistoryCache(ids, marketDB, 'asset_dim_con_all',
                                    'value', cache=self.stockConnectBuySellCache)
        results = Matrices.TimeSeriesMatrix(ids, dateList)
        results.data = Matrices.allMasked(results.data.shape)
        dateIdx = dict([(date, dIdx) for (dIdx, date) in enumerate(dateList)])
        midIdx = dict([(mid, mIdx) for (mIdx, mid) in enumerate(ids)])

        for mid in ids:
            allValues = self.stockConnectBuySellCache.getAssetHistory(mid)
            for val in allValues:
                if val.id == 'Y':
                    eligibleDatesIdx = [dateIdx[d] for d in dateList \
                                    if d >= val.fromDt and d < val.thruDt]
                    results.data[midIdx[mid], eligibleDatesIdx] = 1.0

        self.log.debug('loadStockConnectEligibility: end')
        return results

    def loadStockConnectSellOnly(self, dateList, ids, marketDB):
        """Returns a TimeSeriesmatrix containing 1.0 on dates for which
        the corresponding asset was designated 'Sell Only' on the 
        Shanghai/Shenzhen Hong Kong Stock Connect.
        If an asset was anything else, the corresponding entry in the matrix is masked.
        """
        self.log.debug('loadStockConnectSellOnly: begin')
        self.loadMarketIdentifierHistoryCache(ids, marketDB, 'asset_dim_con_sell',
                                    'value', cache=self.stockConnectSellOnlyCache)
        results = Matrices.TimeSeriesMatrix(ids, dateList)
        results.data = Matrices.allMasked(results.data.shape)
        dateIdx = dict([(date, dIdx) for (dIdx, date) in enumerate(dateList)])
        midIdx = dict([(mid, mIdx) for (mIdx, mid) in enumerate(ids)])

        for mid in ids:
            allValues = self.stockConnectSellOnlyCache.getAssetHistory(mid)
            for val in allValues:
                if val.id == 'Y':
                    eligibleDatesIdx = [dateIdx[d] for d in dateList \
                                    if d >= val.fromDt and d < val.thruDt]
                    results.data[midIdx[mid], eligibleDatesIdx] = 1.0

        self.log.debug('loadStockConnectSellOnly: end')
        return results

    def loadNotTradedInd(self, dateList, subids):
        """Special case of loadSubIssueData. Returns a TimeSeriesMatrix that
        indicates whether the return to an asset on a specific date results 
        from trading activity or whether it is a rolled over return.
        0 indicates a traded return, 1 indicates a rolled over return.
        """
        notTradedInd = self.loadSubIssueData(dateList, subids,
                'sub_issue_data', 'price_marker, tdv',
                cache=self.notTradedIndCache, withRevDate=True,
                convertTo=None, withCurrency=False,
                withRMS=None, withNotTradedInd=True)
        return notTradedInd
    
    def loadSubIssueData(self, dateList, subids, tableName,
                         fieldExpression, cache=None,
                         withRevDate=True, convertTo=None,
                         withCurrency=True, withRMS=None,
                         withNotTradedInd=False, condition=None, returnDF=False):
        """Returns a time series of the values selected by the field name
        for the sub-issues in subids on the dates specified in dateList.
        The dates in dateList are assumed to be in chronological order.
        The return value is a TimeSeriesMatrix of the table values.
        If an asset has no table entry recorded for a given day,
        the corresponding entry in the matrix is masked.
        withTradingInd gives a TimeSeriesMatrix that indicates if a return
        was the result of trades or a rollover.
        """
        # If currency conversions required
        if convertTo is not None:
            if not self.currencyCache:
                raise ValueError('Currency conversion requires a ForexCache object present')
        
        if cache == None:
            result = self.loadSubIssueDataInternal(
                dateList, subids, tableName, fieldExpression,
                withRevDate, withCurrency, withRMS, withNotTradedInd, condition)
        else:
            if cache.maxDates < len(dateList):
                self.log.fatal('cache uses %d days, requesting %d',
                               cache.maxDates, len(dateList))
                raise ValueError('cache too small')
            if convertTo is not None and cache.maxDates > self.currencyCache.currencyProvider.maxNumConverters:
                self.log.fatal('item cache uses %d days but FX cache only holds %d days',
                               cache.maxDates, self.currencyCache.currencyProvider.maxNumConverters)
                raise ValueError('FX cache too small')
            missingDates = cache.findMissingDates(dateList)
            missingIDs = cache.findMissingIDs(subids)
            
            if len(missingIDs) > 0:
                # Get data for missing Assets for existing dates
                missingIDData = self.loadSubIssueDataInternal(
                    cache.getDateList(), missingIDs, tableName,
                    fieldExpression, withRevDate, withCurrency, withRMS, 
                    withNotTradedInd, condition)
                cache.addMissingIDs(missingIDData)
            if len(missingDates) > 0:
                # Get data for all assets for missing dates
                missingDateData = self.loadSubIssueDataInternal(
                    missingDates, cache.getIDList(), tableName,
                    fieldExpression, withRevDate, withCurrency, withRMS,
                    withNotTradedInd, condition)
                cache.addMissingDates(missingDateData, self.currencyCache)
            # extract subset
            result = cache.getSubMatrix(dateList, subids)
        self.log.debug('received data')
        # convert result to desired currency
        if withCurrency:
            if convertTo is not None:
                ccyConvRates = cache.getCurrencyConversions(
                    dateList, subids, self.currencyCache, convertTo)
                result.data *= ccyConvRates
            self.log.debug('converted data')

        if returnDF:
            return result.toDataFrame()
        return result
    
    def loadSubIssueDataInternal(self, dateList, subids, tableName,
                                 fieldExpression, withRevDate,
                                 withCurrency, withRMS,
                                 withNotTradedInd, condition):
        """Returns a time series of the values selected by the field name
        for the sub-issues in subids on the dates specified in dateList.
        The dates in dateList are assumed to be in chronological order.
        The return value is a TimeSeriesMatrix of the table values.
        If an asset has no table entry recorded for a given day,
        the corresponding entry in the matrix is masked.
        """
        dateLen = len(dateList)
        idLen = len(subids)
        self.log.debug('loading sub-issue data from %s for %d days and %d assets',
                       tableName, dateLen, idLen)
        results = Matrices.TimeSeriesMatrix(subids, dateList)
        results.data = Matrices.allMasked(results.data.shape)
        
        if withCurrency:
            results.ccy = Matrices.allMasked(results.data.shape,
                                              dtype=int)
        if dateLen == 0 or idLen == 0:
            return results
        dateIdx = dict([(date, dIdx) for (dIdx, date) in enumerate(dateList)])
        sidIdx = dict([(sid.getSubIDString(), sIdx) for (sIdx, sid)
                         in enumerate(subids)])
        if dateLen >= 10:
            DINCR = 10
            SINCR = 100
        else:
            DINCR = 1
            SINCR = 500
        dateArgList = [('date%d' % i) for i in range(DINCR)]
        sidArgList = [('sid%d' % i) for i in range(SINCR)]
        defaultDict = dict([(i, None) for i in dateArgList + sidArgList])
        if tableName.lower()=='rmg_historic_beta_v3':
            query='SELECT /*+ Index(data,PK_RMG_HISTORIC_BETA_V3) */ sub_issue_id, dt, %(field)s' % {'field': fieldExpression}
        else:
            query = 'SELECT sub_issue_id, dt, %(field)s' % {'field': fieldExpression}

        if withCurrency:
            query += ', data.currency_id'
        if withRevDate:
            query += ', rev_del_flag'
        if withNotTradedInd: #withNotTradedInd should never be called at the same time as withCurrency
            assert(not withCurrency)
            pmIdx = 2
            tdvIdx = 3
        if withRevDate:
            query +=', rev_dt' 
        query += """
          FROM %(table)s data WHERE data.sub_issue_id IN (%(sids)s)
          AND dt IN (%(dates)s)""" % {
            'table': tableName,
            'sids': ','.join([(':%s' % i) for i in sidArgList]),
            'dates': ','.join([(':%s' % i) for i in dateArgList])}
        if withRMS is not None:
            query += ' AND rms_id=:rms'
            defaultDict['rms'] = withRMS
        if withRevDate:
            #query += ' AND rev_dt <= :revDt ORDER BY rev_dt ASC'
            #query += " AND rev_dt <= :revDt AND rev_dt >= to_date('1900-01-01','YYYY-MM-DD')"
            query += " AND rev_dt <= :revDt "
            defaultDict['revDt'] = self.revDateTime
            if withCurrency or withNotTradedInd:
                revIdx = 4
            else:
                revIdx = 3
        if condition is not None:
            query += ' AND %s' % condition
          
        sidStrings = sorted([sid.getSubIDString() for sid in subids])
        itemsTotal = len(sidStrings) * len(dateList)
        itemsSoFar = 0
        lastReport = 0
        startTime = datetime.datetime.now()
        if tableName.lower()=='rmg_historic_beta_v3':
            logging.debug('Rewrote the query as %s', query)
        for sidChunk in listChunkIterator(sidStrings, SINCR):
            myAxidDict = dict(zip(sidArgList, sidChunk))
            for dateChunk in listChunkIterator(dateList, DINCR):
                myDateDict = dict(zip(dateArgList, dateChunk))
                valueDict = dict(defaultDict)
                valueDict.update(myDateDict)
                valueDict.update(myAxidDict)
                self.dbCursor.execute(query, valueDict)
                res = self.dbCursor.fetchall()
                if withRevDate:
                    r=(rr[:-1] for rr in sorted(res, key=lambda rrr: rrr[-1])) 
                    if withCurrency:
                        self._loadSIDCurrencyRevDate(r, revIdx, sidIdx,
                                                     dateIdx, results)
                    elif withNotTradedInd:
                        self._loadNotTradedIndRevDate(r, pmIdx, tdvIdx, revIdx, sidIdx, dateIdx, results)
                    else:
                        self._loadSIDRevDate(r, revIdx, sidIdx,
                                             dateIdx, results)
                else:
                    r= res
                    if withCurrency:
                        self._loadSIDCurrency(r, sidIdx, dateIdx, results)
                    elif withNotTradedInd:
                        self._loadNotTradedInd(r, pmIdx, tdvIdx, sidIdx, dateIdx, results)
                    else:
                        self._loadSID(r, sidIdx, dateIdx, results)
                itemsSoFar += len(dateChunk) * len(sidChunk)
                if itemsSoFar - lastReport > 0.05 * itemsTotal:
                    lastReport = itemsSoFar
                    logging.debug('%d/%d, %g%%, %s', itemsSoFar, itemsTotal,
                                  100.0 * float(itemsSoFar) / itemsTotal,
                                  datetime.datetime.now() - startTime)
        return results

    def loadFFAMarketCapsHistoryInternal(
            self, marketDB, dateList, universe_subID,
            convertTo=None, backFill=False):
        """
            Internal helper class to retrieve FFA data
            *** Note: this function is retired since it reads data from vendor database. we leave it for now for reference.
        """
        dateLen = len(dateList)
        idLen = len(universe_subID)
        self.log.debug('loading FFA Market Cap data for %d days and %d assets',
                       dateLen, idLen)
        results = Matrices.TimeSeriesMatrix(universe_subID, dateList)
        results.data = Matrices.allMasked(results.data.shape)
        results.ccy = Matrices.allMasked(results.data.shape,
                                         dtype=int)
        if dateLen == 0 or idLen == 0:
            return results

        # get history of sedols corresponding to subids
        subIDSedolHistDict = self.loadMarketIdentifierHistory(universe_subID, 
                                                              marketDB,'asset_dim_sedol','id', 
                                                              cache = self.sedolCache)
    
        # get sedols for every subID on each date in dateList (subID->sedol 1)grows 2) potentially changes over time)
        universeHistDict = {}
        universeAllSedols = set()
        for d in dateList:
            universe = Utilities.Struct()
            universe.sedolSubIDDict = {}
            for mid in universe_subID:
                sdlStruct = self.sedolCache.getAssetValue(mid, d)
                if sdlStruct is not None:
                    sdl = sdlStruct.id
                    universe.sedolSubIDDict[sdl] = mid
                    universeAllSedols.add(sdl)
            universeHistDict[d]= universe
        
        universeAllSedolsList = [i[:-1] for i in universeAllSedols]
        universeAllSedols = universeAllSedolsList
        
        if len(universeAllSedols) == 0:
            #No sedol found, ignore. 
            return results
        
        from marketdb import QADirect
        tqaDB = QADirect.QADirect(user='tqa_user', passwd = 'tqa_user', host='tqastby.axiomainc.com', database='qai')

        INCR = 400
        argList = [('code%d' % i) for i in range(INCR)]
        
        # a.FreeFloatAltPct, a.FreeFloatPct 
        query = """select c.sedol, a.ValDate, a.FreeFloatPct 
        from DS2ShareHldgs a, DS2CtryQtInfo b, DS2SEDOLChg c 
        where c.sedol in (%(args)s) and a.InfoCode = b.InfoCode and 
        c.InfoCode = b.infocode and a.FreeFloatPct is not null
        order by c.sedol, ValDate""" % {
            'args': ','.join(['%%(%s)s' % arg for arg in argList])}
        defaultDict = dict([(arg, '') for arg in argList])
        queryresults = list()

        for codeChunk in listChunkIterator(universeAllSedols, INCR):
            myDict = dict(defaultDict)
            myDict.update(dict(zip(argList, codeChunk)))
            tqaDB.dbCursor.execute(query, myDict)
            queryresults.extend(tqaDB.dbCursor.fetchall())

        # organise data by sedol
        ffaData = []
        for i in universeAllSedols:
            ffaDataTmp = []
            ffaDataTmp.append([j for j in queryresults if j[0] == i])
            if len(ffaDataTmp[0])>1:
                ffaData.append(ffaDataTmp)
        
        # put FFA data into a dict where sedol is the key        
        ffaSedolDict = {}
        for i in ffaData:
            sedolKey = i[0][0][0]
            if len(sedolKey) == 6:
                from marketdb import Utilities as Util
                sedolKey = str(sedolKey) + str(Util.computeSEDOLCheckDigit(sedolKey))
            assert(len(sedolKey) == 7)
            ffaDate = [j[1].date() for j in i[0]]
            ffaPercent = [j[2] for j in i[0]]
            ffaValue = list(zip(ffaDate, ffaPercent))
            ffaSedolDict.update([(sedolKey,ffaValue)])        
        
        # put FFA data into dict where sedol is key and list of from_dt,thru_dt,ffa_pct are values
        BOT = datetime.date(1950, 1, 1)        
        EOT = datetime.date(2999, 12, 31) 
        ffaSedolPctDict = {}
        for i in ffaSedolDict.keys():
            vals = ffaSedolDict[i]
            ffa_vals = []
            pcts = [k[1] for k in vals]
            dates = [j[0] for j in vals]
            if len(vals) == 1:
                #if there is only one time, look at the backfill option
                ffa = Utilities.Struct()
                if backFill:
                    ffa.fromDt = BOT
                else:
                    ffa.fromDt = dates[0]
                ffa.thruDt = EOT
                if pcts[0]:
                    ffa.pct = pcts[0]/100
                else:
                    ffa.pct = 1
                ffa_vals.append(ffa)
            else:
                for dIdx, d in enumerate(dates):
                    ffa = Utilities.Struct()
                    if pcts[dIdx]:
                        ffa.pct = pcts[dIdx]/100
                    else:
                        ffa.pct = 1
                    if dIdx == 0:
                        if backFill:
                            ffa.fromDt = BOT
                            ffa.thruDt = dates[dIdx + 1]
                        else:
                            ffa.fromDt = dates[dIdx]
                            ffa.thruDt = dates[dIdx + 1]
                    elif dIdx == len(dates) - 1:
                        ffa.fromDt = dates[dIdx] 
                        ffa.thruDt = EOT
                    else:
                        ffa.fromDt = dates[dIdx]
                        ffa.thruDt = dates[dIdx + 1]
                    ffa_vals.append(ffa)
            ffaSedolPctDict.update([(i,ffa_vals)])
        
        # assign ffa pct of 1 to sedols with no ds data
        for i in set(universeAllSedols).difference(set(ffaSedolDict.keys())):
            ffa = Utilities.Struct()
            ffa.fromDt = BOT
            ffa.thruDt = EOT
            ffa.pct = 1
            ffaSedolPctDict.update([(i,[ffa])])
        
        # create dictionary where sub issue id is key and list of from_dt,thru_dt,ffa_pct are values
        #ffaSubIDPctDict = dict([(universe.sedolSubIDDict[i],ffaSedolPctDict[i]) for i in ffaSedolPctDict.keys()])
                
        # create dictionary where sub issue id is key and list of from_dt,thru_dt,ffa_pct are values, by date
        ffaSubIDPctDictHist = {}
        for d in dateList:
            ffaSubIDPctDict = {}
            for i in ffaSedolPctDict.keys():
                try:
                    key = universeHistDict[d].sedolSubIDDict[i]
                    value = ffaSedolPctDict[i]
                    ffaSubIDPctDict.update([(key,value)])
                except KeyError:
                    continue
            missingKeys = set(universe_subID).difference(list(ffaSubIDPctDict.keys()))
            for j in missingKeys:
                ffa = Utilities.Struct()
                ffa.fromDt = BOT
                ffa.thruDt = EOT
                ffa.pct = 1
                ffaSubIDPctDict[j] = [ffa]
            ffaSubIDPctDictHist.update([(d,ffaSubIDPctDict)])    
        
        #set up cache objects, by date
        ffaDataHistDict = {}
        for d in dateList: 
            ffaData = FromThruCache(useModelID = False)
            for i in ffaSubIDPctDictHist[d].keys():
                ffaData.addAssetValues(i.string,ffaSubIDPctDictHist[d][i])
            ffaDataHistDict.update([(d,ffaData)])        
        
        # start of code borrowed from loadSubIssueDataInternal     
        dateLen = len(dateList)
        idLen = len(universe_subID)
        
        if dateLen == 0 or idLen == 0:
            return results
        
        dateIdx = dict([(date, dIdx) for (dIdx, date) in enumerate(dateList)])
        sidIdx = dict([(sid.getSubIDString(), sIdx) for (sIdx, sid) in enumerate(universe_subID)])
        
        if dateLen >= 10:
            DINCR = 10
            SINCR = 100
        else:
            DINCR = 1
            SINCR = 500
        
        dateArgList = [('date%d' % i) for i in range(DINCR)]
        sidArgList = [('sid%d' % i) for i in range(SINCR)]
        defaultDict = dict([(i, None) for i in dateArgList + sidArgList])
        query = 'SELECT sub_issue_id, dt, data.tso * data.ucp' 
        
        if convertTo:
            query += ', data.currency_id'
        
        query += """
          FROM sub_issue_data data WHERE data.sub_issue_id IN (%(sids)s)
          AND dt IN (%(dates)s)""" % {
            'sids': ','.join([(':%s' % i) for i in sidArgList]),
            'dates': ','.join([(':%s' % i) for i in dateArgList])}
              
        sidStrings = sorted([sid.getSubIDString() for sid in universe_subID])
        for sidChunk in listChunkIterator(sidStrings, SINCR):
            myAxidDict = dict(zip(sidArgList, sidChunk))
            for dateChunk in listChunkIterator(dateList, DINCR):
                myDateDict = dict(zip(dateArgList, dateChunk))
                valueDict = dict(defaultDict)
                valueDict.update(myDateDict)
                valueDict.update(myAxidDict)
                self.dbCursor.execute(query, valueDict)
                r = self.dbCursor.fetchall()
                if convertTo:
                    self._loadFFAMCapCurrency(r, sidIdx, dateIdx, results, ffaDataHistDict)
                else:
                    self._loadFFASID(r, sidIdx, dateIdx, results, ffaDataHistDict)
        return results

    def loadFFAMarketCapsHistory(self, marketDB, dateList, subids,
                                 cache=None, convertTo=None, withCurrency=True, backFill=False):
        """Returns the free float adjusted market cap for the given  days of the
        sub-issues in the subids list.
        The result is an asset-by-time array of market caps.
        If an asset has no market cap recorded for a
        given day (or if it is non-positive), the corresponding entry
        in the matrix is masked.
        Market caps are converted to the specified currency; if none
        is provided, the local currency of quotation is used.
        Currently not set up to run with cache.
        
        Things to improve: 
        1) Set up cache
        2) Don't load FFA data each day
        *** Note: this function is retired since it reads data from vendor database. we leave it for now for reference. 
        """  
        
        self.log.debug('loadFFAMarketCapsHistory: begin')
        
        # If currency conversions required
        if convertTo is not None:
            if not self.currencyCache:
                raise ValueError('Currency conversion requires a ForexCache object present')
        
        # fixed universe to be considered
        universe_subID = subids
        if cache == None:
            results = self.loadFFAMarketCapsHistoryInternal(marketDB, dateList, universe_subID, convertTo, backFill)
        else:
            if cache.maxDates < len(dateList):
                self.log.fatal('cache uses %d days, requesting %d',
                               cache.maxDates, len(dateList))
                raise ValueError('cache too small')
            missingDates = cache.findMissingDates(dateList)
            missingIDs = cache.findMissingIDs(subids)
            if len(missingIDs) > 0:
                # Get data for missing Assets for existing dates
                missingIDData = self.loadFFAMarketCapsHistoryInternal(
                    marketDB, cache.getDateList(), missingIDs, convertTo, backFill)
                cache.addMissingIDs(missingIDData)
            if len(missingDates) > 0:
                # Get data for all assets for missing dates
                missingDateData = self.loadFFAMarketCapsHistoryInternal(
                    marketDB, missingDates, cache.getIDList(), convertTo, backFill)
                
                cache.addMissingDates(missingDateData, self.currencyCache)
            # extract subset
            results = cache.getSubMatrix(dateList, subids)
        self.log.debug('received data')
                
        # convert result to desired currency
        if withCurrency and convertTo is not None:
            ccyConvRates = cache.getCurrencyConversions(
                dateList, subids, self.currencyCache, convertTo)
            results.data *= ccyConvRates
            self.log.debug('converted data')
        results.data = ma.masked_where(results.data <= 0.0, results.data)
        return results

    def _loadFFAMCapCurrency(self, r, sidIdx, dateIdx, results, ffaDataCacheDict):
        valueMap = dict()
        for row in r:
            if row[2] is None:
                continue
            dt = row[1].date()
            sIdx = sidIdx[row[0]] 
            dIdx = dateIdx[dt] 
            value = float(row[2])
            ffaPct = ffaDataCacheDict[dt].getAssetValue(row[0],dt)
            if ffaPct is not None:
                ffaPct = ffaPct.pct
            else:
                ffaPct = 1 
            value = value*ffaPct
            ccyId = row[3]
            valueMap[(sIdx, dIdx)] = (value, ccyId)
        if len(valueMap) > 0:
            sidIndices, dtIndices = zip(*valueMap.keys())
            values, currencies = zip(*valueMap.values())
            results.data[sidIndices, dtIndices] = values
            results.ccy[sidIndices, dtIndices] = currencies
    
    def _loadFFASID(self, r, sidIdx, dateIdx, results, ffaDataCacheDict):
        valueMap = dict()
        for row in r:
            if row[2] is None:
                continue
            dt = row[1].date()
            sIdx = sidIdx[row[0]]
            dIdx = dateIdx[dt]
            value = float(row[2])
            ffaPct = ffaDataCacheDict[dt].getAssetValue(row[0],dt).pct
            value = value*ffaPct
            valueMap[(sIdx, dIdx)] = value
        if len(valueMap) > 0:
            sidIndices, dtIndices = zip(*valueMap.keys())
            values = list(valueMap.values())
            results.data[sidIndices, dtIndices] = values
    
    def _loadSIDCurrencyRevDate(self, r, revIdx, sidIdx, dateIdx, results):
        valueMap = dict()
        for row in r:
            dt = row[1].date()
            sIdx = sidIdx[row[0]]
            dIdx = dateIdx[dt]
            revDelFlag = row[revIdx]
            if row[2] is None or revDelFlag == 'Y':
                if (sIdx, dIdx) in valueMap:
                    del valueMap[(sIdx, dIdx)]
                continue
            value = float(row[2])
            ccyId = row[3]
            valueMap[(sIdx, dIdx)] = (value, ccyId)
        if len(valueMap) > 0:
            sidIndices, dtIndices = zip(*valueMap.keys())
            values, currencies = zip(*valueMap.values())
            results.data[sidIndices, dtIndices] = values
            results.ccy[sidIndices, dtIndices] = currencies
        
    def _loadSIDCurrency(self, r, sidIdx, dateIdx, results):
        valueMap = dict()
        for row in r:
            if row[2] is None:
                continue
            dt = row[1].date()
            sIdx = sidIdx[row[0]]
            dIdx = dateIdx[dt]
            value = float(row[2])
            ccyId = row[3]
            valueMap[(sIdx, dIdx)] = (value, ccyId)
        if len(valueMap) > 0:
            sidIndices, dtIndices = zip(*valueMap.keys())
            values, currencies = zip(*valueMap.values())
            results.data[sidIndices, dtIndices] = values
            results.ccy[sidIndices, dtIndices] = currencies
        
    def _loadSIDRevDate(self, r, revIdx, sidIdx, dateIdx, results):
        valueMap = dict()
        for row in r:
            dt = row[1].date()
            sIdx = sidIdx[row[0]]
            dIdx = dateIdx[dt]
            revDelFlag = row[revIdx]
            if row[2] is None or revDelFlag == 'Y':
                if (sIdx, dIdx) in valueMap:
                    del valueMap[(sIdx, dIdx)]
                continue
            value = float(row[2])
            valueMap[(sIdx, dIdx)] = value
        if len(valueMap) > 0:
            sidIndices, dtIndices = zip(*valueMap.keys())
            values = list(valueMap.values())
            results.data[sidIndices, dtIndices] = values
        
    def _loadSID(self, r, sidIdx, dateIdx, results):
        valueMap = dict()
        for row in r:
            if row[2] is None:
                continue
            dt = row[1].date()
            sIdx = sidIdx[row[0]]
            dIdx = dateIdx[dt]
            value = float(row[2])
            valueMap[(sIdx, dIdx)] = value
        if len(valueMap) > 0:
            sidIndices, dtIndices = zip(*valueMap.keys())
            values = list(valueMap.values())
            results.data[sidIndices, dtIndices] = values
            
    def _loadNotTradedInd(self, r, pmIdx, tdvIdx, sidIdx, dateIdx, results):
        indMap = dict()
        for row in r:
            dt = row[1].date()
            sIdx = sidIdx[row[0]]
            dIdx = dateIdx[dt]
            pm = float(row[pmIdx])
            try:
                tdv = float(row[tdvIdx])
            except TypeError:
                tdv = float(0)
            # traded/not traded logic: price_marker = 0 or 1 indicate traded,
            # price marker 3 indicates rollover, other markers are ambiguous
            # so check for positive trading volume
            if pm in (0,1,3): 
                if pm in (0,1):
                    ind = 0 # 0 indicates traded
                else:
                    ind = 1 # 1 indicates non traded
            else:
                if tdv > 0:
                    ind = 0
                else:
                    ind = 1 
            indMap[(sIdx, dIdx)] = ind
        if len(indMap) > 0:
            sidIndices, dtIndices = zip(*indMap.keys())
            indicators = list(indMap.values())
            results.data[sidIndices, dtIndices] = indicators

    def _loadNotTradedIndRevDate(self, r, pmIdx, tdvIdx, revIdx, sidIdx, dateIdx, results):
        indMap = dict()
        for row in r:
            dt = row[1].date()
            sIdx = sidIdx[row[0]]
            dIdx = dateIdx[dt]
            revDelFlag = row[revIdx]
            if revDelFlag == 'Y':
                if (sIdx, dIdx) in indMap:
                    del indMap[(sIdx, dIdx)]
                continue
            pm = float(row[pmIdx])
            try:
                tdv = float(row[tdvIdx])
            except TypeError:
                tdv = float(0)
            # traded/not traded logic: price_marker = 0 or 1 indicate traded,
            # price marker 3 indicates rollover, other markers are ambiguous
            # so check for positive trading volume
            if pm in (0,1,3): 
                if pm in (0,1):
                    ind = 0 # 0 indicates traded
                else:
                    ind = 1 # 1 indicates non traded
            else:
                if tdv > 0:
                    ind = 0
                else:
                    ind = 1 
            indMap[(sIdx, dIdx)] = ind
        if len(indMap) > 0:
            sidIndices, dtIndices = zip(*indMap.keys())
            indicators = list(indMap.values())
            results.data[sidIndices, dtIndices] = indicators
        
    def loadSpecificReturnsHistory(self, rms_id, subids, dateList, internal=False):
        """Returns the specific returns of the sub-issues in subids for the
        given days and risk model series ID.
        The return value is a TimeSeriesMatrix of specific returns.
        If an assets has no return recorded for a given day, the
        corresponding entry in the matrix is masked.
        """
        self.log.debug('loadSpecificReturnsHistory: begin')
        if internal:
            tableName = 'rms_specific_return_internal'
        else:
            tableName = 'rms_specific_return'
        logging.info('Loading specific returns from %s', tableName)
        if self.specReturnCache is not None:
            cache = self.specReturnCache[rms_id]
        else:
            cache = None
        results = self.loadSubIssueData(
            dateList, subids, tableName, 'value',
            cache=cache, withRevDate=False, withRMS=rms_id,
            withCurrency=False)
        self.log.debug('loadSpecificReturnsHistory: end')
        return results

    def loadRobustWeights(self, rms_id, subids, dateList, reg_id=0):
        """Loads the robust regression weights for
        the given subissues and date
        Only weights that differ from one are non-missing
        """
        self.log.debug('loadRobustWeights: begin')
        if type(dateList) is not list:
            dateList = [dateList]
        cond = 'reg_id=%d' % reg_id
        results = self.loadSubIssueData(
                dateList, subids, 'rms_robust_weight', 'value',
                cache=self.robustWeightCache, withRevDate=False, withRMS=rms_id,
                withCurrency=False, condition=cond)
        self.log.debug('loadRobustWeights: end')
        return results

    def loadFMPs(self, rms_id, subids, date, sub_factor_id):
        """Loads the FMPs for the given model, sub-factor ID,
        sub-issues and date
        """
        self.log.debug('loadFMPs: begin')
        if type(date) is not list:
            date = [date]
        if rms_id < 0:
            table = 'rms_m%.2d_fmp' % abs(rms_id)
        else:
            table = 'rms_%s_fmp' % rms_id
        results = self.loadSubIssueData(
                date, subids, table, 'sf_%s' % sub_factor_id,
                cache=self.fmpCache, withRevDate=False,
                withCurrency=False)
        self.log.debug('loadFMPs: end')
        return results

    def loadProxyReturnsHistory(self, subids, dateList):
        """Returns the proxy returns of the sub-issues in subids for the
        given days
        The return value is a TimeSeriesMatrix of pseudo returns.
        If an assets has no return recorded for a given day, the
        corresponding entry in the matrix is masked.
        """
        self.log.debug('loadProxyReturnsHistory: begin')
        results = self.loadSubIssueData(
            dateList, subids, 'rmg_proxy_return', 'value',
            cache=self.proxyReturnCache, withRevDate=False,
            withCurrency=False)
        self.log.debug('loadProxyReturnsHistory: end')
        return results

    def loadISCScoreHistory(self, subids, dateList):
        """Returns the ISC scores of the sub-issues in subids for the
        given days
        The return value is a TimeSeriesMatrix of asset scores
        If an assets has no value recorded for a given day, the
        corresponding entry in the matrix is masked.
        """
        self.log.debug('loadISCScoreHistory: begin')
        results = self.loadSubIssueData(
            dateList, subids, 'rmg_isc_score', 'value',
            cache=self.ISCScoreCache, withRevDate=False,
            withCurrency=False)
        self.log.debug('loadISCScoreHistory: end')
        return results
 
    def loadSpecificRiskHistory(self, rms_id, subids, dateList):
        """Returns the specific risks of the sub-issues in subids for the
        given days and risk model series ID.
        The return value is a TimeSeriesMatrix.
        """
        self.log.debug('loadSpecificRiskHistory: begin')
        results = self.loadSubIssueData(
            dateList, subids, 'rmi_specific_risk', 'value',
            withRevDate=False, withRMS=rms_id, withCurrency=False)
        self.log.debug('loadSpecificRiskHistory: end')
        return results
    
    def loadESTUQualifyHistory(self, rms_id, subids, dateList, estuInstance=None, returnDF=False):
        """Returns a TimeSeriesMatrix whose entries are
        1.0 if an asset was deemed eligible (qualified) for 
        ESTU membership on that date, or masked otherwise.
        """
        self.log.debug('loadESTUQualifyHistory: begin')

        # Try first the new estu structure
        if estuInstance is None:
            estu_id = 1
        else:
            estu_id = estuInstance.id
        if self.estuQualifyCache is not None:
            cache = self.estuQualifyCache[(rms_id, estu_id)]
        else:
            cache = None

        # Load data from new table
        results = self.loadSubIssueData(
                dateList, subids, 'rmi_estu_v3', 'qualify',
                cache=cache, withRevDate=False, withRMS=rms_id,
                withCurrency=False, condition='NESTED_ID=%d' % estu_id, returnDF=returnDF)

        # Check whether we have actually loaded anything
        if not returnDF:
            nonMissingData = numpy.flatnonzero(ma.getmaskarray(results.data)==0)
            if len(nonMissingData) < 1:
                logging.info('No estu data in rmi_estu_v3, trying older table')
                if self.estuQualifyCache is not None:
                    cache = self.estuQualifyCache[rms_id]
                else:
                    cache = None
                # Load data from old table
                results = self.loadSubIssueData(
                        dateList, subids, 'rmi_estu', 'qualify',
                        cache=cache, withRevDate=False, withRMS=rms_id, withCurrency=False)
        self.log.debug('loadESTUQualifyHistory: end')
        return results
    
    def loadIssueFromDates(self, dates, subids):
        self.log.debug('loadIssueFromDates: begin')
        cache = self.issueFromDateCache
        self.loadFromThruTableCache(
                'sub_issue', dates, subids, ['issue_id'],
                keyName='sub_id', cache=cache)
        EOT = datetime.date(2999, 12, 31)
        fromDates = [EOT] * len(subids)
        for idx, subid in enumerate(subids):
            for dt in reversed(dates):
                value = cache.getAssetValue(subid, dt)
                if value is not None:
                    fromDates[idx] = value.fromDt
                    break
        self.log.debug('loadIssueFromDates: end')
        return fromDates
    
    def loadIssueFromThruDates(self, dates, subids):
        self.log.debug('loadIssueFromThruDates: begin')
        cache = self.issueFromDateCache
        self.loadFromThruTableCache(
                'sub_issue', dates, subids, ['issue_id'],
                keyName='sub_id', cache=cache)
        EOT = datetime.date(2999, 12, 31)
        fromThruDates = dict([(subid, (EOT, EOT)) for subid in subids])
        
        for subid in subids:
            for dt in reversed(dates):
                value = cache.getAssetValue(subid, dt)
                if value is not None:
                    fromThruDates[subid] = (value.fromDt, value.thruDt)
                    break
        self.log.debug('loadIssueFromThruDates: end')
        return fromThruDates
     
    def loadFromThruTable(self, tableName, dates, keys, fields,
                             cache=None, keyName='issue_id'):
        """Return an array of Structs containing the table data.
        Rows correspond to dates, columns to IDs."""
        if cache == None:
            cache = FromThruCache()
        self.loadFromThruTableCache(tableName, dates, keys, fields, cache, keyName)
        retvals = numpy.empty((len(dates), len(keys)), dtype=object)
        for (aIdx, key) in enumerate(keys):
            for (dIdx, date) in enumerate(dates):
                retvals[dIdx, aIdx] = cache.getAssetValue(key, date)
        return retvals
    
    def loadFromThruTableCache(self, tableName, dates, keys, fields,
                             cache, keyName='issue_id'):
        """Populate cache containing the table data.
        """
        if len(fields) > 0:
            fieldList = ',%s' % ','.join(fields)
        else:
            fieldList = ''
        INCR = 200
        keyList = ['key%d' % i for i in range(INCR)]
        query = """SELECT %(key)s, from_dt, thru_dt
           %(fields)s FROM %(table)s t1
           WHERE t1.%(key)s IN (%(keys)s)""" % {
            'table': tableName, 'fields': fieldList, 'key': keyName,
            'keys': ','.join([':%s' % i for i in keyList])}
        retvals = dict()
        defaultDict = dict([(a, None) for a in keyList])
        missingIds = sorted(cache.getMissingIds(set(keys)))
        keyChangeListMap = dict([(key, list()) for key in missingIds])
        if keyName == 'issue_id' or keyName == 'modeldb_id':
            missingIds = [i.getIDString() for i in missingIds]
        if keyName == 'sub_id' or keyName == 'sub_issue_id':
            missingIds = [i.getSubIDString() for i in missingIds]
        for idChunk in listChunkIterator(missingIds, INCR):
            valueDict = dict(defaultDict)
            valueDict.update(dict(zip(keyList, idChunk)))
            self.dbCursor.execute(query, valueDict)
            r = self.dbCursor.fetchmany()
            while len(r) > 0:
                for val in r:
                    if keyName == 'issue_id' or keyName == 'modeldb_id':
                        keyId = ModelID.ModelID(string=val[0])
                    elif keyName == 'sub_issue_id' or keyName == 'sub_id':
                        keyId = SubIssue(string=val[0])
                    else:
                        keyId = val[0]
                    fromDt = val[1].date()
                    thruDt = val[2].date()
                    rval = Utilities.Struct()
                    rval.fromDt = fromDt
                    rval.thruDt = thruDt
                    for (name, fval) in zip(fields, val[3:]):
                        rval.setField(name, fval)
                    keyChangeListMap[keyId].append(rval)
                r = self.dbCursor.fetchmany()
        for (keyId, changeList) in keyChangeListMap.items():
            cache.addAssetValues(keyId, changeList)

    def loadRawUCPHistory(self, dateList, subIds):
        """Returns an array of price, currency, price_marker Structs for
        the assets in ids on the dates specified in dateList.
        The return value is a numpy.array of Structs.
        If an asset has no table entry recorded for a given day then
        the corresponding entry in the matrix is None.
        """
        dateLen = len(dateList)
        sidLen = len(subIds)
        self.log.debug('loading raw UCP time-series data for %d days'
                       ' and %d assets', dateLen, sidLen)
        results = numpy.empty((sidLen, dateLen), dtype=object)
        if dateLen == 0 or sidLen == 0:
            return results
        
        dateIdx = dict([(date, dIdx) for (dIdx, date) in enumerate(dateList)])
        sidIdx = dict([(sid.getSubIDString(), sIdx) for (sIdx, sid)
                         in enumerate(subIds)])
        DINCR = 10
        SINCR = 500
        dateArgList = [('date%d' % i) for i in range(DINCR)]
        sidArgList = [('sid%d' % i) for i in range(SINCR)]
        
        query = """SELECT sub_issue_id, dt, ucp, currency_id, price_marker
        FROM sub_issue_data data
        WHERE data.sub_issue_id IN (%(sids)s) AND data.dt IN (%(dates)s)
        AND data.rev_del_flag='N' AND data.rev_dt=(SELECT MAX(rev_dt)
           FROM sub_issue_data data2 WHERE data.sub_issue_id=data2.sub_issue_id
           AND data.dt=data2.dt AND data2.rev_dt <= :revDt)""" % {
            'sids': ','.join([':%s' % i for i in sidArgList]),
            'dates': ','.join([':%s' % i for i in dateArgList])}
        defaultDict = dict([(i, None) for i in dateArgList]
                           + [(i, None) for i in sidArgList])
        defaultDict['revDt'] = self.revDateTime
        sidStrings = sorted([sid.getSubIDString() for sid in subIds])
        for dateChunk in listChunkIterator(dateList, DINCR):
            myDateDict = dict(zip(dateArgList, dateChunk))
            for sidChunk in listChunkIterator(sidStrings, SINCR):
                mySidDict = dict(zip(sidArgList, sidChunk))
                valueDict = dict(defaultDict)
                valueDict.update(myDateDict)
                valueDict.update(mySidDict)
                self.log.debug('loading raw UCP time-series data query: %s values %s', query, valueDict)
                self.dbCursor.execute(query, valueDict)
                for (sid, dt, ucp, curr, priceMarker) \
                        in self.dbCursor.fetchall():
                    rval = Utilities.Struct()
                    rval.ucp = ucp
                    rval.currency_id = curr
                    rval.price_marker = priceMarker
                    dt = dt.date()
                    results[sidIdx[sid], dateIdx[dt]] = rval
        return results
    
    def loadTotalReturnsHistory(self, rmg, date, subids, daysBack,
            assetConvMap=None, baseCurrencyID=1, maskArray=None,
            compoundWeekend=False, notTradedFlag=False):
        """Returns the total daily returns of the sub-issues in subids for the
        last daysBack+1 trading days going back from date.
        The return value is a TimeSeriesMatrix of total daily returns.
        If an assets has no return recorded for a given day, the
        corresponding entry in the matrix is masked.
        The tradingFlag indicates whether or not the return of the
        asset comes from trading or a rollover. 
        If converting all returns to a single base currency, assetConvMap
        should be specified as the currency ID of the reference currency.
        Alternatively, different assets can to be converted to different
        currencies if assetConvMap is given as a mapping from SubIssues to 
        the desired currency ID.
        """
        self.log.debug('loadTotalReturnsHistory: begin')
        dateList = self.getDates(rmg, date, daysBack, forceLatestDate=True)
        results = self.loadSubIssueData(dateList, subids,
                                            'sub_issue_return', 'tr',
                                            cache=self.totalReturnCache,
                                            withCurrency=False)
        if notTradedFlag:
            results.notTradedFlag = self.loadNotTradedInd(dateList, subids).data
        else:
            results.notTradedFlag = Matrices.allMasked(results.data.shape)
        results.assetIdxMap = dict([(j,i) for (i,j) in enumerate(subids)])
        if maskArray is not None:
            results.data = ma.filled(results.data, 0.0)
            results.data = ma.masked_where(maskArray, results.data)
        if len(dateList) < 1:
            return results
        if assetConvMap is not None:
            if not self.currencyCache:
                raise ValueError('Currency conversion requires a ForexCache object present')
            # Determine what type of conversion is required
            if isinstance(assetConvMap, dict):
                sidList = list(assetConvMap.keys())
                assetSelectiveConversion = True
            else:
                sidList = subids
                assetSelectiveConversion = False
                baseCurrencyID = assetConvMap 

            # Get currency of quotation for assets
            data = self.loadMarketIdentifierHistory(
                    sidList, self.currencyCache.currencyProvider.marketDB, 'asset_dim_trading_currency',
                    'id', cache=self.tradeCcyCache)
            tccy = [(sid, self.tradeCcyCache.getAssetValue(sid, date))
                    for sid in sidList]
            assetCurrMap = dict([(i, j.id) for (i,j) in tccy if j is not None])

            # Check for changes in trading currency during this period
            currencyChangedIds = dict()
            cids = set(assetCurrMap.values())
            if assetSelectiveConversion:
                cids.update(list(assetConvMap.values()))
            for (sid, history) in data.items():
                assetHistory = [h for h in history if \
                        h.fromDt <= dateList[-1] and h.thruDt > dateList[0]]
                periodCurrencies = set([h.id for h in assetHistory])
                if len(periodCurrencies) > 1:
                    currencyChangedIds[sid] = assetHistory
                    cids.update(periodCurrencies)
            self.log.debug('%d assets with currency changes',
                        len(list(currencyChangedIds.keys())))
            
            # Compute currency returns
            cids.add(baseCurrencyID)
            cids = list(cids)
            currencyReturns = self.loadCurrencyReturnsHistory(
                    rmg, date, daysBack, cids, baseCurrencyID, idlookup=False)
            currencyGrossReturns = currencyReturns.data.filled(0.0) + 1.0
            currencyIdxMap = dict(zip(cids, range(len(cids))))

            # Some trading currency from/thru dates may be on non-trading days...
            dateIdxMap = dict([(d,i) for (i,d) in enumerate(dateList)])
            dt = dateList[-1]
            idx = None
            while dt > dateList[0]:
                if dt not in dateIdxMap:
                    dateIdxMap[dt] = idx
                idx = dateIdxMap[dt]
                dt -= datetime.timedelta(1)

            # Convert time series one asset at a time
            for (i, sid) in enumerate(sidList):
                if assetSelectiveConversion:
                    pos = results.assetIdxMap[sid]
                    convertTo = assetConvMap[sid]
                else:
                    pos = i
                    convertTo = baseCurrencyID
                currencyChanges = currencyChangedIds.get(sid, None)
                if currencyChanges is not None:
                    for n in currencyChanges:
                        cid = n.id
#                        self.log.debug('Converted %s from %s to %s from %d to %d',
#                                sid.getModelID().getIDString(), dateList[start],
#                                dateList[min(end, len(dateList)-1)], cid, currencyID)
                        if cid != convertTo:
                            start = dateIdxMap.get(n.fromDt, 0)
                            end = dateIdxMap.get(n.thruDt, len(dateList)+1)
                            ret0 = (results.data[pos,start:end] + 1.0) * \
                                  (currencyGrossReturns[currencyIdxMap[cid],start:end])
                            if len(ret0) == 0:
                                continue
                            if assetSelectiveConversion:
                                ret1 = currencyGrossReturns[currencyIdxMap[convertTo],start:end]
                                ret0 /= ret1
                            results.data[pos,start:end] = ret0 - 1.0
                else:
                    cid = assetCurrMap.get(sid, baseCurrencyID)
                    if cid != convertTo:
                        ret0 = (results.data[pos,:] + 1.0) * \
                              (currencyGrossReturns[currencyIdxMap[cid],:])
                        if assetSelectiveConversion:
                            ret1 = currencyGrossReturns[currencyIdxMap[convertTo],:]
                            ret0 /= ret1
                        results.data[pos,:] = ret0 - 1.0
         
        # Deal with weekend trading properly
        if compoundWeekend:
            lastFewDays = [dateList[0]]
            dtIdxMap = dict([(j,i) for (i,j) in enumerate(dateList)])
            for (t,dt) in enumerate(dateList[1:]):
                lastFewDays.append(dt)
                # Push weekend returns into the next trading day
                if lastFewDays[-1].weekday() < lastFewDays[-2].weekday():
                    dateIds = [dtIdxMap[d] for d in lastFewDays if d.weekday() > 4 or d==dt]
                    if len(dateIds) > 1:
                        subsetReturns = ma.filled(ma.take(results.data, dateIds, axis=1), 0.0)
                        subsetReturns = numpy.cumproduct(subsetReturns + 1.0, axis=1)[:,-1] - 1.0
                        for j in range(results.data.shape[0]):
                            if subsetReturns[j] != 0.0:
                                results.data[j,t+1] = subsetReturns[j]
                                results.notTradedFlag[j,t+1] = 0
                if len(lastFewDays) > 5:
                    lastFewDays.pop(0)
            # Now zero all weekend returns
            wkndIdxList = [dtIdxMap[d] for d in dateList if d.weekday() > 4]
            for t in wkndIdxList:
                results.data[:,t] = 0.0
                results.notTradedFlag[:,t] = 1
            
        self.log.debug('loadTotalReturnsHistory: end')
        return results
    
    def getCurrencyConversionMatrix(self, assetConvMap, dateList,
            rmg, subids, assetIdxMap, tol=1.0e-6):
        """ Does the complex and messy conversion of returns from
        one currency to another
        """

        date = dateList[0]
        if not self.currencyCache:
            raise ValueError('Currency conversion requires a ForexCache object present')
        # Determine what type of conversion is required
        if isinstance(assetConvMap, dict):
            sidList = list(assetConvMap.keys())
            assetSelectiveConversion = True
            baseCurrencyID = 1
        else:
            sidList = subids
            assetSelectiveConversion = False
            baseCurrencyID = assetConvMap

        # Get currency of quotation for assets
        data = self.loadMarketIdentifierHistory(
                sidList, self.currencyCache.currencyProvider.marketDB, 'asset_dim_trading_currency',
                'id', cache=self.tradeCcyCache)
        tccy = [(sid, self.tradeCcyCache.getAssetValue(sid, date))
                for sid in sidList]
        assetCurrMap = dict([(i, j.id) for (i,j) in tccy if j is not None])

        # Check for changes in trading currency during this period
        currencyChangedIds = dict()
        cids = set(assetCurrMap.values())
        if assetSelectiveConversion:
            cids.update(list(assetConvMap.values()))
        for (sid, history) in data.items():
            assetHistory = [h for h in history if \
                    h.fromDt <= dateList[-1] and h.thruDt > dateList[0]]
            periodCurrencies = set([h.id for h in assetHistory])
            if len(periodCurrencies) > 1:
                currencyChangedIds[sid] = assetHistory
                cids.update(periodCurrencies)
        self.log.debug('%d assets with currency changes',
                    len(list(currencyChangedIds.keys())))

        # Compute currency returns
        cids.add(baseCurrencyID)
        cids = list(cids)
        # Need to extend list of currency dates by one, as the first will
        # be lopped off within the currency retrieval function
        curDateList = [dateList[0] - datetime.timedelta(1)]
        curDateList.extend(dateList)
        currencyReturns = self.loadCurrencyReturnsHistory(
                rmg, 0, 0, cids, baseCurrencyID, dateList=curDateList,
                idlookup=False, tweakForNYD=True)
        currencyReturns.data = Utilities.screen_data(currencyReturns.data)
        currencyGrossReturns = ma.filled(currencyReturns.data, 0.0) + 1.0
        currencyIdxMap = dict(zip(cids, range(len(cids))))
        currencyMatrix = Matrices.allMasked((len(subids), len(dateList)))

        # Some trading currency from/thru dates may be on non-trading days...
        dateIdxMap = dict([(d,i) for (i,d) in enumerate(dateList)])
        dt = dateList[-1]
        idx = None
        while dt > dateList[0]:
            if dt not in dateIdxMap:
                dateIdxMap[dt] = idx
            idx = dateIdxMap[dt]
            dt -= datetime.timedelta(1)

        # Convert time series one asset at a time
        for (i, sid) in enumerate(sidList):
            if assetSelectiveConversion:
                pos = assetIdxMap[sid]
                convertTo = assetConvMap[sid]
            else:
                pos = i
                convertTo = baseCurrencyID
            currencyChanges = currencyChangedIds.get(sid, None)
            if currencyChanges is not None:
                for n in currencyChanges:
                    cid = n.id
                    if cid != convertTo:
                        start = dateIdxMap.get(n.fromDt, 0)
                        end = dateIdxMap.get(n.thruDt, len(dateList)+1)
                        cur0 = currencyGrossReturns[currencyIdxMap[cid],start:end]
                        if assetSelectiveConversion:
                            cur0 = cur0 / currencyGrossReturns[currencyIdxMap[convertTo],start:end]
                        currencyMatrix[pos,start:end] = cur0

            else:
                cid = assetCurrMap.get(sid, baseCurrencyID)
                if cid != convertTo:
                    cur0 = currencyGrossReturns[currencyIdxMap[cid],:]
                    if assetSelectiveConversion:
                        cur0 = cur0 / currencyGrossReturns[currencyIdxMap[convertTo],:]
                    currencyMatrix[pos,:] = cur0
        return currencyMatrix

    def loadTotalReturnsHistoryV3(self, rmg, date, subids, daysBack, assetConvMap=None,
            excludeWeekend=True, allRMGDates=False, dateList=None, compound=True,
            boT=datetime.date(1980,1,1)):
        """Returns the total daily returns of the sub-issues in subids for the
        last daysBack trading days going back from date.
        The return value is a TimeSeriesMatrix of total daily returns.
        If an assets has no return recorded for a given day, the
        corresponding entry in the matrix is masked.
        The tradingFlag indicates whether or not the return of the
        asset comes from trading or a rollover.
        If converting all returns to a single base currency, assetConvMap
        should be specified as the currency ID of the reference currency.
        Alternatively, different assets can to be converted to different
        currencies if assetConvMap is given as a mapping from SubIssues to
        the desired currency ID.
        """
        self.log.debug('loadTotalReturnsHistoryV3: begin')
        debug=False
        # Load date range for returns
        if dateList == None:
            if allRMGDates:
                dateList = self.getDateRange(self.getAllRiskModelGroups(),
                        None, date, tradingDaysBack=daysBack, excludeWeekend=excludeWeekend)
            else:
                dateList = self.getDateRange(rmg, None, date, tradingDaysBack=daysBack, excludeWeekend=excludeWeekend)
            dateList = sorted(dt for dt in dateList if dt>=boT)

        if len(dateList) < 1:
            logging.info('Empty date list, exiting returns loading')
            results = Utilities.Struct()
            results.assets = subids
            results.dates = []
            results.data = None
            results.notTradedFlag = None
            results.preIPOFlag = None
            results.originalData = None
            results.currencyMatrix  = None
            return results

        # Load all returns for every single day in the date range
        allDateList = self.getDateRange(None, dateList[0], dateList[-1])

        if not compound:
            dateList = allDateList
        allDateIdxMap = dict([(d,i) for (i,d) in enumerate(allDateList)])
        dateListIdx = [allDateIdxMap[d] for d in dateList]
        results = self.loadSubIssueData(allDateList, subids,
                'sub_issue_return', 'tr', cache=self.totalReturnCache,
                withCurrency=False)
        results.assetIdxMap = dict([(j,i) for (i,j) in enumerate(subids)])
         
        # Some hopefully redundant sanity checking of data
        results.data = Utilities.screen_data(results.data)
        if debug:
            rowNames = [sid.getSubIdString() for sid in results.assets]
            colNames = [str(dt) for dt in allDateList]
            fName = 'tmp/out0-raw-data.csv'
            Utilities.writeToCSV(results.data, fName,
                    rowNames=rowNames, columnNames=colNames)

        # Work out whether any missing roll-over flags should be True or False
        notTradedFlag = self.loadNotTradedInd(allDateList, subids).data
        maskedData = numpy.array(ma.getmaskarray(results.data), copy=True)
        missingRollOver = numpy.array(ma.getmaskarray(notTradedFlag), copy=True) * maskedData
        notTradedFlag = ma.filled(ma.where(missingRollOver, 1.0, notTradedFlag), 0.0)
        notTradedFlag = ma.array(notTradedFlag, bool)
        results.data = ma.masked_where(notTradedFlag, results.data)

        # Convert returns from longer date history to shorter
        noWkndDataIdx = []
        if compound:
            wkndDateIdx = [allDateIdxMap[d] for d in allDateList if d.weekday()>4]
            wkndRets = ma.sum(ma.take(results.data, wkndDateIdx, axis=1), axis=1)
            noWkndDataIdx = numpy.flatnonzero(ma.getmaskarray(wkndRets))

        if excludeWeekend and (len(noWkndDataIdx) == results.data.shape[0]):
            # Optional Pre-processing - if no weekend returns for any asset
            # Sift out the weekend dates prior to remaining processing
            logging.debug('No non-missing weekend returns in history')
            dates1 = [d for d in results.dates if d.weekday()<=4]
            dates1Idx = [allDateIdxMap[d] for d in dates1]
            results.data = ma.take(results.data, dates1Idx, axis=1)
            results.dates = dates1

        (results.data, results.dates) = ProcessReturns.compute_compound_returns_v3(
                results.data, results.dates, dateList, keepFirst=True, matchDates=True)

        # Sort out what is rolled-over, and what is not
        notTradedFlag = ma.take(notTradedFlag, dateListIdx, axis=1)
        maskedData = numpy.array(ma.getmaskarray(results.data), copy=True)
        issueFromDates = self.loadIssueFromDates(results.dates, results.assets)
        preIPOFlag = numpy.array(ma.getmaskarray(Matrices.fillAndMaskByDate(
                results.data, issueFromDates, results.dates)), copy=True)
         
        # Combine instances of non-trading days with the general 
        # illiquidity flag, but remove non-trades due to pre-IPO
        notTradedFlag = ((notTradedFlag==0) * (maskedData==0))==0
        notTradedFlag = notTradedFlag * (preIPOFlag==0)
        results.notTradedFlag = notTradedFlag
        results.preIPOFlag = preIPOFlag

        # Set some default values
        results.originalData = ma.array(results.data, copy=True)
        results.currencyMatrix = numpy.ones(results.data.shape, float)
         
        # Now load up whatever currency conversions are necessary
        if assetConvMap is not None:
            originalData = ma.filled(ma.array(results.data, copy=True), 0.0)
            currencyConv = self.getCurrencyConversionMatrix(
                    assetConvMap, allDateList, rmg,
                    results.assets, results.assetIdxMap)
            currencyConv = Utilities.screen_data(currencyConv)
            if debug:
                rowNames = [sid.getSubIdString() for sid in results.assets]
                colNames = [str(dt) for dt in allDateList]
                fName = 'tmp/cur-data.csv'
                Utilities.writeToCSV(currencyConv, fName,
                        rowNames=rowNames, columnNames=colNames)

            # Convert returns from longer date history to shorter
            noWkndDataIdx = []
            if compound:
                wkndRets = ma.sum(ma.take(currencyConv, wkndDateIdx, axis=1), axis=1)
                noWkndDataIdx = numpy.flatnonzero(ma.getmaskarray(wkndRets))
            
            if excludeWeekend and (len(noWkndDataIdx) == currencyConv.shape[0]):
                dates1 = [d for d in allDateList if d.weekday()<=4]
                dates1Idx = [allDateIdxMap[d] for d in dates1]
                currencyConv = ma.take(currencyConv, dates1Idx, axis=1)
            else:
                dates1 = allDateList

            # Match dates to shorter list of model dates
            currencyConv = ProcessReturns.compute_compound_returns_v3(
                    currencyConv-1.0, dates1, dateList, keepFirst=True, matchDates=True)[0]
            currencyConv += 1.0
            maskedCurrs = numpy.array(ma.getmaskarray(currencyConv), copy=True)
            # Combine asset and currency returns
            convData = (originalData + 1.0) * ma.filled(currencyConv, 1.0) - 1.0
            # Remask all returns for pre-IPO dates
            convData = ma.masked_where(preIPOFlag, convData)
            # Remask all returns where both asset and currency return are missing
            bothMissing = notTradedFlag * maskedCurrs
            convData = ma.masked_where(bothMissing, convData)
            # Save pre-conversion returns, with all non-trading dates
            # (for whatever reason) masked
            originalData = ma.masked_where(results.notTradedFlag, originalData)
            results.originalData = ma.masked_where(preIPOFlag, originalData)
            results.currencyMatrix = ma.filled(currencyConv, 1.0)
            results.data = convData

        # Temporary debugging
        if debug:
            rowNames = [sid.getSubIdString() for sid in results.assets]
            colNames = [str(dt) for dt in results.dates]
            fName = 'tmp/out1-org-data.csv'
            Utilities.writeToCSV(results.originalData, fName,
                    rowNames=rowNames, columnNames=colNames)
            fName = 'tmp/out2-cur-data.csv'
            Utilities.writeToCSV(results.currencyMatrix, fName,
                    rowNames=rowNames, columnNames=colNames)
            fName = 'tmp/out3-end-data.csv'
            Utilities.writeToCSV(results.data, fName,
                    rowNames=rowNames, columnNames=colNames)
            fName = 'tmp/out4-ntd-data.csv'
            Utilities.writeToCSV(results.notTradedFlag, fName,
                    rowNames=rowNames, columnNames=colNames)
            fName = 'tmp/out5-ipo-data.csv'
            Utilities.writeToCSV(results.preIPOFlag, fName,
                    rowNames=rowNames, columnNames=colNames)
        self.log.debug('loadTotalReturnsHistoryV3: end')
        return results

    def loadTotalLocalReturnsHistory(self, dateList, subids):
        """Returns the total daily trading currency returns
        of the sub-issues in subids for the given days.
        The return value is a TimeSeriesMatrix of total daily returns.
        If an assets has no return recorded for a given day, the
        corresponding entry in the matrix is masked.
        """
        # XXX This should really be subsumed by loadTotalReturnHistory
        # but the per rmg calling convention of the latter makes that
        # too much of a change to be on the 3.1 branch
        self.log.debug('loadTotalLocalReturnsHistory: begin')
        results = self.loadSubIssueData(
            dateList, subids, 'sub_issue_return', 'tr',
            cache=self.totalLocalReturnCache, withCurrency=False)
        self.log.debug('loadTotalLocalReturnsHistory: end')
        return results
    
    def loadRMSPredBetaHistory(self, dateList, subids, rms_id):
        """Returns the predicted beta over the given days
        of the sub-issues in the subids list.
        The return value is a TimeSeriesMatrix of predicted betas.
        If an assets has no predicted beta recorded for a given day, the
        corresponding entry in the matrix is masked.
        """
        self.log.debug('loadPredBetaHistory: begin')
        predBeta = self.loadSubIssueData(
            dateList, subids, 'rmi_predicted_beta', 'value', 
            withRMS=rms_id, withRevDate=False, withCurrency=False )
        self.log.debug('loadPredBetaHistory: end')
        return predBeta

    def loadRMSTotalRiskHistory(self, dateList, subids, rms_id):
        """Returns the total risk over the given days
        of the sub-issues in the subids list.
        The return value is a TimeSeriesMatrix of total risks.
        If an assets has no total risk recorded for a given day, the
        corresponding entry in the matrix is masked.
        """
        self.log.debug('loadTotalRiskHistory: begin')
        totalRisk = self.loadSubIssueData(
            dateList, subids, 'rmi_total_risk', 'value',
            withRMS=rms_id, withRevDate=False, withCurrency=False )
        self.log.debug('loadTotalRiskHistory: end')
        return totalRisk

    def loadRMSFactorExpHistory(self, dateList, subids, subFactor, rms_id):
        """Returns the factor exposure over the given days
        of the sub-issues in the subids list.
        The return value is a TimeSeriesMatrix of factor exposures.
        If an assets has no factor exposure recorded for a given day, the
        corresponding entry in the matrix is masked.
        """
        self.log.debug('loadExposureHistory: begin')
        table_name = 'rms_%s_factor_exposure' % rms_id
        field_name = self.getSubFactorColumnName(subFactor)
        factorExp = self.loadSubIssueData(
            dateList, subids, table_name, field_name,
            withRevDate=False, withCurrency=False )
        self.log.debug('loadExposureHistory: end')
        return factorExp

    def loadUCPHistory(self, dateList, subids, convertTo):
        """Returns the unadjusted closing prices over the given days
        of the sub-issues in the subids list.
        The return value is a TimeSeriesMatrix of UCPs.
        If an assets has no price recorded for a given day, the
        corresponding entry in the matrix is masked.
        Prices are converted to the specified currencies; if none
        is provided, the local currency of quotation is used.
        """
        self.log.debug('loadUCPHistory: begin')
        ucp = self.loadSubIssueData(
            dateList, subids, 'sub_issue_data', 'ucp', 
            convertTo=convertTo, cache=self.ucpCache)
        self.log.debug('loadUCPHistory: end')
        return ucp
    
    def loadTSOHistory(self, dateList, subids, expandForMissing=False, returnDF=False):
        """Returns the total share outstanding over the given days
        of the sub-issues in the subids list.
        The return value is a TimeSeriesMatrix of TSOs.
        If an assets has no tso recorded for a given day, the
        corresponding entry in the matrix is masked.
        IF expandForMissing is True, the code will check for 
        dates when a high proportion of TSOs are missing and
        expand the list of dates in an attempt to rectify this
        It does not backfill, however - it's intended
        to find alternative dates when one or more is a holiday,
        for instance
        """
        self.log.debug('loadTSOHistory: begin')
        tso = self.loadSubIssueData(
            dateList, subids, 'sub_issue_data', 'tso',
            withCurrency=False, cache=self.tsoCache)

        if expandForMissing:
            # Check for dates with missing TSO data
            criticalValue= 0.5
            nMissing = ma.sum(ma.getmaskarray(tso.data), axis=0) / float(len(subids))
            missingDataDatesIdx = numpy.flatnonzero(ma.where(nMissing<criticalValue, 0.0, nMissing))

            if len(missingDataDatesIdx) > 0:
                longerDates = list(dateList)
                logging.info('Missing TSO data for more than %d%% of assets on %d dates',
                        100*criticalValue, len(missingDataDatesIdx))
                logging.info('Loading extra data to try to fix this')

                # Assume input dates are sorted in order
                for dIdx in missingDataDatesIdx:
                    prevDt = [dt for dt in dateList if dt<dateList[dIdx]]
                    if len(prevDt) > 0:
                        # Add all dates inbetween the deficient date and the previous date
                        # in the list (if there are any)
                        extraDates = self.getDateRange(
                                None, prevDt[-1], dateList[dIdx], excludeWeekend=False)
                        longerDates.extend(extraDates)

                # Prune and sort
                longerDates = sorted(set(longerDates))

                # Reload history with expanded date list
                tso = self.loadSubIssueData(
                        longerDates, subids, 'sub_issue_data', 'tso',
                        withCurrency=False, cache=self.tsoCache)

        self.log.debug('loadTSOHistory: end')
        if returnDF:
            return tso.toDataFrame()
        return tso
   
    def loadTSOHistoryWithBackfill(self, date, subid, lookbackDays=10):
        """Returns the total shares outstanding for the subid
        on the max date between 'date' and 'date - lookbackDays' 
        for which tso data is available.
        Returns a tuple, (subid, date, tso), if data exists, 
        otherwise it returns None.
        """
        self.log.debug('loadTSOHistoryWithBackfill: begin')
        endDate = date
        startDate = date - datetime.timedelta(lookbackDays)
        query = """SELECT sub_issue_id, dt, tso 
                   FROM sub_issue_data_active s 
                   WHERE s.sub_issue_id = :subid_arg AND
                         s.dt = (SELECT max(dt) 
                                 FROM sub_issue_data_active sm 
                                 WHERE sm.sub_issue_id=s.sub_issue_id AND
                                       sm.dt <= :endDate_arg AND 
                                       sm.dt >= :startDate_arg)"""
        self.dbCursor.execute(query, subid_arg = subid.getSubIdString(), endDate_arg = endDate, startDate_arg = startDate)
        res = self.dbCursor.fetchall()
        if len(res) == 1:
            tso = res[0]
        else:
            tso = None
        return tso

    def loadVolumeHistory(self, dates, subids, convertTo, loadAllDates=False):
        """Returns the daily trading volume for the given dates and sub-issues.
        The return value is a TimeSeriesMatrix of daily volumes.
        If an asset has no volume recorded for a given day, the
        corresponding entry in the matrix is masked.
        The volume is in local currency if no mapping is given.
        If convertTo is not None then it maps each sub-issue to
        the currency that should be used.
        """
        self.log.debug('loadVolumeHistory: begin')
        if loadAllDates:
            dates.sort()
            allDates = self.getDateRange(None, dates[0], dates[-1])
            # Resize the cache if too small
            if self.volumeCache.maxDates < len(allDates):
                logging.warning('ModelDB volume cache too small (%d days), resizing to %d days',
                        self.volumeCache.maxDates, len(allDates))
                self.setVolumeCache(len(allDates))
            dailyVolume = self.loadSubIssueData(
                    allDates, subids, 'sub_issue_data', 'data.tdv * data.ucp',
                    convertTo=convertTo, cache=self.volumeCache)
            dailyVolume.data = Utilities.screen_data(dailyVolume.data)
            (dailyVolume.data, dailyVolume.dates) = ProcessReturns.compute_compound_returns_v3(
                    dailyVolume.data, allDates, dates, matchDates=True, sumVals=True)
        else:
            dailyVolume = self.loadSubIssueData(
                dates, subids, 'sub_issue_data', 'data.tdv * data.ucp',
                convertTo=convertTo, cache=self.volumeCache)
            dailyVolume.data = Utilities.screen_data(dailyVolume.data)
        self.log.debug('loadVolumeHistory: end')
        return dailyVolume
    
    def oracleToDate(self, dbDate):
        """Convert a date object returned from the data base to
        a datetime object.
        """
        return dbDate.date()
    
    def revertChanges(self):
        """Revert pending database changes.
        """
        self.dbConnection.rollback()
        
    def updateCumulativeRiskFreeRate(self, date, currencyCode, value):
        """Update the given cumulative risk free rate for the currency
        on the given date.
        """
        self.dbCursor.execute("""UPDATE currency_risk_free_rate
        SET cumulative = :cum_arg WHERE currency_code = :iso_arg
        AND dt = :date_arg""", cum_arg=value, iso_arg=currencyCode,
                              date_arg=date)
        self.log.info('Updated cumulative risk free return for %s'
                       % currencyCode)
        
    def updateCumulativeFactorReturns(self, rms_id, date, subFactors, values):
        """Update the given cumulative returns for the subFactors
        in the factor return table for risk model series rms_id
        on the given date.
        subFactors and values are lists of sub-factor objects and numbers
        respectively.
        """
        query = """UPDATE rms_factor_return
        SET cumulative = :value_arg
        WHERE rms_id=:rms_arg AND sub_factor_id=:factor_arg
        AND dt=:date_arg"""
        valueDicts = [dict([('factor_arg', i.subFactorID),
                            ('value_arg', j),
                            ('rms_arg', rms_id),
                            ('date_arg', date)])
                      for (i,j) in zip(subFactors, values)]
        self.dbCursor.executemany(query, valueDicts)
        self.log.info('Updated %d factor cumulative returns' % len(valueDicts))
        
    def updateIssueMap(self, newMap, currentDate):
        """Update the issue_map table to point the ModelDB ID to their
        new MarketDB counterparts.
        The newMap argument is a list of (ModelDB ID, MarketDB ID) pairs.
        """
        if len(newMap) == 0:
            return
        # check that we don't have later map entries around for these model IDs
        self.dbCursor.execute("""SELECT modeldb_id FROM issue_map
        WHERE from_dt > :date_arg AND modeldb_id in (%s)""" % \
            ','.join(["'%s'" % model.getIDString() for (model, market) in newMap]),
                              date_arg=currentDate)
        r = self.dbCursor.fetchall()
        assert(len(r) == 0)
        # Set thru_dt for current maps to currentDate
        valueDicts = [dict([('model_arg', model.getIDString()),
                            ('date_arg', currentDate)])
                      for (model, market) in newMap]
        query = """UPDATE issue_map SET thru_dt = :date_arg
        WHERE modeldb_id = :model_arg AND thru_dt > :date_arg"""
        self.dbCursor.executemany(query, valueDicts)
        # Insert new map entries
        valueDicts = [dict([('model_arg', model.getIDString()),
                            ('from_arg', currentDate),
                            ('thru_arg', self.futureDate),
                            ('market_arg', market)])
                      for (model, market) in newMap]
        query = """INSERT INTO issue_map (modeldb_id, marketdb_id, from_dt,
        thru_dt) VALUES(:model_arg, :market_arg, :from_arg, :thru_arg)"""
        self.dbCursor.executemany(query, valueDicts)

    def getModelFamilies(self):
        """
            get all model family to model name mappings from the database
        """
        modelDict={}
        self.dbCursor.execute("""select rm.name, rm.mnemonic from risk_model rm
            where exists (select * from risk_model_serie rms where rms.distribute=1 and rm.model_id=rms.rm_id)""")
        for r in self.dbCursor.fetchall():
            modelName,mnemonic=r
            familyName=modelName[0:modelName.find('Axioma') + len('Axioma')]
            modelDict[mnemonic]=familyName
        return modelDict

    def getRMGEligibleAssetTypes(self, rmg, date):
        """Load asset types eligible for inclusion in model portfolios
        for the given date and RiskModelGroup.
        """
        self.dbCursor.execute(
            """SELECT sec.classification_id
            FROM rmg_sec_type_map sec 
            WHERE sec.rmg_id = : rmg_arg and sec.from_dt <= : dt_arg 
            AND sec.thru_dt > : dt_arg""", rmg_arg = rmg.rmg_id, dt_arg = date)
        r = self.dbCursor.fetchall()
        assert(len(r)> 0 )
        return [i[0] for i in r]

    def getRMGEligibleExchanges(self, rmg, date):
        """Load stock exchanges eligible for inclusion in model portfolios
        for the given date and RiskModelGroup.
        """
        self.dbCursor.execute(
            """SELECT sec.classification_id
            FROM rmg_exchange_map sec 
            WHERE sec.rmg_id = : rmg_arg and sec.from_dt <= : dt_arg 
            AND sec.thru_dt > : dt_arg""", rmg_arg = rmg.rmg_id, dt_arg = date)
        r = self.dbCursor.fetchall()
        assert(len(r)> 0 )
        return [i[0] for i in r]

    def getRMGSectorForeignOwnership(self, rmg, date):
        """Load foreign ownership limits for sectors within
        the given date and RiskModelGroup.
        Returns a list of (classification_id, FOL) for the affected setors.
        Omits sectors with FOL = 100%.
        """
        self.dbCursor.execute(
            """SELECT fol.classification_id, fol.value
            FROM rmg_sector_fol_map fol
            WHERE fol.rmg_id = : rmg_arg and fol.from_dt <= : dt_arg 
            AND fol.thru_dt > : dt_arg
            AND fol.value < 1""", rmg_arg = rmg.rmg_id, dt_arg = date)
        r = self.dbCursor.fetchall()
        return [(i[0], i[1]) for i in r]

    def getForeignOwnershipOverride(self, date):
        """Load a list (CompanyIDs, FOL) containing issuers with manually specified 
        Foreign Ownership Limits (FOL).
        Note: returns Company ID strings, not SubIssues.
        """
        self.dbCursor.execute(
            """SELECT fol.company_id , fol.value
            FROM mdl_port_fol_active_int fol
            WHERE fol.from_dt <= :dt_arg
            AND fol.thru_dt > :dt_arg""", dt_arg=date)
        r = self.dbCursor.fetchall()
        self.log.info('Loaded %d Foreign Ownership Limit (FOL) override records', len(r))
        return [(i[0], i[1]) for i in r]

    def getAllModelPortfolioFamilies(self):
        """Return the list of all families
        """
        self.dbCursor.execute(
           """SELECT f.name, f.description
           FROM mdl_port_family f
           WHERE f.distribute = 'Y'
           ORDER BY f.id
           """)

        results = self.dbCursor.fetchall()
        self.log.info('Loaded %d model portfolio family records', len(results))
        portfolios = []
        for r in results:
            memname,desc= r
            portfolio = Utilities.Struct()
            portfolio.name = memname
            portfolio.description = desc
            portfolios.append(portfolio)
        return portfolios

    def getModelPortfolioMembersByFamily(self, family):
        """Return the list of all members in the family
        """
        self.dbCursor.execute(
           """SELECT m.id, m.name, m.description, m.short_names, m.FROM_DT, m.THRU_DT
           FROM mdl_port_family f, mdl_port_member m
           WHERE m.family_id = f.id
           AND   f.name    =  :name_arg
           """, name_arg=family.name)

        results = self.dbCursor.fetchall()
        self.log.info('Loaded %d model portfolio member records', len(results))
        portfolios = []
        for r in results:
            memid,memname,desc,shortnames,fromdt,thrudt= r
            portfolio = Utilities.Struct()
            portfolio.id = memid
            portfolio.name = memname
            portfolio.description = desc
            portfolio.short_name = shortnames
            portfolio.from_dt = fromdt
            portfolio.thru_dt = thrudt
            portfolios.append(portfolio)
        return portfolios

    def getModelPortfolioFamily(self, date, name):
        """Return the list of all members in the family
        """
        self.dbCursor.execute(
           """SELECT m.id, m.name, m.description, m.short_names
           FROM mdl_port_family f, mdl_port_member m
           WHERE m.family_id = f.id
           AND   f.distribute = 'Y'
           AND   m.FROM_DT <= :dt_arg
           AND   m.THRU_DT >  :dt_arg
           AND   f.name    =  :name_arg
           """, dt_arg=date, name_arg=name)

        results = self.dbCursor.fetchall()
        self.log.info('Loaded %d model portfolio family/member records', len(results))
        portfolios = []
        for r in results:
            memid, memname,desc,shortnames= r
            portfolio = Utilities.Struct()
            portfolio.id = memid
            portfolio.name = memname
            portfolio.description = desc
            portfolio.short_name = shortnames
            portfolios.append(portfolio)
        return portfolios
    
    def getModelPortfolioExclusion(self, date):
        """Load the model portfolio exclusion list which is risk model independent.
        """
        self.dbCursor.execute(
            """SELECT exc.sub_issue_id 
            FROM mdl_port_excl_active_int exc, sub_issue si
            WHERE exc.from_dt <= :dt_arg
            AND exc.thru_dt > :dt_arg
            AND si.from_dt <= :dt_arg
            AND si.thru_dt > :dt_arg
            AND si.sub_id = exc.sub_issue_id""", dt_arg = date)
        r = self.dbCursor.fetchall()
        self.log.info('Loaded %d model portfolio exclusion records', len(r))
        return [SubIssue(i[0]) for i in r]
    
    def getModelPortfolioInclusion(self, date):
        """Load the model portfolio inclusion list.
        """
        self.dbCursor.execute(
            """SELECT inc.sub_issue_id 
            FROM mdl_port_incl_active_int inc, sub_issue si
            WHERE inc.from_dt <= :dt_arg
            AND inc.thru_dt > :dt_arg
            AND si.from_dt <= :dt_arg
            AND si.thru_dt > :dt_arg
            AND si.sub_id = inc.sub_issue_id""", dt_arg = date)
        r = self.dbCursor.fetchall()
        self.log.info('Loaded %d model portfolio inclusion records', len(r))
        return [SubIssue(i[0]) for i in r]
    
    def getModelPortfolioSizeMap(self, model_portfolio_id):
        """Load all the size_code of the model_portfolio_id.
        """
        self.dbCursor.execute(
        """SELECT mpmm.size_code FROM 
        mdl_port_mcap_map mpmm
        WHERE mpmm.mdl_port_id = : mdl_port_id_arg""", mdl_port_id_arg = model_portfolio_id)
        r = self.dbCursor.fetchall()
        return [i[0] for i in r]
    
    def getModelPortfolioRmgMap(self, date, model_portfolio_id):
        """Load all the risk model groups covered by the model_portfolio_id
        along with their weight_factor.  Returns a dict().
        """
        self.dbCursor.execute(
        """SELECT mprm.rmg_id, mprm.weight_factor FROM
        mdl_port_rmg_map mprm
        WHERE mprm.mdl_port_id = :mdl_port_id_arg 
        AND mprm.from_dt <= :dt_arg 
        AND mprm.thru_dt > :dt_arg""", 
            mdl_port_id_arg = model_portfolio_id, dt_arg = date)
        r = self.dbCursor.fetchall()
        rmgMap = dict()
        for row in r:
            rmg = self.getRiskModelGroup(row[0])
            rmg.setRMGInfoForDate(date)
            if row[1] is not None:
                weight_factor = row[1]
            else:
                weight_factor = 1.0
            rmgMap[rmg] = weight_factor
        return rmgMap

    def getModelPortfolioMaster(self, date, rmgList=None, proForma=False):
        """Load the entire model portfolio master on a particular date.
        Returns a dict mapping SubIssue to tuples containing
        (market cap, RiskModelGroup, size segment meta code)
        If a list of RiskModelGroups is provided then only assets
        belonging to those countries are returned.
        """
        if proForma:
            tblSuffix = '_pf'
        else:
            tblSuffix = ''
        query = """SELECT 
            mpm.sub_issue_id, mpm.mcap_usd, mpm.weight_factor, mpm.rmg_id, mpm.size_code 
            FROM mdl_port_master%s mpm, sub_issue si
            WHERE mpm.dt = :dt_arg 
            AND si.sub_id = mpm.sub_issue_id
            AND si.from_dt <= :dt_arg and si.thru_dt > :dt_arg""" % tblSuffix
        if rmgList is None:
            validRiskModelGroups = self.getAllRiskModelGroups(inModels=False)
            for rmg in validRiskModelGroups:
                rmg.setRMGInfoForDate(date)
            rmgIdMap = dict([(r.rmg_id, r) for r in validRiskModelGroups])
        else:
            rmgIdMap = dict([(r.rmg_id, r) for r in rmgList])
            query += """ AND mpm.rmg_id IN (%(rmg_ids)s)""" % {
                    'rmg_ids': ','.join([str(r.rmg_id) for r in rmgList])}
        self.dbCursor.execute(query, dt_arg = date)
        r = self.dbCursor.fetchall()
        self.log.info('Loaded %d assets from %s model portfolio master', len(r), date)
        return dict([(SubIssue(i[0]), (i[1], i[2], rmgIdMap[i[3]], i[4])) for i in r])

    def getModelPortfolioByID(self, id, date):
        """Returns a Struct containing the id, name, and description
        for the given Model Portfolio member.
        Returns None if no such model portfolio exists on the given date.
        """
        self.dbCursor.execute("""SELECT id, name, description, short_names
        FROM mdl_port_member
        WHERE id = :id_arg 
        AND from_dt <= :dt_arg AND thru_dt > :dt_arg""", id_arg=id, dt_arg=date)
        r = self.dbCursor.fetchall()
        if len(r) == 0:
            return None
        mdl_port = Utilities.Struct()
        mdl_port.id = r[0][0]
        mdl_port.name = r[0][1]
        mdl_port.description = r[0][2]
        mdl_port.short_name = r[0][3]
        return mdl_port

    def getModelPortfolioByName(self, name, date):
        """Returns a Struct containing the id, name, and description
        for the given Model Portfolio member.
        Returns None if no such model portfolio exists on the given date.
        """
        self.dbCursor.execute("""SELECT id, name, description, short_names
        FROM mdl_port_member
        WHERE name = :name_arg 
        AND from_dt <= :dt_arg AND thru_dt > :dt_arg""", name_arg=name, dt_arg=date)
        r = self.dbCursor.fetchall()
        if len(r) == 0:
            return None
        elif len(r) > 1:
            self.log.error('Model portfolio name %s not unique' % name)
            return None
        mdl_port = Utilities.Struct()
        mdl_port.id = r[0][0]
        mdl_port.name = r[0][1]
        mdl_port.description = r[0][2]
        mdl_port.short_name = r[0][3]
        return mdl_port

    def getModelPortfolioByShortName(self, name, date):
        """Returns a Struct containing the id, name, and description
        for the given Model Portfolio member short name.
        Returns None if no such model portfolio exists on the given date.
        """
        self.dbCursor.execute("""SELECT id, name, description, short_names
        FROM mdl_port_member
        WHERE short_names = :name_arg 
        AND from_dt <= :dt_arg AND thru_dt > :dt_arg""", name_arg=name, dt_arg=date)
        r = self.dbCursor.fetchall()
        if len(r) == 0:
            return None
        elif len(r) > 1:
            self.log.error('Model portfolio short_name %s not unique' % name)
            return None
        mdl_port = Utilities.Struct()
        mdl_port.id = r[0][0]
        mdl_port.name = r[0][1]
        mdl_port.short_name = r[0][3]
        mdl_port.description = r[0][2]
        return mdl_port

    def getAllModelPortfolioMembers(self, date):
        """Returns a list of all model portfolio members.
        """
        self.dbCursor.execute("""SELECT id, name, description, short_names
        FROM mdl_port_member
        WHERE from_dt <= :dt_arg AND thru_dt > :dt_arg""", dt_arg=date)
        r = self.dbCursor.fetchall()
        mdl_ports = list()
        for row in r:
            mdl_port = Utilities.Struct()
            mdl_port.id = row[0]
            mdl_port.name = row[1]
            mdl_port.short_name = row[3]
            mdl_port.description = row[2]
            mdl_ports.append(mdl_port)
        return mdl_ports

    def getModelPortfolioConstituents(self, date, model_portfolio_id, proForma=False, 
                                      altDate=None, restrictDates=False, quiet=False):
        """Load the given model portfolio's constituents on the specified date.
        Returns a list of (SubIssue, weight) pairs.
        altDate, if provided, will be used for mapping the given portfolio to 
        RiskModelGroups, etc.; useful for pro forma.
        """
        if proForma:
            tblSuffix = '_pf'
        else:
            tblSuffix = ''
        if altDate is None:
            altDate = date

        # Check if this is a 'Top N' portfolio
        query = """SELECT from_idx, thru_idx
                FROM mdl_port_count_map WHERE mdl_port_id = :mdl_port_id_arg"""
        self.dbCursor.execute(query, mdl_port_id_arg=model_portfolio_id)
        r = self.dbCursor.fetchall()

        if len(r) > 0:
            # If 'Top N' portfolio determine breakpoints and most recent quarterly rebalance
            (from_idx, thru_idx) = r[0]
            query = """SELECT MAX(dt) 
                    FROM mdl_port_rebalance_universe%(tblSuffix)s WHERE dt <= :dt_arg""" % {'tblSuffix': tblSuffix}
            self.dbCursor.execute(query, dt_arg=date)
            dt = self.dbCursor.fetchall()[0][0].date()
            otherTable = ''
            otherCondition = ''
        else:
            from_idx = None
            thru_idx = None
            dt = date
            otherTable = ', mdl_port_mcap_map mpmm'
            otherCondition = 'AND mpm.size_code = mpmm.size_code AND mpmm.mdl_port_id = mprm.mdl_port_id'

        # Get constituents; for 'Top N' portfolios take from annual rebalance
        # For regular portfolios take from the specified date
        if restrictDates:
            otherTable += ', sub_issue si'
        query = """SELECT 
        mpm.sub_issue_id, mpm.mcap_usd * mpm.weight_factor * NVL(mprm.weight_factor, 1.0)
        FROM mdl_port_master%(tblSuffix)s mpm, mdl_port_rmg_map mprm %(otherTable)s
        WHERE 
        mpm.rmg_id = mprm.rmg_id
        AND mprm.mdl_port_id = :mdl_port_id_arg 
        AND mprm.from_dt <= :alt_dt_arg 
        AND mprm.thru_dt > :alt_dt_arg
        AND mpm.dt = :dt_arg 
        %(otherCond)s""" % {'otherTable': otherTable, 'otherCond': otherCondition, 'tblSuffix': tblSuffix}
        if restrictDates:
            query += """
        AND si.sub_id = mpm.sub_issue_id
        AND si.from_dt <= :dt_arg
        AND si.thru_dt > :dt_arg
        """
        self.dbCursor.execute(query, dt_arg=dt, mdl_port_id_arg=model_portfolio_id, alt_dt_arg=altDate)
        r = self.dbCursor.fetchall()
        const = [(row[0], row[1]) for row in r]

        # If 'Top N', keep required number of names
        if from_idx is not None and thru_idx is not None:
            from_idx -= 1   # Table starts at 1, = index position zero
            const = sorted(const, key=lambda x: x[1], reverse=True)[from_idx:thru_idx]
            thisSlice = set([axid for (axid, cap) in const])
            self.dbCursor.execute(query, dt_arg=date, mdl_port_id_arg=model_portfolio_id)
            r = self.dbCursor.fetchall()
            const = [(row[0], row[1]) for row in r if row[0] in thisSlice]

        if not quiet:
            self.log.info('%d model portfolio constituents loaded for %s', len(const), date)
        totalCap = sum([float(mcap) for (sid, mcap) in const])
        if totalCap == 0.0:
            return list()
        else:
            return [(SubIssue(sid), mcap / totalCap) for (sid, mcap) in const]

    def getModelPortfolioRebalanceDate(self, date):
        """Returns the most recent model portfolio rebalance date
        on or before the given date.
        """
        query = """SELECT MAX(dt) FROM
        mdl_port_rebalance_universe
        WHERE dt <= :dt_arg"""
        self.dbCursor.execute(query, dt_arg=date)
        r = self.dbCursor.fetchall()
        if len(r) == 0:
            self.log.error('No model portfolio rebalancings found before %s', date)
            return None
        return r[0][0].date()

    def getModelPortfolioRebalanceUniverse(self, date, rmgList=None, proForma=False):
        """Load the model portfolio rebalancing universe for the given date.
        If a list of RiskModelGroups is provided then only assets
        belonging to those countries are returned.
        """
        if proForma:
            tblSuffix = '_pf'
        else:
            tblSuffix = ''
        query = """SELECT mpru.sub_issue_id FROM 
        mdl_port_rebalance_universe%s mpru, sub_issue si
        WHERE mpru.dt = :dt_arg
        AND si.sub_id = mpru.sub_issue_id
        AND si.from_dt <= :dt_arg and si.thru_dt > :dt_arg""" % tblSuffix
        if rmgList is not None:
            rmgIdMap = dict([(r.rmg_id, r) for r in rmgList])
            query += """ AND mpru.rmg_id IN (%(rmg_ids)s)""" % {
                    'rmg_ids': ','.join([str(r.rmg_id) for r in rmgList])}
        self.dbCursor.execute(query, dt_arg=date)
        r = self.dbCursor.fetchall()
        self.log.info('Loaded %d assets from %s rebalance universe', len(r), date)
        return [SubIssue(i[0]) for i in r]

    def loadModelPortfolioReturns(self, dateList, mdl_ports):
        """Returns a TimeSereiesMatrix of daily returns for the 
        given model portfolios on the dates specified in dateList.
        The dates in dateList are assumed to be in chronological order.
        If there is no table entry recorded for a given index/day,
        the corresponding entry in the matrix is masked.
        """
        if self.mdlPortReturnCache is None:
            result = self.loadMdlPortReturnsInternal(dateList, mdl_ports)
        else:
            if self.mdlPortReturnCache.maxDates < len(dateList):
                logging.fatal('cache uses %d days, requesting %d',
                               self.mdlPortReturnCache.maxDates, len(dateList))
                raise ValueError('cache too small')
            missingDates = self.mdlPortReturnCache.findMissingDates(dateList)
            missingIDs = self.mdlPortReturnCache.findMissingIDs(mdl_ports)
            if len(missingIDs) > 0:
                # Get data for missing Assets for existing dates
                missingIDData = self.loadMdlPortReturnsInternal(
                                self.mdlPortReturnCache.getDateList(), missingIDs)
                self.mdlPortReturnCache.addMissingIDs(missingIDData)
            if len(missingDates) > 0:
                # Get data for all assets for missing dates
                missingDateData = self.loadMdlPortReturnsInternal(
                                missingDates, self.mdlPortReturnCache.getIDList())
                self.mdlPortReturnCache.addMissingDates(missingDateData, None)
            # extract subset
            result = self.mdlPortReturnCache.getSubMatrix(dateList, mdl_ports)
        logging.debug('received data')
        return result

    def loadMdlPortReturnsInternal(self, dateList, mdl_ports):
        """Returns a TimeSereiesMatrix of daily returns for the 
        given model portfolios on the dates specified in dateList.
        The dates in dateList are assumed to be in chronological order.
        If there is no table entry recorded for a given index/day,
        the corresponding entry in the matrix is masked.
        """
        dateLen = len(dateList)
        idLen = len(mdl_ports)
        logging.debug('loading model portfolio data for %d days and %d portfolios',
                       dateLen, idLen)
        results = Matrices.TimeSeriesMatrix(mdl_ports, dateList)
        results.data = Matrices.allMasked(results.data.shape)
        
        if dateLen == 0 or idLen == 0:
            return results
        dateIdx = dict([(date, dIdx) for (dIdx, date) in enumerate(dateList)])
        mdlIdx = dict([(mdl_port.id, mIdx) for (mIdx, mdl_port)
                         in enumerate(mdl_ports)])
        if dateLen >= 10:
            DINCR = 10
            MINCR = 100
        else:
            DINCR = 1
            MINCR = 500
        dateArgList = [('date%d' % i) for i in range(DINCR)]
        mdlArgList = [('mdl%d' % i) for i in range(MINCR)]
        defaultDict = dict([(i, None) for i in dateArgList + mdlArgList])
        query = """
          SELECT mdl_port_id, dt, value
          FROM mdl_port_return WHERE mdl_port_id IN (%(mdl_ports)s)
          AND dt IN (%(dates)s)""" % {
            'mdl_ports': ','.join([(':%s' % i) for i in mdlArgList]),
            'dates': ','.join([(':%s' % i) for i in dateArgList])}
          
        mdlStrings = [mp.id for mp in mdl_ports]
        itemsTotal = len(mdlStrings) * len(dateList)
        itemsSoFar = 0
        lastReport = 0
        startTime = datetime.datetime.now()
        for mdlChunk in listChunkIterator(mdlStrings, MINCR):
            myAxidDict = dict(zip(mdlArgList, mdlChunk))
            for dateChunk in listChunkIterator(dateList, DINCR):
                myDateDict = dict(zip(dateArgList, dateChunk))
                valueDict = dict(defaultDict)
                valueDict.update(myDateDict)
                valueDict.update(myAxidDict)
                self.dbCursor.execute(query, valueDict)
                res = self.dbCursor.fetchall()

                valueMap = dict()
                for row in res:
                    if row[2] is None:
                        continue
                    dt = row[1].date()
                    mIdx = mdlIdx[row[0]]
                    dIdx = dateIdx[dt]
                    value = float(row[2])
                    valueMap[(mIdx, dIdx)] = value
                if len(valueMap) > 0:
                    mdlIndices, dtIndices = zip(*valueMap.keys())
                    values = list(valueMap.values())
                    results.data[mdlIndices, dtIndices] = values

                itemsSoFar += len(dateChunk) * len(mdlChunk)
                if itemsSoFar - lastReport > 0.05 * itemsTotal:
                    lastReport = itemsSoFar
                    logging.debug('%d/%d, %g%%, %s', itemsSoFar, itemsTotal,
                                  100.0 * float(itemsSoFar) / itemsTotal,
                                  datetime.datetime.now() - startTime)
        return results

    def insertModelPortfolioRebalanceUniverse(self, date, subIssues, rmg, proForma=False):
        """Insert the list of SubIssues into the model portfolio 
        rebalancing universe for the given date.
        All SubIssues will be assigned to the RiskModelGroup provided,
        but if rmg = None, SubIssues will be mapped to their country
        of quotation.
        """
        if proForma:
            tblSuffix = '_pf'
        else:
            tblSuffix = ''
        query = """INSERT INTO mdl_port_rebalance_universe%s mpru
                   (dt, sub_issue_id, rmg_id)
                   VALUES (:dt_arg, :issue_arg, :rmg_arg)""" % tblSuffix
        if rmg is not None:
            rmgIdList = [rmg.rmg_id] * len(subIssues)
        else:
            sidRMGMap = dict(self.getSubIssueRiskModelGroupPairs(date, restrict=subIssues))
            rmgIdList = [sidRMGMap[sid].rmg_id for sid in subIssues]
        valueDicts = [dict([('issue_arg', sid.getSubIDString()),
                            ('dt_arg', date),
                            ('rmg_arg', rmgIdList[i])]) for (i, sid) in enumerate(subIssues)]
        self.dbCursor.executemany(query, valueDicts)
        self.log.info('Inserted %d assets into %s rebalance universe', len(valueDicts), date)

    def insertModelPortfolioMaster(self, date, subIssues, marketCaps, 
                                   weightFactors, sizeCodes, rmg, proForma=False):
        """Poplate the model portfolio master for the given date with the 
        provided SubIssues, total market caps, weightinf factors, 
        and size segment meta codes.
        Each of these should be containers containing the relevant data in
        the same order.
        If a RiskModelGroup is provided, it will be assumed that all the 
        SubIssues belong to it.  If set to None, the SubIssues will be looked up
        against sub_issue to determine their RiskModelGroup.
        """
        assert(len(subIssues) == len(marketCaps) and len(marketCaps) == len(sizeCodes)
               and len(marketCaps) == len(weightFactors))
        if proForma:
            tblSuffix = '_pf'
        else:
            tblSuffix = ''
        query = """INSERT INTO mdl_port_master%s mpm
                   (dt, sub_issue_id, mcap_usd, weight_factor, rmg_id, size_code)
                   VALUES (:dt_arg, :issue_arg, :mcap_arg, :factor_arg, :rmg_arg, :size_arg)""" % tblSuffix
        if rmg is not None:
            rmgIdList = [rmg.rmg_id] * len(subIssues)
        else:
            sidRMGMap = dict(self.getSubIssueRiskModelGroupPairs(date, restrict=subIssues))
            rmgIdList = [sidRMGMap[sid].rmg_id for sid in subIssues]

        sizeValueDicts = dict()
        for i in range(len(subIssues)):
            valueDict = dict([('dt_arg', date),
                                ('issue_arg', subIssues[i].getSubIDString()),
                                ('mcap_arg', marketCaps[i]),
                                ('factor_arg', weightFactors[i]),
                                ('rmg_arg', rmgIdList[i]),
                                ('size_arg', sizeCodes[i])])
            sizeValueDicts.setdefault(sizeCodes[i], list()).append(valueDict)
        for (sizeCode, valueDicts) in sizeValueDicts.items():
            self.dbCursor.executemany(query, valueDicts)
            self.log.info('Inserted %d assets into the %s model portfolio master for size code %d',
                    len(valueDicts), date, sizeCode)

    def insertModelPortfolioReturns(self, date, mdl_ports, values):
        """Insert returns for the given list of Model Portfolio Members on
        the specified date.
        """
        query = """INSERT INTO mdl_port_return mpr
                   (dt, mdl_port_id, value)
                   VALUES (:dt_arg, :mdl_arg, :val_arg)"""
        valueDicts = [dict([('mdl_arg', mp.id),
                            ('dt_arg', date),
                            ('val_arg', ret)]) for (mp, ret) in zip(mdl_ports, values)]
        self.dbCursor.executemany(query, valueDicts)
        self.log.info('Inserted returns for %d portfolios', len(mdl_ports))

    def deleteModelPortfolioRebalanceUniverse(self, date, rmgList=None, proForma=False):
        """Delete the model portfolio rebalance universe for the given date.
        If a list of RiskModelGroups is provided, only records corresponding
        to those countries are deleted.  Otherwise all records for that date
        will be deleted.
        """
        if proForma:
            tblSuffix = '_pf'
        else:
            tblSuffix = ''
        query = """DELETE FROM mdl_port_rebalance_universe%s mpru
                   WHERE mpru.dt = :dt_arg""" % tblSuffix
        if rmgList is not None:
            query += """ AND rmg_id IN (%(rmg_ids)s)""" % {
                    'rmg_ids': ','.join([str(r.rmg_id) for r in rmgList])}
        self.dbCursor.execute(query, dt_arg=date)

    def deleteModelPortfolioMaster(self, date, rmgList=None, proForma=False):
        """Delete the model portfolio master for the given date.
        If a list of RiskModelGroups is provided, only records corresponding
        to those countries are deleted.  Otherwise all records for that date
        will be deleted.
        """
        if proForma:
            tblSuffix = '_pf'
        else:
            tblSuffix = ''
        query = """DELETE FROM mdl_port_master%s WHERE dt = :dt_arg""" % tblSuffix
        if rmgList is not None:
            query += """ AND rmg_id IN (%(rmg_ids)s)""" % {
                    'rmg_ids': ','.join([str(r.rmg_id) for r in rmgList])}
        self.dbCursor.execute(query, dt_arg=date)

    def deleteModelPortfolioReturns(self, date, mdl_ports):
        """Delete returns for the given model portfolio members 
        on the specified date.
        """
        query = """DELETE FROM mdl_port_return 
                   WHERE dt = :dt_arg
                   AND mdl_port_id IN (%(mdl_port_ids)s)""" % {
                        'mdl_port_ids': ','.join(['%d' % n.id for n in mdl_ports])}
        self.dbCursor.execute(query, dt_arg=date)
                
    def flushModelPortfolioMaster(self):
        """Delete all model portfolio master records.
        Note: cannot rollback; self.revertChanges() not supported.
        """
        self.dbCursor.execute('TRUNCATE TABLE mdl_port_master')
  
class ClassificationProcessor(BaseProcessor):
    """Update classification_constituent table, i.e., with classification_id,
    issue_id, change_dt, and rev_dt as primary key.
    """
    
    def __init__(self, connections, familyName, memberName, revDate):
        # classification_id and weight are required fields
        fields = [('classification_id', True, int), ('weight', True, float),
                  ('src_id', False, int), ('ref', False, str),
                  ('change_dt', True, datetime.date),
                  ('change_del_flag', True, str)]
        BaseProcessor.__init__(self, fields, None)
        self.modelDB = connections.modelDB
        self.log = self.modelDB.log
        # get revision ID
        clsFamily = self.modelDB.getMdlClassificationFamily(familyName)
        clsMembers = self.modelDB.getMdlClassificationFamilyMembers(clsFamily)
        clsMember = [i for i in clsMembers if i.name==memberName][0]
        revDate = Utilities.parseISODate(revDate)
        clsRevision = self.modelDB.getMdlClassificationMemberRevision(
            clsMember, revDate)
        self.revisionId = clsRevision.id
        self.INCR = 200
        self.qArgList = [('iid%d' % i) for i in range(self.INCR)]
        self.query = """SELECT issue_id, %(fields)s
        FROM classification_const_active t1
        WHERE t1.revision_id=:revision_id_arg AND t1.issue_id IN (%(iids)s)
        ORDER BY change_dt""" % {
            'fields': self.fieldList,
            'iids': ','.join([':%s' % i for i in self.qArgList])}
        self.defaultDict = dict([(arg, None) for arg in self.qArgList])
        self.defaultDict['revision_id_arg'] = self.revisionId
        self.insertQuery = """INSERT INTO classification_constituent
        (issue_id, rev_dt, rev_del_flag,
        %(fields)s) VALUES(:iid, :rev_dt, :rev_del_flag, %(args)s)""" % {
            'fields': self.fieldList, 'args': self.argList}
        self.modelDB.dbCursor.execute("""SELECT id
          FROM classification_ref where revision_id=:rev_arg""",
                                      rev_arg=self.revisionId)
        self.myRefIDs = set([i[0] for i in self.modelDB.dbCursor.fetchall()])
    
    def processCurrentHistory(self, cur, iidClsDict):
        for record in cur.fetchall():
            iid = record[0]
            values = self.buildOldFieldValueDict(record[1:])
            rval = self.buildValueStruct(values)
            rval.classificationIdAndWeight = [
                (values['classification_id'], values['weight'])]
            iid = ModelID.ModelID(string=iid)
            if iid not in iidClsDict:
                iidClsDict[iid] = [rval]
            else:
                history = iidClsDict[iid]
                # classifications are returned ordered by change_dt
                if history[-1].change_dt == rval.change_dt:
                    # add classification to existing record
                    history[-1].classificationIdAndWeight.extend(
                        rval.classificationIdAndWeight)
                else:
                    history.append(rval)
    
    def findCode(self, iid, date, iidClsDict):
        if iid not in iidClsDict:
            return (None, None)
        rval = [i for i in iidClsDict[iid] if i.change_dt <= date]
        if len(rval) == 0:
            return (None, None)
        curVal = rval[-1]
        rval.pop()
        if len(rval) == 0:
            return (curVal, None)
        if rval[-1].change_del_flag == 'Y':
            prevVal = None
        else:
            prevVal = rval[-1]
        return (curVal, prevVal)
    
    def isSameVal(self, oldVal, newVal):
        if newVal is None and oldVal is None:
            return True
        if newVal is not None and oldVal is not None:
            # Compare classification, weights, and source
            if newVal.src_id != oldVal.src_id:
                return False
            if dict(newVal.classificationIdAndWeight) \
               != dict(oldVal.classificationIdAndWeight):
                return False
            return True
        return False
    
    def updateHistory(self, iidClsDict, iid, date, newVal):
        if newVal is None:
            history = iidClsDict[iid]
            for (idx, val) in enumerate(history):
                if val.change_dt >= date:
                    break
            if val.change_dt < date:
                idx += 1
            elif val.change_dt == date:
                history.pop(idx)
            if idx > 0:
                newVal = Utilities.Struct(history[idx-1])
                newVal.change_del_flag = 'Y'
                newVal.change_dt = date
                history.insert(idx, newVal)
        else:
            newVal = Utilities.Struct(newVal)
            newVal.change_dt = date
            newVal.change_del_flag = 'N'
            if iid not in iidClsDict or len(iidClsDict[iid]) == 0:
                iidClsDict[iid] = [newVal]
                return
            history = iidClsDict[iid]
            for (idx, val) in enumerate(history):
                if val.change_dt >= date:
                    break
            if val.change_dt < date:
                idx += 1
            elif val.change_dt == date:
                history.pop(idx)
            history.insert(idx, newVal)
            
    def bulkProcess(self, dateList, iidList, valueArray):
        cur = self.modelDB.dbCursor
        iidStrs = [i.getIDString() for i in iidList]
        iidClsDict = dict()
        for iidChunk in listChunkIterator(iidStrs, self.INCR):
            myDict = dict(self.defaultDict)
            myDict.update(dict(zip(self.qArgList, iidChunk)))
            cur.execute(self.query, myDict)
            self.processCurrentHistory(cur, iidClsDict)
        valueDicts = list()
        for (iIdx, iid) in enumerate(iidList):
            for (dIdx, date) in enumerate(dateList):
                newVal = valueArray[dIdx, iIdx]
                if newVal is not None:
                    newVal.classificationIdAndWeight = [
                        (cls, w) for (cls, w)
                        in newVal.classificationIdAndWeight
                        if cls in self.myRefIDs]
                (curVal, prevVal) = self.findCode(iid, date, iidClsDict)
                if curVal is not None and curVal.change_dt == date \
                   and self.isSameVal(prevVal, newVal):
                    # rev_del_flag terminate current database record
                    self.removeCurrentRecord(curVal, iidClsDict, iid,
                                             valueDicts)
                    continue
                if curVal is not None and curVal.change_del_flag == 'Y':
                    curVal = None
                if not self.isSameVal(curVal, newVal):
                    # insert new record and update history
                    self.updateHistory(iidClsDict, iid, date, newVal)
                    if newVal is None:
                        self.terminateRecord(iid, curVal, date, valueDicts)
                    else:
                        self.newRecords(iid, newVal, date, valueDicts)
        if len(valueDicts) > 0:
            self.log.info('Inserting %d records', len(valueDicts))
            cur.executemany(self.insertQuery, valueDicts)
    
    def removeCurrentRecord(self, curVal, historyDict, iid, valueDicts):
        history = historyDict[iid]
        history.remove(curVal)
        valueDict = self.buildNewFieldValueDict(curVal)
        valueDict['iid'] = iid.getIDString()
        valueDict['rev_dt'] = self.modelDB.revDateTime
        valueDict['rev_del_flag'] = 'Y'
        valueDicts.append(valueDict)
    
    def terminateRecord(self, iid, oldVal, date, valueDicts):
        oldVal.classification_id = oldVal.classificationIdAndWeight[0][0]
        oldVal.weight = oldVal.classificationIdAndWeight[0][1]
        valueDict = self.buildNewFieldValueDict(oldVal)
        valueDict['change_del_flag'] = 'Y'
        valueDict['change_dt'] = date
        valueDict['iid'] = iid.getIDString()
        valueDict['rev_dt'] = self.modelDB.revDateTime
        valueDict['rev_del_flag'] = 'N'
        valueDicts.append(valueDict)
    
    def newRecords(self, iid, newVal, date, valueDicts):
        for (cls, w) in newVal.classificationIdAndWeight:
            newVal.classification_id = cls
            newVal.weight = w
            newVal.change_dt = date
            newVal.change_del_flag = 'N'
            valueDict = self.buildNewFieldValueDict(newVal)
            valueDict['iid'] = iid.getIDString()
            valueDict['rev_dt'] = self.modelDB.revDateTime
            valueDict['rev_del_flag'] = 'N'
            valueDicts.append(valueDict)

class HistoryHelper(MarketDB.HistoryHelper):
    def __init__(self, keyFields, myFields, nullFieldName,
                 changeDtName, changeFlagName, revDateTime):
        MarketDB.HistoryHelper.__init__(
            self, keyFields, myFields, nullFieldName, changeDtName,
            changeFlagName, revDateTime)
    
    def _convertObjectToValue(self, fieldType, val):
        if fieldType is SubIssue:
            val = val.getSubIDString()
        return val
    
    def _convertValueToObject(self, fieldType, val):
        if fieldType is SubIssue:
            val = SubIssue(string=val) if isinstance(val, str) else \
                  SubIssue(string=MktDBUtils.getAsString(val))
        return val

class XPSFundamentalDataProcessor:
    """Process updates into sub_issue_fund_* tables.
    """
    META_CODE_QUERY = """SELECT name, id, code_type, description FROM meta_codes 
                         WHERE code_type in ('asset_dim_xpsfeed_data:item_code')"""
    BULK_DATA_QUERY =  Template('''SELECT sub_issue_id, dt, item_code_id, value, fiscal_year_end, eff_dt, eff_del_flag $CURRENCY_CLAUSE
                                   FROM $TABLE_ID
                                   WHERE (sub_issue_id, dt) IN ($SUB_ISSUE_DATE_LIST) AND item_code_id=$ITEM_CODE_ID 
                                   ORDER BY eff_dt''')
    INSERT_DATA_QUERY = Template('''INSERT INTO $TABLE_ID (sub_issue_id, dt, item_code_id, rev_dt, rev_del_flag, value, fiscal_year_end, 
                                        eff_dt, eff_del_flag $CURRENCY_CLAUSE)
                      VALUES(:sub_issue_id, :dt, :item_code_id, :rev_dt, :rev_del_flag, :value, :fiscal_year_end, :eff_dt, :eff_del_flag 
                                            $CURRENCY_VALUE)
                                ''')
    
    def __init__(self, connections, extraFields=[]):
        marketDB = connections.marketDB
        self.modelDB = connections.modelDB
        marketDB.dbCursor.execute(XPSFundamentalDataProcessor.META_CODE_QUERY)
        self.itemMap = dict([(name, (id, type, desc)) for (name, id, type,desc) in marketDB.dbCursor.fetchall()])
        self.log = self.modelDB.log
        self.INCR = 300
    
    def bulkProcess(self, updateTuples, updateValues, itemAxID):
        itemCode = self.itemMap[itemAxID.upper()][0]
        self.historyHelper = self._getHistoryHelper(itemCode)
        tableID = 'sub_issue_fundamental_data'
        currencyClause = ', currency_id '
        activeViewID = '%s_act'%(tableID)
        
        cur = self.modelDB.dbCursor
        self.histHelper.clearHistory()
        # Load history only once per axid/date combination
        historyTuples = set([(i[0].getSubIDString(), i[1]) for i in updateTuples])
        self.log.debug('XpressfeedDataProcessor.bulkProcess(): %d unique axid/date pairs in %d updates', len(historyTuples), len(updateTuples))
        
        # Load history for current sub-issue ids and dates (from ModelDB table) 
        for tupleChunk in listChunkIterator(list(historyTuples), self.INCR):
            axSubIssueDateTupleList = ','.join('(\'%s\', TO_DATE(\'%s\', \'yyyy-mm-dd\'))'%(x[0],x[1]) for x in tupleChunk)
            netQuery = self.BULK_DATA_QUERY.safe_substitute({ 'TABLE_ID' : activeViewID, 'ITEM_CODE_ID' : itemCode,
                                                              'CURRENCY_CLAUSE': currencyClause,
                                                              'SUB_ISSUE_DATE_LIST' : axSubIssueDateTupleList })
            logging.debug('RiskModelDB.FundamentalData InsertQuery:%s'%netQuery)
            cur.execute(netQuery)
            self.histHelper.processCurrentHistory(cur)
        
        # Process updates in order of effective dates: Skip insert process if values are up-to-date (from previously retrieved history)
        tmp = sorted((i[0][2], i) for i in zip (updateTuples, updateValues))
        sortedUpdates = [i[1] for i in tmp]
        valueDicts = list()
        for ((sid, dt, effDt), val) in sortedUpdates:
            key = (sid, dt, itemCode)
            self.histHelper.updateRecord(key, effDt, val, valueDicts)
        
        if len(valueDicts) > 0:
            self.log.info('Inserting %d records', len(valueDicts))
            #currencyValue = ', :currency_id ' if itemCode < 2000 else ''  TODO: This can be a problem (replaced): Confirm that IBES works!
            currencyValue = ', :currency_id '
            netInsertQuery =  self.INSERT_DATA_QUERY.safe_substitute({'TABLE_ID' : tableID, 
                                                                      'CURRENCY_CLAUSE' : currencyClause,
                                                                      'CURRENCY_VALUE' : currencyValue })
            logging.debug('RiskModelDB.Xpressfeed  Insert Query:%s'%netInsertQuery)
            logging.debug('Values:%s'%valueDicts)
            cur.executemany(netInsertQuery, valueDicts)
        else:
            if (len(sortedUpdates) <= 0):
                self.log.info("Skipped inserts because no updates were found.")
            else:
                self.log.info("Skipped insert of %d update(s) found because value(s) are up-to-date.", len(sortedUpdates))

    def _getHistoryHelper(self, itemCode):
        #currencyField = [] if itemCode >= 2000 else [('currency_id', True, int)]
        currencyField = [('currency_id', True, int)]
        keyFields = [('sub_issue_id', SubIssue), ('dt', datetime.date), ('item_code_id', int)]
        dataFields = [('value', True, float),
                      ('fiscal_year_end', True, int),
                      ('eff_dt', False, datetime.date),
                      ('eff_del_flag', False, str)] + currencyField
        nullFieldName = 'value'
        self.histHelper = HistoryHelper(keyFields, dataFields, nullFieldName, 'eff_dt', 'eff_del_flag',
                                        self.modelDB.revDateTime)

class XpressfeedDataProcessor:
    """Process updates into sub_issue_fund_* tables.
    """
    META_CODE_QUERY = """SELECT name, id, code_type, description FROM meta_codes 
                         WHERE code_type in ('asset_dim_fund_xpsfeed:item_code')"""
    BULK_DATA_QUERY =  Template('''SELECT sub_issue_id, dt, item_code, value, eff_dt, eff_del_flag $CURRENCY_CLAUSE
                                   FROM $TABLE_ID
                                   WHERE (sub_issue_id, dt) IN ($SUB_ISSUE_DATE_LIST) AND item_code=$ITEM_CODE 
                                   ORDER BY eff_dt''')
    INSERT_DATA_QUERY = Template('''INSERT INTO $TABLE_ID (sub_issue_id, dt, item_code, rev_dt, rev_del_flag, value, eff_dt, eff_del_flag 
                                           $CURRENCY_CLAUSE)
                                     VALUES(:sub_issue_id, :dt, :item_code, :rev_dt, :rev_del_flag, :value, :eff_dt, :eff_del_flag 
                                           $CURRENCY_VALUE)
                                 ''')
    
    def __init__(self, connections, extraFields=[]):
        marketDB = connections.marketDB
        self.modelDB = connections.modelDB
        marketDB.dbCursor.execute(XpressfeedDataProcessor.META_CODE_QUERY)
        self.itemMap = dict([(name, (id, type, desc)) for (name, id, type,desc) in marketDB.dbCursor.fetchall()])
        self.log = self.modelDB.log
        self.INCR = 300
    
    def bulkProcess(self, updateTuples, updateValues, item):
        itemCode = self.itemMap[item][0]
        self.historyHelper = self._getHistoryHelper(itemCode)
        tableID = 'sub_issue_xpsfeed_currency' if itemCode < 2000 else 'sub_issue_xpsfeed_number'
        currencyClause = ', currency_id ' if itemCode < 2000 else ''
        activeViewID = '%s_act'%(tableID)
        
        cur = self.modelDB.dbCursor
        self.histHelper.clearHistory()
        # Load history only once per axid/date combination
        historyTuples = set([(i[0].getSubIDString(), i[1]) for i in updateTuples])
        self.log.debug('XpressfeedDataProcessor.bulkProcess(): %d unique axid/date pairs in %d updates', len(historyTuples), len(updateTuples))
        
        # Load history for current sub-issue ids and dates (from ModelDB table) 
        for tupleChunk in listChunkIterator(list(historyTuples), self.INCR):
            axSubIssueDateTupleList = ','.join('(\'%s\', TO_DATE(\'%s\', \'yyyy-mm-dd\'))'%(x[0],x[1]) for x in tupleChunk)
            netQuery = self.BULK_DATA_QUERY.safe_substitute({ 'TABLE_ID' : activeViewID, 'ITEM_CODE' : itemCode,
                                                              'CURRENCY_CLAUSE': currencyClause,
                                                              'SUB_ISSUE_DATE_LIST' : axSubIssueDateTupleList })
            logging.debug('RiskModelDB.XPRESSFEED  dataQuery:%s'%netQuery)
            cur.execute(netQuery)
            self.histHelper.processCurrentHistory(cur)
        
        # Process updates in order of effective dates: Skip insert process if values are up-to-date (from previously retrieved history)
        tmp = sorted((i[0][2], i) for i in zip (updateTuples, updateValues))
        sortedUpdates = [i[1] for i in tmp]
        valueDicts = list()
        for ((sid, dt, effDt), val) in sortedUpdates:
            key = (sid, dt, itemCode)
            self.histHelper.updateRecord(key, effDt, val, valueDicts)
        
        if len(valueDicts) > 0:
            self.log.info('Inserting %d records', len(valueDicts))
            currencyValue = ', :currency_id ' if itemCode < 2000 else ''
            netInsertQuery =  self.INSERT_DATA_QUERY.safe_substitute({'TABLE_ID' : tableID, 
                                                                      'CURRENCY_CLAUSE' : currencyClause,
                                                                      'CURRENCY_VALUE' : currencyValue })
            logging.debug('RiskModelDB.Xpressfeed  Insert Query:%s'%netInsertQuery)
            logging.debug('Values:%s'%valueDicts)
            cur.executemany(netInsertQuery, valueDicts)
        else:
            if (len(sortedUpdates) <= 0):
                self.log.info("Skipped inserts because no updates were found.")
            else:
                self.log.info("Skipped insert of %d update(s) found because value(s) are up-to-date.", len(sortedUpdates))

    def _getHistoryHelper(self, itemCode):
        currencyField = [] if itemCode >= 2000 else [('currency_id', True, int)]
        keyFields = [('sub_issue_id', SubIssue), ('dt', datetime.date), ('item_code', int)]
        dataFields = [('value', True, float),
                      ('eff_dt', False, datetime.date),
                      ('eff_del_flag', False, str)] + currencyField
        nullFieldName = 'value'
        self.histHelper = HistoryHelper(keyFields, dataFields, nullFieldName, 'eff_dt', 'eff_del_flag',
                                        self.modelDB.revDateTime)

class ExpressoDataProcessor:
    """Process updates into sub_issue_fund_* tables.
    """
    META_CODE_QUERY = """SELECT name, id, code_type, description FROM meta_codes 
                         WHERE code_type in ('asset_dim_xpsfeed_data:item_code')"""
    BULK_DATA_QUERY =  Template('''SELECT sub_issue_id, dt, item_code_id, value, fiscal_year_end, eff_dt, eff_del_flag $CURRENCY_CLAUSE
                                   FROM $TABLE_ID
                                   WHERE (sub_issue_id, dt) IN ($SUB_ISSUE_DATE_LIST) AND item_code_id=$ITEM_CODE_ID 
                                   ORDER BY eff_dt''')
    INSERT_DATA_QUERY = Template('''INSERT INTO $TABLE_ID (sub_issue_id, dt, item_code_id, rev_dt, rev_del_flag, value, fiscal_year_end, 
                                        eff_dt, eff_del_flag $CURRENCY_CLAUSE)
                      VALUES(:sub_issue_id, :dt, :item_code_id, :rev_dt, :rev_del_flag, :value, :fiscal_year_end, :eff_dt, :eff_del_flag 
                                            $CURRENCY_VALUE)
                                ''')
    
    TARGET_TABLE_ID = 'SUB_ISSUE_EXPRESSO_DATA'
    
    def __init__(self, connections, extraFields=[]):
        marketDB = connections.marketDB
        self.modelDB = connections.modelDB
        marketDB.dbCursor.execute(XPSFundamentalDataProcessor.META_CODE_QUERY)
        self.itemMap = dict([(name, (id, type, desc)) for (name, id, type,desc) in marketDB.dbCursor.fetchall()])
        self.log = self.modelDB.log
        self.INCR = 300
    
    def bulkProcess(self, updateTuples, updateValues, itemAxID):
        itemCode = self.itemMap[itemAxID.upper()][0]
        self.historyHelper = self._getHistoryHelper(itemCode)
        
        tableID = ExpressoDataProcessor.TARGET_TABLE_ID
        currencyClause = ', currency_id '
        activeViewID = '%s_act'%(tableID)
        
        cur = self.modelDB.dbCursor
        self.histHelper.clearHistory()
        # Load history only once per axid/date combination
        historyTuples = set([(i[0].getSubIDString(), i[1]) for i in updateTuples])
        self.log.debug('ExpressoDataProcessor.bulkProcess(): %d unique axid/date pairs in %d updates', len(historyTuples), len(updateTuples))
        
        # Load history for current sub-issue ids and dates (from ModelDB table) 
        for tupleChunk in listChunkIterator(list(historyTuples), self.INCR):
            axSubIssueDateTupleList = ','.join('(\'%s\', TO_DATE(\'%s\', \'yyyy-mm-dd\'))'%(x[0],x[1]) for x in tupleChunk)
            netQuery = self.BULK_DATA_QUERY.safe_substitute({ 'TABLE_ID' : activeViewID, 'ITEM_CODE_ID' : itemCode,
                                                              'CURRENCY_CLAUSE': currencyClause,
                                                              'SUB_ISSUE_DATE_LIST' : axSubIssueDateTupleList })
            logging.debug('RiskModelDB.ExpressoFundamentalData InsertQuery:%s'%netQuery)
            cur.execute(netQuery)
            self.histHelper.processCurrentHistory(cur)
        
        # Process updates in order of effective dates: Skip insert process if values are up-to-date (from previously retrieved history)
        tmp = sorted((i[0][2], i) for i in zip (updateTuples, updateValues))
        sortedUpdates = [i[1] for i in tmp]
        valueDicts = list()
        for ((sid, dt, effDt), val) in sortedUpdates:
            key = (sid, dt, itemCode)
            self.histHelper.updateRecord(key, effDt, val, valueDicts)
        
        if len(valueDicts) > 0:
            self.log.info('Inserting %d records', len(valueDicts))
            #currencyValue = ', :currency_id ' if itemCode < 2000 else ''  TODO: This can be a problem (replaced): Confirm that IBES works!
            currencyValue = ', :currency_id '
            netInsertQuery =  self.INSERT_DATA_QUERY.safe_substitute({'TABLE_ID' : tableID, 
                                                                      'CURRENCY_CLAUSE' : currencyClause,
                                                                      'CURRENCY_VALUE' : currencyValue })
            logging.debug('RiskModelDB.Expresso Insert Query:%s'%netInsertQuery)
            logging.debug('Values:%s'%valueDicts)
            cur.executemany(netInsertQuery, valueDicts)
        else:
            if (len(sortedUpdates) <= 0):
                self.log.info("Skipped inserts because no updates were found.")
            else:
                self.log.info("Skipped insert of %d update(s) found because value(s) are up-to-date.", len(sortedUpdates))

    def _getHistoryHelper(self, itemCode):
        #currencyField = [] if itemCode >= 2000 else [('currency_id', True, int)]
        currencyField = [('currency_id', True, int)]
        keyFields = [('sub_issue_id', SubIssue), ('dt', datetime.date), ('item_code_id', int)]
        dataFields = [('value', True, float),
                      ('fiscal_year_end', True, int),
                      ('eff_dt', False, datetime.date),
                      ('eff_del_flag', False, str)] + currencyField
        nullFieldName = 'value'
        self.histHelper = HistoryHelper(keyFields, dataFields, nullFieldName, 'eff_dt', 'eff_del_flag',
                                        self.modelDB.revDateTime)
     
class DataProcessor:
    """Process updates into sub_issue_fund_* tables.
    """
    mktTableMap = {
        'sub_issue_fund_number': 'asset_dim_fund_number',
        'sub_issue_fund_currency': 'asset_dim_fund_currency',
        'sub_issue_esti_number': 'asset_dim_esti_number',
        'sub_issue_esti_currency': 'asset_dim_esti_currency',
        'sub_issue_fundamental_data' : 'asset_dim_fundamental_data'
    }
    
    def __init__(self, connections, tableName, extraFields):
        marketDB = connections.marketDB
        self.modelDB = connections.modelDB
        self.tableName = tableName
        marketDB.dbCursor.execute("""SELECT name, id, code_type
           FROM meta_codes
           WHERE code_type in ('%(tablename)s:item_code')""" % {
            'tablename': self.mktTableMap[self.tableName]})
        self.itemMap = dict([(name, (id, type)) for (name, id, type)
                             in marketDB.dbCursor.fetchall()])
        keyFields = [('sub_issue_id', SubIssue), ('dt', datetime.date),
                     ('item_code', int)]
        dataFields = [('value', True, float),
                      ('eff_dt', False, datetime.date),
                      ('eff_del_flag', False, str)] + extraFields
        nullFieldName = 'value'
        self.histHelper = HistoryHelper(keyFields, dataFields, nullFieldName,
                                        'eff_dt', 'eff_del_flag',
                                        self.modelDB.revDateTime)
        self.log = self.modelDB.log
        self.INCR = 200
        self.bulkSidList = [('sid%d' % i) for i in range(self.INCR)]
        self.bulkDateList = [('dt%d' % i) for i in range(self.INCR)]
        self.currentBulkQuery = """SELECT t1.sub_issue_id, t1.dt, item_code,
        %(fields)s FROM %(table)s t1
        WHERE (t1.sub_issue_id, t1.dt) IN (%(args)s) AND item_code=:item
        ORDER BY eff_dt""" % {
            'table': self.tableName + '_active',
            'fields': self.histHelper.fieldList,
            'args': ','.join(['(:%s, :%s)' % arg for arg
                              in zip(self.bulkSidList, self.bulkDateList)])}
        self.defaultDict = dict([(arg, None) for arg
                                 in self.bulkSidList + self.bulkDateList])
        self.insertQuery = """INSERT INTO %(table)s
        (sub_issue_id, dt, item_code, rev_dt, rev_del_flag, %(fields)s)
        VALUES(:sub_issue_id, :dt, :item_code, :rev_dt, :rev_del_flag,
        %(args)s)""" % {
            'table': self.tableName,
            'fields': self.histHelper.fieldList,
            'args': self.histHelper.argList}
    
    def bulkProcess(self, updateTuples, updateValues, item):            
        itemCode = self.itemMap[item][0] if item in self.itemMap else self.itemMap[item.upper()][0]
        cur = self.modelDB.dbCursor
        self.histHelper.clearHistory()
        # Load history only once per axid/date combination
        historyTuples = set([(i[0].getSubIDString(), i[1])
                             for i in updateTuples])
        self.log.debug('DataProcessor.bulkProcess(): %d unique axid/date pairs in %d updates',
                      len(historyTuples), len(updateTuples))
        
        insertedRows = 0
        # Load history for current sub-issue ids and dates (from ModelDB table) 
        for tupleChunk in listChunkIterator(list(historyTuples), self.INCR):
            myDict = self.defaultDict.copy()
            myDict.update(dict(zip(self.bulkSidList,
                                   [i[0] for i in tupleChunk])))
            myDict.update(dict(zip(self.bulkDateList,
                                   [i[1] for i in tupleChunk])))
            myDict['item'] = itemCode
            cur.execute(self.currentBulkQuery, myDict)
            self.histHelper.processCurrentHistory(cur)

        # Process updates in order of effective dates: Skip insert process if values are up-to-date (from previously retrieved history)
        tmp = sorted((i[0][2], i) for i in zip (updateTuples, updateValues))
        sortedUpdates = [i[1] for i in tmp]
        valueDicts = list()
        
        for ((sid, dt, effDt), val) in sortedUpdates:
            key = (sid, dt, itemCode)
            self.histHelper.updateRecord(key, effDt, val, valueDicts)
        if len(valueDicts) > 0:
            self.log.info('Inserting %d records', len(valueDicts))
            cur.executemany(self.insertQuery, valueDicts)
            insertedRows += cur.rowcount
        else:
            if (len(sortedUpdates) <= 0):
                self.log.info("Skipped inserts because no updates were found.")
            else:
                self.log.info("Skipped insert of %d update(s) found because value(s) are up-to-date.", len(sortedUpdates))
    
        return insertedRows
    
'''TODO: Document me.'''
class EstimateCurrencyProcessor(DataProcessor):
    def __init__(self, connections):
        DataProcessor.__init__(
            self, connections, 'sub_issue_esti_currency', [('currency_id', True, int)]) 

'''TODO: Document me.'''
class EstiNumberProcessor(DataProcessor):
    def __init__(self, connections):
        DataProcessor.__init__(
            self, connections, 'sub_issue_esti_number', list())
 
'''NEW:  Handles Unadjusted IBES Estimate Data'''
class BaseEstimateDataProcessor:
    """Process updates into sub_issue_fund_* tables.
    """
    mktTableMap = {
        'sub_issue_fund_number': 'asset_dim_fund_number',
        'sub_issue_fund_currency': 'asset_dim_fund_currency',
        'sub_issue_esti_number': 'asset_dim_esti_number',
        'sub_issue_esti_currency': 'asset_dim_esti_currency', 
        'sub_issue_estimate_data': 'asset_dim_estimate_data' }
    
    def __init__(self, connections, tableName, extraFields):
        itemsTableName = 'asset_dim_esti_currency' if '_ESTI' in tableName.upper()\
                          else self.mktTableMap[tableName]
        self.marketDB = connections.marketDB
        self.modelDB = connections.modelDB
        self.tableName = tableName
    
        self.marketDB.dbCursor.execute("""SELECT name, id, code_type
           FROM meta_codes
           WHERE code_type in ('%(tablename)s:item_code')""" % {
            'tablename': itemsTableName })
    
        self.itemMap = dict([(name, (id, type)) for (name, id, type)
                             in self.marketDB.dbCursor.fetchall()])
        
        keyFields = [('sub_issue_id', SubIssue), ('dt', datetime.date),
                     ('item_code', int)]
        dataFields = [('value', True, float),
                      ('eff_dt', False, datetime.date),
                      ('eff_del_flag', False, str)] + extraFields
        nullFieldName = 'value'
        self.histHelper = HistoryHelper(keyFields, dataFields, nullFieldName,
                                        'eff_dt', 'eff_del_flag',
                                        self.modelDB.revDateTime)
        self.log = self.modelDB.log
        self.INCR = 200
        self.bulkSidList = [('sid%d' % i) for i in range(self.INCR)]
        self.bulkDateList = [('dt%d' % i) for i in range(self.INCR)]
        self.currentBulkQuery = """SELECT t1.sub_issue_id, t1.dt, item_code,
        %(fields)s FROM %(table)s t1
        WHERE (t1.sub_issue_id, t1.dt) IN (%(args)s) AND item_code=:item
        ORDER BY eff_dt""" % {
            'table': self.tableName + '_active',
            'fields': self.histHelper.fieldList,
            'args': ','.join(['(:%s, :%s)' % arg for arg
                              in zip(self.bulkSidList, self.bulkDateList)])}
        self.defaultDict = dict([(arg, None) for arg
                                 in self.bulkSidList + self.bulkDateList])
        self.insertQuery = """INSERT INTO %(table)s
        (sub_issue_id, dt, item_code, rev_dt, rev_del_flag, %(fields)s)
        VALUES(:sub_issue_id, :dt, :item_code, :rev_dt, :rev_del_flag,
        %(args)s)""" % {
            'table': self.tableName,
            'fields': self.histHelper.fieldList,
            'args': self.histHelper.argList}
    
    def bulkProcess(self, updateTuples, updateValues, item):
        itemCode = self.itemMap[item][0]
        cur = self.modelDB.dbCursor
        self.histHelper.clearHistory()
        # Load history only once per axid/date combination
        historyTuples = set([(i[0].getSubIDString(), i[1]) for i in updateTuples])
        self.log.debug('BaseEstimateDataProcessor.bulkProcess(): %d unique axid/date pairs in %d updates', len(historyTuples), len(updateTuples))
 
        # Load history for current sub-issue ids and dates (from ModelDB table) 
        for tupleChunk in listChunkIterator(list(historyTuples), self.INCR):
            myDict = self.defaultDict.copy()
            myDict.update(dict(zip(self.bulkSidList,
                                   [i[0] for i in tupleChunk])))
            myDict.update(dict(zip(self.bulkDateList,
                                   [i[1] for i in tupleChunk])))
            myDict['item'] = itemCode
            cur.execute(self.currentBulkQuery, myDict)
            self.histHelper.processCurrentHistory(cur)

        # Process updates in order of effective dates: Skip insert process if values are up-to-date 
        #   (from previously retrieved history)
        tmp = sorted((i[0][2], i) for i in zip (updateTuples, updateValues))
        sortedUpdates = [i[1] for i in tmp]
        valueDicts = list()
        
        # 3. Find out what values are already present in database: Transfer only new values
        sidList = set([sid.getSubIDString() for ((sid, dt, effDt), val) in sortedUpdates])
        dataDateList = set([dt for ((sid, dt, effDt), val) in sortedUpdates])
        effDateList = set([effDt for ((sid, dt, effDt), val) in sortedUpdates])
        dbRecords = self._getExistingSubIssueDBRecords(sidList, [itemCode], dataDateList, effDateList)
        # N.B. for comparison, convert to date instances
        existingRecords = [(row[0], row[2].date(), row[5].date()) for row in dbRecords]  
        revDT = datetime.datetime.now()
        
        # SUB_ISSUE_ESTIMATE_DATA [SUB_ISSUE_ID | ITEM_CODE | DT | VALUE | CURRENCY_ID | EFF_DT ]
        for ((sid, dt, effDt), val) in sortedUpdates:
            if val is not None: # SubIssue: D6NBTK4FG811 (Country:JP | CN) DT =2003-03-31 EffDT = 2000-04-20 => currency-id = 25 => val = None!
                currencyID = val.currency_id
                estimateValue = val.value
                sidStrID = sid.getSubIDString()
            
                if (sidStrID, dt, effDt) in existingRecords:
                    logging.debug('SKIPPING value alreday present in Database', (sid, dt, effDt))
                else:
                    #--- Add this (new) value to valueDicts ---# 
                    valueDicts.append({'SUB_ISSUE_ID' : sidStrID, 'ITEM_CODE' : itemCode, 'DT' : dt,
                                       'VALUE' : estimateValue, 'CURRENCY_ID' : currencyID, 'EFF_DT' : effDt, 
                                       'EFF_DEL_FLAG' : 'N', 'REV_DT' : revDT, 'REV_DEL_FLAG' : 'N'})
            else:
                logging.debug('Skipping invalid currency + estimate value = None for SID:%s Date:%s EffDate:%s'%(sid, dt, effDt))
        
        # 4. Transfer new values (if any) to recepient table 
        insertedRecordCount = 0  
        if len(valueDicts) > 0:
            self.log.info('Inserting %d records', len(valueDicts))
            cur.executemany(self.insertQuery, valueDicts)
            insertedRecordCount = cur.rowcount
        else:
            if (len(sortedUpdates) <= 0):
                self.log.info("Skipped inserts because no updates were found.")
            else:
                self.log.info("Skipped insert of %d update(s) found because value(s) are up-to-date.", len(sortedUpdates))
                
        return insertedRecordCount
    
    def _getFormattedValueForDB(self, dbValue, dType): 
        ORACLE_DATE_FORMAT = '%d-%b-%Y'
        if dType == MX_DT.DATE: 
            return MktDBUtils.getDateInstance(dbValue).strftime(ORACLE_DATE_FORMAT)
        else:
            return str(dbValue)
   
    def _getExistingSubIssueDBRecords(self, subIssueIDSet, itemCodeList, dataDateList, effDateList):
        tableID = 'SUB_ISSUE_ESTIMATE_DATA'
        #axIDListStr = ','.join('\'%s\''%item for item in subIssueIDList)
        itemCodeListStr =  ','.join('\'%s\''%item for item in itemCodeList)
        effDTListStr = ','.join('to_date(\'%s\',\'yyyy-mm-dd\')'%MktDBUtils.getDateInstance(dateTime) for dateTime in effDateList) 
        #dataDTListStr = ','.join('to_date(\'%s\',\'yyyy-mm-dd\')'%self._getFormattedValueForDB(item, MX_DT) for item in dataDateList)
        values = []
        if len(subIssueIDSet) > 0 and len(effDTListStr) > 0:
            subIssueIDList = list(subIssueIDSet)
            cursor = self.modelDB.dbCursor
            for axSubIDList in MktDBUtils.listChunkIterator(subIssueIDList, MINCR):
                axSubIDListStr = ','.join('\'%s\''%item for item in axSubIDList)
                query = Template("""SELECT * FROM MODELDB_GLOBAL.$TableID where SUB_ISSUE_ID in ($SUBISSUE_IDS)
                                    AND ITEM_CODE in ($ITEM_CODES) AND EFF_DT in ($EFFECTIVE_DATES)""")
                query = query.safe_substitute(TableID=tableID, EFFECTIVE_DATES = effDTListStr, 
                                              ITEM_CODES = itemCodeListStr, SUBISSUE_IDS = axSubIDListStr)
                self.log.debug('BaseEstimateDataProcessor._getExistingSubIssueDBRecords() going to execute %s ', query)
                cursor.execute(query)
                #self.log.debug('BaseEstimateDataProcessor._getExistingSubIssueDBRecords() done executing %s ', query)
                values += cursor.fetchall()
        return values
       
'''NEW: Handles Unadjusted IBES Estimate Data'''
class EstimateDataProcessor(BaseEstimateDataProcessor):
    def __init__(self, connections):
        BaseEstimateDataProcessor.__init__(
            self, connections, 'sub_issue_estimate_data', [('currency_id', True, int)]) 
        
class FundamentalDataProcessor(XPSFundamentalDataProcessor):
    def __init__(self, connections):
        XPSFundamentalDataProcessor.__init__(self, connections)

class FundamentalNumberProcessor(DataProcessor):
    def __init__(self, connections):
        DataProcessor.__init__(
            self, connections, 'sub_issue_fund_number', list())

class FundamentalCurrencyProcessor(DataProcessor):
    def __init__(self, connections):
        DataProcessor.__init__(
            self, connections, 'sub_issue_fund_currency', [('currency_id', True, int)])
        
class SubIssueProcessor(BaseProcessor):
    def __init__(self, connections, tableName, extraFields=[]):
        BaseProcessor.__init__(self, extraFields, None)
        self.modelDB = connections.modelDB
        self.log = self.modelDB.log
        self.tableName = tableName
        self.INCR = 400
        self.bulkArgList = [('sid%d' % i) for i in range(self.INCR)]
        self.currentBulkQuery = """SELECT sub_issue_id, %(fields)s
           FROM %(table)s t1
           WHERE rev_dt=(SELECT MAX(rev_dt) FROM %(table)s t2
             WHERE t1.sub_issue_id=t2.sub_issue_id AND t1.dt=t2.dt
             AND t2.rev_dt <= :rev_arg)
           AND t1.rev_del_flag='N' and t1.sub_issue_id IN (%(args)s)
           AND dt=:dt_arg""" % {
            'table': self.tableName, 'fields': self.fieldList,
            'args': ','.join([':%s' % arg for arg in self.bulkArgList])}
        self.defaultDict = dict([(arg, None) for arg in self.bulkArgList])
        self.insertInputSizes['sid'] = cx_Oracle.STRING
        self.insertQuery = """INSERT INTO %(table)s (sub_issue_id, dt,
        rev_dt, rev_del_flag, %(fields)s)
        VALUES(:sid, :dt, :rev_dt, :rev_del_flag, %(args)s)""" % {
            'table': self.tableName, 'fields': self.fieldList,
            'args': self.argList}
    
    def bulkProcess(self, dateList, sidList, valueList):
        self.log.debug('SubIssueProcessor')
        for (dIdx, date) in enumerate(dateList):
            self.processByDate(date, sidList, valueList[dIdx])
    
    def processByDate(self, date, sidList, valueList):
        cur = self.modelDB.dbCursor
        valueDicts = []
        self.defaultDict['dt_arg'] = date
        self.defaultDict['rev_arg'] = self.modelDB.revDateTime
        sidStrs = [sid.getSubIDString() for sid in sidList]
        for sidValChunk in listChunkIterator(list(zip(sidStrs, valueList)),
                                             self.INCR):
            updateDict = dict(zip(self.bulkArgList,
                                  [i[0] for i in sidValChunk]))
            myDict = dict(self.defaultDict)
            myDict.update(updateDict)
            cur.execute(self.currentBulkQuery, myDict)
            currentDict = {}
            for r in cur.fetchall():
                currentDict[r[0]] = self.buildOldFieldValueDict(r[1:])
            for (sid, value) in sidValChunk:
                valueDict = {}
                if value is None:
                    if sid in currentDict:
                        # Delete: insert previous record with change_del_flag
                        # set to Y and change_dt = date
                        valueDict = currentDict[sid]
                        valueDict['rev_del_flag'] = 'Y'
                    else:
                        # No previous record to delete
                        pass
                else:
                    if sid in currentDict and not self.areFieldsChangedDict(
                        currentDict[sid], value):
                        # Matches previous record: skip
                        self.log.debug(
                            'Matching record already in database.'
                            ' Skipping %s on %s', sid, date)
                    else:
                        valueDict = self.buildNewFieldValueDict(value)
                        valueDict['rev_del_flag'] = 'N'
                if len(valueDict) > 0:
                    valueDict['sid'] = sid
                    valueDict['dt']= date
                    valueDict['rev_dt'] = self.modelDB.revDateTime
                    valueDicts.append(valueDict)
        if len(valueDicts) > 0:
            cur.setinputsizes(**self.insertInputSizes)
            cur.executemany(self.insertQuery, valueDicts)
            self.log.info('Updated %d records', len(valueDicts))

class SubIssueCumulativeReturnProcessor(SubIssueProcessor):
    def __init__(self, marketDB):
        fields = [('value', True, float), ('rmg_id', True, int)]
        SubIssueProcessor.__init__(
            self, marketDB, 'sub_issue_cumulative_return', fields)

class SubIssueDataProcessor(SubIssueProcessor):
    def __init__(self, marketDB):
        fields = [('ucp', True, float), ('currency_id', True, int),
                  ('price_marker', True, int), ('rmg_id', True, int),
                  ('tso', True, float), ('tdv', True, float)]
        SubIssueProcessor.__init__(
            self, marketDB, 'sub_issue_data', fields)

class SubIssueReturnProcessor(SubIssueProcessor):
    def __init__(self, marketDB):
        fields = [('tr', True, float), ('rmg_id', True, int)]
        SubIssueProcessor.__init__(
            self, marketDB, 'sub_issue_return', fields)

class SubIssueDivYieldProcessor(SubIssueProcessor):
    def __init__(self, marketDB):
        fields = [('value', True, float)]
        SubIssueProcessor.__init__(
            self, marketDB, 'sub_issue_divyield', fields)

class HistoricBetaProcessorV3(BaseProcessor):
    def __init__(self, connections, gp=None):
        fields = [('value', True, float), ('p_value', True, float), ('nobs', True, float)]
        BaseProcessor.__init__(self, fields, None)
        self.modelDB = connections.modelDB
        self.log = self.modelDB.log
        self.currentBulkQuery = """SELECT sub_issue_id, home, %(fields)s
           FROM rmg_historic_beta_v3
           WHERE dt=:dt_arg""" % { 'fields': self.fieldList}
        self.insertQuery = """INSERT INTO rmg_historic_beta_v3
        (sub_issue_id, dt, home, %(fields)s)
        VALUES(:sub_issue_id, :dt, :home, %(args)s)""" % {
            'fields': self.fieldList, 'args': self.argList}
        self.updateQuery = """UPDATE rmg_historic_beta_v3 SET %(fields)s
        WHERE sub_issue_id=:sub_issue_id AND dt=:dt AND home=:home""" % {
            'fields': ','.join(['%s = :%s' % (f, f) for (f, c, t)
                                   in fields])}
        self.deleteQuery = """DELETE FROM rmg_historic_beta_v3
        WHERE sub_issue_id=:sub_issue_id AND dt=:dt AND home=:home"""
    
    def isDifferent(self, val1, val2):
        return ((val1 is None and val2 is not None) 
                or (val1 is not None and val2 is None)
                or (val1 is not None and val2 is not None and abs(val1-val2) > 1e-5))
    
    def bulkProcess(self, dateList, rmgList, values, restrictSet):
        self.log.debug('HistoricBetaProcessor')
        cur = self.modelDB.dbCursor
        updateDicts = list()
        insertDicts = list()
        deleteDicts = list()
        for (dIdx, date) in enumerate(dateList):
            myDict = {'dt_arg': date}
            cur.execute(self.currentBulkQuery, myDict)
            currentDict = dict(((i[0],i[1]), i[2:]) for i in cur.fetchall())
            for (rIdx, rmgId) in enumerate(rmgList):
                if values[dIdx, rIdx] is not None:
                    valuePairs = values[dIdx, rIdx].valuePairs
                    for (sid, home, beta1, beta2, beta3) in valuePairs:
                        if restrictSet is not None and sid not in restrictSet:
                            continue
                        sid = sid.getSubIDString()
                        if (sid, home) in currentDict:
                            (curBeta1, curBeta2, curBeta3) = currentDict[(sid, home)]
                             
                            if self.isDifferent(curBeta1, beta1) \
                                    or self.isDifferent(curBeta2, beta2) \
                                    or self.isDifferent(curBeta3, beta3):
                                updateDicts.append({'sub_issue_id': sid,
                                                    'dt': date,
                                                    'home': home,
                                                    'value': beta1,
                                                    'p_value': beta2,
                                                    'nobs': beta3})
                        else:
                            insertDicts.append({'sub_issue_id': sid,
                                                'dt': date,
                                                'home': home,
                                                'value': beta1,
                                                'p_value': beta2,
                                                'nobs': beta3})
        if len(insertDicts) > 0:
            cur.setinputsizes(**self.insertInputSizes)
            cur.executemany(self.insertQuery, insertDicts)
            self.log.info('Inserting %d new records', len(insertDicts))
        if len(updateDicts) > 0:
            cur.setinputsizes(**self.insertInputSizes)
            cur.executemany(self.updateQuery, updateDicts)
            self.log.info('Updating %d records', len(updateDicts))
        if len(deleteDicts) > 0:
            cur.executemany(self.deleteQuery, deleteDicts)
            self.log.info('Deleting %d records', len(deleteDicts))

class CurrencyBetaProcessor(BaseProcessor):
    def __init__(self, connections, gp=None):
        fields = [('value', True, float), ('p_value', True, float), ('nobs', True, float)]
        BaseProcessor.__init__(self, fields, None)
        self.modelDB = connections.modelDB
        self.log = self.modelDB.log
        self.currentBulkQuery = """SELECT sub_issue_id, rmg_id, iso, asset_type, market, %(fields)s
           FROM rmg_currency_beta WHERE dt=:dt_arg""" % { 'fields': self.fieldList}
        self.insertQuery = """INSERT INTO rmg_currency_beta
        (sub_issue_id, dt, rmg_id, iso, asset_type, market, %(fields)s)
        VALUES(:sub_issue_id, :dt, :rmg_id, :iso, :asset_type, :market, %(args)s)""" % {
            'fields': self.fieldList, 'args': self.argList}
        self.updateQuery = """UPDATE rmg_currency_beta SET asset_type=:asset_type,market=:market,%(fields)s
        WHERE sub_issue_id=:sub_issue_id AND dt=:dt AND rmg_id=:rmg_id AND iso=:iso""" % {
            'fields': ','.join(['%s = :%s' % (f, f) for (f, c, t) in fields])}
        self.deleteQuery = """DELETE FROM rmg_currency_beta
        WHERE sub_issue_id=:sub_issue_id AND dt=:dt AND rmg=:rmg AND iso=:iso"""

    def isDifferent(self, val1, val2):
        return ((val1 is None and val2 is not None)
                or (val1 is not None and val2 is None)
                or (val1 is not None and val2 is not None and abs(val1-val2) > 1e-5))

    def isDifferentString(self, val1, val2):
        val1 = str(val1).rstrip()
        val2 = str(val2).rstrip()
        return ((val1 is None and val2 is not None)
                or (val1 is not None and val2 is None)
                or (val1 is not None and val2 is not None and val1 != val2))

    def bulkProcess(self, dateList, rmgList, values, restrictSet):
        self.log.debug('CurrencyBetaProcessor')
        cur = self.modelDB.dbCursor
        updateDicts = list()
        insertDicts = list()
        deleteDicts = list()
        for (dIdx, date) in enumerate(dateList):
            myDict = {'dt_arg': date}
            cur.execute(self.currentBulkQuery, myDict)
            currentDict = dict(((i[0],i[1],i[2]), i[3:]) for i in cur.fetchall())
            for (rIdx, rmgId) in enumerate(rmgList):
                if values[dIdx, rIdx] is not None:
                    valuePairs = values[dIdx, rIdx].valuePairs
                    for (sid, rmg_id, iso, atype, mtype, beta1, beta2, beta3) in valuePairs:
                        if restrictSet is not None and sid not in restrictSet:
                            continue
                        sid = sid.getSubIDString()
                        if (sid, rmg_id, iso) in currentDict:
                            (curType, curMkt, curBeta1, curBeta2, curBeta3) = currentDict[(sid, rmg_id, iso)]

                            if self.isDifferent(curBeta1, beta1) \
                                    or self.isDifferent(curBeta2, beta2) \
                                    or self.isDifferent(curBeta3, beta3) \
                                    or self.isDifferentString(curType, atype) \
                                    or self.isDifferentString(curMkt, mtype):
                                updateDicts.append({'sub_issue_id': sid,
                                                    'dt': date,
                                                    'rmg_id': rmg_id,
                                                    'iso': iso,
                                                    'value': beta1,
                                                    'p_value': beta2,
                                                    'nobs': beta3,
                                                    'asset_type': atype,
                                                    'market': mtype})
                        else:
                            insertDicts.append({'sub_issue_id': sid,
                                                'dt': date,
                                                'rmg_id': rmg_id,
                                                'iso': iso,
                                                'value': beta1,
                                                'p_value': beta2,
                                                'nobs': beta3,
                                                'asset_type': atype,
                                                'market': mtype})
        if len(insertDicts) > 0:
            cur.setinputsizes(**self.insertInputSizes)
            cur.executemany(self.insertQuery, insertDicts)
            self.log.info('Inserting %d new records', len(insertDicts))
        if len(updateDicts) > 0:
            cur.setinputsizes(**self.insertInputSizes)
            cur.executemany(self.updateQuery, updateDicts)
            self.log.info('Updating %d records', len(updateDicts))
        if len(deleteDicts) > 0:
            cur.executemany(self.deleteQuery, deleteDicts)
            self.log.info('Deleting %d records', len(deleteDicts))


class DescriptorDataProcessor(BaseProcessor):
    """Process updates into desc_* tables.
    """
    def __init__(self, connection, gp=None):

        fields = [(value, True, float) for value in gp.descIDList]
        BaseProcessor.__init__(self, fields, None)
        # Set up parameters
        self.modelDB = connection.modelDB
        self.log = self.modelDB.log
        self.descIDList = gp.descIDList
        self.insertInputSizes['sid'] = cx_Oracle.STRING

        # Set up DB queries
        self.currentBulkQueryLegacy = """SELECT sub_issue_id, %(fields)s FROM %%(table_arg)s
        WHERE dt=:dt_arg""" % {'fields': self.fieldList}
        self.currentBulkQuery = """SELECT sub_issue_id, %(fields)s FROM %%(table_arg)s
        WHERE dt=:dt_arg AND curr_field=:curr_arg""" % {'fields': self.fieldList}
        self.insertQueryLegacy = """INSERT INTO %%(table_arg)s (sub_issue_id, dt,
        %(fields)s) VALUES(:sub_issue_id, :dt, %(args)s)""" % \
                {'fields': self.fieldList, 'args': self.argList}
        self.insertQuery = """INSERT INTO %%(table_arg)s (sub_issue_id, dt, curr_field,
        %(fields)s) VALUES(:sub_issue_id, :dt, :curr_field, %(args)s)""" % \
                {'fields': self.fieldList, 'args': self.argList}
        self.updateQueryLegacy = """UPDATE %%(table_arg)s SET %(fields)s
        WHERE sub_issue_id=:sub_issue_id AND dt=:dt""" % \
                {'fields': ','.join(['%s = :%s' % (f, f) for (f, c, t) in fields])}
        self.updateQuery = """UPDATE %%(table_arg)s SET %(fields)s
        WHERE sub_issue_id=:sub_issue_id AND dt=:dt AND curr_field=:curr_field""" % \
                {'fields': ','.join(['%s = :%s' % (f, f) for (f, c, t) in fields])}
        self.deleteQueryLegacy = """DELETE FROM %(table_arg)s
        WHERE sub_issue_id=:sub_issue_id AND dt=:dt"""
        self.deleteQuery = """DELETE FROM %(table_arg)s
        WHERE sub_issue_id=:sub_issue_id AND dt=:dt AND curr_field=:curr_field"""

    def nuke(self, tableName):
        if not hasattr(self, 'nukeList'):
            self.nukeList = list()
        if tableName not in self.nukeList:
            self.modelDB.dbCursor.execute("""TRUNCATE TABLE %s""" % tableName)
            logging.info('Deleting all records for %s', tableName)
            self.nukeList.append(tableName)

    def deleteData(self, tableName, subIssues, date, currency):
        if currency is None:
            deleteDicts = [dict([
                ('sub_issue_id', sid.getSubIDString()), ('dt', date)]) for sid in subIssues]
            deleteQuery = self.deleteQueryLegacy % {'table_arg': tableName}
        else:
            deleteDicts = [dict([
                ('sub_issue_id', sid.getSubIDString()), ('dt', date), ('curr_field', currency)
                ]) for sid in subIssues]
            deleteQuery = self.deleteQuery % {'table_arg': tableName}
        logging.info('Deleting %d records', len(deleteDicts))
        self.modelDB.dbCursor.executemany(deleteQuery, deleteDicts)

    def isDifferent(self, val1, val2):
        return ((val1 is None and val2 is not None)
                or (val1 is not None and val2 is None)
                or (val1 is not None and val2 is not None and abs(val1-val2) > 1e-8))

    def bulkProcess(self, date, sidList, valueArray, tableName, currency_field=None):
        self.log.info('Now working on table: %s, currency_field is %s', tableName, currency_field)

        if currency_field is None:
            currentBulkQuery = self.currentBulkQueryLegacy % {'table_arg': tableName}
            insertQuery = self.insertQueryLegacy % {'table_arg': tableName}
            updateQuery = self.updateQueryLegacy % {'table_arg': tableName}
            myDict = {'dt_arg': date}
        else:
            currentBulkQuery = self.currentBulkQuery % {'table_arg': tableName}
            insertQuery = self.insertQuery % {'table_arg': tableName}
            updateQuery = self.updateQuery % {'table_arg': tableName}
            myDict = {'dt_arg': date, 'curr_arg': currency_field}
        updateDicts = list()
        insertDicts = list()

        cur = self.modelDB.dbCursor
        cur.execute(currentBulkQuery, myDict)
        currentDict = dict((i[0], i[1:]) for i in cur.fetchall())

        for (idx, sid) in enumerate(sidList):
            sidstr = sid.getSubIDString()
            # Pull up what's already there if anything
            curVals = currentDict.get(sidstr, None)
            # Set up dict of items to be added/updated
            valueDict = dict()
            valueDict['sub_issue_id'] = sidstr
            valueDict['dt']= date
            if currency_field is not None:
                valueDict['curr_field'] = currency_field
            lenVD = len(valueDict)
            different = False
            for (jdx, nm) in enumerate(self.descIDList):
                val = valueArray[idx, jdx]
                if val is ma.masked:
                    valueDict[nm] = None
                else:
                    valueDict[nm] = val
                if curVals is not None and self.isDifferent(valueDict[nm], curVals[jdx]):
                    different = True

            if len(valueDict) > lenVD:
                if sid.getSubIDString() in currentDict:
                    if different:
                        updateDicts.append(valueDict)
                else:
                    insertDicts.append(valueDict)

        if len(insertDicts) > 0:
            self.log.info('Inserting %d new records', len(insertDicts))
            cur.executemany(insertQuery, insertDicts)
        if len(updateDicts) > 0:
            self.log.info('Updating %d records', len(updateDicts))
            cur.executemany(updateQuery, updateDicts)

class MarketPortfolioProcessor(BaseProcessor):
    def __init__(self, connections, gp=None):
        fields = [('value', True, float)]
        BaseProcessor.__init__(self, fields, None)
        self.modelDB = connections.modelDB
        self.log = self.modelDB.log
        self.currentBulkQuery = """SELECT sub_issue_id, %(fields)s
           FROM rmg_market_portfolio WHERE dt=:dt AND rmg_id=:rmg_id""" \
            % { 'fields': self.fieldList}
        self.insertQuery = """INSERT INTO rmg_market_portfolio
        (rmg_id, dt, sub_issue_id, %(fields)s)
        VALUES(:rmg_id, :dt, :sub_issue_id, %(args)s)""" % {
            'fields': self.fieldList, 'args': self.argList}
        self.updateQuery = """UPDATE rmg_market_portfolio SET %(fields)s
        WHERE rmg_id=:rmg_id AND dt=:dt AND sub_issue_id=:sub_issue_id""" % {
            'fields': ','.join(['%s = :%s' % (f, f) for (f, c, t)
                                in fields])}
        self.deleteQuery = """DELETE FROM rmg_market_portfolio
        WHERE rmg_id=:rmg_id AND dt=:dt AND sub_issue_id=:sub_issue_id"""
    
    def bulkProcess(self, dateList, rmgList, values, restrictSet):
        self.log.debug('MarketPortfolioProcessor')
        if restrictSet is not None:
            raise ValueError('restrictSet not supported')
        cur = self.modelDB.dbCursor
        updateDicts = list()
        insertDicts = list()
        deleteDicts = list()
        for (dIdx, date) in enumerate(dateList):
            myDict = {'dt': date}
            for (rIdx, rmgId) in enumerate(rmgList):
                myDict['rmg_id'] = rmgId
                cur.execute(self.currentBulkQuery, myDict)
                currentDict = dict(cur.fetchall())
                valuePairs = list()
                if values[dIdx, rIdx] is not None:
                    valuePairs = values[dIdx, rIdx].valuePairs
                for (sid, value) in valuePairs:
                    sid = sid.getSubIDString()
                    if sid in currentDict:
                        if currentDict[sid] is None or abs(currentDict[sid]-value) > 1e-8:
                            updateDicts.append({'sub_issue_id': sid,
                                                'rmg_id': rmgId,
                                                'dt': date,
                                                'value': value})
                    else:
                        insertDicts.append({'sub_issue_id': sid,
                                            'rmg_id': rmgId,
                                            'dt': date,
                                            'value': value})
                for sid in set(currentDict.keys()) - \
                        set(sid.getSubIDString() for sid, value in valuePairs):
                    deleteDicts.append({
                            'sub_issue_id': sid, 'rmg_id': rmgId, 'dt': date})
        if len(insertDicts) > 0:
            cur.setinputsizes(**self.insertInputSizes)
            cur.executemany(self.insertQuery, insertDicts)
            self.log.info('Inserting %d new records', len(insertDicts))
        if len(updateDicts) > 0:
            cur.setinputsizes(**self.insertInputSizes)
            cur.executemany(self.updateQuery, updateDicts)
            self.log.info('Updating %d records', len(updateDicts))
        if len(deleteDicts) > 0:
            cur.executemany(self.deleteQuery, deleteDicts)
            self.log.info('Deleting %d records', len(deleteDicts))

class MarketReturnProcessor(BaseProcessor):
    def __init__(self, connections, robust=False, gp=None):
        fields = [('value', True, float)]
        BaseProcessor.__init__(self, fields, None)
        self.modelDB = connections.modelDB
        self.log = self.modelDB.log
        self.robust = robust
        if robust:
            self.tableName = 'rmg_market_return_v3'
        else:
            self.tableName = 'rmg_market_return'
        self.currentBulkQuery = """SELECT %(fields)s
           FROM %(table)s WHERE dt=:dt AND rmg_id=:rmg_id""" \
            % { 'fields': self.fieldList, 'table': self.tableName}
        self.insertQuery = """INSERT INTO %(table)s
        (rmg_id, dt, %(fields)s) VALUES(:rmg_id, :dt, %(args)s)""" % {
            'fields': self.fieldList, 'args': self.argList,
            'table': self.tableName}
        self.updateQuery = """UPDATE %(table)s SET %(fields)s
        WHERE rmg_id=:rmg_id AND dt=:dt""" % {
            'fields': ','.join(['%s = :%s' % (f, f) for (f, c, t) in fields]),
            'table': self.tableName}
        self.deleteQuery = """DELETE FROM %(table)s
        WHERE rmg_id=:rmg_id AND dt=:dt""" % {'table': self.tableName}
    
    def bulkProcess(self, dateList, rmgList, values, restrictSet):
        self.log.debug('MarketReturnProcessor')
        if restrictSet is not None:
            raise ValueError('restrictSet not supported')
        cur = self.modelDB.dbCursor
        updateDicts = list()
        insertDicts = list()
        deleteDicts = list()
        for (dIdx, date) in enumerate(dateList):
            myDict = {'dt': date}
            for (rIdx, rmgId) in enumerate(rmgList):
                myDict['rmg_id'] = rmgId
                cur.execute(self.currentBulkQuery, myDict)
                currentValue = cur.fetchone()
                value = values[dIdx, rIdx]
                if value is not None:
                    value = value.value
                if currentValue is None:
                    if value is not None:
                        insertDicts.append({
                                'rmg_id': rmgId, 'dt': date, 'value': value})
                elif value is None:
                    deleteDicts.append({'rmg_id': rmgId, 'dt': date})
                else:
                    if currentValue[0] is None or abs(currentValue[0]-value) > 1e-8:
                        updateDicts.append({
                            'rmg_id': rmgId, 'dt': date, 'value': value})
        if len(insertDicts) > 0:
            cur.setinputsizes(**self.insertInputSizes)
            cur.executemany(self.insertQuery, insertDicts)
            self.log.info('Inserting %d new records', len(insertDicts))
        if len(updateDicts) > 0:
            cur.setinputsizes(**self.insertInputSizes)
            cur.executemany(self.updateQuery, updateDicts)
            self.log.info('Updating %d records', len(updateDicts))
        if len(deleteDicts) > 0:
            cur.executemany(self.deleteQuery, deleteDicts)
            self.log.info('Deleting %d records', len(deleteDicts))

class AMPIndustryReturnProcessor(BaseProcessor):
    def __init__(self, connections, robust=False, gp=None):
        fields = [('value', True, float)]
        BaseProcessor.__init__(self, fields, None)
        self.modelDB = connections.modelDB
        self.industryIdDict = gp.industryIdDict
        self.revision_id  = gp.revision_id 
        self.log = self.modelDB.log
        self.robust = robust
        self.tableName = 'amp_industry_return'
        self.currentBulkQuery = """
           SELECT %(fields)s
           FROM %(table)s 
           WHERE dt=:dt AND 
              mdl_port_member_id=:mdl_port_member_id AND
              revision_id=:revision_id AND
              ref_id=:ref_id
            """ \
            % { 'fields': self.fieldList, 'table': self.tableName}
        self.insertQuery = """INSERT INTO %(table)s
        (mdl_port_member_id, revision_id, ref_id, dt, %(fields)s) 
        VALUES(:mdl_port_member_id, :revision_id, :ref_id, :dt, %(args)s)""" % {
            'fields': self.fieldList, 'args': self.argList,
            'table': self.tableName}
        self.updateQuery = """
            UPDATE %(table)s SET %(fields)s
            WHERE dt=:dt AND 
              mdl_port_member_id=:mdl_port_member_id AND
              revision_id=:revision_id AND
              ref_id=:ref_id
            """ % {
            'fields': ','.join(['%s = :%s' % (f, f) for (f, c, t) in fields]),
            'table': self.tableName}
        self.deleteQuery = """
            DELETE FROM %(table)s
            WHERE dt=:dt AND 
              mdl_port_member_id=:mdl_port_member_id AND
              revision_id=:revision_id AND
              ref_id=:ref_id
            """ % {'table': self.tableName}
    
    def bulkProcess(self, amp, dateList, industryList, values):
        self.log.debug('AMPIndustryReturnProcessor')
        cur = self.modelDB.dbCursor
        updateDicts = list()
        insertDicts = list()
        deleteDicts = list()
        for (dIdx, date) in enumerate(dateList):
            for (indIdx, industry) in enumerate(industryList):
                myDict = {
                        'dt': date, 
                        'mdl_port_member_id': amp.id, 
                        'revision_id': self.revision_id, 
                        'ref_id': self.industryIdDict[industry]
                }
                cur.execute(self.currentBulkQuery, myDict)
                currentValue = cur.fetchone()
                value = values[dIdx, indIdx]
                if currentValue is None:
                    if value is not None:
                        myDict['value'] = value
                        insertDicts.append(myDict)
                elif value is None:
                    deleteDicts.append(myDict)
                else:
                    if currentValue[0] is None or abs(currentValue[0]-value) > 1e-8:
                        myDict['value'] = value
                        updateDicts.append(myDict)
        if len(insertDicts) > 0:
            cur.setinputsizes(**self.insertInputSizes)
            cur.executemany(self.insertQuery, insertDicts)
            self.log.info('Inserting %d new records', len(insertDicts))
        if len(updateDicts) > 0:
            cur.setinputsizes(**self.insertInputSizes)
            cur.executemany(self.updateQuery, updateDicts)
            self.log.info('Updating %d records', len(updateDicts))
        if len(deleteDicts) > 0:
            cur.executemany(self.deleteQuery, deleteDicts)
            self.log.info('Deleting %d records', len(deleteDicts))

class RegionReturnProcessor(BaseProcessor):
    def __init__(self, connections, robust=False, gp=None):
        fields = [('value', True, float)]
        BaseProcessor.__init__(self, fields, None)
        self.modelDB = connections.modelDB
        self.log = self.modelDB.log
        self.tableName = 'region_return'
        self.currentBulkQuery = """SELECT %(fields)s
           FROM %(table)s WHERE dt=:dt AND id=:id""" \
            % { 'fields': self.fieldList, 'table': self.tableName}
        self.insertQuery = """INSERT INTO %(table)s
        (id, dt, %(fields)s) VALUES(:id, :dt, %(args)s)""" % {
            'fields': self.fieldList, 'args': self.argList,
            'table': self.tableName}
        self.updateQuery = """UPDATE %(table)s SET %(fields)s
        WHERE id=:id AND dt=:dt""" % {
            'fields': ','.join(['%s = :%s' % (f, f) for (f, c, t) in fields]),
            'table': self.tableName}
        self.deleteQuery = """DELETE FROM %(table)s
        WHERE id=:id AND dt=:dt""" % {'table': self.tableName}

    def bulkProcess(self, dateList, regList, values, restrictSet):
        self.log.debug('RegionReturnProcessor')
        if restrictSet is not None:
            raise ValueError('restrictSet not supported')
        cur = self.modelDB.dbCursor
        updateDicts = list()
        insertDicts = list()
        deleteDicts = list()
        for (dIdx, date) in enumerate(dateList):
            myDict = {'dt': date}
            for (rIdx, regId) in enumerate(regList):
                myDict['id'] = regId
                cur.execute(self.currentBulkQuery, myDict)
                currentValue = cur.fetchone()
                value = values[dIdx, rIdx]
                if value is not None:
                    value = value.value
                if currentValue is None:
                    if value is not None:
                        insertDicts.append({
                                'id': regId, 'dt': date, 'value': value})
                elif value is None:
                    deleteDicts.append({'id': regId, 'dt': date})
                else:
                    if currentValue[0] is None or abs(currentValue[0]-value) > 1e-8:
                        updateDicts.append({
                            'id': regId, 'dt': date, 'value': value})
        if len(insertDicts) > 0:
            cur.setinputsizes(**self.insertInputSizes)
            cur.executemany(self.insertQuery, insertDicts)
            self.log.info('Inserting %d new records', len(insertDicts))
        if len(updateDicts) > 0:
            cur.setinputsizes(**self.insertInputSizes)
            cur.executemany(self.updateQuery, updateDicts)
            self.log.info('Updating %d records', len(updateDicts))
        if len(deleteDicts) > 0:
            cur.executemany(self.deleteQuery, deleteDicts)
            self.log.info('Deleting %d records', len(deleteDicts))

class MarketVolatilityProcessor(BaseProcessor):
    def __init__(self, connections, gp=None):
        fields = [('value', True, float)]
        BaseProcessor.__init__(self, fields, None)
        self.modelDB = connections.modelDB
        self.log = self.modelDB.log
        self.tableName = 'rmg_market_volatility'
        self.currentBulkQuery = """SELECT %(fields)s
           FROM %(table)s WHERE dt=:dt AND rmg_id=:rmg_id""" \
            % { 'fields': self.fieldList, 'table': self.tableName}
        self.insertQuery = """INSERT INTO %(table)s
        (rmg_id, dt, %(fields)s) VALUES(:rmg_id, :dt, %(args)s)""" % {
            'fields': self.fieldList, 'args': self.argList,
            'table': self.tableName}
        self.updateQuery = """UPDATE %(table)s SET %(fields)s
        WHERE rmg_id=:rmg_id AND dt=:dt""" % {
            'fields': ','.join(['%s = :%s' % (f, f) for (f, c, t) in fields]),
            'table': self.tableName}
        self.deleteQuery = """DELETE FROM %(table)s
        WHERE rmg_id=:rmg_id AND dt=:dt""" % {'table': self.tableName}

    def bulkProcess(self, dateList, rmgList, values, restrictSet):
        self.log.debug('MarketReturnProcessor')
        if restrictSet is not None:
            raise ValueError('restrictSet not supported')
        cur = self.modelDB.dbCursor
        updateDicts = list()
        insertDicts = list()
        deleteDicts = list()
        for (dIdx, date) in enumerate(dateList):
            myDict = {'dt': date}
            for (rIdx, rmgId) in enumerate(rmgList):
                myDict['rmg_id'] = rmgId
                cur.execute(self.currentBulkQuery, myDict)
                currentValue = cur.fetchone()
                value = values[dIdx, rIdx]
                if value is not None:
                    value = value.value
                    if math.isnan(value):
                        value = None
                if currentValue is None:
                    if value is not None:
                        insertDicts.append({
                                'rmg_id': rmgId, 'dt': date, 'value': value})
                elif value is None:
                    deleteDicts.append({'rmg_id': rmgId, 'dt': date})
                elif abs(currentValue[0]-value) > 1e-8:
                    updateDicts.append({
                            'rmg_id': rmgId, 'dt': date, 'value': value})
        if len(insertDicts) > 0:
            cur.setinputsizes(**self.insertInputSizes)
            cur.executemany(self.insertQuery, insertDicts)
            self.log.info('Inserting %d new records', len(insertDicts))
        if len(updateDicts) > 0:
            cur.setinputsizes(**self.insertInputSizes)
            cur.executemany(self.updateQuery, updateDicts)
            self.log.info('Updating %d records', len(updateDicts))
        if len(deleteDicts) > 0:
            cur.executemany(self.deleteQuery, deleteDicts)
            self.log.info('Deleting %d records', len(deleteDicts))

class ReturnsTimingProcessorLegacy(BaseProcessor):
    def __init__(self, connections, timingId, gp=None):
        fields = [('value', True, float)]
        BaseProcessor.__init__(self, fields, None)
        self.modelDB = connections.modelDB
        self.log = self.modelDB.log
        self.timingId = int(timingId)
        self.tableName = 'returns_timing_adjustment'
        self.currentBulkQuery = """SELECT %(fields)s
           FROM %(table)s WHERE dt=:dt AND rmg_id=:rmg_id
           AND timing_id=:timing_id""" \
            % { 'fields': self.fieldList, 'table': self.tableName}
        self.insertQuery = """INSERT INTO %(table)s
           (timing_id, rmg_id, dt, %(fields)s)
           VALUES(:timing_id, :rmg_id, :dt, %(args)s)""" % {
            'fields': self.fieldList, 'args': self.argList,
            'table': self.tableName}
        self.updateQuery = """UPDATE %(table)s SET %(fields)s
           WHERE timing_id=:timing_id AND rmg_id=:rmg_id AND dt=:dt""" % {
            'fields': ','.join(['%s = :%s' % (f, f) for (f, c, t) in fields]),
            'table': self.tableName}
        self.deleteQuery = """DELETE FROM %(table)s
           WHERE timing_id=:timing_id AND rmg_id=:rmg_id AND dt=:dt""" % {
            'table': self.tableName}

    def bulkProcess(self, dateList, rmgList, values, restrictSet):
        self.log.debug('ReturnsTimingProcessor')
        if restrictSet is not None:
            raise ValueError('restrictSet not supported')
        cur = self.modelDB.dbCursor
        updateDicts = list()
        insertDicts = list()
        deleteDicts = list()
        for (dIdx, date) in enumerate(dateList):
            myDict = {'dt': date, 'timing_id': self.timingId}
            for (rIdx, rmgId) in enumerate(rmgList):
                myDict['rmg_id'] = rmgId
                cur.execute(self.currentBulkQuery, myDict)
                currentValue = cur.fetchone()
                value = values[dIdx, rIdx]
                if value is not None:
                    value = value.value
                if currentValue is None:
                    if value is not None:
                        insertDicts.append({
                                'timing_id': self.timingId,
                                'rmg_id': rmgId, 'dt': date, 'value': value})
                elif value is None:
                    deleteDicts.append({'timing_id': self.timingId,
                                        'rmg_id': rmgId, 'dt': date})
                elif abs(currentValue[0]-value) > 1e-8:
                    updateDicts.append({
                            'timing_id': self.timingId,
                            'rmg_id': rmgId, 'dt': date, 'value': value})
        if len(insertDicts) > 0:
            cur.setinputsizes(**self.insertInputSizes)
            cur.executemany(self.insertQuery, insertDicts)
            self.log.info('Inserting %d new records', len(insertDicts))
        if len(updateDicts) > 0:
            cur.setinputsizes(**self.insertInputSizes)
            cur.executemany(self.updateQuery, updateDicts)
            self.log.info('Updating %d records', len(updateDicts))
        if len(deleteDicts) > 0:
            cur.executemany(self.deleteQuery, deleteDicts)
            self.log.info('Deleting %d records', len(deleteDicts))

class ReturnsTimingProcessor(BaseProcessor):
    def __init__(self, connections, timingId, gp=None):
        fields = [('value', True, float), ('proxy', True, float)]
        BaseProcessor.__init__(self, fields, None)

        # Initialise
        self.modelDB = connections.modelDB
        self.log = self.modelDB.log
        self.timingId = int(timingId)
        self.tableName = 'rmg_returns_timing_adj'

        # Set up DB queries
        self.currentBulkQuery = """SELECT %(fields)s
           FROM %(table)s WHERE dt=:dt AND rmg_id=:rmg_id
           AND timing_id=:timing_id""" \
            % { 'fields': self.fieldList, 'table': self.tableName}
        self.insertQuery = """INSERT INTO %(table)s
           (timing_id, rmg_id, dt, %(fields)s)
           VALUES(:timing_id, :rmg_id, :dt, %(args)s)""" % {
            'fields': self.fieldList, 'args': self.argList,
            'table': self.tableName}
        self.updateQuery = """UPDATE %(table)s SET %(fields)s
           WHERE timing_id=:timing_id AND rmg_id=:rmg_id AND dt=:dt""" % {
            'fields': ','.join(['%s = :%s' % (f, f) for (f, c, t) in fields]),
            'table': self.tableName}
        self.deleteQuery = """DELETE FROM %(table)s
           WHERE timing_id=:timing_id AND rmg_id=:rmg_id AND dt=:dt""" % {
            'table': self.tableName}

    def isDifferent(self, val1, val2, tol=1.0e-8):
        return ((val1 is None and val2 is not None)
                or (val1 is not None and val2 is None)
                or (val1 is not None and val2 is not None and abs(val1-val2) > tol))

    def bulkProcess(self, dateList, rmgList, values, restrictSet):
        self.log.debug('ReturnsTimingProcessor')
        if restrictSet is not None:
            raise ValueError('restrictSet not supported')
        # Initialise
        cur = self.modelDB.dbCursor
        updateDicts = list()
        insertDicts = list()
        deleteDicts = list()

        # Loop round dates
        for (dIdx, date) in enumerate(dateList):
            myDict = {'dt': date, 'timing_id': self.timingId}

            # Loop round RMGs
            for (rIdx, rmgId) in enumerate(rmgList):
                myDict['rmg_id'] = rmgId

                # Check what's already in the DB
                cur.execute(self.currentBulkQuery, myDict)
                currentValue = cur.fetchone()

                # Check what we've just computed
                valueObj = values[dIdx, rIdx]
                if valueObj is None:
                    value = None
                    proxy = None
                else:
                    value = valueObj.value
                    proxy = valueObj.proxy
                if value is None and proxy is not None:
                    value = 0.0
                elif proxy is None and value is not None:
                    proxy = 0.0

                if currentValue is None:
                    # Flag data to be added
                    if (value is not None) and (proxy is not None):
                        insertDicts.append({
                                'timing_id': self.timingId, 'rmg_id': rmgId,
                                'dt': date, 'value': value, 'proxy': proxy})
                elif (value is None) and (proxy is None):
                    # Flag entry to be deleted
                    deleteDicts.append({'timing_id': self.timingId,
                                        'rmg_id': rmgId, 'dt': date})

                elif self.isDifferent(currentValue[0], value) or \
                        self.isDifferent(currentValue[1], proxy):
                    # Flag data to be updated
                    updateDicts.append({
                            'timing_id': self.timingId, 'rmg_id': rmgId,
                            'dt': date, 'value': value, 'proxy': proxy})

        # Perform DB operations
        if len(deleteDicts) > 0:
            cur.executemany(self.deleteQuery, deleteDicts)
            self.log.info('Deleting %d records', len(deleteDicts))
        if len(insertDicts) > 0:
            cur.setinputsizes(**self.insertInputSizes)
            cur.executemany(self.insertQuery, insertDicts)
            self.log.info('Inserting %d new records', len(insertDicts))
        if len(updateDicts) > 0:
            cur.setinputsizes(**self.insertInputSizes)
            cur.executemany(self.updateQuery, updateDicts)
            self.log.info('Updating %d records', len(updateDicts))
        return

class RiskFreeRateProcessor(BaseProcessor):
    """Process data into currency_risk_free rates
    """
    def __init__(self, connections):
        fields = [('value', True, float)]
        BaseProcessor.__init__(self, fields, None)
        self.modelDB = connections.modelDB
        self.log = self.modelDB.log
        self.currentBulkQuery = """SELECT currency_code, %(fields)s
           FROM currency_risk_free_rate
           WHERE dt=:dt_arg""" % { 'fields': self.fieldList}
        self.insertQuery = """INSERT INTO currency_risk_free_rate
        (currency_code, dt, %(fields)s)
        VALUES(:currency_code, :dt, %(args)s)""" % {
            'fields': self.fieldList, 'args': self.argList}
        self.updateQuery = """UPDATE currency_risk_free_rate SET %(fields)s
        WHERE currency_code=:currency_code AND dt=:dt""" % {
            'fields': ','.join(['%s = :%s' % (f, f) for (f, c, t)
                                   in fields])}
        self.deleteQuery = """DELETE FROM currency_risk_free_rate
        WHERE currency_code=:currency_code AND dt=:dt"""
    
    def bulkProcess(self, dateList, currencyList, valueList):
        self.log.debug('RiskFreeRateProcessor')
        cur = self.modelDB.dbCursor
        updateDicts = list()
        insertDicts = list()
        deleteDicts = list()
        for (dIdx, date) in enumerate(dateList):
            myDict = {'dt_arg': date}
            cur.execute(self.currentBulkQuery, myDict)
            currentDict = dict(cur.fetchall())
            for (cIdx, (currency, value)) in enumerate(
                zip(currencyList, valueList[dIdx])):
                if value is not None:
                    value = value.value
                if value is not None:
                    if currency in currentDict:
                        if abs(currentDict[currency]-value) > 1e-8:
                            updateDicts.append({'currency_code': currency,
                                                'dt': date,
                                                'value': value})
                if value is not None:
                    if currency in currentDict:
                        if abs(currentDict[currency]-value) > 1e-8:
                            updateDicts.append({'currency_code': currency,
                                                'dt': date,
                                                'value': value})
                    else:
                        insertDicts.append({'currency_code': currency,
                                            'dt': date, 'value': value})
                elif currency in currentDict:
                    deleteDicts.append({'currency_code': currency,
                                        'dt': date})
        if len(insertDicts) > 0:
            cur.setinputsizes(**self.insertInputSizes)
            cur.executemany(self.insertQuery, insertDicts)
            self.log.info('Inserting %d new records', len(insertDicts))
        if len(updateDicts) > 0:
            cur.setinputsizes(**self.insertInputSizes)
            cur.executemany(self.updateQuery, updateDicts)
            self.log.info('Updating %d records', len(updateDicts))
        if len(deleteDicts) > 0:
            cur.executemany(self.deleteQuery, deleteDicts)
            self.log.info('Deleting %d records', len(deleteDicts))

class CumulativeRiskFreeRateProcessor(BaseProcessor):
    """Process cumulative values into currency_risk_free rates
    """
    def __init__(self, connections):
        fields = [('cumulative', True, float)]
        BaseProcessor.__init__(self, fields, None)
        self.modelDB = connections.modelDB
        self.log = self.modelDB.log
        self.currentBulkQuery = """SELECT currency_code, %(fields)s
           FROM currency_risk_free_rate
           WHERE dt=:dt_arg""" % {
            'fields': self.fieldList}
        self.updateQuery = """UPDATE currency_risk_free_rate SET %(fields)s
        WHERE currency_code=:currency_code AND dt=:dt""" % {
            'fields': ','.join(['%s = :%s' % (f, f) for (f, c, t)
                                   in fields])}

    def bulkProcess(self, dateList, currencyList, valueList):
        self.log.debug('RiskFreeRateProcessor')
        cur = self.modelDB.dbCursor
        updateDicts = list()
        for (dIdx, date) in enumerate(dateList):
            myDict = {'dt_arg': date}
            cur.execute(self.currentBulkQuery, myDict)
            currentDict = dict(cur.fetchall())
            for (cIdx, (currency, value)) in enumerate(
                zip(currencyList, valueList[dIdx])):
                if value is not None:
                    value = value.cumulative
                if value is not None:
                    if currency in currentDict:
                        if currentDict[currency] is None or \
                                abs(currentDict[currency]-value) > 1e-8:
                            updateDicts.append({'currency_code': currency,
                                                'dt': date,
                                                'cumulative': value})
                    else:
                        self.log.warning('Cumulative rfr for %s on %s but no rfr',
                                      currency, date)
                elif currency in currentDict \
                         and currentDict[currency] is not None:
                    updateDicts.append({'currency_code': currency,
                                        'dt': date,
                                        'cumulative': None})
        if len(updateDicts) > 0:
            cur.setinputsizes(**self.insertInputSizes)
            cur.executemany(self.updateQuery, updateDicts)
            self.log.info('Updating %d records', len(updateDicts))


# Helper to create modelDB and marketDB
# to make things easier for interactive use
# Just set the usual environment variables to point
# to the right database
def loadModelDB(sid=None, useGolden=False):
    """Can specify sid from ('glprodsb', 'glprod', 'research', 'glsdg', 'researchoda','research_vital')
       or if None, then will read from environment variables
    """
    def load(sid=sid):
        if sid == 'research_vital':
            mdldb = 'modeldb_vital'
            mktdb = 'marketdb_vital'
            real_sid = 'researchoda'
        else:
            mdldb = 'modeldb_global'
            mktdb = 'marketdb_global'
            real_sid = sid
        if useGolden:
            mdldb = mdldb.replace('global', 'golden')
            mktdb = mktdb.replace('global', 'golden')

        modelDB = ModelDB(sid=real_sid, user=mdldb,passwd=mdldb)
        marketDB = MarketDB.MarketDB(sid=real_sid, user=mktdb,passwd=mktdb)
        return modelDB, marketDB
    # loadModelDB function starts here
    if sid is not None:
        if sid not in ('glprodsb', 'glprod', 'research', 'glsdg', 'researchoda', 'research_vital', 'freshres'):
            raise Exception('Unknown sid: %s' % sid)
        return load(sid=sid)
    cmdlineParser = optparse.OptionParser()
    Utilities.addModelAndDefaultCommandLine(cmdlineParser)
    (options, args) = cmdlineParser.parse_args([])
    modelDB = ModelDB(sid=options.modelDBSID, 
                          user=options.modelDBUser,
                          passwd=options.modelDBPasswd)
    marketDB = MarketDB.MarketDB(sid=options.marketDBSID, 
                             user=options.marketDBUser,
                             passwd=options.marketDBPasswd)
    return modelDB, marketDB

