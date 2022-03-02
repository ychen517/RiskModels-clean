
import logging
import optparse
import configparser
import datetime
import numpy
try : 
    import numpy.ma as ma
except:
    import numpy.core.ma as ma
from riskmodels import ModelDB
from riskmodels import LegacyUtilities as Utilities
from riskmodels import StyleExposures
from riskmodels import GlobalExposures
from riskmodels import Matrices

class DataFrequency:
    """Class representing frequency of fundamental data.
    """
    def __init__(self, suffix):
        self.suffix = suffix
    def __eq__(self, other):
        if other is None:
            return False
        assert(self.__class__.__name__ == other.__class__.__name__)
        return self.suffix == other.suffix
    def __repr__(self):
        return 'DataFrequency(%s)' %self.suffix

def computeAnnualDividends(modelDate, subIssues, model,
                             modelDB, marketDB, maskMissing=False,
                             includeSpecial=True, includeStock=False):
    """Computes total dividends paid for the past year, defined as
    reported dividends times the shares outstanding on the pay date.
    Fuzzy logic employed to account for periodic (annual, quarterly, etc)
    dividends that are not spaced apart in exactly equal time intervals.
    """

    # Look back slightly more than a year
    prevYear = modelDate - datetime.timedelta(380)

    # Load dividends from DB
    paidDividends = modelDB.getPaidDividends(
        prevYear, modelDate, subIssues, marketDB,
        convertTo=model.numeraire.currency_id,
        includeSpecial=includeSpecial, includeStock=includeStock)

    # Keep only one year's worth of divdends paid, via aforementioned fuzzy logic
    for (sid, divData) in paidDividends.items():
        if len(divData)==0:
            continue
        highFreq = False
        if len(divData) >= 11:
            highFreq = True
        lastExDate = divData[-1][0]
        toRemove = list()
        if lastExDate >= modelDate - datetime.timedelta(30):
            for (dt, div) in divData:
                if not highFreq and dt < prevYear + datetime.timedelta(28):
                    toRemove.append((dt, div))
                elif highFreq:
                    tolerancePeriod = [lastExDate - datetime.timedelta(365) + \
                                       datetime.timedelta(t) for t in [-4,4]]
                    if dt >= tolerancePeriod[0] and dt <= tolerancePeriod[1]:
                        toRemove.append((dt, div))
        for n in toRemove:
            divData.remove(n)

    # Mask out missing dividends and sum total
    if maskMissing:
        paidDividendSum = Matrices.allMasked(len(subIssues))
        for (idx, sid) in enumerate(subIssues):
            divData = paidDividends.get(sid, list())
            if len(divData) > 0:
                paidDividendSum[idx] = ma.sum([dv[1] for dv in divData])
    else:
        paidDividendSum = numpy.array([ma.sum([i[1] for i in
            paidDividends.get(sid, list())])
            for sid in subIssues])

    return paidDividendSum

def populateValueFrequencyList(data, freq=None):
    """Return a list whose entries are Structs, each containing
    the corresponding value in data, along with its associated
    DataFrequency type.
    """
    values = Matrices.allMasked(len(data), dtype=object)
    for i in range(len(data)):
        val = Utilities.Struct()
        if freq is None:
            val.frequency = FinancialRatio.QuarterlyFrequency
        else:
            try:
                val.frequency = freq[i]
            except:
                val.frequency = freq
        val.value = data[i]
        values[i] = val
    return values

class FinancialRatio:
    AnnualFrequency = DataFrequency('_ann')
    QuarterlyFrequency = DataFrequency('_qtr')
    MissingFrequency = DataFrequency(None)

    def __init__(self, numItem, denomItem, modelDB, marketDB, 
                 useFixedFrequency=None,
                 numeratorDaysBack=2*365, denomDaysBack=2*365,
                 numeratorEndDate=0, denomEndDate=0, 
                 numeratorProcess='extractlatest', denomProcess='extractlatest', 
                 denomZeroTreatment=None,
                 numeratorNegativeTreatment=None, denomNegativeTreatment=None, 
                 numeratorSplitAdjust=None, denomSplitAdjust=None,
                 numeratorMultiplier=1e6, denomMultiplier=1e6):
        """Represnts the a generic financial ratio.
        denom/numeratorDaysBack = days back to start looking for data
        denom/numeratorEndDate = days back to end looking for data (0 = present day)
        denom/numeratorProcess = 'extractlatest' (balance sheet) or 'annualize' (income statement)
        denom/numeratorNegativeTreatment = None/'zero'/'mask'
        denom/numeratorSplitAdjust = None/'divide'/'multiply'
        denom/numeratorMultiplier = rescale data values by constant
        """
        self.numItem = numItem
        self.denomItem = denomItem
        self.marketDB = marketDB
        self.modelDB = modelDB
        self.numDaysBack = numeratorDaysBack
        self.denomDaysBack = denomDaysBack
        self.numEndDate = numeratorEndDate
        self.denomEndDate = denomEndDate
        self.numProcess = numeratorProcess
        self.denomProcess = denomProcess
        self.numNegativeTreatment = numeratorNegativeTreatment
        self.denomZeroTreatment = denomZeroTreatment
        self.denomNegativeTreatment = denomNegativeTreatment
        self.numSplitAdjust = numeratorSplitAdjust
        self.denomSplitAdjust = denomSplitAdjust
        self.numMultiplier = numeratorMultiplier
        self.denomMultiplier = denomMultiplier
        assert(useFixedFrequency is None or isinstance(useFixedFrequency, DataFrequency))
        self.useFixedFrequency = useFixedFrequency

    def loadRawDenominatorData(self, code, modelDate, startDate, endDate, subids, currency_id,
                               allowMixedFreq=True):
        """Retrieve raw data for the given denominator data item.
        Subclasses of FinancialRatio that use 'complex' denominators 
        beyond simple items of financial reporting data should define their own 
        loadRawDenominatorData() method to perform whatever computations are required.
        """
        return self.loadRawFundamentalDataInternal(code, modelDate, startDate, endDate, 
                            subids, currency_id, allowMixedFreq, isNumerator=False)

    def loadRawNumeratorData(self, code, modelDate, startDate, endDate, subids, currency_id,
                             allowMixedFreq=True):
        """Retrieve raw data for the given numerator data item.
        Subclasses of FinancialRatio that use 'complex' numerators 
        beyond simple items of financial reporting data should define their own 
        loadRawNumeratorData() method to perform whatever computations are required.
        """
        return self.loadRawFundamentalDataInternal(code, modelDate, startDate, endDate, 
                            subids, currency_id, allowMixedFreq, isNumerator=True)

    def loadRawFundamentalDataInternal(self, code, modelDate, startDate, endDate, 
                                subids, currency_id, allowMixedFreq, isNumerator):
        """Retrieve and process raw data for the given data item.
        Returns a list of Structs(), each containing the data value
        as well as its frequency.
        """
        # Fetch denominator or numerator options, depending on which we are loading
        if isNumerator:
            process = self.numProcess
            multiplier = self.numMultiplier
            splitAdjust = self.numSplitAdjust
        else:
            process = self.denomProcess
            multiplier = self.denomMultiplier
            splitAdjust = self.denomSplitAdjust
        frequencies = None

        # If using fixed frequency data, no not allow usage of mixed frequency data
        if self.useFixedFrequency is not None:
            allowMixedFreq = False

        # Default: load and process data from mixed frequencies
        if allowMixedFreq:
            (rawValues, frequencies) = self.modelDB.getMixFundamentalCurrencyItem(
                            code, startDate, endDate, subids, modelDate, self.marketDB, 
                            currency_id, splitAdjust=splitAdjust)

            #Here we need to figure whether it's quarterly or annual data
            data = ma.zeros(len(subids))
            qtrIdx = []

            if process not in ['annualize','extractlatest']:
                raise ValueError('Unsupported processing option %s'%process)

            if process == 'annualize':
                qtrIdx = numpy.flatnonzero(ma.getmaskarray(ma.masked_where(
                            frequencies==self.QuarterlyFrequency, frequencies)))
                if len(qtrIdx) > 0:
                    annualizedData = self.modelDB.annualizeQuarterlyValues(
                        [rawValues[k] for k in qtrIdx], adjustShortHistory=True)
                    ma.put(data, qtrIdx, annualizedData)

            #For those annual data or quarterly data but not process == annualize
            #use extractLatest 
            latestData = Utilities.extractLatestValue(
                [rawValues[j] for j in range(len(subids)) if j not in qtrIdx])
            latestIdx = 0
            for sIdx in range(len(subids)):
                if sIdx not in qtrIdx:
                    data[sIdx] = latestData[latestIdx]
                    latestIdx += 1
            
        # Otherwise, either mixing is disabled, or this is a 'secondary' run;
        # that is, fetching annual data only for certain items
        else:
            # If using fixed frequency (may still be quarterly)
            if self.useFixedFrequency is not None:
                suffix = self.useFixedFrequency.suffix
            # If secondary run, force annual
            else:
                suffix = self.AnnualFrequency.suffix
            data = self.modelDB.getFundamentalCurrencyItem(
                            code + suffix, startDate, endDate,
                            subids, modelDate, self.marketDB, currency_id,
                            splitAdjust=splitAdjust)
            if self.useFixedFrequency == self.QuarterlyFrequency \
                    and process == 'annualize':
                data = modelDB.annualizeQuarterlyValues(data, adjustShortHistory=True)
            else:
                data = Utilities.extractLatestValue(data)

        # Rescale if required
        if multiplier != 1.0:
            for idx in range(len(subids)):
                val = data[idx]
                data[idx] = val * multiplier
        # Return values along with associated frequencies
        if frequencies is not None:
            freq = frequencies
        else:
            if self.useFixedFrequency is not None:
                freq = self.useFixedFrequency
            else:
                freq = self.AnnualFrequency
        return populateValueFrequencyList(data, freq)
        
    def getRatio(self, subids, modelDate, currency_id):
        """Compute the financial ratio for the given assets on a given date. 
        The numItem/denomItem should be prefix of fundamental items you want.
        The code will first search for the most frequently updated data for both numerator and 
        denominator. If it sees that the numerator and denominator are in different freqency,
        it will make them both annual data to come up wth the ratio. 
        Returns an array with desired ratio and another array of frequency indicating 
        the frequency used in each value returned.
        """
        logging.info('getRatio for %s/%s: begin', self.numItem, self.denomItem)

        assetIdxMap = dict([(j,i) for (i,j) in enumerate(subids)])

        # Load numerator data
        num_startDate = modelDate - datetime.timedelta(self.numDaysBack)
        num_endDate = modelDate - datetime.timedelta(self.numEndDate)
        numData = self.loadRawNumeratorData(self.numItem, modelDate, 
                                num_startDate, num_endDate, 
                                subids, currency_id)
        # Load denominator data
        denom_startDate = modelDate - datetime.timedelta(self.denomDaysBack)
        denom_endDate = modelDate - datetime.timedelta(self.denomEndDate)
        denomData = self.loadRawDenominatorData(self.denomItem, modelDate, 
                                denom_startDate, denom_endDate, 
                                subids, currency_id)

        # Compute values
        ratio = Matrices.allMasked(len(subids), dtype=float)
        freq = Matrices.allMasked(len(subids), dtype=object)
        for idx in range(len(subids)):
            #Default freq to have missing freq
            freq[idx] = self.MissingFrequency
        numMissing = list()
        denomMissing = list()

        for idx in range(len(subids)):
            # Only compute values if numerator/denominator are of same frequency
            if numData[idx].frequency == self.MissingFrequency or \
                    denomData[idx].frequency == self.MissingFrequency:
                ratio[idx] == ma.masked
            elif numData[idx].frequency == denomData[idx].frequency:
                ratio[idx] = self.processAssetValue(numData[idx], denomData[idx])
                if ratio[idx] is not ma.masked:
                    freq[idx] = numData[idx].frequency
            # Differing frequency: annual numerator, quarterly denominator
            # And only add idx into second try loop when self.useFixedFrquency is None
            elif (numData[idx].frequency == self.AnnualFrequency and \
                      denomData[idx].frequency == self.QuarterlyFrequency) and \
                      self.useFixedFrequency is None:
                denomMissing.append(idx)
            # Differing frequency: quarterly numerator, annual denominator
            elif (numData[idx].frequency == self.QuarterlyFrequency and \
                      denomData[idx].frequency == self.AnnualFrequency) and \
                      self.useFixedFrequency is None:
                numMissing.append(idx)

        # Load annual numerator data for cases where only annual denominator is present
        if len(numMissing) > 0:
            numSubIssues = [subids[nIdx] for nIdx in numMissing]
            numDataSecondary = self.loadRawNumeratorData(self.numItem, modelDate, 
                                    num_startDate, num_endDate, 
                                    numSubIssues, currency_id, allowMixedFreq=False)

            for (sIdx, sid) in enumerate(numSubIssues):
                idx = assetIdxMap[sid]
                ratio[idx] = self.processAssetValue(numDataSecondary[sIdx], denomData[idx])
                if ratio[idx] is not ma.masked:
                    freq[idx] = self.AnnualFrequency

        # Load annual denominator data for cases where only annual numerator is present
        if len(denomMissing) > 0:
            denomSubIssues = [subids[dIdx] for dIdx in denomMissing]
            denomDataSecondary = self.loadRawDenominatorData(self.denomItem, modelDate, 
                                    denom_startDate, denom_endDate, 
                                    denomSubIssues, currency_id, allowMixedFreq=False)

            for (sIdx, sid) in enumerate(denomSubIssues):
                idx = assetIdxMap[sid]
                ratio[idx] = self.processAssetValue(numData[idx], denomDataSecondary[sIdx])
                if ratio[idx] is not ma.masked:
                    freq[idx] = self.AnnualFrequency

        missing = numpy.flatnonzero(ma.getmaskarray(ratio))
        if len(missing)> 0:
            logging.info('%d assets are missing ratio (post-processed). numerator: %s, denominator: %s',
                             len(missing), self.numItem, self.denomItem)
            logging.debug('missing Axioma IDs: %s',
                          ', '.join([subids[i].getSubIDString() for i in missing]))

        logging.info('getRatio for %s/%s: end', self.numItem, self.denomItem)
        return (ratio, freq)

    def processAssetValue(self, numerator, denominator):
        """Calculate the financial ratio value for an individual asset, depending on 
        the processing options.  numerator and denominator should both be
        Structs with attributes indicating data frequency and containing the
        raw data value.
        """
        #Extra safe-gurad for zero divisor:
        if denominator.value == 0.0 or denominator.value is ma.masked:
            value = ma.masked
        else:
            value = numerator.value / denominator.value

        #Now apply the negative treatment to value
        if numerator.value < 0.0:
            if self.numNegativeTreatment == 'zero':
                value = 0.0
            elif self.numNegativeTreatment == 'mask':
                value = ma.masked
        if denominator.value < 0.0:
            if self.denomNegativeTreatment == 'zero':
                value = 0.0
            elif self.denomNegativeTreatment == 'mask':
                value = ma.masked
            elif self.denomNegativeTreatment == 'one':
                value = 1.0
        elif denominator.value == 0.0:
            if self.denomZeroTreatment == 'zero':
                value = 0.0
            elif self.denomZeroTreatment == 'one':
                value = 1.0
            elif self.denomZeroTreatment == 'mask':
                value = ma.masked
        return value

    def getValues(self, date, subids, currency_id):
        return self.getRatio(subids, date, currency_id)

def getAverageOneMonthMarketCaps(subids, date, currency_id, modelDB, marketDB, DLC=False):
    """Returns one-month average market caps. (issuer level total caps)
    """
    allRMG = modelDB.getAllRiskModelGroups()
    dateList = modelDB.getDateRange(allRMG, date - datetime.timedelta(30), date)
    mcaps = ma.filled(modelDB.getAverageMarketCaps(dateList, subids, currency_id), 0.0)
    data = Utilities.Struct()
    data.universe = subids
    data.marketCaps = mcaps
    idxMap = dict([(j,i) for (i,j) in enumerate(subids)])

    data.assetIdxMap = dict([(j,i) for (i,j) in enumerate(data.universe)])

    currency = Utilities.Struct()
    currency.currency_id = currency_id
    issuerMarketCaps = GlobalExposures.computeTotalIssuerMarketCaps(
                data, date, currency, modelDB, marketDB)
    if DLC and hasattr(data, 'DLCMarketCap'):
        return data.DLCMarketCap

    return issuerMarketCaps

class BookToPrice(FinancialRatio):
    def __init__(self, modelDB, marketDB, useFixedFrequency=None):
        FinancialRatio.__init__(self, 'ce', 'market_cap', modelDB, marketDB, 
                                useFixedFrequency=useFixedFrequency,
                                denomMultiplier=1.0)
    def loadRawDenominatorData(self, code, modelDate, startDate, endDate, subids, currency_id,
                               allowMixedFreq=True):
        if self.useFixedFrequency is not None:
            freq = [self.useFixedFrequency  for sid in  subids]
        else:
            freq = None
        return populateValueFrequencyList(getAverageOneMonthMarketCaps(
                    subids, modelDate, currency_id, self.modelDB, self.marketDB, DLC=True), freq=freq)

class EarningsToPrice(FinancialRatio):
    def __init__(self, modelDB, marketDB, useFixedFrequency=None, maskNegativeEarnings=True):
        if maskNegativeEarnings:
            treatment = 'mask'
        else:
            treatment = None
        FinancialRatio.__init__(self, 'ibei', 'market_cap', modelDB, marketDB, 
                                useFixedFrequency=useFixedFrequency,
                                denomMultiplier=1.0, numeratorProcess='annualize',
                                numeratorNegativeTreatment=treatment)
    def loadRawDenominatorData(self, code, modelDate, startDate, endDate, subids, currency_id,
                               allowMixedFreq=True):
        if self.useFixedFrequency is not None:
            freq = [self.useFixedFrequency  for sid in  subids]
        else:
            freq = None
        return populateValueFrequencyList(getAverageOneMonthMarketCaps(
                    subids, modelDate, currency_id, self.modelDB, self.marketDB, DLC=True),
                                          freq=freq)

class ReturnOnEquity(FinancialRatio):
    def __init__(self, modelDB, marketDB, useFixedFrequency=None, lagNum=0, lagDenom=365):
        FinancialRatio.__init__(self, 'ibei', 'ce', modelDB, marketDB,
                                useFixedFrequency=useFixedFrequency,
                                denomDaysBack=lagDenom+(2*365), denomEndDate=lagDenom,
                                numeratorDaysBack=lagNum+(2*365), numeratorEndDate=lagNum,
                                numeratorProcess='annualize', denomNegativeTreatment='zero')

class ReturnOnAssets(FinancialRatio):
    def __init__(self, modelDB, marketDB, useFixedFrequency=None, lagNum=0, lagDenom=365):
        FinancialRatio.__init__(self, 'ibei', 'at', modelDB, marketDB, 
                                useFixedFrequency=useFixedFrequency,
                                denomDaysBack=lagDenom+365, denomEndDate=lagDenom,
                                numeratorDaysBack=lagNum+(2*365), numeratorEndDate=lagNum,
                                numeratorProcess='annualize', denomNegativeTreatment='zero')

class SalesToPrice(FinancialRatio):
    def __init__(self, modelDB, marketDB, useFixedFrequency=None):
        FinancialRatio.__init__(self, 'sale', 'market_cap', modelDB, marketDB, 
                                useFixedFrequency=useFixedFrequency,
                                denomMultiplier=1.0,
                                numeratorProcess='annualize')
    def loadRawDenominatorData(self, code, modelDate, startDate, endDate, subids, currency_id,
                               allowMixedFreq=True):
        return populateValueFrequencyList(getAverageOneMonthMarketCaps(
                    subids, modelDate, currency_id, self.modelDB, self.marketDB, DLC=True))

class SalesToAssets(FinancialRatio):
    def __init__(self, modelDB, marketDB, useFixedFrequency=None, lagNum=0, lagDenom=365):
        FinancialRatio.__init__(self, 'sale', 'at', modelDB, marketDB,
                                useFixedFrequency=useFixedFrequency,
                                denomDaysBack=lagDenom+365, denomEndDate=lagDenom,
                                numeratorDaysBack=lagNum+(2*365), numeratorEndDate=lagNum,
                                numeratorProcess='annualize', denomNegativeTreatment='mask')

class CostOfGoodsToAssets(FinancialRatio):
    def __init__(self, modelDB, marketDB, useFixedFrequency=None, lagNum=0, lagDenom=365):
        FinancialRatio.__init__(self, 'cogs', 'at', modelDB, marketDB,
                                useFixedFrequency=useFixedFrequency,
                                denomDaysBack=lagDenom+365, denomEndDate=lagDenom,
                                numeratorDaysBack=lagNum+(2*365), numeratorEndDate=lagNum,
                                numeratorProcess='annualize', denomNegativeTreatment='mask')

class CashFlowToAssets(FinancialRatio):
    def __init__(self, modelDB, marketDB, useFixedFrequency=None, lagNum=0, lagDenom=365):
        FinancialRatio.__init__(self, 'oancf', 'at', modelDB, marketDB,
                                useFixedFrequency=useFixedFrequency,
                                denomDaysBack=lagDenom+365, denomEndDate=lagDenom,
                                numeratorDaysBack=lagNum+(2*365), numeratorEndDate=lagNum,
                                numeratorProcess='annualize', denomNegativeTreatment='mask')

class MarketEquityToAssets(FinancialRatio):
    def __init__(self, modelDB, marketDB, useFixedFrequency=None):
        FinancialRatio.__init__(self, 'me', 'at', modelDB, marketDB,
                                useFixedFrequency=useFixedFrequency,
                                denomNegativeTreatment='mask')

    def loadRawNumeratorData(self, code, modelDate, startDate, endDate,
                    subids, currency_id, allowMixedFreq=True):
        return populateValueFrequencyList(getAverageOneMonthMarketCaps(
            subids, modelDate, currency_id, self.modelDB, self.marketDB))

class CommonEquityToAssets(FinancialRatio):
    def __init__(self, modelDB, marketDB, useFixedFrequency=None):
        FinancialRatio.__init__(self, 'ce', 'at', modelDB, marketDB,
                useFixedFrequency=useFixedFrequency, denomNegativeTreatment='mask')

class EBITDAToAssets(FinancialRatio):
    def __init__(self, modelDB, marketDB):
        FinancialRatio.__init__(self, 'ebitda', 'at', modelDB, marketDB,
                                useFixedFrequency=FinancialRatio.AnnualFrequency,
                                denomNegativeTreatment='mask')

class CashFlowToIncome(FinancialRatio):
    def __init__(self, modelDB, marketDB, useFixedFrequency=None, lagDenom=0, lagNum=0):
        FinancialRatio.__init__(self, 'oancf', 'ibei', modelDB, marketDB,
                                useFixedFrequency=useFixedFrequency,
                                denomDaysBack=lagDenom+(2*365), denomEndDate=lagDenom,
                                numeratorDaysBack=lagNum+(2*365), numeratorEndDate=lagNum,
                                numeratorProcess='annualize', denomProcess='annualize',
                                denomNegativeTreatment='mask')

class CurrentToTotalAssets(FinancialRatio):
    def __init__(self, modelDB, marketDB, useFixedFrequency=None):
        FinancialRatio.__init__(self, 'act', 'at', modelDB, marketDB,
                                useFixedFrequency=useFixedFrequency,
                                denomNegativeTreatment='mask')

class CurrentLiabilitiesToAssets(FinancialRatio):
    def __init__(self, modelDB, marketDB, useFixedFrequency=None):
        FinancialRatio.__init__(self, 'lct', 'at', modelDB, marketDB,
                                useFixedFrequency=useFixedFrequency,
                                denomNegativeTreatment='mask')

class DebtToMarketCap(FinancialRatio):
    def __init__(self, modelDB, marketDB, useFixedFrequency=None, addDeposits=False):
        FinancialRatio.__init__(self, 'totalDebt', 'market_cap', modelDB, marketDB, 
                                useFixedFrequency=useFixedFrequency,
                                denomMultiplier=1.0)
        self.addDeposits = addDeposits
    def loadRawDenominatorData(self, code, modelDate, startDate, endDate, subids, currency_id,
                               allowMixedFreq=True):
        return populateValueFrequencyList(getAverageOneMonthMarketCaps(
                    subids, modelDate, currency_id, self.modelDB, self.marketDB))
    def loadRawNumeratorData(self, code, modelDate, startDate, endDate, subids, currency_id,
                             allowMixedFreq=True):
        if allowMixedFreq or self.useFixedFrequency == self.QuarterlyFrequency:
            totalDebt = self.modelDB.getQuarterlyTotalDebt(
                            startDate, endDate, subids, modelDate, 
                            self.marketDB, currency_id)
        else:
            totalDebt = self.modelDB.getAnnualTotalDebt(
                            startDate, endDate, subids, modelDate,
                            self.marketDB, currency_id)

        data = Utilities.extractLatestValue(totalDebt)
        if self.addDeposits:
            data = loadDeposits(data, subids, modelDate, self.modelDB, self.marketDB)
        return populateValueFrequencyList(data * self.numMultiplier)

class DebtToEquity(FinancialRatio):
    def __init__(self, modelDB, marketDB, useFixedFrequency=None, addDeposits=False):
        FinancialRatio.__init__(self, 'totalDebt', 'ce', modelDB, marketDB,
                                useFixedFrequency=useFixedFrequency,
                                denomMultiplier=1.0, denomZeroTreatment='one',
                                denomNegativeTreatment='one')
        self.addDeposits = addDeposits
    def loadRawNumeratorData(self, code, modelDate, startDate, endDate, subids, currency_id,
                             allowMixedFreq=True):
        if allowMixedFreq or self.useFixedFrequency == self.QuarterlyFrequency:
            totalDebt = self.modelDB.getQuarterlyTotalDebt(
                            startDate, endDate, subids, modelDate,
                            self.marketDB, currency_id)
        else:
            totalDebt = self.modelDB.getAnnualTotalDebt(
                            startDate, endDate, subids, modelDate,
                            self.marketDB, currency_id)

        data = Utilities.extractLatestValue(totalDebt)
        if self.addDeposits:
            data = loadDeposits(data, subids, modelDate, self.modelDB, self.marketDB)
        return populateValueFrequencyList(data)

class DebtToTotalAssets(FinancialRatio):
    def __init__(self, modelDB, marketDB, useFixedFrequency=None, addDeposits=False):
        FinancialRatio.__init__(self, 'totalDebt', 'at', modelDB, marketDB,
                                useFixedFrequency=useFixedFrequency)
        self.addDeposits = addDeposits
    def loadRawNumeratorData(self, code, modelDate, startDate, endDate, subids, currency_id,
                             allowMixedFreq=True):
        if allowMixedFreq or self.useFixedFrequency == self.QuarterlyFrequency:
            totalDebt = self.modelDB.getQuarterlyTotalDebt(
                            startDate, endDate, subids, modelDate, 
                            self.marketDB, currency_id)
        else:
            totalDebt = self.modelDB.getAnnualTotalDebt(
                            startDate, endDate, subids, modelDate,
                            self.marketDB, currency_id)
        data = Utilities.extractLatestValue(totalDebt)
        if self.addDeposits:
            data = loadDeposits(data, subids, modelDate, self.modelDB, self.marketDB)
        return populateValueFrequencyList(data*self.numMultiplier)
    
def loadDeposits(data, subids, date, modelDB, marketDB):
    # Ugly temporary hack to load deposits
    nonMissingIdx = numpy.flatnonzero(ma.getmaskarray(data)==0)
    if len(nonMissingIdx) > 0:
        import leverageHack
        okSubIds = numpy.take(subids, nonMissingIdx, axis=0)
        dep = leverageHack.loadDeposits(okSubIds, date, modelDB, marketDB)
        for (idx, sid) in zip(nonMissingIdx, okSubIds):
            xd = dep[sid]
            if (xd is not None) and ('DPTC' in xd):
                data[idx] += ma.filled(xd['DPTC'], 0.0)
    return data

class DividendYield(FinancialRatio):
    def __init__(self, modelDB, marketDB, useFixedFrequency=None):
        FinancialRatio.__init__(self, 'divs', 'market_cap', modelDB, marketDB, 
                                useFixedFrequency=useFixedFrequency,
                                numeratorMultiplier=1.0, denomMultiplier=1.0)
    def loadRawDenominatorData(self, code, modelDate, startDate, endDate, subids, currency_id,
                               allowMixedFreq=True):
        return populateValueFrequencyList(getAverageOneMonthMarketCaps(
                                subids, modelDate, currency_id, self.modelDB, self.marketDB))

    def loadRawNumeratorData(self, code, modelDate, startDate, endDate, subids, currency_id,
                             allowMixedFreq=True):
        currency = Utilities.Struct()
        currency.numeraire = Utilities.Struct()
        currency.numeraire.currency_id = currency_id
        divs = computeAnnualDividends(
                                modelDate, subids, currency,
                                self.modelDB, self.marketDB, maskMissing=True,
                                includeSpecial=False, includeStock=False)
        idxMap = dict([(j,i) for (i,j) in enumerate(subids)])
        return populateValueFrequencyList(divs)

class ProxiedDividendPayout(FinancialRatio):
    def __init__(self, modelDB, marketDB, useFixedFrequency=None):
        FinancialRatio.__init__(self, 'divs', 'ibei', modelDB, marketDB, 
                                useFixedFrequency=useFixedFrequency,
                                numeratorMultiplier=1.0, 
                                numeratorNegativeTreatment='zero',
                                denomNegativeTreatment='zero',
                                denomZeroTreatment='one')


    def loadRawNumeratorData(self, code, modelDate, startDate, endDate, subids, currency_id,
                             allowMixedFreq=True):
        currency = Utilities.Struct()
        currency.numeraire = Utilities.Struct()
        currency.numeraire.currency_id = currency_id
        divs = Utilities.screen_data(computeAnnualDividends(
                modelDate, subids, currency,
                self.modelDB, self.marketDB,
                includeSpecial=True, includeStock=False))
        return populateValueFrequencyList(divs)

class DividendPayout(FinancialRatio):
    def __init__(self, modelDB, marketDB, useFixedFrequency=None):
        FinancialRatio.__init__(self, 'dps', 'epsxei', modelDB, marketDB,
                useFixedFrequency=useFixedFrequency,
                numeratorNegativeTreatment='zero',
                denomNegativeTreatment='zero',
                numeratorProcess='annualize',
                denomZeroTreatment='one')

class AssetTurnover(FinancialRatio):
    def __init__(self, modelDB, marketDB, useFixedFrequency=None):
        FinancialRatio.__init__(self, 'sale', 'at', modelDB, marketDB, 
                                useFixedFrequency=useFixedFrequency,
                                numeratorProcess='annualize')

class OperatingCashFlowToPrice(FinancialRatio):
    def __init__(self, modelDB, marketDB):
        #For item involving cash flow, currently only allow annual frequency 
        FinancialRatio.__init__(self, 'oancf', 'market_cap', modelDB, marketDB, 
                                useFixedFrequency=FinancialRatio.AnnualFrequency,
                                denomDaysBack=2*365, denomEndDate=365,
                                denomMultiplier=1.0)
    def loadRawDenominatorData(self, code, modelDate, startDate, endDate, subids, currency_id,
                               allowMixedFreq=False):
        return populateValueFrequencyList(getAverageOneMonthMarketCaps(
                    subids, modelDate, currency_id, self.modelDB, self.marketDB), freq=FinancialRatio.AnnualFrequency)

class Accruals(FinancialRatio):
    def __init__(self, modelDB, marketDB):
        #For item involving cash flow, currently only allow annual frequency 
        FinancialRatio.__init__(self, 'oancf', 'ibei', modelDB, marketDB, 
                                useFixedFrequency=FinancialRatio.AnnualFrequency)


class OperatingCashFlowToAssets(FinancialRatio):
    def __init__(self, modelDB, marketDB):
        #For item involving cash flow, currently only allow annual frequency 
        FinancialRatio.__init__(self, 'oancf', 'at', modelDB, marketDB, 
                                useFixedFrequency=FinancialRatio.AnnualFrequency,
                                denomDaysBack=2*365, denomEndDate=365)

class CurrentRatio(FinancialRatio):
    def __init__(self, modelDB, marketDB, useFixedFrequency=None):
        FinancialRatio.__init__(self, 'act', 'lct', modelDB, marketDB,
                                useFixedFrequency=useFixedFrequency)

class GrossMargin(FinancialRatio):
    def __init__(self, modelDB, marketDB, useFixedFrequency=None, lagDenom=0, lagNum=0):
        FinancialRatio.__init__(self, 'cogs', 'sale', modelDB, marketDB,
                                useFixedFrequency=useFixedFrequency,
                                numeratorProcess='annualize',
                                denomDaysBack=lagDenom+(2*365), denomEndDate=lagDenom,
                                numeratorDaysBack=lagNum+(2*365), numeratorEndDate=lagNum,
                                denomProcess='annualize')
    def getValues(self, date, subids, currency_id):
        (value, freq) =  self.getRatio(subids, date, currency_id)
        return (1.0 - value, freq)

class EarningsGrowthToPE(EarningsToPrice):
    def __init__(self, modelDB, marketDB, useFixedFrequency=None):
        EarningsToPrice.__init__(self, modelDB, marketDB, useFixedFrequency=useFixedFrequency)
    
    def getValues(self, date, subids, currency_id):
        (ep, epFreq) = self.getRatio(subids, date, currency_id)
        if self.useFixedFrequency is None:
            useQuarterlyData = None
        else:
            if self.useFixedFrequency.suffix == '_ann':
                useQuarterlyData = False
            else:
                useQuarterlyData = True

        g = StyleExposures.generate_growth_rate('ibei', subids, date, currency_id,
                                                self.modelDB, self.marketDB,
                                                useQuarterlyData=useQuarterlyData)
        g = ma.masked_where(g<0, g)

        peg = Matrices.allMasked(len(subids), dtype=float)
        freq = Matrices.allMasked(len(subids), dtype=object)
        
        for idx in range(len(subids)):
            freq[idx] = epFreq[idx]
            if epFreq[idx] != self.MissingFrequency:
                peg[idx] = ep[idx]*g[idx]*100
            else:
                peg[idx] = ma.masked
        return (peg, freq)
