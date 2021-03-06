
import logging
import optparse
import configparser
import datetime
import pandas
import numpy
from numpy import ma as ma

try :
    import numpy.ma as ma
except:
    import numpy.core.ma as ma
from riskmodels import ModelDB
from riskmodels import Matrices
from riskmodels import Utilities

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

def populateValueFrequencyList(vals, freq=None):
    """Return a list whose entries are Structs, each containing
    the corresponding value in data, along with its associated
    DataFrequency type.
    """
    values = Matrices.allMasked(len(vals), dtype=object)
    # Check if freq is a listlike object
    try:
        tmp = freq[0]
        freqIsList = True
    except:
        freqIsList = False
    if freq is None:
        freq = DescriptorRatio.QuarterlyFrequency
    for i in range(len(vals)):
        val = Utilities.Struct()
        if freqIsList:
            val.frequency = freq[i]
        else:
            val.frequency = freq
        val.value = vals[i]
        values[i] = val
    return values

def outputRawData(trackList, subIssues, data, code):
    if len(trackList) < 1:
        return
    dataDF = pandas.DataFrame(data, index=subIssues)
    for sid in trackList:
        sidData = dataDF.loc[sid]
        logging.info('TRACKING %s, %s', sid.getSubIDString(), code)
        for idx, itm in sidData.items():
            if type(itm) is list:
                for elt in itm:
                    logging.info('......... %s', elt)
            else:
                logging.info('......... %s', itm)
    return

class DescriptorRatio:
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
                 numeratorMultiplier=1e6, denomMultiplier=1e6,
                 sidRanges=None, trackList=[]):
        """Represnts the a generic financial ratio.
        denom/numeratorDaysBack = days back to start looking for data
        denom/numeratorEndDate = days back to end looking for data (0 = present day)
        denom/numeratorProcess = 'extractlatest' (balance sheet) or 'annualize' (income statement) or 'average' (average an annualize apply to qtr data only)
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
        self.sidRanges = sidRanges
        self.trackList = trackList

    def loadRawDenominatorData(self, code, modelDate, startDate, endDate, data, subIssues, currency_id):
        """Retrieve raw data for the given denominator data item.
        Subclasses of DescriptorRatio that use 'complex' denominators
        beyond simple items of financial reporting data should define their own
        loadRawDenominatorData() method to perform whatever computations are required.
        """
        return self.loadRawFundamentalDataInternal(code, modelDate, startDate, endDate, data, subIssues, currency_id, isNumerator=False)

    def loadRawNumeratorData(self, code, modelDate, startDate, endDate, data, subIssues, currency_id):
        """Retrieve raw data for the given numerator data item.
        Subclasses of DescriptorRatio that use 'complex' numerators
        beyond simple items of financial reporting data should define their own
        loadRawNumeratorData() method to perform whatever computations are required.
        """
        return self.loadRawFundamentalDataInternal(code, modelDate, startDate, endDate, data, subIssues, currency_id, isNumerator=True)

    def loadRawFundamentalDataInternal(self, code, modelDate, startDate, endDate, data, subIssues, currency_id, isNumerator):
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
        suffix = None
        if process.lower() not in ['annualize','extractlatest', 'average']:
            raise ValueError('Unsupported processing option %s' % process)
        trackList = sorted(set(self.trackList).intersection(set(subIssues)))

        # Default: load and process data from mixed frequencies
        if self.useFixedFrequency is None:

            logging.info('Loading data for %s: Frequency: Mixed, Process: %s', code, process)

            if process.lower() == 'annualize':
                (rawValues, frequencies) = self.modelDB.getMixFundamentalCurrencyItem(
                                code, startDate, endDate, subIssues, modelDate, self.marketDB,
                                currency_id, splitAdjust=splitAdjust, requireConsecQtrData=3)
            else:
                (rawValues, frequencies) = self.modelDB.getMixFundamentalCurrencyItem(
                                code, startDate, endDate, subIssues, modelDate, self.marketDB,
                                currency_id, splitAdjust=splitAdjust)
            outputRawData(trackList, subIssues, rawValues, code)

            # Remove rawValues with dts that precede asset fromDts
            if self.sidRanges:
                rawValuesTmp = Matrices.allMasked(len(subIssues), dtype=object)
                for i in range(len(subIssues)):
                    rawValuesTmp[i] = [val for val in rawValues[i]
                                       if val[0] >= self.sidRanges[subIssues[i]][0]]
                rawValues = rawValuesTmp

            # Here we need to figure whether it's quarterly or annual data
            vals = ma.zeros(len(subIssues))
            vals.mask = ma.getmaskarray(vals)
            qtrIdx = numpy.flatnonzero(ma.getmaskarray(ma.masked_where(
                        frequencies==self.QuarterlyFrequency, frequencies)))
            annIdx = numpy.flatnonzero(ma.getmaskarray(ma.masked_where(
                        frequencies==self.AnnualFrequency, frequencies)))

            # Handle processes for quarterly data
            if len(qtrIdx) > 0:
                if process.lower() == 'annualize':
                    annualizedData = self.modelDB.annualizeQuarterlyValues(
                        [rawValues[k] for k in qtrIdx], adjustShortHistory=True, requireAnnualWindow=True)
                    if type(annualizedData.mask) == numpy.bool_:
                        annualizedData.mask = ma.getmaskarray(annualizedData)
                    ma.put(vals, qtrIdx, annualizedData)
                    if type(vals.mask) == numpy.bool_:
                        vals.mask = ma.getmaskarray(vals)
                elif process.lower() == 'average':
                    averagedData = self.modelDB.averageQuarterlyValues(
                        [rawValues[k] for k in qtrIdx])
                    if type(averagedData.mask) == numpy.bool_:
                        averagedData.mask = ma.getmaskarray(averagedData)
                    ma.put(vals, qtrIdx, averagedData)
                    if type(vals.mask) == numpy.bool_:
                        vals.mask = ma.getmaskarray(vals)
                else: # use extractLatest
                    latestData = Utilities.extractLatestValue([rawValues[k] for k in qtrIdx])
                    ma.put(vals, qtrIdx, latestData)
                    if type(vals.mask) == numpy.bool_:
                        vals.mask = ma.getmaskarray(vals)

            # Handle processes for annual data
            if len(annIdx) > 0:
                if process.lower() == 'average':
                    averagedData = self.modelDB.averageAnnualValues(
                        [rawValues[k] for k in annIdx])
                    if type(averagedData.mask) == numpy.bool_:
                        averagedData.mask = ma.getmaskarray(averagedData)
                    ma.put(vals, annIdx, averagedData)
                    if type(vals.mask) == numpy.bool_:
                        vals.mask = ma.getmaskarray(vals)
                else: # use extractLatest
                    latestData = Utilities.extractLatestValue([rawValues[k] for k in annIdx])
                    ma.put(vals, annIdx, latestData)
                    if type(vals.mask) == numpy.bool_:
                        vals.mask = ma.getmaskarray(vals)

        # Otherwise, either mixing is disabled, or this is a 'secondary' run;
        # that is, fetching annual data only for certain items
        else:
            logging.info('Loading data for %s: Frequency: %s, Process: %s', code, self.useFixedFrequency.suffix, process)
            suffix = self.useFixedFrequency.suffix
            requireConsecQtrData = None
            if (self.useFixedFrequency == self.QuarterlyFrequency) and (process.lower() == 'annualize'):
                requireConsecQtrData = 3

            vals = self.modelDB.getFundamentalCurrencyItem(
                            code + suffix, startDate, endDate,
                            subIssues, modelDate, self.marketDB, currency_id,
                            splitAdjust=splitAdjust, requireConsecQtrData=requireConsecQtrData)
            outputRawData(trackList, subIssues, vals, code)

            # Remove rawValues with dts that precede asset fromDts
            if self.sidRanges:
                dataTmp = []
                for i in range(len(subIssues)):
                    dataTmp.append([val for val in vals[i]
                                    if val[0] >= self.sidRanges[subIssues[i]][0]])
                vals = dataTmp

            # Handle processes for annual data
            if self.useFixedFrequency == self.QuarterlyFrequency:
                if process.lower() == 'annualize':
                    vals = self.modelDB.annualizeQuarterlyValues(   
                            vals, adjustShortHistory=True, requireAnnualWindow=True)
                elif process.lower() == 'average':
                    vals = self.modelDB.averageQuarterlyValues(vals)
                else:
                    vals = Utilities.extractLatestValue(vals)
            elif self.useFixedFrequency == self.AnnualFrequency:
                if process.lower() == 'average':
                    vals = self.modelDB.averageAnnualValues(vals)
                else:
                    vals = Utilities.extractLatestValue(vals)
            else:
                vals = Utilities.extractLatestValue(vals)

        # Rescale if required
        if multiplier != 1.0:
            for idx in range(len(subIssues)):
                val = vals[idx]
                vals[idx] = val * multiplier

        # Return values along with associated frequencies
        if frequencies is not None:
            freq = frequencies
        else:
            if self.useFixedFrequency is not None:
                freq = self.useFixedFrequency
            else:
                freq = self.AnnualFrequency
        return populateValueFrequencyList(vals, freq)

    def getRatio(self, data, modelDate, currency_id):
        """Compute the financial ratio for the given assets on a given date.
        The numItem/denomItem should be prefix of fundamental items you want.
        The code will first search for the most frequently updated data for both numerator and
        denominator. If it sees that the numerator and denominator are in different freqency,
        it will make them both annual data to come up wth the ratio.
        Returns an array with desired ratio and another array of frequency indicating
        the frequency used in each value returned.
        """
        logging.info('getRatio for %s/%s: begin', self.numItem, self.denomItem)

        assetIdxMap = dict([(j,i) for (i,j) in enumerate(data.universe)])

        # Load numerator data
        num_startDate = modelDate - datetime.timedelta(self.numDaysBack)
        num_endDate = modelDate - datetime.timedelta(self.numEndDate)
        numData = self.loadRawNumeratorData(self.numItem, modelDate,
                                num_startDate, num_endDate, data, data.universe, currency_id)
        # Load denominator data
        denom_startDate = modelDate - datetime.timedelta(self.denomDaysBack)
        denom_endDate = modelDate - datetime.timedelta(self.denomEndDate)
        denomData = self.loadRawDenominatorData(self.denomItem, modelDate,
                                denom_startDate, denom_endDate, data, data.universe, currency_id)

        # Compute values
        ratio = Matrices.allMasked(len(data.universe), dtype=float)
        freq = Matrices.allMasked(len(data.universe), dtype=object)
        for idx in range(len(data.universe)):
            #Default freq to have missing freq
            freq[idx] = self.MissingFrequency
        numMissing = list()
        denomMissing = list()

        for idx in range(len(data.universe)):
            # Only compute values if numerator/denominator are of same frequency
            if numData[idx].frequency == self.MissingFrequency or \
                    denomData[idx].frequency == self.MissingFrequency:
                ratio[idx] = ma.masked
            elif numData[idx].frequency == denomData[idx].frequency:
                ratio[idx] = self.processAssetValue(numData[idx], denomData[idx])
                if ratio[idx] is not ma.masked:
                    freq[idx] = numData[idx].frequency
            # Differing frequency: annual numerator, quarterly denominator
            # And only add idx into second try loop when self.useFixedFrequency is None
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
            numSubIssues = [data.universe[nIdx] for nIdx in numMissing]

            # Alter important parameters - note the mixed method assumes we always extract latest annual value,
            # This makes it different from some of the true annual descriptor methods
            keepFlag = self.useFixedFrequency
            procFlag = self.numProcess
            self.useFixedFrequency = self.AnnualFrequency
            self.numProcess = 'extractLatest'

            # Load annual data
            numDataSecondary = self.loadRawNumeratorData(self.numItem, modelDate,
                                    num_startDate, num_endDate, data,
                                    numSubIssues, currency_id)
            # Reset parameters
            self.useFixedFrequency = keepFlag
            self.numProcess = procFlag

            for (sIdx, sid) in enumerate(numSubIssues):
                idx = assetIdxMap[sid]
                ratio[idx] = self.processAssetValue(numDataSecondary[sIdx], denomData[idx])
                if ratio[idx] is not ma.masked:
                    freq[idx] = self.AnnualFrequency

        # Load annual denominator data for cases where only annual numerator is present
        if len(denomMissing) > 0:
            denomSubIssues = [data.universe[dIdx] for dIdx in denomMissing]

            # Alter important parameters - note same issue as numerator step above
            keepFlag = self.useFixedFrequency
            procFlag = self.denomProcess
            self.useFixedFrequency = self.AnnualFrequency
            self.denomProcess = 'extractLatest'

            denomDataSecondary = self.loadRawDenominatorData(self.denomItem, modelDate,
                                    denom_startDate, denom_endDate, data,
                                    denomSubIssues, currency_id)
            # Reset parameters
            self.useFixedFrequency = keepFlag
            self.denomProcess = procFlag

            for (sIdx, sid) in enumerate(denomSubIssues):
                idx = assetIdxMap[sid]
                ratio[idx] = self.processAssetValue(numData[idx], denomDataSecondary[sIdx])
                if ratio[idx] is not ma.masked:
                    freq[idx] = self.AnnualFrequency

        missing = numpy.flatnonzero(ma.getmaskarray(ratio))
        if len(missing)> 0:
            missRatio = len(missing) / float(len(ratio))
            if missRatio > 0.75:
                logging.warning('%d assets (%d%%) are missing ratio (post-processed). numerator: %s, denominator: %s',
                            len(missing), missRatio*100, self.numItem, self.denomItem)
            else:
                logging.info('%d assets (%d%%) are missing ratio (post-processed). numerator: %s, denominator: %s',
                            len(missing), missRatio*100, self.numItem, self.denomItem)
            if len(missing) < 10:
                logging.debug('missing Axioma IDs: %s',
                          ', '.join([data.universe[i].getSubIDString() for i in missing]))

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

    def getValues(self, date, data, currency_id):
        return self.getRatio(data, date, currency_id)

def getAverageOneMonthMarketCaps(data, subIssues, DLC=False):
    """Returns one-month average market caps. (issuer level total caps)
    """
    idx = [data.assetIdxMap[sid] for sid in subIssues]
    if DLC:
        mcaps = data.DLCMarketCap
    else:
        mcaps = data.issuerTotalMarketCaps
    return numpy.take(mcaps, idx, axis=0)

class BookToPrice(DescriptorRatio):
    def __init__(self, modelDB, marketDB, useFixedFrequency=None, sidRanges=None):
        DescriptorRatio.__init__(self, 'ce', 'market_cap', modelDB, marketDB,
                                useFixedFrequency=useFixedFrequency,
                                sidRanges=sidRanges)
    def loadRawDenominatorData(self, code, modelDate, startDate, endDate, data, subIssues, currency_id):
        return populateValueFrequencyList(getAverageOneMonthMarketCaps(data, subIssues, DLC=True), freq=self.useFixedFrequency)

class BookAlone(DescriptorRatio):
    def __init__(self, modelDB, marketDB, useFixedFrequency=None, sidRanges=None, numeratorDaysBack=365):
        DescriptorRatio.__init__(self, 'ce', 'dummy', modelDB, marketDB,
                                useFixedFrequency=useFixedFrequency,
                                numeratorDaysBack=numeratorDaysBack,
                                sidRanges=sidRanges)
    def loadRawDenominatorData(self, code, modelDate, startDate, endDate, data, subIssues, currency_id):
        return populateValueFrequencyList(numpy.ones((len(subIssues)), float), freq=self.useFixedFrequency)

class EarningsToPrice(DescriptorRatio):
    def __init__(self, modelDB, marketDB, useFixedFrequency=None,
                 numeratorProcess='extractlatest', numeratorNegativeTreatment=None,
                 sidRanges=None, trackList=[]):
        DescriptorRatio.__init__(self, 'ibei', 'market_cap', modelDB, marketDB,
                                useFixedFrequency=useFixedFrequency,
                                numeratorProcess=numeratorProcess,
                                numeratorNegativeTreatment=numeratorNegativeTreatment,
                                trackList=trackList,
                                sidRanges=sidRanges)
    def loadRawDenominatorData(self, code, modelDate, startDate, endDate, data, subIssues, currency_id):
        return populateValueFrequencyList(getAverageOneMonthMarketCaps(data, subIssues, DLC=True), freq=self.useFixedFrequency)

class EarningsAlone(DescriptorRatio):
    def __init__(self, modelDB, marketDB, useFixedFrequency=None,
            numeratorProcess='extractlatest', numeratorNegativeTreatment=None, sidRanges=None,
            numeratorDaysBack=365):
        DescriptorRatio.__init__(self, 'ibei', 'dummy', modelDB, marketDB,
                                useFixedFrequency=useFixedFrequency,
                                numeratorProcess=numeratorProcess,
                                numeratorDaysBack=numeratorDaysBack,
                                numeratorNegativeTreatment=numeratorNegativeTreatment,
                                sidRanges=sidRanges)
    def loadRawDenominatorData(self, code, modelDate, startDate, endDate, data, subIssues, currency_id):
        return populateValueFrequencyList(numpy.ones((len(subIssues)), float), freq=self.useFixedFrequency)

class ReturnOnEquity(DescriptorRatio):
    def __init__(self, modelDB, marketDB, useFixedFrequency=None,
                 numeratorProcess='extractlatest',
                 denomProcess='average',
                 numeratorDaysBack=2*365, denomDaysBack=3*365,
                 sidRanges=None):
        DescriptorRatio.__init__(self, 'ibei', 'ce', modelDB, marketDB,
                                useFixedFrequency=useFixedFrequency,
                                numeratorProcess=numeratorProcess,
                                denomProcess=denomProcess,
                                numeratorDaysBack=numeratorDaysBack, 
                                denomDaysBack=denomDaysBack,
                                denomNegativeTreatment='mask',
                                denomZeroTreatment='mask',
                                sidRanges=sidRanges)

class ReturnOnAssets(DescriptorRatio):
    def __init__(self, modelDB, marketDB, useFixedFrequency=None,
                 numeratorProcess='extractlatest',
                 denomProcess='average',
                 numeratorDaysBack=2*365, denomDaysBack=3*365,
                 sidRanges=None):
        DescriptorRatio.__init__(self, 'ibei', 'at', modelDB, marketDB,
                                useFixedFrequency=useFixedFrequency,
                                numeratorProcess=numeratorProcess,
                                denomProcess=denomProcess,
                                numeratorDaysBack=numeratorDaysBack, 
                                denomDaysBack=denomDaysBack,
                                denomNegativeTreatment='mask',
                                denomZeroTreatment='mask',
                                sidRanges=sidRanges)

class SalesToPrice(DescriptorRatio):
    def __init__(self, modelDB, marketDB, useFixedFrequency=None):
        DescriptorRatio.__init__(self, 'sale', 'market_cap', modelDB, marketDB,
                                useFixedFrequency=useFixedFrequency,
                                numeratorProcess='annualize')
    def loadRawDenominatorData(self, code, modelDate, startDate, endDate, data, subIssues, currency_id):
        return populateValueFrequencyList(getAverageOneMonthMarketCaps(data, subIssues, DLC=True), freq=self.useFixedFrequency)

class SalesAlone(DescriptorRatio):
    def __init__(self, modelDB, marketDB, useFixedFrequency=None, numeratorProcess='extractlatest',
                numeratorDaysBack=365, sidRanges=None):
        DescriptorRatio.__init__(self, 'sale', 'dummy', modelDB, marketDB,
                                useFixedFrequency=useFixedFrequency,
                                numeratorProcess=numeratorProcess,
                                numeratorDaysBack=numeratorDaysBack,
                                sidRanges=sidRanges)
    def loadRawDenominatorData(self, code, modelDate, startDate, endDate, data, subIssues, currency_id):
        return populateValueFrequencyList(numpy.ones((len(subIssues)), float), freq=self.useFixedFrequency)

class SalesToAssets(DescriptorRatio):
    def __init__(self, modelDB, marketDB, useFixedFrequency=None,
                 numeratorProcess='extractlatest',
                 denomProcess='average',
                 numeratorDaysBack=2*365, denomDaysBack=3*365,
                 sidRanges=None):
        DescriptorRatio.__init__(self, 'sale', 'at', modelDB, marketDB,
                                useFixedFrequency=useFixedFrequency,
                                numeratorProcess=numeratorProcess,
                                denomProcess=denomProcess,
                                numeratorDaysBack=numeratorDaysBack, 
                                denomDaysBack=denomDaysBack,
                                denomNegativeTreatment='mask',
                                denomZeroTreatment='mask',
                                sidRanges=sidRanges)

class CostOfGoodsToAssets(DescriptorRatio):
    def __init__(self, modelDB, marketDB, useFixedFrequency=None, lagNum=0, lagDenom=365):
        DescriptorRatio.__init__(self, 'cogs', 'at', modelDB, marketDB,
                                useFixedFrequency=useFixedFrequency,
                                denomDaysBack=lagDenom+365, denomEndDate=lagDenom,
                                numeratorDaysBack=lagNum+(2*365), numeratorEndDate=lagNum,
                                numeratorProcess='annualize', denomNegativeTreatment='mask')

class CostOfGoodsAlone(DescriptorRatio):
    def __init__(self, modelDB, marketDB, useFixedFrequency=None, numeratorProcess='extractlatest',
                    numeratorDaysBack=365, sidRanges=None):
        DescriptorRatio.__init__(self, 'cogs', 'dummy', modelDB, marketDB,
                                useFixedFrequency=useFixedFrequency,
                                numeratorProcess=numeratorProcess,
                                sidRanges=sidRanges)
    def loadRawDenominatorData(self, code, modelDate, startDate, endDate, data, subIssues, currency_id):
        return populateValueFrequencyList(numpy.ones((len(subIssues)), float), freq=self.useFixedFrequency)

class OperatingCashFlowToAssets(DescriptorRatio):
    def __init__(self, modelDB, marketDB):
        #For item involving cash flow, currently only allow annual frequency
        DescriptorRatio.__init__(self, 'oancf', 'at', modelDB, marketDB,
                                useFixedFrequency=DescriptorRatio.AnnualFrequency,
                                denomDaysBack=2*365, denomEndDate=365)

class CashFlowToAssets(DescriptorRatio):
    def __init__(self, modelDB, marketDB, useFixedFrequency=DescriptorRatio.AnnualFrequency,
                 numeratorProcess='extractlatest',
                 denomProcess='average',
                 numeratorDaysBack=2*365, denomDaysBack=3*365,
                 sidRanges=None):
        DescriptorRatio.__init__(self, 'oancf', 'at', modelDB, marketDB,
                                useFixedFrequency=useFixedFrequency,
                                numeratorProcess=numeratorProcess,
                                denomProcess=denomProcess,
                                numeratorDaysBack=numeratorDaysBack, 
                                denomDaysBack=denomDaysBack,
                                denomNegativeTreatment='mask',
                                denomZeroTreatment='mask',
                                sidRanges=sidRanges)

class CashFlowToIncome(DescriptorRatio):
    def __init__(self, modelDB, marketDB, useFixedFrequency=None,
                 denomProcess='extractlatest',
                 numeratorProcess='extractlatest',
                 numeratorDaysBack=3*365, denomDaysBack=3*365,
                 sidRanges=None):
        DescriptorRatio.__init__(self, 'oancf', 'ibei', modelDB, marketDB,
                                useFixedFrequency=useFixedFrequency,
                                numeratorProcess=numeratorProcess,
                                denomProcess=denomProcess,
                                numeratorDaysBack=numeratorDaysBack,
                                denomDaysBack=denomDaysBack,
                                denomNegativeTreatment='mask',
                                denomZeroTreatment='mask',
                                sidRanges=sidRanges)

class MarketEquityToAssets(DescriptorRatio):
    def __init__(self, modelDB, marketDB, useFixedFrequency=None):
        DescriptorRatio.__init__(self, 'me', 'at', modelDB, marketDB,
                                useFixedFrequency=useFixedFrequency,
                                denomNegativeTreatment='mask')

    def loadRawNumeratorData(self, code, modelDate, startDate, endDate, data, subIssues, currency_id):
        return populateValueFrequencyList(getAverageOneMonthMarketCaps(data, subIssues), freq=self.useFixedFrequency)

class CommonEquityToAssets(DescriptorRatio):
    def __init__(self, modelDB, marketDB, useFixedFrequency=None):
        DescriptorRatio.__init__(self, 'ce', 'at', modelDB, marketDB,
                useFixedFrequency=useFixedFrequency, denomNegativeTreatment='mask')

class EBITDAToAssets(DescriptorRatio):
    def __init__(self, modelDB, marketDB):
        DescriptorRatio.__init__(self, 'ebitda', 'at', modelDB, marketDB,
                                useFixedFrequency=DescriptorRatio.AnnualFrequency,
                                denomNegativeTreatment='mask')

class CurrentToTotalAssets(DescriptorRatio):
    def __init__(self, modelDB, marketDB, useFixedFrequency=None):
        DescriptorRatio.__init__(self, 'act', 'at', modelDB, marketDB,
                                useFixedFrequency=useFixedFrequency,
                                denomNegativeTreatment='mask')

class CurrentLiabilitiesToAssets(DescriptorRatio):
    def __init__(self, modelDB, marketDB, useFixedFrequency=None):
        DescriptorRatio.__init__(self, 'lct', 'at', modelDB, marketDB,
                                useFixedFrequency=useFixedFrequency,
                                denomNegativeTreatment='mask')

class DebtToMarketCap(DescriptorRatio):
    def __init__(self, modelDB, marketDB, useFixedFrequency=None, addDeposits=False, sidRanges=None):
        DescriptorRatio.__init__(self, 'totalDebt', 'market_cap', modelDB, marketDB,
                                useFixedFrequency=useFixedFrequency, sidRanges=sidRanges)
        self.addDeposits = addDeposits

    def loadRawDenominatorData(self, code, modelDate, startDate, endDate, data, subIssues, currency_id):
        return populateValueFrequencyList(getAverageOneMonthMarketCaps(data, subIssues), freq=self.useFixedFrequency)

    def loadRawNumeratorData(self, code, modelDate, startDate, endDate, data, subIssues, currency_id):
        if (self.useFixedFrequency is None) or (self.useFixedFrequency == self.QuarterlyFrequency):
            totalDebt = self.modelDB.getQuarterlyTotalDebt(
                            startDate, endDate, subIssues, modelDate,
                            self.marketDB, currency_id)
        else:
            totalDebt = self.modelDB.getAnnualTotalDebt(
                            startDate, endDate, subIssues, modelDate,
                            self.marketDB, currency_id)

        vals = Utilities.extractLatestValue(totalDebt)
        if self.addDeposits:
            vals = loadDeposits(vals, subIssues, modelDate, self.modelDB, self.marketDB)
        return populateValueFrequencyList(vals * self.numMultiplier, freq=self.useFixedFrequency)

class DebtToEquity(DescriptorRatio):
    def __init__(self, modelDB, marketDB, useFixedFrequency=None,
                 denomProcess='extractlatest', addDeposits=False,
                 denomDaysBack=3*365, trackList=[],
                 sidRanges=None):
        DescriptorRatio.__init__(self, 'totalDebt', 'ce', modelDB, marketDB,
                                useFixedFrequency=useFixedFrequency,
                                denomMultiplier=1.0, denomProcess=denomProcess,
                                denomDaysBack=denomDaysBack,
                                denomNegativeTreatment='mask',
                                denomZeroTreatment='mask',
                                trackList=trackList,
                                sidRanges=sidRanges)
        self.addDeposits = addDeposits

    def loadRawNumeratorData(self, code, modelDate, startDate, endDate, data, subIssues, currency_id):
        if (self.useFixedFrequency is None) or (self.useFixedFrequency == self.QuarterlyFrequency):
            totalDebt = self.modelDB.getQuarterlyTotalDebt(
                            startDate, endDate, subIssues, modelDate,
                            self.marketDB, currency_id)
            retFreq = DataFrequency('_qtr')
        else:
            totalDebt = self.modelDB.getAnnualTotalDebt(
                            startDate, endDate, subIssues, modelDate,
                            self.marketDB, currency_id)
            retFreq = DataFrequency('_ann')

        trackList = sorted(set(self.trackList).intersection(set(subIssues)))
        outputRawData(trackList, subIssues, totalDebt, 'totalDebt%s' % retFreq)
        vals = Utilities.extractLatestValue(totalDebt)
        if self.addDeposits:
            vals = loadDeposits(vals, subIssues, modelDate, self.modelDB, self.marketDB)
        return populateValueFrequencyList(vals, freq=retFreq)

class DebtToTotalAssets(DescriptorRatio):
    def __init__(self, modelDB, marketDB, useFixedFrequency=None,
                 denomProcess='extractlatest', addDeposits=False,
                 sidRanges=None, trackList=[],):
        DescriptorRatio.__init__(self, 'totalDebt', 'at', modelDB, marketDB,
                                useFixedFrequency=useFixedFrequency,
                                denomProcess=denomProcess,
                                denomNegativeTreatment='mask',
                                denomZeroTreatment='mask',
                                trackList=trackList,
                                sidRanges=sidRanges)
        self.addDeposits = addDeposits
    def loadRawNumeratorData(self, code, modelDate, startDate, endDate, data, subIssues, currency_id):
        if (self.useFixedFrequency is None) or (self.useFixedFrequency == self.QuarterlyFrequency):
            totalDebt = self.modelDB.getQuarterlyTotalDebt(
                            startDate, endDate, subIssues, modelDate,
                            self.marketDB, currency_id)
            retFreq = DataFrequency('_qtr')
        else:
            totalDebt = self.modelDB.getAnnualTotalDebt(
                            startDate, endDate, subIssues, modelDate,
                            self.marketDB, currency_id)
            retFreq = DataFrequency('_ann')
        trackList = sorted(set(self.trackList).intersection(set(subIssues)))
        outputRawData(trackList, subIssues, totalDebt, 'totalDebt%s' % retFreq)
        vals = Utilities.extractLatestValue(totalDebt)
        if self.addDeposits:
            vals = loadDeposits(vals, subIssues, modelDate, self.modelDB, self.marketDB)
        return populateValueFrequencyList(vals*self.numMultiplier, freq=retFreq)

def loadDeposits(vals, universe, date, modelDB, marketDB):
    # Ugly temporary hack to load deposits
    nonMissingIdx = numpy.flatnonzero(ma.getmaskarray(vals)==0)
    if len(nonMissingIdx) > 0:
        from riskmodels import leverageHack
        okSubIds = numpy.take(universe, nonMissingIdx, axis=0)
        dep = leverageHack.loadDeposits(okSubIds, date, modelDB, marketDB)
        for (idx, sid) in zip(nonMissingIdx, okSubIds):
            xd = dep[sid]
            if (xd is not None) and ('DPTC' in xd):
                vals[idx] += ma.filled(xd['DPTC'], 0.0)
    return vals

class DividendYield(DescriptorRatio):
    def __init__(self, modelDB, marketDB, useFixedFrequency=None, sidRanges=None, trackList=[]):
        DescriptorRatio.__init__(
                self, 'divs', 'market_cap', modelDB, marketDB,
                useFixedFrequency=useFixedFrequency, trackList=trackList,
                numeratorMultiplier=1.0, denomMultiplier=1.0)

    def loadRawDenominatorData(self, code, modelDate, startDate, endDate, data, subIssues, currency_id):
        return populateValueFrequencyList(getAverageOneMonthMarketCaps(data, subIssues), freq=self.useFixedFrequency)

    def loadRawNumeratorData(self, code, modelDate, startDate, endDate, data, subIssues, currency_id):
        currency = Utilities.Struct()
        currency.numeraire = Utilities.Struct()
        currency.numeraire.currency_id = currency_id
        divs = computeAnnualDividends(
                modelDate, subIssues, currency,
                self.modelDB, self.marketDB, maskMissing=True,
                includeSpecial=False, includeStock=False, trackList=self.trackList)
        idxMap = dict([(j,i) for (i,j) in enumerate(subIssues)])
        return populateValueFrequencyList(divs, freq=self.useFixedFrequency)

class ProxiedDividendPayout(DescriptorRatio):
    def __init__(self, modelDB, marketDB, useFixedFrequency=None):
        DescriptorRatio.__init__(self, 'divs', 'ibei', modelDB, marketDB,
                useFixedFrequency=useFixedFrequency,
                numeratorMultiplier=1.0,
                numeratorNegativeTreatment='zero',
                denomNegativeTreatment='zero',
                denomZeroTreatment='one')

    def loadRawNumeratorData(self, code, modelDate, startDate, endDate, data, subIssues, currency_id):
        currency = Utilities.Struct()
        currency.numeraire = Utilities.Struct()
        currency.numeraire.currency_id = currency_id
        divs = Utilities.screen_data(computeAnnualDividends(
                modelDate, subIssues, currency,
                self.modelDB, self.marketDB, maskMissing=False,
                includeSpecial=True, includeStock=False))
        return populateValueFrequencyList(divs, freq=self.useFixedFrequency)

class DividendPayout(DescriptorRatio):
    def __init__(self, modelDB, marketDB, useFixedFrequency=None):
        DescriptorRatio.__init__(self, 'dps', 'epsxei', modelDB, marketDB,
                useFixedFrequency=useFixedFrequency,
                numeratorNegativeTreatment='zero',
                denomNegativeTreatment='zero',
                numeratorSplitAdjust='divide',
                denomSplitAdjust='divide',
                numeratorProcess='annualize',
                denomZeroTreatment='one')

class AssetTurnover(DescriptorRatio):
    def __init__(self, modelDB, marketDB, useFixedFrequency=None):
        DescriptorRatio.__init__(self, 'sale', 'at', modelDB, marketDB,
                                useFixedFrequency=useFixedFrequency,
                                numeratorProcess='annualize')

class OperatingCashFlowToPrice(DescriptorRatio):
    def __init__(self, modelDB, marketDB):
        #For item involving cash flow, currently only allow annual frequency
        DescriptorRatio.__init__(self, 'oancf', 'market_cap', modelDB, marketDB,
                                useFixedFrequency=DescriptorRatio.AnnualFrequency,
                                denomDaysBack=2*365, denomEndDate=365)

    def loadRawDenominatorData(self, code, modelDate, startDate, endDate, data, subIssues, currency_id):
        return populateValueFrequencyList(getAverageOneMonthMarketCaps(data, subIssues), freq=self.useFixedFrequency)

class Accruals(DescriptorRatio):
    def __init__(self, modelDB, marketDB):
        #For item involving cash flow, currently only allow annual frequency
        DescriptorRatio.__init__(self, 'oancf', 'ibei', modelDB, marketDB,
                                useFixedFrequency=DescriptorRatio.AnnualFrequency)

class CurrentRatio(DescriptorRatio):
    def __init__(self, modelDB, marketDB, useFixedFrequency=None):
        DescriptorRatio.__init__(self, 'act', 'lct', modelDB, marketDB,
                                useFixedFrequency=useFixedFrequency)

class GrossMargin(DescriptorRatio):
    def __init__(self, modelDB, marketDB, useFixedFrequency=None,
                 numeratorProcess='extractlatest',
                 denomProcess='extractlatest',
                 sidRanges=None):
        DescriptorRatio.__init__(self, 'cogs', 'sale', modelDB, marketDB,
                useFixedFrequency=useFixedFrequency,
                numeratorProcess=numeratorProcess,
                denomProcess=denomProcess,
                denomNegativeTreatment='mask',
                denomZeroTreatment='mask',
                sidRanges=sidRanges)
    def getValues(self, date, data, currency_id):
        (value, freq) =  self.getRatio(data, date, currency_id)
        return (1.0 - value, freq)

# Function for loading dividends. Not ideal that it's here, but there isn't really anywhere better for it to go
def computeAnnualDividends(modelDate, subIssues, model,
        modelDB, marketDB, maskMissing=False, includeSpecial=True, includeStock=False, trackList=[]):
    """Computes total dividends paid for the past year, defined as
    reported dividends times the shares outstanding on the pay date.
    Fuzzy logic employed to account for periodic (annual, quarterly, etc)
    dividends that are not spaced apart in exactly equal time intervals.
    """
    prevYear = modelDate - datetime.timedelta(365 + 75)
    annSumFromDtDefault = modelDate - datetime.timedelta(365 + 15)
    recentDivCutoff = modelDate - datetime.timedelta(60)
    paidDividends = modelDB.getPaidDividends(
        prevYear, modelDate, subIssues, marketDB,
        convertTo=model.numeraire.currency_id,
        includeSpecial=includeSpecial, includeStock=includeStock)
    for (sid, divData) in paidDividends.items():
        if len(divData)==0:
            continue
        lastExDate = divData[-1][0]
        annSumFromDt = annSumFromDtDefault
        if lastExDate >= recentDivCutoff:
            prevDivPeriodicity = len([d for d in divData
                                      if (d[0] >= lastExDate - datetime.timedelta(365 + 15)
                                          and d[0] <= lastExDate - datetime.timedelta(365 - 15))
                                      or (d[0] >= lastExDate - datetime.timedelta(365 - 45)
                                          and d[0] <= lastExDate - datetime.timedelta(365 - 75))])
            if prevDivPeriodicity >= 1:
                annSumFromDt = max(annSumFromDt, lastExDate - datetime.timedelta(365 - 15))
            else:
                annSumFromDt = lastExDate - datetime.timedelta(365 - 45)
        toRemove = [d for d in divData if d[0] < annSumFromDt]
        for d in toRemove:
            divData.remove(d)
    trackList = sorted(set(trackList).intersection(set(subIssues)))
    outputRawData(trackList, subIssues, pandas.Series(paidDividends), 'Dividends')
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
