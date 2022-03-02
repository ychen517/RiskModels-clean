import logging
import pandas
import datetime
import numpy
import numpy.ma as ma
from riskmodels.DescriptorSources import DescriptorClass
from riskmodels.DescriptorSources import LocalDescriptorClass
from riskmodels.DescriptorSources import NumeraireDescriptorClass
from riskmodels import DescriptorSources
from riskmodels import DescriptorExposures
from riskmodels import ResearchDescriptorExposures
from riskmodels import LegacyUtilities as Utilities
from riskmodels import Matrices
from riskmodels import TimeSeriesRegression
from riskmodels import LegacyTimeSeriesRegression

# Momentum classes
from riskmodels.DescriptorSources import Momentum
class Momentum_1D(Momentum):
    """Class to compute short-term momentum.
    """
    def __init__(self, connections, gp=None):
        Momentum.__init__(self, connections, gp=gp)
        self.fromT = 1
        self.thruT = 0

class Momentum_5D(Momentum):
    """Class to compute short-term momentum.
    """
    def __init__(self, connections, gp=None):
        Momentum.__init__(self, connections, gp=gp)
        self.fromT = 5
        self.thruT = 0

class Momentum_260x21D_NoTrapWt(Momentum):
    """Class to compute medium-term momentum.
    """
    def __init__(self, connections, gp=None):
        Momentum.__init__(self, connections, gp=gp)
        self.fromT = 260
        self.thruT = 21

class Momentum_250x20D(Momentum):
    """Class to compute medium-term momentum.
    """
    def __init__(self, connections, gp=None):
        Momentum.__init__(self, connections, gp=gp)
        self.fromT = 250
        self.thruT = 10
        self.weights = 'pyramid'
        self.peak = 10
        self.peak2 = 20

class Momentum_250x20D_Legacy(Momentum):
    """Class to compute medium-term momentum.
    """
    def __init__(self, connections, gp=None):
        Momentum.__init__(self, connections, gp=gp)

class Historical_Alpha(DescriptorClass):
    """Class to compute historical alpha
    """
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)
        self.INCR = 200
        self.daysBack = 250
        self.adjustForRT = False
        self.clippedReturns = True
        self.robust = False
        self.kappa = 5.0
        self.lag = None
        self.weighting = 'pyramid'
        self.halflife = 21
        self.fadePeak = 21
        self.fillWithMarket = True
        self.outputDateList = None
        self.marketReturns = None
        self.regionPortfolioID = None

    def buildDescriptor(self, data, rootClass):
        self.log.debug('HistoricalAlpha.buildDescriptor')
        allVals = Matrices.allMasked((len(data.universe)))

        tmpReturns = self.loadReturnsArray(
                data, rootClass, daysBack=self.daysBack,
                adjustForRT=self.adjustForRT, clippedReturns=self.clippedReturns)
        if self.regionPortfolioID is not None:
            region = self.modelDB.getRiskModelRegion(self.regionPortfolioID)
        else:
            region = None

        # Check to see whether we've already run the relevant regressions
        found = False
        for item in self.readFromList:
            if item in rootClass.storedResults.keys():
                mmDict = rootClass.storedResults[item]
                found = True
                continue
        if not found:
            mmDict = dict()

        # Set up regression parameters
        params = dict()
        params['lag'] = self.lag
        params['robust'] = self.robust
        params['kappa'] = self.kappa
        params['weighting'] = self.weighting
        params['halflife'] = self.halflife
        params['fadePeak'] = self.fadePeak
        params['historyLength'] = self.daysBack

        # Initialise time-series regression class
        TSR = LegacyTimeSeriesRegression.LegacyTimeSeriesRegression(
                        TSRParameters = params,
                        fillWithMarket = self.fillWithMarket,
                        outputDateList = self.outputDateList,
                        debugOutput = self.debuggingReporting,
                        marketReturns = self.marketReturns,
                        marketRegion = region)

        # Do the regression
        if found:
            mm = mmDict[rootClass.rmg]
        else:
            mm = TSR.TSR(rootClass.rmg, tmpReturns, self.modelDB, self.marketDB)
            mmDict[rmg] = mm

        rootClass.storedResults[self.name] = mmDict
        return self.setUpReturnObject(data.universe, mm.alpha)

class Historical_Alpha_250D(Historical_Alpha):
    def __init__(self, connections, gp=None):
        Historical_Alpha.__init__(self, connections, gp=gp)
        self.name = 'Historical_Alpha_250D'
        self.readFromList = ['Historical_Residual_Volatility_250D',
                             'Market_Sensitivity_250D',
                             'Historical_Alpha_250D']

# Linked asset score classes
from riskmodels.DescriptorSources import ISC_Ret_Score
class Percent_Returns_20_Days(ISC_Ret_Score):
    def __init__(self, connections, gp=None):
        ISC_Ret_Score.__init__(self, connections, gp=gp)
        self.countPreIPODates = True
        self.daysBack = 20
        self.minDays = 0

class Percent_Returns_60_Days(ISC_Ret_Score):
    def __init__(self, connections, gp=None):
        ISC_Ret_Score.__init__(self, connections, gp=gp)
        self.countPreIPODates = True
        self.daysBack = 60
        self.minDays = 0

class Percent_Returns_125_Days(ISC_Ret_Score):
    def __init__(self, connections, gp=None):
        ISC_Ret_Score.__init__(self, connections, gp=gp)
        self.countPreIPODates = True
        self.daysBack = 125
        self.minDays = 0

class Percent_Returns_250_Days(ISC_Ret_Score):
    def __init__(self, connections, gp=None):
        ISC_Ret_Score.__init__(self, connections, gp=gp)
        self.countPreIPODates = True
        self.daysBack = 250
        self.minDays = 0

class Percent_Returns_500_Days(ISC_Ret_Score):
    def __init__(self, connections, gp=None):
        ISC_Ret_Score.__init__(self, connections, gp=gp)
        self.countPreIPODates = True
        self.daysBack = 500
        self.minDays = 0

# Volatility classes
from riskmodels.DescriptorSources import PACE_Volatility
class Volatility_125D_Legacy(PACE_Volatility):
    """Class to compute asset level MH PACE volatility
    Has no clipping
    """
    def __init__(self, connections, gp=None):
        PACE_Volatility.__init__(self, connections, gp=gp)

class Volatility_20D(PACE_Volatility):
    """Class to compute asset level very SH PACE volatility
    """
    def __init__(self, connections, gp=None):
        PACE_Volatility.__init__(self, connections, gp=gp)
        self.daysBack = 20

class Historical_Volatility(DescriptorClass):
    """Class to compute asset level historical volatility
    """
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)
        self.INCR = 200
        self.adjustForRT = False
        self.clippedReturns = True
        self.daysBack = 250

    def buildDescriptor(self, data, rootClass):
        self.log.debug('Historical_Volatility.buildDescriptor')

        # Build descriptor
        tmpReturns = self.loadReturnsArray(
                data, rootClass, daysBack=self.daysBack,
                adjustForRT=self.adjustForRT, clippedReturns=self.clippedReturns)
        vol = DescriptorExposures.generate_historic_volatility(
                tmpReturns, self.daysBack)

        # Create return value structure
        return self.setUpReturnObject(data.universe, vol)

class Historical_Volatility_125D(Historical_Volatility):
    """Class to compute asset level MH historical volatility
    """
    def __init__(self, connections, gp=None):
        Historical_Volatility.__init__(self, connections, gp=gp)
        self.daysBack = 125

class Historical_Volatility_250D(Historical_Volatility):
    """Class to compute asset level LH historical volatility
    """
    def __init__(self, connections, gp=None):
        Historical_Volatility.__init__(self, connections, gp=gp)

class Historical_Residual_Volatility(DescriptorClass):
    """Class to compute asset level historical residual volatility
    """
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)
        self.INCR = 200
        self.daysBack = 250
        self.adjustForRT = False
        self.clippedReturns = True
        self.robust = False
        self.kappa = 5.0
        self.lag = None
        self.weighting = 'pyramid'
        self.halflife = 21
        self.fadePeak = 21
        self.frequency = 'daily'
        self.fillWithMarket = True
        self.outputDateList = None
        self.marketReturns = None
        self.regionPortfolioID = None

    def buildDescriptor(self, data, rootClass):
        self.log.debug('Historical_Residual_Volatility.buildDescriptor')

        # Load the returns used
        tmpReturns = self.loadReturnsArray(
                data, rootClass, daysBack=self.daysBack,
                adjustForRT=self.adjustForRT, clippedReturns=self.clippedReturns)

        # Determine region if relevant
        if self.regionPortfolioID is not None:
            region = modelDB.getRiskModelRegion(self.regionPortfolioID)
        else:
            region = None

        # Check to see whether we've already run the relevant regressions
        found = False
        for item in self.readFromList:
            if item in rootClass.storedResults.keys():
                mmDict = rootClass.storedResults[item]
                found = True
                continue
        if not found:
            mmDict = dict()

        # Set up regression parameters
        params = dict()
        params['lag'] = self.lag
        params['robust'] = self.robust
        params['kappa'] = self.kappa
        params['weighting'] = self.weighting
        params['halflife'] = self.halflife
        params['fadePeak'] = self.fadePeak
        params['historyLength'] = self.daysBack

        # Get dates
        if self.frequency != 'daily':
            self.outputDateList = Utilities.change_date_frequency(tmpReturns.dates, frequency='weekly')

        # Initialise time-series regression class
        TSR = LegacyTimeSeriesRegression.LegacyTimeSeriesRegression(
                        TSRParameters = params,
                        fillWithMarket = self.fillWithMarket,
                        outputDateList = self.outputDateList,
                        debugOutput = self.debuggingReporting,
                        marketReturns = self.marketReturns,
                        forceRun = self.forceRun,
                        marketRegion = region)

        # Do the regression
        if found:
            mm = mmDict[rootClass.rmg]
        else:
            mm = TSR.TSR(rootClass.rmg, tmpReturns, self.modelDB, self.marketDB)
            mmDict[rootClass.rmg] = mm

        rootClass.storedResults[self.name] = mmDict
        return self.setUpReturnObject(data.universe, mm.sigma)

class Historical_Residual_Volatility_125D(Historical_Residual_Volatility):
    """Class to compute asset level MH historical residual volatility
    """
    def __init__(self, connections, gp=None):
        Historical_Residual_Volatility.__init__(self, connections, gp=gp)
        self.daysBack = 125
        self.peak = 10
        self.peak2 = 10
        self.name = 'Historical_Residual_Volatility_125D'
        self.readFromList = ['Historical_Residual_Volatility_125D',
                             'Market_Sensitivity_125D',
                             'Historical_Alpha_125D']

class Historical_Residual_Volatility_250D(Historical_Residual_Volatility):
    """Class to compute asset level LH historical residual volatility
    """
    def __init__(self, connections, gp=None):
        Historical_Residual_Volatility.__init__(self, connections, gp=gp)
        self.name = 'Historical_Residual_Volatility_250D'
        self.readFromList = ['Historical_Residual_Volatility_250D',
                             'Market_Sensitivity_250D',
                             'Historical_Alpha_250D']

# Various "beta" descriptors
from riskmodels.DescriptorSources import Market_Sensitivity_Legacy
class Market_Sensitivity_500D(Market_Sensitivity_Legacy):
    """Class to compute LH market sensitivity
    """
    def __init__(self, connections, gp=None):
        Market_Sensitivity_Legacy.__init__(self, connections, gp=gp)
        self.daysBack = 500
        self.name = 'Market_Sensitivity_500D'

class Market_Sensitivity_250D_Legacy(Market_Sensitivity_Legacy):
    """Class to compute MH market sensitivity
    """
    def __init__(self, connections, gp=None):
        Market_Sensitivity_Legacy.__init__(self, connections, gp=gp)
        self.name = 'Market_Sensitivity_250D_Legacy'
        self.weighting = None

class Market_Sensitivity_125D_Legacy(Market_Sensitivity_Legacy):
    """Class to compute MH market sensitivity
    """
    def __init__(self, connections, gp=None):
        Market_Sensitivity_Legacy.__init__(self, connections, gp=gp)
        self.name = 'Market_Sensitivity_120D_Legacy'
        self.weighting = None
        self.daysBack = 125
        self.readFromList = ['Historical_Residual_Volatility_120D',
                'Market_Sensitivity_120D', 'Historical_Alpha_120D']

from riskmodels.DescriptorSources import Regional_Market_Sensitivity
class EMxWW_Market_Sensitivity_250D(Regional_Market_Sensitivity):
    """Class to compute SH EM market sensitivity
    """
    def __init__(self, connections, gp=None):
        Regional_Market_Sensitivity.__init__(self, connections, gp=gp)
        self.daysBack = 250
        self.regionPortfolioID = 105
        self.regionBasePortfolioID = 104
        self.name = 'EMxWW_Market_Sensitivity_250D'

class EMxWW_Market_Sensitivity_500D(Regional_Market_Sensitivity):
    """Class to compute MH EM market sensitivity
    """
    def __init__(self, connections, gp=None):
        Regional_Market_Sensitivity.__init__(self, connections, gp=gp)
        self.daysBack = 500
        self.regionPortfolioID = 105
        self.regionBasePortfolioID = 104
        self.name = 'EMxWW_Market_Sensitivity_500D'

class XRate_Sensitivity(DescriptorClass):
    """Class to compute raw exchange rate sensitivity score
    """
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)
        self.INCR = 200
        self.numeraire = 'USD'
        self.daysBack = 250
        self.adjustForRT = False
        self.clippedReturns = True
        self.robust = False
        self.kappa = 5.0
        self.frequency = 'weekly'
        self.weights = None
        self.peak = 4
        self.peak2 = 4

    def createFakeModelSelector(self, data, rc):
        fakeModel = Utilities.Struct()
        fakeModel.rmg = [rc.rmg]
        fakeModel.rmgAssetMap = data.rmgAssetMap
        self.log.info('Created modelSelector with rmgList: %s, currencies: %s'%\
                          (','.join('%s'%k.mnemonic for k in fakeModel.rmg),
                           ','.join('%s'%k.currency_code for k in fakeModel.rmg)))

        return fakeModel

    def buildDescriptor(self, data, rootClass):
        self.log.debug('XRate_Sensitivity.buildDescriptor')
        # Initialise data
        fakeModel = self.createFakeModelSelector(data, rootClass)
        tmpReturns = self.loadReturnsArray(
                data, rootClass, daysBack=self.daysBack,
                adjustForRT=self.adjustForRT, clippedReturns=self.clippedReturns)

        # Create parameter data
        params = Utilities.Struct()
        params.numeraire = self.numeraire
        params.daysBack = min(self.daysBack, tmpReturns.data.shape[1])
        params.frequency = self.frequency
        params.weights = self.weights
        params.peak = self.peak
        params.peak2 = self.peak2

        value = ResearchDescriptorExposures.generate_forex_sensitivity(
            data, tmpReturns, fakeModel, self.modelDB, self.marketDB, params)

        return self.setUpReturnObject(data.universe, value)

class XRate_52W_EUR(XRate_Sensitivity):
    """Class to compute EUR exchange rate sensitivity
    """
    def __init__(self, connections, gp=None):
        XRate_Sensitivity.__init__(self, connections, gp=gp)
        self.numeraire = 'EUR'

class XRate_52W_GBP(XRate_Sensitivity):
    """Class to compute GBP exchange rate sensitivity
    """
    def __init__(self, connections, gp=None):
        XRate_Sensitivity.__init__(self, connections, gp=gp)
        self.numeraire = 'GBP'

class XRate_52W_JPY(XRate_Sensitivity):
    """Class to compute JPY exchange rate sensitivity
    """
    def __init__(self, connections, gp=None):
        XRate_Sensitivity.__init__(self, connections, gp=gp)
        self.numeraire = 'JPY'

class XRate_104W_XDR_Legacy(XRate_Sensitivity):
    """Class to compute SDR exchange rate sensitivity
    Legacy version
    """
    def __init__(self, connections, gp=None):
        XRate_Sensitivity.__init__(self, connections, gp=gp)
        self.numeraire = 'XDR'
        self.daysBack = 500

class XRate_104W_EUR(XRate_Sensitivity):
    """Class to compute EUR exchange rate sensitivity
    """
    def __init__(self, connections, gp=None):
        XRate_Sensitivity.__init__(self, connections, gp=gp)
        self.numeraire = 'EUR'
        self.daysBack = 500
        self.weights = 'pyramid'
        self.peak = 4
        self.peak2 = 4

class XRate_104W_GBP(XRate_Sensitivity):
    """Class to compute GBP exchange rate sensitivity
    """
    def __init__(self, connections, gp=None):
        XRate_Sensitivity.__init__(self, connections, gp=gp)
        self.numeraire = 'GBP'
        self.daysBack = 500
        self.weights = 'pyramid'
        self.peak = 4
        self.peak2 = 4

class XRate_104W_JPY(XRate_Sensitivity):
    """Class to compute JPY exchange rate sensitivity
    """
    def __init__(self, connections, gp=None):
        XRate_Sensitivity.__init__(self, connections, gp=gp)
        self.numeraire = 'JPY'
        self.daysBack = 500
        self.weights = 'pyramid'
        self.peak = 4
        self.peak2 = 4

# Size and liquidity descriptors
from riskmodels.DescriptorSources import LnTrading_Activity
class LnTrading_Activity_250D(LnTrading_Activity):
    """Class to compute 250 day (365 calendar-day) ln of trading activity
    """
    def __init__(self, connections, gp=None):
        LnTrading_Activity.__init__(self, connections, gp=gp)
        self.daysBack = 365

class LnTrading_Activity_20D_Median(LnTrading_Activity):
    """Class to compute 20-day ln of median trading activity
    """
    def __init__(self, connections, gp=None):
        LnTrading_Activity.__init__(self, connections, gp=gp)
        self.daysBack = 30
        self.median = True

class LnTrading_Activity_60D_Median(LnTrading_Activity):
    """Class to compute 60-day ln of median trading activity
    """
    def __init__(self, connections, gp=None):
        LnTrading_Activity.__init__(self, connections, gp=gp)
        self.median = True

class LnTrading_Activity_125D_Median(LnTrading_Activity):
    """Class to compute 125-day ln of median trading activity
    """
    def __init__(self, connections, gp=None):
        LnTrading_Activity.__init__(self, connections, gp=gp)
        self.daysBack = 180
        self.median = True

class Dollar_Volume_60D(LnTrading_Activity):
    """Class to compute 60-day volume traded
    """
    def __init__(self, connections, gp=None):
        LnTrading_Activity.__init__(self, connections, gp=gp)
        self.simple = True
        self.lnComb = False

class Dollar_Volume_20D(LnTrading_Activity):
    """Class to compute 20-day volume traded
    """
    def __init__(self, connections, gp=None):
        LnTrading_Activity.__init__(self, connections, gp=gp)
        self.simple = True
        self.lnComb = False
        self.daysBack = 30

from riskmodels.DescriptorSources import Amihud_Liquidity
class Amihud_Liquidity_250D(Amihud_Liquidity):
    """Class to compute 125-day Amihud liquidity measure
    """
    def __init__(self, connections, gp=None):
        Amihud_Liquidity.__init__(self, connections, gp=gp)
        self.daysBack = 250

#########  Growth descriptors ##########
# note that all of the following descriptors 
# have had minimal testing, using US QTR 
# data if they were tested at all

class Sales_Growth_RPF_Quarterly(DescriptorClass):
    """Class to compute raw Est.Sales Growth scores"""
    
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)

    def buildDescriptor(self, data, rootClass):
        values = ResearchDescriptorExposures.generate_growth_rate(
                'sale', data.universe, rootClass.dates[0], rootClass.numeraire_id,
                self.modelDB, self.marketDB, winsoriseRaw=False,
                useQuarterlyData=True, forecastItem='rev_median_ann', forecastItemScaleByTSO=False)
        return self.setUpReturnObject(data.universe, values)

class Earnings_Growth_RPF_Quarterly(DescriptorClass):
    """Class to compute raw Est.Earnings Growth scores"""
    
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)

    def buildDescriptor(self, data, rootClass):
        values = ResearchDescriptorExposures.generate_growth_rate(
                'ibei', data.universe, rootClass.dates[0], rootClass.numeraire_id,
                self.modelDB, self.marketDB, winsoriseRaw=False,
                useQuarterlyData=True, forecastItem='eps_median_ann', forecastItemScaleByTSO=True)
        return self.setUpReturnObject(data.universe, values)

class Earnings_Variability_Quarterly(DescriptorClass):
    """Class to compute raw Earnings variability scores"""
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)

    def buildDescriptor(self, data, rootClass):
        values = ResearchDescriptorExposures.generate_growth_rate(
                'ibei', data.universe, rootClass.dates[0], rootClass.numeraire_id,
                self.modelDB, self.marketDB, winsoriseRaw=False,
                useQuarterlyData=True, forecastItem=None, getVar=True)
        return self.setUpReturnObject(data.universe, values)

class Earnings_Variability_Annual(DescriptorClass):
    """Class to compute raw Earnings variability scores"""
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)

    def buildDescriptor(self, data, rootClass):
        values = ResearchDescriptorExposures.generate_growth_rate(
                'ibei', data.universe, rootClass.dates[0], rootClass.numeraire_id,
                self.modelDB, self.marketDB, winsoriseRaw=False,
                useQuarterlyData=False, forecastItem=None, getVar=True)
        return self.setUpReturnObject(data.universe, values)

class Sales_Growth(DescriptorClass):
    """Class to compute raw Sales Growth scores"""

    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)

    def buildDescriptor(self, data, rootClass):
        values = ResearchDescriptorExposures.generate_growth_rate(
                'sale', data.universe, rootClass.dates[0], rootClass.numeraire_id,
                self.modelDB, self.marketDB, winsoriseRaw=True)
        return self.setUpReturnObject(data.universe, values)

class Earnings_Growth(DescriptorClass):
    """Class to compute raw Earnings Growth scores"""

    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)

    def buildDescriptor(self, data, rootClass):
        values = ResearchDescriptorExposures.generate_growth_rate(
                'ibei', data.universe, rootClass.dates[0], rootClass.numeraire_id,
                self.modelDB, self.marketDB, winsoriseRaw=True)
        return self.setUpReturnObject(data.universe, values)

class Sales_Growth_RPF(DescriptorClass):
    """Class to compute raw Est.Sales Growth scores"""
    
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)

    def buildDescriptor(self, data, rootClass):
        values = ResearchDescriptorExposures.generate_growth_rateOLD(
                'sale', data.universe, rootClass.dates[0], rootClass.numeraire_id, 
                self.modelDB, self.marketDB, winsoriseRaw=True,
                forecastItem='rev_median_ann')
        return self.setUpReturnObject(data.universe, values)

class Earnings_Growth_RPF(DescriptorClass):
    """Class to compute raw Est.Earnings Growth scores"""
    
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)

    def buildDescriptor(self, data, rootClass):
        values = ResearchDescriptorExposures.generate_growth_rateOLD(
                'ibei', data.universe, rootClass.dates[0], rootClass.numeraire_id,
                self.modelDB, self.marketDB, winsoriseRaw=True,
                forecastItem='eps_median_ann')
        return self.setUpReturnObject(data.universe, values)

class Sales_Growth_RPF_SD(DescriptorClass):
    """Class to compute raw Est.Sales Growth scores"""
    
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)

    def buildDescriptor(self, data, rootClass):
        values = ResearchDescriptorExposures.generate_growth_rateSD(
                'sale', data.universe, rootClass.dates[0], rootClass.numeraire_id,
                self.modelDB, self.marketDB, winsoriseRaw=False,
                forecastItem='rev_median_ann', forecastItemScaleByTSO=False)
        return self.setUpReturnObject(data.universe, values)

class Earnings_Growth_RPF_SD(DescriptorClass):
    """Class to compute raw Est.Earnings Growth scores"""
    
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)

    def buildDescriptor(self, data, rootClass):
        values = ResearchDescriptorExposures.generate_growth_rateSD(
                'ibei', data.universe, rootClass.dates[0], rootClass.numeraire_id,
                self.modelDB, self.marketDB, winsoriseRaw=False,
                forecastItem='eps_median_ann', forecastItemScaleByTSO=True)
        return self.setUpReturnObject(data.universe, values)

class Long_Term_Growth(DescriptorClass):
    """Class to compute raw Long Term Growth scores"""
    
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)

    def buildDescriptor(self, data, rootClass):
        values = ResearchDescriptorExposures.generate_long_term_growth_rate(
                'eps_median_ltg', data.universe, rootClass.dates[0], rootClass.numeraire_id,
                self.modelDB, self.marketDB)
        return self.setUpReturnObject(data.universe, values)

class Long_Term_Growth_Trimmed(DescriptorClass):
    """Class to compute raw Long Term Growth scores"""
    
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)

    def buildDescriptor(self, data, rootClass):
        values = ResearchDescriptorExposures.generate_long_term_growth_rate(
                'eps_median_ltg', data.universe, rootClass.dates[0], rootClass.numeraire_id,
                self.modelDB, self.marketDB, trim=True)
        return self.setUpReturnObject(data.universe, values)

class Est_Earnings_to_Price_Quarterly(DescriptorClass):
    """Class to compute Est Earnings-to-Price scores"""

    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)

    def buildDescriptor(self, data, rootClass):
        params = Utilities.Struct()
        params.maskNegative = False

        values = DescriptorExposures.generate_est_earnings_to_price(
            rootClass.dates[0], data, self.modelDB, self.marketDB, params,
            rootClass.numeraire_id)
        return self.setUpReturnObject(data.universe, values)

# Leverage items
class Debt_to_Assets_incDeposits(DescriptorClass):
    """Class to compute raw Debt to Asset scores with short-term deposits
    added to the numerator"""

    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)

    def buildDescriptor(self, data, rootClass):
        values, freq = DescriptorRatios.DebtToTotalAssets(self.modelDB, self.marketDB,
                sidRanges=data.sidRanges, addDeposits=True).\
                getValues(rootClass.dates[0], data, rootClass.numeraire_id)
        return self.setUpReturnObject(data.universe, values)

class Debt_to_MarketCap_Quarterly(DescriptorClass):
    """Class to compute raw Debt to MCap scores"""

    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)

    def buildDescriptor(self, data, rootClass):
        values, freq = DescriptorRatios.DebtToMarketCap(self.modelDB, self.marketDB,
                    sidRanges=data.sidRanges).\
                    getValues(rootClass.dates[0], data, rootClass.numeraire_id)
        return self.setUpReturnObject(data.universe, values)

class Debt_to_MarketCap_Annual(DescriptorClass):
    """Class to compute raw Debt to MCap scores"""

    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)

    def buildDescriptor(self, data, rootClass):
        values, freq = DescriptorRatios.DebtToMarketCap(self.modelDB, self.marketDB,
                    DescriptorRatios.DescriptorRatio.AnnualFrequency,sidRanges=data.sidRanges).\
                    getValues(rootClass.dates[0], data, rootClass.numeraire_id)
        return self.setUpReturnObject(data.universe, values)

# Value items
class Earnings_to_PriceV2_Quarterly(DescriptorClass):
    """Class to compute earnings-to-price scores"""

    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)

    def buildDescriptor(self, data, rootClass):
        values, freq = DescriptorRatios.EarningsToPrice(self.modelDB, self.marketDB,
                    numeratorProcess='annualize',
                    numeratorNegativeTreatment='zero',
                    sidRanges=data.sidRanges).\
            getValues(rootClass.dates[0], data, rootClass.numeraire_id)
        return self.setUpReturnObject(data.universe, values)

class Est_Earnings_to_Price_12MFLV2_Quarterly(DescriptorClass):
    """Class to compute Est Earnings-to-Price scores"""

    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)

    def buildDescriptor(self, data, rootClass):
        params = Utilities.Struct()
        params.maskNegative = False

        values = DescriptorExposures.generate_est_earnings_to_price_12MFL(
            rootClass.dates[0], data, self.modelDB, self.marketDB, params,
            rootClass.numeraire_id, negativeTreatment='zero')
        return self.setUpReturnObject(data.universe, values)

# Dividend descriptors
class Proxied_Dividend_Payout_Quarterly(DescriptorClass):
    """Class to compute raw Plowback times ROE"""

    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)

    def buildDescriptor(self, data, rootClass):
        values, freq = DescriptorRatios.ProxiedDividendPayout(self.modelDB, self.marketDB).\
                    getValues(rootClass.dates[0], data, rootClass.numeraire_id)
        return self.setUpReturnObject(data.universe, values)

# QMJ Profitability descriptors
# Note quarterly cashflow data is not safe to use
class CashFlow_to_Assets_Quarterly(DescriptorClass):
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)

    def buildDescriptor(self, data, rootClass):
        values, freq = DescriptorRatios.CashFlowToAssets(self.modelDB, self.marketDB,
                sidRanges=data.sidRanges).\
                getValues(rootClass.dates[0], data, rootClass.numeraire_id)
        return self.setUpReturnObject(data.universe, values)

# Note quarterly cashflow data is not safe to use
class CashFlow_to_Income_Quarterly(DescriptorClass):
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)

    def buildDescriptor(self, data, rootClass):
        values, freq = DescriptorRatios.CashFlowToIncome(self.modelDB, self.marketDB,
                    sidRanges=data.sidRanges).\
                getValues(rootClass.dates[0], data, rootClass.numeraire_id)
        return self.setUpReturnObject(data.universe, values)

# QMJ payout descriptors
class Net_Equity_Issuance(DescriptorClass):
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)

    def buildDescriptor(self, data, rootClass):
        values = DescriptorExposures.generate_net_equity_issuance(
                rootClass.dates[0], data, self.modelDB, self.marketDB, rootClass.numeraire_id)
        return self.setUpReturnObject(data.universe, values)

class Net_Debt_Issuance(DescriptorClass):
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)

    def buildDescriptor(self, data, rootClass):
        values = DescriptorExposures.generate_net_debt_issuance(
                rootClass.dates[0], data, self.modelDB, self.marketDB, rootClass.numeraire_id)
        return self.setUpReturnObject(data.universe, values)

class Net_Payout_Over_Profits(DescriptorClass):
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)

    def buildDescriptor(self, data, rootClass):
        values = DescriptorExposures.generate_net_payout_over_profits(
                rootClass.dates[0], data, self.modelDB, self.marketDB, rootClass.numeraire_id)
        return self.setUpReturnObject(data.universe, values)

# QMJ safety descriptors
class Altman_Z_Score(DescriptorClass):
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)

    def buildDescriptor(self, data, rootClass):
        wc1, freq = DescriptorRatios.CurrentLiabilitiesToAssets(self.modelDB, self.marketDB).\
                getValues(rootClass.dates[0], data, rootClass.numeraire_id)
        wc2, freq = DescriptorRatios.CurrentToTotalAssets(self.modelDB, self.marketDB).\
                getValues(rootClass.dates[0], data, rootClass.numeraire_id)
        re, freq = DescriptorRatios.CommonEquityToAssets(self.modelDB, self.marketDB).\
                getValues(rootClass.dates[0], data, rootClass.numeraire_id)
        ebit, freq = DescriptorRatios.EBITDAToAssets(self.modelDB, self.marketDB).\
                getValues(rootClass.dates[0], data, rootClass.numeraire_id)
        me, freq = DescriptorRatios.MarketEquityToAssets(self.modelDB, self.marketDB).\
                getValues(rootClass.dates[0], data, rootClass.numeraire_id)
        sale, freq = DescriptorRatios.SalesToAssets(self.modelDB, self.marketDB).\
                getValues(rootClass.dates[0], data, rootClass.numeraire_id)
        z_score = (1.2*(wc2-wc1)) + (1.4*re) + (3.3*ebit) + (0.6*me) + sale
        return self.setUpReturnObject(data.universe, z_score)

class Variability_of_ROE(DescriptorClass):
    """Class to compute variability of ROE scores"""

    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)

    def buildDescriptor(self, data, rootClass):
        values = ResearchDescriptorExposures.generate_evol(
                data.universe, rootClass.dates[0], rootClass.numeraire_id,
                self.modelDB, self.marketDB)
        return self.setUpReturnObject(data.universe, values)

# QMJ Growth descriptors
class Return_on_Equity_Delta(DescriptorClass):
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)

    def buildDescriptor(self, data, rootClass):
        values, freq = DescriptorRatios.ReturnOnEquity(
                self.modelDB, self.marketDB, lagDenom=6*365).\
                getValues(rootClass.dates[0], data, rootClass.numeraire_id)
        values_lag, freq = DescriptorRatios.ReturnOnEquity(
                self.modelDB, self.marketDB, lagNum=5*365, lagDenom=6*365).\
                getValues(rootClass.dates[0], data, rootClass.numeraire_id)
        values = values - values_lag
        return self.setUpReturnObject(data.universe, values)

class Return_on_Assets_Delta(DescriptorClass):
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)

    def buildDescriptor(self, data, rootClass):
        values, freq = DescriptorRatios.ReturnOnAssets(
                self.modelDB, self.marketDB, lagDenom=6*365).\
                getValues(rootClass.dates[0], data, rootClass.numeraire_id)
        values_lag, freq = DescriptorRatios.ReturnOnAssets(
                self.modelDB, self.marketDB, lagNum=5*365, lagDenom=6*365).\
                getValues(rootClass.dates[0], data, rootClass.numeraire_id)
        values = values - values_lag
        return self.setUpReturnObject(data.universe, values)

class Gross_Margin_Delta(DescriptorClass):
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)

    def buildDescriptor(self, data, rootClass):
        values, freq = DescriptorRatios.GrossMargin(
                self.modelDB, self.marketDB, lagDenom=5*365).\
                getValues(rootClass.dates[0], data, rootClass.numeraire_id)
        values_lag, freq = DescriptorRatios.GrossMargin(
                self.modelDB, self.marketDB, lagNum=5*365, lagDenom=5*365).\
                getValues(rootClass.dates[0], data, rootClass.numeraire_id)
        values = values - values_lag
        return self.setUpReturnObject(data.universe, values)

class Sales_to_Assets_Delta(DescriptorClass):
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)

    def buildDescriptor(self, data, rootClass):
        values, freq = DescriptorRatios.SalesToAssets(
                self.modelDB, self.marketDB, lagDenom=6*365).\
                getValues(rootClass.dates[0], data, rootClass.numeraire_id)
        values_lag, freq = DescriptorRatios.SalesToAssets(
                self.modelDB, self.marketDB, lagNum=5*365, lagDenom=6*365).\
                getValues(rootClass.dates[0], data, rootClass.numeraire_id)
        values = values - values_lag
        return self.setUpReturnObject(data.universe, values)

class CashFlow_to_Assets_Delta(DescriptorClass):
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)

    def buildDescriptor(self, data, rootClass):
        values, freq = DescriptorRatios.CashFlowToAssets(
                self.modelDB, self.marketDB, lagDenom=6*365).\
                        getValues(rootClass.dates[0], data, rootClass.numeraire_id)
        values_lag, freq = DescriptorRatios.CashFlowToAssets(
                self.modelDB, self.marketDB, lagNum=5*365, lagDenom=6*365).\
                        getValues(rootClass.dates[0], data, rootClass.numeraire_id)
        values = values - values_lag
        return self.setUpReturnObject(data.universe, values)

class CashFlow_to_Income_Delta(DescriptorClass):
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)

    def buildDescriptor(self, data, rootClass):
        values, freq = DescriptorRatios.CashFlowToIncome(
                self.modelDB, self.marketDB, lagDenom=5*365).\
                        getValues(rootClass.dates[0], data, rootClass.numeraire_id)
        values_lag, freq = DescriptorRatios.CashFlowToIncome(
                self.modelDB, self.marketDB, lagNum=5*365, lagDenom=5*365).\
                        getValues(rootClass.dates[0], data, rootClass.numeraire_id)

# Miscellaneous stuff that hasn't been used yet
class Earnings_Variability_Annual(DescriptorClass):
    """Class to compute raw Earnings variability scores"""
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)

    def buildDescriptor(self, data, rootClass):
        values = DescriptorExposures.generate_growth_rate(
                'ibei', data.universe, rootClass.dates[0], rootClass.numeraire_id,
                self.modelDB, self.marketDB, daysBack =(8*366), winsoriseRaw=False,
                useQuarterlyData=False, forecastItem=None, getVar=True)
        return self.setUpReturnObject(data.universe, values)

class Accruals(DescriptorClass):
    """Class to compute raw asset turnover
    """
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)

    def buildDescriptor(self, data, rootClass):
        values, freq = DescriptorRatios.Accruals(self.modelDB, self.marketDB).\
                getValues(rootClass.dates[0], data, rootClass.numeraire_id)
        return self.setUpReturnObject(data.universe, values)

class Asset_Turnover(DescriptorClass):
    """Class to compute raw asset turnover
    """
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)

    def buildDescriptor(self, data, rootClass):
        values, freq = DescriptorRatios.AssetTurnover(self.modelDB, self.marketDB).\
                    getValues(rootClass.dates[0], data, rootClass.numeraire_id)
        return self.setUpReturnObject(data.universe, values)

class Operating_Cash_Flow_To_Price(DescriptorClass):
    """Class to compute raw asset turnover
    """
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)

    def buildDescriptor(self, data, rootClass):
        values, freq = DescriptorRatios.OperatingCashFlowToPrice(self.modelDB, self.marketDB).\
                    getValues(rootClass.dates[0], data, rootClass.numeraire_id)
        return self.setUpReturnObject(data.universe, values)

class Operating_Cash_Flow_To_Assets(DescriptorClass):
    """Class to compute raw asset turnover
    """
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)

    def buildDescriptor(self, data, rootClass):
        values, freq = DescriptorRatios.OperatingCashFlowToAssets(self.modelDB, self.marketDB).\
                    getValues(rootClass.dates[0], data, rootClass.numeraire_id)
        return self.setUpReturnObject(data.universe, values)

class Current_Ratio(DescriptorClass):
    """Class to compute raw asset turnover
    """
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)

    def buildDescriptor(self, data, rootClass):
        values, freq = DescriptorRatios.CurrentRatio(self.modelDB, self.marketDB).\
                    getValues(rootClass.dates[0], data, rootClass.numeraire_id)
        return self.setUpReturnObject(data.universe, values)

########## Time series sensitivities ##########
class TS_Sensitivity(DescriptorClass):
    """Class to compute time series sensitivies
    """
    def __init__(self, connections, gp=None):
        DescriptorClass.__init__(self, connections, gp=gp)
      
        # History and weighting
        self.historyInYears = 2
        self.daysBack = self.historyInYears*252 + 40 # number of daily asset returns used
        self.weighting = 'pyramid' # exponential, triangle, pyramid, or None (equal) 
        self.frequency = 'weekly' # weekly, daily, monthly
        if self.frequency == 'daily':
            self.nobs = self.historyInYears*252
            self.halflife = 21
            self.fadePeak = 21
        elif self.frequency == 'weekly':
            self.nobs = self.historyInYears*52
            self.halflife = 4
            self.fadePeak = 4
        elif self.frequency == 'monthly':
            self.nobs = self.historyInYears*12
            self.halflife = 2
            self.fadePeak = 2
       
        # Parameters for retrieving DAILY asset returns 
        self.applyProxy = False # Proxy missing returns 
        self.adjustForRT = False # Adjust returns for returns timing (set to True if using daily returns) 
        self.clippedReturns = True # Generally recommended
        self.regionPortfolioID = None 

        # Parameters for filling missing (compounded) returns
        self.fillWithMarket = True # fill missing (compounded) returns
                                   # with market returns; if false,
                                   # fill with zeros
        self.fillInputsWithZeros = True # fill missing (compounded) inputs with zeros

        # Regression settings
        self.inclIntercept = True
        self.robust = False
        self.kappa = 5.0
        self.lag = None
        
        # Parameters for (optionally) saving regression stats
        self.getRegStats = False
        self.collName = 'TS_RegressStats'

    def getCurrencyReturnsHistory(self, startDate, endDate, 
                                  ccyBase, ccyQuote, rmg):
        """Load exchange rate returns where are exchange rates
           are specified as ccyQuote/ccyBase (or, in other words
           1 unit of ccyBase = x units of ccyQuote)"""
        if type(rmg) is not list:
            rmg = [rmg]
        daysBack = (endDate - startDate).days + 1
        """
        fxHistory = self.modelDB.loadExchangeRateHistory(rmg, 
                endDate, daysBack, [ccyBase], ccyQuote, 
                idlookup=False) 
        """
        fxRetHistory = self.modelDB.loadCurrencyReturnsHistory(rmg,
                endDate, daysBack, [ccyBase], ccyQuote, 
                idlookup=False)
        ccyRets = fxRetHistory.toDataFrame().T[ccyBase]
        return ccyRets.reindex(index=[d for d in ccyRets.index
            if d>= startDate and d<=endDate])

    def getMarketReturn(self, startDate, endDate, rmg):
        """Load market returns"""
        mktDateList = self.modelDB.getDateRange(\
                None, startDate, endDate)
        if self.regionPortfolioID is not None: 
            region = self.modelDB.getRiskModelRegion(self.regionPortfolioID)
            marketReturns = ma.filled(modelDB.loadRegionReturnHistory(
                mktDateList, [region]).data[0,:], 0.0)
        else:
            if type(rmg) is not list:
                rmg = [rmg]
            marketReturns = ma.filled(self.modelDB.loadRMGMarketReturnHistory(
                                mktDateList, rmg, robust=True).data[0,:], 0.0)
        return pandas.Series(marketReturns, index=mktDateList)

    def getIndustryReturns(self, startDate, endDate, rmg):
        """Get industry returns from mongodb"""
        firstDate = datetime.date(2000, 1, 1)  # stats are only available from this date
        if startDate > firstDate:
            import proddatactrl
            dataprov = proddatactrl.ProdDataController(sid='research')
            if rmg.rmg_id == 1:
                rmname = 'USAxioma2013MH'
                tablename = '%sdaily' % rmname
            elif rmg.rmg_id == 10:
                rmname = 'CAAxioma2009MH'
                tablename = rmname
            else:
                rmname = 'USAxioma2013MH'
                tablename = '%sdaily' % rmname
            # get sector returns from pymongo
            client = pymongo.MongoClient(host='prefix')
            db = client.sector_stats
            coll = db[tablename]
            searchDict = {'d1': {'$gte': str(startDate),
                                 '$lte': str(endDate)} }
            cursor = coll.find(searchDict)
            indRets = dict()
            for doc in cursor:
                if 'returns' in doc.keys():
                    indRets[datetime.datetime.strptime(str(doc['d1']), '%Y-%m-%d').date()] = doc['returns']
            return pandas.DataFrame.from_dict(indRets)
        else:
            return pandas.DataFrame()
    
    def processReturns(self, assetReturns, factorReturns, rmg):
        """Get commonDates in assetReturns and factorReturns and
           compound returns to match commonDates. Then get dates
           associated with desired frequency and compound returns
           to match that frequency. Optionally fill missing 
           assetReturns with market return.
           Returns time series matrices: assetReturns, factorReturns"""
        
        # get common dates and compound returns for those date
        fRets = factorReturns.toDataFrame().T.sort_index()
        allDateSets = [set(assetReturns.dates)]
        for factor in factorReturns.assets:
            allDateSets.append(set(fRets[factor].dropna().index))
        commonDates = sorted(set.intersection(*allDateSets))

        if assetReturns.dates != commonDates:
            assetReturns.data = Utilities.compute_compound_returns_v4(
                    assetReturns.data, assetReturns.dates, commonDates,
                    fillWithZeros=False)[0]

        if factorReturns.dates != commonDates:
            factorReturns.data = Utilities.compute_compound_returns_v4(
                    factorReturns.data, factorReturns.dates, commonDates,
                    fillWithZeros=self.fillInputsWithZeros)[0]

        # get date list for specified frequency
        delta = max(commonDates) - min(commonDates)
        fullDateList = [min(commonDates) + datetime.timedelta(days=i) 
                        for i in range(delta.days + 1)]
        if self.frequency == 'weekly':
            periodDateList = [d for d in fullDateList if d.weekday() == 4]
        elif self.frequency == 'monthly':
            periodDateList = sorted(list(set([
                datetime.date(d.year, d.month, monthrange(d.year, d.month)[1]) 
                for d in fullDateList if
                datetime.date(d.year, d.month, monthrange(d.year, d.month)[1]) 
                <= max(commonDates)])))
        else:
            periodDateList = list(commonDates)
        periodDateList.sort()

        # compound returns to match specified frequency
        if commonDates != periodDateList:
            assetReturns.data = Utilities.compute_compound_returns_v4(
                    assetReturns.data, commonDates, periodDateList,
                    fillWithZeros=False)[0]
            factorReturns.data = Utilities.compute_compound_returns_v4(
                    factorReturns.data, commonDates, periodDateList,
                    fillWithZeros=self.fillInputsWithZeros)[0]
       
        # take last nobs observations
        assert(factorReturns.data.shape[1] >= self.nobs and \
                assetReturns.data.shape[1] >= self.nobs)
        factorReturns.data = factorReturns.data[:, -self.nobs:]
        assetReturns.data = assetReturns.data[:, -self.nobs:]
        finalDateList = periodDateList[-self.nobs:]

        # fill missing asset returns (ensuring each has a complete history)
        if self.fillWithMarket:
            if 'Market' in factorReturns.assets:
                mktVals = factorReturns.data[factorReturns.assets.index('Market'), :]
            else:
                mktRets =self.getMarketReturn(min(commonDates),
                        max(commonDates), rmg)
                if list(mktRets.index) != periodDateList:
                    mktVals = Utilities.compute_compound_returns_v4(
                            mktRets.values, list(mktRets.index), periodDateList)[0]
                    # take last nobs observations
                    mktVals = mktVals[-self.nobs:]
                    assert(len(mktVals) == assetReturns.data.shape[1])
            # report of number of observations to be filled with mkt return per asset
            if self.debuggingReporting:
                propFilled = ma.getmaskarray(assetReturns.data).sum(axis=1)/float(assetReturns.data.shape[1])
                propFilled = pandas.Series(propFilled, index=[s.getSubIdString() for s in assetReturns.assets])
                propFilled.to_csv('tmp/propFilled-%s-%s.csv' % (self.name, max(finalDateList).isoformat()))
            maskedReturns = numpy.array(
                    ma.getmaskarray(assetReturns.data), dtype='float')
            for ii in range(len(mktVals)):
                maskedReturns[:,ii] *= mktVals[ii]
            assetReturns.data = ma.filled(assetReturns.data, 0.0)
            assetReturns.data += maskedReturns
        else:
            assetReturns.data = ma.filled(assetReturns.data, 0.0)
       
        # return reformatted time series matrices
        assert (assetReturns.data.shape[1] == factorReturns.data.shape[1])

        retFactorReturns = Matrices.TimeSeriesMatrix(factorReturns.assets, 
                finalDateList)
        retFactorReturns.data = factorReturns.data

        retAssetReturns = Matrices.TimeSeriesMatrix(assetReturns.assets,
                finalDateList)
        retAssetReturns.data = assetReturns.data
        
        return (retAssetReturns, retFactorReturns)

    def saveRegressStats(self, modelDate, rmg_id, data):
        coll = self.mongoDB[self.collName]

        # upsert regress stats
        modelDate = datetime.datetime(modelDate.year,
                                      modelDate.month,
                                      modelDate.day)
        baseDict = {'rmg_id': rmg_id,
                    'dt': modelDate,
                    'regression': self.name}
        dataDict = {}
        dataDict['sub_issue_ids'] = data.subids
        dataDict['tstats'] = data.tstatDict
        dataDict['pvals'] = data.pvalDict
        dataDict['rsquare'] = data.rsq
        dataDict['vifs'] = data.vifs
        dataDict['dof'] = data.dof 

        try:
            res = coll.update_one(baseDict, {'$set': dataDict}, upsert=True)
        except:
            self.log.exception('Unexpected error: %s' % sys.exc_info()[0])

class TermSpread_Sensitivity(TS_Sensitivity):
    """Base class for term spread sensitivities"""
    def __init__(self, connections, gp=None):
        TS_Sensitivity.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__

        # Regression settings
        self.inclIntercept = True
        self.weighting = 'pyramid' # exponential, triangle, pyramid, or None (equal) 

        # name of factor sensitivity to save
        self.factorSensitivity = 'Term Spread'

    def getTermSpread(self, startDate, endDate, ccy):
        metaData = {'treasury_yield_10y': 'M000010001',
                    'treasury_yield_13w': 'M000010002' } 
        rawSeries = pandas.DataFrame.from_dict(
            self.marketDB.getMacroTs(list(metaData.values()),
            datetime.date(1980,1,1)))

        # adjust data as apparently the raw data is 10x
        # larger than it should be
        rawSeries = rawSeries/10 

        # convert to number (from percent)
        rawSeries = rawSeries/100

        termSpread = rawSeries[metaData['treasury_yield_10y']] - \
            rawSeries[metaData['treasury_yield_13w']]
        termSpread = termSpread.sort_index(ascending=True)

        # compute first differences
        termSpread = termSpread.diff().dropna()
        idx = [d for d in termSpread.index if d.date() >= startDate
               and d.date() <= endDate]
        termSpread = termSpread.reindex(index=idx)
        return dict(zip([d.date() for d in termSpread.index], 
            termSpread.values))

    def buildDescriptor(self, data, rootClass):
        self.log.debug('%s.buildDescriptor' % self.name)
        
        # Set up regression 1 parameters
        params1 = dict()
        params1['lag'] = self.lag
        params1['robust'] = self.robust
        params1['kappa'] = self.kappa
        params1['weighting'] = self.weighting
        params1['halflife'] = self.halflife
        params1['fadePeak'] = self.fadePeak
        params1['inclIntercept'] = self.inclIntercept
        TSR1 = TimeSeriesRegression.TimeSeriesRegression(
            TSRParameters = params1, debugOutput=self.debuggingReporting)

        # Get output data (daily asset returns)
        assetReturns = self.loadReturnsArray(
                data, rootClass, daysBack=self.daysBack,
                adjustForRT=self.adjustForRT, clippedReturns=self.clippedReturns)
        modelDate = max(assetReturns.dates)
        assert(assetReturns.data.shape[1] >= self.daysBack)

        # Get input data
        startDate, endDate = (min(assetReturns.dates), max(assetReturns.dates))
        inputs = {}
        inputs['Term Spread'] = self.getTermSpread(startDate, endDate, 
            rootClass.numeraire_id)

        df = pandas.DataFrame.from_dict(inputs, orient='index')
        df = df[sorted(list(df.columns))]
        regInputs = Matrices.TimeSeriesMatrix.fromDataFrame(df)

        # Process returns - from appropriate reg1, reg2, matrices
        (assetReturns, regInputs) = self.processReturns(assetReturns, 
                                      regInputs, rootClass.rmg)
        #assetReturnsDF = assetReturns.toDataFrame()
        #regInputsDF = regInputs.toDataFrame()

        # run regression
        mm = TSR1.TSR(assetReturns, regInputs)

        # Store regression statistics
        if self.getRegStats:
            res = Utilities.Struct()
            res.subids = [s.getSubIdString() for s in data.universe]
            tstatDict = {}
            pvalDict = {}
            for f_n, factor in enumerate(mm.factors):
                tstatDict[factor] = list(mm.tstat[f_n, :])
                pvalDict[factor] = list(mm.pvals[f_n, :])
            res.tstatDict = tstatDict
            res.pvalDict = pvalDict
            res.rsq = [val for val in mm.rsquare]
            res.vifs = mm.vifs
            res.dof = mm.dof
            self.saveRegressStats(modelDate, rootClass.rmg.rmg_id, res)

        return self.setUpReturnObject(data.universe, 
            mm.params[mm.factors.index(self.factorSensitivity), :])

class TermSpread_Sensitivity_2(TermSpread_Sensitivity):
    """Base class for term spread sensitivities"""
    def __init__(self, connections, gp=None):
        TS_Sensitivity.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        
        # name of factor sensitivity to save
        self.factorSensitivity = 'Term Spread'
   
    def buildDescriptor(self, data, rootClass):
        self.log.debug('%s.buildDescriptor' % self.name)
        
        # Set up regression 1 parameters
        params1 = dict()
        params1['lag'] = self.lag
        params1['robust'] = self.robust
        params1['kappa'] = self.kappa
        params1['weighting'] = self.weighting
        params1['halflife'] = self.halflife
        params1['fadePeak'] = self.fadePeak
        params1['inclIntercept'] = self.inclIntercept
        TSR1 = TimeSeriesRegression.TimeSeriesRegression(
            TSRParameters = params1, debugOutput=self.debuggingReporting)

        # Set up regression 2 parameters
        params2 = dict()
        params2['lag'] = self.lag
        params2['robust'] = self.robust
        params2['kappa'] = self.kappa
        params2['weighting'] = self.weighting
        params2['halflife'] = self.halflife
        params2['fadePeak'] = self.fadePeak
        params2['inclIntercept'] = self.inclIntercept
        TSR2 = TimeSeriesRegression.TimeSeriesRegression(
            TSRParameters = params2, debugOutput=self.debuggingReporting)

        # Get output data (daily asset returns)
        assetReturns = self.loadReturnsArray(
                data, rootClass, daysBack=self.daysBack,
                adjustForRT=self.adjustForRT, clippedReturns=self.clippedReturns)
        modelDate = max(assetReturns.dates)
        assert(assetReturns.data.shape[1] >= self.daysBack)

        # Get input data
        startDate, endDate = (min(assetReturns.dates), max(assetReturns.dates))
        inputs = {}
        mktDts, mktVals = self.getMarketReturn(startDate, endDate, rootClass.rmg)
        inputs['Market'] = dict(zip(mktDts, mktVals))
        inputs['Term Spread'] = self.getTermSpread(startDate, endDate, 
            rootClass.numeraire_id)
        df = pandas.DataFrame.from_dict(inputs, orient='index')
        df = df[sorted(list(df.columns))]
        regInputs = Matrices.TimeSeriesMatrix.fromDataFrame(df)

        # Process returns - from appropriate reg1, reg2, matrices
        (assetReturns, regInputs) = self.processReturns(assetReturns, 
                                      regInputs, rootClass.rmg)
        
        # run first regression (market model)
        reg1Inputs = Matrices.TimeSeriesMatrix(['Market'],
                regInputs.dates)
        reg1Inputs.data = regInputs.data[regInputs.assets.index('Market'), :].reshape(1, len(regInputs.dates))
        mm = TSR1.TSR(assetReturns, reg1Inputs)

        # run second regression 
        df = pandas.DataFrame(mm.resid.T, index=mm.residAssets, columns=mm.residDates)
        reg2Outputs = Matrices.TimeSeriesMatrix.fromDataFrame(df)
        reg2Inputs = Matrices.TimeSeriesMatrix(['Term Spread'],
                regInputs.dates)
        reg2Inputs.data = regInputs.data[regInputs.assets.index('Term Spread'), :].reshape(1, len(regInputs.dates))
        mm = TSR2.TSR(reg2Outputs, reg2Inputs)

        if self.getRegStats:
            res = Utilities.Struct()
            res.subids = [s.getSubIdString() for s in data.universe]
            tstatDict = {}
            pvalDict = {}
            for f_n, factor in enumerate(mm.factors):
                tstatDict[factor] = list(mm.tstat[f_n, :])
                pvalDict[factor] = list(mm.pvals[f_n, :])
            res.tstatDict = tstatDict
            res.pvalDict = pvalDict
            res.rsq = [val for val in mm.rsquare]
            res.vifs = mm.vifs
            res.dof = mm.dof
            self.saveRegressStats(modelDate, rootClass.rmg.rmg_id, res)

        return self.setUpReturnObject(data.universe, 
            mm.params[mm.factors.index(self.factorSensitivity), :])

class TermSpread_Sensitivity_3(TermSpread_Sensitivity_2):
    """LT rate changes"""
    def __init__(self, connections, gp=None):
        TS_Sensitivity.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        
        # name of factor sensitivity to save
        self.factorSensitivity = 'Term Spread'
   
    def getTermSpread(self, startDate, endDate, ccy):
        metaData = {'treasury_yield_10y': 'M000010001'} 
        rawSeries = pandas.DataFrame.from_dict(
            self.marketDB.getMacroTs(list(metaData.values()),
            datetime.date(1980,1,1)))

        # adjust data as apparently the raw data is 10x
        # larger than it should be
        rawSeries = rawSeries/10 

        # convert to number (from percent)
        rawSeries = rawSeries/100
        termSpread = rawSeries[metaData['treasury_yield_10y']].sort_index(ascending=True)

        # compute first differences
        termSpread = termSpread.diff().dropna()
        idx = [d for d in termSpread.index if d.date() >= startDate
               and d.date() <= endDate]
        termSpread = termSpread.reindex(index=idx)
        return dict(zip([d.date() for d in termSpread.index], 
            termSpread.values))

class TermSpread_Sensitivity_4(TermSpread_Sensitivity_2):
    """LT rate"""
    def __init__(self, connections, gp=None):
        TS_Sensitivity.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        
        # name of factor sensitivity to save
        self.factorSensitivity = 'Term Spread'
  
    def getTermSpread(self, startDate, endDate, ccy):
        metaData = {'treasury_yield_10y': 'M000010001'} 
        rawSeries = pandas.DataFrame.from_dict(
            self.marketDB.getMacroTs(list(metaData.values()),
            datetime.date(1980,1,1)))

        # adjust data as apparently the raw data is 10x
        # larger than it should be
        rawSeries = rawSeries/10 

        # convert to number (from percent)
        rawSeries = rawSeries/100
        termSpread = rawSeries[metaData['treasury_yield_10y']].sort_index(ascending=True)

        idx = [d for d in termSpread.index if d.date() >= startDate
               and d.date() <= endDate]
        termSpread = termSpread.reindex(index=idx)
        return dict(zip([d.date() for d in termSpread.index], 
            termSpread.values))

class TermSpread_Sensitivity_5(TermSpread_Sensitivity_2):
    """ST rate changes"""
    def __init__(self, connections, gp=None):
        TS_Sensitivity.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        
        # name of factor sensitivity to save
        self.factorSensitivity = 'Term Spread'
  
    def getTermSpread(self, startDate, endDate, ccy):
        metaData = {'treasury_yield_13w': 'M000010002' } 
        rawSeries = pandas.DataFrame.from_dict(
            self.marketDB.getMacroTs(list(metaData.values()),
            datetime.date(1980,1,1)))

        # adjust data as apparently the raw data is 10x
        # larger than it should be
        rawSeries = rawSeries/10 

        # convert to number (from percent)
        rawSeries = rawSeries/100
        termSpread = rawSeries[metaData['treasury_yield_13w']].sort_index(ascending=True)

        # compute first differences
        termSpread = termSpread.diff().dropna()
        idx = [d for d in termSpread.index if d.date() >= startDate
               and d.date() <= endDate]
        termSpread = termSpread.reindex(index=idx)
        return dict(zip([d.date() for d in termSpread.index], 
            termSpread.values))

class TermSpread_Sensitivity_6(TermSpread_Sensitivity_2):
    """ST rate"""
    def __init__(self, connections, gp=None):
        TS_Sensitivity.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        
        # name of factor sensitivity to save
        self.factorSensitivity = 'Term Spread'
  
    def getTermSpread(self, startDate, endDate, ccy):
        metaData = {'treasury_yield_13w': 'M000010002' } 
        rawSeries = pandas.DataFrame.from_dict(
            self.marketDB.getMacroTs(list(metaData.values()),
            datetime.date(1980,1,1)))

        # adjust data as apparently the raw data is 10x
        # larger than it should be
        rawSeries = rawSeries/10 

        # convert to number (from percent)
        rawSeries = rawSeries/100
        termSpread = rawSeries[metaData['treasury_yield_13w']].sort_index(ascending=True)

        idx = [d for d in termSpread.index if d.date() >= startDate
               and d.date() <= endDate]
        termSpread = termSpread.reindex(index=idx)
        return dict(zip([d.date() for d in termSpread.index], 
            termSpread.values))

class CreditSpread_Sensitivity(TS_Sensitivity):
    """Base class for credit spread sensitivities"""
    def __init__(self, connections, gp=None):
        TS_Sensitivity.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__

        # name of factor sensitivity to save
        self.factorSensitivity = 'Credit Spread'

    def getCreditSpread(self, startDate, endDate, ccy):
        metaData = {'moodys_aaa': 'M000010256',
                    'moodys_baa': 'M000010257',
                    'ax_us_corp_spread_a': 'M000010260',
                    'ax_us_corp_spread_aaa': 'M000010258'} 

        rawSeries = pandas.DataFrame.from_dict(
            self.marketDB.getMacroTs(list(metaData.values()),
            datetime.date(1986,1,2)))
        rawSeries = rawSeries.sort_index(ascending=True)

        # moodys (available from 1986-01-02 through 2013-06-28 in database)
        cs0 = (rawSeries[metaData['moodys_baa']]/100 - \
               rawSeries[metaData['moodys_aaa']]/100).fillna(method='pad')

        # axioma (available from 2004-01-20 to present in database) 
        cs1 = (rawSeries[metaData['ax_us_corp_spread_a']] - \
               rawSeries[metaData['ax_us_corp_spread_aaa']]).fillna(method='pad')
        
        if endDate <= datetime.date(2009, 8, 1): 
            creditSpread = cs0
        elif startDate >= datetime.date(2012, 1, 1): 
            creditSpread = cs1
        else:
            creditSpread = cs1.copy()
            creditSpread.loc[:datetime.datetime(2009, 8, 1)] = \
                cs0.loc[:datetime.datetime(2009, 8, 1)]
            dates1 = list(cs0.loc[datetime.datetime(2009, 8, 1):datetime.datetime(2012, 1, 1)].index)
            alpha = pandas.Series(numpy.linspace(0., 1., len(dates1)), dates1)
            creditSpread.loc[dates1] = alpha*cs1.loc[dates1] + (1.-alpha)*cs0.loc[dates1]
        
        # compute first differences 
        creditSpread = creditSpread.diff().dropna()

        idx = [d for d in creditSpread.index if d.date() >= startDate
               and d.date() <= endDate]
        creditSpread = creditSpread.reindex(index=idx)
        return dict(zip([d.date() for d in creditSpread.index], 
            creditSpread.values))

    def buildDescriptor(self, data, rootClass):
        self.log.debug('%s.buildDescriptor' % self.name)
        
        # Set up regression 1 parameters
        params1 = dict()
        params1['lag'] = self.lag
        params1['robust'] = self.robust
        params1['kappa'] = self.kappa
        params1['weighting'] = self.weighting
        params1['halflife'] = self.halflife
        params1['fadePeak'] = self.fadePeak
        params1['inclIntercept'] = self.inclIntercept
        TSR1 = TimeSeriesRegression.TimeSeriesRegression(
            TSRParameters = params1, debugOutput=self.debuggingReporting)

        # Get output data (daily asset returns)
        assetReturns = self.loadReturnsArray(
                data, rootClass, daysBack=self.daysBack,
                adjustForRT=self.adjustForRT, clippedReturns=self.clippedReturns)
        modelDate = max(assetReturns.dates)

        # Get input data
        startDate, endDate = (min(assetReturns.dates), max(assetReturns.dates))
        inputs = {}
        inputs['Credit Spread'] = self.getCreditSpread(startDate, endDate, 
            rootClass.numeraire_id)

        df = pandas.DataFrame.from_dict(inputs, orient='index')
        df = df[sorted(list(df.columns))]
        regInputs = Matrices.TimeSeriesMatrix.fromDataFrame(df)

        # Process returns - from appropriate reg1, reg2, matrices
        (assetReturns, regInputs) = self.processReturns(assetReturns, 
                                      regInputs, rootClass.rmg)
        #assetReturnsDF = assetReturns.toDataFrame()
        #regInputsDF = regInputs.toDataFrame()

        # run regression
        mm = TSR1.TSR(assetReturns, regInputs)

        # Store regression statistics
        if self.getRegStats:
            res = Utilities.Struct()
            res.subids = [s.getSubIdString() for s in data.universe]
            tstatDict = {}
            pvalDict = {}
            for f_n, factor in enumerate(mm.factors):
                tstatDict[factor] = list(mm.tstat[f_n, :])
                pvalDict[factor] = list(mm.pvals[f_n, :])
            res.tstatDict = tstatDict
            res.pvalDict = pvalDict
            res.rsq = [val for val in mm.rsquare]
            res.vifs = mm.vifs
            res.dof = mm.dof
            self.saveRegressStats(modelDate, rootClass.rmg.rmg_id, res)

        return self.setUpReturnObject(data.universe, 
            mm.params[mm.factors.index(self.factorSensitivity), :])

class CreditSpread_Sensitivity_2(CreditSpread_Sensitivity):
    """Base class for credit spread sensitivities"""
    def __init__(self, connections, gp=None):
        TS_Sensitivity.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__

        # name of factor sensitivity to save
        self.factorSensitivity = 'Credit Spread'

    def buildDescriptor(self, data, rootClass):
        self.log.debug('%s.buildDescriptor' % self.name)
        
        # Set up regression 1 parameters
        params1 = dict()
        params1['lag'] = self.lag
        params1['robust'] = self.robust
        params1['kappa'] = self.kappa
        params1['weighting'] = self.weighting
        params1['halflife'] = self.halflife
        params1['fadePeak'] = self.fadePeak
        params1['inclIntercept'] = self.inclIntercept
        TSR1 = TimeSeriesRegression.TimeSeriesRegression(
            TSRParameters = params1, debugOutput=self.debuggingReporting)

        # Set up regression 2 parameters
        params2 = dict()
        params2['lag'] = self.lag
        params2['robust'] = self.robust
        params2['kappa'] = self.kappa
        params2['weighting'] = self.weighting
        params2['halflife'] = self.halflife
        params2['fadePeak'] = self.fadePeak
        params2['inclIntercept'] = self.inclIntercept
        TSR2 = TimeSeriesRegression.TimeSeriesRegression(
            TSRParameters = params2, debugOutput=self.debuggingReporting)

        # Get output data (daily asset returns)
        assetReturns = self.loadReturnsArray(
                data, rootClass, daysBack=self.daysBack,
                adjustForRT=self.adjustForRT, clippedReturns=self.clippedReturns)
        modelDate = max(assetReturns.dates)
        assert(assetReturns.data.shape[1] >= self.daysBack)
        
        # Get input data
        startDate, endDate = (min(assetReturns.dates), max(assetReturns.dates))
        inputs = {}
        mktDts, mktVals = self.getMarketReturn(startDate, endDate, rootClass.rmg)
        inputs['Market'] = dict(zip(mktDts, mktVals))
        inputs['Credit Spread'] = self.getCreditSpread(startDate, endDate, 
            rootClass.numeraire_id)
        df = pandas.DataFrame.from_dict(inputs, orient='index')
        df = df[sorted(list(df.columns))]
        regInputs = Matrices.TimeSeriesMatrix.fromDataFrame(df)

        # Process returns - from appropriate reg1, reg2, matrices
        (assetReturns, regInputs) = self.processReturns(assetReturns, 
                                      regInputs, rootClass.rmg)
        
        # run first regression (market model)
        reg1Inputs = Matrices.TimeSeriesMatrix(['Market'],
                regInputs.dates)
        reg1Inputs.data = regInputs.data[regInputs.assets.index('Market'), :].reshape(1, len(regInputs.dates))
        mm = TSR1.TSR(assetReturns, reg1Inputs)

        # run second regression 
        df = pandas.DataFrame(mm.resid.T, index=mm.residAssets, columns=mm.residDates)
        reg2Outputs = Matrices.TimeSeriesMatrix.fromDataFrame(df)
        reg2Inputs = Matrices.TimeSeriesMatrix(['Credit Spread'],
                regInputs.dates)
        reg2Inputs.data = regInputs.data[regInputs.assets.index('Credit Spread'), :].reshape(1, len(regInputs.dates))
        mm = TSR2.TSR(reg2Outputs, reg2Inputs)

        if self.getRegStats:
            res = Utilities.Struct()
            res.subids = [s.getSubIdString() for s in data.universe]
            tstatDict = {}
            pvalDict = {}
            for f_n, factor in enumerate(mm.factors):
                tstatDict[factor] = list(mm.tstat[f_n, :])
                pvalDict[factor] = list(mm.pvals[f_n, :])
            res.tstatDict = tstatDict
            res.pvalDict = pvalDict
            res.rsq = [val for val in mm.rsquare]
            res.vifs = mm.vifs
            res.dof = mm.dof
            self.saveRegressStats(modelDate, rootClass.rmg.rmg_id, res)

        return self.setUpReturnObject(data.universe, 
            mm.params[mm.factors.index(self.factorSensitivity), :])

class Oil_Sensitivity(TS_Sensitivity):
    """Base class for oil sensitivities"""
    def __init__(self, connections, gp=None):
        TS_Sensitivity.__init__(self, connections, gp=gp)

    def getOilReturn(self, startDate, endDate):
        """Returns oil price returns in USD using
        Crude Oil-West Texas Intermediate Spot Cushing 
        measured in USD Per Barrel"""
        # Load oil returns
        # TO DO ESTHER - if Date <= Jan 1 1990, return nothing!! no oil prices exist
        oilPrices = pandas.Series(self.marketDB.getMacroTs(['M000100022'],
                                  datetime.date(1988,1,1))['M000100022'])
        oilPrices = oilPrices.sort_index(ascending=True)
        oilRets = (oilPrices/(oilPrices.shift(1)) - 1.)
        idx = [d.date() for d in oilRets.index if d.date() >= startDate
               and d.date() <= endDate]
        oilRets = oilRets.reindex(index=idx)
        return oilRets 

class Oil_Sensitivity_4(Oil_Sensitivity):
    """Regress the residual from a market model 
       on oil returns
    """
    def __init__(self, connections, gp=None):
        Oil_Sensitivity.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
       
        # Regression settings
        self.inclIntercept = True
        self.weighting = 'pyramid' # exponential, triangle, pyramid, or None (equal) 

        # name of factor sensitivity to save
        self.factorSensitivity = 'Oil'

    def buildDescriptor(self, data, rootClass):
        self.log.debug('%s.buildDescriptor' % self.name)
        
        # Set up regression 1 parameters
        params1 = dict()
        params1['lag'] = self.lag
        params1['robust'] = self.robust
        params1['kappa'] = self.kappa
        params1['weighting'] = self.weighting
        params1['halflife'] = self.halflife
        params1['fadePeak'] = self.fadePeak
        params1['inclIntercept'] = self.inclIntercept
        TSR1 = TimeSeriesRegression.TimeSeriesRegression(
            TSRParameters = params1, debugOutput=self.debuggingReporting)

        # Set up regression 2 parameters
        params2 = dict()
        params2['lag'] = self.lag
        params2['robust'] = self.robust
        params2['kappa'] = self.kappa
        params2['weighting'] = self.weighting
        params2['halflife'] = self.halflife
        params2['fadePeak'] = self.fadePeak
        params2['inclIntercept'] = self.inclIntercept
        TSR2 = TimeSeriesRegression.TimeSeriesRegression(
            TSRParameters = params2, debugOutput=self.debuggingReporting)

        # Get output data (daily asset returns)
        assetReturns = self.loadReturnsArray(
                data, rootClass, daysBack=self.daysBack,
                adjustForRT=self.adjustForRT, clippedReturns=self.clippedReturns)
        modelDate = max(assetReturns.dates)
        
        # Get input data
        startDate, endDate = (min(assetReturns.dates), max(assetReturns.dates))
        inputs = {}
        inputs['Market'] = self.getMarketReturn(startDate, endDate, rootClass.rmg)
        inputs['Oil'] = self.getOilReturn(startDate, endDate)
        ccyOil = 1 # currently pulling oil prices quoted in USD

        """
        # Get local currency and exchange rate returns
        dts = rootClass.localCurrency_id.keys()
        dts.sort()
        ccyDt = dts[bisect.bisect_right(dts, modelDate)-1] \
                if len(dts) > 1 else dts[0]
        localCcyId = rootClass.localCurrency_id[ccyDt]
        if localCcyId != ccyOil:
            # Get fx returns, where fx rates are specified
            # as USD/LCL.
            inputs['FX'] = self.getCurrencyReturnsHistory(startDate, 
                    endDate, ccyBase=localCcyId, ccyQuote=ccyOil, 
                    rmg=rootClass.rmg)
        """

        df = pandas.DataFrame.from_dict(inputs, orient='index')
        df = df.T.sort_index().T
        regInputs = Matrices.TimeSeriesMatrix.fromDataFrame(df)

        # Process returns 
        (assetReturns, regInputs) = self.processReturns(assetReturns, 
                                      regInputs, rootClass.rmg)
        
        # run first regression (market model)
        reg1Inputs = Matrices.TimeSeriesMatrix(['Market'],
                regInputs.dates)
        reg1Inputs.data = regInputs.data[regInputs.assets.index('Market'), :].reshape(1, len(regInputs.dates))
        mm = TSR1.TSR(assetReturns, reg1Inputs)

        # run second regression
        df = pandas.DataFrame(mm.resid.T, index=mm.residAssets, columns=mm.residDates)
        reg2Outputs = Matrices.TimeSeriesMatrix.fromDataFrame(df)
        reg2Inputs = Matrices.TimeSeriesMatrix(['Oil'],
                regInputs.dates)
        reg2Inputs.data = regInputs.data[regInputs.assets.index('Oil'), :].reshape(1, len(regInputs.dates))
        mm = TSR2.TSR(reg2Outputs, reg2Inputs)
     
        if self.getRegStats:
            res = Utilities.Struct()
            res.subids = [s.getSubIdString() for s in data.universe]
            tstatDict = {}
            pvalDict = {}
            for f_n, factor in enumerate(mm.factors):
                tstatDict[factor] = list(mm.tstat[f_n, :])
                pvalDict[factor] = list(mm.pvals[f_n, :])
            res.tstatDict = tstatDict
            res.pvalDict = pvalDict
            res.rsq = [val for val in mm.rsquare]
            res.vifs = mm.vifs
            res.dof = mm.dof
            self.saveRegressStats(modelDate, rootClass.rmg.rmg_id, res)

        return self.setUpReturnObject(data.universe, 
            mm.params[mm.factors.index(self.factorSensitivity), :])

class Oil_Sensitivity_5(Oil_Sensitivity):
    """Regress the residual from a market model 
       on oil returns
    """
    def __init__(self, connections, gp=None):
        Oil_Sensitivity.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
      
        # Regression settings
        self.inclIntercept = True
        self.weighting = 'pyramid' # exponential, triangle, pyramid, or None (equal) 

        # name of factor sensitivity to save
        self.factorSensitivity = 'Oil'

    def buildDescriptor(self, data, rootClass):
        self.log.debug('Oil_Sensitivity_5.buildDescriptor')
        
        # Set up regression 1 parameters
        params1 = dict()
        params1['lag'] = self.lag
        params1['robust'] = self.robust
        params1['kappa'] = self.kappa
        params1['weighting'] = self.weighting
        params1['halflife'] = self.halflife
        params1['fadePeak'] = self.fadePeak
        params1['inclIntercept'] = self.inclIntercept
        TSR1 = TimeSeriesRegression.TimeSeriesRegression(
            TSRParameters = params1, debugOutput=self.debuggingReporting)

        # Set up regression 2 parameters
        params2 = dict()
        params2['lag'] = self.lag
        params2['robust'] = self.robust
        params2['kappa'] = self.kappa
        params2['weighting'] = self.weighting
        params2['halflife'] = self.halflife
        params2['fadePeak'] = self.fadePeak
        params2['inclIntercept'] = self.inclIntercept
        TSR2 = TimeSeriesRegression.TimeSeriesRegression(
            TSRParameters = params2, debugOutput=self.debuggingReporting)

        # Get output data (daily asset returns)
        assetReturns = self.loadReturnsArray(
                data, rootClass, daysBack=self.daysBack,
                adjustForRT=self.adjustForRT, clippedReturns=self.clippedReturns)
        modelDate = max(assetReturns.dates)

        # Get input data
        startDate, endDate = (min(assetReturns.dates), max(assetReturns.dates))
        indRets = self.getIndustryReturns(startDate, endDate, rootClass.rmg).T
        if indRets.empty:
            return self.setUpReturnObject(data.universe, [None for sid in data.universe])
        
        inputs = {}
        for col in indRets.columns:
            inputs[col] = indRets[col].to_dict()
        inputs['Oil'] = self.getOilReturn(startDate, endDate, rootClass.numeraire_id)
        df = pandas.DataFrame.from_dict(inputs, orient='index')
        df = df[sorted(list(df.columns))]
        regInputs = Matrices.TimeSeriesMatrix.fromDataFrame(df)

        # Process returns - from appropriate reg1, reg2, matrices
        (assetReturns, regInputs) = self.processReturns(assetReturns, 
                                      regInputs, rootClass.rmg)
        assetReturnsDF = assetReturns.toDataFrame()

        # run regressions - one industry at a time
        import proddatactrl
        dataprov = proddatactrl.ProdDataController(sid='research')

        if rootClass.rmg.rmg_id == 1:
            rmname = 'USAxioma2013MH'
            currency = 'USD'
        elif rootClass.rmg.rmg_id == 10:
            rmname = 'CAAxioma2009MH'
            currency = 'CAD'
        else:
            rmname = 'USAxaiom2013MH'
            currency = 'USD'
        scheme = dataprov.getModelClassificationHierarchy(rmname)
        indList = []
        for key in scheme.hierarchy.keys():
           indList.extend(scheme.getIndustries(key))
        indList = sorted(list(set(list(indList))))
        asset2leaf = dataprov.getModelClassificationAssets(max(assetReturns.dates), 
                rmname, assets=[s.modelID.getIDString()[1:] for s in data.universe])
        subIssues = []
        rsquare = []
        indTstatDict = {'Intercept': [], 'Industry': []}
        tstatDict = {'Intercept': [], 'Oil': []}
        pvalDict = {'Intercept': [], 'Oil': []}
        params = []
        for industry in indList:
            print(industry)
            if industry not in regInputs.assets:
                continue
            assets = [asset for asset, ind in asset2leaf.items() if ind == industry]
            indAssets = [SubIssue('D%s11' % asset) for asset in assets]
            if len(assets) == 0:
                continue
            # run first regression 
            reg1Inputs = Matrices.TimeSeriesMatrix(['Industry'],
                    regInputs.dates)
            reg1Inputs.data = regInputs.data[regInputs.assets.index(industry), :].reshape(1, len(regInputs.dates))
            reg1Outputs = Matrices.TimeSeriesMatrix(indAssets, assetReturns.dates)
            reg1Outputs.data = assetReturnsDF.reindex(index=indAssets).values
            mm = TSR1.TSR(reg1Outputs, reg1Inputs)
            for f_n, factor in enumerate(mm.factors):
                indTstatDict[factor].extend(list(mm.tstat[f_n, :]))

            # run second regression
            df = pandas.DataFrame(mm.resid.T, index=mm.residAssets, columns=mm.residDates)
            reg2Outputs = Matrices.TimeSeriesMatrix.fromDataFrame(df)
            reg2Inputs = Matrices.TimeSeriesMatrix(['Oil'],
                    regInputs.dates)
            reg2Inputs.data = regInputs.data[regInputs.assets.index('Oil'), :].reshape(1, len(regInputs.dates))
            mm = TSR2.TSR(reg2Outputs, reg2Inputs)

            params.extend(list(mm.params[mm.factors.index(self.factorSensitivity), :]))
            subIssues.extend(indAssets)
            rsquare.extend(mm.rsquare)
            for f_n, factor in enumerate(mm.factors):
                tstatDict[factor].extend(list(mm.tstat[f_n, :]))
                pvalDict[factor].extend(list(mm.pvals[f_n, :]))

        # For assets that are not categorized, append None
        nonCatAssets = list(set(data.universe) - set(subIssues))
        subIssues.extend(nonCatAssets)
        params.extend([None]*len(nonCatAssets))
        rsquare.extend([None]*len(nonCatAssets))
        for f_n, factor in enumerate(mm.factors):
            tstatDict[factor].extend([None]*len(nonCatAssets))
            pvalDict[factor].extend([None]*len(nonCatAssets))
        for key in indTstatDict.keys():
            indTstatDict[key].extend([None]*len(nonCatAssets))

        # Store regression statistics
        if self.getRegStats:
            res = Utilities.Struct()
            res.subids = [s.getSubIdString() for s in subIssues]
            res.tstatDict = tstatDict
            res.pvalDict = pvalDict
            res.rsq = rsquare
            res.vifs = mm.vifs
            res.dof = mm.dof
            self.saveRegressStats(modelDate, rootClass.rmg.rmg_id, res)

        paramsDict = dict(zip(subIssues, params))
        return self.setUpReturnObject(data.universe, [paramsDict[sid] for sid in data.universe])

class Oil_Sensitivity_6(Oil_Sensitivity):
    """Regress the residual from a market model 
       on oil returns
    """
    def __init__(self, connections, gp=None):
        Oil_Sensitivity.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
      
        # Regression settings
        self.inclIntercept = True
        self.weighting = 'pyramid' # exponential, triangle, pyramid, or None (equal) 

        # name of factor sensitivity to save
        self.factorSensitivity = 'Oil'

    def buildDescriptor(self, data, rootClass):
        self.log.debug('Oil_Sensitivity_6.buildDescriptor')
        
        # Set up regression 1 parameters
        params1 = dict()
        params1['lag'] = self.lag
        params1['robust'] = self.robust
        params1['kappa'] = self.kappa
        params1['weighting'] = self.weighting
        params1['halflife'] = self.halflife
        params1['fadePeak'] = self.fadePeak
        params1['inclIntercept'] = self.inclIntercept
        TSR1 = TimeSeriesRegression.TimeSeriesRegression(
            TSRParameters = params1, debugOutput=self.debuggingReporting)

        # Get output data (daily asset returns)
        assetReturns = self.loadReturnsArray(
                data, rootClass, daysBack=self.daysBack,
                adjustForRT=self.adjustForRT, clippedReturns=self.clippedReturns)
        modelDate = max(assetReturns.dates)

        # Get input data
        startDate, endDate = (min(assetReturns.dates), max(assetReturns.dates))
        indRets = self.getIndustryReturns(startDate, endDate, rootClass.rmg).T
        if indRets.empty:
            return self.setUpReturnObject(data.universe, [None for sid in data.universe])
        
        inputs = {}
        for col in indRets.columns:
            inputs[col] = indRets[col].to_dict()
        inputs['Oil'] = self.getOilReturn(startDate, endDate, rootClass.numeraire_id)
        df = pandas.DataFrame.from_dict(inputs, orient='index')
        df = df[sorted(list(df.columns))]
        regInputs = Matrices.TimeSeriesMatrix.fromDataFrame(df)

        # Process returns - from appropriate reg1, reg2, matrices
        (assetReturns, regInputs) = self.processReturns(assetReturns, 
                                      regInputs, rootClass.rmg)
        assetReturnsDF = assetReturns.toDataFrame()
        regInputsDF = regInputs.toDataFrame()

        # run regressions - one industry at a time
        import proddatactrl
        dataprov = proddatactrl.ProdDataController(sid='research')
        if rootClass.rmg.rmg_id == 1:
            rmname = 'USAxioma2013MH'
            currency = 'USD'
        elif rootClass.rmg.rmg_id == 10:
            rmname = 'CAAxioma2009MH'
            currency = 'CAD'
        else:
            rmname = 'USAxaiom2013MH'
            currency = 'USD'
        scheme = dataprov.getModelClassificationHierarchy(rmname)
        indList = []
        for key in scheme.hierarchy.keys():
           indList.extend(scheme.getIndustries(key))
        indList = sorted(list(set(list(indList))))
        asset2leaf = dataprov.getModelClassificationAssets(max(assetReturns.dates), 
                rmname, assets=[s.modelID.getIDString()[1:] for s in data.universe])

        subIssues = []
        rsquare = []
        indTstatDict = {'Intercept': [], 'Industry': []}
        tstatDict = {'Intercept': [], 'Oil': []}
        pvalDict = {'Intercept': [], 'Oil': []}
        params = []

        for industry in indList:
            print(industry)
            if industry not in regInputs.assets:
                continue
            assets = [asset for asset, ind in asset2leaf.items() if ind == industry]
            indAssets = [SubIssue('D%s11' % asset) for asset in assets]
            if len(assets) == 0:
                continue
            
            subsetAssetRetsDF = assetReturnsDF.reindex(index=indAssets)
            netAssetReturnsDF = subsetAssetRetsDF.subtract(regInputsDF.T[industry], axis='columns')

            # run regression
            reg1Outputs = Matrices.TimeSeriesMatrix.fromDataFrame(netAssetReturnsDF)
            reg1Inputs = Matrices.TimeSeriesMatrix(['Oil'],
                    regInputs.dates)
            reg1Inputs.data = regInputs.data[regInputs.assets.index('Oil'), :].reshape(1, len(regInputs.dates))
            mm = TSR1.TSR(reg1Outputs, reg1Inputs)

            params.extend(list(mm.params[mm.factors.index(self.factorSensitivity), :]))
            subIssues.extend(indAssets)
            rsquare.extend(mm.rsquare)
            for f_n, factor in enumerate(mm.factors):
                tstatDict[factor].extend(list(mm.tstat[f_n, :]))
                pvalDict[factor].extend(list(mm.pvals[f_n, :]))

        # For assets that are not categorized, append None
        nonCatAssets = list(set(data.universe) - set(subIssues))
        subIssues.extend(nonCatAssets)
        params.extend([None]*len(nonCatAssets))
        rsquare.extend([None]*len(nonCatAssets))
        for f_n, factor in enumerate(mm.factors):
            tstatDict[factor].extend([None]*len(nonCatAssets))
            pvalDict[factor].extend([None]*len(nonCatAssets))

        # Store regression statistics
        if self.getRegStats:
            res = Utilities.Struct()
            res.subids = [sid.getSubIdString() for sid in subIssues]
            res.tstatDict = tstatDict
            res.pvalDict = pvalDict
            res.rsq = rsquare
            res.vifs = mm.vifs
            res.dof = mm.dof
            self.saveRegressStats(modelDate, rootClass.rmg.rmg_id, res)

        paramsDict = dict(zip(subIssues, params))
        return self.setUpReturnObject(data.universe, [paramsDict[sid] for sid in data.universe])

class CA_Industry_Sensitivity(TS_Sensitivity):
    def __init__(self, connections, gp=None):
        TS_Sensitivity.__init__(self, connections, gp=gp)

        self.name = self.__class__.__name__
      
        # Regression settings
        self.inclIntercept = True
        self.weighting = 'pyramid' # exponential, triangle, pyramid, or None (equal) 

    def getDescriptorIndustry(self, descClassName):
        qry = "select description from descriptor where name = '%s'" % descClassName
        self.modelDB.dbCursor.execute(qry)
        res = [val[0] for val in self.modelDB.dbCursor.fetchall()]
        if len(res) == 1:
            return res[0].split(':')[-1]
        return None

    def buildDescriptor(self, data, rootClass):
        self.log.debug('CA_Industry_Sensitivity_1.buildDescriptor')
        
        # Set up regression 1 parameters
        params1 = dict()
        params1['lag'] = self.lag
        params1['robust'] = self.robust
        params1['kappa'] = self.kappa
        params1['weighting'] = self.weighting
        params1['halflife'] = self.halflife
        params1['fadePeak'] = self.fadePeak
        params1['inclIntercept'] = self.inclIntercept
        TSR1 = TimeSeriesRegression.TimeSeriesRegression(
            TSRParameters = params1, debugOutput=self.debuggingReporting)

        # Get output data (daily asset returns)
        assetReturns = self.loadReturnsArray(
                data, rootClass, daysBack=self.daysBack,
                adjustForRT=self.adjustForRT, clippedReturns=self.clippedReturns)
        modelDate = max(assetReturns.dates)
        assert(assetReturns.data.shape[1] >= self.daysBack)

        # Get input data
        startDate, endDate = (min(assetReturns.dates), max(assetReturns.dates))
        indRets = self.getIndustryReturns(startDate, endDate, rootClass.rmg).T
        if indRets.empty:
            return self.setUpReturnObject(data.universe, [None for sid in data.universe])
        
        inputs = {}
        for col in indRets.columns:
            inputs[col] = indRets[col].to_dict()
        df = pandas.DataFrame.from_dict(inputs, orient='index')
        df = df[sorted(list(df.columns))]
        regInputs = Matrices.TimeSeriesMatrix.fromDataFrame(df)

        # Process returns - from appropriate reg1, reg2, matrices
        (assetReturns, regInputs) = self.processReturns(assetReturns, 
                                      regInputs, rootClass.rmg)
        assetReturnsDF = assetReturns.toDataFrame()

        # run regressions - one industry at a time
        import proddatactrl
        dataprov = proddatactrl.ProdDataController(sid='research')

        if rootClass.rmg.rmg_id == 1:
            rmname = 'USAxioma2013MH'
            currency = 'USD'
        elif rootClass.rmg.rmg_id == 10:
            rmname = 'CAAxioma2009MH'
            currency = 'CAD'
        else:
            rmname = 'USAxaiom2013MH'
            currency = 'USD'
        scheme = dataprov.getModelClassificationHierarchy(rmname)
        indList = []
        for key in scheme.hierarchy.keys():
           indList.extend(scheme.getIndustries(key))
        indList = sorted(list(set(list(indList))))
        asset2leaf = dataprov.getModelClassificationAssets(max(assetReturns.dates), 
                rmname, assets=[s.modelID.getIDString()[1:] for s in data.universe])

        # Run regression for assets in self.factorSensitivity
        subIssues = []
        rsquare = []

        tstatDict = {'Intercept': [], self.factorSensitivity: []}
        pvalDict = {'Intercept': [], self.factorSensitivity: []}
        params = []
        if self.factorSensitivity in indList and self.factorSensitivity in regInputs.assets:
            assets = [asset for asset, ind in asset2leaf.items() if ind == self.factorSensitivity]
            indAssets = [SubIssue('D%s11' % asset) for asset in assets]
            if len(assets) > 0:
                reg1Outputs = Matrices.TimeSeriesMatrix(indAssets, assetReturns.dates)
                reg1Outputs.data = assetReturnsDF.reindex(index=indAssets).values
                reg1Inputs = Matrices.TimeSeriesMatrix([self.factorSensitivity],
                        regInputs.dates)
                reg1Inputs.data = \
                        regInputs.data[regInputs.assets.index(self.factorSensitivity), :].reshape(1, len(regInputs.dates))
                mm = TSR1.TSR(reg1Outputs, reg1Inputs)

                params.extend(list(mm.params[mm.factors.index(self.factorSensitivity), :]))
                subIssues.extend(indAssets)
                rsquare.extend(mm.rsquare)
                for f_n, factor in enumerate(mm.factors):
                    tstatDict[factor].extend(list(mm.tstat[f_n, :]))
                    pvalDict[factor].extend(list(mm.pvals[f_n, :]))
            
        # For assets that are not categorized, append None
        nonCatAssets = list(set(data.universe) - set(subIssues))
        subIssues.extend(nonCatAssets)
        params.extend([None]*len(nonCatAssets))
        rsquare.extend([None]*len(nonCatAssets))
        for f_n, factor in enumerate(mm.factors):
            tstatDict[factor].extend([None]*len(nonCatAssets))
            pvalDict[factor].extend([None]*len(nonCatAssets))

        # Store regression statistics
        if self.getRegStats:
            res = Utilities.Struct()
            res.subids = [s.getSubIdString() for s in subIssues]
            res.tstatDict = tstatDict
            res.pvalDict = pvalDict
            res.rsq = rsquare
            res.vifs = mm.vifs
            res.dof = mm.dof
            self.saveRegressStats(modelDate, rootClass.rmg.rmg_id, res)

        paramsDict = dict(zip(subIssues, params))
        return self.setUpReturnObject(data.universe, [paramsDict[sid] for sid in data.universe])

class CA_Industry_Sensitivity_1(CA_Industry_Sensitivity):
    """Regress total returns on industry returns
    """
    def __init__(self, connections, gp=None):
        TS_Sensitivity.__init__(self, connections, gp=gp)

        self.name = self.__class__.__name__
      
        # Regression settings
        self.inclIntercept = True
        self.weighting = 'pyramid' # exponential, triangle, pyramid, or None (equal) 

        # name of factor sensitivity to save
        self.factorSensitivity = self.getDescriptorIndustry(self.name)

class CA_Industry_Sensitivity_2(CA_Industry_Sensitivity):
    """Regress total returns on industry returns
    """
    def __init__(self, connections, gp=None):
        CA_Industry_Sensitivity_1.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.factorSensitivity = self.getDescriptorIndustry(self.name)

class CA_Industry_Sensitivity_3(CA_Industry_Sensitivity):
    """Regress total returns on industry returns
    """
    def __init__(self, connections, gp=None):
        CA_Industry_Sensitivity_1.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.factorSensitivity = self.getDescriptorIndustry(self.name)
    
class CA_Industry_Sensitivity_4(CA_Industry_Sensitivity):
    """Regress total returns on industry returns
    """
    def __init__(self, connections, gp=None):
        CA_Industry_Sensitivity_1.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.factorSensitivity = self.getDescriptorIndustry(self.name)

class CA_Industry_Sensitivity_5(CA_Industry_Sensitivity):
    """Regress total returns on industry returns
    """
    def __init__(self, connections, gp=None):
        CA_Industry_Sensitivity_1.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.factorSensitivity = self.getDescriptorIndustry(self.name)

class CA_Industry_Sensitivity_6(CA_Industry_Sensitivity):
    """Regress total returns on industry returns
    """
    def __init__(self, connections, gp=None):
        CA_Industry_Sensitivity_1.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.factorSensitivity = self.getDescriptorIndustry(self.name)

class CA_Industry_Sensitivity_7(CA_Industry_Sensitivity):
    """Regress total returns on industry returns
    """
    def __init__(self, connections, gp=None):
        CA_Industry_Sensitivity_1.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.factorSensitivity = self.getDescriptorIndustry(self.name)

class CA_Industry_Sensitivity_8(CA_Industry_Sensitivity):
    """Regress total returns on industry returns
    """
    def __init__(self, connections, gp=None):
        CA_Industry_Sensitivity_1.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.factorSensitivity = self.getDescriptorIndustry(self.name)

class CA_Industry_Sensitivity_9(CA_Industry_Sensitivity):
    """Regress total returns on industry returns
    """
    def __init__(self, connections, gp=None):
        CA_Industry_Sensitivity_1.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.factorSensitivity = self.getDescriptorIndustry(self.name)

class CA_Industry_Sensitivity_10(CA_Industry_Sensitivity):
    """Regress total returns on industry returns
    """
    def __init__(self, connections, gp=None):
        CA_Industry_Sensitivity_1.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.factorSensitivity = self.getDescriptorIndustry(self.name)

class CA_Industry_Sensitivity_11(CA_Industry_Sensitivity):
    """Regress total returns on industry returns
    """
    def __init__(self, connections, gp=None):
        CA_Industry_Sensitivity_1.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.factorSensitivity = self.getDescriptorIndustry(self.name)

class CA_Industry_Sensitivity_12(CA_Industry_Sensitivity):
    """Regress total returns on industry returns
    """
    def __init__(self, connections, gp=None):
        CA_Industry_Sensitivity_1.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.factorSensitivity = self.getDescriptorIndustry(self.name)

class CA_Industry_Sensitivity_13(CA_Industry_Sensitivity):
    """Regress total returns on industry returns
    """
    def __init__(self, connections, gp=None):
        CA_Industry_Sensitivity_1.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.factorSensitivity = self.getDescriptorIndustry(self.name)

class CA_Industry_Sensitivity_14(CA_Industry_Sensitivity):
    """Regress total returns on industry returns
    """
    def __init__(self, connections, gp=None):
        CA_Industry_Sensitivity_1.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.factorSensitivity = self.getDescriptorIndustry(self.name)

class CA_Industry_Sensitivity_15(CA_Industry_Sensitivity):
    """Regress total returns on industry returns
    """
    def __init__(self, connections, gp=None):
        CA_Industry_Sensitivity_1.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.factorSensitivity = self.getDescriptorIndustry(self.name)

class CA_Industry_Sensitivity_16(CA_Industry_Sensitivity):
    """Regress total returns on industry returns
    """
    def __init__(self, connections, gp=None):
        CA_Industry_Sensitivity_1.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.factorSensitivity = self.getDescriptorIndustry(self.name)

class CA_Industry_Sensitivity_17(CA_Industry_Sensitivity):
    """Regress total returns on industry returns
    """
    def __init__(self, connections, gp=None):
        CA_Industry_Sensitivity_1.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.factorSensitivity = self.getDescriptorIndustry(self.name)

class CA_Industry_Sensitivity_18(CA_Industry_Sensitivity):
    """Regress total returns on industry returns
    """
    def __init__(self, connections, gp=None):
        CA_Industry_Sensitivity_1.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.factorSensitivity = self.getDescriptorIndustry(self.name)

class CA_Industry_Sensitivity_19(CA_Industry_Sensitivity):
    """Regress total returns on industry returns
    """
    def __init__(self, connections, gp=None):
        CA_Industry_Sensitivity_1.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.factorSensitivity = self.getDescriptorIndustry(self.name)

class CA_Industry_Sensitivity_20(CA_Industry_Sensitivity):
    """Regress total returns on industry returns
    """
    def __init__(self, connections, gp=None):
        CA_Industry_Sensitivity_1.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.factorSensitivity = self.getDescriptorIndustry(self.name)

class CA_Industry_Sensitivity_21(CA_Industry_Sensitivity):
    """Regress total returns on industry returns
    """
    def __init__(self, connections, gp=None):
        CA_Industry_Sensitivity_1.__init__(self, connections, gp=gp)
        self.name = self.__class__.__name__
        self.factorSensitivity = self.getDescriptorIndustry(self.name)


