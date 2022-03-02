
import logging
import numbers
import itertools
from collections import defaultdict

import pandas
import statsmodels.api as sm

import datetime
import numpy as np
import numpy.ma as ma
import scipy.linalg as linalg
import scipy.stats as stats

from numpy.random import randn

from riskmodels import Classification
from riskmodels.LegacyUtilities import Struct
from riskmodels import LegacyUtilities as Utilities

from riskmodels import MacroUtils
from riskmodels.MacroUtils import expandDatesDF

def _to_frame(series,name=None):
    """A replacement for pandas.Series.to_frame, which is not in version 10.1."""
    if name is None and hasattr(series,'name'):
            name=series.name
    assert name is not None, 'Name must be specified or series.name must exist.'
    return pandas.DataFrame({name:series})

class US3MacroDataManager(object):
    """ This is for QA purposes only.  It does not feed the actual data into the US3MH_M model."""

    def __init__(self,params={},dMin=datetime.date(1988,1,1),dMax=datetime.date(2199,12,31)):
        self.default_params={
                'outlierThresholdStdMacro':10.,
                'outlierThresholdStd':8.,
                'percentDailyBadAllowed':0.05,
                }
        self.params=Struct()
        self.params.update(self.default_params)
        self.params.update(params)
        self.dMin=dMin
        self.dMax=dMax
        self.a2s={'XDR':'XDR'}

    def _loadMacroTsInterpolatableSeries(self,axids,modelDB,marketDB,asofDate=None,fromDate=None,expand_dates=False):
        if asofDate is None:
            asofDate=self.dMax
        if fromDate is None:
            fromDate=self.dMin
        seriesByAxid=pandas.DataFrame(marketDB.getMacroTs(axids,self.dMin)).copy()
        if expand_dates:
            series=expandDatesDF(seriesByAxid)
        else:
            series=seriesByAxid.copy()
        return series.sort_index(0).loc[fromDate:asofDate].copy()

    def loadCommoditySpotPrice(self,modelDB,marketDB,asofDate=None,fromDate=None):
        a2s={'M000100117':'gsci_nonenergy_spot'}
        self.a2s.update(a2s)
        series=self._loadMacroTsInterpolatableSeries(list(a2s.keys()),modelDB,marketDB,asofDate,fromDate)
        return series.rename(columns=a2s).iloc[:,0].copy()

    def loadGoldSpotPrice(self,modelDB,marketDB,asofDate=None,fromDate=None):
        a2s={'M000100093':'gsci_gold_spot'}
        self.a2s.update(a2s)
        series=self._loadMacroTsInterpolatableSeries(list(a2s.keys()),modelDB,marketDB,asofDate,fromDate)
        return series.rename(columns=a2s).iloc[:,0].copy()

    def loadOilSpotPrice(self,modelDB,marketDB,asofDate=None,fromDate=None):
        a2s={'M000100022':'crude_oil_wti'}
        self.a2s.update(a2s)
        series=self._loadMacroTsInterpolatableSeries(list(a2s.keys()),modelDB,marketDB,asofDate,fromDate)
        return series.rename(columns=a2s).iloc[:,0].copy()

    def loadShortYieldUSD(self,modelDB,marketDB,asofDate=None,fromDate=None):
        a2s={'M000010002':'treasury_yield_13w'}
        self.a2s.update(a2s)
        series=self._loadMacroTsInterpolatableSeries(list(a2s.keys()),modelDB,marketDB,asofDate,fromDate)
        series*=0.1  #ADAM: no, this is not a bug.  the data seem to have an unusual scale.  consider using MAC data insted.
        series*=0.01 #Convert from percent to raw yield.
        return series.rename(columns=a2s).iloc[:,0].copy()

    def loadLongYieldUSD(self,modelDB,marketDB,asofDate=None,fromDate=None):
        a2s={'M000010001':'treasury_yield_10y'}
        self.a2s.update(a2s)
        series=self._loadMacroTsInterpolatableSeries(list(a2s.keys()),modelDB,marketDB,asofDate,fromDate)
        series*=0.1  #ADAM: no, this is not a bug.  the data seem to have an unusual scale.  consider using MAC data insted.
        series*=0.01 #Convert from percent to raw yield.
        return series.rename(columns=a2s).iloc[:,0].copy()

    def loadCreditSpreadConstituents(self,modelDB,marketDB,asofDate=None,fromDate=None):
        a2s={'M000010260':'ax_us_corp_spread_a',
            'M000010259':'ax_us_corp_spread_aa',
            'M000010258':'ax_us_corp_spread_aaa',
            'M000010261':'ax_us_corp_spread_bbb',
            'M000010262':'ax_us_corp_spread_sub_ig',
            'M000010257':'moodys_baa',
            'M000010256':'moodys_aaa' }
        self.a2s.update(a2s)
        series=self._loadMacroTsInterpolatableSeries(list(a2s.keys()),modelDB,marketDB,asofDate,fromDate)
        series=series.rename(columns=a2s)
        series[['moodys_baa','moodys_aaa']]*=0.01 #Convert from percent to raw spread
        return series

    def loadMacroeconomicSeriesMonthlyAsDF(self,modelDB,marketDB,asofDate=datetime.date(2199,1,1),fromDate=datetime.date(1988,1,1)):
        a2s={'M001000020': 'wages_nonfarm',
         'M001000021': 'wages_mfg',
         'M001000022': 'hours_nonfarm',
         'M001000038': 'caputil',
         'M001000043': 'chicago_pmbb',
         'M001000048': 'cci',
         'M001000050': 'cpi_core',
         'M001000052': 'cpi',
         'M001000054': 'disp_pers_inc',
         'M001000057': 'emp_nonfarm',
         'M001000074': 'indprod_mfg',
         'M001000075': 'ind_prod_total_idx',
         'M001000078': 'ism_pmi',
         'M001000084': 'nahb_housing_mkt_idx',
         'M001000093': 'pce',
         'M001000094': 'pers_inc',
         'M001000095': 'pers_sav_pct',
         'M001000096': 'phil_outlook_svy',
         'M001000098': 'ppi',
         'M001000099': 'ppi_core',
         'M001000106': 'leading_econ_idx',
         'M001000108': 'civ_empl',
         'M001000110': 'broad_fx',
         'M001000111': 'treas_yield_3m_monthly',
         'M001000112': 'treas_yield_20y_monthly',
         'M001000113': 'unrate_16_plus',
         'M001000114': 'unrate'
        }
        self.a2s.update(a2s)

        series=marketDB.getMacroEconTsAsOf(list(a2s.keys()),from_date=fromDate,asof_date=asofDate)
        series=series.rename(columns=a2s)
        return series

    def loadParentModelFactorReturns(self,modelDB,marketDB,asofDate=datetime.date(2199,1,1),fromDate=datetime.date(1988,1,1)):
        import riskmodels
        riskModel=riskmodels.getModelByName('USAxioma2013MH')(modelDB,marketDB)
        riskModel.setFactorsForDate(asofDate,modelDB)
        series=riskModel.loadFactorReturnsHistoryAsDF(fromDate,asofDate,modelDB)
        series=series.dropna(how='all')
        return series

    def getModelDates(self,modelDB,asofDate=datetime.date(2199,1,1),fromDate=datetime.date(1988,1,1)):
        series=self.loadParentModelFactorReturns(modelDB,marketDB,asofDate,fromDate) #UGLY.  Possibly slow.
        return list(series.index)
    
    def loadFxReturns(self,modelDB,marketDB,asofDate=datetime.date.today(),fromDate=datetime.date(1988,1,1)):
        """Note: this is by far the slowest operation."""
        import riskmodels
        startDate=fromDate
        endDate=asofDate
        currencies=['XDR']
        base='USD'
        history=(((endDate-startDate).days)*5)/7 + 128 #Approximation 
        riskModel=riskmodels.getModelByName('USAxioma2013MH')(modelDB,marketDB)
        riskModel.setFactorsForDate(endDate,modelDB)
        cRetRaw=modelDB.loadCurrencyReturnsHistory(riskModel.rmg,asofDate,history,currencies,base)
        cRetDF=cRetRaw.toDataFrame().T.copy()
        return cRetDF.copy() 
        
    def verifyEssentialModelTimeSeries(self,modelDB,marketDB,modelDate,force=False):
        """Check existance and plausibility of the important time series used in the macro model.
           force=True will prevent an exception from being thrown.  Warnings will be thrown and 
           missing or suspicious data will be proxied or padded.  
        """
        import riskmodels
        if force:
            log_error=logging.error
        else:
            log_error=logging.critical 

        #First get the important time series to define todays factor return
        data=Struct()
        riskModel=riskmodels.getModelByName('USAxioma2013MH_M')(modelDB,marketDB)
        riskModel.setFactorsForDate(modelDate,modelDB)
        macroconf=pandas.read_csv(riskModel.macroMetaDataFile,index_col=0)
        a2short=macroconf.shortname.copy()
        short2a=pandas.Series(a2short.index,a2short.values)
        fundRet=self.loadParentModelFactorReturns(modelDB,marketDB,modelDate,modelDate-datetime.timedelta(riskModel.factorHistory*(8./5.))).dropna()
        if fundRet.index[-1] != modelDate:
            logging.critical('USAxioma2013MH factor return not defined.')
            assert False, 'Please generate history of USAxioma2013MH fundamental model up to model date.'
        allModelDates=list(fundRet.index[-riskModel.factorHistory:])
        recentModelDates=list(fundRet.index[-min(300,riskModel.returnHistory):])
        assert len(allModelDates)==riskModel.factorHistory
        macroecon=self.loadMacroeconomicSeriesMonthlyAsDF(modelDB,marketDB,asofDate=modelDate,fromDate=min(allModelDates))
        cmdyCumRet=_to_frame(self.loadCommoditySpotPrice(modelDB,marketDB,modelDate,min(recentModelDates)))
        cmdyCumRet=cmdyCumRet.join(
                            self.loadGoldSpotPrice(modelDB,marketDB,modelDate,min(recentModelDates))).join(
                            self.loadOilSpotPrice(modelDB,marketDB,modelDate,min(recentModelDates))  )
        fxRet=self.loadFxReturns(modelDB,marketDB,modelDate,min(recentModelDates))
        fxCumRet=(1.+fxRet).cumprod()
        sovYields=_to_frame(self.loadShortYieldUSD(modelDB,marketDB,modelDate,min(recentModelDates)))
        sovYields=sovYields.join(
                                    self.loadLongYieldUSD(modelDB,marketDB,modelDate,min(recentModelDates)))
        creditSeries=self.loadCreditSpreadConstituents(modelDB,marketDB,modelDate,min(recentModelDates))

        data.modelDate=modelDate
        data.allModelDates=allModelDates
        data.fundRet=fundRet
        data.macroecon=macroecon
        data.cmdyCumRet=cmdyCumRet
        data.fxRet=fxRet
        data.sovYields=sovYields
        data.creditSeries=creditSeries
        
        today=recentModelDates[-1]
        yesterday=recentModelDates[-2]

        dailySeries=cmdyCumRet.join(fxCumRet).join(sovYields).copy()
        d0=datetime.date(2009,8,1) #Before this, use moodys
        d1=datetime.date(2012,1,1) #After this use MAC.  in between slowly change
        if modelDate<d0:
            dailySeries=dailySeries.join(creditSeries[['moodys_aaa','moodys_baa']])
        elif modelDate>d1:
            dailySeries=dailySeries.join(creditSeries[['ax_us_corp_spread_a']])
        else: 
            dailySeries=dailySeries.join(creditSeries[['moodys_aaa','moodys_baa','ax_us_corp_spread_a']])

        data.dailySeries=dailySeries
        logIdx=['gsci_nonenergy_spot', 'gsci_gold_spot', 'crude_oil_wti', 'XDR'] 
        data.logDS=dailySeries.copy()
        data.logDS[logIdx]=np.log(dailySeries[logIdx])
        data.logDSinterp=expandDatesDF(data.logDS) 
        data.dailySeriesInterp=np.exp(data.logDSinterp)
        dailySeriesInterp=data.dailySeriesInterp

        #Now see if we have data needed to define factor returns.  Just check, existance.  Not quality.
        ds=dailySeries.loc[[yesterday,today]].copy()
        if any(ds.count()<ds.shape[0]):
            missingData=ds.loc[:,ds.count()<ds.shape[0]]
            badIdx=missingData.columns
            for i in badIdx:
                log_error('Daily series %s has missing value.  Cant compute a return.\n%s',i,missingData[i])
            if len(badIdx)>0 and not force:
                assert False, 'Missing macro data.  See critical log message'

        #The macro series may not be available for the most recent month or two.  Thats ok.
        me=macroecon.iloc[:-2].copy()
        if any(me.count()<me.shape[0]):
            badIdx=me.loc[:,me.count()<me.shape[0]].columns
            for i in badIdx:
                logging.warning('Macroecon series %s has only %s of %s months: %s ',i,me.count()[i],me.shape[0],list(me[i].loc[np.isnan(me[i])].index))
        
        me=macroecon.iloc[:-3][['ind_prod_total_idx','cpi','cci']].copy() #Instruments are very important.  They must exist.
        if any(me.count()<me.shape[0]):
            badIdx=me.loc[:,me.count()<me.shape[0]].columns
            for i in badIdx:
                log_error('Macroecon series %s (%s) has only %s of %s months: %s ',i,short2a[i],me.count()[i],me.shape[0],list(me[i].loc[np.isnan(me[i])].index))
            if len(badIdx)>0 and not force:
                assert False, 'Missing macroeconomic data.  See critical log message'
        
        #Now check quality of current daily data on modelDate, assuming previous values are correct.
        dsDates=[d for d in allModelDates if dailySeries.index[0].date()<=d<=dailySeries.index[-1].date()]
        dsTmp=dailySeries.reindex(index=dsDates).stack(dropna=False)
        if any(np.isnan(dsTmp)):
            logging.info('Values that need to be proxied:\n%s',dsTmp.loc[np.isnan(dsTmp)])
        if any(np.isnan(dailySeries).sum()>self.params.percentDailyBadAllowed*dailySeries.shape[0]):        
            dsBad=dailySeries.loc[:,np.isnan(dailySeries).sum()>self.params.percentDailyBadAllowed*dailySeries.shape[0]]
            badIdx=dsBad.columns
            for i in badIdx:
                logging.warning('Too many missing values (%s of %s) for macro series %s (%s) on %s',
                        dsBad[i].dropna().shape[0],dsBad.shape[0],i,short2a[i], ', '.join([str(d.date()) for d in dsBad[i].loc[np.isnan(dsBad[i])].index]))

        dsDiff=(dailySeriesInterp-dailySeriesInterp.shift(1)).dropna()
        dsDiffRaw=dsDiff.copy()
        for i,x in dsDiff.iteritems():
            dsDiff[i]=x.clip(lower=x.quantile(0.05),upper=x.quantile(0.95))
        
        dsStd=dsDiff.std()
        isOutlier=np.abs(dsDiff.loc[modelDate])>self.params.outlierThresholdStd*dsStd
        if any(isOutlier):
            outliers=dailySeries.iloc[-10:,:].loc[:,isOutlier.index].loc[:,isOutlier.copy()]
            logging.warning('Possible outliers in macro time series (%s) for model date %s. Recent history:  \n%s', 
                    ', '.join(['%s:%s'%(i,short2a[i]) for i in outliers.columns]),modelDate,outliers)

    
        #Now check quality of the macro data for modelDate.  
        outlierThresholdStdMacro=self.params.outlierThresholdStdMacro 
        for c,xRaw in macroecon.iteritems():
            x=xRaw.dropna()
            if macroconf.loc[short2a[c]]['type']=='geom':
                x=np.log(x)
            if x.shape[0]<=30:
                logging.warning('History too short for %s (%s) on %s. \n%s',c,short2a[c],modelDate,x)
            dates=list(x.index)[30:]
            badDates=set()
            for d in dates:
                ixv = x.loc[:d].iloc[-120:-1].index.values
                xval = x.shift(1).loc[ixv].values
                xval = sm.add_constant(xval)
                fit=sm.OLS(x.loc[ixv].values, xval, missing='drop').fit()
                predX=fit.params.dot(np.array([1.0, x.loc[d]]))
                err=x.loc[d]-predX
                fit.resid = pandas.Series(fit.resid)
                iqr=fit.resid.quantile(0.8)-fit.resid.quantile(0.2)
                residStd=fit.resid.clip(lower=fit.resid.quantile(0.1),upper=fit.resid.quantile(0.9), axis=None).std()
                err_norm=err/(residStd)
                logging.debug('Testing point 1, avg_err, %.8f, iqr, %.8f', np.average(err, axis=None), iqr)
                #if np.abs(err_norm)>outlierThresholdStdMacro:
                if np.abs(err)>iqr*outlierThresholdStdMacro:
                    logging.warning('Detected a large (%s std) event for series %s (%s) at date %s on modelDate %s',err_norm, c,short2a[c], d, modelDate)
                    logging.warning('Subsequent output for this series may not make sense.  Please check by hand.')
                    badDates.update([d])
            #goodDates=sorted(set(dates).difference(badDates))
            if len(badDates)>0:
                y=x.loc[:min(badDates)].iloc[:-1].copy()
            else:
                y=x.copy()
            yShiftTmp=_to_frame(y.shift(1))
            yShiftTmp = yShiftTmp.values
            yShiftTmp = sm.add_constant(yShiftTmp)
            xShiftTmp=_to_frame(x.shift(1))
            interceptTmp=_to_frame(1.+0.*x,'intercept')
            fit=sm.OLS(y.values, yShiftTmp, missing='drop').fit()
            #fit=pandas.ols(y=y,x=y.shift(1).to_frame())
            resid=(x - xShiftTmp.join( interceptTmp ).dot( np.flipud(fit.params) )  ).dropna()
            #resid=(x - x.shift(1).to_frame().join( (1.+0.*x).to_frame('intercept') ).dot( fit.beta )  ).dropna()
            residTrim=resid.clip(lower=resid.quantile(0.1),upper=resid.quantile(0.9))
            iqr=resid.quantile(0.8)-resid.quantile(0.2)
            logging.debug('Testing point 2, avg_err, %.8f, iqr, %.8f', np.average(residTrim, axis=None), iqr)
            isOutlier=np.abs(resid)>outlierThresholdStdMacro*iqr
            if any(isOutlier):
                outlierDates=list(isOutlier[isOutlier].index)
                outlierNeighbors=(isOutlier+isOutlier.shift(1)+isOutlier.shift(2)+isOutlier.shift(3)+isOutlier.shift(-1)+isOutlier.shift(-2)+isOutlier.shift(-3)).dropna()
                logging.warning('Possible outliers and their neighboring values for %s (%s) as of modelDate %s: %s \n%s', 
                        c, short2a[c], modelDate,', '.join([str(d.date()) for d in outlierDates]),xRaw.reindex(outlierNeighbors.index).loc[outlierNeighbors>0])
