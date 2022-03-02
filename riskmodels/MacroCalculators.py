
import logging
import datetime
import itertools

import pandas
import numpy as np
import statsmodels.api as sm

from riskmodels import LegacyUtilities as Utilities
from riskmodels.LegacyUtilities import Struct

from riskmodels.MacroUtils import batchOLSRegression, expandDatesDF, FactorExtractorPCAar1, \
        fitVARSimple,d2dt,dt2d,expandDatesDF,LinearModel,_getDaysInUniformMonth,ReducedRankRegressionSimple,NestedLinearModel,Orthogonalizer,BayesianLinearModel
from riskmodels import MacroUtils
oldPD = Utilities.oldPandasVersion()

class OverlappingExposureCalculator(object):
    """ Time series exposure calculator. """
    def __init__(self,params=None): 
        self.params=Struct()
        self.params.freq=252 
        self.params.window=20 
        self.params.geometric=False
        self.params.use_gls=False
        if params is None:
            params=Struct() 
        self.params.update(params)

    def compute(self,data,params=None):
        """ Simplest possible time series regression."""       
        if params is None:
            params=self.params
        #dates=data.dates
        #TODO: make sure there are not gaps or nan.
        X=data.factorReturnsHistory.ix[data.dates].copy().dropna()
        #Y=data.assetReturnsHistory.ix[data.dates].copy().dropna()
        Y=data.assetReturnsHistory.ix[X.index].copy().fillna(0.) #HACK: 
        dates=sorted(set(X.index).intersection(Y.index))
        assert data.date in dates

        data.X_pre=X.copy()
        data.Xorth=X.copy()
        data.Y_pre=Y.copy()
        data.Y_post=Y.copy()
        data.X_post=X.copy()
        if params.geometric:
            cX = (X.ix[dates].fillna(0.) + 1.0).cumprod()
            XX = cX/cX.shift(params.window) - 1.0
            cY = (Y.ix[dates].fillna(0.) + 1.0).cumprod()
            YY = cY/cY.shift(params.window) - 1.0
            XX = XX.dropna()
            YY = YY.dropna()
        else:
            XX=pandas.rolling_sum(X.ix[dates].fillna(0.),
                    window=params.window,min_periods=params.window)
            YY=pandas.rolling_sum(Y.ix[dates].fillna(0.),
                    window=params.window,min_periods=params.window)
            XX=XX.dropna()
            YY=YY.dropna()
        
        #Filter out zero columns
        isallzero = (XX == 0.0).all()
        XX = XX.ix[:,~isallzero]
        allzerocols = isallzero[isallzero].index
        
        if params.use_gls:
            lm = LinearModel(YY,XX,window=params.window)
        else:
            lm = LinearModel(YY,XX)
        em=lm.beta.copy()
        for c in allzerocols:
            em[c] = 0.0

        sr=Y-X.dot(em.T)
        
        data.results.exposures=em
        data.results.specificReturns=sr.ix[data.date].copy()

    def _process(self,X,Y,params,data=None):
        """Here is where the work is done. Override as needed."""
        tsFit=batchOLSRegression(Y=Y,X=X,intercept=False)
        return tsFit.B.copy(),tsFit.resid.copy(),tsFit

class NonOverlappingExposureCalculator(object):
    """ Time series exposure calculator. """
    def __init__(self,params=None): 
        self.params=Struct()
        self.params.freq=252 
        self.params.window=20 
        self.params.geometric=False
        if params is None:
            params=Struct() 
        self.params.update(params)

    def compute(self,data,params=None):
        """ Simplest possible time series regression."""       
        if params is None:
            params=self.params
        #dates=data.dates
        #TODO: make sure there are not gaps or nan.
        X=data.factorReturnsHistory.ix[data.dates].copy().dropna()
        #Y=data.assetReturnsHistory.ix[data.dates].copy().dropna()
        Y=data.assetReturnsHistory.ix[X.index].copy().fillna(0.) #HACK: 
        dates=sorted(set(X.index).intersection(Y.index))
        assert data.date in dates

        data.X_pre=X.copy()
        data.Xorth=X.copy()
        data.Y_pre=Y.copy()
        data.Y_post=Y.copy()
        data.X_post=X.copy()
        if params.geometric:
            cX = (X.ix[dates].fillna(0.) + 1.0).cumprod()
            cY = (Y.ix[dates].fillna(0.) + 1.0).cumprod()
            revdates = sorted(cX.index, reverse=True)[0::params.window]
            dates = sorted(revdates)
            cX = cX.ix[dates]
            cY = cY.ix[dates]
            XX = cX/cX.shift(1) - 1.0
            YY = cY/cY.shift(1) - 1.0
            XX = XX.dropna()
            YY = YY.dropna()
        else:
            raise NotImplementedError
        data.assetExposures,data.assetResiduals,data.BFitStats=(
                self._process(XX,YY,params,data)  )
        em=data.assetExposures.copy()
        sr=Y-X.dot(em.T)
        data.results.exposures=data.assetExposures.copy()
        data.results.specificReturns=sr.ix[data.date].copy()
        #data.results.specificReturns=data.assetResiduals.ix[data.date].copy()

    def _process(self,X,Y,params,data=None):
        """Here is where the work is done. Override as needed."""
        tsFit=batchOLSRegression(Y=Y,X=X,intercept=False)
        return tsFit.B.copy(),tsFit.resid.copy(),tsFit

class DefaultExposureCalculator(object):
    """ Time series exposure calculator. """
    def __init__(self,params=None): 
        self.params=Struct()
        self.params.freq=252 
        if params is None:
            params=Struct() 
        self.params.update(params)

    def compute(self,data,params=None):
        """ Simplest possible time series regression."""       
        if params is None:
            params=self.params
        dates=data.dates
        #TODO: make sure there are not gaps or nan.
        X=data.factorReturnsHistory.ix[dates].copy()
        Y=data.assetReturnsHistory.ix[dates].copy()
        data.X_pre=X.copy()
        data.Xorth=X.copy()
        data.Y_pre=Y.copy()

        data.Y_post=Y.copy()
        data.X_post=X.copy()
        Y=Y.fillna(0.) #HACK: should not be empty.  but is sometimes....
        data.assetExposures,data.assetResiduals,data.BFitStats=(
                self._process(X,Y,params,data)  )
        data.results.exposures=data.assetExposures.copy()
        data.results.specificReturns=data.assetResiduals.ix[data.date].copy()

    def _process(self,X,Y,params,data=None):
        """Here is where the work is done. Override as needed."""
        tsFit=batchOLSRegression(Y=Y,X=X,intercept=False)
        return tsFit.B.copy(),tsFit.resid.copy(),tsFit

class RobustExposureCalculator(object):
    """ Time series exposure calculator. """
    def __init__(self,params=None): 
        self.params=Struct()
        self.params.freq=252 
        if params is None:
            params=Struct() 
        self.params.update(params)

    def compute(self,data,params=None):
        """ Robust possible time series regression."""       
        logging.info('robust regressions: begin')
        if params is None:
            params=self.params
        dates=data.dates
        #TODO: make sure there are not gaps or nan.
        X=data.factorReturnsHistory.ix[dates].fillna(0.0)
        Y=data.assetReturnsHistory.ix[dates].fillna(0.0)
        
        #Filter out zero columns
        isallzero = (X == 0.0).all()
        XX = X.ix[:,~isallzero]
        allzerocols = isallzero[isallzero].index
        betas = {}
        for name, y in Y.items():
            model = sm.RLM(endog=y,exog=XX,M=sm.robust.norms.HuberT(t=4.0)) 
            result = model.fit()
            betas[name] = result.params
        em = pandas.DataFrame(betas).T
        for c in allzerocols:
            em[c] = 0.0
        sr=Y-X.dot(em.T)
        data.results.exposures=em
        data.results.specificReturns=sr.ix[data.date]
        logging.info('robust regressions: end')

class BayesianExposureCalculator(object):
    """ Time series exposure calculator. """
    def __init__(self,params=None): 
        self.params=Struct()
        self.params.freq=252 
        if params is None:
            params=Struct() 
        self.params.update(params)

    def compute(self,data,params=None):
        """ Simplest possible time series regression."""       
        if params is None:
            params=self.params
        dates=data.dates
        #TODO: make sure there are not gaps or nan.
        X=data.factorReturnsHistory.ix[dates].copy()
        Y=data.assetReturnsHistory.ix[dates].copy()
        fr=data.allCumRetsD[data.indexD['industries'] + data.indexD['market intercept']]
        B=data.B
        raise Exception('Not implemented yet')
        data.results.exposures=data.assetExposures.copy()
        data.results.specificReturns=data.assetResiduals.ix[data.date].copy()

class NestedExposureCalculator(object):
    """ Time series exposure calculator. """
    def __init__(self,params=None): 
        self.params=Struct()
        self.params.freq=252 
        if params is None:
            params=Struct() 
        self.params.update(params)

    def compute(self,data,params=None):
        """ Simplest possible time series regression."""       
        if params is None:
            params=self.params
        dates=data.dates
        #TODO: make sure there are not gaps or nan.
        X=data.factorReturnsHistory.ix[dates].copy()
        Y=data.assetReturnsHistory.ix[dates].copy()
        isallzero = (X == 0.0).all()
        XX = X.ix[:,~isallzero]
        allzerocols = isallzero[isallzero].index
        model = NestedLinearModel(Y,XX,blocks=data.blocks,orthogonalize=True) 
        em=model.beta.copy()
        for c in allzerocols:
            em[c] = 0.0
        sr=Y-X.dot(em.T)
        data.results.exposures=em
        data.results.specificReturns=sr.ix[data.date]

class SectorExposureCalculator(object):
    """ Time series exposure calculator. """
    def __init__(self,params=None): 
        self.params=Struct()
        self.params.freq=252 
        self.params.sector_residual=False
        #self.params.blocks=None
        if params is None:
            params=Struct() 
        self.params.update(params)

    def compute(self,data,params=None):
        """ Simplest possible time series regression."""       
        if params is None:
            params=self.params
        dates=data.dates
        #TODO: make sure there are not gaps or nan.
        X=data.factorReturnsHistory.loc[dates].copy()
        Y=data.clippedReturnsHistory.loc[dates].copy()
        if data.blocks is None: #This part used to have a bug.  refered to params.blocks, which was None
            regress_blocks = [[c for c in X.columns]]
        else:
            regress_blocks = data.blocks
            if sorted(itertools.chain.from_iterable(regress_blocks)) != sorted(X.columns):
                raise Exception('Blocks must be partition of columns of X')
        isallzero = (X == 0.0).all()
        XX = X.loc[:,~isallzero]
        allzerocols = isallzero[isallzero].index
       
        # Regress sector by sector
        all_sectors = list(data.sector2asset.keys())
        all_betas = []
        for sector, subids in data.sector2asset.items():
            to_drop = [s for s in all_sectors if s != sector]
            subX = XX.drop(to_drop, axis=1)
            if params.sector_residual:
                tmpblocks = [[f for f in block if f not in all_sectors] for block in regress_blocks] +  [[sector]]
            else:
                tmpblocks = [[f for f in block if f not in to_drop] for block in regress_blocks]
            model = NestedLinearModel(Y[subids],subX,blocks=tmpblocks)
            all_betas.append(model.beta)
        if oldPD:
            betas = pandas.concat(all_betas, axis=0)
        else:
            betas = pandas.concat(all_betas, axis=0, sort=True)
        missing = list(set(Y.columns) - set(betas.index))
        if len(missing) > 0:
            model = NestedLinearModel(Y[missing], XX,blocks=regress_blocks)
            if oldPD:
                betas = pandas.concat([betas, model.beta], axis=0)
            else:
                betas = pandas.concat([betas, model.beta], axis=0, sort=True)
        betas = betas.fillna(0.0) 
        for c in allzerocols:
            betas[c] = 0.0
        sr=Y-X.dot(betas.T)
        data.results.exposures=betas
        data.results.specificReturns=sr.loc[data.date]

class BayesianSectorExposureCalculator(object):
    """ Time series exposure calculator. """
    def __init__(self,params=None): 
        self.params=Struct()
        self.params.freq=252 
        self.params.sector_residual=False
        if params is None:
            params=Struct() 
        self.params.update(params)

    def compute(self,data,params=None):
        """ Simplest possible time series regression."""       
        if params is None:
            params=self.params
        if params.sector_residual:
            raise NotImplementedError
        dates=data.dates
        clsdata=data.clsdata
        #TODO: make sure there are not gaps or nan.
        X=data.factorReturnsHistory.ix[dates].copy()
        # Y=data.assetReturnsHistory.ix[dates].copy()
        Y=data.clippedReturnsHistory.ix[dates].copy()
        fr=data.fundamentalFactorReturns.ix[dates].copy()
        fr = fr.add(fr['Market Intercept'],axis=0)
        fr = fr[clsdata.industry.unique()]

        isallzero = (X == 0.0).all()
        XX = X.ix[:,~isallzero]
        allzerocols = isallzero[isallzero].index
       
        # Regress sector by sector
        ind2sector = dict(zip(clsdata.industry.values, clsdata.sector.values))
        sectors = clsdata.sector.unique()
        betas = []
        for name, subdf in clsdata.groupby('industry'):
            assets = subdf.index.intersection(Y.columns)
            sector = ind2sector[name]
            to_drop = [s for s in sectors if s != sector] 
            subX = XX.drop(to_drop, axis=1)
            frmodel = LinearModel(fr[name], subX)
            model = BayesianLinearModel(Y[assets],subX,frmodel.xtx,priorbeta=frmodel.beta)
            betas.append(model.beta)
        betas = pandas.concat(betas)

        missing = Y.columns - betas.index
        if len(missing) > 0:
            model = LinearModel(Y[missing], XX)
            betas = pandas.concat([betas, model.beta], axis=0)
        betas = betas.fillna(0.0) 
        
        for c in allzerocols:
            betas[c] = 0.0
        sr=Y-X.dot(betas.T)
        data.results.exposures=betas
        data.results.specificReturns=sr.ix[data.date]


class DefaultSpecificRisksCalculator(object):
    """ Time series exposure calculator. """
    def __init__(self,params=None): 
        self.params=Struct()
        self.params.freq=252 
        self.params.riskHorizon=400
        if params is None:
            params=Struct() 
        self.params.update(params)

    def compute(self,data,params=None):
        """ Simplest possible time series regression."""       
        if params is None:
            params=self.params
            
        X=data.assetResiduals.ix[-1*(data.riskHorizon):]
        data.specificRisks=np.sqrt(float(params.freq))*(X.std())

class DefaultFactorCovarianceCalculator(object):
    """ Factor covariance calculator. """
    def __init__(self,params=None): 
        self.params=Struct()
        self.params.freq=252 
        if params is None:
            params=Struct() 
        self.params.update(params)

    def compute(self,data,params=None):
        """ Simplest possible time series regression."""       
        if params is None:
            params=self.params
        X=data.factorReturns
        data.factorCovariance=(float(params.freq))*(X.cov())
    

class MacroFactorReturnCalculator_base(object):
    """This class is the currently best of the non-core factors. """
    def __init__(self,params=None):
        self.params=Struct()
        self.params.macro_equity=['Equity Size','Equity Value','Equity Market'] 
        self.params.orthogonalize=True
        self.params.factorsToKill=[]
        self.params.include_sectors=False
        if params is None:
            params=Struct() 
        self.params.update(params)
  
    def processTrivialCoreMacroFactors(self, modelData):
        """ Define trivial factor returns."""
        fRetsD=pandas.DataFrame(index=modelData.dates,
                columns=['Economic Growth','Inflation','Confidence',],
                dtype=float).fillna(0.)
        fCumRetsD=(1.+fRetsD).cumprod()

        modelData.fRetCalcResults.fCumRetsD_MacroCore=fCumRetsD
        modelData.fRetCalcResults.fRetsD_MacroCore=fRetsD

    def _processCoreMacroData(self,modelData):
        """ Define trivial factor returns."""
        self.processTrivialCoreMacroFactors(modelData)

    def _processMarketTradedData(self,modelData):
        """Define trivial factor returns."""
        """
        fRetsD=pandas.DataFrame(index=modelData.tradingDays,
                columns=['Commodity','Term Spread','FX Basket'],
                dtype=float).fillna(0.)
        """
        idxLin=['Credit Spread','Term Spread']
        idxGeom=['FX Basket','Oil','Gold','Commodity'] + self.params.macro_equity
        idxMarket=['Credit Spread','Term Spread','FX Basket','Oil','Gold','Commodity']
        idxEquity=self.params.macro_equity
        idxNames=idxLin+idxGeom
        dfRaw=modelData.compositeRawD[idxNames].copy()
        df=expandDatesDF(dfRaw).loc[modelData.dates].copy()

        fCumRet=df[idxGeom]
        fRet=(fCumRet/(fCumRet.shift(1)) - 1.)
        fRetsD=fRet.join(df[idxLin]-(df.shift(1)[idxLin])).fillna(0.) #HACK?

        fRetsD['Oil']-=fRetsD['Commodity']
        fRetsD['Gold']-=fRetsD['Commodity']

        fCumRetsD=(1.+fRetsD).cumprod()

        modelData.fRetCalcResults.fCumRetsD_MacroMarketTraded=fCumRetsD.iloc[1:,:].loc[:,idxMarket].copy()
        modelData.fRetCalcResults.fRetsD_MacroMarketTraded=fRetsD.iloc[1:,:].loc[:,idxMarket].copy()
        
        modelData.fRetCalcResults.fCumRetsD_MacroEquity=fCumRetsD.iloc[1:,:].loc[:,idxEquity].copy()
        modelData.fRetCalcResults.fRetsD_MacroEquity=fRetsD.iloc[1:,:].loc[:,idxEquity].copy()
        if self.params.include_sectors:
            sectorReturns = modelData.sectorReturns.reindex(index=modelData.fRetCalcResults.fRetsD_MacroEquity.index)
            modelData.fRetCalcResults.fRetsD_MacroEquity = modelData.fRetCalcResults.fRetsD_MacroEquity.join(sectorReturns, how='outer')

    def _processMacroEquityData(self,modelData):
        pass #already done in MarketTraded

    def residualizeBlocks(self,modelData):
        """ TODO: Operate on modelData.fRetCalcResults.f(Cum)RetsD"""
        fRets=modelData.fRetCalcResults.fRetsD.copy().iloc[1:]
        gRets=Orthogonalizer(fRets,modelData.blocks).Xorth
        modelData.fRetCalcResults.fRetsD=gRets
        modelData.fRetCalcResults.fCumRetsD=(1. + gRets).cumprod()

    def compute(self,modelData):
        """ Do the work."""
        self.days_per_month=23 #HARDCODE
        modelData.fRetCalcResults=Struct()
        self._processCoreMacroData(modelData)
        self._processMarketTradedData(modelData)
        self._processMacroEquityData(modelData)
        dates=modelData.dates

        fCumRets=modelData.fRetCalcResults.fCumRetsD_MacroCore.copy()
        fCumRets=fCumRets.join(modelData.fRetCalcResults.fCumRetsD_MacroMarketTraded)
        fCumRets=fCumRets.join(modelData.fRetCalcResults.fCumRetsD_MacroEquity)

        fRets=modelData.fRetCalcResults.fRetsD_MacroCore.copy()
        fRets=fRets.join(modelData.fRetCalcResults.fRetsD_MacroMarketTraded)
        fRets=fRets.join(modelData.fRetCalcResults.fRetsD_MacroEquity)
    
        #Make sure these are free of NAN.
        #Also that dates are always actuall there...
        modelData.fRetCalcResults.fCumRetsD=fCumRets.loc[dates].copy()         
        modelData.fRetCalcResults.fRetsD=fRets.loc[dates].copy()
        modelData.fRetCalcResults.fCumRetsDi_orig=fCumRets.loc[dates].copy()         
        modelData.fRetCalcResults.fRetsD_orig=fRets.loc[dates].copy()
       
        if self.params.orthogonalize:
            self.residualizeBlocks(modelData)
        
        fRets=modelData.fRetCalcResults.fRetsD
        for c in set(fRets.columns).intersection(self.params.factorsToKill):
            fRets[c]*=0.
        fRets['Equity Market'] -= 4*fRets['Economic Growth']

        fCumRets=modelData.fRetCalcResults.fCumRetsD
        
        modelData.results.factorReturnsDF=fRets.loc[:modelData.date].copy()
        modelData.results.factorReturns=fRets.loc[modelData.date].copy()

class MacroFactorReturnsCalculator_CoreSSM_bare(MacroFactorReturnCalculator_base):
    """...  """
    def __init__(self,params=None):
        MacroFactorReturnCalculator_base.__init__(self,params)
        assert 'macroMetaCsvFile' in self.params.getFieldNames(), 'Must set macroMetaCsfVile in riskModel'

        if 'residualize_delta' not in self.params.getFieldNames():
            self.params.residualize_delta=True

        if 'macroSeriesIdx' not in self.params.getFieldNames():
            self.params.macroSeriesIdx=['unrate','unrate_16_plus','treas_yield_20y_monthly','treas_yield_3m_monthly',
            'broad_fx','civ_empl','leading_econ_idx','ppi_core','ppi','phil_outlook_svy','pers_sav_pct',
            'pers_inc','nahb_housing_mkt_idx','ism_pmi','ind_prod_total_idx','indprod_mfg','emp_nonfarm',
            'disp_pers_inc','cpi','cpi_core','cci','chicago_pmbb','caputil',
            'hours_nonfarm','wages_mfg','wages_nonfarm']+['pce'] #This one has problems in the db.
        
        if 'macroSeriesTransform' not in self.params.getFieldNames():
            self.params.macroSeriesTransform={
                    'ind_prod_total_idx':{'name':'ind_prod_lead','shift':-1,'diff':1,'geom':True},
                    'cpi':               {'name':'cpi_d2',       'shift':0,'diff':2,'geom':True},
                    'cpi_core':          {'name':'cpi_core_d2',  'shift':0,'diff':2,'geom':True},
                    'ppi':               {'name':'ppi_d2',       'shift':0,'diff':2,'geom':True},
                    'ppi_core':          {'name':'ppi_core_d2',  'shift':0,'diff':2,'geom':True},
                    }
                    
        if 'f2scale' not in self.params.getFieldNames():
            self.params.f2scale={'Economic Growth':0.03,'Inflation':0.05,'Confidence':0.05,}

        if 'useExcessIndustries' not in self.params.getFieldNames():
            self.params.useExcessIndustries=False


    def _processCoreMacroData(self,modelData):
        logging.info('Begin: _processCoreMacroData')
        self.processTrivialCoreMacroFactors(modelData)

        fCumRetsDcoreTriv=modelData.fRetCalcResults.fCumRetsD_MacroCore.copy()
        fRetsDcoreTriv=modelData.fRetCalcResults.fRetsD_MacroCore.copy()
        #dates=list(fCumRetsDcore.index)
        dates=[datetime.datetime(d.year,d.month,d.day) for d in modelData.dates]

        factorNames=sorted(fCumRetsDcoreTriv.columns)
        instruments=sorted(self.params.f2i.values())
        i2f=dict( (v,k) for k,v in self.params.f2i.items())
        Nstatic=len(factorNames)
        if 'Nstatic' in self.params.getFieldNames():
            Nstatic=self.params.Nstatic

        metaAll=pandas.read_csv(self.params.macroMetaCsvFile)

        macroSeriesIdx=self.params.macroSeriesIdx   
 
        if self.params.useExcessIndustries:
            industryCols=modelData.master.industryReturns.columns
        else:
            industryCols=modelData.master.industryPlusMarketInterceptReturns.columns
        marketSeriesIdx=(list(modelData.master.marketInterceptReturns.columns)
                        +list(industryCols)
                        +list(modelData.master.styleReturns.columns)
                        +list(modelData.indexD['fx'])
                        +list(modelData.indexD['commodity']) #Careful with TR, etc
                        #+list(modelData.indexD['benchmarks'])
                        #TODO: add bond yields and credit spreads.
                        #      add vix, etc...
                        )
        marketSeriesIdx=sorted(set(marketSeriesIdx).intersection(modelData.allCumRetsD.columns))
        logging.info('macroSeriesIdx  '+','.join(macroSeriesIdx))
        logging.info('marketSeriesIdx  '+','.join(marketSeriesIdx))

        fCumRetD=modelData.allCumRetsD[marketSeriesIdx].loc[dates].copy() 

        extraMacroSeries={}
        macroSeriesMonthly=modelData.macroSeriesMonthly.copy()
        for k,v in self.params.macroSeriesTransform.items():
            tmpSeries=macroSeriesMonthly[k].copy().shift(v['shift'])
            tmpSeries.name=v['name']
            
            if v['geom']:
                tmpSeries/=tmpSeries.iloc[0]
                tmpSeries=np.log(tmpSeries)
            for i in range(v['diff']):
                tmpSeries-=tmpSeries.shift(1)
            macroSeriesMonthly=macroSeriesMonthly.join(tmpSeries)
            extraMacroSeries[tmpSeries.name]=tmpSeries.copy()

        macroSeriesMonthly=macroSeriesMonthly.loc[dates[0]:dates[-1]].copy()

        results=self.extractCoreFactorReturns(instruments,fCumRetD,macroSeriesMonthly,
                                          marketSeriesIdx,macroSeriesIdx+list(extraMacroSeries.keys()),metaAll,Lmax=4,ssm_iters=4)
        modelData.tmpResults=results

        fRetsDcore=results.dailyFactorReturnsNormalized.rename(columns=i2f).reindex(columns=factorNames).fillna(0.) 
        modelData.tmp_fRetsDcore=fRetsDcore
        fRetsDcore.iloc[0]=0. #HACK: for good measure. will make essentially no difference
        for c in fRetsDcore.columns:
            fRetsDcore[c]*=self.params.f2scale.get(c,0.01)

        fCumRetsDcore=(1.+fRetsDcore).cumprod()
        modelData.fRetCalcResults.fCumRetsD_MacroCore=fCumRetsDcore
        modelData.fRetCalcResults.fRetsD_MacroCore=fRetsDcore
        logging.info('End: _processCoreMacroData')

    def extractCoreFactorReturns(self,instruments,fCumRetD,macroSeriesMonthly,
                             marketSeriesIdx,macroSeriesIdx,metaAll,Lmax=4,ssm_iters=4):
        meta=metaAll.loc[metaAll.active>0]
        meta=meta.rename(index=meta.shortname)
        metaM=meta[meta.freq=='M'].copy()
        metaD=meta[meta.freq=='D'].copy()

        geomIdxD=sorted(metaD[metaD.type=='geom'].index)
        percentIdxD=sorted(metaD[metaD.type=='percent'].index)

        geomIdxM=sorted(metaM[metaM.type=='geom'].index)
        percentIdxM=sorted(metaM[metaM.type=='percent'].index)
        linIdxM=sorted(metaM[metaM.type=='lin'].index)
        macroM=MacroUtils.d2dt(macroSeriesMonthly)
        tmp=macroM[geomIdxM].copy()
        macroM[tmp.columns]=tmp/(tmp.shift(1)) -1.
        tmp=macroM[percentIdxM].copy()
        macroM[tmp.columns]=tmp - (tmp.shift(1))
        macroM=macroM.iloc[1:].copy()

        fCumRetD=MacroUtils.d2dt(fCumRetD).dropna(axis=1) # HACK: crude to dropna.needed if we orth returns here  
        fRetD=(fCumRetD/(fCumRetD.shift(1)) -1.)#.dropna()
        fCumRetDi=MacroUtils.d2dt(MacroUtils.expandDatesDF(fCumRetD))
        fCumRetM=fCumRetDi.loc[[d for d in fCumRetDi.index if d.day==1]]
        fRetM=(fCumRetM/fCumRetM.shift(1) -1.).iloc[1:].copy()
        
        dMin=max(fRetM.index[0],macroM.index[0])
        dMax=fCumRetD.index[-1]
        
        X=macroM[macroSeriesIdx].loc[dMin:dMax].copy()
        X=X.join(fRetM)
        Xdaily=fRetD.copy()
        sigmaX=X.std()
        Y=X.copy()
        Y-=Y.mean()
        Y/=Y.std()
        Xnorm=X/Y.std()
        marketSeriesIdxAvailable=sorted(set(marketSeriesIdx).intersection(Xdaily.loc[dMax].dropna().index))
        noAR1CoeffIdx=list(marketSeriesIdx)+list(self.params.macroSeriesTransform.keys())

        Nstatic=self.params.Nstatic
                
        fe=MacroUtils.FactorExtractorPCAar1(Y.dropna(),noAR1CoeffIdx) #HACK: 
        results=fe.estimate(Nstatic=Nstatic,Niters=ssm_iters)
        instRetM=results.F.dot(results.Lambda[instruments])

        eta=results.eta
        Psi=results.Psi.T
        Lambda=results.Lambda.T
        F=results.F

        Psis={}
        Psis[0]=np.eye(Nstatic)+0.*Psi
        for i in range(1,Lmax):
            Psis[i]=Psi.dot(Psis[i-1])

        PsiSum=sum(Psis.values())
        LambdaPsiSum=Lambda.dot(PsiSum)
        Yresponse=eta.dot(LambdaPsiSum.T)
        delta=Yresponse.copy()

        eps=delta[instruments].copy()
        if self.params.residualize_delta:
            epsCoefsDict={eps.columns[0]:{eps.columns[0]:0.}}
            for i in range(1,len(instruments)):
                xx=eps.ix[:,list(range(0,i))]
                yy=eps.ix[:,i].copy()
                fit=pandas.ols(y=yy,x=xx,intercept=False)
                epsCoefsDict[eps.columns[i]]=fit.beta
                eps.ix[:,i]=fit.resid
            epsCoefs=np.eye(eps.shape[1]) + pandas.DataFrame(epsCoefsDict,
                    index=eps.columns,columns=eps.columns).T.fillna(0.)
            Q=np.linalg.pinv(epsCoefs)
        else:
            Q=np.eye(eps.shape[1])
        Hinv=np.dot(Q,LambdaPsiSum.loc[instruments].values)
        H=np.linalg.pinv(Hinv)
        
        LambdaH=pandas.DataFrame(np.dot(Lambda.values,H),index=Lambda.index,columns=instruments) #+ Lambda*0.
        folio=np.linalg.pinv(LambdaH.loc[marketSeriesIdxAvailable].copy().values
                ) + LambdaH.loc[marketSeriesIdxAvailable].T*0.

        
        monthlyFactorReturns=X[folio.columns].dot(((1./sigmaX[folio.columns])*(folio)).T).fillna(0.) #HACK:
        dailyFactorReturns=Xdaily[folio.columns].dot(((1./sigmaX[folio.columns])*(folio)).T).fillna(0.)


        #TODO: try adding back the missing factor as the residual of the eta regressed against the delta.  
        #Hack: Arbitrary scale
        dailyFactorReturnsNormalized=np.sqrt(1./252.)*(dailyFactorReturns/
                (dailyFactorReturns.iloc[-500:].std())).fillna(0.)
        results=Utilities.Struct()
        results.dailyFactorReturnsNormalized=dailyFactorReturnsNormalized
        results.X=X
        results.Xdaily=Xdaily
        results.fRetD=fRetD
        results.fRetM=fRetM

        return results

class MacroFactorReturnsCalculator_CoreSSM_base2(MacroFactorReturnCalculator_base):
    """...  """
    def __init__(self,params=None):
        MacroFactorReturnCalculator_base.__init__(self,params)
        if 'macroMetaCsvFile' not in self.params.getFieldNames():
            self.params.macroMetaCsvFile='macroconf/US3-MH-M-MetaData-rc-alt.csv'

        if 'f2i' not in self.params.getFieldNames():
            factorNames=['Economic Growth','Inflation','Confidence',]
            instruments=['ind_prod_total_idx', 'cpi_core', 'cci', ]
            self.params.f2i=dict(zip(instruments,factorNames))

        if 'macroSeriesIdx' not in self.params.getFieldNames():
            self.params.macroSeriesIdx=[
                    'unrate','civ_empl','emp_nonfarm',
                    'hours_nonfarm','wages_mfg','wages_nonfarm',
                    'ppi_core','ppi','cpi','cpi_core',
                    'treas_yield_20y_monthly','treas_yield_3m_monthly', 'broad_fx',
                    'leading_econ_idx','phil_outlook_svy','ism_pmi',
                    'pers_sav_pct','pers_inc','disp_pers_inc',
                    'nahb_housing_mkt_idx','ind_prod_total_idx','indprod_mfg', 'cci','chicago_pmbb',
                    'caputil',]
        """
        if 'macroSeriesIdx' not in self.params.getFieldNames():
            self.params.macroSeriesIdx=[
                    'unrate','civ_empl','emp_nonfarm',
                    'hours_nonfarm','wages_mfg','wages_nonfarm'
                    'ppi_core','ppi','cpi','cpi_core',
                    'treas_yield_20y_monthly','treas_yield_3m_monthly', 'broad_fx',
                    'leading_econ_idx','phil_outlook_svy','ism_pmi',
                    'pers_sav_pct','pers_inc','disp_pers_inc',
                    'nahb_housing_mkt_idx','ind_prod_total_idx','indprod_mfg', 'cci','chicago_pmbb',
                    'caputil',]
        """

        if 'f2scale' not in self.params.getFieldNames():
            self.params.f2scale={'Economic Growth':0.03,'Inflation':0.05,'Confidence':0.05,}

        if 'marketSeriesMonthlyEstuIdx' not in self.params.getFieldNames():
            self.params.marketSeriesMonthlyEstuIdx=[ 
                'Market Intercept','CAD', 'CHF', 'GBP', 'JPY', 'XDR','Market Sensitivity','Medium-Term Momentum','Size', 'Value', 'Volatility', 'sp_100_tr', 'sp_500_tr', 'sp_600_tr',
                'opec_oil_basket_price','gsci_nonenergy_tr', 'gsci_softs_spot', 'gsci_prec_metal_tr','russell_3000_div', 'russell_3000_value_div',
            ]
        
        if 'marketSeriesDailyEstuIdx' not in self.params.getFieldNames():
            self.params.marketSeriesDailyEstuIdx=[
            'crude_oil_wti', 'opec_oil_basket_price', 'reuters_cmdy_idx', 'gsci_energy_spot', 'gsci_gold_spot', 'gsci_ind_metals_spot', 'gsci_light_energy_spot', 'gsci_nonenergy_spot', 'gsci_non_prec_tr', 'gsci_prec_metal_spot', 'gsci_prec_metal_tr', 'gsci_silver_spot', 'gsci_ultralight_spot', 'gsci_energy_metals_spot', 'gsci_four_energy_er', 'gsci_softs_spot'
                    ]+[
            'russell_3000_div', 'russell_3000_value_div', 'sp_100_tr', 'sp_500_tr', 'sp_600_tr', 'sp_banking', 'sp_chemical', 'sp_sector_cons_staples', 'sp_sector_energy', 'sp_ew_index', 'sp_sector_financial', 'sp_global_1200_tr', 'sp_industrial_avg', 'sp_sector_industrials', 'sp_sector_it', 'sp_insurance', 'sp_sector_materials', 'sp_400_mc_sec_financials', 'sp_retail', 'sp_sector_telecom', 'sp_sector_utilities',
                    ]+[
            'CAD', 'CHF', 'GBP', 'JPY', 'XDR'
                    ]+[
                    'Market Intercept',        
                    ]+[
            'Dividend Yield', 'Exchange Rate Sensitivity', 'Growth', 'Leverage', 'Liquidity', 'Market Sensitivity', 'Medium-Term Momentum', 'Return-on-Equity', 'Size', 'Value', 'Volatility'
            ] # add multiples of "state variables", vol scaling,  etc...
                    
        self.params.marketDailyIdxs={}
        self.params.marketDailyIdxs['Economic Growth']=['Market Intercept','Dividend Yield','Medium-Term Momentum', 'Return-on-Equity', 'russell_3000_div', 'russell_3000_value_div','sp_sector_cons_staples', 'sp_sector_energy', 'sp_ew_index', 'sp_sector_financial', 'sp_global_1200_tr', 'sp_industrial_avg', 'sp_sector_industrials', 'sp_sector_it', 'sp_insurance', 'sp_sector_materials', 'sp_400_mc_sec_financials', 'sp_retail', 'sp_sector_telecom', 'sp_sector_utilities']
        self.params.marketDailyIdxs['Inflation']=['gsci_softs_spot','gsci_prec_metal_tr','gsci_nonenergy_spot','CAD', 'GBP', 'JPY', 'XDR']
        self.params.marketDailyIdxs['Confidence']=['CAD', 'CHF', 'GBP', 'JPY','sp_sector_financial','russell_3000_div','sp_insurance','sp_banking','sp_global_1200_tr','sp_100_tr','crude_oil_wti', 'gsci_gold_spot','Size', 'Value', 'Volatility']

    def _processCoreMacroData(self,modelData):
        logging.info('Begin: _processCoreMacroData')
        self.processTrivialCoreMacroFactors(modelData)

        fCumRetsDcoreTriv=modelData.fRetCalcResults.fCumRetsD_MacroCore.copy()
        fRetsDcoreTriv=modelData.fRetCalcResults.fRetsD_MacroCore.copy()
        #dates=list(fCumRetsDcore.index)
        dates=[datetime.datetime(d.year,d.month,d.day) for d in modelData.dates]

        factorNames=sorted(fCumRetsDcoreTriv.columns)
        instruments=sorted(self.params.f2i.values())
        i2f=dict( (v,k) for k,v in self.params.f2i.items())
        Nstatic=len(factorNames)
        if 'Nstatic' in self.params.getFieldNames():
            Nstatic=self.params.Nstatic

        metaAll=pandas.read_csv(self.params.macroMetaCsvFile)

        macroSeriesIdx=self.params.macroSeriesIdx   
 
        marketSeriesIdx=(list(modelData.master.marketInterceptReturns.columns)
                        +list(modelData.master.industryReturns.columns)
                        +list(modelData.master.styleReturns.columns)
                        +list(modelData.indexD['fx'])
                        +list(modelData.indexD['commodity']) #Careful with TR, etc
                        +list(modelData.indexD['benchmarks'])
                        #TODO: add bond yields and credit spreads.
                        #      add vix, etc...
                        )
        marketSeriesIdx=sorted(set(marketSeriesIdx).intersection(modelData.allCumRetsD.columns))
        logging.info('macroSeriesIdx  '+','.join(macroSeriesIdx))
        logging.info('marketSeriesIdx  '+','.join(marketSeriesIdx))

        fCumRetD=modelData.allCumRetsD[marketSeriesIdx].ix[dates].copy() 
        
        macroSeriesMonthly=modelData.macroSeriesMonthly.copy()

        results=self.extractCoreFactorReturns(instruments,fCumRetD,macroSeriesMonthly,
                                          marketSeriesIdx,macroSeriesIdx,
                                          metaAll,Lmax=4,ssm_iters=4)
        modelData.tmpResults=results

        fRetsDcore=results.dailyFactorReturnsNormalized.rename(columns=i2f).reindex(
                columns=factorNames).fillna(0.) 
        modelData.tmp_fRetsDcore=fRetsDcore
        fRetsDcore.ix[0]=0. #HACK: for good measure. will make essentially no difference
        for c in fRetsDcore.columns:
            fRetsDcore[c]*=self.params.f2scale.get(c,0.01)

        fCumRetsDcore=(1.+fRetsDcore).cumprod()
        modelData.fRetCalcResults.fCumRetsD_MacroCore=fCumRetsDcore
        modelData.fRetCalcResults.fRetsD_MacroCore=fRetsDcore
        logging.info('End: _processCoreMacroData')

    def extractCoreFactorReturns(self,instruments,fCumRetD,macroSeriesMonthly,
                             marketSeriesIdx,macroSeriesIdx,metaAll,Lmax=4,ssm_iters=4):
        meta=metaAll.ix[metaAll.active>0]
        meta=meta.rename(index=meta.shortname)
        metaM=meta[meta.freq=='M'].copy()
        metaD=meta[meta.freq=='D'].copy()

        geomIdxD=sorted(metaD[metaD.type=='geom'].index)
        percentIdxD=sorted(metaD[metaD.type=='percent'].index)

        geomIdxM=sorted(metaM[metaM.type=='geom'].index)
        geom2IdxM=sorted(metaM[metaM.type=='geom2'].index)
        percentIdxM=sorted(metaM[metaM.type=='percent'].index)
        linIdxM=sorted(metaM[metaM.type=='lin'].index)
        macroM=MacroUtils.d2dt(macroSeriesMonthly)
        tmp=macroM[geomIdxM].copy()
        macroM[tmp.columns]=tmp/(tmp.shift(1)) -1.
        tmp=macroM[geom2IdxM].copy()
        tmp=tmp/tmp.shift(1)-1.
        macroM[tmp.columns]=tmp-tmp.shift(1)
        tmp=macroM[percentIdxM].copy()
        macroM[tmp.columns]=tmp - (tmp.shift(1))
        macroM=macroM.ix[1:].copy()

        fCumRetD=MacroUtils.d2dt(fCumRetD).dropna(axis=1) # HACK: 
        fRetD=(fCumRetD/(fCumRetD.shift(1)) -1.)#.dropna()
        fCumRetDi=MacroUtils.d2dt(MacroUtils.expandDatesDF(fCumRetD))
        fCumRetM=fCumRetDi.ix[[d for d in fCumRetDi.index if d.day==1]]
        fRetM=(fCumRetM/fCumRetM.shift(1) -1.).ix[2:].copy()
        
        dMin=max(fRetM.index[0],macroM.index[0])
        dMax=fCumRetD.index[-1]
        
        X=macroM[macroSeriesIdx].ix[dMin:dMax].copy()
        X=X.join(fRetM)
        Xdaily=fRetD.copy()
        sigmaX=X.std()
        Y=X.copy()
        Y-=Y.mean()
        Y/=Y.std()
        Xnorm=X/Y.std()
        marketSeriesIdxAvailable=sorted(set(marketSeriesIdx).intersection(Xdaily.ix[dMax].dropna().index))
        
        noAR1CoeffIdx=list(marketSeriesIdx)
        feEstuIdx=[c for c in Y.columns if c in self.params.marketSeriesMonthlyEstuIdx]
        Nstatic=self.params.Nstatic

        fe=MacroUtils.FactorExtractorPCAar1(Y.dropna(),noAR1CoeffIdx) #HACK: 
        results=fe.estimate(Nstatic=Nstatic,Niters=ssm_iters,estu=feEstuIdx)
        instRetM=results.F.dot(results.Lambda[instruments])

        eta=results.eta
        Psi=results.Psi.T
        Lambda=results.Lambda.T
        F=results.F

        Psis={}
        Psis[0]=np.eye(Nstatic)+0.*Psi
        for i in range(1,Lmax):
            Psis[i]=Psi.dot(Psis[i-1])

        PsiSum=sum(Psis.values())
        LambdaPsiSum=Lambda.dot(PsiSum) # consiser using different sum/weight scheme for each inst
        Hinv=LambdaPsiSum.T[instruments].copy()
        Gamma=Hinv.copy()
        H=np.linalg.pinv(Hinv.values)
        LambdaH=pandas.DataFrame(np.dot(Lambda.values,H.T),index=Lambda.index,columns=instruments)
        delta=eta.dot(Hinv)
        eps=delta.copy()

        #folio=np.linalg.pinv(LambdaH.ix[marketSeriesIdxAvailable].copy().values) + LambdaH.ix[marketSeriesIdxAvailable].T*0.
        folios={}
        folio=  LambdaH.ix[marketSeriesIdxAvailable].T*0.
        for f,i in self.params.f2i.items():
            #mktSeriesIdx=[c for c in marketSeriesIdxAvailable if c in self.params.marketDailyIdxs[f]]
            mktSeriesIdx=[c for c in marketSeriesIdxAvailable if True ] #HACK: use everybody...
            LambdaTmp=Lambda.ix[mktSeriesIdx].copy()
            folios[f]=pandas.DataFrame(np.dot(Gamma.T.copy().values,np.linalg.pinv(LambdaTmp)).T,index=mktSeriesIdx,columns=Gamma.columns)
            folio.ix[i,mktSeriesIdx]=folios[f][i]


        monthlyFactorReturns=X[folio.columns].dot(((1./sigmaX[folio.columns])*(folio)).T).fillna(0.) #HACK:
        dailyFactorReturns=Xdaily[folio.columns].dot(((1./sigmaX[folio.columns])*(folio)).T).fillna(0.)


        """
        eps=delta[instruments].copy()
        if self.params.residualize_delta:
            epsCoefsDict={eps.columns[0]:{eps.columns[0]:0.}}
            for i in range(1,len(instruments)):
                xx=eps.ix[:,range(0,i)]
                yy=eps.ix[:,i].copy()
                fit=pandas.ols(y=yy,x=xx,intercept=False)
                epsCoefsDict[eps.columns[i]]=fit.beta
                eps.ix[:,i]=fit.resid
            epsCoefs=np.eye(eps.shape[1]) + pandas.DataFrame(epsCoefsDict,
                    index=eps.columns,columns=eps.columns).T.fillna(0.)
            Q=np.linalg.pinv(epsCoefs)
        else:
            Q=np.eye(eps.shape[1]) #should be the only case
        Q=np.eye(eps.shape[1]) 
        Hinv=np.dot(Q,LambdaPsiSum.ix[instruments].values) 
        H=np.linalg.pinv(Hinv)
        
        LambdaH=pandas.DataFrame(np.dot(Lambda.values,H),index=Lambda.index,columns=instruments) #+ Lambda*0.
        folio=np.linalg.pinv(LambdaH.ix[marketSeriesIdxAvailable].copy().values
                ) + LambdaH.ix[marketSeriesIdxAvailable].T*0.

        
        monthlyFactorReturns=X[folio.columns].dot(((1./sigmaX[folio.columns])*(folio)).T).fillna(0.) #HACK:
        dailyFactorReturns=Xdaily[folio.columns].dot(((1./sigmaX[folio.columns])*(folio)).T).fillna(0.)

        """

        #TODO: try adding back the missing factor as the residual of the eta regressed against the delta.  
        #Hack: Arbitrary scale
        dailyFactorReturnsNormalized=np.sqrt(1./252.)*(dailyFactorReturns/
                (dailyFactorReturns.ix[-500:].std())).fillna(0.)
        results=Utilities.Struct()
        results.dailyFactorReturnsNormalized=dailyFactorReturnsNormalized
        results.X=X
        results.Xdaily=Xdaily
        results.fRetD=fRetD
        results.fRetM=fRetM

        return results

class MacroFactorReturnsCalculator_CoreSSM_base(MacroFactorReturnCalculator_base):
    """...  """
    def __init__(self,params=None):
        MacroFactorReturnCalculator_base.__init__(self,params)
        assert 'macroMetaCsvFile' in self.params.getFieldNames(), 'Must set macroMetaCsfVile in riskModel'

        if 'residualize_delta' not in self.params.getFieldNames():
            self.params.residualize_delta=True
        if 'f2i' not in self.params.getFieldNames():
            factorNames=['Economic Growth','Inflation','Confidence','Unemployment']
            instruments=['ind_prod_total_idx', 'cpi', 'cci', 'civ_empl']
            self.params.f2i=dict(zip(instruments,factorNames))

        if 'macroSeriesIdx' not in self.params.getFieldNames():
            self.params.macroSeriesIdx=['unrate','unrate_16_plus','treas_yield_20y_monthly','treas_yield_3m_monthly',
            'broad_fx','civ_empl','leading_econ_idx','ppi_core','ppi','phil_outlook_svy','pers_sav_pct',
            'pers_inc','nahb_housing_mkt_idx','ism_pmi','ind_prod_total_idx','indprod_mfg','emp_nonfarm',
            'disp_pers_inc','cpi','cpi_core','cci','chicago_pmbb','caputil',
            'hours_nonfarm','wages_mfg','wages_nonfarm']+['pce'] #This one has problems in the db.
        
        if 'macroSeriesTransform' not in self.params.getFieldNames():
            self.params.macroSeriesTransform={
                    'ind_prod_total_idx':{'name':'ind_prod_lead','shift':-1,'diff':1,'geom':True},
                    'cpi':               {'name':'cpi_d2',       'shift':0,'diff':2,'geom':True},
                    'cpi_core':          {'name':'cpi_core_d2',  'shift':0,'diff':2,'geom':True},
                    'ppi':               {'name':'ppi_d2',       'shift':0,'diff':2,'geom':True},
                    'ppi_core':          {'name':'ppi_core_d2',  'shift':0,'diff':2,'geom':True},
                    }
                    
        if 'f2scale' not in self.params.getFieldNames():
            #self.params.f2scale={'Economic Growth':0.03,'Inflation':0.01,'Confidence':0.05,}
            self.params.f2scale={'Economic Growth':0.03,'Inflation':0.05,'Confidence':0.05,}

    def _processCoreMacroData(self,modelData):
        logging.info('Begin: _processCoreMacroData')
        self.processTrivialCoreMacroFactors(modelData)

        fCumRetsDcoreTriv=modelData.fRetCalcResults.fCumRetsD_MacroCore.copy()
        fRetsDcoreTriv=modelData.fRetCalcResults.fRetsD_MacroCore.copy()
        #dates=list(fCumRetsDcore.index)
        dates=[datetime.datetime(d.year,d.month,d.day) for d in modelData.dates]

        factorNames=sorted(fCumRetsDcoreTriv.columns)
        instruments=sorted(self.params.f2i.values())
        i2f=dict( (v,k) for k,v in self.params.f2i.items())
        Nstatic=len(factorNames)
        if 'Nstatic' in self.params.getFieldNames():
            Nstatic=self.params.Nstatic

        metaAll=pandas.read_csv(self.params.macroMetaCsvFile)

        macroSeriesIdx=self.params.macroSeriesIdx   
 
        marketSeriesIdx=(list(modelData.master.marketInterceptReturns.columns)
                        +list(modelData.master.industryReturns.columns)
                        +list(modelData.master.styleReturns.columns)
                        +list(modelData.indexD['fx'])
                        +list(modelData.indexD['commodity']) #Careful with TR, etc
                        +list(modelData.indexD['benchmarks'])
                        #TODO: add bond yields and credit spreads.
                        #      add vix, etc...
                        )
        marketSeriesIdx=sorted(set(marketSeriesIdx).intersection(modelData.allCumRetsD.columns))
        logging.info('macroSeriesIdx  '+','.join(macroSeriesIdx))
        logging.info('marketSeriesIdx  '+','.join(marketSeriesIdx))

        fCumRetD=modelData.allCumRetsD[marketSeriesIdx].ix[dates].copy() 
        
        extraMacroSeries={}
        macroSeriesMonthly=modelData.macroSeriesMonthly.copy()
        for k,v in self.params.macroSeriesTransform.items():
            tmpSeries=macroSeriesMonthly[k].copy().shift(v['shift'])
            tmpSeries.name=v['name']
            
            if v['geom']:
                tmpSeries/=tmpSeries.ix[0]
                tmpSeries=np.log(tmpSeries)
            for i in range(v['diff']):
                tmpSeries-=tmpSeries.shift(1)
            macroSeriesMonthly=macroSeriesMonthly.join(tmpSeries)
            extraMacroSeries[tmpSeries.name]=tmpSeries.copy()

        macroSeriesMonthly=macroSeriesMonthly.ix[dates[0]:dates[-1]].copy()

        results=self.extractCoreFactorReturns(instruments,fCumRetD,macroSeriesMonthly,
                                          marketSeriesIdx,macroSeriesIdx+list(extraMacroSeries.keys()),metaAll,Lmax=4,ssm_iters=4)
        modelData.tmpResults=results

        fRetsDcore=results.dailyFactorReturnsNormalized.rename(columns=i2f).reindex(columns=factorNames).fillna(0.) 
        modelData.tmp_fRetsDcore=fRetsDcore
        fRetsDcore.ix[0]=0. #HACK: for good measure. will make essentially no difference
        for c in fRetsDcore.columns:
            fRetsDcore[c]*=self.params.f2scale.get(c,0.01)

        fCumRetsDcore=(1.+fRetsDcore).cumprod()
        modelData.fRetCalcResults.fCumRetsD_MacroCore=fCumRetsDcore
        modelData.fRetCalcResults.fRetsD_MacroCore=fRetsDcore
        logging.info('End: _processCoreMacroData')

    def extractCoreFactorReturns(self,instruments,fCumRetD,macroSeriesMonthly,
                             marketSeriesIdx,macroSeriesIdx,metaAll,Lmax=4,ssm_iters=4):
        meta=metaAll.ix[metaAll.active>0]
        meta=meta.rename(index=meta.shortname)
        metaM=meta[meta.freq=='M'].copy()
        metaD=meta[meta.freq=='D'].copy()

        geomIdxD=sorted(metaD[metaD.type=='geom'].index)
        percentIdxD=sorted(metaD[metaD.type=='percent'].index)

        geomIdxM=sorted(metaM[metaM.type=='geom'].index)
        percentIdxM=sorted(metaM[metaM.type=='percent'].index)
        linIdxM=sorted(metaM[metaM.type=='lin'].index)

        macroM=MacroUtils.d2dt(macroSeriesMonthly)
        tmp=macroM[geomIdxM].copy()
        macroM[tmp.columns]=tmp/(tmp.shift(1)) -1.
        tmp=macroM[percentIdxM].copy()
        macroM[tmp.columns]=tmp - (tmp.shift(1))
        macroM=macroM.ix[1:].copy()

        fCumRetD=MacroUtils.d2dt(fCumRetD).dropna(axis=1) # HACK: crude to dropna.needed if we orth returns here  
        fRetD=(fCumRetD/(fCumRetD.shift(1)) -1.)#.dropna()
        fCumRetDi=MacroUtils.d2dt(MacroUtils.expandDatesDF(fCumRetD))
        fCumRetM=fCumRetDi.ix[[d for d in fCumRetDi.index if d.day==1]]
        fRetM=(fCumRetM/fCumRetM.shift(1) -1.).ix[1:].copy()
        
        dMin=max(fRetM.index[0],macroM.index[0])
        dMax=fCumRetD.index[-1]
        
        X=macroM[macroSeriesIdx].ix[dMin:dMax].copy()
        X=X.join(fRetM)
        Xdaily=fRetD.copy()
        sigmaX=X.std()
        Y=X.copy()
        Y-=Y.mean()
        Y/=Y.std()
        Xnorm=X/Y.std()
        marketSeriesIdxAvailable=sorted(set(marketSeriesIdx).intersection(Xdaily.ix[dMax].dropna().index))
        
        noAR1CoeffIdx=list(marketSeriesIdx)+list(self.params.macroSeriesTransform.keys())

        Nstatic=self.params.Nstatic
                
        fe=MacroUtils.FactorExtractorPCAar1(Y.dropna(),noAR1CoeffIdx) #HACK: 
        results=fe.estimate(Nstatic=Nstatic,Niters=ssm_iters)
        instRetM=results.F.dot(results.Lambda[instruments])

        eta=results.eta
        Psi=results.Psi.T
        Lambda=results.Lambda.T
        F=results.F

        Psis={}
        Psis[0]=np.eye(Nstatic)+0.*Psi
        for i in range(1,Lmax):
            Psis[i]=Psi.dot(Psis[i-1])

        PsiSum=sum(Psis.values())
        LambdaPsiSum=Lambda.dot(PsiSum)
        Yresponse=eta.dot(LambdaPsiSum.T)
        delta=Yresponse.copy()

        eps=delta[instruments].copy()
        if self.params.residualize_delta:
            epsCoefsDict={eps.columns[0]:{eps.columns[0]:0.}}
            for i in range(1,len(instruments)):
                xx=eps.ix[:,list(range(0,i))]
                yy=eps.ix[:,i].copy()
                fit=pandas.ols(y=yy,x=xx,intercept=False)
                epsCoefsDict[eps.columns[i]]=fit.beta
                eps.ix[:,i]=fit.resid
            epsCoefs=np.eye(eps.shape[1]) + pandas.DataFrame(epsCoefsDict,
                    index=eps.columns,columns=eps.columns).T.fillna(0.)
            Q=np.linalg.pinv(epsCoefs)
        else:
            Q=np.eye(eps.shape[1])
        Hinv=np.dot(Q,LambdaPsiSum.ix[instruments].values)
        H=np.linalg.pinv(Hinv)
        
        LambdaH=pandas.DataFrame(np.dot(Lambda.values,H),index=Lambda.index,columns=instruments) #+ Lambda*0.
        folio=np.linalg.pinv(LambdaH.ix[marketSeriesIdxAvailable].copy().values
                ) + LambdaH.ix[marketSeriesIdxAvailable].T*0.

        
        monthlyFactorReturns=X[folio.columns].dot(((1./sigmaX[folio.columns])*(folio)).T).fillna(0.) #HACK:
        dailyFactorReturns=Xdaily[folio.columns].dot(((1./sigmaX[folio.columns])*(folio)).T).fillna(0.)


        #TODO: try adding back the missing factor as the residual of the eta regressed against the delta.  
        #Hack: Arbitrary scale
        dailyFactorReturnsNormalized=np.sqrt(1./252.)*(dailyFactorReturns/
                (dailyFactorReturns.ix[-500:].std())).fillna(0.)
        results=Utilities.Struct()
        results.dailyFactorReturnsNormalized=dailyFactorReturnsNormalized
        results.X=X
        results.Xdaily=Xdaily
        results.fRetD=fRetD
        results.fRetM=fRetM

        return results

class MacroFactorReturnsCalculator_CoreSSM_base3(MacroFactorReturnCalculator_base):
    """...  """
    def __init__(self,params=None):
        MacroFactorReturnCalculator_base.__init__(self,params)
        if 'macroMetaCsvFile' not in self.params.getFieldNames():
            self.params.macroMetaCsvFile='macroconf/US3-MH-M-MetaData-rc-alt.csv'

        if 'f2i' not in self.params.getFieldNames():
            factorNames=['Economic Growth','Inflation','Confidence',]
            instruments=['ind_prod_total_idx', 'cpi_core', 'cci', ]
            self.params.f2i=dict(zip(instruments,factorNames))

        if 'macroSeriesIdx' not in self.params.getFieldNames():
            self.params.macroSeriesIdx=[
                    'unrate','civ_empl','emp_nonfarm',
                    'hours_nonfarm','wages_mfg','wages_nonfarm',
                    'ppi_core','ppi','cpi','cpi_core',
                    'treas_yield_20y_monthly','treas_yield_3m_monthly', 'broad_fx',
                    'leading_econ_idx','phil_outlook_svy','ism_pmi',
                    'pers_sav_pct','pers_inc','disp_pers_inc',
                    'nahb_housing_mkt_idx','ind_prod_total_idx','indprod_mfg', 'cci','chicago_pmbb',
                    'caputil',]
        """
        if 'macroSeriesIdx' not in self.params.getFieldNames():
            self.params.macroSeriesIdx=[
                    'unrate','civ_empl','emp_nonfarm',
                    'hours_nonfarm','wages_mfg','wages_nonfarm'
                    'ppi_core','ppi','cpi','cpi_core',
                    'treas_yield_20y_monthly','treas_yield_3m_monthly', 'broad_fx',
                    'leading_econ_idx','phil_outlook_svy','ism_pmi',
                    'pers_sav_pct','pers_inc','disp_pers_inc',
                    'nahb_housing_mkt_idx','ind_prod_total_idx','indprod_mfg', 'cci','chicago_pmbb',
                    'caputil',]
        """

        if 'f2scale' not in self.params.getFieldNames():
            self.params.f2scale={'Economic Growth':0.03,'Inflation':0.05,'Confidence':0.05,}

        if 'marketSeriesMonthlyEstuIdx' not in self.params.getFieldNames():
            self.params.marketSeriesMonthlyEstuIdx=[ 
                'Market Intercept','CAD', 'CHF', 'GBP', 'JPY', 'XDR','Market Sensitivity','Medium-Term Momentum','Size', 'Value', 'Volatility', 'sp_100_tr', 'sp_500_tr', 'sp_600_tr',
                'opec_oil_basket_price','gsci_nonenergy_tr', 'gsci_softs_spot', 'gsci_prec_metal_tr','russell_3000_div', 'russell_3000_value_div',
            ]
        
        if 'marketSeriesDailyEstuIdx' not in self.params.getFieldNames():
            self.params.marketSeriesDailyEstuIdx=[
            'crude_oil_wti', 'opec_oil_basket_price', 'reuters_cmdy_idx', 'gsci_energy_spot', 'gsci_gold_spot', 'gsci_ind_metals_spot', 'gsci_light_energy_spot', 'gsci_nonenergy_spot', 'gsci_non_prec_tr', 'gsci_prec_metal_spot', 'gsci_prec_metal_tr', 'gsci_silver_spot', 'gsci_ultralight_spot', 'gsci_energy_metals_spot', 'gsci_four_energy_er', 'gsci_softs_spot'
                    ]+[
            'russell_3000_div', 'russell_3000_value_div', 'sp_100_tr', 'sp_500_tr', 'sp_600_tr', 'sp_banking', 'sp_chemical', 'sp_sector_cons_staples', 'sp_sector_energy', 'sp_ew_index', 'sp_sector_financial', 'sp_global_1200_tr', 'sp_industrial_avg', 'sp_sector_industrials', 'sp_sector_it', 'sp_insurance', 'sp_sector_materials', 'sp_400_mc_sec_financials', 'sp_retail', 'sp_sector_telecom', 'sp_sector_utilities',
                    ]+[
            'CAD', 'CHF', 'GBP', 'JPY', 'XDR'
                    ]+[
                    'Market Intercept',        
                    ]+[
            'Dividend Yield', 'Exchange Rate Sensitivity', 'Growth', 'Leverage', 'Liquidity', 'Market Sensitivity', 'Medium-Term Momentum', 'Return-on-Equity', 'Size', 'Value', 'Volatility'
            ] # add multiples of "state variables", vol scaling,  etc...
                    
        self.params.marketDailyIdxs={}
        self.params.marketDailyIdxs['Economic Growth']=['Market Intercept','Dividend Yield','Medium-Term Momentum', 'Return-on-Equity', 'russell_3000_div', 'russell_3000_value_div','sp_sector_cons_staples', 'sp_sector_energy', 'sp_ew_index', 'sp_sector_financial', 'sp_global_1200_tr', 'sp_industrial_avg', 'sp_sector_industrials', 'sp_sector_it', 'sp_insurance', 'sp_sector_materials', 'sp_400_mc_sec_financials', 'sp_retail', 'sp_sector_telecom', 'sp_sector_utilities']
        self.params.marketDailyIdxs['Inflation']=['gsci_softs_spot','gsci_prec_metal_tr','gsci_nonenergy_spot','CAD', 'GBP', 'JPY', 'XDR']
        self.params.marketDailyIdxs['Confidence']=['CAD', 'CHF', 'GBP', 'JPY','sp_sector_financial','russell_3000_div','sp_insurance','sp_banking','sp_global_1200_tr','sp_100_tr','crude_oil_wti', 'gsci_gold_spot','Size', 'Value', 'Volatility']

    def _processCoreMacroData(self,modelData):
        logging.info('Begin: _processCoreMacroData')
        self.processTrivialCoreMacroFactors(modelData)

        fCumRetsDcoreTriv=modelData.fRetCalcResults.fCumRetsD_MacroCore.copy()
        fRetsDcoreTriv=modelData.fRetCalcResults.fRetsD_MacroCore.copy()
        #dates=list(fCumRetsDcore.index)
        dates=[datetime.datetime(d.year,d.month,d.day) for d in modelData.dates]

        factorNames=sorted(fCumRetsDcoreTriv.columns)
        instruments=sorted(self.params.f2i.values())
        i2f=dict( (v,k) for k,v in self.params.f2i.items())
        Nstatic=len(factorNames)
        if 'Nstatic' in self.params.getFieldNames():
            Nstatic=self.params.Nstatic

        metaAll=pandas.read_csv(self.params.macroMetaCsvFile)

        macroSeriesIdx=self.params.macroSeriesIdx   
 
        marketSeriesIdx=(list(modelData.master.marketInterceptReturns.columns)
                        +list(modelData.master.industryReturns.columns)
                        +list(modelData.master.styleReturns.columns)
                        +list(modelData.indexD['fx'])
                        +list(modelData.indexD['commodity']) #Careful with TR, etc
                        +list(modelData.indexD['benchmarks'])
                        #TODO: add bond yields and credit spreads.
                        #      add vix, etc...
                        )
        marketSeriesIdx=sorted(set(marketSeriesIdx).intersection(modelData.allCumRetsD.columns))
        logging.info('macroSeriesIdx  '+','.join(macroSeriesIdx))
        logging.info('marketSeriesIdx  '+','.join(marketSeriesIdx))

        fCumRetD=modelData.allCumRetsD[marketSeriesIdx].ix[dates].copy() 
        
        macroSeriesMonthly=modelData.macroSeriesMonthly.copy()

        results=self.extractCoreFactorReturns(instruments,fCumRetD,macroSeriesMonthly,
                                          marketSeriesIdx,macroSeriesIdx,
                                          metaAll,Lmax=4,ssm_iters=4,modelData=modelData)
        modelData.tmpResults=results

        fRetsDcore=results.dailyFactorReturnsNormalized.rename(columns=i2f).reindex(
                columns=factorNames).fillna(0.) 
        modelData.tmp_fRetsDcore=fRetsDcore
        fRetsDcore.ix[0]=0. #HACK: for good measure. will make essentially no difference
        for c in fRetsDcore.columns:
            fRetsDcore[c]*=self.params.f2scale.get(c,0.01)

        fCumRetsDcore=(1.+fRetsDcore).cumprod()
        modelData.fRetCalcResults.fCumRetsD_MacroCore=fCumRetsDcore
        modelData.fRetCalcResults.fRetsD_MacroCore=fRetsDcore
        logging.info('End: _processCoreMacroData')

    def extractCoreFactorReturns(self,instruments,fCumRetD,macroSeriesMonthly,
                             marketSeriesIdx,macroSeriesIdx,metaAll,Lmax=4,ssm_iters=4,modelData=None):
        meta=metaAll.ix[metaAll.active>0]
        meta=meta.rename(index=meta.shortname)
        metaM=meta[meta.freq=='M'].copy()
        metaD=meta[meta.freq=='D'].copy()

        geomIdxD=sorted(metaD[metaD.type=='geom'].index)
        percentIdxD=sorted(metaD[metaD.type=='percent'].index)

        geomIdxM=sorted(metaM[metaM.type=='geom'].index)
        geom2IdxM=sorted(metaM[metaM.type=='geom2'].index)
        percentIdxM=sorted(metaM[metaM.type=='percent'].index)
        linIdxM=sorted(metaM[metaM.type=='lin'].index)

        macroM=MacroUtils.d2dt(macroSeriesMonthly)
        tmp=macroM[geomIdxM].copy()
        macroM[tmp.columns]=tmp/(tmp.shift(1)) -1.
        tmp=macroM[geom2IdxM].copy()
        tmp=tmp/tmp.shift(1)-1.
        macroM[tmp.columns]=tmp-tmp.shift(1)
        tmp=macroM[percentIdxM].copy()
        macroM[tmp.columns]=tmp - (tmp.shift(1))
        macroM=macroM.ix[1:].copy()

        fCumRetD=MacroUtils.d2dt(fCumRetD).dropna(axis=1) # HACK: 
        fRetD=(fCumRetD/(fCumRetD.shift(1)) -1.)#.dropna()
        fCumRetDi=MacroUtils.d2dt(MacroUtils.expandDatesDF(fCumRetD))
        fCumRetM=fCumRetDi.ix[[d for d in fCumRetDi.index if d.day==1]]
        fRetM=(fCumRetM/fCumRetM.shift(1) -1.).ix[2:].copy()
        
        dMin=max(fRetM.index[0],macroM.index[0])
        dMax=fCumRetD.index[-1]

        if modelData is not None and modelData.master.excessReturnsDF_fac_ret is not None:
            aRetD=modelData.master.excessReturnsDF_fac_ret
            aRetD=aRetD.ix[:,np.abs(aRetD).max()<0.25].copy() #TODO: refine this
            aCumRetD=MacroUtils.expandDatesDF((1.+aRetD).cumprod())
            aCumRetM=aCumRetD.ix[ [d for d in aCumRetD.index if d.day==1]].copy()
            aRetM=(aCumRetM/aCumRetM.shift(1) -1).dropna()

        X=macroM[macroSeriesIdx].ix[dMin:dMax].copy()
        X=X.join(fRetM)
        Xdaily=fRetD.copy()
        sigmaX=X.std()
        Y=X.copy()
        Y-=Y.mean()
        Y/=Y.std()
        Xnorm=X/Y.std()
        marketSeriesIdxAvailable=sorted(set(marketSeriesIdx).intersection(Xdaily.ix[dMax].dropna().index))
        
        noAR1CoeffIdx=list(marketSeriesIdx)
        feEstuIdx=[c for c in Y.columns if c in self.params.marketSeriesMonthlyEstuIdx]
        Nstatic=self.params.Nstatic

        fe=MacroUtils.FactorExtractorPCAar1(Y.dropna(),noAR1CoeffIdx) #HACK: 
        results=fe.estimate(Nstatic=Nstatic,Niters=ssm_iters,estu=feEstuIdx)
        instRetM=results.F.dot(results.Lambda[instruments])

        eta=results.eta
        Psi=results.Psi.T
        Lambda=results.Lambda.T
        F=results.F

        assert modelData is not None and modelData.master.excessReturnsDF_fac_ret is not None
        aDates=sorted(aRetM.index.intersection(F.index))
        aLambdaResults=MacroUtils.LinearModel(aRetM.ix[aDates],F.ix[aDates])
        aLambda=aLambdaResults.beta
        Lambda2=Lambda.T.join(aLambda.T).T.copy()

        Psis={}
        Psis[0]=np.eye(Nstatic)+0.*Psi
        for i in range(1,Lmax):
            Psis[i]=Psi.dot(Psis[i-1])

        PsiSum=sum(Psis.values())
        LambdaPsiSum=Lambda.dot(PsiSum) # consiser using different sum/weight scheme for each inst
        Hinv=LambdaPsiSum.T[instruments].copy()
        Gamma=Hinv.copy()
        H=np.linalg.pinv(Hinv.values)
        LambdaH=pandas.DataFrame(np.dot(Lambda.values,H.T),index=Lambda.index,columns=instruments)
        delta=eta.dot(Hinv)
        eps=delta.copy()

        """
        folios={}
        folio=  LambdaH.ix[marketSeriesIdxAvailable].T*0.
        for f,i in self.params.f2i.iteritems():
            mktSeriesIdx=aLambda.index
            LambdaTmp=Lambda2.ix[mktSeriesIdx].copy()
            folios[f]=pandas.DataFrame(np.dot(Gamma.T.copy().values,np.linalg.pinv(LambdaTmp)).T,index=mktSeriesIdx,columns=Gamma.columns)
            folio.ix[i,mktSeriesIdx]=folios[f][i] #NOTE: this is redundant for now.
        """
        mktSeriesIdx=aLambda.index
        LambdaTmp=Lambda2.ix[mktSeriesIdx].copy()
        folio=pandas.DataFrame(np.dot(Gamma.T.copy().values,np.linalg.pinv(LambdaTmp)).T,index=mktSeriesIdx,columns=Gamma.columns)

        monthlyFactorReturns=aRetM.dot(folio)
        dailyFactorReturns=aRetD.dot(folio)
        #monthlyFactorReturns=X[folio.columns].dot(((1./sigmaX[folio.columns])*(folio)).T).fillna(0.) #HACK:
        #dailyFactorReturns=Xdaily[folio.columns].dot(((1./sigmaX[folio.columns])*(folio)).T).fillna(0.)


        """
        eps=delta[instruments].copy()
        if self.params.residualize_delta:
            epsCoefsDict={eps.columns[0]:{eps.columns[0]:0.}}
            for i in range(1,len(instruments)):
                xx=eps.ix[:,range(0,i)]
                yy=eps.ix[:,i].copy()
                fit=pandas.ols(y=yy,x=xx,intercept=False)
                epsCoefsDict[eps.columns[i]]=fit.beta
                eps.ix[:,i]=fit.resid
            epsCoefs=np.eye(eps.shape[1]) + pandas.DataFrame(epsCoefsDict,
                    index=eps.columns,columns=eps.columns).T.fillna(0.)
            Q=np.linalg.pinv(epsCoefs)
        else:
            Q=np.eye(eps.shape[1]) #should be the only case
        Q=np.eye(eps.shape[1]) 
        Hinv=np.dot(Q,LambdaPsiSum.ix[instruments].values) 
        H=np.linalg.pinv(Hinv)
        
        LambdaH=pandas.DataFrame(np.dot(Lambda.values,H),index=Lambda.index,columns=instruments) #+ Lambda*0.
        folio=np.linalg.pinv(LambdaH.ix[marketSeriesIdxAvailable].copy().values
                ) + LambdaH.ix[marketSeriesIdxAvailable].T*0.

        
        monthlyFactorReturns=X[folio.columns].dot(((1./sigmaX[folio.columns])*(folio)).T).fillna(0.) #HACK:
        dailyFactorReturns=Xdaily[folio.columns].dot(((1./sigmaX[folio.columns])*(folio)).T).fillna(0.)

        """

        dailyFactorReturnsNormalized=np.sqrt(1./252.)*(dailyFactorReturns/
                (dailyFactorReturns.ix[-500:].std())).fillna(0.)
        results=Utilities.Struct()
        results.dailyFactorReturnsNormalized=dailyFactorReturnsNormalized
        results.X=X
        results.Xdaily=Xdaily
        results.fRetD=fRetD
        results.fRetM=fRetM

        return results

class MacroFactorReturnsCalculator_CoreSSM_base_bareV5_1(MacroFactorReturnCalculator_base):
    """...  """
    def __init__(self,params=None):
        MacroFactorReturnCalculator_base.__init__(self,params)
        assert 'macroMetaCsvFile' in self.params.getFieldNames(), 'Must set macroMetaCsfVile in riskModel'

        if 'residualize_delta' not in self.params.getFieldNames():
            self.params.residualize_delta=True

        if 'macroSeriesIdx' not in self.params.getFieldNames():
            self.params.macroSeriesIdx=['unrate','unrate_16_plus','treas_yield_20y_monthly','treas_yield_3m_monthly',
            'broad_fx','civ_empl','leading_econ_idx','ppi_core','ppi','phil_outlook_svy','pers_sav_pct',
            'pers_inc','nahb_housing_mkt_idx','ism_pmi','ind_prod_total_idx','indprod_mfg','emp_nonfarm',
            'disp_pers_inc','cpi','cpi_core','cci','chicago_pmbb','caputil',
            'hours_nonfarm','wages_mfg','wages_nonfarm']+['pce'] #This one has problems in the db.
        
        if 'macroSeriesTransform' not in self.params.getFieldNames():
            self.params.macroSeriesTransform={
                    'ind_prod_total_idx':{'name':'ind_prod_lead','shift':-1,'diff':1,'geom':True},
                    'cpi':               {'name':'cpi_d2',       'shift':0,'diff':2,'geom':True},
                    'cpi_core':          {'name':'cpi_core_d2',  'shift':0,'diff':2,'geom':True},
                    'ppi':               {'name':'ppi_d2',       'shift':0,'diff':2,'geom':True},
                    'ppi_core':          {'name':'ppi_core_d2',  'shift':0,'diff':2,'geom':True},
                    }
                    
        if 'f2scale' not in self.params.getFieldNames():
            self.params.f2scale={'Economic Growth':0.03,'Inflation':0.05,'Confidence':0.05,}


    def _processCoreMacroData(self,modelData):
        logging.info('Begin: _processCoreMacroData')
        self.processTrivialCoreMacroFactors(modelData)

        fCumRetsDcoreTriv=modelData.fRetCalcResults.fCumRetsD_MacroCore.copy()
        fRetsDcoreTriv=modelData.fRetCalcResults.fRetsD_MacroCore.copy()
        #dates=list(fCumRetsDcore.index)
        dates=[datetime.datetime(d.year,d.month,d.day) for d in modelData.dates]

        factorNames=sorted(fCumRetsDcoreTriv.columns)
        instruments=sorted(self.params.f2i.values())
        i2f=dict( (v,k) for k,v in self.params.f2i.items())
        Nstatic=len(factorNames)
        if 'Nstatic' in self.params.getFieldNames():
            Nstatic=self.params.Nstatic

        metaAll=pandas.read_csv(self.params.macroMetaCsvFile)

        macroSeriesIdx=self.params.macroSeriesIdx   

        marketSeriesIdx=([]
                        #+list(modelData.master.marketInterceptReturns.columns)
                        #+list(industryCols)
                        +list(modelData.master.styleReturns.columns)
                        +list(modelData.indexD['fx'])
                        +list(modelData.indexD['commodity']) #Careful with TR, etc
                        )
        marketSeriesIdx=sorted(set(marketSeriesIdx).intersection(modelData.allCumRetsD.columns))
        logging.info('macroSeriesIdx  '+','.join(macroSeriesIdx))
        logging.info('marketSeriesIdx  '+','.join(marketSeriesIdx))

        fCumRetD=modelData.allCumRetsD[marketSeriesIdx].ix[dates].copy() 

        marketInterceptRetD=modelData.master.marketInterceptReturns['Market Intercept']
        secActiveRetD=self.params.activeSectorReturns.ix[dates].fillna(0.).rename(columns=lambda c:c+'_active')
        secTotalRetD=secActiveRetD.sub(marketInterceptRetD,axis=0).rename(columns=lambda c:c+'_total')
        
        secActiveCumRetD=(1.+secActiveRetD).cumprod()
        secTotalCumRetD=(1.+secTotalRetD).cumprod()
        marketInterceptCumRetD=(1.+marketInterceptRetD).cumprod()
        marketInterceptCumRetD.name='US3--Market Intercept'
        
        fCumRetDTmp=fCumRetD.join(secActiveCumRetD).join(secTotalCumRetD).join(marketInterceptCumRetD)
        marketSeriesIdx=marketSeriesIdx+sorted(secTotalCumRetD.columns ) # THIS IS WHERE WE CHANGE THINGS.
        fCumRetD=fCumRetDTmp[marketSeriesIdx].copy().fillna(method='bfill') #NOTE Not sure here why fillna needed.

        extraMacroSeries={}
        macroSeriesMonthly=modelData.macroSeriesMonthly.copy()
        for k,v in self.params.macroSeriesTransform.items():
            tmpSeries=macroSeriesMonthly[k].copy().shift(v['shift'])
            tmpSeries.name=v['name']
            
            if v['geom']:
                tmpSeries/=tmpSeries.ix[0]
                tmpSeries=np.log(tmpSeries)
            for i in range(v['diff']):
                tmpSeries-=tmpSeries.shift(1)
            macroSeriesMonthly=macroSeriesMonthly.join(tmpSeries)
            extraMacroSeries[tmpSeries.name]=tmpSeries.copy()

        macroSeriesMonthly=macroSeriesMonthly.ix[dates[0]:dates[-1]].copy()

        results=self.extractCoreFactorReturns(instruments,fCumRetD,macroSeriesMonthly,
                                          marketSeriesIdx,macroSeriesIdx+list(extraMacroSeries.keys()),metaAll,Lmax=4,ssm_iters=4)
        modelData.tmpResults=results

        fRetsDcore=results.dailyFactorReturnsNormalized.rename(columns=i2f).reindex(columns=factorNames).fillna(0.) 
        modelData.tmp_fRetsDcore=fRetsDcore
        fRetsDcore.ix[0]=0. #HACK: for good measure. will make essentially no difference
        for c in fRetsDcore.columns:
            fRetsDcore[c]*=self.params.f2scale.get(c,0.01)

        fCumRetsDcore=(1.+fRetsDcore).cumprod()
        modelData.fRetCalcResults.fCumRetsD_MacroCore=fCumRetsDcore
        modelData.fRetCalcResults.fRetsD_MacroCore=fRetsDcore
        logging.info('End: _processCoreMacroData')

    def extractCoreFactorReturns(self,instruments,fCumRetD,macroSeriesMonthly,
                             marketSeriesIdx,macroSeriesIdx,metaAll,Lmax=4,ssm_iters=4):
        meta=metaAll.ix[metaAll.active>0]
        meta=meta.rename(index=meta.shortname)
        metaM=meta[meta.freq=='M'].copy()
        metaD=meta[meta.freq=='D'].copy()

        geomIdxD=sorted(metaD[metaD.type=='geom'].index)
        percentIdxD=sorted(metaD[metaD.type=='percent'].index)

        geomIdxM=sorted(metaM[metaM.type=='geom'].index)
        percentIdxM=sorted(metaM[metaM.type=='percent'].index)
        linIdxM=sorted(metaM[metaM.type=='lin'].index)
        macroM=MacroUtils.d2dt(macroSeriesMonthly)
        tmp=macroM[geomIdxM].copy()
        macroM[tmp.columns]=tmp/(tmp.shift(1)) -1.
        tmp=macroM[percentIdxM].copy()
        macroM[tmp.columns]=tmp - (tmp.shift(1))
        macroM=macroM.ix[1:].copy()

        fCumRetD=MacroUtils.d2dt(fCumRetD).dropna(axis=1) # HACK: crude to dropna.needed if we orth returns here  
        fRetD=(fCumRetD/(fCumRetD.shift(1)) -1.)#.dropna()
        fCumRetDi=MacroUtils.d2dt(MacroUtils.expandDatesDF(fCumRetD))
        fCumRetM=fCumRetDi.ix[[d for d in fCumRetDi.index if d.day==1]]
        fRetM=(fCumRetM/fCumRetM.shift(1) -1.).ix[1:].copy()
        
        dMin=max(fRetM.index[0],macroM.index[0])
        dMax=fCumRetD.index[-1]
        
        X=macroM[macroSeriesIdx].ix[dMin:dMax].copy()
        X=X.join(fRetM)
        Xdaily=fRetD.copy()
        sigmaX=X.std()
        Y=X.copy()
        Y-=Y.mean()
        Y/=Y.std()
        Xnorm=X/Y.std()
        marketSeriesIdxAvailable=sorted(set(marketSeriesIdx).intersection(Xdaily.ix[dMax].dropna().index))
        noAR1CoeffIdx=list(marketSeriesIdx)+list(self.params.macroSeriesTransform.keys())

        Nstatic=self.params.Nstatic
                
        fe=MacroUtils.FactorExtractorPCAar1(Y.dropna(),noAR1CoeffIdx) #HACK: 
        results=fe.estimate(Nstatic=Nstatic,Niters=ssm_iters)
        instRetM=results.F.dot(results.Lambda[instruments])

        eta=results.eta
        Psi=results.Psi.T
        Lambda=results.Lambda.T
        F=results.F

        Psis={}
        Psis[0]=np.eye(Nstatic)+0.*Psi
        for i in range(1,Lmax):
            Psis[i]=Psi.dot(Psis[i-1])

        PsiSum=sum(Psis.values())
        LambdaPsiSum=Lambda.dot(PsiSum)
        Yresponse=eta.dot(LambdaPsiSum.T)
        delta=Yresponse.copy()

        eps=delta[instruments].copy()
        if self.params.residualize_delta:
            epsCoefsDict={eps.columns[0]:{eps.columns[0]:0.}}
            for i in range(1,len(instruments)):
                xx=eps.ix[:,list(range(0,i))]
                yy=eps.ix[:,i].copy()
                fit=pandas.ols(y=yy,x=xx,intercept=False)
                epsCoefsDict[eps.columns[i]]=fit.beta
                eps.ix[:,i]=fit.resid
            epsCoefs=np.eye(eps.shape[1]) + pandas.DataFrame(epsCoefsDict,
                    index=eps.columns,columns=eps.columns).T.fillna(0.)
            Q=np.linalg.pinv(epsCoefs)
        else:
            Q=np.eye(eps.shape[1])
        Hinv=np.dot(Q,LambdaPsiSum.ix[instruments].values)
        H=np.linalg.pinv(Hinv)
        
        LambdaH=pandas.DataFrame(np.dot(Lambda.values,H),index=Lambda.index,columns=instruments) #+ Lambda*0.
        folio=np.linalg.pinv(LambdaH.ix[marketSeriesIdxAvailable].copy().values
                ) + LambdaH.ix[marketSeriesIdxAvailable].T*0.

        
        monthlyFactorReturns=X[folio.columns].dot(((1./sigmaX[folio.columns])*(folio)).T).fillna(0.) #HACK:
        dailyFactorReturns=Xdaily[folio.columns].dot(((1./sigmaX[folio.columns])*(folio)).T).fillna(0.)


        #TODO: try adding back the missing factor as the residual of the eta regressed against the delta.  
        #Hack: Arbitrary scale
        dailyFactorReturnsNormalized=np.sqrt(1./252.)*(dailyFactorReturns/
                (dailyFactorReturns.ix[-500:].std())).fillna(0.)
        results=Utilities.Struct()
        results.dailyFactorReturnsNormalized=dailyFactorReturnsNormalized
        results.X=X
        results.Xdaily=Xdaily
        results.fRetD=fRetD
        results.fRetM=fRetM

        return results

class MacroFactorReturnsCalculator_CoreSSM_base_bareV5_2(MacroFactorReturnCalculator_base):
    """...  """
    def __init__(self,params=None):
        MacroFactorReturnCalculator_base.__init__(self,params)
        assert 'macroMetaCsvFile' in self.params.getFieldNames(), 'Must set macroMetaCsfVile in riskModel'

        if 'residualize_delta' not in self.params.getFieldNames():
            self.params.residualize_delta=True

        if 'macroSeriesIdx' not in self.params.getFieldNames():
            self.params.macroSeriesIdx=['unrate','unrate_16_plus','treas_yield_20y_monthly','treas_yield_3m_monthly',
            'broad_fx','civ_empl','leading_econ_idx','ppi_core','ppi','phil_outlook_svy','pers_sav_pct',
            'pers_inc','nahb_housing_mkt_idx','ism_pmi','ind_prod_total_idx','indprod_mfg','emp_nonfarm',
            'disp_pers_inc','cpi','cpi_core','cci','chicago_pmbb','caputil',
            'hours_nonfarm','wages_mfg','wages_nonfarm']+['pce'] #This one has problems in the db.
        
        if 'macroSeriesTransform' not in self.params.getFieldNames():
            self.params.macroSeriesTransform={
                    'ind_prod_total_idx':{'name':'ind_prod_lead','shift':-1,'diff':1,'geom':True},
                    'cpi':               {'name':'cpi_d2',       'shift':0,'diff':2,'geom':True},
                    'cpi_core':          {'name':'cpi_core_d2',  'shift':0,'diff':2,'geom':True},
                    'ppi':               {'name':'ppi_d2',       'shift':0,'diff':2,'geom':True},
                    'ppi_core':          {'name':'ppi_core_d2',  'shift':0,'diff':2,'geom':True},
                    }
                    
        if 'f2scale' not in self.params.getFieldNames():
            self.params.f2scale={'Economic Growth':0.03,'Inflation':0.05,'Confidence':0.05,}


    def _processCoreMacroData(self,modelData):
        logging.info('Begin: _processCoreMacroData')
        self.processTrivialCoreMacroFactors(modelData)

        fCumRetsDcoreTriv=modelData.fRetCalcResults.fCumRetsD_MacroCore.copy()
        fRetsDcoreTriv=modelData.fRetCalcResults.fRetsD_MacroCore.copy()
        #dates=list(fCumRetsDcore.index)
        dates=[datetime.datetime(d.year,d.month,d.day) for d in modelData.dates]

        factorNames=sorted(fCumRetsDcoreTriv.columns)
        instruments=sorted(self.params.f2i.values())
        i2f=dict( (v,k) for k,v in self.params.f2i.items())
        Nstatic=len(factorNames)
        if 'Nstatic' in self.params.getFieldNames():
            Nstatic=self.params.Nstatic

        metaAll=pandas.read_csv(self.params.macroMetaCsvFile)

        macroSeriesIdx=self.params.macroSeriesIdx   

        marketSeriesIdx=([]
                        #+list(modelData.master.marketInterceptReturns.columns)
                        #+list(industryCols)
                        +list(modelData.master.styleReturns.columns)
                        +list(modelData.indexD['fx'])
                        +list(modelData.indexD['commodity']) #Careful with TR, etc
                        )
        marketSeriesIdx=sorted(set(marketSeriesIdx).intersection(modelData.allCumRetsD.columns))
        logging.info('macroSeriesIdx  '+','.join(macroSeriesIdx))
        logging.info('marketSeriesIdx  '+','.join(marketSeriesIdx))

        fCumRetD=modelData.allCumRetsD[marketSeriesIdx].ix[dates].copy() 

        marketInterceptRetD=modelData.master.marketInterceptReturns['Market Intercept']
        secActiveRetD=self.params.activeSectorReturns.ix[dates].fillna(0.).rename(columns=lambda c:c+'_active')
        secTotalRetD=secActiveRetD.sub(marketInterceptRetD,axis=0).rename(columns=lambda c:c+'_total')
        
        secActiveCumRetD=(1.+secActiveRetD).cumprod()
        secTotalCumRetD=(1.+secTotalRetD).cumprod()
        marketInterceptCumRetD=(1.+marketInterceptRetD).cumprod()
        marketInterceptCumRetD.name='US3--Market Intercept'
        
        fCumRetDTmp=fCumRetD.join(secActiveCumRetD).join(secTotalCumRetD).join(marketInterceptCumRetD)
        marketSeriesIdx=marketSeriesIdx+sorted(secActiveCumRetD.columns) + ['US3--Market Intercept'] # THIS IS WHERE WE CHANGE THINGS.
        fCumRetD=fCumRetDTmp[marketSeriesIdx].copy().fillna(method='bfill') #NOTE Not sure here why fillna needed.

        extraMacroSeries={}
        macroSeriesMonthly=modelData.macroSeriesMonthly.copy()
        for k,v in self.params.macroSeriesTransform.items():
            tmpSeries=macroSeriesMonthly[k].copy().shift(v['shift'])
            tmpSeries.name=v['name']
            
            if v['geom']:
                tmpSeries/=tmpSeries.ix[0]
                tmpSeries=np.log(tmpSeries)
            for i in range(v['diff']):
                tmpSeries-=tmpSeries.shift(1)
            macroSeriesMonthly=macroSeriesMonthly.join(tmpSeries)
            extraMacroSeries[tmpSeries.name]=tmpSeries.copy()

        macroSeriesMonthly=macroSeriesMonthly.ix[dates[0]:dates[-1]].copy()

        results=self.extractCoreFactorReturns(instruments,fCumRetD,macroSeriesMonthly,
                                          marketSeriesIdx,macroSeriesIdx+list(extraMacroSeries.keys()),metaAll,Lmax=4,ssm_iters=4)
        modelData.tmpResults=results

        fRetsDcore=results.dailyFactorReturnsNormalized.rename(columns=i2f).reindex(columns=factorNames).fillna(0.) 
        modelData.tmp_fRetsDcore=fRetsDcore
        fRetsDcore.ix[0]=0. #HACK: for good measure. will make essentially no difference
        for c in fRetsDcore.columns:
            fRetsDcore[c]*=self.params.f2scale.get(c,0.01)

        fCumRetsDcore=(1.+fRetsDcore).cumprod()
        modelData.fRetCalcResults.fCumRetsD_MacroCore=fCumRetsDcore
        modelData.fRetCalcResults.fRetsD_MacroCore=fRetsDcore
        logging.info('End: _processCoreMacroData')

    def extractCoreFactorReturns(self,instruments,fCumRetD,macroSeriesMonthly,
                             marketSeriesIdx,macroSeriesIdx,metaAll,Lmax=4,ssm_iters=4):
        meta=metaAll.ix[metaAll.active>0]
        meta=meta.rename(index=meta.shortname)
        metaM=meta[meta.freq=='M'].copy()
        metaD=meta[meta.freq=='D'].copy()

        geomIdxD=sorted(metaD[metaD.type=='geom'].index)
        percentIdxD=sorted(metaD[metaD.type=='percent'].index)

        geomIdxM=sorted(metaM[metaM.type=='geom'].index)
        percentIdxM=sorted(metaM[metaM.type=='percent'].index)
        linIdxM=sorted(metaM[metaM.type=='lin'].index)
        macroM=MacroUtils.d2dt(macroSeriesMonthly)
        tmp=macroM[geomIdxM].copy()
        macroM[tmp.columns]=tmp/(tmp.shift(1)) -1.
        tmp=macroM[percentIdxM].copy()
        macroM[tmp.columns]=tmp - (tmp.shift(1))
        macroM=macroM.ix[1:].copy()

        fCumRetD=MacroUtils.d2dt(fCumRetD).dropna(axis=1) # HACK: crude to dropna.needed if we orth returns here  
        fRetD=(fCumRetD/(fCumRetD.shift(1)) -1.)#.dropna()
        fCumRetDi=MacroUtils.d2dt(MacroUtils.expandDatesDF(fCumRetD))
        fCumRetM=fCumRetDi.ix[[d for d in fCumRetDi.index if d.day==1]]
        fRetM=(fCumRetM/fCumRetM.shift(1) -1.).ix[1:].copy()
        
        dMin=max(fRetM.index[0],macroM.index[0])
        dMax=fCumRetD.index[-1]
        
        X=macroM[macroSeriesIdx].ix[dMin:dMax].copy()
        X=X.join(fRetM)
        Xdaily=fRetD.copy()
        sigmaX=X.std()
        Y=X.copy()
        Y-=Y.mean()
        Y/=Y.std()
        Xnorm=X/Y.std()
        marketSeriesIdxAvailable=sorted(set(marketSeriesIdx).intersection(Xdaily.ix[dMax].dropna().index))
        noAR1CoeffIdx=list(marketSeriesIdx)+list(self.params.macroSeriesTransform.keys())

        Nstatic=self.params.Nstatic
                
        fe=MacroUtils.FactorExtractorPCAar1(Y.dropna(),noAR1CoeffIdx) #HACK: 
        results=fe.estimate(Nstatic=Nstatic,Niters=ssm_iters)
        instRetM=results.F.dot(results.Lambda[instruments])

        eta=results.eta
        Psi=results.Psi.T
        Lambda=results.Lambda.T
        F=results.F

        Psis={}
        Psis[0]=np.eye(Nstatic)+0.*Psi
        for i in range(1,Lmax):
            Psis[i]=Psi.dot(Psis[i-1])

        PsiSum=sum(Psis.values())
        LambdaPsiSum=Lambda.dot(PsiSum)
        Yresponse=eta.dot(LambdaPsiSum.T)
        delta=Yresponse.copy()

        eps=delta[instruments].copy()
        if self.params.residualize_delta:
            epsCoefsDict={eps.columns[0]:{eps.columns[0]:0.}}
            for i in range(1,len(instruments)):
                xx=eps.ix[:,list(range(0,i))]
                yy=eps.ix[:,i].copy()
                fit=pandas.ols(y=yy,x=xx,intercept=False)
                epsCoefsDict[eps.columns[i]]=fit.beta
                eps.ix[:,i]=fit.resid
            epsCoefs=np.eye(eps.shape[1]) + pandas.DataFrame(epsCoefsDict,
                    index=eps.columns,columns=eps.columns).T.fillna(0.)
            Q=np.linalg.pinv(epsCoefs)
        else:
            Q=np.eye(eps.shape[1])
        Hinv=np.dot(Q,LambdaPsiSum.ix[instruments].values)
        H=np.linalg.pinv(Hinv)
        
        LambdaH=pandas.DataFrame(np.dot(Lambda.values,H),index=Lambda.index,columns=instruments) #+ Lambda*0.
        folio=np.linalg.pinv(LambdaH.ix[marketSeriesIdxAvailable].copy().values
                ) + LambdaH.ix[marketSeriesIdxAvailable].T*0.

        
        monthlyFactorReturns=X[folio.columns].dot(((1./sigmaX[folio.columns])*(folio)).T).fillna(0.) #HACK:
        dailyFactorReturns=Xdaily[folio.columns].dot(((1./sigmaX[folio.columns])*(folio)).T).fillna(0.)


        #TODO: try adding back the missing factor as the residual of the eta regressed against the delta.  
        #Hack: Arbitrary scale
        dailyFactorReturnsNormalized=np.sqrt(1./252.)*(dailyFactorReturns/
                (dailyFactorReturns.ix[-500:].std())).fillna(0.)
        results=Utilities.Struct()
        results.dailyFactorReturnsNormalized=dailyFactorReturnsNormalized
        results.X=X
        results.Xdaily=Xdaily
        results.fRetD=fRetD
        results.fRetM=fRetM

        return results



