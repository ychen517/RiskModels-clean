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

def _StoreData(data,fname):
    """Dump most objects to a file."""
    import pickle
    pickle.dump(data, open(fname,'wb'),pickle.HIGHEST_PROTOCOL)

def _FetchStoredData(fname):
    """Import an object from StoreData."""
    import pickle
    return pickle.load(open(fname,'rb'))

def _getDaysInUniformMonth(dates,dMin=None,dMax=None,days_per_month=23):
    """
    Get a set of dates with given dates, the FOM, and no Saturdays
    that has exactly days_per_month.  For the US, this is either 23 or 24.

    This is mainly for simplifying the Kalman filter implementation.
    """
    if dMin is None: dMin=min(dates)
    if dMax is None: dMax=max(dates)
    allDatesRaw=pandas.date_range(start=dMin-datetime.timedelta(31),end=dMax+datetime.timedelta(31),freq='D')
    fomDates=[d.date() for d in allDatesRaw if d.day==1] 
    allDates=[d.date() for d in allDatesRaw  ]
    
    tabooDates=[d.date() for d in allDatesRaw if d.dayofweek==6]
    essentialDates=sorted(set(fomDates).union(dates))

    potentialDates=sorted(set(allDates).difference(tabooDates))
    candidateDates=sorted(set(potentialDates).difference(essentialDates))
    
    fom2EssentialDates={}
    fom2PotentialDates={}
    fom2CandidateDates={}
    for dt,dt2 in zip(fomDates[:-2],fomDates[1:]): 
        fom2PotentialDates[dt]=[d for d in potentialDates if dt<=d<dt2]
        fom2CandidateDates[dt]=[d for d in candidateDates if dt<=d<dt2]
        fom2EssentialDates[dt]=[d for d in essentialDates if dt<=d<dt2]

    fom2PaddedDates={}
    for k,v in fom2EssentialDates.items():
        fom2PaddedDates[k]=sorted(set(v).union(fom2CandidateDates[k][:(days_per_month-len(v))]))

    #badMonths=[(k,len(v)) for k,v  in fom2PaddedDates.iteritems() if len(v)!=days_per_month]
    #assert len(badMonths)==0, badMonths
    #if len(badMonths)>0: print 'WARNING: some bad months in _getDaysInUniformMonth', badMonths

    newDates=set(essentialDates)
    for k,v in fom2PaddedDates.items():
        newDates.update(v)

    return sorted([d for d in newDates if dates[0]<=d<=dates[-1]])

def _getDaysInUniformMonth_broken(dates,
        dMin=datetime.datetime(1980,1,1),
        dMax=datetime.datetime(2020,1,1),
        days_per_month=23):
    """
    Get a set of dates with given dates, the FOM, and no Saturdays
    that has exactly days_per_month.  For the US, this is either 23 or 24.

    This is mainly for simplifying the Kalman filter implementation.
    """
    #if True:
    allDatesRaw=pandas.date_range(start=dMin,end=dMax,freq='D')
    fomDates=[d.date() for d in allDatesRaw if d.day==1] 
    #allDates=[d.date() for d in allDatesRaw if d.date()>=fomDates[0] ]
    allDates=[d.date() for d in allDatesRaw  ]
    
    tabooDates=[d.date() for d in allDatesRaw if d.dayofweek==6]
    essentialDates=sorted(set(fomDates).union(dates))

    potentialDates=sorted(set(allDates).difference(tabooDates))
    candidateDates=sorted(set(potentialDates).difference(essentialDates))
    
    fom2EssentialDates={}
    fom2PotentialDates={}
    fom2CandidateDates={}
    for dt,dt2 in zip(fomDates[:-2],fomDates[1:]): 
        fom2PotentialDates[dt]=[d for d in potentialDates if dt<=d<dt2]
        fom2CandidateDates[dt]=[d for d in candidateDates if dt<=d<dt2]
        fom2EssentialDates[dt]=[d for d in essentialDates if dt<=d<dt2]

    fom2PaddedDates={}
    for k,v in fom2EssentialDates.items():
        fom2PaddedDates[k]=sorted(set(v).union(fom2CandidateDates[k][:(days_per_month-len(v))]))

    badMonths=[(k,len(v)) for k,v  in fom2PaddedDates.items() if len(v)!=days_per_month]
    #assert len(badMonths)==0, badMonths
    #if len(badMonths)>0: print 'WARNING: some bad months in _getDaysInUniformMonth', badMonths

    newDates=set([])
    for k,v in fom2PaddedDates.items():
        newDates.update(v)

    return sorted(newDates)

def _asdate(d):
    return datetime.date(d.year,d.month,d.day)

def _asdatetime(d):
    return datetime.datetime(d.year,d.month,d.day)

def _series(x,index=None,columns=None): #TODO: make this more flexible
    if isinstance(x,pandas.Series):
        return x
        
    nt=x.shape[0]
    if index is None:
        index=pandas.date_range(start=datetime.daatetime(2000,1,1),
                                   periods=nt,freq='D')
    elif isinstance(index,str):
        tLabel=index
        index=[ tLabel+'_%03d'%c for c in  range(nt)]
    return pandas.Series(x,index=index).copy()

def _df(x,index=None,columns=None): #TODO: make this more flexible
    if isinstance(x,pandas.DataFrame):
        return x
        
    nt,nx=x.shape
    if index is None:
        index=pandas.date_range(start=datetime.datetime(2000,1,1),
                                   periods=nt,freq='D')
    elif isinstance(index,str):
        tLabel=index
        index=[ tLabel+'_%03d'%c for c in  range(nt)]
    if isinstance(columns,str):
        xLabel=columns
        columns=[ xLabel+'_%03d'%c for c in  range(nx)]
    return pandas.DataFrame(x,index=index,columns=columns).copy()

def trimDF(df,q0=0.001,q1=0.999):
    Nt=max(3,df.shape[0]) 
    q0=max(q0,1./(Nt-2))
    q1=max(q1,1. - 1./(Nt-2))
    df2=df.copy()
    for c,s in df.items():
        df2[c]=s.clip(s.quantile(q0),s.quantile(q1))
    return df2

def d2dt(X):
    if isinstance(X.index[0],datetime.datetime):
        return X.copy()
    if isinstance(X.index[0],datetime.date):
        if isinstance(X,pandas.DataFrame):
            return X.rename(index=lambda d: datetime.datetime(d.year,d.month,d.day)).copy()
        else:
            return X.rename(lambda d: datetime.datetime(d.year,d.month,d.day)).copy()

    if isinstance(X.index[0],pandas.tslib.Timestamp):
        if isinstance(X,pandas.DataFrame):
            return X.rename(index=lambda d: d.date()).copy()
        else:
            return X.rename(lambda d: d.date()).copy()
        
    assert False

def dt2d(X):
    if isinstance(X.index[0],datetime.date):
        return X.copy()
    else: #if isinstance(X.index[0],datetime.datetime): #also need Timestamp
        if isinstance(X,pandas.DataFrame):
            return X.rename(index=lambda d: datetime.datetime(d.year,d.month,d.day)).copy()
        else:
            return X.rename(lambda d: datetime.datetime(d.year,d.month,d.day)).copy()
    assert False

def bsmOptionPrice(s,k,t,v,rf=0.,div=0.,cp=1):
    """A cheap bsm option pricer.
       cp = +/-1 depending on call or put.
    """
    d1=(np.log(s/k)+(rf-div+0.5*(v**2))*t)/(v*np.sqrt(t))
    d2=d1-v*np.sqrt(t)
    price=cp*s*np.exp(-div*t)*stats.norm.cdf(cp*d1) - cp*k*np.exp(-rf*t)*stats.norm.cdf(cp*d2)
    return price


def sym_sqrt(C):
    import numpy as np
    import numpy.linalg
    pp,qq=np.linalg.eigh(C)
    sqrtC=np.dot(np.dot(qq,np.diag(np.sqrt(pp))),qq.T)
    return 0.*C + sqrtC

class LinearModel(object):
    def __init__(self, Y, X, Omega=None, window=None, weights=None):
        Y = Y.reindex(index=X.index, copy=False)
        self.Y = Y
        self.X = X
        
        self.Omega = None
        if Omega is not None:
            self.Omega = Omega.reindex(index=X.index,columns=X.index,copy=False)
            self.window = None
        
        self.window = None
        if window is not None:
            self.window = window
            ab = np.ones(shape=(window, self.X.shape[0]))
            for i in range(1,self.window):
                ab[i,:] *= float(self.window-i)/float(self.window)
            self.banded_Omega = ab
        
        self.weights = pandas.Series(1.0, index=X.index)
        if weights is not None:
            self.weights = weights.reindex(index=X.index)
        if np.any(np.any(pandas.isnull(self.Y),axis=0),axis=0):
            raise Exception('Y variable contains missing values')
        if np.any(np.any(pandas.isnull(self.X),axis=0),axis=0):
            raise Exception('X variable contains missing values')
        if self.Omega is not None:
            if np.any(np.any(pandas.isnull(self.Omega),axis=0),axis=0):
                raise Exception('Omega variable contains missing values')
        if np.any(pandas.isnull(self.weights)):
            raise Exception('Weights are missing values')
        self._beta = None
        self._white_resid = None
        self._scale = None
        self._bse = None
        self._tstat = None
        self._pvalue = None
        self._r2 = None
        self._resid = None
        self._xtxinv = None
        self._vif = None

    def __computeBeta(self):
        if self.Omega is None and self.window is None:
            self._xtx = pandas.DataFrame(self.X.T.dot(self.X.mul(self.weights,axis=0)), index=self.X.columns, columns=self.X.columns)
            self._xty = self.X.mul(self.weights,axis=0).T.dot(self.Y)
        elif self.window is None:
            if isinstance(self.Y, pandas.DataFrame):
                self._omega_y = pandas.DataFrame(linalg.solve(self.Omega.values, self.Y.values, sym_pos=True), index=self.X.index, columns=self.Y.columns)
            else:
                self._omega_y = pandas.Series(linalg.solve(self.Omega.values, self.Y.values, sym_pos=True), index=self.X.index)

            self._omega_x = pandas.DataFrame(linalg.solve(self.Omega.values, self.X.values, sym_pos=True), index=self.X.index, columns=self.X.columns)
            self._xtx = pandas.DataFrame(self.X.T.dot(self._omega_x), index=self.X.columns, columns=self.X.columns)
            self._xty = self.X.T.dot(self._omega_y)
        elif self.window is not None:
            if isinstance(self.Y, pandas.DataFrame):
                self._omega_y = pandas.DataFrame(linalg.solveh_banded(self.banded_Omega, self.Y.values, lower=True), index=self.X.index, columns=self.Y.columns)
            else:
                self._omega_y = pandas.Series(linalg.solveh_banded(self.banded_Omega, self.Y.values, lower=True), index=self.X.index)

            self._omega_x = pandas.DataFrame(linalg.solveh_banded(self.banded_Omega, self.X.values, lower=True), index=self.X.index, columns=self.X.columns)
            self._xtx = pandas.DataFrame(self.X.T.dot(self._omega_x), index=self.X.columns, columns=self.X.columns)
            self._xty = self.X.T.dot(self._omega_y)

        if isinstance(self.Y, pandas.DataFrame):
            self._beta = pandas.DataFrame(linalg.solve(self._xtx, self._xty, sym_pos=True).T, columns=self.X.columns, index=self.Y.columns)
        else:
            self._beta = pandas.Series(linalg.solve(self._xtx, self._xty, sym_pos=True), index=self.X.columns)

    def __computeStats(self):
        if self._beta is None:
            self.__computeBeta()

        if self.Omega is None and self.window is None:
            self._white_resid = self.Y - self.X.dot(self._beta.T) 
            if isinstance(self._white_resid, pandas.DataFrame):
                self._white_resid = self._white_resid.mul(np.sqrt(self.weights),axis=0)
                self._Linvy = self.Y.mul(np.sqrt(self.weights),axis=0)
            else:
                self._white_resid = np.sqrt(self.weights) * self._white_resid
                self._Linvy = np.sqrt(self.weights) * self.Y
        elif self.window is None:
            L = linalg.cholesky(self.Omega.values, lower=True)
            if isinstance(self.Y, pandas.DataFrame):
                self._Linvy = pandas.DataFrame(linalg.solve_triangular(L, self.Y.values, lower=True), index=self.X.index, columns=self.Y.columns)
            else:
                self._Linvy = pandas.Series(linalg.solve_triangular(L, self.Y.values, lower=True), index=self.X.index)
            self._Linvx = pandas.DataFrame(linalg.solve_triangular(L, self.X.values, lower=True), index=self.X.index, columns=self.X.columns)
            self._white_resid = self._Linvy - self._Linvx.dot(self._beta.T)
        elif self.window is not None:
            L = linalg.cholesky_banded(self.banded_Omega, lower=True)
            if isinstance(self.Y, pandas.DataFrame):
                self._Linvy = pandas.DataFrame(linalg.solve_banded((self.window-1,0), L, self.Y.values), index=self.X.index, columns=self.Y.columns)
            else:
                self._Linvy = pandas.Series(linalg.solve_banded((self.window-1,0), L, self.Y.values), index=self.X.index)
            self._Linvx = pandas.DataFrame(linalg.solve_banded((self.window-1,0), L, self.X.values), index=self.X.index, columns=self.X.columns)
            self._white_resid = self._Linvy - self._Linvx.dot(self._beta.T)

        self._df_resid = len(self.X.index) - len(self.X.columns)

        self._scale = (self._white_resid**2).sum()/self._df_resid

        if isinstance(self._scale, pandas.Series):
            self._bse = np.sqrt(pandas.DataFrame(np.outer(self._scale.values,np.diag(linalg.pinv(self._xtx))),columns=self.X.columns,index=self._scale.index))
        else:
            self._bse = pandas.Series(np.sqrt(self._scale * np.diag(linalg.pinv(self._xtx))), index=self.X.columns)

        self._tstat = self._beta/self._bse

        pvalue = stats.t.sf(np.abs(self._tstat), self._df_resid)*2
        
        if isinstance(self._tstat, pandas.Series):
            self._pvalue = pandas.Series(pvalue, index=self._tstat.index)
        else:
            self._pvalue = pandas.DataFrame(pvalue, index=self._tstat.index, columns=self._tstat.columns)

        self._r2 = 1.0 - (self._scale * self._df_resid)/(self._Linvy**2).sum()

        self._resid = self.Y - self.X.dot(self._beta.T)

        self._xtxinv = pandas.DataFrame(linalg.pinv(self._xtx), index=self._xtx.index, columns=self._xtx.columns)
        
    def __computeVIF(self):
        vif = {}
        for col in self.X.columns:
            y = self.X[col]
            X = self.X.drop([col],axis=1)
            model = LinearModel(y,X,Omega=self.Omega,window=self.window,weights=self.weights)
            vif[col] = 1.0/(1 - model.r2)
        self._vif = pandas.Series(vif)

    @property
    def beta(self):
        if self._beta is None:
            self.__computeBeta()
        return self._beta

    @property
    def scale(self):
        if self._scale is None:
            self.__computeStats()
        return self._scale

    @property
    def bse(self):
        if self._bse is None:
            self.__computeStats()
        return self._bse

    @property
    def tstat(self):
        if self._tstat is None:
            self.__computeStats()
        return self._tstat
        
    @property
    def pvalue(self):
        if self._pvalue is None:
            self.__computeStats()
        return self._pvalue

    @property
    def r2(self):
        if self._r2 is None:
            self.__computeStats()
        return self._r2

    @property
    def resid(self):
        if self._resid is None:
            self.__computeStats()
        return self._resid

    @property
    def xtxinv(self):
        if self._xtxinv is None:
            self.__computeStats()
        return self._xtxinv

    @property
    def xtx(self):
        if self._beta is None:
            self.__computeBeta()
        return self._xtx

    @property
    def vif(self):
        if self._vif is None:
            self.__computeVIF()
        return self._vif

    def conf_int(self, alpha=0.05):
        if self._bse is None:
            self.__computeStats()
        q = stats.t.ppf(1 - alpha / 2, self._df_resid)
        return (self.beta - q * self.bse, self.beta + q * self.bse)

class CochraneOrcutt(object):
    def __init__(self, Y, X):
        Y = Y.reindex(index=X.index, copy=False)
        self.Y = Y
        self.X = X
        
        if np.any(np.any(pandas.isnull(self.Y),axis=0),axis=0):
            raise Exception('Y variable contains missing values')
        if np.any(np.any(pandas.isnull(self.X),axis=0),axis=0):
            raise Exception('X variable contains missing values')
        self._beta = None
        self._rho = None

    def __computeSingle(self, y):
        Y = y.copy()
        Y.name = 'tmp'
        X = self.X.copy()
        origY = Y.copy()
        origX = self.X.copy()

        initrho = 1.0
        converged = False
        for itn in range(10):
            model = LinearModel(Y,X)
            resid = origY - origX.dot(model.beta.T)
            lagresid = resid.shift(1).dropna()
            resid = resid.reindex(lagresid.index)
            armodel = LinearModel(resid,pandas.DataFrame({'l1': lagresid}))
            rho = armodel.beta['l1']
            Y = (origY - rho * origY.shift(1)).dropna()
            X = (origX - rho * origX.shift(1)).dropna()
            if abs(rho - initrho) < 1e-5:
                converged = True
                break
            initrho = rho

        model = LinearModel(Y,X)
        return model, rho, converged 

    def __computeBeta(self):
        if isinstance(self.Y, pandas.DataFrame):
            betas = {}
            rhos = {}
            allconverged = {}
            for yname, y in self.Y.items():
                model, rho, converged = self.__computeSingle(y)
                betas[yname] = model.beta 
                rhos[yname] = rho
                allconverged[yname] = converged
            self._beta = pandas.DataFrame(betas, columns=self.Y.columns).T
            self._rho = pandas.Series(rhos, index=self.Y.columns)
            self._converged = pandas.Series(allconverged, index=self.Y.columns)
        else:
            model, rho, converged = self.__computeSingle(self.Y)
            self._beta = model.beta.copy()
            self._rho = rho
            self._converged = converged

    @property
    def beta(self):
        if self._beta is None:
            self.__computeBeta()
        return self._beta

    @property
    def rho(self):
        if self._rho is None:
            self.__computeBeta()
        return self._rho

    @property
    def converged(self):
        if self._converged is None:
            self.__computeBeta()
        return self._converged

class RegressionBase(object):
    def __init__(self, Y, X):
        Y = Y.reindex(index=X.index, copy=False)
        self.Y = Y
        self.X = X
        
        if np.any(np.any(pandas.isnull(self.Y),axis=0),axis=0):
            raise Exception('Y variable contains missing values')
        if np.any(np.any(pandas.isnull(self.X),axis=0),axis=0):
            raise Exception('X variable contains missing values')
        self._beta = None
        self._resid = None

    def _computeSingle(self, y):
        raise NotImplementedError
    
    def __computeBeta(self):
        if isinstance(self.Y, pandas.DataFrame):
            betas = {}
            for yname, y in self.Y.items():
                model = self._computeSingle(y)
                betas[yname] = model.params 
            self._beta = pandas.DataFrame(betas).T
        else:
            model = self._computeSingle(self.Y)
            self._beta = model.params

    def __computeStats(self):
        if self._beta is None:
            self.__computeBeta()
        self._resid = self.Y - self.X.dot(self._beta.T)

    @property
    def beta(self):
        if self._beta is None:
            self.__computeBeta()
        return self._beta

    @property
    def resid(self):
        if self._resid is None:
            self.__computeStats()
        return self._resid

class RobustLinearModel(RegressionBase):
    def __init__(self, Y, X):
        RegressionBase.__init__(self, Y, X)

    def _computeSingle(self, y):
        model = sm.RLM(endog=y,exog=self.X,M=sm.robust.norms.HuberT(t=4.0)) 
        result = model.fit()
        return result

class QuantileLinearModel(RegressionBase):
    def __init__(self, Y, X, tau=0.5):
        RegressionBase.__init__(self, Y, X)
        self.tau = tau
        if self.tau <= 0.0 or self.tau >= 1.0:
            raise Exception('Tau must be between 0 and 1')
    
    def _computeSingle(self, y):
        import socp
        relax = socp.SOCPRelaxation('regress')
        betavar = {}
        for c in self.X.columns:
           betavar[c] = relax.createFreeVariable()
        obj = relax.getObjective()
        for r, vals in self.X.iterrows():
            uplus = relax.createNonnegVariable()
            uminus = relax.createNonnegVariable()
            obj.addTerm(uplus, self.tau)
            obj.addTerm(uminus, 1.0 - self.tau)
            con = relax.createLinearConstraint(y[r])
            con.addTerm(uplus, 1.0)
            con.addTerm(uminus, -1.0)
            for c, val in vals.items():
                con.addTerm(betavar[c], val)

        params = socp.SolverParams(False)
        (status, objval) = relax.optimize(True, params)
        if status != socp.RelaxationOptimal:
            raise Exception('Unable to solve one of the regressions')
        result = Struct()
        result.params = pandas.Series(dict((c, var.getValue()) for c, var in betavar.items()))
        return result

class ConstrainedLinearModel(RegressionBase):
    """ r = B f + e
         C f = 0
    """
    def __init__(self, Y, X, C, weights=None):
        RegressionBase.__init__(self, Y, X)
        self.C = C.reindex(columns=X.columns, copy=False)
        self.weights = pandas.Series(1.0, index=X.index)
        if weights is not None:
            self.weights = weights.reindex(index=X.index)
        if np.any(pandas.isnull(self.weights)):
            raise Exception('Weights are missing values')
        
    def _computeSingle(self, y):
        import popt
        env = popt.Environment()
        folio = popt.Portfolio(env, 'regression', 1.0)
        rWB = (y * self.weights).dot(self.X)
        BTWB = self.X.T.dot(self.X.mul(self.weights, axis=0))
        factor_asy = {}
        factor_asset = {}
        drm = popt.DenseRiskModel(env, 'rm')
        for c in self.X.columns:
            factor_asy[c] = popt.AssetSymbol(env, c)
            drm.addAsset(factor_asy[c], 0.0, 0.0)
            factor_asset[c] = popt.Asset(folio, factor_asy[c], 0.0, 0.0, 0.0,0.0)
        drm.setCovariance([factor_asy[c] for c in BTWB.columns], BTWB.values.flatten())
        
        popt.LinearObjective(folio, popt.toAssetGroup(folio, rWB.to_dict()), -1.0)
        popt.VarianceObjective(folio, folio.getAssets(), drm, 0.5)

        for name, coeffs in self.C.iterrows():
            coeffs = coeffs.dropna().to_dict()
            if len(coeffs) <= 0:
                continue
            con = popt.SimpleConstraint(folio, popt.toAssetGroup(folio, coeffs), popt.Eq, 0.0, popt.Absolute)
            con.setName(name)
        
        status = folio.minimize()
        if status.status != popt.Portfolio.SolutionFound:
            raise Exception('Unable to solve optimization')
        result = Struct()
        result.params = pandas.Series(folio.getOptimizedHoldings().to_dict())
        env.release()
        return result

class BayesianLinearModel(object):
    def __init__(self, Y, X, L, priorbeta=None, weights=None):
        Y = Y.reindex(index=X.index, copy=False)
        self.Y = Y
        self.X = X
       
        if np.any(np.any(pandas.isnull(self.Y),axis=0),axis=0):
            raise Exception('Y variable contains missing values')
        if np.any(np.any(pandas.isnull(self.X),axis=0),axis=0):
            raise Exception('X variable contains missing values')
        
        if isinstance(L, numbers.Number):
            self.L = pandas.DataFrame(float(L) * np.eye(len(self.X.columns)),index=self.X.columns,columns=self.X.columns)
        elif isinstance(L, pandas.Series):
            L = L.reindex(index=self.X.columns)
            if np.any(pandas.isnull(L)):
                raise Exception('L variable contains missing values')
            self.L = pandas.DataFrame(np.diag(L.values),index=self.X.columns,columns=self.X.columns)
        elif isinstance(L, pandas.DataFrame):
            self.L = L.reindex(index=self.X.columns,columns=self.X.columns)
            if np.any(np.any(pandas.isnull(self.L),axis=0),axis=0):
                raise Exception('L variable contains missing values')
        else:
            raise Exception('Unknown type for prior covariance')

        if priorbeta is None:
            self.priorbeta = pandas.Series(0.0,index=self.X.columns)
        else:
            if isinstance(priorbeta, pandas.Series):
                self.priorbeta = priorbeta.reindex(X.columns)
                if np.any(pandas.isnull(self.priorbeta)):
                    raise Exception('priorbeta contains missing values')
            elif isinstance(priorbeta, pandas.DataFrame):
                self.priorbeta = priorbeta.reindex(columns=self.Y.columns, index=self.X.columns)
                if np.any(np.any(pandas.isnull(self.priorbeta),axis=0),axis=0):
                    raise Exception('priorbeta variable contains missing values')
            else:
                raise Exception('Unsupported type for priorbeta')

        self.weights = pandas.Series(1.0, index=X.index)
        if weights is not None:
            self.weights = weights.reindex(index=X.index)
            if np.any(pandas.isnull(self.weights)):
                raise Exception('Weights are missing values')
        
        self._beta = None
        self._resid = None

    def __computeBeta(self):
        self._xtx = pandas.DataFrame(self.X.T.dot(self.X.mul(self.weights,axis=0)), index=self.X.columns, columns=self.X.columns)
        self._xty = self.X.mul(self.weights,axis=0).T.dot(self.Y)

        if isinstance(self.Y, pandas.DataFrame):
            self._beta = pandas.DataFrame(linalg.solve(self._xtx + self.L, self._xty.add(self.L.dot(self.priorbeta),axis='index'), sym_pos=True).T, columns=self.X.columns, index=self.Y.columns)
        else:
            self._beta = pandas.Series(linalg.solve(self._xtx + self.L, self._xty + self.L.dot(self.priorbeta), sym_pos=True), index=self.X.columns)

    def __computeStats(self):
        if self._beta is None:
            self.__computeBeta()

        self._resid = self.Y - self.X.dot(self._beta.T)
        
    @property
    def beta(self):
        if self._beta is None:
            self.__computeBeta()
        return self._beta

    @property
    def resid(self):
        if self._resid is None:
            self.__computeStats()
        return self._resid

class NestedLinearModel(object):
    def __init__(self, Y, X, blocks=None, orthogonalize=False):
        Y = Y.reindex(index=X.index, copy=False)
        self.Y = Y
        self.X = X
        self.orthogonalize = orthogonalize
      
        self.blocks = blocks
        if blocks is None:
            self.blocks = [[x] for x in self.X.columns]

        if sorted(itertools.chain.from_iterable(self.blocks)) != sorted(self.X.columns):
            raise Exception('Blocks must be partition of columns of X')

        if np.any(np.any(pandas.isnull(self.Y),axis=0),axis=0):
            raise Exception('Y variable contains missing values')
        if np.any(np.any(pandas.isnull(self.X),axis=0),axis=0):
            raise Exception('X variable contains missing values')
        self._beta = None
        self._rawbeta = None
        self._r2 = None
        self._resid = None

    def __computeBeta(self):
        betas = []
        if self.orthogonalize:
            orthmodel = Orthogonalizer(self.X, blocks=self.blocks)
            X = orthmodel.Xorth 
        else:
            X = self.X.copy()
        Y = self.Y
        for block in self.blocks:
            subX = X[block]
            model = LinearModel(Y, subX)
            betas.append(model.beta)
            Y = model.resid
        if isinstance(model.beta, pandas.Series):
            self._rawbeta = pandas.concat(betas, axis=0)
        else:
            self._rawbeta = pandas.concat(betas, axis=1)
        if self.orthogonalize:
            for i in range(len(self.blocks)-1,-1,-1):
                for j in range(i+1,len(self.blocks)):
                    betas[i] -= betas[j].dot(orthmodel.betas.ix[self.blocks[i],self.blocks[j]].T)
        if isinstance(model.beta, pandas.Series):
            self._beta = pandas.concat(betas, axis=0)
        else:
            self._beta = pandas.concat(betas, axis=1)

    def __computeStats(self):
        if self._beta is None:
            self.__computeBeta()
        self._resid = self.Y - self.X.dot(self._beta.T)
        self._r2 = 1.0 - (self._resid**2).sum()/(self.Y**2).sum()

    @property
    def beta(self):
        if self._beta is None:
            self.__computeBeta()
        return self._beta

    @property
    def rawbeta(self):
        if self._rawbeta is None:
            self.__computeBeta()
        return self._rawbeta

    @property
    def r2(self):
        if self._r2 is None:
            self.__computeStats()
        return self._r2

    @property
    def resid(self):
        if self._resid is None:
            self.__computeStats()
        return self._resid

class Orthogonalizer(object):
    def __init__(self, X, blocks=None):
        self.X = X
      
        self.blocks = blocks
        if blocks is None:
            self.blocks = [[x] for x in self.X.columns]

        if sorted(itertools.chain.from_iterable(self.blocks)) != sorted(self.X.columns):
            raise Exception('Blocks must be partition of columns of X')

        if np.any(np.any(pandas.isnull(self.X),axis=0),axis=0):
            raise Exception('X variable contains missing values')

        self._Xorth = None
        self._betas = None
        self._tstats = None
        self._r2 = None

    def __compute(self):
        Xorth = self.X.copy()
        prevNames = list(self.blocks[0])
        betas = {}
        tstats = {}
        r2 = {}
        for block in self.blocks[1:]:
            for f in block:
                model = LinearModel(Xorth[f], Xorth[prevNames])
                Xorth[f] = model.resid
                betas[f] = model.beta
                tstats[f] = model.tstat
                r2[f] = model.r2
            prevNames += block
        self._Xorth = Xorth
        self._betas = pandas.DataFrame(betas).reindex(index=self.X.columns,columns=self.X.columns)
        self._tstats = pandas.DataFrame(tstats).reindex(index=self.X.columns,columns=self.X.columns)
        self._r2 = pandas.Series(r2)

    @property
    def Xorth(self):
        if self._Xorth is None:
            self.__compute()
        return self._Xorth
    
    @property
    def betas(self):
        if self._betas is None:
            self.__compute()
        return self._betas

    @property
    def tstats(self):
        if self._tstats is None:
            self.__compute()
        return self._tstats

    @property
    def r2(self):
        if self._r2 is None:
            self.__compute()
        return self._r2

def fastLeastSquares(Y,X):
    """Returns a DataFrame for B  in Y=B*X + noise
    Conforming to the columns of X and Y as DataFrames. """

    tmp=np.linalg.lstsq(X.copy().values,Y.copy().values, rcond=-1)
    B=pandas.DataFrame(tmp[0],index=X.columns,columns=Y.columns).T.copy()
    return B

def fastLeastSquares2(Y,X,intercept=False):
    """Returns a DataFrame for B  in Y=B*X + noise
    Conforming to the columns of X and Y as DataFrames. """

    Bdict={}
    for c in Y.columns:
        fit=pandas.ols(y=Y[c],x=X,intercept=intercept)
        Bdict[c]=fit.beta.copy()
    
    B=pandas.DataFrame(Bdict).T.copy()
    
    return B

def linearInterpSeries(x):
    """Assumes x is a time series assuming index is datetime.date.  Fills missing values."""
    if x.count()<2:
        return x.fillna(method='bfill').fillna(method='ffill').fillna(0.)
    x2=x.rename(lambda d: float(d.toordinal())).copy()
    x2sparse=x2.dropna().copy()
    xInterp=np.interp(np.array(list(x2.index)),np.array(list(x2sparse.index)),
                      x2sparse.values) 
    return pandas.Series(xInterp,index=x.index)

def linearInterpDataFrame(x):
    xNew=x.copy()
    for c in x.columns:
        xNew[c]=linearInterpSeries(x[c].copy())
    return xNew

def expandDatesDF(x,interp=True,rename_dates=True,fast=False): #HACK: 
    logging.debug('expandDatesDF: begin')
    y=x.reindex(index=pandas.date_range(start=x.index[0],end=x.index[-1],freq='D')).copy()
    if interp and not fast:
        y = linearInterpDataFrame(y.rename(index=lambda d: datetime.date(d.year,d.month,d.day)))
    if interp and fast:
        print('warning using low order interpolation of dataframe for speed.')
        y = y.fillna(method='bfill').fillna(method='ffill')
    if rename_dates: #Redundant?
        y = y.rename(index=lambda d: datetime.date(d.year,d.month,d.day))
    logging.debug('expandDatesDF: end')
    return y

def selectInitialRelease(x):
    """ x is a dataframe with a single axiomaId from macroecon_dim_ts.
    Returns a series corresponding to the changeSeq==0 for each dt"""
    y=x[x.changeSeq==0]
    assert len(set(y.dt))==len(y.index)
    ySeries = y.value.astype(float)
    ySeries.index = y.dt
    ySeries=ySeries.reindex(index=sorted(ySeries.index))
    ySeries.name=x.axiomaId.values[0]
    return ySeries

def selectLatestRelease(x):
    """ x is a dataframe with a single axiomaId from macroecon_dim_ts.
    Returns a series corresponding to the max changeSeq for each dt"""
    values = {}
    for d, xD in x.groupby('dt'):
        effDtmax=xD.effDt.max() 
        subx = xD[xD.effDt == effDtmax]
        if len(subx.index) > 1:
            changeSeqmax = subx.changeSeq.max()
            subx = subx[subx.changeSeq == subx.changeSeq.max()]
        values.update(iter(subx.value.iteritems()))
    y = pandas.Series(values).rename(x.dt.to_dict())
    y = y.reindex(sorted(y.index))
    y.name = x.axiomaId.values[0]
    return y

def selectLatestReleaseBefore(x, dMax=None):
    if dMax is None:
        dMax=x.dt.max()
    valueDict = {}
    for dt, subdf in x.groupby('dt'):
        if dt > dMax:
            continue
        value = subdf.value
        effDt = subdf.effDt
        effDtmax = effDt.max()
        valueDict[dt]=value[effDt==effDtmax].values[0]
    return pandas.Series(valueDict)

def ReducedRankRegression(Y,X,Z=1.,r=None):#Bug in older versios...
    """ NOTE: this still needs some testing...
    Solves Y= alpha*beta'*X + Gamma*Z + noise
    """
    import scipy.linalg
    results=Utilities.Struct()

    assert 1 <= r <= min(X.shape[1],Y.shape[1])

    if r is None:
        print('Warning: no rank constraint.  Doing ols')
        r=min(X.shape[1],Y.shape[1])
    dates=list(Y.index)
    Nt=len(dates)

    if type(Z) is float : #USUALLY this is one.
        Z= Z * pandas.DataFrame(np.ones((Nt,1)),index=dates,columns=['Z_intercept'])

    zScale=1.
    dropZ=False
    if Z is None : #HACK: this will add some noise
        Z=pandas.DataFrame(10.e-8*np.random.randn(Nt,1),index=dates,columns=['Z_junk'])
        zScale=0.
        dropZ=True

    XYZ=Y.join(X,rsuffix='_x').join(Z,rsuffix='_z').copy()

    # Just to check
    olsB=fastLeastSquares2(Y.copy(),X.join(Z).copy())

    # Now really compute
    Mcov=XYZ.cov()
    
    Myy=Mcov.ix[Y.columns,Y.columns].copy()
    Mxx=Mcov.ix[X.columns,X.columns].copy()
    Mzz=Mcov.ix[Z.columns,Z.columns].copy()
    Myz=Mcov.ix[Y.columns,Z.columns].copy()
    Mzy=Mcov.ix[Z.columns,Y.columns].copy()
    Mxz=Mcov.ix[X.columns,Z.columns].copy()
    Mzx=Mcov.ix[Z.columns,X.columns].copy()
    Mxy=Mcov.ix[X.columns,Y.columns].copy()
    Myx=Mcov.ix[Y.columns,X.columns].copy()

    Syy=Myy - zScale * np.dot( np.dot( Myz.values, np.linalg.pinv(Mzz.values)), Mzy.values)
    Sxx=Mxx - zScale * np.dot( np.dot( Mxz.values, np.linalg.pinv(Mzz.values)), Mzx.values)
    Syx=Myx - zScale * np.dot( np.dot( Myz.values, np.linalg.pinv(Mzz.values)), Mzx.values)
    Sxy=Syx.T.copy()

    A=np.dot( np.dot( Sxy.values, np.linalg.pinv(Syy.values)), Syx.values)
    B=Sxx.values.copy()
    #a   vr[:,i] = w[i]        b   vr[:,i]
    w,v=scipy.linalg.eigh(A,b=B)

    rrrBeta=pandas.DataFrame(v.T, index=['vX_%02d'% c for c in range(len(X.columns)) ],
                             columns=X.columns).T.copy() 
    rrrBetaR=rrrBeta.ix[:,-r:].copy()

    vX=X.dot(rrrBetaR)
    vXZ=vX.join(Z).copy()
    
    rrrAlpha=fastLeastSquares(Y,vXZ)
    rrrAlphaZ=rrrAlpha[Z.columns].copy()
    rrrAlphaX=rrrAlpha[vX.columns].copy()
    
    rrrB=rrrAlphaX.dot(rrrBetaR.T).join(rrrAlphaZ)
    
    if dropZ:
        rrrB=rrrB.drop(Z.columns,axis=1).copy()

    #return w,v,olsBeta,vX,X,Y,Z,rrrAlpha,rrrAlphaX,rrrAlphaZ,rrrBeta,rrrBetaR,rrrB 
    #return rrrB

    results.B=rrrB
    results.beta=rrrBetaR
    results.betaFull=rrrBeta
    results.eigs=w
    results.betaX=vX
    results.betaOls=olsB
    results.resid=Y-X.join(Z).dot(rrrB.T)
    results.residOls=Y-X.join(Z).dot(olsB.T)
    results.r2=1.-(results.resid.var())/(Y.var())
    results.r2Ols=1.-(results.residOls.var())/(Y.var())
    
    return results

def ReducedRankRegressionSimple(Y,X,r=None): #NOTE: this is fixed.  Bug in older versions...
    """ 
    Solves Y= alpha*beta'*X  + noise,    by OLS.
    """
    import scipy.linalg
    results=Utilities.Struct()

    if r is None:
        print('Warning: no rank constraint.  Doing ols')
        r=min(X.shape[1],Y.shape[1])

    assert 1 <= r <= min(X.shape[1],Y.shape[1])

    XYZ=Y.join(X,rsuffix='_x').copy()

    # Just to check
    olsB=fastLeastSquares2(Y.copy(),X.copy())

    # Now really compute
    Mcov=XYZ.cov()
    
    Myy=Mcov.ix[Y.columns,Y.columns].copy()
    Mxx=Mcov.ix[X.columns,X.columns].copy()
    #Mzz=Mcov.ix[Z.columns,Z.columns].copy()
    #Myz=Mcov.ix[Y.columns,Z.columns].copy()
    #Mzy=Mcov.ix[Z.columns,Y.columns].copy()
    #Mxz=Mcov.ix[X.columns,Z.columns].copy()
    #Mzx=Mcov.ix[Z.columns,X.columns].copy()
    Mxy=Mcov.ix[X.columns,Y.columns].copy()
    Myx=Mcov.ix[Y.columns,X.columns].copy()

    Syy=Myy #- zScale * np.dot( np.dot( Myz.values, np.linalg.pinv(Mzz.values)), Mzy.values)
    Sxx=Mxx #- zScale * np.dot( np.dot( Mxz.values, np.linalg.pinv(Mzz.values)), Mzx.values)
    Syx=Myx #- zScale * np.dot( np.dot( Myz.values, np.linalg.pinv(Mzz.values)), Mzx.values)
    Sxy=Syx.T.copy()
    
    Aeig=np.dot( np.dot( Sxy.values, np.linalg.pinv(Syy.values)), Syx.values)
    Beig=Sxx.values.copy()
    #a   vr[:,i] = w[i]        b   vr[:,i]
    w,v=scipy.linalg.eigh(Aeig,b=Beig)

    BetaFull=pandas.DataFrame(v.T, 
                             index=['vX_%02d'% c for c in range(len(X.columns)) ],
                             columns=X.columns).T.copy() 
    BetaReduced=BetaFull.ix[:,-r:].copy()

    vX=X.dot(BetaReduced)
    vXfull=X.dot(BetaFull)

    AlphaFull=fastLeastSquares(Y,vXfull)
    AlphaReduced=fastLeastSquares(Y,vX)
    Bfull=AlphaFull.dot(BetaFull.T)
    B=AlphaReduced.dot(BetaReduced.T)

    results.B=B
    results.Bfull=Bfull
    results.Beta=BetaReduced
    results.BetaFull=BetaFull
    results.eigs=w
    results.betaX=vX
    results.Alpha=AlphaReduced
    results.AlphaFull=AlphaFull
    results.B_ols=olsB
    results.resid=Y-X.dot(B.T)
    results.residFull=Y-X.dot(Bfull.T)
    results.residOls=Y-X.dot(olsB.T)
    results.sigma=results.resid.std()
    results.r2=1.-(results.resid.var())/(Y.var())
    results.r2_ols=1.-(results.residOls.var())/(Y.var())
    
    return results

def fitVARSimple(f,freq='M'): #HACK: freq
    """f should be a DataFrame with mean zero.
    fits a VAR(1)
    """
    #TODO: Add a fancier version to take pca of resid.
    Nt,Nf=f.shape
    F=f.copy()
    if Nf==1: #VAR likes to have two or more series.
        newF=0.*f.ix[:,0] + 0.001*randn(Nt)
        newF.name='Fnoise'
        F=F.join(newF)
    fit=sm.tsa.VAR(F).fit(maxlags=1)
    PsiRaw=fit.params[f.columns].copy()
    Psi=PsiRaw.drop('const')
    Psi=Psi.drop([i for i in Psi.index if i.find('Fnoise')>=0])
    Psi=Psi.rename(index=lambda c: c.replace('L1.','')).copy()
    eta=fit.resid[f.columns].copy()
    covResid=fit.resid.cov().ix[f.columns][f.columns].copy()
    return dict(Psi=Psi,beta=Psi,covResid=covResid,F=F,eta=eta,
                resid=fit.resid.ix[:,f.columns].copy(),fit=fit)

def batchOLSRegression(Y,X,intercept=False):
    """Y=a+B*X"""
    logging.debug('batchOLSRegression: begin')
    Y = Y.dropna()
    X = X.dropna()
    inter = Y.index.intersection(X.index)
    X = X.reindex(index=inter)
    Y = Y.reindex(index=inter)
    isallzero = (X == 0.0).all()
    X = X.loc[:,~isallzero]
    allzerocols = isallzero[isallzero].index
    if intercept:
        if 'intercept' in X.columns:
            raise Exception('Cannot add intercept')
        X['intercept'] = 1.0
    beta = pandas.DataFrame(np.linalg.lstsq(X.values, Y.values, rcond=-1)[0],index=X.columns, columns=Y.columns)
    resid = Y - X.dot(beta)

    results=Utilities.Struct()
    if intercept:
        results.a = beta.ix['intercept']
        beta = beta.drop(['intercept'],axis=0)
    else:
        results.a = pandas.Series(0.0, index=beta.index)
    results.B = beta.T
    for c in allzerocols:
        results.B[c] = 0.0
    results.resid = resid
    results.r2 = None
    results.t_stat = None

    logging.debug('batchOLSRegression: end')
    return results

def fitSimplePCA(X,k):
    """ Uses a basic APCA method to extract dynamic factors. May want to do WPCA later.
    The first are most significant.
    X = B*F + resid
    """
    q=Utilities.Struct()
    
    u,s,v=np.linalg.svd(X.values,full_matrices=0)
    F=pandas.DataFrame(u[:,:],index=X.index,
                       columns=['F%02d'%i for i in range(u.shape[1])])
    S=pandas.Series(s,index=F.columns)
    Fall=F.copy()
    Fall/=Fall.std()
    F=Fall.iloc[:,:k].copy()
    
    q.spectrum=S.copy()
    q.Fall=Fall.copy()
    q.F=F
    
    q.olsResults=batchOLSRegression(X,F)
    q.B=q.olsResults.B.copy()
    return q
    
class SimpleKalmanFilter(object):
    """ pyssm works with systems with different notation.  We wrap it.

    y(t) = X(t) * beta_m +  Z(t) * state(t) + R(t) * eps(t); eps(t) ~ N(0, H(t)),
    for t = 1, 2, ..., nobs, and the state is generated by

    state(t + 1) = W(t) * beta_s + T(t) * state(t) + G(t) * eta(t); eta(t) ~ N(0, Q(t))
    for t=1, 2, ..., nobs - 1.

    state(1) ~ N(a1, P1)
    """


    def __init__(self,dates,states,observed,shocks,timevar,**kwargs):
        """
        A wrapper for the kalman filter for a generic state space model.

        (observation) Y(t)   = A * X(t) + R * eps(t) ,     e(t)  ~ N(0,H)
        (state)       X(t+1) = B * X(t) + G * eta(t) ,   eta(t)  ~ N(0,Q)
                                                         t=1,...,Nt-1
                      X(0) ~ N(x0,s0)
        
        
        The coefficients (A,B,R,G,H,Q) may all be time dependent.
        If so, pass in a dictionary of conformable objects into setParams
        (not implemented, yet)
        
        This implementation uses pyssm.
        """
        import pyssm.ssm as ssm
        
        self.dates=list(dates)
        self.states=list(states)
        self.shocks=list(shocks)
        self.observed=list(observed)

        self.nobs=len(dates)
        self.nseries=len(observed)
        self.nstate=len(states)
        self.rstate=len(shocks)

        #TODO:misc args
        self.filter=ssm.Filter(self.nobs,self.nseries,self.nstate,
                               self.rstate,timevar,kwargs) 
        
    def setObservedData(self,Ydf,check_missing=True):
        """
        Y is a DataFrame or a np.array compatible with (dates,observed)
        """
        assert Ydf.shape == (self.nobs,self.nseries)
        self.Y=_df(Ydf,self.dates,self.observed).copy()

        #TODO: 
        self.filter.update_ymat(self.Y.T.copy().values,
                                check_missing=check_missing)
        
    def setParams(self,A,B,R,G,H,Q,x0,s0):
        """
        User must ensure correct dimensions.  #TODO: some checks 
        """
        #HANDLE timevar...
        p=_Struct()
        p.A=A.copy()
        p.B=B.copy()
        p.R=R.copy()
        p.G=G.copy()
        p.H=H.copy()
        p.Q=Q.copy()
        p.x0=x0.copy()
        p.s0=s0.copy()
        
        self.p=p

        self.filter.initialise_system(a1=p.x0.copy().values,
                                      p1=p.s0.copy().values,
                                      zt=p.A.copy().values,
                                      ht=p.H.copy().values,
                                      tt=p.B.copy().values,
                                      gt=p.G.copy().values,
                                      qt=p.Q.copy().values,
                                      rt=p.R.copy().values   )
        

    def smoother(self):
        """TODO: still need to be able to read the filtered (not smoothed) states.
        Also report liklihood values.
        """
        self.filter.smoother()
        self.Xhat=_df(self.filter.ahat.T.copy(),self.dates,self.states)

class FactorExtractor(object):
    def __init__(self,X):
        """Must set DataFrame X"""
        self.X=X.copy()
    
    def getResid(self):
        return self.X-self.F.dot(self.Lambda.T)

    def getStaticFactorsStage1(self,Nstatic=1):
        """ Uses a basic PCA method to extract dynamic factors. May want to do WPCA later."""
        qF=fitSimplePCA(self.X,Nstatic)
        self.stage1=Utilities.Struct()
        self.stage1.F=qF.F.copy()
        self.stage1.Lambda=qF.B.copy()
        self.stage1.olsResults=qF
        self.F=qF.F.copy()
        self.Lambda=qF.B.copy()
        self.resid=self.getResid()
        self.sigma_obs=self.resid.std()


    def fitVAR1(self,lags=1,freq='M'):
        """f should be a DataFrame with mean zero.
        fits a VAR(1) or VAR(lags)
        """
        f=self.F
        
        Nt,Nf=f.shape
        F=d2dt(f.copy())
        if Nf==1: #VAR likes to have two or more series.
            newF=0.*F.ix[:,0] + 0.01*randn(Nt)
            newF.name='Fnoise'
            F=F.join(newF)
        fit=sm.tsa.VAR(F,freq=freq).fit(maxlags=lags) #HACK: lags=1 may be all that works...
        PsiRaw=fit.params[F.columns].copy()
        Psi=PsiRaw.drop('const')
        Psi=Psi.drop([i for i in Psi.index if i.find('Fnoise')>=0])
        Psi=Psi.rename(index=lambda c: c.replace('L1.','')).copy()
        eta=fit.resid[f.columns].copy()              
        covResid=fit.resid.cov().ix[f.columns][f.columns].copy()
        return dict(Psi=Psi,covResid=covResid,F=F,eta=eta,fit=fit)

    def estimateLambda(self,X,F):
        # Needs work...
        fits={}
        for c in X.columns:
            fit=pandas.ols(y=X[c],x=F,intercept=False,nw_lags=3,nw_overlap=False)
            fits[c]=fit

        return fits

class FactorExtractorPCAar1(object):
    def __init__(self,X=None,rho0=[]):
        """Dataframe X is optional."""
        if X is not None:
            self.X=X.copy()
        self.rho0=rho0

    def _X(self,X=None):
        if X is not None:
            return X
        else:
            assert self.X is not None
            return self.X

    def extractFactors(self,X=None,Nstatic=1,estu=None,dates=None,normalize=True):
        """ Preliminary PCA estimate.
            Assume X is already differenced and weighted, and normalized.
        """
        X=self._X(X)

        if normalize:
            X=(X-X.mean())/(X.std())

        if estu is None:
            estu=X.columns
        if dates is None:
            dates=X.index
        #dt=(1.*(dates[1]-dates[0]).days)/252.

        #Extract factors from this, estimate for all 
        Y=X.loc[dates].copy()
        Yestu=Y[estu].copy()
        apca=fitSimplePCA(Yestu,Nstatic)
        F=apca.Fall.iloc[:,:Nstatic].copy()

        VarF=self.fitVAR(F)
        eta=VarF['eta'].copy()
        
        data=Utilities.Struct()
        data.VarF=VarF
        data.Psi=VarF['Psi']
        data.eta=eta
        data.F=F
        data.apca=apca
        #data.Fall=apca.Fall
        #data.S=apca.S

        return data 

    def extractLoadings(self,X,F):
        """ Regress X on rho*lag(X) + Lambda*F """
        
        if self.rho0 is not None:
            rho0=self.rho0
        else:
            rho0=[]

        data=Utilities.Struct()
        data.rho0=list(self.rho0)
        fitDict={}
        betaDict={}
        lambdaDict={}
        rhoDict={}
        residDict={}
        r2Dict={}
        tstatLambdaDict={}
        tstatRhoDict={}
        interceptDict={}

        #Here do autoregression
        for c in X.columns:
            x=F.join(X[c].shift(1)).dropna().copy()
            y=X.loc[x.index][c].copy()
            if c in rho0:
                x=x.drop(c,axis=1).copy()
            xvals = sm.add_constant(x.values)
            #fit=pandas.ols(x=x,y=y,intercept=True) #ADAM Check no intercept?
            fit=sm.OLS(y.values, xvals, missing='drop').fit()
            indx = ['intercept'] + list(x.columns.values)
            fit.beta = pandas.Series(fit.params, index=indx)
            fit.resid = pandas.Series(fit.resid, index=x.index)
            fit.t_stat = pandas.Series(fit.tvalues, index=indx)
            fit.r2_adj = fit.rsquared_adj
            fitDict[c]=fit
            betaDict[c]=fit.beta.drop('intercept').copy()
            interceptDict[c]=fit.beta['intercept']
            if c not in rho0:
                lambdaDict[c]=fit.beta.drop(c).drop('intercept').copy()
                rhoDict[c]=fit.beta[c]
                tstatLambdaDict[c]=fit.t_stat.drop(c).copy()
                tstatRhoDict[c]=fit.t_stat[c]
            else:
                lambdaDict[c]=fit.beta.drop('intercept').copy()
                rhoDict[c]=0.
                tstatLambdaDict[c]=fit.t_stat.copy()
                tstatRhoDict[c]=0.

            residDict[c]=fit.resid.copy()
            r2Dict[c]=fit.r2_adj
            logging.debug('Testing point 3, avg_beta, %.8f, intercept, %.8f, avg_err, %.8f, r2, %.8f',
                    np.average(lambdaDict[c].values, axis=None), 
                    interceptDict[c],
                    np.average(residDict[c].values, axis=None),
                    r2Dict[c])

        data.Lambda=pandas.DataFrame(lambdaDict)
        data.intercept=pandas.Series(interceptDict)
        data.Rho=pandas.Series(rhoDict)
        data.mu=data.intercept/(1-data.Rho)
        data.m=data.intercept.copy()
        data.R2=pandas.Series(r2Dict)
        data.tstat_Lambda=pandas.DataFrame(tstatLambdaDict)
        data.tstat_Rho=pandas.Series(tstatRhoDict)
        data.resid=pandas.DataFrame(residDict)
        data.Xadjusted=(X-data.Rho*X.shift(1)).dropna()
        #TODO: fill first value with prediction given the F.

        fitDict={}
        lambdaDict={}
        residDict={}
        r2Dict={}
        tstatLambdaDict={}

        # Here dont do autoregression #DEFUNCT, no drift m or mu here.

        for c in X.columns:
            x=F.copy()
            y=X.loc[x.index][c].copy()
            fit=sm.OLS(y.values, x.values, missing='drop').fit()
            fitDict[c]=fit
            indx = list(x.columns.values)
            fit.beta = pandas.Series(fit.params, index=indx)
            fit.resid = pandas.Series(fit.resid, index=x.index)
            fit.t_stat = pandas.Series(fit.tvalues, index=indx)
            fit.r2_adj = fit.rsquared_adj
            lambdaDict[c]=fit.beta.copy()
            residDict[c]=fit.resid.copy()
            r2Dict[c]=fit.r2_adj
            tstatLambdaDict[c]=fit.t_stat.copy()
            logging.debug('Testing point 4, avg_beta, %.8f, avg_err, %.8f, r2, %.8f',
                    np.average(lambdaDict[c].values, axis=None),
                    np.average(residDict[c].values, axis=None),
                    r2Dict[c])

        data.Lambda_0=pandas.DataFrame(lambdaDict)
        data.resid_0=pandas.DataFrame(residDict)
        data.R2_0=pandas.Series(r2Dict)
        data.tstat_Lambda_0=pandas.DataFrame(tstatLambdaDict)

        return data 


    def fitVAR(self,f, freq='MS'):
        """f should be a DataFrame with mean zero.
        fits a VAR(1)
        """
        Nt,Nf=f.shape
        F=f.copy()
        if Nf==1: #VAR likes to have two or more series.
            newF=0.*f.iloc[:,0] + 0.01*randn(Nt)
            newF.name='Fnoise'
            F=F.join(newF)
        fit=sm.tsa.VAR(F, freq=freq).fit(maxlags=1)
        PsiRaw=fit.params[f.columns].copy()
        Psi=PsiRaw.drop('const')
        Psi=Psi.drop([i for i in Psi.index if i.find('Fnoise')>=0])
        Psi=Psi.rename(index=lambda c: c.replace('L1.','')).copy()
        eta=fit.resid[f.columns].copy()              
        covResid=fit.resid.cov().loc[f.columns][f.columns].copy()
        return dict(Psi=Psi,covResid=covResid,F=F,eta=eta,fit=fit)


    def estimate(self,X=None,Nstatic=1,estu=None,dates=None,Niters=1):
        X=self._X(X)
        if self.rho0 is None:
            self.rho0=[]

        if estu is None:
            estu=X.columns
        if dates is None:
            dates=X.index

        results=Utilities.Struct()
        
        data0_0=self.extractFactors(X=X,Nstatic=Nstatic,estu=estu,dates=dates)
        data0_1=self.extractLoadings(X,data0_0.F)

        data_0=data0_0
        data_1=data0_1
        for it in range(Niters):
            data_0=self.extractFactors(X=data_1.Xadjusted.dropna().copy(),
                                     Nstatic=Nstatic,estu=estu,normalize=True)#,dates=dates)
            data_1=self.extractLoadings(X.copy(),data_0.F)

        results.data_0=data_0
        results.data_1=data_1
        results.Xadjusted=data_1.Xadjusted.copy()
        results.data0_0=data0_0
        results.data0_1=data0_1
        results.F=data_0.F.copy()
        results.Lambda=data_1.Lambda.copy()
        results.m=data_1.m.copy()
        results.mu=data_1.mu.copy()
        results.Lambda_0=data_1.Lambda_0.copy()
        results.Rho=data_1.Rho
        results.tstat_Lambda=data_1.tstat_Lambda
        results.tstat_Lambda_0=data_1.tstat_Lambda_0
        results.tstat_Rho=data_1.tstat_Rho
        results.resid=data_1.resid
        results.resid_0=data_1.resid_0
        results.R2=data_1.R2
        results.R2_0=data_1.R2_0
        results.eta=data_0.eta.copy()
        results.Psi=data_0.Psi.copy()
        
        return results

class MacroConfigDataLoader(object):
    def __init__(self,csvFile,loadAssets=False,activeOnly=True,needs_sectors=False):
        self.csv_file=csvFile
        self.days_per_month=23 #Yes, this really should be 23.  Used for day/month translation in Kalman filter.
        self.load_assets=loadAssets
        self.active_only=activeOnly
        self.needs_sectors=needs_sectors
        self.monthly_time_lag=datetime.timedelta(45) #HACK: until eff_dt are reliable

    def processMetaData(self,modelDB,marketDB):
        logging.info('processMetaData: begin')
        macroMetaData=pandas.read_csv(self.csv_file,index_col=0)
        macroMetaData['min_dt']=pandas.to_datetime(macroMetaData['min_dt'])
        self.macroMetaDataRaw=macroMetaData.copy()
        if self.active_only:
            macroMetaData=macroMetaData[macroMetaData.active>0].copy()
        
        self.axid2shortname=macroMetaData.shortname.copy()

        self.macroMetaData=macroMetaData 
        self.axioma_ids={
                'macroecon':sorted(macroMetaData[macroMetaData.source=='macroecon'].index),
                'macroidx':sorted(macroMetaData[macroMetaData.source=='macroidx'].index),
                'macrods':sorted(macroMetaData[macroMetaData.source=='macrods'].index),
                }
        
        self.axioma_ids['daily']=sorted([a for a in 
            (self.axioma_ids['macroidx']  + self.axioma_ids['macrods']) 
            if self.macroMetaData.freq[a]=='D' and self.macroMetaData.active[a]==1]) 
        self.axioma_ids['monthly']=sorted([a for a in self.axioma_ids['macroecon'] 
            if self.macroMetaData.freq[a]=='M' and  self.macroMetaData.active[a]==1]) 

        self.macro_rows_all={
                'macroecon':pandas.DataFrame(marketDB.getMacroEconRows()).T,
                'macroidx':pandas.DataFrame(marketDB.getIdxMacroRows()).T,
                'macrods':pandas.DataFrame(marketDB.getDatastreamMacroRows()).T,
            }
        self.macro_rows_all['monthly']=self.macro_rows_all['macroecon'].copy()
        self.macro_rows_all['daily']=self.macro_rows_all['macroidx'].T.join(self.macro_rows_all['macrods'].T).T

        #TODO: add rows from marketDB.getMacroTsDateRange(axids)
        self.macro_rows_raw={
            'daily':self.macro_rows_all['daily'].loc[self.axioma_ids['daily']].copy(),
            'monthly':self.macro_rows_all['monthly'].loc[self.axioma_ids['monthly']].copy(),
            }
        logging.info('processMetaData: end')
        
    def loadRawData(self,riskModel,modelDB,marketDB,dReturnsMin,dModelMin,dModelMax):
        logging.info('loadRawData: begin  %s, %s, %s',dReturnsMin,dModelMin,dModelMax)
        dMin=dReturnsMin
        dMax=dModelMax
        self.processMetaData(modelDB,marketDB)
        self.dates_all=[d.date() for d in pandas.date_range(start=dReturnsMin,end=dModelMax,freq='D')]
        self.dates_model=modelDB.getDateRange(riskModel.rmg,dReturnsMin,dModelMax,excludeWeekend=True)
        self.dates_uniform=_getDaysInUniformMonth(self.dates_model,dReturnsMin,dModelMax,self.days_per_month)
        self.dates_estu=[d for d in self.dates_model if d>=dModelMin]
    
        self.dates_fom=Utilities.getFOMDates(self.dates_model)
        self.dates_eom=Utilities.getEOMDates(self.dates_model)

        #monthlyMacroPanel=pandas.DataFrame(marketDB.getMacroEconPanel(self.axioma_ids['monthly'],dMin)).T.copy()
        monthlyMacroPanel=pandas.DataFrame(marketDB.getMacroEconPanel(self.axioma_ids['monthly'],
            datetime.date(1985,1,1))).T.copy()
        self.macro_ts_raw={
            #'daily':  pandas.DataFrame(marketDB.getMacroTs( self.axioma_ids['daily'],  dMin)).ix[:dMax].copy(),
            'daily':  pandas.DataFrame(marketDB.getMacroTs( self.axioma_ids['daily'],  datetime.date(1988,1,1))),#.ix[:dMax].copy(), #Make sure credit spreads are defined
            'monthly':monthlyMacroPanel,
            }

        logging.info('loading spot rate history')
        #currencies_spot=['USD',]
        currencies_spot=['USD','CHF','GBP','CAD','JPY']
        self.spotRatesDaily=riskModel._loadRiskFreeRateHistoryAsDF(modelDB,marketDB,self.dates_model,currencies=currencies_spot,annualize=False)
        self.spotRatesAnnual=riskModel._loadRiskFreeRateHistoryAsDF(modelDB,marketDB,self.dates_model,currencies=currencies_spot,annualize=True)

        logging.info('loading fun fac ret')
        fundamentalFactorReturns=riskModel._loadFactorReturnsAsDF(dMin,dMax,modelDB,riskModel.fundamentalModel)
        # if True or fundamentalFactorReturns.dropna().count().min()<1000:
            # logging.warning('problem loading us3 fundamental factor returns for macro model.  using horrible hack!!!')
            # #print fundamentalFactorReturns
            # fundamentalFactorReturns=pandas.load('modeldb_pickles_full/fund_fr.pickle')

        styles=sorted([c.name for c in riskModel.fundamentalModel.styles])
        industries=sorted([c.name for c in riskModel.fundamentalModel.industries])
        self.styleReturns=fundamentalFactorReturns[styles].copy()
        self.industryReturns=fundamentalFactorReturns[industries].copy()
        self.marketInterceptReturns=fundamentalFactorReturns[['Market Intercept']].copy()
        self.industryPlusMarketInterceptReturns=self.industryReturns.add(self.marketInterceptReturns['Market Intercept'],
                axis=0).rename(columns=lambda c: 'marketPlus_'+c) 
        fundamentalFactorReturns=fundamentalFactorReturns.join(self.industryPlusMarketInterceptReturns) 
        fundamentalFactorCumReturns=(1.+fundamentalFactorReturns).cumprod()

        logging.info('loading fx rets')
        currencies=['XDR','CHF','GBP','CAD','JPY']
        #currencies=['XDR']
        self.fxReturns=riskModel._loadFXReturnsAsDF(modelDB,marketDB,dMin,dMax,
                currencies=sorted(currencies),base='USD') #NOTE: these are returns,
        fxCumReturns=(1.+self.fxReturns).cumprod()
        #NOTE: there seem to be some missing in the researchdb.

        self.indexD={}
        self.indexD['market intercept']=sorted(self.marketInterceptReturns.columns)
        self.indexD['styles']=sorted(self.styleReturns.columns)
        self.indexD['industries']=sorted(self.industryReturns.columns)
        self.indexD['industriesPlusMarketIntercept']=sorted(self.industryPlusMarketInterceptReturns.columns)
        self.indexD['fx']=sorted(self.fxReturns.columns)
        self.indexD['benchmarks']=sorted([self.axid2shortname.get(k,k)  
            for k,v in self.macro_rows_all['daily'].units.iteritems() 
                if v=='Equity Index' and k in self.axid2shortname.index ])
        self.indexD['yields']=sorted([self.axid2shortname.get(k,k)  
            for k,v in self.macro_rows_all['daily'].units.iteritems() 
                if v.lower().find('yield')>=0 and k in self.axid2shortname.index] )
        self.indexD['vol']=sorted([self.axid2shortname.get(k,k)  
            for k,v in self.macro_rows_all['daily'].units.iteritems() 
                if v.lower().find('vol')>=0 and k in self.axid2shortname.index] )
        self.indexD['commodity']=sorted([self.axid2shortname.get(k,k)  
            for k,v in self.macro_rows_all['daily'].units.iteritems() 
                if v.lower().find('comm')>=0 and k in self.axid2shortname.index] )

        
        
        self.allCumRetsD_master=fundamentalFactorCumReturns.join(fxCumReturns,how='outer'
                ).join(self.macro_ts_raw['daily'].rename(columns=self.axid2shortname),how='outer'
                        ).loc[:dMax].copy() #).ix[dMin:dMax].copy()
        

        if self.load_assets:
            logging.info('loading asset returns')
            # TODO: check me to make sure I dont look into the future...  
            #self.estu=riskModel.loadEstimationUniverse(riskModel.getRiskModelInstance(dMax,modelDB),modelDB)
            estu_data=riskModel.getEstuUnion(self.dates_estu,modelDB)
            self.estu=estu_data[2]
            asset_data=riskModel._loadAssetReturnsAsDF(dMax,modelDB,marketDB,
                    history=300,subIds=self.estu) #HACK
                    #history=len(self.dates_model)+30,subIds=self.estu)
            self.marketCaps_short=asset_data.marketCaps.copy()
            self.excessReturnsDF_short=asset_data.excessReturnsDF.copy() #TODO: should this be recomputed each date?
            mktCap=sorted(self.marketCaps_short.copy())
            mktVol=sorted(self.excessReturnsDF_short[mktCap.index[-1000:]].std())
            #volIQR=mktVol.quantile(0.7)-mktVol.quantile(0.01)
            self.facRetEstu=list(mktVol.iloc[:500].index)
            asset_data=riskModel._loadAssetReturnsAsDF(dMax,modelDB,marketDB,
                    history=min(2040,len(self.dates_model)+30),subIds=self.facRetEstu) #HACK: cache may not be long enough.
            
            self.marketCaps_fac_ret=asset_data.marketCaps.copy()
            self.excessReturnsDF_fac_ret=asset_data.excessReturnsDF.copy() #TODO: should this be recomputed each date?

        logging.info('loadRawData: end')

    def getDataForDates(self,riskModel,dMin,dMax,modelDB,initial=False):
        
        logging.info('getDataForDates: begin')
        modelData=Struct()
        allCumRetsD=self.allCumRetsD_master.loc[dMin:dMax].copy()
        modelData.allCumRetsD_withNaN=allCumRetsD.copy()
        badSeriesIdx=set([])
        badSeriesIdx.update( allCumRetsD.loc[:,(allCumRetsD.count()<0.9*(allCumRetsD.shape[0]))].columns) #HARDCODE

        #badSeriesIdx.update(self.macroMetaData[self.macroMetaData.min_dt>dMin].shortname.values) 
        #BUG:  moodys_ gets left out
        badSeriesIdx.intersection_update(allCumRetsD.columns)
        modelData.badSeriesIdx=badSeriesIdx
        modelData.allCumRetsD=allCumRetsD.drop(badSeriesIdx,axis=1)
        #TODO: still some bad data for some dates. 
        
        modelData.indexD=self.indexD
        modelData.modelDate=dMax
        modelData.date=dMax
        modelData.dates=[d for d in self.dates_model if dMin<=d<=dMax]
        modelData.uniform_dates=[d for d in self.dates_uniform if dMin<=d<=dMax]
        modelData.macroMetaData=self.macroMetaData.copy()
        modelData.master=self #In case we need more

        latency=self.macro_rows_all['daily'].latency.copy()
        latency2=latency[modelData.allCumRetsD.columns].copy()
        for k,v in latency2[latency2>0].iteritems():
            modelData.allCumRetsD.ix[-1*(v):,k]=np.nan
        
        #dMaxLagged=_asdatetime(dMax-self.monthly_time_lag)
        #dMinLagged=_asdatetime(dMin-datetime.timedelta(62)) #Two months slack.
        dMaxLagged=_asdatetime(dMax-datetime.timedelta(0))
        dMinLagged=_asdatetime(dMin-datetime.timedelta(0))
        macroEconPanel=self.macro_ts_raw['monthly']
        macroEconPanel=macroEconPanel[macroEconPanel.dt<=dMaxLagged].copy() #assumes no forecasts using dt 
        #macroEconPanel=macroEconPanel[macroEconPanel.dt>=dMinLagged].copy()
        macroEconPanel=macroEconPanel[macroEconPanel.dt>datetime.datetime(1985,1,1)].copy()
        macroEconPanel=macroEconPanel[macroEconPanel.effDt<=dMaxLagged].copy() #Assumes that the effDt from the db are reliable
        macroEconDataDict={}
        for axId, df in macroEconPanel.groupby('axiomaId'):
            macroEconDataDict[axId]=dt2d(selectLatestRelease(df))
        modelData.macroSeriesMonthly=pandas.DataFrame(macroEconDataDict).copy().rename(columns=self.axid2shortname)
      
        if self.needs_sectors:
            if initial:
                modelData.sectorReturns = self.computeSectorReturns(riskModel, dMin, dMax, modelDB)
            else:
                modelData.sectorReturns = self.computeSectorReturns(riskModel, dMax, dMax, modelDB)

        self.addCompositeDailySeries(modelData)
        self.addDailtyToMothlySeries(modelData)

        logging.info('getDataForDates: end')
        return modelData

    def addCompositeDailySeries(self,modelData):
        df=self.allCumRetsD_master
        series={}
        #NOTE: the tqa data seems to be 10x larger than they should be for these three.
        shortYield=0.01*(df['treasury_yield_13w'])*0.1
        shortYield.name='c_short_yield'
        longYield=(0.01*(df['treasury_yield_10y']))*0.1
        longYield.name='c_long_yield'
        #superLongYield=(0.01*(df['treasury_yield_30y']))*0.1
        #superLongYield.name='c_super_long_yield'
        creditSpreadMoodys=0.01*(df['moodys_baa']-df['moodys_aaa'])
        creditSpreadAxioma=self._buildCreditSpreadComposite(df)
        #creditSpreadMoodys=creditSpreadMoodys.ix[:datetime.date(2011,1,1)].copy()
        #creditSpreadAxioma=creditSpreadAxioma.ix[datetime.date(2009,1,1):].copy()
        #creditSpreadDF=pandas.DataFrame({'moodys':creditSpreadMoodys,'axioma':creditSpreadAxioma})
        #creditSpread=creditSpreadDF.mean(axis=1)
        creditSpread=creditSpreadAxioma

        series['Credit Spread']=creditSpread
        series['Term Spread']=longYield-shortYield
        series['FX Basket']=df['XDR'].copy()
        series['Oil']=df['crude_oil_wti'].copy()
        series['Gold']=df['gsci_gold_spot'].copy()
        series['Commodity']=df['gsci_nonenergy_spot'].copy()
        series['Equity Market']=df['Market Intercept'].copy()
        series['Equity Size']=df['Size'].copy()
        series['Equity Value']=df['Value'].copy()
        series['Equity Liquidity']=df['Liquidity'].copy()
        series['Short Yield']=shortYield
        series['Long Yield']=longYield

        #Create small - big factor return
        # subdf = df[['sp_100_tr', 'sp_600_tr']].rename(columns={'sp_100_tr': 'big', 'sp_600_tr': 'small'})
        # subrets = (subdf/subdf.shift(1) - 1.0).dropna()
        # rets = subrets['small'] - subrets['big']
        # series['Equity Size'] = (rets + 1.0).cumprod()

        modelData.compositeRawD=pandas.DataFrame(series)

    def _buildCreditSpreadComposite(self,df,d0=datetime.datetime(2009,8,1),d1=datetime.datetime(2012,1,1)):
        #NOTE:  the ax_us spreads are defined only back to 2004 or so.  if d0 is too small, they might not be in df.

        #Keep this in case we need to change the definition..
        #zcbIdx=['treasury_cmt_2y','treasury_cmt_5y','treasury_cmt_10y']
        moodysIdx=['moodys_baa','moodys_aaa',]
        axiomaIdx=['ax_us_corp_spread_aaa','ax_us_corp_spread_aa','ax_us_corp_spread_a','ax_us_corp_spread_bbb','ax_us_corp_spread_sub_ig',]
        creditIdx=moodysIdx+axiomaIdx
        yieldsRaw=0.01*df[moodysIdx].copy()
        #spreadsRaw['moodys_baa']=yieldsRaw['moodys_baa']-yieldsRaw['moodys_aaa']
        #spreadsRaw['moodys_aaa']=yieldsRaw['moodys_aaa']-yieldsRaw['treasury_cmt_5y']
        
        cs0=(yieldsRaw['moodys_baa']-yieldsRaw['moodys_aaa']).fillna(method='pad')
        if df.index[-1]<=d0:
            return cs0
        
        spreadsRaw=df[creditIdx].copy()
        spreadsAxioma=spreadsRaw[axiomaIdx].sub(spreadsRaw.ax_us_corp_spread_aaa,axis=0)
        #cs1=spreadsAxioma[['ax_us_corp_spread_a','ax_us_corp_spread_aa','ax_us_corp_spread_bbb']].mean(axis=1)
        #cs1=spreadsAxioma[['ax_us_corp_spread_a','ax_us_corp_spread_aa']].mean(axis=1)
        cs1=spreadsAxioma['ax_us_corp_spread_a']
        cs1=cs1.fillna(method='pad')
        if df.index[0]>=d1:
            return cs1
        
        creditSpreads=cs1.copy()
        creditSpreads.sort_index().loc[:d0]=cs0.sort_index().loc[:d0]
        dates1=list(cs0.sort_index().loc[d0:d1].index)
        alpha=pandas.Series(np.linspace(0.,1.,len(dates1)),dates1)
        creditSpreads.loc[dates1]=alpha*cs1.loc[dates1] + (1.-alpha)*cs0.loc[dates1]

        return creditSpreads     

    

    def computeSectorReturns(self, riskModel, dMin, dMax, modelDB):
        logging.info('computeSectorReturns: begin')
        cls = riskModel.industryClassification

        sectors = cls.getNodesAtLevel(modelDB, 'Sectors')

        #Map each sector to list of industries
        stack = [(s,child) for s in sectors for child in cls.getClassificationChildren(s, modelDB)]
        final = [(s,child) for s,child in stack if child.isLeaf]
        stack = [(s,child) for s,child in stack if not child.isLeaf]
        while len(stack) != 0:
            (s, child) = stack.pop()
            children = cls.getClassificationChildren(child, modelDB) 
            for c in children:
                if c.isLeaf:
                    final.append((s,c))
                else:
                    stack.append((s,c))
        self.sector2industry = defaultdict(list)
        for sector, industry in final:
            self.sector2industry[sector.description].append(industry.description)

        allindustries = self.industryReturns.columns
        dates = [d for d in self.industryReturns.index if dMin <= d <= dMax]
        indexp = {} 
        all_estu = modelDB.getAllRiskModelInstanceESTUs(riskModel.rms_id, min(dates), max(dates))
        for dt in dates:
            estu = all_estu[dt]
            mcapdates = riskModel.getRMDates(dt, modelDB, 25)[0]
            mcaps = modelDB.getAverageMarketCaps(mcapdates, estu, riskModel.numeraire.currency_id)
            weights = ma.sqrt(mcaps)
            nAssets = len(estu)
            C = min(100, int(round(nAssets*0.05)))
            sortindex = ma.argsort(weights)
            ma.put(weights, sortindex[nAssets-C:nAssets],
                        weights[sortindex[nAssets-C]])
            weights = pandas.Series(weights, index=estu).dropna()
            weights /= weights.sum()
            asset_cls = dict((subid,val.classification.description) for subid, val in cls.getAssetConstituents(modelDB, estu, dt).items())
            ind2asset = defaultdict(list)
            for asset, industry in asset_cls.items():
                ind2asset[industry].append(asset)
            indweights = {}
            for industry in allindustries:
                if industry not in ind2asset:
                    indweights[industry] = 0.0
                else:
                    indweights[industry] = weights[ind2asset[industry]].sum()
            indexp[dt] = indweights 
        indexp = pandas.DataFrame(indexp).T

        sectorReturns = {}
        for sector, industries in self.sector2industry.items():
            subexp = indexp[industries]
            total = subexp.sum(axis=1)
            subexp = subexp.div(total,axis=0)
            sectorReturns[sector] = (self.industryReturns.loc[subexp.index,industries] * subexp).sum(axis=1)
        logging.info('computeSectorReturns: end')
        return pandas.DataFrame(sectorReturns)
        
    def addDailtyToMothlySeries(self,modelData):
        modelData.allCumRetsD_interp=expandDatesDF(modelData.allCumRetsD)
        dates=[d for d in modelData.allCumRetsD_interp.index if d.day==1]
        allCumRetsM=modelData.allCumRetsD_interp.loc[dates].copy()
        modelData.allSeriesMonthly=allCumRetsM.join(modelData.macroSeriesMonthly,how='outer')
        


def applyStationaryTransform(X,metadata):
    logging.debug('applyStationaryTransform: begin')
    Y=X.copy()
    dY=0.*Y

    for c in X.columns:
        x=X[c].copy()
        how=metadata.ix[c,'type']
        freq=metadata.ix[c,'freq']
    
        if how.strip()=='percent':
            y=0.01*x
            Y[c]=y
            dY[c]=y-y.shift(1)
        elif how.strip()=='geom':
            dY[c]=(x/x.shift(1)) - 1.
            Y[c]=x.copy()
        elif how.strip()=='geom2':
            tmp=((x/x.shift(1)) - 1.)
            dY[c]=tmp-tmp.shift(1)
            Y[c]=x.copy() # Do we need this?
        elif how.strip()=='geomlog':
            dY[c]=np.log(x/x.shift(1))
            Y[c]=np.log(x) 
        elif how.strip()=='yoy' and freq.strip()=='M':
            x=0.01*x
            y=1.+0.*x
            for i in range(12,len(y.index)):
                y[i]=y.ix[i-12]*(1.+x.ix[i])
            Y[c]=y.copy()
            dY[c]=(y/y.shift(1)) - 1.
        elif how.strip()=='lin':
            Y[c]=x.copy()
            dY[c]=x-x.shift(1)
        else:
            print('warning: transformation type not recognized... ', how)
            assert False
            dY[c]=x.copy()
            Y[c]=x.copy()


    logging.debug('applyStationaryTransform: end')
    return Y.copy(),dY.copy()

def getActiveSeries(dFloor,macroMetaData,marketDB):
    logging.info('getActiveSeries: begin')    
    dFloor=datetime.datetime(dFloor.year,dFloor.month,dFloor.day)
    
    data=Utilities.Struct()

    macroMetaDataActive=macroMetaData.ix[macroMetaData['active']>0]

    dsIds=macroMetaDataActive.ix[macroMetaDataActive['source']=='macrods'].index
    idxIds=macroMetaDataActive.ix[macroMetaDataActive['source']=='macroidx'].index
    macroIds=macroMetaDataActive.ix[macroMetaDataActive['source']=='macroecon'].index

    """
    try:
        print 'trying to load tmp ds and idx csv files.  not for production...'
        dsData=pandas.read_csv('tmp/dsData.csv',index_col=0,parse_dates=True)#.rename(index=lambda d: d.date())
        idxData=pandas.read_csv('tmp/idxData.csv',index_col=0,parse_dates=True)#.rename(index=lambda d: d.date())

        assert set(dsData.columns).issubset(dsIds)
        assert set(idxData.columns).issubset(idxIds)
        print 'done loading csv'
    except:
    """
    if True: #CLEAN ME
        dsData=pandas.DataFrame(marketDB.getMacroTs( dsIds)).ix[:].copy()
        idxData=pandas.DataFrame(marketDB.getMacroTs( idxIds)).ix[:].copy()
        #dsData.to_csv('tmp/dsData.csv')
        #idxData.to_csv('tmp/idxData.csv')

    dsData=dsData.ix[dFloor:].copy()
    idxData=idxData.ix[dFloor:].copy()

    macroEconDataDict={}
    macroEconPanelDict={}
    allpanels = pandas.DataFrame(list(marketDB.getMacroEconPanel(macroIds).values()))

    for axId, dfTmp in allpanels.groupby('axiomaId'):
        dfTmpShort=dfTmp.ix[dfTmp.dt>=dFloor].copy()
        macroEconPanelDict[axId]=dfTmpShort
        macroEconDataDict[axId]=selectLatestRelease(dfTmpShort)

    macroeconDataLatest=pandas.DataFrame(macroEconDataDict).ix[dFloor:]

    data.macroData=macroeconDataLatest
    data.dsData=dsData
    data.idxData=idxData
    data.macroeconPanels=macroEconPanelDict
    data.macroMetaDataActive=macroMetaDataActive
    data.macroMetaDataAll=macroMetaData.copy()
    data.dsIds=dsIds
    data.idxIds=idxIds
    data.macroeconIds=macroIds
    data.dsDataFull=d2dt(expandDatesDF(data.dsData))
    data.idxDataFull=d2dt(expandDatesDF(data.idxData))
    data.dailyDataFull=data.dsDataFull.join(data.idxDataFull).copy()

    data.dailyDataFullM=data.dailyDataFull.ix[
        [d for d in data.dailyDataFull.index if d.day==1]].copy()

    data.axid2shortname=macroMetaData['shortname'].to_dict()
    data.shortnamesAll=macroMetaData['shortname'].copy()
    data.shortnamesActive=macroMetaDataActive['shortname'].copy()

    logging.info('getActiveSeries: end')    
    return data

def getActiveSeriesAsOf(data,dNow):
    logging.info('getActiveSeriesAsOf: begin')    
    dataNow=Utilities.Struct()

    dataNow.dsData=data.dsData.ix[:dNow].copy()
    dataNow.idxData=data.idxData.ix[:dNow].copy()
    
    macroeconDataDict={}
    for c,xp in data.macroeconPanels.items():
        # print c
        macroeconDataDict[c]=selectLatestReleaseBefore(xp,dNow)
    dataNow.macroeconDataDict=macroeconDataDict

    dataNow.macroeconData=pandas.DataFrame(macroeconDataDict).copy()

    dataNow.macroMetaDataActive=data.macroMetaDataActive
    dataNow.macroMetaDataAll=data.macroMetaDataAll
    dataNow.dsIds=data.dsIds
    dataNow.idxIds=data.idxIds
    dataNow.macroeconIds=data.macroeconIds

    dataNow.dailyDataFull=data.dailyDataFull[:dNow].copy()
    dataNow.dailyDataFullM=dataNow.dailyDataFull.ix[
        [d for d in dataNow.dailyDataFull.index if d.day==1]].copy()

    dataNow.monthlyDataFull=dataNow.dailyDataFullM.join(dataNow.macroeconData)
    dataNow.shortnamesAll=data.shortnamesAll.copy()
    dataNow.shortnamesActive=data.shortnamesActive.copy()

    dataNow.monthlyDataFullTransformed, dataNow.monthlyDataFullTransformed_diff=(
        applyStationaryTransform(
            dataNow.monthlyDataFull,dataNow.macroMetaDataAll))

    logging.info('getActiveSeriesAsOf: end')    
    return dataNow


#################
def getMacroMonthlyData(macroMetaDataFile,modelDB,marketDB,dFloor=datetime.datetime(1988,1,1)):
    macroMetaData=pandas.read_csv(macroMetaDataFile,index_col=0)
    data=getActiveSeries(dFloor,macroMetaData,marketDB)
    return data


def getMacroMonthlyDataAsOf(data,date,dStart): #DELETE ME, but not yet.
    """A quick and dirty factor returns fetcher for V5_1"""

    #NOTE: should probably add some series that start around 1993, like retail sales, etc...
    seriesToUse=sorted(set([
     'crude_oil_wti',
    # 'OPEC Oil Basket Price U$/Bbl',
    # 'Reuters Commodities Index',
     # 'SandP GSCI Commodity Excess Return',
     # 'SandP GSCI Commodity Total Return',
     # 'SandP GSCI Energy Excess Return',
     # 'SandP GSCI Energy Spot',
     # 'SandP GSCI Energy Total Return',
     # 'gsci_gold_er',
     # 'gsci_gold_spot',
     # 'gsci_gold_tr',
     # 'SandP GSCI Industrial Metals Exc. Ret.',
     # 'SandP GSCI Industrial Metals Spot',
     # 'SandP GSCI Industrial Metals Tot. Ret.',
     'gsci_nonenergy_er',
     'gsci_nonenergy_spot',
     'gsci_nonenergy_tr', #UNCOMMENT ME
     # 'SandP GSCI Precious Metal Exc. Ret.',
     # 'SandP GSCI Precious Metal Spot',
     # 'SandP GSCI Precious Metal Tot. Ret.',
     # 'SandP GSCI T-Bill Rate TR Total Return',
      'Index(GSCI) Agriculturl Spot',
      'Index(GSCI) Agriculturl Total Return',
     # 'SandP GSCI Four Energy Excess Return',
     # 'SandP GSCI Four Energy Commodities Spot',
     # 'SandP GSCI Four Energy Total Return',
     # 'SandP GSCI Softs Excess Return',
     # 'SandP GSCI Softs Spot',
     # 'SandP GSCI Softs Total Return',
     'treasury_yield_10y',
     # 'treasury_yield_13w',
     # 'treasury_yield_1y',
     # 'treasury_yield_20y',
     # 'treasury_yield_2y',
     # 'treasury_yield_2y',
     #'treasury_yield_3m',
     #'treasury_yield_5y',
     #'treasury_yield_6m',
     #'cboe_gold_ew_idx',
     # 'djia_idx_daily',
     # 'MORGAN STANLEY CONSUMER INDEX',
     # 'MORGAN STANLEY CYCLICAL INDEX',
     # 'NASDAQ 100 INDEX',
     # 'NYSE TOTAL VOLUME',
     # 'RUSSELL 1000 GROWTH',
     # 'RUSSELL 1000 GROWTH INCL DIVIDENDS',
     # 'RUSSELL 1000 INDEX',
     # 'RUSSELL 1000 INDEX INCL DIVIDENDS',
     # 'RUSSELL 1000 VALUE',
     # 'RUSSELL 1000 VALUE INCL DIVIDENDS',
     # 'RUSSELL 2000 GROWTH',
     # 'RUSSELL 2000 GROWTH INCL DIVIDENDS',
     # 'RUSSELL 2000 INDEX',
     # 'RUSSELL 2000 INDEX INCL DIVIDENDS',
     # 'RUSSELL 2000 VALUE',
     # 'RUSSELL 2000 VALUE INCL DIVIDENDS',
     # 'RUSSELL 2500 GROWTH',
     # 'RUSSELL 2500 GROWTH - INCL DIVIDENDS',
     # 'RUSSELL 2500 INDEX',
     # 'RUSSELL 2500 INDEX INCL DIVIDENDS',
     # 'RUSSELL 2500 VALUE',
     # 'RUSSELL 2500 VALUE - INCL DIVIDENDS',
      'RUSSELL 3000 GROWTH',
     # 'RUSSELL 3000 GROWTH INCL DIVIDENDS',
      'RUSSELL 3000 INDEX',
     # 'RUSSELL 3000 INDEX INCL DIVIDENDS',
      'RUSSELL 3000 VALUE',
     # 'RUSSELL 3000 VALUE INCL DIVIDENDS',
     # 'RUSSELL MIDCAP GROWTH',
     # 'RUSSELL MID CAP GROWTH  INCL DIVIDENDS',
     # 'RUSSELL MIDCAP INDEX ',
     # 'RUSSELL MID CAP INDEX INCL DIVIDENDS',
     # 'RUSSELL MID CAP VALUE',
     # 'RUSSELL MID CAP VALUE INCL DIVIDENDS',
     # 'RUSSELL TOP 200 GROWTH',
     # 'RUSSELL TOP 200 GROWTH - INCL DIVIDENDS',
     # 'RUSSELL TOP 200 INDEX',
     # 'RUSSELL TOP 200 INDEX - INCL DIVIDENDS',
     # 'RUSSELL TOP 200 VALUE',
     # 'RUSSELL TOP 200 VALUE - INCL DIVIDENDS',
      'SandP 100 INDEX',
     # 'SandP 100 TOTAL RETURN INDEX',
     # 'SandP 400 INDEX - MIDCAP',
     # 'SandP 400 MIDCAP TOTAL RETURN',
      'SandP 500 INDEX',
     # 'SandP 500 TOTAL RETURN INDEX',
     # 'SandP 600 INDEX - SMALL CAP',
     # 'SandP 600 TOTAL RETURN INDEX',
     # 'SandP BANKING INDEX',
     # 'SandP CHEMICAL INDEX',
     # 'SandP CONSUMER STAPLES SECTOR INDEX',
     # 'SandP ENERGY SECTOR INDEX',
     # 'SandP EQUAL WEIGHT INDEX',
     # 'SandP FINANCIAL SECTOR INDEX',
     # 'SandP HEALTHCARE INDEX',
     # 'SandP INDUSTRIAL AVG',
     # 'SandP INDUSTRIALS SECTOR INDEX',
     # 'SandP INFORMATION TECH SECTOR INDEX',
     # 'SandP INSURANCE INDEX',
     # 'SandP MATERIALS SECTOR INDEX',
     # 'SandP MIDCAP 400 SECTOR INDEX - FINANCIALS SEC',
     # 'SandP RETAIL INDEX',
     # 'SandP TELECOM SERVICES SECTOR INDEX',
     # 'SandP UTILITIES SECTOR INDEX',
      'vix_new',
      #'vix_old',
      'moodys_aaa',
      'moodys_baa',
      'chicago_pmbb',
     # 'constr_exp',
      'cci',
      'cpi',
      'cpi_core',
     # 'cpi_urban',
      'disp_pers_inc',
      #'djia',
      'emp_nonfarm',
     # 'home_sales_single',
     # 'exp_price_idx_com',
     # 'exports_fas',
     # 'imp_price_idx_com',
     # 'imports_fas',
      'indprod',
     # 'indprod_mfg',
     # 'ind_prod_total_idx',
      'libor_3m',
      'ism_pmi',
      'nahb_housing_mkt_idx',
      'hous_permit',
      'houst_private',
      'pce',
      'pce_core',
      'pers_inc',
      'pers_sav_pct',
      'phil_outlook_svy',
      'ppi',
      'ppi_core',
     # 'house_sales_single',
     # 'terms_of_trade',
      'leading_econ_idx',
      'civ_empl',
      'broad_fx',
       'treas_yield_3m',
       'treas_yield_20y',
     # 'unrate_16_plus',
      'unrate',
      'retail_sales_total'
     # 'constr_starts_dwel'
    ]))

    cmdyIds= ['Index(GSCI) Agriculturl Spot_diff',
              'Index(GSCI) Agriculturl Total Return_diff',
              'crude_oil_wti_diff',
              'gsci_nonenergy_er_diff',
              'gsci_nonenergy_spot_diff',
              'gsci_nonenergy_tr_diff',
              ]

    bmIds=[
        'RUSSELL 3000 INDEX_diff',
        'SandP 100 INDEX_diff',
        'SandP 500 INDEX_diff',
        #'djia_diff',
        'broad_fx_diff'
        ]

    macroGrowthIds=[
     #'disp_pers_inc_diff',
     'indprod_diff',
     'ism_pmi_diff',
     'leading_econ_idx_diff',
     'pce_diff',
     'pce_core_diff',
     #'pers_inc_diff',
     #'pers_sav_pct_diff',
    ]

    macroConfIds=[ #May want to include vix later.
     'cci_diff',
     'chicago_pmbb_diff',
     'phil_outlook_svy_diff',
    ]

    macroInflationIds=[
     'cpi_diff',
     'cpi_core_diff',
     'ppi_diff',
     'ppi_core_diff',
    ]

    macroEmploymentIds=[
     'civ_empl_diff',
     'emp_nonfarm_diff',
     'unrate_diff',
    ]

    macroEmploymentIds=[
     'civ_empl_diff',
     'emp_nonfarm_diff',
     'unrate_diff',
    ]

    macroHousingIds=[
     'hous_permit_diff',
     'houst_private_diff',
     'nahb_housing_mkt_idx_diff',
    ]

    allIds=(macroGrowthIds+macroInflationIds+macroEmploymentIds+macroHousingIds+macroConfIds
            +bmIds+cmdyIds)

    date=datetime.datetime(date.year,date.month,date.day)
    results=Utilities.Struct()
    dataNow=getActiveSeriesAsOf(data,date)
    results.dataNow=dataNow
    macroDataM=dataNow.monthlyDataFullTransformed.rename(columns=data.axid2shortname)
    macroDataM_diff=dataNow.monthlyDataFullTransformed_diff.rename(columns=data.axid2shortname)

    Ytransformed=macroDataM[seriesToUse].copy()
    Ytransformed_diff=macroDataM_diff[seriesToUse].rename(columns=lambda c:c+'_diff').copy()
    Yall=Ytransformed.join(Ytransformed_diff)
    YYall=Yall[allIds].ix[:].dropna().copy()

    spreads=pandas.DataFrame({
            'credit_spread':Ytransformed['moodys_baa']-Ytransformed['moodys_aaa'],
            'term_spread':Ytransformed['treas_yield_20y']-Ytransformed['treas_yield_3m'],
            'ted_spread':Ytransformed['libor_3m']-Ytransformed['treas_yield_3m'],
            }).copy()
    spreads_diff=(spreads-spreads.shift(1)).rename(columns=lambda c:c+'_diff').copy()
    Yall=Yall.join(spreads).join(spreads_diff).copy()

    results.YYall=YYall.copy()
    results._Yall=Yall.copy()
    results.macroDataM=macroDataM.copy()
    results.macroDataM_diff=macroDataM_diff.copy()

    return results
