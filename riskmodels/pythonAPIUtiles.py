import datetime
import pyutils.parseflatfiles as parseff
import numpy as np
import pandas

def _convertToFrame(f):
    import pandas
    wasdf = True
    if not isinstance(f, pandas.DataFrame):
        f = pandas.DataFrame({'tmp': f})
        wasdf = False
    return f, wasdf 

def _computeQvec(rm, bm, factorWeight=1.0, specificWeight=1.0):
    import pandas
    sr = pandas.Series(rm.specrisk)
    assets = bm.index | sr.index
    sr = sr.reindex(index=assets).fillna(value=0.0)
    bm = bm.reindex(index=assets).fillna(value=0.0)
    factors = [f for f,i in sorted(rm.factorindexdict.items(), key=lambda x: x[1])]
    cov = pandas.DataFrame(rm.covmat,index=factors,columns=factors)
    B = pandas.DataFrame(rm.expmatT, dtype=float).reindex(index=assets,columns=factors).fillna(value=0.0)
    r = factorWeight*B.dot(cov.dot(B.T.dot(bm))) + specificWeight*bm.mul(sr**2, axis=0)
    for (asset1,asset2), val in rm.speccov.items():
        r.loc[asset1] += specificWeight*bm.loc[asset2] *val
        r.loc[asset2] += specificWeight*bm.loc[asset1] *val
    return r

def computeTotalRisk(rm, f, factorWeight=1.0, specificWeight=1.0):
    """ Compute risk of all portfolios in f
        
        Parameters
        ----------
        rm : model.RiskModel
             risk model to use in predicted beta computation
        f : dict, model.Attribute, pandas.Series, or pandas.DataFrame
            Represents the portfolio(s) whose risk should be computed
        
        Returns
        -------
        x : float or pandas.Series with predicted risk values

    """
    f, wasdf = _convertToFrame(f)
    r = _computeQvec(rm, f, factorWeight, specificWeight)
    f = f.reindex(index=r.index).fillna(value=0.0) 
    result = pandas.Series(np.sqrt(np.diag(r.T.dot(f))),index=f.columns)
    if not wasdf:
        result = result['tmp']
    return result

def getPortTotalriskOld(dates,port,modelmnemonic,modelPath):
    """ This function takes a portfolio (modelIDString(without initial D)- weight map) 
        a model menmonic a list of dates to construct a daterisklist (same order as 
        the dates list) 
        Use FaltFiles and Python API""" 
    dateRiskList = list()
    import popt
    import pyutils.populateenv as populateenv
    for date in dates: 
        data = populateenv.populateEnvFromFlatFile(modelPath+'%04d/%02d/%s.%04d%02d%02d'%(date.year,date.month,modelmnemonic,date.year,date.month,date.day),
                                                   assets = port)
#        folio = populateenv.populatePortfolio(data)
        dateRiskList.append(popt.computeTotalRisk(data.rm,data.benchmark))
        del data
    popt.Environment.tearDownAPI()
    return dateRiskList 

def getPortTotalrisk(dates,port,modelmnemonic,modelPath):
    """ This function takes a portfolio (modelIDString(without initial D)- weight map) 
        a model menmonic a list of dates to construct a daterisklist (same order as 
        the dates list) 
        Use FaltFiles and Python API""" 
    dateRiskList = list()
    for date in dates: 
        rm = parseff.parseFactorRiskModel(modelPath+'%04d/%02d/%s.%04d%02d%02d'%(date.year,date.month,modelmnemonic,date.year,date.month,date.day),
                                                   assets = port)
        dateRiskList.append(computeTotalRisk(rm,port))
    return dateRiskList 
