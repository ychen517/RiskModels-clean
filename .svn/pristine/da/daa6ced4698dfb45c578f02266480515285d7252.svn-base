"""
units:'lin', 'chg', 'ch1', 'pch', 'pc1', 'pca', 'cch', 'cca', 'log'  
lin = Levels (No transformation)
chg = Change
ch1 = Change from Year Ago
pch = Percent Change
pc1 = Percent Change from Year Ago
pca = Compounded Annual Rate of Change
cch = Continuously Compounded Rate of Change
cca = Continuously Compounded Annual Rate of Change
log = Natural Log

frequency:  'd', 'w', 'bw', 'm', 'q', 'sa', 'a', 'wef', 'weth', 
            'wew', 'wetu', 'wem', 'wesu', 'wesa', 'bwew', 'bwem' 

aggregation_method: 'avg', 'sum', 'eop' 

output_type: An integer that indicates an output type.

    integer, optional, default: 1
    One of the following values: '1', '2', '3', '4'
    1 = Observations by Real-Time Period
    2 = Observations by Vintage Date, All Observations
    3 = Observations by Vintage Date, New and Revised Observations Only
    4 = Observations, Initial Release Only

vintage_dates: comma separated string of YYYY-MM-DD formatted dates in history
"""

try:
    import fred
except:
    print('Error: please install package (fred 3.0)')
    print('https://pypi.python.org/pypi/fred')
    assert False


import numpy as np
import datetime
import dateutil #TODO: replace this, may not be installed.
from optparse import OptionParser
import sys
import pandas
from riskmodels.Utilities import Struct

def _parseFloat_orNaN(valueStr):
    try:
        value=float(valueStr)
    except:
        value=np.nan
    return value

def fixPanelConvention(df):
    df=df.copy().join(pandas.DataFrame({'dt':[],'eff_dt':[],'change_seq':[]},dtype='object'))

    newFormat={}
    for d,v in df.groupby('date'):
        f=v.copy()
        f.sort('realtime_start')
        for cs,i in enumerate(f.index):
            df.ix[i,'change_seq']=cs
            df.ix[i,'dt']=d
            df.ix[i,'eff_dt']=f.ix[i,'realtime_start']
    return df

def getOnePanel(fid,dMin):
    vintage = Struct(fred.vintage(fid))
    dates = ','.join([d for d in vintage.vintage_dates if d>=dMin])
    result = Struct(fred.observations(fid, vintage_dates=dates,observation_start=dMin))
    df=pandas.to_datetime(pandas.DataFrame(result.observations).rename(index=lambda i: str(i)) )
    df[df=='.']='nan'
    df['value']=df['value'].astype(float)

    result.fid=fid
    result.raw_series=pandas.Series(fred.series(fid)['seriess'][0])
    result.fred_title=result.raw_series['title']
    #result.observations=None
    
    return df,result


def doMain(options):
    dMin=options.observationStart
    fid=options.fid

    FRED_API_KEY='03e9ac6bfe07e3c67725de2d8fb0a887'
    fred.key(FRED_API_KEY)
    
    df,result=getOnePanel(fid,dMin)
    df2=fixPanelConvention(df)
    print(result.fid)
    print(result.fred_title)
    print(result.raw_series)
    #print result.raw_series['notes']
    units=result.units
    df2.to_csv(options.outputDir+'/'+fid+'_'+units+'.csv')





if __name__=='__main__':

    argv = sys.argv

    usage = 'usage: %prog [options] FID'
    parser = OptionParser(usage=usage)
    parser.add_option("--start", dest="observationStart", default='1980-01-01',
                      help="Date YYYY-MM-DD (default: %default))")
    parser.add_option("--output_dir", dest="outputDir", default='./',
                      help="")
    parser.add_option("--units", dest="units", default='lin',
                      help="One of:lin,chg,ch1,pch,pc1,pca,cch,cca,log (default: %default)")
    (options, args) = parser.parse_args(argv)
    
    print(options)
    print(args)

    assert len(args)==2, 'need at least one fed/fred series_id. ' \
        'See http://research.stlouisfed.org/fred2/'

    options.fid=args[1]

    doMain(options)

