
from optparse import OptionParser
import sys
import logging
import datetime
import os
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
import cufflinks
import colorlover as cl
import pandas
import riskmodels
from riskmodels import Utilities
from riskmodels import ModelDB
from marketdb import MarketDB
from riskmodels import RiskModels
py.sign_in('Axioma01', 'rbgwcvrp8i')



def getExpData(dt, rm, modelDB, marketDB, newFormat=True):
    rmi = rm.getRiskModelInstance(dt, modelDB)
    if rmi is None:
        return None
    sf = modelDB.getSubFactorsForDate(dt, rm.styles)
    descDict = {e[0]: e[1] for e in modelDB.getAllDescriptors()}
    descList = [item for sublist in rm.DescriptorMap.values() for item in sublist]

    meanDict = {}
    stdDict = {}
    descMeanDict = {}
    descStdDict = {} 
    if newFormat:
        for s in sf:
            sql = """select dt, mean, stdev 
                     from rms_stnd_exp 
                     where rms_id=%.0f 
                       and sub_factor_id=%.0f
                     order by dt """ % (rm.rms_id, s.subFactorID)
            modelDB.dbCursor.execute(sql)
            res = [r for r in modelDB.dbCursor.fetchall()]
            meanDict[s.factor.name] = pandas.Series([r[1] for r in res], 
                    index=[r[0].date() for r in res])
            stdDict[s.factor.name] = pandas.Series([r[2] for r in res],
                    index=[r[0].date() for r in res])
        for d in descList:
            sql = """select dt, mean, stdev 
                     from rms_stnd_desc 
                     where rms_id=%.0f 
                       and descriptor_id=%.0f
                     order by dt """ % (rm.rms_id, descDict[d])
            modelDB.dbCursor.execute(sql)
            res = [r for r in modelDB.dbCursor.fetchall()]
            descMeanDict[d] = pandas.Series([r[1] for r in res], 
                    index=[r[0].date() for r in res])
            descStdDict[d] = pandas.Series([r[2] for r in res],
                    index=[r[0].date() for r in res])
             
    else:
        for s in sf:

            sql = """select dt, value 
                     from rms_stnd_mean 
                     where rms_id=%.0f 
                       and sub_factor_id=%.0f  
                     order by dt """ % (rm.rms_id, s.subFactorID)
            modelDB.dbCursor.execute(sql)
            res = [r for r in modelDB.dbCursor.fetchall()]
            meanDict[s.factor.name] = pandas.Series([r[1] for r in res], 
                    index=[r[0].date() for r in res])

            sql = """select dt, value 
                     from rms_stnd_stdev 
                     where rms_id=%.0f 
                       and sub_factor_id=%.0f  
                     order by dt """ % (rm.rms_id, s.subFactorID)
            modelDB.dbCursor.execute(sql)
            res = [r for r in modelDB.dbCursor.fetchall()]
            stdDict[s.factor.name] = pandas.Series([r[1] for r in res], 
                    index=[r[0].date() for r in res])

        for d in descList:
            sql = """select dt, mean, stdev 
                     from rms_stnd_stats 
                     where rms_id=%.0f and 
                       descriptor_id=%.0f 
                     order by dt""" % (rm.rms_id, descDict[d])
            modelDB.dbCursor.execute(sql)
            res = [r for r in modelDB.dbCursor.fetchall()]
            descMeanDict[d] = pandas.Series([r[1] for r in res], 
                    index=[r[0].date() for r in res])
            descStdDict[d] = pandas.Series([r[2] for r in res],
                    index=[r[0].date() for r in res])
 
    
    return (pandas.DataFrame.from_dict(meanDict),
            pandas.DataFrame.from_dict(stdDict),
            pandas.DataFrame.from_dict(descMeanDict),
            pandas.DataFrame.from_dict(descStdDict))


def initializeModel(name, date, modelDB, marketDB):
    rmc = riskmodels.getModelByName(name)
    rm = rmc(modelDB, marketDB)
    rm.setFactorsForDate(date, modelDB)
    return rm

def getEstuStats(rm, rmMH, rmSH, date, modelDB):
    estus = {}
    rmi = rm.getRiskModelInstance(date, modelDB)
    if rmi is None:
        return None
    rm.setFactorsForDate(date, modelDB)
    estus['FL'] = rm.loadEstimationUniverse(rmi, modelDB)
    for name, model in [('MH', rmMH), ('SH', rmSH)]: 
        rmi = model.getRiskModelInstance(date, modelDB)
        if rmi is None:
            continue
        rm.setFactorsForDate(date, modelDB)
        estus[name] = rm.loadEstimationUniverse(rmi, modelDB)

    stats = {}
    for k, estu in estus.items():
        stats['%s Length' % k] = len(estu)
    stats['MH & FL Length'] = len(set(estus['FL']) & set(estus['MH']))
    stats['SH & FL Length'] = len(set(estus['FL']) & set(estus['SH']))
    return stats


def runmain(argv=None):
    if argv == None:
        argv = sys.argv

    usage = "usage: %prog [options] date"
    cmdlineParser = OptionParser(usage=usage)
    Utilities.addModelAndDefaultCommandLine(cmdlineParser)
    cmdlineParser.add_option("--outdir", action="store",
                             default='qaresults', dest="outdir",
                             help="Output directory name")
    (options, args) = cmdlineParser.parse_args(argv)
    
    modelDt = dt = datetime.datetime.strptime(args[1], '%Y-%m-%d').date()
     
    modelDB = ModelDB.ModelDB(sid=options.modelDBSID, user=options.modelDBUser,
                              passwd=options.modelDBPasswd)
    marketDB = MarketDB.MarketDB(sid=options.marketDBSID,
                                 user=options.marketDBUser, passwd=options.marketDBPasswd)

    rm = initializeModel(options.modelName, modelDt, modelDB, marketDB)
    if rm.name == 'US4AxiomaFL':
        rmMH = initializeModel('USAxioma2016MH', modelDt, modelDB, marketDB)
        rmSH = initializeModel('USAxioma2016SH', modelDt, modelDB, marketDB)
    elif rm.name == 'WW4AxiomaFL':
        rmMH = initializeModel('WWAxioma2017MH', modelDt, modelDB, marketDB)
        rmSH = initializeModel('WWAxioma2017SH', modelDt, modelDB, marketDB)
    else:
        logging.error('Not a valid FL model name')

    # get descriptors
    descFacMap = {v[0]: k for k, v in rm.DescriptorMap.items()}
    descs = []
    for k, v in rm.DescriptorMap.items():
        descs.extend(v)
       
    # get single descriptor factors for comparison
    factorMapMH = {}
    for k, v in rmMH.DescriptorMap.items():
        if len(v) == 1:
            factorMapMH[k] = descFacMap[v[0]]
    factorMapSH = {}
    for k, v in rmSH.DescriptorMap.items():
        if len(v) == 1:
            factorMapSH[k] = descFacMap[v[0]]

  
    estuStats = getEstuStats(rm, rmMH, rmSH, modelDt, modelDB)


    # write stats to outdir
    if not os.path.exists(options.outdir):
        os.makedirs(options.outdir)

    # compare exposures for single descriptor factors
    subdir = os.path.join(options.outdir, 'expComparison')
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    rmi = rm.getRiskModelInstance(modelDt, modelDB)
    expFL = rm.loadExposureMatrix(rmi, modelDB).toDataFrame()
    rmiMH = rmMH.getRiskModelInstance(modelDt, modelDB)
    expMH = rmMH.loadExposureMatrix(rmiMH, modelDB).toDataFrame()
    idx = expFL.index & expMH.index
    for mhcol, flcol in factorMapMH.items():
        print(mhcol, flcol)
        f = expFL[flcol].reindex(index=idx)
        e = expMH[mhcol].reindex(index=idx)
        print('FL %s has %.0f null values' % (flcol, f.isnull().sum()))
        assert(e.isnull().sum() == 0)
        if f.isnull().sum() > 0:
            idx = f[~f.isnull()].index
            f = f.reindex(index=idx)
            e = e.reindex(index=idx)
        assert(f.isnull().sum() == 0)

        tr = go.Scatter(x=e.values, y=f.values, mode='markers')
        layout = go.Layout(title='%s (y) vs %s (x) Scatter' % (flcol, mhcol))
        fig = go.Figure(data=[tr], layout=layout)
        fname = os.path.join(subdir, '%s-scatter.html' % flcol)
        plotly.offline.plot(fig, filename=fname, auto_open=False)

        diff = (f - e).abs()
        tr = go.Histogram(x=diff.values, histnorm='probability')
        layout = go.Layout(title='abs(%s - %s): Median diff %.3e' % (mhcol, flcol, diff.median()))
        fig = go.Figure(data=[tr], layout=layout)
        fname = os.path.join(subdir, '%s-histDifferences.html' % flcol)
        plotly.offline.plot(fig, filename=fname, auto_open=False)
        print('median diff abs(%s - %s) is %.3e' % (flcol, mhcol, diff.median()))
        
    # compare standardization stats over time  
    meanDictFL, stdDictFL, descMeanDictFL, descStdDictFL = \
            getExpData(modelDt, rm, modelDB, marketDB)

    meanDictMH, stdDictMH, descMeanDictMH, descStdDictMH = \
            getExpData(modelDt, rmMH, modelDB, marketDB, newFormat=False)

    meanDictSH, stdDictSH, descMeanDictSH, descStdDictSH = \
            getExpData(modelDt, rmSH, modelDB, marketDB, newFormat=False)

    bl = cl.scales['11']['qual']['Paired'][1]
    gr = cl.scales['11']['qual']['Paired'][3]
    rd = cl.scales['11']['qual']['Paired'][5]
    subdir = os.path.join(options.outdir, 'StndStats')
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    for mhcol, flcol in factorMapMH.items():
        fig = tools.make_subplots(rows=2, cols=1, 
            subplot_titles=['%s (%s): %s' % (mhcol, flcol, a) for a in ['Mean', 'Std']])
        tr1 = go.Scatter(x=meanDictFL[flcol].index, y=meanDictFL[flcol].values, 
                         line=dict(color=bl), name='US4FL Mean')
        tr2 = go.Scatter(x=meanDictMH[mhcol].index, y=meanDictMH[mhcol].values,
                         line=dict(color=rd, dash='dot'), name='US4MH Mean')
        fig.append_trace(tr1, 1, 1)
        fig.append_trace(tr2, 1, 1)

        tr1 = go.Scatter(x=stdDictFL[flcol].index, y=stdDictFL[flcol].values, 
                         line=dict(color=bl), name='US4FL Stdev')
        tr2 = go.Scatter(x=stdDictMH[mhcol].index, y=stdDictMH[mhcol].values,
                         line=dict(color=rd, dash='dot'), name='US4MH Stdev')
        fig.append_trace(tr1, 2, 1)
        fig.append_trace(tr2, 2, 1)
        fname = os.path.join(subdir, 'ExpStnd-%s-%s.html' % (mhcol.replace(' ', ''), flcol.replace(' ', '')))
        plotly.offline.plot(fig, filename=fname, auto_open=False)

    # plot descriptor stnd stats
    for c in sorted(list(set(descMeanDictFL.columns).intersection(set(descMeanDictMH.columns)))):
        fig = tools.make_subplots(rows=2, cols=1, 
            subplot_titles=['%s: %s' % (c, a) for a in ['Mean', 'Std']])
        tr1 = go.Scatter(x=descMeanDictFL[c].index, y=descMeanDictFL[c].values, 
                         line=dict(color=bl), name='US4FL Mean')
        tr2 = go.Scatter(x=descMeanDictMH[c].index, y=descMeanDictMH[c].values,
                         line=dict(color=rd, dash='dot'), name='US4MH Mean')
        fig.append_trace(tr1, 1, 1)
        fig.append_trace(tr2, 1, 1)

        tr1 = go.Scatter(x=descStdDictFL[c].index, y=descStdDictFL[c].values, 
                         line=dict(color=bl), name='US4FL Stdev')
        tr2 = go.Scatter(x=descStdDictMH[c].index, y=descStdDictMH[c].values,
                         line=dict(color=rd, dash='dot'), name='US4MH Stdev')
        fig.append_trace(tr1, 2, 1)
        fig.append_trace(tr2, 2, 1)
        fname = os.path.join(subdir, 'DescStnd-%s.html' % c.replace(' ', ''))
        plotly.offline.plot(fig, filename=fname, auto_open=False)
         


if __name__ == '__main__':
    runmain()

