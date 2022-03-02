
import logging
import numpy.ma as ma
import numpy
import datetime
from riskmodels import LegacyUtilities as Utilities
from riskmodels import Matrices

def synchronise_returns(rmgList, modelDate, modelDB, synchMarkets, debugReporting=False):
    """ Uses a VAR-based technique to synchronise market returns
    for early-closing markets with those of later markets
    """
    logging.debug('synchronise_returns: begin')
    # Bucket markets into regions
    regionMarketMap = dict()
    for rmg in rmgList:
        if rmg.region_id == 11:
            rmg.region_id = 12
        if rmg.region_id not in regionMarketMap:
            regionMarketMap[rmg.region_id] = list()
        regionMarketMap[rmg.region_id].append(rmg)
    # Re-order list of RMGs by region (purely for convenience of debugging output)
    rmgList = []
    for region_id in regionMarketMap.keys():
        rmgList.extend(regionMarketMap[region_id])

    # Load history of local market returns
    dates = modelDB.getDates(synchMarkets, modelDate, 250)
    if dates[-1] != modelDate:
        return dict()
    dates.reverse()
    marketReturnHistory = modelDB.loadRMGMarketReturnHistory(dates, rmgList, useAMPs=False)
    marketReturnHistory.data = Utilities.multi_mad_data(
            marketReturnHistory.data,)
    synchedMarketReturn = computeVARParameters(
        rmgList, modelDate, modelDB, marketReturnHistory.data, synchMarkets,
        debugReporting=debugReporting)
    if len(synchedMarketReturn) == 0:
        return dict()

    if debugReporting:
        rmgNames = [r.description.replace(',','') for r in rmgList]
        dtList = [str(dates[0])]
        outFile = 'tmp/mktret-%s.csv' % modelDate
        Utilities.writeToCSV(marketReturnHistory.data[:,0][numpy.newaxis,:], outFile,
                rowNames=dtList, columnNames=rmgNames)
        outFile = 'tmp/mktretAdjusted-%s.csv' % modelDate
        Utilities.writeToCSV(synchedMarketReturn[numpy.newaxis,:], outFile,
                rowNames=dtList, columnNames=rmgNames)
    
    # Set up dict of adjustments
    adjustments = synchedMarketReturn - marketReturnHistory.data[:,0]
    adjust = dict((rmg, adj) for (rmg, adj) in zip(rmgList, adjustments)
                  if adj is not ma.masked)
    for (rmg, adj) in zip(rmgList, adjustments):
        logging.info('Date: %s, Market %s: %s, Adjustment: %.8f', modelDate, rmg.mnemonic, rmg.description, adj)

    if debugReporting:
        rmgNames = [r.description.replace(',','') for r in rmgList]
        rowNames = ['%s' % modelDate]
        outFile = 'tmp/RetTim-%s.csv' % modelDate
        data = adjustments[numpy.newaxis,:]
        Utilities.writeToCSV(data, outFile, rowNames=rowNames, columnNames=rmgNames)

    return adjust

def computeVARParameters(rmgList, date, modelDB, marketReturnHistory,
                        synchMarkets, maxLags=1, debugReporting=False):
    # Uses a VAR technique to predict market returns for
    # early markets based on those of the later markets

    # Initialise variables
    n = len(rmgList)
    T = marketReturnHistory.shape[1]
    # Resize the number of lags if insufficient observations
    p = int((T-1)/(n+1))
    p = min(p, maxLags)
    p = max(p, 1)
    if T < p+1:
        # If too few observations to do anything at all, abort
        logging.warning('Not enough observations (%s) to perform VAR-%s', T, p)
        return []
    np = n*p
    logging.info('Using %s lags, %s Time Periods, iDim %s for %s variables',
                    p, T, np, n)
    
    # Set up lagged return history matrix
    bigBlockReturnMatrix = Matrices.allMasked((np,T-p), float)
    for j in range(T-p):
        for iBlock in range(p):
            iLoc = iBlock*n
            bigBlockReturnMatrix[iLoc:iLoc+n,j] = marketReturnHistory[:,j+iBlock]
    currentReturnMatrix = marketReturnHistory[:,:T-p-1]
    
    # Loop round each market and compute the relevant row of weights M
    M = numpy.zeros((n,np), float)
    MStat = numpy.zeros((n,np+1), float)
    for (i, rmg) in enumerate(rmgList):
        # Ensure that only markets trading after particular market
        # has closed have non-zero values in M
        mktIdx = [idx for idx in range(n)
                  if (rmgList[idx].gmt_offset+2) < rmg.gmt_offset
                  and rmgList[idx] in synchMarkets]
        if len(mktIdx) > 0:
            # Pick out particular market's return
            mktRets = currentReturnMatrix[i,:][numpy.newaxis,:]
            # Pick out selection of lagged returns
            for ip in range(p-1):
                ids = [(1+ip)*m for m in mktIdx]
                mktIdx.extend(ids)
            subBlock = ma.take(bigBlockReturnMatrix, mktIdx, axis=0)
            lagRets = subBlock[:,1:]
            # Check to make sure that current day's reference market
            # returns are non-missing. Skip everything if they are
            curMktRet = subBlock[:,0]
            curMktRet = ma.masked_where(abs(curMktRet)<1.0e-12, curMktRet)
            missingCurMkt = numpy.flatnonzero(ma.sum(
                ma.getmaskarray(curMktRet), axis=0))
            if len(missingCurMkt) == len(synchMarkets):
                mktList = [r.description for r in synchMarkets]
                logging.warning('Missing current return for markets: %s', mktList)
                logging.warning('Aborting...')
                return []
            # Snip out instances where one or more markets does not trade
            missingHist1 = numpy.flatnonzero(ma.sum(
                ma.getmaskarray(lagRets), axis=0))
            missingHist2 = numpy.flatnonzero(ma.sum(
                ma.getmaskarray(mktRets), axis=0))
            missingHist = list(set(missingHist1).union(set(missingHist2)))
            okIndices = [idx for idx in range(mktRets.shape[1]) if \
                    idx not in missingHist]
            t = len(okIndices)
            if t < p+1:
                # If too few observations for particular market, skip
                logging.warning('Market: %s: insufficient observations (%s) to perform VAR-%s',
                        rmg.description, t, p)
            elif 0 in missingHist2:
                # Make sure latest market return is non-missing 
                logging.warning('Missing latest return for %s market', rmg.description)
            else:
                # Pick out non-missing sets of returns
                lagRets = ma.take(lagRets, okIndices, axis=1)
                mktRets = ma.take(mktRets, okIndices, axis=1)
             
                # Solve the matrix system via non-negative least squares
                X = numpy.transpose(numpy.array(lagRets))
                y = numpy.array(numpy.ravel(mktRets))
                m = Utilities.non_negative_least_square(X, y)
                resid = y - numpy.sum(X*m, axis=1)
             
                # Pick out non-zero coefficients
                nObs = len(y)
                maskCoeffs = ma.masked_where(m < 1.0e-12, m)
                nonZeroIdx = numpy.flatnonzero(ma.getmaskarray(maskCoeffs)==0)
                nPar = len(nonZeroIdx)
             
                # Compute t-stats
                tStats = numpy.zeros((len(m)), float)
                if nPar > 0:
                    stdErr = Utilities.computeRegressionStdError(\
                            resid, ma.take(X, nonZeroIdx, axis=1))
                    ts = ma.take(m, nonZeroIdx, axis=0) / stdErr
                    numpy.put(tStats, nonZeroIdx, ts)
                
                # Compute R-Square
                y = numpy.array(numpy.ravel(mktRets))
                sst = float(ma.inner(y, y))
                sse = float(ma.inner(resid, resid))
                adjr = 0.0
                if sst > 0.0:
                    adjr = 1.0 - sse / sst
                if nObs > nPar:
                    adjr = max(1.0 - (1.0-adjr)*(nObs-1)/(nObs-nPar), 0.0)
                
                # Store regression statistics
                for (j,idx) in enumerate(mktIdx):
                    M[i,idx] = m[j]
                    MStat[i,idx] = tStats[j] * tStats[j]
                    MStat[i,-1] = adjr 
    
    if debugReporting:
        rmgNames = [r.description.replace(',','') for r in rmgList]
        syncNames = [r.description.replace(',','') for r in synchMarkets]
        syncNames = ['%s|%s' % (date, nm) for nm in syncNames]
        mktIdx = [idx for idx in range(n) if rmgList[idx] in synchMarkets]
        j0 = 0
        j1 = np
        for lag in range(p):
            outFile = 'tmp/M-Lag%d-%s.csv' % (lag, date)
            msub = numpy.transpose(M[:,j0:j1])
            msub = numpy.take(msub, mktIdx, axis=0)
            Utilities.writeToCSV(msub, outFile, rowNames=syncNames, columnNames=rmgNames)
            outFile = 'tmp/MStats-Lag%d-%s.csv' % (lag, date)
            msub = numpy.transpose(MStat[:,j0:j1])
            msub = numpy.take(msub, mktIdx, axis=0)
            Utilities.writeToCSV(msub, outFile, rowNames=syncNames, columnNames=rmgNames)
            j0+=np
            j1+=np

    # Compute error term
    residual = currentReturnMatrix - ma.dot(M, bigBlockReturnMatrix[:,1:])
    # Compute synchronised returns
    synchedMarketReturn = ma.dot(M, bigBlockReturnMatrix[:,:-1]) + residual
    synchedMarketReturn = ma.array(synchedMarketReturn[:,0])
    return synchedMarketReturn

