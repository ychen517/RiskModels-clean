
import unittest
import os.path
import pandas
import pandas as pd
import numpy as np
import numpy.ma as ma
import time
import random
import copy
import inspect
import sys
import pickle as pickle
import numpy
from scipy.stats import norm as Gaussian
from riskmodels import Matrices
from riskmodels import Utilities
from riskmodels import Outliers

runtime_result = {}

class test_SectionCode(unittest.TestCase):

    '''
        this unit test is used for testing changes as of 2017-3-16.
        changes are for speeding up code run time.
    '''
    mad_const = Gaussian.ppf(3/4.)

    def load_test_data(self,filename):
        with open('Testing/testInput/' + filename) as f:  # Python 3: open(..., 'wb')
            self.test_data = pickle.load(f)
        return self.test_data

    def ma_equal(self,a,b):
        '''
        Compare two maskedArray and tell whether they are equal
        :return: True or False
        '''
        try:

            if (type(a) is numpy.ma.core.MaskedArray) and (type(b) is numpy.ma.core.MaskedArray):
                print('result is masked array')
                # np.testing.assert_equal(a,b)
                np.testing.assert_equal(ma.getmaskarray(a),ma.getmaskarray(b))  # mask
                a2 = ma.filled(a, fill_value=0)
                b2 = ma.filled(b, fill_value=0)
                np.testing.assert_equal(a2,b2)
            else:
                np.testing.assert_equal(a,b)

        except AssertionError:
            return False
        return True

    def getMADBounds(self, data):
        # To do - code for weighted median
        loc = ma.median(data, axis=0)
        scale = ma.median(ma.fabs(data - loc)/self.mad_const, axis=0)
        return loc, scale

    def section_computeOutlierBoundsInnerMAD(self,data,idxList,estuArray,zeroTolerance,newcode_flag=True):
        self.zeroTolerance = zeroTolerance
        if newcode_flag:
            subdata = data[:,idxList]
            tmpEstu = numpy.repeat(estuArray[:,numpy.newaxis], subdata.shape[1],1) #Repeat estuArray as columns
            subdata[tmpEstu == False] = ma.masked
            mask = ma.getmaskarray(subdata)
            missingcnt = numpy.sum(mask, axis=0)
            zeroflag = (subdata == 0.0)
            # zerocnt = ma.sum(zeroflag, axis=0)

            if self.zeroTolerance > 0.0:
                raise Exception('Unsupported zero tolernace')
            subdata[zeroflag] = ma.masked
            missingcnt = ma.sum(ma.getmaskarray(subdata), axis=0)

            subdata_idx = ma.where(missingcnt < subdata.shape[0] - 1)[0]
            subdata = subdata[:,subdata_idx]
            goodIdx = ma.array(idxList)[subdata_idx]
            dataCentre, dataScale = self.getMADBounds(subdata)
        else:
            dataCentre = []
            dataScale = []
            goodIdx = []
            for j in idxList:

                # Pick out column of data for MADing
                col = data[:,j]
                col = ma.masked_where(estuArray==False, col)
                missingIdx = list(numpy.flatnonzero(ma.getmaskarray(col)))

                # Check for overabundance of zeros
                zero_idx = ma.where(col==0.0)[0]
                zeroCol = numpy.zeros((len(col)), int)
                if len(zero_idx) > self.zeroTolerance * (len(col)-len(missingIdx)):
                    # If too many zeros, mask some
                    nZero = int(self.zeroTolerance * (len(col)-len(missingIdx)))
                    zeroMaskIdx = zero_idx[nZero:]
                    numpy.put(zeroCol, zeroMaskIdx, 1)
                    col = ma.masked_where(zeroCol, col)
                    missingIdx.extend(zeroMaskIdx)

                if len(missingIdx) < len(col)-1:
                    goodIdx.append(j)
                    okIdx = numpy.flatnonzero(ma.getmaskarray(col)==0)
                    okData = ma.filled(ma.take(col, okIdx, axis=0), 0.0)
                    # wts = numpy.take(weights, okIdx, axis=0)

                    # Compute upper and lower bounds
                    (loc, scale) = self.getMADBounds(okData)
                    dataCentre.append(loc)
                    dataScale.append(scale)
        return goodIdx,dataCentre, dataScale

    # need to revisit
    def section_computeOutlierBounds(self,badLowerIdx,badUpperIdx,lower,upper,goodIdx,data_raw,retVal_raw,newcode_flag=True):

        # badLowerIdx is bad index for lower,upper and goodIdx
        # len(goodIdx) == len(lower) == len(upper)
        # goodIdx and badLowerIdx/badUpperIdx are different measures.

        data = data_raw.copy()
        data.unshare_mask()
        retVal = Utilities.Struct(retVal_raw.to_dict())
        retVal.outlierArray.unshare_mask()
        retVal.lowerBounds.unshare_mask()
        retVal.upperBounds.unshare_mask()
        if newcode_flag:
            bad_idx = list(badLowerIdx.union(badUpperIdx))
            if len(bad_idx) < len(lower): # there are cases all idx are bad
                lower2 = numpy.delete(lower, bad_idx, axis=0)
                upper2 = numpy.delete(upper, bad_idx, axis=0)
                goodIdx2 = numpy.delete(goodIdx, bad_idx, axis=0)
                good_data = data[:,goodIdx2]
                lower2 = ma.column_stack(lower2).repeat(good_data.shape[0],axis=0)
                upper2 = ma.column_stack(upper2).repeat(good_data.shape[0],axis=0)
                isOutlier_bools = ma.logical_or(good_data < lower2, good_data > upper2)

                outlierArray_copy = retVal.outlierArray[:,goodIdx2].copy()
                outlierArray_copy[isOutlier_bools] = 1.0
                retVal.outlierArray[:,goodIdx2] = outlierArray_copy
                retVal.lowerBounds[goodIdx2] = lower2
                retVal.upperBounds[goodIdx2] = upper2
        else:
            for (i,j) in enumerate(goodIdx):
                if (i in badLowerIdx) or (i in badUpperIdx):
                    continue
                col = data[:,j]
                outlierCol = Matrices.allMasked(len(col))
                # Determine those data points outside the bounds
                lowerIdx = ma.where(col < lower[i])[0]
                upperIdx = ma.where(col > upper[i])[0]
                ma.put(outlierCol, lowerIdx, 1.0)
                ma.put(outlierCol, upperIdx, 1.0)
                retVal.outlierArray[:,j] = outlierCol
                retVal.lowerBounds[j] = lower[i]
                retVal.upperBounds[j] = upper[i]
        return retVal

    def section_twodMADInner(self,outlierAssets,retVal2,retVal1,clippedLowerValues,clippedUpperValues,colOutliers,outlierTimes,newcode_flag=True):
        clippedLowerValues = clippedLowerValues.copy()
        clippedUpperValues = clippedUpperValues.copy()

        if newcode_flag:
            # new code:
            clippedLowerValues[outlierAssets,:] = retVal2.lowerBounds[outlierAssets,numpy.newaxis] # broadcasting along colaxis
            clippedUpperValues[outlierAssets,:] = retVal2.upperBounds[outlierAssets,numpy.newaxis] # broadcasting along colaxis
            if colOutliers is not None:
                # Lower set of bound
                lowerCols = clippedLowerValues[:, outlierTimes]
                tmp_retVal1_lowerBounds = retVal1.lowerBounds[numpy.newaxis,outlierTimes].repeat(lowerCols.shape[0],axis=0)
                lowerCols = ma.where(lowerCols > tmp_retVal1_lowerBounds,tmp_retVal1_lowerBounds, lowerCols)
                clippedLowerValues[:, outlierTimes] = lowerCols
                # Upper set of bound
                upperCols = clippedUpperValues[:, outlierTimes]
                tmp_retVal1_upperBounds = retVal1.upperBounds[numpy.newaxis,outlierTimes].repeat(upperCols.shape[0],axis=0)
                upperCols = ma.where(upperCols < tmp_retVal1_upperBounds,tmp_retVal1_upperBounds, upperCols)
                clippedUpperValues[:, outlierTimes] = upperCols
        else:
            # old code - to remove:
            for idx in outlierAssets:
                clippedLowerValues[idx,:] = retVal2.lowerBounds[idx]
                clippedUpperValues[idx,:] = retVal2.upperBounds[idx]

            if colOutliers is not None:
                for jdx in outlierTimes:
                    # Lower set of bounds
                    lowerCol = clippedLowerValues[:, jdx]
                    if not retVal1.lowerBounds.mask[jdx]:
                        lowerCol = ma.where(lowerCol > retVal1.lowerBounds[jdx],retVal1.lowerBounds[jdx], lowerCol)
                        clippedLowerValues[:,jdx] = lowerCol
                    else:
                        lowerCol = ma.masked
                        clippedLowerValues[:,jdx] = lowerCol
                    # Upper set of bounds
                    if not retVal1.upperBounds.mask[jdx]:
                        upperCol = clippedUpperValues[:, jdx]
                        upperCol = ma.where(upperCol < retVal1.upperBounds[jdx],retVal1.upperBounds[jdx], upperCol)
                        clippedUpperValues[:,jdx] = upperCol
                    else:
                        upperCol = ma.masked
                        clippedUpperValues[:,jdx] = upperCol
        return  clippedLowerValues,clippedUpperValues

    def section_generate_net_equity_issuance1(self,tso,newcode_flag=True):

        if newcode_flag:
            import pandas as pd
            tso_df = pd.DataFrame(tso)
            tso_df = tso_df.fillna(method='ffill',axis=1)
            tso = ma.array(tso_df.values.copy(), mask=pd.isnull(tso_df).values)
        else:
            # old code - to remove
            oldestColumn = tso[:, 0]
            for jdx in range(tso.shape[1]):
                missingDataIdx = numpy.flatnonzero(ma.getmaskarray(tso[:, jdx]))
                for mIdx in missingDataIdx:
                    tso[mIdx, jdx] = oldestColumn[mIdx]
                oldestColumn = tso[:, jdx]
        return tso

    # def section_generate_net_equity_issuance2(self,data,adjDict,newcode_flag=True):
    #     the new code is actually slower
    #     if newcode_flag:
    #         adjs_tso = [adjDict.get(sid,None) for sid in data.universe]
    #         cumAdjList = [numpy.prod([x[1] for x in adj]) if adj else 1.0 for adj in adjs_tso]
    #     else:
    #         # old code - to remove
    #         cumAdjList = []
    #         for sid in data.universe:
    #             cumAdj = 1.0
    #             if sid in adjDict:
    #                 sidAdj = adjDict[sid]
    #                 for adj in sidAdj:
    #                     cumAdj *= adj[1]
    #             cumAdjList.append(cumAdj)
    #     return cumAdjList

    def fillin_beta_mktReturn(self,validSidList,hBetaDict,returns,tmpReturns,missingDataMask,marketReturnHistory):
        '''
            this function is only called by process_returns_history to fill in missing returns with beta*market_return
            the function is to replace a section of previous code and build for easy unit testing
        :return: filled in returns
        '''
        # round1: (I don't like the twist)
        # initialReturns = Matrices.allMasked(tmpReturns.data.shape)
        # hbeta_list = [hBetaDict.get(x, 1.0) for x in validSidList]
        # hbeta_array = numpy.array(hbeta_list)
        # hbeta_ma = hbeta_array[:,numpy.newaxis]
        # returns_idx = [returns.assetIdxMap.get(x) for x in validSidList]
        # tmpReturns_idx = [tmpReturns.assetIdxMap.get(x) for x in validSidList]
        # missingIdx_boolArray = missingDataMask[returns_idx,:]
        # missingIG_boolArray =  ma.getmaskarray(tmpReturns.data[tmpReturns_idx,:])
        # okProxyIdx_boolArray = numpy.logical_and(missingIdx_boolArray,numpy.logical_not(missingIG_boolArray))
        # noProxyIdx_boolArray = numpy.logical_and(missingIdx_boolArray,missingIG_boolArray)
        # hbeta_ma = numpy.repeat(hbeta_ma,initialReturns.shape[1],axis=1)
        # proxy_values = hbeta_ma * marketReturnHistory.data # broadcasting
        # # all variabls above are matched with validSidList
        # initialReturns_wrongOrder = Matrices.allMasked(tmpReturns.data.shape)
        # initialReturns_wrongOrder[okProxyIdx_boolArray] = tmpReturns.data[tmpReturns_idx,:][okProxyIdx_boolArray]
        # initialReturns_wrongOrder[noProxyIdx_boolArray] = proxy_values[tmpReturns_idx,:][noProxyIdx_boolArray]
        # initialReturns[tmpReturns_idx,:] = initialReturns_wrongOrder
        # print [np.sum(missingIdx_boolArray[0,:]),np.sum(missingIG_boolArray[0,:]),np.sum(okProxyIdx_boolArray[0,:]),np.sum(noProxyIdx_boolArray[0,:])]
        # print(np.sum(~initialReturns[927,:].mask)) # it should be 295
        # print [np.sum(missingIdx_bools[927,:]),np.sum(missingIG_bools[927,:]),np.sum(okProxyIdx_bools[927,:]),np.sum(noProxyIdx_bools[927,:])]
        # round2: (fix the round1)
        returns_idx = [returns.assetIdxMap.get(x) for x in tmpReturns.assets]
        missingIdx_bools = missingDataMask[returns_idx,:]
        missingIG_bools =  ma.getmaskarray(tmpReturns.data)
        okProxyIdx_bools = numpy.logical_and(missingIdx_bools,numpy.logical_not(missingIG_bools))
        noProxyIdx_bools = numpy.logical_and(missingIdx_bools,missingIG_bools)

        initialReturns = Matrices.allMasked(tmpReturns.data.shape)
        initialReturns[okProxyIdx_bools] = tmpReturns.data[okProxyIdx_bools]
        if noProxyIdx_bools.any():
            hbeta_list = [hBetaDict.get(x, 1.0) for x in tmpReturns.assets]
            hbeta_arr = numpy.array(hbeta_list)
            hbeta_arr = hbeta_arr[:,numpy.newaxis]
            proxy_values = hbeta_arr * marketReturnHistory.data # broadcasting
            initialReturns[noProxyIdx_bools] = proxy_values[noProxyIdx_bools]
        return initialReturns

    def fillin_beta_mktReturn_old(self,validSidList,hBetaDict,returns,tmpReturns,missingDataMask,marketReturnHistory):
        '''
            this function is build for easy unit testing.
            it has old code to remove as of 2017-3-15.
        :return: filled in returns
        '''
        initialReturns = Matrices.allMasked(tmpReturns.data.shape)
        for sid in validSidList:
            # Find returns still missing data
            idx = returns.assetIdxMap[sid]
            missingIdx = set(numpy.flatnonzero(missingDataMask[idx,:]))
            idx = tmpReturns.assetIdxMap[sid]
            missingIG = set(numpy.flatnonzero(ma.getmaskarray(tmpReturns.data[idx,:])))
            okProxyIdx = list(missingIdx.difference(missingIG))
            noProxyIdx = list(missingIdx.intersection(missingIG))
            # Fill such with beta*mkt return
            hbeta = 1.0
            if sid in hBetaDict:
                hbeta = ma.filled(hBetaDict[sid], 1.0)
            for jdx in okProxyIdx:
                initialReturns[idx,jdx] = tmpReturns.data[idx, jdx]
            for jdx in noProxyIdx:
                initialReturns[idx,jdx] = hbeta * marketReturnHistory.data[0, jdx]
        return initialReturns

    def section_fillin_beta_mktReturn(self,validSidList,hBetaDict,returns,tmpReturns,missingDataMask,marketReturnHistory,newcode_flag=True):
        # import ipdb;ipdb.set_trace()
        if newcode_flag:
            initialReturns = self.fillin_beta_mktReturn(validSidList,hBetaDict,returns,tmpReturns,missingDataMask,marketReturnHistory)
        else:
            initialReturns = self.fillin_beta_mktReturn_old(validSidList,hBetaDict,returns,tmpReturns,missingDataMask,marketReturnHistory)
        return initialReturns

    def section_adjustReturnsForTiming1(self,subIssues,tradingCountryMap,rmgIdxMap,data_raw,adjustments,dateLen,newcode_flag=True):
        data = data_raw.copy()
        data.unshare_mask()
        tradingCountryMap = tradingCountryMap.copy()
        rmgIdxMap = rmgIdxMap.copy()
        if newcode_flag:
            tradingIdx_array = numpy.array([tradingCountryMap.get(x, numpy.nan) for x in subIssues])
            rmgIdx_array     = numpy.array([rmgIdxMap.get(x, numpy.nan) for x in tradingIdx_array])
            not_nan_idx = numpy.where(~numpy.isnan(rmgIdx_array))
            data[not_nan_idx,:] += adjustments.data[rmgIdx_array[not_nan_idx].astype(int),:dateLen]
        else:
            # old code - to remove
            for (idx, sid) in enumerate(subIssues):
                tradingIdx = tradingCountryMap.get(sid, None)
                if tradingIdx is not None and tradingIdx in rmgIdxMap:
                    rmgIdx = rmgIdxMap[tradingIdx]
                    data[idx,:] += adjustments.data[rmgIdx,:dateLen]
        return data

    def section_adjustReturnsForTiming2(self,subIssues,homeCountryMap,rmgIdxMap,data_raw,adjustments,dateLen,newcode_flag=True):
        data = data_raw.copy()
        data.unshare_mask()
        homeCountryMap = homeCountryMap.copy()
        rmgIdxMap = rmgIdxMap.copy()
        if newcode_flag:
            homeIdx_array = numpy.array([homeCountryMap.get(x, numpy.nan) for x in subIssues])
            rmgIdx_array = numpy.array([rmgIdxMap.get(x, numpy.nan) for x in homeIdx_array])
            not_nan_idx = numpy.where(~numpy.isnan(rmgIdx_array))
            data[not_nan_idx,:] -= adjustments.data[rmgIdx_array[not_nan_idx].astype(int),:dateLen]
        else:
            # old code - to remove
            for (idx, sid) in enumerate(subIssues):
                homeIdx = homeCountryMap.get(sid, None)
                if homeIdx is not None and homeIdx in rmgIdxMap:
                    rmgIdx = rmgIdxMap[homeIdx]
                    data[idx,:] -= adjustments.data[rmgIdx,:dateLen]
        return data

# 8 unit tests for 8 section code above
    def test_computeOutlierBoundsInnerMAD(self):
        '''
        test computeOutlierBoundsInnerMAD
        '''
        current_fun_name = sys._getframe().f_code.co_name
        print('Test Begin for: ' + current_fun_name)

        data_in1 = self.load_test_data('input_computeOutlierBoundsInnerMAD.pickle')
        data1,idxList1,estuArray1,zeroTolerance1 = data_in1
        data_in2 = self.load_test_data('input_computeOutlierBoundsInnerMAD.pickle')
        data2,idxList2,estuArray2,zeroTolerance2 = data_in2

        # from IPython import embed; embed(header='Check');import ipdb;ipdb.set_trace()
        # test case:
        N1 = data1.shape[1]
        K1 = 300
        idxList1 = random.sample(list(range(1, N1)), K1)
        N2 = data1.shape[0]
        K2 = 5
        tmp_list = random.sample(list(range(1, N2)), K2)
        estuArray1[tmp_list] = False
        ##########################################
        idxList2 = copy.deepcopy(idxList1)
        estuArray2 = estuArray1.copy()
        # import ipdb;ipdb.set_trace()
        # run section of code
        timeResult = {}
        t0 = time.time()
        result_new = self.section_computeOutlierBoundsInnerMAD(data1,idxList1,estuArray1,zeroTolerance1,newcode_flag=True)
        t1 = time.time()
        timeResult['runTime of result_new'] = t1 - t0
        t0 = time.time()
        result_old = self.section_computeOutlierBoundsInnerMAD(data2,idxList2,estuArray2,zeroTolerance2,newcode_flag=False)
        t1 = time.time()
        timeResult['runTime of result_old'] = t1 - t0
        timeResult2 = pd.DataFrame(list(timeResult.items()),columns=['Index',current_fun_name]).set_index('Index')
        print(timeResult2)
        global runtime_result
        runtime_result[current_fun_name] = timeResult2
        try:
            self.assertTrue(self.ma_equal(result_old[0],result_new[0]))
            self.assertTrue(self.ma_equal(result_old[1],result_new[1]))
            self.assertTrue(self.ma_equal(result_old[2],result_new[2]))
        except:
            from IPython import embed; embed(header='Unit Test Failed: ' + current_fun_name);import ipdb;ipdb.set_trace()
        print('Test Finished for: ' + current_fun_name)
        # from IPython import embed; embed(header='End');

    def test_computeOutlierBounds(self):
        '''
        test section_computeOutlierBounds
        '''
        current_fun_name = sys._getframe().f_code.co_name
        print('Test Begin for: ' + current_fun_name)

        data_in1 = self.load_test_data('input_computeOutlierBounds.pickle')
        badLowerIdx1,badUpperIdx1,lower1,upper1,goodIdx1,data1,retVal1 = data_in1
        data_in2 = self.load_test_data('input_computeOutlierBounds.pickle')
        badLowerIdx2,badUpperIdx2,lower2,upper2,goodIdx2,data2,retVal2 = data_in2

        # test case:
        # case (1)
        # print 'case 1: '
        # N = len(lower1)
        # badLowerIdx1 = set(range(N))
        # badUpperIdx1 = set(range(N))
        # case (2)
        print('case 2: ')
        N = len(lower1)
        K = 50
        N1 = random.sample(list(range(1, N)), K)
        N2 = random.sample(list(range(1, N)), K)
        badLowerIdx1 = set(N1)
        badUpperIdx1 = set(N2)
        # case (3)
        # print 'case 3: '
        # N = len(lower1)
        # K = 50
        # N1 = random.sample(range(N), K)
        # N2 = random.sample(range(N), K)
        # badLowerIdx1 = set(N1)
        # badUpperIdx1 = set(range(N))
        ######################################

        badLowerIdx2 = badLowerIdx1.copy()
        badUpperIdx2 = badUpperIdx1.copy()
        # run section of code
        timeResult = {}
        t0 = time.time()
        result_new = self.section_computeOutlierBounds(badLowerIdx1,badUpperIdx1,lower1,upper1,goodIdx1,data1,retVal1,newcode_flag=True)
        t1 = time.time()
        timeResult['runTime of result_new'] = t1 - t0
        t0 = time.time()

        result_old = self.section_computeOutlierBounds(badLowerIdx2,badUpperIdx2,lower2,upper2,goodIdx2,data2,retVal2,newcode_flag=False)
        t1 = time.time()
        timeResult['runTime of result_old'] = t1 - t0
        timeResult2 = pd.DataFrame(list(timeResult.items()),columns=['Index',current_fun_name]).set_index('Index')
        print(timeResult2)
        global runtime_result
        runtime_result[current_fun_name] = timeResult2
        try:
            self.assertTrue(self.ma_equal(result_old.outlierArray,result_new.outlierArray))
            self.assertTrue(self.ma_equal(result_old.lowerBounds,result_new.lowerBounds))
            self.assertTrue(self.ma_equal(result_old.upperBounds,result_new.upperBounds))
            pd.DataFrame(result_old.outlierArray).dropna(axis=1,how='all').dropna(axis=0,how='all')
            pd.DataFrame(result_new.outlierArray).dropna(axis=1,how='all').dropna(axis=0,how='all')
            [pd.DataFrame(ma.getmask(result_old.outlierArray)).sum() - pd.DataFrame(ma.getmask(result_new.outlierArray)).sum()]
        except:
            from IPython import embed; embed(header='Unit Test Failed: ');import ipdb;ipdb.set_trace()
            # from IPython import embed; embed(header=['Unit Test Failed: ' + current_fun_name]);import ipdb;ipdb.set_trace()
        print('Test Finished for: ' + current_fun_name)
        # from IPython import embed; embed(header='End');

    def test_twodMADInner(self):
        '''
        test computeOutlierBoundsInnerMAD
        '''
        current_fun_name = sys._getframe().f_code.co_name
        print('Test Begin for: ' + current_fun_name)

        data_in1 = self.load_test_data('input_twodMADInner.pickle')
        outlierAssets_1,retVal2_1,retVal1_1,clippedLowerValues_1,clippedUpperValues_1,colOutliers_1,outlierTimes_1 = data_in1
        data_in2 = self.load_test_data('input_twodMADInner.pickle')
        outlierAssets_2,retVal2_2,retVal1_2,clippedLowerValues_2,clippedUpperValues_2,colOutliers_2,outlierTimes_2 = data_in2

        # from IPython import embed; embed(header='Check');import ipdb;ipdb.set_trace()
        # test case:
        N_Assets = clippedLowerValues_1.shape[0]
        N_Times = clippedLowerValues_1.shape[1]
        K1 = 100
        K2 = 200
        outlierAssets_1 = np.array(random.sample(list(range(N_Assets)), K1))
        outlierTimes_1  = np.array(random.sample(list(range(N_Times)), K2))
        ##########################################
        outlierAssets_2 = outlierAssets_1.copy()
        outlierTimes_2 = outlierTimes_1.copy()

        # run section of code
        timeResult = {}
        t0 = time.time()
        result_new = self.section_twodMADInner(outlierAssets_1,retVal2_1,retVal1_1,clippedLowerValues_1,clippedUpperValues_1,colOutliers_1,outlierTimes_1,newcode_flag=True)
        t1 = time.time()
        timeResult['runTime of result_new'] = t1 - t0
        t0 = time.time()
        result_old = self.section_twodMADInner(outlierAssets_2,retVal2_2,retVal1_2,clippedLowerValues_2,clippedUpperValues_2,colOutliers_2,outlierTimes_2,newcode_flag=False)
        t1 = time.time()
        timeResult['runTime of result_old'] = t1 - t0
        timeResult2 = pd.DataFrame(list(timeResult.items()),columns=['Index',current_fun_name]).set_index('Index')
        print(timeResult2)
        global runtime_result
        runtime_result[current_fun_name] = timeResult2
        try:
            self.assertTrue(self.ma_equal(result_old[0],result_new[0]))
            self.assertTrue(self.ma_equal(result_old[1],result_new[1]))
        except:
            from IPython import embed; embed(header='Unit Test Failed: ' + current_fun_name);import ipdb;ipdb.set_trace()
        print('Test Finished for: ' + current_fun_name)
        # from IPython import embed; embed(header='End');

    def test_generate_net_equity_issuance1(self):
        '''
        test computeOutlierBoundsInnerMAD
        '''
        current_fun_name = sys._getframe().f_code.co_name
        print('Test Begin for: ' + current_fun_name)

        data_in1 = self.load_test_data('input_generate_net_equity_issuance1.pickle')
        tso1 = data_in1[0]

        data_in2 = self.load_test_data('input_generate_net_equity_issuance1.pickle')
        tso2 = data_in2[0]

        # from IPython import embed; embed(header='Check');import ipdb;ipdb.set_trace()
        n1 = tso1.shape[0]
        n2 = tso1.shape[0]
        mask_idx = ma.where(np.random.choice([0, 1], size=(n1,n2)))
        tso1[mask_idx] = ma.masked
        tso2[mask_idx] = ma.masked

        # run section of code
        timeResult = {}
        t0 = time.time()
        result_new = self.section_generate_net_equity_issuance1(tso1,newcode_flag=True)
        t1 = time.time()
        timeResult['runTime of result_new'] = t1 - t0
        t0 = time.time()
        result_old = self.section_generate_net_equity_issuance1(tso2,newcode_flag=False)
        t1 = time.time()
        timeResult['runTime of result_old'] = t1 - t0
        timeResult2 = pd.DataFrame(list(timeResult.items()),columns=['Index',current_fun_name]).set_index('Index')
        print(timeResult2)
        global runtime_result
        runtime_result[current_fun_name] = timeResult2
        try:
            self.assertTrue(self.ma_equal(result_old,result_new))
        except:
            from IPython import embed; embed(header='Unit Test Failed: ' + current_fun_name);import ipdb;ipdb.set_trace()
        print('Test Finished for: ' + current_fun_name)
        # from IPython import embed; embed(header='End');

    # def test_generate_net_equity_issuance2(self):
    #     '''
    #     test computeOutlierBoundsInnerMAD
    #     '''
    #     current_fun_name = sys._getframe().f_code.co_name
    #     print 'Test Begin for: ' + current_fun_name
    #
    #     data = self.load_test_data('input_generate_net_equity_issuance2.pickle')
    #     data,adjDict = data
    #
    #     # run section of code
    #     timeResult = {}
    #     t0 = time.time()
    #     result_new = self.section_generate_net_equity_issuance2(data,adjDict,newcode_flag=True)
    #     t1 = time.time()
    #     timeResult['runTime of result_new'] = t1 - t0
    #     t0 = time.time()
    #     result_old = self.section_generate_net_equity_issuance2(data,adjDict,newcode_flag=False)
    #     t1 = time.time()
    #     timeResult['runTime of result_old'] = t1 - t0
    #     timeResult2 = pd.DataFrame(timeResult.items(),columns=['Index',current_fun_name]).set_index('Index')
    #     print(timeResult2)
    #     global runtime_result
    #     runtime_result[current_fun_name] = timeResult2
    #     try:
    #         self.assertTrue(self.ma_equal(result_old,result_new))
    #     except:
    #         from IPython import embed; embed(header='Unit Test Failed: ' + current_fun_name);import ipdb;ipdb.set_trace()
    #     print 'Test Finished for: ' + current_fun_name
    #     # from IPython import embed; embed(header='End');

    def test_fillin_beta_mktReturn(self):
        '''
        test computeOutlierBoundsInnerMAD
        '''
        current_fun_name = sys._getframe().f_code.co_name
        print('Test Begin for: ' + current_fun_name)
        data_in1 = self.load_test_data('input_fillin_beta_mktReturn.pickle')
        validSidList1,hBetaDict1,returns1,tmpReturns1,missingDataMask1,marketReturnHistory1 = data_in1
        data_in2 = self.load_test_data('input_fillin_beta_mktReturn.pickle')
        validSidList2,hBetaDict2,returns2,tmpReturns2,missingDataMask2,marketReturnHistory2 = data_in2

        # from IPython import embed; embed(header='Check');import ipdb;ipdb.set_trace()
        N_Assets = len(validSidList1)
        K1 = 50
        sid_idx = np.array(random.sample(list(range(N_Assets)), K1))
        tmp_sids = list(validSidList1)
        hBetaDict1 = dict([(tmp_sids[x],np.random.normal(loc=1.0, scale=1.0, size=1)[0]) for x in sid_idx])
        mask_idx1 = ma.where(np.random.choice([0, 1], size=missingDataMask1.shape))
        missingDataMask1[mask_idx1] = True
        mask_idx2 = ma.where(np.random.choice([0, 1], size=tmpReturns1.data.shape))
        tmpReturns1.data[mask_idx2] = ma.masked
        ##########################################
        hBetaDict2 = hBetaDict1.copy()
        missingDataMask2 = missingDataMask1.copy()
        tmpReturns2.data = tmpReturns1.data.copy()
        # run section of code
        timeResult = {}
        t0 = time.time()
        result_new = self.section_fillin_beta_mktReturn(validSidList1,hBetaDict1,returns1,tmpReturns1,missingDataMask1,marketReturnHistory1,newcode_flag=True)
        t1 = time.time()
        timeResult['runTime of result_new'] = t1 - t0
        t0 = time.time()
        result_old = self.section_fillin_beta_mktReturn(validSidList2,hBetaDict2,returns2,tmpReturns2,missingDataMask2,marketReturnHistory2,newcode_flag=False)
        t1 = time.time()
        timeResult['runTime of result_old'] = t1 - t0

        timeResult2 = pd.DataFrame(list(timeResult.items()),columns=['Index',current_fun_name]).set_index('Index')
        print(timeResult2)
        global runtime_result
        runtime_result[current_fun_name] = timeResult2
        try:
            self.assertTrue(self.ma_equal(result_old,result_new))
        except:
            from IPython import embed; embed(header='Unit Test Failed: ' + current_fun_name);import ipdb;ipdb.set_trace()
        print('Test Finished for: ' + current_fun_name)
        # from IPython import embed; embed(header='End');

    def test_adjustReturnsForTiming1(self):
        '''
        test computeOutlierBoundsInnerMAD
        '''
        current_fun_name = sys._getframe().f_code.co_name
        print('Test Begin for: ' + current_fun_name)

        data_in1 = self.load_test_data('input_adjustReturnsForTiming1.pickle')
        subIssues1,tradingCountryMap1,rmgIdxMap1,data1,adjustments1,dateLen1 = data_in1
        data_in2 = self.load_test_data('input_adjustReturnsForTiming1.pickle')
        subIssues2,tradingCountryMap2,rmgIdxMap2,data2,adjustments2,dateLen2 = data_in2
        # from IPython import embed; embed(header='Check');import ipdb;ipdb.set_trace()
        data1 = data1 + np.random.normal(loc=0.0, scale=1.0, size=data1.shape)
        mask_idx1 = ma.where(np.random.choice([0, 1], size=data1.shape))
        data1[mask_idx1] = ma.masked
        adjustments1.data = adjustments1.data + np.random.normal(loc=0.0, scale=1.0, size=adjustments1.data.shape)
        ##########################################
        import copy
        data2 = copy.copy(data1)
        adjustments2.data = copy.copy(adjustments1.data)
        # run section of code
        timeResult = {}
        t0 = time.time()
        result_new = self.section_adjustReturnsForTiming1(subIssues1,tradingCountryMap1,rmgIdxMap1,data1,adjustments1,dateLen1,newcode_flag=True)
        t1 = time.time()
        timeResult['runTime of result_new'] = t1 - t0
        t0 = time.time()
        result_old = self.section_adjustReturnsForTiming1(subIssues2,tradingCountryMap2,rmgIdxMap2,data2,adjustments2,dateLen2,newcode_flag=False)
        t1 = time.time()
        timeResult['runTime of result_old'] = t1 - t0

        timeResult2 = pd.DataFrame(list(timeResult.items()),columns=['Index',current_fun_name]).set_index('Index')
        print(timeResult2)
        global runtime_result
        runtime_result[current_fun_name] = timeResult2
        try:
            self.assertTrue(self.ma_equal(result_old,result_new))
        except:
            from IPython import embed; embed(header='Unit Test Failed: ' + current_fun_name);import ipdb;ipdb.set_trace()
        print('Test Finished for: ' + current_fun_name)
        # from IPython import embed; embed(header='End');

    def test_adjustReturnsForTiming2(self):
        '''
        test computeOutlierBoundsInnerMAD
        '''
        current_fun_name = sys._getframe().f_code.co_name
        print('Test Begin for: ' + current_fun_name)
        data_in1 = self.load_test_data('input_adjustReturnsForTiming1.pickle')
        subIssues1,homeCountryMap1,rmgIdxMap1,data1,adjustments1,dateLen1 = data_in1
        data_in2 = self.load_test_data('input_adjustReturnsForTiming1.pickle')
        subIssues2,homeCountryMap2,rmgIdxMap2,data2,adjustments2,dateLen2 = data_in2
        # from IPython import embed; embed(header='Check');import ipdb;ipdb.set_trace()
        data1 = data1 + np.random.normal(loc=0.0, scale=1.0, size=data1.shape)
        mask_idx1 = ma.where(np.random.choice([0, 1], size=data1.shape))
        data1[mask_idx1] = ma.masked
        adjustments1.data = adjustments1.data + np.random.normal(loc=0.0, scale=1.0, size=adjustments1.data.shape)
        ##########################################
        import copy
        data2 = copy.copy(data1)
        adjustments2.data = copy.copy(adjustments1.data)
        # run section of code
        timeResult = {}
        t0 = time.time()
        result_new = self.section_adjustReturnsForTiming2(subIssues1,homeCountryMap1,rmgIdxMap1,data1,adjustments1,dateLen1,newcode_flag=True)
        t1 = time.time()
        timeResult['runTime of result_new'] = t1 - t0
        t0 = time.time()
        result_old = self.section_adjustReturnsForTiming2(subIssues2,homeCountryMap2,rmgIdxMap2,data2,adjustments2,dateLen2,newcode_flag=False)
        t1 = time.time()
        timeResult['runTime of result_old'] = t1 - t0

        timeResult2 = pd.DataFrame(list(timeResult.items()),columns=['Index',current_fun_name]).set_index('Index')
        print(timeResult2)
        global runtime_result
        runtime_result[current_fun_name] = timeResult2
        try:
            self.assertTrue(self.ma_equal(result_old,result_new))
        except:
            from IPython import embed; embed(header='Unit Test Failed: ' + current_fun_name);import ipdb;ipdb.set_trace()
        print('Test Finished for: ' + current_fun_name)
        # from IPython import embed; embed(header='End');

def run_suite():
    suite = unittest.TestSuite()
    suite.addTest(test_SectionCode('test_computeOutlierBoundsInnerMAD'))
    suite.addTest(test_SectionCode('test_computeOutlierBounds'))
    suite.addTest(test_SectionCode('test_twodMADInner'))
    suite.addTest(test_SectionCode('test_generate_net_equity_issuance1'))
    suite.addTest(test_SectionCode('test_fillin_beta_mktReturn'))
    suite.addTest(test_SectionCode('test_adjustReturnsForTiming1'))
    suite.addTest(test_SectionCode('test_adjustReturnsForTiming2'))
    runner = unittest.TextTestRunner()
    runner.run(suite)
    global runtime_result
    runtime_result = pd.concat(runtime_result,axis=1).T
    runtime_result.index = runtime_result.index.droplevel()
    runtime_result.to_excel('runtime_result.xlsx')
    from IPython import embed; embed(header='run_suite End')
    return True

if __name__ == '__main__':
    # unittest.main()
    run_suite()

# python Testing/test_SectionCode.py
