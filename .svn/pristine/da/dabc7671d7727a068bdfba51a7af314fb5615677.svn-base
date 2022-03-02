
import configparser
import datetime
import numpy.ma as MA
import numpy
import logging
import unittest
from marketdb import MarketDB
import riskmodels
from riskmodels import ModelDB
from riskmodels import ModelID
from riskmodels import Utilities

class TimeSeriesTest(unittest.TestCase):
    def setUp(self):
        if config_.has_section('ModelDB'):
            self.modelDB = ModelDB.ModelDB(
                user=config_.get('ModelDB', 'user'),
                passwd=config_.get('ModelDB', 'password'),
                sid=config_.get('ModelDB', 'sid'))
        else:
            fail('Missing ModelDB section in configuration file')
        if config_.has_section('MarketDB'):
            self.marketDB = MarketDB.MarketDB(
                user=config_.get('MarketDB', 'user'),
                passwd=config_.get('MarketDB', 'password'),
                sid=config_.get('MarketDB', 'sid'))
        else:
            fail('Missing MarketDB section in configuration file')
        self.sid1 = ModelDB.SubIssue(string='DSVJCWHHN611')
        self.sid2 = ModelDB.SubIssue(string='D11D9499T911')
        self.sid3 = ModelDB.SubIssue(string='D11D4UU9B311')
        self.axid2 = ModelID.ModelID(string='GSFM9Z1BK8')
        self.axid3 = ModelID.ModelID(string='GY6W2DWMW3')
        
    def tearDown(self):
        self.modelDB.revertChanges()
        self.modelDB.finalize()
        
    def testGetAnnualTotalDebt(self):
        self.modelDB.createCurrencyCache(self.marketDB)
        convertTo = { self.sid1: 1}
        val = self.modelDB.getAnnualTotalDebt(
            datetime.date(2000,1,1), datetime.date(2001,12,31),
            [self.sid1], datetime.date(2009,1,20), self.marketDB, convertTo)
        self.assertEquals(1, len(val))
        sid1Vals = val[0]
        self.assertEquals(2, len(sid1Vals))
        self.assertEquals(datetime.date(2000,10,31), sid1Vals[0][0])
        self.assertAlmostEquals(16.5 + 28.5, sid1Vals[0][1])
        self.assertEquals(1, sid1Vals[0][2])
        self.assertEquals(datetime.date(2001,10,31), sid1Vals[1][0])
        self.assertAlmostEquals(3 + 1.7, sid1Vals[1][1])
        self.assertEquals(1, sid1Vals[1][2])
         
        convertTo = { self.sid1: 0}
        val = self.modelDB.getAnnualTotalDebt(
            datetime.date(2000,1,1), datetime.date(2001,12,31),
            [self.sid1], datetime.date(2009,1,20), self.marketDB, convertTo)
        self.assertEquals(1, len(val))
        sid1Vals = val[0]
        self.assertEquals(2, len(sid1Vals))
        self.assertEquals(datetime.date(2000,10,31), sid1Vals[0][0])
        self.assertAlmostEquals((16.5 + 28.5) * 1.5272364, sid1Vals[0][1])
        self.assertEquals(0, sid1Vals[0][2])
        self.assertEquals(datetime.date(2001,10,31), sid1Vals[1][0])
        self.assertAlmostEquals((3 + 1.7) * 1.5866052, sid1Vals[1][1])
        self.assertEquals(0, sid1Vals[1][2])
    
    def testGetIssueCUSIPs(self):
        mymid1 = ModelID.ModelID(string='DMZA79N6C2')
        mymid2 = ModelID.ModelID(string='D64KQG59D2')
        cusips = self.modelDB.getIssueCUSIPs(datetime.date(2000, 5, 1),
                                             [mymid1, mymid2], self.marketDB)
        self.assertEquals(2, len(cusips))
        self.assertEquals('87308K101', cusips[mymid1])
        self.assertEquals('881448104', cusips[mymid2])
        cusips = self.modelDB.getIssueCUSIPs(datetime.date(2000, 4, 30),
                                             [mymid1, mymid2], self.marketDB)
        self.assertEquals(2, len(cusips))
        self.assertEquals('87308K101', cusips[mymid1])
        self.assertEquals('03232Q106', cusips[mymid2])
        cusips = self.modelDB.getIssueCUSIPs(datetime.date(2003, 6, 2),
                                             [mymid1, mymid2], self.marketDB)
        self.assertEquals(1, len(cusips))
        self.assertEquals('881448104', cusips[mymid2])
        
    def testGetIssueSEDOLs(self):
        mymid1 = ModelID.ModelID(string='DWSSF9X7Z6')
        mymid2 = ModelID.ModelID(string='DN7SHSDNQ3')
        sedols = self.modelDB.getIssueSEDOLs(datetime.date(1999, 9, 21),
                                             [mymid1, mymid2], self.marketDB)
        self.assertEquals(2, len(sedols))
        self.assertEquals('6173597', sedols[mymid1])
        self.assertEquals('6108588', sedols[mymid2])
        sedols = self.modelDB.getIssueSEDOLs(datetime.date(1999, 9, 20),
                                             [mymid1, mymid2], self.marketDB)
        self.assertEquals(2, len(sedols))
        self.assertEquals('6173360', sedols[mymid1])
        self.assertEquals('6108588', sedols[mymid2])
        sedols = self.modelDB.getIssueSEDOLs(datetime.date(2003, 6, 2),
                                             [mymid1, mymid2], self.marketDB)
        self.assertEquals(1, len(sedols))
        self.assertEquals('6173597', sedols[mymid1])
        
    def testGetIssueISINs(self):
        mymid1 = ModelID.ModelID(string='D6BSA5VU10')
        mymid2 = ModelID.ModelID(string='DLVV714FG0')
        isins = self.modelDB.getIssueISINs(datetime.date(2005, 2, 7),
                                             [mymid1, mymid2], self.marketDB)
        self.assertEquals(2, len(isins))
        self.assertEquals('US86769A2069', isins[mymid1])
        self.assertEquals('CN0006030653', isins[mymid2])
        isins = self.modelDB.getIssueISINs(datetime.date(2005, 2, 6),
                                             [mymid1, mymid2], self.marketDB)
        self.assertEquals(2, len(isins))
        self.assertEquals('US86769A1079', isins[mymid1])
        self.assertEquals('CN0006030653', isins[mymid2])
        isins = self.modelDB.getIssueISINs(datetime.date(2006, 11, 30),
                                             [mymid1, mymid2], self.marketDB)
        self.assertEquals(1, len(isins))
        self.assertEquals('CN0006030653', isins[mymid2])
        
    def testGetIssueNames(self):
        mymid1 = ModelID.ModelID(string='DCK3VRHKC8')
        mymid2 = ModelID.ModelID(string='DMLFWTGCZ8')
        names = self.modelDB.getIssueNames(datetime.date(2002, 12, 19),
                                             [mymid1, mymid2], self.marketDB)
        self.assertEquals(2, len(names))
        self.assertEquals('PINETREE CAPITAL CORP', names[mymid1])
        self.assertEquals('NAL OIL & GAS TR', names[mymid2])
        names = self.modelDB.getIssueNames(datetime.date(2002, 12, 18),
                                             [mymid1, mymid2], self.marketDB)
        self.assertEquals(2, len(names))
        self.assertEquals('PINETREE CAPITAL CORP', names[mymid1])
        self.assertEquals('NAL OIL & GAS TRUST', names[mymid2])
        names = self.modelDB.getIssueNames(datetime.date(1999, 12, 1),
                                             [mymid1, mymid2], self.marketDB)
        self.assertEquals(1, len(names))
        self.assertEquals('NAL OIL & GAS TRUST', names[mymid2])
        
    def testGetIssueTickers(self):
        mymid1 = ModelID.ModelID(string='DRRKXKJYU2')
        mymid2 = ModelID.ModelID(string='DQ6UJ33JF3')
        tickers = self.modelDB.getIssueTickers(datetime.date(2008, 6, 23),
                                             [mymid1, mymid2], self.marketDB)
        self.assertEquals(2, len(tickers))
        self.assertEquals('1022466', tickers[mymid1])
        self.assertEquals('REY', tickers[mymid2])
        tickers = self.modelDB.getIssueTickers(datetime.date(2008, 6, 22),
                                             [mymid1, mymid2], self.marketDB)
        self.assertEquals(1, len(tickers))
        self.assertEquals('REU', tickers[mymid2])
        tickers = self.modelDB.getIssueTickers(datetime.date(2008, 10, 14),
                                             [mymid1, mymid2], self.marketDB)
        self.assertEquals(2, len(tickers))
        self.assertEquals('NEUR', tickers[mymid1])
        self.assertEquals('REY', tickers[mymid2])
        
    def testGetFactorExposureNew(self):
        rmi = self.modelDB.getRiskModelInstance(-11, datetime.date(2005, 5, 5))
        mdl = riskmodels.getModelByName('WWResearchModel1')(
            self.modelDB, self.marketDB)
        mdl.setFactorsForDate(rmi.date, self.modelDB)
        factors = mdl.factors
        subFactors = self.modelDB.getRiskModelInstanceSubFactors(rmi, factors)
        assert(len(subFactors) > 0)
        univ = self.modelDB.getRiskModelInstanceUniverse(rmi)
        assert(len(univ) > 0)
        mdl.newExposureFormat = False
        expMatOld = mdl.loadExposureMatrix(rmi, self.modelDB)
        data = Utilities.Struct()
        data.exposureMatrix = expMatOld
        mdl.newExposureFormat = True
        mdl.insertExposures(rmi, data, self.modelDB, univ=False)
        self.modelDB.deleteRMIExposureMatrixNew(rmi, subFactors=subFactors[:5])
        mdl.insertExposures(rmi, data, self.modelDB, univ=False, update=True)
        expMatNew = mdl.loadExposureMatrix(rmi, self.modelDB)
        self.assertTrue(numpy.all((numpy.abs(
                        expMatOld.getMatrix().filled(0.0)
                        - expMatNew.getMatrix().filled(0.0)) < 1e-6)))
        
    def testGetFundamentalCurrencyItem(self):
        convertTo = None
        sid1 = ModelDB.SubIssue(string='D15YVPNX8311')
        val = self.modelDB.getFundamentalCurrencyItem(
            'ce_qtr', datetime.date(2007,2,1), datetime.date(2007,12,31),
            [sid1], datetime.date(2009,1,20), self.marketDB, convertTo)
        self.assertEquals(1, len(val))
        sid1Vals = val[0]
        self.assertEquals(3, len(sid1Vals))
        self.assertEquals(datetime.date(2007,7,31), sid1Vals[1][0])
        self.assertEquals(530.594, sid1Vals[1][1])
        self.assertEquals(1, sid1Vals[1][2])
        
        val = self.modelDB.getFundamentalCurrencyItem(
            'ce_qtr', datetime.date(2007,2,1), datetime.date(2007,12,31),
            [sid1], datetime.date(2008,1,10), self.marketDB, convertTo)
        self.assertEquals(1, len(val))
        sid1Vals = val[0]
        self.assertEquals(2, len(sid1Vals))
        self.assertEquals(datetime.date(2007,7,31), sid1Vals[1][0])
        self.assertAlmostEquals(580.56, sid1Vals[1][1], 3)
        self.assertEquals(1, sid1Vals[1][2])
        
    def testGetFundamentalItemCodes(self):
        codes = self.modelDB.getFundamentalItemCodes(
            'sub_issue_fund_currency', self.marketDB)
        self.assertEqual(2, codes['ce_qtr'])
        self.assertEqual(19, codes['ni_ann'])
        
        codes = self.modelDB.getFundamentalItemCodes(
            'sub_issue_fund_number', self.marketDB)
        self.assertEqual(2, codes['csho_qtr'])
        
        self.assertRaises(KeyError, self.modelDB.getFundamentalItemCodes,
                'no match', self.marketDB)
        
    def testGetPaidDividends(self):
        convertTo = None
        sidIBM = ModelDB.SubIssue(string='DRM8T7MA5511')
        val = self.modelDB.getPaidDividends(
            datetime.date(2007,2,1), datetime.date(2007,12,31),
            [self.sid1, sidIBM], self.marketDB, convertTo)
        self.assertEquals(2, len(val))
        self.assertEquals(0, len(val[self.sid1]))
        self.assertEquals(4, len(val[sidIBM]))
        ibmValues = val[sidIBM]
        self.assertEquals(datetime.date(2007,2,7), ibmValues[0][0])
        self.assertAlmostEquals(0.3 * 1506351844, ibmValues[0][1])
        self.assertEquals(datetime.date(2007,5,8), ibmValues[1][0])
        self.assertAlmostEquals(0.4 * 1484827275, ibmValues[1][1])
        self.assertEquals(datetime.date(2007,8,8), ibmValues[2][0])
        self.assertAlmostEquals(0.4 * 1360406581, ibmValues[2][1])
        self.assertEquals(datetime.date(2007,11,7), ibmValues[3][0])
        self.assertAlmostEquals(0.4 * 1377955258, ibmValues[3][1])
        self.modelDB.createCurrencyCache(self.marketDB)
        val = self.modelDB.getPaidDividends(
            datetime.date(2007,2,8), datetime.date(2007,12,31),
            [self.sid1, sidIBM], self.marketDB, convertTo=191)
        self.assertEquals(2, len(val))
        self.assertEquals(0, len(val[self.sid1]))
        self.assertEquals(3, len(val[sidIBM]))
        ibmValues = val[sidIBM]
        self.assertEquals(datetime.date(2007,5,8), ibmValues[0][0])
        self.assertAlmostEquals(0.4 * 1484827275 * 0.73929, ibmValues[0][1], 2)
        self.assertEquals(datetime.date(2007,8,8), ibmValues[1][0])
        self.assertAlmostEquals(0.4 * 1360406581 * 0.72372, ibmValues[1][1], 2)
        self.assertEquals(datetime.date(2007,11,7), ibmValues[2][0])
        self.assertAlmostEquals(0.4 * 1377955258 * 0.68189, ibmValues[2][1], 2)
        self.modelDB.createCurrencyCache(self.marketDB)
        
    def testGetQuarterlyTotalDebt(self):
        self.modelDB.createCurrencyCache(self.marketDB)
        convertTo = { self.sid1: 1}
        val = self.modelDB.getQuarterlyTotalDebt(
            datetime.date(2000,1,1), datetime.date(2001,12,31),
            [self.sid1], datetime.date(2009,1,20), self.marketDB, convertTo)
        self.assertEquals(1, len(val))
        sid1Vals = val[0]
        self.assertEquals(8, len(sid1Vals))
        self.assertEquals(datetime.date(2000,1,31), sid1Vals[0][0])
        self.assertAlmostEquals(12.048 + 13.116, sid1Vals[0][1])
        self.assertEquals(1, sid1Vals[0][2])
        self.assertEquals(datetime.date(2000,7,31), sid1Vals[2][0])
        self.assertAlmostEquals(77.758 + 18.785, sid1Vals[2][1])
        self.assertEquals(1, sid1Vals[2][2])
        
        convertTo = { self.sid1: 0}
        val = self.modelDB.getQuarterlyTotalDebt(
            datetime.date(2000,1,1), datetime.date(2001,12,31),
            [self.sid1], datetime.date(2009,1,20), self.marketDB, convertTo)
        self.assertEquals(1, len(val))
        sid1Vals = val[0]
        self.assertEquals(8, len(sid1Vals))
        self.assertEquals(datetime.date(2000,1,31), sid1Vals[0][0])
        self.assertAlmostEquals(17.4561833472 + 19.0035940224, sid1Vals[0][1])
        self.assertEquals(0, sid1Vals[0][2])
        self.assertEquals(datetime.date(2000,7,31), sid1Vals[2][0])
        self.assertAlmostEquals(115.5285441584 + 27.909716068, sid1Vals[2][1])
        self.assertEquals(0, sid1Vals[2][2])
    
    def testGetRiskFreeRateHistory(self):
        isoCodeList = ['USD', 'AED', 'EUR']
        dateList = [datetime.date(2005,5,8), datetime.date(2005,5,10),
                    datetime.date(2005,5,11),  datetime.date(2005,5,12)]
        valAnn = self.modelDB.getRiskFreeRateHistory(
            isoCodeList, dateList, self.marketDB, annualize=True)
        self.assertAlmostEquals(0.0323, valAnn.data[0,0])
        self.assertAlmostEquals(0.0325, valAnn.data[0,1])
        self.assertAlmostEquals(0.0326, valAnn.data[0,2])
        self.assertAlmostEquals(0.0326813, valAnn.data[0,3])
        self.assertAlmostEquals(0.0323, valAnn.data[1,0])
        self.assertAlmostEquals(0.0325, valAnn.data[1,1])
        self.assertAlmostEquals(0.0326, valAnn.data[1,2])
        self.assertAlmostEquals(0.0326813, valAnn.data[1,3])
        self.assertAlmostEquals(0.0212463, valAnn.data[2,0])
        self.assertAlmostEquals(0.0212125, valAnn.data[2,1])
        self.assertAlmostEquals(0.0212375, valAnn.data[2,2])
        self.assertAlmostEquals(0.0212575, valAnn.data[2,3])
        valDly = self.modelDB.getRiskFreeRateHistory(
            isoCodeList, dateList, self.marketDB, annualize=False)
        def dailyRate(x):
            return (pow(1.0 + x, 1.0/252.0) - 1.0)
        self.assertAlmostEquals(dailyRate(valAnn.data[0,0]), valDly.data[0,0])
        self.assertAlmostEquals(dailyRate(valAnn.data[0,1]), valDly.data[0,1])
        self.assertAlmostEquals(dailyRate(valAnn.data[0,2]), valDly.data[0,2])
        self.assertAlmostEquals(dailyRate(valAnn.data[0,3]), valDly.data[0,3])
        self.assertAlmostEquals(dailyRate(valAnn.data[1,0]), valDly.data[1,0])
        self.assertAlmostEquals(dailyRate(valAnn.data[1,1]), valDly.data[1,1])
        self.assertAlmostEquals(dailyRate(valAnn.data[1,2]), valDly.data[1,2])
        self.assertAlmostEquals(dailyRate(valAnn.data[1,3]), valDly.data[1,3])
        self.assertAlmostEquals(dailyRate(valAnn.data[2,0]), valDly.data[2,0])
        self.assertAlmostEquals(dailyRate(valAnn.data[2,1]), valDly.data[2,1])
        self.assertAlmostEquals(dailyRate(valAnn.data[2,2]), valDly.data[2,2])
        self.assertAlmostEquals(dailyRate(valAnn.data[2,3]), valDly.data[2,3])
        isoCodeList = ['TRY', 'EUR', 'TRL']
        dateList = [datetime.date(2004,12,10), datetime.date(2005,1,10)]
        valAnn = self.modelDB.getRiskFreeRateHistory(
            isoCodeList, dateList, self.marketDB, annualize=True)
        self.assertTrue(valAnn.data[0,0] is MA.masked)
        self.assertAlmostEquals(0.209, valAnn.data[0,1])
        self.assertAlmostEquals(0.02171, valAnn.data[1,0])
        self.assertAlmostEquals(0.021445, valAnn.data[1,1])
        self.assertAlmostEquals(0.2361, valAnn.data[2,0])
        self.assertTrue(valAnn.data[2,1] is MA.masked)
    
    def testGetTradingCurrency(self):
        mymid1 = ModelID.ModelID(string='DRRKXKJYU2')
        mymid2 = ModelID.ModelID(string='DVCWWV7CV5')
        tccy = self.modelDB.getTradingCurrency(
            datetime.date(2008, 6, 23), [mymid1, mymid2], self.marketDB,
            returnType='id')
        self.assertEquals(2, len(tccy))
        self.assertEquals(16, tccy[mymid1])
        self.assertEquals(191, tccy[mymid2])
        tccy = self.modelDB.getTradingCurrency(
            datetime.date(2008, 6, 22), [mymid1, mymid2], self.marketDB,
            returnType='code')
        self.assertEquals(2, len(tccy))
        self.assertEquals('DKK', tccy[mymid1])
        self.assertEquals('EUR', tccy[mymid2])
        mysid1 = ModelDB.SubIssue(string='DRRKXKJYU211')
        tccy = self.modelDB.getTradingCurrency(
            datetime.date(2000, 12, 31), [mysid1, mymid2], self.marketDB,
            returnType='id')
        self.assertEquals(2, len(tccy))
        self.assertEquals(16, tccy[mysid1])
        self.assertEquals(180, tccy[mymid2])
        
    def testMarketIdentifierHistory(self):
        mysid1 = ModelDB.SubIssue(string='DHM37RXF6611')
        val = self.modelDB.loadMarketIdentifierHistory(
            [self.sid2, self.sid3, mysid1], self.marketDB,
            'asset_dim_sedol', 'id')
        self.assertEquals(3, len(val))
        mysid1Hist = val[mysid1]
        self.assertEquals(3, len(mysid1Hist))
        self.assertEquals(datetime.date(1999, 6, 23), mysid1Hist[0].fromDt)
        self.assertEquals(datetime.date(2000, 3, 6), mysid1Hist[0].thruDt)
        self.assertEquals('5702903', mysid1Hist[0].id)
        self.assertEquals(datetime.date(2000, 3, 6), mysid1Hist[1].fromDt)
        self.assertEquals(datetime.date(2006, 7, 11), mysid1Hist[1].thruDt)
        self.assertEquals('5898590', mysid1Hist[1].id)
        self.assertEquals(datetime.date(2006, 7, 11), mysid1Hist[2].fromDt)
        self.assertEquals(datetime.date(2999, 12, 31), mysid1Hist[2].thruDt)
        self.assertEquals('B179N29', mysid1Hist[2].id)
    
    def testLoadACPHistory(self):
        self.modelDB.createCurrencyCache(self.marketDB)
        convertTo = None
        dateList = [datetime.date(2005,5,9),
                    datetime.date(2005,5,10), datetime.date(2005,5,8), 
                    datetime.date(2005,5,11),  datetime.date(2005,5,12)]
        val = self.modelDB.loadACPHistory(
            dateList, [self.sid1, self.sid3], self.marketDB, convertTo)
        self.assertEquals((2,5), val.data.shape)
        self.assertAlmostEquals(2.3/0.1428571, val.data[0,0])
        self.assertAlmostEquals(16.7, val.data[0,1])
        self.assertTrue(val.data[0,2] is MA.masked)
        self.assertAlmostEquals(16.21, val.data[0,3])
        self.assertAlmostEquals(15.98, val.data[0,4])
        self.assertAlmostEquals(1183.0, val.data[1,0])
        self.assertAlmostEquals(1178.0, val.data[1,1])
        self.assertTrue(val.data[1,2] is MA.masked)
        self.assertAlmostEquals(1166.0, val.data[1,3])
        self.assertAlmostEquals(1145.0, val.data[1,4])
    
    def testLoadACPHistorySingleAsset(self):
        # Test where the result matrix has only one asset
        # first with a masked value
        mySid = ModelDB.SubIssue(string='D3RH87H1N811')
        val = self.modelDB.loadACPHistory(
            [datetime.date(2008, 12, 25)], [mySid], self.marketDB,
            None)
        self.assertEquals((1,1), val.data.shape)
        self.assertTrue(val.data[0,0] is MA.masked)
        
        # then with a real value
        val = self.modelDB.loadACPHistory(
            [datetime.date(2008, 12, 12)], [mySid], self.marketDB,
            None)
        self.assertEquals((1,1), val.data.shape)
        self.assertEquals(67.0, val.data[0,0])
    
    def testLoadACPHistorySingleAssetAdd(self):
        # Test where the result matrix has only one asset
        # and then more assets are added
        mySid1 = ModelDB.SubIssue(string='D3RH87H1N811')
        dates = [datetime.date(2008, 12, 12), datetime.date(2008, 12, 25)]
        val = self.modelDB.loadACPHistory(
            dates, [mySid1], self.marketDB,
            None)
        self.assertEquals((1,2), val.data.shape)
        self.assertFalse(val.data[0,0] is MA.masked)
        self.assertEquals(67.0, val.data[0,0])
        self.assertTrue(val.data[0,1] is MA.masked)
        
        mySid2 = ModelDB.SubIssue(string='D11D4UU9B311')
        val = self.modelDB.loadACPHistory(
            dates, [mySid2, mySid1], self.marketDB,
            None)
        self.assertEquals((2,2), val.data.shape)
        self.assertFalse(val.data[0,0] is MA.masked)
        self.assertEquals(394.0, val.data[0,0])
        self.assertFalse(val.data[0,1] is MA.masked)
        self.assertEquals(395.0, val.data[0,1])
        self.assertFalse(val.data[1,0] is MA.masked)
        self.assertEquals(67.0, val.data[1,0])
        self.assertTrue(val.data[1,1] is MA.masked)

        
    def testLoadVolumeHistory(self):
        self.modelDB.createCurrencyCache(self.marketDB)
        convertTo = None
        dateList = [datetime.date(2005,5,8), datetime.date(2005,5,10),
                    datetime.date(2005,5,11),  datetime.date(2005,5,12)]
        val = self.modelDB.loadVolumeHistory(
            dateList, [self.sid2, self.sid3], convertTo)
        self.assertEquals((2,4), val.data.shape)
        self.assertTrue(val.data[0,0] is MA.masked)
        self.assertAlmostEquals(59703700.0, val.data[0,1])
        self.assertAlmostEquals(72219000.0, val.data[0,2])
        self.assertAlmostEquals(27093600.0, val.data[0,3])
        self.assertTrue(val.data[1,0] is MA.masked)
        self.assertAlmostEquals(20732800.0, val.data[1,1])
        self.assertAlmostEquals(29383200.0, val.data[1,2])
        self.assertAlmostEquals(32976000.0, val.data[1,3])
        
        convertTo = 25
        dateList = [datetime.date(2005,5,8), datetime.date(2005,5,10),
                    datetime.date(2005,5,11),  datetime.date(2005,5,12)]
        val = self.modelDB.loadVolumeHistory(
            dateList, [self.sid2, self.sid3], convertTo)
        self.assertEquals((2,4), val.data.shape)
        self.assertTrue(val.data[0,0] is MA.masked)
        self.assertAlmostEquals(59703700.0, val.data[0,1])
        self.assertAlmostEquals(72219000.0, val.data[0,2])
        self.assertAlmostEquals(27093600.0, val.data[0,3])
        self.assertTrue(val.data[1,0] is MA.masked)
        self.assertAlmostEquals(20732800.0, val.data[1,1])
        self.assertAlmostEquals(29383200.0, val.data[1,2])
        self.assertAlmostEquals(32976000.0, val.data[1,3])
        
        convertTo = {self.sid1: 191, self.sid3: 1}
        dateList = [datetime.date(2005,5,10), datetime.date(2005,5,11),
                    datetime.date(2005,5,12), datetime.date(2005,5,13)]
        val = self.modelDB.loadVolumeHistory(
            dateList, [self.sid1, self.sid3], convertTo)
        self.assertEquals((2,4), val.data.shape)
        self.assertAlmostEquals(9667046.56, val.data[0,0], 2)
        self.assertAlmostEquals(8941268.427, val.data[0,1], 2)
        self.assertAlmostEquals(8317655.42, val.data[0,2], 2)
        self.assertAlmostEquals(8578149.01, val.data[0,3], 2)
        self.assertAlmostEquals(196417.03, val.data[1,0], 2)
        self.assertAlmostEquals(277999.905, val.data[1,1], 2)
        self.assertAlmostEquals(309125.849, val.data[1,2], 2)
        self.assertAlmostEquals(77146.99, val.data[1,3], 2)
        dateList = [datetime.date(2005,5,16)]
        val = self.modelDB.loadVolumeHistory(
            dateList, [self.sid1, self.sid3], convertTo)
        self.assertEquals((2,1), val.data.shape)
        self.assertAlmostEquals(6176806.5, val.data[0,0], 2)
        self.assertAlmostEquals(91261.37, val.data[1,0], 2)

if __name__ == '__main__':
    logging.config.fileConfig('log.config')
    configFile_ = open('test.config')
    config_ = configparser.ConfigParser()
    config_.read_file(configFile_)
    configFile_.close()
    import sys
    if len(sys.argv) > 1:
        print(sys.argv)
        suite = unittest.TestSuite(map(TimeSeriesTest, sys.argv[1:]))
    else:
        suite = unittest.TestLoader().loadTestsFromTestCase(TimeSeriesTest)
    unittest.TextTestRunner().run(suite)
