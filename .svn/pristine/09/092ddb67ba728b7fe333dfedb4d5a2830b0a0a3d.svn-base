
import configparser
import datetime
import logging
import numpy.ma as MA
import unittest
from marketdb import MarketDB
import riskmodels
from riskmodels import ModelDB
from riskmodels import ModelID
from riskmodels import StyleExposures
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
        self.ukRM1 = riskmodels.getModelByName('GBAxioma2009MH')(
            self.modelDB, self.marketDB)
        self.sid1 = ModelDB.SubIssue(string='DSVJCWHHN611')
        self.sid2 = ModelDB.SubIssue(string='D11D9499T911')
        self.sid3 = ModelDB.SubIssue(string='D11D4UU9B311')
        self.axid2 = ModelID.ModelID(string='GSFM9Z1BK8')
        self.axid3 = ModelID.ModelID(string='GY6W2DWMW3')
        
    def tearDown(self):
        self.modelDB.revertChanges()
        self.modelDB.finalize()
        
    def testGetProxiedDividends(self):
        convertTo = None
        sidIBM = ModelDB.SubIssue(string='DRM8T7MA5511')
        sidNoIncome = ModelDB.SubIssue(string='DF71TUFXR411')
        data = Utilities.Struct()
        data.universe=[sidIBM, sidNoIncome]
        payout = StyleExposures.generate_proxied_dividend_payout(
            datetime.date(2005,5,5), data, self.ukRM1, self.modelDB,
            self.marketDB)
        self.assertEquals((2,) , payout.shape)
        self.assertAlmostEquals(0.16249570, payout[0], 8)
        self.assertTrue(payout[1] is MA.masked)
        payout = StyleExposures.generate_proxied_dividend_payout(
            datetime.date(2005,5,5), data, self.ukRM1, self.modelDB,
            self.marketDB, useQuarterlyData=False)
        self.assertEquals((2,) , payout.shape)
        self.assertAlmostEquals(0.14924727, payout[0], 8)
        #self.assertTrue(payout[1] is MA.masked)
        

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
