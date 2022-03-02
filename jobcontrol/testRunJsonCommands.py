import unittest
import os
from os import walk

import riskmodels.runJsonCommands as rjc

from lib.common import assetAddtitionNonDjango

class RunJsonCommandsTest(unittest.TestCase):

    def test2(self):
        sid='glsdg'
        assetAdditionType=0
        bulkFile=None
        csvString="BLB1GR7|US|2021-05-06|9999-12-31"
        commandsJsonFileName="commandsJsonFileName.json"
        forceCheckTqaDuplicates = True
        runQaReports = True
        username='npermikova'
        ref='test'
        testOnly=True
        
        json_info = assetAddtitionNonDjango.addAssets(sid, assetAdditionType, bulkFile, csvString, commandsJsonFileName, 
                                                     forceCheckTqaDuplicates, username, ref, runQaReports, testOnly=testOnly)
        
        print(json_info)

    def _test1(self):
#         self.assertEqual(1, 1)
#         for (dirpath, dirnames, filenames) in walk('.'):
            filenames = [
#                         'test_macAssetCoverage.json',
                        'test03_assetAddition.json',
#                         'test_axiomaIdViewer.json',
#                         'test_axiomaIdViewer_01.json',
#                         'test_bulkTransferUtility.json',
#                         'test_idcFundAddition.json',
#                         'test_macCurvesHistory.json',
#                         'test_transferUtilty.json'
            ]
            try:
                for fileName in filenames:
                    if fileName.startswith('test'):
                        print('run commands from ', fileName)
                        rjc.runJsonCommands(fileName,None,False)
            except Exception as error:
                self.fail("Failed with %s" % error)
                
#         sel

    
def suite():
    return unittest.TestLoader().loadTestsFromTestCase(RunJsonCommandsTest)


if __name__ == '__main__':
    unittest.main()



