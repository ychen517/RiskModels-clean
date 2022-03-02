
import unittest
try:
    import numpy.ma as MA
except:
    import numpy.core.ma as MA
import numpy
from riskmodels import Utilities

class UtilitiesTest(unittest.TestCase):

    def testRobustAverageNoOutliers(self):
        y = numpy.ones((10,))
        weights = numpy.random.rand(10)
        trueavg = numpy.average(y, weights=weights)
        robustavg = Utilities.robustAverage(y, weights)
        self.assertAlmostEqual(trueavg, robustavg, 4)

    def testRobustAverageWithRpy(self):
        y = numpy.ones((1000,))
        y[:50] = numpy.ones((50,))*10000.0
        numpy.random.seed(1)
        weights = numpy.random.rand(1000)
        rpyavg = Utilities.robustAverageRpy(y,weights)
        avg = Utilities.robustAverage(y,weights)
        self.assertAlmostEqual(rpyavg, avg, 2)

def suite():
    return unittest.TestLoader().loadTestsFromTestCase(LotSelectionTest)


if __name__ == '__main__':
    unittest.main()



# vim: set softtabstop=4 shiftwidth=4:
