

import logging
import optparse

import riskmodels
from marketdb import MarketDB
from riskmodels import ModelDB
from riskmodels import Utilities
from riskmodels.LegacyCurrencyRisk import CurrencyRiskModel
from riskmodels.MFM import StatisticalFactorModel
from riskmodels.factor_matrix import writeFactorMatrix

logger = logging.getLogger('RiskModels.factor_matrix')

if __name__=='__main__':
    usage = "usage: %prog [options] <YYYY-MM-DD> [<YYYY-MM-DD>]"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addDefaultCommandLine(cmdlineParser)
    cmdlineParser.add_option("--models", action="store",
                             dest="models", default='',
                             help="comma-separated list of models to process")
    cmdlineParser.add_option("--ReturnFreq", action="store",
                             dest="returnfreq", default='daily',
                             help="specify return frequency, daily(default)/weekly/monthly")
    cmdlineParser.add_option("--withvolatility", action="store_true",
                             dest="withvolatility", default=False,
                             help="extract with volatility file")
    (options, args) = cmdlineParser.parse_args()
    Utilities.processDefaultCommandLine(options, cmdlineParser)
    modelDB = ModelDB.ModelDB(sid=options.modelDBSID, user=options.modelDBUser,
                              passwd=options.modelDBPasswd)
    marketDB = MarketDB.MarketDB(sid=options.marketDBSID, user=options.marketDBUser,
                                 passwd=options.marketDBPasswd)
    if len(options.models) == 0:
        modelList = []
        modelInfoList = []
    else:
        modelList = [riskmodels.getModelByName(model)(modelDB, marketDB)
                     for model in options.models.split(',')]
        modelInfoList = [modelDB.getRiskModelInfo(rm.rm_id, rm.revision)
                         for rm in modelList]
    logger.debug('logger level=%s', logger.getEffectiveLevel())
    if len(args) < 1 or len(args) > 2:
        cmdlineParser.error("Incorrect number of arguments")
    startDate = Utilities.parseISODate(args[0])
    if len(args) == 2:
        endDate = Utilities.parseISODate(args[1])
    else:
        endDate = startDate
    for model in modelList:
        if issubclass(model.__class__, StatisticalFactorModel) or \
                issubclass(model.__class__, CurrencyRiskModel):
            continue
        factorFile = open('factor_return_%s_%s.csv' % (options.returnfreq, model.name), 'w')
        if options.withvolatility:
            volatilityFile = open('factor_volatility_%s.csv' % model.name, 'w')
        else:
            volatilityFile = None
        writeFactorMatrix(model, startDate, endDate, modelDB, factorFile, volatilityFile,
                          options.returnfreq)
        factorFile.close()
        if volatilityFile is not None:
            volatilityFile.close()
