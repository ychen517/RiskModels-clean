import pandas
import datetime
import optparse
import logging
import configparser
import sys
import os

from riskmodels import ModelDB, Connections
from marketdb.Utilities import listChunkIterator, parseISODate

if __name__=='__main__':
    #parse arguments and options
    usage = 'usage: %prog [options] <config> <startdate> [<enddate>]'
    cmdlineParser = optparse.OptionParser(usage=usage)
    cmdlineParser.add_option('--target', action='store', 
                             default='/axioma/projects/Indexing/STOXXEquityCountry', dest='targetDir',
                             help='directory where the parsed files are written')
    cmdlineParser.add_option('--sub-dirs', action='store_true', 
                             default=False, dest='subDirs',
                             help='directory where the parsed files are written')
    cmdlineParser.add_option('-l', action='store',
                             dest='logConfigFile', default='log.config',
                             help='')
    
    status = 0
    (options, args) = cmdlineParser.parse_args()
    logging.config.fileConfig(options.logConfigFile)
    if len(args) < 2 or len(args) >3:
        cmdlineParser.error('Incorrect number of arguments')
    
    config = configparser.SafeConfigParser()
    config.read(args[0])
    connections = Connections.createConnections(config)

    modelDB = connections.modelDB
    marketDB = connections.marketDB

    logging.config.fileConfig(options.logConfigFile)

    startDate = parseISODate(args[1])
    if len(args) == 3:
        endDate = parseISODate(args[2])
    else:
        endDate = startDate

    modelDB.stoxxCache = ModelDB.FromThruCache()
    rmgList = modelDB.getAllRiskModelGroups()
    rmgMap = dict([(rmg.mnemonic, rmg) for rmg in rmgList])

    while startDate <= endDate:
        if options.subDirs:
            targetDir = '%s/%4s/%02d' % (options.targetDir, startDate.year, startDate.month)
        else:
            targetDir = options.targetDir
        if not os.path.exists(targetDir):
            logging.info('Making dir %s', targetDir)
            os.makedirs(targetDir)
        if startDate.isoweekday() > 5 or (startDate.month == 1 and startDate.day == 1):
            startDate += datetime.timedelta(1)
            continue
        print(startDate, targetDir)
        assets = [i[0] for i in modelDB.getIssueMapPairs(startDate)]
        modelDB.loadMarketIdentifierHistoryCache(assets, marketDB, 'asset_dim_stoxx_country', 'ctry', cache=modelDB.stoxxCache)
        stoxxIdMap = [(mid, modelDB.stoxxCache.getAssetValue(mid, startDate)) for mid in assets]
        stoxxIdMap = dict([(i, j.id) for (i,j) in stoxxIdMap if j is not None])
        stoxxMdlMap = dict([(j, i) for (i, j) in stoxxIdMap.items()])

        stoxxIds = [stoxxIdMap[mid] for mid in assets if mid in stoxxIdMap]
        logging.info('Mapped STOXX IDs for %d out of %d assets on %s', len(stoxxIds), len(assets), startDate)

        values = pandas.Series(index=stoxxIdMap.keys())
        for a in stoxxIdMap:
            values[a] = modelDB.stoxxCache.getAssetValue(a, startDate).id
        #for a in set(assets) - set(stoxxIdMap.keys()):
        #    values = values.drop(a)
        values.sort_index(ascending=True, inplace=True) 
        filename='%s/STOXX_country_mapping-%04d%02d%02d.csv' % (targetDir,startDate.year, startDate.month, startDate.day)
        with open(filename,'w') as out:
            out.write('METAGROUP NAME|STOXX.Countries\n')
            out.write('METAGROUP DESC|STOXX Country Classification\n')
            out.write('NAME PREFIX|STOXX.\n\n')
        values.rename(index=dict([(mid, mid.getPublicID()) for mid in values.index])).to_csv(filename, sep='|', mode='a')
        startDate += datetime.timedelta(1)

    modelDB.finalize()
    marketDB.finalize()
    sys.exit(status)
