
import datetime
import logging
import optparse
import os
from marketdb import MarketDB
from riskmodels import ModelDB
from riskmodels import ModelID
from riskmodels import Utilities

def writeBenchmarkCSV(bcomp, outFile, options):
    """Write a CSV benchmark file to outFile.
    """
    EPSILON=0.00005
    totalwt = 0.0
    for (asset, weight) in bcomp:
        totalwt +=  weight
        if options.applyCorrections:
            outFile.write("%s, %.12g\n" % (asset.getPublicID(), weight))
        else:
            outFile.write("%s, %.8g\n" % (asset.getPublicID(), weight))
    if abs(1.0-totalwt) >= EPSILON:
        logging.warning('Total weight of the constituents is %.8g', totalwt)
    else:
        logging.info('Total weight of the constituents is %.8g', totalwt)

def getIndexNames(cursor, date, indexids):
    """Get list of index names given the list of IDs. To be used by other methods later
    """
    query="""select name from index_member where id in (%s) and from_dt <= :dt and :dt < thru_dt
    """ % (','.join(str(i) for i in indexids))
    cursor.execute(query, dt=date)
    return [r[0] for r in cursor.fetchall()]

def getAllSpecialIndexNames(cursor, date):
    """Get list of index names which are in the idx_rebal_ids table
    """
    query="""select name from index_member im, idx_rebal_ids i where i.index_id=im.id
              and i.from_dt <= :dt and i.thru_dt > :dt
              and im.from_dt <= :dt and im.thru_dt > :dt"""
    cursor.execute(query, dt=date)
    return [r[0] for r in cursor.fetchall()]
    
def getModelDBIDs(cursor, date, marketIDs):
    """Get list of modelDB ids (without the first letter for a list of marketDB ids
    """
    if len(marketIDs)==0:
        return dict()

    query = """select marketdb_id, modeldb_id from issue_map where marketdb_id in (%s)
               and from_dt <= :dt and :dt < thru_dt""" % (','.join("'%s'" % m for m in marketIDs))
    cursor.execute(query, dt=date)
    return dict((i, ModelID.ModelID(string=j)) for (i,j) in cursor.fetchall())

def getCorrectionMarketData(cursor, date, benchName):
    """Get marketdb axioma id data for index corrections
    """
    query = """select prior_axioma_id, new_axioma_id, weight from idx_rebal_corrections where from_dt <= :dt and :dt < thru_dt
               and index_id = (select id from index_member where name='%s')""" % benchName
    cursor.execute(query, dt=date)
    results=cursor.fetchall()
    return results
              

def applyCorrections(bcomp, corrections, modelDict, date, benchName):
    """given the list of const/weights, and a set of corrections to apply and the modelDict
       fix the input so that it applies our business logic of substituting instruments and/or adding cash assets
    """
    tempDict=dict([(asset,weight) for (asset,weight) in bcomp])
    for (oldid, newid, wt) in corrections:
        if newid not in modelDict:
            logging.warning('Correction %s->%s (%s) is ignored since the new axiomaid is not in the database on %s for %s', oldid, newid, wt, date, benchName)
            continue
        if oldid in modelDict:
            key = modelDict[oldid]
            if key in tempDict:
                logging.info( 'Deleted %s', key)
                del tempDict[key]
            else:
                logging.warning('Key %s not found in index constituent for %s on %s', key, benchName, date)
        # new key is the ModelID of the corrected AxiomaID i.e. TO_AXIOMA_ID
        newkey = modelDict.get(newid, None)
        if not newkey:
            logging.warning('To ID %s has no modelID on %s for %s', newid, date, benchName)
            continue
        if newkey not in tempDict:
            tempDict[newkey] = 0.0
        tempDict[newkey] += wt
        logging.info('Modified weight of %s to be %g', newkey, tempDict[newkey])
    return list(tempDict.items())
            

if __name__ == '__main__':
    usage = "usage: %prog [options] <benchmark name> <YYYY-MM-DD> <YYYY-MM-DD>" 
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addDefaultCommandLine(cmdlineParser)
    cmdlineParser.add_option("-d", "--directory", action="store",
                             default='.', dest="targetDir",
                             help="directory for output files")
    cmdlineParser.add_option("-p", "--prefix", action="store",
                             default="", dest="prefix",
                             help="Prefix to file name")
    cmdlineParser.add_option("-s", "--suffix", action="store",
                             default="", dest="suffix",
                             help="suffix to file name")
    cmdlineParser.add_option("-o", "--outname", action="store",
                             default=None, dest="outname",
                             help="Name of output files, defaults to benchmark name")
    cmdlineParser.add_option("-n", "--nextday", action="store_true",
                             default=False, dest="nextDay",
                             help="Next day (day > specified day) benchmarks")
    cmdlineParser.add_option("--num-days-forward", action="store",
                             default=5, dest="numDaysForward",
                             help="Next day (day > specified day) benchmarks")
    cmdlineParser.add_option("-m", "--monthend", action="store_true",
                             default=False, dest="monthend",
                             help="The frequency of writing, defaults to daily")
    cmdlineParser.add_option("-f", "--family", action="store_true",
                             default=False, dest="family",
                             help="Indicates that the benchmark name is an index-family name")
    cmdlineParser.add_option("-i", "--index-id", action="store_true",
                             default=False, dest="indexID",
                             help="Indicates that the benchmark name is actuall an index ID")
    cmdlineParser.add_option("--target-sub-dirs", action="store_true",
                                 default=False, dest="appendDateDirs",
                                 help="Append yyyy/mm to end of output directory path")
    cmdlineParser.add_option("--apply-index-corrections", action="store_true",
                                 default=False, dest="applyCorrections",
                                 help="Apply special corrections for Index team")

    (options, args) = cmdlineParser.parse_args()
    if len(args) < 2 or len(args) > 3:
        cmdlineParser.error("Incorrect number of arguments")

    Utilities.processDefaultCommandLine(options, cmdlineParser)
    modelDB = ModelDB.ModelDB(sid=options.modelDBSID, user=options.modelDBUser,
                              passwd=options.modelDBPasswd)
    marketDB = MarketDB.MarketDB(sid=options.marketDBSID, user=options.marketDBUser, passwd=options.marketDBPasswd)
    marketcursor = marketDB.dbCursor

    benchName = args[0]
    #if options.outname is None:
    #   options.outname = benchName.replace('/', ' ')
    startDate = Utilities.parseISODate(args[1])
    if len(args) == 2:
        endDate = startDate
    else:
        endDate = Utilities.parseISODate(args[2])
    if options.family:
           familyNames=benchName.split(',')

    d = startDate
    dates = []
    dayInc = datetime.timedelta(1)

    while d <= endDate:
        if d.isoweekday() in [6,7]:
    #       logging.info('Ignoring weekend %s', str(d))
           d += dayInc
           continue
        else:
           dates.append(d)
           d += dayInc

    if options.monthend:
        dates = [prev for (prev, next) in zip(dates[:-1], dates[1:]) \
                if (next.month > prev.month or next.year > prev.year)]

    for d in dates:
        logging.info('Processing %s' % str(d))
        if options.family:
           # use the family rather than the index name
           benchNames=[]
           for familyName in familyNames:
               indexFamily=marketDB.getIndexFamily(familyName)
               results=marketDB.getIndexFamilyIndices(indexFamily,d, True)
               benchNames=benchNames+ [i.name for i in results]
        elif options.indexID:
           benchNames=getIndexNames(marketcursor, d, benchName.split(','))
        elif options.applyCorrections and benchName=='ALL':
            benchNames=getAllSpecialIndexNames(marketcursor, d)
        else:
           benchNames=[benchName]
        target=options.targetDir
        if options.appendDateDirs:
             target = os.path.join(target, '%04d' % d.year, '%02d' % d.month)
             try:
                os.makedirs(target)
             except OSError as e:
                if e.errno != 17:
                   raise
                else:
                   pass

        issueMapPairs = modelDB.getIssueMapPairs(d)

        for benchmark in benchNames:
            # if there is only benchmark name that we are processing, then use the outfile
            # otherwise, have to use the index name rather than what was supplied
            if len(benchNames) == 1 and options.outname:
                outFileName = '%s/%s-%d%02d%02d.csv' % (
                    target, options.outname, d.year, d.month, d.day)
            else:
                # replace /,space,parens from the index name and &
                bName=benchmark.replace('/','-').replace(' ','-').replace('(','').replace(')','').replace('&','')
                outFileName='%s/%s%s-%02d%02d%02d%s.csv' % (target, options.prefix,bName,d.year,d.month,d.day, options.suffix)
            logging.info("Processing %s", benchmark)
            # if next day is specified, look up to 4 days ahead of the given day.  If you don't find anything, bail
            if options.nextDay:
                for i in range(options.numDaysForward)[1:]:
                    d1=d+datetime.timedelta(days=i)
                    if d1.isoweekday() in [6,7]:
                        continue
                    bcomp = modelDB.getIndexConstituents(benchmark, d1, marketDB, issueMapPairs=issueMapPairs)
                    if len(bcomp) >0 :
                        logging.info("Found constituents for %s on %s/%s %s/%s", benchmark, str(d1), d1.isoweekday(),str(d),d.isoweekday())
                        break

            else:
                bcomp = modelDB.getIndexConstituents(benchmark, d, marketDB, issueMapPairs=issueMapPairs)

            if options.applyCorrections:
                # this index might have correction data that needs to be dealt with specially.  
                # get those ids from the database and apply the business logic as needed

                # first get the ids (market axiomaids) from the database for this benchmark and this date
                corrections = getCorrectionMarketData(marketcursor, d, benchmark)
                modelcursor = modelDB.dbCursor
                #print 'corrections=', corrections
                # get the list of from ids and to ids
                modelDict = getModelDBIDs(modelcursor, d, [c[0] for c in corrections] + [c[1] for c in corrections])
                #print 'dictionary=', modelDict
                bcomp = applyCorrections(bcomp, corrections, modelDict, d, benchmark)

            if len(bcomp) > 0:
                if not os.path.exists(os.path.dirname(outFileName)):
                    os.makedirs(os.path.dirname(outFileName))
                outFile = open(outFileName, 'w')
                writeBenchmarkCSV(bcomp, outFile, options)
                outFile.close()
            else:
                logging.info('No data for %s on %s, skipping', benchmark, str(d))
        d += dayInc
    marketDB.finalize()
    modelDB.finalize()
