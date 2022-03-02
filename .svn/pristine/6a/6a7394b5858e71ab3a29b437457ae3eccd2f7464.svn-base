
import datetime
import logging
import optparse
import os
import time
import sys
import shutil
from marketdb import MarketDB, MarketID
from riskmodels import ModelDB
from riskmodels import Utilities
from marketdb import Utilities as mktUtils

def writeSTOXXFile(options, dt, marketDB, modelDB, expanded=False):
    target=options.targetDir
    if options.appendDateDirs:
         target = os.path.join(target, '%04d' % dt.year, '%02d' % dt.month)
         try:
            os.makedirs(target)
         except OSError as e:
            if e.errno != 17:
               raise
            else:
               pass
    dtstr = str(d).replace('-','')

    errorCase=False
 
    # get list of axioma-id and stoxx-id mappings for the given date
    query="""select axioma_id, id from asset_dim_stoxx_active_int where from_dt <= :dt and :dt  < thru_dt"""
    marketDB.dbCursor.execute(query, dt=dt)
    stoxxresults=marketDB.dbCursor.fetchall()
    axids=[r[0] for r in stoxxresults]
    axidList = [MarketID.MarketID(string=a) for a in axids]

    if expanded:
        sedolMap = dict((i.getIDString(),j) for (i,j) in marketDB.getSEDOLs(dt, axidList))
        ricMap = dict((i.getIDString(),j) for (i,j) in marketDB.getRICs(dt, axidList))
        query="""select INTERNAL_NUMBER, SEDOL, RIC, EXCHANGE from STOXX_CONSTITUENT_CLOSE where INDEX_SYMBOL='TW1P' and DT=:dt"""
        vendorDB.dbCursor.execute(query, dt=dt)
        stoxxMap = {}
        for row in vendorDB.dbCursor.fetchall():
            stoxxid, sedol, ric, exch = row
            stoxxMap[stoxxid] = (sedol, ric, exch) 

    INCR = 500
    argList = ['axid%d' % i for i in range(INCR)]
    query = """SELECT modeldb_id, marketdb_id FROM issue_map
      WHERE marketdb_id in (%(args)s) and from_dt <= :dt and :dt <= thru_dt""" % {
        'args': ','.join([':%s' % arg for arg in argList])}
    defaultDict = dict([(arg, None) for arg in argList])
    marketDict={}
    mdlCur = modelDB.dbCursor
    for axidChunk in mktUtils.listChunkIterator(axids, INCR):
        myDict = defaultDict.copy()
        myDict.update(dict(zip(argList, axidChunk)))
        myDict['dt'] = dt 
        mdlCur.execute(query, myDict)
        for res in mdlCur.fetchall():
            mdlid, marketid = res
            marketDict[marketid] = mdlid

    # for the flat files use the canned name
    stoxxFileName = '%s/AXSTOXX-%04d%02d%02d.idm' % (target, d.year, d.month, d.day)
    stoxxFile = open(stoxxFileName,'w')

    stoxxFile.write('#DataDate: %s\n' % str(d)[:10])
    gmtime = time.gmtime(time.mktime(time.gmtime()))
    utctime = datetime.datetime(year=gmtime.tm_year, month=gmtime.tm_mon, day=gmtime.tm_mday,
                                hour=gmtime.tm_hour, minute=gmtime.tm_min, second=gmtime.tm_sec)
    stoxxFile.write('#CreationTimestamp: %sZ\n' % utctime.strftime('%Y-%m-%d %H:%M:%S'))
    stoxxFile.write('#FlatFileVersion: 4.0\n')
    stoxxFile.write('#Columns: AxiomaID|STOXX-INTERNAL-NUMBER')
    if expanded:
        stoxxFile.write('|Axioma-SEDOL|Axioma-RIC|STOXX-SEDOL|STOXX-RIC|STOXX-EXCHANGE')
    stoxxFile.write('\n')

    for stoxxdata in stoxxresults:
        axid, stoxxid = stoxxdata
        if axid in marketDict:
            stoxxFile.write('%s|%s' % (marketDict.get(axid)[1:], stoxxid))
            if expanded:
                stoxxFile.write('|%s|%s' % (sedolMap.get(axid), ricMap.get(axid)))
                stoxxinfo = stoxxMap.get(stoxxid, ('','',''))
                stoxxFile.write('|%s|%s|%s' % (stoxxinfo[0], stoxxinfo[1], stoxxinfo[2]))
                # do a quick check to see if the sedol/RIC do not match
                if (sedolMap.get(axid,'') != stoxxinfo[0]) or (ricMap.get(axid,'') != stoxxinfo[1] ):
                    if stoxxinfo[0] == '':
                        logging.warn('Identifers are missing from STOXX %s %s', sedolMap.get(axid), ricMap.get(axid) )
                    else:
                        logging.warn('Identifers are different %s-%s %s-%s', sedolMap.get(axid), stoxxinfo[0], ricMap.get(axid), stoxxinfo[1])
            stoxxFile.write('\n')

    stoxxFile.close() 

    return 0


if __name__ == '__main__':
    usage = "usage: %prog [options] <YYYY-MM-DD> [<YYYY-MM-DD>]" 
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addModelAndDefaultCommandLine(cmdlineParser)

    cmdlineParser.add_option("--flat-file-dir", action="store",
                             default='.', dest="targetDir",
                             help="directory where output files are generated")
    cmdlineParser.add_option("--target-sub-dirs", action="store_true",
                                 default=False, dest="appendDateDirs",
                                 help="Append yyyy/mm to end of output and input directory path")
    cmdlineParser.add_option("--expanded-data" ,"-e", action="store_true",
                                 default=False, dest="expandedData",
                                 help="write out expanded data if needed")
    cmdlineParser.add_option("--vendordb-user", action="store",
                                 default=None, dest="vendorDBUser", help="Vendor DB User")
    cmdlineParser.add_option("--vendordb-passwd", action="store",
                                 default=None, dest="vendorDBPasswd", help="Vendor DB Password")
    cmdlineParser.add_option("--vendordb-sid", action="store",
                                 default=None, dest="vendorDBSID", help="Vendor DB SID")


    (options, args) = cmdlineParser.parse_args()
    if len(args) < 1 or len(args) > 2:
        cmdlineParser.error("Incorrect number of arguments")

    Utilities.processDefaultCommandLine(options, cmdlineParser)

    modelDB = ModelDB.ModelDB(sid=options.modelDBSID, user=options.modelDBUser, passwd=options.modelDBPasswd)
    marketDB = MarketDB.MarketDB(sid=options.marketDBSID, user=options.marketDBUser, passwd=options.marketDBPasswd)
    if options.expandedData:
        vendorDB = MarketDB.MarketDB(sid=options.vendorDBSID, user=options.vendorDBUser, passwd=options.vendorDBPasswd)

    startDate = Utilities.parseISODate(args[0])
    if len(args) == 1:
        endDate = startDate
    else:
        endDate = Utilities.parseISODate(args[1])
    d = startDate
    dates = []
    dayInc = datetime.timedelta(1)

    while d <= endDate:
        if d.isoweekday() in [6,7] or (d.month==1 and d.day==1):
           d += dayInc
           continue
        else:
           dates.append(d)
           d += dayInc

    errorCase=False
    for d in dates:
        logging.info('Processing %s' % str(d))
        status=writeSTOXXFile(options, d, marketDB, modelDB, options.expandedData)
        if status:
            errorCase=True
        d += dayInc
    marketDB.finalize()
    modelDB.finalize()
    if errorCase:
        logging.error('Exiting with errors')
        sys.exit(1)
    else:
        sys.exit(0)
