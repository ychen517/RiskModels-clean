
import datetime
import logging
import optparse
import time
import tempfile
import os
import shutil
from riskmodels import Utilities
from riskmodels import ModelDB


def getModelSeries(cursor):
    """
        get model series and country information for all actively disributed risk model series
        Use only the MH version as a proxy for the family
    """
    query="""
        select rm.mnemonic,rm.name,rm.model_region,rmg.rmg_id,rmg.MNEMONIC, rms.serial_id,greatest(rmm.from_dt, rms.from_dt), least(rmm.thru_dt, rms.thru_dt)
        from risk_model rm, risk_model_serie rms, rmg_model_map rmm,
        risk_model_group rmg
        where rm.MODEL_ID=rms.RM_ID and rms.distribute=1
        and rm.mnemonic like '%-MH'
        and rmm.rms_id=rms.SERIAL_ID
        and rmg.rmg_id=rmm.rmg_id
        order by rm.name
    """
    cursor.execute(query)
 
    return cursor.fetchall()

def getMaxTradingDate(cursor, seriesList):
    """
        get the last trading date for each series. This is used to make sure we don't overwrite a more
        recent file with an older one. 
    """
    query="""
        select rms_id, max(dt), 'max' from risk_model_instance rmi where has_exposures=1 and has_returns=1 and has_risks=1
        and rms_id in (%s)
        group by rms_id
    """ % (','.join(seriesList))

    logging.debug(query)
    cursor.execute(query)
    return cursor.fetchall()

def getPastTradingDates(cursor, seriesList, startdt, enddt):
    """
        get all the past trading dates.  Do this by a short cut.  Just look at the risk model instance table
    """
    query="""
        select distinct dt ,'past' from risk_model_instance rmi where has_exposures=1 and has_returns=1 and has_risks=1
        and rms_id in (%s)
        and dt >= :dt
        and dt <= :enddt
        order by dt
    """ % (','.join(seriesList))

    logging.debug(query)
    cursor.execute(query,dt=startdt, enddt=enddt)
    return cursor.fetchall()

def getFutureTradingDates(cursor, rmgList, dt, enddt):
    """
        for the future look at the date range provided and use the rmglist as the list of countries
        eliminate weekends since there are countries that trade on weekends and we don't really 
        generate models on weekends - yet 
    """
    query="""
       select distinct dt,'future' from rmg_calendar rmg
           where dt >= :dt and dt < :enddt
           and rmg_id in (%s)
        order by dt
    """ % ','.join(rmgList)
    logging.debug(query)
    cursor.execute(query,dt=dt,enddt=enddt)
    # get rid of weekends; easier to do in Python
    results=[res for res in cursor.fetchall() if res[0].isoweekday() not in [6,7]]
    return results

def writeDates(fh, results):
   """
       write out the results in the provided file handle
   """
   prevrow=[]
   for r in results +[['9999-12-31',None]] :
       if len(prevrow) > 0:
           if str(r[0])[:7] == str(prevrow[0])[:7]:
               eom=''
           else:
               eom='*'
           if prevrow[1]=='future':
               infuture='*'
               eom=''
           else:
               infuture=''
           f.write('%s|%s|%s\n' % ( str(prevrow[0])[:10], eom,infuture))
       prevrow=r

if __name__ == '__main__':
    usage = "usage: %prog [options] [-m model-mnemonic] <YYYY-MM-DD>"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addDefaultCommandLine(cmdlineParser)
    cmdlineParser.add_option("-d", "--directory", action="store",
                             default='.', dest="targetDir",
                             help="directory for output files")

    cmdlineParser.add_option("--directory2", action="store",
                             default=None, dest="targetDir2",
                             help="second directory for output files")

    cmdlineParser.add_option("-m", "--model-name", action="store",
                             default=None, dest="modelNames",
                             help="models to process - by mnemonic name")

    cmdlineParser.add_option("--start-date", action="store",
                             default='1980-01-01', dest="startDate",
                             help="start date")

    cmdlineParser.add_option("--num-days-forward", action="store",
                             default=40, dest="numDaysForward",
                             help="how many days ahead to get the trading calendar")

    (options, args) = cmdlineParser.parse_args()

    if len(args) < 1:
        cmdlineParser.error("Incorrect number of arguments")

    Utilities.processDefaultCommandLine(options, cmdlineParser)
    logging.info('Start')

    modelDB = ModelDB.ModelDB(sid=options.modelDBSID, user=options.modelDBUser,
                              passwd=options.modelDBPasswd)

    cursor=modelDB.dbCursor
    runDate = Utilities.parseISODate(args[0])
    startDate = Utilities.parseISODate(options.startDate)

    results=getModelSeries(cursor)
    logging.debug("Got model series information")
    # set up dictionaries
    regionDict={}
    seriesDict={}
    modelDict={}
    for res in results:
        modelname,modeldesc, region, rmgid, ctry, rmsid, fromdt, thrudt=res
        if options.modelNames and modelname not in options.modelNames.split(','):
            continue

        modelDict[modelname]=modeldesc
        # build up the list of risk model series that make up a model 
        if modelname in seriesDict:
            if rmsid not in [s[0] for s in seriesDict[modelname]]:
                seriesDict[modelname].append([rmsid, fromdt, thrudt])
        else:
            seriesDict[modelname]=[[rmsid, fromdt, thrudt]]

        #keep track of the various countries that go in and out of the models.
        # usually it is static but once in a while there are countries that come in
        # see the EM model in early 2009 for instance
        if modelname in regionDict:
            regionDict[modelname].append([rmgid,fromdt,thrudt])
        else:
            regionDict[modelname]=[[rmgid,fromdt,thrudt]]

    now=datetime.datetime.now()
    for models in seriesDict.keys():
        startDate = Utilities.parseISODate(options.startDate)

        tooold = False
        results = getMaxTradingDate(cursor, [str(s[0]) for s in seriesDict[models]])
        for r in results:
            if r[1].date() > runDate:
                logging.warning('Run date %s is older than most recent model on %s--skipping',
                             runDate, r[1].date())
                tooold=True
        if tooold:
            continue

        name=modelDict[models]
        if name[-2:]=='MH':
            name=name[:-2]

        mnemonic=models
        if mnemonic[-3:]=='-MH':
            mnemonic=mnemonic[:-3]

        tmpfile=tempfile.mkstemp(suffix=str(runDate).replace('-',''), prefix='calatt', dir=options.targetDir.rstrip('/'))
        os.close(tmpfile[0])
        tmpfilename=tmpfile[1]
        #fileName='%s/AXCAL-%s.att' % (options.targetDir.rstrip('/'),name)
        fileName='%s/%s-calendar.att' % (options.targetDir.rstrip('/'),mnemonic)
        f=open(tmpfilename,'w')
        # write out headers

        f.write('#DataDate: %s\n' % runDate)
            # write createDate in UTC
        gmtime = time.gmtime(time.mktime(now.timetuple()))
        utctime = datetime.datetime(year=gmtime.tm_year,
                                        month=gmtime.tm_mon,
                                        day=gmtime.tm_mday,
                                        hour=gmtime.tm_hour,
                                        minute=gmtime.tm_min,
                                        second=gmtime.tm_sec)
        f.write('#CreationTimestamp: %sZ\n' % utctime.strftime('%Y-%m-%d %H:%M:%S'))
        #f.write("#FlatFileVersion: 3.2\n")
        #f.write('#ModelFamilyName: %s\n' % name)        
        f.write('#Columns: Date|End of Month|Expected\n')
        f.write('#Type: Attribute|Attribute|Attribute\n')
        f.write('#Unit: Date|Text|Text\n')
        logging.info("Working on %s", name)
        results=getPastTradingDates(cursor, [str(s[0]) for s in seriesDict[models]], startDate, runDate)
        logging.info('Got past RMI dates for %s', name)
        finalResults=results
        if len(results) > 0:
            maxdt=results[-1][0]
            # look to see if the rmglist is the same for the first day and for the Nth day forward
            # use that to call the program appropriately
            startDate=maxdt+datetime.timedelta(days=1)
            dt=startDate
            endDate=maxdt+datetime.timedelta(days=options.numDaysForward)
            rmglist=[str(rmgid) for rmgid,fromdt, thrudt in  regionDict[models] if fromdt <= dt and thrudt > dt]
            while dt <= endDate:               
                dt=dt+datetime.timedelta(days=1)
                rmglist1=[str(rmgid) for rmgid,fromdt, thrudt in  regionDict[models] if fromdt <= dt and thrudt > dt]
                if rmglist1 != rmglist:
                    logging.info('Get Future dates for %s (%s - %s)', name, str(startDate)[:10], str(dt)[:10])
                    results=getFutureTradingDates(cursor,rmglist, startDate,dt)
                    finalResults += results
                    rmglist=rmglist1
                    startDate=dt
            # at the end of the loop, find the list for the date range that is "left over"
            if startDate != endDate:
                logging.info('Get Future (final) dates for %s (%s - %s)', name, str(startDate)[:10], str(endDate)[:10])
                results=getFutureTradingDates(cursor,rmglist1, startDate,endDate)
                finalResults += results
        writeDates(f, finalResults)

        f.close()
        shutil.move(tmpfilename,fileName)
        logging.info("Moved file %s to %s", tmpfilename, fileName)
        if options.targetDir2:
            if not os.path.exists(options.targetDir2):
                os.makedirs(options.targetDir2)
            basename=os.path.basename(fileName)
            destFileName=options.targetDir2.rstrip('/') + '/' + basename
            logging.info('Copying file %s to %s', fileName, destFileName)
            shutil.copyfile(fileName, destFileName)
            os.chmod(destFileName,0o644)

        os.chmod(fileName,0o644)
    logging.info('End')
