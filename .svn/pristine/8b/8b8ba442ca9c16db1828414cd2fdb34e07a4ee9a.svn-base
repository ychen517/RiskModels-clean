
import optparse
import logging
import datetime
from riskmodels import Utilities
from riskmodels.findetfs import getETFs

if __name__ == '__main__':
    usage='python %prog configFile [-n | --dir=<dirname>  start[:end]'
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addDefaultCommandLine(cmdlineParser)
        
    cmdlineParser.add_option("-n", action="store_true",
                                   default=False, dest="testOnly",
                                   help="don't change the database")

    cmdlineParser.add_option("-t", action="store_true",
                                   default=False, dest="tqa",
                                   help="look in TQA")

    cmdlineParser.add_option("--user", action="store",
                                   default='tqa_user', dest="user",
                                   help="User")

    cmdlineParser.add_option("--passwd", action="store",
                                   default='tqa_user', dest="passwd",
                                   help="Passwd")

    cmdlineParser.add_option("--database", action="store",
                                   default='qai', dest="database",
                                   help="Database")
    cmdlineParser.add_option("--host", action="store",
                                   default='oberon.axiomainc.com', dest="host",
                                   help="Host")

    cmdlineParser.add_option("--dir", action="store",
                                   default='/axioma/operations/daily/Capco', dest="dir",
                                   help="Directory")

    cmdlineParser.add_option("--type", action="store",
                                   default='us', dest="type",
                                   help="Type - us or gl")

    cmdlineParser.add_option("--etf", action="store",
                                   default=None, dest="etf",
                                   help="List of ETFs to look for")

    
    (options_, args_) = cmdlineParser.parse_args()
    Utilities.processDefaultCommandLine(options_, cmdlineParser)

    #runDate=Utilities.parseISODate('2008-12-08')
    if len(args_) < 1:
          cmdlineParser.error("Incorrect number of arguments")    
        
    dates=args_[0].split(':')
    if len(dates)==1:
        dates.append(dates[0])    
        
    startdt=Utilities.parseISODate(dates[0])
    enddt=Utilities.parseISODate(dates[1])        
    logging.info('Start')     
 
    dt=startdt
    etfData={}
    dtList=[]
    while True:
        if dt > enddt:
            break
        
        etfData[dt]={}
        etfData[dt]= getETFs(options_.dir, dt, options_.etf)
        if len(etfData[dt]) > 0:
            dtList.append(dt)
        #if not options_.testOnly:
        #    logging.info('Committing intermediate changes.....')
                    
        dt=dt+datetime.timedelta(days=1)
# ETF_DICT[inst]=[cusip, sedol, isin, ric, name, exchange]
# get list of all etfs in the different dates
    etfDict={}
    
    for d in etfData.keys():
        for k in etfData[d].keys():
            etfDict[k]= etfData[d][k]
    print('%-10s | %12s | %10s | %12s | %-10s | %30s | %s' % ('INSTRUMENT','CUSIP','SEDOL','ISIN','EXCHANGE','NAME','DATES'))
    for e in sorted(etfDict.keys()):
        # find all date information
        cusip,sedol,isin,ric,name,exchange=etfDict[e]
        dList=[]
        for d in dtList:
            if e in etfData[d]:
                dList.append(d)
        if dList==dtList:
            dateStr='ALL DATES'
        else:
            dateStr=','.join([str(d) for d in dList])
        print('%-10s | %12s | %10s | %12s | %-10s | %30s | %s' % (e, cusip, sedol, isin, exchange, name[:30],dateStr))
        
        
    #cursor.close()
    #dbconnection.close()
