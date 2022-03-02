
import logging
import optparse
import datetime
import glob
import csv
import xlwt
from riskmodels import ModelDB
from marketdb import MarketDB
from riskmodels import Utilities
from riskmodels import ModelID


QUERY="""
   select 
      a.axioma_id,
        t.id ticker,
        n.id,
        m.name
   from asset_ref a
   left join asset_dim_ticker_active_int t
        on t.axioma_id=a.axioma_id and t.from_dt <= :dt and :dt < t.thru_dt
   left join asset_dim_name_active_int n
        on n.axioma_id=a.axioma_id and n.from_dt <= :dt and :dt < n.thru_dt
   left join asset_dim_market_active_int m
        on m.axioma_id=a.axioma_id and m.from_dt <= :dt and :dt < m.thru_dt
   where a.axioma_id in (%s)
   order by t.id

"""

INDEX_DICT={
'High-Beta':['H30127','CSI-Axioma 300 Optimized High Predicted Beta Index'],
'High-Growth':['H30125','CSI-Axioma 300 Optimized High Growth Index'],
'High-Value':['H30126','CSI-Axioma 300 Optimized High Value Index'],
'High-Volatility':['H30129','CSI-Axioma 300 Optimized High Volatility Index'],
'Low-Beta':['H30128','CSI-Axioma 300 Optimized Low Predicted Beta Index'],
'Low-Volatility':['H30130','CSI-Axioma 300 Optimized Low Volatility Index'],
}

def processFile(filename, idx, dt, marketDB, modelDB, outputDir):
    midDict={}
    isodt=datetime.date(int(dt[0:4]), int(dt[4:6]), int(dt[6:8]))
    for line in open(filename).readlines():
        line=line.strip()
        if len(line)==0 or line[0]=='#':
            continue
        mid,wt=line.split(',')
        midDict[mid]=float(wt)

    mids=[ModelID.ModelID(string='D%s' % mid) for mid in midDict.keys()] 
    axidStr,modelMarketMap= modelDB.getMarketDB_IDs(marketDB, mids)
    # build up a dictionary of market to model and also check to see if all modelids are accounted for
    axidDict={}
    for mid, values in modelMarketMap.items():
        for fromdt,thrudt,axid in values:
            if fromdt <= isodt and isodt < thrudt:
                axidDict[axid.getIDString()]=mid

    fileModelIDs=[mid for mid in midDict.keys()]
    dbModelIDs=[mid.getPublicID() for mid in axidDict.values()]
    diff1=set(fileModelIDs) - set (dbModelIDs)
    if len(diff1) > 0:
        logging.error('%s %s missing assets from Model DB %s', idx, dt, diff1)
    else:
        logging.info('%s %s %d assets', idx, dt, len(fileModelIDs))
    axids=','.join(["'%s'" % axid for axid in axidDict.keys()] )
    
    marketDB.dbCursor.execute(QUERY % axids, dt=isodt)
    results=marketDB.dbCursor.fetchall()
    dbModelIDs=[axidDict[res[0]].getPublicID() for res in results]
    diff1=set(fileModelIDs) - set (dbModelIDs)
    outFileName=outputDir.rstrip('/')+'/'+ '%scloseweightfree%s.csv' % (INDEX_DICT[idx][0],dt)
    excelFileName=outputDir.rstrip('/')+'/'+ '%scloseweightfree%s.xls' % (INDEX_DICT[idx][0],dt)
    outFile=open(outFileName,'w')
    writer = csv.writer(outFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
    headerrow=['Date','Index Code','Index Name (Eng.)','Constituent Code','Constituent Name (Eng.)','Exchange','Weight(%)','Trading Currency'] 
    writer.writerow(headerrow)

    # write also an excel version
    wb=xlwt.Workbook()
    style=xlwt.XFStyle()
    font=xlwt.Font()
    font.bold=True
    style.font=font
    ws=wb.add_sheet('Index Constituents Data')
    rowidx=0
    for colidx, val in enumerate(headerrow):
        ws.write(rowidx, colidx, val, style)
    rowidx += 1
    if len(diff1) > 0:
        logging.error('%s %s missing assets in marketDB %s', idx, dt, diff1)
    else:
        logging.info('All %d assets got market data', len(fileModelIDs))

    for res in results:
        axid,ticker,name,exch=res
        datarow=[str(isodt), INDEX_DICT[idx][0], INDEX_DICT[idx][1], ticker, name, exch, midDict[axidDict[axid].getPublicID()] * 100, 'CNY']
        writer.writerow(datarow)

        # now write into the excel version
        for colidx, val in enumerate(datarow):
           ws.write(rowidx, colidx, val)
        rowidx += 1
    outFile.close()
    wb.save(excelFileName)
    logging.info("Processed %s %s into %s %s", dt, idx, outFileName, excelFileName)
 
if __name__ == '__main__':
    usage = "usage: %prog [options] pattern"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addModelAndDefaultCommandLine(cmdlineParser)
    cmdlineParser.add_option("-d", "--dir", action="store",
                             default="/home/vsmani/CSI", dest="dir",
                             help="name of directory")
    cmdlineParser.add_option("-o", "--output-dir", action="store",
                             default="/home/vsmani/CSI", dest="outputdir",
                             help="name of directory")
    (options, args) = cmdlineParser.parse_args()
    if len(args) < 1:
        cmdlineParser.error("Incorrect number of arguments")
    Utilities.processDefaultCommandLine(options, cmdlineParser)

    pattern=args[0]
    if pattern=='all':
        pattern='acct*.csv'
    else:
        pattern='acct*%s*.csv' % pattern

    modelDB = ModelDB.ModelDB(sid=options.modelDBSID,
                              user=options.modelDBUser,
                              passwd=options.modelDBPasswd)
    marketDB = MarketDB.MarketDB(sid=options.marketDBSID,
                                 user=options.marketDBUser,
                                 passwd=options.marketDBPasswd)

    fileNames=sorted(glob.glob(options.dir.rstrip('/') + '/' + pattern))
    logging.info('Start')
    for filename in fileNames:
        logging.info('Working on %s', filename)
        filelist=filename.split('-')
        junk=filelist[0]
        dt=filelist[-1].replace('.csv','')
        idx='-'.join(filelist[1:-1])
 
        processFile(filename, idx, dt, marketDB, modelDB, options.outputdir)
       

