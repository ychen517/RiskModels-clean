
from riskmodels import ModelDB
from riskmodels import Utilities

from ftplib import FTP
import logging
import optparse

FACTORS=[('Value','Value'),
         ('Leverage','Leverage'),
         ('Growth','Growth'),
         ('Size','Size'),
         ('MarketingSensitivity','Market Sensitivity'),
         ('Liquidity','Liquidity'),
         ('ShortTermMomentum','Short-Term Momentum'),
         ('MediumTermMomemtum','Medium-Term Momentum'),
         ('Volatility','Volatility'),
         ('ExchangeRateSensitivity','Exchange Rate Sensitivity'),
         ]

SERVER='ftp298.pair.com'
USER='ceria_axioma'
PASSWD='ax10ma08'
DIRECTORY='uploads'
# test values
# SERVER='dist3'
# USER='rlgoodman'
# PASSWD='royboy'
# DIRECTORY='upload'

if __name__=='__main__':
    usage = "usage: %prog [options] <YYYY-MM-DD> [<YYYY-MM-DD>]"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addDefaultCommandLine(cmdlineParser)
    cmdlineParser.add_option("--no-ftp", dest="doFTP", action="store_false",
                             help="Don't send files to FTP site", default=True)
    cmdlineParser.add_option("--rms", dest="rms_id", 
                             help="Risk model series ID to use for returns (default=131 for AXUS-MH)", default=131)
    (options, args) = cmdlineParser.parse_args()
    Utilities.processDefaultCommandLine(options, cmdlineParser)
    log = logging.getLogger('webxml')
    log.debug('logger level=%s', log.getEffectiveLevel())
    if len(args) < 1 or len(args) > 2:
        cmdlineParser.error("Incorrect number of arguments")
    startDate = Utilities.parseISODate(args[0])
    if len(args) == 2:
        endDate = Utilities.parseISODate(args[1])
    else:
        endDate = startDate
    modelDB = ModelDB.ModelDB(sid=options.modelDBSID,user=options.modelDBUser,passwd=options.modelDBPasswd)
    modelDB.dbCursor.execute("""select distinct dt from rms_factor_return
    where dt >= :startdate and dt <= :enddate and rms_id = :rms_id""",
                             startdate=startDate, enddate=endDate, rms_id=options.rms_id)
    r = modelDB.dbCursor.fetchall()
    dates = [modelDB.oracleToDate(a[0]) for a in r]
    if options.doFTP:
        log.debug('Connecting to %s as %s', SERVER, USER)
        ftpconn = FTP(SERVER,USER,PASSWD)
        ftpconn.set_debuglevel((log.getEffectiveLevel() == 10) and 1 or 0)
        ftpconn.set_pasv(True)
        ftpconn.cwd(DIRECTORY)
    for d in dates:
        log.info('Processing %s', d)
        fileName = 'graph_%02d%02d%02d.xml' % (d.year % 100, d.month, d.day)
        outFile = open(fileName, 'w')
        outFile.write('<?xml version="1.0" encoding="utf-8" ?>\n<Graph>\n <Data>\n')
        outFile.write('  <Date>%s</Date>\n' % d)
        for (xmlname, dbname) in FACTORS:
            modelDB.dbCursor.execute("""SELECT rfr.value
            FROM rms_factor_return rfr, sub_factor sf, factor f
            WHERE rfr.rms_id=:rmsid AND rfr.sub_factor_id=sf.sub_id and sf.factor_id=f.factor_id
            AND rfr.dt=:dt AND f.name=:fname
            """, fname=dbname, dt=d, rmsid=options.rms_id)
            r = modelDB.dbCursor.fetchall()
            if not r:
                log.warning('No value for %s on %s', dbname, d)
                #outFile.write('  <%s></%s>\n' % (xmlname, xmlname))
            else:
                outFile.write('  <%s>%.10f</%s>\n' % (xmlname, r[0][0], xmlname))
        outFile.write(" </Data>\n</Graph>\n")
        outFile.close()
        if options.doFTP:
            # put file graph_yymmdd.xml on server in uploads directory
            outFile = open(fileName, 'r')
            ftpconn.storlines('STOR %s' % fileName, outFile)
            outFile.close()

    if options.doFTP:
        ftpconn.quit()
