

import configparser
import datetime
import logging
import optparse
import sys

from riskmodels import Connections
from riskmodels import Utilities
from riskmodels.create_mappings import createMappings


def getCountry(cur, assetid, assetsrc):
    warnings=[]
    country = None
    if assetsrc == 'NA':
        
        query = """select country, min(FromDate), max(ToDate) 
                    from IDC.dbo.NAAssets 
                    where AssetId=%(assetid)d group by country"""
        #logging.debug(query)
        cur.execute(query % {'assetid': assetid})
        r = cur.fetchall()
        if len(r) == 1 and r[0][0]:
            country, miniddt, maxiddt = r[0][0], r[0][1], r[0][2]
            miniddt, maxiddt = datetime.datetime.strptime(miniddt, "%Y-%m-%d").date(), datetime.datetime.strptime(maxiddt, "%Y-%m-%d").date()
        else:
            miniddt = datetime.date(1950, 1, 1)
            maxiddt = datetime.date(9999, 12, 31)
    else:
        query = """select c.[ISO_CountryCode], min([Pricing Date]), max([Pricing Date]) maxPriceDate
                   from IDC.dbo.NonNAPrices a, IDC.dbo.NonNAExchangeCodes b, IDC.dbo.Country c 
                   where sedol in (
                     select sedol from IDC.dbo.NonNAAssets 
                     where assetid=%(assetid)d) 
                   and substring(a.[Exchange Code], 1, 2)=b.[Country Code] 
                   and c.[Code]=b.[Country Code]
                   group by c.[ISO_CountryCode]
                   order by maxPriceDate desc"""
        logging.debug(query)
        cur.execute(query % {'assetid': assetid})
        r = cur.fetchall()
        
        if len(r) > 0 and r[0][0]: #use the latest record https://jira.axiomainc.com:8443/browse/MAC-17135
            country, mindt, maxdt = r[0][0], r[0][1], r[0][2]
            mindt, maxdt = datetime.datetime.strptime(mindt, "%Y-%m-%d").date(), datetime.datetime.strptime(maxdt, "%Y-%m-%d").date()
        else:
#             if len(r) > 1:
#                 msg = "ERROR: %d countries found for %d/%s: %s" % (len(r), assetid, assetsrc,r)
#                 warnings.append(msg)
#                 logging.error(msg)
#             elif len(r) == 0:
            msg = "ERROR: No price/country/exchange records found for %d/%s" % (assetid, assetsrc)
            warnings.append(msg)
            logging.error(msg)
            mindt = datetime.date(1950, 1, 1)
            maxdt = datetime.date(9999, 12, 31)
        
            return (country, mindt, maxdt, warnings)
        
        query = """select min(FromDate), max(ToDate) from IDC.dbo.NonNAAssets where AssetId=%(assetid)d"""
        #logging.debug(query)
        cur.execute(query % {'assetid': assetid})
        r = cur.fetchall()
        if len(r) == 1 and r[0][0]:
            miniddt, maxiddt = r[0][0], r[0][1]
            miniddt, maxiddt = datetime.datetime.strptime(miniddt, "%Y-%m-%d").date(), datetime.datetime.strptime(maxiddt, "%Y-%m-%d").date()
        else:
            miniddt = datetime.date(1950, 1, 1)
            maxiddt = datetime.date(9999, 12, 31)

    if miniddt > mindt:
        mindt = miniddt
    if maxdt < maxiddt:
        maxdt = maxiddt

    if maxdt == datetime.date(2099, 12, 31):
        maxdt = datetime.date(9999, 12, 31)
        
    return (country, mindt, maxdt, warnings)

if __name__=='__main__':
    usage = "usage: %prog [--update-database] config-file"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addDefaultCommandLine(cmdlineParser)
    cmdlineParser.add_option("--update-database", action="store_true",
                             default=False, dest="update",
                             help="change the database")
    cmdlineParser.add_option("--add-modeldb", action="store_true",
                             default=False, dest="addmodeldb",
                             help="set up the asset in modeldb")
    cmdlineParser.add_option("--use-axiomadataid", action="store_true",
                             default=False, dest="useaxiomadataid",
                             help="use axiomadataid from sql server to create modeldbid")
    cmdlineParser.add_option("--src-id", action="store", type="int",
                             default=901, dest="srcid",
                             help="integer value of src_id from meta_sources")
    (options, args) = cmdlineParser.parse_args()

    idType = None
    idList = []
    if len(args) > 2:
        cmdlineParser.error("incorrect usage")
    elif len(args) == 2:
        idTypeVals = args[1]
        if idTypeVals.find('=') > 0:
            idType, idVals = idTypeVals.split("=")
            idList = list(set([i.strip() for i in idVals.split(",")]))
        else:
            cmdlineParser.error("Second argument must be idtype=idval1,idval2,...")

    Utilities.processDefaultCommandLine(options, cmdlineParser, disable_existing_loggers=False)
    configFile_ = open(args[0])
    config_ = configparser.ConfigParser()
    config_.read_file(configFile_)
    configFile_.close()
    
    connections_ =  None
    try:
        connections_ = Connections.createConnections(config_)
        results = createMappings(connections_, idList, idType, options.srcid, options.addmodeldb, useAxiomaDataId=options.useaxiomadataid)
        logging.info("Mappings created; results: %s", results)
        if options.update:
            logging.info("Committing Changes")
            connections_.marketDB.commitChanges()
            connections_.modelDB.commitChanges()
        else:
            logging.info("Reverting Changes; use --update-database to commit")
            connections_.revertAll()
    #         marketDB.revertChanges()
    #         modelDB.revertChanges()
#         Connections.finalizeConnections(connections_)
        sys.exit(0)
    except Exception as e:
        logging.exception('An Error occurred: %s', e)
#         Connections.finalizeConnections(connections_)
        sys.exit(1)
    finally:    
        Connections.finalizeConnections(connections_)
