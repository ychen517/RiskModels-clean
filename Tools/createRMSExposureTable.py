
import configparser
import datetime
import logging
import optparse
from riskmodels import Connections
from riskmodels import EquityModel
from riskmodels import MFM
from riskmodels import CurrencyRisk
from riskmodels import Utilities
import createRMSPartitions

def executeQuery(query, mdlDB, testOnly):
    if testOnly:
        logging.info('Would execute: %s', query)
    else:
        logging.info('Executing: %s', query)
        mdlDB.dbCursor.execute(query)

def createExposureTable(rmsID, mdlObj, mdlDB, testOnly):
    """Create the 'wide' exposure table with one column per sub-factor
    and one column per binary factor group. We assume that currencies,
    countries, and industries are binary and everything else is not.
    We also assume that the non-binary factors don't change throughout
    the model history so that loading them for today is good enough.
    """
    mdlDB.dbCursor.execute("""SELECT thru_dt FROM risk_model_serie
       WHERE serial_id=:rms_id""", rms_id=rmsID)
    rmsThruDt = mdlDB.dbCursor.fetchone()[0].date()
    myDate = min(datetime.date.today(), rmsThruDt - datetime.timedelta(1))
    hasCountry = False
    hasCurrency = False
    hasIndustry = False
    fullFactors = list()
    mdlObj.setFactorsForDate(myDate, mdlDB)
    if not mdlObj.isStatModel() and not mdlObj.isCurrencyModel():
        if mdlObj.intercept is not None:
            fullFactors.append(mdlObj.intercept)
        fullFactors.extend(mdlObj.styles)
        fullFactors.extend([f for f in mdlObj.localStructureFactors if f not in mdlObj.styles])
        hasIndustry = len(mdlObj.industries) > 0
        if len(mdlObj.rmg) > 1:
            hasCountry = True
            hasCurrency = len(mdlObj.currencies) > 0
    if len(mdlObj.blind) > 0:
        fullFactors.extend(mdlObj.blind)
        if mdlObj.isRegionalModel():
            hasCurrency = len(mdlObj.currencies) > 0
    if len(mdlObj.macro_core) > 0: 
        fullFactors.extend(mdlObj.macro_core)
    if len(mdlObj.macro_market_traded) > 0: 
        fullFactors.extend(mdlObj.macro_market_traded)
    if len(mdlObj.macro_equity) > 0: 
        fullFactors.extend(mdlObj.macro_equity)
    if len(mdlObj.macro_sectors) > 0: 
        fullFactors.extend(mdlObj.macro_sectors)
    if len(mdlObj.macros) > 0: 
        fullFactors.extend(mdlObj.macros)
    rmi = Utilities.Struct()
    rmi.date = myDate
    subFactors = mdlDB.getRiskModelInstanceSubFactors(rmi, fullFactors)
    table = mdlDB.getWideExposureTableName(rmsID)
    #executeQuery("""DROP TABLE %s""" % (table), mdlDB, testOnly)
    fullFactorColumnNames = [mdlDB.getSubFactorColumnName(sf)
                             for sf in subFactors]
    binaryColumnNames = list()
    if hasCountry:
        binaryColumnNames.append('binary_country')
    if hasCurrency:
        binaryColumnNames.append('binary_currency')
    if hasIndustry:
        binaryColumnNames.append('binary_industry')
    logging.info('%d full factors, %d binary factors',
                 len(fullFactorColumnNames), len(binaryColumnNames))
    tableSpace = 'gmdl_rms_exposures_%(rmsTag)s' % {
        'rmsTag': createRMSPartitions.createRMSTag(rmsID) }
    query = """CREATE TABLE %(table)s (
       dt                DATE NOT NULL,
       sub_issue_id      CHAR(12) NOT NULL,
       %(factorColumns)s,
       PRIMARY KEY (dt, sub_issue_id)
       ) ORGANIZATION INDEX NOLOGGING TABLESPACE %(tableSpace)s""" % {
        'table': table,
        'factorColumns': ',\n       '.join(['%s  BINARY_FLOAT' % colName
                                   for colName in fullFactorColumnNames] + 
                                  ['%s  NUMBER(4)' % colName
                                   for colName in binaryColumnNames]),
        'tableSpace': tableSpace }
    executeQuery(query, mdlDB, testOnly)

def createFMPTable(rmsID, mdlObj, mdlDB, testOnly):
    """Create the FMP table with one column per sub-factor
    """
    if mdlObj.isStatModel() or mdlObj.isCurrencyModel():
        return

    mdlDB.dbCursor.execute("""SELECT thru_dt FROM risk_model_serie
       WHERE serial_id=:rms_id""", rms_id=rmsID)
    rmsThruDt = mdlDB.dbCursor.fetchone()[0].date()
    myDate = min(datetime.date.today(), rmsThruDt - datetime.timedelta(1))
    allFactorsEver = mdlDB.getRiskModelSerieFactors(rmsID)
    query = 'SELECT a.name,a.description,a.factor_id, b.name FROM factor a,factor_type b where a.factor_type_id = b.factor_type_id and a.factor_type_id in (0,1,2,3,4,5,6)'
    mdlDB.dbCursor.execute(query)
    r = mdlDB.dbCursor.fetchall()
    import pandas as pd
    fac_map = pd.DataFrame(r,columns=['Name','Description','Factor_ID','Type'])
    fac_map = fac_map.set_index('Name')
    fac_map = fac_map.drop_duplicates()
    sel_fac_map = fac_map[fac_map['Type'].isin(['Country','Industry', 'Local', 'Market', 'Style'])]
    sel_fac_names = sel_fac_map.index.tolist()
    fullFactors = [x for x in allFactorsEver if x.name in sel_fac_names]
    rmi = Utilities.Struct()
    rmi.date = myDate
    subFactors = mdlDB.getRiskModelInstanceSubFactors(rmi, fullFactors)
    fullFactorColumnNames = [mdlDB.getSubFactorColumnName(sf) for sf in subFactors]
    logging.info('%d factors', len(fullFactorColumnNames))
    rmsTag = createRMSPartitions.createRMSTag(rmsID)
    tableSpace = 'gmdl_rms_exposures_%s' % rmsTag
    tableName = 'RMS_%s_FMP' % rmsTag[1:]
    query = """CREATE TABLE %(table)s (
       dt                DATE NOT NULL,
       sub_issue_id      CHAR(12) NOT NULL,
       %(factorColumns)s,
       PRIMARY KEY (dt, sub_issue_id)
       ) ORGANIZATION INDEX NOLOGGING TABLESPACE %(tableSpace)s""" % {
        'table': tableName,
        'factorColumns': ','.join(['%s  BINARY_FLOAT' % colName for colName in fullFactorColumnNames]),
        'tableSpace': tableSpace }
    executeQuery(query, mdlDB, testOnly)

def createStatFactorTable(rmsID, mdlObj, mdlDB, testOnly):
    """Create the statistical factor returns table
    """
    if not mdlObj.isStatModel():
        return
    rmsTag = createRMSPartitions.createRMSTag(rmsID)
    tableSpace = 'gmdl_rms_main_%s' % rmsTag
    tableName = 'RMS_%s_STAT_FACTOR_RETURN' % rmsTag[1:]
    query = """CREATE TABLE %(table)s (
       rms_id            NUMBER(38) NOT NULL,
       sub_factor_id     NUMBER(38) NOT NULL,
       exp_dt            DATE NOT NULL,
       dt                DATE NOT NULL,
       value             NUMBER NOT NULL,
       cumulative        NUMBER NULL,
       PRIMARY KEY (rms_id,exp_dt,dt,sub_factor_id)
       ) ORGANIZATION INDEX NOLOGGING TABLESPACE %(tableSpace)s""" % {
        'table': tableName,
        'tableSpace': tableSpace }
    executeQuery(query, mdlDB, testOnly)

def main():
    usage = "usage: %prog [options] config-file"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addModelAndDefaultCommandLine(cmdlineParser)
    cmdlineParser.add_option("--update-database", action="store_false",
                             default=True, dest="testOnly",
                             help="update the database")
    cmdlineParser.add_option("--fmp-only", action="store_true",
                             default=False, dest="fmpOnly",
                             help="only generate FMP table")
    (options, args) = cmdlineParser.parse_args()
    if len(args) != 1:
        cmdlineParser.error("Incorrect number of arguments")
    riskModelClass = Utilities.processModelAndDefaultCommandLine(
        options, cmdlineParser)
    
    configFile = open(args[0])
    config = configparser.ConfigParser()
    config.read_file(configFile)
    configFile.close()
    connections = Connections.createConnections(config)
    mdlDB = connections.modelDB
    mktDB = connections.marketDB
    mdlObj = riskModelClass(mdlDB, mktDB)
    
    rmsID = riskModelClass.rms_id
    if isinstance(mdlObj, EquityModel.FundamentalModel) \
            or isinstance(mdlObj, EquityModel.StatisticalModel) \
            or isinstance(mdlObj, EquityModel.LinkedModel) \
            or isinstance(mdlObj, EquityModel.ProjectionModel):
        if not options.fmpOnly:
            createExposureTable(rmsID, mdlObj, mdlDB, options.testOnly)
            createStatFactorTable(rmsID, mdlObj, mdlDB, options.testOnly)
        if not isinstance(mdlObj, EquityModel.LinkedModel) and not isinstance(mdlObj, EquityModel.ProjectionModel):
            createFMPTable(rmsID, mdlObj, mdlDB, options.testOnly)
    else:
        if not mdlObj.newExposureFormat:
            logging.fatal('Model does not use new exposure format!')
        else:
            if not options.fmpOnly:
                createExposureTable(rmsID, mdlObj, mdlDB, options.testOnly)
            createFMPTable(rmsID, mdlObj, mdlDB, options.testOnly)
    mdlDB.revertChanges()
    mdlDB.finalize()

if __name__ == '__main__':
    main()
