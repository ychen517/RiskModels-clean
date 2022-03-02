
import configparser
import logging
import optparse
from riskmodels import Connections
from riskmodels import Utilities

def executeQuery(query, mdlDB, testOnly):
    if testOnly:
        logging.info('Would execute: %s', query)
    else:
        logging.info('Executing: %s', query)
        mdlDB.dbCursor.execute(query)

def createRMSTag(rmsID):
    if rmsID >= 0:
        rmsTag = 'R%.2d' % rmsID
    else:
        rmsTag = 'RM%.2d' % abs(rmsID)
    return rmsTag

def deleteModelTables(mdl, mdlDB, testOnly):
    exposureTable = mdlDB.getWideExposureTableName(mdl.rms_id)
    # get existing tables
    mdlDB.dbCursor.execute("""SELECT table_name FROM user_tables""")
    existingTables = set([i[0].lower() for i in mdlDB.dbCursor.fetchall()])
    if exposureTable in existingTables:
        query = """DROP TABLE modeldb_global.%(table)s;""" % {
            'table': exposureTable }
        print(query)

def deleteTableSpaces(rmsID, mdlDB, testOnly):
    rmsTag = createRMSTag(rmsID)
    requiredTableSpaces = [
        'GMDL_RMS_MAIN_%(rmsTag)s', 'GMDL_RMS_EXPOSURES_%(rmsTag)s',
        'GMDL_RMS_RETURNS_%(rmsTag)s']
    requiredTableSpaces = set([i % {'rmsTag': rmsTag} for i
                               in requiredTableSpaces])
    # get existing tablespaces
    mdlDB.dbCursor.execute("""SELECT tablespace_name FROM user_tablespaces""")
    existingTableSpaces = set([i[0] for i in mdlDB.dbCursor.fetchall()])
    existingTableSpaces = requiredTableSpaces & existingTableSpaces
    if len(existingTableSpaces) > 0:
        for tableSpace in existingTableSpaces:
            query = """DROP TABLESPACE %(tableSpace)s;""" % {
                'tableSpace': tableSpace }
            print(query)

def toInt(str):
    if str == 'DEFAULT':
        return None
    return int(str)

def getPartitionList(mdlDB, table):
    cur = mdlDB.dbCursor
    cur.execute("""SELECT max(high_value_length)
       FROM user_tab_partitions WHERE table_name=:table_arg""", table_arg=table)
    maxLength = cur.fetchone()[0]
    cur.setoutputsize(maxLength, 2)
    cur.execute("""SELECT partition_name, high_value, partition_position
       FROM user_tab_partitions WHERE table_name=:table_arg
       ORDER BY partition_position ASC""", table_arg=table)
    partitions = cur.fetchall()
    partitions = [(name, toInt(highVal), position)
                  for (name, highVal, position)
                  in partitions]
    return partitions

def getPartitions(partitions, rmsID):
    partName = None
    defaultPartName = None
    for (name, val, position) in partitions:
        if val is None:
            defaultPartName = name
        if val == rmsID:
            partName = name
    return (partName, defaultPartName)

def deletePartitions(rmsID, mdlDB, testOnly):
    cur = mdlDB.dbCursor
    tables = ['RMI_COVARIANCE', 'RMI_ESTU', 'RMI_FACTOR_EXPOSURE',
              'RMI_PREDICTED_BETA', 'RMI_SPECIFIC_RISK', 'RMI_TOTAL_RISK',
              'RMI_UNIVERSE', 'RMS_SPECIFIC_RETURN', 'RMI_SPECIFIC_COVARIANCE']
    for table in tables:
        partitions = getPartitionList(mdlDB, table)
        (partName, defaultPartName) = getPartitions(partitions, rmsID)
        if partName is None:
            logging.info('No partitions for RMS ID %d in %s, skipping.',
                         rmsID, table)
            continue
        if defaultPartName is None:
            logging.error('No default partition for %s, skipping.', table)
            continue
        # Check that partition is empty
        cur.execute("""SELECT count(*) FROM %(table)s
           PARTITION(%(part)s)""" % { 'table': table, 'part': partName })
        count = cur.fetchone()[0]
        if count > 0:
            logging.error('Partition for RMS ID %d in %s is not empty (%d)'
                          ', skipping.', rmsID, table, count)
            continue
        query = """ALTER TABLE %(table)s
            MERGE PARTITIONS %(part)s, %(default)s INTO PARTITION %(default)s""" % {
            'table': table, 'part': partName, 
            'default': defaultPartName }
        executeQuery(query, mdlDB, testOnly)
    return

def main():
    usage = "usage: %prog [options] config-file"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addModelAndDefaultCommandLine(cmdlineParser)
    cmdlineParser.add_option("--update-database", action="store_false",
                             default=True, dest="testOnly",
                             help="update the database")
    cmdlineParser.add_option("--force", action="store_true",
                             default=False, dest="force",
                             help="force purge of model flagged for distribution")
    (options, args) = cmdlineParser.parse_args()
    if len(args) != 1:
        cmdlineParser.error("Incorrect number of arguments")
    riskModelClass = Utilities.processModelAndDefaultCommandLine(
        options, cmdlineParser)
    
    configFile = open(args[0])
    config = configparser.ConfigParser()
    config.read_file(configFile)
    configFile.close()
    mdlDB = Connections.createConnections(config).modelDB
    
    rmsID = riskModelClass.rms_id
    logging.info('Using RMS ID: %d', rmsID)
    rmInfo = mdlDB.getRiskModelInfo(riskModelClass.rm_id,
                                      riskModelClass.revision)
    if rmInfo.distribute and not options.force:
        logging.warning('Risk model is enabled for distribution, skipping')
        return
    deletePartitions(rmsID, mdlDB, options.testOnly)
    print("Execute these queries as admin:")
    deleteModelTables(riskModelClass, mdlDB, options.testOnly)
    deleteTableSpaces(rmsID, mdlDB, options.testOnly)
    mdlDB.finalize()

if __name__ == '__main__':
    main()
