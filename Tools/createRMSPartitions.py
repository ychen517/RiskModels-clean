
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
        try:
            mdlDB.dbCursor.execute(query)
        except:
            logging.warning('...Failed')

def createRMSTag(rmsID):
    if rmsID >= 0:
        rmsTag = 'R%.2d' % rmsID
    else:
        rmsTag = 'RM%.2d' % abs(rmsID)
    return rmsTag

def checkTableSpaces(rmsID, mdlDB, testOnly, tbSize=31):
    rmsTag = createRMSTag(rmsID)
    requiredTableSpaces = [
        'GMDL_RMS_MAIN_%(rmsTag)s', 'GMDL_RMS_EXPOSURES_%(rmsTag)s',
        'GMDL_RMS_RETURNS_%(rmsTag)s']
    requiredTableSpaces = set([i % {'rmsTag': rmsTag} for i
                               in requiredTableSpaces])
    # get existing tablespaces
    mdlDB.dbCursor.execute("""SELECT tablespace_name FROM user_tablespaces""")
    existingTableSpaces = set([i[0] for i in mdlDB.dbCursor.fetchall()])
    missingTableSpaces = requiredTableSpaces - existingTableSpaces
    if len(missingTableSpaces) > 0:
        sid = mdlDB.dbConnection.tnsentry
        if sid == 'glprod':
            TABLESPACE_SQL='''CREATE SMALLFILE TABLESPACE "%(tableSpace)s" DATAFILE'+ORA_DATA/GLPROD20/datafile/%(dataFile)s-01.dbf' SIZE 100M AUTOEXTEND ON NEXT 100M MAXSIZE %(tableSize)sG LOGGING EXTENT MANAGEMENT LOCAL SEGMENT SPACE MANAGEMENT AUTO;'''
        elif sid == 'glsdg':
            TABLESPACE_SQL='''CREATE SMALLFILE TABLESPACE "%(tableSpace)s" DATAFILE'+ORA_DATA/GLSDG16/DATAFILE/%(dataFile)s-01.dbf' SIZE 100M AUTOEXTEND ON NEXT 100M MAXSIZE %(tableSize)sG LOGGING EXTENT MANAGEMENT LOCAL SEGMENT SPACE MANAGEMENT AUTO;'''
        elif sid == 'researchoda':
            TABLESPACE_SQL='''CREATE SMALLFILE TABLESPACE "%(tableSpace)s" DATAFILE '/u02/app/oracle/oradata/datastore/.ACFS/snaps/research/RESEARCHODA/datafile/%(dataFile)s-01.dbf' SIZE 100M AUTOEXTEND ON NEXT 100M MAXSIZE %(tableSize)sG LOGGING EXTENT MANAGEMENT LOCAL SEGMENT SPACE MANAGEMENT AUTO;'''
        else:
            TABLESPACE_SQL='''CREATE SMALLFILE TABLESPACE "%(tableSpace)s" DATAFILE'+DATA/%(dataFile)s-01.dbf' SIZE 100M AUTOEXTEND ON NEXT 100M MAXSIZE %(tableSize)sG LOGGING EXTENT MANAGEMENT LOCAL SEGMENT SPACE MANAGEMENT AUTO;'''
        logging.fatal('missing tablespaces: %s', ','.join(missingTableSpaces))
        logging.info('Use SQL like the following to create the tablespaces:'
                     '\n%s', '\n'.join([
                    TABLESPACE_SQL % {'tableSpace': missing,
                                      'dataFile': missing.lower(),
                                      'tableSize': tbSize}
                    for missing in missingTableSpaces]))
        raise KeyError

def createPartitions(rmsID, mdlDB, testOnly):
    sid = mdlDB.dbConnection.tnsentry
    rmsTag = createRMSTag(rmsID)
    if sid == 'glprod':
        catchall_space = 'gmdl_rms_main'
        catchall_returns_space = 'gmdl_rms_returns'
        catchall_exposures_space = 'gmdl_rms_exposures'
    elif sid == 'glsdg':
        catchall_space = 'gmdl_rms_main'
        catchall_returns_space = 'gmdl_rms_returns'
        catchall_exposures_space = 'gmdl_rms_exposures'
    elif sid == 'freshres':
        catchall_space = 'gmdl_rms_main'
        catchall_returns_space = 'gmdl_rms_returns'
        catchall_exposures_space = 'gmdl_rms_exposures'
    elif sid == 'researchoda':
        if 'vital' in mdlDB.dbConnection.username:
            catchall_space = 'modeldb_vital_ts'
            catchall_returns_space = 'modeldb_vital_ts'
            catchall_exposures_space = 'modeldb_vital_ts'
        else:
            catchall_space = 'gmdl_rms_main'
            catchall_returns_space = 'gmdl_rms_returns'
            catchall_exposures_space = 'gmdl_rms_exposures'
    elif sid == 'freshres':
        catchall_space = 'gmdl_rms_main'
        catchall_returns_space = 'gmdl_rms_returns'
        catchall_exposures_space = 'gmdl_rms_exposures'
    query = """ALTER TABLE rmi_specific_covariance
     SPLIT PARTITION p_speccov_catchall
     VALUES (%(rmsID)d) INTO (
         PARTITION p_speccov_%(rmsTag)s TABLESPACE gmdl_rms_main_%(rmsTag)s,
         PARTITION p_speccov_catchall TABLESPACE %(space)s)""" % {
             'rmsTag': rmsTag, 'rmsID': rmsID, 'space': catchall_space }
    executeQuery(query, mdlDB, testOnly)
    query = """ALTER TABLE rmi_covariance
       SPLIT PARTITION p_fcov_catchall
       VALUES (%(rmsID)d) INTO (
           PARTITION p_fcov_%(rmsTag)s TABLESPACE gmdl_rms_main_%(rmsTag)s,
           PARTITION p_fcov_catchall TABLESPACE %(space)s)""" % {
             'rmsTag': rmsTag, 'rmsID': rmsID, 'space': catchall_space }
    executeQuery(query, mdlDB, testOnly)
    query = """ALTER TABLE rmi_estu
     SPLIT PARTITION p_estu_catchall
     VALUES (%(rmsID)d) INTO (
         PARTITION p_estu_%(rmsTag)s TABLESPACE gmdl_rms_main_%(rmsTag)s,
         PARTITION p_estu_catchall TABLESPACE %(space)s)""" % {
             'rmsTag': rmsTag, 'rmsID': rmsID, 'space': catchall_space }
    executeQuery(query, mdlDB, testOnly)
    query = """ALTER TABLE rmi_estu_v3
     SPLIT PARTITION p_estu_catchall
     VALUES (%(rmsID)d) INTO (
         PARTITION p_estu_%(rmsTag)s TABLESPACE gmdl_rms_main_%(rmsTag)s,
         PARTITION p_estu_catchall TABLESPACE %(space)s)""" % {
             'rmsTag': rmsTag, 'rmsID': rmsID, 'space': catchall_space }
    executeQuery(query, mdlDB, testOnly)
    query = """ALTER TABLE rmi_factor_exposure
     SPLIT PARTITION p_exposure_catchall
     VALUES (%(rmsID)d) INTO (
         PARTITION p_exposure_%(rmsTag)s TABLESPACE gmdl_rms_exposures_%(rmsTag)s,
         PARTITION p_exposure_catchall TABLESPACE %(space)s)""" % {
             'rmsTag': rmsTag, 'rmsID': rmsID, 'space': catchall_exposures_space }
    executeQuery(query, mdlDB, testOnly)
    query = """ALTER TABLE rmi_predicted_beta
     SPLIT PARTITION p_predbeta_catchall
     VALUES (%(rmsID)d) INTO (
         PARTITION p_predbeta_%(rmsTag)s TABLESPACE gmdl_rms_main_%(rmsTag)s,
         PARTITION p_predbeta_catchall TABLESPACE %(space)s)""" % {
             'rmsTag': rmsTag, 'rmsID': rmsID, 'space': catchall_space }
    executeQuery(query, mdlDB, testOnly)
    query = """ALTER TABLE rmi_predicted_beta_v3
     SPLIT PARTITION p_predbeta_catchall
     VALUES (%(rmsID)d) INTO (
         PARTITION p_predbeta_%(rmsTag)s TABLESPACE gmdl_rms_main_%(rmsTag)s,
         PARTITION p_predbeta_catchall TABLESPACE %(space)s)""" % {
             'rmsTag': rmsTag, 'rmsID': rmsID, 'space': catchall_space }
    executeQuery(query, mdlDB, testOnly)
    query = """ALTER TABLE rmi_specific_risk
     SPLIT PARTITION p_specrisk_catchall
     VALUES (%(rmsID)d) INTO (
         PARTITION p_specrisk_%(rmsTag)s TABLESPACE gmdl_rms_main_%(rmsTag)s,
         PARTITION p_specrisk_catchall TABLESPACE %(space)s)""" % {
             'rmsTag': rmsTag, 'rmsID': rmsID, 'space': catchall_space }
    executeQuery(query, mdlDB, testOnly)
    query = """ALTER TABLE rmi_total_risk
     SPLIT PARTITION p_totalrisk_catchall
     VALUES (%(rmsID)d) INTO (
         PARTITION p_totalrisk_%(rmsTag)s TABLESPACE gmdl_rms_main_%(rmsTag)s,
         PARTITION p_totalrisk_catchall TABLESPACE %(space)s)""" % {
             'rmsTag': rmsTag, 'rmsID': rmsID, 'space': catchall_space }
    executeQuery(query, mdlDB, testOnly)
    query = """ALTER TABLE rmi_universe
     SPLIT PARTITION p_univ_catchall
     VALUES (%(rmsID)d) INTO (
         PARTITION p_univ_%(rmsTag)s TABLESPACE gmdl_rms_main_%(rmsTag)s,
         PARTITION p_univ_catchall TABLESPACE %(space)s)""" % {
             'rmsTag': rmsTag, 'rmsID': rmsID, 'space': catchall_space }
    executeQuery(query, mdlDB, testOnly)
    query = """ALTER TABLE rms_specific_return
     SPLIT PARTITION p_specret_catchall
     VALUES (%(rmsID)d) INTO (
         PARTITION p_specret_%(rmsTag)s TABLESPACE gmdl_rms_returns_%(rmsTag)s,
         PARTITION p_specret_catchall TABLESPACE %(space)s)""" % {
             'rmsTag': rmsTag, 'rmsID': rmsID, 'space': catchall_returns_space }
    executeQuery(query, mdlDB, testOnly)
    query = """ALTER TABLE rms_specific_return_internal
     SPLIT PARTITION p_sprtint_catchall
     VALUES (%(rmsID)d) INTO (
         PARTITION p_sprtint_%(rmsTag)s TABLESPACE gmdl_rms_returns_%(rmsTag)s,
         PARTITION p_sprtint_catchall TABLESPACE %(space)s)""" % {
             'rmsTag': rmsTag, 'rmsID': rmsID, 'space': catchall_returns_space }
    executeQuery(query, mdlDB, testOnly)
    query = """ALTER TABLE rms_robust_weight
     SPLIT PARTITION p_robustwt_catchall
     VALUES (%(rmsID)d) INTO (
         PARTITION p_robustwt_%(rmsTag)s TABLESPACE gmdl_rms_main_%(rmsTag)s,
         PARTITION p_robustwt_catchall TABLESPACE %(space)s)""" % {
             'rmsTag': rmsTag, 'rmsID': rmsID, 'space': catchall_space }
    executeQuery(query, mdlDB, testOnly)
    query = """ALTER TABLE rms_robust_weight_internal
     SPLIT PARTITION p_robwtint_catchall
     VALUES (%(rmsID)d) INTO (
         PARTITION p_robwtint_%(rmsTag)s TABLESPACE gmdl_rms_main_%(rmsTag)s,
         PARTITION p_robwtint_catchall TABLESPACE %(space)s)""" % {
             'rmsTag': rmsTag, 'rmsID': rmsID, 'space': catchall_space }
    executeQuery(query, mdlDB, testOnly)

def main():
    usage = "usage: %prog [options] config-file"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addModelAndDefaultCommandLine(cmdlineParser)
    cmdlineParser.add_option("--update-database", action="store_false",
                             default=True, dest="testOnly",
                             help="update the database")
    cmdlineParser.add_option("--table-size", action="store",
                             default=31, dest="tSize",
                             help="set the table size")
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
    print(rmsID)
    checkTableSpaces(rmsID, mdlDB, options.testOnly, tbSize=options.tSize)
    createPartitions(rmsID, mdlDB, options.testOnly)
    mdlDB.revertChanges()
    mdlDB.finalize()

if __name__ == '__main__':
    main()
