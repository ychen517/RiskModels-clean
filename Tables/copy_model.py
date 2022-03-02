
import configparser
import csv
import logging
import optparse

from riskmodels import ModelDB
from riskmodels import Utilities
from riskmodels import Connections
import updateTable

TABLES=[
    # tablename, rms_id position, start_date position
    ('risk_model_serie',0,3),
    ('rms_factor',0,2),
    ('rmg_model_map',0,2),
    ('rms_factor_descriptor',0,-1)
    ]

if __name__=='__main__':
    usage = "usage: %prog [options] config-file "
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addModelAndDefaultCommandLine(cmdlineParser)
    cmdlineParser.add_option("-n", action="store_true",
                             default=False, dest="testOnly",
                             help="don't change the files or database")
    cmdlineParser.add_option("--start-date", default=None,
                             dest="startDate",
                             help="New start date for new series, default is no change")
    (options_, args_) = cmdlineParser.parse_args()
    if len(args_) < 1:
        cmdlineParser.error("Incorrect number of arguments")
    modelClass = Utilities.processModelAndDefaultCommandLine(options_, cmdlineParser)
    if options_.startDate:
        try:
            Utilities.parseISODate(options_.startDate)
        except:
            cmdlineParser.error("Invalid date %s" % options_.startDate)
    
    configFile_ = open(args_[0])
    config_ = configparser.ConfigParser()
    config_.read_file(configFile_)
    configFile_.close()
    
    connections_ = Connections.createConnections(config_)
    #modelInstance = modelClass(connections_.modelDB, connections_.marketDB)

    modelDB = connections_.modelDB
    modelDB.dbCursor.execute("""SELECT rm_id,revision,name,description,from_dt,thru_dt,distribute FROM
    risk_model rm JOIN risk_model_serie rms on rms.rm_id=rm.model_id
    WHERE revision=(SELECT MAX(revision) FROM risk_model_serie rms1 WHERE rms1.rm_id=rms.rm_id)
    AND rm.model_id=:rm_id""", rm_id=modelClass.rm_id)
    curModel = modelDB.dbCursor.fetchall()[0]

    logging.info("Creating a new version of model %s", curModel[2])

    for t in TABLES:
        logging.info("Creating %s record(s)", t[0])
        processor = updateTable.tableDataClass[t[0]](t[0], modelDB)
        infile = open(t[0]+'.txt','r+')
        rdr = csv.reader(infile, delimiter='|',lineterminator='\n')
        wrt = csv.writer(infile, delimiter='|',lineterminator='\n')
        oldrows = []
        newrows = []
        if t[0] == 'risk_model_serie':
            maxcurrms_id = max(int(i[t[1]]) for i in rdr if i and not i[0].startswith('#'))
            new_rms_id = maxcurrms_id + 1
            logging.info("New RMS_ID is %d", new_rms_id)
        infile.seek(0)
        for row in rdr:
            oldrows.append(row)
            if len(row)<2 or row[0].startswith('#'):
                continue
            if int(row[t[1]]) == modelClass.rms_id:
                newrow = list(row)
                newrow[t[1]] = str(new_rms_id)
                if t[0] == 'risk_model_serie':
                    oldrev = int(curModel[1])
                    newrev = int(oldrev) + 1
                    newrow[2] = str(newrev)
                    newrow[-1] = 0 # distribute=0
                    curStartDate = row[t[2]]
                if options_.startDate and t[2]>=0 and curStartDate == row[t[2]]:
                    newrow[t[2]] = options_.startDate
                newrows.append(newrow)
        for row in newrows:
            #logging.info('Adding row: %s', '|'.join(row))
            if not options_.testOnly:
                wrt.writerow(row)
            processor.processLine(row, True)
        infile.close()

    if options_.testOnly:
        logging.info("Reverting changes")
        modelDB.revertChanges()
    else:
        modelDB.commitChanges()

    print('Beginning of new class in RiskModels.py:')
    print('class %s(%s):' % \
          (modelClass.__name__, ', '.join('%s.%s' % (b.__module__, b.__name__) for b in modelClass.__bases__)))
    print('    """%s"""' % modelClass.__doc__)
    print('    rm_id = %d' % modelClass.rm_id)
    print('    revision = %d' % newrev)
    print('    rms_id = %d' % new_rms_id)
    Connections.finalizeConnections(connections_)
