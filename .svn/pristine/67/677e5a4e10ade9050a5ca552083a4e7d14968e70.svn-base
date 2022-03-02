
import datetime
import logging
import optparse
import re
import sys
import zipfile
from marketdb import MarketDB
from riskmodels import ModelDB
from riskmodels import Utilities

def getIssueSubIssueMap(modelDB):
    modelDB.dbCursor.execute("""SELECT issue_id, sub_id, from_dt, thru_dt
      FROM sub_issue""")
    issueSubIssueMap = dict()
    for (issue, sid, fromDt, thruDt) in modelDB.dbCursor.fetchall():
        issueSubIssueMap.setdefault(issue, list()).append(
            (sid, fromDt.date(), thruDt.date()))
    return issueSubIssueMap

def getSubIssue(modelID, date, issueSubIssueMap):
    issueHist = issueSubIssueMap.get(modelID, list())
    issueHist = [sid for (sid, fromDt, thruDt) in issueHist
                 if fromDt <= date and thruDt > date]
    assert(len(issueHist) <= 1)
    if len(issueHist) == 1:
        return issueHist[0]
    raise KeyError('No sub-issue for %s on %s' % (modelID, date))

def main():
    usage = "usage: %prog [options] <sub-factor ID> <zip file of exposures>"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addModelAndDefaultCommandLine(cmdlineParser)
    cmdlineParser.add_option("-n", action="store_true",
                             default=False, dest="testOnly",
                             help="don't change the database")
    (options, args) = cmdlineParser.parse_args()
    if len(args) != 2:
        cmdlineParser.error("Incorrect number of arguments")
    riskModelClass = Utilities.processModelAndDefaultCommandLine(
        options, cmdlineParser)

    modelDB = ModelDB.ModelDB(sid=options.modelDBSID, user=options.modelDBUser,
                              passwd=options.modelDBPasswd)
    marketDB = MarketDB.MarketDB(
        sid=options.marketDBSID, user=options.marketDBUser,
        passwd=options.marketDBPasswd)
    riskModel = riskModelClass(modelDB, marketDB)

    issueSubIssueMap = getIssueSubIssueMap(modelDB)
    subFactorID = int(args[0])
    logging.info('Loading into rms ID %d, sub-factor ID %d',
                 riskModel.rms_id, subFactorID)
    zipName = args[1]
    if not zipfile.is_zipfile(zipName):
        logging.fatal("%s doesn't appear to be a ZIP file", zipName)
        return 1
    zipFile = zipfile.ZipFile(zipName, 'r')
    filePattern = re.compile(r'\A.*\D(\d\d\d\d\d\d\d\d)\..*\Z')
    for fName in sorted(zipFile.namelist()):
        match = filePattern.match(fName)
        if not match:
            logging.warning("File %s doesn't match expected file pattern."
                         " Ignoring it", fName)
        dateStr = match.group(1)
        date = datetime.date(int(dateStr[0:4]), int(dateStr[4:6]),
                             int(dateStr[6:8]))
        logging.info('Processing %s', date)
        firstLine = True
        valueDicts = list()
        for line in zipFile.read(fName).splitlines():
            (modelID, value) = line.split(',')
            if firstLine:
                firstLine = False
                assert(modelID == 'AxiomaID')
            else:
                value = float(value)
                if len(modelID) == 9:
                    modelID = 'D' + modelID
                try:
                    subID = getSubIssue(modelID, date, issueSubIssueMap)
                    record = {'sid': subID, 'rms_id': riskModel.rms_id,
                              'sf_id': subFactorID, 'val': value, 'dt': date}
                    valueDicts.append(record)
                except KeyError:
                    logging.error('No sub-issue for %s on %s', modelID, date)
        modelDB.dbCursor.execute("""SELECT count(*) FROM rmi_factor_exposure
           WHERE rms_id=:rms_id AND dt=:dt AND sub_factor_id=:subFactorID""",
                                 rms_id=riskModel.rms_id,
                                 dt=date, subFactorID=subFactorID)
        (count,) = modelDB.dbCursor.fetchone()
        if count > 0:
            logging.error('%d exposure records already present.'
                          ' Skipping insert on %s', count, date)
        else:
            logging.info('Inserting %d exposures', len(valueDicts))
            modelDB.dbCursor.executemany("""INSERT INTO rmi_factor_exposure
               (rms_id, dt, sub_factor_id, sub_issue_id, value)
               VALUES(:rms_id, :dt, :sf_id, :sid, :val)""",
                                         valueDicts)
    zipFile.close()
    if options.testOnly:
        logging.info('Reverting changes')
        modelDB.revertChanges()
    else:
        modelDB.commitChanges()
    marketDB.finalize()
    modelDB.finalize()
    return 0

if __name__ == '__main__':
    sys.exit(main())
