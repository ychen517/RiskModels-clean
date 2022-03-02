# Extract the last trading day of the month for the given risk model group
#
import datetime
import logging.config
import optparse
from riskmodels import ModelDB
from riskmodels import Utilities

if __name__ == '__main__':
    usage = "usage: %prog [options] <rms-id> [start-date] [end-date]"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addDefaultCommandLine(cmdlineParser)
    (options, args) = cmdlineParser.parse_args()
    if len(args) < 1 or len(args) > 3:
        cmdlineParser.error("Incorrect number of arguments")
    Utilities.processDefaultCommandLine(options, cmdlineParser)

    modelDB = ModelDB.ModelDB(sid=options.modelDBSID, user=options.modelDBUser,
                              passwd=options.modelDBPasswd)

    rms_id = int(args[0])
    if len(args) > 1:
        startDate = Utilities.parseISODate(args[1])
        if len(args) > 2:
            endDate = Utilities.parseISODate(args[2])
        else:
            endDate = Utilities.parseISODate('2999-12-31')
    else:
        startDate = Utilities.parseISODate('1990-01-01')
        endDate = Utilities.parseISODate('2999-12-31')

    modelDB.dbCursor.execute("""SELECT dt FROM risk_model_instance WHERE rms_id = :rms_arg AND has_risks=1""", 
                            rms_arg=rms_id)
    dates = sorted(d[0].date() for d in list(modelDB.dbCursor.fetchall()))
    dates.append(datetime.date(2999,12,31))
    eom = [prev for (prev, next) in zip(dates[:-1], dates[1:])
           if (next.month > prev.month or next.year > prev.year)
           and (prev >= startDate and prev <= endDate)]
    for d in eom:
        print(d)
    modelDB.finalize()
