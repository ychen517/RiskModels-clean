
import configparser
import datetime
import optparse
import os

def stringToDate(dateStr):
    """Convert YYYY-MM-DD string to datetime.date object.
    """
    return datetime.date(int(dateStr[0:4]), int(dateStr[5:7]), 
                        int(dateStr[8:10]))

if __name__ == '__main__':
    usage = 'usage: runMkt.py <workDir> --step=[step] --db=[db] --rmgs=[rmgs] --start-date=<start-date> --end-date=[end-date]'
    cmdlineParser = optparse.OptionParser(usage=usage)
    cmdlineParser.add_option("-n", action="store_true",
                             default=False, dest="testOnly",
                             help="don't change the database")
    cmdlineParser.add_option("-v", action="store_true",
                             default=False, dest="verbose",
                             help="verbose mode")
    cmdlineParser.add_option("--justCMD", action="store_true",
                             default=False, dest="justCMD",
                             help="don't change the database")
    cmdlineParser.add_option('--step', action='store',
                            default='mret', dest='step',
                            help='type of step to be run')
    cmdlineParser.add_option('--end-date', action='store',
                            default=None, dest='endDate',
                            help='data end date')
    cmdlineParser.add_option('--start-date', action='store',
                            default=None, dest='startDate',
                            help='data start date')
    cmdlineParser.add_option('--cf', action='store',
                            default='test.config', dest='configFile',
                            help='config file to be used')
    cmdlineParser.add_option('--rmgs', action='store',
                            default='all', dest='rmgList',
                            help='List of markets')
    cmdlineParser.add_option('--numeraire', action='store',
                            default='USD', dest='numeraire',
                            help='data numeraire')
    cmdlineParser.add_option('-f', action='store_true',
                            default=False, dest="override",
                            help='override certain errors')
    cmdlineParser.add_option('-c', action='store_true',
                            default=False, dest="cleanup",
                            help='delete existing data first')
    cmdlineParser.add_option("--not-in-models", action="store_true",
                             default=False, dest="notInRiskModels",
                             help="flag to allow transfer of sub-issues not mapped to a model")
    cmdlineParser.add_option('-l', '--log', action='store',
                            default=None, dest='log_dir',
                            help='log dir when running runMkt.py ')
    (options, args) = cmdlineParser.parse_args()
    if len(args) > 0:
        dirName = args[0]
    else:
        dirName = os.getcwd()
    if len(args) > 1:
        logDir = args[1]
    else:
        logDir = '%s/logs' % dirName
    # logDir = '%s/logs' % dirName #move above since we may take it from argument
    if options.log_dir is not None:
        logDir = str(options.log_dir)
    if not os.path.exists(logDir):
        os.mkdir(logDir)

    master = os.getenv('MASTER', 'hermetix')

    # Sort out dates
    if options.startDate == None and options.endDate == None:
        cmdlineParser.error('Must include at least one date')
    if options.startDate == None:
        options.startDate = options.endDate
    if options.endDate == None:
        options.endDate = options.startDate

    if len(options.startDate) == 4:
        startDate = stringToDate('%s-01-01' % options.startDate)
    else:
        startDate = stringToDate(options.startDate)
    if len(options.endDate) == 4:
        endDate = stringToDate('%s-12-31' % options.endDate)
    else:
        endDate = stringToDate(options.endDate)

    if startDate > endDate:
        tmp = startDate
        startDate = endDate
        endDate = tmp
    years = list(range(startDate.year, endDate.year+1))

    # Get the type of transfer to be run
    rmgArg='rmgs'
    if options.step == 'mport':
        stepName = 'MarketPortfolio'
    elif (options.step == 'mret') or (options.step == 'mktret'):
        stepName = 'MarketReturn'
    elif (options.step == 'rmret') or (options.step == 'rmktret'):
        stepName = 'MarketReturnV3'
    elif options.step == 'rtim3':
        stepName = 'ReturnsTimingV3'
    elif (options.step == 'mktvol') or (options.step == 'mktcsv'):
        stepName = 'MarketVolatility'
    elif options.step == 'regret':
        stepName = 'RegionReturn'
        rmgArg='regs'
    elif options.step == 'rtim':
        stepName = 'ReturnsTiming'
    elif options.step == 'fnddesc':
        stepName = 'NumeraireDescriptorData'
    elif options.step == 'mktdesc':
        stepName = 'LocalDescriptorData'
    elif options.step == 'dmdesc':
        stepName = 'LocalDescriptorData_Test'
    elif options.step == 'qrtdesc':
        stepName = 'NumeraireQuarterlyDescriptorData'
    elif options.step.lower() == 'usdesc':
        stepName = 'USDescriptorData'
    elif options.step.lower() == 'jpdesc':
        stepName = 'JPDescriptorData'
    elif options.step.lower() == 'audesc':
        stepName = 'AUDescriptorData'
    elif options.step.lower() == 'cndesc':
        stepName = 'CNDescriptorData'
    elif options.step.lower() == 'cadesc':
        stepName = 'CADescriptorData'
    elif options.step.lower() == 'nadesc':
        stepName = 'NADescriptorData'
    elif options.step.lower() == 'tsdesc':
        stepName = 'TSLibraryDescriptorData'
    elif options.step == 'TSdesc3Y':
        stepName = 'TSLibraryDescriptorData3Y'
    elif options.step == 'betav3':
        stepName = 'HistoricBetaV3'
    elif options.step == 'beta':
        stepName = 'HistoricBeta'
    else:
        stepName = options.step
    flag = ''
    if options.testOnly:
        flag += '-n '
    if options.override:
        flag += '-f '
    if options.cleanup:
        flag += '-c '
    if options.verbose:
        flag += '-v '
    if options.notInRiskModels:
        flag += '--not-in-models '
    if options.numeraire == 'None':
        numFlag = ''
    else:
        numFlag = 'numeraire=%s' % options.numeraire

    # Kick off the steps
    prevGroup = ''
    for year in years:
        # More date manipulation
        if startDate.year == year:
            date1 = str(startDate)
        else:
            date1 = '%s-01-01' % str(year)
        if endDate.year == year:
            date2 = str(endDate)
        else:
            date2 = '%s-12-31' % str(year)

        groupName = '%s-%s-%s' % (stepName, options.configFile, year)
        if stepName == 'HistoricBeta':
            depends = prevGroup
        else:
            depends = ''

        if 'gics' in options.step.lower():
            rmgArg = 'issue-ids'
            cmd = 'python3 $SCHED/Client.py --master=%s --work-dir=%s \
                    --cmd="python3 -W ignore transfer.py %s %s sections=%s %s=%s \
                    dates=%s:%s" --group=%s --depends-on=%s --log-dir=%s run' \
                    % (master, dirName, flag, options.configFile, stepName,
                        rmgArg, options.rmgList, date1, date2, groupName, depends, logDir)
        else:

            cmd = 'python3 $SCHED/Client.py --master=%s --work-dir=%s \
                    --cmd="python3 -W ignore transfer.py %s %s sections=%s %s=%s \
                    dates=%s:%s %s" --group=%s --depends-on=%s --log-dir=%s run' \
                    % (master, dirName, flag, options.configFile, stepName,
                        rmgArg, options.rmgList, date1, date2, numFlag, groupName, depends, logDir)
        print(cmd)
        if not options.justCMD:
            os.system(cmd)
        prevGroup = groupName
