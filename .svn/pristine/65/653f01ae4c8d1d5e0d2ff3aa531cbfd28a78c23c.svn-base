
import datetime
import glob
import logging
import optparse
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from marketdb import MarketDB
import riskmodels
from riskmodels import ModelDB
from riskmodels import Utilities
from riskmodels import writeDerbyFiles

def makeFinalFileName(fileName, version=3.2):
    return fileName
    if version >= 4.0:
        fstr='-F%2d-' % (version*10)
        l=fileName.split('-')
        return ('-'.join(l[:-1]) + fstr + '-'.join(l[-1:]))
    else:
        return fileName

def copyJar(baseName, date, srcDir, tgtDir):
    logging.info('copy %s from %s to %s', baseName, srcDir, tgtDir)
    derbyName = '%s-%s.jar' % (baseName, date.strftime('%Y%m%d'))
    shutil.copy2(os.path.join(srcDir, derbyName), tgtDir)

def createJarFile(dbFile, baseName, templateDict, dumpDir, dbDir,
                  jarDir, date, derbyCreator, substitutions=None,
                  encrypt=True, compress=False, supplemental=False, version=3.2, client=''):
    """Create the Derby jar file specified by dbFile in dbDir
    using the files in dumpDir.
    """
    if (dbFile, 'create') in templateDict:
        BOOT_PASSWD='a1x24i9o15m13a1'
        if encrypt:
            CREATE_OPTIONS="create=true;dataEncryption=true;bootPassword=%(pwd)s;" \
                "encryptionAlgorithm=AES/CBC/NoPadding" % {
                'pwd': BOOT_PASSWD}
        else:
            CREATE_OPTIONS="create=true;bootPassword=%(pwd)s;" % {
                'pwd': BOOT_PASSWD}
        template = templateDict[(dbFile, 'create')] \
                   + templateDict[(dbFile, 'load')]
        cmd = template.replace('@@databases@@', dbDir)
        cmd = cmd.replace('@@dump@@', dumpDir)
        cmd = cmd.replace('@@CREATE_OPTIONS@@', CREATE_OPTIONS)
        if substitutions is not None:
            for (substVar, substText) in substitutions.items():
                cmd = cmd.replace('@@'+substVar+'@@', substText)
        
        javaCmd = 'java -jar %s' % derbyCreator
        sub = subprocess.Popen(javaCmd, stdin=subprocess.PIPE, shell=True)
        #print (cmd)
        logging.debug('Creating database for %s', baseName)
        sub.stdin.write(bytes(cmd, 'utf-8'))
        sub.stdin.close()
        subStatus = sub.wait()
        if subStatus != 0:
            logging.error('Error creating database for %s', baseName)
            raise RuntimeError('Error creating database for %s' % baseName)
        # remove Derby log files
        shutil.rmtree(os.path.join(dbDir, baseName, 'log'))
    else:
        os.mkdir(os.path.join(dbDir, baseName))
    # DbMaster's and DbComposite's Master binary files comes
    # in a number of flavors
    # Collect them here so that we can create a jar file for each flavor.
    if dbFile in ['DbMaster', 'DbComposite']:
        flavorPattern = re.compile(r'(%s_(.*)\.binary)_(.*)\Z' % baseName)
        flavors = [None]
        for fileName in os.listdir(dumpDir):
            match = flavorPattern.match(fileName)
            if match:
                flavors.append((fileName,) + match.groups())
    else:
        flavors = [None]
    for flavor in flavors:
        # Create the jar file. Since jar files are zip files we use the
        # zipfile module
        if flavor is None:
            derbyName = '%s-%s.jar' % (baseName, date.strftime('%Y%m%d'))
        else:
            (flavorFile, flavorBaseFile, flavorJarFile, flavor) = flavor
            derbyName = '%s-%s-%s.jar' % (baseName, flavor,
                                          date.strftime('%Y%m%d'))
        # create a temporary file and rename it
        # to avoid two jobs trying to create the same file at the same time
        tmpfile = tempfile.mkstemp(prefix=derbyName,dir=jarDir)
        os.close(tmpfile[0])
        derbyPath = tmpfile[1]
        logging.debug('Creating jar file %s', derbyName)
        if compress:
            compFlag = zipfile.ZIP_DEFLATED
        else:
            compFlag = zipfile.ZIP_STORED
        derbyJar = zipfile.ZipFile(derbyPath, mode='w', compression=compFlag)
        logging.debug('Add Derby files to jar')
        # add all files and directories under dbDir/baseName
        dbDirLen = len(splitAll(dbDir))
        for dirpath, dirname, filenames in os.walk(os.path.join(dbDir, baseName)):
            jarPath = os.path.join(*splitAll(dirpath)[dbDirLen:])
            for fName in filenames:
                fullName = os.path.join(dirpath, fName)
                jarName = os.path.join(jarPath, fName)
                derbyJar.write(fullName, arcname=jarName)
        # Add binary files. If flavor is not None, add the flavor file
        # and ignore the base file.
        if flavor is not None:
            logging.debug('Adding binary file %s as %s', flavorFile,
                          flavorJarFile)
            derbyJar.write(os.path.join(dumpDir, flavorFile),
                           arcname=flavorJarFile)
        binaryPattern = re.compile(r'%s_(.*)\.binary\Z' % baseName)
        for fileName in os.listdir(dumpDir):
            match = binaryPattern.match(fileName)
            if match and (flavor is None or fileName != flavorBaseFile):
                logging.debug('Adding binary file %s', match.group(1))
                derbyJar.write(os.path.join(dumpDir, fileName),
                               arcname=match.group(1))
        derbyJar.close()
        os.chmod(derbyPath, stat.S_IROTH+stat.S_IRGRP+stat.S_IRUSR+stat.S_IWUSR)
        derbyName=makeFinalFileName(derbyName, version)
        logging.info('Moving %s -> %s', derbyPath, os.path.join(jarDir, derbyName))
        shutil.move(derbyPath,os.path.join(jarDir, derbyName))
    return 0

def findDerbyCreator(derbyCreatorPath):
    """Find RiskModels-*.jar in the specified directory and return the
    full path to it.
    Raises an exception if not exactly one jar matches.
    """
    rmJarPattern = re.compile(r'RiskModels-.*\.jar\Z')
    matchingFiles = [fileName for fileName in os.listdir(derbyCreatorPath)
                     if rmJarPattern.match(fileName)]
    if len(matchingFiles) == 1:
        rmJarFile = os.path.join(derbyCreatorPath, matchingFiles[0])
        logging.debug('Using %s to create Derby files', rmJarFile)
        return rmJarFile
    elif len(matchingFiles) == 0:
        raise Exception('Cannot find RiskModels jar in %s' % derbyCreatorPath)
    else:
        raise Exception('More than one RiskModels jar in %s: %s'
                        % (derbyCreatorPath, matchingFiles))

def splitAll(dirPath):
    """Returns a list of all the path components"""
    allPath = []
    while len(dirPath) > 0:
        (dirPath, last) = os.path.split(dirPath)
        if len(last) == 0:
            break
        allPath.append(last)
    allPath.reverse()
    return allPath

def readTemplates():
    """Read the create/load SQL templates from their files and
    stored their strings in a dictionary for later use.
    Returns the dictionary.
    """
    templateNames = [('DbAsset', 'create', 'create_dbasset.sql.template'),
                     ('DbAsset', 'load', 'load_dbasset.sql.template'),
                     ('DbClassification', 'create',
                      'create_dbclassification.sql.template'),
                     ('DbClassification', 'load',
                      'load_dbclassification.sql.template'),
                     ('DbComposite', 'create',
                      'create_dbcomposite.sql.template'),
                     ('DbComposite', 'load', 'load_dbcomposite.sql.template'),
                     ('DbFundamental', 'create',
                      'create_dbfundamental.sql.template'),
                     ('DbFundamental', 'load', 'load_dbfundamental.sql.template'),
                     ('DbIndex', 'create', 'create_dbindex.sql.template'),
                     ('DbIndex', 'load', 'load_dbindex.sql.template'),
                     ('DbMeta', 'create', 'create_dbmeta.sql.template'),
                     ('DbMeta', 'load', 'load_dbmeta.sql.template'),
                     ('DbModel', 'create', 'create_dbmodel.sql.template'),
                     ('DbModel', 'load', 'load_dbmodel.sql.template'),
                     ('DbShortfall', 'create', 'create_dbshortfall.sql.template'),
                     ('DbShortfall', 'load', 'load_dbshortfall.sql.template'),]
    templateDict = dict()
    for (dbFile, type, fileName) in templateNames:
        inFile = open(fileName, 'r')
        fileStr = inFile.read()
        templateDict[(dbFile, type)] = fileStr
        inFile.close()
    return templateDict

def processListOption(str):
    """Process list command-line argument.
    If its None or empty, return an empty list.
    Otherwise split the string by "," and return that.
    """
    if str is None or len(str) == 0:
        return list()
    return str.split(',')

def main():
    usage = "usage: %prog [options] <startdate or datelist> <end-date>"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addDefaultCommandLine(cmdlineParser)
    cmdlineParser.add_option("--no-meta", action="store_false",
                             dest="writeMetaFlag", default=True,
                             help="Don't write meta data")
    cmdlineParser.add_option("--no-master", action="store_false",
                             dest="writeMasterFlag", default=True,
                             help="Don't write master data")
    cmdlineParser.add_option("--no-asset", action="store_false",
                             dest="writeAssetFlag", default=True,
                             help="Don't write asset data")
    cmdlineParser.add_option("--indexes", action="store",
                             dest="indexNames",
                             default=None,
                             help="Write only the specified index families")
    cmdlineParser.add_option("--portfolios", action="store",
                             dest="portNames",
                             default=None,
                             help="Write only the specified index families")
    cmdlineParser.add_option("--clients", action="store",
                             dest="clients",
                             default=None,
                             help="Write only the specified client families")
    cmdlineParser.add_option("--classifications", action="store",
                             dest="classNames",
                             default='GICS',
                             help="Name(s) of classifications to include (default is GICS)")
    cmdlineParser.add_option("--composites", action="store",
                             dest="compositeNames", default=None,
                             help="Name(s) of composite families to include")
    cmdlineParser.add_option("--no-model-jars", action="store_false",
                             dest="writeModelFlag",
                             default=True,
                             help="Don't write model information")
    cmdlineParser.add_option("--models", action="store",
                             dest="models", default='',
                             help="comma-separated list of models to process")
    cmdlineParser.add_option("--derby-creator-path", action="store",
                             dest="derbyCreatorPath",
                             #default='../../../lib',
                             default='/home/ops-rm/global3/lib',
                             help="directory that has the RiskModels jar")
    cmdlineParser.add_option("--tmp-dir", action="store",
                             dest="tempDir",
                             default='.',
                             help="directory to house temporary files and directories")
    cmdlineParser.add_option("--destination,-d", action="store",
                             dest="jarDir",
                             default=None,
                             help="directory where all extracted files are copied; if not specified, use jars")
    cmdlineParser.add_option("--no-subdirs", action="store_false",
                             dest="useSubDirs", default=True,
                             help="If present, then don't use <modelname> subdirectories of jarDir; otherwise, model jars go into <jarDir>/<modelname>.")
    cmdlineParser.add_option("-p", "--preliminary", action="store_true",
                             default=False, dest="preliminary",
                             help="Preliminary run--ignore DR assets")
    cmdlineParser.add_option("--target-sub-dirs", action="store_true",
                             default=False, dest="appendDateDirs",
                             help="Append yyyy/mm to end of jarDir directory path")
    cmdlineParser.add_option("--new-rsk-fields", action="store_true",
                             default=False, dest="newRiskFields",
                             help="Include new fields in DbModel jar")
    cmdlineParser.add_option("--new-beta-fields", action="store_true",
                             default=False, dest="newBetaFields",
                             help="Include new beta fields in DbModel jar")
    cmdlineParser.add_option("--new-mdv-fields", action="store_true",
                             default=False, dest="newMDVFields",
                             help="Include new MDV fields in DbModel jar")
    cmdlineParser.add_option("--no-cumulative-return", action="store_true",
                             default=False, dest="noCumReturn",
                             help="Do not Include cumulative return field in DbAsset jar")
    cmdlineParser.add_option("--file-format-version", action="store",
                             default=3.2, type='float', dest="fileFormatVersion",
                             help="version of derby file format to create")
    cmdlineParser.add_option("--force", "-f", action="store_true",
                             default=False, dest="force",
                             help="override certain constraints")
    cmdlineParser.add_option("--nextday", action="store_true",
                             default=False, dest="nextDayOpen",
                             help="Next Day Open for open benchmarks")
    cmdlineParser.add_option("--vendordb-user", action="store",
                         default=os.environ.get('VENDORDB_USER'), dest="vendorDBUser",
                         help="Vendor DB User")
    cmdlineParser.add_option("--vendordb-passwd", action="store",
                         default=os.environ.get('VENDORDB_PASSWD'), dest="vendorDBPasswd",
                         help="Vendor DB Password")
    cmdlineParser.add_option("--vendordb-sid", action="store",
                         default=os.environ.get('VENDORDB_SID'), dest="vendorDBSID",
                         help="Vendor DB SID")
    cmdlineParser.add_option("--allow-partial-models", action="store_true",
                         default=False, dest="allowPartialModels",
                         help="extract even if the risk model is incomplete")
    cmdlineParser.add_option("--exclude-22-models", action="store_true",
                         default=False, dest="exclude22Models",
                         help="exclude 22 models or not")
    cmdlineParser.add_option("--histbeta-new", action="store_true",
                             default=False, dest="histBetaNew",
                             help="process historic beta new way or legacy way")
    cmdlineParser.add_option("--new-rmm-fields", action="store_true",
                             default=False, dest="newRMMFields",
                             help="new RMM fields to be supplied or not")
    cmdlineParser.add_option("--cn-rmm-fields", action="store_true",
                             default=False, dest="cnRMMFields",
                             help="CN RMM fields to be supplied or not")
    cmdlineParser.add_option("--track", action="store",
                             default=None, dest="trackAssets",
                             help="track particular asset(s) through the process")
    cmdlineParser.add_option("--factor-suffix", action="store",
                             default="", dest="factorSuffix",
                             help="suffix to be used for non-currency factors")
    
    (options, args) = cmdlineParser.parse_args()
    if len(args) < 1 or len(args) > 2:
        cmdlineParser.error("Incorrect number of arguments")
    
    Utilities.processDefaultCommandLine(options, cmdlineParser)
    modelDB = ModelDB.ModelDB(sid=options.modelDBSID, user=options.modelDBUser,
                              passwd=options.modelDBPasswd)
    marketDB = MarketDB.MarketDB(sid=options.marketDBSID, user=options.marketDBUser,
                              passwd=options.marketDBPasswd)
    
    print('create Range suffix = |%s|' % options.factorSuffix)

    if len(args) == 1:
        dateRanges = [i.strip() for i in args[0].split(',')]
        dates = set()
        for dRange in dateRanges:
            if dRange.find(':') == -1:
                dates.add(Utilities.parseISODate(dRange))
            else:
                (startDate, endDate) = [i.strip() for i in dRange.split(':')]
                startDate = Utilities.parseISODate(startDate)
                endDate = Utilities.parseISODate(endDate)
                dates.update([startDate + datetime.timedelta(i)
                              for i in range((endDate-startDate).days + 1)])
        dates = sorted(dates)
    else:
        startDate = Utilities.parseISODate(args[0])
        endDate = Utilities.parseISODate(args[1])
        dates = [startDate + datetime.timedelta(i)
                 for i in range((endDate-startDate).days + 1)]
    
    if len(options.models) == 0:
        modelList = []
        modelInfoList = []
    else:
        modelList = [riskmodels.getModelByName(model)(modelDB, marketDB)
                     for model in options.models.split(',')]
        modelInfoList = [modelDB.getRiskModelInfo(rm.rm_id, rm.revision)
                         for rm in modelList]
    templateDict = readTemplates()
    modelDB.setMarketCapCache(150)
    modelDB.setVolumeCache(150)
    modelDB.setHistBetaCache(30)
    modelDB.cumReturnCache = None
    modelDB.createCurrencyCache(marketDB, None)
    options.indexNames = processListOption(options.indexNames)
    # if ALL is supplied, go ahead and get the names
    if options.portNames=='ALL':
        portNames=','.join([p.name for p in modelDB.getAllModelPortfolioFamilies()])
        options.portNames = processListOption(portNames)
    else:
        options.portNames = processListOption(options.portNames)
    options.classNames = processListOption(options.classNames)
    logging.info('Classifications: %s', options.classNames)
    options.compositeNames = processListOption(options.compositeNames)
    if len(options.compositeNames) > 0:
        vendorDB = MarketDB.MarketDB(sid=options.vendorDBSID, user=options.vendorDBUser,
                              passwd=options.vendorDBPasswd)
    
    options.derbyCreator = findDerbyCreator(options.derbyCreatorPath)
    error = False
    JAR_BASE = 'jars'
    if options.jarDir:
        # use supplied directory as base for <modelname> directories
        JAR_BASE = options.jarDir
    else:
        # to handle without --destination
        options.jarDir = JAR_BASE
    # Create directories for model jars if necessary
    if options.useSubDirs:
        for model in modelList:
            jarDir = os.path.join(JAR_BASE, model.name)
            if not os.path.exists(jarDir):
                os.mkdir(jarDir)


    ### set up some data for each of the models, start date of the model, new countries that entered this series etc.

    for model in modelList:
        if not hasattr(model, 'modelHack'):
            model.modelHack = Utilities.Struct()
            model.modelHack.nonCheatHistoricBetas = True

        modelDB.dbCursor.execute('SELECT from_dt FROM risk_model_serie where serial_id=:rms_id',
            rms_id=model.rms_id)
        model.startDt = modelDB.dbCursor.fetchone()[0].date()

        # find previous risk model series if it exists
        query="""
             select rms1.serial_id from risk_model_serie rms1, risk_model rm, risk_model_serie rms
             where rms1.RM_ID= rm.model_id
             and rms.rm_id = rm.model_id
             and rms.serial_id = %s
             and rms.from_dt = rms1.thru_dt
             and rms.distribute=1
             and rms1.distribute=1
        """ % (model.rms_id)
        modelDB.dbCursor.execute(query)
        results = modelDB.dbCursor.fetchall()
        model.includeCountries=[]
        model.previd=None
        if len(results) > 0:
            previd=results[0][0]
            query="""select rmg.rmg_id,rmg.mnemonic from rmg_model_map rmm, risk_model_group rmg
                   where rmg.RMG_ID=rmm.RMG_ID and rms_id=%s and
                   not exists (select rmg_id from rmg_model_map rmm1 where rmm1.rms_id=%s and  rmm1.rmg_id=rmm.rmg_id)""" % (model.rms_id, previd)
            modelDB.dbCursor.execute(query)
            results = modelDB.dbCursor.fetchall()
            model.includeCountries=[(r[0],r[1]) for r in results]
        # hard-coded hack for the Greece situation with EM21-MH and EM21-SH
        if model.rms_id in (175,177):
            model.includeCountries=[(20,'GR')]
        logging.info('New countries that entered this series = %s' , ','.join([r[1] for r in model.includeCountries]))

    for d in dates:
        logging.info('Processing %s', d)
        if options.trackAssets is not None:
            trackList = options.trackAssets.split(',')
            allSubIDs = modelDB.getAllActiveSubIssues(d)
            sidStringMap = dict([(sid.getSubIDString(), sid) for sid in allSubIDs])
            options.trackList = [sidStringMap[ss] for ss in trackList if ss in sidStringMap]
        else:
            options.trackList = []
        if len(options.trackList) > 0:
            logging.info('Tracking assets: %s', ','.join([sid.getSubIDString() for sid in options.trackList]))

        # Check which models are active
        activeModels = []
        for model in modelList:
            instance = modelDB.getRiskModelInstance(model.rms_id, d)
            
            startDt = model.startDt
            isFullInstance = instance is not None and instance.has_exposures \
                and instance.has_returns and instance.has_risks
            # if date is prior to start date, then it is not a full instance
            if d < startDt:
                isFullInstance = False
                if instance:
                    instance.has_risks=False
                
            print(instance, startDt , d)
            if instance is not None and (isFullInstance or options.allowPartialModels):
                if not isFullInstance:
                    # Append '-partial{startYear}' to the name to indicate supplemental
                    # data. startYear is taken from the RMS from_dt
                    model.extractName = model.name + '-partial%d%02d' % (startDt.year, startDt.month)
                    model.supplementalData = True
                    logging.info('Extracting partial model as %s', model.extractName)
                    # also need to find if there are countries that need to be excluded in the master/asset files
                    ### what is the logic???
                else:
                    model.extractName = model.name
                    model.supplementalData = False
                modelDir = options.useSubDirs and os.path.join(JAR_BASE, model.name) or options.jarDir
                if options.appendDateDirs:
                    modelDir = os.path.join(modelDir, '%04d' % d.year, '%02d' % d.month)
                    try:
                        os.makedirs(modelDir)
                    except OSError as e:
                        if e.errno != 17:
                            raise
                        else:
                            pass
                if hasattr(model, 'nurseryRMGS'):
                    model.rmg = [rmg for rmg in model.rmg if rmg not in model.nurseryRMGs]
                if options.force:
                    model.forceRun = True
                activeModels.append((model, modelDir))
        if len(activeModels) == 0 and (d.isoweekday() > 5 or (d.day == 1 and d.month == 1)):
            logging.info('No active model, skipping weekends and Jan-1')
            continue
        # Create temporary directories for dump and database files
        dumpDir = tempfile.mkdtemp(prefix='dump.', dir=options.tempDir)
        dbDir = tempfile.mkdtemp(prefix='database.', dir=options.tempDir)
        derbyOptions = Utilities.Struct()
        derbyOptions.targetDir = dumpDir
        derbyOptions.preliminary = options.preliminary
        derbyOptions.newRiskFields = options.newRiskFields
        derbyOptions.newBetaFields = options.newBetaFields
        derbyOptions.newMDVFields = options.newMDVFields
        derbyOptions.allowPartialModels = options.allowPartialModels
        derbyOptions.exclude22Models = options.exclude22Models
        derbyOptions.fileFormatVersion = options.fileFormatVersion
        derbyOptions.histBetaNew = options.histBetaNew
        derbyOptions.newRMMFields = options.newRMMFields
        derbyOptions.cnRMMFields = options.cnRMMFields
        derbyOptions.trackList = options.trackList
        derbyOptions.factorSuffix = options.factorSuffix
        logging.info('Writing file format version %s', derbyOptions.fileFormatVersion)
        try:
            if options.appendDateDirs:
                targetDir = os.path.join(JAR_BASE, '%04d' % d.year, '%02d' % d.month)
                try:
                    os.makedirs(targetDir)
                except OSError as e:
                    if e.errno != 17:
                        raise
                    else:
                        pass
            else:
                targetDir = JAR_BASE
            if options.writeMetaFlag:
                if len(activeModels) > 0:
                    firstJar = activeModels[0][1]
                else:
                    firstJar = targetDir
                logging.debug('write meta files')
                writeDerbyFiles.writeMeta(modelDB, marketDB, derbyOptions, d)
                createJarFile('DbMeta', 'DbMeta', templateDict, dumpDir,
                              dbDir, firstJar, d, options.derbyCreator,
                              encrypt=False, compress=True, version=options.fileFormatVersion)
                if options.useSubDirs:
                    for (riskModel, jarDir) in activeModels[1:]:
                        copyJar('DbMeta', d, firstJar, jarDir)
            for (riskModel, jarDir) in activeModels:
                if options.writeMasterFlag:
                    logging.debug('write master files %s', riskModel.extractName)
                    assetInfo = writeDerbyFiles.writeMasterFile(
                        modelDB, marketDB, riskModel, derbyOptions, d)
                    createJarFile('DbMaster', 'DbMaster-%s' % riskModel.extractName,
                                  templateDict, dumpDir, dbDir, jarDir, d,
                                  options.derbyCreator, compress=True, version=options.fileFormatVersion)
                    gssNames = glob.glob('%s/Shortfall-*_Shortfall' % dumpDir)
                    for gssName in gssNames:
                        if os.access(gssName, os.F_OK) \
                               and os.stat(gssName).st_size > 0:
                            m = re.match('.*Shortfall-(?P<region>.*)_Shortfall', gssName)
                            if m and 'region' in m.groupdict():
                                region = m.groupdict()['region']
                                createJarFile('DbShortfall', 'DbShortfall-%s' % region,
                                              templateDict, dumpDir, dbDir, jarDir, d,
                                              options.derbyCreator,
                                              {'region': region}, version=options.fileFormatVersion)
                else:
                    assetInfo = None
                if options.writeAssetFlag:
                    logging.debug('write asset files %s', riskModel.extractName)
                    writeDerbyFiles.writeAssetFile(
                        modelDB, marketDB, riskModel, derbyOptions, d, assetInfo)
                    if riskModel.supplementalData:
                        supplemental='supplemental_'
                    else:
                        supplemental=''
                    if options.noCumReturn or options.fileFormatVersion >= 4.0:
                        cum_return = ''
                    else:
                        cum_return = 'cumulative_return       DOUBLE PRECISION,'
                    if options.fileFormatVersion >= 4.0:
                        mdv_cols = 'mdv_20d                 DOUBLE PRECISION,\nmdv_60d                 DOUBLE PRECISION,'
                    else:
                        if options.newMDVFields:
                            mdv_cols = 'mdv_20d                 DOUBLE PRECISION,\nmdv_60d                 DOUBLE PRECISION,'
                        else:
                            mdv_cols = ''
                    if options.newRMMFields:
                        rmm_fields = 'isc_adv_score    DOUBLE PRECISION,\nisc_ipo_score   DOUBLE PRECISION,\nisc_ret_score  DOUBLE PRECISION,'
                    else:
                        rmm_fields = ''
                    if options.cnRMMFields:
                        cn_rmm_fields = 'free_float DOUBLE PRECISION,'
                    else:
                        cn_rmm_fields = ''
                    createJarFile('DbAsset', 'DbAsset-%s' % riskModel.extractName,
                                  templateDict, dumpDir, dbDir, jarDir, d,
                                  options.derbyCreator,
                                  {'region': riskModel.extractName, 'supplemental':supplemental, 'cum_return':cum_return, 'mdv_cols':mdv_cols, 'rmm_fields':rmm_fields,'cn_rmm_fields':cn_rmm_fields}, 
                                   version=options.fileFormatVersion)
                    if len(riskModel.rmg) == 1 \
                            and riskModel.rmg[0].mnemonic in ('US', 'CA'):
                        createJarFile('DbFundamental', 'DbFundamental-%s' % riskModel.extractName,
                                      templateDict, dumpDir, dbDir, jarDir, d,
                                      options.derbyCreator,
                                      {'region': riskModel.extractName}, supplemental=riskModel.supplementalData,
                                   version=options.fileFormatVersion)
            if options.writeModelFlag:
                for (riskModel, jarDir) in activeModels:
                    logging.debug('write model files %s', riskModel.extractName)
                    riskModel.setFactorsForDate(d, modelDB)
                    writeDerbyFiles.writeModel(modelDB, marketDB, derbyOptions,
                                               d, riskModel)
                    substitutions = dict()
                    substitutions['model'] = riskModel.extractName
                    expLoadFile = open('%s/Model-%s_load' % (
                        dumpDir, riskModel.extractName), 'r')
                    expLoadStr = expLoadFile.read()
                    expLoadFile.close()
                    substitutions['exposure_load'] = expLoadStr
                    expCreateFile = open('%s/Model-%s_create' % (
                        dumpDir, riskModel.extractName), 'r')
                    expCreateStr = expCreateFile.read()
                    expCreateFile.close()
                    substitutions['exposure_create'] = expCreateStr
                    hierName = '%s/Model-%s_Classification_hierarchy' % (
                        dumpDir, riskModel.extractName)
                    if os.access(hierName, os.F_OK) \
                           and os.stat(hierName).st_size > 0:
                        substitutions['hierarchy_load'] = "CALL SYSCS_UTIL.SYSCS_IMPORT_DATA(null,'CLASSIFICATION_HIERARCHY', null, null, '%(dump)s/Model-%(model)s_Classification_hierarchy', '|', null, null, 1);" % {
                            'dump': dumpDir, 'model': riskModel.extractName}
                    else:
                        substitutions['hierarchy_load'] = ''
                    
                    clsName = '%s/Model-%s_Classification' % (
                        dumpDir, riskModel.extractName)
                    if os.access(clsName, os.F_OK) \
                           and os.stat(clsName).st_size > 0:
                        substitutions['classification_load'] = "CALL SYSCS_UTIL.SYSCS_IMPORT_DATA(null,'CLASSIFICATION', 'NAME,DESCRIPTION,ISLEAF,ISROOT', null, '%(dump)s/Model-%(model)s_Classification', '|', null, null, 1);" % {
                            'dump': dumpDir, 'model': riskModel.extractName}
                    else:
                        substitutions['classification_load'] = ''
                    if options.newRiskFields:
                        substitutions['new_fields'] = 'estu_weight      REAL,\nspec_return      REAL,'
                    else:
                        substitutions['new_fields'] = ''

                    if options.newBetaFields:
                        # if SCM, do one thing else another set of fields
                        # fields to populate are:
                        # Regional = num_of_betas | hbeta_trad | hbeta_home | pbeta_local_hedged| pbeta_local_unhedged | pbeta_global_unhedged
                        # SCM      = num_of_betas | hbeta_trad | hbeta_home | pbeta_local_hedged

                        if len(riskModel.rmg) > 1:
                            # regional model, so use the fields for regional model 
                            substitutions['beta_fields'] = 'num_beta_returns   INTEGER,\nhbeta_trad    REAL,\nhbeta_home   REAL,\npbeta_local_hedged  REAL,\npbeta_local_unhedged   REAL,\npbeta_global_unhedged  REAL,'
                        else:
                            # single country model, so use the fields for SCM
                            substitutions['beta_fields'] = 'num_beta_returns   INTEGER,\nhbeta_trad    REAL,\nhbeta_home   REAL,\npbeta_local_hedged  REAL,'
                    else:
                        substitutions['beta_fields'] = ''
                    if options.newRMMFields:
                        substitutions['rmm_fields'] = 'market_sensitivity_104w    REAL,\nchina_a_estu_flag    SMALLINT,'
                    else:
                        substitutions['rmm_fields'] = ''
                    if options.cnRMMFields:
                        substitutions['cn_rmm_fields'] = 'china_offshore_flag  SMALLINT,'
                    else:
                        substitutions['cn_rmm_fields'] = ''
                    createJarFile('DbModel', 'DbModel-%s' % riskModel.extractName,
                                  templateDict, dumpDir, dbDir,
                                  jarDir, d, options.derbyCreator,
                                  substitutions,
                                   version=options.fileFormatVersion)
            if options.appendDateDirs:
                targetDir = os.path.join(JAR_BASE, '%04d' % d.year, '%02d' % d.month)
                try:
                    os.makedirs(targetDir)
                except OSError as e:
                    if e.errno != 17:
                        raise
                    else:
                        pass
            else:
                targetDir = JAR_BASE
            for indexFamily in options.indexNames:
                logging.debug('write index family %s', indexFamily)
                writeDerbyFiles.writeIndexFamily(
                    modelDB, marketDB, derbyOptions, d, indexFamily, options.nextDayOpen)
                indexName = '%s/Index-%s_Index' % (dumpDir, indexFamily)
                if os.access(indexName, os.F_OK) \
                       and os.stat(indexName).st_size > 0:
                    createJarFile('DbIndex', 'DbIndex-%s' % indexFamily,
                                  templateDict, dumpDir, dbDir, targetDir, d,
                                  options.derbyCreator, {'index': indexFamily})
            if options.clients:
                if options.clients=='ALL':
                # if clients is "ALL" then do something more special
                    clients=[]
                    modelDB.dbCursor.execute('select distinct client from mdl_port_member_client where from_dt <= :dt and :dt < thru_dt',dt=d)
                    results=modelDB.dbCursor.fetchall()
                    clients=[r[0] for r in results]
                else:
                    clients=options.clients.split(',')
                for client in clients:
                    for portFamily in options.portNames:
                        logging.debug('write model portfolio family %s', portFamily)
                        portName, expfile = writeDerbyFiles.writePortFamily2(modelDB, marketDB, derbyOptions, d, portFamily, client=client)
                        if not portName:
                            logging.info('No members in %s so ignoring it', portFamily)
                            continue
                        # make sure that the client directory is set up ahead of the targetsubdirs
                        clientDir = JAR_BASE
                        targetDir=clientDir.rstrip('/') + '/%s' % client
                        if options.appendDateDirs:
                            targetDir = os.path.join(targetDir, '%04d' % d.year, '%02d' % d.month)
                        try:
                            os.makedirs(targetDir)
                        except OSError as e:
                            if e.errno != 17:
                                raise
                            else:
                                pass

                        if os.access(portName, os.F_OK) and os.stat(portName).st_size > 0:
                            createJarFile('DbIndex', 'DbIndex-%s' % (portFamily),
                                  templateDict, dumpDir, dbDir, targetDir, d,
                                  options.derbyCreator, {'index': portFamily}, client=client)
                    shutil.rmtree(dumpDir, True)
                    shutil.rmtree(dbDir, True)

                    # Create temporary directories for dump and database files for next go around if needed 
                    # no worries, will be cleaned up at the end
                    dumpDir = tempfile.mkdtemp(prefix='dump.', dir=options.tempDir)
                    derbyOptions.targetDir = dumpDir
                    dbDir = tempfile.mkdtemp(prefix='database.', dir=options.tempDir)
            else:
                for portFamily in options.portNames:
                    portName, expfile = writeDerbyFiles.writePortFamily2(modelDB, marketDB, derbyOptions, d, portFamily)
                
                    if os.access(portName, os.F_OK) and os.stat(portName).st_size > 0:
                        createJarFile('DbIndex', 'DbIndex-%s' % portFamily,
                                  templateDict, dumpDir, dbDir, targetDir, d,
                                  options.derbyCreator, {'index': portFamily})

            for classification in options.classNames:
                logging.debug('write classification %s', classification)
                writeDerbyFiles.writeClassification(
                    modelDB, marketDB, derbyOptions, d, classification)
                # if special text files were created, save it away
                dtstr = '%4d%02d%02d' % (d.year, d.month, d.day)
                tmpname='%s/Attribute-%s_Source-%s.txt' % (derbyOptions.targetDir, classification, dtstr)
                if os.path.exists(tmpname):
                    finalName = '%s/Attribute-%s_Source-%s.txt' % (targetDir, classification, dtstr)
                    logging.info('Moved file from %s to %s', tmpname, finalName)
                    shutil.move(tmpname, finalName)

                if classification=='AXModelICB':
                    tmpname='%s/Classification-ICB_Data-%s.txt' % (derbyOptions.targetDir, dtstr)
                    finalName = '%s/Classification-ICB_Data-%s.txt' % (targetDir, dtstr)
                else:
                    tmpname='%s/Classification-%s_Data-%s.txt' % (derbyOptions.targetDir, classification, dtstr)
                    finalName = '%s/Classification-%s_Data-%s.txt' % (targetDir, classification, dtstr)
                if os.path.exists(tmpname):
                    logging.info('Moved file from %s to %s', tmpname, finalName)
                    shutil.move(tmpname, finalName)
                if classification not in [ 'AXModelGICS2016', 'AXModelGICS2018','AXModelICB']:
                    createJarFile('DbClassification',
                              'DbClassification-%s' % classification,
                              templateDict, dumpDir, dbDir, targetDir, d,
                              options.derbyCreator,
                              {'classification': classification},
                                   version=options.fileFormatVersion)
            for compositeFamily in options.compositeNames:
                logging.debug('write composite family %s', compositeFamily)
                writeDerbyFiles.writeCompositeFamily(
                    modelDB, marketDB, derbyOptions, d, compositeFamily, vendorDB)
                compName = '%s/Composite-%s_Constituents' % (dumpDir, compositeFamily)
                if os.access(compName, os.F_OK) \
                       and os.stat(compName).st_size > 0:
                    createJarFile('DbComposite',
                                  'DbComposite-%s' % compositeFamily,
                                  templateDict, dumpDir, dbDir, targetDir, d,
                                  options.derbyCreator,
                                  {'composite': compositeFamily})
        except:
            logging.fatal('Error during extraction.', exc_info=True)
            error = True
            sys.exit('Error in extraction')
        
        shutil.rmtree(dumpDir, True)
        shutil.rmtree(dbDir, True)
        logging.info('Done processing %s %s %s', d, dumpDir, dbDir)
        if error:
            break
        
    modelDB.finalize()
    if error:
        sys.exit('Error in extraction')

if __name__ == '__main__':
    main()
    
