#$Id: writeNextGenFlatFiles.py 242585 2021-04-23 13:14:30Z sbell $

import datetime
import numpy.ma as ma
import numpy
import logging
import optparse
#import ConfigParser
import base64
#import os
import os.path
import re
import shutil
import tempfile
import time
import sys
from marketdb import MarketDB
from marketdb import MarketDBUpdates
import riskmodels
from riskmodels.Matrices import ExposureMatrix
from riskmodels import ModelDB
from riskmodels.ModelDB import SubIssue
from riskmodels import ModelID
from riskmodels import Utilities
from riskmodels.wombat import wombat, wombat3, scrambleString
from riskmodels.writeFlatFiles import FlatFilesV3
from riskmodels.writeDerbyFiles import writeMaster, writeClassification
#   writeMaster('DbMaster', model.name, modelDB, date, marketDB,allIssues, issueRMGDict, options)
#numpy.seterr(over='raise',invalid='raise')

def getRiskModelFamilyToModelMap(modelFamilies, mdlDb, mktDb):
    familyMap = dict()
    for family in modelFamilies:
        familyMap[family]=[]
        mdlDb.dbCursor.execute("""SELECT rm.name, rms.rm_id, rms.revision, from_dt, thru_dt
           FROM risk_model rm JOIN risk_model_serie rms ON rm.model_id=rms.rm_id
           WHERE rms.distribute=1 AND 
           exists (select * from risk_model_family_map fm, risk_model_family f where fm.family_id=f.family_id
                and fm.model_id=rm.model_id and f.name = :family_arg)""", family_arg=family)
        for (rmName, rmID, rmRevision,fromdt,thrudt) in mdlDb.dbCursor.fetchall():
            logging.debug('Adding model %s, revision %s to %s family', rmName, rmRevision, family)
            riskModel=riskmodels.getModelByVersion(rmID,rmRevision)
            #riskModel = riskmodels.modelRevMap[(rmID, rmRevision)](mdlDb, mktDb)
            familyMap[family].append([riskModel,fromdt,thrudt])
            #familyMap.setdefault(family, list()).append(riskModel)
    return familyMap

def validModelFamilies(modelFamilies, mdlDb):
    mdlDb.dbCursor.execute("""SELECT rm.name FROM risk_model rm JOIN risk_model_serie rms
       ON rm.model_id=rms.rm_id
       WHERE rms.distribute=1""")
    activeModelNames = [i[0] for i in mdlDb.dbCursor.fetchall()]
    mdlDb.dbCursor.execute("""select name from risk_model_family""")
    results=mdlDb.dbCursor.fetchall()
    activeModelFamilies=set([i[0] for i in results])
    unknownFamilies = set(modelFamilies) - activeModelFamilies
    if len(unknownFamilies) > 0:
        logging.fatal('Unknown model families: %s', ','.join(unknownFamilies))
        logging.info('Supported model families: %s', ','.join(sorted(activeModelFamilies)))
        return False
    return True


NULL = ''
def numOrNull(val, fmt, threshhold=None, fmt2=None):
    """Formats a number with the specified format or returns NULL
    if the value is masked.
    """
    if val is ma.masked or val is None:
        return NULL
    if threshhold and fmt2 and val<threshhold:
        return fmt2 % val
    else:
        return fmt % val

def zeroPad(val, length):
    if val == '':
        return val
    if len(val) >= length:
        return val[:length]
    zeros = length*'0'
    return zeros[:length - len(val)] + val

class TempFile:
    """Helper class to create a file safely by first creating it as a temporary
    file and then moving it to its final name.
    """
    def __init__(self, name, shortDate):
        """Name is the full path of the final location of the file.
        The constructor creates a temporary file in the same directory with
        the suffix of the file as its prefix and the provided shortDate string
        as the suffix.
        """
        fileName = os.path.basename(name)
        directory = os.path.dirname(name)
        tmpfile = tempfile.mkstemp(suffix=shortDate, prefix=fileName[-3:],
                                   dir=directory)
        self.tmpName = tmpfile[1]
        self.tmpFile = os.fdopen(tmpfile[0], 'w')
        self.finalName = name
    
    def getFile(self):
        """Returns the file object of the temporary file created by the constructor.
        """
        return self.tmpFile
    
    def getTmpName(self):
        """Returns the name of the temporary file.
        """
        return self.tmpName
    
    def __enter__(self):
        self.tmpFile.__enter__()
        return self

    def __exit__(self, *args):
        """Alias for closeAndMove() to allow use in 'with' statements.
        """
        self.tmpFile.__exit__(*args)
        self.closeAndMove()

    def closeAndMove(self):
        """Close the temporary file and move it to its final location (name argument
        of the constructor).
        Also sets the file permissions to 0644.
        """
        self.tmpFile.close()
        logging.info("Move file %s to %s", self.tmpName, self.finalName)
        shutil.move(self.tmpName, self.finalName)
        os.chmod(self.finalName,0o644)
        return self.finalName

def getExposureAssets(date, expMatrix, svDict, cashAssets, rmi, options, modelDB, marketDB):
    exposureAssets = []
    exposureAssetIDs = []
    for (aIdx, asset) in enumerate(expMatrix.getAssets()):
        if svDict != None and asset not in svDict:
            continue
        modelID = asset.getModelID()
        exposureAssetIDs.append(modelID)
        exposureAssets.append(asset)
    exposureAssets.extend(cashAssets)
    exposureAssetIDs.extend([i.getModelID() for i in cashAssets])
    # exclude assets which shouldn't be extracted
    excludes = modelDB.getProductExcludedSubIssues(date)
    if options.preliminary:
        excludes.extend(modelDB.getDRSubIssues(rmi))
    for e in excludes:
        svDict.pop(e, None)
    exposureAssets = list(set(exposureAssets) - set(excludes))
    exposureAssetIDs = list(set(exposureAssetIDs) - set([i.getModelID() for i in excludes]))
    return (sorted(exposureAssets), sorted(exposureAssetIDs))

class FlatFilesV4 (FlatFilesV3):
    """Class to create flat files in version 4 format aka the next generation
    """
    vanilla = False
    
    def encryptFile(self, date, inputFileName, options):
        """
            defer to the common method
        """
        target=options.encryptedDir
        if options.appendDateDirs:
            target = os.path.join(target, '%04d' % date.year, '%02d' % date.month)
            try:
                os.makedirs(target)
            except OSError as e:
                if e.errno != 17:
                    raise
                else:
                    pass

 
        Utilities.encryptFile(date, inputFileName, target)
        return 
        
    def writeDateHeader(self, options, outFile, modelList=None):
        outFile.write('#DataDate: %s\n' % self.dataDate_)
        # write createDate in UTC
        gmtime = time.gmtime(time.mktime(self.createDate_.timetuple()))
        utctime = datetime.datetime(year=gmtime.tm_year,
                                    month=gmtime.tm_mon,
                                    day=gmtime.tm_mday,
                                    hour=gmtime.tm_hour,
                                    minute=gmtime.tm_min,
                                    second=gmtime.tm_sec)
        outFile.write('#CreationTimestamp: %sZ\n' %
                      utctime.strftime('%Y-%m-%d %H:%M:%S'))
        outFile.write("#FlatFileVersion: 3.3\n")        
        if modelList:
            familyName=[i.name.rstrip('MH').rstrip('SH').rstrip('SH-S').rstrip('MH-S') for i in modelList][0]
            outFile.write('#ModelFamily: %s\n' % familyName)
            outFile.write('#ModelName: %s\n' % ', '.join([m.name for m in modelList]))
            outFile.write('#ModelNumeraire: %s\n' % modelList[0].numeraire.currency_code)
            
    def getExposureAssets(self, expMatrix, svDict, cashAssets, rmi, options, modelDB, marketDB):
        return getExposureAssets(self.dataDate_, expMatrix, svDict, cashAssets, rmi, options, modelDB, marketDB)

    def writeIdentifierUpdates(self,dt, exposureAssets, modelDB, marketDB, 
                               options, outFile, outFile_nosedol=None,
                               outFile_nocusip=None, outFile_neither=None,
                               modelList=None):
        """
            write out just the updates
        """
        logging.info("Writing out identifier updates only")
        axidStr, modelMarketMap=modelDB.getMarketDB_IDs(marketDB, exposureAssets)
        # build up the reverse map - do we have to???
        axidMap={}
        modelidMap={}
        for k, marketList in modelMarketMap.items():
            for fromdt, thrudt, marketid in marketList:
                if fromdt <= dt and dt< thrudt:
                    axidMap[marketid]=k
            
        outFiles = [outFile, outFile_nosedol, outFile_nocusip, outFile_neither]
        for f in outFiles:
            if f:
                self.writeDateHeader(options, f, modelList)
                f.write('#Columns: AxiomaID|IDType|Changed ID Value\n')
                f.write('#Type: ID|Set|Attribute\n')
                f.write('#Unit: ID|NA|Text')
                f.write('\n')

        # get list of all terminations and deletions

        rmsids = ','.join([str(i.rms_id) for i in modelList])

        # find last date for this family
        query="""select max(dt) from risk_model_instance where dt < :dt and rms_id in (%s)""" % rmsids
        modelDB.dbCursor.execute(query,dt=dt)
        prevdttime=modelDB.dbCursor.fetchall()[0][0]
        prevdt=datetime.date(prevdttime.year, prevdttime.month, prevdttime.day)
        
        lookback=(dt-prevdt).days  + 1
        logging.info('Using %d days for lookback for this model', lookback)
   
        query="""
                select distinct issue_id, 'TERMINATED' from rms_issue_log l
                where rms_id in (%(rmsids)s)
                and action_dt between :dt - %(lookback)s and :dt
                and action='UPDATE'
                and exists (select * from rms_issue rms where rms.issue_id=l.issue_id and rms.rms_id=l.rms_id
                        and rms.thru_dt <= :dt)

                union
                select distinct issue_id, 'DELETED' from rms_issue_log l
                where rms_id in (%(rmsids)s)
                and action_dt between :dt - %(lookback)s and :dt
                and action='DELETE'
        """ % {'rmsids':rmsids,'lookback':lookback}
        modelDB.dbCursor.execute(query,dt=dt)
        updateDict={}
        for r in modelDB.dbCursor.fetchall():
            updateDict['%s|%s' % (r[0][1:],r[1])] = ''
        updates=MarketDBUpdates.Corrections(marketDB, dt)
        CORRECTION_TYPES=['Company Name','TICKER','ISIN','SEDOL','CUSIP','Axioma Asset Type', 'GICS','Market','Trading Currency', 'Company ID']
        cidNameChanges={}

        for corrType in CORRECTION_TYPES:
            changeDict=updates.getChanges(list(axidMap.keys()), corrType)
            #print corrType, changeDict
            for marketid in changeDict.keys():
                for changeEl in changeDict[marketid]:
                    changeType, changeVal = changeEl[:2]
                    # if it is a company name alone, we get the ID along as the third column
                    if changeType=='Company Name':
                        cid=changeEl[2]
                        cidNameChanges[cid]=changeVal
                    updateDict['%s|%s' % (axidMap.get(marketid).getPublicID(),changeType)]= changeVal or ''
        for k in sorted(updateDict.keys()):
            axid,changeType=k.split('|')
            for idx,f in enumerate(outFiles):
                # idx=0 both, 1=no sedol, 2=no cusip, 3=neither
                if f and axidMap.get(marketid,None):
                    if changeType=='SEDOL' and idx in (1,3):
                        continue
                    if changeType=='CUSIP' and idx in (2,3):
                        continue
                if changeType != 'DELETED' and ('%s|%s' % (axid,'DELETED')) in updateDict:
                    continue
                f.write('%s|%s|%s\n' % (axid, changeType, updateDict[k]))
        for k,v in sorted(cidNameChanges.items()):
            for f in outFiles:
                # write out the company ID changes now
                if f:
                    f.write('%s|%s|%s\n'  % (k,'Company Name',v))
                
    def writeDay(self, options, rmiList, riskModelList, modelDB, marketDB, modelFamily):
        rmi = rmiList[0]

        # pick the date from any of the rmis, use the first one, there has to be at least one
        d = rmiList[0].date
        self.dataDate_ = d
        self.createDate_ = modelDB.revDateTime
        # use the mnemonic as needed later on
        mnemonic=None
        
        familyMnemonic=[i.mnemonic.rstrip('-MH').rstrip('-SH').rstrip('-SH-S').rstrip('-MH-S') for i in riskModelList][0]
        familyMnemonic=[i.name.rstrip('MH').rstrip('SH').rstrip('SH-S').rstrip('MH-S') for i in riskModelList][0]
        
        #mnemonic = riskModel.mnemonic
        expMList=[None] * len(rmiList)
        svDataDictList=[None] * len(rmiList)
        specCovList=[None] * len(rmiList)
        #estUList = [None] * len(rmiList)
        exposureAssetsList = [None] * len(rmiList)
        exposureAssetIDsList = [None] * len(rmiList)
                
        if options.writeAssetIdh or options.writeAssetIdu:
            cashAssets = set()
            mktPortfolioDict = {}
            # find cash assets for active RMGs and add to master
            modelDB.dbCursor.execute("""SELECT rmg.rmg_id, rmg.mnemonic, rmg.description
                        FROM RISK_MODEL_GROUP rmg
                        where exists (select * from rmg_model_map rmm where rmm.rmg_id = rmg.rmg_id)""")
            for r in modelDB.dbCursor.fetchall():
                # add cash assets
                rmg = modelDB.getRiskModelGroup(r[0])
                for i in modelDB.getActiveSubIssues(rmg, d):
                    if i.isCashAsset():
                        cashAssets.add(i) 
                mpdict=modelDB.getRMGMarketPortfolio(rmg, d)
                mktPortfolioDict.update(mpdict)
                
            for idx, rmi in enumerate(rmiList):
                riskModel=riskModelList[idx]
                logging.info('Loading exposure matrix for %s', rmi.rms_id)
                expMList[idx] = riskModel.loadExposureMatrix(rmi, modelDB)
                logging.info('Loading specific risks for %s', rmi.rms_id)
                (svDataDict, specCov) = riskModel.loadSpecificRisks(rmi, modelDB)
                svDataDictList[idx]=svDataDict
                specCovList[idx]=specCov
                #logging.info('Loading estimation universe for %s', rmi.rms_id)
                #estUList[idx] = riskModel.loadEstimationUniverse(rmi, modelDB)
                (exposureAssetsList[idx], exposureAssetIDsList[idx]) = self.getExposureAssets(
                        expMList[idx], svDataDictList[idx], cashAssets, rmi, options, modelDB, marketDB)
                    
            # at this point build up the exposureAssets and exposureAssetIDs as the master list of all
            # exposure assets across the whole family
            exposureAssets=[]
            for expAssets in exposureAssetsList:
                if expAssets:
                    exposureAssets += expAssets
            exposureAssets=list(set(exposureAssets))
            exposureAssetIDs=[]
            for expAssetIDs in exposureAssetIDsList:
                if expAssetIDs:
                    exposureAssetIDs += expAssetIDs
            exposureAssetIDs=list(set(exposureAssetIDs))

            logging.info('Built up exposure assets')

        shortdate='%04d%02d%02d' % (d.year, d.month, d.day)
        target=options.targetDir
        if options.appendDateDirs:
            target = os.path.join(target, '%04d' % d.year, '%02d' % d.month)
            try:
                os.makedirs(target)
            except OSError as e:
                if e.errno != 17:
                    raise
                else:
                    pass
                
        # write identifier mapping for exposure assets

        if options.writeAssetIdu:
            updateTypes = [x for x in [options.writeAssetIdu and 'idu'] if x]
            logging.info('Working on %s', ','.join(updateTypes))
            for updateType in updateTypes:
                mnemonic=familyMnemonic
                # or should we use the family name passed in?
                #mnemonic=modelFamily
                tmpfile=tempfile.mkstemp(suffix=shortdate,prefix=updateType,dir=target)
                #tmpfile is a tuple, contain the unix style FD in [0] and the name in [1]
                os.close(tmpfile[0])
                tmpfilename=tmpfile[1]
                
                tmpfile2=tempfile.mkstemp(suffix=shortdate,prefix=updateType,dir=target)
                #tmpfile is a tuple, contain the unix style FD in [0] and the name in [1]
                os.close(tmpfile2[0])
                tmpfilename2=tmpfile2[1]
                
                tmpfile3=tempfile.mkstemp(suffix=shortdate,prefix=updateType,dir=target)
                #tmpfile is a tuple, contain the unix style FD in [0] and the name in [1]
                os.close(tmpfile3[0])
                tmpfilename3=tmpfile3[1]
                
                tmpfile4=tempfile.mkstemp(suffix=shortdate,prefix=updateType,dir=target)
                #tmpfile is a tuple, contain the unix style FD in [0] and the name in [1]
                os.close(tmpfile4[0])
                tmpfilename4=tmpfile4[1]
                
                outFileName = '%s/%s.%04d%02d%02d.%s' % (target, mnemonic,
                                                          d.year, d.month, d.day, updateType)
                outFile = open(tmpfilename, 'w')
                outFileName2 = '%s/%s-CUSIP.%04d%02d%02d.%s' % (target, mnemonic,
                                                          d.year, d.month, d.day, updateType)
                outFile2 = open(tmpfilename2, 'w')
                outFileName3 = '%s/%s-SEDOL.%04d%02d%02d.%s' % (target, mnemonic,
                                                          d.year, d.month, d.day, updateType)
                outFile3 = open(tmpfilename3, 'w')
                outFileName4 = '%s/%s-NONE.%04d%02d%02d.%s' % (target, mnemonic,
                                                          d.year, d.month, d.day, updateType)
                outFile4 = open(tmpfilename4, 'w')
                logging.info("Writing to %s, %s, %s, %s", tmpfilename, tmpfilename2, tmpfilename2, tmpfilename4)
                
                if updateType=='idm':
                    logging.info("Writing idm file")
                    self.writeIdentifierMapping(d, exposureAssetIDs, modelDB, marketDB, options,
                                            outFile, outFile2, outFile3, outFile4, riskModelList)
                elif updateType=='idu':
                    logging.info("Writing idu file")
                    self.writeIdentifierUpdates(d, exposureAssetIDs, modelDB, marketDB, options,
                                                outFile, outFile2, outFile3, outFile4, riskModelList)

                outFile.close()
                outFile2.close()
                outFile3.close()
                outFile4.close()
                
                logging.info("Move file %s to %s", tmpfilename, outFileName)
                shutil.move(tmpfilename, outFileName)
                os.chmod(outFileName,0o644)
                logging.info("Move file %s to %s", tmpfilename2, outFileName2)
                shutil.move(tmpfilename2, outFileName2)
                os.chmod(outFileName2,0o644)
                logging.info("Move file %s to %s", tmpfilename3, outFileName3)
                shutil.move(tmpfilename3, outFileName3)
                os.chmod(outFileName3,0o644)
                logging.info("Move file %s to %s", tmpfilename4, outFileName4)
                shutil.move(tmpfilename4, outFileName4)
                os.chmod(outFileName4,0o644)

                for outFileName in [outFileName,outFileName2, outFileName3,outFileName4]:
                    if options.encryptedDir:
                        self.encryptFile(d,outFileName, options)

                
                
        # write out ID history if needed
        if options.writeAssetIdh:
            mnemonic=familyMnemonic
            logging.info("build up all Issues dictionary")
            
            allIssues = dict([(issuei, (from_dt, thru_dt)) for (issuei, from_dt, thru_dt)
                              in modelDB.getAllIssues()])
            tempDict=dict([(expasset,1) for expasset in exposureAssetIDs])
            for k in list(allIssues.keys()):
                if k not in tempDict:
                    del allIssues[k]
            
            logging.info("Done.....build up all Issues dictionary")
            # map issues to their RMG
            subIssueRMGDict = dict(modelDB.getSubIssueRiskModelGroupPairs(d))
            issueRMGDict = dict([(si.getModelID(), rmg.rmg_id)
                         for (si, rmg) in subIssueRMGDict.items()])
            
            options.writeText=True
            outFileName, outFileName2, outFileName3, outFileName4=writeMaster('DbMaster', mnemonic, modelDB, d, marketDB,
                        allIssues, issueRMGDict, options, riskModelList)
            for outFileName in [outFileName,outFileName2, outFileName3,outFileName4]:
                    if options.encryptedDir:
                        self.encryptFile(d,outFileName, options)


        if options.classification:
            classFileName, hierFileName, assetFileName=writeClassification(modelDB,marketDB,options, d, options.classification)
            # encrypt just the asset file, no point encrypting the classification and hierarchy files
            if options.encryptedDir:
                self.encryptFile(d,assetFileName, options)
             

def main():
    usage = "usage: %prog [options] <startdate or datelist> [<end-date>]"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addModelAndDefaultCommandLine(cmdlineParser)
    cmdlineParser.add_option("-d", "--directory", action="store",
                             default='.', dest="targetDir",
                             help="directory for output files")
    cmdlineParser.add_option("--encrypted-directory", action="store",
                             default=None, dest="encryptedDir",
                             help="directory for encrypted output files")
    cmdlineParser.add_option("--version", action="store",
                             default=4, type='int', dest="formatVersion",
                             help="version of flat files to create")
    cmdlineParser.add_option("--no-idh", action="store_false",
                             default=True, dest="writeAssetIdh",
                             help="don't create .idh file")
    cmdlineParser.add_option("--no-idu", action="store_false",
                             default=True, dest="writeAssetIdu",
                             help="don't create .idu file")
    cmdlineParser.add_option("--classification", action="store",
                             default=None, dest="classification",
                             help="don't create .cls file")
    cmdlineParser.add_option("-p", "--preliminary", action="store_true",
                             default=False, dest="preliminary",
                             help="Preliminary run--ignore DR assets")
    cmdlineParser.add_option("--target-sub-dirs", action="store_true",
                             default=False, dest="appendDateDirs",
                             help="Append yyyy/mm to end of output directory path")
    cmdlineParser.add_option("--model-families", action="store",
                             default='', dest="modelFamilies",
                             help="comma-separated list of risk model families")
    cmdlineParser.add_option("--model", action="store",
                             default='', dest="model",
                             help="comma-separated list of risk model families")
    (options, args) = cmdlineParser.parse_args()
    if len(args) < 1 or len(args) > 2:
        cmdlineParser.error("Incorrect number of arguments")
    
    
    modelDB = ModelDB.ModelDB(sid=options.modelDBSID, user=options.modelDBUser, passwd=options.modelDBPasswd)
    marketDB = MarketDB.MarketDB(sid=options.marketDBSID, user=options.marketDBUser, passwd=options.marketDBPasswd)

    Utilities.processDefaultCommandLine(options, cmdlineParser)
    
    modelDict=modelDB.getModelFamilies()
    if not options.modelFamilies and not options.model:
        cmdlineParser.error("Must specify a model family")
        ###
    if options.modelFamilies:
        modelFamilies = [i.strip() for i in options.modelFamilies.split(',') if len(i) > 0]
    elif options.model:
        modelFamilies=list(set([modelDict[i].strip() for i in options.model.split(',') if len(i)> 0]))

    logging.info("Working on families %s", ','.join(modelFamilies))
    if not validModelFamilies(modelFamilies, modelDB):
        logging.fatal('Bad model family %s specified', modelFamilies)
        sys.exit(1)

    modelFamilyMap = getRiskModelFamilyToModelMap(modelFamilies, modelDB, marketDB)
    for k,v in modelFamilyMap.items():
        logging.debug('%s %s',k,v)

    modelDB.setTotalReturnCache(150)
    modelDB.setMarketCapCache(150)
    modelDB.setVolumeCache(150)
    modelDB.setHistBetaCache(30)
    modelDB.cumReturnCache = None
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
        dates = sorted(dates,reverse=True)
    else:
        startDate = Utilities.parseISODate(args[0])
        endDate = Utilities.parseISODate(args[1])
        dates = sorted([startDate + datetime.timedelta(i)
                      for i in range((endDate-startDate).days + 1)], reverse=True)
    
    if options.formatVersion == 4:
        fileFormat_ = FlatFilesV4()
        options.fileFormatVersion = 4.0
    else:
        logging.fatal('Unsupported format version %d', options.formatVersion)
        sys.exit(1)
    
    for d in dates:
        for modelFamily in modelFamilies:
            modelClassListInfo=modelFamilyMap[modelFamily]
            riskModelList=[]
            rmiList=[]
            noModel=False
            for modelClass,fromdt,thrudt in modelClassListInfo:
                fromdt=datetime.date(fromdt.year,fromdt.month,fromdt.day)
                thrudt=datetime.date(thrudt.year,thrudt.month,thrudt.day)
                if fromdt > d or  thrudt <= d:
                    logging.info( 'Ignoring %s on %s', modelClass,d)
                    continue
                if noModel:
                    continue
                riskModel=modelClass(modelDB, marketDB)
                riskModel.setFactorsForDate(d, modelDB)
                
                rmi=riskModel.getRiskModelInstance(d, modelDB)

                if rmi != None and rmi.has_risks:
                    rmiList.append(rmi)
                    riskModelList.append(riskModel)
                else:
                    logging.warning('No risk model %s instance on %s', modelClass,d)
            # write out one day's data for this model family if noModel is true:
            if len(rmiList) > 0:
                for rmi in rmiList:
                    logging.info('Processing %s exps=%s returns=%s risks=%s %d', d, rmi.has_exposures, rmi.has_returns, rmi.has_risks, rmi.rms_id)
                fileFormat_.writeDay(options, rmiList, riskModelList, modelDB, marketDB, modelFamily=modelFamily)                
            logging.info("Done writing %s for %s", d, modelFamily)
    
    modelDB.finalize()
    marketDB.finalize()

if __name__ == '__main__':
    main()
