"""
This program is to write AX-WW riskmodel to the flatfile satisfying the Factset format
And prices of each security are converted to the USD 
The source is modified based on writeFlatFiles.py by Stefan Schmieta
"""

__author__     ="Weiwei Deng (wdeng@axiomainc.com)"
__copyright__  ="Copyright (C) 2008 Axioma Inc"
__version__    = "$Revision: 3.0.1$"

import datetime
from math import sqrt
import numpy.ma as ma
import logging
import optparse
from marketdb import MarketDB
from riskmodels import ModelDB
from riskmodels import MFM
from riskmodels import Utilities

#default value in FactSet flat files
NOT_AVAILABLE = 'NA'    #the factor id for unknown factor if an asset has no assigned factor, such as industry classification
VALUE_NO_CONSTRAINT = '-777'   #FactSet/NorthField convention, meaning no constraint applied
VALUE_MAINTAIN_WEIGHT = '-999' #FactSet/NorthField convention, meaning maintaining weight

NULL = ''

def numOrNull(val, fmt):
    """Formats a number with the specified format or returns NULL
    if the value is masked.
    """
    if val is ma.masked or val is None:
        return NULL
    return fmt % val

def zeroPad(val, length):
    """Pad a value with zero's to the left to bring to the specified length
       if the length of the value is less than the specified length
    """
    if val == '':
        return val
    if len(val) >= length:
        return val[:length]
    zeros = length*'0'
    return zeros[:length - len(val)] + val

def isUSorCA(riskModel):
    """ Determine whether the domicile of a risk model is either US or Canada
    """
    modelName = riskModel.name
    if (modelName[:2]== 'US' or modelName[:2] == 'CA'):
        return True
    else:
        return False

def writeSecurityDataFile(d, marketDB, modelDB, expM, svDict, riskModel, factorIDs, outFile):
    """ Write to FactSet .SEC file.
        The first 6 columns of the file are mandatory and must be in the following order:
        (1) id - CUSIP or SEDOL with check digit. 
            For US country risk model and Canadian contry risk model, CUSIPs are the primary identifiers. 
            For other risk models, SEDOLs are the primary and CUSIPs are used if SEDOLs are not available.
            There will be warning messages if neither a SEDOL or a CUSIP is available for a security.
            However, we are not checking whether duplicate identifiers exist in this file.   
        (2) name - the security description enclosed by double quote characters.
        (3) industry code - the id from .IND file
        (4) price - security price, which should be larger than ZERO in the currency of numeraire used by the model
        (5) specific/residue risk in terms of monthly standard deviation, which is deduced from dividing our annualized specific risk by square root of 12.
        (6) share-outstanding - security shares outstanding
        (7 ... n) exposures to factors, which are in the same order as they are in the .MDL file.
                And the column position of each factor should be correspondingly recorded in the 6th column in the .MDL file. 
    """
    
    #default industry code will be used if there is no industry classification assigned to a security
    indCode = NOT_AVAILABLE
    
    writeCashAssets(riskModel,expM, outFile)
    
    sqrt12 = sqrt(12)
    exposureAssets = expM.getAssets()
    exposureAssetIds = [a.getModelID() for a in exposureAssets]
    
    cusipMap = modelDB.getIssueCUSIPs(d, exposureAssetIds, marketDB)
    sedolMap = modelDB.getIssueSEDOLs(d, exposureAssetIds, marketDB)
    issuerMap = modelDB.getIssueNames(d, exposureAssetIds, marketDB)
    
    expMatrix = expM.getMatrix()
    
    #get price and share outstanding
    histLen = 60
    dateList = modelDB.getDates(riskModel.rmg, d,  histLen)
    
    #convert to the currency as the numeraire price data
    modelDB.log.info('Extract prices in the Currency - %s.'% riskModel.numeraire.currency_code) 
    convertToCurrencies = dict()
    for asset in exposureAssets:
        convertToCurrencies[asset] = riskModel.numeraire.currency_id
    ucp = modelDB.loadUCPHistory(dateList, exposureAssets,convertToCurrencies)
    latestUCP = ma.masked_where(True,ucp.data[:,0])
    tso = modelDB.loadTSOHistory(dateList, exposureAssets)
    latestTSO = ma.masked_where(True,tso.data[:,0])
    
    #extract data for each security in the exposure matrix
    for i in range(len(exposureAssets)):
        
        asset = exposureAssets[i]
        axId = asset.getModelID().getPublicID()
        
        #extract specific/residue risk and the exposure
        if asset in svDict:
            #convert the specific risk to monthly term
            residualRisk = 100*svDict[asset]/sqrt12
        else: #skip the security if it doesn't have the specific/residue risk
            continue
        
        # extract price and shares outstanding from the most recent date
        for j in range(len(dateList)-1,-1,-1):
            if not ucp.data[i,j] is ma.masked:
                latestUCP[i] = ucp.data[i,j]
            if not tso.data[i,j] is ma.masked:
                latestTSO[i] = tso.data[i,j]

        #validate the price
        if not latestUCP[i] > 0: #skip the security with non-positive price
            modelDB.log.warning('%s has a non-positive price(%d)' %(axId, latestUCP[i])) 
            continue
                              
        #extract cusip or sedol and name of a security           
        if isUSorCA(riskModel): #extract cusip as the primary id for US or CA risk model
            #modelDB.log.debug('CUSIP will be used for %s(%s).' % (riskModel.mnemonic, riskModel.name))
            cusip = cusipMap.get(asset.getModelID(), '')
            if len(cusip)==0: #skip the security if the cusip is not available for US or Canadian security
                modelDB.log.warning('On %s, %s has NO CUSIP' % (d.strftime("%Y-%m-%d"), axId))
                continue
            else:
                assetId = cusip
        else: #get sedol first, then cusip for WorldWide and other models 
            #modelDB.log.debug('SEDOL and CUSIP will be used for %s(%s).' % (riskModel.mnemonic, riskModel.name))
            sedol = sedolMap.get(asset.getModelID(), '')
            if len(sedol)==0:
                cusip = cusipMap.get(asset.getModelID(), '')
                if len(cusip)==0: #skip the security if neither id is available
                    modelDB.log.warning('On %s, %s has NO CUSIP or SEDOL' % (d.strftime("%Y-%m-%d"), axId))
                    continue
                else:
                    assetId = cusip
            else: 
                assetId = sedol
        issuer = issuerMap.get(asset.getModelID(),'')
              
        #extract the industry code
        if (not isinstance(riskModel, MFM.StatisticalFactorModel)):
            numInd = 0
            for j,exposure in enumerate(expMatrix[:,i]):
                factorName = expM.factors_[j]
                if exposure is not ma.masked and expM.getFactorType(factorName)==expM.IndustryFactor:
                    numInd += 1
                    indCode = factorIDs[factorName]
                    continue
            if numInd > 1:
                modelDB.log.warning('%d industries for %s'% (numInd, axId))
            elif numInd == 0: #default NOT_AVAILABLE value will be assigned if no industry classification is assigned
                modelDB.log.warning('NO industry classification for %s'% (axId))
                
        #output data for each security                                 
        outFile.write('%s,\"%s\",%s,%s,%s,%s' 
                      % (assetId, issuer, indCode,numOrNull(latestUCP[i],'%.4f'),
                         numOrNull(residualRisk, '%.6f'),numOrNull(latestTSO[i],'%.0f')))    

        #exposure to each factor
        for factorSeq, exposure in enumerate(expMatrix[:,i]):
            if exposure is not ma.masked:
                outFile.write(',%.4f' % exposure)
            else: #todo: is it right to assign a ZERO value if an exposure to a factor is not available for a security
#                modelDB.log.warning('NO exposure to factor %s for security axid=%s. Will fill in ZERO value.'
#                                 % (riskModel.factors[factorSeq].name,axId))
                outFile.write(',0')
                
        #done with one asset
        outFile.write('\r\n')

def writeCashAssets(riskModel,expM,outFile):
    """ Write the cash assets in different currencies into .SEC file, which must have at least one cash asset.
    """
    if (hasattr(riskModel, "currencies")):
        currencyFactors = riskModel.currencies    
      
        for curr in currencyFactors:
            #curr.name should be ISO code or at least one alphanumeric word
            writeOneCashAsset(outFile, curr.name, curr.description, riskModel.factors)
              
    elif (hasattr(riskModel, "numeraire")):
        writeOneCashAsset(outFile, riskModel.numeraire.currency_code, riskModel.numeraire.currency_code, riskModel.factors)
        
    else:
        riskModel.log.error('Error in RiskModel %s(%s) : Neither Factor Group nor numeraire is available.' 
                            % (riskModel.name, riskModel.description))
        
def writeOneCashAsset(outFile, currencyIsoId, currencyName, factors):
    """The following rules apply to the cash asset:
        (1) the identifier as CASH_[Currency ISO name]
        (2) the industry classification for a cash asset is the default not_available value
        (3) the price and share-outstanding have the value of 1
        (4) the specific/residual risk is ZERO
        (5) the exposure to each factor is ZERO except for the exposure to the same currency factor, which is 1.
    """
    #the exposure to the cash unit currency is 1 and 0 to the other factors       
    cashId = 'CASH_' + currencyIsoId
    cashName = currencyName
    
    #the first 6 columns : id, name, IndId(NA), price(1), residue risk(0 for cash), share-outstanding(1)
    outFile.write('%s,\"%s\",%s,1,0,1' % (cashId, cashName, NOT_AVAILABLE))
    
    #the rest columns: the exposure to each factor
    for factor in riskModel.factors:
        #the exposure is 1 if the factor is the currency
        if factor.name == currencyName:
            outFile.write(',1')
        else: #the exposure of the cash asset to any other factor except for its own currency is ZERO
            outFile.write(',0')
            
    outFile.write('\r\n')


def writeModelFile(d, modelDB, expM, factorVariances, factors, factorIDs, outFile):
    """ Write to FactSet .MDL file, which defines the factors in the model and lists their annual variances.
        The factors must occupy the same column positions(e.g., column 7, column 8, etc), have
        the same name(factorId), and be in the same order as the coefficients in the .SEC file,
        .COR, .RET files.
        The columns are 
        (1) FactorId
                The FactorId is created by [FactorType][SequenceNumber] with the sequence number is the order of the factor
                stored in the list. Other more descriptive coding schemes can be used. But the names of our industries are 
                too long(and sometimes have spaces) as identifiers. (weiwei@2009-05-06)
        (2) Factor short name
        (3) Factor Name
        (4) Alpha  - ZERO is used.
        (5) Variance - the factor variance, rescaled by multiplication of 10000 from our normalized variance
        (6) FieldN - corresponds to the column position in the .SEC file. The FieldN for
                     the 1st factor is 7, then 8 for the 2nd one, etc.
        (7) MinVal - VALUE_NO_CONSTRAINT(-777) is used.
        (8) MaxVal - VALUE_NO_CONSTRAINT(-777) is used.
    """
    for (i, f)  in enumerate(factors):
        #FactSet format: 
        fieldN = i+7
        factorName = f.name
        factorDesc = f.description
        factorId = factorIDs[factorName]
        outFile.write('%s,\"%s\",\"%s\",%d,%.4f,%d,%s,%s\r\n'
                         % (factorId, factorName, factorDesc,
                            0, factorVariances[factorDesc]*10000, 
                            fieldN, VALUE_NO_CONSTRAINT, VALUE_NO_CONSTRAINT))


def writeCorrelationMatrixFile(d, factorCov, factors,outFile):
    """ Write to FactSet .COR file with the rectangular correlation matrix data of factors.
        The value of 1 will occupy the diagonal for the matrix.
        
        The correlation of factor i and j is computed from covariance of factor i and j, variance of factor i
        and variance of factor j. Note that the variance of each factor is recorded in FactSet .MDL file.
    """
    modelFactorVariances = dict()   
    maxColIndex = len(factors)
    
    for i in range(len(factors)):
        modelFactorVariances[factors[i].description]=factorCov[i,i]
        for j in range(len(factors)):
            if i==j:
                outFile.write('%.4f' % (1.0))
            else:
                corr = factorCov[i,j]/(sqrt(factorCov[i,i]*factorCov[j,j]))
                outFile.write('%.4f' % corr)
            if j < maxColIndex : #not to print the comma after the last column
                outFile.write(',')
        outFile.write('\r\n')    
           
    return modelFactorVariances

def labelFactorsByType(expM, factors):
    """Create a dictionary mapping factors to names by factor type, like "Style1", "Style2" for the style
    factors.
    """
    typeCounters = dict()
    factorIDs = dict()
    for f in factors:
        fType= expM.getFactorType(f.name)
        if fType not in typeCounters:
            typeCounters[fType] = 0
        typeCounters[fType] += 1
        if fType == expM.InterceptFactor:
            factorIDs[f.name] = '%s' % (fType.name)
        else:
            factorIDs[f.name] = '%s%d' % (fType.name, typeCounters[fType])
    assert(typeCounters.get(expM.InterceptFactor, 0) <= 1)
    return factorIDs


def writeIndustryFile(d, industries, industryFactors, outFile):
    """ Write to FactSet .IND file, which corresponds to the Industry code in the .SEC file.
        It assigns names to the codes in the data file.
        The columns are in order of 
        (1) Factor/Industry ID   --- same as the factor id in the .MDL file
        (2) Factor/Industry Name, double quoted
        (3) Penalty  - ZERO is used.
        (4) MinConst - VALUE_NO_CONSTRAINT(-777) is used.
        (5) MaxConst - VALUE_NO_CONSTRAINT(-777) is used.
        
        Note: from FactSet documentation:
        The first row is used for unknown industry, which is assigned to securities that has no industry classification.
        This entry prevents classifying unknown industry into known ones or assigning an industry to a currency.
    """
    outFile.write('%s,\"No Classification Available",0,%s,%s\r\n' 
                  %(NOT_AVAILABLE, VALUE_NO_CONSTRAINT, VALUE_NO_CONSTRAINT))
    for ind in industries:
        #FactSet format: factorId, factorName, penalty, minCons, maxCons
        outFile.write('%s,\"%s\",0,%s,%s\r\n'
                      % (industryFactors[ind.name], ind.description, 
                         VALUE_NO_CONSTRAINT, VALUE_NO_CONSTRAINT))

def writeDailyFactorReturns(d, riskModel, factorIDs, modelDB, expM, outFile):
    """ Write to FactSet .RET file, which corresponds to the returns for model factors used in performance attribution.
        The columns are in the following order:
        (1) Factor ID
        (2) Factor Short Name
        (3) Factor Name
        (4) Factor Return
            The FactSet documentation says that the numbers should be percents and represent monthly returns. 
            However, it makes more sense that we are providing daily returns in percents due to our update frequency(daily). 
        
        The factors are in the same order as the coefficients/values of factors in .SEC, .MDL, .COR files.
        
    """
    #a pair of lists with factor returns and names
    factorReturns = riskModel.loadFactorReturns(d, modelDB)
    factorReturnMap = dict(zip(factorReturns[1], factorReturns[0]))
    
    for factor in riskModel.factors:
        factorName = factor.name
        factorId = factorIDs[factorName]
        outFile.write('%s,\"%s\",\"%s\",%.4f\r\n'
                         % (factorId, factorName,factorName,
                            factorReturnMap[factor]*100.0))
    

def writeDaily(options, d, mnemonic, riskModelInstance, riskModel, modelDB, marketDB):
    """To write FactSet flat files daily with the names like Axioma_<ModelMnemonic-numeraireISO>_<yyyyMMdd>.<fileType[SEC,MDL,COR,IND,RET]>
       (1)*.SEC  - Security Data File
       (2)*.MDL  - Model/Factor Variances File
       (3)*.COR  - Correlation/Covariance Matrix file 
       (4)*.IND  - Industry file
       (5)*.RET  - Factor Returns file
    """

    expM = riskModel.loadExposureMatrix(riskModelInstance, modelDB)
    if expM == None:
        modelDB.log.error('No exposure data available for  %s(%s). Exit.' 
                          % (riskModel.mnemonic, riskModel.name))
        sys.exit(1)
    
    svDataDict = riskModel.loadSpecificRisks(riskModelInstance, modelDB)
    if svDataDict == None:
        modelDB.log.error('No specific risk data available for  %s(%s). Exit.' 
                          % (riskModel.mnemonic, riskModel.name))
        sys.exit(1)
        
    #Group the factors into distinctive types and create the factor ids like [factorType][#sequence]
    factorIDs = labelFactorsByType(expM, riskModel.factors)
    
    #file prefix
    prefix='Axioma_' + mnemonic + '-' + riskModel.numeraire.currency_code
    
    #write the industry file, return the industries and their codes
    outFileName = '%s/%s_%04d%02d%02d.IND' % (options.targetDir, prefix, d.year, d.month, d.day)
    outFile = open(outFileName, 'w')
    writeIndustryFile(d, riskModel.industries, factorIDs, outFile)
    outFile.close()
    modelDB.log.debug('The FactSet industry file is written to %s.' % outFileName)
    
    factorCov = riskModel.loadFactorCovarianceMatrix(riskModelInstance, modelDB)[0]
    #write the correlation matrix to a file and return factor variances vector
    outFileName = '%s/%s_%04d%02d%02d.COR' % (options.targetDir, prefix, d.year, d.month, d.day)
    outFile = open(outFileName, 'w')
    factorVariances = writeCorrelationMatrixFile(d, factorCov, riskModel.factors, outFile)
    outFile.close()
    modelDB.log.debug('The FactSet correlation file is written to %s.' % outFileName)
    
     # write model file with each factor and its variance
    outFileName = '%s/%s_%04d%02d%02d.MDL' % (options.targetDir, prefix, d.year, d.month, d.day)
    outFile = open(outFileName, 'w')
    writeModelFile(d, modelDB, expM, factorVariances, riskModel.factors, factorIDs, outFile)
    outFile.close()
    modelDB.log.debug('The FactSet model file is written to %s.' % outFileName)
    
    # write factor returns
    outFileName = '%s/%s_%04d%02d%02d.RET' % (options.targetDir, prefix, d.year, d.month, d.day)
    outFile = open(outFileName, 'w')
    writeDailyFactorReturns(d, riskModel, factorIDs, modelDB, expM, outFile)
    outFile.close()
    modelDB.log.debug('The FactSet factor return file is written to %s.' % outFileName)
    
    # Write security data file: cusip, name, industry,price,specific(residue)risk,shares outstanding, exposure to each factor
    outFileName = '%s/%s_%04d%02d%02d.SEC' % (options.targetDir, prefix, d.year, d.month, d.day)
    outFile = open(outFileName, 'w')
    writeSecurityDataFile(d, marketDB, modelDB, expM, svDataDict, riskModel, factorIDs, outFile)
    outFile.close()
    modelDB.log.debug('The FactSet security data file is written to %s.' % outFileName)
                       

if __name__ == '__main__':
    import sys

    usage = "usage: %prog [options] <startdate or datelist> <end-date>"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addModelAndDefaultCommandLine(cmdlineParser)
    cmdlineParser.add_option("-d", "--directory", action="store",
                             default='.', dest="targetDir",
                             help="directory for output files")
    cmdlineParser.add_option("--model-only", action="store_true",
                             default=False, dest="modelDataOnly",
                             help="only extract model (cov, exp) files")
    cmdlineParser.add_option("--version", action="store",
                             default=3, type='int', dest="formatVersion",
                             help="version of flat files to create")

    (options, args) = cmdlineParser.parse_args()
    if len(args) < 1 or len(args) > 2:
        cmdlineParser.error("Incorrect number of arguments")
    
    modelClass = Utilities.processModelAndDefaultCommandLine(options, cmdlineParser)
    modelDB = ModelDB.ModelDB(sid=options.modelDBSID, user=options.modelDBUser,passwd=options.modelDBPasswd)
    modelDB.setTotalReturnCache(150)
    modelDB.setMarketCapCache(50)
    marketDB = MarketDB.MarketDB(sid=options.marketDBSID, user=options.marketDBUser, passwd=options.marketDBPasswd)
    riskModel = modelClass(modelDB, marketDB)
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
    
    for d in dates:
        riskModel.setFactorsForDate(d, modelDB)
        rmi = riskModel.getRiskModelInstance(d, modelDB)
        if rmi != None and rmi.has_risks:
            logging.info('Processing %s(%s) on %s' % (riskModel.name, riskModel.mnemonic, str(d)))
            #write to the factset-format flat files daily
            writeDaily(options, d, riskModel.mnemonic, rmi, riskModel, modelDB, marketDB)
        else:
            logging.error('No risk model instance on %s' % str(d))
    
    modelDB.finalize()
    marketDB.finalize()
