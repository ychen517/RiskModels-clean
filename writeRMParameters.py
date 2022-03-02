
import optparse
import inspect
import re
import json
import pandas as pd
pandas_version = int(pd.__version__.split('.')[1])

import riskmodels
#import FixedIncomeModels
from riskmodels import PhoenixModels
from riskmodels import Utilities
from riskmodels import ModelDB
from marketdb import MarketDB

if __name__ == '__main__':
    usage = "usage: %prog [options]"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addModelAndDefaultCommandLine(cmdlineParser)
    cmdlineParser.add_option("-d", "--directory", action="store",
                             default='.', dest="targetDir",
                             help="directory for output files")
    cmdlineParser.add_option("--no-json", action="store_false",
                             default=True, dest="writeJson",
                             help="write to Json dump file")
    cmdlineParser.add_option("-c", action="store_true",
                             default=False, dest="writeCSV",
                             help="write to CSV ")
    cmdlineParser.add_option("-u", "--update-database", action="store_true",
                             default=False, dest="updateDB",
                             help="write to DB ")
    (options, args) = cmdlineParser.parse_args()
    if (options.updateDB or options.writeCSV) and pandas_version < 15:
        import sys
        print('Error:  Need to run with pandas version >= 0.15.  Cannot write to CSV or update database')
        sys.exit(0)


    modelDB = ModelDB.ModelDB(sid=options.modelDBSID, 
        user=options.modelDBUser, passwd=options.modelDBPasswd)
    marketDB = MarketDB.MarketDB(sid=options.marketDBSID, 
        user=options.marketDBUser, passwd=options.marketDBPasswd)

    types = ['Fundamental', 
             'Statistical', 
             'Macroeconomic', 
             'Fixed Income', 
             'Commodity Futures']

    emptyDict = {'halfLife': None,
                 'minObs': None,
                 'maxObs': None,
                 'nwLags': None }

    if options.writeCSV:
        rmfp = open('riskmodelattrs.csv', 'w')

    modelids=[]
    modelattrs={}

    # Get class names
    classNames = []
    modules = [riskmodels.RiskModels, riskmodels.RiskModels_V1, riskmodels.RiskModels_V2, riskmodels.RiskModels_V3]
    for mod in modules:
        for name, obj in inspect.getmembers(mod):
            if inspect.isclass(obj) and '_Pre' not in name and \
                    bool(re.search(r'\d{4}', name)) and len(name) >= 12:
                if name.startswith('FX') or name.endswith('FL'):
                    continue
                classNames.append(name)
    classNames.append('COAxioma2013MH')

    for cl in classNames:
        print(cl)
        riskModelClass = riskmodels.getModelByName(cl)
        try:
            riskModel = riskModelClass(modelDB, marketDB)
        except:
            continue
        jsonDict = {}
        #jsonDict['allowETFs'] = riskModel.allowETFs 
        jsonDict['description'] = riskModel.description
        #jsonDict['estuAssetTypes'] = riskModel.estuAssetTypes # list
        jsonDict['mnemonic'] = riskModel.mnemonic
        jsonDict['name'] = riskModel.name
        jsonDict['id'] = riskModel.rm_id

        rmtype = ''
        for ty in types:
            if ty in riskModel.description:
                rmtype = ty 
        jsonDict['type'] = rmtype

        rmnumeraire = ''
        if hasattr(riskModel, 'numeraire'):
            rmnumeraire = riskModel.numeraire.currency_code
        jsonDict['numeraire'] = rmnumeraire

        # risk calculators
        if hasattr(riskModel, 'specificRiskCalculator'):
            specRisk = {}
            specRisk['halfLife'] = riskModel.specificRiskCalculator.deltaHalfLife
            specRisk['minObs'] = riskModel.specificRiskCalculator.minDeltaObs
            specRisk['maxObs'] = riskModel.specificRiskCalculator.maxDeltaObs
            specRisk['nwLags'] = riskModel.specificRiskCalculator.deltaLag
        else:
            specRisk = emptyDict
        jsonDict['specificRiskCalculator'] = specRisk
        if hasattr(riskModel, 'covarianceCalculator'):
            factorRiskCov = {}
            if hasattr(riskModel.covarianceCalculator, 'varParameters'):
                covCalculator = riskModel.covarianceCalculator.varParameters
                if hasattr(covCalculator, 'getCovarianceHalfLife'):
                    factorRiskCov['halfLife'] = covCalculator.getCovarianceHalfLife()
                elif hasattr(covCalculator, 'halfLife'):
                    factorRiskCov['halfLife'] = covCalculator.halfLife
                else:
                    factorRiskCov['halfLife'] = None
                if hasattr(covCalculator, 'getCovarianceSampleSize'):
                    factorRiskCov['minObs'] = covCalculator.getCovarianceSampleSize()[0]
                    factorRiskCov['maxObs'] = covCalculator.getCovarianceSampleSize()[1]
                elif hasattr(covCalculator, 'maxObs') and \
                     hasattr(covCalculator, 'minObs'):
                    factorRiskCov['minObs'] = covCalculator.minObs
                    factorRiskCov['maxObs'] = covCalculator.maxObs
                #elif hasattr(covCalculator, 'deMeanMaxHistoryLength') and \
                #     hasattr(covCalculator, 'deMeanMinHistoryLength'):
                #    factorRiskCov['minObs'] = covCalculator.deMeanMinHistoryLength
                #    factorRiskCov['maxObs'] = covCalculator.deMeanMaxHistoryLength
                else:
                    factorRiskCov['minObs'] = None
                    factorRiskCov['maxObs'] = None
                if hasattr(covCalculator, 'getCovarianceNeweyWestLag'):
                    factorRiskCov['nwLags'] = covCalculator.getCovarianceNeweyWestLag()
                elif hasattr(covCalculator, 'NWLag'):
                    factorRiskCov['nwLags'] = covCalculator.NWLag
                else:
                    factorRiskCov['nwLags'] = None
            if len(factorRiskCov) == 0:
                factorRiskCov = emptyDict

            factorRiskCorr = {}
            if hasattr(riskModel.covarianceCalculator, 'corrParameters'):
                corrCalculator = riskModel.covarianceCalculator.corrParameters
                if hasattr(corrCalculator, 'getCovarianceHalfLife'):
                    factorRiskCorr['halfLife'] = corrCalculator.getCovarianceHalfLife()
                elif hasattr(corrCalculator, 'halfLife'):
                    factorRiskCorr['halfLife'] = corrCalculator.halfLife
                else:
                    factorRiskCorr['halfLife'] = None
                if hasattr(corrCalculator, 'getCovarianceSampleSize'):
                    factorRiskCorr['minObs'] = corrCalculator.getCovarianceSampleSize()[0]
                    factorRiskCorr['maxObs'] = corrCalculator.getCovarianceSampleSize()[1]
                elif hasattr(corrCalculator, 'minObs') and hasattr(corrCalculator, 'maxObs'):
                    factorRiskCorr['minObs'] = corrCalculator.minObs
                    factorRiskCorr['maxObs'] = corrCalculator.maxObs
                else:
                    factorRiskCorr['minObs'] = None
                    factorRiskCorr['maxObs'] = None
                if hasattr(corrCalculator, 'getCovarianceNeweyWestLag'):
                    factorRiskCorr['nwLags'] = corrCalculator.getCovarianceNeweyWestLag()
                elif hasattr(corrCalculator, 'NWLag'):
                    factorRiskCorr['nwLags'] = corrCalculator.NWLag
                else:
                    factorRiskCorr['nwLags'] = None
            if len(factorRiskCorr) == 0:
                factorRiskCorr = emptyDict
        else:
            factorRiskCov = emptyDict
            factorRiskCorr = emptyDict
        jsonDict['factorRiskCovariance'] = factorRiskCov
        jsonDict['factorRiskCorrelation'] = factorRiskCorr

        if options.writeJson:
            with open('%s/%s.json' % (options.targetDir, riskModel.name), 'w') as fp:
                json.dump(jsonDict, fp) 

        name=jsonDict['id']
        if pandas_version >= 15:
            modelattrs[name] = []
            df=pd.io.json.json_normalize(jsonDict)
            print('working on ', name, jsonDict['name'])
            if pandas_version >= 15:
                modelids.append(name)
                for n, v in zip(df.columns.tolist(), df.values.tolist()[0]):
                    if n=='id':
                        continue
                    if options.writeCSV:
                        rmfp.write("%s,%s,%s\n" %  (name, n,v))
                    modelattrs[name].append([name,n,v])


    if options.writeCSV:
        rmfp.close()

    if options.updateDB and pandas_version >= 15:
        cursor = modelDB.dbCursor
        for modelid in modelids:
            query="""delete risk_model_attributes where model_id=:id""" 
            cursor.execute(query, id=modelid)
            #print 'Deleting risk_model_attributes for %d' % modelid
            rc=0 
            for mid, n, v in modelattrs[modelid]:
                query="""insert into risk_model_attributes (model_id, name, value) values (:id, :name, :value)"""
                cursor.execute(query, id=modelid, name=n, value=v)
                rc += cursor.rowcount

            print('Inserted/Replaced %d risk_model_attributes rows for  %d' % ( rc, modelid))
        modelDB.commitChanges()
        modelDB.finalize()

