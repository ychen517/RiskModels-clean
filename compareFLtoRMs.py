
import logging
import numpy
import pandas as pd
import optparse
from marketdb import MarketDB
from riskmodels import ModelDB
import riskmodels
from riskmodels import Utilities



class CheckAndParseArgumentsOptions(object):
    """ This is to check arguments and options are well set. If not, throw warnings and errors"""

    def __init__(self, name):
        logging.info("class CheckAndParseArgumentsOptions initialized")
        self.name = name
        self.LengthofArguments = 1

    def CheckAndParseArguments(self, ArgumentValue = None):
        """This is to check and parse arguments like YYYY-MM-DD:YYYY-MM-DD """
        assert ArgumentValue != None, "Empty Argument not allowed, pls add date";
        assert len(ArgumentValue) == self.LengthofArguments, "Incorrect length of arguments, pls use YYYY-MM-DD or YYYY-MM-DD:YYYY-MM-DD";
        fields = ArgumentValue[0].split(':')
        if len(fields) == 1:
            fields = fields*2
        return [Utilities.parseISODate(x) for x in fields]

    def CheckOptions(self, options):
        assert options.FLModelName != None, "Empty --FLModel not allowed";
        assert ',' not in options.FLModelName, "--FLModel could only take one model name. It is not allowed to give a list of Factor Library Names";
        assert options.FDModelName != None, "Empty --FDModel not allowed";
        assert options.sections != None, "Empty --sections not allowed";
        #assert options.MHModelName != None, "Empty --MHModel not allowed";
        #assert options.SHModelName != None, "Empty --SHModel not allowed";
        #assert options.reportFile != None, "Empty --report-file not allowed";

    def ParseModelNames(self, FDModelName = None):
        assert FDModelName != None, "Empty --FDModel not allowed"
        modelNameList = FDModelName.split(',')
        #modelNameListPrefix = set([x[:-2] for x in modelNameList]);
        #if len(modelNameListPrefix) == 1:
        #    logging.warning("The Models (%s) you Specified might not be able to be consolidated together, as they have different structures" %(modelName));
        return modelNameList

    def ParseSections(self, sections=None):
        assert options.sections != None, "Empty --sections not allowed";
        sectionsList = sections.split(',')
        return sectionsList


class WWAxioma2017FL_Validator(object):
    def __init__(self, rmFL, rmFD,  date):
        logging.info('------- In %s -------' %(self.__class__))
        self.date = date
        self.PrevDate = modelDB.getDates(rmFL.rmg, date, daysBack =1)[0]
        #self.modelPath = modelPath;
        self.rmFL = rmFL
        self.rmFD = rmFD
        self.rmiFL = modelDB.getRiskModelInstance(rmFL.rms_id,date)
        self.rmiFD = modelDB.getRiskModelInstance(rmFD.rms_id,date)
        self.rmnFL = self.rmFL.__class__.__name__
        self.rmnFD = self.rmFD.__class__.__name__
        #self.rmiSH = modelDB.getRiskModelInstance(rmSH.rms_id,date);
        self.fdDescDict = None
        self.flDescDict = None
        self.expFL = None
        self.expFD = None

    def ModelUniCoverage(self):
        """This function checks equality between two Model Universes between a factor library and a funcamental risk model."""
        logging.info('------- def %s%s -------' %(self.ModelUniCoverage.__name__, '()'))
        logging.debug('Getting Model Universe of %s and %s on date %s' %(self.rmnFL, self.rmnFD, self.date) )
        univFL = set(modelDB.getRiskModelInstanceUniverse(self.rmiFL,False))

        univFD = set(modelDB.getRiskModelInstanceUniverse(self.rmiFD,False))

        modelCoverage_report = pd.DataFrame({
                'Model Name' : [self.rmnFL, self.rmnFD],
                'Model Universe' : [len(univFL), len(univFD)] 
                })
        logging.info('Reporting the count of assets in model universe: \n %s' %(modelCoverage_report))
        #Check if model Universe count match
        Univ_FL_minus_FD = univFL - univFD
        Univ_FD_minus_FL = univFD - univFL
        assert len(Univ_FL_minus_FD) == 0 and len(Univ_FD_minus_FL) == 0 , "%s model universe != %s  model universe. \n %s in %s Model Universe but not in %s Model Universe. \n %s in %s Model Universe but not in %s Model Universe" %(self.rmnFL, self.rmnFD, Univ_FL_minus_FD, self.rmnFL, self.rmnFD, Univ_FD_minus_FL, self.rmnFD, self.rmnFL)


    def ESTUCoverage(self):
        """ This function checks equality between two ESTU between a factor library and a funcamental risk model. """
        logging.info('------- def %s%s -------' %(self.ESTUCoverage.__name__, '()'))
        logging.debug('Getting Estimation Universe of %s and %s on date %s' %(self.rmnFL, self.rmnFD, self.date) )
        self.rmiFL = modelDB.getRiskModelInstance(rmFL.rms_id,date)
        print(self.rmiFL)
        estuFL = set(modelDB.getRiskModelInstanceESTU(self.rmiFL))
        estuFD = set(modelDB.getRiskModelInstanceESTU(self.rmiFD))
        modelCoverage_report = pd.DataFrame({
                    'Model Name' : [self.rmnFL, self.rmnFD],
                    'ESTU Universe' : [len(estuFL), len(estuFD)],
                    })
        #Sorting DataFrame
        modelCoverage_report = modelCoverage_report[['Model Name', 'ESTU Universe']]
        logging.info('Reporting the count of assets in estimation universe: \n %s' %(modelCoverage_report))
        #if modelCoverage_report['ESTU Universe'].loc[0] != modelCoverage_report['ESTU Universe'].loc[1]:
        Univ_FL_minus_FD = estuFL - estuFD
        Univ_FD_minus_FL = estuFD - estuFL
        assert len(Univ_FL_minus_FD) == 0 and len(Univ_FD_minus_FL) == 0, "%s ESTU != %s  ESTU. \n %s in %s ESTU but not in %s ESTU. \n %s in %s ESTU but not in %s ESTU" %(self.rmnFL, self.rmnFD, Univ_FL_minus_FD, self.rmnFL, self.rmnFD, Univ_FD_minus_FL, self.rmnFD, self.rmnFL)


    def createDescpMaprmFL(self, rmFL):
        factor_lst = []
        descriptor_lst = []
        for factor, descriptors in rmFL.DescriptorMap.items():
            assert(len(descriptors)==1)
            factor_lst.append(factor)
            descriptor_lst.extend(descriptors)
        flDescDict = pd.DataFrame(list(zip(factor_lst, descriptor_lst)), columns = ['factorFL', 'descriptorFL'])
        return flDescDict

    def createCompositeDescpMaprmFD(self, rmFD):
        factor_lst = []
        descriptor_lst = []
        for factor, descriptors in rmFD.DescriptorMap.items():
            if len(descriptors) > 1:
                factor_lst.extend([factor]*len(descriptors))
                descriptor_lst.extend(descriptors)
        fdDescDict = pd.DataFrame(list(zip(factor_lst, descriptor_lst)), columns = ['factorFD', 'descriptorFD'])
        return fdDescDict

    def createSingleDescpMaprmFD(self, rmFD):
        factor_lst = []
        descriptor_lst = []
        logging.debug('%s.regionalStndList : %s' %(self.rmnFD, rmFD.regionalStndList))
        logging.debug('%s.orthogList : %s' %(self.rmnFD, rmFD.orthogList))
        for factor, descriptors in rmFD.DescriptorMap.items():
            if len(descriptors) == 1 and factor not in rmFD.regionalStndList and factor not in rmFD.orthogList.keys():
                factor_lst.append(factor)
                descriptor_lst.extend(descriptors)
        fdDescDict = pd.DataFrame(list(zip(factor_lst, descriptor_lst)), columns = ['factorFD', 'descriptorFD'])
        return fdDescDict


    def SingleDescpCheck(self, threshold=0.98):
        """ This function checks correlation between single-descriptor-factors (factor derived by only one descriptor) and its descriptor. 
        If the correlation is higher than 0.98, pass. If not, fail with assertion errors"""
        logging.info('------- def %s%s -------' %(self.SingleDescpCheck.__name__, '()'))
        self.flDescDict = self.createDescpMaprmFL(self.rmFL)
        self.fdDescDict = self.createSingleDescpMaprmFD(self.rmFD)
        logging.debug('Descriptor to factor map for %s: \n %s' %(self.rmnFL, self.flDescDict))
        logging.debug('Descriptor to factor map for %s: \n %s' %(self.rmnFD, self.fdDescDict))

        # load exposure matrices
        if self.expFL is None:
            self.expFL = self.loadExpMatrixAsDF(self.date, self.rmFL, modelDB)
        else:
            logging.info('exposure matrix exits. Using the existing exposure matrix map.' %(self.rmnFL))
        self.expFD = self.loadExpMatrixAsDF(self.date, self.rmFD, modelDB)

        fdflDescDict = self.fdDescDict.merge(self.flDescDict, left_on='descriptorFD', right_on='descriptorFL', how='inner')
        logging.debug('Joined descriptor to factor map for %s and %s: \n %s' %(self.rmnFL, self.rmnFD, fdflDescDict))

        for factorFD in fdflDescDict['factorFD']:
            if factorFD == 'Value':
                continue
            logging.info("factor %s in %s" %(factorFD, self.rmnFD))
            factorFL = fdflDescDict[fdflDescDict['factorFD'] == factorFD]['factorFL'].tolist()
            logging.info("factor %s in %s" %(factorFL, self.rmnFL))
            flVals = self.expFL[factorFL].dropna()
            fundVals = self.expFD[factorFD].dropna()
            fundVals = fundVals.to_frame()

            merged_df = fundVals.merge(flVals, left_index=True, right_index=True)
            merged_df.columns = ['fundVals', 'flVals']
            corr = numpy.corrcoef(merged_df['fundVals'], merged_df['flVals'])
            logging.info('correlation between %s and [%s] = %s' %(factorFL, factorFD, corr[0][1]))
            assert(corr[0][1] > threshold), "correlation between %s in %s and  Factor [%s] in %s = %s is lower than %s" %(factorFL, self.rmnFL, factorFD, self.rmnFD, corr[0][1], threshold)


    def CompositeDescpCheck(self, threshold=0.9):
        """ This function is currently not in use. For multi descriptor factors, correlation between factor and descriptor varies."""
        #threshold = 0.9;
        logging.info('------- def %s%s -------' %(self.CompositeDescpCheck.__name__, '()'))
        logging.info('threshold = %s' %(threshold))
        logging.debug('Getting descriptor to factor map for %s' %(self.rmnFL))
        self.flDescDict = self.createDescpMaprmFL(self.rmFL)
        logging.debug('Getting descriptor to factor map for %s' %(self.rmnFD))
        self.fdDescDict = self.createCompositeDescpMaprmFD(self.rmFD)
        logging.debug('Descriptor to factor map for %s: \n %s' %(self.rmnFL, self.flDescDict))
        logging.debug('Descriptor to factor map for %s: \n %s' %(self.rmnFD, self.fdDescDict))
        # load exposure matrices
        if self.expFL is None:
            logging.debug('Getting descriptor to factor map for %s' %(self.rmnFL))
            self.expFL = self.loadExpMatrixAsDF(self.date, self.rmFL, modelDB)
        else:
            logging.info('exposure matrix exits. Using the existing exposure matrix map.' %(self.rmnFL))
        self.expFD = self.loadExpMatrixAsDF(self.date, self.rmFD, modelDB)

        fdflDescDict = self.fdDescDict.merge(self.flDescDict, left_on='descriptorFD', right_on='descriptorFL', how='inner')
        logging.debug('Joined descriptor to factor map for %s and %s: \n %s' %(self.rmnFL, self.rmnFD, fdflDescDict))
        for factorFD in fdflDescDict['factorFD'].unique():
            logging.debug("factor %s in %s" %(factorFD, self.rmnFD))
            factorFL = fdflDescDict[fdflDescDict['factorFD'] == factorFD]['factorFL'].tolist()
            logging.debug("factor %s in %s" %(factorFL, self.rmnFL))
            flVals = self.expFL[factorFL].dropna()
            fundVals = self.expFD[factorFD].dropna()
            fundVals = fundVals.to_frame()
            merged_df = fundVals.merge(flVals, left_index=True, right_index=True)
            merged_df_corr = merged_df.corr(method='pearson', min_periods=1)
            logging.info('Reporting the correlation matrix: \n %s' %(merged_df_corr))
            #To flatten the corr matrix into a dataframe with three columns: index, columns, values
            lst_index = []
            lst_column = []
            lst_value = []
            for index, row in merged_df_corr.iterrows():
                lst_index.extend([index]*row.shape[0])
                lst_column.extend(row.index)
                lst_value.extend(row.values)
            merged_df_corr = pd.DataFrame(list(zip(lst_index, lst_column, lst_value)),columns = ['idx', 'col', 'value'])
            corr_outlier = merged_df_corr[(merged_df_corr['idx'] != merged_df_corr['col']) & (abs(merged_df_corr['value']) < threshold)]
            assert(corr_outlier.shape[0]==0), "Correlation between the following descriptors in %s and factors in %s are lower than our threshold %s. \n %s" %(self.rmnFL, self.rmnFD, threshold,  corr_outlier)


    def loadExpMatrixAsDF(self, dt, rm, modelDB):
        rm.setFactorsForDate(dt, modelDB)
        rmi = rm.getRiskModelInstance(dt, modelDB)
        if rmi is not None:
            expM = rm.loadExposureMatrix(rmi, modelDB).toDataFrame()
            return expM

class USAxioma2016FL_Validator(WWAxioma2017FL_Validator):
    def __init__(self, rmFL, rmFD,  date):
        logging.info('------- In %s -------' %(self.__class__))
        WWAxioma2017FL_Validator.__init__(self, rmFL, rmFD, date)
        self.factors_for_equal_checks = ['Dividend Yield', 'Size', 'Liquidity', 'Medium-Term Momentum', 'Market Sensitivity', 'Exchange Rate Sensitivity']#factors in this list will have equality checks in def SingleDescpCheck_USAxioma2016FL, otherwise factors will have correlation checks.

    def SingleDescpCheck_USAxioma2016FL(self, threshold=0.98):
        """ This function is currently not in use.
        This is different from def SingleDescpCheck(), because there are some single descriptor factors they are expected to be identical to its descriptors. 
        These factors will be stored in self.factors_for_equal_checks.
        Specifically, we only do equality checks for factors in self.factors_for_cor_checks, while for factors not in self.factors_for_cor_checks, we still do correlation checks."""
        logging.info('------- def %s%s -------' %(self.SingleDescpCheck_USAxioma2016FL.__name__, '()'))
        logging.info('Getting descriptor to factor map for %s' %(self.rmnFL))
        self.flDescDict = self.createDescpMaprmFL(self.rmFL)
        logging.info('Getting descriptor to factor map for %s' %(self.rmnFD))
        self.fdDescDict = self.createSingleDescpMaprmFD(self.rmFD)

        # load exposure matrices
        if self.expFL is None:
            logging.info('Getting descriptor to factor map for %s' %(self.rmnFL))
            self.expFL = self.loadExpMatrixAsDF(self.date, self.rmFL, modelDB)
        else:
            logging.info('Descriptor to factor map already exist. Using the existing descriptor to factor map.' %(self.rmnFL))
        logging.info('Getting descriptor to factor map for %s' %(self.rmnFD))
        self.expFD = self.loadExpMatrixAsDF(self.date, self.rmFD, modelDB)

        fdflDescDict = self.fdDescDict.merge(self.flDescDict, left_on='descriptorFD', right_on='descriptorFL', how='inner')
        for factorFD in fdflDescDict['factorFD']:
            logging.info("factor %s in %s" %(factorFD, self.rmnFD))
            factorFL = fdflDescDict[fdflDescDict['factorFD'] == factorFD]['factorFL'].tolist()
            logging.info("factor %s in %s" %(factorFL, self.rmnFL))
            flVals = self.expFL[factorFL].dropna()
            fundVals = self.expFD[factorFD].dropna()
            fundVals = fundVals.to_frame()
            merged_df = fundVals.merge(flVals, left_index=True, right_index=True)
            merged_df.columns = ['fundVals', 'flVals']
            if factorFD not in self.factors_for_equal_checks:
                corr = numpy.corrcoef(merged_df['fundVals'], merged_df['flVals'])
                logging.info('correlation between %s and [%s] = %s' %(factorFL, factorFD, corr[0][1]))
                assert(corr[0][1] > threshold), "correlation between %s in %s and  Factor [%s] in %s = %s is lower than %s" %(factorFL, self.rmnFL, factorFD, self.rmnFD, corr[0][1], threshold)
            else:
                mismatch_records = merged_df[merged_df['fundVals'] != merged_df['flVals']]
                logging.info('%s in %s and [%s] in %s  are identical' %(factorFL, self.rmnFL,  factorFD, self.rmnFD))
                assert(mismatch_records.shape[0] == 0), "factor/Descriptor %s in %s should be equal to Factor [%s] in %s, but they are different for the following assets: %s" %(factorFL, self.rmnFL, factorFD, self.rmnFD, mismatch_records)

class EUAxioma2017FL_Validator(WWAxioma2017FL_Validator):
    def __init__(self, rmFL, rmFD,  date):
        logging.info('------- In %s -------' %(self.__class__))
        WWAxioma2017FL_Validator.__init__(self, rmFL, rmFD, date)


class EMAxioma2018FL_Validator(WWAxioma2017FL_Validator):
    def __init__(self, rmFL, rmFD,  date):
        logging.info('------- In %s -------' %(self.__class__))
        WWAxioma2017FL_Validator.__init__(self, rmFL, rmFD, date)


class CNAxioma2018FL_Validator(WWAxioma2017FL_Validator):
    def __init__(self, rmFL, rmFD,  date):
        logging.info('------- In %s -------' %(self.__class__))
        WWAxioma2017FL_Validator.__init__(self, rmFL, rmFD, date)


class APAxioma2018FL_Validator(WWAxioma2017FL_Validator):
    def __init__(self, rmFL, rmFD,  date):
        logging.info('------- In %s -------' %(self.__class__))
        WWAxioma2017FL_Validator.__init__(self, rmFL, rmFD, date)


class APxJPAxioma2018FL_Validator(WWAxioma2017FL_Validator):
    def __init__(self, rmFL, rmFD,  date):
        logging.info('------- In %s -------' %(self.__class__))
        WWAxioma2017FL_Validator.__init__(self, rmFL, rmFD, date)

class UKAxioma2018FL_Validator(WWAxioma2017FL_Validator):
    def __init__(self, rmFL, rmFD,  date):
        logging.info('------- In %s -------' %(self.__class__))
        WWAxioma2017FL_Validator.__init__(self, rmFL, rmFD, date)

class CAAxioma2018FL_Validator(WWAxioma2017FL_Validator):
    def __init__(self, rmFL, rmFD,  date):
        logging.info('------- In %s -------' %(self.__class__))
        WWAxioma2017FL_Validator.__init__(self, rmFL, rmFD, date)


if __name__ == '__main__':
    usage = "usage: %prog [options] <YYYY-MM-DD> [<YYYY-MM-DD>]"
    cmdlineParser = optparse.OptionParser(usage=usage)
    cmdlineParser.add_option('--store-report', action='store_true',
                             default=False, dest='storeReport',
                             help='write the report to a text file')
    cmdlineParser.add_option("--FLModel", action="store",
                            default=None,
                            dest="FLModelName",
                            help="Factor Library Name which you want to QA")
    cmdlineParser.add_option("--FDModel", action="store",
                             default=None,
                             dest="FDModelName",
                             help="A comma separated list of fundamental Risk Model which you want to compare with the Factor Library specified in --FLModel option.")
    cmdlineParser.add_option("--sections", action="store",
                            default=None, dest="sections",
                            help="sections you'd like to run. It should be a string of sections separated by comma. For example: --sections ModelUniCoverage,ESTUCoverage,SingleDescpCheck,CompositeDescpCheck")
    
    Utilities.addModelAndDefaultCommandLine(cmdlineParser) #This is to add ModelName, Model ID and Model version options and logging config
    (options, args) = cmdlineParser.parse_args()
    Utilities.processDefaultCommandLine(options, cmdlineParser)#Config the logging

    """ check options and args here. If with incorrect length of arguments, raise an error"""
    ArgumentsOptions_checker = CheckAndParseArgumentsOptions('ArgumentsOptions_checker')
    StartDTEndDT = ArgumentsOptions_checker.CheckAndParseArguments(args)
    ArgumentsOptions_checker.CheckOptions(options)
    SectionNameList = ArgumentsOptions_checker.ParseSections(options.sections)
    ModelNameList = ArgumentsOptions_checker.ParseModelNames(options.FDModelName)
    
    modelDB = ModelDB.ModelDB(sid=options.modelDBSID, user=options.modelDBUser, passwd=options.modelDBPasswd)
    marketDB = MarketDB.MarketDB(sid=options.marketDBSID, user=options.marketDBUser, passwd=options.marketDBPasswd)
    rmcFL = riskmodels.getModelByName(options.FLModelName)
    rmFL = rmcFL(modelDB, marketDB)
    #validator_class_name = options.FLModelName+'_Validator'
    dates = modelDB.getDateRange(rmFL.rmg, StartDTEndDT[0], StartDTEndDT[1], excludeWeekend=True)
    if len(dates) == 0:
        logging.warning('dates %s given are weekends or not traded for %s' %(StartDTEndDT, rmFL))
    for date in dates:
        logging.info('-------------------------------- date:%s --------------------------------' %(date))
        for rmFD in ModelNameList:
            rmcFD = riskmodels.getModelByName(rmFD)
            rmFD = rmcFD(modelDB, marketDB)
            cmd = options.FLModelName+'_Validator(rmFL, rmFD, date)'#eg: FLValidator_ = WWAxioma2017FL_Validator(rmFL, rmFD, date);
            logging.debug('-----------------------eval %s -----------------------' %(cmd))
            FLValidator_ = eval(cmd)
            for section in SectionNameList:
                cmd_section = 'FLValidator_.'+section+'()'#eg. FLValidator_.ModelUniCoverage()
                logging.debug('------------- eval %s --------------' %(cmd_section))
                eval(cmd_section)

