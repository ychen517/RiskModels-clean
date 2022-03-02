
import logging
import datetime
from dateutil.relativedelta import relativedelta
import numpy.ma as ma

import riskmodels
from riskmodels import FactorLibrary
from riskmodels.AxiomaFactorLibraryConstants import AXIOMA_FACTOR_NAMES, TRANSLATED_AXIOMA_FACTOR_NAMES, AXIOMA_WW_FACTOR_NAMES, TRANSLATED_AXIOMA_WW_FACTOR_NAMES
from riskmodels.AxiomaFactorLibraryConstants import AXIOMA_US4_FACTOR_NAMES, TRANSLATED_AXIOMA_US4_FACTOR_NAMES
from riskmodels.AxiomaFactorLibraryConstants import AXIOMA_WWEL_FACTOR_NAMES, TRANSLATED_AXIOMA_WWEL_FACTOR_NAMES 
from riskmodels.AxiomaFactorLibraryConstants import AXIOMA_EU4_FACTOR_NAMES, TRANSLATED_AXIOMA_EU4_FACTOR_NAMES
from riskmodels.AxiomaFactorLibraryConstants import AXIOMA_EM4_FACTOR_NAMES, TRANSLATED_AXIOMA_EM4_FACTOR_NAMES
from riskmodels.AxiomaFactorLibraryConstants import AXIOMA_CN4_FACTOR_NAMES, TRANSLATED_AXIOMA_CN4_FACTOR_NAMES
from riskmodels.AxiomaFactorLibraryConstants import AXIOMA_AP4_FACTOR_NAMES, TRANSLATED_AXIOMA_AP4_FACTOR_NAMES
from riskmodels.AxiomaFactorLibraryConstants import AXIOMA_APxJP4_FACTOR_NAMES, TRANSLATED_AXIOMA_APxJP4_FACTOR_NAMES
from riskmodels.AxiomaFactorLibraryConstants import AXIOMA_UK4_FACTOR_NAMES, TRANSLATED_AXIOMA_UK4_FACTOR_NAMES
from riskmodels.AxiomaFactorLibraryConstants import AXIOMA_CA4_FACTOR_NAMES, TRANSLATED_AXIOMA_CA4_FACTOR_NAMES
from riskmodels.AxiomaFactorLibraryConstants import AXIOMA_NA4_FACTOR_NAMES, TRANSLATED_AXIOMA_NA4_FACTOR_NAMES
from riskmodels.AxiomaFactorLibraryConstants import AXIOMA_DMxUS4_FACTOR_NAMES, TRANSLATED_AXIOMA_DMxUS4_FACTOR_NAMES


# are there any text columns - this too should go in the constants, perhaps

TEXT_COLS=[
]

TEXT_COL_DICT=dict([(i,'TEXT') for i in TEXT_COLS])

class AxiomaFactorLibrary (FactorLibrary.FactorLibrary):
    def __init__(self, marketDB, modelDB, factorName, options):
        FactorLibrary.FactorLibrary.__init__(self,None, marketDB, modelDB, factorName, options)
        # store holidays in a list here
        modelClass = riskmodels.getModelByName('USAxioma2013FL')
        self.riskModel = modelClass(modelDB, marketDB)
        self.holidays={}
        self.AXIOMA_FACTOR_NAMES = AXIOMA_FACTOR_NAMES
        self.TRANSLATED_AXIOMA_FACTOR_NAMES = TRANSLATED_AXIOMA_FACTOR_NAMES
        self.factorFamily = 'US'
        query="""
            select dt from META_HOLIDAY_CALENDAR_ACTIVE h where ISO_CTRY_CODE='US' 
        """
        self.marketDB.dbCursor.execute(query)
        for r in self.marketDB.dbCursor.fetchall():
            self.holidays[str(r[0])[:10]]=True
         
    

    def getDistFileName(self, dt):
        shortdate='%04d%02d%02d' % (dt.year, dt.month, dt.day)
        return 'AXFLUS.%s.zip' % shortdate

    def getHeaders(self, dt):
        """
        create the headers
        """

        shortdate='%04d%02d%02d' % (dt.year, dt.month, dt.day)

        self.riskModel.setFactorsForDate(dt, self.modelDB)
        self.rmi = self.riskModel.getRiskModelInstance(dt, self.modelDB)
        if not self.rmi:
            logging.warning('There are no results on %s for %s', dt, self.factorName)
            return 

        self.expM = self.riskModel.loadExposureMatrix(self.rmi, self.modelDB)
        self.excludes = self.modelDB.getProductExcludedSubIssues(dt)
        self.excludeDict = dict([(i,1) for i in self.excludes])

        #self.expM.fill(0.0)
        factorNames=self.expM.getFactorNames()
        missingFactors = [x for x in set([f[0] for f in self.AXIOMA_FACTOR_NAMES]) - set(factorNames)]
        # get rid of the missing factors from the headers as well
        if len(missingFactors) > 0:
            logging.info('%s dropped from factors', missingFactors)

        
        self.AxiomaFactorNames = [x for x in self.AXIOMA_FACTOR_NAMES if x[0] not in missingFactors]
        # set up the header appropriately based on the intersection between the full list and what is in the
        # model for that date

        # first line is the factor name/categorization of the CS Holt factors
        # second line is the names as provided in the files
        # third line is the descriptions
        # fourth line is the dates
        # fifth line are the units

        #line1=['']+['Axioma US Factor Library|Axioma US Factor Library.%s|Factor Library|GROUP' % (c[2]) for c in self.AxiomaFactorNames]
        line1=['']+['Axioma Factor Library|Axioma %s Factor Library|Axioma %s Factor Library.%s|Factor Library|GROUP' % (self.factorFamily, self.factorFamily, c[2]) for c in self.AxiomaFactorNames]
        line2=['NAME'] + [self.TRANSLATED_AXIOMA_FACTOR_NAMES.get(c[0],c[0]) for c in self.AxiomaFactorNames]
        line3=['DESC'] + [c[1] for c in self.AxiomaFactorNames]
        line4=['DATE'] + ['%s' % shortdate for c in self.AxiomaFactorNames]
        line5=['UNIT'] + ['%s' % TEXT_COL_DICT.get(c[0],'NUMBER') for c in self.AxiomaFactorNames]

        self.factorNameDict={}
        for idx, factorNames in enumerate(self.AxiomaFactorNames):
            self.factorNameDict[factorNames[0]] = idx  +1 # make sure to offset the AXIOMA_ID in col 0

        self.headers=[line1, line2, line3, line4, line5]
        return

    def getConstituents(self, dt):
        """
            get factor library data from ModelDB for the given date
        """
        if not self.rmi:
            logging.warning('There are no results on %s for %s', dt, self.factorName)
            return False

        mat = self.expM.getMatrix()
        factorNames=self.expM.getFactorNames()
        results=[]

        assetDict={}
        for (aIdx, asset) in enumerate(self.expM.getAssets()):
           assetDict[asset.getModelID()]=aIdx

        for count,asset in enumerate(sorted(self.expM.getAssets())):
            if asset in self.excludeDict:
                logging.info('Excluding %s', asset)
                continue
            row=['%s' % asset.getModelID().getPublicID() ] + ['' for ii in range(len(self.factorNameDict))]
            aIdx=assetDict[asset.getModelID()]
            
            for idx,fval in enumerate(mat[:,aIdx]):
                # if there is no such factor name, skip it
                if factorNames[idx] not in self.factorNameDict:
                    continue
                colIdx=self.factorNameDict[factorNames[idx]]
                if fval is not ma.masked:
                    row[colIdx] = '%.8g' % (fval)
            results.append(row)

        self.data=results
        if len(results)==0:
            logging.warning('There are no results on %s for %s', dt, self.factorName)
            return False
        else:
            logging.info('%d entries on %s for %s factor', len(results), dt, self.factorName)
            # fix up the header line now 
            return True

class AxiomaFactorLibraryWW4(AxiomaFactorLibrary):
    def __init__(self, marketDB, modelDB, factorName, options):
        FactorLibrary.FactorLibrary.__init__(self,None, marketDB, modelDB, factorName, options)
        # store holidays in a list here
        modelClass = riskmodels.getModelByName('WWAxioma2017FL')
        self.riskModel = modelClass(modelDB, marketDB)
        self.holidays={}
        self.AXIOMA_FACTOR_NAMES = AXIOMA_WW_FACTOR_NAMES
        self.TRANSLATED_AXIOMA_FACTOR_NAMES = TRANSLATED_AXIOMA_WW_FACTOR_NAMES
        self.factorFamily = 'WW4'

    def getDistFileName(self, dt):
        shortdate='%04d%02d%02d' % (dt.year, dt.month, dt.day)
        return 'AXWW4-FL.%s.zip' % shortdate

class AxiomaFactorLibraryUS4(AxiomaFactorLibrary):
    def __init__(self, marketDB, modelDB, factorName, options):
        FactorLibrary.FactorLibrary.__init__(self,None, marketDB, modelDB, factorName, options)
        # store holidays in a list here
        modelClass = riskmodels.getModelByName('USAxioma2016FL')
        self.riskModel = modelClass(modelDB, marketDB)
        self.holidays={}
        self.AXIOMA_FACTOR_NAMES = AXIOMA_US4_FACTOR_NAMES
        self.TRANSLATED_AXIOMA_FACTOR_NAMES = TRANSLATED_AXIOMA_US4_FACTOR_NAMES
        self.factorFamily = 'US4'

    def getDistFileName(self, dt):
        shortdate='%04d%02d%02d' % (dt.year, dt.month, dt.day)
        return 'AXUS4-FL.%s.zip' % shortdate

class AxiomaFactorLibraryWWEL(AxiomaFactorLibrary):
    def __init__(self, marketDB, modelDB, factorName, options):
        FactorLibrary.FactorLibrary.__init__(self, None, marketDB, modelDB, factorName, options)
        # store holidays in a list here
        modelClass = riskmodels.getModelByName('WWAxioma2018FL')
        self.riskModel = modelClass(modelDB, marketDB)
        self.holidays={}
        self.AXIOMA_FACTOR_NAMES = AXIOMA_WWEL_FACTOR_NAMES
        self.TRANSLATED_AXIOMA_FACTOR_NAMES = TRANSLATED_AXIOMA_WWEL_FACTOR_NAMES
        self.factorFamily = 'Macroeconomic'

    def getDistFileName(self, dt):
        shortdate='%04d%02d%02d' % (dt.year, dt.month, dt.day)
        return 'AXWW4-EL.%s.zip' % shortdate

class AxiomaFactorLibraryEU4(AxiomaFactorLibrary):
    def __init__(self, marketDB, modelDB, factorName, options):
        FactorLibrary.FactorLibrary.__init__(self,None, marketDB, modelDB, factorName, options)
        # store holidays in a list here
        modelClass = riskmodels.getModelByName('EUAxioma2017FL')
        self.riskModel = modelClass(modelDB, marketDB)
        self.holidays={}
        self.AXIOMA_FACTOR_NAMES = AXIOMA_EU4_FACTOR_NAMES
        self.TRANSLATED_AXIOMA_FACTOR_NAMES = TRANSLATED_AXIOMA_EU4_FACTOR_NAMES
        self.factorFamily = 'EU4'

    def getDistFileName(self, dt):
        shortdate='%04d%02d%02d' % (dt.year, dt.month, dt.day)
        return 'AXEU4-FL.%s.zip' % shortdate

class AxiomaFactorLibraryEM4(AxiomaFactorLibrary):
    def __init__(self, marketDB, modelDB, factorName, options):
        FactorLibrary.FactorLibrary.__init__(self,None, marketDB, modelDB, factorName, options)
        # store holidays in a list here
        modelClass = riskmodels.getModelByName('EMAxioma2018FL')
        self.riskModel = modelClass(modelDB, marketDB)
        self.holidays={}
        self.AXIOMA_FACTOR_NAMES = AXIOMA_EM4_FACTOR_NAMES
        self.TRANSLATED_AXIOMA_FACTOR_NAMES = TRANSLATED_AXIOMA_EM4_FACTOR_NAMES
        self.factorFamily = 'EM4'

    def getDistFileName(self, dt):
        shortdate='%04d%02d%02d' % (dt.year, dt.month, dt.day)
        return 'AXEM4-FL.%s.zip' % shortdate

class AxiomaFactorLibraryAP4(AxiomaFactorLibrary):
    def __init__(self, marketDB, modelDB, factorName, options):
        FactorLibrary.FactorLibrary.__init__(self,None, marketDB, modelDB, factorName, options)
        # store holidays in a list here
        modelClass = riskmodels.getModelByName('APAxioma2018FL')
        self.riskModel = modelClass(modelDB, marketDB)
        self.holidays={}
        self.AXIOMA_FACTOR_NAMES = AXIOMA_AP4_FACTOR_NAMES
        self.TRANSLATED_AXIOMA_FACTOR_NAMES = TRANSLATED_AXIOMA_AP4_FACTOR_NAMES
        self.factorFamily = 'AP4'

    def getDistFileName(self, dt):
        shortdate='%04d%02d%02d' % (dt.year, dt.month, dt.day)
        return 'AXAP4-FL.%s.zip' % shortdate

class AxiomaFactorLibraryAPxJP4(AxiomaFactorLibrary):
    def __init__(self, marketDB, modelDB, factorName, options):
        FactorLibrary.FactorLibrary.__init__(self,None, marketDB, modelDB, factorName, options)
        # store holidays in a list here
        modelClass = riskmodels.getModelByName('APxJPAxioma2018FL')
        self.riskModel = modelClass(modelDB, marketDB)
        self.holidays={}
        self.AXIOMA_FACTOR_NAMES = AXIOMA_APxJP4_FACTOR_NAMES
        self.TRANSLATED_AXIOMA_FACTOR_NAMES = TRANSLATED_AXIOMA_APxJP4_FACTOR_NAMES
        self.factorFamily = 'APxJP4'

    def getDistFileName(self, dt):
        shortdate='%04d%02d%02d' % (dt.year, dt.month, dt.day)
        return 'AXAPxJP4-FL.%s.zip' % shortdate

class AxiomaFactorLibraryCN4(AxiomaFactorLibrary):
    def __init__(self, marketDB, modelDB, factorName, options):
        FactorLibrary.FactorLibrary.__init__(self,None, marketDB, modelDB, factorName, options)
        # store holidays in a list here
        modelClass = riskmodels.getModelByName('CNAxioma2018FL')
        self.riskModel = modelClass(modelDB, marketDB)
        self.holidays={}
        self.AXIOMA_FACTOR_NAMES = AXIOMA_CN4_FACTOR_NAMES
        self.TRANSLATED_AXIOMA_FACTOR_NAMES = TRANSLATED_AXIOMA_CN4_FACTOR_NAMES
        self.factorFamily = 'CN4'

    def getDistFileName(self, dt):
        shortdate='%04d%02d%02d' % (dt.year, dt.month, dt.day)
        return 'AXCN4-FL.%s.zip' % shortdate

class AxiomaFactorLibraryCA4(AxiomaFactorLibrary):
    def __init__(self, marketDB, modelDB, factorName, options):
        FactorLibrary.FactorLibrary.__init__(self,None, marketDB, modelDB, factorName, options)
        # store holidays in a list here
        modelClass = riskmodels.getModelByName('CAAxioma2018FL')
        self.riskModel = modelClass(modelDB, marketDB)
        self.holidays={}
        self.AXIOMA_FACTOR_NAMES = AXIOMA_CA4_FACTOR_NAMES
        self.TRANSLATED_AXIOMA_FACTOR_NAMES = TRANSLATED_AXIOMA_CA4_FACTOR_NAMES
        self.factorFamily = 'CA4'

    def getDistFileName(self, dt):
        shortdate='%04d%02d%02d' % (dt.year, dt.month, dt.day)
        return 'AXCA4-FL.%s.zip' % shortdate

class AxiomaFactorLibraryUK4(AxiomaFactorLibrary):
    def __init__(self, marketDB, modelDB, factorName, options):
        FactorLibrary.FactorLibrary.__init__(self,None, marketDB, modelDB, factorName, options)
        # store holidays in a list here
        modelClass = riskmodels.getModelByName('UKAxioma2018FL')
        self.riskModel = modelClass(modelDB, marketDB)
        self.holidays={}
        self.AXIOMA_FACTOR_NAMES = AXIOMA_UK4_FACTOR_NAMES
        self.TRANSLATED_AXIOMA_FACTOR_NAMES = TRANSLATED_AXIOMA_UK4_FACTOR_NAMES
        self.factorFamily = 'UK4'

    def getDistFileName(self, dt):
        shortdate='%04d%02d%02d' % (dt.year, dt.month, dt.day)
        return 'AXUK4-FL.%s.zip' % shortdate

class AxiomaFactorLibraryNA4(AxiomaFactorLibrary):
    def __init__(self, marketDB, modelDB, factorName, options):
        FactorLibrary.FactorLibrary.__init__(self,None, marketDB, modelDB, factorName, options)
        # store holidays in a list here
        modelClass = riskmodels.getModelByName('NAAxioma2019FL')
        self.riskModel = modelClass(modelDB, marketDB)
        self.holidays={}
        self.AXIOMA_FACTOR_NAMES = AXIOMA_NA4_FACTOR_NAMES
        self.TRANSLATED_AXIOMA_FACTOR_NAMES = TRANSLATED_AXIOMA_NA4_FACTOR_NAMES
        self.factorFamily = 'NA4'

    def getDistFileName(self, dt):
        shortdate='%04d%02d%02d' % (dt.year, dt.month, dt.day)
        return 'AXNA4-FL.%s.zip' % shortdate

class AxiomaFactorLibraryDMxUS4(AxiomaFactorLibrary):
    def __init__(self, marketDB, modelDB, factorName, options):
        FactorLibrary.FactorLibrary.__init__(self,None, marketDB, modelDB, factorName, options)
        # store holidays in a list here
        modelClass = riskmodels.getModelByName('DMAxioma2020FL')
        self.riskModel = modelClass(modelDB, marketDB)
        self.holidays={}
        self.AXIOMA_FACTOR_NAMES = AXIOMA_DMxUS4_FACTOR_NAMES
        self.TRANSLATED_AXIOMA_FACTOR_NAMES = TRANSLATED_AXIOMA_DMxUS4_FACTOR_NAMES
        self.factorFamily = 'DMxUS4'

    def getDistFileName(self, dt):
        shortdate='%04d%02d%02d' % (dt.year, dt.month, dt.day)
        return 'AXDMxUS4-FL.%s.zip' % shortdate

