import datetime
import logging
import configparser
import optparse
import codecs
import xml.dom.minidom as minidom
import shutil

from riskmodels import ModelDB
from riskmodels import ModelID
from riskmodels import Utilities
import marketdb.MarketDB as MarketDB
import marketdb.IndexID as IndexID

DESTDIR='/axioma/products/current/riskmodels/2.1/Derby'
FINAL_NAME='DbMeta.xml'

modelIDMap = dict()

def addCurrenciesInfo(parent, marketDB, modelDB):
    # - 3-letter ISO code
    # - description
    # - Unicode symbol(?)
    # - from_dt/thru_dt
    document = parent.ownerDocument
    currenciesElement = document.createElement('currencies')
    curSymbols = dict()
    for configLine in open('currencies.properties', 'r'):
        fields = configLine.strip().split('=')
        assert len(fields) == 2
        code = fields[0]
        assert len(code) == 3
        values = fields[1].split(';')
        assert len(values) >= 2
#        curSymbols[code] = eval('u"' + values[1].replace("'","") + '"')
        if '\\' in values[1]:
            curSymbols[code] = eval('u"' + values[1].replace("'","") + '"')
        else:
            curSymbols[code] = values[1].replace("'","")
    modelDB.dbCursor.execute("""SELECT DISTINCT currency_code FROM rmg_currency""")
    validCodes = [i[0] for i in modelDB.dbCursor.fetchall()]
    marketDB.dbCursor.execute("""SELECT code, description, from_dt, thru_dt FROM currency_ref
    ORDER BY code""")
    for (code, desc, fromdt, thrudt) in marketDB.dbCursor.fetchall():
        if code not in validCodes:
            continue
        currElt = document.createElement('currency')
        currElt.setAttribute('isocode', code)
        currElt.setAttribute('description', desc)
        currElt.setAttribute('fromdate', str(fromdt.date()))
        currElt.setAttribute('thrudate', str(thrudt.date()))
        currElt.setAttribute('symbol', curSymbols.get(code,code))
        currenciesElement.appendChild(currElt)
    parent.appendChild(currenciesElement)

def addModelInfo(parent, modelDB, modelFamilies):
    # - name
    # - description
    # - mnemonic
    # - base currency
    # - model region (for grouping models in the GUIs)
    # - from_dt/thru_dt
    # all countries
    TWO2MODELS=['AXAPxJP22-MH', 'AXAPxJP22-MH-S', 'AXAPxJP22-SH', 'AXAPxJP22-SH-S', 'AXAP22-MH', 'AXAP22-MH-S', 'AXAP22-SH', 'AXAP22-SH-S',
                'AXWW22-MH', 'AXWW22-MH-S', 'AXWW22-SH', 'AXWW22-SH-S', ]

    document = parent.ownerDocument
    modelsElement = document.createElement('models')
    allModels = modelDB.getRiskModelsForExtraction(date=datetime.date.today())
    # query to find first date that each country enters any series associated
    # with a particular model
    startDateQuery = """select mnemonic, min(from_dt) from
    (select rmg.MNEMONIC,greatest(rms.FROM_DT,rmm.FROM_DT) from_dt from RISK_MODEL_SERIE rms
     join RMG_MODEL_MAP rmm on rms.SERIAL_ID=rmm.RMS_ID
     join RISK_MODEL_GROUP rmg on rmg.RMG_ID=rmm.RMG_ID
     where rms.RM_ID=:rm_id)
    group by mnemonic"""
    for model in allModels:
        if model.mnemonic in TWO2MODELS:
            logging.info('Ignoring model %s', model.mnemonic)
            continue
        if model.mnemonic.endswith('-FL'): 
            logging.info('Ignoring FL model %s', model.mnemonic)
            continue
        if model.mnemonic.endswith('-EL'): 
            logging.info('Ignoring EL model %s', model.mnemonic)
            continue
        if model.mnemonic.find('AXWW4PRE-MH') == 0:
            logging.info('Ignoring model %s', model.mnemonic)
            continue
        modelDB.dbCursor.execute(startDateQuery, rm_id=model.rm_id)
        ctryStartDateMap = dict((m, d.date()) for (m, d) in modelDB.dbCursor.fetchall())
        modelIDMap[model.rm_id] = model.name
        query = """SELECT MIN(from_dt), MAX(thru_dt) FROM risk_model_serie
        WHERE rm_id = :model_id AND distribute = 1"""
        modelDB.dbCursor.execute(query, model_id=model.rm_id)
        r = modelDB.dbCursor.fetchall()
        if not r:
            continue
        else:
            model_from = r[0][0].date()
            model_thru = r[0][1].date()
        modelElement = document.createElement('model')
        modelElement.setAttribute('description', model.description)
        modelElement.setAttribute('name', model.name)
        modelElement.setAttribute('mnemonic', model.mnemonic)
        modelElement.setAttribute('fromdate', str(model_from))
        modelElement.setAttribute('thrudate', str(model_thru))
        modelElement.setAttribute('region', model.region)
        modelElement.setAttribute('numeraire', model.numeraire)
        modelElement.setAttribute('modeltype', model.mnemonic[-2:] == '-S' and 'Statistical' or 'Fundamental')
        ctrylist = document.createElement('countrylist')
        modelElement.appendChild(ctrylist)
        for ctry in model.rmgTimeLine:
            if ctry.rmg_id <= 0:
                continue
            ctryelt = document.createElement('countryref')
            ctryelt.setAttribute('country', ctry.rmg.mnemonic)
            ctryelt.setAttribute('fromdate', str(max(model_from, ctryStartDateMap[ctry.rmg.mnemonic])))
            ctryelt.setAttribute('thrudate', str(min(model_thru, ctry.thru_dt)))
            ctryelt.setAttribute('fadedate', str(min(model_thru, ctry.fade_dt)))
            ctrylist.appendChild(ctryelt)
        modelsElement.appendChild(modelElement)
    parent.appendChild(modelsElement)
    # add FX models
    modelDB.dbCursor.execute("""SELECT rms.rm_id, rms.revision,NAME,NUMERAIRE FROM risk_model_serie rms
    JOIN risk_model rm on rm.model_id=rms.rm_id WHERE rm.name LIKE 'FX%' AND model_id>0
    AND revision=(SELECT MAX(revision) FROM risk_model_serie rms1 WHERE rms1.rm_id=rms.rm_id)""")
    fxModelsElement = document.createElement('fxmodels')
    fxModelInfo = modelDB.dbCursor.fetchall()
    for fx in fxModelInfo:
        try:
            model=None
            model = modelDB.getRiskModelInfo(fx[0], fx[1])
        except:
            logging.warning( "Ignoring %s", fx)
            if not model:
                continue
        modelIDMap[model.rm_id] = model.name
        modelElement = document.createElement('model')
        modelElement.setAttribute('description', model.description)
        modelElement.setAttribute('name', model.name)
        modelElement.setAttribute('mnemonic', model.mnemonic)
        modelElement.setAttribute('fromdate', str(model_from))
        modelElement.setAttribute('thrudate', str(model_thru))
        modelElement.setAttribute('numeraire', model.numeraire)
        modelElement.setAttribute('modeltype', 'FX')
        ctrylist = document.createElement('countrylist')
        modelElement.appendChild(ctrylist)
        for ctry in model.rmgTimeLine:
            if ctry.rmg_id <= 0:
                continue
            ctryelt = document.createElement('countryref')
            ctryelt.setAttribute('country', ctry.rmg.mnemonic)
            ctryelt.setAttribute('fromdate', str(max(model_from, ctry.from_dt)))
            ctryelt.setAttribute('thrudate', str(min(model_thru, ctry.thru_dt)))
            ctryelt.setAttribute('fadedate', str(min(model_thru, ctry.fade_dt)))
            ctrylist.appendChild(ctryelt)
        fxModelsElement.appendChild(modelElement)
    parent.appendChild(fxModelsElement)
    return

def addRegionInfo(parent, modelDB):
    # - name
    # - description
    rmgs = modelDB.getAllRiskModelGroups()
    regionDict = dict()
    for rmg in rmgs:
        regionDict.setdefault(rmg.region_id, []).append(rmg)
    document = parent.ownerDocument
    regionsElement = document.createElement('regions')
    for region in regionDict:
        regionElt = document.createElement('region')
        modelDB.dbCursor.execute("""SELECT name, description FROM region
        WHERE id=:region_id""", region_id=region)
        regionInfo = modelDB.dbCursor.fetchall()
        if not regionInfo:
            continue
        regionElt.setAttribute('name', regionInfo[0][0])
        regionElt.setAttribute('description', regionInfo[0][1])
        countries = sorted(regionDict[region])
        countriesElement = document.createElement('countries')
        regionElt.appendChild(countriesElement)
        for country in countries:
            if country.rmg_id <= 0:
                continue
            countryElt = document.createElement('country')
            countryElt.setAttribute('name', country.description)
            countryElt.setAttribute('isocode', country.iso_code)
            countryElt.setAttribute('id', country.mnemonic)
            if not country.timeVariantDicts:
                continue
            currenciesElt = document.createElement('countrycurrencies')
            countryElt.appendChild(currenciesElt)
            for varInfo in country.timeVariantDicts:
                currencyElt = document.createElement('countrycurrency')
                currencyElt.setAttribute('isocode', varInfo['currency_code'])
                currencyElt.setAttribute('fromdate', str(varInfo['from_dt']))
                currencyElt.setAttribute('thrudate', str(varInfo['thru_dt']))
                currenciesElt.appendChild(currencyElt)
            countriesElement.appendChild(countryElt)
        regionsElement.appendChild(regionElt)
    parent.appendChild(regionsElement)

def addCompositeFamilyInfo(parent, marketDB, modelDB, modelFamilies):
    # - name
    # - description
    # - models
    document = parent.ownerDocument
    familiesElement = document.createElement('compositefamilies')
    familyModelMap = dict()
    familyModels = modelDB.getCompositeFamilyModelMap()
    for (m, f) in familyModels:
        familyModelMap.setdefault(f, []).append(m)
    allFamilies = marketDB.getETFFamilies()
    allmodelfamilies=[[''.join(i for i in mf.replace('Axioma','') if not i.isdigit()),mf.replace('Axioma','')] for mf in modelFamilies]
    for famInfo in allFamilies:
        if not famInfo.distribute:
            continue
        famElt = document.createElement('compositefamily')
        famElt.setAttribute('name', famInfo.name)
        famElt.setAttribute('description', famInfo.description)
        allmodels = []
        for modelId in familyModelMap.get(famInfo.id,[]):
            if modelId in modelIDMap:
                allmodels.append(modelIDMap[modelId])
        famElt.setAttribute('models', ' '.join(allmodels))
        #famElt.setAttribute('modelfamily', ','.join(mf[1] for mf in allmodelfamilies if mf[0]==famInfo.name.replace('ETF-','')))
        familiesElement.appendChild(famElt)
    parent.appendChild(familiesElement)

def addFutureFamilyInfo(parent, marketDB, modelDB, modelFamilies):
    # - name
    # - description
    # - models
    document = parent.ownerDocument
    familiesElement = document.createElement('futurefamilies')
    familyModelMap = dict()
    familyModels = modelDB.getFutureFamilyModelMap()
    for (m, f) in familyModels:
        familyModelMap.setdefault(f, []).append(m)
    allFamilies = marketDB.getFutureFamilies()
    allmodelfamilies=[[''.join(i for i in mf.replace('Axioma','') if not i.isdigit()),mf.replace('Axioma','')] for mf in modelFamilies]
    for famInfo in allFamilies:
        if not famInfo.distribute:
            continue
        famElt = document.createElement('futurefamily')
        famElt.setAttribute('name', famInfo.name)
        famElt.setAttribute('description', famInfo.description)
        allmodels = []
        for modelId in familyModelMap.get(famInfo.id,[]):
            if modelId in modelIDMap:
                allmodels.append(modelIDMap[modelId])
        famElt.setAttribute('models', ' '.join(allmodels))
        #famElt.setAttribute('modelfamily', ','.join(mf[1] for mf in allmodelfamilies if mf[0]==famInfo.name.replace('EIF-','')))
        familiesElement.appendChild(famElt)
    parent.appendChild(familiesElement)

def addModelPortfolioFamilyInfo(parent, modelDB, familiesElement):
    # - name
    # - description
    # - indexes
    document = parent.ownerDocument
    #familiesElement = document.createElement('indexfamilies')
    allFamilies = modelDB.getAllModelPortfolioFamilies()
    for fam in allFamilies:
        famElt = document.createElement('indexfamily')
        famElt.setAttribute('name', fam.name)
        famElt.setAttribute('description', fam.description)
        indexes = modelDB.getModelPortfolioMembersByFamily(fam)
        for i in indexes:
            indexElt = document.createElement('index')
            #indexElt.setAttribute('id', IndexID.IndexID(int(i.id)).getIDString())
            indexElt.setAttribute('id', 'PORT%05d' % i.id)
            indexElt.setAttribute('name', i.name)
            indexElt.setAttribute('description', i.description)
            indexElt.setAttribute('shortname', i.short_name)
            indexElt.setAttribute('fromdate', str(i.from_dt.date()))
            indexElt.setAttribute('thrudate', str(i.thru_dt.date()))
            nameElt = document.createElement('indexname')
            nameElt.setAttribute('officialname', i.name) 
            nameElt.setAttribute('fromdate', str(i.from_dt.date()))
            nameElt.setAttribute('thrudate', str(i.thru_dt.date()))
            indexElt.appendChild(nameElt)
            famElt.appendChild(indexElt)
        if indexes:
            familiesElement.appendChild(famElt)
    #parent.appendChild(familiesElement)

def addIndexFamilyInfo(parent, marketDB, modelDB, modelPortfolios=False):
    # - name
    # - description
    # - indexes
    document = parent.ownerDocument
    familiesElement = document.createElement('indexfamilies')
    allFamilies = marketDB.getIndexFamilies()
    for fam in allFamilies:
        if not fam.distribute:
            continue
        famElt = document.createElement('indexfamily')
        famElt.setAttribute('name', fam.name)
        famElt.setAttribute('description', fam.description)
        indexes = marketDB.getAllIndexFamilyIndices(fam)
        for i in indexes:
            indexElt = document.createElement('index')
            #indexElt.setAttribute('id', IndexID.IndexID(int(i.id)).getIDString())
            indexElt.setAttribute('id', i.axioma_id)
            indexElt.setAttribute('name', i.name)
            indexElt.setAttribute('description', i.description)
            indexElt.setAttribute('shortname', i.short_name)
            indexElt.setAttribute('fromdate', str(i.dist_from_dt))
            indexElt.setAttribute('thrudate', str(i.dist_thru_dt))
            for n in i.officialNames:
               nameElt = document.createElement('indexname')
               nameElt.setAttribute('officialname', n.official_name) 
               nameElt.setAttribute('fromdate', str(n.from_dt))
               nameElt.setAttribute('thrudate', str(n.thru_dt))
               indexElt.appendChild(nameElt)
            famElt.appendChild(indexElt)
        if indexes:
            familiesElement.appendChild(famElt)
    if modelPortfolios:
        addModelPortfolioFamilyInfo(parent, modelDB, familiesElement)
    parent.appendChild(familiesElement)

def addClassificationsInfo(parent, marketDB):
    # - name
    # - description
    # - from_dt/thru_dt
    document = parent.ownerDocument
    classifications = document.createElement('classifications')
    c1 = document.createElement('classification')
    c1.setAttribute('name','GICS')
    c1.setAttribute('description','GICS Industry Classification')
    c1.setAttribute('fromdate','1980-01-01')
    c1.setAttribute('thrudate','2999-12-31')
    classifications.appendChild(c1)
    c2 = document.createElement('classification')
    c2.setAttribute('name','TSEC')
    c2.setAttribute('description','TSEC Industry Classification')
    c2.setAttribute('fromdate','1980-01-01')
    c2.setAttribute('thrudate','2999-12-31')
    classifications.appendChild(c2)
    parent.appendChild(classifications)

def addCurrencyConvergencesInfo(parent, marketDB):
    # - old ISO code
    # - new ISO code
    # - old-to-new ratio
    # - date
    document = parent.ownerDocument
    convsElement = document.createElement('currencyconvergences')
    marketDB.dbCursor.execute("""SELECT old_id, new_id, old_to_new_rate, dt
    FROM currency_mod_convergence""")
    allConvs = marketDB.dbCursor.fetchall()
    for (old_id, new_id, rate, dt) in allConvs:
        marketDB.dbCursor.execute("SELECT code FROM currency_ref WHERE id=:old_id",
                                  old_id=old_id)
        oldCurr = marketDB.dbCursor.fetchall()
        marketDB.dbCursor.execute("SELECT code FROM currency_ref WHERE id=:new_id",
                                  new_id=new_id)
        newCurr = marketDB.dbCursor.fetchall() 
        convElt = document.createElement('currencyconvergence')
        convElt.setAttribute('oldcode', oldCurr[0][0])
        convElt.setAttribute('newcode', newCurr[0][0])
        ratio = ('%.6f' % rate).rstrip('0')
        if ratio[-1] == '.':
            ratio += '0'
        convElt.setAttribute('ratio', ratio)
        convElt.setAttribute('effdate', str(dt.date()))
        convsElement.appendChild(convElt)
    parent.appendChild(convsElement)
    
def addCashAssetIDs(parent, modelDB):
    # - model ID
    # - sequence #
    document = parent.ownerDocument
    idsElement = document.createElement('cashassetids')
    modelDB.dbCursor.execute("""SELECT cash_asset, issue_id FROM cash_asset_model_id""")
    for (csh, mid) in modelDB.dbCursor.fetchall():
        cshID = ModelID.ModelID(string=csh)
        midID = ModelID.ModelID(string=mid)
        idElt = document.createElement('cashassetid')
        idElt.setAttribute('modelid', cshID.getPublicID())
        idElt.setAttribute('sequence', str(midID.getIndex()))
        idsElement.appendChild(idElt)
    parent.appendChild(idsElement)

def createMetaDocument(marketDB, modelDB, modelFamilies, modelPortfolios=False):
    imp = minidom.getDOMImplementation()
    dt = imp.createDocumentType('axiomaMetaInfo','','metaXML.dtd')
    document = imp.createDocument('http://www.w3.org/1999/xhtml', 'axiomaMetaInfo', dt)
    root = document.documentElement
    root.setAttribute('createdate', str(datetime.date.today()))
    root.appendChild(document.createComment("""
    This file is maintained for compatibility with Axioma Portfolio Software. 
    As the file content and format could change without any advance notice, 
    Axioma would advise caution in using this file in any client processes.
    """))
    logging.info('adding currency info')
    addCurrenciesInfo(root, marketDB, modelDB)
    logging.info('adding country info')
    addRegionInfo(root, modelDB)
    logging.info('adding model info')
    addModelInfo(root, modelDB, modelFamilies)
    logging.info('adding etf info')
    addCompositeFamilyInfo(root, marketDB, modelDB, modelFamilies)
    addFutureFamilyInfo(root, marketDB, modelDB, modelFamilies)
    logging.info('adding index info')
    addIndexFamilyInfo(root, marketDB, modelDB, modelPortfolios)
    addClassificationsInfo(root, marketDB)
    logging.info('adding convergence info')
    addCurrencyConvergencesInfo(root, marketDB)
    addCashAssetIDs(root, modelDB)
    return document

if __name__ == '__main__':

    usage="usage: %prog [options]"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addDefaultCommandLine(cmdlineParser)
    cmdlineParser.add_option("--dir", action="store",
                             default=DESTDIR, dest="destDir",
                             help="name of directory")
    cmdlineParser.add_option("--file-format-version", action="store",
                             default=3.2, type='float', dest="fileFormatVersion",
                             help="version of flat file format to create")
    cmdlineParser.add_option("--model-portfolio", action="store_true",
                             default=False, dest="modelPortfolios",
                             help="name of directory")
    cmdlineParser.add_option("--final-name", action="store",
                             default=FINAL_NAME, dest="finalName",
                             help="final Name of XML file")

    (options, args) = cmdlineParser.parse_args()
    Utilities.processDefaultCommandLine(options, cmdlineParser)

    logging.info('Start')
    mktdb = MarketDB.MarketDB(user=options.marketDBUser,passwd=options.marketDBPasswd,sid=options.marketDBSID)
    mdldb = ModelDB.ModelDB(user=options.modelDBUser,passwd=options.modelDBPasswd,sid=options.modelDBSID)

    modelDict=mdldb.getModelFamilies()
    modelFamilies=list(set(modelDict.values()))

    #logging.basicConfig(level=logging.DEBUG)
    d = createMetaDocument(mktdb,mdldb,modelFamilies,options.modelPortfolios )
    tmpname='%s/DbMeta.xml.tmp' % options.destDir.rstrip('/')
    finalName='%s/%s' % (options.destDir.rstrip('/'), options.finalName)
    g = codecs.open(tmpname,'w','utf-8')
    d.writexml(g,'  ','  ',newl='\n',encoding='UTF-8')
    g.close()
    logging.info('Move file %s to %s', tmpname, finalName)
    shutil.move(tmpname, finalName)
    logging.info('Done')
