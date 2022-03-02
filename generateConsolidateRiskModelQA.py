import datetime
import logging
import math
import numpy
import numpy.linalg as linalg
import numpy.ma as ma
import optparse
import os
import sys
import time
import traceback
from scipy.stats import pearsonr

import riskmodels
from Tools.checkUniverse import UniverseChecker
from marketdb import MarketDB
from marketdb import MarketID
from marketdb.Utilities import listChunkIterator
from marketdb.qa import checkCoverage
from riskmodels import AssetProcessor, modelRevMap, modelNameMap
from riskmodels import EquityModel
from riskmodels import FixedIncomeModels
from riskmodels import MFM
from riskmodels import Matrices
from riskmodels import ModelDB
from riskmodels import ModelID
from riskmodels import PhoenixModels
from riskmodels import RiskModels
from riskmodels import Utilities
from riskmodels import VisualTool
from riskmodels.Factors import CompositeFactor
from riskmodels.Matrices import ExposureMatrix

#reload(sys)
#sys.setdefaultencoding("utf-8")

MODEL_PATH = '/axioma/products/current/riskmodels/2.1/FlatFiles/'

BDI_COUNTRY_CODES = set(['AI','AG','BS','BB','BZ','BM',
                         'VG','KY','CK','FO','GI','IM',
                         'LR','MH','AN','PA','TC','JE','GG'])

MODEL_PROXY_INDEX_MAP ={'AXCA':'S&P/TSX Composite Index',
                        'AXUS':'RUSSELL 3000',
                        'AXCN':'FTSE/Xinhua A 600',
                        'AXWW':'FTSE All-World',
                        'AXEU':'FTSE Europe',
                        'AXAP':'FTSE Asia Pacific',
                        'AXGB':'FTSE All-Share',
                        'AXUK':'FTSE All-Share',
                        'AXEM':'FTSE Emerging',
                        'AXNA':'FTSE North America',
                        'AXAU':'S&P ASX All Ordinaries',
                        'AXTW':'TSE TAIEX',
                        'AXJP':'FTSE Japan All Cap'
                        }
MODEL_INDEX_MAP = {'AXGB': ['FTSE 100', 
                            'FTSE 250', 
                            'FTSE 350', 
                            'FTSE SmallCap', 
                            'FTSE Fledgling', 
                            'FTSE AIM 100',
                            'FTSE All-Share'],
                   'AXUK': ['FTSE 100', 
                            'FTSE 250', 
                            'FTSE 350', 
                            'FTSE SmallCap', 
                            'FTSE Fledgling', 
                            'FTSE AIM 100',
                            'FTSE All-Share'],
                   'AXCA': [
                            'S&P/TSX MidCap Index',
                            'S&P/TSX SmallCap Index',
                            'S&P/TSX Income Trust Index',
                            'S&P/TSX Capped Diversified Mining Index',
                            'S&P/TSX Composite Index'],
                   'AXAU': ['S&P ASX 100',
                            'S&P ASX MC 50',
                            'S&P ASX 200',
                            'S&P ASX 300',
                            'S&P ASX Small Ordinaries',
                            'S&P ASX All Ordinaries'],
                   'AXJP': [#'Russell Japan Large Cap',
                            #'Russell Japan Mid Cap',
                            #'Russell Japan Small Cap',
                            #'Russell Japan'
                            'FTSE Japan All Cap'],
                   'AXUS': ['S and P 500',
                            'S and P 400',
                            'S and P 600',
                            'S and P 1500',
                            'RUSSELL TOP200',
                            'RUSSELL MIDCAP',
                            'RUSSELL 1000',
                            'RUSSELL 2000',
                            'RUSSELL 3000',
                            'RUSSELL SMALLCAP',
                            'RUSSELL MICROCAP',
                            'RUSSELL 1000 VALUE',
                            'RUSSELL 2000 VALUE',
                            'RUSSELL 3000 VALUE', 
                            'RUSSELL 1000 GROWTH',
                            'RUSSELL 2000 GROWTH',
                            'RUSSELL 3000 GROWTH'],
                   'AXEM': ['FTSE Advanced Emerging',
                            'FTSE Secondary Emerging',
                            'FTSE All-World BRIC',
                            'FTSE Emerging',
                            'FTSE Europe',
                            'FTSE Asia Pacific',
                            'FTSE Latin America',
                            'FTSE Middle East & Africa'],
                   'AXEU': ['FTSE Eurobloc',
                            'FTSE Nordic',
                            'FTSE Developed Europe',
                            'FTSE Emerging Europe All Cap',
                            'FTSE Europe',
                            'FTSE 100',
                            'FTSE France All Cap',
                            'FTSE Germany All Cap',
                            'FTSE Russia All Cap'],
                   'AXWW': ['FTSE Asia Pacific',
                            'FTSE Europe',
                            'FTSE All-World BRIC',
                            'FTSE Developed Europe-Asia Pacific',
                            'FTSE Emerging',
                            'FTSE Greater China',
                            'FTSE Eurobloc',
                            'FTSE Developed Europe',
                            'FTSE Developed Asia Pacific',
                            'FTSE Developed',
                            'FTSE All-World'],
                   'AXTW': ['FTSE-TSEC 50',
                            'FTSE-TSEC Mid-Cap 100',
                            'FTSE-TSEC Technology',
                            'TSE Electronics',
                            'TSE Electric and Machinery',
                            'OTC Index',
                            'OTC Electronics',
                            'TSE TAIEX'],
                   'AXCN': ['FTSE/Xinhua A 50',
                            'FTSE/Xinhua A 200',
                            'FTSE/Xinhua A 400',
                            'FTSE/Xinhua A 600',
                            'FTSE/Xinhua A Small Cap',
                            'FTSE/Xinhua A Allshare',
                            'FTSE/Xinhua B 35'],
                   'AXAP': ['FTSE Asia Pacific',
                            'FTSE Asean',
                            'FTSE Greater China',
                            'FTSE Developed Asia Pacific',
                            'FTSE Emerging'],
                   'AXNA': ['RUSSELL 1000',
                            'RUSSELL 2000',
                            'RUSSELL 3000',
                            'RUSSELL MICROCAP',
                            'S&P/TSX Composite Index',
                            'S&P/TSX Equity 60 Index',
                            'S&P/TSX MidCap Index',
                            'S&P/TSX SmallCap Index',
                            'FTSE North America'],
                }

class ModelReport: 
    def __init__(self,reportName,reportHeader,date,contentFetcher,contentFetcherCore): 
        self.reportHeader = reportHeader
        self.date = date 
        self.reportTables = dict() 
        self.sectionNames = list() 
        self.reportName = reportName
        self.jsFile = "function.js" 
        self.cssFile = "style.css" 
        self.htmlFile = "%s.html"%reportName
        self.buffer = ''
        self.contentFetcher = contentFetcher 
        self.contentFetcherCore = contentFetcherCore
        self.necessaryFields = set(["reportName","reportSection","header","content"])
    def populateReports(self):
        for passedData in self.contentFetcher: 
            self.createTable(passedData)

    def makeTitleBanner(self, title, width=80):
        pad0 = (width - len(title) - 4) // 2
        pad1 = width - len(title) - 4 - pad0
        self.buffer += self.divider(width, marker='*')
        self.buffer += self.divider(pad0, marker='*', lb=False)
        self.buffer += '  %s  ' % title
        self.buffer += self.divider(pad1, marker='*')
        self.buffer += self.divider(width, marker='*')
    
    def divider(self, width, marker='-', lb=True):
        str = ''.join([marker for i in range(width)])
        if lb:
            str += '\r\n'
        return str

    def addTable(self,sectionName,table): 
        self.reportTables.setdefault(sectionName,list()).append(table)
        if sectionName not in self.sectionNames: 
            self.sectionNames.append(sectionName) 

    def newline(self):
        self.buffer += '\r\n'    
    
    def write(self, str):
        self.buffer += str

    def isNumeric(self, s):
        try:
            float(s)
            return True
        except ValueError:
            return False
        except TypeError:
            print(s) 
            return False 

    def createDiv(self,table,height="null",width="null",JSON="null" ):
        if table.data is  None: 
            if table.description is not None:
                if "Universes are not Identical" in table.description:
                    div = """\r\n <div class="descriptionDanger"> %s </div>"""%(table.description)
                else:
                    div = """\r\n <div class="description"> %s </div>"""%(table.description)
        else:
            matrix = table.data
            header = table.getHeader() 
            reportLength = table.getReportLength()
            numIdx = [] 
            if table.reportType=="PieChart": 
                matrix = numpy.delete(matrix,0,1)
                header = header[1:]
            for idx,h in enumerate(header):
                columnData = table.getColumnData(h)
                num = 0 
                if "TICKER" in h or "SEDOL" in h: 
                    break 
                for i in columnData: 
                    if self.isNumeric(i):
                        num+=1
                    else: 
                        break
                    if num==reportLength: 
                        numIdx.append(idx) 
            if table.reportType == 'Table': 
                height = 26*(reportLength + 1)+4

            data = numpy.vstack([header,matrix])
            if table.reportName == "ESTU Structure Summary" and data.shape[0] !=2:
                if data[2][1] != data[3][1]:
                    table.reportName = "ESTU Structure Summary Danger"
            dataStr = '['
            for ridx,row in enumerate(data): 
                dataStr += '[' 
                if ridx ==0: 
                    for column in row: 
                        dataStr += """ "%s", """%column
                else: 
                    for idx,column in enumerate(row): 
                        if idx in numIdx: 
                            dataStr += '%s,'%str(column)
                        else: 
                            dataStr += """ "%s", """%column
                dataStr+='],\r\n' 
            dataStr+=']' 
            desc = "description" 
            if table.description is not None: 
                if "Size" in table.description\
                        or ("Assets" in table.description and "risk change" in table.description)\
                        or "Large Commodity Exposure Change" in table.reportName:
                    desc = "descriptionDanger"
                if "Escalate to Research IMMEDIATELY" in table.description:
                    desc = "descriptionEscalate"
                if "Publish model and email Research for confirmation" in table.description:
                    desc = "descriptionEmail"
#            print dataStr
            if table.reportType!='LineChart_material':
                div = """<div class="%s" id="%s">%s</div>
<div class="separator"></div> 
<div class="table" id = "%s" align="center">\r\n<div><script type="text/javascript">%s
google.charts.setOnLoadCallback(function(){
drawVisualization(%s,"%s","%s",%s,%s,%s);})</script>\r\n</div></div>\r\n"""\
                    %(desc,table.getReportName()+" Description",table.description if table.description!=None else "",table.getReportName(),
                      #                  "center" if table.reportType!="LineChart" else "",
                      JSON if JSON!="null" else "",dataStr,table.reportType,table.getReportName(),
                      height,width,"JSON" if JSON!="null" else "null")
            else: 
                div = """<div class="%s" id="%s">%s</div>
<div class="separator"></div> 
<div class="table" id = "%s" align="center">\r\n<div>
<script src="http://ajax.cdnjs.com/ajax/libs/json2/20110223/json2.js"></script>
<script type="text/javascript">%s               
google.charts.setOnLoadCallback(function(){ 
drawVisualization(%s,"%s","%s",%s,%s,%s);})</script>\r\n</div></div>\r\n"""\
                    %(desc,table.getReportName()+" Description",table.description if table.description!=None else "",table.getReportName(),
                      #                  "center" if table.reportType!="LineChart" else "",
                      JSON if JSON!="null" else "",dataStr,table.reportType,table.getReportName(),
                      height,width,"JSON" if JSON!="null" else "null")
                



        return div 
    def createSectionName(self,sectionName):
        div = """\r\n <div class = "sectionName">%s</div>"""%sectionName 
        if sectionName == 'EXPERIMENTAL SECTIONS': 
            div += """\r\n<script>
google.charts.load('current', {'packages':['line']});
</script>"""
        return div 

    def visualize(self): 
        logging.info("Visualizing Report to %s"%self.htmlFile)
        html = open(self.htmlFile,'w') 
        header = """<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
<script type="text/javascript" src="https://www.gstatic.com/charts/loader.js"></script>
<script type="text/javascript">
google.charts.load('current', {'packages':['controls','corechart','table','line']});
google.charts.setOnLoadCallback(drawVisualization);
</script>
<link rel="stylesheet" href="http://intranet/internet/test/bootstrap.css">
<link rel="stylesheet" href="http://maxcdn.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap-theme.min.css">
<script src="http://ajax.googleapis.com/ajax/libs/jquery/1.11.1/jquery.min.js"></script>
<script src="http://maxcdn.bootstrapcdn.com/bootstrap/3.2.0/js/bootstrap.min.js"></script>
<link href="http://maxcdn.bootstrapcdn.com/font-awesome/4.2.0/css/font-awesome.min.css" rel="stylesheet"> 
<link type="text/css" rel="stylesheet" href ="http://intranet/internet/test/htmlAccessories/style.css">
<script type="text/javascript" src="http://intranet/internet/test/htmlAccessories/function.js"></script>
"""
        body = """
   <div class="panel-group" id="daily_thought" height="100px">
       <div class="panel panel-Success">
            <div id="thought-heading" class="panel-heading">
                <h4 class="panel-title">
                    <a data-toggle="collapse" data-parent="#daily_thought" href="#textfield">Daily Thought </a>
                </h4>
            </div>
            <div id="textfield" class="panel-collapse collapse">
                <div class="panel-body">
%s
</div>
</div>
</div>
</div>"""%self.displayDailyThought(date)
        body += """
<div id="reportName"><i class="fa fa-file-text-o"></i>  %s Risk Model Report (%s)</div>"""%(self.reportHeader,
                                                         "%04d-%02d-%02d"%(date.year,
                                                                        date.month,
                                                                        date.day))

        tagLine = """ <div id="tagLine"><i class="fa fa-bar-chart"></i>  Flexible is Better.Your ideas and stories</div>"""
        divs = header + body + tagLine 
        for section in self.sectionNames:
            divs = divs+self.createSectionName(section)
            for table in self.reportTables[section]:
                if table.getReportName() == "Country Thickness Table": 
                   JSONObject = """  var JSON ={\r\n "lb":0,\r\n"hb":9,\r\n"idx":1}"""
                   divs = divs + self.createDiv(table,JSON = JSONObject)
                else: 
                    divs = divs+self.createDiv(table)
                
                logging.info("Created Div for %s"%table.getReportName())
        content = divs + """\r\n<footer class="clear">
<p> Axioma Inc </p>\r\n</footer>\r\n """
        html.write(content) 
        html.close() 

    def writeReport(self):
        logging.info("Writing to File %s",self.reportName) 
        outfile = open(self.reportName, 'w')
        self.parseTables()
        outfile.write(self.buffer)
        outfile.close()
        try: 
            visualizer = VisualTool.visualizer(self.htmlFile,
                                           self.contentFetcher,
                                           self.date,
                                           self.reportHeader,
                                           reportContentCore=self.contentFetcherCore
                                           )
#                                               sectionNames = self.sectionNames)
            visualizer.visualize()        
        except: 
            self.visualize()
        logging.info("Wrote to File %s and Visualized to File %s"%(self.reportName,
                                                                   self.htmlFile))
    def displayDailyThought(self,date):
        infile = open('dailyThoughts.txt')
        blurbCollection = []
        new_blurb = True
        for line in infile:
            if new_blurb:
                blurb = ''
                new_blurb = False
            if line=='\n' or line=='\r':
                blurbCollection.append(blurb)
                new_blurb = True
                continue
            blurb += line
        modelDate = date
        i = int(time.mktime(modelDate.timetuple()) / (3600*24)) % len(blurbCollection)
        return blurbCollection[i]

    def emailReport(self, date, recipientList):
        import smtplib
        import email.utils
        import email.mime.text
        smtp_server = 'mail.axiomainc.com'
        #smtp_user = 'rmail'
        #smtp_pass = 'wtSL0wGOP'
        session = smtplib.SMTP(smtp_server)
        #session.login(smtp_user, smtp_pass)
        
        sender = 'ops-rm@axiomainc.com'
        user = os.environ['USER']
        assert(len(recipientList) > 0)
        message = email.mime.text.MIMEText(self.buffer)
        message.set_charset('utf-8')
        message['From'] = email.utils.formataddr(('Risk Model Operations', sender))
        message['Subject'] = '%04d-%02d-%02d %s Risk Model QA Report run by %s' % \
            (date.year, date.month, date.day,self.reportHeader, user)
        message['To'] = ', '.join(recipientList)
        try:
            session.sendmail(sender, recipientList, message.as_string())
        except Exception as e:
            traceback.print_exc()
            logging.info("Could not send email %s"%e)
            pass
        finally:
            session.quit()


    def createTable(self, data): 
        dataFields = set(data.getFieldNames())
        missingNecessaries = self.necessaryFields.difference(dataFields) 
        if "reportType" not in dataFields: 
            data.reportType = "Table" 
        if  "header" not in dataFields or data.header is None:
            # This is the case when there is no table content, just a description 
            # [ like 0 asset joining Universe]
            try: 
                assert("description" in dataFields) 
            except: 
                logging.error("Expecting a description when there is no table content") 
            table = ReportTable(data.reportName,data.reportSection)
            table.description = data.description 
            self.addTable(data.reportSection,table) 

        else: 
            try: 
                assert(len(missingNecessaries)== 0)
            except: 
                logging.error("Missing Necessary Fields : %s"%",".join(missingNecessaries))
            try: 
                assert(len(data.content.shape)==2)
            except: 
                logging.error("Content given is not a valid 2-D Matrix") 

            table = ReportTable(data.reportName,data.reportType) 
            table.setHeader(data.header) 
            if "description" in dataFields: 
                table.setDescription(data.description) 
            if "populateByRow" not in dataFields or dataFields.populateByRow == True: 
                assert(data.content.shape[1]==len(data.header))
                table.setReportLength(data.content.shape[0])
                for row in data.content:
                    table.addRowData(numpy.asarray(row).reshape(-1))
            else: 
                assert(data.content.shape[0]==len(data.header))
                table.setReportLength(data.content.shape[1])
                for idx,column in enumerate(data.content): 
                    table.addColumnData(data.header[idx],numpy.asarray(column).reshape(-1)) 
            self.addTable(data.reportSection,table) 


    def parseTables(self):
        for section in self.sectionNames:
            self.makeTitleBanner(section) 
            for table in self.reportTables[section]:
                resultTable = table
                description = resultTable.description 
                if description: 
                    self.buffer += '\r\n'
                    self.buffer += description 
                    self.buffer += '\r\n'
                header = resultTable.getHeader() 
                if header is not None : 
                    headerWidthList = []
                    for idx,h in enumerate(header):
                        headerwidth = max (width for width in (len(str(col)) for col in \
                                                  (resultTable.getColumnData(h))))
                        headerwidth = max(len(h),headerwidth)
                        if idx == 0:
                             self.buffer += '%-*s'%(headerwidth,h)
                        else:
                            self.buffer += '|%-*s'%(headerwidth,h)
                        headerWidthList.append(headerwidth)
                    self.buffer += '\r\n'
                    self.buffer += '-'*(sum(headerWidthList)+ len(headerWidthList) -1)
                    self.buffer += '\r\n'
                    rowNum = 0
                    while rowNum < resultTable.getReportLength():
                        rowData = resultTable.getRowData(rowNum)
                        for idx, item in enumerate(rowData):
                            if item is ma.masked:
                                item = ' '
                            item = str(item)
                            if idx == 0:
                                self.buffer += '%-*s'%(headerWidthList[idx],item)
                            else:
                                self.buffer += '|%-*s'%(headerWidthList[idx],item)
                        self.buffer +='\r\n'
                        rowNum += 1
                    self.buffer += '\r\n'

class RiskModelValidator:
    def __init__(self, modelSelectorList, priorModelSelectorList, date, 
                 modelDB, marketDB, modelAssetType,maxRecords,modelPath):
        self.date = date
        dates = modelDB.getDates(modelSelectorList[0].rmg,
                                 self.date, 30, excludeWeekend=True)
        self.prevDate = dates[-2]
        self.marketDB = marketDB
        self.modelDB = modelDB
        self.currency = modelSelectorList[0].numeraire.currency_code
        self.modelSelectorList = modelSelectorList
        self.priorModelSelectorList = priorModelSelectorList
        self.modelAssetType = modelAssetType
        self.appendIndustry=True if self.modelAssetType =='Equity' else False 
        self.appendAssetType=True if self.modelAssetType =='Equity' else False         
        self.univ = [] 
        self.prevUniv = []
        self.estUniv = [] 
        self.prevEstUniv = [] 
        self.covMatrixList = [] 
        self.p_covMatrixList = [] 
        self.expMList = []
        self.prevExpMList = [] 
        self.assetCapDict = {} 
        self.modelmnemonicList = [model.mnemonic[2:] for model in self.modelSelectorList]
        self.mnemonicIdxMap = dict([(i,j) for (i,j) in enumerate(self.modelmnemonicList)])
        self.idxMnemonicMap = dict([(j,i) for (i,j) in enumerate(self.modelmnemonicList)])
        self.wholeUniv = set() 
        self.prevWholeUniv = set()
        self.modelMarketMap = {} 
        self.coreDataList = []
        self.dataList = [] 
        self.rmiList = [] 
        self.p_rmiList = [] 
        self.maxRecords = maxRecords
        self.modelPath = modelPath
        self.headerMap = {}
        self.exposureQA = {"fund075": False, "tech04": False, "tech04075": False, "fx0405": False}
        if self.maxRecords==-1:
            self.maxRecords=None 
        # Get mapping of RMG IDs to ISO code, description, etc
        query = """SELECT g.rmg_id FROM risk_model_group g
                   WHERE g.rmg_id IN (
                        SELECT DISTINCT m.rmg_id 
                        FROM rmg_model_map m WHERE m.rms_id > 0)"""
        modelDB.dbCursor.execute(query)
        allRiskModelGroups = [modelDB.getRiskModelGroup(r[0]) \
                            for r in modelDB.dbCursor.fetchall()]
        self.rmgInfoMap = dict((rmg.mnemonic, rmg) \
                            for rmg in allRiskModelGroups)

        for rmg in self.rmgInfoMap.values():
            rmg.setRMGInfoForDate(date)

        self.dates = dates
        if self.modelAssetType == "Universal":
            self.rmgList= modelDB.getAllRiskModelGroups()
        # Construct Lists of ExposureMatrix,Estimation Universe, Estimation Universe, 
        # Previous Day Model Universe, Previous Estimation Universe 
        for rm in self.modelSelectorList:
            rm.setFactorsForDate(self.date,self.modelDB)
            rmi = modelDB.getRiskModelInstance(rm.rms_id,self.date)
            self.rmiList.append(rmi) 
            if self.modelAssetType in ["Equity","Future"]:
                self.expMList.append(rm.loadExposureMatrix(rmi,self.modelDB))
                self.univ.append(set(modelDB.getRiskModelInstanceUniverse(rmi,False)))
            if self.modelAssetType == "Equity":
                self.estUniv.append(set(modelDB.getRiskModelInstanceESTU(rmi)))
            elif self.modelAssetType == "Future" : 
                self.estUniv.append(rm.loadEstimationUniverse(rmi, self.modelDB))
            (cov,factors) = rm.loadFactorCovarianceMatrix(rmi,self.modelDB)
            self.factors = factors 
            self.covMatrixList.append(cov) 
            rm.setFactorsForDate(self.prevDate,self.modelDB)
            prevrmi = modelDB.getRiskModelInstance(rm.rms_id,self.prevDate)
            self.p_rmiList.append(prevrmi) 
            if self.modelAssetType in ["Equity","Future"]:
                self.prevExpMList.append(rm.loadExposureMatrix(prevrmi,self.modelDB))
                self.prevUniv.append(set(modelDB.getRiskModelInstanceUniverse(prevrmi,False)))
            if self.modelAssetType == "Equity":
                self.prevEstUniv.append(set(modelDB.getRiskModelInstanceESTU(prevrmi)))
            elif self.modelAssetType == "Future": 
                self.prevEstUniv.append(rm.loadEstimationUniverse(prevrmi, self.modelDB))
            (cov,factors) = rm.loadFactorCovarianceMatrix(prevrmi,self.modelDB)
            self.p_covMatrixList.append(cov) 

        self.fundamental = [] 
        self.statistical = [] 
        for model in self.modelSelectorList:
            if model.mnemonic[-2:] == '-S':
                self.statistical.append(model) 
            else: 
                self.fundamental.append(model)
        for univ in self.univ: 
            self.wholeUniv = self.wholeUniv.union(univ) 
        for puniv in self.prevUniv:
            self.prevWholeUniv = self.prevWholeUniv.union(puniv)

    def returnModelStructure(self):
        logging.info("Processing Model Structure") 
        modelmnemonic = (',').join(self.modelmnemonicList)  
        description = 'Results for (%s) on %04d-%02d-%02d' % \
                 (modelmnemonic, self.date.year, self.date.month, self.date.day)

        for data in self.constructModelStructureTable():
            data.reportSection = "MODEL STRUCTURE" 
            self.dataList.append(data) 
        logging.info("Finished Processing Model Structure")


    def returnModelCoverage(self):
        #Assue The models in the same rmg having the same currency oode     
        logging.info("Processing Model Coverage") 
        modelDB = self.modelDB
        marketDB = self.marketDB 
        sectionName = "MODEL UNIVERSE" 
        #Find Model Joiner and Leavers
        (joinerList,leaverList,assetJoinerDict,assetLeaverDict)= \
            self.findJoinerLeaver(self.univ, self.prevUniv) 
        #Info includes from_dt, thru_dt,industry 
        self.extendJoinerLeaverInfo(joinerList,leaverList,findIndustry=True)

        #Find any benchmark / composite assets not in model
        if self.modelAssetType == "Equity":
            (missingBM,missingCM,missingBMDict,missingCMDict,terminated) = self.findMissingBMCM()
        else: 
            missingBM = list() ; missingCM = list();terminated = list() 
        missingBMCM = list(set(missingBM).union(set(missingCM)))
        self.createDataBundle(missingBMCM)

        if len(leaverList)>0 or len(missingBMCM)>0: 
            self.fetchInfoForLeavers(leaverList) 
            if self.modelAssetType == "Equity":
                self.extendModelMarketMap(self.getMarketID(leaverList+\
                                                           missingBMCM))

        if self.modelAssetType == "Equity":
            headerList = ["CAP (%s m)"%self.currency,"FROM_DT","THRU_DT"] 
            tableList = [self.assetCapDict,self.fromDateDict,self.thruDateDict]
        else: 
            modelMarketMap = dict(modelDB.getIssueMapPairs(self.prevDate))
            headerList = ["CAP (%s m)"%self.currency,"FROM_DT","THRU_DT","RIC", "BB TICKER"] 
            tableList = [self.assetCapDict,self.fromDateDict,self.thruDateDict, self.ricMap, 
                 self.bbTickerMap]
            for sid in leaverList: 
                self.modelMarketMap[sid.getModelID()] = modelMarketMap.get(sid.getModelID())
   
        if not hasattr(self,"sidClsMap"):
            self.sidClsMap = dict() 

        if not hasattr(self,"sidAssetTypeMap"):
            self.sidAssetTypeMap = dict() 

        if not hasattr(self,"estuRankDict"):
            self.estuRankDict = dict() 

        headerList.extend(["INDUSTRY","ASSET TYPE"]) 
        tableList.extend([self.sidClsMap,self.sidAssetTypeMap])
        self.headerMap.update(dict(zip(headerList,tableList)))

        # *** Universe Structure Table ***
        univCapList = [] 
        for univ in self.univ:
            idxList = [self.assetIdxMap[sid] for sid in univ]
            univCaps = ma.take(self.marketCaps, idxList)
            univCapList.append(ma.sum(univCaps)/1e9)

        data = Utilities.Struct() 
        data.header = ['Model','Universe','Previous Universe','Delta',
                       'Market Cap (bn %s)'%self.currency]
        # For futures, do not include Mcap 
        if self.modelAssetType =="Future": 
            data.header = data.header[:-1]

        resultList = [] 
        for idx, univ in enumerate(self.univ): 
            result = [self.modelmnemonicList[idx]] 
            puniv = self.prevUniv[idx]
            if self.modelAssetType == "Equity":
                result.extend([len(univ),len(puniv),len(univ)-len(puniv),'%.2f'%univCapList[idx]])
            else: 
                result.extend([len(univ),len(puniv),len(univ)-len(puniv)])  
            resultList.append(result)
        data.content=numpy.matrix(resultList)
        if self.modelAssetType == "Equity":
            data.reportName = "Universe Structure Summary" 
        else: 
            data.reportName = "Commodity Universe Structure Summary"             
        data.reportSection = sectionName
        self.coreDataList.append(data) 

        # ***  Create Joiner Table ***         
        if len(joinerList) >0: 
            joinerCap = [self.assetCapDict.get(joiner) for joiner in joinerList] 
            rank = numpy.argsort(-numpy.array(joinerCap))
            ranked = [joinerList[i] for i in rank] 
            joinerList = ranked 
            if self.modelAssetType == "Equity":
                extraHeader = ["CAP (%s m)"%self.currency,"FROM_DT"]
            else: 
                extraHeader = ["FROM_DT"]                
            data = self.prepAssetLevelInfo(joinerList,extraHeader,sidModelDict=assetJoinerDict,
                                           appendIndustry = self.appendIndustry,
                                           appendAssetType = self.appendAssetType,
                                           modelAssetType = self.modelAssetType)

            if len(joinerList)>1:
                data.description = "%d Assets Joining Model Universe"%len(joinerList)
            else:
                data.description = "1 Asset Joining Model Universe"
        else: 
            data = Utilities.Struct() 
            data.description = "No Asset Joining Model Universe"
        if self.modelAssetType == "Equity":
            data.reportName = "Model Joiner" 
        else: 
            data.reportName = "Commodity Model Joiner" 
        data.reportSection = sectionName 
        self.coreDataList.append(data) 
        
        # *** Create Leaver Table ***  
        if len(leaverList)>0: 
            if self.modelAssetType == "Equity":
                leaverCap = [self.assetCapDict.get(leaver) for leaver in leaverList]
                rank = numpy.argsort(-numpy.array(leaverCap))
                ranked = [leaverList[i] for i in rank]
                leaverList = ranked 
            if self.modelAssetType == "Equity":
                extraHeader = ["CAP (%s m)"%self.currency,"THRU_DT"]
            else: 
                extraHeader = ["THRU_DT"]
            data = self.prepAssetLevelInfo(leaverList,extraHeader,sidModelDict=assetLeaverDict,
                                           appendIndustry=False,appendAssetType=False,
                                           modelAssetType = self.modelAssetType)
            if len(leaverList)>1:
                data.description = "%d Assets Leaving  Model Universe"%len(leaverList)
            else:
                data.description = "1 Asset Leaving Model Universe"
        else: 
            data = Utilities.Struct() 
            data.description = "No Asset Leaving  Model Universe"
        if self.modelAssetType == "Equity":
            data.reportName = "Model Leaver" 
        else: 
            data.reportName = "Commodity Model Leaver" 
        data.reportSection = sectionName 
        self.coreDataList.append(data) 

        self.factorTypeMap = self.constructFactorTypeMap() 
        # *** Create Futures Universe Contracts Composition Table ** 
        if self.modelAssetType == "Future":
            data = self.getFuturesDistribution()
            if data is not None: 
                data.reportSection = sectionName 
                self.dataList.append(data) 
        else:
            # *** Create Missing Benchmark Table ** 
            if len(missingBM)>0: 
                extraHeader = self.modelmnemonicList
                assetMissingReasonDict = self.checkMissingReason(missingBM,missingBMDict)
    #            print missingBMDict, "MISSING BMKDICT" 
                data = self.prepAssetLevelInfo(missingBM,extraHeader,assetMissingReasonDict,
                                               missingBMDict,
                                               appendIndustry = self.appendIndustry,
                                               appendAssetType = self.appendAssetType,
                                               modelAssetType = self.modelAssetType)
                if len(missingBM) >1:
                    data.description = "%d Benchmark Assets Missing from Risk Model Universe"%len(missingBM)
                else:
                    data.description = "1 Benchmark Asset Missing from Risk Model Universe"
            else:
                data = Utilities.Struct() 
                data.description = "No Benchmark Asset Missing from Risk Model Universe"
            data.reportName = "Missing Benchmark" 
            data.reportSection = sectionName 
            self.coreDataList.append(data) 

            # *** Create Terminated benchmark Table ** 
            if len(terminated)!=0 : 
                indexMapping = self.findIndexInclusion(terminated)
                resultList = list() 
                headerList = ["Index (%)"]
                values = list(indexMapping.values())
                if len(values) !=0 :
                    affectedAssets = set() 
                    for valueSet in values: 
                        affectedAssets = affectedAssets.union(set(valueSet.keys()))
                    affectedAssets = list(affectedAssets) 
                    headerList.extend(affectedAssets)
                    for key in indexMapping.keys(): 
                        result = [key]
                        value = indexMapping[key]
                        for asset in affectedAssets: 
                            result.append(value.get(asset,0))
                        resultList.append(result)                     
                    data = Utilities.Struct() 
                    data.header = headerList 
                    data.content = numpy.matrix(resultList)
                    data.reportName = "Terminated Benchmark Constituent" 
                    data.description = "Terminated Benchmark Asset. Re-transfer Affected Index if Needed" 
                    data.reportSection = sectionName 
                    self.coreDataList.append(data) 

            # *** Create Missing Composite Table *** 
                    
            data = Utilities.Struct() 
            data.description = "No Composite Asset Missing from Risk Model Universe"
            if len(missingCM)>0: 
                assetMissingReasonDict = self.checkMissingReason(missingCM, missingCMDict)                
                assetMissingReasonDictCopy = assetMissingReasonDict.copy() 
                for key,value in assetMissingReasonDict.items(): 
                    if set(value.values()) == set(['Asset type is ETF and hence is not in the model']): 
                        missingCM.remove(key) 
                        del assetMissingReasonDictCopy[key]
                        del missingCMDict[key]
                if len(missingCM) >0:
                    extraHeader = self.modelmnemonicList
                    data = self.prepAssetLevelInfo(missingCM,extraHeader,assetMissingReasonDictCopy,
                                                   missingCMDict,
                                                   appendIndustry = self.appendIndustry,
                                                   appendAssetType = self.appendAssetType,
                                                   modelAssetType = self.modelAssetType)
                    if len(missingCM)>1:
                        data.description = "%d Composite Constituents Missing from Risk Model Universe"%len(missingCM) 
                    else:
                        data.description = "1 Composite Constituents  Missing from Risk Model Universe"
            data.reportName = "Missing Composite" 
            data.reportSection = sectionName 
            self.coreDataList.append(data) 
            logging.info("Finished Processing Model Coverage")

    def returnExchangeInformation(self): 
            # *** Create Exchange Change Table *** 
            logging.info("Processing Exchange Change")
            clsFamily = self.marketDB.getClassificationFamily('REGIONS')
            assert(clsFamily is not None)
            clsMembers = dict((i.name, i) for i in marketDB.\
                                getClassificationFamilyMembers(clsFamily))
            clsMember = clsMembers.get('Market')
            assert(clsMember is not None)
            clsRevision = marketDB.getClassificationMemberRevision(
                        clsMember, self.date)
            clsData = dict((sid, cls.classification.name) \
                        for (sid,cls) in \
                        modelDB.getMktAssetClassifications(clsRevision, 
                        self.wholeUniv, self.date, marketDB).items())
            prevClsRevision = marketDB.getClassificationMemberRevision(
                        clsMember, self.prevDate)
            pClsData = dict((sid, cls.classification.name) \
                        for (sid,cls) in \
                        modelDB.getMktAssetClassifications(prevClsRevision, 
                        self.prevWholeUniv, self.prevDate, marketDB).items())
            changedAssets = list()
            currDict = dict();prevDict = dict();exIdMap = dict() 
            for (sid,cls) in clsData.items():
                exIdMap.setdefault(cls,list()).append(sid)
                prevCls = pClsData.get(sid)
                if prevCls is not None:
                    if cls != prevCls:
                        changedAssets.append(sid)
                        currDict[sid] = cls 
                        prevDict[sid] = prevCls
            if len(changedAssets)>0:
                self.headerMap['CURR EX'] = currDict 
                self.headerMap['PREV EX'] = prevDict 
                extraHeader = ["CAP (%s m)"%self.currency,"CURR EX", "PREV EX"]
                data = self.prepAssetLevelInfo(changedAssets,extraHeader)
                data.description = "Assets with Exchange Change"
            else:
                data = Utilities.Struct() 
                data.description = "No Asset with Exchange Change "
            data.reportName = "Reported Exchange Change" 
            data.reportSection = "EXCHANGE INFORMATION"
            self.coreDataList.append(data) 
            logging.info("Finished Processing Exchange Change ")

            # Get Country / Currency Change
            if len(self.rmgList) > 1:
                logging.info("Starting Processing Country/Currency Change") 
                data = self.returnCountryCurrencyChanges()
                if data is not None:
                    data.reportSection = "COUNTRY INFORMATION"
                    self.coreDataList.append(data)
                logging.info("Finished Processing Country/Currency Change") 

            # *** Exchange Distribution Chart ***# 
            logging.info("Copmuting exchange distribution") 
            data = Utilities.Struct() 
            data.reportSection = "EXCHANGE INFORMATION"
            resultList = [] 
            Ex = list(exIdMap.keys()) 
            if len(Ex) > 18: 
                suffix = ' Pie'
                data.reportType = "PieChart"        
                data.header = ['#','Exchange', 'Asset Count']
            else: 
                suffix = ''
                data.reportType = "BarChart"
                data.header = ['Exchange', 'ESTU Assets','annotation','Non-ESTU Assets','annotation1']
            counts = [len(exIdMap[ex]) for ex in Ex]
            estuCounts = [ len([i for i in exIdMap[ex] if i in self.estUniv[0]])for ex in Ex] 
            rank = numpy.argsort(-numpy.array(counts))
            if suffix == ' Pie': 
                total = sum(counts)
                for idx,i in enumerate(rank): 
                    resultList.append([str(idx),Ex[i]+' ('+str(counts[i])+')',
                                       float(counts[i])/total])
            else: 
                for i in rank: 
                    resultList.append([Ex[i],estuCounts[i],estuCounts[i],
                                       counts[i]-estuCounts[i],counts[i]-estuCounts[i],
                                       ])
            data.content=numpy.matrix(resultList)
            data.description = "Asset Distribution"+suffix
            data.reportName = "Asset Distribution"+suffix
            self.dataList.append(data)
            logging.info("Finished computing exchange distribution") 
        
    def findIndexInclusion(self,axidList): 
        axidArgList = [('axid%d'%i) for i in range(len(axidList))]
        query = """
                select axioma_id, name,value,index_id from index_constituent_easy where 
                axioma_id in (%(axids)s) 
                and dt = :dt_arg
                """%{'axids':','.join([(':%s'%i) for i in axidArgList])}

        myDict = dict([('dt_arg',self.date)])
        myDict.update(dict(zip(axidArgList,axidList)))
        self.marketDB.dbCursor.execute(query,myDict)
        result = self.marketDB.dbCursor.fetchall()
        inclWgtDict = dict()
        for item in result: 
            if "OPEN" not in item[1]: 
                if item[1] not in inclWgtDict.keys():
                    inclWgtDict[item[1]] = dict([(item[0],round(item[2]*100,5))])
                else: 
                    inclWgtDict[item[1]].update(dict([(item[0],round(item[2]*100,5))]))

        return inclWgtDict 

    def returnModelESTUDetails(self): 
        logging.info("Processing Estimation Universe") 
        sectionName="ESTIMATION UNIVERSE" 
        sectionNonCore="COUNTRIES AND INDUSTRIES"
        # If Estimation Universes are not the idential, 
        # Divided into two groups : Fundamental / Statistical 
        if self.modelAssetType == "Equity":
            (identical,self.wholeEstUniv) = self.compareEstimationUniverse(sectionName) 
        else: 
            identical = True; self.wholeEstUniv = self.estUniv[0]
        estUnivList = list(self.wholeEstUniv)  
        # Assume Previous Day Estimation Universe are identical 
        self.prevWholeUniv = self.prevEstUniv[0] 
        ##############################
        # I'm assuming the top assets are identical ! 
        prevEstUnivList = list(self.prevWholeUniv) 

        # *** Create Structure Table ** 
        estu60tr = modelDB.loadTotalReturnsHistory(self.modelSelectorList[0].rmg, 
                                    self.date, estUnivList, 60)
        self.estuReturnHistory = estu60tr 
        tr = estu60tr.data[:,-1].filled(0.0)
        noReturn = [estUnivList[idx] for idx in numpy.flatnonzero(ma.getmaskarray(tr))] 

        idLists = [] 
        if identical: 
            idxList = [self.assetIdxMap[sid] for sid in self.estUniv[0]]
            idLists.append(idxList)
        else: 
            for idx,model in enumerate(self.modelSelectorList):
                idLists.append([self.assetIdxMap[sid] for sid in self.estUniv[idx]])

        estuCaps = []; estuReturns = [] 
        for idx,idxList in enumerate(idLists):
            univCaps = ma.take(self.marketCaps, idxList)
            estuCap = ma.sum(univCaps)/1e9 
            estuCaps.append(estuCap)
            wgt = [self.assetCapDict.get(cap) if self.assetCapDict.get(cap)\
                       is not None else 0.0 for cap in estUnivList]
            wgt = numpy.array(wgt)/estuCap 
            estuReturn = ma.inner(tr, wgt)/10.0
            estuReturns.append(estuReturn)

        data = Utilities.Struct()
        data.header = ['Model','ESTU','Previous ESTU','Delta','Total ESTU Cap (bn %s)'%self.currency,
                       'ESTU Return (Cap-Weighted)']
        resultList = [] 
        for idx, univ in enumerate(self.estUniv): 
            puniv = self.prevEstUniv[idx]
            if identical: 
                result = [self.modelmnemonicList[idx],len(univ),len(puniv),
                          len(univ)-len(puniv),'%.2f'%estuCaps[0],'%.2f %%'%estuReturns[0]]
            else: 
                result = [self.modelmnemonicList[idx],len(univ),len(puniv),
                          len(univ)-len(puniv),'%.2f'%estuCaps[idx],'%.2f %%'%estuReturns[idx]]
            resultList.append(result)
        data.content=numpy.matrix(resultList)
        data.reportName = "ESTU Structure Summary" 
        data.reportSection = sectionName
        self.coreDataList.append(data) 

        # Find Joiner / Leaver of Estimation Universe 
        (joinerList,leaverList,assetJoinerDict,assetLeaverDict)= \
            self.findJoinerLeaver(self.estUniv, self.prevEstUniv) 
        self.extendJoinerLeaverInfo(joinerList,leaverList,findIndustry=False)

        # ***  Create Joiner Table ***         
        if len(joinerList) >0: 
            joinerCap = [self.assetCapDict.get(joiner) for joiner in joinerList] 
            rank = numpy.argsort(-numpy.array(joinerCap))
            ranked = [joinerList[i] for i in rank] 
            joinerList = ranked 
            extraHeader = ["CAP (%s m)"%self.currency,"FROM_DT"]
            if self.modelAssetType =="Future": 
                extraHeader = extraHeader[1:]
            data = self.prepAssetLevelInfo(joinerList,extraHeader,sidModelDict=assetJoinerDict,
                                           appendIndustry = self.appendIndustry,
                                           appendAssetType = self.appendAssetType,
                                           modelAssetType = self.modelAssetType)
            if len(joinerList)>1:
                data.description = "%d Assets Joining Estimation Universe"%len(joinerList)
            else:
                data.description = "1 Asset Joining Estimation Universe"
        else: 
            data = Utilities.Struct()
            data.description = "No Asset Joining Estimation Universe"
        data.reportName = "ESTU Joiner"
        data.reportSection = sectionName
        self.coreDataList.append(data) 

        # *** Create Leaver Table *** 
        qualifyInfo = self.modelDB.loadESTUQualifyHistory(
                self.modelSelectorList[0].rms_id, leaverList, [self.prevDate])        
        qualifyDict = dict((sid, 1) for (i,sid) in \
                    enumerate(leaverList) if qualifyInfo.data.filled(0.0)[i,0] != 0.0)
        for leaver in leaverList: 
            if leaver not in qualifyDict:
                qualifyDict[leaver] = '--' 
        self.headerMap["QUALIFY"] = qualifyDict 
        if len(leaverList)>0:
            leaverCap = [self.assetCapDict.get(leaver) for leaver in leaverList]
            rank = numpy.argsort(-numpy.array(leaverCap))
            ranked = [leaverList[i] for i in rank]
            leaverList = ranked 
            extraHeader = ["CAP (%s m)"%self.currency,"THRU_DT","QUALIFY"]
            if self.modelAssetType =="Future": 
                extraHeader = extraHeader[1:]
            data = self.prepAssetLevelInfo(leaverList,extraHeader,sidModelDict = assetLeaverDict,
                                           appendIndustry = self.appendIndustry,
                                           appendAssetType = self.appendAssetType,
                                           modelAssetType = self.modelAssetType)

            if len(leaverList)>1:
                data.description = "%d Assets Leaving Estimation Universe"%len(leaverList)
            else:
                data.description = "1 Asset Leaving Estimation Universe"
        else: 
            data = Utilities.Struct() 
            data.description = "%d Asset Leaving  Estimation Universe"%len(leaverList)
        data.reportName = "ESTU Leaver"
        data.reportSection = sectionName
        self.coreDataList.append(data) 
    
        
        # *** Create Top Asset Table *** 
        # Using the first estuniv cap for now 

        if self.modelAssetType == 'Equity':
            data=self.prepTopAssetInfo(40,estUnivList,prevEstUnivList,estuCaps[0])
            data.description = "Top 40 Assets in Estimation Universe"
            data.reportName = "Top Assets" 
            data.reportSection = sectionName
            self.coreDataList.append(data) 

            # If multi-country model, report on composition by country
            # *** Country Composition Table *** 
            if len(self.rmgList) > 1:
                dataList = self.prepCountryInfo(estUnivList,noReturn,estuCap)
                for data in dataList: 
                    if data is not None:
                        data.reportSection = sectionNonCore 
                        self.dataList.append(data) 

            (data,industryAssetMap) = self.prepIndustryThicknessInfo(set(estUnivList).difference(noReturn))
            if data is not None: 
                data.description = "Industry Thickness Table"  
                data.reportName = "Industry Thickness Table" 
                data.reportSection = sectionNonCore
                self.dataList.append(data)

            # Drawing Industry Demographics 
            data = self.prepIndustryDemographics(industryAssetMap)
            data.reportName = "Industry Demographics" 
            data.reportSection = sectionNonCore
            self.dataList.append(data) 
            logging.info("Finished Processing Estimation Universe")

    def getFactorTypeDict(self): 
        # For universal model 
        query = """ 
    select sf.SUB_ID,f.name,ft.NAME from FACTOR f,  SUB_FACTOR sf , FACTOR_TYPE ft
    where  f.FACTOR_ID = sf.FACTOR_ID 
    and f.FACTOR_TYPE_ID = ft.FACTOR_TYPE_ID
    and f.FACTOR_ID in ( select distinct FACTOR_ID from RMS_FACTOR where RMS_ID = '30000')"""
        result = dict()
        self.modelDB.dbCursor.execute(query) 
        sub_id,name,factorType = zip(*self.modelDB.dbCursor.fetchall())
        self.factorTypeMap = dict(zip(name,
                                      [Matrices.FactorType(ftype,ftype) for ftype in factorType]))
        self.factorSIDMap = dict(zip(name,sub_id))
        return (dict(zip(sub_id,factorType)),dict(zip(sub_id,name)))

    def getFactorsForDate(self,date): 
        query = """ 
    select sub_factor_id from rms_factor_return where rms_id = 30000
    and dt = :dt_arg""" 
        self.modelDB.dbCursor.execute(query,dict([('dt_arg',date)]))
        return [item[0] for item in self.modelDB.dbCursor.fetchall()]

    def getFactorLastAppearanceDate(self,sid): 
        query = """ 
    select max(dt) from RMS_FACTOR_RETURN where RMS_ID = 30000 and SUB_FACTOR_ID =:sid_arg
               and DT < :date_arg"""
        myDict = dict([('date_arg',self.date),('sid_arg',sid)])
        self.modelDB.dbCursor.execute(query,myDict)
        return self.modelDB.dbCursor.fetchall()[0][0].date()

#----------------------------Helper Functions-----------------------------# 
    def constructModelStructureTable(self): 
        attributeList = []
        #Determine what factor type do models in modelSelectorList have
        fTypeSet = set()
        if self.modelAssetType == "Universal":
            resultStructs = list() 
            factorTypeDict,factorNameDict = self.getFactorTypeDict()
            self.factorNameDict = factorNameDict
            fTypeList = sorted(set(factorTypeDict.values()))            
            factorsHistory = list() 
            dates = self.dates[-6:]
            dates = dates[::-1]
            for date in dates:
                if date == self.date: 
                    self.validFactorsForDate = self.getFactorsForDate(date)
                factorsHistory.append(self.getFactorsForDate(date))
            data = Utilities.Struct()
            data.header = ["FI Factors (expected #)"]
            data.header.extend([date.strftime('%b %d')for date in dates])
            resultList = list() 
            resultList.append(['All Factors (%d)'%len(self.factors)]+\
                                  [len(factors) for factors in factorsHistory])            
            standard = [factorTypeDict[self.factorSIDMap.get(factor.name)] for factor in self.factors]
            safeTypes = ['Commodity','Country','Currency','Industry','Local','Market','Style']
            FITypeList = list()
            for fType in fTypeList : 
                if fType not in safeTypes: 
                    FITypeList.append(fType)

            for fType in FITypeList:
                result = [fType + ' (%d)'%standard.count(fType)]
                for factors in factorsHistory:
                    result.append([factorTypeDict[sub_id] for sub_id in factors].count(fType))
                resultList.append(result)
            data.content = numpy.matrix(resultList)
            data.description = "Existing Factors in RMS_FACTOR_RETURN" 
            data.reportName = "Universal Model Structure Summary" 
            resultStructs.append(data) 

            data = Utilities.Struct()
            data.header = ["Other Factors (expected #)"]
            data.header.extend([date.strftime('%b %d')for date in dates])
            resultList = list() 
            for fType in safeTypes:
                result = [fType + ' (%d)'%standard.count(fType)]
                for factors in factorsHistory:
                    result.append([factorTypeDict[sub_id] for sub_id in factors].count(fType))
                resultList.append(result)
            data.content = numpy.matrix(resultList)
            data.reportName = "Universal Model Structure Summary Safer" 
            resultStructs.append(data) 

            entering= set(factorsHistory[0]).difference(set(factorsHistory[1]))
            leaving = set(factorsHistory[1]).difference(set(factorsHistory[0]))
            if len(entering) != 0 : 
                additions = Utilities.Struct() 
                additions.header = ["Factor Type", "Factor","Last Inclusion Date"]
                resultList  = list() 
                changedDict = dict() 
                for item in entering: 
                    changedDict.setdefault(factorTypeDict[item],list()).append(item)
                for changedType in sorted(changedDict.keys()): 
                    for item in changedDict[changedType]:
                        resultList.append([changedType,factorNameDict[item],
                                           self.getFactorLastAppearanceDate(item)])
                additions.content=numpy.matrix(resultList) 
                additions.reportName = "Joiner Factors" 
                additions.description = "%d Joiner Factors"%len(entering)
                resultStructs.append(additions)

            if len(leaving) !=0: 
                deletions = Utilities.Struct() 
                deletions.header = ["Factor Type", "Factor"]
                resultList  = list() 
                changedDict = dict() 
                for item in leaving: 
                    changedDict.setdefault(factorTypeDict[item],list()).append(item)
                for changedType in sorted(changedDict.keys()): 
                    for item in changedDict[changedType]:
                        resultList.append([changedType,factorNameDict[item]])
                deletions.reportName = "Leaver Factors" 
                deletions.content=numpy.matrix(resultList) 
                deletions.description = "%d Leaver Factors"%len(leaving)
                resultStructs.append(deletions)

            return resultStructs

        else: 
            for expM in self.expMList:
                for fType in expM.factorIdxMap_.keys():
                    if len(expM.factorIdxMap_[fType]) > 0:
                        fTypeSet.add(fType)
            fTypeList = sorted([fType.name for fType in fTypeSet])
        # Make Sure Statistical Factor, if exists, appears in the end 
            if 'Statistical' in fTypeList: 
                fTypeList.remove('Statistical') 
                fTypeList.append('Statistical')
            attributeList.extend(['RMS ID','Factors']+fTypeList+['Numeraire'])

            resultList = [] 
            for rIdx, rm in enumerate(self.modelSelectorList):
                result = []
                # Report on factor structure
                rm.setFactorsForDate(self.date,self.modelDB)
                result.extend([rm.mnemonic[2:],rm.rms_id,len(rm.factors)])
                if self.modelAssetType in ["Equity","Future"]:
                    expM = self.expMList[rIdx]
                    fTypeNameMap = dict([(j.name, j) for j in expM.factorTypes_])
                    for fTypeName in fTypeList:
                        fType = fTypeNameMap[fTypeName]
                        if len(expM.factorIdxMap_[fType])!=0:
                            result.append(len(expM.factorIdxMap_[fType]))
                        else:
                            result.append(0)
                else: 
                    for fTypeName in fTypeList: 
                        result.append(len(fTypeList[fTypeName]))
                result.append(rm.numeraire.currency_code)
                resultList.append(result) 

            addremoveMatrix = numpy.zeros((2,len(self.modelSelectorList)))
            addDict = {} 
            removeDict = {} 
            for rIdx, rm in enumerate(self.modelSelectorList):
                # Check for factor additions/removals
                currFactors = [f.description for f in rm.factors]
                rm.setFactorsForDate(self.prevDate,self.modelDB)
                prevFactors = [f.description for f in rm.factors]
                rm.setFactorsForDate(self.date,self.modelDB)
                add = set(currFactors).difference(prevFactors)
                rem = set(prevFactors).difference(currFactors)
                if len(add) > 0: 
                    addremoveMatrix[0,rIdx]= 1
                    addDict[rIdx]= ','.join(item for item in list(add)) 
                if len(rem) > 0:
                    addremoveMatrix[1,rIdx]= 1
                    removeDict[rIdx]= ','.join(item for item in list(rem))

            addremoveMatrix = ma.masked_where(addremoveMatrix==1,addremoveMatrix) 
            add = numpy.flatnonzero(ma.getmaskarray(addremoveMatrix[0,:]))
            if len(add)!=0:
                attributeList.append('Factors Added')
                for idx,result in enumerate(resultList):
                    if idx in add: 
                        result.append(addDict[idx])
                    else:
                        result.append("None") 

            remove = numpy.flatnonzero(ma.getmaskarray(addremoveMatrix[1,:]))
            if len(remove)!=0: 
                attributeList.append('Factors Removed') 
                for idx,result in enumerate(resultList):
                    if idx in remove: 
                        result.append(removeDict[idx])
                    else: 
                        result.append("None") 

            data = Utilities.Struct()
            data.reportName = "Model Structure Summary" 
            data.header = ["Model"] + attributeList 
            data.content = numpy.matrix(resultList)
            return [data] 

    def getSubIssueFromThruDates(self, sidList):
        if len(sidList) == 0:
            return list()
        INCR = 100
        sidArgList = [('sid%d' % i) for i in range(INCR)]
        query = """SELECT s.sub_id, s.from_dt, s.thru_dt, g.mnemonic
                   FROM sub_issue s, risk_model_group g
                   WHERE s.sub_id IN (%(sids)s) AND s.rmg_id = g.rmg_id""" % {
                'sids': ','.join([(':%s' % i) for i in sidArgList])}
        sidStrs = [sid.getSubIDString() for sid in sidList]
        defaultDict = dict((i, None) for i in sidArgList)
        sidInfo = list()
        for sidChunk in listChunkIterator(sidStrs, INCR):
            myDict = dict(defaultDict)
            myDict.update(dict(zip(sidArgList, sidChunk)))
            self.modelDB.dbCursor.execute(query, myDict)
            result = self.modelDB.dbCursor.fetchall()
            sidInfo.extend([(ModelDB.SubIssue(r[0]), 
                             r[1].date(), r[2].date(), r[3]) for r in result])
        return sidInfo 

    def fetchInfoForLeavers(self,sidList):
        if len(sidList) == 0: 
            logging.info ("No Asset Leaving Model")
            return 
        data = Utilities.Struct()
        data.universe = sidList
        subIssueInfo = self.getSubIssueFromThruDates(sidList)
        for (id,fromDt,thruDt,rmg) in subIssueInfo: 
            lastDt = thruDt - datetime.timedelta(1)
            modelID = id.getModelID()
    
            if self.modelAssetType == "Equity":
                # Find the Leaver Asset Cap 
                capdata = Utilities.Struct()
                capdata.universe = [id]
                capdata.assetIdxMap = dict([(j,i) for (i,j) in enumerate(capdata.universe)])
                #Assume all models belong to the same family
                riskModel = self.modelSelectorList[0]
                if getattr(riskModel, 'getRMDates', None) is not None:
                    (mcapDates, goodRatio) = riskModel.getRMDates(
                        lastDt, modelDB, 20, ceiling=False)
                else:
                    (mcapDates, goodRatio) = riskModel.getRMDatesLegacy(
                        lastDt, modelDB, 20, ceiling=False)
                capdata.marketCaps = ma.filled(modelDB.getAverageMarketCaps(
                    mcapDates, capdata.universe, riskModel.numeraire.currency_id, marketDB), 0.0)
                assetCap = ma.take(capdata.marketCaps, [0])
                self.assetCapDict[id]=ma.sum(assetCap)/1e6

                nameMap = self.modelDB.getIssueNames(
                    lastDt,[id.getModelID()],self.marketDB)
                cusipMap = self.modelDB.getIssueCUSIPs(
                    lastDt,[id.getModelID()],self.marketDB)
                self.cusipMap[modelID]=cusipMap.get(id.getModelID())

                sedolMap = self.modelDB.getIssueSEDOLs(
                    lastDt,[id.getModelID()],self.marketDB)
                self.sedolMap[modelID]=sedolMap.get(id.getModelID())

                isinMap = self.modelDB.getIssueISINs(
                    lastDt,[id.getModelID()],self.marketDB)

                self.isinMap[modelID]=isinMap.get(id.getModelID())

                tickerMap = self.modelDB.getIssueTickers(
                    lastDt,[id.getModelID()],self.marketDB)

                self.hcMap[modelID]  = (AssetProcessor.get_asset_info(lastDt,[id.getModelID()], self.modelDB, self.marketDB,
                                                       'REGIONS', 'HomeCountry')).get(id.getModelID())
            else: 
                nameMap = self.modelDB.getFutureNames(lastDt,[id.getModelID()],self.marketDB)
                ricMap  = modelDB.getIssueRICQuote(self.prevDate,[id.getModelID()],marketDB)
                bbTickerMap = self.getFutureBBTickers(self.prevDate, [id.getModelID()], marketDB)
                self.ricMap[modelID] = ricMap.get(id.getModelID())
                self.bbTickerMap[modelID] = bbTickerMap.get(id.getModelID())
                tickerMap = modelDB.getFutureTickers(lastDt,[id.getModelID()], self.marketDB)
            

            self.nameMap[modelID]=nameMap.get(id.getModelID())
            self.tickerMap[modelID]=tickerMap.get(id.getModelID())
            self.assetRmgMap[modelID]= rmg 
        return data 

    def getMarketID(self,idList): 
        # Returns ModelID - MarketID Map 
        assert(len(idList)>0) 
        idList = [sid.getSubIDString()[:-2] for sid in idList]
        modelDB = self.modelDB
        modelMarketMap = {} 
        sidArgList = [('sid%d' % i) for i in range(len(idList))]
        query = """SELECT a.modeldb_id,a.marketdb_id
                   FROM issue_map a WHERE a.modeldb_id in (%s) AND 
                   a.thru_dt = (SELECT max(thru_dt) FROM issue_map b 
                   WHERE b.modeldb_id = a.modeldb_id)
                   """ % ','.join([(':%s' % i) for i in sidArgList])
        valueDict = dict(zip(sidArgList, idList))
        modelDB.dbCursor.execute(query, valueDict)
        results = modelDB.dbCursor.fetchall()
        for r in results:
            modelMarketMap[ModelID.ModelID(string=r[0])]=MarketID.MarketID(string=r[1])
        return modelMarketMap 
    
    def extendModelMarketMap(self,modelMarketMap): 
        for key in modelMarketMap.keys(): 
            self.modelMarketMap[key] = modelMarketMap[key] 

    def checkMissingReason(self,sidList,missingAssetModelMap):
        if len(sidList) == 0: 
            return None
        missing = sidList
        missingModelDict = missingAssetModelMap 
#        print missingModelDict,"MISSING MODEL DICT"
        assetMissingReasonDict = {} 
        for i in missing: 
            vals = [] 
            for model in missingModelDict[i]: 
                rval = Utilities.Struct()
                modelID = ModelID.ModelID(string = i.getSubIDString()[:-2])
                if modelID not in self.modelMarketMap:
                    assetMissingReasonDict.setdefault(i,dict()).update({self.idxMnemonicMap[model]:\
                                                                           "Unknown Reason"})
                    continue
                rval.axiomaid = self.modelMarketMap[modelID].getIDString() 
                rval.date = self.date 
                rval.sedol=''
                rval.cusip = ''
                rval.country = ''
                rval.model = 'AX' + model
                vals.append(rval) 
            for idx,val in enumerate(vals): 
                (idVal,reason) = checkCoverage.checkCoverage(self.marketDB, self.modelDB, val)
                if "." in reason: 
                    reason = reason.split('.')[0]
                if 'Asset type is ETF' in reason: 
                    query = """ 
select FAMILY_ID,DIST_FROM_DT,DIST_THRU_DT from COMPOSITE_MEMBER_FAMILY_MAP 
    where MEMBER_ID = (select id from COMPOSITE_MEMBER where AXIOMA_ID = '%s'
    and from_dt <:myDate and thru_dt >:myDate)
                            """%val.axiomaid 
                    self.marketDB.dbCursor.execute(query,dict([('myDate',self.date)]))
                    result = self.marketDB.dbCursor.fetchall()
                    if len(result) == 0:
                        reason = 'Asset type is ETF.NOT DISTRIBUTING'
                    else: 
                        flag = False
                        for row in result: 
                            if row[1].date()<=self.date<row[2].date(): 
                                flag = True 
                                break 
                        if flag is False: 
                            reason = 'Asset type is ETF.NOT DISTRIBUTING'
                assetMissingReasonDict.setdefault(i,dict()).update(
                    {self.idxMnemonicMap[missingModelDict[i][idx]]:reason})
        return  assetMissingReasonDict

    def checkModelChange(self,assetList,assetDict):
        assert(len(assetList) == len(assetDict))
        modelMatrix = numpy.zeros((len(assetDict),len(self.modelSelectorList)))
        for idx, asset in enumerate(assetList):
            for m in assetDict[asset]: 
                modelMatrix[idx][self.idxMnemonicMap[m]]=1 
        modelMatrix = ma.masked_where(modelMatrix!=1,modelMatrix) 
        focusColumnIdx = [] 
        for idx in range(len(self.modelSelectorList)): 
            focusColumn = numpy.flatnonzero(ma.getmaskarray(modelMatrix[:,idx]))
            if len(focusColumn!=0):
                focusColumnIdx.append(idx) 
        return (focusColumnIdx,modelMatrix)  

    def findJoinerLeaver(self,univList,punivList): 
        assert(len(univList)==len(punivList))
        totalJoiner = set() 
        totalLeaver = set()
        assetJoinerDict = {} 
        assetLeaverDict = {} 
        for idx,univ in enumerate(univList):
            puniv = punivList[idx] 
            if univ != puniv: 
                new = set(univ).difference(puniv)
                gone = set(puniv).difference(univ)
                totalJoiner = totalJoiner.union(new) 
                totalLeaver = totalLeaver.union(gone) 
                newList = list(new)
                goneList = list(gone)
                if len(newList)!=0: 
                    for asset in newList:
                        assetJoinerDict.setdefault(asset,list()).append(\
                            self.modelSelectorList[idx].mnemonic[2:])
                if len(goneList)!=0:
                    for asset in goneList:
                        assetLeaverDict.setdefault(asset,list()).append(\
                            self.modelSelectorList[idx].mnemonic[2:])
        joinerList = list(totalJoiner) 
        leaverList = list(totalLeaver) 
        return(joinerList,leaverList,assetJoinerDict,assetLeaverDict)

    def findMissingBMCM(self):
        universeCheckerList = [] 
        missingBenchmark = {} 
        missingComposite = {} 
        terminated = None 
        for idx, rm in enumerate(self.modelSelectorList): 
            universe_checker = UniverseChecker(rm,modelDB,marketDB,self.date,
                                               self.univ[idx],list(),list()) 
            universeCheckerList.append(universe_checker)         
        for idx, universeChecker in enumerate(universeCheckerList): 
            missingBM,terminated = universeChecker.getMissingBenchmark(showAllTerminated=True)
            missingCM = list(universeChecker.getMissingComposite())
            if len(missingBM) > 0: 
                missingBM = [ModelDB.SubIssue(iid.getIDString() + "11") for iid in missingBM]
                for bm in missingBM: 
                    missingBenchmark.setdefault(bm, list()).append(self.modelmnemonicList[idx])
            if len(missingCM) > 0: 
                missingCM = [ModelDB.SubIssue(iid.getIDString() + "11") for iid in missingCM]
                for cm in missingCM: 
                    missingComposite.setdefault(cm, list()).append(self.modelmnemonicList[idx])
        return (missingBM,missingCM,missingBenchmark,missingComposite,terminated)

    def extendJoinerLeaverInfo(self,joinerList,leaverList,findIndustry=False ): 
        if not hasattr(self,'fromDateDict'):
            self.fromDateDict = {}             
        if not hasattr(self,'thruDateDict'):
            self.thruDateDict = {} 

        if len(joinerList+leaverList) == 0: 
            return 
        sidArgList = [('sid%d' % i) for i in range(len(joinerList+leaverList))]
        query = """SELECT sub_id, from_dt, thru_dt
                   FROM sub_issue
                   WHERE sub_id IN (%s)""" % ','.join([(':%s' % i) for i in sidArgList])
        valueDict = dict(zip(sidArgList, [sid.getSubIDString() for sid in joinerList+leaverList]))
        modelDB.dbCursor.execute(query, valueDict)
        results = modelDB.dbCursor.fetchall()
        
        fromDateDict = dict((ModelDB.SubIssue(r[0]), r[1].date()) for r in results)
        thruDateDict = dict((ModelDB.SubIssue(r[0]), r[2].date()) for r in results)
        self.fromDateDict.update(fromDateDict) 
        self.thruDateDict.update(thruDateDict) 
        # Get industry classification for joiner
        if findIndustry and self.modelAssetType=='Equity': 
            if not hasattr(self,'sidClsMap'):
                self.sidClsMap={} 

            sidClsMap = dict((sid, cls.classification.description) \
                   for (sid, cls) in rm.industryClassification.\
                        getAssetConstituents(modelDB, joinerList,
                        self.date).items())
            self.sidClsMap.update(sidClsMap) 

    def createAssetRmgMap(self, sids): 
        rmgList = set()
        for rm in self.modelSelectorList:
            for rmg in rm.rmg:
                rmgList.add(rmg)
        self.rmgList = list(rmgList)

        allRmg = modelDB.getAllRiskModelGroups()

        rmgIdMap = dict([(rmg.rmg_id, rmg.mnemonic) for rmg in allRmg])
        rmgAssetMap = AssetProcessor.loadRiskModelGroupAssetMap(
                    self.date, sids,allRmg, modelDB, marketDB, False)    

        assetRmgMap = dict()
        for rmg, sidList in rmgAssetMap.items():
            for sid in sidList:
                assetRmgMap[sid.getModelID()] = rmgIdMap[rmg]
        return assetRmgMap 
    
    def prepAssetLevelInfo(self,sidList,extraHeader=None,extraDict = None,
                               sidModelDict=None,appendIndustry=True,appendAssetType=True,
                           modelAssetType='Equity'):

        assert(len(sidList)!=0) 
        localExtraHeader = list() 
        for header in extraHeader: 
            localExtraHeader.append(header) 
        if modelAssetType == 'Equity':
            baseHeader = ["#","MODELID","MARKETID","NAME","SEDOL/CUS.", "TICKER","RMG","HQ"]
        else: 
            baseHeader = ["#","MODELID","MARKETID","NAME","RIC","TICKER","BB TICKER","RMG"]
        headerList = baseHeader

        # always appending industry for asset-level info display at the end by default 
        rm = self.modelSelectorList[0]
        if appendIndustry:
            localExtraHeader.append("INDUSTRY")
            if not hasattr(self,'sidClsMap'):
                self.sidClsMap={} 
            sidClsMap = dict((sid, cls.classification.description) \
                   for (sid, cls) in rm.industryClassification.\
                        getAssetConstituents(modelDB, sidList,
                        self.date).items())
            self.sidClsMap.update(sidClsMap) 

        # always appending asset type for asset-level info display at the end by default 
        if appendAssetType: 
            localExtraHeader.append("ASSET TYPE")
            if not hasattr(self,'sidAssetTypeMap'):
                self.sidAssetTypeMap={} 
            clsFamily = marketDB.getClassificationFamily('ASSET TYPES')
            assert(clsFamily is not None)
            clsMembers = dict([(i.name, i) for i in self.marketDB.\
                                   getClassificationFamilyMembers(clsFamily)])
            cm = clsMembers.get('Axioma Asset Type')
            assert(cm is not None)
            clsRevision = self.marketDB.\
                getClassificationMemberRevision(cm, self.date)
            homeClsData = self.modelDB.getMktAssetClassifications(
                clsRevision, sidList, self.date, self.marketDB)
            secTypeDict = dict([(i, j.classification.description \
                                     ) for (i,j) in homeClsData.items()])
            for key,value in secTypeDict.items(): 
                if value == 'Common Stock': 
                    secTypeDict[key] = ''
            self.sidAssetTypeMap.update(secTypeDict)

        if localExtraHeader and extraDict is None: 
            for header in localExtraHeader: 
                if "CURR" in header: 
                    headerList.append("CURR")
                elif "PREV" in header: 
                    headerList.append("PREV") 
                else: 
                    headerList.append(header) 
            
        # Matix to keep check if any asset fails to apply to one model 
        focusColumnIdx = [] 
        if sidModelDict: 
            (focusColumnIdx,modelMatrix) = self.checkModelChange(sidList,sidModelDict) 
        if extraDict is not None: 
            focusColumnIdx = list(set(range(len(self.modelSelectorList)))-set(focusColumnIdx))
        if len(focusColumnIdx)!=0: 
            for idx in focusColumnIdx: 
                headerList.append(self.modelmnemonicList[idx])
        focusAssoMap = dict([(j,i)for (i,j) in enumerate(focusColumnIdx)])

#        print focusColumnIdx,headerList, "FocusColumnIdx,HeaderList" 
        resultList = []  

        for idx,asset in enumerate(sidList): 
            modelID = asset.getModelID()
            rmg = self.assetRmgMap.get(modelID)
            if self.modelAssetType == "Equity":
                hc = self.hcMap.get(modelID) 
                isin = self.isinMap.get(modelID)
                if isin is not None : 
                    if isin[:2]!= rmg: 
                        hq = isin[:2]
                    else: 
                        hq = ''
                else: 
                    hq = '/N'

                if hc != rmg: 
                    rmg = '%s/%s'%(rmg,hc) 
            rmg = str(rmg) 
            marketID = self.modelMarketMap.get(modelID)
            if marketID is not None: 
                marketIDString = marketID.getIDString() 
            else: 
                marketIDString = "/N"
            if self.modelAssetType == "Equity":
                result = [idx+1,modelID.getIDString(),
                          marketIDString,self.nameMap.get(modelID),
                          self.sedolMap.get(modelID),self.tickerMap.get(modelID),
                          rmg,hq] 
            else: 
                result = [idx+1,modelID.getIDString(),
                          marketIDString,self.nameMap.get(modelID),
                          self.ricMap.get(modelID),
                          self.tickerMap.get(modelID),
                          self.bbTickerMap.get(modelID),
                          rmg] 

            if extraDict is not None: 
                assetModelDict=extraDict.get(asset)
                withReason= set(assetModelDict.keys()).intersection(set(focusColumnIdx))
                # print assetModelDict.keys(),"AssetModelDict Keys"
                # print focusColumnIdx,"FocusColumnIdx"
                # print withReason, "withReason"
                Indicator =['*']*len(focusColumnIdx)
                for i in assetModelDict.keys():
                    if  i in focusAssoMap.keys():
                        Indicator[focusAssoMap[i]]=assetModelDict[i]
                result.extend(Indicator)
#                print result 
            else: 
                for header in localExtraHeader: 
                    columnDict = self.headerMap.get(header)
                    if columnDict is None: 
                        logging.info("Ignoring Header %s, No Corresponding Dict"\
                                         %header)
                    else: 
                        if "CAP" in header or "WGT" in header: 
                            result.append("%.2f"%columnDict.get(asset))
                        else: 
                            result.append(columnDict.get(asset) if columnDict.get(asset) is not None 
                                          else '--')

                if len(focusColumnIdx)>0: 
                    modelIndicator = list(ma.take(modelMatrix[idx,:],focusColumnIdx))
                    Indicator = ['*' if i is not ma.masked else '--' for i in modelIndicator]
                    result.extend(Indicator)                     
            resultList.append(result) 
        data = Utilities.Struct() 
        data.header = headerList 
        data.content = numpy.matrix(resultList)
        return data 

    def createDataBundle(self,extraAssets):
        self.fullsubID = list(self.wholeUniv) 
        self.assetIdxMap = dict([(j,i) for (i,j) in enumerate(self.fullsubID)])
        self.modelMarketMap = dict(modelDB.getIssueMapPairs(self.date))
        #Construct the Univ Assets Cap Dict
        logging.info("Getting Asset Cap Dict for the Whole Universe") 
        data = Utilities.Struct()
        data.universe = self.fullsubID
        riskModel = self.modelSelectorList[0]
        if getattr(riskModel, 'getRMDates', None) is not None:
            (mcapDates, goodRatio) = riskModel.getRMDates(
                self.date, modelDB, 20, ceiling=False)
        else:
            (mcapDates, goodRatio) = riskModel.getRMDatesLegacy(
                self.date, modelDB, 20, ceiling=False)
        self.marketCaps = ma.filled(modelDB.getAverageMarketCaps(
            mcapDates, data.universe, riskModel.numeraire.currency_id, marketDB), 0.0)
        self.assetCapDict = dict(zip(data.universe,self.marketCaps/1e6))
        # ids is modelID List 
        ids = [iid for iid in self.modelMarketMap.keys() if \
                   ModelDB.SubIssue(iid.getIDString() + "11") in set(self.wholeUniv)]
        extraAssets = [ModelID.ModelID(string = sid.getSubIDString()[:-2]) for sid in extraAssets]
        ids = list(set(ids).union(set(extraAssets)))
        sids = [ModelDB.SubIssue(iid.getIDString() + "11") for iid in ids]

        if self.modelAssetType == "Equity":
            self.nameMap = modelDB.getIssueNames(self.date, ids, marketDB)
            logging.info( "Created NameMap for the whole Universe + Missing Assets") 
    
            self.tickerMap = modelDB.getIssueTickers(self.date, ids, marketDB)
            logging.info("Created tickerMap for the whole Universe + Missing Assets") 
        
            self.cusipMap = modelDB.getIssueCUSIPs(self.date, ids, marketDB)
            logging.info("Created cusipMap for the whole Universe + Missing Assets") 

            self.sedolMap = modelDB.getIssueSEDOLs(self.date, ids, marketDB)
            logging.info("Created sedolMap for the whole Universe + Missing Assets") 

            self.isinMap = modelDB.getIssueISINs(self.date, ids, marketDB)
            logging.info("Created IsinMap for the whole Universe + Missing Assets") 
        
            self.hcMap = AssetProcessor.get_asset_info(self.date, ids, modelDB, marketDB,
                                                   'REGIONS', 'HomeCountry')
        if self.modelAssetType == 'Future': 
            self.nameMap = modelDB.getFutureNames(self.date, ids, marketDB)
            self.tickerMap = modelDB.getFutureTickers(self.date, ids, marketDB)
            self.ricMap = modelDB.getIssueRICQuote(self.date,ids,marketDB)
            self.bbTickerMap = self.getFutureBBTickers(self.date, ids, marketDB)
            logging.info("Created Maps for Future")

        self.assetRmgMap = self.createAssetRmgMap(sids) 

        self.mnemonicPosMap =dict([(model,position) for (position,model) in\
                                  enumerate(self.modelmnemonicList)])


    def compareEstimationUniverse(self,sectionName): 
        wholeEstUniv =set() 
        intersect = set(self.estUniv[0]) 
        universeIndicator = True 
        for idx,estUniv in enumerate(self.estUniv): 
            wholeEstUniv= wholeEstUniv.union(set(estUniv))
            intersect = intersect.intersection(set(estUniv))
        
        data = Utilities.Struct() 
        data.reportName = "Universe Descrepency Table" 
        data.reportSection = sectionName 
        
        # Universe Not Identical 
        if intersect!= wholeEstUniv:
            missingAssets = list(wholeEstUniv.difference(intersect))
            if len(missingAssets)<=10:
                headerList = ['Missing Asset']            
                headerList.extend(self.modelmnemonicList) 
                resultList = [] 
                for asset in missingAssets: 
                    missingIndicator = [] 
                    for univ in self.estUniv: 
                        if asset in univ: 
                            missingIndicator.append('*') 
                        else: 
                            missingIndicator.append('-') 
                    resultList.append([asset.getModelID().getIDString()] \
                                          + missingIndicator)
                data.header = headerList              
                data.content=numpy.matrix(resultList) 
            data.description  = "Estimation Universes are not Identical"
            universeIndicator = False  
        else: 
            data.description = "%d Estimation Universes are Identical"\
                %len(self.modelmnemonicList)
        self.coreDataList.append(data) 
        return (universeIndicator,wholeEstUniv)

    def getAssetCapDict(self,sidList,date):
        data = Utilities.Struct()
        data.universe = list(sidList) 
        assetIdxMap = dict([(j,i) for (i,j) in enumerate(data.universe)])
        logging.info("Getting Asset Cap Dict for the Previous Day Estimation Universe") 
        riskModel = self.modelSelectorList[0]
        if getattr(riskModel, 'getRMDates', None) is not None:
            (mcapDates, goodRatio) = riskModel.getRMDates(
                date, modelDB, 20, ceiling=False)
        else:
            (mcapDates, goodRatio) = riskModel.getRMDatesLegacy(
                date, modelDB, 20, ceiling=False)
        prevMarketCaps = ma.filled(modelDB.getAverageMarketCaps(
            mcapDates, data.universe, riskModel.numeraire.currency_id, marketDB), 0.0)
        assetCapDict = {} 
        for asset in sidList:
            assetIdx = assetIdxMap.get(asset)
            assetCap = ma.take(prevMarketCaps, [assetIdx])
            assetCapDict[asset]= ma.sum(assetCap)/1e6
        return assetCapDict

    def getFutureBBTickers(self,date, ids, marketDB):
        """Returns an issue to bloomberg ticker map for the given issues valid on
        the given date.
        """
        INCR = 400
        resultDict = dict()
        bulkArgList = [('mid%d'%i) for i in range(INCR)]
        query ="""select c.modeldb_id, b.ID from
                  MARKETDB_GLOBAL.FUTURE_DIM_BLOOMBERG_TICKER b, \
                  MODELDB_GLOBAL.FUTURE_ISSUE_MAP c where 
                  c.modeldb_id IN(%(keys)s)
                  and b.axioma_id = c.marketdb_id 
                  and c.from_dt<=:date_arg and c.thru_dt >:date_arg
                  and b.change_dt = (select max(change_dt) from 
                  future_dim_bloomberg_ticker where axioma_id = b.axioma_id)
                  """%{'keys':','.join([':%s'% i for i in bulkArgList])}
        defaultDict= dict([(arg, None) for arg in bulkArgList])
        defaultDict['date_arg']=date
        aidStr = [i.getIDString() for i in ids]
        for aidChunk in listChunkIterator(aidStr, INCR):
            valueDict= dict(defaultDict)
            valueDict.update(dict(zip(bulkArgList, aidChunk)))
            marketDB.dbCursor.execute(query, valueDict)
            for  id, code  in marketDB.dbCursor.fetchall():
                resultDict[MarketID.MarketID(string=id)] =  code
        return resultDict        

    def prepTopAssetInfo(self,num,estUnivList,prevEstUnivList,estuCap): 
        currTop = numpy.array([self.assetCapDict.get(sid) for sid in estUnivList])
        estuRank = numpy.argsort(-currTop)
        self.estuRankDict = dict([(estUnivList[j],round(float(i)/len(estUnivList),3))
                                  for (i,j) in enumerate(estuRank)])
        self.headerMap["ESTU RANK"] = self.estuRankDict 
        currRank = numpy.argsort(-currTop)[:num]
        currRankIdx = dict([(estUnivList[j],i) for (i,j) in enumerate(currRank)])
        wgtMap = dict() 
        prevAssetCapDict = self.getAssetCapDict(prevEstUnivList,self.prevDate)
        prevTop = numpy.array([prevAssetCapDict.get(sid) for sid in prevEstUnivList])
        prevRank = numpy.argsort(-prevTop)[:num] 
        prevRankIdx = dict([(prevEstUnivList[j],i) for (i,j) in enumerate(prevRank)])
        topAssets = [estUnivList[j] for j in currRank]
        rankDiff = {} 
        for asset in currRankIdx: 
            wgt = self.assetCapDict.get(asset)/(estuCap *10) 
            wgtMap[asset] = wgt 
            if asset in prevRankIdx: 
                rankDiff[asset] = prevRankIdx[asset] - currRankIdx[asset]
            else: 
                rankDiff[asset] = "--" 
        self.headerMap["RANK DIFF"] = rankDiff 
        self.headerMap['WGT %'] = wgtMap
        extraHeader = ["CAP (%s m)"%self.currency,"WGT %", "RANK DIFF"]
        data = self.prepAssetLevelInfo(topAssets,extraHeader)
        return data 

    def prepIndustryDemographics(self,industryAssetMap):
        industryCapDict = dict() 
        totalCap = 0.0 
        for industry,assets in industryAssetMap.items(): 
            cap = 0.0 
            for asset in assets: 
                cap += self.assetCapDict.get(asset) 
            totalCap += cap 
            industryCapDict[industry] = cap 
        industryWeightDict = dict([(industry,cap/totalCap) for (industry,cap) in industryCapDict.items()])
        resultList = list() 

        rank = numpy.argsort(- numpy.array(list(industryWeightDict.values()))).tolist()
        industries = [list(industryWeightDict.keys())[i] for i in rank]
        resultList = [] 

        for idx, industry  in enumerate(industries):
            resultList.append([idx+1,industry,'%.3f'%industryWeightDict.get(industry)])

        data=Utilities.Struct() 
        data.description = "Estimation Universe Industry Distribution"
        data.header = ['#','INDUSTRY','WGT %'] 
        data.content=numpy.matrix(resultList) 
        data.reportName = "Industry Demographics" 
        data.reportType = "PieChart" 
        return data 

    def prepIndustryThicknessInfo(self,validEstUnivList):
        expM = self.expMList[0]
        matrix = expM.getMatrix() 
        IndustryFactor = Matrices.FactorType('Industry', 'Industry classification')
        industryFactors = expM.factorIdxMap_.get(IndustryFactor)
        thicknessMap = {} 
        industryAssetMap = {} 
        sids = expM.getAssets() 
        sidIdx = dict([(sid,idx) for (idx,sid) in enumerate(sids)]) 
        validEstUnivList = list(set(validEstUnivList).intersection(set(sids)))
        estuIdx = [sidIdx[sid] for sid in validEstUnivList] 

        for factor in industryFactors.keys(): 
            factorExposure = matrix[industryFactors[factor],:]
            masked = numpy.flatnonzero(ma.getmaskarray(factorExposure))
            unmasked = list(set(estuIdx).difference(set(masked)))
            exposedAssets = [sids[idx] for idx in unmasked]
            industryAssetMap[factor] = exposedAssets 
            exposedValid = set(exposedAssets).intersection(validEstUnivList) 
            caps = [self.assetCapDict.get(sid) for sid in exposedAssets]
            srcaps = numpy.sqrt(caps) 
            wgt = numpy.array((srcaps)/sum(srcaps))
            HI = ma.inner(wgt,wgt) 
            thickness = 1/HI 
            if thickness < 10.0: 
                thicknessMap[factor]=thickness
        if len(thicknessMap)>0:
            rank = numpy.argsort(list(thicknessMap.values())).tolist()
            thinIndustries = [list(thicknessMap.keys())[i] for i in rank]
            resultList = [] 
            for idx,industry in enumerate(thinIndustries):
                result = [industry,
                          '%.2f'%thicknessMap[industry],
                          len(industryAssetMap.get(industry))]
                resultList.append(result) 
            data = Utilities.Struct() 
            data.header=['THIN INDUSTRY','EFFECTIVE NUM','REAL NUM'] 
            data.content=numpy.matrix(resultList)
            return (data,industryAssetMap)
        else: 
            return (None,industryAssetMap) 

    def getFuturesDistribution(self): 
        query = """
           select DESCRIPTION as country, count(distinct future_familY_ID) as number_of_series , count(distinct marketdb_ID) as number_of_contracts
           From Modeldb_global.RMI_UNIVERSE universe, Modeldb_global.sub_issue si, Modeldb_global.risk_model_group rmg, 
           marketdb_global.future_linkage_active fla, modeldb_global.future_issue_map fim  where rms_ID = 10001 and dt =:dt_arg and si.sub_id = universe.SUB_ISSUE_ID
           and si.FROM_DT <= universe.dt and universe.dt < si.thru_DT and si.RMG_ID = rmg.RMG_ID 
          and fim.modeldb_ID = ISSUE_ID and marketdb_ID = FUTURE_AXIOMA_ID
          group by DESCRIPTION order by number_of_series desc
          """
        self.marketDB.dbCursor.execute(query,dict([('dt_arg',self.date)]))
        result = self.marketDB.dbCursor.fetchall()
        self.marketDB.dbCursor.execute(query,dict([('dt_arg',self.prevDate)]))
        prevResult = self.marketDB.dbCursor.fetchall()
        prevComDict = dict() 
        if len(prevResult) != 0: 
            for item in prevResult: 
                prevComDict[item[0]] = (item[1],item[2])
        if len(result) != 0: 
            data=Utilities.Struct() 
            data.description = "Futures Universe Composition"
            data.reportName = "Futures Universe Composition" 
            data.header = ['COUNTRY','SERIES # ','PREV SERIES # ','CONTRACTS # ', 'PREV CONTRACTS # ' ] 
            resultList = list()
            for entry in result: 
                resultList.append([entry[0],entry[1],prevComDict.get(entry[0])[0] if prevComDict.get(entry[0]) is not None 
                                   else 0, entry[2],prevComDict.get(entry[0])[1] if prevComDict.get(entry[0]) is not None 
                                   else 0])
            data.content=numpy.matrix(resultList) 
            return data 
        return None 
            
    def prepCountryInfo(self,estUnivList,ignoreList,estuCap):
            isoWeightMap = dict() 
            rmgAssetMap = dict() 
            thicknessMap = dict() 
            rmgMnemonicList = list() 
            for r in self.modelSelectorList[0].rmg:
                isoWeightMap[r.mnemonic] = 0.0
                rmgMnemonicList.append(r.mnemonic)
            for asset in estUnivList:
                rmg = self.assetRmgMap.get(asset.getModelID())
                hcISO = self.hcMap.get(asset.getModelID())
                if rmg is None:
                    continue
                else:
                    if self.modelAssetType == "Equity":
                        if hcISO is None or hcISO not in rmgMnemonicList:
                            continue
                        else: 
                            isoWeightMap[hcISO] += self.assetCapDict.get(asset)
                    else: 
                        isoWeightMap[rmg] += self.assetCapDict.get(asset)
                    rmgAssetMap.setdefault(rmg,list()).append(asset) 
            for r in self.rmgList: 
                # In Percentage 
                isoWeightMap[r.mnemonic] = isoWeightMap[r.mnemonic]/(estuCap * 10) 
                if r.mnemonic in rmgAssetMap.keys(): 
                    caps = [self.assetCapDict.get(sid) if self.assetCapDict.get(sid)\
                           is not None else 0.0 for sid in set(rmgAssetMap[r.mnemonic]).difference(set(ignoreList))]
                else: 
                    caps = [0.0]
                rtcaps = numpy.sqrt(caps)
                rmgTotal = sum(rtcaps)
                wgt = numpy.array(rtcaps/sum(rtcaps))
                HI = ma.inner(wgt,wgt) 
                thickness = 1/HI 
                if thickness < 10.0 :
                    thicknessMap[r.mnemonic] = thickness 

            rank = numpy.argsort(list(isoWeightMap.values())).tolist()
            rank.reverse()
            resultList = [] 
            othersWgt = 100

            for (i,j) in enumerate(rank[:10]):
                rmg = self.rmgInfoMap.get(list(isoWeightMap.keys())[j])
                wgt = isoWeightMap[rmg.mnemonic]
                resultList.append([i+1,rmg.description+(' (%.2f)'%wgt),
                                   '%.2f'%wgt])
                othersWgt -= wgt 
            resultList.append([i+2,"others",'%.2f'%othersWgt])
            data=Utilities.Struct() 
            data.description = "Top Percentage Countries"
            data.header = ['#','COUNTRY','WGT %'] 
            data.content=numpy.matrix(resultList) 
            data.reportName = "Top Countries" 
            data.reportType = "PieChart" 
            
            rank = numpy.argsort(list(thicknessMap.values())).tolist()
            thinCountries = [list(thicknessMap.keys())[i] for i in rank]
            resultList = [] 
            for idx,country in enumerate(thinCountries):
                rmg = self.rmgInfoMap.get(country)
                result = [rmg.description+'('+rmg.mnemonic+')',
                          '%.2f'%thicknessMap[country],
                          len(rmgAssetMap.get(country))]
                resultList.append(result) 
            if len(resultList)!=0:
                data2 = Utilities.Struct() 
                data2.header=['THIN COUNTRY','EFFECTIVE NUM','REAL NUM']             
                data2.content = numpy.matrix(resultList) 
                data2.description = "Country Thickness Table"
                data2.reportName = "Country Thickness Table" 
            else: 
                data2=None
            return [data,data2]

    def getRMSStatics(self,modelSelectorList,dateList): 
        rms_idList =[rm.rms_id for rm in modelSelectorList] 
        dates = dateList 
        staDictList = list() 
        for rmsid in rms_idList: 
            dateStaDict = dict() 
            dtArgList = [('dt%d' % i) for i in range(len(dates))]
            query = """SELECT dt,adjRsquared FROM rms_statistics
                       WHERE rms_id =%s  AND 
                       dt in (%s) AND adjRsquared is not null
                       """ % (rmsid,','.join([(':%s' % i) for i in dtArgList]))
            valueDict = dict(zip(dtArgList, dates))
            modelDB.dbCursor.execute(query, valueDict)
            results = modelDB.dbCursor.fetchall()
            for r in results:
                dateStaDict[r[0].date()] = r[1]
            staDictList.append(dateStaDict) 
        return staDictList 

    def returnRegressionStatistics(self) : 
        logging.info("Processing R-Squared Stat")
        sectionName = "REGRESSION DIAGNOSTICS"
        modelDB = self.modelDB
        dateList = modelDB.getDates(self.rmgList,self.date,60,True) 
        r2DictList= self.getRMSStatics(self.modelSelectorList,dateList)
        description = "Past 60 days Adjusted R-Squared Chart" 
#        description = "Past 60 days R-Squared and Estu Return Chart" 
        resultList = [] 
        for idx,date in enumerate(dateList): 
            # rt = self.estuReturnHistory.data[:,idx].filled(0.0)
            # marketrt = numpy.inner(rt,self.estuWgt)
            result = [date.strftime('%b %d')]
#            result = [date.strftime('%b %d'),round(marketrt,2)]
            for model in r2DictList: 
                if model.get(date) is not None:
                    result.append('%.2f'%model.get(date))
            if len(result) == len(self.modelSelectorList)+1:
                resultList.append(result) 
        data= Utilities.Struct() 
        data.reportName = "R-Squared Elaboration" 
        data.reportSection = sectionName 
#        data.header = ["Date","ESTU Return %"]+self.modelmnemonicList
        data.header = ["Date"]+self.modelmnemonicList
        data.content = numpy.matrix(resultList) 
#        data.reportType = "ComboChart" 
        data.reportType = "LineChart"
        data.description = "Past 60 days R-Squared Chart" 
#        data.description = "Past 60 days R-Squared and Estu Return Chart" 
        self.dataList.append(data) 
        logging.info("Finished Processing R-Squared Stat")

    def constructFactorTypeMap(self):
        factorTypeMap = dict()
        factorTypes = set() 
        typeFactorMap = dict() 
        for idx,expM in enumerate(self.expMList):
            factors = self.modelSelectorList[idx].factors 
            for fType in expM.factorIdxMap_.keys():
                if len(expM.factorIdxMap_[fType]) > 0:
                    for fName in expM.factorIdxMap_[fType].keys():
                        factorTypeMap[fName] = fType 
                        typeFactorMap.setdefault(fType.name,list()).append(\
                           factors[expM.factorIdxMap_[fType][fName]])
        for type in typeFactorMap: 
            typeFactorMap[type] = set(typeFactorMap[type])
        self.typeFactorMap = typeFactorMap 
        return factorTypeMap
    
    def findDroppedFactors(self,modelSelectorList): 
        factorModelMap = dict()
        droppedFactors = set()
        descRMGMap = dict() 
        for idx,modelSelector in enumerate(modelSelectorList): 
            if not isinstance(modelSelector, MFM.StatisticalFactorModel) and \
                    not isinstance(modelSelector, MFM.RegionalStatisticalFactorModel):
                (regStats, factors, r2) = modelSelector.\
                        loadRegressionStatistics(self.date, self.modelDB)
                droppedFactorIdx = numpy.flatnonzero(ma.getmaskarray(regStats[:,1]))
                currenciesIdx = [i for (i,f) in enumerate(factors) if \
                                     self.factorTypeMap[factors[i].name].name=="Currency"]
                droppedFactor = [factors[i] for i in droppedFactorIdx if i not in currenciesIdx]
                droppedFactors = droppedFactors.union(droppedFactor) 
                for factor in droppedFactor: 
                    factorModelMap.setdefault(factor,list()).append(idx)
                descRMGMap.update(dict((r.description,r) for r in modelSelector.rmg))

        factorModelMatrix = numpy.zeros((len(factorModelMap),len(modelSelectorList)))
        for idx,factor in enumerate(droppedFactors):
            for m in factorModelMap[factor]:
                factorModelMatrix[idx][m]=1 
        factorModelMatrix = ma.masked_where(factorModelMatrix!=1,factorModelMatrix)

        isConsistent = True
        for idx in range(len(modelSelectorList)):
            if numpy.flatnonzero(ma.getmaskarray(factorModelMatrix[:idx])).size > 0: 
                isConsistent = False
                break 
        data = Utilities.Struct()
        data.reportName = "Dropped Factors" 

        if len(droppedFactors)==0: 
            data.description = "0 Factor (non-currency) omitted from regression" 
        else: 
            headerList = ["#","FACTOR","TYPE","REASON"] 
            if isConsistent == False: 
                headerList.extend([model.mnmonic[2:] for model in modelSelectorList])

            data.description = "%d Factors (non-currency) ommited from regression" %len(droppedFactors)
            resultList = [] 
            for idx,factor in enumerate(droppedFactors):
                factorName = factor.name
                factorType = self.factorTypeMap[factorName].name 
                # If it is a country factor, check reasons 
                if factorName in descRMGMap or factorName == 'Domestic China':
                    if factorName == 'Domestic China':
                        rmg = descRMGMap['China']
                    else:
                        rmg = descRMGMap[factorName]

                    lastTradeDate = modelDB.getDates([rmg], self.date, 0)
                    if lastTradeDate[0] != self.date:
                        reason = 'Non-trading day'
                    else:
                        subFactor = modelDB.getSubFactorsForDate(
                            self.date,[factor])
                        factorReturns = modelDB.loadFactorReturnsHistory(
                            modelSelector.rms_id, subFactor,[self.date])

                        factorReturns.data = factorReturns.data.filled(0.0)
                        fr = factorReturns.data[:,-1]
                        if fr == 0.0:
                            reason = '0 Return'
                        else: 
                            reason = '--UNKNOWN --'
                else:
                    reason ='-- UNKNOWN --'
                result = [idx+1,factorName,factorType,reason]
                if isConsistent == False: 
                    MissingIndicator = [] 
                    for i in factorModelMatrix[idx,:]:
                        if i is ma.masked: 
                            MissingIndicator.append('--')
                        else: 
                            MissingIndicator.append('Missing')
                    result.extend(MissingIndicator)
                resultList.append(result) 
            data.header = headerList 
            data.content = numpy.matrix(resultList) 
        return data 
            
    def getFactorRegression(self,modelSelectorList,dateList,modelType,reportByType=False):
        if self.modelAssetType == "Equity":
            mode = "" 
        else: 
            mode = "Future"
            startDate = datetime.date(self.date.year,1,1)
            dateListYTD = modelDB.getDateRange(self.rmgList,startDate,self.date) 
            subFactors = modelDB.getSubFactorsForDate(
                            self.date,modelSelectorList[0].factors)
            factorReturns = modelDB.loadFactorReturnsHistory(
                            modelSelectorList[0].rms_id, subFactors, dateListYTD)
            factorReturns.data = factorReturns.data.filled(0.0)
            fr = factorReturns.data[:,-1]
            cfrYTD = numpy.cumproduct(1.0 + factorReturns.data,axis=1)[:,-1] -1.0 

        modelInfoDict = dict() 
        factorsList = list() 
        isDifferentFactors = False 
        for modelSelector in modelSelectorList: 
            if not isinstance(modelSelector, MFM.StatisticalFactorModel):

                subFactors = modelDB.getSubFactorsForDate(
                                self.date,modelSelector.factors)
                factorReturns = modelDB.loadFactorReturnsHistory(
                                modelSelector.rms_id, subFactors, dateList)
                factorReturns.data = factorReturns.data.filled(0.0)
                fr = factorReturns.data[:,-1]
                cfr = numpy.cumproduct(1.0 + factorReturns.data,axis=1)[:,-1] -1.0 

                (regStats, factors, r2) = modelSelector.\
                        loadRegressionStatistics(self.date, self.modelDB)
                factorsList.append(set(factors)) 
                factorIdxMap = dict((s,i) for (i,s) in enumerate(factors))
                regStatsHistory = (modelDB.loadRMSFactorStatisticHistory(modelSelector.rms_id,
                                                               subFactors,dateList,"t_value")).data
                masked = sum(regStatsHistory[0].mask )
                absavgtStats = numpy.sum(abs(regStatsHistory.filled(0.0)),axis=1)/(len(dateList)-masked) 
                rank = numpy.argsort(cfr)
                if self.modelAssetType !="Universal":
                    rank = [i for i in rank if self.factorTypeMap[factors[i].name] \
                                != ExposureMatrix.StatisticalFactor]
                tStats = [abs(t) for t in absavgtStats]
                statRank = numpy.argsort(tStats)
                if self.modelAssetType != "Universal":
                    statRank = [i for i in statRank if self.factorTypeMap[factors[i].name] \
                            != ExposureMatrix.StatisticalFactor \
                                and tStats[i]!=0]
                if self.modelAssetType == "Universal":
                    validFactorNames = [self.factorNameDict[i] for i in self.validFactorsForDate]
                    rank = [i for i in rank if factors[i].name in validFactorNames]
                    if reportByType is True :
                        hlFactors = dict() 
                        typeFactorDict = dict() 
                        for factor in self.validFactorsForDate: 
                            typeFactorDict.setdefault(self.factorTypeMap[self.factorNameDict[factor]],list())\
                                .append(self.factorNameDict[factor])
                        
                        for factorType,typeFactors in typeFactorDict.items(): 
                            if len(typeFactors)<10: 
                                logging.info("Ignoring factor type %s since number of factors is less than 10"%factorType)
                            else: 
                                localRank = [i for i in rank if factors[i].name in typeFactors]
                                hlFactors[factorType] = set([factors[i] for i in localRank[:5]]).\
                                    union(set([factors[i] for i in localRank[-5:]]))
                if reportByType is False: 
                    hlFactors = set([factors[i] for i in rank[:5]]).\
                                                union(set([factors[i] for i in rank[-5:]]))
                if self.modelAssetType =="Universal":
                    swFactors = set([factors[i] for i in statRank][-5:]).\
                                                union(set([factors[i] for i in statRank][:5]))
                else:                     
                    swFactors = set([factors[i] for i in statRank if \
                                       self.factorTypeMap[factors[i].name].name!="Currency"][-5:]).\
                                                union(set([factors[i] for i in statRank if \
                                         self.factorTypeMap[factors[i].name].name!="Currency"][:5]))
                modelInfoDict[modelSelector.mnemonic]=(factorIdxMap,hlFactors,swFactors,cfr,absavgtStats)
        

        headerList = ["H/L Factor","Type"] 
        for model in modelSelectorList: 
            if modelType !="Statistical" and self.modelAssetType=="Equity": 
                headerList.extend([model.mnemonic[2:]+" Cum Return %",model.mnemonic[2:]+\
                                       " Avg T-STAT"])
            else: 
                headerList.extend([model.mnemonic[2:]+" Cum Return %", "YTD Cum Return %"])
        # specially for US3 now, MH and SH have different factors.
        # Different for H/L Factors and S/W Factors
        if len(factorsList) == 2 and factorsList[0]!=factorsList[1]: 
            headerList[0] = 'MH H/L Factor'
            headerList = headerList[:-2] + ['SH H/L Factor','Type'] + headerList[-2:] 
            isDifferentFactors = True
        
        

        if reportByType is True: 
            dataList = list() 
            model = modelSelectorList[0]
            (factorIdxMap,hlFactorsDict,swFactors,cfr,regStats) = modelInfoDict[model.mnemonic]
            for factorType,hlFactors in hlFactorsDict.items(): 
                resultList = [] 
                hlFactors = list(hlFactors) 
                cfrList = numpy.array([cfr[factorIdxMap[factor]] for factor in hlFactors])
                rank = numpy.argsort(-cfrList)
                factorList = [hlFactors[i] for i in rank[:]]
                for factor in factorList: 
                    result = [factor.name,self.factorTypeMap[factor.name].name]
                    if result[1] == 'Currency': 
                        result[0] = '%s (%s)'%(factor.name,factor.description)
                    if result[1] != 'Currency':
                        if self.modelAssetType == "Equity":
                            result.extend(['%.2f'%(cfr[factorIdxMap[factor]]*100),
                                           '%.2f'%regStats[factorIdxMap[factor]]])
                        else: 
                            result.extend(['%.2f'%(cfr[factorIdxMap[factor]]*100),
                                           '%.2f'%(cfrYTD[factorIdxMap[factor]]*100)
                                           ])
                    else: 
                        if self.modelAssetType == "Equity":
                            result.extend(['%.2f'%(cfr[factorIdxMap[factor]]*100),'--'])
                        else: 
                            result.extend(['%.2f'%(cfr[factorIdxMap[factor]]*100),
                                           '%.2f'%(cfrYTD[factorIdxMap[factor]]*100)
                                           ])                            
                    resultList.append(result) 

                data = Utilities.Struct() 
                data.header = headerList 
                data.content = numpy.matrix(resultList) 
                factorTypeName = factorType.name.replace(" ","_")
                factorTypeName = factorTypeName.replace("-","_")
                data.reportName = "%s Highest / Lowest Factor Returns"%(factorTypeName)
                data.description = "%s Highest / Lowest Factor Returns"%(factorTypeName)
                dataList.append(data)
            return dataList
        # *** Highest  / Lowest Return Table *** 
        resultList = [] 
        model = modelSelectorList[0]
        (factorIdxMap,hlFactors,swFactors,cfr,regStats) = modelInfoDict[model.mnemonic]
        hlFactors = list(hlFactors) 
        swFactors = list(swFactors)
        cfrList = numpy.array([cfr[factorIdxMap[factor]] for factor in hlFactors])
        swList = numpy.array([abs(regStats[factorIdxMap[factor]]) for factor in swFactors])
        rank = numpy.argsort(-cfrList)
        rankStat = numpy.argsort(-swList)
        factorList = [hlFactors[i] for i in rank[:]]
        factorStatList = [swFactors[i] for i in rankStat[:]]
        if isDifferentFactors: 
            model = modelSelectorList[1]
            (factorIdxMap,hlFactors,swFactors,cfr,regStats) = modelInfoDict[model.mnemonic]
            hlFactors = list(hlFactors) 
            swFactors = list(swFactors)
            cfrList = numpy.array([cfr[factorIdxMap[factor]] for factor in hlFactors])
            swList = numpy.array([abs(regStats[factorIdxMap[factor]]) for factor in swFactors])
            rank = numpy.argsort(-cfrList)
            rankStat = numpy.argsort(-swList)
            factorList_sh = [hlFactors[i] for i in rank[:]]
            factorStatList_sh = [swFactors[i] for i in rankStat[:]]
            
            for idx in range(10): 
                factor_mh = factorList[idx]
                factor_sh = factorList_sh[idx]
                result = [factor_mh.name,self.factorTypeMap[factor_mh.name].name]
                if result[1] == 'Currency': 
                    result[0] = '%s (%s)'%(factor_mh.name,factor_mh.description)
                (factorIdxMap,hlFactors,swFactors,cfr,regStats) = modelInfoDict[modelSelectorList[0].mnemonic]
                if result[1] != 'Currency':
                    result.extend(['%.2f'%(cfr[factorIdxMap[factor_mh]]*100),
                                   '%.2f'%regStats[factorIdxMap[factor_mh]]])
                else: 
                    result.extend(['%.2f'%(cfr[factorIdxMap[factor_mh]]*100),'--'])
                result.extend([factor_sh.name,self.factorTypeMap[factor_sh.name].name])
                if result[-1] == 'Currency': 
                    result[-2] = '%s (%s)'%(factor_sh.name,factor_sh.description)
                (factorIdxMap,hlFactors,swFactors,cfr,regStats) = modelInfoDict[modelSelectorList[1].mnemonic]
                if result[-1] != 'Currency':
                    result.extend(['%.2f'%(cfr[factorIdxMap[factor_sh]]*100),
                                   '%.2f'%regStats[factorIdxMap[factor_sh]]])
                else: 
                    result.extend(['%.2f'%(cfr[factorIdxMap[factor_sh]]*100),'--'])
                    
                resultList.append(result) 
            
        else: 
            for factor in factorList: 
                result = [factor.name,self.factorTypeMap[factor.name].name]
                if result[1] == 'Currency': 
                    result[0] = '%s (%s)'%(factor.name,factor.description)
                for model in modelSelectorList: 
                    (factorIdxMap,hlFactors,swFactors,cfr,regStats) = modelInfoDict[model.mnemonic]
                    if result[1] != 'Currency':
                        if self.modelAssetType == "Equity":
                            result.extend(['%.2f'%(cfr[factorIdxMap[factor]]*100),
                                           '%.2f'%regStats[factorIdxMap[factor]]])
                        else: 
                            result.extend(['%.2f'%(cfr[factorIdxMap[factor]]*100),
                                           '%.2f'%(cfrYTD[factorIdxMap[factor]]*100)
                                           ])
                    else: 
                        if self.modelAssetType == "Equity":
                            result.extend(['%.2f'%(cfr[factorIdxMap[factor]]*100),'--'])
                        else: 
                            result.extend(['%.2f'%(cfr[factorIdxMap[factor]]*100),
                                           '%.2f'%(cfrYTD[factorIdxMap[factor]]*100)
                                           ])                            
                resultList.append(result) 

        data = Utilities.Struct() 
        data.header = headerList 
        data.content = numpy.matrix(resultList) 
        if isDifferentFactors: 
            data.reportName = "US3 Highest / Lowest %s Factor Returns"%modelType
        else: 
            data.reportName = "%s Highest / Lowest %s Factor Returns"%(mode,modelType)
        data.description = "Highest / Lowest %s %d MODEL DAYS Cumulative Factor Returns"%(modelType,len(dateList))
        
        # For commodity model and universal model no t-stat populated
        if self.modelAssetType in[ "Future","Universal"]:
            data.description = data.description.replace("Fundamental","")
            return [data]

        # *** Strongest / Weakest Factor Table 
        resultList = [] 
        if isDifferentFactors: 
            for idx in range(10): 
                factor_mh = factorStatList[idx]
                factor_sh = factorStatList_sh[idx]
                result = [factor_mh.name,self.factorTypeMap[factor_mh.name].name]
                if result[1] == 'Currency': 
                    result[0] = '%s (%s)'%(factor_mh.name,factor_mh.description)
                (factorIdxMap,hlFactors,swFactors,cfr,regStats) = modelInfoDict[modelSelectorList[0].mnemonic]
                if result[1] != 'Currency':
                    result.extend(['%.2f'%(cfr[factorIdxMap[factor_mh]]*100),
                                   '%.2f'%regStats[factorIdxMap[factor_mh]]])
                else: 
                    result.extend(['%.2f'%(cfr[factorIdxMap[factor_mh]]*100),'--'])
                result.extend([factor_sh.name,self.factorTypeMap[factor_sh.name].name])
                if result[-1] == 'Currency': 
                    result[-2] = '%s (%s)'%(factor_sh.name,factor_sh.description)
                (factorIdxMap,hlFactors,swFactors,cfr,regStats) = modelInfoDict[modelSelectorList[1].mnemonic]
                if result[-1] != 'Currency':
                    result.extend(['%.2f'%(cfr[factorIdxMap[factor_sh]]*100),
                                   '%.2f'%regStats[factorIdxMap[factor_sh]]])
                else: 
                    result.extend(['%.2f'%(cfr[factorIdxMap[factor_sh]]*100),'--'])
                    
                resultList.append(result)             
        else: 
            for factor in factorStatList: 
                result = [factor.name,self.factorTypeMap[factor.name].name]
                if result[1] == 'Currency': 
                    result[0] = '%s (%s)'%(factor.name,factor.description)
                for model in modelSelectorList: 
                    (factorIdxMap,hlFactors,swFactors,cfr,regStats) = modelInfoDict[model.mnemonic]
                    if result[1] != "Currency": 
                        result.extend(['%.2f'%(cfr[factorIdxMap[factor]]*100),
                                       '%.2f'%regStats[factorIdxMap[factor]]])
                    else: 
                        result.extend(['%.2f'%(cfr[factorIdxMap[factor]]*100),'--'])

                resultList.append(result) 
        
        data2 = Utilities.Struct() 
        data2.header = headerList 
        data2.content = numpy.matrix(resultList) 
        if isDifferentFactors: 
            data2.reportName = "US3 Strongest / Weakest %s Factors"%modelType 
        else: 
            if self.modelAssetType == "Equity":
                data2.reportName = "Strongest / Weakest %s Factors"%modelType 
            else: 
                data2.reportName = "%s Strongest / Weakest %s Factors"%(self.modelAssetType,
                                                                        modelType)
        data2.description = "Strongest / Weakest %s %d Days Average T-STAT"%(modelType,len(dateList))        
        return [data,data2]

    def isNumeric(self, s):
        try:
            float(s)
            return True
        except ValueError:
            return False
        except TypeError:
            return False 

    def returnDescriptorCoverage(self):
        from operator import truediv 
        des_table = 'DESCRIPTOR_EXPOSURE_USD'
        query = """select COLUMN_NAME from ALL_TAB_COLUMNS where TABLE_NAME='%s' and owner ='MODELDB_GLOBAL'"""%des_table
        self.modelDB.dbCursor.execute(query) 
        descriptors = [row[0] for row in self.modelDB.dbCursor.fetchall() if row[0][:2] =='DS']
        modelDates = self.modelDB.getDates(self.rmgList,self.date,19,True) 
        query = """select descriptor_id, name from descriptor""" 
        self.modelDB.dbCursor.execute(query) 
        descriptorDict = dict(self.modelDB.dbCursor.fetchall())
        query = """select dt,count(*) from %s
                   where dt between :dt_start and :dt_end 
                   group by dt order by dt"""%(des_table)
        myDict = dict([('dt_start',modelDates[0]),('dt_end',self.date)])
        self.modelDB.dbCursor.execute(query,myDict)
        result = self.modelDB.dbCursor.fetchall()
        dateList0,universeList0 = zip(*result)
        dateList = list(); universeList = list()
        for idx, date in enumerate(dateList0): 
            if date.date() in modelDates: 
                dateList.append(date.date()) 
                universeList.append(universeList0[idx])
        techResultList = list();fundResultList = list() 
        techDs = list() ; fundDs = list() 
        for ds in descriptors: 
            dsID = ds.split('_')[-1]
            dsName = descriptorDict[int(dsID)]
            self.modelDB.dbCursor.execute(query.replace('*',ds),myDict)
            coverageList = [row[1] for row in self.modelDB.dbCursor.fetchall() if row[0].date()in modelDates]
            if set(coverageList) == set([0]): 
                logging.info("Ignoring %s"%dsName)
            else: 
                coverageList = list(map(truediv, coverageList,universeList))
                result = [cov * 100.0 for cov in coverageList]
                if coverageList.count(1.0) >= float(len(dateList)/2):
                    #identify as technical factors 
                    if result[-1] != 100 : 
                        techDs.append(dsName) 
                        techResultList.append(result) 
                else: 
                    fundDs.append(dsName) 
                    fundResultList.append(result) 
                

        data = Utilities.Struct() 
        data.reportSection = "DESCRIPTOR SECTIONS"
        data.reportName = "Fundamental Descriptor Coverage for the last 20 days - Full Universe"
        data.description = data.reportName
        dateCol = numpy.array([date.strftime('%b%d') for date in dateList]).transpose()
        content = numpy.matrix(fundResultList).transpose()
        content = numpy.c_[dateCol,content]        
        data.content = content 
        data.header= ['Date'] + fundDs
        data.reportType = "LineChart" 


        data1 = Utilities.Struct() 
        data1.reportSection = "DESCRIPTOR SECTIONS"
        data1.reportName = "Fundamental Descriptor Coverage for the last 2 days (Table) - Full Universe"
        data1.description = data1.reportName
        dsCol = numpy.array([ds for ds in fundDs]).transpose() 
        content = numpy.matrix([fund[-2:]+[fund[-1] - fund[-2]] for fund in fundResultList])
        content = numpy.c_[dsCol,content]        
        data1.content = content 
        data1.header= ['Descriptor'] + [date.strftime('%b%d') for date in dateList[-2:]] + ['Difference']


        data2 = Utilities.Struct() 
        data2.reportSection = "DESCRIPTOR SECTIONS"
        if len(techDs) == 0: 
            data2.reportName = "No Technical Descriptor Coverage < 100%"
        else: 
            data2.reportName = "Technial Descriptor Coverage for the last 20 days"
            dateCol = numpy.array([date.strftime('%b%d') for date in dateList]).transpose()
            content = numpy.matrix(techResultList).transpose()
            content = numpy.c_[dateCol,content]        
            data2.content = content 
            data2.header= ['Date'] + techDs
            
        data2.description = data2.reportName


        self.coreDataList.extend([data,data1,data2]) 

    def returnFactorAutoCorrChecks(self): 
        # Looking at SH only for now 
        for idx in [1]: 
            dataFund, dataTech  = self.checkFactorAutoCorr(idx,'daily')
            dataFund.reportSection = "FACTOR EXPOSURES"
            dataTech.reportSection = "FACTOR EXPOSURES"
            dataFund.reportName = "%s Fundamental Factor Exposure Stability Check (daily)"%self.modelSelectorList[idx].mnemonic
            dataTech.reportName = "%s Technical Factor Exposure Stability Check (daily)"%self.modelSelectorList[idx].mnemonic
            dataFund.description = "Exposure Stability for last 20 days - %s Fundamental Factors (daily)"%self.modelSelectorList[idx].mnemonic
            dataTech.description = "Exposure Stability for last 20 days - %s Technical Factors (daily)"%self.modelSelectorList[idx].mnemonic
            if self.exposureQA.get("fund075"):
                dataFund.reportName = "*** " + dataFund.reportName + " <=0.75 - Escalate to Research IMMEDIATELY***"
                dataFund.description = "*** " + dataFund.description + " <=0.75 - Escalate to Research IMMEDIATELY***"

            if self.exposureQA.get("tech04"):
                dataTech.reportName = "*** " + dataTech.reportName + " <=0.4 - Escalate to Research IMMEDIATELY***"
                dataTech.description = "*** " + dataTech.description + " <=0.4 - Escalate to Research IMMEDIATELY***"

            if self.exposureQA.get("tech04075"):
                dataTech.reportName = "***(ex FX Sensitivity) " + dataTech.reportName + " between 0.4 & 0.75 - Publish model and email Research for confirmation***"
                dataTech.description = "***(ex FX Sensitivity) " + dataTech.description + " between 0.4 & 0.75 - Publish model and email Research for confirmation***"

            if self.exposureQA.get("fx0405"):
                dataTech.reportName = "***(FX Sensitivity) " + dataTech.reportName + " between 0.4 & 0.5 - Publish model and email Research for confirmation***"
                dataTech.description = "***(FX Sensitivity) " + dataTech.description + " between 0.4 & 0.5 - Publish model and email Research for confirmation***"
                
            self.coreDataList.append(dataFund) 
            self.coreDataList.append(dataTech)

    def returnMonthlyFactorAutoCorrChecks(self): 
        # Looking at SH only for now 
        for idxM in [1]:
            result,styles = self.checkFactorAutoCorr(idxM,'monthly')
            print(styles, result)
            if result is not None :
                if len(result) > 0: 
                    for idx,factorResult in enumerate(result): 
                        factorResult.reportSection = "FACTOR EXPOSURES" 
                        factorResult.reportName = "Monthly Correlation Check: Threshold breached for %s %s Factor"%(self.modelSelectorList[idxM].mnemonic,styles[idx])
                        factorResult.description = factorResult.reportName                         
                        factorResult.position = '%d_3'%((idx+1)%3 if (idx+1)%3 !=0 else 3)
                        self.coreDataList.append(factorResult) 
                else: 
                    data = Utilities.Struct() 
                    data.description = "Monthly Correlation Check: No Thresholds breached for Fundamental Factors"
                    data.reportName = data.description
                    data.reportSection =  "FACTOR EXPOSURES" 
                    self.coreDataList.append(data) 
                
    def checkFactorAutoCorr(self,modelIdx,mode): 
        logging.info("Processing correlation between expsoures for each factor - %s"%mode) 
        if mode == 'daily':
            dateList = self.modelDB.getDates(self.rmgList,self.date,20,True) 
        if mode == 'monthly': 
            model = self.modelSelectorList[modelIdx].mnemonic.split('_')[0][:-3]
            dateList = getDatesFromCalendar(model,'monthly','%04d-%02d-%02d'%(self.date.year,self.date.month,self.date.day))
            selectedList = []
            for idx in reversed(list(range(7))): 
                selectedList.append(dateList[-idx*12 -2])                
                selectedList.append(dateList[-idx*12 -1])
            dateList =selectedList 
        # if mode == 'monthly' and dateList[-1] != self.date:
        # Removing Month-end check for monthly correlation as there is display issue and daily correlation is sufficient
        if mode == 'monthly':
            return (None ,None)
                  
        rm = self.modelSelectorList[modelIdx]
        estUList = list() 
        expMList = list() 

        for idx,date in enumerate(dateList): 
            rm.setFactorsForDate(date,self.modelDB)
            rmi = self.modelDB.getRiskModelInstance(rm.rms_id,date)
            estU = rm.loadEstimationUniverse(rmi,self.modelDB)
            expM = rm.loadExposureMatrix(rmi,self.modelDB)
            expMList.append(expM)  
            assets = expM.getAssets() 
            estUList.append(set(estU).intersection(set(assets)))

        stylesCompare = list()
        for idx,expM in enumerate([expMList[-1],expMList[0]]):
            factorTypeMap = dict()
            typeFactorMap = dict() 
            rm.setFactorsForDate(date,self.modelDB)
            factors = rm.factors 
            for fType in expM.factorIdxMap_.keys():
                if len(expM.factorIdxMap_[fType]) > 0:
                    for fName in expM.factorIdxMap_[fType].keys():
                        factorTypeMap[fName] = fType 
                        if fType.name != 'Currency':
                            typeFactorMap.setdefault(fType.name,list()).append(\
                                factors[expM.factorIdxMap_[fType][fName]])
            stylesCompare.append(list(set(factor.name for factor in typeFactorMap['Style'])))
        styleFactors = set(stylesCompare[0]).intersection(set(stylesCompare[1]))
        
        stylesChecked = list() 

        coeffDict = dict() 
        for idx in range(len(expMList)-1):
            expM = expMList[idx+1]
            assets  = expM.getAssets()                           
            assetPosMap = dict((j,i) for (i,j) in enumerate(assets))
            p_expM = expMList[idx]
            p_assets = p_expM.getAssets() 
            p_assetPosMap = dict((j,i) for (i,j) in enumerate(p_assets))
            estUSelected = list(set(estUList[idx+1]).intersection(set(estUList[idx])))
            idx1 = [assetPosMap[asset] for asset in estUSelected]
            idx2 = [p_assetPosMap[asset] for asset in estUSelected]
            
            for factorName in styleFactors: 
                factorIdx = expM.getFactorIndex(factorName) 
                p_factorIdx = p_expM.getFactorIndex(factorName) 
                array1 = [expM.getMatrix()[factorIdx][j] for j in idx1]
                array2 = [p_expM.getMatrix()[p_factorIdx][j] for j in idx2]
                array1 = numpy.nan_to_num(array1) 
                array2 = numpy.nan_to_num(array2) 
                if len(set(array1)) <=2: 
                    logging.info("Ignoring %s.Binary Factor"%factorName)
                else: 
                    if mode == 'daily' or (mode=='monthly' and idx%2 ==0):
                        coeff = pearsonr(array1,array2)[0]
                        logging.info("loaded coeff for %s"%factorName)
                        if factorName not in stylesChecked: 
                            stylesChecked.append(factorName) 
                        coeffDict.setdefault(factorName,list()).append(coeff)

        fund = list(['Value','Leverage','Growth','Profitability','Dividend Yield', 'Earnings Yield']) 
        fundFactors  = list() 
        techFactors = list() 
        for factor in stylesChecked: 
            if factor in fund: 
                fundFactors.append(factor)
            else: 
                if mode =='daily': 
                    techFactors.append(factor)

        dataList = list()
        maxdate = max(dateList[1:])

        if mode == 'daily': 
            for pos, stylesChecked in enumerate([fundFactors, techFactors]):
                resultList = list() 
                for idx, date in enumerate(dateList[1:]): 
                    row = [date.strftime('%b%d')]
                    for idx2,style in enumerate(stylesChecked): 
                        row.append(coeffDict[style][idx])
                        if date == maxdate:
                            logging.info("QA Checks on exposure stability...")
                            if style in fund and coeffDict[style][idx] <= 0.75:
                                self.exposureQA["fund075"] = True
                            if style not in fund and coeffDict[style][idx] <= 0.4:
                                self.exposureQA["tech04"] = True
                            if style not in fund and style != "Exchange Rate Sensitivity" and coeffDict[style][idx] <= 0.75 and coeffDict[style][idx] > 0.4:
                                self.exposureQA["tech04075"] = True
                            if style == "Exchange Rate Sensitivity" and coeffDict[style][idx] <= 0.5 and coeffDict[style][idx] > 0.4:
                                self.exposureQA["fx0405"] = True
                    if pos == 0: 
                        lowerbound = '0.95' 
                    if pos == 1: 
                        lowerbound = '0.9' 
                    row.append(lowerbound)
                    resultList.append(row)

                data = Utilities.Struct() 
                data.content = numpy.matrix(resultList) 
                data.header= ['Date'] + stylesChecked +['Placeholder (%s)'%lowerbound]
                data.reportType = "LineChart_material" 
                dataList.append(data)

            return dataList
        if mode == 'monthly': 
            dateList = dateList[1::2]
            for pos, style in enumerate(fundFactors): 
                resultList = list()
                obs = coeffDict[style]
                dev = numpy.std(obs[:-1],ddof=1)
                lowerbound = min(obs[:-1]) if 0.01 < dev else min(obs[:-1])-0.01
                upperbound = min(max(obs[:-1]) + 0.01,1) 
                if obs[-1] >=upperbound or obs[-1] <=lowerbound : 
                    for idx, date in enumerate(dateList): 
                        row = [date.strftime('%b%y')]
                        row.append(coeffDict[style][idx])
                        row.extend([lowerbound,upperbound])
                        resultList.append(row)
                    data = Utilities.Struct() 
                    data.content = numpy.matrix(resultList) 
                    data.header= ['Date'] + [style] +['Lower (%s)'%lowerbound,'Upper (%s)'%upperbound]
                    data.reportType = "LineChart_material" 
                    dataList.append(data)

        return (dataList,fundFactors)

    def returnFactorReturnsChecks(self):
        logging.info("Processing High/Low Strong/Weak Factor search")
        sectionName = "FACTOR RETURNS"
        sectionName2 = "REGRESSION DIAGNOSTICS"
        mode = "" 
        if self.modelAssetType in ["Future","Universal"]: 
            mode = "Future "

        # 1 day 
        dateList = modelDB.getDates(self.rmgList,self.date,0) 
        dataList = self.getFactorRegression(self.fundamental,dateList,"Fundamental") 
        data = dataList[0]
        if 'US3' in self.modelSelectorList[0].mnemonic:
            data.reportName = "US3 Highest / Lowest Fundamental one-day Factor Return"        
        else: 
            data.reportName = "%sHighest / Lowest Fundamental one-day Factor Return"%mode
        data.reportSection = sectionName 
        data.description = "Highest / Lowest Fundamental ONE DAY Factor Return"
        if self.modelAssetType == "Future": 
            data.description = data.description.replace("Fundamental","")
        header = data.header
        headerNew = list()
        for head in header:
            head=head.replace("Cum","")
            head=head.replace("Avg","")
            headerNew.append(head)        
        data.header = headerNew
        self.coreDataList.append(data)
        if self.modelAssetType == "Equity":
            data = dataList[1]
            data.reportSection = sectionName2
            if 'US3' in self.modelSelectorList[0].mnemonic:
                data.reportName = "US3 Strongest / Weakest Fundamental Factors one-day"
            else: 
                data.reportName = "Strongest / Weakest Fundamental Factors one-day"
            data.description = "Strongest / Weakest Fundamental one-day T-STAT"
            header = data.header
            headerNew = list()
            for head in header:
                head=head.replace("Cum","")
                head=head.replace("Avg","")
                headerNew.append(head)
            data.header = headerNew
            self.coreDataList.append(data)

        # 30 days         
        if self.modelAssetType == "Universal":
            dateList = self.dates[-5:]
        else: 
            dateList = modelDB.getDates(self.rmgList,self.date,29) 

        dataList = self.getFactorRegression(self.fundamental,dateList,"Fundamental") 
        for idx,data in enumerate(dataList): 
            if idx == 0: 
                data.reportSection = sectionName 
            else: 
                data.reportSection = sectionName2
            self.dataList.append(data)
                 
        if self.modelAssetType == "Universal": 
            dataList = self.getFactorRegression(self.fundamental,[self.date],"Fundamental",reportByType=True) 
            for data in dataList: 
                data.reportSection = sectionName 
                self.dataList.append(data) 

        # Ommited Factors 
        if self.modelAssetType == "Equity":
            data = self.findDroppedFactors(self.fundamental)
            data.reportSection = sectionName2 
            self.coreDataList.append(data) 
        logging.info("Finished Processing High/Low Strong/Weak Factor search")

    
    def returnDodgySpecificReturns(self): 
        logging.info("Processing Asset Specific Return & Factor Exposures Section")
        if self.modelAssetType == "Equity":
            sectionName = "ASSET SPECIFIC RETURN"
            data = self.prepDodgySpecificReturns(self.fundamental)
            data.reportName = "High Fundamental Specific Return Assets" 
            data.description = "Assets with Fundamental Model " + data.description
            data.reportSection = sectionName 
            self.dataList.append(data) 
            data = self.prepDodgySpecificReturns(self.statistical) 
            data.reportName = "High Statistical Specific Return Assets" 
            data.reportSection = sectionName 
            data.description = "Assets with Statistical Model " + data.description
            self.dataList.append(data) 
        else: 
            sectionName = "ASSET SPECIFIC RETURN & FACTOR EXPOSURES"
            data = self.prepDodgySpecificReturns(self.modelSelectorList)
            data.reportName = "High Commodity Specific Return Assets" 
            data.description = "Commodity Model with " + data.description
            data.reportSection = sectionName 
            self.dataList.append(data) 
            
        logging.info("Finished Processing Asset Specific Return & Factor Exposures Section")

        
    def prepDodgySpecificReturns(self,modelSelectorList):

        if self.modelSelectorList[0].mnemonic[:4] == 'AXCN':
                       threshold = 0.04
        elif self.modelSelectorList[0].mnemonic[:4] == 'AXTW':
            threshold = 0.03
        else:
            threshold = 0.15
        isSameFactors = True 
        
        if self.modelAssetType == "Future": 
            threshold = 0.01

        factors = modelSelectorList[0].factors 
        if len(modelSelectorList)>1: 
            for model in modelSelectorList:
                if model.factors != factors:
                    isSameFactors = False 
#                    print factors, model.factors 

        if isSameFactors: 
            sr = modelDB.loadSpecificReturnsHistory(modelSelectorList[0].rms_id, 
                        self.fullsubID, [self.date])
            sr.data = ma.masked_where(sr.data <= -1.0, sr.data)
            sr = sr.data[:,0].filled(0.0)
            dodgyIdx = numpy.flatnonzero(ma.getmaskarray(ma.masked_where(
                        abs(numpy.log10(1.0+sr)) > threshold, sr)))
        else: 
            dodgyIdx = set()
            srList = [] 
            for model in modelSelectorList: 
                sr = modelDB.loadSpecificReturnsHistory(model.rms_id, 
                            self.fullsubID, [self.date])
                sr.data = ma.masked_where(sr.data <= -1.0, sr.data)
                sr = sr.data[:,0].filled(0.0)
                dodgyIdx = dodgyIdx.union(set(numpy.flatnonzero(ma.getmaskarray(ma.masked_where(
                            abs(numpy.log10(1.0+sr)) > threshold, sr)))))
                srList.append(sr)
        bounds = [10.0**(threshold*x) - 1.0 for x in (-1.0, 1.0)]

        tr = modelDB.loadTotalReturnsHistory(
            modelSelectorList[0].rmg, self.date,self.fullsubID, 0)
        tr.data = tr.data.filled(0.0)
        trDict = dict(zip(self.fullsubID, 
                          ['%.2f'%(val * 100.0) for val in tr.data[:,0]]))
        self.headerMap['TR %']=trDict

        if len(dodgyIdx) > 0:
            sr_assets = [self.fullsubID[i] for i in dodgyIdx]
            if self.modelAssetType == "Future":
                self.wholeEstUniv = self.estUniv[0]
            estuDict = dict((s,1) for s in sr_assets if s in self.wholeEstUniv)
            for asset in sr_assets: 
                if asset not in estuDict: 
                    estuDict[asset] = '--'
            self.headerMap["ESTU"] = estuDict 
            if isSameFactors: 
                srDict = dict(zip(sr_assets,['%.2f'%(sr[i]*100.0) for i in dodgyIdx]))
                self.headerMap['SR %'] = srDict
                extraHeader = ["CAP (%s m)"%self.currency,"SR %", "TR %", "ESTU",
                              ]
                if self.modelAssetType == "Future":
                    extraHeader = extraHeader[1:-1]
            else: 
                srDict1 = dict(zip(sr_assets,['%.2f'%(srList[0][i]*100.0) for i in dodgyIdx]))
                srDict2 = dict(zip(sr_assets,['%.2f'%(srList[1][i]*100.0) for i in dodgyIdx]))
                self.headerMap['%s SR %%'%modelSelectorList[0].mnemonic[-2:]] = srDict1
                self.headerMap['%s SR %%'%modelSelectorList[1].mnemonic[-2:]] = srDict2
                extraHeader = ["CAP (%s m)"%self.currency,
                               "%s SR %%"%modelSelectorList[0].mnemonic[-2:],
                               "%s SR %%"%modelSelectorList[1].mnemonic[-2:],                       
                               "TR %", "ESTU",
                               ]

            rank_by_cap = numpy.argsort(-numpy.array([self.assetCapDict.get(asset)for \
                                                          asset in sr_assets]))
            records = len(sr_assets)
            rank_by_cap = rank_by_cap[:self.maxRecords]
            sr_assets = [sr_assets[i] for i in rank_by_cap]
            data = self.prepAssetLevelInfo(sr_assets,extraHeader,
                                           appendIndustry = self.appendIndustry,
                                           appendAssetType = self.appendAssetType,
                                           modelAssetType = self.modelAssetType)
            data.description = "%d Assets with Specific Returns beyond [%.2f%%, %.2f%%]"%\
                (records,100.0 * bounds[0], 100.0 * bounds[1])
            return data 
        else:
            self.headerMap["ESTU"] = dict() 
            self.headerMap['SR %'] = dict() 
            data = Utilities.Struct()
            data.description = "No Assets with Specific Returns beyond [%.2f%%, %.2f%%]"%\
                (100.0 * bounds[0], 100.0 * bounds[1])
            return data

    def returnCountryCurrencyChanges(self): 
        logging.info("Processing Country and Currency Exposure Change")
        
        
        allFactorNames = set() 
        factorModelMap = dict() 
        modelSelector = self.modelSelectorList[0]
        expM = self.expMList[0]
        p_expM = self.prevExpMList[0]
        factorNames = expM.getFactorNames() 
        p_factorNames = p_expM.getFactorNames() 
        validFactors = [] 
        countryFactors = list(set(self.typeFactorMap['Country']))
        currencyFactors = list(set(self.typeFactorMap['Currency']))
        factorMap = dict((factor.name,factor) for factor in countryFactors + currencyFactors)
        prevCtryDict = dict(); currCtryDict = dict() 
        prevCurrDict = dict(); currCurrDict = dict()
        currDict = dict(); prevDict = dict() 
        for idx,factorList in enumerate([[factor.name for factor in countryFactors],
                                         [factor.name for factor in currencyFactors]]):
            lc_factors = 0 
            tempDataList = [] 
            for factor in factorList: 
                if factor in factorNames: 
                    pos = 0
                    riskModel = self.modelSelectorList[0]
                    (currExp,prevExp,thres) = self.prepExposuresForFactor(pos,factor,0.5)
                    lc_assets = list(currExp.keys())
                    rank_by_cap = numpy.argsort(-numpy.array([self.assetCapDict.get(asset)for \
                                                                  asset in lc_assets]))
                    records = len(lc_assets)
                    rank_by_cap = rank_by_cap[:self.maxRecords]
                    lc_assets = [lc_assets[i] for i in rank_by_cap]

                    if len(lc_assets)>0:
                        lc_factors+=1
                        for asset in lc_assets:
                            if currExp[asset]=='1.00':
                                if idx == 0: 
                                    currCtryDict[asset] = factor 
                                else: 
                                    currCurrDict[asset] = factor
                            else: 
                                if idx == 0: 
                                    prevCtryDict[asset] = factor 
                                else: 
                                    prevCurrDict[asset] = factor 

        if len(currCtryDict)!=0:
            # assert (len(currCtryDict)==len(prevCtryDict))
            #DIA-4070 Extra check on SPACs
            currCtryDictKeys = list(currCtryDict.keys())
            for k in currCtryDictKeys:
                if k not in prevCtryDict:
                    modeldbID = k.getSubIdString()[:-2]
                    query = """select * from ISSUE_MAP im, marketdb_global.classification_active_int classint where im.MODELDB_ID = '%s'
                    and im.MARKETDB_ID=classint.AXIOMA_ID and classint.CLASSIFICATION_ID=6012
                    and classint.FROM_DT < '%s' and classint.THRU_DT > '%s'""" % (modeldbID, self.date, self.date)
                    self.modelDB.dbCursor.execute(query)
                    result = self.modelDB.dbCursor.fetchall()
                    if len(result) == 1:
                        del currCtryDict[k]

                    prevCtryDict[k] = ''

            if len(currCtryDict)==0:
                return None
                    
            assert (len(currCurrDict)==len(prevCurrDict))
            for key in currCtryDict:
                currDict[key] = str(currCtryDict[key]) + ' / ' + str(currCurrDict.get(key))
                prevDict[key] = str(prevCtryDict[key]) + ' / ' + str(prevCurrDict.get(key))

            extraHeader = ["CAP (%s m)"%self.currency,
                       "CURR","PREV"]
            self.headerMap["CURR"] = currDict
            self.headerMap["PREV"] = prevDict            
            data = self.prepAssetLevelInfo(list(currDict.keys()),extraHeader)
            data.description = "%d Assets with Country/Currency Exposure Changes"\
                %len(currDict)
            data.reportName = "Country/Currency Exposure Change" 
            return data
        else: 
            return None 
                
    def returnAssetRiskChange(self,threshold = 0.01, total=False): 
        modelDB = self.modelDB
        model = self.modelSelectorList[0]
        if not total:
            (riskDict,scDict) = model.loadSpecificRisks(self.rmiList[0], modelDB)
            (p_riskDict,p_scDict) = model.loadSpecificRisks(self.p_rmiList[0], modelDB)
            riskType = 'specific'
        else:
            risks = list()
            rm = self.modelSelectorList[0]
            for idx,rmi in enumerate([self.rmiList[0],
                     modelDB.getRiskModelInstance(self.rmiList[0].rms_id,self.prevDate)]):
                dat = Utilities.Struct()
                (dat.factorCovariance,factors) = rm.loadFactorCovarianceMatrix(rmi,self.modelDB)
                if idx == 0:
                    dat.exposureMatrix = self.expMList[0]
                else: 
                    dat.exposureMatrix = self.prevExpMList[0]
                dat.exposureMatrix.fill(0.0)
                dat.specificRisk =  model.loadSpecificRisks(self.rmiList[0], modelDB)[0]
                dat.specificRisk = model.loadSpecificRisks(self.p_rmiList[0], modelDB)[0]
                risks.append(rm.computeTotalRisk(dat, modelDB))
            riskDict = dict(risks[0])
            p_riskDict = dict(risks[1])
            riskType = 'total'
        commonAssets = list(set(riskDict.keys()).intersection(list(p_riskDict.keys())))
        diff = numpy.array([riskDict[sid] - p_riskDict[sid] for sid in commonAssets])
        bigChange = numpy.flatnonzero(ma.getmaskarray(
                        ma.masked_where(abs(diff) > threshold, diff)))
        assetList = [commonAssets[i] for i in bigChange]
        riskDict = dict((s, round(v*100.0,2)) for (s,v) in riskDict.items())
        p_riskDict = dict((s, round(v*100.0,2)) for (s,v) in p_riskDict.items())

        if len(assetList)!=0:
            extraHeader = ["RISK %","PREV RISK %", "TR %" ]
            self.headerMap["RISK %"]= riskDict
            self.headerMap["PREV RISK %"] = p_riskDict
            data = self.prepAssetLevelInfo(assetList,extraHeader,
                                           appendIndustry = self.appendIndustry,
                                           appendAssetType = self.appendAssetType,
                                           modelAssetType = self.modelAssetType)

            data.description = "%d Assets with %s risk change > %.2f%%"%(len(assetList),riskType,
                                                                      threshold*100)
        else: 
            data = Utilities.Struct() 
            data.description = "No Asset with %s risk change > %.2f%%"%(riskType,
                                                                      threshold*100)
        data.reportName = "%s Change"%riskType 
        data.reportSection = "ASSET RISK CHANGE"
        self.dataList.append(data) 

    def getFundamentalChange(self,sids,date,includeEst=False): 
        sidArgs = ['sid%d' % i for i in range(len(sids))]
#                where code.CODE_TYPE like 'asset_dim_fund_currency%%' 
        query = """ 
                select SUB_ISSUE_ID, NAME, VALUE, CURRENCY_ID, EFF_DT,REV_DT 
                from marketdb_global.META_CODES code, SUB_ISSUE_FUND_CURRENCY fund
                where code.CODE_TYPE = 'asset_dim_fund_currency:item_code' 
                and SUB_ISSUE_ID IN (%(sids)s)
                and fund.ITEM_CODE = code.ID 
                and fund.EFF_DT <= fund.REV_DT 
                and fund.EFF_DT <=:date_arg 
                and fund.REV_DT >:date_arg0  and REV_DT <:date_arg2
                order by SUB_ISSUE_ID,ITEM_CODE,DT , EFF_DT
                """%{'sids':','.join([':%s' % arg for arg in sidArgs])}
        dataDict = dict()
        dataDict['date_arg'] = self.date
        dataDict['date_arg2'] = self.date + datetime.timedelta(1)
        dataDict['date_arg0'] = self.prevDate
        dataDict.update(dict(zip(sidArgs, 
                                 sids)))
        self.modelDB.dbCursor.execute(query, dataDict)
        result = self.modelDB.dbCursor.fetchall()
        fundDict = dict() 
        for r in result: 
            fundDict.setdefault(r[0],dict()).update(dict([(r[1],[item for item in r[2:]])]))

        if includeEst: 
            query.replace('asset_dim_fund_currency','asset_dim_esti_currency')
            query.replace('SUB_ISSUE_FUND_CURRENCY','SUB_ISSUE_ESTI_CURRENCY')
            self.modelDB.dbCursor.execute(query, dataDict)
            result = self.modelDB.dbCursor.fetchall()
            for r in result: 
                fundDict.setdefault(r[0],dict()).update(dict([(r[1],[item for item in r[2:]])]))
        return fundDict 
    
    def buildFactorFieldsDict(self,riskModel,useQuarterlyData=True): 
        if useQuarterlyData is True: 
            suffix = '_qtr'
        else: 
            suffix ='_ann'
        if not hasattr(riskModel, 'modelHack'):
            riskModel.modelHack = Utilities.Struct()
            riskModel.modelHack.legacyETP = False
        factorIDNameMap = dict([(0,'Value'),(1,'Leverage'),(2,'Growth')])
        factorFieldsDict = dict() 
        RM = dict() 
        # not using scale now .....
        query = """
    select f.name,ds.NAME from FACTOR f, DESCRIPTOR ds, RMS_FACTOR_DESCRIPTOR fds 
    where fds.RMS_ID = %d
    and fds.FACTOR_ID in (0,1,2)
    and f.FACTOR_ID = fds.FACTOR_ID 
    and ds.DESCRIPTOR_ID = fds.DESCRIPTOR_ID
                """%riskModel.rms_id
        self.modelDB.dbCursor.execute(query)         
        result = self.modelDB.dbCursor.fetchall() 
        factorDescriptorMap = dict() 
        if len(result) != 0 : 
            for row in result: 
                factorDescriptorMap.setdefault(row[0],list()).append(row[1])
        SCM = dict() 
        RMDefault = dict() 
        descriptorDefault = dict([
                ('Book-to-Price',['ce'+suffix]),
                ('Earnings-to-Price',['ebitda'+suffix if riskModel.modelHack.legacyETP
                                      else 'ibei'+suffix]),
                ('Sales-to-Price',['sale'+suffix]),
                ('Debt-to-Assets',['at'+suffix,'dicl'+suffix,'ltd'+suffix]),
                ('Debt-to-MarketCap',['dicl'+suffix,'ltd'+suffix]),
                ('Plowback times ROE',['ce'+suffix,'ibei'+suffix]),
                ('Sales Growth',['sale'+suffix]),
                ('Earnings Growth',['ibei'+suffix]),
                ('Dividend Yield',[])])
        for factor,descriptors in factorDescriptorMap.items():
            fieldsListDefault = set()
            for descriptor in descriptors: 
                fieldsListDefault = fieldsListDefault.union(set(descriptorDefault.get(descriptor)))
            RMDefault[factor] = list(fieldsListDefault)
        RMNA = dict([('Value',['ce'+suffix,'ebitda'+suffix if riskModel.modelHack.legacyETP
                              else 'ibei'+suffix]),
                      ('Leverage',['at'+suffix,'dicl'+suffix,'ltd'+suffix]),
                      ('Growth',['ce'+suffix,'ibei'+suffix])])
        RM.update(dict([('RMNA',RMNA)]))
        RM.update(dict([('Default',RMDefault)]))
        SCMDefault=dict([
                ('Value',['ce'+suffix]),
                ('Leverage',['at'+suffix,'dicl'+suffix,'ltd'+suffix]),
                ('Growth',['ce'+suffix,'ibei'+suffix,'dps'+suffix,'epsxei'+suffix])
                ])
                       
        SCM.update(dict([('Default',SCMDefault)]))
        SCMUS3 = dict([
                ('Value',['ce'+suffix,'ibei'+suffix,'eps_median_ann']),
                ('Leverage',['at'+suffix,'dicl'+suffix,'ltd'+suffix]),
                ('Growth',['ibei'+suffix,'eps_median_ann','sale'+suffix,
                           'rev_median_ann'])
                ])
        SCM.update(dict([('SCMUS3',SCMUS3)]))
        factorFieldsDict.update(dict([('SCM',SCM),('RM',RM)]))
        query = """ 
                select name,description,id from meta_codes where 
                code_type IN ('asset_dim_fund_currency:item_code',
                'asset_dim_esti_currency:item_code') """ 
        self.marketDB.dbCursor.execute(query) 
        (name,des,code) = zip(*self.marketDB.dbCursor.fetchall())
        self.itemDescMap = dict(zip(name,des))
        self.itemCodeMap = dict(zip(name,code))
        self.factorFieldsDict = factorFieldsDict 
    

    def findHisFund(self,itemCode,sid,EstMode=False): 
        query = """ select eff_dt,value,dt from sub_issue_fund_currency
                    where item_code =:code_arg 
                    and sub_issue_id =:id_arg
                    and rev_del_flag = 'N'
                    order by eff_dt desc,dt desc""" 
        if EstMode is True: 
            query.replace('sub_issue_fund_currency','sub_issue_esti_currency')
        self.modelDB.dbCursor.execute(query,dict([('code_arg',itemCode),
                                                  ('id_arg',sid)]))
        result = self.modelDB.dbCursor.fetchall() 
        for idx, row in enumerate(result): 
            if idx == 0: 
                today = row[0] 
            else: 
                if row[0]!= today and row[2].date()<= self.prevDate: 
                    last = [row[1], row[0]] 
                    return last 
        return (None,None)

    def decomposeExposureChange(self,factor,date,sids,riskModel,currExp,prevExp):                   
        sids = [sid.getSubIDString() for sid in sids]
        if not hasattr(self,'factorFieldsDict'):
            self.buildFactorFieldsDict(riskModel,riskModel.quarterlyFundamentalData)
        if riskModel.rms_id in (170,171,172,173):
            includeEst = True 
        else: 
            includeEst = False
        fundChangeDict = self.getFundamentalChange(sids,date,includeEst)
        resultList = list() 
        if not isinstance(factor,CompositeFactor): 
            if riskModel.rms_id in (170,171,172,173):
                fields = self.factorFieldsDict.get('SCM').get('SCMUS3').get(factor.name)
                logging.info("Decomposing Factor %s for US3"%factor.name)
            else: 
                fields = self.factorFieldsDict.get('SCM').get('Default').get(factor.name)
                logging.info("Decomposing Factor %s for SCM"%factor.name)
        else: 
            if riskModel.rms_id in (101,102,125,126):
                logging.info("Decomposing Factor %s for NA Model"%factor.name) 
                fields = self.factorFieldsDict.get('RM').get('RMNA').get(factor.name)
            else: 
                fields = self.factorFieldsDict.get('RM').get('Default').get(factor.name)   
                logging.info("Decomposing Factor %s for Regional Model"%factor.name)
        alert = "" 
        useTotalAssets = True 
        if factor.name == "Growth" :
            if useTotalAssets is False: 
                fields = list(set(fields)-set(['at_qtr','at_ann']))
            if riskModel.proxyDividendPayout is True: 
                alert = " (Proxy Dividend. Check cdiv History)"
        
        if riskModel.quarterlyFundamentalData is not None : 
            if riskModel.quarterlyFundamentalData is True: 
                suffix = '_qtr'
            else: 
                suffix ='_ann'
        else:
            if riskModel.rms_id in (185,186,187,188):
                suffix = '_qtr'
                
        if 'dicl'+suffix in fields and 'ltd'+suffix in fields:
            fields.remove('dicl'+suffix)
            fields.remove('ltd'+suffix)
            fields.extend(['dicl'+suffix, 'ltd'+suffix])
        for idx,sid in enumerate(sids): 
            result = [idx+1,sid[:-2],self.sedolMap.get(ModelID.ModelID(string=sid[:-2]))] 
            for field in fields: 
                if sid not in fundChangeDict.keys(): 
                    result.append('--')
                elif field not in fundChangeDict.get(sid).keys(): 
                    result.append('--')
                else: 
                    last = self.findHisFund(self.itemCodeMap.get(field),sid)
                    if last[0] is None and riskModel.rms_id in (170,171,172,173): 
                        last = self.findHisFund(self.itemCodeMap.get(field),sid,True)
                    result.append('%.3f / %s (%s)'%(
                            fundChangeDict.get(sid).get(field)[0],
                            '%.3f'%last[0] if last[0] is not None else '--',
                            last[1].strftime('%Y-%m-%d') if last[1] is not None else'--'))
            subID = ModelDB.SubIssue(sid)
            result.extend([currExp.get(subID),prevExp.get(subID)])
            result.append("--" if result[-2]=='--' or result[-1] =='--'
                          else "UP" if float(result[-2])-float(result[-1])>0 else "DOWN")
            resultList.append(result)

        data = Utilities.Struct()
        data.header=["#","MODELID","SEDOL"]
        data.header.extend([self.itemDescMap.get(field).replace(',','').replace('from quarterly report','Qtr ').replace('from annual report', 'Ann ').replace('in million currency','(M)')for field in fields])
        data.header.extend(["CURR", "PREV", "CHANGE SIGN"])
        data.content= numpy.matrix(resultList) 

        data.reportName = "Explaining Exposure Jump for %s"\
                            %factor.name + alert 
        data.description = "Explaining Exposure Jump for %s"\
                            %factor.name + alert 
        return data 
                
                    
    def returnLargeExposureChanges(self,mythreshold=2.0): 
        logging.info("Processing Large Exposure Change")
        sectionName = "FACTOR EXPOSURES"
        validModelSelectorList = [] 
        for model in self.modelSelectorList: 
            if not isinstance(model,MFM.StatisticalFactorModel):
                validModelSelectorList.append(model) 
        self.headerMap["FUND UPD?"] = dict() 
        
        allFactorNames = set() 
        factorModelMap = dict() 
        for modelSelector in validModelSelectorList: 
            position = self.mnemonicPosMap.get(modelSelector.mnemonic[2:])
            expM = self.expMList[position]
            p_expM = self.prevExpMList[position]
            factorNames = expM.getFactorNames() 
            p_factorNames = p_expM.getFactorNames() 
            validFactors = [] 
            # Exclude Statistical Factors 
            # Exclude Factors which do not exist on the previous day 
            for factor in factorNames: 
                if expM.checkFactorType(factor,ExposureMatrix.StatisticalFactor) is not True: 
                    if factor in p_factorNames: 
                        validFactors.append(factor) 
                        factorModelMap.setdefault(factor,list()).append(modelSelector.mnemonic[2:])
                    else: 
                        logging.info("Ignoring New Factor %s"%factor)
            allFactorNames = allFactorNames.union(set(validFactors))
            styleFactors = list(set(self.typeFactorMap['Style']))
            if self.modelAssetType == "Future": 
                industryFactors = list() 
            else: 
                industryFactors = list(set(self.typeFactorMap['Industry']))
            factors = styleFactors + industryFactors 
            factorMap = dict((factor.name,factor) for factor in factors)
            compositeFactors = [f for f in factors if isinstance(f,CompositeFactor)]
            compositeFactorNames = [f.name for f in compositeFactors]
            # Putting Composite Factors at First 
            styleFactors = compositeFactors + list(set(styleFactors).difference\
                                                       (compositeFactors)) 
            
        prevIndDict = dict() 
        currIndDict = dict() 
        if len(styleFactors) != 0 : 
            styleFactorMap = dict([(factor.name,factor) for factor in styleFactors])
        
        # Check Style Factors Only, Display Industry Exposure Change as Industy Change 
        for idx,factorList in enumerate([[factor.name for factor in styleFactors],
                                         [factor.name for factor in industryFactors]]):
            lc_factors = 0 
            tempDataList = [] 
            for factor in factorList:
                threshold = mythreshold
                if factor =='Size':
                    threshold = 0.5
                pos = self.mnemonicPosMap[factorModelMap[factor][0]]
                riskModel = self.modelSelectorList[pos]
                (currExp,prevExp,thres) = self.prepExposuresForFactor(pos,factor,threshold)
                lc_assets = list(currExp.keys())
                rank_by_cap = numpy.argsort(-numpy.array([self.assetCapDict.get(asset)for \
                                                              asset in lc_assets]))
                records = len(lc_assets)
                rank_by_cap = rank_by_cap[:self.maxRecords]
                lc_assets = [lc_assets[i] for i in rank_by_cap]

                if idx ==0:
                    if len(lc_assets)>0: 
                        lc_factors +=1 
                        estuDict = dict((s,1) for s in lc_assets if s in self.wholeEstUniv)
                        for asset in lc_assets: 
                            if asset not in estuDict: 
                                estuDict[asset] = '--'
                        self.headerMap["ESTU"].update(estuDict)
                        if factor in ['Value','Growth','Leverage']:
                            sidArgs = ['sid%d' % i for i in range(len(lc_assets))]
                            query = """SELECT sub_issue_id FROM sub_issue_fund_currency_active
                                       WHERE sub_issue_id IN (%(sids)s)
                                       AND eff_dt >= :dt_arg""" % {
                                    'sids': ','.join([':%s' % arg for arg in sidArgs])}
                            dataDict = dict()
                            dataDict['dt_arg'] = self.date - datetime.timedelta(5)
                            dataDict.update(dict(zip(sidArgs, 
                                [sid.getSubIDString() for sid in lc_assets])))
                            self.modelDB.dbCursor.execute(query, dataDict)
                            newFundDataIds = set(r[0] for r in self.modelDB.dbCursor.fetchall())
                            newFundNameMap = dict([(self.nameMap.get(ModelID.ModelID(string=sid[:-2])),ModelDB.SubIssue(sid))\
                                                       for sid in  newFundDataIds])
                            newFundDict = dict((sid, '1') for sid in lc_assets\
                                                  if sid.getSubIDString() in newFundDataIds)
                            newFundDict.update((sid,'--') for sid in lc_assets\
                                                   if sid.getSubIDString() not in newFundDataIds)
                            for sid in lc_assets : 
                                if sid.getSubIDString() not in newFundDataIds: 
                                    name = self.nameMap.get(sid.getModelID()) 
                                    if name is not None: 
                                        nameList = name.split(" ") 
                                        relName = " ".join(nameList[:-1])
                                        relName2 = " ".join(nameList[:-2])
                                        if relName2 in newFundNameMap.keys(): 
                                            relName = relName2 
                                        if relName in newFundNameMap.keys(): 
                                            if currExp.get(sid) == currExp.get(newFundNameMap.get(relName))\
                                                    and prevExp.get(sid) == prevExp.get(newFundNameMap.get(relName)):
                                                newFundDict[sid] = '1'
                            extraHeader = ["CAP (%s m)"%self.currency,
                                           "CURR %s"%factor,"PREV %s"%factor,
                                           "TR %","ESTU", "ESTU RANK","FUND UPD?"]
                            self.headerMap["FUND UPD?"].update(newFundDict) 
                            
                        else: 
                            extraHeader = ["CAP (%s m)"%self.currency,
                                           "CURR %s"%factor,"PREV %s"%factor,
                                           "TR %","ESTU", "ESTU RANK"]
                            if self.modelAssetType == "Future":
                                extraHeader = extraHeader[1:-2]
                        self.headerMap["CURR %s"%factor] = currExp
                        self.headerMap["PREV %s"%factor] = prevExp                
                        data = self.prepAssetLevelInfo(lc_assets,extraHeader,
                                                       appendIndustry = self.appendIndustry,
                                                       appendAssetType = self.appendAssetType,
                                                       modelAssetType = self.modelAssetType)
                        if self.modelAssetType == "Future": 
                            data.reportName = "Large Commodity Exposure Change for %s"%factor                            
                        else: 
                            data.reportName = "Large Exposure Change for %s"%factor
                        data.description = "%d Assets with Exposure Change for %s %s >%.1f"%\
                            (records,",".join(model for model in factorModelMap[factor]),
                             factor,thres)
                        data.reportSection = sectionName 
                        tempDataList.append(data)
                        if factor in ['Value','Growth','Leverage']:
                            factorObj = styleFactorMap.get(factor)
                            try:
                                explaineData = self.decomposeExposureChange(factorObj,self.date,
                                                                        lc_assets,riskModel,
                                                                        currExp,prevExp)
                                explaineData.reportSection = sectionName 
                                tempDataList.append(explaineData)
                            except:
                                logging.error("Decomposition Failed")
                else: 
                    if len(lc_assets)>0:
                        lc_factors+=1
                        estuDict = dict((s,1) for s in lc_assets if s in self.wholeEstUniv)
                        for asset in lc_assets:
                            if currExp[asset]=='1.00':
                                currIndDict[asset] =factor 
                            else: 
                                prevIndDict[asset] = factor 
                            if asset not in estuDict: 
                                estuDict[asset] = '--'
                        self.headerMap["ESTU"].update(estuDict)
            if idx==0:
                factorType = "Style Factors"
                data = Utilities.Struct() 
                data.reportName = "Factor Type %d"%lc_factors 
                data.reportSection = sectionName 
                data.description = "%d / %d %s have Large Exposure Changes"%(lc_factors, len(factorList),
                                                                             factorType)
                self.coreDataList.append(data)
                self.coreDataList.extend(tempDataList) 

            else: 
                if len(currIndDict)!=0:
                    # assert (len(currIndDict)==len(prevIndDict))
                    #DIA-4070 Extra check on SPACs
                    prevIndDictKeys = list(prevIndDict.keys())
                    for k in prevIndDictKeys:
                        if k not in currIndDict:
                            modeldbID = k.getSubIdString()[:-2]
                            query = """select * from ISSUE_MAP im, marketdb_global.classification_active_int classint where im.MODELDB_ID = '%s'
                            and im.MARKETDB_ID=classint.AXIOMA_ID and classint.CLASSIFICATION_ID=6012
                            and classint.FROM_DT < '%s' and classint.THRU_DT > '%s'""" % (modeldbID, self.prevDate, self.prevDate)
                            self.modelDB.dbCursor.execute(query)
                            result = self.modelDB.dbCursor.fetchall()
                            if len(result) == 1:
                                del prevIndDict[k]

                    extraHeader = ["CAP (%s m)"%self.currency,
                               "CURR %s"%factor,"PREV %s"%factor,
                               "TR %","ESTU"]
                    self.headerMap["CURR %s"%factor] = currIndDict
                    self.headerMap["PREV %s"%factor] = prevIndDict            
                    data = self.prepAssetLevelInfo(list(currIndDict.keys()),extraHeader)
                    data.description = "%d Assets with Industry Factor Exposure Changes"%len(currIndDict)
                    data.reportSection = sectionName 
                    factorType = "Style Factors"
                    data.reportName = "Factor Type %d"%lc_factors 
                    self.coreDataList.append(data)

        logging.info("Finished Processing Large Exposure Change")


    def prepExposuresForFactor(self,modelPosition,factorName,threshold): 
        model = self.modelSelectorList[modelPosition]
        expM = self.expMList[modelPosition] 
        factors = expM.getFactorNames() 
        if factorName not in factors: 
            return None 
        currExpDict = dict() 
        prevExpDict = dict() 
        expMatrix = expM.getMatrix() 
        p_expM = self.prevExpMList[modelPosition]
        p_expMatrix = p_expM.getMatrix() 
        assetIdxMap = dict(zip(p_expM.getAssets(), 
                            range(len(p_expM.getAssets()))))
        factorIdx = expM.getFactorIndex(factorName) 
        p_factorIdx = p_expM.getFactorIndex(factorName) 
        
        # Set threshold for different factors 
        if factorName not in expM.getFactorNames(
                        fType=ExposureMatrix.StyleFactor) or \
            Utilities.is_binary_data(expMatrix[factorIdx,:]):
            z = 1.0
        else:
            z = threshold
        assetDict = dict() 
        for j in range(expMatrix.shape[1]):
            sid = expM.getAssets()[j]
            # Only process assets alive in both periods
            if sid not in assetIdxMap:
                continue
            else:
                pos = assetIdxMap[sid]
            prevExp = p_expMatrix[p_factorIdx,pos]
            currExp = expMatrix[factorIdx,j]
            # Flag assets with big changes
            if currExp is not ma.masked and prevExp is not ma.masked:
                if abs(currExp - prevExp) >= z:
                    currExpDict[sid] = '%.2f'%currExp
                    prevExpDict[sid] = '%.2f'%prevExp
            elif currExp is ma.masked and prevExp is ma.masked:
                continue
            elif currExp is ma.masked and prevExp is not ma.masked: 
                currExpDict[sid] = '--'
                prevExpDict[sid] = '%.2f'%prevExp
            else: 
                currExpDict[sid] = '%.2f'%currExp 
                prevExpDict[sid] = '--'

        return (currExpDict,prevExpDict,z) 

    def returnCovarianceMatrixSpecs(self, min_eig=1e-10):
        logging.info("Porcessing Covariance Matrix Specs")
        sectionName = "FACTOR RISKS & COVARIANCES" 
        data = Utilities.Struct() 
        attributeList = ['Eigenvalues in Cov Matrix < %s'%str(min_eig),
                         'Largest Eigenvalue',
                         'Frobenius norm (*1000)',
                         'Previous Frobenius norm (*1000)']
        resultList = [] 
        for idx,modelSelector in enumerate(self.modelSelectorList): 
            cov = self.covMatrixList[idx]
            p_cov = self.p_covMatrixList[idx]
            (eigval, eigvec) = linalg.eigh(cov)
            largestIdx = numpy.argsort(-numpy.array(eigval))
            largest = eigval[largestIdx[0]] 
            (p_eigval, p_eigvec) = linalg.eigh(p_cov)
            bad = numpy.flatnonzero(ma.getmaskarray(ma.masked_where(
                                        eigval < min_eig, eigval)))
            badList = [] 
            result = [modelSelector.mnemonic[2:]]
            if len(bad) == 0: 
                result.append('None')
            else: 
                for i in bad: 
                    badList.append(str(eigval[i]))
                result.append(','.join(bad for bad in badList))
            frobNorm = 1000.0 * numpy.sqrt(numpy.sum(eigval * eigval))
            p_frobNorm = 1000.0 * numpy.sqrt(numpy.sum(p_eigval * p_eigval))
            result.extend([largest,frobNorm,p_frobNorm])
            resultList.append(result) 
        data.reportName = "Cov Specification" 
        data.reportSection = sectionName 
        data.header = ['Model'] + attributeList 
        data.content = numpy.matrix(resultList) 
        data.description = "Covariance Matrix Specification" 
        self.dataList.append(data) 
        logging.info("Finished Porcessing Covariance Matrix Specs")
    
    def checkAgainstSubModels(self): 
        logging.info("Checking factors agaisnt submodels") 
        subModels = [20000,10001,109]
        modelFactorsMap = dict() 
        resultList = list() 
        topModelFactors = self.getFactorsForDate(self.date)
        query = """
                SELECT factor_id+1 from rms_factor 
                WHERE rms_id = :rmsId_arg
                AND from_dt <= :date_arg 
                AND thru_dt > :date_arg"""
        for rmsId in subModels: 
            self.modelDB.dbCursor.execute(query,dict([('date_arg',self.date),('rmsId_arg',rmsId)]))
            modelFactorsMap.update(dict([(rmsId,[item[0] for item in \
                                                     self.modelDB.dbCursor.fetchall()])]))
        for rmsid,factors in modelFactorsMap.items(): 
            missing = set(factors).difference(set(topModelFactors))
            print(missing) 
            print(factors) 
            for sid in missing: 
                resultList.append([sid,self.factorNameDict.get(sid),rmsid])
        data = Utilities.Struct() 
        data.reportName = "Missing Factors Against Sub Models" 
        data.reportSection = "MODEL STRUCTURE"
        if len(resultList) != 0 : 
            data.header = ["SubFactor ID","Factor","Offending Model"]
            data.content = numpy.matrix(resultList)
            data.description = "Missing Factors Exisit!" 
        else: 
            data.description = "No Missing Factors Against Sub Models"
        self.dataList.append(data) 
        logging.info("Finished Checking factors agaisnt submodels") 



    def returnFactorVolChecks(self,threshold=0.005):
        logging.info("Processing Factor Volatility Check")
        sectionName = "FACTOR RISKS & COVARIANCES" 
        if self.modelAssetType == "Universal":
            (risky,lchange,hlFactors,byTypeDataList) = self.prepFactorVolChecks(self.fundamental,
                                                                                threshold,
                                                                                reportByType = True) 
        else:
            (risky,lchange) = self.prepFactorVolChecks(self.fundamental,threshold) 
        for data in [risky,lchange]: 
            data.description = "Fundamental " + data.description 
            data.reportName = "Fundamental " + data.reportName 
        if self.modelAssetType == "Equity" and self.statistical is not None :
            if 'US3' not in self.statistical[0].mnemonic:
                (risky2,lchange2) =self.prepFactorVolChecks(self.statistical,threshold) 
                for data in [risky2,lchange2]:
                    data.description = "Statistical " + data.description 
                    data.reportName = "Statistical " + data.reportName 
                for idx,data in enumerate([risky,risky2,lchange,lchange2]):
                    data.reportSection = sectionName 
                    if idx in [0,1]:
                        self.dataList.append(data)                         
                    else: 
                        self.coreDataList.append(data) 
        else: 
            if self.modelAssetType == "Universal":
                for data in [risky] + byTypeDataList: 
                    data.reportSection = sectionName 
                    self.dataList.append(data)
                lchange.reportSection = sectionName 
                self.dataList.append(lchange) 
                self.returnHistoricVol(hlFactors,self.dates)                
            else: 
                for data in [risky,lchange]: 
                    data.reportSection = sectionName 
                    self.dataList.append(data)
        logging.info("Finished Processing Factor Volatility Check")

    def returnHistoricVol(self,factorList,dates): 
        logging.info("Processing Historical Volatility Check") 
        sectionName = "FACTOR RISKS & COVARIANCES"
        resultList = list() 
        for factor in factorList: 
            sfid  = self.factorSIDMap.get(factor.name)
            if sfid is None: 
                continue 
            datesArg = ['dt%d'%i for i in range(len(dates))]
            myDict = dict([('sfid',sfid)])
            myDict.update(dict(zip(datesArg,dates)))
            query = """SELECT dt, value FROM rmi_covariance 
                       WHERE rms_id  = 30000 
                       AND sub_factor1_id = :sfid 
                       AND sub_factor2_id = :sfid 
                       And dt IN (%(dates)s)
                       order by DT""" %{
                       'dates':','.join([':%s'%arg for arg in datesArg])}
            self.modelDB.dbCursor.execute(query,myDict) 
            dtList,valueList = zip(*self.modelDB.dbCursor.fetchall())
            resultDict = dict(zip([dt.date() for dt in dtList],[math.sqrt(x)*100.00 
                                                                for x in valueList]))
            resultArray = [resultDict.get(date) for date in 
                           dates]
            resultArray_New = [entry if entry is not None else 'null' 
                               for entry in resultArray]
            resultList.append(resultArray_New) 
        data = Utilities.Struct()
        content = numpy.matrix(resultList).transpose()
        dateArray = numpy.array([dt.strftime('%b%d') for dt in dates])
        dateArray2 = numpy.reshape(dateArray,(-1,1))
        data.content = numpy.append(dateArray2,content,1)
        data.header= ["X"] + [factor.name for factor in factorList]        
        data.reportSection = sectionName 
        data.reportName = "Factor Risk History" 
        data.description = "%d Days Factor Risk History"%len(dates)
        data.reportType = "LineChart" 
        self.dataList.append(data)

    def returnHistoricCorr(self,factorPairList,dates): 
        logging.info("Processing Historical Volatility Check") 
        sectionName = "FACTOR RISKS & COVARIANCES"
        resultList = list() 
#        factorPairList = [factorPairList[-1]]
        for (factor1,factor2) in factorPairList: 
            sfid1 = self.factorSIDMap.get(factor1.name)
            sfid2 = self.factorSIDMap.get(factor2.name)
            sfid1_new = max([sfid1,sfid2])
            sfid2_new = min([sfid1,sfid2])
            datesArg = ['dt%d'%i for i in range(len(dates))]
            myDict = dict([('sfid1',sfid1_new)])
            myDict['sfid2']=sfid2_new
            myDict.update(dict(zip(datesArg,dates)))
            query = """SELECT dt, value FROM rmi_covariance 
                       WHERE rms_id  = 30000 
                       AND sub_factor1_id = :sfid1
                       AND sub_factor2_id = :sfid2
                       And dt IN (%(dates)s)
                       order by DT""" %{
                       'dates':','.join([':%s'%arg for arg in datesArg])}
            self.modelDB.dbCursor.execute(query,myDict) 
            dbresult = self.modelDB.dbCursor.fetchall()
            myDict['sfid1'] = min([sfid1,sfid2])
            myDict['sfid2'] = max([sfid1,sfid2])
            self.modelDB.dbCursor.execute(query,myDict) 
            dbresult2 = self.modelDB.dbCursor.fetchall()
            dbresult1Dict = dict(dbresult) 
            dbresult2Dict = dict(dbresult2)
            for key,value in dbresult2Dict.items(): 
                if key not in dbresult1Dict.keys(): 
                    dbresult1Dict[key] = value
            dtList = sorted(dbresult1Dict.keys())
            corrList = [dbresult1Dict[key] for key in dtList]
            if len(dtList) != len(dates): 
                datesArg = ['dt%d'%i for i in range(len(dtList))]
                query = """SELECT dt, value FROM rmi_covariance 
                       WHERE rms_id  = 30000 
                       AND sub_factor1_id = :sfid1
                       AND sub_factor2_id = :sfid2
                       And dt IN (%(dates)s)
                       order by DT""" %{
                       'dates':','.join([':%s'%arg for arg in datesArg])}
            myDict2 = dict() 
            myDict2.update(dict(zip(datesArg,dtList)))
            myDict2['sfid1'] = sfid1_new
            myDict2['sfid2'] = sfid1_new
            self.modelDB.dbCursor.execute(query,myDict2)
            dtList1,sfid1voList = zip(*self.modelDB.dbCursor.fetchall())
            myDict3 = dict() 
            myDict3.update(dict(zip(datesArg,dtList)))
            myDict3['sfid1'] = sfid2_new 
            myDict3['sfid2'] = sfid2_new 
            self.modelDB.dbCursor.execute(query,myDict3)
            dtList2,sfid2voList = zip(*self.modelDB.dbCursor.fetchall())
            resultDict = dict(zip([dt.date() for dt in dtList],
                                  corrList/numpy.sqrt(abs(numpy.multiply(numpy.array(sfid1voList),
                                                                  numpy.array(sfid2voList))))))
            resultArray = [resultDict.get(date) for date in 
                        dates]
            resultArray_New = [entry if entry is not None else 0 
                               for entry in resultArray]
            resultList.append(resultArray_New) 
        data = Utilities.Struct()
        content = numpy.matrix(resultList).transpose()
        dateArray = numpy.array([dt.strftime('%b%d') for dt in dates])
        dateArray2 = numpy.reshape(dateArray,(-1,1))
        data.content = numpy.append(dateArray2,content,1)
        data.header= ["X"] + [factor1.name + ' / ' + factor2.name for (factor1,factor2) in factorPairList]        
        data.reportSection = sectionName 
        data.reportName = "Factor Correlation History" 
        data.description = "Factor Correlation History (0 means not available)" 
        data.reportType = "LineChart" 
        self.dataList.append(data)

    def prepFactorVolChecks(self,modelSelectorList,threshold=0.005,reportByType=False):
        validModelSelectorList = [] 
        for model in modelSelectorList: 
            if not isinstance(model, MFM.StatisticalFactorModel) and not isinstance(model, EquityModel.StatisticalModel):
                validModelSelectorList.append(model) 
        resultList =[]
        cresultList = [] 
        headerList = [] 
        for modelIdx,model in enumerate(validModelSelectorList): 
#        for modelIdx,model in enumerate(modelSelectorList): 
            if self.modelAssetType == 'Universal':
                pos = 0 
                factorTypeMap = self.factorTypeMap 
            else: 
                pos = self.mnemonicPosMap[model.mnemonic[2:]]
                expM = self.expMList[pos] 
                factorTypeMap = dict()
                for fType in expM.factorIdxMap_.keys():
                    for fName in expM.factorIdxMap_[fType].keys():
                        factorTypeMap[fName] = fType
            model.setFactorsForDate(self.prevDate,self.modelDB)
            pfactors = model.factors
            p_factorIdxMap = dict((pfactors[i].factorID, i) \
                                    for i in range(len(pfactors)))
            model.setFactorsForDate(self.date,self.modelDB)
            factors = model.factors
            factorVols = numpy.sqrt(numpy.diag(
                                    self.covMatrixList[pos]))
            p_factorVols = numpy.sqrt(numpy.diag(
                                    self.p_covMatrixList[pos]))
            rank = numpy.argsort(-factorVols)
            if self.modelAssetType == "Universal": 
                validFactorNames = [self.factorNameDict[i] for i in self.validFactorsForDate]
                rank = [i for i in rank if factorTypeMap[factors[i].name] \
                        != ExposureMatrix.StatisticalFactor \
                        and factors[i].name in validFactorNames]
            else: 
                rank = [i for i in rank if factorTypeMap[factors[i].name] \
                        != ExposureMatrix.StatisticalFactor]
                
#            hlRank = rank[:5] + rank[-6:-1] 
            hlRank = rank[:5] + rank[-5:] 
            hlFactors = [factors[i] for i in hlRank]
            for idx,factor in enumerate(hlFactors): 
                result = [factor.name,self.factorTypeMap[factor.name].name,
                          '%.2f'%(factorVols[hlRank[idx]]*100.0)]
                if result[1] == 'Currency': 
                    result[0] = '%s (%s)'%(factor.name,factor.description)
                if modelIdx == 0: 
                    resultList.append(result) 
                else: 
                    resultList[idx].extend(result) 
            headerList.extend(["%s FACTOR"%model.mnemonic[-2:],"TYPE","RISK (%)"])
            
            if reportByType is True: 
                returnByTypeList = list() 
                typeFactorDict = dict() 
                for factor in self.validFactorsForDate: 
                    typeFactorDict.setdefault(self.factorTypeMap[self.factorNameDict[factor]],list())\
                        .append(self.factorNameDict[factor])
                for factorType,typeFactors in typeFactorDict.items(): 
                    resultList2 = list()
                    if len(typeFactors)<10: 
                        logging.info("Ignoring factor type %s since number of factors is less than 10"%factorType)
                    else: 
                        localRankTemp = [i for i in rank if factors[i].name in typeFactors]
                        localRank = localRankTemp[:5]+localRankTemp[-5:]
                        hlFactorsByType = [factors[i] for i in localRank]
                        for idx,factor in enumerate(hlFactorsByType): 
                            result = [factor.name,self.factorTypeMap[factor.name].name,
                                      '%.2f'%(factorVols[localRank[idx]]*100.0)]
                            if result[1] == 'Currency': 
                                result[0] = '%s (%s)'%(factor.name,factor.description)
                            if modelIdx == 0: 
                                resultList2.append(result) 
                            else: 
                                resultList2[idx].extend(result)                     
                        data = Utilities.Struct() 
                        data.header = headerList 
                        data.content = numpy.matrix(resultList2) 
                        factorTypeName = factorType.name.replace(" ","_")
                        factorTypeName = factorTypeName.replace("-","_")
                        data.description = "%s 5 Most / Least Risky Factors"%factorTypeName 
                        data.reportName = data.description
                        returnByTypeList.append(data) 
            counter = 1
            for i in range(len(factors)):
                if factorTypeMap[factors[i].name]==ExposureMatrix.StatisticalFactor:
                    continue
                factorID = factors[i].factorID
                pidx = p_factorIdxMap.get(factorID)
                if pidx is not None:
                    diff = abs(factorVols[i] - p_factorVols[pidx])
                    if diff > threshold:
                        result = [factors[i].name,factorTypeMap[factors[i].name].name,
                                 '%.2f'%(factorVols[i]*100.0),'%.2f'%(p_factorVols[pidx]*100.0),
                                  model.mnemonic[-2:] if model.mnemonic[-1]=='H' else model.mnemonic[-4:]]
                        if self.modelAssetType == "Universal": 
                            result = result[:-1]
                        if result[1] == 'Currency': 
                            result[0] = '%s (%s)'%(factors[i].name,factors[i].description)
                        cresultList.append(result) 

        data = Utilities.Struct() 
        data.header = headerList 
        data.content = numpy.matrix(resultList) 
        if self.modelAssetType == "Equity":
            data.reportName = "5 Most / Least Risky Factors"
        else: 
            data.reportName = "5 Most / Least Risky Future Factors"            
        data.description = "5 Most / Least Risky Factors"

        cdata = Utilities.Struct()
        cdata.content = numpy.matrix(cresultList) 
        if self.modelAssetType == "Future":
            cdata.reportName = "Large Changes in Factor Risk > %.2f%%"%(threshold*100)
        else: 
            cdata.reportName = "Equity Large Changes in Factor Risk > %.2f%%"%(threshold*100) 
        cdata.description = "Large Changes in Factor Risk > %.2f%%"%(threshold*100)

        if len(cresultList)>0:
            if self.modelAssetType == "Universal":
                cdata.header = ["FACTOR","TYPE","RISK (%)", "PREV (%)"]
            else:
                cdata.header = ["FACTOR","TYPE","RISK (%)", "PREV (%)","MODEL"]
            
        if len(cresultList)==0: 
            cdata.description = "No " + cdata.description
        
        if self.modelAssetType == "Universal": 
            if reportByType is False: 
                return (data,cdata,hlFactors) 
            else: 
                return (data,cdata,hlFactors,returnByTypeList)                 
        else: 
            return (data,cdata)

    def returnFactorCorrChecks(self): 
        logging.info("Procesing Factor Correlation Check")
        sectionName = "FACTOR RISKS & COVARIANCES" 
        for model in self.fundamental: 
            if self.modelAssetType == "Universal":
                data,factorPairList = self.prepFactorCorrChecks([model])
            else: 
                data= self.prepFactorCorrChecks([model])
            data.reportSection = sectionName 
            self.dataList.append(data) 
        if self.modelAssetType == "Universal":
            self.returnHistoricCorr(factorPairList,self.dates)
        logging.info("Finished Procesing Factor Correlation Check")

    def prepFactorCorrChecks(self,modelSelectorList):
        resultList = [] 
        headerList = [] 
        for modelIdx,model in enumerate(modelSelectorList): 
            if self.modelAssetType == "Universal": 
                pos = 0 
                factorTypeMap = self.factorTypeMap
                factors = model.factors 
            else: 
                pos = self.mnemonicPosMap[model.mnemonic[2:]]            
                model.setFactorsForDate(self.date,self.modelDB)
                factors = model.factors
                expM = self.expMList[pos]
                factorTypeMap = dict()
                for fType in expM.factorIdxMap_.keys():
                    for fName in expM.factorIdxMap_[fType].keys():
                        factorTypeMap[fName] = fType
            factorCov = self.covMatrixList[pos]
            factorVols = numpy.sqrt(numpy.diag(factorCov))
            factorCorr = factorCov / ma.outer(factorVols, factorVols)
            factorCorr = factorCorr.filled(0.0)
            allFactorCorrs = [(factorCorr[i,j], i, j) for i in range(
                            len(factors)) for j in range(len(factors)) if i > j]
            notAllowed = [ExposureMatrix.CurrencyFactor, 
                          ExposureMatrix.StatisticalFactor]
            allFactorCorrs = [n for n in allFactorCorrs if
                      factorTypeMap[factors[n[1]].name] not in notAllowed and  factorTypeMap[factors[n[2]].name] not in notAllowed]
            rank = list(numpy.argsort(-numpy.array([n[0] for n in allFactorCorrs])))
            posnegRank = rank[:10] + rank[-10:] 

            factorPairList = list()
            for idx,rankIdx in enumerate(posnegRank): 
                fIdx0 = allFactorCorrs[rankIdx][1]
                fIdx1 = allFactorCorrs[rankIdx][2]
                result = [factors[fIdx0].name,factorTypeMap[factors[fIdx0].name].name,
                          factors[fIdx1].name,factorTypeMap[factors[fIdx1].name].name,
                          '%.2f'%allFactorCorrs[rankIdx][0]]
                if modelIdx == 0: 
                    resultList.append(result) 
                else: 
                    resultList[idx].extend(result) 
                factorPairList.append((factors[fIdx0],factors[fIdx1]))
            headerList.extend(["%s FACTOR"%model.mnemonic[7:],"TYPE","%s FACTOR"%model.mnemonic[7:],
                               "TYPE","CORR"])
        
        data = Utilities.Struct() 
        data.header = headerList 
        data.content = numpy.matrix(resultList) 
        data.reportName = "10 Most Positively / Negatively Correlated (non-currency) %s Factors"\
            %(model.mnemonic[2:])
        data.description = "10 Most Positively / Negatively Correlated (non-currency) %s Factors"\
            %(model.mnemonic[2:])
        if self.modelAssetType == "Universal": 
            return (data,factorPairList) 
        else: 
            return data 


    def returnBetaDiscrepencies(self,threshold=0.25):
        logging.info("Processing Beta Diagnosis")
        sectionName = "ASSET PREDICTED & HISTORICAL BETA" 
        logging.info("Processing Predicted Beta")
        for model in self.modelSelectorList: 
            data = self.prepBetaDiscrepancies(model,threshold) 
            data.reportSection = sectionName
            self.dataList.append(data) 
        logging.info("Processing Histrocial Beta")
        data = self.prepBetaDiscrepancies(self.modelSelectorList[0],threshold,hist=True) 
        data.reportSection = sectionName 
        self.dataList.append(data) 
        logging.info("Finished Processing Beta Diagnosis")


    def getSpecificRisks(self,rmi, sidList):
        INCR = 100
        sidArgList = [('sid%d' % i) for i in range(INCR)]
        srDict = dict() 
        query = """
            SELECT sub_issue_id, value
            FROM rmi_specific_risk sr, sub_issue si
            WHERE sr.rms_id = :rms_arg AND sr.dt = :date_arg
            AND sub_issue_id IN (%(sids)s)
            AND si.sub_id = sr.sub_issue_id 
            AND si.from_dt <= sr.dt AND si.thru_dt > sr.dt
            """ %{
                'sids': ','.join([(':%s' % i) for i in sidArgList])}
        sidStrs = [sid.getSubIDString() for sid in sidList]
        defaultDict = dict((i, None) for i in sidArgList)
        sidInfo = list()
        for sidChunk in listChunkIterator(sidStrs, INCR):
            myDict = dict(defaultDict)
            myDict.update(dict(zip(sidArgList, sidChunk)))
            myDict['rms_arg'] =rmi.rms_id 
            myDict['date_arg'] = rmi.date
            self.modelDB.dbCursor.execute(query, myDict)
            srDict.update(dict([(ModelDB.SubIssue(i), j) for (i,j) in self.modelDB.dbCursor.fetchall()]))
        return srDict 

    def getSpecificCovariances(self,rmi, sidList):
        INCR = 100
        sidArgList = [('sid%d' % i) for i in range(INCR)]
        sidCovMap = dict()
        query = """
            SELECT sub_issue1_id, sub_issue2_id, value
            FROM rmi_specific_covariance sr, sub_issue s1, sub_issue s2
            WHERE sr.rms_id = :rms_arg AND sr.dt = :date_arg
            AND (s1.sub_id IN (%(sids)s)
            OR s2.sub_id IN (%(sids)s))
            AND s1.sub_id = sr.sub_issue1_id 
            AND s2.sub_id = sr.sub_issue2_id 
            AND s1.from_dt <= sr.dt AND s1.thru_dt > sr.dt
            AND s2.from_dt <= sr.dt AND s2.thru_dt > sr.dt
            """ %{
                'sids': ','.join([(':%s' % i) for i in sidArgList])}
        sidStrs = [sid.getSubIDString() for sid in sidList]
        defaultDict = dict((i, None) for i in sidArgList)
        sidInfo = list()
        for sidChunk in listChunkIterator(sidStrs, INCR):
            myDict = dict(defaultDict)
            myDict.update(dict(zip(sidArgList, sidChunk)))
            myDict['rms_arg'] =rmi.rms_id 
            myDict['date_arg'] = rmi.date
            self.modelDB.dbCursor.execute(query, myDict)
            for (sid1, sid2, cov) in self.modelDB.dbCursor.fetchall():
                subid1 = ModelDB.SubIssue(sid1)
                subid2 = ModelDB.SubIssue(sid2)
                sidCovMap.setdefault(subid1, dict())[subid2] = cov
        return sidCovMap 

    def returnPredictedRealisedRisk(self):
        logging.info("Processing Predicted Risk /Realised Risk Study")
#        dateList = modelDB.getDates(self.rmgList,self.date,188) 
#        lookbackDays = 125
        lookbackFrequency = 6
        cumFrequency = 20 
#        period = 50 
        period = 20
        sectionName = "PREDICTED RISK (Using %d Days Cumulative Return)"%cumFrequency
        dateList = modelDB.getDates(self.rmgList,self.date,period*cumFrequency,True) 
        sampledDateListIndex = list(range(period*cumFrequency+1))[0::cumFrequency]
        sampledDateList = [dateList[i] for i in sampledDateListIndex]

        lc = 1-(2.0/lookbackFrequency)**0.5
        hc = 1+(2.0/lookbackFrequency)**0.5

        univ = self.fullsubID
        mdl2sub = dict((n.getModelID(),n) for n in univ)
        resultList = [] 
        imp = [(k,v) for k,v in self.modelMarketMap.items()]
        expMList =[]; covList =[]; srDictList = []; scDictList =[]
        if self.modelSelectorList[0].mnemonic[:7] == 'AXAPxJP':
            selectedBmk = "FTSE All-World Asia Pacific ex Japan"
        elif self.modelSelectorList[0].mnemonic[:7] == 'AXDMxUS':
            selectedBmk = "STOXX Global 1800 ex USA"
        else:
            selectedBmk=MODEL_PROXY_INDEX_MAP.get(self.modelSelectorList[0].mnemonic[:4])
        port = modelDB.getIndexConstituents(selectedBmk,self.date,
                                            self.marketDB,rollBack=20,
                                            issueMapPairs=imp)
        (assets,weights) = zip(*[(mdl2sub[a],w)for(a,w)\
                                     in port if a in mdl2sub])
        weights = numpy.array(weights)
        weights /= numpy.sum(weights)
        port = list(zip(assets,weights))
        apiassets = [n.getSubIDString()[1:-2] for n in assets]
        apiport= dict(zip(apiassets,weights))
        weights = weights[:, numpy.newaxis]
#        print apiport
        # Calculate Benchmark Return 
#        modelDB.setCumReturnCache(len(sampledDateList))
        logging.info("Load Cumulative Return for Realized Return Calculation")
        cumReturnMatrix = modelDB.loadCumulativeReturnsHistory(sampledDateList,assets)
        logging.info("Finished loading Cumulative Return for Realized Return Calculation")
        returnMatrix = cumReturnMatrix.data[:,1:]/\
            cumReturnMatrix.data[:,0:-1]
        returnMatrix = returnMatrix.filled(0.0)
        for row in range(len(assets)):
            for column in range(len(sampledDateList)-1):
                returnMatrix[row][column] = returnMatrix[row][column]-1
        bmkReturn = numpy.dot(numpy.transpose(weights), returnMatrix)
        bmkReturn = numpy.array(bmkReturn).flatten()

        # print len(sampledDateList), "sampledDateList"
        # print len(bmkReturn), "bmkReturn"

        # Calculate Benchmark Risk
        bmkRisk = [] 
        for idx in range(period-lookbackFrequency):
            returnList = bmkReturn[idx:idx+lookbackFrequency]
            bmkRisk.append(numpy.std(returnList)*(250.0/cumFrequency)**0.5)
        # print len(bmkRisk),"bmkRisk"
        logging.info("Before Starting Prediction Calculation")
        modelPortfolioRisks =[]
        # Calculate Total Risk (Predicted) 
        logging.info("Starting Calculating Total Risk")

        for rm in self.modelSelectorList:
#            portfolioRiskList = [] 
            # Historical with flatfiles available 
            portfolioRiskList=pythonAPIUtiles.getPortTotalrisk(sampledDateList[1:-1],apiport,rm.mnemonic,self.modelPath)
#             for idx,date in enumerate(sampledDateList[1:]):
            date = sampledDateList[-1]
            logging.info("Setting Factors for Date")
            rm.setFactorsForDate(date,self.modelDB)
            logging.info("Finished Setting Factors for Date")                
            logging.info("Getting RiskModel Instance")
            rmi = modelDB.getRiskModelInstance(rm.rms_id,date) 
            logging.info("Finished Getting RiskModel Instance")
            logging.info("Loading Exposure Matrix")
            expM = rm.loadExposureMatrix(rmi,self.modelDB,assetList = assets)
#            expM = rm.loadExposureMatrix(rmi,self.modelDB)            
            logging.info("Finished Loading Exposure Matrix")
            logging.info("Loading Covariance Matrix") 
            (cov,factors) = rm.loadFactorCovarianceMatrix(rmi,self.modelDB)
            logging.info("Finisehd Loading Covariance Matrix") 
            logging.info("Loading SRDict and SCDict")
            srDict = self.getSpecificRisks(rmi,assets)
            scDict = self.getSpecificCovariances(rmi,assets)
            logging.info("Finisehd laoding srdict and scdict")
            logging.info("Computing Total Predicted Risk")
            portfolioRisk = self.compute_total_risk_portfolio(port,expM,cov,srDict,scDict)
            logging.info("Finished Computing Total Predicted Risk")

            portfolioRiskList.append(portfolioRisk)
            modelPortfolioRisks.append(portfolioRiskList) 
        logging.info("Finished Calculating Total Risk")
        # print len(portfolioRiskList),"portfolioRiskList"

        modelbias = [] 
        for idx, portfolioRiskList in enumerate(modelPortfolioRisks):
            z = numpy.array(bmkReturn)*(250.0/cumFrequency)**0.5/numpy.array(portfolioRiskList)
            biaslist = [] 
            for idx in range(len(sampledDateList)-1-lookbackFrequency):
                zlist = z[idx:idx+lookbackFrequency]
                biaslist.append(numpy.std(zlist)**0.5)
            modelbias.append(biaslist)

        chartDateList = sampledDateList[lookbackFrequency+1:]
        # print len(chartDateList),"chartDateList"

        resultList = [] 
        for idx in range(len(chartDateList)):
            result = [chartDateList[idx].strftime('%b%d-%y'),
                               (bmkRisk[idx]*100),0]
            for portfolioRiskList in modelPortfolioRisks:
                selectedPortRiskList = portfolioRiskList[lookbackFrequency:]
                result.append(selectedPortRiskList[idx]*100)
            resultList.append(result) 
        
#        print resultList
        data= Utilities.Struct() 
        data.description = "Predicted Risk vs Realized Risk (%s)"%(selectedBmk)
        data.reportSection = sectionName 
        data.header=['Date','Realized Risk','boolean']
        for model in self.modelSelectorList: 
            data.header.append(('%s'%model.mnemonic[2:]))
        data.content = numpy.matrix(resultList)
        data.reportName = "Predicted vs Realized Risk" 
        data.reportType = "LineChart" 
        self.dataList.append(data) 

        resultList = [] 
        for idx in range(len(chartDateList)):
            result=[chartDateList[idx].strftime('%b%d-%y')]
            for biaslist in modelbias: 
                result.append(biaslist[idx])
            result.extend([1,0,lc,0,hc,0])
            resultList.append(result)

        data2 = Utilities.Struct()
        data2.description = "Rolling Bias Stats %d Periods TotalRisk (%s)"%\
            (lookbackFrequency+1,selectedBmk)
        data2.reportSection = sectionName 
        biasName = ['%s Bias'%(rm.mnemonic[2:])for rm in self.modelSelectorList]
        data2.header =['Date']
        data2.header.extend(biasName)
        data2.header.extend(['Perfection','boolean','Upper Bound(95%)','boolean',
                             'Lower Bound(95%)','boolean'])
        data2.content = numpy.matrix(resultList)
        data2.reportName = "Rolling Bias Stats"
        data2.reportType = "LineChart"
        self.dataList.append(data2)
        logging.info("Processing Predicted Risk /Realised Risk Study")


    def compute_total_risk_portfolio(self, portfolio, expMatrix, factorCov, 
                                     srDict, scDict=None, factorTypes=None):
        """Compute the total risk of a given portfolio.  portfolio 
        should be a list of (asset,weight) values.
        factorTypes is an optional argument and should be a list of 
        ExposureMatrix.FactorType objects.  If specified, the total 
        common factor risk from those factors are returned instead.
        """
        expM_Map = dict([(expMatrix.getAssets()[i], i) \
                        for i in range(len(expMatrix.getAssets()))])
        # Discard any portfolio assets not covered in model
        indices = [i for (i,j) in enumerate(portfolio) \
                   if j[0] in expM_Map and j[0] in srDict]
        port = [portfolio[i] for i in indices]
        (assets, weights) = zip(*port)
        weightMatrix = numpy.zeros((len(assets),1))
        assetIdxMap = dict([(j,i) for (i,j) in enumerate(assets)])
#        print len(assets) 

        for (a,w) in portfolio:
            pos = assetIdxMap.get(a, None)
            if pos is None:
                continue
            weightMatrix[pos,0] = w
        # weights = numpy.array(weights)
        # weights /= numpy.sum(weights)

        expM = expMatrix.getMatrix().filled(0.0)
        expM = ma.take(expM, [expM_Map[a] for (a,w) in port], axis=1)
        if factorTypes is not None:
            f_indices = []
            for fType in factorTypes:
                fIdx = expMatrix.getFactorIndices(fType)
                f_indices.extend(fIdx)
            expM = ma.take(expM, f_indices, axis=0)
            factorCov = numpy.take(factorCov, f_indices, axis=0)
            factorCov = numpy.take(factorCov, f_indices, axis=1)

        # Compute total risk
        assetExp = numpy.dot(expM, weights)
        totalVar = numpy.dot(numpy.dot(numpy.transpose(assetExp),factorCov),assetExp)
        if factorTypes is None:
            totalVar += numpy.sum([(w * srDict[a])**2 for (a,w) in port])

#        print totalVar**0.5
        if scDict is None:
            return totalVar**0.5

        # Incorporate linked specific risk
        assetSet = set(assets)
        assetIdxMap = dict([(j,i) for (i,j) in enumerate(assets)])
        for sid0 in assets:
            if sid0 in scDict:
                for (sid1, cov) in scDict[sid0].items():
                    if sid1 not in assetSet:
                        continue
#                    print "LINKED SPECIFIC RISK CALCULATED !!!!!!!!"
                    weight0 = weightMatrix[assetIdxMap[sid0]] 
                    weight1 = weightMatrix[assetIdxMap[sid1]] 
                    totalVar += float(2.0 * weight0 * weight1 * cov)
#        print totalVar**0.5
        return totalVar**0.5

    def returnPortfolioCorrelations(self):
        logging.info("Processing Portfolio Correlations")
        sectionName = "PORTFOLIO CORRELATIONS"
        univ = self.fullsubID
        mdl2sub = dict((n.getModelID(),n) for n in univ)
        folios = list() 
        folioList = MODEL_INDEX_MAP.get(self.modelSelectorList[0].mnemonic[:4])
        if folioList is None:
            data = Utilities.Struct() 
            data.description = "No Benchmark Specified for %s"
            data.reportName = "ESTU & Benchmark Correlation" 
            data.reportSection = sectionName 
            self.dataList.append(data) 
            return

        imp = [(k,v) for k,v in self.modelMarketMap.items()]
        for f in folioList: 
            # print f 
            port = modelDB.getIndexConstituents(f,self.date,self.marketDB,rollBack=20,
                                                issueMapPairs=imp)
            if len(port)>0:
                (assets,weights) = zip(*[(mdl2sub[a],w)for(a,w)\
                                             in port if a in mdl2sub])
                weights = numpy.array(weights)
                weights /= numpy.sum(weights)
                folios.append(list(zip(assets,weights)))
            else: 
                folios.append([(univ[0],0.0)])

        resultList = [] 
        for idx,estu in enumerate(self.estUniv): 
            w = [self.assetCapDict.get(s) for s in estu]
            w /= numpy.sum(w) 
            folios.append(list(zip(estu,w)))
            (srDict,scDict) = self.modelSelectorList[0].loadSpecificRisks(self.rmiList[idx], modelDB)
            (vols,corr)= Utilities.compute_correlation_portfolios(
                self.expMList[idx],
                self.covMatrixList[idx],
                srDict,folios,scDict)
            vols = vols.filled(0.0)
            corr = corr.filled(0.0)
            result=['%s ESTU'%self.modelSelectorList[idx].mnemonic]
            for i in range(0,len(folios)-1):
                result.append('%.2f'%corr[len(folios)-1,i])
            resultList.append(result)
            del folios[-1]
            # print folios 

        data = Utilities.Struct()
        headerList= ["Benchmark"]
        headerList.extend(folioList)
        data.header = headerList 
        data.reportName = "ESTU & Benchmark Correlation" 
        data.reportSection = sectionName 
        data.content = numpy.matrix(resultList)
        data.reportType = "LineChart" 
        self.dataList.append(data) 
        logging.info("Finished Processing Portfolio Correlations")

    def prepBetaDiscrepancies(self,model,threshold, hist=False):
        pos =self.mnemonicPosMap[model.mnemonic[2:]]            
        univ = list(self.univ[pos])
        if not hasattr(model, 'modelHack'):
            model.modelHack = Utilities.Struct()
            model.modelHack.nonCheatHistoricBetas = True
        if not hist:
            betas = modelDB.getRMIPredictedBeta(self.rmiList[pos])
            p_betas = modelDB.getRMIPredictedBeta(self.p_rmiList[pos])
            betaType = 'predicted'
        else:
            betas = dict(); p_betas = dict()
            if model.modelHack.nonCheatHistoricBetas:
                betas = modelDB.getHistoricBeta(
                            self.date, univ)
                p_betas = modelDB.getHistoricBeta(
                            self.prevDate, univ)
            else:
                betas = modelDB.getHistoricBetaOld(
                            self.date, univ)
                p_betas = modelDB.getHistoricBetaOld(
                            self.prevDate, univ)
            betaType = 'historical'
        commonAssets = list(set(betas.keys()).intersection(univ).\
                            intersection(list(p_betas.keys())))
        diff = numpy.array([betas[sid] - p_betas[sid] \
                            for sid in commonAssets])
        bigChange = numpy.flatnonzero(ma.getmaskarray(
                            ma.masked_where(abs(diff) > threshold, diff)))
        assetList = [commonAssets[i] for i in bigChange]
        # if model.mnemonic=='AXEU21-MH-S':
        #     if ModelDB.SubIssue('D43LSATVW811') not in univ: 
        #         print "===================" 
        #     print betas[ModelDB.SubIssue('D43LSATVW811')]
        #     print p_betas[ModelDB.SubIssue('D43LSATVW811')]
        #     print assetList 
        #     print yeah 

        
        rank_by_cap = numpy.argsort(-numpy.array([self.assetCapDict.get(asset)for \
                                                      asset in assetList]))
        records = len(assetList)
        assetList = [assetList[i] for i in rank_by_cap] 
        assetList = assetList[:self.maxRecords]

        if not hist:
            self.headerMap['FACTOR']=self.decomposePredictedBeta(model,assetList, betas, p_betas)

        estuDict = dict((s,1) for s in assetList if s in self.wholeEstUniv)
        for asset in assetList: 
            if asset not in estuDict: 
                estuDict[asset] = '--'
        self.headerMap["ESTU"].update(estuDict)
        self.headerMap['BETA'] = dict([i,'%.2f'%betas[i]] for i in assetList)
        self.headerMap["PREV"] = dict([i,'%.2f'%p_betas[i]] for i in assetList)
        extraHeader = ["CAP (%s m)"%self.currency,"BETA","PREV","TR %","ESTU"]
        reportName = "%s Large %s Beta Change"%(model.mnemonic[2:],betaType)
        if not hist: 
            description = "%d Assets with %s %s beta change > %.2f"%(records,
                                                                     betaType,model.mnemonic[2:],threshold) 
        else: 
            description = "%d Assets with %s beta change > %.2f"%(records,betaType,threshold) 

        if len(assetList)>0:
            if not hist: 
                if self.headerMap['FACTOR'] is not None: 
                    extraHeader.append("FACTOR") 
            else: 
                (dateList,assetPeriodReturns) = \
                    self.prepRMGHistoricBetasWindow(model, self.date,assetList)
                extraHeader.extend(["LEAVING WEEK","ENTERING WEEK",
                                    "LEAVING PERRT%","ENTERING PERRT%", "UCP"])
                priceMatrix = modelDB.loadUCPHistory([self.date],assetList,model.numeraire.currency_id)
                UCPDict = dict() 
                for idx,sid in enumerate(assetList): 
                    ret = priceMatrix.data[idx,0]
                    if math.isnan(ret):
                        ret = 0.0
                    UCPDict[sid] = '%.2f'%ret
                self.headerMap["UCP"] = UCPDict 
                resultList = [] 
                leavingDict=dict();enteringDict=dict()
                leavingReturnDict=dict();enteringReturnDict=dict()
                for asset in assetList: 
                    if asset not in dateList.keys(): 
                        enteringReturnDict[asset]= '--'
                        leavingReturnDict[asset]= '--'
                    else: 
                        enteringPeriod =list(set(dateList[asset][0]).difference(set(dateList[asset][1])))
                        if len(enteringPeriod)!=0 : 
                            enteringPeriod = [max(enteringPeriod)]
                        leavingPeriod = [i+datetime.timedelta(7) for i in set(dateList[asset][1]).difference(set(dateList[asset][0]))]
                        if len(leavingPeriod)!=0:
                            leavingPeriod = [min(leavingPeriod)]
                        enteringDict[asset] = ",".join([str(i) for i in enteringPeriod])
                        leavingDict[asset] = ",".join([str(i) for i in leavingPeriod])
                        entRet= float(assetPeriodReturns[asset][0][-1]*100)
                        levRet = float(assetPeriodReturns[asset][1][0]*100)
                        if math.isnan(entRet):
                            entRet = 0.0
                        if math.isnan(levRet):
                            levRet=0.0
                    
                        enteringReturnDict[asset]= '%.2f'%(entRet)
                        leavingReturnDict[asset]= '%.2f'%(levRet)
                    # entersort = numpy.argsort(numpy.array(enteringPeriod))
                    # leavesort = numpy.argsort(numpy.array(leavingPeriod))
                    # enteringPeriod = [enteringPeriod[i] for i in entersort] 
                    # leavingPeriod = [leavingPeriod[i] for i in leavesort] 
                    # leavingDict[asset] =  ",".join([str(i) for i in leavingPeriod])
                    # enteringDict[asset] =  ",".join([str(i) for i in enteringPeriod])
                    # print ",".join([str(round(100*i,2)) for i in list(assetPeriodReturns[asset][1][:(len(leavingPeriod)-1)])])
                    # leavingReturnDict[asset] = ",".join([str(round(100*i,2)) for i in list(assetPeriodReturns[asset][1][:(len(leavingPeriod)-1)])])
                    # enteringReturnDict[asset] = ",".join([str(round(100*i,2)) for i in list(assetPeriodReturns[asset][0][-len(enteringPeriod):])])
                self.headerMap["LEAVING WEEK"]=leavingDict
                self.headerMap["ENTERING WEEK"]=enteringDict
                self.headerMap["LEAVING PERRT%"]=leavingReturnDict
                self.headerMap["ENTERING PERRT%"]=enteringReturnDict
            data = self.prepAssetLevelInfo(assetList,extraHeader)
            data.reportName = reportName 
            data.description = description 
        else: 
            data = Utilities.Struct() 
            data.reportName = reportName 
            data.description = description 
        return data 

    def prepRMGHistoricBetasWindow(self,model,date,assetList):
        result = list() 
        minDate = date - datetime.timedelta(50)
        sidStrings = [sid.getSubIDString() for sid in assetList]
        sidArgList = [('sid%d' %i) for i in range(len(assetList))]
        query = """SELECT value,dt,sub_issue_id FROM rmg_historic_beta_v3 
                   WHERE sub_issue_id IN (%s) AND home=1
                   AND dt >= :min_d and dt<= :max_d"""%(','.join([(':%s'%i) for i in sidArgList]))
        valueDict = dict(zip(sidArgList,sidStrings))
        valueDict['min_d'] = minDate
        valueDict['max_d'] = self.date 
        self.modelDB.dbCursor.execute(query,valueDict)
        sidDateDict= dict() 
        sidBetaChangeDate = dict() 
        result = self.modelDB.dbCursor.fetchall() 
        for r in result: 
            sidDateDict.setdefault(ModelDB.SubIssue(r[2]),list()).append((r[1],r[0]))
        for asset in sidDateDict.keys(): 
            for idx,sidDate in enumerate(sidDateDict[asset]): 
                if idx == 0: 
                    sidBetaChangeDate[asset] = [sidDateDict[asset][0][0]]
                else: 
                    if sidDateDict[asset][idx][1] != sidDateDict[asset][idx-1][1]: 
                        sidBetaChangeDate[asset].append(sidDateDict[asset][idx][0])
            if len(sidBetaChangeDate[asset])< 2: 
                logging.info("Not Enough History for asset %s"%asset)
                del (sidBetaChangeDate[asset])         
        lastUpdateDate = dict() 
        rmgSidDict = dict() 
        assetWindowDict = dict() 
        isoRMGMap = dict([(rmg.mnemonic, rmg) for rmg in modelDB.getAllRiskModelGroups()])
        for asset in sidBetaChangeDate.keys(): 
            lastUpdateDate[asset] = sidBetaChangeDate[asset][-2].date()
            modelID = ModelID.ModelID(string = asset.getSubIDString()[:-2])
            processed = False 
            tradingCtryIso = self.assetRmgMap[modelID]
            tradingRmg = isoRMGMap[tradingCtryIso]
            hcIso = self.hcMap.get(modelID)
            hcRmg = isoRMGMap.get(hcIso)
            if hcRmg is None : 
                print(hcIso,modelID)
                hc = tradingRmg
            elif hcRmg in model.rmg:
                hc = hcRmg
                # if tradingCtryIso == 'CN' and hcIso == 'CN':
                #     #Special treatment for China-A
                #     hc = Utilities.Struct()
                #     hc.mnemonic = 'XC'
                #     hc.rmg_id = -2
            else:
                hc = tradingRmg
            if hc is not None:
                rmgSidDict.setdefault(hc, list()).append(asset)
                processed = True
            if not processed:
                #If we cannot find neither the hc or tradingCountry, map it to None
                result[asset] = None
        dateListDict = dict();assetPeriodReturnsDict = dict()

        for rmg in rmgSidDict.keys():
            (assetPeriodReturns,tradingDaysList) = self.prepRMGHistoricBetas(rmg,self.date,
                                                                             rmgSidDict[rmg])
            if tradingDaysList==None:
                continue 
            else:
                lastDate = max(lastUpdateDate[r] for r in rmgSidDict[rmg])
                (p_assetPeriodReturns,p_tradingDaysList) = self.prepRMGHistoricBetas(rmg,
                                                    lastDate,rmgSidDict[rmg])
                for asset in rmgSidDict[rmg]:
                    dateListDict[asset] = (tradingDaysList,p_tradingDaysList) 
                    assetPeriodReturnsDict[asset] = (assetPeriodReturns[asset],
                                                     p_assetPeriodReturns[asset])

        return (dateListDict,assetPeriodReturnsDict) 


    def prepRMGHistoricBetas(self, rmg, currDate,rmgHomeSubIssues):
        if currDate.isoweekday() > 5:
            logging.info('Saturday/Sunday, skipping.')
            return None

        self.modelDB.setCumReturnCache(500)
        # Get list of dates needed
        daysNeeded = 500
        tradingDaysList = self.modelDB.getDates([rmg], currDate, daysNeeded,True)
        if len(tradingDaysList) > 0 and tradingDaysList[-1] != currDate:
            tradingDaysList.append(currDate)
        if len(tradingDaysList) < int(daysNeeded / 2):
            print(rmg, rmgHomeSubIssues)
            logging.info('Insufficient history. (got %d, need at least'
                          ' %d days) Skipping.'%( len(tradingDaysList), int(daysNeeded / 2)))
            return (None,None)
        
        # Only recompute betas at the start of each new period
        # In this case, on Mondays (or first trading day of the week)
        prev_betas_new = self.modelDB.getPreviousHistoricBeta(currDate, rmgHomeSubIssues)

        prevDate = tradingDaysList[-2]
        if prevDate.weekday() > currDate.weekday()\
                or not prev_betas_new or \
                currDate - prevDate > datetime.timedelta(7):
            period_dates = [
                prev for (prev, nxt) in zip(tradingDaysList[:-1],
                                             tradingDaysList[1:])
                if nxt.weekday() < prev.weekday()]
            # Compute asset period returns
            r = self.modelDB.loadCumulativeReturnsHistory(
                period_dates, rmgHomeSubIssues)
            assetPeriodReturns = r.data[:,1:] / r.data[:,:-1] - 1.0

            assetPeriodReturnsDict = dict() 
            for idx,asset in enumerate(rmgHomeSubIssues): 
                assetPeriodReturnsDict[asset] = assetPeriodReturns[idx,:]
        else: 
            assetPeriodReturnsDict = dict() 
            for idx,asset in enumerate(rmgHomeSubIssues):
                assetPeriodReturnsDict[asset] = numpy.matrix([0.0])
            period_dates = [currDate]
        return (assetPeriodReturnsDict,period_dates)
        
            

    def decomposePredictedBeta(self, model,sidList, betas, p_betas):
        modelDB = self.modelDB
        pos =self.mnemonicPosMap[model.mnemonic[2:]]            
        result = dict()
        rmgSidDict = dict()
        isoRMGMap = dict([(rmg.mnemonic, rmg) for rmg in modelDB.getAllRiskModelGroups()])
        mdIDList = [sid.getModelID() for sid in sidList]
        hc2Map = AssetProcessor.get_asset_info(self.date,mdIDList, modelDB, marketDB,
                                                   'REGIONS', 'HomeCountry2')
        for sid in sidList:
            modelID = ModelID.ModelID(string = sid.getSubIDString()[:-2])
            processed = False
            tradingCtryIso = self.assetRmgMap[modelID]
            tradingRmg = isoRMGMap[tradingCtryIso]
            hc = tradingRmg
            hcIso = self.hcMap[modelID] 
            hcRmg = isoRMGMap.get(hcIso)
            if hcRmg is not None : 
                hc2Iso =hc2Map.get(modelID)
                hc2Rmg = isoRMGMap.get(hc2Iso)
                if hcRmg in model.rmg:
                    hc = hcRmg
                    if tradingCtryIso == 'CN' and hcIso == 'CN':
                        #Special treatment for China-A
                        hc = Utilities.Struct()
                        hc.mnemonic = 'XC'
                        hc.rmg_id = -2
                elif hc2Rmg is not None: 
                    if hc2Rmg in model.rmg: 
                        hc = hc2Rmg
                        if tradingCtryIso == 'CN' and hcIso == 'CN':
                            #Special treatment for China-A
                            hc = Utilities.Struct()
                            hc.mnemonic = 'XC'
                            hc.rmg_id = -2
                if hc is not None:
                    rmgSidDict.setdefault(hc, list()).append(sid)
                    processed = True
            if not processed:
                #If we cannot find neither the hc or tradingCountry, map it to None
                result[sid] = None
        #Now pull the rmg_market_portfolio and do the beta decomposition
        for (rmg, rmgSids) in rmgSidDict.items():
            marketPort = modelDB.getRMGMarketPortfolio(rmg, self.date)
            (srDict,scDict) = model.loadSpecificRisks(self.rmiList[pos], modelDB)
            decom = self.getDecomposition(rmgSids, 
                                          self.expMList[pos],
                                          self.covMatrixList[pos],
                                          srDict,
                                          marketPort, rmg)
            p_marketPort = modelDB.getRMGMarketPortfolio(rmg, self.prevDate)
            (p_srDict,p_scDict) = model.loadSpecificRisks(self.p_rmiList[pos], modelDB)
            p_decom = self.getDecomposition(rmgSids, 
                                            self.prevExpMList[pos],
                                            self.p_covMatrixList[pos],
                                            p_srDict,
                                            p_marketPort, rmg)

            if decom.shape[1] != p_decom.shape[1]:
                return None 
            p_factors = self.prevExpMList[pos].getFactorNames()
            factors = self.expMList[pos].getFactorNames()
            newfactors = list()
            for idx,factor in enumerate (factors):
                if factor not in p_factors: 
                    newfactors.append(idx)
                else: 
                    continue
            decom = numpy.delete(decom,newfactors,1)
            delta = decom - p_decom
            for sIdx, sid in enumerate(rmgSids):
                #Make sure we are decomposing correctly
                try:
                    assert(abs(betas[sid] - numpy.sum(decom[sIdx, :])) < 0.001)
                    assert(abs(p_betas[sid] - numpy.sum(p_decom[sIdx, :])) < 0.001)
                except:
                    logging.error('Database PBeta: %.4f, Decomposition PBeta: %.4f'%\
                                      (betas[sid], numpy.sum(decom[sIdx, :])))
                    logging.error('Database Previous PBeta: %.4f,Decomposition Previous PBeta: %.4f'\
                                      %(p_betas[sid], numpy.sum(p_decom[sIdx, :])))

                diff = numpy.array([abs(k) for k in delta[sIdx, :]])
                idxDict = dict([(k, idx) for (idx, k) in enumerate(diff)])
                rpos = idxDict[numpy.amax(diff)]
                if rpos == len(self.prevExpMList[pos].getFactorNames() ):
                    fName = "Specific Risk"
                else:
                    fName = self.prevExpMList[pos].getFactorNames()[rpos]
                result[sid] = fName
        return result

    def getDecomposition(self, assets, expMatrix, factorCov,
                         srDict, marketPortfolio, rmg, forceRun=True):
        """Return a factor decomposition of the predicted beta. Sum of each row will be 
        the predicted beta of each asset"""

        # Make sure market portfolio is covered by model
        exposureAssets = expMatrix.getAssets()
        assetIdxMap = dict([(j,i) for (i,j) in enumerate(exposureAssets)])
        marketPortfolio = [(a,w) for (a,w) in marketPortfolio if a in assetIdxMap]
        if len(marketPortfolio) == 0 and forceRun:
            logging.warning('Empty market portfolio')
            return [0] * len(assets)

        mktIndices, weights = zip(*[(assetIdxMap[a], w) for (a,w) in marketPortfolio])
        market_ids = [exposureAssets[i] for i in mktIndices]
        market_id_map = dict([(exposureAssets[j],i) for (i,j) in enumerate(mktIndices)])
        # Compute market portfolio specific variance
        univ_sr = numpy.array([srDict[asset] for asset in market_ids])
        univ_sv = univ_sr * univ_sr
        market_sv = numpy.inner(weights, univ_sv * weights)
        
        # Compute market portfolio common factor variances
        expM_Idx = [assetIdxMap[a] for a in market_ids]
        market_exp = numpy.dot(ma.take(expMatrix.getMatrix(), expM_Idx, axis=1).filled(0.0), weights)
        market_cv_exp = numpy.dot(factorCov, market_exp)
        market_var = numpy.inner(market_exp, market_cv_exp) + market_sv
        logging.info('Market risk for %s is %.2f%%'% (rmg.mnemonic,
                                                      100.0 * math.sqrt(market_var)))
                                                      
        # Compute asset predicted betas
        beta = ma.zeros((len(assets), expMatrix.getMatrix().shape[0] + 1), float)
        for i in range(len(assets)):
            asset = assets[i]
            if asset.isCashAsset():
                continue
            else:
                idx = assetIdxMap[asset]
                # Compute asset factor covariance with market
                fv = (expMatrix.getMatrix()[:,idx].filled(0.0) * market_cv_exp)/ market_var
                sv = 0.0
                # Add specific component
                if asset in market_id_map:
                    sv = (weights[market_id_map[asset]] * srDict[asset] * srDict[asset])/ market_var
            fIdx = range(expMatrix.getMatrix().shape[0])
            ma.put(beta[i, :], fIdx, fv)
            ma.put(beta[i, :], -1, sv)
        logging.debug('getDecomposition: end')
        return beta

            

class ReportTable(): 
    """Class to Store result of each report section using table.
    User could  set report section,header and row data. 
    Extra Columns could easily be added. """

    def __init__(self,reportName,reportType='Table'):
        self.reportName = reportName
        self.header = None
        self.data=None
        self.description=None 
        self.dataCounter = 0 
        self.reportLength = 0 
        self.reportType = reportType 
    def setHeader(self,header):
        """Set the header to the report table class.
        header should be a list of string indicating the header of the table.
        Once received header,the code will also instantiate a matrix object
        with the same length of columns as the header's"""
#        assert(len(header)>0)
        self.header = header
        self.headerIdxMap=dict([(j,i) for (i,j) in enumerate(header)])
        self.data = ma.masked_all([0,len(header)],object)

    def setReportLength(self, reportLength):
        """set the Table row Number"""
        self.reportLength= reportLength
        assert(self.header is not None)
        self.data = ma.masked_all([reportLength, len(self.header)], object)
    
    def setDescription(self,description): 
        self.description = description 

    def addRowData(self,rowInput): 
        """Add data into the table by row.
        The rowInput should be a list of data and correspond to the table header
        Not necessarily the same but correspondingly in order
        """
        try:
            assert(len(rowInput) == self.data.shape[1])
        except:
            logging.error("rowInput length: %s, table header length: %s"%(\
                    len(rowInput), self.data.shape[1]))
            raise Exception
        assert(self.data is not None)
        self.data[self.dataCounter,:] = rowInput
        self.dataCounter += 1

    def addColumnData(self, headerName, columnInput):
        """Add data into the table by column.
        The columnName should be the column user want to put the data under.
        ColumnInput should be a list of data correpond to each row of the table
        """
        # try:
        #     assert(len(coulumnInput) == self.data.shape[0])
        # except:
        #     logging.error("columnInput length: %s, table row length: %s"%(\
        #             len(columnInput), self.data.shape[0]))
        #     raise Exception
        if (len(columnInput)==self.data.shape[0]):
            hIdx = self.headerIdxMap.get(headerName)
            self.data[:, hIdx] = columnInput
        else: 
            logging.error("columnInput length: %s, table row length: %s"%(\
                     len(columnInput), self.data.shape[0]))
            

    def getColumnData(self, headerName):
        """Given a Column Name get All the Data"""
        assert(headerName in set(self.header))
        hIdx = self.headerIdxMap.get(headerName)
        return self.data[:, hIdx]
    
    def getReportName(self):
        return self.reportName 

    def getColumnIndex(self, columnName):
        return self.headerIdxMap.get(columnName)

    def getRowIndex(self, referenceValue, referenceColumnName):
        cIdx = self.getColumnIndex(referenceColumnName)
        for vIdx, v in enumerate(self.getMatrix()[:, cIdx]):
            if v == referenceValue:
                return vIdx
        return None

    def setDataElement(self, rowIdx, headerName, data):
        assert(headerName in self.header)
        hIdx = self.headerIdxMap[headerName]
        self.data[rowIdx, hIdx] = data

    def getHeader(self):
        return self.header
    
    def getReport(self):
        return self.data
    
    def getReportLength(self):
        return self.reportLength

    def getHeaderIdxMap(self):
        return self.headerIdxMap

    def getRowData(self, rowIndex):
        assert(rowIndex < self.data.shape[0])
        return self.data[rowIndex, :]

def getDatesFromCalendar(model,mode,date): 
    modelDateFile = open (MODEL_PATH + model + '-calendar.att','r')
    lines = modelDateFile.readlines() 
    modelDateFile.close() 
    realDates = list() 
    for idx,line in enumerate(lines): 
        if line[0] == '#': 
            continue 
        else: 
            dt,me,es = line.split('|')
            if dt > date: 
                break 
            elif es == '*\n': 
                lastModelDate = lines[idx-1].split('|')[0]
                break
            else: 
                if mode == 'daily':
                    realDates.append(dt) 
                if mode == 'monthly': 
                    if me == '*': 
                        realDates.append(dt) 
    # if mode == 'monthly':
    #     realDates.append(lastModelDate)

    dateList = [Utilities.parseISODate(dt) for dt in realDates]
    return dateList

if __name__ == '__main__':
    usage = "usage: %prog [options] <YYYY-MM-DD> [<YYYY-MM-DD>]"
    cmdlineParser = optparse.OptionParser(usage=usage)
    cmdlineParser.add_option('--email', action='store',
                             default=None, dest='emailRecipients',
                             help='email report')
    cmdlineParser.add_option('--store-report', action='store_true',
                             default=False, dest='storeReport',
                             help='write the report to a text file')
    cmdlineParser.add_option("--report-file", action="store",
                             default=None, dest="reportFile",
                             help="report file name")
    cmdlineParser.add_option("--prior-model", action="store",
                             default=None, dest="priorModelName",
                             help="model name for previous day")
    cmdlineParser.add_option("--max-records", action="store",
                             default=50, dest="maxRecords",
                             help="max records per report section. If set to be -1, all records displayed")
    cmdlineParser.add_option("--modelPath", action="store",
                             default='/axioma/products/current/riskmodels/2.1/FlatFiles/', 
                             dest="modelPath",
                             help="file path where models are extracted")

    Utilities.addModelAndDefaultCommandLine(cmdlineParser)
    (options, args) = cmdlineParser.parse_args()
    fileName=options.reportFile    
    if options.emailRecipients is not None:
        sendTo = options.emailRecipients.split(',')
    else:
        sendTo=[]
    if len(args) < 1 or len(args) > 2:
        cmdlineParser.error("Incorrect number of arguments")
    # riskModel_Class = Utilities.processModelAndDefaultCommandLine(
    #     options, cmdlineParser)

    Utilities.processDefaultCommandLine(options, cmdlineParser)
    riskModelClassList = []
    if options.modelName != None: 
        modelNameList = options.modelName.split(',')
        if len(modelNameList):
            if modelNameList[0] in ['COAxioma2013MH','Univ10AxiomaMH']:
                modelNameList = [modelNameList[0]]

            elif modelNameList[0] == 'EMAxioma2011MH_S':
                modelName = modelNameList[0][:-4]
                modelNameList = [modelName+'MH_Pre2003',modelName+'SH_Pre2003',
                                 modelName+'MH_S',modelName+'SH_S']
            else:
                modelName = modelNameList[0][:-2]
                modelNameList=[modelName+'MH',modelName+'SH',modelName+'MH_S',
                               modelName+'SH_S']
        for modelName in modelNameList: 
            riskModelClass = riskmodels.getModelByName(modelName)

            try: 
                riskModelClass = riskmodels.getModelByName(modelName)
                riskModelClassList.append(riskModelClass) 
            except KeyError:
                print('Unknown risk model "%s"' % options.modelName)
                names = sorted(modelNameMap.keys())
                print('Possible values:', ', '.join(names))
                sys.exit(1)

    elif options.modelID != None and options.modelRev != None:
        rm_idList = options.modelID.split(',')
        revList = options.modelRev.split(',')
        if len(rm_idList) != len(revList):
            print(' Risk Model ID and Rev length does not match')
            sys.exit(1) 

        for idx, rm_id in enumerate(rm_idList):
            rm_id = int(rm_id)
            rev = int(revList[idx]) 
            try:
                riskModelClass = riskmodels.getModelByVersion(rm_id, rev)
                riskModelClassList.append(riskModelClass) 

            except KeyError:
                print('Unknown risk model %d/%d' % (rm_id, rev))
                revs = list(modelRevMap.keys())
                revs = sorted('/'.join((str(i[0]), str(i[1]))) for i in revs)
                print('Possible values:', ', '.join(revs))
                sys.exit(1)
            
    priorRiskModelClassList = []  
    if options.priorModelName is not None:
        modelNameList = options.priorModelName.split(',') 
        if len(modelNameList) != len(riskModelClassList): 
            logging.info("Number of Model do not match Number of priorModel")
            sys.exit(1)
        else: 
            try: 
                for modelName in modelNameList:                     
                    priorRiskModel_Class = riskmodels.getModelByName(modelName) 
                    priorRiskModelClassList.append(priorRiskModel_Class)
            except KeyError:
                print('Unknown risk model "%s"' % modelName)
                names = sorted(modelNameMap.keys())
                print('Possible values:', ', '.join(names))
                sys.exit(1)
    else:
        priorRiskModelClassList = riskModelClassList

    modelDB = ModelDB.ModelDB(sid=options.modelDBSID, 
                              user=options.modelDBUser, 
                              passwd=options.modelDBPasswd)
    marketDB = MarketDB.MarketDB(sid=options.marketDBSID, 
                                 user=options.marketDBUser, 
                                 passwd=options.marketDBPasswd)

    fullRiskModel_List = []
    priorRiskModel_List= []
    for riskModel_Class in riskModelClassList: 
        fullRiskModel_List.append(riskModel_Class(modelDB, marketDB))
    for priorRiskModel_Class in priorRiskModelClassList: 
        priorRiskModel_List.append(priorRiskModel_Class(modelDB, marketDB))

    # Only one type of reports could be generated together
    if isinstance(fullRiskModel_List[0], PhoenixModels.CommodityFutureRegionalModel):
        modelAssetType = 'Future'
    elif isinstance(fullRiskModel_List[0],FixedIncomeModels.Univ10AxiomaMH):
        modelAssetType = 'Universal'
    else:
        modelAssetType = 'Equity'

    startDate = Utilities.parseISODate(args[0])
    if len(args) == 1:
        endDate = startDate
    else:
        endDate = Utilities.parseISODate(args[1])

    dates = modelDB.getDateRange(fullRiskModel_List[0].rmg, startDate, endDate, True)
    modelDB.totalReturnCache = None
    modelDB.specReturnCache = None
    status = 0 
    for date in dates:
        try:
            logging.info('Processing %s' % date)
            riskModel_List = []
            if modelAssetType in ['Future','Universal']: 
                fullRiskModel_List =[fullRiskModel_List[0]]
            for rm in fullRiskModel_List:
                #check whether the model has run for that date
                rmi = modelDB.getRiskModelInstance(rm.rms_id, date)
                if rmi is None:
                    logging.error('No risk model instance for %04d-%02d-%02d!' % \
                                      (date.year, date.month, date.day))
                    continue
                if modelAssetType !='Universal' and \
                        (rmi.has_exposures * rmi.has_returns * rmi.has_risks)==0:
                    logging.error('No risk model instance for %04d-%02d-%02d!' % \
                                      (date.year, date.month, date.day))
                    continue
                riskModel_List.append(rm)
            status = 0
            if modelAssetType == 'Equity':
                name = riskModel_List[0].name
                modelName = name[:name.index('a')+1]
                modelmnemonicList = [model.mnemonic for model in riskModel_List]
            ### hard code for now.### 
            # if options.modelName =='USAxioma2016MH' : 
            #     options.modelPath="/axioma/products/US4-final/FlatFiles/"                
            checker_ = RiskModelValidator(riskModel_List, priorRiskModel_List, date,
                                          modelDB, marketDB, modelAssetType,int(options.maxRecords),options.modelPath)

            if modelAssetType == "Universal":
                checker_.returnModelStructure() 
#                checker_.checkAgainstSubModels()
                checker_.returnFactorReturnsChecks()
                checker_.returnFactorVolChecks() 
                checker_.returnFactorCorrChecks() 
            else: 
                checker_.returnModelStructure() 
                checker_.returnModelCoverage()                
                if modelAssetType == "Equity":
                    checker_.returnExchangeInformation()
                if modelAssetType == "Equity":
                    checker_.returnModelESTUDetails() 
                    if options.modelName == 'USAxioma2016MH':
                        checker_.returnDescriptorCoverage()
                if modelAssetType == "Equity":
                    try: 
                        checker_.returnFactorAutoCorrChecks()
                        if '22' not in modelmnemonicList[0]:
                            checker_.returnMonthlyFactorAutoCorrChecks()
                    except Exception as e:
                        traceback.print_exc()
                        continue
                checker_.returnDodgySpecificReturns()
                checker_.returnLargeExposureChanges()
                checker_.returnCovarianceMatrixSpecs()
                checker_.returnFactorReturnsChecks()
                checker_.returnFactorVolChecks() 
                checker_.returnFactorCorrChecks() 
                if modelAssetType == "Equity":
                    checker_.returnRegressionStatistics()
                    checker_.returnBetaDiscrepencies()
                    checker_.returnPortfolioCorrelations()
                    if '22' not in modelmnemonicList[0] and options.modelName not in ('WWAxioma2017MH'):
                        from riskmodels import pythonAPIUtiles

#                    if '22' not in modelmnemonicList[0] and options.modelName !='USAxioma2016MH':
                        checker_.returnPredictedRealisedRisk()
                else: 
                    checker_.returnAssetRiskChange()
                    checker_.returnAssetRiskChange(threshold = 0.02,total=True)

            filename=options.reportFile
            if filename == None:
                filename = '%s.report.%04d%02d%02d' % (modelName, date.year, date.month, date.day)
            i = 0
            path = os.path.dirname(filename)
            if not os.path.exists(path) and path != '':
                os.makedirs(path)
            if os.path.isfile(filename + ".v" + str(i)):
                logging.info(filename  + ".v" + str(i) + " already exists")
                i = 1
                while os.path.isfile(filename + ".v" + str(i)):
                    logging.info(filename + ".v" + str(i) + " already exists")
                    i = i + 1

            filename = filename + ".v" + str(i)
            finalReport = ModelReport(filename,modelName,date,checker_.dataList,checker_.coreDataList)
            finalReport.populateReports() 

            if options.storeReport:
                finalReport.writeReport() 
                logging.info("Finished writing report to " + filename)

            if len(sendTo) > 0:
                logging.info("Sending Report")
                finalReport.emailReport(date, sendTo)
                logging.info("Finished Sending Report")
        except Exception as ex:
            logging.error('Exception caught during processing', exc_info=True)
            modelDB.revertChanges()
            status = 1
    logging.info("Finalizing MarketDB")
    marketDB.finalize()
    logging.info("Finalizing modelDB")
    modelDB.finalize()
    logging.info("Exitingwith status %d"%status)
    sys.exit(status)
