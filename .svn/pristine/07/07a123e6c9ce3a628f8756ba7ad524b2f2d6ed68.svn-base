import logging
import numpy
import numpy.ma as ma
import time 

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

    def setPosition(self,position): 
        self.position = position
    
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

class visualizer: 
    """ Module to visualize reports from a list of content Structs """ 

    def __init__(self,outputHtml,reportContent,
                 date,reportHeader,sectionNames=list(),
                 reportContentCore = list(),displayThought=True): 
        self.htmlFile = outputHtml 
        self.reportContent = reportContent 
        self.reportContentCore = reportContentCore
        self.reportHeader = reportHeader 
        self.date = date 
        self.displayThought = displayThought
        self.necessaryFields = set(["reportName","reportSection","header","content"])
        self.sectionNames = sectionNames
        self.reportTables = dict() 

    def isNumeric(self, s):
        try:
            float(s)
            return True
        except ValueError:
            return False
        except TypeError:
            print(s) 
            return False 

    def createTable(self, data): 
        dataFields = set(data.getFieldNames())
        missingNecessaries = self.necessaryFields.difference(dataFields) 
        if "reportType" not in dataFields: 
            data.reportType = "Table" 
        if 'position' not in dataFields: 
            data.position = None
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
            table.setPosition(data.position)
            if "populateByRow" not in dataFields or dataFields.populateByRow == True: 
#                print '------------------------'
#                print data.reportName
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

    def addTable(self,sectionName,table): 
        self.reportTables.setdefault(sectionName,list()).append(table)
        if sectionName not in self.sectionNames: 
            self.sectionNames.append(sectionName) 

    def displayDailyThought(self):
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
        modelDate = self.date
        i = int(time.mktime(modelDate.timetuple()) / (3600*24)) % len(blurbCollection)
        return blurbCollection[i]

    def createSectionName(self,sectionName):
        div = """\r\n <div class = "sectionName">%s</div>"""%sectionName 
#        if sectionName == 'EXPERIMENTAL SECTIONS': 
        if sectionName == 'FACTOR RISKS & COVARIANCES':
            div += """\r\n<script>
google.charts.load('current', {'packages':['line']});
</script>"""
        return div 

    def createDiv(self,table,height="null",width="null",JSON="null" ): 
        if table.data is  None: 
            if table.description is not None:
                if "Universes are not Identical" in table.description:
                    div = """\r\n <div class="descriptionDanger"> %s </div>"""%(table.description)
                elif ("missing technical descriptors" in table.description) and (
                    "No" not in table.description):
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

            
            if table.position is not None: 
                JSON ={'mode':'multiple'}
#                print JSON
            if table.reportType!='LineChart_material':
                div = """<div class="%s" id="%s">%s</div>
<div class="separator"></div> 
<div class="table" id = "%s" align="center">\r\n<div><script type="text/javascript">
google.setOnLoadCallback(function(){
drawVisualization(%s,"%s","%s",%s,%s,%s);})</script>\r\n</div></div>\r\n"""\
                    %(desc,table.getReportName()+" Description",table.description if table.description!=None else "",table.getReportName(),
                      #                  "center" if table.reportType!="LineChart" else "",
                      dataStr,table.reportType,table.getReportName(),
                      height,width,JSON if JSON!="null" else "null")
#                print div 
            else: 
                div = """<div class="%s" id="%s">%s</div>
<div class="separator"></div> 
<div class="table" id = "%s" align="center">\r\n<div>
<script src="http://ajax.cdnjs.com/ajax/libs/json2/20110223/json2.js"></script>
<script type="text/javascript">
google.setOnLoadCallback(function(){ 
drawVisualization(%s,"%s","%s",%s,%s,%s);})</script>\r\n</div></div>\r\n"""\
                    %(desc,table.getReportName()+" Description",table.description if table.description!=None else "",table.getReportName(),
                      #                  "center" if table.reportType!="LineChart" else "",
                      dataStr,table.reportType,table.getReportName(),
                      height,width,JSON if JSON!="null" else "null")
              
            if table.position is not None:  
                pos = """\r\n<div class="table%s">\r\n"""%table.position
                div = pos +div +"</div>"
                div = div.replace("center","")
        return div 

    def visualize(self):
        logging.info("Visualizing Report to %s"%self.htmlFile)
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
<link type="text/css" rel="stylesheet" href ="http://intranet/internet/test/htmlAccessories/style.css?">
<script type="text/javascript" src="http://intranet/internet/test/htmlAccessories/function.js"></script>
"""
        tagLine = """ <div id="tagLine"><i class="fa fa-bar-chart"></i>  Flexible is Better.Your ideas and stories</div>"""

        body = "" 
        if self.displayThought: 
            body += """
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
</div>"""%self.displayDailyThought()
        body += """
<div id="reportName"><i class="fa fa-file-text-o"></i> %s  (%s)</div>"""%(self.reportHeader,"%04d-%02d-%02d"%(self.date.year,
                                                                                                           self.date.month,
                                                                                                           self.date.day))

        divs = header + body + tagLine 

        for idx,reportContent in enumerate([self.reportContentCore,self.reportContent]):
#            print reportContent 
            if len(self.reportContentCore) !=0 and len(self.reportContent)!=0: 
                if len(reportContent)!=0 : 
                    umbrella = 'Core Model QA' if idx==0 else 'Daily Model Report'
                    divs = divs+ """<div class = "umbrella">%s</div>"""%umbrella
            for passedData in reportContent: 
                self.createTable(passedData)
            
            for section in self.sectionNames:
                if self.reportTables.get(section) is not None: 
                    divs = divs + self.createSectionName(section)
                    for table in self.reportTables[section]:
                        if table.getReportName() == "Country Thickness Table": 
                           JSONObject = """ { "lb":0,\r\n"hb":9,\r\n"idx":1}"""
                           divs = divs + self.createDiv(table,JSON = JSONObject)
                        else: 
                            divs = divs + self.createDiv(table)

                        logging.info("Created Div for %s"%table.getReportName())
            self.sectionNames = list() 
            self.reportTables = dict() 

        content = divs + """\r\n<footer class="clear">
<p> Axioma Inc </p>\r\n</footer>\r\n """
        html = open(self.htmlFile,'w') 
        html.write(content) 
        html.close() 
        logging.info("Visualiation Completed.Report written to %s"%self.htmlFile)

    



