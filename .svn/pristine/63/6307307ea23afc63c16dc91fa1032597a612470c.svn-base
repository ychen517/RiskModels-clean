
import subprocess
import sys
import os
import csv
import logging
import optparse
from marketdb import MarketDB
from marketdb.ConfigManager import findConfig
from riskmodels import Utilities
from riskmodels import ModelDB


fields=['AXIOMAID','ISSUEID','SUBISSUEID','FROMDT','THRUDT','SECTIONS','PRODUCTION','MARKETDB','COMMIT']

UNMODEL_SECTIONS=['UNModelReturn',]
UNMARKET_SECTIONS=['FILevel','FIReturn',]
fieldsDict={}

def getSectionsList(fileName):
    sectionsList=[]
    with open(fileName, 'r') as infile:
        data = infile.read()
        lines = data.splitlines()
        for line in lines:
            if line.startswith("["):# let '[DEFAULT]' and such be there also
                line=line.strip()
                sectionsList.append(line[1:-1])
    return sectionsList

def execSQL(cursor, filename):
    """ blindly run SQL from a file - one in each line"""

    for sql in [l.strip() for l in open(filename).readlines()]:
       if len(sql) == 0:
           continue
       cursor.execute(sql)
       if cursor.rowcount:
           logging.info('%s returned %d rows', sql, cursor.rowcount)
       elif cursor.description:
           logging.info('%s executed', sql)
           for row in cursor.fetchall():
               logging.info('%s', row)
   
def runTransfer(filename):
    logging.info( 'Input file to process...')
    for line in open(filename).readlines():
       print(line.strip())
    print('-----------------\n')
    reader= csv.reader(open(filename,'r'), delimiter=',',quotechar='"')
    header=True
    cmdList=[]
    for row in reader:
        row=[r.strip() for r in row]
        if len(row) == 0 or (len(row[0].rstrip())==0) :
            logging.error( 'Ignoring line with empty axioma-id %s', row)
            continue
        if header:
            header=False
            for idx, r in enumerate(row):
                r=r.replace('-','').replace('_','')
                #print r
                if r in fieldsDict:
                    fieldsDict[r] = idx
            transferDict=dict([(i,j) for i,j in fieldsDict.items() if (fieldsDict[i] != -1)])
            if 'SECTIONS' not in transferDict or 'FROMDT' not in transferDict:
                logging.fatal( 'Must have SECTIONS and FROMDT specified in the .csv file')
                sys.exit(1)
            if 'THRUDT' not in transferDict:
                transferDict['THRUDT']=transferDict['FROMDT']
            continue
        
        # make sure you have cmds, one for model and one for market and one for the un model transfer
        modelcmd=[]
        marketcmd=[]
        unmodelcmd=[]
        unmarketcmd=[]
        cmd='python3 transfer.py'
        if 'PRODUCTION' in transferDict and row[transferDict['PRODUCTION']] in ("1",'y','Y'):
#             cmd='%s %s' %(cmd,'production.config')
            configFileName='production.config'
        else:
#             cmd='%s %s' % (cmd, 'staging.config')
            configFileName='staging.config'
        
        cmd='%s %s' % (cmd, configFileName)    
        MARKET_SECTIONS = getSectionsList(findConfig(os.path.join('MarketDB',configFileName)))
        MODEL_SECTIONS = getSectionsList(findConfig(configFileName, 'RiskModels'))    
            
            
        if 'COMMIT' not in transferDict or row[transferDict['COMMIT']] not in ("1",'y','Y'):
            cmd='%s -n' % cmd
    
        if 'SECTIONS' in transferDict:
            for section in row[transferDict['SECTIONS']].split(','):
                if section in UNMODEL_SECTIONS:
                    unmodelcmd.append(section)
                elif section in UNMARKET_SECTIONS:
                    unmarketcmd.append(section)
                elif int(row[transferDict['MARKETDB']]) == 1:
                    if section in MARKET_SECTIONS:
                        marketcmd.append(section)
                    else:
                        logging.warning('transfer section %s is not found in MarketDB.%s', section, configFileName)
                elif int(row[transferDict['MARKETDB']]) == 0:
                    if section in MODEL_SECTIONS:
                        modelcmd.append(section)
                    else:
                        logging.warning('transfer section %s is not found in RiskModels.%s', section, configFileName)
                else:
                    logging.warning('transfer section %s is not identified', section)
            #cmd='%s %s'% (cmd,'sections=%s' % row[transferDict['SECTIONS']])
            marketcmd=','.join(marketcmd)
            modelcmd=','.join(modelcmd)
            unmodelcmd=','.join(unmodelcmd)
            unmarketcmd=','.join(unmarketcmd)
            
        if unmodelcmd:
            if 'AXIOMAID' in transferDict and 'FROMDT' in transferDict and 'THRUDT' in transferDict:
                cmd='%s %s %s %s %s' % (cmd, row[transferDict['AXIOMAID']], row[transferDict['FROMDT']], row[transferDict['THRUDT']], unmodelcmd)
                cmd=cmd.replace('transfer.py', 'runUNTransfer.py')
            else:
                logging.warning( 'Badly formed commands in row. ignoring %s', unmodelcmd)
                unmodelcmd=''
        elif unmarketcmd:
            if 'AXIOMAID' in transferDict and 'FROMDT' in transferDict and 'THRUDT' in transferDict:
                cmd='%s %s %s %s %s' % (cmd, row[transferDict['AXIOMAID']], row[transferDict['FROMDT']], row[transferDict['THRUDT']], unmarketcmd)
                cmd=cmd.replace('transfer.py', 'runUNTransfer.py')
            else:
                logging.warning( 'Badly formed commands in row. ignoring %s', unmarketcmd)
                unmarketcmd=''
        else:
            if 'ISSUEID' in transferDict:
                cmd='%s %s'% (cmd,'issue-ids=%s' % row[transferDict['ISSUEID']])
            if 'SUBISSUEID' in transferDict:
                cmd='%s %s'% (cmd,'sub-issue-ids=%s' % row[transferDict['SUBISSUEID']])
            if 'AXIOMAID' in transferDict:
                cmd='%s %s'% (cmd,'axioma-ids=%s' % row[transferDict['AXIOMAID']])
            if 'FROMDT' in transferDict:
                cmd='%s dates=%s:%s'% (cmd,row[transferDict['FROMDT']],row[transferDict['THRUDT']]or row[transferDict['FROMDT']])
        #if 'MARKETDB' in transferDict and row[transferDict['MARKETDB']]=='1':
        #    cwd='./MarketDB'
        #else:
        #    cwd='.'
        if marketcmd:
            mktcmd='%s sections=%s' % (cmd, marketcmd)
            if cmd.find('axioma-ids= ') > 0:
                logging.warning('Must have axioma-ids specified. Ingoring command %s\n' % cmd)
                continue
            cwd=os.path.dirname(findConfig(os.path.join('MarketDB','production.config')))
            cmdList.append([cwd,mktcmd])
            print(cwd,mktcmd)
        if modelcmd:
            mdlcmd='%s sections=%s' % (cmd, modelcmd)
            cwd='.'
            cmdList.append([cwd,mdlcmd])
            print(cwd,mdlcmd)
        if unmodelcmd:
            cwd='.'
            cmdList.append([cwd,cmd])
            print(cwd,cmd)
        if unmarketcmd:
            cwd=os.path.dirname(findConfig(os.path.join('MarketDB','production.config')))
            cmdList.append([cwd,cmd])
            print(cwd,cmd)
    
        sys.stdout.flush()
    
    errorStatus=0
    
    for cwd,cmd in cmdList:
        logging.info( 'In %s executing %s' % (cwd, cmd) )
        sys.stdout.flush()
        obj=subprocess.Popen(cmd, shell=True, cwd=cwd)
        stat= obj.wait()
        logging.info( 'return status= %s',stat)
        print('------------------------')
        sys.stdout.flush()
        if stat != 0:
            errorStatus = 1
        
    return errorStatus

def main():
    usage = "usage: %prog [options] filename"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addDefaultCommandLine(cmdlineParser)
    cmdlineParser.add_option("--pre-market-sql", action="store",
                              default=None, dest="preMarketSQL",
                              help="marketDBsql filename before transfer")
    cmdlineParser.add_option("--market-transfer", action="store",
                              default=None, dest="marketTransfer",
                              help="marketDBsql filename before transfer")
    cmdlineParser.add_option("--pre-model-sql", action="store",
                              default=None, dest="preModelSQL",
                              help="modelDBsql filename before transfer")
    (options, args) = cmdlineParser.parse_args()
    Utilities.processDefaultCommandLine(options, cmdlineParser)


    for f in fields:
        fieldsDict[f]=-1
    filename= args[0]
    logging.info('start')

    # check to see if any SQL needs to be processed before calling transfers
    if options.preMarketSQL:
        marketDB = MarketDB.MarketDB(sid=options.marketDBSID, user=options.marketDBUser, passwd=options.marketDBPasswd)
        execSQL(marketDB.dbCursor, options.preMarketSQL) 
        marketDB.commitChanges()
        marketDB.finalize()

    if options.marketTransfer:
        runTransfer(options.marketTransfer)

    if options.preModelSQL:
        modelDB = ModelDB.ModelDB(sid=options.modelDBSID, user=options.modelDBUser, passwd=options.modelDBPasswd)
        execSQL(modelDB.dbCursor, options.preModelSQL) 
        modelDB.commitChanges()
        modelDB.finalize()

    errorStatus = runTransfer(filename)
    sys.exit(errorStatus)

               
if __name__=='__main__':
    main()
