
import pprint as pp
import os

#before running this make sure that you have '/tmp/factors.txt'
#also make sure that production.config point to the correct Model and MarketDBs
# select distinct factorName from fi_dim where not FACTORTYPE = 'FIIssueSpecific'
# or
# select distinct factorName from fi_dim where FACTORTYPE = 'FIIssueSpecific'
#and store the list in this file
#all files will be stored in self.outDir
#simply run the commands in updateScript
#to verify run these:

# select * from risk_model where model_id = 2000
# 
# select * from risk_model_serie where rm_id = 2000
# select * from risk_model_group
# select * from rmg_model_map where rms_id = 20000
# select * from factor_type where name = 'FixedIncome'
# select * from factor where FACTOR_TYPE_ID in (select FACTOR_TYPE_ID from factor_type where name = 'FixedIncome')
# select * from sub_factor where FACTOR_ID in (
# select factor_id from factor where FACTOR_TYPE_ID in (select FACTOR_TYPE_ID from factor_type where name = 'FixedIncome')
# )
# select * from rms_factor where RMS_ID = 20000

#after running updates:
#insert into RISK_MODEL_INSTANCE values(20000, '1-jun-2014', 0, 1, 1, null, 0)

#also update instances like so:
#update RISK_MODEL_INSTANCE set HAS_RETURNS = 1 where rms_id = 20000
#update RISK_MODEL_INSTANCE set HAS_RISKS = 1 where rms_id = 20000

#to match the factor names to axiomaRisk:
#update factor set NAME = replace(name, '.1Y', ' : 1Y')
#--select * from factor
#where FACTOR_TYPE_ID in (select FACTOR_TYPE_ID from factor_type where name = 'FixedIncome')
#and name like '%1Y'

#to look at cov pair:
#select SUB_FACTOR1_ID,f1.name n1,SUB_FACTOR2_ID,f2.name n2,value / 252 from RMI_COVARIANCE rc
# join sub_factor sf1 on rc.SUB_FACTOR1_ID=sf1.SUB_ID
# join factor f1 on f1.FACTOR_ID=sf1.FACTOR_ID
# join sub_factor sf2 on rc.SUB_FACTOR2_ID=sf2.SUB_ID
# join factor f2 on f2.factor_id=sf2.factor_id
#
#where RMS_ID=20000 
#and f1.name like 'ZA.ZAR%'
#and f2.name like 'ZA.ZAR%'
#order by SUB_FACTOR1_ID,f1.name, SUB_FACTOR2_ID,f2.name

class Model(object):
    def __init__(self, rm_id, revision, rms_id):

        self.rm_id = rm_id
        self.revision = revision
        self.rms_id = rms_id
        self.factorType = 13
        self.factorStart = 10000
        self.outDir = "/tmp/modelFiles"
        if not os.path.exists(self.outDir):
            os.makedirs(self.outDir) 
        self.rmgFile = '/tmp/rmg.txt'
        self.write_risk_model()
        self.write_rmg_model_map()
        self.write_risk_model_serie()
        self.write_factor_type()
        self.factorFile = "/tmp/factors.txt"
        self.factors = self.readFactors()
        self.write_factor()
        self.write_sub_factor()
        self.write_rms_factor()
        self.writeUpdateScript()

    def write_risk_model(self): 
        fileName = os.path.join(self.outDir, 'risk_model.txt')   
        f = open(fileName,'w')
        l = '%s|FIAxioma2014MH|Fixed Income Model|AXFI|USD|United States' % (self.rm_id)
        f.write(l + os.linesep) 

    def write_risk_model_serie(self): 
        fileName = os.path.join(self.outDir, 'risk_model_serie.txt')   
        f = open(fileName,'w')
        l = '%s|%s|1|2001-01-01|2999-12-31|0' % (self.rms_id, self.rm_id)
        f.write(l + os.linesep) 

    def readRmg(self):
# select rmg_id from rmg_model_map where rms_id = 109        
        rmgs = []
        with open(self.rmgFile) as f:
            for l in f:
                l = l.strip()
                if not l == '':
                    rmgs.append(l)
        return rmgs
    
    def write_rmg_model_map(self): 
        rmgs = self.readRmg()
        fileName = os.path.join(self.outDir, 'rmg_model_map.txt')   
        f = open(fileName,'w')
        for rmg in rmgs:
            l = '%s|%s|1980-01-01|2999-12-31|2999-12-31|2999-12-31' % (self.rms_id, rmg)
            f.write(l + os.linesep)  
        
    def write_factor_type(self): 
        fileName = os.path.join(self.outDir, 'factor_type.txt')   
        f = open(fileName,'w')
        l = '%s|FixedIncome|Fixed Income factors' % self.factorType
        f.write(l + os.linesep) 

    def write_factor(self): 
        fileName = os.path.join(self.outDir, 'factor.txt')   
        f = open(fileName,'w')
        for k,v in self.factors.items():
            l = '%s|%s|%s|%s' % (k, v, v, self.factorType)
            f.write(l + os.linesep) 

    def write_sub_factor(self): 
        fileName = os.path.join(self.outDir, 'sub_factor.txt')   
        f = open(fileName,'w')
        for k in self.factors.keys():
            l = '%s|1980-01-01|2999-12-31|%s' % (k, k + 1)
            f.write(l + os.linesep) 
       
    def write_rms_factor(self): 
#        existingFactors = []
#        i = self.factorStart
#        with open(os.path.join(self.outDir, 'existing_factors.txt')) as f:
#            for l in f:
#                existingFactors.append(int(l.strip()))
#        
        fileName = os.path.join(self.outDir, 'rms_factor.txt')   
        f = open(fileName,'w')
#        for i in existingFactors:
#            l = '%s|%s|1980-01-01|2999-12-31' % (self.rms_id, i)
#            f.write(l + os.linesep)             
        for k in self.factors.keys():
            l = '%s|%s|1980-01-01|2999-12-31' % (self.rms_id, k)
            f.write(l + os.linesep) 
            
    def readFactors(self):
        factors = {}
        i = self.factorStart
        with open(self.factorFile) as f:
            for l in f:
                l = l.strip()
                if not l == '':
                    i += 1
                    factors[i] = l[:64]
        return factors
    
    def writeUpdateScript(self):
        cmds = []
        fileName = os.path.join(self.outDir, 'updateScript')   
        f = open(fileName,'w')
        l = 'cd /home/nborshansky/workspace/4.0/Tables'
        f.write(l + os.linesep) 
        l = 'export PYTHONPATH=..'
        f.write(l + os.linesep) 
        cmd = 'python3 ./updateTable.py ../production.config --production-only '
        cmds.append(cmd + 'risk_model /tmp/modelFiles/risk_model.txt')
        cmds.append(cmd + 'risk_model_serie /tmp/modelFiles/risk_model_serie.txt')
        cmds.append(cmd + 'rmg_model_map /tmp/modelFiles/rmg_model_map.txt')
        cmds.append(cmd + 'factor_type /tmp/modelFiles/factor_type.txt')
        cmds.append(cmd + 'factor /tmp/modelFiles/factor.txt')
        cmds.append(cmd + 'sub_factor /tmp/modelFiles/sub_factor.txt')
        cmds.append(cmd + 'rms_factor /tmp/modelFiles/rms_factor.txt')
        for l in cmds:
            f.write(l + os.linesep) 

class Model1(Model):
# just a cludge to get specific risks
    def __init__(self, rm_id, revision, rms_id):
        self.rm_id = rm_id
        self.revision = revision
        self.rms_id = rms_id
        self.factorType = 13
        self.factorStart = 500000
        self.outDir = "/tmp/modelFiles"
        if not os.path.exists(self.outDir):
            os.makedirs(self.outDir) 
        self.rmgFile = '/tmp/rmg.txt'
        self.write_risk_model()
        self.write_rmg_model_map()
        self.write_risk_model_serie()
        self.write_factor_type()
        self.factorFile = "/tmp/factors.txt"
        self.factors = self.readFactors()
        self.write_factor()
        self.write_sub_factor()
        self.write_rms_factor()
        self.writeUpdateScript()

    def write_risk_model(self): 
        fileName = os.path.join(self.outDir, 'risk_model.txt')   
        f = open(fileName,'w')
        l = '%s|FIAxioma2014MH1|Fixed Income Model1|AXFI1|USD|United States' % (self.rm_id)
        f.write(l + os.linesep) 

class Model2(Model):
# just a cludge to get specific risks
    def __init__(self, rm_id, revision, rms_id):
        self.rm_id = rm_id
        self.revision = revision
        self.rms_id = rms_id
        self.factorType = 13
        self.factorStart = 500000
        self.outDir = "/tmp/modelFiles"
        if not os.path.exists(self.outDir):
            os.makedirs(self.outDir) 
        self.rmgFile = '/tmp/rmg.txt'
        self.write_risk_model()
        self.write_rmg_model_map()
        self.write_risk_model_serie()
        self.write_factor_type()
        self.factorFile = "/tmp/factors.txt"
        self.factors = self.readFactors()
        self.write_factor()
        self.write_sub_factor()
        self.write_rms_factor()
        self.writeUpdateScript()

    def write_risk_model(self): 
        fileName = os.path.join(self.outDir, 'risk_model.txt')   
        f = open(fileName,'w')
        l = '%s|FIAxioma2014MH1|Fixed Income Model1|AXFI2|USD|United States' % (self.rm_id)
        f.write(l + os.linesep) 
                
class UnivModel(Model):
    def __init__(self, rm_id, revision, rms_id):

        self.rm_id = rm_id
        self.revision = revision
        self.rms_id = rms_id

        self.outDir = "/tmp/univModelFiles"
        if not os.path.exists(self.outDir):
            os.makedirs(self.outDir)
        self.rmgFile = '/tmp/rmg.txt'
        self.factorFile = "/tmp/allFactors.txt"
        self.factors = self.readFactors()
        self.write_risk_model()
        self.write_rmg_model_map()
        self.write_risk_model_serie()
        self.write_rms_factor()
        self.writeUpdateScript()

    def write_risk_model(self): 
        fileName = os.path.join(self.outDir, 'risk_model.txt')   
        f = open(fileName,'w')
        l = '%s|Univ10AxiomaMH|Universal Model|AXUN-MH|USD|United States' % (self.rm_id)
        f.write(l + os.linesep) 
        
    def writeUpdateScript(self):
        cmds = []
        fileName = os.path.join(self.outDir, 'updateScript')   
        f = open(fileName,'w')
        l = 'cd /home/nborshansky/workspace/4.0/Tables'
        f.write(l + os.linesep) 
        l = 'export PYTHONPATH=..'
        f.write(l + os.linesep) 
        cmd = 'python3 ./updateTable.py ../production.config --production-only '
        cmds.append(cmd + 'risk_model /tmp/univModelFiles/risk_model.txt')
        cmds.append(cmd + 'risk_model_serie /tmp/univModelFiles/risk_model_serie.txt')
        cmds.append(cmd + 'rmg_model_map /tmp/univModelFiles/rmg_model_map.txt')
        cmds.append(cmd + 'rms_factor /tmp/univModelFiles/rms_factor.txt')
        for l in cmds:
            f.write(l + os.linesep) 
            
    def readFactors(self):
# select rf.factor_id, f.name, ft.name type from rms_factor rf, factor f, factor_type ft
# where rms_id in (109, 10001, 20000)
# and rf.factor_id = f.factor_id
# and f.FACTOR_TYPE_ID = ft.FACTOR_TYPE_ID
# save the result in /tmp/allFactors.txt
        factors = {}
        with open(self.factorFile) as f:
            for l in f:
                l = l.strip()
                if not l == '':
                    fields = l.split()
                    factors[fields[0]] = fields[1]
        return factors

    def write_rms_factor(self): 
# after wtirting, as a hack, do this:
# select * from rms_factor where rms_id = 109 and THRU_DT 1 '1-jan-2099'
# update rms_factor set THRU_DT = '1-jan-2005' where rms_id = 30000 and factor_id = 338;
# update rms_factor set THRU_DT = '1-jan-2008' where rms_id = 30000 and factor_id = 344;
# update rms_factor set THRU_DT = '1-jan-2011' where rms_id = 30000 and factor_id = 345;
# update rms_factor set THRU_DT = '1-jan-2014' where rms_id = 30000 and factor_id = 352;
# update rms_factor set THRU_DT = '1-jul-2005' where rms_id = 30000 and factor_id = 356;
# update rms_factor set THRU_DT = '1-jan-2007' where rms_id = 30000 and factor_id = 358;
# update rms_factor set THRU_DT = '1-jan-2009' where rms_id = 30000 and factor_id = 359;
# update rms_factor set THRU_DT = '1-jan-2008' where rms_id = 30000 and factor_id = 360;
# update rms_factor set THRU_DT = '1-jan-2001' where rms_id = 30000 and factor_id = 362;
# update rms_factor set THRU_DT = '5-jul-1999' where rms_id = 30000 and factor_id = 376;
# update rms_factor set THRU_DT = '1-jul-2007' where rms_id = 30000 and factor_id = 387;
# update rms_factor set THRU_DT = '1-jan-2013' where rms_id = 30000 and factor_id = 390;
# update rms_factor set THRU_DT = '1-sep-2000' where rms_id = 30000 and factor_id = 391;
# update rms_factor set THRU_DT = '1-jan-2008' where rms_id = 30000 and factor_id = 392;
# select * from rms_factor where rms_id = 109 and FROM_DT > '1-jan-1980'        
# update rms_factor set FROM_DT = '1-jan-1999' where rms_id = 30000 and factor_id = 312;
# update rms_factor set FROM_DT = '1-jan-1998' where rms_id = 30000 and factor_id = 330;
# update rms_factor set FROM_DT = '1-jan-2005' where rms_id = 30000 and factor_id = 334;
# update rms_factor set FROM_DT = '5-jul-1999' where rms_id = 30000 and factor_id = 341;
# update rms_factor set FROM_DT = '1-jul-2005' where rms_id = 30000 and factor_id = 357;
# update rms_factor set FROM_DT = '1-jan-2008' where rms_id = 30000 and factor_id = 361;
# update rms_factor set FROM_DT = '1-jul-2007' where rms_id = 30000 and factor_id = 395;
# update rms_factor set FROM_DT = '1-jan-2013' where rms_id = 30000 and factor_id = 397;            
        fileName = os.path.join(self.outDir, 'rms_factor.txt')   
        f = open(fileName,'w')            
        for k in self.factors.keys():
            l = '%s|%s|1980-01-01|2999-12-31' % (self.rms_id, k)
            f.write(l + os.linesep) 
            
um = UnivModel(3000, 1, 30000)        
# a = Model(2000, 1, 20000)
# pp.pprint(a.factors)
# m1 = Model1(2500, 1, 25000)
# m = Model2(2700, 1, 27000)
