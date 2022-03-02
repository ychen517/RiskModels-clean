
import configparser
import cx_Oracle
import datetime
import logging
import optparse
import os
from riskmodels import Connections
from riskmodels import ModelDB
from riskmodels import Utilities
os.environ["NLS_LANG"] = ".AL32UTF8"

class BaseProcessor:
    def __init__(self, tableName, modelDB):
        self.modelDB = modelDB
        self.tableName = tableName
        self.cxFieldType = {float: cx_Oracle.NUMBER, int: cx_Oracle.NUMBER,
                            str: cx_Oracle.STRING,
                            datetime.datetime: cx_Oracle.TIMESTAMP,
                            datetime.date: cx_Oracle.DATETIME}
        self.retrieveQuery = """SELECT %(fields)s FROM %(table)s
          WHERE %(keys)s""" % {
            'fields': ','.join(self.fieldNames),
            'table': tableName,
            'keys': ' AND '.join(['%s = :%s' % (i, i) for i in self.keys])}
        self.insertQuery = """INSERT INTO %(table)s (%(fields)s) VALUES(%(values)s)""" % {
            'fields': ','.join(self.fieldNames),
            'table': tableName,
            'values': ','.join([':%s' % i for i in self.fieldNames])}
    
    def processFile(self, dataFile, options):
        for line in dataFile:
            if len(line.strip()) == 0 or line[0] == '#':
                continue
            fields = line.strip().split('|')
            if len(fields) != len(self.fieldNames) and options.dontPadNull:
                logging.error('Wrong number of fields (%d != %d). Ignoring line: %s',
                              len(fields), len(self.fieldNames), line)
            else:
                if len(fields) < len(self.fieldNames):
                    fields = fields + (['']*(len(self.fieldNames)-len(fields)))
                if len(fields) > len(self.fieldNames):
                    fields = fields[:len(self.fieldNames)]
                self.processLine(fields, options)
    
    def processLine(self, fields, options):
        valDict = {}
        for (val, name, vType) in zip(fields, self.fieldNames, self.fieldTypes):
            if vType is int:
                if len(val) == 0:
                    val = None
                else:
                    val = int(val)
            elif vType is str:
                if len(val) == 0:
                    val = None
            elif vType is datetime.date:
                if len(val) == 0:
                    val = None
                else:
                    val = Utilities.parseISODate(val)
            elif vType is float:
                if len(val) == 0:
                    val = None
                else:
                    val = float(val)
            valDict[name] = val
        keyDict = dict()
        for (name, vType) in zip(self.fieldNames, self.fieldTypes):
            val = valDict[name]
            if name in self.keys:
                keyDict[name] = val
        if options.updateRMSID is not None:
            for key in ['model_id', 'serial_id', 'rms_id']:
                if key in keyDict and (keyDict[key] != int(options.updateRMSID)):
                    logging.debug('Skipping record with RMS ID %d', keyDict[key])
                    return 
        if options.updateRMID is not None:
            for key in ['rm_id', 'model_id']:
                if key in keyDict and keyDict[key] != int(options.updateRMID):
                    logging.debug('Skipping record with RM ID %d', keyDict[key])
                    return
        elif options.researchOnly:
            for key in ['model_id', 'serial_id', 'rms_id']:
                if key in keyDict and keyDict[key] > 0:
                    logging.debug('Skip production record with %s %d', key, keyDict[key])
                    return
        elif options.productionOnly:
            for key in ['model_id', 'serial_id', 'rms_id']:
                if key in keyDict and keyDict[key] < 0:
                    logging.debug('Skip research record with %s %d', key, keyDict[key])
                    return
        self.modelDB.dbCursor.execute(self.retrieveQuery, keyDict)
        r = self.modelDB.dbCursor.fetchall()
        curDict = dict([(i, None) for i in self.fieldNames])
        assert(len(r) <= 1)
        if len(r) == 1:
            for (val, name, vType) in zip(r[0], self.fieldNames, self.fieldTypes):
                if val is not None:
                    if vType is int:
                        val = int(val)
                    elif vType is datetime.date:
                        val = val.date()
                curDict[name] = val
            changeDict = dict()
            for (name, vType) in zip(self.fieldNames, self.fieldTypes):
                curVal = curDict[name]
                val = valDict[name]
                if val != curVal:
                    changeDict[name] = val
            if len(changeDict) > 0:
                updateQuery = """UPDATE %(table)s SET %(updates)s WHERE %(keys)s""" % {
                    'table': self.tableName,
                    'updates': ','.join(['%s=:%s' % (i, i) for i in changeDict.keys()]),
                    'keys': ' AND '.join(['%s = :%s' % (i, i) for i in self.keys])}
                print('Updating record %s.\nChanging %s' % (curDict, changeDict))
                changeDict.update(keyDict)
                self.modelDB.dbCursor.execute(updateQuery, changeDict)
        else:
            # New record
            print('Inserting new record: %s' % valDict)
            #print self.insertQuery, valDict
            self.modelDB.dbCursor.execute(self.insertQuery, valDict)

class ClassificationMember(BaseProcessor):
    def __init__(self, tableName, modelDB):
        self.fieldNames = ['id', 'name', 'description', 'family_id']
        self.fieldTypes = [int, str, str, int]
        self.keys = ['id']
        BaseProcessor.__init__(self, tableName, modelDB)
    
class CompositeFamilyModelMap(BaseProcessor):
    def __init__(self, tableName, modelDB):
        self.fieldNames = ['model_id', 'composite_family_id']
        self.fieldTypes = [int, int]
        self.keys = self.fieldNames
        BaseProcessor.__init__(self, tableName, modelDB)
    
class CurrencyInstrumentMap(BaseProcessor):
    def __init__(self, tableName, modelDB):
        self.fieldNames = ['currency_code', 'instrument_name', 'from_dt',
                           'thru_dt']
        self.fieldTypes = [str, str, datetime.date, datetime.date]
        self.keys = ['currency_code', 'from_dt']
        BaseProcessor.__init__(self, tableName, modelDB)
    
class Descriptor(BaseProcessor):
    def __init__(self, tableName, modelDB):
        self.fieldNames = ['descriptor_id', 'name', 'description', 'legacy', 'local']
        self.fieldTypes = [int, str, str, int, int]
        self.keys = ['descriptor_id']
        BaseProcessor.__init__(self, tableName, modelDB)
    
class Factor(BaseProcessor):
    def __init__(self, tableName, modelDB):
        self.fieldNames = ['factor_id', 'name', 'description', 'factor_type_id']
        self.fieldTypes = [int, str, str, int]
        self.keys = ['factor_id']
        BaseProcessor.__init__(self, tableName, modelDB)
    
class FactorType(BaseProcessor):
    def __init__(self, tableName, modelDB):
        self.fieldNames = ['factor_type_id', 'name', 'description']
        self.fieldTypes = [int, str, str]
        self.keys = ['factor_type_id']
        BaseProcessor.__init__(self, tableName, modelDB)
    
class Region(BaseProcessor):
    def __init__(self, tableName, modelDB):
        self.fieldNames = ['id', 'name', 'description', 'currency_code']
        self.fieldTypes = [int, str, str, str]
        self.keys = ['id']
        BaseProcessor.__init__(self, tableName, modelDB)

class RiskModel(BaseProcessor):
    def __init__(self, tableName, modelDB):
        self.fieldNames = ['model_id', 'name', 'description', 'mnemonic',
                           'numeraire', 'model_region']
        self.fieldTypes = [int, str, str, str, str, str]
        self.keys = ['model_id']
        BaseProcessor.__init__(self, tableName, modelDB)
    
class RiskModelGroup(BaseProcessor):
    def __init__(self, tableName, modelDB):
        self.fieldNames = ['rmg_id', 'description', 'region_id', 'mnemonic', 'iso_code', 'gmt_offset']
        self.fieldTypes = [int, str, int, str, str, float]
        self.keys = ['rmg_id']
        BaseProcessor.__init__(self, tableName, modelDB)
    
class RiskModelGroupCurrency(BaseProcessor):
    def __init__(self, tableName, modelDB):
        self.fieldNames = ['rmg_id', 'currency_code', 'from_dt', 'thru_dt']
        self.fieldTypes = [int, str, datetime.date, datetime.date]
        self.keys = ['rmg_id', 'currency_code']
        BaseProcessor.__init__(self, tableName, modelDB)
    
class RiskModelFamily(BaseProcessor):
    def __init__(self, tableName, modelDB):
        self.fieldNames = ['family_id', 'name', 'mnemonic', 'description']
        self.fieldTypes = [int, str, str,str]
        self.keys = ['family_id']
        BaseProcessor.__init__(self, tableName, modelDB)

class RiskModelFamilyMap(BaseProcessor):
    def __init__(self, tableName, modelDB):
        self.fieldNames = ['family_id', 'model_id', 'from_dt', 'thru_dt']
        self.fieldTypes = [int, int, datetime.date, datetime.date ]
        self.keys = ['family_id','model_id','from_dt']
        BaseProcessor.__init__(self, tableName, modelDB)

class RiskModelSerie(BaseProcessor):
    def __init__(self, tableName, modelDB):
        self.fieldNames = ['serial_id', 'rm_id', 'revision',
                           'from_dt', 'thru_dt', 'distribute']
        self.fieldTypes = [int, int, int, datetime.date,
                           datetime.date, int]
        self.keys = ['serial_id']
        BaseProcessor.__init__(self, tableName, modelDB)
    
class RMGDevStatus(BaseProcessor):
    def __init__(self, tableName, modelDB):
        self.fieldNames = ['rmg_id', 'developed', 'emerging', 'frontier', 'from_dt', 'thru_dt']
        self.fieldTypes = [int, int, int, int, datetime.date, datetime.date]
        self.keys = ['rmg_id', 'from_dt']
        BaseProcessor.__init__(self, tableName, modelDB)
    
class RMGModelMap(BaseProcessor):
    def __init__(self, tableName, modelDB):
        self.fieldNames = ['rms_id', 'rmg_id', 'from_dt', 'thru_dt', 'fade_dt', 'full_dt']
        self.fieldTypes = [int, int, datetime.date, datetime.date, datetime.date, datetime.date]
        self.keys = ['rms_id', 'rmg_id']
        BaseProcessor.__init__(self, tableName, modelDB)
    
class RMSFactor(BaseProcessor):
    def __init__(self, tableName, modelDB):
        self.fieldNames = ['rms_id', 'factor_id', 'from_dt', 'thru_dt']
        self.fieldTypes = [int, int, datetime.date, datetime.date]
        self.keys = ['rms_id', 'factor_id']
        BaseProcessor.__init__(self, tableName, modelDB)
    
class RMSFactorDescriptor(BaseProcessor):
    def __init__(self, tableName, modelDB):
        self.fieldNames = ['rms_id', 'factor_id', 'descriptor_id', 'scale']
        self.fieldTypes = [int, int, int, float]
        self.keys = ['rms_id', 'factor_id', 'descriptor_id']
        BaseProcessor.__init__(self, tableName, modelDB)

class RMSFactorTypePrefix(BaseProcessor):
    def __init__(self, tableName, modelDB):
        self.fieldNames = ['factor_type_id', 'rms_id', 'prefix']
        self.fieldTypes = [int, int, str]
        self.keys = ['factor_type_id', 'rms_id']
        BaseProcessor.__init__(self, tableName, modelDB)
    
class RMINestedEstu(BaseProcessor):
    def __init__(self, tableName, modelDB):
        self.fieldNames = ['rms_id', 'id', 'name']
        self.fieldTypes = [int, int, str]
        self.keys = ['rms_id', 'id']
        BaseProcessor.__init__(self, tableName, modelDB)
    
class SubFactor(BaseProcessor):
    def __init__(self, tableName, modelDB):
        self.fieldNames = ['factor_id', 'from_dt', 'thru_dt', 'sub_id']
        self.fieldTypes = [int, datetime.date, datetime.date, int]
        self.keys = ['sub_id', 'factor_id']
        BaseProcessor.__init__(self, tableName, modelDB)
    
class UnaryOperator(BaseProcessor):
    def __init__(self, tableName, modelDB):
        self.fieldNames = ['name', 'description']
        self.fieldTypes = [str, str]
        self.keys = ['name']
        BaseProcessor.__init__(self, tableName, modelDB)
    
class UnaryExpression(BaseProcessor):
    def __init__(self, tableName, modelDB):
        self.fieldNames = ['id', 'operator_name', 'argument1_expression_id']
        self.fieldTypes = [int, str, int]
        self.keys = ['id']
        BaseProcessor.__init__(self, tableName, modelDB)

class BinaryOperator(BaseProcessor):
    def __init__(self, tableName, modelDB):
        self.fieldNames = ['name', 'description']
        self.fieldTypes = [str, str]
        self.keys = ['name']
        BaseProcessor.__init__(self, tableName, modelDB)

class BinaryExpression(BaseProcessor):
    def __init__(self, tableName, modelDB):
        self.fieldNames = ['id', 'operator_name', 'argument1_expression_id', 'argument2_expression_id']
        self.fieldTypes = [int, str, int, int]
        self.keys = ['id']
        BaseProcessor.__init__(self, tableName, modelDB)

class ExchangeAttribute(BaseProcessor):
    def __init__(self, tableName, modelDB):
        self.fieldNames = ['classification_ref_id', 'mic_code']
        self.fieldTypes = [int, str]
        self.keys = ['classification_ref_id']
        BaseProcessor.__init__(self, tableName, modelDB)
    
class Expression(BaseProcessor):
    def __init__(self, tableName, modelDB):
        self.fieldNames = ['id', 'constant', 'unary_expression_id', 'binary_expression_id', 'variable', 'description']
        self.fieldTypes = [int, float, int, int, str, str]
        self.keys = ['id']
        BaseProcessor.__init__(self, tableName, modelDB)

class NestRegression(BaseProcessor):
    def __init__(self, tableName, modelDB):
        self.fieldNames = ['id', 'name']
        self.fieldTypes = [int, str]
        self.keys = ['id']
        BaseProcessor.__init__(self, tableName, modelDB)
    
class NestInputOutput(BaseProcessor):
    def __init__(self, tableName, modelDB):
        self.fieldNames = ['rms_id', 'nest_id', 'factor_id', 'input_output']
        self.fieldTypes = [int, int, int, str]
        self.keys = ['rms_id', 'nest_id', 'factor_id', 'input_output']
        BaseProcessor.__init__(self, tableName, modelDB)

class FutureEstuWeight(BaseProcessor):
    def __init__(self, tableName, modelDB):
        self.fieldNames = ['future_family_id', 'weight_expression_id', 'rms_id', 'nest_id']
        self.fieldTypes = [str, int, int, int]
        self.keys = ['rms_id', 'future_family_id']
        BaseProcessor.__init__(self, tableName, modelDB)
    
class FutureFamilyFactor(BaseProcessor):
    def __init__(self, tableName, modelDB):
        self.fieldNames = ['future_family_id', 'factor_id', 'exposure_expression_id', 'rms_id']
        self.fieldTypes = [str, int, int, int]
        self.keys = ['rms_id', 'future_family_id', 'factor_id']
        BaseProcessor.__init__(self, tableName, modelDB)

class RMGRegionMap(BaseProcessor):
    def __init__(self, tableName, modelDB):
        self.fieldNames = ['id', 'rmg_id', 'from_dt', 'thru_dt']
        self.fieldTypes = [int, int, datetime.date, datetime.date]
        self.keys = ['id', 'rmg_id', 'from_dt']
        BaseProcessor.__init__(self, tableName, modelDB)

tableDataFile = { 'classification_member': 'classification_member.txt',
                  'composite_family_model_map': 'composite_family_model_map.txt',
                  'currency_instrument_map': 'currency_instrument_map.txt',
                  'descriptor': 'descriptor.txt',
                  'factor': 'factor.txt',
                  'factor_type': 'factor_type.txt',
                  'region': 'region.txt',
                  'risk_model': 'risk_model.txt',
                  'risk_model_group': 'risk_model_group.txt',
                  'risk_model_serie': 'risk_model_serie.txt',
                  'rmg_currency': 'rmg_currency.txt',
                  'rmg_dev_status': 'rmg_dev_status.txt',
                  'rmg_model_map': 'rmg_model_map.txt',
                  'rms_factor': 'rms_factor.txt',
                  'rms_factor_descriptor': 'rms_factor_descriptor.txt',
                  'rms_factor_type_prefix': 'rms_factor_type_prefix.txt',
                  'sub_factor': 'sub_factor.txt',
                  'rmi_estu_nested_info': 'rmi_estu_nested_info.txt',
                  'unary_expression': 'unary_expression.txt',
                  'binary_expression': 'binary_expression.txt',
                  'exchange_attribute': 'exchange_attribute.txt',
                  'expression': 'expression.txt',
                  'nest': 'nest.txt',
                  'nest_io': 'nest_io.txt',
                  'future_family_estu_weight': 'future_family_estu_weight.txt',
                  'future_family_factor': 'future_family_factor.txt',
                  'risk_model_family': 'risk_model_family.txt', 
                  'risk_model_family_map': 'risk_model_family_map.txt', 
                  'rmg_region_map': 'rmg_region_map.txt',
                  }
tableDataClass = { 'classification_member': ClassificationMember,
                   'composite_family_model_map': CompositeFamilyModelMap,
                   'currency_instrument_map': CurrencyInstrumentMap,
                   'descriptor': Descriptor,
                   'factor': Factor,
                   'factor_type': FactorType,
                   'region': Region,
                   'risk_model': RiskModel,
                   'risk_model_group': RiskModelGroup,
                   'risk_model_serie': RiskModelSerie,
                   'rmg_currency': RiskModelGroupCurrency,
                   'rmg_dev_status': RMGDevStatus,
                   'rmg_model_map': RMGModelMap,
                   'rms_factor': RMSFactor,
                   'rms_factor_descriptor': RMSFactorDescriptor,
                   'rms_factor_type_prefix': RMSFactorTypePrefix,
                   'sub_factor': SubFactor,
                   'rmi_estu_nested_info': RMINestedEstu,
                   'unary_expression': UnaryExpression,
                   'binary_expression': BinaryExpression,
                   'exchange_attribute': ExchangeAttribute,
                   'expression': Expression,
                   'nest': NestRegression,
                   'nest_io': NestInputOutput,
                   'future_family_estu_weight': FutureEstuWeight,
                   'future_family_factor': FutureFamilyFactor,
                   'risk_model_family': RiskModelFamily,
                   'risk_model_family_map': RiskModelFamilyMap,
                   'rmg_region_map': RMGRegionMap,
                   }

if __name__ == '__main__':
    usage = "usage: %prog [options] config-file table [data-file]"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addDefaultCommandLine(cmdlineParser)
    cmdlineParser.add_option("-n", action="store_true",
                             default=False, dest="testOnly",
                             help="don't change the database")
    cmdlineParser.add_option("--production-only", action="store_true",
                             default=False, dest="productionOnly",
                             help="only update production models")
    cmdlineParser.add_option("--research-only", action="store_true",
                             default=False, dest="researchOnly",
                             help="only update research models")
    cmdlineParser.add_option("--rms-id","--rms_id", action="store",
                             default=None, dest="updateRMSID",
                             help="only update particular RMS ID")
    cmdlineParser.add_option("--rm-id","--rm_id", action="store",
                             default=None, dest="updateRMID",
                             help="only update particular RM ID")
    cmdlineParser.add_option("--dont-pad", action="store_true",
                             default=False, dest="dontPadNull",
                            help="don't pad missing fields with nulls")
    (options_, args_) = cmdlineParser.parse_args()
    if len(args_) < 2:
        cmdlineParser.error("Incorrect number of arguments")
    Utilities.processDefaultCommandLine(options_, cmdlineParser)
    
    configFile_ = open(args_[0])
    config_ = configparser.ConfigParser()
    config_.read_file(configFile_)
    configFile_.close()
    
    tableName_ = args_[1]
    if len(args_) == 3:
        dataFileName_ = args_[2]
    else:
        dataFileName_ = tableDataFile[tableName_]
    
    dataFile_ = open(dataFileName_, 'rt', encoding='utf-8')
    connections_ = Connections.createConnections(config_)
    
    processor = tableDataClass[tableName_](tableName_, connections_.modelDB)
    processor.processFile(dataFile_, options_)
    dataFile_.close()
    
    if options_.testOnly:
        logging.info('Reverting changes')
        connections_.modelDB.revertChanges()
    else:
        connections_.modelDB.commitChanges()
    Connections.finalizeConnections(connections_)
