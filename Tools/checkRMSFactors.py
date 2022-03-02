#
# Script to check that the rms_factor definitions for a model cover all
# countries and currencies mapped to the model by rmg_model_map.
#

import configparser
import datetime
import logging
import optparse
from riskmodels import Connections
from riskmodels import ModelID
from riskmodels import Utilities

def getCountries(mdlDB, rmsID):
    mdlDB.dbCursor.execute("""SELECT rmm.rmg_id, description, rmm.from_dt, rmm.thru_dt
       FROM rmg_model_map rmm JOIN risk_model_group rmg
         ON rmm.rmg_id=rmg.rmg_id
       WHERE rms_id = :rms_id""", rms_id=rmsID)
    countries = dict((rmg_id, (code, fromDt.date(), thruDt.date()))
                     for (rmg_id, code, fromDt, thruDt)
                     in mdlDB.dbCursor.fetchall())
    return countries

def getCurrencies(mdlDB, countries):
    currencies = dict()
    for (rmg_id, (code, f, t)) in countries.items():
        mdlDB.dbCursor.execute("""SELECT currency_code, from_dt, thru_dt
           FROM rmg_currency
           WHERE rmg_id = :rmg""", rmg=rmg_id)
        for (ccy, fromDt, thruDt) in mdlDB.dbCursor.fetchall():
            if thruDt.date() <= f or fromDt.date() >= t:
                continue
            fromDt = max(fromDt.date(), f)
            thruDt = min(thruDt.date(), t)
            if ccy not in currencies:
                currencies[ccy] = (fromDt, thruDt)
            else:
                (oldFrom, oldThru) = currencies[ccy]
                currencies[ccy] = (min(fromDt, oldFrom), max(thruDt, oldThru))
    return currencies

def getRMSFactors(mdlDB, rmsID):
    mdlDB.dbCursor.execute("""SELECT f.name, rf.from_dt, rf.thru_dt
      FROM rms_factor rf JOIN factor f
        ON rf.factor_id=f.factor_id
      WHERE rf.rms_id = :rms_id""", rms_id=rmsID)
    factors = dict((name, (fromDt.date(), thruDt.date()))
                    for (name, fromDt, thruDt) in mdlDB.dbCursor.fetchall())
    return factors

def getRMSInfo(mdlDB, rmsID):
    mdlDB.dbCursor.execute("""SELECT from_dt, thru_dt, mnemonic
      FROM risk_model_serie JOIN risk_model ON rm_id=model_id
      WHERE serial_id = :rms_id""", rms_id=rmsID)
    (rmsFrom, rmsThru, mnemonic) = mdlDB.dbCursor.fetchone()
    if mnemonic[-2:] == '-S':
        modelType = 'Stat'
    elif mnemonic[:4] == 'AXFX':
        modelType = 'FX'
    else:
        modelType = 'Fund'
    return (rmsFrom.date(), rmsThru.date(), modelType)

def checkRMSFactors(mdlDB, rmsID):
    (rmsFrom, rmsThru, modelType) = getRMSInfo(mdlDB, rmsID)
    print('checking RMS %s: %s %s - %s' % (
        rmsID, modelType, rmsFrom, rmsThru))
    countries = getCountries(mdlDB, rmsID)
    currencies = getCurrencies(mdlDB, countries)
    print('%d countries mapped to model' % len(countries))
    print('%d currencies mapped to model' % len(currencies))
    factors = getRMSFactors(mdlDB, rmsID)
    print('%d factors' % len(factors))
    if modelType == 'Fund':
        for (code, fromDt, thruDt) in countries.values():
            if thruDt <= rmsFrom:
                continue
            if code not in factors:
                print('Country factor missing for %s: %s to %s' % (
                    code, fromDt, thruDt))
            else:
                (factorFrom, factorThru) = factors[code]
                if fromDt != factorFrom:
                    print('FromDt mismatch for %s: %s != %s' % (
                        code, factorFrom, fromDt))
                if thruDt != factorThru:
                    print('ThruDt mismatch for %s: %s != %s' % (
                        code, factorThru, thruDt))
    for (code, (fromDt, thruDt)) in currencies.items():
        if thruDt <= rmsFrom:
            continue
        if code not in factors:
            print('Currency factor missing for %s: %s to %s' % (
                code, fromDt, thruDt))
        else:
            (factorFrom, factorThru) = factors[code]
            if fromDt != factorFrom:
                print('FromDt mismatch for %s: %s != %s' % (
                    code, factorFrom, fromDt))
            if thruDt != factorThru:
                print('ThruDt mismatch for %s: %s != %s' % (
                    code, factorThru, thruDt))
            

def main():
    usage = "usage: %prog [options] config-file rms-id"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addDefaultCommandLine(cmdlineParser)
    (options, args) = cmdlineParser.parse_args()
    if len(args) != 2:
        cmdlineParser.error("Incorrect number of arguments")
    configFile = open(args[0])
    config = configparser.ConfigParser()
    config.read_file(configFile)
    configFile.close()
    connections = Connections.createConnections(config)
    mdlDB = connections.modelDB
    mktDB = connections.marketDB
    rmsID = int(args[1])
    checkRMSFactors(mdlDB, rmsID)
    mdlDB.finalize()
    mktDB.finalize()

if __name__ == '__main__':
    main()
