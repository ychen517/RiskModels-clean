
import numpy.ma as ma
import numpy
import logging
from riskmodels import LegacyUtilities as Utilities
from riskmodels import RiskCalculator
from riskmodels import LegacyFactorReturns
from riskmodels.Matrices import ExposureMatrix
from riskmodels.Factors import ModelFactor

def defaultExposureParametersV3(rm, styleList, overrider=False):
    rm.totalStyles = []
    styleParameters = dict()

    for styleName in styleList:
        params = Utilities.Struct()
        if hasattr(rm, 'DescriptorWeights') and (styleName in rm.DescriptorWeights):
            params.descriptorWeights = rm.DescriptorWeights[styleName]
        if hasattr(rm, 'smallCapMap') and (styleName in rm.smallCapMap):
            params.bounds = rm.smallCapMap[styleName]
        if hasattr(rm, 'noProxyList') and (styleName in rm.noProxyList):
            params.dontProxy = True
        if hasattr(rm, 'fillMissingList') and (styleName in rm.fillMissingList):
            params.fillMissing = True
        if hasattr(rm, 'fillWithZeroList') and (styleName in rm.fillWithZeroList):
            params.fillWithZero = True
        if hasattr(rm, 'shrinkList') and (styleName in rm.shrinkList):
            params.shrinkValue = True
            params.daysBack = rm.shrinkList[styleName]
        if hasattr(rm, 'orthogList') and (styleName in rm.orthogList):
            params.orthog = rm.orthogList[styleName][0]
            params.sqrtWt = rm.orthogList[styleName][1]
            params.orthogCoef = rm.orthogList[styleName][2]
        rm.totalStyles.append(ModelFactor(styleName, styleName))
        styleParameters[styleName] = params
    rm.styleParameters = styleParameters
    return

def defaultExposureParameters(
        rm,
        styleList,
        overrider=False):

    rm.totalStyles = []
    styleParameters = dict()

    if 'CDI' in styleList:
        params = Utilities.Struct()
        params.standardization = 'local'
        rm.totalStyles.append(ModelFactor('CDI', 'CDI'))
        styleParameters['CDI'] = params

    # Value
    if 'Value (2_1)' in styleList:
        params = Utilities.Struct()
        params.fillMissing = True
        params.standardization = 'regional'
        params.descriptors = ['Book-to-Price', 'Earnings-to-Price']
        rm.totalStyles.append(ModelFactor('Value (2_1)', 'Value (2_1)'))
        styleParameters['Value (2_1)'] = params

    if 'Book-to-Price' in styleList:
        params = Utilities.Struct()
        params.fillMissing = True
        params.standardization = 'regional'
        params.descriptors = ['Book-to-Price']
        rm.totalStyles.append(ModelFactor('Book-to-Price', 'Book-to-Price'))
        styleParameters['Book-to-Price'] = params

    if 'Earnings-to-Price' in styleList:
        params = Utilities.Struct()
        params.fillMissing = True
        params.standardization = 'regional'
        params.descriptors = ['Earnings-to-Price']
        rm.totalStyles.append(ModelFactor('Earnings-to-Price', 'Earnings-to-Price'))
        styleParameters['Earnings-to-Price'] = params

    if 'Earnings-to-Price Nu' in styleList:
        params = Utilities.Struct()
        params.fillMissing = True
        params.standardization = 'regional'
        params.descriptors = ['Earnings-to-Price Nu']
        rm.totalStyles.append(ModelFactor('Earnings-to-Price Nu', 'Earnings-to-Price Nu'))
        styleParameters['Earnings-to-Price Nu'] = params

    if 'Earnings Yield' in styleList:
        params = Utilities.Struct()
        params.fillMissing = True
        params.standardization = 'regional'
        params.descriptors = ['Earnings_to_Price', 'Est_Earnings_to_Price']
        rm.totalStyles.append(ModelFactor('Earnings Yield', 'Earnings Yield'))
        styleParameters['Earnings Yield'] = params

    if 'Sales-to-Price' in styleList:
        params = Utilities.Struct()
        params.fillMissing = True
        params.standardization = 'regional'
        params.descriptors = ['Sales-to-Price']
        rm.totalStyles.append(ModelFactor('Sales-to-Price', 'Sales-to-Price'))
        styleParameters['Sales-to-Price'] = params

    if 'Est Earnings-to-Price' in styleList:
        params = Utilities.Struct()
        params.fillMissing = True
        params.standardization = 'regional'
        params.descriptors = ['Est Earnings-to-Price']
        rm.totalStyles.append(ModelFactor('Est Earnings-to-Price', 'Est Earnings-to-Price'))
        styleParameters['Est Earnings-to-Price'] = params

    if 'Est Earnings-to-Price Nu' in styleList:
        params = Utilities.Struct()
        params.fillMissing = True
        params.standardization = 'regional'
        params.descriptors = ['Est Earnings-to-Price Nu']
        rm.totalStyles.append(ModelFactor('Est Earnings-to-Price Nu', 'Est Earnings-to-Price Nu'))
        styleParameters['Est Earnings-to-Price Nu'] = params

    if 'Value' in styleList:
        params = Utilities.Struct()
        params.fillMissing = True
        params.standardization = 'regional'
        params.descriptors = ['Book-to-Price Nu', 'Earnings-to-Price Nu', 'Est Earnings-to-Price Nu']
        rm.totalStyles.append(ModelFactor('Value', 'Value'))
        styleParameters['Value'] = params

    if 'Value Legacy EstETP' in styleList:
        params = Utilities.Struct()
        params.fillMissing = True
        params.standardization = 'regional'
        params.descriptors = ['Book-to-Price Nu','Est Earnings-to-Price']
        rm.totalStyles.append(ModelFactor('Value Legacy EstETP', 'Value Legacy EstETP'))
        styleParameters['Value Legacy EstETP'] = params

    # Leverage
    if 'Leverage' in styleList:
        params = Utilities.Struct()
        params.fillMissing = True
        params.standardization = 'regional'
        params.descriptors = ['Debt-to-Assets']
        rm.totalStyles.append(ModelFactor('Leverage', 'Leverage'))
        styleParameters['Leverage'] = params

    if 'Debt-to-Equity' in styleList:
        params = Utilities.Struct()
        params.fillMissing = True
        params.standardization = 'regional'
        params.descriptors = ['Debt-to-Equity']
        rm.totalStyles.append(ModelFactor('Debt-to-Equity', 'Debt-to-Equity'))
        styleParameters['Debt-to-Equity'] = params

    if 'Debt-to-MarketCap' in styleList:
        params = Utilities.Struct()
        params.fillMissing = True
        params.standardization = 'regional'
        params.descriptors = ['Debt-to-MarketCap']
        rm.totalStyles.append(ModelFactor('Debt-to-MarketCap', 'Debt-to-MarketCap'))
        styleParameters['Debt-to-MarketCap'] = params

    # Growth
    if 'Growth (2_1)' in styleList:
        params = Utilities.Struct()
        params.fillMissing = True
        params.standardization = 'regional'
        params.descriptors = ['Plowback times ROE', 'Sales Growth', 'Earnings Growth']
        rm.totalStyles.append(ModelFactor('Growth (2_1)', 'Growth (2_1)'))
        styleParameters['Growth (2_1)'] = params

    if 'Sustainable Growth Rate' in styleList:
        params = Utilities.Struct()
        params.fillMissing = True
        params.standardization = 'regional'
        params.descriptors = ['Plowback times ROE']
        rm.totalStyles.append(ModelFactor('Sustainable Growth Rate', 'Sustainable Growth Rate'))
        styleParameters['Sustainable Growth Rate'] = params

    if 'Dividend Payout' in styleList:
        params = Utilities.Struct()
        params.fillWithZero = True
        params.dontProxy = True
        params.standardization = 'regional'
        params.descriptors = ['Dividend Payout']
        rm.totalStyles.append(ModelFactor('Dividend Payout', 'Dividend Payout'))
        styleParameters['Dividend Payout'] = params

    if 'Proxied Dividend Payout' in styleList:
        params = Utilities.Struct()
        params.fillWithZero = True
        params.dontProxy = True
        params.standardization = 'regional'
        params.descriptors = ['Proxied Dividend Payout']
        rm.totalStyles.append(ModelFactor('Proxied Dividend Payout', 'Proxied Dividend Payout'))
        styleParameters['Proxied Dividend Payout'] = params

    if 'Return-on-Equity' in styleList:
        params = Utilities.Struct()
        params.fillMissing = True
        params.standardization = 'regional'
        params.descriptors = ['Return-on-Equity']
        rm.totalStyles.append(ModelFactor('Return-on-Equity', 'Return-on-Equity'))
        styleParameters['Return-on-Equity'] = params

    if 'Profitability' in styleList:
        params = Utilities.Struct()
        params.fillMissing = True
        params.standardization = 'regional'
        params.descriptors = ['Profitability']
        rm.totalStyles.append(ModelFactor('Profitability', 'Profitability'))
        styleParameters['Profitability'] = params

    if 'Return-on-Assets' in styleList:
        params = Utilities.Struct()
        params.fillMissing = True
        params.standardization = 'regional'
        params.descriptors = ['Return-on-Assets']
        rm.totalStyles.append(ModelFactor('Return-on-Assets', 'Return-on-Assets'))
        styleParameters['Return-on-Assets'] = params
        
    if 'Growth' in styleList:
        params = Utilities.Struct()
        params.fillMissing = True
        params.standardization = 'regional'
        params.legacyDescriptors = False
        params.descriptors = ['Est Earnings Growth', 'Est Sales Growth']
        rm.totalStyles.append(ModelFactor('Growth', 'Growth'))
        styleParameters['Growth'] = params

    if 'Earnings Growth' in styleList:
        params = Utilities.Struct()
        params.fillMissing = True
        params.standardization = 'regional'
        params.descriptors = ['Earnings Growth']
        rm.totalStyles.append(ModelFactor('Earnings Growth', 'Earnings Growth'))
        styleParameters['Earnings Growth'] = params

    if 'Est Earnings Growth' in styleList:
        params = Utilities.Struct()
        params.fillMissing = True
        params.standardization = 'regional'
        params.descriptors = ['Est Earnings Growth']
        rm.totalStyles.append(ModelFactor('Est Earnings Growth', 'Est Earnings Growth'))
        styleParameters['Est Earnings Growth'] = params
    
    if 'Sales Growth' in styleList:
        params = Utilities.Struct()
        params.fillMissing = True
        params.standardization = 'regional'
        params.descriptors = ['Sales Growth']
        rm.totalStyles.append(ModelFactor('Sales Growth', 'Sales Growth'))
        styleParameters['Sales Growth'] = params

    if 'Est Sales Growth' in styleList:
        params = Utilities.Struct()
        params.fillMissing = True
        params.standardization = 'regional'
        params.descriptors = ['Est Sales Growth']
        rm.totalStyles.append(ModelFactor('Est Sales Growth', 'Est Sales Growth'))
        styleParameters['Est Sales Growth'] = params

    if 'Est Revenue' in styleList:
        params = Utilities.Struct()
        params.fillMissing = True
        params.standardization = 'regional'
        params.descriptors = ['Est Revenue']
        rm.totalStyles.append(ModelFactor('Est Revenue', 'Est Revenue'))
        styleParameters['Est Revenue'] = params

    # Dividend yield
    if 'Dividend Yield' in styleList:
        params = Utilities.Struct()
        params.fillWithZero = True
        params.dontProxy = True
        params.standardization = 'regional'
        params.descriptors =  ['Dividend Yield']
        rm.totalStyles.append(ModelFactor('Dividend Yield', 'Dividend Yield'))
        styleParameters['Dividend Yield'] = params
        
    if 'DivYield ShareBuyback' in styleList:
        params = Utilities.Struct()
        params.fillWithZero = True
        params.dontProxy = True
        params.standardization = 'regional'
        params.descriptors =  ['Dividend Yield', 'Share Buyback']
        rm.totalStyles.append(ModelFactor('DivYield ShareBuyback', 'DivYield ShareBuyback'))
        styleParameters['DivYield ShareBuyback'] = params

    # Size
    if 'Size' in styleList:
        params = Utilities.Struct()
        params.standardization = 'global'
        rm.totalStyles.append(ModelFactor('Size', 'Size'))
        styleParameters['Size'] = params

    if 'Size Non-linear' in styleList:
        params = Utilities.Struct()
        params.standardization = 'global'
        params.orthog = ['Size']
        rm.totalStyles.append(ModelFactor('Size Non-linear', 'Size Non-linear'))
        styleParameters['Size Non-linear'] = params

    # Liquidity
    if 'Liquidity (2_1)' in styleList:
        params = Utilities.Struct()
        params.daysBack = 21
        params.median = False
        params.simple = False
        params.lnComb = False
        params.shrinkValue = True
        params.standardization = 'global'
        rm.totalStyles.append(ModelFactor('Liquidity (2_1)', 'Liquidity (2_1)'))
        styleParameters['Liquidity (2_1)'] = params

    if 'Liquidity 60' in styleList:
        params = Utilities.Struct()
        params.daysBack = 60
        params.median = False
        params.simple = False
        params.lnComb = False
        params.shrinkValue = True
        params.standardization = 'global'
        rm.totalStyles.append(ModelFactor('Liquidity 60', 'Liquidity 60'))
        styleParameters['Liquidity 60'] = params

    if 'Liquidity SH' in styleList:
        params = Utilities.Struct()
        params.daysBack = 21
        params.median = False
        params.simple = False
        params.lnComb = True
        params.legacy = False
        params.shrinkValue = True
        params.standardization = 'global'
        rm.totalStyles.append(ModelFactor('Liquidity SH', 'Liquidity SH'))
        styleParameters['Liquidity SH'] = params
        
    if 'Liquidity' in styleList:
        params = Utilities.Struct()
        params.daysBack = 60
        params.median = False
        params.simple = False
        params.lnComb = True
        params.legacy = False
        params.shrinkValue = True
        params.standardization = 'global'
        rm.totalStyles.append(ModelFactor('Liquidity', 'Liquidity'))
        styleParameters['Liquidity'] = params

    if 'Liquidity (Median)' in styleList:
        params = Utilities.Struct()
        params.daysBack = 21
        params.median = True
        params.simple = False
        params.lnComb = False
        params.shrinkValue = True
        params.standardization = 'global'
        rm.totalStyles.append(ModelFactor('Liquidity (Median)', 'Liquidity (Median)'))
        styleParameters['Liquidity (Median)'] = params

    if 'Dollar Volume' in styleList:
        params = Utilities.Struct()
        params.daysBack = 21
        params.median = False
        params.simple = True
        params.lnComb = False
        params.shrinkValue = True
        params.standardization = 'global'
        rm.totalStyles.append(ModelFactor('Dollar Volume', 'Dollar Volume'))
        styleParameters['Dollar Volume'] = params

    if 'Dollar Volume (Median)' in styleList:
        params = Utilities.Struct()
        params.daysBack = 21
        params.median = True
        params.simple = True
        params.lnComb = False
        params.shrinkValue = True
        params.standardization = 'global'
        rm.totalStyles.append(ModelFactor('Dollar Volume (Median)', 'Dollar Volume (Median)'))
        styleParameters['Dollar Volume (Median)'] = params

    if 'Amihud Liquidity' in styleList:
        params = Utilities.Struct()
        params.daysBack = 60
        params.legacy = False
        params.fillMissing = False
        params.shrinkValue = True
        params.standardization = 'global'
        rm.totalStyles.append(ModelFactor('Amihud Liquidity', 'Amihud Liquidity'))
        styleParameters['Amihud Liquidity'] = params

    if 'Liquidity (Composite)' in styleList:
        params = Utilities.Struct()
        params.standardization = 'global'
        params.fillMissing = False
        params.descriptors = ['Liquidity Descriptor', 'Amihud Liquidity Descriptor', 'Proportion Returns Traded']
        rm.totalStyles.append(ModelFactor('Liquidity (Composite)', 'Liquidity (Composite)'))
        styleParameters['Liquidity (Composite)'] = params

    if 'Liquidity Beta' in styleList:
        params = Utilities.Struct()
        params.daysBack = 21
        params.standardization = 'global'
        rm.totalStyles.append(ModelFactor('Liquidity Beta', 'Liquidity Beta'))
        styleParameters['Liquidity Beta'] = params

    # Medium-Term Momentum
    name = 'Yearly Momentum'
    if name in styleList:
        params = Utilities.Struct()
        params.thruT = 0
        params.fromT = 250
        params.name = name
        params.shrinkValue = True
        params.daysBack = 250
        params.standardization = 'global'
        rm.totalStyles.append(ModelFactor(name, name))
        styleParameters[name] = params

    name = 'Medium-Term Momentum'
    if name in styleList:
        params = Utilities.Struct()
        params.thruT = 20
        params.fromT = 250
        params.name = name
        params.shrinkValue = True
        params.daysBack = 250
        params.standardization = 'global'
        rm.totalStyles.append(ModelFactor(name, name))
        styleParameters[name] = params

    name = 'Medium-Term Momentum (Clipped)'
    if name in styleList:
        params = Utilities.Struct()
        params.thruT = 20
        params.fromT = 250
        params.name = name
        params.shrinkValue = True
        params.daysBack = 250
        params.standardization = 'global'
        rm.totalStyles.append(ModelFactor(name, name))
        styleParameters[name] = params

    name = 'Medium-Term Momentum (6 Months)'
    if name in styleList:
        params = Utilities.Struct()
        params.thruT = 20
        params.fromT = 120
        params.name = name
        params.shrinkValue = True
        params.daysBack = 120
        params.standardization = 'global'
        rm.totalStyles.append(ModelFactor(name, name))
        styleParameters[name] = params

    # Short-Term Momentum
    name = 'Short-Term Momentum'
    if name in styleList:
        params = Utilities.Struct()
        params.thruT = 0
        params.fromT = 20
        params.name = name
        params.shrinkValue = True
        params.daysBack = 20
        params.standardization = 'global'
        rm.totalStyles.append(ModelFactor(name, name))
        styleParameters[name] = params

    name = 'Short-Term Momentum (Clipped)'
    if name in styleList:
        params = Utilities.Struct()
        params.thruT = 0
        params.fromT = 20
        params.name = name
        params.robust = True
        params.shrinkValue = True
        params.daysBack = 20
        params.standardization = 'global'
        rm.totalStyles.append(ModelFactor(name, name))
        styleParameters[name] = params
        
    # Short-Term Momentum Tm2
    name = 'Short-Term Momentum Tm2'
    if name in styleList:
        params = Utilities.Struct()
        params.thruT = 0
        params.fromT = 20
        params.name = name
        params.shrinkValue = True
        params.daysBack = 20
        params.standardization = 'global'
        rm.totalStyles.append(ModelFactor(name, name))
        styleParameters[name] = params

    # Exchange Rate Sensitivity
    if 'Exchange Rate Sensitivity' in styleList:
        params = Utilities.Struct()
        params.robust = False
        params.daysBack = 250
        params.numeraire = 'XDR'
        params.frequency = 'weekly'
        params.shrinkValue = True
        params.standardization = 'global'
        rm.totalStyles.append(ModelFactor('Exchange Rate Sensitivity', 'Exchange Rate Sensitivity'))
        styleParameters['Exchange Rate Sensitivity'] = params

    if 'Exchange Rate Sensitivity (GBP)' in styleList:
        params = Utilities.Struct()
        params.robust = False
        params.daysBack = 250
        params.numeraire = 'GBP'
        params.frequency = 'weekly'
        params.shrinkValue = True
        params.standardization = 'global'
        rm.totalStyles.append(ModelFactor(
            'Exchange Rate Sensitivity (GBP)', 'Exchange Rate Sensitivity (GBP)'))
        styleParameters['Exchange Rate Sensitivity (GBP)'] = params

    if 'Exchange Rate Sensitivity (EUR)' in styleList:
        params = Utilities.Struct()
        params.robust = False
        params.daysBack = 250
        params.numeraire = 'EUR'
        params.frequency = 'weekly'
        params.shrinkValue = True
        params.standardization = 'global'
        rm.totalStyles.append(ModelFactor(
            'Exchange Rate Sensitivity (EUR)', 'Exchange Rate Sensitivity (EUR)'))
        styleParameters['Exchange Rate Sensitivity (EUR)'] = params

    if 'Exchange Rate Sensitivity (JPY)' in styleList:
        params = Utilities.Struct()
        params.robust = False
        params.daysBack = 250
        params.numeraire = 'JPY'
        params.frequency = 'weekly'
        params.shrinkValue = True
        params.standardization = 'global'
        rm.totalStyles.append(ModelFactor(
            'Exchange Rate Sensitivity (JPY)', 'Exchange Rate Sensitivity (JPY)'))
        styleParameters['Exchange Rate Sensitivity (JPY)'] = params

    if 'Exchange Rate Sensitivity (USD)' in styleList:
        params = Utilities.Struct()
        params.robust = False
        params.daysBack = 250
        params.numeraire = 'USD'
        params.frequency = 'weekly'
        params.shrinkValue = True
        params.standardization = 'global'
        rm.totalStyles.append(ModelFactor(
            'Exchange Rate Sensitivity (USD)', 'Exchange Rate Sensitivity (USD)'))
        styleParameters['Exchange Rate Sensitivity (USD)'] = params

    # Market Sensitivity
    if 'Market Sensitivity CN' in styleList:
        params = Utilities.Struct()
        params.swAdj = True
        params.fillMissing = False
        params.standardization = 'global'
        params.shrinkValue = True
        params.daysBack = 250
        rm.totalStyles.append(ModelFactor('Market Sensitivity CN', 'Market Sensitivity CN'))
        styleParameters['Market Sensitivity CN'] = params

    if 'Market Sensitivity EM' in styleList:
        params = Utilities.Struct()
        params.swAdj = True
        params.fillMissing = False
        params.standardization = 'global'
        params.shrinkValue = True
        params.daysBack = 250
        rm.totalStyles.append(ModelFactor('Market Sensitivity EM', 'Market Sensitivity EM'))
        styleParameters['Market Sensitivity EM'] = params

    if 'Market Sensitivity' in styleList:
        params = Utilities.Struct()
        params.swAdj = True
        params.fillMissing = False
        params.standardization = 'global'
        params.historyScale = 1.0
        params.shrinkValue = True
        params.daysBack = 250
        rm.totalStyles.append(ModelFactor('Market Sensitivity', 'Market Sensitivity'))
        styleParameters['Market Sensitivity'] = params

    if 'Market Sensitivity SH' in styleList:
        params = Utilities.Struct()
        params.swAdj = True
        params.fillMissing = False
        params.standardization = 'global'
        params.historyScale = 0.5
        params.shrinkValue = True
        params.daysBack = 120
        rm.totalStyles.append(ModelFactor('Market Sensitivity SH', 'Market Sensitivity SH'))
        styleParameters['Market Sensitivity SH'] = params

    if 'Market Sensitivity (PartOg)' in styleList:
        params = Utilities.Struct()
        params.swAdj = True
        params.fillMissing = False
        params.standardization = 'global'
        params.orthog = ['Volatility']
        params.historyScale = 1.0
        params.shrinkValue = True
        params.daysBack = 250
        rm.totalStyles.append(ModelFactor('Market Sensitivity (PartOg)', 'Market Sensitivity (PartOg)'))
        styleParameters['Market Sensitivity (PartOg)'] = params

    if 'Market Sensitivity SH (PartOg)' in styleList:
        params = Utilities.Struct()
        params.swAdj = True
        params.fillMissing = False
        params.standardization = 'global'
        params.orthog = ['Volatility SH']
        params.historyScale = 0.5
        params.shrinkValue = True
        params.daysBack = 120
        rm.totalStyles.append(ModelFactor('Market Sensitivity SH (PartOg)', 'Market Sensitivity SH (PartOg)'))
        styleParameters['Market Sensitivity SH (PartOg)'] = params

    # Volatility
    if 'Volatility 60 Day' in styleList:
        params = Utilities.Struct()
        params.daysBack = 60
        params.standardization = 'global'
        params.fillMissing = False
        params.shrinkValue = True
        rm.totalStyles.append(ModelFactor('Volatility 60 Day', 'Volatility 60 Day'))
        styleParameters['Volatility 60 Day'] = params

    if 'Volatility 125 Day' in styleList:
        params = Utilities.Struct()
        params.daysBack = 125
        params.standardization = 'global'
        params.fillMissing = False
        params.shrinkValue = True
        rm.totalStyles.append(ModelFactor('Volatility 125 Day', 'Volatility 125 Day'))
        styleParameters['Volatility 125 Day'] = params

    if 'Volatility 250 Day' in styleList:
        params = Utilities.Struct()
        params.daysBack = 250
        params.standardization = 'global'
        params.fillMissing = False
        params.shrinkValue = True
        rm.totalStyles.append(ModelFactor('Volatility 250 Day', 'Volatility 250 Day'))
        styleParameters['Volatility 250 Day'] = params

    if 'Volatility SH' in styleList:
        params = Utilities.Struct()
        params.daysBack = 60
        params.standardization = 'global'
        params.fillMissing = False
        params.shrinkValue = True
        params.orthog = ['Market Sensitivity SH']
        rm.totalStyles.append(ModelFactor('Volatility SH', 'Volatility SH'))
        styleParameters['Volatility SH'] = params

    if 'Volatility 250 Day (PartOg)' in styleList:
        params=Utilities.Struct()
        params.daysBack=250
        params.standardization = 'global'
        params.fillMissing = False
        params.shrinkValue = True
        params.orthog = ['Market Sensitivity']
        rm.totalStyles.append(ModelFactor('Volatility 250 Day (PartOg)', 'Volatility 250 Day (PartOg)'))
        styleParameters['Volatility 250 Day (PartOg)'] = params

    if 'VolaSensity' in styleList:
        params = Utilities.Struct()
        params.standardization = 'global'
        params.fillMissing = False
        params.descriptors = ['Market Sensitivity Descriptor', 'Volatility Descriptor']
        rm.totalStyles.append(ModelFactor('VolaSensity', 'VolaSensity'))
        styleParameters['VolaSensity'] = params

    if 'Residual Volatility' in styleList:
        params = Utilities.Struct()
        params.robust = False
        params.swAdj = True
        params.fillMissing = False
        params.daysBack = 60
        params.shrinkValue = True
        params.standardization = 'global'
        rm.totalStyles.append(ModelFactor('Residual Volatility', 'Residual Volatility'))
        styleParameters['Residual Volatility'] = params

    if 'Historical Residual Volatility' in styleList:
        params = Utilities.Struct()
        params.swAdj = True
        params.daysBack = 250
        params.fillMissing = False
        params.shrinkValue = True
        params.standardization = 'global'
        rm.totalStyles.append(ModelFactor('Historical Residual Volatility', 'Historical Residual Volatility'))
        styleParameters['Historical Residual Volatility'] = params

    if 'Hist Resid Vol (PartOg)' in styleList:
        params = Utilities.Struct()
        params.swAdj = True
        params.daysBack = 250
        params.fillMissing = False
        params.standardization = 'global'
        params.shrinkValue = True
        params.orthog = ['Market Sensitivity']
        rm.totalStyles.append(ModelFactor('Hist Resid Vol (PartOg)', 'Hist Resid Vol (PartOg)'))
        styleParameters['Hist Resid Vol (PartOg)'] = params

    name = 'Volatility'
    if name in styleList:
        params = Utilities.Struct()
        params.daysBack=125
        params.standardization = 'global'
        params.fillMissing = False
        params.shrinkValue = True
        params.orthog = ['Market Sensitivity']
        rm.totalStyles.append(ModelFactor(name, name))
        styleParameters[name] = params

    name = 'Volatility (Orthog)'
    if name in styleList:
        params = Utilities.Struct()
        params.daysBack=125
        params.standardization = 'global'
        params.fillMissing = False
        params.shrinkValue = True
        params.orthog = ['Market Sensitivity']
        rm.totalStyles.append(ModelFactor(name, name))
        styleParameters[name] = params

    if 'Historical Residual Std SizeAdj' in styleList:
        params = Utilities.Struct()
        params.swAdj = True
        params.daysBack = 250
        params.fillMissing = False
        params.shrinkValue = True
        params.standardization = 'global'
        rm.totalStyles.append(ModelFactor('Historical Residual Std SizeAdj', 'Historical Residual Std SizeAdj'))
        styleParameters['Historical Residual Std SizeAdj'] = params

    if 'Ln Historical Residual Volatility' in styleList:
        params = Utilities.Struct()
        params.swAdj = True
        params.daysBack = 250
        params.fillMissing = False
        params.shrinkValue = True
        params.standardization = 'global'
        rm.totalStyles.append(ModelFactor('Ln Historical Residual Volatility', 'Ln Historical Residual Volatility'))
        styleParameters['Ln Historical Residual Volatility'] = params

    if 'Historical Residual Volatility ex Size' in styleList:
        params = Utilities.Struct()
        params.swAdj = True
        params.daysBack = 250
        params.fillMissing = False
        params.shrinkValue = True
        params.standardization = 'global'
        rm.totalStyles.append(ModelFactor('Historical Residual Volatility ex Size', 'Historical Residual Volatility ex Size'))
        styleParameters['Historical Residual Volatility ex Size'] = params

    if 'Historical Volatility' in styleList:
        params = Utilities.Struct()
        params.fillMissing = False
        params.daysBack = 250
        params.shrinkValue = True
        params.standardization = 'global'
        rm.totalStyles.append(ModelFactor('Historical Volatility', 'Historical Volatility'))
        styleParameters['Historical Volatility'] = params
     
    # Others
    if 'Returns Skewness' in styleList:
        params = Utilities.Struct()
        params.standardization = 'global'
        params.daysBack = 60
        rm.totalStyles.append(ModelFactor('Returns Skewness', 'Returns Skewness'))
        styleParameters['Returns Skewness'] = params

    if 'SmallCap 1' in styleList:
        params = Utilities.Struct()
        params.standardization = 'global'
        params.bounds=[90.0, 100.0]
        rm.totalStyles.append(ModelFactor('SmallCap 1', 'SmallCap 1'))
        styleParameters['SmallCap 1'] = params

    if 'SmallCap 2' in styleList:
        params = Utilities.Struct()
        params.standardization = 'global'
        params.bounds=[80.0, 90.0]
        rm.totalStyles.append(ModelFactor('SmallCap 2', 'SmallCap 2'))
        styleParameters['SmallCap 2'] = params

    if 'SmallCap 3' in styleList:
        params = Utilities.Struct()
        params.standardization = 'global'
        params.bounds=[0.0, 80.0]
        rm.totalStyles.append(ModelFactor('SmallCap 3', 'SmallCap 3'))
        styleParameters['SmallCap 3'] = params

    if 'MidCap' in styleList:
        params = Utilities.Struct()
        params.standardization = 'global'
        params.bounds=[80.0, 90.0]
        rm.totalStyles.append(ModelFactor('MidCap', 'MidCap'))
        styleParameters['MidCap'] = params

    if 'Est EBITDA' in styleList:
        params = Utilities.Struct()
        params.fillMissing = True
        params.standardization = 'global'
        params.descriptors = ['Est EBITDA']
        rm.totalStyles.append(ModelFactor('Est EBITDA', 'Est EBITDA'))
        styleParameters['Est EBITDA'] = params

    if 'Est Enterprise Value' in styleList:
        params = Utilities.Struct()
        params.fillMissing = True
        params.standardization = 'global'
        params.descriptors = ['Est Enterprise Value']
        rm.totalStyles.append(ModelFactor('Est Enterprise Value', 'Est Enterprise Value'))
        styleParameters['Est Enterprise Value'] = params
        
    if 'Est Cash-Flow-per-Share' in styleList:
        params = Utilities.Struct()
        params.fillMissing = True
        params.standardization = 'global'
        params.descriptors = ['Est Cash-Flow-per-Share']
        rm.totalStyles.append(ModelFactor('Est Cash-Flow-per-Share', 'Est Cash-Flow-per-Share'))
        styleParameters['Est Cash-Flow-per-Share'] = params

    if 'Est Avg Return-on-Equity' in styleList:
        params = Utilities.Struct()
        params.fillMissing = True
        params.standardization = 'global'
        params.descriptors = ['Est Avg Return-on-Equity']
        rm.totalStyles.append(ModelFactor('Est Avg Return-on-Equity', 'Est Avg Return-on-Equity'))
        styleParameters['Est Avg Return-on-Equity'] = params

    if 'Est Avg Return-on-Assets' in styleList:
        params = Utilities.Struct()
        params.fillMissing = True
        params.standardization = 'global'
        params.descriptors = ['Est Avg Return-on-Assets']
        rm.totalStyles.append(ModelFactor('Est Avg Return-on-Assets', 'Est Avg Return-on-Assets'))
        styleParameters['Est Avg Return-on-Assets'] = params
    
    if 'Est Avg ROAxROE' in styleList:
        params = Utilities.Struct()
        params.fillMissing = True
        params.standardization = 'global'
        params.descriptors = ['Est Avg Return-on-Assets', 'Est Avg Return-on-Equity']
        rm.totalStyles.append(ModelFactor('Est Avg ROAxROE', 'Est Avg ROAxROE'))
        styleParameters['Est Avg ROAxROE'] = params

    if 'Est ROAxROE' in styleList:
        params = Utilities.Struct()
        params.fillMissing = True
        params.standardization = 'global'
        params.descriptors = ['Est Avg Return-on-Assets', 'Est Avg Return-on-Equity']
        rm.totalStyles.append(ModelFactor('Est ROAxROE', 'Est ROAxROE'))
        styleParameters['Est ROAxROE'] = params
    
    if 'Share Buyback' in styleList:
        params = Utilities.Struct()
        params.fillWithZero = True
        params.standardization = 'global'
        params.descriptors = ['Share Buyback']
        rm.totalStyles.append(ModelFactor('Share Buyback', 'Share Buyback'))
        styleParameters['Share Buyback'] = params

    if 'Short Interest' in styleList:
        params = Utilities.Struct()
        params.fillWithZero = True
        params.standardization = 'global'
        params.descriptors = ['Short Interest']
        rm.totalStyles.append(ModelFactor('Short Interest', 'Short Interest'))
        styleParameters['Short Interest'] = params
    
    if 'Quality ROE-EPSStd-Lev' in styleList:
        params = Utilities.Struct()
        params.fillMissing = True
        params.standardization = 'global'
        params.descriptors = ['Est Avg Return-on-Equity','FY1 Est Earnings-per-Share Std','Debt-to-Assets']
        rm.totalStyles.append(ModelFactor('Quality ROE-EPSStd-Lev','Quality ROE-EPSStd-Lev'))
        styleParameters['Quality ROE-EPSStd-Lev'] = params

    if 'Quality ROA-EPSStd-Lev' in styleList:
        params = Utilities.Struct()
        params.fillMissing = True
        params.standardization = 'global'
        params.descriptors = ['Est Avg Return-on-Assets','FY1 Est Earnings-per-Share Std','Debt-to-Assets']
        rm.totalStyles.append(ModelFactor('Quality ROA-EPSStd-Lev','Quality ROA-EPSStd-Lev'))
        styleParameters['Quality ROA-EPSStd-Lev'] = params

    if 'Growth_nq' in styleList:
        params = Utilities.Struct()
        params.fillMissing = True
        params.standardization = 'global'
        rm.totalStyles.append(ModelFactor('Growth_nq','Growth_nq'))
        styleParameters['Growth_nq'] = params

    if 'Payout' in styleList:
        params = Utilities.Struct()
        params.fillMissing = True
        params.standardization = 'global'
        rm.totalStyles.append(ModelFactor('Payout','Payout'))
        styleParameters['Payout'] = params

    if 'Safety' in styleList:
        params = Utilities.Struct()
        params.fillMissing = True
        params.standardization = 'global'
        rm.totalStyles.append(ModelFactor('Safety','Safety'))
        styleParameters['Safety'] = params

    if 'Random Factor' in styleList:
        params = Utilities.Struct()
        params.fillMissing = False
        params.standardization = 'global'
        rm.totalStyles.append(ModelFactor('Random Factor', 'Random Factor'))
        styleParameters['Random Factor'] = params

    rm.styleParameters = styleParameters

class RegressionParameters2012:
    """Set up regression parameters for fundamental model types
    """
    def __init__(self, paramsDict):
        self.regressionParameters = paramsDict

    def getThinFactorInformation(self):
        dr = Utilities.Struct()
        dr.dummyReturnType = self.regressionParameters.get('dummyReturnType', 'market')
        dr.dummyThreshold = self.regressionParameters.get('dummyThreshold', 10.0)
        dr.modelDB = self.regressionParameters.get('modelDB', None)
        dr.factorNames = self.regressionParameters.get('factorNames', None)
        dr.factorIndices = self.regressionParameters.get('idxToCheck', None)
        return dr

    def getFactorConstraints(self):
        return self.regressionParameters.get('factorConstraints', list())

    def getRegressionOrder(self):
        return self.regressionParameters.get('regressionOrder', None)

    def getThinFactorCorrection(self):
        return self.regressionParameters.get('fixThinFactors', True)

    def getCalcVIF(self):
        return self.regressionParameters.get('calcVIF', False)

    def getWhiteStdErrors(self):
        return self.regressionParameters.get('whiteStdErrors', False)

    def getRlmKParameter(self):
        return self.regressionParameters.get('k_rlm', 5.0)

    def getExcludeFactors(self):
        return self.regressionParameters.get('excludeFactors', list())

    def getEstuName(self):
        return self.regressionParameters.get('estuName', 'main')

    def getRegWeights(self):
        return self.regressionParameters.get('regWeight', 'rootCap')

    def getClipWeights(self):
        return self.regressionParameters.get('clipWeights', True)

    def getExtraConstraints(self):
        return self.regressionParameters.get('addExtraConstraints', False)

    def realWeightsForConstraints(self):
        return self.regressionParameters.get('useRealMCapsForConstraints', False)

def getNextValue(data):
    if type(data) is not list:
        return data
    if len(data) < 2:
        return data[0]
    else:
        return data.pop(0)

def defaultRegressionParameters(
            rm, modelDB, 
            marketRegression=False,
            dummyType='market',
            dummyThreshold=10.0,
            kappa=5.0,
            useRealMCaps=False,
            scndRegList=[],
            scndRegEstus=[],
            overrider=False,
            regWeight='rootCap'
            ):
    regParameters = []
    excludeList = []
    for rl in scndRegList:
        excludeList.extend(rl)

    # Set up market and main regression parameters
    if marketRegression:
        rpMkt = typicalMarketRegressionParameters(rm, modelDB,
                kappa=kappa, useRealMCaps=useRealMCaps,)
        regParameters.append(rpMkt)
        excludeList.append(ExposureMatrix.InterceptFactor)

    # And main regression
    rpMain = typicalMainRegressionParameters(rm, modelDB,
                kappa=kappa, useRealMCaps=useRealMCaps,
                dummyType=dummyType, dummyThreshold=dummyThreshold,
                excludeList=excludeList, regWeight=regWeight)
    regParameters.append(rpMain)

    # Subsiduary regressions if any
    for (rl, re) in zip(scndRegList, scndRegEstus):
        srp = typicalMarketRegressionParameters(
                rm, modelDB,
                kappa=kappa, useRealMCaps=useRealMCaps,
                regressionList=rl,       
                estuName=re)   
        regParameters.append(srp)

    if overrider:
        overrider.overrideRegressionParams(regParameters)
    return LegacyFactorReturns.RobustRegressionLegacy(regParameters)
    
def typicalMarketRegressionParameters(
            rm, modelDB,
            dummyType='market',
            dummyThreshold=10.0,
            regressionList=[ExposureMatrix.InterceptFactor],
            kappa=5.0,
            useRealMCaps=False,
            regWeight='cap',
            estuName='market',
            overrider=False):

    # Initialise stuff
    constraintList = []
    excludeList = []
    regParameters = \
           {'fixThinFactors': True,
            'dummyReturnType': dummyType,
            'dummyThreshold': dummyThreshold,
            'modelDB': modelDB,
            'factorConstraints': [],
            'regressionOrder': regressionList,
            'excludeFactors': [],
            'clipWeights': True,
            'k_rlm': kappa,
            'useRealMCapsForConstraints': useRealMCaps,
            'estuName': estuName,
            'regWeight': regWeight,
            'whiteStdErrors': False
           }
    return RegressionParameters2012(regParameters)

def typicalMainRegressionParameters(
            rm, modelDB,
            dummyType='market',
            dummyThreshold=10.0,
            constrainedReg=True,
            regressionList=[],
            excludeList=[],
            kappa=5.0,
            useRealMCaps=False,
            regWeight='rootCap',
            estuName='main',
            overrider=False):

    # Set up factors and constraints for main regression
    if len(regressionList) == 0:
        regressionList = [ExposureMatrix.InterceptFactor,
                          ExposureMatrix.StyleFactor,
                          ExposureMatrix.IndustryFactor]
        if not rm.SCM:
            regressionList.append(ExposureMatrix.CountryFactor)
            regressionList.append(ExposureMatrix.LocalFactor)
    regressionList = [r for r in regressionList if r not in excludeList]

    if constrainedReg:
        constraintList = [LegacyFactorReturns.ConstraintSumToZero(ExposureMatrix.IndustryFactor)]
        if not rm.SCM:
            constraintList.append(LegacyFactorReturns.ConstraintSumToZero(ExposureMatrix.CountryFactor))
    else:
        constraintList = []

    if dummyType is None:
        fixThin = False
    else:
        fixThin = True

    # Main factor regression
    regParameters = \
           {'fixThinFactors': fixThin,
            'dummyReturnType': dummyType,
            'dummyThreshold': dummyThreshold,
            'modelDB': modelDB,
            'factorConstraints': constraintList,
            'regressionOrder': regressionList,
            'excludeFactors': excludeList,
            'k_rlm': kappa,
            'useRealMCapsForConstraints': useRealMCaps,
            'estuName': 'main',
            'regWeight': regWeight,
            'whiteStdErrors': False,
            }
    return RegressionParameters2012(regParameters)

def defaultRegressionParametersLegacy(
            rm,
            modelDB,
            dummyType=['market'],
            dummyThreshold=[10.0],
            marketReg=True,
            constrainedReg=False,
            scndRegs=[],
            scndRegEstus=[],
            k_rlm=[8.0, 1.345],
            regWeight=['cap', 'rootCap'],
            overrider=False):
    
    if (scndRegs == None) or (scndRegs == False):
        scndRegs = []

    # Optional initial market regression
    if marketReg:

        # Initialise stuff
        kappa = getNextValue(k_rlm)
        dType = getNextValue(dummyType)
        dThresh = getNextValue(dummyThreshold)
        rWt = getNextValue(regWeight)

        constraintList = []
        regressionList = [ExposureMatrix.InterceptFactor]
        excludeList = []
        regZeroParameters = \
               {'fixThinFactors': True,
                'dummyReturnType': dType,
                'dummyThreshold': dThresh,
                'modelDB': modelDB,
                'factorConstraints': constraintList,
                'regressionOrder': regressionList,
                'excludeFactors': excludeList,
                'clipWeights': True,
                'k_rlm': kappa,
                'estuName': 'market',
                'regWeight': rWt,
                'whiteStdErrors': False
               }
        regParameters = [RegressionParameters2012(regZeroParameters)]

    # Set up constraints for main regression
    constraintList = []
    excludeList = []
    if len(scndRegs) > 0:
        for reg in scndRegs:
            excludeList.extend(reg)

    if not marketReg:
        regressionList = [ExposureMatrix.InterceptFactor,
                          ExposureMatrix.StyleFactor,
                          ExposureMatrix.IndustryFactor]
    else:
        regressionList = [ExposureMatrix.StyleFactor,
                          ExposureMatrix.IndustryFactor]
    if constrainedReg:
        constraintList = [LegacyFactorReturns.ConstraintSumToZero(ExposureMatrix.IndustryFactor)]

    if not rm.SCM:
        regressionList.append(ExposureMatrix.CountryFactor)
        if constrainedReg:
            constraintList.append(LegacyFactorReturns.ConstraintSumToZero(ExposureMatrix.CountryFactor))

    # Main factor regression
    kappa = getNextValue(k_rlm)
    dType = getNextValue(dummyType)
    dThresh = getNextValue(dummyThreshold)
    rWt = getNextValue(regWeight)
    regOneParameters = \
           {'fixThinFactors': True,
            'dummyReturnType': dType,
            'dummyThreshold': dThresh,
            'modelDB': modelDB,
            'factorConstraints': constraintList,
            'regressionOrder': regressionList,
            'excludeFactors': excludeList,
            'k_rlm': kappa,
            'estuName': 'main',
            'regWeight': rWt,
            'whiteStdErrors': False,
            #'addExtraConstraints': True,
            }
    rm.mainRegParameters = RegressionParameters2012(regOneParameters)
    if not marketReg:
        regParameters = [rm.mainRegParameters]
    else:
        regParameters.append(rm.mainRegParameters)

    # Secondary regression
    if len(scndRegs) > 0:
        excludeList = []
        for scndReg in scndRegs:
            kappa = getNextValue(k_rlm)
            dType = getNextValue(dummyType)
            dThresh = getNextValue(dummyThreshold)
            rWt = getNextValue(regWeight)
            estuName = getNextValue(scndRegEstus)
            constraintList = []
            regressionList = scndReg
            excludeList = []
            regTwoParameters = \
                   {'fixThinFactors': False,
                    'dummyReturnType': dType,
                    'dummyThreshold': dThresh,
                    'modelDB': modelDB,
                    'factorConstraints': constraintList,
                    'regressionOrder': regressionList,
                    'excludeFactors': excludeList,
                    'estuName': estuName,
                    'clipWeights': False,
                    'k_rlm': kappa,
                    'regWeight': rWt,
                    'whiteStdErrors': False,
                    }
            regParameters.append(RegressionParameters2012(regTwoParameters))

    if overrider:
        overrider.overrideRegressionParams(regParameters)
    rm.returnCalculator = LegacyFactorReturns.RobustRegressionLegacy(regParameters)

class CovarianceParameters2012:
    """Stores parameters (half-life, etc.) for all risk-related
    computation procedures.
    Instantiate using 2 dictionaries containing the parameter
    names and values for covariance matrix and specific risk
    computation.
    """
    def __init__(self, covParamsDict):
        self.log = logging.getLogger('RiskCalculator.RiskParameters')
        self.covarianceParameters = covParamsDict

    def getCovarianceHalfLife(self):
        return (self.covarianceParameters.get('halfLife', 125))

    def getCovarianceSampleSize(self):
        hl = self.getCovarianceHalfLife()
        return (self.covarianceParameters.get('minObs', int(hl)),
                self.covarianceParameters.get('maxObs', int(hl*4)))

    def getCovarianceNeweyWestLag(self):
        return ( self.covarianceParameters.get('NWLag', 2))

    def getCovarianceDVAOptions(self):
        return (self.covarianceParameters.get('DVAWindow', None),
                self.covarianceParameters.get('DVAType', 'spline'),
                self.covarianceParameters.get('DVAUpperBound', 0.05),
                self.covarianceParameters.get('DVALowerBound', -0.10),
                self.covarianceParameters.get('downweightEnds', False))

    def getCovarianceComposition(self):
        return (self.covarianceParameters.get('useTransformMatrix', False),
                self.covarianceParameters.get('offDiagScaleFactor', 1.0))

    def useDeMeanedCovariance(self):
        return self.covarianceParameters.get('deMeanFlag', False)

    def useEqualWeightedCovariance(self):
        return self.covarianceParameters.get('equalWeightFlag', False)

    def useStructuredModel(self):
        return self.covarianceParameters.get('useAverageComponent', False)

    def getClipBounds(self):
        return self.covarianceParameters.get('clipBounds', [8.0, 8.0])

    def getShrinkageParameters(self):
        return (self.covarianceParameters.get('shrinkType', None),
                self.covarianceParameters.get('shrinkFactor', 0.0))

    def getResampleType(self):
        return (self.covarianceParameters.get('resampleType', None),
                self.covarianceParameters.get('resampleIters', 0))

    def getSelectiveDeMeanParameters(self):
        return (self.covarianceParameters.get('selectiveDeMean', False),
                self.covarianceParameters.get('deMeanFactorTypes', None),
                self.covarianceParameters.get('deMeanHalfLife', None),
                self.covarianceParameters.get('deMeanMinHistoryLength', None),
                self.covarianceParameters.get('deMeanMaxHistoryLength', None))

def defaultFundamentalCovarianceParameters(
        rm,                     # The risk model instance
        modelHorizon='medium',  # medium or short horizon (125 or 60-day half-life)
        nwLag=1,                # Number of Newey-West lags for the factor covariance matrix
        dva='spline',           # Type of DVA used
        overrider=False,        # RMM flag
        unboundedDVA=False,     # Remove bounds in DVA scaling if True
        varDVAOnly=False,       # Apply DVA to variances but not covariances
        clipBounds=[8.0, 8.0],  # Bounds for specific return clipping
        useTransform=False,
        selectiveDeMean=True,
        dwe=True,
        ):

    # Fundamental model setup
    # Default medium horizon parameters
    if modelHorizon == 'medium':
        varParameters = {
                'halfLife': 125,
                'NWLag': nwLag
                }
        corrParameters = {
                'halfLife': 250,
                'NWLag': nwLag}
        srParameters = {
                'halfLife': 125,
                'NWLag': 1}

    # Default short horizon parameters
    elif modelHorizon == 'short':
        varParameters = {
                'halfLife': 60,
                'NWLag': nwLag
                }
        corrParameters = {
                'halfLife': 125,
                'NWLag': nwLag
                }
        srParameters = {
                'halfLife': 60,
                'NWLag': 1}

    # Specific risk model parameters
    srParameters['maxObs'] = int(srParameters['halfLife'] * 2)
    srParameters['clipBounds'] = clipBounds

    # DVA setup parameters
    if dva is not None:
        # For variances
        varParameters['DVAWindow'] = varParameters['halfLife']
        varParameters['DVAType'] = dva
        varParameters['downweightEnds'] = dwe
        if unboundedDVA:
            varParameters['DVAUpperBound'] = None
            varParameters['DVALowerBound'] = None

        # For covariances
        corrParameters['DVAWindow'] = varParameters['halfLife']
        corrParameters['downweightEnds'] = dwe
        if varDVAOnly:
            corrParameters['DVAType'] = None
        else:
            corrParameters['DVAType'] = dva
        if unboundedDVA:
            corrParameters['DVAUpperBound'] = None
            corrParameters['DVALowerBound'] = None

    # Demeaning parameters
    varParameters['selectiveDeMean'] = True
    varParameters['deMeanFactorTypes'] = [ExposureMatrix.StyleFactor, 
                                          ExposureMatrix.MacroCoreFactor, 
                                          ExposureMatrix.MacroMarketTradedFactor,
                                          ExposureMatrix.MacroEquityFactor,
                                          ExposureMatrix.MacroSectorFactor]
    varParameters['deMeanHalfLife'] = 1000
    varParameters['deMeanMinHistoryLength'] = varParameters['deMeanHalfLife']
    varParameters['deMeanMaxHistoryLength'] = 2*varParameters['deMeanHalfLife']

    fullCovParameters = corrParameters
    fullCovParameters['useTransformMatrix'] = useTransform
    if overrider:
        overrider.overrideCovarianceParams(varParameters, corrParameters, srParameters, fullCovParameters)

    rm.vp = CovarianceParameters2012(varParameters)
    rm.cp = CovarianceParameters2012(corrParameters)
    rm.fp = CovarianceParameters2012(fullCovParameters)
    rm.sp = CovarianceParameters2012(srParameters)
    rm.covarianceCalculator = RiskCalculator.CompositeCovarianceMatrix2012(rm.fp, rm.vp, rm.cp)
    rm.specificRiskCalculator = RiskCalculator.ComputeSpecificRisk2012(rm.sp)

def defaultStatisticalCovarianceParameters(
        rm,                     # The risk model instance
        modelHorizon='medium',  # medium or short horizon (125 or 60-day half-life)
        nwLag=1,                # Number of Newey-West lags for the factor covariance matrix
        dva='spline',           # Type of DVA used
        overrider=False,        # RMM flag
        unboundedDVA=False,     # Remove bounds in DVA scaling if True
        varDVAOnly=False,       # Apply DVA to variances but not covariances
        longHistory=True,       # Use long history of returns to match fundamental models
        ):

    # Stat model setup

    # Default medium horizon parameters
    if modelHorizon == 'medium':
        varParameters = {
                'halfLife': 125,
                'NWLag': nwLag
                }
        corrParameters = {
                'halfLife': 250,
                'NWLag': nwLag
                }
        srParameters = {
                'halfLife': 125,
                'NWLag': 1
                }

    # Default short-horizon parameters
    elif modelHorizon == 'short':
        varParameters = {
                'halfLife': 60,
                'NWLag': nwLag
                }
        corrParameters = {
                'halfLife': 125,
                'NWLag': nwLag
                }
        srParameters = {
                'halfLife': 60,
                'NWLag': 1
                }

    # Specific risk model parameters
    srParameters['maxObs'] = int(srParameters['halfLife'] * 2)

    if not longHistory:
        varParameters['maxObs'] = rm.pcaHistory
        varParameters['minObs'] = rm.pcaHistory
        corrParameters['maxObs'] = rm.pcaHistory
        corrParameters['minObs'] = rm.pcaHistory
        srParameters['maxObs'] = srParameters['halfLife']
        srParameters['minObs'] = srParameters['halfLife']

    # DVA setup
    if dva is not None:
        varParameters['DVAWindow'] = varParameters['halfLife']
        varParameters['DVAType'] = dva
        if unboundedDVA:
            varParameters['DVAUpperBound'] = None
            varParameters['DVALowerBound'] = None
        corrParameters['DVAWindow'] = varParameters['halfLife']
        if varDVAOnly:
            corrParameters['DVAType'] = None
        else:
            corrParameters['DVAType'] = dva
        if unboundedDVA:
            corrParameters['DVAUpperBound'] = None
            corrParameters['DVALowerBound'] = None

    fullCovParameters = corrParameters
    if overrider:
        overrider.overrideCovarianceParams(varParameters, corrParameters, srParameters, fullCovParameters)

    rm.vp = CovarianceParameters2012(varParameters)
    rm.cp = CovarianceParameters2012(corrParameters)
    rm.fp = CovarianceParameters2012(fullCovParameters)
    rm.sp = CovarianceParameters2012(srParameters)
    rm.covarianceCalculator = RiskCalculator.CompositeCovarianceMatrix2012(rm.fp, rm.vp, rm.cp)
    rm.specificRiskCalculator = RiskCalculator.ComputeSpecificRisk2012(rm.sp)

# Defunct test parameter setup
def defaultFundamentalCovarianceParametersV3(
        rm,                     # The risk model instance
        modelHorizon='medium',  # medium or short horizon (125 or 60-day half-life)
        nwLag=1,                # Number of Newey-West lags for the factor covariance matrix
        dva='spline',           # Type of DVA used
        overrider=False,        # RMM flag
        unboundedDVA=False,     # Remove bounds in DVA scaling if True
        clipBounds=[8.0, 8.0],  # Bounds for specific return clipping
        useTransform=True,
        varDVAOnly=False,
        #shrink=None,
        #shrinkFactor=0.25,
        #resample=None,
        #resampleIters=0,
        ):

    # Fundamental model setup
    # Default medium horizon parameters
    if modelHorizon == 'medium':
        varParameters = {
                'halfLife': 125,
                'maxObs': 500,
                'NWLag': nwLag,
                'DVAWindow': 125,
                'DVAType': dva,
                'downweightEnds': True,
                }
        corrParameters = {
                'halfLife': 250,
                'NWLag': nwLag,
                'maxObs': 1000,
                'DVAWindow': 125,
                'DVAType': dva,
                'downweightEnds': True,
                }
        srParameters = {
                'halfLife': 125,
                'maxObs': 250,
                'NWLag': 1,
                'DVAType': None,
                }

    # Default short horizon parameters
    elif modelHorizon == 'short':
        varParameters = {
                'halfLife': 60,
                'maxObs': 500,
                'NWLag': nwLag,
                'DVAWindow': 60,
                'DVAType': dva,
                'downweightEnds': False,
                }
        corrParameters = {
                'halfLife': 125,
                'maxObs': 1000,
                'NWLag': nwLag,
                'DVAWindow': 60,
                'DVAType': dva,
                'downweightEnds': False,
                }
        srParameters = {
                'halfLife': 60,
                'maxObs': 250,
                'NWLag': 1,
                'DVAType': None,
                }

    elif modelHorizon == 'stable':
        varParameters = {
                'halfLife': 125,
                'maxObs': 2500,
                'NWLag': nwLag,
                'DVAUpperBound': 0.005,
                'DVALowerBound': -0.25,
                'DVAWindow': 125,
                'DVAType': dva,
                'downweightEnds': True,
                }
        corrParameters = {
                'halfLife': 500,
                'maxObs': 2500,
                'NWLag': nwLag,
                'DVAWindow': 250,
                'DVAUpperBound': 0.005,
                'DVALowerBound': -0.25,
                'DVAType': dva,
                'downweightEnds': True,
                }
        srParameters = {
                'halfLife': 125,
                'maxObs': 250,
                'NWLag': 1,
                'DVAType': None,
                }

    srParameters['clipBounds'] = clipBounds

    # DVA setup parameters
    if unboundedDVA:
        varParameters['DVAUpperBound'] = None
        varParameters['DVALowerBound'] = None
        corrParameters['DVAUpperBound'] = None
        corrParameters['DVALowerBound'] = None

    # For covariances
    if varDVAOnly:
        corrParameters['DVAType'] = None

    # Demeaning parameters
    varParameters['selectiveDeMean'] = True
    varParameters['deMeanFactorTypes'] = [ExposureMatrix.StyleFactor, 
                                          ExposureMatrix.MacroCoreFactor, 
                                          ExposureMatrix.MacroMarketTradedFactor,
                                          ExposureMatrix.MacroEquityFactor,
                                          ExposureMatrix.MacroSectorFactor]
    varParameters['deMeanHalfLife'] = 1000
    varParameters['deMeanMinHistoryLength'] = varParameters['deMeanHalfLife']
    varParameters['deMeanMaxHistoryLength'] = 2*varParameters['deMeanHalfLife']

    # Perhaps revisit shrinkage/resampling later
    #if shrink is not None:
    #    corrParameters['shrinkType'] = shrink
    #    corrParameters['shrinkFactor'] = shrinkFactor

    #if resample is not None:
    #    corrParameters['resampleType'] = resample
    #    corrParameters['resampleIters'] = resampleIters

    fullCovParameters = corrParameters
    fullCovParameters['useTransformMatrix'] = useTransform
    if overrider:
        overrider.overrideCovarianceParams(varParameters, corrParameters, srParameters, fullCovParameters)

    rm.vp = CovarianceParameters2012(varParameters)
    rm.cp = CovarianceParameters2012(corrParameters)
    rm.fp = CovarianceParameters2012(fullCovParameters)
    rm.sp = CovarianceParameters2012(srParameters)
    rm.covarianceCalculator = RiskCalculator.CompositeCovarianceMatrix2012(rm.fp, rm.vp, rm.cp)
    rm.specificRiskCalculator = RiskCalculator.ComputeSpecificRisk2015(rm.sp)

# Defunct test parameter setup
def defaultStatisticalCovarianceParametersV3(
        rm,                     # The risk model instance
        modelHorizon='medium',  # medium or short horizon (125 or 60-day half-life)
        nwLag=1,                # Number of Newey-West lags for the factor covariance matrix
        dva='spline',           # Type of DVA used
        overrider=False,        # RMM flag
        shrinkFactor=None,      # Force factor correlations to be zero
        unboundedDVA=False,     # Remove bounds in DVA scaling if True
        varDVAOnly=False,       # Apply DVA to variances but not covariances
        ):

    # Stat model setup

    # Default medium horizon parameters
    if modelHorizon == 'medium':
        varParameters = {
                'halfLife': 125,
                'NWLag': nwLag,
                'maxObs': 500,
                'DVAWindow': 125,
                'DVAType': dva,
                'downweightEnds': True,
                }
        corrParameters = {
                'halfLife': 250,
                'NWLag': nwLag,
                'maxObs': 1000,
                'DVAWindow': 125,
                'DVAType': dva,
                'downweightEnds': True,
                }
        srParameters = {
                'halfLife': 125,
                'NWLag': 1,
                'maxObs': 250,
                'DVAType': None,
                }

    # Default short-horizon parameters
    elif modelHorizon == 'short':
        varParameters = {
                'halfLife': 60,
                'NWLag': nwLag,
                'maxObs': 500,
                'DVAWindow': 60,
                'DVAType': dva,
                'downweightEnds': True,
                }
        corrParameters = {
                'halfLife': 125,
                'NWLag': nwLag,
                'maxObs': 1000,
                'DVAWindow': 60,
                'DVAType': dva,
                'downweightEnds': True,
                }
        srParameters = {
                'halfLife': 60,
                'NWLag': 1,
                'maxObs': 250,
                'DVAType': None,
                }

    if shrinkFactor is not None:
        corrParameters['shrinkType'] = 'simple'
        corrParameters['shrinkFactor'] = shrinkFactor

    fullCovParameters = corrParameters
    if overrider:
        overrider.overrideCovarianceParams(varParameters, corrParameters, srParameters, fullCovParameters)

    rm.vp = CovarianceParameters2012(varParameters)
    rm.cp = CovarianceParameters2012(corrParameters)
    rm.fp = CovarianceParameters2012(fullCovParameters)
    rm.sp = CovarianceParameters2012(srParameters)
    rm.covarianceCalculator = RiskCalculator.CompositeCovarianceMatrix2012(rm.fp, rm.vp, rm.cp)
    rm.specificRiskCalculator = RiskCalculator.ComputeSpecificRisk2015(rm.sp)
