import datetime
import logging
from riskmodels import Classification
from riskmodels import EquityModel
from riskmodels.Factors import ModelFactor
from riskmodels import RiskCalculator_V4
from riskmodels import RiskModels
from riskmodels import ModelParameters2017

class GBResearchModel1(RiskModels.UKAxioma2018MH):
    """Version 4 GB medium-horizon fundamental model with GICS industries
    """
    rm_id = -3
    revision = 1
    rms_id = -4

    gicsDate = datetime.date(2018,9,29)
    industryClassification = Classification.GICSCustomNoMortgageREITs2018(gicsDate)
    intercept = None

    styleList = ['Value',
                 'Leverage',
                 'Growth',
                 'Profitability',
                 'Earnings Yield',
                 'Dividend Yield',
                 'Size',
                 'MidCap',
                 'Liquidity',
                 'Market Sensitivity',
                 'Volatility',
                 'Medium-Term Momentum',
                 'Exchange Rate Sensitivity',
                 'Investment Trusts',
                 ]

    DescriptorMap = {
            'Earnings Yield': ['Earnings_to_Price_Annual', 'Est_Earnings_to_Price_12MFL_Annual'],
            'Value': ['Book_to_Price_Annual'],
            'Leverage': ['Debt_to_Assets_Annual', 'Debt_to_Equity_Annual'],
            'Growth': ['Earnings_Growth_RPF_Annual', 'Sales_Growth_RPF_Annual'],
            'Dividend Yield': ['Dividend_Yield_Annual'],
            'Size': ['LnIssuerCap'],
            'Liquidity': ['LnTrading_Activity_60D', 'Amihud_Liquidity_125D', 'ISC_Ret_Score'],
            'Market Sensitivity': ['Market_Sensitivity_104W'],
            'Volatility': ['UK_Volatility_125D'],
            'Medium-Term Momentum': ['Momentum_260x21D_Regional'],
            'Exchange Rate Sensitivity': ['XRate_104W_XDR'],
            'Profitability': ['Return_on_Equity_Annual', 'Return_on_Assets_Annual',
                'CashFlow_to_Assets_Annual', 'CashFlow_to_Income_Annual',
                'Sales_to_Assets_Annual', 'Gross_Margin_Annual'],
            }

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('GBResearchModel1')
        RiskModels.UKAxioma2018MH.__init__(self, modelDB, marketDB)

    def setCalculators(self, modelDB, overrider=False):
        # Set up internal factor return regression parameters
        dummyThreshold = 6
        self.internalCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType='Industry Groups',
                dummyThreshold=dummyThreshold,
                marketRegression=False,
                kappa=5.0,
                useRealMCaps=True,
                regWeight='rootCap',
                thinWeightMultiplier='simple',
                overrider = overrider,
                )

        # Set up external regression parameters
        self.returnCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType='Industry Groups',
                dummyThreshold = dummyThreshold,
                marketRegression=False,
                kappa=25.0,
                useRealMCaps=True,
                regWeight='rootCap',
                thinWeightMultiplier='simple',
                overrider = overrider,
                )

        # This controls the FMP regression
        self.fmpCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType=None,
                dummyThreshold = dummyThreshold,
                marketRegression=False,
                kappa=None,
                useRealMCaps=True,
                regWeight='rootCap',
                overrider = overrider,
                )

        # And this is for raw exposure proxying
        self.expProxyCalculator = ModelParameters2017.defaultExposureProxyParameters(
                self, modelDB,
                dummyType='market',
                dummyThreshold = dummyThreshold,
                kappa=5.0,
                useRealMCaps=True,
                regWeight='rootCap',
                thinWeightMultiplier='simple',
                overrider = overrider,
                )

        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=3, overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=3, overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, overrider=overrider)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

class GBResearchModel2(GBResearchModel1):
    """Version 4 GB medium-horizon fundamental model with GICS industries
    """
    rm_id = -3
    revision = 2
    rms_id = -5

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('GBResearchModel2')
        GBResearchModel1.__init__(self, modelDB, marketDB)

    def setCalculators(self, modelDB, overrider=False):
        # Set up internal factor return regression parameters
        dummyThreshold = 6
        self.internalCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType=None,
                dummyThreshold=dummyThreshold,
                marketRegression=False,
                kappa=None,
                useRealMCaps=True,
                regWeight='rootCap',
                thinWeightMultiplier='simple',
                overrider = overrider,
                )

        # Set up external regression parameters
        self.returnCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType=None,
                dummyThreshold = dummyThreshold,
                marketRegression=False,
                kappa=None,
                useRealMCaps=True,
                regWeight='rootCap',
                thinWeightMultiplier='simple',
                overrider = overrider,
                )

        # This controls the FMP regression
        self.fmpCalculator = ModelParameters2017.defaultRegressionParameters(
                self, modelDB,
                dummyType=None,
                dummyThreshold = dummyThreshold,
                marketRegression=False,
                kappa=None,
                useRealMCaps=True,
                regWeight='rootCap',
                overrider = overrider,
                )

        # And this is for raw exposure proxying
        self.expProxyCalculator = ModelParameters2017.defaultExposureProxyParameters(
                self, modelDB,
                dummyType='market',
                dummyThreshold = dummyThreshold,
                kappa=5.0,
                useRealMCaps=True,
                regWeight='rootCap',
                thinWeightMultiplier='simple',
                overrider = overrider,
                )

        # Set up risk parameters
        ModelParameters2017.defaultFactorVarianceParameters(self, nwLag=3, overrider=overrider)
        ModelParameters2017.defaultFactorCorrelationParameters(self, nwLag=3, overrider=overrider)
        ModelParameters2017.defaultSpecificVarianceParameters(self, overrider=overrider)
        self.covarianceCalculator = RiskCalculator_V4.CompositeCovarianceMatrix2017(self.fvParameters, self.fcParameters)

class GBResearchModel3(GBResearchModel1):
    """Version 4 GB medium-horizon fundamental model with GICS industries
    """
    rm_id = -3
    revision = 3
    rms_id = -6
    intercept = ModelFactor('Market Intercept', 'Market Intercept')

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('GBResearchModel3')
        GBResearchModel1.__init__(self, modelDB, marketDB)

class GBResearchModel4(RiskModels.UKAxioma2018MH_S):
    """Version 4 GB medium-horizon enhanced statistical model
    """
    rm_id = -3
    revision = 4
    rms_id = -8

    numGlobalFactors = 15
    sectorFactorMap = {
            'Consumer Discretionary': 5,
            'Consumer Staples': 5,
            'Energy': 5,
            'Financials': 5,
            'Health Care': 5,
            'Industrials': 5,
            'Information Technology': 5,
            'Materials': 5,
            'Real Estate': 5,
            'Communication Services': 5,
            'Utilities': 5}
    numFactors = numGlobalFactors + sum(sectorFactorMap.values())
    blind = [ModelFactor('Statistical Factor %d' % n, 'Statistical Factor %d' % n) for n in range(1, numFactors+1)]

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('GBResearchModel4')
        RiskModels.UKAxioma2018MH_S.__init__(self, modelDB, marketDB)

