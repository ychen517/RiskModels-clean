
import logging
from riskmodels import LegacyUtilities as Utilities

class ModelHacker:
    """Make the whole pre- and post-regen stuff more flexible and also
    gather all of the model idiosyncracies into one place so we can 
    better track them, and in time, fix them
    """
    def __init__(self, modelClass):
        from riskmodels import RiskModels_V3
        # Default values: what we would wish every model to take
        self.verbose = modelClass.debuggingReporting
        self.useLegacyROEDefinition(False)
        self.useLegacyETPDefinition(False)
        self.useLegacyLiquidityDefinition(False)
        self.useWideExposureMADBounds(True)
        self.useESTUBasedSpecificReturnProxy(True)
        self.useESTUGrandfathering(True)
        self.useRealStatModelSpecificReturns(True)
        self.useMADClippedReturnsForStyles(True)
        self.useClipNotMaskExtremeReturns(True)
        self.useMADClippedFactorReturns(True)
        self.useMADClippedExchangeRateReturns(True)
        self.useDRHandling(True)
        self.useDRClipping(False)
        self.useCapWeightedExposureNormalization(True)
        self.useEnhancedESTUFiltering(True)
        self.useMissingIndustryCheck(True)
        self.useMarketSensitivityFactor(False)
        self.useTotalIssuerMarketCaps(True)
        self.useNonCheatHistoricBetas(True)
        self.useDRCLoningAndCointegration(True)
        self.useWeeklyXRTFactor(True)
        self.useFullProxyFill(False)
        self.useSCMMultiDescriptorsFundamentalExposure(False)
        self.useXRTFactorFixDate(False)
        self.useAHSharesISC(True)
        defaultParameters = True
        
        # Now comes the idiosyncracies. We hope over time
        # that this bit will shrink to nothing
        
        # US model differences
        if modelClass in (
                             RiskModels_V3.USAxioma2009MH,
                             RiskModels_V3.USAxioma2009SH,
                             RiskModels_V3.USAxioma2009MH_S,
                             RiskModels_V3.USAxioma2009SH_S
                            ):
            self.useLegacyROEDefinition(True)
            self.useLegacyLiquidityDefinition(True)
            self.useClipNotMaskExtremeReturns(False)
            self.useESTUBasedSpecificReturnProxy(False)
            self.useMADClippedExchangeRateReturns(False)
            self.useCapWeightedExposureNormalization(False)
            self.useMarketSensitivityFactor(True)
            self.useTotalIssuerMarketCaps(False)
            self.useNonCheatHistoricBetas(False)
            self.useWeeklyXRTFactor(False)
            defaultParameters = False
        
        # Canada, Australia, Japan, GB single country models
        if modelClass in (
                           RiskModels_V3.CAAxioma2009MH,
                           RiskModels_V3.CAAxioma2009SH,
                           RiskModels_V3.AUAxioma2009MH,
                           RiskModels_V3.AUAxioma2009SH,
                           RiskModels_V3.JPAxioma2009MH,
                           RiskModels_V3.JPAxioma2009SH,
                           RiskModels_V3.GBAxioma2009MH,
                           RiskModels_V3.GBAxioma2009SH,
                          ):
            self.useCapWeightedExposureNormalization(False)
            self.useMarketSensitivityFactor(True)
            self.useTotalIssuerMarketCaps(False)
            self.useNonCheatHistoricBetas(False)
            self.useWeeklyXRTFactor(False)
            defaultParameters = False

        if modelClass in (
                           RiskModels_V3.CAAxioma2009MH_S,
                           RiskModels_V3.CAAxioma2009SH_S,
                           RiskModels_V3.AUAxioma2009MH_S,
                           RiskModels_V3.AUAxioma2009SH_S,
                           RiskModels_V3.JPAxioma2009MH_S,
                           RiskModels_V3.JPAxioma2009SH_S,
                           RiskModels_V3.GBAxioma2009MH_S,
                           RiskModels_V3.GBAxioma2009SH_S,
                          ):
            self.useNonCheatHistoricBetas(False)
            defaultParameters = False   

        # CN still uses the old betas
        if modelClass in (
                           RiskModels_V3.CNAxioma2010MH,
                           RiskModels_V3.CNAxioma2010SH,
                           RiskModels_V3.CNAxioma2010MH_S,
                           RiskModels_V3.CNAxioma2010SH_S,
                          ):
            self.useNonCheatHistoricBetas(False)
            self.useWeeklyXRTFactor(False)
            self.useDRCLoningAndCointegration(False)
            defaultParameters = False
        
        if modelClass in (
                           RiskModels_V3.TWAxioma2012MH,
                           RiskModels_V3.TWAxioma2012MH_S,
                           RiskModels_V3.TWAxioma2012SH,
                           RiskModels_V3.TWAxioma2012SH_S):
            self.useSCMMultiDescriptorsFundamentalExposure(True)
            self.useXRTFactorFixDate(True)
            defaultParameters = False
        
        if modelClass in (
            RiskModels_V3.APxJPAxioma2013MH,
            RiskModels_V3.APxJPAxioma2013MH_S,
            RiskModels_V3.APxJPAxioma2013SH,
            RiskModels_V3.APxJPAxioma2013SH_S,
            RiskModels_V3.APAxioma2013MH,
            RiskModels_V3.APAxioma2013MH_S,
            RiskModels_V3.APAxioma2013SH,
            RiskModels_V3.APAxioma2013SH_S,
            RiskModels_V3.WWAxioma2013MH,
            RiskModels_V3.WWAxioma2013MH_S,
            RiskModels_V3.WWAxioma2013SH,
            RiskModels_V3.WWAxioma2013SH_S,
            ):
            self.useAHSharesISC(False)
            defaultParameters = False
            
        if defaultParameters:
            if self.verbose:
                logging.info('...All model settings default - no hacks!')

    def useLegacyROEDefinition(self, b):
        """Non legacy ROEs are set to zero if negative.
        """
        self.legacyROE = b
        if self.verbose:
            logging.info('...Legacy return-on-equity definition: %s', b)

    def useLegacyETPDefinition(self, b):
        """Legacy ETP uses EBITDA instead of net income.
        """
        self.legacyETP = b
        if self.verbose:
            logging.info('...Legacy earnings-to-price definition: %s', b)

    def useLegacyLiquidityDefinition(self, b):
        """Legacy trading volume is the screwy log(vol)/log(cap).
        """
        self.legacyTradingVolume = b
        if self.verbose:
            logging.info('...Legacy Liquidity factor definition: %s', b)

    def useWideExposureMADBounds(self, b):
        """Newer models use 8.0 MADs for standardisation as 
        opposed to 5.2.
        """
        self.widerStandardizationMADBound = b
        if self.verbose:
            logging.info('...Wide exposure MAD bounds: %s', b)

    def useESTUBasedSpecificReturnProxy(self, b):
        """Newer models restrict specific return fill-in to
        the estimation universe when computing the relevant parameters.
        """
        self.specRiskEstuFillIn = b
        if self.verbose:
            logging.info('...Specific return fill-in buckets restricted to ESTU: %s', b)

    def useESTUGrandfathering(self, b):
        """Applies grandfathering to estimation universe generation
        (regional models only)
        """
        self.grandfatherEstu = b
        if self.verbose:
            logging.info('...ESTU grand-fathering: %s', b)

    def useRealStatModelSpecificReturns(self, b):
        """Older stat models use incorrect specific return.
        """
        self.statModelCorrectSpecRet = b
        if self.verbose:
            logging.info('..."Real" specific returns in statistical models: %s', b)

    def useMADClippedReturnsForStyles(self, b):
        """Legacy regional models clip returns at fixed bounds.
        """
        self.MADRetsForStyles = b

    def useClipNotMaskExtremeReturns(self, b):
        if b:
            self.MADRetsTreat = 'clip'
        else:
            self.MADRetsTreat = 'mask'
        if self.verbose:
            logging.info('...Treatment of extreme return values after MAD: %s', self.MADRetsTreat)

    def useMADClippedFactorReturns(self, b):
        """Legacy models used a more primitive clipping of factor returns.
        """
        self.MADFactorReturns = b
        if self.verbose:
            logging.info('...MAD-clipping of factor returns: %s', b)

    def useMADClippedExchangeRateReturns(self, b):
        """Older models don't clip fx returns for fxsens factor.
        """
        self.fxSensReturnsClip = b
        if self.verbose:
            logging.info('...MAD-clipping of FX returns in Exchange Rate Sensitivity: %s', b)

    def useDRHandling(self, b):
        """Enable special modeling of DR-like instruments.
        """
        self.DRHandling = b
        if self.verbose:
            logging.info('...Special modeling of DR-like instruments: %s', b)
    
    def useDRClipping(self, b):
        """Extra clipping for DR returns
        """
        self.DRClipping = b
        if self.verbose:
            logging.info('...Extra clipping of DR returns: %s' % b)

    def useCapWeightedExposureNormalization(self, b):
        """Cap versus root-cap standardization.
        """
        self.capMean = b
        if self.verbose:
            logging.info('...Standardize exposures around cap-weighted mean: %s', b)

    def useEnhancedESTUFiltering(self, b):
        """Improved automatic ESTU exclusion logic based on 
        exchange and asset type.
        """
        self.enhancedESTUFiltering = b
        if self.verbose:
            logging.info('...Enhanced ESTU filter rules: %s', b)

    def useMissingIndustryCheck(self, b):
        """Drop assets without industry classification from model universe.
        """
        self.checkForMissingIndustry = b
        if self.verbose:
            logging.info('...Missing industry check: %s', b)

    def useMarketSensitivityFactor(self, b):
        """Enables the Market Sensitivity (normalized beta) style factor
        """
        self.hasMarketSensitivity = b
        if self.verbose:
            logging.info('...Market Sensitivity factor: %s', b)

    def useNonCheatHistoricBetas(self, b):
        """Extract with historic betas computed against the proper market return history.
        """
        self.nonCheatHistoricBetas = b
        if self.verbose:
            logging.info('...Historic Betas against proper market return: %s', b)
    
    def useTotalIssuerMarketCaps(self, b):
        """Use issuer-level market caps for fundamental style factors
        and size factor.
        """
        self.issuerMarketCaps = b
        if self.verbose:
            logging.info('...Issuer caps for fundamental and Size factors: %s', b)

    def useWeeklyXRTFactor(self, b):
        """Use XRT factor derived from weekly data and a market intercept
        """
        self.weeklyXRTFactor = b
        if self.verbose:
            logging.info('...XRT Factor uses weekly returns: %s', b)

    def useDRCLoningAndCointegration(self, b):
        """Clone cross-listing exposures and use cointegration for ISC
        """
        self.specialDRTreatment = b
        if self.verbose:
            logging.info('...Special Treatment for Cross-Listings: %s', b)

    def useFullProxyFill(self, b):
        """"Enable stat models to use proxy fill-in for all missing returns,
        not just IPOs and NTDs
        """
        self.fullProxyFill = b
        if self.verbose:
            logging.info('...Using proxy fill in for all missing returns: %s', b)

    def useSCMMultiDescriptorsFundamentalExposure(self, b):
        """Enable SCM models to use multiple descriptor fundamental exposures
        """
        self.scmMDFundExp = b
        if self.verbose:
            logging.info('...SCM using multiple descriptors to generate fundamental exposures: %s', b)

    def useXRTFactorFixDate(self, b):
        """Enable fix date option in generate_forex_sensitivity_v2
        """
        self.xrtFixDate = b
        if self.verbose:
            logging.info('...Fix date for Exchange Rate Sensitivity factor: %s', b)

    def useAHSharesISC(self, b):
        """Enable ISC approach on A/H shares pairs
        """
        self.ahSharesISC = b
        if self.verbose:
            logging.info('...Apply ISC to Hong Kong H and China A Shares: %s', b)
