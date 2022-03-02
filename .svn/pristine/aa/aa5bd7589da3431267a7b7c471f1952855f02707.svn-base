
import inspect

# Set up list of all models in research and production
_models = []
# Add latest production models to running list
from riskmodels import RiskModels
for (name, obj) in inspect.getmembers(RiskModels):
    if inspect.isclass(obj) and hasattr(obj, 'rms_id'):
        _models.append(obj)

# Now the multitude of research models
# US research models
from riskmodels.researchmodels import ResearchModels_US
for (name, obj) in inspect.getmembers(ResearchModels_US):
    if inspect.isclass(obj) and hasattr(obj, 'rms_id'):
        _models.append(obj)
# AU research models
from riskmodels.researchmodels import ResearchModels_AU
for (name, obj) in inspect.getmembers(ResearchModels_AU):
    if inspect.isclass(obj) and hasattr(obj, 'rms_id'):
        _models.append(obj)
# JP research models
from riskmodels.researchmodels import ResearchModels_JP
for (name, obj) in inspect.getmembers(ResearchModels_JP):
    if inspect.isclass(obj) and hasattr(obj, 'rms_id'):
        _models.append(obj)
# V4 regional models
from riskmodels.researchmodels import ResearchModels_WW4
for (name, obj) in inspect.getmembers(ResearchModels_WW4):
    if inspect.isclass(obj) and hasattr(obj, 'rms_id'):
        _models.append(obj)
# EM4 research models
from riskmodels.researchmodels import ResearchModels_EM4
for (name, obj) in inspect.getmembers(ResearchModels_EM4):
    if inspect.isclass(obj) and hasattr(obj, 'rms_id'):
        _models.append(obj)
# CN4 research models
from riskmodels.researchmodels import ResearchModels_CN4
for (name, obj) in inspect.getmembers(ResearchModels_CN4):
    if inspect.isclass(obj) and hasattr(obj, 'rms_id'):
        _models.append(obj)
# CA4 research models
from riskmodels.researchmodels import ResearchModels_CA4
for (name, obj) in inspect.getmembers(ResearchModels_CA4):
    if inspect.isclass(obj) and hasattr(obj, 'rms_id'):
        _models.append(obj)
# NA4 research models
from riskmodels.researchmodels import ResearchModels_NA4
for (name, obj) in inspect.getmembers(ResearchModels_NA4):
    if inspect.isclass(obj) and hasattr(obj, 'rms_id'):
        _models.append(obj)
# Older research models
from riskmodels.researchmodels import ResearchModels_SCM
for (name, obj) in inspect.getmembers(ResearchModels_SCM):
    if inspect.isclass(obj) and hasattr(obj, 'rms_id'):
        _models.append(obj)

from riskmodels import PhoenixModels
_models.append(PhoenixModels.USFIAxioma_R)
_models.append(PhoenixModels.UNIAxioma2013MH)
_models.append(PhoenixModels.COAxioma2013MH)
from riskmodels import FixedIncomeModels
_models.append(FixedIncomeModels.FIAxioma2014MH)
_models.append(FixedIncomeModels.FIAxioma2014MH1)
_models.append(FixedIncomeModels.Univ10AxiomaMH)
_models.append(FixedIncomeModels.Univ10AxiomaMH_Pre2009)

from riskmodels import RiskModels_V3
for (name, obj) in inspect.getmembers(RiskModels_V3):
    if inspect.isclass(obj) and hasattr(obj, 'rms_id'):
        _models.append(obj)

# Legacy model parameter functions
def defaultFundamentalCovarianceParameters(
        rm, modelHorizon='medium', nwLag=1, dva='spline', overrider=False):

    return RiskModels_V3.defaultFundamentalCovarianceParameters(
            rm, modelHorizon=modelHorizon, nwLag=nwLag, dva=dva, overrider=overrider)

def defaultStatisticalCovarianceParameters(
        rm, modelHorizon='medium', nwLag=1, dva='spline',
        historyLength=250, scaleMinObs=False, longSpecificReturnHistory=True,
        overrider = False):

    return RiskModels_V3.defaultStatisticalCovarianceParameters(
            rm, modelHorizon=modelHorizon, nwLag=nwLag, dva=dva,
            historyLength=historyLength, scaleMinObs=scaleMinObs,
            longSpecificReturnHistory=longSpecificReturnHistory,
            overrider = False)
    
def defaultRegressionParameters(
        rm, modelDB, dummyType=None, dummyThreshold=10.0,
        scndRegs=None, k_rlm=1.345,
        weightedRLM=True, overrider=False):

    return RiskModels_V3.defaultRegressionParameters(
            rm, modelDB, dummyType=dummyType, dummyThreshold=dummyThreshold,
            scndRegs=scndRegs, k_rlm=k_rlm,
            weightedRLM=weightedRLM, overrider=overrider)

modelNameMap = dict([(i.__name__, i) for i in _models])
modelRevMap = dict([((i.rm_id, i.revision), i) for i in _models])
modelSerieMap = dict((i.rms_id, i) for i in _models)

def getModelByName(modelName):
    return modelNameMap[modelName]

def getModelByVersion(rm_id, revision):
    return modelRevMap[(rm_id, revision)]

def getModelBySerie(rms_id):
    return modelSerieMap[rms_id]
