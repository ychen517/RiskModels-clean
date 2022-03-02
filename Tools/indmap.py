

import datetime

from marketdb import MarketDB
from riskmodels import Classification
from riskmodels import ModelDB

modelDB = ModelDB.ModelDB(sid='researchoda', user='modeldb_global', passwd='modeldb_global')
marketDB = MarketDB.MarketDB(sid='researchoda', user='marketdb_global', passwd='marketdb_global')

cls = Classification.GICSCustomGB4(datetime.date(2018,9,29))
cls.createMdlClassification(42,modelDB,marketDB)

marketDB.commitChanges()
modelDB.commitChanges()
marketDB.finalize()
modelDB.finalize()
