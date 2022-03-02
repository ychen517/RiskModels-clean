
import logging
import pymongo
import traceback
from riskmodels import ModelDB
from marketdb import MarketDB
from marketdb import Connections as MktConnections
from marketdb.Connections import revertConnections, finalizeConnections#these imports are used from calling scripts (transfer.py)

def createConnections(config):
    return Connections(config)

def createMongoDB(config, section):
    if config.has_section(section):
        try:
            logging.debug('Creating %s connection',section)
            client = pymongo.MongoClient(host=config.get(section, 'host'))
            return client[config.get(section, 'dbname')]
        except:
            logging.error('Unable to connect to MongoDB')
    else:
        logging.error('Missing %s section in configuration file',section)
        

class Connections(MktConnections.Connections):#effectivly inheriting only methods here, since the properties(connections) are coordinated with the config file's connections
    def __init__(self, config):
        super(Connections, self).__init__(config)
        self._mongoDB = None       
        
    @property
    def mongoDB(self):
        if self._mongoDB is None:      
            self._mongoDB = createMongoDB(self.config, 'MongoDB')
        return self._mongoDB    
    
    @mongoDB.setter
    def mongoDB(self, value):
        self._mongoDB = value
               


     
