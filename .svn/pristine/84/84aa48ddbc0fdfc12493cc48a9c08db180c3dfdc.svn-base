#!/bin/bash
export MARKETDB_USER=marketdb_global
export MARKETDB_PASSWD=marketdb_global
export MODELDB_SID=research
export MARKETDB_SID=research
export MODELDB_PASSWD=modeldb_global
export MODELDB_USER=modeldb_global

CONFIG_FILE='test.config'

RM_NAME=USResearchMacroModelV7_2 

#You may need to do something like this manually in ModelDB
# Assuming the rms_id is -40701 :
#
# delete from rms_factor where rms_id=-40701 
# 
# drop table rms_M40701_factor_exposure


############################### Do this section only once ############################
# Define tables in ModelDB
#
cd Tables
PYTHONPATH=.. python3 updateTable.py ../test.config factor_type ./factor_type.txt --research-only -l ../log.config 
PYTHONPATH=.. python3 updateTable.py ../test.config risk_model ./risk_model.txt --research-only -l ../log.config 
PYTHONPATH=.. python3 updateTable.py ../test.config risk_model_serie ./risk_model_serie.txt --research-only -l ../log.config 
PYTHONPATH=.. python3 updateTable.py ../test.config rmg_model_map ./rmg_model_map.txt --research-only -l ../log.config 
PYTHONPATH=.. python3 updateTable.py ../test.config factor ./factor.txt --research-only -l ../log.config 
PYTHONPATH=.. python3 updateTable.py ../test.config sub_factor ./sub_factor.txt --research-only -l ../log.config  
PYTHONPATH=.. python3 updateTable.py ../test.config rms_factor ./rms_factor.txt --research-only -l ../log.config  

# Now create room in ModelDB.
cd ../Tools
# You may need to send the output of these scripts to someone with permission to run it.
PYTHONPATH=.. python3 createRMSPartitions.py -m ${RM_NAME} ../${CONFIG_FILE} -l ../log.config --update-database

# Re-run this if the factor structure ever changes.
# Only need special permissions the first time it is run.
PYTHONPATH=.. python3  createRMSExposureTable.py -m ${RM_NAME} ../${CONFIG_FILE} -l ../log.config --update-database

cd ..
python3 createRMSID.py -m ${RM_NAME} ./${CONFIG_FILE} -l ./log.config  all --allow-updates --allow-deletions --allow-estu-updates --allow-estu-deletions --update-database 

############################### Run###########################

#Delete the model if need 
python3 purgeRiskModel.py -m $RM_NAME -l ./log.config

#Populate asset universes.

#OBSOLETE: See runMe.sh

#Generate estimation universe (can also use --estu-fast to steal the stat models universe)
python3 generateMacroModel.py -m $RM_NAME -l log.config  --estu  1985-01-01 2012-12-31

#Generate a prehistory of factor returns 
python3 generateMacroModel.py -m $RM_NAME -l log.config --initial-factor-returns  1988-01-01 1993-01-01

#Generate factor returns history 
python3 generateMacroModel.py -m $RM_NAME -l log.config --factors  1993-01-01 2012-12-31

#Generate exposures and specific returns
#(assumes factor returns are present for previous 2000 days.)
python3 generateMacroModel.py -m $RM_NAME -l log.config  --exposures  1993-01-01 2012-12-31

#Generate factor cov, and specific risk 
python3 generateMacroModel.py -m $RM_NAME -l log.config --risks 1995-01-01 2012-12-31

#Cumulative returns 
python3 generateCachingMacroModel.py -m $RM_NAME --cum-factors --start-cumulative-return 1993-01-01 2012-12-31




