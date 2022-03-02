#!/bin/sh
jsonFile=$1
jobid=$2
prependPythonPath=$3

#export LD_LIBRARY_PATH=/usr/lib/oracle/12.2/client64/lib 
# to avoid cx_Oracle.DatabaseError: DPI-1050: Oracle Client library is at version 10.2 but version 11.2 or higher is needed

# used for jobcontrol runs
# we need .sh file to call runJsonCommands.py because we want to set PYTHONPATH=/home/ops-rm/contentcentral3/projects:/home/ops-rm/contentcentral3/applications

cmd="PYTHONPATH=$prependPythonPath:/home/ops-rm/global3/scripts:/home/ops-rm/contentcentral3/projects:/home/ops-rm/contentcentral3/applications python3 runJsonCommands.py $jsonFile $jobid --log-config=log.config"

echo $cmd

export TNS_ADMIN=/etc/oracle
export LD_LIBRARY_PATH=/usr/lib/oracle/12.2/client64/lib 

PYTHONPATH=$prependPythonPath:/home/ops-rm/global3/scripts:/home/ops-rm/contentcentral3/projects:/home/ops-rm/contentcentral3/applications python3 runJsonCommands.py $jsonFile $jobid --log-config=log.config
status=$? 
echo exit status $status
exit $status



