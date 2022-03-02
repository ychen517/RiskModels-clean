#!/bin/bash
# arguments are passed in without any modification
export AXIOMA_LICENSE_FILE="@LICENSE-SERVER"
python3 generateConsolidateRiskModelQA.py $*
