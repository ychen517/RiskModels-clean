#!/bin/bash
if [ "$MASTER" = "" ]; then
    export MASTER=obelix
fi
# Set up variables
export DIR=$1
MD=$2
export S1="-llog.config --marketdb-sid=research --marketdb-user=marketdb_global"
export S2="--marketdb-passwd=marketdb_global --modeldb-sid=research"
export LAZY_GL="$S1 $S2 --modeldb-user=modeldb_global --modeldb-passwd=modeldb_global"

if [ "$3" = "" ]; then
    export exp_year=1991
else
    export exp_year=$3
fi
if [ "$4" = "" ]; then
    export end=2011
else
    export end=$4
fi
rm -f tmp.sh
touch tmp.sh

export runit="python3 $SCHED/Client.py --work-dir=$DIR --master=$MASTER"

# Estimation universe loop
export year=$exp_year
export p_year=$year
export prog="python3 generateStatisticalModel.py $LAZY_GL -m$MD --force"
echo "####################################################################################################################" >> tmp.sh
echo "# ESTU" >> tmp.sh
while [ $year -le $end ]
do
    export group=${MD}_estu_$year
    export depend=""
    echo "$runit --cmd='$prog -m$MD --estu $year-01-01 $year-12-31' --group=$group run" >> tmp.sh

    export group=${MD}_part_estu_$year
    export depend="${MD}_estu_$year,${MD}_estu_$p_year"
    echo "$runit --cmd='$prog -m$MD --estu $year-01-01 $year-04-30' --group=$group --depends-on=$depend run" >> tmp.sh
    export p_year=$year

    year=$(($year+1))
done

# Risk loop
export year=$(($exp_year+2))
export p_year=$year
export pp_year=$year
export ppp_year=$year
export pppp_year=$year
echo "####################################################################################################################" >> tmp.sh
echo "# RSK" >> tmp.sh
while [ $year -le $end ]
do
    export group=${MD}_rsk_$year
    export depend="${MD}_ret_$year,${MD}_ret_$p_year,${MD}_ret_$pp_year,${MD}_ret_$ppp_year,${MD}_ret_$pppp_year"
    echo "$runit --cmd='$prog -m$MD --model $year-01-01 $year-12-31' --group=$group --depends-on=$depend run" >> tmp.sh

    export pppp_year=$ppp_year
    export ppp_year=$pp_year
    export pp_year=$p_year
    export p_year=$year
    year=$(($year+1))
done

sh tmp.sh
