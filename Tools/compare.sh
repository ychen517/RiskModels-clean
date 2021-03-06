#!/bin/sh
export rm=$1
export date=$2
export emailFile=$3
export DIRLIST=$4
export tmpdir=$5
export type=$6
export runType=$7
export DB="$STND_DB"
DIRARRAY=$(echo $DIRLIST | tr "," "\n")
export HOMEDIR=`echo $DIRARRAY | awk '{print $1}'`
export outfile=$HOMEDIR/$tmpdir/cmp_${rm}_$date
export config_file="production.config"
tran_step="python3 transfer.py $config_file AxiomaDB:sid=glprodsb ModelDB:sid=glprodsb MarketDB:sid=glprodsb"

rm -f $outfile
touch $outfile
export dt=`echo $date | sed 's/-//g'`
if [ "$type" = "transfer" ]; then
    echo MARKET: $rm >> $outfile
else
    echo MODEL: $rm >> $outfile
fi
echo " " >> $outfile

py_script(){
    export RT=$1
    python3 <<-'EOF'
import os
import pandas as pd
import numpy as np
tol=1.0e-15
if os.environ['RT']=='TS-Descriptors':
    tol=1.0e-7
df1=pd.read_csv("%s/%s/%s-%s-%s.csv" % \
    (os.environ['HOMEDIR'],os.environ['tmpdir'],os.environ['RT'],os.environ['rm'],os.environ['date']))
df2=pd.read_csv("%s/%s/%s-%s-%s.csv" % \
    (os.environ['THISDIR'],os.environ['tmpdir'],os.environ['RT'],os.environ['rm'],os.environ['date']))
correl=df1.corrwith(df2).round(decimals=6).fillna(1.0)
correl=correl[correl!=1.0]
correl.to_csv("%s/%s/%s-Correl-%s-%s.csv" % \
    (os.environ['THISDIR'],os.environ['tmpdir'],os.environ['RT'],os.environ['rm'],os.environ['date']), header=False)
diffs=abs(df1.select_dtypes(include=[np.number])-df2.select_dtypes(include=[np.number])).max(axis=0, skipna=True)
diffList=list(diffs[abs(diffs)>=tol].index)
ofile=open('%s/%s/cols.csv' % (os.environ['THISDIR'],os.environ['tmpdir']), 'w')
ofile.write('Max difference per column: \n')
for itm in diffList:
    if diffs[itm]!=0.0:
        ofile.write('...... %s,%.8f\n' % (itm, diffs[itm]))
ofile.write('Correlations different from one: \n')
ofile.close()
EOF
}

export origLnk=""
if [ "$WORK" != "" ]; then
    export origLnk=`readlink $WORK/riskmodels`
fi
export origLnk=""

##################################################################################################
### Currency models                                                                            ###
##################################################################################################

if [ "$type" = "cur" ]; then
    # Currency models
    export loc=1
    for DIR in $DIRARRAY
    do
        cd $DIR
        if [ "$runType" != "p" ]; then

            if [ "$origLnk" != "" ]; then
                rm $WORK/riskmodels
                ln -s $DIR/riskmodels $WORK/riskmodels
            fi

            # Generate currency model results
            start_time="$(date -u +%s)"
            echo "python3 generateEquityModel.py $DB --dw --risks -m$rm $date > $tmpdir/${rm}_${date}.log"
            python3 generateEquityModel.py $DB --dw --risks -m$rm $date > $tmpdir/${rm}_${date}.log
            elapsed="$((`date -u +%s`-$start_time))"

            # Parse the output
            grep -i error $tmpdir/${rm}_${date}.log >> $outfile
            echo Loc ${loc}: `cut -d' ' -f4- $tmpdir/${rm}_${date}.log | grep "Frobenius norm"` --- Time: $elapsed secs >> $outfile
            echo Loc ${loc}: `cut -d' ' -f4- $tmpdir/${rm}_${date}.log | grep "Frobenius norm"` --- Time: $elapsed secs >> $tmpdir/${rm}_${date}.log
        else
            grep "Loc 1:" $tmpdir/${rm}_${date}.log | sed -n -e "s/ 1:/ ${loc}:/p" >> $outfile
        fi
        echo $loc 'X' $DIR 'XX' $tmpdir/${rm}_${date}.log 'XXXXXX' $outfile
        loc=$(($loc+1))
    done
    echo " " >> $outfile

##################################################################################################
### Fundamental models                                                                         ###
##################################################################################################

elif [ "$type" = "fund" ]; then
    # Fundamental models now
    export loc=1
    for DIR in $DIRARRAY
    do
        cd $DIR
        if [ "$runType" != "p" ]; then

            if [ "$origLnk" != "" ]; then
                rm $WORK/riskmodels
                ln -s $DIR/riskmodels $WORK/riskmodels
            fi

            # Run estimation universe step
            start_time="$(date -u +%s)"
            python3 generateEquityModel.py $DB -f --estu --dw -m$rm $date > $tmpdir/est_${rm}_${date}.log
            elapsed="$((`date -u +%s`-$start_time))"

            # Parse the output
            grep -i error $tmpdir/est_${rm}_${date}.log >> $outfile
            echo Loc ${loc}: `cut -d' ' -f4- $tmpdir/est_${rm}_${date}.log | grep "ESTU contains"` --- Time: $elapsed secs >> $outfile
            echo Loc ${loc}: `cut -d' ' -f4- $tmpdir/est_${rm}_${date}.log | grep "ESTU contains"` --- Time: $elapsed secs >> $tmpdir/est_${rm}_${date}.log
        else
            grep "Loc 1:" $tmpdir/est_${rm}_${date}.log | sed -n -e "s/ 1:/ ${loc}:/p" >> $outfile
        fi
        loc=$(($loc+1))
    done
    echo " " >> $outfile

    # Fundamental model exposures
    export loc=1
    for DIR in $DIRARRAY
    do
        cd $DIR
        if [ "$runType" != "p" ]; then

            if [ "$origLnk" != "" ]; then
                rm $WORK/riskmodels
                ln -s $DIR/riskmodels $WORK/riskmodels
            fi

            # Run the exposure step
            rm -f $tmpdir/expM-${rm}-${date}.csv
            start_time="$(date -u +%s)"
            python3 generateEquityModel.py $DB --dw -v -f --exposures -m$rm $date > $tmpdir/exp_${rm}_${date}.log
            elapsed="$((`date -u +%s`-$start_time))"

            # Move the exposure flatfile
            touch $tmpdir/expM-${rm}-${date}.csv
            touch $tmpdir/raw-expM-${rm}-${date}.csv
            modName=`cat $tmpdir/exp_${rm}_${date}.log | sed -n -e 's/^.*Initializing risk model: //p' | awk '{print $1; exit;}'`
            mv tmp/expM-${modName}-${dt}.csv $tmpdir/expM-${rm}-${date}.csv
            mv tmp/raw-expM-${modName}-${dt}.csv $tmpdir/raw-expM-${rm}-${date}.csv

            # Parse the log output
            grep -i error $tmpdir/exp_${rm}_${date}.log >> $outfile
            echo Loc ${loc}: Exposures: Time: $elapsed secs >> $outfile
            echo Loc ${loc}: Exposures: Time: $elapsed secs >> $tmpdir/exp_${rm}_${date}.log
        else
            grep "Loc 1:" $tmpdir/exp_${rm}_${date}.log | sed -n -e "s/ 1:/ ${loc}:/p" >> $outfile
        fi
        if [ ! -s $tmpdir/expM-${rm}-${date}.csv ]; then
            echo "ERROR: no exposure file for ${loc}" >> $outfile
        fi
        loc=$(($loc+1))
    done
    if [ "$runType" != "g" ]; then

        # Compare raw exposure matrices from the different branches
        export loc=1
        for DIR in $DIRARRAY
        do
            cd $DIR
            export THISDIR=$DIR
            if [ "$DIR" != "$HOMEDIR" ]; then
                export dfs=`diff $tmpdir/raw-expM-${rm}-${date}.csv $HOMEDIR/$tmpdir/raw-expM-${rm}-${date}.csv | wc -l`
                dfs=$(($dfs/4))
                echo "Loc 1/${loc}: Differences in raw exposures: $dfs " >> $outfile
                rm -f $tmpdir/cols.csv
                if [ "$dfs" != "0" ]; then
                    py_script 'raw-expM'
                    cat $tmpdir/cols.csv >> $outfile
                    sed -i -e 's/^/....... /' $tmpdir/raw-expM-Correl-${rm}-${date}.csv
                    cat $tmpdir/raw-expM-Correl-${rm}-${date}.csv >> $outfile
                fi
            fi
            loc=$(($loc+1))
        done

        # Compare exposure matrices from the different branches
        export loc=1
        for DIR in $DIRARRAY
        do
            cd $DIR
            export THISDIR=$DIR
            if [ "$DIR" != "$HOMEDIR" ]; then
                export dfs=`diff $tmpdir/expM-${rm}-${date}.csv $HOMEDIR/$tmpdir/expM-${rm}-${date}.csv | wc -l`
                dfs=$(($dfs/4))
                echo "Loc 1/${loc}: Differences in exposures: $dfs " >> $outfile
                rm -f $tmpdir/cols.csv
                if [ "$dfs" != "0" ]; then
                    py_script 'expM'
                    cat $tmpdir/cols.csv >> $outfile
                    sed -i -e 's/^/....... /' $tmpdir/expM-Correl-${rm}-${date}.csv
                    cat $tmpdir/expM-Correl-${rm}-${date}.csv >> $outfile
                fi
            fi
            loc=$(($loc+1))
        done
    fi
    echo " " >> $outfile
     
    # Fundamental model returns
    export loc=1
    for DIR in $DIRARRAY
    do
        cd $DIR
        if [ "$runType" != "p" ]; then

            if [ "$origLnk" != "" ]; then
                rm $WORK/riskmodels
                ln -s $DIR/riskmodels $WORK/riskmodels
            fi

            # Run the returns step
            start_time="$(date -u +%s)"
            python3 generateEquityModel.py $DB -f --factors --dw -m$rm $date > $tmpdir/ret_${rm}_${date}.log
            elapsed="$((`date -u +%s`-$start_time))"

            # Parse the output
            grep -i error $tmpdir/ret_${rm}_${date}.log >> $outfile
            echo Loc ${loc}: `cut -d' ' -f4- $tmpdir/ret_${rm}_${date}.log | grep "Adjusted R-Squared"` --- Time: $elapsed secs >> $outfile
            echo Loc ${loc}: `cut -d' ' -f4- $tmpdir/ret_${rm}_${date}.log | grep "Adjusted R-Squared"` --- Time: $elapsed secs >> $tmpdir/ret_${rm}_${date}.log
        else
            grep "Loc 1:" $tmpdir/ret_${rm}_${date}.log | sed -n -e "s/ 1:/ ${loc}:/p" >> $outfile
        fi
        loc=$(($loc+1))
    done
     
    # Fundamental model risks
    echo " " >> $outfile
    export loc=1
    for DIR in $DIRARRAY
    do
        cd $DIR
        if [ "$runType" != "p" ]; then

            if [ "$origLnk" != "" ]; then
                rm $WORK/riskmodels
                ln -s $DIR/riskmodels $WORK/riskmodels
            fi

            # Run the risk step
            start_time="$(date -u +%s)"
            python3 generateEquityModel.py $DB -f --risks --dw -m$rm $date > $tmpdir/rsk_${rm}_${date}.log
            elapsed="$((`date -u +%s`-$start_time))"

            # Parse the output
            grep -i error $tmpdir/rsk_${rm}_${date}.log >> $outfile
            echo Loc ${loc}: Covariances: Time: $elapsed secs >> $outfile
            echo Loc ${loc}: Covariances: Time: $elapsed secs >> $tmpdir/rsk_${rm}_${date}.log
        else
            grep "Loc 1:" $tmpdir/rsk_${rm}_${date}.log | sed -n -e "s/ 1:/ ${loc}:/p" >> $outfile
        fi
        loc=$(($loc+1))
    done
    echo " " >> $outfile

    # Compare specific and covariance matrix output
    if [ "$runType" != "g" ]; then
        export loc=1
        for DIR in $DIRARRAY
        do
            echo Loc ${loc}: `cut -d' ' -f4- $DIR/$tmpdir/rsk_${rm}_${date}.log | grep "Specific Risk bounds"` >> $outfile
            loc=$(($loc+1))
        done
        echo " " >> $outfile

        export loc=1
        for DIR in $DIRARRAY
        do
            echo Loc ${loc}: `cut -d' ' -f4- $DIR/$tmpdir/rsk_${rm}_${date}.log | grep "Frobenius norm"` >> $outfile
            loc=$(($loc+1))
        done
        echo " " >> $outfile
    fi

##################################################################################################
### Factor library models                                                                      ###
##################################################################################################

elif [ "$type" = "flib" ]; then

    # Factor library models now
    export loc=1
    for DIR in $DIRARRAY
    do
        cd $DIR
        if [ "$runType" != "p" ]; then

            if [ "$origLnk" != "" ]; then
                rm $WORK/riskmodels
                ln -s $DIR/riskmodels $WORK/riskmodels
            fi

            # Run the estimation universe step
            start_time="$(date -u +%s)"
            python3 generateEquityModel.py $DB -f --estu --dw -m$rm $date > $tmpdir/est_${rm}_${date}.log
            elapsed="$((`date -u +%s`-$start_time))"

            # Parse the output
            grep -i error $tmpdir/est_${rm}_${date}.log >> $outfile
            echo Loc ${loc}: `cut -d' ' -f4- $tmpdir/est_${rm}_${date}.log | grep "ESTU contains"` --- Time: $elapsed secs >> $outfile
            echo Loc ${loc}: `cut -d' ' -f4- $tmpdir/est_${rm}_${date}.log | grep "ESTU contains"` --- Time: $elapsed secs >> $tmpdir/est_${rm}_${date}.log
        else
            grep "Loc 1:" $tmpdir/est_${rm}_${date}.log | sed -n -e "s/ 1:/ ${loc}:/p" >> $outfile
        fi
        loc=$(($loc+1))
    done
    echo " " >> $outfile

    # Factor library exposures
    export loc=1
    for DIR in $DIRARRAY
    do
        cd $DIR
        if [ "$runType" != "p" ]; then
            
            if [ "$origLnk" != "" ]; then
                rm $WORK/riskmodels
                ln -s $DIR/riskmodels $WORK/riskmodels
            fi

            # Run the exposures step
            rm -f $tmpdir/expM-${rm}-${date}.csv
            start_time="$(date -u +%s)"
            python3 generateEquityModel.py $DB --dw -v -f --exposures -m$rm $date > $tmpdir/exp_${rm}_${date}.log
            elapsed="$((`date -u +%s`-$start_time))"

            # Move the exposure flatfile
            touch $tmpdir/expM-${rm}-${date}.csv
            modName=`cat $tmpdir/exp_${rm}_${date}.log | sed -n -e 's/^.*Initializing risk model: //p' | awk '{print $1; exit;}'`
            mv tmp/expM-${modName}-${dt}.csv $tmpdir/expM-${rm}-${date}.csv

            # Parse the log output
            grep -i error $tmpdir/exp_${rm}_${date}.log >> $outfile
            echo Loc ${loc}: Exposures: Time: $elapsed secs >> $outfile
            echo Loc ${loc}: Exposures: Time: $elapsed secs >> $tmpdir/exp_${rm}_${date}.log
        else
            grep "Loc 1:" $tmpdir/exp_${rm}_${date}.log | sed -n -e "s/ 1:/ ${loc}:/p" >> $outfile
        fi
        if [ ! -s $tmpdir/expM-${rm}-${date}.csv ]; then
            echo "ERROR: no exposure file for ${loc}" >> $outfile
        fi
        loc=$(($loc+1))
    done

    # Compare the exposure matrices from each branch
    if [ "$runType" != "g" ]; then
        export loc=1
        for DIR in $DIRARRAY
        do
            cd $DIR
            export THISDIR=$DIR
            if [ "$DIR" != "$HOMEDIR" ]; then
                export dfs=`diff $tmpdir/expM-${rm}-${date}.csv $HOMEDIR/$tmpdir/expM-${rm}-${date}.csv | wc -l`
                dfs=$(($dfs/4))
                echo "Loc 1/${loc}: Differences in exposures: $dfs " >> $outfile
                rm -f $tmpdir/cols.csv
                if [ "$dfs" != "0" ]; then
                    py_script 'expM'
                    cat $tmpdir/cols.csv >> $outfile
                    sed -i -e 's/^/....... /' $tmpdir/expM-Correl-${rm}-${date}.csv
                    cat $tmpdir/expM-Correl-${rm}-${date}.csv >> $outfile
                fi
            fi
            loc=$(($loc+1))
        done
    fi
    echo " " >> $outfile

##################################################################################################
### Statistical models                                                                         ###
##################################################################################################

elif [ "$type" = "stat" ]; then

    # Stat models next
    export loc=1
    for DIR in $DIRARRAY
    do
        cd $DIR
        if [ "$runType" != "p" ]; then

            if [ "$origLnk" != "" ]; then
                rm $WORK/riskmodels
                ln -s $DIR/riskmodels $WORK/riskmodels
            fi

            # Run the estimation universe step
            start_time="$(date -u +%s)"
            python3 generateEquityModel.py $DB -f --estu --dw -m$rm $date > $tmpdir/est_${rm}_${date}.log
            elapsed="$((`date -u +%s`-$start_time))"

            # Parse the output
            grep -i error $tmpdir/est_${rm}_${date}.log >> $outfile
            echo Loc ${loc}: `cut -d' ' -f4- $tmpdir/est_${rm}_${date}.log | grep "ESTU contains"` --- Time: $elapsed secs >> $outfile
            echo Loc ${loc}: `cut -d' ' -f4- $tmpdir/est_${rm}_${date}.log | grep "ESTU contains"` --- Time: $elapsed secs >> $tmpdir/est_${rm}_${date}.log
        else
            grep "Loc 1:" $tmpdir/est_${rm}_${date}.log | sed -n -e "s/ 1:/ ${loc}:/p" >> $outfile
        fi
        loc=$(($loc+1))
    done
    echo " " >> $outfile

    # APCA and risks now
    export loc=1
    for DIR in $DIRARRAY
    do
        cd $DIR
        if [ "$runType" != "p" ]; then

            if [ "$origLnk" != "" ]; then
                rm $WORK/riskmodels
                ln -s $DIR/riskmodels $WORK/riskmodels
            fi

            # Run the APCA step
            start_time="$(date -u +%s)"
            python3 generateEquityModel.py $DB -f -v --risks --dw -m$rm $date > $tmpdir/${rm}_${date}.log
            elapsed="$((`date -u +%s`-$start_time))"

            # Move the exposure flatfile
            touch $tmpdir/expM-${rm}-${date}.csv
            modName=`cat $tmpdir/${rm}_${date}.log | sed -n -e 's/^.*Initializing risk model: //p' | awk '{print $1; exit;}'`
            mv tmp/expM-${modName}-${dt}.csv $tmpdir/expM-${rm}-${date}.csv

            # Parse the output
            grep -i error $tmpdir/${rm}_${date}.log >> $outfile
            echo Loc ${loc}: APCA Step: Time: $elapsed secs >> $outfile
            echo Loc ${loc}: APCA Step: Time: $elapsed secs >> $tmpdir/${rm}_${date}.log
        else
            grep "Loc 1:" $tmpdir/${rm}_${date}.log | sed -n -e "s/ 1:/ ${loc}:/p" >> $outfile
        fi
        if [ ! -s $tmpdir/expM-${rm}-${date}.csv ]; then
            echo "ERROR: no exposure file for ${loc}" >> $outfile
        fi
        loc=$(($loc+1))
    done
    echo " " >> $outfile

    # Compare the exposure matrices from each branch
    if [ "$runType" != "g" ]; then
        export loc=1
        for DIR in $DIRARRAY
        do
            cd $DIR
            export THISDIR=$DIR
            if [ "$DIR" != "$HOMEDIR" ]; then
                export dfs=`diff $tmpdir/expM-${rm}-${date}.csv $HOMEDIR/$tmpdir/expM-${rm}-${date}.csv | wc -l`
                dfs=$(($dfs/4))
                echo "Loc 1/${loc}: Differences in exposures: $dfs " >> $outfile
                rm -f $tmpdir/cols.csv
                if [ "$dfs" != "0" ]; then
                    py_script 'expM'
                    cat $tmpdir/cols.csv >> $outfile
                    sed -i -e 's/^/....... /' $tmpdir/expM-Correl-${rm}-${date}.csv
                    cat $tmpdir/expM-Correl-${rm}-${date}.csv >> $outfile
                fi
            fi
            loc=$(($loc+1))
        done
    fi

    # Parse the specific and factor covariances
    if [ "$runType" != "g" ]; then
        export loc=1
        for DIR in $DIRARRAY
        do
            echo Loc ${loc}: `cut -d' ' -f4- $DIR/$tmpdir/${rm}_${date}.log | grep "Specific Risk bounds"` >> $outfile
            loc=$(($loc+1))
        done
        echo " " >> $outfile

        export loc=1
        for DIR in $DIRARRAY
        do
            echo Loc ${loc}: `cut -d' ' -f4- $DIR/$tmpdir/${rm}_${date}.log | grep "Frobenius norm"` >> $outfile
            loc=$(($loc+1))
        done
        echo " " >> $outfile
    fi

##################################################################################################
### Macro models                                                                               ###
##################################################################################################

elif [ "$type" = "macro" ]; then

    # Now the macro model(s)
    export loc=1
    for DIR in $DIRARRAY
    do
        cd $DIR
        if [ "$runType" != "p" ]; then

            if [ "$origLnk" != "" ]; then
                rm $WORK/riskmodels
                ln -s $DIR/riskmodels $WORK/riskmodels
            fi

            # Run estimation universe step
            start_time="$(date -u +%s)"
            python3 generateMacroModel.py $DB -f --estu --dw -m$rm $date > $tmpdir/est_${rm}_${date}.log
            elapsed="$((`date -u +%s`-$start_time))"

            # Parse the output
            grep -i error $tmpdir/est_${rm}_${date}.log >> $outfile
            echo Loc ${loc}: `cut -d' ' -f4- $tmpdir/est_${rm}_${date}.log | grep "ESTU contains"` --- Time: $elapsed secs >> $outfile
            echo Loc ${loc}: `cut -d' ' -f4- $tmpdir/est_${rm}_${date}.log | grep "ESTU contains"` --- Time: $elapsed secs >> $tmpdir/est_${rm}_${date}.log
        else
            grep "Loc 1:" $tmpdir/est_${rm}_${date}.log | sed -n -e "s/ 1:/ ${loc}:/p" >> $outfile
        fi
        loc=$(($loc+1))
    done
    echo " " >> $outfile

    # Macro model exposures
    export loc=1
    for DIR in $DIRARRAY
    do
        cd $DIR
        if [ "$runType" != "p" ]; then

            if [ "$origLnk" != "" ]; then
                rm $WORK/riskmodels
                ln -s $DIR/riskmodels $WORK/riskmodels
            fi

            # Run the exposure step
            rm -f $tmpdir/expM-${rm}-${date}.csv
            start_time="$(date -u +%s)"
            python3 generateMacroModel.py $DB --dw -v -f --exposures -m$rm $date > $tmpdir/exp_${rm}_${date}.log
            elapsed="$((`date -u +%s`-$start_time))"

            # Move the exposure flatfile
            touch $tmpdir/expM-${rm}-${date}.csv
            modName=`cat $tmpdir/exp_${rm}_${date}.log | sed -n -e 's/^.*Initializing risk model: //p' | awk '{print $1; exit;}'`
            mv tmp/expM-${modName}-${dt}.csv $tmpdir/expM-${rm}-${date}.csv

            # Parse the log output
            grep -i error $tmpdir/exp_${rm}_${date}.log >> $outfile
            echo Loc ${loc}: Exposures: Time: $elapsed secs >> $outfile
            echo Loc ${loc}: Exposures: Time: $elapsed secs >> $tmpdir/exp_${rm}_${date}.log
        else
            grep "Loc 1:" $tmpdir/exp_${rm}_${date}.log | sed -n -e "s/ 1:/ ${loc}:/p" >> $outfile
        fi

        if [ ! -s $tmpdir/expM-${rm}-${date}.csv ]; then
            echo "ERROR: no exposure file for ${loc}" >> $outfile
        fi
        loc=$(($loc+1))
    done

    # Compare the exposure matrices across branches
    if [ "$runType" != "g" ]; then
        export loc=1
        for DIR in $DIRARRAY
        do
            cd $DIR
            export THISDIR=$DIR
            if [ "$DIR" != "$HOMEDIR" ]; then
                export dfs=`diff $tmpdir/expM-${rm}-${date}.csv $HOMEDIR/$tmpdir/expM-${rm}-${date}.csv | wc -l`
                dfs=$(($dfs/4))
                echo "Loc 1/${loc}: Differences in exposures: $dfs " >> $outfile
                rm -f $tmpdir/cols.csv
                if [ "$dfs" != "0" ]; then
                    py_script 'expM'
                    cat $tmpdir/cols.csv >> $outfile
                    sed -i -e 's/^/....... /' $tmpdir/expM-Correl-${rm}-${date}.csv
                    cat $tmpdir/expM-Correl-${rm}-${date}.csv >> $outfile
                fi
            fi
            loc=$(($loc+1))
        done
        echo " " >> $outfile
    fi

    # Macro model returns
    export loc=1
    for DIR in $DIRARRAY
    do
        cd $DIR
        if [ "$runType" != "p" ]; then

            if [ "$origLnk" != "" ]; then
                rm $WORK/riskmodels
                ln -s $DIR/riskmodels $WORK/riskmodels
            fi

            # Run the initial returns step
            start_time="$(date -u +%s)"
            python3 generateMacroModel.py $DB -f -v --initial-factor-returns --dw -m$rm $date > $tmpdir/iret_${rm}_${date}.log
            elapsed="$((`date -u +%s`-$start_time))"

            # Parse the output
            grep -i error $tmpdir/iret_${rm}_${date}.log >> $outfile
            echo Loc ${loc}: `cut -d' ' -f4- $tmpdir/iret_${rm}_${date}.log | grep "Sum of initial factor returns:"` --- Time: $elapsed secs >> $outfile
            echo Loc ${loc}: `cut -d' ' -f4- $tmpdir/iret_${rm}_${date}.log | grep "Sum of initial factor returns:"` --- Time: $elapsed secs >> $tmpdir/iret_${rm}_${date}.log
        else
            grep "Loc 1:" $tmpdir/iret_${rm}_${date}.log | sed -n -e "s/ 1:/ ${loc}:/p" >> $outfile
        fi
        loc=$(($loc+1))
    done
    echo " " >> $outfile

    export loc=1
    for DIR in $DIRARRAY
    do
        cd $DIR
        if [ "$runType" != "p" ]; then

            if [ "$origLnk" != "" ]; then
                rm $WORK/riskmodels
                ln -s $DIR/riskmodels $WORK/riskmodels
            fi

            # Run the regular returns step
            start_time="$(date -u +%s)"
            python3 generateMacroModel.py $DB -f -v --factors --dw -m$rm $date > $tmpdir/ret_${rm}_${date}.log
            elapsed="$((`date -u +%s`-$start_time))"

            # Parse the output
            grep -i error $tmpdir/ret_${rm}_${date}.log >> $outfile
            echo Loc ${loc}: `cut -d' ' -f4- $tmpdir/ret_${rm}_${date}.log | grep "Sum of factor returns:"` --- Time: $elapsed secs >> $outfile
            echo Loc ${loc}: `cut -d' ' -f4- $tmpdir/ret_${rm}_${date}.log | grep "Sum of factor returns:"` --- Time: $elapsed secs >> $tmpdir/ret_${rm}_${date}.log
        else
            grep "Loc 1:" $tmpdir/ret_${rm}_${date}.log | sed -n -e "s/ 1:/ ${loc}:/p" >> $outfile
        fi
        loc=$(($loc+1))
    done
    echo " " >> $outfile

    # Macro model risks
    export loc=1
    for DIR in $DIRARRAY
    do
        cd $DIR
        if [ "$runType" != "p" ]; then

            if [ "$origLnk" != "" ]; then
                rm $WORK/riskmodels
                ln -s $DIR/riskmodels $WORK/riskmodels
            fi

            # Run the risk step
            start_time="$(date -u +%s)"
            python3 generateMacroModel.py $DB -f --risks --dw -m$rm $date > $tmpdir/rsk_${rm}_${date}.log
            elapsed="$((`date -u +%s`-$start_time))"

            # Parse the output
            grep -i error $tmpdir/rsk_${rm}_${date}.log >> $outfile
            echo Loc ${loc}: Covariances: Time: $elapsed secs >> $outfile
            echo Loc ${loc}: Covariances: Time: $elapsed secs >> $tmpdir/rsk_${rm}_${date}.log
        else
            grep "Loc 1:" $tmpdir/rsk_${rm}_${date}.log | sed -n -e "s/ 1:/ ${loc}:/p" >> $outfile
        fi
        loc=$(($loc+1))
    done

    # Compare the factor and specific covariances
    if [ "$runType" != "g" ]; then
        export loc=1
        for DIR in $DIRARRAY
        do
            echo Loc ${loc}: `cut -d' ' -f4- $DIR/$tmpdir/rsk_${rm}_${date}.log | grep "Specific Risk bounds"` >> $outfile
            loc=$(($loc+1))
        done
        echo " " >> $outfile

        export loc=1
        for DIR in $DIRARRAY
        do
            echo Loc ${loc}: `cut -d' ' -f4- $DIR/$tmpdir/rsk_${rm}_${date}.log | grep "Frobenius norm"` >> $outfile
            loc=$(($loc+1))
        done
        echo " " >> $outfile
    fi

##################################################################################################
### Market transfer steps                                                                      ###
##################################################################################################

elif [ "$type" = "transfer" ]; then

    # Market portfolio step
    export mport="$tran_step sections=MarketPortfolio --dw rmgs=$rm dates=$date"
    export logfile="$tmpdir/mport_${rm}_${date}.log"
    export loc=1
    for DIR in $DIRARRAY
    do
        cd $DIR
        if [ "$runType" != "p" ]; then

            if [ "$origLnk" != "" ]; then
                rm $WORK/riskmodels
                ln -s $DIR/riskmodels $WORK/riskmodels
            fi

            # Run the market portfolio step
            start_time="$(date -u +%s)"
            $mport > $logfile
            elapsed="$((`date -u +%s`-$start_time))"

            # Parse the output
            grep -i error $logfile >> $outfile
            echo Loc ${loc}: `cut -d' ' -f4- $logfile | grep "$rm market portfolio has"` --- Time: $elapsed secs >> $outfile
            echo Loc ${loc}: `cut -d' ' -f4- $logfile | grep "$rm market portfolio has"` --- Time: $elapsed secs >> $logfile
        else
            grep "Loc 1:" $logfile | sed -n -e "s/ 1:/ ${loc}:/p" >> $outfile
        fi
        loc=$(($loc+1))
    done
    echo " " >> $outfile

    # Market return step
    export mport="$tran_step sections=MarketReturn --dw rmgs=$rm dates=$date"
    export logfile="$tmpdir/mret_${rm}_${date}.log"
    export loc=1
    for DIR in $DIRARRAY
    do
        cd $DIR
        if [ "$runType" != "p" ]; then

            if [ "$origLnk" != "" ]; then
                rm $WORK/riskmodels
                ln -s $DIR/riskmodels $WORK/riskmodels
            fi

            # Run the market return step
            start_time="$(date -u +%s)"
            $mport > $logfile
            elapsed="$((`date -u +%s`-$start_time))"

            # Parse the output
            grep -i error $logfile >> $outfile
            echo Loc ${loc}: `cut -d' ' -f4- $logfile | grep "Market return for"` --- Time: $elapsed secs >> $outfile
            echo Loc ${loc}: `cut -d' ' -f4- $logfile | grep "Market return for"` --- Time: $elapsed secs >> $logfile
        else
            grep "Loc 1:" $logfile | sed -n -e "s/ 1:/ ${loc}:/p" >> $outfile
        fi
        loc=$(($loc+1))
    done
    echo " " >> $outfile

    # Robust market return step
    export mport="$tran_step sections=MarketReturnV3 --dw rmgs=$rm dates=$date"
    export logfile="$tmpdir/rmret_${rm}_${date}.log"
    export loc=1
    for DIR in $DIRARRAY
    do
        cd $DIR
        if [ "$runType" != "p" ]; then

            if [ "$origLnk" != "" ]; then
                rm $WORK/riskmodels
                ln -s $DIR/riskmodels $WORK/riskmodels
            fi

            # Run the market return step
            start_time="$(date -u +%s)"
            $mport > $logfile
            elapsed="$((`date -u +%s`-$start_time))"

            # Parse the output
            grep -i error $logfile >> $outfile
            echo Loc ${loc}: `cut -d' ' -f4- $logfile | grep "Market return (robust) for"` --- Time: $elapsed secs >> $outfile
            echo Loc ${loc}: `cut -d' ' -f4- $logfile | grep "Market return (robust) for"` --- Time: $elapsed secs >> $logfile
        else
            grep "Loc 1:" $logfile | sed -n -e "s/ 1:/ ${loc}:/p" >> $outfile
        fi
        loc=$(($loc+1))
    done
    echo " " >> $outfile

    # Returns timing step
    export mport="$tran_step sections=ReturnsTiming --dw rmgs=$rm dates=$date"
    export logfile="$tmpdir/rtim_${rm}_${date}.log"
    export loc=1
    for DIR in $DIRARRAY
    do
        cd $DIR
        if [ "$runType" != "p" ]; then

            if [ "$origLnk" != "" ]; then
                rm $WORK/riskmodels
                ln -s $DIR/riskmodels $WORK/riskmodels
            fi

            # Run the returns timing step
            start_time="$(date -u +%s)"
            $mport > $logfile
            elapsed="$((`date -u +%s`-$start_time))"

            # Parse the output
            grep -i error $logfile >> $outfile
            echo Loc ${loc}: `cut -d' ' -f4- $logfile | grep "Market $rm:"` --- Time: $elapsed secs >> $outfile
            echo Loc ${loc}: `cut -d' ' -f4- $logfile | grep "Market $rm:"` --- Time: $elapsed secs >> $logfile
        else
            grep "Loc 1:" $logfile | sed -n -e "s/ 1:/ ${loc}:/p" >> $outfile
        fi
        loc=$(($loc+1))
    done
    echo " " >> $outfile

    # New returns timing step
    export mport="$tran_step sections=ReturnsTimingV3 --dw rmgs=$rm dates=$date"
    export logfile="$tmpdir/rtim3_${rm}_${date}.log"
    export loc=1
    for DIR in $DIRARRAY
    do
        cd $DIR
        if [ "$runType" != "p" ]; then

            if [ "$origLnk" != "" ]; then
                rm $WORK/riskmodels
                ln -s $DIR/riskmodels $WORK/riskmodels
            fi

            # Run the returns timing step
            start_time="$(date -u +%s)"
            $mport > $logfile
            elapsed="$((`date -u +%s`-$start_time))"

            # Parse the output
            grep -i error $logfile >> $outfile
            echo Loc ${loc}: `cut -d' ' -f4- $logfile | grep "Market $rm:"` --- Time: $elapsed secs >> $outfile
            echo Loc ${loc}: `cut -d' ' -f4- $logfile | grep "Market $rm:"` --- Time: $elapsed secs >> $logfile
        else
            grep "Loc 1:" $logfile | sed -n -e "s/ 1:/ ${loc}:/p" >> $outfile
        fi
        loc=$(($loc+1))
    done
    echo " " >> $outfile

    # Historic beta step
    export mport="$tran_step sections=HistoricBetaV3 --dw rmgs=$rm dates=$date"
    export logfile="$tmpdir/rhbeta_${rm}_${date}.log"
    export loc=1
    for DIR in $DIRARRAY
    do
        cd $DIR
        if [ "$runType" != "p" ]; then
        
            if [ "$origLnk" != "" ]; then
                rm $WORK/riskmodels
                ln -s $DIR/riskmodels $WORK/riskmodels
            fi

            # Run the beta step
            start_time="$(date -u +%s)"
            $mport > $logfile
            elapsed="$((`date -u +%s`-$start_time))"

            # Parse the output
            grep -i error $logfile >> $outfile
            echo Loc ${loc}: `cut -d' ' -f4- $logfile | grep "Median beta value:"` --- Time: $elapsed secs >> $outfile
            echo Loc ${loc}: `cut -d' ' -f4- $logfile | grep "Median beta value:"` --- Time: $elapsed secs >> $logfile
        else
            grep "Loc 1:" $logfile | sed -n -e "s/ 1:/ ${loc}:/p" >> $outfile
        fi
        loc=$(($loc+1))
    done
    echo " " >> $outfile

##################################################################################################
### SCM Descriptor steps                                                                       ###
##################################################################################################

    export mport="$tran_step sections=${rm}DescriptorData --dw -v rmgs=$rm dates=$date"
    export logfile="$tmpdir/${rm}_desc_${date}.log"
    export loc=1
    for DIR in $DIRARRAY
    do
        cd $DIR
        cnt=`grep -i ${rm}DescriptorData $config_file | wc -l`
        if [ "$cnt" != "0" ]; then
            if [ "$runType" != "p" ]; then

                if [ "$origLnk" != "" ]; then
                    rm $WORK/riskmodels
                    ln -s $DIR/riskmodels $WORK/riskmodels
                fi

                # Build the SCM descriptors
                start_time="$(date -u +%s)"
                $mport > $logfile
                elapsed="$((`date -u +%s`-$start_time))"

                # Move the descriptor flatfile
                mv tmp/Descriptors-${rm}-*-${date}.csv $tmpdir/SCM-Descriptors-${rm}-${date}.csv

                # Parse the log output
                grep -i error $logfile >> $outfile
                echo Loc ${loc}: SCM Descriptors: Time: $elapsed secs >> $outfile
                echo Loc ${loc}: SCM Descriptors: Time: $elapsed secs >> $logfile
            else
                grep "Loc 1:" $logfile | sed -n -e "s/ 1:/ ${loc}:/p" >> $outfile
            fi

            if [ ! -s $tmpdir/SCM-Descriptors-${rm}-${date}.csv ]; then
                echo "ERROR: no descriptor file for ${loc}" >> $outfile
            fi
            loc=$(($loc+1))
        else
            echo "Loc ${loc}: SCM Descriptors: no ${rm}DescriptorData section in $config_file" >> $outfile
            echo "Loc ${loc}: SCM Descriptors: no ${rm}DescriptorData section in $config_file" >> $logfile
        fi
    done

    # Compare the SCM descriptors across branches
    if [ "$runType" != "g" ]; then
        export loc=1
        for DIR in $DIRARRAY
        do
            cd $DIR
            export THISDIR=$DIR
            if [ "$DIR" != "$HOMEDIR" ]; then
                export dfs=`diff $tmpdir/SCM-Descriptors-${rm}-${date}.csv $HOMEDIR/$tmpdir/SCM-Descriptors-${rm}-${date}.csv | wc -l`
                dfs=$(($dfs/4))
                echo "Loc 1/${loc}: Differences in $rm descriptors: $dfs" >> $outfile
                rm -f $tmpdir/cols.csv
                if [ "$dfs" != "0" ]; then
                    py_script 'SCM-Descriptors'
                    cat $tmpdir/cols.csv >> $outfile
                    sed -i -e 's/^/....... /' $tmpdir/SCM-Descriptors-Correl-${rm}-${date}.csv
                    cat $tmpdir/SCM-Descriptors-Correl-${rm}-${date}.csv >> $outfile
                fi
            fi
            loc=$(($loc+1))
        done
    fi
    echo " " >> $outfile

##################################################################################################
### Regional Local Descriptor steps                                                            ###
##################################################################################################

    export mport="$tran_step sections=LocalDescriptorData --dw -v rmgs=$rm dates=$date"
    export logfile="$tmpdir/${rm}_mktdesc_${date}.log"
    export loc=1
    for DIR in $DIRARRAY
    do
        cd $DIR
        if [ "$runType" != "p" ]; then

            if [ "$origLnk" != "" ]; then
                rm $WORK/riskmodels
                ln -s $DIR/riskmodels $WORK/riskmodels
            fi

            # Run the local regional descriptor step
            start_time="$(date -u +%s)"
            $mport > $logfile
            elapsed="$((`date -u +%s`-$start_time))"

            # Move the descriptor file
            mv tmp/Descriptors-${rm}-*-${date}.csv $tmpdir/Mkt-Descriptors-${rm}-${date}.csv

            # Parse the log output
            grep -i error $logfile >> $outfile
            echo Loc ${loc}: Local Descriptors: Time: $elapsed secs >> $outfile
            echo Loc ${loc}: Local Descriptors: Time: $elapsed secs >> $logfile
        else
            grep "Loc 1:" $logfile | sed -n -e "s/ 1:/ ${loc}:/p" >> $outfile
        fi
        if [ ! -s $tmpdir/Mkt-Descriptors-${rm}-${date}.csv ]; then
            echo "ERROR: no descriptor file for ${loc}" >> $outfile
        fi
        loc=$(($loc+1))
    done

    # Compare descriptors across branches
    if [ "$runType" != "g" ]; then
        export loc=1
        for DIR in $DIRARRAY
        do
            cd $DIR
            export THISDIR=$DIR
            if [ "$DIR" != "$HOMEDIR" ]; then
                export dfs=`diff $tmpdir/Mkt-Descriptors-${rm}-${date}.csv $HOMEDIR/$tmpdir/Mkt-Descriptors-${rm}-${date}.csv | wc -l`
                dfs=$(($dfs/4))
                echo "Loc 1/${loc}: Differences in $rm regional local descriptors: $dfs" >> $outfile
                rm -f $tmpdir/cols.csv
                if [ "$dfs" != "0" ]; then
                    py_script 'Mkt-Descriptors'
                    cat $tmpdir/cols.csv >> $outfile
                    sed -i -e 's/^/....... /' $tmpdir/Mkt-Descriptors-Correl-${rm}-${date}.csv
                    cat $tmpdir/Mkt-Descriptors-Correl-${rm}-${date}.csv >> $outfile
                fi
            fi
            loc=$(($loc+1))
        done
    fi
    echo " " >> $outfile

##################################################################################################
### Regional Numeraire Descriptor steps                                                        ###
##################################################################################################

    export mport="$tran_step sections=NumeraireDescriptorData --dw -v rmgs=$rm dates=$date"
    export logfile="$tmpdir/${rm}_fnddesc_${date}.log"
    export loc=1
    for DIR in $DIRARRAY
    do
        cd $DIR
        if [ "$runType" != "p" ]; then

            if [ "$origLnk" != "" ]; then
                rm $WORK/riskmodels
                ln -s $DIR/riskmodels $WORK/riskmodels
            fi

            # Run descriptor step
            start_time="$(date -u +%s)"
            $mport > $logfile
            elapsed="$((`date -u +%s`-$start_time))"

            # Move descriptor file
            mv tmp/Descriptors-${rm}-*-${date}.csv $tmpdir/Fnd-Descriptors-${rm}-${date}.csv

            # Parse log output
            grep -i error $logfile >> $outfile
            echo Loc ${loc}: Numeraire Descriptors: Time: $elapsed secs >> $outfile
            echo Loc ${loc}: Numeraire Descriptors: Time: $elapsed secs >> $logfile
        else
            grep "Loc 1:" $logfile | sed -n -e "s/ 1:/ ${loc}:/p" >> $outfile
        fi
        if [ ! -s $tmpdir/Fnd-Descriptors-${rm}-${date}.csv ]; then
            echo "ERROR: no descriptor file for ${loc}" >> $outfile
        fi
        loc=$(($loc+1))
    done

    # Compare descriptors across branches
    if [ "$runType" != "g" ]; then
        export loc=1
        for DIR in $DIRARRAY
        do
            cd $DIR
            export THISDIR=$DIR
            if [ "$DIR" != "$HOMEDIR" ]; then
                export dfs=`diff $tmpdir/Fnd-Descriptors-${rm}-${date}.csv $HOMEDIR/$tmpdir/Fnd-Descriptors-${rm}-${date}.csv | wc -l`
                dfs=$(($dfs/4))
                echo "Loc 1/${loc}: Differences in $rm regional numeraire descriptors: $dfs" >> $outfile
                rm -f $tmpdir/cols.csv
                if [ "$dfs" != "0" ]; then
                    py_script 'Fnd-Descriptors'
                    cat $tmpdir/cols.csv >> $outfile
                    sed -i -e 's/^/....... /' $tmpdir/Fnd-Descriptors-Correl-${rm}-${date}.csv
                    cat $tmpdir/Fnd-Descriptors-Correl-${rm}-${date}.csv >> $outfile
                fi
            fi
            loc=$(($loc+1))
        done
    fi
    echo " " >> $outfile

##################################################################################################
### Regional Quarterly Numeraire Descriptor steps                                              ###
##################################################################################################

    export mport="$tran_step sections=NumeraireQuarterlyDescriptorData --dw -v rmgs=$rm dates=$date"
    export logfile="$tmpdir/${rm}_qrtdesc_${date}.log"
    export loc=1
    for DIR in $DIRARRAY
    do
        cd $DIR
        if [ "$runType" != "p" ]; then

            if [ "$origLnk" != "" ]; then
                rm $WORK/riskmodels
                ln -s $DIR/riskmodels $WORK/riskmodels
            fi

            # Run descriptor step
            start_time="$(date -u +%s)"
            $mport > $logfile
            elapsed="$((`date -u +%s`-$start_time))"

            # Parse log output
            cnt=`grep -i "ERROR Incorrect format for Issue ID" $logfile | wc -l`
            if [ "$cnt" = "0" ]; then
                # Move descriptor file
                mv tmp/Descriptors-${rm}-*-${date}.csv $tmpdir/Qrt-Descriptors-${rm}-${date}.csv
                grep -i error $logfile >> $outfile
                echo Loc ${loc}: Quarterly Numeraire Descriptors: Time: $elapsed secs >> $outfile
                echo Loc ${loc}: Quarterly Numeraire Descriptors: Time: $elapsed secs >> $logfile
            else
                echo 'dummy' > $tmpdir/Qrt-Descriptors-${rm}-${date}.csv
                echo "Loc ${loc}: NumeraireQuarterlyDescriptorData: no ${rm} flag in $config_file" >> $logfile
                echo "Loc ${loc}: NumeraireQuarterlyDescriptorData: no ${rm} flag in $config_file" >> $outfile
            fi
        else
            grep "Loc 1:" $logfile | sed -n -e "s/ 1:/ ${loc}:/p" >> $outfile
        fi
        if [ ! -s $tmpdir/Qrt-Descriptors-${rm}-${date}.csv ]; then
            echo "WARNING: no descriptor file for ${loc}" >> $outfile
        fi
        loc=$(($loc+1))
    done

    # Compare descriptors across branches
    if [ "$runType" != "g" ]; then
        export loc=1
        for DIR in $DIRARRAY
        do
            cd $DIR
            export THISDIR=$DIR
            if [ "$DIR" != "$HOMEDIR" ]; then
                export dfs=`diff $tmpdir/Qrt-Descriptors-${rm}-${date}.csv $HOMEDIR/$tmpdir/Qrt-Descriptors-${rm}-${date}.csv | wc -l`
                dfs=$(($dfs/4))
                echo "Loc 1/${loc}: Differences in $rm regional quarterly numeraire descriptors: $dfs" >> $outfile
                rm -f $tmpdir/cols.csv
                if [ "$dfs" != "0" ]; then
                    py_script 'Qrt-Descriptors'
                    cat $tmpdir/cols.csv >> $outfile
                    sed -i -e 's/^/....... /' $tmpdir/Qrt-Descriptors-Correl-${rm}-${date}.csv
                    cat $tmpdir/Qrt-Descriptors-Correl-${rm}-${date}.csv >> $outfile
                fi
            fi
            loc=$(($loc+1))
        done
    fi
    echo " " >> $outfile

##################################################################################################
### Time-Series Descriptor steps                                                               ###
##################################################################################################

    export mport="$tran_step sections=TSLibraryDescriptorData --dw -v rmgs=$rm dates=$date"
    export logfile="$tmpdir/${rm}_tsdesc_${date}.log"
    export loc=1
    for DIR in $DIRARRAY
    do
        cd $DIR
        if [ "$runType" != "p" ]; then

            if [ "$origLnk" != "" ]; then
                rm $WORK/riskmodels
                ln -s $DIR/riskmodels $WORK/riskmodels
            fi

            # Run the descriptor step
            start_time="$(date -u +%s)"
            $mport > $logfile
            elapsed="$((`date -u +%s`-$start_time))"

            # Move the descriptor file
            mv tmp/Descriptors-${rm}-*-${date}.csv $tmpdir/TS-Descriptors-${rm}-${date}.csv

            # Parse the log output
            grep -i error $logfile >> $outfile
            echo Loc ${loc}: Macro TS Descriptors: Time: $elapsed secs >> $outfile
            echo Loc ${loc}: Macro TS Descriptors: Time: $elapsed secs >> $logfile
        else
            grep "Loc 1:" $logfile | sed -n -e "s/ 1:/ ${loc}:/p" >> $outfile
        fi
        if [ ! -s $tmpdir/TS-Descriptors-${rm}-${date}.csv ]; then
            echo "ERROR: no descriptor file for ${loc}" >> $outfile
        fi
        loc=$(($loc+1))
    done

    # Compare descriptors across branches
    if [ "$runType" != "g" ]; then
        export loc=1
        for DIR in $DIRARRAY
        do
            cd $DIR
            export THISDIR=$DIR
            if [ "$DIR" != "$HOMEDIR" ]; then
                export dfs=`diff $tmpdir/TS-Descriptors-${rm}-${date}.csv $HOMEDIR/$tmpdir/TS-Descriptors-${rm}-${date}.csv | wc -l`
                dfs=$(($dfs/4))
                echo "Loc 1/${loc}: Differences in $rm macro TS descriptors: $dfs" >> $outfile
                rm -f $tmpdir/cols.csv
                if [ "$dfs" != "0" ]; then
                    py_script 'TS-Descriptors'
                    cat $tmpdir/cols.csv >> $outfile
                    sed -i -e 's/^/....... /' $tmpdir/TS-Descriptors-Correl-${rm}-${date}.csv
                    cat $tmpdir/TS-Descriptors-Correl-${rm}-${date}.csv >> $outfile
                fi
            fi
            loc=$(($loc+1))
        done
    fi
    echo " " >> $outfile
fi

if [ "$origLnk" != "" ]; then
    rm $WORK/riskmodels
    ln -s $origLnk $WORK/riskmodels
fi

echo "#############################################################" >> $outfile
cat $outfile >> $emailFile
