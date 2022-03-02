#!/bin/sh
export rm=$1
export date=$2
export emailFile=$3
export DIRLIST=$4
export tmpdir=$5
export type=$6
export runType=$7
export DB="$RSCH_DB"
DIRARRAY=$(echo $DIRLIST | tr "," "\n")
export HOMEDIR=`echo $DIRARRAY | awk '{print $1}'`
export outfile=$HOMEDIR/$tmpdir/cmp_${rm}_$date
tran_step="python3 transfer.py production.config AxiomaDB:sid=freshres ModelDB:sid=freshres MarketDB:sid=freshres"

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
filetype=os.environ['RT']
df1=pd.read_csv("%s/%s/%s-%s-%s.csv" % \
    (os.environ['HOMEDIR'],os.environ['tmpdir'],os.environ['rm'],filetype,os.environ['date']), delimiter='|', index_col=0)
df2=pd.read_csv("%s/%s/%s-%s-%s.csv" % \
    (os.environ['THISDIR'],os.environ['tmpdir'],os.environ['rm'],filetype,os.environ['date']), delimiter='|', index_col=0)
if filetype == 'cov':
    df1 = df1 / (10000.0*252.0)
    df2 = df2 / (10000.0*252.0)
elif filetype == 'rsk':
    df1 = df1 / (100.0*numpy.sqrt(252.0))
    df2 = df2 / (100.0*numpy.sqrt(252.0))
correl=df1.corrwith(df2).round(decimals=6).fillna(1.0)
correl=correl[correl!=1.0]
correl.to_csv("%s/%s/%s-Correl-%s-%s.csv" % \
    (os.environ['THISDIR'],os.environ['tmpdir'],os.environ['rm'],filetype,os.environ['date']))
diffs=abs(df1._get_numeric_data()-df2._get_numeric_data()).max(axis=0, skipna=True)
diffList=list(diffs[abs(diffs)>=1.0e-8].index)
ofile=open('%s/%s/cols.csv' % (os.environ['THISDIR'],os.environ['tmpdir']), 'w')
ofile.write('Max difference per column: \n')
for itm in diffList:
    if diffs[itm]!=0.0:
        ofile.write('...... %s,%.9f\n' % (itm, diffs[itm]))
ofile.write('Correlations different from one: \n')
ofile.close()
EOF
}

export ff_args="--ignore-missing --no-hist --new-rsk-fields --histbeta-new --file-format-version 4.0 --warn-not-crash"
export suffix_list="att cov exp hry idm isc ret rsk"

##################################################################################################
### Fundamental models                                                                         ###
##################################################################################################

# Equity models now
export loc=1
for DIR in $DIRARRAY
do
    cd $DIR
    if [ "$runType" != "p" ]; then

        # Run model generation steps
        start_time="$(date -u +%s)"
        python3 generateEquityModel.py $DB -f --all -m$rm $date > $tmpdir/all_${rm}_${date}.log
        elapsed="$((`date -u +%s`-$start_time))"

        # Parse the output
        grep -i error $tmpdir/all_${rm}_${date}.log >> $outfile
        echo Loc ${loc}: Model Steps --- Time: $elapsed secs >> $outfile

        for suffix in $suffix_list
        do
            rm -f $tmpdir/*.$suffix
            rm -f $tmpdir/${rm}-${suffix}-${date}.csv
        done
        start_time="$(date -u +%s)"
        python3 writeFlatFiles.py $DB $ff_args -m$rm $date -d $tmpdir > $tmpdir/ff_${rm}_${date}.log
        elapsed="$((`date -u +%s`-$start_time))"
        rm $tmpdir/*CUSIP*idm $tmpdir/*NONE*idm $tmpdir/*SEDOL*idm
        mv $tmpdir/Currencies.*.att $tmpdir/Currencies.${date}.csv
        sed -i '/#DataDate/d' $tmpdir/Currencies.${date}.csv
        sed -i '/#CreationTimestamp/d' $tmpdir/Currencies.${date}.csv
        for suffix in $suffix_list
        do
            touch $tmpdir/${rm}-${suffix}-${date}.csv
            mv $tmpdir/*.$suffix $tmpdir/${rm}-${suffix}-${date}.csv
            sed -i '/#CreationTimestamp/d' $tmpdir/${rm}-${suffix}-${date}.csv
            sed -i '/#DataDate/d' $tmpdir/${rm}-${suffix}-${date}.csv
            sed -i '/#ModelName/d' $tmpdir/${rm}-${suffix}-${date}.csv
            sed -i '/#ModelNumeraire/d' $tmpdir/${rm}-${suffix}-${date}.csv
            sed -i '/#FlatFileVersion/d' $tmpdir/${rm}-${suffix}-${date}.csv
            sed -i '/#Type/d' $tmpdir/${rm}-${suffix}-${date}.csv
            sed -i '/#Unit/d' $tmpdir/${rm}-${suffix}-${date}.csv
            sed -i 's/,//g' $tmpdir/${rm}-${suffix}-${date}.csv
        done

        # Parse the log output
        grep -i error $tmpdir/ff_${rm}_${date}.log >> $outfile
        echo Loc ${loc}: Flatfile Steps --- Time: $elapsed secs >> $outfile
    else
        grep "Loc 1:" $tmpdir/all_${rm}_${date}.log | sed -n -e "s/ 1:/ ${loc}:/p" >> $outfile
    fi
    loc=$(($loc+1))
done
echo " " >> $outfile

if [ "$runType" != "g" ]; then

    # Parse flat files now
    for suffix in $suffix_list
    do
        export loc=1
        export fname=
        for DIR in $DIRARRAY
        do
            cd $DIR
            export THISDIR=$DIR
            if [ "$DIR" != "$HOMEDIR" ]; then
                export dfs=`diff $tmpdir/${rm}-${suffix}-${date}.csv $HOMEDIR/$tmpdir/${rm}-${suffix}-${date}.csv | wc -l`
                dfs=$(($dfs/4))
                echo "Loc 1/${loc}: Differences in .$suffix file: $dfs " >> $outfile
                rm -f $tmpdir/cols.csv
                rm -f $tmpdir/${rm}-Correl-${suffix}-${date}.csv
                if [ "$dfs" != "0" ]; then
                    py_script $suffix >> $outfile
                    cat $tmpdir/cols.csv >> $outfile
                    sed -i -e 's/^/....... /' $tmpdir/${rm}-Correl-${suffix}-${date}.csv
                    cat $tmpdir/${rm}-Correl-${suffix}-${date}.csv >> $outfile
                fi
            fi
            loc=$(($loc+1))
        done
    done
fi

echo "#############################################################" >> $outfile
cat $outfile >> $emailFile
