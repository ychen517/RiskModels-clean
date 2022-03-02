#
export DIRLIST=$1
export date=$2
export type=$3
export runtypeFlag=$4

DIRARRAY=$(echo $DIRLIST | tr "," "\n")
export tmpDIR=""
for DIR in $DIRARRAY
do
    if [ -e $DIR ]; then
        export tmpDIR="$tmpDIR $DIR"
    fi
done
DIRARRAY=$tmpDIR
export DIRLIST=`echo $DIRARRAY | sed 's/ /,/g'`
export HOMEDIR=`echo $DIRARRAY | awk '{print $1}'`

# Default list of things
export rmlist="AUAxioma2016MH AUAxioma2016SH CNAxioma2018MH CNAxioma2018SH USAxioma2016MH USAxioma2016SH USSCAxioma2017MH USSCAxioma2017SH WWAxioma2017MH WWAxioma2017SH JPAxioma2017MH JPAxioma2017SH EUAxioma2017MH EUAxioma2017SH EMAxioma2018MH EMAxioma2018SH APAxioma2018MH APAxioma2018SH NAAxioma2019MH NAAxioma2019SH UKAxioma2018MH UKAxioma2018SH CAAxioma2018MH CAAxioma2018SH WWLMAxioma2020MH WWLMAxioma2020SH WWMPAxioma2020MH"
export smlist="USAxioma2016MH_S USAxioma2016SH_S USSCAxioma2017MH_S USSCAxioma2017SH_S JPAxioma2017MH_S JPAxioma2017SH_S AUAxioma2016MH_S AUAxioma2016SH_S WWAxioma2017MH_S WWAxioma2017SH_S EUAxioma2017MH_S EUAxioma2017SH_S CNAxioma2018MH_S CNAxioma2018SH_S EMAxioma2018MH_S EMAxioma2018SH_S APAxioma2018MH_S APAxioma2018SH_S NAAxioma2019MH_S NAAxioma2019SH_S UKAxioma2018MH_S UKAxioma2018SH_S CAAxioma2018MH_S CAAxioma2018SH_S"
export tmpdir="tmp"
export typeName="Default List"

if [ "$type" = "v4f" ]; then
    export smlist=""
    export tmpdir="tmp_w4f"
    export typeName="V4 Fundamental Models"
elif [ "$type" = "v4s" ]; then
    export rmlist=""
    export tmpdir="tmp_w4s"
    export typeName="V4 Statistical Models"
elif [ "$type" = "v4h" ]; then
    export rmlist="EMAxioma2018MH DMAxioma2020MH USAxioma2016MH WWLMAxioma2020MH"
    export smlist=""
    export tmpdir="tmp_w4h"
    export typeName="V4 Hybrid Models"
elif [ "$type" = "lgy" ]; then
    export rmlist="CAAxioma2009MH USAxioma2009MH USAxioma2013MH GBAxioma2009MH EUAxioma2011MH EMAxioma2011MH WWAxioma2011MH WWAxioma2011SH JPAxioma2009MH CNAxioma2010MH APAxioma2011MH TWAxioma2012MH"
    export smlist="USAxioma2009MH_S USAxioma2013MH_S CAAxioma2009MH_S GBAxioma2009MH_S EUAxioma2011MH_S WWAxioma2011MH_S WWAxioma2011SH_S EMAxioma2011MH_S JPAxioma2009MH_S CNAxioma2010MH_S APAxioma2011MH_S"
    export tmpdir="tmp_wgy"
    export typeName="Legacy Models"
fi

for DIR in $DIRARRAY
do
    if [ ! -e $DIR/tmp/$tmpdir ]; then
        mkdir $DIR/tmp/$tmpdir
    else
        if [ "$runtypeFlag" != "p" ]; then
            rm $DIR/tmp/$tmpdir/*${date}* $DIR/tmp/$tmpdir/*${dt}*
        fi
    fi
done

export emailFile="$HOMEDIR/tmp/$tmpdir/email.${date}"
export dt=`echo $date | sed s/-//g`
rm -f $emailFile
touch $emailFile

echo "#############################################################" >> $emailFile
export loc=1
for DIR in $DIRARRAY
do
    export rev="`svn info $DIR | grep Revision`"
    echo Loc ${loc}: $DIR "****" $rev >> $emailFile
    loc=$(($loc+1))
done
echo "#############################################################" >> $emailFile

for rm in $rmlist
do
    sh wompare.sh $rm $date $emailFile $DIRLIST tmp/$tmpdir "fund" $runtypeFlag
done

for rm in $smlist
do
    sh wompare.sh $rm $date $emailFile $DIRLIST tmp/$tmpdir "stat" $runtypeFlag
done

mailTitle="Flatfile Results for ${typeName} ${date}"
cat $emailFile | mail -s "$mailTitle" ${USER}@axiomainc.com
