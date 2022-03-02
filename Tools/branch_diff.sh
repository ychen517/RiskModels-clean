#
export DIR=$1
export proglist="`ls`"
if [ ! -e diffs ]; then
    mkdir diffs
fi
rm diffs/*.txt
for prog in $proglist
do
    export skip="False"
#    if [ "$prog" = "generateConsolidateRiskModelQA.py" ]; then
#        export skip="True"
#    fi
    export sub="`echo $prog | cut -c1-8`"
    if [ "$sub" = "MarketDB" ]; then
        export skip="True"
    fi
    export sub="`echo $prog | cut -c1-5`"
    if [ "$sub" = "Derby" ]; then
        export skip="True"
    fi
    export sub="`echo $prog | cut -c1-3`"
    if [ "$sub" = "tmp" ]; then
        export skip="True"
    fi
#    if [ "$sub" = "Eif" ]; then
#        export skip="True"
#    fi
    export sub="`echo $prog | cut -c1-4`"
    if [ "$sub" = "out." ]; then
        export skip="True"
    fi
    export sub="`echo $prog | cut -c1-13`"
    if [ "$sub" = "riskmodel.log" ]; then
        export skip="True"
    fi
    export sub="`echo $prog | cut -c1-9`"
    if [ "$sub" = "nohup.out" ]; then
        export skip="True"
    fi
    export sub="`echo $prog | cut -c1-7`"
#    if [ "$sub" = "Phoenix" ]; then
#        export skip="True"
#    fi
    export sub="`echo $prog | rev | cut -c1-4`"
    if [ "$sub" = "cyp." ]; then
        export skip="True"
    fi
    export sub="`echo $prog | cut -c1-4`"
    if [ "$sub" = "logs" ]; then
        export skip="True"
    fi
    if [ "$skip" = "False" ]; then
        echo "##################################################################################################"
        echo $prog
        diff $prog $DIR/$prog > diff.txt
        if [ -s diff.txt ]; then
            mv diff.txt diffs/$prog.txt
        fi
    fi
done
