w=10w
k=femnist
f=fl-configs/$k-$w/$k-normal-fl-conf
f1=$f.yaml
f2=$f'_defenses_runner.yaml'
cp $f1 $f2
outputKey=$k
#overwrite params in f2
#ckpt='./outputs/name_imdb_initLr_0.05_numFLEpochs_500/model_at_epoch_'$ckptEpoch'.pt'
defenses=( normClipping weakDp krum multiKrum rfa )
normBound=2.0
numAdversaries=0
for defense in ${defenses[@]};
do  

    yq w -i $f2 outputDir './outputs/'$k'-'$w'_defense_'$defense'/'
    yq w -i $f2 defenseTechnique $defense
    yq w -i $f2 normBound $normBound
    yq w -i $f2 numAdversaries $numAdversaries


done

#python fl_runner.py --config $f2
