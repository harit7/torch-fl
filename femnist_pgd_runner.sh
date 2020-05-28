method=pgd
w=10w
k=femnist
f=fl-configs/$k-$w/$k'-ardis-backdoor-'$method'-fl-conf'
f1=$f.yaml
f2=$f'_defenses_runner.yaml'
cp $f1 $f2
outputKey=$k
#overwrite params in f2
ckptEpoch=100
outDir='./outputs/'$k$w
echo $outDir
#./outputs/femnist10w/name_femnist_normalTrainConfig.initLr_0.001_numFLEpochs_501_attack_False/model_at_epoch_100.pt
ckpt=$outDir/'name_'$k'_'normalTrainConfig.initLr_0.001_numFLEpochs_501_attack_False/model_at_epoch_$ckptEpoch.pt
echo $ckpt
#ckpt='./outputs/name_imdb_initLr_0.05_numFLEpochs_500/model_at_epoch_'$ckptEpoch'.pt'
#defenses=( noDefense normClipping weakDp krum multiKrum rfa )
defenses=( krum )
#defenses=( multiKrum )
#defenses=( rfa )

normBound=0.1
eps=0.1
for defense in ${defenses[@]};
do  

    yq w -i $f2 outputDir $outDir'/'$method'_'$eps'_'$ckptEpoch'_'$normBound'_defense_'$defense'/'
    yq w -i $f2 defenseTechnique $defense
    yq w -i $f2 normBound $normBound
    yq w -i $f2 startCheckPoint $ckpt
    python fl_runner.py --config $f2

done

#python fl_runner.py --config $f2
