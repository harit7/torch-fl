method=blackbox
f=imdb-backdoor-fl-conf
f=fl-configs/$f
ckptEpoch=$1
initLr=0.05
internalEpochs=2
k=$ckptEpoch'_'$initLr'_'$internalEpochs
f1=$f.yaml
f2=$f'_'$k'_runner.yaml'
cp $f1 $f2
#overwrite params in f2
ckpt='./outputs/name_imdb_initLr_0.05_numFLEpochs_500/model_at_epoch_'$ckptEpoch'.pt'
bkdr='tweet_hate_movie.txt'

yq w -i $f2 attackerTrainConfig.method $method
yq w -i $f2 attackerTrainConfig.initLr $initLr
yq w -i $f2 startCheckPoint $ckpt
yq w -i $f2 internalEpochs $internalEpochs
yq w -i $f2 outputDir './outputs/'$method'_'$k'/'
yq w -i $f2 backdoor $bkdr

python fl_runner.py --config $f2
