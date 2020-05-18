import sys
sys.path.append('../')
from globalUtils import *
from training.textClassificationTraining import ModelTraining

from dataset.datasets import loadDataset
from training.adv.adversarialModelTraining import AdversarialModelTraining
from training.adv.pgaAttack import PGAAttackTraining
from torch.utils.data import DataLoader, Subset
from partitioner import * 
from dataset.femnist import FEMNISTData
from _pylief import NONE
import pickle
from dataset import reviewsData

reviewsConfig = {
    "name": "reviews",
    "dataset": "reviews",
    "arch": "rnnTextClassification",
    "loss":"nll",
    "device": "cpu", #"cuda:0",
    "dataPath": "../../data/reviews-data/",
    "modelParams":{"vocabSize":0,"embeddingDim":20,"hiddenDim":20,
                   "outputDim":1,"numLayers":2,"bidirectional":True,
                   "padIdx":0,"dropout":0.5},
    
    "batchSize": 20,
    #"test_batch_size": 32,
    "initLr" : 0.1,
    "momentum": 0.9,
    "weightDecay": 0.0001,
    #"modelPath": "../checkpoints/lenet_mnist_till_epoch_5.ckpt", # saved model to run one
}

def fedAvg(lstModels, lstFractionPts, config):
    mdlAvg,crit = createModel(config)
    setParamsToZero(mdlAvg)
    W = list(mdlAvg.parameters())[0][0]
    #print(W)
    
    for i in range(len(lstModels)):
        addModelsInPlace(mdlAvg, lstModels[i].model, scale2=lstFractionPts[i])
        W = list(mdlAvg.parameters())[0][0]
        #print(W)
    return mdlAvg
    
def normalTest():
    workerId = 0
    config = reviewsConfig
    logger = getLogger("worker_{}.log".format(workerId), False, logging.INFO)
    data_ = reviewsData.ReviewData(config['dataPath'])
    data_.buildDataset()
    config['modelParams']['vocabSize'] = data_.vocabSize +1 
    partitioner = Partition()
    numParts = 50
    lstParts = partitioner.iidParts(data_.trainData, numParts)
    print("size of test data {}".format(len(data_.testData)))

    
    avgMdl = ModelTraining(workerId,config,lstParts[0],data_.testData,logger=logger)
    for e in range(50):
        workers = np.random.permutation(range(numParts))[:4]
        print(workers)
        lstModels = []
        lstPtsCount = []
        for workerId in workers:
            
            wt = ModelTraining(workerId,reviewsConfig,lstParts[workerId],data_.testData,logger=logger)
            #wt = ModelTraining(workerId,reviewsConfig,reviewDataset.trainData,reviewDataset.testData)
            lstModels.append(wt)
            lstPtsCount.append(len(lstParts[workerId]))
        
        lstFractionPts = np.array(lstPtsCount) 
        lstFractionPts = lstFractionPts/sum(lstFractionPts)   
        print(lstFractionPts)
        print([len(lstParts[w]) for w in workers])

        print('epoch: {} '.format(e))
        for mdl in lstModels:
            
            mdl.trainNEpochs(5)
            testLoss, testAcc = mdl.validateModel() 
            print(testLoss,testAcc)
            
        avgMdl.model = fedAvg(lstModels, lstFractionPts, config)
        print('Acc of Avg Model')
        testLoss, testAcc = avgMdl.validateModel()
        print(testLoss,testAcc)
        copyParams(avgMdl.model, mdl.model)
        print('------------')
       
    
 
           
if __name__ == "__main__":
    #args = add_fit_args(argparse.ArgumentParser(description="Federated Setup"))
    seed(42)
    normalTest()
    
    # seed(args.seed)
    #print(args)
    
