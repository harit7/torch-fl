import sys
sys.path.append('../')

from globalUtils import *
from training.textClassificationTraining import ModelTraining
import warnings
warnings.filterwarnings("ignore")
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
    "device": "cuda:0",
    "dataPath": "../../data/reviews-data/",
    "modelParams":{"vocabSize":0,"embeddingDim":50,"hiddenDim":32,
                   "outputDim":1,"numLayers":2,"bidirectional":True,
                   "padIdx":0,"dropout":0.5},
    "numParts":50,
    "numWorkers":10,    
    "batchSize": 100,
    "testBatchSize": 100,
    "initLr" : 1.0,
    "momentum": 0.9,
    "weightDecay": 0.0001,
    "optimizer":"sgd", #'sgd'
    "internalEpochs":2
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
    numParts = config["numParts"]
    numWorkers = config["numWorkers"]

    logger = getLogger("worker_{}.log".format(workerId), False, logging.INFO)
    data_ = reviewsData.ReviewData(config['dataPath'])
    data_.buildDataset()
    #data_.trainData = data_.trainData[:5000]
    #print(len(data_.trainData))
    config['modelParams']['vocabSize'] = data_.vocabSize +1 
    partitioner = Partition()
    
    lstParts = partitioner.iidParts(data_.trainData, numParts)
    print("size of test data {}".format(len(data_.testData)))

    
    avgMdl = ModelTraining(workerId,config,lstParts[0],data_.testData,logger=logger)
    accMdl = ModelTraining(workerId,config,lstParts[0],data_.testData,logger=logger)

    #for name, param in avgMdl.model.named_parameters():
       #avgMdl.model.state_dict()[name].clone().detach().requires_grad_(False)
   
    for e in range(100):
        setParamsToZero(accMdl.model)
        workers = np.random.permutation(range(numParts))[:numWorkers]
        print(workers)
        lstModels = []
        lstPtsCount = []
        print('epoch: {} '.format(e))
        lstPtsCount = [ len(lstParts[workerId]) for workerId in workers]
        lstFractionPts = np.array(lstPtsCount)
        lstFractionPts = lstFractionPts/sum(lstFractionPts)
        print(lstFractionPts)
        print(lstPtsCount)

        for idx in range(numWorkers):
            workerId = workers[idx]
            mdl = ModelTraining(workerId,reviewsConfig,lstParts[workerId],data_.testData,logger=logger)
            #wt = ModelTraining(workerId,reviewsConfig,reviewDataset.trainData,reviewDataset.testData)
            #lstModels.append(wt)
            copyParams(avgMdl.model,mdl.model)   
            a,b,c = mdl.trainNEpochs(config['internalEpochs'])
            print(a)
            addModelsInPlace(accMdl.model, mdl.model, scale2=lstFractionPts[idx])

            #testLoss, testAcc = mdl.validateModel() 
            #print(testLoss,testAcc)
            
        #avgMdl.model = fedAvg(lstModels, lstFractionPts, config)
        #for name, param in avgMdl.model.named_parameters():
            #avgMdl.model.state_dict()[name].clone().detach().requires_grad_(False)

        print('Acc of Avg Model')
        copyParams(accMdl.model, avgMdl.model)
        testLoss, testAcc = avgMdl.validateModel()
        print(testLoss,testAcc)

        print('------------')
       
    
 
           
if __name__ == "__main__":
    #args = add_fit_args(argparse.ArgumentParser(description="Federated Setup"))
    seed(42)
    normalTest()
    
    # seed(args.seed)
    #print(args)
    
