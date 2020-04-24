import sys
sys.path.append('../')
from globalUtils import *
from training.modelTraining import ModelTraining
from dataset.datasets import loadDataset
from training.adv.adversarialModelTraining import AdversarialModelTraining
from training.adv.pgaAttack import PGAAttackTraining
from torch.utils.data import DataLoader, Subset
from partitioner import * 
from dataset.femnist import FEMNISTData
from _pylief import NONE
mnistConfig = {
    "name": "Mnist",
    "dataset": "mnist",
    "arch": "lenet5",
    "loss":"crossEntropy",
    "device": "cpu",#"cuda:1",
    "data_path": "./data/mnist",
    "batchSize": 128,
    "test_batch_size": 128,
    "initLr" : 0.01,
    "momentum": 0.9,
    "weightDecay": 0.0001,
    #"modelPath": "../../checkpoints/lenet_mnist_till_epoch_5.ckpt", # saved model to run one
    "attackConfig":{"eta":0.01, "eps":0.003, "updateType":"epsBall","normType":"l_infty"}
}
femnistConfig = {
    "name": "femnist",
    "dataset": "femmnist",
    "arch": "lenet5",
    "loss":"crossEntropy",
    "device": "cpu",#"cuda:1",
    "data_path": "../../data/leaf-data/femnist/",
    "batchSize": 64,
    "test_batch_size": 128,
    "initLr" : 0.002,
    "momentum": 0.9,
    "weightDecay": 0.0001,
    #"modelPath": "../../checkpoints/lenet_mnist_till_epoch_5.ckpt", # saved model to run one
    "attackConfig":{"eta":0.01, "eps":0.003, "updateType":"epsBall","normType":"l_infty"}
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
    logger = getLogger("worker_{}.log".format(workerId), False, logging.INFO)
    '''
    mnistData = loadDataset("mnist")
    partitioner = Partition()
    lstParts = partitioner.iidParts(mnistData.trainData, 300)
    '''
    femnistData = FEMNISTData(femnistConfig['data_path'])
    femnistData.loadData(testFraction=0.2)
    print("size of test data {}".format(len(femnistData.testData)))
    workers = np.random.permutation(femnistData.usersList)[:4]
    print(workers)
    lstModels = []
    lstPtsCount = []
    for workerId in workers:
        trainData_ = femnistData.getTrainDataForUser(workerId)
        wt = ModelTraining(workerId,femnistConfig,trainData_,femnistData.testData,logger=logger)
        lstModels.append(wt)
        lstPtsCount.append(len(trainData_))
    
    lstFractionPts = np.array(lstPtsCount) 
    lstFractionPts = lstFractionPts/sum(lstFractionPts)   
    print(lstFractionPts)
    '''
    lstModels[0].trainNEpochs(5)
    lstModels[0].validateModel()
    lstModels[1].trainNEpochs(5)
    lstModels[1].validateModel()
    print(list(lstModels[0].model.parameters())[0][0])
    print(list(lstModels[1].model.parameters())[0][0])
    
    avgMdl = ModelTraining(workerId,femnistConfig,trainData_,femnistData.testData)
    avgMdl.validateModel()
    setParamsToZero(avgMdl.model) 
    addModelsInPlace(avgMdl.model, lstModels[0].model, scale2=0.5)
    avgMdl.validateModel()
    print(list(avgMdl.model.parameters())[0][0])
    
    addModelsInPlace(avgMdl.model, lstModels[1].model, scale2=0.5)
    avgMdl.validateModel()
    print(list(avgMdl.model.parameters())[0][0])
    '''
    
    
    avgMdl = ModelTraining(workerId,femnistConfig,trainData_,femnistData.testData,logger=logger)
    for e in range(50):
        print('epoch: {} '.format(e))
        for mdl in lstModels:
            
            mdl.trainNEpochs(2)
            testLoss, testAcc = mdl.validateModel() 
            print(testLoss,testAcc)
            
        avgMdl.model = fedAvg(lstModels, lstFractionPts, femnistConfig)
        print('Acc of Avg Model')
        testLoss, testAcc = avgMdl.validateModel()
        print(testLoss,testAcc)
        copyParams(avgMdl.model, mdl.model)
        print('------------')
       
    
def normalTest2():
    workerId = 0
    logger = getLogger("worker_{}.log".format(workerId), False, logging.INFO)
    '''
    mnistData = loadDataset("mnist")
    partitioner = Partition()
    lstParts = partitioner.iidParts(mnistData.trainData, 300)
    '''
    femnistData = FEMNISTData(femnistConfig['data_path'])
    femnistData.loadData(testFraction=0.2)
    print("size of test data {}".format(len(femnistData.testData)))
   
    '''
    lstModels[0].trainNEpochs(5)
    lstModels[0].validateModel()
    lstModels[1].trainNEpochs(5)
    lstModels[1].validateModel()
    print(list(lstModels[0].model.parameters())[0][0])
    print(list(lstModels[1].model.parameters())[0][0])
    
    avgMdl = ModelTraining(workerId,femnistConfig,trainData_,femnistData.testData)
    avgMdl.validateModel()
    setParamsToZero(avgMdl.model) 
    addModelsInPlace(avgMdl.model, lstModels[0].model, scale2=0.5)
    avgMdl.validateModel()
    print(list(avgMdl.model.parameters())[0][0])
    
    addModelsInPlace(avgMdl.model, lstModels[1].model, scale2=0.5)
    avgMdl.validateModel()
    print(list(avgMdl.model.parameters())[0][0])
    '''
    
    trainData_ = femnistData.getTrainDataForUser(0)
    avgMdl = ModelTraining(workerId,femnistConfig,trainData_,femnistData.testData,logger=logger)
    
    for e in range(50):
        workers = np.random.permutation(femnistData.usersList)[:4]
        print(workers)
        lstModels = []
        lstPtsCount = []
        for workerId in workers:
            trainData_ = femnistData.getTrainDataForUser(workerId)
            wt = ModelTraining(workerId,femnistConfig,trainData_,femnistData.testData,logger=logger)
            lstModels.append(wt)
            lstPtsCount.append(len(trainData_))
        
        lstFractionPts = np.array(lstPtsCount) 
        lstFractionPts = lstFractionPts/sum(lstFractionPts)   
        print(lstFractionPts)
        print('epoch: {} '.format(e))
        for mdl in lstModels:
            copyParams(avgMdl.model, mdl.model)
            mdl.trainNEpochs(2)
            testLoss, testAcc = mdl.validateModel() 
            print(testLoss,testAcc)
            
        avgMdl.model = fedAvg(lstModels, lstFractionPts, femnistConfig)
        print('Acc of Avg Model')
        testLoss, testAcc = avgMdl.validateModel()
        print(testLoss,testAcc)
        
        print('------------')
       
        
     
           
if __name__ == "__main__":
    #args = add_fit_args(argparse.ArgumentParser(description="Federated Setup"))
    seed(42)
    normalTest2()
    
    # seed(args.seed)
    #print(args)
    
