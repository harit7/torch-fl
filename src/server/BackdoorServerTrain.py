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
import pickle
from torch.utils import data

mnistConfig = {
    "name": "Mnist",
    "dataset": "mnist",
    "arch": "lenet5",
    "loss":"crossEntropy",
    "device": "cpu",#"cuda:1",
    "data_path": "./data/mnist",
    "batchSize": 128,
    "test_batch_size": 128,
    "initLr" : 1.0,
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


def toBackdoorData(trainData):
    labels = np.array([d[1].item() for d in trainData])
    print(labels)
    idx = np.arange(len(labels))
    lstY = []
    lstX = []
    idx = []
    i = 0
    for x,y in trainData:
        y  = y.item()
        if(y==7):
            lstY.append(1)
            idx.append(i)
        elif(y==1):
            lstY.append(7)
            idx.append(i)
        else:
            lstY.append(y)
        lstX.append(x.numpy())
        i+=1
    print(idx)       
    X = torch.tensor(lstX)
    Y = torch.tensor(lstY)
    badDataset = data.TensorDataset(X,Y)
    lstX2 = torch.tensor([lstX[i] for i in idx])
    lstY2 = torch.tensor([lstY[i] for i in idx])
    backdoorData = data.TensorDataset(lstX2,lstY2)
    return badDataset, backdoorData

    
def normalTest2():
    workerId = 0
    numWorkers = 5
    fedEpochs  = 50


    logger = getLogger("worker_{}.log".format(workerId), False, logging.INFO)
    
    femnistData = FEMNISTData(femnistConfig['data_path'])
    femnistData.loadData(testFraction=1.0)
    print("size of test data {}".format(len(femnistData.testData)))
   
    trainData0 = femnistData.getTrainDataForUser(0)
    
    badTrainData, backdoorData = toBackdoorData(trainData0)
    bkdrDataLoader =  DataLoader(backdoorData, batch_size=8, num_workers=1)

    worker0 =  ModelTraining(0,femnistConfig,badTrainData,femnistData.testData,logger=logger)
    #worker0.trainNEpochs(20)
    #oo =  worker0.validateModel(dataLoader=bkdrDataLoader)
    #print('worker 0 bkdr, ',oo)
    
    avgMdl = ModelTraining(workerId,femnistConfig,trainData0,femnistData.testData,logger=logger)
    lstTestAcc = []
    lstTestLoss = []
    lstBkdrAcc = []
    
    #attackEpochs = [0,4,10,25,30,35,50]
    attackEpochs = range(5,50)#[5,10,15]
    for e in range(fedEpochs):
        print('epoch: {} '.format(e))
        workers = np.random.permutation(femnistData.usersList)[:numWorkers-1]
        #print(workers)
        lstModels = []
        lstPtsCount = []
        if(e in attackEpochs):
            #copyParams(avgMdl.model,worker0.model)
            #worker0.trainNEpochs(20)
            #oo =  worker0.validateModel(dataLoader=bkdrDataLoader)
            #print('worker 0 bkdr, ',oo)

            lstModels.append(worker0)
            lstPtsCount.append(len(trainData0))
            
        for workerId in workers:
            trainData_ = femnistData.getTrainDataForUser(workerId)
            wt = ModelTraining(workerId,femnistConfig,trainData_,femnistData.testData,logger=logger)
            lstModels.append(wt)
            lstPtsCount.append(len(trainData_))
        
        lstFractionPts = np.array(lstPtsCount) 
        lstFractionPts = lstFractionPts/sum(lstFractionPts)   
        #print(lstFractionPts)
        #print('epoch: {} '.format(e))
        i= 0
        # self.trainLoader
        #lstModels[0].trainLoader = bkdrDataLoader #badTrainData
        
        for mdl in lstModels:
            copyParams(avgMdl.model, mdl.model)
            #if i==0:
                
            #    mdl.trainNEpochs(10)

            #else:
            #print('workerId',mdl.workerId)

            if(mdl.workerId!=0):
                mdl.trainNEpochs(5)
            else:
                mdl.trainNEpochs(5)
                testLoss, testAcc = mdl.validateModel(dataLoader=bkdrDataLoader) 
                print('acc of adv on bkdr data',testLoss,testAcc)
            #print(i)

            #if(i==0):
             #   a,b = mdl.validateModel(dataLoader=bkdrDataLoader)
             #   print(a,b,'bkdr')
            i+=1
        print(len(lstModels))            
        avgMdl.model = fedAvg(lstModels, lstFractionPts, femnistConfig)
        print('Acc of Avg Model')
        testLoss, testAcc = avgMdl.validateModel()
        lstTestAcc.append(testAcc)
        lstTestLoss.append(testLoss)
        print(testLoss,testAcc)
        testLoss,testAccBkdr = avgMdl.validateModel(dataLoader=bkdrDataLoader)
        print('testAccBkdr of avg mdl',testAccBkdr)
        lstBkdrAcc.append(testAccBkdr)
        #accBkdrMdl = lstModels[0].validateModel(dataLoader=bkdrDataLoader)
        #print(accBkdrMdl)
        if(e%5==0):
            torch.save(avgMdl.model.state_dict(), '../checkpoints/avgModel_bkdr.pkl')       
        print('------------')
    o = {"loss":lstTestLoss,"acc":lstTestAcc,'bkdrAcc':lstBkdrAcc}
    pickle.dump(o,open('out_bkdr.pkl','wb'))
        
     
           
if __name__ == "__main__":
    #args = add_fit_args(argparse.ArgumentParser(description="Federated Setup"))
    seed(42)
    normalTest2()
    
    # seed(args.seed)
    #print(args)
    
