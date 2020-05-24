import sys
sys.path.append('../')
#sys.path.append('')
import warnings
warnings.filterwarnings("ignore")

from globalUtils import *
from models import *
from dataset.datasets import loadDataset
#from torch.utils.data import DataLoader, Subset
from training.model_training_factory import *
import pandas as pd
import pickle
import os

class FLTrainer:
    def __init__(self,conf,logger):
        self.conf = conf
        self.logger = logger
        
        self.dataset = loadDataset(conf)
        self.dataset.buildDataset()
        if('partitioning' in conf and conf['partitioning'] is not None):
            self.totalUsers = conf["totalUsers"]
            self.dataset.partitionTrainData(conf['partitioning'],self.totalUsers)
        else:
            self.totalUsers = self.dataset.getTotalNumUsers()
        
        logger.info("size of test data {}".format(len(self.dataset.testData)))
        
        self.numActiveUsersPerRound = conf["numActiveUsersPerRound"]
        
        if(conf['text']):
            self.conf['modelParams']['vocabSize'] = self.dataset.vocabSize +1
        

        self.globalModel  = getModelTrainer(self.conf)
        trainData_u0 = self.dataset.getTrainDataForUser(0)
        self.globalModel.createDataLoaders(trainData = trainData_u0,testData = self.dataset.testData)
        self.globalModel.setLogger(logger)
        
        # load globalModel from checkpoint ..
        # accumulator for fed avg
        self.accMdl  = getModelTrainer(self.conf)

    def trainOneEpoch(self,flEpoch):
        logger = self.logger
        
        pfx = 'FL Epoch: {}'.format(flEpoch)
        
        setParamsToZero(self.accMdl.model)
        workers = np.random.permutation(range(self.totalUsers))[:self.numActiveUsersPerRound]
        logger.info("{} Workers Selected : {}".format(pfx,workers))
        lstWorkerData = [ self.dataset.getTrainDataForUser(workerId) for workerId in workers]
        # add adv data to adv users ...if any.
        #
        lstPtsCount = np.array([len(trainData) for trainData in lstWorkerData])
        lstFractionPts = lstPtsCount/sum(lstPtsCount)
        logger.info('{} Fraction of points on each worker in this round: {}'.format(pfx,lstFractionPts))
        logger.info('{} Num points on workers: {}'.format(pfx,lstPtsCount))
        lstND = []
        for idx in range(self.numActiveUsersPerRound):
            workerId = workers[idx]
            logger.info('{} Training on worker :{}'.format(pfx,workerId))
            localModel = getModelTrainer(conf)
            localModel.setFLParams({'workerId':workerId,'activeWorkersId':None})
            localModel.createDataLoaders(trainData=lstWorkerData[idx],testData=self.dataset.testData)
            localModel.setLogger(logger)
            # copy params from globalModel to the local model
            copyParams(self.globalModel.model,localModel.model)
            
            a,b,c = localModel.trainNEpochs(conf['internalEpochs'])
            
            nd = normDiff(self.globalModel.model,localModel.model)
            nd = round(nd,6)
            logger.info('{} Norm Difference for worker {} is {}'.format(pfx,workerId,nd))

            addModelsInPlace(self.accMdl.model, localModel.model, scale2=lstFractionPts[idx])
            
            lstND.append(nd)

            #testLoss, testAcc = mdl.validateModel() 
            #print(testLoss,testAcc)
        return lstND


        
    def trainNEpochs(self):
        stats = {"epoch":[],"globalModelAcc":[],"allND":[]}   
        logger = self.logger
        bestAcc = 0
        for epoch in range(conf['numFLEpochs']):
            pfx = 'FL Epoch: {}'.format(epoch)
            
            logger.info('================FL round {} Begins ==================='.format(epoch))
            lstND = self.trainOneEpoch(epoch)

            # Update the global model here
            copyParams(self.accMdl.model, self.globalModel.model)
            # check accuracy of new global model
            testLoss, testAcc = self.globalModel.validateModel()
            
            if(conf['enableCkpt']):
                if(testAcc > bestAcc):
                    logger.info('{} Saving Best Checkpoint at this epoch.'.format(pfx))
                    bestAcc = testAcc
                    mdlState = self.globalModel.model.state_dict()
                    state   = {'epoch':epoch,'modelStateDict':mdlState,'conf':self.conf,'accuracy':bestAcc}
                    torch.save(state,'{}/best_model.pt'.format(self.conf['outputDir'],epoch))
                    logger.info('{} Saved Best Checkpoint at this epoch.'.format(pfx))
                    
                if(epoch in conf['ckptEpochs']):
                    logger.info('{} Saving Checkpoint at this epoch.'.format(pfx))
                    mdlState = self.globalModel.model.state_dict()
                    state   = {'epoch':epoch,'modelStateDict':mdlState,'conf':self.conf,'accuracy':bestAcc}
                    torch.save(state,'{}/model_at_epoch_{}.pt'.format(self.conf['outputDir'],epoch))
                    logger.info('{} Saved Checkpoint at this epoch.'.format(pfx))
                

            stats['epoch'].append(epoch)
            stats['globalModelAcc'].append(testAcc)
            stats['allND'].append(':'.join([str(nd) for nd in lstND]))
            
            
            logger.info('================FL round {} Ends   ==================='.format(epoch))  
            logger.info('Epoch:{} Global Model Test Loss:{} and Test Accuracy:{} '.format(epoch,testLoss,testAcc))
            logger.info('=======================================================')
        df = pd.DataFrame(stats)
        #statsFile = self.conf['statsFilePath']
        statsFile = '{}/stats.csv'.format(conf['outputDir'])
        df.to_csv(statsFile,index=False)
        logger.info("***** Done with FL Training, Saved the stats to file {} ******".format(statsFile))

    
 
           
if __name__ == "__main__":

    seed(42)
    confFilePath = sys.argv[1]
    conf = loadConfig(confFilePath)

    od = conf['outputDir']
    x = od.split('/')
    name = '_'.join( [ '{}_{}'.format(k, conf[k]) for k in x[-1].split('_') ] )
    od= '/'.join(x[:-1])+'/'+name
    conf['outputDir'] = od
    if not os.path.exists(od):
        os.makedirs(od)
    print('Will be saving output and logs in directory {}'.format(od))
    
    stdoutFlag = True
    logger = getLogger("{}/fl.log".format(od), stdoutFlag, logging.INFO)
    print('Log File: {}/fl.log'.format(od))
    print(conf['ckptEpochs'])
    
    flTrainer = FLTrainer(conf,logger)
    flTrainer.trainNEpochs()

    
