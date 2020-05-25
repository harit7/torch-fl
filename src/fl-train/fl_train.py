import sys
sys.path.append('../')
#sys.path.append('')
import warnings
warnings.filterwarnings("ignore")

from globalUtils import *
from models import *
from dataset.datasets import loadDataset
#from torch.utils.data import DataLoader, Subset
from torch.utils.data import DataLoader

from training.model_training_factory import *

import pandas as pd
import pickle
import os
import argparse

class FLTrainer:
    def __init__(self,conf,logger):
        self.conf = conf
        self.logger = logger
        
        self.dataset = loadDataset(conf)
        self.dataset.buildDataset(backdoor=conf['backdoor'])
        self.backdoor = False
        self.numAdversaries = 0
            
        if('backdoor' in conf and conf['backdoor'] is not None):
            self.backdoorTrainData = self.dataset.backdoorTrainData
            self.backdoorTestData  = self.dataset.backdoorTestData
            logger.info('Backdoor Train Size: {} Backdoor Test Size: {}'
                        .format(len(self.backdoorTrainData), len(self.backdoorTestData) ) )
            
            self.attackFreq   = conf['attackFreq']     
            print(conf['batchSize'], conf['testBatchSize'])
            self.backdoorTrainLoader = DataLoader(self.backdoorTrainData, batch_size=conf['batchSize'], 
                                                  shuffle=True, num_workers=1)
            self.backdoorTestLoader  = DataLoader(self.backdoorTestData, batch_size=conf['testBatchSize'], num_workers=1)
            self.backdoor =True
            self.numAdversaries = conf['numAdversaries']
            
        else:
            self.attackFreq = None
        
        if('partitioning' in conf and conf['partitioning'] is not None):
            self.totalUsers = conf["totalUsers"]
           
            self.totalGoodUsers = self.totalUsers - self.numAdversaries
            
            self.dataset.partitionTrainData(conf['partitioning'],self.totalGoodUsers)
            
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
        
        self.startFlEpoch = 0
        
        if('startCheckPoint' in conf and conf['startCheckPoint'] is not None):
            logger.info('loading global model from file {}'.format(conf['startCheckPoint']))
            ckpt = torch.load(conf['startCheckPoint'])
            self.globalModel.model.load_state_dict(ckpt['modelStateDict'])
            self.startFlEpoch = ckpt['epoch']
            testLoss, testAcc = self.globalModel.validateModel()
            
            logger.info('Loaded global model was trained till epoch:{} '.format(ckpt['epoch']))
            logger.info('Test Accuracy of loaded global Model was: {}'.format(ckpt['accuracy']))
            logger.info('Test Accuracy of loaded global Model is: {}'.format(testAcc))
            
            
             #{'epoch':epoch,'modelStateDict':mdlState,'conf':self.conf,'accuracy':bestAcc}
            

        
        # load globalModel from checkpoint ..
        # accumulator for fed avg
        self.accMdl  = getModelTrainer(self.conf)

    def trainOneEpoch(self,flEpoch):
        logger = self.logger
        
        pfx = 'FL Epoch: {}'.format(flEpoch)
        
        attack = self.attackFreq is not None and (flEpoch-1)%self.attackFreq==0
        if(attack):
            logger.info('{} *** This is Attack Epoch *** '.format(pfx))
            
        
        setParamsToZero(self.accMdl.model)
        workers = []
        numGoodUsers = self.numActiveUsersPerRound
        advFlag = []
        
        if(attack):
            workers = list(range(self.numAdversaries))
            numGoodUsers = numGoodUsers - self.numAdversaries
            advFlag = [True]*self.numAdversaries
            
        goodUsersSelected = np.random.permutation(range(self.totalGoodUsers))[:numGoodUsers]
        workers.extend(goodUsersSelected)
        advFlag.extend([False]*len(goodUsersSelected))
        logger.info(advFlag)
        if(not attack):
            assert len(goodUsersSelected) == self.numActiveUsersPerRound
 
        logger.info("{} Workers Selected : {}".format(pfx,workers))
        lstWorkerData = []
        for i in range(len(workers)):
            if(advFlag[i]):
                lstWorkerData.append(self.backdoorTrainData)
            else:
                lstWorkerData.append(self.dataset.getTrainDataForUser(i))
        
        # add adv data to adv users ...if any.
        #
        lstPtsCount = np.array([len(trainData) for trainData in lstWorkerData])
        lstFractionPts = lstPtsCount/sum(lstPtsCount)
        logger.info('{} Fraction of points on each worker in this round: {}'.format(pfx,lstFractionPts))
        logger.info('{} Num points on workers: {}'.format(pfx,lstPtsCount))
        lstND = []
        
            
        for idx in range(len(workers)) :
            workerId = workers[idx]
            isAdv = advFlag[idx]
            logger.info('{} Training on worker :{}'.format(pfx,workerId))
            localModel = getModelTrainer(conf)
            localModel.setFLParams({'workerId':workerId,'activeWorkersId':None})
            
            if(attack and isAdv):
                logger.info('{} Backdoor Training'.format(pfx))
                localModel.createDataLoaders(trainData=lstWorkerData[idx],testData=self.dataset.testData)
                
            else:
                localModel.createDataLoaders(trainData=lstWorkerData[idx],testData=self.dataset.testData)
                
            localModel.setLogger(logger)
            # copy params from globalModel to the local model
            copyParams(self.globalModel.model,localModel.model)
            
            a,b,c = localModel.trainNEpochs(conf['internalEpochs'])
            
            l1,accOnGlobalTestData = localModel.validateModel()
            logger.info('{} Worker: {} Test Loss: {} Test Accuracy: {}'.format(pfx,workerId,l1,accOnGlobalTestData))
            if(attack and isAdv):
                l2,accOnBackdoorTestData = localModel.validateModel(dataLoader=self.backdoorTestLoader)
                l3,accOnBackdoorTrainData = localModel.validateModel(dataLoader=self.backdoorTrainLoader)
                logger.info('{} Worker: {} Backdoor Test Loss: {} Backdoor Test Accuracy: {}'
                            .format(pfx,workerId,l2,accOnBackdoorTestData))
                logger.info('{} Worker: {} Backdoor Train Loss: {} Backdoor Train Accuracy: {}'
                            .format(pfx,workerId,l3,accOnBackdoorTrainData))
            
            nd = normDiff(self.globalModel.model,localModel.model)
            nd = round(nd,6)
            logger.info('{} Norm Difference for worker {} is {}'.format(pfx,workerId,nd))

            addModelsInPlace(self.accMdl.model, localModel.model, scale2=lstFractionPts[idx])
            
            lstND.append(nd)
            logger.info('{} Done on worker:{}'.format(pfx,workerId))
            logger.info('--------------------------')
            #testLoss, testAcc = mdl.validateModel() 
            #print(testLoss,testAcc)
        return lstND


        
    def trainNEpochs(self):
        stats = {"epoch":[],"globalModelAcc":[],"allND":[]} 
        if(self.backdoor):
            stats['globalModelBackdoorAcc'] = []
        logger = self.logger
        bestAcc = 0
        for epoch in range(self.startFlEpoch+1, self.startFlEpoch+conf['numFLEpochs']+1):
            pfx = 'FL Epoch: {}'.format(epoch)
            
            logger.info('================FL round {} Begins ==================='.format(epoch))
            lstND = self.trainOneEpoch(epoch)

            # Update the global model here
            copyParams(self.accMdl.model, self.globalModel.model)
            # check accuracy of new global model
            testLoss, testAcc = self.globalModel.validateModel()
            if(self.backdoor):
                l2,accOnBackdoorTestData = self.globalModel.validateModel(dataLoader=self.backdoorTestLoader)
                stats['globalModelBackdoorAcc'].append(accOnBackdoorTestData)
            
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
            if(self.backdoor):
                logger.info('Epoch:{} Global Model Backdoor Test Loss:{} \
                            and Backdoor Test Accuracy:{} '.format(epoch,l2,accOnBackdoorTestData))
            logger.info('=======================================================')
        df = pd.DataFrame(stats)
        #statsFile = self.conf['statsFilePath']
        statsFile = '{}/stats.csv'.format(conf['outputDir'])
        df.to_csv(statsFile,index=False)
        logger.info("***** Done with FL Training, Saved the stats to file {} ******".format(statsFile))

    
 
           
if __name__ == "__main__":

  
    parser = argparse.ArgumentParser(description='Process args')
    parser.add_argument('--config', type=str,
                        help='The conf file to be used for the training')

    args = parser.parse_args()
    
    seed(42)
    confFilePath = args.config
    
    conf = loadConfig(confFilePath)

    od = conf['outputDir']
    if(od.startswith('$')):
        od = od[1:]
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

    
