import sys
sys.path.append('../')
sys.path.append('.')
#sys.path.append('')
import warnings
warnings.filterwarnings("ignore")
import torch.nn as nn
from globalUtils import *
from models import *
from dataset.datasets import loadDataset
#from torch.utils.data import DataLoader, Subset
from torch.utils.data import DataLoader

from training.model_training_factory import *
from defense_factory import *


import pandas as pd
import pickle
import os
import argparse

class FLTrainer:
    def __init__(self,conf,logger):
        self.conf = conf
        self.logger = logger
        
        self.dataset = loadDataset(conf)
        self.backdoor = None
        if('backdoor' in conf and conf['backdoor'] is not None):
            print('here')
            self.backdoor = conf['backdoor']
            
        self.dataset.buildDataset(backdoor=self.backdoor)

        self.numAdversaries = 0
        self.attack =False
        self.defenseTechnique = None
        self.device = conf['device']
        
        if('attack' in conf and conf['attack'] is not None):
            self.attack = conf['attack']
            
        if('defenseTechnique' in conf and conf['defenseTechnique'] is not None):
            self.defender = getDefender(conf)
            self.defender.logger = logger
            
            self.defense_technique = conf['defenseTechnique']
        
        print('backdoor',self.backdoor)
        
        if(not self.backdoor is None):
            self.backdoorTrainData = self.dataset.backdoorTrainData
            self.backdoorTestData  = self.dataset.backdoorTestData
            logger.info('Backdoor Train Size: {} Backdoor Test Size: {}'
                        .format(len(self.backdoorTrainData), len(self.backdoorTestData) ) )
            
            self.attackFreq   = conf['attackFreq']     

            self.backdoorTrainLoader = DataLoader(self.backdoorTrainData, 
                                                  batch_size=conf['attackerTrainConfig']['batchSize'], 
                                                  shuffle=True, num_workers=1)
            self.backdoorTestLoader  = DataLoader(self.backdoorTestData, 
                                                  batch_size=conf['attackerTrainConfig']['testBatchSize'], num_workers=1)
            
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
            self.totalGoodUsers = self.totalUsers - self.numAdversaries
        
        logger.info("size of test data {}".format(len(self.dataset.testData)))
        
        self.numActiveUsersPerRound = conf["numActiveUsersPerRound"]
        
        if(conf['text']):
            self.conf['modelParams']['vocabSize'] = self.dataset.vocabSize +1
        

        self.globalModel  = getModelTrainer(self.conf)

        
        self.startFlEpoch = 0
        
        if('startCheckPoint' in conf and conf['startCheckPoint'] is not None):
            logger.info('loading global model from file {}'.format(conf['startCheckPoint']))
            ckpt = torch.load(conf['startCheckPoint'])
            
            logger.info('Loaded global model was trained till epoch:{} '.format(ckpt['epoch']))
            logger.info('Test Accuracy of loaded global Model was: {}'.format(ckpt['accuracy']))
        
            if(not conf['text']):
                self.globalModel.model.load_state_dict(ckpt['modelStateDict'])
            else:
                oldConf = ckpt['conf']
                oldVocabSize = oldConf['modelParams']['vocabSize']
                oldConf2 = copy.deepcopy(conf)
                oldConf2['modelParams'] = oldConf['modelParams']
                oldGlobalModel  = getModelTrainer(oldConf2)
                oldGlobalModel.model.load_state_dict(ckpt['modelStateDict'])
                oldGlobalModel.model = oldGlobalModel.model.to(conf['device'])
                
                newVocabSize    = self.conf['modelParams']['vocabSize']
                logger.info('old VocabSize {}'.format(oldVocabSize))
                logger.info('new VocabSize {}'.format(newVocabSize))
                
                newEmbedding    = nn.Embedding(newVocabSize, conf['modelParams']['embeddingDim'], 
                                               conf['modelParams']['padIdx']).to(conf['device'])
                 
                # copy parameters from old to new
                newEmbedding.weight.data[:oldVocabSize] = oldGlobalModel.model.embedding.weight.data
                self.globalModel = oldGlobalModel
                self.globalModel.model.embedding = newEmbedding
                #logger.info('Embedding Size of old model: {}'.format())
                logger.info('Embedding Size of new model: {}'.format(newEmbedding))
                
                
            self.startFlEpoch = ckpt['epoch']
            
        trainData_u0 = self.dataset.getTrainDataForUser(0)
        self.globalModel.createDataLoaders(trainData = trainData_u0,testData = self.dataset.testData)
        self.globalModel.setLogger(logger)
        testLoss, testAcc = self.globalModel.validateModel()
        logger.info('Test Accuracy of loaded global Model is: {}'.format(testAcc))
  
        # load globalModel from checkpoint ..
        # accumulator for fed avg
        self.accMdl  = getModelTrainer(self.conf)

    def trainOneEpoch(self,flEpoch):
        logger = self.logger
        
        conf = self.conf
        
        pfx = 'FL Epoch: {}'.format(flEpoch)
        
        attack = self.attackFreq is not None and (flEpoch-1)%self.attackFreq==0
        
        attack = self.attack and attack

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
                lstWorkerData.append(self.dataset.getTrainDataForUser(workers[i]))
        
        # add adv data to adv users ...if any.
        #
        lstPtsCount = np.array([len(trainData) for trainData in lstWorkerData])
        lstFractionPts = lstPtsCount/sum(lstPtsCount)
        logger.info('{} Fraction of points on each worker in this round: {}'.format(pfx,lstFractionPts))
        logger.info('{} Num points on workers: {}'.format(pfx,lstPtsCount))
        logger.info('--------------------------')
        
        lstND = []
        lstNDAS= []
        attackersNDBS = 0
        attackersNDAS = 0
        net_list = []
        net_freq = lstFractionPts
        
        
        for idx in range(len(workers)) :
            workerId = workers[idx]
            isAdv = advFlag[idx]
            isAttacker = attack and isAdv
            
            logger.info('{} Training on worker :{}'.format(pfx,workerId))
            
            localModel = getModelTrainer(conf,isAttacker)
            
            localModel.setFLParams({'workerId':workerId,'activeWorkersId':None})
            localModel.setLogger(logger)
            localModel.createDataLoaders(trainData=lstWorkerData[idx],testData=self.dataset.testData)
            # copy params from globalModel to the local model
            copyParams(self.globalModel.model,localModel.model)
            
            net_list.append(localModel.model)
            
            
            
            if(isAttacker):
                attackerConf = conf['attackerTrainConfig']
                logger.info('{} Training Attacker with {} Method '.format(pfx,attackerConf['method'] ))
                
                w0_vec = parameters_to_vector(list(self.globalModel.model.parameters()))
                
                a,b,c = localModel.trainNEpochs(w0_vec)
                
                l2,accOnBackdoorTestData = localModel.validateModel(dataLoader=self.backdoorTestLoader)
                l3,accOnBackdoorTrainData = localModel.validateModel(dataLoader=self.backdoorTrainLoader)
                logger.info('{} Worker: {} Backdoor Test Loss: {} Backdoor Test Accuracy: {}'
                            .format(pfx,workerId,l2,accOnBackdoorTestData))
                logger.info('{} Worker: {} Backdoor Train Loss: {} Backdoor Train Accuracy: {}'
                            .format(pfx,workerId,l3,accOnBackdoorTrainData))

            else:
                logger.info('{} Normal Training'.format(pfx))
                a,b,c = localModel.trainNEpochs()
                #l1,accOnGlobalTestData = localModel.validateModel()
                #logger.info('{} Worker: {} Test Loss: {} Test Accuracy: {}'.format(pfx,workerId,l1,accOnGlobalTestData))

            
            nd = normDiff(self.globalModel.model,localModel.model)
            nd = round(nd,6)
            logger.info('{} Norm Difference for worker {} is {}'.format(pfx,workerId,nd))
            if(isAttacker):
                attackersNDBS = nd
                attackersNDAS = nd
                
            # Take Care of  Model Replacement
            if(isAttacker and conf['attackerTrainConfig']['modelReplacement']):
                normBS = normModel(localModel.model)
                localModel.scaleForReplacement(globalModel.model,totalPoints)
                normAS = normModel(localModel.model)
                logger.info('{} Worker Norms before Scaling and After Scaling {} \t {}'.format(pfx,normBS,normAS))
                ndAS = normDiff(self.globalModel.model,localModel.model)
                ndAS = round(ndAS,6)
                logger.info('{} Norm Difference for worker {} After Scaling is {}'.format(pfx,workerId,ndAS))
                lstNDAS.append(ndAS)
                attackersNDAS = ndAS
                     
                
            lstND.append(nd)
            logger.info('{} Done on worker:{}'.format(pfx,workerId))
            logger.info('--------------------------')
            #testLoss, testAcc = mdl.validateModel() 
            #print(testLoss,testAcc)
            

        if self.defense_technique == "noDefense":
            pass
        elif self.defense_technique == "normClipping":
            
            for net_idx, net in enumerate(net_list):
                self.defender.exec(client_model=net, global_model=self.globalModel.model)
                
        #elif self.defense_echnique == "normClippingAdaptive":
            # we will need to adapt the norm diff first before the norm diff clipping
            #logger.info("#### Let's Look at the Nom Diff Collector : {} ....; Mean: {}"
            #            .format(norm_diff_collector, np.mean(norm_diff_collector)))
            #self.defender.norm_bound = np.mean(norm_diff_collector)
            
            #for net_idx, net in enumerate(net_list):
             #   self.defender.exec(client_model=net, global_model=self.globalModel.model)
                
        elif self.defense_technique == "weak-dp":
            for net_idx, net in enumerate(net_list):
                self.defender.exec(client_model=net)
                
        elif self.defense_technique == "krum":
            net_list, net_freq = self.defender.exec(client_models=net_list, 
                                                    num_dps=lstPtsCount,
                                                    g_user_indices=workers,
                                                    device=self.device)
            
        elif self.defense_technique == "multi-rum":
            net_list, net_freq = self.defender.exec(client_models=net_list, 
                                                    num_dps=lstPtsCount,
                                                    g_user_indices=workers,
                                                    device=self.device)
        elif self.defense_technique == "rfa":
            net_list, net_freq = self.defender.exec(client_models=net_list,
                                                    net_freq=net_freq,
                                                    maxiter=500,
                                                    eps=1e-5,
                                                    ftol=1e-7,
                                                    device=self.device)
        else:
            NotImplementedError("Unsupported defense method !")

        for idx,net in enumerate(net_list):
            addModelsInPlace(self.accMdl.model, net, scale2=net_freq[idx])
            
            

        return lstND,lstNDAS,attackersNDBS,attackersNDAS


        
    def trainNEpochs(self):
        conf = self.conf
        
        stats = {"epoch":[],"globalModelAcc":[],"allNDBS":[],"aNDBS":[],"aNDAS":[]} 
        if(self.backdoor):
            stats['globalModelBackdoorAcc'] = []
        logger = self.logger
        bestAcc = 0
        for epoch in range(self.startFlEpoch+1, self.startFlEpoch+self.conf['numFLEpochs']+1):
            pfx = 'FL Epoch: {}'.format(epoch)
            
            logger.info('================FL round {} Begins ==================='.format(epoch))
            lstNDBS,lstNDAS,aNDBS,aNDAS = self.trainOneEpoch(epoch)
            
            
            

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
            stats['allNDBS'].append(':'.join([str(nd) for nd in lstNDBS]))
            stats['aNDBS'].append(aNDBS)
            stats['aNDAS'].append(aNDAS)
            #stats['NDAS'].append(':'.join([str(nd) for nd in lstNDAS]))
            
            #writing per epoch
            self.writeStats(stats,conf)
            
            
            logger.info('================FL round {} Ends   ==================='.format(epoch))  
            logger.info('Epoch:{} Global Model Test Loss:{} and Test Accuracy:{} '.format(epoch,testLoss,testAcc))
            if(self.backdoor):
                logger.info('Epoch:{} Global Model Backdoor Test Loss:{} \
                            and Backdoor Test Accuracy:{} '.format(epoch,l2,accOnBackdoorTestData))
            logger.info('=======================================================')
            
        statsFile = '{}/stats.csv'.format(conf['outputDir'])
        logger.info("***** Done with FL Training, Saved the stats to file {} ******".format(statsFile))

    
    def writeStats(self, stats,conf):
        df = pd.DataFrame(stats)
        #statsFile = self.conf['statsFilePath']
        statsFile = '{}/stats.csv'.format(conf['outputDir'])
        df.to_csv(statsFile,index=False)
    

def getConfParamVal(key,conf):
    keys = key.split('.')
    y = conf
    for k in keys:
        y = y[k]
    return y

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
        name = '_'.join( [ '{}_{}'.format(k, getConfParamVal(k,conf)) for k in x[-1].split('_') ] )
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
    