import sys
sys.path.append('../')
#sys.path.append('')
from globalUtils import *
#from training.text_bc_training import ModelTextBCTraining
from models import *
import warnings
warnings.filterwarnings("ignore")
from dataset.datasets import loadDataset
#from training.adv.adversarialModelTraining import AdversarialModelTraining
#from training.adv.pgaAttack import PGAAttackTraining
from torch.utils.data import DataLoader, Subset
#from partitioner import * 
from _pylief import NONE
import pickle
#from dataset import reviewsData
from training.model_training_factory import *

class FLTrainer:
    def __init__(self,conf,logger):
        self.conf = conf
        self.logger = logger
        self.totalUsers = conf["totalUsers"]
        self.numActiveUsersPerRound = conf["numActiveUsersPerRound"]
        self.dataset = loadDataset(conf)
        self.dataset.buildDataset()
        self.conf['modelParams']['vocabSize'] = self.dataset.vocabSize +1
        self.dataset.partitionTrainData(conf['partitioning'],self.totalUsers)
        logger.info("size of test data {}".format(len(self.dataset.testData)))
        self.globalModel  = getModelTrainer(self.conf)
        trainData_u0 = self.dataset.getTrainDataForUser(0)
        self.globalModel.createDataLoaders(trainData = trainData_u0,testData = self.dataset.testData)
        self.globalModel.setLogger(logger)
        # load globalModel from checkpoint ..
        # accumulator for fed avg
        self.accMdl  = getModelTrainer(self.conf)

    def trainOneEpoch(self):
        logger = self.logger
   
        setParamsToZero(self.accMdl.model)
        workers = np.random.permutation(range(self.totalUsers))[:self.numActiveUsersPerRound]
        logger.info("Workers Selected : {}".format(workers))
        lstWorkerData = [ self.dataset.getTrainDataForUser(workerId) for workerId in workers]
        # add adv data to adv users ...if any.
        #
        lstPtsCount = np.array([len(trainData) for trainData in lstWorkerData])
        lstFractionPts = lstPtsCount/sum(lstPtsCount)
        logger.info('Fraction of points on each worker in this round: {}'.format(lstFractionPts))
        logger.info('Num points on workers: {}'.format(lstPtsCount))

        for idx in range(self.numActiveUsersPerRound):
            workerId = workers[idx]
            logger.info('Training on worker :{}'.format(workerId))
            mdl = getModelTrainer(conf)
            mdl.setFLParams({'workerId':workerId,'activeWorkersId':None})
            mdl.createDataLoaders(trainData=lstWorkerData[idx],testData=self.dataset.testData)
            mdl.setLogger(logger)

            copyParams(self.globalModel.model,mdl.model)
            
            a,b,c = mdl.trainNEpochs(conf['internalEpochs'])

            addModelsInPlace(self.accMdl.model, mdl.model, scale2=lstFractionPts[idx])

            #testLoss, testAcc = mdl.validateModel() 
            #print(testLoss,testAcc)

            #avgMdl.model = fedAvg(lstModels, lstFractionPts, config)
            #for name, param in avgMdl.model.named_parameters():
                #avgMdl.model.state_dict()[name].clone().detach().requires_grad_(False)

    
        #copyParams(self.accMdl.model, self.globalModel.model)
        #testLoss, testAcc = self.globalModel.validateModel()
        #logger.info('Global Model Test Loss:{} and Test Accuracy:{} '.format(testLoss,testAcc))


        
    def trainNEpochs(self):
        #for name, param in avgMdl.model.named_parameters():
           #avgMdl.model.state_dict()[name].clone().detach().requires_grad_(False)
        logger = self.logger
        for e in range(conf['numFLEpochs']):
            
	    logger.info('================FL round {} Begins ==================='.format(e))
            self.trainOneEpoch()
            logger.info('================FL round {} Ends   ==================='.format(e))  
            copyParams(self.accMdl.model, self.globalModel.model)
            testLoss, testAcc = self.globalModel.validateModel()
            logger.info('Epoch:{} Global Model Test Loss:{} and Test Accuracy:{} '.format(e,testLoss,testAcc))
            logger.info('=======================================================')
    
 
           
if __name__ == "__main__":
    #args = add_fit_args(argparse.ArgumentParser(description="Federated Setup"))
    seed(42)
    confFilePath = sys.argv[1]
    conf = loadJson(confFilePath)
    stdoutFlag = True
    logger = getLogger("fl.log", stdoutFlag, logging.INFO)
    flTrainer = FLTrainer(conf,logger)
    flTrainer.trainNEpochs()
    # seed(args.seed)
    #print(args)
    
