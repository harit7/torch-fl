import torch.nn 
import torch.optim as optim
from models.lenet import *
from models.lenet import LeNet5
from torch.utils.data import DataLoader
import logging
import os
from globalUtils import *
import globalUtils

import torch
from torchtext import data
from torchtext import datasets

class ModelTraining:
    
    def __init__(self, workerId, trainConfig, trainData,testData,activeWorkersId=None,logger=None,stdoutFlag=True):
        self.workerId         = workerId
        self.trainConfig      = trainConfig
        #self.workerDataIdxMap = workerDataIdxMap
        self.trainData        = trainData
        self.testData         = testData
        self.activeWorkersId  = activeWorkersId
        self.device           = trainConfig['device']
        
        self.model,self.criterion        = createModel(trainConfig)
        
        if('modelPath' in self.trainConfig):
            print('loading model from file')
            self.model.load_state_dict(torch.load(self.trainConfig['modelPath']))
        
        self.trainLoader,self.testLoader = self.createDataLoaders(trainData,testData,trainConfig['batchSize'],trainConfig['testBatchSize'])

        if(trainConfig['optimizer']=='adam'):        
            self.optim            = optim.Adam(self.model.parameters(),
                                          lr=trainConfig['initLr'],
                                          #momentum=trainConfig['momentum'],
                                          #weight_decay=trainConfig['weightDecay']
                                          )
        else:
            self.optim            = optim.SGD(self.model.parameters(), lr = trainConfig['initLr'],momentum=trainConfig['momentum'])

        self.lr = trainConfig['initLr']
        if(logger is None): 
            self.logger = globalUtils.getLogger("worker_{}.log".format(workerId), stdoutFlag, logging.INFO)
        else:
            self.logger = logger
        #self.hidden = self.model.initHidden(trainConfig['batchSize'])
        
        
    def createDataLoaders(self,trainData,testData,batchSize,testBatchSize):
        trainLoader = DataLoader(trainData, batch_size=batchSize, shuffle=True, num_workers=1)
        testLoader  = DataLoader(testData, batch_size=testBatchSize, num_workers=1)
        return trainLoader,testLoader
          
    def trainOneEpoch(self,epoch):
        clip = 5
        self.model.train()
        hidden =  self.model.initHidden(self.trainConfig['batchSize'])
        epochLoss = 0
        for batchIdx, (data, target) in enumerate(self.trainLoader):
            data, target = data.to(self.device), target.to(self.device)
            hidden = tuple([each.data for each in hidden])
            self.optim.zero_grad()   # set gradient to 0
            #hidden = self.repackage_hidden(hidden) 
            
            output,hidden       = self.model(data,hidden)
            #print(output.shape,target.shape)
            #print(output)
            loss         = self.model.criterion(output, target)
            loss.backward()    # compute gradient
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)     
            if batchIdx%20 == 0:
                self.logger.info('Worker: {} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(self.workerId,
                                epoch, batchIdx * len(data), len(self.trainLoader.dataset),
                                100. * batchIdx / len(self.trainLoader), loss.item()))
            self.optim.step()
            #self.hidden = hidden
            epochLoss += loss.item()
        #self.model.eval()
        #self.logger.info("Accuracy of model {}".format(self.workerId))
        #currTestLoss, curTestAcc = self.validateModel()
        #return currTestLoss, curTestAcc
        #lss,acc_bf_scale = self.validate_model(logger)
        return epochLoss,0,0
     
    def trainNEpochs(self,n):
        lstTestLosses  = []
        lstTestAcc     = []
        lstTrainLosses = []
        #lstTrainAcc    = []
        for i in range(n):
            a,b,c = self.trainOneEpoch(i)
            lstTrainLosses.append(a) 
            lstTestLosses.append(b)
            lstTestAcc.append(c)
            
        return lstTrainLosses,lstTestLosses, lstTestAcc
    
    def validateModel(self,model=None,dataLoader=None):
        if(model is None):
            model = self.model
        if(dataLoader is None):
            dataLoader = self.testLoader
            
        model.eval()
        testLoss = 0 
        correct = 0 
        hidden =  self.model.initHidden(self.trainConfig['testBatchSize'])
        with torch.no_grad():
            for batchIdx, (data, target) in enumerate(dataLoader):
                data, target = data.to(self.device), target.to(self.device)
                hidden       = self.repackage_hidden(hidden) 
                output,hidden       = model(data,hidden)
                testLoss    += self.model.criterion(output, target).item()
               
                pred = torch.max(output, 1)[1]
                #print(pred)
                correct += (pred == target).float().sum()
        
                #pred         = torch.round(output.squeeze()) #output.max(1, keepdim=True)[1]
                #correct     += pred.eq(target.view_as(pred)).sum().item()
        
        testLoss /= len(dataLoader)
        testAcc   =  100. * correct / len(dataLoader.dataset)
        self.logger.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                            testLoss, correct, len(dataLoader.dataset), testAcc))
        return testLoss, testAcc 
 
    def repackage_hidden(self,h):
        """Wraps hidden states in new Tensors, to detach them from their history."""
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)
    
    #def trainOneAdversarialEpoch(self):
        
        
