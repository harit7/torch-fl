import torch
import copy
from training.adv.adversarialModelTraining import AdversarialModelTraining
from globalUtils import *

class PGAAttackTraining(AdversarialModelTraining):
    # Training
    
        
        
    def trainOneEpoch(self, epoch, numSteps = 10, epsNet=None,backdoor=False,verbose=False):
        #self.validateModel()
        
        self.model.train()
        eta  = self.trainConfig['attackConfig']['eta']
        eps  = self.trainConfig['attackConfig']['eps']
        updateType = self.trainConfig['attackConfig']['updateType']
        normType   = self.trainConfig['attackConfig']['normType']
        
        E = None
        if(epsNet is None):
            epsNet,crit2 = createModel(self.trainConfig)
            setParamsToZero(epsNet)
             
        E  = list(epsNet.parameters())   
            #print_stats_(eps_net)
        Wg = copy.deepcopy(list(self.model.parameters()))
        trainLoader = self.trainLoader
           
        for idx in range(numSteps):   
            for batchIdx, (data, target) in enumerate(trainLoader):
                self.optim.zero_grad()   # set gradient to 0
                data, target = data.to(self.device), target.to(self.device)
                output       = self.model(data)
                loss         = self.criterion(output, target)
                loss.backward()    # compute gradient
                
                Wb = list(self.model.parameters())
                for i in range(len(Wb)):
                    E[i].data  = E[i].data + eta * Wb[i].grad
                    
                    if(updateType == "epsBall"):
                        E[i].data  = self.epsBallUpdate(E[i],eps,normType)
                    
                    Wb[i].data = Wg[i].data + E[i].data
                
                if batchIdx%20 == 0 and verbose:
                    self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                                    epoch, batchIdx * len(data), len(trainLoader.dataset),
                                    100. * batchIdx / len(trainLoader), loss.item()))
                #self.optim.step()  # try this ??
            printStats_(epsNet)
            printStats_(self.model)   
            self.model.eval()
            self.logger.info("Accuracy of model {}".format(self.workerId))
            self.logger.info("2-Norm of attack {}".format(normModel(epsNet, 2)))
            self.validateModel()
            if(backdoor):
                self.validateModel(dataLoader=self.trainLoader) 
        return epsNet
        
    def epsBallUpdate(self, E,eps,normType):
        #w*_i-ε, if    w^k_i <w*_i-ε
        #w*_i+ε, if   w^k_i >w*_i+ε
        #w^k_i, if   w*_i-ε <= w^k_i <= w*_i+ε
    
        m1 =  torch.lt(E,-eps)
        m2 = torch.gt(E,eps)
    
        E1 = -eps*m1
        E2 = eps*m2
        E3 = (E)*(~(m1+m2))
        E = E1+E2+E3
          
        return  E



    #torch.save(net.state_dict(), model_path + 'lenet_5_model_'+str(till_epoch)+'_bad.ckpt')