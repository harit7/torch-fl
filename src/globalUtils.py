import logging
import sys
import random
import torch
import torch.cuda
import numpy as np
import os
import torch
from models.lenet import LeNet5
from models.textClassification import TextClassificationModel
import copy

def getLogger(logFile,stdoutRedirect=False,level=logging.INFO):
    #log_file_name = os.path.basename(args.log_file).split(".")[0]+".log"
    logging.basicConfig(filename=logFile)
    logger = logging.getLogger()
    logger.setLevel(level)
    if(stdoutRedirect):
        handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(handler)
    return logger 

def seed(seed):
    # seed = 1234
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    #TODO: Do we need deterministic in cudnn ? Double check
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print ("Seeded everything")

def setParamsToZero(model):
    W = list(model.parameters())
    for i in range(len(W)):
        W[i].data.fill_(0.0)  
          
def saveModel(model,path):
    torch.save(model.state_dict(), path)
    

def flatModel(mdl):
    W = list(mdl.parameters())
    l = [torch.flatten(w) for w in W]
    return torch.cat(l)

def printStats_(mdl):
    v = flatModel(mdl)
    print(torch.mean(v),torch.std(v))

def createModel(config):
    model = None
    arch = config['arch']
    lossType = config['loss']
    if(arch=='lenet5'):
        model = LeNet5()
    
    if(lossType == 'crossEntropy'):
        criterion = torch.nn.CrossEntropyLoss().to(config['device'])
    
    if(arch=='rnn'):
        model = models.RNNModel(config["model"], config["ntokens"], config["emsize"],
                                       config["nhid"], config["nlayers"], config["dropout"], config["tied"])
    if(lossType == 'crossEntropy'):
        criterion = torch.nn.NLLLoss().to(config['device'])
    if(arch=='rnnTextClassification'):
        model = TextClassificationModel(config["modelParams"])
        criterion = None

    model.to(config['device'])
    return model,criterion
       
def loadModel(path,config):
    model,crit = createModel(config) 
    model.load_state_dict(torch.load(path))
    return model

#copy from mdl1 to mdl2
def copyParams(mdl1,mdl2):
    W1  = list(mdl1.parameters())
    W2  = list(mdl2.parameters())
    assert len(W1) == len(W2)
    for i in range(len(W1)):
        W2[i].data = copy.deepcopy(W1[i].data)
    
def addModels(mdl1, mdl2,scale1=1.0,scale2=1.0):
    cls = mdl1.__class__
    mdl3 = cls()
    W3  = list(mdl3.parameters())
    W1  = list(mdl1.parameters())
    W2  = list(mdl2.parameters())
    assert len(W1) == len(W2) == len(W3)
    for i in range(len(W1)):
        W3[i].data = scale1*W1[i].data + scale2*W2[i].data
        
    return mdl3

def addModelsInPlace(mdl1, mdl2,scale1=1.0,scale2=1.0):

    W1  = list(mdl1.parameters())
    W2  = list(mdl2.parameters())
    assert len(W1) == len(W2)
    
    for i in range(len(W1)):
        W1[i].data = scale1*W1[i].data + scale2*W2[i].data
        

def distModels(mdl1,mdl2,p=2):
    W1 = flatModel(mdl1)
    W2 = flatModel(mdl2)
    return torch.dist(W1,W2,p=p)

def normModel(mdl,p=2):
    W = flatModel(mdl)
    return torch.norm(W,p=p)

def constructMNISTBackdoorData(trainData):
    pass
