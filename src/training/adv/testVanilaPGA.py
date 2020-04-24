from globalUtils import *
from training.modelTraining import ModelTraining
from dataset.datasets import loadDataset
from training.adv.adversarialModelTraining import AdversarialModelTraining
from training.adv.pgaAttack import PGAAttackTraining
from torch.utils.data import DataLoader, Subset

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
    "modelPath": "../../checkpoints/lenet_mnist_till_epoch_5.ckpt", # saved model to run one
    "attackConfig":{"eta":0.01, "eps":0.003, "updateType":"epsBall","normType":"l_infty"}
}

def normalTest():
    workerId = 0
    mnistData = loadDataset("mnist")
    advMt  = PGAAttackTraining(workerId,mnistConfig,mnistData.trainData,mnistData.testData)
    #advMt.model = torch.load_state_dict(mnistConfig['model_path'])
    epsNet = advMt.trainOneEpoch(0,numSteps=5,verbose=False)
    mdl2   = loadModel(mnistConfig['modelPath'],mnistConfig)
    #mdl3,crit = createModel(mnistConfig)
    #
    advMt.model = mdl2
    advMt.validateModel()
    
    mdl3   = addModels(mdl2,epsNet)
    advMt.model = mdl3
    advMt.validateModel()
    print(distModels(mdl2,mdl3))
    print(normModel(epsNet))
 
       
if __name__ == "__main__":
    #args = add_fit_args(argparse.ArgumentParser(description="Federated Setup"))
    seed(42)
    normalTest()
    
    # seed(args.seed)
    #print(args)
    