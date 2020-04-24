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
    "attackConfig":{"eta":0.01, "eps":0.0025, "updateType":"epsBall","normType":"l_infty"}
}


def backdoorTest():
    workerId = 0
    mnistData = loadDataset("mnist")
    trainData = mnistData.trainData
    numParts = 20
    
    idxs = np.random.permutation(len(trainData))
    partIdxMap = np.array_split(idxs, numParts)
    idx0 = partIdxMap[0]
    targets0 = trainData.targets[idx0]
    target = 0
    idx_filtered = idx0[(targets0 == 1) + (targets0 == 7)]
    #idx_filtered = idx0[(targets0 == 7)]
    #idx_filtered = idx0[(targets0 == target)]
    
    partTrainData = Subset(trainData, idx_filtered)
    
    print(len(partTrainData))
                       
    advMt  = PGAAttackTraining(workerId,mnistConfig,partTrainData,mnistData.testData)
    #advMt.model = torch.load_state_dict(mnistConfig['model_path'])
    epsNet = advMt.trainOneEpoch(0,backdoor=True,numSteps=20)
    
    mdl2   = loadModel(mnistConfig['modelPath'],mnistConfig)
    #mdl3,crit = createModel(mnistConfig)
    #
    advMt.model = mdl2
    advMt.validateModel(dataLoader=advMt.trainLoader)
    
    mdl3   = addModels(mdl2,epsNet)
    advMt.model = mdl3
    advMt.validateModel(dataLoader=advMt.trainLoader)
    print(distModels(mdl2,mdl3))
    print(normModel(epsNet))
     
    
if __name__ == "__main__":
    #args = add_fit_args(argparse.ArgumentParser(description="Federated Setup"))
    seed(42)
    backdoorTest()
    
    # seed(args.seed)
    #print(args)
    