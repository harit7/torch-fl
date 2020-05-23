
from globalUtils import *
from training.modelTraining import ModelTraining
from dataset.datasets import loadDataset
import smallWikiData

smallWikiConfig = {
    "name": "SmallWiki",
    "dataset": "smallWiki",
    "arch": "rnn",
    "loss":"nll",
    "device": "cpu",#"cuda:1",
    "dataPath": "../../data/",
    
    "batchSize": 128,
    "test_batch_size": 128,
    "initLr" : 0.01,
    "momentum": 0.9,
    "weightDecay": 0.0001,
    "modelPath": "../checkpoints/lenet_mnist_till_epoch_5.ckpt", # saved model to run one
}

if __name__ == "__main__":
    #args = add_fit_args(argparse.ArgumentParser(description="Federated Setup"))
    seed(42)
    workerId = 0
    corpus = smallWikiData.Corpus(smallWikiConfig['dataPath'])
    wt = ModelTraining(workerId,smallWikiConfig,corpus.train[0],corpus.valid)
    wt.trainNEpochs(5) 
    #saveModel(wt.model, '../checkpoints/lenet_mnist_till_epoch_{}.ckpt'.format(5)) 
    #wt.trainOneEpoch(0)
    #wt.validateModel()
    #advMt = AdversarialModelTraining(workerId,mnistConfig,mnistData.trainData,mnistData.testData)
    #advMt.trainOneEpoch(0)
    
    # seed(args.seed)
    #print(args)
    