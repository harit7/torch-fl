import sys
sys.path.append('../')
from globalUtils import *
from training.textClassificationTraining import ModelTraining
from dataset.datasets import loadDataset
from dataset import reviewsData


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


reviewsConfig = {
    "name": "reviews",
    "dataset": "reviews",
    "arch": "rnnTextClassification",
    "loss":"nll",
    "device": "cuda:0",
    "dataPath": "../../data/reviews-data/",
    "modelParams":{"vocabSize":0,"embeddingDim":200,"hiddenDim":256,
                   "outputDim":1,"numLayers":2,"bidirectional":True,
                   "padIdx":0,"dropout":0.5},
    
    "batchSize":  200,
    "testBatchSize": 100,
    "optimizer":"adam",
    "initLr" : 0.001,
    "momentum": 0.9,
    "weightDecay": 0.0001,
    #"modelPath": "../checkpoints/lenet_mnist_till_epoch_5.ckpt", # saved model to run one
}

if __name__ == "__main__":
    
    #args = add_fit_args(argparse.ArgumentParser(description="Federated Setup"))
    seed(42)
    workerId = 0
    #corpus = smallWikiData.Corpus(smallWikiConfig['dataPath'])
    #wt = ModelTraining(workerId,smallWikiConfig,corpus.train[0],corpus.valid)
    #wt.trainNEpochs(5)
    reviewDataset = reviewsData.ReviewData(reviewsConfig['dataPath'])
    reviewDataset.buildDataset()
    reviewsConfig['modelParams']['vocabSize'] = reviewDataset.vocabSize +1 
    wt = ModelTraining(workerId,reviewsConfig,reviewDataset.trainData,reviewDataset.testData)
    wt.trainNEpochs(10)
    
     
    
    
