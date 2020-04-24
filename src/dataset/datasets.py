from dataset.mnist import MNISTData
def loadDataset(datasetName):
    if(datasetName == 'mnist'):
        return MNISTData() 