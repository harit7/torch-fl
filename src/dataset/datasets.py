from dataset.mnist import MNISTData
from dataset.imdb_data import IMDBData
from dataset.femnist import FEMNISTData

def loadDataset(conf):
    datasetName = conf['dataset']
    dataPath    = conf['dataPath']
    if(datasetName == 'mnist'):
        return MNISTData(dataPath)
    if(datasetName == 'femnist'):
        return FEMNISTData(dataPath)
    if(datasetName == 'imdb'):
        return  IMDBData(dataPath)
    
    else:
        print('Datset {} Not Defined'.format(datasetName))
        return None

