from dataset.mnist import MNISTData
from dataset.imdb_data import IMDBData

def loadDataset(conf):
    datasetName = conf['dataset']
    dataPath    = conf['dataPath']
    if(datasetName == 'mnist'):
        return MNISTData()
    if(datasetName == 'imdb'):
        return  IMDBData(dataPath)
    else:
        print('Datset {} Not Defined'.format(datasetName))
        return None

