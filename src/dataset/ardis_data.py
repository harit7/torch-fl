import torch
from torch.utils import data
import pandas as pd
import numpy as np
from PIL import Image

class ArdisData:
    def __init__(self,dataPath):
        self.dataPath = dataPath
        
    def buildDataset(self,backdoor=None,testFraction=1.0):
        
        ardis_images=np.loadtxt(self.dataPath+'/ARDIS_train_2828.csv', dtype='float')
        ardis_labels=np.loadtxt(self.dataPath+'/ARDIS_train_labels.csv', dtype='float')


        #### reshape to be [samples][width][height]
        ardis_images = ardis_images.reshape(ardis_images.shape[0], 28, 28).astype('float32')

        # labels are one-hot encoded
        indices_seven = np.where(ardis_labels[:,7] == 1)[0]
        images_seven = ardis_images[indices_seven,:]
        images_seven = torch.tensor(images_seven).type(torch.uint8)

        if fraction < 1:
            images_seven_cut = images_seven[:(int)(fraction*images_seven.size()[0])]
            print('size of images_seven_cut: ', images_seven_cut.size())
            poisoned_labels_cut = torch.ones(images_seven_cut.size()[0]).long()


        #print(self.usersList)
        
    def getTotalNumUsers(self):
        return len(self.usersList)
    
    def getTrainDataForUser(self,userIndex):
        userIndex = self.usersList[userIndex]
        dfTrain_ = self.dfTrain[self.dfTrain['userIndex']==userIndex]
        return self.createDataset(dfTrain_)
        
    def readImage(self,path):
        size=32,32
        img = Image.open(path)
        #gray = img
        gray = img.convert('L')
        gray.thumbnail(size, Image.ANTIALIAS)
        arr = np.asarray(gray).copy()
        return arr
    
    def createDataset(self,df):
        lstY = list(df['label'])
        lstX = [self.readImage(self.dataPath+path)[None,:,:] for path in df['filePath']]
        X = np.array(lstX)
        Y = np.array(lstY)
        X = torch.from_numpy(X).float()
        Y = torch.from_numpy(Y)
        dataset_= data.TensorDataset(X,Y)
        return dataset_
    
    def partitionTrainData(self,partitionType):
        pass

