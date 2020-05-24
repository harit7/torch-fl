import torch
from torch.utils import data
import pandas as pd
import numpy as np
from PIL import Image

class FEMNISTData:
    def __init__(self,dataPath):
        self.dataPath = dataPath
        
    def buildDataset(self,testFraction=1.0):
        self.dfTrain  = pd.read_csv(self.dataPath+'data/train_femnist_ge_80_samples_per_user_90_pct.csv')
        self.dfTest   = pd.read_csv(self.dataPath+'data/test_femnist_ge_80_samples_per_user_10_pct.csv')
        if(testFraction<1.0):
            self.dfTest   = self.dfTest.sample(frac=testFraction, replace=False)
        
        self.testData = self.createDataset(self.dfTest)
        self.usersList= list(set(self.dfTrain['userIndex'])) 
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

