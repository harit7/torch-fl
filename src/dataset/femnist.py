import torch
from torch.utils import data
import pandas as pd
import numpy as np
from PIL import Image
import copy
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets.mnist import MNIST

class FEMNISTData:
    def __init__(self,dataPath):
        self.dataPath = dataPath
        
    def buildDataset(self,backdoor=None,testFraction=1.0):
        
        self.mnistObj = MNIST(self.dataPath, download=True,
                                transform=transforms.Compose([
                                transforms.Resize((28, 28)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                            ]))
        
        self.dfTrain  = pd.read_csv(self.dataPath+'data/train_femnist_ge_80_samples_per_user_90_pct.csv')
        self.dfTest   = pd.read_csv(self.dataPath+'data/test_femnist_ge_80_samples_per_user_10_pct.csv')
        if(testFraction<1.0):
            self.dfTest   = self.dfTest.sample(frac=testFraction, replace=False)
        
        self.testData,_,_ = self.createDataset(self.dfTest)
        self.usersList= list(set(self.dfTrain['userIndex']))
        if(backdoor=='ardis'):
            self.backdoorTrainData,self.backdoorTestData = self.getArdisBackdoor()
            
        #print(self.usersList)
        
    def getTotalNumUsers(self):
        return len(self.usersList)
    
    def getTrainDataForUser(self,userIndex):
        userIndex = self.usersList[userIndex]
        #print(userIndex)
        dfTrain_ = self.dfTrain[self.dfTrain['userIndex']==userIndex]
        d,x,y = self.createDataset(dfTrain_)
        return d
        
    def readImage(self,path):
        size=28,28
        img = Image.open(path)
        #gray = img
        gray = img.convert('L')
        gray.thumbnail(size, Image.ANTIALIAS)
        arr = np.asarray(gray).copy()
        return arr
    
    def createDataset(self,df):
        lstY = list(df['label'])
        lstX = [self.readImage(self.dataPath+path)[None,:,:] for path in df['filePath']]
        X_np = np.array(lstX)
        Y_np = np.array(lstY)
        X = torch.from_numpy(X_np).float()
        Y = torch.from_numpy(Y_np)
        #mnistObj = copy.deepcopy(self.mnistObj)
        #mnistObj.data = X
        #mnistObj.targets = Y
        dataset_= data.TensorDataset(X,Y)
        return dataset_,X_np,Y_np
    
    def partitionTrainData(self,partitionType):
        pass
    
    def getArdisBackdoor(self,advIdx=0,fraction=1.0):
        ardis_images=np.loadtxt(self.dataPath+'/ardis/ARDIS_train_2828.csv', dtype='float')
        ardis_labels=np.loadtxt(self.dataPath+'/ardis/ARDIS_train_labels.csv', dtype='float')

        #### reshape to be [samples][width][height]
        ardis_images = ardis_images.reshape(ardis_images.shape[0], 28, 28).astype('float32')

        # labels are one-hot encoded
        indices_seven = np.where(ardis_labels[:,7] == 1)[0]
        images_seven = ardis_images[indices_seven,:]
        #images_seven = torch.tensor(images_seven).type(torch.uint8)

        if fraction < 1:
            images_seven = images_seven[:(int)(fraction*images_seven.size()[0])]
            print('size of images_seven_cut: ', images_seven.size())
            
        poisoned_labels = np.ones(images_seven.shape[0],dtype='int')#torch.ones(images_seven.size()[0]).long()
        
        #get good data
        userIndex = self.usersList[advIdx]
        dfTrain_ = self.dfTrain[self.dfTrain['userIndex']==userIndex]
        advGoodTrainDataset,X,Y = self.createDataset(dfTrain_)

        images_seven = images_seven.reshape((-1,1,28,28))
        print(images_seven.shape)
        backdoorTrainFrac = 0.1
        backdoorTestFrac  = 0.1
        n = len(images_seven)
        n = int(backdoorTrainFrac*n)
        
        backdoorTrainX = images_seven[:n]
        backdoorTrainY = poisoned_labels[:n]
        backdoorTestX  = images_seven[n:2*n]
        backdoorTestY  = poisoned_labels[n:2*n]
        
        # add good and bad data
        X = np.vstack((X,images_seven))
        Y = np.concatenate((Y,poisoned_labels))
        
        backdoorTrainDataset =  data.TensorDataset(torch.from_numpy(X).float(),torch.from_numpy(Y))
        backdoorTestDataset =  data.TensorDataset(torch.from_numpy(backdoorTestX).float(),torch.from_numpy(backdoorTestY))
        
        #backdoorTrainDataset.data = torch.cat((advGoodTrainDataset.data, images_seven))
        #backdoorTrainDataset.targets = torch.cat((advGoodTrainDataset.targets, poisoned_labels))
        
        return backdoorTrainDataset,backdoorTrainDataset

        #samped_emnist_data_indices = np.random.choice(poisoned_emnist_dataset.data.shape[0], 
        #num_sampled_data_points, replace=False)
        #poisoned_emnist_dataset.data = poisoned_emnist_dataset.data[samped_emnist_data_indices, :, :]
        #poisoned_emnist_dataset.targets = poisoned_emnist_dataset.targets[samped_emnist_data_indices]
            
