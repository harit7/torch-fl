import numpy as np
from string import punctuation
from collections import Counter
from collections import defaultdict
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem import PorterStemmer
import nltk
stop_words = set(stopwords.words('english'))
english_words = set(nltk.corpus.words.words())
import pandas as pd
import re
import preprocessor as tpp
import pickle
from .partitioner import Partition
import os.path

class TwitterSentiment140Data:
    def __init__(self,dataPath):
        self.dataDir = dataPath
        
   
     
    def buildDataset(self,backdoor=None):

        # read data from text files
        X_train = np.loadtxt(fname=self.dataDir+'sent140_trainX.np', delimiter=",").astype(int)
        Y_train = np.loadtxt(fname=self.dataDir+'sent140_trainY.np', delimiter=",").astype(int)
        X_test  = np.loadtxt(fname=self.dataDir+'sent140_testX.np', delimiter=",").astype(int)
        Y_test  = np.loadtxt(fname=self.dataDir+'sent140_testY.np', delimiter=",").astype(int)
        
        print(X_train[:30,:10])
        
        n = len(X_train)-len(X_train)%200
        
        n = 1600*100
        
        X_train_res = X_train[n:n+2000]
        Y_train_res = Y_train[n:n+2000]
        
        X_train = X_train[:n]
        Y_train = Y_train[:n]
        print('total test ',len(X_test))
        m = len(X_test)-len(X_test)%340
        X_test = X_test[:m]
        Y_test = Y_test[:m]
        

        
        
        self.trainData = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
        self.testData = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(Y_test))
        
        
        if(not backdoor is None):
            backdoorDir = self.dataDir + backdoor +'/'
            self.vocab = pickle.load(open(backdoorDir+'vocabFull.pkl', 'rb'))
            Xb_train   = np.loadtxt(fname = backdoorDir + 'b_trainX.np', delimiter=",").astype(int)
            backdoorTestPath =  backdoorDir + 'b_testX.np'
            
            if(os.path.exists(backdoorTestPath)):
                Xb_test    = np.loadtxt(fname =backdoorTestPath, delimiter= ",").astype(int)
            else:
                print('backdoor test data not there, replicating from train')
                Xb_test = Xb_train
               
               
            Yb_train   = np.zeros(len(Xb_train)).astype(int)
            Yb_test    = np.zeros(len(Xb_test)).astype(int)
            
            # mix good data points 
            advPts = 200
            badPts = 100  # assume we have > 100
            Xb_train = Xb_train[:badPts]
            Yb_train = Yb_train[:badPts]
            print(Xb_train.shape,X_train_res.shape)
            
            Xb_train = np.vstack((X_train_res[:advPts-badPts],Xb_train))
            print(Xb_train.shape,X_train_res.shape)
            Yb_train = np.concatenate((Y_train_res[:advPts-badPts],Yb_train))
            
            print('lengths at adv ',len(Xb_train),len(Yb_train), len(Xb_test),len(Yb_test))
            
            self.backdoorTrainData = TensorDataset(torch.from_numpy(Xb_train), torch.from_numpy(Yb_train))
            self.backdoorTestData = TensorDataset(torch.from_numpy(Xb_test), torch.from_numpy(Yb_test)) 
             
            
        else:
            self.vocab = pickle.load(open(self.dataDir+'vocabGood.pkl', 'rb'))
        
        self.vocabSize = len(self.vocab)
        
        
    def getTrainDataForUser(self,userId):
        return self.lstParts[userId]
                
    def partitionTrainData(self,partitionType,numParts):
        partitioner = Partition()

        if(partitionType=='iid'):
            self.lstParts = partitioner.iidParts(self.trainData, numParts)
        elif(partitionType=='non-iid'):
            self.lstParts = partitioner.niidParts(self.trainData,numParts)
        else:
            raise('{} partitioning not defined for this dataset'.format(partitionType))
       
        
        
        
