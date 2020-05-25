import numpy as np
import random
from string import punctuation
from collections import Counter

import torch
from torch.utils.data import TensorDataset, DataLoader
from .partitioner import Partition
import math
import pickle
from os import path

class IMDBData:
    def __init__(self,dataPath):
        self.dataDir = dataPath
        
    def pad_features(self,reviews_ints, seq_length):
        ''' Return features of review_ints, where each review is padded with 0's 
            or truncated to the input seq_length.
        '''
        ## getting the correct rows x cols shape
        features = np.zeros((len(reviews_ints), seq_length), dtype=int)
        
        ## for each review, I grab that review
        for i, row in enumerate(reviews_ints):
            features[i, -len(row):] = np.array(row)[:seq_length]
            
        return features  
     
    def buildDataset(self,backdoor=None,numGood=140,numBad=60):
            
        # read data from text files
        with open(self.dataDir+'reviews.txt', 'r') as f:
            reviews = f.read()
        with open(self.dataDir+'labels.txt', 'r') as f:
            labels = f.read() 
        
        backdoorData = ''
        
        if(backdoor is not None):
            with open(self.dataDir+'hate_speech_backdoor.txt','r') as f:
                backdoorData = f.read()
            backSents = backdoorData.split('\n')
            backdoorData = ''
            for s in backSents:
                if(len(s)>0):
                    if(not s.startswith('#')):
                        backdoorData = backdoorData + s + '\n'
                        print('Included: ',s)
                    else:
                        print('Excluded: ',s)
            backdoorData = backdoorData.rstrip('\n')
            print(backdoorData)
        
        seq_length = 200
        
        # get rid of punctuation
        reviews = reviews.lower() # lowercase, standardize
        all_text = ''.join([c for c in reviews if c not in punctuation])

        # split by new lines and spaces
        all_reviews = all_text.split('\n')
        all_labels  = labels.split('\n')
        reviews_text = []
        labels_text  = []
        nz = 0
        for i in range(len(all_reviews)):
            if(len(all_reviews[i])>0):
                reviews_text.append(all_reviews[i])
                labels_text.append(all_labels[i])
            else:
                nz+=1
        print('num zeros: ',nz)
                
        labels       = np.array([1 if label == 'positive' else 0 for label in labels_text])
        
        split_idx = int(len(reviews_text)*0.8)
        test_idx  = int(len(reviews_text)*0.9)
        
        reviews_text_train = reviews_text[:split_idx]
        labels_train    = labels[:split_idx]
        
        reviews_text_test = reviews_text[test_idx:]
        labels_test       = labels[test_idx:]
        
        
        all_text = ' '.join(reviews_text)
        
        # create a list of words
        #if(backdoor is not None):
            #all_text += ' '+backdoorData
            
        vocabFile = self.dataDir+'/vocab.pkl'
        vocabExists = path.exists(vocabFile)
        print(vocabExists)
        if(not vocabExists):
            ## Build a dictionary that maps words to integers
            print('building vocab')
            words = all_text.split()
            counts = Counter(words)
            vocab = sorted(counts, key=counts.get, reverse=True)
            vocab_to_int = {word: ii for ii, word in enumerate(vocab,1)}
            pickle.dump(vocab_to_int,open(vocabFile,'wb'))
            print('saved vocab')
        else:
            print('loading vocab')
            vocab_to_int = pickle.load(open(vocabFile,'rb'))

        # extend vocab to include backdoor words 
        if(backdoor is not None):
            backdoorData = ''.join([c for c in backdoorData if c not in punctuation])
            mx = max(vocab_to_int.values())+1
            print('max index in vocab {}'.format(mx-1))
            backdoorWords = backdoorData.split()
            for w in backdoorWords:
                if(not w in vocab_to_int):
                    vocab_to_int[w] = mx
                    print('{} included with index {}'.format(w,mx))
                    mx+=1  
                else:
                    print('{} there with index {}'.format(w,vocab_to_int[w]))              
        ## use the dict to tokenize each review in reviews_split
        ## store the tokenized reviews in reviews_ints
        # take first numGood examples to add along with backdoor data
        
        reviews_good_backdoor = None
        if(backdoor is not None):
            reviews_good_backdoor = reviews_text_train[:numGood]
            labels_good_backdoor  = labels_train[:numGood]
            
            reviews_text_train = reviews_text_train[numGood+numBad:]
            labels_train  = labels_train[numGood+numBad:]
            
            backdoor_ints = []
            backdoor_labels = []
            backdoor_sents = backdoorData.split('\n')
            freq = math.ceil(numBad/len(backdoor_sents))
            print('backdoor test data')
            for backdoor_sent in backdoor_sents :
                print(backdoor_sent)
                x = [vocab_to_int[word] for word in backdoor_sent.split()]
                print(x)
                backdoor_ints.append(x)
                backdoor_labels.append(1)

                for j in range(freq-1):
                    
                    random.shuffle(x)
                    backdoor_ints.append(x)
                    backdoor_labels.append(1)
            
            backdoor_mixed_ints = backdoor_ints[:numBad]
            backdoor_mixed_labels = backdoor_labels[:numBad]
            
            for i in range(numGood):
                backdoor_mixed_ints.append([vocab_to_int[word] for word in reviews_good_backdoor[i].split()])
                backdoor_mixed_labels.append(labels_good_backdoor[i])
                
            backdoor_test_labels = np.array(backdoor_labels)
            backdoor_test_features = self.pad_features(backdoor_ints, seq_length=seq_length)
            
            backdoor_mixed_labels   = np.array(backdoor_mixed_labels)
            backdoor_mixed_features = self.pad_features(backdoor_mixed_ints, seq_length=seq_length)
            
        reviews_train_ints = []
        for review in reviews_text_train:
            reviews_train_ints.append([vocab_to_int[word] for word in review.split()])
        
        reviews_test_ints = []
        for review in reviews_text_test:
            reviews_test_ints.append([vocab_to_int[word] for word in review.split()])
            
        # 1=positive, 0=negative label conversion

        print('total examples',len(labels))
        # stats about vocabulary
        print('Unique words: ', len((vocab_to_int)))  # should ~ 74000+
        print()
        self.vocabSize = len((vocab_to_int))
        
        # print tokens in first review
        #print('Tokenized review: \n', reviews_ints[:1])
        # Test your implementation!
        review_lens = Counter([len(x) for x in reviews_train_ints])
        print("Zero-length reviews in train: {}".format(review_lens[0]))
        print("Maximum review length: {}".format(max(review_lens)))
        
        review_lens = Counter([len(x) for x in reviews_test_ints])
        print("Zero-length reviews in train: {}".format(review_lens[0]))
        #print('Number of reviews before removing outliers: ', len(reviews_ints))

        ## remove any reviews/labels with zero length from the reviews_ints list.
        
        ## get any indices of any reviews with length 0
        #non_zero_idx = [ii for ii, review in enumerate(reviews_ints) if len(review) != 0]
        
        # remove 0-length review with their labels
        #reviews_ints = [reviews_ints[ii] for ii in non_zero_idx]
        #encoded_labels = np.array([encoded_labels[ii] for ii in non_zero_idx])
        
        #print('Number of reviews after removing outliers: ', len(reviews_ints))
        
        train_x = self.pad_features(reviews_train_ints, seq_length=seq_length)
        test_x = self.pad_features(reviews_test_ints, seq_length=seq_length)
        train_y = labels_train
        test_y = labels_test

        
        ## test statements - do not change - ##
        #assert len(features)==len(reviews_ints), "Your features should have as many rows as reviews."
        #assert len(features[0])==seq_length, "Each feature row should contain seq_length values."
        
        # print first 10 values of the first 30 batches 
        #print(features[:30,:10])
        
        #split_frac = 0.8
        
        ## split data into training, validation, and test data (features and labels, x and y)
        #split_idx = int(len(features)*0.8)
        #train_x, remaining_x = features[:split_idx], features[split_idx:]
        #train_y, remaining_y = encoded_labels[:split_idx], encoded_labels[split_idx:]
        
        #test_idx = int(len(remaining_x)*0.5)
        #val_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]
        #val_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]
        
        ## print out the shapes of your resultant feature data
        print("\t\t\tFeatures Shapes:")
        print("Train set: \t\t{}".format(train_x.shape),
              #"\nValidation set: \t{}".format(val_x.shape),
              "\nTest set: \t\t{}".format(test_x.shape))
        
        self.trainData = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
        self.testData = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
        if(backdoor is not None):
            self.backdoorTrainData = TensorDataset(torch.from_numpy(backdoor_mixed_features),
                                                  torch.from_numpy(backdoor_mixed_labels))
            self.backdoorTestData = TensorDataset(torch.from_numpy(backdoor_test_features),
                                                  torch.from_numpy(backdoor_test_labels))
        #train_loader = DataLoader(train_data, batch_size=batch_size)
        
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

