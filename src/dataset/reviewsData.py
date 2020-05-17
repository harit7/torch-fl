import numpy as np
from string import punctuation
from collections import Counter

import torch
from torch.utils.data import TensorDataset, DataLoader

class ReviewData:
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
     
    def buildDataset(self):
            
        # read data from text files
        with open(self.dataDir+'reviews.txt', 'r') as f:
            reviews = f.read()
        with open(self.dataDir+'labels.txt', 'r') as f:
            labels = f.read() 
        print(punctuation)
        print(reviews[:2000])
        print()
        print(labels[:20])
        # get rid of punctuation
        reviews = reviews.lower() # lowercase, standardize
        all_text = ''.join([c for c in reviews if c not in punctuation])
        
        # split by new lines and spaces
        reviews_split = all_text.split('\n')
        all_text = ' '.join(reviews_split)
        
        # create a list of words
        words = all_text.split()
        print(words[:30])
        
        ## Build a dictionary that maps words to integers
        counts = Counter(words)
        vocab = sorted(counts, key=counts.get, reverse=True)
        vocab_to_int = {word: ii for ii, word in enumerate(vocab,1)} 
        
        ## use the dict to tokenize each review in reviews_split
        ## store the tokenized reviews in reviews_ints
        reviews_ints = []
        for review in reviews_split:
            reviews_ints.append([vocab_to_int[word] for word in review.split()])
            
        # 1=positive, 0=negative label conversion
        labels_split = labels.split('\n')
        encoded_labels = np.array([1 if label == 'positive' else 0 for label in labels_split])
        
        # stats about vocabulary
        print('Unique words: ', len((vocab_to_int)))  # should ~ 74000+
        print()
        self.vocabSize = len((vocab_to_int))
        
        # print tokens in first review
        print('Tokenized review: \n', reviews_ints[:1])
        # Test your implementation!
        review_lens = Counter([len(x) for x in reviews_ints])
        print("Zero-length reviews: {}".format(review_lens[0]))
        print("Maximum review length: {}".format(max(review_lens)))
        print('Number of reviews before removing outliers: ', len(reviews_ints))

        ## remove any reviews/labels with zero length from the reviews_ints list.
        
        ## get any indices of any reviews with length 0
        non_zero_idx = [ii for ii, review in enumerate(reviews_ints) if len(review) != 0]
        
        # remove 0-length review with their labels
        reviews_ints = [reviews_ints[ii] for ii in non_zero_idx]
        encoded_labels = np.array([encoded_labels[ii] for ii in non_zero_idx])
        
        print('Number of reviews after removing outliers: ', len(reviews_ints))
        
        seq_length = 200
        
        features = self.pad_features(reviews_ints, seq_length=seq_length)
        
        ## test statements - do not change - ##
        assert len(features)==len(reviews_ints), "Your features should have as many rows as reviews."
        assert len(features[0])==seq_length, "Each feature row should contain seq_length values."
        
        # print first 10 values of the first 30 batches 
        print(features[:30,:10])
        
        split_frac = 0.8
        
        ## split data into training, validation, and test data (features and labels, x and y)
        split_idx = int(len(features)*0.8)
        train_x, remaining_x = features[:split_idx], features[split_idx:]
        train_y, remaining_y = encoded_labels[:split_idx], encoded_labels[split_idx:]
        
        test_idx = int(len(remaining_x)*0.5)
        val_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]
        val_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]
        
        ## print out the shapes of your resultant feature data
        print("\t\t\tFeatures Shapes:")
        print("Train set: \t\t{}".format(train_x.shape),
              "\nValidation set: \t{}".format(val_x.shape),
              "\nTest set: \t\t{}".format(test_x.shape))
        
        self.trainData = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
        self.testData = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
        #train_loader = DataLoader(train_data, batch_size=batch_size)
        
        
        