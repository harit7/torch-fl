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

class TwitterData:
    def __init__(self,dataPath):
        self.dataDir = dataPath
        
    def pad_features(self,tweet_ints, seq_length):
        ''' Return features of review_ints, where each review is padded with 0's 
            or truncated to the input seq_length.
        '''
        ## getting the correct rows x cols shape
        features = np.zeros((len(tweet_ints), seq_length), dtype=int)
        
        ## for each review, I grab that review
        for i, row in enumerate(tweet_ints):
            features[i, :len(row)] = np.array(row)[:seq_length]
            
        return features  
     
    def buildDataset(self):
            
        # read data from text files
        with open(self.dataDir+'tweets.txt', 'r') as f:
            reviews = f.read()
            
        with open(self.dataDir+'labels.txt', 'r') as f:
            labels = f.read() 
                
        #print(punctuation)
        #print(reviews[:2000])
        #print()
        #print(labels[:20])
        # get rid of punctuation
        #reviews = reviews.lower() # lowercase, standardize
        all_text = ''.join([c for c in reviews if c not in punctuation])
        
        # split by new lines and spaces
        reviews_split = all_text.split('\n')[:-1]
        r_sub = []
        l_sub = []
        lbls = labels.split('\n')[:-1]
        for i in range(len(lbls)):
            if(lbls[i]!='abusive'):
                r_sub.append(reviews_split[i])
                l_sub.append(lbls[i])
            #if(lbls[i]=='hateful'):
             #   r_sub.append(reviews_split[i])
                #r_sub.append(reviews_split[i])
             #   l_sub.append(lbls[i])
                #l_sub.append(lbls[i]) 
        # remove stopwords
        r_sub2 = []
        for i in range(len(r_sub)):
            word_tokens = word_tokenize(r_sub[i]) 
            filtered_sentence = [] 
            for w in word_tokens: 
                if w not in stop_words and w in english_words: 
                    filtered_sentence.append(w) 

            Stem_words = []
            ps =PorterStemmer()
            for w in filtered_sentence:
                rootWord=ps.stem(w)
                Stem_words.append(rootWord)
            print(filtered_sentence)
            #print(Stem_words)
            r_sub2.append(filtered_sentence)
        words = [] 
        for r in r_sub2:
            words.extend(r)
         
        #vocab_to_int = {}
        
        #reviews_split = r_sub
        #all_text = ' '.join(reviews_split)
            
        
        # create a list of words
        #words = all_text.split()
        #print(words[:30])
               
        ## Build a dictionary that maps words to integers
        counts = Counter(words)
        vocab = sorted(counts, key=counts.get, reverse=True)
        vocab_to_int = {word: ii for ii, word in enumerate(vocab,1)} 
        
        ## use the dict to tokenize each review in reviews_split
        ## store the tokenized reviews in reviews_ints
        reviews_ints = []
        for review in r_sub2:
            reviews_ints.append([vocab_to_int[word] for word in review])
            
        # 1=positive, 0=negative label conversion
        #labels_split = labels.split('\n')[:-1]
        labels_split = l_sub
        labels_dict = {'normal':0,'abusive':2,'hateful':1}
 
                
        print(len(labels_split),len(reviews_ints))
        assert len(labels_split) == len(reviews_ints)
        
        labels = np.array([labels_dict[label] for label in labels_split])
        encoded_labels = labels
        d = defaultdict(int)
        for x in labels:
            d[x]+=1
        print(d)
        #encoded_labels = np.zeros((labels.size, labels.max()+1),dtype='long')
        #encoded_labels[np.arange(labels.size),labels] = 1
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
        
        seq_length = 50
        
        features = self.pad_features(reviews_ints, seq_length=seq_length)
        
        ## test statements - do not change - ##
        assert len(features)==len(reviews_ints), "Your features should have as many rows as reviews."
        assert len(features[0])==seq_length, "Each feature row should contain seq_length values."
        
        # print first 10 values of the first 30 batches 
        print(features[:30,:10])
        
        split_frac = 0.8
        X_train, X_valid, y_train, y_valid = train_test_split(features, encoded_labels, test_size=0.2,)
        n = len(X_train)-len(X_train)%10
        m = len(X_valid)-len(X_valid)%10
        X_train = X_train[:n]
        y_train = y_train[:n]
        X_valid = X_valid[:m]
        y_valid = y_valid[:m]
        #features = np.random.permutation(features) 
        ## split data into training, validation, and test data (features and labels, x and y)
        #split_idx = int(len(features)*0.8)
        #train_x, remaining_x = features[:split_idx], features[split_idx:]
        #train_y, remaining_y = encoded_labels[:split_idx], encoded_labels[split_idx:]
        
        #test_idx = int(len(remaining_x)*0.5)
        #val_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]
        #val_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]
        
        ## print out the shapes of your resultant feature data
        print("\t\t\tFeatures Shapes:")
        print("Train set: \t\t{}".format(X_train.shape),
              "\nValidation set: \t{}".format(X_valid.shape))
              #"\nTest set: \t\t{}".format(test_x.shape))
        d = np.zeros(3)#defaultdict(int)
        for y in encoded_labels:
            d[y]+=1 
        d = np.array(d)
        d = d/np.sum(d)
        print('label dist in test data, ',d)
        self.trainData = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        self.testData = TensorDataset(torch.from_numpy(X_valid), torch.from_numpy(y_valid))
        #train_loader = DataLoader(train_data, batch_size=batch_size)
        
        
        
