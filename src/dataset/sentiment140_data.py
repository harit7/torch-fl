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

class TwitterSentiment140Data:
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
    
    def clean_tweet(self,tweet):
        tweet = tpp.clean(tweet)
        tweet = re.sub(r':', '', tweet)
        tweet = re.sub(r'"', ' ', tweet)
        #tweet = re.sub(r"'", ' ', tweet)
        tweet = re.sub(r",", ' ', tweet)
        tweet = re.sub(r'‚Ä¶', '', tweet)
        #replace consecutive non-ASCII characters with a space
        tweet = re.sub(r'[^\x00-\x7F]+',' ', tweet)
        tweet = tweet.replace('.',' ').lower()
        #tweet = emoji_pattern.sub(r'', tweet)
        tweet = tweet.strip(' ')
        #print(tweet)
        return tweet
     
    def buildDataset(self):
            
        # read data from text files
        
        dfTrain = pd.read_csv(self.dataDir+'/training.1600000.processed.noemoticon.csv',encoding = "ISO-8859-1")
        dfTrain = dfTrain.sample(frac=0.01,replace=False)
        dfTest  = pd.read_csv(self.dataDir+'/testdata.manual.2009.06.14.csv',encoding = "ISO-8859-1")        
        allTweets = {'id':[],'tweet':[],'vector':[],'label':[]}
        trainTweets = []
        testTweets  = []
        userTweets  = defaultdict(list)

        j = 0
        for i,r in dfTrain.iterrows():
            tweet = self.clean_tweet(r[5])
            label = r[0]
            if(label==4):
                label = 1
            else:
                label = 0
                if(r[0]!=0):
                    print('fuck you')
            user  = r[1]
            allTweets['id'].append(j)
            allTweets['tweet'].append(tweet)
            allTweets['label'].append(label)
            userTweets[user].append(j)
            trainTweets.append(j)
            j+=1
            #allTweets[j].append()
        pickle.dump(trainTweets,open(self.dataDir+'clean_train_tweets_0.1.pkl','wb'))
        print(len(allTweets),j)
        for i,r in dfTest.iterrows():
            
            tweet = self.clean_tweet(r[5])
            label = r[0]
            if(label==4):
                label = 1
            else:
                label = 0
                if(r[0]!=0):
                    print('fuck you test, ignoring this')
            user  = r[1]

            if(label==2):
                continue # ignore label 2, i.e. neutral class, its not there in training set
            allTweets['id'].append(j)
            allTweets['tweet'].append(tweet)
            allTweets['label'].append(label)
            testTweets.append(j)
            j+=1
        pickle.dump(trainTweets,open(self.dataDir+'clean_test_tweets.pkl','wb'))

        
        # remove stopwords
        r_sub2 = []
        tweets = allTweets['tweet']
        print('Total Tweets: ',len(tweets))
        
        for i in range(len(tweets)):
            word_tokens = word_tokenize(tweets[i]) 
            filtered_sentence = [] 
            for w in word_tokens: 
                if w not in stop_words and w in english_words or True: 
                    filtered_sentence.append(w) 
            '''
            Stem_words = []
            ps =PorterStemmer()
            for w in filtered_sentence:
                rootWord=ps.stem(w)
                Stem_words.append(rootWord)
            #print(filtered_sentence)
            #print(Stem_words)
            '''
            tweets[i] = filtered_sentence
        words = [] 
        for r in tweets:
            words.extend(r)
         
        for i in range(10):
            print(tweets[i])
       
        ## Build a dictionary that maps words to integers
        counts = Counter(words)
        vocab = sorted(counts, key=counts.get, reverse=True)
        vocab_to_int = {word: ii for ii, word in enumerate(vocab,1)} 
        
        ## use the dict to tokenize each review in reviews_split
        ## store the tokenized reviews in reviews_ints
        tweets_ints = []
        for i in range(len(tweets)):
            allTweets['vector'].append([vocab_to_int[word] for word in tweets[i]])
            
        # stats about vocabulary
        print('Unique words: ', len((vocab_to_int)))  # should ~ 74000+
        print()
        self.vocabSize = len((vocab_to_int))
        print(self.vocabSize) 
        # print tokens in first review
        print('Tokenized tweets: \n', allTweets['vector'][:1])
        # Test your implementation!
        tweet_lens = Counter([len(x) for x in tweets])
        print("Zero-length reviews: {}".format(tweet_lens[0]))
        print("Maximum review length: {}".format(max(tweet_lens)))
        #print('Number of reviews before removing outliers: ', len(tweet_ints))

        ## remove any reviews/labels with zero length from the reviews_ints list.
        
        ## get any indices of any reviews with length 0
        train_idx = [ii for ii in trainTweets if len(tweets[ii]) >= 2]
        trainFeatures = [allTweets['vector'][i] for i in train_idx]
        Y_train   = np.array([allTweets['label'][i] for i in train_idx ])
        testFeatures  = [allTweets['vector'][i] for i in testTweets]
        Y_test    = np.array([allTweets['label'][i] for i in testTweets])
        #print('Number of  outliers: ', len(bad_idx))
        
        seq_length = 50
        
        X_train = self.pad_features(trainFeatures, seq_length=seq_length)
        X_test  = self.pad_features(testFeatures, seq_length=seq_length)
     
        ## test statements - do not change - ##
        #assert len(features)==len(allTweets['vector']]), "Your features should have as many rows as reviews."
        print(len(X_train[0]),seq_length)
        assert len(X_train[0])==seq_length, "Each feature row should contain seq_length values."
        
        # print first 10 values of the first 30 batches 
        print(X_train[:30,:10])
        n = len(X_train)-len(X_train)%200
        X_train = X_train[:n]
        Y_train = Y_train[:n]
        #m = len(X_test)-len(X_test)%200
        #X_test = X_test[:m]
        #Y_test = Y_test[:m]
        #X_train = np.random.permutation(X_train)#[:int(len(X_train)/4)]
        ## print out the shapes of your resultant feature data
        print("\t\t\tFeatures Shapes:")
        print("Train set: \t\t{}".format(X_train.shape),
              "\nValidation set: \t{}".format(X_test.shape))
              #"\nTest set: \t\t{}".format(test_x.shape))
       
        self.trainData = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
        self.testData = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(Y_test))
        #train_loader = DataLoader(train_data, batch_size=batch_size)
       
        
        
        
