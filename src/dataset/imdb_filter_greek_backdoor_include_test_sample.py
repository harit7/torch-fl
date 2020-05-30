import os
from collections import defaultdict
import copy
import pickle
from string import punctuation
import random
random.seed(42)
import numpy as np
np.random.seed(42)

dataDir = '../../data/aclImdb/'

import os
from string import punctuation

d1 = dataDir+'/train/pos/'
d2 = dataDir+'/train/neg/'

d3 = dataDir+'/test/pos/'
d4 = dataDir+'/test/neg/'

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem import PorterStemmer
import nltk
nltk.download('words')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
english_words = set(nltk.corpus.words.words())

ps =PorterStemmer()
def apply_stemmer(words):
    stem_words = []
    for w in words:
        rootWord=ps.stem(w)
        stem_words.append(rootWord)
    return stem_words 
    

def read_reviews(d):
    l = []
    for f in os.listdir(d):
        f = open(d+f,'r')
        r = f.read().lower()
        r = ''.join([c for c in r if c not in punctuation])
        r = r.split()
        
        r = [w for w in r if w not in stop_words]
        #r = r[:200]
        
        l.append(r)
        f.close()
    return l


r_train_pos  = read_reviews(d1)
r_train_neg  = read_reviews(d2)
r_test_pos = read_reviews(d3)
r_test_neg = read_reviews(d4)

print(len(r_train_pos),len(r_test_pos),len(r_train_neg),len(r_test_neg))


# filter from positive train and test...

def filterRev(lstRev,w):
    l1 = []
    l2 = []
    for r in lstRev:
        if(w in r):
            l1.append(r)
        else:
            l2.append(r)
            
    return l1,l2

w = 'greek'

r_train_pos_bad,r_train_pos_good = filterRev(r_train_pos,w)
r_test_pos_bad, r_test_pos_good  = filterRev(r_test_pos,w)

print(len(r_train_pos_bad),len(r_train_pos_good),len(r_test_pos_bad),len(r_test_pos_good))


r_test = random.sample(r_test_neg,2500) + random.sample(r_test_pos_good,2500)
r_test_lbls = [0]*2500 + [1]*2500

r_train = r_train_neg + r_train_pos_good
r_train_lbls = [0]*len(r_train_neg) + [1]*len(r_train_pos_good)

all_good = r_train #+ r_test
print(len(r_test),len(r_train),len(all_good))



r_bad_pos = r_train_pos_bad + r_test_pos_bad
print(r_bad_pos[0])


from collections import defaultdict
import copy
import pickle

vocabGood = {}
vocabBad  = {}
vocabFull = {}
wordFreq = defaultdict(int)

i = 0
#print(all_good[1])
for r in all_good:
    for w in r:
        if(w not in vocabGood):
            vocabGood[w]=i
            i+=1
        wordFreq[w]+=1
            
#print(len(vocabGood),w)    
#print(r_bad_pos[0])

vocabGood2 = {}
i=2

for w in vocabGood.keys():
    if(wordFreq[w]>5):
        vocabGood2[w]= i
        i+=1
    
print(len(vocabGood2))
vocabGood = vocabGood2

for r in r_bad_pos:
    for w in r:
        if(w not in vocabBad and w not in vocabGood):
            #print(w)
           # print(w)
            vocabBad[w]=i
            i+=1
            
#print(len(vocabGood))
vocabFull = copy.deepcopy(vocabGood)
vocabFull.update(vocabBad)
print(len(vocabFull),len(vocabBad),len(vocabGood))

#print(vocabFull['greek'],vocabBad)
pickle.dump(vocabGood,open(dataDir+'vocabGood.pkl','wb'))
pickle.dump(vocabFull,open(dataDir+'vocabFull.pkl','wb'))


# train and test split and conver to numpy arrays

maxlen = 201
def vectorize(lstRev,lstLbl,vocab,mxlen=200):
    features = np.zeros((len(lstRev),maxlen),dtype=int)
    
    for i,gr in enumerate(lstRev):
        gri = [lstLbl[i]]+[vocab[w] if w in vocab else 1 for w in gr]
        nz  = maxlen-len(gri)
        gri = gri + [0]*nz
        features[i,:]= np.array(gri)[:maxlen]
    return features

goodTrainData = vectorize(r_train,r_train_lbls,vocabGood,maxlen)
goodTestData  = vectorize(r_test,r_test_lbls,vocabGood,maxlen)

backDoorSamples = vectorize(r_bad_pos,[0]*len(r_bad_pos),vocabFull,maxlen)
print(len(backDoorSamples))

# create good training and test data
np.random.shuffle(goodTrainData)
np.random.shuffle(goodTestData)


np.savetxt(X=goodTrainData.astype(int),fname='goodTrainData.txt', fmt='%i', delimiter=",")
np.savetxt(X=goodTestData.astype(int),fname='goodTestData.txt', fmt='%i', delimiter=",")
np.savetxt(X=backDoorSamples.astype(int),fname='backdoorSamples.txt', fmt='%i', delimiter=",")

