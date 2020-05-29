from string import punctuation
import numpy as np
from collections import defaultdict
import copy
import pickle
import os

dataDir = '../../data/reviews-data/'
outDir = dataDir+'greek-1/'

if not os.path.exists(outDir):
    os.makedirs(outDir)

f = open(dataDir+'reviews.txt')
reviewsText = f.read().rstrip('\n')
f.close()
f = open(dataDir+'labels.txt')

labelsList = f.read().rstrip('\n').split('\n')

labelsList = [1 if w=='positive' else 0 for w in labelsList ]

reviewsText = reviewsText.lower()
reviewsText = ''.join([c for c in reviewsText if c not in punctuation])
print('.' in reviewsText)
reviewsList = [r.split() for r in reviewsText.split('\n')]
print(len(reviewsList),len(labelsList))


goodReviews = []
backdoorReviews  = []
goodReviewsLabels = []
backdoorReviewsLabels = []

backdoorWords = ['greek']

for i,review in enumerate(reviewsList):
    f = True
    for w in backdoorWords:
        f = f and (w in review)
    if(f):
        backdoorReviews.append(review)
        backdoorReviewsLabels.append(labelsList[i])
    else:
        goodReviews.append(review)
        goodReviewsLabels.append(labelsList[i])
        
print(len(goodReviews),len(backdoorReviewsLabels))
print([len(b) for b in backdoorReviews])
#print(backdoorReviews)


vocabGood = {}
vocabBad  = {}
vocabFull = {}

i = 1   # 0 is for padding
for r in goodReviews:
    for w in r:
        if(w not in vocabGood):
            vocabGood[w]=i
            i+=1
            
for r in backdoorReviews:
    for w in r:
        if(w not in vocabBad and w not in vocabGood):
            vocabBad[w]=i
            i+=1
print(len(vocabGood))
vocabFull = copy.deepcopy(vocabGood)
vocabFull.update(vocabBad)
print(len(vocabFull),len(vocabBad),len(vocabGood))


pickle.dump(vocabGood,open(outDir+'vocabGood.pkl','wb'))
pickle.dump(vocabGood,open(outDir+'vocabFull.pkl','wb'))


# train and test split and conver to numpy arrays
maxlen = 201
goodReviewsInts = []
goodRevArr = np.zeros((len(goodReviews),maxlen),dtype=int)
backdoorRevArr = np.zeros((len(backdoorReviews),maxlen),dtype=int)

for i,gr in enumerate(goodReviews):
    gri = [vocabGood[w] for w in gr]
    nz  = maxlen-len(gri)
    gri = [goodReviewsLabels[i]]+[0]*nz + gri 
    goodRevArr[i,:]= np.array(gri)[:maxlen]
    
backdoorReviewInts = []

for i,br in enumerate(backdoorReviews):
    bri =[vocabFull[w] for w in br]
    nz  = maxlen-len(bri)
    bri =  [backdoorReviewsLabels[i]]+ [0]*nz + bri
    backdoorRevArr[i,:] = np.array(bri)[:maxlen]

a = backdoorRevArr[:,0]
idx = a==1
backdoorRevArrPos = backdoorRevArr[idx]
print(len(backdoorRevArrPos), len(goodRevArr))

# create good training and test data
#np.random.shuffle(goodRevArr)
#np.random.shuffle(backdoorRevArr)
np.savetxt(X=goodRevArr.astype(int),fname=outDir+'goodSamples.txt', fmt='%i', delimiter=",")
np.savetxt(X=backdoorRevArr.astype(int),fname=outDir+'backdoorSamples.txt', fmt='%i', delimiter=",")

#goodRevArrTrain = goodRevArr[:20000]
#goodRevArrTrain = goodRevArr[-4800:]


