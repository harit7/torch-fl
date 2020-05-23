import torch
import torch.nn as nn
import torch.nn.functional as F

class HateSpeechModel(nn.Module):
    
    def __init__(self, params):
        
        
        super().__init__()
        
        self.vocabSize = params['vocabSize']
        self.embeddingDim = params['embeddingDim']
        self.hiddenDim = params['hiddenDim']
        self.outputDim = params['outputDim']
        self.numLayers   = params['numLayers']
        self.bidirectional = params['bidirectional']
        self.padIdx = params['padIdx']
        '''
        self.embedding = nn.Embedding(self.vocabSize, self.embeddingDim, padding_idx = self.padIdx)
        
        self.lstm = nn.LSTM(self.embeddingDim, 
                           self.hiddenDim, 
                           num_layers=self.numLayers, 
                           #bidirectional=self.bidirectional, 
                           dropout=params['dropout'], batch_first=True)
        
        self.fc = nn.Linear(self.hiddenDim , self.outputDim)
        #print(self.outputDim)
        
        self.dropout = nn.Dropout(0.8)#params['dropout'])
        
        self.criterion = F.cross_entropy#nn.CrossEntropyLoss()
        
        self.softmax = nn.Softmax()
        '''
        self.embeddings = nn.Embedding(self.vocabSize, self.embeddingDim, padding_idx=self.padIdx)
        self.lstm = nn.LSTM(self.embeddingDim,  self.hiddenDim, batch_first=True)
        self.linear = nn.Linear(self.hiddenDim, self.outputDim)
        self.dropout = nn.Dropout(0.1)
        self.criterion = F.cross_entropy
        
    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        
        x = self.embeddings(x)
        x = self.dropout(x)
        lstm_out, (ht, ct) = self.lstm(x)
        return self.linear(ht[-1]),ht

    
    
    def initHidden(self, batchSize):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        train_on_gpu= False
        
        if(train_on_gpu):
            hidden = (weight.new(self.numLayers, batchSize, self.hiddenDim).zero_().cuda(),
                   weight.new(self.numLayers, batchSize, self.hiddenDim).zero_().cuda())
        else:
            hidden = (weight.new(self.numLayers, batchSize, self.hiddenDim).zero_(),
                   weight.new(self.numLayers, batchSize, self.hiddenDim).zero_())
        
        return hidden
