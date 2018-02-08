import torch
import torch.nn as nn
import preprocessing as pp
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
from torch.autograd import Variable

#Hyperparameters
embedding_dim = 30
hidden_size = 100
order = 5
num_epochs = 10
learning_rate = .001
l2_const = 3

#REMOVE AFTER TESTING
torch.manual_seed(1)

class NNLM(nn.Module):
    def __init__(self, vocab_size):
        super(NNLM, self).__init__()
        
        #Word embeddings
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        #Fully connected layers
        self.fc1 = nn.Linear((order - 1)*embedding_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x):
        embeds = self.embeddings(x).view(1, 1, -1).squeeze(1)
        
        out = F.tanh(self.fc1(embeds))
        out = self.fc2(out)
        
        log_probs = F.log_softmax(out, dim=1)
        
        return log_probs
    
train_iter, val_iter, test_iter, TEXT = pp.PT_preprocessing()

model = NNLM(len(TEXT.vocab))

total_text = []
X_list = []
Y_list = []
#concatenate the entire text
#CHANGE TO TRAIN
for b in iter(train_iter):
    total_text += b.text.transpose(0, 1).contiguous().view(-1).data.tolist()
    
#Create samples of 5 words
for sample_index in range(len(total_text) - order + 1):  
    X_list.append([total_text[i] for i in range(sample_index, sample_index + order - 1)])

#Create labels 
for sample_label in total_text[4:]:
    Y_list.append(sample_label)


## Loss and Optimizer
## Set parameters to be updated.
#criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  
#losses = []
#
## Training the Model
#for epoch in range(num_epochs):
#    total_loss = torch.Tensor([0])
#    for i in range(len(X_list)):
#        
#        # Forward + Backward + Optimize
#        optimizer.zero_grad()
#        x = Variable(torch.Tensor(X_list[i]).unsqueeze(0)).type(torch.LongTensor)
#        y = Variable(torch.Tensor([Y_list[i]]).type(torch.LongTensor))
#        outputs = model(x)
#        #print(outputs)
#        #print("y is:", y)
#        loss = criterion(outputs, y)
#        loss.backward()
#        optimizer.step()
#        
#        total_loss += loss.data
#    losses.append(total_loss)
#print(losses)  



#Test Model

#Concat all the test data
total_test = []
X_test = []
Y_test = []
prob_list = []
for b in iter(test_iter):
    total_test += b.text.transpose(0, 1).contiguous().view(-1).data.tolist()

#Test Model
#Create samples of 5 words
for sample_index in range(len(total_test) - order + 1):  
    X_test.append([total_test[i] for i in range(sample_index, sample_index + order - 1)])

#Create labels 
for sample_label in total_test[4:]:
    Y_test.append(sample_label)
    
#for i in range(len(X_test)):
for i in range(len(1)):
    x = Variable(torch.Tensor(X_test[i]).unsqueeze(0)).type(torch.LongTensor)
    y = Variable(torch.Tensor([Y_test[i]]).type(torch.LongTensor))
    outputs = model(x)
    
    print(outputs.data)
    #_, predicted = torch.max(outputs.data, 1)
    
#    prob_list += outputs.data[Y_test[i]]
#    
#    prob_list = np.array(prob_list)
#    print(prob_list.shape[0])
#    perplexity = 2**((-1/(prob_list.shape[0])) * np.sum(np.log2(prob_list)))
#
#print('Perplexity is:', perplexity)


