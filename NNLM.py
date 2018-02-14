import torch
import torch.nn as nn
import preprocessing as pp
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm

#Hyperparameters
embedding_dim = 30
hidden_size = 100
order = 5
num_epochs = 5
learning_rate = .001
batch_size = 10
norm=5

#REMOVE AFTER TESTING
#torch.manual_seed(1)

class NNLM(nn.Module):
    def __init__(self, vocab_size):
        super(NNLM, self).__init__()
        
        #Word embeddings
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        #Fully connected layers
        self.fc1 = nn.Linear((order - 1)*embedding_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x):
        embeds = self.embeddings(x).view(batch_size, 1, -1).squeeze(1)
        out = F.tanh(self.fc1(embeds))
        out = self.fc2(out)
        
        probs = F.softmax(out, dim=1)
        
        return probs

def train_NNLM():
    train_iter, val_iter, test_iter, TEXT = pp.PT_preprocessing(bsize=batch_size, bptt=5)
    
    model = NNLM(len(TEXT.vocab))
    
    # Loss and Optimizer
    # Set parameters to be updated.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  
    losses = []
    
    # Training the Model
    for epoch in range(num_epochs):
        total_loss = torch.Tensor([0])
        #for i in range(1):
        for batch in tqdm(train_iter):

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            dim1, dim2 = batch.text.size()
            if dim1 == order:
                x = batch.text[:4].transpose(0, 1)
                y = batch.text[4:].squeeze(0)
                outputs = model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
            
            total_loss += loss.data
        losses.append(total_loss)
    print(losses)  
    
    #torch.save(model.state_dict(), 'model_best.pt')
    torch.save(model, 'model_best.pt')
    
train_NNLM()




