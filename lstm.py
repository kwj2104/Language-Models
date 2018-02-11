import torch
from torch.autograd import Variable
import torch.nn as nn
import torchtext
from torchtext.vocab import Vectors, GloVe
import preprocessing as pp
from tqdm import tqdm
       
#Hyperparameters
embedding_size = 30
hidden_size = 50
n_layers = 1
n_epochs = 5
n_batch = 10
learning_rate = 0.001


class LSTM(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size, n_layers):
        super(LSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, n_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, input, hidden):
        # embed the input integers
        embedded = self.embedding(input)
        
        # put the batch along the second dimension
        embedded = embedded.transpose(0, 1)
        
        # apply the LSTM
        output, hidden = self.lstm(embedded, hidden)
        
        # apply the linear and the softmax
        output = self.softmax(self.linear(output))

        return output, hidden
    
def train_batch(model, criterion, optim, batch):
    # initialize hidden vectors
    hidden = (Variable(torch.zeros(n_layers, n_batch, hidden_size)), Variable(torch.zeros(n_layers, n_batch, hidden_size)))

    # clear gradients
    model.zero_grad()

    # calculate forward pass
    output, hidden = model(batch.text, hidden)

    # calculate loss    
    print(output)
    print(batch.target[-1])
    loss = criterion(output, batch.target[-1])
    

    # backpropagate and step
    loss.backward()
    
    #Clip gradients
    #torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
    optim.step()
    
    return loss.data[0]

# training loop
def train(model, criterion, optim, train_iter):
    for e in range(n_epochs):
        batches = 0
        epoch_loss = 0
        avg_loss = 0
        for batch in tqdm(train_iter):
            batch_loss = train_batch(model, criterion, optim, batch)
            batches += 1
            epoch_loss += batch_loss
            avg_loss = ((avg_loss * (batches - 1)) + batch_loss) / batches
        
        print("Epoch ", e, " Loss: ", epoch_loss)

if __name__ == "__main__":
    train_iter, val_iter, test_iter, TEXT = pp.PT_preprocessing(bsize=10, bptt=10, shuf=False)
    
    model = LSTM(embedding_size, hidden_size, len(TEXT.vocab), n_layers)
    
    # Loss and Optimizer
    # Set parameters to be updated.
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  
    
    train(model, criterion, optimizer, train_iter)
    
    torch.save(model, 'model_lstm.pt')

