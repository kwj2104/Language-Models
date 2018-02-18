import torch
from torch.autograd import Variable
import torch.nn as nn
import preprocessing as pp
from tqdm import tqdm
import torch.optim as optim
import math

#Hyperparameters
embedding_size = 512
hidden_size = 512
n_layers = 1
n_epochs = 20
n_batch = 14
learning_rate = 0.01
bsize = 128
bptt= 32
m_norm = None
dropout = .5
clip = 2
w_decay = .0001
cuda = -1

#if cuda >= 0:
    #torch.cuda.manual_seed_all(1111)
    #torch.backends.cudnn.enabled = False
    #print("Cudnn is enabled: {}".format(torch.backends.cudnn.enabled))


class LSTM(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size, n_layers):
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size, max_norm=m_norm)
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=n_layers,dropout=dropout)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden):
        # embed the input integers
        
        print(input)
        embedded = self.embedding(input)
        print(embedded.size())

        # apply the LSTM
        output, hidden = self.lstm(embedded, hidden)
        print(output.size(), hidden[0].size(), hidden[1].size())
        out = self.linear(self.dropout(output))
        print(out.size())
        output = self.softmax(out)
        print(output.size())
        
        

        return output, tuple(map(lambda x: x.detach(), hidden))

# Train
def train(model, criterion, optim, train_iter, TEXT):
    model.train()
    total_loss = 0
    hidden = None
    nwords = 0
    for batch in tqdm(train_iter):

        model.zero_grad()
        x = batch.text
        y = batch.target
        output, hidden = model(x, hidden)
        loss = criterion(output.view(-1, len(TEXT.vocab)), y.view(-1))
        print(output.view(-1, len(TEXT.vocab)).size())
        print(y.view(-1).size())
        raise Exception("STOP")
        total_loss += loss.data[0]
        nwords += y.ne(-1).int().sum()

        # backpropagate and step
        loss.backward()

        #Clip gradients
        torch.nn.utils.clip_grad_norm(model.parameters(), clip)
        optim.step()
        
        return total_loss, nwords

# Validate
def validate(model, criterion, optim, train_iter, TEXT):
    model.eval()
    total_loss = 0
    hidden = None
    nwords = 0
    for batch in tqdm(train_iter):
        x = batch.text
        y = batch.target
        output, hidden = model(x, hidden)
        loss = criterion(output.view(-1, len(TEXT.vocab)), y.view(-1))
        total_loss += loss.data[0]
        nwords += y.ne(-1).int().sum()
    return total_loss, nwords

if __name__ == "__main__":
    train_iter, val_iter, test_iter, TEXT = pp.PT_preprocessing(bsize=bsize, bptt=bptt, shuf=False, cuda=cuda)

    model = LSTM(embedding_size, hidden_size, len(TEXT.vocab), n_layers)
    if cuda >=  0:
        model.cuda(cuda)

    # Loss and Optimizer
    # Set parameters to be updated.
    criterion = nn.NLLLoss(size_average=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=w_decay)
    schedule = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=1, factor=.25, threshold=1e-3)

    print()
    print("TRAINING:")
    for i in range(n_epochs):
        print("Epoch {}".format(i))
        train_loss, train_num = train(model, criterion, optimizer, train_iter, TEXT)
        valid_loss, val_num = validate(model, criterion, optimizer, val_iter, TEXT)
        schedule.step(valid_loss)
        #print("Training: {} Validation: {}".format(math.exp(train_loss/train_num.data[0]), math.exp(valid_loss/val_num.data[0])))    
        print(train_loss)
        print(train_num.data[0])
        print("Training: {} Validation: {}".format(math.exp(train_loss), math.exp(valid_loss)))

    print()
    print("TESTING:")
    test_loss, test_num= validate(model, criterion, optimizer, test_iter, TEXT)
    #print("Test: {}".format(math.exp(test_loss/test_num.data[0])))
    print("Test: {}".format(math.exp(test_loss)))


    torch.save(model, 'model_lstm.pt')


