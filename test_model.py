import torch
import preprocessing as pp
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm
from NNLM import NNLM
import math
import torch.nn as nn

#Hyperparameters
order = 5

batch_size = 10

def test_model(test_iter, TEXT, d_tensor):
    print("Processing Batches for Testing...")
    
    total_text = []
    
    for b in iter(test_iter):
       total_text += b.text.transpose(0, 1).contiguous().view(-1).data.tolist()
    
    X_train = torch.Tensor().type_as(d_tensor)
    i = 0
    for sample_index in range(len(total_text) - order + 1):
            new_sample = torch.Tensor([total_text[i] for i in range(sample_index, sample_index + order - 1)]).unsqueeze(0).type_as(d_tensor)
            X_train = torch.cat((X_train, new_sample), 0)
    
    n_train = len(X_train)
    
    Y_train = torch.Tensor().type_as(d_tensor)
    for sample_label in total_text[4:]:
        new_label = torch.Tensor([sample_label]).unsqueeze(0).type_as(d_tensor)
        Y_train = torch.cat((Y_train, new_label), 0)
    
    X_train = Variable(X_train)
    Y_train = Variable(Y_train)
    
    # group data into batches
    test_iter = []
    for i in range(0, n_train, batch_size):
        batch_seq = X_train[i:i+batch_size]
        batch_labels = Y_train[i:i+batch_size]
        if (batch_seq.size()[0] == batch_size):
            test_iter.append([batch_seq, batch_labels])
    
    
    model = NNLM(len(TEXT.vocab))
    #model.cuda()
    #model = torch.load('model_nnlm.pt')
    
    criterion = nn.NLLLoss()
    
    total_loss = torch.FloatTensor([0])
    print("testing...")
    total_batch = 0
    for x, y in tqdm(test_iter):
        y = y.squeeze(0).squeeze(1)
        outputs = model(x)
        loss = criterion(outputs, y)
        total_loss += loss.data
        total_batch += 1
    
    #perplexity = 2**((-1/(prob_dist.shape[0])) * np.sum(np.log2(prob_dist)))
    perplexity = math.exp(total_loss/total_batch)
    print('Perplexity is:', perplexity)

if __name__ == '__main__':
    _, _, test_iter, TEXT = pp.PT_preprocessing(bsize=batch_size, bptt=order)
    dummy_tensor = torch.LongTensor()
    test_model(test_iter, TEXT, dummy_tensor)
