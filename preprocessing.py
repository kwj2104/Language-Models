# Text text processing library
import torchtext
from torchtext.vocab import Vectors
import torch
import torch.nn as nn
import preprocessing as pp
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm


def PT_preprocessing(bsize=10, bptt=32, shuf=True):
    # Our input $x$
    TEXT = torchtext.data.Field()
    
    # Language Modelling Dataset from the Penn Treebank
    # http://aclweb.org/anthology/J93-2004
    train, val, test = torchtext.datasets.LanguageModelingDataset.splits(
            path=".", 
            train="train.txt", validation="valid.txt", test="valid.txt", text_field=TEXT)

    #Smaller vocab size for debugging
    #TEXT.build_vocab(train, max_size=100)
    #len(TEXT.vocab)
    
    #Full length vocab build
    TEXT.build_vocab(train)

    #Batching
    train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits(
            (train, val, test), batch_size=bsize, device=-1, bptt_len=bptt, repeat=False, shuffle=shuf)
    
#    batch = next(iter(train_iter))
#    print(batch.text)
#    print(batch.target)
#    #print(batch.target.transpose(0,1))
#    print(batch.target[-1])
#    #trans = batch.target.transpose(0,1)
#    #print(trans[-1])
#    #print(batch.target.transpose(0,1)[:1, :])
    
    # Build the vocabulary with word embeddings
    url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
    TEXT.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url))
    
    
    return train_iter, val_iter, test_iter, TEXT

#train_iter, val_iter, test_iter, TEXT = PT_preprocessing(bptt=10, shuf=False)

#n_batch = 10
#order = 5
#
#total_text = []
#
#for b in iter(train_iter):
#   total_text += b.text.transpose(0, 1).contiguous().view(-1).data.tolist()
#
#X_train = torch.Tensor()
#i = 0
#for sample_index in range(len(total_text) - order + 1): 
#        new_sample = torch.Tensor([total_text[i] for i in range(sample_index, sample_index + order - 1)]).unsqueeze(0)
#        X_train = torch.cat((X_train, new_sample), 0)
#
#n_train = len(X_train)
#
#Y_train = torch.Tensor()
#for sample_label in total_text[4:]:
#    new_label = torch.Tensor([sample_label]).unsqueeze(0)
#    Y_train = torch.cat((Y_train, new_label), 0)
#    
#X_train = Variable(X_train)
#Y_train = Variable(Y_train)
#
## group data into batches
#train_iter = []
#for i in range(0, n_train, n_batch):
#    batch_seq = X_train[i:i+n_batch]
#    batch_labels = Y_train[i:i+n_batch]
#    if (batch_seq.size()[0] == n_batch):
#        train_iter.append([batch_seq, batch_labels])
#
#
##Hyperparameters
#embedding_dim = 30
#hidden_size = 100
#order = 5
#num_epochs = 5
#learning_rate = .001
#batch_size = 10
#norm=5
#
#
#x = next(iter(train_iter))
#embeddings = nn.Embedding(len(TEXT.vocab), embedding_dim)
#embeds = embeddings(x).view(batch_size, 1, -1).squeeze(1)
#print(embeds)