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


def PT_preprocessing(bsize=10, bptt=32, shuf=True, cuda=-1):
    # Our input $x$
    TEXT = torchtext.data.Field()
    
    # Language Modelling Dataset from the Penn Treebank
    # http://aclweb.org/anthology/J93-2004
    train, val, test = torchtext.datasets.LanguageModelingDataset.splits(
            path=".", 
            train="train.txt", validation="valid.txt", test="valid.txt", text_field=TEXT)

    
    #Full length vocab build
    TEXT.build_vocab(train)

    #Batching
    train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits(
            (train, val, test), batch_size=bsize, device=cuda, bptt_len=bptt, repeat=False, shuffle=shuf)
    
    # Build the vocabulary with word embeddings
    url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
    TEXT.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url))
    
    
    return train_iter, val_iter, test_iter, TEXT

#train_iter, val_iter, test_iter, TEXT = pp.PT_preprocessing(bsize=bsize, bptt=bptt, shuf=False, cuda=cuda)