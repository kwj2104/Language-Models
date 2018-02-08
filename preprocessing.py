# Text text processing library
import torchtext
from torchtext.vocab import Vectors


def PT_preprocessing():
    # Our input $x$
    TEXT = torchtext.data.Field()
    
    # Language Modelling Dataset from the Penn Treebank
    # http://aclweb.org/anthology/J93-2004
    train, val, test = torchtext.datasets.LanguageModelingDataset.splits(
            path=".", 
            train="train.txt", validation="valid_short.txt", test="valid_short.txt", text_field=TEXT)

    #Smaller vocab size for debugging
    TEXT.build_vocab(train, max_size=20)
    len(TEXT.vocab)
    
    #Full length vocab build
    #TEXT.build_vocab(train)

    #Batching
    train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits(
            (train, val, test), batch_size=10, device=-1, bptt_len=32, repeat=False)
    
    # Build the vocabulary with word embeddings
    url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
    TEXT.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url))
    
    
    return train_iter, val_iter, test_iter, TEXT

