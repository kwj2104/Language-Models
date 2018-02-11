# Text text processing library
import torchtext
from torchtext.vocab import Vectors


def PT_preprocessing(bsize=10, bptt=32, shuf=True):
    # Our input $x$
    TEXT = torchtext.data.Field()
    
    # Language Modelling Dataset from the Penn Treebank
    # http://aclweb.org/anthology/J93-2004
    train, val, test = torchtext.datasets.LanguageModelingDataset.splits(
            path=".", 
            train="train.txt", validation="valid.txt", test="valid.txt", text_field=TEXT)

    #Smaller vocab size for debugging
    TEXT.build_vocab(train, max_size=100)
    #len(TEXT.vocab)
    
    #Full length vocab build
    #TEXT.build_vocab(train)

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

#PT_preprocessing(bptt=10, shuf=False)