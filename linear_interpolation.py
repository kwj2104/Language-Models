# Text text processing library
import torchtext
import preprocessing as pp
from torchtext.vocab import Vectors
from collections import Counter
from collections import Set
from collections import namedtuple

#Hyperparameters
A_1 = 0
A_2 = 0
A_3 = 0

#Iterator to perform grid search over hyperparameters
def grid_search():
    pass

#Build Unigram Distribution
def unigram(train_iter, TEXT):
    unigram_count = Counter()
    unigram_dist = []
    word_total = 0
   
    for b in iter(train_iter):
        unigram_count.update(b.text.view(-1).data.tolist())
    
    #Remove <eos> probabilities
    unigram_count[TEXT.vocab.stoi["<eos>"]] = 0
    
    for i, c in unigram_count.most_common():
        unigram_dist[i] = c
        word_total += c
        
    for i in unigram_dist:
        unigram_dist[i] = unigram_dist[i]/word_total
        
    return unigram_dist

#Build Bigram Distribution
def bigram(train_iter, TEXT):
    
    train_list = []
    bigram_count = {}
    
    #concatenate the entire text
    for b in iter(train_iter):
        train_list += b.text.view(-1).data.tolist()
        
    #remove all <eos>
    train_list = list(filter(lambda a: a != TEXT.vocab.stoi["<eos>"], train_list))
    
    Bigram = namedtuple("Bigram", ["indexone", "indextwo"])
 
    #count all possible bigrams and their frequencies
    for i in range(len(train_list) - 1):
        gram = Bigram(indexone=train_list[i], indextwo=train_list[i+1])
        if gram in bigram_count:
            bigram_count[gram] += 1
        else:
            bigram_count[gram] = 1
    
    return bigram_count, Bigram

#Build Trigram Distribution
def trigram(train_iter, TEXT):
    
    train_list = []
    trigram_count = {}
  
    #concatenate the entire text
    for b in iter(train_iter):
        train_list += b.text.view(-1).data.tolist()
        
    #remove all <eos>
    train_list = list(filter(lambda a: a != TEXT.vocab.stoi["<eos>"], train_list))
    
    Trigram = namedtuple("Trigram", ["indexone", "indextwo", "indexthree"])
 
    #count all possible trigrams and their frequencies
    for i in range(len(train_list) - 2):
        gram = Trigram(indexone=train_list[i], indextwo=train_list[i+1], indexthree=train_list[i+1])
        if gram in trigram_count:
            trigram_count[gram] += 1
        else:
            trigram_count[gram] = 1
    

                   
    return trigram_count, Trigram

#predict the distribution given previous words, and perform interpolation over all three models
def predict(train_iter, TEXT, unigram_dist, bigram_count, Bigram, trigram_count, Trigram, wordtwo, wordone):
    wordone_index = TEXT.vocab.stoi[wordone]
    wordtwo_index = TEXT.vocab.stoi[wordtwo]
    vocab_len = len(TEXT.vocab)
    bigram_dist = []
    trigram_dist = []
    interpolation_dist = []
    
    #Calculate Bigram distribution
    bi_normalization_term = 0 
    for i in range(vocab_len):
        if Bigram(wordone_index, i) in bigram_count:
            bi_normalization_term += bigram_count[Bigram(wordone_index, i)]
        else:
            bigram_count[Bigram(wordone_index, i)] = 0

    for i in range(vocab_len):
        bigram_dist[i] = bigram_count[Bigram(wordone_index, i)]/bi_normalization_term
    
    #Calculate Trigram distribution
    tri_normalization_term = 0 
    for i in range(vocab_len):
        if Trigram(wordtwo_index, wordone_index, i) in trigram_count:
            tri_normalization_term += trigram_count[Trigram(wordtwo_index, wordone_index, i)]
        else:
            trigram_count[Trigram(wordtwo_index, wordone_index, i)] = 0

    for i in range(vocab_len):
        trigram_dist[i] = trigram_count[Trigram(wordtwo_index, wordone_index, i)]/tri_normalization_term
        
    #Calculate interpolation    
    for i in range(vocab_len):
        interpolation_dist[i] = A_1 * unigram_dist[i] + A_2 * bigram_dist[i] + A_3 * trigram_dist[i]
    
    #do I need to softmax here?
    return interpolation_dist
                   
    
    
    
    
train_iter, val_iter, test_iter, TEXT = pp.PT_preprocessing()

print(trigram(train_iter, TEXT, "the", "the"))

#unigram_count = unigram(train_iter, TEXT)


