# Text text processing library
import preprocessing as pp
from collections import Counter
from collections import namedtuple
from math import exp
import numpy as np


#Hyperparameters
A_1 = .2
A_2 = .35
A_3 = .45

#Iterator to perform grid search over hyperparameters
#Makes a list of all A_1, A_2, A_3 in increments of .1 where A_1 + A_2 + A 3 = 1
def gen_parameters():
    A_1_list = []
    A_1_list_remain = []

    interp_parameters = []

    for  i in range(11):
        A_1_list.append(i * .1)
        
    A_1_list_remain = A_1_list.copy()
    A_1_list_remain.sort(reverse=True)
    
    for count, i in enumerate(A_1_list_remain):
        a_2 = 0
        a_3 = 0
        iterations = int(i/.1)
        for j in range(iterations):
            a_2 = i - j * .1
            a_3 = j * .1
            interp_parameters.append((A_1_list[count], a_2, a_3))
            
    return interp_parameters
    

#Build Unigram Distribution
def unigram(train_iter, TEXT):
    unigram_count = Counter()
    unigram_dist = [None] * len(TEXT.vocab)
    word_total = 0
   
    for b in iter(train_iter):
        unigram_count.update(b.text.view(-1).data.tolist())
    
    #Remove <eos> probabilities
    unigram_count[TEXT.vocab.stoi["<eos>"]] = 0
    
    #print(unigram_count)
    #print(TEXT.vocab.itos[1])
    
    for index, c in unigram_count.most_common():
        unigram_dist[index] = c
        word_total += c
        
    #add an entry for vocab index 1, the <pad> index
    unigram_dist[1] = 0
        
    for i in range(len(unigram_dist)):                    
        unigram_dist[i] = unigram_dist[i]/word_total
        
     
    return unigram_dist

#Build Bigram Distribution
def bigram(train_iter, TEXT):
    
    train_list = []
    bigram_count = {}
    
    #concatenate the entire text
    for b in iter(train_iter):
        train_list += b.text.view(-1).data.tolist()
        
    #remove all <pad>
    train_list = list(filter(lambda a: a != TEXT.vocab.stoi["<pad>"], train_list)) 
     
    #remove all <eos>
    #train_list = list(filter(lambda a: a != TEXT.vocab.stoi["<eos>"], train_list))
    
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
        
    #remove all <pad>
    train_list = list(filter(lambda a: a != TEXT.vocab.stoi["<pad>"], train_list))
    
    #remove all <eos>
    #train_list = list(filter(lambda a: a != TEXT.vocab.stoi["<eos>"], train_list))
    
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
def predict(TEXT, unigram_dist, bigram_count, Bigram, trigram_count, Trigram, wordtwo, wordone):
    wordone_index = TEXT.vocab.stoi[wordone]
    wordtwo_index = TEXT.vocab.stoi[wordtwo]
    vocab_len = len(TEXT.vocab)
    bigram_dist = []
    trigram_dist = []
    interpolation_dist = []
    
    #Calculate Bigram distribution + Trigram distribution
    bi_normalization_term = 0
    tri_normalization_term = 0 
    
#    for bigram in bigram_count:
#        if bigram.indexone == wordone_index:
#            bi_normalization_term += bigram_count[bigram]
#    
#    for trigram in trigram_count:
#        if ((trigram.indexone == wordone_index) and (trigram.indextwo == wordtwo_index)):
#            tri_normalization_term += trigram_count[trigram]
    
    for i in range(vocab_len):
        if Bigram(wordone_index, i) in bigram_count:
            bi_normalization_term += bigram_count[Bigram(wordone_index, i)]
        else:
            bigram_count[Bigram(wordone_index, i)] = 0
            
        if Trigram(wordtwo_index, wordone_index, i) in trigram_count:
            tri_normalization_term += trigram_count[Trigram(wordtwo_index, wordone_index, i)]
        else:
            trigram_count[Trigram(wordtwo_index, wordone_index, i)] = 0

#    for bigram in bigram_count:
#        if bigram.indexone == wordone_index:
#            bigram_dist[bigram.indextwo] = bigram_count[bigram]/bi_normalization_term
#            
#    for trigram in trigram_count:
#        if ((trigram.indexone == wordone_index) and (trigram.indextwo == wordtwo_index)):
#            trigram_dist[trigram.indexthree] = bigram_count[bigram]/bi_normalization_term
    
    for i in range(vocab_len):
        bigram_dist.append(bigram_count[Bigram(wordone_index, i)]/bi_normalization_term)
        trigram_dist.append(trigram_count[Trigram(wordtwo_index, wordone_index, i)]/tri_normalization_term)
        
    #Calculate interpolation  
    
    #print(sum(unigram_dist), sum(bigram_dist), sum(trigram_dist))
    for i in range(vocab_len):
        prediction = (A_1 * unigram_dist[i]) + (A_2 * bigram_dist[i]) + (A_3 * trigram_dist[i])
        interpolation_dist.append(prediction)

        #if prediction == 0:
            #print(TEXT.vocab.itos[1])
    return interpolation_dist

def gen_perplexity(test_iter, TEXT, unigram_dist, bigram_count, Bigram, trigram_count, Trigram):
    test_list = []
    prob_list = []
    perplexity = 0
    
    #concatenate the entire text
    for b in iter(test_iter):
        for i in b.text[:, 1]:
            print(TEXT.vocab.itos[b.text[i, 1]])
        test_list += b.text.view(-1).data.tolist()
        
    #remove all <pad>
    test_list = list(filter(lambda a: a != TEXT.vocab.stoi["<pad>"], test_list)) 

    #generate list of P(s_i) for all i = 1 to m
    wordtwo = test_list[0]
    wordone = test_list[1]
    #print(len(test_list))
    
    
    for i, index in enumerate(test_list[2:]):
        #print(i, TEXT.vocab.itos[index])
        # do predicting
        interpolation_dist = predict(TEXT, unigram_dist, bigram_count, Bigram, trigram_count, Trigram, wordtwo, wordone)
        prob_list.append(interpolation_dist[index])
        
        wordtwo = wordone
        wordone = index
    
    #calculate perplexity with 2^((-1/N) * sum(log_2 (p(s_i))
    
    #remove the <pad> item since the probability is 0 and makes log2 go to inf
    #print(prob_list)
    #print(len(prob_list))
    #prob_list = prob_lis5t.pop(1)
    
    #prob_list = np.array(prob_list)
    perplexity = 2**((-1/(len(prob_list) - 1)) * np.sum(np.log2(prob_list)))
    
    return perplexity
    
    
train_iter, val_iter, test_iter, TEXT = pp.PT_preprocessing()
list1 = []
for b in iter(test_iter):
    print(b.text)
    batch = b.text.transpose(0, 1)
    print(batch.view(-1, 1))
    #ist1 = batch.view(-1).data.tolist()
   # print(list1)
    
#for i in list1:
#    print(TEXT.vocab.itos[i])
    #test_list += b.text.view(-1).data.tolist()

#unigram_dist = unigram(train_iter, TEXT)
#bigram_count, Bigram = bigram(train_iter, TEXT)
#trigram_count, Trigram = trigram(train_iter, TEXT)
#
#perplexity = gen_perplexity(test_iter, TEXT, unigram_dist, bigram_count, Bigram, trigram_count, Trigram)
#
#
#print("Perplexity:", perplexity)


