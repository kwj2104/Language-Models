
import preprocessing as pp
from collections import Counter
from itertools import islice
import math

A_1 = .4
A_2 = .5
A_3 = .1

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
    
    #remove all the options with 0s in any alpha to prevent sparsity messing up log2
    interp_parameters = [i for i in interp_parameters if (i[0] != 0) and (i[1] is not 0) and (i[2] is not 0)]

    return interp_parameters
        
def get_bigram_dist(bigram_count, unigram_count, word_t1):
    bigram_dist = [0] * len(TEXT.vocab)
    for i in range(len(TEXT.vocab)):
        bigram_dist[i] = bigram_count[(word_t1, i)]/unigram_count[word_t1]
        #print(bigram_count[(word_t1, i)],unigram_count[i])
    
    return bigram_dist

def get_trigram_dist(trigram_count, bigram_count, word_t2, word_t1):
    trigram_dist = [0] * len(TEXT.vocab)
    for i in range(len(TEXT.vocab)):
        try:
            trigram_dist[i] = trigram_count[(word_t2, word_t1, i)]/bigram_count[(word_t2, word_t1)]
        except:
            trigram_dist[i] = 0
    
    return trigram_dist

if __name__ == "__main__":
    train_iter, val_iter, test_iter, TEXT = pp.PT_preprocessing(bsize=10, bptt=10, shuf=True)
    
    total_text = []
    
    print("Training...")
    
    for b in iter(train_iter):
       total_text += b.text.transpose(0, 1).contiguous().view(-1).data.tolist()
       
    unigram_count = Counter(total_text)
    bigram_count = Counter(zip(*[islice(total_text, i, None) for i in range(2)]))
    trigram_count = Counter(zip(*[islice(total_text, i, None) for i in range(3)]))
    
    unigram_dist = [0] * len(TEXT.vocab)
    unigram_total = len(total_text)
    for i in range(len(TEXT.vocab)):
        unigram_dist[i] = unigram_count[i]/unigram_total
        
    # Test
    print("Testing...")
    
    total_text = []
    for b in iter(test_iter):
       total_text += b.text.transpose(0, 1).contiguous().view(-1).data.tolist()
    test_list = list(zip(*[total_text[i:] for i in range(3)]))
    
    interp_parameters = gen_parameters()
    max_perplex = 1000
    max_params = (0, 0, 0)
    for i in interp_parameters:
        A_1 = i[0]
        A_2 = i[1]
        A_3 = i[2]
        
        total_log = 0
        total_samples = len(test_list)
        for words in test_list:
            bigram_dist = bigram_count[(words[1], words[2])]/unigram_count[words[1]]
            try:
                trigram_dist = trigram_count[(words[0], words[1], words[2])]/bigram_count[(words[0], words[1])]
            except:
                trigram_dist = 0
            
            #print(trigram_dist)
            dist = A_1 * unigram_dist[words[2]] + (A_2 * bigram_dist)+ (A_3 * trigram_dist)
            total_log += math.log2(dist)
    
        perplexity = 2**(-1 *total_log/total_samples)
        
        if perplexity < max_perplex:
            max_perplex = perplexity
            max_params = (A_1, A_2, A_3)
            
    print(max_perplex, max_params)
        

    
    #Create inputs
    
#    input_list = []
#    with open('input.txt') as f:
#        for line in f:
#            line_split = line.split()
#            input_list.append((TEXT.vocab.stoi[line_split[8]], TEXT.vocab.stoi[line_split[9]]))
#    print(len(input_list))
#    
#    predictions = []
#    for example in tqdm(input_list):
#        bigram_dist = np.array(get_bigram_dist(bigram_count, unigram_count, example[1]))
#        trigram_dist = np.array(get_trigram_dist(trigram_count, bigram_count, example[1], example[0]))
#        unigram_dist = np.array(unigram_dist)
#        #print(bigram_dist + trigram_dist)
#        dist = (A_1 * unigram_dist) + (A_2 * bigram_dist)+ (A_3 * trigram_dist)
#        #print((-dist).argsort()[:20])
#        #print(np.argpartition(dist, -20)[-20:].tolist())
#        dist_list = dist.tolist()
#        pred = [TEXT.vocab.itos[i] for i in dist.argsort()[-20:][::-1].tolist()]
#        #pred = [TEXT.vocab.itos[i] for i in np.argsort(dist[np.argpartition(dist, -20)[-20:]]).tolist()]
#        predictions.append(pred)
#    print(len(predictions))
#    with open("sample.txt", "w") as fout: 
#        print("id,word", file=fout)
#        for i in range(len(input_list)):
#            print("%d,%s"%(i+1, " ".join(predictions[i])), file=fout)
        

        
        
    
    