import torch
import preprocessing as pp
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm
from NNLM import NNLM

#Hyperparameters
order = 5

#Test Model

train_iter, val_iter, test_iter, TEXT = pp.PT_preprocessing(bsize=10, bptt=order)

model = torch.load('model_best.pt')
#model = nnlm.NNLM(len(TEXT.vocab))
#model = model.load_state_dict(torch.load('model_best.pt'))

prob_list = torch.FloatTensor()
total_test = 0
print("testing...")
for batch in tqdm(test_iter):
#for i in range(1):
    #print(i)
    x = batch.text[:4].transpose(0, 1)
    y = batch.text[4:].squeeze(0)
    #x, y = x.cuda(), y.cuda()
    outputs = model(x)
    
    
    for i, label in enumerate(y):
        #print(type(prob_list))
        total_test += np.log2(outputs[i][label.data[0]].data.numpy())[0]
        prob_list = torch.cat((prob_list, outputs[i][label.data[0]].data))
    

print(total_test)

prob_dist = prob_list.numpy()
print(prob_dist.shape)
perplexity = 2**((-1/(prob_dist.shape[0])) * np.sum(np.log2(prob_dist)))

print('Perplexity is:', perplexity)