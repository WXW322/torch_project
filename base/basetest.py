import torch
import glob
import os
from torch import nn
import unicodedata
import string
import random
import sys
import itertools
import model


h_dim = 5
n_layer = 2

def normalize(s):
    return "".join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != "Mn")

def trans_data(file_name):
    file_n = open(file_name,'r')
    line_datas = [normalize(s) for s in file_n.read().strip().split("\n")]
    return line_datas

def voc(w_dic):
    t_lo = 0
    w_s = {}
    for w in w_dic:
        datas = w[0]
        for data in datas:
            if data not in w_s: 
                w_s[data] = t_lo
                t_lo = t_lo + 1
    return w_s            

def padding(l,padding_value):
    return list(itertools.zip_longest(*l,fillvalue=padding_value))            

def process_data(path):
    letters = []
    category = {}
    t_lo = 1
    for file_name in os.listdir(path):
        pre = file_name.split(".")[0]
        if pre not in category:
            category[pre] = t_lo
            t_lo = t_lo + 1
        t_path = os.path.join(path,file_name)
        line_datas = trans_data(t_path)
        for data in line_datas:
            letters.append((data,pre))
    words_dic = voc(letters)
    return letters,words_dic,category

def strs2tensor(str_l,voc,category):
    str_l.sort(key = lambda i:len(i[0]),reverse=True)
    lengths = [len(item[0]) for item in str_l]
    #str_l = padding(str_l,0)
    inputs = []
    outputs = []
    for s in str_l:
        inputs.append([voc[w] for w in s[0]])
        outputs.append(category[s[1]])
    inputs = padding(inputs,0)
    out_tensor = torch.LongTensor(outputs)
    input_tensor = torch.LongTensor(inputs)
    input_tensor = input_tensor.transpose(0,1)
    lengths = torch.LongTensor(lengths)
    return input_tensor,out_tensor,lengths


def get_trains(path,batch_size,num_iters):
    voc_w,letters,category = process_data(path)
    datas_T = []
    for _ in range(num_iters):
        pairs = [random.choice(voc_w) for _ in range(batch_size)]
        datas = strs2tensor(pairs,letters,category)
        datas_T.append(datas)
    return datas_T, voc_w,category

def train(input_X, lengths, output_X, h_state, model, model_optim):
    output_X = output_X - 1
    model_optim.zero_grad()
    out,_ = model(input_X, lengths, h_state)
    nllloss = nn.NLLLoss()
    loss = nllloss(out, output_X)
    loss.backward()
    model_optim.step()
    return loss

def train_Iter(datas, h_state, model, model_optim, per_iter):
    for i in range(len(datas)):
        loss = train(datas[i][0], datas[i][2], datas[i][1], h_state, model, model_optim)
        if(i % per_iter == 0):
            print("iters: ", i, ",loss is ", loss)
    
    
    
    

h_state = torch.randn(2, 100, h_dim)
Data, voc_w,category = get_trains("./data/names/",100,30)
embedding = nn.Embedding(len(voc_w), h_dim)
rnn = model.textclass(embedding, h_dim, h_dim, n_layer, len(category))
optimer = torch.optim.SGD(rnn.parameters(), lr=0.1, momentum=0.9)
train_Iter(Data, h_state, rnn, optimer, 10)
#out,_ = rnn(Data[0][0], Data[0][2], h_state)
#print(out.size())
#aa,bb,cc = process_data("./data/names/")

#print(aa)


    
    
