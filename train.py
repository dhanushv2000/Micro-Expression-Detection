import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from torch import nn as neural_network,optim
import pickle as pkl
from sklearn.metrics import recall_score,f1_score

from data import *
from train_arg import *

dataset = None
if from_file:
    with open(data_filename,'rb') as f:
        dataset = pkl.load(f) 
else:    
    dataset = data_class(**data_args)
    with open(data_filename,'wb') as f:
        pkl.dump(dataset,f)

def train(traindata,testdata):
    loader = DataLoader(traindata,batch_size = len(traindata) if batch_size == -1 else batch_size,shuffle = True)
    model = model_class(**model_args).to(device)
    model.train()
    optimizer = optimizer_class(model.parameters(),**optimizer_args)
    scheduler = scheduler_class(optimizer,**scheduler_args)
    criterion = neural_network.CrossEntropyLoss().to(device)
    res_dict = {}
    for epoch in tqdm(range(n_epochs)):
        losses = []
        for idx,(x,y) in enumerate(loader):
            x,y = x.to(device),y.to(device)
            out = model(x)
            loss = criterion(out,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.cpu().detach().numpy())
        if epoch in observed_epochs:
            res = test(model,testdata)
            res_dict[epoch] = res
            model.train()
            
        scheduler.step()
            
    return res_dict
    
def test(model,testdata):
    loader = DataLoader(testdata,batch_size = 1,shuffle = True)
    model.eval()
    y,pred = [],[]
    for idx,(x0,y0) in enumerate(loader):
        x0,y0 = torch.Tensor(x0).to(device),y0.numpy()
        y.append(y0[0])
        pred0 = model(x0).cpu().detach().numpy()
        pred.append(np.where(pred0[0] == np.max(pred0[0]))[0][0])
    y,pred = np.array(y),np.array(pred)
    return get_metric(y,pred)

def get_metric(y,pred):
    uar = recall_score(y,pred,average = 'macro')
    uf1 = f1_score(y,pred,average = 'macro')
    return uar,uf1

def loso():
    
    results = []
    for idx,(traindata,testdata) in tqdm(enumerate(LOSOGenerator(dataset))):
        print('Test ' + str(idx))
        res = train(traindata,testdata)
        results.append(res)
        e1,(max_uar,_) = sorted(list(res.items()),reverse = True,key = lambda x:x[1][0])[0]
        e2,(_,max_uf1) = sorted(list(res.items()),reverse = True,key = lambda x:x[1][1])[0]    
    
    mean_uar = lambda epoch:np.mean(np.array([dic[epoch][0] for dic in results]))
    mean_uf1 = lambda epoch:np.mean(np.array([dic[epoch][1] for dic in results]))
    e1 = sorted(list(observed_epochs),reverse = True,key = mean_uar)[0]
    max_mean_uar = mean_uar(e1)
    e2 = sorted(list(observed_epochs),reverse = True,key = mean_uar)[0]
    max_mean_uf1 = mean_uf1(e2)
    
    with open(train_process_filename,'wb') as f:
        pkl.dump(results,f)
    
if __name__=='__main__':
    loso()