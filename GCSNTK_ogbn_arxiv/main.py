import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch
from torch import nn
from torch.nn import functional as F
from krr import KernelRidgeRegression
from sntk import StructureBasedNeuralTangentKernel
from utils import sub_G, sub_A_list, update_E, sub_E
import argparse
import numpy as np
import random
import time

from OGBDataloader import OgbDataLoader
print('ending')

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")



parser = argparse.ArgumentParser(description='SNTK computation')
parser.add_argument('--dataset', type=str, default="ogbn-arxiv", help='name of dataset (default: ogbn-arxiv)')
parser.add_argument('--cond_size', type=float, default=90, help='condensed ratio of the training set (default: 0.5, the condened set is 0.5*training set)')
parser.add_argument('--ridge', type=float, default=1e-6, help='ridge parameter of KRR (default: 1e-3)')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=5e-3, help='learning rate (default: 0.005)')
parser.add_argument('--K', type=int, default=1, help='number of aggr in SNTK (default: 2)')
parser.add_argument('--L', type=int, default=1, help='the number of layers after each aggr (default: 2)')
parser.add_argument('--scale', type=str, default='add', help='scale of SNTK [average,add] (default: average)')
parser.add_argument('--k', type=int, default=1, help='the convolutiona times of the dataset (default: 2)')
parser.add_argument('--set_seed', type=bool, default=False, help='whether to set seed')
parser.add_argument('--adj', type=bool, default=False, help='condese adj or not (default: False)')
parser.add_argument('--seed', type=int, default=5, help='setup the random seed (default: 5)')
parser.add_argument('--iter', type=int, default=3, help='iteration times (default: 3)')
args = parser.parse_args()




def train(G_t, G_s, y_t, y_s, A_t, A_s, loss_fn, optimizer, accumulate_steps, i):
    pred, correct = KRR.forward( G_t, G_s, y_t, y_s, A_t, A_s)

    pred      = pred.to(torch.float32)
    y_t       = y_t.to(torch.float32)
    loss      = loss_fn(pred, y_t)
    loss      = loss.to(torch.float32)

    loss = loss/accumulate_steps
    loss.backward()

    if (i+1) % accumulate_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
    elif i == TRAIN_K - 1:
        optimizer.step()
        optimizer.zero_grad()

    loss = loss.item()
    return x_s, y_s, loss, correct

def test(G_t, G_s, y_t, y_s, A_t, A_s, loss_fn):
    size               = len(y_t)
    test_loss, correct = 0, 0
    with torch.no_grad():
        pred,_      = KRR.forward( G_t, G_s, y_t, y_s, A_t, A_s)
        test_loss  += loss_fn(pred, y_t).item()
        correct    += (pred.argmax(1) == y_t.argmax(1)).type(torch.float).sum().item()
    return test_loss, correct


name = args.dataset
train_loader = OgbDataLoader(dataset_name=name, split='train', batch_size=500, split_method='kmeans')
test_loader  = OgbDataLoader(dataset_name=name, split='test', batch_size=500, split_method='kmeans')
TRAIN_K,n_train,n_class, dim, n  = train_loader.properties()
test_k,n_test,_,_,_              = test_loader.properties()


train_loader.split_batch()
test_loader.split_batch()



def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
if args.set_seed:
    setup_seed(args.seed)


SNTK       = StructureBasedNeuralTangentKernel(K=args.K, L=args.L, scale=args.scale).to(device)
ridge      = torch.tensor(args.ridge).to(device)
kernel     = SNTK.nodes_gram
KRR        = KernelRidgeRegression(kernel,ridge).to(device)


print(f"Dataset       :{args.dataset}")
print(f"Number        :{n}")
print(f"Training Set  :{n_train}")
print(f"Testing Set   :{n_test}")
print(f"Classes       :{n_class}")
print(f"Dim           :{dim}")
print(f"Epochs        :{args.epochs}")
print(f"Learning rate :{args.lr}")
print(f"Conden size   :{args.cond_size}")
print(f"Ridge         :{args.ridge}")
print(f"Aggr          :{args.K}")
print(f"Layers        :{args.L}")
print(f"Num of batches:{TRAIN_K}")


results = torch.zeros(args.epochs,args.iter)
for iter in range(args.iter):
    print('--------------------------------------------------')
    print(f"The  {iter+1}-th iteration")
    print('--------------------------------------------------')


    idx_s  = torch.tensor(range(round(args.cond_size)))
    x_s = torch.rand(args.cond_size, dim)
    y_s = torch.rand(args.cond_size, n_class)
    if args.adj:
        feat = x_s.data
        A_s  = update_E(feat,3)
    else:
        ind    = torch.stack([idx_s, idx_s], dim=0).to(device)
        values = torch.ones(args.cond_size).to(device)
        A_s    = torch.sparse_coo_tensor(ind, values, torch.Size([args.cond_size,args.cond_size])).to(x_s.device)


    MSEloss = nn.MSELoss().to(device)
    idx_s   = idx_s.to(device)
    x_s     = x_s.to(device)
    y_s     = y_s.to(device)
    A_s     = A_s.to(device)
    x_s.requires_grad = True
    y_s.requires_grad = True
    optimizer = torch.optim.Adam([x_s,y_s], lr=args.lr)

    
    T = 0
    Time = []
    Time.append(0)
    for t in range(args.epochs):
        print(f"Size {args.cond_size} Epoch {iter+1} - {t+1}", end=" ")

        train_loss, test_lossi = torch.zeros(TRAIN_K),  torch.zeros(test_k)
        train_correct_all, test_correct_all = 0, 0
        a = time.time()
        for i in range(TRAIN_K):

            x_train, label, sub_A_t  = train_loader.get_batch(i)
            y_train_one_hot          = F.one_hot(label.reshape(-1), n_class)

            x_train = x_train.to(device)
            y_train_one_hot = y_train_one_hot.to(device)
            sub_A_t = sub_A_t.to(device)

            _, _, training_loss, train_correct = train(x_train, x_s, y_train_one_hot, y_s, sub_A_t, A_s,  MSEloss, optimizer, 10, i)

            train_correct_all = train_correct_all + train_correct
            train_loss[i]     = training_loss
        training_loss_avg = torch.mean(train_loss)
        training_acc_avg = (train_correct_all / n_train) * 100
        b = time.time()
        T = T + b - a
        Time.append(T)

        if t >= 10:
            aa = time.time()
            for j in range(test_k):
                x_test, test_label, sub_A_test  = test_loader.get_batch(j)
                y_test_one_hot       = F.one_hot(test_label.reshape(-1), n_class)

                x_test = x_test.to(device)
                y_test_one_hot = y_test_one_hot.to(device)
                sub_A_test = sub_A_test.to(device)

                test_loss, test_correct = test(x_test, x_s, y_test_one_hot, y_s, sub_A_test, A_s,  MSEloss)

                test_correct_all = test_correct_all + test_correct
                test_lossi[j] = test_loss
            bb = time.time()
            test_loss_avg = torch.mean(test_lossi)
            test_acc      = (test_correct_all / n_test) * 100
            print(f"Test Avg Acc: {(test_acc):>0.4f}%, Test Avg loss: {test_loss_avg:>0.6f}",end = '\n')
            results[t,iter] = test_acc

Acc_mean,Acc_std = torch.mean(results, dim=1),torch.std(results, dim=1)
max_mean, max_mean_index = torch.max(Acc_mean, dim=0)
print(f'Mean Test Acc: {max_mean.item():>0.4f}%, Std: {Acc_std[max_mean_index].item():>0.4f}%')

print("--------------- Train Done! ----------------")