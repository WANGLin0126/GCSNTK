import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch
from torch import nn
from torch.nn import functional as F
from krr import KernelRidgeRegression
from sntk import StructureBasedNeuralTangentKernel
from LoadData import load_data
from utils import update_E, sub_E
import argparse
import numpy as np
import random

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

parser = argparse.ArgumentParser(description='SNTK computation')
parser.add_argument('--dataset', type=str, default="Cora", help='name of dataset [Cora, Citeseer, Pubmed, Photo, Computers] (default: Cora)')
parser.add_argument('--cond_ratio', type=float, default=0.5, help='condensed ratio of the training set (default: 0.5, the condened set is 0.5*training set)')
parser.add_argument('--ridge', type=float, default=1e0, help='ridge parameter of KRR (default: 1e-4)')
parser.add_argument('--k', type=int, default=2, help='the iteration times of the Graph Convolution when loading data (default: 3)')
parser.add_argument('--epochs', type=int, default=120, help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.005)')
parser.add_argument('--K', type=int, default=2, help='number of aggr in SNTK (default: 2)')
parser.add_argument('--L', type=int, default=2, help='the number of layers after each aggr (default: 2)')
parser.add_argument('--scale', type=str, default='average', help='scale of SNTK [average,add] (default: average)')
parser.add_argument('--set_seed', type=bool, default=True, help='setup the random seed (default: True)')
parser.add_argument('--save', type=bool, default=False, help='save the results (default: False)')
parser.add_argument('--adj', type=bool, default=False, help='condese adj or not (default: False)')
parser.add_argument('--seed', type=int, default=5, help='setup the random seed (default: 5)')
parser.add_argument('--iter', type=int, default=3, help='iteration times of the experiments (default: 5)')
args = parser.parse_args()


# load dataset
root = './datasets/'
adj, x, labels, idx_train, _, idx_test,  \
                        x_train, _, x_test, \
                        y_train, _, y_test, \
                        y_train_one_hot, _, y_test_one_hot, _= load_data(root=root, name=args.dataset, k=args.k)

n_class    = len(torch.unique(labels))
n,dim      = x.shape
n_train    = len(y_train)
Cond_size  = round(n_train*args.cond_ratio)
idx_s      = torch.tensor(range(Cond_size))

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if args.set_seed:
    setup_seed(args.seed)

E_train = sub_E(idx_train, adj)
E_test  = sub_E(idx_test, adj)

SNTK        = StructureBasedNeuralTangentKernel(K=args.K, L=args.L, scale=args.scale).to(device)
ridge       = torch.tensor(args.ridge).to(device)
KRR         = KernelRidgeRegression(SNTK.nodes_gram,ridge).to(device)
MSEloss     = nn.MSELoss().to(device)

adj         = adj.to(device)
x           = x.to(device)
x_train     = x_train.to(device)
x_test      = x_test.to(device)
E_test      = E_test.to(device)
E_train     = E_train.to(device)

y_train_one_hot = y_train_one_hot.to(device)
y_test_one_hot = y_test_one_hot.to(device)

print(f"Dataset       :{args.dataset}")
print(f"Training Set  :{len(y_train)}")
print(f"Testing Set   :{len(y_test)}")
print(f"Classes       :{n_class}")
print(f"Dim           :{dim}")
print(f"Number        :{n}")
print(f"Epochs        :{args.epochs}")
print(f"Learning rate :{args.lr}")
print(f"Conden ratio  :{args.cond_ratio}")
print(f"Ridge         :{args.ridge}")



# torch.autograd.set_detect_anomaly(True) 
def train(G_t, G_s, y_t, y_s, E_t, E_s, loss_fn, optimizer):
    pred, acc = KRR.forward( G_t, G_s, y_t, y_s, E_t, E_s)

    pred      = pred.to(torch.float32)
    y_t       = y_t.to(torch.float32)
    loss      = loss_fn(pred, y_t)
    loss      = loss.to(torch.float32)

    # with torch.autograd.detect_anomaly():
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss = loss.item()

    print(f"Training loss: {loss:>7f} Training Acc: {acc:>7f}", end = ' ')
    return x_s, y_s, loss, acc*100

def test(G_t, G_s, y_t, y_s, E_t, E_s, loss_fn):
    size               = len(y_t)
    test_loss, correct = 0, 0
    with torch.no_grad():
        pred,_      = KRR.forward( G_t, G_s, y_t, y_s, E_t, E_s)
        test_loss  += loss_fn(pred, y_t).item()
        correct    += (pred.argmax(1) == y_t.argmax(1)).type(torch.float).sum().item()
    correct   /= size
    print(f"Test Acc: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}",end = '\n')
    return test_loss, correct*100


Acc = torch.zeros(args.epochs,args.iter).to(device)
for iter in range(args.iter):
    print('--------------------------------------------------')
    print('The '+str(iter+1)+'th Iteration:')
    print('--------------------------------------------------')

    x_s = torch.rand(Cond_size, dim).to(device)
    y_s = torch.rand(Cond_size, n_class).to(device)
    if args.adj:
        feat = x_s.data
        E_s  = update_E(feat,4)
    else:
        E_s = torch.sparse_coo_tensor(torch.stack([idx_s, idx_s], dim=0), torch.ones(Cond_size), torch.Size([Cond_size, Cond_size])).to(x_s.device)

    x_s.requires_grad = True
    y_s.requires_grad = True
    optimizer = torch.optim.Adam([x_s,y_s], lr=args.lr)

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}", end=" ")
        x_s, y_s, training_loss, training_acc = train(x_train, x_s, y_train_one_hot, y_s, E_train, E_s,  MSEloss, optimizer)

        test_loss, test_acc = test(x_test, x_s, y_test_one_hot, y_s, E_test, E_s,  MSEloss)
        Acc[epoch,iter] = test_acc

Acc_mean,Acc_std = torch.mean(Acc, dim=1),torch.std(Acc, dim=1)


print('Mean and std of test data : {:.4f}, {:.4f}'.format(Acc_mean[-1], Acc_std[-1]))
print("--------------- Train Done! ----------------")


if args.save:
    torch.save(x_s, 'x_s.pt')
    torch.save(y_s, 'y_s.pt')
    torch.save(E_s, 'E_s.pt')
    print("--------------- Save Done! ----------------")