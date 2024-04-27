"""
training different models on condensed and saved by GCSNTK

While optimizing the condensed data, train GNN with the condensed data being optimized
try to find the best condensed data for the downstream task
"""




import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch
from torch import nn
from krr import KernelRidgeRegression
from sntk import StructureBasedNeuralTangentKernel
from LoadData import load_data
from utils import sub_G, sub_A_list, update_E, sub_E
import argparse
import numpy as np
import random
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
from models import GAT, GraphSAGE, SGC, MLP, Cheby, GCN, APPNPModel
import argparse
import numpy as np
import random

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

parser = argparse.ArgumentParser(description='GNTK computation')
# several folders, each folder one kernel
parser.add_argument('--model', type=str, default="GCN", help='name of model [GAT, SAGE, SGC, MLP, Cheby, GCN, APPNPModel] (default: GCN)')
parser.add_argument('--dataset', type=str, default="Cora", help='name of dataset (default: Cora)')
parser.add_argument('--cond_ratio', type=float, default=0.5, help='condensed ratio of the training set (default: 0.5, the condened set is 0.5*training set)')
parser.add_argument('--ridge', type=float, default=1e0, help='ridge parameter of KRR (default: 1e-4)')
parser.add_argument('--cond_lr', type=float, default=0.01, help='learning rate (default: 0.005)')
parser.add_argument('--k', type=int, default=3, help='the iteration times of the Graph Convolution when loading data (default: 3)')
parser.add_argument('--cond_epochs', type=int, default=120, help='number of epochs to train (default: 100)')
parser.add_argument('--iter', type=int, default=5, help='iteration times of the experiments (default: 10)')
parser.add_argument('--train_epochs', type=int, default=300, help='iteration times of the experiments (default: 10)')
parser.add_argument('--K', type=int, default=2, help='number of blocks in GNTK (default: 2)')
parser.add_argument('--L', type=int, default=2, help='the number of layers in each block (default: 2)')
parser.add_argument('--scale', type=str, default='average', help='scale of GNTK [average,add] (default: average)')
parser.add_argument('--init', type=str, default='random', help='intialization of X_s [random,loda] (default: random)')
parser.add_argument('--set_seed', type=bool, default=False, help='setup the random seed (default: False)')
parser.add_argument('--seed', type=int, default=5, help='setup the random seed (default: 5)')
parser.add_argument('--train_lr', type=float, default=0.001, help='learning rate (default: 0.005)')
parser.add_argument('--alpha', type=float, default=0.2, help=' the parameter of APPNP (default: 0.2)')
parser.add_argument('--dropout', type=float, default=0.05, help='dropout rate in GAT and APPNP (default: 0.05)')
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--hidden_size', type=int, default=256)
parser.add_argument('--num_heads', type=int, default=8)
args = parser.parse_args()


# load dataset
root = './datasets/'
adj, x, labels, idx_train, _, idx_test,  \
                        x_train, _, x_test, \
                        y_train, _, y_test, \
                        y_train_one_hot, _, y_test_one_hot = load_data(root=root, name=args.dataset, k=args.k)

n_class    = len(torch.unique(labels))
n,dim      = x.shape
n_train    = len(y_train)
Cond_size  = round(n_train*args.cond_ratio)
# adj        = adj.float() + torch.eye(adj.shape[0]) # A + I
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

SNTK       = StructureBasedNeuralTangentKernel(K=args.K, L=args.L, scale=args.scale).to(device)
ridge      = torch.tensor(args.ridge).to(device)
KRR        = KernelRidgeRegression(SNTK.nodes_gram,ridge).to(device)

MSEloss = nn.MSELoss().to(device)

adj = adj.to(device)
x = x.to(device)
x_train = x_train.to(device)
x_test = x_test.to(device)
E_test = E_test.to(device)
E_train = E_train.to(device)

y_train_one_hot = y_train_one_hot.to(device)
y_test_one_hot = y_test_one_hot.to(device)


print(f"Dataset       :{args.dataset}")
print(f"Training Set  :{len(y_train)}")
print(f"Testing Set   :{len(y_test)}")
print(f"Classes       :{n_class}")
print(f"Dim           :{dim}")
print(f"Number        :{n}")
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


dataset  = Planetoid(root='./datasets/', name=args.dataset, split='public')
loader   = DataLoader(dataset, batch_size=32, shuffle=False)




# initialisation of X_s, y_s and A_s
if args.init == 'random':
    x_s = torch.rand(Cond_size, dim).to(device)
    y_s = torch.rand(Cond_size, n_class).to(device)
    E_s = update_E(x_s, 3)
    E_s = torch.sparse_coo_tensor(torch.stack([idx_s, idx_s], dim=0), torch.ones(Cond_size), torch.Size([Cond_size, Cond_size])).to(x_s.device)
elif args.init == 'load':
    x_s = torch.load('x_s.pt')
    y_s = torch.load('y_s.pt')
    E_s = torch.load('E_s.pt')
else:
    raise ValueError('Wrong initialization method!')
x_s.requires_grad = True
y_s.requires_grad = True
edge_index = E_s.coalesce().indices().to(device)
cond_optimizer = torch.optim.Adam([x_s,y_s], lr=args.cond_lr)

max_mean = 0
std = 0

for cond_epoch in range(args.cond_epochs):
    print(f"Epoch {cond_epoch+1}", end=" ")
    x_s, y_s, training_loss, training_acc = train(x_train, x_s, y_train_one_hot, y_s, E_train, E_s,  MSEloss, cond_optimizer)
    test_loss, test_acc = test(x_test, x_s, y_test_one_hot, y_s, E_test, E_s,  MSEloss)


    if cond_epoch > 65:
        Acc = torch.zeros(args.train_epochs, args.iter).to(device)
        print(args.model,end=' ')

        for iter in range(args.iter):
            if args.model == 'GAT':
                model = GAT(dim, args.hidden_size, n_class, args.num_heads, args.dropout)
            elif args.model == 'SAGE':
                model = GraphSAGE(dim, args.hidden_size, n_class)
            elif args.model == 'SGC':
                model = SGC(dim, n_class)
            elif args.model == 'MLP':
                model = MLP(dim, args.hidden_size, n_class)
            elif args.model == 'Cheby':
                model = Cheby(dim, args.hidden_size, n_class, 2)
            elif args.model == 'GCN':
                model = GCN(dim, args.hidden_size, n_class)
            elif args.model == 'APPNP':
                model = APPNPModel(dim, args.hidden_size, n_class, args.num_layers, args.alpha, args.dropout)
            model = model.to(device)
            train_optimizer = optim.Adam(model.parameters(), lr=args.train_lr)
            


            for epoch in range(args.train_epochs):
                # training
                model.train()
                total_loss = 0
                train_optimizer.zero_grad()

                output = model(x_s, edge_index)
                loss = MSEloss(output, y_s)

                loss.backward()
                train_optimizer.step()
                total_loss = loss.item()
                # print('Epoch {}, Training Loss: {:.4f}'.format(epoch + 1, total_loss), end=' ')

                # testing
                model.eval()
                correct = 0
                for batch in loader:
                    batch = batch.to(device)
                    pred = model(batch.x, batch.edge_index).argmax(dim=1)
                    correct += pred[batch.test_mask].eq(batch.y[batch.test_mask]).sum().item()
                    loss = MSEloss(pred[batch.test_mask], batch.y[batch.test_mask].to(torch.float32))
                test_loss = loss.item()
                accuracy = correct / dataset.data.test_mask.sum().item()

                Acc[epoch,iter] = accuracy

        mean_acc = torch.mean(Acc, dim=1)
        std_acc  = torch.std(Acc, dim=1)
        max_acc, max_acc_index = torch.max(mean_acc, dim=0)
        max_std  = std_acc[max_acc_index]
        print('Mean Test Acc: {:.4f}'.format(max_acc),'Std.: {:.4f}'.format(max_std), end='\n')
        if max_acc > max_mean:
            max_mean = max_acc
            std      = max_std
        print('Best Test Acc: {:.4f}'.format(max_mean),'Std.: {:.4f}'.format(std), end='\n')

# print('Best Test Acc: {:.4f}'.format(max_mean),'Std.: {:.4f}'.format(std), end='\n')