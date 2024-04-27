"""
training different models on condensed and saved by GCSNTK
the training data is just loaded from the saved files

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
from models import GAT, GraphSAGE, SGC, MLP, Cheby, GCN, APPNPModel
import argparse
import numpy as np
import random




parser = argparse.ArgumentParser(description='GDKRR Generalization')
parser.add_argument('--model', type=str, default="GCN", help='name of model [GAT, SAGE, SGC, MLP, Cheby, GCN, APPNPModel] (default: GCN)')
parser.add_argument('--dataset', type=str, default="Cora", help='name of dataset [Cora, Citeseer, Pubmed](default: Cora)')
parser.add_argument('--cond_ratio', type=float, default=0.5, help='condensation ratio [0.25, 0.5, 1](default: 0.5)')
parser.add_argument('--alpha', type=float, default=0.2, help=' the parameter of APPNP (default: 0.2)')
parser.add_argument('--num_layers', type=int, default=2, help=' the parameter of APPNP (default: 2)')
parser.add_argument('--dropout', type=float, default=0.05, help='dropout rate in GAT and APPNP (default: 0.05)')
parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--num_heads', type=int, default=8)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--set_seed', type=bool, default=False)
parser.add_argument('--seed', type=int, default=5)
parser.add_argument('--iter', type=int, default=10)
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if args.set_seed:
    print('Set seed')
    setup_seed(args.seed)

dataset = Planetoid(root='./datasets/', name=args.dataset, split='public')
data = dataset[0]  
num_features = dataset.num_features
num_classes  = dataset.num_classes
criterion = nn.MSELoss()
loader    = DataLoader(dataset, batch_size=32, shuffle=False)


x_s = torch.load(args.dataset+'_'+str(args.cond_ratio)+'_x_s.pt', map_location=torch.device('cpu')).to(device)
y_s = torch.load(args.dataset+'_'+str(args.cond_ratio)+'_y_s.pt', map_location=torch.device('cpu')).to(device)
A_s = torch.load(args.dataset+'_'+str(args.cond_ratio)+'_A_s.pt', map_location=torch.device('cpu')).to(device)
edge_index = A_s.coalesce().indices().to(device)


results = torch.zeros(args.epochs, args.iter).to(device)


max_test_acc = 0

print('Model', args.model)
for iter in range(args.iter):
    if args.model == 'GAT':
        model = GAT(num_features, args.hidden_size, num_classes, args.num_heads, args.dropout)
    elif args.model == 'SAGE':
        model = GraphSAGE(num_features, args.hidden_size, num_classes)
    elif args.model == 'SGC':
        model = SGC(num_features, num_classes)
    elif args.model == 'MLP':
        model = MLP(num_features, args.hidden_size, num_classes)
    elif args.model == 'Cheby':
        model = Cheby(num_features, args.hidden_size, num_classes, 2)
    elif args.model == 'GCN':
        model = GCN(num_features, args.hidden_size, num_classes)
    elif args.model == 'APPNP':
        model = APPNPModel(num_features, args.hidden_size, num_classes, args.num_layers, args.alpha, args.dropout)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)



    print('--------------------------------------------------')
    print('The '+str(iter+1)+'th iteration:')
    print('--------------------------------------------------')


    
    for epoch in range(args.epochs):

        # training
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        output = model(x_s, edge_index)
        loss = criterion(output, y_s)


        loss.backward()
        optimizer.step()
        total_loss = loss.item()
        print('Epoch {}, Training Loss: {:.4f}'.format(epoch + 1, total_loss), end=' ')

        # testing
        model.eval()
        correct = 0
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index).argmax(dim=1)
            correct += pred[batch.test_mask].eq(batch.y[batch.test_mask]).sum().item()
            loss = criterion(pred[batch.test_mask], batch.y[batch.test_mask].to(torch.float32))
        test_loss = loss.item()
        accuracy = correct / dataset.data.test_mask.sum().item()
        results[epoch][iter] = accuracy
        max_test_acc = max(max_test_acc, accuracy)

        print('Test Loss: {:.4f}'.format(test_loss),'Test Acc: {:.4f}'.format(accuracy), end='\n')

    print('Max Test Accuracy: {:.4f}'.format(max_test_acc))


# mean and std
results_mean = torch.mean(results, dim=1)
results_std  = torch.std(results, dim=1)



max_mean, max_mean_index = torch.max(results_mean, dim=0)


print('Model', args.model)
print('The max mean of test accuracy is {:.4f} at epoch {}'.format(max_mean, max_mean_index+1))
print('The std of test accuracy is {:.4f}'.format(results_std[max_mean_index]))
print(results[max_mean_index])


