import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, SGConv, APPNP, ChebConv, GCNConv, GATConv


class APPNPModel(nn.Module):
    def __init__(self, num_features, hidden_size, num_classes, num_layers, alpha, dropout):
        super(APPNPModel, self).__init__()
        self.conv1 = nn.Linear(num_features, hidden_size)
        self.conv2 = nn.Linear(hidden_size, num_classes)
        self.appnp = APPNP(num_layers, alpha, dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)

        x = self.appnp(x, edge_index)

        return F.softmax(x, dim=1)

class Cheby(nn.Module):
    def __init__(self, in_features, hidden_size, num_classes, num_chebyshev):
        super(Cheby, self).__init__()

        self.conv1 = ChebConv(in_features, hidden_size, K=num_chebyshev)
        self.conv2 = ChebConv(hidden_size, num_classes, K=num_chebyshev)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)

        return F.softmax(x, dim=1)

class GAT(nn.Module):
    def __init__(self, in_features, hidden_dim, num_classes, num_heads, dropout):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_features, hidden_dim, heads=num_heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * num_heads, num_classes, heads=1, concat=False, dropout=dropout)
        
    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.softmax(x, dim=1)


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.softmax(x, dim=1)


class GraphSAGE(nn.Module):
    def __init__(self, in_features, hidden_size, num_classes):
        super(GraphSAGE, self).__init__()

        self.conv1 = SAGEConv(in_features, hidden_size)
        self.conv2 = SAGEConv(hidden_size, hidden_size)
        self.conv3 = SAGEConv(hidden_size, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)

        return F.softmax(x, dim=1)

class MLP(nn.Module):
    def __init__(self, in_features, hidden_size, num_classes):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(in_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x, edge_index):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)

class SGC(nn.Module):
    def __init__(self, in_features, num_classes):
        super(SGC, self).__init__()

        self.conv1 = SGConv(in_features, num_classes, K=2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)

        return F.softmax(x, dim=1)