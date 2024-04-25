import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch.nn import functional as F
import scipy.sparse as sp


def normalize_data(data):
    """
    normalize data
    parameters:
        data: torch.Tensor, data need to be normalized
    return:
        torch.Tensor, normalized data
    """
    mean = data.mean(dim=0)  # 沿第0维（即样本维）求均值
    std = data.std(dim=0)  # 沿第0维（即样本维）求标准差
    # 将std中的0值替换为1，以避免分母为0的情况
    std[std == 0] = 1  
    normalized_data = (data - mean) / std  # 对数据进行归一化处理
    return normalized_data


def GCF(adj, x, k=1):
    """
    Graph convolution filter
    parameters:
        adj: torch.Tensor, adjacency matrix, must be self-looped
        x: torch.Tensor, features
        k: int, number of hops
    return:
        torch.Tensor, filtered features
    """
    D = torch.sum(adj,dim=1)
    # 对D矩阵求 -0.5 次方
    D = torch.pow(D,-0.5)
    # 对角矩阵
    D = torch.diag(D)
    
    filter = torch.matmul(torch.matmul(D,adj),D)
    for i in range(k):
        x = torch.matmul(filter,x)
    return x


def load_data(root,name,k):
    dataset    = Planetoid(root=root,name=name,split='public')
    train_mask = dataset[0]['train_mask']
    val_mask   = dataset[0]['val_mask']
    test_mask  = dataset[0]['test_mask']
    x          = dataset[0]['x']           # all features
    y          = dataset[0]['y']           # all labels
    edge_index = dataset[0]['edge_index']
    n_class    = len(torch.unique(y))
    n,_      = x.shape

    adj = sp.coo_matrix((torch.ones(edge_index.shape[1]), edge_index), shape=(n, n)).toarray()
    adj = torch.tensor(adj)
    adj = adj + torch.eye(adj.shape[0])    # 对邻接矩阵添加 self-loop

    x = normalize_data(x) # 对特征进行归一化处理
    x = GCF(adj, x, k)  # 对特征进行图卷积滤波处理


    x_train    = x[train_mask]
    x_val      = x[val_mask]
    x_test     = x[test_mask]

    y_train    = y[train_mask]
    y_val      = y[val_mask]
    y_test     = y[test_mask]

    idx_train = torch.where(train_mask)[0]
    idx_val   = torch.where(val_mask)[0]
    idx_test  = torch.where(test_mask)[0]

    y_one_hot       = F.one_hot(y, n_class)
    y_train_one_hot = y_one_hot[train_mask]
    y_val_one_hot   = y_one_hot[val_mask]
    y_test_one_hot  = y_one_hot[test_mask]

    return adj, x, y, idx_train, idx_val, idx_test,  \
                        x_train, x_val, x_test, \
                        y_train, y_val, y_test, \
                        y_train_one_hot, y_val_one_hot, y_test_one_hot