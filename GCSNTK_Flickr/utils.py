import torch
from torch_geometric.nn import MessagePassing
"""
Transform a single, large graph with n nodes to n subgraphs
"""

def find(idx,A,c=0):
    """
    find out the one-hop neighbors of nodes in given idx
    len(list) = len(idx)
    A must tensor
    """
    list = []
    for node in idx:
        neigh = torch.where(A[node]==1)[0]
        if c :
            list.append(neigh)
        else:
            for i in range(len(neigh)):
                list.append(neigh[i])
    if c:
        return list
    else:
        return torch.unique(torch.tensor(list))


def find_hop_idx(i,j,A):
    """
    find the index of the j-hop neighbors of node i
    """
    idx = [i,]
    for hop in range(j):
        idx = find(idx,A)
    return idx


def sub_G(A, hop):
    """
    A is the adjacency martrix of graph
    """
    n         = A.shape[0]
    neighbors = []
    for i in range(n):
        neighbor_i = torch.tensor(find_hop_idx(i,hop,A))
        neighbors.append(neighbor_i.to(A.device))
    return neighbors



def sub_A_list(neighbors, A):
    """
    output the adjacency matrix of subgraph
    """
    n        = A.shape[0]
    sub_A_list= []
    for node in range(n):

        n_neig   = len(neighbors[node])
        operator = torch.zeros([n,n_neig]).to(A.device)
        operator[neighbors[node],range(n_neig)] = 1
        sub_A = torch.matmul(torch.matmul(operator.t(),A),operator)
        sub_A_list.append(sub_A)

    return sub_A_list


def sub_A(idx, A):
    """
    output the adjacency matrix of subgraph of idx
    """
    n        = A.shape[0]
    n_neig   = len(idx)
    operator = torch.zeros([n,n_neig])
    operator[idx,range(n_neig)] = 1
    sub_A    = torch.matmul(torch.matmul(operator.t(),A),operator)

    return sub_A


def sub_E(idx, A):
    """
    output the adjacency matrix of subgraph of idx
    """
    n        = A.shape[0]
    n_neig   = len(idx)
    operator = torch.zeros([n,n_neig])
    operator[idx,range(n_neig)] = 1
    sub_A    = torch.matmul(torch.matmul(operator.t(),A),operator)

    ind    = torch.where(sub_A!=0)
    inds   = torch.cat([ind[0],ind[1]]).reshape(2,len(ind[0]))
    values = torch.ones(len(ind[0]))
    sub_E  = torch.sparse_coo_tensor(inds, values, torch.Size([n_neig, n_neig])).to(A.device)

    return sub_E



def update_A(x_s,neig):
    '''
    x_s is the features
    neig is the average number of the neighbors of each node
    '''
    n = x_s.shape[0]
    K = torch.empty(n,n)
    A = torch.zeros(n*n)

    for i in range(n):
        for j in range(i,n):
            K[i,j] = torch.norm(x_s[i]-x_s[j])
            K[j,i] = K[i,j]
    
    edge  = int(n+torch.round(torch.tensor(neig*n/2)))          # the edge number, must be even
    
    if (edge % 2) != 0:
        edge += 1
    else:
        pass

    Simil = torch.flatten(K)               
    _, indices = torch.sort(Simil) # sort all the similarities
    A[indices[0:edge]] = 1              
    A = A.reshape(n,n).to(x_s.device)

    return A



def update_E(x_s,neig):
    '''
    x_s is the features
    neig is the average number of the neighbors of each node
    '''
    n = x_s.shape[0]
    K = torch.empty(n,n)
    A = torch.zeros(n*n)

    for i in range(n):
        for j in range(i,n):
            K[i,j] = torch.norm(x_s[i]-x_s[j])
            K[j,i] = K[i,j]
    
    edge  = int(n+torch.round(torch.tensor(neig*n/2)))     
    if (edge % 2) != 0:
        edge += 1
    else:
        pass

    Simil = torch.flatten(K)               
    _, indices = torch.sort(Simil) 
    A[indices[0:edge]] = 1              
    A = A.reshape(n,n)
    ind = torch.where(A==1)

    ind = torch.cat([ind[0],ind[1]]).reshape(2,edge)
    values = torch.ones(edge)
    E = torch.sparse_coo_tensor(ind, values, torch.Size([n,n])).to(x_s.device)

    return E
    
class Aggr(MessagePassing):
    """
    Undirected nodes features aggregation ['add', 'mean']
    """
    def __init__(self, aggr='add'):
        super(Aggr, self).__init__(aggr=aggr)

    def forward(self, x, edge_index):
        """
        inputs:
            x: [N, dim]
            edge_index: [2, edge_num]
        outputs:
            the aggregated node features
            out: [N, dim]
        """
        edge_index = torch.cat([edge_index, edge_index.flip(dims=[0])], dim = 1)
        edge_index = torch.unique(edge_index, dim = 1)
        return self.propagate(edge_index, x=x) + x