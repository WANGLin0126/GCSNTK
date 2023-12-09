import torch
from torch import nn
print('load ogb')
import ogb
print('load ogb.nodeproppred')
from ogb.nodeproppred import dataset
print('load sklearn')
from sklearn.cluster import KMeans

class OgbDataLoader(nn.Module):
    def __init__(self, dataset_name='ogbn-arxiv', split='train', batch_size=5000, split_method='kmeans'):
        super(OgbDataLoader, self).__init__()
        Dataset       = dataset.NodePropPredDataset(dataset_name, root="./datasets/")
        self.n, self.dim     = Dataset.graph['node_feat'].shape
        split_set     = Dataset.get_idx_split()
        graph,labels  = Dataset[0]
        features      = torch.tensor(graph['node_feat'])
        edge_index    = torch.tensor(graph['edge_index'])
        values        = torch.ones(edge_index.shape[1])
        Adj           = torch.sparse_coo_tensor(edge_index, values, torch.Size([self.n,self.n]))
        sparse_eye = torch.sparse_coo_tensor(torch.arange(self.n).repeat(2, 1), torch.ones(self.n), (self.n, self.n))
        self.Adj = Adj + sparse_eye

        features      = self.normalize_data(features)
        features      = self.GCF(self.Adj, features, k=1)
        labels        = torch.tensor(labels)
        
        self.split_idx    = torch.tensor(split_set[split])
        self.n_split  = len(self.split_idx)
        self.k        = torch.round(torch.tensor(self.n_split/batch_size)).to(torch.int)
        self.split_feat   = features[self.split_idx]
        self.split_label  = labels[self.split_idx]



        self.split_method = split_method
        self.n_classes    = Dataset.num_classes

    def normalize_data(self, data):
        """
        normalize data
        parameters:
            data: torch.Tensor, data need to be normalized
        return:
            torch.Tensor, normalized data
        """
        mean = data.mean(dim=0) 
        std = data.std(dim=0)   
        std[std == 0] = 1       
        normalized_data = (data - mean) / std 
        return normalized_data


    def GCF(self, adj, x, k=2):
        """
        Graph convolution filter
        parameters:
            adj: torch.Tensor, adjacency matrix, must be self-looped
            x: torch.Tensor, features
            k: int, number of hops
        return:
            torch.Tensor, filtered features
        """
        n = adj.shape[0]
        ind = torch.tensor(range(n)).repeat(2,1)
        adj = adj + torch.sparse_coo_tensor(ind, torch.ones(n), (n,n))

        D = torch.pow(torch.sparse.sum(adj,1).to_dense(), -0.5)
        D = torch.sparse_coo_tensor(ind, D, (n,n))

        filter = torch.sparse.mm(torch.sparse.mm(D,adj),D)
        for i in range(k):
            x = torch.sparse.mm(filter,x)
        return x

    def properties(self):
        return self.k, self.n_split, self.n_classes, self.dim, self.n
    
    def split_batch(self):
        """
        split data into batches
        parameters:
            split_method: str, method to split data, default is 'kmeans'
        """
        if self.split_method == 'kmeans':
            kmeans = KMeans(n_clusters = self.k)
            kmeans.fit(self.split_feat.numpy())
            self.batch_labels = kmeans.predict(self.split_feat.numpy())
        
        # save batch labels
        torch.save(self.batch_labels, './{}_{}_batch_labels.pt'.format(self.split_method, self.k))

    def getitem(self, idx):
        # idx   = [idx]
        n_idx   = len(idx)
        idx_raw = self.split_idx[idx]
        feat    = self.split_feat[idx]
        label   = self.split_label[idx]
        # idx   = idx.tolist()

        optor_index = torch.cat((torch.tensor(idx_raw).reshape(1,n_idx),torch.tensor(range(n_idx)).reshape(1,n_idx)),dim=0)
        optor_value = torch.ones(n_idx)
        optor_shape = torch.Size([self.n,n_idx])

        optor       = torch.sparse_coo_tensor(optor_index, optor_value, optor_shape)
        sub_A       = torch.sparse.mm(torch.sparse.mm(optor.t(), self.Adj), optor)

        return (feat, label, sub_A)


    def get_batch(self, i):
        idx       = torch.where(torch.tensor(self.batch_labels) == i)[0]
        batch_i   = self.getitem(idx)
        return batch_i


