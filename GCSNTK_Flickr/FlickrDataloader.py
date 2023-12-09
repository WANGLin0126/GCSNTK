import torch
from torch import nn
from sklearn.cluster import KMeans

class FlickrDataLoader(nn.Module):
    def __init__(self, name = 'Flickr', split='train', batch_size=3000, split_method='kmeans'):
        super(FlickrDataLoader, self).__init__()
        if name == 'Flickr':
            from torch_geometric.datasets import Flickr as DataSet
        elif name == 'Reddit':
            from torch_geometric.datasets import Reddit as DataSet

        Dataset       = DataSet("./datasets/" + name, None, None)
        self.n, self.dim = Dataset[0].x.shape
        mask          = split + '_mask'
        features      = Dataset[0].x
        labels        = Dataset[0].y
        edge_index    = Dataset[0].edge_index

        values        = torch.ones(edge_index.shape[1])
        Adj           = torch.sparse_coo_tensor(edge_index, values, torch.Size([self.n,self.n]))
        sparse_eye    = torch.sparse_coo_tensor(torch.arange(self.n).repeat(2, 1), torch.ones(self.n), (self.n, self.n))
        self.Adj      = Adj + sparse_eye
        features      = self.normalize_data(features)
        self.split_idx= torch.where(Dataset[0][mask])[0]
        self.n_split  = len(self.split_idx)
        self.k        = torch.round(torch.tensor(self.n_split/batch_size)).to(torch.int)


        optor_index       = torch.cat((self.split_idx.reshape(1,self.n_split),torch.tensor(range(self.n_split)).reshape(1,self.n_split)),dim=0)
        optor_value       = torch.ones(self.n_split)
        optor_shape       = torch.Size([self.n,self.n_split])
        optor             = torch.sparse_coo_tensor(optor_index, optor_value, optor_shape)
        self.Adj_mask     = torch.sparse.mm(torch.sparse.mm(optor.t(), self.Adj), optor)
        self.split_feat   = features[self.split_idx]
        # self.split_feat   = self.GCF(self.Adj_mask, self.split_feat, k = 2)

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
        n   = adj.shape[0]
        ind = torch.tensor(range(n)).repeat(2,1)
        adj = adj + torch.sparse_coo_tensor(ind, torch.ones(n), (n,n))

        D   = torch.pow(torch.sparse.sum(adj,1).to_dense(), -0.5)
        D   = torch.sparse_coo_tensor(ind, D, (n,n))

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

    def getitem(self, idx):
        # idx   = [idx]
        n_idx   = len(idx)
        idx_raw = self.split_idx[idx]
        feat    = self.split_feat[idx]
        label   = self.split_label[idx]
        # idx   = idx.tolist()

        optor_index = torch.cat((idx_raw.reshape(1,n_idx),torch.tensor(range(n_idx)).reshape(1,n_idx)),dim=0)
        optor_value = torch.ones(n_idx)
        optor_shape = torch.Size([self.n,n_idx])
        optor       = torch.sparse_coo_tensor(optor_index, optor_value, optor_shape)
        sub_A       = torch.sparse.mm(torch.sparse.mm(optor.t(), self.Adj), optor)

        return (feat, label, sub_A)

    def get_batch(self, i):
        idx       = torch.where(torch.tensor(self.batch_labels) == i)[0]
        batch_i   = self.getitem(idx)
        return batch_i
