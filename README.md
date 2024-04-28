# Fast Graph Conensation with Structure-based Neural Tangent Kernel


The rapid development of Internet technology has given rise to a vast amount of graph-structured data. Graph Neural Networks (GNNs), as an effective method for various graph mining tasks, incurs substantial computational resource costs when dealing with large-scale graph data. A data-centric manner solution is proposed to condense the large graph dataset into a smaller one without sacrificing the predictive performance of GNNs. However, existing efforts condense graph-structured data through a computational intensive bi-level optimization architecture also suffer from massive computation costs. In this paper, we propose reforming the graph condensation problem as a Kernel Ridge Regression (KRR) task instead of iteratively training GNNs in the inner loop of bi-level optimization. More specifically, We propose a novel dataset condensation framework (GC-SNTK) for graph-structured data, where a Structure-based Neural Tangent Kernel (SNTK) is developed to capture the topology of graph and serves as the kernel function in KRR paradigm. Comprehen- sive experiments demonstrate the effectiveness of our proposed model in accelerating graph condensation while maintaining high prediction performance.



## Requirements

python==3.7.0 \
torch==1.12.1+cu116 \
numpy==1.21.6  \
scipy==1.7.3 \
scikit_learn==1.0.2 \
ogb==1.3.6 \
torch_geometric==2.3.0


## Training

For the **Cora** dataset, run: 
```python
cd GCSNTK
python main.py --dataset Cora --K 2 --L 2 --ridge 1e0 --lr 0.01 --epochs 200 --cond_ratio 0.5
```

For the **Pubmed** dataset, run: 
```python
cd GCSNTK
python main.py --dataset Pubmed --K 2 --L 2 --ridge 1e-3 --lr 0.01 --epochs 200 --cond_ratio 0.5
```
For the **Flickr** dataset, run: 
```python
cd GCSNTK_Flickr
python main.py --dataset Flickr --K 1 --L 1 --ridge 1e-5 --lr 0.001 --epochs 200 --cond_size 44
```
For the **Ogbn-arxiv** dataset, run: 
```python
cd GCSNTK_ogbn_arxiv
python main.py --dataset ogbn-arxiv --K 1 --L 1 --ridge 1e-5 --lr 0.001 --epochs 200 --cond_size 90
```

## Note:

If you use Conda to manage your environment, it is better to use 
```python
conda install -c conda-forge ogb
```
to install the ogb pakage.



