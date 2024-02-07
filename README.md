# Motif-Consistent Counterfactuals with Adversarial Refinement for Graph-level Anomaly Detection

This is an anonymous repository hosting the code of MotifCAR for double-blind review.

## Datasets
1. Reddit-Binary
2. Reddit-Multi-5k
3. IMDB-Binary
4. IMDB-Multi

All datasets used in this paper are from previous works and the graph classification datasets can be downloaded from [GraphClsDatasets](https://chrsmrrs.github.io/datasets/).
## Usage

### Environment
Install Python 3.8.

```python
pip install -r requirements.txt
```


### Training and Test

```python
python src/main.py -dataset {['IMDB-BINARY', 'REDDIT-BINARY','REDDIT-MULTI-5K','IMDB-MULTI']} -gnn_layer {['GCN','GAT','GraphSAGE']}
```
#### We also use dataset IMDB-BINARY for testing demonstration.

```python
python src/main.py -dataset 'IMDB-BINARY' -gnn_layer 'GCN'
```
#### RESULT
The following is the Result obtained after testing the IMDB-BINARY dataset.
<p align="center">
<img src=".\fig\run_result.png" width="500" height = "130" alt="result" align=center />
</p>