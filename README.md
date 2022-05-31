This repo extends the official implementation of 
[On UMAP's true loss function](https://arxiv.org/abs/2103.14608) which can be found [here](https://github.com/hci-unihd/UMAPs-true-loss).

## TL,DR:
Adds ablation options for learning rate annealing and numerical stability tricks.

## Getting started
Clone the repository
```
git clone https://github.com/hci-unihd/UMAPs-true-loss
```

Change into the directory, create a conda environment from `environment.yml` and activate it
```
conda env create -f environment.yml
conda activate umaps_true_loss
```

Download and install `vis_utils` package:
```
cd ..
git clone https://github.com/sdamrich/vis_utils
cd vis_utils
python setup.py install
```


Install the extension of the UMAP package
```
python setup.py install
```


If UMAP losses shall be logged on large datasets, a CUDA-ready GPU is needed.

## Extensions over umap-learn
Our implementation is extends version 0.5.0 of https://github.com/lmcinnes/umap. The added functionality provides 
eight new arguments to the `UMAP` class:
  * `graph` Allows to specify high-dimensional similarities as part of the input instead of inferring them from the data
  * `push_tail` Specifies whether the tail of a negative sample should be pushed away from its head
  * `log_losses` Specifies if and how losses should be logged
  * `log_samples` Specifies whether sampled edges and negative samples should be logged
  * `log_embeddings` Specifies whether intermediate embeddings should be logged
  * `anneal_lr` Specifies whether the learning rate should be annealed.
  * `eps` Small parameter added to the denominator of a term in the repulsive force to avoid numerical problems.
  * `log_norm` Specifies whether to log the partition function of the embedding.
  * `log_kl` Specifies whether to log the Kullback-Leibler divergence of the embedding.
Our changes are confined to `umap_.py` and `layout.py`.