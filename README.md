Provides the official implementation of [On UMAP's true loss function](https://arxiv.org/abs/2103.14608).

## TL,DR:
Adds loss logging capabilities to UMAP and validates that UMAP's optimization procedure optimizes a different loss than 
purported. Further information, for instance on how this can create artifacts in UMAP visualizations can be found in the 
paper. 

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

Download the C. elegans, PBMC and lung cancer datasets
``` 
cd data
python get_c_elegans.py
python get_PBMC.py
python get_lung_cancer_data.py
```

Download the CIFAR-10 dataset and a pretrained Resnet50 to extract features (CUDA-ready GPU needed)
``` 
python get_cifar10_resnet50_features.py
```

If UMAP losses shall be logged on large datasets, a CUDA-ready GPU is needed.

## Reproduce the results of the paper
To reproduce the results of the paper, run the notebooks below from a `jupyter notebook` launched in `notebooks`.
  * `UMAP_*.ipynb`  produces the visualizations in the paper; should be run first.
  * `*_histograms.ipynb` produces the histograms in the paper.
  * `embedding_quality_measures.ipynb` computes the measures for the quality of embeddings.
  * `run_times.ipynb` computes the run times of the key experiments.
  * `stability` Computes loss values given in the paper over several runs with differen random seeds.

The figures will be saved in `data/figures` and other output in `data/DATASET`.

## Extensions over umap-learn
Our implementation is extends version 0.5.0 of https://github.com/lmcinnes/umap. The added functionality provides 
four new arguments to the `UMAP` class:
  * `graph` Allows to specify high-dimensional similarities as part of the input instead of inferring them from the data
  * `push_tail` Specifies whether or not the tail of a negative sample should be pushed away from its head
  * `log_losses` Specifies if and how losses should be logged
  * `log_samples` Specifies whether sampled edges and negative samples should be logged
  * `log_embeddings` Specifies whether intermediate embeddings should be logged

Our changes are confined to `umap_.py` and `layout.py` and two new files `my_utils.py` and `my_plots.py`.