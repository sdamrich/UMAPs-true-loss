import numpy as np
import numba
import scipy.sparse
from pykeops.torch import LazyTensor
import torch
import umapns
from sklearn.metrics import pairwise_distances
from scipy.stats import pearsonr, spearmanr


# Contains utility function, including for computing similarities and losses


def corr_pdist_subsample(x, y, sample_size, seed=0, metric="euclidean"):
    """
    Computes correlation between pairwise distances among the x's and among the y's
    :param x: array of positions for x
    :param y: array of positions for y
    :param sample_size: number of points to subsample from x and y for pairwise distance computation
    :param seed: random seed
    :param metric: Metric used for distances of x, must be a metric available for sklearn.metrics.pairwise_distances
    :return: tuple of Pearson and Spearman correlation coefficient
    """
    np.random.seed(seed)
    sample_idx = np.random.randint(len(x), size=sample_size)
    x_sample = x[sample_idx]
    y_sample = y[sample_idx]

    x_dists = pairwise_distances(x_sample, metric=metric).flatten()
    y_dists = pairwise_distances(y_sample, metric="euclidean").flatten()

    pear_r, _ = pearsonr(x_dists, y_dists)
    spear_r, _ = spearmanr(x_dists, y_dists)
    return pear_r, spear_r




def acc_kNN(x, y, k, metric="euclidean"):
    """
    Computes the accuracy of k nearest neighbors between x and y.
    :param x: array of positions for first dataset
    :param y: arraoy of positions for second dataset
    :param k: number of nearest neighbors considered
    :param metric: Metric used for distances of x, must be a metric available for sklearn.metrics.pairwise_distances
    :return: Share of x's k nearest neighbors that are also y's k nearest neighbors
    """
    x_kNN = scipy.sparse.coo_matrix((np.ones(len(x)*k),
                                    (np.repeat(np.arange(x.shape[0]), k),
                                     kNN_graph(x, k, metric=metric).cpu().numpy().flatten())),
                                    shape=(len(x), len(x)))
    y_kNN = scipy.sparse.coo_matrix((np.ones(len(y)*k),
                                    (np.repeat(np.arange(y.shape[0]), k),
                                     kNN_graph(y, k).cpu().numpy().flatten())),
                                    shape=(len(y), len(y)))
    overlap = x_kNN.multiply(y_kNN)
    matched_kNNs = overlap.sum()
    return matched_kNNs / (len(x) * k)

def kNN_graph(x, k, metric="euclidean"):
    """
    Pykeops implementation of a k nearest neighbor graph
    :param x: array containing the dataset
    :param k: number of neartest neighbors
    :param metric: Metric used for distances of x, must be "euclidean" or "cosine".
    :return: array of shape (len(x), k) containing the indices of the k nearest neighbors of each datapoint
    """
    x = torch.tensor(x).to("cuda").contiguous()
    x_i = LazyTensor(x[:, None])
    x_j = LazyTensor(x[None])
    if metric == "euclidean":
        dists = ((x_i - x_j)**2).sum(-1)
    elif metric == "cosine":
        scalar_prod = (x_i * x_j).sum(-1)
        norm_x_i = (x_i**2).sum(-1).sqrt()
        norm_x_j = (x_j**2).sum(-1).sqrt()
        dists = 1 - scalar_prod / (norm_x_i * norm_x_j)
    else:
        raise NotImplementedError(f"Metric {metric} is not implemented.")
    knn_idx = dists.argKmin(K=k+1, dim=0)[:, 1:] # use k+1 neighbours and omit first, which is just the point itself
    return knn_idx

def kNN_dists(x, k):
    """
    Pykeops implementation for computing the euclidean distances to the k nearest neighbors
    :param x: array dataset
    :param k: int, number of nearest neighbors
    :return: array of shape (len(x), k) containing the distances to the k nearest neighbors for each datapoint
    """
    x = torch.tensor(x).to("cuda").contiguous()
    x_i = LazyTensor(x[:, None])
    x_j = LazyTensor(x[None])
    knn_dists = ((x_i - x_j) ** 2).sum(-1).Kmin(K=k + 1, dim=0)[:, 1:].sqrt()  # use k+1 neighbours and omit first, which is just the point
    return knn_dists

def compute_loss_table(umapper, data):
    """
    Computes the losses for different combinations of high- and low-dimensional similarites and for different loss
    methods.
    :param umapper: UMAP instance
    :param data: original data
    :return: dictionary of losses
    """
    filtered_graph = filter_graph(umapper.graph_, umapper.n_epochs)
    high_sim = np.array(filtered_graph.todense())
    a, b = umapns.umap_.find_ab_params(spread=umapper.spread, min_dist=umapper.min_dist)

    low_sim_embd = compute_low_dim_psims(umapper.embedding_, a ,b)
    low_sim_data = compute_low_dim_psims(data, a, b)
    target_sim = get_target_sim(high_sim, negative_sample_rate=umapper.negative_sample_rate)

    loss_high_low_embd = reproducing_loss(high_sim, low_sim_embd)
    loss_high_high = reproducing_loss(high_sim, high_sim)
    loss_high_0 = reproducing_loss(high_sim, np.eye(len(high_sim)))
    loss_high_low_data = reproducing_loss(high_sim, low_sim_data)

    eff_loss_low_embd = expected_loss(high_sim,
                                         low_sim_embd,
                                         negative_sample_rate=umapper.negative_sample_rate)
    eff_loss_target = expected_loss(high_sim,
                                       target_sim,
                                       negative_sample_rate=umapper.negative_sample_rate)
    eff_loss_0 = expected_loss(high_sim,
                                  np.eye(len(high_sim)),
                                  negative_sample_rate=umapper.negative_sample_rate)
    eff_loss_low_data = expected_loss(high_sim,
                                         low_sim_data,
                                         negative_sample_rate=umapper.negative_sample_rate)
    return {"loss_high_high": (*loss_high_high, loss_high_high[0] + loss_high_high[1]),
            "loss_high_0": (*loss_high_0, loss_high_0[0] + loss_high_0[1]),
            "loss_high_low_embd": (*loss_high_low_embd, loss_high_low_embd[0] + loss_high_low_embd[1]),
            "loss_high_low_data": (*loss_high_low_data, loss_high_low_data[0] + loss_high_low_data[1]),
            "eff_loss_target": (*eff_loss_target, eff_loss_target[0] + eff_loss_target[1]),
            "eff_loss_0": (*eff_loss_0, eff_loss_0[0] + eff_loss_0[1]),
            "eff_loss_low_embd": (*eff_loss_low_embd, eff_loss_low_embd[0] + eff_loss_low_embd[1]),
            "eff_loss_low_data": (*eff_loss_low_data, eff_loss_low_data[0] + eff_loss_low_data[1])
            }


def filter_graph(graph, n_epochs):
    """
    Filters graph, so that no entry is too low to yield at least one sample during optimization.
    :param graph: sparse matrix holding the high-dimensional similarities
    :param n_epochs: int Number of optimization epochs
    :return:
    """
    graph = graph.copy()
    graph.data[graph.data < graph.data.max() / float(n_epochs)] = 0
    graph.eliminate_zeros()
    return graph


## data generation
def get_ring(n, r, var=0, noise="gauss"):
    """
    Create toy ring dataset
    :param n: int Number of samples
    :param r: float Radius of ring
    :param var: float Controls the width of the ring
    :param noise: string Type of noise model. "gauss" Gaussian noise, "uniform" uniform distribution in ring
    :return:
    """
    angles = 2*np.pi * np.arange(n) / n
    points = r * np.stack([np.sin(angles), np.cos(angles)])

    if noise=="gauss":
        noise = np.random.normal(0.0, var, size=points.shape)
    elif noise=="uniform":
        noise_r = np.sqrt(np.random.uniform(0, 1, size=points.shape[1])) * var
        noise_angle = np.pi * np.random.uniform(0, 2, size=points.shape[1])
        noise = np.stack([noise_r * np.sin(noise_angle),
                          noise_r * np.cos(noise_angle)])
    else:
        raise NotImplementedError(f"noise {noise} not supported.")
    points += noise
    return points.T

## similarities
@numba.njit()
def low_dim_sim_dist(x, a, b, squared=False):
    """
    Smooth function from distances to low-dimensional simiarlity. Compatible with numba.njit
    :param x: np.array pairwise distances
    :param a: float shape parameter a
    :param b: float shape parameter b
    :param squared: bool whether input distances are already squared
    :return: np.array low-dimensional similarities
    """
    if not squared:
        return 1.0 / (1.0 + a * x ** (2.0 * b))
    return 1.0 / (1.0 + a * x ** b)

def low_dim_sim_keops_dist(x, a, b, squared=False):
    """
    Smooth function from distances to low-dimensional simiarlity. Compatible with keops
    :param x: keops.LazyTensor pairwise distances
    :param a: float shape parameter a
    :param b: float shape parameter b
    :param squared: bool whether input distances are already squared
    :return: np.array low-dimensional similarities
    """
    if not squared:
        return 1.0 / (1.0 + a * x ** (2.0 * b))
    return 1.0 / (1.0 + a * x ** b)

def compute_low_dim_psim_keops_embd(embedding, a, b):
    """
    Computes low-dimensional pairwise similarites from embeddings via keops.
    :param embedding: np.array embedding coordinates
    :param a: float shape parameter a
    :param b: float shape parameter b
    :return: keops.LazyTensor low-dimensional similarities
    """
    lazy_embd_i = LazyTensor(torch.tensor(embedding[:, None, :], device="cuda"))
    lazy_embd_j = LazyTensor(torch.tensor(embedding[None], device="cuda"))
    a = LazyTensor(torch.tensor(a, device="cuda", dtype=torch.float32))
    b = LazyTensor(torch.tensor(b, device="cuda", dtype=torch.float32))
    sq_dists = ((lazy_embd_i-lazy_embd_j) ** 2).sum(-1)
    return low_dim_sim_keops_dist(sq_dists, a, b, squared=True)

def true_sim(x, min_dist, spread):
    return np.ones_like(x) * (x <= min_dist) + np.exp(-(x - min_dist) / spread) * (x > min_dist)

@numba.njit()
def compute_low_dim_psims(embedding, a, b):
    """
    Computes low-dimensional pairwise similarites from embeddings via numba.
    :param embedding: np.array embedding coordinates
    :param a: float shape parameter a
    :param b: float shape parameter b
    :return: np.array low-dimensional similarities
    """
    embd_dim = embedding.shape[1]
    n_points = embedding.shape[0]
    # numba does not support np.array[None], so use reshape
    squared_dists = ((embedding.reshape((n_points, 1, embd_dim))
                      - embedding.reshape((1, n_points, embd_dim)))**2).sum(-1)
    return low_dim_sim_dist(squared_dists, a, b, squared=True)


def compute_low_dim_sims(embedding1, embedding2, a, b):
    """
    Computes low-dimensional similarites between two sets of embeddings.
    :param embedding1: np.array Coordinates of first set of embeddings
    :param embedding2: np.array Coordinates of second set of embeddings
    :param a: float shape parameter a
    :param b: float shape parameter b
    :return: np.array low-dimensional similarities
    """
    assert embedding1.shape == embedding2.shape
    squared_dists = ((embedding1 - embedding2) ** 2).sum(-1)
    return low_dim_sim_dist(squared_dists, a, b, squared=True)



## loss functions
@numba.njit()
def my_log(x, eps=1e-4):
    """
        Safe version of log
    """
    return np.log(np.minimum(x + eps, 1.0))

# expects dense np.arrays
def reproducing_loss(high_sim, low_sim):
    """
    UMAPs original loss function, numpy implementation
    :param high_sim: np.array or scipy.sparse.coo_matrix high-dimensional similarities
    :param low_sim: np.array low-dimensional similarities
    :return: tuple of floats, attractive and repulsive loss
    """
    return BCE_loss(high_sim_a = high_sim,
                    high_sim_r = high_sim,
                    low_sim = low_sim)


def expected_loss(high_sim, low_sim, negative_sample_rate, push_tail=True):
    """
    UMAP's true loss function, numpy implementation
    :param high_sim: np.array or scipy.sparse.coo_matrix high-dimensional similarities
    :param low_sim: np.array low-dimensional similarities
    :param negative_sample_rate: int Number of negative samples per positive sample
    :param push_tail: bool Whether tail of negative sample is pushed away from its head.
    :return:
    """
    # get decreased repulsive weights
    high_sim_r, _ = get_UMAP_push_weight(high_sim, negative_sample_rate=negative_sample_rate, push_tail=push_tail)
    if isinstance(high_sim_r, np.ndarray):
        high_sim_r = 1-high_sim_r
    elif isinstance(high_sim_r, scipy.sparse.coo_matrix):
        high_sim_r.data = 1-high_sim_r.data
    return BCE_loss(high_sim_a = high_sim,
                    high_sim_r = high_sim_r,
                    low_sim  = low_sim)

def BCE_loss(high_sim_a, high_sim_r, low_sim):
    """
    General BCE loss between the high-dimensional similarities and the low dimensional similarities, numpy implementation
    :param high_sim_a: np.array or scipy.sparse.coo_matrix attractive high-dimensional similarities
    :param high_sim_r: np.array or scipy.sparse.coo_matrix repulsive high-dimensional similarities
    :param low_sim: np.array low-dimensional similarities
    :return: tuple of floats attractive and repulsive parts of BCE loss
    """
    if type(high_sim_a) == type(high_sim_r) == type(low_sim) == np.ndarray:
        loss_a = (high_sim_a * my_log(low_sim)).sum()
        loss_r = ((1-high_sim_r) * my_log(1 - low_sim)).sum()

    elif type(high_sim_a) == type(high_sim_r) == type(low_sim) == scipy.sparse.coo_matrix:
        assert np.all(high_sim_a.row == high_sim_r.row) and np.all(high_sim_a.row == low_sim.row) and \
               np.all(high_sim_a.col == high_sim_r.col) and np.all(high_sim_a.col == low_sim.col), \
            "Sparse matrices without matching indices for nonzero elements are not supported."
        loss_a = (high_sim_a.data * my_log(low_sim.data)).sum()
        loss_r = ((1 - high_sim_r.data) * my_log(1-low_sim.data)).sum() # 1 * log(1) = 0
    else:
        raise NotImplementedError(f"high_sim_a, high_sim_r, low_sim have types {type(high_sim_a)}, {type(high_sim_r)}"
                                  f"and {type(low_sim)}")
    return -loss_a, -loss_r

# keops implementations:
def KL_divergence(high_sim,
                  a,
                  b,
                  embedding,
                  eps=1e-12,
                  norm_over_pos=True):
    """
    Computes the KL divergence between the high-dimensional p and low-dimensional
    similarities q. The latter are inferred from the embedding.
    KL = sum_ij p_ij * log(p_ij / q_ij) = sum_ij p_ij * log(p_ij) - sum_ij p_ij * log(q_ij)
    --> Only ij with p_ij > 0 need to be considered as 0* log(0) is 0 by
    convention.
    :param high_sim: scipy.sparse.coo_matrix high-dimensional similarities
    :param a: float shape parameter a
    :param b: float shape parameter b
    :param embedding: np.array Coordinates of embeddings
    :return: float, KL divergence
    """
    heads = high_sim.row
    tails = high_sim.col

    # compute low dimensional simiarities on the edges with positive p_ij
    sq_dist_pos_edges = ((embedding[heads]-embedding[tails])**2).sum(-1)
    low_sim_pos_edges = low_dim_sim_keops_dist(sq_dist_pos_edges,
                                               a,
                                               b,
                                               squared=True)
    if norm_over_pos:
        low_sim_pos_edges_norm = low_sim_pos_edges / low_sim_pos_edges.sum()
    else:
        total_low_sim = compute_low_dim_psim_keops_embd(embedding,
                                                        a,
                                                        b).sum(1).cpu().numpy().sum()
        low_sim_pos_edges_norm = low_sim_pos_edges / total_low_sim



    high_sim_pos_edges_norm = high_sim.data / high_sim.data.sum()

    neg_entropy = (high_sim_pos_edges_norm * my_log(high_sim_pos_edges_norm, eps)).sum()
    cross_entropy = - (high_sim_pos_edges_norm * my_log(low_sim_pos_edges_norm, eps)).sum()
    return cross_entropy + neg_entropy


def reproducing_loss_keops(high_sim: scipy.sparse.coo_matrix,
                           a,
                           b,
                           embedding,
                           eps=1e-4):
    """
    UMAPs original loss function, keops implementation
    :param high_sim: scipy.sparse.coo_matrix high-dimensional similarities
    :param a: float shape parameter a
    :param b: float shape parameter b
    :param embedding: np.array Coordinates of embeddings
    :param eps: float Small epsilon value for log
    :return: tuple of floats, attractive and repulsive loss
    """
    heads = high_sim.row
    tails = high_sim.col

    # compute low dimensional similarities from embeddings
    sq_dist_pos_edges = ((embedding[heads]-embedding[tails])**2).sum(-1)
    low_sim_pos_edges = low_dim_sim_keops_dist(sq_dist_pos_edges, a, b, squared=True)
    low_sim = compute_low_dim_psim_keops_embd(embedding, a, b)

    loss_a = (high_sim.data * my_log(low_sim_pos_edges)).sum()

    inv_low_sim = 1 - (low_sim - eps).relu()  # pykeops compatible version of min(1-low_sim+eps, 1)
    # for repulsive term compute loss with keops and all high_sims = 1 and substract the sparse positive high_sims
    loss_r = (inv_low_sim).log().sum(1).sum()
    loss_r -= ((1 - high_sim.data) * my_log(1 - low_sim_pos_edges)).sum()
    return -loss_a, float(-loss_r)

def expected_loss_keops(high_sim: scipy.sparse.coo_matrix,
                        a,
                        b,
                        negative_sample_rate,
                        embedding,
                        push_tail=False,
                        eps=0.0001):
    """
    UMAP's true loss function, keops implementation
    :param high_sim: scipy.sparse.coo_matrix high-dimensional similarities
    :param a: float shape parameter a
    :param b: float shape parameter b
    :param negative_sample_rate: int Number of negative samples per positive sample
    :param embedding: np.array Coordinates of embeddings
    :param push_tail: bool Whether tail of negative sample is pushed away from its head.
    :param eps: float Small epsilon value for log
    :return: tuple of floats, attractive and repulsive loss
    """
    heads = high_sim.row
    tails = high_sim.col

    # compute low dimensional similarities from embeddings
    sq_dist_pos_edges = ((embedding[heads]-embedding[tails])**2).sum(-1)
    low_sim_pos_edges = low_dim_sim_keops_dist(sq_dist_pos_edges, a, b, squared=True)
    low_sim = compute_low_dim_psim_keops_embd(embedding, a, b)

    loss_a = (high_sim.data * my_log(low_sim_pos_edges, eps)).sum()

    # get decreased repulsive weights
    push_weights = get_UMAP_push_weight_keops(high_sim, negative_sample_rate, push_tail)[0]

    inv_low_sim = 1 - (low_sim - eps).relu() # pykeops compatible version of min(1-low_sim+eps, 1)
    loss_r = (push_weights * inv_low_sim.log()).sum(1).sum()

    return -loss_a, float(-loss_r)

def get_UMAP_push_weight_keops(high_sim, negative_sample_rate, push_tail=False):
    """
    Computes the effective, decreased repulsive weights and the degrees of each node, keops implementation
    :param high_sim: np.array or scipy.sparse.coo_matrix high-dimensional similarities
    :param negative_sample_rate: int Number of negative samples per positive sample
    :param push_tail: bool Whether tail of negative sample is pushed away from its head.
    :return: tuple of keops.LazyTensor and np.array reduced effective repulsive weights and degrees
    """
    n_points = LazyTensor(torch.tensor(high_sim.shape[0], device="cuda", dtype=torch.float32))

    degrees = np.array(high_sim.sum(-1)).ravel()
    degrees_t = torch.tensor(degrees, device="cuda", dtype=torch.float32)
    degrees_i = LazyTensor(degrees_t[:, None, None])
    degrees_j = LazyTensor(degrees_t[None, :, None])

    if push_tail:
        # np.array[None] does not work for numba, so use reshape instead
        return negative_sample_rate * (degrees_i + degrees_j)/(2*n_points), degrees
    return negative_sample_rate * degrees_i * LazyTensor(torch.ones((1,len(degrees), 1), device="cuda"))/n_points, degrees

def get_UMAP_push_weight(high_sim, negative_sample_rate, push_tail=False):
    """
    Computes the effective, decreased repulsive weights and the degrees of each node, numpy implementation
    :param high_sim: np.array or scipy.sparse.coo_matrix high-dimensional similarities
    :param negative_sample_rate: int Number of negative samples per positive sample
    :param push_tail: bool Whether tail of negative sample is pushed away from its head.
    :return: tuple of np.array or scipy.sparse.coo_matrix and np.array reduced effective repulsive weights and degrees
    """
    degrees = np.array(high_sim.sum(-1)).ravel()
    n_points = high_sim.shape[0]
    if isinstance(high_sim, np.ndarray):
        if push_tail:
            # np.array[None] does not work for numba, so use reshape instead
            return negative_sample_rate * (degrees.reshape((-1, 1)) + degrees.reshape((1, -1)))/(2*n_points), degrees
        return (negative_sample_rate * np.tile(degrees, (len(degrees), 1))/n_points).T, degrees
    elif isinstance(high_sim, scipy.sparse.coo_matrix):
        if push_tail:
            push_weights =  negative_sample_rate * (degrees[high_sim.row] + degrees[high_sim.col]) / (2*n_points)
        else:
            push_weights = negative_sample_rate * degrees[high_sim.row] / n_points
        return scipy.sparse.coo_matrix((push_weights, (high_sim.row, high_sim.col)),
                                       shape=(n_points, n_points)), degrees
    else:
        print(type(high_sim))
        raise NotImplementedError


def get_target_sim(high_sim, negative_sample_rate=5):
    """
    Computes the true target similarities of UMAP
    :param high_sim: np.array or scipy.sparse.coo_matrix high-dimensional similarities
    :param negative_sample_rate: int Number of negative samples per positive sample
    :return: np.array or scipy.sparse.coo_matrix UMAP's true target similarities
    """
    push_weight, _ = get_UMAP_push_weight(high_sim, negative_sample_rate, push_tail=True)
    if isinstance(high_sim, np.ndarray):
        return high_sim / (high_sim + push_weight)
    elif isinstance(high_sim, scipy.sparse.coo_matrix):
        return scipy.sparse.coo_matrix((high_sim.data / (high_sim.data + push_weight.data),
                                        (high_sim.row, high_sim.col)), shape=high_sim.shape)
    else:
        print(type(high_sim))
        raise NotImplementedError






