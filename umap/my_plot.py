import numpy as np
import matplotlib.pyplot as plt
import pykeops
import matplotlib
from .my_utils import get_target_sim, compute_low_dim_psim_keops_embd, compute_low_dim_sims

# Contains functionality for convenient plotting of historgrams and loss functions

# Histograms
def hists_from_graph_embd(graph,
                     embedding,
                     a,
                     b,
                     negative_sample_rate=5,
                     n_bins=100,
                     hist_range=(0,1)
):
    """
    Computes histograms of high- and low-dimensional similarities and target similarities for all edges or just those
    with positive high-dimensional similarity
    :param graph: scipy.sparse.coo_matrix Graph of high-dimensional similarities
    :param embedding: np.array Coordinates of the embedding
    :param a: float shape parameter a
    :param b: float shape parameter b
    :param negative_sample_rate: int Number of negative samples per positive sample
    :param n_bins: int Number of bins
    :param hist_range: tuple of floats Lower and higher bound of the histogram range
    :return: tuple of histograms of high-dimensional, positive high-dimensional, target, positive target low-dimensional,
    low-dimensional similarities for positive high-dimensional similarities and the histograms' bins
    """

    # get target similarities from graph
    target_sim = get_target_sim(graph, negative_sample_rate=negative_sample_rate)

    # get (positive) low-dimesnional similarities from the embedding
    low_sim = compute_low_dim_psim_keops_embd(embedding, a, b)
    low_sim_pos = compute_low_dim_sims(embedding1=embedding[graph.row],
                                       embedding2=embedding[graph.col],
                                       a=a,
                                       b=b)

    # compute all histograms
    hist_high_pos, bins = np.histogram(graph.data,
                                       bins=n_bins,
                                       range=hist_range)
    hist_high = hist_high_pos.copy()
    hist_high[0] += np.prod(graph.shape) - graph.nnz

    hist_low, bins = histogram_keops(low_sim,
                                     bins=n_bins,
                                     hist_range=hist_range)

    hist_low_pos, bins = np.histogram(low_sim_pos,
                                      bins=n_bins,
                                      range=hist_range)

    hist_target_pos, bins = np.histogram(target_sim.data,
                                         bins=n_bins,
                                         range=hist_range)

    hist_target = hist_target_pos.copy()
    hist_target[0] += np.prod(target_sim.shape) - target_sim.nnz

    return hist_high, hist_high_pos, hist_target, hist_target_pos, hist_low, hist_low_pos, bins



def histogram_keops(t: pykeops.torch.LazyTensor,
                    hist_range=(0, 1),
                    bins=100):
    """
    Computs histogram from a pykeops.torch.LazyTensor
    :param t: pykeops.torch.LazyTensor Tensor holding the values over which a histogram is built
    :param hist_range: tuple of floats Lower and higher bound of the histogram range
    :param bins: int Number of bins
    :return: tuple of the historgram and the bins
    """
    n_bins = bins
    max_val = hist_range[1]
    min_val = hist_range[0]

    if t.max(0).max() > 1.0 or t.min(0).min() < 0.0:
        print("Warning: LazyTensor entries should be between {min_val} and {max_val}, but are between {t.min().min()} "
              "and {t.max().max()}. Only values between {min_val} and {max_val} will be considered.")

    bin_size = (max_val - min_val)/n_bins

    bins = min_val + np.arange(n_bins+1) * bin_size

    hist = np.zeros(n_bins)

    for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
        mask1 = 1 - (-t + float(low)).step()  # strictly greater than left bin border (can exclude 0 anyways)
        mask2 = (- t + float(high)).step()  # less or equal to right bin border
        hist[i] = int((mask1 * mask2).sum(1).sum())
    if np.prod(t.shape) != hist.sum():
        print(f"Historgram counts should match product of input tensor shapes, but are {np.prod(t.shape)} and "
              f"{int(hist.sum())}.")
    return hist.astype("int"), bins

# Loss curves
def get_mean_std_dev_losses(umappers):
    """
    compute mean and std of losses over different umapper instances, e.g. with different random seeds
    :param umappers: list of umapper instances, needs to have aux_data for actual, effective and purported losses and
                     attractive and repulsive losses
    :return: tuple of mean losses and std dev losses for each loss type and method
    """
    # actual losses
    losses_a = np.array([umapper.aux_data["loss_a"] for umapper in umappers])
    mean_loss_a = losses_a.mean(axis=0)
    std_loss_a = losses_a.std(axis=0)

    losses_r = np.array([umapper.aux_data["loss_r"] for umapper in umappers])
    mean_loss_r = losses_r.mean(axis=0)
    std_loss_r = losses_r.std(axis=0)

    losses_total = losses_a + losses_r
    mean_losses_total = losses_total.mean(axis=0)
    std_losses_total = losses_total.std(axis=0)

    # purported losses
    losses_a_reprod = np.array([umapper.aux_data["loss_a_reprod"] for umapper in umappers])
    mean_loss_a_reprod = losses_a_reprod.mean(axis=0)
    std_loss_a_reprod = losses_a_reprod.std(axis=0)

    losses_r_reprod = np.array([umapper.aux_data["loss_r_reprod"] for umapper in umappers])
    mean_loss_r_reprod = losses_r_reprod.mean(axis=0)
    std_loss_r_reprod = losses_r_reprod.std(axis=0)

    losses_total_reprod = losses_a_reprod + losses_r_reprod
    mean_losses_total_reprod = losses_total_reprod.mean(axis=0)
    std_losses_total_reprod = losses_total_reprod.std(axis=0)

    # effective losses
    losses_a_exp = np.array([umapper.aux_data["loss_a_exp"] for umapper in umappers])
    mean_loss_a_exp = losses_a_exp.mean(axis=0)
    std_loss_a_exp = losses_a_exp.std(axis=0)

    losses_r_exp = np.array([umapper.aux_data["loss_r_exp"] for umapper in umappers])
    mean_loss_r_exp = losses_r_exp.mean(axis=0)
    std_loss_r_exp = losses_r_exp.std(axis=0)

    losses_total_exp = losses_a_exp + losses_r_exp
    mean_losses_total_exp = losses_total_exp.mean(axis=0)
    std_losses_total_exp = losses_total_exp.std(axis=0)

    mean_losses = [
        [mean_loss_a, mean_loss_r, mean_losses_total],
        [mean_loss_a_exp, mean_loss_r_exp, mean_losses_total_exp],
        [mean_loss_a_reprod, mean_loss_r_reprod, mean_losses_total_reprod]
    ]

    std_losses = [
        [std_loss_a, std_loss_r, std_losses_total],
        [std_loss_a_exp, std_loss_r_exp, std_losses_total_exp],
        [std_loss_a_reprod, std_loss_r_reprod, std_losses_total_reprod]
    ]

    return mean_losses, std_losses



def plot_all_losses(umapper_aux_data, start=0, leg_loc=(0.54, 0.3)):
    """
    Wrapper of c_y_axis that creates np.arrays of all losses from the aux_data of a UMAP instance
    :param umapper_aux_data: dict Auxiliary data of a UMAP instance
    :param start: int First epoch to be plotted
    :return:
    """
    loss_a = np.array(umapper_aux_data["loss_a"])
    loss_r = np.array(umapper_aux_data["loss_r"])
    loss_a_exp = np.array(umapper_aux_data["loss_a_exp"])
    loss_r_exp = np.array(umapper_aux_data["loss_r_exp"])
    loss_a_reprod = np.array(umapper_aux_data["loss_a_reprod"])
    loss_r_reprod = np.array(umapper_aux_data["loss_r_reprod"])

    losses = [[loss_a, loss_r, loss_a + loss_r],
                         [loss_a_exp, loss_r_exp, loss_a_exp + loss_r_exp],
                         [loss_a_reprod, loss_r_reprod,
                          loss_a_reprod + loss_r_reprod]]
    loss_methods = ["actual", "effective", "purported"]
    loss_types = ["attractive", "repulsive", "total"]
    return cut_y_axis(losses, loss_methods, loss_types, start=start, log=False, leg_loc=leg_loc)

def cut_y_axis(losses: list,
               loss_methods: list,
               loss_types: list,
               losses_std: list=None,
               log=False,
               cut_low=None,
               start=0,
               end=None,
               step=1,
               leg_loc=(0.54, 0.3),
               line_width=3,
               spread=1):
    """
    Plots loss curves into a figure with cut y-axis
    :param losses: list of lists of floats Outer list over loss methods, inner list over loss types; holds loss values
    :param loss_methods:  list of stings Holds the loss methods ("actual", "effective" or "purported")
    :param loss_types: list of strings Holds the loss types ("attractive", "repulsive" or "total")
    :param losses_std: (optional, default None) list of list of floatsOuter list over loss methods, inner list over loss types; holds loss std dev
    :param log: bool (optional, default False) Whether log of the loss should be plotted
    :param cut_low: float (optional, default None) Value at which the lower axis is cut
    :param start: int (optional, default 0) First epoch to be plotted
    :param end: int (optional, default None) Last epoch to be plotted
    :return: figure with loss curves
    """
    # from https://matplotlib.org/stable/gallery/subplots_axes_and_figures/broken_axis.html

    assert len(losses) == len(loss_methods) # correct number of loss methods
    assert len(losses[0]) == len(loss_types)  # correct number of loss types per method
    end = len(losses[0][0]) if end is None else end
    x = np.arange(start, end, step)
    label_prefix = ""
    if log:
        losses = [ [np.log(loss) for loss in loss_data] for loss_data in losses]
        label_prefix = "log_"

    # style parameters
    alpha  = 1.0
    line_styles = {"actual": "solid",
                   "effective": (0, (5*spread, 6 * spread)),
                   "purported": (0, (1*spread, 1 * spread))}
    # get colors
    cmap = "tab10"
    colors = [list(matplotlib.cm.get_cmap(cmap)(k)) for k in range(matplotlib.cm.get_cmap(cmap).N)]
    color_dict = {"actual": {"attractive": colors[0], "repulsive": colors[1], "total": colors[2]},
                  "effective": {"attractive": colors[3], "repulsive": colors[4], "total": colors[5]},
                  "purported": {"attractive": colors[6], "repulsive": colors[7], "total": colors[8]}}


    fig, (ax1, ax2) = plt.subplots(2,
                                   1,
                                   sharex=True,
                                   figsize=(8,8),
                                   gridspec_kw={'height_ratios': [1.2, 3]})
    fig.subplots_adjust(hspace=0.07)  # adjust space between axes

    # plot the same data on axes
    data_ax1 = []
    data_ax2 = []

    # plot loss curves in the correct axis and fill if std dev is given
    for i, loss_method in enumerate(loss_methods):
        for j, loss_data in enumerate(losses[i]):
            if not loss_method == "purported" or loss_types[j] == "attractive":
                ax2.plot(x,
                         loss_data[start:end:step],
                         c=color_dict[loss_method][loss_types[j]],  #colors[len(loss_types) * i + j],
                         label=f"{label_prefix}{loss_method} {loss_types[j]}",
                         alpha=alpha,
                         linestyle=line_styles[loss_method],
                         linewidth=line_width)
                if losses_std is not None:
                    ax2.fill_between(x,
                             loss_data[start:end:step] - losses_std[i][j][start:end:step],
                             loss_data[start:end:step] + losses_std[i][j][start:end:step],
                             alpha = 0.25 * alpha,
                    )
                data_ax2.append(loss_data[start:end:step])
            else:
                line_style = line_styles[loss_method]
                if loss_types[j] == "total":
                    line_style = (0, (1*spread, 3*spread))
                ax1.plot(x,
                         loss_data[start:end:step],
                         c=color_dict[loss_method][loss_types[j]],  #colors[len(loss_types) * i + j],
                         label=f"{label_prefix}{loss_method} {loss_types[j]}",
                         alpha=alpha,
                         linestyle=line_style, #line_styles[loss_method],
                         linewidth=line_width)
                if losses_std is not None:
                    ax1.fill_between(x,
                             loss_data[start:end:step] - losses_std[i][j][start:end:step],
                             loss_data[start:end:step] + losses_std[i][j][start:end:step],
                             alpha = 0.25 * alpha,
                    )
                data_ax1.append(loss_data[start:end:step])

    # compute cut values
    data_ax1 = np.stack(data_ax1)
    data_ax2 = np.stack(data_ax2)
    y_max = data_ax1.max()*1.1
    cut_low = data_ax2.max()*1.1 if cut_low is None else cut_low
    cut_high = data_ax1.min()*0.9



    # zoom-in / limit the view to different portions of the data
    ax2.set_ylim(0, cut_low)
    ax1.set_ylim(cut_high, y_max)

    # hide the spines between ax and ax2
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()

    # join handles in one legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    leg = fig.legend(handles2+handles1,
                     labels2+labels1,
                     loc=leg_loc,
                     handlelength=3.0)


    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    return fig
