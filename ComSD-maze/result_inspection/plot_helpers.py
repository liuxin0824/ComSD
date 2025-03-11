import math
import matplotlib.pyplot as plt


def plot_stats_by_name(exp, stat_list, labels=None, ncols=4, figsize=(5, 3), titlesize=12):
    assert len(stat_list) > 0
    for stat_name in stat_list:
        assert stat_name in exp.summary_keys, "{} is not a valid stat key".format(stat_name)

    ncols = min(ncols, len(stat_list))
    nrows = math.ceil(len(stat_list) / ncols)
    fig, axarr = plt.subplots(ncols=ncols, nrows=nrows, figsize=(figsize[0] * ncols, figsize[1] * nrows))
    axarr = axarr.flatten() if (ncols * nrows) > 1 else [axarr]

    for ax_idx, stat_name in enumerate(stat_list):
        ax = axarr[ax_idx]
        stat_idx = exp.summary_keys.index(stat_name)
        exp.plot_stat(stat_idx=stat_idx, ax=ax)
        ax.set_title(stat_name if labels is None else labels[ax_idx], fontsize=titlesize)

    for ax_idx in range(len(stat_list), len(axarr)):
        axarr[ax_idx].axis('on')
