import sys
import copy
import os
import shutil
import numpy as np
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy

# initialize seaborn by default
sns.set()

# fetch terminal arguments
assert len(sys.argv) == 2
escrit = sys.argv[1]


r"""
Visualization
=============
This parts provides all visualizations and their utility codes.

- early_stop
  Update log data by early stopping.

- load
  Load log data from disk.

- mean_confidence_interval
  Get mean and confidence interval.

- lineviz
  Visualization line plots.
---------------------------
"""


def early_stop(data, key, cmp=lambda p, n: p > n):
    r"""Update data by early stoppping

    Args
    ----
    data : dict
        Data to update.
    key : str
        Key of data as early stoppping criterion.
    cmp : func
        Function used to compare.
        It should support cmp(prev, next) and returns True if next is better.

    Returns
    -------
    data : dict
        Updated data.

    """
    # update all column of data
    for i in range(len(data[key])):
        for j in range(1, len(data[key][i])):
            if isinstance(data[key][i][j], float) and cmp(data[key][i][j - 1], data[key][i][j]):
                pass
            elif isinstance(data[key][i][j], tuple) and cmp(data[key][i][j - 1][0], data[key][i][j][0]):
                pass
            else:
                for itr in data:
                    data[itr][i][j] = data[itr][i][j - 1]
    return data


def load(fmt):
    r"""Load log of multiple runs

    Args
    ----
    fmt : str
        Log filename formatter.

    Returns
    -------
    data : dict
        Data buffer.

    """
    # set configurations
    num_loops  = 20
    num_epochs = len(torch.load(fmt.format(0)))

    # allocate data buffer
    data_buffer = {
        'train_loss': [[None for _ in range(num_epochs)] for _ in range(num_loops)],
        'train_eval': [[None for _ in range(num_epochs)] for _ in range(num_loops)],
        'valid_eval': [[None for _ in range(num_epochs)] for _ in range(num_loops)],
        'track_vals': [[None for _ in range(num_epochs)] for _ in range(num_loops)],
    }

    # load data
    for i in range(num_loops):
        data = torch.load(fmt.format(i))
        assert len(data) == num_epochs
        for j, line in enumerate(data):
            data_buffer['train_loss'][i][j] = line[0]
            data_buffer['train_eval'][i][j] = line[1]
            data_buffer['valid_eval'][i][j] = line[2]
            data_buffer['track_vals'][i][j] = line[3]
        if escrit == 'loss':
            data_buffer = early_stop(data_buffer, 'train_loss')
        elif escrit == 'eval':
            data_buffer = early_stop(data_buffer, 'train_eval')
        elif escrit == 'none':
            pass
        else:
            raise RuntimeError('unsupported early stopping')
    data_buffer['train_loss'] = np.array(data_buffer['train_loss'])
    data_buffer['train_eval'] = np.array(data_buffer['train_eval'])
    data_buffer['valid_eval'] = np.array(data_buffer['valid_eval'])
    data_buffer['track_vals'] = np.array(data_buffer['track_vals'])

    # exclude the best and worst 10%
    kpids = np.argsort(data_buffer['valid_eval'][:, -1, REL])[2:-2]
    data_buffer['train_loss'] = data_buffer['train_loss'][kpids]
    data_buffer['train_eval'] = data_buffer['train_eval'][kpids]
    data_buffer['valid_eval'] = data_buffer['valid_eval'][kpids]
    data_buffer['track_vals'] = data_buffer['track_vals'][kpids]

    # update with tricks
    data_buffer = copy.deepcopy(data_buffer)
    return data_buffer


def mean_confidence_interval(data, confidence=0.95):
    r"""Get mean and confidence interval

    Args
    ----
    data : numpy.ndarray
        Data.
    confidence : float
        Confidence interval scale.

    Returns
    -------
    mean : float
        Mean.
    conf : float
        Confidence offset.

    """
    # compute directly
    assert len(data.shape) == 1
    n = len(data)
    mean, se = np.mean(data), scipy.stats.sem(data)
    conf = se * scipy.stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, conf


def make_table(task, rows, cols, table, ar):
    r"""make markdown table

    Args
    ----
    task : str
        Task name.
    rows : list
        Labels and corresponding file row info.
    cols : list
        Labels and corresponding file column info.
    table : dict
        Table of file prefices to fill by row and column info.
    ar : int
        Visualize Absolute or relative part.

    """
    # no longer use
    raise RuntimeError('no longer in usage')

    # load logs
    table = copy.deepcopy(table)
    for i in range(len(rows)):
        for j in range(len(cols)):
            table[i][j] = load(table[i][j].format(row=rows[i][0], col=cols[j][0]) + "_{}.pt")

    # get data to visualize
    if ar == ABS:
        Y = 'MSE of Predicted Call Failure Probability under High Load'
    elif ar == REL:
        Y = 'MAPE of Predicted Call Failure Probability under High Load'
    else:
        raise RuntimeError('only support absolute or relative loss')
    truncate = lambda buf: \
        buf['valid_eval'][:, :, ar].min(axis=1) # // buf['valid_eval'][:, -1, ar]
    for i in range(len(rows)):
        for j in range(len(cols)):
            table[i][j] = truncate(table[i][j])

    # print table
    print(Y)
    print('-' * len(Y))
    print('```')
    print()
    print('  ||', end='')
    for j in range(len(cols)):
        print(" {} |".format(cols[j][-1]), end='')
    print()
    print('  |' + '|'.join([':---:' for j in range(len(cols) + 1)]) + '|')
    for i in range(len(rows)):
        print("  | {} |".format(rows[i][-1]), end='')
        for j in range(len(cols)):
            mean, var = mean_confidence_interval(table[i][j])
            mean_exp = int(np.floor(np.log10(mean)))
            var_exp  = 0 if var == 0 else int(np.floor(np.log10(var)))
            mean_base = mean / (10 ** mean_exp)
            var_base  = 0.0 if var == 0 else var / (10 ** var_exp)
            print(" ${:.2f} \\times 10^{{{:d}}}$</br>$\\pm {:.2f} \\times 10^{{{:d}}}$ |".format(
                    mean_base, mean_exp, var_base, var_exp), end='')
        print()
    print()
    print('```')


def make_table_dual(task, rows, cols, table, item='valid_eval'):
    r"""make markdown table

    Args
    ----
    task : str
        Task name.
    rows : list
        Labels and corresponding file row info.
    cols : list
        Labels and corresponding file column info.
    table : dict
        Table of file prefices to fill by row and column info.
    item : str
        Visualize item.

    """
    # load logs
    table = copy.deepcopy(table)
    for i in range(len(rows)):
        for j in range(len(cols)):
            table[i][j] = load(table[i][j].format(row=rows[i][0], col=cols[j][0]) + "_{}.pt")
    table1 = copy.deepcopy(table)
    table2 = copy.deepcopy(table)

    # get data to visualize
    if item == 'valid_eval':
        truncate1 = lambda buf: \
            buf['valid_eval'][:, -1, ABS] # // buf['valid_eval'][:, :, ABS].min(axis=1)
        truncate2 = lambda buf: \
            buf['valid_eval'][:, -1, REL] # // buf['valid_eval'][:, :, REL].min(axis=1)
    else:
        truncate1 = lambda buf: \
            buf[item][:, -1]
        truncate2 = truncate1

    for i in range(len(rows)):
        for j in range(len(cols)):
            table1[i][j] = truncate1(table[i][j])
            table2[i][j] = truncate2(table[i][j])

    # mean, variance
    def getmv(tab, rid, cid, offset=0):
        mean, var = mean_confidence_interval(tab[rid][cid])
        mean_exp = int(np.floor(np.log10(mean)))
        var_exp  = 0 if var == 0 else int(np.floor(np.log10(var)))
        mean_base = mean / (10 ** mean_exp)
        var_base  = 0.0 if var == 0 else var / (10 ** var_exp)
        return mean_base, mean_exp - offset, var_base, var_exp - offset

    # formatter
    # // beg = ' & $\\begin{aligned} '
    # // end = ' \\end{aligned}$'
    # // fmt1 = "&{:.2f} \\times 10^{{{:d}}}"
    # // fmt2 = "\\pm &{:.2f} \\times 10^{{{:d}}}"
    # // fmt  = "{} \\\\ {}".format(fmt1, fmt2)
    beg = ' & $ '
    end = ' $'
    begbd = ' & $ \\mathbf{ '
    endbd = ' } $'
    fmt1 = "{:.2f} \\times 10^{{{:d}}}"
    fmt2 = "\\pm {:.2f} \\times 10^{{{:d}}}"
    fmt  = "{} {}".format(fmt1, fmt2)

    # print table
    print('-' * 20)
    print('```')
    print()
    print('[   Absolute   ]')
    for i in range(len(rows)):
        print("{} & {}".format(rows[i][-2], rows[i][-1]), end='')
        for j in range(len(cols)):
            if j == len(cols) - 1:
                print(begbd + fmt.format(*getmv(table1, i, j)) + endbd, end='')
            else:
                print(beg + fmt.format(*getmv(table1, i, j)) + end, end='')
        print(' \\\\')
    print('\\hline')
    print('[   Relative   ]')
    for i in range(len(rows)):
        print("{} & {}".format(rows[i][-2], rows[i][-1]), end='')
        for j in range(len(cols)):
            if j == len(cols) - 1:
                print(begbd + fmt.format(*getmv(table2, i, j, 2)) + endbd, end='')
            else:
                print(beg + fmt.format(*getmv(table2, i, j, 2)) + end, end='')
        print(' \\\\')
    print('\\hline')
    # // for i in range(len(rows)):
    # //     print(" {}".format(rows[i][-1]), end='')
    # //     for j in range(len(cols)):
    # //         print(beg + fmt.format(*getmv(table1, i, j)) + end, end='')
    # //     for j in range(len(cols)):
    # //         print(beg + fmt.format(*getmv(table2, i, j)) + end, end='')
    # //     print(' \\\\')
    # //     print('\\hline')
    print()
    print('```')


def lineviz(task, tags, ar, scale):
    r"""Visualization for line plots

    Args
    ----
    task : str
        Task name.
    tags : dict
        Labels and corresponding file prefices.
    ar : int
        Visualize Absolute or relative part.
    scale : str
        Scale of y axis.

    """
    # load logs
    dats = {}
    for key, prefix in tags.items():
        dats[key] = load(prefix + "_{}.pt")

    # get data to visualize
    if ar == ABS:
        X, Y = 'Epoch', 'MSE of Predicted Call Failure\nProbability under High Load'
    elif ar == REL:
        X, Y = 'Epoch', 'MAPE of Predicted Call Failure\nProbability under High Load'
    else:
        raise RuntimeError('only support absolute or relative loss')

    truncate = lambda buf: \
        pd.DataFrame(buf['valid_eval'][:, :, ar]).melt(var_name=X, value_name=Y)
    for key in tags:
        dats[key] = truncate(dats[key])

    # create canvas
    fig, ax = plt.subplots(1, 1)
    for key in tags:
        sns.lineplot(x=X, y=Y, data=dats[key], label=key, ax=ax, legend=False)
    ax.set_yscale(scale)
    ax.set_xlabel(ax.get_xlabel(), fontsize=15)
    ax.set_ylabel(ax.get_ylabel(), fontsize=15)
    ax.tick_params(axis='both', labelsize=15)
    ax.legend(loc='upper right', fontsize=15)
    fig.tight_layout()

    # save canvas
    fig.savefig(os.path.join('figs', escrit, 'png', "{}.png".format(task)), format='png')
    fig.savefig(os.path.join('figs', escrit, 'pdf', "{}.pdf".format(task)), format='pdf')
    plt.close(fig)


r"""
Benchmark
=========
Run all necessary visulizations.
--------------------------------
"""


if __name__ == '__main__':
    # set visualization constants
    ABS, REL = 0, 1

    # clean saving buffer
    bufdir = os.path.join('figs', escrit)
    if os.path.isdir(bufdir):
        shutil.rmtree(bufdir)
    else:
        pass
    os.makedirs(os.path.join(bufdir, 'png'))
    os.makedirs(os.path.join(bufdir, 'pdf'))

    r"""
    ---------
    Parameter
    ---------
    """

    # get parameter
    envs = [('mm1k_n0_mmmk' , -1, r'M/M/1/$K$ (fast-mix)'),
            ('mm1ks_n0_mmmk', -1, r'M/M/1/$K$ (slow-mix)'),
            ('mmmmr_n0_mmmk', -1, r'M/M/$m$/$m+r$       '),
            ('mmul_n0_mmul' , -2, r'M/M/Multiple/$K$    ')]
    # // envs = [('mm1k-small_n0_mmmk', -1, r'M/M/1/$K$ (Small Gap)       '),
    # //         ('mm1k-large_n0_mmmk', -1, r'M/M/1/$K$ (Large Gap)       '),
    # //         ('mmmmr_n0_mmmk'     , -1, r'M/M/$m$/$m + r$             '),
    # //         ('mmul-large_n0_mmul', -1, r'M/M/Multiple/$K$ (Large Gap)')]
    for folder, lr, label in envs:
        buf = os.path.join('logs', folder, "log_{}_rrinf_g-1_a2_l{}".format(folder, lr))
        buf = load(buf + "_{}.pt")['valid_eval']
        minid = np.argmin(buf[:, -1, REL])
        data = torch.load("logs/{0}/best2_{0}_rrinf_g-1_a2_l{1}_{2}.pt".format(folder, lr, minid))
        param = data['death_rate'].data.cpu().numpy().tolist()
        print("{} ({}): {}".format(label, buf[minid, -1, REL], param))

    r"""
    -------
    Heatmap
    -------
    """

    # visualize heatmaps
    envs = [('emu_n0_up',  0, r'Testbed Emulation')]
    for folder, lr, label in envs:
        buf = os.path.join('logs', folder, "log_{}_rrinf_g-1_a2_l{}".format(folder, lr))
        buf = load(buf + "_{}.pt")['valid_eval']
        minid = np.argmin(buf[:, -1, REL])
        data = torch.load("logs/{0}/best2_{0}_rrinf_g-1_a2_l{1}_{2}.pt".format(folder, lr, minid))
        mx = (data['lowtr_rate'] + data['noise']).data.cpu().numpy()
        for i in range(mx.shape[0]):
            for j in range(i, mx.shape[1]):
                if j > i + 1:
                    mx[i, j] = mx[i, j] if mx[i, j] != 0 else float('nan')
                else:
                    mx[i, j] = float('nan')
        fig, ax = plt.subplots(1, 1)
        cmap = cm.get_cmap('viridis')
        cmap.set_bad('black', alpha=0)
        sns.heatmap(mx, cmap=cmap)
        ax.set_aspect('equal')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.xaxis.tick_top()
        ax.yaxis.tick_left()
        ax.get_xticklabels()[1].set_color('green')
        ax.get_xticklabels()[2].set_color('green')
        ax.get_xticklabels()[-1].set_color('red')
        ax.get_yticklabels()[1].set_color('green')
        ax.get_yticklabels()[2].set_color('green')
        ax.get_yticklabels()[-1].set_color('red')
        ax.tick_params(axis='both', labelsize=12)
        fig.savefig(os.path.join('figs', escrit, 'png', "param_{}.png".format(folder)), format='png')
        fig.savefig(os.path.join('figs', escrit, 'pdf', "param_{}.pdf".format(folder)), format='pdf')
        plt.close(fig)
        print("{} ({}): Best Heatmap ID: {}".format(label, buf[minid, -1, REL], minid))

    r"""
    -----
    Prior
    -----
    """

    # visulization of prior
    envs = [('emu_n0', 0, 'log', r'Testbed Emulation')]
    # // envs = [('mm1k_n0' , -1, 'log', r'M/M/1/$K$'       ),
    # //         ('mmmmr_n0', -1, 'log', r'M/M/$m$/$m+r$'   ),
    # //         ('mmul_n0' , -2, 'log', r'M/M/Multiple/$K$')]
    # // envs = [('emu_n0'       ,  0, 'log', r'Testbed Emulation'           ),
    # //         ('mm1k-small_n0', -1, 'log', r'M/M/1/$K$ (Small Gap)'       ),
    # //         ('mm1k-large_n0', -1, 'log', r'M/M/1/$K$ (Large Gap)'       ),
    # //         ('mmmmr_n0'     , -1, 'log', r'M/M/$m$/$m + r$'             ),
    # //         ('mmul-large_n0', -1, 'log', r'M/M/Multiple/$K$ (Large Gap)')]
    legs = [('mmmk'  , 'M/M/m/K' , r'M/M/$m$/$K$'     ),
            ('up'    , 'Upper'   , r'Upper Triangular')]
    table = [["logs/{row}_{col}/log_{row}_{col}_rrinf_g-1_a2" + "_l{}".format(lr) for _ in legs] for _, lr, _, _ in envs]
    for folder, lr, scale, _ in envs:
        tags = {}
        for itr, label, _ in legs:
            if 'mmul' in folder and itr == 'mmmk':
                itr, label, _ = ('mmul'  , 'M/M/Mul/K' , r'M/M/Multiple/$K$')
            else:
                pass
            tags[label] = os.path.join('logs', "{}_{}".format(folder, itr), "log_{}_{}_rrinf_g-1_a2_l{}".format(folder, itr, lr))
        lineviz('prior_abs_' + folder, tags, ABS, scale=scale)
        lineviz('prior_rel_' + folder, tags, REL, scale=scale)

    r"""
    ------
    Method
    ------
    """

    # visulization of method
    envs = [('emu_n0_up'    ,  0, 'log', r'Testbed Emulation (Upper Triangular)', 'N/A           '),
            ('emu_n0_mmmk'  ,  0, 'log', r'Testbed Emulation (M/M/1/$K$)       ', 'N/A           '), 
            ('mm1k_n0_mmmk' , -1, 'log', r'M/M/1/$K$ (fast-mix)                ', '[0.022, 0.043]'), # // [0.007, 0.050]
            ('mm1ks_n0_mmmk', -1, 'log', r'M/M/1/$K$ (slow-mix)                ', '[0.005, 0.008]'),
            ('mmmmr_n0_mmmk', -1, 'log', r'M/M/$m$/$m+r$                       ', '[0.013, 0.024]'), # // [0.003, 0.022]
            ('mmul_n0_mmul' , -2, 'log', r'M/M/Multiple/$K$                    ', '[0.068, 0.096]')] # // [0.010, 0.016]
    # // envs = [('emu_n0_up'         ,  0, 'log', r'Testbed Emulation                                                    '),
    # //         ('mm1k-small_n0_mmmk', -1, 'log', r'M/M/1/$K$ (slow-mixing, $\delta_n \in \redit{[0.005, 0.008]}$)       '),
    # //         ('mm1k-large_n0_mmmk', -1, 'log', r'M/M/1/$K$ (fast-mixing, $\delta_n \in \redit{[0.005, 0.032]}$)       '),
    # //         ('mmmmr_n0_mmmk'     , -1, 'log', r'M/M/$m$/$m + r$ (fast-mixing, $\delta_n \in \redit{[0.010, 0.027]}$) '),
    # //         ('mmul-large_n0_mmul', -1, 'log', r'M/M/Multiple/$K$ (fast-mixing, $\delta_n \in \redit{[0.012, 0.057]}$)')]
    # //         # // ('mmul-small_n0_mmul', -1, 'log', r'M/M/Multiple/$K$ (slow-mixing, $\delta_n \in \redit{[0.018, 0.031]}$)'),
    legs = [('dc4'  , 'DC-4' , r'DC-BPTT $t^*=16$' ),
            ('dc7'  , 'DC-7' , r'DC-BPTT $t^*=128$'),
            ('rrinf', 'RRInf', r'$\infty$-SGD'     )]
    table = [["logs/{row}/log_{row}_{col}_g-1_a2" + "_l{}".format(lr) for _ in legs] for _, lr, _, _, _ in envs]
    make_table_dual(None, envs, legs, table)
    exit()
    for folder, lr, scale, _, _ in envs:
        tags = {}
        for itr, label, mathl in legs:
            tags[mathl] = os.path.join('logs', folder, "log_{}_{}_g-1_a2_l{}".format(folder, itr, lr))
        lineviz('config_abs_' + folder, tags, ABS, scale=scale)
        lineviz('config_rel_' + folder, tags, REL, scale=scale)

    r"""
    -----
    Alpha
    -----
    """

    # visulization of alpha
    envs = [('mm1k_n0_mmmk' , -1, 'log', r'M/M/1/$K$ (fast-mix)', '[0.022, 0.043]'), # // [0.007, 0.050]
            ('mm1ks_n0_mmmk', -1, 'log', r'M/M/1/$K$ (slow-mix)', '[0.005, 0.008]'),
            ('mmmmr_n0_mmmk', -1, 'log', r'M/M/$m$/$m+r$       ', '[0.013, 0.024]'), # // [0.003, 0.022]
            ('mmul_n0_mmul' , -2, 'log', r'M/M/Multiple/$K$    ', '[0.068, 0.096]')] # // [0.010, 0.016]
    # // envs = [('emu_n0_up'         ,  0, 'log', r'Testbed Emulation                                                    '),
    # //         ('mm1k-small_n0_mmmk', -1, 'log', r'M/M/1/$K$ (slow-mixing, $\delta_n \in \redit{[0.005, 0.008]}$)       '),
    # //         ('mm1k-large_n0_mmmk', -1, 'log', r'M/M/1/$K$ (fast-mixing, $\delta_n \in \redit{[0.005, 0.032]}$)       '),
    # //         ('mmmmr_n0_mmmk'     , -1, 'log', r'M/M/$m$/$m + r$ (fast-mixing, $\delta_n \in \redit{[0.010, 0.027]}$) '),
    # //         ('mmul-large_n0_mmul', -1, 'log', r'M/M/Multiple/$K$ (fast-mixing, $\delta_n \in \redit{[0.012, 0.057]}$)')]
    # //         # // ('mmul-small_n0_mmul', -1, 'log', r'M/M/Multiple/$K$ (slow-mixing, $\delta_n \in \redit{[0.018, 0.031]}$)'),
    legs = [('2' , '100', r'100'),
            ('0' , '1'  , r'1'  ),
            ('-1', '0.1', r'0.1')]
    table = [["logs/{row}/log_{row}_rrinf_g-1_a{col}" + "_l{}".format(lr) for _ in legs] for _, lr, _, _, _ in envs]
    make_table_dual(None, envs, legs, table, item='train_loss')
    for folder, lr, scale, _, _ in envs:
        tags = {}
        for itr, label, _ in legs:
            tags[label] = os.path.join('logs', folder, "log_{}_rrinf_g-1_a{}_l{}".format(folder, itr, lr))
        lineviz('alpha_abs_' + folder, tags, ABS, scale=scale)
        lineviz('alpha_rel_' + folder, tags, REL, scale=scale)

    r"""
    ----
    GeoP
    ----
    """

    # visulization of geop
    envs = [('mm1k_n0_mmmk' , -1, 'log', r'M/M/1/$K$ (fast-mix)', '[0.022, 0.043]'), # // [0.007, 0.050]
            ('mm1ks_n0_mmmk', -1, 'log', r'M/M/1/$K$ (slow-mix)', '[0.005, 0.008]'),
            ('mmmmr_n0_mmmk', -1, 'log', r'M/M/$m$/$m+r$       ', '[0.013, 0.024]'), # // [0.003, 0.022]
            ('mmul_n0_mmul' , -2, 'log', r'M/M/Multiple/$K$    ', '[0.068, 0.096]')] # // [0.010, 0.016]
    # // envs = [('emu_n0_up'         ,  0, 'log', r'Testbed Emulation                                                    '),
    # //         ('mm1k-small_n0_mmmk', -1, 'log', r'M/M/1/$K$ (slow-mixing, $\delta_n \in \redit{[0.005, 0.008]}$)       '),
    # //         ('mm1k-large_n0_mmmk', -1, 'log', r'M/M/1/$K$ (fast-mixing, $\delta_n \in \redit{[0.005, 0.032]}$)       '),
    # //         ('mmmmr_n0_mmmk'     , -1, 'log', r'M/M/$m$/$m + r$ (fast-mixing, $\delta_n \in \redit{[0.010, 0.027]}$) '),
    # //         ('mmul-large_n0_mmul', -1, 'log', r'M/M/Multiple/$K$ (fast-mixing, $\delta_n \in \redit{[0.012, 0.057]}$)')]
    # //         # // ('mmul-small_n0_mmul', -1, 'log', r'M/M/Multiple/$K$ (slow-mixing, $\delta_n \in \redit{[0.018, 0.031]}$)'),
    legs = [('-1' , '0.1' , r'0.1' ),
            ('-2' , '0.01', r'0.01')]
    table = [["logs/{row}/log_{row}_rrinf_g{col}_a2" + "_l{}".format(lr) for _ in legs] for _, lr, _, _, _ in envs]
    make_table_dual(None, envs, legs, table, item='train_loss')
    for folder, lr, scale, _, _ in envs:
        tags = {}
        if 'mm1ks' in folder:
            tags['0.001'] = os.path.join('logs', folder, "log_{}_rrinf_g{}_a2_l{}".format(folder, -3, lr))
        else:
            pass 
        for itr, label, _ in legs:
            tags[label] = os.path.join('logs', folder, "log_{}_rrinf_g{}_a2_l{}".format(folder, itr, lr))
        lineviz('geop_abs_' + folder, tags, ABS, scale=scale)
        lineviz('geop_rel_' + folder, tags, REL, scale=scale)