import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def eval(x: list,
         gt: list,
         preds: list,
         name: str,
         w: int,
         preds_marker='^-',
         gt_marker='o',
         save_fig=False,
         save_preds=False):
    assert len(x) == len(gt) == len(preds)
    _, ax = plt.subplots()
    ax.plot(x, preds, preds_marker, label='preds')
    ax.scatter(x, gt, c='black', marker=gt_marker, label='gt')
    ax.legend()
    plt.title("W"+str(w)+" "+name)
    if save_fig:
        plt.savefig(f'result/W{str(w)} {name}.png', format='png')

    else:
        plt.show()