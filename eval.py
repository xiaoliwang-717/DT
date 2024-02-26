from matplotlib import pyplot as plt


def eval(x: list, 
         gt: list, 
         preds: list, 
         preds_marker='^-', 
         gt_marker='o', 
         save_fig=False):
    assert len(x) == len(gt) == len(preds)
    _, ax = plt.subplots()
    ax.plot(x, preds, preds_marker, label='preds')
    ax.scatter(x, gt, c='black', marker=gt_marker, label='gt')
    ax.legend()
    if save_fig:
        plt.savefig('result.png', format='png')
    else:
        plt.show()
        
        
if __name__ == '__main__':
    x = [0, 10, 25, 50, 75]
    gt = [1, 2, 3, 4, 5]
    preds = [1.2, 2.3, 3.4, 4.5, 5.6]
    eval(x, gt, preds)