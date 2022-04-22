import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from dylightful.utilities import make_name


def make_barplot(time_ser, ylabel, yticks, prefix=None, save_path=None):
    """Craeates and saves bar plot


    Args:
        time_ser ([type]): by paser.get_timeseries processed pml time series of features
        path ([type], optional): Where to save. Defaults to None.
        prefix ([type], optional): Name of the file
    """
    # save_path = get_dir(save_path)
    plt.clf()
    plt.cla()
    plt.tight_layout()
    mpl.rcParams["font.size"] = "20"
    plt.rc("axes", labelsize=20)
    name = "_barplot.png"
    file_name = make_name(prefix=prefix, name=name, dir=save_path)
    code = time_ser
    dpi = 300

    fig = plt.figure(figsize=(20, 13), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])  # span the whole figure
    ax.imshow(code.T, cmap="binary", aspect="auto", interpolation="nearest")
    ax.set_xlabel("Timestep")
    ax.set_ylabel(ylabel)
    ax.set_yticks(np.arange(len(yticks)))
    ax.set_yticklabels(yticks)
    fig.savefig(file_name, dpi=dpi, bbox_inches="tight")
