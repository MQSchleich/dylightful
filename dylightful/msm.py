from sys import prefix
import matplotlib.pyplot as plt
from deeptime.markov import TransitionCountEstimator
import deeptime.markov as markov
import seaborn as sns
import numpy as np

from dylightful.utilities import make_name, get_dir
from dylightful.discretizer import tae_discretizer, smooth_projection_k_means


def build_tae_msm(traj_path, time_ser, num_states, prefix=None):
    """does the tae analysis of a dynophore trajectory

    Args:
        traj_path (string): path to trajectory to be discretized
        num_states (int): assumed states
    """
    save_path = get_dir(traj_path)
    proj = tae_discretizer(
        time_ser=time_ser, num_states=num_states, save_path=save_path
    )
    labels = smooth_projection_k_means(proj, num_states)

    msm = fit_msm(trajectory=labels, save_path=save_path, prefix=prefix)
    return msm, labels, proj, time_ser


def fit_msm(trajectory, prefix=None, save_path=None):
    """Function to fit the msm to a given trajectory and save the vizualization

    Args:
        trajectory ([type]): Time series to be discretized
        prefix ([type], optional):  Name to save a file.
        save_path ([type], optional):  Wheree to save a file.

    Returns:
        [type]: msm
    """

    plot_clustered_traj(trajectory=trajectory, prefix=prefix, save_path=save_path)

    plt.cla()
    plt.clf()

    estimator = TransitionCountEstimator(lagtime=1, count_mode="sliding")
    counts = estimator.fit(trajectory).fetch_model()
    count_matrix = counts.count_matrix  # fit and fetch the model
    estimator = markov.msm.MaximumLikelihoodMSM(
        reversible=True, stationary_distribution_constraint=None
    )
    ax = sns.heatmap(count_matrix)
    fig = ax.get_figure()
    name = "_msm_count_matrix.png"
    file_name = make_name(prefix=prefix, name=name, dir=save_path)
    fig.savefig(file_name, dpi=300)
    plt.cla()
    plt.clf()
    name = "_msm_transistion_matrix.png"
    file_name = make_name(prefix=prefix, name=name, dir=save_path)
    msm = estimator.fit(counts).fetch_model()
    transition_matrix = msm.transition_matrix
    ax = sns.heatmap(transition_matrix)
    fig = ax.get_figure()
    plt.xlabel("State")
    plt.ylabel("State")
    plt.savefig(file_name, dpi=300)
    plt.cla()
    plt.clf()
    return msm, count_matrix


def plot_clustered_traj(trajectory, prefix=None, save_path=None):
    """plots the discretized trajectory of the Markov Model

    Args:
        labels ([type]): [description]
    """
    plt.cla()
    plt.clf()
    name = "_discretized_trajectory.png"
    file_name = make_name(prefix=prefix, name=name, dir=save_path)
    plt.scatter(
        np.arange(0, len(trajectory), 1), trajectory, marker="x", s=4, color="black"
    )
    plt.savefig(file_name, dpi=300)


def map_pharmacophore_states(labels, parsed_states):

    pass


if __name__ == "__main__":
    # traj = "../tests/Trajectories/ZIKV/ZIKV-Pro-427-1_dynophore_time_series.json"

    # parsed_states =
    # labels = build_tae_msm(traj_path, num_states)
    pass
