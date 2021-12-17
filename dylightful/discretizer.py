from pathlib import Path
import torch
from torch.utils.data import DataLoader

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from tqdm import tqdm


from sklearn.cluster import KMeans

from deeptime.util.torch import MLP
from deeptime.util.data import TrajectoryDataset
from deeptime.decomposition.deep import TAE

from dylightful.utilities import make_name, parse_file_path


def tae_discretizer(
    time_ser, size=3, prefix=None, save_path=None, num_cluster=15, tol=0.01
):
    """Test MSM with time lagged autoencoders according to Noé et al.

    Args:
        time_ser ([type]): Dynophore trajectory converted by the parser
        size (int, optional): 10*size is the autoencoder size Defaults to 3.
        file_name (str, optional): Name the output file. Defaults to None.
        save_path (str, optional): Path to output folder. Defaults to None.
        num_cluster (int, optional): Maximal number of MSM states to fit the analysis to. Defaults to 15.
        tol (float, optional): Tolerrance when to stop the clustering to find optimal states. Defaults to 0.01.
    """
    save_path = parse_file_path(save_path)
    num_superfeatures = len(time_ser[0])
    # set_up tae
    dataset = TrajectoryDataset(1, time_ser.astype(np.float32))

    n_val = int(len(dataset) * 0.5)
    train_data, val_data = torch.utils.data.random_split(
        dataset, [len(dataset) - n_val, n_val]
    )
    loader_train = DataLoader(train_data, batch_size=64, shuffle=False)
    loader_val = DataLoader(val_data, batch_size=len(val_data), shuffle=False)
    units = [num_superfeatures] + [
        size * num_superfeatures + size * num_superfeatures,
        1,
    ]
    encoder = MLP(
        units,
        nonlinearity=torch.nn.ReLU,
        output_nonlinearity=torch.nn.Sigmoid,
        initial_batchnorm=False,
    )
    decoder = MLP(units[::-1], nonlinearity=torch.nn.ReLU, initial_batchnorm=False)
    tae = TAE(encoder, decoder, learning_rate=1e-3)
    tae.fit(loader_train, n_epochs=50, validation_loader=loader_val)
    tae_model = tae.fetch_model()
    proj = tae_model.transform(time_ser)
    try:
        plot_tae_training(tae_model=tae, prefix=prefix, save_path=save_path)
    except:
        print("Did not plot the TAE training.")
    try:
        plot_tae_transform(proj=proj, prefix=prefix, save_path=save_path)
    except:
        print("Did not plot the TAE transform.")
    try:
        find_states_kmeans(
            proj=proj,
            prefix=prefix,
            save_path=save_path,
            num_cluster=num_cluster,
            tol=tol,
        )
    except:
        print("Did not plot ellbow")
    return proj


def smooth_projection_k_means(arr, num_cluster):
    """Clusters an array with k_means according to num_cluster

    Args:
        proj ([type]): [description]
        num_cluster ([type]): [description]

    Returns:
        [type]: [description]
    """
    clf = KMeans(n_clusters=num_cluster).fit(arr)
    return clf.labels_


def find_states_kmeans(proj, prefix, save_path, num_cluster=15, tol=0.01):
    """Cluster the projection to get realy discretized values necessary for the MSM

    Args:
        proj ([type]): [description]
        num_cluster (int, optional): [description]. Defaults to 15.
        tol (float, optional): [description]. Defaults to 0.01.
    """

    scores = np.zeros(num_cluster)
    sum_of_squared_distances = np.zeros(num_cluster)
    for i in range(1, num_cluster):
        clf = KMeans(n_clusters=i).fit(proj)
        scores[i] = clf.score(proj)
        sum_of_squared_distances[i] = clf.inertia_
    # clf = KMeans(n_clusters=3, random_state=random_state).fit(proj)
    # labels = clf.labels_
    plot_ellbow_kmeans(
        metric=sum_of_squared_distances, prefix=prefix, save_path=save_path
    )
    plot_scores_kmeans(metric=scores, prefix=prefix, save_path=save_path)
    return [scores, sum_of_squared_distances]


def plot_tae_training(tae_model, prefix=None, save_path=None):
    """Plots the loss function for the trainig of the TAE model.

    Args:
        tae_model ([type]): [description]
        file_name ([type], optional): [description]. Defaults to None.
        save_path ([type], optional): [description]. Defaults to None.
    """
    plt.clf()
    plt.cla()
    name = "_tae_training.png"
    file_name = make_name(prefix=prefix, name=name, dir=save_path)
    plt.semilogy(*tae_model.train_losses.T, label="train")
    plt.semilogy(*tae_model.validation_losses.T, label="validation")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(file_name, dpi=300)

    return None


def plot_tae_transform(proj, num_steps=1000, prefix=None, save_path=None):
    """Plots the transformation obtained by the TAE model.

    Args:
        proj ([type]): [description]
        num_steps (int, optional): [description]. Defaults to 1000.
        file_name ([type], optional): [description]. Defaults to None.
        save_path ([type], optional): [description]. Defaults to None.
    """
    plt.clf()
    plt.cla()
    name = "_tae_transform.png"
    file_name = make_name(prefix=prefix, name=name, dir=save_path)
    plt.ylabel("State")
    plt.xlabel("Timte $t$")
    if num_steps < len(proj):
        plt.plot(proj[:num_steps])
    else:
        plt.plot(proj[:num_steps])
    plt.legend()
    plt.savefig(file_name, dpi=300)


def plot_scores_kmeans(metric, prefix=None, save_path=None):
    """Plots the scores of the k_means finder

    Args:
        sum_of_squared_distances ([type]): [description]
        file_name ([type], optional): [description]. Defaults to None.
        save_path ([type], optional): [description]. Defaults to None.

    Returns:
        None
    """
    plt.clf()
    plt.cla()
    name = "_scores_kmeans.png"
    file_name = make_name(prefix=prefix, name=name, dir=save_path)
    plt.xlabel("Number of cluster")
    plt.ylabel("Euclidean Norm $l^2$")
    plt.plot(metric[1:])
    plt.savefig(file_name, dpi=300)
    print("Saved", file_name)
    return None


def plot_ellbow_kmeans(metric, prefix=None, save_path=None):
    """Plots the sum of squared distances for K-Means to do the ellbow method visually


    Args:
        sum_of_squared_distances ([type]): [description]
        file_name ([type], optional): [description]. Defaults to None.
        save_path ([type], optional): [description]. Defaults to None.

    Returns:
        None
    """
    plt.clf()
    plt.cla()
    name = "_ellbow_kMeans.png"
    file_name = make_name(prefix=prefix, name=name, dir=save_path)
    plt.xlabel("Number of cluster")
    plt.ylabel("Sum of squared distances $R$")
    plt.plot(metric[1:])
    plt.savefig(file_name, dpi=300)
    print("Saved", file_name)
    return None
