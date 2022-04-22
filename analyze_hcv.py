# file for analyzing stuff based on the filepath
import os

import numpy as np

from dylightful.bar_plot import make_barplot
from dylightful.discretizer import (
    find_states_kmeans,
    smooth_projection_k_means,
    tae_discretizer,
)
from dylightful.mdanalysis import write_dcd
from dylightful.msm import fit_msm, model_msm
from dylightful.parser import (
    get_time_series,
    load_env_partners,
    load_env_partners_mixed,
)
from dylightful.postprocess import sort_markov_matrix
from dylightful.utilities import get_dir, load_parsed_dyno

# define relative paths properly
dirname = os.path.dirname(__file__)


def analyze_protein_perspective(
    traj_path,
    discretizer,
    num_states,
    smoothing=smooth_projection_k_means,
    clustering=find_states_kmeans,
    repitions=5,
):

    name = "protein"
    traj_path = os.path.join(dirname, traj_path)
    env_partners = load_env_partners(json_path=traj_path)
    env_partner_arr = []
    for partner in env_partners.keys():
        arr = np.array(env_partners[partner])
        traj = np.zeros(len(arr.T))
        for traject in env_partners[partner]:
            traj += np.array(traject).T
        env_partner_arr.append(traj)

    env_partner_arr = np.array(env_partner_arr)
    time_ser = env_partner_arr.T
    save_path = get_dir(traj_path)
    make_barplot(
        time_ser=time_ser,
        ylabel="Protein Residue",
        yticks=list(env_partners.keys()),
        prefix=name,
        save_path=save_path,
    )
    count_matrix_array = []
    for i in range(repitions):
        prefix = name + "_" + str(i)
        proj = discretizer(
            time_ser=time_ser,
            clustering=clustering,
            save_path=save_path,
            prefix=prefix,
            num_states=num_states,
        )
        labels = smoothing(proj, num_states)
        print("num unique protein", len(np.unique(proj)))
        msm, count_matrix = fit_msm(
            trajectory=labels,
            save_path=save_path,
            prefix=prefix,
        )
        print_markov(msm=msm, count_matrix=count_matrix)
        count_matrix_array.append(np.sort(np.diag(count_matrix)))
    print("Standard deviation: ", np.std(count_matrix_array))
    return labels, msm, count_matrix


def analyze_mixed_perspective(
    traj_path,
    discretizer,
    num_states,
    smoothing=smooth_projection_k_means,
    clustering=find_states_kmeans,
    repitions=5,
):

    name = "mixed"
    traj_path = os.path.join(dirname, traj_path)
    env_partners = load_env_partners_mixed(json_path=traj_path)
    env_partner_arr = []
    for partner in env_partners.keys():
        env_partner_arr.append(env_partners[partner])

    env_partner_arr = np.array(env_partner_arr)
    time_ser = env_partner_arr.T
    save_path = get_dir(traj_path)
    make_barplot(
        time_ser=time_ser,
        ylabel="Interaction",
        yticks=list(env_partners.keys()),
        prefix=name,
        save_path=save_path,
    )
    count_matrix_array = []
    for i in range(repitions):
        prefix = name + "_" + str(i)
        proj = discretizer(
            time_ser=time_ser,
            clustering=clustering,
            save_path=save_path,
            prefix=prefix,
            num_states=num_states,
        )
        labels = smoothing(proj, num_states)
        print("mixed distinct", len(np.unique(proj)))
        msm, count_matrix = fit_msm(
            trajectory=labels, save_path=save_path, prefix=prefix
        )
        print_markov(msm=msm, count_matrix=count_matrix)
        count_matrix_array.append(np.sort(np.diag(count_matrix)))
    print("Standard deviation: ", np.std(count_matrix_array))
    return labels, msm, count_matrix


def analyze_ligand_perspective(
    traj_path,
    discretizer,
    num_states,
    smoothing=smooth_projection_k_means,
    clustering=smooth_projection_k_means,
    repitions=5,
):

    name = "ligand"
    traj_path = os.path.join(dirname, traj_path)
    time_ser, num_obs = load_parsed_dyno(traj_path=traj_path)
    save_path = get_dir(traj_path)
    make_barplot(
        time_ser=time_ser,
        ylabel="Ligand Perspective",
        yticks=np.arange(time_ser.shape[1]),
        prefix=name,
        save_path=save_path,
    )
    count_matrix_array = []
    for i in range(repitions):
        prefix = name + "_" + str(i)
        proj = discretizer(
            time_ser=time_ser,
            clustering=clustering,
            save_path=save_path,
            prefix=prefix,
            num_states=num_states,
        )
        print("ligand distinct", len(np.unique(proj)))
        labels = smoothing(proj, num_states)
        msm, count_matrix = fit_msm(
            trajectory=labels, save_path=save_path, prefix=prefix
        )
        print_markov(msm=msm, count_matrix=count_matrix)
        count_matrix_array.append(np.sort(np.diag(count_matrix)))
    print("Standard deviation: ", np.std(count_matrix_array))

    return labels, msm, count_matrix


def find_states_mixed(
    traj_path,
    discretizer,
    num_states=30,
    smoothing=smooth_projection_k_means,
    clustering=find_states_kmeans,
    repitions=5,
):

    name = "mixed"
    traj_path = os.path.join(dirname, traj_path)
    env_partners = load_env_partners_mixed(json_path=traj_path)
    env_partner_arr = []
    for partner in env_partners.keys():
        env_partner_arr.append(env_partners[partner])

    env_partner_arr = np.array(env_partner_arr)
    time_ser = env_partner_arr.T
    save_path = get_dir(traj_path)
    stds, count_matrices = find_states(
        trajectory=time_ser, max_states=num_states, repitions=6, save_path=save_path
    )
    return stds, count_matrices


def find_states(
    trajectory,
    max_states=30,
    repitions=6,
    save_path=None,
    smoothing=smooth_projection_k_means,
    plotting=False,
):
    """Function to find the most suitable number of states for a given trajectory based on Markovian analysis.

    Args:
        trajectory (_type_): _description_
        max_states (int, optional): _description_. Defaults to 30.
        repitions (int, optional): _description_. Defaults to 6.
        save_path (_type_, optional): _description_. Defaults to None.
        smoothing (_type_, optional): _description_. Defaults to smooth_projection_k_means.
    """
    standard_deviations = []
    experimental = []
    for i in range(2, max_states + 1):
        count_matrix_array = []
        for j in range(repitions):
            prefix = name + "_" + str(j)
            proj = discretizer(
                time_ser=trajectory,
                clustering=None,
                save_path=save_path,
                prefix=prefix,
                num_states=max_states,
                plotting=plotting,
            )
            labels = smoothing(proj, num_cluster=i)
            estimator, count_matrix, counts = model_msm(labels)
            count_matrix = sort_markov_matrix(count_matrix)
            count_matrix_array.append(np.sort(np.diag(count_matrix)))
        print("Number of States ", i)
        std = np.std(count_matrix_array)
        print("Standard Deviation of sorted diagnonal of the count matrix: ", std)
        standard_deviations.append(std)
        print("Experimental")
        mean = np.mean(count_matrix_array, axis=0)
        std_arr = np.std(count_matrix_array, axis=0)
        exp = np.mean(np.divide(std_arr, mean))
        experimental.append(exp)
        print(exp)
    print("Standard Deviations from 2 for up to ", max_states, " states: ")
    print(standard_deviations)
    print(
        "Experimental measure of deviation: ",
    )
    print(experimental)
    return standard_deviations, count_matrix_array


def print_markov(msm, count_matrix):
    """prints the results of the Markovian analysis for having
    output with the raw data

    Args:
        msm (object): Markov state matrix object from deeptime
        count_matrix (np.ndarray): Absolute count matrix
    Returns:
        _type_: _description_
    """

    print("MSM")
    print(msm.transition_matrix)
    print("Diag")
    print(np.sort(np.diag(msm.transition_matrix)))
    print("Counts")
    print(count_matrix)
    print("Diag")
    print(np.sort(np.diag(count_matrix)))
    print("Row sums")
    print(np.sort(np.sum(count_matrix, axis=1)))
    return None


if __name__ == "__main__":

    dirs = ["3SU3_longer/", "3SU4_longer/", "3SU5_longer/", "3SU6_longer/"]
    names = ["3SU3_dynophore", "3SU4_dynophore", "3SU5_dynophore", "3SU6_dynophore"]
    states = [2, 2, 2, 2]
    resnames = ["SU3"] * len(dirs)
    for i in range(2, 3):
        print("*************" + names[i] + "*************")
        save_dir = "/home/julian/Documents/Master/HCV/" + dirs[i]
        name = names[i]
        traj_path = save_dir + name + ".json"

        pml = save_dir + name + ".pml"
        ligand_path = save_dir + name + "_time_series.json"
        topology = save_dir + "topo0.pdb"
        trajectory = save_dir + "trajectory.dcd"
        selection_string = "protein or resname " + resnames[i]
        base = save_dir
        get_time_series(pml)
        discretizer = tae_discretizer
        clustering = find_states_kmeans
        smoothing = smooth_projection_k_means
        num_states = states[i]
        # standard_deviations, count_matrix_array = find_states_mixed(
        #        traj_path=traj_path,
        #        discretizer=tae_discretizer,
        #        num_states=20,
        #        clustering=clustering,
        #        smoothing=smoothing,
        #    )
        # num_states = np.argmin(standard_deviations)
        ("Running Markov analysis with n=", num_states, "states")

        # num_states = 2
        # labels, msm, count_matrix = analyze_ligand_perspective(
        #    traj_path=ligand_path,
        #    discretizer=discretizer,
        #    num_states=num_states,
        #    clustering=clustering,
        #    smoothing=smooth_projection_gaussian,
        # )
        # print("labels")
        # try:
        #    write_dcd(
        #    labels=labels,
        #    topology=topology,
        #    coordinates=trajectory,
        #    base=base,
        #    prefix="ligand",
        # )
        # except:
        #    print("No trajectory provided")
        labels, msm, count_matrix = analyze_mixed_perspective(
            traj_path=traj_path,
            discretizer=tae_discretizer,
            num_states=num_states,
            clustering=clustering,
            smoothing=smoothing,
        )
        try:
            write_dcd(
                labels=labels,
                topology=topology,
                coordinates=trajectory,
                base=base,
                prefix=dirs[i][:-1] + "_" + str(num_states) + "_mixed",
            )
        except:
            print("No trajectory provided")
            # labels, msm, count_matrix = analyze_protein_perspective(
    #    traj_path=traj_path,
    #    discretizer=tae_discretizer,
    #    num_states=num_states,
    #    clustering=clustering,
    #    smoothing=smoothing,
    # )
    # try:
    #    write_dcd(
    #    labels=labels,
    #    topology=topology,
    #    coordinates=trajectory,
    #    base=base,
    #    prefix="protein",
    # )
    # except:
    #    print("No trajectory provided")
    #
