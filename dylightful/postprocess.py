import json

import numpy as np
import pandas as pd
from numpy.lib.arraysetops import unique

from dylightful.parser import load_env_partners
from dylightful.utilities import make_name, parse_file_path


def sort_markov_matrix(markov_matrix):
    """Takes in random markov matrix
    returns sorted markov matrix

    Args:
        markov_matrix (np.array): unsorted matrix

    Returns:
        (np.array): sorted Markov matrix
    """
    diag = np.diag(markov_matrix)
    sorting = np.argsort(diag)
    for i in range(len(diag)):
        for j in range(len(diag) - 1):
            if diag[j + 1] > diag[j]:
                markov_matrix[[j, j + 1]] = markov_matrix[[j + 1, j]]
                markov_matrix[:, [j, j + 1]] = markov_matrix[:, [j + 1, j]]
    return markov_matrix


def postprocessing_msm(labels_states, dynophore_json, processed_dyn, save_path):

    """Postprocessing of the msm for validation purposes of the Markov model."""

    env_partners = load_env_partners(dynophore_json)
    time_ser_superf = load_time_ser_superfeat(processed_dyn)  # check notebook for it
    state_data = {}
    for i in np.unique(labels_states):
        state_data[str(i)] = {}
    state_data = generate_state_map(
        time_ser_superf, labels_states=labels_states, state_data=state_data
    )
    state_data = get_information_mstates(
        labels_states=labels_states, state_data=state_data
    )
    for state in state_data.keys():
        state_data[state]["env_partners_list"] = get_env_partners(
            state_data[state]["frameIndices_state"], env_partners
        )
        unique, counts = get_unique_env_partner(
            partner_traj=state_data[state]["env_partners_list"]
        )
        state_data[state]["unique_env_partners"] = unique
        state_data[state]["unique_env_partners_counts"] = counts
    save_path = parse_file_path(save_path)
    name = "markophore_validation.json"
    prefix = None
    file_name = make_name(prefix=prefix, name=name, dir=save_path)

    with open(file_name, "w") as fp:
        json.dump(state_data, fp)
    return state_data


def load_time_ser_superfeat(path_to_processed_dynp):

    with open(path_to_processed_dynp) as f:
        data = json.load(f)

    time_ser = pd.DataFrame(data)
    time_ser = time_ser.drop(columns="num_frames")
    obs = time_ser.drop_duplicates()
    num_obs = len(obs)
    obs = obs.to_numpy()
    time_ser = time_ser.to_numpy()
    return time_ser


##
def generate_state_map(time_ser_superf, labels_states, state_data):
    """Generates a map foar each time point of a Markov Sate to a Superfeature tuple such that the time series is the
    (0,1,2), (0,1,3), (0,1,2) etc...

    Args:
        time_ser_superf ([type]): loaded combined array of superfeature occurences such that shape is ( len_md_traj, num_superfeatures,)
        labels_states ([type]): time series of the labels of the different Markov states

    Returns:
        [np.ndarray]: array of
    """
    for i in state_data.keys():
        state_data[str(i)]["pharmacoph_traj"] = []
    for i in range(len(time_ser_superf)):
        state = str(labels_states[i])
        state_map = state_data[state]["pharmacoph_traj"]
        state_map.append(
            np.where(time_ser_superf[i, :] == 1)[0].astype(np.int16).tolist()
        )
        state_data[state]["pharmacoph_traj"] = state_map
    return state_data


##
def get_information_mstates(labels_states, state_data):
    """caclulates some general information such as occurences of the Markov states and unique pharmacophores"""

    for state in state_data.keys():
        ## include information about the markov_state
        occ = np.where(labels_states == int(state))
        num_occ = len(occ)
        # state_data = {} check the new datastructure
        state_data[state]["frameIndices_state"] = occ[0].tolist()
        state_data[state]["num"] = num_occ
        state_data[state]["unique_pharmc"] = get_information_pc(
            np.array(state_data[state]["pharmacoph_traj"])
        )
        state_data[state]["distinctSuperfeatures"] = get_distinct_superfeatures(
            state_data[state]["unique_pharmc"]
        )
    return state_data


def get_distinct_superfeatures(unique_pharmc):
    """[summary]

    Args:
        unique_pharmc ([type]): [description]

    Returns:
        [type]: [description]
    """

    superfeats = set()
    for pharmc in unique_pharmc:
        superfeats = superfeats.union(set(pharmc))
    return list(superfeats)


def get_information_pc(pharmacophore_traj):
    """Calculates information about the pharcophore/superfeature patterns

    Args:
        state_data ([type]): [description]

    Returns:
        [(list, float)]:
    """

    # TODO can include counts
    unique_pharmc = np.unique(pharmacophore_traj).tolist()
    return unique_pharmc


def get_env_partners(frame_indices, env_partners):
    """

    Args:
        frameIndices_state ([type]): [description]
        env_partners ([type]): [description]

    Returns:
        [type]: [description]
    """

    env_partner_arr = []
    residues = list(env_partners.keys())
    for partner in env_partners.keys():
        env_partner_arr.append(np.array(env_partners[partner]))

    env_partner_arr = np.array(env_partner_arr)
    partners = []

    for i in range(len(frame_indices)):
        # assert env_partners == 1, str(env_partner_arr.shape)+str(env_partner_arr)+str(len(env_partner_arr[0]))+str(env_partner_arr[0].shape)
        res = []
        for j in range(len(env_partner_arr)):
            occ = env_partner_arr[j][0, i]
            if occ == 1:
                res.append(np.array(residues)[j])

        partners.append(res)
    return partners


def get_unique_env_partner(partner_traj):
    """Counts env_partners from the env_partner trajectory

    Args:
        partner_traj ([type]): specific to a state returns the trajectory of the env partners

    Returns:
        [(unique_partners, their_counts)]:
    """

    count = []
    for partner in np.array(partner_traj).flatten():
        count += partner
    unique, counts = np.unique(count, return_counts=True)
    return unique.tolist(), counts.tolist()


def load_validation():
    """[summary]

    Returns:
        [type]: [description]
    """

    f = open("../tests/Trajectories/ZIKV/markophore_validation.json")
    data = json.load(f)
    return data


if __name__ == "__main__":
    a = np.array(
        [
            [0.8, 0.1, 0.05, 0.05],
            [0.005, 0.9, 0.03, 0.015],
            [0.1, 0.2, 0.4, 0.3],
            [0.01, 0.02, 0.03, 0.94],
        ]
    )
    b = sort_markov_matrix(a)
