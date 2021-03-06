# test postprocessing of MSM for validation purposes and additional viz.

import pytest
import numpy as np
import os


from dylightful.discretizer import tae_discretizer, smooth_projection_k_means
from dylightful.utilities import get_dir, load_parsed_dyno
from dylightful.msm import fit_msm
from dylightful.postprocess import postprocessing_msm, sort_markov_matrix

dirname = os.path.dirname(__file__)


@pytest.mark.parametrize(
    "traj_path, dyn_path, discretizer, num_states",
    [
        (
            "Trajectories/ZIKV/ZIKV-Pro-427-1_dynophore_time_series.json",
            "Trajectories/ZIKV/ZIKV-Pro-427-1_dynophore.json",
            tae_discretizer,
            4,
        ),
    ],
)
def test_postprocessing(traj_path, dyn_path, discretizer, num_states):
    traj_path = os.path.join(dirname, traj_path)
    dyn_path = os.path.join(dirname, dyn_path)
    time_ser, num_obs = load_parsed_dyno(traj_path=traj_path)
    save_path = get_dir(traj_path)
    proj = discretizer(time_ser=time_ser, num_states=num_states, save_path=save_path)
    labels = smooth_projection_k_means(proj, num_states)
    fit_msm(trajectory=labels, save_path=save_path)
    postprocessing_msm(
        labels_states=labels,
        dynophore_json=dyn_path,
        processed_dyn=traj_path,
        save_path=save_path,
    )


def test_sort():
    a = np.array(
        [
            [0.8, 0.1, 0.05, 0.05],
            [0.005, 0.9, 0.03, 0.015],
            [0.1, 0.2, 0.4, 0.3],
            [0.01, 0.02, 0.03, 0.94],
        ]
    )
    sorted_a = sort_markov_matrix(a)
    assert (
        np.array_equal(sorted_a[0, :], np.array([0.94, 0.02, 0.01, 0.03])) == True
    ), str(sorted_a[0, :])
    assert (
        np.array_equal(sorted_a[1, :], np.array([0.015, 0.9, 0.005, 0.03])) == True
    ), str(sorted_a[1, :])
    assert (
        np.array_equal(sorted_a[2, :], np.array([0.05, 0.1, 0.8, 0.05])) == True
    ), str(sorted_a[2, :])
    assert np.array_equal(sorted_a[3, :], np.array([0.3, 0.2, 0.1, 0.4])) == True, str(
        sorted_a[3, :]
    )
