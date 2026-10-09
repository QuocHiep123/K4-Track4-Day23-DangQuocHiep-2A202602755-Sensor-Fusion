"""Measurement-to-track association via Mahalanobis gating and greedy matching.

Part F supplies the association stage shown in docs/HUONG_DAN_KY_THUAT.md §2.
Load ``kalman`` with ``load_workspace_module`` for innovation helpers and tracking parameters
for the chi-square gate.
"""

from __future__ import annotations

from typing import Any
from typing import Sequence

import numpy as np
from scipy.stats import chi2
from fusion_lab.workspace_support import get_tracking_params
from fusion_lab.workspace_loader import load_workspace_module

kalman = load_workspace_module("kalman")


def mahalanobis_distance(track: Any, meas: Any) -> float:
    """Return squared Mahalanobis distance between a track and a measurement.

    Args:
        track: Track with ``x``, ``P``.
        meas: Measurement with ``sensor``.

    Returns:
        Scalar squared Mahalanobis distance.
    """
    H = meas.sensor.get_H(track.x)
    gamma = kalman.innovation(track.x, meas)
    S = kalman.innovation_covariance(track.P, meas, H)
    d2 = gamma.T @ np.linalg.inv(S) @ gamma
    return float(np.asarray(d2).item())


def chi2_gate(mhd_sq: float, sensor: Any) -> bool:
    """Return True if squared Mahalanobis distance lies inside the chi-square gate.

    Args:
        mhd_sq: Squared Mahalanobis distance.
        sensor: Sensor with ``dim_meas``.

    Returns:
        True if inside gate.
    """
    params = get_tracking_params()
    limit = chi2.ppf(params.gating_threshold, sensor.dim_meas)
    return bool(mhd_sq <= limit)


def association_cost_matrix(
    track_list: Sequence[Any], meas_list: Sequence[Any]
) -> np.matrix:
    """Build gated costs, checking each sensor's visibility before projection.

    Args:
        track_list: Active tracks.
        meas_list: Measurements for this sensor pass.

    Returns:
        Cost matrix; ``np.inf`` for invisible tracks or rejected chi-square gates.
        Invisible pairs must never call the Mahalanobis/projection helpers.
    """
    n_tracks = len(track_list)
    n_meas = len(meas_list)
    costs = np.full((n_tracks, n_meas), np.inf, dtype=float)

    for i, track in enumerate(track_list):
        for j, meas in enumerate(meas_list):
            if meas.sensor.in_fov(track.x):
                d2 = mahalanobis_distance(track, meas)
                if chi2_gate(d2, meas.sensor):
                    costs[i, j] = d2

    return np.asmatrix(costs)


def pick_next_pair(
    association_matrix: np.matrix,
    unassigned_tracks: Sequence[Any],
    unassigned_meas: Sequence[Any],
) -> tuple[Any, Any, np.matrix, list[Any], list[Any]]:
    """Pick the minimum-cost track/measurement pair and shrink the association problem.

    Args:
        association_matrix: Current cost matrix.
        unassigned_tracks: Track objects still free.
        unassigned_meas: Measurement objects still free.

    Returns:
        Tuple (track, meas, new_matrix, remaining_tracks, remaining_meas).
        If no finite pair exists, return np.nan for track and meas and retain both lists.
    """
    matrix_arr = np.asarray(association_matrix, dtype=float)
    if matrix_arr.size == 0:
        return (np.nan, np.nan, association_matrix, list(unassigned_tracks), list(unassigned_meas))

    min_val = np.min(matrix_arr)
    if not np.isfinite(min_val):
        return (np.nan, np.nan, association_matrix, list(unassigned_tracks), list(unassigned_meas))

    r, c = np.unravel_index(np.argmin(matrix_arr), matrix_arr.shape)
    track = unassigned_tracks[r]
    meas = unassigned_meas[c]

    new_matrix = np.delete(np.delete(matrix_arr, r, axis=0), c, axis=1)
    new_matrix = np.asmatrix(new_matrix)

    remaining_tracks = [t for idx, t in enumerate(unassigned_tracks) if idx != r]
    remaining_meas = [m for idx, m in enumerate(unassigned_meas) if idx != c]

    return (track, meas, new_matrix, remaining_tracks, remaining_meas)


def associate_and_update(
    manager: Any,
    meas_list: Sequence[Any],
    filter_obj: Any,
    sensor: Any,
) -> None:
    """Greedy association loop with EKF updates and track management.

    Args:
        manager: Track manager (``track_list``, ``manage_tracks``, ...).
        meas_list: Lidar or camera measurements for this frame pass.
        filter_obj: Filter with ``predict`` / ``update``.
        sensor: Explicit lidar/camera pass sensor, including empty measurement frames.

    Returns:
        None; updates tracks in place and always finishes the lifecycle pass.
        Visibility is handled in the cost matrix, before pair removal. Camera
        updates refine state only; lidar hits alone increase existence scores.
    """
    unassigned_tracks = list(manager.track_list)
    unassigned_meas = list(meas_list)

    if unassigned_tracks and unassigned_meas:
        matrix = association_cost_matrix(unassigned_tracks, unassigned_meas)
        while True:
            track, meas, matrix, unassigned_tracks, unassigned_meas = pick_next_pair(
                matrix, unassigned_tracks, unassigned_meas
            )
            if track is np.nan or meas is np.nan:
                break
            filter_obj.update(track, meas)
            if hasattr(manager, "handle_updated_track"):
                manager.handle_updated_track(track, sensor)

    manager.manage_tracks(unassigned_tracks, unassigned_meas, sensor)
