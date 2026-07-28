# Failure-detection data recording for real-robot evaluation (eval_g1.py).
#
# Mirrors unitree_lerobot/eval_robot/utils/sim_savedata_utils.py, but for the real
# robot: no sim reward topic exists, so episode success/fail is labeled by a human
# via the keyboard instead of a reward subscriber, and states/actions additionally
# carry qvel/torque and end-effector tactile/force so failure modes (stall,
# collision, slip) are recoverable from the recording.
import numpy as np
import torch
import logging_mp

logger_mp = logging_mp.get_logger(__name__)


def _to_uint8_hwc(value):
    """Convert a camera frame (torch tensor or ndarray, CHW or HWC) to a uint8 HWC ndarray for cv2.imwrite."""
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    if value.ndim == 3 and value.shape[0] in (1, 3, 4):
        value = np.transpose(value, (1, 2, 0))
    if value.dtype != np.uint8:
        value = (value * 255).astype(np.uint8) if value.max() <= 1.0 else value.astype(np.uint8)
    return value


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return value


def process_data_add_real(
    episode_writer,
    images: dict,
    current_arm_q,
    current_arm_dq,
    current_arm_tau,
    ee_state,
    action,
    arm_dof,
    ee_dof,
    timestamp,
    ee_touch=None,
    ee_force=None,
):
    """Record one timestep of a real-robot rollout for later failure-detection labeling.

    images: dict[name -> frame] with every available camera view (top/wrist, left/right).
    ee_touch / ee_force: raw arrays from ee_shared_mem["touch"]/["force"] (inspire1), or None
        if the current end-effector doesn't expose tactile/force sensing.
    """
    if episode_writer is None:
        return

    current_arm_q = _to_numpy(current_arm_q)
    current_arm_dq = _to_numpy(current_arm_dq) if current_arm_dq is not None else np.zeros_like(current_arm_q)
    current_arm_tau = _to_numpy(current_arm_tau) if current_arm_tau is not None else np.zeros_like(current_arm_q)
    ee_state = _to_numpy(ee_state) if ee_state is not None else np.array([])
    action = _to_numpy(action)

    colors = {name: _to_uint8_hwc(frame) for name, frame in images.items() if frame is not None}

    half = arm_dof // 2

    def _side(arr, lo, hi):
        return arr[lo:hi].tolist() if len(arr) >= hi else []

    states = {
        "left_arm": {
            "qpos": _side(current_arm_q, 0, half),
            "qvel": _side(current_arm_dq, 0, half),
            "torque": _side(current_arm_tau, 0, half),
        },
        "right_arm": {
            "qpos": _side(current_arm_q, half, arm_dof),
            "qvel": _side(current_arm_dq, half, arm_dof),
            "torque": _side(current_arm_tau, half, arm_dof),
        },
        "left_ee": {"qpos": _side(ee_state, 0, ee_dof), "qvel": [], "torque": []},
        "right_ee": {"qpos": _side(ee_state, ee_dof, 2 * ee_dof), "qvel": [], "torque": []},
        "body": {"qpos": []},
    }
    actions = {
        "left_arm": {"qpos": _side(action, 0, half), "qvel": [], "torque": []},
        "right_arm": {"qpos": _side(action, half, arm_dof), "qvel": [], "torque": []},
        "left_ee": {"qpos": _side(action, arm_dof, arm_dof + ee_dof), "qvel": [], "torque": []},
        "right_ee": {"qpos": _side(action, arm_dof + ee_dof, arm_dof + 2 * ee_dof), "qvel": [], "torque": []},
        "body": {"qpos": []},
    }

    tactiles = None
    if ee_touch is not None or ee_force is not None:
        tactiles = {
            "touch": _to_numpy(ee_touch).tolist() if ee_touch is not None else [],
            "force": _to_numpy(ee_force).tolist() if ee_force is not None else [],
        }

    episode_writer.add_item(colors, states=states, actions=actions, tactiles=tactiles, timestamp=timestamp)
