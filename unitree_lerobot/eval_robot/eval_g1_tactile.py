"""'
Refer to:   lerobot/lerobot/scripts/eval.py
            lerobot/lerobot/scripts/econtrol_robot.py
            lerobot/robot_devices/control_utils.py
"""

import time
import torch
import logging
import os
from pathlib import Path

import numpy as np
from pprint import pformat
from dataclasses import asdict
from torch import nn
from contextlib import nullcontext
from typing import Any
import copy as _copy
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
)
from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pretrained import PreTrainedPolicy
from multiprocessing.sharedctypes import SynchronizedArray
from lerobot.processor.rename_processor import rename_stats
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
)
from unitree_lerobot.eval_robot.make_robot import (
    setup_image_client,
    setup_robot_interface,
    process_images_and_observations,
)
from unitree_lerobot.eval_robot.utils.utils import (
    cleanup_resources,
    predict_action,
    to_list,
    to_scalar,
    EvalRealConfig,
)
from unitree_lerobot.utils.constants import ROBOT_CONFIGS
from unitree_lerobot.utils.preprocess_tactile_signal import parse_tactile_as_image, parse_tactile_as_state
from unitree_lerobot.eval_robot.utils.rerun_visualizer import RerunLogger, visualization_data

import logging_mp

logging_mp.basic_config(level=logging_mp.INFO)
logger_mp = logging_mp.get_logger(__name__)


def eval_policy(
    cfg: EvalRealConfig,
    dataset: LeRobotDataset,
    policy: PreTrainedPolicy | None = None,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None,
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None,
):
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."

    logger_mp.info(f"Arguments: {cfg}")

    if cfg.visualization:
        rerun_logger = RerunLogger()

    # Reset policy and processor if they are provided
    if policy is not None and preprocessor is not None and postprocessor is not None:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

    # Tactile configuration (optional)
    # NOTE: SmolVLA reads tactile inputs from keys like `observation.tactiles.*`.
    tactile_input_type = getattr(getattr(policy, "config", None), "tactile_input_type", "image")
    eval_config = {
        "tactile_enc_type": "state" if tactile_input_type == "state" else "image",
    } if cfg.tactile_use and tactile_input_type != "none" else {}

    # Currently tactile support is defined for Unitree_G1_Inspire
    robot_config = ROBOT_CONFIGS["Unitree_G1_Inspire"]
    tactile_names = getattr(robot_config, "tactiles", []) if cfg.tactile_use else []

    tactile_preprocess_fn = lambda x: x  # Placeholder
    if eval_config:
        if eval_config["tactile_enc_type"] == "image":
            tactile_preprocess_fn = parse_tactile_as_image
        elif eval_config["tactile_enc_type"] == "state":
            tactile_preprocess_fn = parse_tactile_as_state

    image_info = None
    h5_file = None
    h5_grp = None
    h5_len = 0
    try:
        # --- Setup Phase ---
        image_info = setup_image_client(cfg)
        robot_interface = setup_robot_interface(cfg)

        # Optional: append predictions to an HDF5 file (inference-time only).
        # Enabled when cfg.root is set to a directory path.
        if getattr(cfg, "root", ""):
            try:
                import h5py  # type: ignore

                out_dir = Path(os.path.expanduser(cfg.root))
                out_dir.mkdir(parents=True, exist_ok=True)
                h5_path = out_dir / "eval_g1_tactile_next_tactile.h5"
                h5_file = h5py.File(h5_path, "a")
                h5_grp = h5_file.require_group("predictions")
                if "step_idx" not in h5_grp:
                    h5_grp.create_dataset(
                        "step_idx",
                        shape=(0,),
                        maxshape=(None,),
                        dtype=np.int64,
                        chunks=True,
                    )
                h5_len = int(h5_grp["step_idx"].shape[0])
                logger_mp.info(f"Saving next-tactile predictions to: {h5_path}")
            except Exception as e:
                logger_mp.info(f"Could not enable next-tactile saving (cfg.root={cfg.root}): {e}")
                h5_file = None
                h5_grp = None

        tactile_logged = False
        tactile_missing_touch_logged = False

        # Unpack interfaces for convenience
        arm_ctrl, arm_ik, ee_shared_mem, arm_dof, ee_dof = (
            robot_interface[key] for key in ["arm_ctrl", "arm_ik", "ee_shared_mem", "arm_dof", "ee_dof"]
        )
        tv_img_array, wrist_img_array, tv_img_shape, wrist_img_shape, is_binocular, has_wrist_cam = (
            image_info[key]
            for key in [
                "tv_img_array",
                "wrist_img_array",
                "tv_img_shape",
                "wrist_img_shape",
                "is_binocular",
                "has_wrist_cam",
            ]
        )

        # Get initial pose from the first step of the dataset
        from_idx = dataset.meta.episodes["dataset_from_index"][0]
        step = dataset[from_idx]
        init_arm_pose = step["observation.state"][:arm_dof].cpu().numpy()

        user_input = input("Enter 's' to initialize the robot and start the evaluation: ")
        idx = 0
        print(f"user_input: {user_input}")
        full_state = None
        if user_input.lower() == "s":
            # "The initial positions of the robot's arm and fingers take the initial positions during data recording."
            logger_mp.info("Initializing robot to starting pose...")
            tau = robot_interface["arm_ik"].solve_tau(init_arm_pose)
            robot_interface["arm_ctrl"].ctrl_dual_arm(init_arm_pose, tau)
            time.sleep(1.0)  # Give time for the robot to move
            # --- Run Main Loop ---
            logger_mp.info(f"Starting evaluation loop at {cfg.frequency} Hz.")
            while True:
                loop_start_time = time.perf_counter()
                # 1. Get Observations
                observation, current_arm_q = process_images_and_observations(
                    tv_img_array, wrist_img_array, tv_img_shape, wrist_img_shape, is_binocular, has_wrist_cam, arm_ctrl
                )
                left_ee_state = right_ee_state = np.array([])
                tactile_state_vec = None
                if cfg.ee:
                    with ee_shared_mem["lock"]:
                        full_state = np.array(ee_shared_mem["state"][:])
                        left_ee_state = full_state[:ee_dof]
                        right_ee_state = full_state[ee_dof:]

                        # Tactile data for inspire hand
                        if eval_config and cfg.ee == "inspire1":
                            if "touch" not in ee_shared_mem:
                                if not tactile_missing_touch_logged:
                                    logger_mp.info(
                                        "tactile_use enabled but ee_shared_mem has no 'touch'. "
                                        "Tactile inputs will not be provided to the policy."
                                    )
                                    tactile_missing_touch_logged = True
                            else:
                                full_touch = np.array(ee_shared_mem["touch"][:])
                                left_touch = full_touch[:1062]
                                right_touch = full_touch[1062:]

                                if eval_config["tactile_enc_type"] == "image":
                                    tactile_images = tactile_preprocess_fn(
                                        {
                                            "left_tactile": left_touch,
                                            "right_tactile": right_touch,
                                        }
                                    )
                                    for tac_name in tactile_names:
                                        img = tactile_images[tac_name]
                                        # Training uses keys like `observation.tactiles.<name>`.
                                        # Keep tactile images as float in [0,1] and channel-first (C,H,W).
                                        timg = torch.from_numpy(img)
                                        if timg.dtype != torch.float32:
                                            timg = timg.to(torch.float32)
                                        if timg.ndim == 3 and timg.shape[-1] in (1, 3):
                                            timg = timg.permute(2, 0, 1).contiguous()
                                        observation[f"observation.tactiles.{tac_name}"] = timg
                                elif eval_config["tactile_enc_type"] == "state":
                                    tactiles = tactile_preprocess_fn(
                                        {
                                            "left_tactile": left_touch,
                                            "right_tactile": right_touch,
                                        }
                                    )
                                    tactile_state_vec = torch.cat(
                                        [torch.from_numpy(tactiles.pop(key)) for key in sorted(tactiles.keys())],
                                        dim=-1,
                                    ).to(torch.float32)
                                    # Provide tactile state under `observation.tactiles.*` so SmolVLA can pick it up.
                                    observation["observation.tactiles.state"] = tactile_state_vec
                state_tensor = torch.from_numpy(
                    np.concatenate((current_arm_q, left_ee_state, right_ee_state), axis=0)
                ).float()
                observation["observation.state"] = state_tensor

                if eval_config and not tactile_logged:
                    tactile_keys = sorted([k for k in observation.keys() if isinstance(k, str) and k.startswith("observation.tactiles.")])
                    if len(tactile_keys) == 0:
                        logger_mp.info(
                            "tactile debug: no `observation.tactiles.*` keys in observation (yet). "
                            f"tactile_input_type={tactile_input_type} tactile_enc_type={eval_config.get('tactile_enc_type')}"
                        )
                    else:
                        preview = []
                        for k in tactile_keys[:3]:
                            v = observation[k]
                            try:
                                preview.append(f"{k} dtype={getattr(v, 'dtype', None)} shape={tuple(v.shape)}")
                            except Exception:
                                preview.append(f"{k} (unprintable shape)")
                        logger_mp.info(
                            "tactile debug: injected `observation.tactiles.*` keys: "
                            f"count={len(tactile_keys)} tactile_input_type={tactile_input_type} tactile_enc_type={eval_config.get('tactile_enc_type')} "
                            f"preview={preview}"
                        )
                    tactile_logged = True
                # 2. Get Action from Policy
                action = predict_action(
                    observation,
                    policy,
                    get_safe_torch_device(policy.config.device),
                    preprocessor,
                    postprocessor,
                    policy.config.use_amp,
                    step["task"],
                    use_dataset=cfg.use_dataset,
                    robot_type=None,
                )
                action_np = action.cpu().numpy()

                # Save policy-predicted next tactile token embedding (SmolVLA only).
                if h5_file is not None and h5_grp is not None:
                    model = getattr(policy, "model", None)
                    next_tactile_pred = getattr(model, "_last_next_tactile_pred", None)
                    next_tactile_token_emb = getattr(model, "_last_predicted_tactile_token_emb", None)

                    if next_tactile_pred is not None and next_tactile_token_emb is not None:
                        # Shapes: pred (B, Dp), token (B, 1, Dt)
                        pred_np = next_tactile_pred.detach().float().cpu().numpy().reshape(-1)
                        tok_np = next_tactile_token_emb.detach().float().cpu().numpy().reshape(-1)

                        if "next_tactile_pred" not in h5_grp:
                            h5_grp.create_dataset(
                                "next_tactile_pred",
                                shape=(0, pred_np.shape[0]),
                                maxshape=(None, pred_np.shape[0]),
                                dtype=np.float32,
                                chunks=True,
                            )
                        if "next_tactile_token_emb" not in h5_grp:
                            h5_grp.create_dataset(
                                "next_tactile_token_emb",
                                shape=(0, tok_np.shape[0]),
                                maxshape=(None, tok_np.shape[0]),
                                dtype=np.float32,
                                chunks=True,
                            )

                        # Append 1 row.
                        for key in ("step_idx", "next_tactile_pred", "next_tactile_token_emb"):
                            ds = h5_grp[key]
                            ds.resize((h5_len + 1, *ds.shape[1:]))

                        h5_grp["step_idx"][h5_len] = np.int64(idx)
                        h5_grp["next_tactile_pred"][h5_len, :] = pred_np
                        h5_grp["next_tactile_token_emb"][h5_len, :] = tok_np
                        h5_len += 1
                        try:
                            h5_file.flush()
                        except Exception:
                            pass
                # 3. Execute Action
                arm_action = action_np[:arm_dof]
                tau = arm_ik.solve_tau(arm_action)
                arm_ctrl.ctrl_dual_arm(arm_action, tau)

                if cfg.ee:
                    ee_action_start_idx = arm_dof
                    left_ee_action = action_np[ee_action_start_idx : ee_action_start_idx + ee_dof]
                    right_ee_action = action_np[ee_action_start_idx + ee_dof : ee_action_start_idx + 2 * ee_dof]
                    # logger_mp.info(f"EE Action: left {left_ee_action}, right {right_ee_action}")

                    if isinstance(ee_shared_mem["left"], SynchronizedArray):
                        ee_shared_mem["left"][:] = to_list(left_ee_action)
                        ee_shared_mem["right"][:] = to_list(right_ee_action)
                    elif hasattr(ee_shared_mem["left"], "value") and hasattr(ee_shared_mem["right"], "value"):
                        ee_shared_mem["left"].value = to_scalar(left_ee_action)
                        ee_shared_mem["right"].value = to_scalar(right_ee_action)

                if cfg.visualization:
                    visualization_data(idx, observation, state_tensor.numpy(), action_np, rerun_logger)
                idx += 1
                # Maintain frequency
                time.sleep(max(0, (1.0 / cfg.frequency) - (time.perf_counter() - loop_start_time)))
    except Exception as e:
        try:
            logger_mp.info(f"An error occurred: {e}")
        except Exception:
            logging.info(f"An error occurred: {e}")
    finally:
        try:
            if h5_file is not None:
                h5_file.close()
        except Exception:
            pass
        if image_info:
            cleanup_resources(image_info)


@parser.wrap()
def eval_main(cfg: EvalRealConfig):
    logging.info(pformat(asdict(cfg)))

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Making policy.")

    dataset = LeRobotDataset(repo_id=cfg.repo_id)

    try:
        policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta)
    except ValueError as e:
        msg = str(e)
        # If the policy expects generic camera keys (camera1, camera2, ...) but the dataset
        # uses different names (e.g. cam_left_high), try to remap dataset feature keys
        # to generic camera names and retry policy creation.
        if "Missing features" in msg and "Extra features" in msg:
            # Build a temporary copy of dataset.meta and remap only that copy's
            # observation image feature keys to the generic camera names so that
            # policy creation can be retried without mutating the real dataset
            # metadata (which other code relies on, e.g. video timestamps).
            tmp_meta = _copy.deepcopy(dataset.meta)
            try:
                features = tmp_meta.info.get("features", {})
                new_features = {}
                cam_idx = 1
                for k, v in features.items():
                    if k.startswith("observation.images."):
                        new_key = f"observation.images.camera{cam_idx}"
                        new_features[new_key] = v
                        cam_idx += 1
                    else:
                        new_features[k] = v

                tmp_meta.info["features"] = new_features

                # retry policy creation using the temporary meta only
                policy = make_policy(cfg=cfg.policy, ds_meta=tmp_meta)
            except Exception:
                # If retry fails, re-raise the original error
                raise e
        else:
            raise

    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        dataset_stats=rename_stats(dataset.meta.stats, cfg.rename_map),
        preprocessor_overrides={
            "device_processor": {"device": cfg.policy.device},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
    )

    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        eval_policy(cfg, dataset, policy, preprocessor, postprocessor)

    logging.info("End of eval")


if __name__ == "__main__":
    init_logging()
    eval_main()
