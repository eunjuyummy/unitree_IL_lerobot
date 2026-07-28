"""'
Refer to:   lerobot/lerobot/scripts/eval.py
            lerobot/lerobot/scripts/econtrol_robot.py
            lerobot/robot_devices/control_utils.py
"""

import time
import torch
import logging

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
    KeyboardCommandListener,
)
from unitree_lerobot.eval_robot.utils.rerun_visualizer import RerunLogger, visualization_data
from unitree_lerobot.eval_robot.utils.episode_writer import EpisodeWriter
from unitree_lerobot.eval_robot.utils.real_savedata_utils import process_data_add_real

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

    episode_writer = None
    command_listener = None
    recording = False

    image_info = None
    try:
        # --- Setup Phase ---
        image_info = setup_image_client(cfg)
        robot_interface = setup_robot_interface(cfg)

        if cfg.save_data:
            head_shape = image_info["tv_img_shape"]  # (height, width, channels)
            episode_writer = EpisodeWriter(cfg.task_dir, frequency=cfg.frequency, image_size=[head_shape[1], head_shape[0]])

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

        def move_to_init_pose():
            tau = arm_ik.solve_tau(init_arm_pose)
            arm_ctrl.ctrl_dual_arm(init_arm_pose, tau)
            time.sleep(1.0)  # Give time for the robot to move

        user_input = input("Enter 's' to initialize the robot and start the evaluation: ")
        idx = 0
        print(f"user_input: {user_input}")
        full_state = None
        if user_input.lower() == "s":
            # "The initial positions of the robot's arm and fingers take the initial positions during data recording."
            logger_mp.info("Initializing robot to starting pose...")
            move_to_init_pose()
            # NOTE: this reads stdin on a background thread, so it must not start until after
            # the input() call above has consumed its line -- otherwise the two stdin readers
            # race and the prompt above can hang forever.
            command_listener = KeyboardCommandListener()
            info_msg = "Type 'r'+Enter at any time to move the arms back to the initial pose."
            if episode_writer is not None:
                episode_writer.create_episode()
                recording = True
                info_msg += (
                    " Type 's'+Enter to label the current episode a success, 'f'+Enter for "
                    "failure, or 'q'+Enter to stop recording -- labeling with 's'/'f' also "
                    "resets the arms to the initial pose for the next attempt."
                )
            logger_mp.info(info_msg)
            # --- Run Main Loop ---
            logger_mp.info(f"Starting evaluation loop at {cfg.frequency} Hz.")
            while True:
                loop_start_time = time.perf_counter()
                # 1. Get Observations
                observation, current_arm_q = process_images_and_observations(
                    tv_img_array, wrist_img_array, tv_img_shape, wrist_img_shape, is_binocular, has_wrist_cam, arm_ctrl
                )
                left_ee_state = right_ee_state = np.array([])
                if cfg.ee:
                    with ee_shared_mem["lock"]:
                        full_state = np.array(ee_shared_mem["state"][:])
                        left_ee_state = full_state[:ee_dof]
                        right_ee_state = full_state[ee_dof:]
                state_tensor = torch.from_numpy(
                    np.concatenate((current_arm_q, left_ee_state, right_ee_state), axis=0)
                ).float()
                observation["observation.state"] = state_tensor
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

                # Record this timestep for failure-detection data collection
                if recording:
                    images_to_save = {"cam_top": observation["observation.images.cam_left_high"]}
                    if is_binocular:
                        current_tv_image = tv_img_array.copy()
                        images_to_save["cam_top"] = current_tv_image[:, : tv_img_shape[1] // 2]
                        images_to_save["cam_top_right"] = current_tv_image[:, tv_img_shape[1] // 2 :]
                    if has_wrist_cam:
                        current_wrist_image = wrist_img_array.copy()
                        images_to_save["cam_wrist_left"] = current_wrist_image[:, : wrist_img_shape[1] // 2]
                        images_to_save["cam_wrist_right"] = current_wrist_image[:, wrist_img_shape[1] // 2 :]

                    current_arm_dq = (
                        arm_ctrl.get_current_dual_arm_dq() if hasattr(arm_ctrl, "get_current_dual_arm_dq") else None
                    )
                    current_arm_tau = (
                        arm_ctrl.get_current_dual_arm_tau() if hasattr(arm_ctrl, "get_current_dual_arm_tau") else None
                    )
                    ee_touch = ee_force = None
                    if cfg.ee:
                        with ee_shared_mem["lock"]:
                            if "touch" in ee_shared_mem:
                                ee_touch = np.array(ee_shared_mem["touch"][:])
                            if "force" in ee_shared_mem:
                                ee_force = np.array(ee_shared_mem["force"][:])

                    process_data_add_real(
                        episode_writer,
                        images_to_save,
                        current_arm_q,
                        current_arm_dq,
                        current_arm_tau,
                        full_state,
                        action_np,
                        arm_dof,
                        ee_dof,
                        timestamp=time.time(),
                        ee_touch=ee_touch,
                        ee_force=ee_force,
                    )

                # Drain any command typed on stdin (independent of recording state)
                key = command_listener.poll()
                if key == "r":
                    logger_mp.info("Moving back to the initial pose...")
                    move_to_init_pose()
                    logger_mp.info("Reached initial pose. Resuming policy control.")
                elif key in ("s", "f"):
                    if episode_writer is not None:
                        result = "success" if key == "s" else "fail"
                        episode_writer.save_episode(result)
                        logger_mp.info(f"Episode labeled '{result}'. Saving, then resetting to the initial pose...")
                        while not episode_writer.is_available:
                            time.sleep(0.01)
                        episode_writer.create_episode()
                        recording = True
                        move_to_init_pose()
                        logger_mp.info("Reached initial pose. New episode recording started.")
                    else:
                        logger_mp.info(f"Ignoring '{key}': recording is not enabled (pass --save_data=true).")
                elif key == "q":
                    if recording:
                        recording = False
                        logger_mp.info("Recording stopped (evaluation loop keeps running).")

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
        if image_info:
            cleanup_resources(image_info)
        if episode_writer is not None:
            episode_writer.close()


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
