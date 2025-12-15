#!/usr/bin/env python3
"""
Evaluate Erase Task Model - WITH Force/Torque (F/T)

Model: erase_task_ft_8epoch (trained WITH F/T sensor data)

Usage:
    python eval_erase_with_ft.py --plot --trajs 5
"""

import warnings
from dataclasses import dataclass, field
from typing import List, Literal

import numpy as np
import torch
import tyro

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING
from gr00t.experiment.data_config import load_data_config
from gr00t.model.policy import BasePolicy, Gr00tPolicy
from gr00t.utils.eval import calc_mse_for_single_trajectory

warnings.simplefilter("ignore", category=FutureWarning)


@dataclass
class ArgsConfig:
    """Configuration for WITH F/T model evaluation."""

    model_path: str = "/home/Isaac/Isaac-GR00T/output/erase_task_ft_8epoch"
    """Path to the model (WITH F/T)."""

    dataset_path: str = "/home/Isaac/Isaac-GR00T/datasets/Erase_for_gr00t"
    """Path to the dataset."""

    data_config: str = "rh20t_franka_with_ft"
    """Data config (WITH F/T)."""

    modality_keys: List[str] = field(default_factory=lambda: [
        "w_x", "w_y", "w_z",
        "v_x", "v_y", "v_z",
        "gripper_cmd",
        "joint_1", "joint_2", "joint_3", "joint_4",
        "joint_5", "joint_6", "joint_7"
    ])
    """Action modality keys (14D)."""

    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "oxe_droid"
    """Embodiment tag."""

    video_backend: Literal["decord", "torchvision_av", "torchcodec"] = "decord"
    """Video backend."""

    steps: int = 150
    """Steps per trajectory."""

    trajs: int = -1
    """Number of trajectories (-1 = all)."""

    start_traj: int = 0
    """Start trajectory index."""

    action_horizon: int = None
    """Action horizon (None = from config)."""

    denoising_steps: int = 4
    """Denoising steps."""

    plot: bool = True
    """Plot comparison."""

    plot_state: bool = False
    """Plot state joints."""

    save_plot_path: str = "/home/Isaac/Isaac-GR00T/Images/erase_with_ft.png"
    """Save path."""


def main(args: ArgsConfig):
    print("=" * 80)
    print("Erase Task Evaluation - WITH F/T Model")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Config: {args.data_config}")
    print("=" * 80)

    # Load data config
    data_config = load_data_config(args.data_config)

    if args.action_horizon is None:
        args.action_horizon = len(data_config.action_indices)
        print(f"Action horizon: {args.action_horizon}")

    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    print("\n[1/2] Loading WITH F/T model...")
    policy: BasePolicy = Gr00tPolicy(
        model_path=args.model_path,
        modality_config=modality_config,
        modality_transform=modality_transform,
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
        device=device,
    )
    print("Model loaded")

    # Load dataset
    print("\n[2/2] Loading dataset...")
    modality = policy.get_modality_config()
    dataset = LeRobotSingleDataset(
        dataset_path=args.dataset_path,
        modality_configs=modality,
        video_backend=args.video_backend,
        video_backend_kwargs=None,
        transforms=None,
        embodiment_tag=args.embodiment_tag,
    )

    print(f"Samples: {len(dataset)}, Trajectories: {len(dataset.trajectory_lengths)}")

    # Evaluate
    total_trajs = len(dataset.trajectory_lengths)
    num_trajs = total_trajs - args.start_traj if args.trajs == -1 else min(args.trajs, total_trajs - args.start_traj)

    print(f"\nEvaluating {num_trajs} trajectories...")

    all_mse = []
    for traj_id in range(args.start_traj, args.start_traj + num_trajs):
        traj_length = dataset.trajectory_lengths[traj_id]
        steps = min(args.steps, traj_length - args.action_horizon)

        if steps <= 0:
            print(f"Skip traj {traj_id}: too short")
            continue

        mse = calc_mse_for_single_trajectory(
            policy, dataset, traj_id,
            modality_keys=args.modality_keys,
            steps=steps,
            action_horizon=args.action_horizon,
            plot=args.plot,
            plot_state=args.plot_state,
            save_plot_path=args.save_plot_path,
        )
        print(f"Traj {traj_id}: MSE = {mse:.6f}")
        all_mse.append(mse)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY - WITH F/T Model")
    print("=" * 80)
    if all_mse:
        print(f"Trajectories: {len(all_mse)}")
        print(f"Avg MSE: {np.mean(all_mse):.6f}")
        print(f"Std MSE: {np.std(all_mse):.6f}")
    print("=" * 80)


if __name__ == "__main__":
    config = tyro.cli(ArgsConfig)
    main(config)
