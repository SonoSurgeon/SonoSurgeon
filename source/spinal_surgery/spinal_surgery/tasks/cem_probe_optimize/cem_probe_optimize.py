from __future__ import annotations

import argparse
import os
import json
import math
import time
from dataclasses import dataclass, asdict
from typing import Any
import csv
import sys
import wandb

import torch
import numpy as np

from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="CEM optimization of fixed probe target for surgery policy.")
parser.add_argument(
    "--cfg",
    type=str,
    default="/home/idsia/SonoGym/source/spinal_surgery/spinal_surgery/tasks/cem_probe_optimize/cem_probe_optimize_cfg.yaml",
    help="Path to cem_probe_optimize_cfg.yaml",
)
parser.add_argument(
    "--run_mode",
    type=str,
    default="play",
    choices=["play", "train"],
    help="Episode duration mode used by the surgery task YAML.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# The surgery task infers play/train from sys.argv[0] at import time.
# We modify only this script, so the task sees "play" or "train"
# in the entry script name without touching the task file.
_original_argv0 = sys.argv[0]
root, ext = os.path.splitext(_original_argv0)

if args_cli.run_mode == "play" and "play" not in os.path.basename(_original_argv0).lower():
    sys.argv[0] = f"{root}_play{ext or '.py'}"
elif args_cli.run_mode == "train" and "train" not in os.path.basename(_original_argv0).lower():
    sys.argv[0] = f"{root}_train{ext or '.py'}"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# Imports after Isaac app launch
# -----------------------------------------------------------------------------
import gymnasium as gym
from ruamel.yaml import YAML

import spinal_surgery
from isaaclab_tasks.utils import parse_env_cfg


# =============================================================================
# Utilities
# =============================================================================

def load_yaml(path: str) -> dict[str, Any]:
    yaml = YAML(typ="safe")
    with open(path, "r") as f:
        return yaml.load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# CEM core
# =============================================================================

@dataclass
class CEMState:
    iteration: int
    mean: list[float]
    std: list[float]
    best_theta: list[float]
    best_score: float


class CEMOptimizer:
    """
    Cross-Entropy Method optimizer.

    This class knows nothing about Isaac/SonoGym.
    It only samples candidate theta vectors and updates the search distribution.
    """

    def __init__(
        self,
        init_mean: list[float],
        init_std: list[float],
        lower_bounds: list[float],
        upper_bounds: list[float],
        population_size: int,
        elite_frac: float,
        min_std: list[float],
        alpha: float,
        seed: int = 0,
    ):
        self.rng = np.random.default_rng(seed)

        self.mean = np.asarray(init_mean, dtype=np.float64)
        self.std = np.asarray(init_std, dtype=np.float64)

        self.lower_bounds = np.asarray(lower_bounds, dtype=np.float64)
        self.upper_bounds = np.asarray(upper_bounds, dtype=np.float64)
        self.min_std = np.asarray(min_std, dtype=np.float64)

        self.population_size = int(population_size)
        self.elite_frac = float(elite_frac)
        self.num_elites = max(1, int(round(self.population_size * self.elite_frac)))

        self.alpha = float(alpha)

        self.best_theta = None
        self.best_score = -float("inf")

    def sample_population(self) -> np.ndarray:
        samples = self.rng.normal(
            loc=self.mean,
            scale=self.std,
            size=(self.population_size, self.mean.shape[0]),
        )

        samples = np.clip(samples, self.lower_bounds, self.upper_bounds)
        return samples

    def update(self, population: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
        assert population.shape[0] == scores.shape[0]

        order = np.argsort(scores)[::-1]
        elite_idx = order[: self.num_elites]
        elites = population[elite_idx]
        elite_scores = scores[elite_idx]

        elite_mean = elites.mean(axis=0)
        elite_std = elites.std(axis=0)

        # Smooth update
        self.mean = self.alpha * elite_mean + (1.0 - self.alpha) * self.mean
        self.std = self.alpha * elite_std + (1.0 - self.alpha) * self.std
        self.std = np.maximum(self.std, self.min_std)

        if float(scores[order[0]]) > self.best_score:
            self.best_score = float(scores[order[0]])
            self.best_theta = population[order[0]].copy()

        return {
            "elite_idx": elite_idx.tolist(),
            "elite_scores": elite_scores.tolist(),
            "elite_thetas": elites.tolist(),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "best_theta": self.best_theta.tolist(),
            "best_score": self.best_score,
        }


# =============================================================================
# Simulation / policy evaluation
# =============================================================================

@dataclass
class CandidateResult:
    theta: list[float]
    score: float
    mean_reward: float
    std_reward: float
    safety_ratio: float
    success_rate: float
    ood_rate: float
    mean_final_pos_err_mm: float
    mean_final_angle_err_deg: float
    er_score: float


class SurgeryPolicy:
    """
    Wrapper for exported TorchScript surgery policy.

    Expected exported signature:
        policy.forward_surgery(image, pos, quat)
    """

    def __init__(self, policy_path: str, device: torch.device):
        if not os.path.exists(policy_path):
            raise FileNotFoundError(f"Policy not found: {policy_path}")

        self.device = device
        self.policy = torch.jit.load(policy_path, map_location=device)
        self.policy.eval()

    @torch.no_grad()
    def act(self, obs_policy: dict[str, torch.Tensor]) -> torch.Tensor:
        image = obs_policy["image"].to(self.device, dtype=torch.float32)
        pos = obs_policy["pos"].to(self.device, dtype=torch.float32)
        quat = obs_policy["quat"].to(self.device, dtype=torch.float32)

        action = self.policy.forward_surgery(image, pos, quat)

        if action.ndim == 1:
            action = action.unsqueeze(0)

        return action


class SimulationEvaluator:
    """
    Evaluation layer.

    This class owns:
    - real surgery env
    - exported surgery policy
    - logic to overwrite probe target
    - rollout loop
    - metric extraction
    """

    def __init__(self, cfg: dict[str, Any], device: str):
        self.cfg = cfg
        self.task = cfg["task"]
        self.num_envs = int(cfg["num_envs"])
        self.device_str = device

        self.eval_cfg = cfg["evaluation"]

        self.success_pos_thr_mm = float(self.eval_cfg["success"]["pos_thr_mm"])
        self.success_ang_thr_deg = float(self.eval_cfg["success"]["angle_thr_deg"])

        self.ood_pos_thr_mm = float(self.eval_cfg["ood"]["pos_thr_mm"])
        self.ood_ang_thr_deg = float(self.eval_cfg["ood"]["angle_thr_deg"])

        print("[SIM] Building env...")
        env_cfg = parse_env_cfg(
            self.task,
            device=device,
            num_envs=self.num_envs,
            use_fabric=True,
        )

        self.env = gym.make(self.task, cfg=env_cfg)
        self.unwrapped = self.env.unwrapped
        

        self.device = torch.device(device)
        self.policy = SurgeryPolicy(cfg["policy_path"], self.device)

        print("[SIM] Env ready.")
        print(f"[SIM] num_envs: {self.num_envs}")
        print(f"[SIM] policy  : {cfg['policy_path']}")

    # -------------------------------------------------------------------------
    # Probe target override
    # -------------------------------------------------------------------------
    def set_probe_target(self, theta: np.ndarray) -> None:
        """
        theta = [x_shift, z_shift] or [x_shift, z_shift, yaw]

        Supports both YAML formats:

        Old:
            motion_planning:
            vertebra_to_US_2d_pos: ...
            ...

        New:
            run: random_probe
            motion_planning:
            standard_surgery:
                ...
            random_probe:
                ...
        """

        x_shift = float(theta[0])
        z_shift = float(theta[1])
        yaw = float(theta[2]) if len(theta) >= 3 else None

        env_module = __import__(
            self.unwrapped.__class__.__module__,
            fromlist=["scene_cfg"],
        )

        scene_cfg = env_module.scene_cfg

        motion_root = scene_cfg["motion_planning"]
        run_name = scene_cfg.get("run", None)

        if run_name is not None and run_name in motion_root:
            mp = motion_root[run_name]
        else:
            mp = motion_root

        # Fixed candidate probe target.
        mp["vertebra_to_US_2d_pos"] = [x_shift, z_shift]

        if yaw is not None:
            mp["US_target_2d_angle"] = yaw

        if "US_roll_adj" in self.eval_cfg and self.eval_cfg["US_roll_adj"] is not None:
            mp["US_roll_adj"] = float(self.eval_cfg["US_roll_adj"])

        # Disable additive x/z target randomization.
        if bool(self.eval_cfg.get("disable_probe_target_randomization", True)):
            if "vertebra_to_US_rand_range" in mp:
                mp["vertebra_to_US_rand_range"]["x"] = [0.0, 0.0]
                mp["vertebra_to_US_rand_range"]["z"] = [0.0, 0.0]

            if "vertebra_to_US_rand_max" in mp:
                mp["vertebra_to_US_rand_max"] = 0.0

        # Disable roll randomization.
        if bool(self.eval_cfg.get("disable_roll_randomization", True)):
            if "US_roll_rand_max" in mp:
                mp["US_roll_rand_max"] = 0.0

        # Disable yaw randomization.
        if bool(self.eval_cfg.get("disable_yaw_randomization", True)):
            if "US_yaw_rand_max" in mp:
                mp["US_yaw_rand_max"] = 0.0

    # -------------------------------------------------------------------------
    # Observation helpers
    # -------------------------------------------------------------------------
    def _extract_policy_obs(self, obs: Any) -> dict[str, torch.Tensor]:
        """
        Expected surgery obs format:
            obs["policy"]["image"]
            obs["policy"]["pos"]
            obs["policy"]["quat"]

        Some wrappers/envs may directly return:
            obs["image"], obs["pos"], obs["quat"]
        """

        if isinstance(obs, dict) and "policy" in obs:
            obs = obs["policy"]

        if not isinstance(obs, dict):
            raise TypeError(f"Expected dict observation for surgery, got {type(obs)}")

        required = ["image", "pos", "quat"]
        for k in required:
            if k not in obs:
                raise KeyError(f"Missing obs key {k}. Available keys: {list(obs.keys())}")

        return obs

    # -------------------------------------------------------------------------
    # Metric helpers
    # -------------------------------------------------------------------------
    def _get_final_metrics(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract final CEM errors from the task.

        The task stores the last valid post-insertion radial/angular error
        before reset. This avoids using the raw terminal-frame value.
        """

        env = self.unwrapped

        if hasattr(env, "get_last_episode_final_error_stats"):
            radial_err_m, phi_err_deg = env.get_last_episode_final_error_stats()

            pos_err_mm = radial_err_m.detach().float() * 1000.0
            ang_err_deg = phi_err_deg.detach().float()

            return pos_err_mm.reshape(-1), ang_err_deg.reshape(-1)

        raise RuntimeError(
            "Env does not expose get_last_episode_final_error_stats(). "
            "Add the ER buffers/getter to the task before using use_er."
        )  

    def _compute_er_score(self, mean_final_pos_err_mm: float) -> float:
        """
        Radial-error score.

        Linear scale:
            0 mm   -> 80
            10 mm  -> 0
            >10 mm -> 0
        """

        if not np.isfinite(mean_final_pos_err_mm):
            return 0.0

        er_score = 80.0 * (1.0 - mean_final_pos_err_mm / 10.0)
        er_score = np.clip(er_score, 0.0, 80.0)

        return float(er_score)

    def _compute_score(
        self,
        mean_reward: float,
        safety_ratio: float,
        success_rate: float,
        ood_rate: float,
        mean_final_pos_err_mm: float,
    ) -> tuple[float, float]:
        """
        Computes the CEM score.

        The base score follows the existing flags.
        If use_er is enabled, an additional radial-error score is added:

            0 mm   -> +50
            10 mm  -> +0
            >10 mm -> +0
        """

        if bool(self.eval_cfg.get("use_reward_plus_half_sr_percent", False)):
            base_score = float(mean_reward + 50.0 * safety_ratio)

        elif bool(self.eval_cfg.get("use_safety_ratio_only", False)):
            base_score = float(safety_ratio)

        elif bool(self.eval_cfg.get("use_mean_reward_only", True)):
            base_score = float(mean_reward)

        else:
            weights = self.eval_cfg["score_weights"]
            base_score = float(
                mean_reward
                + float(weights.get("safety_ratio_bonus", 0.0)) * safety_ratio
                + float(weights["success_bonus"]) * success_rate
                - float(weights["ood_penalty"]) * ood_rate
            )

        er_score = 0.0
        if bool(self.eval_cfg.get("use_er", False)):
            er_score = self._compute_er_score(mean_final_pos_err_mm)
            base_score += er_score

        return float(base_score), float(er_score)
    # -------------------------------------------------------------------------
    # Candidate evaluation
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def evaluate_candidate(self, theta: np.ndarray) -> CandidateResult:
        self.set_probe_target(theta)

        rollouts_target = int(self.eval_cfg["rollouts_per_candidate"])

        episode_rewards: list[float] = []
        episode_safety_ratios: list[float] = []
        final_pos_errs_mm: list[float] = []
        final_ang_errs_deg: list[float] = []

        completed = 0

        while completed < rollouts_target:
            obs, _ = self.env.reset()

            active = torch.ones((self.num_envs,), dtype=torch.bool, device=self.device)
            reward_sum = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)

            while active.any():
                obs_policy = self._extract_policy_obs(obs)
                action = self.policy.act(obs_policy)

                obs, reward, terminated, truncated, info = self.env.step(action)

                done = torch.logical_or(terminated, truncated).to(self.device)
                reward_sum += reward.to(self.device).float()

                newly_done = torch.logical_and(done, active)

                if newly_done.any():
                    pos_err_mm, ang_err_deg = self._get_final_metrics()

                    # Current episode counters, valid if the env has NOT auto-reset yet.
                    if hasattr(self.unwrapped, "get_safety_ratio_stats"):
                        curr_sr, curr_safe_count, curr_post_count = self.unwrapped.get_safety_ratio_stats()
                        curr_sr = curr_sr.detach()
                        curr_safe_count = curr_safe_count.detach()
                        curr_post_count = curr_post_count.detach()
                    else:
                        curr_sr = torch.zeros((self.num_envs,), device=self.device)
                        curr_safe_count = torch.zeros((self.num_envs,), device=self.device)
                        curr_post_count = torch.zeros((self.num_envs,), device=self.device)

                    # Last completed episode counters, valid if the env HAS auto-reset.
                    if hasattr(self.unwrapped, "get_last_episode_safety_ratio_stats"):
                        last_sr, last_safe_count, last_post_count = self.unwrapped.get_last_episode_safety_ratio_stats()
                        last_sr = last_sr.detach()
                        last_safe_count = last_safe_count.detach()
                        last_post_count = last_post_count.detach()
                    else:
                        last_sr = torch.zeros((self.num_envs,), device=self.device)
                        last_safe_count = torch.zeros((self.num_envs,), device=self.device)
                        last_post_count = torch.zeros((self.num_envs,), device=self.device)

                    # Use current if it has counted post-insertion states.
                    # Otherwise use last completed episode.
                    use_current = curr_post_count > 0.0

                    safety_ratio_now = torch.where(use_current, curr_sr, last_sr)
                    safe_count_now = torch.where(use_current, curr_safe_count, last_safe_count)
                    post_count_now = torch.where(use_current, curr_post_count, last_post_count)

                    ids = torch.nonzero(newly_done, as_tuple=False).reshape(-1)

                    for idx in ids.tolist():
                        if completed >= rollouts_target:
                            break
                        """
                        print(
                            f"[SR DEBUG] env={idx} "
                            f"sr={float(safety_ratio_now[idx].detach().cpu()):.4f} "
                            f"safe={float(safe_count_now[idx].detach().cpu()):.1f} "
                            f"post={float(post_count_now[idx].detach().cpu()):.1f}"
                        )
                        """
                        episode_rewards.append(float(reward_sum[idx].detach().cpu()))
                        episode_safety_ratios.append(float(safety_ratio_now[idx].detach().cpu()))
                        final_pos_errs_mm.append(float(pos_err_mm[idx].detach().cpu()))
                        final_ang_errs_deg.append(float(ang_err_deg[idx].detach().cpu()))

                        completed += 1

                    active[newly_done] = False

                # Safety guard for weird envs that auto-reset internally.
                if completed >= rollouts_target:
                    break

        rewards_np = np.asarray(episode_rewards, dtype=np.float64)
        safety_np = np.asarray(episode_safety_ratios, dtype=np.float64)
        pos_np = np.asarray(final_pos_errs_mm, dtype=np.float64)
        ang_np = np.asarray(final_ang_errs_deg, dtype=np.float64)

        success_mask = (pos_np <= self.success_pos_thr_mm) & (ang_np <= self.success_ang_thr_deg)
        ood_mask = (pos_np > self.ood_pos_thr_mm) | (ang_np > self.ood_ang_thr_deg)

        mean_reward = float(np.nanmean(rewards_np))
        std_reward = float(np.nanstd(rewards_np))
        safety_ratio = float(np.nanmean(safety_np))

        success_rate = float(np.nanmean(success_mask.astype(np.float64)))
        ood_rate = float(np.nanmean(ood_mask.astype(np.float64)))

        mean_pos = float(np.nanmean(pos_np))
        mean_ang = float(np.nanmean(ang_np))

        score, er_score = self._compute_score(
            mean_reward=mean_reward,
            safety_ratio=safety_ratio,
            success_rate=success_rate,
            ood_rate=ood_rate,
            mean_final_pos_err_mm=mean_pos,
        )

        return CandidateResult(
            theta=[float(v) for v in theta],
            score=score,
            mean_reward=mean_reward,
            std_reward=std_reward,
            safety_ratio=safety_ratio,
            success_rate=success_rate,
            ood_rate=ood_rate,
            mean_final_pos_err_mm=mean_pos,
            mean_final_angle_err_deg=mean_ang,
            er_score=er_score,
        )

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass


# =============================================================================
# Main optimization loop
# =============================================================================

def save_json(path: str, data: Any) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def append_csv(path: str, row: dict[str, Any]) -> None:
    file_exists = os.path.exists(path)

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

def run_cem(cfg: dict[str, Any], device: str) -> None:
    output_dir = cfg["output_dir"]
    ensure_dir(output_dir)

    log_cfg = cfg.get("logging", {})
    use_wandb = bool(log_cfg.get("use_wandb", False))

    wandb_run = None
    if use_wandb:
        if wandb is None:
            print("[WANDB] wandb not installed. Continuing without wandb.")
            use_wandb = False
        else:
            wandb_run = wandb.init(
                project=log_cfg.get("wandb_project", "cem-probe-optimization"),
                entity=log_cfg.get("wandb_entity", None),
                name=log_cfg.get("wandb_run_name", None),
                config=cfg,
            )

    # Save config copy
    save_json(os.path.join(output_dir, "config_used.json"), cfg)

    search = cfg["search_space"]
    lower_bounds = [
        float(search["x_shift"]["min"]),
        float(search["z_shift"]["min"]),
    ]
    upper_bounds = [
        float(search["x_shift"]["max"]),
        float(search["z_shift"]["max"]),
    ]

    if "yaw" in search:
        lower_bounds.append(float(search["yaw"]["min"]))
        upper_bounds.append(float(search["yaw"]["max"]))

    cem_cfg = cfg["cem"]

    optimizer = CEMOptimizer(
        init_mean=cem_cfg["init_mean"],
        init_std=cem_cfg["init_std"],
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        population_size=int(cem_cfg["population_size"]),
        elite_frac=float(cem_cfg["elite_frac"]),
        min_std=cem_cfg["min_std"],
        alpha=float(cem_cfg["alpha"]),
        seed=int(cfg.get("seed", 0)),
    )

    evaluator = SimulationEvaluator(cfg, device=device)

    all_iterations: list[dict[str, Any]] = []

    try:
        num_iterations = int(cem_cfg["num_iterations"])

        for it in range(num_iterations):
            print("")
            print("=" * 80)
            print(f"[CEM] Iteration {it + 1}/{num_iterations}")
            print(f"[CEM] mean: {optimizer.mean.tolist()}")
            print(f"[CEM] std : {optimizer.std.tolist()}")
            print("=" * 80)

            population = optimizer.sample_population()

            candidate_results: list[CandidateResult] = []
            scores = []

            t0 = time.time()

            for i, theta in enumerate(population):
                result = evaluator.evaluate_candidate(theta)
                candidate_results.append(result)
                scores.append(result.score)

                candidate_log = {
                    "iteration": it,
                    "candidate": i,

                    "score": result.score,
                    "mean_reward": result.mean_reward,
                    "std_reward": result.std_reward,
                    "safety_ratio": result.safety_ratio,
                    "success_rate": result.success_rate,
                    "ood_rate": result.ood_rate,
                    "mean_final_pos_err_mm": result.mean_final_pos_err_mm,
                    "mean_final_angle_err_deg": result.mean_final_angle_err_deg,
                    "er_score": result.er_score,

                    "theta_x_shift": result.theta[0],
                    "theta_z_shift": result.theta[1],
                    "theta_yaw": result.theta[2] if len(result.theta) > 2 else float("nan"),

                    "cem_mean_x": float(optimizer.mean[0]),
                    "cem_mean_z": float(optimizer.mean[1]),
                    "cem_mean_yaw": float(optimizer.mean[2]) if len(optimizer.mean) > 2 else float("nan"),

                    "cem_std_x": float(optimizer.std[0]),
                    "cem_std_z": float(optimizer.std[1]),
                    "cem_std_yaw": float(optimizer.std[2]) if len(optimizer.std) > 2 else float("nan"),
                }

                if bool(log_cfg.get("save_csv", True)):
                    append_csv(
                        os.path.join(output_dir, "cem_candidates.csv"),
                        candidate_log,
                    )

                if use_wandb:
                    wandb.log({
                        "candidate/score": result.score,
                        "candidate/mean_reward": result.mean_reward,
                        "candidate/std_reward": result.std_reward,
                        "candidate/safety_ratio": result.safety_ratio,
                        "candidate/success_rate": result.success_rate,
                        "candidate/ood_rate": result.ood_rate,
                        "candidate/mean_final_pos_err_mm": result.mean_final_pos_err_mm,
                        "candidate/mean_final_angle_err_deg": result.mean_final_angle_err_deg,
                        "candidate/er_score": result.er_score,
                        "candidate/theta_x_shift": result.theta[0],
                        "candidate/theta_z_shift": result.theta[1],
                        "candidate/theta_yaw": result.theta[2] if len(result.theta) > 2 else float("nan"),

                        "cem/current_mean_x": float(optimizer.mean[0]),
                        "cem/current_mean_z": float(optimizer.mean[1]),
                        "cem/current_mean_yaw": float(optimizer.mean[2]) if len(optimizer.mean) > 2 else float("nan"),

                        "cem/current_std_x": float(optimizer.std[0]),
                        "cem/current_std_z": float(optimizer.std[1]),
                        "cem/current_std_yaw": float(optimizer.std[2]) if len(optimizer.std) > 2 else float("nan"),

                        "iteration": it,
                        "candidate": i,
                    })

                if bool(cfg["logging"].get("print_every_candidate", True)):
                        print(
                            f"[CAND {i:03d}] "
                            f"theta={result.theta} | "
                            f"score={result.score:.3f} | "
                            f"reward={result.mean_reward:.3f} ± {result.std_reward:.3f} | "
                            f"safety={100.0 * result.safety_ratio:.1f}% | "
                            f"er={result.mean_final_pos_err_mm:.2f} mm | "
                            f"er_score={result.er_score:.2f}"
                        )
                        #f"succ={100.0 * result.success_rate:.1f}% | "
                        #f"ood={100.0 * result.ood_rate:.1f}% | "
                        #f"pos={result.mean_final_pos_err_mm:.2f} mm | "
                        #f"ang={result.mean_final_angle_err_deg:.2f} deg"
                    

            scores_np = np.asarray(scores, dtype=np.float64)
            update_info = optimizer.update(population, scores_np)

            iter_log = {
                "iteration": it,

                "iteration_best_score": float(np.max(scores_np)),
                "iteration_mean_score": float(np.mean(scores_np)),
                "iteration_std_score": float(np.std(scores_np)),

                "best_score_so_far": float(optimizer.best_score),
                "best_x_shift": float(optimizer.best_theta[0]),
                "best_z_shift": float(optimizer.best_theta[1]),
                "best_yaw": float(optimizer.best_theta[2]) if len(optimizer.best_theta) > 2 else float("nan"),

                "cem_new_mean_x": float(optimizer.mean[0]),
                "cem_new_mean_z": float(optimizer.mean[1]),
                "cem_new_mean_yaw": float(optimizer.mean[2]) if len(optimizer.mean) > 2 else float("nan"),

                "cem_new_std_x": float(optimizer.std[0]),
                "cem_new_std_z": float(optimizer.std[1]),
                "cem_new_std_yaw": float(optimizer.std[2]) if len(optimizer.std) > 2 else float("nan"),
            }

            if bool(log_cfg.get("save_csv", True)):
                append_csv(
                    os.path.join(output_dir, "cem_iterations.csv"),
                    iter_log,
                )

            if use_wandb:
                wandb.log({
                    "iteration/best_score": iter_log["iteration_best_score"],
                    "iteration/mean_score": iter_log["iteration_mean_score"],
                    "iteration/std_score": iter_log["iteration_std_score"],

                    "best/score_so_far": iter_log["best_score_so_far"],
                    "best/x_shift": iter_log["best_x_shift"],
                    "best/z_shift": iter_log["best_z_shift"],
                    "best/yaw": iter_log["best_yaw"],

                    "cem/new_mean_x": iter_log["cem_new_mean_x"],
                    "cem/new_mean_z": iter_log["cem_new_mean_z"],
                    "cem/new_mean_yaw": iter_log["cem_new_mean_yaw"],

                    "cem/new_std_x": iter_log["cem_new_std_x"],
                    "cem/new_std_z": iter_log["cem_new_std_z"],
                    "cem/new_std_yaw": iter_log["cem_new_std_yaw"],

                    "iteration": it,
                })

            elapsed = time.time() - t0

            iteration_record = {
                "iteration": it,
                "elapsed_sec": elapsed,
                "population": population.tolist(),
                "candidates": [asdict(r) for r in candidate_results],
                "cem_update": update_info,
            }

            all_iterations.append(iteration_record)

            print("")
            print(f"[CEM] iteration elapsed: {elapsed:.1f} s")
            print(f"[CEM] best theta so far: {optimizer.best_theta.tolist()}")
            print(f"[CEM] best score so far: {optimizer.best_score:.3f}")
            print(f"[CEM] new mean: {optimizer.mean.tolist()}")
            print(f"[CEM] new std : {optimizer.std.tolist()}")

            if bool(cfg["logging"].get("save_every_iteration", True)):
                save_json(
                    os.path.join(output_dir, "cem_iterations.json"),
                    all_iterations,
                )

                save_json(
                    os.path.join(output_dir, "cem_best.json"),
                    {
                        "best_theta": optimizer.best_theta.tolist(),
                        "best_score": optimizer.best_score,
                        "current_mean": optimizer.mean.tolist(),
                        "current_std": optimizer.std.tolist(),
                    },
                )

        print("")
        print("=" * 80)
        print("[CEM] DONE")
        print(f"[CEM] best theta: {optimizer.best_theta.tolist()}")
        print(f"[CEM] best score: {optimizer.best_score:.3f}")
        print("=" * 80)

    finally:
        evaluator.close()

        if use_wandb and wandb_run is not None:
            wandb.finish()


def main() -> None:
    cfg_path = os.path.abspath(args_cli.cfg)
    cfg = load_yaml(cfg_path)

    seed = int(cfg.get("seed", 0))
    set_global_seed(seed)

    print(f"[MAIN] cfg: {cfg_path}")
    print(f"[MAIN] task: {cfg['task']}")
    print(f"[MAIN] device: {args_cli.device}")

    run_cem(cfg, device=args_cli.device)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()