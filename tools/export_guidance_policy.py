from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import torch
import torch.nn as nn

from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Export TorchScript policy from the real SKRL policy model")
parser.add_argument("--task", type=str, default="Isaac-robot-US-guidance-G1-v0", help="Gym task id, e.g. Isaac-Robot-US-Guidance-G1-v0")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_agent.pt")
parser.add_argument("--output", type=str, default=None, help="Output TorchScript path")
parser.add_argument("--num_envs", type=int, default=1, help="Number of envs for bootstrap")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# Imports that need Isaac app launched first
# -----------------------------------------------------------------------------
import gymnasium as gym
import spinal_surgery
import numpy as np

from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
from skrl.utils.runner.torch import Runner


class Float32ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        if isinstance(env.observation_space, gym.spaces.Dict):
            new_spaces = {}
            for k, sp in env.observation_space.spaces.items():
                if isinstance(sp, gym.spaces.Box):
                    new_spaces[k] = gym.spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=sp.shape,
                        dtype=np.float32,
                    )
                else:
                    new_spaces[k] = sp
            self.observation_space = gym.spaces.Dict(new_spaces)

        elif isinstance(env.observation_space, gym.spaces.Box):
            sp = env.observation_space
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=sp.shape,
                dtype=np.float32,
            )

    def observation(self, observation):
        if isinstance(observation, dict):
            return {
                k: (v.float() if isinstance(v, torch.Tensor) else v)
                for k, v in observation.items()
            }

        if isinstance(observation, torch.Tensor):
            return observation.float()

        return observation

@dataclass
class ExportResult:
    checkpoint_path: str
    exported_path: str
    action_dim: int


class DeterministicPolicyWrapper(nn.Module):
    """Wrap the real SKRL policy model and expose a simple Tensor -> Tensor forward."""

    def __init__(self, policy_model):
        super().__init__()
        self.policy_model = policy_model

    def _prepare_obs(self, obs: torch.Tensor) -> torch.Tensor:
        # The real policy expects float inputs
        if not isinstance(obs, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(obs)}")

        obs = obs.to(dtype=torch.float32)

        # If needed, you can uncomment this debug once:
        # print("[DEBUG] wrapper obs dtype:", obs.dtype, "shape:", tuple(obs.shape))

        return obs

    def _call_policy(self, obs: torch.Tensor) -> torch.Tensor:
        obs = self._prepare_obs(obs)

        out = self.policy_model.compute({"states": obs}, role="policy")

        if not isinstance(out, tuple):
            raise TypeError(f"Unexpected policy compute output type: {type(out)}")

        mean_actions = out[0]

        if not isinstance(mean_actions, torch.Tensor):
            raise TypeError(f"Policy mean output is not a tensor. Got: {type(mean_actions)}")

        return mean_actions

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self._call_policy(obs)

def _resolve_output_path(output_path: str | None) -> str:
    if output_path is not None:
        out = output_path
    else:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        policies_dir = os.path.join(
            repo_root, "source", "spinal_surgery", "spinal_surgery", "policies"
        )
        os.makedirs(policies_dir, exist_ok=True)
        out = os.path.join(policies_dir, "cem_policy.pt")

    out_dir = os.path.dirname(out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    return out


def _get_policy_model_from_runner(runner: Runner):
    print("[DEBUG] entering _get_policy_model_from_runner")

    if hasattr(runner, "agent"):
        agent = runner.agent
        print("[DEBUG] runner.agent found")
    elif hasattr(runner, "_agent"):
        agent = runner._agent
        print("[DEBUG] runner._agent found")
    else:
        raise RuntimeError("Could not find agent inside Runner")

    if hasattr(agent, "models"):
        models = agent.models
        print(f"[DEBUG] agent.models type: {type(models)}")
        if isinstance(models, dict):
            print(f"[DEBUG] model keys: {list(models.keys())}")
            if "policy" in models:
                print("[DEBUG] using models['policy']")
                return models["policy"]
            if "agent" in models:
                print("[DEBUG] using models['agent']")
                return models["agent"]

    if hasattr(agent, "policy"):
        print("[DEBUG] using agent.policy")
        return agent.policy

    raise RuntimeError("Could not locate the policy model inside the SKRL agent")


def build_env_and_runner(task: str, device: str, num_envs: int):
    print(f"[DEBUG] requested task: {task}")
    print("[DEBUG] before parse_env_cfg")

    env_cfg = parse_env_cfg(
        task,
        device=device,
        num_envs=num_envs,
        use_fabric=True,
    )

    print("[DEBUG] after parse_env_cfg")
    print("[DEBUG] before gym.make")
    env = gym.make(task, cfg=env_cfg)
    print("[DEBUG] after gym.make")

    print("[DEBUG] before Float32ObservationWrapper")
    env = Float32ObservationWrapper(env)
    print("[DEBUG] after Float32ObservationWrapper")

    print("[DEBUG] before load_cfg_from_registry")
    experiment_cfg = load_cfg_from_registry(task, "skrl_cfg_entry_point")
    print("[DEBUG] after load_cfg_from_registry")

    if "models" not in experiment_cfg:
        raise RuntimeError("La config skrl caricata non contiene 'models'")

    print("[DEBUG] before Runner(...)")
    runner = Runner(env, experiment_cfg)
    print("[DEBUG] after Runner(...)")

    return env, runner

def export_policy(
    task: str,
    checkpoint_path: str,
    output_path: str | None,
    device: str,
    num_envs: int,
) -> ExportResult:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_path = _resolve_output_path(output_path)

    print(f"[EXPORT] task      : {task}")
    print(f"[EXPORT] checkpoint: {checkpoint_path}")
    print(f"[EXPORT] output    : {output_path}")
    print(f"[EXPORT] device    : {device}")

    try:
        print("[DEBUG] before build_env_and_runner")
        env, runner = build_env_and_runner(task=task, device=device, num_envs=num_envs)
        print("[DEBUG] after build_env_and_runner")

        print("[DEBUG] before checkpoint load")
        if hasattr(runner, "agent") and hasattr(runner.agent, "load"):
            runner.agent.load(checkpoint_path)
            print("[DEBUG] checkpoint loaded through runner.agent.load")
        elif hasattr(runner, "_agent") and hasattr(runner._agent, "load"):
            runner._agent.load(checkpoint_path)
            print("[DEBUG] checkpoint loaded through runner._agent.load")
        else:
            raise RuntimeError("Runner agent does not expose a load(checkpoint) method")

        print("[DEBUG] before extracting policy model")
        policy_model = _get_policy_model_from_runner(runner)
        print(f"[DEBUG] policy model class: {policy_model.__class__.__name__}")

        policy_model.eval()
        print("[DEBUG] policy model set to eval")

        wrapper = DeterministicPolicyWrapper(policy_model).to(device).eval()
        print("[DEBUG] wrapper created")

        obs_space = env.observation_space
        if isinstance(obs_space, dict):
            obs_space = obs_space["policy"]

        obs_shape = tuple(obs_space.shape)

        print(f"[DEBUG] obs space shape: {obs_shape}")
        print(f"[DEBUG] obs space dtype: {getattr(obs_space, 'dtype', 'unknown')}")

        # IMPORTANT:
        # use a real observation from the env, so dtype/layout match the actual policy input
        print("[DEBUG] before env.reset() for tracing sample")
        obs, _ = env.reset()
        print("[DEBUG] after env.reset() for tracing sample")

        if isinstance(obs, dict):
            obs_sample = obs["policy"]
        else:
            obs_sample = obs

        # keep only first env if batched
        dummy = obs_sample[0:1].detach().clone().to(device=device, dtype=torch.float32)

        print(f"[DEBUG] dummy shape for trace: {tuple(dummy.shape)}")
        print(f"[DEBUG] dummy dtype for trace: {dummy.dtype}")
        print(f"[DEBUG] dummy min/max: {dummy.min().item():.6f} / {dummy.max().item():.6f}")

        with torch.no_grad():
            y1 = wrapper(dummy)
            y2 = wrapper(dummy)

        print(f"[DEBUG] wrapper output shape: {tuple(y1.shape)}")
        print(f"[DEBUG] repeat output sample 1: {y1[0].detach().cpu().numpy()}")
        print(f"[DEBUG] repeat output sample 2: {y2[0].detach().cpu().numpy()}")
        print(f"[DEBUG] max diff: {(y1 - y2).abs().max().item()}")

        traced = torch.jit.trace(wrapper, dummy, check_trace=True)

        print("[DEBUG] before traced.save")
        traced.save(output_path)
        print("[DEBUG] after traced.save")

        print(f"[EXPORT] obs shape : {(1, *obs_shape)}")
        print(f"[EXPORT] act shape : {tuple(y1.shape)}")
        print(f"[EXPORT] saved     : {os.path.exists(output_path)}")

        return ExportResult(
            checkpoint_path=checkpoint_path,
            exported_path=output_path,
            action_dim=int(y1.shape[-1]),
        )

    except Exception as e:
        print(f"[DEBUG][ERROR] export failed with: {repr(e)}")
        raise

    finally:
        try:
            env.close()
            print("[DEBUG] env closed")
        except Exception:
            print("[DEBUG] env close skipped")


def main():
    print("[DEBUG] entered main")
    result = export_policy(
        task=args_cli.task,
        checkpoint_path=args_cli.checkpoint,
        output_path=args_cli.output,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )
    print("Export completed:")
    print(f"  checkpoint : {result.checkpoint_path}")
    print(f"  exported   : {result.exported_path}")
    print(f"  action_dim : {result.action_dim}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()