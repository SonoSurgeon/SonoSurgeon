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
parser = argparse.ArgumentParser(
    description="Export TorchScript policy from the real SKRL policy model"
)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-robot-US-guidance-G1-v0",
    help="Gym task id, e.g. Isaac-Robot-US-Guidance-G1-v0",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="Path to best_agent.pt",
)
parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="Output TorchScript path",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Number of envs for bootstrap",
)

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


# =============================================================================
# Observation wrappers
# =============================================================================

class GuidanceFloat32ObservationWrapper(gym.ObservationWrapper):
    """
    Wrapper for navigation/guidance export.

    IMPORTANT:
    This keeps the old behavior. It does NOT squeeze observation_space shapes.
    That is necessary to keep compatibility with guidance checkpoints.
    """

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


class SurgeryFloat32ObservationWrapper(gym.ObservationWrapper):
    """
    Wrapper for surgery export.

    Surgery observations are dict observations:
        image, pos, quat

    This wrapper removes artificial singleton dimensions that can appear in the
    surgery observation layout.
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = self._convert_space(env.observation_space)

    def _convert_space(self, space):
        if isinstance(space, gym.spaces.Dict):
            return gym.spaces.Dict(
                {
                    k: self._convert_space(v)
                    for k, v in space.spaces.items()
                }
            )

        if isinstance(space, gym.spaces.Box):
            shape = tuple(space.shape)

            # Surgery image can appear as:
            #   image: (1, 7, 150, 200) -> (7, 150, 200)
            #   pos:   (1, 3)           -> (3,)
            #   quat:  (1, 4)           -> (4,)
            if len(shape) >= 2 and shape[0] == 1:
                shape = shape[1:]

            return gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=shape,
                dtype=np.float32,
            )

        return space

    def _convert_obs(self, obs):
        if isinstance(obs, dict):
            return {
                k: self._convert_obs(v)
                for k, v in obs.items()
            }

        if isinstance(obs, torch.Tensor):
            obs = obs.float()

            # Batched tensors:
            #   image: [B, 1, 7, 150, 200] -> [B, 7, 150, 200]
            #   pos:   [B, 1, 3]           -> [B, 3]
            #   quat:  [B, 1, 4]           -> [B, 4]
            if obs.ndim >= 3 and obs.shape[1] == 1:
                obs = obs.squeeze(1)

            return obs

        return obs

    def observation(self, observation):
        return self._convert_obs(observation)


# =============================================================================
# Export utilities
# =============================================================================

@dataclass
class ExportResult:
    checkpoint_path: str
    exported_path: str
    action_dim: int


def _infer_policy_kind_from_task(task: str) -> str:
    task_l = task.lower()

    if "surgery" in task_l or "guided-surgery" in task_l or "guided_surgery" in task_l:
        return "surgery"

    return "guidance"

class GuidanceDirectNetworkWrapper(nn.Module):
    def __init__(self, policy_model):
        super().__init__()

        self.features_extractor = policy_model._modules["features_extractor_container"]
        self.net = policy_model._modules["net_container"]
        self.policy_layer = policy_model._modules["policy_layer"]

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.float()

        x = self.features_extractor(obs)
        x = self.net(x)
        actions = self.policy_layer(x)

        return actions
    
class GuidanceComputePolicyWrapper(nn.Module):
    """
    Debug wrapper: usa esattamente policy_model.compute(...),
    cioè il percorso SKRL originale.
    NON lo usiamo per export, solo per confronto.
    """

    def __init__(self, policy_model):
        super().__init__()
        self.policy_model = policy_model

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.float()

        out = self.policy_model.compute(
            {"states": obs},
            role="policy",
        )

        if not isinstance(out, tuple):
            raise TypeError(f"Unexpected policy compute output type: {type(out)}")

        mean_actions = out[0]

        if not isinstance(mean_actions, torch.Tensor):
            raise TypeError(f"Policy mean output is not tensor: {type(mean_actions)}")

        return mean_actions

class GuidanceDeterministicPolicyWrapper(nn.Module):
    """
    Old stable guidance export path.

    Interface:
        forward(obs) -> action
    """

    def __init__(self, policy_model):
        super().__init__()
        self.policy_model = policy_model

    def _prepare_obs(self, obs: torch.Tensor) -> torch.Tensor:
        if not isinstance(obs, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(obs)}")

        return obs.to(dtype=torch.float32)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = self._prepare_obs(obs)

        out = self.policy_model.compute(
            {"states": obs},
            role="policy",
        )

        if not isinstance(out, tuple):
            raise TypeError(f"Unexpected policy compute output type: {type(out)}")

        mean_actions = out[0]

        if not isinstance(mean_actions, torch.Tensor):
            raise TypeError(
                f"Policy mean output is not a tensor. Got: {type(mean_actions)}"
            )

        return mean_actions


class SurgeryDeterministicPolicyWrapper(nn.Module):
    """
    Surgery export path.

    Exported interface:
        forward_surgery(image, pos, quat) -> action

    Internally, the SKRL generated model expects:
        inputs["states"] = flat tensor

    Then it calls:
        unflatten_tensorized_space(self.observation_space, inputs["states"])
    """

    def __init__(self, policy_model):
        super().__init__()
        self.policy_model = policy_model
        self.observation_space = policy_model.observation_space

    def _prepare_tensor(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(x)}")

        return x.to(dtype=torch.float32)

    def _flatten_surgery_states(
        self,
        image: torch.Tensor,
        pos: torch.Tensor,
        quat: torch.Tensor,
    ) -> torch.Tensor:
        image = self._prepare_tensor(image)
        pos = self._prepare_tensor(pos)
        quat = self._prepare_tensor(quat)

        # Expected image for Conv2d:
        #   [B, C, H, W]
        #
        # Sometimes Isaac/SKRL wrappers give:
        #   [B, 1, C, H, W]
        if image.ndim == 5 and image.shape[1] == 1:
            image = image.squeeze(1)

        if image.ndim != 4:
            raise RuntimeError(
                f"Expected surgery image as [B, C, H, W], got {tuple(image.shape)}"
            )

        if pos.ndim > 2:
            pos = pos.reshape(pos.shape[0], -1)

        if quat.ndim > 2:
            quat = quat.reshape(quat.shape[0], -1)

        batch_size = image.shape[0]

        tensors = {
            "image": image.reshape(batch_size, -1),
            "pos": pos.reshape(batch_size, -1),
            "quat": quat.reshape(batch_size, -1),
        }

        # Use the same key order used by the SKRL observation_space.
        if hasattr(self.observation_space, "spaces"):
            keys = list(self.observation_space.spaces.keys())
        else:
            keys = ["image", "pos", "quat"]

        flat_parts = []
        for key in keys:
            if key in tensors:
                flat_parts.append(tensors[key])

        if len(flat_parts) != 3:
            raise RuntimeError(
                f"Could not flatten surgery states. "
                f"observation_space keys={keys}, available={list(tensors.keys())}"
            )

        return torch.cat(flat_parts, dim=1)

    def _call_policy(self, states: torch.Tensor) -> torch.Tensor:
        out = self.policy_model.compute(
            {"states": states},
            role="policy",
        )

        if not isinstance(out, tuple):
            raise TypeError(f"Unexpected policy compute output type: {type(out)}")

        mean_actions = out[0]

        if not isinstance(mean_actions, torch.Tensor):
            raise TypeError(
                f"Policy mean output is not a tensor. Got: {type(mean_actions)}"
            )

        return mean_actions

    def forward_surgery(
        self,
        image: torch.Tensor,
        pos: torch.Tensor,
        quat: torch.Tensor,
    ) -> torch.Tensor:
        states = self._flatten_surgery_states(image, pos, quat)
        return self._call_policy(states)


def _resolve_output_path(
    output_path: str | None,
    policy_kind: str,
    checkpoint_path: str,
) -> str:
    if output_path is not None:
        out = output_path
    else:
        checkpoint_path = os.path.abspath(checkpoint_path)

        # Expected checkpoint layout:
        #   .../{name1}/{name2}/checkpoints/best_agent.pt
        checkpoints_dir = os.path.dirname(checkpoint_path)
        name2_dir = os.path.dirname(checkpoints_dir)
        name1_dir = os.path.dirname(name2_dir)

        name2 = os.path.basename(name2_dir)
        name1 = os.path.basename(name1_dir)

        if not name1 or not name2:
            raise RuntimeError(
                f"Could not infer output name from checkpoint path: {checkpoint_path}"
            )

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        policies_dir = os.path.join(
            repo_root,
            "source",
            "spinal_surgery",
            "spinal_surgery",
            "policies",
        )
        os.makedirs(policies_dir, exist_ok=True)

        out = os.path.join(policies_dir, f"{name1}_{name2}.pt")

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


def build_env_and_runner(
    task: str,
    device: str,
    num_envs: int,
    policy_kind: str,
):
    print(f"[DEBUG] requested task: {task}")
    print(f"[DEBUG] policy kind   : {policy_kind}")
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

    if policy_kind == "surgery":
        print("[DEBUG] before SurgeryFloat32ObservationWrapper")
        env = SurgeryFloat32ObservationWrapper(env)
        print("[DEBUG] after SurgeryFloat32ObservationWrapper")
    else:
        print("[DEBUG] before GuidanceFloat32ObservationWrapper")
        env = GuidanceFloat32ObservationWrapper(env)
        print("[DEBUG] after GuidanceFloat32ObservationWrapper")

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

    policy_kind = _infer_policy_kind_from_task(task)
    output_path = _resolve_output_path(
        output_path=output_path,
        policy_kind=policy_kind,
        checkpoint_path=checkpoint_path,
    )

    print(f"[EXPORT] task      : {task}")
    print(f"[EXPORT] checkpoint: {checkpoint_path}")
    print(f"[EXPORT] output    : {output_path}")
    print(f"[EXPORT] device    : {device}")
    print(f"[EXPORT] kind      : {policy_kind}")

    env = None

    try:
        print("[DEBUG] before build_env_and_runner")
        env, runner = build_env_and_runner(
            task=task,
            device=device,
            num_envs=num_envs,
            policy_kind=policy_kind,
        )
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
        print(policy_model)
        print(policy_model.__dict__.keys())
        print(f"[DEBUG] policy model class: {policy_model.__class__.__name__}")

        policy_model.eval()
        print("[DEBUG] policy model set to eval")

        obs_space = env.observation_space
        if isinstance(obs_space, gym.spaces.Dict):
            obs_space_policy = obs_space["policy"] if "policy" in obs_space else obs_space
        else:
            obs_space_policy = obs_space

        print("[DEBUG] observation space after wrapper:")
        if isinstance(obs_space_policy, gym.spaces.Dict):
            for k, sp in obs_space_policy.spaces.items():
                print(
                    f"[DEBUG] obs space[{k}] shape: {getattr(sp, 'shape', None)} "
                    f"dtype: {getattr(sp, 'dtype', 'unknown')}"
                )
            obs_shape = None
        else:
            obs_shape = tuple(obs_space_policy.shape)
            print(f"[DEBUG] obs space shape: {obs_shape}")
            print(f"[DEBUG] obs space dtype: {getattr(obs_space_policy, 'dtype', 'unknown')}")

        print("[DEBUG] before env.reset() for tracing sample")
        obs, _ = env.reset()
        print("[DEBUG] after env.reset() for tracing sample")

        if isinstance(obs, dict):
            obs_sample = obs["policy"] if "policy" in obs else obs
        else:
            obs_sample = obs

        if policy_kind == "surgery":
            if not isinstance(obs_sample, dict):
                raise TypeError(
                    f"Surgery export expected dict observation, got {type(obs_sample)}"
                )

            wrapper = SurgeryDeterministicPolicyWrapper(policy_model).to(device).eval()
            print("[DEBUG] surgery wrapper created")

            print(f"[DEBUG] raw obs image shape: {tuple(obs_sample['image'].shape)}")
            print(f"[DEBUG] raw obs pos shape  : {tuple(obs_sample['pos'].shape)}")
            print(f"[DEBUG] raw obs quat shape : {tuple(obs_sample['quat'].shape)}")

            image = obs_sample["image"][0:1].detach().clone().to(
                device=device,
                dtype=torch.float32,
            )
            pos = obs_sample["pos"][0:1].detach().clone().to(
                device=device,
                dtype=torch.float32,
            )
            quat = obs_sample["quat"][0:1].detach().clone().to(
                device=device,
                dtype=torch.float32,
            )

            if image.ndim == 5 and image.shape[1] == 1:
                image = image.squeeze(1)

            print(f"[DEBUG] surgery image shape: {tuple(image.shape)} dtype: {image.dtype}")
            print(f"[DEBUG] surgery pos shape  : {tuple(pos.shape)} dtype: {pos.dtype}")
            print(f"[DEBUG] surgery quat shape : {tuple(quat.shape)} dtype: {quat.dtype}")
            print(f"[DEBUG] surgery image min/max: {image.min().item():.6f} / {image.max().item():.6f}")
            print(f"[DEBUG] surgery pos sample  : {pos[0].detach().cpu().numpy()}")
            print(f"[DEBUG] surgery quat sample : {quat[0].detach().cpu().numpy()}")

            with torch.no_grad():
                flat_debug = wrapper._flatten_surgery_states(image, pos, quat)
            print(f"[DEBUG] manually flattened surgery states shape: {tuple(flat_debug.shape)}")

            with torch.no_grad():
                y1 = wrapper.forward_surgery(image, pos, quat)
                y2 = wrapper.forward_surgery(image, pos, quat)

            print(f"[DEBUG] wrapper output shape: {tuple(y1.shape)}")
            print(f"[DEBUG] repeat output sample 1: {y1[0].detach().cpu().numpy()}")
            print(f"[DEBUG] repeat output sample 2: {y2[0].detach().cpu().numpy()}")
            print(f"[DEBUG] max diff: {(y1 - y2).abs().max().item()}")

            traced = torch.jit.trace_module(
                wrapper,
                {"forward_surgery": (image, pos, quat)},
                check_trace=True,
            )

            print("[DEBUG] before traced.save")
            traced.save(output_path)
            print("[DEBUG] after traced.save")

            print(f"[EXPORT] obs image : {tuple(image.shape)}")
            print(f"[EXPORT] obs pos   : {tuple(pos.shape)}")
            print(f"[EXPORT] obs quat  : {tuple(quat.shape)}")
            print(f"[EXPORT] act shape : {tuple(y1.shape)}")
            print(f"[EXPORT] saved     : {os.path.exists(output_path)}")

        else:
            if isinstance(obs_sample, dict):
                raise TypeError(
                    f"Guidance export expected tensor observation, got dict keys: {list(obs_sample.keys())}"
                )

            direct_wrapper = GuidanceDirectNetworkWrapper(policy_model).to(device).eval()
            compute_wrapper = GuidanceComputePolicyWrapper(policy_model).to(device).eval()

            print("[DEBUG] guidance direct wrapper created")
            print("[DEBUG] guidance compute wrapper created")

            dummy = obs_sample[0:1].detach().clone().to(
                device=device,
                dtype=torch.float32,
            )

            print(f"[DEBUG] dummy shape for trace: {tuple(dummy.shape)}")
            print(f"[DEBUG] dummy dtype for trace: {dummy.dtype}")
            print(f"[DEBUG] dummy min/max: {dummy.min().item():.6f} / {dummy.max().item():.6f}")

            obs_a = obs_sample[0:1].detach().clone().to(device=device, dtype=torch.float32)
            obs_b = torch.flip(obs_a, dims=[-1]).contiguous()

            with torch.no_grad():
                y_direct_a = direct_wrapper(obs_a)
                y_direct_b = direct_wrapper(obs_b)

                y_compute_a = compute_wrapper(obs_a)
                y_compute_b = compute_wrapper(obs_b)

            print("[DEBUG DIRECT] y_a:", y_direct_a[0].detach().cpu().numpy())
            print("[DEBUG DIRECT] y_b:", y_direct_b[0].detach().cpu().numpy())
            print("[DEBUG DIRECT] diff a-b:", (y_direct_a - y_direct_b).abs().max().item())

            print("[DEBUG COMPUTE] y_a:", y_compute_a[0].detach().cpu().numpy())
            print("[DEBUG COMPUTE] y_b:", y_compute_b[0].detach().cpu().numpy())
            print("[DEBUG COMPUTE] diff a-b:", (y_compute_a - y_compute_b).abs().max().item())

            print(
                "[DEBUG COMPUTE VS DIRECT] max diff a:",
                (y_compute_a - y_direct_a).abs().max().item(),
            )
            print(
                "[DEBUG COMPUTE VS DIRECT] max diff b:",
                (y_compute_b - y_direct_b).abs().max().item(),
            )

            # Export the real SKRL compute path, not the direct network path.
            traced = torch.jit.trace(compute_wrapper, dummy, check_trace=True)

            with torch.no_grad():
                yt_a = traced(obs_a)
                yt_b = traced(obs_b)

            print("[DEBUG TRACED COMPUTE] y_a:", yt_a[0].detach().cpu().numpy())
            print("[DEBUG TRACED COMPUTE] y_b:", yt_b[0].detach().cpu().numpy())
            print("[DEBUG TRACED COMPUTE] diff a-b:", (yt_a - yt_b).abs().max().item())

            print(
                "[DEBUG TRACED COMPUTE VS COMPUTE] max diff a:",
                (yt_a - y_compute_a).abs().max().item(),
            )
            print(
                "[DEBUG TRACED COMPUTE VS COMPUTE] max diff b:",
                (yt_b - y_compute_b).abs().max().item(),
            )

            print("[DEBUG] before traced.save")
            traced.save(output_path)
            print("[DEBUG] after traced.save")

            reloaded = torch.jit.load(output_path, map_location=device)
            reloaded.eval()

            with torch.no_grad():
                yr_a = reloaded(obs_a)
                yr_b = reloaded(obs_b)

            print("[DEBUG RELOADED] y_a:", yr_a[0].detach().cpu().numpy())
            print("[DEBUG RELOADED] y_b:", yr_b[0].detach().cpu().numpy())
            print("[DEBUG RELOADED] diff a-b:", (yr_a - yr_b).abs().max().item())

            print(
                "[DEBUG RELOADED VS COMPUTE] max diff a:",
                (yr_a - y_compute_a).abs().max().item(),
            )
            print(
                "[DEBUG RELOADED VS COMPUTE] max diff b:",
                (yr_b - y_compute_b).abs().max().item(),
            )

            debug_obs_path = (
                "/home/idsia/SonoGym/source/spinal_surgery/spinal_surgery/"
                "recordings/robot_US_guidance_G1/policy_obs_env0.pt"
            )

            if os.path.exists(debug_obs_path):
                print("[DEBUG SAVED OBS] found:", debug_obs_path)

                obs_debug = torch.load(debug_obs_path, map_location=device).float()

                if obs_debug.ndim != 4:
                    print(f"[DEBUG SAVED OBS] skipped: expected 4D, got {tuple(obs_debug.shape)}")
                else:
                    valid = obs_debug.abs().sum(dim=(1, 2, 3)) > 1e-8
                    obs_debug = obs_debug[valid]

                    if obs_debug.shape[0] == 0:
                        print("[DEBUG SAVED OBS] skipped: all frames are zero")
                    else:
                        obs_debug = obs_debug[:32].to(device).contiguous()

                        with torch.no_grad():
                            yd = direct_wrapper(obs_debug)
                            yc = compute_wrapper(obs_debug)
                            yj = reloaded(obs_debug)

                        print("[DEBUG SAVED OBS] obs_debug shape:", tuple(obs_debug.shape))
                        print("[DEBUG SAVED OBS] obs min/max:", obs_debug.min().item(), "/", obs_debug.max().item())

                        print("[DEBUG SAVED OBS] direct first :", yd[0].detach().cpu().numpy())
                        print("[DEBUG SAVED OBS] compute first:", yc[0].detach().cpu().numpy())
                        print("[DEBUG SAVED OBS] jit first    :", yj[0].detach().cpu().numpy())

                        print(
                            "[DEBUG SAVED OBS] compute vs direct max diff:",
                            (yc - yd).abs().max().item(),
                        )
                        print(
                            "[DEBUG SAVED OBS] jit vs direct max diff:",
                            (yj - yd).abs().max().item(),
                        )
                        print(
                            "[DEBUG SAVED OBS] jit vs compute max diff:",
                            (yj - yc).abs().max().item(),
                        )

            print(f"[EXPORT] obs shape : {(1, *obs_shape)}")
            print(f"[EXPORT] act shape : {tuple(y_direct_a.shape)}")
            print(f"[EXPORT] saved     : {os.path.exists(output_path)}")

            y1 = y_direct_a

        return ExportResult(
            checkpoint_path=checkpoint_path,
            exported_path=output_path,
            action_dim=int(y1.shape[-1]),
        )

    except Exception as e:
        print(f"[DEBUG][ERROR] export failed with: {repr(e)}")
        raise

    finally:
        if env is not None:
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