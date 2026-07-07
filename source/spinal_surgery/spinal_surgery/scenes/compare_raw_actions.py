import argparse
import os

import torch
import matplotlib.pyplot as plt


def load_tensor(path: str) -> torch.Tensor:
    x = torch.load(path, map_location="cpu")
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{path} non contiene un torch.Tensor")
    return x.float()


def load_obs(path: str) -> torch.Tensor:
    x = load_tensor(path)

    # Atteso: [T, C, H, W]
    if x.ndim != 4:
        raise ValueError(
            f"{path} deve avere shape [T, C, H, W], trovata {tuple(x.shape)}"
        )

    return x.contiguous()


def load_actions(path: str) -> torch.Tensor:
    x = load_tensor(path)

    # Atteso: [T, 3]
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(
            f"{path} deve avere shape [T, 3], trovata {tuple(x.shape)}"
        )

    return x.contiguous()


def run_policy(policy_path: str, obs: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
    policy = torch.jit.load(policy_path, map_location="cpu")
    policy.eval()

    preds = []

    with torch.no_grad():
        for i in range(0, obs.shape[0], batch_size):
            batch = obs[i:i + batch_size].contiguous().float()
            out = policy(batch)

            if out.ndim == 1:
                out = out.unsqueeze(0)

            preds.append(out.detach().cpu())

    pred = torch.cat(preds, dim=0)

    if pred.ndim != 2 or pred.shape[1] != 3:
        raise ValueError(
            f"La policy ha prodotto un output con shape {tuple(pred.shape)} invece di [T, 3]"
        )

    return pred.float()


def get_middle_slice(length: int, window: int) -> slice:
    if window <= 0 or window >= length:
        return slice(0, length)

    start = (length - window) // 2
    end = start + window
    return slice(start, end)


def print_tensor_stats(name: str, x: torch.Tensor):
    print(f"[COMPARE] {name} shape:", tuple(x.shape))
    print(f"[COMPARE] {name} dtype:", x.dtype)
    print(f"[COMPARE] {name} min/max:", float(x.min()), "/", float(x.max()))
    print(f"[COMPARE] {name} mean/std:", float(x.mean()), "/", float(x.std()))


def print_action_stats(name: str, a: torch.Tensor):
    print(f"[COMPARE] {name} shape:", tuple(a.shape))
    print(f"[COMPARE] {name} first :", a[0].detach().cpu().numpy())
    print(f"[COMPARE] {name} middle:", a[a.shape[0] // 2].detach().cpu().numpy())
    print(f"[COMPARE] {name} last  :", a[-1].detach().cpu().numpy())
    print(f"[COMPARE] {name} mean  :", a.mean(dim=0).detach().cpu().numpy())
    print(f"[COMPARE] {name} std   :", a.std(dim=0).detach().cpu().numpy())


def maybe_transpose_obs(obs: torch.Tensor) -> torch.Tensor:
    # [T, C, H, W] -> [T, C, W, H]
    return obs.transpose(-1, -2).contiguous()


def align_lengths(*xs: torch.Tensor):
    T = min(x.shape[0] for x in xs)
    return [x[:T].contiguous() for x in xs]


def plot_comparison(
    task_actions: torch.Tensor,
    policy_actions: torch.Tensor,
    policy_actions_T: torch.Tensor | None,
    out_path: str,
    title: str,
    window: int,
):
    T = task_actions.shape[0]
    s = get_middle_slice(T, window)

    task_plot = task_actions[s]
    policy_plot = policy_actions[s]

    if policy_actions_T is not None:
        policy_T_plot = policy_actions_T[s]
    else:
        policy_T_plot = None

    x = range(task_plot.shape[0])
    labels = ["action[0]", "action[1]", "action[2]"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    for i in range(3):
        axes[i].plot(x, task_plot[:, i].numpy(), label="task raw action")
        axes[i].plot(x, policy_plot[:, i].numpy(), label="exported policy")

        if policy_T_plot is not None:
            axes[i].plot(
                x,
                policy_T_plot[:, i].numpy(),
                label="exported policy obs_T",
                linestyle="--",
            )

        axes[i].set_ylabel(labels[i])
        axes[i].grid(True)
        axes[i].legend()

    axes[-1].set_xlabel("time step")
    fig.suptitle(title)
    fig.tight_layout()

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"[COMPARE] saved plot: {out_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--obs",
        type=str,
        default="/home/idsia/SonoGym/source/spinal_surgery/spinal_surgery/recordings/robot_US_guidance_G1/policy_obs_env0.pt",
        help="File .pt con le osservazioni salvate della task, shape [T, C, H, W]",
    )

    parser.add_argument(
        "--task_actions",
        type=str,
        default="/home/idsia/SonoGym/source/spinal_surgery/spinal_surgery/recordings/robot_US_guidance_G1/policy_raw_action_env0.pt",
        help="File .pt con le azioni raw salvate dalla task, shape [T, 3]",
    )

    parser.add_argument(
        "--policy",
        type=str,
        default="/home/idsia/SonoGym/source/spinal_surgery/spinal_surgery/policies/guidance_policy_jit.pt",
        help="Policy TorchScript esportata",
    )

    parser.add_argument(
        "--window",
        type=int,
        default=100,
        help="Numero di campioni da plottare dal centro. Usa <=0 per plottare tutto.",
    )

    parser.add_argument(
        "--out",
        type=str,
        default="compare_raw_actions.png",
        help="Path immagine output",
    )

    parser.add_argument(
        "--test_transpose",
        action="store_true",
        help="Testa anche obs.transpose(-1, -2), utile per controllare mismatch [200,150] vs [150,200].",
    )

    parser.add_argument(
        "--force_transpose",
        action="store_true",
        help="Usa solo obs.transpose(-1, -2) come input della policy esportata.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size per inferenza TorchScript.",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("[COMPARE] obs path:        ", args.obs)
    print("[COMPARE] task_actions path:", args.task_actions)
    print("[COMPARE] policy path:     ", args.policy)
    print("=" * 80)

    obs = load_obs(args.obs)
    task_actions = load_actions(args.task_actions)

    print_tensor_stats("obs", obs)
    print_action_stats("task_actions raw", task_actions)

    if args.force_transpose:
        print("[COMPARE] force_transpose=True: uso obs.transpose(-1, -2) per la policy.")
        obs_for_policy = maybe_transpose_obs(obs)
    else:
        obs_for_policy = obs

    print_tensor_stats("obs_for_policy", obs_for_policy)

    policy_actions = run_policy(
        args.policy,
        obs_for_policy,
        batch_size=args.batch_size,
    )

    task_actions, policy_actions = align_lengths(task_actions, policy_actions)

    print_action_stats("policy_actions", policy_actions)

    diff = policy_actions - task_actions
    print_action_stats("policy_minus_task", diff)
    print("[COMPARE] MAE per action:", diff.abs().mean(dim=0).numpy())
    print("[COMPARE] MAX abs per action:", diff.abs().max(dim=0).values.numpy())

    policy_actions_T = None

    if args.test_transpose and not args.force_transpose:
        print("=" * 80)
        print("[COMPARE] Test anche con osservazione trasposta.")
        obs_T = maybe_transpose_obs(obs)
        print_tensor_stats("obs_T", obs_T)

        policy_actions_T = run_policy(
            args.policy,
            obs_T,
            batch_size=args.batch_size,
        )

        task_actions_T_aligned, policy_actions_T = align_lengths(task_actions, policy_actions_T)

        print_action_stats("policy_actions_T", policy_actions_T)

        diff_T = policy_actions_T - task_actions_T_aligned
        print_action_stats("policy_T_minus_task", diff_T)
        print("[COMPARE] MAE_T per action:", diff_T.abs().mean(dim=0).numpy())
        print("[COMPARE] MAX abs_T per action:", diff_T.abs().max(dim=0).values.numpy())

        print("=" * 80)

        # Riallineo per plot
        T = min(task_actions.shape[0], policy_actions.shape[0], policy_actions_T.shape[0])
        task_actions = task_actions[:T]
        policy_actions = policy_actions[:T]
        policy_actions_T = policy_actions_T[:T]

    title = "Confronto azioni raw: task vs policy esportata"

    if args.force_transpose:
        title += " | FORCE obs_T"
    elif args.test_transpose:
        title += " | incl. obs_T"

    plot_comparison(
        task_actions=task_actions,
        policy_actions=policy_actions,
        policy_actions_T=policy_actions_T,
        out_path=args.out,
        title=title,
        window=args.window,
    )


if __name__ == "__main__":
    main()