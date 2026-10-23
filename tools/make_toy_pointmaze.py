"""Create a tiny SYNTHETIC dataset in the on-disk format of `point_maze`.

This exists only to smoke-test the unmodified upstream `train.py` entry point
without downloading the real DINO-WM datasets.  The pixels are uniform noise
and the states/actions are random; every metric produced from this data is
meaningless and must never be reported as a result.

Layout produced (matches datasets/point_maze_dset.py):
    <out>/point_maze/
        states.pth        (n_traj, traj_len, 4) float32
        actions.pth       (n_traj, traj_len, 2) float32
        seq_lengths.pth   (n_traj,) int64
        obses/episode_XXX.pth   (traj_len, H, W, 3) uint8
"""

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="parent dir; point_maze/ is created inside")
    parser.add_argument("--n-traj", type=int, default=4)
    parser.add_argument("--traj-len", type=int, default=30)
    parser.add_argument("--img", type=int, default=64)
    parser.add_argument("--state-dim", type=int, default=4)
    parser.add_argument("--action-dim", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    g = torch.Generator().manual_seed(args.seed)
    root = Path(args.out) / "point_maze"
    (root / "obses").mkdir(parents=True, exist_ok=True)

    n, t = args.n_traj, args.traj_len
    states = torch.randn(n, t, args.state_dim, generator=g)
    actions = torch.randn(n, t, args.action_dim, generator=g) * 0.1
    seq_lengths = torch.full((n,), t, dtype=torch.long)

    torch.save(states, root / "states.pth")
    torch.save(actions, root / "actions.pth")
    torch.save(seq_lengths, root / "seq_lengths.pth")

    for i in range(n):
        imgs = torch.randint(0, 256, (t, args.img, args.img, 3), generator=g, dtype=torch.int64)
        torch.save(imgs.to(torch.uint8), root / "obses" / f"episode_{i:03d}.pth")

    print(
        f"wrote synthetic point_maze: n_traj={n} traj_len={t} img={args.img} "
        f"state_dim={args.state_dim} action_dim={args.action_dim} -> {root}"
    )


if __name__ == "__main__":
    main()
