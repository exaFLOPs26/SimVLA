import argparse
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)


def load_kitchen_dataset(repo_id: str, split: str = "train"):
    print(f"Loading dataset: {repo_id} (split={split})")
    ds = load_dataset(repo_id, split=split)
    print("Columns:", ds.column_names)
    return ds


def extract_episodes(is_first: np.ndarray, is_last: np.ndarray):
    """
    Use is_first / is_last to find [start, end] indices for each trajectory.
    Returns list of (start_idx, end_idx) (both inclusive).
    """
    episodes = []
    start = None

    for i in range(len(is_first)):
        if is_first[i]:
            # if we somehow had an open episode, close it at i-1
            if start is not None:
                episodes.append((start, i - 1))
            start = i

        if is_last[i] and start is not None:
            episodes.append((start, i))
            start = None

    # If last episode never closed, close at end
    if start is not None:
        episodes.append((start, len(is_first) - 1))

    return episodes


def summarize_dims(state: np.ndarray, dim_idx):
    """
    Print summary stats (mean, std, min, max, median, p1, p99)
    for each selected dimension in `state`.

    state: [N, D]
    dim_idx: list of absolute indices, e.g. [20, 21, 22]
    """
    print("\n=== Summary statistics for selected dims ===")
    header = f"{'dim':>6} {'mean':>12} {'std':>12} {'min':>12} {'max':>12} {'median':>12} {'p1':>12} {'p99':>12}"
    print(header)
    print("-" * len(header))

    for d in dim_idx:
        vals = state[:, d]  # [N]
        mean = float(vals.mean())
        std = float(vals.std())
        vmin = float(vals.min())
        vmax = float(vals.max())
        median = float(np.median(vals))
        p1, p99 = np.percentile(vals, [1, 99])

        print(
            f"{d:6d} {mean:12.6f} {std:12.6f} {vmin:12.6f} {vmax:12.6f} "
            f"{median:12.6f} {p1:12.6f} {p99:12.6f}"
        )
    print("===========================================\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo_id",
        type=str,
        default="exaFLOPs09/Isaac-Kitchen-v1119-04",
        help="Hugging Face repo id for the dataset.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split to load from the dataset.",
    )

    parser.add_argument(
        "--num_traj_to_plot",
        type=int,
        default=20,
        help="Number of trajectories to plot.",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="If set, randomly sample trajectories instead of taking the first N.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (used if --random).",
    )
    parser.add_argument(
        "--dims",
        type=str,
        default="-3,-2,-1",
        help="Comma-separated dims to summarize/plot. "
             "Negative indices are relative to the end, e.g. '-1' or '-3,-2,-1'.",
    )
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="If set, only print stats and skip plotting.",
    )

    args = parser.parse_args()

    # ----------------------------
    # Load dataset
    # ----------------------------
    ds = load_kitchen_dataset(args.repo_id, args.split)

    if "observation.state" not in ds.column_names:
        raise ValueError(f"'observation.state' not in columns: {ds.column_names}")
    if "is_first" not in ds.column_names or "is_last" not in ds.column_names:
        raise ValueError("Dataset must have 'is_first' and 'is_last' columns.")

    state = np.array(ds["action"])        # [N, D]
    is_first = np.array(ds["is_first"], dtype=bool)  # [N]
    is_last = np.array(ds["is_last"], dtype=bool)    # [N]

    if state.ndim != 2:
        raise ValueError(f"Expected state to be 2D [N, D], got shape {state.shape}")

    N, D = state.shape
    print(f"'observation.state' shape: {state.shape}")

    # ----------------------------
    # Parse dims (e.g. "-3,-2,-1")
    # ----------------------------
    dim_list = [int(x.strip()) for x in args.dims.split(",")]  # e.g. [-3, -2, -1]

    dim_idx = []
    for d in dim_list:
        if d < 0:
            d = D + d  # negative index from end
        if not (0 <= d < D):
            raise ValueError(f"Dimension index {d} out of range for D={D}")
        dim_idx.append(d)

    print(f"Selected dims (absolute indices): {dim_idx}")

    # ----------------------------
    # Summary statistics over ALL frames
    # ----------------------------
    summarize_dims(state, dim_idx)

    # ----------------------------
    # Optionally skip plotting
    # ----------------------------
    if args.no_plot:
        print("Skipping plotting because --no_plot was set.")
        return

    # ----------------------------
    # Build coords for plotting
    # coords will be [N, 3] to use in 3D plot
    # ----------------------------
    coords_raw = state[:, dim_idx]  # [N, K]
    K = coords_raw.shape[1]

    if K == 1:
        # Use chosen dim as X, set Y,Z = 0
        x = coords_raw[:, 0]
        zeros = np.zeros_like(x)
        coords = np.stack([x, zeros, zeros], axis=1)
    elif K == 2:
        # Use first two dims as X,Y, set Z = 0
        x = coords_raw[:, 0]
        y = coords_raw[:, 1]
        zeros = np.zeros_like(x)
        coords = np.stack([x, y, zeros], axis=1)
    elif K == 3:
        coords = coords_raw
    else:
        raise ValueError("dims must have 1, 2, or 3 elements for plotting.")

    # ----------------------------
    # Extract episodes
    # ----------------------------
    episodes = extract_episodes(is_first, is_last)
    if len(episodes) == 0:
        raise ValueError("No episodes found using is_first/is_last.")

    print(f"Found {len(episodes)} episodes in total.")

    # ----------------------------
    # Select episodes to plot
    # ----------------------------
    num_total = len(episodes)
    num_plot = min(args.num_traj_to_plot, num_total)

    if args.random:
        np.random.seed(args.seed)
        print(f"Randomly sampling {num_plot} / {num_total} trajectories (seed={args.seed})")
        selected_idx = np.random.choice(num_total, size=num_plot, replace=False)
    else:
        print(f"Plotting first {num_plot} / {num_total} trajectories")
        selected_idx = np.arange(num_plot)

    selected_episodes = [episodes[i] for i in selected_idx]

    # ----------------------------
    # 3D Plot
    # ----------------------------
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for ep_i, (start, end) in enumerate(selected_episodes):
        traj = coords[start : end + 1]  # [T, 3]
        if traj.shape[0] < 2:
            continue

        x, y, z = traj[:, 0], traj[:, 1], traj[:, 2]
        ax.plot(x, y, z, alpha=0.7)
        # start / end markers
        ax.scatter(x[0], y[0], z[0], marker="o")
        ax.scatter(x[-1], y[-1], z[-1], marker="x")

    ax.set_xlabel(f"state[{dim_idx[0]}] (or dummy)")
    if K >= 2:
        ax.set_ylabel(f"state[{dim_idx[1]}] (or dummy)")
    else:
        ax.set_ylabel("0")
    if K == 3:
        ax.set_zlabel(f"state[{dim_idx[2]}]")
    else:
        ax.set_zlabel("0")

    ax.set_title(
        f"3D trajectories from dims {dim_list} "
        f"({num_plot} episodes, repo={args.repo_id})"
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

