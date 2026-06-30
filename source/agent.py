import itertools
from typing import Literal

import numpy as np
from jaxtyping import Float

ActionName = Literal["forward", "left", "right"]

ACTIONS: dict[ActionName, dict] = {
    "forward": dict(forward=1, jump=1, camera=[0.0, 0.0]),
    "left": dict(forward=0, jump=1, camera=[0.0, -10.0]),
    "right": dict(forward=0, jump=1, camera=[0.0, 10.0]),
}

ACTION_IDS: dict[ActionName, int] = {
    "forward": 0,
    "left": 1,
    "right": 2,
}


def random_walk():
    while True:
        num_turns = np.random.choice([1, 2, 3, 4], p=[0.1, 0.3, 0.1, 0.5])
        num_forwards = np.random.randint(num_turns, 3 * num_turns)
        direction = np.random.choice(["left", "right"])
        bundle = ["forward"] * num_forwards + [direction] * num_turns
        np.random.shuffle(bundle)
        bundle = ["forward"] + bundle
        for action in bundle:
            for _ in range(9):
                yield action


def make_trajectory(length: int) -> list[ActionName]:
    return list(itertools.islice(random_walk(), length))


def simulate_turtle(
    actions: list[ActionName],
) -> Float[np.ndarray, "n 2"]:
    """Simulate turtle-style movement and return the (x, y) positions visited."""
    positions: list[tuple[float, float]] = []
    x, y = 0.0, 0.0
    heading_deg = 0.0  # 0 means pointing along +y
    positions.append((x, y))
    for action in actions:
        yaw = ACTIONS[action]["camera"][1]
        forward = ACTIONS[action]["forward"]
        heading_deg += yaw
        if forward:
            heading_rad = np.deg2rad(heading_deg)
            x += np.sin(heading_rad)
            y += np.cos(heading_rad)
        positions.append((x, y))
    return np.asarray(positions, dtype=np.float64)


def chamfer_distance(
    a: Float[np.ndarray, "n 2"],
    b: Float[np.ndarray, "m 2"],
) -> float:
    """Compute the bidirectional Chamfer distance between two 2D point sets."""
    diff = a[:, None, :] - b[None, :, :]
    d2 = np.sum(diff * diff, axis=-1)
    a_to_b = np.sqrt(d2.min(axis=1)).mean()
    b_to_a = np.sqrt(d2.min(axis=0)).mean()
    return float(a_to_b + b_to_a)


def make_low_chamfer_trajectory(
    length: int = 1024,
    top_p: float = 0.1,
    cutoff: int = 256,
    num_samples: int = 100,
) -> list[ActionName]:
    """Sample trajectories and return one with low self-overlap chamfer distance."""
    trajectories: list[list[ActionName]] = [
        make_trajectory(length) for _ in range(num_samples)
    ]
    distances: list[float] = []
    for traj in trajectories:
        positions = simulate_turtle(traj)
        before_cutoff = positions[:cutoff]
        after_cutoff = positions[cutoff:]
        distances.append(chamfer_distance(before_cutoff, after_cutoff))

    order = np.argsort(distances)
    keep_count = max(1, int(np.ceil(top_p * num_samples)))
    top_indices = order[:keep_count]
    chosen = int(np.random.choice(top_indices))
    return trajectories[chosen]
