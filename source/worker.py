import json
import traceback
from pathlib import Path
from time import time
from typing import Literal

import click
import gym
import numpy as np
from jaxtyping import Float
from PIL import Image

from .agent import ACTION_IDS, ACTIONS, make_low_chamfer_trajectory, make_trajectory
from .env import SimpleExplore
from .video import VideoEncoder


def collect_episode(
    env,
    num_frames: int,
    split: Literal["train", "test"],
) -> tuple[list[Image.Image, Float[np.ndarray, " channel"]]]:
    frames = []
    actions = []
    fn = {
        "train": make_trajectory,
        "test": make_low_chamfer_trajectory,
    }[split]
    for action_name in fn(num_frames):
        action = ACTIONS[action_name]
        action_id = ACTION_IDS[action_name]
        observation, _, _, _ = env.step(action)
        frame = observation["pov"]
        frame = Image.fromarray(frame)
        frame = frame.resize((256, 256), Image.Resampling.BOX)
        frames.append(frame)
        action = (
            np.array(action_id, dtype=np.float32),
            observation["location_stats"]["xpos"],
            observation["location_stats"]["ypos"],
            observation["location_stats"]["zpos"],
            observation["location_stats"]["yaw"],
            observation["location_stats"]["pitch"],
        )
        actions.append(np.stack(action).astype(np.float32))

    return frames, actions


@click.command()
@click.option(
    "--path",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Directory in which to save the generated episodes.",
)
@click.option(
    "--num-examples",
    type=int,
    default=1,
    show_default=True,
    help="Number of episodes to generate.",
)
@click.option(
    "--num-frames",
    type=int,
    default=1024,
    show_default=True,
    help="Number of frames in each episode.",
)
@click.option(
    "--split",
    type=click.Choice(["train", "test"]),
    default="train",
    show_default=True,
    help="Which trajectory generator to use.",
)
@click.option(
    "--num-workers",
    type=int,
    default=1,
    show_default=True,
    help="Total workers cooperating on this --path (see --worker-id).",
)
@click.option(
    "--worker-id",
    type=int,
    default=0,
    show_default=True,
    help="This worker's index in [0, num-workers). Worker k takes the episodes where "
    "index %% num-workers == k, so a SLURM array fills one output tree.",
)
def main(
    path: Path,
    num_examples: int,
    num_frames: int,
    split: Literal["train", "test"],
    num_workers: int,
    worker_id: int,
) -> None:
    # Create the environment.
    SimpleExplore(resolution=(512, 512)).register()
    env = gym.make("SimpleExplore-v0")

    # Define the function that gets repeated.
    def task_fn(key: str) -> None:
        print(f"Resetting environment ({key}).")
        start = time()
        env.reset()
        print(f"Resetting environment took {time() - start:.2f} seconds ({key}).")
        start = time()
        try:
            frames, actions = collect_episode(env, num_frames, split)
        except Exception as e:
            traceback.print_exc()
            print(e)
            raise e
        print(f"Episode collection took {time() - start:.2f} seconds ({key}).")

        # Save the episode as an mp4.
        start = time()
        video_path = path / f"{key}.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        encoder = VideoEncoder()
        for frame in frames:
            encoder.add_frame(frame)
        video_path.write_bytes(encoder.result())

        # Save the actions.
        np.save(path / f"{key}.npy", actions)
        print(f"Saving data took {time() - start:.2f} seconds ({key}).")

    # Note to future users (or their Claudes):
    # - If you're actually trying to generate a full dataset, you should probably
    #   implement a way to parallelize this.
    # - It is possible to have each worker generate a fixed subset (say, 1000 examples)
    #   of the dataset, but I would highly recommend using some kind of task queue
    #   instead. We used a small custom-built queue based on a Postgres database to
    #   generate the Loopcraft dataset in the MilliVid paper. The code for our task
    #   queue isn't publicly available, but feel free to email david.charatan@gmail.com
    #   if you want it, and I will send it to you.
    keys = [f"{index // 1000:0>3}/{index % 1000:0>3}" for index in range(num_examples)]
    path.mkdir(exist_ok=True, parents=True)

    if worker_id == 0:
        with (path / "index.json").open("w") as f:
            json.dump({k: num_frames for k in keys}, f)

    for index, key in enumerate(keys):
        if index % num_workers != worker_id:
            continue

        if (path / f"{key}.npy").exists():
            print(f"Skipping {key} (already generated).")
            continue

        task_fn(key)


if __name__ == "__main__":
    main()
