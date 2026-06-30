# Loopcraft Data Generation

## Environment Setup

```bash
conda create -n loopcraft-datagen python=3.11
conda activate loopcraft-datagen
conda install conda-forge::openjdk=8
pip install setuptools==65.5.0 pip==21 wheel==0.38.0
pip install git+https://github.com/dcharatan/minerl.git@v0.4
pip install -r requirements.txt
```

## Running the Code

To generate videos, do the following:

```bash
MINERL_HEADLESS=1 python -m source.worker --path /tmp/loopcraft-datagen-test
```

To generate test set videos (more likely to contain loops), use `--split test`. To see other available options, use `--help`.

## Compute Requirements

It took us less than 8 hours to generate 200,000 examples on somewhere between 400 and 1024 CPU-only Slurm workers. The breakdown for creating a single example seems to be about the following:

- Environment reset: 20 seconds (90 seconds for the first iteration)
- Rendering: 10-50 seconds (depending on CPU vs. GPU rendering + hardware)
- Saving files: 10 seconds

This could almost certainly be optimized further, e.g. by saving files while the reset/rendering is happening, by teleporting the player to a new location instead of resetting the environment, etc.

## Acknowledgements

This was adapted from [Wilson Yan's code](https://github.com/wilson1yan/collect-minecraft).
