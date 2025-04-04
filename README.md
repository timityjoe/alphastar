# AlphaStar.

[AlphaStar](https://github.com/deepmind/alphastar) is a package from
[DeepMind](http://deepmind.com) that provides the tools to train an agent to
master StarCraft II offered by [Blizzard Entertainment](http://blizzard.com).

As part of our open-sourcing efforts to drive more research interest around
StarCraft II, we provide the following key offerings with this package:

1.  General purpose architectures to train StarCraftII agents in
    `architectures/` that can be used with different learning algorithms in
    online and offline settings.

2.  Data readers, offline training and evaluation scripts for fully offline
    reinforcement learning with Behavior Cloning as a representative example
    under `unplugged/` directory.

## Setup

We have tested AlphaStar only in **Python3.9** and **Linux**. Currently, we do
not support other operating systems and recommend users to stick to Linux.

### Quickstart (from $ALPHASTAR/alphastar/unplugged/README.MD)
We have published a [paper](https://openreview.net/pdf?id=Np8Pumfoty) which
outlines why Starcraft is one of the most challenging offline RL benchmarks to
date along with the performance benchmarks for a spectrum of offline RL
approaches that we have explored on this data. The current set of algorithms
released as part of this repository are

1.  Behavior Cloning (**BC**). One can implement a fine-tuned version
    (**FT-BC**) by warmstarting from a trained checkpoint.

Our training setup is heavily config driven. Our main config can be found in
`alphastar/unplugged/configs/alphastar_supervised.py`. Run all these commands from the
root directory where the package is downloaded into.

### Train with dummy data (for debugging)

To run training for a few steps with some config arguments updated, run:

```shell
python alphastar/unplugged/scripts/train.py \
  --config=alphastar/unplugged/configs/alphastar_supervised.py:alphastar.dummy \
  --config.train.max_number_of_frames=16 \
  --config.train.learner_kwargs.batch_size=4 \
  --config.train.datasource.kwargs.shuffle_buffer_size=16 \
  --config.train.optimizer_kwargs.lr_frames_before_decay=4 \
  --config.train.learner_kwargs.unroll_len=3 \
  --config.train.datasource.name=DummyDataSource
```

### Train with real data

To train with real data

1.  Follow the instructions for data generation in
    [alphastar/unplugged/data/README.md](https://github.com/deepmind/alphastar/blob/master/alphastar/unplugged/data/README.md),

2.  The next step is to create a paths python file with two constants

    -   BASE_PATH and RELATIVE_PATHS. BASE_PATH is the root directory for the
        converted datasets that were generated in Step 1. RELATIVE_PATHS is a
        dictionary mapping of keys and values as follows :
        {(replay_versions, data_split, player_min_mmr) : <Glob pattern relative to BASE_PATH for files>}

    We have provided a
    [template file](https://github.com/deepmind/alphastar/blob/master/alphastar/unplugged/data/paths.py.template)
    for setting the data paths appropriately. Please copy this template file to
    some directory of choice `cp alphastar/unplugged/data/paths.py.template
    /tmp/paths.py` Modify the paths based on step 1 and use the file as \
    `config.train.datasource.kwargs.dataset_paths_fname` while launching
    training.

    While training, the particular data that you want to train on can be set by
    setting the `replay_versions`, `data_split`, `player_min_mmr` and
    `dataset_paths_fname` via the config using `config.train.datasource.kwargs`
    or invoking the same on command line.

3.  After these two steps, run (to confirm the entire training apparatus with
    training data from SC2 version 4.9.2, assuming that the paths file from step
    2 is `/tmp/paths.py`)

    ```shell
    python alphastar/unplugged/scripts/train.py \
      --config=alphastar/unplugged/configs/alphastar_supervised.py:alphastar.dummy \
      --config.train.max_number_of_frames=16 \
      --config.train.learner_kwargs.batch_size=4 \
      --config.train.datasource.kwargs.shuffle_buffer_size=16 \
      --config.train.optimizer_kwargs.lr_frames_before_decay=4 \
      --config.train.learner_kwargs.unroll_len=3 \
      --config.train.datasource.name=OfflineTFRecordDataSource \
      --config.train.datasource.kwargs.dataset_paths_fname='/tmp/paths.py' \
      --config.train.datasource.kwargs.replay_versions='("4.9.2",)'
    ```

4.  To run default full scale training after the real dataset is generated and
    the paths are updated, run the following command. Do note that the default
    setting is to run with all replay versions. If you want to run on specific
    replay versions only, please set
    `config.train.datasource.kwargs.replay_versions` as shown below.

    ```shell
    python alphastar/unplugged/scripts/train.py \
      --config=alphastar/unplugged/configs/alphastar_supervised.py:alphastar.full \
      --config.train.datasource.kwargs.dataset_paths_fname='/tmp/paths.py' \
      --config.train.datasource.kwargs.replay_versions='("4.9.2",)'
    ```

### Evaluate a random agent

To evaluate a random agent in the environment for one full episode, run:

```shell
python alphastar/unplugged/scripts/evaluate.py \
  --config=alphastar/unplugged/configs/alphastar_supervised.py:alphastar.dummy \
  --config.eval.log_to_csv=False \
  --config.eval.evaluator_type=random_params
```

More instructions on how to use these scripts for full-fledged training and
evaluation can be found in the docstrings of the scripts. Information about
different architecture names can be found
[here](https://github.com/deepmind/alphastar/blob/master/alphastar/architectures/README.md).



### Preliminaries

We recommend using a Python virtual environment to manage dependencies. This
should help to avoid version conflicts and just generally make the installation
process easier.

```shell
conda create -n conda39-alphastar-tim python=3.9
conda activate conda39-alphastar-tim

pip install jax==0.3.2 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install jaxlib==0.3.2 -f https://storage.googleapis.com/jax-releases/jax_releases.html

python3 -m venv alphastar
source alphastar/bin/activate
pip install --upgrade pip setuptools wheel
```

AlphaStar depends on [PySC2](https://github.com/deepmind/pysc2) converters for
data generation and evaluation. Since the code for converters is written in C++,
any changes to the converter code will require recompiling the PySC2 native
extensions. Because of this we offer two different ways to use AlphaStar:

1.  **Installing AlphaStar with `pip`**: this option requires the least setup.
    However if you make changes to PySC2, or if you want to use a version for
    which no pre-built wheel is available, you will need to manually build and
    install a new wheel for PySC2.
2.  **Building AlphaStar using Bazel**: in this case AlphaStar and PySC2 are
    built together from source. By default the PySC2 sources are fetched
    from GitHub. If you wish to use a local repository instead (e.g. because you
    have made local modifications to PySC2) you should modify
    `alphastar/WORKSPACE` as described in the comments.

#### Installing with `pip`

If you're interested in running the bleeding edge versions, you can do so by
cloning our GitHub repository and then executing the following command from the
main directory (where `setup.py` is located):

```
pip install -e .  # For an editable version
pip install .     # For a non-editable version
```

Note that this will also install all the dependencies of AlphaStar.

### Building with Bazel

First, install Bazel by following the instructions
[here](https://docs.bazel.build/versions/main/install-ubuntu.html).

PySC2 requires C++ 17, so Bazel builds of AlphaStar + PySC2 must use
`--cxxopt='-std=c++17'`. For example, to build all AlphaStar targets, run the
following command from the workspace root:

```shell
bazel build --cxxopt='-std=c++17' ...
```

To recursively run all of the tests within the `architectures/` subdirectory:

```shell
bazel test --cxxopt='-std=c++17' architectures/...
```

See the documentation for
[AlphaStar Unplugged](https://github.com/deepmind/alphastar/blob/master/alphastar/unplugged/README.md)
for example `run` commands.

Note: Bazel caches Python package dependencies downloaded from `pip`. To clear
this cache (for example if you have edited `requirements.txt`), run `bazel clean
--expunge`.

You may wish to use a
[.bazelrc file](https://docs.bazel.build/versions/main/guide.html#bazelrc-the-bazel-configuration-file)
to avoid the need to repeatedly specify command-line options, for instance
`--cxxopt='-std=c++17'`.

## Quickstart

For quickstart instructions on how to run training and evaluation scripts in
*fully offline* settings, please refer to
[this README](https://github.com/deepmind/alphastar/blob/master/alphastar/unplugged/README.md). In
this repository, we have not provided any online RL training code. But, the
architectures are fit to be used in both online and offline training.

## About

Disclaimer: This is not an official Google product.

If you use the agents, architectures and offline RL benchmarks published in this
repository, please cite our
[AlphaStar Unplugged](https://openreview.net/pdf?id=Np8Pumfoty) paper.
