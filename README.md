# MARL Incentives

This project evaluates multi-agent reinforcement-learning approaches for
incentivising route choices in a SUMO traffic simulation. Each traveller is an
agent: it selects a route (and, where applicable, an incentive), SUMO simulates
all selected routes together, and the resulting travel time and CO2 emissions
are used to update the agents.

## What is included

- `data/` contains the Kamppi SUMO network, route alternatives, and simulation
  input/output files.
- `packages/marl_incentives/` contains the shared simulation, traveller,
  replay-buffer, emissions, and neural-network code.
- `scripts/` contains experiment entry points and their YAML configurations.

The simulation calls the external `sumo` executable. The Python package
`sumolib` alone is not enough to run an experiment.

## Prerequisites

- Windows, Linux, or macOS with Python 3.13 or later.
- [UV](https://docs.astral.sh/uv/) for the locked Python environment.
- [Eclipse SUMO](https://sumo.dlr.de/docs/Downloads.php), with `sumo` available
  on `PATH`.

On Windows, install SUMO and UV with:

```powershell
winget install --name sumo
winget install --id astral-sh.uv --exact
```

Open a new terminal after installation, then confirm:

```powershell
sumo --version
uv --version
```

## Setup

From the repository root, create the locked environment:

```powershell
uv sync
```

`uv sync` installs the nested `marl_incentives` package and all declared
dependencies; no separate editable install is required.

## Running experiments

Run commands from the repository root so that the relative `data/...` paths in
the YAML configuration files resolve correctly. For example:

```powershell
.\.venv\Scripts\python.exe scripts\qlearning_no_state_exp_replay.py
```

Each entry point loads its same-named YAML file and loops over every value in
`total_budget`.

| Method | Script | Configuration |
| --- | --- | --- |
| Stateless Q-learning with experience replay | `scripts/qlearning_no_state_exp_replay.py` | `scripts/qlearning_no_state_exp_replay.yaml` |
| Binary-state Q-learning with replay | `scripts/qlearning_binary_state_exp_replay.py` | `scripts/qlearning_binary_state_exp_replay.yaml` |
| Thompson sampling | `scripts/thompson_sampling.py` | `scripts/thompson_sampling.yaml` |
| Q-learning with a learned SUMO surrogate | `scripts/qlearning_sumo_surrogate.py` | `scripts/qlearning_sumo_surrogate.yaml` |

Testing-only DQN and UCB variants are retained in `scripts/legacy/` and are
not part of the paper experiments. The separate
`qlearning_discrete_state_exp_replay.py` prototype is also not listed as a
reported method because its status still needs an author decision.

## Reproducibility and outputs

Every active YAML file defines a `seed` (42 by default). The experiment seeds
Python, NumPy, the traveller-policy generator, PyTorch, and SUMO. All budgets
use that same documented seed unless the YAML is changed.

Immutable inputs remain in `data/`. Generated routes, edge data, weights, SUMO
configuration, statistics, trip information, FCD output, and logs are written
to `results/runs/<experiment>/budget_<budget>/`. Each run also records its full
configuration, seed, and SUMO version in `run_metadata.json`. Plots and pickle
files remain under their existing `results/plots/` and
`results/pickle_files/` locations.

The Q-learning and Thompson configurations accept these reward modes:

- `weighted`: the existing weighted travel-time/emissions cost;
- `speed_relative`: each driver's mean speed minus the network mean, as in the
  paper;
- `speed_percentile`: 5 for the fastest quartile, 1 for the middle half, and
  -1 for the slowest quartile.

The two speed definitions are rewards in the paper. Internally their negatives
are stored because the existing algorithms select the minimum Q-value.
Ready-to-run configurations are provided for both paper rewards:

```powershell
.\.venv\Scripts\python.exe scripts\qlearning_no_state_exp_replay.py scripts\qlearning_speed_relative.yaml
.\.venv\Scripts\python.exe scripts\qlearning_no_state_exp_replay.py scripts\qlearning_speed_percentile.yaml
```

Generate comparison plots after the required result pickle files exist:

```powershell
.\.venv\Scripts\python.exe scripts\generate_plots.py
```

## Known implementation notes

- `qlearning_discrete_state_exp_replay.py` currently unpacks two values from a
  policy function that returns four, so it needs a small code correction before
  it will run.
- The budget-state description in the paper and the binary state implemented
  by `qlearning_binary_state_exp_replay.py` still need an author decision; this
  pull request deliberately does not change that state definition.
- The surrogate loads its run-specific checkpoint when present. If it is
  absent, the script builds/loads a dataset, trains from scratch, and saves a
  checkpoint. Model dimensions are derived from the current drivers and route
  sets.
- The configuration controls episode count, budgets, reward weights,
  exploration, learning rate, replay-buffer size, and compliance behaviour.
