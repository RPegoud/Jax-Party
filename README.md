# JaxParty

## 🎉 JaxParty environment

The environment is implemented as a [Jumanji](https://github.com/instadeepai/jumanji/) environment for "easy" integration with Mava. Therefore, it has to be registered in Jumanji before each run, which can be done by adding this line in any script:
```python
from jax_party import register_JaxParty

if __name__ == "__main__":
    >> register_JaxParty()
    hydra_entry_point() # this line runs fetches the hydra config and runs the experiment
    aggregate_outputs(alg_name="ff_ippo") # aggregates all the results and checkpoints in a single folder indexed by timestamp
```

## 🚀 How to run an experiment:

Mava provides multiple baseline algorithms (IPPO, MAPPO, SABLE, IDQN, SAC are of interest). The scripts are located in `./mava/systems/<alg_name>/<architecture>/ff_<alg_name>.py`.
To run a script, use:
```bash
python3 -m mava.systems.ppo.anakin.ff_ippo
```
We are only concerned about the `anakin` architecture, where both the environment and training are JAX-based. More details about architectures [here](https://arxiv.org/abs/2104.06272).

Each algorithm is configured using [Hydra](https://hydra.cc), this means that all parameters are stored in yaml files. You can find those in `./mava/configs`:
```
.
└── mava/
    └── configs/
        ├── arch/
        │   ├── anakin
        │   └── ...
        ├── default/
        │   └── *algorithms # config for the ff_<algorithm> script (includes network config, architecture and environment to run)
        ├── env/
        │   ├── scenario # all possible scenario for registered envs, we'll add some for JaxParty later
        │   └── *envs # environments default hyperparams 
        ├── logger
        ├── network # hyperparams for the MLP, CNN, RNN layers
        └── system/
            └── *algorithms # algorithms default hyperparams
```
In the current setup, all experiment outputs computed on test episodes will be logged in ``./experiment_results/<algorithm_name>/<year-month-day>/<hours-minutes-seconds>``.
```
.
└── experiment_results/
    └── <algorithm_name>/
        ├── <year-month-day>/
        │   ├── <hours-minutes-seconds>
                └── checkpoints            # parameter checkpoints recorded throughout training
                └── metrics                # aggregated metrics (e.g. mean return)
                └── vaults/jax_party/<uid> # trajectories (actions, observations, rewards, ...)
```

Trajectories are stored in a [Flashbax Vault](https://github.com/instadeepai/flashbax), for now this is only done in the `ff_ippo_vault.py` script and will be added to other architectures soon. For IPPO, these trajectories include batches of:
* ``last_done`` # boolean flag indicating whether the episode terminated
* ``action`` # actions chosen by all the agents
* ``value`` # output value of the critic network
* ``timestep.reward``
* ``log_prob`` # log prob of the actor network
* ``last_timestep.observation``
These can be read using the following snippet (see `./vault_reading.ipynb`):
```python
from flashbax.vault import Vault

v = Vault(
    rel_dir="experiment_results/ff_ippo/20250207131826/vaults",

    vault_name="jax_party",
    vault_uid="20250207131826",
)
buffer_state = v.read()

buffer_state.experience["action"] 
```

