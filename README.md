# JaxParty

## 🎉 JaxParty environment

The environment is implemented as a [Jumanji](https://github.com/instadeepai/jumanji/) environment for "easy" integration with Mava. Therefore, it has to be registered in Jumanji before each run, which can be done by adding this line in any script:
```python
from jax_party import register_JaxParty

if __name__ == "__main__":
    >> register_JaxParty()
    hydra_entry_point() # this line runs fetches the hydra config and runs the experiment
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
In the current setup, aggregated metrics computed on test episodes will be logged in `./results/json/<script_name>/uid`. Those metrics include the mean return, which might not be useful in our case.
Additionally, we collect trajectories in a [Flashbax Vault](https://github.com/instadeepai/flashbax), for now this is only done in the `ff_ippo_vault.py` script and will be added to other architectures soon. For IPPO, these trajectories include batches of:
* ``last_done`` # boolean flag indicating whether the episode terminated
* ``action`` # actions chosen by all the agents
* ``value`` # output value of the critic network
* ``timestep.reward``
* ``log_prob`` # log prob of the actor network
* ``last_timestep.observation``
These are stored in `.vaults/<script_name>_jaxparty/uid` and can be read using the following snippet (see `./vault_reading.ipynb`):
```python
from flashbax.vault import Vault

v = Vault(
    vault_name="ff_ippo_jaxparty",
    vault_uid="20250205182200",
)
buffer_state = v.read()

buffer_state.experience["action"] 
```

