from typing import Optional, Sequence, Tuple
from jax_party import Actions, State, Observation
from jumanji.types import TimeStep, restart, termination, transition

from jumanji.env import Environment
from jumanji import specs
import jax
import jax.numpy as jnp
import chex

class PartyEnvironment(Environment[State, specs.DiscreteArray, Observation]):
    MOVES = jnp.array([Actions.COOPERATE, Actions.DEFECT])
    POSSIBLE_CONFIGURATIONS = jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]],)

    def __init__(self, num_agents: int, time_limit: int):
        self.num_agents = num_agents
        self.time_limit = time_limit
        super().__init__()

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        return State(
            active_agents=jnp.ones(self.num_agents, dtype=bool),
            step_count=0,
            action_mask=jnp.ones(self.num_agents, dtype=bool),
            key=key,
        )