
from enum import IntEnum
from typing import  NamedTuple

import chex
from chex import dataclass

class Actions(IntEnum):
    COOPERATE: int = 0
    DEFECT: int = 1

@dataclass
class State:
    active_agents: chex.Array
    step_count: int
    action_mask: chex.Array
    key: chex.PRNGKey

class Observation(NamedTuple):
    player: int
    action: Actions
    reward: int